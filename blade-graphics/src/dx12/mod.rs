use std::{ptr, sync::Mutex};
use windows::{
    core::Interface,
    Win32::{
        Foundation::{HANDLE, WAIT_EVENT},
        Graphics::{
            Direct3D12::*,
            Dxgi::{Common::*, *},
        },
        System::Threading::{WaitForSingleObjectEx, INFINITE},
    },
};

mod command;
mod init;
mod pipeline;
mod resource;
mod surface;

// ── Descriptor heap sizes ─────────────────────────────────────────────────────

pub(super) const RTV_HEAP_SIZE: u32 = 512;
pub(super) const DSV_HEAP_SIZE: u32 = 128;
pub(super) const STAGING_HEAP_SIZE: u32 = 65536;
pub(super) const STAGING_SAMPLER_SIZE: u32 = 2048;
// Per-encoder GPU-visible CBV/SRV/UAV ring. Each `bind()` consumes descriptors,
// so a frame with N draws needs ~N entries. 1,000,000 is the D3D12 tier-1 max
// for a shader-visible CBV/SRV/UAV heap. Reset each `start()`.
pub(super) const ENCODER_HEAP_SIZE: u32 = 1_000_000;
// Every encoder's CBV/SRV/UAV heap STARTS at this size and grows on demand up to
// its tier max (see DescriptorRing::grow). Small start = transient transfer/copy
// encoders (which bind ~nothing) stay tiny instead of each reserving a huge heap;
// a heavy render encoder grows a few times on its first frames, then is stable.
pub(super) const ENCODER_HEAP_START: u32 = 16384;
pub(super) const ENCODER_SAMPLER_SIZE: u32 = 2048;
// Per-encoder upload scratch for plain/CBV data: ~256 B per uniform bind, so a
// frame with many per-draw binds needs room. 64 MiB ~= 256k binds. Reset each frame.
pub(super) const UPLOAD_RING_SIZE: u64 = 64 * 1024 * 1024;

// Register spaces for naga 28's HLSL sampler-heap model. naga's BindTarget.space
// is a u8, so these MUST stay <= 255 (and match the root signature, which uses
// the same numeric values). Chosen high to avoid colliding with per-group spaces
// (group `g` uses `space g`; group counts are tiny).
//   nagaSamplerHeap[]           : register(s0, space SAMPLER_HEAP_SPACE)
//   nagaComparisonSamplerHeap[] : register(s0, space SAMPLER_HEAP_SPACE + 1)
//   nagaGroup{g}SamplerIndexArray : register(t0, space SAMPLER_INDEX_SPACE_BASE + g)
pub(super) const SAMPLER_HEAP_SPACE: u32 = 250;
pub(super) const SAMPLER_INDEX_SPACE_BASE: u32 = 100;

// ── Error type ────────────────────────────────────────────────────────────────

pub type PlatformError = windows::core::Error;

// ── Descriptor heap helper ────────────────────────────────────────────────────

pub(super) struct DescriptorHeap {
    pub raw: ID3D12DescriptorHeap,
    pub cpu_start: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub gpu_start: D3D12_GPU_DESCRIPTOR_HANDLE,
    pub increment: u32,
    pub capacity: u32,
}

unsafe impl Send for DescriptorHeap {}
unsafe impl Sync for DescriptorHeap {}

impl DescriptorHeap {
    pub fn new(
        device: &ID3D12Device,
        ty: D3D12_DESCRIPTOR_HEAP_TYPE,
        capacity: u32,
        gpu_visible: bool,
    ) -> Self {
        let flags = if gpu_visible {
            D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE
        } else {
            D3D12_DESCRIPTOR_HEAP_FLAG_NONE
        };
        let desc = D3D12_DESCRIPTOR_HEAP_DESC {
            Type: ty,
            NumDescriptors: capacity,
            Flags: flags,
            NodeMask: 0,
        };
        let raw: ID3D12DescriptorHeap = unsafe { device.CreateDescriptorHeap(&desc).unwrap() };
        let cpu_start = unsafe { raw.GetCPUDescriptorHandleForHeapStart() };
        let gpu_start = if gpu_visible {
            unsafe { raw.GetGPUDescriptorHandleForHeapStart() }
        } else {
            D3D12_GPU_DESCRIPTOR_HANDLE { ptr: 0 }
        };
        let increment = unsafe { device.GetDescriptorHandleIncrementSize(ty) };
        Self { raw, cpu_start, gpu_start, increment, capacity }
    }

    pub fn cpu_handle(&self, index: u32) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: self.cpu_start.ptr + (index * self.increment) as usize,
        }
    }

    pub fn gpu_handle(&self, index: u32) -> D3D12_GPU_DESCRIPTOR_HANDLE {
        D3D12_GPU_DESCRIPTOR_HANDLE {
            ptr: self.gpu_start.ptr + (index * self.increment) as u64,
        }
    }
}

pub(super) struct LinearAllocator {
    pub offset: u32,
}
impl LinearAllocator {
    pub fn alloc(&mut self, count: u32) -> u32 {
        let idx = self.offset;
        self.offset += count;
        idx
    }
}

// ── Queue ─────────────────────────────────────────────────────────────────────

pub(super) struct Queue {
    pub raw: ID3D12CommandQueue,
    pub fence: ID3D12Fence,
    pub fence_event: HANDLE,
    pub last_progress: u64,
}

unsafe impl Send for Queue {}
unsafe impl Sync for Queue {}

// ── Context ───────────────────────────────────────────────────────────────────

pub struct Context {
    pub(super) device: ID3D12Device,
    pub(super) queue: Mutex<Queue>,
    pub(super) rtv_heap: DescriptorHeap,
    pub(super) dsv_heap: DescriptorHeap,
    pub(super) staging_heap: DescriptorHeap,
    pub(super) staging_sampler_heap: DescriptorHeap,
    pub(super) rtv_alloc: Mutex<LinearAllocator>,
    pub(super) dsv_alloc: Mutex<LinearAllocator>,
    pub(super) staging_alloc: Mutex<LinearAllocator>,
    pub(super) staging_sampler_alloc: Mutex<LinearAllocator>,
    pub(super) draw_sig: ID3D12CommandSignature,
    pub(super) draw_indexed_sig: ID3D12CommandSignature,
    pub(super) dispatch_sig: ID3D12CommandSignature,
    pub(super) validation_mode: bool,
    pub(super) device_information: crate::DeviceInformation,
    pub(super) vendor: crate::GpuVendor,
    pub(super) factory: IDXGIFactory4,
    /// Per-encoder shader-visible CBV/SRV/UAV heap capacity. The heap is split
    /// into `buffer_count` per-frame segments, so this must be large enough that
    /// one frame's binds fit in `size / buffer_count`. Resource-binding tier 1/2
    /// cap a shader-visible heap at 1,000,000; tier 3 (most discrete GPUs) has no
    /// such cap, so we use a larger heap there to keep per-frame headroom.
    pub(super) cbv_srv_uav_heap_size: u32,
}

unsafe impl Send for Context {}
unsafe impl Sync for Context {}

// ── Swapchain / surface ───────────────────────────────────────────────────────

pub struct Surface {
    pub(super) hwnd: Option<windows::Win32::Foundation::HWND>,
    pub(super) swapchain: Option<IDXGISwapChain3>,
    pub(super) frames: Vec<SurfaceFrame>,
    pub(super) format: crate::TextureFormat,
    pub(super) dxgi_format: DXGI_FORMAT,
    pub(super) alpha: crate::AlphaMode,
    pub(super) target_size: [u16; 2],
    pub(super) next_present_id: u64,
    /// Swapchain frame-latency waitable. `acquire_frame` waits on it so the GPU
    /// does not render into a back buffer still being presented. Required for
    /// correctness: a blade SyncPoint is a GPU-render-completion fence, which is
    /// NOT present completion — DXGI FLIP has no acquire-semaphore (Vulkan's
    /// GPU-side present-readiness sync), so this CPU wait is the substitute.
    pub(super) frame_latency_waitable: Option<windows::Win32::Foundation::HANDLE>,
}

unsafe impl Send for Surface {}
unsafe impl Sync for Surface {}

pub(super) struct SurfaceFrame {
    pub resource: ID3D12Resource,
    pub rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub fence_value: u64,
}

// ── Frame ─────────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct Frame {
    pub(super) resource: ID3D12Resource,
    pub(super) swapchain: IDXGISwapChain3,
    pub(super) rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub(super) buffer_index: u32,
    pub(super) present_id: u64,
    pub(super) format: crate::TextureFormat,
    pub(super) target_size: [u16; 2],
    pub(super) display_sync: bool,
}

unsafe impl Send for Frame {}
unsafe impl Sync for Frame {}

impl Frame {
    pub fn texture(&self) -> Texture {
        // Frame textures are not owned — memory_handle=!0 marks this as un-destroyable.
        // resource_ptr is null because init_texture is a no-op in DX12.
        Texture {
            resource_ptr: ptr::null_mut(),
            memory_handle: !0,
            format: self.format,
            target_size: self.target_size,
            usage: crate::TextureUsage::TARGET,
            mip_levels: 1,
            array_layers: 1,
            sample_count: 1,
        }
    }

    pub fn texture_view(&self) -> TextureView {
        // Store the raw COM vtable pointer (without AddRef — borrowed from the frame).
        let resource_ptr = unsafe { self.resource.as_raw() as *mut std::ffi::c_void };
        TextureView {
            resource_ptr,
            srv_handle: 0,
            uav_handle: 0,
            rtv_dsv_handle: self.rtv.ptr as u64,
            format: self.format,
            aspects: crate::TexelAspects::COLOR,
            target_size: self.target_size,
            base_mip: 0,
            mip_count: 1,
            base_array: 0,
            array_count: 1,
            mip_levels: 1,
            array_layers: 1,
        }
    }
}

// ── Synchronization ───────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct SyncPoint {
    progress: u64,
}

// ── Resource types ────────────────────────────────────────────────────────────

/// A raw CPU descriptor handle stored as u64 (0 = not present).
pub(super) type RawCpuHandle = u64;

pub(super) fn raw_cpu_handle(v: RawCpuHandle) -> D3D12_CPU_DESCRIPTOR_HANDLE {
    D3D12_CPU_DESCRIPTOR_HANDLE { ptr: v as usize }
}

/// Drain the D3D12 debug layer's message queue to the log (only populated when
/// validation is enabled). Call this right after a failed D3D12 call to learn
/// the precise reason the runtime rejected it.
pub(super) fn log_debug_messages(device: &ID3D12Device) {
    let info_queue: ID3D12InfoQueue = match device.cast() {
        Ok(q) => q,
        Err(_) => return, // debug layer not enabled
    };
    unsafe {
        let count = info_queue.GetNumStoredMessages();
        for i in 0..count {
            let mut len: usize = 0;
            if info_queue.GetMessage(i, None, &mut len).is_err() || len == 0 {
                continue;
            }
            let mut buf = vec![0u8; len];
            let msg = buf.as_mut_ptr() as *mut D3D12_MESSAGE;
            if info_queue.GetMessage(i, Some(msg), &mut len).is_ok() {
                let m = &*msg;
                let text = std::slice::from_raw_parts(m.pDescription, m.DescriptionByteLength);
                let text = String::from_utf8_lossy(text);
                let text = text.trim_end_matches('\0');
                match m.Severity {
                    D3D12_MESSAGE_SEVERITY_CORRUPTION | D3D12_MESSAGE_SEVERITY_ERROR => {
                        log::error!("[D3D12] {text}")
                    }
                    D3D12_MESSAGE_SEVERITY_WARNING => log::warn!("[D3D12] {text}"),
                    _ => {}
                }
            }
        }
        info_queue.ClearStoredMessages();
    }
}

/// If the device has been removed, return the reason as a string.
pub(super) fn device_removed_reason(device: &ID3D12Device) -> Option<String> {
    match unsafe { device.GetDeviceRemovedReason() } {
        Ok(()) => None,
        Err(e) => Some(format!("{e}")),
    }
}

unsafe fn pcwstr_to_string(s: windows::core::PCWSTR) -> String {
    if s.is_null() {
        return String::new();
    }
    s.to_string().unwrap_or_default()
}

// Per-pass label tracking for DRED. DRED gives the hung op's *type* and the
// command-list debug name, but not which blade pass it belonged to. We name
// each list `blade-enc-{id}` and record the label of every pass (in marker-emit
// order) keyed by id; on a hang, counting the SetMarker ops up to the hung op
// indexes this table to the exact pass. Only active under validation.
static DRED_ENABLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static NEXT_ENCODER_ID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
static PASS_LABELS: std::sync::OnceLock<Mutex<std::collections::HashMap<u32, Vec<String>>>> =
    std::sync::OnceLock::new();

fn pass_label_table() -> &'static Mutex<std::collections::HashMap<u32, Vec<String>>> {
    PASS_LABELS.get_or_init(|| Mutex::new(std::collections::HashMap::new()))
}

pub(super) fn dred_set_enabled() {
    DRED_ENABLED.store(true, std::sync::atomic::Ordering::Relaxed);
}

pub(super) fn dred_next_encoder_id() -> u32 {
    NEXT_ENCODER_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

pub(super) fn dred_record_pass(encoder_id: u32, label: &str) {
    if !DRED_ENABLED.load(std::sync::atomic::Ordering::Relaxed) {
        return;
    }
    pass_label_table()
        .lock()
        .unwrap()
        .entry(encoder_id)
        .or_default()
        .push(label.to_string());
}

pub(super) fn dred_clear_passes(encoder_id: u32) {
    if !DRED_ENABLED.load(std::sync::atomic::Ordering::Relaxed) {
        return;
    }
    if let Some(v) = pass_label_table().lock().unwrap().get_mut(&encoder_id) {
        v.clear();
    }
}

/// Map a hung command list's debug name + completed-op count to the blade pass
/// label: parse `blade-enc-{id}`, count SetMarker ops up to `done` (each pass
/// emits exactly one marker at its start), and index the recorded labels.
fn dred_pass_for(
    list_name: &str,
    history: *const D3D12_AUTO_BREADCRUMB_OP,
    done: u32,
) -> Option<String> {
    // The debug layer appends suffixes (e.g. "blade-enc-1 (debug layer
    // indirect)"), so take the leading run of digits after the prefix.
    let rest = list_name.strip_prefix("blade-enc-")?;
    let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    let id: u32 = digits.parse().ok()?;
    if history.is_null() {
        return None;
    }
    let markers = (0..=done)
        .filter(|&i| unsafe { *history.add(i as usize) } == D3D12_AUTO_BREADCRUMB_OP_SETMARKER)
        .count();
    if markers == 0 {
        return None;
    }
    let table = pass_label_table().lock().unwrap();
    table.get(&id).and_then(|v| v.get(markers - 1).cloned())
}

fn breadcrumb_op_name(op: D3D12_AUTO_BREADCRUMB_OP) -> &'static str {
    match op {
        D3D12_AUTO_BREADCRUMB_OP_SETMARKER => "SetMarker",
        D3D12_AUTO_BREADCRUMB_OP_BEGINEVENT => "BeginEvent",
        D3D12_AUTO_BREADCRUMB_OP_ENDEVENT => "EndEvent",
        D3D12_AUTO_BREADCRUMB_OP_DRAWINSTANCED => "DrawInstanced",
        D3D12_AUTO_BREADCRUMB_OP_DRAWINDEXEDINSTANCED => "DrawIndexedInstanced",
        D3D12_AUTO_BREADCRUMB_OP_EXECUTEINDIRECT => "ExecuteIndirect",
        D3D12_AUTO_BREADCRUMB_OP_DISPATCH => "Dispatch",
        D3D12_AUTO_BREADCRUMB_OP_COPYBUFFERREGION => "CopyBufferRegion",
        D3D12_AUTO_BREADCRUMB_OP_COPYTEXTUREREGION => "CopyTextureRegion",
        D3D12_AUTO_BREADCRUMB_OP_RESOLVESUBRESOURCE => "ResolveSubresource",
        D3D12_AUTO_BREADCRUMB_OP_CLEARRENDERTARGETVIEW => "ClearRenderTargetView",
        D3D12_AUTO_BREADCRUMB_OP_CLEARDEPTHSTENCILVIEW => "ClearDepthStencilView",
        D3D12_AUTO_BREADCRUMB_OP_CLEARUNORDEREDACCESSVIEW => "ClearUnorderedAccessView",
        _ => "other",
    }
}

/// Dump DRED auto-breadcrumbs and page-fault data after a device removal. The
/// last completed breadcrumb value vs. the op count identifies the command-list
/// operation the GPU was executing when it hung; the breadcrumb context (the
/// pass label emitted as a marker in `begin_pass`) names the pass; the
/// page-fault VA + named allocation identifies a bad address. Only populated
/// when DRED was armed at init (validation builds).
pub(super) fn log_dred(device: &ID3D12Device) {
    // Prefer the v1 output, which carries per-op context strings (our pass
    // markers); fall back to v0 (op type only) on older runtimes.
    if let Ok(dred1) = device.cast::<ID3D12DeviceRemovedExtendedData1>() {
        unsafe { log_dred_breadcrumbs1(&dred1) };
    } else if let Ok(dred) = device.cast::<ID3D12DeviceRemovedExtendedData>() {
        if let Ok(bc) = unsafe { dred.GetAutoBreadcrumbsOutput() } {
            unsafe { log_breadcrumb_node_v0(bc.pHeadAutoBreadcrumbNode) };
        }
    }

    let dred: ID3D12DeviceRemovedExtendedData = match device.cast() {
        Ok(d) => d,
        Err(_) => return,
    };
    unsafe {
        if let Ok(pf) = dred.GetPageFaultAllocationOutput() {
            if pf.PageFaultVA != 0 {
                log::error!("[DRED] GPU page fault at VA {:#x}", pf.PageFaultVA);
                for (label, mut alloc) in [
                    ("existing", pf.pHeadExistingAllocationNode),
                    ("recently-freed", pf.pHeadRecentFreedAllocationNode),
                ] {
                    while !alloc.is_null() {
                        let a = &*alloc;
                        log::error!(
                            "[DRED]   {label} allocation '{}' type {:?}",
                            pcwstr_to_string(a.ObjectNameW),
                            a.AllocationType,
                        );
                        alloc = a.pNext;
                    }
                }
            }
        }
    }
}

/// Report a hung breadcrumb node: the op the GPU stopped on, the few ops
/// preceding it (the pass's command pattern), and `context` (the pass label
/// marker active at the hung op, empty if unavailable).
unsafe fn report_hung_node(
    list: &str,
    queue: &str,
    history: *const D3D12_AUTO_BREADCRUMB_OP,
    count: u32,
    done: u32,
    context: &str,
) {
    // Skip nodes that fully executed (done == count) — those didn't hang. Also
    // skip the debug layer's own GBV helper lists (1-op internal command lists).
    if done >= count || count <= 1 {
        return;
    }
    let hung = if history.is_null() {
        D3D12_AUTO_BREADCRUMB_OP(-1)
    } else {
        *history.add(done as usize)
    };
    // Prefer the DRED-provided context string; fall back to our own marker-count
    // table (works even when the driver doesn't surface marker contexts).
    let resolved = if context.is_empty() {
        dred_pass_for(list, history, done)
    } else {
        Some(context.to_string())
    };
    let pass = resolved.as_deref().unwrap_or("<unknown>");
    log::error!(
        "[DRED] hung in pass '{pass}' (list '{list}', queue '{queue}') after {done}/{count} ops; hung op = {} {:?}",
        breadcrumb_op_name(hung),
        hung,
    );
    if !history.is_null() {
        let start = done.saturating_sub(8);
        let recent: Vec<&str> = (start..done)
            .map(|i| breadcrumb_op_name(*history.add(i as usize)))
            .collect();
        log::error!("[DRED]   preceding ops [{start}..{done}]: {}", recent.join(", "));
    }
}

unsafe fn log_dred_breadcrumbs1(dred: &ID3D12DeviceRemovedExtendedData1) {
    let bc = match dred.GetAutoBreadcrumbsOutput1() {
        Ok(bc) => bc,
        Err(_) => return,
    };
    let mut node = bc.pHeadAutoBreadcrumbNode;
    while !node.is_null() {
        let n = &*node;
        let done = if n.pLastBreadcrumbValue.is_null() {
            0
        } else {
            *n.pLastBreadcrumbValue
        };
        // Find the marker context whose breadcrumb index is the greatest <= the
        // hung op — that is the pass label that was active when the GPU stopped.
        let mut context = String::new();
        let mut best: i64 = -1;
        for c in 0..n.BreadcrumbContextsCount {
            let ctx = &*n.pBreadcrumbContexts.add(c as usize);
            if ctx.BreadcrumbIndex <= done && ctx.BreadcrumbIndex as i64 > best {
                best = ctx.BreadcrumbIndex as i64;
                context = pcwstr_to_string(ctx.pContextString);
            }
        }
        report_hung_node(
            &pcwstr_to_string(n.pCommandListDebugNameW),
            &pcwstr_to_string(n.pCommandQueueDebugNameW),
            n.pCommandHistory,
            n.BreadcrumbCount,
            done,
            &context,
        );
        node = n.pNext;
    }
}

unsafe fn log_breadcrumb_node_v0(mut node: *const D3D12_AUTO_BREADCRUMB_NODE) {
    while !node.is_null() {
        let n = &*node;
        let done = if n.pLastBreadcrumbValue.is_null() {
            0
        } else {
            *n.pLastBreadcrumbValue
        };
        report_hung_node(
            &pcwstr_to_string(n.pCommandListDebugNameW),
            &pcwstr_to_string(n.pCommandQueueDebugNameW),
            n.pCommandHistory,
            n.BreadcrumbCount,
            done,
            "",
        );
        node = n.pNext;
    }
}

/// If the device has been removed, dump debug messages + DRED and panic with
/// `what` as context. A no-op while the device is healthy, so it can be called
/// after fallible calls (Present, ExecuteCommandLists, Signal) whose own error
/// is otherwise swallowed — turning a later silent abort into an actionable
/// device-removal report at the point it actually happened.
pub(super) fn panic_if_device_removed(device: &ID3D12Device, what: &str) {
    if let Some(reason) = device_removed_reason(device) {
        log_debug_messages(device);
        log_dred(device);
        panic!("DX12 device removed during {what}: {reason}");
    }
}

/// Unwrap a D3D12 call's result, dumping debug-layer messages and the
/// device-removed reason on failure so the panic is actionable.
pub(super) fn expect_d3d12<T>(device: &ID3D12Device, what: &str, r: windows::core::Result<T>) -> T {
    match r {
        Ok(v) => v,
        Err(e) => {
            log_debug_messages(device);
            if let Some(reason) = device_removed_reason(device) {
                log_dred(device);
                panic!("DX12 {what} failed: {e}; device removed reason: {reason}");
            }
            panic!("DX12 {what} failed: {e}");
        }
    }
}

/// Pointer stored in `Buffer.resource_ptr` and `Texture.resource_ptr`:
///   - Box ptr (`Box::into_raw(Box::new(ID3D12Resource))`)
///   - null for frame textures (memory_handle == !0)
/// Pointer stored in `TextureView.resource_ptr`:
///   - COM vtable ptr (`resource.as_raw() as *mut c_void`)
pub(super) type ResourcePtr = *mut std::ffi::c_void;

/// Borrow a Box-stored resource (for `Buffer` and `Texture`).
/// # Safety
/// `ptr` must have been created by `Box::into_raw(Box::new(ID3D12Resource))`.
pub(super) unsafe fn box_resource_ref(ptr: ResourcePtr) -> &'static ID3D12Resource {
    debug_assert!(!ptr.is_null(), "null resource_ptr in box_resource_ref");
    &*(ptr as *const ID3D12Resource)
}

#[derive(Clone, Copy, Debug, Hash, PartialEq)]
pub struct Buffer {
    /// `Box::into_raw(Box::new(ID3D12Resource))`.
    pub(super) resource_ptr: ResourcePtr,
    pub(super) gpu_address: u64,
    pub(super) mapped_ptr: *mut u8,
    pub(super) srv_handle: RawCpuHandle,
    pub(super) uav_handle: RawCpuHandle,
    pub(super) size: u64,
}

unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

impl Default for Buffer {
    fn default() -> Self {
        Self {
            resource_ptr: ptr::null_mut(),
            gpu_address: 0,
            mapped_ptr: ptr::null_mut(),
            srv_handle: 0,
            uav_handle: 0,
            size: 0,
        }
    }
}

impl Buffer {
    pub fn data(&self) -> *mut u8 {
        self.mapped_ptr
    }

    pub(super) fn resource(&self) -> &ID3D12Resource {
        unsafe { box_resource_ref(self.resource_ptr) }
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq)]
pub struct Texture {
    /// `Box::into_raw(Box::new(ID3D12Resource))`, or null when `memory_handle == !0`.
    pub(super) resource_ptr: ResourcePtr,
    /// `!0` signals a "borrowed" frame texture that must not be destroyed.
    pub(super) memory_handle: usize,
    pub(super) format: crate::TextureFormat,
    pub(super) target_size: [u16; 2],
    /// Which views are legal to create — D3D12 forbids e.g. an RTV on a
    /// resource not created with ALLOW_RENDER_TARGET (can remove the device).
    pub(super) usage: crate::TextureUsage,
    /// For per-subresource state tracking (subresource = mip + layer*mip_levels).
    pub(super) mip_levels: u32,
    pub(super) array_layers: u32,
    /// MSAA sample count; >1 requires multisample view dimensions (TEXTURE2DMS).
    pub(super) sample_count: u32,
}

unsafe impl Send for Texture {}
unsafe impl Sync for Texture {}

impl Default for Texture {
    fn default() -> Self {
        // Null placeholder (matches Vulkan/Metal). memory_handle=!0 marks it
        // un-destroyable; it must be overwritten before any real use.
        Self {
            resource_ptr: ptr::null_mut(),
            memory_handle: !0,
            format: crate::TextureFormat::Rgba8Unorm,
            target_size: [0; 2],
            usage: crate::TextureUsage::empty(),
            mip_levels: 1,
            array_layers: 1,
            sample_count: 1,
        }
    }
}

impl Texture {
    pub(super) fn resource(&self) -> &ID3D12Resource {
        unsafe { box_resource_ref(self.resource_ptr) }
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq)]
pub struct TextureView {
    /// COM vtable pointer (`resource.as_raw() as *mut c_void`), borrowed.
    pub(super) resource_ptr: ResourcePtr,
    pub(super) srv_handle: RawCpuHandle,
    pub(super) uav_handle: RawCpuHandle,
    pub(super) rtv_dsv_handle: RawCpuHandle,
    pub(super) format: crate::TextureFormat,
    pub(super) aspects: crate::TexelAspects,
    pub(super) target_size: [u16; 2],
    /// Subresource range this view covers, for per-subresource state tracking.
    pub(super) base_mip: u32,
    pub(super) mip_count: u32,
    pub(super) base_array: u32,
    pub(super) array_count: u32,
    pub(super) mip_levels: u32,
    pub(super) array_layers: u32,
}

unsafe impl Send for TextureView {}
unsafe impl Sync for TextureView {}

impl Default for TextureView {
    fn default() -> Self {
        Self {
            resource_ptr: ptr::null_mut(),
            srv_handle: 0,
            uav_handle: 0,
            rtv_dsv_handle: 0,
            format: crate::TextureFormat::Rgba8Unorm,
            aspects: crate::TexelAspects::COLOR,
            target_size: [0; 2],
            base_mip: 0,
            mip_count: 1,
            base_array: 0,
            array_count: 1,
            mip_levels: 1,
            array_layers: 1,
        }
    }
}

impl TextureView {
    /// Total subresources of the underlying texture, including planes. A
    /// depth+stencil format has 2 planes (0=depth, 1=stencil), each a distinct
    /// subresource that must be transitioned — missing the stencil plane left it
    /// in COMMON and removed the device on ClearDepthStencilView.
    pub(super) fn total_subresources(&self) -> u32 {
        self.mip_levels * self.array_layers * plane_count(self.format)
    }
    /// Subresource indices this view covers, across all planes
    /// (index = plane*mip_levels*array_layers + layer*mip_levels + mip).
    pub(super) fn subresources(&self) -> Vec<u32> {
        let ml = self.mip_levels;
        let al = self.array_layers;
        let planes = plane_count(self.format);
        let mut out = Vec::with_capacity((self.mip_count * self.array_count * planes) as usize);
        for p in 0..planes {
            for a in self.base_array..self.base_array + self.array_count {
                for m in self.base_mip..self.base_mip + self.mip_count {
                    out.push(p * ml * al + a * ml + m);
                }
            }
        }
        out
    }
}

/// Number of subresource planes for a format. Depth+stencil formats have a
/// separate stencil plane; everything else is single-plane.
pub(super) fn plane_count(format: crate::TextureFormat) -> u32 {
    match format {
        crate::TextureFormat::Depth32FloatStencil8Uint => 2,
        _ => 1,
    }
}


#[derive(Clone, Copy, Debug, Hash, PartialEq)]
pub struct Sampler {
    pub(super) cpu_handle: RawCpuHandle,
}

unsafe impl Send for Sampler {}
unsafe impl Sync for Sampler {}

impl Default for Sampler {
    fn default() -> Self {
        Self { cpu_handle: 0 }
    }
}

#[derive(Clone, Copy, Debug, Default, Hash, PartialEq)]
pub struct AccelerationStructure {}

// ── Pipeline types ────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum BindingKind {
    Srv,
    Uav,
    Cbv,
    Sampler,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct BindingSlot {
    pub heap_offset: u32,
    pub register: u32,
    pub kind: BindingKind,
}

#[derive(Clone, Debug)]
pub(super) struct GroupDescriptors {
    /// Root parameter index of this group's CBV/SRV/UAV descriptor table.
    pub cbv_srv_uav_root_index: Option<u32>,
    /// Root parameter index of this group's sampler-index-buffer root SRV.
    /// naga 28's HLSL backend addresses samplers as
    /// `nagaSamplerHeap[nagaGroupNSamplerIndexArray[register]]`, so each group
    /// with samplers needs a `StructuredBuffer<uint>` holding the heap indices.
    pub sampler_index_root_index: Option<u32>,
    pub cbv_srv_uav_count: u32,
    pub sampler_count: u32,
    pub slots: Vec<BindingSlot>,
}

#[derive(Debug)]
pub(super) struct PipelineLayout {
    pub root_signature: ID3D12RootSignature,
    pub groups: Vec<GroupDescriptors>,
    /// Root parameter indices of the two global sampler heaps
    /// (standard, comparison). Present iff any group has samplers.
    pub sampler_heap_roots: Option<(u32, u32)>,
}

pub struct ComputePipeline {
    pub(super) pso: ID3D12PipelineState,
    pub(super) layout: PipelineLayout,
    pub(super) wg_size: [u32; 3],
}

unsafe impl Send for ComputePipeline {}
unsafe impl Sync for ComputePipeline {}

impl ComputePipeline {
    pub fn get_workgroup_size(&self) -> [u32; 3] {
        self.wg_size
    }
}

pub struct RenderPipeline {
    pub(super) pso: ID3D12PipelineState,
    pub(super) layout: PipelineLayout,
    pub(super) topology: windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY,
    pub(super) vertex_strides: Vec<u32>,
}

unsafe impl Send for RenderPipeline {}
unsafe impl Sync for RenderPipeline {}

// ── Per-encoder GPU-visible descriptor ring ───────────────────────────────────

/// A GPU-visible descriptor heap partitioned into `segment_count` regions, one
/// per in-flight frame. The heap is reused across frames, so the region a frame
/// writes into MUST NOT be reset while a prior frame's command list is still
/// reading it on the GPU. `begin_segment` rewinds only the current frame's
/// region; the caller (`CommandEncoder::start`) advances the segment in lockstep
/// with the allocator rotation, having already fence-waited on the submission
/// that last used this segment. A single offset-0 reset (which is what a
/// single-buffered ring does) would clobber descriptors of frames still in
/// flight → garbage bindings, the classic source of intermittent glitches.
fn create_shader_visible_heap(
    device: &ID3D12Device,
    ty: D3D12_DESCRIPTOR_HEAP_TYPE,
    capacity: u32,
) -> windows::core::Result<ID3D12DescriptorHeap> {
    unsafe {
        device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
            Type: ty,
            NumDescriptors: capacity,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            NodeMask: 0,
        })
    }
}

#[derive(Clone)]
struct HeapBlock {
    heap: ID3D12DescriptorHeap,
    cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    gpu: D3D12_GPU_DESCRIPTOR_HANDLE,
}

impl HeapBlock {
    fn new(device: &ID3D12Device, ty: D3D12_DESCRIPTOR_HEAP_TYPE, cap: u32) -> Self {
        let heap = create_shader_visible_heap(device, ty, cap)
            .unwrap_or_else(|e| panic!("CreateDescriptorHeap({cap}) failed: {e}"));
        let cpu = unsafe { heap.GetCPUDescriptorHandleForHeapStart() };
        let gpu = unsafe { heap.GetGPUDescriptorHandleForHeapStart() };
        Self { heap, cpu, gpu }
    }
}

/// Per-frame GPU-visible descriptor allocator. The `base..limit` window is the
/// region the current frame allocates into; it normally lives in a segment of
/// the permanent `main` heap (per-frame isolation by offset), but on overflow it
/// either grows the main heap (copy live region into a bigger heap) or, once the
/// main heap is at the hardware cap, SPILLS into a dedicated cap-sized heap and
/// keeps going. Spill heaps are pooled and reused per frame; outgrown main heaps
/// are returned to the caller to destroy (they are smaller and never reused).
pub(super) struct DescriptorRing {
    // Active allocation cursor — the main segment or a spill heap.
    pub heap: ID3D12DescriptorHeap,
    pub cpu_start: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub gpu_start: D3D12_GPU_DESCRIPTOR_HANDLE,
    pub increment: u32,
    pub offset: u32,
    base: u32,
    limit: u32,
    on_spill: bool,
    // Permanent, segmented, growable main heap.
    main: HeapBlock,
    main_capacity: u32,
    /// Hard cap any single heap may reach (the hardware shader-visible limit for
    /// this type: 1,000,000 CBV/SRV/UAV on tier 1/2, 2048 samplers).
    pub max_capacity: u32,
    ty: D3D12_DESCRIPTOR_HEAP_TYPE,
    segment_count: u32,
    segment: u32,
    // Cap-sized spill heaps in use by the in-progress frame.
    spill_in_use: Vec<HeapBlock>,
    // Spill heaps from past frames, reusable once `free_after` <= frame_index.
    spill_pool: Vec<(u64, HeapBlock)>,
}

unsafe impl Send for DescriptorRing {}

impl DescriptorRing {
    fn new(
        device: &ID3D12Device,
        ty: D3D12_DESCRIPTOR_HEAP_TYPE,
        capacity: u32,
        max_capacity: u32,
        segment_count: u32,
    ) -> Self {
        let max_capacity = max_capacity.max(segment_count);
        let cap = capacity.max(segment_count).min(max_capacity);
        let main = HeapBlock::new(device, ty, cap);
        let increment = unsafe { device.GetDescriptorHandleIncrementSize(ty) };
        Self {
            heap: main.heap.clone(),
            cpu_start: main.cpu,
            gpu_start: main.gpu,
            increment,
            offset: 0,
            base: 0,
            limit: cap / segment_count,
            on_spill: false,
            main,
            main_capacity: cap,
            max_capacity,
            ty,
            segment_count,
            segment: 0,
            spill_in_use: Vec::new(),
            spill_pool: Vec::new(),
        }
    }

    fn main_segment_capacity(&self) -> u32 {
        self.main_capacity / self.segment_count
    }

    /// Reset to a fresh frame: bind the main heap's segment, and recycle the
    /// previous frame's spill heaps (tagged complete-after `frame_index +
    /// buffer_count - 1`, since they were used by the frame that just ended).
    pub fn begin_segment(&mut self, segment: u32, frame_index: u64, buffer_count: u64) {
        let prev_frame = frame_index.saturating_sub(1);
        for block in self.spill_in_use.drain(..) {
            self.spill_pool.push((prev_frame + buffer_count, block));
        }
        let seg_cap = self.main_segment_capacity();
        self.segment = segment;
        self.on_spill = false;
        self.heap = self.main.heap.clone();
        self.cpu_start = self.main.cpu;
        self.gpu_start = self.main.gpu;
        self.base = segment * seg_cap;
        self.limit = self.base + seg_cap;
        self.offset = self.base;
    }

    fn used(&self) -> u32 {
        self.offset - self.base
    }

    /// Offset relative to the current `base` (stable across a grow, which copies
    /// the live region to the same relative layout).
    fn rel(&self, abs: u32) -> u32 {
        abs - self.base
    }

    fn would_overflow(&self, count: u32) -> bool {
        self.offset + count > self.limit
    }

    /// Grow the main heap so the current segment holds `used() + count`, copying
    /// the live region into the bigger heap. Returns the OLD main heap (still read
    /// by already-recorded draws — keep alive until they complete). Returns `None`
    /// if already on a spill heap or at `max_capacity`.
    fn grow_main(&mut self, device: &ID3D12Device, count: u32) -> Option<ID3D12DescriptorHeap> {
        if self.on_spill {
            return None;
        }
        let used = self.used();
        let mut new_seg_cap = self.main_segment_capacity();
        while new_seg_cap < used + count {
            new_seg_cap = new_seg_cap.saturating_mul(2);
        }
        let new_cap = new_seg_cap.saturating_mul(self.segment_count).min(self.max_capacity);
        if new_cap <= self.main_capacity {
            return None; // at the hardware cap
        }
        let new_seg_cap = new_cap / self.segment_count;
        let new = HeapBlock::new(device, self.ty, new_cap);
        if used > 0 {
            let new_seg_base = self.segment * new_seg_cap;
            let dst = D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: new.cpu.ptr + (new_seg_base * self.increment) as usize,
            };
            let src = D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: self.main.cpu.ptr + (self.base * self.increment) as usize,
            };
            unsafe { device.CopyDescriptorsSimple(used, dst, src, self.ty) };
        }
        let old = std::mem::replace(&mut self.main, new);
        self.main_capacity = new_cap;
        let new_seg_base = self.segment * new_seg_cap;
        self.heap = self.main.heap.clone();
        self.cpu_start = self.main.cpu;
        self.gpu_start = self.main.gpu;
        self.base = new_seg_base;
        self.limit = new_seg_base + new_seg_cap;
        self.offset = new_seg_base + used;
        Some(old.heap)
    }

    /// Take a cap-sized spill heap (reuse a pooled one proven GPU-complete, else
    /// allocate). Switch the active cursor to it with the cursor at `carry` (the
    /// caller has copied `carry` carried-over descriptors to its front). The
    /// previous active heap stays alive (main is permanent; a previous spill heap
    /// is still in `spill_in_use`), so already-recorded draws keep reading it.
    fn spill(&mut self, device: &ID3D12Device, carry: u32, frame_index: u64) {
        let reuse = self
            .spill_pool
            .iter()
            .position(|(free_after, _)| *free_after <= frame_index);
        let block = match reuse {
            Some(i) => self.spill_pool.swap_remove(i).1,
            None => HeapBlock::new(device, self.ty, self.max_capacity),
        };
        self.heap = block.heap.clone();
        self.cpu_start = block.cpu;
        self.gpu_start = block.gpu;
        self.base = 0;
        self.limit = self.max_capacity;
        self.offset = carry;
        self.on_spill = true;
        self.spill_in_use.push(block);
    }

    /// CPU handle of the current active heap at relative offset `rel`.
    fn cpu_handle_rel(&self, rel: u32) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: self.cpu_start.ptr + ((self.base + rel) * self.increment) as usize,
        }
    }

    /// GPU handle of the current active heap at relative offset `rel`.
    fn gpu_handle_rel(&self, rel: u32) -> D3D12_GPU_DESCRIPTOR_HANDLE {
        D3D12_GPU_DESCRIPTOR_HANDLE {
            ptr: self.gpu_start.ptr + ((self.base + rel) * self.increment) as u64,
        }
    }

    pub fn alloc(&mut self, count: u32) -> (D3D12_CPU_DESCRIPTOR_HANDLE, D3D12_GPU_DESCRIPTOR_HANDLE) {
        assert!(self.offset + count <= self.limit, "descriptor ring segment overflow");
        let cpu = D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: self.cpu_start.ptr + (self.offset * self.increment) as usize,
        };
        let gpu = D3D12_GPU_DESCRIPTOR_HANDLE {
            ptr: self.gpu_start.ptr + (self.offset * self.increment) as u64,
        };
        self.offset += count;
        (cpu, gpu)
    }
}

// ── Upload scratch ring (for CBV plain-data) ──────────────────────────────────

/// Host-visible scratch buffer, partitioned per in-flight frame exactly like
/// `DescriptorRing` (see its doc): the GPU reads this buffer's CBV/upload data
/// while a frame is in flight, so a frame must only write into its own segment.
pub(super) struct UploadRing {
    pub resource: ID3D12Resource,
    pub gpu_address: u64,
    pub mapped: *mut u8,
    pub offset: u64,
    pub capacity: u64,
    segment_count: u64,
    segment: u64,
}

unsafe impl Send for UploadRing {}

impl UploadRing {
    fn new(device: &ID3D12Device, capacity: u64, segment_count: u64) -> Self {
        let mut resource: Option<ID3D12Resource> = None;
        unsafe {
            device.CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() },
                D3D12_HEAP_FLAG_NONE,
                &D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: capacity,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    ..Default::default()
                },
                D3D12_RESOURCE_STATE_GENERIC_READ,
                None,
                &mut resource,
            ).unwrap();
        }
        let resource = resource.unwrap();
        let gpu_address = unsafe { resource.GetGPUVirtualAddress() };
        let mut mapped_ptr: *mut std::ffi::c_void = ptr::null_mut();
        unsafe { resource.Map(0, None, Some(&mut mapped_ptr)).unwrap() };
        Self { resource, gpu_address, mapped: mapped_ptr as *mut u8, offset: 0, capacity, segment_count, segment: 0 }
    }

    fn segment_capacity(&self) -> u64 {
        self.capacity / self.segment_count
    }

    fn segment_limit(&self) -> u64 {
        (self.segment + 1) * self.segment_capacity()
    }

    pub fn begin_segment(&mut self, segment: u64) {
        self.segment = segment;
        self.offset = segment * self.segment_capacity();
    }

    /// Write `data` at an `align`-aligned offset; returns its GPU address.
    /// Used for the per-group sampler-index buffer (root SRV → 4-byte aligned).
    pub fn write_aligned(&mut self, data: &[u8], align: u64) -> u64 {
        let aligned = (self.offset + align - 1) & !(align - 1);
        let end = aligned + data.len() as u64;
        assert!(end <= self.segment_limit(), "upload ring segment overflow");
        unsafe { ptr::copy_nonoverlapping(data.as_ptr(), self.mapped.add(aligned as usize), data.len()) };
        self.offset = end;
        self.gpu_address + aligned
    }

    /// Reserve `size` bytes at an `align`-aligned offset; returns the *byte
    /// offset* into the ring resource (for use as a placed-footprint Offset).
    /// The caller writes into `self.mapped + offset` and copies from
    /// `self.resource`. Used to repack texture uploads into a D3D12-legal layout.
    pub fn alloc(&mut self, size: u64, align: u64) -> u64 {
        let aligned = (self.offset + align - 1) & !(align - 1);
        let end = aligned + size;
        assert!(end <= self.segment_limit(), "upload ring segment overflow");
        self.offset = end;
        aligned
    }

    /// Write `data` at a 256-aligned offset and reserve a 256-aligned span (a CBV
    /// reads `SizeInBytes` rounded up to 256, so the whole span must stay in bounds).
    /// Returns the GPU address of the written data.
    pub fn write_cbv(&mut self, data: &[u8]) -> u64 {
        const CBV_ALIGN: u64 = 256;
        let aligned = (self.offset + CBV_ALIGN - 1) & !(CBV_ALIGN - 1);
        let size = data.len() as u64;
        let aligned_size = (size + CBV_ALIGN - 1) & !(CBV_ALIGN - 1);
        assert!(aligned + aligned_size <= self.segment_limit(), "upload ring segment overflow");
        unsafe { ptr::copy_nonoverlapping(data.as_ptr(), self.mapped.add(aligned as usize), data.len()) };
        self.offset = aligned + aligned_size;
        self.gpu_address + aligned
    }
}

// ── Presentation ──────────────────────────────────────────────────────────────

pub(super) struct Presentation {
    pub swapchain: IDXGISwapChain3,
    pub buffer_index: u32,
    pub present_id: u64,
    pub display_sync: bool,
}

// ── Command encoder ────────────────────────────────────────────────────────────

pub struct CommandEncoder {
    pub(super) list: Option<ID3D12GraphicsCommandList>,
    /// Stable id used to name the list `blade-enc-{id}` and key its per-pass
    /// label table for DRED hang attribution.
    pub(super) id: u32,
    pub(super) allocators: Vec<ID3D12CommandAllocator>,
    pub(super) device: ID3D12Device,
    pub(super) cbv_srv_uav_ring: DescriptorRing,
    pub(super) sampler_ring: DescriptorRing,
    pub(super) upload_ring: UploadRing,
    pub(super) present: Option<Presentation>,
    pub(super) timings: crate::Timings,
    pub(super) staging_heap: ID3D12DescriptorHeap,
    pub(super) staging_sampler_heap: ID3D12DescriptorHeap,
    pub(super) staging_increment: u32,
    pub(super) staging_sampler_increment: u32,
    pub(super) draw_sig: ID3D12CommandSignature,
    pub(super) draw_indexed_sig: ID3D12CommandSignature,
    pub(super) dispatch_sig: ID3D12CommandSignature,
    /// When true, a global UAV barrier is inserted automatically at the start of
    /// every pass (the DX12 analog of Vulkan's full memory barrier).
    pub(super) auto_barriers: bool,
    /// Current D3D12 state of every buffer touched this command list, keyed by
    /// COM vtable ptr. Absent => COMMON. Reset in `start()`.
    pub(super) buffer_states: std::collections::HashMap<usize, D3D12_RESOURCE_STATES>,
    /// Per-subresource state of every texture touched this command list, keyed by
    /// COM vtable ptr; Vec is indexed by subresource (mip + layer*mip_levels).
    /// Per-subresource granularity lets e.g. mip-generation read mip i-1 (SRV)
    /// while writing mip i (UAV) of the same texture.
    pub(super) texture_states: std::collections::HashMap<usize, Vec<D3D12_RESOURCE_STATES>>,
    /// Vertex buffers bound this render pass, keyed by slot. blade lets
    /// `bind_vertex` be called on the pass *before* a pipeline is bound, but in
    /// D3D12 the per-vertex stride lives in the vertex-buffer view, which comes
    /// from the pipeline. So binds are recorded here and (re)applied with the
    /// pipeline's strides at each `with()`. Cleared at `render()`.
    pub(super) pending_vertex: Vec<(u32, crate::BufferPiece)>,
    /// Queue fence (cloned) + a private event, used by `start()` to CPU-wait for
    /// the GPU to finish with an allocator before resetting it. Resetting an
    /// allocator the GPU is still reading is illegal and removes the device.
    pub(super) fence: ID3D12Fence,
    pub(super) fence_event: HANDLE,
    /// Parallel to `allocators`: the fence value each allocator's last submission
    /// was signaled at (0 = never submitted). Rotated in lockstep in `start()`.
    pub(super) allocator_fences: Vec<u64>,
    /// Monotonic count of `start()` calls; `frame_index % buffer_count` selects
    /// the per-frame ring segment. Advances in lockstep with the allocator
    /// rotation, so the segment chosen now is the same one used `buffer_count`
    /// frames ago — whose fence `start()` has just waited on.
    pub(super) frame_index: u64,
    /// (resource, subresource) pairs bound as a color/depth attachment of the
    /// currently-open render pass. Vulkan keeps every sampled texture in GENERAL
    /// layout, so a subresource that is both an attachment and an (unsampled)
    /// bound resource is harmless there; DX12 has no such shared state, so
    /// transitioning it to SHADER_RESOURCE on bind would break the in-flight
    /// draws. We keep exactly those subresources in their attachment state and
    /// skip only their transition. This MUST be per-subresource: bloom binds mip
    /// `i-1` (SRV) of the same texture whose mip `i` is the render target — those
    /// are different subresources and the SRV mip absolutely must be transitioned
    /// (keying on the resource alone would skip it → stale RT data sampled).
    /// Repopulated by `render()`, cleared by `begin_pass()`.
    pub(super) active_render_targets: Vec<(ResourcePtr, u32)>,
    /// CBV/SRV/UAV heaps replaced by a mid-frame grow, paired with the
    /// `frame_index` at which they were retired. Already-recorded draws in the
    /// current command list still read from them, so they are freed only in a
    /// later `start()` once that frame's submission is known complete.
    pub(super) retired_cbv_heaps: Vec<(u64, ID3D12DescriptorHeap)>,
    /// CBV/SRV/UAV root descriptor tables bound since the last root-signature
    /// change (set by `with()`), as (root_index, relative offset, descriptor
    /// count, is_compute). On a grow these are re-pointed into the new heap; on a
    /// spill their descriptors are copied (hence the count) and re-pointed. A
    /// descriptor-heap switch invalidates table bindings and a later draw may
    /// reuse a group without re-binding it, so all active tables are replayed.
    pub(super) active_cbv_tables: Vec<(u32, u32, u32, bool)>,
    /// The pipeline's sampler-heap root table indices (std, cmp, is_compute),
    /// set by `with()`; re-issued after a grow's `SetDescriptorHeaps`.
    pub(super) bound_sampler_roots: Option<(u32, u32, bool)>,
}

unsafe impl Send for CommandEncoder {}

pub struct TransferCommandEncoder<'a> {
    pub(super) encoder: &'a mut CommandEncoder,
}

pub struct AccelerationStructureCommandEncoder<'a> {
    pub(super) list: &'a ID3D12GraphicsCommandList,
}

pub struct ComputeCommandEncoder<'a> {
    pub(super) encoder: &'a mut CommandEncoder,
}

/// A `FinishOp::ResolveTo` deferred to the end of the render pass.
pub(super) struct PendingResolve {
    pub(super) src: TextureView,
    pub(super) dst: TextureView,
    pub(super) mode: crate::ResolveMode,
}

pub struct RenderCommandEncoder<'a> {
    pub(super) encoder: &'a mut CommandEncoder,
    pub(super) target_size: [u16; 2],
    pub(super) resolves: Vec<PendingResolve>,
}

impl Drop for RenderCommandEncoder<'_> {
    fn drop(&mut self) {
        let resolves = std::mem::take(&mut self.resolves);
        if resolves.is_empty() {
            return;
        }
        // Unbind the render targets before resolving: transitioning an attachment
        // out of RENDER_TARGET while it is still bound on the OM stage is flagged
        // by the debug layer.
        unsafe {
            self.encoder
                .list
                .as_ref()
                .unwrap()
                .OMSetRenderTargets(0, None, false, None);
        }
        for r in resolves {
            self.encoder.resolve_view(&r.src, &r.dst, r.mode);
        }
    }
}

pub struct ComputePipelineContext<'a> {
    pub(super) encoder: &'a mut CommandEncoder,
    pub(super) layout: &'a PipelineLayout,
}

pub struct RenderPipelineContext<'a> {
    pub(super) encoder: &'a mut CommandEncoder,
    pub(super) layout: &'a PipelineLayout,
    pub(super) vertex_strides: &'a [u32],
}

pub struct PipelineContext<'a> {
    pub(super) encoder: &'a mut CommandEncoder,
    pub(super) group_descriptors: &'a GroupDescriptors,
    /// Base of this group's CBV/SRV/UAV descriptors in the GPU-visible ring.
    pub(super) ring_cpu_base: D3D12_CPU_DESCRIPTOR_HANDLE,
    /// Base of this group's samplers in the GPU-visible sampler ring (CPU handle).
    /// Each `Sampler::bind_to` copies into `sampler_cpu_base + register`.
    pub(super) sampler_cpu_base: D3D12_CPU_DESCRIPTOR_HANDLE,
}

// ── Trait: CommandDevice ──────────────────────────────────────────────────────

#[hidden_trait::expose]
impl crate::traits::CommandDevice for Context {
    type CommandEncoder = CommandEncoder;
    type SyncPoint = SyncPoint;

    fn create_command_encoder(&self, desc: super::CommandEncoderDesc) -> CommandEncoder {
        let allocators: Vec<ID3D12CommandAllocator> = (0..desc.buffer_count)
            .map(|_| unsafe {
                self.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap()
            })
            .collect();

        let list: ID3D12GraphicsCommandList = unsafe {
            self.device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocators[0], None).unwrap()
        };
        unsafe { list.Close().unwrap() };

        // Name the list so DRED reports it on a hang and we can map the hung node
        // back to this encoder's per-pass label table.
        let id = dred_next_encoder_id();
        let list_name: Vec<u16> = format!("blade-enc-{id}")
            .encode_utf16()
            .chain(std::iter::once(0))
            .collect();
        let _ = unsafe { list.SetName(windows::core::PCWSTR(list_name.as_ptr())) };

        // Each ring is split into `buffer_count` per-frame segments so that
        // resetting one frame's region never clobbers descriptors/upload data a
        // still-in-flight frame is reading (the allocator pool already buffers
        // `buffer_count` frames; the rings must match).
        let segments = desc.buffer_count.max(1);
        // Start small and grow on demand (DescriptorRing::grow), capped at the
        // device's tier max. A transient transfer/copy encoder that binds nothing
        // stays at the start size instead of reserving the full tier heap; the
        // render encoder grows over its first frames to its steady need.
        let cbv_srv_uav_ring = DescriptorRing::new(
            &self.device, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            ENCODER_HEAP_START, self.cbv_srv_uav_heap_size, segments,
        );
        // Samplers cannot grow past the 2048 shader-visible-sampler-heap limit,
        // so start at and cap at that maximum.
        let sampler_ring = DescriptorRing::new(
            &self.device, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
            ENCODER_SAMPLER_SIZE, ENCODER_SAMPLER_SIZE, segments,
        );
        let upload_ring = UploadRing::new(&self.device, UPLOAD_RING_SIZE, segments as u64);

        CommandEncoder {
            list: Some(list),
            id,
            allocators,
            device: self.device.clone(),
            cbv_srv_uav_ring,
            sampler_ring,
            upload_ring,
            present: None,
            timings: Vec::new(),
            staging_heap: self.staging_heap.raw.clone(),
            staging_sampler_heap: self.staging_sampler_heap.raw.clone(),
            staging_increment: self.staging_heap.increment,
            staging_sampler_increment: self.staging_sampler_heap.increment,
            draw_sig: self.draw_sig.clone(),
            draw_indexed_sig: self.draw_indexed_sig.clone(),
            dispatch_sig: self.dispatch_sig.clone(),
            auto_barriers: true,
            buffer_states: std::collections::HashMap::new(),
            texture_states: std::collections::HashMap::new(),
            pending_vertex: Vec::new(),
            fence: self.queue.lock().unwrap().fence.clone(),
            active_render_targets: Vec::new(),
            fence_event: unsafe {
                windows::Win32::System::Threading::CreateEventW(None, false, false, None).unwrap()
            },
            allocator_fences: vec![0u64; desc.buffer_count as usize],
            frame_index: 0,
            retired_cbv_heaps: Vec::new(),
            active_cbv_tables: Vec::new(),
            bound_sampler_roots: None,
        }
    }

    fn destroy_command_encoder(&self, encoder: &mut CommandEncoder) {
        encoder.allocators.clear();
        encoder.list = None;
        unsafe {
            let _ = windows::Win32::Foundation::CloseHandle(encoder.fence_event);
        }
    }

    fn submit(&self, encoder: &mut CommandEncoder, after: Option<&SyncPoint>) -> SyncPoint {
        profiling::function_scope!();

        // Final barrier before closing — matches Vulkan's finish() calling barrier()
        if encoder.auto_barriers {
            encoder.barrier();
        }

        // Return every resource to COMMON before closing. Unlike buffers and
        // implicitly-promoted read states, resources we *explicitly* transitioned
        // (render targets, storage textures, etc.) do NOT decay at the
        // ExecuteCommandLists boundary — so without this they'd carry a non-COMMON
        // state into the next command list while `start()` assumes COMMON. The
        // back buffer is already COMMON here (present() reset it), so it is skipped.
        encoder.flush_to_common();

        let list = encoder.list.as_ref().unwrap();
        expect_d3d12(&self.device, "GraphicsCommandList::Close", unsafe { list.Close() });

        let mut queue = {
            profiling::scope!("lock queue");
            self.queue.lock().unwrap()
        };

        if let Some(sp) = after {
            if let Err(e) = unsafe { queue.raw.Wait(&queue.fence, sp.progress) } {
                panic_if_device_removed(&self.device, "queue Wait");
                panic!("DX12 queue Wait failed: {e}");
            }
        }

        let cmd_lists: [Option<ID3D12CommandList>; 1] =
            [Some(list.cast::<ID3D12CommandList>().unwrap())];
        unsafe { queue.raw.ExecuteCommandLists(&cmd_lists) };
        // ExecuteCommandLists reports a bad command list only via device removal;
        // surface it here (with DRED) instead of at some later silent failure.
        panic_if_device_removed(&self.device, "ExecuteCommandLists");

        queue.last_progress += 1;
        let progress = queue.last_progress;
        if let Err(e) = unsafe { queue.raw.Signal(&queue.fence, progress) } {
            panic_if_device_removed(&self.device, "Signal");
            panic!("DX12 queue Signal failed: {e}");
        }

        // Record this submission's fence value for the allocator currently in use
        // (index 0 — no rotation happens between start() and submit()), so the
        // next start() that rotates back to it CPU-waits before resetting it.
        if let Some(slot) = encoder.allocator_fences.first_mut() {
            *slot = progress;
        }

        if let Some(pres) = encoder.present.take() {
            let sync_interval: u32 = if pres.display_sync { 1 } else { 0 };
            let _ = unsafe { pres.swapchain.Present(sync_interval, DXGI_PRESENT(0)) };
            // Present swallows benign errors (occluded/resize), but a removed
            // device must not pass silently — it is the usual place a prior
            // frame's GPU fault first becomes visible on the CPU.
            panic_if_device_removed(&self.device, "Present");
        }

        SyncPoint { progress }
    }

    fn wait_for(&self, sp: &SyncPoint, timeout_ms: u32) -> bool {
        let queue = self.queue.lock().unwrap();
        if unsafe { queue.fence.GetCompletedValue() } >= sp.progress {
            return true;
        }
        // A removed device resets the fence to UINT64_MAX, so GetCompletedValue
        // above passes and we never reach here — but if it didn't, the event
        // would never fire. Surface a removal rather than blocking/timing out.
        panic_if_device_removed(&self.device, "wait_for");
        unsafe {
            queue.fence.SetEventOnCompletion(sp.progress, queue.fence_event).unwrap();
        }
        let timeout = if timeout_ms == !0 { INFINITE } else { timeout_ms };
        unsafe { WaitForSingleObjectEx(queue.fence_event, timeout, false) == WAIT_EVENT(0u32) }
    }
}

impl Context {
    pub fn wait_for_present(&self, _surface: &Surface, _present_id: u64, _timeout_ms: u32) -> bool {
        true
    }
}

// ── Format mapping ────────────────────────────────────────────────────────────

pub(super) fn map_texture_format(format: crate::TextureFormat) -> DXGI_FORMAT {
    use crate::TextureFormat as Tf;
    match format {
        Tf::R8Unorm => DXGI_FORMAT_R8_UNORM,
        Tf::Rg8Unorm => DXGI_FORMAT_R8G8_UNORM,
        Tf::Rg8Snorm => DXGI_FORMAT_R8G8_SNORM,
        Tf::Rgba8Unorm => DXGI_FORMAT_R8G8B8A8_UNORM,
        Tf::Rgba8UnormSrgb => DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,
        Tf::Bgra8Unorm => DXGI_FORMAT_B8G8R8A8_UNORM,
        Tf::Bgra8UnormSrgb => DXGI_FORMAT_B8G8R8A8_UNORM_SRGB,
        Tf::Rgba8Snorm => DXGI_FORMAT_R8G8B8A8_SNORM,
        Tf::R16Float => DXGI_FORMAT_R16_FLOAT,
        Tf::Rg16Float => DXGI_FORMAT_R16G16_FLOAT,
        Tf::Rgba16Float => DXGI_FORMAT_R16G16B16A16_FLOAT,
        Tf::R32Float => DXGI_FORMAT_R32_FLOAT,
        Tf::Rg32Float => DXGI_FORMAT_R32G32_FLOAT,
        Tf::Rgba32Float => DXGI_FORMAT_R32G32B32A32_FLOAT,
        Tf::R32Uint => DXGI_FORMAT_R32_UINT,
        Tf::Rg32Uint => DXGI_FORMAT_R32G32_UINT,
        Tf::Rgba32Uint => DXGI_FORMAT_R32G32B32A32_UINT,
        Tf::Depth32Float => DXGI_FORMAT_D32_FLOAT,
        Tf::Depth32FloatStencil8Uint => DXGI_FORMAT_D32_FLOAT_S8X24_UINT,
        Tf::Stencil8Uint => DXGI_FORMAT_R8_UINT,
        Tf::Bc1Unorm => DXGI_FORMAT_BC1_UNORM,
        Tf::Bc1UnormSrgb => DXGI_FORMAT_BC1_UNORM_SRGB,
        Tf::Bc2Unorm => DXGI_FORMAT_BC2_UNORM,
        Tf::Bc2UnormSrgb => DXGI_FORMAT_BC2_UNORM_SRGB,
        Tf::Bc3Unorm => DXGI_FORMAT_BC3_UNORM,
        Tf::Bc3UnormSrgb => DXGI_FORMAT_BC3_UNORM_SRGB,
        Tf::Bc4Unorm => DXGI_FORMAT_BC4_UNORM,
        Tf::Bc4Snorm => DXGI_FORMAT_BC4_SNORM,
        Tf::Bc5Unorm => DXGI_FORMAT_BC5_UNORM,
        Tf::Bc5Snorm => DXGI_FORMAT_BC5_SNORM,
        Tf::Bc6hUfloat => DXGI_FORMAT_BC6H_UF16,
        Tf::Bc6hFloat => DXGI_FORMAT_BC6H_SF16,
        Tf::Bc7Unorm => DXGI_FORMAT_BC7_UNORM,
        Tf::Bc7UnormSrgb => DXGI_FORMAT_BC7_UNORM_SRGB,
        Tf::Rgb10a2Unorm => DXGI_FORMAT_R10G10B10A2_UNORM,
        Tf::Rg11b10Ufloat => DXGI_FORMAT_R11G11B10_FLOAT,
        Tf::Rgb9e5Ufloat => DXGI_FORMAT_R9G9B9E5_SHAREDEXP,
    }
}

pub(super) fn map_depth_srv_format(format: crate::TextureFormat) -> DXGI_FORMAT {
    match format {
        crate::TextureFormat::Depth32Float => DXGI_FORMAT_R32_FLOAT,
        crate::TextureFormat::Depth32FloatStencil8Uint => DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS,
        crate::TextureFormat::Stencil8Uint => DXGI_FORMAT_R8_UINT,
        other => map_texture_format(other),
    }
}

/// The format to create the *resource* with. Depth formats use the TYPELESS base
/// so the resource can simultaneously host a typed DSV and a depth-readable SRV
/// (D3D12 only allows that on a typeless resource). Color formats stay typed —
/// they already support both RTV and SRV directly.
/// Attach a debug name to any D3D12 object (resource, pipeline state, …) so
/// RenderDoc/PIX, the debug layer, and DRED show the blade name instead of an
/// auto-generated number — DRED in particular reports the bound PSO by name, so
/// naming pipelines is what turns a device-removal report into something
/// localizable.
pub(super) fn set_resource_name<T: windows::core::Interface>(object: &T, name: &str) {
    if name.is_empty() {
        return;
    }
    let Ok(object) = object.cast::<ID3D12Object>() else { return };
    let wide: Vec<u16> = name.encode_utf16().chain(std::iter::once(0)).collect();
    let _ = unsafe { object.SetName(windows::core::PCWSTR(wide.as_ptr())) };
}

pub(super) fn map_resource_format(format: crate::TextureFormat) -> DXGI_FORMAT {
    match format {
        crate::TextureFormat::Depth32Float => DXGI_FORMAT_R32_TYPELESS,
        crate::TextureFormat::Depth32FloatStencil8Uint => DXGI_FORMAT_R32G8X24_TYPELESS,
        crate::TextureFormat::Stencil8Uint => DXGI_FORMAT_R8_TYPELESS,
        other => map_texture_format(other),
    }
}

pub(super) fn map_comparison(fun: crate::CompareFunction) -> D3D12_COMPARISON_FUNC {
    use crate::CompareFunction as Cf;
    match fun {
        Cf::Never => D3D12_COMPARISON_FUNC_NEVER,
        Cf::Less => D3D12_COMPARISON_FUNC_LESS,
        Cf::LessEqual => D3D12_COMPARISON_FUNC_LESS_EQUAL,
        Cf::Equal => D3D12_COMPARISON_FUNC_EQUAL,
        Cf::GreaterEqual => D3D12_COMPARISON_FUNC_GREATER_EQUAL,
        Cf::Greater => D3D12_COMPARISON_FUNC_GREATER,
        Cf::NotEqual => D3D12_COMPARISON_FUNC_NOT_EQUAL,
        Cf::Always => D3D12_COMPARISON_FUNC_ALWAYS,
    }
}

pub(super) fn map_index_format(ty: crate::IndexType) -> DXGI_FORMAT {
    match ty {
        crate::IndexType::U16 => DXGI_FORMAT_R16_UINT,
        crate::IndexType::U32 => DXGI_FORMAT_R32_UINT,
    }
}

pub(super) fn classify_binding(
    binding: &crate::ShaderBinding,
    access: naga::StorageAccess,
) -> BindingKind {
    match binding {
        crate::ShaderBinding::Texture | crate::ShaderBinding::TextureArray { .. } => {
            // naga's HLSL backend emits every storage texture as RWTexture2D (a `u`
            // register) regardless of read/write access; only sampled textures
            // (empty access) become Texture2D SRVs. A read-only storage texture
            // therefore must be a UAV here, or its register collides with the `u`
            // space naga assigns it.
            if access.is_empty() {
                BindingKind::Srv
            } else {
                BindingKind::Uav
            }
        }
        crate::ShaderBinding::Sampler => BindingKind::Sampler,
        crate::ShaderBinding::Buffer | crate::ShaderBinding::BufferArray { .. } => {
            if access.contains(naga::StorageAccess::STORE) {
                BindingKind::Uav
            } else {
                BindingKind::Srv
            }
        }
        crate::ShaderBinding::AccelerationStructure => BindingKind::Srv,
        crate::ShaderBinding::Plain { .. } => BindingKind::Cbv,
    }
}

pub(super) fn analyze_group(
    layout: &crate::ShaderDataLayout,
    info: &crate::ShaderDataInfo,
    _group_index: u32,
    root_base: &mut u32,
) -> GroupDescriptors {
    let mut srv = 0u32;
    let mut uav = 0u32;
    let mut cbv = 0u32;
    let mut smp = 0u32;
    for (bi, (_, binding)) in layout.bindings.iter().enumerate() {
        match classify_binding(binding, info.binding_access[bi]) {
            BindingKind::Srv => srv += 1,
            BindingKind::Uav => uav += 1,
            BindingKind::Cbv => cbv += 1,
            BindingKind::Sampler => smp += 1,
        }
    }
    let cbv_srv_uav_count = srv + uav + cbv;
    let cbv_srv_uav_root_index = if cbv_srv_uav_count > 0 {
        let i = *root_base; *root_base += 1; Some(i)
    } else { None };
    // Root SRV holding this group's sampler-index buffer (see GroupDescriptors).
    let sampler_index_root_index = if smp > 0 {
        let i = *root_base; *root_base += 1; Some(i)
    } else { None };

    let mut srv_off = 0u32; let mut uav_off = srv;
    let mut cbv_off = srv + uav;
    let mut srv_reg = 0u32; let mut uav_reg = 0u32;
    let mut cbv_reg = 0u32; let mut smp_reg = 0u32;

    let slots = layout.bindings.iter().enumerate().map(|(bi, (_, binding))| {
        let access = info.binding_access[bi];
        let kind = classify_binding(binding, access);
        let (heap_offset, register) = match kind {
            BindingKind::Srv => { let v = (srv_off, srv_reg); srv_off += 1; srv_reg += 1; v }
            BindingKind::Uav => { let v = (uav_off, uav_reg); uav_off += 1; uav_reg += 1; v }
            BindingKind::Cbv => { let v = (cbv_off, cbv_reg); cbv_off += 1; cbv_reg += 1; v }
            // For samplers `heap_offset` is unused; `register` is the index into
            // this group's sampler-index buffer (== slot in the sampler ring).
            BindingKind::Sampler => { let v = (0u32, smp_reg); smp_reg += 1; v }
        };
        BindingSlot { heap_offset, register, kind }
    }).collect();

    GroupDescriptors {
        cbv_srv_uav_root_index,
        sampler_index_root_index,
        cbv_srv_uav_count,
        sampler_count: smp,
        slots,
    }
}
