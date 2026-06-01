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
    /// Frame-latency waitable handle of the swapchain. `acquire_frame` blocks on
    /// it so the CPU does not race ahead and render into a back buffer still
    /// queued for presentation — the DX12 analog of Vulkan's blocking
    /// `acquire_next_image` (DXGI's FLIP model otherwise lets the CPU outrun the
    /// present queue, which shows up as occasional whole-frame flicker/tearing).
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

/// Dump DRED auto-breadcrumbs and page-fault data after a device removal. The
/// last completed breadcrumb value vs. the op count identifies the command-list
/// operation the GPU was executing when it hung; the page-fault VA + the named
/// allocation it falls in identifies a bad address. Only populated when DRED was
/// armed at init (validation builds).
pub(super) fn log_dred(device: &ID3D12Device) {
    let dred: ID3D12DeviceRemovedExtendedData = match device.cast() {
        Ok(d) => d,
        Err(_) => return, // DRED not armed
    };
    unsafe {
        if let Ok(bc) = dred.GetAutoBreadcrumbsOutput() {
            let mut node = bc.pHeadAutoBreadcrumbNode;
            while !node.is_null() {
                let n = &*node;
                let list = pcwstr_to_string(n.pCommandListDebugNameW);
                let queue = pcwstr_to_string(n.pCommandQueueDebugNameW);
                let done = if n.pLastBreadcrumbValue.is_null() {
                    0
                } else {
                    *n.pLastBreadcrumbValue
                };
                // A node fully executed (done == count) didn't hang; report the
                // one that stopped partway, where op[done] is the hung operation.
                if done < n.BreadcrumbCount {
                    log::error!(
                        "[DRED] command list '{list}' (queue '{queue}') hung after {done}/{} ops; hung op = {:?}",
                        n.BreadcrumbCount,
                        if n.pCommandHistory.is_null() {
                            D3D12_AUTO_BREADCRUMB_OP(-1)
                        } else {
                            *n.pCommandHistory.add(done as usize)
                        },
                    );
                }
                node = n.pNext;
            }
        }
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
pub(super) struct DescriptorRing {
    pub heap: ID3D12DescriptorHeap,
    pub cpu_start: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub gpu_start: D3D12_GPU_DESCRIPTOR_HANDLE,
    pub increment: u32,
    pub offset: u32,
    pub capacity: u32,
    segment_count: u32,
    segment: u32,
}

unsafe impl Send for DescriptorRing {}

impl DescriptorRing {
    fn new(
        device: &ID3D12Device,
        ty: D3D12_DESCRIPTOR_HEAP_TYPE,
        capacity: u32,
        segment_count: u32,
    ) -> Self {
        // A large shader-visible heap can exceed available descriptor memory
        // (especially with GPU-based validation, which shadows descriptors). Rather
        // than panic on E_OUTOFMEMORY, halve and retry down to a usable floor so the
        // app still boots; if the result is too small for a frame, the per-segment
        // overflow assert fires with a clear message instead of a startup crash.
        let mut cap = capacity.max(segment_count);
        let floor = (65536 * segment_count).min(capacity);
        let (heap, capacity): (ID3D12DescriptorHeap, u32) = loop {
            let result: windows::core::Result<ID3D12DescriptorHeap> = unsafe {
                device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: ty,
                    NumDescriptors: cap,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    NodeMask: 0,
                })
            };
            match result {
                Ok(heap) => break (heap, cap),
                Err(e) if cap > floor => {
                    log::warn!("CreateDescriptorHeap({cap}) failed ({e}); retrying smaller");
                    cap = (cap / 2).max(floor);
                }
                Err(e) => panic!("CreateDescriptorHeap({cap}) failed: {e}"),
            }
        };
        let cpu_start = unsafe { heap.GetCPUDescriptorHandleForHeapStart() };
        let gpu_start = unsafe { heap.GetGPUDescriptorHandleForHeapStart() };
        let increment = unsafe { device.GetDescriptorHandleIncrementSize(ty) };
        Self { heap, cpu_start, gpu_start, increment, offset: 0, capacity, segment_count, segment: 0 }
    }

    fn segment_capacity(&self) -> u32 {
        self.capacity / self.segment_count
    }

    pub fn begin_segment(&mut self, segment: u32) {
        self.segment = segment;
        self.offset = segment * self.segment_capacity();
    }

    pub fn alloc(&mut self, count: u32) -> (D3D12_CPU_DESCRIPTOR_HANDLE, D3D12_GPU_DESCRIPTOR_HANDLE) {
        let limit = (self.segment + 1) * self.segment_capacity();
        assert!(self.offset + count <= limit, "descriptor ring segment overflow");
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

        // Each ring is split into `buffer_count` per-frame segments so that
        // resetting one frame's region never clobbers descriptors/upload data a
        // still-in-flight frame is reading (the allocator pool already buffers
        // `buffer_count` frames; the rings must match).
        let segments = desc.buffer_count.max(1);
        let cbv_srv_uav_ring = DescriptorRing::new(
            &self.device, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, self.cbv_srv_uav_heap_size, segments,
        );
        let sampler_ring = DescriptorRing::new(
            &self.device, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, ENCODER_SAMPLER_SIZE, segments,
        );
        let upload_ring = UploadRing::new(&self.device, UPLOAD_RING_SIZE, segments as u64);

        CommandEncoder {
            list: Some(list),
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
            unsafe { queue.raw.Wait(&queue.fence, sp.progress).unwrap() };
        }

        let cmd_lists: [Option<ID3D12CommandList>; 1] =
            [Some(list.cast::<ID3D12CommandList>().unwrap())];
        unsafe { queue.raw.ExecuteCommandLists(&cmd_lists) };

        queue.last_progress += 1;
        let progress = queue.last_progress;
        unsafe { queue.raw.Signal(&queue.fence, progress).unwrap() };

        // Record this submission's fence value for the allocator currently in use
        // (index 0 — no rotation happens between start() and submit()), so the
        // next start() that rotates back to it CPU-waits before resetting it.
        if let Some(slot) = encoder.allocator_fences.first_mut() {
            *slot = progress;
        }

        if let Some(pres) = encoder.present.take() {
            let sync_interval: u32 = if pres.display_sync { 1 } else { 0 };
            let _ = unsafe { pres.swapchain.Present(sync_interval, DXGI_PRESENT(0)) };
        }

        SyncPoint { progress }
    }

    fn wait_for(&self, sp: &SyncPoint, timeout_ms: u32) -> bool {
        let queue = self.queue.lock().unwrap();
        if unsafe { queue.fence.GetCompletedValue() } >= sp.progress {
            return true;
        }
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
/// Attach a debug name to a resource so RenderDoc/PIX/the debug layer show the
/// blade name (e.g. "color with mips") instead of an auto-generated number.
pub(super) fn set_resource_name(resource: &ID3D12Resource, name: &str) {
    if name.is_empty() {
        return;
    }
    let wide: Vec<u16> = name.encode_utf16().chain(std::iter::once(0)).collect();
    let _ = unsafe { resource.SetName(windows::core::PCWSTR(wide.as_ptr())) };
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
