use std::{mem, ptr, sync::Mutex};
use windows::{
    core::Interface,
    Win32::{
        Foundation::HANDLE,
        Graphics::{
            Direct3D12::*,
            Dxgi::{Common::*, *},
        },
        System::Threading::{CreateEventW, WaitForSingleObjectEx, INFINITE},
    },
};

mod command;
mod init;
mod pipeline;
mod resource;
mod surface;

// ── Descriptor heap sizes ─────────────────────────────────────────────────────

/// CPU-only heap for render target views.
pub(super) const RTV_HEAP_SIZE: u32 = 512;
/// CPU-only heap for depth/stencil views.
pub(super) const DSV_HEAP_SIZE: u32 = 128;
/// CPU-only staging heap for CBV/SRV/UAV descriptors (used when creating resources).
pub(super) const STAGING_HEAP_SIZE: u32 = 65536;
/// CPU-only staging heap for sampler descriptors.
pub(super) const STAGING_SAMPLER_SIZE: u32 = 2048;
/// Per-encoder GPU-visible CBV/SRV/UAV heap.
pub(super) const ENCODER_HEAP_SIZE: u32 = 65536;
/// Per-encoder GPU-visible sampler heap (DX12 max is 2048).
pub(super) const ENCODER_SAMPLER_SIZE: u32 = 2048;
/// Per-encoder upload scratch ring size (for Plain/CBV data).
pub(super) const UPLOAD_RING_SIZE: u64 = 4 * 1024 * 1024; // 4 MiB

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
        let increment =
            unsafe { device.GetDescriptorHandleIncrementSize(ty) };
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

// ── Descriptor allocator (linear, used for static heaps) ─────────────────────

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
    queue: Mutex<Queue>,
    // Static CPU-only descriptor heaps
    pub(super) rtv_heap: DescriptorHeap,
    pub(super) dsv_heap: DescriptorHeap,
    pub(super) staging_heap: DescriptorHeap,
    pub(super) staging_sampler_heap: DescriptorHeap,
    // Linear allocators for static heaps
    pub(super) rtv_alloc: Mutex<LinearAllocator>,
    pub(super) dsv_alloc: Mutex<LinearAllocator>,
    pub(super) staging_alloc: Mutex<LinearAllocator>,
    pub(super) staging_sampler_alloc: Mutex<LinearAllocator>,
    // Command signatures for indirect execution (no root arg changes)
    pub(super) draw_sig: ID3D12CommandSignature,
    pub(super) draw_indexed_sig: ID3D12CommandSignature,
    pub(super) dispatch_sig: ID3D12CommandSignature,
    pub(super) validation_mode: bool,
    pub(super) device_information: crate::DeviceInformation,
    pub(super) vendor: crate::GpuVendor,
    pub(super) factory: IDXGIFactory4,
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
        Texture {
            resource: self.resource.clone(),
            format: self.format,
            target_size: self.target_size,
        }
    }

    pub fn texture_view(&self) -> TextureView {
        TextureView {
            resource: self.resource.clone(),
            srv_handle: 0,
            uav_handle: 0,
            rtv_dsv_handle: self.rtv.ptr as u64,
            format: self.format,
            aspects: crate::TexelAspects::COLOR,
            target_size: self.target_size,
        }
    }
}

// ── Synchronization ───────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct SyncPoint {
    progress: u64,
}

// ── Resource types ────────────────────────────────────────────────────────────

/// CPU descriptor handle stored as raw usize value (0 = not present).
type RawCpuHandle = u64;

pub(super) fn cpu_handle_raw(h: D3D12_CPU_DESCRIPTOR_HANDLE) -> RawCpuHandle {
    h.ptr as u64
}
pub(super) fn raw_cpu_handle(v: RawCpuHandle) -> D3D12_CPU_DESCRIPTOR_HANDLE {
    D3D12_CPU_DESCRIPTOR_HANDLE { ptr: v as usize }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq)]
pub struct Buffer {
    pub(super) resource_ptr: *mut std::ffi::c_void,
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
        unsafe { &*(self.resource_ptr as *const ID3D12Resource) }
    }
}

/// A raw, non-refcounted handle to a `ID3D12Resource`.
/// Created by `Box::into_raw(Box::new(resource)) as *mut c_void`.
/// Destroyed by `Box::from_raw(ptr as *mut ID3D12Resource)` + drop.
pub(super) type ResourcePtr = *mut std::ffi::c_void;

pub(super) unsafe fn resource_ref(ptr: ResourcePtr) -> &'static ID3D12Resource {
    &*(ptr as *const ID3D12Resource)
}

#[derive(Clone, Copy, Debug, Hash, PartialEq)]
pub struct Texture {
    pub(super) resource_ptr: ResourcePtr,
    pub(super) format: crate::TextureFormat,
    pub(super) target_size: [u16; 2],
}

unsafe impl Send for Texture {}
unsafe impl Sync for Texture {}

impl Default for Texture {
    fn default() -> Self {
        panic!("Texture::default() not supported on DX12")
    }
}

impl Texture {
    pub(super) fn resource(&self) -> &ID3D12Resource {
        unsafe { resource_ref(self.resource_ptr) }
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq)]
pub struct TextureView {
    pub(super) resource_ptr: ResourcePtr,
    pub(super) srv_handle: RawCpuHandle,
    pub(super) uav_handle: RawCpuHandle,
    pub(super) rtv_dsv_handle: RawCpuHandle,
    pub(super) format: crate::TextureFormat,
    pub(super) aspects: crate::TexelAspects,
    pub(super) target_size: [u16; 2],
}

unsafe impl Send for TextureView {}
unsafe impl Sync for TextureView {}

impl Default for TextureView {
    fn default() -> Self {
        panic!("TextureView::default() not supported on DX12")
    }
}

impl TextureView {
    pub(super) fn resource(&self) -> &ID3D12Resource {
        unsafe { resource_ref(self.resource_ptr) }
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
        panic!("Sampler::default() not supported on DX12")
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

/// Per-binding metadata in a group descriptor layout.
#[derive(Clone, Copy, Debug)]
pub(super) struct BindingSlot {
    /// Index into the group's CBV/SRV/UAV or sampler heap allocation.
    pub heap_offset: u32,
    /// Re-mapped HLSL register number (0-based within its type).
    pub register: u32,
    pub kind: BindingKind,
}

/// Descriptor layout for one bind group.
#[derive(Clone, Debug)]
pub(super) struct GroupDescriptors {
    /// Root parameter index for the CBV/SRV/UAV table (None if no such bindings).
    pub cbv_srv_uav_root_index: Option<u32>,
    /// Root parameter index for the sampler table (None if no sampler bindings).
    pub sampler_root_index: Option<u32>,
    /// Number of CBV/SRV/UAV descriptors needed per bind call.
    pub cbv_srv_uav_count: u32,
    /// Number of sampler descriptors needed per bind call.
    pub sampler_count: u32,
    /// One entry per binding in group layout order.
    pub slots: Vec<BindingSlot>,
}

#[derive(Debug)]
pub(super) struct PipelineLayout {
    pub root_signature: ID3D12RootSignature,
    pub groups: Vec<GroupDescriptors>,
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
    /// Stride per vertex buffer binding slot (indexed by slot index).
    pub(super) vertex_strides: Vec<u32>,
}

unsafe impl Send for RenderPipeline {}
unsafe impl Sync for RenderPipeline {}

// ── Command encoder ────────────────────────────────────────────────────────────

/// Per-encoder GPU-visible descriptor ring (resets each frame).
pub(super) struct DescriptorRing {
    pub heap: ID3D12DescriptorHeap,
    pub cpu_start: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub gpu_start: D3D12_GPU_DESCRIPTOR_HANDLE,
    pub increment: u32,
    pub offset: u32,
    pub capacity: u32,
}

unsafe impl Send for DescriptorRing {}

impl DescriptorRing {
    fn new(device: &ID3D12Device, ty: D3D12_DESCRIPTOR_HEAP_TYPE, capacity: u32) -> Self {
        let heap: ID3D12DescriptorHeap = unsafe {
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: ty,
                NumDescriptors: capacity,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            }).unwrap()
        };
        let cpu_start = unsafe { heap.GetCPUDescriptorHandleForHeapStart() };
        let gpu_start = unsafe { heap.GetGPUDescriptorHandleForHeapStart() };
        let increment = unsafe { device.GetDescriptorHandleIncrementSize(ty) };
        Self { heap, cpu_start, gpu_start, increment, offset: 0, capacity }
    }

    pub fn reset(&mut self) {
        self.offset = 0;
    }

    /// Allocate `count` consecutive descriptors; return (cpu_start, gpu_start).
    pub fn alloc(&mut self, count: u32) -> (D3D12_CPU_DESCRIPTOR_HANDLE, D3D12_GPU_DESCRIPTOR_HANDLE) {
        assert!(self.offset + count <= self.capacity, "descriptor ring overflow");
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

/// Per-encoder upload scratch ring (for CBV plain-data copies).
pub(super) struct UploadRing {
    pub resource: ID3D12Resource,
    pub gpu_address: u64,
    pub mapped: *mut u8,
    pub offset: u64,
    pub capacity: u64,
}

unsafe impl Send for UploadRing {}

impl UploadRing {
    fn new(device: &ID3D12Device, capacity: u64) -> Self {
        let mut resource: Option<ID3D12Resource> = None;
        unsafe {
            device.CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES {
                    Type: D3D12_HEAP_TYPE_UPLOAD,
                    ..Default::default()
                },
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
        unsafe {
            resource.Map(0, None, Some(&mut mapped_ptr)).unwrap();
        }
        Self {
            resource,
            gpu_address,
            mapped: mapped_ptr as *mut u8,
            offset: 0,
            capacity,
        }
    }

    pub fn reset(&mut self) {
        self.offset = 0;
    }

    /// Write `data` to the ring, aligned to 256 bytes (DX12 CBV alignment).
    /// Returns the GPU address of the written data.
    pub fn write_cbv(&mut self, data: &[u8]) -> u64 {
        const CBV_ALIGN: u64 = 256;
        let aligned_offset = (self.offset + CBV_ALIGN - 1) & !(CBV_ALIGN - 1);
        let end = aligned_offset + data.len() as u64;
        assert!(end <= self.capacity, "upload ring overflow");
        unsafe {
            ptr::copy_nonoverlapping(data.as_ptr(), self.mapped.add(aligned_offset as usize), data.len());
        }
        self.offset = end;
        self.gpu_address + aligned_offset
    }
}

pub(super) struct Presentation {
    pub swapchain: IDXGISwapChain3,
    pub buffer_index: u32,
    pub present_id: u64,
    pub display_sync: bool,
}

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
}

unsafe impl Send for CommandEncoder {}

pub struct TransferCommandEncoder<'a> {
    pub(super) list: &'a ID3D12GraphicsCommandList,
    pub(super) device: &'a ID3D12Device,
}

pub struct AccelerationStructureCommandEncoder<'a> {
    pub(super) list: &'a ID3D12GraphicsCommandList,
}

pub struct ComputeCommandEncoder<'a> {
    pub(super) encoder: &'a mut CommandEncoder,
}

pub struct RenderCommandEncoder<'a> {
    pub(super) encoder: &'a mut CommandEncoder,
    pub(super) target_size: [u16; 2],
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
    /// Base CPU address in the GPU-visible ring for CBV/SRV/UAV of this group.
    pub(super) ring_cpu_base: D3D12_CPU_DESCRIPTOR_HANDLE,
    /// Base GPU address in the ring for CBV/SRV/UAV.
    pub(super) ring_gpu_base: D3D12_GPU_DESCRIPTOR_HANDLE,
    /// Base CPU address in the sampler ring for this group.
    pub(super) sampler_cpu_base: D3D12_CPU_DESCRIPTOR_HANDLE,
    /// Base GPU address in the sampler ring.
    pub(super) sampler_gpu_base: D3D12_GPU_DESCRIPTOR_HANDLE,
    pub(super) is_compute: bool,
    pub(super) root_index_cbv_srv_uav: Option<u32>,
    pub(super) root_index_sampler: Option<u32>,
}

// ── Trait: CommandDevice ──────────────────────────────────────────────────────

#[hidden_trait::expose]
impl crate::traits::CommandDevice for Context {
    type CommandEncoder = CommandEncoder;
    type SyncPoint = SyncPoint;

    fn create_command_encoder(&self, desc: super::CommandEncoderDesc) -> CommandEncoder {
        let allocators: Vec<ID3D12CommandAllocator> = (0..desc.buffer_count)
            .map(|_| unsafe {
                self.device
                    .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                    .unwrap()
            })
            .collect();

        let list: ID3D12GraphicsCommandList = unsafe {
            self.device
                .CreateCommandList(
                    0,
                    D3D12_COMMAND_LIST_TYPE_DIRECT,
                    &allocators[0],
                    None,
                )
                .unwrap()
        };
        // Close immediately; we'll reset/open at start()
        unsafe { list.Close().unwrap() };

        let cbv_srv_uav_ring =
            DescriptorRing::new(&self.device, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, ENCODER_HEAP_SIZE);
        let sampler_ring =
            DescriptorRing::new(&self.device, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, ENCODER_SAMPLER_SIZE);
        let upload_ring = UploadRing::new(&self.device, UPLOAD_RING_SIZE);

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
        }
    }

    fn destroy_command_encoder(&self, encoder: &mut CommandEncoder) {
        // Drop all allocators (and list) — COM refs released on drop
        encoder.allocators.clear();
        encoder.list = None;
    }

    fn submit(&self, encoder: &mut CommandEncoder, after: Option<&SyncPoint>) -> SyncPoint {
        profiling::function_scope!();
        let list = encoder.list.as_ref().unwrap();
        unsafe { list.Close().unwrap() };

        let mut queue = self.queue.lock().unwrap();

        // GPU-side dependency on a previous sync point
        if let Some(sp) = after {
            unsafe {
                queue.raw.Wait(&queue.fence, sp.progress).unwrap();
            }
        }

        let cmd_lists: [Option<ID3D12CommandList>; 1] =
            [Some(list.cast::<ID3D12CommandList>().unwrap())];
        unsafe { queue.raw.ExecuteCommandLists(&cmd_lists) };

        queue.last_progress += 1;
        let progress = queue.last_progress;
        unsafe { queue.raw.Signal(&queue.fence, progress).unwrap() };

        if let Some(pres) = encoder.present.take() {
            // Record which fence value the swapchain buffer was last used at
            let sync_interval: u32 = if pres.display_sync { 1 } else { 0 };
            unsafe { pres.swapchain.Present(sync_interval, 0).ok() };
        }

        SyncPoint { progress }
    }

    fn wait_for(&self, sp: &SyncPoint, timeout_ms: u32) -> bool {
        let queue = self.queue.lock().unwrap();
        let completed = unsafe { queue.fence.GetCompletedValue() };
        if completed >= sp.progress {
            return true;
        }
        unsafe {
            queue
                .fence
                .SetEventOnCompletion(sp.progress, queue.fence_event)
                .unwrap();
        }
        let timeout = if timeout_ms == !0 { INFINITE } else { timeout_ms };
        use windows::Win32::Foundation::WAIT_EVENT;
        let result = unsafe { WaitForSingleObjectEx(queue.fence_event, timeout, false) };
        result == WAIT_EVENT(0u32) // WAIT_OBJECT_0
    }
}

impl Context {
    pub fn wait_for_present(&self, _surface: &Surface, _present_id: u64, _timeout_ms: u32) -> bool {
        // DXGI doesn't expose a direct present-wait primitive; return true immediately.
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

/// For depth formats, return the typeless parent format suitable for SRV.
pub(super) fn map_depth_srv_format(format: crate::TextureFormat) -> DXGI_FORMAT {
    match format {
        crate::TextureFormat::Depth32Float => DXGI_FORMAT_R32_FLOAT,
        crate::TextureFormat::Depth32FloatStencil8Uint => DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS,
        crate::TextureFormat::Stencil8Uint => DXGI_FORMAT_R8_UINT,
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
            if access.contains(naga::StorageAccess::STORE) {
                BindingKind::Uav
            } else {
                BindingKind::Srv
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
    group_index: u32,
    root_base: &mut u32,
) -> GroupDescriptors {
    let mut srv_counter = 0u32;
    let mut uav_counter = 0u32;
    let mut cbv_counter = 0u32;
    let mut sampler_counter = 0u32;

    // First pass: count by kind
    for (bi, (_, binding)) in layout.bindings.iter().enumerate() {
        let access = info.binding_access[bi];
        match classify_binding(binding, access) {
            BindingKind::Srv => srv_counter += 1,
            BindingKind::Uav => uav_counter += 1,
            BindingKind::Cbv => cbv_counter += 1,
            BindingKind::Sampler => sampler_counter += 1,
        }
    }

    let cbv_srv_uav_count = srv_counter + uav_counter + cbv_counter;

    // Assign root parameter indices
    let cbv_srv_uav_root_index = if cbv_srv_uav_count > 0 {
        let idx = *root_base;
        *root_base += 1;
        Some(idx)
    } else {
        None
    };
    let sampler_root_index = if sampler_counter > 0 {
        let idx = *root_base;
        *root_base += 1;
        Some(idx)
    } else {
        None
    };

    // Second pass: assign heap_offset and remapped register per binding
    let mut srv_heap_offset = 0u32;
    let mut uav_heap_offset = srv_counter;
    let mut cbv_heap_offset = srv_counter + uav_counter;
    let mut srv_reg = 0u32;
    let mut uav_reg = 0u32;
    let mut cbv_reg = 0u32;
    let mut sampler_heap_offset = 0u32;
    let mut sampler_reg = 0u32;

    let slots = layout
        .bindings
        .iter()
        .enumerate()
        .map(|(bi, (_, binding))| {
            let access = info.binding_access[bi];
            let kind = classify_binding(binding, access);
            let (heap_offset, register) = match kind {
                BindingKind::Srv => {
                    let (h, r) = (srv_heap_offset, srv_reg);
                    srv_heap_offset += 1;
                    srv_reg += 1;
                    (h, r)
                }
                BindingKind::Uav => {
                    let (h, r) = (uav_heap_offset, uav_reg);
                    uav_heap_offset += 1;
                    uav_reg += 1;
                    (h, r)
                }
                BindingKind::Cbv => {
                    let (h, r) = (cbv_heap_offset, cbv_reg);
                    cbv_heap_offset += 1;
                    cbv_reg += 1;
                    (h, r)
                }
                BindingKind::Sampler => {
                    let (h, r) = (sampler_heap_offset, sampler_reg);
                    sampler_heap_offset += 1;
                    sampler_reg += 1;
                    (h, r)
                }
            };
            BindingSlot { heap_offset, register, kind }
        })
        .collect();

    GroupDescriptors {
        cbv_srv_uav_root_index,
        sampler_root_index,
        cbv_srv_uav_count,
        sampler_count: sampler_counter,
        slots,
    }
}
