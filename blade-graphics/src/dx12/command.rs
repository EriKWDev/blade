use std::mem;
use windows::{
    core::Interface,
    Win32::{
        Foundation::RECT,
        Graphics::{Direct3D12::*, Dxgi::Common::DXGI_FORMAT_R32_TYPELESS},
    },
};

// ── ShaderBindable impls ──────────────────────────────────────────────────────

impl<T: bytemuck::Pod> crate::ShaderBindable for T {
    fn bind_to(&self, ctx: &mut super::PipelineContext, index: u32) {
        let slot = &ctx.group_descriptors.slots[index as usize];
        debug_assert_eq!(slot.kind, super::BindingKind::Cbv);

        // Copy the full plain data into the upload ring (arbitrary size up to the
        // 64 KiB CBV limit — e.g. large `materials` uniform arrays), then point a
        // CBV at it. SizeInBytes must be a multiple of 256.
        let data = bytemuck::bytes_of(self);
        let gpu_addr = ctx.encoder.upload_ring.write_cbv(data);
        let size = ((data.len() as u32) + 255) & !255;

        let cbv_cpu = D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: ctx.ring_cpu_base.ptr
                + (slot.heap_offset * ctx.encoder.cbv_srv_uav_ring.increment) as usize,
        };
        unsafe {
            ctx.encoder.device.CreateConstantBufferView(
                Some(&D3D12_CONSTANT_BUFFER_VIEW_DESC {
                    BufferLocation: gpu_addr,
                    SizeInBytes: size.max(256),
                }),
                cbv_cpu,
            );
        }
    }
}

impl crate::ShaderBindable for super::TextureView {
    fn bind_to(&self, ctx: &mut super::PipelineContext, index: u32) {
        let slot = &ctx.group_descriptors.slots[index as usize];
        let src = match slot.kind {
            super::BindingKind::Srv => {
                // Storage/sampled textures do NOT auto-promote to a read state from
                // an arbitrary prior state (e.g. UNORDERED_ACCESS), so transition.
                ctx.encoder.require_view_state(self, READ_STATE);
                super::raw_cpu_handle(self.srv_handle)
            }
            super::BindingKind::Uav => {
                // Regular textures never auto-promote COMMON->UNORDERED_ACCESS, so
                // this explicit transition is mandatory for storage textures.
                ctx.encoder
                    .require_view_state(self, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                super::raw_cpu_handle(self.uav_handle)
            }
            other => panic!("unexpected binding kind {:?} for TextureView", other),
        };
        let dst = D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: ctx.ring_cpu_base.ptr
                + (slot.heap_offset * ctx.encoder.cbv_srv_uav_ring.increment) as usize,
        };
        unsafe {
            ctx.encoder
                .device
                .CopyDescriptorsSimple(1, dst, src, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
        };
    }
}

impl<'a, const N: crate::ResourceIndex> crate::ShaderBindable for &'a crate::TextureArray<N> {
    fn bind_to(&self, _ctx: &mut super::PipelineContext, _index: u32) {
        unimplemented!("TextureArray binding not yet implemented for DX12")
    }
}

impl crate::ShaderBindable for super::Sampler {
    fn bind_to(&self, ctx: &mut super::PipelineContext, index: u32) {
        let slot = &ctx.group_descriptors.slots[index as usize];
        debug_assert_eq!(slot.kind, super::BindingKind::Sampler);
        // Copy the sampler into ring slot (base + register). The shader reads
        // nagaSamplerHeap[indexbuf[register]] where indexbuf[register] = base + register.
        let src = super::raw_cpu_handle(self.cpu_handle);
        let dst = D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: ctx.sampler_cpu_base.ptr
                + (slot.register * ctx.encoder.sampler_ring.increment) as usize,
        };
        unsafe {
            ctx.encoder.device.CopyDescriptorsSimple(1, dst, src, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER)
        };
    }
}

impl crate::ShaderBindable for crate::BufferPiece {
    fn bind_to(&self, ctx: &mut super::PipelineContext, index: u32) {
        let slot = &ctx.group_descriptors.slots[index as usize];
        let dst = D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: ctx.ring_cpu_base.ptr
                + (slot.heap_offset * ctx.encoder.cbv_srv_uav_ring.increment) as usize,
        };
        // The buffer's pre-made SRV/UAV cover the whole buffer from offset 0, but a
        // BufferPiece can point mid-buffer (belt sub-allocations). Create the view
        // inline with FirstElement = offset/4 so the shader reads from the right
        // place — otherwise sub-allocated bindings (e.g. egui vertices) read garbage.
        let raw_first = (self.offset / 4) as u32;
        let raw_num = (self.buffer.size.saturating_sub(self.offset) / 4) as u32;
        match slot.kind {
            super::BindingKind::Srv => {
                ctx.encoder.require_buffer_state(&self.buffer, READ_STATE);
                unsafe {
                    ctx.encoder.device.CreateShaderResourceView(
                        self.buffer.resource(),
                        Some(&D3D12_SHADER_RESOURCE_VIEW_DESC {
                            Format: DXGI_FORMAT_R32_TYPELESS,
                            ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                                Buffer: D3D12_BUFFER_SRV {
                                    FirstElement: raw_first as u64,
                                    NumElements: raw_num.max(1),
                                    StructureByteStride: 0,
                                    Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                                },
                            },
                        }),
                        dst,
                    )
                };
            }
            super::BindingKind::Uav => {
                ctx.encoder
                    .require_buffer_state(&self.buffer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                unsafe {
                    ctx.encoder.device.CreateUnorderedAccessView(
                        self.buffer.resource(),
                        None,
                        Some(&D3D12_UNORDERED_ACCESS_VIEW_DESC {
                            Format: DXGI_FORMAT_R32_TYPELESS,
                            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                                Buffer: D3D12_BUFFER_UAV {
                                    FirstElement: raw_first as u64,
                                    NumElements: raw_num.max(1),
                                    StructureByteStride: 0,
                                    CounterOffsetInBytes: 0,
                                    Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                                },
                            },
                        }),
                        dst,
                    )
                };
            }
            super::BindingKind::Cbv => {
                ctx.encoder
                    .require_buffer_state(&self.buffer, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
                let addr = self.buffer.gpu_address + self.offset;
                let size = ((self.buffer.size.saturating_sub(self.offset) as u32) + 255) & !255;
                unsafe {
                    ctx.encoder.device.CreateConstantBufferView(
                        Some(&D3D12_CONSTANT_BUFFER_VIEW_DESC {
                            BufferLocation: addr,
                            SizeInBytes: size.max(256),
                        }),
                        dst,
                    )
                };
            }
            other => panic!("unexpected binding kind {:?} for BufferPiece", other),
        }
    }
}

impl<'a, const N: crate::ResourceIndex> crate::ShaderBindable for &'a crate::BufferArray<N> {
    fn bind_to(&self, _ctx: &mut super::PipelineContext, _index: u32) {
        unimplemented!("BufferArray binding not yet implemented for DX12")
    }
}

impl crate::ShaderBindable for crate::AccelerationStructure {
    fn bind_to(&self, _ctx: &mut super::PipelineContext, _index: u32) {
        unimplemented!("AccelerationStructure not yet implemented for DX12")
    }
}

// ── Barrier helpers ───────────────────────────────────────────────────────────

/// Create a transition barrier that borrows the resource without changing its refcount.
fn transition_barrier(
    resource_vtbl: super::ResourcePtr,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
    subresource: u32,
) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                // from_raw does NOT AddRef; ManuallyDrop prevents Release → net 0 refcount change.
                pResource: mem::ManuallyDrop::new(Some(unsafe {
                    ID3D12Resource::from_raw(resource_vtbl as *mut _)
                })),
                Subresource: subresource,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

/// Combined shader-read state (works for both compute and graphics reads).
const READ_STATE: D3D12_RESOURCE_STATES = D3D12_RESOURCE_STATES(
    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE.0 | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE.0,
);

// ── CommandEncoder: private helpers ──────────────────────────────────────────

impl super::CommandEncoder {
    /// Transition a specific set of texture subresources to `needed`, tracking
    /// per-subresource state. Per-subresource granularity is what lets e.g.
    /// mip-generation read mip i-1 (SRV) while writing mip i (UAV) of the same
    /// texture — a whole-resource transition would put both mips in one state.
    fn require_subresources(
        &mut self,
        vtbl: super::ResourcePtr,
        total: u32,
        subs: impl Iterator<Item = u32>,
        needed: D3D12_RESOURCE_STATES,
    ) {
        if vtbl.is_null() {
            return; // null placeholder view
        }
        let mut barriers: Vec<D3D12_RESOURCE_BARRIER> = Vec::new();
        {
            let entry = self
                .texture_states
                .entry(vtbl as usize)
                .or_insert_with(|| vec![D3D12_RESOURCE_STATE_COMMON; total.max(1) as usize]);
            for s in subs {
                let cur = entry[s as usize];
                if cur.0 != needed.0 {
                    barriers.push(transition_barrier(vtbl, cur, needed, s));
                    entry[s as usize] = needed;
                }
            }
        }
        if !barriers.is_empty() {
            unsafe { self.list.as_ref().unwrap().ResourceBarrier(&barriers) };
        }
    }

    /// Transition the subresources a texture *view* covers.
    pub(super) fn require_view_state(
        &mut self,
        view: &super::TextureView,
        needed: D3D12_RESOURCE_STATES,
    ) {
        let total = view.total_subresources();
        let subs = view.subresources();
        self.require_subresources(view.resource_ptr, total, subs.into_iter(), needed);
    }

    /// Transition the single (plane-0) subresource a texture *piece* (copy)
    /// addresses. `total` must include planes so the tracking Vec stays the same
    /// size as the view path uses for the same texture.
    pub(super) fn require_piece_state(
        &mut self,
        piece: &crate::TexturePiece,
        needed: D3D12_RESOURCE_STATES,
    ) {
        let tex = &piece.texture;
        let sub = piece.mip_level + piece.array_layer * tex.mip_levels;
        let total = tex.mip_levels * tex.array_layers * super::plane_count(tex.format);
        let vtbl = unsafe { tex.resource().as_raw() as super::ResourcePtr };
        self.require_subresources(vtbl, total, std::iter::once(sub), needed);
    }

    /// Transition a buffer (whole resource). Host-visible (UPLOAD/Shared) buffers
    /// are permanently locked in GENERIC_READ and must never be transitioned;
    /// GENERIC_READ already permits vertex/index/constant/SRV/copy-source reads.
    pub(super) fn require_buffer_state(
        &mut self,
        buffer: &super::Buffer,
        needed: D3D12_RESOURCE_STATES,
    ) {
        if !buffer.mapped_ptr.is_null() || buffer.resource_ptr.is_null() {
            return;
        }
        let vtbl = unsafe { buffer.resource().as_raw() as super::ResourcePtr };
        let key = vtbl as usize;
        let cur = self
            .buffer_states
            .get(&key)
            .copied()
            .unwrap_or(D3D12_RESOURCE_STATE_COMMON);
        if cur.0 != needed.0 {
            let barrier =
                transition_barrier(vtbl, cur, needed, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES);
            unsafe { self.list.as_ref().unwrap().ResourceBarrier(&[barrier]) };
            self.buffer_states.insert(key, needed);
        }
    }

    pub(super) fn current_list(&self) -> &ID3D12GraphicsCommandList {
        self.list.as_ref().unwrap()
    }

    /// Record a vertex-buffer bind for this slot (replacing any prior one), to be
    /// applied with the pipeline's stride. See `pending_vertex`.
    pub(super) fn record_vertex(&mut self, slot: u32, buf: crate::BufferPiece) {
        match self.pending_vertex.iter_mut().find(|(s, _)| *s == slot) {
            Some(entry) => entry.1 = buf,
            None => self.pending_vertex.push((slot, buf)),
        }
    }

    /// Issue IASetVertexBuffers for one slot with an explicit stride.
    pub(super) fn set_vertex_buffer(&self, slot: u32, buf: &crate::BufferPiece, stride: u32) {
        let vbv = D3D12_VERTEX_BUFFER_VIEW {
            BufferLocation: buf.buffer.gpu_address + buf.offset,
            SizeInBytes: (buf.buffer.size - buf.offset) as u32,
            StrideInBytes: stride,
        };
        unsafe { self.current_list().IASetVertexBuffers(slot, Some(&[vbv])) };
    }

    /// (Re)apply all recorded vertex binds with the given pipeline's strides.
    pub(super) fn apply_pending_vertex(&mut self, vertex_strides: &[u32]) {
        let binds = std::mem::take(&mut self.pending_vertex);
        for (slot, buf) in &binds {
            let stride = vertex_strides.get(*slot as usize).copied().unwrap_or(0);
            self.set_vertex_buffer(*slot, buf, stride);
        }
        self.pending_vertex = binds;
    }

    /// Global UAV barrier — the DX12 analog of Vulkan's
    /// `MEMORY_WRITE -> MEMORY_READ|MEMORY_WRITE` over `ALL_COMMANDS`.
    ///
    /// Per-resource *state* hazards (UAV<->SRV, render-target<->sample, write
    /// then indirect-read, etc.) are resolved by `require_state` at each use,
    /// which also carries the necessary write-visibility for that resource.
    /// What remains is same-state UAV->UAV write-after-write / read-after-write
    /// (e.g. one compute pass writing a storage buffer, the next reading it as a
    /// UAV): a global UAV barrier covers exactly that, for all resources at once.
    pub fn barrier(&mut self) {
        profiling::function_scope!();
        let uav_barrier = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                UAV: mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                    pResource: mem::ManuallyDrop::new(None),
                }),
            },
        };
        unsafe { self.list.as_ref().unwrap().ResourceBarrier(&[uav_barrier]) };
    }

    /// Matches Vulkan's `barrier_modifies_indirect` — a focused barrier
    /// when indirect buffers were written in the previous compute pass.
    pub fn barrier_modifies_indirect(&mut self) {
        self.barrier();
    }

    /// Transition every tracked resource back to COMMON and clear the state map.
    /// Called at submit so that the next command list (which assumes everything
    /// is COMMON) starts from a correct baseline — explicitly-transitioned
    /// resources do not decay at the ExecuteCommandLists boundary.
    pub(super) fn flush_to_common(&mut self) {
        let mut barriers: Vec<D3D12_RESOURCE_BARRIER> = Vec::new();
        for (&key, &state) in self.buffer_states.iter() {
            if state.0 != D3D12_RESOURCE_STATE_COMMON.0 {
                barriers.push(transition_barrier(
                    key as super::ResourcePtr,
                    state,
                    D3D12_RESOURCE_STATE_COMMON,
                    D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                ));
            }
        }
        for (&key, states) in self.texture_states.iter() {
            for (sub, &state) in states.iter().enumerate() {
                if state.0 != D3D12_RESOURCE_STATE_COMMON.0 {
                    barriers.push(transition_barrier(
                        key as super::ResourcePtr,
                        state,
                        D3D12_RESOURCE_STATE_COMMON,
                        sub as u32,
                    ));
                }
            }
        }
        if !barriers.is_empty() {
            unsafe { self.list.as_ref().unwrap().ResourceBarrier(&barriers) };
        }
        self.buffer_states.clear();
        self.texture_states.clear();
    }

    /// Called at the start of every pass. Inserts the automatic barrier
    /// when `auto_barriers` is enabled (matching Vulkan's `begin_pass`).
    fn begin_pass(&mut self, _label: &str) {
        if self.auto_barriers {
            self.barrier();
        }
    }

    pub fn set_auto_barriers(&mut self, auto_barriers: bool) {
        self.auto_barriers = auto_barriers;
    }

    pub fn transfer(&mut self, label: &str) -> super::TransferCommandEncoder<'_> {
        self.begin_pass(label);
        super::TransferCommandEncoder { encoder: self }
    }

    pub fn acceleration_structure(
        &mut self,
        label: &str,
    ) -> super::AccelerationStructureCommandEncoder<'_> {
        self.begin_pass(label);
        super::AccelerationStructureCommandEncoder {
            list: self.list.as_ref().unwrap(),
        }
    }

    pub fn compute(&mut self, label: &str) -> super::ComputeCommandEncoder<'_> {
        self.begin_pass(label);
        super::ComputeCommandEncoder { encoder: self }
    }

    pub fn render(
        &mut self,
        label: &str,
        targets: crate::RenderTargetSet,
    ) -> super::RenderCommandEncoder<'_> {
        self.begin_pass(label);
        self.pending_vertex.clear();

        let mut target_size = [0u16; 2];
        let mut rtv_handles: Vec<D3D12_CPU_DESCRIPTOR_HANDLE> = Vec::new();
        let mut dsv_handle: Option<D3D12_CPU_DESCRIPTOR_HANDLE> = None;

        // Transition all attachments first (require_state borrows the list internally),
        // then record clears / OMSetRenderTargets under a single list borrow.
        for rt in targets.colors {
            target_size = rt.view.target_size;
            // -> RENDER_TARGET (not a promotable state; tracked so a later sample
            // or present transitions out of it correctly).
            self.require_view_state(&rt.view, D3D12_RESOURCE_STATE_RENDER_TARGET);
            rtv_handles.push(super::raw_cpu_handle(rt.view.rtv_dsv_handle));
        }
        if let Some(ref rt) = targets.depth_stencil {
            target_size = rt.view.target_size;
            self.require_view_state(&rt.view, D3D12_RESOURCE_STATE_DEPTH_WRITE);
            dsv_handle = Some(super::raw_cpu_handle(rt.view.rtv_dsv_handle));
        }

        let list = self.list.as_ref().unwrap();

        for (i, rt) in targets.colors.iter().enumerate() {
            if let crate::InitOp::Clear(color) = rt.init_op {
                let rgba = map_texture_color(color);
                unsafe { list.ClearRenderTargetView(rtv_handles[i], &rgba, None) };
            }
        }

        if let Some(ref rt) = targets.depth_stencil {
            let dsv = dsv_handle.unwrap();
            let mut clear_flags = D3D12_CLEAR_FLAGS(0);
            let mut depth_v = 0.0f32;
            let mut stencil_v = 0u8;
            if let crate::InitOp::Clear(v) = rt.depth_init_op {
                clear_flags |= D3D12_CLEAR_FLAG_DEPTH;
                depth_v = v;
            }
            if let crate::InitOp::Clear(v) = rt.stencil_init_op {
                clear_flags |= D3D12_CLEAR_FLAG_STENCIL;
                stencil_v = v as u8;
            }
            if clear_flags.0 != 0 {
                unsafe { list.ClearDepthStencilView(dsv, clear_flags, depth_v, stencil_v, &[]) };
            }
        }

        unsafe {
            list.OMSetRenderTargets(
                rtv_handles.len() as u32,
                if rtv_handles.is_empty() { None } else { Some(rtv_handles.as_ptr()) },
                false,
                dsv_handle.as_ref().map(|h| h as *const _),
            );
            list.RSSetViewports(&[D3D12_VIEWPORT {
                TopLeftX: 0.0, TopLeftY: 0.0,
                Width: target_size[0] as f32, Height: target_size[1] as f32,
                MinDepth: 0.0, MaxDepth: 1.0,
            }]);
            list.RSSetScissorRects(&[RECT {
                left: 0, top: 0,
                right: target_size[0] as i32,
                bottom: target_size[1] as i32,
            }]);
        }

        super::RenderCommandEncoder { encoder: self, target_size }
    }
}

// ── CommandEncoder trait ──────────────────────────────────────────────────────

#[hidden_trait::expose]
impl crate::traits::CommandEncoder for super::CommandEncoder {
    type Texture = super::Texture;
    type Frame = super::Frame;

    fn start(&mut self) {
        profiling::function_scope!();
        self.allocators.rotate_left(1);
        self.allocator_fences.rotate_left(1);
        // The allocator now at the front may still be in use by the GPU from its
        // previous submission (it was last used `buffer_count` submits ago).
        // Resetting an allocator while the GPU is still reading the command list
        // recorded from it is illegal — the debug layer reports "command allocator
        // was reset after the command list was recorded" and the device is removed
        // (which then cascades into bogus barrier-state and closed-list errors). So
        // CPU-wait for that submission's fence value before resetting.
        let target = self.allocator_fences[0];
        if target != 0 && unsafe { self.fence.GetCompletedValue() } < target {
            unsafe {
                self.fence
                    .SetEventOnCompletion(target, self.fence_event)
                    .unwrap();
                let _ = windows::Win32::System::Threading::WaitForSingleObjectEx(
                    self.fence_event,
                    windows::Win32::System::Threading::INFINITE,
                    false,
                );
            }
        }
        let allocator = &self.allocators[0];
        unsafe { allocator.Reset().unwrap() };
        unsafe { self.list.as_ref().unwrap().Reset(allocator, None).unwrap() };

        self.cbv_srv_uav_ring.reset();
        self.sampler_ring.reset();
        self.upload_ring.reset();
        // All resources decay to COMMON at the ExecuteCommandLists boundary, so a
        // fresh command list starts with every resource implicitly in COMMON.
        self.buffer_states.clear();
        self.texture_states.clear();

        let heaps: [Option<ID3D12DescriptorHeap>; 2] = [
            Some(self.cbv_srv_uav_ring.heap.clone()),
            Some(self.sampler_ring.heap.clone()),
        ];
        unsafe { self.list.as_ref().unwrap().SetDescriptorHeaps(&heaps) };

        self.timings.clear();
    }

    fn init_texture(&mut self, _texture: super::Texture) {
        // Resources start in D3D12_RESOURCE_STATE_COMMON; no transition needed.
    }

    fn present(&mut self, frame: super::Frame) {
        profiling::function_scope!();

        // The presentable state is COMMON (== D3D12_RESOURCE_STATE_PRESENT == 0).
        // The back buffer is a single-subresource texture (1 mip, 1 layer).
        // require_subresources is a no-op if it's already COMMON, so it never
        // emits an illegal before==after barrier.
        let frame_vtbl = unsafe { frame.resource.as_raw() as super::ResourcePtr };
        self.require_subresources(frame_vtbl, 1, std::iter::once(0), D3D12_RESOURCE_STATE_COMMON);

        self.present = Some(super::Presentation {
            swapchain: frame.swapchain,
            buffer_index: frame.buffer_index,
            present_id: frame.present_id,
            display_sync: frame.display_sync,
        });
    }

    fn timings(&self) -> &crate::Timings {
        &self.timings
    }
}

// No Drop impls on the sub-encoders: render targets are left in
// RENDER_TARGET/DEPTH_WRITE state and lazily transitioned back to COMMON by
// the next `barrier()` (auto-inserted at the next pass / at submit), or
// directly to PRESENT by `present()`. This mirrors Vulkan, where the pipeline
// barrier between passes — not the pass teardown — is what re-synchronizes,
// and it keeps the back buffer at the minimal 2 transitions per frame.

// ── TransferEncoder ───────────────────────────────────────────────────────────

#[hidden_trait::expose]
impl crate::traits::TransferEncoder for super::TransferCommandEncoder<'_> {
    type BufferPiece = crate::BufferPiece;
    type TexturePiece = crate::TexturePiece;

    fn fill_buffer(&mut self, dst: crate::BufferPiece, _size: u64, value: u8) {
        // DX12 ClearUnorderedAccessViewUint fills in DWORD chunks with a repeated-byte
        // pattern. Note: it clears the whole UAV (the buffer's full-range raw UAV), so
        // `_size`/offset sub-ranges are not honored — exact for whole-buffer fills.
        let value_u32 = (value as u32) * 0x01010101;

        // The buffer must be in UNORDERED_ACCESS to be cleared through a UAV.
        self.encoder
            .require_buffer_state(&dst.buffer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        let enc = &mut *self.encoder;

        let (ring_cpu, ring_gpu) = enc.cbv_srv_uav_ring.alloc(1);
        let uav_cpu = super::raw_cpu_handle(dst.buffer.uav_handle);

        unsafe {
            enc.device.CopyDescriptorsSimple(
                1, ring_cpu, uav_cpu, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            );

            // ClearUnorderedAccessViewUint fills the entire resource; we'd need
            // a per-range UAV to fill an offset+size range precisely.
            // For offset=0 / full-buffer fills this is exact.  For other cases
            // it fills the entire buffer (conservatively safe).
            enc.current_list().ClearUnorderedAccessViewUint(
                ring_gpu,
                uav_cpu,
                dst.buffer.resource(),
                &[value_u32, value_u32, value_u32, value_u32],
                &[], // no scissor rects → clear whole resource
            );
        }
    }

    fn copy_buffer_to_buffer(
        &mut self,
        src: crate::BufferPiece,
        dst: crate::BufferPiece,
        size: u64,
    ) {
        self.encoder
            .require_buffer_state(&src.buffer, D3D12_RESOURCE_STATE_COPY_SOURCE);
        self.encoder
            .require_buffer_state(&dst.buffer, D3D12_RESOURCE_STATE_COPY_DEST);
        unsafe {
            self.encoder.current_list().CopyBufferRegion(
                dst.buffer.resource(), dst.offset,
                src.buffer.resource(), src.offset,
                size,
            );
        }
    }

    fn copy_texture_to_texture(
        &mut self,
        src: crate::TexturePiece,
        dst: crate::TexturePiece,
        size: crate::Extent,
    ) {
        let src_res = src.texture.resource();
        let dst_res = dst.texture.resource();
        self.encoder.require_piece_state(&src, D3D12_RESOURCE_STATE_COPY_SOURCE);
        self.encoder.require_piece_state(&dst, D3D12_RESOURCE_STATE_COPY_DEST);
        let src_loc = D3D12_TEXTURE_COPY_LOCATION {
            pResource: mem::ManuallyDrop::new(unsafe { Some(ID3D12Resource::from_raw(src_res.as_raw())) }),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: subresource_index(src.mip_level, src.array_layer, src.texture.mip_levels),
            },
        };
        let dst_loc = D3D12_TEXTURE_COPY_LOCATION {
            pResource: mem::ManuallyDrop::new(unsafe { Some(ID3D12Resource::from_raw(dst_res.as_raw())) }),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: subresource_index(dst.mip_level, dst.array_layer, dst.texture.mip_levels),
            },
        };
        let src_box = D3D12_BOX {
            left: src.origin[0], top: src.origin[1], front: src.origin[2],
            right: src.origin[0] + size.width,
            bottom: src.origin[1] + size.height,
            back: src.origin[2] + size.depth,
        };
        unsafe {
            self.encoder.current_list().CopyTextureRegion(
                &dst_loc, dst.origin[0], dst.origin[1], dst.origin[2], &src_loc, Some(&src_box),
            );
        }
    }

    fn copy_buffer_to_texture(
        &mut self,
        src: crate::BufferPiece,
        bytes_per_row: u32,
        dst: crate::TexturePiece,
        size: crate::Extent,
    ) {
        self.encoder.require_piece_state(&dst, D3D12_RESOURCE_STATE_COPY_DEST);

        // D3D12 placed footprints require the source offset to be a multiple of
        // 512 (D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT) and the row pitch a multiple
        // of 256 (D3D12_TEXTURE_DATA_PITCH_ALIGNMENT). blade buffers are packed
        // tightly at `bytes_per_row` with arbitrary offsets, so when either is
        // unaligned we repack the rows into the upload ring at a legal layout.
        // Wide textures (pitch already 256-aligned) and aligned offsets take the
        // direct path with no copy.
        let aligned_pitch = align_pitch(bytes_per_row as u64);
        let direct = src.offset % D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT as u64 == 0
            && bytes_per_row as u64 == aligned_pitch;

        let (footprint_res, footprint_offset) = if direct {
            self.encoder
                .require_buffer_state(&src.buffer, D3D12_RESOURCE_STATE_COPY_SOURCE);
            (
                unsafe { src.buffer.resource().as_raw() },
                src.offset,
            )
        } else {
            assert!(
                !src.buffer.mapped_ptr.is_null(),
                "DX12: copy_buffer_to_texture with an unaligned offset/row-pitch \
                 requires a host-visible (Upload/Shared) source buffer"
            );
            let num_rows = (size.height * size.depth) as u64;
            let staging_off = self.encoder.upload_ring.alloc(
                aligned_pitch * num_rows,
                D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT as u64,
            );
            unsafe {
                let dst_base = self.encoder.upload_ring.mapped.add(staging_off as usize);
                let src_base = src.buffer.mapped_ptr.add(src.offset as usize);
                for r in 0..num_rows {
                    std::ptr::copy_nonoverlapping(
                        src_base.add((r * bytes_per_row as u64) as usize),
                        dst_base.add((r * aligned_pitch) as usize),
                        bytes_per_row as usize,
                    );
                }
                (self.encoder.upload_ring.resource.as_raw(), staging_off)
            }
        };

        let src_loc = D3D12_TEXTURE_COPY_LOCATION {
            pResource: mem::ManuallyDrop::new(unsafe { Some(ID3D12Resource::from_raw(footprint_res)) }),
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                    Offset: footprint_offset,
                    Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                        Format: super::map_texture_format(dst.texture.format),
                        Width: size.width, Height: size.height, Depth: size.depth,
                        RowPitch: aligned_pitch as u32,
                    },
                },
            },
        };
        let dst_loc = D3D12_TEXTURE_COPY_LOCATION {
            pResource: mem::ManuallyDrop::new(unsafe { Some(ID3D12Resource::from_raw(dst.texture.resource().as_raw())) }),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: subresource_index(dst.mip_level, dst.array_layer, dst.texture.mip_levels),
            },
        };
        unsafe {
            self.encoder.current_list().CopyTextureRegion(
                &dst_loc, dst.origin[0], dst.origin[1], dst.origin[2], &src_loc, None,
            );
        }
    }

    fn copy_texture_to_buffer(
        &mut self,
        src: crate::TexturePiece,
        dst: crate::BufferPiece,
        bytes_per_row: u32,
        size: crate::Extent,
    ) {
        let src_res = src.texture.resource();
        self.encoder.require_piece_state(&src, D3D12_RESOURCE_STATE_COPY_SOURCE);
        self.encoder
            .require_buffer_state(&dst.buffer, D3D12_RESOURCE_STATE_COPY_DEST);
        let aligned_pitch = align_pitch(bytes_per_row as u64);
        let src_loc = D3D12_TEXTURE_COPY_LOCATION {
            pResource: mem::ManuallyDrop::new(unsafe { Some(ID3D12Resource::from_raw(src_res.as_raw())) }),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: subresource_index(src.mip_level, src.array_layer, src.texture.mip_levels),
            },
        };
        let dst_loc = D3D12_TEXTURE_COPY_LOCATION {
            pResource: mem::ManuallyDrop::new(unsafe { Some(ID3D12Resource::from_raw(dst.buffer.resource().as_raw())) }),
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                    Offset: dst.offset,
                    Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                        Format: super::map_texture_format(src.texture.format),
                        Width: size.width, Height: size.height, Depth: size.depth,
                        RowPitch: aligned_pitch as u32,
                    },
                },
            },
        };
        let src_box = D3D12_BOX {
            left: src.origin[0], top: src.origin[1], front: src.origin[2],
            right: src.origin[0] + size.width,
            bottom: src.origin[1] + size.height,
            back: src.origin[2] + size.depth,
        };
        unsafe {
            self.encoder.current_list().CopyTextureRegion(
                &dst_loc, 0, 0, 0, &src_loc, Some(&src_box),
            );
        }
    }
}

fn align_pitch(pitch: u64) -> u64 {
    (pitch + D3D12_TEXTURE_DATA_PITCH_ALIGNMENT as u64 - 1)
        & !(D3D12_TEXTURE_DATA_PITCH_ALIGNMENT as u64 - 1)
}

// ── AccelerationStructureEncoder ─────────────────────────────────────────────

#[hidden_trait::expose]
impl crate::traits::AccelerationStructureEncoder
    for super::AccelerationStructureCommandEncoder<'_>
{
    type AccelerationStructure = super::AccelerationStructure;
    type AccelerationStructureMesh = crate::AccelerationStructureMesh;
    type BufferPiece = crate::BufferPiece;

    fn build_bottom_level(
        &mut self,
        _acceleration_structure: super::AccelerationStructure,
        _meshes: &[crate::AccelerationStructureMesh],
        _scratch_data: crate::BufferPiece,
    ) {
        unimplemented!("DX12 acceleration structures not yet implemented")
    }
    fn build_top_level(
        &mut self,
        _acceleration_structure: super::AccelerationStructure,
        _bottom_level: &[super::AccelerationStructure],
        _instance_count: u32,
        _instance_data: crate::BufferPiece,
        _scratch_data: crate::BufferPiece,
    ) {
        unimplemented!("DX12 acceleration structures not yet implemented")
    }
}

// ── ComputeCommandEncoder ─────────────────────────────────────────────────────

impl super::ComputeCommandEncoder<'_> {
    /// Insert a memory barrier within a compute pass (mirrors Metal/Vulkan).
    pub fn barrier(&mut self) {
        let uav_barrier = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                UAV: mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                    pResource: mem::ManuallyDrop::new(None),
                }),
            },
        };
        unsafe { self.encoder.current_list().ResourceBarrier(&[uav_barrier]) };
    }

    pub fn with<'p>(&'p mut self, pipeline: &'p super::ComputePipeline)
        -> super::ComputePipelineContext<'p>
    {
        let sampler_gpu = self.encoder.sampler_ring.gpu_start;
        let list = self.encoder.list.as_ref().unwrap();
        unsafe {
            list.SetPipelineState(&pipeline.pso);
            list.SetComputeRootSignature(&pipeline.layout.root_signature);
                if let Some((std_root, cmp_root)) = pipeline.layout.sampler_heap_roots {
                list.SetComputeRootDescriptorTable(std_root, sampler_gpu);
                list.SetComputeRootDescriptorTable(cmp_root, sampler_gpu);
            }
        }
        super::ComputePipelineContext { encoder: self.encoder, layout: &pipeline.layout }
    }
}

// ── RenderCommandEncoder ──────────────────────────────────────────────────────

impl super::RenderCommandEncoder<'_> {
    pub fn with<'p>(&'p mut self, pipeline: &'p super::RenderPipeline)
        -> super::RenderPipelineContext<'p>
    {
        let sampler_gpu = self.encoder.sampler_ring.gpu_start;
        let list = self.encoder.list.as_ref().unwrap();
        unsafe {
            list.SetPipelineState(&pipeline.pso);
            list.SetGraphicsRootSignature(&pipeline.layout.root_signature);
            list.IASetPrimitiveTopology(pipeline.topology);
            if let Some((std_root, cmp_root)) = pipeline.layout.sampler_heap_roots {
                list.SetGraphicsRootDescriptorTable(std_root, sampler_gpu);
                list.SetGraphicsRootDescriptorTable(cmp_root, sampler_gpu);
            }
        }
        // Apply vertex buffers bound on the pass (their stride is known only now).
        self.encoder.apply_pending_vertex(&pipeline.vertex_strides);
        super::RenderPipelineContext {
            encoder: self.encoder,
            layout: &pipeline.layout,
            vertex_strides: &pipeline.vertex_strides,
        }
    }
}

// `RenderEncoder` on the pass itself (matches Vulkan/Metal): scissor/viewport/
// stencil are command-list state set directly; vertex binds are deferred until a
// pipeline supplies the stride (see `with`).
#[hidden_trait::expose]
impl crate::traits::RenderEncoder for super::RenderCommandEncoder<'_> {
    type BufferPiece = crate::BufferPiece;

    fn set_scissor_rect(&mut self, rect: &crate::ScissorRect) {
        let r = RECT { left: rect.x, top: rect.y, right: rect.x + rect.w as i32, bottom: rect.y + rect.h as i32 };
        unsafe { self.encoder.current_list().RSSetScissorRects(&[r]) };
    }

    fn set_viewport(&mut self, viewport: &crate::Viewport) {
        let vp = D3D12_VIEWPORT {
            TopLeftX: viewport.x, TopLeftY: viewport.y,
            Width: viewport.w, Height: viewport.h,
            MinDepth: viewport.depth.start,
            MaxDepth: viewport.depth.end,
        };
        unsafe { self.encoder.current_list().RSSetViewports(&[vp]) };
    }

    fn set_stencil_reference(&mut self, reference: u32) {
        unsafe { self.encoder.current_list().OMSetStencilRef(reference) };
    }

    fn bind_vertex(&mut self, index: u32, vertex_buf: crate::BufferPiece) {
        self.encoder
            .require_buffer_state(&vertex_buf.buffer, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
        // Deferred — applied with the pipeline's stride at `with()`.
        self.encoder.record_vertex(index, vertex_buf);
    }
}

// ── PipelineEncoder (bind) ────────────────────────────────────────────────────

fn bind_group<D: crate::ShaderData>(
    encoder: &mut super::CommandEncoder,
    layout: &super::PipelineLayout,
    group: u32,
    data: &D,
    is_compute: bool,
) {
    let gd = &layout.groups[group as usize];
    if gd.slots.is_empty() { return; }

    let (ring_cpu_base, ring_gpu_base) = if gd.cbv_srv_uav_count > 0 {
        encoder.cbv_srv_uav_ring.alloc(gd.cbv_srv_uav_count)
    } else {
        (D3D12_CPU_DESCRIPTOR_HANDLE { ptr: 0 }, D3D12_GPU_DESCRIPTOR_HANDLE { ptr: 0 })
    };
    // Reserve a contiguous run in the sampler ring for this group's samplers.
    // `sampler_base_index` is the absolute heap index the index buffer references.
    let (sampler_cpu_base, sampler_base_index) = if gd.sampler_count > 0 {
        let base_index = encoder.sampler_ring.offset;
        let (cpu, _gpu) = encoder.sampler_ring.alloc(gd.sampler_count);
        (cpu, base_index)
    } else {
        (D3D12_CPU_DESCRIPTOR_HANDLE { ptr: 0 }, 0)
    };

    let root_cbv = gd.cbv_srv_uav_root_index;
    let root_sampler_index = gd.sampler_index_root_index;

    // `fill` consumes the PipelineContext (matching ShaderData::fill's by-value
    // signature) and copies each sampler into ring slot (base + register).
    data.fill(super::PipelineContext {
        encoder: &mut *encoder,
        group_descriptors: gd,
        ring_cpu_base,
        sampler_cpu_base,
    });

    // Build and bind this group's sampler-index buffer: [base, base+1, ...].
    if let Some(idx) = root_sampler_index {
        let indices: Vec<u32> = (0..gd.sampler_count)
            .map(|r| sampler_base_index + r)
            .collect();
        let va = encoder
            .upload_ring
            .write_aligned(bytemuck::cast_slice(&indices), 16);
        let list = encoder.list.as_ref().unwrap();
        if is_compute { unsafe { list.SetComputeRootShaderResourceView(idx, va) }; }
        else          { unsafe { list.SetGraphicsRootShaderResourceView(idx, va) }; }
    }

    let list = encoder.list.as_ref().unwrap();
    if let Some(idx) = root_cbv {
        if ring_gpu_base.ptr != 0 {
            if is_compute { unsafe { list.SetComputeRootDescriptorTable(idx, ring_gpu_base) }; }
            else          { unsafe { list.SetGraphicsRootDescriptorTable(idx, ring_gpu_base) }; }
        }
    }
}

#[hidden_trait::expose]
impl crate::traits::PipelineEncoder for super::ComputePipelineContext<'_> {
    fn bind<D: crate::ShaderData>(&mut self, group: u32, data: &D) {
        let layout = self.layout;
        bind_group(self.encoder, layout, group, data, true);
    }
}

#[hidden_trait::expose]
impl crate::traits::PipelineEncoder for super::RenderPipelineContext<'_> {
    fn bind<D: crate::ShaderData>(&mut self, group: u32, data: &D) {
        let layout = self.layout;
        bind_group(self.encoder, layout, group, data, false);
    }
}

// ── ComputePipelineEncoder ────────────────────────────────────────────────────

#[hidden_trait::expose]
impl crate::traits::ComputePipelineEncoder for super::ComputePipelineContext<'_> {
    type BufferPiece = crate::BufferPiece;

    fn dispatch(&mut self, groups: [u32; 3]) {
        unsafe {
            self.encoder.current_list().Dispatch(groups[0], groups[1], groups[2]);
        }
    }

    fn dispatch_indirect(&mut self, indirect_buf: crate::BufferPiece) {
        self.encoder
            .require_buffer_state(&indirect_buf.buffer, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
        let sig = self.encoder.dispatch_sig.clone();
        unsafe {
            self.encoder.current_list().ExecuteIndirect(
                &sig, 1, indirect_buf.buffer.resource(), indirect_buf.offset, None, 0,
            );
        }
    }
}

// ── RenderEncoder ─────────────────────────────────────────────────────────────

#[hidden_trait::expose]
impl crate::traits::RenderEncoder for super::RenderPipelineContext<'_> {
    type BufferPiece = crate::BufferPiece;

    fn set_scissor_rect(&mut self, rect: &crate::ScissorRect) {
        let r = RECT { left: rect.x, top: rect.y, right: rect.x + rect.w as i32, bottom: rect.y + rect.h as i32 };
        unsafe { self.encoder.current_list().RSSetScissorRects(&[r]) };
    }

    fn set_viewport(&mut self, viewport: &crate::Viewport) {
        let vp = D3D12_VIEWPORT {
            TopLeftX: viewport.x, TopLeftY: viewport.y,
            Width: viewport.w, Height: viewport.h,
            MinDepth: viewport.depth.start,
            MaxDepth: viewport.depth.end,
        };
        unsafe { self.encoder.current_list().RSSetViewports(&[vp]) };
    }

    fn set_stencil_reference(&mut self, reference: u32) {
        unsafe { self.encoder.current_list().OMSetStencilRef(reference) };
    }

    fn bind_vertex(&mut self, index: u32, vertex_buf: crate::BufferPiece) {
        // A vertex buffer produced by a compute pass (e.g. post-culling instances)
        // is in UNORDERED_ACCESS; bring it to the vertex-buffer read state.
        self.encoder
            .require_buffer_state(&vertex_buf.buffer, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
        // Apply now with this pipeline's stride, and record it so a subsequent
        // `with()` (next pipeline in the pass) re-applies it.
        let stride = self.vertex_strides.get(index as usize).copied().unwrap_or(0);
        self.encoder.set_vertex_buffer(index, &vertex_buf, stride);
        self.encoder.record_vertex(index, vertex_buf);
    }
}

// ── RenderPipelineEncoder ─────────────────────────────────────────────────────

#[hidden_trait::expose]
impl crate::traits::RenderPipelineEncoder for super::RenderPipelineContext<'_> {
    fn draw(&mut self, first_vertex: u32, vertex_count: u32, first_instance: u32, instance_count: u32) {
        unsafe {
            self.encoder.current_list().DrawInstanced(vertex_count, instance_count, first_vertex, first_instance);
        }
    }

    fn draw_indexed(
        &mut self,
        index_buf: crate::BufferPiece,
        index_type: crate::IndexType,
        index_count: u32,
        base_vertex: i32,
        start_instance: u32,
        instance_count: u32,
    ) {
        self.encoder
            .require_buffer_state(&index_buf.buffer, D3D12_RESOURCE_STATE_INDEX_BUFFER);
        let ibv = D3D12_INDEX_BUFFER_VIEW {
            BufferLocation: index_buf.buffer.gpu_address + index_buf.offset,
            SizeInBytes: (index_buf.buffer.size - index_buf.offset) as u32,
            Format: super::map_index_format(index_type),
        };
        unsafe {
            let list = self.encoder.current_list();
            list.IASetIndexBuffer(Some(&ibv));
            list.DrawIndexedInstanced(index_count, instance_count, 0, base_vertex, start_instance);
        }
    }

    fn draw_indirect(&mut self, indirect_buf: crate::BufferPiece, draw_count: u32) {
        // Compute-produced draw args are in UNORDERED_ACCESS; move to INDIRECT_ARGUMENT.
        self.encoder
            .require_buffer_state(&indirect_buf.buffer, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
        let sig = self.encoder.draw_sig.clone();
        unsafe {
            self.encoder.current_list().ExecuteIndirect(
                &sig, draw_count, indirect_buf.buffer.resource(), indirect_buf.offset, None, 0,
            );
        }
    }

    fn draw_indexed_indirect(
        &mut self,
        index_buf: crate::BufferPiece,
        index_type: crate::IndexType,
        indirect_buf: crate::BufferPiece,
        draw_count: u32,
    ) {
        self.encoder
            .require_buffer_state(&index_buf.buffer, D3D12_RESOURCE_STATE_INDEX_BUFFER);
        self.encoder
            .require_buffer_state(&indirect_buf.buffer, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
        let ibv = D3D12_INDEX_BUFFER_VIEW {
            BufferLocation: index_buf.buffer.gpu_address + index_buf.offset,
            SizeInBytes: (index_buf.buffer.size - index_buf.offset) as u32,
            Format: super::map_index_format(index_type),
        };
        let sig = self.encoder.draw_indexed_sig.clone();
        unsafe {
            let list = self.encoder.current_list();
            list.IASetIndexBuffer(Some(&ibv));
            list.ExecuteIndirect(
                &sig, draw_count, indirect_buf.buffer.resource(), indirect_buf.offset, None, 0,
            );
        }
    }

    fn draw_indexed_indirect_count(
        &mut self,
        index_buf: crate::BufferPiece,
        index_type: crate::IndexType,
        indirect_buf: crate::BufferPiece,
        count_buf: crate::BufferPiece,
        max_draw_count: u32,
    ) {
        self.encoder
            .require_buffer_state(&index_buf.buffer, D3D12_RESOURCE_STATE_INDEX_BUFFER);
        // Both the argument buffer and the count buffer must be INDIRECT_ARGUMENT.
        self.encoder
            .require_buffer_state(&indirect_buf.buffer, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
        self.encoder
            .require_buffer_state(&count_buf.buffer, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
        let ibv = D3D12_INDEX_BUFFER_VIEW {
            BufferLocation: index_buf.buffer.gpu_address + index_buf.offset,
            SizeInBytes: (index_buf.buffer.size - index_buf.offset) as u32,
            Format: super::map_index_format(index_type),
        };
        let sig = self.encoder.draw_indexed_sig.clone();
        unsafe {
            let list = self.encoder.current_list();
            list.IASetIndexBuffer(Some(&ibv));
            list.ExecuteIndirect(
                &sig, max_draw_count,
                indirect_buf.buffer.resource(), indirect_buf.offset,
                Some(count_buf.buffer.resource()), count_buf.offset,
            );
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn subresource_index(mip: u32, array_layer: u32, mip_count: u32) -> u32 {
    mip + array_layer * mip_count
}

fn map_texture_color(color: crate::TextureColor) -> [f32; 4] {
    match color {
        crate::TextureColor::TransparentBlack => [0.0, 0.0, 0.0, 0.0],
        crate::TextureColor::OpaqueBlack => [0.0, 0.0, 0.0, 1.0],
        crate::TextureColor::White => [1.0, 1.0, 1.0, 1.0],
        crate::TextureColor::RgbaFloat { rgba } => rgba,
    }
}
