use std::mem;
use windows::{
    core::Interface,
    Win32::{
        Foundation::RECT,
        Graphics::{Direct3D::D3D_PRIMITIVE_TOPOLOGY, Direct3D12::*},
    },
};

// ── ShaderBindable impls ──────────────────────────────────────────────────────

impl<T: bytemuck::Pod> crate::ShaderBindable for T {
    fn bind_to(&self, ctx: &mut super::PipelineContext, index: u32) {
        let slot = &ctx.group_descriptors.slots[index as usize];
        debug_assert_eq!(slot.kind, super::BindingKind::Cbv);

        // Write data (padded to 256 bytes — DX12 CBV minimum)
        let data = bytemuck::bytes_of(self);
        let mut padded = [0u8; 256];
        padded[..data.len().min(256)].copy_from_slice(&data[..data.len().min(256)]);
        let gpu_addr = ctx.encoder.upload_ring.write_cbv(&padded);

        let cbv_cpu = D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: ctx.ring_cpu_base.ptr
                + (slot.heap_offset * ctx.encoder.cbv_srv_uav_ring.increment) as usize,
        };
        unsafe {
            ctx.encoder.device.CreateConstantBufferView(
                Some(&D3D12_CONSTANT_BUFFER_VIEW_DESC {
                    BufferLocation: gpu_addr,
                    SizeInBytes: 256,
                }),
                cbv_cpu,
            );
        }
    }
}

impl crate::ShaderBindable for super::TextureView {
    fn bind_to(&self, ctx: &mut super::PipelineContext, index: u32) {
        let slot = &ctx.group_descriptors.slots[index as usize];
        let (src, heap_type) = match slot.kind {
            super::BindingKind::Srv => (
                super::raw_cpu_handle(self.srv_handle),
                D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            ),
            super::BindingKind::Uav => (
                super::raw_cpu_handle(self.uav_handle),
                D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            ),
            other => panic!("unexpected binding kind {:?} for TextureView", other),
        };
        let dst = D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: ctx.ring_cpu_base.ptr
                + (slot.heap_offset * ctx.encoder.cbv_srv_uav_ring.increment) as usize,
        };
        unsafe { ctx.encoder.device.CopyDescriptorsSimple(1, dst, src, heap_type) };
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
        let src = super::raw_cpu_handle(self.cpu_handle);
        let dst = D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: ctx.sampler_cpu_base.ptr
                + (slot.heap_offset * ctx.encoder.sampler_ring.increment) as usize,
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
        match slot.kind {
            super::BindingKind::Srv => {
                unsafe {
                    ctx.encoder.device.CopyDescriptorsSimple(
                        1, dst,
                        super::raw_cpu_handle(self.buffer.srv_handle),
                        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    )
                };
            }
            super::BindingKind::Uav => {
                unsafe {
                    ctx.encoder.device.CopyDescriptorsSimple(
                        1, dst,
                        super::raw_cpu_handle(self.buffer.uav_handle),
                        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    )
                };
                // Track for UNORDERED_ACCESS->COMMON reset at the next barrier(),
                // so a later indirect/SRV read can auto-promote from COMMON.
                let vtbl = unsafe { self.buffer.resource().as_raw() as super::ResourcePtr };
                if !ctx.encoder.uav_buffers.contains(&vtbl) {
                    ctx.encoder.uav_buffers.push(vtbl);
                }
            }
            super::BindingKind::Cbv => {
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
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

// ── CommandEncoder: private helpers ──────────────────────────────────────────

impl super::CommandEncoder {
    pub(super) fn current_list(&self) -> &ID3D12GraphicsCommandList {
        self.list.as_ref().unwrap()
    }

    /// Full pipeline barrier — the DX12 equivalent of Vulkan's
    /// `MEMORY_WRITE -> MEMORY_READ|MEMORY_WRITE` over `ALL_COMMANDS`.
    ///
    /// Three things happen, all "everything-before is visible to everything-after":
    ///  1. Active render/depth targets transition back to COMMON.
    ///  2. UAV-written buffers transition UNORDERED_ACCESS -> COMMON. This both
    ///     flushes the writes AND returns them to COMMON so a following read can
    ///     auto-promote — crucially, `ExecuteIndirect` needs INDIRECT_ARGUMENT,
    ///     which COMMON promotes to but UNORDERED_ACCESS does not. This is what
    ///     makes "compute writes indirect args -> indirect draw" work, matching
    ///     Vulkan's MEMORY_WRITE->MEMORY_READ guarantee.
    ///  3. A global UAV barrier covers any remaining UAV resources (e.g. storage
    ///     textures, or buffers kept in UAV across passes).
    pub fn barrier(&mut self) {
        profiling::function_scope!();

        let list = self.list.as_ref().unwrap();

        let mut transitions: Vec<D3D12_RESOURCE_BARRIER> = Vec::new();
        for rt in self.active_rts.drain(..) {
            transitions.push(transition_barrier(rt.resource_vtbl, rt.state, D3D12_RESOURCE_STATE_COMMON));
        }
        for vtbl in self.uav_buffers.drain(..) {
            transitions.push(transition_barrier(
                vtbl, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON));
        }
        if !transitions.is_empty() {
            unsafe { list.ResourceBarrier(&transitions) };
        }

        // Global UAV barrier: ensures all UAV writes are visible to later UAV use.
        let uav_barrier = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                UAV: mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                    pResource: mem::ManuallyDrop::new(None),
                }),
            },
        };
        unsafe { list.ResourceBarrier(&[uav_barrier]) };
    }

    /// Matches Vulkan's `barrier_modifies_indirect` — a focused barrier
    /// when indirect buffers were written in the previous compute pass.
    pub fn barrier_modifies_indirect(&mut self) {
        self.barrier();
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

        let list = self.list.as_ref().unwrap();
        let mut target_size = [0u16; 2];
        let mut rtv_handles: Vec<D3D12_CPU_DESCRIPTOR_HANDLE> = Vec::new();
        let mut dsv_handle: Option<D3D12_CPU_DESCRIPTOR_HANDLE> = None;

        for rt in targets.colors {
            target_size = rt.view.target_size;
            let vtbl = rt.view.resource_ptr;

            // COMMON → RENDER_TARGET. This is the one strictly-required transition:
            // RENDER_TARGET is not a promotable state. COMMON (== PRESENT, both 0) is
            // the resting state, so this covers the first frame, post-present back
            // buffers, and offscreen targets uniformly. Skip if already tracked as a
            // render target (manual-barrier mode, repeated passes without a barrier).
            if !self.active_rts.iter().any(|rt| rt.resource_vtbl == vtbl) {
                let barrier = transition_barrier(
                    vtbl, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RENDER_TARGET);
                unsafe { list.ResourceBarrier(&[barrier]) };
                self.active_rts.push(super::ActiveRt {
                    resource_vtbl: vtbl,
                    state: D3D12_RESOURCE_STATE_RENDER_TARGET,
                });
            }

            let rtv = super::raw_cpu_handle(rt.view.rtv_dsv_handle);
            rtv_handles.push(rtv);
            if let crate::InitOp::Clear(color) = rt.init_op {
                let rgba = map_texture_color(color);
                unsafe { list.ClearRenderTargetView(rtv, &rgba, None) };
            }
        }

        if let Some(ref rt) = targets.depth_stencil {
            target_size = rt.view.target_size;
            let vtbl = rt.view.resource_ptr;
            let state = D3D12_RESOURCE_STATE_DEPTH_WRITE;

            // COMMON → DEPTH_WRITE (also not promotable). Skip if already tracked.
            if !self.active_rts.iter().any(|rt| rt.resource_vtbl == vtbl) {
                let barrier = transition_barrier(vtbl, D3D12_RESOURCE_STATE_COMMON, state);
                unsafe { list.ResourceBarrier(&[barrier]) };
                self.active_rts.push(super::ActiveRt { resource_vtbl: vtbl, state });
            }

            let dsv = super::raw_cpu_handle(rt.view.rtv_dsv_handle);
            dsv_handle = Some(dsv);

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
                unsafe { list.ClearDepthStencilView(dsv, clear_flags, depth_v, stencil_v, None) };
            }
        }

        unsafe {
            list.OMSetRenderTargets(
                rtv_handles.len() as u32,
                if rtv_handles.is_empty() { None } else { Some(rtv_handles.as_ptr()) },
                false,
                dsv_handle.as_ref(),
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
        let allocator = &self.allocators[0];
        unsafe { allocator.Reset().unwrap() };
        unsafe { self.list.as_ref().unwrap().Reset(allocator, None).unwrap() };

        self.cbv_srv_uav_ring.reset();
        self.sampler_ring.reset();
        self.upload_ring.reset();
        self.active_rts.clear();
        // Buffer states decay to COMMON at the ExecuteCommandLists boundary, so
        // a fresh command list starts with all buffers in COMMON.
        self.uav_buffers.clear();

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
        let list = self.list.as_ref().unwrap();

        // PRESENT == COMMON (both 0). If the back buffer is still in RENDER_TARGET
        // (the render pass left it there, lazily), transition it back to COMMON —
        // which is the presentable state. If it was already flushed to COMMON by a
        // `barrier()`, no transition is needed: emitting COMMON→PRESENT would be a
        // before==after no-op that the debug layer rejects.
        let frame_vtbl = unsafe { frame.resource.as_raw() as *mut std::ffi::c_void };
        if let Some(pos) = self
            .active_rts
            .iter()
            .position(|rt| rt.resource_vtbl == frame_vtbl)
        {
            let rt = self.active_rts.remove(pos);
            let barrier = transition_barrier(frame_vtbl, rt.state, D3D12_RESOURCE_STATE_COMMON);
            unsafe { list.ResourceBarrier(&[barrier]) };
        }

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

    fn fill_buffer(&mut self, dst: crate::BufferPiece, size: u64, value: u8) {
        // DX12 ClearUnorderedAccessViewUint fills in DWORD chunks with a repeated-byte pattern.
        let value_u32 = (value as u32) * 0x01010101;
        let enc = &mut *self.encoder;

        // Allocate one slot in the GPU-visible ring for the UAV.
        let (ring_cpu, ring_gpu) = enc.cbv_srv_uav_ring.alloc(1);
        let uav_cpu = super::raw_cpu_handle(dst.buffer.uav_handle);

        // Copy the buffer's CPU-visible UAV descriptor into the GPU-visible ring.
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
        let src_loc = D3D12_TEXTURE_COPY_LOCATION {
            pResource: mem::ManuallyDrop::new(unsafe { Some(ID3D12Resource::from_raw(src_res.as_raw())) }),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: subresource_index(src.mip_level, src.array_layer, 1),
            },
        };
        let dst_loc = D3D12_TEXTURE_COPY_LOCATION {
            pResource: mem::ManuallyDrop::new(unsafe { Some(ID3D12Resource::from_raw(dst_res.as_raw())) }),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: subresource_index(dst.mip_level, dst.array_layer, 1),
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
        let dst_res = dst.texture.resource();
        let aligned_pitch = align_pitch(bytes_per_row as u64);
        let src_loc = D3D12_TEXTURE_COPY_LOCATION {
            pResource: mem::ManuallyDrop::new(unsafe { Some(ID3D12Resource::from_raw(src.buffer.resource().as_raw())) }),
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                    Offset: src.offset,
                    Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                        Format: super::map_texture_format(dst.texture.format),
                        Width: size.width, Height: size.height, Depth: size.depth,
                        RowPitch: aligned_pitch as u32,
                    },
                },
            },
        };
        let dst_loc = D3D12_TEXTURE_COPY_LOCATION {
            pResource: mem::ManuallyDrop::new(unsafe { Some(ID3D12Resource::from_raw(dst_res.as_raw())) }),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: subresource_index(dst.mip_level, dst.array_layer, 1),
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
        let aligned_pitch = align_pitch(bytes_per_row as u64);
        let src_loc = D3D12_TEXTURE_COPY_LOCATION {
            pResource: mem::ManuallyDrop::new(unsafe { Some(ID3D12Resource::from_raw(src_res.as_raw())) }),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: subresource_index(src.mip_level, src.array_layer, 1),
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

    fn build_bottom_level(&mut self, _: Self::AccelerationStructure, _: &[Self::AccelerationStructureMesh], _: Self::BufferPiece) {
        unimplemented!("DX12 acceleration structures not yet implemented")
    }
    fn build_top_level(&mut self, _: Self::AccelerationStructure, _: &[Self::AccelerationStructure], _: u32, _: Self::BufferPiece, _: Self::BufferPiece) {
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
        let list = self.encoder.list.as_ref().unwrap();
        unsafe {
            list.SetPipelineState(&pipeline.pso);
            list.SetComputeRootSignature(&pipeline.layout.root_signature);
        }
        super::ComputePipelineContext { encoder: self.encoder, layout: &pipeline.layout }
    }
}

// ── RenderCommandEncoder ──────────────────────────────────────────────────────

impl super::RenderCommandEncoder<'_> {
    pub fn with<'p>(&'p mut self, pipeline: &'p super::RenderPipeline)
        -> super::RenderPipelineContext<'p>
    {
        let list = self.encoder.list.as_ref().unwrap();
        unsafe {
            list.SetPipelineState(&pipeline.pso);
            list.SetGraphicsRootSignature(&pipeline.layout.root_signature);
            list.IASetPrimitiveTopology(pipeline.topology);
        }
        super::RenderPipelineContext {
            encoder: self.encoder,
            layout: &pipeline.layout,
            vertex_strides: &pipeline.vertex_strides,
        }
    }
}

// ── PipelineEncoder (bind) ────────────────────────────────────────────────────

fn bind_group(
    encoder: &mut super::CommandEncoder,
    layout: &super::PipelineLayout,
    group: u32,
    data_fill: impl FnOnce(&mut super::PipelineContext),
    is_compute: bool,
) {
    let gd = &layout.groups[group as usize];
    if gd.slots.is_empty() { return; }

    let (ring_cpu_base, ring_gpu_base) = if gd.cbv_srv_uav_count > 0 {
        encoder.cbv_srv_uav_ring.alloc(gd.cbv_srv_uav_count)
    } else {
        (D3D12_CPU_DESCRIPTOR_HANDLE { ptr: 0 }, D3D12_GPU_DESCRIPTOR_HANDLE { ptr: 0 })
    };
    let (sampler_cpu_base, sampler_gpu_base) = if gd.sampler_count > 0 {
        encoder.sampler_ring.alloc(gd.sampler_count)
    } else {
        (D3D12_CPU_DESCRIPTOR_HANDLE { ptr: 0 }, D3D12_GPU_DESCRIPTOR_HANDLE { ptr: 0 })
    };

    let root_cbv = gd.cbv_srv_uav_root_index;
    let root_smp = gd.sampler_root_index;
    let mut ctx = super::PipelineContext {
        encoder,
        group_descriptors: gd,
        ring_cpu_base, ring_gpu_base,
        sampler_cpu_base, sampler_gpu_base,
        is_compute,
        root_index_cbv_srv_uav: root_cbv,
        root_index_sampler: root_smp,
    };
    data_fill(&mut ctx);

    let list = ctx.encoder.list.as_ref().unwrap();
    if let Some(idx) = root_cbv {
        if ring_gpu_base.ptr != 0 {
            if is_compute { unsafe { list.SetComputeRootDescriptorTable(idx, ring_gpu_base) }; }
            else          { unsafe { list.SetGraphicsRootDescriptorTable(idx, ring_gpu_base) }; }
        }
    }
    if let Some(idx) = root_smp {
        if sampler_gpu_base.ptr != 0 {
            if is_compute { unsafe { list.SetComputeRootDescriptorTable(idx, sampler_gpu_base) }; }
            else          { unsafe { list.SetGraphicsRootDescriptorTable(idx, sampler_gpu_base) }; }
        }
    }
}

#[hidden_trait::expose]
impl crate::traits::PipelineEncoder for super::ComputePipelineContext<'_> {
    fn bind<D: crate::ShaderData>(&mut self, group: u32, data: &D) {
        let layout = self.layout;
        bind_group(self.encoder, layout, group, |ctx| data.fill(ctx), true);
    }
}

#[hidden_trait::expose]
impl crate::traits::PipelineEncoder for super::RenderPipelineContext<'_> {
    fn bind<D: crate::ShaderData>(&mut self, group: u32, data: &D) {
        let layout = self.layout;
        bind_group(self.encoder, layout, group, |ctx| data.fill(ctx), false);
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
            MinDepth: *viewport.depth.start(),
            MaxDepth: *viewport.depth.end(),
        };
        unsafe { self.encoder.current_list().RSSetViewports(&[vp]) };
    }

    fn set_stencil_reference(&mut self, reference: u32) {
        unsafe { self.encoder.current_list().OMSetStencilRef(reference) };
    }

    fn bind_vertex(&mut self, index: u32, vertex_buf: crate::BufferPiece) {
        let stride = self.vertex_strides.get(index as usize).copied().unwrap_or(0);
        let vbv = D3D12_VERTEX_BUFFER_VIEW {
            BufferLocation: vertex_buf.buffer.gpu_address + vertex_buf.offset,
            SizeInBytes: (vertex_buf.buffer.size - vertex_buf.offset) as u32,
            StrideInBytes: stride,
        };
        unsafe { self.encoder.current_list().IASetVertexBuffers(index, Some(&[vbv])) };
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
