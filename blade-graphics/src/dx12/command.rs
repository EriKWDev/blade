use std::mem;
use windows::Win32::{
    Foundation::RECT,
    Graphics::{Direct3D::D3D_PRIMITIVE_TOPOLOGY, Direct3D12::*},
};

// ── ShaderBindable impls ──────────────────────────────────────────────────────

impl<T: bytemuck::Pod> crate::ShaderBindable for T {
    fn bind_to(&self, ctx: &mut super::PipelineContext, index: u32) {
        let slot = &ctx.group_descriptors.slots[index as usize];
        assert_eq!(slot.kind, super::BindingKind::Cbv);

        let data = bytemuck::bytes_of(self);
        // Pad to 256 bytes (DX12 CBV minimum size)
        let mut padded = [0u8; 256];
        let len = data.len().min(256);
        padded[..len].copy_from_slice(&data[..len]);

        let gpu_addr = ctx.encoder.upload_ring.write_cbv(&padded[..256]);

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
        let (src_handle, heap_type) = match slot.kind {
            super::BindingKind::Srv => (
                super::raw_cpu_handle(self.srv_handle),
                D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            ),
            super::BindingKind::Uav => (
                super::raw_cpu_handle(self.uav_handle),
                D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            ),
            _ => panic!("unexpected binding kind for TextureView"),
        };
        let dst_cpu = D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: ctx.ring_cpu_base.ptr
                + (slot.heap_offset * ctx.encoder.cbv_srv_uav_ring.increment) as usize,
        };
        unsafe {
            ctx.encoder.device.CopyDescriptorsSimple(1, dst_cpu, src_handle, heap_type);
        }
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
        assert_eq!(slot.kind, super::BindingKind::Sampler);
        let src_handle = super::raw_cpu_handle(self.cpu_handle);
        let dst_cpu = D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: ctx.sampler_cpu_base.ptr
                + (slot.heap_offset * ctx.encoder.sampler_ring.increment) as usize,
        };
        unsafe {
            ctx.encoder.device.CopyDescriptorsSimple(
                1,
                dst_cpu,
                src_handle,
                D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
            );
        }
    }
}

impl crate::ShaderBindable for crate::BufferPiece {
    fn bind_to(&self, ctx: &mut super::PipelineContext, index: u32) {
        let slot = &ctx.group_descriptors.slots[index as usize];
        let (src_handle, heap_type) = match slot.kind {
            super::BindingKind::Srv => (
                super::raw_cpu_handle(self.buffer.srv_handle),
                D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            ),
            super::BindingKind::Uav => (
                super::raw_cpu_handle(self.buffer.uav_handle),
                D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            ),
            super::BindingKind::Cbv => {
                // Use root CBV (buffer GPU address + offset) - create CBV inline
                let addr = self.buffer.gpu_address + self.offset;
                let size = ((self.buffer.size - self.offset) as u32 + 255) & !255;
                let cbv_cpu = D3D12_CPU_DESCRIPTOR_HANDLE {
                    ptr: ctx.ring_cpu_base.ptr
                        + (slot.heap_offset * ctx.encoder.cbv_srv_uav_ring.increment) as usize,
                };
                unsafe {
                    ctx.encoder.device.CreateConstantBufferView(
                        Some(&D3D12_CONSTANT_BUFFER_VIEW_DESC {
                            BufferLocation: addr,
                            SizeInBytes: size.max(256),
                        }),
                        cbv_cpu,
                    );
                }
                return;
            }
            _ => panic!("unexpected binding kind for BufferPiece"),
        };
        let dst_cpu = D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: ctx.ring_cpu_base.ptr
                + (slot.heap_offset * ctx.encoder.cbv_srv_uav_ring.increment) as usize,
        };
        unsafe {
            ctx.encoder.device.CopyDescriptorsSimple(1, dst_cpu, src_handle, heap_type);
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

// ── CommandEncoder trait ──────────────────────────────────────────────────────

#[hidden_trait::expose]
impl crate::traits::CommandEncoder for super::CommandEncoder {
    type Texture = super::Texture;
    type Frame = super::Frame;

    fn start(&mut self) {
        profiling::function_scope!();
        // Rotate allocators (ring-buffer of allocators, one per in-flight frame)
        self.allocators.rotate_left(1);
        let allocator = &self.allocators[0];
        unsafe { allocator.Reset().unwrap() };

        let list = self.list.as_ref().unwrap();
        unsafe { list.Reset(allocator, None).unwrap() };

        // Reset descriptor ring allocators
        self.cbv_srv_uav_ring.reset();
        self.sampler_ring.reset();
        self.upload_ring.reset();

        // Bind the GPU-visible heaps to the command list
        let heaps: [Option<ID3D12DescriptorHeap>; 2] = [
            Some(self.cbv_srv_uav_ring.heap.clone()),
            Some(self.sampler_ring.heap.clone()),
        ];
        unsafe { list.SetDescriptorHeaps(&heaps) };

        self.timings.clear();
    }

    fn init_texture(&mut self, _texture: super::Texture) {
        // DX12 textures start in D3D12_RESOURCE_STATE_COMMON; no transition needed.
    }

    fn present(&mut self, frame: super::Frame) {
        profiling::function_scope!();
        // Transition back buffer from RENDER_TARGET → PRESENT
        let barrier = transition_barrier(
            &frame.resource,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_RESOURCE_STATE_PRESENT,
        );
        let list = self.list.as_ref().unwrap();
        unsafe { list.ResourceBarrier(&[barrier]) };

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

// ── CommandEncoder public methods ─────────────────────────────────────────────

impl super::CommandEncoder {
    pub(super) fn current_list(&self) -> &ID3D12GraphicsCommandList {
        self.list.as_ref().unwrap()
    }

    pub fn set_auto_barriers(&mut self, _auto_barriers: bool) {
        // DX12 backend uses explicit barriers; this is a no-op for now.
    }

    pub fn barrier(&mut self) {
        // Full UAV barrier to flush all pending writes/reads
        let barrier = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                UAV: mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                    pResource: unsafe { mem::zeroed() },
                }),
            },
        };
        unsafe { self.current_list().ResourceBarrier(&[barrier]) };
    }

    pub fn barrier_modifies_indirect(&mut self) {
        self.barrier();
    }

    pub fn transfer(&mut self, _label: &str) -> super::TransferCommandEncoder<'_> {
        super::TransferCommandEncoder {
            list: self.list.as_ref().unwrap(),
            device: &self.device,
        }
    }

    pub fn acceleration_structure(
        &mut self,
        _label: &str,
    ) -> super::AccelerationStructureCommandEncoder<'_> {
        super::AccelerationStructureCommandEncoder {
            list: self.list.as_ref().unwrap(),
        }
    }

    pub fn compute(&mut self, _label: &str) -> super::ComputeCommandEncoder<'_> {
        super::ComputeCommandEncoder { encoder: self }
    }

    pub fn render(
        &mut self,
        _label: &str,
        targets: crate::RenderTargetSet,
    ) -> super::RenderCommandEncoder<'_> {
        let list = self.list.as_ref().unwrap();

        let mut target_size = [0u16; 2];
        let mut rtv_handles: Vec<D3D12_CPU_DESCRIPTOR_HANDLE> = Vec::new();
        let mut dsv_handle: Option<D3D12_CPU_DESCRIPTOR_HANDLE> = None;

        for rt in targets.colors {
            target_size = rt.view.target_size;

            // Transition back buffer (or texture) to RENDER_TARGET state
            let barrier = transition_barrier(
                rt.view.resource(),
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
            );
            unsafe { list.ResourceBarrier(&[barrier]) };

            let rtv = super::raw_cpu_handle(rt.view.rtv_dsv_handle);
            rtv_handles.push(rtv);

            // Clear if requested
            match rt.init_op {
                crate::InitOp::Clear(color) => {
                    let rgba = map_texture_color(color, rt.view.format);
                    unsafe { list.ClearRenderTargetView(rtv, &rgba, None) };
                }
                _ => {}
            }
        }

        if let Some(ref rt) = targets.depth_stencil {
            target_size = rt.view.target_size;
            let dsv = super::raw_cpu_handle(rt.view.rtv_dsv_handle);
            dsv_handle = Some(dsv);

            let mut clear_flags = D3D12_CLEAR_FLAGS(0);
            let mut clear_depth = 0.0f32;
            let mut clear_stencil = 0u8;

            match rt.depth_init_op {
                crate::InitOp::Clear(v) => {
                    clear_flags |= D3D12_CLEAR_FLAG_DEPTH;
                    clear_depth = v;
                }
                _ => {}
            }
            match rt.stencil_init_op {
                crate::InitOp::Clear(v) => {
                    clear_flags |= D3D12_CLEAR_FLAG_STENCIL;
                    clear_stencil = v as u8;
                }
                _ => {}
            }
            if clear_flags.0 != 0 {
                unsafe {
                    list.ClearDepthStencilView(dsv, clear_flags, clear_depth, clear_stencil, None)
                };
            }
        }

        unsafe {
            list.OMSetRenderTargets(
                rtv_handles.len() as u32,
                if rtv_handles.is_empty() { None } else { Some(rtv_handles.as_ptr()) },
                false,
                dsv_handle.as_ref(),
            );
        }

        // Set default viewport and scissor
        let w = target_size[0] as f32;
        let h = target_size[1] as f32;
        let viewport = D3D12_VIEWPORT {
            TopLeftX: 0.0,
            TopLeftY: 0.0,
            Width: w,
            Height: h,
            MinDepth: 0.0,
            MaxDepth: 1.0,
        };
        let scissor = RECT {
            left: 0,
            top: 0,
            right: target_size[0] as i32,
            bottom: target_size[1] as i32,
        };
        unsafe {
            list.RSSetViewports(&[viewport]);
            list.RSSetScissorRects(&[scissor]);
        }

        super::RenderCommandEncoder { encoder: self, target_size }
    }
}

// ── TransferEncoder ───────────────────────────────────────────────────────────

#[hidden_trait::expose]
impl crate::traits::TransferEncoder for super::TransferCommandEncoder<'_> {
    type BufferPiece = crate::BufferPiece;
    type TexturePiece = crate::TexturePiece;

    fn fill_buffer(&mut self, dst: crate::BufferPiece, size: u64, value: u8) {
        // DX12 fill via ClearUnorderedAccessViewUint requires a UAV and a GPU heap.
        // For simplicity, fill via a staging upload approach is complex.
        // Use a workaround: write to a staging buffer and copy.
        // For now, panic if value != 0, and use a zero-init workaround otherwise.
        if value == 0 {
            // Create a small zeroed staging resource and copy repeatedly
            // This is complex; use CopyBufferRegion with a zero source if possible.
            // For now, just note this limitation.
            log::warn!("fill_buffer with value=0 on DX12 is a no-op in this implementation");
        } else {
            log::warn!("fill_buffer with non-zero value not fully implemented on DX12");
        }
    }

    fn copy_buffer_to_buffer(
        &mut self,
        src: crate::BufferPiece,
        dst: crate::BufferPiece,
        size: u64,
    ) {
        unsafe {
            self.list.CopyBufferRegion(
                dst.buffer.resource(),
                dst.offset,
                src.buffer.resource(),
                src.offset,
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
        let src_loc = D3D12_TEXTURE_COPY_LOCATION {
            pResource: unsafe { mem::ManuallyDrop::new(Some(src.texture.resource().clone())) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: subresource_index(src.mip_level, src.array_layer, 1),
            },
        };
        let dst_loc = D3D12_TEXTURE_COPY_LOCATION {
            pResource: unsafe { mem::ManuallyDrop::new(Some(dst.texture.resource().clone())) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: subresource_index(dst.mip_level, dst.array_layer, 1),
            },
        };
        let src_box = D3D12_BOX {
            left: src.origin[0],
            top: src.origin[1],
            front: src.origin[2],
            right: src.origin[0] + size.width,
            bottom: src.origin[1] + size.height,
            back: src.origin[2] + size.depth,
        };
        unsafe {
            self.list.CopyTextureRegion(
                &dst_loc,
                dst.origin[0],
                dst.origin[1],
                dst.origin[2],
                &src_loc,
                Some(&src_box),
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
        let block_info = dst.texture.format.block_info();
        let row_pitch = bytes_per_row as u64;
        let aligned_row_pitch = (row_pitch + D3D12_TEXTURE_DATA_PITCH_ALIGNMENT as u64 - 1)
            & !(D3D12_TEXTURE_DATA_PITCH_ALIGNMENT as u64 - 1);

        let src_loc = D3D12_TEXTURE_COPY_LOCATION {
            pResource: unsafe { mem::ManuallyDrop::new(Some(src.buffer.resource().clone())) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                    Offset: src.offset,
                    Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                        Format: super::map_texture_format(dst.texture.format),
                        Width: size.width,
                        Height: size.height,
                        Depth: size.depth,
                        RowPitch: aligned_row_pitch as u32,
                    },
                },
            },
        };
        let dst_loc = D3D12_TEXTURE_COPY_LOCATION {
            pResource: unsafe { mem::ManuallyDrop::new(Some(dst.texture.resource().clone())) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: subresource_index(dst.mip_level, dst.array_layer, 1),
            },
        };
        unsafe {
            self.list.CopyTextureRegion(
                &dst_loc,
                dst.origin[0],
                dst.origin[1],
                dst.origin[2],
                &src_loc,
                None,
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
        let aligned_row_pitch = (bytes_per_row as u64
            + D3D12_TEXTURE_DATA_PITCH_ALIGNMENT as u64 - 1)
            & !(D3D12_TEXTURE_DATA_PITCH_ALIGNMENT as u64 - 1);

        let src_loc = D3D12_TEXTURE_COPY_LOCATION {
            pResource: unsafe { mem::ManuallyDrop::new(Some(src.texture.resource().clone())) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: subresource_index(src.mip_level, src.array_layer, 1),
            },
        };
        let dst_loc = D3D12_TEXTURE_COPY_LOCATION {
            pResource: unsafe { mem::ManuallyDrop::new(Some(dst.buffer.resource().clone())) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                    Offset: dst.offset,
                    Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                        Format: super::map_texture_format(src.texture.format),
                        Width: size.width,
                        Height: size.height,
                        Depth: size.depth,
                        RowPitch: aligned_row_pitch as u32,
                    },
                },
            },
        };
        let src_box = D3D12_BOX {
            left: src.origin[0],
            top: src.origin[1],
            front: src.origin[2],
            right: src.origin[0] + size.width,
            bottom: src.origin[1] + size.height,
            back: src.origin[2] + size.depth,
        };
        unsafe {
            self.list.CopyTextureRegion(
                &dst_loc,
                0,
                0,
                0,
                &src_loc,
                Some(&src_box),
            );
        }
    }
}

// ── AccelerationStructureEncoder ──────────────────────────────────────────────

#[hidden_trait::expose]
impl crate::traits::AccelerationStructureEncoder
    for super::AccelerationStructureCommandEncoder<'_>
{
    type AccelerationStructure = super::AccelerationStructure;
    type AccelerationStructureMesh = crate::AccelerationStructureMesh;
    type BufferPiece = crate::BufferPiece;

    fn build_bottom_level(
        &mut self,
        _acceleration_structure: Self::AccelerationStructure,
        _meshes: &[Self::AccelerationStructureMesh],
        _scratch_data: Self::BufferPiece,
    ) {
        unimplemented!("DX12 acceleration structures not yet implemented")
    }

    fn build_top_level(
        &mut self,
        _acceleration_structure: Self::AccelerationStructure,
        _bottom_level: &[Self::AccelerationStructure],
        _instance_count: u32,
        _instance_data: Self::BufferPiece,
        _scratch_data: Self::BufferPiece,
    ) {
        unimplemented!("DX12 acceleration structures not yet implemented")
    }
}

// ── ComputeCommandEncoder ─────────────────────────────────────────────────────

impl super::ComputeCommandEncoder<'_> {
    pub fn with_pipeline<'p>(
        &mut self,
        pipeline: &'p super::ComputePipeline,
    ) -> super::ComputePipelineContext<'_> {
        let list = self.encoder.list.as_ref().unwrap();
        unsafe {
            list.SetPipelineState(&pipeline.pso);
            list.SetComputeRootSignature(&pipeline.layout.root_signature);
        }
        super::ComputePipelineContext {
            encoder: self.encoder,
            layout: &pipeline.layout,
        }
    }
}

// ── RenderCommandEncoder ──────────────────────────────────────────────────────

impl super::RenderCommandEncoder<'_> {
    pub fn with_pipeline<'p>(
        &mut self,
        pipeline: &'p super::RenderPipeline,
    ) -> super::RenderPipelineContext<'_> {
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
    if gd.slots.is_empty() {
        return;
    }

    // Allocate descriptor space in the GPU rings for this group
    let (ring_cpu_base, ring_gpu_base) = if gd.cbv_srv_uav_count > 0 {
        encoder.cbv_srv_uav_ring.alloc(gd.cbv_srv_uav_count)
    } else {
        (
            D3D12_CPU_DESCRIPTOR_HANDLE { ptr: 0 },
            D3D12_GPU_DESCRIPTOR_HANDLE { ptr: 0 },
        )
    };
    let (sampler_cpu_base, sampler_gpu_base) = if gd.sampler_count > 0 {
        encoder.sampler_ring.alloc(gd.sampler_count)
    } else {
        (
            D3D12_CPU_DESCRIPTOR_HANDLE { ptr: 0 },
            D3D12_GPU_DESCRIPTOR_HANDLE { ptr: 0 },
        )
    };

    let root_cbv = gd.cbv_srv_uav_root_index;
    let root_smp = gd.sampler_root_index;

    let mut ctx = super::PipelineContext {
        encoder,
        group_descriptors: gd,
        ring_cpu_base,
        ring_gpu_base,
        sampler_cpu_base,
        sampler_gpu_base,
        is_compute,
        root_index_cbv_srv_uav: root_cbv,
        root_index_sampler: root_smp,
    };

    data_fill(&mut ctx);

    // Set the root descriptor tables
    let list = ctx.encoder.list.as_ref().unwrap();
    if let Some(root_idx) = root_cbv {
        if ring_gpu_base.ptr != 0 {
            if is_compute {
                unsafe { list.SetComputeRootDescriptorTable(root_idx, ring_gpu_base) };
            } else {
                unsafe { list.SetGraphicsRootDescriptorTable(root_idx, ring_gpu_base) };
            }
        }
    }
    if let Some(root_idx) = root_smp {
        if sampler_gpu_base.ptr != 0 {
            if is_compute {
                unsafe { list.SetComputeRootDescriptorTable(root_idx, sampler_gpu_base) };
            } else {
                unsafe { list.SetGraphicsRootDescriptorTable(root_idx, sampler_gpu_base) };
            }
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
            self.encoder
                .list
                .as_ref()
                .unwrap()
                .Dispatch(groups[0], groups[1], groups[2]);
        }
    }

    fn dispatch_indirect(&mut self, indirect_buf: crate::BufferPiece) {
        let sig = self.encoder.dispatch_sig.clone();
        unsafe {
            self.encoder.current_list().ExecuteIndirect(
                &sig,
                1,
                indirect_buf.buffer.resource(),
                indirect_buf.offset,
                None,
                0,
            );
        }
    }
}

// ── RenderEncoder ─────────────────────────────────────────────────────────────

#[hidden_trait::expose]
impl crate::traits::RenderEncoder for super::RenderPipelineContext<'_> {
    type BufferPiece = crate::BufferPiece;

    fn set_scissor_rect(&mut self, rect: &crate::ScissorRect) {
        let r = RECT {
            left: rect.x,
            top: rect.y,
            right: rect.x + rect.w as i32,
            bottom: rect.y + rect.h as i32,
        };
        unsafe { self.encoder.current_list().RSSetScissorRects(&[r]) };
    }

    fn set_viewport(&mut self, viewport: &crate::Viewport) {
        let vp = D3D12_VIEWPORT {
            TopLeftX: viewport.x,
            TopLeftY: viewport.y,
            Width: viewport.w,
            Height: viewport.h,
            MinDepth: *viewport.depth.start(),
            MaxDepth: *viewport.depth.end(),
        };
        unsafe { self.encoder.current_list().RSSetViewports(&[vp]) };
    }

    fn set_stencil_reference(&mut self, reference: u32) {
        unsafe { self.encoder.current_list().OMSetStencilRef(reference) };
    }

    fn bind_vertex(&mut self, index: u32, vertex_buf: crate::BufferPiece) {
        let stride = self
            .vertex_strides
            .get(index as usize)
            .copied()
            .unwrap_or(0);
        let vbv = D3D12_VERTEX_BUFFER_VIEW {
            BufferLocation: vertex_buf.buffer.gpu_address + vertex_buf.offset,
            SizeInBytes: (vertex_buf.buffer.size - vertex_buf.offset) as u32,
            StrideInBytes: stride,
        };
        unsafe {
            self.encoder
                .current_list()
                .IASetVertexBuffers(index, Some(&[vbv]));
        }
    }
}

// ── RenderPipelineEncoder ─────────────────────────────────────────────────────

#[hidden_trait::expose]
impl crate::traits::RenderPipelineEncoder for super::RenderPipelineContext<'_> {
    fn draw(
        &mut self,
        first_vertex: u32,
        vertex_count: u32,
        first_instance: u32,
        instance_count: u32,
    ) {
        unsafe {
            self.encoder
                .current_list()
                .DrawInstanced(vertex_count, instance_count, first_vertex, first_instance);
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
            list.DrawIndexedInstanced(
                index_count,
                instance_count,
                0,
                base_vertex,
                start_instance,
            );
        }
    }

    fn draw_indirect(&mut self, indirect_buf: crate::BufferPiece, draw_count: u32) {
        let sig = self.encoder.draw_sig.clone();
        unsafe {
            self.encoder.current_list().ExecuteIndirect(
                &sig,
                draw_count,
                indirect_buf.buffer.resource(),
                indirect_buf.offset,
                None,
                0,
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
                &sig,
                draw_count,
                indirect_buf.buffer.resource(),
                indirect_buf.offset,
                None,
                0,
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
                &sig,
                max_draw_count,
                indirect_buf.buffer.resource(),
                indirect_buf.offset,
                Some(count_buf.buffer.resource()),
                count_buf.offset,
            );
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn subresource_index(mip: u32, array_layer: u32, mip_count: u32) -> u32 {
    mip + array_layer * mip_count
}

fn transition_barrier(
    resource: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: unsafe { mem::ManuallyDrop::new(Some(resource.clone())) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

fn map_texture_color(color: crate::TextureColor, _format: crate::TextureFormat) -> [f32; 4] {
    match color {
        crate::TextureColor::TransparentBlack => [0.0, 0.0, 0.0, 0.0],
        crate::TextureColor::OpaqueBlack => [0.0, 0.0, 0.0, 1.0],
        crate::TextureColor::White => [1.0, 1.0, 1.0, 1.0],
        crate::TextureColor::RgbaFloat { rgba } => rgba,
    }
}
