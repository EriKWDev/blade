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
                // an arbitrary prior state (e.g. UNORDERED_ACCESS), so transition —
                // except for any subresource that is the open pass's live
                // attachment (kept in its RT/DEPTH state).
                ctx.encoder
                    .require_view_state_skip_active_rt(self, READ_STATE);
                // No SRV exists unless the texture had RESOURCE usage; a null
                // handle would fault inside the runtime with no message.
                assert_ne!(
                    self.srv_handle, 0,
                    "DX12: sampled binding of a texture view that has no SRV — its                      texture was created without TextureUsage::RESOURCE"
                );
                super::raw_cpu_handle(self.srv_handle)
            }
            super::BindingKind::Uav => {
                // Regular textures never auto-promote COMMON->UNORDERED_ACCESS, so
                // this explicit transition is mandatory for storage textures.
                ctx.encoder
                    .require_view_state_skip_active_rt(self, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                assert_ne!(
                    self.uav_handle, 0,
                    "DX12: storage binding of a texture view that has no UAV — its                      texture was created without TextureUsage::STORAGE (or its format                      is depth/stencil, which cannot have one)"
                );
                super::raw_cpu_handle(self.uav_handle)
            }
            other => panic!("unexpected binding kind {:?} for TextureView", other),
        };
        let dst = D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: ctx.ring_cpu_base.ptr
                + (slot.heap_offset * ctx.encoder.cbv_srv_uav_ring.increment) as usize,
        };
        unsafe {
            ctx.encoder.device.CopyDescriptorsSimple(
                1,
                dst,
                src,
                D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            )
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
        // Resolved as a group in `bind_group`, so a sampler shared by many draws
        // takes one ring slot. The shader reads nagaSamplerHeap[indexbuf[register]].
        ctx.encoder.sampler_binds.push((slot.register, self.cpu_handle));
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
        // A whole-buffer piece matches the pre-made descriptor, so copy that
        // instead of re-creating a view on every bind.
        let raw_first = (self.offset / 4) as u32;
        let avail = self.buffer.size.saturating_sub(self.offset);
        // Belt sub-allocations set size and cover only part of the remainder.
        let extent = if self.size == 0 {
            avail
        } else {
            self.size.min(avail)
        };
        let raw_num = (extent / 4) as u32;
        let whole_buffer = self.offset == 0 && extent >= self.buffer.size;
        match slot.kind {
            super::BindingKind::Srv => {
                ctx.encoder.require_buffer_state(&self.buffer, READ_STATE);
                if whole_buffer && self.buffer.srv_handle != 0 {
                    unsafe {
                        ctx.encoder.device.CopyDescriptorsSimple(
                            1,
                            dst,
                            super::raw_cpu_handle(self.buffer.srv_handle),
                            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                        )
                    };
                    return;
                }
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
                if whole_buffer && self.buffer.uav_handle != 0 {
                    unsafe {
                        ctx.encoder.device.CopyDescriptorsSimple(
                            1,
                            dst,
                            super::raw_cpu_handle(self.buffer.uav_handle),
                            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                        )
                    };
                    return;
                }
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
                ctx.encoder.require_buffer_state(
                    &self.buffer,
                    D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
                );
                let addr = self.buffer.gpu_address + self.offset;
                let size = ((extent as u32) + 255) & !255;
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

/// Wrap a COM resource pointer for a windows-rs descriptor field (barrier /
/// copy-location `pResource`), which is typed as an *owned*
/// `ManuallyDrop<Option<ID3D12Resource>>`. We are only borrowing for the
/// duration of the API call: `from_raw` does not AddRef and `ManuallyDrop`
/// suppresses the Release, so the refcount is unchanged — the owning reference
/// stays with the Texture/Buffer and is released exactly once in `destroy_*`.
fn borrowed_resource(raw: super::ResourcePtr) -> mem::ManuallyDrop<Option<ID3D12Resource>> {
    mem::ManuallyDrop::new(Some(unsafe { ID3D12Resource::from_raw(raw as *mut _) }))
}

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
                pResource: borrowed_resource(resource_vtbl),
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

/// Buffer states that are read-only and therefore OR-able into a single
/// combined state — the same buffer can legally sit in several of these at once
/// (a device-belt chunk read as both index buffer and a vertex-stage storage
/// SRV within one draw is exactly this). Write states (UNORDERED_ACCESS,
/// COPY_DEST, RENDER_TARGET, …) are exclusive and never combined.
const BUFFER_READ_STATES: D3D12_RESOURCE_STATES = D3D12_RESOURCE_STATES(
    D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER.0
        | D3D12_RESOURCE_STATE_INDEX_BUFFER.0
        | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE.0
        | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE.0
        | D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT.0
        | D3D12_RESOURCE_STATE_COPY_SOURCE.0,
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
        skip_active_rt: bool,
    ) {
        if vtbl.is_null() {
            return; // null placeholder view
        }
        let mut barriers: Vec<D3D12_RESOURCE_BARRIER> = Vec::new();
        {
            // Disjoint field borrows: the state map is written while the
            // open pass's attachment list is only read.
            let active_render_targets = &self.active_render_targets;
            let entry = self
                .texture_states
                .entry(vtbl as usize)
                .or_insert_with(|| vec![D3D12_RESOURCE_STATE_COMMON; total.max(1) as usize]);
            for s in subs {
                if skip_active_rt && active_render_targets.contains(&(vtbl, s)) {
                    continue;
                }
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
        self.require_subresources(view.resource_ptr, total, view.subresources(), needed, false);
    }

    /// Like `require_view_state`, but leaves any subresource that is a live
    /// attachment of the open render pass in its RENDER_TARGET/DEPTH state. Used
    /// when binding a sampled/storage texture: a subresource that is also the
    /// current attachment cannot be transitioned without breaking the in-flight
    /// draws (Vulkan tolerates the overlap via GENERAL layout; DX12 cannot). Any
    /// *other* subresource of the same resource (e.g. a different bloom mip) is
    /// transitioned normally.
    pub(super) fn require_view_state_skip_active_rt(
        &mut self,
        view: &super::TextureView,
        needed: D3D12_RESOURCE_STATES,
    ) {
        let total = view.total_subresources();
        self.require_subresources(view.resource_ptr, total, view.subresources(), needed, true);
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
        let vtbl = tex.resource().as_raw() as super::ResourcePtr;
        self.require_subresources(vtbl, total, std::iter::once(sub), needed, false);
    }

    /// Transition a buffer (whole resource). Host-visible buffers are tracked the
    /// same as device buffers: they live on a CUSTOM heap (NOT the GENERIC_READ-
    /// locked UPLOAD heap), so they are transitionable, and they MUST be
    /// transitioned. The "buffers auto-promote from COMMON" rule only holds while
    /// the buffer is still in COMMON — once a belt copy promotes it to COPY_DEST
    /// (write state), it stays there and does NOT auto-promote to
    /// INDIRECT_ARGUMENT / UNORDERED_ACCESS for a later use. Skipping host-visible
    /// buffers here left compute-produced indirect-arg/count buffers in COPY_DEST
    /// at ExecuteIndirect → device removal (DEVICE_HUNG).
    pub(super) fn require_buffer_state(
        &mut self,
        buffer: &super::Buffer,
        needed: D3D12_RESOURCE_STATES,
    ) {
        if buffer.resource_ptr.is_null() {
            return;
        }
        let vtbl = buffer.resource().as_raw() as super::ResourcePtr;
        let key = vtbl as usize;
        let cur = self
            .buffer_states
            .get(&key)
            .copied()
            .unwrap_or(D3D12_RESOURCE_STATE_COMMON);
        // When both current and needed are read-only, widen to their union so a
        // buffer bound two ways at once (index buffer + shader SRV) ends up in a
        // state valid for every concurrent read, rather than the last writer
        // winning and leaving the other access in an incompatible state.
        let both_read =
            (cur.0 & !BUFFER_READ_STATES.0) == 0 && (needed.0 & !BUFFER_READ_STATES.0) == 0;
        let target = if both_read {
            D3D12_RESOURCE_STATES(cur.0 | needed.0)
        } else {
            needed
        };
        if cur.0 != target.0 {
            let barrier =
                transition_barrier(vtbl, cur, target, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES);
            unsafe { self.list.as_ref().unwrap().ResourceBarrier(&[barrier]) };
            self.buffer_states.insert(key, target);
        }
    }

    pub(super) fn current_list(&self) -> &ID3D12GraphicsCommandList {
        self.list.as_ref().unwrap()
    }

    /// Allocate `count` CBV/SRV/UAV descriptors, first ensuring the active heap
    /// has room (grow the main heap, or spill to another heap once at the
    /// hardware cap). Returns the base CPU/GPU handles and the relative offset
    /// (recorded so the binding can be re-pointed if a later grow/spill moves the
    /// active heap).
    pub(super) fn cbv_alloc(
        &mut self,
        count: u32,
    ) -> (
        D3D12_CPU_DESCRIPTOR_HANDLE,
        D3D12_GPU_DESCRIPTOR_HANDLE,
        u32,
    ) {
        if self.cbv_srv_uav_ring.would_overflow(count) {
            self.ensure_cbv_space(count);
        }
        let rel = self.cbv_srv_uav_ring.rel(self.cbv_srv_uav_ring.offset);
        let (_cpu, gpu) = self.cbv_srv_uav_ring.alloc(count);
        (self.cbv_srv_uav_ring.write_cpu_rel(rel), gpu, rel)
    }

    /// Make room for `count` more descriptors. First try to grow the main heap
    /// (bigger heap, live region copied, relative offsets preserved). If the main
    /// heap is already at the hardware cap, spill: copy the live binding set into
    /// a fresh cap-sized heap (compacted), and continue there. Either way the old
    /// heap stays alive for already-recorded draws, the heaps are rebound, and
    /// every active root table is re-issued (a heap switch invalidates table
    /// bindings, and a later draw may reuse a group without re-binding it).
    fn ensure_cbv_space(&mut self, count: u32) {
        let device = self.device.clone();
        if let Some(old) = self.cbv_srv_uav_ring.grow_main(&device, count) {
            self.retired_cbv_heaps.push((self.frame_index, old));
            self.rebind_cbv_heap();
            if !self.cbv_srv_uav_ring.would_overflow(count) {
                return;
            }
            // Grew to the cap but still short — fall through to spill.
        }
        // Spill: compact the active binding set into a fresh cap-sized heap. The
        // set any single draw needs at once is small (a few groups) and <= cap;
        // only the frame total exceeds it.
        let carry: u32 = self.active_cbv_tables.iter().map(|&(_, _, c, _)| c).sum();
        let needed = carry + count;
        let _ = carry;
        assert!(
            needed <= self.cbv_srv_uav_ring.max_capacity,
            "single draw needs {needed} descriptors, more than the {} one heap holds",
            self.cbv_srv_uav_ring.max_capacity
        );
        // Compact the active groups into the new heap and rewrite their offsets.
        let tables: Vec<(u32, u32)> = self
            .active_cbv_tables
            .iter()
            .map(|&(_, rel, c, _)| (rel, c))
            .collect();
        let offsets = self
            .cbv_srv_uav_ring
            .spill_tables(&device, &tables, count, self.frame_index);
        for (entry, off) in self.active_cbv_tables.iter_mut().zip(offsets) {
            entry.1 = off;
        }
        self.rebind_cbv_heap();
    }

    /// After a grow/spill switched the active CBV/SRV/UAV heap: rebind both heaps
    /// and re-issue every active root descriptor table into the new heap (offsets
    /// in `active_cbv_tables` are current — grow preserves them, spill rewrote
    /// them).
    fn rebind_cbv_heap(&mut self) {
        let heaps: [Option<ID3D12DescriptorHeap>; 2] = [
            Some(self.cbv_srv_uav_ring.heap.clone()),
            Some(self.sampler_ring.heap.clone()),
        ];
        let list = self.list.as_ref().unwrap();
        unsafe { list.SetDescriptorHeaps(&heaps) };
        for &(root, rel, _count, is_compute) in &self.active_cbv_tables {
            let gpu = self.cbv_srv_uav_ring.gpu_handle_rel(rel);
            unsafe {
                if is_compute {
                    list.SetComputeRootDescriptorTable(root, gpu);
                } else {
                    list.SetGraphicsRootDescriptorTable(root, gpu);
                }
            }
        }
        // Sampler heap is unchanged, but SetDescriptorHeaps invalidated its
        // table binding too; re-issue it.
        if let Some((std_root, cmp_root, is_compute)) = self.bound_sampler_roots {
            let sampler_gpu = self.sampler_ring.gpu_start;
            unsafe {
                if is_compute {
                    list.SetComputeRootDescriptorTable(std_root, sampler_gpu);
                    list.SetComputeRootDescriptorTable(cmp_root, sampler_gpu);
                } else {
                    list.SetGraphicsRootDescriptorTable(std_root, sampler_gpu);
                    list.SetGraphicsRootDescriptorTable(cmp_root, sampler_gpu);
                }
            }
        }
    }

    /// Ring slot holding `cpu_handle`, copying the sampler into the GPU-visible
    /// heap the first time this frame that it is asked for. See `sampler_slots`.
    pub(super) fn sampler_slot(&mut self, cpu_handle: u64) -> u32 {
        if let Some(&index) = self.sampler_slots.get(&cpu_handle) {
            return index;
        }
        assert!(
            !self.sampler_ring.would_overflow(1),
            "DX12: more than {} distinct samplers used in one frame; a shader-visible              sampler heap cannot exceed the hardware limit",
            self.sampler_ring.max_capacity
        );
        let index = self.sampler_ring.offset;
        let (dst, _gpu) = self.sampler_ring.alloc(1);
        unsafe {
            self.device.CopyDescriptorsSimple(
                1,
                dst,
                super::raw_cpu_handle(cpu_handle),
                D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
            )
        };
        self.sampler_slots.insert(cpu_handle, index);
        index
    }

    /// Record that `root` now points at the CBV/SRV/UAV table of `count`
    /// descriptors at relative offset `rel`, so a later grow/spill can re-point
    /// (and, for spill, copy) it. Replaces any prior entry for the same root.
    pub(super) fn record_cbv_table(&mut self, root: u32, rel: u32, count: u32, is_compute: bool) {
        match self
            .active_cbv_tables
            .iter_mut()
            .find(|(r, _, _, _)| *r == root)
        {
            Some(e) => *e = (root, rel, count, is_compute),
            None => self.active_cbv_tables.push((root, rel, count, is_compute)),
        }
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

    pub fn barrier_compute_to_indirect_and_vertex(&mut self) {
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
    fn begin_pass(&mut self, label: &str) {
        self.active_render_targets.clear();
        // Emit the pass label as a marker (PIX/RenderDoc surface it) and record it
        // in the DRED label table. Each begin_pass emits exactly one SetMarker op,
        // so on a hang, counting marker ops up to the hung op indexes this label.
        if !label.is_empty() {
            if let Some(list) = self.list.as_ref() {
                let wide: Vec<u16> = label.encode_utf16().chain(std::iter::once(0)).collect();
                // PIX_EVENT_UNICODE_VERSION: the marker payload is a UTF-16 string.
                const PIX_EVENT_UNICODE_VERSION: u32 = 2;
                unsafe {
                    list.SetMarker(
                        PIX_EVENT_UNICODE_VERSION,
                        Some(wide.as_ptr() as *const std::ffi::c_void),
                        (wide.len() * 2) as u32,
                    );
                }
            }
            super::dred_record_pass(self.id, label);
        }
        if self.auto_barriers {
            self.barrier();
        }
    }

    /// Whether variable-rate shading is usable at all on this device+runtime.
    pub(super) fn supports_shading_rate(&self) -> bool {
        self.list5.is_some()
            && self.shading_rate_tier.0 >= D3D12_VARIABLE_SHADING_RATE_TIER_1.0
    }

    /// Set the per-draw shading rate. `use_image` picks the second combiner:
    /// OVERRIDE lets the shading-rate image win, PASSTHROUGH keeps this rate.
    /// Mirrors Vulkan's KEEP/REPLACE pair.
    pub(super) fn set_shading_rate(&mut self, rate: crate::FragmentShadingRate, use_image: bool) {
        let Some(list5) = self.list5.as_ref() else {
            return;
        };
        let combiners = [
            D3D12_SHADING_RATE_COMBINER_PASSTHROUGH,
            if use_image {
                D3D12_SHADING_RATE_COMBINER_OVERRIDE
            } else {
                D3D12_SHADING_RATE_COMBINER_PASSTHROUGH
            },
        ];
        unsafe { list5.RSSetShadingRate(map_shading_rate(rate), Some(combiners.as_ptr())) };
    }

    /// Bind (or, with `None`, unbind) the screen-space shading-rate image.
    pub(super) fn set_shading_rate_image(&mut self, view: Option<&super::TextureView>) {
        let Some(list5) = self.list5.as_ref() else {
            return;
        };
        match view {
            Some(view) => {
                let resource = borrowed_resource(view.resource_ptr);
                unsafe { list5.RSSetShadingRateImage(resource.as_ref().unwrap()) };
            }
            None => unsafe { list5.RSSetShadingRateImage(None) },
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
            _list: self.list.as_ref().unwrap(),
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
        let mut resolves: Vec<super::PendingResolve> = Vec::new();

        // Transition all attachments first (require_state borrows the list internally),
        // then record clears / OMSetRenderTargets under a single list borrow.
        for rt in targets.colors {
            target_size = rt.view.target_size;
            // -> RENDER_TARGET (not a promotable state; tracked so a later sample
            // or present transitions out of it correctly).
            self.require_view_state(&rt.view, D3D12_RESOURCE_STATE_RENDER_TARGET);
            for sub in rt.view.subresources() {
                self.active_render_targets.push((rt.view.resource_ptr, sub));
            }
            // No RTV unless the texture had TARGET usage.
            assert_ne!(
                rt.view.rtv_dsv_handle, 0,
                "DX12: color attachment view has no RTV — its texture was created                  without TextureUsage::TARGET"
            );
            rtv_handles.push(super::raw_cpu_handle(rt.view.rtv_dsv_handle));
            if let crate::FinishOp::ResolveTo { view, mode, .. } = rt.finish_op {
                resolves.push(super::PendingResolve {
                    src: rt.view,
                    dst: view,
                    mode,
                });
            }
        }
        if let Some(ref rt) = targets.depth_stencil {
            target_size = rt.view.target_size;
            self.require_view_state(&rt.view, D3D12_RESOURCE_STATE_DEPTH_WRITE);
            for sub in rt.view.subresources() {
                self.active_render_targets.push((rt.view.resource_ptr, sub));
            }
            assert_ne!(
                rt.view.rtv_dsv_handle, 0,
                "DX12: depth-stencil attachment view has no DSV — its texture was                  created without TextureUsage::TARGET"
            );
            dsv_handle = Some(super::raw_cpu_handle(rt.view.rtv_dsv_handle));
            if let crate::FinishOp::ResolveTo { view, mode, .. } = rt.depth_finish_op {
                resolves.push(super::PendingResolve {
                    src: rt.view,
                    dst: view,
                    mode,
                });
            }
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
                if rtv_handles.is_empty() {
                    None
                } else {
                    Some(rtv_handles.as_ptr())
                },
                false,
                dsv_handle.as_ref().map(|h| h as *const _),
            );
            list.RSSetViewports(&[D3D12_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: target_size[0] as f32,
                Height: target_size[1] as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            }]);
            list.RSSetScissorRects(&[RECT {
                left: 0,
                top: 0,
                right: target_size[0] as i32,
                bottom: target_size[1] as i32,
            }]);
        }

        super::RenderCommandEncoder {
            encoder: self,
            resolves,
            has_shading_rate_image: false,
        }
    }

    /// Render pass with a shading-rate image (VRS tier 2). The image starts
    /// inactive; a draw opts in via `set_fragment_shading_rate`, as on Vulkan.
    pub fn render_with_fragment_shading_rate_attachment(
        &mut self,
        label: &str,
        targets: crate::RenderTargetSet,
        attachment: crate::TextureView,
        texel_size: [u32; 2],
    ) -> super::RenderCommandEncoder<'_> {
        // The attachment is read as a rate source for the whole pass; transition
        // it before the pass binds its render targets.
        self.require_view_state(&attachment, D3D12_RESOURCE_STATE_SHADING_RATE_SOURCE);
        let supported = self.supports_shading_rate()
            && self.shading_rate_tier.0 >= D3D12_VARIABLE_SHADING_RATE_TIER_2.0;
        debug_assert!(
            !supported || texel_size[0] == texel_size[1],
            "DX12 shading-rate image tiles are square"
        );
        let mut pass = self.render(label, targets);
        if supported {
            pass.encoder.set_shading_rate_image(Some(&attachment));
            pass.encoder
                .set_shading_rate(crate::FragmentShadingRate::FULL, false);
            pass.has_shading_rate_image = true;
        }
        pass
    }

    pub fn render_multisampled_to_single_sampled_with_fragment_shading_rate_attachment(
        &mut self,
        _label: &str,
        _targets: crate::RenderTargetSet,
        _sample_count: u32,
        _attachment: crate::TextureView,
        _texel_size: [u32; 2],
    ) -> super::RenderCommandEncoder<'_> {
        unreachable!("multisampled render-to-single-sampled is not supported by DX12")
    }

    pub fn render_multisampled_to_single_sampled(
        &mut self,
        _label: &str,
        _targets: crate::RenderTargetSet,
        _sample_count: u32,
    ) -> super::RenderCommandEncoder<'_> {
        unreachable!("multisampled render-to-single-sampled is not supported by DX12")
    }

    /// Put a shading-rate image into the state it is read in, so the per-frame
    /// upload path only has to bounce it through COPY_DEST and back.
    pub fn init_fragment_shading_rate_attachment(&mut self, texture: super::Texture) {
        let total = texture.mip_levels * texture.array_layers * super::plane_count(texture.format);
        let vtbl = texture.resource().as_raw() as super::ResourcePtr;
        self.require_subresources(
            vtbl,
            total,
            0..total,
            D3D12_RESOURCE_STATE_SHADING_RATE_SOURCE,
            false,
        );
    }

    /// Perform an MSAA resolve from `src` (the multisampled attachment) into `dst`.
    /// Emitted at render-pass end for each `FinishOp::ResolveTo`. Mirrors Vulkan's
    /// resolve-attachment, which blade expresses as a finish op.
    pub(super) fn resolve_view(
        &mut self,
        src: &super::TextureView,
        dst: &super::TextureView,
        mode: crate::ResolveMode,
    ) {
        self.require_view_state(src, D3D12_RESOURCE_STATE_RESOLVE_SOURCE);
        self.require_view_state(dst, D3D12_RESOURCE_STATE_RESOLVE_DEST);
        let format = super::map_texture_format(dst.format);
        let src_res = borrowed_resource(src.resource_ptr);
        let dst_res = borrowed_resource(dst.resource_ptr);
        let list = self.list.as_ref().unwrap();
        for (s, d) in src.subresources().zip(dst.subresources()) {
            match mode {
                crate::ResolveMode::Average => unsafe {
                    list.ResolveSubresource(
                        dst_res.as_ref().unwrap(),
                        d,
                        src_res.as_ref().unwrap(),
                        s,
                        format,
                    );
                },
                _ => {
                    // Min/Max/Sample0 (e.g. depth resolve) need the typed-mode
                    // entry point on ID3D12GraphicsCommandList1.
                    let d3d_mode = match mode {
                        crate::ResolveMode::Min => D3D12_RESOLVE_MODE_MIN,
                        crate::ResolveMode::Max => D3D12_RESOLVE_MODE_MAX,
                        // Sample0 has no DX12 equivalent; MIN matches the common
                        // "nearest depth" reduction used for depth resolves.
                        crate::ResolveMode::Sample0 => D3D12_RESOLVE_MODE_MIN,
                        crate::ResolveMode::Average => D3D12_RESOLVE_MODE_AVERAGE,
                    };
                    let list1: ID3D12GraphicsCommandList1 = list.cast().unwrap();
                    unsafe {
                        list1.ResolveSubresourceRegion(
                            dst_res.as_ref().unwrap(),
                            d,
                            0,
                            0,
                            src_res.as_ref().unwrap(),
                            s,
                            None,
                            format,
                            d3d_mode,
                        );
                    }
                }
            }
        }
    }
}

impl super::TransferCommandEncoder<'_> {
    /// Upload a complete shading-rate image. The state tracker handles the
    /// COPY_DEST/SHADING_RATE_SOURCE transitions, so this is just a copy.
    pub fn copy_buffer_to_fragment_shading_rate_attachment(
        &mut self,
        src: crate::BufferPiece,
        bytes_per_row: u32,
        dst: crate::TexturePiece,
        size: crate::Extent,
    ) {
        self.copy_buffer_to_texture(src, bytes_per_row, dst, size);
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

        let buffer_count = self.allocators.len() as u64;
        let segment = (self.frame_index % buffer_count) as u32;
        self.frame_index += 1;
        self.cbv_srv_uav_ring
            .begin_segment(segment, self.frame_index, buffer_count);
        self.sampler_ring
            .begin_segment(segment, self.frame_index, buffer_count);
        self.upload_ring.begin_segment(segment as u64);
        // Free heaps retired by a mid-frame grow once their frame's submission is
        // complete. start() just CPU-waited the allocator last used buffer_count
        // frames ago, so every frame with index <= frame_index - buffer_count is
        // finished; a heap tagged with such a frame (referenced only by that
        // frame's already-recorded draws) is now safe to drop.
        let completed = self.frame_index.saturating_sub(buffer_count);
        self.retired_cbv_heaps.retain(|(f, _)| *f > completed);

        // Root-table bindings do not survive across command lists; replay state
        // starts empty each frame (re-populated by with()/bind()).
        self.active_cbv_tables.clear();
        self.bound_sampler_roots = None;
        // The sampler ring has just rewound to this frame's segment, so the
        // slots resolved for the previous frame no longer describe it.
        self.sampler_slots.clear();
        // All resources decay to COMMON at the ExecuteCommandLists boundary, so a
        // fresh command list starts with every resource implicitly in COMMON.
        self.buffer_states.clear();
        self.texture_states.clear();
        super::dred_clear_passes(self.id);

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
        let frame_vtbl = frame.resource.as_raw() as super::ResourcePtr;
        self.require_subresources(
            frame_vtbl,
            1,
            std::iter::once(0),
            D3D12_RESOURCE_STATE_COMMON,
            false,
        );

        self.present = Some(super::Presentation {
            swapchain: frame.swapchain,
            sync_interval: frame.sync_interval,
            allow_tearing: frame.allow_tearing,
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
        // The clear covers exactly what the UAV describes, so the view must be
        // built for this range — the buffer's whole-range UAV would wipe every
        // other belt sub-allocation in the same buffer. Raw UAVs are DWORD-addressed.
        let value_u32 = (value as u32) * 0x01010101;
        let available = dst.buffer.size.saturating_sub(dst.offset);
        let size = size.min(available);
        if size < 4 {
            return;
        }
        debug_assert_eq!(dst.offset % 4, 0, "DX12 raw UAV clears are DWORD-addressed");

        // The buffer must be in UNORDERED_ACCESS to be cleared through a UAV.
        self.encoder
            .require_buffer_state(&dst.buffer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

        let uav_desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_UAV {
                    FirstElement: dst.offset / 4,
                    NumElements: (size / 4) as u32,
                    StructureByteStride: 0,
                    CounterOffsetInBytes: 0,
                    Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                },
            },
        };

        // The clear needs the same view twice: once in the bound shader-visible
        // heap (which may grow here, hence `cbv_alloc`) and once in a CPU-only
        // heap.
        let (ring_cpu, ring_gpu, rel) = self.encoder.cbv_alloc(1);
        let enc = &mut *self.encoder;
        unsafe {
            let resource = dst.buffer.resource();
            enc.device
                .CreateUnorderedAccessView(resource, None, Some(&uav_desc), ring_cpu);
            let device = enc.device.clone();
            enc.cbv_srv_uav_ring.publish(&device, rel, 1);
            enc.current_list().ClearUnorderedAccessViewUint(
                ring_gpu,
                // ring_cpu is the shadow: non-shader-visible, as required here.
                ring_cpu,
                resource,
                &[value_u32; 4],
                &[], // no scissor rects → clear the view's whole range
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
        let src_res = src.texture.resource();
        let dst_res = dst.texture.resource();
        self.encoder
            .require_piece_state(&src, D3D12_RESOURCE_STATE_COPY_SOURCE);
        self.encoder
            .require_piece_state(&dst, D3D12_RESOURCE_STATE_COPY_DEST);
        let src_loc = D3D12_TEXTURE_COPY_LOCATION {
            pResource: borrowed_resource(src_res.as_raw()),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: subresource_index(
                    src.mip_level,
                    src.array_layer,
                    src.texture.mip_levels,
                ),
            },
        };
        let dst_loc = D3D12_TEXTURE_COPY_LOCATION {
            pResource: borrowed_resource(dst_res.as_raw()),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: subresource_index(
                    dst.mip_level,
                    dst.array_layer,
                    dst.texture.mip_levels,
                ),
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
            self.encoder.current_list().CopyTextureRegion(
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
        self.encoder
            .require_piece_state(&dst, D3D12_RESOURCE_STATE_COPY_DEST);

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
            (src.buffer.resource().as_raw(), src.offset)
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
            pResource: borrowed_resource(footprint_res),
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                    Offset: footprint_offset,
                    Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                        Format: super::map_texture_format(dst.texture.format),
                        Width: size.width,
                        Height: size.height,
                        Depth: size.depth,
                        RowPitch: aligned_pitch as u32,
                    },
                },
            },
        };
        let dst_loc = D3D12_TEXTURE_COPY_LOCATION {
            pResource: borrowed_resource(dst.texture.resource().as_raw()),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: subresource_index(
                    dst.mip_level,
                    dst.array_layer,
                    dst.texture.mip_levels,
                ),
            },
        };
        unsafe {
            self.encoder.current_list().CopyTextureRegion(
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
        let src_res = src.texture.resource();
        self.encoder
            .require_piece_state(&src, D3D12_RESOURCE_STATE_COPY_SOURCE);
        self.encoder
            .require_buffer_state(&dst.buffer, D3D12_RESOURCE_STATE_COPY_DEST);
        let aligned_pitch = align_pitch(bytes_per_row as u64);
        let src_loc = D3D12_TEXTURE_COPY_LOCATION {
            pResource: borrowed_resource(src_res.as_raw()),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: subresource_index(
                    src.mip_level,
                    src.array_layer,
                    src.texture.mip_levels,
                ),
            },
        };
        let dst_loc = D3D12_TEXTURE_COPY_LOCATION {
            pResource: borrowed_resource(dst.buffer.resource().as_raw()),
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                    Offset: dst.offset,
                    Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                        Format: super::map_texture_format(src.texture.format),
                        Width: size.width,
                        Height: size.height,
                        Depth: size.depth,
                        RowPitch: aligned_pitch as u32,
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
            self.encoder.current_list().CopyTextureRegion(
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

    pub fn with<'p>(
        &'p mut self,
        pipeline: &'p super::ComputePipeline,
    ) -> super::ComputePipelineContext<'p> {
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
        // Setting the root signature invalidated all root-table bindings; reset
        // the heap-grow replay cache to this pipeline's state.
        self.encoder.active_cbv_tables.clear();
        self.encoder.bound_sampler_roots = pipeline
            .layout
            .sampler_heap_roots
            .map(|(s, c)| (s, c, true));
        super::ComputePipelineContext {
            encoder: self.encoder,
            layout: &pipeline.layout,
        }
    }
}

// ── RenderCommandEncoder ──────────────────────────────────────────────────────

impl super::RenderCommandEncoder<'_> {
    pub fn set_fragment_shading_rate(&mut self, rate: crate::FragmentShadingRate) -> bool {
        if !self.encoder.supports_shading_rate() {
            return false;
        }
        // With an attachment bound, a non-full request means "let the image
        // decide"; a full request means "go back to shading every pixel".
        let use_image = self.has_shading_rate_image && rate != crate::FragmentShadingRate::FULL;
        self.encoder.set_shading_rate(rate, use_image);
        true
    }

    pub fn begin_pipeline_statistics(&mut self, _label: &str) {}
    pub fn end_pipeline_statistics(&mut self) {}

    /// No-op: DX12 has no timestamp queries here, so `timings()` stays empty.
    pub fn diagnostic_timestamp(&mut self, _label: &str) {}

    pub fn with<'p>(
        &'p mut self,
        pipeline: &'p super::RenderPipeline,
    ) -> super::RenderPipelineContext<'p> {
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
        // Setting the root signature invalidated all root-table bindings; reset
        // the heap-grow replay cache to this pipeline's state.
        self.encoder.active_cbv_tables.clear();
        self.encoder.bound_sampler_roots = pipeline
            .layout
            .sampler_heap_roots
            .map(|(s, c)| (s, c, false));
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
            MinDepth: viewport.depth.start,
            MaxDepth: viewport.depth.end,
        };
        unsafe { self.encoder.current_list().RSSetViewports(&[vp]) };
    }

    fn set_stencil_reference(&mut self, reference: u32) {
        unsafe { self.encoder.current_list().OMSetStencilRef(reference) };
    }

    fn bind_vertex(&mut self, index: u32, vertex_buf: crate::BufferPiece) {
        self.encoder.require_buffer_state(
            &vertex_buf.buffer,
            D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
        );
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
    if gd.slots.is_empty() {
        return;
    }

    // Allocate BEFORE fill() so any heap grow happens before descriptors are
    // written — fill() writes into `ring_cpu_base` and issues no ring allocs of
    // its own, so the base stays valid through it. `cbv_rel` is the in-segment
    // offset, recorded below so a later grow can re-point this table.
    let (ring_cpu_base, ring_gpu_base, cbv_rel) = if gd.cbv_srv_uav_count > 0 {
        encoder.cbv_alloc(gd.cbv_srv_uav_count)
    } else {
        (
            D3D12_CPU_DESCRIPTOR_HANDLE { ptr: 0 },
            D3D12_GPU_DESCRIPTOR_HANDLE { ptr: 0 },
            0,
        )
    };
    let root_cbv = gd.cbv_srv_uav_root_index;
    let root_sampler_index = gd.sampler_index_root_index;

    // `fill` consumes the PipelineContext (matching ShaderData::fill's by-value
    // signature); samplers it touches land in `encoder.sampler_binds`.
    encoder.sampler_binds.clear();
    data.fill(super::PipelineContext {
        encoder: &mut *encoder,
        group_descriptors: gd,
        ring_cpu_base,
    });

    // Build and bind this group's sampler-index buffer: one ring slot per
    // shader register, resolved through the frame's dedup table.
    if let Some(idx) = root_sampler_index {
        let binds = std::mem::take(&mut encoder.sampler_binds);
        let mut indices = std::mem::take(&mut encoder.sampler_indices);
        indices.clear();
        indices.resize(gd.sampler_count as usize, 0);
        for &(register, cpu_handle) in &binds {
            let slot = encoder.sampler_slot(cpu_handle);
            indices[register as usize] = slot;
        }
        let va = encoder
            .upload_ring
            .write_aligned(bytemuck::cast_slice(&indices), 16);
        let list = encoder.list.as_ref().unwrap();
        if is_compute {
            unsafe { list.SetComputeRootShaderResourceView(idx, va) };
        } else {
            unsafe { list.SetGraphicsRootShaderResourceView(idx, va) };
        }
        encoder.sampler_binds = binds;
        encoder.sampler_indices = indices;
    }

    if gd.cbv_srv_uav_count > 0 {
        let device = encoder.device.clone();
        encoder
            .cbv_srv_uav_ring
            .publish(&device, cbv_rel, gd.cbv_srv_uav_count);
    }

    if let Some(idx) = root_cbv {
        if ring_gpu_base.ptr != 0 {
            let list = encoder.list.as_ref().unwrap();
            if is_compute {
                unsafe { list.SetComputeRootDescriptorTable(idx, ring_gpu_base) };
            } else {
                unsafe { list.SetGraphicsRootDescriptorTable(idx, ring_gpu_base) };
            }
            // Remember this table so a mid-frame heap grow/spill can re-point it.
            encoder.record_cbv_table(idx, cbv_rel, gd.cbv_srv_uav_count, is_compute);
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
            self.encoder
                .current_list()
                .Dispatch(groups[0], groups[1], groups[2]);
        }
    }

    fn dispatch_indirect(&mut self, indirect_buf: crate::BufferPiece) {
        self.encoder
            .require_buffer_state(&indirect_buf.buffer, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
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
        self.encoder.require_buffer_state(
            &vertex_buf.buffer,
            D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
        );
        // Apply now with this pipeline's stride, and record it so a subsequent
        // `with()` (next pipeline in the pass) re-applies it.
        let stride = self
            .vertex_strides
            .get(index as usize)
            .copied()
            .unwrap_or(0);
        self.encoder.set_vertex_buffer(index, &vertex_buf, stride);
        self.encoder.record_vertex(index, vertex_buf);
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
            self.encoder.current_list().DrawInstanced(
                vertex_count,
                instance_count,
                first_vertex,
                first_instance,
            );
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

/// blade's shading-rate encoding — `(log2(width) << 2) | log2(height)` — is
/// deliberately identical to `D3D12_SHADING_RATE`, so this is a cast, and an
/// uploaded shading-rate image needs no per-texel translation either.
fn map_shading_rate(rate: crate::FragmentShadingRate) -> D3D12_SHADING_RATE {
    D3D12_SHADING_RATE(rate.attachment_texel_value() as i32)
}

fn map_texture_color(color: crate::TextureColor) -> [f32; 4] {
    match color {
        crate::TextureColor::TransparentBlack => [0.0, 0.0, 0.0, 0.0],
        crate::TextureColor::OpaqueBlack => [0.0, 0.0, 0.0, 1.0],
        crate::TextureColor::White => [1.0, 1.0, 1.0, 1.0],
        crate::TextureColor::RgbaFloat { rgba } => rgba,
    }
}
