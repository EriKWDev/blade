use std::ptr;
use windows::{
    core::Interface,
    Win32::Graphics::{Direct3D12::*, Dxgi::Common::*},
};

impl super::Context {
    pub(super) fn alloc_rtv(&self) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let idx = self.rtv_alloc.lock().unwrap().alloc(1);
        assert!(idx < super::RTV_HEAP_SIZE, "RTV heap exhausted");
        self.rtv_heap.cpu_handle(idx)
    }

    pub(super) fn alloc_dsv(&self) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let idx = self.dsv_alloc.lock().unwrap().alloc(1);
        assert!(idx < super::DSV_HEAP_SIZE, "DSV heap exhausted");
        self.dsv_heap.cpu_handle(idx)
    }

    pub(super) fn alloc_staging(&self, count: u32) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let idx = self.staging_alloc.lock().unwrap().alloc(count);
        assert!(idx + count <= super::STAGING_HEAP_SIZE, "staging heap exhausted");
        self.staging_heap.cpu_handle(idx)
    }

    pub(super) fn alloc_staging_sampler(&self) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let idx = self.staging_sampler_alloc.lock().unwrap().alloc(1);
        assert!(idx < super::STAGING_SAMPLER_SIZE, "staging sampler heap exhausted");
        self.staging_sampler_heap.cpu_handle(idx)
    }
}

#[hidden_trait::expose]
impl crate::traits::ResourceDevice for super::Context {
    type Buffer = super::Buffer;
    type Texture = super::Texture;
    type TextureView = super::TextureView;
    type Sampler = super::Sampler;
    type AccelerationStructure = super::AccelerationStructure;

    fn create_buffer(&self, desc: super::super::BufferDesc) -> super::Buffer {
        let is_host_visible = desc.memory.is_host_visible();
        let heap_type = match desc.memory {
            crate::Memory::Device | crate::Memory::External(_) => D3D12_HEAP_TYPE_DEFAULT,
            crate::Memory::Shared | crate::Memory::Upload => D3D12_HEAP_TYPE_UPLOAD,
        };
        let initial_state = match heap_type {
            D3D12_HEAP_TYPE_UPLOAD => D3D12_RESOURCE_STATE_GENERIC_READ,
            _ => D3D12_RESOURCE_STATE_COMMON,
        };
        let aligned_size = (desc.size + 255) & !255;

        // A UAV requires ALLOW_UNORDERED_ACCESS on the resource. That flag is only
        // legal on DEFAULT-heap buffers (not UPLOAD), so device buffers get it (and
        // a UAV); upload/shared buffers get neither. Creating a UAV on a buffer
        // without the flag removes the device (DXGI_ERROR_INVALID_CALL).
        let res_flags = if heap_type == D3D12_HEAP_TYPE_DEFAULT {
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
        } else {
            D3D12_RESOURCE_FLAG_NONE
        };

        let mut resource: Option<ID3D12Resource> = None;
        super::expect_d3d12(&self.device, "CreateCommittedResource(buffer)", unsafe {
            self.device.CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES { Type: heap_type, ..Default::default() },
                D3D12_HEAP_FLAG_NONE,
                &D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: aligned_size,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    Flags: res_flags,
                    ..Default::default()
                },
                initial_state,
                None,
                &mut resource,
            )
        });
        let resource = resource.unwrap();
        let gpu_address = unsafe { resource.GetGPUVirtualAddress() };

        let mapped_ptr = if is_host_visible {
            let mut p: *mut std::ffi::c_void = ptr::null_mut();
            unsafe { resource.Map(0, None, Some(&mut p)).unwrap() };
            p as *mut u8
        } else {
            ptr::null_mut()
        };

        // SRV needs no special resource flag, so it is always available.
        let srv_handle = self.alloc_staging(1);
        unsafe {
            self.device.CreateShaderResourceView(
                &resource,
                Some(&D3D12_SHADER_RESOURCE_VIEW_DESC {
                    Format: DXGI_FORMAT_R32_TYPELESS,
                    ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                    Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                    Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                        Buffer: D3D12_BUFFER_SRV {
                            FirstElement: 0,
                            NumElements: (aligned_size / 4) as u32,
                            StructureByteStride: 0,
                            Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                        },
                    },
                }),
                srv_handle,
            );
        }

        // UAV only for DEFAULT-heap buffers (which carry ALLOW_UNORDERED_ACCESS).
        let uav_handle = if heap_type == D3D12_HEAP_TYPE_DEFAULT {
            let h = self.alloc_staging(1);
            unsafe {
                self.device.CreateUnorderedAccessView(
                    &resource,
                    None,
                    Some(&D3D12_UNORDERED_ACCESS_VIEW_DESC {
                        Format: DXGI_FORMAT_R32_TYPELESS,
                        ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                        Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                            Buffer: D3D12_BUFFER_UAV {
                                FirstElement: 0,
                                NumElements: (aligned_size / 4) as u32,
                                StructureByteStride: 0,
                                CounterOffsetInBytes: 0,
                                Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                            },
                        },
                    }),
                    h,
                );
            }
            h.ptr as u64
        } else {
            0
        };

        // Box the COM object so `resource_ptr` is a stable `Box<ID3D12Resource>` pointer.
        let resource_ptr = Box::into_raw(Box::new(resource)) as super::ResourcePtr;

        super::Buffer {
            resource_ptr,
            gpu_address,
            mapped_ptr,
            srv_handle: srv_handle.ptr as u64,
            uav_handle,
            size: desc.size,
        }
    }

    fn sync_buffer(&self, _buffer: super::Buffer) {}

    fn destroy_buffer(&self, buffer: super::Buffer) {
        if !buffer.resource_ptr.is_null() {
            unsafe { drop(Box::from_raw(buffer.resource_ptr as *mut ID3D12Resource)) };
        }
    }

    fn create_texture(&self, desc: super::super::TextureDesc) -> super::Texture {
        let format = super::map_texture_format(desc.format);
        let dimension = match desc.dimension {
            crate::TextureDimension::D1 => D3D12_RESOURCE_DIMENSION_TEXTURE1D,
            crate::TextureDimension::D2 => D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            crate::TextureDimension::D3 => D3D12_RESOURCE_DIMENSION_TEXTURE3D,
        };
        let mut flags = D3D12_RESOURCE_FLAG_NONE;
        if desc.usage.contains(crate::TextureUsage::TARGET) {
            if desc.format.aspects().contains(crate::TexelAspects::DEPTH) {
                flags |= D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
            } else {
                flags |= D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
            }
        }
        if desc.usage.contains(crate::TextureUsage::STORAGE) {
            flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        }

        let mut resource: Option<ID3D12Resource> = None;
        super::expect_d3d12(&self.device, "CreateCommittedResource(texture)", unsafe {
            self.device.CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
                D3D12_HEAP_FLAG_NONE,
                &D3D12_RESOURCE_DESC {
                    Dimension: dimension,
                    Width: desc.size.width as u64,
                    Height: desc.size.height,
                    DepthOrArraySize: if desc.dimension == crate::TextureDimension::D3 {
                        desc.size.depth as u16
                    } else {
                        desc.array_layer_count as u16
                    },
                    MipLevels: desc.mip_level_count as u16,
                    Format: format,
                    SampleDesc: DXGI_SAMPLE_DESC { Count: desc.sample_count, Quality: 0 },
                    Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
                    Flags: flags,
                    ..Default::default()
                },
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut resource,
            )
        });
        let resource = resource.unwrap();
        let resource_ptr = Box::into_raw(Box::new(resource)) as super::ResourcePtr;
        let array_layers = if desc.dimension == crate::TextureDimension::D3 {
            1
        } else {
            desc.array_layer_count
        };
        super::Texture {
            resource_ptr,
            usage: desc.usage,
            memory_handle: 0,
            format: desc.format,
            target_size: [desc.size.width as u16, desc.size.height as u16],
            mip_levels: desc.mip_level_count,
            array_layers,
        }
    }

    fn destroy_texture(&self, texture: super::Texture) {
        if texture.memory_handle != !0 && !texture.resource_ptr.is_null() {
            unsafe { drop(Box::from_raw(texture.resource_ptr as *mut ID3D12Resource)) };
        }
    }

    fn create_texture_view(
        &self,
        texture: super::Texture,
        desc: super::super::TextureViewDesc,
    ) -> super::TextureView {
        let format = super::map_texture_format(desc.format);
        let srv_format = super::map_depth_srv_format(desc.format);
        let aspects = desc.aspects.unwrap_or_else(|| desc.format.aspects());

        let resource = texture.resource();

        // Store the COM vtable pointer for barrier use — borrowed from the Box.
        let resource_ptr = unsafe { resource.as_raw() as *mut std::ffi::c_void };

        // D3D12 forbids creating a view kind the resource wasn't created for
        // (e.g. an RTV on a non-ALLOW_RENDER_TARGET resource), which can remove
        // the device. Only create the views the texture's usage permits. An SRV
        // is always legal (no special creation flag is required for it).
        let usage = texture.usage;
        let can_srv = usage.contains(crate::TextureUsage::RESOURCE);
        let can_uav = usage.contains(crate::TextureUsage::STORAGE)
            && !aspects.contains(crate::TexelAspects::DEPTH)
            && !aspects.contains(crate::TexelAspects::STENCIL);
        let is_depth = aspects.contains(crate::TexelAspects::DEPTH)
            || aspects.contains(crate::TexelAspects::STENCIL);
        let can_target = usage.contains(crate::TextureUsage::TARGET);

        let srv_handle = if can_srv {
            let h = self.alloc_staging(1);
            let srv_desc = make_srv_desc(srv_format, desc.dimension, desc.subresources);
            unsafe { self.device.CreateShaderResourceView(resource, Some(&srv_desc), h) };
            h.ptr as u64
        } else {
            0
        };

        let uav_handle = if can_uav {
            let h = self.alloc_staging(1);
            let uav_desc = make_uav_desc(format, desc.dimension, desc.subresources);
            unsafe { self.device.CreateUnorderedAccessView(resource, None, Some(&uav_desc), h) };
            h.ptr as u64
        } else {
            0
        };

        let rtv_dsv_handle = if can_target && is_depth {
            let h = self.alloc_dsv();
            unsafe {
                self.device.CreateDepthStencilView(
                    resource,
                    Some(&make_dsv_desc(format, desc.dimension, desc.subresources)),
                    h,
                );
            }
            h.ptr as u64
        } else if can_target {
            let h = self.alloc_rtv();
            unsafe {
                self.device.CreateRenderTargetView(
                    resource,
                    Some(&make_rtv_desc(format, desc.dimension, desc.subresources)),
                    h,
                );
            }
            h.ptr as u64
        } else {
            0
        };

        let mip_count = desc
            .subresources
            .mip_level_count
            .map_or(texture.mip_levels - desc.subresources.base_mip_level, |n| n.get());
        let array_count = desc
            .subresources
            .array_layer_count
            .map_or(texture.array_layers - desc.subresources.base_array_layer, |n| n.get());

        super::TextureView {
            resource_ptr,
            srv_handle,
            uav_handle,
            rtv_dsv_handle,
            format: desc.format,
            aspects,
            target_size: texture.target_size,
            base_mip: desc.subresources.base_mip_level,
            mip_count,
            base_array: desc.subresources.base_array_layer,
            array_count,
            mip_levels: texture.mip_levels,
            array_layers: texture.array_layers,
        }
    }

    fn destroy_texture_view(&self, _view: super::TextureView) {
        // TextureView borrows the resource; destruction is a no-op.
    }

    fn create_sampler(&self, desc: super::super::SamplerDesc) -> super::Sampler {
        let handle = self.alloc_staging_sampler();
        let [au, av, aw] = desc.address_modes;
        let lod_max = desc.lod_max_clamp.unwrap_or(f32::MAX);
        unsafe {
            self.device.CreateSampler(
                &D3D12_SAMPLER_DESC {
                    Filter: map_filter(desc.mag_filter, desc.min_filter, desc.mipmap_filter, desc.compare),
                    AddressU: map_address_mode(au),
                    AddressV: map_address_mode(av),
                    AddressW: map_address_mode(aw),
                    MipLODBias: 0.0,
                    MaxAnisotropy: desc.anisotropy_clamp.max(1),
                    ComparisonFunc: desc.compare.map_or(
                        D3D12_COMPARISON_FUNC_ALWAYS, super::map_comparison),
                    BorderColor: map_border_color(desc.border_color),
                    MinLOD: desc.lod_min_clamp,
                    MaxLOD: lod_max,
                },
                handle,
            );
        }
        super::Sampler { cpu_handle: handle.ptr as u64 }
    }

    fn destroy_sampler(&self, _sampler: super::Sampler) {}

    fn create_acceleration_structure(
        &self,
        _desc: super::super::AccelerationStructureDesc,
    ) -> super::AccelerationStructure {
        unimplemented!("DX12 acceleration structures not yet implemented")
    }

    fn destroy_acceleration_structure(&self, _as: super::AccelerationStructure) {
        unimplemented!("DX12 acceleration structures not yet implemented")
    }
}

impl super::Context {
    pub fn get_bottom_level_acceleration_structure_sizes(
        &self,
        _meshes: &[crate::AccelerationStructureMesh],
    ) -> crate::AccelerationStructureSizes {
        unimplemented!("DX12 acceleration structures not yet implemented")
    }

    pub fn get_top_level_acceleration_structure_sizes(
        &self,
        _instance_count: u32,
    ) -> crate::AccelerationStructureSizes {
        unimplemented!("DX12 acceleration structures not yet implemented")
    }

    pub fn create_acceleration_structure_instance_buffer(
        &self,
        _instances: &[crate::AccelerationStructureInstance],
        _bottom_level: &[super::AccelerationStructure],
    ) -> super::Buffer {
        unimplemented!("DX12 acceleration structures not yet implemented")
    }

    pub fn get_external_texture_source(
        &self,
        _texture: super::Texture,
    ) -> Option<crate::ExternalMemorySource> {
        // DX12 external memory is not supported yet.
        None
    }

    pub fn get_external_buffer_source(
        &self,
        _buffer: super::Buffer,
    ) -> Option<crate::ExternalMemorySource> {
        None
    }
}

// ── SRV / UAV / RTV / DSV descriptor builders ─────────────────────────────────

fn mip_range(sr: &crate::TextureSubresources) -> (u32, u32) {
    (sr.base_mip_level, sr.mip_level_count.map_or(u32::MAX, |n| n.get()))
}
fn arr_range(sr: &crate::TextureSubresources) -> (u32, u32) {
    (sr.base_array_layer, sr.array_layer_count.map_or(u32::MAX, |n| n.get()))
}

fn make_srv_desc(
    fmt: DXGI_FORMAT,
    dim: crate::ViewDimension,
    sr: &crate::TextureSubresources,
) -> D3D12_SHADER_RESOURCE_VIEW_DESC {
    let (mb, mc) = mip_range(sr);
    let (ab, ac) = arr_range(sr);
    let m = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    match dim {
        crate::ViewDimension::D1 => D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: fmt, ViewDimension: D3D12_SRV_DIMENSION_TEXTURE1D,
            Shader4ComponentMapping: m,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Texture1D: D3D12_TEX1D_SRV { MostDetailedMip: mb, MipLevels: mc, ResourceMinLODClamp: 0.0 } },
        },
        crate::ViewDimension::D1Array => D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: fmt, ViewDimension: D3D12_SRV_DIMENSION_TEXTURE1DARRAY,
            Shader4ComponentMapping: m,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Texture1DArray: D3D12_TEX1D_ARRAY_SRV { MostDetailedMip: mb, MipLevels: mc, FirstArraySlice: ab, ArraySize: ac, ResourceMinLODClamp: 0.0 } },
        },
        crate::ViewDimension::D2 => D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: fmt, ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: m,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Texture2D: D3D12_TEX2D_SRV { MostDetailedMip: mb, MipLevels: mc, PlaneSlice: 0, ResourceMinLODClamp: 0.0 } },
        },
        crate::ViewDimension::D2Array => D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: fmt, ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2DARRAY,
            Shader4ComponentMapping: m,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Texture2DArray: D3D12_TEX2D_ARRAY_SRV { MostDetailedMip: mb, MipLevels: mc, FirstArraySlice: ab, ArraySize: ac, PlaneSlice: 0, ResourceMinLODClamp: 0.0 } },
        },
        crate::ViewDimension::Cube => D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: fmt, ViewDimension: D3D12_SRV_DIMENSION_TEXTURECUBE,
            Shader4ComponentMapping: m,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 { TextureCube: D3D12_TEXCUBE_SRV { MostDetailedMip: mb, MipLevels: mc, ResourceMinLODClamp: 0.0 } },
        },
        crate::ViewDimension::CubeArray => D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: fmt, ViewDimension: D3D12_SRV_DIMENSION_TEXTURECUBEARRAY,
            Shader4ComponentMapping: m,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 { TextureCubeArray: D3D12_TEXCUBE_ARRAY_SRV { MostDetailedMip: mb, MipLevels: mc, First2DArrayFace: ab, NumCubes: ac / 6, ResourceMinLODClamp: 0.0 } },
        },
        crate::ViewDimension::D3 => D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: fmt, ViewDimension: D3D12_SRV_DIMENSION_TEXTURE3D,
            Shader4ComponentMapping: m,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Texture3D: D3D12_TEX3D_SRV { MostDetailedMip: mb, MipLevels: mc, ResourceMinLODClamp: 0.0 } },
        },
    }
}

fn make_uav_desc(
    fmt: DXGI_FORMAT,
    dim: crate::ViewDimension,
    sr: &crate::TextureSubresources,
) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    let mip = sr.base_mip_level;
    let (ab, ac) = arr_range(sr);
    match dim {
        crate::ViewDimension::D2 => D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: fmt, ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 { Texture2D: D3D12_TEX2D_UAV { MipSlice: mip, PlaneSlice: 0 } },
        },
        crate::ViewDimension::D2Array => D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: fmt, ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2DARRAY,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 { Texture2DArray: D3D12_TEX2D_ARRAY_UAV { MipSlice: mip, FirstArraySlice: ab, ArraySize: ac, PlaneSlice: 0 } },
        },
        crate::ViewDimension::D3 => D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: fmt, ViewDimension: D3D12_UAV_DIMENSION_TEXTURE3D,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 { Texture3D: D3D12_TEX3D_UAV { MipSlice: mip, FirstWSlice: ab, WSize: ac } },
        },
        _ => D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: fmt, ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 { Texture2D: D3D12_TEX2D_UAV { MipSlice: mip, PlaneSlice: 0 } },
        },
    }
}

fn make_rtv_desc(
    fmt: DXGI_FORMAT,
    dim: crate::ViewDimension,
    sr: &crate::TextureSubresources,
) -> D3D12_RENDER_TARGET_VIEW_DESC {
    let mip = sr.base_mip_level;
    let (ab, ac) = arr_range(sr);
    match dim {
        crate::ViewDimension::D2Array => D3D12_RENDER_TARGET_VIEW_DESC {
            Format: fmt, ViewDimension: D3D12_RTV_DIMENSION_TEXTURE2DARRAY,
            Anonymous: D3D12_RENDER_TARGET_VIEW_DESC_0 { Texture2DArray: D3D12_TEX2D_ARRAY_RTV { MipSlice: mip, FirstArraySlice: ab, ArraySize: ac, PlaneSlice: 0 } },
        },
        _ => D3D12_RENDER_TARGET_VIEW_DESC {
            Format: fmt, ViewDimension: D3D12_RTV_DIMENSION_TEXTURE2D,
            Anonymous: D3D12_RENDER_TARGET_VIEW_DESC_0 { Texture2D: D3D12_TEX2D_RTV { MipSlice: mip, PlaneSlice: 0 } },
        },
    }
}

fn make_dsv_desc(
    fmt: DXGI_FORMAT,
    dim: crate::ViewDimension,
    sr: &crate::TextureSubresources,
) -> D3D12_DEPTH_STENCIL_VIEW_DESC {
    let mip = sr.base_mip_level;
    let (ab, ac) = arr_range(sr);
    match dim {
        crate::ViewDimension::D2Array => D3D12_DEPTH_STENCIL_VIEW_DESC {
            Format: fmt, ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2DARRAY,
            Flags: D3D12_DSV_FLAG_NONE,
            Anonymous: D3D12_DEPTH_STENCIL_VIEW_DESC_0 { Texture2DArray: D3D12_TEX2D_ARRAY_DSV { MipSlice: mip, FirstArraySlice: ab, ArraySize: ac } },
        },
        _ => D3D12_DEPTH_STENCIL_VIEW_DESC {
            Format: fmt, ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2D,
            Flags: D3D12_DSV_FLAG_NONE,
            Anonymous: D3D12_DEPTH_STENCIL_VIEW_DESC_0 { Texture2D: D3D12_TEX2D_DSV { MipSlice: mip } },
        },
    }
}

// ── Sampler helpers ────────────────────────────────────────────────────────────

fn map_address_mode(m: crate::AddressMode) -> D3D12_TEXTURE_ADDRESS_MODE {
    match m {
        crate::AddressMode::ClampToEdge => D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        crate::AddressMode::Repeat => D3D12_TEXTURE_ADDRESS_MODE_WRAP,
        crate::AddressMode::MirrorRepeat => D3D12_TEXTURE_ADDRESS_MODE_MIRROR,
        crate::AddressMode::ClampToBorder => D3D12_TEXTURE_ADDRESS_MODE_BORDER,
    }
}

fn map_filter(
    mag: crate::FilterMode,
    min: crate::FilterMode,
    mip: crate::FilterMode,
    compare: Option<crate::CompareFunction>,
) -> D3D12_FILTER {
    match (min, mag, mip, compare.is_some()) {
        (crate::FilterMode::Nearest, crate::FilterMode::Nearest, crate::FilterMode::Nearest, false) => D3D12_FILTER_MIN_MAG_MIP_POINT,
        (crate::FilterMode::Nearest, crate::FilterMode::Nearest, crate::FilterMode::Linear,  false) => D3D12_FILTER_MIN_MAG_POINT_MIP_LINEAR,
        (crate::FilterMode::Nearest, crate::FilterMode::Linear,  crate::FilterMode::Nearest, false) => D3D12_FILTER_MIN_POINT_MAG_LINEAR_MIP_POINT,
        (crate::FilterMode::Nearest, crate::FilterMode::Linear,  crate::FilterMode::Linear,  false) => D3D12_FILTER_MIN_POINT_MAG_MIP_LINEAR,
        (crate::FilterMode::Linear,  crate::FilterMode::Nearest, crate::FilterMode::Nearest, false) => D3D12_FILTER_MIN_LINEAR_MAG_MIP_POINT,
        (crate::FilterMode::Linear,  crate::FilterMode::Nearest, crate::FilterMode::Linear,  false) => D3D12_FILTER_MIN_LINEAR_MAG_POINT_MIP_LINEAR,
        (crate::FilterMode::Linear,  crate::FilterMode::Linear,  crate::FilterMode::Nearest, false) => D3D12_FILTER_MIN_MAG_LINEAR_MIP_POINT,
        (crate::FilterMode::Linear,  crate::FilterMode::Linear,  crate::FilterMode::Linear,  false) => D3D12_FILTER_MIN_MAG_MIP_LINEAR,
        (_, _, _, true) => D3D12_FILTER_COMPARISON_MIN_MAG_MIP_LINEAR,
    }
}

fn map_border_color(color: Option<crate::TextureColor>) -> [f32; 4] {
    match color {
        None | Some(crate::TextureColor::TransparentBlack) => [0.0, 0.0, 0.0, 0.0],
        Some(crate::TextureColor::OpaqueBlack) => [0.0, 0.0, 0.0, 1.0],
        Some(crate::TextureColor::White) => [1.0, 1.0, 1.0, 1.0],
        Some(crate::TextureColor::RgbaFloat { rgba }) => rgba,
    }
}
