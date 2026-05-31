use std::collections::BTreeMap;
use windows::{
    core::PCSTR,
    Win32::Graphics::{
        Direct3D::{Fxc::D3DCompile, *},
        Direct3D12::*,
        Dxgi::Common::*,
    },
};

use super::{BindingKind, GroupDescriptors, PipelineLayout};

// ── Binding map construction ──────────────────────────────────────────────────

/// Build the naga HLSL binding map so each binding is re-mapped to
/// a dense register index (per type) within its group's register space.
fn make_binding_map(groups: &[GroupDescriptors]) -> BTreeMap<naga::ResourceBinding, naga::back::hlsl::BindTarget> {
    let mut map = BTreeMap::new();
    for (g, gd) in groups.iter().enumerate() {
        for (bi, slot) in gd.slots.iter().enumerate() {
            map.insert(
                naga::ResourceBinding { group: g as u32, binding: bi as u32 },
                naga::back::hlsl::BindTarget {
                    space: g as u8,
                    register: slot.register,
                    binding_array_size: None,
                    dynamic_storage_buffer_offsets_index: None,
                    restrict_indexing: false,
                },
            );
        }
    }
    map
}

// ── Root signature construction ───────────────────────────────────────────────

fn build_root_signature(
    device: &ID3D12Device,
    groups: &[GroupDescriptors],
) -> ID3D12RootSignature {
    // Collect all ranges in stable Vecs before taking pointers.
    let mut all_cbv_srv_uav_ranges: Vec<Vec<D3D12_DESCRIPTOR_RANGE>> = Vec::new();
    let mut all_sampler_ranges: Vec<Vec<D3D12_DESCRIPTOR_RANGE>> = Vec::new();
    let mut params: Vec<D3D12_ROOT_PARAMETER> = Vec::new();

    for (g, gd) in groups.iter().enumerate() {
        let space = g as u32;

        // CBV/SRV/UAV descriptor table
        if gd.cbv_srv_uav_count > 0 {
            let mut ranges = Vec::new();
            let (srv_count, uav_count, cbv_count) = count_types(&gd.slots);

            // Offset within the table: SRV first, then UAV, then CBV
            let mut offset = 0u32;

            if srv_count > 0 {
                ranges.push(D3D12_DESCRIPTOR_RANGE {
                    RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
                    NumDescriptors: srv_count,
                    BaseShaderRegister: 0,
                    RegisterSpace: space,
                    OffsetInDescriptorsFromTableStart: offset,
                });
                offset += srv_count;
            }
            if uav_count > 0 {
                ranges.push(D3D12_DESCRIPTOR_RANGE {
                    RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
                    NumDescriptors: uav_count,
                    BaseShaderRegister: 0,
                    RegisterSpace: space,
                    OffsetInDescriptorsFromTableStart: offset,
                });
                offset += uav_count;
            }
            if cbv_count > 0 {
                ranges.push(D3D12_DESCRIPTOR_RANGE {
                    RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_CBV,
                    NumDescriptors: cbv_count,
                    BaseShaderRegister: 0,
                    RegisterSpace: space,
                    OffsetInDescriptorsFromTableStart: offset,
                });
            }

            all_cbv_srv_uav_ranges.push(ranges);
        } else {
            all_cbv_srv_uav_ranges.push(Vec::new()); // placeholder
        }

        // Sampler descriptor table
        if gd.sampler_count > 0 {
            let ranges = vec![D3D12_DESCRIPTOR_RANGE {
                RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER,
                NumDescriptors: gd.sampler_count,
                BaseShaderRegister: 0,
                RegisterSpace: space,
                OffsetInDescriptorsFromTableStart: 0,
            }];
            all_sampler_ranges.push(ranges);
        } else {
            all_sampler_ranges.push(Vec::new()); // placeholder
        }
    }

    // Now build the root parameters pointing into the stable range Vecs
    for (g, gd) in groups.iter().enumerate() {
        if gd.cbv_srv_uav_root_index.is_some() {
            let ranges = &all_cbv_srv_uav_ranges[g];
            params.push(D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                        NumDescriptorRanges: ranges.len() as u32,
                        pDescriptorRanges: ranges.as_ptr(),
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            });
        }
        if gd.sampler_root_index.is_some() {
            let ranges = &all_sampler_ranges[g];
            params.push(D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                        NumDescriptorRanges: ranges.len() as u32,
                        pDescriptorRanges: ranges.as_ptr(),
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            });
        }
    }

    let desc = D3D12_ROOT_SIGNATURE_DESC {
        NumParameters: params.len() as u32,
        pParameters: if params.is_empty() { std::ptr::null() } else { params.as_ptr() },
        NumStaticSamplers: 0,
        pStaticSamplers: std::ptr::null(),
        Flags: D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT,
    };

    let mut blob: Option<windows::Win32::Graphics::Direct3D::ID3DBlob> = None;
    let mut error_blob: Option<windows::Win32::Graphics::Direct3D::ID3DBlob> = None;
    unsafe {
        D3D12SerializeRootSignature(
            &desc,
            D3D_ROOT_SIGNATURE_VERSION_1,
            &mut blob,
            Some(&mut error_blob),
        ).unwrap_or_else(|e| {
            if let Some(err) = error_blob {
                let ptr = err.GetBufferPointer();
                let size = err.GetBufferSize();
                let msg = std::slice::from_raw_parts(ptr as *const u8, size);
                eprintln!("Root signature error: {}", String::from_utf8_lossy(msg));
            }
            panic!("Failed to serialize root signature: {e}");
        });
    }

    let blob = blob.unwrap();
    let rs: ID3D12RootSignature = unsafe {
        device.CreateRootSignature(
            0,
            std::slice::from_raw_parts(
                blob.GetBufferPointer() as *const u8,
                blob.GetBufferSize(),
            ),
        ).unwrap()
    };
    rs
}

fn count_types(slots: &[super::BindingSlot]) -> (u32, u32, u32) {
    let mut srv = 0u32;
    let mut uav = 0u32;
    let mut cbv = 0u32;
    for s in slots {
        match s.kind {
            BindingKind::Srv => srv += 1,
            BindingKind::Uav => uav += 1,
            BindingKind::Cbv => cbv += 1,
            BindingKind::Sampler => {}
        }
    }
    (srv, uav, cbv)
}

// ── HLSL compilation helpers ──────────────────────────────────────────────────

fn compile_hlsl(source: &str, entry: &str, target: &str, debug: bool) -> Vec<u8> {
    let source_bytes = source.as_bytes();
    let entry_cstr = std::ffi::CString::new(entry).unwrap();
    let target_cstr = std::ffi::CString::new(target).unwrap();
    use windows::Win32::Graphics::Direct3D::Fxc::*;
    let flags: u32 = if debug {
        D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION
    } else {
        D3DCOMPILE_OPTIMIZATION_LEVEL3
    };
    let mut bytecode: Option<windows::Win32::Graphics::Direct3D::ID3DBlob> = None;
    let mut errors: Option<windows::Win32::Graphics::Direct3D::ID3DBlob> = None;

    let result = unsafe {
        D3DCompile(
            source_bytes.as_ptr() as *const _,
            source_bytes.len(),
            PCSTR::null(),
            None,
            None,
            PCSTR(entry_cstr.as_ptr() as _),
            PCSTR(target_cstr.as_ptr() as _),
            flags,
            0,
            &mut bytecode,
            Some(&mut errors),
        )
    };

    if let Some(err) = errors {
        let ptr = unsafe { err.GetBufferPointer() };
        let size = unsafe { err.GetBufferSize() };
        let msg = unsafe { std::slice::from_raw_parts(ptr as *const u8, size) };
        let msg_str = String::from_utf8_lossy(msg);
        if result.is_err() {
            panic!("HLSL compile error ({target}):\n{msg_str}\n\nSource:\n{source}");
        } else {
            log::warn!("HLSL compile warning ({target}): {msg_str}");
        }
    }

    result.unwrap();
    let blob = bytecode.unwrap();
    let size = unsafe { blob.GetBufferSize() };
    let ptr = unsafe { blob.GetBufferPointer() } as *const u8;
    unsafe { std::slice::from_raw_parts(ptr, size).to_vec() }
}

fn compile_shader_function(
    sf: crate::ShaderFunction,
    stage: naga::ShaderStage,
    group_layouts: &[&crate::ShaderDataLayout],
    group_infos: &mut [crate::ShaderDataInfo],
    vertex_fetch_states: &[crate::VertexFetchState],
    groups: &[GroupDescriptors],
    debug: bool,
) -> (Vec<u8>, [u32; 3]) {
    let ep_index = sf.entry_point_index();
    let ep = &sf.shader.module.entry_points[ep_index];
    let ep_info = sf.shader.info.get_entry_point(ep_index);

    let (mut module, module_info) = sf.shader.resolve_constants(&sf.constants);
    crate::Shader::fill_resource_bindings(
        &mut module,
        group_infos,
        stage,
        ep_info,
        group_layouts,
    );
    if stage == naga::ShaderStage::Vertex {
        crate::Shader::fill_vertex_locations(&mut module, ep_index, vertex_fetch_states);
    }

    let binding_map = make_binding_map(groups);
    let hlsl_options = naga::back::hlsl::Options {
        shader_model: naga::back::hlsl::ShaderModel::V5_1,
        binding_map,
        fake_missing_bindings: true,
        ..Default::default()
    };

    let pipeline_options = naga::back::hlsl::PipelineOptions {
        entry_point: Some((stage, sf.entry_point.to_string())),
    };

    let mut hlsl_source = String::new();
    let mut writer = naga::back::hlsl::Writer::new(&mut hlsl_source, &hlsl_options, &pipeline_options);
    let _reflection = writer.write(&module, &module_info, None).unwrap();

    let wg_size = if stage == naga::ShaderStage::Compute {
        let wg = ep.workgroup_size;
        [wg[0], wg[1], wg[2]]
    } else {
        [1, 1, 1]
    };

    let target_str = match stage {
        naga::ShaderStage::Compute => "cs_5_1",
        naga::ShaderStage::Vertex => "vs_5_1",
        naga::ShaderStage::Fragment => "ps_5_1",
        _ => panic!("unsupported shader stage"),
    };

    let bytecode = compile_hlsl(&hlsl_source, sf.entry_point, target_str, debug);
    (bytecode, wg_size)
}

// ── ShaderDevice impl ─────────────────────────────────────────────────────────

#[hidden_trait::expose]
impl crate::traits::ShaderDevice for super::Context {
    type ComputePipeline = super::ComputePipeline;
    type RenderPipeline = super::RenderPipeline;

    fn create_compute_pipeline(
        &self,
        desc: super::super::ComputePipelineDesc,
    ) -> super::ComputePipeline {
        let _num_groups = desc.data_layouts.len();
        let mut group_infos: Vec<crate::ShaderDataInfo> = desc
            .data_layouts
            .iter()
            .map(|l| l.to_info())
            .collect();

        // Preliminary group analysis (before we know access modes from shader inspection).
        // We'll re-analyze after fill_resource_bindings has been called.
        // But analyze_group needs group_infos, which aren't filled yet at this point.
        // So we do a two-phase approach: call fill_resource_bindings to get access modes,
        // then rebuild the GroupDescriptors.

        // Phase 1: fill bindings to get access modes
        let ep_index = desc.compute.entry_point_index();
        let ep_info = desc.compute.shader.info.get_entry_point(ep_index);
        let (mut module_temp, _) = desc.compute.shader.resolve_constants(&desc.compute.constants);
        crate::Shader::fill_resource_bindings(
            &mut module_temp,
            &mut group_infos,
            naga::ShaderStage::Compute,
            ep_info,
            desc.data_layouts,
        );

        // Phase 2: build group descriptors now that we have access modes
        let mut root_base = 0u32;
        let groups: Vec<GroupDescriptors> = desc.data_layouts
            .iter()
            .enumerate()
            .map(|(g, layout)| {
                super::analyze_group(layout, &group_infos[g], g as u32, &mut root_base)
            })
            .collect();

        // Phase 3: compile the shader with the correct binding map
        let debug = self.validation_mode;

        // Re-init group_infos for actual compilation
        let mut group_infos2: Vec<crate::ShaderDataInfo> = desc
            .data_layouts
            .iter()
            .map(|l| l.to_info())
            .collect();

        let (bytecode, wg_size) = compile_shader_function(
            desc.compute,
            naga::ShaderStage::Compute,
            desc.data_layouts,
            &mut group_infos2,
            &[],
            &groups,
            debug,
        );

        let root_signature = build_root_signature(&self.device, &groups);

        let pso: ID3D12PipelineState = unsafe {
            self.device.CreateComputePipelineState(
                &D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::ManuallyDrop::new(Some(root_signature.clone())),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: bytecode.as_ptr() as *const _,
                        BytecodeLength: bytecode.len(),
                    },
                    NodeMask: 0,
                    CachedPSO: D3D12_CACHED_PIPELINE_STATE::default(),
                    Flags: D3D12_PIPELINE_STATE_FLAG_NONE,
                },
            ).unwrap()
        };

        super::ComputePipeline {
            pso,
            layout: PipelineLayout { root_signature, groups },
            wg_size,
        }
    }

    fn destroy_compute_pipeline(&self, _pipeline: &mut super::ComputePipeline) {}

    fn create_render_pipeline(
        &self,
        desc: super::super::RenderPipelineDesc,
    ) -> super::RenderPipeline {
        let _num_groups = desc.data_layouts.len();
        let mut group_infos_vs: Vec<crate::ShaderDataInfo> = desc
            .data_layouts
            .iter()
            .map(|l| l.to_info())
            .collect();
        let mut group_infos_fs: Vec<crate::ShaderDataInfo> = desc
            .data_layouts
            .iter()
            .map(|l| l.to_info())
            .collect();

        // Fill resource bindings for vertex shader
        let vs_ep_index = desc.vertex.entry_point_index();
        let vs_ep_info = desc.vertex.shader.info.get_entry_point(vs_ep_index);
        let (mut vs_module, _) = desc.vertex.shader.resolve_constants(&desc.vertex.constants);
        crate::Shader::fill_resource_bindings(
            &mut vs_module,
            &mut group_infos_vs,
            naga::ShaderStage::Vertex,
            vs_ep_info,
            desc.data_layouts,
        );

        // Fill resource bindings for fragment shader
        if let Some(fs) = desc.fragment {
            let fs_ep_index = fs.entry_point_index();
            let fs_ep_info = fs.shader.info.get_entry_point(fs_ep_index);
            let (mut fs_module, _) = fs.shader.resolve_constants(&fs.constants);
            crate::Shader::fill_resource_bindings(
                &mut fs_module,
                &mut group_infos_fs,
                naga::ShaderStage::Fragment,
                fs_ep_info,
                desc.data_layouts,
            );
        }

        // Merge access info (union of vs + fs accesses)
        let mut merged_infos: Vec<crate::ShaderDataInfo> = desc
            .data_layouts
            .iter()
            .enumerate()
            .map(|(g, l)| {
                let mut info = l.to_info();
                for (bi, access) in info.binding_access.iter_mut().enumerate() {
                    *access |= group_infos_vs[g].binding_access[bi];
                    *access |= group_infos_fs[g].binding_access[bi];
                }
                info.visibility = group_infos_vs[g].visibility | group_infos_fs[g].visibility;
                info
            })
            .collect();

        let mut root_base = 0u32;
        let groups: Vec<GroupDescriptors> = desc.data_layouts
            .iter()
            .enumerate()
            .map(|(g, layout)| {
                super::analyze_group(layout, &merged_infos[g], g as u32, &mut root_base)
            })
            .collect();

        let debug = self.validation_mode;

        // Compile vertex shader
        let mut group_infos_vs2: Vec<crate::ShaderDataInfo> = desc
            .data_layouts
            .iter()
            .map(|l| l.to_info())
            .collect();
        let (vs_bytecode, _) = compile_shader_function(
            desc.vertex,
            naga::ShaderStage::Vertex,
            desc.data_layouts,
            &mut group_infos_vs2,
            desc.vertex_fetches,
            &groups,
            debug,
        );

        // Compile fragment shader
        let fs_bytecode = if let Some(fs) = desc.fragment {
            let mut group_infos_fs2: Vec<crate::ShaderDataInfo> = desc
                .data_layouts
                .iter()
                .map(|l| l.to_info())
                .collect();
            let (bytecode, _) = compile_shader_function(
                fs,
                naga::ShaderStage::Fragment,
                desc.data_layouts,
                &mut group_infos_fs2,
                &[],
                &groups,
                debug,
            );
            bytecode
        } else {
            Vec::new()
        };

        let root_signature = build_root_signature(&self.device, &groups);

        let topology = map_topology(desc.primitive.topology);
        let topology_type = match desc.primitive.topology {
            crate::PrimitiveTopology::PointList => D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT,
            crate::PrimitiveTopology::LineList | crate::PrimitiveTopology::LineStrip =>
                D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE,
            crate::PrimitiveTopology::TriangleList | crate::PrimitiveTopology::TriangleStrip =>
                D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
        };

        let raster = D3D12_RASTERIZER_DESC {
            FillMode: if desc.primitive.wireframe {
                D3D12_FILL_MODE_WIREFRAME
            } else {
                D3D12_FILL_MODE_SOLID
            },
            CullMode: match desc.primitive.cull_mode {
                None => D3D12_CULL_MODE_NONE,
                Some(crate::Face::Front) => D3D12_CULL_MODE_FRONT,
                Some(crate::Face::Back) => D3D12_CULL_MODE_BACK,
            },
            FrontCounterClockwise: (desc.primitive.front_face == crate::FrontFace::Ccw).into(),
            DepthBias: 0,
            DepthBiasClamp: 0.0,
            SlopeScaledDepthBias: 0.0,
            DepthClipEnable: (!desc.primitive.unclipped_depth).into(),
            MultisampleEnable: false.into(),
            AntialiasedLineEnable: false.into(),
            ForcedSampleCount: 0,
            ConservativeRaster: D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF,
        };

        let depth_stencil = build_depth_stencil(&desc.depth_stencil);

        let blend = build_blend_state(desc.color_targets);

        let sample_mask = desc.multisample_state.sample_mask as u32;

        let color_formats: Vec<DXGI_FORMAT> = desc.color_targets
            .iter()
            .map(|ct| super::map_texture_format(ct.format))
            .collect();
        let depth_format = desc.depth_stencil
            .as_ref()
            .map_or(DXGI_FORMAT_UNKNOWN, |ds| super::map_texture_format(ds.format));

        // Input layout from vertex fetch states
        let (input_elements, element_strs) =
            build_input_layout(desc.vertex_fetches, desc.vertex);

        let mut rtv_formats = [DXGI_FORMAT_UNKNOWN; 8];
        for (i, fmt) in color_formats.iter().enumerate() {
            rtv_formats[i] = *fmt;
        }

        let pso_desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::ManuallyDrop::new(Some(root_signature.clone())),
            VS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: vs_bytecode.as_ptr() as *const _,
                BytecodeLength: vs_bytecode.len(),
            },
            PS: if !fs_bytecode.is_empty() {
                D3D12_SHADER_BYTECODE {
                    pShaderBytecode: fs_bytecode.as_ptr() as *const _,
                    BytecodeLength: fs_bytecode.len(),
                }
            } else {
                D3D12_SHADER_BYTECODE::default()
            },
            BlendState: blend,
            SampleMask: sample_mask,
            RasterizerState: raster,
            DepthStencilState: depth_stencil,
            InputLayout: D3D12_INPUT_LAYOUT_DESC {
                pInputElementDescs: if input_elements.is_empty() {
                    std::ptr::null()
                } else {
                    input_elements.as_ptr()
                },
                NumElements: input_elements.len() as u32,
            },
            PrimitiveTopologyType: topology_type,
            NumRenderTargets: color_formats.len() as u32,
            RTVFormats: rtv_formats,
            DSVFormat: depth_format,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: desc.multisample_state.sample_count,
                Quality: 0,
            },
            ..Default::default()
        };

        let pso: ID3D12PipelineState = unsafe {
            self.device.CreateGraphicsPipelineState(&pso_desc).unwrap()
        };

        let vertex_strides: Vec<u32> = desc.vertex_fetches
            .iter()
            .map(|vf| vf.layout.stride)
            .collect();

        super::RenderPipeline {
            pso,
            layout: PipelineLayout { root_signature, groups },
            topology,
            vertex_strides,
        }
    }

    fn destroy_render_pipeline(&self, _pipeline: &mut super::RenderPipeline) {}
}

// ── Blend / depth / input layout helpers ─────────────────────────────────────

fn build_blend_state(color_targets: &[crate::ColorTargetState]) -> D3D12_BLEND_DESC {
    let mut rt_blend = [D3D12_RENDER_TARGET_BLEND_DESC::default(); 8];
    for (i, ct) in color_targets.iter().enumerate() {
        rt_blend[i] = if let Some(blend) = ct.blend {
            D3D12_RENDER_TARGET_BLEND_DESC {
                BlendEnable: true.into(),
                LogicOpEnable: false.into(),
                SrcBlend: map_blend_factor(blend.color.src_factor),
                DestBlend: map_blend_factor(blend.color.dst_factor),
                BlendOp: map_blend_op(blend.color.operation),
                SrcBlendAlpha: map_blend_factor(blend.alpha.src_factor),
                DestBlendAlpha: map_blend_factor(blend.alpha.dst_factor),
                BlendOpAlpha: map_blend_op(blend.alpha.operation),
                LogicOp: D3D12_LOGIC_OP_NOOP,
                RenderTargetWriteMask: map_color_writes(ct.write_mask) as u8,
            }
        } else {
            D3D12_RENDER_TARGET_BLEND_DESC {
                BlendEnable: false.into(),
                RenderTargetWriteMask: map_color_writes(ct.write_mask) as u8,
                ..Default::default()
            }
        };
    }
    D3D12_BLEND_DESC {
        AlphaToCoverageEnable: false.into(),
        IndependentBlendEnable: (color_targets.len() > 1).into(),
        RenderTarget: rt_blend,
    }
}

fn build_depth_stencil(ds: &Option<crate::DepthStencilState>) -> D3D12_DEPTH_STENCIL_DESC {
    match ds {
        None => D3D12_DEPTH_STENCIL_DESC::default(),
        Some(ds) => {
            let aspects = ds.format.aspects();
            D3D12_DEPTH_STENCIL_DESC {
                DepthEnable: aspects.contains(crate::TexelAspects::DEPTH).into(),
                DepthWriteMask: if ds.depth_write_enabled {
                    D3D12_DEPTH_WRITE_MASK_ALL
                } else {
                    D3D12_DEPTH_WRITE_MASK_ZERO
                },
                DepthFunc: super::map_comparison(ds.depth_compare),
                StencilEnable: aspects.contains(crate::TexelAspects::STENCIL).into(),
                StencilReadMask: (ds.stencil.read_mask & 0xFF) as u8,
                StencilWriteMask: (ds.stencil.write_mask & 0xFF) as u8,
                FrontFace: map_stencil_op_desc(&ds.stencil.front),
                BackFace: map_stencil_op_desc(&ds.stencil.back),
            }
        }
    }
}

fn map_stencil_op_desc(face: &crate::StencilFaceState) -> D3D12_DEPTH_STENCILOP_DESC {
    D3D12_DEPTH_STENCILOP_DESC {
        StencilFailOp: map_stencil_op(face.fail_op),
        StencilDepthFailOp: map_stencil_op(face.depth_fail_op),
        StencilPassOp: map_stencil_op(face.pass_op),
        StencilFunc: super::map_comparison(face.compare),
    }
}

fn map_stencil_op(op: crate::StencilOperation) -> D3D12_STENCIL_OP {
    match op {
        crate::StencilOperation::Keep => D3D12_STENCIL_OP_KEEP,
        crate::StencilOperation::Zero => D3D12_STENCIL_OP_ZERO,
        crate::StencilOperation::Replace => D3D12_STENCIL_OP_REPLACE,
        crate::StencilOperation::Invert => D3D12_STENCIL_OP_INVERT,
        crate::StencilOperation::IncrementClamp => D3D12_STENCIL_OP_INCR_SAT,
        crate::StencilOperation::DecrementClamp => D3D12_STENCIL_OP_DECR_SAT,
        crate::StencilOperation::IncrementWrap => D3D12_STENCIL_OP_INCR,
        crate::StencilOperation::DecrementWrap => D3D12_STENCIL_OP_DECR,
    }
}

fn map_blend_factor(f: crate::BlendFactor) -> D3D12_BLEND {
    match f {
        crate::BlendFactor::Zero => D3D12_BLEND_ZERO,
        crate::BlendFactor::One => D3D12_BLEND_ONE,
        crate::BlendFactor::Src => D3D12_BLEND_SRC_COLOR,
        crate::BlendFactor::OneMinusSrc => D3D12_BLEND_INV_SRC_COLOR,
        crate::BlendFactor::SrcAlpha => D3D12_BLEND_SRC_ALPHA,
        crate::BlendFactor::OneMinusSrcAlpha => D3D12_BLEND_INV_SRC_ALPHA,
        crate::BlendFactor::Dst => D3D12_BLEND_DEST_COLOR,
        crate::BlendFactor::OneMinusDst => D3D12_BLEND_INV_DEST_COLOR,
        crate::BlendFactor::DstAlpha => D3D12_BLEND_DEST_ALPHA,
        crate::BlendFactor::OneMinusDstAlpha => D3D12_BLEND_INV_DEST_ALPHA,
        crate::BlendFactor::SrcAlphaSaturated => D3D12_BLEND_SRC_ALPHA_SAT,
        crate::BlendFactor::Constant => D3D12_BLEND_BLEND_FACTOR,
        crate::BlendFactor::OneMinusConstant => D3D12_BLEND_INV_BLEND_FACTOR,
    }
}

fn map_blend_op(op: crate::BlendOperation) -> D3D12_BLEND_OP {
    match op {
        crate::BlendOperation::Add => D3D12_BLEND_OP_ADD,
        crate::BlendOperation::Subtract => D3D12_BLEND_OP_SUBTRACT,
        crate::BlendOperation::ReverseSubtract => D3D12_BLEND_OP_REV_SUBTRACT,
        crate::BlendOperation::Min => D3D12_BLEND_OP_MIN,
        crate::BlendOperation::Max => D3D12_BLEND_OP_MAX,
    }
}

fn map_color_writes(mask: crate::ColorWrites) -> u32 {
    let mut out = 0u32;
    if mask.contains(crate::ColorWrites::RED) { out |= D3D12_COLOR_WRITE_ENABLE_RED.0 as u32; }
    if mask.contains(crate::ColorWrites::GREEN) { out |= D3D12_COLOR_WRITE_ENABLE_GREEN.0 as u32; }
    if mask.contains(crate::ColorWrites::BLUE) { out |= D3D12_COLOR_WRITE_ENABLE_BLUE.0 as u32; }
    if mask.contains(crate::ColorWrites::ALPHA) { out |= D3D12_COLOR_WRITE_ENABLE_ALPHA.0 as u32; }
    out
}

fn map_topology(t: crate::PrimitiveTopology) -> D3D12_PRIMITIVE_TOPOLOGY {
    match t {
        crate::PrimitiveTopology::PointList => D3D_PRIMITIVE_TOPOLOGY_POINTLIST,
        crate::PrimitiveTopology::LineList => D3D_PRIMITIVE_TOPOLOGY_LINELIST,
        crate::PrimitiveTopology::LineStrip => D3D_PRIMITIVE_TOPOLOGY_LINESTRIP,
        crate::PrimitiveTopology::TriangleList => D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
        crate::PrimitiveTopology::TriangleStrip => D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP,
    }
}

fn map_vertex_format(vf: crate::VertexFormat) -> DXGI_FORMAT {
    match vf {
        crate::VertexFormat::F32 => DXGI_FORMAT_R32_FLOAT,
        crate::VertexFormat::F32Vec2 => DXGI_FORMAT_R32G32_FLOAT,
        crate::VertexFormat::F32Vec3 => DXGI_FORMAT_R32G32B32_FLOAT,
        crate::VertexFormat::F32Vec4 => DXGI_FORMAT_R32G32B32A32_FLOAT,
        crate::VertexFormat::U32 => DXGI_FORMAT_R32_UINT,
        crate::VertexFormat::U32Vec2 => DXGI_FORMAT_R32G32_UINT,
        crate::VertexFormat::U32Vec3 => DXGI_FORMAT_R32G32B32_UINT,
        crate::VertexFormat::U32Vec4 => DXGI_FORMAT_R32G32B32A32_UINT,
        crate::VertexFormat::I32 => DXGI_FORMAT_R32_SINT,
        crate::VertexFormat::I32Vec2 => DXGI_FORMAT_R32G32_SINT,
        crate::VertexFormat::I32Vec3 => DXGI_FORMAT_R32G32B32_SINT,
        crate::VertexFormat::I32Vec4 => DXGI_FORMAT_R32G32B32A32_SINT,
    }
}

/// Returns (input_element_descs, semantic_name_strings).
/// The strings must outlive the descs, so we return them together.
fn build_input_layout(
    vertex_fetches: &[crate::VertexFetchState],
    vs_sf: crate::ShaderFunction,
) -> (Vec<D3D12_INPUT_ELEMENT_DESC>, Vec<std::ffi::CString>) {
    let mut elements: Vec<D3D12_INPUT_ELEMENT_DESC> = Vec::new();
    let mut strings: Vec<std::ffi::CString> = Vec::new();

    // Location index counter (matches what fill_vertex_locations assigned)
    let mut location = 0u32;

    for (buf_index, fetch) in vertex_fetches.iter().enumerate() {
        for (attr_index, (name, attr)) in fetch.layout.attributes.iter().enumerate() {
            let semantic = std::ffi::CString::new(format!("TEXCOORD{location}")).unwrap();
            elements.push(D3D12_INPUT_ELEMENT_DESC {
                SemanticName: PCSTR(semantic.as_ptr() as _),
                SemanticIndex: 0,
                Format: map_vertex_format(attr.format),
                InputSlot: buf_index as u32,
                AlignedByteOffset: attr.offset,
                InputSlotClass: if fetch.instanced {
                    D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA
                } else {
                    D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA
                },
                InstanceDataStepRate: if fetch.instanced { 1 } else { 0 },
            });
            strings.push(semantic);
            location += 1;
        }
    }

    (elements, strings)
}
