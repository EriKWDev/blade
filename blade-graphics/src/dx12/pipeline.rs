use std::collections::BTreeMap;
use windows::{
    core::PCSTR,
    Win32::Graphics::{
        Direct3D::*,
        Direct3D12::*,
        Dxgi::Common::*,
    },
};

use super::{BindingKind, GroupDescriptors, PipelineLayout};

// ── Binding map construction ──────────────────────────────────────────────────

/// Build the naga HLSL binding map: each binding is re-mapped to a dense
/// per-type register within its group's register space (`space == group`).
/// For samplers, `register` is the index into that group's sampler-index
/// buffer (naga emits `nagaSamplerHeap[nagaGroupNSamplerIndexArray[register]]`).
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

/// naga HLSL sampler options: where the sampler heaps and per-group sampler
/// index buffers live. Must agree with the root signature built below.
fn make_sampler_options(
    groups: &[GroupDescriptors],
) -> (
    naga::back::hlsl::SamplerHeapBindTargets,
    std::collections::BTreeMap<naga::back::hlsl::SamplerIndexBufferKey, naga::back::hlsl::BindTarget>,
) {
    let heap = naga::back::hlsl::SamplerHeapBindTargets {
        standard_samplers: naga::back::hlsl::BindTarget {
            space: super::SAMPLER_HEAP_SPACE as u8,
            register: 0,
            binding_array_size: None,
            dynamic_storage_buffer_offsets_index: None,
            restrict_indexing: false,
        },
        comparison_samplers: naga::back::hlsl::BindTarget {
            space: (super::SAMPLER_HEAP_SPACE + 1) as u8,
            register: 0,
            binding_array_size: None,
            dynamic_storage_buffer_offsets_index: None,
            restrict_indexing: false,
        },
    };
    let mut index_map = std::collections::BTreeMap::new();
    for (g, gd) in groups.iter().enumerate() {
        if gd.sampler_count > 0 {
            index_map.insert(
                naga::back::hlsl::SamplerIndexBufferKey { group: g as u32 },
                naga::back::hlsl::BindTarget {
                    space: (super::SAMPLER_INDEX_SPACE_BASE + g as u32) as u8,
                    register: 0,
                    binding_array_size: None,
                    dynamic_storage_buffer_offsets_index: None,
                    restrict_indexing: false,
                },
            );
        }
    }
    (heap, index_map)
}

// ── Root signature construction ───────────────────────────────────────────────

fn build_root_signature(
    device: &ID3D12Device,
    groups: &[GroupDescriptors],
) -> (ID3D12RootSignature, Option<(u32, u32)>) {
    // Stable storage for descriptor ranges (params hold raw pointers into these).
    let mut all_cbv_srv_uav_ranges: Vec<Vec<D3D12_DESCRIPTOR_RANGE>> =
        groups.iter().map(|_| Vec::new()).collect();
    // Build per-group CBV/SRV/UAV ranges.
    for (g, gd) in groups.iter().enumerate() {
        if gd.cbv_srv_uav_count == 0 {
            continue;
        }
        let space = g as u32;
        let (srv_count, uav_count, cbv_count) = count_types(&gd.slots);
        let mut ranges = Vec::new();
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
        all_cbv_srv_uav_ranges[g] = ranges;
    }

    // Two SAMPLER ranges (standard + comparison heaps), both [2048] arrays.
    let any_samplers = groups.iter().any(|gd| gd.sampler_count > 0);
    let std_sampler_range = [D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER,
        NumDescriptors: super::ENCODER_SAMPLER_SIZE,
        BaseShaderRegister: 0,
        RegisterSpace: super::SAMPLER_HEAP_SPACE,
        OffsetInDescriptorsFromTableStart: 0,
    }];
    let cmp_sampler_range = [D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER,
        NumDescriptors: super::ENCODER_SAMPLER_SIZE,
        BaseShaderRegister: 0,
        RegisterSpace: super::SAMPLER_HEAP_SPACE + 1,
        OffsetInDescriptorsFromTableStart: 0,
    }];

    // Root parameters, in the exact order analyze_group assigned root indices:
    //   per group: [CBV/SRV/UAV table?] [sampler-index root SRV?]
    //   then (if any samplers): [standard sampler heap] [comparison sampler heap]
    let mut params: Vec<D3D12_ROOT_PARAMETER> = Vec::new();
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
        if gd.sampler_index_root_index.is_some() {
            // Root SRV (StructuredBuffer<uint>) at (t0, space SAMPLER_INDEX_SPACE_BASE+g).
            params.push(D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    Descriptor: D3D12_ROOT_DESCRIPTOR {
                        ShaderRegister: 0,
                        RegisterSpace: super::SAMPLER_INDEX_SPACE_BASE + g as u32,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            });
        }
    }

    let sampler_heap_roots = if any_samplers {
        let std_idx = params.len() as u32;
        params.push(D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: 1,
                    pDescriptorRanges: std_sampler_range.as_ptr(),
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        });
        let cmp_idx = params.len() as u32;
        params.push(D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: 1,
                    pDescriptorRanges: cmp_sampler_range.as_ptr(),
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        });
        Some((std_idx, cmp_idx))
    } else {
        None
    };

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
    let rs: ID3D12RootSignature = super::expect_d3d12(device, "CreateRootSignature", unsafe {
        device.CreateRootSignature(
            0,
            std::slice::from_raw_parts(
                blob.GetBufferPointer() as *const u8,
                blob.GetBufferSize(),
            ),
        )
    });
    (rs, sampler_heap_roots)
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

fn wide_nul(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

/// Drop top-level function definitions whose signature was already emitted.
///
/// naga's HLSL backend keys its image-query/sample helpers (`NagaRWDimensions2D`
/// etc.) on `ImageClass`, which for storage textures includes the format and
/// access — but the generated function *name* collapses all storage to `"RW"`.
/// Two storage textures that differ only in format/access thus emit two
/// identically-named, identical-bodied helpers. FXC tolerated that; DXC rejects
/// it ("redefinition of 'NagaRWDimensions2D'"). The duplicates are byte-identical,
/// so dropping the repeats is semantically inert.
///
/// A top-level item is a brace-balanced block (functions, `cbuffer`, `struct`)
/// or a `;`-terminated statement. Only function definitions are deduplicated, and
/// only when an identical normalized signature has already been kept — distinct
/// overloads (e.g. `naga_mod(int,int)` vs `naga_mod(uint,uint)`) have different
/// signatures and are preserved.
fn dedup_hlsl_functions(src: &str) -> String {
    let bytes = src.as_bytes();
    let n = bytes.len();
    let mut out = String::with_capacity(n);
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut item_start = 0usize;
    let mut i = 0usize;
    while i < n {
        match bytes[i] {
            b';' => {
                out.push_str(&src[item_start..=i]);
                item_start = i + 1;
                i += 1;
            }
            b'{' => {
                let header = &src[item_start..i];
                let mut depth = 1u32;
                let mut j = i + 1;
                while j < n && depth != 0 {
                    match bytes[j] {
                        b'{' => depth += 1,
                        b'}' => depth -= 1,
                        _ => {}
                    }
                    j += 1;
                }
                let block = &src[item_start..j];
                let trimmed = header.trim_start();
                let is_fn = header.contains(')')
                    && !trimmed.starts_with("cbuffer")
                    && !trimmed.starts_with("struct");
                if is_fn {
                    let sig: String = header.split_whitespace().collect::<Vec<_>>().join(" ");
                    if seen.insert(sig) {
                        out.push_str(block);
                    }
                } else {
                    out.push_str(block);
                }
                item_start = j;
                i = j;
            }
            _ => i += 1,
        }
    }
    out.push_str(&src[item_start..]);
    out
}

/// Read an IDxcBlob's bytes as a UTF-8 string (for the DXC error/warning buffer).
unsafe fn dxc_blob_str(blob: &windows::Win32::Graphics::Direct3D::Dxc::IDxcBlob) -> String {
    let ptr = blob.GetBufferPointer() as *const u8;
    let len = blob.GetBufferSize();
    if ptr.is_null() || len == 0 {
        return String::new();
    }
    String::from_utf8_lossy(std::slice::from_raw_parts(ptr, len)).into_owned()
}

/// Compile HLSL to DXIL with DXC (SM 6.0). FXC/SM5.1 can't compile constructs
/// this renderer relies on (texture Sample inside dynamic loops force-unroll and
/// fail X3511); DXC handles them natively. Requires dxcompiler.dll (+ dxil.dll
/// for signing) at runtime.
fn compile_hlsl(source: &str, entry: &str, target: &str, debug: bool) -> Vec<u8> {
    use windows::core::PCWSTR;
    use windows::Win32::Graphics::Direct3D::Dxc::*;

    let source = dedup_hlsl_functions(source);
    let source = source.as_str();

    unsafe {
        let utils: IDxcUtils = DxcCreateInstance(&CLSID_DxcUtils)
            .expect("DxcCreateInstance(DxcUtils) failed — is dxcompiler.dll present?");
        let compiler: IDxcCompiler = DxcCreateInstance(&CLSID_DxcCompiler)
            .expect("DxcCreateInstance(DxcCompiler) failed — is dxcompiler.dll present?");

        let source_blob = utils
            .CreateBlobFromPinned(
                source.as_ptr() as *const _,
                source.len() as u32,
                DXC_CP(65001), // UTF-8
            )
            .unwrap();

        let entry_w = wide_nul(entry);
        let target_w = wide_nul(target);
        // naga emits HLSL-2018-style code (notably vector `?:` ternaries for
        // `select`); DXC defaults to HLSL 2021, which rejects them. Pin 2018.
        // -Zi (debug info) + -Od (no opt) in validation builds; DXC defaults to
        // full optimization otherwise. Keep the DXIL validator on so dxil.dll
        // signs the output (unsigned DXIL is rejected by the runtime).
        // -Qembed_debug stores the PDB inside the shader blob — without it -Zi
        // warns "no output provided for debug" since we never emit a separate PDB.
        let hv = [wide_nul("-HV"), wide_nul("2018")];
        // naga always emits `if ((a == b))` (redundant parens) and writes vectors
        // to narrower storage targets; both are harmless codegen artifacts that
        // otherwise flood the log on every shader. Silence those categories.
        let wno = [
            wide_nul("-Wno-parentheses-equality"),
            wide_nul("-Wno-conversion"),
        ];
        let dbg = [
            wide_nul("-Zi"),
            wide_nul("-Od"),
            wide_nul("-Qembed_debug"),
        ];
        let mut args: Vec<PCWSTR> = vec![
            PCWSTR(hv[0].as_ptr()),
            PCWSTR(hv[1].as_ptr()),
            PCWSTR(wno[0].as_ptr()),
            PCWSTR(wno[1].as_ptr()),
        ];
        if debug {
            for d in &dbg {
                args.push(PCWSTR(d.as_ptr()));
            }
        }

        let result: IDxcOperationResult = compiler
            .Compile(
                &source_blob,
                PCWSTR::null(),
                PCWSTR(entry_w.as_ptr()),
                PCWSTR(target_w.as_ptr()),
                Some(&args),
                &[],
                None,
            )
            .unwrap();

        let status = result.GetStatus().unwrap();
        if status.is_err() {
            let msg = result
                .GetErrorBuffer()
                .ok()
                .map(|e| dxc_blob_str(&e))
                .unwrap_or_default();
            panic!("HLSL compile error ({target}):\n{msg}\n\nSource:\n{source}");
        }
        if let Ok(err) = result.GetErrorBuffer() {
            let msg = dxc_blob_str(&err);
            if !msg.trim().is_empty() {
                log::warn!("HLSL compile warning ({target}): {msg}");
            }
        }

        let object = result.GetResult().unwrap();
        let ptr = object.GetBufferPointer() as *const u8;
        let len = object.GetBufferSize();
        std::slice::from_raw_parts(ptr, len).to_vec()
    }
}

fn compile_shader_function(
    sf: crate::ShaderFunction,
    stage: naga::ShaderStage,
    group_layouts: &[&crate::ShaderDataLayout],
    group_infos: &mut [crate::ShaderDataInfo],
    vertex_fetch_states: &[crate::VertexFetchState],
    groups: &[GroupDescriptors],
    debug: bool,
) -> (Vec<u8>, [u32; 3], Vec<crate::VertexAttributeMapping>) {
    let ep_index = sf.entry_point_index();
    let ep = &sf.shader.module.entry_points[ep_index];

    // Compute dispatches sub-allocate read inputs and read-write outputs from the
    // same belt buffer; bind them all as UAVs so the shared D3D12 resource stays
    // uniformly in UNORDERED_ACCESS (see `prepare`). Graphics draws only read
    // storage (no in-draw writes), so they keep read-only buffers as SRVs.
    let force_storage_read_write = ep.stage == naga::ShaderStage::Compute;

    // `Shader::prepare` resolves overrides, fills bindings + vertex locations, and
    // — crucially for HLSL — prunes the module to just this entry point and
    // compacts it. naga's HLSL backend emits *every* function/global in the
    // module and unwraps each sampler binding (write_global_sampler) / panics in
    // write_array_size on constructs in otherwise-dead functions. Compaction
    // drops that unreachable code, which is why this no longer needs a bespoke
    // `assign_unused_bindings` pass for unused samplers.
    let (module, module_info, attribute_mappings) = crate::Shader::prepare(
        sf,
        group_infos,
        group_layouts,
        vertex_fetch_states,
        force_storage_read_write,
    );

    let binding_map = make_binding_map(groups);
    let (sampler_heap_target, sampler_buffer_binding_map) = make_sampler_options(groups);
    let hlsl_options = naga::back::hlsl::Options {
        shader_model: naga::back::hlsl::ShaderModel::V6_0,
        binding_map,
        fake_missing_bindings: true,
        sampler_heap_target,
        sampler_buffer_binding_map,
        // naga's loop-bounding wraps every loop in a `while(true)` guarded by a
        // ~4-billion countdown. FXC must unroll loops containing gradient ops
        // (texture Sample), and that bound makes it give up: "loop does not
        // terminate in a timely manner" (X3511). Metal disables it for the same
        // reason; do the same here.
        force_loop_bounding: false,
        ..Default::default()
    };

    let pipeline_options = naga::back::hlsl::PipelineOptions {
        entry_point: Some((stage, sf.entry_point.to_string())),
    };

    // naga's HLSL backend can `unreachable!()`/panic on constructs it can't
    // express (it does not return a Result for those). Catch it so we (a) don't
    // abort the process via a worker-thread panic across the FFI boundary, and
    // (b) report the exact entry point + WGSL source that triggered it, which is
    // otherwise invisible because the panic happens inside naga.
    let mut hlsl_source = String::new();
    let write_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut writer =
            naga::back::hlsl::Writer::new(&mut hlsl_source, &hlsl_options, &pipeline_options);
        writer.write(&module, &module_info, None)
    }));
    match write_result {
        Ok(Ok(_reflection)) => {}
        Ok(Err(e)) => panic!("naga HLSL generation failed for '{}': {e}", sf.entry_point),
        Err(_panic) => {
            // The offending construct lives in `#import`-ed / `#ifdef`-preprocessed
            // files, so `sf.shader.source` (the primary file) is not enough to
            // reproduce it. Round-trip the fully-assembled module back to WGSL via
            // naga's wgsl backend so the complete source is visible.
            let dump = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                naga::back::wgsl::write_string(
                    &module,
                    &module_info,
                    naga::back::wgsl::WriterFlags::EXPLICIT_TYPES,
                )
            }));
            let assembled = match dump {
                Ok(Ok(s)) => s,
                _ => format!(
                    "<wgsl-out also failed; falling back to primary source>\n{}",
                    sf.shader.source
                ),
            };
            panic!(
                "naga HLSL backend panicked compiling entry point '{}' ({:?}).\n\
                 This is a naga HLSL limitation for some WGSL construct. Full \
                 assembled module source follows:\n{}",
                sf.entry_point, stage, assembled
            );
        }
    }

    let wg_size = if stage == naga::ShaderStage::Compute {
        let wg = ep.workgroup_size;
        [wg[0], wg[1], wg[2]]
    } else {
        [1, 1, 1]
    };

    let target_str = match stage {
        naga::ShaderStage::Compute => "cs_6_0",
        naga::ShaderStage::Vertex => "vs_6_0",
        naga::ShaderStage::Fragment => "ps_6_0",
        _ => panic!("unsupported shader stage"),
    };

    let bytecode = compile_hlsl(&hlsl_source, sf.entry_point, target_str, debug);
    (bytecode, wg_size, attribute_mappings)
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

        // analyze_group needs the per-binding access modes, which only
        // fill_resource_bindings produces — so fill into a throwaway module first,
        // build the descriptors, then compile from a fresh group_infos.
        let ep_index = desc.compute.entry_point_index();
        let ep_info = desc.compute.shader.info.get_entry_point(ep_index);
        let (mut module_temp, _) = desc.compute.shader.resolve_constants(&desc.compute.constants);
        // Match compile_shader_function (which forces compute storage to UAV): the
        // register/space assignment derived here MUST use the same classification
        // as the emitted HLSL, or SRV/UAV registers collide (e.g. a read-only
        // `renderables` and a read-write `visible_buffer` both landing at u0).
        crate::Shader::force_storage_read_write(&mut module_temp);
        crate::Shader::fill_resource_bindings(
            &mut module_temp,
            &mut group_infos,
            naga::ShaderStage::Compute,
            ep_info,
            desc.data_layouts,
        );

        let mut root_base = 0u32;
        let groups: Vec<GroupDescriptors> = desc.data_layouts
            .iter()
            .enumerate()
            .map(|(g, layout)| {
                super::analyze_group(layout, &group_infos[g], g as u32, &mut root_base)
            })
            .collect();

        let debug = self.validation_mode;

        let mut group_infos2: Vec<crate::ShaderDataInfo> = desc
            .data_layouts
            .iter()
            .map(|l| l.to_info())
            .collect();

        let (bytecode, wg_size, _) = compile_shader_function(
            desc.compute,
            naga::ShaderStage::Compute,
            desc.data_layouts,
            &mut group_infos2,
            &[],
            &groups,
            debug,
        );

        let (root_signature, sampler_heap_roots) = build_root_signature(&self.device, &groups);

        let pso: ID3D12PipelineState = super::expect_d3d12(
            &self.device,
            "CreateComputePipelineState",
            unsafe {
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
                )
            },
        );

        super::set_resource_name(&pso, desc.name);
        super::ComputePipeline {
            pso,
            layout: PipelineLayout { root_signature, groups, sampler_heap_roots },
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

        let merged_infos: Vec<crate::ShaderDataInfo> = desc
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

        let mut group_infos_vs2: Vec<crate::ShaderDataInfo> = desc
            .data_layouts
            .iter()
            .map(|l| l.to_info())
            .collect();
        let (vs_bytecode, _, vs_attribute_mappings) = compile_shader_function(
            desc.vertex,
            naga::ShaderStage::Vertex,
            desc.data_layouts,
            &mut group_infos_vs2,
            desc.vertex_fetches,
            &groups,
            debug,
        );

        let fs_bytecode = if let Some(fs) = desc.fragment {
            let mut group_infos_fs2: Vec<crate::ShaderDataInfo> = desc
                .data_layouts
                .iter()
                .map(|l| l.to_info())
                .collect();
            let (bytecode, _, _) = compile_shader_function(
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

        let (root_signature, sampler_heap_roots) = build_root_signature(&self.device, &groups);

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

        // Input layout, keyed by the locations the shader actually got.
        let (input_elements, _element_strs) =
            build_input_layout(desc.vertex_fetches, &vs_attribute_mappings);

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

        let pso: ID3D12PipelineState = super::expect_d3d12(
            &self.device,
            "CreateGraphicsPipelineState",
            unsafe { self.device.CreateGraphicsPipelineState(&pso_desc) },
        );

        let vertex_strides: Vec<u32> = desc.vertex_fetches
            .iter()
            .map(|vf| vf.layout.stride)
            .collect();

        super::set_resource_name(&pso, desc.name);
        super::RenderPipeline {
            pso,
            layout: PipelineLayout { root_signature, groups, sampler_heap_roots },
            topology,
            vertex_strides,
        }
    }

    fn destroy_render_pipeline(&self, _pipeline: &mut super::RenderPipeline) {}
}

// ── Blend / depth / input layout helpers ─────────────────────────────────────

/// A blend-disabled render target with *valid* (non-zero) enum fields. A
/// zeroed `D3D12_RENDER_TARGET_BLEND_DESC` has `SrcBlend=0`/`LogicOp=0`, which
/// are invalid enum values the runtime rejects (E_INVALIDARG) even when blend
/// is disabled — so unused targets must use this, not `Default::default()`.
fn disabled_blend_rt(write_mask: u8) -> D3D12_RENDER_TARGET_BLEND_DESC {
    D3D12_RENDER_TARGET_BLEND_DESC {
        BlendEnable: false.into(),
        LogicOpEnable: false.into(),
        SrcBlend: D3D12_BLEND_ONE,
        DestBlend: D3D12_BLEND_ZERO,
        BlendOp: D3D12_BLEND_OP_ADD,
        SrcBlendAlpha: D3D12_BLEND_ONE,
        DestBlendAlpha: D3D12_BLEND_ZERO,
        BlendOpAlpha: D3D12_BLEND_OP_ADD,
        LogicOp: D3D12_LOGIC_OP_NOOP,
        RenderTargetWriteMask: write_mask,
    }
}

fn build_blend_state(color_targets: &[crate::ColorTargetState]) -> D3D12_BLEND_DESC {
    let mut rt_blend = [disabled_blend_rt(D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8); 8];
    for (i, ct) in color_targets.iter().enumerate() {
        let write_mask = map_color_writes(ct.write_mask) as u8;
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
                RenderTargetWriteMask: write_mask,
            }
        } else {
            disabled_blend_rt(write_mask)
        };
    }
    D3D12_BLEND_DESC {
        AlphaToCoverageEnable: false.into(),
        IndependentBlendEnable: (color_targets.len() > 1).into(),
        RenderTarget: rt_blend,
    }
}

/// Depth/stencil with depth and stencil disabled, but valid (non-zero) enum
/// fields — see `disabled_blend_rt` for the same E_INVALIDARG rationale.
fn disabled_depth_stencil() -> D3D12_DEPTH_STENCIL_DESC {
    let keep = D3D12_DEPTH_STENCILOP_DESC {
        StencilFailOp: D3D12_STENCIL_OP_KEEP,
        StencilDepthFailOp: D3D12_STENCIL_OP_KEEP,
        StencilPassOp: D3D12_STENCIL_OP_KEEP,
        StencilFunc: D3D12_COMPARISON_FUNC_ALWAYS,
    };
    D3D12_DEPTH_STENCIL_DESC {
        DepthEnable: false.into(),
        DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ZERO,
        DepthFunc: D3D12_COMPARISON_FUNC_ALWAYS,
        StencilEnable: false.into(),
        StencilReadMask: 0,
        StencilWriteMask: 0,
        FrontFace: keep,
        BackFace: keep,
    }
}

fn build_depth_stencil(ds: &Option<crate::DepthStencilState>) -> D3D12_DEPTH_STENCIL_DESC {
    match ds {
        None => disabled_depth_stencil(),
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

fn map_topology(t: crate::PrimitiveTopology) -> D3D_PRIMITIVE_TOPOLOGY {
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
///
/// naga's HLSL backend names vertex inputs `LOC{n}`, where `n` is the location
/// `fill_vertex_locations` assigned — and it assigns them in *shader member*
/// order, recording which fetch attribute feeds each one in `attribute_mappings`
/// (location == index into that list). Walking the fetch layouts and counting
/// instead only agrees when the shader happens to declare its inputs in fetch
/// order and use every attribute; otherwise the elements silently shift, so a
/// draw reads the wrong bytes for every attribute past the first mismatch — the
/// debug layer reports it only as a format/type mismatch warning. Vulkan builds
/// its attribute descriptions from the same mappings; do the same here.
fn build_input_layout(
    vertex_fetches: &[crate::VertexFetchState],
    attribute_mappings: &[crate::VertexAttributeMapping],
) -> (Vec<D3D12_INPUT_ELEMENT_DESC>, Vec<std::ffi::CString>) {
    let mut elements: Vec<D3D12_INPUT_ELEMENT_DESC> = Vec::with_capacity(attribute_mappings.len());
    let mut strings: Vec<std::ffi::CString> = Vec::with_capacity(attribute_mappings.len());

    for (location, mapping) in attribute_mappings.iter().enumerate() {
        let fetch = &vertex_fetches[mapping.buffer_index];
        let (_name, attr) = &fetch.layout.attributes[mapping.attribute_index];
        let semantic = std::ffi::CString::new("LOC").unwrap();
        elements.push(D3D12_INPUT_ELEMENT_DESC {
            SemanticName: PCSTR(semantic.as_ptr() as _),
            SemanticIndex: location as u32,
            Format: map_vertex_format(attr.format),
            InputSlot: mapping.buffer_index as u32,
            AlignedByteOffset: attr.offset,
            InputSlotClass: if fetch.instanced {
                D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA
            } else {
                D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA
            },
            InstanceDataStepRate: if fetch.instanced { 1 } else { 0 },
        });
        strings.push(semantic);
    }

    (elements, strings)
}
