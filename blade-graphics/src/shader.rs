use once_cell::sync::Lazy;
use std::borrow::Cow;

impl From<naga::ShaderStage> for super::ShaderVisibility {
    fn from(stage: naga::ShaderStage) -> Self {
        match stage {
            naga::ShaderStage::Compute => Self::COMPUTE,
            naga::ShaderStage::Vertex => Self::VERTEX,
            naga::ShaderStage::Fragment => Self::FRAGMENT,

            _ => Self::empty(),
        }
    }
}

impl super::Context {
    pub fn try_create_shader(
        &self,
        desc: super::ShaderDesc,
    ) -> Result<super::Shader, &'static str> {
        let module = naga::front::wgsl::parse_str(desc.source).map_err(|e| {
            let err = e.emit_to_string_with_path(desc.source, "");
            eprintln!("{err}");
            "compilation failed"
        })?;

        let device_caps = self.capabilities();

        // Bindings are set up at pipeline creation, ignore here
        let flags = naga::valid::ValidationFlags::all() ^ naga::valid::ValidationFlags::BINDINGS;
        // let mut caps = naga::valid::Capabilities::empty();
        let mut caps = naga::valid::Capabilities::all();
        caps.set(
            naga::valid::Capabilities::RAY_QUERY
                | naga::valid::Capabilities::TEXTURE_AND_SAMPLER_BINDING_ARRAY_NON_UNIFORM_INDEXING
                | naga::valid::Capabilities::STORAGE_TEXTURE_BINDING_ARRAY_NON_UNIFORM_INDEXING
                | naga::valid::Capabilities::STORAGE_BUFFER_BINDING_ARRAY_NON_UNIFORM_INDEXING,
            !device_caps.ray_query.is_empty(),
        );
        let info = naga::valid::Validator::new(flags, caps)
            .validate(&module)
            .map_err(|e| {
                crate::util::emit_annotated_error(&e, "", desc.source);
                crate::util::print_err(&e);
                "validation failed"
            })?;

        Ok(super::Shader {
            module,
            info,
            source: desc.source.to_owned(),
        })
    }

    pub fn create_shader(&self, desc: super::ShaderDesc) -> super::Shader {
        self.try_create_shader(desc).unwrap()
    }
}

pub static EMPTY_CONSTANTS: Lazy<super::PipelineConstants> = Lazy::new(Default::default);

impl super::Shader {
    pub fn at<'a>(&'a self, entry_point: &'a str) -> super::ShaderFunction<'a> {
        super::ShaderFunction {
            shader: self,
            entry_point,
            constants: Lazy::force(&EMPTY_CONSTANTS),
        }
    }

    pub fn with_constants<'a>(
        &'a self,
        entry_point: &'a str,
        constants: &'a super::PipelineConstants,
    ) -> super::ShaderFunction<'a> {
        super::ShaderFunction {
            shader: self,
            entry_point,
            constants,
        }
    }

    pub fn resolve_constants<'a>(
        &'a self,
        constants: &super::PipelineConstants,
    ) -> (naga::Module, Cow<'a, naga::valid::ModuleInfo>) {
        let (module, info) = naga::back::pipeline_constants::process_overrides(
            &self.module,
            &self.info,
            None,
            constants,
        )
        .unwrap();
        (module.into_owned(), info)
    }

    pub fn get_struct_size(&self, struct_name: &str) -> u32 {
        match self
            .module
            .types
            .iter()
            .find(|&(_, ty)| ty.name.as_deref() == Some(struct_name))
        {
            Some((_, ty)) => match ty.inner {
                naga::TypeInner::Struct { members: _, span } => span,
                _ => panic!("Type '{struct_name}' is not a struct in the shader"),
            },
            None => panic!("Struct '{struct_name}' is not found in the shader"),
        }
    }

    pub fn check_struct_size<T>(&self) {
        use std::{any::type_name, mem::size_of};
        let name = type_name::<T>().rsplit("::").next().unwrap();
        assert_eq!(
            size_of::<T>(),
            self.get_struct_size(name) as usize,
            "Host struct '{name}' size doesn't match the shader"
        );
    }

    pub(crate) fn fill_resource_bindings(
        module: &mut naga::Module,
        sd_infos: &mut [crate::ShaderDataInfo],
        naga_stage: naga::ShaderStage,
        ep_info: &naga::valid::FunctionInfo,
        group_layouts: &[&crate::ShaderDataLayout],
    ) {
        let mut layouter = naga::proc::Layouter::default();
        layouter.update(module.to_ctx()).unwrap();

        for (handle, var) in module.global_variables.iter_mut() {
            if ep_info[handle].is_empty() {
                continue;
            }
            let var_access = match var.space {
                naga::AddressSpace::Storage { access } => access,
                naga::AddressSpace::Uniform | naga::AddressSpace::Handle => {
                    naga::StorageAccess::empty()
                }
                _ => continue,
            };

            assert_eq!(var.binding, None);
            let var_name = var.name.as_ref().unwrap();
            for (group_index, (&layout, info)) in
                group_layouts.iter().zip(sd_infos.iter_mut()).enumerate()
            {
                if let Some((binding_index, &(_, proto_binding))) = layout
                    .bindings
                    .iter()
                    .enumerate()
                    .find(|&(_, &(name, _))| name == var_name)
                {
                    let (expected_proto, access) = match module.types[var.ty].inner {
                        naga::TypeInner::Image {
                            class: naga::ImageClass::Storage { access, format: _ },
                            ..
                        } => (crate::ShaderBinding::Texture, access),
                        naga::TypeInner::Image { .. } => {
                            (crate::ShaderBinding::Texture, naga::StorageAccess::empty())
                        }
                        naga::TypeInner::Sampler { .. } => {
                            (crate::ShaderBinding::Sampler, naga::StorageAccess::empty())
                        }
                        naga::TypeInner::AccelerationStructure { .. } => (
                            crate::ShaderBinding::AccelerationStructure,
                            naga::StorageAccess::empty(),
                        ),
                        naga::TypeInner::BindingArray { base, size: _ } => {
                            //Note: we could extract the count from `size` for more rigor
                            let count = match proto_binding {
                                crate::ShaderBinding::TextureArray { count } => count,
                                crate::ShaderBinding::BufferArray { count } => count,
                                _ => 0,
                            };
                            let proto = match module.types[base].inner {
                                naga::TypeInner::Image { .. } => {
                                    crate::ShaderBinding::TextureArray { count }
                                }
                                naga::TypeInner::Struct { .. } => {
                                    crate::ShaderBinding::BufferArray { count }
                                }
                                ref other => panic!("Unsupported binding array for {:?}", other),
                            };
                            (proto, var_access)
                        }
                        _ => {
                            let type_layout = &layouter[var.ty];
                            let proto = if var_access.is_empty()
                                && proto_binding != crate::ShaderBinding::Buffer
                            {
                                crate::ShaderBinding::Plain {
                                    size: type_layout.size,
                                }
                            } else {
                                crate::ShaderBinding::Buffer
                            };
                            (proto, var_access)
                        }
                    };
                    assert_eq!(
                        proto_binding, expected_proto,
                        "Mismatched type for binding '{}'",
                        var_name
                    );
                    assert_eq!(var.binding, None);
                    var.binding = Some(naga::ResourceBinding {
                        group: group_index as u32,
                        binding: binding_index as u32,
                    });
                    info.visibility |= naga_stage.into();
                    info.binding_access[binding_index] |= access;
                    break;
                }
            }

            assert!(
                var.binding.is_some(),
                "Unable to resolve binding for '{}' in stage '{:?}'",
                var_name,
                naga_stage,
            );
        }
    }

    pub(crate) fn fill_vertex_locations(
        module: &mut naga::Module,
        selected_ep_index: usize,
        fetch_states: &[crate::VertexFetchState],
    ) -> Vec<crate::VertexAttributeMapping> {
        let mut attribute_mappings = Vec::new();
        for (ep_index, ep) in module.entry_points.iter().enumerate() {
            let mut location = 0;
            if ep.stage != naga::ShaderStage::Vertex {
                continue;
            }

            for argument in ep.function.arguments.iter() {
                if argument.binding.is_some() {
                    continue;
                }

                let arg_name = match argument.name {
                    Some(ref name) => name.as_str(),
                    None => "?",
                };
                let mut ty = module.types[argument.ty].clone();
                let members = match ty.inner {
                    naga::TypeInner::Struct {
                        ref mut members, ..
                    } => members,
                    ref other => {
                        log::error!("Unexpected type for '{}': {:?}", arg_name, other);
                        continue;
                    }
                };

                if ep_index == selected_ep_index {
                    log::debug!("Processing vertex argument: {}", arg_name);

                    'member: for member in members.iter_mut() {
                        let member_name = match member.name {
                            Some(ref name) => name.as_str(),
                            None => "?",
                        };
                        if let Some(ref binding) = member.binding {
                            log::warn!(
                                "Member '{}' alread has binding: {:?}",
                                member_name,
                                binding
                            );
                            continue;
                        }

                        let binding = naga::Binding::Location {
                            location: attribute_mappings.len() as u32,
                            blend_src: None,
                            interpolation: None,
                            sampling: None,
                            per_primitive: false,
                        };

                        for (buffer_index, vertex_fetch) in fetch_states.iter().enumerate() {
                            for (attribute_index, &(at_name, _)) in
                                vertex_fetch.layout.attributes.iter().enumerate()
                            {
                                if at_name == member_name {
                                    log::debug!("Assigning location({}) for member '{}' to be using input {}:{}",
                                        attribute_mappings.len(), member_name, buffer_index, attribute_index);
                                    member.binding = Some(binding);
                                    attribute_mappings.push(crate::VertexAttributeMapping {
                                        buffer_index,
                                        attribute_index,
                                    });
                                    continue 'member;
                                }
                            }
                        }
                        assert_ne!(
                            member.binding, None,
                            "Field {} is not covered by the vertex fetch layouts!",
                            member_name
                        );
                    }
                } else {
                    // Just fill out the locations for the module to be valid
                    for member in members.iter_mut() {
                        if member.binding.is_none() {
                            member.binding = Some(naga::Binding::Location {
                                location,
                                interpolation: None,
                                sampling: None,
                                blend_src: None,
                                per_primitive: false,
                            });
                            location += 1;
                        }
                    }
                }

                module.types.replace(argument.ty, ty);
            }
        }
        attribute_mappings
    }

    /// Prepare a single entry point's module for backend code generation.
    ///
    /// Shared by all backends: resolves pipeline-override constants, assigns
    /// resource bindings and vertex input locations, then prunes the module to
    /// just the requested entry point and compacts it — dropping every function,
    /// type, and global that is unreachable from that entry point.
    ///
    /// Compaction matters because some backends emit the *entire* module rather
    /// than only what the entry point reaches. naga's HLSL backend in particular
    /// emits every function and panics (`write_array_size` hits `unreachable!()`)
    /// on constructs that only appear in otherwise-dead functions; pruning them
    /// here, in the common path, sidesteps that and also shrinks the SPIR-V/MSL
    /// output. Doing it once here keeps behavior identical across backends rather
    /// than adding per-backend special cases.
    ///
    /// Returns the prepared module, a freshly validated `ModuleInfo` matching it
    /// (compaction remaps handles, invalidating the original), and the vertex
    /// attribute mappings.
    pub(crate) fn prepare(
        sf: crate::ShaderFunction,
        group_infos: &mut [crate::ShaderDataInfo],
        group_layouts: &[&crate::ShaderDataLayout],
        vertex_fetch_states: &[crate::VertexFetchState],
        force_storage_read_write: bool,
    ) -> (
        naga::Module,
        naga::valid::ModuleInfo,
        Vec<crate::VertexAttributeMapping>,
    ) {
        profiling::function_scope!();

        let ep_index = sf.entry_point_index();
        let ep_stage = sf.shader.module.entry_points[ep_index].stage;
        let ep_info = sf.shader.info.get_entry_point(ep_index);

        let (mut module, _) = sf.shader.resolve_constants(sf.constants);

        // DX12 only: promote every storage buffer to read_write so naga emits it
        // as an RWByteAddressBuffer (UAV) rather than a read-only ByteAddressBuffer
        // (SRV). blade sub-allocates many logical buffers (e.g. a cull dispatch's
        // visibility input and its draw-command output) from one belt buffer; on
        // Vulkan they share a GENERAL-layout VkBuffer and can be read and written
        // freely, but a single D3D12 resource has ONE state and cannot be both
        // SHADER_RESOURCE (for an SRV binding) and UNORDERED_ACCESS (for a UAV
        // binding) within one dispatch. Binding all storage as UAV keeps the
        // resource uniformly in UNORDERED_ACCESS — the DX12 analog of blade's
        // existing "every buffer gets a UAV" model. Done before
        // fill_resource_bindings so binding_access (which classify_binding reads)
        // and the emitted HLSL agree. Storage *textures* are AddressSpace::Handle,
        // not Storage, so they are unaffected.
        if force_storage_read_write {
            for (_, var) in module.global_variables.iter_mut() {
                if let naga::AddressSpace::Storage { ref mut access } = var.space {
                    *access |= naga::StorageAccess::LOAD | naga::StorageAccess::STORE;
                }
            }
        }

        Self::fill_resource_bindings(&mut module, group_infos, ep_stage, ep_info, group_layouts);
        let attribute_mappings =
            Self::fill_vertex_locations(&mut module, ep_index, vertex_fetch_states);

        // Keep only the target entry point so compaction can drop functions/types/
        // globals reachable only from the others.
        module
            .entry_points
            .retain(|e| e.name.as_str() == sf.entry_point);

        {
            profiling::scope!("compact shader, dead code elim");
            naga::compact::compact(&mut module, naga::compact::KeepUnused::No);
        }

        // Same flags as the initial validation in `Shader::new`: skip BINDINGS
        // (blade assigns its own) and allow all capabilities.
        let flags = naga::valid::ValidationFlags::all() ^ naga::valid::ValidationFlags::BINDINGS;
        let module_info = naga::valid::Validator::new(flags, naga::valid::Capabilities::all())
            .validate(&module)
            .expect("re-validation after shader compaction failed");

        (module, module_info, attribute_mappings)
    }
}
