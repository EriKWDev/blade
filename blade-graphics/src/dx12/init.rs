use windows::core::Interface;
use windows::Win32::{
        Foundation::HANDLE,
        Graphics::{
            Direct3D::D3D_FEATURE_LEVEL_12_0,
            Direct3D12::*,
            Dxgi::{Common::*, *},
        },
        System::Threading::CreateEventW,
    };

use super::{Context, DescriptorPool, Queue};

/// Mute the mismatched-clear-value spam: blade has no per-texture optimized
/// clear value, so every clear reports the slow path, thousands of lines a second.
unsafe fn mute_noisy_debug_messages(device: &ID3D12Device) {
    let Ok(info_queue) = device.cast::<ID3D12InfoQueue>() else {
        return;
    };
    let mut denied = [
        D3D12_MESSAGE_ID_CLEARRENDERTARGETVIEW_MISMATCHINGCLEARVALUE,
        D3D12_MESSAGE_ID_CLEARDEPTHSTENCILVIEW_MISMATCHINGCLEARVALUE,
    ];
    let mut filter = D3D12_INFO_QUEUE_FILTER {
        DenyList: D3D12_INFO_QUEUE_FILTER_DESC {
            NumIDs: denied.len() as u32,
            pIDList: denied.as_mut_ptr(),
            ..Default::default()
        },
        ..Default::default()
    };
    let _ = info_queue.PushStorageFilter(&mut filter);
}

/// Typed `CheckFeatureSupport`. False (with `data` untouched) if the runtime
/// does not know the feature.
unsafe fn check_feature<T>(device: &ID3D12Device, feature: D3D12_FEATURE, data: &mut T) -> bool {
    device
        .CheckFeatureSupport(
            feature,
            data as *mut T as *mut std::ffi::c_void,
            std::mem::size_of::<T>() as u32,
        )
        .is_ok()
}

/// Supported MSAA counts as a mask of the counts themselves (bit 4 == 4x).
/// D3D12 answers per format, so intersect a color and a depth format.
unsafe fn query_sample_count_mask(device: &ID3D12Device) -> u32 {
    let mut mask = 0u32;
    for count in [1u32, 2, 4, 8, 16, 32] {
        let supported = [DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_D32_FLOAT]
            .into_iter()
            .all(|format| {
                let mut levels = D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS {
                    Format: format,
                    SampleCount: count,
                    ..Default::default()
                };
                check_feature(
                    device,
                    D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS,
                    &mut levels,
                ) && levels.NumQualityLevels > 0
            });
        if supported {
            mask |= count;
        }
    }
    // Single-sampled is always available; never report an empty mask.
    mask | 1
}

/// VRS support in blade's shape. Tier 1 is per-draw only; tier 2 adds the
/// shading-rate image. blade's texel encoding is already `D3D12_SHADING_RATE`.
unsafe fn query_shading_rates(
    device: &ID3D12Device,
    sample_count_mask: u32,
) -> (
    D3D12_VARIABLE_SHADING_RATE_TIER,
    Vec<crate::FragmentShadingRateSupport>,
    Option<[u32; 2]>,
) {
    let mut options6 = D3D12_FEATURE_DATA_D3D12_OPTIONS6::default();
    if !check_feature(device, D3D12_FEATURE_D3D12_OPTIONS6, &mut options6) {
        return (D3D12_VARIABLE_SHADING_RATE_TIER_NOT_SUPPORTED, Vec::new(), None);
    }
    let tier = options6.VariableShadingRateTier;
    if tier.0 < D3D12_VARIABLE_SHADING_RATE_TIER_1.0 {
        return (tier, Vec::new(), None);
    }
    // The base set every VRS-capable device supports, plus the coarser sizes
    // gated behind AdditionalShadingRatesSupported.
    let mut sizes: Vec<[u8; 2]> = vec![[1, 1], [1, 2], [2, 1], [2, 2]];
    if options6.AdditionalShadingRatesSupported.as_bool() {
        sizes.extend_from_slice(&[[2, 4], [4, 2], [4, 4]]);
    }
    let rates = sizes
        .into_iter()
        .map(|[w, h]| crate::FragmentShadingRateSupport {
            rate: crate::FragmentShadingRate::new(w, h),
            sample_count_mask,
        })
        .collect();
    // The shading-rate image is tier 2 and its tile is square.
    let texel_size = (tier.0 >= D3D12_VARIABLE_SHADING_RATE_TIER_2.0).then(|| {
        let tile = options6.ShadingRateImageTileSize.max(1);
        [tile, tile]
    });
    (tier, rates, texel_size)
}

impl Context {
    pub unsafe fn init(desc: super::super::ContextDesc) -> Result<Self, super::super::NotSupportedError> {
        // GPU-based validation is enormously expensive (it patches every shader
        // memory access) and can itself push a heavy frame past the Windows TDR
        // timeout — a device "hang" that is a validation artifact, not an app bug.
        // It is on by default under validation, but can be disabled with
        // BLADE_DX12_GPU_VALIDATION=0 to keep the base debug layer + DRED while
        // isolating whether a hang is real. The base layer's API checks and DRED
        // breadcrumbs stay on regardless.
        let gpu_validation = desc.validation
            && !matches!(
                std::env::var("BLADE_DX12_GPU_VALIDATION").as_deref(),
                Ok("0") | Ok("off") | Ok("false")
            );
        if desc.validation {
            let mut debug: Option<ID3D12Debug> = None;
            if D3D12GetDebugInterface(&mut debug).is_ok() {
                if let Some(debug) = debug {
                    debug.EnableDebugLayer();
                    // GPU-based validation instruments shader execution and is the
                    // only layer that catches GPU-side hazards — descriptor-heap
                    // corruption, out-of-bounds / uninitialized reads, a descriptor
                    // pointing at the wrong resource — none of which the base debug
                    // layer (API-parameter checks only) reports.
                    if gpu_validation {
                        if let Ok(debug1) = debug.cast::<ID3D12Debug1>() {
                            debug1.SetEnableGPUBasedValidation(true);
                            debug1.SetEnableSynchronizedCommandQueueValidation(true);
                        }
                    }
                }
            }
            // DRED captures GPU auto-breadcrumbs + the page-fault VA, the only way
            // to localize a DEVICE_HUNG/DEVICE_REMOVED that the debug layer and GBV
            // do not flag with a specific message. Must be armed before device
            // creation; read back via `log_dred` on removal.
            let mut dred: Option<ID3D12DeviceRemovedExtendedDataSettings> = None;
            if D3D12GetDebugInterface(&mut dred).is_ok() {
                if let Some(dred) = dred {
                    dred.SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
                    dred.SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
                    super::dred_set_enabled();
                }
            }
        }

        let factory_flags = if desc.validation {
            DXGI_CREATE_FACTORY_DEBUG
        } else {
            DXGI_CREATE_FACTORY_FLAGS(0)
        };
        let factory: IDXGIFactory4 = CreateDXGIFactory2(factory_flags)
            .map_err(|e| super::super::NotSupportedError::Platform(e))?;

        let (adapter, adapter_desc) = pick_adapter(&factory, desc.device_id)?;
        let vendor = map_vendor(adapter_desc.VendorId);

        let device: ID3D12Device = {
            let mut dev: Option<ID3D12Device> = None;
            D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut dev)
                .map_err(|e| super::super::NotSupportedError::Platform(e))?;
            dev.unwrap()
        };

        if desc.validation {
            mute_noisy_debug_messages(&device);
        }

        if gpu_validation {
            // GBV defaults to STATE_TRACKING_ONLY; GUARDED_VALIDATION additionally
            // patches every shader resource access with a bounds guard, catching
            // out-of-bounds buffer/texture/descriptor reads at the shader level
            // (the deepest GPU-side check). Set per-device after creation.
            if let Ok(debug_device) = device.cast::<ID3D12DebugDevice1>() {
                let settings = D3D12_DEBUG_DEVICE_GPU_BASED_VALIDATION_SETTINGS {
                    MaxMessagesPerCommandList: 1024,
                    DefaultShaderPatchMode:
                        D3D12_GPU_BASED_VALIDATION_SHADER_PATCH_MODE_GUARDED_VALIDATION,
                    PipelineStateCreateFlags:
                        D3D12_GPU_BASED_VALIDATION_PIPELINE_STATE_CREATE_FLAG_NONE,
                };
                let _ = debug_device.SetDebugParameter(
                    D3D12_DEBUG_DEVICE_PARAMETER_GPU_BASED_VALIDATION_SETTINGS,
                    &settings as *const _ as *const std::ffi::c_void,
                    std::mem::size_of::<D3D12_DEBUG_DEVICE_GPU_BASED_VALIDATION_SETTINGS>() as u32,
                );
            }
        }

        // The per-encoder CBV/SRV/UAV ring is split into per-frame segments, so a
        // heavy frame can need well over the tier-1/2 cap of 1,000,000 once halved.
        // Resource-binding tier 3 lifts that cap, so use a larger heap there.
        let cbv_srv_uav_heap_size = {
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
            let tier = if check_feature(&device, D3D12_FEATURE_D3D12_OPTIONS, &mut options) {
                options.ResourceBindingTier
            } else {
                D3D12_RESOURCE_BINDING_TIER_1
            };
            // Tier 3 has no 1M cap, so use 2M → 1M per frame at buffer_count=2,
            // matching the capacity that worked single-buffered. (4M overflowed
            // descriptor memory on a laptop GPU under GPU-based validation.)
            if tier.0 >= D3D12_RESOURCE_BINDING_TIER_3.0 {
                2_000_000
            } else {
                super::ENCODER_HEAP_TIER12_MAX
            }
        };

        // Everything below was previously hard-coded; ask the device instead.
        let mut options4 = D3D12_FEATURE_DATA_D3D12_OPTIONS4::default();
        let shader_float16 = check_feature(&device, D3D12_FEATURE_D3D12_OPTIONS4, &mut options4)
            && options4.Native16BitShaderOpsSupported.as_bool();
        let sample_count_mask = query_sample_count_mask(&device);
        let (shading_rate_tier, fragment_shading_rates, fragment_shading_rate_attachment_texel_size) =
            query_shading_rates(&device, sample_count_mask);

        let capabilities = crate::Capabilities {
            // Ray queries need DXR + the acceleration-structure backend, which
            // this backend does not implement yet.
            ray_query: crate::ShaderVisibility::empty(),
            shader_float16,
            sample_count_mask,
            multidraw_indirect: true,
            draw_indexed_indirect_count: true,
            vendor,
            present_wait: false,
            multisampled_render_to_single_sampled: false,
            fragment_shading_rates,
            fragment_shading_rate_attachment_texel_size,
        };

        let queue_desc = D3D12_COMMAND_QUEUE_DESC {
            Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
            Flags: D3D12_COMMAND_QUEUE_FLAG_NONE,
            NodeMask: 0,
            Priority: 0,
        };
        let queue_raw: ID3D12CommandQueue =
            device.CreateCommandQueue(&queue_desc).map_err(|e| super::super::NotSupportedError::Platform(e))?;

        let fence: ID3D12Fence = device
            .CreateFence(0, D3D12_FENCE_FLAG_NONE)
            .map_err(|e| super::super::NotSupportedError::Platform(e))?;
        let fence_event: HANDLE = CreateEventW(None, false, false, None)
            .map_err(|e| super::super::NotSupportedError::Platform(e))?;

        // CPU-visible descriptor pools: pages are allocated lazily and freed
        // handles are recycled, so there is no fixed view/target budget.
        let rtv_pool = DescriptorPool::new(D3D12_DESCRIPTOR_HEAP_TYPE_RTV, super::RTV_PAGE_SIZE);
        let dsv_pool = DescriptorPool::new(D3D12_DESCRIPTOR_HEAP_TYPE_DSV, super::DSV_PAGE_SIZE);
        let staging_pool = DescriptorPool::new(
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            super::STAGING_PAGE_SIZE,
        );
        let staging_sampler_pool = DescriptorPool::new(
            D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
            super::STAGING_SAMPLER_PAGE_SIZE,
        );

        let draw_sig = create_command_sig(&device, D3D12_INDIRECT_ARGUMENT_TYPE_DRAW, 16);
        let draw_indexed_sig = create_command_sig(&device, D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED, 20);
        let dispatch_sig = create_command_sig(&device, D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH, 12);

        let device_name = {
            let mut raw = [0u16; 128];
            raw[..adapter_desc.Description.len()].copy_from_slice(&adapter_desc.Description);
            String::from_utf16_lossy(
                &raw[..raw.iter().position(|&c| c == 0).unwrap_or(128)],
            )
        };

        // DXGI_ADAPTER_DESC1 does not expose a driver version field.
        let driver_info = String::new();

        Ok(Context {
            device,
            queue: std::sync::Mutex::new(Queue {
                raw: queue_raw,
                fence,
                fence_event,
                last_progress: 0,
            }),
            rtv_pool: std::sync::Mutex::new(rtv_pool),
            dsv_pool: std::sync::Mutex::new(dsv_pool),
            staging_pool: std::sync::Mutex::new(staging_pool),
            staging_sampler_pool: std::sync::Mutex::new(staging_sampler_pool),
            draw_sig,
            draw_indexed_sig,
            dispatch_sig,
            validation_mode: desc.validation,
            device_information: crate::DeviceInformation {
                is_software_emulated: is_software_adapter(&adapter_desc),
                device_name,
                driver_name: "D3D12".to_string(),
                driver_info,
            },
            factory,
            cbv_srv_uav_heap_size,
            capabilities,
            shading_rate_tier,
        })
    }

    pub fn capabilities(&self) -> crate::Capabilities {
        self.capabilities.clone()
    }

    pub fn device_information(&self) -> &crate::DeviceInformation {
        &self.device_information
    }
}

fn create_command_sig(
    device: &ID3D12Device,
    ty: D3D12_INDIRECT_ARGUMENT_TYPE,
    byte_stride: u32,
) -> ID3D12CommandSignature {
    let arg_desc = D3D12_INDIRECT_ARGUMENT_DESC {
        Type: ty,
        Anonymous: Default::default(),
    };
    let sig_desc = D3D12_COMMAND_SIGNATURE_DESC {
        ByteStride: byte_stride,
        NumArgumentDescs: 1,
        pArgumentDescs: &arg_desc,
        NodeMask: 0,
    };
    let mut sig: Option<ID3D12CommandSignature> = None;
    unsafe {
        device
            .CreateCommandSignature(&sig_desc, None::<&ID3D12RootSignature>, &mut sig)
            .unwrap();
    }
    sig.unwrap()
}

/// D3D12-capable hardware adapters, best first. `EnumAdapters1` order is not
/// performance-ordered (it often yields the iGPU first), so prefer
/// `IDXGIFactory6`, falling back to a largest-VRAM sort.
unsafe fn enumerate_adapters(factory: &IDXGIFactory4) -> Vec<(IDXGIAdapter1, DXGI_ADAPTER_DESC1)> {
    let mut out: Vec<(IDXGIAdapter1, DXGI_ADAPTER_DESC1)> = Vec::new();
    let by_preference = factory.cast::<IDXGIFactory6>().ok();
    for i in 0.. {
        let adapter: IDXGIAdapter1 = match &by_preference {
            Some(f) => {
                match f.EnumAdapterByGpuPreference(i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE) {
                    Ok(a) => a,
                    Err(_) => break,
                }
            }
            None => match factory.EnumAdapters1(i) {
                Ok(a) => a,
                Err(_) => break,
            },
        };
        let Ok(desc) = adapter.GetDesc1() else { continue };
        if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0 {
            continue;
        }
        let mut test: Option<ID3D12Device> = None;
        if D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut test).is_err() {
            continue;
        }
        out.push((adapter, desc));
    }
    if by_preference.is_none() {
        // No preference order available: approximate it by dedicated video memory.
        out.sort_by_key(|(_, desc)| std::cmp::Reverse(desc.DedicatedVideoMemory));
    }
    out
}

unsafe fn pick_adapter(
    factory: &IDXGIFactory4,
    preferred_device_id: u32,
) -> Result<(IDXGIAdapter1, DXGI_ADAPTER_DESC1), super::super::NotSupportedError> {
    let mut adapters = enumerate_adapters(factory);
    if preferred_device_id != 0 {
        if let Some(i) = adapters
            .iter()
            .position(|(_, desc)| desc.DeviceId == preferred_device_id)
        {
            return Ok(adapters.swap_remove(i));
        }
        log::warn!(
            "DX12: requested device id {preferred_device_id:#x} not found; using the              highest-performance adapter instead"
        );
    }
    if adapters.is_empty() {
        return Err(super::super::NotSupportedError::NoSupportedDeviceFound);
    }
    Ok(adapters.swap_remove(0))
}

fn is_software_adapter(desc: &DXGI_ADAPTER_DESC1) -> bool {
    desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0
}

fn map_vendor(vendor_id: u32) -> crate::GpuVendor {
    match vendor_id {
        0x8086 => crate::GpuVendor::Intel,
        0x10DE => crate::GpuVendor::Nvidia,
        0x1002 => crate::GpuVendor::Amd,
        0x13B5 => crate::GpuVendor::Arm,
        0x5143 => crate::GpuVendor::Qualcomm,
        _ => crate::GpuVendor::Unknown,
    }
}
