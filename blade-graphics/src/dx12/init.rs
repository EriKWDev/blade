use windows::core::Interface;
use windows::Win32::{
        Foundation::HANDLE,
        Graphics::{
            Direct3D::D3D_FEATURE_LEVEL_12_0,
            Direct3D12::*,
            Dxgi::*,
        },
        System::Threading::CreateEventW,
    };

use super::{Context, DescriptorHeap, LinearAllocator, Queue};

impl Context {
    pub unsafe fn init(desc: super::super::ContextDesc) -> Result<Self, super::super::NotSupportedError> {
        if desc.validation {
            let mut debug: Option<ID3D12Debug> = None;
            if D3D12GetDebugInterface(&mut debug).is_ok() {
                if let Some(debug) = debug {
                    debug.EnableDebugLayer();
                    // GPU-based validation instruments shader execution and is the
                    // only layer that catches GPU-side hazards — descriptor-heap
                    // corruption, out-of-bounds / uninitialized reads, a descriptor
                    // pointing at the wrong resource — none of which the base debug
                    // layer (API-parameter checks only) reports. It is expensive but
                    // exactly what surfaces otherwise-silent intermittent glitches.
                    if let Ok(debug1) = debug.cast::<ID3D12Debug1>() {
                        debug1.SetEnableGPUBasedValidation(true);
                        debug1.SetEnableSynchronizedCommandQueueValidation(true);
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
            let tier = if device
                .CheckFeatureSupport(
                    D3D12_FEATURE_D3D12_OPTIONS,
                    &mut options as *mut _ as *mut std::ffi::c_void,
                    std::mem::size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS>() as u32,
                )
                .is_ok()
            {
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
                super::ENCODER_HEAP_SIZE
            }
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

        let rtv_heap = DescriptorHeap::new(&device, D3D12_DESCRIPTOR_HEAP_TYPE_RTV, super::RTV_HEAP_SIZE, false);
        let dsv_heap = DescriptorHeap::new(&device, D3D12_DESCRIPTOR_HEAP_TYPE_DSV, super::DSV_HEAP_SIZE, false);
        let staging_heap = DescriptorHeap::new(
            &device,
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            super::STAGING_HEAP_SIZE,
            false,
        );
        let staging_sampler_heap = DescriptorHeap::new(
            &device,
            D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
            super::STAGING_SAMPLER_SIZE,
            false,
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
            rtv_heap,
            dsv_heap,
            staging_heap,
            staging_sampler_heap,
            rtv_alloc: std::sync::Mutex::new(LinearAllocator { offset: 0 }),
            dsv_alloc: std::sync::Mutex::new(LinearAllocator { offset: 0 }),
            staging_alloc: std::sync::Mutex::new(LinearAllocator { offset: 0 }),
            staging_sampler_alloc: std::sync::Mutex::new(LinearAllocator { offset: 0 }),
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
            vendor,
            factory,
            cbv_srv_uav_heap_size,
        })
    }

    pub fn capabilities(&self) -> crate::Capabilities {
        crate::Capabilities {
            ray_query: crate::ShaderVisibility::empty(),
            sample_count_mask: 0x1 | 0x2 | 0x4 | 0x8 | 0x10 | 0x20, // 1,2,4,8,16,32
            multidraw_indirect: true,
            draw_indexed_indirect_count: true,
            vendor: self.vendor,
            present_wait: false,
        }
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

unsafe fn pick_adapter(
    factory: &IDXGIFactory4,
    preferred_device_id: u32,
) -> Result<(IDXGIAdapter1, DXGI_ADAPTER_DESC1), super::super::NotSupportedError> {
    let mut best: Option<(IDXGIAdapter1, DXGI_ADAPTER_DESC1)> = None;

    for i in 0.. {
        let adapter: IDXGIAdapter1 = match factory.EnumAdapters1(i) {
            Ok(a) => a,
            Err(_) => break,
        };
        let desc: DXGI_ADAPTER_DESC1 = match adapter.GetDesc1() {
            Ok(d) => d,
            Err(_) => continue,
        };

        if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0 {
            continue;
        }

        let mut test: Option<ID3D12Device> = None;
        if D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut test).is_err() {
            continue;
        }

        if preferred_device_id != 0 && desc.DeviceId == preferred_device_id {
            return Ok((adapter, desc));
        }

        if best.is_none() {
            best = Some((adapter, desc));
        }
    }

    best.ok_or(super::super::NotSupportedError::NoSupportedDeviceFound)
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
