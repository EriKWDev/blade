use windows::{
    core::Interface,
    Win32::{
        Foundation::HANDLE,
        Graphics::{
            Direct3D::D3D_FEATURE_LEVEL_12_0,
            Direct3D12::*,
            Dxgi::*,
        },
        System::Threading::CreateEventW,
    },
};

use super::{Context, DescriptorHeap, LinearAllocator, Queue};

impl Context {
    pub unsafe fn init(desc: super::super::ContextDesc) -> Result<Self, super::super::NotSupportedError> {
        // Enable debug layer in validation mode
        if desc.validation {
            if let Ok(debug) = D3D12GetDebugInterface::<ID3D12Debug>() {
                debug.EnableDebugLayer();
            }
        }

        let factory_flags = if desc.validation {
            DXGI_CREATE_FACTORY_DEBUG
        } else {
            DXGI_CREATE_FACTORY_FLAGS(0)
        };
        let factory: IDXGIFactory4 = CreateDXGIFactory2(factory_flags)
            .map_err(|e| super::super::NotSupportedError::Platform(e))?;

        // Enumerate adapters and pick one
        let (adapter, adapter_desc) = pick_adapter(&factory, desc.device_id)?;
        let vendor = map_vendor(adapter_desc.VendorId);

        let device: ID3D12Device = {
            let mut dev: Option<ID3D12Device> = None;
            D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut dev)
                .map_err(|e| super::super::NotSupportedError::Platform(e))?;
            dev.unwrap()
        };

        // Command queue
        let queue_desc = D3D12_COMMAND_QUEUE_DESC {
            Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
            Flags: D3D12_COMMAND_QUEUE_FLAG_NONE,
            NodeMask: 0,
            Priority: 0,
        };
        let queue_raw: ID3D12CommandQueue =
            device.CreateCommandQueue(&queue_desc).map_err(|e| super::super::NotSupportedError::Platform(e))?;

        // Fence + event for CPU synchronization
        let fence: ID3D12Fence = device
            .CreateFence(0, D3D12_FENCE_FLAG_NONE)
            .map_err(|e| super::super::NotSupportedError::Platform(e))?;
        let fence_event: HANDLE = CreateEventW(None, false, false, None)
            .map_err(|e| super::super::NotSupportedError::Platform(e))?;

        // Static descriptor heaps
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

        // Command signatures for indirect rendering (no root argument modifications)
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

        let driver_version = adapter_desc.DriverVersion;
        let driver_info = format!(
            "{}.{}.{}.{}",
            (driver_version >> 48) & 0xFFFF,
            (driver_version >> 32) & 0xFFFF,
            (driver_version >> 16) & 0xFFFF,
            driver_version & 0xFFFF,
        );

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
    unsafe { device.CreateCommandSignature(&sig_desc, None).unwrap() }
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

        // Skip the software adapter
        if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0 {
            continue;
        }

        // Test that D3D12 is supported
        let mut test: Option<ID3D12Device> = None;
        if D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut test).is_err() {
            continue;
        }

        if preferred_device_id != 0 && desc.DeviceId == preferred_device_id {
            return Ok((adapter, desc));
        }

        // Otherwise take the first usable discrete adapter
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
