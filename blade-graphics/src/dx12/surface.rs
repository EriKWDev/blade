use windows::{
    core::Interface,
    Win32::{
        Foundation::HWND,
        Graphics::{
            Direct3D12::*,
            Dxgi::{Common::*, *},
        },
        System::Threading::WaitForSingleObjectEx,
    },
};

impl super::Context {
    pub fn create_surface<
        I: raw_window_handle::HasWindowHandle + raw_window_handle::HasDisplayHandle,
    >(
        &self,
        window: &I,
    ) -> Result<super::Surface, crate::NotSupportedError> {
        let hwnd = match window.window_handle().unwrap().as_raw() {
            raw_window_handle::RawWindowHandle::Win32(handle) => {
                HWND(handle.hwnd.get() as *mut _)
            }
            _ => return Err(crate::NotSupportedError::PlatformNotSupported),
        };

        Ok(super::Surface {
            hwnd: Some(hwnd),
            swapchain: None,
            frames: Vec::new(),
            format: crate::TextureFormat::Bgra8Unorm,
            dxgi_format: DXGI_FORMAT_B8G8R8A8_UNORM,
            alpha: crate::AlphaMode::Ignored,
            target_size: [0; 2],
            next_present_id: 1,
        })
    }

    pub fn destroy_surface(&self, surface: &mut super::Surface) {
        self.flush_gpu();
        for frame in surface.frames.drain(..) {
            drop(frame.resource);
        }
        surface.swapchain = None;
    }

    pub fn reconfigure_surface(
        &self,
        surface: &mut super::Surface,
        config: crate::SurfaceConfig,
    ) {
        self.flush_gpu();

        let hwnd = surface.hwnd.expect("surface has no window handle");

        let (format, dxgi_format, alpha) = pick_format(config.color_space, config.transparent);

        let num_frames: u32 = if config.requested_num_frames == 0 {
            match config.display_sync {
                crate::DisplaySync::Block => 3,
                crate::DisplaySync::Recent => 3,
                crate::DisplaySync::Tear => 2,
            }
        } else {
            config.requested_num_frames
        };

        let tearing_flag = match config.display_sync {
            crate::DisplaySync::Tear => DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING,
            _ => DXGI_SWAP_CHAIN_FLAG(0),
        };

        // Drop old swapchain frames before resizing
        for frame in surface.frames.drain(..) {
            drop(frame.resource);
        }

        let queue = self.queue.lock().unwrap();

        if let Some(ref sc) = surface.swapchain {
            // Resize existing swapchain
            unsafe {
                sc.ResizeBuffers(
                    num_frames,
                    config.size.width,
                    config.size.height,
                    dxgi_format,
                    tearing_flag,
                )
                .unwrap();
            }
        } else {
            // Create new swapchain
            let swap_desc = DXGI_SWAP_CHAIN_DESC1 {
                Width: config.size.width,
                Height: config.size.height,
                Format: dxgi_format,
                Stereo: false.into(),
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
                BufferCount: num_frames,
                Scaling: DXGI_SCALING_STRETCH,
                SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
                AlphaMode: map_alpha_mode(alpha),
                Flags: tearing_flag.0 as u32,
            };
            let sc1: IDXGISwapChain1 = unsafe {
                self.factory
                    .CreateSwapChainForHwnd(
                        &queue.raw,
                        hwnd,
                        &swap_desc,
                        None,
                        None,
                    )
                    .unwrap()
            };
            surface.swapchain = Some(sc1.cast::<IDXGISwapChain3>().unwrap());
        }
        drop(queue);

        surface.format = format;
        surface.dxgi_format = dxgi_format;
        surface.alpha = alpha;
        surface.target_size = [config.size.width as u16, config.size.height as u16];
        surface.next_present_id = 1;

        // Acquire RTVs for the new back buffers
        let sc = surface.swapchain.as_ref().unwrap();
        let buffer_count: u32 = {
            let mut desc: DXGI_SWAP_CHAIN_DESC1 = unsafe { std::mem::zeroed() };
            unsafe { sc.GetDesc1(&mut desc).unwrap() };
            desc.BufferCount
        };

        for i in 0..buffer_count {
            let resource: ID3D12Resource = unsafe { sc.GetBuffer(i).unwrap() };
            let rtv = self.alloc_rtv();
            unsafe {
                self.device.CreateRenderTargetView(
                    &resource,
                    Some(&D3D12_RENDER_TARGET_VIEW_DESC {
                        Format: dxgi_format,
                        ViewDimension: D3D12_RTV_DIMENSION_TEXTURE2D,
                        Anonymous: D3D12_RENDER_TARGET_VIEW_DESC_0 {
                            Texture2D: D3D12_TEX2D_RTV { MipSlice: 0, PlaneSlice: 0 },
                        },
                    }),
                    rtv,
                );
            }
            surface.frames.push(super::SurfaceFrame {
                resource,
                rtv,
                fence_value: 0,
            });
        }
    }

    fn flush_gpu(&self) {
        let mut queue = self.queue.lock().unwrap();
        let progress = queue.last_progress + 1;
        unsafe {
            queue.raw.Signal(&queue.fence, progress).unwrap();
            queue
                .fence
                .SetEventOnCompletion(progress, queue.fence_event)
                .unwrap();
            WaitForSingleObjectEx(queue.fence_event, u32::MAX, false);
        }
        queue.last_progress = progress;
    }
}

impl super::Surface {
    pub fn info(&self) -> crate::SurfaceInfo {
        crate::SurfaceInfo {
            format: self.format,
            alpha: self.alpha,
        }
    }

    pub fn acquire_frame(&mut self) -> super::Frame {
        let sc = self.swapchain.as_ref().expect("surface not configured");
        let index = unsafe { sc.GetCurrentBackBufferIndex() };
        let frame = &self.frames[index as usize];

        // CPU-wait if this back buffer is still in use by the GPU
        // (fence_value was set when we last submitted work using this buffer)
        // NOTE: We don't have direct access to the queue fence here, so we rely on
        // the per-frame fence tracking being done in submit() externally.
        // For correctness, the user must call wait_for() on the SyncPoint returned
        // by the previous submit that used this buffer. This matches how Vulkan
        // works with acquire blocking on the acquire semaphore.

        let present_id = self.next_present_id;
        self.next_present_id += 1;

        super::Frame {
            resource: frame.resource.clone(),
            swapchain: sc.clone(),
            rtv: frame.rtv,
            buffer_index: index,
            present_id,
            format: self.format,
            target_size: self.target_size,
            display_sync: true, // default; reconfigure_surface can override
        }
    }

    pub fn last_present_id(&self) -> u64 {
        self.next_present_id.saturating_sub(1)
    }
}

fn pick_format(
    color_space: crate::ColorSpace,
    transparent: bool,
) -> (crate::TextureFormat, DXGI_FORMAT, crate::AlphaMode) {
    match (color_space, transparent) {
        (crate::ColorSpace::Linear, false) => (
            crate::TextureFormat::Bgra8Unorm,
            DXGI_FORMAT_B8G8R8A8_UNORM,
            crate::AlphaMode::Ignored,
        ),
        (crate::ColorSpace::Srgb, false) => (
            crate::TextureFormat::Bgra8UnormSrgb,
            DXGI_FORMAT_B8G8R8A8_UNORM_SRGB,
            crate::AlphaMode::Ignored,
        ),
        (_, true) => (
            crate::TextureFormat::Bgra8Unorm,
            DXGI_FORMAT_B8G8R8A8_UNORM,
            crate::AlphaMode::PreMultiplied,
        ),
    }
}

fn map_alpha_mode(mode: crate::AlphaMode) -> DXGI_ALPHA_MODE {
    match mode {
        crate::AlphaMode::Ignored => DXGI_ALPHA_MODE_IGNORE,
        crate::AlphaMode::PreMultiplied => DXGI_ALPHA_MODE_PREMULTIPLIED,
        crate::AlphaMode::PostMultiplied => DXGI_ALPHA_MODE_STRAIGHT,
    }
}
