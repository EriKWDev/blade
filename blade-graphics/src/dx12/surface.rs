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
            frame_latency_waitable: None,
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
        // FLIP-model swapchains reject _SRGB backbuffer formats. Create the
        // swapchain with the UNORM base and put the _SRGB view on the RTV, so the
        // hardware still encodes linear shader output to sRGB on write.
        let swapchain_format = swapchain_base_format(dxgi_format);

        let num_frames: u32 = if config.requested_num_frames == 0 {
            match config.display_sync {
                crate::DisplaySync::Block => 3,
                crate::DisplaySync::Recent => 3,
                crate::DisplaySync::Tear => 2,
            }
        } else {
            config.requested_num_frames
        };

        // ALLOW_TEARING may only be set if the adapter actually supports it;
        // setting it otherwise makes CreateSwapChain fail with DXGI_ERROR_INVALID_CALL
        // (which is what broke exclusive-fullscreen creation on this adapter).
        let tearing_flag = match config.display_sync {
            crate::DisplaySync::Tear if self.tearing_supported() => {
                DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING
            }
            _ => DXGI_SWAP_CHAIN_FLAG(0),
        };
        // FRAME_LATENCY_WAITABLE_OBJECT gives us a handle to block on in
        // acquire_frame, bounding how far the CPU runs ahead of presentation.
        let swap_flags = tearing_flag.0 | DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT.0;

        // Drop old swapchain frames before resizing
        for frame in surface.frames.drain(..) {
            drop(frame.resource);
        }

        let queue = self.queue.lock().unwrap();

        if let Some(ref sc) = surface.swapchain {
            unsafe {
                sc.ResizeBuffers(
                    num_frames,
                    config.size.width,
                    config.size.height,
                    swapchain_format,
                    DXGI_SWAP_CHAIN_FLAG(swap_flags),
                )
                .unwrap();
            }
        } else {
            let swap_desc = DXGI_SWAP_CHAIN_DESC1 {
                Width: config.size.width,
                Height: config.size.height,
                Format: swapchain_format,
                Stereo: false.into(),
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
                BufferCount: num_frames,
                Scaling: DXGI_SCALING_STRETCH,
                SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
                AlphaMode: map_alpha_mode(alpha),
                Flags: swap_flags as u32,
            };
            let sc_result = unsafe {
                self.factory.CreateSwapChainForHwnd(&queue.raw, hwnd, &swap_desc, None, None)
            };
            let sc1: IDXGISwapChain1 = match sc_result {
                Ok(sc) => sc,
                Err(e) => {
                    log_dxgi_messages();
                    panic!(
                        "CreateSwapChainForHwnd failed: {e}\n  desc: {}x{} fmt={:?} buffers={} flags={:#x}",
                        swap_desc.Width, swap_desc.Height, swap_desc.Format,
                        swap_desc.BufferCount, swap_desc.Flags
                    );
                }
            };
            let sc3 = sc1.cast::<IDXGISwapChain3>().unwrap();
            // Bound the present queue to one frame so the waitable releases as
            // soon as the prior frame's present is consumed, then grab the handle
            // acquire_frame blocks on.
            unsafe {
                let _ = sc3.SetMaximumFrameLatency(1);
            }
            surface.frame_latency_waitable =
                Some(unsafe { sc3.GetFrameLatencyWaitableObject() });
            surface.swapchain = Some(sc3);
        }
        drop(queue);

        surface.format = format;
        surface.dxgi_format = dxgi_format;
        surface.alpha = alpha;
        surface.target_size = [config.size.width as u16, config.size.height as u16];
        surface.next_present_id = 1;

        let sc = surface.swapchain.as_ref().unwrap();
        let buffer_count: u32 = {
            let desc = unsafe { sc.GetDesc1().unwrap() };
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

    /// Whether the adapter supports DXGI present tearing (required before the
    /// ALLOW_TEARING swap-chain flag may be used).
    fn tearing_supported(&self) -> bool {
        let factory5: IDXGIFactory5 = match self.factory.cast() {
            Ok(f) => f,
            Err(_) => return false,
        };
        let mut allow: windows::Win32::Foundation::BOOL = false.into();
        let ok = unsafe {
            factory5.CheckFeatureSupport(
                DXGI_FEATURE_PRESENT_ALLOW_TEARING,
                &mut allow as *mut _ as *mut std::ffi::c_void,
                std::mem::size_of::<windows::Win32::Foundation::BOOL>() as u32,
            )
        };
        ok.is_ok() && allow.as_bool()
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
        // Block until the swapchain is ready for a new frame — the DX12 analog of
        // Vulkan's acquire blocking on the image-available semaphore. Without this
        // the CPU races ahead of the present queue and can record into a back
        // buffer still owned by the presenter, producing occasional flicker.
        if let Some(waitable) = self.frame_latency_waitable {
            unsafe {
                WaitForSingleObjectEx(waitable, 1000, false);
            }
        }

        let sc = self.swapchain.as_ref().expect("surface not configured");
        let index = unsafe { sc.GetCurrentBackBufferIndex() };
        let frame = &self.frames[index as usize];

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

/// Drain the DXGI debug message queue to the log. DXGI errors (e.g. from
/// CreateSwapChainForHwnd) go here, NOT to the D3D12 info queue, so this is the
/// only way to see why a swapchain call failed. No-op if the DXGI debug layer
/// isn't installed.
fn log_dxgi_messages() {
    let info_queue: IDXGIInfoQueue = match unsafe { DXGIGetDebugInterface1(0) } {
        Ok(q) => q,
        Err(_) => return,
    };
    unsafe {
        let count = info_queue.GetNumStoredMessages(DXGI_DEBUG_ALL);
        for i in 0..count {
            let mut len: usize = 0;
            if info_queue.GetMessage(DXGI_DEBUG_ALL, i, None, &mut len).is_err() || len == 0 {
                continue;
            }
            let mut buf = vec![0u8; len];
            let msg = buf.as_mut_ptr() as *mut DXGI_INFO_QUEUE_MESSAGE;
            if info_queue.GetMessage(DXGI_DEBUG_ALL, i, Some(msg), &mut len).is_ok() {
                let m = &*msg;
                let text = std::slice::from_raw_parts(m.pDescription, m.DescriptionByteLength);
                log::error!("[DXGI] {}", String::from_utf8_lossy(text).trim_end_matches('\0'));
            }
        }
    }
}

/// The UNORM base of an `_SRGB` format. FLIP swapchains require the non-sRGB
/// format for the backbuffer; the sRGB-ness is applied via the RTV instead.
fn swapchain_base_format(format: DXGI_FORMAT) -> DXGI_FORMAT {
    match format {
        DXGI_FORMAT_B8G8R8A8_UNORM_SRGB => DXGI_FORMAT_B8G8R8A8_UNORM,
        DXGI_FORMAT_R8G8B8A8_UNORM_SRGB => DXGI_FORMAT_R8G8B8A8_UNORM,
        other => other,
    }
}

fn pick_format(
    color_space: crate::ColorSpace,
    transparent: bool,
) -> (crate::TextureFormat, DXGI_FORMAT, crate::AlphaMode) {
    // `ColorSpace::Linear` means the app writes LINEAR color and expects the
    // presentation path to encode it to sRGB — matching Vulkan, which uses
    // EXTENDED_SRGB_LINEAR_EXT (compositor encodes) or falls back to a _SRGB
    // format (hardware encodes). DX12 has no scRGB-linear standard swapchain, so
    // we take the fallback: report a _SRGB format and put a _SRGB RTV on a UNORM
    // backbuffer (swapchain_base_format strips it) — the RTV hardware-encodes
    // linear->sRGB on write. Reporting plain UNORM here (no encode) is what made
    // the whole image render darker than on Vulkan.
    //
    // `ColorSpace::Srgb` means the app already writes sRGB-encoded values, so the
    // RTV must NOT re-encode: plain UNORM.
    match (color_space, transparent) {
        (crate::ColorSpace::Linear, false) => (
            crate::TextureFormat::Bgra8UnormSrgb,
            DXGI_FORMAT_B8G8R8A8_UNORM_SRGB,
            crate::AlphaMode::Ignored,
        ),
        (crate::ColorSpace::Srgb, false) => (
            crate::TextureFormat::Bgra8Unorm,
            DXGI_FORMAT_B8G8R8A8_UNORM,
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
