use codespan_reporting::{
    diagnostic::{Diagnostic, Label},
    files::SimpleFile,
    term::{
        self,
        termcolor::{ColorChoice, StandardStream},
    },
};
use std::error::Error;

pub fn print_err(error: &dyn Error) {
    eprint!("{}", error);

    let mut e = error.source();
    if e.is_some() {
        eprintln!(": ");
    } else {
        eprintln!();
    }

    while let Some(source) = e {
        eprintln!("\t{}", source);
        e = source.source();
    }
}

pub fn emit_annotated_error<E: Error>(ann_err: &naga::WithSpan<E>, filename: &str, source: &str) {
    let files = SimpleFile::new(filename, source);
    let config = term::Config::default();
    let writer = StandardStream::stderr(ColorChoice::Auto);

    let diagnostic = Diagnostic::error().with_labels(
        ann_err
            .spans()
            .map(|&(span, ref desc)| {
                Label::primary((), span.to_range().unwrap()).with_message(desc.to_owned())
            })
            .collect(),
    );

    term::emit(&mut writer.lock(), &config, &files, &diagnostic).expect("cannot write error");
}

impl super::TextureFormat {
    pub fn block_info(&self) -> super::TexelBlockInfo {
        fn uncompressed(size: u8) -> super::TexelBlockInfo {
            super::TexelBlockInfo {
                dimensions: (1, 1),
                size,
            }
        }
        fn cx_bc(size: u8) -> super::TexelBlockInfo {
            super::TexelBlockInfo {
                dimensions: (4, 4),
                size,
            }
        }
        match *self {
            Self::R8Unorm => uncompressed(1),
            Self::Rg8Unorm => uncompressed(2),
            Self::Rg8Snorm => uncompressed(2),
            Self::Rgba8Unorm => uncompressed(4),
            Self::Rgba8UnormSrgb => uncompressed(4),
            Self::Bgra8Unorm => uncompressed(4),
            Self::Bgra8UnormSrgb => uncompressed(4),
            Self::Rgba8Snorm => uncompressed(4),
            Self::R16Float => uncompressed(2),
            Self::Rg16Float => uncompressed(4),
            Self::Rgba16Float => uncompressed(8),
            Self::R32Float => uncompressed(4),
            Self::Rg32Float => uncompressed(8),
            Self::Rgba32Float => uncompressed(16),
            Self::R32Uint => uncompressed(4),
            Self::Rg32Uint => uncompressed(8),
            Self::Rgba32Uint => uncompressed(16),
            Self::Depth32Float => uncompressed(4),

            Self::Depth32FloatStencil8Uint => {
                log::warn!("Requested 'block_info' on depth-stencil format, information most likely incorrect");
                uncompressed(5)
            }
            Self::Stencil8Uint => {
                log::warn!(
                    "Requested 'block_info' on stencil format, information most likely incorrect"
                );
                uncompressed(1)
            }

            Self::Bc1Unorm => cx_bc(8),
            Self::Bc1UnormSrgb => cx_bc(8),
            Self::Bc2Unorm => cx_bc(16),
            Self::Bc2UnormSrgb => cx_bc(16),
            Self::Bc3Unorm => cx_bc(16),
            Self::Bc3UnormSrgb => cx_bc(16),
            Self::Bc4Unorm => cx_bc(8),
            Self::Bc4Snorm => cx_bc(8),
            Self::Bc5Unorm => cx_bc(16),
            Self::Bc5Snorm => cx_bc(16),
            Self::Bc6hUfloat => cx_bc(16),
            Self::Bc6hFloat => cx_bc(16),
            Self::Bc7Unorm => cx_bc(16),
            Self::Bc7UnormSrgb => cx_bc(16),
            Self::Rgb10a2Unorm => uncompressed(4),
            Self::Rg11b10Ufloat => uncompressed(4),
            Self::Rgb9e5Ufloat => uncompressed(4),
        }
    }

    pub const fn aspects(&self) -> super::TexelAspects {
        let mut aspects = super::TexelAspects::empty();
        aspects = aspects.union(self.depth_stencil_color());
        aspects = aspects.union(self.float_int_uint());
        if self.is_srgb() {
            aspects = aspects.union(super::TexelAspects::SRGB)
        }
        aspects
    }

    pub const fn is_srgb(&self) -> bool {
        match *self {
            crate::TextureFormat::R8Unorm
            | crate::TextureFormat::Rg8Unorm
            | crate::TextureFormat::Rg8Snorm
            | crate::TextureFormat::Rgba8Unorm
            | crate::TextureFormat::Bgra8Unorm
            | crate::TextureFormat::Rgba8Snorm
            | crate::TextureFormat::R16Float
            | crate::TextureFormat::Rg16Float
            | crate::TextureFormat::Rgba16Float
            | crate::TextureFormat::R32Float
            | crate::TextureFormat::Rg32Float
            | crate::TextureFormat::Rgba32Float
            | crate::TextureFormat::R32Uint
            | crate::TextureFormat::Rg32Uint
            | crate::TextureFormat::Rgba32Uint
            | crate::TextureFormat::Depth32Float
            | crate::TextureFormat::Depth32FloatStencil8Uint
            | crate::TextureFormat::Stencil8Uint
            | crate::TextureFormat::Bc1Unorm
            | crate::TextureFormat::Bc2Unorm
            | crate::TextureFormat::Bc3Unorm
            | crate::TextureFormat::Bc4Unorm
            | crate::TextureFormat::Bc4Snorm
            | crate::TextureFormat::Bc5Unorm
            | crate::TextureFormat::Bc5Snorm
            | crate::TextureFormat::Bc6hUfloat
            | crate::TextureFormat::Bc6hFloat
            | crate::TextureFormat::Bc7Unorm
            | crate::TextureFormat::Rgb10a2Unorm
            | crate::TextureFormat::Rg11b10Ufloat
            | crate::TextureFormat::Rgb9e5Ufloat => false,

            crate::TextureFormat::Bc7UnormSrgb
            | crate::TextureFormat::Rgba8UnormSrgb
            | crate::TextureFormat::Bgra8UnormSrgb
            | crate::TextureFormat::Bc1UnormSrgb
            | crate::TextureFormat::Bc2UnormSrgb
            | crate::TextureFormat::Bc3UnormSrgb => true,
        }
    }

    const fn depth_stencil_color(&self) -> super::TexelAspects {
        match *self {
            Self::Depth32Float => super::TexelAspects::DEPTH,
            Self::Depth32FloatStencil8Uint => {
                super::TexelAspects::DEPTH.union(super::TexelAspects::STENCIL)
            }
            Self::Stencil8Uint => super::TexelAspects::STENCIL,

            _ => super::TexelAspects::COLOR,
        }
    }

    const fn float_int_uint(&self) -> super::TexelAspects {
        match *self {
            crate::TextureFormat::Rg8Snorm
            | crate::TextureFormat::Rgba8Snorm
            | crate::TextureFormat::R32Uint
            | crate::TextureFormat::Rg32Uint
            | crate::TextureFormat::Rgba32Uint
            | crate::TextureFormat::Stencil8Uint
            | crate::TextureFormat::Bc4Snorm
            | crate::TextureFormat::Bc5Snorm => super::TexelAspects::INT,

            crate::TextureFormat::R8Unorm
            | crate::TextureFormat::Rg8Unorm
            | crate::TextureFormat::Rgba8Unorm
            | crate::TextureFormat::Rgba8UnormSrgb
            | crate::TextureFormat::Bgra8Unorm
            | crate::TextureFormat::Bgra8UnormSrgb
            | crate::TextureFormat::Bc1Unorm
            | crate::TextureFormat::Bc1UnormSrgb
            | crate::TextureFormat::Bc2Unorm
            | crate::TextureFormat::Bc2UnormSrgb
            | crate::TextureFormat::Bc3Unorm
            | crate::TextureFormat::Bc3UnormSrgb
            | crate::TextureFormat::Bc4Unorm
            | crate::TextureFormat::Bc5Unorm
            | crate::TextureFormat::Bc7Unorm
            | crate::TextureFormat::Bc7UnormSrgb => super::TexelAspects::UINT,

            crate::TextureFormat::R16Float
            | crate::TextureFormat::Rg16Float
            | crate::TextureFormat::Rgba16Float
            | crate::TextureFormat::R32Float
            | crate::TextureFormat::Rg32Float
            | crate::TextureFormat::Rgba32Float
            | crate::TextureFormat::Depth32Float
            | crate::TextureFormat::Depth32FloatStencil8Uint
            | crate::TextureFormat::Bc6hUfloat
            | crate::TextureFormat::Bc6hFloat
            | crate::TextureFormat::Rgb10a2Unorm
            | crate::TextureFormat::Rg11b10Ufloat
            | crate::TextureFormat::Rgb9e5Ufloat => super::TexelAspects::FLOAT,
        }
    }
}

impl super::ComputePipeline {
    /// Return the dispatch group counts sufficient to cover the given extent.
    pub fn get_dispatch_for(&self, extent: super::Extent) -> [u32; 3] {
        let wg_size = self.get_workgroup_size();
        [
            (extent.width + wg_size[0] - 1) / wg_size[0],
            (extent.height + wg_size[1] - 1) / wg_size[1],
            (extent.depth + wg_size[2] - 1) / wg_size[2],
        ]
    }
}
