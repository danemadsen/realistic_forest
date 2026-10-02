//! Shared render-world textures: the noise field, erosion atlases, blend
//! mask, SSAO noise, the two PBR terrain texture arrays, and the CPU texture
//! loading half (LoadTerrainTextures port) that feeds them.

use crate::constants::*;
use crate::erosion::ErosionCache;
use crate::noise::NoiseField;
use bevy::prelude::*;
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::render::{ExtractSchedule, MainWorld};
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// CPU loading (main world)
// ---------------------------------------------------------------------------

fn asset_path(relative: &str) -> PathBuf {
    std::path::Path::new("assets").join(relative)
}

fn load_image_rgba(path: &PathBuf) -> Option<(Vec<u8>, usize)> {
    let decoded = image::ImageReader::open(path).ok()?.decode().ok()?;
    let size = decoded.width() as usize;
    Some((decoded.to_rgba8().into_raw(), size))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TextureEncoding {
    /// Albedo RGB is sRGB; the packed ambient occlusion alpha is linear.
    Srgb,
    /// Normal vectors, roughness and grayscale masks are linear data.
    Linear,
}

fn srgb_to_linear(value: f32) -> f32 {
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb(value: f32) -> f32 {
    if value <= 0.0031308 {
        value * 12.92
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    }
}

/// Average each ratio x ratio block in linear light, then encode color
/// back to sRGB for storage. Averaging encoded albedo darkens fine gravel
/// and grass as they recede into the distance. Alpha always remains linear
/// because it contains a material parameter, not color or transparency.
/// Both images are square; `size` divides `src_size` exactly. Tile-aligned
/// blocks preserve the periodic boundary used by the repeat sampler.
fn downscale_tile_wrapping(
    src: &[u8],
    src_size: usize,
    size: usize,
    encoding: TextureEncoding,
) -> Vec<u8> {
    assert!(size > 0 && src_size.is_multiple_of(size));
    assert_eq!(src.len(), src_size * src_size * 4);
    if src_size == size {
        return src.to_vec();
    }
    let ratio = src_size / size;
    let count = ratio * ratio;
    // Decode once per possible byte instead of applying the transfer
    // function to every source texel in every material.
    let srgb_lookup: [f32; 256] = std::array::from_fn(|i| srgb_to_linear(i as f32 / 255.0));
    let mut dst = vec![0u8; size * size * 4];
    for oy in 0..size {
        for ox in 0..size {
            let mut sum = [0usize; 4];
            let mut linear_rgb = [0.0; 3];
            for dy in 0..ratio {
                for dx in 0..ratio {
                    let sx = (ox * ratio + dx) % src_size;
                    let sy = (oy * ratio + dy) % src_size;
                    let base = (sy * src_size + sx) * 4;
                    for channel in 0..4 {
                        sum[channel] += src[base + channel] as usize;
                    }
                    if encoding == TextureEncoding::Srgb {
                        for channel in 0..3 {
                            linear_rgb[channel] += srgb_lookup[src[base + channel] as usize];
                        }
                    }
                }
            }
            let base = (oy * size + ox) * 4;
            for channel in 0..4 {
                // Round to nearest at the base and each mip to avoid
                // systematically darkening material data at every level.
                dst[base + channel] = if channel < 3 && encoding == TextureEncoding::Srgb {
                    (linear_to_srgb(linear_rgb[channel] / count as f32) * 255.0)
                        .round()
                        .clamp(0.0, 255.0) as u8
                } else {
                    ((sum[channel] + count / 2) / count) as u8
                };
            }
        }
    }
    dst
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn downscale_averages_whole_blocks() {
        // 2x2 source, one value per pixel.
        let src: Vec<u8> = vec![
            4, 0, 0, 255, 8, 0, 0, 255, //
            16, 0, 0, 255, 32, 0, 0, 255,
        ];
        let out = downscale_tile_wrapping(&src, 2, 1, TextureEncoding::Linear);
        assert_eq!(out, vec![15, 0, 0, 255]);
    }

    #[test]
    fn linear_mips_round_without_a_dark_bias() {
        let blocks: [[u8; 4]; 4] = [[1, 1, 1, 2], [1, 1, 1, 3], [1, 1, 1, 4], [1, 1, 2, 6]];
        let mut src = vec![0u8; 4 * 4 * 4];
        for block in 0..4 {
            let (bx, by) = (block % 2, block / 2);
            for dy in 0..2 {
                for dx in 0..2 {
                    let index = ((by * 2 + dy) * 4 + bx * 2 + dx) * 4;
                    src[index] = blocks[block][dy * 2 + dx];
                    src[index + 3] = 255;
                }
            }
        }

        let mip = downscale_tile_wrapping(&src, 4, 2, TextureEncoding::Linear);
        let red =
            |bytes: &[u8]| -> Vec<u8> { (0..bytes.len() / 4).map(|i| bytes[i * 4]).collect() };
        assert_eq!(red(&mip), vec![1, 2, 2, 3]);
        assert_eq!(
            red(&downscale_tile_wrapping(
                &mip,
                2,
                1,
                TextureEncoding::Linear
            )),
            vec![2]
        );
    }

    #[test]
    fn albedo_filters_in_linear_light_but_occlusion_stays_linear() {
        // Half black, half white has 50% linear reflectance (sRGB 188).
        // The same values in the AO channel must average to 128 instead.
        let src = vec![
            0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 255, 255, 255, 255,
        ];
        assert_eq!(
            downscale_tile_wrapping(&src, 2, 1, TextureEncoding::Srgb),
            vec![188, 188, 188, 128]
        );
        assert_eq!(
            downscale_tile_wrapping(&src, 2, 1, TextureEncoding::Linear),
            vec![128, 128, 128, 128]
        );
    }

    #[test]
    fn srgb_transfer_preserves_shadows_and_round_trips_bytes() {
        // A power-of-2.2 approximation loses the sRGB linear shadow segment.
        assert!((srgb_to_linear(0.02) - 0.02 / 12.92).abs() < 1e-7);
        assert!((linear_to_srgb(0.001) - 0.01292).abs() < 1e-7);
        for byte in 0..=255 {
            let encoded = linear_to_srgb(srgb_to_linear(byte as f32 / 255.0));
            assert_eq!((encoded * 255.0).round() as u32, byte);
        }
    }
}

// ---------------------------------------------------------------------------
// Data movement
// ---------------------------------------------------------------------------

/// One layer's mip chain, level 0 at TERRAIN_TILE_SIZE², RGBA8.
#[derive(Debug)]
pub struct MipChain8 {
    pub levels: Vec<Vec<u8>>,
}

impl MipChain8 {
    fn blank() -> Self {
        Self {
            levels: vec![vec![0u8; TERRAIN_TILE_SIZE * TERRAIN_TILE_SIZE * 4]],
        }
    }
}

/// Terrain PBR layer chains decoded on the CPU at startup (main world).
#[derive(Debug, Resource)]
pub struct TerrainLayerDataLoaded {
    pub albedo_layers: Box<[MipChain8; TERRAIN_ATLAS_SLOTS]>,
    pub normal_rough_layers: Box<[MipChain8; TERRAIN_ATLAS_SLOTS]>,
}

/// Render-world staging slot for the layer chains, filled once by extraction.
#[derive(Default, Resource)]
pub struct TerrainLayerDataSlot(pub Option<Box<TerrainLayerDataLoaded>>);

/// CPU copy of the erosion blend mask (extracted for the GPU upload).
#[derive(Default, Resource)]
pub struct BlendMaskData(pub Option<Vec<f32>>);

/// Extraction: move the CPU layer data into the render world once;
/// `prepare_gpu_textures` consumes it.
pub fn extract_terrain_layers(
    mut main_world: ResMut<MainWorld>,
    mut slot: ResMut<TerrainLayerDataSlot>,
) {
    let Some(data) = main_world.remove_resource::<TerrainLayerDataLoaded>() else {
        return;
    };
    slot.0 = Some(Box::new(data));
}

/// Extraction: snapshot the blend mask before the GPU texture is created.
pub fn extract_blend_mask(main_world: Res<MainWorld>, mut data: ResMut<BlendMaskData>) {
    if data.0.is_some() {
        return;
    }
    if let Some(cache) = main_world.get_resource::<ErosionCache>() {
        data.0 = Some(cache.blend_mask_samples.clone());
    }
}

/// `BuildTerrainLayers`: the TERRAIN_ATLAS_SLOTS RGBA8 array layers for one
/// channel pair. mapSuffix selects the RGB source (_Color.png /
/// _NormalGL.png); maskSuffix (optional) packs a grayscale map into the
/// layer's alpha channel. Missing files leave a blank layer.
fn build_terrain_layers(
    map_suffix: &str,
    mask_suffix: Option<&str>,
    encoding: TextureEncoding,
) -> [MipChain8; TERRAIN_ATLAS_SLOTS] {
    let tile_area = TERRAIN_TILE_SIZE * TERRAIN_TILE_SIZE;
    let mut layers: Vec<Option<MipChain8>> = (0..TERRAIN_ATLAS_SLOTS).map(|_| None).collect();
    for (index, folder) in TERRAIN_MATERIALS.iter().enumerate() {
        let map_path = asset_path(&format!("textures/{folder}/{folder}{map_suffix}"));
        let Some((map_rgba, map_size)) = load_image_rgba(&map_path) else {
            log::warn!("Terrain texture missing: {}", map_path.display());
            continue;
        };
        let mut tile_bytes =
            downscale_tile_wrapping(&map_rgba, map_size, TERRAIN_TILE_SIZE, encoding);

        if let Some(mask_suffix) = mask_suffix {
            let mask_path = asset_path(&format!("textures/{folder}/{folder}{mask_suffix}"));
            if let Some((mask_rgba, mask_size)) = load_image_rgba(&mask_path) {
                let mask_bytes = downscale_tile_wrapping(
                    &mask_rgba,
                    mask_size,
                    TERRAIN_TILE_SIZE,
                    TextureEncoding::Linear,
                );
                for pixel in 0..tile_area {
                    tile_bytes[pixel * 4 + 3] = mask_bytes[pixel * 4];
                }
            }
        }

        // Successive 2x2 averages until 1x1, in the same color space used
        // by GPU filtering. Normals stay linear and normalize in the shader.
        let mut chain = vec![tile_bytes];
        while chain.last().unwrap().len() > 4 {
            let previous = chain.last().unwrap();
            let previous_size = previous.len() / 4;
            let previous_size = (previous_size as f32).sqrt() as usize;
            chain.push(downscale_tile_wrapping(
                previous,
                previous_size,
                previous_size / 2,
                encoding,
            ));
        }
        layers[index] = Some(MipChain8 { levels: chain });
    }

    layers
        .into_iter()
        .map(|layer| layer.unwrap_or_else(MipChain8::blank))
        .collect::<Vec<_>>()
        .try_into()
        .expect("fixed slot count")
}

/// `LoadTerrainTextures` — decode every material folder twice (the albedo
/// pair then the normal pair). Startup-only; heavy.
fn load_terrain_layers(mut commands: Commands) {
    log::info!("TERRAIN TEXTURES: loading PBR arrays");
    let albedo_layers = build_terrain_layers(
        "_Color.png",
        Some("_AmbientOcclusion.png"),
        TextureEncoding::Srgb,
    );
    let normal_rough_layers = build_terrain_layers(
        "_NormalGL.png",
        Some("_Roughness.png"),
        TextureEncoding::Linear,
    );
    commands.insert_resource(TerrainLayerDataLoaded {
        albedo_layers: Box::new(albedo_layers),
        normal_rough_layers: Box::new(normal_rough_layers),
    });
}

/// Startup registration for the main world.
pub fn register_main_texture_systems(app: &mut bevy::app::App) {
    app.add_systems(bevy::app::PreStartup, load_terrain_layers);
}

// ---------------------------------------------------------------------------
// GPU textures (render world)
// ---------------------------------------------------------------------------

pub const SIM_TEXTURE_SIZE: u32 = EROSION_RESOLUTION as u32;

/// All world textures held by the render passes; created once by
/// `prepare_gpu_textures` after CPU data extraction.
pub struct GpuWorldTextures {
    /// R32 1024², 11 mips, trilinear + repeat (base noise). Held so the
    /// texture outlives the view the samplers bind.
    #[allow(dead_code)]
    pub noise_texture: wgpu::Texture,
    pub noise_view: wgpu::TextureView,
    /// RGBA32F atlas (bilinear, repeat).
    pub height_atlas_view: wgpu::TextureView,
    /// RGBA32F atlas (bilinear, repeat).
    pub flow_atlas_view: wgpu::TextureView,
    /// RGBA32F 11x11 tile lookup (point); kept for per-frame updates.
    pub lookup_texture: wgpu::Texture,
    pub lookup_view: wgpu::TextureView,
    /// R32 blend mask (bilinear, clamp — the C++ sets CLAMP here).
    pub blend_mask_view: wgpu::TextureView,
    /// RGBA32F pairs (point) for terrain/water/flux/drainage simulation state.
    pub sim_terrain: [wgpu::Texture; 2],
    pub sim_water: [wgpu::Texture; 2],
    pub sim_flux: [wgpu::Texture; 2],
    pub sim_drainage: [wgpu::Texture; 2],
    /// Each cell's drainage-routing normaliser, written by the water pass for
    /// the terrain pass of the same iteration (R; see erosion-water.wgsl).
    pub sim_routing: wgpu::Texture,
    pub albedo_array_view: wgpu::TextureView,
    pub normal_rough_array_view: wgpu::TextureView,
    /// Rgba8Unorm 4x4 rotation noise (point, repeat).
    pub ssao_noise_view: wgpu::TextureView,
}

pub const SSAO_NOISE_WIDTH: u32 = 4;

fn sim_texture(device: &RenderDevice, label: &str) -> wgpu::Texture {
    device
        .wgpu_device()
        .create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: SIM_TEXTURE_SIZE,
                height: SIM_TEXTURE_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        })
}

fn float_texture(
    device: &RenderDevice,
    label: &str,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    mips: u32,
    usage: wgpu::TextureUsages,
) -> wgpu::Texture {
    device
        .wgpu_device()
        .create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: mips,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        })
}

/// Default SSAO 4x4 rotation-noise pixels, ported from `CreateSSAONoise`.
pub fn ssao_noise_pixels() -> Vec<u8> {
    let count = SSAO_NOISE_WIDTH as usize * SSAO_NOISE_WIDTH as usize;
    let mut pixels = vec![0u8; count * 4];
    for (index, pixel) in pixels.chunks_exact_mut(4).enumerate() {
        let angle = index as f32 * 2.39996323;
        pixel[0] = ((angle.cos() * 0.5 + 0.5) * 255.0) as u8;
        pixel[1] = ((angle.sin() * 0.5 + 0.5) * 255.0) as u8;
        pixel[2] = 128;
        pixel[3] = 255;
    }
    pixels
}

/// Noise texture mip chain on the CPU: 2x2 float averages with wrap
/// (matching glGenerateMipmap on the REPEAT-wrapped GL texture).
fn noise_mip_chain(samples: &[f32], size: usize) -> Vec<Vec<f32>> {
    let mut chain = vec![samples.to_vec()];
    let mut previous_size = size;
    while previous_size > 1 {
        let previous = chain.last().unwrap();
        let next_size = previous_size / 2;
        let mut next = vec![0.0f32; next_size * next_size];
        for oy in 0..next_size {
            for ox in 0..next_size {
                let mut sum = 0.0f32;
                for dy in 0..2 {
                    for dx in 0..2 {
                        let sx = (ox * 2 + dx) % previous_size;
                        let sy = (oy * 2 + dy) % previous_size;
                        sum += previous[sy * previous_size + sx];
                    }
                }
                next[oy * next_size + ox] = sum * 0.25;
            }
        }
        chain.push(next);
        previous_size = next_size;
    }
    chain
}

/// queue.write_texture with rows padded to wgpu's 256-byte copy alignment.
/// `data` holds rows of `width * bytes_per_pixel` contiguous bytes.
fn write_padded(
    queue: &RenderQueue,
    texture: &wgpu::Texture,
    data: &[u8],
    width: u32,
    height: u32,
    bytes_per_pixel: u32,
    mip_level: u32,
) {
    write_padded_at(
        queue,
        texture,
        data,
        width,
        height,
        bytes_per_pixel,
        [0, 0],
        mip_level,
    );
}

/// `write_padded` into a sub-rectangle: the erosion node patches 260x260
/// atlas tiles (4160-byte rows) and the 11x11 lookup (44-byte rows), neither
/// of which meets wgpu's 256-byte row alignment on its own.
///
/// `mip_level` is threaded through rather than fixed at 0: mip chains uploaded
/// level by level (the base noise) must land in their own levels, and a
/// hardcoded 0 silently stacks them all into the base level — leaving the base
/// level holding only the final 1x1 mip and every real mip zero-filled.
pub fn write_padded_at(
    queue: &RenderQueue,
    texture: &wgpu::Texture,
    data: &[u8],
    width: u32,
    height: u32,
    bytes_per_pixel: u32,
    origin: [u32; 2],
    mip_level: u32,
) {
    let raw_row = (width * bytes_per_pixel) as usize;
    let row = raw_row.next_multiple_of(256);
    let extent = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };
    let destination = wgpu::TexelCopyTextureInfo {
        texture,
        mip_level,
        origin: wgpu::Origin3d {
            x: origin[0],
            y: origin[1],
            z: 0,
        },
        aspect: wgpu::TextureAspect::All,
    };
    if row == raw_row {
        queue.write_texture(
            destination,
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(raw_row as u32),
                rows_per_image: Some(height),
            },
            extent,
        );
        return;
    }
    let rows = height as usize;
    let mut padded = vec![0u8; row * rows];
    for y in 0..rows {
        padded[y * row..y * row + raw_row].copy_from_slice(&data[y * raw_row..(y + 1) * raw_row]);
    }
    queue.write_texture(
        destination,
        &padded,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(row as u32),
            rows_per_image: Some(height),
        },
        extent,
    );
}

/// Padded write for per-level f32 chains (the noise texture).
fn write_f32_texture_levels(texture: &wgpu::Texture, queue: &RenderQueue, levels: &[Vec<f32>]) {
    let width = texture.width();
    for (level, data) in levels.iter().enumerate() {
        let size = (width >> level).max(1);
        let bytes: Vec<u8> = data.iter().flat_map(|s| s.to_le_bytes()).collect();
        write_padded(queue, texture, &bytes, size, size, 4, level as u32);
    }
}

#[derive(Resource, Default)]
pub struct GpuWorldTexturesOption(pub Option<Box<GpuWorldTextures>>);

/// Creates every shared texture once the CPU layer chains and the noise
/// field have been extracted into the render world.
pub fn prepare_gpu_textures(
    mut option: ResMut<GpuWorldTexturesOption>,
    noise_field: Res<NoiseField>,
    mut layer_slot: ResMut<TerrainLayerDataSlot>,
    blend_mask: Res<BlendMaskData>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    if option.0.is_some() || layer_slot.0.is_none() {
        return;
    }
    let device = &device;
    let queue = &*queue;
    let layers = layer_slot.0.take().expect("checked above");
    let atlas_size = EROSION_ATLAS_SIZE as u32;

    // Base noise: R32 1024², 11 mips, trilinear, repeat.
    let noise_chain = noise_mip_chain(&noise_field.samples, NOISE_RESOLUTION);
    let noise_texture = float_texture(
        device,
        "noise",
        NOISE_RESOLUTION as u32,
        NOISE_RESOLUTION as u32,
        wgpu::TextureFormat::R32Float,
        noise_chain.len() as u32,
        wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
    );
    write_f32_texture_levels(&noise_texture, queue, &noise_chain);
    let noise_view = noise_texture.create_view(&Default::default());

    // Erosion atlases: RGBA32F, bilinear (the C++ leaves wrap at raylib's
    // REPEAT default). Initial content zero, patched later from readbacks.
    let zero_atlas: Vec<u8> = vec![0u8; atlas_size as usize * atlas_size as usize * 16];
    let make_atlas = |label: &str| {
        let texture = float_texture(
            device,
            label,
            atlas_size,
            atlas_size,
            wgpu::TextureFormat::Rgba32Float,
            1,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        );
        // 2640 * 16 = 42240 bytes per row, already 256-aligned.
        queue.write_texture(
            texture.as_image_copy(),
            &zero_atlas,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(atlas_size * 16),
                rows_per_image: Some(atlas_size),
            },
            wgpu::Extent3d {
                width: atlas_size,
                height: atlas_size,
                depth_or_array_layers: 1,
            },
        );
        texture
    };
    let height_atlas_view = make_atlas("erosion_height_atlas").create_view(&Default::default());
    let flow_atlas_view = make_atlas("erosion_flow_atlas").create_view(&Default::default());

    // Tile lookup: RGBA32F 11x11 point-sampled, per-frame updates.
    let lookup_texture = float_texture(
        device,
        "erosion_lookup",
        EROSION_LOOKUP_DIAMETER as u32,
        EROSION_LOOKUP_DIAMETER as u32,
        wgpu::TextureFormat::Rgba32Float,
        1,
        wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
    );
    let lookup_view = lookup_texture.create_view(&Default::default());

    // Blend mask: R32, bilinear, CLAMP (the C++ sets CLAMP here).
    let blend_samples = blend_mask
        .0
        .clone()
        .unwrap_or_else(|| vec![1.0f32; EROSION_OUTPUT_RESOLUTION * EROSION_OUTPUT_RESOLUTION]);
    let blend_texture = float_texture(
        device,
        "erosion_blend_mask",
        EROSION_OUTPUT_RESOLUTION as u32,
        EROSION_OUTPUT_RESOLUTION as u32,
        wgpu::TextureFormat::R32Float,
        1,
        wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
    );
    {
        let bytes: Vec<u8> = blend_samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        // 256 * 4 = 1024 bytes per row, already 256-aligned.
        queue.write_texture(
            blend_texture.as_image_copy(),
            &bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(EROSION_OUTPUT_RESOLUTION as u32 * 4),
                rows_per_image: Some(EROSION_OUTPUT_RESOLUTION as u32),
            },
            wgpu::Extent3d {
                width: EROSION_OUTPUT_RESOLUTION as u32,
                height: EROSION_OUTPUT_RESOLUTION as u32,
                depth_or_array_layers: 1,
            },
        );
    }
    let blend_mask_view = blend_texture.create_view(&Default::default());

    // Simulation target pairs.
    let sim_terrain = [
        sim_texture(device, "sim_terrain_0"),
        sim_texture(device, "sim_terrain_1"),
    ];
    let sim_water = [
        sim_texture(device, "sim_water_0"),
        sim_texture(device, "sim_water_1"),
    ];
    let sim_flux = [
        sim_texture(device, "sim_flux_0"),
        sim_texture(device, "sim_flux_1"),
    ];
    let sim_drainage = [
        sim_texture(device, "sim_drainage_0"),
        sim_texture(device, "sim_drainage_1"),
    ];
    let sim_routing = sim_texture(device, "sim_routing");

    // PBR texture arrays with CPU-built mip chains (wgpu cannot generate
    // array mips on the GPU), trilinear + repeat + max anisotropy.
    let make_array = |label: &str,
                      layers: &Box<[MipChain8; TERRAIN_ATLAS_SLOTS]>,
                      format: wgpu::TextureFormat| {
        let max_levels = layers
            .iter()
            .map(|layer| layer.levels.len())
            .max()
            .unwrap_or(1) as u32;
        let texture = device
            .wgpu_device()
            .create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: TERRAIN_TILE_SIZE as u32,
                    height: TERRAIN_TILE_SIZE as u32,
                    depth_or_array_layers: TERRAIN_ATLAS_SLOTS as u32,
                },
                mip_level_count: max_levels,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
        for level in 0..max_levels as usize {
            let size = (TERRAIN_TILE_SIZE >> level).max(1) as u32;
            // Layers contiguous, each row padded to 256.
            let raw_row = size as usize * 4;
            let row = raw_row.next_multiple_of(256);
            let mut level_bytes = vec![0u8; row * size as usize * TERRAIN_ATLAS_SLOTS];
            for (slot, layer) in layers.iter().enumerate() {
                let Some(data) = layer.levels.get(level) else {
                    continue; // shorter chains leave zero-filled tail levels
                };
                let layer_offset = slot * row * size as usize;
                for y in 0..size as usize {
                    level_bytes[layer_offset + y * row..layer_offset + y * row + raw_row]
                        .copy_from_slice(&data[y * raw_row..(y + 1) * raw_row]);
                }
            }
            queue.write_texture(
                // NOTE: `Texture::as_image_copy()` hardcodes mip_level 0, so
                // it cannot be used here — every iteration would land on the
                // base level, leaving the base level holding only the last
                // (1x1) mip and every real mip zero-filled. The albedo array
                // then samples as pure black at every distance. The copy
                // target is built by hand so each level lands in its own mip.
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: level as u32,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &level_bytes,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row as u32),
                    rows_per_image: Some(size),
                },
                wgpu::Extent3d {
                    width: size,
                    height: size,
                    depth_or_array_layers: TERRAIN_ATLAS_SLOTS as u32,
                },
            );
        }
        texture.create_view(&Default::default())
    };
    // sRGB views decode RGB before bilinear/trilinear interpolation. Doing
    // that in the shader after sampling averages encoded color incorrectly;
    // alpha is deliberately exempt from the GPU's sRGB conversion.
    let albedo_array_view = make_array(
        "terrain_albedo_array",
        &layers.albedo_layers,
        wgpu::TextureFormat::Rgba8UnormSrgb,
    );
    let normal_rough_array_view = make_array(
        "terrain_normal_rough_array",
        &layers.normal_rough_layers,
        wgpu::TextureFormat::Rgba8Unorm,
    );

    // SSAO 4x4 rotation noise (point, repeat).
    let ssao_noise_texture = device
        .wgpu_device()
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("ssao_noise"),
            size: wgpu::Extent3d {
                width: SSAO_NOISE_WIDTH,
                height: SSAO_NOISE_WIDTH,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
    write_padded(
        queue,
        &ssao_noise_texture,
        &ssao_noise_pixels(),
        SSAO_NOISE_WIDTH,
        SSAO_NOISE_WIDTH,
        4,
        0,
    );
    let ssao_noise_view = ssao_noise_texture.create_view(&Default::default());

    option.0 = Some(Box::new(GpuWorldTextures {
        noise_texture,
        noise_view,
        height_atlas_view,
        flow_atlas_view,
        lookup_texture,
        lookup_view,
        blend_mask_view,
        sim_terrain,
        sim_water,
        sim_flux,
        sim_drainage,
        sim_routing,
        albedo_array_view,
        normal_rough_array_view,
        ssao_noise_view,
    }));
}

/// Render-app registration: extract systems and the resources they fill.
/// `prepare_gpu_textures` itself is registered by `ForestRenderPlugin::build`,
/// which owns the `Render`-schedule ordering it shares with
/// `prepare_forest_globals`.
pub fn register_gpu_texture_systems(render_app: &mut bevy::app::SubApp) {
    render_app
        .init_resource::<TerrainLayerDataSlot>()
        .init_resource::<BlendMaskData>()
        .init_resource::<GpuWorldTexturesOption>()
        .add_systems(
            ExtractSchedule,
            (extract_terrain_layers, extract_blend_mask),
        );
}
