//! Imported grass geometry and a deterministic, bounded field of instances.
//!
//! Placement is indexed by world cells rather than camera frames. Terrain and
//! erosion eligibility are evaluated by the render shader using the same height
//! and material functions as the terrain itself.

use bevy::prelude::Resource;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub const SCATTER_RADIUS: f32 = 235.0;
pub const SCATTER_CELL_SIZE: f32 = 1.0;
pub const SCATTER_ANCHOR_SIZE: f32 = 8.0;
// The snapped center can differ from the camera by 4 * sqrt(2) meters.
const CANDIDATE_RADIUS: f32 = 242.0;
// Candidate rings extend beyond their GPU fade so the nearest 8 m anchor
// cannot reveal instance insertion or removal at the ring boundaries.
const FAR_MIDDLE_CANDIDATE_RADIUS: f32 = 207.0;
const CLOSE_MIDDLE_CANDIDATE_RADIUS: f32 = 147.0;
const CARPET_CANDIDATE_RADIUS: f32 = 40.0;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GrassVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub tangent: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GrassInstance {
    pub xz: [f32; 2],
    pub rotation: f32,
    pub scale: f32,
    pub tint: f32,
    pub seed: f32,
    /// World-anchored patch density (0.65..=1.0), then layer:
    /// 0 base, 1 far middle, 2 close middle, 3 carpet.
    pub _pad: [f32; 2],
}

const _: () = assert!(std::mem::size_of::<GrassVertex>() == 48);
const _: () = assert!(std::mem::size_of::<GrassInstance>() == 32);

pub struct GrassModel {
    pub name: String,
    pub vertices: Vec<GrassVertex>,
    pub indices: Vec<u32>,
    pub material: usize,
    pub height: f32,
    pub radius: f32,
}

#[derive(Clone)]
pub struct GrassTexture {
    pub rgba: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

pub struct GrassMaterial {
    pub base_color: GrassTexture,
    pub normal: GrassTexture,
    pub orm: GrassTexture,
    pub base_color_factor: [f32; 4],
    pub alpha_cutoff: f32,
    pub normal_scale: f32,
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub occlusion_strength: f32,
}

pub struct GrassAssets {
    pub models: Vec<GrassModel>,
    pub materials: Vec<GrassMaterial>,
}

/// One CPU allocation shared between Bevy's main world and render world.
#[derive(Resource, Clone)]
pub struct SharedGrassAssets(pub Arc<GrassAssets>);

impl GrassAssets {
    /// Load the extracted, normalized single-primitive GLBs in stable name order.
    /// Restricting discovery to the `grass-` prefix excludes the trees, bushes,
    /// rocks and other models that share the directory.
    pub fn load(models_directory: impl AsRef<Path>) -> Result<Self, String> {
        let directory = models_directory.as_ref();
        let mut paths = std::fs::read_dir(directory)
            .map_err(|e| format!("Reading grass directory {}: {e}", directory.display()))?
            .map(|entry| entry.map(|entry| entry.path()))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("Reading grass directory entry: {e}"))?;
        paths.retain(|path| {
            path.extension().is_some_and(|extension| extension == "glb")
                && path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with("grass-"))
        });
        paths.sort();
        if paths.is_empty() {
            return Err(format!(
                "No extracted grass GLBs in {}",
                directory.display()
            ));
        }

        let mut assets = Self {
            models: Vec::new(),
            materials: Vec::new(),
        };
        let mut materials_by_key = HashMap::<String, usize>::new();
        for path in paths {
            assets
                .load_model(&path, &mut materials_by_key)
                .map_err(|error| format!("Grass model {}: {error}", path.display()))?;
        }
        Ok(assets)
    }

    fn load_model(
        &mut self,
        path: &Path,
        materials_by_key: &mut HashMap<String, usize>,
    ) -> Result<(), String> {
        let gltf = gltf::Gltf::open(path).map_err(|e| e.to_string())?;
        let blob = gltf.blob.as_deref().ok_or("Expected a GLB binary buffer")?;
        let primitives: Vec<_> = gltf.meshes().flat_map(|mesh| mesh.primitives()).collect();
        if primitives.len() != 1 || primitives[0].mode() != gltf::mesh::Mode::Triangles {
            return Err("Expected exactly one triangle primitive per extracted grass model".into());
        }
        let primitive = &primitives[0];
        let reader = primitive.reader(|buffer| match buffer.source() {
            gltf::buffer::Source::Bin => Some(blob),
            gltf::buffer::Source::Uri(_) => None,
        });
        let positions: Vec<_> = reader
            .read_positions()
            .ok_or("Missing positions")?
            .collect();
        let normals: Vec<_> = reader.read_normals().ok_or("Missing normals")?.collect();
        let uvs: Vec<_> = reader
            .read_tex_coords(0)
            .ok_or("Missing UV0")?
            .into_f32()
            .collect();
        let indices: Vec<u32> = reader
            .read_indices()
            .map(|indices| indices.into_u32().collect())
            .unwrap_or_else(|| (0..positions.len() as u32).collect());
        if positions.is_empty()
            || normals.len() != positions.len()
            || uvs.len() != positions.len()
            || indices.is_empty()
            || indices.len() % 3 != 0
            || indices
                .iter()
                .any(|&index| index as usize >= positions.len())
        {
            return Err("Invalid grass vertex or triangle data".into());
        }
        let tangents: Vec<_> = reader
            .read_tangents()
            .map(|values| values.collect())
            .unwrap_or_else(|| generate_tangents(&positions, &normals, &uvs, &indices));
        if tangents.len() != positions.len() {
            return Err("Tangent count does not match vertex count".into());
        }
        let vertices: Vec<_> = (0..positions.len())
            .map(|i| GrassVertex {
                position: positions[i],
                normal: normals[i],
                uv: uvs[i],
                tangent: tangents[i],
            })
            .collect();
        if vertices.iter().any(|v| {
            v.position
                .iter()
                .chain(&v.normal)
                .chain(&v.uv)
                .chain(&v.tangent)
                .any(|v| !v.is_finite())
        }) {
            return Err("Non-finite grass vertex data".into());
        }

        let material = primitive.material();
        let pbr = material.pbr_metallic_roughness();
        let base_image = pbr
            .base_color_texture()
            .map(|texture| texture.texture().source());
        let normal_image = material
            .normal_texture()
            .map(|texture| texture.texture().source());
        let orm_image = pbr
            .metallic_roughness_texture()
            .map(|texture| texture.texture().source());
        let base_color_factor = pbr.base_color_factor();
        let alpha_cutoff = material.alpha_cutoff().unwrap_or(0.5);
        let normal_scale = material
            .normal_texture()
            .map_or(1.0, |texture| texture.scale());
        let occlusion_strength = material
            .occlusion_texture()
            .map_or(1.0, |texture| texture.strength());
        // Each extracted family shares its external maps. Include scalar factors
        // so an artist can change one material without accidentally reusing it.
        let key = format!(
            "{:?}|{:?}|{:?}|{:?}|{alpha_cutoff}|{normal_scale}|{}|{}|{occlusion_strength}",
            image_key(base_image.as_ref(), path),
            image_key(normal_image.as_ref(), path),
            image_key(orm_image.as_ref(), path),
            base_color_factor,
            pbr.metallic_factor(),
            pbr.roughness_factor()
        );
        let material_index = if let Some(&index) = materials_by_key.get(&key) {
            index
        } else {
            let index = self.materials.len();
            self.materials.push(GrassMaterial {
                base_color: load_texture(base_image, path, blob, [255, 255, 255, 255])?,
                normal: load_texture(normal_image, path, blob, [128, 128, 255, 255])?,
                orm: load_texture(orm_image, path, blob, [255, 255, 0, 255])?,
                base_color_factor,
                alpha_cutoff,
                normal_scale,
                metallic_factor: pbr.metallic_factor(),
                roughness_factor: pbr.roughness_factor(),
                occlusion_strength,
            });
            materials_by_key.insert(key, index);
            index
        };
        let minimum_y = positions.iter().map(|p| p[1]).fold(f32::INFINITY, f32::min);
        let maximum_y = positions
            .iter()
            .map(|p| p[1])
            .fold(f32::NEG_INFINITY, f32::max);
        let radius = positions
            .iter()
            .map(|p| p[0].hypot(p[2]))
            .fold(0.0, f32::max);
        self.models.push(GrassModel {
            name: path
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned(),
            vertices,
            indices,
            material: material_index,
            height: maximum_y - minimum_y,
            radius,
        });
        Ok(())
    }
}

fn image_key(image: Option<&gltf::Image<'_>>, model: &Path) -> String {
    match image.map(|image| image.source()) {
        Some(gltf::image::Source::Uri { uri, .. }) => model
            .parent()
            .unwrap_or(Path::new("."))
            .join(uri)
            .to_string_lossy()
            .into_owned(),
        Some(gltf::image::Source::View { view, .. }) => {
            format!("{}#{}", model.display(), view.index())
        }
        None => "default".into(),
    }
}

fn load_texture(
    image: Option<gltf::Image<'_>>,
    model: &Path,
    blob: &[u8],
    fallback: [u8; 4],
) -> Result<GrassTexture, String> {
    let Some(image) = image else {
        return Ok(GrassTexture {
            rgba: fallback.to_vec(),
            width: 1,
            height: 1,
        });
    };
    let image = match image.source() {
        gltf::image::Source::Uri { uri, .. } => {
            let path: PathBuf = model.parent().unwrap_or(Path::new(".")).join(uri);
            image::open(&path).map_err(|e| format!("Loading texture {}: {e}", path.display()))?
        }
        gltf::image::Source::View { view, .. } => {
            let bytes = blob
                .get(view.offset()..view.offset() + view.length())
                .ok_or("Texture buffer view out of bounds")?;
            image::load_from_memory(bytes).map_err(|e| format!("Loading embedded texture: {e}"))?
        }
    }
    .into_rgba8();
    Ok(GrassTexture {
        width: image.width(),
        height: image.height(),
        rgba: image.into_raw(),
    })
}

fn generate_tangents(
    positions: &[[f32; 3]],
    normals: &[[f32; 3]],
    uvs: &[[f32; 2]],
    indices: &[u32],
) -> Vec<[f32; 4]> {
    use bevy::math::{Vec2, Vec3};
    let mut tangent_sum = vec![Vec3::ZERO; positions.len()];
    let mut bitangent_sum = vec![Vec3::ZERO; positions.len()];
    for triangle in indices.chunks_exact(3) {
        let [a, b, c] = [
            triangle[0] as usize,
            triangle[1] as usize,
            triangle[2] as usize,
        ];
        let edge1 = Vec3::from(positions[b]) - Vec3::from(positions[a]);
        let edge2 = Vec3::from(positions[c]) - Vec3::from(positions[a]);
        let uv1 = Vec2::from(uvs[b]) - Vec2::from(uvs[a]);
        let uv2 = Vec2::from(uvs[c]) - Vec2::from(uvs[a]);
        let determinant = uv1.x * uv2.y - uv1.y * uv2.x;
        if determinant.abs() > 1e-8 {
            let tangent = (edge1 * uv2.y - edge2 * uv1.y) / determinant;
            let bitangent = (edge2 * uv1.x - edge1 * uv2.x) / determinant;
            for index in [a, b, c] {
                tangent_sum[index] += tangent;
                bitangent_sum[index] += bitangent;
            }
        }
    }
    normals
        .iter()
        .enumerate()
        .map(|(i, normal)| {
            let normal = Vec3::from(*normal).normalize_or(Vec3::Y);
            let tangent = (tangent_sum[i] - normal * normal.dot(tangent_sum[i]))
                .try_normalize()
                .unwrap_or_else(|| normal.any_orthonormal_vector());
            let handedness = if normal.cross(tangent).dot(bitangent_sum[i]) < 0.0 {
                -1.0
            } else {
                1.0
            };
            [tangent.x, tangent.y, tangent.z, handedness]
        })
        .collect()
}

impl GrassTexture {
    /// Average the visible blade texels in linear light. The render shader
    /// uses this to keep a grayscale atlas's texture contrast while letting
    /// the terrain, rather than atlas brightness, set each clump's mean color.
    pub fn visible_mean_luminance(&self, factor: [f32; 4], alpha_cutoff: f32) -> f32 {
        let mut weighted_luminance = 0.0;
        let mut weight_sum = 0.0;
        for pixel in self.rgba.chunks_exact(4) {
            let alpha = pixel[3] as f32 / 255.0 * factor[3];
            if alpha < alpha_cutoff {
                continue;
            }
            let linear = [
                srgb_to_linear(pixel[0] as f32 / 255.0) * factor[0],
                srgb_to_linear(pixel[1] as f32 / 255.0) * factor[1],
                srgb_to_linear(pixel[2] as f32 / 255.0) * factor[2],
            ];
            weighted_luminance +=
                (linear[0] * 0.2126 + linear[1] * 0.7152 + linear[2] * 0.0722) * alpha;
            weight_sum += alpha;
        }
        if weight_sum > 0.0 {
            weighted_luminance / weight_sum
        } else {
            1.0
        }
    }

    /// Mips filter color in linear light and premultiply cutout alpha to avoid
    /// dark fringes. Coverage preservation keeps thin blades visible in mips.
    pub fn mip_chain(&self, srgb: bool, alpha_cutoff: Option<f32>) -> Vec<Self> {
        assert!(self.width > 0 && self.height > 0);
        assert_eq!(
            self.rgba.len(),
            self.width as usize * self.height as usize * 4
        );
        let cutoff = alpha_cutoff.filter(|value| value.is_finite() && *value > 0.0 && *value < 1.0);
        let reference_coverage = cutoff.map(|cutoff| alpha_coverage(&self.rgba, cutoff, 1.0));
        let mut levels = vec![self.clone()];
        while levels
            .last()
            .is_some_and(|level| level.width > 1 || level.height > 1)
        {
            let previous = levels.last().unwrap();
            let width = (previous.width / 2).max(1);
            let height = (previous.height / 2).max(1);
            let mut rgba = vec![0; width as usize * height as usize * 4];
            for y in 0..height {
                for x in 0..width {
                    let mut sum = [0.0f32; 4];
                    let mut samples = 0.0;
                    for sy in y * previous.height / height..(y + 1) * previous.height / height {
                        for sx in x * previous.width / width..(x + 1) * previous.width / width {
                            let offset = ((sy * previous.width + sx) * 4) as usize;
                            let alpha = previous.rgba[offset + 3] as f32 / 255.0;
                            for channel in 0..3 {
                                let encoded = previous.rgba[offset + channel] as f32 / 255.0;
                                sum[channel] += if srgb {
                                    srgb_to_linear(encoded) * alpha
                                } else {
                                    encoded
                                };
                            }
                            sum[3] += alpha;
                            samples += 1.0;
                        }
                    }
                    let offset = ((y * width + x) * 4) as usize;
                    for channel in 0..3 {
                        let value = if srgb {
                            linear_to_srgb(sum[channel] / sum[3].max(1e-8))
                        } else {
                            sum[channel] / samples
                        };
                        rgba[offset + channel] = (value.clamp(0.0, 1.0) * 255.0).round() as u8;
                    }
                    rgba[offset + 3] = (sum[3] / samples * 255.0).round() as u8;
                }
            }
            if let (Some(cutoff), Some(reference)) = (cutoff, reference_coverage) {
                preserve_alpha_coverage(&mut rgba, cutoff, reference);
            }
            levels.push(Self {
                rgba,
                width,
                height,
            });
        }
        levels
    }
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

fn alpha_coverage(rgba: &[u8], cutoff: f32, scale: f32) -> f32 {
    rgba.chunks_exact(4)
        .filter(|pixel| ((pixel[3] as f32 * scale).round().min(255.0) / 255.0) >= cutoff)
        .count() as f32
        / (rgba.len() / 4) as f32
}

fn preserve_alpha_coverage(rgba: &mut [u8], cutoff: f32, target: f32) {
    let mut low = 0.0;
    let mut high = 16.0;
    let mut best_scale = 1.0;
    let mut best_error = (alpha_coverage(rgba, cutoff, 1.0) - target).abs();
    for _ in 0..16 {
        let scale = (low + high) * 0.5;
        let coverage = alpha_coverage(rgba, cutoff, scale);
        let error = (coverage - target).abs();
        if error < best_error {
            best_error = error;
            best_scale = scale;
        }
        if coverage < target {
            low = scale;
        } else {
            high = scale;
        }
    }
    for pixel in rgba.chunks_exact_mut(4) {
        pixel[3] = (pixel[3] as f32 * best_scale).round().clamp(0.0, 255.0) as u8;
    }
}

/// Nearest 8 m anchor. Flooring the shifted coordinate is also correct west
/// and south of the origin; truncating integer division is not.
pub fn scatter_anchor(center: [f32; 2]) -> [i64; 2] {
    center.map(|value| {
        (((value as f64 / SCATTER_ANCHOR_SIZE as f64) + 0.5).floor() as i64)
            .saturating_mul(SCATTER_ANCHOR_SIZE as i64)
    })
}

/// Generate about 1.07 million bounded candidates grouped by grass variant.
/// Each square metre has one persistent base candidate, three through the far
/// middle distance, eight through the close middle distance, and 64 short
/// blades in the foreground carpet. Every candidate retains an independent,
/// world-anchored identity.
/// Distance thinning and the final 235 m fade happen on the GPU, which avoids
/// camera-dependent CPU selection and repeated uploads within an anchor cell.
pub fn scatter_grass(center: [f32; 2], model_count: usize) -> Vec<Vec<GrassInstance>> {
    let mut groups = vec![Vec::new(); model_count];
    if model_count == 0
        || center
            .iter()
            .any(|value| !value.is_finite() || value.abs() > 1e9)
    {
        return groups;
    }
    let anchor = scatter_anchor(center);
    let cells = (CANDIDATE_RADIUS / SCATTER_CELL_SIZE).ceil() as i64;
    let anchor_cells =
        anchor.map(|coordinate| (coordinate as f64 / SCATTER_CELL_SIZE as f64).floor() as i64);
    for z in anchor_cells[1] - cells..=anchor_cells[1] + cells {
        for x in anchor_cells[0] - cells..=anchor_cells[0] + cells {
            // The closest possible point in this jitter cell gives a cheap,
            // conservative ring bound. Outer cells only generate their one
            // distant candidate; the five close-middle and 56 carpet extras
            // run only where they can contribute to the image.
            let cell_offset = [x - anchor_cells[0], z - anchor_cells[1]];
            let cell_min = cell_offset.map(|offset| {
                if offset > 0 {
                    offset as f32 * SCATTER_CELL_SIZE
                } else {
                    (-offset - 1).max(0) as f32 * SCATTER_CELL_SIZE
                }
            });
            let cell_min_squared = cell_min[0] * cell_min[0] + cell_min[1] * cell_min[1];
            if cell_min_squared > CANDIDATE_RADIUS * CANDIDATE_RADIUS {
                continue;
            }
            let subcandidate_count: u64 = if cell_min_squared
                <= CARPET_CANDIDATE_RADIUS * CARPET_CANDIDATE_RADIUS
            {
                64
            } else if cell_min_squared
                <= CLOSE_MIDDLE_CANDIDATE_RADIUS * CLOSE_MIDDLE_CANDIDATE_RADIUS
            {
                8
            } else if cell_min_squared <= FAR_MIDDLE_CANDIDATE_RADIUS * FAR_MIDDLE_CANDIDATE_RADIUS
            {
                3
            } else {
                1
            };
            let cell_key = cell_hash(x, z);
            for subcandidate in 0..subcandidate_count {
                let layer = match subcandidate {
                    0 => 0,
                    1..=2 => 1,
                    3..=7 => 2,
                    _ => 3,
                };
                let extra = layer != 0;
                let key = if extra {
                    mix64(cell_key ^ subcandidate.wrapping_mul(0xa076_1d64_78bd_642f))
                } else {
                    cell_key
                };
                // Full-cell jitter eliminates regular rows while retaining a
                // bounded number of candidates and a stable identity per tuft.
                let xz = [
                    (x as f64 + random(key, 0) as f64) as f32 * SCATTER_CELL_SIZE,
                    (z as f64 + random(key, 1) as f64) as f32 * SCATTER_CELL_SIZE,
                ];
                let distance_squared =
                    (xz[0] - anchor[0] as f32).powi(2) + (xz[1] - anchor[1] as f32).powi(2);
                let radius = match layer {
                    0 => CANDIDATE_RADIUS,
                    1 => FAR_MIDDLE_CANDIDATE_RADIUS,
                    2 => CLOSE_MIDDLE_CANDIDATE_RADIUS,
                    _ => CARPET_CANDIDATE_RADIUS,
                };
                if distance_squared > radius * radius {
                    continue;
                }
                let model = select_model(key, model_count, layer);
                let patch = 0.72 * patch_noise(xz, 10.0) + 0.28 * patch_noise(xz, 2.7);
                let patch = ((patch - 0.18) / 0.64).clamp(0.0, 1.0);
                let density = 0.65 + 0.35 * patch * patch * (3.0 - 2.0 * patch);
                groups[model].push(GrassInstance {
                    xz,
                    rotation: random(key, 2) * std::f32::consts::TAU,
                    scale: 0.78 + random(key, 3) * 0.46,
                    tint: 0.88 + random(key, 4) * 0.20,
                    seed: random(key, 5),
                    _pad: [density, layer as f32],
                });
            }
        }
    }
    groups
}

fn select_model(key: u64, model_count: usize, layer: u32) -> usize {
    if model_count != 9 {
        return (mix64(key ^ 0x678a_d119_af77) % model_count as u64) as usize;
    }
    // The grass models' sorted order is large, medium, small (three variants
    // each). Larger clumps make the sparse far field legible; root-level
    // carpet favours short groups so the foreground still looks like a meadow.
    let choice = random(key, 11);
    let (start, count) = if layer == 3 {
        if choice < 0.10 {
            (0, 3)
        } else if choice < 0.55 {
            (3, 3)
        } else {
            (6, 3)
        }
    } else if layer == 0 {
        if choice < 0.50 {
            (0, 3)
        } else if choice < 0.85 {
            (3, 3)
        } else {
            (6, 3)
        }
    } else if layer == 1 {
        if choice < 0.45 {
            (0, 3)
        } else if choice < 0.90 {
            (3, 3)
        } else {
            (6, 3)
        }
    } else if choice < 0.20 {
        (0, 3)
    } else if choice < 0.60 {
        (3, 3)
    } else {
        (6, 3)
    };
    start + (random(key, 12) * count as f32) as usize
}

/// Smooth world-space density patches, independent of the camera and model
/// selection. Two scales form meadow openings with smaller irregular clumps.
fn patch_noise(xz: [f32; 2], wavelength: f32) -> f32 {
    let coordinates = xz.map(|value| value as f64 / wavelength as f64);
    let cell = coordinates.map(|value| value.floor() as i64);
    let fraction = [
        (coordinates[0] - cell[0] as f64) as f32,
        (coordinates[1] - cell[1] as f64) as f32,
    ]
    .map(|value| value * value * (3.0 - 2.0 * value));
    let value = |dx, dz| random(cell_hash(cell[0] + dx, cell[1] + dz), 17);
    let south = value(0, 0) + (value(1, 0) - value(0, 0)) * fraction[0];
    let north = value(0, 1) + (value(1, 1) - value(0, 1)) * fraction[0];
    south + (north - south) * fraction[1]
}

fn cell_hash(x: i64, z: i64) -> u64 {
    mix64(
        (x as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
            ^ (z as u64).wrapping_mul(0xd1b5_4a32_d192_ed03)
            ^ 0x4752_4153_53,
    )
}

fn mix64(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn random(key: u64, channel: u64) -> f32 {
    (mix64(key ^ channel.wrapping_mul(0x9e37_79b9_7f4a_7c15)) >> 40) as f32 / 16_777_216.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracted_assets_load_with_shared_textures() {
        let assets = GrassAssets::load(Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/models"))
            .expect("Extracted grass GLBs and their external textures must load");
        assert_eq!(assets.models.len(), 9);
        assert_eq!(assets.materials.len(), 1);
        assert_eq!(
            assets
                .models
                .iter()
                .map(|model| model.indices.len() / 3)
                .sum::<usize>(),
            92
        );
        for model in &assets.models {
            assert!(model.height > 0.0 && model.radius > 0.0, "{}", model.name);
            assert!(model.material < assets.materials.len(), "{}", model.name);
            assert!(
                model
                    .vertices
                    .iter()
                    .all(|vertex| vertex.position[1] >= -1e-5),
                "{}",
                model.name
            );
        }
        for material in &assets.materials {
            for texture in [&material.base_color, &material.normal, &material.orm] {
                assert!(texture.width > 0 && texture.height > 0);
                assert_eq!(
                    texture.rgba.len(),
                    (texture.width * texture.height * 4) as usize
                );
            }
        }
    }

    #[test]
    fn negative_anchors_and_camera_motion_preserve_placement() {
        assert_eq!(scatter_anchor([-4.01, -12.01]), [-8, -16]);
        assert_eq!(scatter_anchor([-3.99, 3.99]), [0, 0]);
        assert_eq!(scatter_grass([-1.0, -1.0], 9), scatter_grass([3.9, 3.9], 9));
        let a = scatter_grass([-8.0, 0.0], 9);
        let b = scatter_grass([0.0, 0.0], 9);
        // Compare the overlap by stable position, including negative cells
        // with multiple candidates of the same variant.
        let mut overlap = 0;
        for (model, instances) in a.iter().enumerate() {
            let lookup: HashMap<_, _> = b[model]
                .iter()
                .map(|instance| {
                    (
                        (instance.xz[0].to_bits(), instance.xz[1].to_bits()),
                        instance,
                    )
                })
                .collect();
            for instance in instances {
                let cell = (instance.xz[0].to_bits(), instance.xz[1].to_bits());
                if let Some(other) = lookup.get(&cell) {
                    assert_eq!(instance, *other);
                    overlap += 1;
                }
            }
        }
        assert!(overlap > 500_000);
    }

    #[test]
    fn candidates_are_bounded_and_include_every_variant() {
        let groups = scatter_grass([-1000.0, 350.0], 9);
        let anchor = scatter_anchor([-1000.0, 350.0]);
        assert!(groups.iter().all(|group| !group.is_empty()));
        let total: usize = groups.iter().map(Vec::len).sum();
        assert!((1_060_000..1_090_000).contains(&total), "{total}");
        let mut central_cells = HashMap::<(i64, i64), usize>::new();
        let mut tier_cell_counts = [0usize; 4];
        let mut layer_counts = [0usize; 4];
        let mut large_far = 0usize;
        let mut large_medium_far = 0usize;
        let mut short_carpet = 0usize;
        for (model, instances) in groups.iter().enumerate() {
            for instance in instances {
                let layer = instance._pad[1] as usize;
                layer_counts[layer] += 1;
                if layer == 0 && model < 3 {
                    large_far += 1;
                }
                if layer == 1 && model < 6 {
                    large_medium_far += 1;
                }
                if layer == 3 && model >= 3 {
                    short_carpet += 1;
                }
                let radius = match layer {
                    0 => CANDIDATE_RADIUS,
                    1 => FAR_MIDDLE_CANDIDATE_RADIUS,
                    2 => CLOSE_MIDDLE_CANDIDATE_RADIUS,
                    _ => CARPET_CANDIDATE_RADIUS,
                };
                assert!(
                    (instance.xz[0] - anchor[0] as f32).hypot(instance.xz[1] - anchor[1] as f32)
                        <= radius + 0.001
                );
                assert!((0.78..1.24).contains(&instance.scale));
                assert!((0.0..1.0).contains(&instance.seed));
                assert!((0.65..=1.0).contains(&instance._pad[0]));
                let cell = (
                    instance.xz[0].floor() as i64 - anchor[0],
                    instance.xz[1].floor() as i64 - anchor[1],
                );
                match cell {
                    (2, 0) => tier_cell_counts[0] += 1,
                    (50, 0) => tier_cell_counts[1] += 1,
                    (160, 0) => tier_cell_counts[2] += 1,
                    (220, 0) => tier_cell_counts[3] += 1,
                    _ => {}
                }
                if (-10..10).contains(&cell.0) && (-10..10).contains(&cell.1) {
                    *central_cells.entry(cell).or_default() += 1;
                }
            }
        }
        assert!((180_000..190_000).contains(&layer_counts[0]));
        assert!((260_000..280_000).contains(&layer_counts[1]));
        assert!((330_000..350_000).contains(&layer_counts[2]));
        assert!((275_000..290_000).contains(&layer_counts[3]));
        assert!(large_far > layer_counts[0] * 45 / 100);
        assert!(large_medium_far > layer_counts[1] * 85 / 100);
        assert!(short_carpet > layer_counts[3] * 85 / 100);
        assert_eq!(tier_cell_counts, [64, 8, 3, 1]);
        assert_eq!(central_cells.len(), 400);
        // F32 world positions can round a jittered point onto a neighbouring
        // metre boundary far from the origin, shifting its observed bin.
        assert!(
            central_cells
                .values()
                .all(|&count| (62..=66).contains(&count))
        );
        assert!(scatter_grass([0.0, 0.0], 0).is_empty());
        assert!(scatter_grass([f32::NAN, 0.0], 9).iter().all(Vec::is_empty));
    }

    #[test]
    fn mip_filter_uses_linear_light_and_premultiplied_alpha() {
        let texture = GrassTexture {
            width: 2,
            height: 1,
            rgba: vec![255, 255, 255, 255, 0, 0, 0, 255],
        };
        let levels = texture.mip_chain(true, None);
        assert_eq!((levels[1].width, levels[1].height), (1, 1));
        assert!((187..=189).contains(&levels[1].rgba[0]));
        let texture = GrassTexture {
            width: 2,
            height: 1,
            rgba: vec![0, 255, 0, 255, 255, 0, 0, 0],
        };
        assert_eq!(&texture.mip_chain(true, None)[1].rgba[..3], &[0, 255, 0]);
    }

    #[test]
    fn mip_coverage_keeps_thin_blades_and_all_odd_edge_texels() {
        let mut rgba = vec![0u8; 8 * 4];
        for (pixel, alpha) in rgba
            .chunks_exact_mut(4)
            .zip([255, 255, 255, 0, 200, 0, 0, 0])
        {
            pixel.copy_from_slice(&[20, 180, 30, alpha]);
        }
        let texture = GrassTexture {
            width: 8,
            height: 1,
            rgba,
        };
        let levels = texture.mip_chain(true, Some(0.5));
        assert!((alpha_coverage(&levels[1].rgba, 0.5, 1.0) - 0.5).abs() <= 0.25);
        let odd = GrassTexture {
            width: 3,
            height: 1,
            rgba: vec![0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255, 255],
        };
        assert_eq!(odd.mip_chain(false, None)[1].rgba, [85, 85, 85, 255]);
    }
}
