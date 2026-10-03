//! Plant models: discovery by file name, GLB decoding and texture preparation.
//!
//! Every plant family in `assets/models` follows one naming scheme,
//! `{species}[-{form}]-{variant}-lod-{0..3}.glb`, with lavender as the single
//! unversioned `lavender.glb`. LODs 0-2 carry two primitives, an opaque bark
//! and an alpha-blended foliage card set; LOD 3 is a crossed-card billboard
//! with its own baked atlas. Everything is converted once, on a background
//! thread, into one shared vertex array, one index array and a deduplicated
//! texture list the render world uploads as they are.
//!
//! The source materials are glTF `BLEND`; the deferred G-buffer cannot blend,
//! so every card is alpha *tested* at 0.5 instead (the foliage alpha is all
//! but binary, see the coverage-preserving mips below). Normal, roughness and
//! ambient occlusion share one RGBA8 texture per material (normal XY,
//! roughness, AO), which halves the detail memory of the whole library.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// One interleaved vertex: the same 48-byte layout the grass uses.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PlantVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub tangent: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<PlantVertex>() == 48);

/// The plant families the scatter knows how to place.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Species {
    Fir,
    Pine,
    Oak,
    Maple,
    Lilac,
    Bush,
    Broadleaf,
    Lavender,
}

impl Species {
    pub const ALL: [Species; 8] = [
        Species::Fir,
        Species::Pine,
        Species::Oak,
        Species::Maple,
        Species::Lilac,
        Species::Bush,
        Species::Broadleaf,
        Species::Lavender,
    ];

    /// The file-name prefix. `lilac-bush` must be tried before `bush`.
    pub fn file_prefix(self) -> &'static str {
        match self {
            Species::Fir => "fir",
            Species::Pine => "pine",
            Species::Oak => "oak",
            Species::Maple => "maple",
            Species::Lilac => "lilac-bush",
            Species::Bush => "bush",
            Species::Broadleaf => "broadleaf-plant",
            Species::Lavender => "lavender",
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Species::Fir => "fir",
            Species::Pine => "pine",
            Species::Oak => "oak",
            Species::Maple => "maple",
            Species::Lilac => "lilac",
            Species::Bush => "bush",
            Species::Broadleaf => "broadleaf plant",
            Species::Lavender => "lavender",
        }
    }
}

/// How a primitive's material is shaded.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Surface {
    /// Opaque trunk and branch wood.
    Bark,
    /// Alpha-tested leaf, needle or flower cards.
    Foliage,
    /// Alpha-tested LOD-3 impostor cards with a baked atlas per model.
    Billboard,
}

/// A texture with its complete mip chain, ready to upload.
pub struct PreparedTexture {
    pub label: String,
    pub width: u32,
    pub height: u32,
    pub srgb: bool,
    /// Level 0 first; every level is tightly packed RGBA8.
    pub levels: Vec<Vec<u8>>,
}

impl PreparedTexture {
    pub fn byte_size(&self) -> usize {
        self.levels.iter().map(Vec::len).sum()
    }
}

pub struct PlantMaterial {
    pub surface: Surface,
    /// Index into [`VegetationAssets::textures`]: sRGB base colour + alpha.
    pub base_color: usize,
    /// Index into [`VegetationAssets::textures`]: linear normal XY,
    /// roughness, ambient occlusion.
    pub detail: usize,
    pub base_color_factor: [f32; 4],
    pub alpha_cutoff: f32,
    pub normal_scale: f32,
    pub roughness_factor: f32,
    pub occlusion_strength: f32,
    pub double_sided: bool,
}

/// One indexed draw over a LOD's shared vertex range.
#[derive(Clone, Copy, Debug)]
pub struct PlantPrimitive {
    pub material: usize,
    /// Into [`VegetationAssets::indices`].
    pub first_index: u32,
    pub index_count: u32,
}

pub struct PlantLod {
    /// Into [`VegetationAssets::vertices`]; the primitives' indices are
    /// relative to it.
    pub base_vertex: u32,
    pub primitives: Vec<PlantPrimitive>,
}

pub struct PlantModel {
    pub name: String,
    pub species: Species,
    /// `small`, `large`, `sapling`, ... or empty for a family with one form.
    pub form: String,
    pub lods: Vec<PlantLod>,
    /// Top of the LOD-0 mesh above its trunk-base pivot, metres.
    pub height: f32,
    /// Horizontal reach of the foliage from the trunk axis (90th percentile
    /// of the LOD-0 foliage vertices, so one stray twig does not set it).
    pub crown_radius: f32,
    /// Height of the lowest foliage (10th percentile), metres.
    pub crown_base: f32,
    /// Largest horizontal extent of any LOD, for conservative culling.
    pub bounding_radius: f32,
}

pub struct VegetationAssets {
    pub models: Vec<PlantModel>,
    pub materials: Vec<PlantMaterial>,
    pub textures: Vec<PreparedTexture>,
    pub vertices: Vec<PlantVertex>,
    pub indices: Vec<u32>,
}

impl VegetationAssets {
    pub fn texture_bytes(&self) -> usize {
        self.textures.iter().map(PreparedTexture::byte_size).sum()
    }
}

// ---------------------------------------------------------------------------
// Discovery
// ---------------------------------------------------------------------------

/// A model file name split into its parts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelName {
    pub species: Species,
    pub form: String,
    pub variant: u32,
    /// `None` for a model without LODs (lavender).
    pub lod: Option<u32>,
}

/// Parse `fir-large-2-lod-1`, `lilac-bush-3-lod-0`, `broadleaf-plant-5-lod-2`
/// or `lavender`. Rocks, grass and anything unrecognised return `None`.
pub fn parse_model_name(stem: &str) -> Option<ModelName> {
    let (body, lod) = match stem.rsplit_once("-lod-") {
        Some((body, lod)) => (body, Some(lod.parse::<u32>().ok()?)),
        None => (stem, None),
    };
    // Longest prefix first, so `lilac-bush-*` never reads as a `bush`.
    let mut species: Vec<Species> = Species::ALL.to_vec();
    species.sort_by_key(|s| std::cmp::Reverse(s.file_prefix().len()));
    let species = species.into_iter().find(|s| {
        body == s.file_prefix()
            || body
                .strip_prefix(s.file_prefix())
                .is_some_and(|rest| rest.starts_with('-'))
    })?;
    let rest = body[species.file_prefix().len()..].trim_start_matches('-');
    if rest.is_empty() {
        // Only an unversioned single model (lavender) has no variant number.
        return (lod.is_none()).then(|| ModelName {
            species,
            form: String::new(),
            variant: 1,
            lod: None,
        });
    }
    let (form, variant) = match rest.rsplit_once('-') {
        Some((form, variant)) => (form, variant),
        None => ("", rest),
    };
    let variant = variant.parse::<u32>().ok()?;
    Some(ModelName {
        species,
        form: form.to_string(),
        variant,
        lod,
    })
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

/// Texture resolution caps. The foliage atlases drive the look of the near
/// forest and keep 2048 texels; bark and the detail maps are read at a
/// coarser scale; billboards are only drawn beyond ~20 tree heights, where a
/// whole crown spans a few dozen pixels.
const FOLIAGE_BASE_MAX: u32 = 2048;
const FOLIAGE_DETAIL_MAX: u32 = 1024;
const BARK_MAX: u32 = 1024;
const BILLBOARD_BASE_MAX: u32 = 256;
const BILLBOARD_DETAIL_MAX: u32 = 128;
/// The foliage alpha is close to binary (under 2% of texels are partial for
/// every family but pine), so the classic half-coverage threshold keeps the
/// authored silhouettes.
pub const ALPHA_CUTOFF: f32 = 0.5;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum TextureKey {
    BaseColor { path: PathBuf, cap: u32, cutout: bool },
    Detail {
        normal: Option<PathBuf>,
        roughness: Option<PathBuf>,
        occlusion: Option<PathBuf>,
        cap: u32,
    },
}

#[derive(Clone, Debug, PartialEq)]
struct MaterialKey {
    surface: Surface,
    base_color: usize,
    detail: usize,
    factors: [u32; 8],
    double_sided: bool,
}

/// One primitive as read from a GLB, before textures are prepared.
struct RawPrimitive {
    vertices: Vec<PlantVertex>,
    indices: Vec<u32>,
    material: usize,
}

struct Loader {
    texture_keys: Vec<TextureKey>,
    texture_lookup: HashMap<TextureKey, usize>,
    materials: Vec<PlantMaterial>,
    material_keys: Vec<MaterialKey>,
}

impl Loader {
    fn texture(&mut self, key: TextureKey) -> usize {
        if let Some(&index) = self.texture_lookup.get(&key) {
            return index;
        }
        let index = self.texture_keys.len();
        self.texture_keys.push(key.clone());
        self.texture_lookup.insert(key, index);
        index
    }

    fn material(&mut self, material: PlantMaterial) -> usize {
        let key = MaterialKey {
            surface: material.surface,
            base_color: material.base_color,
            detail: material.detail,
            factors: [
                material.base_color_factor[0].to_bits(),
                material.base_color_factor[1].to_bits(),
                material.base_color_factor[2].to_bits(),
                material.base_color_factor[3].to_bits(),
                material.alpha_cutoff.to_bits(),
                material.normal_scale.to_bits(),
                material.roughness_factor.to_bits(),
                material.occlusion_strength.to_bits(),
            ],
            double_sided: material.double_sided,
        };
        if let Some(index) = self.material_keys.iter().position(|k| *k == key) {
            return index;
        }
        self.material_keys.push(key);
        self.materials.push(material);
        self.materials.len() - 1
    }

    /// Read one GLB's primitives, registering its materials and textures.
    fn read_glb(&mut self, path: &Path, lod: Option<u32>) -> Result<Vec<RawPrimitive>, String> {
        let gltf = gltf::Gltf::open(path).map_err(|e| e.to_string())?;
        let blob = gltf.blob.as_deref().ok_or("expected a GLB binary chunk")?;
        let directory = path.parent().unwrap_or(Path::new("."));
        let mut primitives = Vec::new();
        for mesh in gltf.meshes() {
            for primitive in mesh.primitives() {
                if primitive.mode() != gltf::mesh::Mode::Triangles {
                    return Err("only triangle lists are supported".into());
                }
                let reader = primitive.reader(|buffer| match buffer.source() {
                    gltf::buffer::Source::Bin => Some(blob),
                    gltf::buffer::Source::Uri(_) => None,
                });
                let positions: Vec<[f32; 3]> =
                    reader.read_positions().ok_or("missing positions")?.collect();
                let normals: Vec<[f32; 3]> =
                    reader.read_normals().ok_or("missing normals")?.collect();
                let uvs: Vec<[f32; 2]> = reader
                    .read_tex_coords(0)
                    .ok_or("missing UV0")?
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
                    || !indices.len().is_multiple_of(3)
                    || indices.iter().any(|&i| i as usize >= positions.len())
                {
                    return Err("invalid vertex or triangle data".into());
                }
                let tangents: Vec<[f32; 4]> = match reader.read_tangents() {
                    Some(tangents) => tangents.collect(),
                    None => crate::grass::generate_tangents(&positions, &normals, &uvs, &indices),
                };
                if tangents.len() != positions.len() {
                    return Err("tangent count does not match vertex count".into());
                }
                let vertices: Vec<PlantVertex> = (0..positions.len())
                    .map(|i| {
                        let n = normals[i];
                        let length = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
                        let normal = if length > 1e-6 {
                            [n[0] / length, n[1] / length, n[2] / length]
                        } else {
                            [0.0, 1.0, 0.0]
                        };
                        PlantVertex {
                            position: positions[i],
                            normal,
                            uv: uvs[i],
                            tangent: tangents[i],
                        }
                    })
                    .collect();
                if vertices.iter().any(|v| {
                    v.position
                        .iter()
                        .chain(&v.normal)
                        .chain(&v.uv)
                        .chain(&v.tangent)
                        .any(|value| !value.is_finite())
                }) {
                    return Err("non-finite vertex data".into());
                }
                let material = self.read_material(&primitive.material(), directory, lod)?;
                primitives.push(RawPrimitive {
                    vertices,
                    indices,
                    material,
                });
            }
        }
        if primitives.is_empty() {
            return Err("no primitives".into());
        }
        Ok(primitives)
    }

    fn read_material(
        &mut self,
        material: &gltf::Material<'_>,
        directory: &Path,
        lod: Option<u32>,
    ) -> Result<usize, String> {
        // Only UV0 is carried; a texture on another set (maple bark's
        // roughness) falls back to the material's constant factor.
        let uri_of = |texture: gltf::Texture<'_>, tex_coord: u32| -> Option<PathBuf> {
            if tex_coord != 0 {
                return None;
            }
            match texture.source().source() {
                gltf::image::Source::Uri { uri, .. } => Some(normalize_path(&directory.join(uri))),
                gltf::image::Source::View { .. } => None,
            }
        };
        let pbr = material.pbr_metallic_roughness();
        let base = pbr
            .base_color_texture()
            .and_then(|info| uri_of(info.texture(), info.tex_coord()))
            .ok_or("plant materials need an external base colour texture")?;
        let normal = material
            .normal_texture()
            .and_then(|info| uri_of(info.texture(), info.tex_coord()));
        let roughness = pbr
            .metallic_roughness_texture()
            .and_then(|info| uri_of(info.texture(), info.tex_coord()));
        let occlusion = material
            .occlusion_texture()
            .and_then(|info| uri_of(info.texture(), info.tex_coord()));
        let blended = material.alpha_mode() != gltf::material::AlphaMode::Opaque;
        let surface = if !blended {
            Surface::Bark
        } else if lod == Some(3) {
            Surface::Billboard
        } else {
            Surface::Foliage
        };
        let (base_cap, detail_cap) = match surface {
            Surface::Bark => (BARK_MAX, BARK_MAX),
            Surface::Foliage => (FOLIAGE_BASE_MAX, FOLIAGE_DETAIL_MAX),
            Surface::Billboard => (BILLBOARD_BASE_MAX, BILLBOARD_DETAIL_MAX),
        };
        let base_color = self.texture(TextureKey::BaseColor {
            path: base,
            cap: base_cap,
            cutout: blended,
        });
        let detail = self.texture(TextureKey::Detail {
            normal,
            roughness,
            occlusion: occlusion.clone(),
            cap: detail_cap,
        });
        Ok(self.material(PlantMaterial {
            surface,
            base_color,
            detail,
            base_color_factor: pbr.base_color_factor(),
            alpha_cutoff: if blended {
                material.alpha_cutoff().unwrap_or(ALPHA_CUTOFF)
            } else {
                0.0
            },
            normal_scale: material.normal_texture().map_or(1.0, |t| t.scale()),
            roughness_factor: pbr.roughness_factor(),
            occlusion_strength: if occlusion.is_some() {
                material.occlusion_texture().map_or(1.0, |t| t.strength())
            } else {
                0.0
            },
            // Every card is drawn two-sided: the LOD-3 billboards of several
            // packs leave doubleSided unset although their crossed planes are
            // single cards that must read from both sides.
            double_sided: true,
        }))
    }
}

/// Collapse `a/b/../c` so the same texture reached from different model
/// directories dedupes to one key.
fn normalize_path(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            std::path::Component::ParentDir => {
                if !out.pop() {
                    out.push("..");
                }
            }
            std::path::Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out
}

impl VegetationAssets {
    /// Load every plant model in `models_directory`. Rocks and grass are
    /// skipped. Fails if no plant model is found or any file is malformed.
    pub fn load(models_directory: impl AsRef<Path>) -> Result<Self, String> {
        Self::load_inner(models_directory.as_ref(), true)
    }

    /// Meshes and model metrics only, with no texture decoded: what the
    /// scatter and its tests need, in a fraction of the time.
    pub fn load_geometry(models_directory: impl AsRef<Path>) -> Result<Self, String> {
        Self::load_inner(models_directory.as_ref(), false)
    }

    fn load_inner(directory: &Path, with_textures: bool) -> Result<Self, String> {
        let mut entries: Vec<(ModelName, PathBuf)> = std::fs::read_dir(directory)
            .map_err(|e| format!("reading {}: {e}", directory.display()))?
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| path.extension().is_some_and(|extension| extension == "glb"))
            .filter_map(|path| {
                let stem = path.file_stem()?.to_str()?.to_string();
                parse_model_name(&stem).map(|name| (name, path))
            })
            .collect();
        entries.sort_by(|a, b| {
            (a.0.species, &a.0.form, a.0.variant, a.0.lod).cmp(&(b.0.species, &b.0.form, b.0.variant, b.0.lod))
        });
        if entries.is_empty() {
            return Err(format!("no plant models in {}", directory.display()));
        }

        let mut loader = Loader {
            texture_keys: Vec::new(),
            texture_lookup: HashMap::new(),
            materials: Vec::new(),
            material_keys: Vec::new(),
        };
        let mut models: Vec<PlantModel> = Vec::new();
        let mut vertices: Vec<PlantVertex> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();
        let mut index = 0;
        while index < entries.len() {
            let (name, _) = &entries[index];
            let group_end = entries[index..]
                .iter()
                .position(|(other, _)| {
                    (other.species, &other.form, other.variant) != (name.species, &name.form, name.variant)
                })
                .map_or(entries.len(), |offset| index + offset);
            let group = &entries[index..group_end];
            // LODs must run 0, 1, 2, ... without gaps (or be a single model).
            for (expected, (entry, path)) in group.iter().enumerate() {
                if entry.lod.is_some_and(|lod| lod as usize != expected) {
                    return Err(format!("{}: LODs must be numbered from 0 without gaps", path.display()));
                }
            }
            let model_name = match name.form.as_str() {
                "" => format!("{}-{}", name.species.file_prefix(), name.variant),
                form => format!("{}-{}-{}", name.species.file_prefix(), form, name.variant),
            };
            let model_name = if name.lod.is_none() {
                name.species.file_prefix().to_string()
            } else {
                model_name
            };
            let mut lods = Vec::new();
            let mut foliage_points: Vec<[f32; 3]> = Vec::new();
            let mut all_points_radius = 0.0f32;
            let mut top = 0.0f32;
            for (lod_index, (entry, path)) in group.iter().enumerate() {
                let primitives = loader
                    .read_glb(path, entry.lod)
                    .map_err(|e| format!("{}: {e}", path.display()))?;
                let base_vertex = vertices.len() as u32;
                let mut lod_primitives = Vec::new();
                let mut local_vertex = 0u32;
                for primitive in primitives {
                    let first_index = indices.len() as u32;
                    indices.extend(primitive.indices.iter().map(|&i| i + local_vertex));
                    for vertex in &primitive.vertices {
                        let p = vertex.position;
                        all_points_radius = all_points_radius.max(p[0].hypot(p[2]));
                        if lod_index == 0 {
                            top = top.max(p[1]);
                            if loader.materials[primitive.material].surface != Surface::Bark {
                                foliage_points.push(p);
                            }
                        }
                    }
                    local_vertex += primitive.vertices.len() as u32;
                    vertices.extend_from_slice(&primitive.vertices);
                    lod_primitives.push(PlantPrimitive {
                        material: primitive.material,
                        first_index,
                        index_count: primitive.indices.len() as u32,
                    });
                }
                lods.push(PlantLod {
                    base_vertex,
                    primitives: lod_primitives,
                });
            }
            let (crown_radius, crown_base) = crown_metrics(&foliage_points, top);
            models.push(PlantModel {
                name: model_name,
                species: name.species,
                form: name.form.clone(),
                lods,
                height: top.max(0.05),
                crown_radius,
                crown_base,
                bounding_radius: all_points_radius.max(0.05),
            });
            index = group_end;
        }

        let textures = if with_textures {
            prepare_textures(&loader.texture_keys)?
        } else {
            Vec::new()
        };
        Ok(Self {
            models,
            materials: loader.materials,
            textures,
            vertices,
            indices,
        })
    }
}

/// Crown reach and base from the LOD-0 foliage. Percentiles rather than
/// extremes, so one long branch card does not define the whole crown.
fn crown_metrics(points: &[[f32; 3]], top: f32) -> (f32, f32) {
    if points.is_empty() {
        return (0.3 * top.max(0.1), 0.0);
    }
    let mut radii: Vec<f32> = points.iter().map(|p| p[0].hypot(p[2])).collect();
    let mut heights: Vec<f32> = points.iter().map(|p| p[1]).collect();
    radii.sort_by(f32::total_cmp);
    heights.sort_by(f32::total_cmp);
    let radius = radii[(radii.len() * 9 / 10).min(radii.len() - 1)];
    let base = heights[heights.len() / 10].max(0.0);
    (radius.max(0.05), base.min(top))
}

// ---------------------------------------------------------------------------
// Texture preparation
// ---------------------------------------------------------------------------

struct Image {
    width: u32,
    height: u32,
    rgba: Vec<u8>,
}

fn decode(path: &Path) -> Result<Image, String> {
    let image = image::open(path)
        .map_err(|e| format!("loading texture {}: {e}", path.display()))?
        .into_rgba8();
    Ok(Image {
        width: image.width(),
        height: image.height(),
        rgba: image.into_raw(),
    })
}

/// Decode and prepare every texture, spreading the work over the available
/// cores: the library is ~150 PNGs, several of them 4096 square.
fn prepare_textures(keys: &[TextureKey]) -> Result<Vec<PreparedTexture>, String> {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};
    let next = AtomicUsize::new(0);
    let results: Mutex<Vec<Option<Result<PreparedTexture, String>>>> =
        Mutex::new((0..keys.len()).map(|_| None).collect());
    let workers = std::thread::available_parallelism()
        .map_or(4, |n| n.get())
        .clamp(1, 8);
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| {
                loop {
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    if index >= keys.len() {
                        break;
                    }
                    let prepared = prepare_texture(&keys[index]);
                    results.lock().unwrap_or_else(|e| e.into_inner())[index] = Some(prepared);
                }
            });
        }
    });
    results
        .into_inner()
        .unwrap_or_else(|e| e.into_inner())
        .into_iter()
        .map(|result| result.expect("every texture prepared"))
        .collect()
}

fn prepare_texture(key: &TextureKey) -> Result<PreparedTexture, String> {
    match key {
        TextureKey::BaseColor { path, cap, cutout } => {
            let image = decode(path)?;
            let levels = base_color_chain(&image, *cap, cutout.then_some(ALPHA_CUTOFF));
            Ok(PreparedTexture {
                label: path.file_name().map_or_else(String::new, |n| n.to_string_lossy().into_owned()),
                width: levels[0].width,
                height: levels[0].height,
                srgb: true,
                levels: levels.into_iter().map(|level| level.rgba).collect(),
            })
        }
        TextureKey::Detail {
            normal,
            roughness,
            occlusion,
            cap,
        } => {
            let normal = normal.as_deref().map(decode).transpose()?;
            let roughness_image = roughness.as_deref().map(decode).transpose()?;
            // Usually the occlusion is the R channel of the same ORM file the
            // roughness came from; decode it again only when it is not.
            let occlusion_image = match (occlusion, roughness) {
                (Some(o), Some(r)) if o == r => None,
                (Some(o), _) => Some(decode(o)?),
                (None, _) => None,
            };
            let occlusion_source = match (occlusion, roughness) {
                (Some(o), Some(r)) if o == r => roughness_image.as_ref(),
                (Some(_), _) => occlusion_image.as_ref(),
                (None, _) => None,
            };
            let packed = pack_detail(normal.as_ref(), roughness_image.as_ref(), occlusion_source);
            let levels = linear_chain(packed, *cap);
            let label = normal
                .as_ref()
                .and(key_label(key))
                .unwrap_or_else(|| "flat-detail".to_string());
            Ok(PreparedTexture {
                label,
                width: levels[0].width,
                height: levels[0].height,
                srgb: false,
                levels: levels.into_iter().map(|level| level.rgba).collect(),
            })
        }
    }
}

fn key_label(key: &TextureKey) -> Option<String> {
    match key {
        TextureKey::Detail { normal: Some(path), .. } => {
            path.file_name().map(|n| n.to_string_lossy().replace("normal", "detail"))
        }
        _ => None,
    }
}

/// Normal XY (tangent space, +Y up as glTF stores it), roughness (glTF G) and
/// occlusion (glTF R) in one RGBA8 texture at the normal map's resolution.
/// Missing maps contribute a flat normal, full roughness and no occlusion,
/// the glTF defaults, and the material factors are applied in the shader.
fn pack_detail(normal: Option<&Image>, roughness: Option<&Image>, occlusion: Option<&Image>) -> Image {
    let (width, height) = normal
        .map(|n| (n.width, n.height))
        .or_else(|| roughness.map(|r| (r.width, r.height)))
        .or_else(|| occlusion.map(|o| (o.width, o.height)))
        .unwrap_or((1, 1));
    let sample = |image: &Image, x: u32, y: u32, channel: usize| -> u8 {
        // Nearest texel of a map that may differ in resolution.
        let sx = (x as u64 * image.width as u64 / width as u64) as u32;
        let sy = (y as u64 * image.height as u64 / height as u64) as u32;
        image.rgba[((sy * image.width + sx) * 4) as usize + channel]
    };
    let mut rgba = vec![0u8; (width * height * 4) as usize];
    for y in 0..height {
        for x in 0..width {
            let offset = ((y * width + x) * 4) as usize;
            rgba[offset] = normal.map_or(128, |n| sample(n, x, y, 0));
            rgba[offset + 1] = normal.map_or(128, |n| sample(n, x, y, 1));
            rgba[offset + 2] = roughness.map_or(255, |r| sample(r, x, y, 1));
            rgba[offset + 3] = occlusion.map_or(255, |o| sample(o, x, y, 0));
        }
    }
    Image { width, height, rgba }
}

struct Level {
    width: u32,
    height: u32,
    rgba: Vec<u8>,
}

fn srgb_decode_table() -> &'static [f32; 256] {
    static TABLE: std::sync::OnceLock<[f32; 256]> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        std::array::from_fn(|i| {
            let value = i as f32 / 255.0;
            if value <= 0.04045 {
                value / 12.92
            } else {
                ((value + 0.055) / 1.055).powf(2.4)
            }
        })
    })
}

const ENCODE_STEPS: usize = 16384;

fn srgb_encode_table() -> &'static [u8] {
    static TABLE: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        (0..=ENCODE_STEPS)
            .map(|i| {
                let value = i as f32 / ENCODE_STEPS as f32;
                let encoded = if value <= 0.0031308 {
                    value * 12.92
                } else {
                    1.055 * value.powf(1.0 / 2.4) - 0.055
                };
                (encoded.clamp(0.0, 1.0) * 255.0).round() as u8
            })
            .collect()
    })
}

fn srgb_encode(linear: f32) -> u8 {
    srgb_encode_table()[(linear.clamp(0.0, 1.0) * ENCODE_STEPS as f32 + 0.5) as usize]
}

/// Average a level down by two in each axis (odd sizes keep every texel).
/// Colour is filtered in linear light and weighted by alpha, so transparent
/// texels never darken the fringe of a card.
fn downsample_base(level: &Level) -> Level {
    let decode = srgb_decode_table();
    let width = (level.width / 2).max(1);
    let height = (level.height / 2).max(1);
    let mut rgba = vec![0u8; (width * height * 4) as usize];
    for y in 0..height {
        let y0 = y * level.height / height;
        let y1 = ((y + 1) * level.height / height).max(y0 + 1);
        for x in 0..width {
            let x0 = x * level.width / width;
            let x1 = ((x + 1) * level.width / width).max(x0 + 1);
            let mut sum = [0.0f32; 4];
            let mut count = 0.0f32;
            for sy in y0..y1 {
                for sx in x0..x1 {
                    let offset = ((sy * level.width + sx) * 4) as usize;
                    let alpha = level.rgba[offset + 3] as f32 / 255.0;
                    sum[0] += decode[level.rgba[offset] as usize] * alpha;
                    sum[1] += decode[level.rgba[offset + 1] as usize] * alpha;
                    sum[2] += decode[level.rgba[offset + 2] as usize] * alpha;
                    sum[3] += alpha;
                    count += 1.0;
                }
            }
            let offset = ((y * width + x) * 4) as usize;
            if sum[3] > 1e-6 {
                for channel in 0..3 {
                    rgba[offset + channel] = srgb_encode(sum[channel] / sum[3]);
                }
            }
            rgba[offset + 3] = (sum[3] / count * 255.0).round() as u8;
        }
    }
    Level { width, height, rgba }
}

/// Plain box filter for linear data (the packed detail maps).
fn downsample_linear(level: &Level) -> Level {
    let width = (level.width / 2).max(1);
    let height = (level.height / 2).max(1);
    let mut rgba = vec![0u8; (width * height * 4) as usize];
    for y in 0..height {
        let y0 = y * level.height / height;
        let y1 = ((y + 1) * level.height / height).max(y0 + 1);
        for x in 0..width {
            let x0 = x * level.width / width;
            let x1 = ((x + 1) * level.width / width).max(x0 + 1);
            let mut sum = [0u32; 4];
            let mut count = 0u32;
            for sy in y0..y1 {
                for sx in x0..x1 {
                    let offset = ((sy * level.width + sx) * 4) as usize;
                    for (total, &value) in sum.iter_mut().zip(&level.rgba[offset..offset + 4]) {
                        *total += value as u32;
                    }
                    count += 1;
                }
            }
            let offset = ((y * width + x) * 4) as usize;
            for (texel, total) in rgba[offset..offset + 4].iter_mut().zip(sum) {
                *texel = ((total + count / 2) / count) as u8;
            }
        }
    }
    Level { width, height, rgba }
}

fn alpha_histogram(rgba: &[u8]) -> [u32; 256] {
    let mut histogram = [0u32; 256];
    for pixel in rgba.chunks_exact(4) {
        histogram[pixel[3] as usize] += 1;
    }
    histogram
}

/// Fraction of texels that pass the alpha test after scaling alpha by `scale`.
fn coverage(histogram: &[u32; 256], cutoff: f32, scale: f32) -> f32 {
    let total: u32 = histogram.iter().sum();
    let passing: u32 = histogram
        .iter()
        .enumerate()
        .filter(|(alpha, _)| ((*alpha as f32 * scale).round().min(255.0) / 255.0) >= cutoff)
        .map(|(_, count)| *count)
        .sum();
    passing as f32 / total.max(1) as f32
}

/// Scale a mip's alpha so the alpha test keeps the same fraction of texels as
/// the top level. Without it a thin card dissolves as it mips away, and a
/// distant forest turns transparent.
fn preserve_coverage(level: &mut Level, cutoff: f32, target: f32) {
    let histogram = alpha_histogram(&level.rgba);
    let (mut low, mut high) = (0.0f32, 16.0f32);
    let mut best = (1.0f32, (coverage(&histogram, cutoff, 1.0) - target).abs());
    for _ in 0..20 {
        let scale = 0.5 * (low + high);
        let achieved = coverage(&histogram, cutoff, scale);
        let error = (achieved - target).abs();
        if error < best.1 {
            best = (scale, error);
        }
        if achieved < target {
            low = scale;
        } else {
            high = scale;
        }
    }
    if (best.0 - 1.0).abs() > 1e-4 {
        for pixel in level.rgba.chunks_exact_mut(4) {
            pixel[3] = (pixel[3] as f32 * best.0).round().clamp(0.0, 255.0) as u8;
        }
    }
}

/// Fill the colour of fully transparent texels from their opaque
/// neighbourhood (pull-push). Bilinear filtering at the alpha-test edge then
/// blends toward the card's own colour instead of the black some packs store
/// behind the cutout.
fn fill_transparent(level: &mut Level) {
    // Pull: premultiplied averages down to 1x1.
    let mut pyramid: Vec<Level> = vec![Level {
        width: level.width,
        height: level.height,
        rgba: level.rgba.clone(),
    }];
    while pyramid.last().is_some_and(|l| l.width > 1 || l.height > 1) {
        let next = downsample_base(pyramid.last().unwrap());
        pyramid.push(next);
    }
    // Push: every texel without coverage takes its parent's colour.
    for index in (0..pyramid.len() - 1).rev() {
        let (fine, coarse) = pyramid.split_at_mut(index + 1);
        let fine = &mut fine[index];
        let coarse = &coarse[0];
        for y in 0..fine.height {
            let cy = (y as u64 * coarse.height as u64 / fine.height as u64) as u32;
            for x in 0..fine.width {
                let offset = ((y * fine.width + x) * 4) as usize;
                if fine.rgba[offset + 3] > 0 {
                    continue;
                }
                let cx = (x as u64 * coarse.width as u64 / fine.width as u64) as u32;
                let source = ((cy * coarse.width + cx) * 4) as usize;
                fine.rgba[offset..offset + 3].copy_from_slice(&coarse.rgba[source..source + 3]);
            }
        }
    }
    let filled = pyramid.swap_remove(0);
    for (pixel, source) in level.rgba.chunks_exact_mut(4).zip(filled.rgba.chunks_exact(4)) {
        if pixel[3] == 0 {
            pixel[..3].copy_from_slice(&source[..3]);
        }
    }
}

fn base_color_chain(image: &Image, cap: u32, cutoff: Option<f32>) -> Vec<Level> {
    let mut top = Level {
        width: image.width,
        height: image.height,
        rgba: image.rgba.clone(),
    };
    let reference = cutoff.map(|cutoff| coverage(&alpha_histogram(&top.rgba), cutoff, 1.0));
    while top.width > cap || top.height > cap {
        top = downsample_base(&top);
        if let (Some(cutoff), Some(reference)) = (cutoff, reference) {
            preserve_coverage(&mut top, cutoff, reference);
        }
    }
    if cutoff.is_some() {
        fill_transparent(&mut top);
    }
    let mut levels = vec![top];
    while levels.last().is_some_and(|l| l.width > 1 || l.height > 1) {
        let mut next = downsample_base(levels.last().unwrap());
        if let (Some(cutoff), Some(reference)) = (cutoff, reference) {
            preserve_coverage(&mut next, cutoff, reference);
        }
        levels.push(next);
    }
    levels
}

fn linear_chain(image: Image, cap: u32) -> Vec<Level> {
    let mut top = Level {
        width: image.width,
        height: image.height,
        rgba: image.rgba,
    };
    while top.width > cap || top.height > cap {
        top = downsample_linear(&top);
    }
    let mut levels = vec![top];
    while levels.last().is_some_and(|l| l.width > 1 || l.height > 1) {
        let next = downsample_linear(levels.last().unwrap());
        levels.push(next);
    }
    levels
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_names_parse_every_family() {
        let parse = |s| parse_model_name(s).unwrap();
        assert_eq!(parse("fir-large-2-lod-1").species, Species::Fir);
        assert_eq!(parse("fir-large-2-lod-1").form, "large");
        assert_eq!(parse("fir-large-2-lod-1").variant, 2);
        assert_eq!(parse("fir-large-2-lod-1").lod, Some(1));
        let lilac = parse("lilac-bush-3-lod-0");
        assert_eq!((lilac.species, lilac.form.as_str(), lilac.variant), (Species::Lilac, "", 3));
        let lilac_tree = parse("lilac-bush-single-tree-2-lod-3");
        assert_eq!(
            (lilac_tree.species, lilac_tree.form.as_str(), lilac_tree.lod),
            (Species::Lilac, "single-tree", Some(3))
        );
        let bush = parse("bush-medium-4-lod-2");
        assert_eq!((bush.species, bush.form.as_str()), (Species::Bush, "medium"));
        let broadleaf = parse("broadleaf-plant-5-lod-0");
        assert_eq!((broadleaf.species, broadleaf.form.as_str(), broadleaf.variant), (Species::Broadleaf, "", 5));
        let lavender = parse("lavender");
        assert_eq!((lavender.species, lavender.lod), (Species::Lavender, None));
        assert!(parse_model_name("rock-12").is_none());
        assert!(parse_model_name("grass-large-1").is_none());
        assert!(parse_model_name("fir-large-x-lod-0").is_none());
    }

    #[test]
    fn coverage_preserving_mips_keep_thin_cards() {
        // A leafy card: 30% of texels opaque in a scattered pattern. A plain
        // box filter averages its alpha to ~77/255, under the cutoff, so the
        // whole card would vanish from the first mips on.
        let width = 32u32;
        let mut rgba = vec![0u8; (width * width * 4) as usize];
        for (index, pixel) in rgba.chunks_exact_mut(4).enumerate() {
            let opaque = crate::vegetation::ecology::random(index as u64, 7) < 0.3;
            pixel.copy_from_slice(&[30, 160, 40, if opaque { 255 } else { 0 }]);
        }
        let image = Image { width, height: width, rgba };
        let reference = coverage(&alpha_histogram(&image.rgba), 0.5, 1.0);
        let levels = base_color_chain(&image, 32, Some(0.5));
        assert_eq!(levels.len(), 6);
        for level in levels.iter().filter(|level| level.width >= 4) {
            let kept = coverage(&alpha_histogram(&level.rgba), 0.5, 1.0);
            assert!((kept - reference).abs() < 0.15, "{}x{} kept {kept} of {reference}", level.width, level.height);
        }
        // Transparent texels inherit the card's colour, not black.
        let transparent = levels[0].rgba.chunks_exact(4).find(|p| p[3] == 0).unwrap();
        assert!(transparent[1] > 100, "{transparent:?}");
    }

    #[test]
    fn detail_packing_defaults_to_flat_rough_unoccluded() {
        let packed = pack_detail(None, None, None);
        assert_eq!((packed.width, packed.height), (1, 1));
        assert_eq!(packed.rgba, vec![128, 128, 255, 255]);
    }

    /// Loading every model decodes ~150 PNGs; run explicitly with
    /// `cargo test --release vegetation_library -- --ignored`.
    #[test]
    #[ignore = "decodes the whole plant library"]
    fn vegetation_library_loads() {
        let start = std::time::Instant::now();
        let assets = VegetationAssets::load(Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/models"))
            .expect("plant library loads");
        println!(
            "{} models, {} materials, {} textures ({:.1} MiB), {} vertices, {} triangles in {:.1?}",
            assets.models.len(),
            assets.materials.len(),
            assets.textures.len(),
            assets.texture_bytes() as f64 / (1024.0 * 1024.0),
            assets.vertices.len(),
            assets.indices.len() / 3,
            start.elapsed()
        );
        assert_eq!(assets.models.len(), 86);
        let mut textures: Vec<_> = assets.textures.iter().collect();
        textures.sort_by_key(|t| std::cmp::Reverse(t.byte_size()));
        for texture in textures.iter().take(24) {
            println!(
                "{:56} {:5}x{:<5} {:6.2} MiB",
                texture.label,
                texture.width,
                texture.height,
                texture.byte_size() as f64 / (1024.0 * 1024.0)
            );
        }
        for model in &assets.models {
            let expected = if model.species == Species::Lavender { 1 } else { 4 };
            assert_eq!(model.lods.len(), expected, "{}", model.name);
            assert!(model.height > 0.2 && model.crown_radius > 0.05, "{}", model.name);
            println!(
                "{:28} h={:5.2} crown r={:5.2} base={:5.2} tris={:?}",
                model.name,
                model.height,
                model.crown_radius,
                model.crown_base,
                model
                    .lods
                    .iter()
                    .map(|lod| lod.primitives.iter().map(|p| p.index_count / 3).sum::<u32>())
                    .collect::<Vec<_>>()
            );
        }
    }
}
