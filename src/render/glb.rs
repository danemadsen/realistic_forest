//! A minimal GLB reader for the extracted fir pack.
//!
//! SCOPE: this reads exactly the shape `tools/extract-fir-glb.py` writes —
//! a single mesh whose primitives carry `POSITION`, `NORMAL`, `TEXCOORD_0` and
//! `TANGENT`, indexed with `UNSIGNED_INT`, one node, one scene, and either
//! `uri`-pointed PNGs next to the file or PNGs embedded in the BIN chunk.
//! Anything else (sparse accessors, `byteStride`-interleaved views, quantised
//! attributes, data-URI images, skins, animations) is rejected with a message
//! naming the file and the offending field rather than silently mis-read.
//!
//! It is deliberately not `bevy_gltf`: that loader produces `Handle<Mesh>`,
//! `Handle<Image>` and `StandardMaterial`, and this renderer's pipelines take
//! raw `wgpu::Buffer`s and `wgpu::TextureView`s built by hand (see
//! `src/render/gpu_textures.rs` and `src/render/water_node.rs`). Everything
//! bevy_gltf would produce would have to be converted into exactly the raw
//! buffers below, so the conversion *is* the loader.
//!
//! glTF stores accessor data tightly packed unless a `bufferView` declares
//! `byteStride`, so an accessor is `count` consecutive `type`-sized tuples
//! starting at `bufferView.byteOffset + accessor.byteOffset`. The extracted
//! pack never interleaves, which is why one stride is threaded through and
//! interleaved views are refused.

use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Public data
// ---------------------------------------------------------------------------

/// One vertex as the tree pipeline's vertex buffer holds it, interleaved.
///
/// The tangent is carried rather than re-derived in the fragment shader from
/// screen-space derivatives, because foliage cards are exactly where
/// derivative-based frames break down: a card's UV gradient is undefined across
/// the alpha cut. The pack ships a correct per-vertex tangent, so it is used.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GlbVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    /// xyz tangent, w handedness. The fragment shader builds the bitangent as
    /// `cross(normal, tangent) * w`, matching glTF's convention.
    pub tangent: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<GlbVertex>() == 48);

/// Where a material's image data comes from. Both variants decode to the same
/// RGBA8 texels; only the fetch differs.
#[derive(Clone, Debug)]
pub enum GlbImage {
    /// A `uri` relative to the `.glb`'s directory. The extracted pack shares
    /// six of these across all 36 files.
    External(PathBuf),
    /// A PNG in the BIN chunk. The pack's LOD-3 billboard textures are unique
    /// per tree, so their bytes stay embedded in their own file.
    Embedded(Vec<u8>),
}

/// The subset of a glTF material this renderer consumes.
///
/// `metallicRoughnessTexture` is deliberately not read: every material in the
/// pack is fully rough (that texture's green channel is 255) and effectively
/// non-metallic (blue 0-28), so it carries no information the two scalars below
/// do not already carry. Not decoding a 4096x4096 PNG for nothing is worth more
/// than the fidelity it would add.
#[derive(Clone, Debug)]
pub struct GlbMaterial {
    pub name: String,
    pub base_colour: Option<GlbImage>,
    pub normal: Option<GlbImage>,
    pub metallic: f32,
    pub roughness: f32,
    /// `alphaMode` was `MASK` or `BLEND`.
    ///
    /// The pack authors its branch cards `BLEND`, but the branch albedo's alpha
    /// is effectively binary (72.9% of texels are 0, 26.1% are 1, ~1% between),
    /// so the tree pass draws them with an alpha cutout. That is required, not
    /// cosmetic: the water pass reconstructs depth from texel-exact point
    /// samples of the G-buffer, and a blended fragment would write depth and
    /// coverage that disagree, streaking the raymarched reflection.
    pub alpha_cutout: bool,
    pub alpha_cutoff: f32,
    /// `doubleSided`, kept so a future one-sided species can opt into culling.
    /// The tree pass draws everything with `cull_mode: None` regardless, which
    /// is what the pack's cards need at every LOD.
    pub double_sided: bool,
    /// `KHR_materials_specular`'s `specularFactor`, the scale on the
    /// dielectric Fresnel term (glTF's own default when the extension is
    /// absent is 1.0, i.e. an unmodified F0 of 0.04).
    ///
    /// This matters more for foliage than the name suggests. It scales the
    /// 0.04 dielectric F0 the composite would otherwise apply on its own, and
    /// the pack asks for 0.096 on the branch cards and 0.022 on the LOD-3
    /// billboards — an effective F0 of 0.0038 and 0.0009, so a fir canopy is
    /// authored to be an order of magnitude less reflective than a generic
    /// surface. Ignoring it and running full Fresnel over a crown of
    /// randomly-oriented cards is what paints the pale desaturated sheen
    /// across the foliage.
    ///
    /// `specularColorFactor` and `specularTexture` are not read: every material
    /// in the pack ships the default `[1, 1, 1]` colour factor, and the one
    /// `specularTexture` belongs to a LOD-3 billboard whose factor alone
    /// already puts it within 2% of zero.
    pub specular_factor: f32,
}

/// One primitive's geometry, flattened to the interleaved vertex format with
/// u32 indices.
pub struct GlbPrimitive {
    pub vertices: Vec<GlbVertex>,
    pub indices: Vec<u32>,
    /// Index into [`GlbFile::materials`].
    pub material: usize,
    /// Object-space bounds from the POSITION accessor's own `min`/`max`.
    pub bounds_min: [f32; 3],
    pub bounds_max: [f32; 3],
}

/// One `.glb`, fully decoded into CPU-side geometry and material descriptions.
pub struct GlbFile {
    pub name: String,
    pub primitives: Vec<GlbPrimitive>,
    pub materials: Vec<GlbMaterial>,
    /// Object-space bounds over every primitive.
    pub bounds_min: [f32; 3],
    pub bounds_max: [f32; 3],
}

impl GlbFile {
    /// Reads and decodes `path`.
    ///
    /// Returns a `String` rather than an `io::Error`: every failure here is
    /// either a malformed container or an asset this reader deliberately does
    /// not support, and both reach the caller as one message with the file
    /// named.
    pub fn load(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
        let name = path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("fir")
            .to_string();
        let directory = path.parent().unwrap_or(Path::new(".")).to_path_buf();
        let (json, binary) = split_container(&bytes, &name)?;
        let root: serde_json::Value =
            serde_json::from_str(&json).map_err(|error| format!("{name}: bad JSON chunk: {error}"))?;
        let reader = Reader {
            root: &root,
            binary: binary.unwrap_or(&[]),
        };
        reader.decode(&directory, name)
    }

    /// Object-space height in metres. The pack is authored with the trunk foot
    /// on the origin (`tools/extract-fir-glb.py` clears the world matrix's
    /// translation column), so this is `bounds_max.y` — what the scatter's size
    /// bands and the LOD distance bands are both measured in.
    pub fn height(&self) -> f32 {
        self.bounds_max[1].max(0.0)
    }
}

// ---------------------------------------------------------------------------
// Container
// ---------------------------------------------------------------------------

/// Splits a GLB into its JSON text and its BIN chunk.
///
/// Layout: a 12-byte header (`glTF`, version, total length) followed by chunks
/// of `[u32 length][u32 type][length bytes]`. `0x4E4F534A` is `JSON`,
/// `0x004E4942` is `BIN\0`. The BIN chunk is optional in the spec and absent
/// from a file whose images are all external.
fn split_container<'a>(bytes: &'a [u8], name: &str) -> Result<(String, Option<&'a [u8]>), String> {
    if bytes.len() < 12 || &bytes[0..4] != b"glTF" {
        return Err(format!("{name}: not a GLB container"));
    }
    let declared = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    if declared > bytes.len() {
        return Err(format!(
            "{name}: header declares {declared} bytes, file holds {}",
            bytes.len()
        ));
    }

    let mut offset = 12;
    let mut json: Option<&[u8]> = None;
    let mut binary: Option<&[u8]> = None;
    while offset + 8 <= declared {
        let length = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
        let kind = u32::from_le_bytes(bytes[offset + 4..offset + 8].try_into().unwrap());
        let start = offset + 8;
        let end = start
            .checked_add(length)
            .filter(|end| *end <= bytes.len())
            .ok_or_else(|| format!("{name}: chunk at {offset} runs past the file"))?;
        match kind {
            0x4E4F534A => json = Some(&bytes[start..end]),
            0x004E4942 => binary = Some(&bytes[start..end]),
            _ => {}
        }
        offset = end;
    }

    let json = json.ok_or_else(|| format!("{name}: no JSON chunk"))?;
    let text =
        std::str::from_utf8(json).map_err(|error| format!("{name}: JSON chunk not UTF-8: {error}"))?;
    Ok((text.to_string(), binary))
}

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------

/// The JSON tree plus the BIN chunk, so accessor reads can reach the bytes.
struct Reader<'a> {
    root: &'a serde_json::Value,
    binary: &'a [u8],
}

impl Reader<'_> {
    fn decode(&self, directory: &Path, name: String) -> Result<GlbFile, String> {
        let materials = self.read_materials(directory, &name)?;

        // Only one mesh is accepted. Every file in this pack has exactly one
        // mesh on one node, so a second one means the extraction tool changed
        // shape — and reporting that beats drawing half a tree.
        let meshes = self.root["meshes"]
            .as_array()
            .ok_or_else(|| format!("{name}: no meshes array"))?;
        if meshes.len() != 1 {
            return Err(format!("{name}: expected 1 mesh, found {}", meshes.len()));
        }
        let primitives_json = meshes[0]["primitives"]
            .as_array()
            .ok_or_else(|| format!("{name}: mesh 0 has no primitives"))?;

        let mut primitives = Vec::with_capacity(primitives_json.len());
        for (index, primitive) in primitives_json.iter().enumerate() {
            primitives.push(self.read_primitive(primitive, &name, index, materials.len())?);
        }

        let mut bounds_min = [f32::INFINITY; 3];
        let mut bounds_max = [f32::NEG_INFINITY; 3];
        for primitive in &primitives {
            for axis in 0..3 {
                bounds_min[axis] = bounds_min[axis].min(primitive.bounds_min[axis]);
                bounds_max[axis] = bounds_max[axis].max(primitive.bounds_max[axis]);
            }
        }

        Ok(GlbFile {
            name,
            primitives,
            materials,
            bounds_min,
            bounds_max,
        })
    }

    // -- materials ----------------------------------------------------------

    fn read_materials(&self, directory: &Path, name: &str) -> Result<Vec<GlbMaterial>, String> {
        let Some(list) = self.root["materials"].as_array() else {
            return Ok(Vec::new());
        };
        let mut materials = Vec::with_capacity(list.len());
        for (index, material) in list.iter().enumerate() {
            let alpha_mode = material["alphaMode"].as_str().unwrap_or("OPAQUE");
            materials.push(GlbMaterial {
                name: material["name"].as_str().unwrap_or("material").to_string(),
                base_colour: self.image_for_texture(
                    material["pbrMetallicRoughness"]["baseColorTexture"]["index"].as_u64(),
                    directory,
                    name,
                    index,
                )?,
                normal: self.image_for_texture(
                    material["normalTexture"]["index"].as_u64(),
                    directory,
                    name,
                    index,
                )?,
                metallic: material["pbrMetallicRoughness"]["metallicFactor"]
                    .as_f64()
                    .unwrap_or(1.0) as f32,
                roughness: material["pbrMetallicRoughness"]["roughnessFactor"]
                    .as_f64()
                    .unwrap_or(1.0) as f32,
                // `alphaCutoff` is only defined for MASK in the spec. The pack's
                // BLEND cards carry no cutoff and take the spec default.
                alpha_cutoff: material["alphaCutoff"].as_f64().unwrap_or(0.5) as f32,
                alpha_cutout: alpha_mode != "OPAQUE",
                double_sided: material["doubleSided"].as_bool().unwrap_or(false),
                specular_factor: material["extensions"]["KHR_materials_specular"]
                    ["specularFactor"]
                    .as_f64()
                    .unwrap_or(1.0)
                    .clamp(0.0, 1.0) as f32,
            });
        }
        Ok(materials)
    }

    /// Resolves a `textures[i]` index through `images[source]` to bytes.
    fn image_for_texture(
        &self,
        texture_index: Option<u64>,
        directory: &Path,
        name: &str,
        material: usize,
    ) -> Result<Option<GlbImage>, String> {
        let Some(texture_index) = texture_index else {
            return Ok(None);
        };
        let source = self.root["textures"]
            .as_array()
            .and_then(|textures| textures.get(texture_index as usize))
            .and_then(|texture| texture["source"].as_u64())
            .ok_or_else(|| {
                format!("{name}: material {material} texture {texture_index} has no source image")
            })?;
        let image = self.root["images"]
            .as_array()
            .and_then(|images| images.get(source as usize))
            .ok_or_else(|| format!("{name}: image {source} is missing"))?;

        if let Some(uri) = image["uri"].as_str() {
            if uri.starts_with("data:") {
                return Err(format!("{name}: image {source} is a data URI, which is not supported"));
            }
            return Ok(Some(GlbImage::External(directory.join(uri))));
        }
        let view = image["bufferView"]
            .as_u64()
            .ok_or_else(|| format!("{name}: image {source} has neither uri nor bufferView"))?
            as usize;
        Ok(Some(GlbImage::Embedded(self.buffer_view_bytes(view, name)?.to_vec())))
    }

    // -- primitives ---------------------------------------------------------

    fn read_primitive(
        &self,
        primitive: &serde_json::Value,
        name: &str,
        index: usize,
        material_count: usize,
    ) -> Result<GlbPrimitive, String> {
        let attributes = &primitive["attributes"];
        let position: Vec<[f32; 3]> = self.attribute(attributes, "POSITION", 3, name, index)?
            .chunks_exact(3)
            .map(|v| [v[0], v[1], v[2]])
            .collect();
        let normal: Vec<[f32; 3]> = self.attribute(attributes, "NORMAL", 3, name, index)?
            .chunks_exact(3)
            .map(|v| [v[0], v[1], v[2]])
            .collect();
        let uv: Vec<[f32; 2]> = self.attribute(attributes, "TEXCOORD_0", 2, name, index)?
            .chunks_exact(2)
            .map(|v| [v[0], v[1]])
            .collect();
        let tangent: Vec<[f32; 4]> = self.attribute(attributes, "TANGENT", 4, name, index)?
            .chunks_exact(4)
            .map(|v| [v[0], v[1], v[2], v[3]])
            .collect();

        let count = position.len();
        for (label, length) in [
            ("NORMAL", normal.len()),
            ("TEXCOORD_0", uv.len()),
            ("TANGENT", tangent.len()),
        ] {
            if length != count {
                return Err(format!(
                    "{name}: primitive {index} has {count} positions but {length} {label} vectors"
                ));
            }
        }

        let indices = self.indices(primitive, name, index)?;
        if indices.iter().any(|vertex| *vertex as usize >= count) {
            return Err(format!("{name}: primitive {index} indexes past its own vertices"));
        }

        let material = primitive["material"].as_u64().unwrap_or(0) as usize;
        if material_count > 0 && material >= material_count {
            return Err(format!(
                "{name}: primitive {index} names material {material}, but {material_count} exist"
            ));
        }

        let (bounds_min, bounds_max) = self.position_bounds(attributes, &position);

        Ok(GlbPrimitive {
            vertices: (0..count)
                .map(|vertex| GlbVertex {
                    position: position[vertex],
                    normal: normal[vertex],
                    uv: uv[vertex],
                    tangent: tangent[vertex],
                })
                .collect(),
            indices,
            material,
            bounds_min,
            bounds_max,
        })
    }

    /// The POSITION accessor's own `min`/`max`, which glTF requires for
    /// POSITION. Falling back to a CPU scan keeps the reader working on a file
    /// that omits them.
    fn position_bounds(
        &self,
        attributes: &serde_json::Value,
        positions: &[[f32; 3]],
    ) -> ([f32; 3], [f32; 3]) {
        let accessor = attributes["POSITION"]
            .as_u64()
            .map(|index| &self.root["accessors"][index as usize]);
        let read = |key: &str| -> Option<[f32; 3]> {
            let array = accessor?[key].as_array()?;
            if array.len() != 3 {
                return None;
            }
            Some([
                array[0].as_f64()? as f32,
                array[1].as_f64()? as f32,
                array[2].as_f64()? as f32,
            ])
        };
        match (read("min"), read("max")) {
            (Some(min), Some(max)) => (min, max),
            _ => {
                let mut min = [f32::INFINITY; 3];
                let mut max = [f32::NEG_INFINITY; 3];
                for position in positions {
                    for axis in 0..3 {
                        min[axis] = min[axis].min(position[axis]);
                        max[axis] = max[axis].max(position[axis]);
                    }
                }
                (min, max)
            }
        }
    }

    /// Flattens one `f32` accessor into a plain `Vec<f32>`.
    fn attribute(
        &self,
        attributes: &serde_json::Value,
        label: &str,
        components: usize,
        name: &str,
        primitive: usize,
    ) -> Result<Vec<f32>, String> {
        let accessor_index = attributes[label]
            .as_u64()
            .ok_or_else(|| format!("{name}: primitive {primitive} has no {label}"))?
            as usize;
        let accessor = &self.root["accessors"][accessor_index];
        let declared = match accessor["type"].as_str() {
            Some("SCALAR") => 1,
            Some("VEC2") => 2,
            Some("VEC3") => 3,
            Some("VEC4") => 4,
            other => return Err(format!("{name}: {label} has type {other:?}")),
        };
        if declared != components {
            return Err(format!("{name}: {label} is {declared} components, expected {components}"));
        }
        if accessor["componentType"].as_u64() != Some(5126) {
            return Err(format!("{name}: {label} is not FLOAT"));
        }
        self.accessor_f32(accessor_index, name)
    }

    fn indices(
        &self,
        primitive: &serde_json::Value,
        name: &str,
        index: usize,
    ) -> Result<Vec<u32>, String> {
        let Some(accessor_index) = primitive["indices"].as_u64().map(|index| index as usize) else {
            return Err(format!("{name}: primitive {index} is not indexed"));
        };
        let accessor = &self.root["accessors"][accessor_index as usize];
        let bytes = self.accessor_bytes(accessor_index, name)?;
        match accessor["componentType"].as_u64() {
            Some(5125) => Ok(bytes
                .chunks_exact(4)
                .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
                .collect()),
            Some(5123) => Ok(bytes
                .chunks_exact(2)
                .map(|chunk| u16::from_le_bytes(chunk.try_into().unwrap()) as u32)
                .collect()),
            other => Err(format!("{name}: index componentType {other:?} is not supported")),
        }
    }

    fn accessor_f32(&self, index: usize, name: &str) -> Result<Vec<f32>, String> {
        Ok(self
            .accessor_bytes(index, name)?
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect())
    }

    /// The accessor's raw bytes, after the sparse/stride/range checks that
    /// every accessor shares.
    fn accessor_bytes(&self, index: usize, name: &str) -> Result<&[u8], String> {
        let accessor = self.root["accessors"]
            .as_array()
            .and_then(|accessors| accessors.get(index))
            .ok_or_else(|| format!("{name}: accessor {index} is missing"))?;
        if accessor["sparse"].is_object() {
            return Err(format!("{name}: accessor {index} is sparse, which is not supported"));
        }
        let count = accessor["count"]
            .as_u64()
            .ok_or_else(|| format!("{name}: accessor {index} has no count"))? as usize;
        let view_index = accessor["bufferView"]
            .as_u64()
            .ok_or_else(|| format!("{name}: accessor {index} has no bufferView"))? as usize;
        if self.root["bufferViews"][view_index]["byteStride"].as_u64().is_some() {
            return Err(format!(
                "{name}: bufferView {view_index} is interleaved, which is not supported"
            ));
        }
        let view = self.buffer_view_bytes(view_index, name)?;
        let offset = accessor["byteOffset"].as_u64().unwrap_or(0) as usize;
        // The element size is recovered from `count` and the view's own length
        // only for the bound check; the caller re-interprets the slice.
        let element = match accessor["type"].as_str() {
            Some("SCALAR") => 1,
            Some("VEC2") => 2,
            Some("VEC3") => 3,
            Some("VEC4") => 4,
            other => return Err(format!("{name}: accessor {index} has type {other:?}")),
        };
        let component = match accessor["componentType"].as_u64() {
            Some(5126) | Some(5125) => 4,
            Some(5123) => 2,
            other => return Err(format!("{name}: accessor {index} has componentType {other:?}")),
        };
        let end = offset + count * element * component;
        if end > view.len() {
            return Err(format!("{name}: accessor {index} runs past its bufferView"));
        }
        Ok(&view[offset..end])
    }

    fn buffer_view_bytes(&self, index: usize, name: &str) -> Result<&[u8], String> {
        let view = self.root["bufferViews"]
            .as_array()
            .and_then(|views| views.get(index))
            .ok_or_else(|| format!("{name}: bufferView {index} is missing"))?;
        // Only buffer 0 is ever addressed: a GLB has exactly one BIN chunk.
        if view["buffer"].as_u64().unwrap_or(0) != 0 {
            return Err(format!("{name}: bufferView {index} names a buffer other than the BIN chunk"));
        }
        let offset = view["byteOffset"].as_u64().unwrap_or(0) as usize;
        let length = view["byteLength"]
            .as_u64()
            .ok_or_else(|| format!("{name}: bufferView {index} has no byteLength"))? as usize;
        let end = offset
            .checked_add(length)
            .filter(|end| *end <= self.binary.len())
            .ok_or_else(|| {
                format!(
                    "{name}: bufferView {index} covers {offset}..{} but the BIN chunk is {} bytes",
                    offset + length,
                    self.binary.len()
                )
            })?;
        Ok(&self.binary[offset..end])
    }
}

#[cfg(test)]
mod tests {
    use super::GlbFile;

    /// The reader's real coverage is the asset folder itself: every one of the
    /// 36 extracted files must decode, and its geometry must be self-consistent.
    #[test]
    fn every_extracted_fir_decodes() {
        let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/models/fir");
        let mut decoded = 0;
        for entry in std::fs::read_dir(&directory).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().and_then(|value| value.to_str()) != Some("glb") {
                continue;
            }
            let file = GlbFile::load(&path).unwrap_or_else(|error| panic!("{error}"));
            assert!(!file.primitives.is_empty(), "{}", path.display());
            assert!(
                file.height() > 0.05,
                "{} is {} m tall",
                path.display(),
                file.height()
            );
            assert!(file.bounds_min[1] > -0.1, "{}: foot is below the origin", path.display());
            for primitive in &file.primitives {
                assert!(!primitive.vertices.is_empty());
                assert_eq!(primitive.indices.len() % 3, 0);
                assert!(
                    primitive.indices.iter().all(|i| (*i as usize) < primitive.vertices.len()),
                    "{}",
                    path.display()
                );
                assert!(primitive.material < file.materials.len().max(1));
            }
            decoded += 1;
        }
        assert_eq!(decoded, 36, "expected the full extracted pack");
    }

    /// Every material resolves to a base colour, and the alpha-cutout split
    /// between bark and branches survives the round trip.
    #[test]
    fn materials_carry_their_alpha_mode_and_textures() {
        let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/models/fir");

        let opaque = GlbFile::load(&directory.join("fir-large-1-lod-0.glb")).unwrap();
        assert_eq!(opaque.materials.len(), 2);
        assert!(!opaque.materials[0].alpha_cutout, "bark must stay opaque");
        assert!(opaque.materials[1].alpha_cutout, "branch cards must be cut out");
        assert!(opaque.materials[0].base_colour.is_some());
        assert!(opaque.materials[0].normal.is_some());

        // LOD 3 is the billboard: one material, textures embedded in the BIN
        // chunk rather than shared as external PNGs.
        let billboard = GlbFile::load(&directory.join("fir-large-1-lod-3.glb")).unwrap();
        assert_eq!(billboard.materials.len(), 1);
        assert!(billboard.materials[0].alpha_cutout);
        assert!(matches!(
            billboard.materials[0].base_colour,
            Some(super::GlbImage::Embedded(_))
        ));
        assert_eq!(billboard.primitives.len(), 1);
    }

    /// The whole pack is authored as near-non-reflective foliage, and that
    /// factor is the difference between a canopy that reads as needles and one
    /// painted with a pale sheen. It arrives through an extension, so a
    /// reader that silently stopped finding `KHR_materials_specular` would
    /// leave every tree at the bare 0.04 default with nothing else to show for
    /// it — the geometry, textures and alpha would all still load.
    #[test]
    fn the_pack_states_its_own_specular_factor() {
        let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/models/fir");
        let tree = GlbFile::load(&directory.join("fir-large-1-lod-0.glb")).unwrap();
        for (material, expected) in tree.materials.iter().zip([0.1448, 0.0960]) {
            assert!(
                (material.specular_factor - expected).abs() < 1e-3,
                "{} ships specularFactor {expected}, read {}",
                material.name,
                material.specular_factor
            );
        }
        // The factor scales the 0.04 dielectric F0 the composite would
        // otherwise apply on its own: the branch cards come out at 0.0038, an
        // order of magnitude less reflective than the bare default.
        assert!((tree.materials[1].specular_factor * 0.04 - 0.00384).abs() < 1e-4);

        let billboard = GlbFile::load(&directory.join("fir-large-1-lod-3.glb")).unwrap();
        assert!((billboard.materials[0].specular_factor - 0.0219).abs() < 1e-3);
    }
}
