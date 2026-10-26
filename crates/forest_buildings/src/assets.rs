//! The building kit: the houses and the sheriff's office exported from
//! ~/buildings by scripts/export_buildings.py, one GLB each in
//! `assets/models/buildings`, sharing one set of textures in
//! `assets/textures/buildings`.
//!
//! A GLB holds three kinds of node, told apart by name:
//!
//! - `part.<object>`: the building's static meshes, in building space (metres,
//!   +Y up, the foundation standing on y = 0).
//! - `door.<object>`: one door each, its node origin on the hinge axis and its
//!   node rotation the door's closed turn about +Y.
//! - `collision.<kind>`: the closed, outward-facing convex pieces the player
//!   collides with (walls, floors, stairs, railings, ...).
//!
//! Every part and door vertex carries COLOR_0: red is the share of the sky the
//! surface sees past the building itself, green the share of the rain that
//! reaches it. The composite dims the sky light indoors by the first, and the
//! weather keeps floors under a roof dry by the second.
//!
//! Geometry loads on the calling thread in a few tens of milliseconds, since
//! placing the buildings needs their footprints; the textures are decoded
//! separately ([`prepare_textures`]), on a background task.

use bevy::math::{Mat3, Mat4, Quat, Vec2, Vec3};
use forest_vegetation::assets::{
    PreparedTexture, base_color_chain, decode, linear_chain, pack_detail,
};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

/// One interleaved vertex: the plants' 48-byte layout plus the sky and rain
/// shares.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BuildingVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub tangent: [f32; 4],
    /// The share of the sky the surface sees past the building, and the share
    /// of the rain that falls on it.
    pub sky: [f32; 2],
}

const _: () = assert!(std::mem::size_of::<BuildingVertex>() == 56);

/// Texture resolution cap. A board or a brick course repeats every metre or
/// two, so 1024 texels still give several hundred texels a metre.
const TEXTURE_MAX: u32 = 1024;

/// What a material looks like, by the kit's own name for it.
pub struct BuildingMaterial {
    pub name: String,
    /// Window glass: not drawn into the G-buffer, so a window shows the room
    /// behind it and lets the sun in, but over the lit scene as a faint
    /// reflection of the sky.
    pub glass: bool,

    /// Index into the texture list: sRGB base colour.
    pub base_color: usize,

    /// Index into the texture list: linear normal XY, roughness, occlusion.
    pub detail: usize,

    pub base_color_factor: [f32; 4],
    pub roughness_factor: f32,
    pub normal_scale: f32,
}

/// One indexed draw: `index_count` indices from `first_index`, relative to
/// `base_vertex`.
#[derive(Clone, Copy, Debug)]
pub struct BuildingPrimitive {
    pub material: usize,
    pub first_index: u32,
    pub index_count: u32,
    pub base_vertex: u32,
}

/// The collision objects the kit builds (`COL_<kind>` in the .blend files).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CollisionKind {
    Wall,
    Floor,
    Ceiling,
    Roof,
    Window,
    Deck,
    Railing,
    Frame,
    Stairs,
    Steps,
    Cells,
    Other,
}

impl CollisionKind {
    fn from_name(name: &str) -> Self {
        match name {
            "wall" => Self::Wall,
            "floor" => Self::Floor,
            "ceiling" => Self::Ceiling,
            "roof" => Self::Roof,
            "window" => Self::Window,
            "deck" => Self::Deck,
            "railing" => Self::Railing,
            "frame" => Self::Frame,
            "stairs" => Self::Stairs,
            "steps" => Self::Steps,
            "cells" => Self::Cells,
            _ => Self::Other,
        }
    }
}

/// A closed convex piece of a building's collision, in building space.
#[derive(Clone, Debug)]
pub struct ConvexPiece {
    pub kind: CollisionKind,
    pub vertices: Vec<Vec3>,
    /// Every edge of its faces, as vertex index pairs: where it crosses a
    /// horizontal plane is where its cross-section's corners are.
    pub edges: Vec<[u16; 2]>,

    /// Outward unit normal and offset of each face: a point is inside where
    /// `normal.dot(point) <= offset` for every face.
    pub planes: Vec<(Vec3, f32)>,

    pub minimum: Vec3,
    pub maximum: Vec3,
}

/// A box over a roof, in building space: rain does not fall inside it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Shelter {
    pub minimum: Vec2,
    pub maximum: Vec2,
    /// The top of the roof over it.
    pub top: f32,
}

/// A hinged door.
#[derive(Clone, Debug)]
pub struct DoorModel {
    pub name: String,
    /// The hinge axis, building space.
    pub hinge: Vec3,

    /// The door's closed turn about +Y, radians.
    pub yaw: f32,

    /// Index into [`BuildingLibrary::door_meshes`].
    pub mesh: usize,

    /// Door-local horizontal direction from the hinge to the door's free
    /// edge (before `yaw`), and the distance to it.
    pub reach: Vec2,

    pub width: f32,
    /// Thickness of the leaf, its handles left out.
    pub thickness: f32,

    /// Bottom and top of the leaf, building space.
    pub bottom: f32,

    pub top: f32,
    /// How far it swings each way from closed, radians: `swing[0] <= 0 <= swing[1]`,
    /// set by the collision analysis when the library loads.
    pub swing: [f32; 2],

    /// Building-space XZ direction its outer face looks: the side of the
    /// leaf that sees the more sky.
    pub outward: Vec2,
}

pub struct DoorMesh {
    pub primitives: Vec<BuildingPrimitive>,
}

pub struct BuildingModel {
    pub name: String,
    /// Drawn wherever the building is.
    pub exterior: Vec<BuildingPrimitive>,

    /// Rooms, stairs and ceiling joists: drawn only near enough to be seen
    /// through a window or a door.
    pub interior: Vec<BuildingPrimitive>,

    pub doors: Vec<DoorModel>,
    pub pieces: Vec<ConvexPiece>,
    pub shelters: Vec<Shelter>,
    /// Bounds of every part, building space.
    pub minimum: Vec3,

    pub maximum: Vec3,
    /// The way the front door faces, building space XZ, along an axis.
    pub front: Vec2,
}

impl BuildingModel {
    /// Centre of the footprint, building space XZ.
    pub fn footprint_centre(&self) -> Vec2 {
        Vec2::new(
            self.minimum.x + self.maximum.x,
            self.minimum.z + self.maximum.z,
        ) * 0.5
    }

    /// Half the footprint's extent along building X and Z.
    pub fn footprint_half(&self) -> Vec2 {
        Vec2::new(
            self.maximum.x - self.minimum.x,
            self.maximum.z - self.minimum.z,
        ) * 0.5
    }

    /// Horizontal radius of a circle about the footprint's centre that holds
    /// the whole building, doors swung open included.
    pub fn radius(&self) -> f32 {
        self.footprint_half().length() + 1.0
    }
}

/// The textures a library needs, decoded later by [`prepare_textures`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum TextureKey {
    White,
    BaseColor(PathBuf),
    Detail {
        normal: Option<PathBuf>,
        roughness: Option<PathBuf>,
    },
}

pub struct BuildingLibrary {
    pub models: Vec<BuildingModel>,
    pub materials: Vec<BuildingMaterial>,
    pub door_meshes: Vec<DoorMesh>,
    pub texture_keys: Vec<TextureKey>,
    pub vertices: Vec<BuildingVertex>,
    pub indices: Vec<u32>,
}

impl BuildingLibrary {
    #[cfg(any(test, feature = "test_support"))]
    pub fn model_index(&self, name: &str) -> Option<usize> {
        self.models.iter().position(|model| model.name == name)
    }
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

/// Parts drawn only when the building is near: everything inside its walls.
const INTERIOR_PARTS: [&str; 4] = ["interior", "rafters", "stairs", "cells"];

/// Corners closer than this are one vertex when collision pieces are welded.
const WELD_PRECISION: f32 = 1000.0;
/// Two face planes of a piece closer than these in angle (one minus the
/// cosine) and offset are one.
const PLANE_ANGLE_TOLERANCE: f32 = 1e-6;
const PLANE_OFFSET_TOLERANCE: f32 = 1e-4;

/// One primitive's attributes as read from a GLB, already in the space the
/// caller asked for.
struct RawPrimitive {
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    uvs: Vec<[f32; 2]>,
    sky: Vec<[f32; 2]>,
    indices: Vec<u32>,
    material: Option<usize>,
}

struct Loader {
    texture_keys: Vec<TextureKey>,
    texture_lookup: HashMap<TextureKey, usize>,
    materials: Vec<BuildingMaterial>,
    vertices: Vec<BuildingVertex>,
    indices: Vec<u32>,
    door_meshes: Vec<DoorMesh>,
    /// Door meshes by the bytes of their vertices and indices: the kit's doors
    /// are copies of one mesh, drawn as instances of it.
    door_lookup: HashMap<Vec<u8>, usize>,
}

impl Loader {
    fn texture(&mut self, key: TextureKey) -> usize {
        if let Some(&index) = self.texture_lookup.get(&key) {
            return index;
        }
        self.texture_keys.push(key.clone());
        let index = self.texture_keys.len() - 1;
        self.texture_lookup.insert(key, index);
        index
    }

    /// The library's material for a GLB material, by the kit's name for it.
    fn material(
        &mut self,
        material: &gltf::Material<'_>,
        directory: &Path,
    ) -> Result<usize, String> {
        let name = material.name().unwrap_or("unnamed").to_string();
        if let Some(index) = self.materials.iter().position(|known| known.name == name) {
            return Ok(index);
        }
        let uri_of = |texture: gltf::Texture<'_>| -> Option<PathBuf> {
            match texture.source().source() {
                gltf::image::Source::Uri { uri, .. } => Some(directory.join(uri)),
                gltf::image::Source::View { .. } => None,
            }
        };
        let resolve = |path: Option<PathBuf>| -> Result<Option<PathBuf>, String> {
            let Some(path) = path else {
                return Ok(None);
            };
            let resolved = forest_assets::asset_io::canonicalize(&path)
                .map_err(|e| format!("texture {}: {e}", path.display()))?;
            Ok(Some(resolved))
        };
        let pbr = material.pbr_metallic_roughness();
        let base_path = resolve(
            pbr.base_color_texture()
                .and_then(|info| uri_of(info.texture())),
        )?;
        let roughness_path = resolve(
            pbr.metallic_roughness_texture()
                .and_then(|info| uri_of(info.texture())),
        )?;
        let normal_path = resolve(
            material
                .normal_texture()
                .and_then(|info| uri_of(info.texture())),
        )?;
        let base_color = match base_path {
            Some(path) => self.texture(TextureKey::BaseColor(path)),
            None => self.texture(TextureKey::White),
        };
        let detail_key = TextureKey::Detail {
            normal: normal_path,
            roughness: roughness_path,
        };
        let detail = self.texture(detail_key);
        let glass = name == "Glass" || material.alpha_mode() == gltf::material::AlphaMode::Blend;
        self.materials.push(BuildingMaterial {
            name,
            glass,
            base_color,
            detail,
            base_color_factor: pbr.base_color_factor(),
            roughness_factor: pbr.roughness_factor(),
            normal_scale: material.normal_texture().map_or(1.0, |info| info.scale()),
        });
        Ok(self.materials.len() - 1)
    }

    /// Read a mesh's primitives, positions and normals carried through
    /// `matrix`.
    fn read_mesh(
        &mut self,
        mesh: gltf::Mesh<'_>,
        blob: &[u8],
        matrix: Mat4,
        directory: &Path,
        with_material: bool,
    ) -> Result<Vec<RawPrimitive>, String> {
        let normal_matrix = Mat3::from_mat4(matrix).inverse().transpose();
        let mut primitives = Vec::new();
        for primitive in mesh.primitives() {
            if primitive.mode() != gltf::mesh::Mode::Triangles {
                return Err("only triangle lists are supported".into());
            }
            let reader = primitive.reader(|buffer| match buffer.source() {
                gltf::buffer::Source::Bin => Some(blob),
                gltf::buffer::Source::Uri(_) => None,
            });
            let local_positions: Vec<[f32; 3]> = reader
                .read_positions()
                .ok_or("missing positions")?
                .collect();
            let local_normals: Vec<[f32; 3]> = match reader.read_normals() {
                Some(normals) => normals.collect(),
                None => vec![[0.0, 1.0, 0.0]; local_positions.len()],
            };
            let uvs: Vec<[f32; 2]> = match reader.read_tex_coords(0) {
                Some(uvs) => uvs.into_f32().collect(),
                None => vec![[0.0, 0.0]; local_positions.len()],
            };
            let sky: Vec<[f32; 2]> = match reader.read_colors(0) {
                Some(colours) => colours.into_rgba_f32().map(|c| [c[0], c[1]]).collect(),
                None => vec![[1.0, 1.0]; local_positions.len()],
            };
            let indices: Vec<u32> = match reader.read_indices() {
                Some(indices) => indices.into_u32().collect(),
                None => (0..local_positions.len() as u32).collect(),
            };
            let count = local_positions.len();
            let consistent =
                local_normals.len() == count && uvs.len() == count && sky.len() == count;
            let in_range = indices.iter().all(|&index| (index as usize) < count);
            if count == 0 || !consistent || !in_range || !indices.len().is_multiple_of(3) {
                return Err("invalid vertex or triangle data".into());
            }
            let place = |p: &[f32; 3]| matrix.transform_point3(Vec3::from_array(*p)).to_array();
            let turn = |n: &[f32; 3]| {
                let turned = normal_matrix * Vec3::from_array(*n);
                turned.try_normalize().unwrap_or(Vec3::Y).to_array()
            };
            let positions: Vec<[f32; 3]> = local_positions.iter().map(place).collect();
            let normals: Vec<[f32; 3]> = local_normals.iter().map(turn).collect();
            let material = if with_material {
                Some(self.material(&primitive.material(), directory)?)
            } else {
                None
            };
            primitives.push(RawPrimitive {
                positions,
                normals,
                uvs,
                sky,
                indices,
                material,
            });
        }
        Ok(primitives)
    }

    /// Vertices with generated tangents, appended to the library.
    fn append_vertices(&mut self, primitive: &RawPrimitive) -> u32 {
        let tangents = forest_vegetation::grass::generate_tangents(
            &primitive.positions,
            &primitive.normals,
            &primitive.uvs,
            &primitive.indices,
        );
        let first = self.vertices.len() as u32;
        for (index, &tangent) in tangents.iter().enumerate() {
            let vertex = BuildingVertex {
                position: primitive.positions[index],
                normal: primitive.normals[index],
                uv: primitive.uvs[index],
                tangent,
                sky: primitive.sky[index],
            };
            self.vertices.push(vertex);
        }
        first
    }

    /// A door's mesh, shared with any door already loaded with the same one.
    fn door_mesh(&mut self, primitives: &[RawPrimitive]) -> usize {
        let mut signature: Vec<u8> = Vec::new();
        for primitive in primitives {
            signature.extend_from_slice(bytemuck::cast_slice(&primitive.positions));
            signature.extend_from_slice(bytemuck::cast_slice(&primitive.uvs));
            signature.extend_from_slice(bytemuck::cast_slice(&primitive.indices));
            signature.extend_from_slice(
                &(primitive.material.unwrap_or(usize::MAX) as u64).to_le_bytes(),
            );
        }
        if let Some(&index) = self.door_lookup.get(&signature) {
            return index;
        }
        let mut drawn = Vec::new();
        for primitive in primitives {
            let base_vertex = self.append_vertices(primitive);
            let first_index = self.indices.len() as u32;
            self.indices.extend_from_slice(&primitive.indices);
            drawn.push(BuildingPrimitive {
                material: primitive.material.unwrap_or(0),
                first_index,
                index_count: primitive.indices.len() as u32,
                base_vertex,
            });
        }
        self.door_meshes.push(DoorMesh { primitives: drawn });
        let index = self.door_meshes.len() - 1;
        self.door_lookup.insert(signature, index);
        index
    }

    fn read_building(&mut self, path: &Path, name: &str) -> Result<BuildingModel, String> {
        let model_bytes = forest_assets::asset_io::read(path)?;
        let gltf = gltf::Gltf::from_slice(&model_bytes).map_err(|e| e.to_string())?;
        let blob = gltf.blob.as_deref().ok_or("expected a GLB binary chunk")?;
        let directory = path.parent().unwrap_or(Path::new("."));
        let scene = gltf
            .default_scene()
            .or_else(|| gltf.scenes().next())
            .ok_or("no scene")?;

        // The building's own vertices, indices by (interior, material), and
        // its pieces and doors, gathered over the node tree.
        let mut model_vertices: Vec<RawPrimitive> = Vec::new();
        let mut groups: HashMap<(bool, usize), Vec<u32>> = HashMap::new();
        let mut vertex_count = 0u32;
        let mut pieces = Vec::new();
        let mut doors = Vec::new();
        let mut stack: Vec<(gltf::Node<'_>, Mat4)> =
            scene.nodes().map(|node| (node, Mat4::IDENTITY)).collect();
        while let Some((node, parent)) = stack.pop() {
            let local = Mat4::from_cols_array_2d(&node.transform().matrix());
            let world = parent * local;
            for child in node.children() {
                stack.push((child, world));
            }
            let Some(mesh) = node.mesh() else {
                continue;
            };
            let node_name = node.name().unwrap_or("");
            if let Some(part) = node_name.strip_prefix("part.") {
                let interior = INTERIOR_PARTS.contains(&part);
                for primitive in self.read_mesh(mesh, blob, world, directory, true)? {
                    let material = primitive.material.unwrap_or(0);
                    let group = groups.entry((interior, material)).or_default();
                    group.extend(primitive.indices.iter().map(|&index| index + vertex_count));
                    vertex_count += primitive.positions.len() as u32;
                    model_vertices.push(primitive);
                }
            } else if let Some(door_name) = node_name.strip_prefix("door.") {
                let primitives = self.read_mesh(mesh, blob, Mat4::IDENTITY, directory, true)?;
                let door_mesh = self.door_mesh(&primitives);
                let door = door_from(door_name, &primitives, world, door_mesh)?;
                doors.push(door);
            } else if let Some(kind) = node_name.strip_prefix("collision.") {
                let kind = CollisionKind::from_name(kind);
                for primitive in self.read_mesh(mesh, blob, world, directory, false)? {
                    let positions: Vec<Vec3> = primitive
                        .positions
                        .iter()
                        .map(|p| Vec3::from_array(*p))
                        .collect();
                    let parts: Vec<u32> = primitive
                        .uvs
                        .iter()
                        .map(|uv| uv[0].max(0.0) as u32)
                        .collect();
                    pieces.extend(convex_pieces(&positions, &parts, &primitive.indices, kind));
                }
            }
        }
        if model_vertices.is_empty() {
            return Err("no part.* meshes".into());
        }

        // One vertex range for the whole building, one draw per class and
        // material over it.
        let mut minimum = Vec3::splat(f32::INFINITY);
        let mut maximum = Vec3::splat(f32::NEG_INFINITY);
        let mut base_vertex = None;
        for primitive in &model_vertices {
            for position in &primitive.positions {
                minimum = minimum.min(Vec3::from_array(*position));
                maximum = maximum.max(Vec3::from_array(*position));
            }
            let first = self.append_vertices(primitive);
            base_vertex.get_or_insert(first);
        }
        let base_vertex = base_vertex.unwrap_or(0);
        let mut keys: Vec<(bool, usize)> = groups.keys().copied().collect();
        keys.sort();
        let mut exterior = Vec::new();
        let mut interior = Vec::new();
        for key in keys {
            let group = &groups[&key];
            let first_index = self.indices.len() as u32;
            self.indices.extend_from_slice(group);
            let primitive = BuildingPrimitive {
                material: key.1,
                first_index,
                index_count: group.len() as u32,
                base_vertex,
            };
            if key.0 {
                interior.push(primitive);
            } else {
                exterior.push(primitive);
            }
        }

        doors.sort_by(|a: &DoorModel, b: &DoorModel| a.name.cmp(&b.name));
        let shelters = pieces
            .iter()
            .filter(|piece| piece.kind == CollisionKind::Roof)
            .map(|piece| Shelter {
                minimum: Vec2::new(piece.minimum.x, piece.minimum.z),
                maximum: Vec2::new(piece.maximum.x, piece.maximum.z),
                top: piece.maximum.y,
            })
            .collect();
        let mut model = BuildingModel {
            name: name.to_string(),
            exterior,
            interior,
            doors,
            pieces,
            shelters,
            minimum,
            maximum,
            front: Vec2::X,
        };
        model.front = front_of(&model);
        Ok(model)
    }
}

/// A door from its node: the hinge where the node stands, its closed turn
/// from the node's rotation, and its leaf measured from its mesh.
fn door_from(
    name: &str,
    primitives: &[RawPrimitive],
    world: Mat4,
    mesh: usize,
) -> Result<DoorModel, String> {
    let (_, rotation, translation) = world.to_scale_rotation_translation();
    let yaw =
        yaw_of(rotation).ok_or_else(|| format!("door {name} is not turned about +Y alone"))?;
    let mut minimum = Vec3::splat(f32::INFINITY);
    let mut maximum = Vec3::splat(f32::NEG_INFINITY);
    for primitive in primitives {
        for position in &primitive.positions {
            minimum = minimum.min(Vec3::from_array(*position));
            maximum = maximum.max(Vec3::from_array(*position));
        }
    }
    // The face that sees more sky is the outer one; weigh each corner's
    // normal by its sky share to find which way it looks.
    let mut facing = Vec2::ZERO;
    for primitive in primitives {
        for (normal, sky) in primitive.normals.iter().zip(&primitive.sky) {
            facing += Vec2::new(normal[0], normal[2]) * sky[0];
        }
    }
    let (sine, cosine) = yaw.sin_cos();
    let turned = Vec2::new(
        facing.x * cosine + facing.y * sine,
        -facing.x * sine + facing.y * cosine,
    );
    let outward = turned.try_normalize().unwrap_or(Vec2::Y);
    let extent_x = maximum.x - minimum.x;
    let extent_z = maximum.z - minimum.z;
    // The leaf runs out from the hinge along its longer horizontal side.
    let (reach, width, thickness) = if extent_x >= extent_z {
        let far = if maximum.x.abs() > minimum.x.abs() {
            maximum.x
        } else {
            minimum.x
        };
        (Vec2::new(far.signum(), 0.0), far.abs(), extent_z)
    } else {
        let far = if maximum.z.abs() > minimum.z.abs() {
            maximum.z
        } else {
            minimum.z
        };
        (Vec2::new(0.0, far.signum()), far.abs(), extent_x)
    };
    Ok(DoorModel {
        name: name.to_string(),
        hinge: translation,
        yaw,
        mesh,
        reach,
        width,
        // The handles stand proud of the leaf; the leaf itself is a board.
        thickness: thickness.min(0.06),
        bottom: translation.y + minimum.y,
        top: translation.y + maximum.y,
        swing: [0.0, 0.0],
        outward,
    })
}

/// The turn about +Y a rotation makes, if that is all it does.
fn yaw_of(rotation: Quat) -> Option<f32> {
    let tilt = rotation.x.abs() + rotation.z.abs();
    if tilt > 1e-3 {
        return None;
    }
    Some(2.0 * rotation.y.atan2(rotation.w))
}

/// The way the front door faces, along the nearer axis. Not toward the door
/// from the footprint's centre: a granny flat in front of a wing (house-g)
/// moves the centre past the front wall.
fn front_of(model: &BuildingModel) -> Vec2 {
    let door = model
        .doors
        .iter()
        .find(|door| door.name == "front-door")
        .or_else(|| model.doors.first());
    let Some(door) = door else {
        return Vec2::Y;
    };
    let outward = door.outward;
    if outward.x.abs() > outward.y.abs() {
        Vec2::new(outward.x.signum(), 0.0)
    } else {
        Vec2::new(0.0, outward.y.signum())
    }
}

/// Split a collision mesh into its closed convex pieces: the exporter writes
/// each face's piece (its loose part in the .blend) into the UVs, since
/// touching pieces share corners and the glTF keeps no other trace of where
/// one ends. Within a piece, the corners the exporter split by face are
/// welded again.
fn convex_pieces(
    positions: &[Vec3],
    parts: &[u32],
    indices: &[u32],
    kind: CollisionKind,
) -> Vec<ConvexPiece> {
    let mut welded: HashMap<(u32, [i32; 3]), u32> = HashMap::new();
    let mut points: Vec<Vec3> = Vec::new();
    let mut ids: Vec<u32> = Vec::with_capacity(positions.len());
    for (position, part) in positions.iter().zip(parts) {
        let key = [
            (position.x * WELD_PRECISION).round() as i32,
            (position.y * WELD_PRECISION).round() as i32,
            (position.z * WELD_PRECISION).round() as i32,
        ];
        let next = points.len() as u32;
        let id = *welded.entry((*part, key)).or_insert(next);
        if id == next {
            points.push(*position);
        }
        ids.push(id);
    }
    let mut groups: HashMap<u32, Vec<[u32; 3]>> = HashMap::new();
    for triangle in indices.as_chunks::<3>().0 {
        let part = parts[triangle[0] as usize];
        let corners = [
            ids[triangle[0] as usize],
            ids[triangle[1] as usize],
            ids[triangle[2] as usize],
        ];
        groups.entry(part).or_default().push(corners);
    }
    let mut order: Vec<u32> = groups.keys().copied().collect();
    order.sort();
    order
        .iter()
        .filter_map(|part| piece_from(&points, &groups[part], kind))
        .collect()
}

fn piece_from(points: &[Vec3], triangles: &[[u32; 3]], kind: CollisionKind) -> Option<ConvexPiece> {
    let mut local: HashMap<u32, u16> = HashMap::new();
    let mut vertices: Vec<Vec3> = Vec::new();
    let mut local_id = |id: u32, vertices: &mut Vec<Vec3>| -> u16 {
        *local.entry(id).or_insert_with(|| {
            vertices.push(points[id as usize]);
            (vertices.len() - 1) as u16
        })
    };
    let mut edges: HashSet<[u16; 2]> = HashSet::new();
    let mut planes: Vec<(Vec3, f32)> = Vec::new();
    for triangle in triangles {
        let corners = triangle.map(|id| local_id(id, &mut vertices));
        for (a, b) in [
            (corners[0], corners[1]),
            (corners[1], corners[2]),
            (corners[2], corners[0]),
        ] {
            edges.insert([a.min(b), a.max(b)]);
        }
        let [a, b, c] = triangle.map(|id| points[id as usize]);
        let Some(normal) = (b - a).cross(c - a).try_normalize() else {
            continue;
        };
        let offset = normal.dot(a);
        let known = planes.iter().any(|(n, d)| {
            n.dot(normal) > 1.0 - PLANE_ANGLE_TOLERANCE
                && (d - offset).abs() < PLANE_OFFSET_TOLERANCE
        });
        if !known {
            planes.push((normal, offset));
        }
    }
    if vertices.len() < 4 || planes.len() < 4 {
        return None;
    }
    // Outward-facing: the piece's own centre lies behind every face.
    let centre = vertices.iter().copied().sum::<Vec3>() / vertices.len() as f32;
    for plane in &mut planes {
        if plane.0.dot(centre) > plane.1 {
            *plane = (-plane.0, -plane.1);
        }
    }
    let minimum = vertices
        .iter()
        .copied()
        .fold(Vec3::splat(f32::INFINITY), Vec3::min);
    let maximum = vertices
        .iter()
        .copied()
        .fold(Vec3::splat(f32::NEG_INFINITY), Vec3::max);
    let mut edges: Vec<[u16; 2]> = edges.into_iter().collect();
    edges.sort();
    Some(ConvexPiece {
        kind,
        vertices,
        edges,
        planes,
        minimum,
        maximum,
    })
}

impl BuildingLibrary {
    /// Every building in `directory`, geometry and collision only: the
    /// textures are listed in `texture_keys` for [`prepare_textures`].
    pub fn load(directory: impl AsRef<Path>) -> Result<Self, String> {
        let directory = directory.as_ref();
        let entries = forest_assets::asset_io::readDirectory(directory)
            .map_err(|e| format!("reading {}: {e}", directory.display()))?;
        let mut paths: Vec<PathBuf> = entries
            .into_iter()
            .filter(|path| path.extension().is_some_and(|extension| extension == "glb"))
            .collect();
        paths.sort();
        if paths.is_empty() {
            return Err(format!("no buildings in {}", directory.display()));
        }
        let mut loader = Loader {
            texture_keys: Vec::new(),
            texture_lookup: HashMap::new(),
            materials: Vec::new(),
            vertices: Vec::new(),
            indices: Vec::new(),
            door_meshes: Vec::new(),
            door_lookup: HashMap::new(),
        };
        let mut models = Vec::new();
        for path in &paths {
            let name = path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .unwrap_or("building")
                .to_string();
            let mut model = loader
                .read_building(path, &name)
                .map_err(|e| format!("{}: {e}", path.display()))?;
            super::collision::fit_door_swings(&mut model);
            models.push(model);
        }
        Ok(Self {
            models,
            materials: loader.materials,
            door_meshes: loader.door_meshes,
            texture_keys: loader.texture_keys,
            vertices: loader.vertices,
            indices: loader.indices,
        })
    }

    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }
}

// ---------------------------------------------------------------------------
// Textures
// ---------------------------------------------------------------------------

fn prepare_texture(key: &TextureKey) -> Result<PreparedTexture, String> {
    match key {
        TextureKey::White => Ok(PreparedTexture {
            label: "building-white".to_string(),
            width: 1,
            height: 1,
            srgb: true,
            levels: vec![vec![255, 255, 255, 255]],
        }),
        TextureKey::BaseColor(path) => {
            let image = decode(path)?;
            let levels = base_color_chain(&image, TEXTURE_MAX, None);
            let label = path
                .file_name()
                .map_or_else(String::new, |n| n.to_string_lossy().into_owned());
            Ok(PreparedTexture {
                label,
                width: levels[0].width,
                height: levels[0].height,
                srgb: true,
                levels: levels.into_iter().map(|level| level.rgba).collect(),
            })
        }
        TextureKey::Detail { normal, roughness } => {
            let normal_image = normal.as_deref().map(decode).transpose()?;
            let roughness_image = roughness.as_deref().map(decode).transpose()?;
            let packed = pack_detail(normal_image.as_ref(), roughness_image.as_ref(), None);
            let levels = linear_chain(packed, TEXTURE_MAX);
            let label = normal.as_deref().and_then(Path::file_name).map_or_else(
                || "building-flat-detail".to_string(),
                |n| n.to_string_lossy().replace("normal", "detail"),
            );
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

/// Decode and prepare every texture a library lists, spread over the cores.
pub fn prepare_textures(keys: &[TextureKey]) -> Result<Vec<PreparedTexture>, String> {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};
    let next = AtomicUsize::new(0);
    let results: Mutex<Vec<Option<Result<PreparedTexture, String>>>> =
        Mutex::new((0..keys.len()).map(|_| None).collect());
    let workers = std::thread::available_parallelism()
        .map_or(4, |n| n.get())
        .clamp(1, 8);
    let worker = || {
        loop {
            let index = next.fetch_add(1, Ordering::Relaxed);
            if index >= keys.len() {
                break;
            }
            let prepared = prepare_texture(&keys[index]);
            results
                .lock()
                .expect("building texture results lock poisoned")[index] = Some(prepared);
        }
    };
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(worker);
        }
    });
    let prepared = results
        .into_inner()
        .expect("building texture results lock poisoned");
    prepared
        .into_iter()
        .map(|result| result.expect("every texture prepared"))
        .collect()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    use super::test_support::library;

    #[test]
    fn the_kit_loads_with_its_doors_collision_and_sky() {
        let library = library();
        let names: Vec<&str> = library
            .models
            .iter()
            .map(|model| model.name.as_str())
            .collect();
        assert_eq!(
            names,
            [
                "house-a",
                "house-b",
                "house-c",
                "house-d",
                "house-e",
                "house-f",
                "house-g",
                "sheriff-office"
            ]
        );
        // Ten kit materials, nothing duplicated across files.
        let mut material_names: Vec<&str> =
            library.materials.iter().map(|m| m.name.as_str()).collect();
        material_names.sort();
        material_names.dedup();
        assert_eq!(material_names.len(), library.materials.len());
        assert!(library.materials.iter().any(|m| m.glass));
        for model in &library.models {
            assert!(
                !model.exterior.is_empty() && !model.interior.is_empty(),
                "{}",
                model.name
            );
            assert!(
                model.doors.iter().any(|door| door.name == "front-door"),
                "{} has no front door",
                model.name
            );
            assert!(
                model.pieces.len() > 50,
                "{}: {} pieces",
                model.name,
                model.pieces.len()
            );
            assert!(!model.shelters.is_empty(), "{}", model.name);
            // The foundation stands on the ground; nothing hangs below it.
            assert!(
                model.minimum.y.abs() < 0.02,
                "{}: {}",
                model.name,
                model.minimum.y
            );
            for door in &model.doors {
                assert!(
                    (0.7..0.85).contains(&door.width),
                    "{} {}: {}",
                    model.name,
                    door.name,
                    door.width
                );
                assert!(door.top - door.bottom > 1.9, "{} {}", model.name, door.name);
            }
        }
        // Each door keeps its own mesh: the sky baked into its corners splits
        // them differently from one door to the next.
        let door_count: usize = library.models.iter().map(|model| model.doors.len()).sum();
        assert!(library.door_meshes.len() <= door_count);
        for vertex in &library.vertices {
            assert!((0.0..=1.0).contains(&vertex.sky[0]) && (0.0..=1.0).contains(&vertex.sky[1]));
        }
    }

    #[test]
    fn collision_pieces_are_convex_and_outward() {
        let library = library();
        for model in &library.models {
            for piece in &model.pieces {
                for vertex in &piece.vertices {
                    for (normal, offset) in &piece.planes {
                        let outside = normal.dot(*vertex) - offset;
                        assert!(
                            outside < 0.005,
                            "{} {:?}: a corner {outside} m outside a face",
                            model.name,
                            piece.kind
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn fronts_face_the_way_the_kit_says() {
        let library = library();
        let front = |name: &str| library.models[library.model_index(name).unwrap()].front;
        // house-a and house-b face east, +X; the generated houses face
        // Blender's -Y, which is +Z here.
        assert_eq!(front("house-a"), Vec2::X);
        assert_eq!(front("house-b"), Vec2::X);
        for name in [
            "house-c",
            "house-d",
            "house-e",
            "house-f",
            "house-g",
            "sheriff-office",
        ] {
            assert_eq!(front(name), Vec2::Y, "{name}");
        }
    }
}

/// Building assets shared by this crate and renderer regression tests.
#[cfg(any(test, feature = "test_support"))]
pub mod test_support {
    use super::*;

    pub fn library() -> BuildingLibrary {
        BuildingLibrary::load(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../assets/models/buildings"),
        )
        .expect("the building kit loads")
    }
}
