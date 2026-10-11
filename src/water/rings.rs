//! Crest-style concentric ocean tile geometry.
//!
//! Ported from bevy-aqua's `bevy-aqua-geom/src/rings.rs`, itself a
//! reimplementation of Crest's `Scripts/OceanBuilder.cs` (`BuildOceanPatch`
//! and `CreateLOD`). See `src/water/ATTRIBUTION.md`.
//!
//! DIVERGENCE: aqua builds `bevy::Mesh` assets and attaches them to entities
//! with a `Transform` per tile. This renderer has no bevy_pbr scene, so the
//! same topology is emitted as raw vertex/index arrays and the per-tile
//! placement becomes an instance buffer the vertex shader reads. The patch
//! shapes, the fat/slim overlap rule, the horizon skirt and the ring layouts
//! are unchanged — those are what make the tiles tile.

use super::{WATER_LOD_COUNT, WATER_TILE_RESOLUTION};

const PATCH_HALF_WIDTH: f32 = 0.5;
const OUTER_TILE_OFFSET: f32 = 1.5;
const INNER_TILE_OFFSET: f32 = 0.5;
const TILE_AXIS: [f32; 4] = [-1.5, -0.5, 0.5, 1.5];
const CENTER_TILE_COUNT: usize = 16;
const RING_TILE_COUNT: usize = 12;

/// Crest supports LOD indices 0 through 15. Its 16-slot capacity sets how far
/// the outermost row stretches toward the far plane.
const CREST_LOD_CAPACITY: usize = 16;
const CREST_EXTENT_BASE: f32 = 100.0;
const OUTER_EXTENT_MULTIPLIER: f32 =
    CREST_EXTENT_BASE * (CREST_LOD_CAPACITY - WATER_LOD_COUNT) as f32;

const REGULAR_EDGE_VERTICES: u32 = WATER_TILE_RESOLUTION as u32 + 1;
const INDICES_PER_QUAD: usize = 6;

/// One reusable ocean-tile patch topology.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(usize)]
pub enum Patch {
    Interior,
    FatX,
    FatXSlimZ,
    FatXOuter,
    FatXZ,
    FatXZOuter,
    SlimX,
    SlimXZ,
    SlimXFatZ,
}

impl Patch {
    /// Every patch variant, in build order.
    pub const ALL: [Self; 9] = [
        Self::Interior,
        Self::FatX,
        Self::FatXSlimZ,
        Self::FatXOuter,
        Self::FatXZ,
        Self::FatXZOuter,
        Self::SlimX,
        Self::SlimXZ,
        Self::SlimXFatZ,
    ];
}

/// One tile placement inside a ring layout.
#[derive(Clone, Copy, Debug)]
pub struct Tile {
    /// Offset from the ring centre in tile units, scaled by the LOD's scale.
    pub offset: [f32; 2],
    pub patch: Patch,
    /// Y rotation in radians, aligning the fat/slim edges outward.
    pub rotation: f32,
}

/// A patch's raw geometry: 2D local position (x, z) and triangle indices.
pub struct PatchMesh {
    pub positions: Vec<[f32; 2]>,
    pub indices: Vec<u32>,
}

#[derive(Clone, Copy)]
enum Edge {
    Slim,
    Regular,
    Fat,
}

impl Edge {
    fn vertex_count(self) -> u32 {
        match self {
            Self::Slim => REGULAR_EDGE_VERTICES - 1,
            Self::Regular => REGULAR_EDGE_VERTICES,
            Self::Fat => REGULAR_EDGE_VERTICES + 1,
        }
    }

    fn end(self, step: f32) -> f32 {
        match self {
            Self::Slim => PATCH_HALF_WIDTH - step,
            Self::Regular => PATCH_HALF_WIDTH,
            Self::Fat => PATCH_HALF_WIDTH + step,
        }
    }
}

#[derive(Clone, Copy)]
struct Edges {
    x: Edge,
    z: Edge,
    outer_x: bool,
    outer_z: bool,
}

impl Edges {
    const fn new(x: Edge, z: Edge) -> Self {
        Self { x, z, outer_x: false, outer_z: false }
    }

    const fn outer_x(mut self) -> Self {
        self.outer_x = true;
        self
    }

    const fn outer_xz(mut self) -> Self {
        self.outer_x = true;
        self.outer_z = true;
        self
    }
}

/// Build one reusable patch. Fat and slim edge variants overlap without holes,
/// while the outer variants stretch one row into the horizon skirt.
pub fn build_patch(patch: Patch) -> PatchMesh {
    let edges = patch_edges(patch);
    let columns = edges.x.vertex_count();
    let rows = edges.z.vertex_count();
    let step = (WATER_TILE_RESOLUTION as f32).recip();
    let end_x = edges.x.end(step);
    let end_z = edges.z.end(step);

    let mut positions = Vec::with_capacity((columns * rows) as usize);
    for row in 0..rows {
        let fraction_z = row as f32 / (rows - 1) as f32;
        let mut z = -PATCH_HALF_WIDTH + (end_z + PATCH_HALF_WIDTH) * fraction_z;
        if edges.outer_z && row == rows - 1 {
            z *= OUTER_EXTENT_MULTIPLIER;
        }
        for column in 0..columns {
            let fraction_x = column as f32 / (columns - 1) as f32;
            let mut x = -PATCH_HALF_WIDTH + (end_x + PATCH_HALF_WIDTH) * fraction_x;
            if edges.outer_x && column == columns - 1 {
                x *= OUTER_EXTENT_MULTIPLIER;
            }
            positions.push([x, z]);
        }
    }

    let quad_count = (columns - 1) * (rows - 1);
    let mut indices = Vec::with_capacity(quad_count as usize * INDICES_PER_QUAD);
    for row in 0..rows - 1 {
        for column in 0..columns - 1 {
            add_quad(&mut indices, column, row, columns);
        }
    }
    PatchMesh { positions, indices }
}

/// Crest's 4x4 centre or 12-patch ring layout. Rotations carry each patch's
/// variable ends — the +x and +z ones `build_patch` scales, which are where the
/// fat overlap and the horizon skirt live — to the side facing away from the
/// ring centre.
pub fn tile_layout(lod: usize) -> Vec<Tile> {
    let capacity = if lod == 0 { CENTER_TILE_COUNT } else { RING_TILE_COUNT };
    let mut tiles = Vec::with_capacity(capacity);
    for &z in TILE_AXIS.iter().rev() {
        for &x in &TILE_AXIS {
            let offset = [x, z];
            if lod > 0 && is_inner(offset) {
                continue;
            }
            let (patch, rotation) = classify_patch(offset, lod == WATER_LOD_COUNT - 1);
            tiles.push(Tile { offset, patch, rotation });
        }
    }
    tiles
}

/// Tile centre in metres, relative to the ring centre. LOD `n` uses tiles
/// `WATER_BASE_SCALE * 2^n` metres across.
pub fn lod_scale(lod: usize) -> f32 {
    super::WATER_BASE_SCALE * (1u32 << lod) as f32
}

/// Camera distance at which wave displacement must be completely flat. The horizon skirt
/// starts at twice the outer tile scale, then stretches a single quad to the
/// far horizon. Carrying displacement into that quad smears it over kilometres.
/// Fragment normals are evaluated independently and remain detailed on the skirt.
/// Leave one snap step inside the skirt so it stays flat while the camera moves
/// relative to the snapped ring centre (at most half a step on either axis).
pub fn horizon_wave_fade_end() -> f32 {
    2.0 * lod_scale(WATER_LOD_COUNT - 1) - super::WATER_SNAP
}

fn add_quad(indices: &mut Vec<u32>, column: u32, row: u32, columns: u32) {
    let lower_left = column + row * columns;
    let lower_right = lower_left + 1;
    let upper_left = lower_left + columns;
    let upper_right = upper_left + 1;
    let quad = if (column + row) % 2 == 0 {
        [upper_right, lower_right, lower_left, lower_left, upper_left, upper_right]
    } else {
        [upper_right, lower_right, upper_left, lower_left, upper_left, lower_right]
    };
    indices.extend_from_slice(&quad);
}

fn classify_patch(offset: [f32; 2], outer: bool) -> (Patch, f32) {
    let corner = is_corner(offset);
    if !corner && !is_border(offset) {
        return (Patch::Interior, 0.0);
    }
    let rotation =
        if corner { corner_rotation(offset) } else { side_rotation(offset) };
    if outer {
        let patch =
            if corner { Patch::FatXZOuter } else { Patch::FatXOuter };
        return (patch, rotation);
    }
    // Border tiles are fat on both variable ends and never slim.
    //
    // `build_patch` pins the -x and -z ends at exactly half a tile, so every
    // seam in this layout is a variable end of the tile on the inner side
    // meeting a pinned end of its neighbour: ring n's outer edge meets ring
    // n+1's inner edge, and within a ring a tile's azimuthal end meets the
    // next tile's pinned one. Only the fat end reaches across such a seam —
    // a slim end stops one quad short and leaves a hole the width of a quad,
    // which at the outer rings is most of a metre. Aqua selects the slim
    // patches where a crossfading LOD underneath covers the seam; this port
    // draws one LOD per pixel and has no such pass, so nothing selects them.
    let patch = if corner { Patch::FatXZ } else { Patch::FatX };
    (patch, rotation)
}

fn side_rotation(offset: [f32; 2]) -> f32 {
    match (
        offset[1].abs() >= offset[0].abs(),
        offset[0].is_sign_negative(),
    ) {
        // Quarter turns that put the +x end — the fat one, and the one the
        // horizon skirt stretches — along the tile's own outward direction.
        // The sign has to follow the tile's z: mapping +x onto -z for a tile
        // at +z aims the skirt back across the ring centre, and a stretched
        // end is 550 tiles long, so it sweeps the whole near field.
        (true, _) => offset[1].signum() * std::f32::consts::FRAC_PI_2,
        (false, true) => std::f32::consts::PI,
        (false, false) => 0.0,
    }
}

fn corner_rotation(offset: [f32; 2]) -> f32 {
    // `atan2(x, z)` measures from +z, so pairing it with the -45 degrees that
    // takes the patch's local (+x, +z) diagonal to +x +z is what aims that
    // diagonal at the tile's own corner. With the arguments the other way
    // round the two anti-diagonal corners come out a half turn off, aiming
    // their skirts inward like the side tiles' above.
    offset[1].atan2(offset[0]) - std::f32::consts::FRAC_PI_4
}

fn patch_edges(patch: Patch) -> Edges {
    match patch {
        Patch::Interior => Edges::new(Edge::Regular, Edge::Regular),
        Patch::FatX => Edges::new(Edge::Fat, Edge::Regular),
        Patch::FatXSlimZ => Edges::new(Edge::Fat, Edge::Slim),
        Patch::FatXOuter => Edges::new(Edge::Fat, Edge::Regular).outer_x(),
        Patch::FatXZ => Edges::new(Edge::Fat, Edge::Fat),
        Patch::FatXZOuter => Edges::new(Edge::Fat, Edge::Fat).outer_xz(),
        Patch::SlimX => Edges::new(Edge::Slim, Edge::Regular),
        Patch::SlimXZ => Edges::new(Edge::Slim, Edge::Slim),
        Patch::SlimXFatZ => Edges::new(Edge::Slim, Edge::Fat),
    }
}

fn is_inner(offset: [f32; 2]) -> bool {
    offset[0].abs() == INNER_TILE_OFFSET && offset[1].abs() == INNER_TILE_OFFSET
}

fn is_corner(offset: [f32; 2]) -> bool {
    offset[0].abs() == OUTER_TILE_OFFSET && offset[1].abs() == OUTER_TILE_OFFSET
}

fn is_border(offset: [f32; 2]) -> bool {
    offset[0].abs() == OUTER_TILE_OFFSET || offset[1].abs() == OUTER_TILE_OFFSET
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn centre_ring_is_filled_and_outer_rings_are_hollow() {
        assert_eq!(tile_layout(0).len(), CENTER_TILE_COUNT);
        for lod in 1..WATER_LOD_COUNT {
            assert_eq!(tile_layout(lod).len(), RING_TILE_COUNT, "lod {lod}");
        }
    }

    #[test]
    fn every_patch_has_geometry() {
        for patch in Patch::ALL {
            let mesh = build_patch(patch);
            assert!(!mesh.positions.is_empty(), "{patch:?}");
            assert_eq!(mesh.indices.len() % 3, 0, "{patch:?} indices must be triangles");
            let vertices = mesh.positions.len() as u32;
            assert!(
                mesh.indices.iter().all(|index| *index < vertices),
                "{patch:?} has an out-of-range index",
            );
        }
    }

    /// One tile's world-space x and z bounds: `build_instances`' placement plus
    /// the rotation `water.wgsl`'s `vs_sea` applies.
    fn tile_world_bounds(tile: &Tile, lod: usize) -> [[f32; 2]; 2] {
        let scale = lod_scale(lod);
        let mesh = build_patch(tile.patch);
        let (cos_r, sin_r) = (tile.rotation.cos(), tile.rotation.sin());
        let origin = [tile.offset[0] * scale, tile.offset[1] * scale];
        let mut bounds = [[f32::MAX, f32::MIN], [f32::MAX, f32::MIN]];
        for [x, z] in mesh.positions {
            let (x, z) = (x * scale, z * scale);
            let world = [
                x * cos_r - z * sin_r + origin[0],
                x * sin_r + z * cos_r + origin[1],
            ];
            for axis in 0..2 {
                bounds[axis][0] = bounds[axis][0].min(world[axis]);
                bounds[axis][1] = bounds[axis][1].max(world[axis]);
            }
        }
        bounds
    }

    #[test]
    fn every_ring_meets_the_rings_either_side_of_it() {
        // The seam rule the fat ends exist for: a tile that owns a side of its
        // ring must reach back to the ring's inner edge, which is where the
        // finer ring's outer edge is, and (if it owns an outer side) out to the
        // ring's own outer edge. A slim end stops one quad short of both and
        // the water has a hole a quad wide.
        for lod in 0..WATER_LOD_COUNT {
            let scale = lod_scale(lod);
            let outer = 2.0 * scale;
            // Ring 0 is the filled centre block, so only the rings outside it
            // have an inner edge for their tiles to reach back to.
            let inner = (lod > 0).then(|| 2.0 * lod_scale(lod - 1));
            for tile in tile_layout(lod) {
                let bounds = tile_world_bounds(&tile, lod);
                for axis in 0..2 {
                    let [low, high] = bounds[axis];
                    if let Some(inner) = inner {
                        let reach_inward = low.abs().min(high.abs());
                        assert!(
                            reach_inward <= inner + 1e-3,
                            "lod {lod} tile {:?} does not reach its inner edge {inner}: \
                             spans {low}..{high} on axis {axis}",
                            tile.offset,
                        );
                    }
                    if tile.offset[axis].abs() == OUTER_TILE_OFFSET {
                        assert!(
                            low.abs().max(high.abs()) >= outer - 1e-3,
                            "lod {lod} tile {:?} does not reach its outer edge {outer}: \
                             spans {low}..{high} on axis {axis}",
                            tile.offset,
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn the_horizon_skirt_stretches_away_from_the_ring_centre() {
        // The outermost tiles carry the water to the far plane, and their
        // stretched end is `OUTER_EXTENT_MULTIPLIER` tiles long. A rotation
        // that aims it back across the ring centre therefore draws triangles
        // hundreds of kilometres across over the near field, whose straight
        // edges and flat shading are what this asserts against.
        let lod = WATER_LOD_COUNT - 1;
        for tile in tile_layout(lod) {
            let bounds = tile_world_bounds(&tile, lod);
            for axis in 0..2 {
                if tile.offset[axis].abs() != OUTER_TILE_OFFSET {
                    continue;
                }
                let [low, high] = bounds[axis];
                let side = tile.offset[axis].signum();
                assert!(
                    side * low >= 0.0 && side * high >= 0.0,
                    "lod {lod} tile {:?} crosses the ring centre on axis {axis}: \
                     spans {low}..{high}",
                    tile.offset,
                );
                assert!(
                    side * if side > 0.0 { high } else { low } > 100_000.0,
                    "lod {lod} tile {:?} does not stretch toward the horizon on axis {axis}: \
                     spans {low}..{high}",
                    tile.offset,
                );
            }
        }
    }

    #[test]
    fn horizon_triangles_start_beyond_the_wave_fade_for_every_camera_snap_offset() {
        let lod = WATER_LOD_COUNT - 1;
        let scale = lod_scale(lod);
        let half_snap = super::super::WATER_SNAP * 0.5;
        let mut skirt_triangles = 0;

        for tile in tile_layout(lod) {
            let mesh = build_patch(tile.patch);
            let (sin_r, cos_r) = tile.rotation.sin_cos();
            let positions: Vec<[f32; 2]> = mesh.positions.iter()
                .map(|[x, z]| {
                    [
                        (x * cos_r - z * sin_r + tile.offset[0]) * scale,
                        (x * sin_r + z * cos_r + tile.offset[1]) * scale,
                    ]
                })
                .collect();
            for triangle in mesh.indices.chunks_exact(3) {
                let vertices: [[f32; 2]; 3] =
                    std::array::from_fn(|i| positions[triangle[i] as usize]);
                // Only the horizon skirt has edges longer than a whole tile.
                if !(0..3).any(|i| {
                    let a = vertices[i];
                    let b = vertices[(i + 1) % 3];
                    (a[0] - b[0]).hypot(a[1] - b[1]) > scale
                }) {
                    continue;
                }
                skirt_triangles += 1;
                for [x, z] in vertices {
                    // The closest point in the whole camera snap cell, covering
                    // every offset rather than only its centre or corners.
                    let camera_x = x.clamp(-half_snap, half_snap);
                    let camera_z = z.clamp(-half_snap, half_snap);
                    let distance = (x - camera_x).hypot(z - camera_z);
                    assert!(
                        distance >= horizon_wave_fade_end(),
                        "tile {:?} has a horizon vertex at {distance} m inside \
                         the wave fade for camera [{camera_x}, {camera_z}]",
                        tile.offset,
                    );
                }
            }
        }
        assert!(skirt_triangles > 0, "the test must exercise horizon geometry");
    }

    #[test]
    fn rotations_are_axis_aligned() {
        // Every rotation is a multiple of a quarter turn, which is what lets
        // the rotated fat/slim edges meet exactly instead of nearly.
        let quarter = std::f32::consts::FRAC_PI_2;
        for lod in 0..WATER_LOD_COUNT {
            for tile in tile_layout(lod) {
                let turns = tile.rotation / quarter;
                assert!(
                    (turns - turns.round()).abs() < 1e-5,
                    "lod {lod} tile {:?} rotation {} is not a quarter turn",
                    tile.offset,
                    tile.rotation,
                );
            }
        }
    }
}
