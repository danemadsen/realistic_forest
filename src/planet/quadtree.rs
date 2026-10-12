//! Per-face quadtree LOD over the cube sphere (CDLOD).
//!
//! Each cube face is the root of a quadtree whose depth-`d` nodes are squares
//! of `2 S / 2^d` chart metres, drawn as one shared [`PATCH_CELLS`]-square
//! grid. A node is drawn where it lies within its depth's range of the camera
//! and refined into its four children where they lie within theirs, so the
//! vertex spacing grows with distance like the clipmap rings did: a vertex of
//! spacing `s` is kept out to about [`LOD_RANGE_SPACINGS`] `s`, the reach of a
//! clipmap ring of the same spacing.
//!
//! The outer [`MORPH_FRACTION`] of every depth's range geomorphs onto the next
//! coarser lattice, exactly the clipmap's ring morph, so where a node meets a
//! coarser neighbour its edge vertices have already collapsed onto the
//! neighbour's and no crack opens. The ranges are chosen so that every node's
//! farthest point stays inside its parent's unmorphed zone, which is what
//! makes that hold (see `selection_is_crack_free`).
//!
//! Selection uses the distance to the patch at one reference altitude (the
//! ground under the camera), like the clipmap's planar ring distance, so the
//! morph each vertex computes is a function of the vertex alone and every
//! node that shares it agrees. Culling uses each node's real height range.

use super::{Face, Planet};
use bevy::math::{DVec2, DVec3};
use std::collections::HashMap;

/// Quads along each side of a drawn patch.
pub const PATCH_CELLS: u32 = 32;
/// Vertex spacing of the deepest nodes, the clipmap's finest.
pub const FINEST_SPACING: f64 = 0.25;
/// A vertex of spacing `s` is drawn out to this many `s` from the camera: the
/// reach of the clipmap ring with that spacing.
pub const LOD_RANGE_SPACINGS: f64 = 100.0;
/// The outer fraction of each depth's range over which it morphs onto its
/// parent's lattice.
pub const MORPH_FRACTION: f64 = 0.3;

/// Lowest and highest ground the terrain functions can produce, in metres
/// about sea level, with room for erosion, carving and snow. The horizon test
/// occludes with the lower; unmeasured nodes span both.
pub const TERRAIN_HEIGHT_FLOOR: f64 = -1_200.0;
pub const TERRAIN_HEIGHT_CEILING: f64 = 1_600.0;

/// How many node height ranges are measured per frame. The rest use their
/// parent's until their turn comes.
pub const BOUNDS_BUDGET_PER_FRAME: usize = 96;
/// Height samples along each side of a node when its range is measured.
const BOUNDS_SAMPLES: usize = 9;

/// A quadtree node: `depth` levels below the face root, at column `x` and row
/// `z` of that depth's `2^depth`-square grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeKey {
    pub face: Face,
    pub depth: u8,
    pub x: u32,
    pub z: u32,
}

impl NodeKey {
    pub fn root(face: Face) -> Self {
        Self { face, depth: 0, x: 0, z: 0 }
    }

    pub fn children(self) -> [NodeKey; 4] {
        let depth = self.depth + 1;
        let (x, z) = (self.x * 2, self.z * 2);
        [
            NodeKey { face: self.face, depth, x, z },
            NodeKey { face: self.face, depth, x: x + 1, z },
            NodeKey { face: self.face, depth, x, z: z + 1 },
            NodeKey { face: self.face, depth, x: x + 1, z: z + 1 },
        ]
    }

    pub fn parent(self) -> Option<NodeKey> {
        (self.depth > 0).then(|| NodeKey { face: self.face, depth: self.depth - 1, x: self.x / 2, z: self.z / 2 })
    }
}

/// The LOD ladder of one planet: node sizes, spacings and ranges per depth.
#[derive(Clone, Debug, PartialEq)]
pub struct LodLadder {
    pub half_width: f64,
    /// The deepest depth, whose vertex spacing is [`FINEST_SPACING`].
    pub max_depth: u8,
}

impl LodLadder {
    pub fn new(planet: &Planet) -> Self {
        let face_cells = 2.0 * planet.half_width / FINEST_SPACING / PATCH_CELLS as f64;
        Self { half_width: planet.half_width, max_depth: face_cells.log2().round().max(0.0) as u8 }
    }

    /// Chart metres along a side of a node at `depth`.
    pub fn node_size(&self, depth: u8) -> f64 {
        2.0 * self.half_width / (1u64 << depth) as f64
    }

    /// Chart metres between neighbouring vertices of a node at `depth`.
    pub fn spacing(&self, depth: u8) -> f64 {
        self.node_size(depth) / PATCH_CELLS as f64
    }

    /// Out to here from the camera a node of `depth` is drawn.
    pub fn range(&self, depth: u8) -> f64 {
        LOD_RANGE_SPACINGS * self.spacing(depth)
    }

    /// The distance band over which `depth` morphs onto its parent's lattice.
    /// The face roots have no parent and never morph.
    pub fn morph(&self, depth: u8) -> (f64, f64) {
        if depth == 0 {
            return (f64::MAX, f64::MAX);
        }
        let end = self.range(depth);
        let start = end - MORPH_FRACTION * (end - self.range(depth + 1));
        (start, end)
    }

    /// Chart-space minimum corner of a node.
    pub fn node_min(&self, key: NodeKey) -> DVec2 {
        let size = self.node_size(key.depth);
        DVec2::new(key.x as f64 * size - self.half_width, key.z as f64 * size - self.half_width)
    }

}

/// The geomorph factor a vertex `distance` metres from the camera takes at
/// `depth`: 0 on its own lattice, 1 fully on its parent's. The terrain vertex
/// shader evaluates the same expression; the tests check its stitching here.
#[cfg_attr(not(test), allow(dead_code))]
pub fn morph_factor(ladder: &LodLadder, depth: u8, distance: f64) -> f64 {
    let (start, end) = ladder.morph(depth);
    if start >= end {
        return 0.0;
    }
    super::smoothstep(start, end, distance)
}

/// Snap a chart coordinate onto the lattice of `spacing`, rounding toward the
/// face centre. Every face rounds toward its own centre and a shared edge's
/// midpoint is zero on both sides of it, so the two faces collapse an edge
/// vertex onto the same neighbour. (The clipmap rounded half up, which on a
/// mirrored neighbouring chart would pick the opposite neighbour.) The
/// terrain vertex shader's `planetSnapTowardCentre` is the same expression.
#[cfg_attr(not(test), allow(dead_code))]
pub fn snap_toward_centre(coordinate: f64, spacing: f64) -> f64 {
    (coordinate.abs() / spacing).floor().copysign(coordinate) * spacing
}

/// Planet-centred view volume the selection culls against.
#[derive(Clone, Copy, Debug)]
pub struct ViewVolume {
    /// Camera position, planet-centred metres.
    pub eye: DVec3,
    /// Inward-facing planes `(normal, offset)`: a point `p` is inside when
    /// `normal . p + offset >= 0` for all of them.
    pub planes: [(DVec3, f64); 6],
}

impl ViewVolume {
    /// A symmetric perspective frustum: `forward`, `up` unit, planet axes.
    pub fn perspective(eye: DVec3, forward: DVec3, up: DVec3, fov_y: f64, aspect: f64, near: f64, far: f64) -> Self {
        let right = forward.cross(up).normalize();
        let up = right.cross(forward);
        let half_y = (0.5 * fov_y).tan();
        let half_x = half_y * aspect;
        let plane = |normal: DVec3, through: DVec3| {
            let normal = normal.normalize();
            (normal, -normal.dot(through))
        };
        let planes = [
            plane(forward, eye + forward * near),
            plane(-forward, eye + forward * far),
            plane(forward * half_x + right, eye),
            plane(forward * half_x - right, eye),
            plane(forward * half_y + up, eye),
            plane(forward * half_y - up, eye),
        ];
        Self { eye, planes }
    }

    /// Accepts everything.
    #[cfg(test)]
    pub fn everything(eye: DVec3) -> Self {
        Self { eye, planes: [(DVec3::ZERO, 1.0); 6] }
    }

    pub fn intersects_sphere(&self, centre: DVec3, radius: f64) -> bool {
        self.planes.iter().all(|(normal, offset)| normal.dot(centre) + offset >= -radius)
    }
}

/// The measured height range of nodes, filled a budgeted number per frame:
/// the terrain's chunk streaming. A node is measured the first time the
/// selection visits it and kept: the bare relief it samples never changes.
#[derive(Default)]
pub struct HeightBounds {
    measured: HashMap<NodeKey, (f32, f32)>,
    pending: Vec<NodeKey>,
}

impl HeightBounds {
    /// How many nodes have been measured.
    #[cfg(test)]
    pub fn measured_count(&self) -> usize {
        self.measured.len()
    }

    /// The tightest range known for `key` without queueing it: its own, its
    /// nearest measured ancestor's, or everything the terrain can reach.
    pub fn known(&self, key: NodeKey) -> (f64, f64) {
        let mut node = Some(key);
        while let Some(current) = node {
            if let Some(&(low, high)) = self.measured.get(&current) {
                return (low as f64, high as f64);
            }
            node = current.parent();
        }
        (TERRAIN_HEIGHT_FLOOR, TERRAIN_HEIGHT_CEILING)
    }

    /// The tightest known range for `key`: its own if measured, else its
    /// nearest measured ancestor's (which contains it), else everything the
    /// terrain can reach. Queues `key` for measuring when it lacks its own.
    fn range(&mut self, key: NodeKey) -> (f64, f64) {
        if let Some(&(low, high)) = self.measured.get(&key) {
            return (low as f64, high as f64);
        }
        if self.pending.len() < 4 * BOUNDS_BUDGET_PER_FRAME {
            self.pending.push(key);
        }
        let mut ancestor = key.parent();
        while let Some(parent) = ancestor {
            if let Some(&(low, high)) = self.measured.get(&parent) {
                return (low as f64, high as f64);
            }
            ancestor = parent.parent();
        }
        (TERRAIN_HEIGHT_FLOOR, TERRAIN_HEIGHT_CEILING)
    }

    /// Measure up to [`BOUNDS_BUDGET_PER_FRAME`] queued nodes with `height`,
    /// the ground at a point of a face's chart. The samples miss whatever
    /// rises between them, so the range is widened by the slope the samples
    /// show plus a fixed allowance for the streamed features.
    pub fn measure(&mut self, ladder: &LodLadder, mut height: impl FnMut(Face, DVec2) -> f64) {
        let mut pending = std::mem::take(&mut self.pending);
        // Shallow nodes first: every node below one inherits its range.
        pending.sort_unstable_by_key(|key| key.depth);
        pending.dedup();
        let mut measured = 0;
        for key in pending {
            if measured >= BOUNDS_BUDGET_PER_FRAME {
                break;
            }
            if self.measured.contains_key(&key) {
                continue;
            }
            let minimum = ladder.node_min(key);
            let size = ladder.node_size(key.depth);
            let step = size / (BOUNDS_SAMPLES - 1) as f64;
            let mut low = f64::MAX;
            let mut high = f64::MIN;
            let mut steepest = 0.0f64;
            let mut row_previous = [0.0; BOUNDS_SAMPLES];
            for j in 0..BOUNDS_SAMPLES {
                let mut previous = 0.0;
                for (i, above) in row_previous.iter_mut().enumerate() {
                    let chart = minimum + DVec2::new(i as f64, j as f64) * step;
                    let value = height(key.face, chart);
                    low = low.min(value);
                    high = high.max(value);
                    if i > 0 {
                        steepest = steepest.max((value - previous).abs());
                    }
                    if j > 0 {
                        steepest = steepest.max((value - *above).abs());
                    }
                    previous = value;
                    *above = value;
                }
            }
            let margin = steepest + super::height::FEATURE_HEIGHT_MARGIN;
            self.measured.insert(
                key,
                (
                    (low - margin).max(TERRAIN_HEIGHT_FLOOR) as f32,
                    (high + margin).min(TERRAIN_HEIGHT_CEILING) as f32,
                ),
            );
            measured += 1;
        }
    }
}

/// One patch to draw: a node, or a quadrant of one whose other quadrants were
/// refined (CDLOD's partial selection), at the node's own vertex spacing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Patch {
    pub key: NodeKey,
    /// Chart-space minimum corner of the drawn area.
    pub chart_min: DVec2,
    /// Quads along each side: [`PATCH_CELLS`] for a whole node, half that for
    /// a quadrant.
    pub cells: u32,
    /// Vertex spacing, chart metres.
    pub spacing: f64,
}

impl Patch {
    pub fn chart_centre(&self) -> DVec2 {
        self.chart_min + DVec2::splat(0.5 * self.spacing * self.cells as f64)
    }
}

/// Everything the selection needs to know about this frame.
#[derive(Clone, Copy, Debug)]
pub struct SelectionView {
    pub volume: ViewVolume,
    /// Altitude of the reference surface LOD distances are measured to: the
    /// ground under the camera.
    pub reference_altitude: f64,
    /// Patches farther than this (straight-line, metres) are not drawn.
    pub far: f64,
    /// Whether ground hidden behind the planet's curve is skipped.
    pub horizon_culling: bool,
}

/// The selection's statistics, for the debug overlay.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SelectionStats {
    pub visited: u32,
    pub frustum_culled: u32,
    pub horizon_culled: u32,
    pub drawn: u32,
}

/// Select this frame's patches over all six faces.
pub fn select(
    planet: &Planet,
    ladder: &LodLadder,
    view: &SelectionView,
    bounds: &mut HeightBounds,
    patches: &mut Vec<Patch>,
) -> SelectionStats {
    patches.clear();
    let mut stats = SelectionStats::default();
    let mut walker = Walker { planet, ladder, view, bounds, patches, stats: &mut stats };
    for face in Face::ALL {
        walker.visit(NodeKey::root(face));
    }
    stats
}

struct Walker<'a> {
    planet: &'a Planet,
    ladder: &'a LodLadder,
    view: &'a SelectionView,
    bounds: &'a mut HeightBounds,
    patches: &'a mut Vec<Patch>,
    stats: &'a mut SelectionStats,
}

/// What the selection knows of a node's extent: a sphere enclosing its patch
/// between two altitudes, and the patch's angular extent as seen from the
/// planet's centre.
struct NodeVolume {
    centre: DVec3,
    radius: f64,
    /// Unit direction of the patch centre.
    direction: DVec3,
    /// Largest angle between `direction` and any point of the patch.
    angular_radius: f64,
}

fn node_volume(planet: &Planet, ladder: &LodLadder, key: NodeKey, low: f64, high: f64) -> NodeVolume {
    let minimum = ladder.node_min(key);
    let size = ladder.node_size(key.depth);
    let centre_chart = minimum + DVec2::splat(0.5 * size);
    let direction = planet.direction(key.face, centre_chart);
    let centre = direction * (planet.radius + 0.5 * (low + high));
    let mut radius: f64 = 0.0;
    let mut cosine: f64 = 1.0;
    // Corners and edge midpoints at both altitudes bound the patch to within
    // its sagitta, which is added below.
    for j in 0..3 {
        for i in 0..3 {
            let chart = minimum + DVec2::new(i as f64, j as f64) * (0.5 * size);
            let sample = planet.direction(key.face, chart);
            cosine = cosine.min(sample.dot(direction));
            for altitude in [low, high] {
                radius = radius.max((sample * (planet.radius + altitude) - centre).length());
            }
        }
    }
    let angle = size / planet.radius;
    let sagitta = (planet.radius + high) * (1.0 - (0.5 * angle).min(std::f64::consts::FRAC_PI_2).cos());
    NodeVolume {
        centre,
        radius: radius + sagitta + 1.0e-3 * size,
        direction,
        // A patch's edges bow out from the chords between these samples by
        // well under a tenth of the patch's own angle.
        angular_radius: cosine.clamp(-1.0, 1.0).acos() * 1.1 + 1.0e-6,
    }
}

/// Whether a patch reaching up to `high` metres is wholly hidden behind the
/// smooth planet. From the eye, the lowest possible ground hides everything
/// more than its horizon angle away; a point at radius `r` shows over that
/// ground only within its own horizon angle of the tangent point. So a patch
/// is hidden when the angle (at the planet's centre) between the eye and the
/// patch's nearest point exceeds the two horizon angles together.
fn behind_horizon(planet: &Planet, eye: DVec3, direction: DVec3, angular_radius: f64, high: f64) -> bool {
    let occluder = planet.radius + TERRAIN_HEIGHT_FLOOR;
    let eye_distance = eye.length();
    if eye_distance <= occluder {
        return false;
    }
    let reach = planet.radius + high;
    if reach <= occluder {
        return true;
    }
    let eye_angle = (occluder / eye_distance).acos();
    let point_angle = (occluder / reach).acos();
    let separation = (eye / eye_distance).dot(direction).clamp(-1.0, 1.0).acos();
    separation - angular_radius > eye_angle + point_angle
}

impl Walker<'_> {
    /// CDLOD's `LODSelect`. Returns false when `key` lies beyond its depth's
    /// range, leaving its area to its parent.
    fn visit(&mut self, key: NodeKey) -> bool {
        self.stats.visited += 1;
        let reference = self.view.reference_altitude;
        let lod = node_volume(self.planet, self.ladder, key, reference, reference);
        let lod_distance = ((lod.centre - self.view.volume.eye).length() - lod.radius).max(0.0);
        if key.depth > 0 && lod_distance > self.ladder.range(key.depth) {
            return false;
        }
        let (low, high) = self.bounds.range(key);
        let volume = node_volume(self.planet, self.ladder, key, low, high);
        if (volume.centre - self.view.volume.eye).length() - volume.radius > self.view.far {
            self.stats.frustum_culled += 1;
            return true;
        }
        if self.view.horizon_culling
            && behind_horizon(self.planet, self.view.volume.eye, volume.direction, volume.angular_radius, high)
        {
            self.stats.horizon_culled += 1;
            return true;
        }
        if !self.view.volume.intersects_sphere(volume.centre, volume.radius) {
            self.stats.frustum_culled += 1;
            return true;
        }
        let spacing = self.ladder.spacing(key.depth);
        if key.depth >= self.ladder.max_depth || lod_distance > self.ladder.range(key.depth + 1) {
            self.emit(key, self.ladder.node_min(key), PATCH_CELLS, spacing);
            return true;
        }
        for child in key.children() {
            if !self.visit(child) {
                // The child is out of its own range: draw its quarter at this
                // node's spacing.
                self.emit(key, self.ladder.node_min(child), PATCH_CELLS / 2, spacing);
            }
        }
        true
    }

    fn emit(&mut self, key: NodeKey, chart_min: DVec2, cells: u32, spacing: f64) {
        self.stats.drawn += 1;
        self.patches.push(Patch { key, chart_min, cells, spacing });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eye_above(planet: &Planet, face: Face, chart: DVec2, altitude: f64) -> DVec3 {
        planet.position(face, chart, altitude)
    }

    fn view(planet: &Planet, eye: DVec3, forward: DVec3, reference: f64, far: f64) -> SelectionView {
        let up = eye.normalize();
        let forward = if forward.cross(up).length() < 1e-6 { up.any_orthonormal_vector() } else { forward };
        let _ = planet;
        SelectionView {
            volume: ViewVolume::perspective(eye, forward.normalize(), up, 68f64.to_radians(), 16.0 / 9.0, 0.1, far),
            reference_altitude: reference,
            far,
            horizon_culling: true,
        }
    }

    fn everything(eye: DVec3, reference: f64) -> SelectionView {
        SelectionView {
            volume: ViewVolume::everything(eye),
            reference_altitude: reference,
            far: f64::MAX,
            horizon_culling: false,
        }
    }

    #[test]
    fn the_ladder_reaches_the_clipmaps_finest_spacing() {
        let planet = Planet::default();
        let ladder = LodLadder::new(&planet);
        assert_eq!(ladder.spacing(ladder.max_depth), FINEST_SPACING);
        assert_eq!(ladder.max_depth, 17);
        // The finest nodes reach as far as the clipmap's finest ring.
        assert_eq!(ladder.range(ladder.max_depth), 25.0);
        // Every depth's morph starts after its children's range ends.
        for depth in 1..ladder.max_depth {
            let (start, end) = ladder.morph(depth);
            assert!(start > ladder.range(depth + 1) && start < end);
        }
    }

    #[test]
    fn snapping_is_mirror_symmetric() {
        for coordinate in [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 5.0, -5.0] {
            assert_eq!(snap_toward_centre(-coordinate, 2.0), -snap_toward_centre(coordinate, 2.0));
        }
        assert_eq!(snap_toward_centre(3.0, 2.0), 2.0);
        assert_eq!(snap_toward_centre(-3.0, 2.0), -2.0);
        assert_eq!(snap_toward_centre(4.0, 2.0), 4.0);
        // Face edges lie on every depth's lattice.
        let planet = Planet::default();
        let ladder = LodLadder::new(&planet);
        for depth in 0..=ladder.max_depth {
            let spacing = ladder.spacing(depth);
            assert_eq!(snap_toward_centre(planet.half_width, 2.0 * spacing), planet.half_width);
        }
    }

    /// The patch on `face` drawing chart point `chart`, if any.
    fn patch_at(patches: &[Patch], face: Face, chart: DVec2) -> Option<&Patch> {
        patches.iter().find(|patch| {
            let size = patch.spacing * patch.cells as f64;
            patch.key.face == face
                && chart.x >= patch.chart_min.x - 1e-9
                && chart.x <= patch.chart_min.x + size + 1e-9
                && chart.y >= patch.chart_min.y - 1e-9
                && chart.y <= patch.chart_min.y + size + 1e-9
        })
    }

    /// Check every patch edge against whatever lies across it, on this face or
    /// the next: neighbours differ by at most one depth, and where they
    /// differ, the finer side's vertices on the edge have fully morphed while
    /// the coarser side's have not begun to.
    fn assert_crack_free(planet: &Planet, ladder: &LodLadder, view: &SelectionView, patches: &[Patch]) {
        let reference = view.reference_altitude;
        let eye = view.volume.eye;
        for patch in patches {
            let size = patch.spacing * patch.cells as f64;
            for sample in 0..=8 {
                let t = sample as f64 / 8.0 * size;
                let edges = [
                    (DVec2::new(0.0, t), DVec2::new(-1.0, 0.0)),
                    (DVec2::new(size, t), DVec2::new(1.0, 0.0)),
                    (DVec2::new(t, 0.0), DVec2::new(0.0, -1.0)),
                    (DVec2::new(t, size), DVec2::new(0.0, 1.0)),
                ];
                for (offset, outward) in edges {
                    let on_edge = patch.chart_min + offset;
                    let across = on_edge + outward * 1e-3;
                    let direction = planet.cube_point(patch.key.face, across).normalize();
                    let (face, chart) = if planet.edge_distance(across) < 0.0 {
                        let face = Face::dominant(direction);
                        (face, planet.chart_of(face, direction).unwrap())
                    } else {
                        (patch.key.face, across)
                    };
                    let Some(neighbour) = patch_at(patches, face, chart) else { continue };
                    let ours = patch.key.depth as i32;
                    let theirs = neighbour.key.depth as i32;
                    assert!((ours - theirs).abs() <= 1, "{:?} next to {:?}", patch.key, neighbour.key);
                    let point = planet.position(patch.key.face, on_edge, reference);
                    let distance = (point - eye).length();
                    if theirs < ours {
                        let morph = morph_factor(ladder, patch.key.depth, distance);
                        assert!(morph >= 1.0, "{:?} meets coarser {:?} at morph {morph}", patch.key, neighbour.key);
                        let coarse = morph_factor(ladder, neighbour.key.depth, distance);
                        assert!(coarse <= 0.0, "{:?} meets finer {:?} while morphing {coarse}", neighbour.key, patch.key);
                    }
                }
            }
        }
    }

    #[test]
    fn selection_is_crack_free_on_the_ground() {
        let planet = Planet::default();
        let ladder = LodLadder::new(&planet);
        for &(face, chart) in &[
            (Face::PosY, DVec2::new(3.3, -7.9)),
            (Face::PosY, DVec2::new(1000.0, 2000.0)),
            // Right beside a cube edge, where the selection crosses faces.
            (Face::PosX, DVec2::new(planet.half_width - 6.0, 1234.0)),
            // Near a cube corner.
            (Face::NegY, DVec2::new(planet.half_width - 3.0, -planet.half_width + 5.0)),
        ] {
            for altitude in [1.75, 60.0, 2_000.0] {
                let eye = eye_above(&planet, face, chart, altitude);
                let view = everything(eye, 0.0);
                let mut bounds = HeightBounds::default();
                let mut patches = Vec::new();
                select(&planet, &ladder, &view, &mut bounds, &mut patches);
                assert!(!patches.is_empty());
                assert!(patches.iter().any(|patch| patch.key.depth == ladder.max_depth) || altitude > 25.0);
                assert_crack_free(&planet, &ladder, &view, &patches);
            }
        }
    }

    #[test]
    fn selection_covers_each_face_once() {
        let planet = Planet::default();
        let ladder = LodLadder::new(&planet);
        let eye = eye_above(&planet, Face::PosY, DVec2::new(-40.0, 15.0), 30.0);
        let view = everything(eye, 0.0);
        let mut bounds = HeightBounds::default();
        let mut patches = Vec::new();
        select(&planet, &ladder, &view, &mut bounds, &mut patches);
        // Unculled, the patches tile all six faces: their areas add up.
        for face in Face::ALL {
            let area: f64 = patches
                .iter()
                .filter(|patch| patch.key.face == face)
                .map(|patch| (patch.spacing * patch.cells as f64).powi(2))
                .sum();
            let face_area = (2.0 * planet.half_width).powi(2);
            assert!((area - face_area).abs() < 1e-6 * face_area, "{face:?}: {area} of {face_area}");
        }
    }

    #[test]
    fn culling_keeps_the_patch_count_bounded() {
        let planet = Planet::default();
        let ladder = LodLadder::new(&planet);
        let eye = eye_above(&planet, Face::PosY, DVec2::new(10.0, 10.0), 1.75);
        let frame = planet.local_frame(Face::PosY, DVec2::new(10.0, 10.0));
        let forward = frame.to_planet(DVec3::new(0.0, -0.18f64.sin(), -0.18f64.cos()));
        let view = view(&planet, eye, forward, 0.0, 5_800.0);
        let mut bounds = HeightBounds::default();
        let mut patches = Vec::new();
        let stats = select(&planet, &ladder, &view, &mut bounds, &mut patches);
        assert!(stats.drawn < 700, "{stats:?}");
        assert!(stats.frustum_culled + stats.horizon_culled > 0);
        // Measuring the visited nodes' heights only ever tightens culling.
        let mut tighter = stats;
        for _ in 0..40 {
            bounds.measure(&ladder, |_, _| 0.0);
            tighter = select(&planet, &ladder, &view, &mut bounds, &mut patches);
        }
        assert!(bounds.measured_count() > 100);
        assert!(tighter.drawn <= stats.drawn, "{tighter:?} vs {stats:?}");
        assert_crack_free(&planet, &ladder, &view, &patches);
    }

    #[test]
    fn the_far_side_is_hidden_from_high_up() {
        let planet = Planet::default();
        let ladder = LodLadder::new(&planet);
        let eye = eye_above(&planet, Face::PosY, DVec2::ZERO, 20_000.0);
        let view = SelectionView { horizon_culling: true, ..everything(eye, 0.0) };
        let mut bounds = HeightBounds::default();
        let mut patches = Vec::new();
        let stats = select(&planet, &ladder, &view, &mut bounds, &mut patches);
        assert!(stats.horizon_culled > 0, "{stats:?}");
        // Nothing past the horizon is refined: the antipodal face is only
        // ever its root, which the depth test hides.
        assert!(patches.iter().filter(|patch| patch.key.face == Face::NegY).all(|patch| patch.key.depth == 0));
        assert_crack_free(&planet, &ladder, &view, &patches);
    }

    #[test]
    fn horizon_culling_is_conservative() {
        let planet = Planet::default();
        let eye = planet.position(Face::PosY, DVec2::ZERO, 2.0);
        let floor_reach = |altitude: f64| planet.horizon_distance(altitude, TERRAIN_HEIGHT_FLOOR);
        // A 1.6 km peak shows over the lowest ground until the two grazing
        // lines of sight no longer meet.
        for distance in [50_000.0, 95_000.0, 105_000.0, 150_000.0] {
            let direction = planet.direction(Face::PosY, DVec2::new(distance, 0.0));
            let peak = direction * (planet.radius + TERRAIN_HEIGHT_CEILING);
            let visible = (peak - eye).length() < floor_reach(2.0) + floor_reach(TERRAIN_HEIGHT_CEILING);
            let hidden = behind_horizon(&planet, eye, direction, 0.0, TERRAIN_HEIGHT_CEILING);
            assert!(!(visible && hidden), "{distance}: a visible peak was culled");
        }
        // Ground at the antipode never shows; ground underfoot always does.
        assert!(behind_horizon(&planet, eye, DVec3::NEG_Y, 0.01, 0.0));
        assert!(!behind_horizon(&planet, eye, DVec3::Y, 0.0, 0.0));
        // A patch whose edge reaches back over the horizon is kept.
        assert!(!behind_horizon(&planet, eye, DVec3::new(1.0, 1.0, 0.0).normalize(), 0.8, 0.0));
    }
}
