//! Where the camera is on the planet, and the world frame the renderer draws
//! in.
//!
//! The renderer's "world" is the planar world bent onto the planet around the
//! camera: a Euclidean frame whose axes are the camera's local east, up and
//! north, with the camera at its terrain plane coordinates and altitude. At
//! the camera it coincides with the terrain plane, so everything drawn near
//! the eye looks as it did on the plane, while the planet curves away from it
//! exactly. See `planet-functions.wgslinc` for the shaders' side.

use super::quadtree::{self, HeightBounds, LodLadder, Patch, SelectionStats, SelectionView, ViewVolume};
use super::{Face, LocalFrame, Planet};
use bevy::math::{DVec2, DVec3, Vec3};
use bevy::prelude::Resource;

/// The camera's place on the planet this frame.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct PlanetView {
    /// The face the camera is over; its chart is the world's plane.
    pub face: Face,
    /// The camera's position in its face's chart.
    pub chart: DVec2,
    /// The camera's terrain plane coordinates and altitude: the world
    /// position it is drawn from.
    pub plane: DVec3,
    /// Planet-centred position of the eye.
    pub eye: DVec3,
    /// The world's axes in planet axes: east, up and north at the camera.
    pub frame: LocalFrame,
    /// Altitude of the ground under the camera, the quadtree's reference.
    pub ground: f64,
}

impl Default for PlanetView {
    fn default() -> Self {
        Self::new(&Planet::default(), Face::PosY, Vec3::ZERO, 0.0)
    }
}

impl PlanetView {
    /// The view from terrain plane position `plane` (x, altitude, z) on `face`.
    pub fn new(planet: &Planet, face: Face, plane: Vec3, ground: f64) -> Self {
        let plane = plane.as_dvec3();
        let chart = Planet::chart_of_plane(face, DVec2::new(plane.x, plane.z));
        Self {
            face,
            chart,
            plane,
            eye: planet.position(face, chart, plane.y),
            frame: planet.local_frame(face, chart),
            ground,
        }
    }

    /// A planet-axis vector in world axes.
    pub fn to_world(&self, planet_vector: DVec3) -> DVec3 {
        self.frame.to_local(planet_vector)
    }

    /// A world-axis vector in planet axes.
    pub fn to_planet(&self, world_vector: DVec3) -> DVec3 {
        self.frame.to_planet(world_vector)
    }

    /// World offset from the camera of a planet-centred point.
    pub fn offset_of(&self, point: DVec3) -> DVec3 {
        self.frame.to_local(point - self.eye)
    }

    /// World position of a planet-centred point.
    pub fn world_of(&self, point: DVec3) -> DVec3 {
        self.plane + self.offset_of(point)
    }

    /// The camera's altitude above the ground under it.
    pub fn height_above_ground(&self) -> f64 {
        self.plane.y - self.ground
    }

    /// How far the view must reach to show everything above the horizon:
    /// the line of sight that grazes sea level, plus what the highest ground
    /// beyond it can show, with the planar build's reach as the floor.
    pub fn far_distance(&self, planet: &Planet, minimum: f64) -> f64 {
        let over_sea = planet.horizon_distance(self.plane.y.max(0.0), 0.0);
        let beyond = planet.horizon_distance(quadtree::TERRAIN_HEIGHT_CEILING, 0.0);
        (over_sea + beyond).max(minimum)
    }
}

/// One drawn terrain patch as the vertex shader reads it, per instance.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PatchInstance {
    /// World offset from the camera of the patch centre at the reference
    /// altitude, computed in f64.
    pub offset: [f32; 3],
    /// The patch's face.
    pub face: u32,
    /// The patch centre in its face's chart.
    pub centre: [f32; 2],
    /// The drawn area's minimum corner in the face's chart.
    pub chart_min: [f32; 2],
    /// Chart metres between neighbouring vertices.
    pub spacing: f32,
    /// The camera distances over which the patch morphs onto the next
    /// coarser lattice.
    pub morph_start: f32,
    pub morph_end: f32,
    /// Bit 0: the patch is on the camera's face, so the streamed features
    /// apply to it.
    pub flags: u32,
}
const _: () = assert!(std::mem::size_of::<PatchInstance>() == 48);

pub const PATCH_ON_CAMERA_FACE: u32 = 1;

/// The quadtree's per-frame output: patches to draw, whole nodes first and
/// then quadrants, so each mesh is one instanced draw.
#[derive(Resource, Default)]
pub struct PlanetTerrain {
    pub bounds: HeightBounds,
    pub patches: Vec<Patch>,
    pub instances: Vec<PatchInstance>,
    /// How many of `instances` (from the front) are whole nodes.
    pub whole: usize,
    pub stats: SelectionStats,
}

impl PlanetTerrain {
    /// Select and lay out this frame's patches for a camera looking along
    /// `forward` (world axes) with a vertical field of view `fov_y` radians.
    #[allow(clippy::too_many_arguments)]
    pub fn select(
        &mut self,
        planet: &Planet,
        ladder: &LodLadder,
        view: &PlanetView,
        forward: Vec3,
        fov_y: f64,
        aspect: f64,
        near: f64,
        far: f64,
    ) {
        let forward = view.to_planet(forward.as_dvec3()).normalize();
        let up = view.frame.up;
        let up = if forward.cross(up).length_squared() < 1e-12 { view.frame.north } else { up };
        let selection = SelectionView {
            volume: ViewVolume::perspective(view.eye, forward, up, fov_y, aspect, near, far),
            reference_altitude: view.ground,
            far,
            horizon_culling: true,
        };
        self.stats = quadtree::select(planet, ladder, &selection, &mut self.bounds, &mut self.patches);
        self.patches.sort_by_key(|patch| std::cmp::Reverse(patch.cells));
        self.whole = self.patches.iter().take_while(|patch| patch.cells == quadtree::PATCH_CELLS).count();
        self.instances.clear();
        self.instances.extend(self.patches.iter().map(|patch| {
            let centre = patch.chart_centre();
            let point = planet.position(patch.key.face, centre, view.ground);
            let (morph_start, morph_end) = ladder.morph(patch.key.depth);
            PatchInstance {
                offset: view.offset_of(point).as_vec3().to_array(),
                face: patch.key.face.index() as u32,
                centre: centre.as_vec2().to_array(),
                chart_min: patch.chart_min.as_vec2().to_array(),
                spacing: patch.spacing as f32,
                morph_start: morph_start.min(f32::MAX as f64) as f32,
                morph_end: morph_end.min(f32::MAX as f64) as f32,
                flags: if patch.key.face == view.face { PATCH_ON_CAMERA_FACE } else { 0 },
            }
        }));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_world_is_the_plane_at_the_spawn() {
        let planet = Planet::default();
        let view = PlanetView::new(&planet, Face::PosY, Vec3::new(0.0, 20.0, 0.0), 2.0);
        assert_eq!(view.to_world(DVec3::new(0.3, 0.4, -0.5)), DVec3::new(0.3, 0.4, -0.5));
        // A point ten metres of sea-level arc east at the same altitude sits
        // a hair over ten metres east and only the curvature's microns lower.
        let east = planet.position(Face::PosY, DVec2::new(10.0, 0.0), 20.0);
        let world = view.world_of(east);
        let angle = 10.0 / planet.radius;
        assert!((world.x - (planet.radius + 20.0) * angle.sin()).abs() < 1e-6);
        assert!(world.z.abs() < 1e-9);
        assert!(world.y < 20.0 && world.y > 20.0 - 1e-3);
    }

    #[test]
    fn the_far_distance_grows_with_altitude() {
        let planet = Planet::default();
        let ground = PlanetView::new(&planet, Face::PosY, Vec3::new(0.0, 2.0, 0.0), 0.0);
        let orbit = PlanetView::new(&planet, Face::PosY, Vec3::new(0.0, 300_000.0, 0.0), 0.0);
        assert!(ground.far_distance(&planet, 5_800.0) < 300_000.0);
        // 300 km up a 667.6 km planet the sea's horizon is 700 km away.
        assert!(orbit.far_distance(&planet, 5_800.0) > 700_000.0);
    }

    #[test]
    fn instances_put_whole_nodes_first() {
        let planet = Planet::default();
        let ladder = LodLadder::new(&planet);
        let view = PlanetView::new(&planet, Face::PosY, Vec3::new(5.0, 3.0, 5.0), 1.0);
        let mut terrain = PlanetTerrain::default();
        terrain.select(&planet, &ladder, &view, Vec3::new(0.0, -0.2, -1.0).normalize(),
                       68f64.to_radians(), 16.0 / 9.0, 0.1, 6_000.0);
        assert!(terrain.whole > 0 && terrain.whole <= terrain.instances.len());
        assert!(terrain.patches[..terrain.whole].iter().all(|patch| patch.cells == quadtree::PATCH_CELLS));
        assert!(terrain.patches[terrain.whole..].iter().all(|patch| patch.cells < quadtree::PATCH_CELLS));
        // The patch under the camera is close; its offset is camera-relative.
        let nearest = terrain.instances.iter()
            .map(|instance| Vec3::from(instance.offset).length())
            .fold(f32::MAX, f32::min);
        assert!(nearest < 10.0, "{nearest}");
    }
}
