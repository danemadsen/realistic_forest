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
    pub fn to_world(self, planet_vector: DVec3) -> DVec3 {
        self.frame.to_local(planet_vector)
    }

    /// A world-axis vector in planet axes.
    pub fn to_planet(self, world_vector: DVec3) -> DVec3 {
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

    /// World position of a point given as plane coordinates and altitude on
    /// the camera's face: where the renderer draws it (`planetPlaneToWorld`).
    pub fn plane_to_world(&self, planet: &Planet, plane: [f32; 3]) -> [f32; 3] {
        let chart = Planet::chart_of_plane(self.face, DVec2::new(plane[0] as f64, plane[2] as f64));
        let world = self.world_of(planet.position(self.face, chart, plane[1] as f64));
        [world.x as f32, world.y as f32, world.z as f32]
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

    /// The view's far plane: the planar build's `minimum` while the camera
    /// is near the ground, where its fog, terrain shadows and LOD were tuned
    /// to that reach, growing to [`PlanetView::far_distance`] as it climbs
    /// clear of the ground (from 150 m to 3 km above it).
    pub fn far_plane(&self, planet: &Planet, minimum: f64) -> f64 {
        let climb = super::smoothstep(150.0, 3_000.0, self.height_above_ground());
        minimum + (self.far_distance(planet, minimum) - minimum) * climb
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
    /// apply to it. Bit 1: its ground may lie below sea level, so the sea is
    /// drawn over it.
    pub flags: u32,
}
const _: () = assert!(std::mem::size_of::<PatchInstance>() == 48);

pub const PATCH_ON_CAMERA_FACE: u32 = 1;
pub const PATCH_REACHES_SEA: u32 = 2;

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
        up: Vec3,
        fov_y: f64,
        aspect: f64,
        near: f64,
        far: f64,
    ) {
        let forward = view.to_planet(forward.as_dvec3()).normalize();
        let up = view.to_planet(up.as_dvec3());
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
            // The face roots never morph; the clipmap's coarsest level sent
            // the same pair of unreachable distances.
            let (morph_start, morph_end) = if patch.key.depth == 0 {
                (1.0e20, 2.0e20)
            } else {
                ladder.morph(patch.key.depth)
            };
            PatchInstance {
                offset: view.offset_of(point).as_vec3().to_array(),
                face: patch.key.face.index() as u32,
                centre: centre.as_vec2().to_array(),
                chart_min: patch.chart_min.as_vec2().to_array(),
                spacing: patch.spacing as f32,
                morph_start: morph_start as f32,
                morph_end: morph_end as f32,
                flags: if patch.key.face == view.face { PATCH_ON_CAMERA_FACE } else { 0 }
                    | if self.bounds.known(patch.key).0 < crate::constants::SEA_LEVEL as f64 {
                        PATCH_REACHES_SEA
                    } else {
                        0
                    },
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
        // 300 km up a 667.5 km planet the sea's horizon is 700 km away.
        assert!(orbit.far_distance(&planet, 5_800.0) > 700_000.0);
        // On the ground the view keeps the planar build's reach, which its
        // fog and terrain shadows were tuned to.
        assert_eq!(ground.far_plane(&planet, 5_800.0), 5_800.0);
        let hill = PlanetView::new(&planet, Face::PosY, Vec3::new(0.0, 150.0, 0.0), 0.0);
        assert_eq!(hill.far_plane(&planet, 5_800.0), 5_800.0);
        assert_eq!(orbit.far_plane(&planet, 5_800.0), orbit.far_distance(&planet, 5_800.0));
    }

    #[test]
    fn instances_put_whole_nodes_first() {
        let planet = Planet::default();
        let ladder = LodLadder::new(&planet);
        let view = PlanetView::new(&planet, Face::PosY, Vec3::new(5.0, 3.0, 5.0), 1.0);
        let mut terrain = PlanetTerrain::default();
        terrain.select(&planet, &ladder, &view, Vec3::new(0.0, -0.2, -1.0).normalize(), Vec3::Y,
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
