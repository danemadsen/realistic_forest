//! The planet: six cube faces projected onto a sphere, KSP-PQS style.
//!
//! Every cube face carries its own *chart*: square coordinates `(x, z)` in
//! `[-S, S]` metres, measured from the face centre along the face's east and
//! north axes. The chart is equi-angular: chart metres run in proportion to
//! the angle subtended at the planet centre, so along both axes through a face
//! centre one chart metre is one metre of arc on the sea-level sphere, and the
//! chart is conformal and isometric at the centre. Toward the face's edges
//! and corners it stretches and shears like any cube map: across an edge at
//! its midpoint a chart metre is 0.71 m of ground, and at a corner the axes
//! meet at 120 degrees. `S` is a power of two times the clipmap's finest
//! spacing, which keeps every quadtree vertex on the same global 0.25 m
//! lattice the clipmap sampled and exactly representable in f32.
//!
//! Each chart is a window onto the existing infinite terrain plane: a face's
//! *terrain plane coordinates* are its chart coordinates plus that face's
//! [`FACE_PLANE_OFFSETS`] entry. The spawn face (+Y) has no offset, so the
//! ground around spawn is the plane the planar build drew, sample for sample,
//! and every 2D system (noise, erosion, rivers, plants, buildings, snow) keeps
//! working in the plane coordinates it already uses.
//!
//! Elevation is a function of the planet-centred direction alone: within a
//! face the face's own chart is used, and across a band either side of every
//! cube edge the neighbouring charts are blended by a smooth partition of
//! unity ([`Planet::chart_weights`]). Both sides of an edge therefore sample
//! the same height for the same point on the sphere, and the surface is
//! continuous across every seam however the faces' charts are oriented.
//!
//! Positions are planet-centred metres in f64 on the CPU. The GPU never sees
//! a planet-centred f32 position: the renderer works relative to the camera
//! (`view::PlanetView` and `planet-functions.wgslinc`), so the planet's
//! radius costs no precision; what remains is the precision of the terrain
//! plane coordinates themselves, bounded by the face's extent.

pub mod height;
pub mod quadtree;
pub mod systems;
pub mod view;

use bevy::math::{DVec2, DVec3, Vec2};
use bevy::prelude::Resource;
use std::f64::consts::{FRAC_PI_4, PI};

/// Chart metres from a face centre to its edge on the default planet: 2^21
/// times the clipmap's finest 0.25 m spacing, a 667.5 km sea-level radius.
pub const PLANET_DEFAULT_HALF_WIDTH: f64 = 524_288.0;
/// The smallest and largest half widths `--planet-radius` may snap to.
pub const PLANET_MIN_HALF_WIDTH: f64 = 4_096.0;
pub const PLANET_MAX_HALF_WIDTH: f64 = 4_194_304.0;

/// Chart metres either side of a cube edge over which neighbouring charts are
/// blended. Wide enough that the change of chart is a gentle cross-fade of two
/// landscapes, narrow enough that almost all of every face is a single chart.
pub const PLANET_SEAM_BAND: f64 = 6_000.0;
/// How far inside the seam band the streamed plane features (erosion, river
/// carving, building pads) have faded out, so a chart change inside the band
/// never moves the ground.
pub const PLANET_FEATURE_FADE: f64 = 4_000.0;

/// Altitude of the top of the atmosphere: the haze and sky shell the composite
/// draws around the planet from orbit.
pub const ATMOSPHERE_TOP: f64 = 60_000.0;

/// The air's density scale height (`PLANET_AIR_SCALE_HEIGHT` in the shaders):
/// the fog's thin floor and the sky's air thin with it, so from orbit the
/// planet wears a finite shell of air.
pub const AIR_SCALE_HEIGHT: f32 = 8_000.0;

/// Standard gravity at sea level. The player's own falls use their tuned
/// constant; this is what scales it with altitude.
pub const PLANET_SURFACE_GRAVITY: f64 = 9.81;

/// One of the six cube faces.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub enum Face {
    PosX = 0,
    NegX = 1,
    #[default]
    PosY = 2,
    NegY = 3,
    PosZ = 4,
    NegZ = 5,
}

/// A face's orthonormal basis: its outward `axis`, and the `east` and `north`
/// directions its chart's `x` and `z` grow along. `(east, axis, north)` is
/// right-handed like the planar world's `(X, Y, Z)`: `east x axis = north`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FaceBasis {
    pub axis: DVec3,
    pub east: DVec3,
    pub north: DVec3,
}

impl Face {
    pub const ALL: [Face; 6] = [Face::PosX, Face::NegX, Face::PosY, Face::NegY, Face::PosZ, Face::NegZ];

    pub fn index(self) -> usize {
        self as usize
    }

    /// The spawn face (+Y) has the planar world's own axes. The four
    /// equatorial faces all keep north toward +Y.
    pub fn basis(self) -> FaceBasis {
        let (axis, east) = match self {
            Face::PosX => (DVec3::X, DVec3::Z),
            Face::NegX => (DVec3::NEG_X, DVec3::NEG_Z),
            Face::PosY => (DVec3::Y, DVec3::X),
            Face::NegY => (DVec3::NEG_Y, DVec3::X),
            Face::PosZ => (DVec3::Z, DVec3::NEG_X),
            Face::NegZ => (DVec3::NEG_Z, DVec3::X),
        };
        FaceBasis { axis, east, north: east.cross(axis) }
    }

    /// The face whose axis is closest to `direction`.
    pub fn dominant(direction: DVec3) -> Face {
        let a = direction.abs();
        if a.x >= a.y && a.x >= a.z {
            if direction.x >= 0.0 { Face::PosX } else { Face::NegX }
        } else if a.y >= a.z {
            if direction.y >= 0.0 { Face::PosY } else { Face::NegY }
        } else if direction.z >= 0.0 {
            Face::PosZ
        } else {
            Face::NegZ
        }
    }
}

/// Where each face's chart sits on the infinite terrain plane. The spawn face
/// keeps the plane's origin; the others look at their own stretches of it, a
/// few tens of kilometres away and off every noise period, so no two faces
/// show the same ground. Kept small so plane coordinates stay precise in f32.
pub const FACE_PLANE_OFFSETS: [[f32; 2]; 6] = [
    [37_123.0, -21_457.0],  // +X
    [-45_311.0, 29_873.0],  // -X
    [0.0, 0.0],             // +Y, spawn
    [61_207.0, 53_119.0],   // -Y
    [-27_449.0, -58_231.0], // +Z
    [13_877.0, 71_363.0],   // -Z
];

/// One chart's share of the elevation at a point: its face, the point in
/// that face's chart, and its weight. The weights of a point sum to one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChartWeight {
    pub face: Face,
    pub chart: DVec2,
    pub weight: f64,
}

/// The planet's shape. Inserted as a resource; everything that maps between
/// chart, plane and planet-centred coordinates goes through it.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct Planet {
    /// Sea-level radius in metres, `4 S / pi`.
    pub radius: f64,
    /// `S`: chart metres from a face centre to its edge.
    pub half_width: f64,
    /// Chart metres either side of a cube edge where charts blend.
    pub seam_band: f64,
}

impl Default for Planet {
    fn default() -> Self {
        Self::new(PLANET_DEFAULT_HALF_WIDTH)
    }
}

impl Planet {
    pub fn new(half_width: f64) -> Self {
        Self {
            radius: 4.0 * half_width / PI,
            half_width,
            seam_band: PLANET_SEAM_BAND.min(half_width * 0.25),
        }
    }

    /// The planet whose face half width is the power of two nearest (in
    /// ratio) to the one a sea-level `radius` in metres asks for. Snapping
    /// keeps every quadtree vertex on the 0.25 m terrain lattice.
    pub fn with_radius(radius: f64) -> Self {
        let wanted = (radius.max(1.0) * PI / 4.0).clamp(PLANET_MIN_HALF_WIDTH, PLANET_MAX_HALF_WIDTH);
        Self::new(2f64.powf(wanted.log2().round()))
    }

    /// Equi-angular warp from a face coordinate `u = x / S` to the cube
    /// coordinate: `tan(pi/4 u)`. Odd by construction and exactly +-1 on a
    /// face edge, so two faces build bit-identical cube points along the edge
    /// they share. Defined for `|u| < 2`, which reaches past a neighbouring
    /// face's centre.
    pub fn warp(u: f64) -> f64 {
        let magnitude = u.abs();
        let warped = if magnitude == 1.0 { 1.0 } else { (FRAC_PI_4 * magnitude).tan() };
        warped.copysign(u)
    }

    /// The unnormalised cube point of chart `(x, z)` on `face`.
    pub fn cube_point(&self, face: Face, chart: DVec2) -> DVec3 {
        let basis = face.basis();
        basis.axis
            + basis.east * Self::warp(chart.x / self.half_width)
            + basis.north * Self::warp(chart.y / self.half_width)
    }

    /// Unit planet-centred direction of chart `(x, z)` on `face`.
    pub fn direction(&self, face: Face, chart: DVec2) -> DVec3 {
        self.cube_point(face, chart).normalize()
    }

    /// Planet-centred position of chart `(x, z)` on `face` at `altitude`
    /// metres above sea level.
    pub fn position(&self, face: Face, chart: DVec2, altitude: f64) -> DVec3 {
        self.direction(face, chart) * (self.radius + altitude)
    }

    /// `direction` in `face`'s chart, extended past the face's edge across its
    /// neighbours. `None` on the far hemisphere, where the chart is undefined.
    pub fn chart_of(&self, face: Face, direction: DVec3) -> Option<DVec2> {
        let basis = face.basis();
        let along = direction.dot(basis.axis);
        if along <= 1.0e-9 {
            return None;
        }
        let u = direction.dot(basis.east).atan2(along) / FRAC_PI_4;
        let v = direction.dot(basis.north).atan2(along) / FRAC_PI_4;
        Some(DVec2::new(u, v) * self.half_width)
    }

    /// The face a planet-centred direction lies on and its chart there.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn locate(&self, direction: DVec3) -> (Face, DVec2) {
        let face = Face::dominant(direction);
        let chart = self.chart_of(face, direction).unwrap_or(DVec2::ZERO);
        let limit = self.half_width;
        (face, chart.clamp(DVec2::splat(-limit), DVec2::splat(limit)))
    }

    /// Chart metres from `chart` to the nearest edge of its own face:
    /// positive inside the face, negative past its edge.
    pub fn edge_distance(&self, chart: DVec2) -> f64 {
        self.half_width - chart.x.abs().max(chart.y.abs())
    }

    /// Every chart with a share of the elevation at `direction`. Outside the
    /// seam bands this is the dominant face alone with weight one; inside a
    /// band, two faces (three near a cube corner) cross-fade. Each face's
    /// weight is a smoothstep of its chart's distance to its own edge, which
    /// is mirror-symmetric across that edge (the equi-angular coordinate past
    /// an edge continues as the neighbour's coordinate, `atan(1/t) = pi/2 -
    /// atan(t)`), so the weights are continuous on the sphere and normalising
    /// them makes a partition of unity.
    pub fn chart_weights(&self, direction: DVec3) -> Vec<ChartWeight> {
        let mut weights: Vec<ChartWeight> = Face::ALL
            .iter()
            .filter_map(|&face| {
                let chart = self.chart_of(face, direction)?;
                let weight = self.seam_weight(self.edge_distance(chart));
                (weight > 0.0).then_some(ChartWeight { face, chart, weight })
            })
            .collect();
        let total: f64 = weights.iter().map(|share| share.weight).sum();
        if total > 0.0 {
            for share in &mut weights {
                share.weight /= total;
            }
        }
        weights
    }

    /// Unnormalised weight of a chart a signed `edge_distance` inside its face.
    pub fn seam_weight(&self, edge_distance: f64) -> f64 {
        smoothstep(-self.seam_band, self.seam_band, edge_distance)
    }

    /// How much of the streamed plane features (erosion, river carving,
    /// building pads) the ground at `chart` keeps: one across the face's
    /// interior, fading to nothing [`PLANET_FEATURE_FADE`] metres before the
    /// seam band begins, so the band blends bare procedural relief only.
    pub fn feature_weight(&self, chart: DVec2) -> f64 {
        let inner = self.seam_band + PLANET_FEATURE_FADE;
        smoothstep(self.seam_band, inner, self.edge_distance(chart))
    }

    /// Terrain plane coordinates of a face chart point.
    pub fn plane_of(face: Face, chart: DVec2) -> DVec2 {
        let offset = FACE_PLANE_OFFSETS[face.index()];
        chart + DVec2::new(offset[0] as f64, offset[1] as f64)
    }

    /// Face chart coordinates of terrain plane coordinates on `face`.
    pub fn chart_of_plane(face: Face, plane: DVec2) -> DVec2 {
        let offset = FACE_PLANE_OFFSETS[face.index()];
        plane - DVec2::new(offset[0] as f64, offset[1] as f64)
    }

    /// The chart's derivatives at `chart`: how far the point at `altitude`
    /// moves, in planet-centred metres, per chart metre along x and along z.
    /// Both are unit length at a face centre at sea level.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn chart_jacobian(&self, face: Face, chart: DVec2, altitude: f64) -> (DVec3, DVec3) {
        let basis = face.basis();
        let cube = self.cube_point(face, chart);
        let length = cube.length();
        let direction = cube / length;
        let scale = (self.radius + altitude) * FRAC_PI_4 / (self.half_width * length);
        let secant_squared = |t: f64| 1.0 + t * t;
        let along_x = (basis.east - direction * direction.dot(basis.east))
            * (scale * secant_squared(Self::warp(chart.x / self.half_width)));
        let along_z = (basis.north - direction * direction.dot(basis.north))
            * (scale * secant_squared(Self::warp(chart.y / self.half_width)));
        (along_x, along_z)
    }

    /// The orthonormal local frame at a chart point: `up` is the outward
    /// radial (opposite gravity), `east` follows the chart's x direction, and
    /// `north = east x up` completes the planar world's right-handed `(X, Y,
    /// Z)`. At a face centre it is the face's own basis.
    pub fn local_frame(&self, face: Face, chart: DVec2) -> LocalFrame {
        let up = self.direction(face, chart);
        let east_axis = face.basis().east;
        let east = (east_axis - up * up.dot(east_axis)).normalize();
        LocalFrame { east, up, north: east.cross(up) }
    }

    /// Gravitational acceleration at `altitude`, falling off with the square
    /// of the distance from the centre.
    pub fn gravity_at(&self, altitude: f64) -> f64 {
        let ratio = self.radius / (self.radius + altitude.max(-0.5 * self.radius));
        PLANET_SURFACE_GRAVITY * ratio * ratio
    }

    /// Distance from an eye `altitude` metres above a sphere of radius
    /// `radius + surface` to the point where its line of sight grazes that
    /// sphere: how far the eye can see over a smooth planet.
    pub fn horizon_distance(&self, altitude: f64, surface: f64) -> f64 {
        let sphere = self.radius + surface;
        let eye = self.radius + altitude;
        (eye * eye - sphere * sphere).max(0.0).sqrt()
    }
}

/// An orthonormal tangent frame on the planet: the planar world's `(X, Y, Z)`
/// at that point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LocalFrame {
    pub east: DVec3,
    pub up: DVec3,
    pub north: DVec3,
}

impl LocalFrame {
    /// Express a frame-local `(x, y, z)` vector in planet axes.
    pub fn to_planet(self, local: DVec3) -> DVec3 {
        self.east * local.x + self.up * local.y + self.north * local.z
    }

    /// Express a planet-axis vector in this frame's `(x, y, z)`.
    pub fn to_local(self, planet: DVec3) -> DVec3 {
        DVec3::new(planet.dot(self.east), planet.dot(self.up), planet.dot(self.north))
    }
}

/// A point on or above the planet in the coordinates the game plays in: the
/// face whose chart it is on, its terrain plane coordinates on that face (the
/// planar world's X and Z), and its altitude above sea level (the planar
/// world's Y).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlanetPosition {
    pub face: Face,
    pub plane: DVec2,
    pub altitude: f64,
}

impl PlanetPosition {
    pub fn chart(&self) -> DVec2 {
        Planet::chart_of_plane(self.face, self.plane)
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn to_planet(self, planet: &Planet) -> DVec3 {
        planet.position(self.face, self.chart(), self.altitude)
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn from_planet(planet: &Planet, position: DVec3) -> Self {
        let distance = position.length();
        let direction = if distance > 0.0 { position / distance } else { DVec3::Y };
        let (face, chart) = planet.locate(direction);
        Self { face, plane: Planet::plane_of(face, chart), altitude: distance - planet.radius }
    }
}

/// A move off the edge of the chart a body was on, onto the chart it now lies
/// on. The heading `yaw` (radians, the player's convention: forward is
/// `(sin yaw, 0, -cos yaw)` in plane axes) and any horizontal plane vector are
/// carried across by the rotation between the two local frames.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChartTransfer {
    pub to: PlanetPosition,
    /// Radians to add to a heading measured on the old chart.
    pub yaw_change: f64,
}

impl ChartTransfer {
    /// Rotate a horizontal plane vector `(x, z)` from the old chart's axes into
    /// the new one's.
    pub fn rotate(&self, vector: Vec2) -> Vec2 {
        let (sin, cos) = (self.yaw_change as f32).sin_cos();
        // A heading's forward is (sin yaw, -cos yaw); adding `yaw_change`
        // rotates every horizontal vector the same way.
        Vec2::new(vector.x * cos - vector.y * sin, vector.x * sin + vector.y * cos)
    }
}

impl Planet {
    /// If `position` has walked past the edge of its face, the same point on
    /// the face it is now over. `None` while it is still on its own face.
    pub fn transfer(&self, position: PlanetPosition) -> Option<ChartTransfer> {
        let chart = position.chart();
        if self.edge_distance(chart) >= 0.0 {
            return None;
        }
        // The warp is defined to just short of twice the half width; a point
        // typed in farther out than that (a teleport) is taken from there.
        // Every point closer than that lies over a neighbouring face.
        let limit = 1.9 * self.half_width;
        let chart = chart.clamp(DVec2::splat(-limit), DVec2::splat(limit));
        let direction = self.cube_point(position.face, chart).normalize();
        let face = Face::dominant(direction);
        if face == position.face {
            return None;
        }
        let to_chart = self.chart_of(face, direction)?;
        let limit = self.half_width;
        let to_chart = to_chart.clamp(DVec2::splat(-limit), DVec2::splat(limit));
        let to = PlanetPosition { face, plane: Planet::plane_of(face, to_chart), altitude: position.altitude };
        // A heading is a horizontal direction on the sphere; read it in both
        // frames and take the angle between them.
        let old_frame = self.local_frame(position.face, chart);
        let new_frame = self.local_frame(face, to_chart);
        // The old east is the new one turned by the heading change.
        let east_in_new = new_frame.to_local(old_frame.east);
        Some(ChartTransfer { to, yaw_change: east_in_new.z.atan2(east_in_new.x) })
    }
}

pub fn smoothstep(edge0: f64, edge1: f64, value: f64) -> f64 {
    let t = ((value - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn planet() -> Planet {
        Planet::default()
    }

    #[test]
    fn the_spawn_face_centre_is_the_planar_origin() {
        let planet = planet();
        let frame = planet.local_frame(Face::PosY, DVec2::ZERO);
        assert_eq!(frame.east, DVec3::X);
        assert_eq!(frame.up, DVec3::Y);
        assert_eq!(frame.north, DVec3::Z);
        assert_eq!(Planet::plane_of(Face::PosY, DVec2::new(12.5, -3.0)), DVec2::new(12.5, -3.0));
    }

    #[test]
    fn every_face_basis_is_right_handed_like_the_plane() {
        for face in Face::ALL {
            let basis = face.basis();
            assert!((basis.east.cross(basis.axis) - basis.north).length() < 1e-12);
            assert!(basis.east.dot(basis.axis).abs() < 1e-12);
            assert!((basis.east.cross(basis.north).dot(basis.axis) + 1.0).abs() < 1e-12,
                "{face:?}: (east, axis, north) must be right-handed");
        }
    }

    #[test]
    fn the_radius_follows_the_half_width() {
        let planet = planet();
        // A quarter of a great circle spans a face from edge to edge.
        assert!((planet.radius * PI / 2.0 - 2.0 * planet.half_width).abs() < 1e-6);
        assert_eq!(Planet::with_radius(600_000.0).half_width, 524_288.0);
        assert_eq!(Planet::with_radius(1.0).half_width, PLANET_MIN_HALF_WIDTH);
    }

    #[test]
    fn charts_are_arc_length_along_their_axes() {
        let planet = planet();
        for distance in [1.0, 100.0, 10_000.0, 300_000.0] {
            let a = planet.position(Face::PosY, DVec2::ZERO, 0.0);
            let b = planet.position(Face::PosY, DVec2::new(distance, 0.0), 0.0);
            // From the chord: acos of a dot product cannot resolve a metre.
            let arc = 2.0 * planet.radius * ((b - a).length() / (2.0 * planet.radius)).asin();
            assert!((arc - distance).abs() < 1e-6 * distance.max(1.0), "{distance}: {arc}");
        }
    }

    #[test]
    fn charts_round_trip_through_directions() {
        let planet = planet();
        for face in Face::ALL {
            for &(x, z) in &[(0.0, 0.0), (1234.5, -987.25), (-500_000.0, 400_000.0), (524_288.0, -1.0)] {
                let chart = DVec2::new(x, z);
                let direction = planet.direction(face, chart);
                let back = planet.chart_of(face, direction).unwrap();
                assert!((back - chart).length() < 1e-6, "{face:?} {chart} -> {back}");
            }
        }
    }

    #[test]
    fn faces_build_identical_points_along_shared_edges() {
        let planet = planet();
        let s = planet.half_width;
        for face in Face::ALL {
            for step in -8..=8 {
                let t = step as f64 * s / 8.0;
                for chart in [DVec2::new(s, t), DVec2::new(-s, t), DVec2::new(t, s), DVec2::new(t, -s)] {
                    let cube = planet.cube_point(face, chart);
                    let direction = cube.normalize();
                    // The same point from every other face that owns it.
                    for other in Face::ALL {
                        if other == face {
                            continue;
                        }
                        let Some(other_chart) = planet.chart_of(other, direction) else { continue };
                        if planet.edge_distance(other_chart) < -1e-6 {
                            continue;
                        }
                        let snapped = (other_chart / s * 8.0).round() * s / 8.0;
                        assert!((snapped - other_chart).length() < 1e-6);
                        assert_eq!(planet.cube_point(other, snapped), cube,
                            "{face:?} {chart} vs {other:?} {snapped}");
                    }
                }
            }
        }
    }

    #[test]
    fn chart_weights_partition_unity_and_vary_continuously() {
        let planet = planet();
        let mut previous: Option<f64> = None;
        // Walk from the +Y face centre over its +X edge to the +X centre.
        for step in 0..=4000 {
            let angle = step as f64 / 4000.0 * std::f64::consts::FRAC_PI_2;
            let direction = DVec3::new(angle.sin(), angle.cos(), 0.0);
            let weights = planet.chart_weights(direction);
            let total: f64 = weights.iter().map(|share| share.weight).sum();
            assert!((total - 1.0).abs() < 1e-12);
            let spawn = weights.iter().find(|share| share.face == Face::PosY).map_or(0.0, |s| s.weight);
            // Steps are 262 m; a smoothstep over the 12 km band rises at most
            // 1.5 * 262 / 12000 = 0.033 per step.
            if let Some(previous) = previous {
                assert!((spawn - previous).abs() < 0.04, "jump at step {step}");
            }
            previous = Some(spawn);
        }
        // Deep inside a face only that face counts.
        let inside = planet.chart_weights(planet.direction(Face::NegZ, DVec2::new(1000.0, -2000.0)));
        assert_eq!(inside.len(), 1);
        assert_eq!(inside[0].face, Face::NegZ);
        assert_eq!(inside[0].weight, 1.0);
        // On an edge two faces share equally; at a corner three do.
        let edge = planet.chart_weights(planet.direction(Face::PosY, DVec2::new(planet.half_width, 0.0)));
        assert_eq!(edge.len(), 2);
        assert!(edge.iter().all(|share| (share.weight - 0.5).abs() < 1e-9));
        let corner = planet.chart_weights(DVec3::ONE.normalize());
        assert_eq!(corner.len(), 3);
        assert!(corner.iter().all(|share| (share.weight - 1.0 / 3.0).abs() < 1e-9));
    }

    #[test]
    fn features_fade_before_the_seam_band() {
        let planet = planet();
        let s = planet.half_width;
        assert_eq!(planet.feature_weight(DVec2::ZERO), 1.0);
        assert_eq!(planet.feature_weight(DVec2::new(s - planet.seam_band, 0.0)), 0.0);
        let inside_band = DVec2::new(s - planet.seam_band - 1.0, 0.0);
        assert_eq!(planet.seam_weight(planet.edge_distance(inside_band)), 1.0);
    }

    #[test]
    fn the_jacobian_is_unit_at_the_spawn() {
        let planet = planet();
        let (x, z) = planet.chart_jacobian(Face::PosY, DVec2::ZERO, 0.0);
        assert!((x - DVec3::X).length() < 1e-12);
        assert!((z - DVec3::Z).length() < 1e-12);
        // And it agrees with a finite difference anywhere.
        let chart = DVec2::new(200_000.0, -350_000.0);
        let (x, z) = planet.chart_jacobian(Face::NegX, chart, 120.0);
        let h = 0.01;
        let fx = (planet.position(Face::NegX, chart + DVec2::new(h, 0.0), 120.0)
            - planet.position(Face::NegX, chart - DVec2::new(h, 0.0), 120.0)) / (2.0 * h);
        let fz = (planet.position(Face::NegX, chart + DVec2::new(0.0, h), 120.0)
            - planet.position(Face::NegX, chart - DVec2::new(0.0, h), 120.0)) / (2.0 * h);
        assert!((x - fx).length() < 1e-6, "{x} vs {fx}");
        assert!((z - fz).length() < 1e-6, "{z} vs {fz}");
    }

    #[test]
    fn positions_round_trip_through_the_planet_frame() {
        let planet = planet();
        let start = PlanetPosition { face: Face::PosZ, plane: DVec2::new(-20_000.0, -60_000.0), altitude: 812.5 };
        let back = PlanetPosition::from_planet(&planet, start.to_planet(&planet));
        assert_eq!(back.face, start.face);
        assert!((back.plane - start.plane).length() < 1e-6);
        assert!((back.altitude - start.altitude).abs() < 1e-6);
    }

    #[test]
    fn walking_off_a_face_keeps_the_point_and_the_heading() {
        let planet = planet();
        let s = planet.half_width;
        for face in Face::ALL {
            for chart in [DVec2::new(s + 3.0, 1000.0), DVec2::new(-s - 3.0, -2000.0),
                          DVec2::new(500.0, s + 3.0), DVec2::new(-700.0, -s - 3.0)] {
                let position = PlanetPosition { face, plane: Planet::plane_of(face, chart), altitude: 10.0 };
                let transfer = planet.transfer(position).expect("past the edge");
                assert_ne!(transfer.to.face, face);
                let before = position.to_planet(&planet);
                let after = transfer.to.to_planet(&planet);
                assert!((before - after).length() < 1e-6);
                // A heading points the same way on the sphere before and after.
                for yaw in [0.0f64, 0.7, -2.0] {
                    let heading = |frame: LocalFrame, yaw: f64| {
                        frame.to_planet(DVec3::new(yaw.sin(), 0.0, -yaw.cos()))
                    };
                    let old_frame = planet.local_frame(face, chart);
                    let new_frame = planet.local_frame(transfer.to.face, transfer.to.chart());
                    let a = heading(old_frame, yaw);
                    let b = heading(new_frame, yaw + transfer.yaw_change);
                    assert!((a - b).length() < 1e-9, "{face:?} {chart} yaw {yaw}: {a} vs {b}");
                    let vector = Vec2::new(yaw.sin() as f32, -yaw.cos() as f32);
                    let turned = transfer.rotate(vector);
                    let c = new_frame.to_planet(DVec3::new(turned.x as f64, 0.0, turned.y as f64));
                    assert!((a - c).length() < 1e-5, "rotated vector {a} vs {c}");
                }
            }
        }
        let inside = PlanetPosition { face: Face::PosY, plane: DVec2::new(100.0, 100.0), altitude: 0.0 };
        assert!(planet.transfer(inside).is_none());
        // A teleport typed in far past the edge still lands on a face.
        for plane in [DVec2::new(1.0e6, 0.0), DVec2::new(-1.0e6, 9.0e5), DVec2::new(3.0e6, -3.0e6)] {
            let far = PlanetPosition { face: Face::PosY, plane, altitude: 5.0 };
            let transfer = planet.transfer(far).expect("past the edge");
            assert!(planet.edge_distance(transfer.to.chart()) >= -1e-6, "{plane}");
            assert!(planet.transfer(transfer.to).is_none());
        }
    }

    /// The shaders' copy of the faces and plane offsets is this module's.
    #[test]
    fn the_shader_planet_block_matches_the_faces() {
        let block = include_str!("../../assets/shaders/planet-functions.wgslinc");
        let vectors = |name: &str| -> Vec<[f64; 3]> {
            let start = block.find(&format!("const {name} = array<vec3<f32>, 6>(")).expect(name);
            let body = &block[start..block[start..].find(");").unwrap() + start];
            // The first piece is the declaration itself.
            body.split("vec3<f32>(")
                .skip(1)
                .map(|item| {
                    let numbers: Vec<f64> = item
                        .split(')')
                        .next()
                        .unwrap()
                        .split(',')
                        .map(|value| value.trim().parse::<f64>().unwrap())
                        .collect();
                    [numbers[0], numbers[1], numbers[2]]
                })
                .collect()
        };
        let axes = vectors("PLANET_FACE_AXIS");
        let easts = vectors("PLANET_FACE_EAST");
        let norths = vectors("PLANET_FACE_NORTH");
        for face in Face::ALL {
            let basis = face.basis();
            assert_eq!(DVec3::from(axes[face.index()]), basis.axis, "{face:?} axis");
            assert_eq!(DVec3::from(easts[face.index()]), basis.east, "{face:?} east");
            assert_eq!(DVec3::from(norths[face.index()]), basis.north, "{face:?} north");
        }
        let start = block.find("const PLANET_FACE_PLANE_OFFSET").unwrap();
        let body = &block[start..block[start..].find(");").unwrap() + start];
        let offsets: Vec<[f32; 2]> = body
            .split("vec2<f32>(")
            .skip(1)
            .map(|item| {
                let numbers: Vec<f32> = item
                    .split(')')
                    .next()
                    .unwrap()
                    .split(',')
                    .map(|value| value.trim().parse::<f32>().unwrap())
                    .collect();
                [numbers[0], numbers[1]]
            })
            .collect();
        assert_eq!(offsets, FACE_PLANE_OFFSETS.to_vec());
    }

    /// Every pass that places geometry on the planet or reads altitude
    /// pastes all of the planet block, unchanged.
    #[test]
    fn every_planet_shader_pastes_the_same_block() {
        let block = include_str!("../../assets/shaders/planet-functions.wgslinc").trim();
        let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/shaders");
        let mut pasted = Vec::new();
        for entry in std::fs::read_dir(directory).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().and_then(|extension| extension.to_str()) != Some("wgsl") {
                continue;
            }
            let source = std::fs::read_to_string(&path).unwrap();
            if source.contains("// BEGIN PLANET FUNCTIONS") {
                assert!(source.contains(block), "{}: the planet block diverged", path.display());
                pasted.push(path.file_name().unwrap().to_string_lossy().into_owned());
            }
        }
        for required in [
            "terrain-vs.wgsl", "vegetation.wgsl", "grass.wgsl", "building.wgsl", "lightning.wgsl",
            "composite.wgsl", "water.wgsl", "cloud-probe.wgsl", "cloud-shadow-map.wgsl",
            "precipitation.wgsl",
        ] {
            assert!(pasted.iter().any(|name| name == required), "{required} lacks the planet block");
        }
        assert!(block.contains(&format!("const PLANET_AIR_SCALE_HEIGHT: f32 = {:.1};", AIR_SCALE_HEIGHT)));
    }

    #[test]
    fn gravity_and_horizon_follow_the_sphere() {
        let planet = planet();
        assert!((planet.gravity_at(0.0) - PLANET_SURFACE_GRAVITY).abs() < 1e-12);
        assert!((planet.gravity_at(planet.radius) - PLANET_SURFACE_GRAVITY / 4.0).abs() < 1e-9);
        // 1.75 m eyes on a 667.5 km planet see about 1.5 km of smooth sea.
        let horizon = planet.horizon_distance(1.75, 0.0);
        assert!((horizon - (2.0 * planet.radius * 1.75f64).sqrt()).abs() < 0.01);
    }
}
