//! The planet's ground for the CPU: collision, spawn placement and the
//! quadtree's height bounds. `terrain-vs.wgsl`'s `planetSurfaceHeight` is the
//! GPU twin and follows the same three rules:
//!
//! - Inside a face, away from its seams, the ground is the terrain plane's at
//!   the face's plane coordinates: the procedural relief, and on the face the
//!   camera stands on, the streamed features over it (erosion, river carving,
//!   building pads), exactly as the planar build computed it.
//! - The streamed features fade out over [`super::PLANET_FEATURE_FADE`] metres
//!   before the seam band, and only the camera's face has them at all: their
//!   caches cover one stretch of one plane, and another face's window onto
//!   the plane may overlap it.
//! - In a seam band the neighbouring charts' bare relief is cross-faded by the
//!   partition of unity, as a function of the direction alone.

use super::{Face, Planet};
use crate::erosion::ErosionCache;
use crate::noise::NoiseField;
use bevy::math::DVec2;

/// How far the streamed features can move the ground from the procedural
/// relief: erosion's deepest incision and deposition, a river's carve and a
/// building pad's cut or fill, with room to spare. Height bounds sampled from
/// the bare relief are widened by this.
pub const FEATURE_HEIGHT_MARGIN: f64 = 60.0;

/// The bare procedural relief at a point of a face's chart, seams blended.
pub fn bare_height(planet: &Planet, noise: &NoiseField, face: Face, chart: DVec2) -> f64 {
    if planet.edge_distance(chart) > planet.seam_band {
        let plane = Planet::plane_of(face, chart);
        return crate::noise::base_height(noise, plane.x as f32, plane.y as f32) as f64;
    }
    let direction = planet.cube_point(face, chart).normalize();
    planet
        .chart_weights(direction)
        .iter()
        .map(|share| {
            let plane = Planet::plane_of(share.face, share.chart);
            share.weight * crate::noise::base_height(noise, plane.x as f32, plane.y as f32) as f64
        })
        .sum()
}

/// The ground the player walks on and the terrain pass draws, at plane
/// coordinates `(x, z)` of the camera's face. `visibility_center` is the
/// erosion fade's centre, as for [`crate::erosion::sample_eroded_height`].
pub fn surface_height(
    planet: &Planet,
    cache: &ErosionCache,
    noise: &NoiseField,
    face: Face,
    x: f32,
    z: f32,
    visibility_center: [f32; 2],
) -> f32 {
    let chart = Planet::chart_of_plane(face, DVec2::new(x as f64, z as f64));
    let features = planet.feature_weight(chart);
    if features >= 1.0 {
        return crate::erosion::sample_eroded_height(cache, noise, x, z, visibility_center);
    }
    let bare = bare_height(planet, noise, face, chart) as f32;
    if features <= 0.0 {
        return bare;
    }
    let full = crate::erosion::sample_eroded_height(cache, noise, x, z, visibility_center);
    bare + (full - bare) * features as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::erosion::ErosionCache;

    #[test]
    fn the_spawn_face_interior_is_the_planar_ground() {
        let noise = NoiseField::new();
        let cache = ErosionCache::default();
        let planet = Planet::default();
        for &(x, z) in &[(0.0f32, 0.0f32), (1500.0, -730.0), (-90_000.0, 210_000.0)] {
            let planar = crate::erosion::sample_eroded_height(&cache, &noise, x, z, [0.0, 0.0]);
            let planetary = surface_height(&planet, &cache, &noise, Face::PosY, x, z, [0.0, 0.0]);
            assert_eq!(planar, planetary);
        }
    }

    #[test]
    fn the_ground_is_continuous_over_a_cube_edge() {
        let noise = NoiseField::new();
        let planet = Planet::default();
        let s = planet.half_width;
        let band = planet.seam_band;
        // March across the +Y/+X edge in 5 cm steps where anything changes:
        // into the band, over the edge and out of the band on the far side.
        // The relief cannot climb a metre in 5 cm; a chart mix-up would.
        for centre in [s - band, s, s + band] {
            let mut previous: Option<f64> = None;
            for step in -400..=400 {
                let chart = DVec2::new(centre + step as f64 * 0.05, 777.0);
                let direction = planet.cube_point(Face::PosY, chart).normalize();
                let (face, local) = planet.locate(direction);
                let height = bare_height(&planet, &noise, face, local);
                if let Some(previous) = previous {
                    assert!((height - previous).abs() < 1.0, "jump at {chart}: {previous} -> {height}");
                }
                previous = Some(height);
            }
        }
    }

    #[test]
    fn both_faces_agree_on_the_edge() {
        let noise = NoiseField::new();
        let planet = Planet::default();
        let s = planet.half_width;
        for t in [-400_000.0, -3.0, 0.0, 12_345.0] {
            let chart = DVec2::new(s, t);
            let direction = planet.cube_point(Face::PosY, chart).normalize();
            let other = planet.chart_of(Face::PosX, direction).unwrap();
            let a = bare_height(&planet, &noise, Face::PosY, chart);
            let b = bare_height(&planet, &noise, Face::PosX, other);
            assert!((a - b).abs() < 1e-3, "{t}: {a} vs {b}");
        }
    }
}
