//! Persistent, world-anchored snow compaction and walking collision.
//!
//! Only touched 8 m tiles are retained. A dense nearby window is the GPU
//! snapshot; scrolling that window copies tile rows rather than resampling
//! every texel or discarding tracks behind the player.

use bevy::math::Vec2;
use bevy::prelude::Resource;
use forest_core::constants::SEA_LEVEL;
#[cfg(test)]
use forest_core::constants::SNOWLINE_ALTITUDE;
use forest_core::noise::NoiseField;
use forest_erosion::{self as erosion, ErosionCache};
use std::collections::HashMap;
use std::sync::Arc;

pub const SNOW_DEPTH: f32 = 0.32;
pub const SNOW_TEXEL_SIZE: f32 = 0.125;
pub const SNOW_MAP_SIZE: usize = 1024;
const TILE_CELLS: usize = 64;
const TILE_SPAN: f32 = TILE_CELLS as f32 * SNOW_TEXEL_SIZE;
const MAP_TILES: usize = SNOW_MAP_SIZE / TILE_CELLS;
const CORE_RADIUS: f32 = 0.40;
const EDGE_RADIUS: f32 = 0.65;
const SLOPE_SAMPLE_DISTANCE: f32 = 4.0;
type TileCoordinates = (i64, i64);
type SnowTile = Box<[u8; TILE_CELLS * TILE_CELLS]>;

#[derive(Resource)]
pub struct SnowState {
    tiles: HashMap<TileCoordinates, SnowTile>,
    minimum_tile: TileCoordinates,
    local_pixels: Arc<[u8]>,
    revision: u64,
}

impl Default for SnowState {
    fn default() -> Self {
        Self {
            tiles: HashMap::new(),
            minimum_tile: (-(MAP_TILES as i64) / 2, -(MAP_TILES as i64) / 2),
            local_pixels: vec![0; SNOW_MAP_SIZE * SNOW_MAP_SIZE].into(),
            revision: 1,
        }
    }
}

impl SnowState {
    /// World minimum XZ, metres per texel, and uncompacted snow depth.
    pub fn mapping(&self) -> [f32; 4] {
        [
            self.minimum_tile.0 as f32 * TILE_SPAN,
            self.minimum_tile.1 as f32 * TILE_SPAN,
            SNOW_TEXEL_SIZE,
            SNOW_DEPTH,
        ]
    }

    /// An immutable upload snapshot. The next write preserves any snapshot
    /// already extracted into the render world through Arc copy-on-write.
    pub fn pixels(&self) -> Arc<[u8]> {
        Arc::clone(&self.local_pixels)
    }

    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// The map moves on the same 8 m lattice as its persistent tiles.
    pub fn recenter(&mut self, position: Vec2) {
        if !position.is_finite() {
            return;
        }
        let half = MAP_TILES as i64 / 2;
        let minimum = (
            (position.x / TILE_SPAN).floor() as i64 - half,
            (position.y / TILE_SPAN).floor() as i64 - half,
        );
        if minimum == self.minimum_tile {
            return;
        }
        self.minimum_tile = minimum;
        let mut pixels = vec![0; SNOW_MAP_SIZE * SNOW_MAP_SIZE];
        for map_z in 0..MAP_TILES {
            for map_x in 0..MAP_TILES {
                let key = (minimum.0 + map_x as i64, minimum.1 + map_z as i64);
                let Some(tile) = self.tiles.get(&key) else {
                    continue;
                };
                for row in 0..TILE_CELLS {
                    let source = row * TILE_CELLS;
                    let target = (map_z * TILE_CELLS + row) * SNOW_MAP_SIZE + map_x * TILE_CELLS;
                    pixels[target..target + TILE_CELLS]
                        .copy_from_slice(&tile[source..source + TILE_CELLS]);
                }
            }
        }
        self.local_pixels = pixels.into();
        self.revision = self.revision.wrapping_add(1);
    }

    fn cell(&self, x: i64, z: i64) -> u8 {
        let key = (
            x.div_euclid(TILE_CELLS as i64),
            z.div_euclid(TILE_CELLS as i64),
        );
        self.tiles.get(&key).map_or(0, |tile| {
            tile[z.rem_euclid(TILE_CELLS as i64) as usize * TILE_CELLS
                + x.rem_euclid(TILE_CELLS as i64) as usize]
        })
    }

    /// Matches linear GPU sampling: world cell i's sample is at i + 0.5,
    /// including the cells on either side of negative tile boundaries.
    pub fn compression(&self, position: Vec2) -> f32 {
        if !position.is_finite() {
            return 0.0;
        }
        let pixel = position / SNOW_TEXEL_SIZE - Vec2::splat(0.5);
        let floor = pixel.floor();
        let x = floor.x as i64;
        let z = floor.y as i64;
        let fraction = pixel - floor;
        let at = |px, pz| self.cell(px, pz) as f32 / 255.0;
        let back = at(x, z) + (at(x + 1, z) - at(x, z)) * fraction.x;
        let front = at(x, z + 1) + (at(x + 1, z + 1) - at(x, z + 1)) * fraction.x;
        back + (front - back) * fraction.y
    }

    /// Compact a swept foot-width capsule without gaps at any walking speed.
    /// Max accumulation keeps repeated footsteps bounded and persistent.
    pub fn stamp_segment(&mut self, start: Vec2, end: Vec2) {
        if !start.is_finite() || !end.is_finite() {
            return;
        }
        let minimum = ((start.min(end) - Vec2::splat(EDGE_RADIUS)) / SNOW_TEXEL_SIZE
            - Vec2::splat(0.5))
        .ceil();
        let maximum = ((start.max(end) + Vec2::splat(EDGE_RADIUS)) / SNOW_TEXEL_SIZE
            - Vec2::splat(0.5))
        .floor();
        let segment = end - start;
        let length_squared = segment.length_squared();
        let mut changed = false;
        for z in minimum.y as i64..=maximum.y as i64 {
            for x in minimum.x as i64..=maximum.x as i64 {
                let center = Vec2::new(x as f32 + 0.5, z as f32 + 0.5) * SNOW_TEXEL_SIZE;
                let along = if length_squared > 0.0 {
                    ((center - start).dot(segment) / length_squared).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let distance = center.distance(start + segment * along);
                if distance >= EDGE_RADIUS {
                    continue;
                }
                let amount = ((1.0 - smooth_hermite(CORE_RADIUS, EDGE_RADIUS, distance)) * 255.0)
                    .round() as u8;
                if amount == 0 {
                    continue;
                }
                let key = (
                    x.div_euclid(TILE_CELLS as i64),
                    z.div_euclid(TILE_CELLS as i64),
                );
                let tile = self
                    .tiles
                    .entry(key)
                    .or_insert_with(|| Box::new([0; TILE_CELLS * TILE_CELLS]));
                let index = z.rem_euclid(TILE_CELLS as i64) as usize * TILE_CELLS
                    + x.rem_euclid(TILE_CELLS as i64) as usize;
                if amount <= tile[index] {
                    continue;
                }
                tile[index] = amount;
                changed = true;
                let local_x = x - self.minimum_tile.0 * TILE_CELLS as i64;
                let local_z = z - self.minimum_tile.1 * TILE_CELLS as i64;
                if (0..SNOW_MAP_SIZE as i64).contains(&local_x)
                    && (0..SNOW_MAP_SIZE as i64).contains(&local_z)
                {
                    Arc::make_mut(&mut self.local_pixels)
                        [local_z as usize * SNOW_MAP_SIZE + local_x as usize] = amount;
                }
            }
        }
        if changed {
            self.revision = self.revision.wrapping_add(1);
        }
    }
}

fn smooth_hermite(start: f32, end: f32, value: f32) -> f32 {
    let t = ((value - start) / (end - start)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

pub use forest_core::snow::*;

/// The base surface and its snow cover are cached for the rest of a walking
/// update, so stamping only needs a fresh compression lookup for collision.
pub struct SnowSurface {
    pub base_height: f32,
    pub coverage: f32,
}

impl SnowSurface {
    pub fn height(&self, snow: &SnowState, position: Vec2) -> f32 {
        self.base_height + self.coverage * SNOW_DEPTH * (1.0 - snow.compression(position))
    }
}

pub fn sample_surface(
    erosion: &ErosionCache,
    noise: &NoiseField,
    position: Vec2,
    visibility_center: [f32; 2],
) -> SnowSurface {
    let base_height =
        erosion::sample_eroded_height(erosion, noise, position.x, position.y, visibility_center);
    if base_height - SEA_LEVEL < SNOW_FLOOR {
        return SnowSurface {
            base_height,
            coverage: 0.0,
        };
    }
    let at = |offset: Vec2| {
        let p = position + offset;
        erosion::sample_eroded_height(erosion, noise, p.x, p.y, visibility_center)
    };
    let dx = Vec2::new(SLOPE_SAMPLE_DISTANCE, 0.0);
    let dz = Vec2::new(0.0, SLOPE_SAMPLE_DISTANCE);
    let gradient = Vec2::new(at(dx) - at(-dx), at(dz) - at(-dz)) / (2.0 * SLOPE_SAMPLE_DISTANCE);
    let concavity =
        erosion::sample_eroded_concavity(erosion, position.x, position.y, visibility_center);
    // Snow lies on dry ground only, thinning to nothing at a river's or
    // lake's waterline, as terrain_vs.wgsl draws it: no bed stands above its
    // water, and a wading player tramples no trail in it.
    let dry = erosion.rivers.as_ref().map_or(1.0, |network| {
        smooth_hermite(
            0.0,
            1.0,
            network
                .envelope(position.x, position.y)
                .bank_at(base_height),
        )
    });
    SnowSurface {
        base_height,
        coverage: dry * snow_coverage(base_height, gradient, concavity),
    }
}

#[cfg(test)]
#[path = "snow_tests.rs"]
mod tests;
