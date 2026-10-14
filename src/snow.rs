//! Persistent, world-anchored snow compaction and walking collision.
//!
//! Only touched 8 m tiles are retained. A dense nearby window is the GPU
//! snapshot; scrolling that window copies tile rows rather than resampling
//! every texel or discarding tracks behind the player.

use crate::constants::{SEA_LEVEL, SNOWLINE_ALTITUDE};
use crate::erosion::{self, ErosionCache};
use crate::noise::NoiseField;
use bevy::math::Vec2;
use bevy::prelude::Resource;
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

/// Half-width of the snowline band, metres.
pub const SNOW_BAND: f32 = 13.0;
/// Metres of snowline per unit of spring-sun exposure.
pub const SNOW_MELT_ASPECT: f32 = 12.0;
/// Scale of the concavity ring, metres: Gaea's coarse cell, under which fine
/// relief is restored in proportion to the pack's depth.
pub const SNOW_BURIAL: f32 = 12.0;
/// 2A/D in band depths (Gaea's adhered snow mass): a full pack holds to tan
/// 0.80, exactly where the uniform pack began to thin.
pub const SNOW_HOLD: f32 = 1.45 / 0.65;
/// Tan of the slip-off angle a vanishing pack still holds (55.4 degrees).
pub const SNOW_SLIP_TAN: f32 = 1.45;
/// Rise per metre toward the lee: the slope test is tilted toward the
/// prevailing wind, so lee faces hold steep snow that windward faces shed.
pub const SNOW_WIND_TILT: f32 = 0.15;
/// The prevailing wind blows toward 70 degrees (+X is 0, +Z is 90), the
/// default cloud wind, as a climate constant rather than the live weather.
pub const SNOW_WIND_TO: Vec2 = Vec2::new(0.34202, 0.93969);
/// Width of the smooth minimum between the pack and what its slope holds,
/// band depths.
pub const SNOW_SMOOTH: f32 = 0.2;
/// Below this height above the sea no slope, aspect or hollow holds snow.
pub const SNOW_FLOOR: f32 = SNOWLINE_ALTITUDE - SNOW_BAND - 0.62 * SNOW_MELT_ASPECT;

/// Keep this cover rule identical to snowCoverage in the terrain shaders
/// (snow-functions.wgslinc): the late-season pack as Gaea's snowfall settles
/// it, in closed form. `concavity` is the 12 m ring's mean height minus the
/// point (+ in hollows), as [`erosion::sample_eroded_concavity`] reads it.
/// Trail slopes do not affect the material's snow cover.
pub fn snow_coverage(base_height: f32, gradient: Vec2, concavity: f32) -> f32 {
    let normal = bevy::math::Vec3::new(-gradient.x, 1.0, -gradient.y).normalize();
    let melt_sun = bevy::math::Vec3::new(-0.22, 0.62, -0.76).normalize();
    let aspect = (0.62 - normal.dot(melt_sun).max(0.0)) * SNOW_MELT_ASPECT;
    let pack = (base_height - SEA_LEVEL + aspect - SNOWLINE_ALTITUDE + SNOW_BAND) / (2.0 * SNOW_BAND);
    let depth = pack * (1.0 + concavity / SNOW_BURIAL).clamp(0.5, 1.5);
    let held = SNOW_HOLD * (1.0 - (gradient + SNOW_WIND_TILT * SNOW_WIND_TO).length() / SNOW_SLIP_TAN);
    let blend = (SNOW_SMOOTH - (depth - held).abs()).max(0.0) / SNOW_SMOOTH;
    smooth_hermite(0.0, 1.0, depth.min(held) - blend * blend * SNOW_SMOOTH * 0.25)
}

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
    let concavity = erosion::sample_eroded_concavity(erosion, position.x, position.y, visibility_center);
    // Snow lies on dry ground only, thinning to nothing at a river's or
    // lake's waterline, as terrain-vs.wgsl draws it: no bed stands above its
    // water, and a wading player tramples no trail in it.
    let dry = erosion.rivers.as_ref().map_or(1.0, |network| {
        smooth_hermite(0.0, 1.0, network.envelope(position.x, position.y).bank_at(base_height))
    });
    SnowSurface {
        base_height,
        coverage: dry * snow_coverage(base_height, gradient, concavity),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracks_survive_scrolling_and_revisiting_negative_tile_boundaries() {
        let mut snow = SnowState::default();
        let start = Vec2::new(-8.35, -0.17);
        let end = Vec2::new(-7.55, 0.31);
        snow.stamp_segment(start, end);
        let before = snow.compression(start.lerp(end, 0.55));
        let saved = snow.pixels();
        snow.recenter(Vec2::new(800.0, -800.0));
        assert!(snow.pixels().iter().all(|&value| value == 0));
        assert_eq!(snow.compression(start.lerp(end, 0.55)), before);
        snow.recenter(Vec2::ZERO);
        assert_eq!(snow.pixels().as_ref(), saved.as_ref());
        assert_eq!(before, 1.0);
    }

    #[test]
    fn bilinear_sampling_uses_texel_centers() {
        let mut snow = SnowState::default();
        let mut tile = Box::new([0; TILE_CELLS * TILE_CELLS]);
        tile[0] = 255;
        snow.tiles.insert((0, 0), tile);
        assert_eq!(snow.compression(Vec2::splat(SNOW_TEXEL_SIZE * 0.5)), 1.0);
        assert_eq!(snow.compression(Vec2::splat(SNOW_TEXEL_SIZE)), 0.25);
        assert_eq!(snow.compression(Vec2::ZERO), 0.25);
        assert_eq!(snow.compression(Vec2::splat(SNOW_TEXEL_SIZE * 1.5)), 0.0);
    }

    #[test]
    fn nonaligned_sweep_has_a_continuous_flush_core_and_soft_edges() {
        let mut snow = SnowState::default();
        let start = Vec2::new(-2.37, 1.19);
        let end = Vec2::new(3.23, 4.68);
        snow.stamp_segment(start, end);
        let normal = Vec2::new(-(end - start).y, (end - start).x).normalize();
        for step in 0..=160 {
            let point = start.lerp(end, step as f32 / 160.0);
            assert_eq!(snow.compression(point), 1.0, "gap at {point}");
            // Leave one bilinear texel footprint inside the rasterized
            // plateau when checking an exactly flush off-center point.
            assert_eq!(snow.compression(point + normal * 0.20), 1.0);
        }
        let center = start.lerp(end, 0.5);
        let edge = snow.compression(center + normal * 0.52);
        assert!(edge > 0.0 && edge < 1.0);
        assert_eq!(snow.compression(center + normal * 0.85), 0.0);
        let surface = SnowSurface {
            base_height: 130.0,
            coverage: 1.0,
        };
        assert_eq!(surface.height(&snow, center), 130.0);
        assert!((surface.height(&snow, center + normal) - 130.32).abs() < 0.0001);
    }

    #[test]
    fn stamps_are_monotonic_and_snapshots_stay_immutable() {
        let mut snow = SnowState::default();
        snow.stamp_segment(Vec2::ZERO, Vec2::new(0.8, 0.0));
        let before = snow.pixels();
        let revision = snow.revision();
        snow.stamp_segment(Vec2::ZERO, Vec2::new(0.8, 0.0));
        assert_eq!(snow.revision(), revision);
        snow.stamp_segment(Vec2::new(0.8, 0.0), Vec2::new(1.6, 0.2));
        assert!(snow.revision() > revision);
        assert!(
            before
                .iter()
                .zip(snow.pixels().iter())
                .all(|(old, new)| old <= new)
        );
        assert!(
            before
                .iter()
                .zip(snow.pixels().iter())
                .any(|(old, new)| old < new)
        );
        let untouched = Vec2::new(1.65, 0.20);
        assert_eq!(before[local_index(&snow, untouched)], 0);
        assert_eq!(snow.compression(untouched), 1.0);
    }

    fn local_index(snow: &SnowState, position: Vec2) -> usize {
        let mapping = snow.mapping();
        let cell = ((position - Vec2::new(mapping[0], mapping[1])) / SNOW_TEXEL_SIZE).floor();
        cell.y as usize * SNOW_MAP_SIZE + cell.x as usize
    }

    #[test]
    fn snow_cover_is_broad_but_releases_walls_and_low_ground() {
        assert_eq!(snow_coverage(90.0, Vec2::ZERO, 0.0), 0.0);
        assert_eq!(snow_coverage(140.0, Vec2::ZERO, 0.0), 1.0);
        assert_eq!(snow_coverage(140.0, Vec2::new(1.5, 0.0), 0.0), 0.0);
        let margin = snow_coverage(112.0, Vec2::ZERO, 0.0);
        assert!(margin > 0.0 && margin < 1.0);
        assert!(
            snow_coverage(112.0, Vec2::new(0.0, -0.4), 0.0)
                > snow_coverage(112.0, Vec2::new(0.0, 0.4), 0.0)
        );
    }

    /// Unit gradients at `count` evenly spaced azimuths.
    fn azimuths(count: usize) -> impl Iterator<Item = Vec2> {
        (0..count).map(move |i| {
            let angle = i as f32 * std::f32::consts::TAU / count as f32;
            Vec2::new(angle.cos(), angle.sin())
        })
    }

    #[test]
    fn no_snow_lies_below_the_floor_whatever_the_ground() {
        for height in [SNOW_FLOOR - 40.0, SNOW_FLOOR - 3.0, SNOW_FLOOR - 0.01, SNOW_FLOOR] {
            for direction in azimuths(24) {
                for steepness in [0.0, 0.2, 0.6, 1.0, 1.5, 3.0] {
                    for concavity in [-20.0, -6.0, -1.0, 0.0, 1.0, 6.0, 20.0] {
                        let coverage =
                            snow_coverage(height + SEA_LEVEL, direction * steepness, concavity);
                        assert_eq!(coverage, 0.0, "snow at {height} m, {steepness}, {concavity}");
                    }
                }
            }
        }
    }

    #[test]
    fn no_hole_opens_above_the_band_on_ground_the_pack_holds() {
        // From the top of the band plus the strongest sun the aspect gives,
        // gentle ground holds at least half a pack however convex it is.
        let top = SNOWLINE_ALTITUDE + SNOW_BAND + 0.38 * SNOW_MELT_ASPECT;
        for height in [top, top + 5.0, top + 30.0, top + 150.0] {
            for direction in azimuths(24) {
                for steepness in [0.0, 0.15, 0.3, 0.45] {
                    for concavity in [-20.0, -6.0, -2.0, 0.0, 2.0, 6.0, 20.0] {
                        let coverage =
                            snow_coverage(height + SEA_LEVEL, direction * steepness, concavity);
                        assert!(coverage >= 0.5, "{coverage} at {height} m, {steepness}, {concavity}");
                    }
                }
            }
        }
    }

    #[test]
    fn hollows_hold_the_margin_lower_and_noses_higher() {
        let at = |height: f32, concavity: f32| snow_coverage(height, Vec2::ZERO, concavity);
        assert!(at(112.0, 2.0) > at(112.0, 0.0) && at(112.0, 0.0) > at(112.0, -2.0));
        // Where the pack crosses half cover on level ground, by bisection.
        let contour = |concavity: f32| {
            let (mut low, mut high) = (80.0f32, 160.0f32);
            for _ in 0..40 {
                let middle = 0.5 * (low + high);
                if at(middle, concavity) < 0.5 { low = middle } else { high = middle }
            }
            0.5 * (low + high)
        };
        let level = contour(0.0);
        assert!((contour(2.0) - level + 1.86).abs() < 0.2, "{}", contour(2.0) - level);
        assert!((contour(-2.0) - level - 2.60).abs() < 0.2, "{}", contour(-2.0) - level);
    }

    #[test]
    fn lee_faces_keep_steep_snow_that_windward_faces_shed() {
        // A 48 degree face well inside the pack, falling with the wind (lee),
        // against it (windward) and across it.
        let tan = 48.0f32.to_radians().tan();
        let across = Vec2::new(-SNOW_WIND_TO.y, SNOW_WIND_TO.x);
        let lee = snow_coverage(160.0, -SNOW_WIND_TO * tan, 0.0);
        let windward = snow_coverage(160.0, SNOW_WIND_TO * tan, 0.0);
        let cross = snow_coverage(160.0, across * tan, 0.0);
        assert!(lee >= 0.8, "{lee}");
        assert!(windward <= 0.25, "{windward}");
        assert!(windward < cross && cross < lee, "{windward} {cross} {lee}");
        // A full pack on moderate ground does not care where the wind blows.
        for direction in azimuths(16) {
            assert_eq!(snow_coverage(160.0, direction * 0.5, 0.0), 1.0);
        }
    }

    #[test]
    fn snow_cover_is_continuous() {
        // A step in the field would be an edge or a hole in the pack; bound
        // its finite differences in height, slope and relief.
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut uniform = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / 16_777_216.0
        };
        for _ in 0..20_000 {
            let height = 80.0 + 100.0 * uniform();
            let angle = std::f32::consts::TAU * uniform();
            let steepness = 2.0 * uniform();
            let gradient = Vec2::new(angle.cos(), angle.sin()) * steepness;
            let concavity = -20.0 + 40.0 * uniform();
            let base = snow_coverage(height, gradient, concavity);
            let step = 0.01;
            let dh = (snow_coverage(height + step, gradient, concavity) - base).abs() / step;
            let dg = (snow_coverage(height, gradient * (1.0 + step / steepness.max(0.01)), concavity)
                - base)
                .abs()
                / step;
            let dc = (snow_coverage(height, gradient, concavity + step) - base).abs() / step;
            assert!(dh < 0.1, "{dh} per metre at {height} {gradient} {concavity}");
            assert!(dg < 2.5, "{dg} per unit of tan at {height} {gradient} {concavity}");
            assert!(dc < 0.2, "{dc} per metre of relief at {height} {gradient} {concavity}");
        }
    }

    #[test]
    fn snow_constants_match_the_shader() {
        let block = include_str!("../assets/shaders/snow-functions.wgslinc");
        let value = |name: &str| -> f32 {
            let line = block
                .lines()
                .find(|line| line.starts_with(&format!("const {name}: f32 = ")))
                .unwrap_or_else(|| panic!("{name} is not in the shared snow block"));
            let expression = line.split('=').nth(1).unwrap().split(';').next().unwrap().trim();
            match expression.split_once('/') {
                Some((numerator, denominator)) => {
                    numerator.trim().parse::<f32>().unwrap() / denominator.trim().parse::<f32>().unwrap()
                }
                None => expression.parse().unwrap(),
            }
        };
        assert_eq!(value("SNOW_BAND"), SNOW_BAND);
        assert_eq!(value("SNOW_MELT_ASPECT"), SNOW_MELT_ASPECT);
        assert_eq!(value("SNOW_BURIAL"), SNOW_BURIAL);
        assert_eq!(value("SNOW_HOLD"), SNOW_HOLD);
        assert_eq!(value("SNOW_SLIP_TAN"), SNOW_SLIP_TAN);
        assert_eq!(value("SNOW_WIND_TILT"), SNOW_WIND_TILT);
        assert_eq!(value("SNOW_SMOOTH"), SNOW_SMOOTH);
        assert!(block.contains(&format!(
            "const SNOW_WIND_TO: vec2<f32> = vec2<f32>({:?}, {:?});",
            SNOW_WIND_TO.x, SNOW_WIND_TO.y
        )));
        assert!(block.contains(
            "return stage.snowline_altitude - SNOW_BAND - 0.62*SNOW_MELT_ASPECT;"
        ));
    }

    /// The terrain's snowbeds (terrain-fs.wgsl) are the pack NIVAL_REACH
    /// metres further down the thaw: they hug the pack's margin, reach about
    /// that far below it on level ground, and are nothing where no pack lay.
    #[test]
    fn snowbeds_follow_the_pack_and_never_reach_below_it() {
        const NIVAL_REACH: f32 = 8.0;
        let shader = include_str!("../assets/shaders/terrain-fs.wgsl");
        assert!(shader.contains(&format!("const NIVAL_REACH: f32 = {NIVAL_REACH:?};")));
        assert!(shader.contains("if (height - stage.sea_level + NIVAL_REACH >= snowFloor())"));
        assert!(shader.contains(
            "snowbed = smoothHermite(0.2, 0.7, snowCoverage(height + NIVAL_REACH, materialNormal,"
        ));
        let earlier = |height: f32, gradient: Vec2, concavity: f32| {
            smooth_hermite(0.2, 0.7, snow_coverage(height + NIVAL_REACH, gradient, concavity))
        };
        for direction in azimuths(12) {
            for steepness in [0.0, 0.3, 0.7, 1.2] {
                for concavity in [-6.0, 0.0, 6.0] {
                    let gradient = direction * steepness;
                    // Below the guard there is no earlier pack at all.
                    assert_eq!(earlier(SNOW_FLOOR - NIVAL_REACH - 0.01, gradient, concavity), 0.0);
                    for height in [95.0, 105.0, 115.0, 125.0] {
                        let now = smooth_hermite(0.2, 0.7, snow_coverage(height, gradient, concavity));
                        assert!(earlier(height, gradient, concavity) >= now);
                    }
                }
            }
        }
        // On level ground, the strip the earlier pack half covered and the
        // pack now does not spans the reach in height.
        let (mut first, mut last) = (f32::MAX, f32::MIN);
        let mut height = 80.0;
        while height < 140.0 {
            let before = snow_coverage(height + NIVAL_REACH, Vec2::ZERO, 0.0);
            if before >= 0.5 && snow_coverage(height, Vec2::ZERO, 0.0) < 0.5 {
                first = first.min(height);
                last = last.max(height);
            }
            height += 0.01;
        }
        assert!((last - first - NIVAL_REACH).abs() < 0.05, "{first}..{last}");
    }

    /// The shader's snowCoverage, run on a GPU (or a software adapter), must
    /// give the player's footing the pack it draws.
    #[test]
    fn the_shader_draws_the_pack_the_player_stands_on() {
        use bevy::tasks::block_on;
        use wgpu::util::DeviceExt;
        let instance = wgpu::Instance::default();
        let Ok(adapter) = block_on(instance.request_adapter(&Default::default())) else {
            eprintln!("skipping the snow shader check: no GPU adapter");
            return;
        };
        let (device, queue) = block_on(adapter.request_device(&Default::default())).expect("a device");
        let block = include_str!("../assets/shaders/snow-functions.wgslinc");
        let source = format!(
            "struct Stage {{ sea_level: f32, snowline_altitude: f32 }};
             var<private> stage: Stage = Stage({:?}, {:?});
             fn smoothHermite(edge0: f32, edge1: f32, value: f32) -> f32 {{
                 let t = clamp((value - edge0) / max(edge1 - edge0, 0.0001), 0.0, 1.0);
                 return t * t * (3.0 - 2.0 * t);
             }}
             {block}
             @group(0) @binding(0) var<storage, read_write> samples: array<vec4<f32>>;
             @compute @workgroup_size(64)
             fn evaluate(@builtin(global_invocation_id) id: vec3<u32>) {{
                 if (id.x >= arrayLength(&samples)) {{ return; }}
                 let input = samples[id.x];
                 let normal = normalize(vec3<f32>(-input.y, 1.0, -input.z));
                 samples[id.x] = vec4<f32>(snowCoverage(input.x, normal, input.w), 0.0, 0.0, 0.0);
             }}",
            SEA_LEVEL, SNOWLINE_ALTITUDE
        );
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("snow coverage check"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("snow coverage check"),
            layout: None,
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: Default::default(),
            cache: None,
        });
        let mut inputs = Vec::new();
        for i in 0..2048u32 {
            let t = |salt: u32| (crate::vegetation::ecology::mix64((i as u64) << 8 | salt as u64) >> 40) as f32 / 16_777_216.0;
            let angle = std::f32::consts::TAU * t(1);
            let steepness = 1.8 * t(2);
            inputs.push([80.0 + 90.0 * t(0), angle.cos() * steepness, angle.sin() * steepness, -12.0 + 24.0 * t(3)]);
        }
        let samples = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("snow coverage samples"),
            contents: bytemuck::cast_slice(&inputs),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("snow coverage readback"),
            size: samples.size(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("snow coverage check"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: samples.as_entire_binding() }],
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups((inputs.len() as u32).div_ceil(64), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&samples, 0, &readback, 0, samples.size());
        queue.submit([encoder.finish()]);
        let (sender, receiver) = std::sync::mpsc::channel();
        readback.map_async(wgpu::MapMode::Read, .., move |result| sender.send(result).unwrap());
        device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        receiver.recv().expect("mapping callback").expect("map readback");
        let bytes = readback.get_mapped_range(..).to_vec();
        let drawn: Vec<[f32; 4]> = bytes.chunks_exact(16).map(bytemuck::pod_read_unaligned).collect();
        for (input, drawn) in inputs.iter().zip(drawn) {
            let footing = snow_coverage(input[0], Vec2::new(input[1], input[2]), input[3]);
            assert!(
                (drawn[0] - footing).abs() < 2.0e-5,
                "shader {} against CPU {footing} at {input:?}",
                drawn[0]
            );
        }
    }
}
