//! All engine constants, ported 1:1 from the C++ project's anonymous namespace.

use bevy::prelude::Resource;

pub const NOISE_RESOLUTION: usize = 1024;
pub const NOISE_RESOLUTION_F: f32 = 1024.0;
pub const NOISE_PERIOD: f32 = 8192.0;

// Independent authoring controls for broad feature width and natural relief.
// Fine terrain and hydraulic erosion remain in physical metres.
pub const LANDFORM_HORIZONTAL_SCALE: f32 = 2.5;
pub const LANDFORM_VERTICAL_SCALE: f32 = 1.5;

// Exponential altitude profile applied to the macro landform around sea
// level (shape_elevation, mirrored by shapeElevation in terrain.wgsl).
pub const LAND_PROFILE_CURVE: f32 = 2.6;
pub const LAND_PROFILE_REFERENCE: f32 = 150.0;
pub const LAND_PROFILE_PEAK: f32 = 280.0;

// Waterline clearance: a decaying stretch of the land signal just above sea
// level. It lifts the near-sea plains out of the water plane's z-fighting
// band and relaxes to zero before the foothills. It acts on the *macro*
// landform, before the fine detail octaves are added.
pub const WATERLINE_CLEARANCE: f32 = 5.0;
pub const WATERLINE_CLEARANCE_SCALE: f32 = 7.0;
pub const WATERLINE_CLEARANCE_DECAY: f32 = 15.0;

// Final waterline push: metres of separation forced between the finished
// surface and sea level. waterline_clearance above cannot hold the coastline
// on its own, because fine_height is added *after* the profile and swings the
// result by roughly +-2 m at 8.6 m and 31 m wavelengths. That swing is what
// turns the coastal plain into a fractal speckle of puddles and islets: the
// terrain crosses the water surface thousands of times per kilometre, and
// every crossing is a depth-buffer tie.
//
// The push adds `WATERLINE_PUSH_LAND * tanh(h / WATERLINE_PUSH_SCALE)` above
// the waterline and `WATERLINE_PUSH_SEA * tanh(h / WATERLINE_PUSH_SCALE)`
// below it. It is sign-preserving and exactly zero at sea level, so the
// coastline's position does not move; it only steepens the approach to it,
// multiplying the local gradient by 1 + PUSH/SCALE. Land that would have sat
// at 0.1 m — inside the band where wave troughs and the depth buffer both
// disagree about who is in front — lands at roughly 0.4 m instead, and land
// at 0.5 m lands near 2 m, clear of any wave crest.
//
// The sea side is deliberately much weaker than the land side: pushing the bed
// down as hard would erase the shallow shelf that the water's shoaling, foam
// and transmission all read, so the bed keeps its shelf and only firms up.
pub const WATERLINE_PUSH_LAND: f32 = 6.0;
pub const WATERLINE_PUSH_SEA: f32 = 2.0;
pub const WATERLINE_PUSH_SCALE: f32 = 2.0;

// The ocean reference is met by the deepest continental signal, and the
// curve holds the depth constant beyond it — a flat abyssal plain.
pub const OCEAN_PROFILE_CURVE: f32 = 3.2;
pub const OCEAN_PROFILE_REFERENCE: f32 = -28.0;
pub const OCEAN_PROFILE_DEPTH: f32 = -300.0;

pub const EROSION_CELL_SIZE: f32 = 4.0;
// Each independent pass retains a 1024 m square, while pass centres are only
// 512 m apart. This exact 50% lattice gives every world point the same four
// candidate simulations (two along each axis).
pub const EROSION_FOOTPRINT_CELLS: usize = 256;
pub const EROSION_STRIDE_CELLS: usize = EROSION_FOOTPRINT_CELLS / 2;
pub const EROSION_FOOTPRINT_SIZE: f32 = EROSION_FOOTPRINT_CELLS as f32 * EROSION_CELL_SIZE;
pub const EROSION_TILE_STRIDE: f32 = EROSION_STRIDE_CELLS as f32 * EROSION_CELL_SIZE;
// Independent passes simulate well beyond their retained footprint. The
// discarded region keeps closed-wall hydraulic boundary effects out of the
// four-way overlap.
pub const EROSION_HALO_CELLS: usize = 112;
pub const EROSION_RESOLUTION: usize = EROSION_FOOTPRINT_CELLS + EROSION_HALO_CELLS * 2;
pub const EROSION_SIMULATION_HALO: f32 = EROSION_HALO_CELLS as f32 * EROSION_CELL_SIZE;
pub const EROSION_OUTPUT_RESOLUTION: usize = EROSION_FOOTPRINT_CELLS;
pub const EROSION_OUTPUT_OFFSET: usize = EROSION_HALO_CELLS;
pub const EROSION_ATLAS_GUTTER: usize = 2;
pub const EROSION_ATLAS_PITCH: usize = EROSION_OUTPUT_RESOLUTION + EROSION_ATLAS_GUTTER * 2;
pub const EROSION_STREAMING_RADIUS: i64 = 4;
pub const EROSION_LOOKUP_RADIUS: i64 = EROSION_STREAMING_RADIUS + 1;
pub const EROSION_LOOKUP_DIAMETER: usize = EROSION_LOOKUP_RADIUS as usize * 2 + 1;
// With a 9x9 cache centred on the nearest lattice point, all four required
// passes exist for at least 3.5 strides (1792 m) in every direction.
pub const EROSION_VISIBILITY_FULL_RADIUS: f32 = 1100.0;
pub const EROSION_VISIBILITY_ZERO_RADIUS: f32 = 1600.0;
pub const EROSION_ATLAS_COLUMNS: usize = 10;
pub const EROSION_ATLAS_SLOTS: usize = EROSION_ATLAS_COLUMNS * EROSION_ATLAS_COLUMNS;
pub const EROSION_ATLAS_SIZE: usize = EROSION_ATLAS_COLUMNS * EROSION_ATLAS_PITCH;
pub const EROSION_ITERATIONS_PER_FRAME: usize = 6;

// PBR terrain texture array: one set each for snow, grass, sand and rock,
// plus two sets each for soil within the grass cover and for gravel.
pub const TERRAIN_TILE_SIZE: usize = 512;
pub const TERRAIN_ATLAS_SLOTS: usize = 8;
pub const EROSION_REVEAL_SECONDS: f32 = 0.9;
pub const SEA_LEVEL: f32 = 0.0;

pub const CLIP_CELLS: usize = 224;
pub const CLIP_LEVELS: usize = 7;
pub const CLIP_ANCHOR_SPACING: f32 = (1u32 << (CLIP_LEVELS - 1)) as f32;
pub const EYE_HEIGHT: f32 = 1.75;
pub const NEAR_PLANE: f32 = 0.1;
pub const FAR_PLANE: f32 = 5800.0;

/// The eight fixed biome material sets, in the shader's blend order.
pub const TERRAIN_MATERIALS: [&str; TERRAIN_ATLAS_SLOTS] = [
    "Snow006_2K-PNG",
    "Grass004_2K-PNG",
    "Ground093C_2K-PNG",
    "Ground103_2K-PNG",
    "Ground106_2K-PNG",
    "Gravel040_2K-PNG",
    "Gravel043_2K-PNG",
    "Rock032_2K-PNG",
];

/// Default user-facing simulation + render settings (mirrors AppSettings
/// and ErosionSettings in the C++).

#[derive(Clone, Copy, Debug, Resource)]
pub struct ErosionSettings {
    pub iterations: usize,
    pub rain: f32,
    pub evaporation: f32,
    pub erosion_rate: f32,
    pub deposition_rate: f32,
    pub sediment_capacity: f32,
    pub transport_rate: f32,
    pub maximum_erosion: f32,
}

impl Default for ErosionSettings {
    fn default() -> Self {
        Self {
            iterations: 140,
            rain: 0.01,
            evaporation: 0.12,
            erosion_rate: 0.08,
            deposition_rate: 0.40,
            sediment_capacity: 3.5,
            transport_rate: 0.80,
            maximum_erosion: 3.5,
        }
    }
}

#[derive(Clone, Copy, Debug, Resource)]
pub struct AppSettings {
    pub ssao_enabled: bool,
    pub flow_debug: bool,
    pub erosion_debug: bool,
    pub show_ui: bool,
    pub ao_radius: f32,
    pub ao_bias: f32,
    pub ao_power: f32,
    pub ao_strength: f32,
    pub fog_density: f32,
    pub sun_intensity: f32,
    pub raymarched_shadows: bool,
    pub volumetric_lighting: bool,
    pub water_reflections: bool,
    /// 0 = low, 1 = balanced, 2 = high.
    pub raymarch_quality: u32,
    pub volumetric_strength: f32,
    pub clouds_enabled: bool,
    pub cloud_coverage: f32,
    pub cloud_density: f32,
    /// Bottom and vertical extent of the cloud layer, in world metres.
    pub cloud_base_height: f32,
    pub cloud_thickness: f32,
    pub cloud_scale: f32,
    pub cloud_shadow_strength: f32,
    /// Horizontal advection in metres per real second; zero degrees is +X.
    pub cloud_wind_speed: f32,
    pub cloud_wind_direction_degrees: f32,
    pub cloud_detail_strength: f32,
    pub texture_scale: f32,
    pub normal_strength: f32,
    pub ao_tex_strength: f32,
    pub variant_scale: f32,
    pub sparkle_strength: f32,
    pub exposure: f32,
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            ssao_enabled: true,
            flow_debug: false,
            erosion_debug: false,
            show_ui: false,
            ao_radius: 2.1,
            ao_bias: 0.08,
            ao_power: 1.25,
            ao_strength: 0.72,
            fog_density: 0.00012,
            // Sun-facing diffuse equals the albedo; ~1.5x pi pushes sunlit
            // snow onto the tonemap's shoulder.
            sun_intensity: 4.8,
            raymarched_shadows: true,
            volumetric_lighting: true,
            water_reflections: true,
            raymarch_quality: 1,
            volumetric_strength: 1.0,
            clouds_enabled: true,
            cloud_coverage: 0.55,
            cloud_density: 1.0,
            cloud_base_height: 1300.0,
            cloud_thickness: 1200.0,
            cloud_scale: 1500.0,
            cloud_shadow_strength: 0.65,
            cloud_wind_speed: 18.0,
            cloud_wind_direction_degrees: 70.0,
            cloud_detail_strength: 0.35,
            texture_scale: 0.16,
            normal_strength: 1.0,
            ao_tex_strength: 0.8,
            // Soil/gravel scan variants only.
            variant_scale: 0.013,
            // Restrained snow glints (0 disables).
            sparkle_strength: 0.7,
            // Linear scale before the ACES tonemap; the shoulder, not this,
            // protects highlights.
            exposure: 1.0,
        }
    }
}
