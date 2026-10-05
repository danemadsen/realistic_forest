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

// Thermal (talus) relaxation. Loose cover - soil, talus, alluvium - stands
// at its angle of repose (~34 degrees); bare bedrock holds much steeper faces,
// from ~48 degrees in weak rock to ~68 degrees in the most resistant beds,
// and sheds rockfall only above them.
pub const EROSION_LOOSE_REPOSE: f32 = 0.67;
pub const EROSION_SOFT_ROCK_SLOPE: f32 = 1.1;
pub const EROSION_HARD_ROCK_SLOPE: f32 = 2.5;
// Fluvial stream power saturates for catchments larger than this many cells
// (~6.4 ha). A tile routes only its own domain, so overlapping tiles truncate
// big rivers differently; beyond the saturation they cut alike.
pub const EROSION_DRAINAGE_SATURATION: f32 = 4000.0;

// PBR terrain texture array: one set each for snow, grass, sand and rock,
// plus two sets each for soil within the grass cover and for gravel.
// Retain the source scans' small stones and grass detail near the camera.
// Two eight-layer RGBA mip arrays use about 85 MiB at this resolution.
pub const TERRAIN_TILE_SIZE: usize = 1024;
pub const TERRAIN_ATLAS_SLOTS: usize = 8;
pub const EROSION_REVEAL_SECONDS: f32 = 0.9;
// The reveal ramps at 1/54 per streamed frame; only the TERRAIN-REVISION hash
// quantises it — to this many steps across the ramp — so the expensive lighting
// heightfield, habitat and shore retakes run 8 times per reveal instead of once
// per frame while the rendered blend in the terrain shaders stays continuous.
pub const EROSION_REVEAL_HASH_QUANTUM: f32 = 8.0;

// The erosion simulation runs on its own wgpu device behind a dedicated
// worker thread (see `erosion_worker`), so the render thread stops recording
// 568-pass prewarm frames head of line. One worker submission still records
// at most this many flux/water/terrain/thermal iterations — Metal command
// buffers stay bounded and, being submitted in order, the chunks execute
// exactly as one long one would.
pub const EROSION_WORKER_MAX_ITERATIONS_PER_SUBMIT: usize = 24;
// Frames `apply_erosion_events` will wait after a tile began, without any
// finalize or failure event, before it requeues the tile like a failed
// readback. A live worker finishes well inside half of this (six iterations
// per frame is ~24 frames plus readback latency), so only a stalled or dying
// worker trips it. A false trip is harmless to results: the re-queued tile
// re-begins and re-stamps its init, converging to the same simulation.
pub const EROSION_WORKER_WATCHDOG_FRAMES: u32 = 240;
// Finished atlas patches cross from the worker thread through
// `ErosionBridge::push_patches`, then the render world uploads once their
// tile's slot appears in the lookup records. If the render side stops
// draining, the oldest patch is dropped with a warning at this many, so the
// bridge cannot grow unbounded.
pub const EROSION_PENDING_PATCH_CAP: usize = 32;
pub const SEA_LEVEL: f32 = 0.0;
// Centre of the persistent (late-season) snowline, metres above sea level.
// A smooth altitude band and climate aspect place a uniform pack;
// falling precipitation turns to snow from ~80 m upward
// (precipitation-functions.wgslinc), so fresh snow can fall a little below
// the ground that keeps it.
pub const SNOWLINE_ALTITUDE: f32 = 112.0;

pub const CLIP_CELLS: usize = 224;
pub const CLIP_LEVELS: usize = 9;
pub const CLIP_FINEST_SPACING: f32 = 0.25;
pub const CLIP_ANCHOR_SPACING: f32 = CLIP_FINEST_SPACING * (1u32 << (CLIP_LEVELS - 1)) as f32;
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
    /// Stream-power transport capacity per square-rooted catchment cell.
    pub fluvial_capacity: f32,
    /// Fraction of a fluvial capacity deficit detached from the bed per step.
    pub fluvial_erosion: f32,
    /// Fraction of a fluvial capacity excess deposited per step.
    pub fluvial_deposition: f32,
    /// Deepest fluvial incision below the base surface, metres.
    pub maximum_incision: f32,
    /// Fraction of a loose slope's excess over its angle of repose that
    /// slides per step (pairwise, at most 1/16).
    pub talus_rate: f32,
    /// The same for over-steepened bedrock faces (rockfall).
    pub rockfall_rate: f32,
}

impl Default for ErosionSettings {
    fn default() -> Self {
        Self {
            iterations: 140,
            rain: 0.018,
            evaporation: 0.12,
            erosion_rate: 0.18,
            deposition_rate: 0.40,
            sediment_capacity: 5.0,
            transport_rate: 1.0,
            maximum_erosion: 5.5,
            fluvial_capacity: 0.050,
            fluvial_erosion: 0.18,
            fluvial_deposition: 0.30,
            maximum_incision: 12.0,
            talus_rate: 0.05,
            rockfall_rate: 0.0075,
        }
    }
}

#[derive(Clone, Copy, Debug, Resource)]
pub struct AppSettings {
    pub ssao_enabled: bool,
    pub flow_debug: bool,
    pub erosion_debug: bool,
    /// Expand advanced controls inside the developer menu.
    pub show_ui: bool,
    pub show_trainer: bool,
    pub show_debug: bool,
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
    /// Loudness of thunder, 0 silent to 1 full.
    pub thunder_volume: f32,
    /// Draw the scattered trees, shrubs and flowers.
    pub vegetation_enabled: bool,
    /// Multiplies every plant LOD switch distance (1 = default detail).
    pub vegetation_detail: f32,
    /// Plants cast sun (and moon) shadows through cascaded shadow maps.
    pub vegetation_shadows: bool,
    /// Draw the rivers' and lakes' water surfaces.
    pub rivers_visible: bool,
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            ssao_enabled: true,
            flow_debug: false,
            erosion_debug: false,
            show_ui: false,
            show_trainer: false,
            show_debug: false,
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
            thunder_volume: 0.8,
            vegetation_enabled: true,
            vegetation_detail: 1.0,
            vegetation_shadows: true,
            rivers_visible: true,
        }
    }
}
