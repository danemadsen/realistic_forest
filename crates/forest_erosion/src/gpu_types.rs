//! GPU uniform layouts shared by erosion workers and forest rendering.

/// The shared `GlobalUniforms` preamble (group 0, binding 0), 416 bytes.
#[repr(C, align(16))]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GlobalUniformsGpu {
    pub view: [f32; 16],
    pub projection: [f32; 16],
    pub camera_position: [f32; 4],

    /// xyz the direction sunlight travels. w is the highest terrain in the
    /// lighting heightfield: the terrain pass reduces it on the GPU and copies
    /// it over [`TERRAIN_HEIGHT_UNKNOWN`] each frame it draws the map.
    pub sun_direction: [f32; 4],

    pub viewport: [f32; 4],
    pub params: [f32; 4],
    pub settings_a: [f32; 4],
    pub settings_b: [f32; 4],
    pub sun_colour: [f32; 4],
    pub moon_direction: [f32; 4],
    pub atmosphere: [f32; 4],
    pub raymarch: [f32; 4],
    pub heightfield: [f32; 4],
    pub clouds: [f32; 4],
    pub cloud_layer: [f32; 4],
    pub cloud_motion: [f32; 4],
    pub weather: [f32; 4],
    pub storm: [f32; 4],
    pub lightning: [f32; 4],
    pub lightning_meta: [f32; 4],
}
const _: () = assert!(std::mem::size_of::<GlobalUniformsGpu>() == 416);

/// `sun_direction.w` until the terrain pass has measured the heightfield: no
/// march ever rises above it, so the shaders' early exits stay off.
pub const TERRAIN_HEIGHT_UNKNOWN: f32 = 1.0e9;
/// Byte offset of `sun_direction.w`, where the terrain pass copies the
/// measured highest terrain.
pub const TERRAIN_HEIGHT_OFFSET: u64 =
    std::mem::offset_of!(GlobalUniformsGpu, sun_direction) as u64 + 12;

/// erosion-init stage uniforms.
#[repr(C, align(8))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ErosionInitStageUniforms {
    pub mode: i32,                 // 0  uMode
    pub(crate) _pad0: f32,         // 4
    pub world_min: [f32; 2],       // 8  uWorldMin
    pub cell_size: f32,            // 16 uCellSize
    pub(crate) _end_pad: [f32; 3], // 20
}
const _: () = assert!(std::mem::size_of::<ErosionInitStageUniforms>() == 32);

/// erosion-flux stage uniforms.
#[repr(C, align(8))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ErosionFluxStageUniforms {
    pub resolution: [f32; 2],      // 0  uResolution
    pub delta_time: f32,           // 8  uDeltaTime
    pub cell_size: f32,            // 12 uCellSize
    pub gravity: f32,              // 16 uGravity
    pub(crate) _end_pad: [f32; 3], // 20
}
const _: () = assert!(std::mem::size_of::<ErosionFluxStageUniforms>() == 32);

/// erosion-water stage uniforms.
#[repr(C, align(8))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ErosionWaterStageUniforms {
    pub resolution: [f32; 2], // 0
    pub delta_time: f32,      // 8
    pub cell_size: f32,       // 12
    pub rain: f32,            // 16
    pub evaporation: f32,     // 20
    pub sea_level: f32,       // 24
    pub(crate) _end_pad: f32, // 28 (uniform size rounds to 32)
}
const _: () = assert!(std::mem::size_of::<ErosionWaterStageUniforms>() == 32);

/// erosion-terrain stage uniforms.
#[repr(C, align(8))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ErosionTerrainStageUniforms {
    pub resolution: [f32; 2],     // 0
    pub delta_time: f32,          // 8
    pub cell_size: f32,           // 12
    pub sea_level: f32,           // 16
    pub erosion_rate: f32,        // 20
    pub deposition_rate: f32,     // 24
    pub sediment_capacity: f32,   // 28
    pub minimum_slope: f32,       // 32
    pub maximum_erosion: f32,     // 36
    pub transport_rate: f32,      // 40
    pub guard_band_pixels: f32,   // 44
    pub brush_strength: f32,      // 48
    pub(crate) _align: f32,       // 52
    pub world_min: [f32; 2],      // 56 uWorldMin
    pub fluvial_capacity: f32,    // 64
    pub fluvial_erosion: f32,     // 68
    pub fluvial_deposition: f32,  // 72
    pub maximum_incision: f32,    // 76
    pub drainage_saturation: f32, // 80
    pub(crate) _end_pad: f32,     // 84 (struct rounds to 88)
}
const _: () = assert!(std::mem::size_of::<ErosionTerrainStageUniforms>() == 88);

/// erosion-thermal stage uniforms.
#[repr(C, align(8))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ErosionThermalStageUniforms {
    pub resolution: [f32; 2],        // 0
    pub cell_size: f32,              // 8
    pub sea_level: f32,              // 12
    pub world_min: [f32; 2],         // 16
    pub loose_rate: f32,             // 24
    pub rock_rate: f32,              // 28
    pub loose_repose: f32,           // 32
    pub soft_rock_slope: f32,        // 36
    pub hard_rock_slope: f32,        // 40
    pub weathering_rate: f32,        // 44
    pub weathering_cover: f32,       // 48
    pub weathering_cover_steep: f32, // 52
    pub weathering_slope: f32,       // 56
    pub(crate) _end_pad: f32,        // 60 (uniform size rounds to 64)
}
const _: () = assert!(std::mem::size_of::<ErosionThermalStageUniforms>() == 64);

/// Width and height of the square erosion simulation textures.
pub const SIM_TEXTURE_SIZE: u32 = forest_core::constants::EROSION_RESOLUTION as u32;
