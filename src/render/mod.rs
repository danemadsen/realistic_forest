//! The forest render schedule: erosion simulation -> terrain G-buffer -> SSAO
//! -> blur -> composite -> FXAA -> egui -> upscale. Ports the C++ main loop
//! render order 1:1 on wgpu.
//!
//! bevy 0.19 replaced render graphs with per-camera schedules, so the order
//! above is an ordered [`ForestRender`] schedule rather than a graph of nodes;
//! `spawn_scene` points the camera at it with
//! `CameraRenderGraph::new(ForestRender)`.

pub mod erosion_node;
pub mod cloud_node;
pub mod gpu_textures;
pub mod grass_node;
pub mod post_nodes;
pub mod precipitation_node;
pub mod terrain_node;
pub mod water_node;

use crate::constants::*;
use crate::day_night::DayNightCycle;
use crate::erosion::{ErosionBridge, ErosionCache};
use crate::noise::NoiseField;
use crate::player::{Player, PlayerCamera};
use crate::WorldOptions;
use bevy::asset::Handle;
use bevy::ecs::schedule::ScheduleLabel;
use bevy::prelude::*;
use bevy::render::render_resource::{
    BindGroup, BindGroupEntry, BindGroupLayoutDescriptor, PipelineCache,
};
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::render::{ExtractSchedule, MainWorld, RenderApp};
use bevy::shader::Shader;
use bevy::window::PrimaryWindow;

/// The camera render schedule; cameras point at it via
/// `CameraRenderGraph::new(ForestRender)`.
///
/// bevy 0.19 renders each camera by running a *schedule* of systems rather
/// than a render graph of nodes, so the pass order the port used to encode
/// with node edges now lives in [`ForestRenderSystems`].
#[derive(ScheduleLabel, Debug, Hash, PartialEq, Eq, Clone)]
pub struct ForestRender;

/// The ordered stages of the forest pipeline. Mirrors the render order the
/// port's render graph used to encode with node edges: erosion -> terrain ->
/// SSAO -> blur -> clouds -> composite -> water -> FXAA -> egui -> upscale.
#[derive(SystemSet, Debug, Hash, PartialEq, Eq, Clone)]
pub enum ForestRenderSystems {
    Erosion,
    Terrain,
    Grass,
    Ssao,
    Blur,
    Clouds,
    Composite,
    WaterSurface,
    WaterUnderwater,
    Precipitation,
    Fxaa,
    Egui,
    Upscale,
}

impl ForestRender {
    /// An empty schedule with the pass stages chained in order. The passes
    /// themselves are added as systems by [`ForestRenderPlugin`].
    pub fn base_schedule() -> bevy::ecs::schedule::Schedule {
        use ForestRenderSystems::*;
        let mut schedule = bevy::ecs::schedule::Schedule::new(Self);
        schedule.configure_sets(
            (
                Erosion,
                Terrain,
                Grass,
                Ssao,
                Blur,
                Clouds,
                Composite,
                WaterSurface,
                WaterUnderwater,
                Precipitation,
                Fxaa,
                Egui,
                Upscale,
            )
                .chain(),
        );
        schedule
    }
}

/// Camera-attached data extracted from the main world each frame.
#[derive(Resource, Default)]
pub struct ExtractedForestView {
    pub player_position: [f32; 3],
    pub player_yaw: f32,
    pub player_pitch: f32,
    pub physical_width: u32,
    pub physical_height: u32,
    pub settings: AppSettings,
    pub day_night: DayNightCycle,
    pub weather_offset: [f32; 2],
    pub weather: crate::weather::WeatherState,
    pub draw_ocean: bool,
    pub lookup_minimum: (i64, i64),
    pub frame: u64,
}

// ---------------------------------------------------------------------------
// Uniform structures — byte layouts matching the WGSL stage uniforms. Every
// struct ends with a `const _` assert pinning its size so WGSL edits cannot
// silently drift.
// ---------------------------------------------------------------------------

/// The shared `GlobalUniforms` preamble (group 0, binding 0), 416 bytes.
#[repr(C, align(16))]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GlobalUniformsGpu {
    pub view: [f32; 16],
    pub projection: [f32; 16],
    pub camera_position: [f32; 4],
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

/// terrain-vs + terrain-fs share one canonical StageUniforms layout
/// (272 bytes); the WGSL files are reconciled to this exact field order.
#[repr(C, align(16))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TerrainStageUniforms {
    pub mat_model: [f32; 16],                 // 0   matModel
    pub inverse_model: [f32; 16],             // 64  inverse(matModel), CPU-supplied
    pub clip_origin: [f32; 2],                // 128 uClipOrigin
    pub spacing: f32,                         // 136 uSpacing
    pub next_spacing: f32,                    // 140 uNextSpacing
    pub morph_start: f32,                     // 144 uMorphStart
    pub morph_end: f32,                       // 148 uMorphEnd
    pub sea_level: f32,                       // 152 uSeaLevel
    pub noise_period: f32,                    // 156 uNoisePeriod
    pub landform_horizontal_scale: f32,       // 160 uLandformHorizontalScale
    pub landform_vertical_scale: f32,         // 164 uLandformVerticalScale
    pub land_profile_curve: f32,              // 168 uLandProfileCurve
    pub land_profile_reference: f32,          // 172 uLandProfileReference
    pub land_profile_peak: f32,               // 176 uLandProfilePeak
    pub waterline_clearance: f32,             // 180 uWaterlineClearance
    pub waterline_clearance_scale: f32,       // 184 uWaterlineClearanceScale
    pub waterline_clearance_decay: f32,       // 188 uWaterlineClearanceDecay
    pub ocean_profile_curve: f32,             // 192 uOceanProfileCurve
    pub ocean_profile_reference: f32,         // 196 uOceanProfileReference
    pub ocean_profile_depth: f32,             // 200 uOceanProfileDepth
    pub erosion_tile_stride: f32,             // 204 uErosionTileStride
    pub erosion_footprint_size: f32,          // 208 uErosionFootprintSize
    pub erosion_output_resolution: f32,       // 212 uErosionOutputResolution
    pub erosion_atlas_pitch: f32,             // 216 uErosionAtlasPitch
    pub erosion_atlas_size: f32,              // 220 uErosionAtlasSize
    pub erosion_atlas_gutter: f32,            // 224 uErosionAtlasGutter
    _pad0: f32,                               // 228 (vec2 alignment)
    pub erosion_lookup_min_tile: [f32; 2],    // 232 uErosionLookupMinTile
    pub erosion_lookup_size: f32,             // 240 uErosionLookupSize
    _pad1: f32,                               // 244 (vec2 alignment)
    pub erosion_visibility_center: [f32; 2],  // 248 uErosionVisibilityCenter
    pub erosion_visibility_full_radius: f32,  // 256 uErosionVisibilityFullRadius
    pub erosion_visibility_zero_radius: f32,  // 260 uErosionVisibilityZeroRadius
    pub waterline_push_land: f32,             // 264 uWaterlinePushLand
    pub waterline_push_sea: f32,              // 268 uWaterlinePushSea
    pub waterline_push_scale: f32,            // 272 uWaterlinePushScale
    _end_pad: [f32; 3],                       // 276 (align(16) tail)
}
const _: () = assert!(std::mem::size_of::<TerrainStageUniforms>() == 288);

impl TerrainStageUniforms {
    /// Per-level uniforms from `DrawClipmap` + the shared block from
    /// `SetTerrainSharedUniforms`. One instance per clipmap level.
    ///
    /// uFlowDebug / uErosionDebug / uTexScale / uNormalStrength / uVariantScale
    /// / uCameraPosition / uSunDirectionWorld / uSparkleStrength moved into the
    /// shared GlobalUniforms buffer; the WGSL reads them from there.
    pub fn build(
        mat_model: [f32; 16],
        level: u32,
        player_position: [f32; 3],
        lookup_minimum: (i64, i64),
        visibility_center: [f32; 2],
    ) -> Self {
        let anchor_spacing = CLIP_ANCHOR_SPACING;
        let clip_origin = [
            (player_position[0] / anchor_spacing + 0.5).floor() * anchor_spacing,
            (player_position[2] / anchor_spacing + 0.5).floor() * anchor_spacing,
        ];
        let spacing = (1u32 << level) as f32;
        let has_coarser_level = level + 1 < CLIP_LEVELS as u32;
        Self {
            mat_model,
            inverse_model: crate::matrices::invert_affine(&mat_model),
            clip_origin,
            spacing,
            next_spacing: spacing * 2.0,
            morph_start: if has_coarser_level {
                CLIP_CELLS as f32 * 0.43 * spacing
            } else {
                1.0e20
            },
            morph_end: if has_coarser_level {
                CLIP_CELLS as f32 * 0.49 * spacing
            } else {
                2.0e20
            },
            sea_level: SEA_LEVEL,
            noise_period: NOISE_PERIOD,
            landform_horizontal_scale: LANDFORM_HORIZONTAL_SCALE,
            landform_vertical_scale: LANDFORM_VERTICAL_SCALE,
            land_profile_curve: LAND_PROFILE_CURVE,
            land_profile_reference: LAND_PROFILE_REFERENCE,
            land_profile_peak: LAND_PROFILE_PEAK,
            waterline_clearance: WATERLINE_CLEARANCE,
            waterline_clearance_scale: WATERLINE_CLEARANCE_SCALE,
            waterline_clearance_decay: WATERLINE_CLEARANCE_DECAY,
            ocean_profile_curve: OCEAN_PROFILE_CURVE,
            ocean_profile_reference: OCEAN_PROFILE_REFERENCE,
            ocean_profile_depth: OCEAN_PROFILE_DEPTH,
            erosion_tile_stride: EROSION_TILE_STRIDE,
            erosion_footprint_size: EROSION_FOOTPRINT_SIZE,
            erosion_output_resolution: EROSION_OUTPUT_RESOLUTION as f32,
            erosion_atlas_pitch: EROSION_ATLAS_PITCH as f32,
            erosion_atlas_size: EROSION_ATLAS_SIZE as f32,
            erosion_atlas_gutter: EROSION_ATLAS_GUTTER as f32,
            _pad0: 0.0,
            erosion_lookup_min_tile: [lookup_minimum.0 as f32, lookup_minimum.1 as f32],
            erosion_lookup_size: EROSION_LOOKUP_DIAMETER as f32,
            _pad1: 0.0,
            erosion_visibility_center: visibility_center,
            erosion_visibility_full_radius: EROSION_VISIBILITY_FULL_RADIUS,
            erosion_visibility_zero_radius: EROSION_VISIBILITY_ZERO_RADIUS,
            waterline_push_land: WATERLINE_PUSH_LAND,
            waterline_push_sea: WATERLINE_PUSH_SEA,
            waterline_push_scale: WATERLINE_PUSH_SCALE,
            _end_pad: [0.0; 3],
        }
    }
}

/// ssao stage uniforms. uProjection moved into the shared GlobalUniforms
/// buffer, which the WGSL reads as globals.projection.
#[repr(C, align(8))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SsaoStageUniforms {
    pub screen_size: [f32; 2],  // 0  uScreenSize (AO target resolution)
    pub radius: f32,            // 8  uRadius
    pub bias: f32,              // 12 uBias
    pub power: f32,             // 16 uPower
    _end_pad: [f32; 3],         // 20
}
const _: () = assert!(std::mem::size_of::<SsaoStageUniforms>() == 32);

/// ssao-blur stage uniforms.
#[repr(C, align(8))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlurStageUniforms {
    pub texel_size: [f32; 2],  // 0  uTexelSize
    pub depth_sharpness: f32,  // 8  uDepthSharpness
    pub normal_sharpness: f32, // 12 uNormalSharpness
}
const _: () = assert!(std::mem::size_of::<BlurStageUniforms>() == 16);

/// composite stage uniforms. uAoTexStrength / uFogDensity / uSunIntensity /
/// uExposure live in the shared GlobalUniforms buffer (settings_a.x/z,
/// params.x/z); only the light direction and the AO strength are per-frame
/// stage data.
#[repr(C, align(16))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CompositeStageUniforms {
    pub light_direction_view: [f32; 4], // 0  uLightDirectionView (w unused)
    pub ao_strength: f32,               // 16 uAoStrength
    _end_pad: [f32; 3],                 // 20
}
const _: () = assert!(std::mem::size_of::<CompositeStageUniforms>() == 32);

/// erosion-init stage uniforms.
#[repr(C, align(8))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ErosionInitStageUniforms {
    pub mode: i32,           // 0  uMode
    _pad0: f32,              // 4
    pub world_min: [f32; 2], // 8  uWorldMin
    pub cell_size: f32,      // 16 uCellSize
    _end_pad: [f32; 3],      // 20
}
const _: () = assert!(std::mem::size_of::<ErosionInitStageUniforms>() == 32);

/// erosion-flux stage uniforms.
#[repr(C, align(8))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ErosionFluxStageUniforms {
    pub resolution: [f32; 2], // 0  uResolution
    pub delta_time: f32,      // 8  uDeltaTime
    pub cell_size: f32,       // 12 uCellSize
    pub gravity: f32,         // 16 uGravity
    _end_pad: [f32; 3],       // 20
}
const _: () = assert!(std::mem::size_of::<ErosionFluxStageUniforms>() == 32);

/// erosion-water stage uniforms.
#[repr(C, align(8))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ErosionWaterStageUniforms {
    pub resolution: [f32; 2],  // 0
    pub delta_time: f32,       // 8
    pub cell_size: f32,        // 12
    pub rain: f32,             // 16
    pub evaporation: f32,      // 20
    pub sea_level: f32,        // 24
    _end_pad: f32,             // 28 (uniform size rounds to 32)
}
const _: () = assert!(std::mem::size_of::<ErosionWaterStageUniforms>() == 32);

/// erosion-terrain stage uniforms.
#[repr(C, align(8))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ErosionTerrainStageUniforms {
    pub resolution: [f32; 2],       // 0
    pub delta_time: f32,            // 8
    pub cell_size: f32,             // 12
    pub sea_level: f32,             // 16
    pub erosion_rate: f32,          // 20
    pub deposition_rate: f32,       // 24
    pub sediment_capacity: f32,     // 28
    pub minimum_slope: f32,         // 32
    pub maximum_erosion: f32,       // 36
    pub transport_rate: f32,        // 40
    pub guard_band_pixels: f32,     // 44
    pub brush_strength: f32,        // 48
    _align: f32,                    // 52
    pub world_min: [f32; 2],        // 56 uWorldMin
}
const _: () = assert!(std::mem::size_of::<ErosionTerrainStageUniforms>() == 64);

/// The group(0) bind group layout every pass shares: one uniform buffer.
///
/// bevy 0.19 pipelines take layout *descriptors* (`RenderPipelineDescriptor`
/// gained a `Vec<BindGroupLayoutDescriptor>`), and the concrete
/// `BindGroupLayout` is resolved through `PipelineCache::get_bind_group_layout`
/// only when a bind group is actually built. So this returns the descriptor.
pub fn globals_layout() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        "forest_globals_layout",
        &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(
                    std::mem::size_of::<GlobalUniformsGpu>() as u64,
                ),
            },
            count: None,
        }],
    )
}

// ---------------------------------------------------------------------------
// Plugin + graph assembly
// ---------------------------------------------------------------------------

/// Shader asset handles for every pipeline.
#[derive(Clone, Resource)]
pub struct ForestShaderHandles {
    pub terrain_vs: Handle<Shader>,
    pub terrain_fs: Handle<Shader>,
    pub grass: Handle<Shader>,
    pub grass_ground_average: Handle<Shader>,
    pub erosion_init: Handle<Shader>,
    pub erosion_flux: Handle<Shader>,
    pub erosion_water: Handle<Shader>,
    pub erosion_terrain: Handle<Shader>,
    pub ssao: Handle<Shader>,
    pub ssao_blur: Handle<Shader>,
    pub composite: Handle<Shader>,
    pub fxaa: Handle<Shader>,
    pub water_surface: Handle<Shader>,
    pub water_underwater: Handle<Shader>,
    pub water_blit: Handle<Shader>,
    pub precipitation: Handle<Shader>,
    pub cloud_probe: Handle<Shader>,
}

pub struct ForestRenderPlugin {
    noise_field: NoiseField,
}

impl ForestRenderPlugin {
    /// Shared CPU data lifted into the render world once at plugin build.
    pub fn new(noise_field: NoiseField) -> Self {
        Self { noise_field }
    }
}

impl Plugin for ForestRenderPlugin {
    fn build(&self, app: &mut App) {
        // Main-world values are read (and the main-world systems registered)
        // BEFORE the render sub-app is borrowed, so `app` is free again by the
        // time `get_sub_app_mut` takes its mutable borrow.
        //
        // Shared cross-world state: the erosion bridge carries one frame of
        // simulation commands at a time from the main world.
        let bridge = app.world().resource::<ErosionBridge>().clone();
        // Shader assets live in the main world's Assets<Shader>; handles work
        // in the render app because PipelineCache watches shader assets.
        let handles = {
            let asset_server = app.world().resource::<AssetServer>();
            ForestShaderHandles {
                terrain_vs: asset_server.load::<Shader>("shaders/terrain-vs.wgsl"),
                terrain_fs: asset_server.load::<Shader>("shaders/terrain-fs.wgsl"),
                grass: asset_server.load::<Shader>("shaders/grass.wgsl"),
                grass_ground_average: asset_server.load::<Shader>("shaders/grass-ground-average.wgsl"),
                erosion_init: asset_server.load::<Shader>("shaders/erosion-init.wgsl"),
                erosion_flux: asset_server.load::<Shader>("shaders/erosion-flux.wgsl"),
                erosion_water: asset_server.load::<Shader>("shaders/erosion-water.wgsl"),
                erosion_terrain: asset_server.load::<Shader>("shaders/erosion-terrain.wgsl"),
                ssao: asset_server.load::<Shader>("shaders/ssao.wgsl"),
                ssao_blur: asset_server.load::<Shader>("shaders/ssao-blur.wgsl"),
                composite: asset_server.load::<Shader>("shaders/composite.wgsl"),
                fxaa: asset_server.load::<Shader>("shaders/fxaa.wgsl"),
                water_surface: asset_server.load::<Shader>("shaders/water-surface.wgsl"),
                water_underwater: asset_server.load::<Shader>("shaders/water-underwater.wgsl"),
                water_blit: asset_server.load::<Shader>("shaders/water-blit.wgsl"),
                precipitation: asset_server.load::<Shader>("shaders/precipitation.wgsl"),
                cloud_probe: asset_server.load::<Shader>("shaders/cloud-probe.wgsl"),
            }
        };
        // The main world's terrain layer images (and the erosion blend mask)
        // are lifted into the render world, since it cannot read them directly.
        gpu_textures::register_main_texture_systems(app);
        // Main-world registration has to happen before the render sub-app is
        // borrowed below.
        water_node::register_water_main_world(app);

        let grass_assets = crate::grass::GrassAssets::load("assets/models")
            .map(|assets| crate::grass::SharedGrassAssets(std::sync::Arc::new(assets)));
        if let Err(error) = &grass_assets {
            error!("Grass assets could not be loaded: {error}");
        }
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        if let Ok(assets) = grass_assets {
            render_app.insert_resource(assets);
        }
        grass_node::register_grass_systems(render_app);
        render_app.insert_resource(bridge);
        render_app.insert_resource(self.noise_field.clone());
        render_app.insert_resource(handles);

        render_app.init_resource::<ExtractedForestView>();
        render_app.init_resource::<ForestGlobals>();
        render_app.init_resource::<gpu_textures::GpuWorldTexturesOption>();
        render_app.init_resource::<erosion_node::ErosionSimState>();
        render_app.init_resource::<terrain_node::TerrainNodeState>();
        render_app.init_resource::<post_nodes::SsaoNodeState>();
        // The vendored ocean. It owns its own extractor, prepare systems and
        // main-world resource so `src/water` stays removable in one piece.
        water_node::register_water_systems(render_app);

        gpu_textures::register_gpu_texture_systems(render_app);
        erosion_node::register_erosion_systems(render_app);
        terrain_node::register_terrain_systems(render_app);
        post_nodes::register_post_systems(render_app);
        cloud_node::register_cloud_systems(render_app);
        precipitation_node::register_precipitation_systems(render_app);

        render_app.add_systems(ExtractSchedule, extract_forest_view);
        render_app.add_systems(
            bevy::render::Render,
            (gpu_textures::prepare_gpu_textures, prepare_forest_globals)
                .chain()
                .in_set(bevy::render::RenderSystems::Prepare),
        );

        // The per-camera pass order. Each pass is a system in this schedule
        // rather than a node in a render graph; the camera opts in through
        // `CameraRenderGraph::new(ForestRender)` in `spawn_scene`.
        render_app.add_schedule(ForestRender::base_schedule());
        render_app.add_systems(
            ForestRender,
            (
                erosion_node::forest_erosion_pass.in_set(ForestRenderSystems::Erosion),
                terrain_node::forest_terrain_pass.in_set(ForestRenderSystems::Terrain),
                grass_node::forest_grass_pass.in_set(ForestRenderSystems::Grass),
                post_nodes::forest_ssao_pass.in_set(ForestRenderSystems::Ssao),
                post_nodes::forest_blur_pass.in_set(ForestRenderSystems::Blur),
                cloud_node::cloud_probe_pass.in_set(ForestRenderSystems::Clouds),
                post_nodes::forest_composite_pass.in_set(ForestRenderSystems::Composite),
                water_node::forest_water_surface_pass.in_set(ForestRenderSystems::WaterSurface),
                water_node::forest_underwater_pass.in_set(ForestRenderSystems::WaterUnderwater),
                precipitation_node::forest_precipitation_pass.in_set(ForestRenderSystems::Precipitation),
                post_nodes::forest_fxaa_pass.in_set(ForestRenderSystems::Fxaa),
                // bevy_egui draws through `egui_pass`, with `prepare_egui_pass`
                // immediately before it to resolve paint callbacks. bevy_egui
                // wires the same pair into its own Core2d/Core3d schedules.
                (
                    bevy_egui::render::prepare_egui_pass,
                    bevy_egui::render::egui_pass,
                )
                    .chain()
                    .in_set(ForestRenderSystems::Egui),
                bevy::core_pipeline::upscaling::upscaling.in_set(ForestRenderSystems::Upscale),
            ),
        );
    }
}

// ---------------------------------------------------------------------------
// Extraction + globals buffer
// ---------------------------------------------------------------------------

/// Camera state + settings + erosion lookup frame copied main -> render.
fn extract_forest_view(
    mut main_world: ResMut<MainWorld>,
    mut view: ResMut<ExtractedForestView>,
) {
    // `ResMut<MainWorld>` derefs twice -- to `MainWorld`, then to `World` --
    // so one `&mut` reaches the main world's storage directly.
    let world: &mut World = &mut main_world;
    let mut player_query = world.query::<&Player>();
    let Ok(player) = player_query.single_mut(world) else {
        return;
    };
    view.player_position = [player.position.x, player.position.y, player.position.z];
    view.player_yaw = player.yaw;
    view.player_pitch = player.pitch;

    if let Ok((window,)) = world.query_filtered::<(&Window,), With<PrimaryWindow>>().single(world) {
        view.physical_width = window.physical_width();
        view.physical_height = window.physical_height();
    }

    view.settings = *world.resource::<AppSettings>();
    view.day_night = *world.resource::<DayNightCycle>();
    view.weather_offset = world.resource::<crate::weather::WeatherMotion>().offset;
    view.weather = *world.resource::<crate::weather::WeatherState>();
    view.draw_ocean = world.resource::<WorldOptions>().draw_ocean;
    let cache = world.resource::<ErosionCache>();
    view.lookup_minimum = (cache.lookup_minimum.x, cache.lookup_minimum.z);
    view.frame += 1;
}

/// The globals uniform buffer shared at group(0) binding(0) by every pass.
#[derive(Resource)]
pub struct ForestGlobals {
    pub globals: GlobalUniformsGpu,
    pub buffer: Option<wgpu::Buffer>,
}

impl Default for ForestGlobals {
    fn default() -> Self {
        Self { globals: GlobalUniformsGpu::default(), buffer: None }
    }
}

pub fn prepare_forest_globals(
    mut globals: ResMut<ForestGlobals>,
    view: Res<ExtractedForestView>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    let width = view.physical_width.max(1) as f32;
    let height = view.physical_height.max(1) as f32;
    let aspect = width / height;
    // GL-style projection with the GL->wgpu z conversion baked in so raster
    // depth matches the C++ (standard depth, near -> 0, far -> 1).
    let projection = crate::matrices::perspective(68.0, aspect, NEAR_PLANE, FAR_PLANE);
    let player = Player {
        position: bevy::math::Vec3::from(view.player_position),
        yaw: view.player_yaw,
        pitch: view.player_pitch,
        ..Player::default()
    };
    let camera = PlayerCamera::from_player(&player);
    let view_matrix = crate::matrices::view_matrix(camera.position, camera.target, camera.up);
    let lighting = crate::day_night::sample(view.day_night.time_hours);
    let weather = view.weather.conditions(&view.settings);
    let heightfield_texel = terrain_node::LIGHTING_HEIGHTFIELD_SPAN
        / terrain_node::LIGHTING_HEIGHTFIELD_SIZE as f32;
    // Move by whole texels so the world-space sampling lattice stays fixed
    // while travelling, avoiding shadow changes from a fractional grid shift.
    let heightfield_snap = heightfield_texel * 8.0;
    globals.globals = GlobalUniformsGpu {
        view: view_matrix,
        projection,
        camera_position: [view.player_position[0], view.player_position[1], view.player_position[2], 0.0],
        sun_direction: lighting.sun_direction,
        viewport: [width, height, 1.0 / width, 1.0 / height],
        params: [
            // Fog extinction is sampled by world position in the shader.
            view.settings.fog_density,
            FAR_PLANE,
            view.settings.exposure,
            view.settings.ssao_enabled as u32 as f32,
        ],
        settings_a: [
            view.settings.sun_intensity * lighting.sun_strength,
            view.settings.texture_scale,
            view.settings.ao_tex_strength,
            view.settings.variant_scale,
        ],
        settings_b: [
            view.settings.normal_strength,
            view.settings.sparkle_strength,
            view.settings.flow_debug as u32 as f32,
            view.settings.erosion_debug as u32 as f32,
        ],
        // The solar disc remains visible at the horizon even after ground
        // irradiance fades out; w keeps the user's unattenuated source power.
        sun_colour: [lighting.sun_colour[0], lighting.sun_colour[1], lighting.sun_colour[2], view.settings.sun_intensity],
        moon_direction: lighting.moon_direction,
        atmosphere: [
            lighting.daylight,
            lighting.moon_intensity,
            weather.sky_overcast,
            view.settings.volumetric_strength,
        ],
        raymarch: [
            view.settings.raymarched_shadows as u32 as f32,
            view.settings.volumetric_lighting as u32 as f32,
            view.settings.water_reflections as u32 as f32,
            view.settings.raymarch_quality.min(2) as f32,
        ],
        heightfield: [
            (view.player_position[0] / heightfield_snap).floor() * heightfield_snap,
            (view.player_position[2] / heightfield_snap).floor() * heightfield_snap,
            terrain_node::LIGHTING_HEIGHTFIELD_SPAN,
            heightfield_texel,
        ],
        clouds: [
            view.settings.clouds_enabled as u32 as f32,
            view.settings.cloud_coverage,
            view.settings.cloud_density,
            view.settings.cloud_base_height,
        ],
        cloud_layer: [
            view.settings.cloud_thickness,
            view.settings.cloud_scale,
            view.settings.cloud_shadow_strength,
            view.settings.raymarch_quality.min(2) as f32,
        ],
        cloud_motion: [
            view.weather_offset[0], view.weather_offset[1],
            view.settings.cloud_detail_strength, 40000.0,
        ],
        weather: [
            view.weather_offset[0], view.weather_offset[1],
            view.weather.climate_bias,
            view.weather.override_bits() as f32,
        ],
        storm: [
            view.weather.precipitation_bias,
            view.weather.precipitation_override() as f32,
            view.weather.elapsed_seconds,
            view.weather.local_precipitation.gust,
        ],
        lightning: [
            view.weather.lightning.position[0],
            view.weather.lightning.position[1],
            view.weather.lightning.position[2],
            view.weather.lightning.flash,
        ],
        lightning_meta: [
            view.weather.lightning.seed,
            view.weather.lightning.age_seconds,
            view.weather.lightning.top_height,
            view.weather.local_precipitation.thunderstorm,
        ],
    };
    if globals.buffer.is_none() {
        // Created straight off the wgpu device so the field can stay a raw
        // `wgpu::Buffer`, the type `globals_bind_group_entries` and the
        // terrain node's layout code take.
        globals.buffer = Some(device.wgpu_device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("forest_global_uniforms"),
            size: std::mem::size_of::<GlobalUniformsGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
    }
    if let Some(buffer) = &globals.buffer {
        queue.write_buffer(buffer, 0, bytemuck::bytes_of(&globals.globals));
    }
}

/// Group-0 bind group entries for the shared globals buffer.
pub fn globals_bind_group_entries(buffer: &wgpu::Buffer) -> [BindGroupEntry<'_>; 1] {
    [BindGroupEntry {
        binding: 0,
        resource: buffer.as_entire_binding(),
    }]
}

/// Builds a bind group from a layout *descriptor*, resolving the cached
/// `BindGroupLayout` through the pipeline cache.
///
/// bevy 0.19 keys pipeline layouts by descriptor, so the passes hold
/// [`BindGroupLayoutDescriptor`]s and only materialise a `BindGroupLayout`
/// here, where wgpu actually needs one.
pub fn bind_group<'a>(
    device: &RenderDevice,
    cache: &PipelineCache,
    label: &'static str,
    descriptor: &BindGroupLayoutDescriptor,
    entries: &'a [BindGroupEntry<'a>],
) -> BindGroup {
    let layout = cache.get_bind_group_layout(descriptor);
    device.create_bind_group(label, &layout, entries)
}

#[cfg(test)]
mod tests {
    use super::GlobalUniformsGpu;

    #[test]
    fn water_and_visible_sky_share_the_same_atmosphere() {
        let common = include_str!("../../assets/shaders/atmosphere-functions.wgslinc").trim();
        for source in [
            include_str!("../../assets/shaders/composite.wgsl"),
            include_str!("../../assets/shaders/water-surface.wgsl"),
            include_str!("../../assets/shaders/water-underwater.wgsl"),
        ] {
            assert!(source.contains(common), "visible and reflected sky helpers diverged");
        }
    }

    #[test]
    fn visible_clouds_reflections_and_shadows_use_one_density_model() {
        let common = include_str!("../../assets/shaders/cloud-functions.wgslinc").trim();
        for source in [
            include_str!("../../assets/shaders/composite.wgsl"),
            include_str!("../../assets/shaders/cloud-probe.wgsl"),
            include_str!("../../assets/shaders/water-surface.wgsl"),
            include_str!("../../assets/shaders/water-underwater.wgsl"),
        ] {
            assert!(source.contains(common), "cloud shape or lighting differs between passes");
        }
    }

    #[test]
    fn rain_snow_and_water_share_one_spatial_storm_field() {
        let common = include_str!("../../assets/shaders/precipitation-functions.wgslinc").trim();
        for source in [
            include_str!("../../assets/shaders/composite.wgsl"),
            include_str!("../../assets/shaders/water-surface.wgsl"),
            include_str!("../../assets/shaders/precipitation.wgsl"),
        ] {
            assert!(source.contains(common), "precipitation field differs between passes");
        }
    }

    /// A stale layout in even an unrelated pass can reinterpret daylight as
    /// a matrix or bind too small a uniform range. Check the actual WGSL ABI
    /// against Rust, including offsets, rather than just matching byte sizes.
    #[test]
    fn every_shader_uses_the_same_global_uniform_layout() {
        let expected = [
            ("view", std::mem::offset_of!(GlobalUniformsGpu, view)),
            ("projection", std::mem::offset_of!(GlobalUniformsGpu, projection)),
            ("camera_position", std::mem::offset_of!(GlobalUniformsGpu, camera_position)),
            ("sun_direction", std::mem::offset_of!(GlobalUniformsGpu, sun_direction)),
            ("viewport", std::mem::offset_of!(GlobalUniformsGpu, viewport)),
            ("params", std::mem::offset_of!(GlobalUniformsGpu, params)),
            ("settings_a", std::mem::offset_of!(GlobalUniformsGpu, settings_a)),
            ("settings_b", std::mem::offset_of!(GlobalUniformsGpu, settings_b)),
            ("sun_colour", std::mem::offset_of!(GlobalUniformsGpu, sun_colour)),
            ("moon_direction", std::mem::offset_of!(GlobalUniformsGpu, moon_direction)),
            ("atmosphere", std::mem::offset_of!(GlobalUniformsGpu, atmosphere)),
            ("raymarch", std::mem::offset_of!(GlobalUniformsGpu, raymarch)),
            ("heightfield", std::mem::offset_of!(GlobalUniformsGpu, heightfield)),
            ("clouds", std::mem::offset_of!(GlobalUniformsGpu, clouds)),
            ("cloud_layer", std::mem::offset_of!(GlobalUniformsGpu, cloud_layer)),
            ("cloud_motion", std::mem::offset_of!(GlobalUniformsGpu, cloud_motion)),
            ("weather", std::mem::offset_of!(GlobalUniformsGpu, weather)),
            ("storm", std::mem::offset_of!(GlobalUniformsGpu, storm)),
            ("lightning", std::mem::offset_of!(GlobalUniformsGpu, lightning)),
            ("lightning_meta", std::mem::offset_of!(GlobalUniformsGpu, lightning_meta)),
        ];
        let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/shaders");
        let mut checked = 0;
        for entry in std::fs::read_dir(directory).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().and_then(|v| v.to_str()) != Some("wgsl") {
                continue;
            }
            let source = std::fs::read_to_string(&path).unwrap();
            let module = naga::front::wgsl::parse_str(&source)
                .unwrap_or_else(|error| panic!("{}: {}", path.display(), error.emit_to_string(&source)));
            for (_, ty) in module.types.iter() {
                if ty.name.as_deref() != Some("GlobalUniforms") {
                    continue;
                }
                let naga::TypeInner::Struct { members, span } = &ty.inner else {
                    panic!("GlobalUniforms must be a struct");
                };
                assert_eq!(*span as usize, std::mem::size_of::<GlobalUniformsGpu>(), "{}", path.display());
                assert_eq!(members.len(), expected.len(), "{}", path.display());
                for (member, (name, offset)) in members.iter().zip(expected) {
                    assert_eq!(member.name.as_deref(), Some(name), "{}", path.display());
                    assert_eq!(member.offset as usize, offset, "{}: {name}", path.display());
                }
                checked += 1;
            }
        }
        assert!(checked >= 10, "expected the complete render pipeline");
    }
}
