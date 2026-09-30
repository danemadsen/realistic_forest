//! The water passes: the ocean surface and the submerged-camera medium.
//!
//! Both draw from `src/water`, the vendored bevy-aqua port (see
//! `src/water/ATTRIBUTION.md`). The split is deliberate: everything aqua's
//! model needs lives under `src/water`, and everything that knows about *this*
//! renderer — the G-buffer, the ping-pong view target, the forest graph — lives
//! here.
//!
//! Graph position: after `CompositePass`, before `FxaaPass`. The water shades
//! using the composite as its refraction/reflection source, undoing gamma and
//! ACES before optical composition. The G-buffer supplies raymarch depth and
//! the terrain heightfield and cloud volume supply sun occlusion. A shared
//! cloud sky probe supplies offscreen reflections and Snell's window.
//! FXAA sees the result.
//!
//! Neither pass has a depth attachment. The surface pass stands in for one by
//! testing the G-buffer's view-space z against the water fragment's own (view
//! space looks down -Z, so a greater z is nearer). That is also why the passes
//! must not be reordered relative to the composite: the G-buffer is written by
//! the terrain pass and stays valid for the whole frame.
//!
//! `--no-water` (`WorldOptions::draw_ocean`) skips the surface pass outright.
//! The placeholder ocean plane it used to skip is gone; the flag now means
//! "no ocean at all", which is what keeps tile seams measurable.

use crate::constants::SEA_LEVEL;
use crate::render::cloud_node::CloudRenderState;
use crate::render::terrain_node::TerrainNodeState;
use crate::render::{globals_layout, ExtractedForestView, ForestGlobals, ForestShaderHandles};
use crate::water::rings::{self, Patch};
use crate::water::{
    displacement_bounds, waves, GpuWave, UnderwaterUniforms, WaterSettings, WaterStageUniforms,
    FRESNEL_EXPONENT, WATER_LOD_COUNT, WATER_SNAP,
};
use bevy::ecs::schedule::IntoScheduleConfigs;
use bevy::mesh::VertexBufferLayout;
use bevy::prelude::{Res, ResMut, Resource, World};
use bevy::render::render_resource::{
    BindGroup, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource,
    BindingType, Buffer, BufferDescriptor, BufferSize, BufferUsages,
    CachedRenderPipelineId, ColorTargetState, ColorWrites, FragmentState, IndexFormat,
    MultisampleState, PipelineCache, PrimitiveState, RenderPipelineDescriptor, SamplerBindingType,
    ShaderStages, TextureFormat, TextureSampleType, TextureViewDimension, VertexAttribute,
    VertexFormat, VertexState, VertexStepMode,
};
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue, ViewQuery};
use bevy::render::view::ViewTarget;
use bevy::render::{ExtractSchedule, Render, RenderSystems};
use std::collections::HashMap;
use std::sync::Mutex;

/// Screen-uv offset applied to the refraction tap, as a fraction of the surface
/// slope, at one metre of optical path and one metre of view distance.
const REFRACTION_SCALE: f32 = 0.5;

/// Multiplier on the shoreline foam coverage; 1.0 is the authored look and 0
/// turns foam off.
const FOAM_SCALE: f32 = 1.0;

/// Eye height, in metres, over which the submerged-camera medium fades in
/// across the surface. Wide enough that crossing the surface does not pop,
/// narrow enough that it never reaches the camera while walking a shore.
const UNDERWATER_FADE_METRES: f32 = 0.6;

/// Artistic gain for the medium's low single-scattering albedo (~0.025).
/// Underwater lighting is reconstructed into HDR before attenuation; this gain
/// preserves the authored water clarity while sunlight follows the day clock.
const MEDIUM_SUN_GAIN: f32 = 15.0;

/// Byte offset of `WaterStageUniforms::params` — 40 waves at 32 bytes plus five
/// LOD ranges at 16. The clock is rewritten here every frame, so the block's
/// static half is uploaded once per sea-state edit.
const SURFACE_PARAMS_OFFSET: u64 = 40 * std::mem::size_of::<GpuWave>() as u64
    + WATER_LOD_COUNT as u64 * 16;

// ---------------------------------------------------------------------------
// Extraction
// ---------------------------------------------------------------------------

/// The main world's water authoring state plus the clock, copied into the
/// render world once per frame. Kept separate from `ExtractedForestView` so the
/// vendored water code stays self-contained: removing `src/water` should mean
/// deleting this file's registration and nothing else.
#[derive(Resource, Clone, Copy, Debug)]
pub struct ExtractedWater {
    pub settings: WaterSettings,
    /// Seconds since startup, the Gerstner phase clock.
    pub elapsed: f32,
    /// World-space eye height, for the submerged-camera admission test.
    pub camera_height: f32,
    /// `--no-water` combined with `settings.enabled`.
    pub draw: bool,
}

impl Default for ExtractedWater {
    fn default() -> Self {
        Self {
            settings: WaterSettings::default(),
            elapsed: 0.0,
            camera_height: 0.0,
            draw: true,
        }
    }
}

/// Copies `WaterSettings`, the clock and the eye height out of the main world.
/// Mirrors `extract_forest_view`: the render world cannot read main-world
/// resources directly, and every `Res<T>` in this system is a *render*-world
/// resource, which is why all four values are fetched through `main_world`.
fn extract_water(
    mut main_world: ResMut<bevy::render::MainWorld>,
    mut extracted: ResMut<ExtractedWater>,
) {
    let world: &mut World = &mut main_world;
    let settings = world
        .get_resource::<WaterSettings>()
        .copied()
        .unwrap_or_default();
    let elapsed = world
        .get_resource::<bevy::time::Time>()
        .map(|time| time.elapsed_secs())
        .unwrap_or(0.0);
    let draw_ocean = world
        .get_resource::<crate::WorldOptions>()
        .is_none_or(|options| options.draw_ocean);
    // The player is the camera; `extract_forest_view` reads the same component
    // to build `globals.camera_position`, so the two agree by construction.
    let camera_height = {
        let mut players = world.query::<&crate::player::Player>();
        players
            .single(world)
            .map(|player| player.position.y)
            .unwrap_or(0.0)
    };

    extracted.settings = settings;
    extracted.elapsed = elapsed;
    extracted.camera_height = camera_height;
    extracted.draw = draw_ocean && settings.enabled;
}

// ---------------------------------------------------------------------------
// Prepared GPU state
// ---------------------------------------------------------------------------

/// One patch's uploaded geometry plus the instance buffer holding every tile
/// that draws it, across all LODs.
struct PatchGeometry {
    vertex: Buffer,
    index: Buffer,
    index_count: u32,
    instance: Buffer,
    instance_count: u32,
}

/// The nine patch meshes with their per-frame tile placements. The geometry is
/// uploaded once; the instance contents are rewritten whenever the ring centre
/// moves.
struct WaterMeshes {
    patches: Vec<PatchGeometry>,
}

/// One tile as the vertex shader sees it: a centre in world XZ, its LOD's tile
/// width in metres, and the Y rotation that points the patch's fat and outer
/// edges away from the ring centre.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct WaterInstance {
    centre: [f32; 2],
    scale: f32,
    rotation: f32,
}

const _: () = assert!(std::mem::size_of::<WaterInstance>() == 16);

/// The two samplers the water samples through. The composited frame is filtered
/// linearly (a resolved colour image); the G-buffer position is point-sampled,
/// because its texel-exact values feed the depth test and the reconstructed
/// world position, the same reason the post passes point-sample it.
struct WaterSamplers {
    linear_clamp: wgpu::Sampler,
    point_clamp: wgpu::Sampler,
}

/// One pass's group-2 uniform slot.
struct WaterStage {
    layout: BindGroupLayoutDescriptor,
    buffer: Buffer,
    group: BindGroup,
}

struct WaterInner {
    globals_layout: Option<BindGroupLayoutDescriptor>,
    globals_group: Option<BindGroup>,
    samplers: Option<WaterSamplers>,
    /// Group 1 for the surface and blit passes: the composited frame at 0/8,
    /// the G-buffer position at 1/9 and terrain heightfield at 2/10.
    surface_screen: Option<BindGroupLayoutDescriptor>,
    /// Group 1 for the underwater pass; the same shape, its own layout so the
    /// two can diverge.
    underwater_screen: Option<BindGroupLayoutDescriptor>,
    surface_stage: Option<WaterStage>,
    underwater_stage: Option<WaterStage>,
    meshes: Option<WaterMeshes>,
    surface_pipeline: HashMap<TextureFormat, CachedRenderPipelineId>,
    blit_pipeline: HashMap<TextureFormat, CachedRenderPipelineId>,
    underwater_pipeline: HashMap<TextureFormat, CachedRenderPipelineId>,
    /// Ring centre the instance buffers currently hold, so a static camera
    /// rewrites nothing.
    snapped_centre: Option<[f32; 2]>,
    /// The sea state the wave block was built from — amplitude, wind and the
    /// flat-surface toggle — as raw bits. All three change every wave in the
    /// block, so they are rebuilt together.
    wave_key: Option<(u32, u32, bool)>,
    /// The per-frame half of the surface uniform: the clock and the sea state
    /// scalars. Rewritten every frame at `SURFACE_PARAMS_OFFSET`.
    surface_frame: [f32; 4],
    /// Submerged-camera admission, from the eye height: 0 fully above the
    /// surface, 1 fully below. Zero skips the pass.
    underwater_fade: f32,
}

/// Shared, interior-mutable state for the two water passes; the passes run in
/// the `ForestRender` schedule and the prepare systems in the `Render`
/// schedule, so the CPU-side resources live behind one mutex.
#[derive(Resource)]
pub struct WaterNodeState {
    inner: Mutex<WaterInner>,
}

impl Default for WaterNodeState {
    fn default() -> Self {
        Self {
            inner: Mutex::new(WaterInner {
                globals_layout: None,
                globals_group: None,
                samplers: None,
                surface_screen: None,
                underwater_screen: None,
                surface_stage: None,
                underwater_stage: None,
                meshes: None,
                surface_pipeline: HashMap::new(),
                blit_pipeline: HashMap::new(),
                underwater_pipeline: HashMap::new(),
                snapped_centre: None,
                wave_key: None,
                surface_frame: [0.0; 4],
                underwater_fade: 0.0,
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// Ring placement
// ---------------------------------------------------------------------------

/// The world XZ the rings are centred on. Snapping to a whole `WATER_SNAP`
/// keeps the innermost tiles from crawling, and here it is exact rather than a
/// compromise: the Gerstner sum is evaluated from each vertex's *world*
/// position, so a tile that jumps by a whole number of its own quads
/// re-evaluates to exactly the same displaced surface. There is nothing to
/// geomorph.
fn snapped_centre(camera_xz: [f32; 2]) -> [f32; 2] {
    [
        (camera_xz[0] / WATER_SNAP).round() * WATER_SNAP,
        (camera_xz[1] / WATER_SNAP).round() * WATER_SNAP,
    ]
}

/// Every tile of every LOD, grouped by patch, in `Patch::ALL` order — one
/// instance per tile. Each group becomes one instanced draw.
fn build_instances(centre: [f32; 2]) -> Vec<Vec<WaterInstance>> {
    let mut grouped: Vec<Vec<WaterInstance>> = vec![Vec::new(); Patch::ALL.len()];
    for lod in 0..WATER_LOD_COUNT {
        let scale = rings::lod_scale(lod);
        for tile in rings::tile_layout(lod) {
            grouped[tile.patch as usize].push(WaterInstance {
                centre: [
                    centre[0] + tile.offset[0] * scale,
                    centre[1] + tile.offset[1] * scale,
                ],
                scale,
                rotation: tile.rotation,
            });
        }
    }
    grouped
}

/// The surface pipeline's two vertex buffers: the patch's local position, and
/// the per-tile placement stepped once per instance.
fn surface_vertex_layouts() -> Vec<VertexBufferLayout> {
    vec![
        VertexBufferLayout {
            array_stride: 8,
            step_mode: VertexStepMode::Vertex,
            attributes: vec![VertexAttribute {
                format: VertexFormat::Float32x2,
                offset: 0,
                shader_location: 0,
            }],
        },
        VertexBufferLayout {
            array_stride: std::mem::size_of::<WaterInstance>() as u64,
            step_mode: VertexStepMode::Instance,
            attributes: vec![
                VertexAttribute {
                    format: VertexFormat::Float32x2,
                    offset: 0,
                    shader_location: 1,
                },
                VertexAttribute {
                    format: VertexFormat::Float32,
                    offset: 8,
                    shader_location: 2,
                },
                VertexAttribute {
                    format: VertexFormat::Float32,
                    offset: 12,
                    shader_location: 3,
                },
            ],
        },
    ]
}

// ---------------------------------------------------------------------------
// Prepare systems
// ---------------------------------------------------------------------------

/// Uploads the nine patch meshes and creates their instance buffers. Runs once;
/// the instance buffers are created empty-ready and filled by
/// [`prepare_water_rings`] on the same frame.
fn prepare_water_meshes(state: Res<WaterNodeState>, device: Res<RenderDevice>) {
    let Ok(mut inner) = state.inner.lock() else {
        return;
    };
    if inner.meshes.is_some() {
        return;
    }
    let grouped = build_instances([0.0, 0.0]);
    let mut patches = Vec::with_capacity(Patch::ALL.len());
    for (patch, instances) in Patch::ALL.into_iter().zip(grouped) {
        let mesh = rings::build_patch(patch);
        patches.push(PatchGeometry {
            vertex: device.create_buffer_with_data(&wgpu::util::BufferInitDescriptor {
                label: Some("forest_water_patch_vertices"),
                contents: bytemuck::cast_slice(&mesh.positions),
                usage: BufferUsages::VERTEX,
            }),
            index: device.create_buffer_with_data(&wgpu::util::BufferInitDescriptor {
                label: Some("forest_water_patch_indices"),
                contents: bytemuck::cast_slice(&mesh.indices),
                usage: BufferUsages::INDEX,
            }),
            index_count: mesh.indices.len() as u32,
            instance: device.create_buffer_with_data(&wgpu::util::BufferInitDescriptor {
                label: Some("forest_water_patch_instances"),
                contents: bytemuck::cast_slice(&instances),
                usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            }),
            instance_count: instances.len() as u32,
        });
    }
    inner.meshes = Some(WaterMeshes { patches });
}

/// Moves the rings with the camera. The instance buffers total 2 KB, so they are
/// rewritten whole whenever the snapped centre changes rather than tracked tile
/// by tile.
fn prepare_water_rings(
    state: Res<WaterNodeState>,
    view: Res<ExtractedForestView>,
    queue: Res<RenderQueue>,
) {
    let Ok(mut inner) = state.inner.lock() else {
        return;
    };
    let centre = snapped_centre([view.player_position[0], view.player_position[2]]);
    if inner.snapped_centre == Some(centre) {
        return;
    }
    let Some(meshes) = inner.meshes.as_ref() else {
        return;
    };
    for (patch, instances) in meshes.patches.iter().zip(build_instances(centre)) {
        queue.write_buffer(&patch.instance, 0, bytemuck::cast_slice(&instances));
    }
    inner.snapped_centre = Some(centre);
}

/// Builds the layouts, samplers, uniform slots and uniform contents both passes
/// draw with. Everything is skipped (and retried next frame) until the G-buffer
/// and the globals buffer exist.
fn prepare_water(
    state: Res<WaterNodeState>,
    globals: Res<ForestGlobals>,
    terrain: Res<TerrainNodeState>,
    water: Res<ExtractedWater>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    pipeline_cache: Res<PipelineCache>,
) {
    let Ok(mut inner) = state.inner.lock() else {
        return;
    };
    let Some(globals_buffer) = globals.buffer.as_ref() else {
        return;
    };
    let Ok(gbuffer_guard) = terrain.gbuffer.lock() else {
        return;
    };
    if gbuffer_guard.is_none() {
        return;
    }

    if inner.globals_layout.is_none() {
        inner.globals_layout = Some(globals_layout());
    }
    if inner.globals_group.is_none() {
        let group = super::bind_group(
            &device,
            &pipeline_cache,
            "forest_water_globals_group",
            inner.globals_layout.as_ref().expect("just built"),
            &[BindGroupEntry {
                binding: 0,
                resource: globals_buffer.as_entire_binding(),
            }],
        );
        inner.globals_group = Some(group);
    }
    if inner.samplers.is_none() {
        inner.samplers = Some(create_samplers(&device));
    }
    if inner.surface_screen.is_none() {
        // Group 1: composited frame and terrain heightfield (bilinear), and
        // G-buffer position (point). Texture N pairs with sampler N + 8.
        inner.surface_screen = Some(screen_layout(
            "forest_water_surface_layout",
            &[true, false, true],
        ));
    }
    if inner.underwater_screen.is_none() {
        inner.underwater_screen = Some(screen_layout(
            "forest_water_underwater_layout",
            &[true, false],
        ));
    }
    if inner.surface_stage.is_none() {
        inner.surface_stage = Some(create_stage(
            &device,
            &pipeline_cache,
            "forest_water_surface_stage",
            std::mem::size_of::<WaterStageUniforms>() as u64,
        ));
    }
    if inner.underwater_stage.is_none() {
        inner.underwater_stage = Some(create_stage(
            &device,
            &pipeline_cache,
            "forest_water_underwater_stage",
            std::mem::size_of::<UnderwaterUniforms>() as u64,
        ));
    }

    let settings = water.settings;
    let flat = settings.flat_surface;
    let wave_key = (
        settings.sea_state_amplitude.to_bits(),
        settings.wind_direction_degrees.to_bits(),
        flat,
    );
    let amplitude = if flat { 0.0 } else { settings.sea_state_amplitude };
    if inner.wave_key != Some(wave_key) {
        let spectrum = waves::build(amplitude, settings.wind_direction_degrees.to_radians());
        let block = build_stage_uniforms(&settings, &spectrum, amplitude, flat);
        inner.surface_frame = block.params;
        inner.wave_key = Some(wave_key);
        if let Some(stage) = inner.surface_stage.as_ref() {
            // The static half: everything outside `params`, which is the one
            // field the clock rewrites every frame. `params` sits in the middle
            // of the block, so this is two writes — the prefix before it and the
            // suffix after it. Writing only the prefix would leave extinction,
            // scatter, surface, sss_tint, misc and flags at their zeroed
            // defaults: no body absorption, a Fresnel exponent of zero (which
            // pins reflectance at 1 and turns the sea into a pure sky mirror),
            // and no foam, subsurface or refraction.
            let bytes = bytemuck::bytes_of(&block);
            let params_end = SURFACE_PARAMS_OFFSET as usize + std::mem::size_of::<[f32; 4]>();
            queue.write_buffer(
                &stage.buffer,
                0,
                &bytes[..SURFACE_PARAMS_OFFSET as usize],
            );
            queue.write_buffer(&stage.buffer, params_end as u64, &bytes[params_end..]);
        }
    }

    // The clock is the only part of the block that moves every frame; it is the
    // first field of `params`, so a 16-byte write at that offset carries it.
    inner.surface_frame[0] = water.elapsed;
    if let Some(stage) = inner.surface_stage.as_ref() {
        queue.write_buffer(
            &stage.buffer,
            SURFACE_PARAMS_OFFSET,
            bytemuck::bytes_of(&inner.surface_frame),
        );
    }

    // Submerged-camera admission. The medium fades in over
    // `UNDERWATER_FADE_METRES` of eye height so crossing the surface does not
    // pop; once fully above it the pass is skipped outright.
    inner.underwater_fade = if settings.underwater_effects {
        ((SEA_LEVEL - water.camera_height) / UNDERWATER_FADE_METRES + 0.5).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let optics = settings.optics;
    let sun_intensity = globals.globals.settings_a[0];
    let sun_colour = globals.globals.sun_colour;
    let underwater = UnderwaterUniforms {
        params: [
            SEA_LEVEL,
            water.camera_height - SEA_LEVEL,
            water.elapsed,
            inner.underwater_fade,
        ],
        extinction: extinction_of(&optics),
        scatter: scatter_of(&optics),
        // Surface sunlight follows both the daylight intensity and sunset tint.
        sun: [
            sun_intensity * sun_colour[0] * 0.30 * MEDIUM_SUN_GAIN,
            sun_intensity * sun_colour[1] * 0.30 * MEDIUM_SUN_GAIN,
            sun_intensity * sun_colour[2] * 0.30 * MEDIUM_SUN_GAIN,
            globals.globals.atmosphere[1] * 0.30 * MEDIUM_SUN_GAIN,
        ],
    };
    if let Some(stage) = inner.underwater_stage.as_ref() {
        queue.write_buffer(&stage.buffer, 0, bytemuck::bytes_of(&underwater));
    }
}

/// Assembles the whole surface block. Split out of [`prepare_water`] so the
/// byte layout lives in one place next to the struct it mirrors.
fn build_stage_uniforms(
    settings: &WaterSettings,
    spectrum: &waves::WaveSpectrum,
    amplitude: f32,
    flat: bool,
) -> WaterStageUniforms {
    let mut block = WaterStageUniforms::default();
    for (slot, wave) in block.waves.iter_mut().zip(spectrum.waves.iter()) {
        *slot = GpuWave {
            direction: wave.direction,
            amplitude: wave.amplitude,
            wave_number: wave.wave_number,
            angular_frequency: wave.angular_frequency,
            phase: wave.phase,
            chop_amplitude: wave.chop_amplitude,
            wavelength: wave.wavelength,
        };
    }
    // Uploaded for parity with aqua's cascade contract, not read by the shader:
    // the band limit here is a continuous function of the pixel footprint (see
    // `sampleSurface`), which is what keeps the ring boundaries seamless
    // without geomorphing.
    for (lod, range) in spectrum.ranges.iter().enumerate() {
        block.ranges[lod] = [range[0], range[1], 0, 0];
    }
    let optics = settings.optics;
    block.params = [0.0, amplitude, SEA_LEVEL, 1.0];
    block.extinction = extinction_of(&optics);
    block.scatter = scatter_of(&optics);
    // The surface's Fresnel inputs, and the only two. `F0` stays at water's own
    // 0.0204 for n = 1.333 — rounding it to 0.02 costs 0.0004 of reflectance at
    // normal incidence and nothing worth the extra constant. The exponent is
    // `FRESNEL_EXPONENT`, whose doc comment carries the fit against the exact
    // curve; it was 5.0 inline here, which is Schlick's general-dielectric fit
    // and reads glossier than water through the 80-88 degree band.
    //
    // Both are written here rather than authored into `WaterOptics` because the
    // underwater pass has no Fresnel: a per-preset value would let the surface
    // and the medium disagree about the same water, which is the one thing the
    // shared-optics contract exists to prevent.
    block.surface = [
        0.02,
        FRESNEL_EXPONENT,
        optics.sun_roughness,
        crate::water::WATER_BASE_SCALE,
    ];
    block.sss_tint = [
        optics.sss_tint[0],
        optics.sss_tint[1],
        optics.sss_tint[2],
        crate::water::WATER_TILE_RESOLUTION as f32,
    ];
    block.misc = [
        REFRACTION_SCALE,
        FOAM_SCALE,
        displacement_bounds(spectrum, flat).max(1e-3),
        0.0,
    ];
    block.flags = [
        if flat { 1.0 } else { 0.0 },
        amplitude,
        settings.wind_direction_degrees.to_radians(),
        rings::horizon_wave_fade_end(),
    ];
    block
}

fn extinction_of(optics: &crate::water::WaterOptics) -> [f32; 4] {
    [
        optics.extinction[0],
        optics.extinction[1],
        optics.extinction[2],
        optics.scatter_scale,
    ]
}

fn scatter_of(optics: &crate::water::WaterOptics) -> [f32; 4] {
    [
        optics.scatter_tint[0],
        optics.scatter_tint[1],
        optics.scatter_tint[2],
        optics.scattering_asymmetry,
    ]
}

// ---------------------------------------------------------------------------
// Render passes
// ---------------------------------------------------------------------------

/// The ocean surface: nine instanced patch draws, one per Crest patch variant,
/// covering the camera's whole ring stack.
pub fn forest_water_surface_pass(
    view: ViewQuery<&ViewTarget>,
    world: &World,
    mut ctx: RenderContext,
) {
    let view = view.into_inner();
    let Some(state) = world.get_resource::<WaterNodeState>() else {
        return;
    };
    let Some(water) = world.get_resource::<ExtractedWater>() else {
        return;
    };
    if !water.draw {
        return;
    }
    let (Some(shaders), Some(extracted)) = (
        world.get_resource::<ForestShaderHandles>(),
        world.get_resource::<ExtractedForestView>(),
    ) else {
        return;
    };
    let Some(terrain) = world.get_resource::<TerrainNodeState>() else {
        return;
    };
    let Some(device) = world.get_resource::<RenderDevice>() else {
        return;
    };
    let Some(pipeline_cache) = world.get_resource::<PipelineCache>() else {
        return;
    };
    let Ok(mut inner) = state.inner.lock() else {
        return;
    };
    let Ok(gbuffer_guard) = terrain.gbuffer.lock() else {
        return;
    };
    let Some(gbuffer) = gbuffer_guard.as_ref() else {
        return;
    };
    let Some(cloud_state) = world.get_resource::<CloudRenderState>() else {
        return;
    };
    let Some(clouds) = cloud_state.resources.as_ref() else {
        return;
    };

    let Some((pipeline_id, blit_id)) = surface_pipelines(
        &mut inner,
        pipeline_cache,
        shaders,
        view.main_texture_format(),
        &clouds.layout,
    ) else {
        return;
    };
    let (Some(pipeline), Some(blit)) = (
        pipeline_cache.get_render_pipeline(pipeline_id),
        pipeline_cache.get_render_pipeline(blit_id),
    ) else {
        return;
    };
    // Wait for both asynchronous pipelines before swapping the frame;
    // otherwise a shader still compiling would leave an unwritten target.
    // Rebuild group 1 from the source the ping-pong hands back each frame.
    let post_process = view.post_process_write();
    let destination: &wgpu::TextureView = post_process.destination;
    let (Some(globals_group), Some(stage), Some(meshes), Some(group)) = (
        inner.globals_group.as_ref(),
        inner.surface_stage.as_ref(),
        inner.meshes.as_ref(),
        surface_group(
            device,
            pipeline_cache,
            &inner,
            &gbuffer.position_view,
            &gbuffer.heightfield_view,
            post_process.source,
        ),
    ) else {
        return;
    };

    let width = extracted.physical_width.max(1) as f32;
    let height = extracted.physical_height.max(1) as f32;
    let mut render_pass = ctx.begin_tracked_render_pass(wgpu::RenderPassDescriptor {
        label: Some("forest_water_surface_pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: destination,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                // The blit below writes every pixel, so nothing is loaded
                // and nothing is cleared.
                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    render_pass.set_viewport(0.0, 0.0, width, height, 0.0, 1.0);
    render_pass.set_bind_group(0, globals_group, &[]);
    render_pass.set_bind_group(1, &group, &[]);

    // The ping-pong's output target holds nothing yet, and the water covers
    // only the pixels it draws, so the composite's frame is copied across
    // first. This is why the pass exists at all rather than drawing the
    // patches straight into the destination.
    render_pass.set_render_pipeline(blit);
    render_pass.draw(0..3, 0..1);

    render_pass.set_render_pipeline(pipeline);
    render_pass.set_bind_group(2, &stage.group, &[]);
    render_pass.set_bind_group(3, &clouds.group, &[]);
    for patch in meshes.patches.iter() {
        if patch.instance_count == 0 {
            continue;
        }
        render_pass.set_vertex_buffer(0, patch.vertex.slice(..));
        render_pass.set_vertex_buffer(1, patch.instance.slice(..));
        render_pass.set_index_buffer(patch.index.slice(..), IndexFormat::Uint32);
        render_pass.draw_indexed(0..patch.index_count, 0, 0..patch.instance_count);
    }
}

/// Queues the surface and blit pipelines for `format` if they are not cached,
/// and returns both ids.
fn surface_pipelines(
    inner: &mut WaterInner,
    pipeline_cache: &PipelineCache,
    shaders: &ForestShaderHandles,
    format: TextureFormat,
    cloud_layout: &BindGroupLayoutDescriptor,
) -> Option<(CachedRenderPipelineId, CachedRenderPipelineId)> {
    let _ = pipeline_cache;
    let surface = match inner.surface_pipeline.get(&format).copied() {
        Some(id) => id,
        None => {
            let mut layout = water_layouts(inner)?;
            layout.push(cloud_layout.clone());
            let id = {
                // `queue_render_pipeline` needs the cache; borrow it through the
                // caller's reference.
                let descriptor = RenderPipelineDescriptor {
                    label: Some("forest_water_surface_pipeline".into()),
                    layout,
                    immediate_size: 0,
                    vertex: VertexState {
                        shader: shaders.water_surface.clone(),
                        shader_defs: Vec::new(),
                        entry_point: Some("vs_main".into()),
                        buffers: surface_vertex_layouts(),
                    },
                    primitive: PrimitiveState::default(),
                    // No depth attachment and no culling: this is a post pass
                    // that tests depth in the shader against the G-buffer, and
                    // the surface has to be drawable from below as well.
                    depth_stencil: None,
                    multisample: MultisampleState::default(),
                    fragment: Some(FragmentState {
                        shader: shaders.water_surface.clone(),
                        shader_defs: Vec::new(),
                        entry_point: Some("fs_main".into()),
                        targets: vec![Some(ColorTargetState {
                            format,
                            // The water writes its own fully resolved colour,
                            // in-scatter included; alpha blending it over the
                            // scene would double-count the medium.
                            blend: None,
                            write_mask: ColorWrites::ALL,
                        })],
                    }),
                    zero_initialize_workgroup_memory: true,
                };
                pipeline_cache.queue_render_pipeline(descriptor)
            };
            inner.surface_pipeline.insert(format, id);
            id
        }
    };
    let blit = match inner.blit_pipeline.get(&format).copied() {
        Some(id) => id,
        None => {
            // Two groups only: the blit reads the composited frame and has no
            // use for the wave block, but shares the surface's group-1 layout
            // so one bind group serves both draws.
            let layout = water_layouts(inner)?.into_iter().take(2).collect();
            let descriptor = RenderPipelineDescriptor {
                label: Some("forest_water_blit_pipeline".into()),
                layout,
                immediate_size: 0,
                vertex: VertexState {
                    shader: shaders.water_blit.clone(),
                    shader_defs: Vec::new(),
                    entry_point: Some("vs_main".into()),
                    buffers: Vec::new(),
                },
                primitive: PrimitiveState::default(),
                depth_stencil: None,
                multisample: MultisampleState::default(),
                fragment: Some(FragmentState {
                    shader: shaders.water_blit.clone(),
                    shader_defs: Vec::new(),
                    entry_point: Some("fs_main".into()),
                    targets: vec![Some(ColorTargetState {
                        format,
                        blend: None,
                        write_mask: ColorWrites::ALL,
                    })],
                }),
                zero_initialize_workgroup_memory: true,
            };
            let id = pipeline_cache.queue_render_pipeline(descriptor);
            inner.blit_pipeline.insert(format, id);
            id
        }
    };
    Some((surface, blit))
}

/// The three group layouts every water pipeline shares, cloned out of the
/// cached state once `prepare_water` has built them.
fn water_layouts(inner: &WaterInner) -> Option<Vec<BindGroupLayoutDescriptor>> {
    Some(vec![
        inner.globals_layout.clone()?,
        inner.surface_screen.clone()?,
        inner.surface_stage.as_ref()?.layout.clone(),
    ])
}

/// The submerged-camera medium, applied to the whole composited frame once the
/// eye crosses the surface.
pub fn forest_underwater_pass(
    view: ViewQuery<&ViewTarget>,
    world: &World,
    mut ctx: RenderContext,
) {
    let view = view.into_inner();
    let Some(state) = world.get_resource::<WaterNodeState>() else {
        return;
    };
    let Some(water) = world.get_resource::<ExtractedWater>() else {
        return;
    };
    if !water.draw {
        return;
    }
    let (Some(shaders), Some(extracted)) = (
        world.get_resource::<ForestShaderHandles>(),
        world.get_resource::<ExtractedForestView>(),
    ) else {
        return;
    };
    let Some(terrain) = world.get_resource::<TerrainNodeState>() else {
        return;
    };
    let Some(device) = world.get_resource::<RenderDevice>() else {
        return;
    };
    let Some(pipeline_cache) = world.get_resource::<PipelineCache>() else {
        return;
    };
    let Ok(mut inner) = state.inner.lock() else {
        return;
    };
    // Above the surface the medium contributes nothing, so the node is pure
    // overhead and is skipped rather than drawn with a zero fade.
    if inner.underwater_fade <= 0.0 {
        return;
    }
    let Ok(gbuffer_guard) = terrain.gbuffer.lock() else {
        return;
    };
    let Some(gbuffer) = gbuffer_guard.as_ref() else {
        return;
    };
    let Some(cloud_state) = world.get_resource::<CloudRenderState>() else {
        return;
    };
    let Some(clouds) = cloud_state.resources.as_ref() else {
        return;
    };

    let format = view.main_texture_format();
    let pipeline_id = match inner.underwater_pipeline.get(&format).copied() {
        Some(id) => id,
        None => {
            let Some((globals_group_layout, screen, stage)) = (|| {
                Some((
                    inner.globals_layout.clone()?,
                    inner.underwater_screen.clone()?,
                    inner.underwater_stage.as_ref()?.layout.clone(),
                ))
            })() else {
                return;
            };
            let descriptor = RenderPipelineDescriptor {
                label: Some("forest_water_underwater_pipeline".into()),
                layout: vec![globals_group_layout, screen, stage, clouds.layout.clone()],
                immediate_size: 0,
                vertex: VertexState {
                    shader: shaders.water_underwater.clone(),
                    shader_defs: Vec::new(),
                    entry_point: Some("vs_main".into()),
                    buffers: Vec::new(),
                },
                primitive: PrimitiveState::default(),
                depth_stencil: None,
                multisample: MultisampleState::default(),
                fragment: Some(FragmentState {
                    shader: shaders.water_underwater.clone(),
                    shader_defs: Vec::new(),
                    entry_point: Some("fs_main".into()),
                    targets: vec![Some(ColorTargetState {
                        format,
                        blend: None,
                        write_mask: ColorWrites::ALL,
                    })],
                }),
                zero_initialize_workgroup_memory: true,
            };
            let id = pipeline_cache.queue_render_pipeline(descriptor);
            inner.underwater_pipeline.insert(format, id);
            id
        }
    };

    let Some(pipeline) = pipeline_cache.get_render_pipeline(pipeline_id) else {
        return;
    };
    let post_process = view.post_process_write();
    let destination: &wgpu::TextureView = post_process.destination;
    let (Some(globals_group), Some(stage), Some(group)) = (
        inner.globals_group.as_ref(),
        inner.underwater_stage.as_ref(),
        underwater_group(
            device,
            pipeline_cache,
            &inner,
            &gbuffer.position_view,
            post_process.source,
        ),
    ) else {
        return;
    };

    let width = extracted.physical_width.max(1) as f32;
    let height = extracted.physical_height.max(1) as f32;
    let mut render_pass = ctx.begin_tracked_render_pass(wgpu::RenderPassDescriptor {
        label: Some("forest_water_underwater_pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: destination,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                // The fullscreen triangle covers every pixel, so the target's
                // previous contents are never read.
                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    render_pass.set_render_pipeline(pipeline);
    render_pass.set_viewport(0.0, 0.0, width, height, 0.0, 1.0);
    render_pass.set_bind_group(0, globals_group, &[]);
    render_pass.set_bind_group(1, &group, &[]);
    render_pass.set_bind_group(2, &stage.group, &[]);
    render_pass.set_bind_group(3, &clouds.group, &[]);
    render_pass.draw(0..3, 0..1);
}

// ---------------------------------------------------------------------------
// Bind group helpers
// ---------------------------------------------------------------------------

/// The surface and blit passes' group-1 bind group: the composited frame at 0/8,
/// the G-buffer position at 1/9 and terrain heightfield at 2/10.
fn surface_group(
    device: &RenderDevice,
    cache: &PipelineCache,
    inner: &WaterInner,
    gbuffer_position: &wgpu::TextureView,
    heightfield: &wgpu::TextureView,
    source: &wgpu::TextureView,
) -> Option<BindGroup> {
    screen_group(
        device,
        cache,
        "forest_water_surface_group",
        inner.surface_screen.as_ref()?,
        inner.samplers.as_ref()?,
        gbuffer_position,
        source,
        Some(heightfield),
    )
}

fn underwater_group(
    device: &RenderDevice,
    cache: &PipelineCache,
    inner: &WaterInner,
    gbuffer_position: &wgpu::TextureView,
    source: &wgpu::TextureView,
) -> Option<BindGroup> {
    screen_group(
        device,
        cache,
        "forest_water_underwater_group",
        inner.underwater_screen.as_ref()?,
        inner.samplers.as_ref()?,
        gbuffer_position,
        source,
        None,
    )
}

fn screen_group(
    device: &RenderDevice,
    cache: &PipelineCache,
    label: &'static str,
    layout: &BindGroupLayoutDescriptor,
    samplers: &WaterSamplers,
    gbuffer_position: &wgpu::TextureView,
    source: &wgpu::TextureView,
    heightfield: Option<&wgpu::TextureView>,
) -> Option<BindGroup> {
    let mut entries = vec![
        BindGroupEntry {
            binding: 0,
            resource: BindingResource::TextureView(source),
        },
        BindGroupEntry {
            binding: 1,
            resource: BindingResource::TextureView(gbuffer_position),
        },
        BindGroupEntry {
            binding: 8,
            resource: BindingResource::Sampler(&samplers.linear_clamp),
        },
        BindGroupEntry {
            binding: 9,
            resource: BindingResource::Sampler(&samplers.point_clamp),
        },
    ];
    if let Some(heightfield) = heightfield {
        entries.extend([
            BindGroupEntry {
                binding: 2,
                resource: BindingResource::TextureView(heightfield),
            },
            BindGroupEntry {
                binding: 10,
                resource: BindingResource::Sampler(&samplers.linear_clamp),
            },
        ]);
    }
    Some(super::bind_group(device, cache, label, layout, &entries))
}

/// One group-1 layout: for every entry in `filterable`, a float texture at
/// binding N whose sampler sits at binding N + 8. The same shape the post
/// passes' `screen_group_layout` builds.
fn screen_layout(label: &'static str, filterable: &[bool]) -> BindGroupLayoutDescriptor {
    let mut entries: Vec<BindGroupLayoutEntry> = Vec::with_capacity(filterable.len() * 2);
    for (index, filterable) in filterable.iter().enumerate() {
        entries.push(BindGroupLayoutEntry {
            binding: index as u32,
            visibility: ShaderStages::FRAGMENT,
            ty: BindingType::Texture {
                sample_type: TextureSampleType::Float { filterable: *filterable },
                view_dimension: TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        });
    }
    for (index, filterable) in filterable.iter().enumerate() {
        entries.push(BindGroupLayoutEntry {
            binding: index as u32 + 8,
            visibility: ShaderStages::FRAGMENT,
            ty: BindingType::Sampler(if *filterable {
                SamplerBindingType::Filtering
            } else {
                SamplerBindingType::NonFiltering
            }),
            count: None,
        });
    }
    BindGroupLayoutDescriptor::new(label, &entries)
}

fn create_samplers(device: &RenderDevice) -> WaterSamplers {
    let wgpu_device = device.wgpu_device();
    WaterSamplers {
        linear_clamp: wgpu_device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("forest_water_linear_clamp"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        }),
        point_clamp: wgpu_device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("forest_water_point_clamp"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        }),
    }
}

/// Creates one pass's group-2 uniform buffer, its single-entry layout and the
/// bind group wrapping the whole buffer.
fn create_stage(
    device: &RenderDevice,
    cache: &PipelineCache,
    label: &'static str,
    size: u64,
) -> WaterStage {
    let layout = BindGroupLayoutDescriptor::new(
        label,
        &[BindGroupLayoutEntry {
            binding: 0,
            // Not fragment-only like the post passes': the surface shader reads
            // the wave block in its vertex stage.
            visibility: ShaderStages::VERTEX_FRAGMENT,
            ty: BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: BufferSize::new(size),
            },
            count: None,
        }],
    );
    let buffer = device.create_buffer(&BufferDescriptor {
        label: Some(label),
        size,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let group = super::bind_group(
        device,
        cache,
        label,
        &layout,
        &[BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    );
    WaterStage { layout, buffer, group }
}

/// Inserts the main-world half of the water setup. Called before
/// `ForestRenderPlugin::build` takes its mutable borrow of the render sub-app,
/// which is why it is separate from [`register_water_systems`].
pub fn register_water_main_world(app: &mut bevy::app::App) {
    app.init_resource::<WaterSettings>();
}

/// Registers the render-world half: the extracted settings, the node state, the
/// extractor and the prepare systems. Called from `ForestRenderPlugin::build`,
/// the only place holding both the main app and the render sub-app.
pub fn register_water_systems(render_app: &mut bevy::app::SubApp) {
    render_app.init_resource::<ExtractedWater>();
    render_app.init_resource::<WaterNodeState>();
    // `ExtractSchedule`, not `RenderSystems::ExtractCommands`: the latter runs
    // in the `Render` schedule, by which time bevy has removed `MainWorld` from
    // the render world, and `extract_water` reads the main world. This is the
    // same schedule `extract_forest_view` and the texture extractors use.
    render_app.add_systems(ExtractSchedule, extract_water);
    render_app.add_systems(
        Render,
        (prepare_water_meshes, prepare_water_rings, prepare_water)
            .chain()
            .in_set(RenderSystems::Prepare)
            .after(crate::render::prepare_forest_globals),
    );
}
