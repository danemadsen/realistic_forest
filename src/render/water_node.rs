//! The water passes: every water surface, and the medium around a submerged
//! eye.
//!
//! Both draw from `src/water`, the vendored bevy-aqua port (see
//! `src/water/ATTRIBUTION.md`), and from the rivers' surfaces
//! (`river_node::ExtractedRivers`). The split is deliberate: everything aqua's
//! model needs lives under `src/water`, and everything that knows about *this*
//! renderer — the G-buffer, the ping-pong view target, the forest graph — lives
//! here.
//!
//! There is one water shader, `assets/shaders/water.wgsl`, and so one set of
//! bindings every water pipeline shares: the globals, the composited frame and
//! the maps the water reads (group 1), one uniform block with the plants'
//! shadow cascades (group 2) and the clouds (group 3). The sea's tiles
//! (`vs_sea`) and the rivers' ribbons and lakes' sheets (`vs_inland`) are two
//! vertex stages feeding the same `fs_water`, which tells them apart only by
//! what their vertices say about the water.
//!
//! Graph position: after `CompositePass`, before `FxaaPass`. The water shades
//! using the composite as its refraction/reflection source, undoing gamma and
//! ACES before optical composition. The G-buffer supplies raymarch depth and
//! the terrain heightfield and cloud volume supply sun occlusion. A shared
//! cloud sky probe supplies offscreen reflections and Snell's window.
//! FXAA sees the result.
//!
//! The surface pass depth-tests against the G-buffer's own depth buffer, which
//! the terrain, plants and grass have finished with by now. The test runs in
//! hardware ahead of the shader, so the sea and rivers hidden behind a
//! hillside or a tree never run their (costly) shading at all; the water
//! writes its own depth too, so a river drawn after the sea never paints over
//! a wave in front of it. The shader still reads the G-buffer's view-space
//! position for the light path through the water. That is also why the
//! passes must not be reordered relative to the composite: the G-buffer is
//! written by the terrain pass and stays valid for the whole frame.
//!
//! The rivers and lakes draw in the same pass after the sea: chunks of their
//! surfaces culled against the view.
//!
//! The medium pass follows the eye into whichever water it is in — the sea,
//! a lake or a river (`Player::water`) — with that water's own optics.
//!
//! `--no-water` (`WorldOptions::draw_ocean`) skips the sea outright. The
//! placeholder ocean plane it used to skip is gone; the flag now means "no
//! ocean at all", which is what keeps tile seams measurable.

use crate::constants::SEA_LEVEL;
use crate::player::{WaterHere, WaterKind};
use crate::render::cloud_node::CloudRenderState;
use crate::render::terrain_node::{TerrainNodeState, shore_heightfield_mapping};
use crate::render::{ExtractedForestView, ForestGlobals, ForestShaderHandles, globals_layout};
use crate::water::rings::{self, Patch};
use crate::water::{
    FRESNEL_EXPONENT, GpuWave, WATER_LOD_COUNT, WATER_SNAP, WaterSettings, WaterStageUniforms,
    displacement_bounds, waves,
};
use bevy::ecs::schedule::IntoScheduleConfigs;
use bevy::mesh::VertexBufferLayout;
use bevy::prelude::{Res, ResMut, Resource, World};
use bevy::render::render_resource::{
    BindGroup, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource,
    BindingType, Buffer, BufferDescriptor, BufferSize, BufferUsages, CachedRenderPipelineId,
    ColorTargetState, ColorWrites, FragmentState, IndexFormat, MultisampleState, PipelineCache,
    PrimitiveState, RenderPipelineDescriptor, SamplerBindingType, ShaderStages, TextureFormat,
    TextureSampleType, TextureViewDimension, VertexAttribute, VertexFormat, VertexState,
    VertexStepMode,
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

/// Group 1's binding for the canopy capture's window (`canopy_window` in
/// water.wgsl), past the five maps at 0..=4.
const CANOPY_WINDOW_BINDING: u32 = 7;

/// Eye height, in metres, over which the submerged-camera medium fades in
/// across the surface. Wide enough that crossing the surface does not pop,
/// narrow enough that it never reaches the camera while walking a shore or
/// floating head out: the band's top half lies under the swimmer's eye.
const UNDERWATER_FADE_METRES: f32 = 0.4;
const _: () = assert!(UNDERWATER_FADE_METRES / 2.0 <= crate::player::SWIM_FREEBOARD);

/// Artistic gain for the medium's low single-scattering albedo (~0.025).
/// Underwater lighting is reconstructed into HDR before attenuation; this gain
/// preserves the authored water clarity while sunlight follows the day clock.
const MEDIUM_SUN_GAIN: f32 = 15.0;

/// The wind at the water, 10 m up, as a share of the wind the clouds ride on.
/// Friction with the ground slows the wind toward the surface: over open
/// water the 10 m wind is commonly a third to a half of the wind a kilometre
/// up, so the default 18 m/s cloud wind is a moderate 6.3 m/s breeze at the
/// water, a few whitecaps on the open sea.
const SURFACE_WIND_SHARE: f32 = 0.35;

/// The frame the viewer is shown, seconds: what a pattern the water carries
/// is judged against, since one carried more than a fraction of its own size
/// between frames strobes and seems to run backwards. That is the frame time
/// the eye (or a camera) has settled into, not the length of the last frame:
/// a single slow frame does not change what the eye integrates, and taking
/// it at face value stripped the ripples and foam from that one frame and
/// spread their slope into a wider glint, a flash at every hitch. So the
/// exposure follows the frame time with a time constant of
/// `EXPOSURE_SETTLE_SECONDS`, and a frame further than a factor of two from
/// it moves it only as far as that factor.
const EXPOSURE_SETTLE_SECONDS: f32 = 0.75;
/// The exposure's range: a 240 Hz display at the short end; at the long end
/// a tenth of a second, past which no frame rate is followed as motion.
const EXPOSURE_MIN_SECONDS: f32 = 1.0 / 240.0;
const EXPOSURE_MAX_SECONDS: f32 = 0.1;
/// The frame a screenshot run is drawn for. Its frames take seconds on a
/// software adapter, which the picture must not show: it is the still of a
/// game running at 60 fps.
const SHOT_EXPOSURE_SECONDS: f32 = 1.0 / 60.0;

/// Byte offset of `WaterStageUniforms::params` — 40 waves at 32 bytes plus five
/// LOD ranges at 16. The clock is rewritten here every frame, so the block's
/// static half is uploaded once per sea-state edit.
const SURFACE_PARAMS_OFFSET: u64 =
    40 * std::mem::size_of::<GpuWave>() as u64 + WATER_LOD_COUNT as u64 * 16;

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
    /// The viewer's exposure, seconds: the frame time the eye has settled
    /// into (see `EXPOSURE_SETTLE_SECONDS`), carried from frame to frame.
    pub exposure_seconds: f32,
    /// World-space eye height, for the submerged-camera admission test.
    pub camera_height: f32,
    /// The water the eye is over or in, if any.
    pub eye_water: Option<WaterHere>,
    /// `--no-water` combined with `settings.enabled`.
    pub draw: bool,
    /// Whether the rivers' and lakes' surfaces are drawn.
    pub rivers_visible: bool,
    /// Baseline wind response. Local storm gusts are sampled per water point
    /// in the shader, while this authored wind change eases over time.
    pub weather_wave_target: f32,
    /// The wind at the water, 10 m up, m/s.
    pub surface_wind: f32,
    /// How gusty it is, 0..1.
    pub gustiness: f32,
}

impl Default for ExtractedWater {
    fn default() -> Self {
        Self {
            settings: WaterSettings::default(),
            elapsed: 0.0,
            exposure_seconds: SHOT_EXPOSURE_SECONDS,
            camera_height: 0.0,
            eye_water: None,
            draw: true,
            rivers_visible: true,
            weather_wave_target: 1.0,
            surface_wind: 0.0,
            gustiness: 0.5,
        }
    }
}

/// Copies `WaterSettings`, the clock, the eye's water and the wind out of the
/// main world. Mirrors `extract_forest_view`: the render world cannot read
/// main-world resources directly, and every `Res<T>` in this system is a
/// *render*-world resource, which is why every value is fetched through
/// `main_world`.
fn extract_water(
    mut main_world: ResMut<bevy::render::MainWorld>,
    mut extracted: ResMut<ExtractedWater>,
) {
    let world: &mut World = &mut main_world;
    let settings = world
        .get_resource::<WaterSettings>()
        .copied()
        .unwrap_or_default();
    let (elapsed, frame_seconds) = world
        .get_resource::<bevy::time::Time>()
        .map_or((0.0, 1.0 / 60.0), |time| (time.elapsed_secs(), time.delta_secs()));
    let draw_ocean = world
        .get_resource::<crate::WorldOptions>()
        .is_none_or(|options| options.draw_ocean);
    let shot = world
        .get_resource::<crate::automation::AutomationSettings>()
        .is_some_and(|automation| automation.shot_path.is_some());
    // The player is the camera; `extract_forest_view` reads the same component
    // to build `globals.camera_position`, so the two agree by construction.
    let (camera_height, eye_water) = {
        let mut players = world.query::<&crate::player::Player>();
        players
            .single(world)
            .map_or((0.0, None), |player| (player.position.y, player.water))
    };
    let app = world.get_resource::<crate::constants::AppSettings>().cloned();
    let conditions = app.as_ref().and_then(|app| {
        world
            .get_resource::<crate::weather::WeatherState>()
            .map(|weather| weather.conditions(app))
    });

    extracted.settings = settings;
    extracted.elapsed = elapsed;
    extracted.exposure_seconds = viewer_exposure(extracted.exposure_seconds, frame_seconds, shot);
    extracted.camera_height = camera_height;
    extracted.eye_water = eye_water;
    extracted.draw = draw_ocean && settings.enabled;
    extracted.rivers_visible = app.as_ref().is_none_or(|app| app.rivers_visible);
    extracted.weather_wave_target = app.as_ref().map_or(1.0, |app| weather_wave_target(app.cloud_wind_speed));
    extracted.surface_wind = conditions.map_or(0.0, |conditions| conditions.wind_speed * SURFACE_WIND_SHARE);
    extracted.gustiness = conditions.map_or(0.5, |conditions| (0.45 + 0.55 * conditions.gust_strength).clamp(0.0, 1.0));
}

/// The viewer's exposure after a frame of `frame_seconds`: an exponential
/// average of the frame time over `EXPOSURE_SETTLE_SECONDS`, into which a
/// frame counts as no more than twice and no less than half the exposure, so
/// a hitch nudges it and a lasting change of frame rate is followed within a
/// second or two. A screenshot run's is always `SHOT_EXPOSURE_SECONDS`.
fn viewer_exposure(exposure: f32, frame_seconds: f32, shot: bool) -> f32 {
    if shot {
        return SHOT_EXPOSURE_SECONDS;
    }
    let exposure = exposure.clamp(EXPOSURE_MIN_SECONDS, EXPOSURE_MAX_SECONDS);
    let frame = frame_seconds.max(0.0);
    let blend = 1.0 - (-frame / EXPOSURE_SETTLE_SECONDS).exp();
    let sample = frame.clamp(0.5 * exposure, 2.0 * exposure);
    (exposure + (sample - exposure) * blend).clamp(EXPOSURE_MIN_SECONDS, EXPOSURE_MAX_SECONDS)
}

/// The square-root wind response gives calm water residual wave energy and
/// prevents strong authored wind from amplifying the spectrum excessively.
fn weather_wave_target(wind_metres_per_second: f32) -> f32 {
    let wind = (wind_metres_per_second.max(0.0) / 18.0).sqrt();
    (0.58 + 0.42 * wind).clamp(0.55, 1.55)
}

/// Wind changes do not instantly rebuild developed waves. The existing
/// spectrum keeps its phase while its gain eases toward the authored wind.
fn settle_wave_gain(current: f32, target: f32, delta_seconds: f32) -> f32 {
    let seconds = if target > current { 18.0 } else { 40.0 };
    let blend = 1.0 - (-delta_seconds.max(0.0) / seconds).exp();
    current + (target - current) * blend
}

impl ExtractedWater {
    /// The water the eye is over or in as it is drawn: the eye's own, or
    /// where that is hidden (the rivers and lakes turned off) the sea under
    /// it, if the sea is drawn and covers the ground there, as it does at a
    /// river's mouth.
    pub fn drawn_eye_water(&self) -> Option<WaterHere> {
        let drawn = |here: &WaterHere| match here.kind {
            WaterKind::Sea => self.draw,
            WaterKind::River | WaterKind::Lake => self.rivers_visible,
        };
        let here = self.eye_water?;
        if drawn(&here) { Some(here) } else { WaterHere::sea(here.ground).filter(drawn) }
    }

    /// Whether the eye is within `margin` metres of going under the drawn
    /// water it is over, or below it.
    pub fn eye_submerged(&self, margin: f32) -> bool {
        self.drawn_eye_water()
            .is_some_and(|here| self.camera_height <= here.surface + margin)
    }
}

/// The water the medium pass puts the eye in: the drawn water it is over
/// (`ExtractedWater::drawn_eye_water`), if the eye is near enough its surface
/// for the medium to have faded in. Returns the uniform block's `eye_water`
/// and `eye_body`.
///
/// `eye_body.w` is how far the eye is under that water, across the same
/// `UNDERWATER_FADE_METRES` band: the side of the water the eye is on, from
/// which `fs_water` sees every surface. It is written whether or not the
/// medium is drawn, since the side the eye is on is where it is, not an
/// effect: with the underwater effects off, an eye under a pond still sees
/// its surface from below.
fn eye_medium(water: &ExtractedWater) -> ([f32; 4], [f32; 4]) {
    let Some(here) = water.drawn_eye_water() else {
        return ([0.0; 4], [0.0; 4]);
    };
    let height = water.camera_height - here.surface;
    let submerged = (-height / UNDERWATER_FADE_METRES + 0.5).clamp(0.0, 1.0);
    if !water.settings.underwater_effects {
        return ([0.0; 4], [0.0, 0.0, 0.0, submerged]);
    }
    // The water's colour is the drawn surface's over the eye (`water_at`), so
    // the medium blends into the sea's and a lake's with the surface.
    ([here.surface, height, submerged, here.sea], [here.still, here.turbulence, here.clarity, submerged])
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

/// The water's uniform block and its group-2 bind group: the block at
/// binding 0, the plants' shadow cascades at 1..=5. Every water pipeline
/// binds the same group.
struct WaterStage {
    layout: BindGroupLayoutDescriptor,
    group: BindGroup,
}

/// The four water pipelines for one target format.
#[derive(Clone, Copy)]
struct WaterPipelines {
    sea: CachedRenderPipelineId,
    inland: CachedRenderPipelineId,
    blit: CachedRenderPipelineId,
    underwater: CachedRenderPipelineId,
}

struct WaterInner {
    globals_layout: Option<BindGroupLayoutDescriptor>,
    globals_group: Option<BindGroup>,
    samplers: Option<WaterSamplers>,
    /// Group 1 for every water pass: the composited frame at 0/8, the
    /// G-buffer position at 1/9, lighting heightfield at 2/10, local seabed
    /// heightfield at 3/11, the canopy over the grass capture at 4/12 and the
    /// capture's window at 7.
    screen: Option<BindGroupLayoutDescriptor>,
    /// The `WaterStageUniforms` block.
    stage_buffer: Option<Buffer>,
    stage: Option<WaterStage>,
    meshes: Option<WaterMeshes>,
    pipelines: HashMap<TextureFormat, WaterPipelines>,
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
    /// (previous real time, settled weather gain). The authored wave spectrum
    /// changes only when the user's WaterSettings change.
    weather_response: Option<(f32, f32)>,
    /// Submerged-camera admission, from the eye height over the eye's water:
    /// 0 fully above the surface, 1 fully below. Zero skips the pass.
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
                screen: None,
                stage_buffer: None,
                stage: None,
                meshes: None,
                pipelines: HashMap::new(),
                snapped_centre: None,
                wave_key: None,
                surface_frame: [0.0; 4],
                weather_response: None,
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

/// Builds the layouts, samplers, uniform block and bind groups every water
/// pass draws with, and writes the block. Everything is skipped (and retried
/// next frame) until the G-buffer, the globals buffer and the plants' shadow
/// targets exist.
#[allow(clippy::too_many_arguments)]
fn prepare_water(
    state: Res<WaterNodeState>,
    globals: Res<ForestGlobals>,
    terrain: Res<TerrainNodeState>,
    water: Res<ExtractedWater>,
    plant_shadows: Res<crate::render::vegetation_shadows::VegetationShadowMaps>,
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
    if inner.screen.is_none() {
        // Group 1: composited frame, the lighting and seabed maps and the
        // canopy over the grass capture (bilinear), plus G-buffer position
        // (point). Texture N pairs with sampler N + 8; the canopy capture's
        // window is the uniform at CANOPY_WINDOW_BINDING.
        inner.screen = Some(screen_layout(
            "forest_water_screen_layout",
            &[true, false, true, true, true],
        ));
    }
    if inner.stage_buffer.is_none() {
        inner.stage_buffer = Some(device.create_buffer(&BufferDescriptor {
            label: Some("forest_water_stage"),
            size: std::mem::size_of::<WaterStageUniforms>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
    }
    if inner.stage.is_none()
        && let (Some(buffer), Some(shadows)) = (inner.stage_buffer.as_ref(), plant_shadows.targets.as_ref())
    {
        inner.stage = Some(create_stage(&device, &pipeline_cache, buffer, shadows));
    }

    let settings = water.settings;
    let flat = settings.flat_surface;
    let wave_key = (
        settings.sea_state_amplitude.to_bits(),
        settings.wind_direction_degrees.to_bits(),
        flat,
    );
    let amplitude = if flat {
        0.0
    } else {
        settings.sea_state_amplitude
    };
    let Some(stage_buffer) = inner.stage_buffer.clone() else {
        return;
    };
    if inner.wave_key != Some(wave_key) {
        let spectrum = waves::build(amplitude, settings.wind_direction_degrees.to_radians());
        let block = build_stage_uniforms(&settings, &spectrum, amplitude, flat);
        inner.surface_frame = block.params;
        inner.wave_key = Some(wave_key);
        // The static half: everything outside `params`. The clock and the
        // per-frame fields are rewritten below. `params` sits in the middle
        // of the block, so this is two writes — the prefix before it and the
        // suffix after it. Writing only the prefix would leave extinction,
        // scatter, surface, sss_tint, misc and flags at their zeroed
        // defaults: no body absorption, a Fresnel exponent of zero (which
        // pins reflectance at 1 and turns the sea into a pure sky mirror),
        // and no foam, subsurface or refraction.
        let bytes = bytemuck::bytes_of(&block);
        let params_end = SURFACE_PARAMS_OFFSET as usize + std::mem::size_of::<[f32; 4]>();
        queue.write_buffer(&stage_buffer, 0, &bytes[..SURFACE_PARAMS_OFFSET as usize]);
        queue.write_buffer(&stage_buffer, params_end as u64, &bytes[params_end..]);
    }

    // Keep the clock and the local seabed mapping current without uploading
    // the whole wave spectrum. The terrain capture uses this same mapping.
    inner.surface_frame[0] = water.elapsed;
    let (last_time, previous_gain) = inner
        .weather_response
        .unwrap_or((water.elapsed, water.weather_wave_target));
    let delta = (water.elapsed - last_time).clamp(0.0, 0.25);
    let weather_gain = settle_wave_gain(previous_gain, water.weather_wave_target, delta);
    inner.weather_response = Some((water.elapsed, weather_gain));
    inner.surface_frame[3] = weather_gain;
    queue.write_buffer(&stage_buffer, SURFACE_PARAMS_OFFSET, bytemuck::bytes_of(&inner.surface_frame));
    let write = |offset: usize, value: [f32; 4]| {
        queue.write_buffer(&stage_buffer, offset as u64, bytemuck::bytes_of(&value));
    };
    // The seabed map is only drawn while the sea is; the rivers and lakes read
    // it too, as the ground of their banks and shores (their ribbons' edges
    // round down onto it), and must know when it is not there. Before its
    // first capture, or the first after a resize, it holds no ground at all:
    // it is there once a capture has been made, and the terrain pass then
    // captures each new window before the water is drawn.
    let shore_captured = gbuffer_guard.as_ref().is_some_and(|gbuffer| gbuffer.shore_heightfield_ready);
    write(
        std::mem::offset_of!(WaterStageUniforms, shore_map),
        if water.draw && shore_captured { shore_heightfield_mapping(globals.globals.camera_position) } else { [0.0; 4] },
    );
    // Where the canopy capture lies is not written here: this runs before
    // the frame's capture, and the capture's own pass writes its window
    // (`GbufferTargets::grass_habitat_mapping_buffer`, group 1).
    write(
        std::mem::offset_of!(WaterStageUniforms, wind),
        [water.surface_wind, water.gustiness, water.exposure_seconds, 0.0],
    );

    // Submerged-camera admission. The medium fades in over
    // `UNDERWATER_FADE_METRES` of eye height across the surface of the
    // eye's water so crossing it does not pop; once fully above it the pass
    // is skipped outright.
    let (eye_water, eye_body) = eye_medium(&water);
    inner.underwater_fade = eye_water[2];
    write(std::mem::offset_of!(WaterStageUniforms, eye_water), eye_water);
    write(std::mem::offset_of!(WaterStageUniforms, eye_body), eye_body);
    let sun_intensity = globals.globals.settings_a[0];
    let sun_colour = globals.globals.sun_colour;
    // Surface sunlight follows both the daylight intensity and sunset tint.
    write(
        std::mem::offset_of!(WaterStageUniforms, medium_sun),
        [
            sun_intensity * sun_colour[0] * 0.30 * MEDIUM_SUN_GAIN,
            sun_intensity * sun_colour[1] * 0.30 * MEDIUM_SUN_GAIN,
            sun_intensity * sun_colour[2] * 0.30 * MEDIUM_SUN_GAIN,
            globals.globals.atmosphere[1] * 0.30 * MEDIUM_SUN_GAIN,
        ],
    );
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
        if flat {
            0.0
        } else {
            let variance = 0.5
                * spectrum
                    .waves
                    .iter()
                    .map(|wave| wave.amplitude * wave.amplitude)
                    .sum::<f32>();
            4.0 * variance.sqrt()
        },
    ];
    block.flags = [
        if flat { 1.0 } else { 0.0 },
        amplitude,
        settings.wind_direction_degrees.to_radians(),
        rings::horizon_wave_fade_end(),
    ];
    block.wind_waves = waves::wind_sea_components(settings.wind_direction_degrees.to_radians());
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

/// Every water surface: the sea's nine instanced patch draws, one per Crest
/// patch variant covering the camera's whole ring stack, then the rivers'
/// ribbons and lakes' sheets the view can see, all through `fs_water`.
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
    let rivers = world
        .get_resource::<super::river_node::ExtractedRivers>()
        .and_then(|rivers| rivers.surface.as_ref())
        .filter(|_| water.rivers_visible);
    if !water.draw && rivers.is_none() {
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
    let Some(globals) = world.get_resource::<ForestGlobals>() else {
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
    // The sea samples the seabed map, which only exists while it is drawn.
    let draw_ocean = water.draw && gbuffer.shore_heightfield_ready;
    if !draw_ocean && rivers.is_none() {
        return;
    }
    // The G-buffer's depth is the water's depth attachment, so the two must
    // agree in size; for the frame a window resize lands in they may not.
    let target = view.main_texture().size();
    if target.width != gbuffer.width || target.height != gbuffer.height {
        return;
    }
    let Some(cloud_state) = world.get_resource::<CloudRenderState>() else {
        return;
    };
    let Some(clouds) = cloud_state.resources.as_ref() else {
        return;
    };

    let Some(ids) = water_pipelines(&mut inner, pipeline_cache, shaders, view.main_texture_format(), &clouds.layout)
    else {
        return;
    };
    // Wait for every pipeline before swapping the frame; otherwise a shader
    // still compiling would leave an unwritten target.
    let (Some(sea), Some(inland), Some(blit), Some(_)) = (
        pipeline_cache.get_render_pipeline(ids.sea),
        pipeline_cache.get_render_pipeline(ids.inland),
        pipeline_cache.get_render_pipeline(ids.blit),
        pipeline_cache.get_render_pipeline(ids.underwater),
    ) else {
        return;
    };
    // Rebuild group 1 from the source the ping-pong hands back each frame.
    let post_process = view.post_process_write();
    let destination: &wgpu::TextureView = post_process.destination;
    let (Some(globals_group), Some(stage), Some(meshes), Some(group)) = (
        inner.globals_group.as_ref(),
        inner.stage.as_ref(),
        inner.meshes.as_ref(),
        screen_bind_group(device, pipeline_cache, &inner, gbuffer, post_process.source),
    ) else {
        return;
    };

    let width = extracted.physical_width.max(1) as f32;
    let height = extracted.physical_height.max(1) as f32;
    let surface_pass_descriptor = wgpu::RenderPassDescriptor {
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
        // The opaque scene's depth: read for the hardware test, written by
        // the water so a river never paints over a nearer wave.
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: &gbuffer.depth_view,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    };
    let mut render_pass = ctx.begin_tracked_render_pass(surface_pass_descriptor);
    render_pass.set_viewport(0.0, 0.0, width, height, 0.0, 1.0);
    render_pass.set_bind_group(0, globals_group, &[]);
    render_pass.set_bind_group(1, &group, &[]);

    // The ping-pong's output target holds nothing yet, and the water covers
    // only the pixels it draws, so the composite's frame is copied across
    // first. This is why the pass exists at all rather than drawing the
    // patches straight into the destination.
    render_pass.set_render_pipeline(blit);
    render_pass.draw(0..3, 0..1);

    render_pass.set_bind_group(2, &stage.group, &[]);
    render_pass.set_bind_group(3, &clouds.group, &[]);
    if draw_ocean {
        render_pass.set_render_pipeline(sea);
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

    if let Some(rivers) = rivers {
        let g = &globals.globals;
        let view_projection = (bevy::math::Mat4::from_cols_array(&g.projection)
            * bevy::math::Mat4::from_cols_array(&g.view))
        .to_cols_array();
        let eye = [g.camera_position[0], g.camera_position[1], g.camera_position[2]];
        let ranges = rivers.visible(&view_projection, eye);
        if !ranges.is_empty() {
            render_pass.set_render_pipeline(inland);
            render_pass.set_vertex_buffer(0, rivers.vertices.slice(..));
            render_pass.set_index_buffer(rivers.indices.slice(..), IndexFormat::Uint32);
            for (first, count) in ranges {
                render_pass.draw_indexed(first..first + count, 0, 0..1);
            }
        }
    }
}

/// The water's depth state: tested against the G-buffer's reverse-Z depth
/// (an exact tie goes to the water, as the terrain pass's comparison does)
/// and written, or for the blit neither.
fn water_depth(test: bool) -> wgpu::DepthStencilState {
    wgpu::DepthStencilState {
        format: TextureFormat::Depth32Float,
        depth_write_enabled: Some(test),
        depth_compare: Some(if test {
            wgpu::CompareFunction::GreaterEqual
        } else {
            wgpu::CompareFunction::Always
        }),
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState::default(),
    }
}

/// The river vertex: `SurfaceVertex` in src/rivers/surface.rs.
fn river_vertex_layout() -> Vec<VertexBufferLayout> {
    let attribute = |location: u32, offset: u64, format: VertexFormat| VertexAttribute {
        format,
        offset,
        shader_location: location,
    };
    vec![VertexBufferLayout {
        array_stride: std::mem::size_of::<crate::rivers::surface::SurfaceVertex>() as u64,
        step_mode: VertexStepMode::Vertex,
        attributes: vec![
            attribute(0, 0, VertexFormat::Float32x3),
            attribute(1, 12, VertexFormat::Float32x2),
            attribute(2, 20, VertexFormat::Float32),
            attribute(3, 24, VertexFormat::Float32),
            attribute(4, 28, VertexFormat::Float32),
            attribute(5, 32, VertexFormat::Float32),
            attribute(6, 36, VertexFormat::Float32),
            attribute(7, 40, VertexFormat::Float32),
            attribute(8, 44, VertexFormat::Float32),
            attribute(9, 48, VertexFormat::Float32),
            attribute(10, 52, VertexFormat::Float32x3),
            attribute(11, 64, VertexFormat::Float32),
            attribute(12, 68, VertexFormat::Float32x4),
        ],
    }]
}

/// Queues the four water pipelines for `format` if they are not cached, and
/// returns their ids. All four are built from the one water shader with the
/// same bind group layouts (the blit uses only the first two groups).
fn water_pipelines(
    inner: &mut WaterInner,
    pipeline_cache: &PipelineCache,
    shaders: &ForestShaderHandles,
    format: TextureFormat,
    cloud_layout: &BindGroupLayoutDescriptor,
) -> Option<WaterPipelines> {
    if let Some(ids) = inner.pipelines.get(&format) {
        return Some(*ids);
    }
    let layout = vec![
        inner.globals_layout.clone()?,
        inner.screen.clone()?,
        inner.stage.as_ref()?.layout.clone(),
        cloud_layout.clone(),
    ];
    let pipeline = |label: &'static str,
                    vertex: &'static str,
                    fragment: &'static str,
                    buffers: Vec<VertexBufferLayout>,
                    layout: Vec<BindGroupLayoutDescriptor>,
                    depth: Option<wgpu::DepthStencilState>| {
        pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
            label: Some(label.into()),
            layout,
            immediate_size: 0,
            vertex: VertexState {
                shader: shaders.water.clone(),
                shader_defs: Vec::new(),
                entry_point: Some(vertex.into()),
                buffers,
            },
            // No culling: a surface has to be drawable from below as well.
            primitive: PrimitiveState::default(),
            depth_stencil: depth,
            multisample: MultisampleState::default(),
            fragment: Some(FragmentState {
                shader: shaders.water.clone(),
                shader_defs: Vec::new(),
                entry_point: Some(fragment.into()),
                targets: vec![Some(ColorTargetState {
                    format,
                    // The water writes its own fully resolved colour,
                    // in-scatter included; alpha blending it over the scene
                    // would double-count the medium.
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            zero_initialize_workgroup_memory: true,
        })
    };
    let ids = WaterPipelines {
        // Depth is the G-buffer's, tested in hardware.
        sea: pipeline(
            "forest_water_sea_pipeline",
            "vs_sea",
            "fs_water",
            surface_vertex_layouts(),
            layout.clone(),
            Some(water_depth(true)),
        ),
        inland: pipeline(
            "forest_water_inland_pipeline",
            "vs_inland",
            "fs_water",
            river_vertex_layout(),
            layout.clone(),
            Some(water_depth(true)),
        ),
        // Two groups only: the blit reads the composited frame and has no
        // use for the uniform block, but shares the surfaces' group-1 layout
        // so one bind group serves every draw.
        blit: pipeline(
            "forest_water_blit_pipeline",
            "vs_fullscreen",
            "fs_blit",
            Vec::new(),
            layout.iter().take(2).cloned().collect(),
            Some(water_depth(false)),
        ),
        underwater: pipeline(
            "forest_water_underwater_pipeline",
            "vs_fullscreen",
            "fs_underwater",
            Vec::new(),
            layout,
            None,
        ),
    };
    inner.pipelines.insert(format, ids);
    Some(ids)
}

/// The submerged-camera medium, applied to the whole composited frame once the
/// eye crosses the surface of the water it is in.
pub fn forest_underwater_pass(view: ViewQuery<&ViewTarget>, world: &World, mut ctx: RenderContext) {
    let view = view.into_inner();
    let Some(state) = world.get_resource::<WaterNodeState>() else {
        return;
    };
    let Some(extracted) = world.get_resource::<ExtractedForestView>() else {
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
    let Ok(inner) = state.inner.lock() else {
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
    // The surface pass queues the pipelines.
    let Some(ids) = inner.pipelines.get(&view.main_texture_format()) else {
        return;
    };
    let Some(pipeline) = pipeline_cache.get_render_pipeline(ids.underwater) else {
        return;
    };
    let post_process = view.post_process_write();
    let destination: &wgpu::TextureView = post_process.destination;
    let (Some(globals_group), Some(stage), Some(group)) = (
        inner.globals_group.as_ref(),
        inner.stage.as_ref(),
        screen_bind_group(device, pipeline_cache, &inner, gbuffer, post_process.source),
    ) else {
        return;
    };

    let width = extracted.physical_width.max(1) as f32;
    let height = extracted.physical_height.max(1) as f32;
    let underwater_pass_descriptor = wgpu::RenderPassDescriptor {
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
    };
    let mut render_pass = ctx.begin_tracked_render_pass(underwater_pass_descriptor);
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

/// Every water pass's group-1 bind group: the composited frame at 0/8, the
/// G-buffer position at 1/9, lighting heightfield at 2/10, local seabed
/// heightfield at 3/11, the canopy over the grass capture at 4/12 and, at
/// CANOPY_WINDOW_BINDING, the window of the capture that canopy was laid over,
/// written with its texels.
fn screen_bind_group(
    device: &RenderDevice,
    cache: &PipelineCache,
    inner: &WaterInner,
    gbuffer: &crate::render::terrain_node::GbufferTargets,
    source: &wgpu::TextureView,
) -> Option<BindGroup> {
    screen_group(
        device,
        cache,
        "forest_water_screen_group",
        inner.screen.as_ref()?,
        inner.samplers.as_ref()?,
        &gbuffer.position_view,
        source,
        &[&gbuffer.heightfield_view, &gbuffer.shore_heightfield_view, &gbuffer.grass_ground_average_view],
        &gbuffer.grass_habitat_mapping_buffer,
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
    // Bilinear maps at bindings 2, 3, ... with their samplers at 10, 11, ...
    maps: &[&wgpu::TextureView],
    canopy_window: &Buffer,
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
        BindGroupEntry {
            binding: CANOPY_WINDOW_BINDING,
            resource: canopy_window.as_entire_binding(),
        },
    ];
    for (index, &map) in maps.iter().enumerate() {
        let binding = 2 + index as u32;
        entries.extend([
            BindGroupEntry {
                binding,
                resource: BindingResource::TextureView(map),
            },
            BindGroupEntry {
                binding: binding + 8,
                resource: BindingResource::Sampler(&samplers.linear_clamp),
            },
        ]);
    }
    Some(super::bind_group(device, cache, label, layout, &entries))
}

/// One group-1 layout: for every entry in `filterable`, a float texture at
/// binding N whose sampler sits at binding N + 8, the same shape the post
/// passes' `screen_group_layout` builds; and the canopy capture's window.
fn screen_layout(label: &'static str, filterable: &[bool]) -> BindGroupLayoutDescriptor {
    assert!(filterable.len() <= CANOPY_WINDOW_BINDING as usize, "a map would take the window's binding");
    let mut entries: Vec<BindGroupLayoutEntry> = Vec::with_capacity(filterable.len() * 2 + 1);
    entries.push(BindGroupLayoutEntry {
        binding: CANOPY_WINDOW_BINDING,
        visibility: ShaderStages::FRAGMENT,
        ty: BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: BufferSize::new(std::mem::size_of::<[f32; 4]>() as u64),
        },
        count: None,
    });
    for (index, filterable) in filterable.iter().enumerate() {
        entries.push(BindGroupLayoutEntry {
            binding: index as u32,
            visibility: if index >= 2 {
                ShaderStages::VERTEX_FRAGMENT
            } else {
                ShaderStages::FRAGMENT
            },
            ty: BindingType::Texture {
                sample_type: TextureSampleType::Float {
                    filterable: *filterable,
                },
                view_dimension: TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        });
    }
    for (index, filterable) in filterable.iter().enumerate() {
        entries.push(BindGroupLayoutEntry {
            binding: index as u32 + 8,
            visibility: if index >= 2 {
                ShaderStages::VERTEX_FRAGMENT
            } else {
                ShaderStages::FRAGMENT
            },
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
    let linear_clamp_descriptor = wgpu::SamplerDescriptor {
        label: Some("forest_water_linear_clamp"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Linear,
        ..Default::default()
    };
    let point_clamp_descriptor = wgpu::SamplerDescriptor {
        label: Some("forest_water_point_clamp"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        ..Default::default()
    };
    let linear_clamp = wgpu_device.create_sampler(&linear_clamp_descriptor);
    let point_clamp = wgpu_device.create_sampler(&point_clamp_descriptor);
    WaterSamplers {
        linear_clamp,
        point_clamp,
    }
}


/// The group-2 layout and bind group every water pipeline shares: the
/// uniform block at binding 0, and the plants' shadow cascades (each depth
/// cascade with its own extent at 1, 4 and 5, their uniform at 2 and their
/// comparison sampler at 3).
fn create_stage(
    device: &RenderDevice,
    cache: &PipelineCache,
    buffer: &Buffer,
    shadows: &crate::render::vegetation_shadows::ShadowTargets,
) -> WaterStage {
    let shadow_texture = |binding| BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::FRAGMENT,
        ty: BindingType::Texture {
            sample_type: TextureSampleType::Depth,
            view_dimension: TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    };
    let entries = [
        BindGroupLayoutEntry {
            binding: 0,
            // Not fragment-only like the post passes': the sea's vertex stage
            // reads the wave block.
            visibility: ShaderStages::VERTEX_FRAGMENT,
            ty: BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: BufferSize::new(std::mem::size_of::<WaterStageUniforms>() as u64),
            },
            count: None,
        },
        shadow_texture(1),
        BindGroupLayoutEntry {
            binding: 2,
            visibility: ShaderStages::FRAGMENT,
            ty: BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: BufferSize::new(
                    std::mem::size_of::<crate::render::vegetation_shadows::ShadowUniform>() as u64,
                ),
            },
            count: None,
        },
        BindGroupLayoutEntry {
            binding: 3,
            visibility: ShaderStages::FRAGMENT,
            ty: BindingType::Sampler(SamplerBindingType::Comparison),
            count: None,
        },
        shadow_texture(4),
        shadow_texture(5),
    ];
    let layout = BindGroupLayoutDescriptor::new("forest_water_stage", &entries);
    let group = super::bind_group(
        device,
        cache,
        "forest_water_stage",
        &layout,
        &[
            BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::TextureView(&shadows.cascade_views[0]),
            },
            BindGroupEntry {
                binding: 2,
                resource: shadows.uniform.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: BindingResource::Sampler(&shadows.sampler),
            },
            BindGroupEntry {
                binding: 4,
                resource: BindingResource::TextureView(&shadows.cascade_views[1]),
            },
            BindGroupEntry {
                binding: 5,
                resource: BindingResource::TextureView(&shadows.cascade_views[2]),
            },
        ],
    );
    WaterStage { layout, group }
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
    let mut config = (prepare_water_meshes, prepare_water_rings, prepare_water).chain();
    config = config.in_set(RenderSystems::Prepare);
    config = config.after(crate::render::prepare_forest_globals);
    render_app.add_systems(Render, config);
}

#[cfg(test)]
mod eye_water_tests {
    use super::*;
    use bevy::math::Vec2;

    /// A river's last reach over the sea floor at its mouth, its surface a
    /// little over the sea's.
    fn estuary() -> WaterHere {
        WaterHere {
            surface: SEA_LEVEL + 0.02,
            ground: SEA_LEVEL - 2.0,
            current: Vec2::new(0.5, 0.0),
            turbulence: 0.1,
            clarity: 0.3,
            sea: 0.8,
            still: 0.6,
            kind: WaterKind::River,
        }
    }

    #[test]
    fn a_hidden_river_leaves_the_eye_in_the_sea_drawn_over_its_mouth() {
        let mut water = ExtractedWater {
            camera_height: SEA_LEVEL - 1.0,
            eye_water: Some(estuary()),
            ..ExtractedWater::default()
        };
        // Drawn, the river's water is the eye's.
        assert!(water.eye_submerged(0.2));
        let (eye, body) = eye_medium(&water);
        assert_eq!((eye[0], eye[3], body[0], body[2]), (SEA_LEVEL + 0.02, 0.8, 0.6, 0.3));
        // Hidden, the sea drawn over the mouth is: the eye is under it, the
        // precipitation stops and the medium is the sea's.
        water.rivers_visible = false;
        assert!(water.eye_submerged(0.2));
        let (eye, body) = eye_medium(&water);
        assert_eq!((eye[0], eye[2], eye[3], body[0], body[2]), (SEA_LEVEL, 1.0, 1.0, 1.0, 0.0));
        // With the sea hidden too, no water is drawn there at all.
        water.draw = false;
        assert!(!water.eye_submerged(0.2));
        assert_eq!(eye_medium(&water), ([0.0; 4], [0.0; 4]));
        // With the sea hidden and the rivers drawn, the river's last reach
        // under its ribbon is drawn, but its mouth past the ribbon is the
        // sea's to draw (`player::water_at`), and nothing is drawn there.
        water.rivers_visible = true;
        assert!(water.eye_submerged(0.2));
        water.eye_water = Some(WaterHere { kind: WaterKind::Sea, ..estuary() });
        assert!(!water.eye_submerged(0.2));
        assert_eq!(eye_medium(&water), ([0.0; 4], [0.0; 4]));
        // A hidden stream over dry ground leaves no sea to fall back to.
        water.rivers_visible = false;
        water.draw = true;
        water.eye_water = Some(WaterHere { surface: SEA_LEVEL + 4.0, ground: SEA_LEVEL + 3.0, ..estuary() });
        water.camera_height = SEA_LEVEL + 3.5;
        assert!(!water.eye_submerged(0.2));
        assert_eq!(eye_medium(&water), ([0.0; 4], [0.0; 4]));
    }

    /// A forest pond, its sheet 6 m over its bed.
    fn pond() -> WaterHere {
        WaterHere {
            surface: 51.46,
            ground: 45.46,
            current: Vec2::ZERO,
            turbulence: 0.0,
            clarity: 0.0,
            sea: 0.0,
            still: 1.0,
            kind: WaterKind::Lake,
        }
    }

    /// The eye's side of the water, `eye_body.w`, which `fs_water` sees every
    /// surface from: an eye in the air sees even a rapid climbing above it
    /// from above, however high it stands over its own water, and only an
    /// eye crossing its own surface leaves the side to each surface's plane.
    #[test]
    fn the_eye_sees_the_water_from_the_side_it_is_on_not_from_its_height() {
        let over = |height: f32| ExtractedWater {
            camera_height: pond().surface + height,
            eye_water: Some(pond()),
            ..ExtractedWater::default()
        };
        // Up in the air, 5 m over the pond: wholly out of it.
        assert_eq!(eye_medium(&over(5.0)).1[3], 0.0);
        // Just over the crossing band, still out of it.
        assert_eq!(eye_medium(&over(UNDERWATER_FADE_METRES * 0.6)).1[3], 0.0);
        // At the surface: half in, the side left to each surface's plane.
        assert!((eye_medium(&over(0.0)).1[3] - 0.5).abs() < 1e-6);
        // A metre under: wholly in it.
        assert_eq!(eye_medium(&over(-1.0)).1[3], 1.0);
        // The side moves with the eye through the band and agrees with the
        // medium's fade wherever the medium is drawn.
        let mut previous = 0.0;
        for step in 0..=20 {
            let height = 0.3 - 0.03 * step as f32;
            let (eye, body) = eye_medium(&over(height));
            assert!(body[3] >= previous, "the side runs back at {height} m");
            assert_eq!(body[3], eye[2]);
            previous = body[3];
        }
        // With the underwater effects off the medium is not drawn, but the
        // eye under the pond still sees its surface from below.
        let mut plain = over(-1.0);
        plain.settings.underwater_effects = false;
        assert_eq!(eye_medium(&plain), ([0.0; 4], [0.0, 0.0, 0.0, 1.0]));
        plain.camera_height = pond().surface + 5.0;
        assert_eq!(eye_medium(&plain), ([0.0; 4], [0.0; 4]));
        // Over dry ground there is no water to be under.
        let dry = ExtractedWater { camera_height: 0.0, eye_water: None, ..ExtractedWater::default() };
        assert_eq!(eye_medium(&dry).1[3], 0.0);
    }
}

#[cfg(test)]
mod exposure_tests {
    use super::*;

    /// Runs `frames` frames of `frame_seconds` from `exposure`.
    fn run(mut exposure: f32, frame_seconds: f32, frames: usize) -> f32 {
        for _ in 0..frames {
            exposure = viewer_exposure(exposure, frame_seconds, false);
        }
        exposure
    }

    #[test]
    fn a_hitch_barely_moves_the_exposure() {
        let steady = run(1.0 / 60.0, 1.0 / 60.0, 600);
        assert!((steady - 1.0 / 60.0).abs() < 1e-6);
        // One 50 ms frame among 60 fps ones, the hitch of a capture or a tile
        // streaming in: the water's detail must not drop out with it.
        let hitch = viewer_exposure(steady, 0.05, false);
        assert!(hitch < steady * 1.08, "{hitch}");
        // Even the longest frame Bevy hands over moves it by under a third.
        assert!(viewer_exposure(steady, 0.25, false) < steady * 1.3);
        // And it settles back.
        assert!((run(hitch, 1.0 / 60.0, 300) - steady).abs() < 1e-5);
    }

    #[test]
    fn a_lasting_frame_rate_is_followed_within_a_second_or_two() {
        let thirty = run(1.0 / 60.0, 1.0 / 30.0, 60);
        assert!(thirty > 0.9 / 30.0 && thirty <= 1.0 / 30.0, "{thirty}");
        let back = run(thirty, 1.0 / 60.0, 120);
        assert!(back < 1.1 / 60.0, "{back}");
        // A paused clock leaves it where it was.
        assert_eq!(viewer_exposure(thirty, 0.0, false), thirty);
    }

    #[test]
    fn the_exposure_keeps_to_its_range() {
        assert_eq!(run(1.0 / 60.0, 1.0 / 1000.0, 10_000), EXPOSURE_MIN_SECONDS);
        assert_eq!(run(1.0 / 60.0, 0.25, 200), EXPOSURE_MAX_SECONDS);
        assert_eq!(viewer_exposure(f32::INFINITY, 0.0, false), EXPOSURE_MAX_SECONDS);
    }

    /// A screenshot run's frames take seconds on a software adapter; its
    /// picture is a still of the game at 60 fps all the same.
    #[test]
    fn a_screenshot_is_exposed_as_a_sixty_fps_frame() {
        assert_eq!(viewer_exposure(EXPOSURE_MAX_SECONDS, 0.25, true), 1.0 / 60.0);
        assert_eq!(viewer_exposure(1.0 / 60.0, 3.0, true), 1.0 / 60.0);
    }
}

#[cfg(test)]
mod shading_tests {
    use super::*;
    use bevy::math::Vec2;
    use bevy::tasks::block_on;
    use wgpu::util::DeviceExt;

    const WATER: &str = include_str!("../../assets/shaders/water.wgsl");

    /// The water shader's module-scope item starting with `head` (`fn name(`
    /// or `const NAME:`), through its end.
    fn item(head: &str) -> &'static str {
        let start = WATER.find(&format!("\n{head}")).unwrap_or_else(|| panic!("water.wgsl has no {head}")) + 1;
        let rest = &WATER[start..];
        let end = if head.starts_with("const") {
            rest.find(";\n").expect("a constant ends") + 2
        } else if head.starts_with("struct") {
            rest.find("\n};\n").expect("a structure ends") + 4
        } else {
            rest.find("\n}\n").expect("a function ends") + 3
        };
        &rest[..end]
    }

    /// Runs `body`, which turns `input` into `samples[id.x]`, over `inputs`
    /// with the water shader's `heads` in scope, on whatever adapter there
    /// is. `None` without one.
    fn evaluate(heads: &[&str], body: &str, inputs: &[[f32; 4]]) -> Option<Vec<[f32; 4]>> {
        evaluate_with("", heads, body, inputs)
    }

    /// [`evaluate`], with `prelude` declared before the shader's items: the
    /// stand-ins for the bindings they read.
    fn evaluate_with(prelude: &str, heads: &[&str], body: &str, inputs: &[[f32; 4]]) -> Option<Vec<[f32; 4]>> {
        let instance = wgpu::Instance::default();
        let Ok(adapter) = block_on(instance.request_adapter(&Default::default())) else {
            eprintln!("skipping water shading check: no GPU adapter");
            return None;
        };
        let (device, queue) = block_on(adapter.request_device(&Default::default())).expect("a device");
        let items = heads.iter().map(|head| item(head)).collect::<Vec<_>>().join("\n");
        let source = format!(
            "{prelude}
             {items}
             @group(0) @binding(0) var<storage, read_write> samples: array<vec4<f32>>;
             @compute @workgroup_size(64)
             fn evaluate(@builtin(global_invocation_id) id: vec3<u32>) {{
                 if (id.x >= arrayLength(&samples)) {{ return; }}
                 let input = samples[id.x];
                 {body}
             }}"
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("water shading check"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("water shading check"),
            layout: None,
            module: &shader,
            entry_point: Some("evaluate"),
            compilation_options: Default::default(),
            cache: None,
        });
        let samples = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("water shading samples"),
            contents: bytemuck::cast_slice(inputs),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("water shading readback"),
            size: samples.size(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("water shading check"),
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
        Some(bytes.chunks_exact(16).map(bytemuck::pod_read_unaligned).collect())
    }

    /// Spread over a rapid at pixel footprints from a few centimetres to tens
    /// of metres, the whitewater over the steps must cover as much of the
    /// river as the near pattern does, and must not change from one pixel to
    /// the next once the pixel can no longer resolve the steps, which is the
    /// crawl and sparkle of a far rapid as the camera moves. The point sample
    /// it was is the measure of that crawl.
    #[test]
    fn distant_rapids_stay_as_white_and_do_not_crawl() {
        let heads = [
            "fn smoothstepf(", "fn hash21(", "fn valueNoise(", "const RIVER_STEP_SPACING:",
            "const VALUE_NOISE_DEVIATION:", "fn octaveResolved(", "fn riverSteps(",
            "fn riverStepShares(", "fn streamFoam(",
        ];
        // Whitewater at full churn, the lace pattern beyond resolving: the
        // shader's own sums, over the thirds of the pixel and at its centre.
        let body = "
            let steps = riverSteps(input.x, input.y, input.zw);
            let shares = riverStepShares(steps.x, steps.y);
            let made = mix(vec3<f32>(0.15), vec3<f32>(0.8), shares);
            let foam = (streamFoam(made.x, 0.5, 0.0) + streamFoam(made.y, 0.5, 0.0)
                        + streamFoam(made.z, 0.5, 0.0))/3.0;
            let point = riverSteps(input.x, input.y, vec2<f32>(0.0));
            let point_made = mix(0.15, 0.8, smoothstepf(0.25, 0.75, point.x));
            samples[id.x] = vec4<f32>(foam, streamFoam(point_made, 0.5, 0.0), 0.0, 0.0);";
        // Metres along and half widths across, scattered down a rapid; each
        // with its neighbours a pixel along and a pixel across.
        let mut seed = 0x2545_f491_u32;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed as f32 / u32::MAX as f32
        };
        let positions: Vec<[f32; 2]> = (0..4096).map(|_| [next() * 4000.0, next() * 2.0 - 1.0]).collect();
        let footprints = [[0.4f32, 0.03], [2.0, 0.15], [6.0, 0.5], [24.0, 1.2]];
        let mut inputs = Vec::new();
        for footprint in footprints {
            for offset in [[0.0, 0.0], [footprint[0], 0.0], [0.0, footprint[1]]] {
                inputs.extend(positions.iter().map(|p| {
                    [p[0] + offset[0], p[1] + offset[1], footprint[0], footprint[1]]
                }));
            }
        }
        let Some(values) = evaluate(&heads, body, &inputs) else {
            return;
        };
        let n = positions.len();
        let mean = |values: &[[f32; 4]], k: usize| values.iter().map(|v| v[k]).sum::<f32>() / values.len() as f32;
        let crawl = |values: &[[f32; 4]], k: usize| {
            let (here, rest) = values.split_at(n);
            let squares: f32 = rest.chunks(n).flat_map(|next| next.iter().zip(here).map(|(a, b)| (a[k] - b[k]).powi(2))).sum();
            (squares / (2 * n) as f32).sqrt()
        };
        let near = mean(&values[..n], 1);
        assert!(near > 0.1, "a rapid is white over its steps: {near}");
        for (index, footprint) in footprints.iter().enumerate() {
            let set = &values[index * 3 * n..(index + 1) * 3 * n];
            let coverage = mean(&set[..n], 0);
            assert!((coverage / near - 1.0).abs() < 0.1, "{footprint:?}: covers {coverage}, near {near}");
            let (filtered, point) = (crawl(set, 0), crawl(set, 1));
            match index {
                // Resolved: the pattern is the near one.
                0 => assert!((filtered - point).abs() < 0.1 * point, "{footprint:?}: {filtered} vs {point}"),
                // The boulders averaged away, the ledges still drawn.
                1 => assert!(filtered < 0.6 * point, "{footprint:?}: {filtered} vs {point}"),
                // Nothing left to resolve: the same from pixel to pixel.
                _ => assert!(filtered < 0.05 * point, "{footprint:?}: {filtered} vs {point}"),
            }
        }
    }

    /// A glint is as wide as the surface's roughness. Where the normal turns
    /// slowly enough across the pixels that the glint spans two of them or
    /// more, it is resolved and must hardly widen; where it turns so fast the
    /// glint would fall between pixels, the glint must spread over its pixel;
    /// and where the normal jumps, the widening stops at its cap.
    #[test]
    fn glints_widen_only_where_a_pixel_cannot_resolve_them() {
        let heads = ["const PIXEL_FILTER_VARIANCE:", "fn pixelRoughness("];
        // x the roughness, y the normal's turn per pixel, z the direction of
        // that turn across the screen.
        let body = "
            let turn = input.y*vec3<f32>(cos(input.z), 0.0, sin(input.z));
            samples[id.x] = vec4<f32>(pixelRoughness(input.x, vec3<f32>(turn.x, 0.0, 0.0),
                                                     vec3<f32>(0.0, 0.0, turn.z)), 0.0, 0.0, 0.0);";
        let mut inputs = Vec::new();
        for alpha in [0.02f32, 0.05, 0.1, 0.3] {
            for direction in [0.0f32, 0.6, 1.3] {
                for turn in [alpha / 2.0, 3.0 * alpha, 3.0] {
                    inputs.push([alpha, turn, direction, 0.0]);
                }
            }
        }
        let Some(values) = evaluate(&heads, body, &inputs) else {
            return;
        };
        for (input, value) in inputs.iter().zip(&values) {
            let [alpha, turn, ..] = *input;
            let seen = value[0];
            assert!(seen >= alpha, "{input:?}: {seen}");
            if turn <= alpha / 2.0 {
                assert!(seen < 1.05 * alpha, "a resolved glint widened: {input:?} {seen}");
            } else if turn < 1.0 {
                assert!(seen / turn > 0.55, "a glint narrower than its pixel: {input:?} {seen}");
            } else {
                assert!((seen - (alpha * alpha + 0.18).sqrt()).abs() < 1e-4, "{input:?}: {seen}");
            }
        }
    }

    /// Which side of the water a surface is seen from is the side the eye is
    /// on, not its height: an eye in the air sees even a rapid standing 20 m
    /// over the eye's own plane from above, foam and all; an eye under the
    /// water sees every surface from below; only an eye crossing its own
    /// surface sees each from the side of the surface's plane it is on, the
    /// foam on it fading out over the last centimetres.
    #[test]
    fn surfaces_are_seen_from_the_side_of_the_water_the_eye_is_on() {
        let heads = ["fn smoothstepf(", "struct SurfaceSide", "fn surfaceSide("];
        let body = "
            let side = surfaceSide(input.x, input.y);
            samples[id.x] = vec4<f32>(select(0.0, 1.0, side.below), side.upper, 0.0, 0.0);";
        // x how far the eye is under its water, y its height over the
        // surface's plane: below a rapid's, a sheet's or a crest's, or over it.
        let overs = [-20.0f32, -1.0, -0.1, -0.01, 0.01, 0.05, 0.1, 1.0, 20.0];
        let unders = [0.0f32, 0.25, 0.5, 0.75, 1.0];
        let inputs: Vec<[f32; 4]> = unders
            .iter()
            .flat_map(|&under| overs.iter().map(move |&over| [under, over, 0.0, 0.0]))
            .collect();
        let Some(values) = evaluate(&heads, body, &inputs) else {
            return;
        };
        for (input, value) in inputs.iter().zip(&values) {
            let [under, over, ..] = *input;
            let (below, upper) = (value[0] > 0.5, value[1]);
            if under <= 0.0 {
                assert!(!below && upper == 1.0, "an eye in the air sees a surface from below: {input:?} {value:?}");
            } else if under >= 1.0 {
                assert!(below && upper == 0.0, "an eye under the water sees a surface from above: {input:?} {value:?}");
            } else {
                assert_eq!(below, over < 0.0, "crossing, the side is the plane's: {input:?} {value:?}");
                assert!((0.0..=1.0).contains(&upper), "{input:?} {value:?}");
            }
        }
        // Crossing, the foam fades in as the eye rises through the surface's
        // plane, and is whole a hand's breadth over it.
        for row in values.chunks(overs.len()).skip(1).take(unders.len() - 2) {
            assert!(row.windows(2).all(|pair| pair[1][1] >= pair[0][1]), "{row:?}");
            assert_eq!((row[2][1], row[overs.len() - 2][1]), (0.0, 1.0), "{row:?}");
        }
    }

    /// The plane a surface is seen by is its triangle's, its normal turned up
    /// out of the water whichever way the triangle is wound, however it lies
    /// across the screen and however small or stretched the pixel's
    /// footprint on it; a triangle whose derivatives line up keeps the
    /// level's.
    #[test]
    fn a_facet_is_turned_up_out_of_the_water() {
        let heads = ["fn facetUp("];
        // x the facet's tilt, y the heading of that tilt, z the footprint's
        // size as a power of ten, w its winding: 1 or -1, or 0 for a
        // footprint along one line.
        let body = "
            let normal = vec3<f32>(sin(input.x)*cos(input.y), cos(input.x), sin(input.x)*sin(input.y));
            let u = normalize(cross(normal, vec3<f32>(0.3, 0.1, 0.95)));
            let v = cross(normal, u);
            let turn = input.x*3.7 + input.y*1.3;
            let size = pow(10.0, input.z);
            let across = (u*cos(turn) + v*sin(turn))*size;
            let down = (v*cos(turn) - u*sin(turn))*size*(1.0 + 19.0*fract(input.y*0.7));
            var dx = across;
            var dy = down;
            if (input.w < 0.0) { dx = down; dy = across; }
            if (input.w == 0.0) { dy = across*2.0; }
            let up = facetUp(dx, dy);
            samples[id.x] = vec4<f32>(up, dot(up, normal));";
        let mut inputs = Vec::new();
        for tilt in [0.0f32, 0.05, 0.4, 0.9, 1.4] {
            for heading in [0.0f32, 1.1, 2.5, 4.0, 5.6] {
                for size in [-4.0f32, -2.0, 0.0, 1.5] {
                    for winding in [1.0f32, -1.0, 0.0] {
                        inputs.push([tilt, heading, size, winding]);
                    }
                }
            }
        }
        let Some(values) = evaluate(&heads, body, &inputs) else {
            return;
        };
        for (input, value) in inputs.iter().zip(&values) {
            if input[3] == 0.0 {
                assert_eq!([value[0], value[1], value[2]], [0.0, 1.0, 0.0], "{input:?}");
            } else {
                assert!(value[3] > 0.9999 && value[1] >= 0.0, "{input:?}: {value:?}");
            }
        }
    }

    /// A bed lies as deep under the surface whatever the angle it is seen
    /// at: a level bed 10 m down shows through 10 m of water at a glance
    /// across a tarn as from high over it, not through the shallower water
    /// a cap on the length along the ray made of it. Only a bed deeper than
    /// the clearest water returns a thousandth of its light from, there and
    /// back, is taken for one that deep.
    #[test]
    fn a_bed_is_as_deep_seen_at_a_glance_as_from_above() {
        let heads = ["const ALPINE_EXTINCTION:", "const BED_DEPTH_MAX:", "fn bedDepth("];
        let body = "
            let returned = exp(-2.0*ALPINE_EXTINCTION*BED_DEPTH_MAX);
            samples[id.x] = vec4<f32>(bedDepth(input.x, input.y), BED_DEPTH_MAX,
                                      max(max(returned.x, returned.y), returned.z), 0.0);";
        // x the straight path from the surface to the bed, y the sine of the
        // eye's elevation over the water.
        let depths = [0.3f32, 2.0, 10.0, 25.0, 60.0];
        let elevations = [3.0f32, 10.0, 30.0, 60.0, 90.0];
        let inputs: Vec<[f32; 4]> = depths
            .iter()
            .flat_map(|&depth| {
                elevations.iter().map(move |&elevation| {
                    let rise = elevation.to_radians().sin();
                    [depth / rise, rise, depth, elevation]
                })
            })
            .collect();
        let Some(values) = evaluate(&heads, body, &inputs) else {
            return;
        };
        let cap = values[0][1];
        assert!((values[0][2] - 1e-3).abs() < 1e-5, "the clearest water returns {} from {cap} m", values[0][2]);
        for (input, value) in inputs.iter().zip(&values) {
            let expected = input[2].min(cap);
            assert!((value[0] - expected).abs() < 1e-3 * expected.max(1.0), "{input:?}: {} not {expected}", value[0]);
        }
        // Looking up from under the surface, no bed lies under it.
        let Some(up) = evaluate(&heads, body, &[[10.0, -0.5, 0.0, 0.0]]) else {
            return;
        };
        assert_eq!(up[0][0], 0.0);
    }

    /// A rapid climbing away above the eye's height, seen from the air, has
    /// its column of water under its tilted face: the eye looks down into it
    /// over the face's own plane, though the ray to it rises. Seen from
    /// below, no bed lies under a surface; a level sheet seen from above is
    /// as deep as it ever was.
    #[test]
    fn a_rapid_above_the_eye_keeps_its_water_column() {
        let heads = ["const ALPINE_EXTINCTION:", "const BED_DEPTH_MAX:", "fn bedDepth(", "fn viewRise("];
        // x the facet's tilt in degrees, y the ray's elevation toward the
        // eye in degrees, z whether the eye sees it from below, w the
        // straight path to the bed behind it.
        let body = "
            let tilt = radians(input.x);
            let facet = vec3<f32>(-sin(tilt), cos(tilt), 0.0);
            let elevation = radians(input.y);
            let to_view = vec3<f32>(-cos(elevation), sin(elevation), 0.0);
            let rise = viewRise(to_view, facet, input.z > 0.5);
            samples[id.x] = vec4<f32>(rise, bedDepth(input.w, rise), 0.0, 0.0);";
        // A 35 degree face climbing away from the eye, which looks up at it
        // from 10 degrees below the point it sees: the ray meets the face 25
        // degrees over its plane.
        let inputs = [[35.0f32, -10.0, 0.0, 2.0], [35.0, -10.0, 1.0, 2.0], [0.0, 30.0, 0.0, 2.0], [0.0, -30.0, 1.0, 2.0]];
        let Some(values) = evaluate(&heads, body, &inputs) else {
            return;
        };
        let expected = 25.0f32.to_radians().sin();
        assert!((values[0][0] - expected).abs() < 1e-4 && (values[0][1] - 2.0 * expected).abs() < 1e-3, "{:?}", values[0]);
        assert_eq!(values[1][1], 0.0, "seen from below, no bed under it: {:?}", values[1]);
        assert!((values[2][0] - 0.5).abs() < 1e-4 && (values[2][1] - 1.0).abs() < 1e-3, "a level sheet: {:?}", values[2]);
        assert_eq!(values[3][1], 0.0, "{:?}", values[3]);
    }

    /// The shader rounds a ribbon's outer strip down onto the ground exactly
    /// as the CPU does (`surface::round_edge`), to the same depth under it.
    #[test]
    fn the_shader_rounds_a_ribbons_edge_as_the_cpu_does() {
        use crate::rivers::surface;
        assert!(item("const EDGE_TUCK:").contains(&format!("= {:?};", surface::EDGE_TUCK)));
        let heads = ["fn roundEdge("];
        let body = "samples[id.x] = vec4<f32>(roundEdge(input.x, input.y, input.z), 0.0, 0.0, 0.0);";
        let mut inputs = Vec::new();
        for (level, cover) in [(10.0f32, 9.2f32), (10.0, 10.4), (52.3, 51.95), (5.0, 2.0)] {
            for out in [0.0f32, 0.3, 0.44, 0.74, 0.97, 1.0, 1.4] {
                inputs.push([level, cover, out, 0.0]);
            }
        }
        let Some(values) = evaluate(&heads, body, &inputs) else {
            return;
        };
        for (input, value) in inputs.iter().zip(&values) {
            let expected = surface::round_edge(input[0], input[1], input[2]);
            assert!((value[0] - expected).abs() < 1e-4, "{input:?}: {} not {expected}", value[0]);
        }
    }

    /// The side each surface is seen from is the side of the water the eye
    /// is on (`surfaceSide`), never how high the eye stands: a source guard,
    /// since every helper may be right while the fragment stage goes back
    /// to the eye's altitude.
    #[test]
    fn the_surface_is_shaded_from_the_eyes_side_not_its_height() {
        let fragment = item("fn fs_water(");
        assert!(fragment.contains("surfaceSide(stage.eye_body.w,"), "fs_water must take its side from surfaceSide");
        assert!(fragment.contains("let underside = side.below;"));
        let squeezed: String = fragment.split_whitespace().collect();
        for altitude in ["camera_position.y<", "camera_position.y>", "<camera_position.y", ">camera_position.y"] {
            assert!(!squeezed.contains(altitude), "fs_water compares the eye's altitude: {altitude}");
        }
    }

    /// The water places the canopy over the habitat capture by the window
    /// that capture's own pass writes, bound beside the canopy texture, and
    /// by nothing in its own block, which is written before the frame's
    /// capture.
    #[test]
    fn the_canopy_is_placed_by_its_capture_window() {
        let binding = format!("@group(1) @binding({CANOPY_WINDOW_BINDING}) var<uniform> canopy_window: vec4<f32>;");
        assert!(WATER.contains(&binding), "water.wgsl must declare {binding}");
        let canopy = item("fn riverCanopy(");
        assert!(canopy.contains("canopy_window.xy") && canopy.contains("canopy_window.z"));
        assert!(!WATER.contains("canopy_map"), "the stage block must not carry a canopy window");
    }

    /// The wind sea's constants are mirrored by hand in the shader.
    #[test]
    fn the_wind_sea_constants_match_their_mirror() {
        use crate::water::waves;
        let constant = |name: &str| item(&format!("const {name}:"));
        assert!(constant("WIND_RUNGS").contains(&format!("= {}u;", waves::WIND_RUNGS)));
        assert!(constant("WIND_PER_RUNG").contains(&format!("= {}u;", waves::WIND_PER_RUNG)));
        assert!(constant("WIND_COMPONENTS").contains(&format!("= {}u;", waves::WIND_COMPONENTS)));
        assert!(constant("WIND_SHORTEST").contains(&format!("= {:?};", waves::WIND_SHORTEST_METRES)));
        assert!(constant("WIND_PERIOD").contains(&format!("= {:?};", waves::WIND_PERIOD_METRES)));
        assert!(constant("SPREAD_FLOOR").contains(&format!("= {:?};", waves::SPREAD_FLOOR)));
    }

    /// Where the wind sea's crests lie and where they head come only from the
    /// table built for the wind's heading. Nothing a pixel computes, its local
    /// wind, its fetch or its maps, may reach a wave vector or a phase, or the
    /// ripples re-phase from frame to frame as those move; nor may the waves
    /// be mirrored pairs about the wind, which cross in a lattice.
    #[test]
    fn the_wind_sea_takes_its_waves_from_the_heading_alone() {
        let sea = item("fn windSea(");
        assert!(sea.contains("stage.wind_waves[rung*WIND_PER_RUNG + j]"));
        for gone in ["hash11", "round(", "wind_direction", "select(-1.0, 1.0"] {
            assert!(!sea.contains(gone), "windSea still uses {gone}");
        }
        let phase = sea.lines().find(|line| line.contains("let phase =")).expect("a phase");
        assert!(phase.contains("dot(wave.xy, wrapped)") && phase.contains("wave.z"), "{phase}");
    }

    /// The shader's spread is the CPU's, which carries the measurements.
    #[test]
    fn the_shader_spreads_the_wind_sea_as_the_cpu_does() {
        use crate::water::waves::spreading_beta;
        let heads = ["fn smoothstepf(", "const SPREAD_FLOOR:", "fn spreadingBeta("];
        let body = "samples[id.x] = vec4<f32>(spreadingBeta(input.x), 0.0, 0.0, 0.0);";
        let inputs: Vec<[f32; 4]> = (0..400).map(|step| [0.05 + step as f32 * 0.02, 0.0, 0.0, 0.0]).collect();
        let Some(values) = evaluate(&heads, body, &inputs) else {
            return;
        };
        for (input, value) in inputs.iter().zip(&values) {
            let expected = spreading_beta(input[0]);
            assert!((value[0] - expected).abs() < 1e-3 * expected, "{}: {} against {expected}", input[0], value[0]);
        }
    }

    /// The wind sea as the shader draws it, over the open sea and a lake: its
    /// slope steeper up the wind than across it, where the crossing pairs
    /// gave the same both ways, and all of its slope accounted for, resolved
    /// or as roughness, however coarse the pixel. On the open sea the ratio
    /// is Cox and Munk's (1.32-1.65 for the whole sea surface); a young sea a
    /// hundred metres from its shore has most of its slope in waves near its
    /// peak, which run closest to the wind, and is more anisotropic still.
    #[test]
    fn the_wind_sea_runs_with_the_wind_and_keeps_its_slope() {
        use crate::water::waves;
        let wind_heading = 0.733f32;
        let table = waves::wind_sea_components(wind_heading)
            .iter()
            .map(|c| format!("vec4<f32>({:?}, {:?}, {:?}, {:?})", c[0], c[1], c[2], c[3]))
            .collect::<Vec<_>>()
            .join(", ");
        let prelude = "struct TestStage { flags: vec4<f32>, wind_waves: array<vec4<f32>, 56> };
                       var<private> stage: TestStage;";
        let heads = [
            "const PI:", "const GRAVITY:", "fn smoothstepf(", "const WIND_RUNGS:", "const WIND_PER_RUNG:",
            "const WIND_COMPONENTS:", "const WIND_SHORTEST:", "const CAPILLARY_CUTOFF:", "const SURFACE_TENSION:",
            "const COX_MUNK_SLOPE:", "const RUNG_LOG_SPAN:", "const WIND_PERIOD:", "const SPREAD_FLOOR:",
            "const WIND_RUNG_STRETCH:", "struct WindSea", "fn windSaturation(", "fn windPeakWavenumber(",
            "fn spreadingBeta(", "fn windSea(",
        ];
        // x, z, the pixel's size, the longest wave the water raises; the
        // wind and the fetch per case.
        let mut seed = 0x9e37_79b9_u32;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed as f32 / u32::MAX as f32
        };
        let positions: Vec<[f32; 2]> = (0..4096).map(|_| [next() * 3000.0 - 1500.0, next() * 3000.0 - 1500.0]).collect();
        let wind = Vec2::from_angle(wind_heading);
        // The open sea under the default breeze, and a lake 100 m down from
        // its upwind shore.
        for (speed, fetch, longest, spread) in [(6.3f32, 1.0e5f32, 2.0f32, 1.2f32..1.9f32), (5.0, 100.0, 3.7, 1.5..4.0)] {
            let body = format!(
                "stage.flags = vec4<f32>(0.0);
                 stage.wind_waves = array<vec4<f32>, 56>({table});
                 let sea = windSea(input.xy, vec2<f32>(0.0), {speed:?}, {fetch:?}, input.w, 0.0,
                                   vec2<f32>(input.z, 0.0), vec2<f32>(0.0, input.z), 0.0, 1.0/60.0);
                 samples[id.x] = vec4<f32>(sea.slope, sea.variance, sea.height);"
            );
            let mut inputs = Vec::new();
            for pixel in [0.002f32, 200.0] {
                inputs.extend(positions.iter().map(|p| [p[0], p[1], pixel, longest]));
            }
            let Some(values) = evaluate_with(prelude, &heads, &body, &inputs) else {
                return;
            };
            let (near, far) = values.split_at(positions.len());
            let n = positions.len() as f32;
            let up = near.iter().map(|v| Vec2::new(v[0], v[1]).dot(wind).powi(2)).sum::<f32>() / n;
            let across = near.iter().map(|v| Vec2::new(v[0], v[1]).perp_dot(wind).powi(2)).sum::<f32>() / n;
            let rough = near.iter().map(|v| v[2]).sum::<f32>() / n;
            let total = far.iter().map(|v| v[2]).sum::<f32>() / n;
            let ratio = up / across;
            assert!(spread.contains(&ratio), "{speed} m/s over {fetch} m: up/cross {ratio}");
            let kept = (up + across + rough) / total;
            assert!((kept - 1.0).abs() < 0.08, "{speed} m/s over {fetch} m: {up} + {across} + {rough} of {total}");
            assert!(far.iter().all(|v| v[0] == 0.0 && v[1] == 0.0), "a coarse pixel resolved a ripple");
        }
    }

    /// The cat's paws are drawn out down the wind, not cells on the world's
    /// axes: the gust field stays alike much further along the wind than
    /// across it, and is as strong as before on average.
    #[test]
    fn gusts_are_drawn_out_down_the_wind() {
        let heads = ["fn hash21(", "fn latticeGradient(", "fn gradientNoise(", "const GUST_STRETCH:", "fn windGust("];
        let heading = 0.733f32;
        let wind = Vec2::from_angle(heading);
        let body = format!(
            "samples[id.x] = vec4<f32>(windGust(input.xy, vec2<f32>({:?}, {:?}), 6.3, input.z), 0.0, 0.0, 0.0);",
            wind.x, wind.y
        );
        let mut seed = 0x2545_f491_u32;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed as f32 / u32::MAX as f32
        };
        let points: Vec<Vec2> = (0..8192).map(|_| Vec2::new(next() * 4000.0 - 2000.0, next() * 4000.0 - 2000.0)).collect();
        let lag = 15.0;
        let mut inputs = Vec::new();
        for offset in [Vec2::ZERO, wind * lag, wind.perp() * lag] {
            inputs.extend(points.iter().map(|p| [p.x + offset.x, p.y + offset.y, 40.0, 0.0]));
        }
        let Some(values) = evaluate(&heads, &body, &inputs) else {
            return;
        };
        let field: Vec<&[[f32; 4]]> = values.chunks(points.len()).collect();
        let mean = field[0].iter().map(|v| v[0]).sum::<f32>() / points.len() as f32;
        let deviation = (field[0].iter().map(|v| (v[0] - mean).powi(2)).sum::<f32>() / points.len() as f32).sqrt();
        let correlation = |other: &[[f32; 4]]| {
            field[0].iter().zip(other).map(|(a, b)| (a[0] - mean) * (b[0] - mean)).sum::<f32>()
                / points.len() as f32
                / (deviation * deviation)
        };
        assert!((mean - 0.5).abs() < 0.02, "mean {mean}");
        // The value noise it replaced varied by 0.158 about its mean.
        assert!((deviation - 0.158).abs() < 0.02, "deviation {deviation}");
        let (along, across) = (correlation(field[1]), correlation(field[2]));
        assert!(along > across + 0.3, "{lag} m down the wind {along}, across it {across}");
    }

    /// The stage block is mirrored by hand on both sides; a field added to one
    /// alone would shift everything after it.
    #[test]
    fn the_stage_block_matches_its_wgsl_mirror() {
        let module = naga::front::wgsl::parse_str(WATER).expect("water.wgsl parses");
        let (span, members) = module
            .types
            .iter()
            .find_map(|(_, ty)| match (&ty.name, &ty.inner) {
                (Some(name), naga::TypeInner::Struct { members, span }) if name == "WaterStageUniforms" => {
                    Some((*span as usize, members.clone()))
                }
                _ => None,
            })
            .expect("water.wgsl declares WaterStageUniforms");
        let wgsl: Vec<(String, usize)> = members
            .iter()
            .map(|member| (member.name.clone().unwrap_or_default(), member.offset as usize))
            .collect();
        macro_rules! offsets {
            ($($field:ident),+) => {
                vec![$((stringify!($field).to_string(), std::mem::offset_of!(WaterStageUniforms, $field))),+]
            };
        }
        let rust = offsets!(
            waves, ranges, params, extinction, scatter, surface, sss_tint, misc, flags, shore_map, wind,
            eye_water, eye_body, medium_sun, wind_waves
        );
        assert_eq!(wgsl, rust);
        assert_eq!(span, std::mem::size_of::<WaterStageUniforms>());
    }
}
