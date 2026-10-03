//! Hydraulic erosion simulation: ports the GPU half of the C++
//! `HydraulicErosion` (`InitErosion`, `InitializeErosionTextures`,
//! `RunErosionIterations`, `FinalizeErosionTile`) into one render pass.
//!
//! Frame flow, in the C++ order:
//!
//! 1. an init frame stamps the tile's base heights and initial loose cover
//!    into the terrain targets, clears water and flux, and stamps the CPU-
//!    routed drainage area (`uMode` 0, 1, then 2), resetting all four
//!    ping-pong indices to zero;
//! 2. `RunErosionIterations` runs flux -> water -> terrain -> thermal into the
//!    opposite targets in that order, so every pass reads the state its
//!    predecessor wrote in the same iteration. The water pass also writes each
//!    cell's drainage-routing normaliser, and the terrain pass writes the
//!    terrain and drainage states together (two colour targets each);
//! 3. a finalize frame copies terrain, water and drainage back to the CPU,
//!    which crops the retained footprint, derives the two atlas patches and
//!    publishes `ErosionEvent::TileFinalized`.
//!
//! The main world hands one frame of work over through [`ErosionBridge`]; the
//! render world owns every GPU object behind [`ErosionSimState`]'s mutex,
//! because a render pass only ever sees `&World`.
//!
//! PORT NOTES (deliberate deviations, all forced by the API):
//!
//! * `queue.write_buffer` lands before the whole submission, so one uniform
//!   buffer cannot hold `uMode` 0 for the terrain pass and `uMode` 1 for the
//!   water/flux passes of the same frame. The init stage therefore owns one
//!   uniform buffer per mode (three, with the drainage stamp), which
//!   preserves the C++ pass order exactly.
//! * `FinalizeErosionTile` reads the state back synchronously; wgpu only maps
//!   a buffer asynchronously, and a `map_async` issued in the frame that
//!   records the copy would wait for the wrong submission. The copy is
//!   recorded at the end of the finalize frame and mapped on a later frame,
//!   so the event lands a frame or two after the tile's last iteration. The
//!   main world already repeats `finalize` until the tile leaves
//!   `Simulating`, so no work is lost.
//! * The atlas slot a tile receives lives in the main world's cache and never
//!   reaches the render world. The finished 260x260 patches are held until the
//!   lookup records publish the tile's slot (`UpdateErosionLookup` writes the
//!   same numbers the C++ uploads from), then uploaded to that slot.
//! * Bevy's pipeline compilation is asynchronous: a frame whose shaders are
//!   still compiling is held and replayed rather than dropped, because `init`
//!   in particular is one-shot. The same holds one level up: a frame the node
//!   never reached — it returns before `take_commands` until the sim exists —
//!   is merged into the next one by `ErosionBridge::set_frame`, so a command
//!   set can carry more than one frame's iteration count.
//! * The finalize readback is asynchronous, so the frame that records the copy
//!   is not the frame that finalizes the tile. `apply_commands` skips issuing a
//!   new readback on a frame that consumed one, and never records a second
//!   copy for a tile it already delivered.

use crate::constants::*;
use crate::erosion::{
    ErosionBridge, ErosionEvent, ErosionFrameCommands, FinalizedTile, TileDiagnostics, TileKey,
};
use crate::render::gpu_textures::{
    write_padded_at, GpuWorldTextures, GpuWorldTexturesOption, SIM_TEXTURE_SIZE,
};
use crate::render::{
    globals_bind_group_entries, globals_layout, ErosionFluxStageUniforms, ErosionInitStageUniforms,
    ErosionTerrainStageUniforms, ErosionThermalStageUniforms, ErosionWaterStageUniforms,
    ExtractedForestView, ForestGlobals, ForestShaderHandles,
};
use bevy::prelude::*;
use bevy::render::render_resource::{
    BindGroup, BindGroupLayoutDescriptor, CachedRenderPipelineId, FragmentState, PipelineCache,
    RenderPassColorAttachment, RenderPassDescriptor, RenderPipeline, RenderPipelineDescriptor,
    VertexState,
};
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue};
use bevy::render::RenderSystems;
use bytemuck::Zeroable;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// `RunErosionIterations`: the simulation steps a fixed `0.055` s, not the
/// frame time, so iteration counts stay comparable across frame rates.
const EROSION_DELTA_TIME: f32 = 0.055;
/// `uGravity` in erosion-flux.fs.
const EROSION_GRAVITY: f32 = 9.81;
/// `uMinimumSlope` in erosion-terrain.fs: below this gradient the shear term
/// is left alone so flat beds do not dissolve.
const EROSION_MINIMUM_SLOPE: f32 = 0.015;
/// `uGuardBandPixels`: the sim shaders leave the outer 16 texels of the
/// retained footprint to their neighbours.
const EROSION_GUARD_BAND_PIXELS: f32 = 16.0;
/// `uBrushStrength` in erosion-terrain.fs.
const EROSION_BRUSH_STRENGTH: f32 = 0.35;

/// One RGBA32F texel of the simulation domain.
const SIM_TEXEL_BYTES: u32 = 16;
/// 480 * 16 = 7680, already 256-byte aligned, so the reads need no padding.
const SIM_ROW_BYTES: u32 = SIM_TEXTURE_SIZE * SIM_TEXEL_BYTES;
/// One full readback plane (terrain, water or drainage): 480 * 480 * 16 bytes.
const SIM_READBACK_BYTES: u64 = SIM_ROW_BYTES as u64 * SIM_TEXTURE_SIZE as u64;
const SIM_PIXELS: usize = EROSION_RESOLUTION * EROSION_RESOLUTION;
/// The lookup texture is one RGBA32F texel per lattice cell.
const LOOKUP_TEXTURE_SIZE: u32 = EROSION_LOOKUP_DIAMETER as u32;

/// A finished atlas patch is dropped once its tile has stayed out of the
/// lookup window for this many frames (the tile was evicted before its slot
/// was ever published), which bounds the pending list.
const MAX_PENDING_ATLAS_AGE: u32 = 600;

// ---------------------------------------------------------------------------
// Render-world state
// ---------------------------------------------------------------------------

/// Render-world state for the erosion simulation.
///
/// Interior mutability is required because a render pass only ever gets
/// `&World` while its schedule is executing; everything the pass touches lives
/// behind this mutex.
#[derive(Resource)]
pub struct ErosionSimState {
    /// Pipelines, bind groups, the scratch textures and the readback pool,
    /// built by `prepare_erosion_sim` once the shared textures and the globals
    /// buffer exist.
    sim: Mutex<Option<ErosionSim>>,
}

impl Default for ErosionSimState {
    fn default() -> Self {
        Self { sim: Mutex::new(None) }
    }
}

/// One frame of the erosion simulation. Consumes the command list the main
/// world left in `ErosionBridge`, runs the requested iterations against the
/// GPU world textures, and reads the results back.
pub fn forest_erosion_pass(world: &World, mut ctx: RenderContext) {
    let Some(state) = world.get_resource::<ErosionSimState>() else {
        return;
    };
    let Some(texture_option) = world.get_resource::<GpuWorldTexturesOption>() else {
        return;
    };
    let Some(textures) = texture_option.0.as_deref() else {
        return;
    };
    let Some(queue) = world.get_resource::<RenderQueue>() else {
        return;
    };
    let Some(device) = world.get_resource::<RenderDevice>() else {
        return;
    };
    let Some(bridge) = world.get_resource::<ErosionBridge>() else {
        return;
    };
    let Some(pipeline_cache) = world.get_resource::<PipelineCache>() else {
        return;
    };
    let Ok(mut guard) = state.sim.lock() else {
        return;
    };
    let Some(sim) = guard.as_mut() else {
        return;
    };

    sim.frame += 1;

    // The C++ refreshes the 11x11 lookup from `UpdateErosionLookup` at the
    // end of every cache update; the bridge carries those records across,
    // and they also carry the atlas slots the pending patches wait for.
    let records = bridge.take_lookup();
    if let Some(records) = records.as_deref() {
        if records.len() == EROSION_LOOKUP_DIAMETER * EROSION_LOOKUP_DIAMETER * 4 {
            textures.revision.note_lookup(records);
            write_padded_at(
                queue,
                &textures.lookup_texture,
                &f32_to_bytes(records),
                LOOKUP_TEXTURE_SIZE,
                LOOKUP_TEXTURE_SIZE,
                SIM_TEXEL_BYTES,
                [0, 0],
                0,
            );
        }
    }

    // Start the maps whose copy commands were submitted on an earlier
    // frame, then give the device a chance to run their callbacks.
    sim.begin_mapping();
    if sim.in_flight.load(Ordering::SeqCst) > 0 {
        let _ = device.poll(wgpu::PollType::Poll);
    }

    // Consume whatever landed since the last frame. The C++ reads its
    // pixels inside FinalizeErosionTile, at the end of the finalize frame;
    // the port consumes the previous frame's readback here, before this
    // frame's commands, so a finished tile is applied as early as
    // possible.
    let outcome = sim.consume_readback();
    let consumed = !matches!(outcome, ReadbackOutcome::Idle);
    match outcome {
        ReadbackOutcome::Finalized(tile) => {
            bridge.push_event(ErosionEvent::TileFinalized(tile));
        }
        ReadbackOutcome::Failed(key) => bridge.push_event(ErosionEvent::ReadbackFailed(key)),
        ReadbackOutcome::Idle => {}
    }

    // Atlas patches whose slot is now published.
    if let (Some(records), Some(view)) = (
        records.as_deref(),
        world.get_resource::<ExtractedForestView>(),
    ) {
        sim.flush_pending_atlas(queue, textures, view.lookup_minimum, records);
    }

    let Some(commands) = bridge.take_commands() else {
        return;
    };

    // The pipelines compile asynchronously; hold the frame (and any
    // `init` it carries) until they are ready instead of dropping it.
    let Some(pipelines) = sim.pipelines(pipeline_cache) else {
        sim.defer_commands(commands);
        return;
    };

    let mut skip_readback = consumed;
    if let Some(deferred) = sim.deferred.take() {
        skip_readback |= sim.apply_commands(
            &deferred,
            &pipelines,
            &mut ctx,
            device,
            queue,
            textures,
            skip_readback,
        );
    }
    sim.apply_commands(
        &commands,
        &pipelines,
        &mut ctx,
        device,
        queue,
        textures,
        skip_readback,
    );

}

// ---------------------------------------------------------------------------
// Readback bookkeeping
// ---------------------------------------------------------------------------

/// Which plane a mapping callback delivers.
#[derive(Clone, Copy)]
enum ReadbackHalf {
    Terrain,
    Water,
    Drainage,
}

/// One half of a readback, as reported by its map callback.
enum HalfResult {
    Ready(Vec<f32>),
    Failed,
}

/// The halves of the readback currently in flight, plus the tile they belong
/// to. The tile's key is set when the copy is recorded and cleared when the
/// pair is consumed, so a late half can never be mistaken for a fresh one.
#[derive(Default)]
struct ReadbackSlots {
    key: Option<TileKey>,
    terrain: Option<HalfResult>,
    water: Option<HalfResult>,
    drainage: Option<HalfResult>,
}

/// A recorded copy whose map has not been started yet: `map_async` only waits
/// for the submissions that exist when it is called, so it must not be issued
/// in the frame that records the copy.
struct StagedReadback {
    terrain: wgpu::Buffer,
    water: wgpu::Buffer,
    drainage: wgpu::Buffer,
    /// Value of `ErosionSim::frame` when the copy was recorded.
    issued_frame: u64,
}

/// What `ErosionSim::consume_readback` found this frame.
enum ReadbackOutcome {
    Idle,
    Finalized(FinalizedTile),
    Failed(TileKey),
}

/// A finished atlas patch waiting for the main world to publish its slot.
struct PendingAtlasPatch {
    key: TileKey,
    atlas_height: Vec<f32>,
    atlas_flow: Vec<f32>,
    /// Frames spent waiting; patches are dropped past `MAX_PENDING_ATLAS_AGE`.
    age: u32,
}

// ---------------------------------------------------------------------------
// Simulation state
// ---------------------------------------------------------------------------

/// The three simulation pipelines plus the init pipeline, borrowed from the
/// pipeline cache for the duration of one frame.
struct ErosionPipelines<'a> {
    init: &'a RenderPipeline,
    flux: &'a RenderPipeline,
    water: &'a RenderPipeline,
    terrain: &'a RenderPipeline,
    thermal: &'a RenderPipeline,
}

/// Every GPU object the erosion node owns, mirroring the C++ `HydraulicErosion`
/// struct's textures, targets, shaders and ping-pong indices.
struct ErosionSim {
    init_pipeline: CachedRenderPipelineId,
    flux_pipeline: CachedRenderPipelineId,
    water_pipeline: CachedRenderPipelineId,
    terrain_pipeline: CachedRenderPipelineId,
    thermal_pipeline: CachedRenderPipelineId,

    /// group(0): the shared per-frame globals buffer.
    globals: BindGroup,

    /// `uMode` 0, 1 and 2 init uniforms (see the module PORT NOTES).
    init_stage_buffers: [wgpu::Buffer; 3],
    init_stage_groups: [BindGroup; 3],
    flux_stage_buffer: wgpu::Buffer,
    flux_stage_group: BindGroup,
    water_stage_buffer: wgpu::Buffer,
    water_stage_group: BindGroup,
    terrain_stage_buffer: wgpu::Buffer,
    terrain_stage_group: BindGroup,
    thermal_stage_buffer: wgpu::Buffer,
    thermal_stage_group: BindGroup,

    /// group(1) inputs, pre-built for every ping-pong combination because
    /// bind groups are immutable.
    init_inputs: BindGroup,
    flux_inputs: [[[BindGroup; 2]; 2]; 2],
    water_inputs: [[[BindGroup; 2]; 2]; 2],
    terrain_inputs: [[[BindGroup; 2]; 2]; 2],
    thermal_inputs: [[BindGroup; 2]; 2],

    /// The short-lived base height map the init passes sample, one Rgba32Float
    /// 480x480 texture reused for every tile (the C++'s `base` Texture2D).
    base_texture: wgpu::Texture,

    terrain_views: [wgpu::TextureView; 2],
    water_views: [wgpu::TextureView; 2],
    flux_views: [wgpu::TextureView; 2],
    drainage_views: [wgpu::TextureView; 2],
    /// Written by every water pass and read by the terrain pass right after,
    /// so it needs no ping-pong partner.
    routing_view: wgpu::TextureView,

    /// `terrainIndex`, `waterIndex`, `fluxIndex`, plus the drainage state.
    terrain_index: usize,
    water_index: usize,
    flux_index: usize,
    drainage_index: usize,

    /// `activeTile` and how many of `settings.iterations` it has run.
    active_tile: Option<TileKey>,
    active_iterations: usize,
    /// Erosion settings in force (`appliedErosionSettings`).
    settings: ErosionSettings,

    /// A tile whose result has already been published; the bridge keeps
    /// sending `finalize` until the main world sees the event, and a second
    /// readback of the same state would be pure waste.
    delivered: Option<TileKey>,

    /// Frames held back while the pipelines compiled.
    deferred: Option<ErosionFrameCommands>,

    pending_atlas: Vec<PendingAtlasPatch>,

    /// Readback machinery, shared with the map callbacks (`Arc` because the
    /// callbacks outlive this frame).
    readback: Arc<Mutex<ReadbackSlots>>,
    pool: Arc<Mutex<Vec<wgpu::Buffer>>>,
    in_flight: Arc<AtomicUsize>,
    staged: Option<StagedReadback>,
    /// Frames seen by `run`, used to delay the maps by one frame.
    frame: u64,
}

impl ErosionSim {
    /// Creates every pipeline, bind group and scratch texture. The shared
    /// world textures and the globals buffer must already exist.
    fn build(
        device: &RenderDevice,
        pipeline_cache: &PipelineCache,
        handles: &ForestShaderHandles,
        textures: &GpuWorldTextures,
        globals_buffer: &wgpu::Buffer,
    ) -> Self {
        // group(0) is the frame's shared globals. It is created here rather
        // than through `globals_bind_group` so the bind group is bevy's
        // wrapper, which is what `TrackedRenderPass::set_bind_group` takes.
        let globals_layout = globals_layout();
        let globals = super::bind_group(
            device,
            pipeline_cache,
            "erosion_globals",
            &globals_layout,
            &globals_bind_group_entries(globals_buffer),
        );

        let stage_layout = stage_uniform_layout();
        let init_inputs_layout = sim_inputs_layout(1);
        let flux_inputs_layout = sim_inputs_layout(3);
        let water_inputs_layout = sim_inputs_layout(3);
        let terrain_inputs_layout = sim_inputs_layout(4);
        let thermal_inputs_layout = sim_inputs_layout(2);

        // The C++ sets TEXTURE_FILTER_POINT on the base map and keeps every
        // simulation texture point-sampled; the sim shaders only ever
        // `textureLoad`, so the samplers they declare (bindings 8..) are
        // present for the layout but never filter.
        let sampler = make_sim_sampler(device);

        let terrain_views = [
            textures.sim_terrain[0].create_view(&Default::default()),
            textures.sim_terrain[1].create_view(&Default::default()),
        ];
        let water_views = [
            textures.sim_water[0].create_view(&Default::default()),
            textures.sim_water[1].create_view(&Default::default()),
        ];
        let flux_views = [
            textures.sim_flux[0].create_view(&Default::default()),
            textures.sim_flux[1].create_view(&Default::default()),
        ];
        let drainage_views = [
            textures.sim_drainage[0].create_view(&Default::default()),
            textures.sim_drainage[1].create_view(&Default::default()),
        ];
        let routing_view = textures.sim_routing.create_view(&Default::default());

        let base_texture = device.wgpu_device().create_texture(&wgpu::TextureDescriptor {
            label: Some("erosion_base_height"),
            size: wgpu::Extent3d {
                width: SIM_TEXTURE_SIZE,
                height: SIM_TEXTURE_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let base_view = base_texture.create_view(&Default::default());
        let init_inputs = target_bind_group(
            device,
            pipeline_cache,
            "erosion_init_inputs",
            &init_inputs_layout,
            &[&base_view],
            &sampler,
        );

        // `RunErosionIterations` feeds each pass the state it must read: flux
        // takes (flux, terrain, water), water takes (water, flux, terrain) —
        // after the flux flip — terrain takes (terrain, water, drainage) and
        // thermal takes the terrain and drainage the terrain pass just wrote
        // (drainage B carries each cell's bedrock resistance).
        let flux_inputs = std::array::from_fn(|flux_index| {
            std::array::from_fn(|terrain_index| {
                std::array::from_fn(|water_index| {
                    target_bind_group(
                        device,
                        pipeline_cache,
                        "erosion_flux_inputs",
                        &flux_inputs_layout,
                        &[
                            &flux_views[flux_index],
                            &terrain_views[terrain_index],
                            &water_views[water_index],
                        ],
                        &sampler,
                    )
                })
            })
        });
        let water_inputs = std::array::from_fn(|water_index| {
            std::array::from_fn(|flux_index| {
                std::array::from_fn(|terrain_index| {
                    target_bind_group(
                        device,
                        pipeline_cache,
                        "erosion_water_inputs",
                        &water_inputs_layout,
                        &[
                            &water_views[water_index],
                            &flux_views[flux_index],
                            &terrain_views[terrain_index],
                        ],
                        &sampler,
                    )
                })
            })
        });
        let terrain_inputs = std::array::from_fn(|terrain_index| {
            std::array::from_fn(|water_index| {
                std::array::from_fn(|drainage_index| {
                    target_bind_group(
                        device,
                        pipeline_cache,
                        "erosion_terrain_inputs",
                        &terrain_inputs_layout,
                        &[
                            &terrain_views[terrain_index],
                            &water_views[water_index],
                            &drainage_views[drainage_index],
                            &routing_view,
                        ],
                        &sampler,
                    )
                })
            })
        });
        let thermal_inputs = std::array::from_fn(|terrain_index| {
            std::array::from_fn(|drainage_index| {
                target_bind_group(
                    device,
                    pipeline_cache,
                    "erosion_thermal_inputs",
                    &thermal_inputs_layout,
                    &[&terrain_views[terrain_index], &drainage_views[drainage_index]],
                    &sampler,
                )
            })
        });

        // One init uniform per mode: `queue.write_buffer` is ordered before
        // the whole submission, so one buffer cannot hold `uMode` 0, 1 and 2
        // within the same frame (see the module PORT NOTES).
        let init_stage_buffers = [
            stage_buffer(device, "erosion_init_mode_0"),
            stage_buffer(device, "erosion_init_mode_1"),
            stage_buffer(device, "erosion_init_mode_2"),
        ];
        let init_stage_groups = [
            stage_bind_group(
                device,
                pipeline_cache,
                &stage_layout,
                "erosion_init_stage_0",
                &init_stage_buffers[0],
            ),
            stage_bind_group(
                device,
                pipeline_cache,
                &stage_layout,
                "erosion_init_stage_1",
                &init_stage_buffers[1],
            ),
            stage_bind_group(
                device,
                pipeline_cache,
                &stage_layout,
                "erosion_init_stage_2",
                &init_stage_buffers[2],
            ),
        ];
        let flux_stage_buffer = stage_buffer(device, "erosion_flux_stage");
        let water_stage_buffer = stage_buffer(device, "erosion_water_stage");
        let terrain_stage_buffer = stage_buffer(device, "erosion_terrain_stage");
        let thermal_stage_buffer = stage_buffer(device, "erosion_thermal_stage");
        let flux_stage_group = stage_bind_group(
            device,
            pipeline_cache,
            &stage_layout,
            "erosion_flux_stage_group",
            &flux_stage_buffer,
        );
        let water_stage_group = stage_bind_group(
            device,
            pipeline_cache,
            &stage_layout,
            "erosion_water_stage_group",
            &water_stage_buffer,
        );
        let terrain_stage_group = stage_bind_group(
            device,
            pipeline_cache,
            &stage_layout,
            "erosion_terrain_stage_group",
            &terrain_stage_buffer,
        );
        let thermal_stage_group = stage_bind_group(
            device,
            pipeline_cache,
            &stage_layout,
            "erosion_thermal_stage_group",
            &thermal_stage_buffer,
        );

        // Layouts: globals, inputs, stage uniforms. The sim shaders take no
        // vertex buffers; all but the terrain pass write one colour target.
        let init_pipeline = queue_erosion_pipeline(
            pipeline_cache,
            "erosion_init_pipeline",
            handles.erosion_init.clone(),
            vec![globals_layout.clone(), init_inputs_layout, stage_layout.clone()],
            1,
        );
        let flux_pipeline = queue_erosion_pipeline(
            pipeline_cache,
            "erosion_flux_pipeline",
            handles.erosion_flux.clone(),
            vec![globals_layout.clone(), flux_inputs_layout, stage_layout.clone()],
            1,
        );
        // Water and the routing normaliser.
        let water_pipeline = queue_erosion_pipeline(
            pipeline_cache,
            "erosion_water_pipeline",
            handles.erosion_water.clone(),
            vec![globals_layout.clone(), water_inputs_layout, stage_layout.clone()],
            2,
        );
        // Terrain and drainage states.
        let terrain_pipeline = queue_erosion_pipeline(
            pipeline_cache,
            "erosion_terrain_pipeline",
            handles.erosion_terrain.clone(),
            vec![globals_layout.clone(), terrain_inputs_layout, stage_layout.clone()],
            2,
        );
        let thermal_pipeline = queue_erosion_pipeline(
            pipeline_cache,
            "erosion_thermal_pipeline",
            handles.erosion_thermal.clone(),
            vec![globals_layout, thermal_inputs_layout, stage_layout.clone()],
            1,
        );

        Self {
            init_pipeline,
            flux_pipeline,
            water_pipeline,
            terrain_pipeline,
            thermal_pipeline,
            globals,
            init_stage_buffers,
            init_stage_groups,
            flux_stage_buffer,
            flux_stage_group,
            water_stage_buffer,
            water_stage_group,
            terrain_stage_buffer,
            terrain_stage_group,
            thermal_stage_buffer,
            thermal_stage_group,
            init_inputs,
            flux_inputs,
            water_inputs,
            terrain_inputs,
            thermal_inputs,
            base_texture,
            terrain_views,
            water_views,
            flux_views,
            drainage_views,
            routing_view,
            terrain_index: 0,
            water_index: 0,
            flux_index: 0,
            drainage_index: 0,
            active_tile: None,
            active_iterations: 0,
            settings: ErosionSettings::default(),
            delivered: None,
            deferred: None,
            pending_atlas: Vec::new(),
            readback: Arc::new(Mutex::new(ReadbackSlots::default())),
            pool: Arc::new(Mutex::new(Vec::new())),
            in_flight: Arc::new(AtomicUsize::new(0)),
            staged: None,
            frame: 0,
        }
    }

    fn pipelines<'a>(&self, cache: &'a PipelineCache) -> Option<ErosionPipelines<'a>> {
        Some(ErosionPipelines {
            init: cache.get_render_pipeline(self.init_pipeline)?,
            flux: cache.get_render_pipeline(self.flux_pipeline)?,
            water: cache.get_render_pipeline(self.water_pipeline)?,
            terrain: cache.get_render_pipeline(self.terrain_pipeline)?,
            thermal: cache.get_render_pipeline(self.thermal_pipeline)?,
        })
    }

    /// Holds a frame that arrived before the pipelines compiled. A stashed
    /// `init` is never dropped in favour of the newer set, since the main
    /// world only sends one per tile.
    fn defer_commands(&mut self, mut commands: ErosionFrameCommands) {
        let stashed_init = self.deferred.take().and_then(|stashed| stashed.init);
        if commands.init.is_none() {
            commands.init = stashed_init;
        }
        self.deferred = Some(commands);
    }

    /// `uMode` 0, 1 and 2 init uniforms, the base height and drainage upload
    /// and the eight init passes (`InitializeErosionTextures`).
    #[allow(clippy::too_many_arguments)]
    fn initialize_tile(
        &mut self,
        init_key: TileKey,
        base_height: &[f32],
        drainage_area: &[f32],
        river: &[f32],
        sim_min: [f32; 2],
        queue: &RenderQueue,
        context: &mut RenderContext<'_, '_>,
        pipelines: &ErosionPipelines<'_>,
    ) {
        if base_height.len() == SIM_PIXELS {
            // The C++ uploads R32; the port uploads Rgba32Float because R32Float
            // is not filterable on every backend, and carries the CPU-routed
            // drainage area in G for the `uMode` 2 stamp, and the river
            // cells in B. One row is 480 * 16 = 7680 bytes, already 256-byte
            // aligned.
            let mut pixels = Vec::with_capacity(SIM_PIXELS * 4);
            for (index, height) in base_height.iter().enumerate() {
                let area = drainage_area.get(index).copied().unwrap_or(1.0);
                let river = river.get(index).copied().unwrap_or(0.0);
                pixels.extend_from_slice(&[*height, area, river, 1.0]);
            }
            queue.write_texture(
                self.base_texture.as_image_copy(),
                bytemuck::cast_slice(&pixels),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(SIM_ROW_BYTES),
                    rows_per_image: Some(SIM_TEXTURE_SIZE),
                },
                wgpu::Extent3d {
                    width: SIM_TEXTURE_SIZE,
                    height: SIM_TEXTURE_SIZE,
                    depth_or_array_layers: 1,
                },
            );
        } else {
            log::warn!(
                "EROSION: tile ({}, {}) base height map has {} texels, expected {}",
                init_key.x,
                init_key.z,
                base_height.len(),
                SIM_PIXELS
            );
        }

        let mut uniforms = ErosionInitStageUniforms::zeroed();
        uniforms.mode = 0;
        uniforms.world_min = sim_min;
        uniforms.cell_size = EROSION_CELL_SIZE;
        queue.write_buffer(&self.init_stage_buffers[0], 0, bytemuck::bytes_of(&uniforms));
        uniforms.mode = 1;
        queue.write_buffer(&self.init_stage_buffers[1], 0, bytemuck::bytes_of(&uniforms));
        uniforms.mode = 2;
        queue.write_buffer(&self.init_stage_buffers[2], 0, bytemuck::bytes_of(&uniforms));

        // for (int i = 0; i < 2; ++i): terrain keeps the stamped state, water
        // and flux are cleared, drainage takes the routed base catchments,
        // and both scratch copies start identical.
        for index in 0..2 {
            record_erosion_pass(
                context,
                pipelines.init,
                &self.globals,
                &self.init_inputs,
                &self.init_stage_groups[0],
                &self.terrain_views[index],
            );
            record_erosion_pass(
                context,
                pipelines.init,
                &self.globals,
                &self.init_inputs,
                &self.init_stage_groups[1],
                &self.water_views[index],
            );
            record_erosion_pass(
                context,
                pipelines.init,
                &self.globals,
                &self.init_inputs,
                &self.init_stage_groups[1],
                &self.flux_views[index],
            );
            record_erosion_pass(
                context,
                pipelines.init,
                &self.globals,
                &self.init_inputs,
                &self.init_stage_groups[2],
                &self.drainage_views[index],
            );
        }

        self.terrain_index = 0;
        self.water_index = 0;
        self.flux_index = 0;
        self.drainage_index = 0;
        self.active_tile = Some(init_key);
        self.active_iterations = 0;
        self.delivered = None;
    }

    /// The per-frame stage uniforms. `RunErosionIterations` re-sets the same
    /// values before every pass; they are frame-constant here, so they are
    /// written once per command set.
    fn write_stage_uniforms(&self, queue: &RenderQueue, sim_min: [f32; 2]) {
        let resolution = [SIM_TEXTURE_SIZE as f32, SIM_TEXTURE_SIZE as f32];

        let mut flux = ErosionFluxStageUniforms::zeroed();
        flux.resolution = resolution;
        flux.delta_time = EROSION_DELTA_TIME;
        flux.cell_size = EROSION_CELL_SIZE;
        flux.gravity = EROSION_GRAVITY;
        queue.write_buffer(&self.flux_stage_buffer, 0, bytemuck::bytes_of(&flux));

        let mut water = ErosionWaterStageUniforms::zeroed();
        water.resolution = resolution;
        water.delta_time = EROSION_DELTA_TIME;
        water.cell_size = EROSION_CELL_SIZE;
        water.rain = self.settings.rain;
        water.evaporation = self.settings.evaporation;
        water.sea_level = SEA_LEVEL;
        queue.write_buffer(&self.water_stage_buffer, 0, bytemuck::bytes_of(&water));

        let mut terrain = ErosionTerrainStageUniforms::zeroed();
        terrain.resolution = resolution;
        terrain.delta_time = EROSION_DELTA_TIME;
        terrain.cell_size = EROSION_CELL_SIZE;
        terrain.sea_level = SEA_LEVEL;
        terrain.erosion_rate = self.settings.erosion_rate;
        terrain.deposition_rate = self.settings.deposition_rate;
        terrain.sediment_capacity = self.settings.sediment_capacity;
        terrain.minimum_slope = EROSION_MINIMUM_SLOPE;
        terrain.maximum_erosion = self.settings.maximum_erosion;
        terrain.transport_rate = self.settings.transport_rate;
        terrain.guard_band_pixels = EROSION_GUARD_BAND_PIXELS;
        terrain.brush_strength = EROSION_BRUSH_STRENGTH;
        terrain.world_min = sim_min;
        terrain.fluvial_capacity = self.settings.fluvial_capacity;
        terrain.fluvial_erosion = self.settings.fluvial_erosion;
        terrain.fluvial_deposition = self.settings.fluvial_deposition;
        terrain.maximum_incision = self.settings.maximum_incision;
        terrain.drainage_saturation = EROSION_DRAINAGE_SATURATION;
        queue.write_buffer(&self.terrain_stage_buffer, 0, bytemuck::bytes_of(&terrain));

        let mut thermal = ErosionThermalStageUniforms::zeroed();
        thermal.resolution = resolution;
        thermal.cell_size = EROSION_CELL_SIZE;
        thermal.sea_level = SEA_LEVEL;
        thermal.world_min = sim_min;
        thermal.loose_rate = self.settings.talus_rate;
        thermal.rock_rate = self.settings.rockfall_rate;
        thermal.loose_repose = EROSION_LOOSE_REPOSE;
        thermal.soft_rock_slope = EROSION_SOFT_ROCK_SLOPE;
        thermal.hard_rock_slope = EROSION_HARD_ROCK_SLOPE;
        queue.write_buffer(&self.thermal_stage_buffer, 0, bytemuck::bytes_of(&thermal));
    }

    /// `RunErosionIterations`: flux, water, terrain (with drainage) and
    /// thermal into the opposite targets, `count` times.
    fn run_iterations(
        &mut self,
        count: usize,
        context: &mut RenderContext<'_, '_>,
        pipelines: &ErosionPipelines<'_>,
    ) {
        for _ in 0..count {
            let next_flux = 1 - self.flux_index;
            record_erosion_pass(
                context,
                pipelines.flux,
                &self.globals,
                &self.flux_inputs[self.flux_index][self.terrain_index][self.water_index],
                &self.flux_stage_group,
                &self.flux_views[next_flux],
            );
            self.flux_index = next_flux;

            let next_water = 1 - self.water_index;
            record_erosion_pass_targets(
                context,
                pipelines.water,
                &self.globals,
                &self.water_inputs[self.water_index][self.flux_index][self.terrain_index],
                &self.water_stage_group,
                &[&self.water_views[next_water], &self.routing_view],
            );
            self.water_index = next_water;

            let next_terrain = 1 - self.terrain_index;
            let next_drainage = 1 - self.drainage_index;
            record_erosion_pass_targets(
                context,
                pipelines.terrain,
                &self.globals,
                &self.terrain_inputs[self.terrain_index][self.water_index][self.drainage_index],
                &self.terrain_stage_group,
                &[&self.terrain_views[next_terrain], &self.drainage_views[next_drainage]],
            );
            self.terrain_index = next_terrain;
            self.drainage_index = next_drainage;

            let next_terrain = 1 - self.terrain_index;
            record_erosion_pass(
                context,
                pipelines.thermal,
                &self.globals,
                &self.thermal_inputs[self.terrain_index][self.drainage_index],
                &self.thermal_stage_group,
                &self.terrain_views[next_terrain],
            );
            self.terrain_index = next_terrain;
        }
    }

    /// The C++'s per-frame iteration budget (`UpdateErosionCache`'s
    /// `min(iterationBudget, remaining)`).
    ///
    /// The bridge carries an explicit `IterateCommand` when the caller fills
    /// it in; until then the C++ budgets are reconstructed: the frame that
    /// initializes and finalizes the same tile is the pre-loop prewarm
    /// (`UpdateErosionCache(..., settings.iterations, true)`), and every other
    /// frame advances `kErosionIterationsPerFrame`.
    fn iteration_count(&self, commands: &ErosionFrameCommands) -> usize {
        let total = self.settings.iterations;
        let remaining = total.saturating_sub(self.active_iterations);
        if let Some(iterate) = &commands.iterate {
            if iterate.count > 0 {
                // Deliberately NOT clamped to `remaining`. An explicit command
                // is only ever filled in by the --measure-overlap driver, which
                // mirrors `RunOverlapMeasurement`'s inner loop: the C++
                // (main.cpp:1606-1614) calls `RunErosionIterations(6)` every
                // pass and only compares `completedIterations` afterwards, so
                // the last batch overshoots and 140 iterations at six per frame
                // end on 144. The `.min(remaining)` this used to apply ran the
                // last batch as two steps instead of six, so the tile was
                // simulated 140 times while its recorded
                // `completed_iterations` said 144 - the diagnostic measured a
                // differently-converged simulation than the one it reported.
                // The streaming path leaves `iterate` as None and keeps the
                // clamp below, which is `UpdateErosionCache`'s
                // `min(iterationBudget, remaining)` (main.cpp:1556-1559).
                return iterate.count;
            }
        }
        let budget = if commands.init.is_some() && commands.finalize {
            total
        } else {
            EROSION_ITERATIONS_PER_FRAME
        };
        budget.min(remaining)
    }

    /// Applies one command set: init, iterations, and the readback that
    /// `FinalizeErosionTile` performs at the end of the frame.
    ///
    /// Returns whether a readback copy was recorded.
    #[allow(clippy::too_many_arguments)]
    fn apply_commands(
        &mut self,
        commands: &ErosionFrameCommands,
        pipelines: &ErosionPipelines<'_>,
        context: &mut RenderContext<'_, '_>,
        device: &RenderDevice,
        queue: &RenderQueue,
        textures: &GpuWorldTextures,
        skip_readback: bool,
    ) -> bool {
        if let Some(iterate) = &commands.iterate {
            self.settings = iterate.settings;
        }

        // `UpdateErosionCache` returns immediately while no tile is active;
        // a frame with neither an active tile nor a fresh init does the same.
        if self.active_tile.is_none() && commands.init.is_none() {
            return false;
        }

        self.write_stage_uniforms(queue, commands.sim_min);

        if let Some(init) = &commands.init {
            self.initialize_tile(
                init.key,
                &init.base_height,
                &init.drainage_area,
                &init.river,
                commands.sim_min,
                queue,
                context,
                pipelines,
            );
        }

        let count = self.iteration_count(commands);
        if count > 0 {
            self.run_iterations(count, context, pipelines);
            self.active_iterations += count;
        }

        if !commands.finalize || skip_readback {
            return false;
        }
        let Some(key) = self.active_tile else {
            return false;
        };
        if self.delivered == Some(key) {
            return false;
        }
        self.issue_readback(context.command_encoder(), device, textures, key)
    }

    /// Records `rlReadTexturePixels`'s two copies into this frame's encoder.
    fn issue_readback(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        device: &RenderDevice,
        textures: &GpuWorldTextures,
        key: TileKey,
    ) -> bool {
        if self.staged.is_some() || self.in_flight.load(Ordering::SeqCst) != 0 {
            return false;
        }
        {
            let Ok(mut slots) = self.readback.lock() else {
                return false;
            };
            // A plane can arrive after its set failed, with no key: drop it,
            // or it would block every later readback.
            if slots.key.is_none() {
                slots.terrain = None;
                slots.water = None;
                slots.drainage = None;
            }
            if slots.key.is_some()
                || slots.terrain.is_some()
                || slots.water.is_some()
                || slots.drainage.is_some()
            {
                return false;
            }
            slots.key = Some(key);
        }

        let terrain_buffer = take_staging_buffer(device, &self.pool);
        let water_buffer = take_staging_buffer(device, &self.pool);
        let drainage_buffer = take_staging_buffer(device, &self.pool);
        let layout = wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(SIM_ROW_BYTES),
            rows_per_image: Some(SIM_TEXTURE_SIZE),
        };
        let extent = wgpu::Extent3d {
            width: SIM_TEXTURE_SIZE,
            height: SIM_TEXTURE_SIZE,
            depth_or_array_layers: 1,
        };
        encoder.copy_texture_to_buffer(
            textures.sim_terrain[self.terrain_index].as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &terrain_buffer,
                layout,
            },
            extent,
        );
        encoder.copy_texture_to_buffer(
            textures.sim_water[self.water_index].as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &water_buffer,
                layout,
            },
            extent,
        );
        encoder.copy_texture_to_buffer(
            textures.sim_drainage[self.drainage_index].as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &drainage_buffer,
                layout,
            },
            extent,
        );

        self.staged = Some(StagedReadback {
            terrain: terrain_buffer,
            water: water_buffer,
            drainage: drainage_buffer,
            issued_frame: self.frame,
        });
        true
    }

    /// Issues the maps for the staged copies once their submission exists.
    fn begin_mapping(&mut self) {
        match &self.staged {
            Some(staged) if staged.issued_frame < self.frame => {}
            _ => return,
        }
        let Some(staged) = self.staged.take() else {
            return;
        };
        // One per plane: each `map_readback` callback subtracts one, and the
        // count has to land back on exactly zero or `issue_readback` (which
        // refuses to start while a readback is outstanding) never runs again.
        self.in_flight.fetch_add(3, Ordering::SeqCst);
        map_readback(
            staged.terrain,
            self.readback.clone(),
            self.pool.clone(),
            self.in_flight.clone(),
            ReadbackHalf::Terrain,
        );
        map_readback(
            staged.water,
            self.readback.clone(),
            self.pool.clone(),
            self.in_flight.clone(),
            ReadbackHalf::Water,
        );
        map_readback(
            staged.drainage,
            self.readback.clone(),
            self.pool.clone(),
            self.in_flight.clone(),
            ReadbackHalf::Drainage,
        );
    }

    /// Consumes a completed readback and runs the finalize half of
    /// `FinalizeErosionTile` on the render thread.
    fn consume_readback(&mut self) -> ReadbackOutcome {
        let Ok(mut slots) = self.readback.lock() else {
            return ReadbackOutcome::Idle;
        };
        let Some(key) = slots.key else {
            return ReadbackOutcome::Idle;
        };
        let outcome = match (slots.terrain.take(), slots.water.take(), slots.drainage.take()) {
            (
                Some(HalfResult::Ready(terrain)),
                Some(HalfResult::Ready(water)),
                Some(HalfResult::Ready(drainage)),
            ) => Ok((terrain, water, drainage)),
            (Some(HalfResult::Failed), _, _)
            | (_, Some(HalfResult::Failed), _)
            | (_, _, Some(HalfResult::Failed)) => Err(()),
            (terrain, water, drainage) => {
                // A plane is still in flight; put back what arrived.
                slots.terrain = terrain;
                slots.water = water;
                slots.drainage = drainage;
                return ReadbackOutcome::Idle;
            }
        };
        slots.key = None;
        drop(slots);

        match outcome {
            Ok((terrain, water, drainage))
                if terrain.len() == SIM_PIXELS * 4
                    && water.len() == SIM_PIXELS * 4
                    && drainage.len() == SIM_PIXELS * 4 =>
            {
                let (tile, atlas_height, atlas_flow) =
                    finalize_erosion_tile(key, &terrain, &water, &drainage);
                self.delivered = Some(key);
                self.pending_atlas.push(PendingAtlasPatch {
                    key,
                    atlas_height,
                    atlas_flow,
                    age: 0,
                });
                ReadbackOutcome::Finalized(tile)
            }
            _ => {
                // A failed or short readback: the C++ counts it and requeues
                // the tile, which the main world does from the event.
                self.delivered = Some(key);
                ReadbackOutcome::Failed(key)
            }
        }
    }

    /// Uploads finished atlas patches once the lookup records reveal the slot
    /// the main world allocated for them (`UpdateErosionLookup` writes the
    /// same slot numbers, in the same 11x11 record layout).
    fn flush_pending_atlas(
        &mut self,
        queue: &RenderQueue,
        textures: &GpuWorldTextures,
        lookup_minimum: (i64, i64),
        records: &[f32],
    ) {
        if self.pending_atlas.is_empty() {
            return;
        }
        let diameter = EROSION_LOOKUP_DIAMETER as i64;
        let mut index = 0;
        while index < self.pending_atlas.len() {
            let patch = &self.pending_atlas[index];
            let relative_x = patch.key.x - lookup_minimum.0;
            let relative_z = patch.key.z - lookup_minimum.1;
            let slot = if relative_x >= 0
                && relative_z >= 0
                && relative_x < diameter
                && relative_z < diameter
            {
                let record = ((relative_z * diameter + relative_x) * 4) as usize;
                if records.get(record + 3).copied().unwrap_or(0.0) > 0.0 {
                    Some((records[record], records[record + 1]))
                } else {
                    None
                }
            } else {
                None
            };

            let Some((slot_x, slot_z)) = slot else {
                self.pending_atlas[index].age += 1;
                index += 1;
                continue;
            };

            let patch = self.pending_atlas.remove(index);
            let origin = [
                slot_x as u32 * EROSION_ATLAS_PITCH as u32,
                slot_z as u32 * EROSION_ATLAS_PITCH as u32,
            ];
            let pitch = EROSION_ATLAS_PITCH as u32;
            write_padded_at(
                queue,
                textures.height_atlas_view.texture(),
                &f32_to_bytes(&patch.atlas_height),
                pitch,
                pitch,
                SIM_TEXEL_BYTES,
                origin,
                0,
            );
            write_padded_at(
                queue,
                textures.flow_atlas_view.texture(),
                &f32_to_bytes(&patch.atlas_flow),
                pitch,
                pitch,
                SIM_TEXEL_BYTES,
                origin,
                0,
            );
            textures.revision.note_atlas_upload();
            log::info!(
                "EROSION: atlas patch ({}, {}) uploaded to slot ({}, {})",
                patch.key.x,
                patch.key.z,
                origin[0],
                origin[1]
            );
        }
        self.pending_atlas.retain(|patch| patch.age <= MAX_PENDING_ATLAS_AGE);
    }
}

// ---------------------------------------------------------------------------
// Finalize (CPU half of `FinalizeErosionTile`)
// ---------------------------------------------------------------------------

/// Crops the retained footprint out of the 480x480 readback and rebuilds the
/// C++'s atlas patch. Returns the tile's event payload (with a placeholder
/// slot, see the module PORT NOTES) plus the 260x260 height and flow patches.
fn finalize_erosion_tile(
    key: TileKey,
    terrain_rgba: &[f32],
    flow_rgba: &[f32],
    drainage_rgba: &[f32],
) -> (FinalizedTile, Vec<f32>, Vec<f32>) {
    let resolution = EROSION_RESOLUTION;
    let output_resolution = EROSION_OUTPUT_RESOLUTION;
    let output_pixels = output_resolution * output_resolution;

    let mut cpu_delta = vec![0.0f32; output_pixels];
    let mut output_height = vec![0.0f32; output_pixels * 4];
    let mut output_flow = vec![0.0f32; output_pixels * 4];

    // Preserve signed displacement in R for rendering and collision. The
    // remaining channels describe the final bed: G concavity (metres),
    // B drainage concentration (positive log2 ratio of contributing area
    // against the surrounding ring), A loose cover thickness (metres). The
    // flow patch keeps the solver's water depth and velocity and carries the
    // routed contributing area, in cells, in A.
    const MATERIAL_RING_OFFSET: usize = 3;
    const DRAINAGE_EPSILON: f32 = 1.0;

    let mut minimum = f32::INFINITY;
    let mut maximum = f32::NEG_INFINITY;
    let mut minimum_delta = f32::INFINITY;
    let mut maximum_delta = f32::NEG_INFINITY;
    let mut flow_axis_bias = 0.0f64;
    let mut moving_flow_cells = 0usize;
    let mut land_cells = 0usize;
    let mut maximum_drainage = 0.0f32;
    let mut bare_cells = 0usize;
    let mut loose_total = 0.0f64;
    for output_z in 0..output_resolution {
        for output_x in 0..output_resolution {
            let source_x = output_x + EROSION_OUTPUT_OFFSET;
            let source_z = output_z + EROSION_OUTPUT_OFFSET;
            let source = source_z * resolution + source_x;
            let height = terrain_rgba[source * 4];
            let base_height = terrain_rgba[source * 4 + 2];
            let delta = height - base_height;
            let destination = output_z * output_resolution + output_x;
            cpu_delta[destination] = delta;

            // Sample the simulation halo too, so the 12 m axial ring does not
            // flatten at the retained footprint's edges.
            let mut surrounding_height = 0.0f32;
            let mut surrounding_drainage = 0.0f32;
            for ring_z in -1i64..=1 {
                for ring_x in -1i64..=1 {
                    if ring_x == 0 && ring_z == 0 {
                        continue;
                    }
                    let neighbour_z =
                        (source_z as i64 + ring_z * MATERIAL_RING_OFFSET as i64) as usize;
                    let neighbour_x =
                        (source_x as i64 + ring_x * MATERIAL_RING_OFFSET as i64) as usize;
                    let neighbour = neighbour_z * resolution + neighbour_x;
                    surrounding_height += terrain_rgba[neighbour * 4];
                    surrounding_drainage += drainage_rgba[neighbour * 4].max(0.0);
                }
            }
            surrounding_height *= 0.125;
            surrounding_drainage *= 0.125;
            let drainage_area = drainage_rgba[source * 4].max(0.0);
            output_height[destination * 4] = delta;
            output_height[destination * 4 + 1] = surrounding_height - height;
            output_height[destination * 4 + 2] = ((drainage_area + DRAINAGE_EPSILON)
                / (surrounding_drainage + DRAINAGE_EPSILON))
                .log2()
                .max(0.0);
            output_height[destination * 4 + 3] = terrain_rgba[source * 4 + 3].max(0.0);
            for channel in 0..3 {
                output_flow[destination * 4 + channel] = flow_rgba[source * 4 + channel];
            }
            output_flow[destination * 4 + 3] = drainage_area;
            minimum = minimum.min(height);
            maximum = maximum.max(height);
            minimum_delta = minimum_delta.min(delta);
            maximum_delta = maximum_delta.max(delta);
            if height > SEA_LEVEL {
                land_cells += 1;
                let loose = terrain_rgba[source * 4 + 3].max(0.0);
                loose_total += loose as f64;
                if loose < 0.1 {
                    bare_cells += 1;
                }
            }
            maximum_drainage = maximum_drainage.max(drainage_area);

            let velocity_x = flow_rgba[source * 4 + 1];
            let velocity_z = flow_rgba[source * 4 + 2];
            let velocity_sum = velocity_x.abs() + velocity_z.abs();
            if velocity_sum > 0.01 {
                flow_axis_bias += (velocity_x.abs().max(velocity_z.abs()) / velocity_sum) as f64;
                moving_flow_cells += 1;
            }
        }
    }

    // The detail metric only looks at the retained footprint.
    let mut detail = 0.0f64;
    let mut detail_samples = 0usize;
    for z in 1..output_resolution - 1 {
        for x in 1..output_resolution - 1 {
            let center_delta = cpu_delta[z * output_resolution + x];
            if center_delta < -0.05 {
                let neighbour_mean = 0.25
                    * (cpu_delta[z * output_resolution + x - 1]
                        + cpu_delta[z * output_resolution + x + 1]
                        + cpu_delta[(z - 1) * output_resolution + x]
                        + cpu_delta[(z + 1) * output_resolution + x]);
                detail += ((center_delta - neighbour_mean).abs()) as f64;
                detail_samples += 1;
            }
        }
    }

    // Upload this pass exactly as simulated. The shader combines four distinct
    // atlas regions; no canonical overlap reconciliation is performed. The
    // gutter duplicates the border texels, so the 12 m ring does not clamp to
    // an atlas neighbour.
    let atlas_pitch = EROSION_ATLAS_PITCH;
    let gutter = EROSION_ATLAS_GUTTER as i64;
    let mut atlas_height = vec![0.0f32; atlas_pitch * atlas_pitch * 4];
    let mut atlas_flow = vec![0.0f32; atlas_pitch * atlas_pitch * 4];
    for atlas_z in 0..atlas_pitch {
        let output_z = (atlas_z as i64 - gutter).clamp(0, output_resolution as i64 - 1) as usize;
        for atlas_x in 0..atlas_pitch {
            let output_x =
                (atlas_x as i64 - gutter).clamp(0, output_resolution as i64 - 1) as usize;
            let source = output_z * output_resolution + output_x;
            let destination = atlas_z * atlas_pitch + atlas_x;
            for channel in 0..4 {
                atlas_height[destination * 4 + channel] = output_height[source * 4 + channel];
                atlas_flow[destination * 4 + channel] = output_flow[source * 4 + channel];
            }
        }
    }

    let stats = TileDiagnostics {
        minimum_height: minimum,
        maximum_height: maximum,
        land_coverage: 100.0 * land_cells as f32 / (output_resolution * output_resolution) as f32,
        maximum_incision: (-minimum_delta).max(0.0),
        maximum_deposition: maximum_delta.max(0.0),
        erosion_detail: if detail_samples > 0 {
            (detail / detail_samples as f64) as f32
        } else {
            0.0
        },
        flow_axis_bias: if moving_flow_cells > 0 {
            (flow_axis_bias / moving_flow_cells as f64) as f32
        } else {
            0.0
        },
        maximum_drainage,
        bedrock_exposure: 100.0 * bare_cells as f32 / land_cells.max(1) as f32,
        mean_loose_cover: (loose_total / land_cells.max(1) as f64) as f32,
    };

    let tile = FinalizedTile {
        key,
        cpu_delta,
        stats,
    };
    (tile, atlas_height, atlas_flow)
}

// ---------------------------------------------------------------------------
// Pipeline + bind group helpers
// ---------------------------------------------------------------------------

/// group(2): one uniform buffer.
fn stage_uniform_layout() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        "erosion_stage_layout",
        &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    )
}

/// group(1): `count` RGBA32F textures at bindings 0..count and the samplers
/// the WGSL declares at 8..8+count.
fn sim_inputs_layout(count: usize) -> BindGroupLayoutDescriptor {
    let mut entries = Vec::with_capacity(count * 2);
    for index in 0..count {
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: index as u32,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        });
    }
    for index in 0..count {
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 8 + index as u32,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
            count: None,
        });
    }
    BindGroupLayoutDescriptor::new("erosion_inputs_layout", &entries)
}

/// The group(1) bind group for one pass.
fn target_bind_group(
    device: &RenderDevice,
    cache: &PipelineCache,
    label: &'static str,
    layout: &BindGroupLayoutDescriptor,
    views: &[&wgpu::TextureView],
    sampler: &wgpu::Sampler,
) -> BindGroup {
    let mut entries = Vec::with_capacity(views.len() * 2);
    for (index, view) in views.iter().copied().enumerate() {
        entries.push(wgpu::BindGroupEntry {
            binding: index as u32,
            resource: wgpu::BindingResource::TextureView(view),
        });
    }
    for index in 0..views.len() {
        entries.push(wgpu::BindGroupEntry {
            binding: 8 + index as u32,
            resource: wgpu::BindingResource::Sampler(sampler),
        });
    }
    super::bind_group(device, cache, label, layout, &entries)
}

/// `SetTextureFilter(base, TEXTURE_FILTER_POINT)`: the simulation state is
/// always fetched by texel, never filtered.
fn make_sim_sampler(device: &RenderDevice) -> wgpu::Sampler {
    device.wgpu_device().create_sampler(&wgpu::SamplerDescriptor {
        label: Some("erosion_sim_sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        ..Default::default()
    })
}

fn stage_buffer(device: &RenderDevice, label: &str) -> wgpu::Buffer {
    // Large enough for every erosion stage struct (the terrain pass's 88
    // bytes is the largest).
    device.wgpu_device().create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: 128,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn stage_bind_group(
    device: &RenderDevice,
    cache: &PipelineCache,
    layout: &BindGroupLayoutDescriptor,
    label: &'static str,
    buffer: &wgpu::Buffer,
) -> BindGroup {
    super::bind_group(
        device,
        cache,
        label,
        layout,
        &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    )
}

/// Queues one simulation pipeline: a full-screen triangle into `targets`
/// Rgba32Float targets, no vertex buffers, no depth, no blending (the C++
/// calls `rlDisableColorBlend` because alpha carries loose cover, flux or
/// discharge).
fn queue_erosion_pipeline(
    cache: &PipelineCache,
    label: &'static str,
    shader: Handle<Shader>,
    layouts: Vec<BindGroupLayoutDescriptor>,
    targets: usize,
) -> CachedRenderPipelineId {
    cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some(label.into()),
        layout: layouts,
        immediate_size: 0,
        vertex: VertexState {
            shader: shader.clone(),
            shader_defs: vec![],
            entry_point: Some("vs_main".into()),
            buffers: Vec::new(),
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            // The erosion targets are sampled by texel and never culled by
            // winding in the C++ (raylib's 2D path disables culling).
            cull_mode: None,
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(FragmentState {
            shader,
            shader_defs: vec![],
            entry_point: Some("fs_main".into()),
            targets: (0..targets)
                .map(|_| {
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba32Float,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })
                })
                .collect(),
        }),
        zero_initialize_workgroup_memory: false,
    })
}

fn record_erosion_pass(
    context: &mut RenderContext<'_, '_>,
    pipeline: &RenderPipeline,
    globals: &BindGroup,
    inputs: &BindGroup,
    stage: &BindGroup,
    target: &wgpu::TextureView,
) {
    record_erosion_pass_targets(context, pipeline, globals, inputs, stage, &[target]);
}

/// One full-screen simulation pass into every view of `targets`, in
/// `@location` order.
fn record_erosion_pass_targets(
    context: &mut RenderContext<'_, '_>,
    pipeline: &RenderPipeline,
    globals: &BindGroup,
    inputs: &BindGroup,
    stage: &BindGroup,
    targets: &[&wgpu::TextureView],
) {
    let color_attachments: Vec<Option<RenderPassColorAttachment>> = targets
        .iter()
        .map(|target| {
            Some(RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    // BeginTextureMode leaves the target's contents in place,
                    // and the full-screen triangle overwrites every texel of
                    // the 480x480 target, so the load value is unobservable.
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })
        })
        .collect();
    let mut pass = context.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("erosion_pass"),
        color_attachments: &color_attachments,
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    pass.set_render_pipeline(pipeline);
    pass.set_bind_group(0, globals, &[]);
    pass.set_bind_group(1, inputs, &[]);
    pass.set_bind_group(2, stage, &[]);
    pass.draw(0..3, 0..1);
}

// ---------------------------------------------------------------------------
// Readback helpers
// ---------------------------------------------------------------------------

fn take_staging_buffer(device: &RenderDevice, pool: &Mutex<Vec<wgpu::Buffer>>) -> wgpu::Buffer {
    if let Ok(mut pooled) = pool.lock() {
        if let Some(buffer) = pooled.pop() {
            return buffer;
        }
    }
    device.wgpu_device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("erosion_readback_staging"),
        size: SIM_READBACK_BYTES,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    })
}

/// Maps one staging buffer and hands the pixels to the slots. wgpu invokes the
/// callback from `RenderDevice::poll` on the render thread; it must not touch
/// the device, the queue or any render resource, so it only copies bytes,
/// returns the buffer to the pool and records the result.
fn map_readback(
    buffer: wgpu::Buffer,
    slots: Arc<Mutex<ReadbackSlots>>,
    pool: Arc<Mutex<Vec<wgpu::Buffer>>>,
    in_flight: Arc<AtomicUsize>,
    half: ReadbackHalf,
) {
    let callback_buffer = buffer.clone();
    buffer.map_async(wgpu::MapMode::Read, .., move |result| {
        let outcome = match result {
            Ok(()) => {
                let pixels = {
                    let view = callback_buffer.slice(..).get_mapped_range();
                    bytes_to_f32(&view)
                };
                callback_buffer.unmap();
                HalfResult::Ready(pixels)
            }
            Err(_) => HalfResult::Failed,
        };
        if let Ok(mut pooled) = pool.lock() {
            pooled.push(callback_buffer);
        }
        if let Ok(mut slots) = slots.lock() {
            match half {
                ReadbackHalf::Terrain => slots.terrain = Some(outcome),
                ReadbackHalf::Water => slots.water = Some(outcome),
                ReadbackHalf::Drainage => slots.drainage = Some(outcome),
            }
        }
        // Last, so a zero count means both halves are already stored.
        in_flight.fetch_sub(1, Ordering::SeqCst);
    });
}

fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn f32_to_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

// ---------------------------------------------------------------------------
// Systems
// ---------------------------------------------------------------------------

/// Builds the simulation's pipelines and buffers once the shared world
/// textures and the globals buffer exist.
///
/// Runs in `Prepare`, ordered after `prepare_forest_globals` (which mod.rs
/// chains after the texture preparation), so both prerequisites are present.
fn prepare_erosion_sim(
    device: Res<RenderDevice>,
    pipeline_cache: Res<PipelineCache>,
    handles: Res<ForestShaderHandles>,
    globals: Res<ForestGlobals>,
    textures: Res<GpuWorldTexturesOption>,
    state: Res<ErosionSimState>,
) {
    let Ok(mut guard) = state.sim.lock() else {
        return;
    };
    if guard.is_some() {
        return;
    }
    let (Some(textures), Some(globals_buffer)) =
        (textures.0.as_deref(), globals.buffer.as_ref())
    else {
        return;
    };
    *guard = Some(ErosionSim::build(
        &device,
        &pipeline_cache,
        &handles,
        textures,
        globals_buffer,
    ));
}

pub fn register_erosion_systems(render_app: &mut bevy::app::SubApp) {
    render_app.add_systems(
        bevy::render::Render,
        prepare_erosion_sim
            .in_set(RenderSystems::Prepare)
            .after(crate::render::prepare_forest_globals),
    );
}
