//! The erosion simulation on its own GPU device, driven by a dedicated worker
//! thread.
//!
//! This is the old `render/erosion_node.rs` simulation moved off the render
//! schedule: the whole GPU half of the C++ `HydraulicErosion` (`init` frame,
//! `RunErosionIterations`, finalize copies) runs on a device the worker opens
//! itself over the render adapter, so the render thread stops recording (and
//! waiting on) 568 pass recordings per prewarm frame inside its own schedule.
//! Two command queues execute concurrently on the adapter; the worker never
//! queues to the render device, so the render thread's own submission
//! ordering is untouched by the sim.
//!
//! Life cycle, in order:
//!
//! 1. the main world publishes the five WGSL sources through
//!    [`ErosionBridge::publish_shader_sources`] (it owns the asset server);
//! 2. the render world's slim erosion node asks [`spawn_erosion_worker`] once
//!    it has the shared `RenderAdapter`; the worker thread opens its own
//!    `wgpu` device, mirrors the render device's requested features and
//!    limits (with an identity assert that the device really opened on the
//!    same adapter), and builds every pipeline, bind group, texture and
//!    sampler there;
//! 3. a loop, forever: fatal device errors from the slot; poll+consume any
//!    outstanding finalize readback (pushing `ErosionEvent`s and atlas
//!    patches onto the bridge); otherwise take one command set from the
//!    bridge and process it.
//!
//! Command processing preserves the C++ ordering exactly: `queue.write_*` is
//! ordered before the whole submission and `queue.submit`s execute in order,
//! so (init frame) -> (iteration chunks) -> (finalize copies) can run as
//! three self-submitted command buffers without changing one byte of sim
//! results. Iterations split into chunks of at most
//! `EROSION_WORKER_MAX_ITERATIONS_PER_SUBMIT` so each Metal command buffer
//! stays bounded; in-order execution makes the result identical to one long
//! buffer.
//!
//! Failure protocol: every fatal error route (device creation failure,
//! adapter/device identity mismatch, an uncaptured device error, a `build`
//! failure, a panic anywhere in the thread) ends in
//! `ErosionBridge::set_worker_dead`, and `apply_erosion_events` panics on
//! reading it — a dead worker is a dead simulation, and the old render-side
//! sim panicked its thread the same way. The thread is detached and never
//! joined: a `wgpu` `Device`'s drop can wait seconds for the GPU to drain,
//! and that must never be able to block process exit.
//!
//! Unlike the render node, the worker compiles its raw WGSL once at startup:
//! asset hot reload does not reach it, so an edited erosion shader needs a
//! restart (documented in PORTING-SPEC.md's shader flow) — and the scratch
//! asset root flow (`cwd/assets`) still works, because the main world
//! publishes whatever assets it actually loaded.

use crate::constants::*;
use crate::erosion::{
    ErosionBridge, ErosionEvent, ErosionFrameCommands, ErosionPhaseCosts, ErosionShaderSources,
    FinalizedTile, TileKey,
};
use crate::erosion_finalize::{bytes_to_f32, finalize_erosion_tile};
use crate::render::gpu_textures::SIM_TEXTURE_SIZE;
use crate::render::{
    ErosionFluxStageUniforms, ErosionInitStageUniforms, ErosionTerrainStageUniforms,
    ErosionThermalStageUniforms, ErosionWaterStageUniforms, GlobalUniformsGpu,
};
use bevy::tasks::block_on;
use bytemuck::Zeroable;
use std::borrow::Cow;
use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

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
/// set is consumed, so a late half can never be mistaken for a fresh one.
#[derive(Default)]
struct ReadbackSlots {
    key: Option<TileKey>,
    terrain: Option<HalfResult>,
    water: Option<HalfResult>,
    drainage: Option<HalfResult>,
}

/// What a complete readback cycle produced for the worker loop.
enum ReadbackOutcome {
    /// Nothing landed yet (a plane is still in flight).
    Idle,
    /// The tile finalized, with its two 260x260 atlas patches.
    Finalized(FinalizedTile, Vec<f32>, Vec<f32>),
    /// The readback failed; the main world requeues the tile.
    Failed(TileKey),
}

// ---------------------------------------------------------------------------
// Simulation state
// ---------------------------------------------------------------------------

/// Every GPU object the erosion worker owns, mirroring the C++
/// `HydraulicErosion` struct's textures, targets, shaders and ping-pong
/// indices — all on the worker's own device.
struct WorkerErosionSim {
    device: wgpu::Device,
    queue: wgpu::Queue,

    init_pipeline: wgpu::RenderPipeline,
    flux_pipeline: wgpu::RenderPipeline,
    water_pipeline: wgpu::RenderPipeline,
    terrain_pipeline: wgpu::RenderPipeline,
    thermal_pipeline: wgpu::RenderPipeline,

    /// group(0): the frame's shared globals. The sim shaders declare
    /// `var<uniform> globals: GlobalUniforms` but never read it (asserted in
    /// the tests), so this binds a zero-filled dummy — the WGSL compiles
    /// unchanged on the worker's device.
    globals: wgpu::BindGroup,

    /// `uMode` 0, 1 and 2 init uniforms (one per mode: `queue.write_buffer`
    /// is ordered before the whole submission, so one buffer cannot hold
    /// `uMode` 0 for the terrain pass and `uMode` 1 for the water/flux passes
    /// of the same frame).
    init_stage_buffers: [wgpu::Buffer; 3],
    init_stage_groups: [wgpu::BindGroup; 3],
    flux_stage_buffer: wgpu::Buffer,
    flux_stage_group: wgpu::BindGroup,
    water_stage_buffer: wgpu::Buffer,
    water_stage_group: wgpu::BindGroup,
    terrain_stage_buffer: wgpu::Buffer,
    terrain_stage_group: wgpu::BindGroup,
    thermal_stage_buffer: wgpu::Buffer,
    thermal_stage_group: wgpu::BindGroup,

    /// group(1) inputs, pre-built for every ping-pong combination because
    /// bind groups are immutable.
    init_inputs: wgpu::BindGroup,
    flux_inputs: [[[wgpu::BindGroup; 2]; 2]; 2],
    water_inputs: [[[wgpu::BindGroup; 2]; 2]; 2],
    terrain_inputs: [[[wgpu::BindGroup; 2]; 2]; 2],
    thermal_inputs: [[wgpu::BindGroup; 2]; 2],

    /// The short-lived base height map the init passes sample, one Rgba32Float
    /// 480x480 texture reused for every tile (the C++'s `base` Texture2D).
    base_texture: wgpu::Texture,

    terrain_targets: [wgpu::Texture; 2],
    water_targets: [wgpu::Texture; 2],
    // No readback plane samples flux or routing; the textures are kept only so
    // the views (which pass validation) keep their GPU objects alive.
    #[allow(dead_code)]
    flux_targets: [wgpu::Texture; 2],
    drainage_targets: [wgpu::Texture; 2],
    /// Written by every water pass and read by the terrain pass right after,
    /// so it needs no ping-pong partner.
    #[allow(dead_code)]
    routing_target: wgpu::Texture,
    terrain_views: [wgpu::TextureView; 2],
    water_views: [wgpu::TextureView; 2],
    flux_views: [wgpu::TextureView; 2],
    drainage_views: [wgpu::TextureView; 2],
    routing_view: wgpu::TextureView,

    /// `terrainIndex`, `waterIndex`, `fluxIndex`, plus the drainage state.
    terrain_index: usize,
    water_index: usize,
    flux_index: usize,
    drainage_index: usize,

    /// `activeTile` and how many of `settings.iterations` it has run.
    active_tile: Option<TileKey>,
    active_iterations: usize,
    /// Microseconds `finalize_erosion_tile` spent on the last consume, for
    /// the frame-time phase attribution (zero unless a tile finalized just
    /// now).
    last_finalize_us: u64,
    /// Erosion settings in force (`appliedErosionSettings`).
    settings: ErosionSettings,

    /// A tile whose result has already been published; the main world keeps
    /// sending `finalize` until it sees the event, and a second readback of
    /// the same state would be pure waste.
    delivered: Option<TileKey>,

    /// Readback machinery, shared with the map callbacks (`Arc` because the
    /// callbacks outlive any borrow of the sim).
    readback: Arc<Mutex<ReadbackSlots>>,
    pool: Arc<Mutex<Vec<wgpu::Buffer>>>,
    in_flight: Arc<AtomicUsize>,
}

/// `true` while a finalize readback is mid-flight: a recorded copy whose
/// halves have not all landed, or map callbacks still pending from a failed
/// set. The worker keeps polling (not command processing) until this clears,
/// because issuing a second copy while the first set's maps can still fire
/// would interleave stale buffers into the new set's slots.
fn readback_outstanding(sim: &WorkerErosionSim) -> bool {
    let keyed = match sim.readback.lock() {
        Ok(slots) => slots.key.is_some(),
        // A poisoned slots mutex cannot prove idleness; keep polling and let
        // the consume path answer with what it can see.
        Err(_) => true,
    };
    keyed || sim.in_flight.load(Ordering::SeqCst) != 0
}

impl WorkerErosionSim {
    /// Creates every pipeline, bind group and scratch texture on the worker's
    /// own device. Raw WGSL in, whole engine out — the published sources are
    /// self-contained files, so no preprocessing runs between.
    fn build(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        sources: &ErosionShaderSources,
    ) -> Result<Self, String> {
        let init_module = create_sim_module(device, "erosion_init_wgsl", &sources.init)?;
        let flux_module = create_sim_module(device, "erosion_flux_wgsl", &sources.flux)?;
        let water_module = create_sim_module(device, "erosion_water_wgsl", &sources.water)?;
        let terrain_module = create_sim_module(device, "erosion_terrain_wgsl", &sources.terrain)?;
        let thermal_module = create_sim_module(device, "erosion_thermal_wgsl", &sources.thermal)?;

        // group(0): the declared-but-unread globals uniform. Buffers are
        // zero-initialized by wgpu; nothing ever writes the dummy.
        let globals_layout_descriptor = wgpu::BindGroupLayoutDescriptor {
            label: Some("erosion_worker_globals_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    // Unread by every sim shader; no minimum to enforce.
                    min_binding_size: None,
                },
                count: None,
            }],
        };
        let globals_layout = device.create_bind_group_layout(&globals_layout_descriptor);
        let globals_buffer_descriptor = wgpu::BufferDescriptor {
            label: Some("erosion_worker_globals"),
            size: size_of::<GlobalUniformsGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        };
        let globals_buffer = device.create_buffer(&globals_buffer_descriptor);
        let globals_descriptor = wgpu::BindGroupDescriptor {
            label: Some("erosion_worker_globals"),
            layout: &globals_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: globals_buffer.as_entire_binding(),
            }],
        };
        let globals = device.create_bind_group(&globals_descriptor);

        // group(2): one uniform buffer.
        let stage_layout_descriptor = wgpu::BindGroupLayoutDescriptor {
            label: Some("erosion_worker_stage_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        };
        let stage_layout = device.create_bind_group_layout(&stage_layout_descriptor);
        let init_inputs_layout = sim_inputs_layout(device, "erosion_init_inputs_layout", 1);
        let flux_inputs_layout = sim_inputs_layout(device, "erosion_flux_inputs_layout", 3);
        let water_inputs_layout = sim_inputs_layout(device, "erosion_water_inputs_layout", 3);
        let terrain_inputs_layout = sim_inputs_layout(device, "erosion_terrain_inputs_layout", 4);
        let thermal_inputs_layout = sim_inputs_layout(device, "erosion_thermal_inputs_layout", 2);

        // The C++ sets TEXTURE_FILTER_POINT on the base map and keeps every
        // simulation texture point-sampled; the sim shaders only ever
        // `textureLoad`, so the samplers they declare (bindings 8..) are
        // present for the layout but never filter.
        let sampler = make_sim_sampler(device);

        // The worker owns its simulation textures; only readback bytes
        // (finalize) and published atlas bytes (uploads) ever leave this
        // device.
        let terrain_targets = [
            sim_texture(device, "erosion_sim_terrain_0"),
            sim_texture(device, "erosion_sim_terrain_1"),
        ];
        let water_targets = [
            sim_texture(device, "erosion_sim_water_0"),
            sim_texture(device, "erosion_sim_water_1"),
        ];
        let flux_targets = [
            sim_texture(device, "erosion_sim_flux_0"),
            sim_texture(device, "erosion_sim_flux_1"),
        ];
        let drainage_targets = [
            sim_texture(device, "erosion_sim_drainage_0"),
            sim_texture(device, "erosion_sim_drainage_1"),
        ];
        let routing_target = sim_texture(device, "erosion_sim_routing");
        let terrain_views = [
            terrain_targets[0].create_view(&Default::default()),
            terrain_targets[1].create_view(&Default::default()),
        ];
        let water_views = [
            water_targets[0].create_view(&Default::default()),
            water_targets[1].create_view(&Default::default()),
        ];
        let flux_views = [
            flux_targets[0].create_view(&Default::default()),
            flux_targets[1].create_view(&Default::default()),
        ];
        let drainage_views = [
            drainage_targets[0].create_view(&Default::default()),
            drainage_targets[1].create_view(&Default::default()),
        ];
        let routing_view = routing_target.create_view(&Default::default());

        let base_texture_descriptor = wgpu::TextureDescriptor {
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
        };
        let base_texture = device.create_texture(&base_texture_descriptor);
        let base_view = base_texture.create_view(&Default::default());
        let init_inputs = make_bind_group(
            device,
            "erosion_worker_init_inputs",
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
                    make_bind_group(
                        device,
                        "erosion_worker_flux_inputs",
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
                    make_bind_group(
                        device,
                        "erosion_worker_water_inputs",
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
                    make_bind_group(
                        device,
                        "erosion_worker_terrain_inputs",
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
                make_bind_group(
                    device,
                    "erosion_worker_thermal_inputs",
                    &thermal_inputs_layout,
                    &[&terrain_views[terrain_index], &drainage_views[drainage_index]],
                    &sampler,
                )
            })
        });

        // One init uniform per mode, all written per command set (the C++
        // sets them per pass inside the init frame; wgpu's write-before-
        // submit ordering makes per-mode buffers the only safe layout).
        let init_stage_buffers = [
            stage_buffer(device, "erosion_init_mode_0"),
            stage_buffer(device, "erosion_init_mode_1"),
            stage_buffer(device, "erosion_init_mode_2"),
        ];
        let init_stage_groups = [
            make_stage_group(
                device,
                "erosion_worker_init_stage_0",
                &stage_layout,
                &init_stage_buffers[0],
            ),
            make_stage_group(
                device,
                "erosion_worker_init_stage_1",
                &stage_layout,
                &init_stage_buffers[1],
            ),
            make_stage_group(
                device,
                "erosion_worker_init_stage_2",
                &stage_layout,
                &init_stage_buffers[2],
            ),
        ];
        let flux_stage_buffer = stage_buffer(device, "erosion_flux_stage");
        let water_stage_buffer = stage_buffer(device, "erosion_water_stage");
        let terrain_stage_buffer = stage_buffer(device, "erosion_terrain_stage");
        let thermal_stage_buffer = stage_buffer(device, "erosion_thermal_stage");
        let flux_stage_group = make_stage_group(
            device,
            "erosion_worker_flux_stage_group",
            &stage_layout,
            &flux_stage_buffer,
        );
        let water_stage_group = make_stage_group(
            device,
            "erosion_worker_water_stage_group",
            &stage_layout,
            &water_stage_buffer,
        );
        let terrain_stage_group = make_stage_group(
            device,
            "erosion_worker_terrain_stage_group",
            &stage_layout,
            &terrain_stage_buffer,
        );
        let thermal_stage_group = make_stage_group(
            device,
            "erosion_worker_thermal_stage_group",
            &stage_layout,
            &thermal_stage_buffer,
        );

        // Layouts: globals, inputs, stage uniforms. The sim shaders take no
        // vertex buffers; all but the terrain pass write one colour target.
        let init_pipeline = make_pipeline(
            device,
            "erosion_init_pipeline",
            &init_module,
            &[&globals_layout, &init_inputs_layout, &stage_layout],
            1,
        );
        let flux_pipeline = make_pipeline(
            device,
            "erosion_flux_pipeline",
            &flux_module,
            &[&globals_layout, &flux_inputs_layout, &stage_layout],
            1,
        );
        // Water and the routing normaliser.
        let water_pipeline = make_pipeline(
            device,
            "erosion_water_pipeline",
            &water_module,
            &[&globals_layout, &water_inputs_layout, &stage_layout],
            2,
        );
        // Terrain and drainage states.
        let terrain_pipeline = make_pipeline(
            device,
            "erosion_terrain_pipeline",
            &terrain_module,
            &[&globals_layout, &terrain_inputs_layout, &stage_layout],
            2,
        );
        let thermal_pipeline = make_pipeline(
            device,
            "erosion_thermal_pipeline",
            &thermal_module,
            &[&globals_layout, &thermal_inputs_layout, &stage_layout],
            1,
        );

        Ok(Self {
            device: device.clone(),
            queue: queue.clone(),
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
            terrain_targets,
            water_targets,
            flux_targets,
            drainage_targets,
            routing_target,
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
            last_finalize_us: 0,
            settings: ErosionSettings::default(),
            delivered: None,
            readback: Arc::new(Mutex::new(ReadbackSlots::default())),
            pool: Arc::new(Mutex::new(Vec::new())),
            in_flight: Arc::new(AtomicUsize::new(0)),
        })
    }

    /// `uMode` 0, 1 and 2 init uniforms, the base height and drainage upload
    /// and the eight init passes (`InitializeErosionTextures`), recorded into
    /// the caller's encoder. Queue-level writes (texture, uniforms) are
    /// ordered before the submission of everything recorded after them.
    fn initialize_tile(
        &mut self,
        init_key: TileKey,
        base_height: &[f32],
        drainage_area: &[f32],
        river: &[f32],
        sim_min: [f32; 2],
        encoder: &mut wgpu::CommandEncoder,
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
            self.queue.write_texture(
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
        self.queue
            .write_buffer(&self.init_stage_buffers[0], 0, bytemuck::bytes_of(&uniforms));
        uniforms.mode = 1;
        self.queue
            .write_buffer(&self.init_stage_buffers[1], 0, bytemuck::bytes_of(&uniforms));
        uniforms.mode = 2;
        self.queue
            .write_buffer(&self.init_stage_buffers[2], 0, bytemuck::bytes_of(&uniforms));

        // for (int i = 0; i < 2; ++i): terrain keeps the stamped state, water
        // and flux are cleared, drainage takes the routed base catchments,
        // and both scratch copies start identical.
        for index in 0..2 {
            record_erosion_pass(
                encoder,
                &self.init_pipeline,
                &self.globals,
                &self.init_inputs,
                &self.init_stage_groups[0],
                &self.terrain_views[index],
            );
            record_erosion_pass(
                encoder,
                &self.init_pipeline,
                &self.globals,
                &self.init_inputs,
                &self.init_stage_groups[1],
                &self.water_views[index],
            );
            record_erosion_pass(
                encoder,
                &self.init_pipeline,
                &self.globals,
                &self.init_inputs,
                &self.init_stage_groups[1],
                &self.flux_views[index],
            );
            record_erosion_pass(
                encoder,
                &self.init_pipeline,
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
    fn write_stage_uniforms(&self, sim_min: [f32; 2]) {
        let resolution = [SIM_TEXTURE_SIZE as f32, SIM_TEXTURE_SIZE as f32];

        let mut flux = ErosionFluxStageUniforms::zeroed();
        flux.resolution = resolution;
        flux.delta_time = EROSION_DELTA_TIME;
        flux.cell_size = EROSION_CELL_SIZE;
        flux.gravity = EROSION_GRAVITY;
        self.queue
            .write_buffer(&self.flux_stage_buffer, 0, bytemuck::bytes_of(&flux));

        let mut water = ErosionWaterStageUniforms::zeroed();
        water.resolution = resolution;
        water.delta_time = EROSION_DELTA_TIME;
        water.cell_size = EROSION_CELL_SIZE;
        water.rain = self.settings.rain;
        water.evaporation = self.settings.evaporation;
        water.sea_level = SEA_LEVEL;
        self.queue
            .write_buffer(&self.water_stage_buffer, 0, bytemuck::bytes_of(&water));

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
        self.queue
            .write_buffer(&self.terrain_stage_buffer, 0, bytemuck::bytes_of(&terrain));

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
        self.queue
            .write_buffer(&self.thermal_stage_buffer, 0, bytemuck::bytes_of(&thermal));
    }

    /// `RunErosionIterations`: flux, water, terrain (with drainage) and
    /// thermal into the opposite targets, `count` times.
    fn run_iterations(&mut self, count: usize, encoder: &mut wgpu::CommandEncoder) {
        for _ in 0..count {
            let next_flux = 1 - self.flux_index;
            record_erosion_pass(
                encoder,
                &self.flux_pipeline,
                &self.globals,
                &self.flux_inputs[self.flux_index][self.terrain_index][self.water_index],
                &self.flux_stage_group,
                &self.flux_views[next_flux],
            );
            self.flux_index = next_flux;

            let next_water = 1 - self.water_index;
            record_erosion_pass_targets(
                encoder,
                &self.water_pipeline,
                &self.globals,
                &self.water_inputs[self.water_index][self.flux_index][self.terrain_index],
                &self.water_stage_group,
                &[&self.water_views[next_water], &self.routing_view],
            );
            self.water_index = next_water;

            let next_terrain = 1 - self.terrain_index;
            let next_drainage = 1 - self.drainage_index;
            record_erosion_pass_targets(
                encoder,
                &self.terrain_pipeline,
                &self.globals,
                &self.terrain_inputs[self.terrain_index][self.water_index][self.drainage_index],
                &self.terrain_stage_group,
                &[&self.terrain_views[next_terrain], &self.drainage_views[next_drainage]],
            );
            self.terrain_index = next_terrain;
            self.drainage_index = next_drainage;

            let next_terrain = 1 - self.terrain_index;
            record_erosion_pass(
                encoder,
                &self.thermal_pipeline,
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

    /// Applies one command set: init, iterations (split across
    /// self-submitted command buffers), and the readback copies that
    /// `FinalizeErosionTile` performs at the end of the frame.
    ///
    /// The worker loop only hands commands over once no readback is
    /// outstanding, so each `queue.submit` lands strictly after the previous
    /// one's — the chunks execute exactly as one long command buffer would.
    fn process_commands(&mut self, commands: &ErosionFrameCommands) {
        if let Some(iterate) = &commands.iterate {
            self.settings = iterate.settings;
        }

        // `UpdateErosionCache` returns immediately while no tile is active;
        // a frame with neither an active tile nor a fresh init does the same.
        if self.active_tile.is_none() && commands.init.is_none() {
            return;
        }

        self.write_stage_uniforms(commands.sim_min);

        if let Some(init) = &commands.init {
            let mut encoder = self.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor {
                    label: Some("erosion_init_frame"),
                },
            );
            self.initialize_tile(
                init.key,
                &init.base_height,
                &init.drainage_area,
                &init.river,
                commands.sim_min,
                &mut encoder,
            );
            self.queue.submit(Some(encoder.finish()));
        }

        let count = self.iteration_count(commands);
        for chunk in chunk_plan(count) {
            let mut encoder = self.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor {
                    label: Some("erosion_iterations"),
                },
            );
            self.run_iterations(chunk, &mut encoder);
            self.active_iterations += chunk;
            self.queue.submit(Some(encoder.finish()));
        }

        if commands.finalize {
            self.issue_readback();
        }
    }

    /// Records the finalize copies (`rlReadTexturePixels`' three planes) and
    /// issues their maps: the worker's submit exists the moment it returns,
    /// so `map_async` waits for exactly this submission — the render node
    /// needed a frame of delay because bevy only submitted its encoder after
    /// the node returned.
    fn issue_readback(&mut self) -> bool {
        let Some(key) = self.active_tile else {
            return false;
        };
        if self.delivered == Some(key) {
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

        let terrain_buffer = take_staging_buffer(&self.device, &self.pool);
        let water_buffer = take_staging_buffer(&self.device, &self.pool);
        let drainage_buffer = take_staging_buffer(&self.device, &self.pool);
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
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("erosion_finalize_frame"),
        });
        encoder.copy_texture_to_buffer(
            self.terrain_targets[self.terrain_index].as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &terrain_buffer,
                layout,
            },
            extent,
        );
        encoder.copy_texture_to_buffer(
            self.water_targets[self.water_index].as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &water_buffer,
                layout,
            },
            extent,
        );
        encoder.copy_texture_to_buffer(
            self.drainage_targets[self.drainage_index].as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &drainage_buffer,
                layout,
            },
            extent,
        );
        self.queue.submit(Some(encoder.finish()));

        // One per plane: each `map_readback` callback subtracts one, and the
        // count has to land back on exactly zero or the issue guard (which
        // refuses to start while a readback is outstanding) never runs again.
        self.in_flight.fetch_add(3, Ordering::SeqCst);
        map_readback(
            terrain_buffer,
            self.readback.clone(),
            self.pool.clone(),
            self.in_flight.clone(),
            ReadbackHalf::Terrain,
        );
        map_readback(
            water_buffer,
            self.readback.clone(),
            self.pool.clone(),
            self.in_flight.clone(),
            ReadbackHalf::Water,
        );
        map_readback(
            drainage_buffer,
            self.readback.clone(),
            self.pool.clone(),
            self.in_flight.clone(),
            ReadbackHalf::Drainage,
        );
        true
    }

    /// Consumes a completed readback and runs the finalize half of
    /// `FinalizeErosionTile`: the crop, the statistics and the atlas patches.
    /// The patches go back with the outcome — the worker pushes them onto the
    /// bridge, and the render world uploads them once the tile's slot is
    /// published through the lookup records.
    fn consume_readback(&mut self) -> ReadbackOutcome {
        self.last_finalize_us = 0;
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
                let started_at = std::time::Instant::now();
                let (tile, atlas_height, atlas_flow) =
                    finalize_erosion_tile(key, &terrain, &water, &drainage);
                self.last_finalize_us = started_at.elapsed().as_micros() as u64;
                self.delivered = Some(key);
                ReadbackOutcome::Finalized(tile, atlas_height, atlas_flow)
            }
            _ => {
                // A failed or short readback: the C++ counts it and requeues
                // the tile, which the main world does from the event.
                self.delivered = Some(key);
                ReadbackOutcome::Failed(key)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Worker thread
// ---------------------------------------------------------------------------

/// Why the worker stopped: an owned slot the map callbacks and the loop both
/// see, so an error raised on any thread converges on one death report.
type ErrorSlot = Arc<Mutex<Option<String>>>;

/// Starts the erosion worker thread. `adapter` is the renderer's shared
/// adapter (the worker opens its own device on it); `features` and `limits`
/// mirror what the render device requested, so the sim's pipelines accept the
/// same resource set the node's did.
///
/// Returns whether the thread spawned. A failure to spawn is reported through
/// the bridge too (`set_worker_dead`), so the main world panics loudly on the
/// next frame instead of waiting on a finalize that can never land.
pub fn spawn_erosion_worker(
    bridge: &ErosionBridge,
    adapter: &wgpu::Adapter,
    features: wgpu::Features,
    limits: wgpu::Limits,
) -> bool {
    let thread_bridge = bridge.clone();
    let adapter = adapter.clone();
    let spawned = std::thread::Builder::new()
        .name("erosion_worker".into())
        .spawn(move || worker_thread_entry(thread_bridge, adapter, features, limits));
    match spawned {
        Ok(_) => true,
        Err(err) => {
            let why = format!("the erosion worker thread failed to start: {err}");
            log::error!("EROSION: {why}");
            bridge.set_worker_dead(why);
            false
        }
    }
}

/// Wraps the thread body in `catch_unwind`: a panicking worker would die
/// silently (a detached thread's panic only prints and the app keeps
/// running), leaving the tile cache waiting on a finalize that never lands.
fn worker_thread_entry(
    bridge: ErosionBridge,
    adapter: wgpu::Adapter,
    features: wgpu::Features,
    limits: wgpu::Limits,
) {
    let panic_bridge = bridge.clone();
    let outcome = std::panic::catch_unwind(AssertUnwindSafe(move || {
        worker_thread_body(bridge, adapter, features, limits)
    }));
    if let Err(panic) = outcome {
        let why = panic_message(&panic);
        log::error!("EROSION: worker thread panicked: {why}");
        panic_bridge.set_worker_dead(format!("worker thread panicked: {why}"));
    }
}

fn worker_thread_body(
    bridge: ErosionBridge,
    adapter: wgpu::Adapter,
    features: wgpu::Features,
    limits: wgpu::Limits,
) {
    let Some(sources) = bridge.take_shader_sources() else {
        // The main world publishes the sources before the render node spawns
        // the worker; a missing set is a wiring bug, not a GPU failure.
        let why = "spawned without shader sources".to_string();
        log::error!("EROSION: {why}");
        bridge.set_worker_dead(why);
        return;
    };

    // The worker opens its own device on the render adapter: two command
    // queues execute concurrently, and the sim's writes can never interleave
    // with a render submission (the single-owner hazard the old render-side
    // machinery had to schedule around does not exist here).
    //
    // `experimental_features` mirrors bevy's own render-device request
    // (`unsafe { ExperimentalFeatures::enabled() }` in bevy_render); the
    // adapter reports those capability bits and a plain request would refuse
    // a device the render device was happily handed.
    let device_descriptor = wgpu::DeviceDescriptor {
        label: Some("erosion_worker"),
        required_features: features,
        required_limits: limits,
        experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
        memory_hints: Default::default(),
        trace: Default::default(),
    };
    let requested = block_on(adapter.request_device(&device_descriptor));
    let (device, queue) = match requested {
        Ok(pair) => pair,
        Err(err) => {
            let why = format!("request_device on {:?} failed: {err}", adapter.get_info());
            log::error!("EROSION: {why}");
            bridge.set_worker_dead(why);
            return;
        }
    };

    // `request_device` must open on the adapter it was handed: a swap would
    // silently move the sim to a different backend than the renderer's.
    let requested_info = format!("{:?}", adapter.get_info());
    let opened_info = format!("{:?}", device.adapter_info());
    if requested_info != opened_info {
        let why = format!(
            "the worker device opened on {opened_info} instead of {requested_info}"
        );
        log::error!("EROSION: {why}");
        bridge.set_worker_dead(why);
        return;
    }

    // Fatal device errors land in this slot and stop the worker on its next
    // tick. wgpu's default handler panics on whichever thread reports the
    // error — possibly neither of ours — so a dedicated handler replaces it
    // before any module, pipeline or texture is created: invalid WGSL lands
    // in the slot instead of killing an arbitrary thread.
    let error_slot: ErrorSlot = Arc::new(Mutex::new(None));
    let error_handler = {
        let error_slot = error_slot.clone();
        Arc::new(move |error: wgpu::Error| {
            let Ok(mut slot) = error_slot.lock() else {
                return;
            };
            slot.get_or_insert_with(|| error.to_string());
        })
    };
    device.on_uncaptured_error(error_handler);

    let mut sim = match WorkerErosionSim::build(&device, &queue, &sources) {
        Ok(sim) => sim,
        Err(why) => {
            let why = format!("building the erosion sim: {why}");
            log::error!("EROSION: {why}");
            bridge.set_worker_dead(why);
            return;
        }
    };
    log::info!(
        "EROSION: worker device ready on {:?}",
        device.adapter_info()
    );

    let mut costs = ErosionPhaseCosts::default();
    loop {
        // 1. Fatal errors first: the slot, map callbacks, readback halves and
        //    command sets are all checked in that order every tick, so an
        //    error raised anywhere stops the worker before more GPU work is
        //    recorded on a device that just failed.
        if let Some(why) = error_slot.lock().ok().and_then(|mut slot| slot.take()) {
            if let Some(key) = sim.active_tile {
                bridge.push_event(ErosionEvent::ReadbackFailed(key));
            }
            log::error!("EROSION: worker device error: {why}");
            bridge.set_worker_dead(why);
            return;
        }

        if readback_outstanding(&sim) {
            let tick = std::time::Instant::now();
            let _ = device.poll(wgpu::PollType::Poll);
            costs.poll_us += tick.elapsed().as_micros() as u64;
            let tick = std::time::Instant::now();
            match sim.consume_readback() {
                ReadbackOutcome::Idle => {
                    // The maps lag the submission by a GPU instant; poll
                    // again shortly rather than spin.
                    std::thread::sleep(Duration::from_millis(1));
                }
                ReadbackOutcome::Finalized(tile, atlas_height, atlas_flow) => {
                    bridge.push_patches(tile.key, atlas_height, atlas_flow);
                    bridge.push_event(ErosionEvent::TileFinalized(tile));
                }
                ReadbackOutcome::Failed(key) => {
                    bridge.push_event(ErosionEvent::ReadbackFailed(key));
                }
            }
            costs.consume_us += tick.elapsed().as_micros() as u64 - sim.last_finalize_us;
            costs.finalize_us += sim.last_finalize_us;
            continue;
        }

        match bridge.take_commands() {
            Some(commands) => {
                let tick = std::time::Instant::now();
                sim.process_commands(&commands);
                costs.sim_record_us += tick.elapsed().as_micros() as u64;
            }
            None => {
                // Parked on the bridge's condvar: `set_frame` signals the
                // next frame's work; the timeout only bounds a missed wake.
                bridge.wait_for_work(Duration::from_millis(2));
            }
        }
        bridge.record_phase_costs(costs);
        costs = ErosionPhaseCosts::default();
    }
}

/// Best-effort panic payload extraction for the death report.
fn panic_message(panic: &(dyn std::any::Any + Send)) -> String {
    if let Some(message) = panic.downcast_ref::<&'static str>() {
        (*message).to_string()
    } else if let Some(message) = panic.downcast_ref::<String>() {
        message.clone()
    } else {
        "unknown panic".to_string()
    }
}

// ---------------------------------------------------------------------------
// Submission planning
// ---------------------------------------------------------------------------

/// Splits `count` iterations into submissions of at most
/// `EROSION_WORKER_MAX_ITERATIONS_PER_SUBMIT`, in order; the sum is exactly
/// `count`, so the sim's iteration total never changes with the chunking.
fn chunk_plan(count: usize) -> Vec<usize> {
    let full_batches = count / EROSION_WORKER_MAX_ITERATIONS_PER_SUBMIT;
    let remainder = count % EROSION_WORKER_MAX_ITERATIONS_PER_SUBMIT;
    let mut chunks = vec![EROSION_WORKER_MAX_ITERATIONS_PER_SUBMIT; full_batches];
    if remainder > 0 {
        chunks.push(remainder);
    }
    chunks
}

// ---------------------------------------------------------------------------
// GPU object helpers
// ---------------------------------------------------------------------------

/// Compiles one published WGSL asset into a worker-side module.
///
/// This mirrors bevy's pipeline cache exactly: `RenderDevice::create_shader_module`
/// compiles engine-managed shaders with `create_shader_module_trusted` and
/// `ShaderRuntimeChecks::unchecked()` (bevy: "the checks required are
/// prohibitively expensive and a poor default for game engines"). The checked
/// path is *not* value-neutral on Metal — its bounds-check instrumentation
/// changes the generated code's low-order bits, and the erosion sim's chaotic
/// river routing amplifies them (drainage statistics diverge up to 45% at
/// confluence cells). Compiling the same sources the same way is what keeps
/// the worker's simulation bit-identical to the old in-graph node's.
///
/// SAFETY: `create_shader_module_trusted` with unchecked runtime checks trusts
/// the WGSL not to read out of bounds. These sources are the engine's own
/// asset files, compiled unchecked by the old render-side node through the
/// same bevy machinery for the whole project's life; every state read goes
/// through `clampCoord` or a bounds-tested donor scan, and the modules are
/// never composed from untrusted input.
fn create_sim_module(
    device: &wgpu::Device,
    label: &'static str,
    bytes: &[u8],
) -> Result<wgpu::ShaderModule, String> {
    let source = String::from_utf8(bytes.to_vec())
        .map_err(|err| format!("published {label} source is not UTF-8: {err}"))?;
    let desc = wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(Cow::Owned(source)),
    };
    // SAFETY: see the function comment; the trusted inputs are the engine's
    // own publication of its asset files.
    Ok(unsafe {
        device.create_shader_module_trusted(desc, wgpu::ShaderRuntimeChecks::unchecked())
    })
}

/// One RGBA32F simulation target, as `InitializeErosionTextures` created them
/// (RENDER_ATTACHMENT | TEXTURE_BINDING | COPY_SRC).
fn sim_texture(device: &wgpu::Device, label: &'static str) -> wgpu::Texture {
    let texture_descriptor = wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: SIM_TEXTURE_SIZE,
            height: SIM_TEXTURE_SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    };
    device.create_texture(&texture_descriptor)
}

fn sim_inputs_layout(device: &wgpu::Device, label: &'static str, count: usize) -> wgpu::BindGroupLayout {
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
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &entries,
    })
}

/// The group(1) bind group for one pass.
fn make_bind_group(
    device: &wgpu::Device,
    label: &'static str,
    layout: &wgpu::BindGroupLayout,
    views: &[&wgpu::TextureView],
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
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
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &entries,
    })
}

/// `SetTextureFilter(base, TEXTURE_FILTER_POINT)`: the simulation state is
/// always fetched by texel, never filtered.
fn make_sim_sampler(device: &wgpu::Device) -> wgpu::Sampler {
    let sampler_descriptor = wgpu::SamplerDescriptor {
        label: Some("erosion_sim_sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        ..Default::default()
    };
    device.create_sampler(&sampler_descriptor)
}

fn stage_buffer(device: &wgpu::Device, label: &'static str) -> wgpu::Buffer {
    // Large enough for every erosion stage struct (the terrain pass's 88
    // bytes is the largest).
    let buffer_descriptor = wgpu::BufferDescriptor {
        label: Some(label),
        size: 128,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    };
    device.create_buffer(&buffer_descriptor)
}

fn make_stage_group(
    device: &wgpu::Device,
    label: &'static str,
    layout: &wgpu::BindGroupLayout,
    buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    let bind_group_descriptor = wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    };
    device.create_bind_group(&bind_group_descriptor)
}

/// Builds one simulation pipeline: a full-screen triangle into `targets`
/// Rgba32Float targets, no vertex buffers, no depth, no blending (the C++
/// calls `rlDisableColorBlend` because alpha carries loose cover, flux or
/// discharge).
fn make_pipeline(
    device: &wgpu::Device,
    label: &'static str,
    module: &wgpu::ShaderModule,
    layouts: &[&wgpu::BindGroupLayout],
    targets: usize,
) -> wgpu::RenderPipeline {
    // wgpu 29 moved push constants to `immediate_size` (the sim uses none)
    // and takes optional layout entries.
    let bind_group_layouts = layouts.iter().map(|layout| Some(*layout)).collect::<Vec<_>>();
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &bind_group_layouts,
        immediate_size: 0,
    });
    let target_states: Vec<Option<wgpu::ColorTargetState>> = (0..targets)
        .map(|_| {
            Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba32Float,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })
        })
        .collect();
    let pipeline_descriptor = wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            // The sim shaders take no vertex buffers.
            buffers: &[],
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
        fragment: Some(wgpu::FragmentState {
            module,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &target_states,
        }),
        multiview_mask: None,
        cache: None,
    };
    device.create_render_pipeline(&pipeline_descriptor)
}

fn record_erosion_pass(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::RenderPipeline,
    globals: &wgpu::BindGroup,
    inputs: &wgpu::BindGroup,
    stage: &wgpu::BindGroup,
    target: &wgpu::TextureView,
) {
    record_erosion_pass_targets(encoder, pipeline, globals, inputs, stage, &[target]);
}

/// One full-screen simulation pass into every view of `targets`, in
/// `@location` order.
fn record_erosion_pass_targets(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::RenderPipeline,
    globals: &wgpu::BindGroup,
    inputs: &wgpu::BindGroup,
    stage: &wgpu::BindGroup,
    targets: &[&wgpu::TextureView],
) {
    let color_attachments: Vec<Option<wgpu::RenderPassColorAttachment>> = targets
        .iter()
        .map(|target| {
            Some(wgpu::RenderPassColorAttachment {
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
    let pass_descriptor = wgpu::RenderPassDescriptor {
        label: Some("erosion_pass"),
        color_attachments: &color_attachments,
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    };
    let mut pass = encoder.begin_render_pass(&pass_descriptor);
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, globals, &[]);
    pass.set_bind_group(1, inputs, &[]);
    pass.set_bind_group(2, stage, &[]);
    pass.draw(0..3, 0..1);
}

// ---------------------------------------------------------------------------
// Readback helpers
// ---------------------------------------------------------------------------

fn take_staging_buffer(device: &wgpu::Device, pool: &Mutex<Vec<wgpu::Buffer>>) -> wgpu::Buffer {
    if let Ok(mut pooled) = pool.lock() {
        if let Some(buffer) = pooled.pop() {
            return buffer;
        }
    }
    let buffer_descriptor = wgpu::BufferDescriptor {
        label: Some("erosion_readback_staging"),
        size: SIM_READBACK_BYTES,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    };
    device.create_buffer(&buffer_descriptor)
}

/// Maps one staging buffer and hands the pixels to the slots. wgpu invokes
/// the callback from `device.poll` on the worker thread; it must not touch
/// the device, the queue or any GPU resource, so it only copies bytes,
/// returns the buffer to the pool and records the result.
fn map_readback(
    buffer: wgpu::Buffer,
    slots: Arc<Mutex<ReadbackSlots>>,
    pool: Arc<Mutex<Vec<wgpu::Buffer>>>,
    in_flight: Arc<AtomicUsize>,
    half: ReadbackHalf,
) {
    let callback_buffer = buffer.clone();
    let on_mapped = move |result| {
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
        // Last, so a zero count means every half is already stored.
        in_flight.fetch_sub(1, Ordering::SeqCst);
    };
    buffer.map_async(wgpu::MapMode::Read, .., on_mapped);
}

#[cfg(test)]
mod worker_tests {
    use super::*;
    use crate::erosion::{tile, tile_simulation_minimum, InitTileCommand};

    /// The worker compiles the published WGSL verbatim; the sim shaders
    /// declare the shared globals uniform but may never read it, because the
    /// worker binds a zero-filled dummy instead of the render world's buffer.
    #[test]
    fn erosion_sim_shaders_never_read_the_globals_uniform() {
        let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        for name in ["init", "flux", "water", "terrain", "thermal"] {
            let path = manifest.join(format!("assets/shaders/erosion-{name}.wgsl"));
            let source = std::fs::read_to_string(&path).unwrap_or_else(|err| {
                panic!("cannot read {}: {err}", path.display())
            });
            assert!(
                !source.contains("globals."),
                "{} reads `globals.`; the worker's zeroed dummy uniform would be visible",
                path.display()
            );
        }
    }

    #[test]
    fn chunk_plan_preserves_the_iteration_total() {
        assert!(chunk_plan(0).is_empty());
        assert_eq!(chunk_plan(6), vec![6]);
        assert_eq!(chunk_plan(24), vec![24]);
        assert_eq!(chunk_plan(25), vec![24, 1]);
        assert_eq!(chunk_plan(140), vec![24, 24, 24, 24, 24, 20]);
        for count in [0usize, 1, 6, 23, 24, 25, 139, 140, 144, 1000] {
            let chunks = chunk_plan(count);
            assert_eq!(chunks.iter().sum::<usize>(), count);
            for chunk in chunks {
                assert!(chunk <= EROSION_WORKER_MAX_ITERATIONS_PER_SUBMIT);
            }
        }
    }

    /// One command set — the app's prewarm shape: init and finalize in the
    /// same frame — driven through the real thread, its own device, the
    /// full 140-iteration chunked submission and the readback maps. The tile
    /// must finalize with finite diagnostics and land one atlas patch.
    #[test]
    fn worker_delivers_a_tile_end_to_end() {
        let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let sources = ErosionShaderSources {
            init: std::fs::read(manifest.join("assets/shaders/erosion-init.wgsl")).unwrap(),
            flux: std::fs::read(manifest.join("assets/shaders/erosion-flux.wgsl")).unwrap(),
            water: std::fs::read(manifest.join("assets/shaders/erosion-water.wgsl")).unwrap(),
            terrain: std::fs::read(manifest.join("assets/shaders/erosion-terrain.wgsl")).unwrap(),
            thermal: std::fs::read(manifest.join("assets/shaders/erosion-thermal.wgsl")).unwrap(),
        };
        let instance = wgpu::Instance::default();
        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .expect("no GPU adapter for the erosion worker test");
        let bridge = ErosionBridge::default();
        bridge.publish_shader_sources(sources);
        assert!(spawn_erosion_worker(
            &bridge,
            &adapter,
            adapter.features(),
            wgpu::Limits::default()
        ));

        let key = tile(1, 2);
        let frame_commands = ErosionFrameCommands {
            sim_min: tile_simulation_minimum(key),
            init: Some(InitTileCommand {
                key,
                base_height: vec![24.0; SIM_PIXELS],
                drainage_area: vec![1.0; SIM_PIXELS],
                river: vec![0.0; SIM_PIXELS],
            }),
            iterate: None,
            finalize: true,
        };
        bridge.set_frame(frame_commands, None);

        for _ in 0..3000 {
            if let Some(why) = bridge.take_worker_dead() {
                panic!("worker died before finalizing: {why}");
            }
            let events = bridge.drain_events();
            let finalized = events.iter().find_map(|event| match event {
                ErosionEvent::TileFinalized(tile) => Some(tile),
                ErosionEvent::ReadbackFailed(key) => {
                    panic!("the test tile's readback failed: ({}, {})", key.x, key.z)
                }
            });
            if let Some(tile) = finalized {
                assert_eq!(tile.key, key);
                assert_eq!(tile.cpu_delta.len(), EROSION_OUTPUT_RESOLUTION * EROSION_OUTPUT_RESOLUTION);
                for value in [
                    tile.stats.minimum_height,
                    tile.stats.maximum_height,
                    tile.stats.land_coverage,
                    tile.stats.maximum_incision,
                    tile.stats.maximum_deposition,
                    tile.stats.erosion_detail,
                    tile.stats.flow_axis_bias,
                    tile.stats.maximum_drainage,
                    tile.stats.bedrock_exposure,
                    tile.stats.mean_loose_cover,
                ] {
                    assert!(value.is_finite(), "{value} is not finite");
                }
                let patches = bridge.drain_patches();
                assert_eq!(patches.len(), 1);
                assert_eq!(patches[0].key, key);
                assert_eq!(patches[0].atlas_height.len(), EROSION_ATLAS_PITCH * EROSION_ATLAS_PITCH * 4);
                assert_eq!(patches[0].atlas_flow.len(), EROSION_ATLAS_PITCH * EROSION_ATLAS_PITCH * 4);
                return;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        panic!("the worker never finalized the test tile (30 s)");
    }
}