//! Hydraulic erosion tile cache orchestration (CPU side), ported from the
//! C++ `main.cpp` erosion functions.
//!
//! The GPU simulation runs on a dedicated worker device; this module owns
//! the tile state machine, blending math for CPU height queries, and the
//! per-frame scheduling that decides which tile initializes, streams
//! iterations, and finalizes. Main and render worlds communicate through
//! [`ErosionBridge`].

use bevy::ecs::prelude::Resource;
use bevy::tasks::{AsyncComputeTaskPool, Task, block_on, poll_once};
use forest_core::constants::*;
use forest_core::noise::NoiseField;
use std::collections::HashMap;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Weak};

#[derive(Clone, Copy, Default, PartialEq, Eq, Debug, Hash)]
pub struct TileKey {
    pub x: i64,
    pub z: i64,
}

/// Identifies one simulation attempt, including retries of the same tile.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SimulationId(u64);

impl SimulationId {
    fn next() -> Self {
        static NEXT_SIMULATION_ID: AtomicU64 = AtomicU64::new(1);
        let mut current = NEXT_SIMULATION_ID.load(Ordering::Relaxed);
        loop {
            let next = current
                .checked_add(1)
                .expect("erosion simulation identifiers are exhausted");
            match NEXT_SIMULATION_ID.compare_exchange_weak(
                current,
                next,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return Self(current),
                Err(observed) => current = observed,
            }
        }
    }
}

pub const fn tile(x: i64, z: i64) -> TileKey {
    TileKey { x, z }
}

/// Cache windows are centred on the nearest pass centre. Keeping this
/// distinct from the lower candidate cell makes their support symmetric as
/// the player crosses the 512 m lattice.
pub fn world_tile(x: f32, z: f32) -> TileKey {
    tile(
        (x as f64 / EROSION_TILE_STRIDE as f64 + 0.5).floor() as i64,
        (z as f64 / EROSION_TILE_STRIDE as f64 + 0.5).floor() as i64,
    )
}

pub fn erosion_candidate_minimum(x: f32, z: f32) -> TileKey {
    tile(
        (x as f64 / EROSION_TILE_STRIDE as f64).floor() as i64,
        (z as f64 / EROSION_TILE_STRIDE as f64).floor() as i64,
    )
}

/// World-space origin of a tile's full simulation domain (retained square
/// plus discarded halo on every side).
pub fn tile_simulation_minimum(key: TileKey) -> [f32; 2] {
    let retained_half_size = EROSION_FOOTPRINT_SIZE * 0.5;
    [
        (key.x as f64 * EROSION_TILE_STRIDE as f64
            - retained_half_size as f64
            - EROSION_SIMULATION_HALO as f64) as f32,
        (key.z as f64 * EROSION_TILE_STRIDE as f64
            - retained_half_size as f64
            - EROSION_SIMULATION_HALO as f64) as f32,
    ]
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ErosionTileState {
    Queued,
    Simulating,
    Ready,
    Failed,
}

#[derive(Clone, Debug)]
pub struct ErosionTile {
    pub key: TileKey,

    /// The attempt producing this tile; zero until simulation begins.
    pub simulation_id: SimulationId,

    pub atlas_slot: usize,
    pub completed_iterations: usize,
    pub failed_readbacks: usize,
    pub reveal: f32,
    pub state: ErosionTileState,

    /// This is the independent result for one hydraulic pass. It deliberately
    /// remains distinct from its three overlapping neighbours.
    pub cpu_delta: Vec<f32>,

    /// The finished bed's 12 m ring concavity per retained texel, metres (+ in
    /// hollows): the surface atlas's G, which the snow pack reads, so the
    /// player's footing on snow matches the pack the terrain draws.
    pub cpu_concavity: Vec<f32>,
}

impl ErosionTile {
    fn new(key: TileKey, atlas_slot: usize) -> Self {
        Self {
            key,
            simulation_id: SimulationId::default(),
            atlas_slot,
            completed_iterations: 0,
            failed_readbacks: 0,
            reveal: 0.0,
            state: ErosionTileState::Queued,
            cpu_delta: Vec::new(),
            cpu_concavity: Vec::new(),
        }
    }
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + t * (b - a)
}

fn smoothstep(edge0: f32, edge1: f32, value: f32) -> f32 {
    let t = ((value - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// UV of a world position inside a tile's retained 1024 m support square.
fn erosion_support_uv(x: f32, z: f32, key: TileKey) -> [f32; 2] {
    let half_footprint = EROSION_FOOTPRINT_SIZE as f64 * 0.5;
    let support_x = key.x as f64 * EROSION_TILE_STRIDE as f64 - half_footprint;
    let support_z = key.z as f64 * EROSION_TILE_STRIDE as f64 - half_footprint;
    [
        ((x as f64 - support_x) / EROSION_FOOTPRINT_SIZE as f64) as f32,
        ((z as f64 - support_z) / EROSION_FOOTPRINT_SIZE as f64) as f32,
    ]
}

/// Hermite tent weight for the candidate tile at a world position, sampled
/// from the CPU copies of the blend mask texture.
pub fn sample_erosion_blend_mask(mask: &[f32], key: TileKey, x: f32, z: f32) -> f32 {
    if mask.is_empty() {
        return 0.0;
    }
    let uv = erosion_support_uv(x, z, key);
    if uv[0] < 0.0 || uv[1] < 0.0 || uv[0] > 1.0 || uv[1] > 1.0 {
        return 0.0;
    }
    let pixel_x = uv[0] * EROSION_OUTPUT_RESOLUTION as f32 - 0.5;
    let pixel_z = uv[1] * EROSION_OUTPUT_RESOLUTION as f32 - 0.5;
    let raw_x = pixel_x.floor() as i32;
    let raw_z = pixel_z.floor() as i32;
    let clamp_i = |v: i32, max: usize| v.clamp(0, max as i32) as usize;
    let x0 = clamp_i(raw_x, EROSION_OUTPUT_RESOLUTION - 1);
    let z0 = clamp_i(raw_z, EROSION_OUTPUT_RESOLUTION - 1);
    let x1 = clamp_i(raw_x + 1, EROSION_OUTPUT_RESOLUTION - 1);
    let z1 = clamp_i(raw_z + 1, EROSION_OUTPUT_RESOLUTION - 1);
    let tx = (pixel_x - pixel_x.floor()).clamp(0.0, 1.0);
    let tz = (pixel_z - pixel_z.floor()).clamp(0.0, 1.0);
    let at = |px: usize, pz: usize| mask[pz * EROSION_OUTPUT_RESOLUTION + px];
    lerp(
        lerp(at(x0, z0), at(x1, z0), tx),
        lerp(at(x0, z1), at(x1, z1), tx),
        tz,
    )
}

/// One of a tile's retained per-texel fields at a world position, bilinear
/// between texel centres and clamped at the footprint's edge, as the shaders
/// sample its atlas patch.
fn sample_tile_field(key: TileKey, field: &[f32], x: f32, z: f32) -> f32 {
    if field.is_empty() {
        return 0.0;
    }
    // The C++ widens only the lattice product to double before narrowing:
    // `static_cast<float>(static_cast<double>(tile.key.x) * kErosionTileStride
    // - kErosionFootprintSize * 0.5)`; every later step is f32.
    let support_x =
        (key.x as f64 * EROSION_TILE_STRIDE as f64 - EROSION_FOOTPRINT_SIZE as f64 * 0.5) as f32;
    let support_z =
        (key.z as f64 * EROSION_TILE_STRIDE as f64 - EROSION_FOOTPRINT_SIZE as f64 * 0.5) as f32;
    let pixel_x = (x - support_x) / EROSION_CELL_SIZE - 0.5;
    let pixel_z = (z - support_z) / EROSION_CELL_SIZE - 0.5;
    let raw_x = pixel_x.floor() as i32;
    let raw_z = pixel_z.floor() as i32;
    let clamp_i = |v: i32, max: usize| v.clamp(0, max as i32) as usize;
    let x0 = clamp_i(raw_x, EROSION_OUTPUT_RESOLUTION - 1);
    let z0 = clamp_i(raw_z, EROSION_OUTPUT_RESOLUTION - 1);
    let x1 = clamp_i(raw_x + 1, EROSION_OUTPUT_RESOLUTION - 1);
    let z1 = clamp_i(raw_z + 1, EROSION_OUTPUT_RESOLUTION - 1);
    let tx = (pixel_x - pixel_x.floor()).clamp(0.0, 1.0);
    let tz = (pixel_z - pixel_z.floor()).clamp(0.0, 1.0);
    let at = |px: usize, pz: usize| field[pz * EROSION_OUTPUT_RESOLUTION + px];
    lerp(
        lerp(at(x0, z0), at(x1, z0), tx),
        lerp(at(x0, z1), at(x1, z1), tx),
        tz,
    )
}

/// The lattice the erosion fade's centre snaps to, metres.
pub const EROSION_FADE_SNAP: f32 = 8.0;

/// The centre of the erosion fade for a player standing at `player_xz`:
/// their position snapped to the nearest point of an `EROSION_FADE_SNAP`
/// lattice, as the clipmap's origin snaps to its own. Every consumer of the
/// fade centres it here (the clipmap, the lighting heightfield, the plants'
/// roots, and the CPU's footing, collision and overlays), so they all agree
/// exactly, and the faded band moves only when the player crosses into the
/// next lattice cell, where any step used to move it across the whole
/// lighting heightfield. The anchor lies at most 4·√2 = 5.66 m from the
/// player. The fade's ramp falls at most 0.003 per metre, so that moves the
/// faded erosion by at most 0.017 of its delta (0.19 m of the largest, 11 m),
/// only 1.1 to 1.6 km away, where it is under a pixel; nearer, the fade is 1
/// either way.
pub fn erosion_fade_anchor(player_xz: [f32; 2]) -> [f32; 2] {
    player_xz.map(|value| (value / EROSION_FADE_SNAP + 0.5).floor() * EROSION_FADE_SNAP)
}

#[cfg(test)]
#[path = "fade_anchor_tests.rs"]
mod fade_anchor_tests;

/// Radial fade of erosion influence beyond the reliable cache window.
pub fn erosion_visibility(x: f32, z: f32, center: [f32; 2]) -> f32 {
    let dx = x - center[0];
    let dz = z - center[1];
    1.0 - smoothstep(
        EROSION_VISIBILITY_FULL_RADIUS,
        EROSION_VISIBILITY_ZERO_RADIUS,
        (dx * dx + dz * dz).sqrt(),
    )
}

/// One per-tile field blended over the four candidate passes at a world
/// position, as the shaders' erosionSurface blends the atlas: four fixed
/// spatial candidate masks are normalized independently of readiness, so a
/// missing or streaming pass contributes zero (the procedural base) and fades
/// in with its own C1 mask, avoiding 512 m quartet-availability steps.
fn blend_tiles(cache: &ErosionCache, x: f32, z: f32, field: fn(&ErosionTile) -> &[f32]) -> f32 {
    let lower = erosion_candidate_minimum(x, z);
    let candidates = [
        lower,
        tile(lower.x + 1, lower.z),
        tile(lower.x, lower.z + 1),
        tile(lower.x + 1, lower.z + 1),
    ];
    let mut masks = [0.0f32; 4];
    let mut total_weight = 0.0f32;
    for (index, key) in candidates.iter().enumerate() {
        masks[index] = sample_erosion_blend_mask(&cache.blend_mask_samples, *key, x, z);
        total_weight += masks[index];
    }

    let mut blended = 0.0f32;
    if total_weight > 0.000001 {
        let inverse_weight = 1.0 / total_weight;
        for (index, key) in candidates.iter().enumerate() {
            if let Some(found) = cache.tiles.get(key)
                && found.state == ErosionTileState::Ready
            {
                let reveal = found.reveal.clamp(0.0, 1.0);
                blended += masks[index]
                    * inverse_weight
                    * reveal
                    * sample_tile_field(found.key, field(found), x, z);
            }
        }
    }
    blended
}

/// Sample the blended eroded surface height at a world position, as the
/// shaders' terrainHeight draws it.
pub fn sample_eroded_height(
    cache: &ErosionCache,
    noise: &NoiseField,
    x: f32,
    z: f32,
    visibility_center: [f32; 2],
) -> f32 {
    let envelope = cache
        .rivers
        .as_ref()
        .map_or(forest_water::rivers::carve::Envelope::NONE, |network| {
            network.envelope(x, z)
        });
    // Beyond the fade's zero radius the delta adds nothing, so its tiles are
    // not sampled at all, as the shaders' fadedErosionSurface skips them.
    let fade = erosion_visibility(x, z, visibility_center);
    let faded_delta = if fade > 0.0 {
        blend_tiles(cache, x, z, |found| &found.cpu_delta) * fade
    } else {
        0.0
    };
    // Erosion ran over the carved base; the channels hold over the result.
    let ground =
        envelope.clamp(envelope.clamp(forest_core::noise::base_height(noise, x, z)) + faded_delta);
    // The buildings' pads level whatever the erosion made of their ground.
    match &cache.buildings {
        Some(pads) => pads.level(x, z, ground),
        None => ground,
    }
}

/// The eroded bed's 12 m ring concavity at a world position, metres (+ in
/// hollows), blended and faded exactly as the shaders' fadedErosionSurface
/// returns it: what the snow pack reads under the player.
pub fn sample_eroded_concavity(
    cache: &ErosionCache,
    x: f32,
    z: f32,
    visibility_center: [f32; 2],
) -> f32 {
    let fade = erosion_visibility(x, z, visibility_center);
    if fade > 0.0 {
        blend_tiles(cache, x, z, |found| &found.cpu_concavity) * fade
    } else {
        0.0
    }
}

// ---------------------------------------------------------------------------
// Main <-> render bridge
// ---------------------------------------------------------------------------

/// Per-frame GPU work the erosion node must run for the active tile. Written
/// by the main-world coordinator, extracted, then executed in the render
/// world in a fully prescribed order.
pub struct ErosionFrameCommands {
    /// Identifies the active attempt independently of its world coordinates.
    pub simulation_id: SimulationId,

    /// The active tile's simulation-domain minimum (uWorldMin).
    pub sim_min: [f32; 2],

    /// A pending tile initialization: base height map for the simulation
    /// domain, uploaded then stamped into both terrain/water/flux targets.
    pub init: Option<InitTileCommand>,

    /// Number of flux/water/terrain iterations for the active tile this frame.
    pub iterate: Option<IterateCommand>,

    /// Read back the active tile's current terrain and water states and
    /// publish the finalize result as an event.
    pub finalize: bool,
}

pub struct InitTileCommand {
    pub key: TileKey,
    pub base_height: Vec<f32>,

    /// Contributing area of every base cell (see [`route_base_drainage`]),
    /// so stream power acts on whole catchments from the first iteration.
    pub drainage_area: Vec<f32>,

    /// 1 where a river's water covers the cell: held fixed, and a sink.
    pub river: Vec<f32>,
}

pub struct IterateCommand {
    pub count: usize,
    pub settings: ErosionSettings,
}

/// A finished atlas patch crossing from the erosion worker thread to the
/// render world, waiting for the main world to publish its slot through the
/// lookup records (`flush_pending_atlas` uploads once it does).
///
/// The 260x260 patches keep their Rust float layout across the bridge; the
/// render side runs them through `write_padded_at`'s `cast_slice` at upload.
pub struct PendingAtlasPatch {
    pub key: TileKey,

    /// The attempt that produced these atlas texels.
    pub simulation_id: SimulationId,

    pub atlas_height: Vec<f32>,
    pub atlas_flow: Vec<f32>,

    /// Frames spent waiting on the render side; dropped past
    /// `MAX_PENDING_ATLAS_AGE`, incremented by the render-side flush.
    pub age: u32,
}

/// The five embedded erosion WGSL sources, published once by the main world
/// and taken once by the worker thread, which compiles
/// them into shader modules on its own device.
///
/// As bytes because that is what crosses threads and what
/// `create_shader_module` takes; every erosion shader is a self-contained
/// WGSL file, so no preprocessing runs between here and the device.
#[derive(Clone, Debug)]
pub struct ErosionShaderSources {
    pub init: Vec<u8>,
    pub flux: Vec<u8>,
    pub water: Vec<u8>,
    pub terrain: Vec<u8>,
    pub thermal: Vec<u8>,
}

/// A finalized tile's data as computed by the render world from its readback.
///
/// The 260x260 atlas patches stay render-side: the C++ wrote them straight
/// into `erosion.heightAtlas`/`flowAtlas` inside `FinalizeErosionTile`, so the
/// port uploads them from the render world (`flush_pending_atlas`) and only
/// the CPU collision field and the summary statistics cross the bridge.
pub struct FinalizedTile {
    pub key: TileKey,

    /// Signed displacement per retained texel, for CPU collision.
    pub cpu_delta: Vec<f32>,

    /// The bed's 12 m ring concavity per retained texel (surface atlas G).
    pub cpu_concavity: Vec<f32>,

    pub stats: TileDiagnostics,
}

pub enum ErosionEvent {
    TileFinalized {
        simulation_id: SimulationId,
        tile: FinalizedTile,
    },
    ReadbackFailed {
        key: TileKey,
        simulation_id: SimulationId,
    },
}

/// Inner state behind the bridge's mutex, shared by the main world (tile
/// scheduling), the render world (lookup writes, atlas uploads), and the
/// erosion worker thread.
#[derive(Default)]
pub struct BridgeState {
    commands: Option<ErosionFrameCommands>,
    lookup_records: Option<Vec<f32>>,
    events: VecDeque<ErosionEvent>,
    phase_costs: ErosionPhaseCosts,

    /// Finished atlas patches the worker published; `flush_pending_atlas`
    /// drains and uploads them once their tile's slot is published.
    patches: VecDeque<PendingAtlasPatch>,

    /// Why the worker thread ended, set only by the worker. `apply_erosion_events`
    /// panics on consumption: a dead worker is a dead simulation, and the
    /// render thread never had a way to fail that quietly (it panicked the
    /// process the same way).
    worker_dead: Option<String>,

    /// The five WGSL sources, published once by the main world and taken once
    /// by the worker spawn; see [`ErosionShaderSources`].
    shader_sources: Option<ErosionShaderSources>,
}

/// Cumulative render-node phase costs in microseconds for `--shot` frame-time
/// attribution: the render node records them each frame, and the main world's
/// frame-time recorder drains the totals. Both sides only touch this under the
/// bridge's mutex.
#[derive(Clone, Copy, Debug, Default)]
pub struct ErosionPhaseCosts {
    /// Writing the 11x11 lookup records and flushing atlas patches into
    /// `GpuWorldTextures`.
    pub lookup_atlas_us: u64,

    /// `begin_mapping` + `device.poll` (map callbacks fire here).
    pub poll_us: u64,

    /// `consume_readback` outside its finalize math: assembling readback
    /// halves and pushing atlas patches.
    pub consume_us: u64,

    /// `finalize_erosion_tile` CPU math (crop, statistics, patch assembly).
    pub finalize_us: u64,

    /// Recording the sim frame's commands (init passes, 4 passes per
    /// iteration, readback copies).
    pub sim_record_us: u64,
}

impl std::ops::AddAssign<&ErosionPhaseCosts> for ErosionPhaseCosts {
    fn add_assign(&mut self, rhs: &Self) {
        self.lookup_atlas_us += rhs.lookup_atlas_us;
        self.poll_us += rhs.poll_us;
        self.consume_us += rhs.consume_us;
        self.finalize_us += rhs.finalize_us;
        self.sim_record_us += rhs.sim_record_us;
    }
}

/// Shared state between the main world (tile scheduling), the render world
/// (GPU pass execution + readbacks), and the erosion worker thread.
///
/// The condvar is a sibling of the guarded state rather than a field of it:
/// `MutexGuard` cannot be moved into `Condvar::wait_timeout` while the call's
/// receiver borrows back into the guarded struct — the pair has to live
/// side by side, which is also the std docs' pattern.
#[derive(Clone, Default, Resource)]
pub struct ErosionBridge {
    state: std::sync::Arc<std::sync::Mutex<BridgeState>>,

    /// Signalled whenever a command set arrives; the worker waits on it when
    /// there is neither readback work nor a pending command set, and
    /// `set_frame` notifies inside the lock, so the classic lost-wakeup race
    /// between a released waiter and a notify cannot happen. An `Arc`: every
    /// clone of the bridge shares one condvar, so a parked worker sees every
    /// notify from any holder.
    work_available: std::sync::Arc<std::sync::Condvar>,
}

impl ErosionBridge {
    fn state(&self) -> std::sync::MutexGuard<'_, BridgeState> {
        self.state.lock().unwrap()
    }
    pub fn take_commands(&self) -> Option<ErosionFrameCommands> {
        self.state().commands.take()
    }
    pub fn take_lookup(&self) -> Option<Vec<f32>> {
        self.state().lookup_records.take()
    }
    pub fn drain_events(&self) -> VecDeque<ErosionEvent> {
        std::mem::take(&mut self.state().events)
    }

    /// Hands one frame of work to the erosion worker thread.
    ///
    /// The worker consumes at most one command set per loop tick, and only
    /// once its device, pipelines and textures exist — for the first seconds
    /// of the app the bridge accumulates sets while the worker thread
    /// compiles its shaders. Whatever is still pending when the next frame
    /// arrives is therefore *merged into* it rather than replaced.
    ///
    /// Without the merge a dropped frame takes its work with it, and `init` is
    /// the one command that cannot be re-sent: `BeginErosionTile` is called
    /// exactly once per tile, so a lost init leaves the main world's cache
    /// holding an active tile in `Simulating` that the render world has never
    /// heard of. The node then bails at its `active_tile.is_none() &&
    /// init.is_none()` guard every frame and no tile ever finalizes — the
    /// whole erosion cache stays empty. Iteration counts are additive (the
    /// main world advances its own `completed_iterations` as it sends them, so
    /// each frame's count is fresh work) and `finalize` is idempotent, so both
    /// survive the same way.
    pub fn set_frame(&self, mut commands: ErosionFrameCommands, lookup_records: Option<Vec<f32>>) {
        let mut state = self.state();
        // Only frames for the same attempt can share work. A reset or retry
        // may reuse both the tile key and its world-space minimum.
        let same_attempt = state.commands.as_ref().is_some_and(|pending| {
            pending.simulation_id == commands.simulation_id && pending.sim_min == commands.sim_min
        });
        if let Some(pending) = state.commands.take().filter(|_| same_attempt) {
            if commands.init.is_none() {
                // `sim_min` belongs to the newest frame's active tile, which
                // is the same tile a carried-over init refers to: the main
                // world only clears `has_active_tile` once the finalize event
                // comes back, so its `sim_min` still describes that tile.
                commands.init = pending.init;
            }
            match (commands.iterate.as_mut(), pending.iterate) {
                (Some(current), Some(earlier)) => current.count += earlier.count,
                (None, Some(earlier)) => commands.iterate = Some(earlier),
                _ => {}
            }
            commands.finalize |= pending.finalize;
        }
        state.commands = Some(commands);
        state.lookup_records = lookup_records;
        // The worker may be parked on this condvar with nothing to do; every
        // frame carries commands unless simulation is paused, and a parked
        // worker costs two frames of latency per missed wake-up. Notifying
        // while still holding the lock is deliberate: it makes the wake-up
        // ordered against this frame's state change.
        self.work_available.notify_one();
    }
    pub fn push_event(&self, event: ErosionEvent) {
        self.state().events.push_back(event);
    }
    /// Hands a finished atlas patch to the render world. The deque is capped:
    /// a render side that stops draining must not grow the bridge unbounded,
    /// so the oldest patch is dropped with a warning past
    /// `EROSION_PENDING_PATCH_CAP` — the tile it belonged to re-streams
    /// through the normal queue rather than stalling everything behind it.
    pub fn push_patches(
        &self,
        key: TileKey,
        simulation_id: SimulationId,
        atlas_height: Vec<f32>,
        atlas_flow: Vec<f32>,
    ) {
        let mut state = self.state();
        state.patches.push_back(PendingAtlasPatch {
            key,
            simulation_id,
            atlas_height,
            atlas_flow,
            age: 0,
        });
        while state.patches.len() > EROSION_PENDING_PATCH_CAP {
            state.patches.pop_front();
            log::warn!(
                "EROSION: dropped tile ({}, {})'s atlas patch past the pending cap; the render side is not draining",
                key.x,
                key.z
            );
        }
    }
    /// Drains every pending atlas patch, oldest first.
    pub fn drain_patches(&self) -> VecDeque<PendingAtlasPatch> {
        std::mem::take(&mut self.state().patches)
    }
    /// Publishes the five WGSL sources; consumed once by the worker spawn.
    ///
    /// # Safety
    ///
    /// Every supplied shader must keep all memory accesses in bounds for
    /// every invocation made by the erosion worker. The worker disables
    /// shader runtime checks to preserve the simulation's numerical results.
    pub unsafe fn publish_shader_sources(&self, sources: ErosionShaderSources) {
        self.state().shader_sources = Some(sources);
    }
    pub fn take_shader_sources(&self) -> Option<ErosionShaderSources> {
        self.state().shader_sources.take()
    }
    /// True once the WGSL publisher has run; the render node uses this to
    /// hold the worker spawn off until the sources exist (the worker treats
    /// an absence at spawn as a wiring bug and dies loudly).
    pub fn shader_sources_ready(&self) -> bool {
        self.state().shader_sources.is_some()
    }
    /// Blocks the calling thread up to `timeout`, returning as soon as
    /// `set_frame` signals work. The worker thread's idle wait: a timed wait
    /// rather than a bare one, so a notification that raced past a wake
    /// (spurious, or consumed by an earlier branch) costs at most `timeout`,
    /// and never the two frames a full sleep round trip would.
    pub fn wait_for_work(&self, timeout: std::time::Duration) {
        let state = self.state();
        let _ = self.work_available.wait_timeout(state, timeout);
    }
    /// Records that the worker thread can no longer progress.
    pub fn set_worker_dead(&self, why: String) {
        self.state().worker_dead = Some(why);
    }
    /// Returns the worker's death reason once; applying erosion events then panics.
    pub fn take_worker_dead(&self) -> Option<String> {
        self.state().worker_dead.take()
    }
    /// Adds one render frame's phase costs to the bridge's running totals.
    pub fn record_phase_costs(&self, costs: ErosionPhaseCosts) {
        self.state().phase_costs += &costs;
    }
    /// Takes the totals accumulated since the last drain (the recorder runs
    /// once per frame, so this is normally a single frame's worth).
    pub fn drain_phase_costs(&self) -> ErosionPhaseCosts {
        std::mem::take(&mut self.state().phase_costs)
    }
}

// ---------------------------------------------------------------------------
// Tile cache
// ---------------------------------------------------------------------------

/// The seven per-tile summary statistics the C++ keeps flat on
/// `HydraulicErosion` (minimumHeight .. flowAxisBias), plus the drainage and
/// loose-cover summaries of the thermal and fluvial passes; they describe the
/// most recently completed tile and are shown in the diagnostics window.
#[derive(Clone, Copy, Debug, Default)]
pub struct TileDiagnostics {
    pub minimum_height: f32,
    pub maximum_height: f32,
    pub land_coverage: f32,
    pub maximum_incision: f32,
    pub maximum_deposition: f32,
    pub erosion_detail: f32,
    pub flow_axis_bias: f32,

    /// Largest routed contributing area in the retained footprint, cells.
    pub maximum_drainage: f32,

    /// Share of land cells with less than 0.1 m of loose cover, percent.
    pub bedrock_exposure: f32,

    /// Mean loose cover over land cells, metres.
    pub mean_loose_cover: f32,

    /// Wall-clock milliseconds the GPU took to initialize the tile and run
    /// all its iterations, measured only where one command set does both
    /// (the prewarm) and 0 for a tile streamed over many frames.
    pub simulation_ms: f32,
}

/// CPU-side erosion tile cache: keys, readiness, reveal progress, and the
/// statistics exposed in the diagnostics window.
#[derive(Resource)]
pub struct ErosionCache {
    pub tiles: HashMap<TileKey, ErosionTile>,
    pub free_atlas_slots: Vec<usize>,
    pub blend_mask_samples: Vec<f32>,
    pub active_tile: TileKey,
    pub has_active_tile: bool,
    pub lookup_minimum: TileKey,
    pub stats: TileDiagnostics,
    pub ready: bool,

    /// Set by reset_erosion_cache so a fresh simulation starts from empty.
    pub cache_dropped: bool,

    /// Keys whose reveal jumps straight to 1.0 when the finalize event lands
    /// (spawn prewarm and UI-triggered re-runs).
    pub force_reveal: Option<TileKey>,

    /// The river network in force: tiles simulate over its carved channels
    /// and height queries hold its channels and banks.
    pub rivers: Option<Arc<forest_water::rivers::network::RiverNetwork>>,

    /// The buildings' pads in force: height queries stand on the ground
    /// they level, as the terrain draws it.
    pub buildings: Option<Arc<forest_core::building_pads::PadIndex>>,

    /// Frames the live active tile has run without a finalize or failure
    /// event (armed at begin, incremented in [`apply_erosion_events`], tripping
    /// `mark_tile_failed` past `EROSION_WORKER_WATCHDOG_FRAMES`).
    pub watchdog_frames: u32,
}

impl Default for ErosionCache {
    fn default() -> Self {
        Self {
            tiles: HashMap::new(),
            free_atlas_slots: (0..EROSION_ATLAS_SLOTS).rev().collect(),
            blend_mask_samples: Vec::new(),
            active_tile: tile(0, 0),
            has_active_tile: false,
            lookup_minimum: tile(0, 0),
            stats: TileDiagnostics::default(),
            ready: true,
            cache_dropped: false,
            force_reveal: None,
            rivers: None,
            buildings: None,
            watchdog_frames: 0,
        }
    }
}

pub fn ensure_erosion_cache(cache: &mut ErosionCache, center: TileKey) {
    let keep_tile = |key: &TileKey, existing: &mut ErosionTile| -> bool {
        let outside = (key.x - center.x).abs() > EROSION_STREAMING_RADIUS
            || (key.z - center.z).abs() > EROSION_STREAMING_RADIUS;
        let active = cache.has_active_tile && *key == cache.active_tile;
        if outside && !active {
            cache.free_atlas_slots.push(existing.atlas_slot);
            false
        } else {
            true
        }
    };
    cache.tiles.retain(keep_tile);
    for z in -EROSION_STREAMING_RADIUS..=EROSION_STREAMING_RADIUS {
        for x in -EROSION_STREAMING_RADIUS..=EROSION_STREAMING_RADIUS {
            let key = tile(center.x + x, center.z + z);
            if !cache.tiles.contains_key(&key) {
                if cache.free_atlas_slots.is_empty() {
                    continue;
                }
                let slot = cache.free_atlas_slots.pop().unwrap();
                cache.tiles.insert(key, ErosionTile::new(key, slot));
            }
        }
    }
    cache.lookup_minimum = tile(
        center.x - EROSION_LOOKUP_RADIUS,
        center.z - EROSION_LOOKUP_RADIUS,
    );
}

fn covers_player_cell(key: TileKey, player_minimum: TileKey) -> bool {
    (key.x == player_minimum.x || key.x == player_minimum.x + 1)
        && (key.z == player_minimum.z || key.z == player_minimum.z + 1)
}

fn choose_next_erosion_tile(
    cache: &ErosionCache,
    center: TileKey,
    player_minimum: TileKey,
) -> Option<TileKey> {
    let mut best_distance = f64::INFINITY;
    let mut best_priority = i32::MAX;
    let mut best_key = tile(0, 0);
    let mut found_any = false;
    for (key, existing) in &cache.tiles {
        if existing.state != ErosionTileState::Queued {
            continue;
        }
        let priority = if covers_player_cell(*key, player_minimum) {
            0
        } else {
            1
        };
        let dx = (key.x - center.x) as f64;
        let dz = (key.z - center.z) as f64;
        let distance = dx * dx + dz * dz;
        let better = priority < best_priority
            || (priority == best_priority && distance < best_distance)
            || (priority == best_priority
                && distance == best_distance
                && (key.z < best_key.z || (key.z == best_key.z && key.x < best_key.x)))
            || !found_any;
        if better {
            best_priority = priority;
            best_distance = distance;
            best_key = *key;
            found_any = true;
        }
    }
    found_any.then_some(best_key)
}

/// Build the 11x11 lookup texture records on the CPU: for each lattice cell
/// inside the window, the ready tile's atlas slot and reveal progress.
pub fn erosion_lookup_records(cache: &ErosionCache) -> Vec<f32> {
    let mut records = vec![0.0f32; EROSION_LOOKUP_DIAMETER * EROSION_LOOKUP_DIAMETER * 4];
    for (key, existing) in &cache.tiles {
        if existing.state != ErosionTileState::Ready {
            continue;
        }
        let relative_x = key.x - cache.lookup_minimum.x;
        let relative_z = key.z - cache.lookup_minimum.z;
        if relative_x < 0
            || relative_z < 0
            || relative_x >= EROSION_LOOKUP_DIAMETER as i64
            || relative_z >= EROSION_LOOKUP_DIAMETER as i64
        {
            continue;
        }
        let destination =
            ((relative_z as usize * EROSION_LOOKUP_DIAMETER) + relative_x as usize) * 4;
        records[destination] = (existing.atlas_slot % EROSION_ATLAS_COLUMNS) as f32;
        records[destination + 1] = (existing.atlas_slot / EROSION_ATLAS_COLUMNS) as f32;
        records[destination + 2] = existing.reveal;
        records[destination + 3] = 1.0;
    }
    records
}

/// A coherent render snapshot of atlas allocations and their simulation attempts.
#[derive(Clone, Debug, Default)]
pub struct ErosionLookup {
    pub minimum: TileKey,
    pub records: Vec<f32>,
    pub simulation_ids: Vec<SimulationId>,
}

/// Captures lookup records, their origin and matching attempts from one cache state.
pub fn erosion_lookup_snapshot(cache: &ErosionCache) -> ErosionLookup {
    let records = erosion_lookup_records(cache);
    let mut simulation_ids = vec![SimulationId::default(); EROSION_LOOKUP_DIAMETER.pow(2)];
    for (key, existing) in &cache.tiles {
        if existing.state != ErosionTileState::Ready {
            continue;
        }
        let relative_x = key.x - cache.lookup_minimum.x;
        let relative_z = key.z - cache.lookup_minimum.z;
        if !(0..EROSION_LOOKUP_DIAMETER as i64).contains(&relative_x)
            || !(0..EROSION_LOOKUP_DIAMETER as i64).contains(&relative_z)
        {
            continue;
        }
        let destination = relative_z as usize * EROSION_LOOKUP_DIAMETER + relative_x as usize;
        simulation_ids[destination] = existing.simulation_id;
    }
    ErosionLookup {
        minimum: cache.lookup_minimum,
        records,
        simulation_ids,
    }
}

pub fn reset_erosion_cache(cache: &mut ErosionCache) {
    cache.tiles.clear();
    cache.free_atlas_slots = (0..EROSION_ATLAS_SLOTS).rev().collect();
    cache.has_active_tile = false;
    cache.cache_dropped = true;
    cache.lookup_minimum = tile(0, 0);
    // A pending immediate-reveal belongs to a tile the reset just discarded.
    cache.force_reveal = None;
    cache.watchdog_frames = 0;
}

/// Contributing area, in cells and counting the cell itself, of every cell of
/// a square height grid. Each cell passes its area to its lower neighbours in
/// proportion to their squared slope, exactly the multiple-flow-direction
/// shares the GPU terrain pass routes every iteration, so the simulation
/// starts from its own converged drainage instead of growing it one cell per
/// iteration. Cells below sea level swallow what reaches them, flats and pits
/// keep theirs, and the grid edge is a closed wall.
pub fn route_base_drainage(heights: &[f32], resolution: usize, cell_size: f32) -> Vec<f32> {
    let count = resolution * resolution;
    let mut area = vec![1.0f32; count.min(heights.len())];
    if area.len() != count {
        return area;
    }
    // Every donor must hold all of its own area before it passes any on. The
    // routing runs strictly downhill, so it is acyclic: a cell is ready once
    // each higher neighbour draining into it has been routed. Counting those
    // donors and releasing cells as their count reaches zero (Kahn's
    // ordering) gives the same partial order as visiting cells highest first,
    // without sorting the whole tile on the main thread as each one begins.
    //
    // Each cell counts its own donors, the higher land cells around it: the
    // neighbour relation is symmetric, so that is the very integer the donors
    // used to add to it one by one, and the release order below is unchanged.
    let steps = interior_steps(resolution);
    let mut waiting = vec![0u8; count];
    for (cell, donors) in waiting.iter_mut().enumerate() {
        let height = heights[cell];
        for_each_neighbour_fast(cell, resolution, &steps, |neighbour, _| {
            let donor = heights[neighbour];
            if height < donor && donor >= SEA_LEVEL {
                *donors += 1;
            }
        });
    }
    let mut ready: Vec<u32> = (0..count as u32)
        .filter(|&cell| waiting[cell as usize] == 0)
        .collect();
    let diagonal = cell_size * std::f32::consts::SQRT_2;
    while let Some(cell) = ready.pop() {
        let cell = cell as usize;
        let height = heights[cell];
        if height < SEA_LEVEL {
            continue;
        }
        let mut shares = [(0usize, 0.0f32); 8];
        let mut receivers = 0;
        let mut share_total = 0.0f32;
        for_each_neighbour_fast(cell, resolution, &steps, |neighbour, diagonal_step| {
            let distance = if diagonal_step { diagonal } else { cell_size };
            let slope = (height - heights[neighbour]) / distance;
            if slope > 0.0 {
                shares[receivers] = (neighbour, slope * slope);
                receivers += 1;
                share_total += slope * slope;
            }
        });
        let passed = area[cell];
        for &(neighbour, share) in &shares[..receivers] {
            if share_total > 0.0 {
                area[neighbour] += passed * share / share_total;
            }
            waiting[neighbour] -= 1;
            if waiting[neighbour] == 0 {
                ready.push(neighbour as u32);
            }
        }
    }
    for (cell, height) in heights.iter().enumerate() {
        if *height < SEA_LEVEL {
            area[cell] = 0.0;
        }
    }
    area
}

/// The eight steps to a cell's neighbours, in the order every visit takes
/// them; the routing's float sums depend on it.
const NEIGHBOURS: [(i64, i64); 8] = [
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
];

/// [`NEIGHBOURS`] as index offsets on a grid `resolution` cells wide, with
/// whether each step is diagonal.
fn interior_steps(resolution: usize) -> [(isize, bool); 8] {
    NEIGHBOURS.map(|(dx, dz)| ((dz * resolution as i64 + dx) as isize, dx != 0 && dz != 0))
}

/// [`for_each_neighbour`] without its bounds tests away from the grid's
/// edge, where all eight neighbours exist and each is a fixed index offset
/// (`steps`, from [`interior_steps`]). The visits come in the same order.
fn for_each_neighbour_fast(
    cell: usize,
    resolution: usize,
    steps: &[(isize, bool); 8],
    mut visit: impl FnMut(usize, bool),
) {
    let (x, z) = (cell % resolution, cell / resolution);
    if x == 0 || z == 0 || x + 1 == resolution || z + 1 == resolution {
        for_each_neighbour(cell, resolution, visit);
        return;
    }
    for &(offset, diagonal) in steps {
        visit(cell.wrapping_add_signed(offset), diagonal);
    }
}

/// Calls `visit(neighbour, diagonal)` for each in-grid cell of the eight
/// surrounding `cell`; the grid edge is a closed wall.
fn for_each_neighbour(cell: usize, resolution: usize, mut visit: impl FnMut(usize, bool)) {
    let x = (cell % resolution) as i64;
    let z = (cell / resolution) as i64;
    for (dx, dz) in NEIGHBOURS {
        let (nx, nz) = (x + dx, z + dz);
        if nx < 0 || nz < 0 || nx >= resolution as i64 || nz >= resolution as i64 {
            continue;
        }
        visit(nz as usize * resolution + nx as usize, dx != 0 && dz != 0);
    }
}

/// A tile's CPU-built simulation inputs: its base heights and their routed
/// drainage.
pub struct PreparedTile {
    pub key: TileKey,
    pub base_height: Vec<f32>,
    pub drainage_area: Vec<f32>,
    pub river: Vec<f32>,
}

/// Builds a tile's simulation inputs: its base heights with the rivers
/// carved in, and their routed drainage. `workers` threads share the height
/// map (the routing is serial); the result is the same for any count.
pub fn prepare_tile(
    noise: &NoiseField,
    rivers: Option<&forest_water::rivers::network::RiverNetwork>,
    key: TileKey,
    workers: usize,
) -> PreparedTile {
    let started = std::time::Instant::now();
    let (base_height, river) = crate::create_base_height_map(noise, key, rivers, workers);
    let drainage_area = route_base_drainage(&base_height, EROSION_RESOLUTION, EROSION_CELL_SIZE);
    log::info!(
        "EROSION: tile ({}, {}) inputs in {:.0} ms on {} thread{}",
        key.x,
        key.z,
        started.elapsed().as_secs_f64() * 1000.0,
        workers,
        if workers == 1 { "" } else { "s" }
    );
    PreparedTile {
        key,
        base_height,
        drainage_area,
        river,
    }
}

/// Threads for a tile prepared inline (the prewarm, screenshot runs and the
/// overlap measurement), where the frame waits for it: every core.
fn inline_preparation_workers() -> usize {
    std::thread::available_parallelism()
        .map_or(4, |n| n.get())
        .clamp(1, 16)
}

/// Inputs being built on the task pool, with the river network they carve:
/// the network in force when the task began, or `None` for none. Held weakly,
/// which keeps a replaced network's allocation, so a later network can never
/// take its address and pass for it.
struct PendingTile {
    key: TileKey,
    rivers: Option<Weak<forest_water::rivers::network::RiverNetwork>>,
    task: Task<PreparedTile>,
}

impl PendingTile {
    /// Whether these inputs carve `rivers`, the network now in force.
    fn carves(&self, rivers: &Option<Arc<forest_water::rivers::network::RiverNetwork>>) -> bool {
        match (&self.rivers, rivers) {
            (None, None) => true,
            (Some(built), Some(current)) => std::ptr::eq(built.as_ptr(), Arc::as_ptr(current)),
            _ => false,
        }
    }
}

/// Builds the next tile's inputs on the async compute pool. Base heights and
/// their drainage routing take tens of milliseconds per tile, several times
/// that in a debug build, and a tile begins every couple of dozen frames while
/// the cache streams; done inline they stalled the frame each time. They are
/// built while the tile before simulates, so the next one can usually begin
/// the frame that one finalizes. One thread builds them, which leaves the
/// other cores to the main and render threads.
#[derive(Resource)]
pub struct TilePreparation {
    noise: Arc<NoiseField>,
    pending: Option<PendingTile>,
}

impl TilePreparation {
    pub fn new(noise: Arc<NoiseField>) -> Self {
        Self {
            noise,
            pending: None,
        }
    }

    /// Drops work that can no longer be begun: its tile has left the queue,
    /// or the rivers changed under it (a region switch), so its heights carve
    /// channels the tile would no longer simulate over.
    fn discard_stale(&mut self, cache: &ErosionCache) {
        let still_queued = |key: &TileKey| {
            cache
                .tiles
                .get(key)
                .is_some_and(|tile| tile.state == ErosionTileState::Queued)
        };
        let stale =
            |pending: &PendingTile| !still_queued(&pending.key) || !pending.carves(&cache.rivers);
        if self.pending.as_ref().is_some_and(stale) {
            self.pending = None;
        }
    }

    /// Starts building `key`'s inputs unless they are already in flight or
    /// built. Work for another tile is dropped: the player has moved and the
    /// queue's best tile with them, and the tile it was for, still queued, is
    /// prepared again when its turn comes. The best tile changes only when
    /// the player crosses a lattice line, so little is thrown away.
    fn start(&mut self, cache: &ErosionCache, key: TileKey) {
        self.discard_stale(cache);
        if self
            .pending
            .as_ref()
            .is_some_and(|pending| pending.key == key)
        {
            return;
        }
        let noise = self.noise.clone();
        let rivers = cache.rivers.clone();
        let built_for = rivers.as_ref().map(Arc::downgrade);
        let task = AsyncComputeTaskPool::get()
            .spawn(async move { prepare_tile(&noise, rivers.as_deref(), key, 1) });
        self.pending = Some(PendingTile {
            key,
            rivers: built_for,
            task,
        });
    }

    /// `next`'s prepared inputs, once they are ready. Starts preparing it
    /// when they are not in flight; work for any other tile is dropped, so
    /// the tile begun is always the queue's best, never one chosen before the
    /// player moved. Work that carves rivers no longer in force is dropped
    /// too.
    fn poll(&mut self, cache: &ErosionCache, next: TileKey) -> Option<PreparedTile> {
        self.discard_stale(cache);
        let Some(mut pending) = self.pending.take_if(|pending| pending.key == next) else {
            self.start(cache, next);
            return None;
        };
        let prepared = block_on(poll_once(&mut pending.task));
        if prepared.is_none() {
            self.pending = Some(pending);
        }
        prepared
    }
}

/// `BeginErosionTile`: stage one tile's base height map for GPU initialization
/// (the render world uploads it and stamps the simulation targets), then mark
/// it as the active simulation. Shared by the per-frame scheduler and the
/// windowless overlap-measurement driver.
pub fn begin_tile(
    cache: &mut ErosionCache,
    commands: &mut ErosionFrameCommands,
    noise: &NoiseField,
    key: TileKey,
) {
    if !cache.tiles.contains_key(&key) {
        return;
    }
    let prepared = prepare_tile(
        noise,
        cache.rivers.as_deref(),
        key,
        inline_preparation_workers(),
    );
    begin_prepared_tile(cache, commands, prepared);
}

/// [`begin_tile`] with inputs already built, synchronously or by
/// [`TilePreparation`].
pub fn begin_prepared_tile(
    cache: &mut ErosionCache,
    commands: &mut ErosionFrameCommands,
    prepared: PreparedTile,
) {
    let key = prepared.key;
    if !cache.tiles.contains_key(&key) {
        return;
    }
    let simulation_id = SimulationId::next();
    commands.simulation_id = simulation_id;
    commands.sim_min = tile_simulation_minimum(key);
    commands.init = Some(InitTileCommand {
        key,
        base_height: prepared.base_height,
        drainage_area: prepared.drainage_area,
        river: prepared.river,
    });
    if let Some(existing) = cache.tiles.get_mut(&key) {
        existing.simulation_id = simulation_id;
        existing.state = ErosionTileState::Simulating;
        existing.completed_iterations = 0;
        existing.reveal = 0.0;
    }
    cache.active_tile = key;
    cache.has_active_tile = true;
    // Arms the main-world watchdog: `apply_erosion_events` counts frames with
    // no delivery from here.
    cache.watchdog_frames = 0;
    log::info!("EROSION: generating tile ({}, {})", key.x, key.z);
}

/// `RunErosionIterations` + the finalize request: feed the active tile
/// `iteration_budget` more iterations and request a readback when complete.
/// The bridge carries exactly one frame's commands, so systems using these
/// helpers must run once per frame.
pub fn advance_active_tile(
    cache: &mut ErosionCache,
    commands: &mut ErosionFrameCommands,
    settings: &ErosionSettings,
    iteration_budget: usize,
    reveal_immediately: bool,
) -> bool {
    if !cache.has_active_tile {
        return false;
    }
    commands.simulation_id = match cache.tiles.get(&cache.active_tile) {
        Some(existing) => existing.simulation_id,
        None => return false,
    };
    let (completed, remaining) = match cache.tiles.get(&cache.active_tile) {
        Some(existing) => (
            existing.completed_iterations,
            settings
                .iterations
                .saturating_sub(existing.completed_iterations),
        ),
        None => return false,
    };
    let iterations = iteration_budget.min(remaining);
    // The C++ hands `settings` and the iteration count straight to
    // `RunErosionIterations`, which runs in the same process; the port must
    // ship both across the bridge so the render world simulates with the
    // settings actually in force (a UI edit included).
    commands.iterate = Some(IterateCommand {
        count: iterations,
        settings: *settings,
    });
    if let Some(existing) = cache.tiles.get_mut(&cache.active_tile) {
        existing.completed_iterations += iterations;
    }
    if completed + iterations >= settings.iterations {
        commands.finalize = true;
        // The render world publishes TileFinalized; the cache applies it in
        // apply_erosion_events below. Reveal-immediately is honoured there
        // for prewarmed tiles.
        if reveal_immediately {
            cache.force_reveal = Some(cache.active_tile);
        }
        return true;
    }
    false
}

/// One frame of erosion streaming. Mirrors `UpdateErosionCache`: ensure the
/// 9x9 cache around the player's tile, advance reveals with wall-clock time,
/// begin the next queued tile, feed the active tile `iteration_budget` more
/// iterations, and finalize when complete. GPU work is expressed as
/// [`ErosionFrameCommands`] on the bridge; atlas patches and readbacks come
/// back as [`ErosionEvent`]s.
///
/// Returns whether a tile was begun this frame: the C++ prewarm loop runs one
/// full-budget pass per tile, so the caller counts down its four passes here.
/// With `preparation`, the next tile's inputs are built off the main thread
/// while the active tile simulates, and it begins once they are ready;
/// without, it begins in this frame.
#[expect(
    clippy::too_many_arguments,
    reason = "coordinates the cache, worker bridge and per-frame streaming inputs"
)]
pub fn update_erosion_cache(
    cache: &mut ErosionCache,
    bridge: &ErosionBridge,
    noise: &NoiseField,
    mut preparation: Option<&mut TilePreparation>,
    settings: &ErosionSettings,
    player_position: [f32; 3],
    iteration_budget: usize,
    reveal_immediately: bool,
    delta_time: f32,
) -> bool {
    if !cache.ready {
        return false;
    }
    let player_tile = world_tile(player_position[0], player_position[2]);
    let player_minimum = erosion_candidate_minimum(player_position[0], player_position[2]);
    ensure_erosion_cache(cache, player_tile);

    let reveal_step = (delta_time.min(0.05) / EROSION_REVEAL_SECONDS).min(1.0);
    for existing in cache.tiles.values_mut() {
        if existing.state == ErosionTileState::Ready {
            existing.reveal = (existing.reveal + reveal_step).min(1.0);
        }
    }

    let mut commands = ErosionFrameCommands {
        simulation_id: cache
            .tiles
            .get(&cache.active_tile)
            .filter(|_| cache.has_active_tile)
            .map_or(SimulationId::default(), |existing| existing.simulation_id),
        sim_min: tile_simulation_minimum(if cache.has_active_tile {
            cache.active_tile
        } else {
            player_tile
        }),
        init: None,
        iterate: None,
        finalize: false,
    };

    let mut began_tile = false;
    if !cache.has_active_tile
        && let Some(next) = choose_next_erosion_tile(cache, player_tile, player_minimum)
    {
        match preparation.as_deref_mut() {
            Some(preparation) => {
                if let Some(prepared) = preparation.poll(cache, next) {
                    begin_prepared_tile(cache, &mut commands, prepared);
                    began_tile = true;
                }
            }
            None => {
                begin_tile(cache, &mut commands, noise, next);
                began_tile = true;
            }
        }
    }
    advance_active_tile(
        cache,
        &mut commands,
        settings,
        iteration_budget,
        reveal_immediately,
    );
    // The active tile is simulating, so the queue's best tile is the one
    // after it: build its inputs now, and it can begin as soon as this one
    // finalizes rather than a preparation later.
    if cache.has_active_tile
        && let Some(preparation) = preparation
        && let Some(next) = choose_next_erosion_tile(cache, player_tile, player_minimum)
    {
        preparation.start(cache, next);
    }
    bridge.set_frame(commands, Some(erosion_lookup_records(cache)));
    began_tile
}

/// Marks a tile whose simulation will never deliver a finalize: one strike
/// against `failed_readbacks`, three of which give up on the tile, and the
/// active slot is released either way. Shared by the readback-failure event
/// and the main-world worker watchdog. A re-queued tile re-begins and
/// re-stamps its init, so it converges to the same simulation as before.
fn mark_tile_failed(cache: &mut ErosionCache, key: TileKey, why: &str) {
    if let Some(existing) = cache.tiles.get_mut(&key) {
        existing.failed_readbacks += 1;
        let failed = existing.failed_readbacks >= 3;
        existing.state = if failed {
            ErosionTileState::Failed
        } else {
            ErosionTileState::Queued
        };
        existing.completed_iterations = 0;
        log::warn!(
            "EROSION: {} for tile ({}, {}), strike {}",
            why,
            key.x,
            key.z,
            existing.failed_readbacks
        );
    }
    // No tile became Ready, so a pending immediate reveal must not carry over
    // to whichever tile finalizes next.
    if cache.force_reveal == Some(key) {
        cache.force_reveal = None;
    }
    cache.has_active_tile = false;
}

fn is_active_simulation(cache: &ErosionCache, key: TileKey, simulation_id: SimulationId) -> bool {
    cache.has_active_tile
        && cache.active_tile == key
        && cache.tiles.get(&key).is_some_and(|existing| {
            existing.state == ErosionTileState::Simulating
                && existing.simulation_id == simulation_id
        })
}

/// Apply events published by the erosion worker, and keep the watchdog
/// running against its death or stall.
pub fn apply_erosion_events(cache: &mut ErosionCache, bridge: &ErosionBridge) {
    // A dead worker is a dead simulation: every later tile would just watch
    // the watchdog count to three. The render thread never had a quieter
    // failure mode (a lost device or a validation error panicked the process
    // the same way), and neither does the worker.
    if let Some(why) = bridge.take_worker_dead() {
        panic!("EROSION: worker thread died; the erosion cache cannot continue: {why}");
    }

    let mut delivery = false;
    for event in bridge.drain_events() {
        match event {
            ErosionEvent::TileFinalized {
                simulation_id,
                tile: finalized,
            } => {
                let key = finalized.key;
                // Resets and retries can reuse a key while an earlier
                // attempt's readback remains in flight.
                if !is_active_simulation(cache, key, simulation_id) {
                    continue;
                }
                if let Some(existing) = cache.tiles.get_mut(&key) {
                    existing.state = ErosionTileState::Ready;
                    existing.reveal = if cache.force_reveal == Some(key) {
                        1.0
                    } else {
                        0.0
                    };
                    existing.cpu_delta = finalized.cpu_delta;
                    existing.cpu_concavity = finalized.cpu_concavity;
                    if cache.force_reveal == Some(key) {
                        cache.force_reveal = None;
                    }
                    cache.stats = finalized.stats;
                    let simulated = if finalized.stats.simulation_ms > 0.0 {
                        format!(", simulated in {:.0} ms", finalized.stats.simulation_ms)
                    } else {
                        String::new()
                    };
                    log::info!(
                        "EROSION: independent pass ({}, {}) ready, range {:.1}..{:.1} m, incision {:.2} m, deposition {:.2} m, detail {:.3}, axis {:.3}, drainage {:.0} cells, bedrock {:.1}%, cover {:.2} m{}",
                        key.x,
                        key.z,
                        finalized.stats.minimum_height,
                        finalized.stats.maximum_height,
                        finalized.stats.maximum_incision,
                        finalized.stats.maximum_deposition,
                        finalized.stats.erosion_detail,
                        finalized.stats.flow_axis_bias,
                        finalized.stats.maximum_drainage,
                        finalized.stats.bedrock_exposure,
                        finalized.stats.mean_loose_cover,
                        simulated,
                    );
                }
                cache.has_active_tile = false;
                delivery = true;
            }
            ErosionEvent::ReadbackFailed { key, simulation_id } => {
                if !is_active_simulation(cache, key, simulation_id) {
                    continue;
                }
                mark_tile_failed(cache, key, "readback failed");
                delivery = true;
            }
        }
    }

    // The watchdog runs while a tile is active with no delivery this frame:
    // a live worker finishes every well-formed tile within ~24 frames plus
    // readback latency, so only a stalled thread reaches the trip.
    if cache.has_active_tile && !delivery {
        cache.watchdog_frames = cache.watchdog_frames.saturating_add(1);
        if cache.watchdog_frames >= forest_core::constants::EROSION_WORKER_WATCHDOG_FRAMES {
            cache.watchdog_frames = 0;
            let key = cache.active_tile;
            mark_tile_failed(
                cache,
                key,
                &format!(
                    "no finalize landed within {} frames of beginning; requeueing",
                    forest_core::constants::EROSION_WORKER_WATCHDOG_FRAMES
                ),
            );
        }
    }
}

/// CSV overlap measurement across the 512 m shared strip of two lattice
/// neighbours, ported from `RunOverlapMeasurement`'s analysis half. The first
/// tile retains [stride*x - 512, stride*x + 512) and the second is offset by
/// one stride, so the overlap pairs (firstColumn = half + i, secondColumn = i)
/// while rows align one-to-one.
pub fn measure_overlap(first: &ErosionTile, second: &ErosionTile) {
    if first.cpu_delta.is_empty() || second.cpu_delta.is_empty() {
        log::warn!(
            "MEASURE: overlap tiles ({}, {}) and ({}, {}) did not finalize",
            first.key.x,
            first.key.z,
            second.key.x,
            second.key.z
        );
        return;
    }
    let half = EROSION_OUTPUT_RESOLUTION / 2;
    let mut column_mean = vec![0.0f32; half];
    let mut column_maximum = vec![0.0f32; half];
    let mut differences: Vec<f32> = Vec::with_capacity(half * EROSION_OUTPUT_RESOLUTION);
    for i in 0..half {
        let mut sum = 0.0f64;
        let mut maximum = 0.0f32;
        for row in 0..EROSION_OUTPUT_RESOLUTION {
            let delta_a = first.cpu_delta[row * EROSION_OUTPUT_RESOLUTION + half + i];
            let delta_b = second.cpu_delta[row * EROSION_OUTPUT_RESOLUTION + i];
            let difference = (delta_a - delta_b).abs();
            sum += difference as f64;
            maximum = maximum.max(difference);
            differences.push(difference);
        }
        column_mean[i] = (sum / EROSION_OUTPUT_RESOLUTION as f64) as f32;
        column_maximum[i] = maximum;
    }
    differences.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    // Mirrors the C++ percentile: (size_t)(fraction * (size - 1)).
    let percentile = |fraction: f64| -> f32 {
        if differences.is_empty() {
            return 0.0;
        }
        let index = (fraction * (differences.len() - 1) as f64) as usize;
        differences[index.min(differences.len() - 1)]
    };
    let overall_mean: f64 =
        differences.iter().sum::<f32>() as f64 / differences.len().max(1) as f64;

    println!("offset_m,mean_abs_delta_m,max_abs_delta_m");
    for i in 0..half {
        let offset = (i as f32 + 0.5) * EROSION_CELL_SIZE - 0.5 * EROSION_TILE_STRIDE;
        println!(
            "{:.1},{:.4},{:.4}",
            offset, column_mean[i], column_maximum[i]
        );
    }
    println!(
        "summary,mean={:.4},p99={:.4},max={:.4}",
        overall_mean as f32,
        percentile(0.99),
        differences.last().copied().unwrap_or(0.0)
    );
}

#[cfg(test)]
#[path = "cache_tests.rs"]
mod tests;

#[cfg(test)]
mod watchdog_frame_limit_check {
    /// The watchdog interval against the live cache settings: a healthy tile
    /// finalizes in well under it, so a trip means the worker stalled.
    #[test]
    fn watchdog_interval_dwarfs_a_healthy_tile() {
        let budget = forest_core::constants::EROSION_ITERATIONS_PER_FRAME.max(1);
        let frames_for_a_tile = forest_core::constants::ErosionSettings::default()
            .iterations
            .div_ceil(budget);
        assert!(
            frames_for_a_tile < forest_core::constants::EROSION_WORKER_WATCHDOG_FRAMES as usize,
            "a healthy tile takes {frames_for_a_tile} frames; the watchdog trips at {}",
            forest_core::constants::EROSION_WORKER_WATCHDOG_FRAMES
        );
    }
}
