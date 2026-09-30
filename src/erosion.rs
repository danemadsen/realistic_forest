//! Hydraulic erosion tile cache orchestration (CPU side), ported from the
//! C++ `main.cpp` erosion functions.
//!
//! The GPU simulation passes themselves live in the render world; this module
//! owns the tile state machine, blending math for CPU height queries, and the
//! per-frame scheduling that decides which tile initializes, streams
//! iterations, and finalizes. Main and render worlds communicate through
//! [`ErosionBridge`].

use crate::constants::*;
use crate::noise::NoiseField;
use bevy::ecs::prelude::Resource;
use std::collections::HashMap;
use std::collections::VecDeque;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct TileKey {
    pub x: i64,
    pub z: i64,
}

pub const fn tile(x: i64, z: i64) -> TileKey {
    TileKey { x, z }
}

/// Cache windows are centred on the nearest pass centre. Keeping this
/// distinct from the lower candidate cell makes their support symmetric as
/// the player crosses the 512 m lattice.
pub fn world_tile(x: f32, z: f32) -> TileKey {
    tile((x as f64 / EROSION_TILE_STRIDE as f64 + 0.5).floor() as i64,
         (z as f64 / EROSION_TILE_STRIDE as f64 + 0.5).floor() as i64)
}

pub fn erosion_candidate_minimum(x: f32, z: f32) -> TileKey {
    tile((x as f64 / EROSION_TILE_STRIDE as f64).floor() as i64,
         (z as f64 / EROSION_TILE_STRIDE as f64).floor() as i64)
}

/// World-space origin of a tile's full simulation domain (retained square
/// plus discarded halo on every side).
pub fn tile_simulation_minimum(key: TileKey) -> [f32; 2] {
    let retained_half_size = EROSION_FOOTPRINT_SIZE * 0.5;
    [(key.x as f64 * EROSION_TILE_STRIDE as f64
        - retained_half_size as f64
        - EROSION_SIMULATION_HALO as f64) as f32,
     (key.z as f64 * EROSION_TILE_STRIDE as f64
        - retained_half_size as f64
        - EROSION_SIMULATION_HALO as f64) as f32]
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
    pub atlas_slot: usize,
    pub completed_iterations: usize,
    pub failed_readbacks: usize,
    pub reveal: f32,
    pub state: ErosionTileState,
    /// This is the independent result for one hydraulic pass. It deliberately
    /// remains distinct from its three overlapping neighbours.
    pub cpu_delta: Vec<f32>,
}

impl ErosionTile {
    fn new(key: TileKey, atlas_slot: usize) -> Self {
        Self {
            key,
            atlas_slot,
            completed_iterations: 0,
            failed_readbacks: 0,
            reveal: 0.0,
            state: ErosionTileState::Queued,
            cpu_delta: Vec::new(),
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
    [((x as f64 - support_x) / EROSION_FOOTPRINT_SIZE as f64) as f32,
     ((z as f64 - support_z) / EROSION_FOOTPRINT_SIZE as f64) as f32]
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
    lerp(lerp(at(x0, z0), at(x1, z0), tx), lerp(at(x0, z1), at(x1, z1), tx), tz)
}

pub fn sample_tile_delta(tile: &ErosionTile, x: f32, z: f32) -> f32 {
    if tile.cpu_delta.is_empty() {
        return 0.0;
    }
    // The C++ widens only the lattice product to double before narrowing:
    // `static_cast<float>(static_cast<double>(tile.key.x) * kErosionTileStride
    // - kErosionFootprintSize * 0.5)`; every later step is f32.
    let support_x = (tile.key.x as f64 * EROSION_TILE_STRIDE as f64
        - EROSION_FOOTPRINT_SIZE as f64 * 0.5) as f32;
    let support_z = (tile.key.z as f64 * EROSION_TILE_STRIDE as f64
        - EROSION_FOOTPRINT_SIZE as f64 * 0.5) as f32;
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
    let at = |px: usize, pz: usize| tile.cpu_delta[pz * EROSION_OUTPUT_RESOLUTION + px];
    lerp(lerp(at(x0, z0), at(x1, z0), tx), lerp(at(x0, z1), at(x1, z1), tx), tz)
}

/// Radial fade of erosion influence beyond the reliable cache window.
pub fn erosion_visibility(x: f32, z: f32, center: [f32; 2]) -> f32 {
    let dx = x - center[0];
    let dz = z - center[1];
    1.0 - smoothstep(EROSION_VISIBILITY_FULL_RADIUS,
                     EROSION_VISIBILITY_ZERO_RADIUS,
                     (dx * dx + dz * dz).sqrt())
}

/// Sample the blended eroded surface height at a world position. Four fixed
/// spatial candidate masks are normalized independently of readiness so a
/// missing/streaming pass contributes zero erosion (the procedural base) and
/// fades in with its own C1 mask, avoiding 512 m quartet-availability steps.
pub fn sample_eroded_height(
    cache: &ErosionCache,
    noise: &NoiseField,
    x: f32,
    z: f32,
    visibility_center: [f32; 2],
) -> f32 {
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

    let mut delta = 0.0f32;
    if total_weight > 0.000001 {
        let inverse_weight = 1.0 / total_weight;
        for (index, key) in candidates.iter().enumerate() {
            if let Some(found) = cache.tiles.get(key) {
                if found.state == ErosionTileState::Ready {
                    let reveal = found.reveal.clamp(0.0, 1.0);
                    delta += masks[index]
                        * inverse_weight
                        * reveal
                        * sample_tile_delta(found, x, z);
                }
            }
        }
    }
    crate::noise::base_height(noise, x, z) + delta * erosion_visibility(x, z, visibility_center)
}

// ---------------------------------------------------------------------------
// Main <-> render bridge
// ---------------------------------------------------------------------------

/// Per-frame GPU work the erosion node must run for the active tile. Written
/// by the main-world coordinator, extracted, then executed in the render
/// world in a fully prescribed order.
pub struct ErosionFrameCommands {
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
}

pub struct IterateCommand {
    pub count: usize,
    pub settings: ErosionSettings,
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
    pub stats: TileDiagnostics,
}

pub enum ErosionEvent {
    TileFinalized(FinalizedTile),
    ReadbackFailed(TileKey),
}

/// Inner state behind the bridge's mutex, shared by both worlds.
#[derive(Default)]
pub struct BridgeState {
    commands: Option<ErosionFrameCommands>,
    lookup_records: Option<Vec<f32>>,
    events: VecDeque<ErosionEvent>,
}

/// Shared state between the main world (tile scheduling) and the render
/// world (GPU pass execution + readbacks).
#[derive(Clone, Default, Resource)]
pub struct ErosionBridge(pub std::sync::Arc<std::sync::Mutex<BridgeState>>);

impl ErosionBridge {
    pub fn take_commands(&self) -> Option<ErosionFrameCommands> {
        self.0.lock().unwrap().commands.take()
    }
    pub fn take_lookup(&self) -> Option<Vec<f32>> {
        self.0.lock().unwrap().lookup_records.take()
    }
    pub fn drain_events(&self) -> VecDeque<ErosionEvent> {
        std::mem::take(&mut self.0.lock().unwrap().events)
    }

    pub fn set_frame(&self, commands: ErosionFrameCommands, lookup_records: Option<Vec<f32>>) {
        let mut state = self.0.lock().unwrap();
        state.commands = Some(commands);
        state.lookup_records = lookup_records;
    }
    pub fn push_event(&self, event: ErosionEvent) {
        self.0.lock().unwrap().events.push_back(event);
    }
}

// ---------------------------------------------------------------------------
// Tile cache
// ---------------------------------------------------------------------------

/// The seven per-tile summary statistics the C++ keeps flat on
/// `HydraulicErosion` (minimumHeight .. flowAxisBias); they describe the most
/// recently completed tile and are shown in the diagnostics window.
#[derive(Clone, Copy, Debug, Default)]
pub struct TileDiagnostics {
    pub minimum_height: f32,
    pub maximum_height: f32,
    pub land_coverage: f32,
    pub maximum_incision: f32,
    pub maximum_deposition: f32,
    pub erosion_detail: f32,
    pub flow_axis_bias: f32,
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
        }
    }
}

pub fn ensure_erosion_cache(cache: &mut ErosionCache, center: TileKey) {
    cache.tiles.retain(|key, existing| {
        let outside = (key.x - center.x).abs() > EROSION_STREAMING_RADIUS
            || (key.z - center.z).abs() > EROSION_STREAMING_RADIUS;
        let active = cache.has_active_tile && *key == cache.active_tile;
        if outside && !active {
            cache.free_atlas_slots.push(existing.atlas_slot);
            false
        } else {
            true
        }
    });
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
    cache.lookup_minimum = tile(center.x - EROSION_LOOKUP_RADIUS,
                                center.z - EROSION_LOOKUP_RADIUS);
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
        let priority = if covers_player_cell(*key, player_minimum) { 0 } else { 1 };
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
        let destination = ((relative_z as usize * EROSION_LOOKUP_DIAMETER)
            + relative_x as usize)
            * 4;
        records[destination] = (existing.atlas_slot % EROSION_ATLAS_COLUMNS) as f32;
        records[destination + 1] = (existing.atlas_slot / EROSION_ATLAS_COLUMNS) as f32;
        records[destination + 2] = existing.reveal;
        records[destination + 3] = 1.0;
    }
    records
}

pub fn reset_erosion_cache(cache: &mut ErosionCache) {
    cache.tiles.clear();
    cache.free_atlas_slots = (0..EROSION_ATLAS_SLOTS).rev().collect();
    cache.has_active_tile = false;
    cache.cache_dropped = true;
    cache.lookup_minimum = tile(0, 0);
    // A pending immediate-reveal belongs to a tile the reset just discarded.
    cache.force_reveal = None;
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
    let base_height = crate::noise::create_base_height_map(noise, key);
    commands.sim_min = tile_simulation_minimum(key);
    commands.init = Some(InitTileCommand {
        key,
        base_height,
    });
    if let Some(existing) = cache.tiles.get_mut(&key) {
        existing.state = ErosionTileState::Simulating;
        existing.completed_iterations = 0;
        existing.reveal = 0.0;
    }
    cache.active_tile = key;
    cache.has_active_tile = true;
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
    let (completed, remaining) = match cache.tiles.get(&cache.active_tile) {
        Some(existing) => (
            existing.completed_iterations,
            settings.iterations.saturating_sub(existing.completed_iterations),
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
#[allow(clippy::too_many_arguments)]
pub fn update_erosion_cache(
    cache: &mut ErosionCache,
    bridge: &ErosionBridge,
    noise: &NoiseField,
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
    if !cache.has_active_tile {
        if let Some(next) = choose_next_erosion_tile(cache, player_tile, player_minimum) {
            begin_tile(cache, &mut commands, noise, next);
            began_tile = true;
        }
    }
    advance_active_tile(cache, &mut commands, settings, iteration_budget, reveal_immediately);
    bridge.set_frame(commands, Some(erosion_lookup_records(cache)));
    began_tile
}

/// Apply events published by the render world after its readbacks.
pub fn apply_erosion_events(cache: &mut ErosionCache, bridge: &ErosionBridge) {
    for event in bridge.drain_events() {
        match event {
            ErosionEvent::TileFinalized(finalized) => {
                let key = finalized.key;
                // The render world sees finalize commands again each frame
                // until the async readback lands; only the first event of a
                // simulation may transition the tile.
                let already_ready = cache
                    .tiles
                    .get(&key)
                    .is_some_and(|existing| existing.state == ErosionTileState::Ready);
                if already_ready {
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
                    if cache.force_reveal == Some(key) {
                        cache.force_reveal = None;
                    }
                    cache.stats = finalized.stats;
                    log::info!(
                        "EROSION: independent pass ({}, {}) ready, range {:.1}..{:.1} m, incision {:.2} m, detail {:.3}, axis {:.3}",
                        key.x, key.z,
                        finalized.stats.minimum_height,
                        finalized.stats.maximum_height,
                        finalized.stats.maximum_incision,
                        finalized.stats.erosion_detail,
                        finalized.stats.flow_axis_bias,
                    );
                }
                cache.has_active_tile = false;
            }
            ErosionEvent::ReadbackFailed(key) => {
                if let Some(existing) = cache.tiles.get_mut(&key) {
                    existing.failed_readbacks += 1;
                    let failed = existing.failed_readbacks >= 3;
                    existing.state = if failed {
                        ErosionTileState::Failed
                    } else {
                        ErosionTileState::Queued
                    };
                    existing.completed_iterations = 0;
                    log::warn!("EROSION: readback failed for tile ({}, {})", key.x, key.z);
                }
                // No tile became Ready, so a pending immediate reveal must not
                // carry over to whichever tile finalizes next.
                if cache.force_reveal == Some(key) {
                    cache.force_reveal = None;
                }
                cache.has_active_tile = false;
            }
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
            first.key.x, first.key.z, second.key.x, second.key.z
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
        println!("{:.1},{:.4},{:.4}", offset, column_mean[i], column_maximum[i]);
    }
    println!("summary,mean={:.4},p99={:.4},max={:.4}",
             overall_mean as f32,
             percentile(0.99),
             differences.last().copied().unwrap_or(0.0));
}