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
    /// Contributing area of every base cell (see [`route_base_drainage`]),
    /// so stream power acts on whole catchments from the first iteration.
    pub drainage_area: Vec<f32>,
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

    /// Hands one frame of work to the render world.
    ///
    /// The render world consumes at most one command set per frame, and only
    /// once its GPU state exists — for the first frames of the app the node
    /// returns before `take_commands` while `prepare_erosion_sim` builds the
    /// pipelines, textures and bind groups. Whatever is still pending when the
    /// next frame arrives is therefore *merged into* it rather than replaced.
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
        let mut state = self.0.lock().unwrap();
        // `sim_min` is a bijection of the tile key, so it identifies which tile
        // a command set describes. Work may only be carried across frames that
        // describe the *same* tile: once the main world moves on, its previous
        // tile has already finalized (that is what lets it move on), so
        // anything still pending for it is a duplicate re-send of `finalize`
        // and merging it into the new tile's frame would finalize the new tile
        // after a single batch.
        let same_tile = state
            .commands
            .as_ref()
            .is_some_and(|pending| pending.sim_min == commands.sim_min);
        if let Some(pending) = state.commands.take().filter(|_| same_tile) {
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
    }
    pub fn push_event(&self, event: ErosionEvent) {
        self.0.lock().unwrap().events.push_back(event);
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

/// Contributing area, in cells and counting the cell itself, of every cell of
/// a square height grid. Each cell passes its area to its lower neighbours in
/// proportion to their squared slope, exactly the multiple-flow-direction
/// shares the GPU terrain pass routes every iteration, so the simulation
/// starts from its own converged drainage instead of growing it one cell per
/// iteration. Cells below sea level swallow what reaches them, flats and pits
/// keep theirs, and the grid edge is a closed wall.
pub fn route_base_drainage(heights: &[f32], resolution: usize, cell_size: f32) -> Vec<f32> {
    const NEIGHBOURS: [(i64, i64); 8] =
        [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)];
    let count = resolution * resolution;
    let mut area = vec![1.0f32; count.min(heights.len())];
    if area.len() != count {
        return area;
    }
    // Highest first, so every donor has received all of its own area before
    // it passes any on: the routing is acyclic because it only runs downhill.
    let mut order: Vec<u32> = (0..count as u32).collect();
    order.sort_unstable_by(|&a, &b| heights[b as usize].total_cmp(&heights[a as usize]));
    let diagonal = cell_size * std::f32::consts::SQRT_2;
    for &cell in &order {
        let cell = cell as usize;
        let height = heights[cell];
        if height < SEA_LEVEL {
            continue;
        }
        let x = (cell % resolution) as i64;
        let z = (cell / resolution) as i64;
        let mut shares = [(0usize, 0.0f32); 8];
        let mut share_total = 0.0f32;
        for (slot, (dx, dz)) in NEIGHBOURS.iter().enumerate() {
            let (nx, nz) = (x + dx, z + dz);
            if nx < 0 || nz < 0 || nx >= resolution as i64 || nz >= resolution as i64 {
                continue;
            }
            let neighbour = nz as usize * resolution + nx as usize;
            let distance = if *dx != 0 && *dz != 0 { diagonal } else { cell_size };
            let slope = (height - heights[neighbour]) / distance;
            if slope > 0.0 {
                shares[slot] = (neighbour, slope * slope);
                share_total += slope * slope;
            }
        }
        if share_total <= 0.0 {
            continue;
        }
        let passed = area[cell];
        for (neighbour, share) in shares {
            if share > 0.0 {
                area[neighbour] += passed * share / share_total;
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
    let drainage_area = route_base_drainage(&base_height, EROSION_RESOLUTION, EROSION_CELL_SIZE);
    commands.sim_min = tile_simulation_minimum(key);
    commands.init = Some(InitTileCommand {
        key,
        base_height,
        drainage_area,
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
                        "EROSION: independent pass ({}, {}) ready, range {:.1}..{:.1} m, incision {:.2} m, deposition {:.2} m, detail {:.3}, axis {:.3}, drainage {:.0} cells, bedrock {:.1}%, cover {:.2} m",
                        key.x, key.z,
                        finalized.stats.minimum_height,
                        finalized.stats.maximum_height,
                        finalized.stats.maximum_incision,
                        finalized.stats.maximum_deposition,
                        finalized.stats.erosion_detail,
                        finalized.stats.flow_axis_bias,
                        finalized.stats.maximum_drainage,
                        finalized.stats.bedrock_exposure,
                        finalized.stats.mean_loose_cover,
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A frame of work for one tile, with no init (the common case: only the
    /// frame that calls `BeginErosionTile` carries one).
    fn frame_for(sim_min: [f32; 2], iterate: usize, finalize: bool) -> ErosionFrameCommands {
        ErosionFrameCommands {
            sim_min,
            init: None,
            iterate: (iterate > 0).then(|| IterateCommand {
                count: iterate,
                settings: ErosionSettings::default(),
            }),
            finalize,
        }
    }

    /// The render world misses frames while `prepare_erosion_sim` builds its
    /// GPU state, and the `init` `BeginErosionTile` sends is one-shot: losing
    /// it leaves a tile `Simulating` with nothing to simulate and no erosion
    /// ever finalizes.
    #[test]
    fn set_frame_carries_a_pending_init_to_the_next_frame() {
        let bridge = ErosionBridge::default();
        bridge.set_frame(
            ErosionFrameCommands {
                sim_min: [0.0, 0.0],
                init: Some(InitTileCommand {
                    key: tile(0, 0),
                    base_height: vec![1.0, 2.0],
                    drainage_area: vec![1.0, 1.0],
                }),
                iterate: Some(IterateCommand {
                    count: 6,
                    settings: ErosionSettings::default(),
                }),
                finalize: false,
            },
            None,
        );
        // Nothing consumed it, and the next frame carries no init of its own.
        bridge.set_frame(frame_for([0.0, 0.0], 6, false), None);

        let taken = bridge.take_commands().expect("commands survived");
        assert!(taken.init.is_some(), "the pending init must not be dropped");
        assert_eq!(taken.iterate.expect("iterate").count, 12);
    }

    /// Once the main world moves to another tile, anything still pending for
    /// the old one is a redundant re-send and must not leak into the new
    /// tile's frame — merging its `finalize` would end the new tile after a
    /// single batch.
    #[test]
    fn set_frame_does_not_carry_work_across_tiles() {
        let bridge = ErosionBridge::default();
        bridge.set_frame(frame_for([0.0, 0.0], 6, true), None);
        bridge.set_frame(frame_for([512.0, 0.0], 6, false), None);

        let taken = bridge.take_commands().expect("commands");
        assert!(!taken.finalize);
        assert_eq!(taken.iterate.expect("iterate").count, 6);
    }

    /// An inclined plane drains down its slope: every row passes everything
    /// it holds to the row below, so row z carries (z + 1) rows of cells, and
    /// columns away from the closed side walls carry exactly z + 1 each.
    #[test]
    fn base_drainage_accumulates_down_a_plane() {
        let resolution = 8;
        let heights: Vec<f32> = (0..resolution * resolution)
            .map(|cell| 100.0 - (cell / resolution) as f32 * 2.0)
            .collect();
        let area = route_base_drainage(&heights, resolution, 4.0);
        for z in 0..resolution {
            let row: f32 = (0..resolution).map(|x| area[z * resolution + x]).sum();
            assert!((row - ((z + 1) * resolution) as f32).abs() < 1e-3, "row {z} = {row}");
            // The walls' influence spreads one column per row.
            for x in (z + 1).min(resolution)..resolution.saturating_sub(z + 1) {
                let value = area[z * resolution + x];
                assert!((value - (z + 1) as f32).abs() < 1e-3, "({x}, {z}) = {value}");
            }
        }
    }

    /// A valley gathers its sides into the thalweg, nothing is created or
    /// lost on the way to the closed outlet row, and the sea swallows its share.
    #[test]
    fn base_drainage_conserves_area_into_the_valley_and_the_sea() {
        let resolution = 9;
        let heights: Vec<f32> = (0..resolution * resolution)
            .map(|cell| {
                let x = (cell % resolution) as f32;
                let z = (cell / resolution) as f32;
                // A V valley along x = 4 falling toward a flat strand at
                // z = 7 that only drains into the sea row at z = 8.
                match z as usize {
                    8 => -5.0,
                    7 => 1.0,
                    _ => 40.0 + (x - 4.0).abs() * 3.0 - z,
                }
            })
            .collect();
        let area = route_base_drainage(&heights, resolution, 4.0);
        let land_cells = resolution * (resolution - 1);
        // The strand cells exchange nothing among themselves, so their areas
        // partition every land cell exactly once.
        let reaching_sea: f32 = (0..resolution)
            .map(|x| area[(resolution - 2) * resolution + x])
            .sum();
        assert!((reaching_sea - land_cells as f32).abs() < 1e-2, "{reaching_sea}");
        let thalweg = area[(resolution - 3) * resolution + 4];
        assert!(thalweg > area[(resolution - 3) * resolution + 2] * 3.0, "{thalweg}");
        assert!(area[(resolution - 1) * resolution + 4] == 0.0);
    }

    /// The ordinary path: a frame that is consumed leaves nothing behind.
    #[test]
    fn take_commands_empties_the_bridge() {
        let bridge = ErosionBridge::default();
        bridge.set_frame(frame_for([0.0, 0.0], 6, false), None);
        assert!(bridge.take_commands().is_some());
        assert!(bridge.take_commands().is_none());
    }
}
