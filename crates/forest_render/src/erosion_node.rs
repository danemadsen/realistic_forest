//! The erosion render world's slim node: the GPU simulation now runs on a
//! dedicated worker thread ([`forest_erosion::worker`]) on its own wgpu
//! device, and this node keeps only the pieces that must touch the render
//! device:
//!
//! 1. spawning the worker once the shared `RenderAdapter` is visible;
//! 2. stamping the C++ `UpdateErosionLookup` records into the shared lookup
//!    texture (the main world's scheduling produces them; the render device
//!    owns the texture);
//! 3. uploading the worker's finished 260x260 atlas patches into the shared
//!    height/flow atlases once those records publish each tile's slot;
//! 4. phase-cost attribution for `--shot` frame-time captures.
//!
//! The simulation, its pipelines, its scratch textures and its readback all
//! live on the worker thread now; only bytes cross between the two worlds,
//! through [`forest_erosion::ErosionBridge`]. The worker also compiles its
//! raw WGSL once at startup — asset hot reload does not reach it, so an
//! edited erosion shader needs a restart (PORTING_SPEC.md documents this;
//! the scratch asset root flow survives, because the main world publishes
//! whatever assets it actually loaded).

use crate::ExtractedForestView;
use crate::gpu_textures::{GpuWorldTextures, GpuWorldTexturesOption, write_padded_at};
use bevy::prelude::*;
use bevy::render::renderer::{RenderAdapter, RenderContext, RenderDevice, RenderQueue};
use forest_core::constants::*;
use forest_erosion::finalize::f32_to_bytes;
use forest_erosion::worker::spawn_erosion_worker;
use forest_erosion::{
    ErosionBridge, ErosionLookup, ErosionPhaseCosts, PendingAtlasPatch, SimulationId, TileKey,
};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// One RGBA32F texel of the simulation domain (the atlas and lookup uploads).
const SIM_TEXEL_BYTES: u32 = 16;
/// The lookup texture is one RGBA32F texel per lattice cell.
const LOOKUP_TEXTURE_SIZE: u32 = EROSION_LOOKUP_DIAMETER as u32;

/// A finished atlas patch is dropped once its tile has stayed out of the
/// lookup window for this many frames (the tile was evicted before its slot
/// was ever published), which bounds the pending list.
const MAX_PENDING_ATLAS_AGE: u32 = 600;

// ---------------------------------------------------------------------------
// Render-world state
// ---------------------------------------------------------------------------

/// Render-world state for the erosion node: whether the simulation's worker
/// thread has been spawned, and the patches still waiting for their atlas
/// slot. The worker owns everything else.
#[derive(Resource)]
pub struct ErosionWorkerState {
    /// The spawn attempt happened (even a failed one, whose reason the
    /// bridge carries); retrying a dead spawn would queue threads forever.
    spawned: AtomicBool,

    /// Patches drained from the bridge but not yet uploaded: their tile's
    /// slot may still be unpublished, and dropping them here would discard a
    /// finished tile's atlas contribution forever. The render schedule only
    /// gets `&World` to this node, hence the mutex.
    pending_patches: Mutex<Vec<PendingAtlasPatch>>,
}

impl Default for ErosionWorkerState {
    fn default() -> Self {
        Self {
            spawned: AtomicBool::new(false),
            pending_patches: Mutex::new(Vec::new()),
        }
    }
}

/// The render world's erosion node, in `ForestRenderSystems::Erosion`:
/// worker spawn, lookup stamping and atlas patch uploads. The simulation's
/// command loop lives in `forest_erosion::worker` now — this node never
/// touches the bridge's command path at all.
pub fn forest_erosion_pass(world: &World, _ctx: RenderContext) {
    let Some(state) = world.get_resource::<ErosionWorkerState>() else {
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
    let Some(bridge) = world.get_resource::<ErosionBridge>() else {
        return;
    };
    let Some(view) = world.get_resource::<ExtractedForestView>() else {
        return;
    };

    // Phase attribution for --shot captures: every early return below simply
    // skips phases, so their costs stay out of the totals.
    let mut costs = ErosionPhaseCosts::default();
    let mut tick = std::time::Instant::now();

    // Spawning runs once, in the render schedule, because only the render
    // app holds the shared `RenderAdapter` the worker must open its device
    // on. It waits for the main world's WGSL publisher: spawning before the
    // sources exist would kill the worker for what is only asset loading
    // latency. A failed spawn has already reported through the bridge (the
    // main world panics on reading it), so there is nothing left to retry.
    if !state.spawned.load(Ordering::SeqCst)
        && bridge.shader_sources_ready()
        && !state.spawned.swap(true, Ordering::SeqCst)
    {
        spawn_worker(world, bridge);
        costs.lookup_atlas_us += tick.elapsed().as_micros() as u64;
        tick = std::time::Instant::now();
    }

    // Pipelined rendering overlaps the next main-world update. Records and
    // allocation identities must come from the same extraction as the origin
    // in this frame's globals, rather than the bridge's newer live state.
    let lookup = &view.erosion_lookup;
    let records = &lookup.records;
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

    // Atlas patches the worker finished. They join the pending list first, so
    // a patch whose slot publishes on a later frame survives; only uploads,
    // age accounting and eviction happen here.
    let patches = bridge.drain_patches();
    let mut pending = state
        .pending_patches
        .lock()
        .expect("erosion pending atlas lock poisoned");
    pending.extend(patches);
    flush_pending_atlas(queue, textures, lookup, &mut pending);
    pending.retain(|patch| patch.age <= MAX_PENDING_ATLAS_AGE);
    drop(pending);
    costs.lookup_atlas_us += tick.elapsed().as_micros() as u64;

    bridge.record_phase_costs(costs);
}

/// Opens the erosion worker's device on the renderer's shared adapter.
fn spawn_worker(world: &World, bridge: &ErosionBridge) {
    let Some(render_adapter) = world.get_resource::<RenderAdapter>() else {
        return;
    };
    let Some(render_device) = world.get_resource::<RenderDevice>() else {
        return;
    };
    // `RenderAdapter` wraps bevy's `WgpuWrapper` around the raw adapter; the
    // worker takes the bare `wgpu::Adapter` and clones it onto its thread,
    // mirroring the render device's requested features and limits.
    spawn_erosion_worker(
        bridge,
        &render_adapter.0,
        render_device.features(),
        render_device.limits(),
    );
}

/// Decides whether the lookup records contain an allocated atlas slot for
/// `key`. `UpdateErosionLookup` writes the same numbers in the same 11x11
/// record layout the lookup texture carries: the slot in (x, z) at offsets
/// 0 and 1, the allocation state at offset 3. The attempt identity must also
/// match, so a reset or retry cannot upload an older simulation's patch.
fn published_slot(
    key: TileKey,
    simulation_id: SimulationId,
    lookup: &ErosionLookup,
) -> Option<(f32, f32)> {
    let diameter = EROSION_LOOKUP_DIAMETER as i64;
    let relative_x = key.x.checked_sub(lookup.minimum.x)?;
    let relative_z = key.z.checked_sub(lookup.minimum.z)?;
    if relative_x < 0 || relative_z < 0 || relative_x >= diameter || relative_z >= diameter {
        return None;
    }
    let cell = (relative_z * diameter + relative_x) as usize;
    if simulation_id == SimulationId::default()
        || lookup.simulation_ids.get(cell) != Some(&simulation_id)
    {
        return None;
    }
    let records = &lookup.records;
    let record = cell * 4;
    if records.get(record + 3).copied().unwrap_or(0.0) > 0.0 {
        Some((records[record], records[record + 1]))
    } else {
        None
    }
}

/// Uploads finished atlas patches once the lookup records reveal the slot
/// the main world allocated for them. Patches still waiting age by one
/// upload attempt; eviction happens in the caller's `retain`.
pub fn flush_pending_atlas(
    queue: &RenderQueue,
    textures: &GpuWorldTextures,
    lookup: &ErosionLookup,
    patches: &mut Vec<PendingAtlasPatch>,
) -> usize {
    let mut uploaded = 0;
    let mut index = 0;
    while index < patches.len() {
        let patch = &patches[index];
        let slot = published_slot(patch.key, patch.simulation_id, lookup);
        let Some((slot_x, slot_z)) = slot else {
            patches[index].age += 1;
            index += 1;
            continue;
        };

        let patch = patches.remove(index);
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
        uploaded += 1;
    }
    uploaded
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod node_tests {
    use super::*;
    use forest_erosion::{
        ErosionCache, ErosionFrameCommands, ErosionTileState, PreparedTile, begin_prepared_tile,
        ensure_erosion_cache, erosion_lookup_snapshot, tile,
    };

    fn begin_attempt(cache: &mut ErosionCache, key: TileKey) -> SimulationId {
        let mut commands = ErosionFrameCommands {
            simulation_id: SimulationId::default(),
            sim_min: [0.0; 2],
            init: None,
            iterate: None,
            finalize: false,
        };
        let prepared = PreparedTile {
            key,
            base_height: Vec::new(),
            drainage_area: Vec::new(),
            river: Vec::new(),
        };
        begin_prepared_tile(cache, &mut commands, prepared);
        commands.simulation_id
    }

    /// The record layout `published_slot` reads: slot in (x, z) at offsets
    /// 0 and 1, the allocation state at offset 3 — what `erosion_lookup_records`
    /// writes and what `flush_pending_atlas` gates on.
    #[test]
    fn published_slot_reads_the_record_layout() {
        let key = tile(2, 1);
        let mut cache = ErosionCache::default();
        ensure_erosion_cache(&mut cache, key);
        cache.lookup_minimum = tile(0, 0);
        let simulation_id = begin_attempt(&mut cache, key);
        // Inside the window but not yet published: wait.
        let unpublished = erosion_lookup_snapshot(&cache);
        assert_eq!(published_slot(key, simulation_id, &unpublished), None);
        // Published: the slot the record carries.
        let allocated = cache.tiles.get_mut(&key).expect("test tile is allocated");
        allocated.atlas_slot = 6 * EROSION_ATLAS_COLUMNS + 5;
        allocated.state = ErosionTileState::Ready;
        let lookup = erosion_lookup_snapshot(&cache);
        assert_eq!(
            published_slot(key, simulation_id, &lookup),
            Some((5.0, 6.0))
        );
        // Outside the 11x11 window: no slot at all.
        for outside in [tile(100, 0), tile(0, -100), tile(i64::MIN, i64::MAX)] {
            assert_eq!(published_slot(outside, simulation_id, &lookup), None);
        }
    }

    #[test]
    fn atlas_slots_follow_the_extracted_origin_and_simulation_attempt() {
        let key = tile(2, 1);
        let mut cache = ErosionCache::default();
        ensure_erosion_cache(&mut cache, key);
        cache.lookup_minimum = tile(0, 0);
        let earlier_id = begin_attempt(&mut cache, key);
        let earlier = cache.tiles.get_mut(&key).expect("test tile is allocated");
        earlier.atlas_slot = 6 * EROSION_ATLAS_COLUMNS + 5;
        earlier.state = ErosionTileState::Ready;
        let extracted = erosion_lookup_snapshot(&cache);

        // The next main-world update can restart the same tile and move its
        // lookup window while the renderer still holds the earlier snapshot.
        let newer_id = begin_attempt(&mut cache, key);
        cache.lookup_minimum = tile(1, 0);
        let newer = cache
            .tiles
            .get_mut(&key)
            .expect("test tile remains allocated");
        newer.atlas_slot = 3;
        newer.state = ErosionTileState::Ready;
        let next_snapshot = erosion_lookup_snapshot(&cache);

        assert_ne!(earlier_id, newer_id);
        assert_eq!(
            published_slot(key, earlier_id, &extracted),
            Some((5.0, 6.0))
        );
        assert_eq!(published_slot(key, newer_id, &extracted), None);
        assert_eq!(published_slot(key, earlier_id, &next_snapshot), None);
        assert_eq!(
            published_slot(key, newer_id, &next_snapshot),
            Some((3.0, 0.0))
        );
    }
}
