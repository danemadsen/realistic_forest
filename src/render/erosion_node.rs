//! The erosion render world's slim node: the GPU simulation now runs on a
//! dedicated worker thread ([`crate::erosion_worker`]) on its own wgpu
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
//! through [`crate::erosion::ErosionBridge`]. The worker also compiles its
//! raw WGSL once at startup — asset hot reload does not reach it, so an
//! edited erosion shader needs a restart (PORTING-SPEC.md documents this;
//! the scratch asset root flow survives, because the main world publishes
//! whatever assets it actually loaded).

use crate::constants::*;
use crate::erosion::{ErosionBridge, ErosionPhaseCosts, PendingAtlasPatch, TileKey};
use crate::erosion_finalize::f32_to_bytes;
use crate::erosion_worker::spawn_erosion_worker;
use crate::render::gpu_textures::{write_padded_at, GpuWorldTextures, GpuWorldTexturesOption};
use crate::render::ExtractedForestView;
use bevy::prelude::*;
use bevy::render::renderer::{RenderAdapter, RenderContext, RenderDevice, RenderQueue};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

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
/// command loop lives in `crate::erosion_worker` now — this node never
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
    if !state.spawned.load(Ordering::SeqCst) && bridge.shader_sources_ready() {
        if !state.spawned.swap(true, Ordering::SeqCst) {
            spawn_worker(world, bridge);
            costs.lookup_atlas_us += tick.elapsed().as_micros() as u64;
            tick = std::time::Instant::now();
        }
    }

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

    // Atlas patches the worker finished. They join the pending list first, so
    // a patch whose slot publishes on a later frame survives; only uploads,
    // age accounting and eviction happen here.
    let patches = bridge.drain_patches();
    if let Ok(mut pending) = state.pending_patches.lock() {
        pending.extend(patches);
        if let (Some(records), Some(view)) = (
            records.as_deref(),
            world.get_resource::<ExtractedForestView>(),
        ) {
            flush_pending_atlas(queue, textures, view.lookup_minimum, records, &mut pending);
            pending.retain(|patch| patch.age <= MAX_PENDING_ATLAS_AGE);
        }
    }
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
        &**render_adapter.0,
        render_device.features(),
        render_device.limits(),
    );
}

/// Decides whether the lookup records contain an allocated atlas slot for
/// `key`. `UpdateErosionLookup` writes the same numbers in the same 11x11
/// record layout the lookup texture carries: the slot in (x, z) at offsets
/// 0 and 1, the allocation state at offset 3.
fn published_slot(
    key: TileKey,
    lookup_minimum: (i64, i64),
    records: &[f32],
) -> Option<(f32, f32)> {
    let diameter = EROSION_LOOKUP_DIAMETER as i64;
    let relative_x = key.x - lookup_minimum.0;
    let relative_z = key.z - lookup_minimum.1;
    if relative_x < 0 || relative_z < 0 || relative_x >= diameter || relative_z >= diameter {
        return None;
    }
    let record = ((relative_z * diameter + relative_x) * 4) as usize;
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
    lookup_minimum: (i64, i64),
    records: &[f32],
    patches: &mut Vec<PendingAtlasPatch>,
) -> usize {
    let mut uploaded = 0;
    let mut index = 0;
    while index < patches.len() {
        let slot = published_slot(patches[index].key, lookup_minimum, records);
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
    use crate::erosion::tile;

    /// The record layout `published_slot` reads: slot in (x, z) at offsets
    /// 0 and 1, the allocation state at offset 3 — what `erosion_lookup_records`
    /// writes and what `flush_pending_atlas` gates on.
    #[test]
    fn published_slot_reads_the_record_layout() {
        let mut records = vec![0.0; EROSION_LOOKUP_DIAMETER * EROSION_LOOKUP_DIAMETER * 4];
        // Inside the window but not yet published: wait.
        assert_eq!(published_slot(tile(0, 0), (0, 0), &records), None);
        // Published: the slot the record carries.
        let diameter = EROSION_LOOKUP_DIAMETER as i64;
        let record = ((1 * diameter + 2) * 4) as usize;
        records[record] = 5.0;
        records[record + 1] = 6.0;
        records[record + 3] = 1.0;
        assert_eq!(published_slot(tile(2, 1), (0, 0), &records), Some((5.0, 6.0)));
        // Outside the 11x11 window: no slot at all.
        assert_eq!(published_slot(tile(100, 0), (0, 0), &records), None);
        assert_eq!(published_slot(tile(0, -100), (0, 0), &records), None);
    }
}