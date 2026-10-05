//! Fixed world chunks with asynchronous, bounded grass preparation.
//!
//! Slots belong to chunk/layer pairs rather than camera anchors. Crossing an
//! anchor boundary queues only missing chunks, and a small retention ring
//! keeps recently visited chunks available when the player turns back.

use std::collections::{HashMap, HashSet};

use bevy::tasks::{AsyncComputeTaskPool, Task, block_on, poll_once};

use crate::grass::{self, GrassInstance, LAYER_COUNT};

/// Begin loading beyond the candidate ring before it can contribute a pixel.
const PREFETCH_METRES: f64 = 8.0;
/// Keep completed chunks for another grid cell to avoid boundary oscillation.
const RETAIN_METRES: f64 = 16.0;
pub const MAX_IN_FLIGHT: usize = 16;
pub const MAX_STARTS_PER_UPDATE: usize = 16;
pub const MAX_UPLOADS_PER_UPDATE: usize = 16;
pub const MAX_UPLOAD_BYTES_PER_UPDATE: usize = 2 * 1024 * 1024;
const CHUNK_METRES: f64 = grass::CHUNK_CELLS as f64 * grass::SCATTER_CELL_SIZE as f64;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GrassChunkKey {
    /// Permanent world-grid coordinates; negative cells use Euclidean division.
    pub coordinate: [i64; 2],
    pub layer: usize,
}

impl GrassChunkKey {
    pub fn centre(self) -> [f32; 2] {
        self.coordinate
            .map(|value| ((value as f64 + 0.5) * CHUNK_METRES) as f32)
    }

    /// Conservative root reach after world coordinates are rounded to the
    /// instance buffer's f32 format. Far from the origin, that rounding can
    /// be wider than one scatter cell or even an entire chunk.
    pub fn root_radius(self) -> f32 {
        let centre = self.centre();
        let half_extent: [f32; 2] = std::array::from_fn(|axis| {
            let low = self.coordinate[axis] as f64 * CHUNK_METRES;
            let high = low + CHUNK_METRES;
            (low as f32 - centre[axis])
                .abs()
                .max((high as f32 - centre[axis]).abs())
        });
        half_extent[0].hypot(half_extent[1]) + 1.0
    }
}

/// One bounded GPU write. Every model shares this layer's fixed-size slot;
/// its ranges within `instances` are available through `ReadySlot`.
pub struct ChunkUpload {
    pub key: GrassChunkKey,
    pub slot: usize,
    pub instances: Vec<GrassInstance>,
}

/// CPU draw metadata for a slot whose upload has been returned by `update`.
pub struct ReadySlot<'a> {
    pub key: GrassChunkKey,
    pub slot: usize,
    pub model_starts: &'a [u32],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct JobTicket {
    key: GrassChunkKey,
    slot: usize,
    generation: u64,
}

enum SlotState {
    Queued,
    Generating,
    Ready(Vec<u32>),
}

struct Resident {
    ticket: JobTicket,
    state: SlotState,
}

/// Slot ownership and selection are independent of threads and GPU resources.
struct Residency {
    anchor: Option<[i64; 2]>,
    capacities: [usize; LAYER_COUNT],
    free: [Vec<usize>; LAYER_COUNT],
    chunks: HashMap<GrassChunkKey, Resident>,
    wanted: HashSet<GrassChunkKey>,
    /// Farthest first, so the closest entering chunk is popped in O(1).
    queued: Vec<JobTicket>,
    generation: u64,
}

impl Residency {
    fn new() -> Self {
        // An 8 m anchor has four possible alignments to the permanent 16 m
        // grid. Taking their exact maximum bounds every retention ring without
        // allocating the corners of a large enclosing square for each layer.
        let capacities = std::array::from_fn(|layer| {
            let radius = retained_radius(layer);
            [[0, 0], [8, 0], [0, 8], [8, 8]]
                .map(|anchor| keys_in_radius(anchor, layer, radius).len())
                .into_iter()
                .max()
                .unwrap()
        });
        Self {
            anchor: None,
            capacities,
            free: std::array::from_fn(|layer| (0..capacities[layer]).rev().collect()),
            chunks: HashMap::new(),
            wanted: HashSet::new(),
            queued: Vec::new(),
            generation: 0,
        }
    }

    fn plan(&mut self, anchor: [i64; 2]) {
        if self.anchor == Some(anchor) {
            return;
        }
        self.anchor = Some(anchor);
        self.wanted.clear();
        for layer in 0..LAYER_COUNT {
            self.wanted
                .extend(keys_in_radius(anchor, layer, desired_radius(layer)));
        }

        // Evict before assigning entering chunks, so slot counts remain fixed
        // even during a teleport. In-flight work is tracked separately until
        // it finishes; it cannot overwrite a slot assigned to its replacement.
        self.chunks.retain(|key, resident| {
            if distance_squared(key.coordinate, anchor) <= retained_radius(key.layer).powi(2) {
                true
            } else {
                self.free[key.layer].push(resident.ticket.slot);
                false
            }
        });
        for &key in &self.wanted {
            self.chunks.entry(key).or_insert_with(|| {
                let slot = self.free[key.layer]
                    .pop()
                    .expect("grass retention capacity exceeded");
                self.generation += 1;
                Resident {
                    ticket: JobTicket {
                        key,
                        slot,
                        generation: self.generation,
                    },
                    state: SlotState::Queued,
                }
            });
        }
        self.queued = self
            .wanted
            .iter()
            .filter_map(|key| {
                let resident = &self.chunks[key];
                matches!(resident.state, SlotState::Queued).then_some(resident.ticket)
            })
            .collect();
        self.queued.sort_unstable_by(|a, b| {
            distance_squared(b.key.coordinate, anchor)
                .total_cmp(&distance_squared(a.key.coordinate, anchor))
                .then_with(|| b.key.layer.cmp(&a.key.layer))
                .then_with(|| b.key.coordinate.cmp(&a.key.coordinate))
        });
    }

    fn next_job(&mut self) -> Option<JobTicket> {
        let ticket = self.queued.pop()?;
        self.chunks.get_mut(&ticket.key).unwrap().state = SlotState::Generating;
        Some(ticket)
    }

    fn owns(&self, ticket: JobTicket) -> bool {
        self.chunks
            .get(&ticket.key)
            .is_some_and(|resident| resident.ticket == ticket)
    }

    fn complete(&mut self, ticket: JobTicket, model_starts: &[u32]) -> bool {
        let Some(resident) = self.chunks.get_mut(&ticket.key) else {
            return false;
        };
        if resident.ticket != ticket {
            return false;
        }
        resident.state = SlotState::Ready(model_starts.to_vec());
        true
    }

    fn ready(&self) -> bool {
        self.anchor.is_some()
            && self
                .wanted
                .iter()
                .all(|key| matches!(self.chunks[key].state, SlotState::Ready(_)))
    }
}

fn desired_radius(layer: usize) -> f64 {
    grass::LAYER_CANDIDATE_RADIUS[layer] as f64 + PREFETCH_METRES
}

fn retained_radius(layer: usize) -> f64 {
    desired_radius(layer) + RETAIN_METRES
}

/// Nearest distance to the closed root AABB. A point on a chunk edge may
/// contribute to either neighbour, so including both is conservative.
fn distance_squared(coordinate: [i64; 2], anchor: [i64; 2]) -> f64 {
    coordinate
        .into_iter()
        .zip(anchor)
        .map(|(chunk, value)| {
            let low = chunk as f64 * CHUNK_METRES;
            let high = low + CHUNK_METRES;
            let value = value as f64;
            (low - value).max(value - high).max(0.0).powi(2)
        })
        .sum()
}

fn keys_in_radius(anchor: [i64; 2], layer: usize, radius: f64) -> Vec<GrassChunkKey> {
    let centre = anchor
        .map(|value| {
            // Scatter cells are metres today; retain the correct floor before
            // integer Euclidean division if their scale changes later.
            (value as f64 / grass::SCATTER_CELL_SIZE as f64).floor() as i64
        })
        .map(|cell| cell.div_euclid(grass::CHUNK_CELLS));
    let span = (radius / CHUNK_METRES).ceil() as i64 + 1;
    let mut keys = Vec::new();
    for z in centre[1] - span..=centre[1] + span {
        for x in centre[0] - span..=centre[0] + span {
            let coordinate = [x, z];
            if distance_squared(coordinate, anchor) <= radius * radius {
                keys.push(GrassChunkKey { coordinate, layer });
            }
        }
    }
    keys
}

struct ChunkData {
    instances: Vec<GrassInstance>,
    model_starts: Vec<u32>,
}

impl ChunkData {
    fn generate(key: GrassChunkKey, model_count: usize) -> Self {
        let groups = grass::scatter_world_chunk(key.coordinate, key.layer, model_count);
        let count = groups.iter().map(Vec::len).sum();
        let mut instances = Vec::with_capacity(count);
        let mut model_starts = Vec::with_capacity(model_count + 1);
        model_starts.push(0);
        for group in groups {
            instances.extend(group);
            model_starts.push(instances.len() as u32);
        }
        Self {
            instances,
            model_starts,
        }
    }

    fn bytes(&self) -> usize {
        self.instances.len() * std::mem::size_of::<GrassInstance>()
    }
}

struct InFlight {
    ticket: JobTicket,
    task: Option<Task<ChunkData>>,
    /// A completed result can wait here when this frame's upload budget is
    /// used up. These results still count against the in-flight limit.
    completed: Option<ChunkData>,
}

pub struct GrassStream {
    model_count: usize,
    valid_center: bool,
    residency: Residency,
    pending: Vec<InFlight>,
}

impl GrassStream {
    pub fn new(model_count: usize) -> Self {
        Self {
            model_count,
            valid_center: false,
            residency: Residency::new(),
            pending: Vec::new(),
        }
    }

    pub fn slot_capacities(&self) -> [usize; LAYER_COUNT] {
        self.residency.capacities
    }

    pub fn anchor(&self) -> Option<[i64; 2]> {
        self.residency.anchor
    }

    /// Every desired slot is prepared. The renderer must apply the returned
    /// uploads before publishing this readiness to capture/automation systems.
    pub fn ready(&self) -> bool {
        self.model_count == 0 || (self.valid_center && self.residency.ready())
    }

    pub fn slots(&self) -> impl Iterator<Item = ReadySlot<'_>> {
        self.residency.chunks.values().filter_map(|resident| {
            let SlotState::Ready(model_starts) = &resident.state else {
                return None;
            };
            Some(ReadySlot {
                key: resident.ticket.key,
                slot: resident.ticket.slot,
                model_starts,
            })
        })
    }

    /// Replan only on an 8 m anchor boundary, poll without waiting, upload at
    /// most 2 MiB, and dispatch a bounded number of entering chunk layers.
    /// Candidate generation and model grouping both run on worker threads.
    pub fn update(&mut self, center: [f32; 2]) -> Vec<ChunkUpload> {
        if self.model_count == 0
            || center
                .iter()
                .any(|value| !value.is_finite() || value.abs() > 1e9)
        {
            self.valid_center = false;
            return Vec::new();
        }
        self.valid_center = true;
        self.residency.plan(grass::scatter_anchor(center));
        let mut uploads = Vec::new();
        let mut uploaded_bytes = 0;
        let mut index = 0;
        while index < self.pending.len() {
            let pending = &mut self.pending[index];
            if pending.completed.is_none() {
                if let Some(task) = &mut pending.task {
                    pending.completed = block_on(poll_once(task));
                    if pending.completed.is_some() {
                        pending.task = None;
                    }
                }
            }
            let Some(data) = &pending.completed else {
                index += 1;
                continue;
            };
            if !self.residency.owns(pending.ticket) {
                self.pending.swap_remove(index);
                continue;
            }
            let bytes = data.bytes();
            if uploads.len() >= MAX_UPLOADS_PER_UPDATE
                || uploaded_bytes + bytes > MAX_UPLOAD_BYTES_PER_UPDATE
            {
                index += 1;
                continue;
            }
            let pending = self.pending.swap_remove(index);
            let data = pending.completed.unwrap();
            if self.residency.complete(pending.ticket, &data.model_starts) {
                uploaded_bytes += bytes;
                uploads.push(ChunkUpload {
                    key: pending.ticket.key,
                    slot: pending.ticket.slot,
                    instances: data.instances,
                });
            }
        }
        for _ in 0..MAX_STARTS_PER_UPDATE {
            if self.pending.len() >= MAX_IN_FLIGHT {
                break;
            }
            let Some(ticket) = self.residency.next_job() else {
                break;
            };
            let model_count = self.model_count;
            let task = AsyncComputeTaskPool::get()
                .spawn(async move { ChunkData::generate(ticket.key, model_count) });
            self.pending.push(InFlight {
                ticket,
                task: Some(task),
                completed: None,
            });
        }
        uploads
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn finish_everything(residency: &mut Residency) {
        while let Some(ticket) = residency.next_job() {
            assert!(residency.complete(ticket, &[0, 256]));
        }
        assert!(residency.ready());
    }

    #[test]
    fn a_boundary_queues_only_entering_chunk_layers_and_preserves_slots() {
        let mut residency = Residency::new();
        residency.plan([0, 0]);
        finish_everything(&mut residency);
        let old: HashMap<_, _> = residency
            .chunks
            .iter()
            .map(|(key, resident)| (*key, resident.ticket.slot))
            .collect();
        residency.plan([8, 0]);
        let entering: HashSet<_> = residency
            .wanted
            .iter()
            .filter(|key| !old.contains_key(key))
            .copied()
            .collect();
        let queued: HashSet<_> = residency.queued.iter().map(|ticket| ticket.key).collect();
        assert_eq!(queued, entering);
        assert!(!queued.is_empty());
        assert!(
            queued.len() < old.len() / 10,
            "queued {} of {}",
            queued.len(),
            old.len()
        );
        for (key, resident) in &residency.chunks {
            if let Some(&slot) = old.get(key) {
                assert_eq!(slot, resident.ticket.slot);
                assert!(matches!(resident.state, SlotState::Ready(_)));
            }
        }
    }

    #[test]
    fn fixed_grid_coordinates_include_the_correct_negative_neighbours() {
        assert_eq!(
            GrassChunkKey {
                coordinate: [-1, -2],
                layer: 0
            }
            .centre(),
            [-8.0, -24.0]
        );
        let keys = keys_in_radius([-8, -8], 0, 0.0);
        assert_eq!(
            keys,
            [GrassChunkKey {
                coordinate: [-1, -1],
                layer: 0
            }]
        );
        let keys = keys_in_radius([-16, -16], 0, 0.0);
        let coordinates: HashSet<_> = keys.iter().map(|key| key.coordinate).collect();
        assert_eq!(
            coordinates,
            HashSet::from([[-2, -2], [-1, -2], [-2, -1], [-1, -1]])
        );
    }

    #[test]
    fn generated_roots_fit_the_culling_radius_after_large_coordinate_rounding() {
        for coordinate in [
            [0, 0],
            [-1, -1],
            [62_499_997, 62_499_999],
            [62_499_998, -62_499_998],
            [62_499_999, -62_499_999],
            [62_500_000, 62_500_001],
            [-62_500_001, 62_500_002],
        ] {
            let key = GrassChunkKey {
                coordinate,
                layer: 3,
            };
            let centre = key.centre();
            let radius = key.root_radius();
            let instances = grass::scatter_world_chunk(coordinate, key.layer, 3);
            for instance in instances.iter().flatten() {
                let distance = (instance.xz[0] - centre[0]).hypot(instance.xz[1] - centre[1]);
                assert!(distance <= radius, "{coordinate:?}: {distance} > {radius}");
            }
        }
    }

    #[test]
    fn desired_chunks_cover_every_layer_fade_at_all_camera_anchor_corners() {
        for anchor in [[0, 0], [8, 8], [-1_000_000, 1_000_000], [999_992, -999_992]] {
            for layer in 0..LAYER_COUNT {
                let wanted: HashSet<_> = keys_in_radius(anchor, layer, desired_radius(layer))
                    .into_iter()
                    .collect();
                let centre = anchor.map(|value| value.div_euclid(grass::CHUNK_CELLS));
                let span = (grass::LAYER_END[layer] as f64 / CHUNK_METRES).ceil() as i64 + 2;
                for corner in [
                    [-3.999, -3.999],
                    [-3.999, 3.999],
                    [3.999, -3.999],
                    [3.999, 3.999],
                ] {
                    let camera = [anchor[0] as f64 + corner[0], anchor[1] as f64 + corner[1]];
                    for z in centre[1] - span..=centre[1] + span {
                        for x in centre[0] - span..=centre[0] + span {
                            let coordinate = [x, z];
                            let nearest: [f64; 2] = std::array::from_fn(|axis| {
                                let low = coordinate[axis] as f64 * CHUNK_METRES;
                                camera[axis].clamp(low, low + CHUNK_METRES)
                            });
                            if (nearest[0] - camera[0]).hypot(nearest[1] - camera[1])
                                <= grass::LAYER_END[layer] as f64
                            {
                                assert!(
                                    wanted.contains(&GrassChunkKey { coordinate, layer }),
                                    "omitted reachable layer {layer} chunk {coordinate:?} for camera {camera:?}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn invalid_centres_cannot_publish_a_previous_ready_anchor() {
        let mut stream = GrassStream::new(1);
        stream.residency.plan([0, 0]);
        finish_everything(&mut stream.residency);
        assert!(stream.update([0.0, 0.0]).is_empty());
        assert!(stream.ready());
        for centre in [[f32::NAN, 0.0], [f32::INFINITY, 0.0], [1.1e9, 0.0]] {
            assert!(stream.update(centre).is_empty());
            assert!(!stream.ready());
        }
        assert!(stream.update([0.0, 0.0]).is_empty());
        assert!(stream.ready());
    }

    #[test]
    fn turning_back_inside_the_retention_ring_reuses_every_ready_chunk() {
        let mut residency = Residency::new();
        residency.plan([0, 0]);
        finish_everything(&mut residency);
        let original: HashMap<_, _> = residency
            .chunks
            .iter()
            .map(|(key, resident)| (*key, resident.ticket))
            .collect();
        residency.plan([8, 0]);
        finish_everything(&mut residency);
        residency.plan([0, 0]);
        assert!(residency.queued.is_empty());
        assert!(residency.ready());
        for (key, ticket) in original {
            assert_eq!(residency.chunks[&key].ticket, ticket);
        }
    }

    #[test]
    fn stale_completions_cannot_fill_an_evicted_or_reused_slot() {
        let mut residency = Residency::new();
        residency.plan([0, 0]);
        let stale = residency.next_job().unwrap();
        residency.plan([4096, 4096]);
        assert!(!residency.complete(stale, &[0, 256]));
        residency.plan([0, 0]);
        let replacement = residency.chunks[&stale.key].ticket;
        assert_ne!(stale.generation, replacement.generation);
        assert!(!residency.complete(stale, &[0, 256]));
        assert!(matches!(
            residency.chunks[&stale.key].state,
            SlotState::Queued
        ));
        assert!(residency.complete(replacement, &[0, 256]));
    }

    #[test]
    fn retained_work_finishing_outside_the_wanted_ring_is_ready_when_reentered() {
        let mut residency = Residency::new();
        residency.plan([0, 0]);
        let mut tickets = Vec::new();
        while let Some(ticket) = residency.next_job() {
            tickets.push(ticket);
        }
        residency.plan([8, 0]);
        let retained = tickets
            .into_iter()
            .find(|ticket| !residency.wanted.contains(&ticket.key) && residency.owns(*ticket))
            .expect("movement should leave some chunks in the retention ring");
        assert!(residency.complete(retained, &[0, 17, 256]));
        residency.plan([0, 0]);
        assert!(residency.wanted.contains(&retained.key));
        assert!(residency.owns(retained));
        let SlotState::Ready(starts) = &residency.chunks[&retained.key].state else {
            panic!("retained completion was discarded");
        };
        assert_eq!(starts, &[0, 17, 256]);
        assert!(
            !residency
                .queued
                .iter()
                .any(|ticket| ticket.key == retained.key)
        );
    }

    #[test]
    fn retention_capacities_bound_all_anchor_phases_movement_and_teleports() {
        let mut residency = Residency::new();
        for anchor in [
            [0, 0],
            [8, 0],
            [8, 8],
            [0, 8],
            [-8, -8],
            [-16, -8],
            [-24, 0],
            [4096, -8192],
            [4104, -8184],
            [-999_992, 999_992],
        ] {
            residency.plan(anchor);
            let mut counts = [0; LAYER_COUNT];
            let mut used = [
                HashSet::new(),
                HashSet::new(),
                HashSet::new(),
                HashSet::new(),
            ];
            for (key, resident) in &residency.chunks {
                counts[key.layer] += 1;
                assert!(used[key.layer].insert(resident.ticket.slot));
            }
            for layer in 0..LAYER_COUNT {
                assert!(counts[layer] <= residency.capacities[layer]);
                assert_eq!(
                    counts[layer] + residency.free[layer].len(),
                    residency.capacities[layer]
                );
            }
        }
        // The foreground layer gets a tight circle allocation, rather than
        // inheriting the far field's hundreds of unnecessary slots.
        assert!(residency.capacities[3] < 100);
    }

    #[test]
    fn update_bounds_queued_work_and_completed_upload_bytes() {
        let mut stream = GrassStream::new(1);
        stream.residency.plan([0, 0]);
        // Finished jobs exercise the update budgets without timing-dependent
        // task-pool tests or starting worker threads.
        stream
            .residency
            .queued
            .retain(|ticket| ticket.key.layer == 3);
        for _ in 0..MAX_IN_FLIGHT {
            let ticket = stream.residency.next_job().unwrap();
            let instance = GrassInstance {
                xz: [0.0, 0.0],
                rotation: 0.0,
                scale: 1.0,
                tint: 1.0,
                seed: 0.0,
                _pad: [1.0, ticket.key.layer as f32],
            };
            stream.pending.push(InFlight {
                ticket,
                task: None,
                completed: Some(ChunkData {
                    instances: vec![instance; 14_336],
                    model_starts: vec![0, 14_336],
                }),
            });
        }
        // Suppress new dispatches: the real scheduler cannot exceed the same
        // fixed in-flight capacity regardless of remaining queued work.
        stream.residency.queued.clear();
        let uploads = stream.update([0.0, 0.0]);
        assert!(!uploads.is_empty());
        assert!(uploads.len() <= MAX_UPLOADS_PER_UPDATE);
        let bytes: usize = uploads
            .iter()
            .map(|upload| upload.instances.len() * std::mem::size_of::<GrassInstance>())
            .sum();
        assert!(bytes <= MAX_UPLOAD_BYTES_PER_UPDATE);
        assert_eq!(uploads.len() + stream.pending.len(), MAX_IN_FLIGHT);
        assert!(
            stream
                .pending
                .iter()
                .all(|pending| pending.completed.is_some())
        );
    }
}
