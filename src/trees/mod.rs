//! The fir forest: the scatter that decides where trees stand, the catalogue
//! that knows what they look like, and the collision that stops the player
//! walking through them.
//!
//! # The division of labour
//!
//! This module is main-world only. It never touches a GPU resource; it decides
//! *what* exists and hands the list to `src/render/tree_node.rs`, which owns
//! every buffer, pipeline and draw. The split matters because the two halves
//! have completely different lifetimes: the scatter changes when a chunk
//! streams in (rarely, and never in the middle of a frame's drawing), while
//! LOD bucketing and instance uploads happen every time the player walks a few
//! metres.
//!
//! # Why the scatter is chunked and deterministic
//!
//! A tree's position, species, yaw, scale and tint are all derived from the
//! integer hash of one lattice cell (see [`placement`]). Nothing is stored
//! between runs, nothing is randomised at runtime, and a chunk can be dropped
//! and rebuilt later without the forest changing — the same guarantees the
//! erosion simulation gives its own tiles, and for the same reason: streaming
//! world content that drifts when it is rebuilt is worse than content that is
//! simply missing.
//!
//! # Adding more vegetation
//!
//! [`TreeField`] stores one [`Vec<TreePlacement>`] per chunk per *kind*, and
//! the render node draws one instanced call per (kind, variant, LOD) group. A
//! second species — bushes, grass tufts, rocks — needs its own catalogue entry,
//! its own group count and its own `evaluate_site` rules; it does not need a
//! new streaming model, a new chunk lifetime, or a new collision structure.

pub mod placement;

use crate::erosion::ErosionCache;
use crate::noise::NoiseField;
use crate::player::Player;
use crate::render::glb::{GlbFile, GlbImage};
use bevy::prelude::*;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Species grid
// ---------------------------------------------------------------------------

pub const TREE_SIZE_COUNT: usize = 3;
pub const TREE_VARIANT_COUNT: usize = 3;
pub const TREE_LOD_COUNT: usize = 4;

/// One instanced draw per (size, variant, LOD). The render node keeps a vertex
/// buffer, an index buffer, one instance buffer per group, and one material
/// bind group per (group, material).
pub const TREE_GROUP_COUNT: usize = TREE_SIZE_COUNT * TREE_VARIANT_COUNT * TREE_LOD_COUNT;

/// `small`, `medium`, `large`, in the order the extracted files name them.
pub const TREE_SIZE_NAMES: [&str; TREE_SIZE_COUNT] = ["small", "medium", "large"];

pub const fn tree_group(size: usize, variant: usize, lod: usize) -> usize {
    (size * TREE_VARIANT_COUNT + variant) * TREE_LOD_COUNT + lod
}

/// Trunk radius by size class, in metres at the pack's authored scale. Used
/// by collision only: the renderer draws the real trunk, so this is the
/// player's clearance, not the tree's geometry. A fir's trunk is roughly 2-3%
/// of its height in radius, which is what these are.
pub const TREE_TRUNK_RADIUS: [f32; TREE_SIZE_COUNT] = [0.11, 0.17, 0.26];

/// The player's own radius, added to the trunk's to get the standoff.
pub const PLAYER_RADIUS: f32 = 0.35;

// ---------------------------------------------------------------------------
// Per-instance ABI
// ---------------------------------------------------------------------------

/// One scattered tree as the vertex shader sees it. 28 bytes, mirrored by
/// `TreeInstance` in assets/shaders/tree-vs.wgsl, which declares the same six
/// fields at `@location(4..9)`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TreeInstance {
    /// World XZ.
    pub centre: [f32; 2],
    /// World Y of the ground under `centre`, as the scatter measured it.
    ///
    /// The vertex shader plants the trunk here rather than calling
    /// `terrainHeight`, which would cost the full height model — twenty-odd
    /// noise samples and four erosion atlas fetches — on every vertex of every
    /// instance, several times over what the whole terrain clipmap costs. See
    /// the header of tree-vs.wgsl for why the two agree.
    pub ground: f32,
    /// Uniform object-space scale.
    pub scale: f32,
    /// Yaw in radians.
    pub rotation: f32,
    /// 0..1 stable per-tree random, driving the fragment stage's tint.
    pub variation: f32,
    /// Metres the trunk foot is pushed below ground, so a tree on a slope is
    /// seated into it rather than balanced on its downhill edge.
    pub sink: f32,
}

const _: () = assert!(std::mem::size_of::<TreeInstance>() == 28);

/// A scattered tree before LOD selection. This is what the main world stores
/// and what crosses to the render world; the render node turns it into a
/// [`TreeInstance`] once it has decided which LOD the camera sees.
#[derive(Clone, Copy, Debug)]
pub struct TreePlacement {
    pub x: f32,
    pub z: f32,
    /// Ground height at (x, z), from the same `sample_eroded_height` call that
    /// judged the site.
    pub ground: f32,
    pub size: u8,
    pub variant: u8,
    pub scale: f32,
    pub rotation: f32,
    pub variation: f32,
    pub sink: f32,
}

/// The whole forest, shared with the render world behind an `Arc` so extraction
/// is a pointer clone rather than a copy of the tree list.
#[derive(Default)]
pub struct TreeScatter {
    /// Bumped whenever the contents change; the render node compares it against
    /// the version its instance buffers hold.
    pub generation: u64,
    pub placements: Vec<TreePlacement>,
    /// LOD-0 trunk height per (size, variant), in metres. The LOD distance
    /// bands are scaled by it so a small fir swaps to a billboard at the same
    /// *apparent* size a large one does.
    pub heights: [[f32; TREE_VARIANT_COUNT]; TREE_SIZE_COUNT],
}

// ---------------------------------------------------------------------------
// Asset payload (main world -> render world, once)
// ---------------------------------------------------------------------------

/// One texture, decoded and with its mip chain built, ready for upload.
///
/// The chain is built here rather than on the GPU because the trees' base
/// colour textures are sRGB and wgpu cannot generate mips for an sRGB view
/// without the write path going through the same transfer function twice.
/// Building it on the CPU also matches what the terrain's PBR arrays already
/// do, so there is one story for mip generation in this renderer.
pub struct TreeImage {
    pub width: u32,
    pub height: u32,
    /// Level 0 (full size) first. Level n is `max(1, width >> n)` square-ish.
    pub levels: Vec<Vec<u8>>,
    /// Whether the bytes are sRGB-encoded. Base colour is; normal maps are not.
    pub srgb: bool,
}

/// One drawable range inside a mesh's index buffer, with the material it draws.
pub struct TreeMaterialRange {
    pub first_index: u32,
    pub index_count: u32,
    pub is_branch: bool,
    pub roughness: f32,
    pub alpha_cutoff: f32,
    /// The material's `KHR_materials_specular` factor, carried to the tree
    /// pass so foliage can be given the near-zero specular the pack authors
    /// for it. See `GlbMaterial::specular_factor`.
    pub specular_factor: f32,
    pub base_colour: Option<Arc<TreeImage>>,
    pub normal: Option<Arc<TreeImage>>,
}

/// One (size, variant, LOD) mesh: every primitive's vertices merged into one
/// buffer, with a [`TreeMaterialRange`] per primitive so each material is a
/// separate indexed draw over its own slice.
///
/// Merging the primitives is what lets one vertex buffer serve the whole group:
/// a fir's bark and its branch cards are the same five vertex attributes in the
/// same format, and one buffer per group keeps the tree pass's vertex-buffer
/// bindings to a single `set_vertex_buffer` per draw.
pub struct TreeMeshData {
    pub vertices: Vec<crate::render::glb::GlbVertex>,
    pub indices: Vec<u32>,
    pub ranges: Vec<TreeMaterialRange>,
    /// Authored trunk height in metres.
    pub height: f32,
}

/// Every mesh, indexed by [`tree_group`]. Produced once at startup and removed
/// from the main world by the extraction system.
#[derive(Resource)]
pub struct TreeAssetData {
    pub meshes: Vec<TreeMeshData>,
}

// ---------------------------------------------------------------------------
// Streaming state
// ---------------------------------------------------------------------------

/// The fir trunk field, as a spatial hash of cylinders. Only the trunk
/// collides; the canopy is above head height and blocking on it would make the
/// forest feel like a maze of invisible walls.
#[derive(Default)]
struct TrunkIndex {
    cells: HashMap<[i32; 2], Vec<(f32, f32, f32)>>,
}

/// Cell side of the trunk hash. Comfortably larger than the player's per-frame
/// step, so a query only ever has to look at the four cells around them.
const TREE_TRUNK_CELL: f32 = 16.0;

impl TrunkIndex {
    fn key(x: f32, z: f32) -> [i32; 2] {
        [
            (x / TREE_TRUNK_CELL).floor() as i32,
            (z / TREE_TRUNK_CELL).floor() as i32,
        ]
    }

    fn insert(&mut self, x: f32, z: f32, radius: f32) {
        self.cells.entry(Self::key(x, z)).or_default().push((x, z, radius));
    }

    /// The trunk the point is inside, if any, as an outward push vector.
    ///
    /// The push is radial, so a player walking into a trunk keeps their
    /// tangential motion and slides around it — the alternative, cancelling
    /// the whole step, stops them dead against a tree they only grazed.
    fn push_out(&self, x: f32, z: f32) -> Option<[f32; 2]> {
        let centre = Self::key(x, z);
        for dz in -1..=1 {
            for dx in -1..=1 {
                let Some(trunks) = self.cells.get(&[centre[0] + dx, centre[1] + dz]) else {
                    continue;
                };
                for &(trunk_x, trunk_z, radius) in trunks {
                    let offset_x = x - trunk_x;
                    let offset_z = z - trunk_z;
                    let distance_squared = offset_x * offset_x + offset_z * offset_z;
                    let standoff = radius + PLAYER_RADIUS;
                    if distance_squared >= standoff * standoff {
                        continue;
                    }
                    // Exactly on the axis: any direction will do, and up is
                    // the one that cannot be wrong for long.
                    if distance_squared < 1.0e-8 {
                        return Some([standoff, 0.0]);
                    }
                    let distance = distance_squared.sqrt();
                    let scale = (standoff - distance) / distance;
                    return Some([offset_x * scale, offset_z * scale]);
                }
            }
        }
        None
    }
}

/// The main world's forest: which chunks have been scattered, the flattened
/// placement list the render world reads, and the trunk index collision uses.
#[derive(Resource, Default)]
pub struct TreeField {
    /// One placement list per scattered chunk. Kept per chunk so eviction is
    /// a single `remove` rather than a scan of the flat list.
    chunks: HashMap<[i32; 2], Vec<TreePlacement>>,
    /// Chunks whose erosion tiles are not all final yet. Re-offered
    /// periodically; see [`TREE_CHUNK_RETRY_FRAMES`].
    deferred: HashMap<[i32; 2], u64>,
    /// Chunks waiting to be scattered, nearest to the player first.
    queue: Vec<[i32; 2]>,
    /// The chunk the queue was built for. When the player crosses into a new
    /// chunk the queue is rebuilt; between crossings it is drained.
    queued_around: Option<[i32; 2]>,
    trunks: TrunkIndex,
    scatter: Arc<TreeScatter>,
    generation: u64,
    frame: u64,
    /// Startup diagnostic: wall time spent scattering chunks and the number of
    /// frames it took. Scattering is the only part of the tree system that
    /// runs on the main thread between frames, so this is the hitch the player
    /// actually feels as the forest appears — and it is the number that moves
    /// when [`placement::TREE_LATTICE`] or `TREE_CHUNKS_PER_FRAME` change.
    /// Logged once, when the initial fill finishes, and never touched again.
    fill_seconds: f32,
    fill_frames: u32,
    fill_reported: bool,
}

impl TreeField {
    pub fn scatter(&self) -> &Arc<TreeScatter> {
        &self.scatter
    }

    /// Every tree currently in the field. Diagnostics only.
    pub fn tree_count(&self) -> usize {
        self.scatter.placements.len()
    }

    /// Rebuild the flat list and the trunk index from the per-chunk lists, and
    /// publish the result to the render world.
    fn republish(&mut self, heights: [[f32; TREE_VARIANT_COUNT]; TREE_SIZE_COUNT], focus: Vec3) {
        self.generation += 1;
        let mut placements = Vec::with_capacity(self.chunks.values().map(Vec::len).sum());
        let mut trunks = TrunkIndex::default();
        for chunk in self.chunks.values() {
            for placement in chunk {
                let radius = TREE_TRUNK_RADIUS[placement.size as usize % TREE_SIZE_COUNT]
                    * placement.scale;
                trunks.insert(placement.x, placement.z, radius);
                placements.push(*placement);
            }
        }
        self.trunks = trunks;
        // The nearest few trees, for aiming a `--camera` capture at one
        // without having to guess coordinates — so "near" means near the
        // player, not near the world origin, which is an arbitrary point the
        // player may never visit. Debug level: it changes on every chunk
        // crossing and is not a number anyone watches in normal play.
        //
        // Sorted from `placements`, the list just built, not from
        // `self.scatter.placements`, which is still the previous generation's
        // and is empty on the first call.
        let mut nearest: Vec<&TreePlacement> = placements.iter().collect();
        nearest.sort_by(|left, right| {
            let a = (left.x - focus.x).powi(2) + (left.z - focus.z).powi(2);
            let b = (right.x - focus.x).powi(2) + (right.z - focus.z).powi(2);
            a.total_cmp(&b)
        });
        let sample: Vec<String> = nearest
            .iter()
            .take(4)
            .map(|placement| {
                format!(
                    "({:.1}, {:.1}) h={:.1} size={} ground={:.1}",
                    placement.x,
                    placement.z,
                    heights[placement.size as usize % TREE_SIZE_COUNT][placement.variant as usize % TREE_VARIANT_COUNT]
                        * placement.scale,
                    placement.size,
                    placement.ground,
                )
            })
            .collect();
        log::debug!(
            "TREES: {} chunks scattered, {} deferred, {} queued; nearest to player: {}",
            self.chunks.len(),
            self.deferred.len(),
            self.queue.len(),
            sample.join(", ")
        );

        self.scatter = Arc::new(TreeScatter {
            generation: self.generation,
            placements,
            heights,
        });
    }
}

/// Chunks scattered per frame. Each one costs a height grid plus one
/// `sample_eroded_height` per accepted candidate, and that height model is the
/// most expensive CPU function in the app (twenty-odd noise samples apiece).
/// Four a frame fills a fresh view in about a second without a visible stall.
const TREE_CHUNKS_PER_FRAME: usize = 4;

/// How long a chunk whose erosion tiles are still simulating waits before it
/// is offered again. Long enough that the retry is not the thing that costs
/// frames, short enough that the forest fills in as the tiles finalise.
const TREE_CHUNK_RETRY_FRAMES: u64 = 30;

// ---------------------------------------------------------------------------
// Scatter
// ---------------------------------------------------------------------------

fn chunk_of(position: Vec3) -> [i32; 2] {
    [
        (position.x / placement::TREE_CHUNK_SIZE).floor() as i32,
        (position.z / placement::TREE_CHUNK_SIZE).floor() as i32,
    ]
}

fn chunk_centre(chunk: [i32; 2]) -> [f32; 2] {
    [
        (chunk[0] as f32 + 0.5) * placement::TREE_CHUNK_SIZE,
        (chunk[1] as f32 + 0.5) * placement::TREE_CHUNK_SIZE,
    ]
}

/// Scatter one chunk. Pure: everything it reads is world-space, and the only
/// state it touches is the erosion cache, which is the ground itself.
fn scatter_chunk(
    cache: &ErosionCache,
    noise: &NoiseField,
    chunk: [i32; 2],
    heights: &[[f32; TREE_VARIANT_COUNT]; TREE_SIZE_COUNT],
) -> Vec<TreePlacement> {
    // The chunk's own centre, never the player's: see `evaluate_site`.
    let visibility_center = chunk_centre(chunk);
    let grid = placement::build_height_grid(cache, noise, chunk, visibility_center);
    let cell_count = grid.side;

    let mut placements = Vec::new();
    // Rejection tally, reported once per chunk at debug level. `candidates` is
    // the denominator every other number here is read against.
    let mut rejected = [0usize; 6];
    for iz in 0..cell_count {
        for ix in 0..cell_count {
            let lattice = [
                chunk[0] as f32 * placement::TREE_CHUNK_SIZE
                    + ix as f32 * placement::TREE_LATTICE,
                chunk[1] as f32 * placement::TREE_CHUNK_SIZE
                    + iz as f32 * placement::TREE_LATTICE,
            ];
            let randoms = placement::cell_randoms(lattice);
            // Jitter stays inside the middle 80% of the cell so two candidates
            // can never land on top of each other, which is what keeps the
            // lattice a guarantee of minimum spacing rather than a suggestion.
            let x = lattice[0] + (0.1 + 0.8 * randoms[0]) * placement::TREE_LATTICE;
            let z = lattice[1] + (0.1 + 0.8 * randoms[1]) * placement::TREE_LATTICE;

            // Grid cell (ix + 1, iz + 1): the one-cell apron shifts every
            // lattice cell up by one.
            let gx = ix + 1;
            let gz = iz + 1;
            let slope = grid.slope(gx, gz);
            let neighbourhood = grid.neighbourhood_mean(gx, gz);

            let site = match placement::evaluate_site(
                cache,
                noise,
                x,
                z,
                visibility_center,
                slope,
                neighbourhood,
            ) {
                Ok(site) => site,
                Err(reject) => {
                    rejected[reject as usize] += 1;
                    continue;
                }
            };

            // Acceptance is a roll against the site's density, so clearings
            // are simply the places the roll keeps failing — which is why they
            // have ragged, organic edges instead of the contour a threshold
            // would draw.
            if randoms[2] >= site.density {
                rejected[5] += 1;
                continue;
            }

            let variant = (randoms[3] * TREE_VARIANT_COUNT as f32) as usize;
            let variant = variant.min(TREE_VARIANT_COUNT - 1);
            // Height-relative size spread: a "large" fir is not one size, it is
            // one species, and a stand of identical trunks reads as a fence.
            let scale = 0.78 + 0.44 * randoms[4];
            let trunk_height = heights[site.size][variant] * scale;
            // Seat the foot below the surface by a share of the local relief,
            // so a tree on a slope is not left standing on the downhill edge of
            // its own base. Capped in absolute terms because the grid's slope
            // is a neighbourhood figure and an outlier should not bury a tree.
            let sink = (slope * 1.6).min(0.22 * trunk_height.max(1.0));

            placements.push(TreePlacement {
                x,
                z,
                ground: site.height,
                size: site.size as u8,
                variant: variant as u8,
                scale,
                rotation: randoms[3] * std::f32::consts::TAU,
                variation: randoms[4],
                sink,
            });
        }
    }
    // The tally, as one line per chunk. `passed` is the candidates that
    // cleared every hard test; `roll` is how many of those the density roll
    // then thinned out. A forest that comes out too thin is either a `passed`
    // that is too small (a hard test is too strict) or a `roll` that is too
    // large (the density field is too low), and this line says which.
    let passed: usize = placements.len() + rejected[5];
    log::debug!(
        "TREES: chunk {chunk:?} — {} candidates, {passed} passed, {} planted; \
         rejected shore {} slope {} rock {} snow {} furrow {}, thinned {}",
        cell_count * cell_count,
        placements.len(),
        rejected[0],
        rejected[1],
        rejected[2],
        rejected[3],
        rejected[4],
        rejected[5],
    );
    placements
}

/// Rebuild the chunk work queue around the player, farthest first so that
/// `pop` hands back the nearest.
///
/// Only the chunks the player has just come into range of are added; the ones
/// already scattered cost nothing. Rebuilding the whole list rather than
/// diffing it is deliberate: the list is a few hundred entries of two integers,
/// and a diff would be the more expensive and more fragile of the two.
fn refill_queue(field: &mut TreeField, centre: [i32; 2], cache: &ErosionCache) {
    let span = (placement::TREE_SCATTER_RADIUS / placement::TREE_CHUNK_SIZE).ceil() as i32;
    // Distances are measured from the centre of the chunk the player is in,
    // not from the world origin: the origin is only where the player happens
    // to start, and a queue built around it stops following them the moment
    // they walk. The chunk centre rather than the player's exact position is
    // deliberate — the queue is only rebuilt when they cross a chunk boundary,
    // so the chunk centre is exactly as stale as the queue is, and using it
    // keeps a chunk's membership fixed for as long as it is in the queue.
    let anchor = chunk_centre(centre);
    let mut wanted: Vec<([i32; 2], f32)> = Vec::new();
    for dz in -span..=span {
        for dx in -span..=span {
            let chunk = [centre[0] + dx, centre[1] + dz];
            let position = chunk_centre(chunk);
            let distance =
                ((position[0] - anchor[0]).powi(2) + (position[1] - anchor[1]).powi(2)).sqrt();
            if distance > placement::TREE_SCATTER_RADIUS {
                continue;
            }
            if field.chunks.contains_key(&chunk) || field.deferred.contains_key(&chunk) {
                continue;
            }
            // A chunk whose erosion tiles have not finalised is deferred
            // rather than skipped: skipping would drop it until the player
            // crossed another chunk boundary, which can be a long walk away.
            if !placement::chunk_is_ready(cache, chunk) {
                let frame = field.frame;
                field.deferred.insert(chunk, frame);
                continue;
            }
            wanted.push((chunk, distance));
        }
    }
    wanted.sort_by(|left, right| right.1.total_cmp(&left.1));
    field.queue = wanted.into_iter().map(|(chunk, _)| chunk).collect();
    field.queued_around = Some(centre);
}

/// Drives the chunk stream: evict what is out of range, refill the queue when
/// the player crosses a chunk boundary, and scatter a few chunks per frame.
pub fn tree_stream_system(
    mut field: ResMut<TreeField>,
    cache: Res<ErosionCache>,
    noise: Res<NoiseField>,
    heights: Res<TreeHeights>,
    players: Query<&Player>,
) {
    let Ok(player) = players.single() else {
        return;
    };
    let centre = chunk_of(player.position);
    field.frame += 1;

    // Eviction first: a chunk is dropped once the player is well past it, so
    // the forest never grows without bound as they walk.
    let retain = (placement::TREE_RETAIN_RADIUS / placement::TREE_CHUNK_SIZE).ceil() as i32;
    let mut evicted = false;
    field.chunks.retain(|chunk, _| {
        let keep = (chunk[0] - centre[0]).abs() <= retain && (chunk[1] - centre[1]).abs() <= retain;
        evicted |= !keep;
        keep
    });
    field.deferred.retain(|chunk, _| {
        (chunk[0] - centre[0]).abs() <= retain && (chunk[1] - centre[1]).abs() <= retain
    });

    // Deferred chunks are re-offered once their tiles have had time to
    // finalise. Retrying every frame would re-run the readiness scan for every
    // waiting chunk on a schedule nothing else keeps.
    let ready: Vec<[i32; 2]> = field
        .deferred
        .iter()
        .filter(|(chunk, since)| {
            field.frame - **since >= TREE_CHUNK_RETRY_FRAMES
                && placement::chunk_is_ready(&cache, **chunk)
        })
        .map(|(chunk, _)| *chunk)
        .collect();
    for chunk in ready {
        field.deferred.remove(&chunk);
        field.queue.push(chunk);
    }

    if field.queued_around != Some(centre) {
        refill_queue(&mut field, centre, &cache);
    }

    let mut scattered = false;
    let started = std::time::Instant::now();
    for _ in 0..TREE_CHUNKS_PER_FRAME {
        let Some(chunk) = field.queue.pop() else {
            break;
        };
        if field.chunks.contains_key(&chunk) {
            continue;
        }
        if !placement::chunk_is_ready(&cache, chunk) {
            let frame = field.frame;
            field.deferred.insert(chunk, frame);
            continue;
        }
        let placements = scatter_chunk(&cache, &noise, chunk, &heights.heights);
        field.chunks.insert(chunk, placements);
        scattered = true;
    }
    let elapsed = started.elapsed().as_secs_f32();

    // One-shot report of the initial fill. Only the frames that actually
    // scattered something are counted, so a frame spent waiting on erosion
    // tiles does not dilute the per-frame cost the number is meant to expose.
    if scattered {
        field.fill_seconds += elapsed;
        field.fill_frames += 1;
    }
    // The fill is over when neither list has work left: an empty `queue` alone
    // is also true on the first frames, before `refill_queue` has run, and it
    // is true every time the scatter briefly catches up between erosion tiles
    // finalising — either would report "0 trees" and call it done. A chunk in
    // `deferred` is a chunk this forest still owes, so its emptying is half of
    // what "fully filled" means.
    if !field.fill_reported
        && field.fill_frames > 0
        && field.queue.is_empty()
        && field.deferred.is_empty()
    {
        field.fill_reported = true;
        let trees = field.tree_count();
        let per_frame = if field.fill_frames == 0 {
            0.0
        } else {
            field.fill_seconds * 1000.0 / field.fill_frames as f32
        };
        info!(
            "TREES: scattered {trees} trees over {} frames ({:.0} ms total, {per_frame:.1} ms per scattering frame)",
            field.fill_frames, field.fill_seconds * 1000.0
        );
    }

    // `republish` rebuilds the trunk index as well, so it must also run when a
    // chunk leaves: an evicted tree must stop blocking the player even though
    // the eviction itself adds nothing.
    if scattered || evicted {
        field.republish(heights.heights, player.position);
    }
}

// ---------------------------------------------------------------------------
// Collision
// ---------------------------------------------------------------------------

/// Pushes the player out of any trunk they have ended up inside.
///
/// Runs after `update_player_system`, so it sees the position the player
/// actually moved to. Pushing out rather than rejecting the step is what lets
/// them slide along a trunk instead of stopping against it, and it also covers
/// the case a rejection cannot: a chunk streaming in around a player who is
/// already standing where a tree now is.
pub fn tree_collision_system(mut players: Query<&mut Player>, field: Res<TreeField>) {
    let Ok(mut player) = players.single_mut() else {
        return;
    };
    if player.flying {
        return;
    }
    // A trunk taller than the player's head still only blocks at trunk height;
    // a player standing on a canopy is above the collision this models.
    if let Some(push) = field.trunks.push_out(player.position.x, player.position.z) {
        player.position.x += push[0];
        player.position.z += push[1];
    }
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

/// The extracted pack's directory, relative to the asset root.
pub const TREE_ASSET_DIRECTORY: &str = "models/fir";

/// Base colour and normal maps are authored at 4K for the branch cards, which
/// is four times the texel density a fir's foliage can resolve at any distance
/// the LOD-0 mesh is drawn from. Halving them once at load takes the pack's
/// texture budget from 39 MB to 11 MB and is invisible; the bark stays at its
/// native 1024x2048, where it is already the limiting factor.
const TREE_TEXTURE_MAXIMUM: u32 = 2048;

fn asset_path(relative: &str) -> PathBuf {
    Path::new("assets").join(relative)
}

fn decode_image(bytes: &[u8]) -> Result<(Vec<u8>, u32, u32), String> {
    let decoded = image::load_from_memory(bytes)
        .map_err(|error| format!("decode: {error}"))?
        .to_rgba8();
    let (width, height) = decoded.dimensions();
    Ok((decoded.into_raw(), width, height))
}

/// Halve an RGBA8 image with a 2x2 box filter until it fits `maximum`.
///
/// Box averaging is the right filter here rather than the terrain's
/// wrap-aware one: a tree texture is not a tiling tile, so there is no seam to
/// preserve and averaging the real neighbours is strictly better than
/// averaging across the wrap.
fn halve_rgba(src: &[u8], width: u32, height: u32) -> (Vec<u8>, u32, u32) {
    let (next_width, next_height) = ((width / 2).max(1), (height / 2).max(1));
    let mut dst = vec![0u8; (next_width * next_height * 4) as usize];
    for y in 0..next_height {
        for x in 0..next_width {
            for channel in 0..4usize {
                let mut total = 0u32;
                for dy in 0..2u32 {
                    for dx in 0..2u32 {
                        let sx = (x * 2 + dx).min(width - 1);
                        let sy = (y * 2 + dy).min(height - 1);
                        total += src[((sy * width + sx) * 4) as usize + channel] as u32;
                    }
                }
                dst[((y * next_width + x) * 4) as usize + channel] = ((total + 2) / 4) as u8;
            }
        }
    }
    (dst, next_width, next_height)
}

/// Build an image's mip chain by repeated halving, down to 1x1.
fn build_mip_chain(mut bytes: Vec<u8>, mut width: u32, mut height: u32, srgb: bool) -> TreeImage {
    bytes = {
        let mut current = bytes;
        let (mut w, mut h) = (width, height);
        while w > TREE_TEXTURE_MAXIMUM || h > TREE_TEXTURE_MAXIMUM {
            let (next, next_w, next_h) = halve_rgba(&current, w, h);
            current = next;
            w = next_w;
            h = next_h;
        }
        width = w;
        height = h;
        current
    };

    let mut levels = vec![bytes];
    let (mut w, mut h) = (width, height);
    while w > 1 || h > 1 {
        let previous = levels.last().unwrap();
        let (next, next_w, next_h) = halve_rgba(previous, w, h);
        levels.push(next);
        w = next_w;
        h = next_h;
    }
    TreeImage {
        width,
        height,
        levels,
        srgb,
    }
}

/// Cache key for an image. External images are keyed by their path, so the six
/// shared textures are decoded once for all 36 files; embedded ones are keyed
/// by a content hash, so a LOD-3 billboard's own textures are decoded once
/// each and never collide with another tree's.
fn image_key(name: &str, image: &GlbImage) -> String {
    match image {
        GlbImage::External(path) => path.display().to_string(),
        GlbImage::Embedded(bytes) => {
            // FNV-1a over the bytes: cheap, and a collision would only ever
            // cost a re-decode, since the key is used as a cache and not as
            // an identity anything else depends on.
            let mut hash = 0xcbf2_9ce4_8422_2325u64;
            for byte in bytes {
                hash ^= *byte as u64;
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
            format!("{name}#{hash:016x}")
        }
    }
}

fn load_tree_image(
    name: &str,
    image: &GlbImage,
    srgb: bool,
    cache: &mut HashMap<String, Arc<TreeImage>>,
) -> Result<Arc<TreeImage>, String> {
    let key = image_key(name, image);
    if let Some(found) = cache.get(&key) {
        return Ok(found.clone());
    }
    let bytes = match image {
        GlbImage::External(path) => {
            std::fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?
        }
        GlbImage::Embedded(bytes) => bytes.clone(),
    };
    let (rgba, width, height) = decode_image(&bytes)?;
    let loaded = Arc::new(build_mip_chain(rgba, width, height, srgb));
    cache.insert(key, loaded.clone());
    Ok(loaded)
}

/// Check the handful of things about the pack that the renderer relies on but
/// cannot enforce, and say so loudly if a re-export breaks one.
///
/// Every one of these is true of the 36 files as extracted, and every one of
/// them would fail *quietly* — floating trees, black foliage, a trunk with a
/// metal sheen — if a future export changed it. A log line is the cheapest way
/// to make that a bug report instead of a mystery.
fn check_pack_assumptions(file: &GlbFile) {
    // The vertex shader plants `foot.y = ground - sink`, which assumes each
    // mesh's origin sits at the trunk's base. A pack whose geometry hung below
    // its origin would bury every tree by that amount.
    if file.bounds_min[1] < -0.1 {
        log::warn!(
            "Tree {}: geometry reaches {:.3} m below the origin; the trunk foot is \
             assumed to be at y = 0 (see tree-vs.wgsl)",
            file.name,
            file.bounds_min[1]
        );
    }
    for material in &file.materials {
        if !material.double_sided {
            log::warn!(
                "Tree {}: material {} is single-sided but the tree pipeline draws with \
                 cull_mode: None, so its back faces will be shaded",
                file.name,
                material.name
            );
        }
        if material.metallic > 0.2 {
            log::warn!(
                "Tree {}: material {} is {:.2} metallic, and the tree material uniform \
                 carries no metallic term — it will render as fully dielectric",
                file.name,
                material.name,
                material.metallic
            );
        }
    }
}

/// Load one mesh: merge every primitive into one vertex buffer, offsetting its
/// indices, and record the material range each primitive occupies.
fn load_tree_mesh(
    path: &Path,
    cache: &mut HashMap<String, Arc<TreeImage>>,
) -> Result<TreeMeshData, String> {
    let file = GlbFile::load(path)?;
    check_pack_assumptions(&file);
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let mut ranges = Vec::new();
    for primitive in &file.primitives {
        let material = file.materials.get(primitive.material);
        let base = vertices.len() as u32;
        let first_index = indices.len() as u32;
        vertices.extend_from_slice(&primitive.vertices);
        indices.extend(primitive.indices.iter().map(|index| index + base));

        let (base_colour, normal) = match material {
            Some(material) => (
                material
                    .base_colour
                    .as_ref()
                    .map(|image| load_tree_image(&file.name, image, true, cache))
                    .transpose()?,
                material
                    .normal
                    .as_ref()
                    .map(|image| load_tree_image(&file.name, image, false, cache))
                    .transpose()?,
            ),
            None => (None, None),
        };
        ranges.push(TreeMaterialRange {
            first_index,
            index_count: indices.len() as u32 - first_index,
            // `alphaMode` is the pack's own statement of which primitive is a
            // foliage card: bark is OPAQUE, branches and every LOD-3 billboard
            // are BLEND. Reading it rather than matching on the material name
            // means a re-export that renames a material cannot silently turn
            // every branch card opaque.
            is_branch: material.is_some_and(|material| material.alpha_cutout),
            roughness: material.map(|material| material.roughness).unwrap_or(1.0),
            alpha_cutoff: match material {
                Some(material) if material.alpha_cutout => material.alpha_cutoff.max(0.5),
                _ => 0.0,
            },
            specular_factor: material.map(|material| material.specular_factor).unwrap_or(1.0),
            base_colour,
            normal,
        });
    }
    Ok(TreeMeshData {
        height: file.height(),
        vertices,
        indices,
        ranges,
    })
}

/// Parse the whole 36-file pack. Startup-only and heavy: it decodes the six
/// shared textures plus the eighteen embedded billboard ones, builds every mip
/// chain, and merges 36 meshes.
fn load_tree_assets(mut commands: Commands) {
    let mut meshes: Vec<Option<TreeMeshData>> = (0..TREE_GROUP_COUNT).map(|_| None).collect();
    let mut cache = HashMap::new();
    for size in 0..TREE_SIZE_COUNT {
        for variant in 1..=TREE_VARIANT_COUNT {
            for lod in 0..TREE_LOD_COUNT {
                let relative = format!(
                    "{TREE_ASSET_DIRECTORY}/fir-{}-{variant}-lod-{lod}.glb",
                    TREE_SIZE_NAMES[size]
                );
                let path = asset_path(&relative);
                match load_tree_mesh(&path, &mut cache) {
                    Ok(mesh) => meshes[tree_group(size, variant - 1, lod)] = Some(mesh),
                    Err(error) => log::warn!("Tree mesh missing: {} ({error})", path.display()),
                }
            }
        }
    }
    // A missing file leaves an empty mesh rather than removing the group: the
    // render node indexes meshes by `tree_group`, and a shorter vector would
    // silently shift every later group onto the wrong geometry.
    let meshes: Vec<TreeMeshData> = meshes
        .into_iter()
        .map(|mesh| {
            mesh.unwrap_or(TreeMeshData {
                vertices: Vec::new(),
                indices: Vec::new(),
                ranges: Vec::new(),
                height: 0.0,
            })
        })
        .collect();
    let total: usize = meshes.iter().map(|mesh| mesh.vertices.len()).sum();
    log::info!(
        "TREES: loaded {TREE_GROUP_COUNT} meshes ({total} vertices), {} textures",
        cache.len()
    );
    // The render world needs the per-(size, variant) heights before it can size
    // a LOD band, and the scatter needs them before it can scale a tree, so
    // they are copied out here rather than re-derived from 36 GLBs twice.
    let heights: [[f32; TREE_VARIANT_COUNT]; TREE_SIZE_COUNT] =
        std::array::from_fn(|size| {
            std::array::from_fn(|variant| {
                // LOD 0 is the only mesh whose height is the authored one; the
                // decimated LODs drift by a few centimetres as the decimator
                // clips the crown, and a species' size should not depend on
                // which LOD happened to be loaded.
                let mesh = &meshes[tree_group(size, variant, 0)];
                if mesh.height > 0.0 {
                    mesh.height
                } else {
                    // Every file for this species failed to load; a nominal
                    // height keeps the LOD bands finite.
                    [4.0, 6.5, 9.2][size]
                }
            })
        });
    commands.insert_resource(TreeAssetData { meshes });
    commands.insert_resource(TreeHeights { heights });
}

/// LOD-0 trunk height per (size, variant). See [`TreeScatter::heights`].
#[derive(Resource)]
pub struct TreeHeights {
    pub heights: [[f32; TREE_VARIANT_COUNT]; TREE_SIZE_COUNT],
}

impl Default for TreeHeights {
    fn default() -> Self {
        Self {
            heights: [[5.0; TREE_VARIANT_COUNT]; TREE_SIZE_COUNT],
        }
    }
}

/// The main-world half of the forest. The render half is
/// `render::tree_node`, registered by `ForestRenderPlugin::build` alongside the
/// other passes; the two are separate plugins because they are separate apps
/// with separate schedules, exactly like the water and terrain nodes.
///
/// The streaming and collision systems are *not* registered here: both need to
/// sit at a specific point in `main.rs`'s existing Update chain — the stream
/// reads the erosion cache that chain's stream system fills, and collision has
/// to see the position the player actually moved to. Adding them from here
/// would leave them unordered with respect to both.
pub struct TreePlugin;

impl Plugin for TreePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<TreeField>()
            .init_resource::<TreeHeights>()
            .add_systems(bevy::app::PreStartup, load_tree_assets);
    }
}
