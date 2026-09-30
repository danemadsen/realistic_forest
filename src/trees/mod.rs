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

/// One scattered tree as the vertex shader sees it. 40 bytes, mirrored by
/// `TreeInstance` in assets/shaders/tree-vs.wgsl, which declares the same six
/// fields at `@location(4..9)`.
///
/// 40 rather than 28 because the orientation is a quaternion: a yaw alone
/// cannot express the lean a crooked tree needs, and three angles would cost
/// the vertex shader six trigonometric calls per vertex to rebuild. See
/// [`placement::tree_rotation`]. At 70,000 instances and an upload only when
/// the buckets change, the twelve extra bytes are 840 KB of buffer that is
/// written a few times a minute.
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
    /// World orientation as a unit quaternion, `[x, y, z, w]`. Full yaw, plus
    /// a lean of at most `TREE_MAX_LEAN`.
    pub rotation: [f32; 4],
    /// 0..1 stable per-tree random, driving the fragment stage's tint.
    pub variation: f32,
    /// Metres the trunk foot is pushed below ground, so a tree on a slope is
    /// seated into it rather than balanced on its downhill edge.
    pub sink: f32,
}

const _: () = assert!(std::mem::size_of::<TreeInstance>() == 40);

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
    /// World orientation as a unit quaternion, `[x, y, z, w]`; see
    /// [`placement::tree_rotation`].
    pub rotation: [f32; 4],
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
    /// The glTF material's `doubleSided`, carried through so the tree pass can
    /// pick a cull mode per material rather than one for the whole pack.
    ///
    /// This is not a cosmetic flag on this pack. The LOD-0/1/2 materials are
    /// `doubleSided` — a fir's branch cards are flat planes and you are meant to
    /// see them from behind — but every LOD-3 billboard material is
    /// single-sided, and its geometry says why: the cross is authored as four
    /// separate cards, one per axis direction, each with its own window of the
    /// atlas (front view at u 0.67-0.99, back view at u 0.34-0.66, and so on).
    /// Exactly one card of each of the two crossed planes faces any given
    /// camera, so culling the back faces removes the other one and its
    /// co-located pixels with it. Drawing them instead doubles the fragment
    /// work of the largest band in the forest for an identical image.
    pub double_sided: bool,
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
/// One chunk's scattered trees, and which tier produced them.
///
/// The tier is stored beside the trees rather than in a set of its own so the
/// two cannot disagree: the only thing the streaming logic needs to know about
/// a chunk is whether the content it is holding is the decimated far version
/// or the full near one, and that question has exactly one answer per chunk.
struct ScatteredChunk {
    placements: Vec<TreePlacement>,
    /// Scattered on the far tier's decimated lattice; see
    /// [`placement::TREE_FAR_STRIDE`].
    coarse: bool,
}

#[derive(Resource, Default)]
pub struct TreeField {
    /// One placement list per scattered chunk. Kept per chunk so eviction is
    /// a single `remove` rather than a scan of the flat list.
    chunks: HashMap<[i32; 2], ScatteredChunk>,
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
        let mut placements =
            Vec::with_capacity(self.chunks.values().map(|chunk| chunk.placements.len()).sum());
        let mut trunks = TrunkIndex::default();
        for chunk in self.chunks.values() {
            for placement in &chunk.placements {
                let radius = TREE_TRUNK_RADIUS[placement.size as usize % TREE_SIZE_COUNT]
                    * placement.scale;
                trunks.insert(placement.x, placement.z, radius);
                placements.push(*placement);
            }
        }
        self.trunks = trunks;

        // Everything below is the debug line, and none of it runs unless that
        // line is going to be emitted. This matters more than the usual
        // "diagnostics should be cheap": `republish` is called once per
        // scattered chunk, up to `TREE_CHUNKS_PER_FRAME` times a frame, and
        // during the fill that is every frame. The nearest-trees list used to
        // be a full sort of every placement in the forest to name four of them
        // — at 100,000 trees that is a few hundred thousand comparisons of two
        // distances each, several times a frame, which is the hitch the player
        // feels as the forest appears. A bounded selection is one pass and four
        // steps of insertion instead; skipping it entirely at info level is
        // nothing at all.
        if log::log_enabled!(log::Level::Debug) {
            // The nearest few trees, for aiming a `--camera` capture at one
            // without having to guess coordinates — so "near" means near the
            // player, not near the world origin, which is an arbitrary point
            // the player may never visit. Debug level: it changes on every
            // chunk crossing and is not a number anyone watches in normal play.
            //
            // Read from `placements`, the list just built, not from
            // `self.scatter.placements`, which is still the previous
            // generation's and is empty on the first call.
            let mut nearest: Vec<(f32, &TreePlacement)> =
                Vec::with_capacity(TREE_NEAREST_REPORTED);
            for placement in &placements {
                let distance =
                    (placement.x - focus.x).powi(2) + (placement.z - focus.z).powi(2);
                // Sorted ascending, so the last entry is the worst of the four
                // kept and one comparison rejects most of the forest.
                if nearest.len() == TREE_NEAREST_REPORTED
                    && distance >= nearest[TREE_NEAREST_REPORTED - 1].0
                {
                    continue;
                }
                let index = nearest.partition_point(|entry| entry.0 <= distance);
                nearest.insert(index, (distance, placement));
                nearest.truncate(TREE_NEAREST_REPORTED);
            }
            // How the stand is spread over distance from the player, in eight
            // bands. A capture that reads as parkland is usually not short of
            // trees overall but short of them in one particular range — a hole
            // in this histogram is a hole the camera sees as a bare band, and a
            // hole here is a placement bug rather than a density one, which the
            // pixel measurements cannot tell apart. Band edges are powers of
            // two from 25 m, which is the order a 1.5 m sapling stops
            // resolving; the last three also straddle the near/far tier
            // boundary at `TREE_SCATTER_RADIUS`, where a hole would mean the
            // far tier is missing rather than thin.
            let mut bands = [0usize; 8];
            for placement in &placements {
                let distance = ((placement.x - focus.x).powi(2)
                    + (placement.z - focus.z).powi(2))
                .sqrt();
                let band = if distance < 25.0 {
                    0
                } else if distance < 50.0 {
                    1
                } else if distance < 100.0 {
                    2
                } else if distance < 200.0 {
                    3
                } else if distance < 400.0 {
                    4
                } else if distance < 900.0 {
                    5
                } else if distance < 1300.0 {
                    6
                } else {
                    7
                };
                bands[band] += 1;
            }
            let sample: Vec<String> = nearest
                .iter()
                .map(|(_, placement)| {
                    format!(
                        "({:.1}, {:.1}) h={:.1} size={} ground={:.1}",
                        placement.x,
                        placement.z,
                        heights[placement.size as usize % TREE_SIZE_COUNT]
                            [placement.variant as usize % TREE_VARIANT_COUNT]
                            * placement.scale,
                        placement.size,
                        placement.ground,
                    )
                })
                .collect();
            log::debug!(
                "TREES: {} chunks scattered, {} deferred, {} queued; \
                 within 25/50/100/200/400/900/1300 m of the player: \
                 {}/{}/{}/{}/{}/{}/{}; nearest: {}",
                self.chunks.len(),
                self.deferred.len(),
                self.queue.len(),
                bands[0],
                bands[1],
                bands[2],
                bands[3],
                bands[4],
                bands[5],
                bands[6],
                sample.join(", ")
            );
        }

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

/// How many trees the debug line names as the nearest. See `republish`.
const TREE_NEAREST_REPORTED: usize = 4;

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
///
/// `cache` is `None` for a far chunk, which is judged against the uneroded
/// landform and therefore needs no erosion tiles to have finalised. `step` is
/// the tier's lattice spacing; see [`placement::build_height_grid`].
fn scatter_chunk(
    cache: Option<&ErosionCache>,
    noise: &NoiseField,
    chunk: [i32; 2],
    step: f32,
    heights: &[[f32; TREE_VARIANT_COUNT]; TREE_SIZE_COUNT],
) -> Vec<TreePlacement> {
    // The chunk's own centre, never the player's: see `evaluate_site`.
    let visibility_center = chunk_centre(chunk);
    let grid = placement::build_height_grid(cache, noise, chunk, step, visibility_center);
    let cell_count = grid.side;

    let mut placements = Vec::new();
    // Rejection tally, reported once per chunk at debug level. `candidates` is
    // the denominator every other number here is read against.
    let mut rejected = [0usize; 6];
    // Ten bins of 0.1 across the vigour range, counted over the candidates that
    // cleared the hard tests. This is the distribution `tree_size`'s boundaries
    // are read against; see `TREE_SMALL_MAX_VIGOUR` for why it is measured
    // rather than reasoned about.
    let mut vigour_bins = [0usize; 10];
    for iz in 0..cell_count {
        for ix in 0..cell_count {
            let lattice = [
                chunk[0] as f32 * placement::TREE_CHUNK_SIZE + ix as f32 * step,
                chunk[1] as f32 * placement::TREE_CHUNK_SIZE + iz as f32 * step,
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

            vigour_bins[((site.vigour.clamp(0.0, 0.999) * 10.0) as usize).min(9)] += 1;

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
            // Lean before yaw, so the tilt is in the tree's own frame; see
            // `placement::tree_rotation`.
            let rotation = placement::tree_rotation(randoms[7], randoms[5], randoms[6]);
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
                rotation,
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
    // The size mix, small/medium/large. A stand that reads as parkland rather
    // than forest is usually a size problem before it is a count problem: the
    // three classes are 1.5 m, 4.4 m and 9.3 m tall, so one class's share of
    // the plantings moves the canopy area far more than the total does, and
    // `tree_size`'s thresholds are read against a vigour field whose own
    // distribution is not obvious from its definition. This line shows it.
    let mut sizes = [0usize; TREE_SIZE_COUNT];
    for placement in &placements {
        sizes[placement.size as usize] += 1;
    }
    log::debug!(
        "TREES: chunk {chunk:?} — {} candidates, {passed} passed, {} planted \
         (small {} medium {} large {}); \
         rejected shore {} slope {} rock {} snow {} furrow {}, thinned {}; \
         vigour per 0.1 {vigour_bins:?}",
        cell_count * cell_count,
        placements.len(),
        sizes[0],
        sizes[1],
        sizes[2],
        rejected[0],
        rejected[1],
        rejected[2],
        rejected[3],
        rejected[4],
        rejected[5],
    );
    placements
}

/// Which lattice a chunk at this distance should be scattered on.
///
/// The two radii overlap on purpose. A chunk is *promoted* — re-scattered on
/// the full lattice — as soon as it comes inside [`TREE_SCATTER_RADIUS`], but
/// it is not *demoted* until it is past [`TREE_RETAIN_RADIUS`], which is the
/// same 150 m of hysteresis that keeps a chunk from being dropped and rebuilt
/// as the player walks along the boundary. Without it, a player standing on the
/// edge would have a ring of chunks re-scattered every time they crossed a
/// chunk line in either direction.
fn wants_coarse(distance: f32, currently_coarse: bool) -> bool {
    if currently_coarse {
        distance > placement::TREE_SCATTER_RADIUS
    } else {
        distance > placement::TREE_RETAIN_RADIUS
    }
}

/// The lattice spacing a tier scatters at.
fn lattice_step(coarse: bool) -> f32 {
    if coarse {
        placement::TREE_LATTICE * placement::TREE_FAR_STRIDE as f32
    } else {
        placement::TREE_LATTICE
    }
}

/// Rebuild the chunk work queue around the player, farthest first so that
/// `pop` hands back the nearest.
///
/// Only the chunks the player has just come into range of are added; the ones
/// already scattered cost nothing. Rebuilding the whole list rather than
/// diffing it is deliberate: the list is a few hundred entries of two integers,
/// and a diff would be the more expensive and more fragile of the two.
///
/// A chunk already in `chunks` is not automatically done with: one holding far
/// content that has come inside the near radius is re-queued so it can be
/// re-scattered at full density. That is the one case where a chunk is
/// scattered twice, and it is what keeps the far tier from leaving a sparse
/// ring in the forest ahead of a player who walks toward it.
fn refill_queue(field: &mut TreeField, centre: [i32; 2], cache: &ErosionCache) {
    let span = (placement::TREE_FAR_RADIUS / placement::TREE_CHUNK_SIZE).ceil() as i32;
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
            if distance > placement::TREE_FAR_RADIUS {
                continue;
            }
            if field.deferred.contains_key(&chunk) {
                continue;
            }
            if let Some(existing) = field.chunks.get(&chunk) {
                // Present and already at the density this distance wants — the
                // common case, and it costs nothing.
                if existing.coarse == wants_coarse(distance, existing.coarse) {
                    continue;
                }
            } else {
                // A chunk whose erosion tiles have not finalised is deferred
                // rather than skipped: skipping would drop it until the player
                // crossed another chunk boundary, which can be a long walk
                // away. Only the near tier reads them, so only it waits.
                let coarse = distance > placement::TREE_SCATTER_RADIUS;
                if !coarse && !placement::chunk_is_ready(cache, chunk) {
                    let frame = field.frame;
                    field.deferred.insert(chunk, frame);
                    continue;
                }
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
    automation: Res<crate::automation::AutomationSettings>,
    mut field: ResMut<TreeField>,
    cache: Res<ErosionCache>,
    noise: Res<NoiseField>,
    heights: Res<TreeHeights>,
    players: Query<&Player>,
) {
    // `--no-trees` prices the forest: the pass still runs, but with no
    // instances it draws nothing, so the frame time it leaves behind is the
    // rest of the pipeline's.
    if automation.no_trees {
        return;
    }
    let Ok(player) = players.single() else {
        return;
    };
    let centre = chunk_of(player.position);
    field.frame += 1;

    // Eviction first: a chunk is dropped once the player is well past it, so
    // the forest never grows without bound as they walk. The box is the far
    // tier's, and it is a box rather than a disc for the same reason it always
    // was: the corners cost a little extra memory and the check costs one
    // comparison. A near chunk inside a far-sized box is kept, which is the
    // point — the alternative is a chunk that was dense when the player walked
    // through it becoming sparse the moment they turned around.
    let retain = (placement::TREE_FAR_RETAIN_RADIUS / placement::TREE_CHUNK_SIZE).ceil() as i32;
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
    let anchor = chunk_centre(centre);
    for _ in 0..TREE_CHUNKS_PER_FRAME {
        let Some(chunk) = field.queue.pop() else {
            break;
        };
        let position = chunk_centre(chunk);
        let distance =
            ((position[0] - anchor[0]).powi(2) + (position[1] - anchor[1]).powi(2)).sqrt();
        // A chunk already holding the density this distance wants is done. The
        // queue can hold one that is not — a far chunk the player has since
        // walked up to — and that one is re-scattered, which is the whole
        // reason the check is against the tier and not against presence.
        let coarse = wants_coarse(distance, true);
        if let Some(existing) = field.chunks.get(&chunk) {
            if existing.coarse == coarse {
                continue;
            }
        }
        if !coarse && !placement::chunk_is_ready(&cache, chunk) {
            let frame = field.frame;
            field.deferred.insert(chunk, frame);
            continue;
        }
        let source = if coarse { None } else { Some(&*cache) };
        let placements =
            scatter_chunk(source, &noise, chunk, lattice_step(coarse), &heights.heights);
        field.chunks.insert(chunk, ScatteredChunk { placements, coarse });
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

/// Halve an RGBA8 image with a 2x2 box filter until it fits `maximum`, keeping
/// the alpha's *maximum* rather than its average.
///
/// Box averaging is the right filter for colour here rather than the terrain's
/// wrap-aware one: a tree texture is not a tiling tile, so there is no seam to
/// preserve and averaging the real neighbours is strictly better than
/// averaging across the wrap.
///
/// Alpha is the exception, and it is the whole reason this function is not a
/// plain box filter. A foliage texture's alpha is *coverage*, and its mean is
/// low — the billboard textures run 0.18 and 0.24 opaque. Averaging preserves
/// that mean at every mip level, so once minification reaches the mips the
/// sampled alpha converges on 0.2 and every fragment fails the 0.5 cutout.
/// Past `TREE_LOD_DISTANCE[2]` every tree is a billboard, so that is every tree
/// further away than 24 m: measured, a fully-filled 68,480-tree forest and a
/// third-filled 23,545-tree one rendered to images differing in 293 pixels of
/// 1,440,000, because the 45,000 extra trees were all far enough to be lost
/// this way. Taking the maximum instead keeps
/// the silhouette at full extent, which is the standard dilation an alpha-tested
/// foliage mip chain needs — the tree stays a tree, and at the coarsest levels
/// it becomes a solid silhouette, which is what a tree 4 px wide should be.
fn halve_rgba(src: &[u8], width: u32, height: u32) -> (Vec<u8>, u32, u32) {
    let (next_width, next_height) = ((width / 2).max(1), (height / 2).max(1));
    let mut dst = vec![0u8; (next_width * next_height * 4) as usize];
    for y in 0..next_height {
        for x in 0..next_width {
            for channel in 0..4usize {
                let mut total = 0u32;
                let mut largest = 0u32;
                for dy in 0..2u32 {
                    for dx in 0..2u32 {
                        let sx = (x * 2 + dx).min(width - 1);
                        let sy = (y * 2 + dy).min(height - 1);
                        let texel = src[((sy * width + sx) * 4) as usize + channel] as u32;
                        total += texel;
                        largest = largest.max(texel);
                    }
                }
                let value = if channel == 3 {
                    largest
                } else {
                    (total + 2) / 4
                };
                dst[((y * next_width + x) * 4) as usize + channel] = value as u8;
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
        // A single-sided material is no longer a warning: the tree pass reads
        // `doubleSided` and builds a back-face-culling pipeline for it. See
        // `TreeMaterialRange::double_sided`.
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
            // A primitive with no material has nothing to say about sidedness,
            // and drawing both faces is the safe default: it can only cost
            // fragments, where the other way can lose geometry.
            double_sided: material.is_none_or(|material| material.double_sided),
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
