//! Where a tree may stand: the CPU half of the scatter.
//!
//! The renderer already has one authoritative answer to "what is the ground
//! here", and it is `terrainHeight` in terrain-height-functions.wgslinc, whose
//! CPU mirror is `crate::erosion::sample_eroded_height`. This module asks that
//! question and then asks a second one the GPU cannot answer: *what is the
//! ground made of*.
//!
//! # What is ported exactly
//!
//! `groundNoise` and its integer hash are pure functions of world position
//! (terrain-fs.wgsl:398-425), so they port to the CPU bit for bit. Everything
//! built only from `groundNoise` therefore ports exactly too: the domain warp,
//! `groundRegion`, `groundGrowth`, the soil fields, `rockRegion` and the four
//! snow-drift fields. So does `slope` (finite differences of the same height
//! field the shader differentiates), the shore-sand term `1 - fGrass`, and the
//! eroded height itself.
//!
//! # What is not, and how that is handled
//!
//! `incision`, `deposition`, `channel`, `hollow`, `transport`, `dischargeAmount`
//! and `ridge` come out of the GPU flow atlas and the erosion simulation
//! (terrain-fs.wgsl:744-783). They exist nowhere on the CPU — a scatter cannot
//! read them and must not pretend to.
//!
//! They are bounded, though, and the bounds are what make a *safe* test
//! possible rather than a guess. Every one of them lies in [0, 1] and enters
//! the material blend through a signed, bounded term, so the CPU can compute
//! the value the shader's expression takes when the unknown terms are at their
//! extreme and reject on that. A rejection is always allowed to be
//! over-cautious — a tree fewer at the edge of a snowfield costs nothing,
//! whereas one tree standing in snow is exactly the artefact the scatter is
//! supposed to prevent. Two of the terms resist that treatment because their
//! only bound is "up to 1 everywhere" (`sedimentSand`, `scouredGravel`); both
//! are features *inside* eroded channels, and the furrow test below rejects the
//! whole channel, so they are covered by a different mechanism rather than by
//! a margin.
//!
//! The structure is deliberately shaped so the day a GPU suitability mask
//! exists (a render target the erosion node fills alongside the height atlas),
//! [`evaluate_site`] can read it and the CPU approximations can be deleted one
//! at a time without touching the scatter or the render node.

use crate::constants::*;
use crate::erosion::{ErosionCache, ErosionTileState, sample_eroded_height};
use crate::noise::{NoiseField, sample_periodic_noise};

// ---------------------------------------------------------------------------
// Scatter tuning
// ---------------------------------------------------------------------------

/// Trees exist within this radius of the player, in world space and
/// independent of which way the camera faces.
///
/// The radius is set by `erosionVisibility`, not by taste — and it is now set
/// so that the visibility is *exactly* 1 everywhere inside it.
/// `EROSION_VISIBILITY_FULL_RADIUS` is 1100 m; the scatter judges every chunk
/// against the fully eroded ground (see [`evaluate_site`]), so inside that
/// radius the scatter and the renderer are reading the same surface and a
/// tree's `ground` is the height the renderer will draw. Past it the renderer
/// starts fading the eroded delta out and the two would disagree, which is
/// what the earlier 1200 m figure traded away 100 m of.
///
/// The second reason for the figure is cost. A closed canopy is one tree per
/// 35 m², so the area of this disc is very nearly the whole vertex budget of
/// the tree pass: 900 m is 2.5 km² and about 70,000 trees, against 4.5 km² and
/// 127,000 at 1200 m. The trees that the smaller radius gives up are ten
/// pixels tall and beyond.
pub const TREE_SCATTER_RADIUS: f32 = 900.0;

/// Chunks are retained this much further out than they are generated, so a
/// chunk is already scattered by the time the player can see it. The 150 m of
/// slack is far beyond the distance at which a LOD-3 billboard is more than a
/// couple of pixels, which is what makes the drop invisible when it comes.
pub const TREE_RETAIN_RADIUS: f32 = 1050.0;

/// How far the coarse far tier reaches, in metres.
///
/// [`TREE_SCATTER_RADIUS`] alone leaves the horizon bare: it is 900 m of
/// forest and then nothing, and the terrain the renderer draws runs out to
/// 1,600 m, so everything between the two is a treeless band the width of the
/// whole near forest. This is the radius that closes it — a little past
/// `EROSION_VISIBILITY_ZERO_RADIUS` so the last trees stand where the erosion
/// fade has already gone to zero.
pub const TREE_FAR_RADIUS: f32 = 1800.0;

/// Chunks holding far content are retained to here, for the same reason the
/// near tier has its own slack: a player who turns around should see the
/// forest they walked through, not the edge of a disc that follows them.
pub const TREE_FAR_RETAIN_RADIUS: f32 = 2000.0;

/// How many lattice cells the far tier skips per tree, per axis.
///
/// The near lattice's 2.8 m is a *sampling* rate — it is what lets the density
/// roll draw a ragged tree line rather than a contour of the lattice — and a
/// sample rate is only worth paying for where the samples resolve. At 900 m a
/// tree is ten pixels tall and one tree per 2.8 m is a solid mass of them; the
/// far tier is not trying to reproduce that, it is trying to put a treeline on
/// the horizon.
///
/// Four puts far candidates on an 11.2 m lattice, which is a sixteenth of the
/// candidates and a sixteenth of the `evaluate_site` calls behind them. The
/// sites it keeps are not new points: they are the `(4i, 4j)` subset of the
/// near lattice, so a far tree and the near tree that would have stood there
/// are the same tree down to the hash — which is what lets a chunk be promoted
/// from one tier to the other without the forest changing underneath it.
pub const TREE_FAR_STRIDE: usize = 4;

/// Side of one scatter chunk. Matches the erosion system's own 256 m streaming
/// tile, so one chunk's readiness gate is one erosion tile's readiness gate.
pub const TREE_CHUNK_SIZE: f32 = 256.0;

/// Candidate lattice spacing. A tree's XZ is a jittered point inside one cell
/// of this lattice, which is what keeps the scatter deterministic and
/// rebuildable: the same world position yields the same candidate forever, on
/// any machine, with no stored state.
///
/// This is the scatter's resolution, not its density: it sets how finely the
/// ground can be divided, and `tree_density` decides how much of that
/// subdivision is used. 4 m puts one candidate in every 16 m², which is the
/// spacing of a thinned stand — close enough that the density roll, not the
/// lattice, is what draws the tree line.
///
/// At 4 m the stand came out at half the density this file designs for: 35,303
/// trees measured against the ~70,000 that one tree per 35 m² over the 900 m
/// disc works out to, with the hard filters (rock, slope, shore, snow, furrow)
/// and the density roll taking the rest. Measured over a whole chunk, 4096
/// candidates yielded 3145 survivors and 1380 plantings — so the roll, not the
/// lattice, was doing the thinning. Tightening the lattice to 2.8 m doubles
/// the candidate count to 8281 without touching the roll, which doubles the
/// stand while leaving every clearing exactly where it was and exactly as
/// ragged. Raising `tree_density`'s floors instead would have flattened the
/// clearings, which is the one thing the scatter is meant to preserve.
pub const TREE_LATTICE: f32 = 2.8;

/// Ground steeper than this carries no trees. Chosen as the point where the
/// rock ramp has already begun on its own (terrain-fs.wgsl:773 turns rock on
/// at `terrainExposure` 0.24, and `slope` is its dominant term), so the slope
/// and rock tests agree instead of fighting.
pub const TREE_MAX_SLOPE: f32 = 0.62;

/// A site is rejected when it sits this far below the mean of its
/// neighbourhood. This is the furrow test, and it is deliberately a *shape*
/// test rather than a reading of the erosion delta: it catches every kind of
/// incision — hydraulic, depositional bank, or a groove the erosion atlas has
/// not finalized yet — without needing any of the GPU-only channel fields.
pub const TREE_FURROW_DEPTH: f32 = 1.35;

/// Radius of the neighbourhood the furrow test averages over. Comfortably
/// wider than the lattice, so a candidate is compared against the hillside it
/// stands on rather than against its own cell.
pub const TREE_FURROW_RADIUS: f32 = 18.0;

/// `fGrass` reaches 0.5 when `grassHeight` reaches sea level + 2.8 m, and
/// `grassDrift` can push the local surface 3.5 m below the true height
/// (terrain-fs.wgsl:703-715), so 7 m is the height at which grass has
/// provably taken over from shore sand. The extra metre is the salt-spray
/// setback the brief asks for: a tree does not grow at the waterline.
pub const TREE_SHORE_SETBACK: f32 = 8.0;

/// `terrainExposure` at which the scatter gives up on a site. The shader
/// starts blending rock at 0.24 and has fully committed by 0.52; 0.30 backs
/// the scatter off before any rock shows while leaving the soil shoulder
/// (0.085-0.22) untouched.
pub const TREE_ROCK_EXPOSURE: f32 = 0.30;

/// How far below the shader's snowline the scatter stops. The shader's own
/// snow height is `height + drifts + dryHollow*28 - ridge*18 -
/// (sunExposure-0.65)*22`, and only the drift terms are portable, so the
/// unknown part spans [-32.3, +28]. Taking the whole lower excursion makes the
/// test safe: a site is only accepted when it is below the snowline even under
/// the most snow-favouring reading of the fields the CPU cannot see.
pub const TREE_SNOWLINE_MARGIN: f32 = 33.0;

/// The shader's snowline begins at sea level + 92 m (terrain-fs.wgsl:817).
pub const TREE_SNOWLINE_HEIGHT: f32 = 92.0;

/// Density floor and ceiling, as an acceptance probability per lattice cell.
///
/// With a 4 m lattice the ceiling is one tree per 17 m² and the floor one per
/// 67 m², so a cell of ground takes between roughly 150 and 590 stems per
/// hectare. Those are the numbers a real closed-canopy conifer stand runs at.
/// The floor is not zero, so a clearing is a *thinning* — a stand you can see
/// through and walk through — rather than a bald patch, and the ceiling is
/// short of 1.0 so even the best soil keeps gaps. The 16x spread between the
/// two ends is what keeps the brief's clearings visible now that both ends
/// have moved up.
const TREE_DENSITY_MINIMUM: f32 = 0.18;
const TREE_DENSITY_MAXIMUM: f32 = 1.00;

/// Where [`tree_vigour`] stops meaning a sapling and starts meaning a medium
/// fir, and then a large one.
///
/// These are the field's 20th and 50th percentiles, measured.
///
/// The first pair was 0.42/0.68, reasoned from the *shape* of the vigour field
/// rather than from its distribution — and the distribution sits far below the
/// middle of its nominal [0, 1] range, because `soilPatches` and `groundGrowth`
/// are products of terrain masks that spend most of their time near zero. The
/// 0.42 boundary therefore landed on the 77th percentile, and the stand came
/// out 67.6% small, 13.9% medium, 18.4% large: two thirds of the forest was
/// 1.5 m saplings, which is what made a scatter of 68,000 trees read as
/// parkland. The scatter histograms the vigour of every site that clears the
/// hard tests — `TREES: chunk ... vigour per 0.1 [...]` — and over 154,405
/// sites in 37 chunks that distribution gives p20 = 0.14 and p50 = 0.26.
///
/// Splitting there yields 20% small, 30% medium, 50% large, which is the
/// gradient the doc comment on `tree_size` describes: the large fir on the
/// deep soil, the smaller classes on the thin ground at the edge of a clearing.
/// It also roughly doubles the stand's canopy area, since canopy goes as
/// height squared and the large class is six times the small one — at the
/// measured 1 tree per 29 m² of the mid field that takes the large trees from
/// 12.5 m apart to 7.6 m, against a 5.3 m crown.
const TREE_SMALL_MAX_VIGOUR: f32 = 0.14;
const TREE_MEDIUM_MAX_VIGOUR: f32 = 0.26;

// ---------------------------------------------------------------------------
// The portable half of the ground-cover field
// ---------------------------------------------------------------------------

/// `surfaceCellHash` (terrain-fs.wgsl:398). Integer hashing keeps cell
/// variation stable across negative world coordinates, which a float hash
/// does not.
fn surface_cell_hash(cell: [f32; 2], material_index: i32, octave: i32) -> u32 {
    let key_x = cell[0] as i32 as u32;
    let key_z = cell[1] as i32 as u32;
    let mut hash = key_x
        .wrapping_mul(0x9e37_79b9)
        .wrapping_add(key_z.wrapping_mul(0x85eb_ca6b))
        .wrapping_add((material_index as u32 + 1).wrapping_mul(0xc2b2_ae35))
        .wrapping_add((octave as u32).wrapping_mul(0x27d4_eb2f));
    hash ^= hash >> 16;
    hash = hash.wrapping_mul(0x7feb_352d);
    hash ^= hash >> 15;
    hash = hash.wrapping_mul(0x846c_a68b);
    hash ^= hash >> 16;
    hash
}

/// Quintic fade, the `w` term of `groundNoise`.
fn quintic(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

/// `groundNoise` (terrain-fs.wgsl:414): unbounded world-space value noise with
/// quintic interpolation, so warped patch boundaries stay smooth.
fn ground_noise(p: [f32; 2]) -> f32 {
    let cell = [p[0].floor(), p[1].floor()];
    let fade = [quintic(p[0] - cell[0]), quintic(p[1] - cell[1])];
    let inverse_hash_range = 1.0 / 16777215.0;
    let corner = |dx: f32, dz: f32| {
        (surface_cell_hash([cell[0] + dx, cell[1] + dz], 17, 0) & 0x00ff_ffff) as f32
            * inverse_hash_range
    };
    let (a, b) = (corner(0.0, 0.0), corner(1.0, 0.0));
    let (c, d) = (corner(0.0, 1.0), corner(1.0, 1.0));
    let top = a + (b - a) * fade[0];
    let bottom = c + (d - c) * fade[0];
    top + (bottom - top) * fade[1]
}

/// `rotateUV` (terrain-fs.wgsl:313).
fn rotate_uv(p: [f32; 2], angle: f32) -> [f32; 2] {
    let (sin, cos) = angle.sin_cos();
    [cos * p[0] - sin * p[1], sin * p[0] + cos * p[1]]
}

/// `mirrorTile` (terrain-fs.wgsl:170). GLSL's `mod` is floor-based, which is
/// why this is written out rather than using `%`.
fn mirror_tile(p: [f32; 2]) -> [f32; 2] {
    let m = [
        p[0] - 2.0 * (p[0] / 2.0).floor(),
        p[1] - 2.0 * (p[1] / 2.0).floor(),
    ];
    [(m[0] - 1.0).abs(), (m[1] - 1.0).abs()]
}

/// The warped domain every ground-cover field is evaluated in
/// (terrain-fs.wgsl:721-724). The slow warp bends the smaller fields so
/// openings bunch together and vary in size instead of making evenly spaced
/// islands of dirt.
fn ground_domain(world: [f32; 2]) -> [f32; 2] {
    let warp_x = ground_noise(offset(rotate_uv(scale(world, 0.011), 0.37), [17.3, -9.1]));
    let warp_z = ground_noise(offset(rotate_uv(scale(world, 0.009), 1.21), [-31.7, 23.4]));
    [
        world[0] + (warp_x - 0.5) * 48.0,
        world[1] + (warp_z - 0.5) * 48.0,
    ]
}

fn scale(p: [f32; 2], factor: f32) -> [f32; 2] {
    [p[0] * factor, p[1] * factor]
}

fn offset(p: [f32; 2], add: [f32; 2]) -> [f32; 2] {
    [p[0] + add[0], p[1] + add[1]]
}

/// `filteredGroundNoise` (terrain-fs.wgsl:429) at zero screen footprint.
///
/// The GPU mixes toward the field's mean once a pixel spans more than a third
/// of a noise cell, which only ever happens at distance. A scatter evaluates a
/// single point, so the filter's weight is exactly zero and the unfiltered
/// field is the correct value — not an approximation of it.
fn filtered_ground_noise(p: [f32; 2]) -> f32 {
    ground_noise(p)
}

/// The ground-cover fields one site is judged against, all of them exact ports.
struct GroundCover {
    ground_region: f32,
    ground_growth: f32,
    soil_patches: f32,
    rock_exposure_base: f32,
    snow_height: f32,
    /// World height at which grass has taken over from shore sand.
    grass_height: f32,
}

fn ground_cover(noise: &NoiseField, world: [f32; 2], height: f32) -> GroundCover {
    let domain = ground_domain(world);

    let ground_region = filtered_ground_noise(offset(
        rotate_uv(scale(domain, 0.007), 0.81),
        [51.8, 7.2],
    ));
    let ground_growth = filtered_ground_noise(offset(
        rotate_uv(scale(domain, 0.021), 1.73),
        [-12.5, 39.6],
    ));
    let soil_large = filtered_ground_noise(offset(
        rotate_uv(scale(domain, 0.043), 0.57),
        [6.2, 81.3],
    ));
    let soil_small = filtered_ground_noise(offset(
        rotate_uv(scale(domain, 0.137), 1.13),
        [73.1, -24.8],
    ));
    let soil_edge = filtered_ground_noise(offset(
        rotate_uv(scale(domain, 0.83), 2.04),
        [-41.6, -63.2],
    ));
    let soil_pattern = soil_large
        + (soil_small - soil_large) * (0.18 + (0.62 - 0.18) * ground_growth)
        + (soil_edge - 0.5) * 0.16;
    let soil_threshold = 0.66 + (0.43 - 0.66) * ground_region;
    let soil_patches = smooth_hermite(soil_threshold, soil_threshold + 0.22, soil_pattern);

    let rock_region = filtered_ground_noise(offset(
        rotate_uv(scale(domain, 0.0031), 0.26),
        [-57.4, 18.9],
    ));
    // Only the portable terms of `terrainExposure` (terrain-fs.wgsl:760). The
    // missing ones are the erosion-atlas fields, handled by the furrow test.
    let rock_exposure_base = (rock_region - 0.5) * 0.22 + (ground_growth - 0.5) * 0.10
        + (soil_large - 0.5) * 0.04;

    let snow_region = filtered_ground_noise(offset(
        rotate_uv(scale(domain, 0.0017), 0.63),
        [91.7, -53.2],
    ));
    let snow_drift = filtered_ground_noise(offset(
        rotate_uv(scale(domain, 0.013), 1.19),
        [-23.8, 67.4],
    ));
    let snow_drift_fine = filtered_ground_noise(offset(
        rotate_uv(scale(domain, 0.05), 0.41),
        [37.1, 12.6],
    ));
    let snow_drift_micro = filtered_ground_noise(offset(
        rotate_uv(scale(domain, 0.15), 1.87),
        [-61.3, -42.9],
    ));
    let snow_height = height
        + (snow_region - 0.5) * 100.0
        + (snow_drift - 0.5) * 42.0
        + (snow_drift_fine - 0.5) * 16.0
        + (snow_drift_micro - 0.5) * 6.0;

    // `grassDrift` is a sample of texture0 — the very noise field the renderer
    // uploads — taken through the same mirror fold `signedNoise` uses, so this
    // is the shader's own value rather than a model of it. `mirrorTile` yields
    // a texture uv at raylib's default repeat wrap; `sample_periodic_noise`
    // takes world coordinates, hence the two conversions either side.
    let grass_uv = mirror_tile(offset(rotate_uv(scale(world, 0.0045), 0.85), [0.43, 0.67]));
    let grass_drift = sample_periodic_noise(
        &noise.samples,
        (grass_uv[0] - 0.5) * NOISE_PERIOD,
        (grass_uv[1] - 0.5) * NOISE_PERIOD,
    ) * 0.5
        + 0.5;

    GroundCover {
        ground_region,
        ground_growth,
        soil_patches,
        rock_exposure_base,
        snow_height,
        grass_height: height + (grass_drift - 0.5) * 7.0,
    }
}

/// `smoothHermite` (terrain-height-functions.wgslinc), matching the shader's
/// floor on the edge span.
fn smooth_hermite(edge0: f32, edge1: f32, value: f32) -> f32 {
    let t = ((value - edge0) / (edge1 - edge0).max(0.0001)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

// ---------------------------------------------------------------------------
// Site evaluation
// ---------------------------------------------------------------------------

/// A site that passed every hard test, with the soft fields the scatter still
/// needs to turn into a yes/no.
pub struct TreeSite {
    pub height: f32,
    /// How favourable the ground is, in [0, 1]. Drives both the acceptance
    /// probability and, through it, where the clearings land.
    pub density: f32,
    /// 0 small, 1 medium, 2 large. Taken from the same cover fields that set
    /// `density`, so the tall trees stand where the soil is deepest and the
    /// small ones ring the clearings — which is what makes a stand read as a
    /// stand rather than as three species sprinkled independently.
    pub size: usize,
    /// The field `size` was bucketed from, before the bucket boundaries. The
    /// three size classes are 1.5 m, 4.4 m and 9.3 m tall, so which bucket a
    /// site lands in moves the canopy far more than the acceptance roll does —
    /// and the boundaries are constants chosen against a field whose real
    /// distribution is not visible from their definition. Carrying the raw
    /// value out lets the scatter histogram it and the boundaries be set from
    /// measurement instead of from an assumption about the terrain.
    pub vigour: f32,
}

/// The number of independent per-cell randoms [`cell_randoms`] returns. Named
/// so the scatter cannot silently index past the end when it grows a field.
///
/// Eight: jitter X, jitter Z, the density roll, the variant, the scale, the
/// lean's direction, the lean's magnitude, and the yaw. The variant, scale and
/// orientation draws are deliberately *separate* indices — while the variant
/// and the yaw shared one, every tree's yaw was a function of its mesh, and
/// with three variants the crossed cards of a stand fell into three yaw bands.
pub const TREE_CELL_RANDOM_COUNT: usize = 8;

/// Stable per-cell randoms in [0, 1).
///
/// Every decision the scatter makes about a candidate — where inside its cell
/// it sits, which of the three variants it is, its yaw, its scale, and its
/// tint — comes from here, and therefore from the lattice cell's integer hash
/// alone. Nothing is stored and nothing is random at runtime: the same world
/// position yields the same tree on every machine and in every session, which
/// is what lets chunks be dropped and rebuilt without the forest changing.
pub fn cell_randoms(cell: [f32; 2]) -> [f32; TREE_CELL_RANDOM_COUNT] {
    std::array::from_fn(|index| {
        // Seed 91+index: outside the ground-cover fields' seed space, so a
        // cell's jitter can never correlate with what grows on it.
        (surface_cell_hash(cell, 91 + index as i32, 0) & 0x00ff_ffff) as f32 / 16777215.0
    })
}

/// How far off vertical a tree may lean, in radians: 5% of a right angle, so
/// 4.5 degrees at the very most.
///
/// A full turn's 5% — 18 degrees — was the first reading of "up to 5%" and it
/// is far too much: at that cap a fir's trunk visibly departs from the ground
/// at an angle, and a stand of them reads as storm-damaged rather than as a
/// forest. Five percent of the *quadrant* keeps the lean inside the range where
/// it reads as a crook in a tree that is still growing at the sky.
///
/// The cap is on the *total* tilt, not per axis, so even the crookedest fir in
/// the forest is still unmistakably pointing at the sky — which is what lets
/// the rest of the pipeline keep assuming a tree's trunk is vertical. See
/// `TreeInstance::ground` in src/trees/mod.rs for where that assumption is
/// load-bearing.
pub const TREE_MAX_LEAN: f32 = std::f32::consts::FRAC_PI_2 * 0.05;

/// A tree's orientation as a world-space quaternion, `[x, y, z, w]`.
///
/// Yaw is a full uniform turn about up: a fir has no front, and at three mesh
/// variants a stand whose trees all face one way shows it.
///
/// The lean is a tilt of at most [`TREE_MAX_LEAN`] about a random *horizontal*
/// axis, composed so the tilt happens before the yaw — the tree leans the same
/// way relative to its own trunk whichever way it is then turned. The
/// magnitude is the signed square of the random, which crowds the distribution
/// towards zero: most trees come out near enough upright that the eye reads the
/// stand as vertical, and a minority carry the cap. A uniform magnitude would
/// put every tree at an average 9 degrees off vertical, which reads as a forest
/// falling over rather than one with a few crooked trees in it.
///
/// A quaternion rather than the yaw angle this used to be, for two reasons. The
/// vertex shader can then rotate with two cross products and no trigonometry,
/// and it does that for every vertex of every instance — eight million vertices
/// a frame in the filled forest, so six `sin`/`cos` calls per vertex is a real
/// cost rather than a rounding error. And three Euler angles would fix the lean
/// axes in world space, so a tree leaning north would lean *sideways* relative
/// to its own trunk once the yaw turned it; the quaternion applies the tilt in
/// the tree's frame.
pub fn tree_rotation(yaw_random: f32, lean_direction_random: f32, lean_random: f32) -> [f32; 4] {
    let signed = 2.0 * lean_random - 1.0;
    let lean = TREE_MAX_LEAN * signed * signed.abs();

    let (lean_sin, lean_cos) = lean.sin_cos();
    let (direction_sin, direction_cos) =
        (lean_direction_random * std::f32::consts::TAU).sin_cos();
    let (yaw_sin, yaw_cos) = (yaw_random * std::f32::consts::TAU).sin_cos();

    // Axis-angle about the horizontal direction (cos d, 0, sin d), then yaw
    // about up. Composing as `yaw * lean` applies the lean first, in the
    // tree's own frame.
    quaternion_product(
        [0.0, yaw_sin, 0.0, yaw_cos],
        [
            direction_cos * lean_sin,
            0.0,
            direction_sin * lean_sin,
            lean_cos,
        ],
    )
}

/// Hamilton product: the rotation `after` applied in the frame `before` leaves.
fn quaternion_product(after: [f32; 4], before: [f32; 4]) -> [f32; 4] {
    [
        after[3] * before[0] + after[0] * before[3] + after[1] * before[2] - after[2] * before[1],
        after[3] * before[1] - after[0] * before[2] + after[1] * before[3] + after[2] * before[0],
        after[3] * before[2] + after[0] * before[1] - after[1] * before[0] + after[2] * before[3],
        after[3] * before[3] - after[0] * before[0] - after[1] * before[1] - after[2] * before[2],
    ]
}

/// Decide whether one candidate point may carry a tree.
///
/// `slope` and `neighbourhood_mean` come from the chunk's height grid rather
/// than from more `sample_eroded_height` calls: both are neighbourhood
/// statistics, and rasterising the chunk once is several times cheaper than
/// asking for four extra heights per candidate.
///
/// `visibility_center` is the *chunk's* centre, never the player's. The
/// erosion fade is the only part of the height model that depends on where the
/// camera is, and a scatter keyed to it would place different trees depending
/// on which direction the player walked in. Passing the chunk centre puts
/// every candidate well inside `EROSION_VISIBILITY_FULL_RADIUS`, so
/// `erosionVisibility` is exactly 1 and the judgement is a pure function of
/// world position: same chunk, same trees, forever.
/// `cache` is `None` for the far tier, which judges the *uneroded* landform
/// instead. That is not a shortcut: erosion's whole contribution to the height
/// is the incision and deposition the flow simulation deposits, and the tiles
/// report it at around a metre (see the `incision` field of the `EROSION:
/// independent pass` line). A metre of error under a tree 1,000 m away is
/// 0.4 px of screen, and past `EROSION_VISIBILITY_ZERO_RADIUS` the renderer
/// fades the erosion term out entirely, so beyond 1,600 m `base_height` is not
/// an approximation of the drawn surface — it *is* the drawn surface. What the
/// far tier gives up is the furrow test's access to erosion-carved channels;
/// the test still runs, against the base landform's own hollows, which is the
/// scale that survives to that distance anyway. What it gains is that a far
/// chunk is ready the moment it is asked for, instead of waiting out the
/// ninety seconds the erosion tiles at that range take to simulate.
pub fn evaluate_site(
    cache: Option<&ErosionCache>,
    noise: &NoiseField,
    x: f32,
    z: f32,
    visibility_center: [f32; 2],
    slope: f32,
    neighbourhood_mean: f32,
) -> Result<TreeSite, SiteReject> {
    let height = match cache {
        Some(cache) => sample_eroded_height(cache, noise, x, z, visibility_center),
        None => crate::noise::base_height(noise, x, z),
    };

    // Underwater, on the beach, and too close to the ocean are one test: the
    // height at which grass has taken over from shore sand.
    if height < SEA_LEVEL + TREE_SHORE_SETBACK {
        return Err(SiteReject::Shore);
    }
    let world = [x, z];
    let cover = ground_cover(noise, world, height);
    if cover.grass_height < SEA_LEVEL + 2.8 {
        return Err(SiteReject::Shore);
    }

    // Steep ground sheds what grows on it, and is where rock begins.
    if slope > TREE_MAX_SLOPE {
        return Err(SiteReject::Slope);
    }
    // Rock: the portable part of terrainExposure, with the slope term folded
    // in. `incision`, `ridge` and `deposition` are the erosion atlas's, and
    // the furrow test below stands in for them.
    if slope + cover.rock_exposure_base > TREE_ROCK_EXPOSURE {
        return Err(SiteReject::Rock);
    }
    // Snow, read at its most snow-favouring.
    if cover.snow_height > SEA_LEVEL + TREE_SNOWLINE_HEIGHT - TREE_SNOWLINE_MARGIN {
        return Err(SiteReject::Snow);
    }
    // Eroded furrows: a site lying well below the ground around it is in a
    // channel, a gully, or a cut bank, whichever process made it.
    if height < neighbourhood_mean - TREE_FURROW_DEPTH {
        return Err(SiteReject::Furrow);
    }

    Ok(TreeSite {
        height,
        density: tree_density(&cover),
        size: tree_size(&cover),
        vigour: tree_vigour(&cover),
    })
}

/// Why a candidate point was refused.
///
/// Returned rather than logged so the scatter can tally a whole chunk and
/// report one line for it. Which test is doing the rejecting is the first
/// thing worth knowing when the forest comes out thinner than the density
/// numbers say it should, and guessing at it from the finished picture has
/// cost more time than the tally does.
#[derive(Clone, Copy, Debug)]
pub enum SiteReject {
    /// Underwater, on the beach, or inside the salt-spray setback.
    Shore,
    /// Steeper than [`TREE_MAX_SLOPE`].
    Slope,
    /// Exposed rock, or the shoulder just before it.
    Rock,
    /// Above the snowline's most snow-favouring reading.
    Snow,
    /// Down in an eroded channel, gully or cut bank.
    Furrow,
}

/// Which of the three firs fits this ground.
///
/// `soilPatches` is whether there is soil at all and `groundGrowth` is how
/// well it grows, so weighting them 60/40 puts the deep-soil stands on the
/// large trees and leaves the medium and small ones on the thin ground at the
/// edge of a clearing — the same gradient a real treeline shows.
fn tree_size(cover: &GroundCover) -> usize {
    let vigour = tree_vigour(cover);
    if vigour < TREE_SMALL_MAX_VIGOUR {
        0
    } else if vigour < TREE_MEDIUM_MAX_VIGOUR {
        1
    } else {
        2
    }
}

/// The bare field behind [`tree_size`], 0 to 1.
fn tree_vigour(cover: &GroundCover) -> f32 {
    0.6 * cover.soil_patches + 0.4 * cover.ground_growth
}

/// Turn the cover fields into an acceptance probability.
///
/// `soilPatches` is the terrain's own answer to "is there soil here", so
/// letting it drive density is what makes the clearings in the scatter line up
/// with the bare patches painted on the ground instead of falling across them.
/// `groundRegion` varies over hundreds of metres and `groundGrowth` over tens,
/// so the two together give stands, gaps and a ragged edge between them at
/// three different scales — which is the whole difference between a forest and
/// a lawn.
fn tree_density(cover: &GroundCover) -> f32 {
    // The floors are high on purpose. Multiplying three fields that each spend
    // most of their time near zero gives a product that is near zero almost
    // everywhere: the first version weighted these 0.12/0.35/0.30 and the
    // mean acceptance came out at 0.18, which is a thin scatter however high
    // the ceiling is set. The floors keep the product near the middle of its
    // range so that the *ceiling* is what varies, and a stand still reads as a
    // stand while a clearing still reads as a clearing.
    let soil = 0.40 + 0.60 * cover.soil_patches;
    let region = 0.62 + 0.38 * cover.ground_region;
    let growth = 0.58 + 0.42 * cover.ground_growth;
    let density = soil * region * growth;
    TREE_DENSITY_MINIMUM + (TREE_DENSITY_MAXIMUM - TREE_DENSITY_MINIMUM) * density.clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// Chunk rasterisation
// ---------------------------------------------------------------------------

/// Terrain height over one chunk's lattice, with a one-cell apron so slopes
/// and neighbourhood means are defined on the chunk's own cells. `side` counts
/// interior cells; the grid is `side + 2` square.
///
/// `step` is the lattice spacing the grid was rasterised at, in metres. It is a
/// property of the grid rather than the constant [`TREE_LATTICE`] because the
/// far tier rasterises the same chunk at [`TREE_FAR_STRIDE`] times the spacing:
/// `slope` is a finite difference and `neighbourhood_mean` a window, and both
/// have to divide by the spacing they were actually sampled at.
pub struct HeightGrid {
    pub side: usize,
    pub step: f32,
    pub heights: Vec<f32>,
}

impl HeightGrid {
    fn at(&self, ix: usize, iz: usize) -> f32 {
        self.heights[iz * (self.side + 2) + ix]
    }

    /// Central difference over the cell neighbourhood, in metres of rise per
    /// metre of run. The shader reconstructs its normal from screen
    /// derivatives; this reconstructs the same quantity from the ground
    /// itself, which is what a scatter can actually see.
    pub fn slope(&self, ix: usize, iz: usize) -> f32 {
        let step = self.step;
        let dx = (self.at(ix + 1, iz) - self.at(ix - 1, iz)) / (2.0 * step);
        let dz = (self.at(ix, iz + 1) - self.at(ix, iz - 1)) / (2.0 * step);
        (dx * dx + dz * dz).sqrt()
    }

    /// Mean height of the cells within `TREE_FURROW_RADIUS`, including this
    /// one. The radius is rounded to whole lattice cells, so the window is the
    /// same shape for every candidate.
    pub fn neighbourhood_mean(&self, ix: usize, iz: usize) -> f32 {
        let reach = (TREE_FURROW_RADIUS / self.step).round() as usize;
        let reach = reach.max(1).min(self.side);
        let mut total = 0.0f32;
        let mut count = 0.0f32;
        for dz in 0..=reach * 2 {
            for dx in 0..=reach * 2 {
                let sx = (ix + dx).saturating_sub(reach).min(self.side + 1);
                let sz = (iz + dz).saturating_sub(reach).min(self.side + 1);
                total += self.at(sx, sz);
                count += 1.0;
            }
        }
        total / count
    }
}

/// Rasterise one chunk's terrain height on the scatter lattice.
///
/// `cache` is `None` for the far tier; see [`evaluate_site`]. `step` is the
/// lattice spacing to sample at — [`TREE_LATTICE`] for the near tier, and
/// [`TREE_FAR_STRIDE`] times that for the far one, whose sample points are
/// therefore a subset of the near tier's and land on the same world positions.
///
/// The grid is rasterised at the tier's own spacing rather than at the near
/// spacing with candidates skipped, and that is the difference between the far
/// tier being cheap and being pointless: the grid is `side + 2` square and one
/// height model call per cell, so a far chunk that rasterised 93 x 93 and then
/// used every sixteenth cell would pay the full 8,649 samples to evaluate 529
/// of them. At the coarse spacing it is 25 x 25.
pub fn build_height_grid(
    cache: Option<&ErosionCache>,
    noise: &NoiseField,
    chunk: [i32; 2],
    step: f32,
    visibility_center: [f32; 2],
) -> HeightGrid {
    let side = (TREE_CHUNK_SIZE / step).round() as usize;
    let origin = [
        chunk[0] as f32 * TREE_CHUNK_SIZE,
        chunk[1] as f32 * TREE_CHUNK_SIZE,
    ];
    let mut heights = Vec::with_capacity((side + 2) * (side + 2));
    for iz in 0..side + 2 {
        for ix in 0..side + 2 {
            // The apron extends one cell past each edge, so cell 0's backward
            // difference reaches into the neighbouring chunk's ground.
            let x = origin[0] + (ix as f32 - 1.0) * step;
            let z = origin[1] + (iz as f32 - 1.0) * step;
            heights.push(match cache {
                Some(cache) => sample_eroded_height(cache, noise, x, z, visibility_center),
                None => crate::noise::base_height(noise, x, z),
            });
        }
    }
    HeightGrid {
        side,
        step,
        heights,
    }
}

/// Whether every erosion tile this chunk's height model actually reads has
/// finished simulating.
///
/// Scattering a chunk before its tiles are ready would judge candidates
/// against the un-eroded `baseHeight`, and the furrow test — the whole reason
/// the erosion fields are needed here — would find no furrows at all. The
/// chunk is simply deferred until the ground it sits on exists.
///
/// The set asked about is exactly the set `sample_eroded_height` reads: the
/// four candidate tiles of `erosion_candidate_minimum`, whose retained
/// footprints are the ones the blend mask weights. `TREE_CHUNK_SIZE` is half
/// `EROSION_TILE_STRIDE`, so a chunk spans two candidate tiles per axis and the
/// union over its corners is a 2x2 block.
///
/// This used to inflate that block by `EROSION_FOOTPRINT_SIZE * 0.5` on every
/// side — a tile of slack for the blend mask's support, giving a 4x4 block of
/// 16 tiles where the height model reads 4. The mask does reach half a
/// footprint past a pass centre, but its Hermite tent weight is *zero* at that
/// boundary (`sample_erosion_blend_mask` returns 0 outside the support square
/// and the tent vanishes at its edge), so the extra twelve tiles were read by
/// nothing and gated everything. The cost was the whole pop-in: the player's
/// own chunk needs tiles {0,1}x{0,1}, which are the first four the simulator
/// ever finishes, but it was waiting on the far corner of a 4x4 block and so
/// went unscattered for 15 s while a chunk 365 m away — the first whose
/// sixteen happened to complete — planted the forest's first tree. Every
/// chunk is now ready as soon as the ground under it is, and the nearest
/// chunks come first rather than whichever one the tile order happened to
/// finish.
pub fn chunk_is_ready(cache: &ErosionCache, chunk: [i32; 2]) -> bool {
    let minimum = [
        chunk[0] as f32 * TREE_CHUNK_SIZE,
        chunk[1] as f32 * TREE_CHUNK_SIZE,
    ];
    let maximum = [
        minimum[0] + TREE_CHUNK_SIZE,
        minimum[1] + TREE_CHUNK_SIZE,
    ];
    let first = crate::erosion::erosion_candidate_minimum(minimum[0], minimum[1]);
    let last = crate::erosion::erosion_candidate_minimum(maximum[0], maximum[1]);
    for z in first.z..=last.z + 1 {
        for x in first.x..=last.x + 1 {
            let found = cache.tiles.get(&crate::erosion::tile(x, z));
            if !found.is_some_and(|found| found.state == ErosionTileState::Ready) {
                return false;
            }
        }
    }
    true
}
