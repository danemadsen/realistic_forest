//! How a river shapes the ground, and what the water is doing at a point.
//!
//! A river is drawn as a chain of short carve segments. Each segment bounds
//! the terrain from above and below near its centreline:
//!
//! - **Upper envelope.** Inside the wetted width the bed follows a skewed
//!   bowl below the water surface, deepest toward the outer bank of a
//!   bend. Beyond the waterline a bank cone rises at the bank slope, then
//!   ever more steeply, so ground that stands above it (a spur a meander
//!   swings into, the saddle a channel breaches) is cut back into a bank.
//! - **Lower envelope.** Just outside the waterline the ground is held a
//!   little above the water, so the channel always contains its river even
//!   where the floodplain dips; a short outer slope returns to the terrain.
//!
//! The terrain height is `min(max(h, lower), upper)`, with the upper bound
//! the minimum and the lower bound the maximum over every nearby segment, so
//! confluences open into each other and a channel always wins over a
//! neighbour's levee. Crossing banks and their shallow shoreline strip round
//! together once, opening a bounded shelf at the junction. Deep channel beds
//! retain their original profile.
//!
//! Everything that shapes the channel (water, width, depth, bank slope,
//! skew, levee) is given at both ends of a segment and interpolated along
//! it, so two segments meeting at a node carve the same ground there. A
//! segment between two others has a round cap at either end, so nothing
//! switches on along a line across the bank; the water in a cap is the reach
//! beside's, rising up the cap before the start as the reach before falls,
//! so the cap never cuts below that reach's bed, and falling down the cap
//! past the end for the levee, so it never holds a ledge over the bank
//! beside the next reach. A segment no reach comes before (a river's first,
//! or its first out of a lake) starts flat: nothing behind it is its.
//!
//! The same arithmetic runs in `river-functions.wgslinc` on the GPU, from the
//! same uploaded segments, so the drawn ground, the player's footing, the
//! seated plants and the erosion simulation all see one channel.

/// Ordinary banks stop shaping the ground this far past the waterline.
/// Shelving banks at still-water junctions spread over up to twice this run.
pub const BANK_REACH: f32 = 12.0;
/// Curvature of an ordinary bank cone; shelving banks stretch its run.
pub const BANK_CURVE: f32 = 0.16;
/// Segments a lookup reads from one grid cell at most, here and on the GPU.
pub const MAX_CANDIDATES: usize = 64;
/// Slope of a levee's outer face back down to the terrain: a natural levee
/// is a broad, low ridge, not a dike.
pub const LEVEE_OUTER_SLOPE: f32 = 0.15;
/// Height over which the carve's creases are rounded where it meets the
/// natural ground: the top of a bank, the foot of a levee.
pub const CARVE_ROUNDING: f32 = 0.8;
/// Round intersecting bank cones over this height difference. Parallel
/// reaches retain their own bank profile, including subdivided reaches.
pub const BANK_UNION_ROUNDING: f32 = 2.8;
/// Shoreline rounding fades out within this distance inside a channel.
pub const BANK_UNION_SHORE_BLEND: f32 = 0.75;
/// Largest new wetted shelf outside either channel's original waterline.
pub const BANK_UNION_INTRUSION: f32 = 0.6;

/// Polynomial smooth minimum: `min(a, b)` with the corner rounded over a
/// difference of `k`, pulled down by at most `k / 4` where `a == b`.
/// `smoothMin` in river-functions.wgslinc.
pub fn smooth_min(a: f32, b: f32, k: f32) -> f32 {
    let h = (k - (a - b).abs()).max(0.0) / k;
    a.min(b) - h * h * k * 0.25
}

/// `smooth_min`'s mirror: `max(a, b)` with the corner rounded.
pub fn smooth_max(a: f32, b: f32, k: f32) -> f32 {
    let h = (k - (a - b).abs()).max(0.0) / k;
    a.max(b) + h * h * k * 0.25
}
/// Side of a cell of the segment lookup grid, metres.
pub const GRID_CELL: f32 = 32.0;

/// One carve segment, as the GPU reads it: 88 bytes, mirrored by
/// `RiverSegment` in river-functions.wgslinc.
///
/// Everything that shapes the channel is given at both ends and
/// interpolated along the segment, so two segments meeting at a node carve
/// the same ground there: a bank slope, skew or levee held constant along
/// each segment would change at every node, and down a bank that is a step.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RiverSegment {
    /// Centreline start and end, world XZ.
    pub a: [f32; 2],
    pub b: [f32; 2],
    /// Water surface height at the start and end.
    pub water: [f32; 2],
    /// Wetted half width at the start and end.
    pub half_width: [f32; 2],
    /// Depth of the thalweg below the water at the start and end.
    pub depth: [f32; 2],
    /// Mean flow speed at the start and end, m/s.
    pub speed: [f32; 2],
    /// Bank slope (rise over run) just past the waterline, at the start and
    /// end.
    pub bank: [f32; 2],
    /// Thalweg skew toward the left (+) or right (-) bank, -1..1: the outer
    /// bank of a bend, where the pool is deep and the bank is cut steep.
    pub skew: [f32; 2],
    /// How firmly the ground beside the channel is held above the water,
    /// 0..1: none where the river meets the sea, whose bed must not rise.
    pub levee: [f32; 2],
    /// How the water runs on round the segment's ends, as the fall per metre
    /// of the reaches beside them: up the river behind its start (as fast as
    /// the steepest of the reaches its start cap reaches back beside), and
    /// down the reach after its end (`segment_envelope`). A start below zero
    /// has no reach before it, and the segment starts flat.
    pub cap_slope: [f32; 2],
    /// Whitewater, 0 calm to 1 a cascade down rapids (rough beds, bare rock
    /// and foam), at the start and end: given at both ends like the rest, so
    /// the bed does not change from gravel to rock along a line at a node.
    pub turbulence: [f32; 2],
}

const _: () = assert!(std::mem::size_of::<RiverSegment>() == 88);

/// `cap_slope[0]` of a segment no reach comes before.
pub const FLAT_START: f32 = -1.0;

/// Bank run for a bank slope: low, shelving banks need a broad shoulder as
/// well as a gentle slope at the waterline. Stretching the same bank cone
/// keeps its quadratic term from turning the mouth's softened banks straight
/// back into walls.
pub fn bank_run(bank: f32) -> f32 {
    2.0 - smoothstep(0.2, 0.55, bank)
}

impl RiverSegment {
    /// The water's fall per metre along the segment.
    pub fn slope(&self) -> f32 {
        let length = (self.b[0] - self.a[0]).hypot(self.b[1] - self.a[1]).max(1e-3);
        (self.water[0] - self.water[1]).max(0.0) / length
    }

    /// A segment by itself, which starts flat and whose end cap's levee
    /// falls on at its own slope.
    #[cfg(test)]
    pub fn alone(mut self) -> Self {
        self.cap_slope = [FLAT_START, self.slope()];
        self
    }

    /// Largest distance from the centreline at which the segment shapes the
    /// ground.
    pub fn reach(&self) -> f32 {
        self.half_width[0].max(self.half_width[1]) + BANK_REACH * bank_run(self.bank[0].min(self.bank[1]))
    }
}

/// What one segment, or the whole river network, says about a point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Envelope {
    /// The ground may not stand above this.
    pub upper: f32,
    /// The ground may not lie below this.
    pub lower: f32,
    /// Distance past the nearest waterline, metres: negative inside a
    /// channel.
    pub bank_distance: f32,
    /// The water surface of the channel nearest its waterline here.
    pub water: f32,
    /// Flow velocity at the point, world XZ, m/s.
    pub velocity: [f32; 2],
    /// Wetted half width of that channel.
    pub half_width: f32,
    /// Its whitewater, 0..1.
    pub turbulence: f32,
    /// Which side of a bend the point lies on: toward +0.6 on the outside
    /// (a cut bank), toward -0.6 on the inside (a point bar).
    pub bend: f32,
    /// The level of a lake whose basin reaches here, or -infinity: ground
    /// below it lies under the lake.
    pub lake: f32,
}

/// Metres from a lake's shore per metre the ground stands above or below
/// its water, for a shore's typical slope; how near the shore a point is,
/// in the same terms as a river's bank distance.
pub const LAKE_SHORE_RUN: f32 = 6.0;

impl Envelope {
    pub const NONE: Envelope = Envelope {
        upper: f32::INFINITY,
        lower: f32::NEG_INFINITY,
        bank_distance: f32::INFINITY,
        water: f32::NEG_INFINITY,
        velocity: [0.0, 0.0],
        half_width: 0.0,
        turbulence: 0.0,
        bend: 0.0,
        lake: f32::NEG_INFINITY,
    };

    /// Distance past the nearest waterline, river or lake, for ground at
    /// `height`: negative under water.
    pub fn bank_at(&self, height: f32) -> f32 {
        self.bank_distance.min((height - self.lake) * LAKE_SHORE_RUN)
    }

    /// The water standing over ground at `height` here, if any: a river's
    /// surface in its channel, a lake's over its bed.
    pub fn water_over(&self, height: f32) -> Option<f32> {
        let river = (self.bank_distance < 0.0).then_some(self.water);
        let lake = (height < self.lake).then_some(self.lake);
        match (river, lake) {
            (Some(r), Some(l)) => Some(r.max(l)),
            (r, l) => r.or(l),
        }
    }

    /// Apply the envelope to a terrain height, with the creases where the
    /// carve meets the natural ground rounded off: a bank's top curves over
    /// into the land above it, and a levee's foot into the land below.
    pub fn clamp(&self, height: f32) -> f32 {
        smooth_min(smooth_max(height, self.lower, CARVE_ROUNDING), self.upper, CARVE_ROUNDING)
    }

    pub fn is_none(&self) -> bool {
        self.bank_distance == f32::INFINITY
    }
}

fn mix(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn smoothstep(edge0: f32, edge1: f32, value: f32) -> f32 {
    let t = ((value - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// One segment's envelope at `p`. Mirrored line for line by
/// `riverSegmentEnvelope` in river-functions.wgslinc.
///
/// A segment between two others has a round cap at either end, so nothing
/// switches on along a line: the cap carries the channel on round its node,
/// and the reach beside carves the same ground there. The water in a cap is
/// the reach beside's: up the cap before its start it rises as the reach
/// before falls, so the cap never cuts below that reach's bed, and down the
/// cap past its end the levee falls as the reach after does, so it never
/// holds a ledge up beside it. A segment no reach comes before starts flat:
/// a river's first segment, or its first out of a lake.
pub fn segment_envelope(segment: &RiverSegment, p: [f32; 2]) -> Envelope {
    let ab = [segment.b[0] - segment.a[0], segment.b[1] - segment.a[1]];
    let ap = [p[0] - segment.a[0], p[1] - segment.a[1]];
    let length_squared = (ab[0] * ab[0] + ab[1] * ab[1]).max(1e-6);
    let length = length_squared.sqrt();
    let along = (ap[0] * ab[0] + ap[1] * ab[1]) / length_squared;
    let behind = (-along).max(0.0) * length;
    if behind > 0.0 && segment.cap_slope[0] < 0.0 {
        return Envelope::NONE;
    }
    let t = along.clamp(0.0, 1.0);
    let beyond = (along - 1.0).max(0.0) * length;
    let offset = [ap[0] - ab[0] * t, ap[1] - ab[1] * t];
    let distance = (offset[0] * offset[0] + offset[1] * offset[1]).sqrt();
    let half_width = mix(segment.half_width[0], segment.half_width[1], t).max(0.05);
    let past_bank = distance - half_width;
    let bank_slope = mix(segment.bank[0], segment.bank[1], t);
    let bank_run = bank_run(bank_slope);
    let bank_reach = BANK_REACH * bank_run;
    if past_bank >= bank_reach {
        return Envelope::NONE;
    }
    let water = mix(segment.water[0], segment.water[1], t) + segment.cap_slope[0].max(0.0) * behind;
    let depth = mix(segment.depth[0], segment.depth[1], t);
    let direction = [ab[0] / length, ab[1] / length];
    // Signed distance from the centreline's line, + on the left of the flow,
    // and which way across the flow the point lies, -1..1: beside the
    // segment it is a side, and round a cap it turns smoothly from one side
    // to the other, so the skew needs no seam where the sides meet.
    let lateral = direction[0] * ap[1] - direction[1] * ap[0];
    let across = lateral / distance.max(1e-6);
    let skew = mix(segment.skew[0], segment.skew[1], t);
    let speed = mix(segment.speed[0], segment.speed[1], t);
    let mut envelope = Envelope {
        upper: f32::INFINITY,
        lower: f32::NEG_INFINITY,
        bank_distance: past_bank,
        water,
        velocity: [direction[0] * speed, direction[1] * speed],
        half_width,
        turbulence: mix(segment.turbulence[0], segment.turbulence[1], t),
        bend: skew * across,
        lake: f32::NEG_INFINITY,
    };
    if past_bank < 0.0 {
        // A flat-bottomed bowl, skewed: (1 - u^4)(1 + skew u) is zero at
        // both banks and deepest toward the outer one. Its sides drop
        // steeply under the waterline, so the water is deep right up to the
        // banks rather than shoaling over a broad parabola; its mean depth
        // is four fifths of the thalweg's.
        let u = distance / half_width;
        let profile = (1.0 - u * u * u * u) * (1.0 + skew * lateral / half_width);
        envelope.upper = water - depth * profile;
        // The current runs fastest over the thalweg and stalls at the banks.
        let lateral_speed = (1.0 - u * u).max(0.0).powf(0.35);
        envelope.velocity = [envelope.velocity[0] * lateral_speed * 1.25, envelope.velocity[1] * lateral_speed * 1.25];
    } else {
        // Cut banks stand steep on the outside of a bend; point bars slope
        // gently into the water on the inside.
        let bank = bank_slope * (1.0 + 0.75 * skew * across).max(0.3);
        let shoulder = past_bank / bank_run;
        // Relax both constraints before their finite lookup support ends.
        // Otherwise a high terrace (or deep hollow under a levee) jumps
        // straight back to its uncarved height at the reach boundary.
        let edge = smoothstep(0.5 * bank_reach, bank_reach, past_bank);
        // Ten kilometres already releases the bound past the entire terrain
        // height range. Saturate it before tiny edge-distance differences
        // amplify into large CPU/GPU discrepancies in an inactive bound.
        let release = (bank_reach * edge * edge / (1.0 - edge).max(1e-5)).min(1.0e4);
        envelope.upper = water + bank * past_bank + BANK_CURVE * shoulder * shoulder + release;
        let freeboard = 0.1 + 0.25 * depth;
        let levee_width = 0.8 + 0.3 * half_width;
        // Past the segment's end its levee falls on as the water after it
        // does: down a rapid steeper than the levee's outer slope, the end
        // held up at its own water would outrank the next segment's lower
        // levee beside it, a ledge at every node, a flight of steps down the
        // bank.
        let fall = segment.cap_slope[1] * beyond;
        // Lower a fading levee gradually into the ground. A fixed 10 km
        // offset made almost any value below 1 switch it off immediately,
        // leaving a step where a channel approached a lake or the sea.
        let levee = mix(segment.levee[0], segment.levee[1], t);
        if levee > 0.0 {
            let retreat = 1.0 - levee;
            let levee_release = (depth + 0.25) * retreat * retreat / levee.max(1e-5);
            envelope.lower = water - fall + (bank * past_bank).min(freeboard)
                - (past_bank - levee_width).max(0.0) * LEVEE_OUTER_SLOPE
                - levee_release - release;
        }
        envelope.velocity = [0.0, 0.0];
    }
    envelope
}

/// Combine one segment's envelope into the running result. Mirrored by
/// `riverCombine` in river-functions.wgslinc.
pub fn combine(total: &mut Envelope, next: &Envelope) {
    if next.is_none() {
        return;
    }
    total.upper = total.upper.min(next.upper);
    total.lower = total.lower.max(next.lower);
    // The water, its motion and the bank distance come from the channel
    // whose waterline is nearest, measured in its own widths so a creek
    // meeting a river hands over to it inside the larger channel.
    let score = |e: &Envelope| e.bank_distance / e.half_width.clamp(0.5, 4.0);
    if total.is_none() || score(next) < score(total) {
        total.bank_distance = next.bank_distance;
        total.water = next.water;
        total.velocity = next.velocity;
        total.half_width = next.half_width;
        total.turbulence = next.turbulence;
        total.bend = next.bend;
    }
}

fn bank_surface_slope(segment: &RiverSegment) -> f32 {
    // Conservative slope at the waterline, including either side of a bend.
    // Keep this fixed across the union's bisector: using the nearest bank's
    // changing secant slope would put another cusp in the waterline itself.
    let at = |end: usize| (segment.bank[end] * (1.0 - 0.75 * segment.skew[end].abs()).max(0.3)).max(0.001);
    at(0).min(at(1))
}

/// Round one pair of intersecting banks against the unchanged raw upper
/// minimum. Taking the lowest pair result once, rather than smoothing the
/// running union repeatedly, prevents extra segments from excavating deeper.
/// Mirrored by `riverBankUnionUpper` in river-functions.wgslinc.
fn bank_union_upper(
    primary: &RiverSegment,
    first: &Envelope,
    next: &RiverSegment,
    second: &Envelope,
) -> f32 {
    let nearest_bank = first.bank_distance.min(second.bank_distance);
    if nearest_bank <= -BANK_UNION_SHORE_BLEND || second.is_none() {
        return first.upper;
    }
    let primary_direction = [primary.b[0] - primary.a[0], primary.b[1] - primary.a[1]];
    let other_direction = [next.b[0] - next.a[0], next.b[1] - next.a[1]];
    let cross = primary_direction[0] * other_direction[1] - primary_direction[1] * other_direction[0];
    let primary_length_squared = primary_direction[0].powi(2) + primary_direction[1].powi(2);
    let other_length_squared = other_direction[0].powi(2) + other_direction[1].powi(2);
    let length_product = (primary_length_squared * other_length_squared).max(1e-12);
    // The squared sine is independent of tangent orientation: straight
    // overlapping reaches cannot deepen one another, and a joining branch
    // eases its bank progressively as its angle opens.
    let turn = (cross * cross / length_product).clamp(0.0, 1.0);
    let bank_slope = bank_surface_slope(primary).min(bank_surface_slope(next));
    // A smooth minimum lowers equal heights by one quarter of its radius.
    // Bound that lowering by the bank's rise over the allowed shelf width.
    let shelf_rounding = 4.0 * BANK_UNION_INTRUSION * bank_slope;
    // Keep the same bound when the two water levels differ: beyond the
    // allowed shelf the union cannot newly cut below either water surface.
    let shelf_clearance = first.upper - first.water.max(second.water)
        + bank_slope * (BANK_UNION_INTRUSION - nearest_bank);
    let bed_fade = smoothstep(-BANK_UNION_SHORE_BLEND, 0.0, nearest_bank);
    let rounding = BANK_UNION_ROUNDING.min(shelf_rounding).min(4.0 * shelf_clearance.max(0.0)) * turn * bed_fade;
    if rounding <= 0.0 {
        return first.upper;
    }
    smooth_min(first.upper, second.upper, rounding)
}

fn round_bank_union(
    total: &mut Envelope,
    segments: &[RiverSegment],
    candidates: &[u32],
    p: [f32; 2],
    primary: Option<usize>,
) {
    // Deep water keeps its exact bed. At the shoreline the same rounding
    // continues into a shallow strip, so its contour does not keep a cusp.
    if total.bank_distance <= -BANK_UNION_SHORE_BLEND {
        return;
    }
    let Some(primary) = primary else { return; };
    let raw_bank_distance = total.bank_distance;
    let first = segment_envelope(&segments[primary], p);
    if first.bank_distance <= -BANK_UNION_SHORE_BLEND {
        return;
    }
    for &index in candidates.iter().take(MAX_CANDIDATES) {
        let next = &segments[index as usize];
        let second = segment_envelope(next, p);
        let rounded_upper = bank_union_upper(&segments[primary], &first, next, &second);
        if rounded_upper < total.upper {
            total.upper = rounded_upper;
            let bank_slope = bank_surface_slope(&segments[primary]).min(bank_surface_slope(next));
            let shelf_water = first.water.max(second.water);
            // A newly cut shelf connects to the participating water above
            // it. Existing flowing beds retain their original water owner.
            if raw_bank_distance >= 0.0 && first.bank_distance >= 0.0 && second.bank_distance >= 0.0 && rounded_upper < shelf_water {
                total.water = total.water.max(shelf_water);
            }
            let shelf_distance = ((rounded_upper - total.water) / bank_slope).max(-BANK_UNION_INTRUSION);
            total.bank_distance = total.bank_distance.min(shelf_distance);
        }
    }
}

/// A uniform grid of segment lists over the network's whole domain, with
/// the lakes' surfaces over each cell they reach. Uploaded to the GPU as one
/// `u32` array: an eight-word header, then `(offset, segment count, lake)`
/// per cell, then the index lists the offsets point into, then the lake
/// records. A cell's lake word is `NO_LAKE_RECORD` or the offset of its
/// record (`LakeRecord`).
#[derive(Clone, Debug, Default)]
pub struct SegmentGrid {
    pub origin: [f32; 2],
    pub resolution: usize,
    /// Per cell: offset into `indices` and segment count.
    pub cells: Vec<[u32; 2]>,
    pub indices: Vec<u32>,
    /// The lakes' surfaces over the cells they reach, by cell.
    pub lakes: std::collections::BTreeMap<u32, LakeRecord>,
}

/// A lake's surface over one grid cell: its height at the corners of the
/// cell's 8 x 8 lake cells (`GRID_CELL / 8`, the flow grid's cells), as the
/// highest corner's height and each corner's depth under it in
/// `LAKE_DEPTH_UNIT`s.
/// Under a lake's sheet the surface is the sheet itself, corner for corner,
/// so ground below it is exactly the ground the drawn water covers; past the
/// sheet it runs on under the ground and sinks away from the water, so a
/// shore measured from it fades out smoothly (`network::lake_surface`).
/// Between corners it is interpolated over the two triangles the sheet's
/// quads are drawn as, split from corner (0, 0) to corner (1, 1).
#[derive(Clone, Debug, PartialEq)]
pub struct LakeRecord {
    pub top: f32,
    pub depths: [u16; LAKE_CORNERS],
}

/// Corners across a lake record.
pub const LAKE_CORNERS_ACROSS: usize = LAKE_CELLS_ACROSS + 1;
pub const LAKE_CORNERS: usize = LAKE_CORNERS_ACROSS * LAKE_CORNERS_ACROSS;
/// Words in a lake record: its top, then the corners' depths two to a word.
pub const LAKE_RECORD_WORDS: usize = 1 + LAKE_CORNERS.div_ceil(2);
/// Metres per unit of a corner's depth: 2 mm, for 131 m of range under a
/// record's top.
pub const LAKE_DEPTH_UNIT: f32 = 0.002;
/// The depth of a corner no lake's surface reaches.
pub const LAKE_NO_CORNER: u16 = u16::MAX;

impl LakeRecord {
    /// The surface at corner `k` (`z * LAKE_CORNERS_ACROSS + x`), or
    /// `NO_LAKE`.
    pub fn corner(&self, k: usize) -> f32 {
        match self.depths[k] {
            LAKE_NO_CORNER => NO_LAKE,
            depth => self.top - depth as f32 * LAKE_DEPTH_UNIT,
        }
    }

    /// The surface at `local`, the point in lake cells from the record's
    /// corner (0..8 on each axis), or `NO_LAKE`.
    pub fn at(&self, local: [f32; 2]) -> f32 {
        let across = LAKE_CORNERS_ACROSS;
        let i = [local[0].floor().clamp(0.0, (LAKE_CELLS_ACROSS - 1) as f32), local[1].floor().clamp(0.0, (LAKE_CELLS_ACROSS - 1) as f32)];
        let (u, v) = (local[0] - i[0], local[1] - i[1]);
        let k = i[1] as usize * across + i[0] as usize;
        let h00 = self.corner(k);
        let h11 = self.corner(k + across + 1);
        let (h, other) = if v >= u {
            let h01 = self.corner(k + across);
            (h00 + (h11 - h01) * u + (h01 - h00) * v, h01)
        } else {
            let h10 = self.corner(k + 1);
            (h00 + (h10 - h00) * u + (h11 - h10) * v, h10)
        };
        if h00.min(h11).min(other) <= NO_LAKE { NO_LAKE } else { h }
    }
}

/// The "no lake" level on the GPU, where infinities are best avoided.
pub const NO_LAKE: f32 = -1.0e30;
/// A cell's lake word when no lake reaches it.
pub const NO_LAKE_RECORD: u32 = u32::MAX;
/// Lake cells across a grid cell.
pub const LAKE_CELLS_ACROSS: usize = 8;
/// Words per cell in the GPU grid.
pub const GRID_CELL_WORDS: usize = 3;

/// Words in the GPU grid header.
pub const GRID_HEADER_WORDS: usize = 8;

impl SegmentGrid {
    pub fn build(origin: [f32; 2], resolution: usize, segments: &[RiverSegment]) -> Self {
        let cell_range = |minimum: [f32; 2], maximum: [f32; 2]| {
            let cell = |value: f32, axis: usize| ((value - origin[axis]) / GRID_CELL).floor() as i64;
            (
                cell(minimum[0], 0).max(0),
                cell(maximum[0], 0).min(resolution as i64 - 1),
                cell(minimum[1], 1).max(0),
                cell(maximum[1], 1).min(resolution as i64 - 1),
            )
        };
        let mut segment_buckets: Vec<Vec<u32>> = vec![Vec::new(); resolution * resolution];
        for (index, segment) in segments.iter().enumerate() {
            let reach = segment.reach();
            let (x0, x1, z0, z1) = cell_range(
                [segment.a[0].min(segment.b[0]) - reach, segment.a[1].min(segment.b[1]) - reach],
                [segment.a[0].max(segment.b[0]) + reach, segment.a[1].max(segment.b[1]) + reach],
            );
            for z in z0..=z1 {
                for x in x0..=x1 {
                    segment_buckets[z as usize * resolution + x as usize].push(index as u32);
                }
            }
        }
        let mut cells = Vec::with_capacity(segment_buckets.len());
        let mut indices = Vec::new();
        for bucket in &segment_buckets {
            cells.push([indices.len() as u32, bucket.len() as u32]);
            indices.extend_from_slice(bucket);
        }
        Self {
            origin,
            resolution,
            cells,
            indices,
            lakes: Default::default(),
        }
    }

    fn cell_index(&self, p: [f32; 2]) -> Option<usize> {
        let x = ((p[0] - self.origin[0]) / GRID_CELL).floor();
        let z = ((p[1] - self.origin[1]) / GRID_CELL).floor();
        if x < 0.0 || z < 0.0 || x >= self.resolution as f32 || z >= self.resolution as f32 {
            return None;
        }
        Some(z as usize * self.resolution + x as usize)
    }

    fn cell(&self, p: [f32; 2]) -> Option<[u32; 2]> {
        self.cell_index(p).map(|index| self.cells[index])
    }

    /// Set the lakes' surfaces from the heights at flow-grid corners
    /// (corner `c` at `c * cell_size` metres; `cell_size` is
    /// GRID_CELL / LAKE_CELLS_ACROSS and the grid's origin lies on a corner).
    /// A corner on a cell's border belongs to every cell sharing it.
    pub fn set_lakes(&mut self, corners: &std::collections::HashMap<[i32; 2], f32>, cell_size: f32) {
        let across = LAKE_CELLS_ACROSS as i64;
        let mut heights: std::collections::BTreeMap<u32, [f32; LAKE_CORNERS]> = Default::default();
        for (&corner, &height) in corners {
            let g = [
                ((corner[0] as f32 * cell_size - self.origin[0]) / cell_size).round() as i64,
                ((corner[1] as f32 * cell_size - self.origin[1]) / cell_size).round() as i64,
            ];
            let cells = |g: i64| {
                let cell = g.div_euclid(across);
                if g.rem_euclid(across) == 0 { vec![cell, cell - 1] } else { vec![cell] }
            };
            for cz in cells(g[1]) {
                for cx in cells(g[0]) {
                    if cx < 0 || cz < 0 || cx >= self.resolution as i64 || cz >= self.resolution as i64 {
                        continue;
                    }
                    let k = ((g[1] - cz * across) * LAKE_CORNERS_ACROSS as i64 + (g[0] - cx * across)) as usize;
                    let entry = heights.entry((cz as usize * self.resolution + cx as usize) as u32).or_insert([NO_LAKE; LAKE_CORNERS]);
                    entry[k] = entry[k].max(height);
                }
            }
        }
        let lake_record = |(index, heights): (u32, [f32; LAKE_CORNERS])| {
            let top = heights.iter().copied().fold(NO_LAKE, f32::max);
            let corner_depth = |h: f32| {
                let depth = ((top - h) / LAKE_DEPTH_UNIT).round();
                if h <= NO_LAKE || depth >= LAKE_NO_CORNER as f32 { LAKE_NO_CORNER } else { depth as u16 }
            };
            let depths = heights.map(corner_depth);
            (index, LakeRecord { top, depths })
        };
        self.lakes = heights.into_iter().map(lake_record).collect();
    }

    /// The surface of the lakes reaching `p`, or `NO_LAKE`.
    pub fn lake(&self, p: [f32; 2]) -> f32 {
        let Some(index) = self.cell_index(p) else {
            return NO_LAKE;
        };
        let Some(record) = self.lakes.get(&(index as u32)) else {
            return NO_LAKE;
        };
        let fine = GRID_CELL / LAKE_CELLS_ACROSS as f32;
        let cell = [(index % self.resolution) as f32, (index / self.resolution) as f32];
        let local = [
            (p[0] - self.origin[0]) / fine - cell[0] * LAKE_CELLS_ACROSS as f32,
            (p[1] - self.origin[1]) / fine - cell[1] * LAKE_CELLS_ACROSS as f32,
        ];
        record.at(local)
    }

    /// The segment indices whose reach may cover `p`.
    pub fn candidates(&self, p: [f32; 2]) -> &[u32] {
        match self.cell(p) {
            Some([offset, count]) => &self.indices[offset as usize..(offset + count) as usize],
            None => &[],
        }
    }

    /// The GPU layout: header, cell table, index lists, lake records.
    pub fn gpu_words(&self, segment_count: usize) -> Vec<u32> {
        let mut words = Vec::with_capacity(GRID_HEADER_WORDS + self.cells.len() * GRID_CELL_WORDS + self.indices.len());
        words.push(self.origin[0].to_bits());
        words.push(self.origin[1].to_bits());
        words.push(GRID_CELL.to_bits());
        words.push(self.resolution as u32);
        words.push(segment_count as u32);
        words.extend_from_slice(&[0, 0, 0]);
        let base = (GRID_HEADER_WORDS + self.cells.len() * GRID_CELL_WORDS) as u32;
        let records = base + self.indices.len() as u32;
        let mut record = 0u32;
        for (index, cell) in self.cells.iter().enumerate() {
            // Offsets are into the whole word array.
            words.push(cell[0] + base);
            words.push(cell[1]);
            if self.lakes.contains_key(&(index as u32)) {
                words.push(records + LAKE_RECORD_WORDS as u32 * record);
                record += 1;
            } else {
                words.push(NO_LAKE_RECORD);
            }
        }
        words.extend_from_slice(&self.indices);
        // In cell order, as the offsets above were handed out.
        for lake in self.lakes.values() {
            words.push(lake.top.to_bits());
            for pair in lake.depths.chunks(2) {
                words.push(pair[0] as u32 | (pair.get(1).copied().unwrap_or(LAKE_NO_CORNER) as u32) << 16);
            }
        }
        words
    }
}

/// How far behind the nearest channel, in the `combine` score (metres past
/// a waterline in that channel's widths), another channel's water still
/// counts: each weighs `exp(-behind / OWNER_BLEND)`.
pub const OWNER_BLEND: f32 = 0.25;

/// The water a point belongs to, blended over every channel near it by how
/// far behind the nearest each lies: where two channels' waters meet (a creek
/// running into its river), the water level, current, width, whitewater and
/// bend under them hand over smoothly instead of switching along a line,
/// where the bed would change from the creek's gravel to the river's rock.
/// Within one channel its neighbouring segments agree, so blending them
/// changes nothing. An online softmax, so one pass serves; mirrored by
/// `RiverOwnerBlend` in river-functions.wgslinc.
#[derive(Clone, Copy, Debug)]
pub struct OwnerBlend {
    best: f32,
    weight: f32,
    water: f32,
    velocity: [f32; 2],
    half_width: f32,
    turbulence: f32,
    bend: f32,
}

impl OwnerBlend {
    pub const NONE: OwnerBlend = OwnerBlend { best: 0.0, weight: 0.0, water: 0.0, velocity: [0.0; 2], half_width: 0.0, turbulence: 0.0, bend: 0.0 };

    pub fn add(&mut self, next: &Envelope) {
        if next.is_none() {
            return;
        }
        let score = next.bank_distance / next.half_width.clamp(0.5, 4.0);
        let k = if self.weight == 0.0 {
            self.best = score;
            1.0
        } else if score < self.best {
            // A nearer channel: what was gathered so far falls behind it.
            let fade = (-(self.best - score) / OWNER_BLEND).exp();
            self.weight *= fade;
            self.water *= fade;
            self.velocity = [self.velocity[0] * fade, self.velocity[1] * fade];
            self.half_width *= fade;
            self.turbulence *= fade;
            self.bend *= fade;
            self.best = score;
            1.0
        } else {
            (-(score - self.best) / OWNER_BLEND).exp()
        };
        self.weight += k;
        self.water += k * next.water;
        self.velocity = [self.velocity[0] + k * next.velocity[0], self.velocity[1] + k * next.velocity[1]];
        self.half_width += k * next.half_width;
        self.turbulence += k * next.turbulence;
        self.bend += k * next.bend;
    }

    /// The blend in place of the nearest channel's own values.
    pub fn apply(&self, total: &mut Envelope) {
        if self.weight <= 0.0 {
            return;
        }
        let inverse = 1.0 / self.weight;
        total.water = self.water * inverse;
        total.velocity = [self.velocity[0] * inverse, self.velocity[1] * inverse];
        total.half_width = self.half_width * inverse;
        total.turbulence = self.turbulence * inverse;
        total.bend = self.bend * inverse;
    }
}

/// The envelope of every segment near `p`.
pub fn envelope_at(segments: &[RiverSegment], grid: &SegmentGrid, p: [f32; 2]) -> Envelope {
    let mut total = Envelope::NONE;
    let mut blend = OwnerBlend::NONE;
    let candidates = grid.candidates(p);
    let mut primary = None;
    for &index in candidates.iter().take(MAX_CANDIDATES) {
        let envelope = segment_envelope(&segments[index as usize], p);
        if envelope.upper < total.upper {
            primary = Some(index as usize);
        }
        combine(&mut total, &envelope);
        blend.add(&envelope);
    }
    blend.apply(&mut total);
    round_bank_union(&mut total, segments, candidates, p, primary);
    let lake = grid.lake(p);
    if lake > NO_LAKE {
        total.lake = lake;
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    fn straight(water: [f32; 2]) -> RiverSegment {
        RiverSegment {
            a: [0.0, 0.0],
            b: [10.0, 0.0],
            water,
            half_width: [2.0, 2.0],
            depth: [0.5, 0.5],
            speed: [1.0, 1.0],
            bank: [0.8; 2],
            skew: [0.0; 2],
            levee: [1.0; 2],
            ..Default::default()
        }
        .alone()
    }

    fn crossing_banks() -> [RiverSegment; 2] {
        let mut along_x = straight([10.0, 10.0]);
        along_x.b = [40.0, 0.0];
        let mut along_z = along_x;
        along_z.b = [0.0, 40.0];
        [along_x.alone(), along_z.alone()]
    }

    fn raw_envelope(segments: &[RiverSegment], p: [f32; 2]) -> Envelope {
        let mut total = Envelope::NONE;
        for segment in segments {
            let envelope = segment_envelope(segment, p);
            combine(&mut total, &envelope);
        }
        total
    }

    #[test]
    fn intersecting_banks_round_the_ridge_between_channels() {
        let segments = crossing_banks();
        let grid = SegmentGrid::build([-32.0, -32.0], 4, &segments);
        let epsilon = 0.01;
        let at = |t| envelope_at(&segments, &grid, [6.0 + t, 6.0 - t]).upper;
        let centre = at(0.0);
        let raw = raw_envelope(&segments, [6.0, 6.0]).upper;
        assert!(centre < raw - 0.4, "{centre} {raw}");
        let left_slope = (centre - at(-epsilon)) / epsilon;
        let right_slope = (at(epsilon) - centre) / epsilon;
        assert!((left_slope - right_slope).abs() < 0.08, "bank cusp: {left_slope} -> {right_slope}");
        let raw_left = raw_envelope(&segments, [6.0 - epsilon, 6.0 + epsilon]).upper;
        assert!((raw - raw_left) / epsilon > 2.0, "unrounded bank should have a ridge");
    }

    #[test]
    fn bank_union_does_not_deepen_when_reaches_are_duplicated_or_subdivided() {
        let original = crossing_banks();
        let duplicate: Vec<_> = original.into_iter().cycle().take(32).collect();
        let divided: Vec<_> = original.iter().flat_map(|segment| {
            (0..8).map(move |i| {
                let mut part = *segment;
                let point = |t| [mix(segment.a[0], segment.b[0], t), mix(segment.a[1], segment.b[1], t)];
                part.a = point(i as f32 / 8.0);
                part.b = point((i + 1) as f32 / 8.0);
                part.alone()
            })
        }).collect();
        let grids = [
            SegmentGrid::build([-32.0, -32.0], 4, &original),
            SegmentGrid::build([-32.0, -32.0], 4, &duplicate),
            SegmentGrid::build([-32.0, -32.0], 4, &divided),
        ];
        for x in 18..70 {
            for z in 18..70 {
                let p = [x as f32 * 0.1 + 0.037, z as f32 * 0.1 + 0.019];
                let expected = envelope_at(&original, &grids[0], p).upper;
                let repeated = envelope_at(&duplicate, &grids[1], p).upper;
                let split = envelope_at(&divided, &grids[2], p).upper;
                assert!((expected - repeated).abs() < 1e-5, "duplicate at {p:?}: {expected} {repeated}");
                assert!((expected - split).abs() < 1e-5, "subdivision at {p:?}: {expected} {split}");
                let original_bank = envelope_at(&original, &grids[0], p).bank_distance;
                let divided_bank = envelope_at(&divided, &grids[2], p).bank_distance;
                assert!((original_bank - divided_bank).abs() < 1e-5, "shelf subdivision at {p:?}: {original_bank} {divided_bank}");
            }
        }
        // A straight reach's bank is unchanged even when its segments overlap.
        let parallel = vec![original[0]; 32];
        let grid = SegmentGrid::build([-32.0, -32.0], 4, &parallel);
        let p = [6.0, 6.0];
        assert_eq!(envelope_at(&parallel, &grid, p).upper, segment_envelope(&original[0], p).upper);
    }

    #[test]
    fn bank_rounding_preserves_deep_beds_and_bounds_the_new_wetted_shelf() {
        for second_water in [10.0, 10.25] {
            let mut segments = crossing_banks();
            segments[1].water = [second_water; 2];
            let grid = SegmentGrid::build([-32.0, -32.0], 4, &segments);
            for x in 0..100 {
                for z in 0..100 {
                    let p = [x as f32 * 0.1, z as f32 * 0.1];
                    let raw = raw_envelope(&segments, p);
                    let rounded = envelope_at(&segments, &grid, p);
                    if raw.bank_distance < 0.0 {
                        assert_eq!(rounded.water, raw.water, "flow water ownership at {p:?}");
                    }
                    if raw.bank_distance <= -BANK_UNION_SHORE_BLEND {
                        assert_eq!(rounded.upper, raw.upper, "bed at {p:?}");
                        assert_eq!(rounded.bank_distance, raw.bank_distance, "bed distance at {p:?}");
                    } else {
                        let first = segment_envelope(&segments[0], p);
                        let second = segment_envelope(&segments[1], p);
                        let water_level = first.water.max(second.water);
                        let nearest_bank = first.bank_distance.min(second.bank_distance);
                        if raw.upper >= water_level && rounded.upper < water_level {
                            assert!(nearest_bank <= BANK_UNION_INTRUSION + 1e-5, "shelf too wide at {p:?}: {rounded:?}");
                            assert!(rounded.bank_distance < 0.0, "wetted shelf unmarked at {p:?}: {rounded:?}");
                            assert!(rounded.water_over(rounded.upper).is_some(), "shelf missing water at {p:?}");
                        }
                        assert!(rounded.upper >= raw.upper - BANK_UNION_ROUNDING * 0.25 - 1e-5);
                    }
                    assert_eq!(rounded.lower, raw.lower);
                }
            }
        }
    }

    #[test]
    fn intersecting_waterlines_open_a_rounded_shore_contour() {
        let segments = crossing_banks();
        let grid = SegmentGrid::build([-32.0, -32.0], 4, &segments);
        // The old corner was exactly (2, 2). A shallow shelf now covers it,
        // while its shoreline bows out within the supported water ribbon.
        let corner = envelope_at(&segments, &grid, [2.0, 2.0]);
        assert!(corner.upper < 9.8 && corner.bank_distance < 0.0, "{corner:?}");
        let shoreline_height = |x, z| envelope_at(&segments, &grid, [x, z]).upper - 10.0;
        let shoreline_z = |x| {
            let mut lower = 2.0;
            let mut upper = 5.0;
            for _ in 0..24 {
                let middle = 0.5 * (lower + upper);
                if shoreline_height(x, middle) < 0.0 {
                    lower = middle;
                } else {
                    upper = middle;
                }
            }
            0.5 * (lower + upper)
        };
        let mut lower = 2.0;
        let mut upper = 2.0 + BANK_UNION_INTRUSION;
        for _ in 0..24 {
            let middle = 0.5 * (lower + upper);
            if shoreline_height(middle, middle) < 0.0 {
                lower = middle;
            } else {
                upper = middle;
            }
        }
        let centre_x = 0.5 * (lower + upper);
        let centre_z = shoreline_z(centre_x);
        assert!((centre_z - centre_x).abs() < 0.002, "shoreline {centre_x} {centre_z}");
        assert!(centre_x > 2.45 && centre_x < 2.6);
        let epsilon = 0.01;
        let left_slope = (centre_z - shoreline_z(centre_x - epsilon)) / epsilon;
        let right_slope = (shoreline_z(centre_x + epsilon) - centre_z) / epsilon;
        assert!((left_slope - right_slope).abs() < 0.10, "shoreline cusp: {left_slope} -> {right_slope}");
        assert!(left_slope < -0.8 && right_slope < -0.8, "shoreline has no rounded arc: {left_slope} {right_slope}");
        assert!(shoreline_height(2.7, 2.7) > 0.0);
        assert!(shoreline_height(2.3, 2.3) < 0.0);
    }

    #[test]
    fn channel_bed_lies_below_the_water_and_meets_it_at_the_banks() {
        let segment = straight([10.0, 10.0]);
        let centre = segment_envelope(&segment, [5.0, 0.0]);
        assert!((centre.upper - 9.5).abs() < 1e-4, "{centre:?}");
        let bank = segment_envelope(&segment, [5.0, 2.0]);
        assert!((bank.upper - 10.0).abs() < 1e-4, "{bank:?}");
        assert!(bank.bank_distance.abs() < 1e-4);
        // Just past the waterline the ground is held above the water and
        // below the bank cone; further out the levee falls back away.
        let outside = segment_envelope(&segment, [5.0, 3.0]);
        assert!(outside.lower > 10.0 && outside.lower <= outside.upper, "{outside:?}");
        assert!((outside.upper - (10.0 + 0.8 + BANK_CURVE)).abs() < 1e-4);
        let beyond = segment_envelope(&segment, [5.0, 6.0]);
        assert!(beyond.lower < outside.lower);
        assert!(segment_envelope(&segment, [5.0, 2.0 + BANK_REACH + 0.5]).is_none());
        // Far away the segment says nothing.
        assert!(segment_envelope(&segment, [5.0, 2.0 + BANK_REACH + 0.5]).is_none());
    }

    /// Neighbouring segments meet at their node with the same bank, skew
    /// and levee, and each one's caps carry the water of the reach beside,
    /// so the carved ground has no step at a joint, however much the banks
    /// change along the river or however fast its water falls.
    #[test]
    fn a_bend_down_a_rapid_carves_one_continuous_bank() {
        let turn = 20f32.to_radians();
        let nodes = [[0.0, 0.0], [10.0, 0.0], [10.0 + 10.0 * turn.cos(), 10.0 * turn.sin()]];
        let water = [10.0, 7.0, 4.0];
        let (bank, skew, levee) = ([0.9, 0.5, 0.3], [0.3, -0.2, 0.1], [1.0, 0.6, 0.2]);
        let fall = |k: usize| (water[k] - water[k + 1]) / 10.0;
        let segment = |k: usize, cap_slope: [f32; 2]| RiverSegment {
            a: nodes[k],
            b: nodes[k + 1],
            water: [water[k], water[k + 1]],
            half_width: [2.0; 2],
            depth: [0.5; 2],
            speed: [1.0; 2],
            bank: [bank[k], bank[k + 1]],
            skew: [skew[k], skew[k + 1]],
            levee: [levee[k], levee[k + 1]],
            cap_slope,
            ..Default::default()
        };
        let segments = [segment(0, [FLAT_START, fall(1)]), segment(1, [fall(0), fall(1)])];
        let grid = SegmentGrid::build([-32.0, -32.0], 4, &segments);
        // High ground the banks are cut into, and ground just under the
        // water, which the levees hold up beside the channel. With a bank,
        // skew and levee constant along each segment and a square start,
        // the banks here stepped by 1.5 m at the node; no slope the carve
        // makes is steeper than 4.
        for (ground, banks_only) in [(13.0, false), (9.5, true)] {
            let envelope = |x: f32, z: f32| envelope_at(&segments, &grid, [x, z]);
            let height = |x: f32, z: f32| envelope(x, z).clamp(ground - 0.3 * x);
            let step = 0.05;
            for i in 0..200 {
                for j in 0..560 {
                    let (x, z) = (5.0 + i as f32 * step, -14.0 + j as f32 * step);
                    for (dx, dz) in [(step, 0.0), (0.0, step)] {
                        let wet = |x: f32, z: f32| envelope(x, z).bank_distance < if banks_only { 0.25 } else { 0.0 };
                        if wet(x, z) != wet(x + dx, z + dz) || (banks_only && wet(x, z)) {
                            continue;
                        }
                        let jump = (height(x + dx, z + dz) - height(x, z)).abs();
                        assert!(jump < 4.0 * step, "step of {jump} m at {x}, {z} over ground {ground}");
                    }
                }
            }
        }
    }

    #[test]
    fn a_segment_never_reaches_back_past_its_start() {
        let segment = straight([10.0, 9.0]);
        assert!(segment_envelope(&segment, [-0.5, 0.0]).is_none());
        // ...but its round end covers the outside of the next bend.
        let ahead = segment_envelope(&segment, [11.0, 0.5]);
        assert!(!ahead.is_none() && ahead.upper < 9.0);
    }

    #[test]
    fn the_outer_bank_of_a_bend_is_deep_and_steep() {
        let mut segment = straight([10.0, 10.0]);
        segment.skew = [0.6; 2];
        let left = segment_envelope(&segment, [5.0, 1.0]);
        let right = segment_envelope(&segment, [5.0, -1.0]);
        assert!(left.upper < right.upper, "{left:?} {right:?}");
        let left_bank = segment_envelope(&segment, [5.0, 4.0]);
        let right_bank = segment_envelope(&segment, [5.0, -4.0]);
        assert!(left_bank.upper > right_bank.upper);
    }

    #[test]
    fn shelving_junction_banks_have_broad_gentle_shoulders() {
        let ordinary = straight([10.0, 10.0]);
        let mut junction = ordinary;
        junction.bank = [0.2; 2];
        junction.levee = [0.0; 2];
        // Across a four-metre shore the low junction rises less than half
        // as far as the ordinary bank, without turning into a steep wall.
        let shore = [5.0, 6.0];
        let ordinary_rise = segment_envelope(&ordinary, shore).upper - 10.0;
        let junction_rise = segment_envelope(&junction, shore).upper - 10.0;
        assert!(junction_rise > 0.0 && junction_rise < ordinary_rise * 0.5);
        let distant = [5.0, 2.0 + BANK_REACH + 2.0];
        assert!(segment_envelope(&ordinary, distant).is_none());
        assert!(!segment_envelope(&junction, distant).is_none());
        let grid = SegmentGrid::build([-32.0, -32.0], 4, &[junction]);
        assert_eq!(envelope_at(&[junction], &grid, distant), segment_envelope(&junction, distant));
    }

    #[test]
    fn carve_returns_to_high_and_low_ground_before_its_support_ends() {
        for bank in [0.2, 0.8] {
            let mut segment = straight([10.0, 10.0]);
            segment.bank = [bank; 2];
            let edge = segment.reach();
            for ground in [-30.0, 100.0] {
                // Both sides of the lookup boundary are exactly the natural
                // terrain, including its slope: no vertical cut or levee step.
                for offset in [-0.1, -0.01, 0.01, 0.1] {
                    let natural = ground + offset * 0.2;
                    let envelope = segment_envelope(&segment, [5.0, edge + offset]);
                    assert!((envelope.clamp(natural) - natural).abs() < 1e-5, "{bank} {ground} {offset}: {envelope:?}");
                }
            }
        }
    }

    #[test]
    fn levees_retreat_gradually_as_the_channel_meets_still_water() {
        let mut segment = straight([10.0, 10.0]);
        let p = [5.0, 2.8];
        let ground = 9.0;
        let full = segment_envelope(&segment, p).clamp(ground);
        segment.levee = [0.99; 2];
        let almost_full = segment_envelope(&segment, p).clamp(ground);
        assert!((full - almost_full).abs() < 0.002, "{full} {almost_full}");
        segment.levee = [0.5; 2];
        let halfway = segment_envelope(&segment, p).clamp(ground);
        assert!(halfway > ground + 0.2 && halfway < full - 0.2, "{ground} {halfway} {full}");
        let mut previous = full;
        for i in (0..100).rev() {
            segment.levee = [i as f32 / 100.0; 2];
            let height = segment_envelope(&segment, p).clamp(ground);
            assert!(height <= previous && previous - height < 0.08, "{i}: {previous} -> {height}");
            previous = height;
        }
        assert_eq!(previous, ground);
    }

    #[test]
    fn broader_junction_shoulders_fit_the_segment_lookup_budget() {
        let noise = crate::noise::NoiseField::new();
        let network = crate::rivers::network::generate(&noise, [0, 0]);
        let busiest = network.grid.cells.iter().map(|cell| cell[1] as usize).max().unwrap_or(0);
        assert!(busiest <= MAX_CANDIDATES, "junction support needs {busiest} candidates, lookup handles {MAX_CANDIDATES}");
    }

    #[test]
    fn grid_finds_every_segment_that_reaches_a_point() {
        let offset_segment = |i: i32| {
            let mut s = straight([10.0, 10.0]);
            s.a = [i as f32 * 10.0, 3.0 * i as f32];
            s.b = [i as f32 * 10.0 + 10.0, 3.0 * i as f32 + 3.0];
            s.alone()
        };
        let segments: Vec<RiverSegment> = (0..20).map(offset_segment).collect();
        let grid = SegmentGrid::build([-100.0, -100.0], 16, &segments);
        let all: Vec<u32> = (0..segments.len() as u32).collect();
        for z in -60..120 {
            for x in -60..260 {
                let p = [x as f32 * 1.3, z as f32 * 1.1];
                let mut brute = Envelope::NONE;
                let mut primary = None;
                for (index, segment) in segments.iter().enumerate() {
                    let next = segment_envelope(segment, p);
                    if next.upper < brute.upper {
                        primary = Some(index);
                    }
                    combine(&mut brute, &next);
                }
                round_bank_union(&mut brute, &segments, &all, p, primary);
                let fast = envelope_at(&segments, &grid, p);
                assert_eq!(brute.upper, fast.upper);
                assert_eq!(brute.lower, fast.lower);
            }
        }
        // The GPU words point at the same lists.
        let words = grid.gpu_words(segments.len());
        let cell = 5 * 16 + 7;
        let offset = words[GRID_HEADER_WORDS + cell * GRID_CELL_WORDS] as usize;
        let count = words[GRID_HEADER_WORDS + cell * GRID_CELL_WORDS + 1] as usize;
        assert_eq!(&words[offset..offset + count], &grid.indices[grid.cells[cell][0] as usize..][..count]);
    }
}

#[cfg(test)]
mod gpu_tests {
    use super::NO_LAKE;
    use bevy::tasks::block_on;
    use wgpu::util::DeviceExt;

    /// The WGSL river block must carve exactly what the CPU carves: the
    /// player walks on the CPU's ground and sees the GPU's.
    /// Run on a GPU (or lavapipe): cargo test river_block_matches_cpu -- --ignored
    #[test]
    #[ignore = "requires a GPU adapter"]
    fn river_block_matches_cpu() {
        let noise = crate::noise::NoiseField::new();
        let network = crate::rivers::network::generate(&noise, [0, 0]);
        let payload = network.gpu_payload();
        let instance = wgpu::Instance::default();
        let adapter = block_on(instance.request_adapter(&Default::default())).expect("adapter");
        let (device, queue) = block_on(adapter.request_device(&Default::default())).expect("device");
        let block = include_str!("../../assets/shaders/river-functions.wgslinc");
        let source = format!(
            "@group(0) @binding(0) var<storage, read> river_grid: array<u32>;
             @group(0) @binding(1) var<storage, read> river_segments: array<RiverSegment>;
             @group(0) @binding(2) var<storage, read_write> samples: array<vec4<f32>>;
             {block}
             @compute @workgroup_size(64)
             fn evaluate(@builtin(global_invocation_id) id: vec3<u32>) {{
                 if (id.x >= arrayLength(&samples)) {{ return; }}
                 let p = samples[id.x].xy;
                 let e = riverEnvelope(p);
                 samples[id.x] = vec4<f32>(e.upper, e.lower, e.bank_distance, e.lake);
             }}"
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("river block"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader,
            entry_point: Some("evaluate"),
            compilation_options: Default::default(),
            cache: None,
        });
        // Points along and across the channels near spawn.
        let mut points = Vec::new();
        let channel_nodes = network.rivers.iter().flat_map(|r| &r.nodes);
        let sampled_nodes = channel_nodes.step_by(7).take(4000);
        for node in sampled_nodes {
            for offset in [-6.0f32, -2.0, -0.7, 0.0, 0.4, 1.3, 3.0, 9.0] {
                points.push([node.position[0] + offset, node.position[1] - offset * 0.5]);
            }
        }
        // Also cover the broader junction shoulders and their return to
        // untouched terrain, on both sides of the finite support boundary.
        for segment in payload.segments.iter().step_by(19).take(2000) {
            let d = [segment.b[0] - segment.a[0], segment.b[1] - segment.a[1]];
            let length = d[0].hypot(d[1]).max(1e-3);
            let centre = [0.5 * (segment.a[0] + segment.b[0]), 0.5 * (segment.a[1] + segment.b[1])];
            let width = 0.5 * (segment.half_width[0] + segment.half_width[1]);
            for fraction in [0.45, 0.75, 0.98, 1.02] {
                let offset = width + super::BANK_REACH * super::bank_run(segment.bank[0].min(segment.bank[1])) * fraction;
                points.push([centre[0] - d[1] / length * offset, centre[1] + d[0] / length * offset]);
            }
        }
        // ...and over the lakes, their shores and the surface past them,
        // off the cells' corners and diagonals.
        let cell = crate::rivers::network::FLOW_CELL as f32;
        for lake in &network.lakes {
            for c in lake.cells.iter().step_by(37).take(20) {
                points.push([(c[0] as f32 + 0.5) * cell, (c[1] as f32 + 0.5) * cell]);
            }
            for c in lake.shore.iter().step_by(11).take(40) {
                for (u, v) in [(0.3, 0.7), (0.71, 0.29), (0.5, 0.5), (-2.4, 0.2), (3.1, -1.7)] {
                    points.push([(c[0] as f32 + u) * cell, (c[1] as f32 + v) * cell]);
                }
            }
        }
        let samples: Vec<[f32; 4]> = points.iter().map(|p| [p[0], p[1], 0.0, 0.0]).collect();
        let grid = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&payload.grid_words),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let segments = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&payload.segments),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let output = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&samples),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: output.size(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: grid.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: segments.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: output.as_entire_binding() },
            ],
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups((samples.len() as u32).div_ceil(64), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, output.size());
        queue.submit([encoder.finish()]);
        let (sender, receiver) = std::sync::mpsc::channel();
        readback.map_async(wgpu::MapMode::Read, .., move |result| sender.send(result).unwrap());
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        receiver.recv().unwrap().unwrap();
        let mapped = readback.get_mapped_range(..);
        let results: &[[f32; 4]] = bytemuck::cast_slice(&mapped);
        let mut inside = 0;
        let mut lakes = 0;
        for (p, gpu) in points.iter().zip(results) {
            let cpu = network.envelope(p[0], p[1]);
            let close = |a: f32, b: f32| (a.min(1e29) - b.min(1e29)).abs() <= 2e-3 * a.abs().clamp(1.0, 1e29);
            assert!(close(gpu[0], cpu.upper.min(1e30)), "upper at {p:?}: GPU {gpu:?} CPU {cpu:?}");
            assert!(close(-gpu[1], -cpu.lower.max(-1e30)), "lower at {p:?}: GPU {gpu:?} CPU {cpu:?}");
            // A lake's surface to the millimetre: the ground under it is the
            // ground the drawn water covers.
            let lake_close = (gpu[3].max(NO_LAKE) - cpu.lake.max(NO_LAKE)).abs() <= 1e-3 + 1e-6 * cpu.lake.abs().min(1e6);
            assert!(lake_close, "lake at {p:?}: GPU {gpu:?} CPU {cpu:?}");
            if cpu.bank_distance < 0.0 {
                inside += 1;
            }
            if cpu.lake > NO_LAKE {
                lakes += 1;
            }
        }
        assert!(inside > 1000, "{inside}");
        assert!(lakes > 100, "{lakes}");
    }
}
