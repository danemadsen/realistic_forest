//! Where the rivers run: drainage routing, channel extraction, the course
//! each river's water finds over the land, lakes, and the long profile, down
//! to the steps of a mountain creek.
//!
//! 1. **Routing.** The base landform is sampled on a world-aligned 32 m grid
//!    over the region plus a margin. A priority flood from the sea and the
//!    grid's edge gives every cell a downhill receiver; closed hollows drain
//!    over their lowest saddle, as a lake would spill. Contributing area then
//!    accumulates down the receiver tree.
//! 2. **Channels.** Cells draining more than [`CHANNEL_AREA`] are channel.
//!    Each channel head is traced downstream; at a confluence the larger
//!    branch keeps the name and the smaller one ends on it, so the network
//!    is a tree of rivers, each ending in the sea or on its parent.
//! 3. **Course.** The coarse path only says which way the water goes. Its
//!    course is traced over a 4 m grid of the natural ground in a corridor
//!    around it: a priority flood from where the river leaves the corridor
//!    fills every hollow to its spill level and the water runs down the
//!    steepest descent of that filled surface, so it keeps to the valley
//!    floor and bends where the land bends it.
//! 4. **Lakes.** A basin the water would stand deeper than [`LAKE_DEPTH`] in
//!    holds a lake, filled over its whole extent to just under the rim it
//!    spills over; the river cuts through the rims of shallower hollows.
//! 5. **Long profile.** The water surface follows the ground below its banks
//!    and only ever falls downstream, held at a lake's level through it and
//!    backed up to it above. Where the fall is steep the surface breaks into
//!    a staircase of pools and drops: step-pools on a steep creek, cascades
//!    and waterfalls where it falls off the mountainside.
//! 6. **Hydraulics.** Bankfull discharge grows with catchment. Width and
//!    depth follow downstream hydraulic geometry with the slope's terms
//!    ([`channel`]): more water makes a channel wider and deeper, a steeper
//!    slope narrower, deeper and faster; it swells and narrows through its
//!    pools and riffles, and a plunge pool is scoured out deep and wide.
//!
//! Everything is a function of world position: the grid is world-aligned and
//! every random choice is keyed by world coordinates or by a river's head
//! cell, so two regions that both contain a whole catchment agree on it.

use super::carve::{RiverSegment, SegmentGrid, GRID_CELL};
use crate::constants::SEA_LEVEL;
use crate::noise::{NoiseField, base_height};
use crate::vegetation::ecology::{cell_key, noise2 as field_noise, random};
use std::collections::BinaryHeap;

/// Side of the square the rivers are extracted from, metres.
pub const REGION_SIZE: f64 = 16384.0;
/// Region centres snap to this lattice.
pub const REGION_SNAP: f64 = 4096.0;
/// Extra routing domain around the region on every side, so the catchments
/// feeding rivers near its edge are mostly seen whole.
pub const ROUTING_MARGIN: f64 = 2048.0;
/// Routing grid cell, metres.
pub const ROUTING_CELL: f64 = 32.0;
pub const ROUTING_RESOLUTION: usize = ((REGION_SIZE + 2.0 * ROUTING_MARGIN) / ROUTING_CELL) as usize;
/// Catchment at which a channel begins on level ground, km² (rainfall
/// weighted); steep ground starts one with less, down to the minimum.
pub const CHANNEL_AREA: f32 = 0.32;
pub const MINIMUM_CHANNEL_AREA: f32 = 0.05;
/// km² per routing cell.
const CELL_AREA_KM2: f32 = (ROUTING_CELL * ROUTING_CELL / 1.0e6) as f32;
/// Slope above which a channel breaks into steps and pools.
pub const STEP_SLOPE: f32 = 0.028;
/// Gravity, m/s².
pub const GRAVITY: f32 = 9.81;
/// A hollow the water would stand deeper than this in holds a lake; the
/// river cuts through the rim of a shallower one.
pub const LAKE_DEPTH: f32 = 1.2;
/// Fine cells a lake may cover (about 2.4 km²); a basin that would take
/// more is cut through instead.
const MAX_LAKE_CELLS: usize = 150_000;

const NONE: u32 = u32::MAX;

/// One centreline node of a finished river.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RiverNode {
    pub position: [f32; 2],
    /// Water surface height.
    pub water: f32,
    /// Wetted half width.
    pub half_width: f32,
    /// Thalweg depth below the water.
    pub depth: f32,
    /// Mean velocity, m/s.
    pub speed: f32,
    /// Bankfull discharge, m³/s.
    pub discharge: f32,
    /// Catchment, km².
    pub area: f32,
    /// Bank slope past the waterline.
    pub bank: f32,
    /// Thalweg skew toward the left (+) bank.
    pub skew: f32,
    /// Whitewater, 0..1.
    pub turbulence: f32,
    /// Water-surface slope of the reach (before steps), rise over run.
    pub slope: f32,
    /// Metres downstream from the river's head.
    pub along: f32,
    /// At a waterfall's lip, the height the water drops to the next node;
    /// zero elsewhere.
    pub fall: f32,
    /// Whether the node lies in a lake, under its still water.
    pub lake: bool,
}

/// How a river ends.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RiverEnd {
    /// In the sea.
    Sea,
    /// On a larger river: (river index, node index of the confluence).
    Confluence(usize, usize),
    /// At the routing grid's edge, out of sight.
    Edge,
}

#[derive(Clone, Debug)]
pub struct River {
    pub nodes: Vec<RiverNode>,
    pub end: RiverEnd,
    /// Water drawn over nodes `0..=surface_end`: a tributary's surface stops
    /// at its parent's bank, a river's at the coast.
    pub surface_end: usize,
}

/// Still water standing in a closed basin a river runs through, up to the
/// rim it spills over.
#[derive(Clone, Debug, PartialEq)]
pub struct Lake {
    pub level: f32,
    /// The flow-grid cells under its water (cell `c` spans
    /// `c * FLOW_CELL .. (c + 1) * FLOW_CELL`).
    pub cells: Vec<[i32; 2]>,
    /// The cells around them whose ground rises through the water on every
    /// side, which its sheet may reach under so the shoreline is wherever
    /// the ground meets the water. Never one beside lower ground outside
    /// the basin (its outlet), where the sheet would hang in the air.
    pub shore: Vec<[i32; 2]>,
}

// ---------------------------------------------------------------------------
// Regions
// ---------------------------------------------------------------------------

/// The region lattice point nearest a world position.
pub fn region_of(x: f64, z: f64) -> [i64; 2] {
    [(x / REGION_SNAP).round() as i64, (z / REGION_SNAP).round() as i64]
}

pub fn region_centre(region: [i64; 2]) -> [f64; 2] {
    [region[0] as f64 * REGION_SNAP, region[1] as f64 * REGION_SNAP]
}

/// World XZ of the routing grid's minimum corner.
pub fn routing_origin(region: [i64; 2]) -> [f64; 2] {
    let centre = region_centre(region);
    let half = REGION_SIZE * 0.5 + ROUTING_MARGIN;
    [centre[0] - half, centre[1] - half]
}

// ---------------------------------------------------------------------------
// Routing
// ---------------------------------------------------------------------------

struct Routing {
    origin: [f64; 2],
    resolution: usize,
    heights: Vec<f32>,
    receiver: Vec<u32>,
    /// Runoff-weighted contributing area, cells: mountains catch more rain
    /// than the coastal plain, so each cell counts by its own rainfall.
    area: Vec<f32>,
    /// Whether a cell carries a channel.
    channel: Vec<bool>,
}

impl Routing {
    fn centre(&self, cell: usize) -> [f64; 2] {
        let x = (cell % self.resolution) as f64;
        let z = (cell / self.resolution) as f64;
        [
            self.origin[0] + (x + 0.5) * ROUTING_CELL,
            self.origin[1] + (z + 0.5) * ROUTING_CELL,
        ]
    }
}

/// Run `work(row)` for every row on all cores.
fn parallel_rows<T: Send + Default + Clone>(rows: usize, width: usize, work: impl Fn(usize, &mut [T]) + Sync) -> Vec<T> {
    let mut out = vec![T::default(); rows * width];
    let workers = std::thread::available_parallelism().map_or(4, |n| n.get()).clamp(1, 16);
    let chunk_rows = rows.div_ceil(workers);
    std::thread::scope(|scope| {
        for (chunk_index, chunk) in out.chunks_mut(chunk_rows * width).enumerate() {
            let work = &work;
            scope.spawn(move || {
                for (row_offset, row) in chunk.chunks_mut(width).enumerate() {
                    work(chunk_index * chunk_rows + row_offset, row);
                }
            });
        }
    });
    out
}

/// `work(item)` for every item on all cores, in order.
fn parallel_map<T: Sync, R: Send>(items: &[T], work: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let next = std::sync::atomic::AtomicUsize::new(0);
    let workers = std::thread::available_parallelism().map_or(4, |n| n.get()).clamp(1, 16);
    let mut done: Vec<(usize, R)> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..workers)
            .map(|_| {
                scope.spawn(|| {
                    let mut out = Vec::new();
                    loop {
                        let index = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if index >= items.len() {
                            break out;
                        }
                        out.push((index, work(&items[index])));
                    }
                })
            })
            .collect();
        handles.into_iter().flat_map(|handle| handle.join().unwrap()).collect()
    });
    done.sort_by_key(|(index, _)| *index);
    done.into_iter().map(|(_, result)| result).collect()
}

#[derive(PartialEq)]
struct Flood(f32, u32);
impl Eq for Flood {}
impl PartialOrd for Flood {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Flood {
    // A min-heap on height; ties go to the lower cell index so the flood is
    // deterministic.
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.0.total_cmp(&self.0).then_with(|| other.1.cmp(&self.1))
    }
}

const NEIGHBOURS: [(i64, i64); 8] = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)];

/// Relative rainfall at a height: the coast gets 1, the high mountains three
/// and a half times as much.
pub fn rainfall(height: f32) -> f32 {
    1.0 + 2.5 * smoothstep(15.0, 140.0, height)
}

fn route(noise: &NoiseField, region: [i64; 2]) -> Routing {
    let origin = routing_origin(region);
    let resolution = ROUTING_RESOLUTION;
    let heights: Vec<f32> = parallel_rows(resolution, resolution, |z, row: &mut [f32]| {
        let wz = origin[1] + (z as f64 + 0.5) * ROUTING_CELL;
        for (x, value) in row.iter_mut().enumerate() {
            let wx = origin[0] + (x as f64 + 0.5) * ROUTING_CELL;
            *value = base_height(noise, wx as f32, wz as f32);
        }
    });
    let count = resolution * resolution;
    let mut receiver = vec![NONE; count];
    let mut filled = heights.clone();
    let mut visited = vec![false; count];
    let mut heap = BinaryHeap::new();
    // The sea and the grid's edge are where water leaves.
    for cell in 0..count {
        let x = cell % resolution;
        let z = cell / resolution;
        let edge = x == 0 || z == 0 || x == resolution - 1 || z == resolution - 1;
        if heights[cell] < SEA_LEVEL || edge {
            visited[cell] = true;
            heap.push(Flood(heights[cell], cell as u32));
        }
    }
    let mut order: Vec<u32> = Vec::with_capacity(count);
    while let Some(Flood(level, cell)) = heap.pop() {
        order.push(cell);
        let x = (cell as usize % resolution) as i64;
        let z = (cell as usize / resolution) as i64;
        for (dx, dz) in NEIGHBOURS {
            let (nx, nz) = (x + dx, z + dz);
            if nx < 0 || nz < 0 || nx >= resolution as i64 || nz >= resolution as i64 {
                continue;
            }
            let neighbour = nz as usize * resolution + nx as usize;
            if visited[neighbour] {
                continue;
            }
            visited[neighbour] = true;
            receiver[neighbour] = cell;
            // A hollow fills to its spill level, plus a little so the
            // routing through it stays strictly downhill.
            filled[neighbour] = heights[neighbour].max(level + 1e-3);
            heap.push(Flood(filled[neighbour], neighbour as u32));
        }
    }
    // Orographic rainfall: uplands wring more water out of the weather.
    let mut area: Vec<f32> = heights.iter().map(|&h| rainfall(h)).collect();
    for &cell in order.iter().rev() {
        let r = receiver[cell as usize];
        if r != NONE {
            area[r as usize] += area[cell as usize];
        }
    }
    // Channels begin where enough water gathers, sooner on steep ground
    // (the slope-area threshold of channel initiation), and once begun they
    // run on to the sea.
    let mut channel = vec![false; count];
    for &cell in order.iter().rev() {
        let cell = cell as usize;
        let r = receiver[cell];
        if heights[cell] < SEA_LEVEL || r == NONE {
            continue;
        }
        let r = r as usize;
        let run = if (cell % resolution) != (r % resolution) && (cell / resolution) != (r / resolution) {
            ROUTING_CELL * std::f64::consts::SQRT_2
        } else {
            ROUTING_CELL
        };
        let slope = ((heights[cell] - heights[r]) as f64 / run).max(0.0) as f32;
        let threshold = (CHANNEL_AREA / (1.0 + 9.0 * slope)).max(MINIMUM_CHANNEL_AREA) / CELL_AREA_KM2;
        if area[cell] >= threshold {
            channel[cell] = true;
        }
        if channel[cell] && heights[r] >= SEA_LEVEL {
            channel[r] = true;
        }
    }
    Routing {
        origin,
        resolution,
        heights,
        receiver,
        area,
        channel,
    }
}

// ---------------------------------------------------------------------------
// Channel extraction
// ---------------------------------------------------------------------------

struct CellPath {
    cells: Vec<u32>,
    end: RiverEnd,
    /// Seed from the head's world cell.
    seed: u64,
}

fn extract_paths(routing: &Routing, region: [i64; 2]) -> Vec<CellPath> {
    let count = routing.heights.len();
    let is_channel = |cell: usize| routing.channel[cell];
    // The largest tributary of each cell keeps its name downstream.
    let mut main_donor = vec![NONE; count];
    let mut has_channel_donor = vec![false; count];
    for cell in 0..count {
        let r = routing.receiver[cell];
        if r == NONE || !is_channel(cell) {
            continue;
        }
        has_channel_donor[r as usize] = true;
        let current = main_donor[r as usize];
        let better = current == NONE
            || routing.area[cell] > routing.area[current as usize]
            || (routing.area[cell] == routing.area[current as usize] && (cell as u32) < current);
        if better {
            main_donor[r as usize] = cell as u32;
        }
    }
    let centre = region_centre(region);
    let half = REGION_SIZE * 0.5;
    let in_region = |p: [f64; 2]| (p[0] - centre[0]).abs() <= half && (p[1] - centre[1]).abs() <= half;
    // Owner path of every channel cell, to find a tributary's parent.
    let mut owner: Vec<(u32, u32)> = vec![(NONE, NONE); count];
    let mut paths: Vec<CellPath> = Vec::new();
    for (head, &fed) in has_channel_donor.iter().enumerate() {
        if !is_channel(head) || fed {
            continue;
        }
        let mut cells = vec![head as u32];
        let mut cell = head;
        let end = loop {
            let r = routing.receiver[cell];
            if r == NONE {
                break RiverEnd::Edge;
            }
            let r = r as usize;
            cells.push(r as u32);
            if routing.heights[r] < SEA_LEVEL {
                break RiverEnd::Sea;
            }
            if main_donor[r] != cell as u32 {
                // Placeholder; resolved to the owning path below.
                break RiverEnd::Confluence(usize::MAX, r);
            }
            cell = r;
        };
        let head_world = routing.centre(head);
        let seed = cell_key(
            (head_world[0] / ROUTING_CELL).floor() as i64,
            (head_world[1] / ROUTING_CELL).floor() as i64,
            0x52_1BE5,
        );
        let path_index = paths.len();
        let owned = match end {
            RiverEnd::Confluence(..) | RiverEnd::Sea => cells.len() - 1,
            RiverEnd::Edge => cells.len(),
        };
        for (index, &c) in cells.iter().take(owned).enumerate() {
            owner[c as usize] = (path_index as u32, index as u32);
        }
        paths.push(CellPath { cells, end, seed });
    }
    for path in &mut paths {
        if let RiverEnd::Confluence(_, cell) = path.end {
            let (parent, index) = owner[cell];
            path.end = if parent == NONE {
                RiverEnd::Edge
            } else {
                RiverEnd::Confluence(parent as usize, index as usize)
            };
        }
    }
    // Keep the rivers that reach into the region, and everything they flow
    // into on their way out of it.
    let mut keep = vec![false; paths.len()];
    for (index, path) in paths.iter().enumerate() {
        if path.cells.iter().any(|&c| in_region(routing.centre(c as usize))) {
            let mut current = index;
            loop {
                if keep[current] {
                    break;
                }
                keep[current] = true;
                match paths[current].end {
                    RiverEnd::Confluence(parent, _) => current = parent,
                    _ => break,
                }
            }
        }
    }
    let mut remap = vec![usize::MAX; paths.len()];
    let mut kept = Vec::new();
    for (index, path) in paths.into_iter().enumerate() {
        if keep[index] {
            remap[index] = kept.len();
            kept.push(path);
        }
    }
    for path in &mut kept {
        if let RiverEnd::Confluence(parent, node) = path.end {
            path.end = RiverEnd::Confluence(remap[parent], node);
        }
    }
    kept
}

// ---------------------------------------------------------------------------
// Polyline helpers
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Default)]
struct PathPoint {
    p: [f64; 2],
    /// Contributing area, km².
    area: f64,
    /// The level of the lake the point lies in, or -infinity.
    lake: f64,
}

fn distance(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

fn lerp_point(a: &PathPoint, b: &PathPoint, t: f64) -> PathPoint {
    PathPoint {
        p: [a.p[0] + (b.p[0] - a.p[0]) * t, a.p[1] + (b.p[1] - a.p[1]) * t],
        area: a.area + (b.area - a.area) * t,
        lake: if t < 0.5 { a.lake } else { b.lake },
    }
}

/// Resample at a spacing that may vary along the path; keeps both ends.
fn resample(points: &[PathPoint], spacing: impl Fn(&PathPoint) -> f64) -> Vec<PathPoint> {
    if points.len() < 2 {
        return points.to_vec();
    }
    let mut out = vec![points[0]];
    let mut segment = 0;
    let mut t = 0.0;
    loop {
        let current = *out.last().unwrap();
        let mut remaining = spacing(&current).max(0.25);
        // Walk `remaining` metres along the polyline from (segment, t).
        loop {
            let a = &points[segment];
            let b = &points[segment + 1];
            let length = distance(a.p, b.p).max(1e-9);
            let left = length * (1.0 - t);
            if remaining <= left {
                t += remaining / length;
                out.push(lerp_point(a, b, t));
                break;
            }
            remaining -= left;
            segment += 1;
            t = 0.0;
            if segment + 1 >= points.len() {
                let last = *points.last().unwrap();
                // Merge a too-short final step into the end point.
                if distance(out.last().unwrap().p, last.p) < spacing(&last) * 0.4 && out.len() > 1 {
                    out.pop();
                }
                out.push(last);
                return out;
            }
        }
    }
}

/// Gaussian smoothing of positions with a kernel of `sigma` points; the ends
/// stay put and the smoothing fades in away from them.
fn smooth(points: &mut [PathPoint], sigma: f64) {
    let n = points.len();
    if n < 3 || sigma <= 0.0 {
        return;
    }
    let radius = (sigma * 3.0).ceil() as i64;
    let weights: Vec<f64> = (0..=radius).map(|k| (-0.5 * (k as f64 / sigma).powi(2)).exp()).collect();
    let source: Vec<[f64; 2]> = points.iter().map(|p| p.p).collect();
    for i in 1..n - 1 {
        let mut sum = [0.0, 0.0];
        let mut total = 0.0;
        // A symmetric window that shrinks near the ends keeps them unbiased.
        let reach = radius.min(i as i64).min((n - 1 - i) as i64);
        for k in -reach..=reach {
            let w = weights[k.unsigned_abs() as usize];
            let q = source[(i as i64 + k) as usize];
            sum[0] += q[0] * w;
            sum[1] += q[1] * w;
            total += w;
        }
        points[i].p = [sum[0] / total, sum[1] / total];
    }
}

fn arc_lengths(points: &[PathPoint]) -> Vec<f64> {
    let mut out = Vec::with_capacity(points.len());
    let mut s = 0.0;
    for (i, point) in points.iter().enumerate() {
        if i > 0 {
            s += distance(points[i - 1].p, point.p);
        }
        out.push(s);
    }
    out
}

/// Unit tangent at every point (central differences).
fn tangents(points: &[PathPoint]) -> Vec<[f64; 2]> {
    let n = points.len();
    (0..n)
        .map(|i| {
            let a = points[i.saturating_sub(1)].p;
            let b = points[(i + 1).min(n - 1)].p;
            let d = [b[0] - a[0], b[1] - a[1]];
            let length = d[0].hypot(d[1]).max(1e-9);
            [d[0] / length, d[1] / length]
        })
        .collect()
}

/// Smooth a scalar sequence with a Gaussian of `sigma` samples.
fn smooth_values(values: &[f64], sigma: f64) -> Vec<f64> {
    let n = values.len();
    if n < 3 || sigma <= 0.0 {
        return values.to_vec();
    }
    let radius = (sigma * 3.0).ceil() as i64;
    (0..n)
        .map(|i| {
            let mut sum = 0.0;
            let mut total = 0.0;
            for k in -radius..=radius {
                let j = (i as i64 + k).clamp(0, n as i64 - 1) as usize;
                let w = (-0.5 * (k as f64 / sigma).powi(2)).exp();
                sum += values[j] * w;
                total += w;
            }
            sum / total
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Hydraulic geometry
// ---------------------------------------------------------------------------

/// Bankfull discharge of a catchment, m³/s.
pub fn discharge(area_km2: f32) -> f32 {
    1.4 * area_km2.max(0.01).powf(0.8)
}

/// A typical wetted half width for a catchment, metres, for spacing work
/// done before the slope is known; [`channel`] gives the real one.
pub fn half_width(area_km2: f32) -> f32 {
    0.5 * (1.0 + 3.2 * area_km2.max(0.0).sqrt())
}

/// The bankfull channel a discharge cuts on a slope: (half width, thalweg
/// depth, mean speed). Downstream hydraulic geometry, the power laws real
/// rivers follow (width ~ Q^0.5, depth ~ Q^0.35, so speed ~ Q^0.15), with
/// the slope's own terms: on a steep slope a stream is held in a narrow,
/// deep slot between boulders and bedrock (width ~ S^-0.35, depth ~ S^0.1);
/// on the flat it spreads wide and shallow over its own gravel and silt.
/// The speed is what carries the discharge through that section, so it
/// rises with slope too. A parabolic section's thalweg is half as deep again
/// as its mean.
pub fn channel(discharge: f32, slope: f32) -> (f32, f32, f32) {
    let q = discharge.max(1e-3);
    let s = (slope / 0.01).clamp(0.08, 30.0);
    let width = 5.0 * q.powf(0.5) * s.powf(-0.35);
    let mean_depth = 0.3 * q.powf(0.35) * s.powf(0.1);
    let half_width = (0.5 * width).max(0.55);
    let speed = q / (2.0 * half_width * mean_depth);
    (half_width, (1.5 * mean_depth).max(0.15), speed.clamp(0.15, 6.0))
}

fn smoothstep(edge0: f32, edge1: f32, value: f32) -> f32 {
    let t = ((value - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

// ---------------------------------------------------------------------------
// Flow paths
// ---------------------------------------------------------------------------

/// Cell of the fine flow grids, metres.
pub const FLOW_CELL: f64 = 4.0;
/// How far either side of its coarse path a river's water may find its way.
pub const FLOW_CORRIDOR: f64 = 72.0;
/// Rise per metre the flood adds across a filled hollow, so water crossing
/// it still runs downhill, toward the spill.
const FLOW_TILT: f32 = 1.0e-4;
/// Side of a corridor block, cells.
const BLOCK: usize = 32;

/// The natural ground in a corridor around a coarse path, on the fine flow
/// grid. Storage comes in 32-cell blocks so a long winding corridor costs
/// only the ground it covers.
struct Corridor {
    /// Cell coordinates of the bounding box's first cell; the grid itself
    /// is world-aligned, so every corridor samples the same ground.
    first: [i64; 2],
    blocks: [usize; 2],
    /// Storage slot of each block of the bounding box.
    slot: Vec<u32>,
    /// Ground height per stored cell; NaN outside the corridor.
    ground: Vec<f32>,
}

impl Corridor {
    fn cell_of(p: [f64; 2]) -> [i64; 2] {
        [(p[0] / FLOW_CELL).floor() as i64, (p[1] / FLOW_CELL).floor() as i64]
    }

    fn centre(cell: [i64; 2]) -> [f64; 2] {
        [(cell[0] as f64 + 0.5) * FLOW_CELL, (cell[1] as f64 + 0.5) * FLOW_CELL]
    }

    /// Storage index of a cell, if its block is stored.
    fn index(&self, cell: [i64; 2]) -> Option<usize> {
        let x = cell[0] - self.first[0];
        let z = cell[1] - self.first[1];
        if x < 0 || z < 0 {
            return None;
        }
        let (x, z) = (x as usize, z as usize);
        let (bx, bz) = (x / BLOCK, z / BLOCK);
        if bx >= self.blocks[0] || bz >= self.blocks[1] {
            return None;
        }
        let slot = self.slot[bz * self.blocks[0] + bx];
        (slot != NONE).then(|| slot as usize * BLOCK * BLOCK + (z % BLOCK) * BLOCK + x % BLOCK)
    }

    /// Storage index of a cell inside the corridor.
    fn inside(&self, cell: [i64; 2]) -> Option<usize> {
        self.index(cell).filter(|&i| !self.ground[i].is_nan())
    }

    /// The ground within `reach` of the path through `points`, and over
    /// any `extra` cells.
    fn new(noise: &NoiseField, points: &[[f64; 2]], reach: f64, extra: &[[i64; 2]]) -> Self {
        let mut minimum = [f64::INFINITY; 2];
        let mut maximum = [f64::NEG_INFINITY; 2];
        for p in points.iter().copied().chain(extra.iter().map(|&c| Self::centre(c))) {
            for axis in 0..2 {
                minimum[axis] = minimum[axis].min(p[axis] - reach);
                maximum[axis] = maximum[axis].max(p[axis] + reach);
            }
        }
        let first = Self::cell_of(minimum);
        let last = Self::cell_of(maximum);
        let blocks = [
            ((last[0] - first[0] + 1) as usize).div_ceil(BLOCK),
            ((last[1] - first[1] + 1) as usize).div_ceil(BLOCK),
        ];
        let mut corridor = Corridor {
            first,
            blocks,
            slot: vec![NONE; blocks[0] * blocks[1]],
            ground: Vec::new(),
        };
        // Mark the cells near each leg of the path, storing blocks as they
        // are first touched.
        let mut marked: Vec<usize> = Vec::new();
        let mut mark = |corridor: &mut Corridor, cell: [i64; 2]| {
            let (lx, lz) = ((cell[0] - first[0]) as usize, (cell[1] - first[1]) as usize);
            let block = (lz / BLOCK) * blocks[0] + lx / BLOCK;
            if corridor.slot[block] == NONE {
                corridor.slot[block] = (corridor.ground.len() / (BLOCK * BLOCK)) as u32;
                corridor.ground.extend(std::iter::repeat_n(f32::NAN, BLOCK * BLOCK));
            }
            let index = corridor.index(cell).unwrap();
            if corridor.ground[index].is_nan() {
                // Marked; sampled below.
                corridor.ground[index] = f32::INFINITY;
                marked.push(index);
            }
        };
        for &cell in extra {
            mark(&mut corridor, cell);
        }
        let legs = points.len().saturating_sub(1).max(1);
        for leg in 0..legs {
            let a = points[leg];
            let b = points[(leg + 1).min(points.len() - 1)];
            let low = Self::cell_of([a[0].min(b[0]) - reach, a[1].min(b[1]) - reach]);
            let high = Self::cell_of([a[0].max(b[0]) + reach, a[1].max(b[1]) + reach]);
            let ab = [b[0] - a[0], b[1] - a[1]];
            let length_squared = (ab[0] * ab[0] + ab[1] * ab[1]).max(1e-9);
            for z in low[1]..=high[1] {
                for x in low[0]..=high[0] {
                    let c = Self::centre([x, z]);
                    let t = (((c[0] - a[0]) * ab[0] + (c[1] - a[1]) * ab[1]) / length_squared).clamp(0.0, 1.0);
                    if distance(c, [a[0] + ab[0] * t, a[1] + ab[1] * t]) <= reach {
                        mark(&mut corridor, [x, z]);
                    }
                }
            }
        }
        // Sample the ground of every marked cell; a block's cells are in
        // row order, so recover each one's position from its block.
        let mut block_cell = vec![[0i64; 2]; corridor.ground.len() / (BLOCK * BLOCK)];
        for (block, &slot) in corridor.slot.iter().enumerate() {
            if slot != NONE {
                block_cell[slot as usize] = [
                    first[0] + ((block % blocks[0]) * BLOCK) as i64,
                    first[1] + ((block / blocks[0]) * BLOCK) as i64,
                ];
            }
        }
        for index in marked {
            let origin = block_cell[index / (BLOCK * BLOCK)];
            let within = index % (BLOCK * BLOCK);
            let c = Self::centre([origin[0] + (within % BLOCK) as i64, origin[1] + (within / BLOCK) as i64]);
            corridor.ground[index] = base_height(noise, c[0] as f32, c[1] as f32);
        }
        corridor
    }
}

/// Where a river's water leaves its corridor: into the sea, into the
/// channel of the river it joins, or off the routing grid.
struct Outlets {
    /// Per stored cell: the water level of an outlet there, or NaN.
    level: Vec<f32>,
}

impl Outlets {
    fn none(corridor: &Corridor) -> Self {
        Outlets { level: vec![f32::NAN; corridor.ground.len()] }
    }

    fn add(&mut self, index: usize, level: f32) {
        let current = self.level[index];
        self.level[index] = if current.is_nan() { level } else { current.min(level) };
    }

    fn any(&self) -> bool {
        self.level.iter().any(|l| !l.is_nan())
    }

    /// The parent's channel, at its water level.
    fn channel(&mut self, corridor: &Corridor, parent: &River) {
        for pair in parent.nodes.windows(2) {
            let (a, b) = (&pair[0], &pair[1]);
            let reach = a.half_width.max(b.half_width).max(FLOW_CELL as f32 * 0.6) as f64;
            let pa = [a.position[0] as f64, a.position[1] as f64];
            let pb = [b.position[0] as f64, b.position[1] as f64];
            let low = Corridor::cell_of([pa[0].min(pb[0]) - reach, pa[1].min(pb[1]) - reach]);
            let high = Corridor::cell_of([pa[0].max(pb[0]) + reach, pa[1].max(pb[1]) + reach]);
            let ab = [pb[0] - pa[0], pb[1] - pa[1]];
            let length_squared = (ab[0] * ab[0] + ab[1] * ab[1]).max(1e-9);
            for z in low[1]..=high[1] {
                for x in low[0]..=high[0] {
                    let Some(index) = corridor.inside([x, z]) else {
                        continue;
                    };
                    let c = Corridor::centre([x, z]);
                    let t = (((c[0] - pa[0]) * ab[0] + (c[1] - pa[1]) * ab[1]) / length_squared).clamp(0.0, 1.0);
                    if distance(c, [pa[0] + ab[0] * t, pa[1] + ab[1] * t]) <= reach {
                        self.add(index, a.water + (b.water - a.water) * t as f32);
                    }
                }
            }
        }
    }

    /// Every cell of the corridor under the sea.
    fn sea(&mut self, corridor: &Corridor) {
        for (index, &ground) in corridor.ground.iter().enumerate() {
            if ground < SEA_LEVEL {
                self.add(index, ground);
            }
        }
    }

    /// The cells around a point, at their own ground.
    fn around(&mut self, corridor: &Corridor, p: [f64; 2], radius: f64) {
        let low = Corridor::cell_of([p[0] - radius, p[1] - radius]);
        let high = Corridor::cell_of([p[0] + radius, p[1] + radius]);
        for z in low[1]..=high[1] {
            for x in low[0]..=high[0] {
                if let Some(index) = corridor.inside([x, z])
                    && distance(Corridor::centre([x, z]), p) <= radius
                {
                    self.add(index, corridor.ground[index]);
                }
            }
        }
    }
}

struct Flow {
    path: Vec<[f64; 2]>,
    /// The corridor's ground with its hollows filled to their spill levels.
    filled: Vec<f32>,
}

impl Flow {
    /// How far a hollow's water would stand over the ground at `p`.
    fn ponding(&self, corridor: &Corridor, p: [f64; 2]) -> f32 {
        corridor
            .inside(Corridor::cell_of(p))
            .map_or(0.0, |index| (self.filled[index] - corridor.ground[index]).max(0.0))
    }
}

/// The way water from `head` takes to an outlet over the corridor's ground:
/// a priority flood from the outlets fills every hollow to the level it
/// spills at, and the water runs down the steepest descent of that filled
/// surface (across a filled hollow, by the shortest way to its spill). The
/// path's cell centres and the filled surface.
fn flow_path(corridor: &Corridor, outlets: &Outlets, head: [f64; 2]) -> Option<Flow> {
    let count = corridor.ground.len();
    let mut filled = vec![f32::NAN; count];
    let mut heap = BinaryHeap::new();
    for (index, &level) in outlets.level.iter().enumerate() {
        if !level.is_nan() {
            filled[index] = level;
            heap.push(Flood(level, index as u32));
        }
    }
    if heap.is_empty() {
        return None;
    }
    // The flood reaches cells by storage index; their grid positions come
    // from a per-slot table.
    let mut slot_origin = vec![[0i64; 2]; count / (BLOCK * BLOCK)];
    for (block, &slot) in corridor.slot.iter().enumerate() {
        if slot != NONE {
            slot_origin[slot as usize] = [
                corridor.first[0] + ((block % corridor.blocks[0]) * BLOCK) as i64,
                corridor.first[1] + ((block / corridor.blocks[0]) * BLOCK) as i64,
            ];
        }
    }
    let cell_at = |index: usize| -> [i64; 2] {
        let origin = slot_origin[index / (BLOCK * BLOCK)];
        let within = index % (BLOCK * BLOCK);
        [origin[0] + (within % BLOCK) as i64, origin[1] + (within / BLOCK) as i64]
    };
    while let Some(Flood(level, index)) = heap.pop() {
        let cell = cell_at(index as usize);
        for (dx, dz) in NEIGHBOURS {
            let Some(neighbour) = corridor.inside([cell[0] + dx, cell[1] + dz]) else {
                continue;
            };
            if !filled[neighbour].is_nan() {
                continue;
            }
            let run = if dx != 0 && dz != 0 { FLOW_CELL * std::f64::consts::SQRT_2 } else { FLOW_CELL } as f32;
            filled[neighbour] = corridor.ground[neighbour].max(level + FLOW_TILT * run);
            heap.push(Flood(filled[neighbour], neighbour as u32));
        }
    }
    // The spring: the lowest ground near the head, where its hollow gathers.
    let head_cell = Corridor::cell_of(head);
    let mut start = None;
    let mut lowest = f32::INFINITY;
    let search = (12.0 / FLOW_CELL).ceil() as i64;
    for dz in -search..=search {
        for dx in -search..=search {
            if let Some(index) = corridor.inside([head_cell[0] + dx, head_cell[1] + dz])
                && !filled[index].is_nan()
                && corridor.ground[index] < lowest
            {
                lowest = corridor.ground[index];
                start = Some(index);
            }
        }
    }
    let mut current = start?;
    let mut path = Vec::new();
    for _ in 0..count {
        let cell = cell_at(current);
        path.push(Corridor::centre(cell));
        if !outlets.level[current].is_nan() {
            return Some(Flow { path, filled });
        }
        let mut best = None;
        let mut steepest = 0.0f32;
        for (dx, dz) in NEIGHBOURS {
            let Some(neighbour) = corridor.inside([cell[0] + dx, cell[1] + dz]) else {
                continue;
            };
            let run = if dx != 0 && dz != 0 { FLOW_CELL * std::f64::consts::SQRT_2 } else { FLOW_CELL } as f32;
            let descent = (filled[current] - filled[neighbour]) / run;
            if descent > steepest {
                steepest = descent;
                best = Some(neighbour);
            }
        }
        current = best?;
    }
    None
}

/// The lakes of a region, and which flow-grid cells lie under each.
#[derive(Default)]
struct Lakes {
    lakes: Vec<Lake>,
    cells: std::collections::HashMap<[i64; 2], u32>,
}

impl Lakes {
    fn level_at(&self, p: [f64; 2]) -> f64 {
        self.cells
            .get(&Corridor::cell_of(p))
            .map_or(f64::NEG_INFINITY, |&lake| self.lakes[lake as usize].level as f64)
    }

    /// The lake filling the basin around `start` up to `level`: every cell
    /// joined to it (edge to edge) whose ground lies below the level. A
    /// lake already holding the basin at about that level is the same lake;
    /// smaller, lower lakes inside the basin are drowned in the new one.
    fn fill(&mut self, noise: &NoiseField, corridor: &Corridor, start: [i64; 2], level: f32) -> Filled {
        if self.cells.contains_key(&start) {
            return Filled::Lake;
        }
        let mut heights: std::collections::HashMap<[i64; 2], f32> = Default::default();
        let mut ground = |cell: [i64; 2]| -> f32 {
            *heights.entry(cell).or_insert_with(|| {
                corridor.inside(cell).map_or_else(
                    || {
                        let c = Corridor::centre(cell);
                        base_height(noise, c[0] as f32, c[1] as f32)
                    },
                    |index| corridor.ground[index],
                )
            })
        };
        if ground(start) >= level {
            return Filled::NotBasin;
        }
        let mut inside = std::collections::HashSet::new();
        let mut queue = std::collections::VecDeque::new();
        let mut cells = Vec::new();
        let mut drowned = std::collections::BTreeSet::new();
        inside.insert(start);
        queue.push_back(start);
        while let Some(cell) = queue.pop_front() {
            if let Some(&other) = self.cells.get(&cell) {
                if self.lakes[other as usize].level >= level - 0.3 {
                    return Filled::Lake;
                }
                drowned.insert(other);
            }
            if cells.len() >= MAX_LAKE_CELLS {
                return Filled::Escaped;
            }
            cells.push(cell);
            for (dx, dz) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                let next = [cell[0] + dx, cell[1] + dz];
                if inside.contains(&next) {
                    continue;
                }
                let height = ground(next);
                // Water that reaches the sea was never held in a basin; what
                // the fill crossed is its way there.
                if height < SEA_LEVEL {
                    let mut way: Vec<[i64; 2]> = inside.into_iter().collect();
                    way.push(next);
                    return Filled::ToSea(way);
                }
                if height < level {
                    inside.insert(next);
                    queue.push_back(next);
                }
            }
        }
        // The shore: cells beside the water with no lower ground beyond.
        let mut shore = Vec::new();
        let mut considered = std::collections::HashSet::new();
        for &cell in &cells {
            for (dx, dz) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                let rim = [cell[0] + dx, cell[1] + dz];
                if inside.contains(&rim) || !considered.insert(rim) {
                    continue;
                }
                // Clearly above the water itself, so the erosion's few
                // centimetres here cannot leave the sheet over dry ground.
                let dry_around = ground(rim) >= level + 0.1
                    && NEIGHBOURS.iter().all(|&(ex, ez)| {
                        let beyond = [rim[0] + ex, rim[1] + ez];
                        inside.contains(&beyond) || ground(beyond) >= level
                    });
                if dry_around {
                    shore.push([rim[0] as i32, rim[1] as i32]);
                }
            }
        }
        let index = self.lakes.len() as u32;
        for &lake in &drowned {
            let lake = &mut self.lakes[lake as usize];
            lake.cells.clear();
            lake.shore.clear();
            lake.level = level;
        }
        for &cell in &cells {
            self.cells.insert(cell, index);
        }
        self.lakes.push(Lake {
            level,
            cells: cells.iter().map(|c| [c[0] as i32, c[1] as i32]).collect(),
            shore,
        });
        Filled::Lake
    }
}

/// What filling a basin came to.
enum Filled {
    Lake,
    /// The water found its way to the sea under the would-be lake's level,
    /// through these cells: a gap too narrow for the coarse routing to see.
    ToSea(Vec<[i64; 2]>),
    /// The water ran out of the basin (to the sea, or across more land than
    /// any lake covers): the corridor it was routed in was too narrow to
    /// show its real way down.
    Escaped,
    NotBasin,
}

/// Raise lakes in the deep basins along a flow path; the river cuts through
/// the rims of the shallow ones. Stops at the first "basin" that turned out
/// to drain beyond the corridor, and says how.
fn raise_lakes(noise: &NoiseField, corridor: &Corridor, flow: &Flow, lakes: &mut Lakes) -> Option<Filled> {
    let ponding: Vec<f32> = flow.path.iter().map(|&p| flow.ponding(corridor, p)).collect();
    let mut k = 0;
    while k < ponding.len() {
        if ponding[k] <= 0.0 {
            k += 1;
            continue;
        }
        let first = k;
        while k < ponding.len() && ponding[k] > 0.0 {
            k += 1;
        }
        let deepest = (first..k).max_by(|&a, &b| ponding[a].total_cmp(&ponding[b])).unwrap();
        if ponding[deepest] < LAKE_DEPTH {
            continue;
        }
        // The filled surface stands at the spill level, rising a hair away
        // from the spill; the water stands a little under the rim.
        let level = (first..k)
            .filter_map(|j| corridor.inside(Corridor::cell_of(flow.path[j])).map(|index| flow.filled[index]))
            .fold(f32::INFINITY, f32::min)
            - 0.05;
        match lakes.fill(noise, corridor, Corridor::cell_of(flow.path[deepest]), level) {
            Filled::Lake | Filled::NotBasin => {}
            escape => return Some(escape),
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Plan form
// ---------------------------------------------------------------------------

/// Pull a path onto the valley floor: each point looks across the path for
/// the lowest ground nearby and moves toward it, then the shifts are
/// smoothed so the path stays a path.
fn snap_to_floor(noise: &NoiseField, points: &mut [PathPoint], reach: f64, step: f64) {
    let n = points.len();
    if n < 3 {
        return;
    }
    let normals: Vec<[f64; 2]> = tangents(points).iter().map(|t| [-t[1], t[0]]).collect();
    let samples = (reach / step).round() as i64;
    let mut shifts = vec![0.0; n];
    for i in 1..n - 1 {
        let mut best = f64::INFINITY;
        let mut best_offset = 0.0;
        for k in -samples..=samples {
            let offset = k as f64 * step;
            let q = [points[i].p[0] + normals[i][0] * offset, points[i].p[1] + normals[i][1] * offset];
            // A mild preference for staying put avoids hopping into a
            // neighbouring valley across a low divide.
            let score = base_height(noise, q[0] as f32, q[1] as f32) as f64 + 0.03 * offset.abs();
            if score < best {
                best = score;
                best_offset = offset;
            }
        }
        shifts[i] = best_offset;
    }
    let shifts = smooth_values(&shifts, 2.0);
    for i in 1..n - 1 {
        // The ends stay where they are: a head, a confluence, the coast.
        let fade = ((i.min(n - 1 - i)) as f64 / 3.0).min(1.0);
        points[i].p[0] += normals[i][0] * shifts[i] * 0.85 * fade;
        points[i].p[1] += normals[i][1] * shifts[i] * 0.85 * fade;
    }
}

/// Relax bends tighter than a channel can turn (radius under 1.6 widths).
fn relax_tight_bends(points: &mut [PathPoint]) {
    let n = points.len();
    if n < 3 {
        return;
    }
    for _ in 0..6 {
        let mut changed = false;
        for i in 1..n - 1 {
            let a = points[i - 1].p;
            let b = points[i].p;
            let c = points[i + 1].p;
            let ab = distance(a, b);
            let bc = distance(b, c);
            let ac = distance(a, c);
            let cross = ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])).abs();
            let radius = if cross < 1e-9 { f64::INFINITY } else { ab * bc * ac / (2.0 * cross) };
            let width = 2.0 * half_width(points[i].area as f32) as f64;
            if radius < 1.6 * width {
                points[i].p = [
                    b[0] * 0.5 + (a[0] + c[0]) * 0.25,
                    b[1] * 0.5 + (a[1] + c[1]) * 0.25,
                ];
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
}

// ---------------------------------------------------------------------------
// Long profile
// ---------------------------------------------------------------------------

struct Profiled {
    points: Vec<PathPoint>,
    s: Vec<f64>,
    water: Vec<f32>,
    slope: Vec<f32>,
}

/// Water surface along a path: below the banks, never rising downstream.
fn water_profile(noise: &NoiseField, points: Vec<PathPoint>, end_level: Option<f32>) -> Profiled {
    let n = points.len();
    let s = arc_lengths(&points);
    let ground: Vec<f32> = points.iter().map(|p| base_height(noise, p.p[0] as f32, p.p[1] as f32)).collect();
    // A lake holds its water level, and backs the river above it up to it:
    // no reach may fall below the next lake downstream.
    let mut floor = vec![f32::NEG_INFINITY; n];
    let mut below = f32::NEG_INFINITY;
    for i in (0..n).rev() {
        if points[i].lake.is_finite() {
            below = points[i].lake as f32;
        }
        floor[i] = below;
    }
    let mut water = vec![0.0f32; n];
    let mut slope = vec![0.01f32; n];
    // Two passes: the bank height depends on the depth, which depends on the
    // slope of the surface the first pass found.
    for pass in 0..2 {
        for i in 0..n {
            let bank = if pass == 0 {
                0.45
            } else {
                // A stream runs a little under its banks, not in a ditch.
                let (_, depth, _) = channel(discharge(points[i].area as f32), slope[i]);
                0.12 + 0.3 * depth
            };
            let raw = ground[i] - bank;
            water[i] = if i == 0 {
                raw
            } else {
                raw.min(water[i - 1] - 4e-4 * (s[i] - s[i - 1]) as f32)
            };
            water[i] = if points[i].lake.is_finite() { points[i].lake as f32 } else { water[i].max(floor[i]) };
        }
        if let Some(level) = end_level {
            // A tributary meets its parent's surface; where it would arrive
            // lower it is held up to it (the parent backs it up).
            for w in water.iter_mut() {
                *w = w.max(level);
            }
            water[n - 1] = level;
        }
        for w in water.iter_mut() {
            *w = w.max(SEA_LEVEL + 0.02);
        }
        // Reach slope over a few channel widths.
        for i in 0..n {
            let width = 2.0 * half_width(points[i].area as f32) as f64;
            let span = (width * 4.0).max(24.0);
            let a = s.partition_point(|&v| v < s[i] - span).min(n - 1);
            let b = s.partition_point(|&v| v <= s[i] + span).saturating_sub(1).max(a);
            slope[i] = if b > a { ((water[a] - water[b]) / (s[b] - s[a]) as f32).max(4e-4) } else { 4e-4 };
        }
    }
    // Smooth the surface; a positive kernel keeps it falling downstream.
    let smoothed = smooth_values(&water.iter().map(|&w| w as f64).collect::<Vec<_>>(), 1.5);
    for i in 1..n.saturating_sub(1) {
        water[i] = (smoothed[i] as f32).max(water[n - 1]);
    }
    for i in 0..n {
        water[i] = if points[i].lake.is_finite() { points[i].lake as f32 } else { water[i].max(floor[i]) };
    }
    for i in 1..n {
        water[i] = water[i].min(water[i - 1]);
    }
    Profiled { points, s, water, slope }
}

/// The steep reaches' staircase: pools whose water stands level with their
/// downstream lip, and the falls between them.
struct Steps {
    /// (start, end, level): a pool's extent along the river and its surface.
    pools: Vec<(f64, f64, f32)>,
    /// (position, upper, lower): a fall at a pool's upstream end, from the
    /// surface above it down to the pool.
    falls: Vec<(f64, f32, f32)>,
}

fn interpolate(s: &[f64], values: &[f32], at: f64) -> f32 {
    let n = s.len();
    let i = s.partition_point(|&v| v < at).clamp(1, n - 1);
    let t = ((at - s[i - 1]) / (s[i] - s[i - 1]).max(1e-9)).clamp(0.0, 1.0) as f32;
    values[i - 1] + (values[i] - values[i - 1]) * t
}

fn find_steps(profile: &Profiled, seed: u64) -> Steps {
    let n = profile.s.len();
    let mut steps = Steps { pools: Vec::new(), falls: Vec::new() };
    if n < 3 {
        return steps;
    }
    let length = profile.s[n - 1];
    let mut position = 0.0;
    let mut step_index = 0u64;
    // (end, level) of the pool just laid, when the next one follows it.
    let mut previous: Option<(f64, f32)> = None;
    while position < length {
        let slope = interpolate(&profile.s, &profile.slope, position);
        let i = profile.s.partition_point(|&v| v < position).min(n - 1);
        // Steps and falls are set by the valley and its bedrock, not by how
        // narrow the stream has become: spaced in the catchment's typical
        // widths.
        let width = 2.0 * half_width(profile.points[i].area as f32);
        if slope < STEP_SLOPE {
            position += (width as f64).max(2.0);
            previous = None;
            continue;
        }
        // Steps about two and a half widths apart, irregular; drops grow
        // with slope into waterfalls where the mountainside is steep.
        let jitter = 0.65 + 0.7 * random(seed ^ 0x57E9, step_index);
        step_index += 1;
        // Each pool is cut into the slope as deep as the drop at its head, so
        // a drop stays a few metres even on a mountainside: it steps down in
        // a cascade rather than one fall in a gorge.
        let maximum_drop = 0.8 + 2.4 * smoothstep(0.10, 0.5, slope) + 2.8 * smoothstep(0.5, 1.2, slope);
        let drop = (slope * 2.6 * width * jitter).clamp(0.22, maximum_drop);
        let spacing = ((drop / slope) as f64).max(1.6 * width as f64);
        // The next step goes where the land itself drops most steeply near
        // there, a natural ledge, so its pool needs the least cutting.
        let window = spacing * 0.3;
        let lo = profile.s.partition_point(|&v| v < position + spacing - window).max(1);
        let hi = profile.s.partition_point(|&v| v <= position + spacing + window).min(n - 1);
        let mut end = position + spacing;
        let mut steepest = f32::NEG_INFINITY;
        for k in lo..hi {
            let run = (profile.s[k + 1] - profile.s[k - 1]).max(0.1) as f32;
            let local = (profile.water[k - 1] - profile.water[k + 1]) / run;
            if local > steepest {
                steepest = local;
                end = profile.s[k];
            }
        }
        let end = end.max(position + 1.2 * width as f64);
        if end >= length - (width as f64) * 1.5 {
            break;
        }
        // The pool lies at the surface's level at its lip: the reach is
        // carved down to it, never filled.
        let level = interpolate(&profile.s, &profile.water, end);
        let upper = match previous {
            Some((previous_end, previous_level)) if (previous_end - position).abs() < 1e-6 => previous_level,
            _ => interpolate(&profile.s, &profile.water, position),
        };
        if upper - level > 0.12 {
            steps.falls.push((position, upper, level));
        }
        steps.pools.push((position, end, level));
        previous = Some((end, level));
        position = end;
    }
    steps
}

// ---------------------------------------------------------------------------
// Assembly
// ---------------------------------------------------------------------------

/// Node spacing for a channel of this half width.
fn node_spacing(half_width: f32) -> f64 {
    (half_width as f64 * 1.5).clamp(2.6, 7.0)
}

#[allow(clippy::too_many_arguments)]
fn build_nodes(noise: &NoiseField, centreline: Vec<PathPoint>, end_level: Option<f32>, seed: u64) -> Vec<RiverNode> {
    let mut centreline = resample(&centreline, |p| node_spacing(half_width(p.area as f32)));
    // Area never shrinks downstream.
    for i in 1..centreline.len() {
        centreline[i].area = centreline[i].area.max(centreline[i - 1].area);
    }
    if centreline.len() < 2 {
        return Vec::new();
    }
    let profile = water_profile(noise, centreline, end_level);
    let steps = find_steps(&profile, seed);
    let n = profile.points.len();

    // Stepped surface: within a pool the water is level with its lip.
    let pool_level = |at: f64, smooth: f32| -> f32 {
        let k = steps.pools.partition_point(|pool| pool.1 < at);
        match steps.pools.get(k) {
            Some(&(start, _, level)) if at > start => smooth.min(level),
            _ => smooth,
        }
    };

    // Lay out nodes, inserting a lip node and a fall-foot node at each fall.
    let mut nodes: Vec<RiverNode> = Vec::with_capacity(n + steps.falls.len() * 2);
    let mut fall_index = 0;
    let sample_point = |at: f64| -> PathPoint {
        let i = profile.s.partition_point(|&v| v < at).clamp(1, n - 1);
        let t = ((at - profile.s[i - 1]) / (profile.s[i] - profile.s[i - 1]).max(1e-9)).clamp(0.0, 1.0);
        lerp_point(&profile.points[i - 1], &profile.points[i], t)
    };
    let make = |point: PathPoint, along: f64, water: f32, slope: f32| RiverNode {
        position: [point.p[0] as f32, point.p[1] as f32],
        water,
        half_width: half_width(point.area as f32),
        depth: 0.0,
        speed: 0.0,
        discharge: discharge(point.area as f32),
        area: point.area as f32,
        bank: 0.0,
        skew: 0.0,
        turbulence: 0.0,
        slope,
        along: along as f32,
        fall: 0.0,
        lake: point.lake.is_finite(),
    };
    for i in 0..n {
        let s = profile.s[i];
        while fall_index < steps.falls.len() && steps.falls[fall_index].0 <= s + 0.6 {
            let (lip, upper, lower) = steps.falls[fall_index];
            let slope = interpolate(&profile.s, &profile.slope, lip);
            if nodes.last().is_none_or(|last: &RiverNode| (last.along as f64) < lip - 0.3) {
                let mut node = make(sample_point(lip), lip, upper, slope);
                node.fall = upper - lower;
                nodes.push(node);
                // The water leaves the lip and lands a little downstream.
                let foot = lip + 0.35;
                nodes.push(make(sample_point(foot), foot, lower, slope));
            }
            fall_index += 1;
        }
        // Keep regular nodes clear of a fall's lip and foot.
        let crowded = nodes.last().is_some_and(|last| (last.along as f64) > s - 0.6);
        if crowded && i > 0 && i + 1 < n {
            continue;
        }
        let water = pool_level(s, profile.water[i]);
        nodes.push(make(profile.points[i], s, water, profile.slope[i]));
    }
    for i in 1..nodes.len() {
        nodes[i].water = nodes[i].water.min(nodes[i - 1].water);
    }

    // Hydraulics, plunge pools and whitewater.
    let count = nodes.len();
    let mut since_fall = f32::INFINITY;
    let mut last_drop = 0.0f32;
    for i in 0..count {
        if i > 0 {
            since_fall += nodes[i].along - nodes[i - 1].along;
        }
        if i > 0 && nodes[i - 1].fall > 0.0 {
            since_fall = 0.0;
            last_drop = nodes[i - 1].fall;
        }
        let node = &mut nodes[i];
        let (half_width, depth, _) = channel(node.discharge, node.slope);
        node.half_width = half_width;
        // A plunge pool is scoured deep beneath the fall and shoals toward
        // the next lip.
        let plunging = 1.0 - smoothstep(0.0, 3.0 * node.half_width + 2.0, since_fall);
        let scour = (0.35 * last_drop).min(2.5) * plunging;
        node.depth = depth + scour;
        // ...and wider than the channel that feeds it.
        node.half_width *= 1.0 + 0.5 * plunging * smoothstep(0.3, 2.0, last_drop);
        let mean_depth = node.depth / 1.5;
        node.speed = (node.discharge / (2.0 * node.half_width * mean_depth.max(0.05))).clamp(0.15, 6.0);
        let cascade = smoothstep(0.02, 0.12, node.slope);
        let plunge = 1.0 - smoothstep(0.0, 2.5 * node.half_width + 3.0, since_fall);
        node.turbulence = (0.15 * smoothstep(0.004, 0.02, node.slope) + 0.6 * cascade
            + plunge * smoothstep(0.1, 1.2, last_drop))
            .min(1.0);
        if node.fall > 0.0 {
            node.turbulence = node.turbulence.max(0.8);
        }
        // Lowland banks are low and grassy; mountain channels cut steep
        // banks into stony ground.
        node.bank = 0.55 + 0.9 * smoothstep(0.003, 0.08, node.slope);
    }
    // Heads begin as a seep that gathers into a channel.
    let head_length = 30.0f32;
    for node in nodes.iter_mut() {
        let grow = 0.35 + 0.65 * smoothstep(0.0, head_length, node.along);
        node.half_width *= grow;
        node.depth *= grow;
        // ...carrying the water the seep has gathered so far.
        node.discharge *= grow * grow;
    }
    // No channel keeps one width: it swells through its pools and narrows
    // over its riffles every few widths, irregularly. A lake's water has
    // no channel to vary.
    let offset = (seed % 100_000) as f64 * 5.3;
    for node in nodes.iter_mut().filter(|node| !node.lake) {
        let width = 2.0 * node.half_width as f64;
        let swell = field_noise([node.along as f64, offset], (5.5 * width).max(9.0), 311);
        node.half_width *= 0.8 + 0.4 * swell;
        // Its banks change too: here a low shelving edge, there a steeper
        // face where roots or a harder layer hold the soil.
        let firmness = field_noise([node.along as f64, offset + 57.0], (4.0 * width).max(8.0), 313);
        node.bank *= 0.7 + 0.6 * firmness;
        // The same water runs faster through the narrows.
        let mean_depth = (node.depth / 1.5).max(0.05);
        node.speed = (node.discharge / (2.0 * node.half_width * mean_depth)).clamp(0.15, 6.0);
    }
    // Skew from the signed curvature: the thalweg hugs the outside of a bend.
    let curvature = signed_curvature(&nodes);
    let curvature = smooth_values(&curvature, 2.0);
    for (node, k) in nodes.iter_mut().zip(curvature) {
        let width = 2.0 * node.half_width as f64;
        // Left turn (k > 0): the outer bank is on the right (-).
        node.skew = (-k * width * 1.6).clamp(-0.6, 0.6) as f32;
    }
    nodes
}

fn signed_curvature(nodes: &[RiverNode]) -> Vec<f64> {
    let n = nodes.len();
    (0..n)
        .map(|i| {
            if i == 0 || i + 1 >= n {
                return 0.0;
            }
            let a = nodes[i - 1].position;
            let b = nodes[i].position;
            let c = nodes[i + 1].position;
            let ab = [(b[0] - a[0]) as f64, (b[1] - a[1]) as f64];
            let bc = [(c[0] - b[0]) as f64, (c[1] - b[1]) as f64];
            let la = ab[0].hypot(ab[1]);
            let lb = bc[0].hypot(bc[1]);
            if la < 0.2 || lb < 0.2 {
                return 0.0;
            }
            let cross = ab[0] * bc[1] - ab[1] * bc[0];
            let dot = ab[0] * bc[0] + ab[1] * bc[1];
            cross.atan2(dot) / ((la + lb) * 0.5)
        })
        .collect()
}

/// The finished network of one region.
pub struct RiverNetwork {
    pub region: [i64; 2],
    pub rivers: Vec<River>,
    pub segments: Vec<RiverSegment>,
    pub grid: SegmentGrid,
    pub lakes: Vec<Lake>,
    /// The water surfaces, ready to upload.
    pub surface: super::surface::SurfaceMesh,
    /// Seconds the generation took.
    pub build_seconds: f32,
}

/// Generate the rivers of a region.
pub fn generate(noise: &NoiseField, region: [i64; 2]) -> RiverNetwork {
    let start = std::time::Instant::now();
    let routing = route(noise, region);
    let paths = extract_paths(&routing, region);

    // Parents before their tributaries.
    let mut order: Vec<usize> = Vec::with_capacity(paths.len());
    {
        let mut children: Vec<Vec<usize>> = vec![Vec::new(); paths.len()];
        let mut roots = Vec::new();
        for (index, path) in paths.iter().enumerate() {
            match path.end {
                RiverEnd::Confluence(parent, _) => children[parent].push(index),
                _ => roots.push(index),
            }
        }
        let mut stack = roots;
        stack.reverse();
        while let Some(index) = stack.pop() {
            order.push(index);
            for &child in children[index].iter().rev() {
                stack.push(child);
            }
        }
    }

    // The natural ground along every river's coarse path, sampled in
    // parallel: no corridor depends on another.
    let corridors: Vec<Corridor> = parallel_map(&paths, |path| {
        let points: Vec<[f64; 2]> = path.cells.iter().map(|&cell| routing.centre(cell as usize)).collect();
        Corridor::new(noise, &points, FLOW_CORRIDOR, &[])
    });

    // Tributaries read their parent's finished centreline.
    let mut rivers: Vec<Option<River>> = vec![None; paths.len()];
    let mut lakes = Lakes::default();
    for &index in &order {
        let path = &paths[index];
        let mut coarse: Vec<PathPoint> = path
            .cells
            .iter()
            .map(|&cell| PathPoint {
                p: routing.centre(cell as usize),
                area: (routing.area[cell as usize] * CELL_AREA_KM2) as f64,
                lake: f64::NEG_INFINITY,
            })
            .collect();
        // A tributary's last cell is its parent's; it should not carry the
        // parent's catchment.
        if let RiverEnd::Confluence(..) = path.end {
            let n = coarse.len();
            if n >= 2 {
                coarse[n - 1].area = coarse[n - 2].area;
            }
        }
        if coarse.len() < 2 {
            continue;
        }
        // The water finds its own way down the corridor to where it leaves:
        // the sea, the parent's channel, or the grid's edge.
        let parent = match path.end {
            RiverEnd::Confluence(parent, _) => rivers[parent].as_ref().map(|river| (parent, river)),
            _ => None,
        };
        let outlets_in = |corridor: &Corridor| {
            let mut outlets = Outlets::none(corridor);
            match (path.end, parent) {
                (RiverEnd::Sea, _) => outlets.sea(corridor),
                (RiverEnd::Confluence(..), Some((_, parent_river))) => outlets.channel(corridor, parent_river),
                _ => {}
            }
            if !outlets.any() {
                outlets.around(corridor, coarse[coarse.len() - 1].p, FLOW_CELL * 1.5);
            }
            outlets
        };
        // The corridor runs along the coarse path; a tributary's must also
        // reach its parent's actual course, which can lie well off the
        // parent's own coarse path.
        let mut route: Vec<[f64; 2]> = coarse.iter().map(|p| p.p).collect();
        let mut wider: Option<Corridor> = None;
        if let Some((_, parent_river)) = parent {
            let (_, point) = nearest_on_river(parent_river, route[route.len() - 1]);
            route.push(point);
            let mut reached = Outlets::none(&corridors[index]);
            reached.channel(&corridors[index], parent_river);
            if !reached.any() {
                wider = Some(Corridor::new(noise, &route, FLOW_CORRIDOR, &[]));
            }
        }
        // Where the water seems held in a basin it can in fact leave (a
        // ridge across the corridor that it would flow around), the river is
        // routed again through a wider one.
        let mut reaches = [FLOW_CORRIDOR * 2.5, FLOW_CORRIDOR * 5.0].into_iter();
        let mut to_sea = false;
        let flow = loop {
            let corridor = wider.as_ref().unwrap_or(&corridors[index]);
            let outlets = if to_sea {
                let mut outlets = Outlets::none(corridor);
                outlets.sea(corridor);
                outlets
            } else {
                outlets_in(corridor)
            };
            let Some(flow) = flow_path(corridor, &outlets, coarse[0].p) else {
                break None;
            };
            match raise_lakes(noise, corridor, &flow, &mut lakes) {
                None => break Some(flow),
                // The water's real way down: out through the gap to the sea.
                Some(Filled::ToSea(way)) if !to_sea => {
                    wider = Some(Corridor::new(noise, &route, FLOW_CORRIDOR, &way));
                    to_sea = true;
                }
                Some(_) => match reaches.next() {
                    Some(reach) => wider = Some(Corridor::new(noise, &route, reach, &[])),
                    None => break Some(flow),
                },
            }
        };
        let end = if to_sea { RiverEnd::Sea } else { path.end };
        let parent = parent.filter(|_| !to_sea);
        let Some(flow) = flow else {
            continue;
        };
        // Each point of the flow path takes its catchment from the coarse
        // path beside it.
        let mut points: Vec<PathPoint> = Vec::with_capacity(flow.path.len() + 1);
        let mut j = 0;
        for &p in &flow.path {
            let ahead = (j + 4).min(coarse.len() - 1);
            for k in j + 1..=ahead {
                if distance(coarse[k].p, p) < distance(coarse[j].p, p) {
                    j = k;
                }
            }
            points.push(PathPoint {
                p,
                area: coarse[j].area,
                lake: f64::NEG_INFINITY,
            });
        }
        let mut end_level = None;
        if let Some((_, parent_river)) = parent {
            // Run on into the parent's thalweg, and meet its surface there:
            // the water has reached its channel, so that is a step across it,
            // never a cut across country.
            let last = *points.last().unwrap();
            let (node, point) = nearest_on_river(parent_river, last.p);
            if distance(point, last.p) <= parent_river.nodes[node].half_width as f64 + 2.0 * FLOW_CELL {
                points.push(PathPoint { p: point, ..last });
                end_level = Some(parent_river.nodes[node].water);
            }
        }
        if points.len() < 2 {
            continue;
        }
        // The grid's stair-steps smoothed out, then the path settled back
        // onto the lowest ground across it.
        let mut centreline = resample(&points, |_| FLOW_CELL);
        smooth(&mut centreline, 1.5);
        snap_to_floor(noise, &mut centreline, 8.0, 1.0);
        let mut centreline = resample(&centreline, |_| FLOW_CELL);
        smooth(&mut centreline, 1.0);
        relax_tight_bends(&mut centreline);
        for point in centreline.iter_mut() {
            point.lake = lakes.level_at(point.p);
        }
        let mut nodes = build_nodes(noise, centreline, end_level, path.seed);
        if nodes.len() < 2 {
            continue;
        }
        let mut surface_end = nodes.len() - 1;
        match end {
            RiverEnd::Confluence(parent, _) => {
                if let Some(parent_river) = rivers[parent].as_ref() {
                    // The tributary's own water stops at the parent's bank.
                    for (i, node) in nodes.iter().enumerate() {
                        let (k, point) = nearest_on_river(parent_river, [node.position[0] as f64, node.position[1] as f64]);
                        let d = distance(point, [node.position[0] as f64, node.position[1] as f64]) as f32;
                        if d < parent_river.nodes[k].half_width + 0.3 {
                            surface_end = i.max(1);
                            break;
                        }
                    }
                }
            }
            RiverEnd::Sea => {
                // The river's water gives way to the sea where the ground
                // drops to it; its channel runs on a few widths into the
                // shallows and no further.
                for (i, node) in nodes.iter().enumerate() {
                    if base_height(noise, node.position[0], node.position[1]) < SEA_LEVEL + 0.25 {
                        surface_end = i.max(1);
                        break;
                    }
                }
                let stop = nodes[surface_end].along + 2.0 * nodes[surface_end].half_width + 4.0;
                let keep = nodes.iter().position(|n| n.along > stop).unwrap_or(nodes.len()).max(surface_end + 1);
                nodes.truncate(keep);
            }
            RiverEnd::Edge => {}
        }
        rivers[index] = Some(River {
            nodes,
            end,
            surface_end,
        });
    }

    // Compact, remapping confluences.
    let mut remap = vec![usize::MAX; rivers.len()];
    let mut finished = Vec::new();
    for (index, river) in rivers.into_iter().enumerate() {
        if let Some(river) = river {
            remap[index] = finished.len();
            finished.push(river);
        }
    }
    for river in &mut finished {
        if let RiverEnd::Confluence(parent, node) = river.end {
            river.end = if remap[parent] == usize::MAX {
                RiverEnd::Edge
            } else {
                RiverEnd::Confluence(remap[parent], node)
            };
        }
    }

    let segments = build_segments(&finished);
    let origin = routing_origin(region);
    let resolution = ((REGION_SIZE + 2.0 * ROUTING_MARGIN) / GRID_CELL as f64).ceil() as usize;
    let grid_origin = [origin[0] as f32, origin[1] as f32];
    let mut grid = SegmentGrid::build(grid_origin, resolution, &segments);
    let lakes = lakes.lakes;
    for lake in &lakes {
        grid.add_lake(lake.level, &lake.cells, FLOW_CELL as f32);
    }
    let surface = super::surface::build(&finished, &lakes);
    RiverNetwork {
        region,
        rivers: finished,
        segments,
        grid,
        lakes,
        surface,
        build_seconds: start.elapsed().as_secs_f32(),
    }
}

/// The node of `river` nearest `p` and the nearest point on its centreline.
fn nearest_on_river(river: &River, p: [f64; 2]) -> (usize, [f64; 2]) {
    let mut best = (0usize, [0.0, 0.0], f64::INFINITY);
    for i in 0..river.nodes.len().saturating_sub(1) {
        let a = [river.nodes[i].position[0] as f64, river.nodes[i].position[1] as f64];
        let b = [river.nodes[i + 1].position[0] as f64, river.nodes[i + 1].position[1] as f64];
        let ab = [b[0] - a[0], b[1] - a[1]];
        let t = (((p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1]) / (ab[0] * ab[0] + ab[1] * ab[1]).max(1e-9)).clamp(0.0, 1.0);
        let q = [a[0] + ab[0] * t, a[1] + ab[1] * t];
        let d = distance(p, q);
        if d < best.2 {
            best = (if t < 0.5 { i } else { i + 1 }, q, d);
        }
    }
    (best.0, best.1)
}

fn build_segments(rivers: &[River]) -> Vec<RiverSegment> {
    let mut segments = Vec::new();
    for river in rivers {
        for pair in river.nodes.windows(2) {
            let (a, b) = (&pair[0], &pair[1]);
            // Under a lake the river has no channel of its own.
            if a.lake && b.lake {
                continue;
            }
            let shore = if a.lake || b.lake { 0.0 } else { 1.0 };
            segments.push(RiverSegment {
                a: a.position,
                b: b.position,
                water: [a.water, b.water],
                half_width: [a.half_width, b.half_width],
                depth: [a.depth, b.depth],
                speed: [a.speed, b.speed],
                bank: 0.5 * (a.bank + b.bank),
                skew: 0.5 * (a.skew + b.skew),
                turbulence: a.turbulence.max(b.turbulence),
                levee: smoothstep(SEA_LEVEL + 0.05, SEA_LEVEL + 0.6, a.water.min(b.water)) * shore,
            });
        }
        // A river's channel opens into the sea a little past its last node.
        if river.end == RiverEnd::Sea && river.nodes.len() >= 2 {
            let a = river.nodes[river.nodes.len() - 2];
            let b = river.nodes[river.nodes.len() - 1];
            let d = [b.position[0] - a.position[0], b.position[1] - a.position[1]];
            let l = d[0].hypot(d[1]).max(1e-3);
            let reach = 3.0 * b.half_width;
            segments.push(RiverSegment {
                a: b.position,
                b: [b.position[0] + d[0] / l * reach, b.position[1] + d[1] / l * reach],
                water: [b.water, b.water],
                half_width: [b.half_width, b.half_width * 1.3],
                depth: [b.depth, b.depth],
                speed: [b.speed, b.speed * 0.6],
                bank: b.bank * 0.5,
                skew: 0.0,
                turbulence: 0.0,
                levee: 0.0,
            });
        }
    }
    segments
}

impl RiverNetwork {
    pub fn envelope(&self, x: f32, z: f32) -> super::carve::Envelope {
        super::carve::envelope_at(&self.segments, &self.grid, [x, z])
    }

    /// Total length of every river, km.
    pub fn total_length_km(&self) -> f32 {
        self.rivers
            .iter()
            .map(|r| r.nodes.last().map_or(0.0, |n| n.along) - r.nodes.first().map_or(0.0, |n| n.along))
            .sum::<f32>()
            / 1000.0
    }

    pub fn waterfall_count(&self) -> usize {
        self.rivers.iter().flat_map(|r| &r.nodes).filter(|n| n.fall > 1.5).count()
    }
}
