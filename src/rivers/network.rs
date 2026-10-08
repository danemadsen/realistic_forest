//! Where the rivers run: drainage routing, channel extraction, the course
//! each river's water finds over the land, lakes, and the long profile, down
//! to the rapids of a mountain creek.
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
//!    backed up to it above. However steep the land, the water runs down it
//!    as rapids, never dropping off a ledge.
//! 6. **Hydraulics.** Bankfull discharge grows with catchment. Width and
//!    depth follow downstream hydraulic geometry with the slope's terms
//!    ([`channel`]): more water makes a channel wider and deeper, a steeper
//!    slope narrower, deeper and faster; it swells and narrows through its
//!    pools and riffles, and churns white down its rapids.
//!
//! Everything is a function of world position: the grid is world-aligned and
//! every random choice is keyed by world coordinates or by a river's head
//! cell, so two regions that both contain a whole catchment agree on it.

use super::carve::{BANK_REACH, CARVE_ROUNDING, GRID_CELL, RiverSegment, SegmentGrid, bank_run};
use crate::constants::SEA_LEVEL;
use crate::noise::{NoiseField, base_height};
use crate::vegetation::ecology::{cell_key, noise2 as field_noise};
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
/// weighted); steeper ground starts one with a little less, down to the
/// minimum.
pub const CHANNEL_AREA: f32 = 0.32;
pub const MINIMUM_CHANNEL_AREA: f32 = 0.05;
/// km² per routing cell.
const CELL_AREA_KM2: f32 = (ROUTING_CELL * ROUTING_CELL / 1.0e6) as f32;
/// Slope above which a reach counts as steep: rapids and cascades.
pub const STEP_SLOPE: f32 = 0.028;
/// Gravity, m/s².
pub const GRAVITY: f32 = 9.81;
/// A hollow the water would stand deeper than this in holds a lake (if it
/// is deep enough over its whole extent, see `LAKE_MEAN_DEPTH`); the river
/// cuts through the rim of a shallower one.
pub const LAKE_DEPTH: f32 = 2.0;
/// A basin whose water would average less than this deep is a flooded flat,
/// not a pond: the river cuts through its rim instead.
const LAKE_MEAN_DEPTH: f32 = 0.6;
/// How far a lake's sheet reaches up its shore, in flow cells.
const SHORE_CELLS: i64 = 3;
/// How far past its sheet a lake's surface field reaches (`lake_surface`),
/// metres: far enough that a shore band measured from it has faded out.
const LAKE_FIELD_REACH: f32 = 24.0;
/// How far under the ground the field runs past the sheet where the ground
/// lies below the lake's level, metres; over ground this much above the
/// level or more it stays at the level, and between the two it blends.
const LAKE_FIELD_SINK: f32 = 2.0;
const LAKE_FIELD_RISE: f32 = 0.5;
/// The field sinks this much per metre past this far from the sheet.
const LAKE_FIELD_FADE: f32 = 0.15;
const LAKE_FIELD_FADE_START: f32 = 4.0;
/// A creek begins where the land along it first eases below this slope:
/// the steep mountainside above is seeps and sheet wash, not a channel.
const HEAD_SLOPE: f64 = 0.22;
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
    /// Water-surface slope of the reach, rise over run.
    pub slope: f32,
    /// Metres downstream from the river's head.
    pub along: f32,
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
    /// The cells around them its sheet reaches over too, so the shoreline
    /// is wherever the ground meets the water: hollows beside the basin and
    /// a few cells up the shore (see `shore_cells`).
    pub shore: Vec<[i32; 2]>,
    /// The corners of the sheet's outer edge (corner `c` at
    /// `c * FLOW_CELL`) sunk to these heights, so the edge runs under
    /// whatever covers it: a little under the ground, or just under a
    /// river's water where its channel meets the lake. Every other corner
    /// lies at the lake's level.
    pub edge: Vec<([i32; 2], f32)>,
    /// The current at the sheet's corners where a river runs into or out of
    /// it (corner `c` at `c * FLOW_CELL`), m/s. Elsewhere the water is still.
    pub current: Vec<([i32; 2], [f32; 2])>,
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
    /// Whether a cell lies under the open sea (`open_sea_cells`).
    open_sea: Vec<bool>,
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
        let worker_loop = || {
            let mut out = Vec::new();
            loop {
                let index = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if index >= items.len() {
                    break out;
                }
                out.push((index, work(&items[index])));
            }
        };
        let handles: Vec<_> = (0..workers).map(|_| scope.spawn(worker_loop)).collect();
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
    let sample_row = |z: usize, row: &mut [f32]| {
        let wz = origin[1] + (z as f64 + 0.5) * ROUTING_CELL;
        for (x, value) in row.iter_mut().enumerate() {
            let wx = origin[0] + (x as f64 + 0.5) * ROUTING_CELL;
            *value = base_height(noise, wx as f32, wz as f32);
        }
    };
    let heights: Vec<f32> = parallel_rows(resolution, resolution, sample_row);
    let count = resolution * resolution;
    let open_sea = open_sea_cells(&heights, resolution);
    let mut receiver = vec![NONE; count];
    let mut filled = heights.clone();
    let mut visited = vec![false; count];
    let mut heap = BinaryHeap::new();
    // The sea and the grid's edge are where water leaves.
    for cell in 0..count {
        let x = cell % resolution;
        let z = cell / resolution;
        let edge = x == 0 || z == 0 || x == resolution - 1 || z == resolution - 1;
        if open_sea[cell] || edge {
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
    // Channels begin where enough water gathers, a little sooner on steeper
    // ground, and once begun they run on to the sea.
    let mut channel = vec![false; count];
    for &cell in order.iter().rev() {
        let cell = cell as usize;
        let r = receiver[cell];
        if open_sea[cell] || r == NONE {
            continue;
        }
        let r = r as usize;
        let run = if (cell % resolution) != (r % resolution) && (cell / resolution) != (r / resolution) {
            ROUTING_CELL * std::f64::consts::SQRT_2
        } else {
            ROUTING_CELL
        };
        let slope = ((heights[cell] - heights[r]) as f64 / run).max(0.0) as f32;
        let threshold = (CHANNEL_AREA / (1.0 + 2.0 * slope)).max(MINIMUM_CHANNEL_AREA) / CELL_AREA_KM2;
        if area[cell] >= threshold {
            channel[cell] = true;
        }
        if channel[cell] && !open_sea[r] {
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
        open_sea,
    }
}

/// The routing cells under the open sea: below the sea's level and joined,
/// edge to edge through cells that are too, to water `OPEN_SEA_DEPTH` deep
/// or to the grid's edge. A coastal hollow dipping just under the sea's
/// level is not the sea: it fills and spills like any basin, and a channel
/// crosses it on its way to the real coast.
fn open_sea_cells(heights: &[f32], resolution: usize) -> Vec<bool> {
    let mut open = vec![false; heights.len()];
    let mut stack: Vec<usize> = Vec::new();
    for (cell, &height) in heights.iter().enumerate() {
        let (x, z) = (cell % resolution, cell / resolution);
        let edge = x == 0 || z == 0 || x == resolution - 1 || z == resolution - 1;
        if height < SEA_LEVEL - OPEN_SEA_DEPTH || (edge && height < SEA_LEVEL) {
            open[cell] = true;
            stack.push(cell);
        }
    }
    while let Some(cell) = stack.pop() {
        let (x, z) = ((cell % resolution) as i64, (cell / resolution) as i64);
        for (dx, dz) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
            let (nx, nz) = (x + dx, z + dz);
            if nx < 0 || nz < 0 || nx >= resolution as i64 || nz >= resolution as i64 {
                continue;
            }
            let next = nz as usize * resolution + nx as usize;
            if !open[next] && heights[next] < SEA_LEVEL {
                open[next] = true;
                stack.push(next);
            }
        }
    }
    open
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
            if routing.open_sea[r] {
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
    let tangent_at = |i: usize| -> [f64; 2] {
        let a = points[i.saturating_sub(1)].p;
        let b = points[(i + 1).min(n - 1)].p;
        let d = [b[0] - a[0], b[1] - a[1]];
        let length = d[0].hypot(d[1]).max(1e-9);
        [d[0] / length, d[1] / length]
    };
    (0..n).map(tangent_at).collect()
}

/// Smooth a falling profile with a Gaussian of `sigma` samples, carrying
/// it on past each end at its own slope (an odd reflection) rather than
/// holding its end values, which would pull a fall down just below its top
/// and up just above its foot.
fn smooth_falling(values: &[f64], sigma: f64) -> Vec<f64> {
    let n = values.len();
    if n < 3 || sigma <= 0.0 {
        return values.to_vec();
    }
    let last = n as i64 - 1;
    let at = |j: i64| -> f64 {
        if j < 0 {
            2.0 * values[0] - values[(-j).min(last) as usize]
        } else if j > last {
            2.0 * values[last as usize] - values[(2 * last - j).max(0) as usize]
        } else {
            values[j as usize]
        }
    };
    let radius = (sigma * 3.0).ceil() as i64;
    let smooth_at = |i: i64| -> f64 {
        let mut sum = 0.0;
        let mut total = 0.0;
        for k in -radius..=radius {
            let w = (-0.5 * (k as f64 / sigma).powi(2)).exp();
            sum += at(i + k) * w;
            total += w;
        }
        sum / total
    };
    (0..n as i64).map(smooth_at).collect()
}

/// Smooth a scalar sequence with a Gaussian of `sigma` samples.
fn smooth_values(values: &[f64], sigma: f64) -> Vec<f64> {
    let n = values.len();
    if n < 3 || sigma <= 0.0 {
        return values.to_vec();
    }
    let radius = (sigma * 3.0).ceil() as i64;
    let smooth_at = |i: usize| -> f64 {
        let mut sum = 0.0;
        let mut total = 0.0;
        for k in -radius..=radius {
            let j = (i as i64 + k).clamp(0, n as i64 - 1) as usize;
            let w = (-0.5 * (k as f64 / sigma).powi(2)).exp();
            sum += values[j] * w;
            total += w;
        }
        sum / total
    };
    (0..n).map(smooth_at).collect()
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

/// Bankfull mean depth of a 1 m³/s channel on a 1 % slope, metres.
const DEPTH_COEFFICIENT: f32 = 0.75;
/// A channel's thalweg over its mean depth: its bed is a flat-bottomed bowl
/// (carve.rs), whose mean depth is four fifths of its deepest.
const THALWEG_OVER_MEAN: f32 = 1.25;

/// The bankfull channel a discharge cuts on a slope: (half width, thalweg
/// depth, mean speed). Downstream hydraulic geometry, the power laws real
/// rivers follow: width ~ Q^0.5 and depth ~ Q^0.4 (Leopold & Maddock 1953),
/// fitted so a small forested creek is ten to fifteen times as wide as it
/// is deep, with the slope's own terms: on a steep slope a stream is held
/// in a narrower, deeper slot between boulders and bedrock (width ~ S^-0.25,
/// depth ~ S^0.1, never narrower than six times its depth); on the flat it
/// spreads wide and shallow over its own gravel and silt. A parabolic
/// section's thalweg is a quarter deeper than its mean (`THALWEG_OVER_MEAN`).
pub fn channel(discharge: f32, slope: f32) -> (f32, f32, f32) {
    let q = discharge.max(1e-3);
    let s = (slope / 0.01).clamp(0.2, 10.0);
    let width = 7.0 * q.powf(0.5) * s.powf(-0.25);
    let half_width = (0.5 * width).max(0.8);
    let mean_depth = (DEPTH_COEFFICIENT * q.powf(0.4) * s.powf(0.1)).min(2.0 * half_width / 6.0);
    (half_width, (THALWEG_OVER_MEAN * mean_depth).max(0.3), flow_speed(mean_depth, 0.01 * s))
}

/// Mean speed of water `mean_depth` deep running down a surface slope:
/// Manning's equation, v = R^(2/3) S^(1/2) / n, with the hydraulic radius
/// taken as the mean depth and a gravel and boulder bed's roughness from
/// Jarrett (1984), n = 0.32 S^0.38 R^-0.16. A steep bed's boulders and steps
/// hold the water back, so a mountain creek runs at one to two metres a
/// second and a lowland one at a few tenths to one.
pub fn flow_speed(mean_depth: f32, slope: f32) -> f32 {
    let r = mean_depth.max(0.05);
    let s = slope.clamp(4e-4, 0.25);
    let n = (0.32 * s.powf(0.38) * r.powf(-0.16)).max(0.032);
    (r.powf(2.0 / 3.0) * s.sqrt() / n).clamp(0.15, 4.0)
}

fn smoothstep(edge0: f32, edge1: f32, value: f32) -> f32 {
    let t = ((value - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

// ---------------------------------------------------------------------------
// Flow paths
// ---------------------------------------------------------------------------

/// Cell of the fine flow grids, metres: the lookup grid's lake cells.
pub const FLOW_CELL: f64 = 4.0;
const _: () = assert!(FLOW_CELL as f32 * super::carve::LAKE_CELLS_ACROSS as f32 == GRID_CELL);
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
    /// The first cell of each stored block, by slot.
    block_origin: Vec<[i64; 2]>,
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

    /// The cell stored at a storage index.
    fn cell_at(&self, index: usize) -> [i64; 2] {
        let origin = self.block_origin[index / (BLOCK * BLOCK)];
        let within = index % (BLOCK * BLOCK);
        [origin[0] + (within % BLOCK) as i64, origin[1] + (within / BLOCK) as i64]
    }

    /// The ground within `reach` of the path through `points`, and over
    /// any `extra` cells.
    fn new(noise: &NoiseField, points: &[[f64; 2]], reach: f64, extra: &[[i64; 2]]) -> Self {
        let mut minimum = [f64::INFINITY; 2];
        let mut maximum = [f64::NEG_INFINITY; 2];
        let extra_centres = extra.iter().map(|&c| Self::centre(c));
        for p in points.iter().copied().chain(extra_centres) {
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
            block_origin: Vec::new(),
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
                corridor.block_origin.push([
                    first[0] + ((block % blocks[0]) * BLOCK) as i64,
                    first[1] + ((block / blocks[0]) * BLOCK) as i64,
                ]);
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
        // Sample the ground of every marked cell.
        for index in marked {
            let c = Self::centre(corridor.cell_at(index));
            corridor.ground[index] = base_height(noise, c[0] as f32, c[1] as f32);
        }
        corridor
    }
}

/// Sea at least this deep is open water rather than a hollow on the shore,
/// and a hollow joins it only through water at least `OPEN_SEA_PASSAGE`
/// deep.
const OPEN_SEA_DEPTH: f32 = 1.0;
const OPEN_SEA_PASSAGE: f32 = 0.3;

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

    /// Every cell of the corridor under the open sea: joined to water a
    /// metre deep, to the routing cell `mouth` where the coarse path reached
    /// the open sea (the routing grid saw it joined to the sea, perhaps only
    /// beyond the corridor) or, failing any, to the corridor's edge, through
    /// water deep enough to show as sea. A hollow on the beach that dips a
    /// little below the sea level is not the sea: a river ending there would
    /// end in a pit, cut off from the water it runs to.
    fn sea(&mut self, corridor: &Corridor, mouth: Option<[f64; 2]>) {
        let below = |index: usize| corridor.ground[index] < SEA_LEVEL - OPEN_SEA_PASSAGE;
        let in_mouth = |index: usize| {
            mouth.is_some_and(|m| {
                let c = Corridor::centre(corridor.cell_at(index));
                (c[0] - m[0]).abs() <= ROUTING_CELL * 0.5 && (c[1] - m[1]).abs() <= ROUTING_CELL * 0.5
            })
        };
        let flood = |seeds: &[usize], through: &dyn Fn(usize) -> bool| -> Vec<bool> {
            let mut reached = vec![false; corridor.ground.len()];
            let mut stack: Vec<usize> = seeds.to_vec();
            for &index in seeds {
                reached[index] = true;
            }
            while let Some(index) = stack.pop() {
                let cell = corridor.cell_at(index);
                for (dx, dz) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                    if let Some(next) = corridor.inside([cell[0] + dx, cell[1] + dz])
                        && !reached[next]
                        && through(next)
                    {
                        reached[next] = true;
                        stack.push(next);
                    }
                }
            }
            reached
        };
        let deep: Vec<usize> =
            (0..corridor.ground.len()).filter(|&index| corridor.ground[index] < SEA_LEVEL - OPEN_SEA_DEPTH).collect();
        // The coarse mouth's cells count only where they join that deep
        // water at this grid's finer scale (if there is any in reach): a bar
        // narrower than a routing cell can still cut a hollow off from it.
        let joined = (!deep.is_empty()).then(|| flood(&deep, &|index| corridor.ground[index] < SEA_LEVEL));
        let mut open = vec![false; corridor.ground.len()];
        let is_open_water = |&index: &usize| -> bool {
            corridor.ground[index] < SEA_LEVEL - OPEN_SEA_DEPTH
                || (corridor.ground[index] < SEA_LEVEL
                    && in_mouth(index)
                    && joined.as_ref().is_none_or(|joined| joined[index]))
        };
        let mut queue: std::collections::VecDeque<usize> = (0..corridor.ground.len()).filter(is_open_water).collect();
        if queue.is_empty() {
            let is_edge_water = |&index: &usize| -> bool {
                below(index) && {
                    let cell = corridor.cell_at(index);
                    [(1, 0), (-1, 0), (0, 1), (0, -1)]
                        .iter()
                        .any(|&(dx, dz)| corridor.inside([cell[0] + dx, cell[1] + dz]).is_none())
                }
            };
            queue = (0..corridor.ground.len()).filter(is_edge_water).collect();
        }
        if queue.is_empty() {
            // No open water in reach at all: whatever lies under the sea.
            for (index, &ground) in corridor.ground.iter().enumerate() {
                if ground < SEA_LEVEL {
                    self.add(index, ground);
                }
            }
            return;
        }
        for &index in &queue {
            open[index] = true;
        }
        while let Some(index) = queue.pop_front() {
            let cell = corridor.cell_at(index);
            for (dx, dz) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                if let Some(next) = corridor.inside([cell[0] + dx, cell[1] + dz])
                    && !open[next]
                    && below(next)
                {
                    open[next] = true;
                    queue.push_back(next);
                }
            }
        }
        for (index, &ground) in corridor.ground.iter().enumerate() {
            if open[index] {
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
    while let Some(Flood(level, index)) = heap.pop() {
        let cell = corridor.cell_at(index as usize);
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
        let cell = corridor.cell_at(current);
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
        self.fill_below(noise, corridor, start, level, 3)
    }

    /// `fill`, lowering the level to where the water first runs out between
    /// the cells' centres up to `tries` times.
    fn fill_below(&mut self, noise: &NoiseField, corridor: &Corridor, start: [i64; 2], level: f32, tries: u32) -> Filled {
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
                // Water that reaches the open sea was never held in a basin;
                // what the fill crossed is its way there. A hollow just under
                // the sea level is part of the basin, like any other.
                if height < SEA_LEVEL - OPEN_SEA_DEPTH {
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
        // The fill saw each cell only at its centre: where the rim is lower
        // between centres, the water runs out there, and stands no higher.
        if let Some(spill) = spill(noise, &cells, level) {
            return if tries > 0 {
                self.fill_below(noise, corridor, start, spill - 0.05, tries - 1)
            } else {
                Filled::NotBasin
            };
        }
        // A basin shallow over most of its extent is a flooded flat.
        let deepest = cells.iter().map(|&cell| level - ground(cell)).fold(0.0f32, f32::max);
        let mean = cells.iter().map(|&cell| level - ground(cell)).sum::<f32>() / cells.len() as f32;
        if deepest < LAKE_DEPTH || mean < LAKE_MEAN_DEPTH {
            return Filled::NotBasin;
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
        let new_lake = Lake {
            level,
            cells: cells.iter().map(|c| [c[0] as i32, c[1] as i32]).collect(),
            shore: Vec::new(),
            edge: Vec::new(),
            current: Vec::new(),
        };
        self.lakes.push(new_lake);
        Filled::Lake
    }
}

/// Samples across a flow cell when a basin's rim is checked.
const RIM_SAMPLES_PER_CELL: i64 = 4;
/// How far past a basin's cells, in samples along the way, water under its
/// level must reach to have run out of it: farther than any closed arm, bay
/// or island the fill stepped past (one it joined through a gap narrower
/// than its cells), which the water floods to its end instead. Measured
/// along each way, not as ground flooded in all, so a long shore's fringe
/// of samples just under the level is no way out.
const RIM_REACH: u32 = 256;

/// Where water standing at `level` in `cells` runs out of them, metre by
/// metre: through a gap in a rim the flood fill, which sees each cell at its
/// centre, took for whole, and on farther than any closed arm of the basin
/// reaches (`RIM_REACH`). The height of the lowest such way's highest
/// ground, or None if the basin holds its water.
fn spill(noise: &NoiseField, cells: &[[i64; 2]], level: f32) -> Option<f32> {
    let per = RIM_SAMPLES_PER_CELL;
    let step = FLOW_CELL as f32 / per as f32;
    let inside: std::collections::HashSet<[i64; 2]> = cells.iter().copied().collect();
    let in_basin = |s: [i64; 2]| inside.contains(&[s[0].div_euclid(per), s[1].div_euclid(per)]);
    let ground = |s: [i64; 2]| base_height(noise, (s[0] as f32 + 0.5) * step, (s[1] as f32 + 0.5) * step);
    let key = |height: f32| (height * 1000.0).round() as i64;
    let mut heap: std::collections::BinaryHeap<std::cmp::Reverse<(i64, u32, [i64; 2])>> = Default::default();
    let mut seen: std::collections::HashSet<[i64; 2]> = Default::default();
    // From the samples of the basin's edge cells under the level.
    for c in cells {
        if NEIGHBOURS.iter().all(|&(dx, dz)| inside.contains(&[c[0] + dx, c[1] + dz])) {
            continue;
        }
        for j in 0..per {
            for i in 0..per {
                let sample = [c[0] * per + i, c[1] * per + j];
                let height = ground(sample);
                if height < level {
                    seen.insert(sample);
                    heap.push(std::cmp::Reverse((key(height), 0, sample)));
                }
            }
        }
    }
    while let Some(std::cmp::Reverse((water, distance, sample))) = heap.pop() {
        if distance > RIM_REACH {
            return Some(water as f32 / 1000.0);
        }
        for (dx, dz) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
            let next = [sample[0] + dx, sample[1] + dz];
            if in_basin(next) || !seen.insert(next) {
                continue;
            }
            let height = ground(next);
            if height < level {
                heap.push(std::cmp::Reverse((key(height).max(water), distance + 1, next)));
            }
        }
    }
    None
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

/// How far under a lake's level the ground along the river above it must lie
/// for the lake's backwater to flood it (`flood_backwater`).
const BACKWATER_DEPTH: f32 = 0.3;

impl Lakes {
    /// Flood the hollow around `start` to the level of lake `lake`, as part
    /// of it: every cell joined to `start` (edge to edge) under that level.
    /// Nothing changes if the water there would not stay (it reaches the sea,
    /// runs out between the cells' centres, or would cover more than any lake
    /// does), or the hollow belongs to another lake.
    fn extend(&mut self, noise: &NoiseField, corridor: &Corridor, start: [i64; 2], lake: u32) {
        let level = self.lakes[lake as usize].level;
        let ground = |cell: [i64; 2]| -> f32 {
            corridor.inside(cell).map_or_else(
                || {
                    let c = Corridor::centre(cell);
                    base_height(noise, c[0] as f32, c[1] as f32)
                },
                |index| corridor.ground[index],
            )
        };
        if self.cells.contains_key(&start) || ground(start) >= level {
            return;
        }
        let mut inside = std::collections::HashSet::from([start]);
        let mut queue = std::collections::VecDeque::from([start]);
        let mut cells = Vec::new();
        while let Some(cell) = queue.pop_front() {
            cells.push(cell);
            if cells.len() >= MAX_LAKE_CELLS {
                return;
            }
            for (dx, dz) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                let next = [cell[0] + dx, cell[1] + dz];
                if inside.contains(&next) {
                    continue;
                }
                match self.cells.get(&next) {
                    Some(&other) if other == lake => continue,
                    Some(_) => return,
                    None => {}
                }
                let height = ground(next);
                if height < SEA_LEVEL - OPEN_SEA_DEPTH {
                    return;
                }
                if height < level {
                    inside.insert(next);
                    queue.push_back(next);
                }
            }
        }
        // Checked metre by metre with the lake it joins, whose water it is.
        let mut basin = cells.clone();
        basin.extend(self.lakes[lake as usize].cells.iter().map(|c| [c[0] as i64, c[1] as i64]));
        if spill(noise, &basin, level).is_some() {
            return;
        }
        for &cell in &cells {
            self.cells.insert(cell, lake);
        }
        self.lakes[lake as usize].cells.extend(cells.iter().map(|c| [c[0] as i32, c[1] as i32]));
    }
}

/// A lake backs the river above it up to its level: no reach upstream falls
/// below it (`water_profile`). Where the course runs through ground under that
/// level, the water stands over it at the lake's level whether or not the
/// hollow seemed shallow enough to cut through, and would only be held there
/// as a river by a dike either side; a hollow the backwater floods is the
/// lake's, every cell of it under the level.
fn flood_backwater(noise: &NoiseField, corridor: &Corridor, flow: &Flow, lakes: &mut Lakes) {
    let mut below: Option<u32> = None;
    for &p in flow.path.iter().rev() {
        let cell = Corridor::cell_of(p);
        if let Some(&lake) = lakes.cells.get(&cell) {
            below = Some(lake);
            continue;
        }
        let Some(lake) = below else {
            continue;
        };
        let ground = corridor.inside(cell).map_or_else(|| base_height(noise, p[0] as f32, p[1] as f32), |index| corridor.ground[index]);
        if ground < lakes.lakes[lake as usize].level - BACKWATER_DEPTH {
            lakes.extend(noise, corridor, cell, lake);
        }
    }
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

/// Start a creek where it really begins: past the steep upper reach of a
/// mountainside, at the first point where the ground along it, over the
/// next 30 m, falls more gently than `HEAD_SLOPE`. False if it never does.
fn trim_steep_head(noise: &NoiseField, points: &mut Vec<PathPoint>) -> bool {
    let n = points.len();
    let s = arc_lengths(points);
    let ground: Vec<f64> = points.iter().map(|p| base_height(noise, p.p[0] as f32, p.p[1] as f32) as f64).collect();
    let ground_eases = |&i: &usize| -> bool {
        let j = s.partition_point(|&v| v < s[i] + 30.0).min(n - 1);
        j <= i || (ground[i] - ground[j]) / (s[j] - s[i]).max(1.0) < HEAD_SLOPE
    };
    let start = (0..n).find(ground_eases);
    match start {
        Some(start) if n - start >= 2 => {
            points.drain(..start);
            true
        }
        _ => false,
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

/// Below a lake its outlet's surface draws down from the lake's level by at
/// most `OUTLET_SLOPE` over the first `OUTLET_REACH` metres. At the sill it
/// stands at least `OUTLET_FREEBOARD` under the ground beside it; by the
/// reach's end it has settled to the channel's own depth under its banks.
const OUTLET_REACH: f32 = 25.0;
const OUTLET_SLOPE: f32 = 0.025;
const OUTLET_FREEBOARD: f32 = 0.15;

/// Drawdown begins tangent to the pond's level surface, reaching the normal
/// sill slope over its first twelve metres. The ground still limits how
/// high the water can stand where the outlet descends a steep hillside.
fn outlet_drawdown(run: f32) -> f32 {
    const EASING: f32 = 12.0;
    let eased = run.min(EASING);
    OUTLET_SLOPE * (0.5 * eased * eased / EASING + (run - EASING).max(0.0))
}

/// How far above its shore a river's lowest reach is graded to the sea.
const MOUTH_GRADE_REACH: f64 = 120.0;
/// A bar across a river's mouth lower than this above the sea is cut
/// through; higher ground is land the river still has to come down.
const MOUTH_BAR: f32 = 1.5;
/// Deepest a mouth's grade cuts under the river's own profile.
const MOUTH_CUT: f32 = 3.0;

/// Bring a river down to the sea along a smooth grade from its surface
/// `MOUTH_GRADE_REACH` above its shore to the sea's level at the shore, and
/// hold it at the sea's level from there on: where the land drops to the
/// sea down a beach's face, the river cuts down through the beach and the
/// rise behind it, as a real mouth does, rather than tumbling down the face.
/// The grade levels out at the coast, where the sea takes over its surface.
/// A river already falling to the sea over that reach (its profile under
/// the grade, as a valley's is) is left as it is. Never below a lake's
/// level above it.
fn grade_to_sea(water: &mut [f32], s: &[f64], floor: &[f32], shore: usize) {
    let sea = SEA_LEVEL + 0.02;
    for i in shore..water.len() {
        water[i] = water[i].min(sea).max(floor[i]);
    }
    // Never from higher than the last lake above the shore: its outlet then
    // falls from the lake's own level, with no step at its edge.
    let last_lake = (0..shore).rev().find(|&i| floor[i].is_finite()).unwrap_or(0);
    let anchor = s.partition_point(|&v| v < s[shore] - MOUTH_GRADE_REACH).max(last_lake).min(shore);
    let run = s[shore] - s[anchor];
    if run < 1.0 {
        return;
    }
    let top = water[anchor].max(sea);
    for i in anchor..shore {
        // A beach and the dune behind it are cut through, but a sea cliff
        // keeps its rapids rather than becoming a gorge.
        let upstream = ((s[shore] - s[i]) / run) as f32;
        let grade = sea + (top - sea) * smoothstep(0.0, 1.0, upstream);
        water[i] = water[i].min(grade.max(water[i] - MOUTH_CUT)).max(floor[i]);
    }
}

/// The water surface that best follows `raw` (the ground less its bank)
/// while never rising downstream and falling at least a little: the
/// least-squares falling fit (pool-adjacent-violators, weighted by the
/// stretch of river each point stands for). Where the ground rises over a
/// bump and falls again, the river neither cuts the whole bump away nor
/// floods the hollow behind it to the bump's height, but meets them halfway
/// as a real stream's long profile does, its surface never above the
/// natural ground (`ground`). Deep basins are lakes and are handled apart.
fn fit_falling(raw: &[f32], ground: &[f32], s: &[f64]) -> Vec<f32> {
    let n = raw.len();
    if n == 0 {
        return Vec::new();
    }
    // Fit the profile with its least fall taken out, so the fit only has
    // to be non-increasing.
    let tilt = |i: usize| 4e-4 * s[i] as f32;
    let weight = |i: usize| {
        let before = if i > 0 { s[i] - s[i - 1] } else { 0.0 };
        let after = if i + 1 < n { s[i + 1] - s[i] } else { 0.0 };
        (0.5 * (before + after)).max(1e-3)
    };
    // Blocks of pooled points: (weighted sum, weight, count).
    let mut blocks: Vec<(f64, f64, usize)> = Vec::with_capacity(n);
    for i in 0..n {
        let w = weight(i);
        blocks.push(((raw[i] + tilt(i)) as f64 * w, w, 1));
        while blocks.len() > 1 {
            let last = blocks[blocks.len() - 1];
            let previous = blocks[blocks.len() - 2];
            if previous.0 / previous.1 >= last.0 / last.1 {
                break;
            }
            blocks.pop();
            let merged = blocks.last_mut().unwrap();
            merged.0 += last.0;
            merged.1 += last.1;
            merged.2 += last.2;
        }
    }
    let mut fitted = Vec::with_capacity(n);
    for (sum, w, count) in blocks {
        let value = (sum / w) as f32;
        for _ in 0..count {
            let i = fitted.len();
            fitted.push((value - tilt(i)).min(ground[i]));
        }
    }
    fitted
}

/// A tributary arrives at its parent's level: over its last `JUNCTION_FLAT`
/// metres it stands at it, and above that it may rise at most
/// `JUNCTION_GRADE` per metre, cutting its channel down to it as far as
/// `JUNCTION_CUT` under its own profile, deeper toward the confluence
/// (`JUNCTION_DEEPENING` more per metre over the last `JUNCTION_GORGE`), and
/// not at all past `JUNCTION_REACH`, where its own profile takes over.
const JUNCTION_FLAT: f32 = 6.0;
const JUNCTION_GRADE: f32 = 0.03;
const JUNCTION_CUT: f32 = 2.5;
const JUNCTION_GORGE: f32 = 20.0;
const JUNCTION_DEEPENING: f32 = 0.4;
const JUNCTION_REACH: f32 = 60.0;

/// The least slope a tributary's reach below a lake falls at toward its
/// confluence when it is graded to its parent: steep enough to leave the
/// lake's level within a few channel widths, as an outlet's rapid.
const JUNCTION_OUTLET_RAPID: f32 = 0.06;

/// Grade a tributary's water down to its parent's at their confluence, so
/// the two meet level as real confluences do, the tributary's mouth drowned
/// in its parent's water, rather than the tributary tumbling into the
/// parent over its last metres. Its channel is cut down into the ground for
/// it, a gorge where it comes down a steep valley side; farther up the cut
/// is held to `JUNCTION_CUT` and the steeper water stays there, as rapids.
/// A lake's water keeps its level: below the last lake above the
/// confluence (`lake`, its last point and level) the reach runs down from
/// the lake's level as an even rapid where it is too short to fall at
/// `JUNCTION_GRADE`, rather than dropping off the lake's edge as a fall.
fn grade_to_parent(water: &mut [f32], s: &[f64], floor: &[f32], lake: Option<(usize, f32)>, level: f32) {
    let n = water.len();
    let end = s[n - 1];
    let first = lake.map_or(0, |(last, _)| last + 1);
    let before = water.to_vec();
    for i in first..n {
        let run = (end - s[i]) as f32;
        let chord = level + JUNCTION_GRADE * (run - JUNCTION_FLAT).max(0.0);
        let cut = (JUNCTION_CUT + JUNCTION_DEEPENING * (JUNCTION_GORGE - run).max(0.0))
            * (1.0 - smoothstep(JUNCTION_GORGE, JUNCTION_REACH, run));
        water[i] = water[i].min(chord.max(water[i] - cut)).max(level).max(floor[i]);
    }
    let Some((last, lake_level)) = lake else {
        return;
    };
    let room = (end - s[last]) as f32 - JUNCTION_FLAT;
    if room <= 0.0 {
        return;
    }
    let rapid = ((lake_level - level) / room).max(JUNCTION_OUTLET_RAPID);
    for i in first..n {
        let line = lake_level - rapid * (s[i] - s[last]) as f32;
        water[i] = water[i].max(line.min(before[i]));
    }
}

/// The point of a river's centreline nearest `p`: how far `p` lies from it,
/// and the river there, its nodes interpolated along the centreline.
pub(super) fn nearest_node(river: &River, p: [f64; 2]) -> (f32, RiverNode) {
    let mut best = (f64::INFINITY, 0usize, 0.0f64);
    for (k, pair) in river.nodes.windows(2).enumerate() {
        let (a, b) = (&pair[0], &pair[1]);
        let pa = [a.position[0] as f64, a.position[1] as f64];
        let ab = [b.position[0] as f64 - pa[0], b.position[1] as f64 - pa[1]];
        let t = (((p[0] - pa[0]) * ab[0] + (p[1] - pa[1]) * ab[1]) / (ab[0] * ab[0] + ab[1] * ab[1]).max(1e-9)).clamp(0.0, 1.0);
        let d = distance(p, [pa[0] + ab[0] * t, pa[1] + ab[1] * t]);
        if d < best.0 {
            best = (d, k, t);
        }
    }
    let Some(a) = river.nodes.get(best.1) else {
        return (f32::INFINITY, RiverNode::default());
    };
    let b = river.nodes.get(best.1 + 1).unwrap_or(a);
    let t = best.2 as f32;
    let mix = |x: f32, y: f32| x + (y - x) * t;
    let node = RiverNode {
        position: [mix(a.position[0], b.position[0]), mix(a.position[1], b.position[1])],
        water: mix(a.water, b.water),
        half_width: mix(a.half_width, b.half_width),
        depth: mix(a.depth, b.depth),
        speed: mix(a.speed, b.speed),
        discharge: mix(a.discharge, b.discharge),
        area: mix(a.area, b.area),
        bank: mix(a.bank, b.bank),
        skew: mix(a.skew, b.skew),
        turbulence: mix(a.turbulence, b.turbulence),
        slope: mix(a.slope, b.slope),
        along: mix(a.along, b.along),
        lake: a.lake && b.lake,
    };
    (best.0 as f32, node)
}

/// How far `p` lies past a river's waterline (negative inside its channel),
/// and the river's water and half width beside it, interpolated along its
/// centreline.
pub(super) fn beside_river(river: &River, p: [f64; 2]) -> (f32, f32, f32) {
    let (distance, node) = nearest_node(river, p);
    (distance - node.half_width, node.water, node.half_width)
}

/// Where a tributary's water joins its parent's: the first point of the
/// last run of its course that lies within reach of the parent's waterline
/// (as `joins_early` reaches a river), and the parent's water beside every
/// point. A tributary that meets a steep parent at a narrow angle runs
/// beside it for some metres before it reaches its thalweg, past water that
/// stands higher than where it ends.
fn parent_contact(points: &[PathPoint], parent: &River) -> (usize, Vec<f32>) {
    let beside: Vec<(f32, f32, f32)> = points.iter().map(|point| beside_river(parent, point.p)).collect();
    let past: Vec<f32> = beside.iter().map(|b| b.0).collect();
    let areas: Vec<f32> = points.iter().map(|point| point.area as f32).collect();
    let lakes: Vec<bool> = points.iter().map(|point| point.lake.is_finite()).collect();
    (first_contact(&past, &areas, &lakes), beside.iter().map(|b| b.1).collect())
}

/// The first point of the last run of a tributary's course within reach of
/// its parent's waterline, from how far past it each point lies, its
/// catchment there and whether it lies in a lake. The run never reaches
/// back into a lake: a lake keeps its own level, and the tributary meets its
/// parent below it.
pub(super) fn first_contact(past: &[f32], areas: &[f32], lakes: &[bool]) -> usize {
    let reached = |i: usize| past[i] <= half_width(areas[i]) + JOIN_REACH as f32 && !lakes[i];
    let mut contact = past.len() - 1;
    while contact > 0 && reached(contact - 1) {
        contact -= 1;
    }
    contact
}

/// Metres up a tributary from where it reaches its parent over which its
/// water takes on the parent's (at least six of its own widths).
const MERGE_REACH: f32 = 12.0;

/// Where a tributary reaches its parent (`first_contact`), and how far its
/// water has become the parent's at each node, 0 to 1: none above
/// `MERGE_REACH` before there, all of it from there on, never in a lake.
pub(super) fn merge_weights(nodes: &[RiverNode], parent: &River) -> (usize, Vec<f32>) {
    if nodes.is_empty() {
        return (0, Vec::new());
    }
    let past: Vec<f32> = nodes.iter().map(|n| beside_river(parent, [n.position[0] as f64, n.position[1] as f64]).0).collect();
    let areas: Vec<f32> = nodes.iter().map(|n| n.area).collect();
    let lakes: Vec<bool> = nodes.iter().map(|n| n.lake).collect();
    let contact = first_contact(&past, &areas, &lakes);
    let at = nodes[contact].along;
    let reach = MERGE_REACH.max(6.0 * nodes[contact].half_width);
    (contact, nodes.iter().map(|n| if n.lake { 0.0 } else { smoothstep(at - reach, at, n.along) }).collect())
}

/// Where a tributary runs into its parent, its water becomes the parent's:
/// over its last metres its current comes to the parent's speed beside it,
/// and its bed deepens to meet the parent's rather than hanging over it, a
/// step down the parent's channel wall. Its whitewater stays its own to the
/// parent's waterline (a calm creek stays calm across its own mouth) and
/// becomes the parent's halfway in to its centre, as the drawn surface does
/// (`surface::merge_into`), so the bed its course carves through the
/// parent's channel is the parent's rock.
fn merge_with_parent(nodes: &mut [RiverNode], parent: &River) {
    let (_, weights) = merge_weights(nodes, parent);
    for (node, merge) in nodes.iter_mut().zip(weights) {
        if merge <= 0.0 {
            continue;
        }
        let (distance, there) = nearest_node(parent, [node.position[0] as f64, node.position[1] as f64]);
        node.speed += (there.speed - node.speed) * merge;
        let inside = smoothstep(0.0, 0.5, 1.0 - distance / there.half_width.max(0.05));
        node.turbulence += (there.turbulence - node.turbulence) * inside * merge;
        // The parent's bed under the node, a flat-bottomed bowl as carved.
        let u = (distance / there.half_width.max(0.05)).min(1.0);
        let bed = there.depth * (1.0 - u * u * u * u);
        node.depth += (node.depth.max(bed) - node.depth) * merge;
    }
}

/// From where a tributary reaches its parent on, its surface is the
/// parent's beside it (a lake's water, at its end, keeps its own level).
fn meet_parent(water: &mut [f32], points: &[PathPoint], contact: usize, beside: &[f32]) {
    for i in contact..water.len() {
        if !points[i].lake.is_finite() {
            water[i] = beside[i];
        }
    }
}

/// The water surface's slope at each point over a few channel widths.
fn reach_slopes(points: &[PathPoint], s: &[f64], water: &[f32]) -> Vec<f32> {
    let n = water.len();
    let slope_at = |i: usize| -> f32 {
        let width = 2.0 * half_width(points[i].area as f32) as f64;
        let span = (width * 4.0).max(24.0);
        let a = s.partition_point(|&v| v < s[i] - span).min(n - 1);
        let b = s.partition_point(|&v| v <= s[i] + span).saturating_sub(1).max(a);
        if b > a { ((water[a] - water[b]) / (s[b] - s[a]) as f32).max(4e-4) } else { 4e-4 }
    };
    (0..n).map(slope_at).collect()
}

/// How widely, in nodes, the water surface is smoothed down a steep reach,
/// and the least share of the reach's slope its water falls by there.
const STEEP_SMOOTHING: f64 = 5.0;
const STEEP_FALL: f32 = 0.35;

/// The least a smoothed water surface keeps under the ground beside it.
const SMOOTHING_FREEBOARD: f32 = 0.1;

/// Water surface along a path: below the banks, never rising downstream.
/// A river running to the sea (`sea`) grades down to it over its lowest
/// reach.
fn water_profile(noise: &NoiseField, points: Vec<PathPoint>, parent: Option<&River>, sea: bool) -> Profiled {
    let n = points.len();
    let s = arc_lengths(&points);
    let ground: Vec<f32> = points.iter().map(|p| base_height(noise, p.p[0] as f32, p.p[1] as f32)).collect();
    // A tributary meets its parent's water where it first comes within reach
    // of its channel, at the level the parent stands at there, and from
    // there on its surface is the parent's beside it: down a steep parent,
    // a tributary graded to where its course ends, on the parent's thalweg,
    // would run in under the parent's higher water where the two channels
    // already lie open to each other.
    let junction = parent.map(|parent| parent_contact(&points, parent));
    // The lower of the ground at its banks and at its centre: across a
    // hillside a channel's downhill bank lies under the ground at its
    // centre, and water a hand under the centre would stand over it, held
    // in only by an embankment. A stream there cuts into the slope.
    let directions = tangents(&points);
    let banks: Vec<f32> = points
        .iter()
        .zip(&directions)
        .zip(&ground)
        .map(|((point, t), &centre)| {
            let across = half_width(point.area as f32) as f64;
            let side = |sign: f64| {
                base_height(noise, (point.p[0] - t[1] * across * sign) as f32, (point.p[1] + t[0] * across * sign) as f32)
            };
            centre.min(side(1.0)).min(side(-1.0))
        })
        .collect();
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
    // Over its first metres below a lake, the outlet's surface draws down
    // from the lake's level over the sill, where the channel is cut only a
    // little into the rim, and settles under its banks downstream. It never
    // stands within `OUTLET_FREEBOARD` of the ground beside it: water level
    // with the land has no bank to end on, and its edge would wander
    // wherever the ground grazes it.
    let mut outlet: Vec<Option<(f32, f32)>> = vec![None; n];
    let mut above: Option<(f32, f64)> = None;
    for i in 0..n {
        if points[i].lake.is_finite() {
            above = Some((points[i].lake as f32, s[i]));
        } else if let Some((level, left)) = above {
            let run = (s[i] - left) as f32;
            if run <= OUTLET_REACH {
                outlet[i] = Some((level, run));
            }
        }
    }
    // Where a river to the sea reaches it: the first point at the sea's
    // level in its final run over ground no higher than a bar.
    let shore = if sea {
        // Past the last lake: a coastal lagoon's bed is not the shore.
        let past_lakes = (0..n).rev().find(|&i| points[i].lake.is_finite()).map_or(0, |i| i + 1);
        let above_the_bar = (0..n).rev().find(|&i| ground[i] >= SEA_LEVEL + MOUTH_BAR);
        let run = above_the_bar.map_or(0, |i| i + 1).max(past_lakes);
        (run..n).find(|&i| ground[i] < SEA_LEVEL + 0.25)
    } else {
        None
    };
    let mut water = vec![0.0f32; n];
    let mut slope = vec![0.01f32; n];
    // Two passes: the bank height depends on the depth, which depends on the
    // slope of the surface the first pass found.
    for pass in 0..2 {
        let bank: Vec<f32> = (0..n)
            .map(|i| {
                if pass == 0 {
                    0.3
                } else {
                    // A stream runs a little under its banks, not in a ditch:
                    // the banks stand a hand or two above the water, more
                    // beside a deeper channel.
                    let (_, depth, _) = channel(discharge(points[i].area as f32), slope[i]);
                    (0.1 + 0.15 * depth).min(0.5)
                }
            })
            .collect();
        let raw: Vec<f32> = (0..n).map(|i| banks[i] - bank[i]).collect();
        let fitted = fit_falling(&raw, &banks, &s);
        for i in 0..n {
            let bank = bank[i];
            water[i] = if i == 0 {
                fitted[0]
            } else {
                fitted[i].min(water[i - 1] - 4e-4 * (s[i] - s[i - 1]) as f32)
            };
            water[i] = if points[i].lake.is_finite() {
                points[i].lake as f32
            } else {
                let held = outlet[i].map_or(f32::NEG_INFINITY, |(level, run)| {
                    let freeboard = OUTLET_FREEBOARD + (bank - OUTLET_FREEBOARD).max(0.0) * (run / OUTLET_REACH);
                    (level - outlet_drawdown(run)).min(banks[i] - freeboard)
                });
                water[i].max(floor[i]).max(held)
            };
        }
        if let Some((contact, beside)) = &junction {
            // A tributary meets its parent's surface; where it would arrive
            // lower it is held up to it (the parent backs it up).
            let level = beside[*contact];
            for w in water[..*contact].iter_mut() {
                *w = w.max(level);
            }
            meet_parent(&mut water, &points, *contact, beside);
        }
        for w in water.iter_mut() {
            *w = w.max(SEA_LEVEL + 0.02);
        }
        if let Some(shore) = shore {
            grade_to_sea(&mut water, &s, &floor, shore);
        }
        slope = reach_slopes(&points, &s, &water);
    }
    // Smooth the surface; a positive kernel keeps it falling downstream.
    // Smoothing rounds a steep drop off on both sides, lifting the water at
    // its foot, but never above the ground there: water above its banks
    // stood on an embankment over the slope a creek tumbles into a lake by.
    // Down a steep reach the fit falls in steps, flat wherever the ground
    // rose a little and steep between: smoothed over a few channel widths
    // there, the water falls evenly down the slope, as a mountain stream
    // does, and its channel and banks follow it instead of terracing the
    // hillside. Below a lake's outlet its drawdown keeps its shape.
    // Smoothing may lift the water toward the ground, but only under the
    // lowest ground it has passed over the last few channel widths: lifted
    // to just under each little rise, it would trace the rises and leave the
    // falling surface a flight of flat steps between them. Below a cliff
    // the fit drops off, the ground just passed stands high, and the water
    // is lifted to round the drop off into a rapid.
    let values: Vec<f64> = water.iter().map(|&w| w as f64).collect();
    let smoothed = smooth_values(&values, 1.5);
    let even = smooth_falling(&values, STEEP_SMOOTHING);
    // How far to blend toward the even surface: each point's steepness
    // spread over the span the wide smoothing reaches (the steepest within
    // it, then smoothed), so the blend changes as gradually as the two
    // surfaces do; a weight switching between neighbours, steep in a drop
    // and gentle just past it, would leave a fall where it switched. An
    // outlet's drawdown, and the reach just past it, keep their shape.
    let span = (2.0 * STEEP_SMOOTHING).ceil() as usize;
    let widest = |values: &[f64]| -> Vec<f64> {
        let spread: Vec<f64> = (0..n)
            .map(|i| values[i.saturating_sub(span)..(i + span + 1).min(n)].iter().fold(0.0, |a: f64, &b| a.max(b)))
            .collect();
        smooth_values(&spread, STEEP_SMOOTHING)
    };
    let steep_weights: Vec<f64> = slope.iter().map(|&v| smoothstep(0.03, 0.12, v) as f64).collect();
    let steepness = widest(&steep_weights);
    let outlet_weights: Vec<f64> = outlet.iter().map(|o| if o.is_some() { 1.0 } else { 0.0 }).collect();
    let held = widest(&outlet_weights);
    for i in 0..n.saturating_sub(1) {
        if i == 0 {
            continue;
        }
        let lowest = banks[i.saturating_sub(span)..=i].iter().fold(f32::INFINITY, |a, &g| a.min(g)) - SMOOTHING_FREEBOARD;
        let steep = if outlet[i].is_some() { 0.0 } else { steepness[i] * (1.0 - held[i]) };
        let level = smoothed[i] + (even[i] - smoothed[i]) * steep;
        let cap = water[i].max(lowest);
        water[i] = (level as f32).min(cap).max(water[n - 1]);
    }
    // Down a steep reach the water keeps falling: across a bench in the
    // hillside it cuts its way down at a share of the reach's slope rather
    // than lying flat behind the bench's lip, a step the channel and its
    // banks would carve into the slope.
    for i in 1..n {
        if points[i].lake.is_finite() || points[i - 1].lake.is_finite() || outlet[i].is_some() {
            continue;
        }
        let fall = STEEP_FALL * slope[i] * smoothstep(0.04, 0.12, slope[i]);
        water[i] = water[i].min(water[i - 1] - fall * (s[i] - s[i - 1]) as f32);
    }
    for w in water.iter_mut() {
        *w = w.max(SEA_LEVEL + 0.02);
    }
    for i in 0..n {
        water[i] = if points[i].lake.is_finite() { points[i].lake as f32 } else { water[i].max(floor[i]) };
    }
    if let Some(shore) = shore {
        grade_to_sea(&mut water, &s, &floor, shore);
    }
    if let Some((contact, beside)) = &junction {
        let contact = *contact;
        let lake = (0..=contact).rev().find(|&i| points[i].lake.is_finite()).map(|i| (i, points[i].lake as f32));
        grade_to_parent(&mut water[..=contact], &s[..=contact], &floor[..=contact], lake, beside[contact]);
        meet_parent(&mut water, &points, contact, beside);
    }
    for i in 1..n {
        water[i] = water[i].min(water[i - 1]);
    }
    // The reach slope of the surface as it finally stands: graded down to
    // the sea or a parent, a river runs calm into it.
    let slope = reach_slopes(&points, &s, &water);
    Profiled { points, s, water, slope }
}

// ---------------------------------------------------------------------------
// Assembly
// ---------------------------------------------------------------------------

/// Node spacing for a channel of this half width.
fn node_spacing(half_width: f32) -> f64 {
    (half_width as f64 * 1.5).clamp(2.6, 7.0)
}

/// How deep a lake's water must stand over the ground at a node of the river
/// crossing it for the node to be in the lake at an end of its crossing.
const LAKE_EDGE_DEPTH: f64 = 0.1;

/// A lake takes its 4 m cells whole, by the ground at their centres, so
/// along a shallow margin a river's course can run on through the lake's
/// cells over ground the water does not cover, the ground rising a little
/// over the lake's level between centres. Such a reach at the end of a
/// crossing is not in the lake: below the lake it is the outlet's channel,
/// cut down through the sill to the water, and above it the inlet's, cut down
/// to meet it. Left in the lake, an outlet would begin as a trench in dry
/// ground, cut off from the water by a strip of shore. A crossing now ends
/// where two nodes in a row stand in the water, so the outlet's submerged
/// connector has water to follow back into the lake; one whose water never
/// does is left as it is: a brush with the shore.
fn trim_dry_lake_ends(noise: &NoiseField, points: &mut [PathPoint]) {
    let n = points.len();
    let deep: Vec<bool> = points
        .iter()
        .map(|point| (base_height(noise, point.p[0] as f32, point.p[1] as f32) as f64) < point.lake - LAKE_EDGE_DEPTH)
        .collect();
    let mut i = 0;
    while i < n {
        if !points[i].lake.is_finite() {
            i += 1;
            continue;
        }
        let first = i;
        while i + 1 < n && points[i + 1].lake.is_finite() {
            i += 1;
        }
        let last = i;
        i += 1;
        let Some(inner) = (first..last).find(|&k| deep[k] && deep[k + 1]) else {
            continue;
        };
        let outer = (first..last).rev().find(|&k| deep[k] && deep[k + 1]).unwrap_or(inner);
        if last + 1 < n {
            for point in &mut points[outer + 2..=last] {
                point.lake = f64::NEG_INFINITY;
            }
        }
        if first > 0 {
            for point in &mut points[first..inner] {
                point.lake = f64::NEG_INFINITY;
            }
        }
    }
}

fn build_nodes(
    noise: &NoiseField,
    centreline: Vec<PathPoint>,
    parent: Option<&River>,
    seed: u64,
    sea: bool,
    lake_at: impl Fn([f64; 2]) -> f64,
) -> Vec<RiverNode> {
    let mut centreline = resample(&centreline, |p| node_spacing(half_width(p.area as f32)));
    // A node lies in a lake exactly where its own flow cell does, so a
    // river's reach through a lake begins and ends at the lake's edge
    // rather than up to half a node off it.
    for point in centreline.iter_mut() {
        point.lake = lake_at(point.p);
    }
    trim_dry_lake_ends(noise, &mut centreline);
    // Area never shrinks downstream.
    for i in 1..centreline.len() {
        centreline[i].area = centreline[i].area.max(centreline[i - 1].area);
    }
    if centreline.len() < 2 {
        return Vec::new();
    }
    let mut profile = water_profile(noise, centreline, parent, sea);
    // A spring in a hollow under the level of the lake, sea or river its
    // water runs to has its first reach held up at that level, over its own
    // ground: the stream rises where its water first lies under the ground,
    // and the hollow above stays dry rather than holding a trickle over it.
    if !profile.points[0].lake.is_finite() {
        let n = profile.points.len();
        let is_covered = |&i: &usize| -> bool {
            let p = profile.points[i].p;
            profile.points[i].lake.is_finite()
                || profile.water[i] <= base_height(noise, p[0] as f32, p[1] as f32) - SPRING_FREEBOARD
        };
        let under = (0..n).find(is_covered);
        if let Some(start) = under.filter(|&start| start > 0 && n - start >= 2) {
            profile.points.drain(..start);
            profile.water.drain(..start);
            profile.slope.drain(..start);
            profile.s.drain(..start);
            let first = profile.s[0];
            profile.s.iter_mut().for_each(|v| *v -= first);
        }
    }
    let node_at = |i: usize| -> RiverNode {
        let point = &profile.points[i];
        RiverNode {
            position: [point.p[0] as f32, point.p[1] as f32],
            water: profile.water[i],
            half_width: half_width(point.area as f32),
            depth: 0.0,
            speed: 0.0,
            discharge: discharge(point.area as f32),
            area: point.area as f32,
            bank: 0.0,
            skew: 0.0,
            turbulence: 0.0,
            slope: profile.slope[i],
            along: profile.s[i] as f32,
            lake: point.lake.is_finite(),
        }
    };
    let mut nodes: Vec<RiverNode> = (0..profile.points.len()).map(node_at).collect();

    // Hydraulics and whitewater. A channel answers to the slope of its
    // valley over a long reach, not to every steeper or gentler stretch, so
    // it widens and narrows gradually.
    let count = nodes.len();
    let mean_spacing = ((nodes[count - 1].along - nodes[0].along) as f64 / (count - 1).max(1) as f64).max(0.5);
    let node_slopes: Vec<f64> = nodes.iter().map(|node| node.slope as f64).collect();
    let reach_slope = smooth_values(&node_slopes, (40.0 / mean_spacing).clamp(2.0, 20.0));
    for (node, &reach) in nodes.iter_mut().zip(&reach_slope) {
        let (half_width, depth, _) = channel(node.discharge, reach as f32);
        node.half_width = half_width;
        node.depth = depth;
        // Rapids churn white as the slope steepens.
        let cascade = smoothstep(0.02, 0.12, node.slope);
        node.turbulence = (0.15 * smoothstep(0.004, 0.02, node.slope) + 0.6 * cascade).min(1.0);
        // Lowland banks are low and grassy; mountain channels cut steeper
        // banks into stony ground.
        node.bank = 0.55 + 0.45 * smoothstep(0.003, 0.08, node.slope);
    }
    // Heads begin as a seep that gathers into a channel: a trickle at the
    // spring, a runnel some tens of metres on. A river that rises in a lake
    // is its outlet, full from the start.
    let from_spring = !nodes[0].lake;
    let head_length = HEAD_GATHER;
    let head_start = if from_spring { 0.15 } else { 1.0 };
    for node in nodes.iter_mut() {
        let grow = head_start + (1.0 - head_start) * smoothstep(0.0, head_length, node.along);
        node.half_width *= grow;
        node.depth *= grow;
        // ...carrying the water the seep has gathered so far.
        node.discharge *= grow * grow;
    }
    // The reach's own section and the speed its water runs at, before
    // pools and riffles vary them.
    let reach_section: Vec<f32> = nodes.iter().map(|n| n.half_width * n.depth).collect();
    let reach_speed: Vec<f32> = nodes.iter().map(|n| flow_speed(n.depth / THALWEG_OVER_MEAN, n.slope)).collect();
    let curvature = smooth_values(&signed_curvature(&nodes), 2.0);
    // No channel keeps exactly one width or depth. It swells a little and
    // narrows again, and its banks change from a low shelving edge to a
    // steeper face where roots or a harder layer hold the soil. Its bed
    // undulates: a pool every five to seven widths (Leopold, Wolman and
    // Miller) and on the outside of every tight bend, deep and slow, with a
    // shallow riffle between that the water breaks over. A lake's water has
    // no channel to vary.
    let offset = (seed % 100_000) as f64 * 5.3;
    let mut phase = 4.0 * field_noise([offset, 0.0], 50.0, 317) as f64;
    for i in 0..nodes.len() {
        let width = 2.0 * nodes[i].half_width as f64;
        if i > 0 {
            let w = width.max(3.0);
            let spacing = w * (5.0 + 2.0 * field_noise([nodes[i].along as f64, offset + 91.0], 30.0 * w, 319) as f64);
            phase += (nodes[i].along - nodes[i - 1].along) as f64 / spacing;
        }
        let node = &mut nodes[i];
        if node.lake {
            continue;
        }
        let swell = field_noise([node.along as f64, offset], (10.0 * width).max(16.0), 311);
        node.half_width *= 0.92 + 0.16 * swell;
        // Banks firm up and slacken over many widths, not node to node: each
        // carve segment keeps one bank slope, and neighbours that differed
        // by much would jog the top of the bank in and out at every joint.
        let firmness = field_noise([node.along as f64, offset + 57.0], (12.0 * width).max(24.0), 313);
        node.bank *= 0.85 + 0.3 * firmness;
        let sequence = 0.5 + 0.5 * (phase * std::f64::consts::TAU).cos() as f32;
        let bend = smoothstep(0.1, 0.35, (curvature[i] * width).abs() as f32);
        let pool = sequence.max(bend);
        node.depth *= 0.75 + 0.6 * pool * pool;
        // Riffles break the surface into small standing waves.
        node.turbulence = (node.turbulence + 0.12 * (1.0 - pool) * smoothstep(0.003, 0.015, node.slope)).min(1.0);
    }
    limit_width_change(&mut nodes);
    // The same water runs faster through the narrows and over the riffles
    // and slows through the pools.
    for ((node, &section), &speed) in nodes.iter_mut().zip(&reach_section).zip(&reach_speed) {
        node.speed = (speed * section / (node.half_width * node.depth).max(1e-3)).clamp(0.15, 4.0);
    }
    still_lakes(&mut nodes);
    // A trickle has too little water to churn white: whitewater builds as
    // the stream gathers below its spring.
    if from_spring {
        for node in nodes.iter_mut() {
            node.turbulence *= smoothstep(0.0, HEAD_GATHER, node.along);
        }
    }
    // Skew from the signed curvature: the thalweg hugs the outside of a bend.
    for (node, &k) in nodes.iter_mut().zip(&curvature) {
        let width = 2.0 * node.half_width as f64;
        // Left turn (k > 0): the outer bank is on the right (-).
        node.skew = (-k * width * 1.6).clamp(-0.6, 0.6) as f32;
    }
    nodes
}

/// Metres below its spring over which a stream gathers from a trickle into
/// its channel.
const HEAD_GATHER: f32 = 40.0;
/// How far under the ground beside it a spring's water stands at least.
const SPRING_FREEBOARD: f32 = 0.05;

/// A lake's water drifts at this speed between its inlets and outlets, m/s.
const LAKE_DRIFT: f32 = 0.12;

/// A lake's water is still. No rapids churn in it, whatever the slope of the
/// valley it drowned, and it only drifts toward its outlet, quickening over
/// the last few metres as it draws down to the sill; an inlet's jet runs on
/// a little way into it and slows.
fn still_lakes(nodes: &mut [RiverNode]) {
    let n = nodes.len();
    let mut next_exit = vec![None; n];
    let mut exit: Option<usize> = None;
    for i in (0..n).rev() {
        if !nodes[i].lake {
            exit = Some(i);
        }
        next_exit[i] = exit;
    }
    let mut last_entry: Option<usize> = None;
    let mut entries = vec![None; n];
    for i in 0..n {
        if !nodes[i].lake {
            last_entry = Some(i);
        }
        entries[i] = last_entry;
    }
    let speeds: Vec<f32> = nodes.iter().map(|node| node.speed).collect();
    for i in 0..n {
        if !nodes[i].lake {
            continue;
        }
        let reach = (2.0 * nodes[i].half_width).max(4.0);
        let mut speed = LAKE_DRIFT;
        if let Some(e) = next_exit[i] {
            let t = 1.0 - smoothstep(0.0, reach, nodes[e].along - nodes[i].along);
            speed = speed.max(LAKE_DRIFT + (speeds[e] - LAKE_DRIFT) * t * t);
        }
        if let Some(e) = entries[i] {
            let t = 1.0 - smoothstep(0.0, 2.0 * reach, nodes[i].along - nodes[e].along);
            speed = speed.max(LAKE_DRIFT + (speeds[e] - LAKE_DRIFT) * t);
        }
        nodes[i].speed = speed.min(speeds[i].max(LAKE_DRIFT));
        nodes[i].turbulence = 0.0;
    }
}

/// Most a channel's half width may change per metre along it: a river
/// widens below a confluence or at its mouth over many metres, never in a
/// step.
const WIDTH_CHANGE: f32 = 0.025;

/// Hold the half width to `WIDTH_CHANGE` per metre, by easing it down
/// toward each narrower neighbour, along the river and back.
fn limit_width_change(nodes: &mut [RiverNode]) {
    for i in 1..nodes.len() {
        let run = (nodes[i].along - nodes[i - 1].along).abs();
        nodes[i].half_width = nodes[i].half_width.min(nodes[i - 1].half_width + WIDTH_CHANGE * run);
    }
    for i in (0..nodes.len().saturating_sub(1)).rev() {
        let run = (nodes[i + 1].along - nodes[i].along).abs();
        nodes[i].half_width = nodes[i].half_width.min(nodes[i + 1].half_width + WIDTH_CHANGE * run);
    }
}

/// Longest stretch of a river between two reaches through the same lake that
/// is taken as the lake's own water, and how far under the lake's level the
/// ground along it may lie: the centreline crossing a bar, a low neck of shore
/// or a cell the lake's flood did not reach, never a separate hollow the
/// lake's sheet would not cover.
const LAKE_GAP: f32 = 80.0;
const LAKE_GAP_DIP: f32 = 0.5;
/// A run of lake nodes shorter than this is the river brushing a lake's shore,
/// not running into it: no mouth opens there.
const MIN_CROSSING: f32 = 8.0;
/// How much wider a river opens where it runs into a lake, and where it
/// leaves one over its outlet's sill.
const INLET_SPREAD: f32 = 0.85;
const OUTLET_SPREAD: f32 = 0.7;
const SEA_SPREAD: f32 = 1.2;
/// How far from the still water's level (a lake's or the sea's) a river's
/// water stands where its mouth gives out: fully open within the lower, not
/// at all past the higher.
const MOUTH_DROWNED: f32 = 0.3;
const MOUTH_CLEAR: f32 = 1.2;

#[derive(Clone, Copy)]
enum MouthKind {
    Inlet,
    Outlet,
    Sea,
}

/// One transition between a channel and a larger water body. The reach is
/// measured from the channel's width at the junction, before any widening:
/// changing widths up the river cannot move the start of the transition.
struct Mouth {
    along: f32,
    reach: f32,
    spread: f32,
    kind: MouthKind,
    /// The still water's level: the mouth opens only where the river's
    /// water stands near it.
    level: Option<f32>,
}

impl Mouth {
    fn new(along: f32, half_width: f32, kind: MouthKind, level: Option<f32>) -> Self {
        let (widths, least, spread) = match kind {
            MouthKind::Inlet => (12.0, 28.0, INLET_SPREAD),
            MouthKind::Outlet => (14.0, 32.0, OUTLET_SPREAD),
            MouthKind::Sea => (18.0, 60.0, SEA_SPREAD),
        };
        Self { along, reach: (widths * half_width).max(least), spread, kind, level }
    }

    fn influence(&self, along: f32, water: f32) -> f32 {
        // An estuary stays open on the seaward side. Folding the distance
        // around the shore would pinch it back into a narrow trench offshore.
        let distance = match self.kind {
            MouthKind::Sea => (self.along - along).max(0.0),
            _ => (self.along - along).abs(),
        };
        let t = (1.0 - distance / self.reach).clamp(0.0, 1.0);
        // Zero slope and curvature at both ends: the bank flares gradually
        // out of the ordinary channel and rounds into the shore.
        let flare = t * t * t * (t * (6.0 * t - 15.0) + 10.0);
        // A mouth is the reach drowned in the still water, wide, calm and
        // low-banked: a cascade down into a lake or to the sea, or an
        // outlet's rapid well below its sill, stays a rapid in its own
        // channel between its banks, not a broad, glassy sheet sliding down
        // the hillside over them.
        let drowned = self.level.map_or(1.0, |level| 1.0 - smoothstep(MOUTH_DROWNED, MOUTH_CLEAR, (water - level).abs()));
        flare * drowned
    }
}

/// Open mouths on both sides of a lake edge, so terrain carving and the
/// water ribbon share the same widening. A brief brush with a lake's shore
/// is not a junction. The depth and flow keep the outlet's sill while its
/// banks become broad shoulders instead of a slot cut through the rim.
fn open_mouths(nodes: &mut [RiverNode], end: RiverEnd, surface_end: usize) {
    let mut mouths = Vec::new();
    for (first, last) in lake_crossings(nodes) {
        if first > 0 {
            let (a, b) = (&nodes[first - 1], &nodes[first]);
            let (along, half_width) = (0.5 * (a.along + b.along), 0.5 * (a.half_width + b.half_width));
            mouths.push(Mouth::new(along, half_width, MouthKind::Inlet, Some(b.water)));
        }
        if last + 1 < nodes.len() {
            let (a, b) = (&nodes[last], &nodes[last + 1]);
            let (along, half_width) = (0.5 * (a.along + b.along), 0.5 * (a.half_width + b.half_width));
            mouths.push(Mouth::new(along, half_width, MouthKind::Outlet, Some(a.water)));
        }
    }
    if end == RiverEnd::Sea {
        let at = &nodes[surface_end];
        mouths.push(Mouth::new(at.along, at.half_width, MouthKind::Sea, Some(SEA_LEVEL)));
    }
    for node in nodes {
        let (mut spread, mut weight, mut depth, mut bank, mut calm, mut openness) = (0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0);
        for mouth in &mouths {
            let t = mouth.influence(node.along, node.water);
            let strength = mouth.spread * t;
            let w = strength * strength;
            let (depth_change, bank_slope, calming) = match mouth.kind {
                MouthKind::Inlet => (-0.24, 0.24, 1.0),
                MouthKind::Outlet => (-0.12, 0.28, 0.35),
                MouthKind::Sea => (-0.2, 0.35, 1.0),
            };
            spread = spread.max(strength);
            weight += w;
            depth += w * depth_change * t;
            bank += w * (node.bank.min(bank_slope) - node.bank) * t;
            calm += w * calming * t;
            openness += w * t;
        }
        if weight <= 0.0 {
            continue;
        }
        // Adjacent mouths can overlap around a small pond. Blend their
        // hydraulic changes so switching the dominant flare leaves no step
        // in depth, speed or the bank slope.
        let widen = 1.0 + spread;
        let deepen = 1.0 + depth / weight;
        node.half_width *= widen;
        node.depth *= deepen;
        // A wider cross-section carries the same discharge more slowly;
        // below a pond's sill the narrowing channel accelerates it again.
        node.speed /= widen * deepen;
        node.turbulence *= 1.0 - calm / weight;
        node.bank += bank / weight;
        node.skew *= 1.0 - 0.8 * openness / weight;
    }
}

/// How far past its mouth an estuary's sill is looked for along the course,
/// metres, and the least depth its channel keeps over it.
const ESTUARY_SILL_REACH: f32 = 40.0;
const ESTUARY_LEAST_DEPTH: f32 = 0.25;

/// An estuary's bed falls toward the sea and meets the seabed. At its mouth
/// it is no deeper than the highest ground its water must still cross to
/// reach open water (the bar a beach throws across it, or the shelving
/// seabed), and up its flare it is nowhere deeper than downstream: deepened
/// toward the coast, it was a closed bowl behind the shore, a pool the sea's
/// own floor rose out of beyond the widened channel's round end.
fn grade_estuary(nodes: &mut [RiverNode], surface_end: usize, ground: impl Fn([f32; 2]) -> f32) {
    let mouth = surface_end.min(nodes.len() - 1);
    // From where the course reaches the sea, across whatever it shelves over
    // before open water.
    let Some(shore) = (mouth..nodes.len()).find(|&i| ground(nodes[i].position) < SEA_LEVEL) else {
        return;
    };
    let mut sill = f32::NEG_INFINITY;
    for node in &nodes[shore..] {
        let height = ground(node.position);
        if height < SEA_LEVEL - OPEN_SEA_DEPTH || node.along - nodes[shore].along > ESTUARY_SILL_REACH {
            break;
        }
        sill = sill.max(height);
    }
    if !sill.is_finite() {
        return;
    }
    // Over the flare's reach, which `Mouth` measures from the channel's
    // width before it widened.
    let reach = Mouth::new(nodes[mouth].along, nodes[mouth].half_width / (1.0 + SEA_SPREAD), MouthKind::Sea, None).reach;
    let bed_at = |node: &RiverNode, floor: f32| -> f32 {
        let bed = (node.water - node.depth).max(floor);
        node.water - (node.water - bed).max(ESTUARY_LEAST_DEPTH)
    };
    for node in nodes[mouth..].iter_mut() {
        node.depth = node.water - bed_at(node, sill);
    }
    let start = nodes[mouth].along - reach;
    let mut floor = sill;
    for node in nodes[..mouth].iter_mut().rev() {
        if node.along < start {
            break;
        }
        let bed = bed_at(node, floor);
        node.depth = node.water - bed;
        floor = bed;
    }
}

/// Join a river's reaches through one lake across the short stretches where
/// its centreline leaves the lake's cells: over a low bar, across a corner of
/// the shore, between two cells of a ragged shoreline. Such a stretch already
/// runs at the lake's level (nothing downstream lets it fall lower). Counted
/// as river, it would open two mouths in mid-lake, cut a channel with levees
/// through the lake's bed, and draw a ribbon in the lake's own plane.
fn settle_lake_runs(nodes: &mut [RiverNode], ground: impl Fn([f32; 2]) -> f32) {
    let mut previous: Option<usize> = None;
    for i in 0..nodes.len() {
        if !nodes[i].lake {
            continue;
        }
        if let Some(p) = previous
            && p + 1 < i
        {
            let level = nodes[i].water;
            let same_lake = (nodes[p].water - level).abs() < 0.01;
            let gap = nodes[i].along - nodes[p].along;
            let short = gap <= LAKE_GAP;
            // A gap no longer than a crossing is a ragged edge, whatever
            // the ground does in it; a longer one must stay near the level.
            let shallow = gap <= MIN_CROSSING
                || nodes[p + 1..i].iter().all(|node| ground(node.position) >= level - LAKE_GAP_DIP);
            if same_lake && short && shallow {
                for node in &mut nodes[p + 1..i] {
                    node.lake = true;
                    node.water = level;
                }
            }
        }
        previous = Some(i);
    }
}

/// A river's reaches through lakes, as (first, last) node, each long enough
/// to be a crossing rather than a brush with a shore.
fn lake_crossings(nodes: &[RiverNode]) -> Vec<(usize, usize)> {
    let mut crossings = Vec::new();
    let mut i = 0;
    while i < nodes.len() {
        if !nodes[i].lake {
            i += 1;
            continue;
        }
        let first = i;
        while i + 1 < nodes.len() && nodes[i + 1].lake {
            i += 1;
        }
        let last = i;
        let start = if first > 0 { 0.5 * (nodes[first - 1].along + nodes[first].along) } else { nodes[first].along };
        let end = if last + 1 < nodes.len() { 0.5 * (nodes[last].along + nodes[last + 1].along) } else { nodes[last].along };
        if end - start >= MIN_CROSSING {
            crossings.push((first, last));
        }
        i += 1;
    }
    crossings
}

fn signed_curvature(nodes: &[RiverNode]) -> Vec<f64> {
    let n = nodes.len();
    let curvature_at = |i: usize| -> f64 {
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
    };
    (0..n).map(curvature_at).collect()
}

/// The finished network of one region.
pub struct RiverNetwork {
    pub region: [i64; 2],
    pub rivers: Vec<River>,
    pub segments: Vec<RiverSegment>,
    pub grid: SegmentGrid,
    pub lakes: Vec<Lake>,
    /// The clarity of the lake over each flow cell its sheet covers
    /// (`surface::lake_clarity`), the highest lake's where two meet.
    pub lake_clarity: std::collections::HashMap<[i32; 2], f32>,
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
    let corridor_of = |path: &CellPath| -> Corridor {
        let points: Vec<[f64; 2]> = path.cells.iter().map(|&cell| routing.centre(cell as usize)).collect();
        Corridor::new(noise, &points, FLOW_CORRIDOR, &[])
    };
    let corridors: Vec<Corridor> = parallel_map(&paths, corridor_of);

    // Tributaries read their parent's finished centreline.
    let mut rivers: Vec<Option<River>> = vec![None; paths.len()];
    let mut lakes = Lakes::default();
    // The rivers built so far, node by node, by `JOIN_CELL` cell.
    let mut built: std::collections::HashMap<[i64; 2], Vec<(usize, usize)>> = Default::default();
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
                (RiverEnd::Sea, _) => outlets.sea(corridor, Some(coarse[coarse.len() - 1].p)),
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
                outlets.sea(corridor, None);
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
        let mut end = if to_sea { RiverEnd::Sea } else { path.end };
        let mut parent = parent.filter(|_| !to_sea);
        let Some(flow) = flow else {
            continue;
        };
        flood_backwater(noise, wider.as_ref().unwrap_or(&corridors[index]), &flow, &mut lakes);
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
        // Where its course comes down beside a river already there (two
        // streams settling onto one valley floor), the water has joined it:
        // it ends there as that river's tributary rather than running on in a
        // channel of its own alongside, which would fold the two together.
        if let Some((k, other)) = joins_early(&points, &rivers, &built, index, |p| lakes.level_at(p)) {
            points.truncate(k + 1);
            end = RiverEnd::Confluence(other, 0);
            parent = rivers[other].as_ref().map(|river| (other, river));
        }
        let mut end_level = None;
        let mut graded_to = None;
        if let Some((_, parent_river)) = parent {
            // Run on into the parent's thalweg, and meet its surface there:
            // the water has reached its channel, so that is a step across it,
            // never a cut across country.
            let last = *points.last().unwrap();
            let (node, point) = nearest_on_river(parent_river, last.p);
            if distance(point, last.p) <= parent_river.nodes[node].half_width as f64 + 2.0 * FLOW_CELL {
                points.push(PathPoint { p: point, ..last });
                end_level = Some(parent_river.nodes[node].water);
                graded_to = Some(parent_river);
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
        if !trim_steep_head(noise, &mut centreline) {
            continue;
        }
        // A river running to the sea, or into an estuary at the sea's level,
        // is graded down to it.
        let to_sea = end == RiverEnd::Sea || end_level.is_some_and(|level| level <= SEA_LEVEL + 0.03);
        let mut nodes = build_nodes(noise, centreline, graded_to, path.seed, to_sea, |p| lakes.level_at(p));
        if nodes.len() < 2 {
            continue;
        }
        settle_lake_runs(&mut nodes, |p| base_height(noise, p[0], p[1]));
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
                // The river's own water runs until its surface has come down
                // to the sea's (`grade_to_sea`); from there the sea fills its
                // channel, across the beach and any bar, out to the open
                // water its course ended in. Nothing is cut off: past the
                // shore the course is the mouth's channel.
                let first_at_sea = nodes.iter().position(|node| node.water <= SEA_LEVEL + 0.03);
                surface_end = first_at_sea
                    .map_or(nodes.len() - 1, |first| first.saturating_sub(1))
                    .clamp(1, nodes.len() - 1);
            }
            RiverEnd::Edge => {}
        }
        // A tributary backed up to its parent's level runs calm into it: its
        // mouth is drowned in the parent's water, with no rapids at the join.
        if matches!(end, RiverEnd::Confluence(..)) {
            let level = nodes[nodes.len() - 1].water;
            for node in nodes.iter_mut() {
                let backed = 1.0 - smoothstep(0.02, 0.3, node.water - level);
                node.turbulence *= 1.0 - 0.8 * backed;
            }
        }
        open_mouths(&mut nodes, end, surface_end);
        if end == RiverEnd::Sea {
            grade_estuary(&mut nodes, surface_end, |p| base_height(noise, p[0], p[1]));
        }
        if let RiverEnd::Confluence(parent, _) = end
            && let Some(parent_river) = rivers[parent].as_ref()
        {
            merge_with_parent(&mut nodes, parent_river);
        }
        for (k, node) in nodes.iter().enumerate() {
            let cell = [
                (node.position[0] as f64 / JOIN_CELL).floor() as i64,
                (node.position[1] as f64 / JOIN_CELL).floor() as i64,
            ];
            built.entry(cell).or_default().push((index, k));
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

    let segments = build_segments(&finished, noise, &lakes);
    let origin = routing_origin(region);
    let resolution = ((REGION_SIZE + 2.0 * ROUTING_MARGIN) / GRID_CELL as f64).ceil() as usize;
    let grid_origin = [origin[0] as f32, origin[1] as f32];
    let mut grid = SegmentGrid::build(grid_origin, resolution, &segments);
    let lakes = lakes.lakes;
    let lake_with_surface = |lake: &Lake| -> Lake {
        let mut lake = lake.clone();
        lake.shore = shore_cells(noise, &segments, &grid, &lake);
        lake.edge = sheet_edge(noise, &segments, &grid, &lake);
        lake.current = sheet_current(&segments, &grid, &lake);
        lake
    };
    let lakes: Vec<Lake> = parallel_map(&lakes, lake_with_surface);
    // The lakes' surfaces: a drawn sheet's corner wherever there is one (the
    // highest sheet's where two meet), and past the sheets the highest field.
    let surfaces = parallel_map(&lakes, |lake| lake_surface(noise, &segments, &grid, lake));
    let mut corners: std::collections::HashMap<[i32; 2], f32> = Default::default();
    let mut sheets: std::collections::HashSet<[i32; 2]> = Default::default();
    for (sheet, _) in &surfaces {
        for &(corner, height) in sheet {
            sheets.insert(corner);
            let entry = corners.entry(corner).or_insert(height);
            *entry = entry.max(height);
        }
    }
    for (_, field) in &surfaces {
        for &(corner, height) in field {
            if !sheets.contains(&corner) {
                let entry = corners.entry(corner).or_insert(height);
                *entry = entry.max(height);
            }
        }
    }
    grid.set_lakes(&corners, FLOW_CELL as f32);
    let surface = super::surface::build(&finished, &lakes);
    let mut lake_clarity: std::collections::HashMap<[i32; 2], (f32, f32)> = Default::default();
    for lake in &lakes {
        let area = lake.cells.len() as f32 * (FLOW_CELL * FLOW_CELL) as f32;
        let clarity = super::surface::lake_clarity(lake.level, area);
        for &cell in lake.cells.iter().chain(&lake.shore) {
            let entry = lake_clarity.entry(cell).or_insert((lake.level, clarity));
            if lake.level > entry.0 {
                *entry = (lake.level, clarity);
            }
        }
    }
    let lake_clarity = lake_clarity.into_iter().map(|(cell, (_, clarity))| (cell, clarity)).collect();
    RiverNetwork {
        region,
        rivers: finished,
        segments,
        grid,
        lakes,
        lake_clarity,
        surface,
        build_seconds: start.elapsed().as_secs_f32(),
    }
}

/// The ground around a lake as the rivers carve it, cell by cell (at cell
/// centres) and point by point.
struct LakeGround<'a> {
    noise: &'a NoiseField,
    segments: &'a [RiverSegment],
    grid: &'a SegmentGrid,
    level: f32,
    cells: std::collections::HashMap<[i64; 2], CellGround>,
}

#[derive(Clone, Copy)]
struct CellGround {
    carved: f32,
    /// In a river's channel: whether its water stands at the lake's level or
    /// above it (an inlet, which the lake backs up into) or runs away below
    /// it (the outlet, or a channel beyond the rim).
    channel: Option<bool>,
}

impl<'a> LakeGround<'a> {
    fn new(noise: &'a NoiseField, segments: &'a [RiverSegment], grid: &'a SegmentGrid, level: f32) -> Self {
        Self { noise, segments, grid, level, cells: Default::default() }
    }

    /// The height a sheet's edge must keep under at `p` to stay hidden: a
    /// little under the ground (as the rivers carve it) or, where a river's
    /// ribbon reaches over `p`, just under the river's surface, whichever is
    /// higher. In a channel meeting the lake, the sheet then slips just under
    /// the river's water instead of diving to the bed beneath it, where it
    /// showed as a pit with the ribbon's cut end standing over it.
    fn cover(&self, p: [f32; 2]) -> f32 {
        let envelope = super::carve::envelope_at(self.segments, self.grid, p);
        let ground = envelope.clamp(base_height(self.noise, p[0], p[1])) - SHEET_SINK;
        if envelope.bank_distance < RIBBON_COVER * envelope.half_width {
            ground.max(envelope.water.min(self.level) - SHEET_UNDER_RIBBON)
        } else {
            ground
        }
    }

    fn cell(&mut self, cell: [i64; 2]) -> CellGround {
        let (noise, segments, grid, level) = (self.noise, self.segments, self.grid, self.level);
        let cell_ground = || -> CellGround {
            let c = Corridor::centre(cell);
            let p = [c[0] as f32, c[1] as f32];
            let envelope = super::carve::envelope_at(segments, grid, p);
            let natural = base_height(noise, p[0], p[1]);
            CellGround {
                carved: envelope.clamp(natural),
                channel: (envelope.bank_distance < 0.0).then_some(envelope.water >= level - 0.02),
            }
        };
        *self.cells.entry(cell).or_insert_with(cell_ground)
    }

    fn in_channel(&mut self, cell: [i64; 2]) -> bool {
        self.cell(cell).channel.is_some()
    }

    /// Ground falling away below the water beyond the rim: the outlet's
    /// channel and its banks, or the far side of a narrow rim.
    fn falls_away(&mut self, cell: [i64; 2]) -> bool {
        let ground = self.cell(cell);
        match ground.channel {
            Some(inlet) => !inlet,
            None => ground.carved < self.level,
        }
    }
}

/// Samples across a flow cell when a lake's basin is flooded metre by metre.
const BASIN_SAMPLES_PER_CELL: i64 = 4;
/// How far past its cells a lake's water may spread, in samples: as far as
/// `spill` follows it before calling it a way out, so an arm the lake keeps
/// its level for is flooded to its end.
const BASIN_SPREAD: usize = RIM_REACH as usize;
/// The most samples a hollow beside a lake may hold for it to fill, and how
/// far one is flooded at most to find out.
const BASIN_HOLLOW: usize = 4000;
const BASIN_HOLLOW_LIMIT: usize = 4 * BASIN_HOLLOW;
/// How near a channel running away below a lake's level ground falls away
/// with it, metres: the outlet's banks below the sill.
const RUNAWAY_BANKS: f32 = 6.0;

#[derive(Clone, Copy, PartialEq)]
enum Basin {
    /// Below the lake's level.
    Under,
    /// Below it but falling away: down a channel running away below the
    /// level, or on its banks.
    Runaway,
    /// At or above the level, or in a channel with its own water at it.
    Above,
}

/// The metre samples outside a lake's cells (sample `s` centred at
/// `(s + 0.5)` metres) its water covers: ground below its level joined to
/// its cells, which the flood fill stepped past moving cell to cell, and the
/// closed hollows beside it, which fill to the same level. Never ground in a
/// channel, whose own water is drawn there, nor ground falling away down a
/// channel running away below the level.
fn basin_samples(noise: &NoiseField, segments: &[RiverSegment], grid: &SegmentGrid, lake: &Lake) -> std::collections::HashSet<[i64; 2]> {
    const SIDES: [(i64, i64); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
    let per = BASIN_SAMPLES_PER_CELL;
    let step = FLOW_CELL as f32 / per as f32;
    let cells: std::collections::HashSet<[i64; 2]> = lake.cells.iter().map(|c| [c[0] as i64, c[1] as i64]).collect();
    let in_lake = |s: [i64; 2]| cells.contains(&[s[0].div_euclid(per), s[1].div_euclid(per)]);
    let mut kinds: std::collections::HashMap<[i64; 2], Basin> = Default::default();
    let mut classify = |s: [i64; 2]| -> Basin {
        *kinds.entry(s).or_insert_with(|| {
            let p = [(s[0] as f32 + 0.5) * step, (s[1] as f32 + 0.5) * step];
            let envelope = super::carve::envelope_at(segments, grid, p);
            let runaway = envelope.water < lake.level - 0.02;
            if envelope.bank_distance < 0.0 {
                return if runaway { Basin::Runaway } else { Basin::Above };
            }
            let ground = envelope.clamp(base_height(noise, p[0], p[1]));
            match (ground < lake.level, runaway && envelope.bank_distance < RUNAWAY_BANKS) {
                (false, _) => Basin::Above,
                (true, true) => Basin::Runaway,
                (true, false) => Basin::Under,
            }
        })
    };
    // Out from the samples of the lake's edge cells.
    let mut edge: Vec<[i64; 2]> = Vec::new();
    for c in &cells {
        if NEIGHBOURS.iter().any(|&(dx, dz)| !cells.contains(&[c[0] + dx, c[1] + dz])) {
            for j in 0..per {
                for i in 0..per {
                    edge.push([c[0] * per + i, c[1] * per + j]);
                }
            }
        }
    }
    let mut wet: std::collections::HashSet<[i64; 2]> = Default::default();
    let mut frontier = edge.clone();
    for _ in 0..BASIN_SPREAD {
        let mut next = Vec::new();
        for s in frontier {
            for (dx, dz) in SIDES {
                let t = [s[0] + dx, s[1] + dz];
                if !in_lake(t) && !wet.contains(&t) && classify(t) == Basin::Under {
                    wet.insert(t);
                    next.push(t);
                }
            }
        }
        if next.is_empty() {
            break;
        }
        frontier = next;
    }
    // Closed hollows within two metres of the water fill to its level: each
    // a patch of ground below it that nowhere runs away and is not too big.
    // Each is flooded whole; a patch found open marks all of it so, and any
    // later patch that reaches it is open too.
    let mut done: std::collections::HashSet<[i64; 2]> = Default::default();
    let mut open_ground: std::collections::HashSet<[i64; 2]> = Default::default();
    let shore: Vec<[i64; 2]> = edge.iter().chain(wet.iter()).copied().collect();
    let mut filled = Vec::new();
    for s in shore {
        for dz in -2..=2 {
            for dx in -2..=2 {
                let start = [s[0] + dx, s[1] + dz];
                if in_lake(start) || wet.contains(&start) || done.contains(&start) || classify(start) != Basin::Under {
                    continue;
                }
                done.insert(start);
                let mut hollow = vec![start];
                let mut open = false;
                let mut i = 0;
                while i < hollow.len() && hollow.len() <= BASIN_HOLLOW_LIMIT {
                    let h = hollow[i];
                    i += 1;
                    for (ex, ez) in SIDES {
                        let t = [h[0] + ex, h[1] + ez];
                        if in_lake(t) || wet.contains(&t) {
                            continue;
                        }
                        if open_ground.contains(&t) {
                            open = true;
                            continue;
                        }
                        match classify(t) {
                            Basin::Under => {
                                if done.insert(t) {
                                    hollow.push(t);
                                }
                            }
                            Basin::Runaway => open = true,
                            Basin::Above => {}
                        }
                    }
                }
                open |= hollow.len() > BASIN_HOLLOW;
                if open {
                    open_ground.extend(hollow);
                } else {
                    filled.extend(hollow);
                }
            }
        }
    }
    wet.extend(filled);
    wet
}

/// The cells around a lake its sheet reaches under: every cell its water
/// covers past those the flood fill took (`basin_samples`), and up its
/// shore for a few cells, over ground standing at or above the water, so
/// the terrain itself draws the shoreline wherever it meets the water. Never
/// a cell beside ground that falls below the water again beyond the rim,
/// where the sheet would hang in the air, nor over a river's channel, whose
/// own surface is drawn there.
fn shore_cells(noise: &NoiseField, segments: &[RiverSegment], grid: &SegmentGrid, lake: &Lake) -> Vec<[i32; 2]> {
    let mut ground = LakeGround::new(noise, segments, grid, lake.level);
    let mut wet: std::collections::HashSet<[i64; 2]> =
        lake.cells.iter().map(|c| [c[0] as i64, c[1] as i64]).collect();
    let mut shore = Vec::new();
    for sample in basin_samples(noise, segments, grid, lake) {
        let per = BASIN_SAMPLES_PER_CELL;
        let cell = [sample[0].div_euclid(per), sample[1].div_euclid(per)];
        if wet.insert(cell) {
            shore.push([cell[0] as i32, cell[1] as i32]);
        }
    }
    // Up the shore.
    let mut seen = wet.clone();
    let mut frontier: Vec<[i64; 2]> = wet.iter().copied().collect();
    for _ in 0..SHORE_CELLS {
        let mut next = Vec::new();
        for cell in frontier {
            for (dx, dz) in NEIGHBOURS {
                let candidate = [cell[0] + dx, cell[1] + dz];
                if !seen.insert(candidate) || ground.in_channel(candidate) || ground.falls_away(candidate) {
                    continue;
                }
                let faces_lower = NEIGHBOURS.iter().any(|&(ex, ez)| {
                    let beyond = [candidate[0] + ex, candidate[1] + ez];
                    !wet.contains(&beyond) && ground.falls_away(beyond)
                });
                if faces_lower {
                    continue;
                }
                shore.push([candidate[0] as i32, candidate[1] as i32]);
                next.push(candidate);
            }
        }
        frontier = next;
    }
    shore
}

/// A lake's surface height at flow-grid corners (corner `c` at `c * FLOW_CELL`).
type LakeCorners = Vec<([i32; 2], f32)>;

/// A lake's surface at the flow-grid corners it reaches (corner `c` at
/// `c * FLOW_CELL`), for the terrain, the plants and the player to measure
/// the water and the shore by. Over the sheet it is the sheet, corner for
/// corner, sunk edges and all, so ground under it is exactly the ground the
/// drawn water covers. Past the sheet it runs on as far as
/// `LAKE_FIELD_REACH`: at the lake's level over ground standing
/// `LAKE_FIELD_RISE` above it or more, so a shore band measured from it
/// carries on up the bank as though the sheet went on, never stepping where
/// the sheet's cells end, but `LAKE_FIELD_SINK` under ground below the
/// level, where no water stands (beyond the rim, down an outlet's channel),
/// blending between, so it always lies half a metre or more under the
/// ground; and it sinks away from the sheet past `LAKE_FIELD_FADE_START`,
/// so the band has faded out wherever the field ends. Nowhere does it end on
/// an edge.
fn lake_surface(
    noise: &NoiseField,
    segments: &[RiverSegment],
    grid: &SegmentGrid,
    lake: &Lake,
) -> (LakeCorners, LakeCorners) {
    let cell = FLOW_CELL as f32;
    let sunk: std::collections::HashMap<[i32; 2], f32> = lake.edge.iter().copied().collect();
    let mut surface: std::collections::HashMap<[i32; 2], f32> = Default::default();
    for &[x, z] in lake.cells.iter().chain(&lake.shore) {
        for corner in [[x, z], [x + 1, z], [x, z + 1], [x + 1, z + 1]] {
            surface.insert(corner, sunk.get(&corner).map_or(lake.level, |&h| h.min(lake.level)));
        }
    }
    // Out from the sheet ring by ring, each corner keeping the sheet corner
    // nearest it, which it measures its distance from.
    let reach = (LAKE_FIELD_REACH / cell).ceil() as i32 + 1;
    let mut nearest: std::collections::HashMap<[i32; 2], [i32; 2]> = surface.keys().map(|&c| (c, c)).collect();
    let mut frontier: Vec<[i32; 2]> = surface.keys().copied().collect();
    let mut field = Vec::new();
    for _ in 0..reach {
        let mut next: std::collections::HashMap<[i32; 2], [i32; 2]> = Default::default();
        for corner in &frontier {
            let seed = nearest[corner];
            for (dx, dz) in NEIGHBOURS {
                let candidate = [corner[0] + dx as i32, corner[1] + dz as i32];
                if nearest.contains_key(&candidate) {
                    continue;
                }
                // Nearest seed, ties broken by the seed itself so the field
                // does not depend on the order the map is walked in.
                let key = |s: [i32; 2]| (((candidate[0] - s[0]).pow(2) + (candidate[1] - s[1]).pow(2)), s);
                let entry = next.entry(candidate).or_insert(seed);
                if key(seed) < key(*entry) {
                    *entry = seed;
                }
            }
        }
        frontier = next.keys().copied().collect();
        for (&corner, &seed) in &next {
            nearest.insert(corner, seed);
            let distance = ((corner[0] - seed[0]) as f32).hypot((corner[1] - seed[1]) as f32) * cell;
            if distance > LAKE_FIELD_REACH {
                continue;
            }
            let p = [corner[0] as f32 * cell, corner[1] as f32 * cell];
            let envelope = super::carve::envelope_at(segments, grid, p);
            let ground = envelope.clamp(base_height(noise, p[0], p[1]));
            let low = 1.0 - smoothstep(lake.level, lake.level + LAKE_FIELD_RISE, ground);
            let mut height = lake.level - (lake.level - ground + LAKE_FIELD_SINK) * low
                - LAKE_FIELD_FADE * (distance - LAKE_FIELD_FADE_START).max(0.0);
            // A channel running away below the level between this corner
            // and the next (an outlet, cut down past its sill between raised
            // banks) is no lake's: the field keeps well under its water.
            if envelope.bank_distance < RUNAWAY_BANKS && envelope.water < lake.level - 0.02 {
                height = height.min(envelope.water - LAKE_FIELD_SINK);
            }
            field.push((corner, height));
        }
    }
    (surface.into_iter().collect(), field)
}

/// How far under the ground a lake sheet's sunk edge lies: enough that the
/// erosion's own small changes there never bare it.
const SHEET_SINK: f32 = 0.2;
/// How far past a channel's waterline, in its half widths, its ribbon surely
/// covers the ground (the ribbon reaches 0.12 to 0.15 past it).
pub const RIBBON_COVER: f32 = 0.1;
/// How far under a river's surface a lake sheet's edge lies where the river's
/// water covers it (its channel at a mouth): enough for the depth test at any
/// range and for the ribbon's rows, which lie on the bisectors of its bends
/// rather than square to each reach, and little enough that the sheet and the
/// ribbon cross within the sheet's last cell rather than the sheet diving to
/// the channel's bed.
const SHEET_UNDER_RIBBON: f32 = 0.06;

/// The corners along a lake sheet's outer edge that must sink for the edge to
/// run under whatever covers it, and the height each sinks to: the lowest
/// `LakeGround::cover` along the edges meeting there (a little under the
/// ground, or just under a river's water in a channel). Wherever the shore
/// rises through the water the corners stay at its level.
fn sheet_edge(noise: &NoiseField, segments: &[RiverSegment], grid: &SegmentGrid, lake: &Lake) -> Vec<([i32; 2], f32)> {
    let ground = LakeGround::new(noise, segments, grid, lake.level);
    let sheet: std::collections::HashSet<[i32; 2]> = lake.cells.iter().chain(&lake.shore).copied().collect();
    let cell = FLOW_CELL as f32;
    let mut corners: std::collections::BTreeMap<[i32; 2], f32> = Default::default();
    for &[x, z] in &sheet {
        for (dx, dz) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
            if sheet.contains(&[x + dx, z + dz]) {
                continue;
            }
            // The edge's two corners, from one end to the other.
            let a = [x + i32::from(dx > 0) , z + i32::from(dz > 0)];
            let b = [a[0] + dz.abs(), a[1] + dx.abs()];
            let cover_at = |k: i32| -> f32 {
                let t = k as f32 / 16.0;
                let p = [
                    (a[0] as f32 + (b[0] - a[0]) as f32 * t) * cell,
                    (a[1] as f32 + (b[1] - a[1]) as f32 * t) * cell,
                ];
                ground.cover(p)
            };
            let lowest = (0..=16).map(cover_at).fold(f32::INFINITY, f32::min);
            for corner in [a, b] {
                let entry = corners.entry(corner).or_insert(f32::INFINITY);
                *entry = entry.min(lowest);
            }
        }
    }
    corners
        .into_iter()
        .filter_map(|(corner, lowest)| (lowest < lake.level).then_some((corner, lowest)))
        .collect()
}

/// The current a river carries into or out of a lake, at the sheet's corners
/// its channel reaches. The sheet's ripples then stream on from the river's
/// where the two surfaces meet, instead of still water meeting running water
/// along the sheet's 4 m cells.
fn sheet_current(segments: &[RiverSegment], grid: &SegmentGrid, lake: &Lake) -> Vec<([i32; 2], [f32; 2])> {
    let cell = FLOW_CELL as f32;
    let corners: std::collections::BTreeSet<[i32; 2]> = lake
        .cells
        .iter()
        .chain(&lake.shore)
        .flat_map(|&[x, z]| [[x, z], [x + 1, z], [x, z + 1], [x + 1, z + 1]])
        .collect();
    let river_current = |corner: [i32; 2]| -> Option<([i32; 2], [f32; 2])> {
        let p = [corner[0] as f32 * cell, corner[1] as f32 * cell];
        if grid.candidates(p).is_empty() {
            return None;
        }
        let envelope = super::carve::envelope_at(segments, grid, p);
        let speed = envelope.velocity[0].hypot(envelope.velocity[1]);
        (envelope.bank_distance < 0.0 && speed > 0.02).then_some((corner, envelope.velocity))
    };
    corners
        .into_iter()
        .filter_map(river_current)
        .collect()
}

/// The node of `river` nearest `p` and the nearest point on its centreline.
/// Side of the cells the rivers already built are indexed by, metres.
const JOIN_CELL: f64 = 16.0;
/// A course this many metres or more past a river's waterline has not
/// reached it (its own half width is added).
const JOIN_REACH: f64 = 1.5;

/// The first point of a course (past its first few, where a stream may
/// spring beside another) that comes within reach of the channel of a river
/// built before it, outside any lake, and that river: where the water joins
/// it. Nearest the head wins, so a stream that settles beside another joins
/// it where they first meet.
fn joins_early(
    points: &[PathPoint],
    rivers: &[Option<River>],
    built: &std::collections::HashMap<[i64; 2], Vec<(usize, usize)>>,
    own: usize,
    lake_at: impl Fn([f64; 2]) -> f64,
) -> Option<(usize, usize)> {
    for (k, point) in points.iter().enumerate().skip(4) {
        let p = point.p;
        if lake_at(p).is_finite() {
            continue;
        }
        let reach = half_width(point.area as f32) as f64 + JOIN_REACH;
        let cell = [(p[0] / JOIN_CELL).floor() as i64, (p[1] / JOIN_CELL).floor() as i64];
        for dz in -1..=1 {
            for dx in -1..=1 {
                let Some(list) = built.get(&[cell[0] + dx, cell[1] + dz]) else {
                    continue;
                };
                for &(river, node) in list {
                    if river == own {
                        continue;
                    }
                    let Some(other) = rivers[river].as_ref() else {
                        continue;
                    };
                    let a = &other.nodes[node];
                    if a.lake || node > other.surface_end {
                        continue;
                    }
                    let b = &other.nodes[(node + 1).min(other.nodes.len() - 1)];
                    let (pa, pb) = ([a.position[0] as f64, a.position[1] as f64], [b.position[0] as f64, b.position[1] as f64]);
                    let ab = [pb[0] - pa[0], pb[1] - pa[1]];
                    let t = (((p[0] - pa[0]) * ab[0] + (p[1] - pa[1]) * ab[1]) / (ab[0] * ab[0] + ab[1] * ab[1]).max(1e-9)).clamp(0.0, 1.0);
                    let q = [pa[0] + ab[0] * t, pa[1] + ab[1] * t];
                    if distance(p, q) <= a.half_width.max(b.half_width) as f64 + reach {
                        return Some((k, river));
                    }
                }
            }
        }
    }
    None
}

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

/// The submerged channel disappears into the existing bed within a few
/// metres of the shore. Follow the river's routed course, including bends,
/// and stop before any dry ground or another shore. Widening belongs above
/// the shore; extending a broad, zero-depth channel offshore still flattens
/// all ground inside its waterline into an artificial lagoon.
fn submerged_transition(
    segments: &mut Vec<RiverSegment>,
    path: &[RiverNode],
    outlet: bool,
    ground: &impl Fn([f32; 2]) -> f32,
    wet: &impl Fn([f32; 2], f32) -> bool,
) {
    let Some(node) = path.first() else { return };
    if path.len() < 2 {
        return;
    }
    let room: f32 = path
        .windows(2)
        .map(|pair| {
            let dx = pair[1].position[0] - pair[0].position[0];
            let dz = pair[1].position[1] - pair[0].position[1];
            dx.hypot(dz)
        })
        .sum();
    let reach = (2.0 * node.half_width).clamp(3.0, 10.0).min(0.4 * room);
    if reach < 0.5 {
        return;
    }
    let mut entered_water = wet(node.position, node.water);
    let shore_cut_reach = if node.lake {
        node.half_width.min(reach)
    } else {
        0.0
    };
    if !entered_water && shore_cut_reach <= 0.0 {
        return;
    }
    let mut points = vec![(node.position, 0.0, true)];
    let mut run = 0.0;
    'course: for pair in path.windows(2) {
        let (a, b) = (pair[0].position, pair[1].position);
        let d = [b[0] - a[0], b[1] - a[1]];
        let length = d[0].hypot(d[1]);
        if length < 1e-4 {
            continue;
        }
        let step_run = length.min(reach - run);
        let steps = (step_run / 0.75).ceil().max(1.0) as usize;
        for step in 1..=steps {
            let advance = step_run * step as f32 / steps as f32;
            let p = [a[0] + d[0] * advance / length, a[1] + d[1] * advance / length];
            let in_water = wet(p, node.water);
            let sampled_run = run + advance;
            if in_water {
                entered_water = true;
            } else if entered_water || sampled_run > shore_cut_reach {
                break 'course;
            }
            // Probe the shore finely, but only routed corners become
            // geometry. Dense carve sections overflow the shared CPU/GPU
            // candidate budget where several outlets are close together.
            points.push((p, sampled_run, step == steps));
        }
        run += step_run;
        if run >= reach - 1e-4 {
            break;
        }
    }
    let actual_reach = points.last().map_or(0.0, |p| p.1);
    if actual_reach < 0.5 || !entered_water {
        return;
    }
    // At a pond's rim the existing outlet has already opened a channel
    // through ground that was dry before carving. Its submerged connector
    // must keep that opening, then hand over to natural pond terrain.
    let needs_rim_cut = ground(node.position) > node.water - node.depth;
    let connector_reach = if node.lake && needs_rim_cut && room > 2.0 * node.half_width {
        node.half_width.min(0.6 * actual_reach)
    } else {
        0.0
    };
    let mut controls: Vec<_> = points
        .iter()
        .filter(|point| point.2)
        .map(|point| (point.0, point.1))
        .collect();
    let tip = points.last().unwrap();
    controls.push((tip.0, tip.1));
    // Two extra sections keep the bed's eased ends even when the whole
    // transition fits inside one original centreline segment.
    for fraction in [0.15, 0.85] {
        let run = actual_reach * fraction;
        let end = points.partition_point(|p| p.1 < run).clamp(1, points.len() - 1);
        let (a, b) = (points[end - 1], points[end]);
        let t = (run - a.1) / (b.1 - a.1).max(1e-5);
        let position = [
            a.0[0] + (b.0[0] - a.0[0]) * t,
            a.0[1] + (b.0[1] - a.0[1]) * t,
        ];
        controls.push((position, run));
    }
    controls.sort_by(|a, b| a.1.total_cmp(&b.1));
    controls.dedup_by(|a, b| (a.1 - b.1).abs() < 0.01);
    let section = |(p, run): ([f32; 2], f32)| {
        let fade = 1.0 - smoothstep(0.0, actual_reach, run);
        let channel_width = node.half_width * fade;
        let mut width = channel_width;
        // A bent route may approach a side bank before its centre reaches
        // it. Keep the entire wetted footprint in the same water body.
        for ray in 0..16 {
            let angle = std::f32::consts::TAU * ray as f32 / 16.0;
            let direction = [angle.cos(), angle.sin()];
            let steps = (width / 0.5).ceil().max(1.0) as usize;
            for step in 1..=steps {
                let radius = width * step as f32 / steps as f32;
                let q = [p[0] + direction[0] * radius, p[1] + direction[1] * radius];
                if !wet(q, node.water) {
                    // Locate the wet edge within the probe interval, then
                    // leave another half metre inside it. Subtracting only
                    // the probe spacing can leave the rounded cap touching
                    // a dry bank between samples.
                    let mut inside = (radius - width / steps as f32).max(0.0);
                    let mut outside = radius;
                    for _ in 0..6 {
                        let middle = 0.5 * (inside + outside);
                        let q = [p[0] + direction[0] * middle, p[1] + direction[1] * middle];
                        if wet(q, node.water) {
                            inside = middle;
                        } else {
                            outside = middle;
                        }
                    }
                    width = (inside - 0.5).max(0.0);
                    break;
                }
            }
        }
        if node.lake {
            if run <= 1e-5 {
                width = node.half_width;
            } else if connector_reach > 0.0 {
                let protection = smoothstep(0.0, connector_reach, run);
                width = channel_width + (width - channel_width) * protection;
            }
        }
        (p, width, node.depth * fade, node.speed * fade, run)
    };
    let mut sections: Vec<_> = controls.into_iter().map(section).collect();
    // Terrain on one side of the rim can briefly leave and re-enter the
    // pond. Its guard must not pinch the connector and swell it again.
    if node.lake {
        for i in 1..sections.len() {
            sections[i].1 = sections[i].1.min(sections[i - 1].1);
        }
    }
    let first_segment = segments.len();
    for pair in sections.windows(2) {
        let (a, b) = if outlet { (pair[1], pair[0]) } else { (pair[0], pair[1]) };
        let mut segment = RiverSegment {
            a: a.0,
            b: b.0,
            water: [node.water; 2],
            half_width: [a.1, b.1],
            depth: [a.2, b.2],
            speed: [a.3, b.3],
            bank: [node.bank.max(0.7); 2],
            ..Default::default()
        };
        // These are submerged bed transitions, not a second set of banks.
        // Lift their bank cone over nearby dry land, including their round
        // caps, so a fan beside an oblique shore cannot excavate that shore.
        for t in [0.0, 0.5, 1.0] {
            let centre = [a.0[0] + (b.0[0] - a.0[0]) * t, a.0[1] + (b.0[1] - a.0[1]) * t];
            let width = a.1 + (b.1 - a.1) * t;
            for ray in 0..16 {
                let angle = std::f32::consts::TAU * ray as f32 / 16.0;
                let direction = [angle.cos(), angle.sin()];
                let mut previous_past = 0.0;
                let mut previous_wet = true;
                for step in 0..=16 {
                    let past = 0.25 + BANK_REACH * step as f32 / 16.0;
                    let p = [centre[0] + direction[0] * (width + past), centre[1] + direction[1] * (width + past)];
                    let natural = ground(p);
                    if natural > node.water {
                        let mut safe_past = past;
                        if previous_wet {
                            let mut inside = previous_past;
                            let mut outside = past;
                            for _ in 0..6 {
                                let middle = 0.5 * (inside + outside);
                                let q = [
                                    centre[0] + direction[0] * (width + middle),
                                    centre[1] + direction[1] * (width + middle),
                                ];
                                if ground(q) > node.water {
                                    outside = middle;
                                } else {
                                    inside = middle;
                                }
                            }
                            safe_past = inside.max(0.01);
                        }
                        let safe_bank = (natural - node.water + CARVE_ROUNDING) / safe_past;
                        segment.bank[0] = segment.bank[0].max(safe_bank);
                    }
                    previous_past = past;
                    previous_wet = natural <= node.water;
                }
            }
        }
        if connector_reach > 0.0 {
            let near_shore = a.4.min(b.4);
            let protection = smoothstep(0.25 * connector_reach, connector_reach, near_shore);
            segment.bank[0] = node.bank + (segment.bank[0] - node.bank) * protection;
        }
        segment.bank[1] = segment.bank[0];
        segments.push(segment);
    }
    if outlet {
        segments[first_segment..].reverse();
    }
    // A chain like any reach, its still water level throughout: each joint
    // at the steeper of the two banks lifted beside it.
    let chain = segments[first_segment..].to_vec();
    for (k, segment) in segments[first_segment..].iter_mut().enumerate() {
        let before = k.checked_sub(1).map(|j| &chain[j]);
        let after = chain.get(k + 1);
        segment.bank = [
            before.map_or(segment.bank[0], |s| s.bank[0].max(segment.bank[0])),
            after.map_or(segment.bank[1], |s| s.bank[0].max(segment.bank[1])),
        ];
        segment.cap_slope = [if before.is_some() { 0.0 } else { super::carve::FLAT_START }, 0.0];
    }
}

fn build_segments(rivers: &[River], noise: &NoiseField, lakes: &Lakes) -> Vec<RiverSegment> {
    let ground = |p: [f32; 2]| base_height(noise, p[0], p[1]);
    let wet = |p: [f32; 2], level: f32| {
        ground(p) < level
            && (level <= SEA_LEVEL + 0.03 || (lakes.level_at([p[0] as f64, p[1] as f64]) - level as f64).abs() < 0.03)
    };
    build_segments_on_ground(rivers, &ground, &wet)
}

/// The first part of the course naturally joined to deep sea water. A
/// coastal hollow before a beach bar is still part of the channel: retain
/// its route across the bar, and stop only on the connected seaward side.
fn sea_shore_index(nodes: &[RiverNode], surface_end: usize, ground: &impl Fn([f32; 2]) -> f32) -> usize {
    let first = surface_end.min(nodes.len() - 1);
    let mut shore = nodes.len() - 1;
    let mut joined_to_deep_water = false;
    for i in (first..nodes.len()).rev() {
        let a = nodes[i].position;
        let b = nodes.get(i + 1).map_or(a, |node| node.position);
        let dx = b[0] - a[0];
        let dz = b[1] - a[1];
        let steps = (dx.hypot(dz) / 0.75).ceil().max(1.0) as usize;
        for step in (0..steps).rev() {
            let t = step as f32 / steps as f32;
            let natural = ground([a[0] + dx * t, a[1] + dz * t]);
            if natural >= SEA_LEVEL {
                joined_to_deep_water = false;
            } else if natural < SEA_LEVEL - OPEN_SEA_DEPTH {
                joined_to_deep_water = true;
            }
        }
        if !joined_to_deep_water && ground(a) < SEA_LEVEL {
            // The routed path may end in shallow connected sea before it
            // reaches a deep node. Look outward through natural wet ground
            // as well; every intervening dry bar blocks that direction.
            let search_reach = (3.0 * nodes[i].half_width + BANK_REACH).clamp(16.0, 64.0);
            let search_steps = (search_reach / 0.75).ceil() as usize;
            'sea: for ray in 0..16 {
                let angle = std::f32::consts::TAU * ray as f32 / 16.0;
                for step in 1..=search_steps {
                    let radius = search_reach * step as f32 / search_steps as f32;
                    let p = [a[0] + angle.cos() * radius, a[1] + angle.sin() * radius];
                    let natural = ground(p);
                    if natural >= SEA_LEVEL {
                        break;
                    }
                    if natural < SEA_LEVEL - OPEN_SEA_DEPTH {
                        joined_to_deep_water = true;
                        break 'sea;
                    }
                }
            }
        }
        if joined_to_deep_water && nodes[i].water <= SEA_LEVEL + 0.03 {
            shore = i;
        }
    }
    shore
}

fn build_segments_on_ground(
    rivers: &[River],
    ground: &impl Fn([f32; 2]) -> f32,
    wet: &impl Fn([f32; 2], f32) -> bool,
) -> Vec<RiverSegment> {
    let mut segments = Vec::new();
    for river in rivers {
        let nodes = &river.nodes;
        if nodes.len() < 2 {
            continue;
        }
        // Routing continues to a deep sea cell and may turn back along a
        // neighbouring beach. The channel ends at its first natural shore,
        // not at that remote routing target.
        let sea_shore = (river.end == RiverEnd::Sea)
            .then(|| sea_shore_index(nodes, river.surface_end, ground));
        // Over its last stretch to the sea a river's banks are the shore:
        // nothing holds the ground up beside it, or the levee of a creek
        // that comes down a sea cliff would dam its own mouth.
        let mouth = (river.end == RiverEnd::Sea).then(|| {
            let end = &river.nodes[river.surface_end.min(river.nodes.len() - 1)];
            end.along - (2.0 * BANK_REACH + 3.0 * end.half_width)
        });
        // So too where it runs into or out of a lake: a levee's outer slope
        // reaches a bank's width round the ends of the segments beside the
        // lake and would raise a bar across the mouth, between the river's
        // water and the lake's. Metres along the river to the nearest lake.
        let mut to_lake = vec![f32::INFINITY; nodes.len()];
        let mut last: Option<f32> = None;
        for (node, near) in nodes.iter().zip(to_lake.iter_mut()) {
            if node.lake {
                last = Some(node.along);
            }
            if let Some(at) = last {
                *near = node.along - at;
            }
        }
        last = None;
        for (node, near) in nodes.iter().zip(to_lake.iter_mut()).rev() {
            if node.lake {
                last = Some(node.along);
            }
            if let Some(at) = last {
                *near = near.min(at - node.along);
            }
        }
        // How firmly each node's banks are held up: not at all in a lake, and
        // fading out toward still water.
        let levee_at = |k: usize| -> f32 {
            let node = &nodes[k];
            if node.lake {
                return 0.0;
            }
            let near_sea = mouth.map_or(1.0, |mouth| 1.0 - smoothstep(mouth - 15.0, mouth, node.along));
            // Clear of the lake by the reach of either segment meeting here.
            let around = &nodes[k.saturating_sub(1)..(k + 2).min(nodes.len())];
            let widest = around.iter().map(|n| n.half_width).fold(0.0f32, f32::max);
            let gentlest = around.iter().map(|n| n.bank).fold(f32::INFINITY, f32::min);
            let clear = 2.0 * widest + BANK_REACH * bank_run(gentlest);
            let near_lake = smoothstep(clear, clear + 20.0, to_lake[k]);
            smoothstep(SEA_LEVEL + 0.05, SEA_LEVEL + 0.6, node.water) * near_sea * near_lake
        };
        // Which reaches have a channel of their own (none under a lake, nor
        // past the shore), and how fast each one's water falls.
        let channelled = |k: usize| -> bool {
            k + 1 < nodes.len() && !(nodes[k].lake && nodes[k + 1].lake) && !sea_shore.is_some_and(|shore| k >= shore)
        };
        let fall_of = |k: usize| -> f32 {
            let (a, b) = (&nodes[k], &nodes[k + 1]);
            let length = (b.position[0] - a.position[0]).hypot(b.position[1] - a.position[1]).max(1e-3);
            (a.water - b.water).max(0.0) / length
        };
        // How fast the water rises up the river behind a segment's start, as
        // far as its start cap reaches back: as fast as the steepest of the
        // reaches there, so the cap never cuts under their banks. Where a
        // wide, low-banked mouth meets the foot of a cascade, the reach
        // just before may fall gently and those above it steeply, and a
        // cap rising only as the one before falls carves a bowl into the
        // hillside beside the cascade.
        let rise_behind = |i: usize| -> f32 {
            let (a, b) = (&nodes[i], &nodes[i + 1]);
            let reach = a.half_width.max(b.half_width) + BANK_REACH * bank_run(a.bank.min(b.bank));
            let mut rise = fall_of(i - 1);
            for j in (0..i).rev() {
                if !channelled(j) {
                    break;
                }
                let run = (a.position[0] - nodes[j].position[0]).hypot(a.position[1] - nodes[j].position[1]);
                if run > reach {
                    break;
                }
                rise = rise.max((nodes[j].water - a.water).max(0.0) / run.max(1e-3));
            }
            rise
        };
        for (i, pair) in river.nodes.windows(2).enumerate() {
            let (a, b) = (&pair[0], &pair[1]);
            if !channelled(i) {
                continue;
            }
            // The first reach, and the first out of a lake, start flat (a
            // spring's narrows away up its lead-in, below).
            let before = if i > 0 && channelled(i - 1) { rise_behind(i) } else { super::carve::FLAT_START };
            let after = if channelled(i + 1) { fall_of(i + 1) } else { fall_of(i) };
            let reach_segment = RiverSegment {
                a: a.position,
                b: b.position,
                water: [a.water, b.water],
                half_width: [a.half_width, b.half_width],
                depth: [a.depth, b.depth],
                speed: [a.speed, b.speed],
                bank: [a.bank, b.bank],
                skew: [a.skew, b.skew],
                levee: [levee_at(i), levee_at(i + 1)],
                cap_slope: [before, after],
                turbulence: [a.turbulence, b.turbulence],
            };
            segments.push(reach_segment);
        }
        // Follow the drowned course at a pond's inlet and outlet. A
        // straight extension of the final tangent can cross its side bank
        // even when the extension is shorter than the lake's crossing.
        for (i, pair) in river.nodes.windows(2).enumerate() {
            let (a, b) = (&pair[0], &pair[1]);
            if a.lake == b.lake {
                continue;
            }
            if sea_shore.is_some_and(|shore| i >= shore) {
                break;
            }
            let path = if a.lake {
                let first = (0..=i).rev().take_while(|&j| nodes[j].lake).last().unwrap_or(i);
                nodes[first..=i]
                    .iter()
                    .rev()
                    .copied()
                    .collect::<Vec<_>>()
            } else {
                let last = (i + 1..nodes.len()).take_while(|&j| nodes[j].lake).last().unwrap_or(i + 1);
                nodes[i + 1..=last].to_vec()
            };
            submerged_transition(&mut segments, &path, a.lake, ground, wet);
        }
        // A stream rising at a spring has no wall at its head: its first
        // segment begins as a square cut across the flow, so a lead-in runs
        // up the slope above the spring, out of the ground, into which the
        // little channel fades. Its top stands over the ground there by the
        // carve's rounding, however steep the hillside, so the cut wedges
        // out into the slope; and it narrows to nothing up the slope, a
        // hollow in the hillside rather than a channel of water.
        if river.nodes.len() >= 2 && !river.nodes[0].lake {
            let (a, b) = (&river.nodes[0], &river.nodes[1]);
            let d = [b.position[0] - a.position[0], b.position[1] - a.position[1]];
            let l = d[0].hypot(d[1]).max(1e-3);
            let reach = (6.0 * a.half_width).max(3.0);
            let top = [a.position[0] - d[0] / l * reach, a.position[1] - d[1] / l * reach];
            let rise = (a.slope.max(0.05) * reach).max(ground(top) + CARVE_ROUNDING - a.water);
            let spring_lead_in = RiverSegment {
                a: top,
                b: a.position,
                water: [a.water + rise, a.water],
                half_width: [0.0, a.half_width],
                depth: [0.0, a.depth],
                speed: [0.0, a.speed],
                bank: [a.bank; 2],
                ..Default::default()
            };
            let after = if channelled(0) { fall_of(0) } else { spring_lead_in.slope() };
            segments.push(RiverSegment { cap_slope: [super::carve::FLAT_START, after], ..spring_lead_in });
        }
        if let Some(shore) = sea_shore {
            submerged_transition(&mut segments, &nodes[shore..], false, ground, wet);
        }
    }
    segments
}

impl RiverNetwork {
    /// The clarity of the lake whose sheet covers `(x, z)`, if one does.
    pub fn lake_clarity_at(&self, x: f32, z: f32) -> Option<f32> {
        let cell = [(x / FLOW_CELL as f32).floor() as i32, (z / FLOW_CELL as f32).floor() as i32];
        self.lake_clarity.get(&cell).copied()
    }

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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reported_pond_outlets_keep_their_full_opening_through_the_rim() {
        let noise = NoiseField::new();
        let network = generate(&noise, [0, 0]);
        for target in [[-765.3, 941.7], [757.7, 1335.2], [710.0, 1256.9]] {
            let candidates = network.rivers.iter().flat_map(|river| {
                river.nodes.windows(2).enumerate()
                    .filter(|(_, pair)| pair[0].lake && !pair[1].lake)
                    .map(move |(i, pair)| {
                        let dx = pair[1].position[0] - target[0];
                        let dz = pair[1].position[1] - target[1];
                        (river, i, dx.hypot(dz))
                    })
            });
            let (river, index, distance) = candidates
                .min_by(|a, b| a.2.total_cmp(&b.2))
                .unwrap();
            // Each now begins where its lake's water does, a node or two
            // back from where the reported outlets left their lakes' cells.
            assert!(distance < 8.0, "the reported outlet should be present: {target:?} nearest {distance}");
            let shore = river.nodes[index];
            let connector = network.segments.iter().find(|segment| {
                segment.b == shore.position && segment.water == [shore.water; 2]
            }).expect("the outlet should have a submerged connector");
            assert_eq!(connector.half_width[1], shore.half_width);
            assert_eq!(connector.depth[1], shore.depth);
            assert_eq!(connector.bank[1], shore.bank);

            let inside = river.nodes[index - 1].position;
            let dx = inside[0] - shore.position[0];
            let dz = inside[1] - shore.position[1];
            let length = dx.hypot(dz);
            let direction = [dx / length, dz / length];
            let normal = [-direction[1], direction[0]];
            for run in [0.4, 1.0] {
                for side in [-0.65, 0.0, 0.65] {
                    let offset = side * shore.half_width;
                    let p = [
                        shore.position[0] + direction[0] * run + normal[0] * offset,
                        shore.position[1] + direction[1] * run + normal[1] * offset,
                    ];
                    let natural = base_height(&noise, p[0], p[1]);
                    let carved = network.envelope(p[0], p[1]).clamp(natural);
                    assert!(carved < shore.water - 0.1,
                        "the pond rim must leave the outlet open at {p:?}: {carved}");
                }
            }
        }
    }

    #[test]
    fn pond_outlets_with_interior_room_keep_their_shore_connectors_open() {
        let noise = NoiseField::new();
        let network = generate(&noise, [0, 0]);
        let mut checked = 0;
        let mut above_water = 0;
        for river in &network.rivers {
            for (index, pair) in river.nodes.windows(2).enumerate() {
                let shore = pair[0];
                if !shore.lake || pair[1].lake || index == 0 {
                    continue;
                }
                let first = (0..=index).rev()
                    .take_while(|&i| river.nodes[i].lake)
                    .last().unwrap();
                let room = shore.along - river.nodes[first].along;
                let reach = (2.0 * shore.half_width).clamp(3.0, 10.0).min(0.4 * room);
                if room <= 2.0 * shore.half_width || reach < 2.5 {
                    continue;
                }
                let connector = network.segments.iter().find(|segment| {
                    segment.b == shore.position && segment.water == [shore.water; 2]
                }).expect("every roomy pond outlet must connect back into its pond");
                assert_eq!(connector.half_width[1], shore.half_width);
                assert_eq!(connector.depth[1], shore.depth);
                let natural = base_height(&noise, shore.position[0], shore.position[1]);
                if natural >= shore.water {
                    above_water += 1;
                    assert_eq!(connector.bank[1], shore.bank);
                }
                let inside = river.nodes[index - 1].position;
                let dx = inside[0] - shore.position[0];
                let dz = inside[1] - shore.position[1];
                let length = dx.hypot(dz);
                let direction = [dx / length, dz / length];
                let normal = [-direction[1], direction[0]];
                for run in [0.0, 0.4, 1.0] {
                    let centre = [
                        shore.position[0] + direction[0] * run,
                        shore.position[1] + direction[1] * run,
                    ];
                    let local_section = network.segments.iter().find_map(|segment| {
                        if segment.water != [shore.water; 2] {
                            return None;
                        }
                        let dx = segment.b[0] - segment.a[0];
                        let dz = segment.b[1] - segment.a[1];
                        let t = ((centre[0] - segment.a[0]) * dx
                            + (centre[1] - segment.a[1]) * dz)
                            / (dx * dx + dz * dz).max(1e-6);
                        if !(0.0..=1.0).contains(&t) {
                            return None;
                        }
                        let q = [segment.a[0] + dx * t, segment.a[1] + dz * t];
                        if (q[0] - centre[0]).hypot(q[1] - centre[1]) > 0.05 {
                            return None;
                        }
                        let width = segment.half_width[0]
                            + (segment.half_width[1] - segment.half_width[0]) * t;
                        let depth = segment.depth[0]
                            + (segment.depth[1] - segment.depth[0]) * t;
                        Some((width, depth))
                    });
                    let Some((width, depth)) = local_section else { continue };
                    if width < 0.1 || depth < 0.1 {
                        continue;
                    }
                    for side in [-0.65, 0.0, 0.65] {
                        let offset = side * width;
                        let p = [
                            centre[0] + normal[0] * offset,
                            centre[1] + normal[1] * offset,
                        ];
                        let natural = base_height(&noise, p[0], p[1]);
                        let carved = network.envelope(p[0], p[1]).clamp(natural);
                        assert!(carved < shore.water - 0.03,
                            "pond outlet {:?} is blocked at {p:?}: {carved} >= {}",
                            shore.position, shore.water);
                    }
                }
                checked += 1;
            }
        }
        assert!(checked > 100);
        // An outlet leaves its lake where the water is, never from a node on
        // the dry ground of the lake's last cells.
        assert_eq!(above_water, 0, "outlets beginning on dry ground: {above_water} of {checked}");
    }

    /// A lake backs the river above it up to its level. Where the river's
    /// course runs through ground below that level, the water stands over it
    /// as the lake's, not as a river held up between dikes.
    #[test]
    fn a_lakes_backwater_floods_the_hollows_below_its_level() {
        let noise = NoiseField::new();
        let network = generate(&noise, [0, 0]);
        let mut backed = 0;
        for river in &network.rivers {
            let (mut level, mut shore) = (f32::NEG_INFINITY, 0.0);
            for node in river.nodes.iter().rev() {
                if node.lake {
                    (level, shore) = (node.water, node.along);
                    continue;
                }
                // Up the river from the lake's 4 m cells, which end anywhere
                // across a cell of a steep shore, and backed up to its level.
                if shore - node.along < 2.0 * FLOW_CELL as f32 || (node.water - level).abs() > 0.01 {
                    continue;
                }
                backed += 1;
                let natural = base_height(&noise, node.position[0], node.position[1]);
                assert!(natural > level - 1.0, "held {:.2} m over the ground at {:?}", level - natural, node.position);
            }
        }
        assert!(backed > 100, "{backed}");
    }

    /// Over its last metres a creek's water becomes its river's: its speed
    /// comes to the river's beside it and its bed deepens to the river's
    /// instead of hanging over it; it keeps its own whitewater to the
    /// river's waterline and takes on the river's inside it. Far above, it
    /// is all its own.
    #[test]
    fn a_tributary_takes_on_its_parents_water_where_it_runs_in() {
        let parent_node = |k: i32| RiverNode {
            position: [-40.0 + 4.0 * k as f32, 0.0],
            water: 5.0,
            half_width: 2.0,
            depth: 0.8,
            speed: 2.0,
            turbulence: 0.8,
            along: 4.0 * k as f32,
            ..Default::default()
        };
        let parent = River { nodes: (0..=20).map(parent_node).collect(), end: RiverEnd::Edge, surface_end: 20 };
        // Straight in from the side, ending on the river's centreline.
        let creek_node = |k: i32| RiverNode {
            position: [0.0, -30.0 + 2.0 * k as f32],
            water: 5.0,
            half_width: 1.0,
            depth: 0.3,
            speed: 0.6,
            turbulence: 0.1,
            along: 2.0 * k as f32,
            ..Default::default()
        };
        let mut nodes: Vec<RiverNode> = (0..=15).map(creek_node).collect();
        merge_with_parent(&mut nodes, &parent);
        let (first, last) = (&nodes[0], &nodes[15]);
        assert_eq!((first.turbulence, first.speed, first.depth), (0.1, 0.6, 0.3), "{first:?}");
        assert!((last.turbulence - 0.8).abs() < 1e-5 && (last.speed - 2.0).abs() < 1e-5, "{last:?}");
        assert!(last.depth >= 0.8 - 1e-5, "{last:?}");
        // Its own mouth, outside the river's waterline (2 m), stays calm.
        let mouth = &nodes[14];
        assert!(mouth.position[1] == -2.0 && mouth.turbulence == 0.1, "{mouth:?}");
        for pair in nodes.windows(2) {
            let [a, b] = [&pair[0], &pair[1]];
            assert!(b.speed >= a.speed && b.depth >= a.depth && b.turbulence >= a.turbulence, "{pair:?}");
        }
    }

    /// Channels are as deep and as fast as real ones of their size.
    #[test]
    fn channels_are_as_deep_and_fast_as_real_ones() {
        // A creek carrying 1 m³/s down a 1 % slope: about 0.95 m over its
        // thalweg, one to one and a half metres a second at bankfull.
        let (_, depth, speed) = channel(1.0, 0.01);
        assert!((0.8..1.1).contains(&depth), "{depth}");
        assert!((0.9..1.6).contains(&speed), "{speed}");
        for q in [0.2f32, 1.0, 5.0, 30.0] {
            for slope in [0.002f32, 0.01, 0.05, 0.12] {
                let (half_width, depth, speed) = channel(q, slope);
                let ratio = 2.0 * half_width / (depth / THALWEG_OVER_MEAN);
                assert!((5.5..50.0).contains(&ratio), "W/D {ratio} at {q} m³/s, slope {slope}");
                assert!((0.15..=4.0).contains(&speed));
            }
        }
        // Deeper water runs faster; a steep bed is rougher, so speed grows
        // slowly with slope.
        assert!(flow_speed(1.0, 0.01) > flow_speed(0.4, 0.01));
        assert!(flow_speed(0.5, 0.1) < 2.5 * flow_speed(0.5, 0.01));
    }

    fn reach(flags: &[bool], water: &[f32]) -> Vec<RiverNode> {
        let node_inputs = flags.iter().zip(water).enumerate();
        node_inputs
            .map(|(i, (&lake, &water))| RiverNode {
                position: [i as f32 * 2.5, 0.0],
                water,
                half_width: 1.0,
                along: i as f32 * 2.5,
                lake,
                ..Default::default()
            })
            .collect()
    }

    /// A river that leaves its lake's cells for a few metres and comes back
    /// stays in the lake; a separate hollow or another lake does not join.
    #[test]
    fn a_river_crossing_a_bar_in_its_lake_stays_in_the_lake() {
        let flags = [false, true, true, false, false, false, false, true, true, false];
        let water = [6.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 4.9];
        let mut nodes = reach(&flags, &water);
        settle_lake_runs(&mut nodes, |_| 5.1);
        let lake_flags: Vec<bool> = nodes.iter().map(|n| n.lake).collect();
        assert_eq!(lake_flags, [false, true, true, true, true, true, true, true, true, false]);
        // A hollow deeper than the lake's sheet would show is not taken in.
        let mut nodes = reach(&flags, &water);
        settle_lake_runs(&mut nodes, |p| if p[0] == 10.0 { 4.0 } else { 5.1 });
        assert!(!nodes[4].lake);
        // A single node out of the lake is a ragged edge, however low.
        let flags = [false, true, true, false, true, true, false];
        let mut nodes = reach(&flags, &[6.0, 5.0, 5.0, 5.0, 5.0, 5.0, 4.9]);
        settle_lake_runs(&mut nodes, |_| 3.0);
        assert!(nodes[3].lake);
        // Nor is a stretch between two different lakes.
        let flags = [false, true, true, false, false, true, true, false];
        let water = [6.0, 5.0, 5.0, 5.0, 4.8, 4.5, 4.5, 4.4];
        let mut nodes = reach(&flags, &water);
        settle_lake_runs(&mut nodes, |_| 5.1);
        assert!(!nodes[3].lake && !nodes[4].lake);
    }

    #[test]
    fn brushing_a_lake_is_not_crossing_it() {
        let flags = [false, true, false, false, true, true, true, true, false];
        let nodes = reach(&flags, &[5.0; 9]);
        assert_eq!(lake_crossings(&nodes), vec![(4, 7)]);
    }

    fn uniform_channel(flags: &[bool]) -> Vec<RiverNode> {
        let mut nodes = reach(flags, &vec![5.0; flags.len()]);
        for node in &mut nodes {
            node.half_width = 2.0;
            node.depth = 0.8;
            node.speed = 1.0;
            node.bank = 0.7;
            node.turbulence = 0.2;
        }
        nodes
    }

    /// Deepened toward the coast, an estuary held a closed pool behind the
    /// shore, deeper than the shelving seabed it opened into. Its bed falls
    /// to the sea and meets the seabed instead.
    #[test]
    fn an_estuary_bed_falls_to_the_seabed_without_a_pool_behind_the_shore() {
        let water: Vec<f32> = (0..=100).map(|i| (1.5 - 0.03 * i as f32).max(SEA_LEVEL + 0.02)).collect();
        let mut nodes = reach(&[false; 101], &water);
        for (i, node) in nodes.iter_mut().enumerate() {
            node.half_width = 2.0;
            node.speed = 1.0;
            node.bank = 0.7;
            // Pools and riffles, as the channel's bed undulates.
            node.depth = if i % 7 < 3 { 1.2 } else { 0.8 };
        }
        // The beach ends 130 m down the river, and the seabed shelves at
        // 0.4 m under the sea before it drops away past 200 m.
        let ground = |p: [f32; 2]| if p[0] < 130.0 { 2.0 } else if p[0] < 200.0 { SEA_LEVEL - 0.4 } else { SEA_LEVEL - 3.0 };
        let surface_end = nodes.iter().position(|n| n.water <= SEA_LEVEL + 0.03).unwrap();
        let upstream = nodes[10];
        open_mouths(&mut nodes, RiverEnd::Sea, surface_end);
        grade_estuary(&mut nodes, surface_end, ground);
        let bed = |n: &RiverNode| n.water - n.depth;
        let mouth = nodes[surface_end].along;
        for pair in nodes.windows(2).filter(|pair| pair[0].along >= mouth - 60.0) {
            assert!(bed(&pair[1]) <= bed(&pair[0]) + 1e-5, "the bed rises toward the sea at {}", pair[1].along);
        }
        for node in &nodes[surface_end..] {
            assert!(bed(node) >= SEA_LEVEL - 0.4 - 1e-5, "deeper than the seabed it opens into: {node:?}");
            assert!(node.depth >= ESTUARY_LEAST_DEPTH - 1e-5);
        }
        assert_eq!(nodes[10], upstream, "the river above its estuary keeps its own bed");
    }

    #[test]
    fn estuary_opens_gradually_before_the_shore_and_stays_open_offshore() {
        // Graded down to the sea (`grade_to_sea`), an estuary's water stands
        // at the sea's level over its mouth.
        let mut nodes = uniform_channel(&[false; 101]);
        for node in &mut nodes {
            node.water = SEA_LEVEL + 0.02;
        }
        let original = nodes.clone();
        let shore = 60;
        open_mouths(&mut nodes, RiverEnd::Sea, shore);
        assert_eq!(nodes[30], original[30], "ordinary channel 75 m above the coast is unchanged");
        assert!(nodes[45].half_width > original[45].half_width + 0.5);
        assert!((nodes[shore].half_width / original[shore].half_width - 2.2).abs() < 1e-5);
        assert!(nodes[shore].bank <= 0.36);
        assert!(nodes.windows(2).all(|p| p[1].half_width >= p[0].half_width));
        assert!(nodes[shore..].iter().all(|n| (n.half_width - nodes[shore].half_width).abs() < 1e-5));
        for (node, old) in nodes.iter().zip(&original) {
            assert_eq!(node.water, old.water);
            let section_flow = node.half_width * node.depth * node.speed;
            assert!((section_flow - old.half_width * old.depth * old.speed).abs() < 1e-5);
        }
        // No corner at either end of the widening, even on a coarse mesh.
        assert!(nodes[37].half_width - nodes[36].half_width < 0.01);
        assert!(nodes[shore].half_width - nodes[shore - 1].half_width < 0.01);
    }

    #[test]
    fn pond_outlet_flare_crosses_the_shore_and_narrows_smoothly_downstream() {
        let flags: Vec<bool> = (0..=80).map(|i| i <= 30).collect();
        let mut nodes = uniform_channel(&flags);
        let original = nodes.clone();
        open_mouths(&mut nodes, RiverEnd::Edge, 80);
        assert!(nodes[30].half_width > 1.65 * original[30].half_width);
        assert!((nodes[30].half_width - nodes[31].half_width).abs() < 1e-5);
        assert!(nodes[35].half_width > 1.4 * original[35].half_width);
        assert_eq!(nodes[44], original[44]);
        assert!(nodes[31..].windows(2).all(|p| p[1].half_width <= p[0].half_width));
        assert!(nodes[30].bank < 0.29);
        assert!(nodes[31].speed < nodes[40].speed);
        assert!(nodes.iter().zip(&original).all(|(a, b)| a.water == b.water));
    }

    /// A lake's mouth is the reach drowned in its water: a cascade down into
    /// a lake keeps its own channel, width, banks and whitewater until its
    /// water comes down to the lake's, rather than spreading into a broad,
    /// glassy, low-banked sheet over the hillside above the shore.
    #[test]
    fn a_cascade_into_a_lake_opens_its_mouth_only_at_the_lakes_level() {
        let flags: Vec<bool> = (0..=60).map(|i| i >= 40).collect();
        // Falling 0.5 m a node (a slope of 0.2) to the lake's level at 5 m.
        let water: Vec<f32> = (0..=60).map(|i| 5.0 + 0.5 * (39 - i.min(39)) as f32).collect();
        let mut cascade = reach(&flags, &water);
        let mut drowned = uniform_channel(&flags);
        for nodes in [&mut cascade, &mut drowned] {
            for node in nodes.iter_mut() {
                node.half_width = 2.0;
                node.depth = 0.8;
                node.speed = 1.0;
                node.bank = 0.7;
                node.turbulence = 0.6;
            }
        }
        let (original, level) = (cascade.clone(), drowned.clone());
        open_mouths(&mut cascade, RiverEnd::Edge, 60);
        open_mouths(&mut drowned, RiverEnd::Edge, 60);
        // A metre and a half and more above the lake, the cascade is
        // untouched...
        assert_eq!(cascade[35], original[35]);
        assert_eq!(cascade[36], original[36]);
        // ...where a river at the lake's level is already opening.
        assert!(drowned[35].half_width > level[35].half_width + 0.5);
        assert!(drowned[36].turbulence < 0.5 * level[36].turbulence);
        // At the lake its mouth opens as any inlet's does.
        assert_eq!(cascade[40], drowned[40]);
        assert!(cascade[40].half_width > 1.8 * original[40].half_width);
    }

    /// A start cap rises as fast as the river does behind it, as far back as
    /// it reaches: at the foot of a cascade, past a short reach that falls
    /// gently into a wide, low-banked mouth, a cap rising only as that reach
    /// falls would stand metres under the cascade's water beside it and
    /// scoop a bowl out of the hillside there.
    #[test]
    fn a_start_cap_rises_with_the_cascade_behind_it() {
        // A cascade falling 1.2 m a metre, then a reach falling 0.3 m a
        // metre into a wide pool at the cascade's foot.
        let mut water: Vec<f32> = (0..12).map(|i| 60.0 - 3.0 * i as f32).collect();
        water.extend([26.25, 25.5, 25.5, 25.5]);
        let flags = vec![false; water.len()];
        let mut nodes = reach(&flags, &water);
        for (i, node) in nodes.iter_mut().enumerate() {
            node.half_width = if i >= 12 { 4.0 } else { 2.0 };
            node.depth = 0.7;
            node.bank = if i >= 12 { 0.25 } else { 1.0 };
        }
        let river = River { nodes, end: RiverEnd::Edge, surface_end: water.len() - 1 };
        let segments = build_segments_on_ground(std::slice::from_ref(&river), &|_| 100.0, &|_, _| false);
        let pool = segments.iter().find(|s| s.a == river.nodes[12].position).unwrap();
        // The reach just before falls 0.3 m a metre; the cascade above it,
        // within the cap's reach, 1.2.
        assert!(pool.cap_slope[0] > 1.0, "{pool:?}");
        // Beside the cascade, behind the pool's start, the cap's bank stands
        // over the cascade's own water there.
        let p = [river.nodes[10].position[0], 5.0];
        let cap = super::super::carve::segment_envelope(pool, p);
        assert!(cap.upper > river.nodes[10].water, "{cap:?}");
    }

    #[test]
    fn submerged_transitions_join_and_taper_to_zero_without_a_depth_step() {
        let path = uniform_channel(&[true; 12]);
        let node = path[0];
        for outlet in [false, true] {
            let mut fan = Vec::new();
            submerged_transition(&mut fan, &path, outlet, &|_| 0.0, &|_, _| true);
            assert!(fan.windows(2).all(|p| p[0].b == p[1].a && p[0].depth[1] == p[1].depth[0]));
            let (channel, bed) = if outlet {
                (fan.last().unwrap().depth[1], fan[0].depth[0])
            } else {
                (fan[0].depth[0], fan.last().unwrap().depth[1])
            };
            assert_eq!(channel, node.depth);
            assert_eq!(bed, 0.0);
            let tip = if outlet { &fan[0] } else { fan.last().unwrap() };
            assert_eq!(tip.half_width[usize::from(!outlet)], 0.0, "a zero-depth tip must also have zero width");
            assert!(fan.iter().all(|s| s.levee == [0.0; 2] && s.water == [node.water; 2]));
            let slope = |s: &RiverSegment| (s.depth[1] - s.depth[0]).abs() / (s.b[0] - s.a[0]).hypot(s.b[1] - s.a[1]);
            let end_slope = slope(&fan[0]).max(slope(fan.last().unwrap()));
            let steepest = fan.iter().map(slope).fold(0.0f32, f32::max);
            assert!(end_slope < 0.5 * steepest, "both ends should ease into their neighbours");
        }
    }

    #[test]
    fn a_shallow_pond_rim_keeps_the_outlet_open_then_tapers_without_pinching() {
        let mut path = uniform_channel(&[true; 7]);
        for node in &mut path {
            node.bank = 0.28;
        }
        // The natural pond narrows to a shallow sill. The channel has
        // already opened that sill to its full two-metre half width.
        let ground = |p: [f32; 2]| {
            let natural_half_width = 0.5 + 0.6 * p[0].max(0.0);
            if p[1].abs() > natural_half_width { 7.0 } else { 4.8 }
        };
        let wet = |p: [f32; 2], level: f32| ground(p) < level;
        let mut fan = Vec::new();
        submerged_transition(&mut fan, &path, true, &ground, &wet);
        let connection = fan.last().unwrap();
        assert_eq!(connection.half_width[1], path[0].half_width);
        assert_eq!(connection.depth[1], path[0].depth);
        assert_eq!(connection.bank[1], path[0].bank);
        assert!(connection.half_width[0] > 1.25,
            "the first submerged reach must remain an open mouth");
        for segment in &fan {
            assert!(segment.half_width[0] <= segment.half_width[1]);
        }
        for pair in fan.windows(2) {
            assert_eq!(pair[0].half_width[1], pair[1].half_width[0]);
            assert_eq!(pair[0].depth[1], pair[1].depth[0]);
        }
        assert_eq!(fan[0].half_width[0], 0.0);
    }

    #[test]
    fn a_pond_connector_crosses_an_initial_dry_sill_then_requires_wet_interior() {
        let mut path = uniform_channel(&[true; 7]);
        for node in &mut path {
            node.bank = 0.28;
        }
        for rise in [0.0, 0.4, 2.0] {
            let ground = |p: [f32; 2]| {
                if p[0] < 0.9 { 5.0 + rise } else { 4.8 }
            };
            let wet = |p: [f32; 2], level: f32| ground(p) < level;
            let mut fan = Vec::new();
            submerged_transition(&mut fan, &path, true, &ground, &wet);
            let connection = fan.last().expect("the initial sill should join real pond water");
            assert_eq!(connection.b, path[0].position);
            assert_eq!(connection.half_width[1], path[0].half_width);
            assert_eq!(connection.depth[1], path[0].depth);
            assert_eq!(connection.bank[1], path[0].bank);
            let p = [0.4, 0.65 * path[0].half_width];
            let mut envelope = super::super::carve::Envelope::NONE;
            for segment in &fan {
                super::super::carve::combine(&mut envelope,
                    &super::super::carve::segment_envelope(segment, p));
            }
            assert!(envelope.clamp(ground(p)) < path[0].water - 0.1);
        }
        let mut fan = Vec::new();
        submerged_transition(&mut fan, &path, true, &|_| 5.4, &|_, _| false);
        assert!(fan.is_empty(), "a connector must reach actual pond water");
        let ground = |p: [f32; 2]| {
            if (0.8..2.4).contains(&p[0]) { 4.8 } else { 5.4 }
        };
        let wet = |p: [f32; 2], level: f32| ground(p) < level;
        submerged_transition(&mut fan, &path, true, &ground, &wet);
        assert!(!fan.is_empty());
        assert!(fan.iter().all(|segment| segment.a[0] < 2.4 && segment.b[0] < 2.4),
            "once inside the pond, a later dry bank must stop the transition");
    }

    #[test]
    fn a_submerged_transition_follows_a_bend_instead_of_crossing_a_ponds_side_bank() {
        let mut path = uniform_channel(&[true; 4]);
        for (node, position) in path.iter_mut().zip([[0.0, 0.0], [1.5, 0.0], [1.5, 3.0], [1.5, 10.0]]) {
            node.position = position;
        }
        let ground = |p: [f32; 2]| if p[0] < 3.0 { 3.0 } else { 9.0 };
        let wet = |p: [f32; 2], level: f32| ground(p) < level;
        for outlet in [false, true] {
            let mut fan = Vec::new();
            submerged_transition(&mut fan, &path, outlet, &ground, &wet);
            assert!(!fan.is_empty());
            assert!(fan.iter().all(|s| s.a[0] <= 1.5 && s.b[0] <= 1.5));
            assert!(fan.iter().any(|s| s.a[1] > 1.0 || s.b[1] > 1.0), "transition should follow the drowned bend");
            let p = [4.0, 0.0];
            let mut envelope = super::super::carve::Envelope::NONE;
            for segment in &fan {
                super::super::carve::combine(&mut envelope, &super::super::carve::segment_envelope(segment, p));
            }
            assert!((envelope.clamp(ground(p)) - ground(p)).abs() < 1e-4, "the adjacent dry bank must keep its height");
        }
    }

    #[test]
    fn transitions_in_a_narrow_pond_stop_before_the_opposite_shore() {
        let mut path = uniform_channel(&[true; 3]);
        for (node, position) in path.iter_mut().zip([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]]) {
            node.position = position;
            node.half_width = 5.0;
        }
        let ground = |p: [f32; 2]| if p[0] <= 4.0 { 3.0 } else { 9.0 };
        let wet = |p: [f32; 2], level: f32| ground(p) < level;
        let mut fan = Vec::new();
        submerged_transition(&mut fan, &path, false, &ground, &wet);
        assert!(!fan.is_empty());
        assert!(fan.iter().all(|s| s.a[0] <= 1.6 && s.b[0] <= 1.6));
        for x in [4.1, 4.5, 5.0, 6.0, 8.0] {
            let p = [x, 0.0];
            let mut envelope = super::super::carve::Envelope::NONE;
            for segment in &fan {
                super::super::carve::combine(&mut envelope, &super::super::carve::segment_envelope(segment, p));
            }
            assert!((envelope.clamp(ground(p)) - ground(p)).abs() < 1e-4, "opposite shore at {x} must not be carved");
        }
    }

    #[test]
    fn a_sea_mouth_stops_before_the_routing_path_turns_onto_an_adjacent_beach() {
        let mut nodes = uniform_channel(&[false; 5]);
        for (node, position) in nodes.iter_mut().zip([[-10.0, 0.0], [0.0, 0.0], [4.0, 0.0], [4.0, 20.0], [20.0, 30.0]]) {
            node.position = position;
            node.water = SEA_LEVEL + 0.02;
        }
        nodes[0].water = SEA_LEVEL + 0.3;
        let river = River { nodes, end: RiverEnd::Sea, surface_end: 1 };
        let ground = |p: [f32; 2]| if p[0] <= 0.0 || p[1] >= 14.0 { 3.0 } else { -2.0 };
        let wet = |p: [f32; 2], level: f32| ground(p) < level;
        let segments = build_segments_on_ground(&[river], &ground, &wet);
        assert!(segments.iter().all(|s| s.a[1] <= 4.0 && s.b[1] <= 4.0));
        for p in [[4.0, 20.0], [20.0, 30.0], [4.0, 14.0]] {
            let mut envelope = super::super::carve::Envelope::NONE;
            for segment in &segments {
                super::super::carve::combine(&mut envelope, &super::super::carve::segment_envelope(segment, p));
            }
            assert!((envelope.clamp(ground(p)) - ground(p)).abs() < 1e-4, "routing-only beach at {p:?} must keep its height");
        }
    }

    #[test]
    fn a_coastal_hollow_does_not_end_the_channel_before_it_crosses_a_beach_bar() {
        let mut nodes = uniform_channel(&[false; 5]);
        for (i, node) in nodes.iter_mut().enumerate() {
            node.position = [i as f32 * 4.0, 0.0];
            node.water = SEA_LEVEL + 0.02;
        }
        let ground = |p: [f32; 2]| {
            if p[0] < 0.0 || (8.5..11.5).contains(&p[0]) {
                1.0
            } else if p[0] < 12.0 {
                -0.2
            } else {
                -2.0
            }
        };
        assert_eq!(sea_shore_index(&nodes, 1, &ground), 3);
        let wet = |p: [f32; 2], level: f32| ground(p) < level;
        let river = River { nodes, end: RiverEnd::Sea, surface_end: 1 };
        let segments = build_segments_on_ground(&[river], &ground, &wet);
        assert!(segments.iter().any(|segment| segment.a[0] == 8.0 && segment.b[0] == 12.0));
    }

    #[test]
    fn pond_drawdown_eases_out_of_still_water_without_a_slope_step() {
        let falls: Vec<f32> = (0..=100).map(|i| outlet_drawdown(i as f32 * 0.25)).collect();
        assert_eq!(falls[0], 0.0);
        assert!(falls.windows(2).all(|p| p[1] >= p[0] && p[1] - p[0] <= OUTLET_SLOPE * 0.25 + 1e-6));
        assert!(falls[1] < OUTLET_SLOPE * 0.25 * 0.02);
        assert!(((falls[49] - falls[48]) - (falls[48] - falls[47])).abs() < 1e-4);
    }

    #[test]
    fn outlet_banks_leave_room_for_their_broad_shoulders_before_raising_a_levee() {
        let flags: Vec<bool> = (0..=80).map(|i| i <= 20).collect();
        let mut nodes = uniform_channel(&flags);
        open_mouths(&mut nodes, RiverEnd::Edge, 80);
        let lake_end = nodes[20].along;
        let river = River { nodes, end: RiverEnd::Edge, surface_end: 80 };
        let segments = build_segments_on_ground(&[river], &|_| 4.0, &|_, _| true);
        // A node raises a levee only clear of the lake by its banks' reach;
        // between it and the last node without one, the levee fades in.
        for segment in &segments {
            let clearance = segment.reach() + segment.half_width[0].max(segment.half_width[1]);
            for (end, position) in [segment.a, segment.b].iter().enumerate() {
                if segment.levee[end] > 0.0 {
                    assert!(position[0] - lake_end > clearance, "{segment:?}");
                }
            }
        }
        assert!(segments.iter().any(|s| s.levee == [1.0; 2]));
    }

    /// A river that drops to the sea down a beach's face is graded down to
    /// it through the beach instead; one that comes down its valley to the
    /// sea anyway keeps its own profile.
    #[test]
    fn a_river_is_graded_to_the_sea_only_where_it_tumbles_down_the_beach() {
        let s: Vec<f64> = (0..=60).map(|i| i as f64 * 4.0).collect();
        let shore = 50;
        let floor = vec![f32::NEG_INFINITY; s.len()];
        // Level at 2.3 m until 10 m from the shore, then down the beach.
        let mut water: Vec<f32> = s.iter().map(|&v| if v < 190.0 { 2.3 } else { (2.3 - 0.22 * (v - 190.0) as f32).max(0.02) }).collect();
        grade_to_sea(&mut water, &s, &floor, shore);
        assert!(water[shore - 2] < 0.2, "{}", water[shore - 2]);
        assert!(water[shore..].iter().all(|&w| w <= SEA_LEVEL + 0.02 + 1e-5));
        assert!(water.windows(2).all(|w| w[1] <= w[0] + 1e-5));
        assert!((water[shore - 30] - 2.3).abs() < 1e-5, "{}", water[shore - 30]);
        assert!(water[shore - 1] - water[shore] < 0.02, "mouth should level into the ocean");
        // A valley falling steeply and then gently to the shore lies under
        // the chord already.
        let valley: Vec<f32> = s.iter().map(|&v| (0.02 + 0.0004 * (200.0 - v).max(0.0).powi(2) as f32).max(0.02)).collect();
        let mut graded = valley.clone();
        grade_to_sea(&mut graded, &s, &floor, shore);
        assert_eq!(graded, valley);
    }
}
