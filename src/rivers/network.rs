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

use super::carve::{BANK_REACH, GRID_CELL, RiverSegment, SegmentGrid};
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
        let mut queue: std::collections::VecDeque<usize> = (0..corridor.ground.len())
            .filter(|&index| {
                corridor.ground[index] < SEA_LEVEL - OPEN_SEA_DEPTH
                    || (corridor.ground[index] < SEA_LEVEL
                        && in_mouth(index)
                        && joined.as_ref().is_none_or(|joined| joined[index]))
            })
            .collect();
        if queue.is_empty() {
            queue = (0..corridor.ground.len())
                .filter(|&index| {
                    below(index) && {
                        let cell = corridor.cell_at(index);
                        [(1, 0), (-1, 0), (0, 1), (0, -1)]
                            .iter()
                            .any(|&(dx, dz)| corridor.inside([cell[0] + dx, cell[1] + dz]).is_none())
                    }
                })
                .collect();
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
        self.lakes.push(Lake {
            level,
            cells: cells.iter().map(|c| [c[0] as i32, c[1] as i32]).collect(),
            shore: Vec::new(),
            edge: Vec::new(),
            current: Vec::new(),
        });
        Filled::Lake
    }
}

/// Samples across a flow cell when a basin's rim is checked.
const RIM_SAMPLES_PER_CELL: i64 = 4;
/// How much ground under its level past a basin's cells water must reach,
/// in samples, or how far, to have run out of it: more than any closed arm,
/// bay or island the fill stepped past (one it joined through a gap
/// narrower than its cells), which the water floods to its end instead.
const RIM_AREA: usize = 20_000;
const RIM_REACH: u32 = 256;

/// Where water standing at `level` in `cells` runs out of them, metre by
/// metre: through a gap in a rim the flood fill, which sees each cell at its
/// centre, took for whole, onto more ground under the level than any closed
/// arm of the basin holds (`RIM_AREA`, `RIM_REACH`). The height of the
/// lowest such way's highest ground, or None if the basin holds its water.
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
    let mut flooded = 0usize;
    while let Some(std::cmp::Reverse((water, distance, sample))) = heap.pop() {
        if distance > 0 {
            flooded += 1;
        }
        if distance > RIM_REACH || flooded > RIM_AREA {
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
    let start = (0..n).find(|&i| {
        let j = s.partition_point(|&v| v < s[i] + 30.0).min(n - 1);
        j <= i || (ground[i] - ground[j]) / (s[j] - s[i]).max(1.0) < HEAD_SLOPE
    });
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

/// How far above its shore a river's lowest reach is graded to the sea.
const MOUTH_GRADE_REACH: f64 = 120.0;
/// A bar across a river's mouth lower than this above the sea is cut
/// through; higher ground is land the river still has to come down.
const MOUTH_BAR: f32 = 1.5;
/// Deepest a mouth's grade cuts under the river's own profile.
const MOUTH_CUT: f32 = 3.0;

/// Bring a river down to the sea along the chord from its surface
/// `MOUTH_GRADE_REACH` above its shore to the sea's level at the shore, and
/// hold it at the sea's level from there on: where the land drops to the
/// sea down a beach's face, the river cuts down through the beach and the
/// rise behind it, as a real mouth does, rather than tumbling down the face.
/// A river already falling to the sea over that reach (its profile under
/// the chord, as a valley's is) is left as it is. Never below a lake's
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
        let chord = sea + (top - sea) * ((s[shore] - s[i]) / run) as f32;
        water[i] = water[i].min(chord.max(water[i] - MOUTH_CUT)).max(floor[i]);
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
/// `JUNCTION_CUT` under its own profile, and deeper toward the confluence
/// (`JUNCTION_DEEPENING` more per metre over the last `JUNCTION_GORGE`).
const JUNCTION_FLAT: f32 = 6.0;
const JUNCTION_GRADE: f32 = 0.03;
const JUNCTION_CUT: f32 = 2.5;
const JUNCTION_GORGE: f32 = 20.0;
const JUNCTION_DEEPENING: f32 = 0.4;

/// Grade a tributary's water down to its parent's at their confluence, so
/// the two meet level as real confluences do, the tributary's mouth drowned
/// in its parent's water, rather than the tributary tumbling into the
/// parent over its last metres. Its channel is cut down into the ground for
/// it, a gorge where it comes down a steep valley side; farther up the cut
/// is held to `JUNCTION_CUT` and the steeper water stays there, as rapids.
fn grade_to_parent(water: &mut [f32], s: &[f64], floor: &[f32], level: f32) {
    let n = water.len();
    let end = s[n - 1];
    for i in 0..n {
        let run = (end - s[i]) as f32;
        let chord = level + JUNCTION_GRADE * (run - JUNCTION_FLAT).max(0.0);
        let cut = JUNCTION_CUT + JUNCTION_DEEPENING * (JUNCTION_GORGE - run).max(0.0);
        water[i] = water[i].min(chord.max(water[i] - cut)).max(level).max(floor[i]);
    }
}

/// The water surface's slope at each point over a few channel widths.
fn reach_slopes(points: &[PathPoint], s: &[f64], water: &[f32]) -> Vec<f32> {
    let n = water.len();
    (0..n)
        .map(|i| {
            let width = 2.0 * half_width(points[i].area as f32) as f64;
            let span = (width * 4.0).max(24.0);
            let a = s.partition_point(|&v| v < s[i] - span).min(n - 1);
            let b = s.partition_point(|&v| v <= s[i] + span).saturating_sub(1).max(a);
            if b > a { ((water[a] - water[b]) / (s[b] - s[a]) as f32).max(4e-4) } else { 4e-4 }
        })
        .collect()
}

/// The least a smoothed water surface keeps under the ground beside it.
const SMOOTHING_FREEBOARD: f32 = 0.1;

/// Water surface along a path: below the banks, never rising downstream.
/// A river running to the sea (`sea`) grades down to it over its lowest
/// reach.
fn water_profile(noise: &NoiseField, points: Vec<PathPoint>, end_level: Option<f32>, sea: bool) -> Profiled {
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
        let run = (0..n).rev().find(|&i| ground[i] >= SEA_LEVEL + MOUTH_BAR).map_or(0, |i| i + 1).max(past_lakes);
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
        let raw: Vec<f32> = (0..n).map(|i| ground[i] - bank[i]).collect();
        let fitted = fit_falling(&raw, &ground, &s);
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
                    (level - OUTLET_SLOPE * run).min(ground[i] - freeboard)
                });
                water[i].max(floor[i]).max(held)
            };
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
        if let Some(shore) = shore {
            grade_to_sea(&mut water, &s, &floor, shore);
        }
        slope = reach_slopes(&points, &s, &water);
    }
    // Smooth the surface; a positive kernel keeps it falling downstream.
    // Smoothing rounds a steep drop off on both sides, lifting the water at
    // its foot, but never above the ground there: water above its banks
    // stood on an embankment over the slope a creek tumbles into a lake by.
    let smoothed = smooth_values(&water.iter().map(|&w| w as f64).collect::<Vec<_>>(), 1.5);
    for i in 1..n.saturating_sub(1) {
        let cap = water[i].max(ground[i] - SMOOTHING_FREEBOARD);
        water[i] = (smoothed[i] as f32).min(cap).max(water[n - 1]);
    }
    for i in 0..n {
        water[i] = if points[i].lake.is_finite() { points[i].lake as f32 } else { water[i].max(floor[i]) };
    }
    if let Some(shore) = shore {
        grade_to_sea(&mut water, &s, &floor, shore);
    }
    if let Some(level) = end_level {
        grade_to_parent(&mut water, &s, &floor, level);
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

fn build_nodes(
    noise: &NoiseField,
    centreline: Vec<PathPoint>,
    end_level: Option<f32>,
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
    // Area never shrinks downstream.
    for i in 1..centreline.len() {
        centreline[i].area = centreline[i].area.max(centreline[i - 1].area);
    }
    if centreline.len() < 2 {
        return Vec::new();
    }
    let profile = water_profile(noise, centreline, end_level, sea);
    let mut nodes: Vec<RiverNode> = (0..profile.points.len())
        .map(|i| {
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
        })
        .collect();

    // Hydraulics and whitewater. A channel answers to the slope of its
    // valley over a long reach, not to every steeper or gentler stretch, so
    // it widens and narrows gradually.
    let count = nodes.len();
    let mean_spacing = ((nodes[count - 1].along - nodes[0].along) as f64 / (count - 1).max(1) as f64).max(0.5);
    let reach_slope = smooth_values(
        &nodes.iter().map(|node| node.slope as f64).collect::<Vec<_>>(),
        (40.0 / mean_spacing).clamp(2.0, 20.0),
    );
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
    // Heads begin as a seep that gathers into a channel.
    let head_length = 30.0f32;
    for node in nodes.iter_mut() {
        let grow = 0.35 + 0.65 * smoothstep(0.0, head_length, node.along);
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
    // Skew from the signed curvature: the thalweg hugs the outside of a bend.
    for (node, &k) in nodes.iter_mut().zip(&curvature) {
        let width = 2.0 * node.half_width as f64;
        // Left turn (k > 0): the outer bank is on the right (-).
        node.skew = (-k * width * 1.6).clamp(-0.6, 0.6) as f32;
    }
    nodes
}

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
const INLET_SPREAD: f32 = 0.4;
const OUTLET_SPREAD: f32 = 0.15;

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
        if !trim_steep_head(noise, &mut centreline) {
            continue;
        }
        // A river running to the sea, or into an estuary at the sea's level,
        // is graded down to it.
        let to_sea = end == RiverEnd::Sea || end_level.is_some_and(|level| level <= SEA_LEVEL + 0.03);
        let mut nodes = build_nodes(noise, centreline, end_level, path.seed, to_sea, |p| lakes.level_at(p));
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
                surface_end = nodes
                    .iter()
                    .position(|node| node.water <= SEA_LEVEL + 0.03)
                    .map_or(nodes.len() - 1, |first| first.saturating_sub(1))
                    .clamp(1, nodes.len() - 1);
            }
            RiverEnd::Edge => {}
        }
        // Where a river meets still water it spreads and slows into it rather
        // than ending as a channel: into a lake it runs into, a little where
        // it leaves one over its outlet's sill, and into the sea at its mouth.
        // Only a real crossing of a lake opens mouths, never a brush with its
        // shore. Nodes on both sides of a mouth widen alike, so the channel
        // and its ribbon open out smoothly instead of stepping wider at the
        // lake's edge. A delta or an outlet's sill shoals where a river meets
        // a lake; the tide scours an estuary deeper toward the sea.
        // (where along the river, how much it spreads, reach in half widths,
        // least reach in metres, whether it is the sea)
        let mut mouths: Vec<(f32, f32, f32, f32, bool)> = Vec::new();
        for (first, last) in lake_crossings(&nodes) {
            if first > 0 {
                mouths.push((0.5 * (nodes[first - 1].along + nodes[first].along), INLET_SPREAD, 4.0, 10.0, false));
            }
            if last + 1 < nodes.len() {
                mouths.push((0.5 * (nodes[last].along + nodes[last + 1].along), OUTLET_SPREAD, 3.0, 8.0, false));
            }
        }
        if end == RiverEnd::Sea {
            mouths.push((nodes[surface_end].along, 0.5, 8.0, 15.0, true));
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
        for node in nodes.iter_mut() {
            let (t, spread, sea) = mouths.iter().fold((0.0f32, 0.0f32, false), |best, &(at, spread, reach, least, sea)| {
                let reach = (reach * node.half_width).max(least);
                let t = 1.0 - smoothstep(0.0, reach, (node.along - at).abs());
                if t * spread > best.0 * best.1 { (t, spread, sea) } else { best }
            });
            node.half_width *= 1.0 + spread * t;
            // Water slows into a lake or the sea, but speeds over a sill as
            // it leaves a lake: an outlet keeps most of its pace and foam.
            let calm = (spread / INLET_SPREAD).min(1.0) * t;
            node.speed *= 1.0 - 0.6 * calm;
            node.turbulence *= 1.0 - calm;
            node.depth *= if sea { 1.0 + 0.25 * t } else { 1.0 - 0.6 * spread * t };
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

    let segments = build_segments(&finished);
    let origin = routing_origin(region);
    let resolution = ((REGION_SIZE + 2.0 * ROUTING_MARGIN) / GRID_CELL as f64).ceil() as usize;
    let grid_origin = [origin[0] as f32, origin[1] as f32];
    let mut grid = SegmentGrid::build(grid_origin, resolution, &segments);
    let lakes = lakes.lakes;
    let lakes: Vec<Lake> = parallel_map(&lakes, |lake| {
        let mut lake = lake.clone();
        lake.shore = shore_cells(noise, &segments, &grid, &lake);
        lake.edge = sheet_edge(noise, &segments, &grid, &lake);
        lake.current = sheet_current(&segments, &grid, &lake);
        lake
    });
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
    natural: f32,
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
        *self.cells.entry(cell).or_insert_with(|| {
            let c = Corridor::centre(cell);
            let p = [c[0] as f32, c[1] as f32];
            let envelope = super::carve::envelope_at(segments, grid, p);
            let natural = base_height(noise, p[0], p[1]);
            CellGround {
                carved: envelope.clamp(natural),
                natural,
                channel: (envelope.bank_distance < 0.0).then_some(envelope.water >= level - 0.02),
            }
        })
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
/// How far past its cells a lake's water may spread, in samples.
const BASIN_SPREAD: usize = 40;
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
) -> (Vec<([i32; 2], f32)>, Vec<([i32; 2], f32)>) {
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
                let distance = |s: [i32; 2]| ((candidate[0] - s[0]) as f32).hypot((candidate[1] - s[1]) as f32);
                let entry = next.entry(candidate).or_insert(seed);
                if distance(seed) < distance(*entry) {
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
            let lowest = (0..=16)
                .map(|k| {
                    let t = k as f32 / 16.0;
                    let p = [
                        (a[0] as f32 + (b[0] - a[0]) as f32 * t) * cell,
                        (a[1] as f32 + (b[1] - a[1]) as f32 * t) * cell,
                    ];
                    ground.cover(p)
                })
                .fold(f32::INFINITY, f32::min);
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
    corners
        .into_iter()
        .filter_map(|corner| {
            let p = [corner[0] as f32 * cell, corner[1] as f32 * cell];
            if grid.candidates(p).is_empty() {
                return None;
            }
            let envelope = super::carve::envelope_at(segments, grid, p);
            let speed = envelope.velocity[0].hypot(envelope.velocity[1]);
            (envelope.bank_distance < 0.0 && speed > 0.02).then_some((corner, envelope.velocity))
        })
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

fn build_segments(rivers: &[River]) -> Vec<RiverSegment> {
    let mut segments = Vec::new();
    for river in rivers {
        // Over its last stretch to the sea a river's banks are the shore:
        // nothing holds the ground up beside it, or the levee of a creek
        // that comes down a sea cliff would dam its own mouth.
        let mouth = (river.end == RiverEnd::Sea).then(|| {
            let end = &river.nodes[river.surface_end.min(river.nodes.len() - 1)];
            end.along - (BANK_REACH + 3.0 * end.half_width)
        });
        // So too where it runs into or out of a lake: a levee's outer slope
        // reaches a bank's width round the ends of the segments beside the
        // lake and would raise a bar across the mouth, between the river's
        // water and the lake's. Metres along the river to the nearest lake.
        let nodes = &river.nodes;
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
        for (i, pair) in river.nodes.windows(2).enumerate() {
            let (a, b) = (&pair[0], &pair[1]);
            // Under a lake the river has no channel of its own.
            if a.lake && b.lake {
                continue;
            }
            let near_sea = mouth.map_or(1.0, |mouth| 1.0 - smoothstep(mouth - 15.0, mouth, b.along));
            let clear = BANK_REACH + 2.0 * a.half_width.max(b.half_width);
            let near_lake = smoothstep(clear, clear + 15.0, to_lake[i].min(to_lake[i + 1]));
            let shore = if a.lake || b.lake { 0.0 } else { near_sea * near_lake };
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
        // Where a river leaves a lake its channel opens out of the lake's
        // bed a little before its first node: a segment's start is flat, and
        // the outlet's first one would otherwise begin as a square cut across
        // the flow, a wall at the lake's edge. This lead-in runs up out of
        // the bed and meets that start with its round end.
        for pair in river.nodes.windows(2) {
            let (a, b) = (&pair[0], &pair[1]);
            if !(a.lake && !b.lake) {
                continue;
            }
            let d = [b.position[0] - a.position[0], b.position[1] - a.position[1]];
            let l = d[0].hypot(d[1]).max(1e-3);
            let reach = 3.0 * a.half_width;
            segments.push(RiverSegment {
                a: [a.position[0] - d[0] / l * reach, a.position[1] - d[1] / l * reach],
                b: a.position,
                water: [a.water, a.water],
                half_width: [a.half_width * 1.3, a.half_width],
                depth: [0.05, a.depth],
                speed: [a.speed * 0.6, a.speed],
                bank: a.bank * 0.5,
                skew: 0.0,
                turbulence: 0.0,
                levee: 0.0,
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
                // Its groove ramps up into the seabed rather than ending in
                // a bowl under the sea.
                depth: [b.depth, 0.05],
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
}

#[cfg(test)]
mod tests {
    use super::*;

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
        flags
            .iter()
            .zip(water)
            .enumerate()
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
        assert_eq!(
            nodes.iter().map(|n| n.lake).collect::<Vec<_>>(),
            [false, true, true, true, true, true, true, true, true, false]
        );
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
        // A valley falling steeply and then gently to the shore lies under
        // the chord already.
        let valley: Vec<f32> = s.iter().map(|&v| (0.02 + 0.0004 * (200.0 - v).max(0.0).powi(2) as f32).max(0.02)).collect();
        let mut graded = valley.clone();
        grade_to_sea(&mut graded, &s, &floor, shore);
        assert_eq!(graded, valley);
    }
}
