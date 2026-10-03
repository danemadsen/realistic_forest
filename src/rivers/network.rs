//! Where the rivers run: drainage routing, channel extraction, meanders and
//! the long profile, down to the steps of a mountain creek.
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
//! 3. **Plan form.** The grid path is smoothed, pulled onto the real valley
//!    floor (the routing grid is coarser than a mountain valley), and given
//!    meanders: a Kinoshita curve whose wavelength follows the channel width
//!    and whose sinuosity grows as the valley flattens, scaled to the room
//!    the valley floor leaves. Steep creeks wander a little; lowland rivers
//!    swing in skewed loops between their bluffs.
//! 4. **Long profile.** The water surface follows the valley floor below its
//!    banks and only ever falls downstream: a reach that would have to climb
//!    (a spur a meander cuts into, the saddle a lake spills over) is carved
//!    through instead. Where the fall is steep the surface breaks into a
//!    staircase of pools and drops: step-pools on a steep creek, cascades
//!    and waterfalls where it falls off the mountainside.
//! 5. **Hydraulics.** Bankfull discharge grows with catchment; width follows
//!    downstream hydraulic geometry, and depth and speed come from Manning's
//!    equation with a roughness that grows with the slope.
//!
//! Everything is a function of world position: the grid is world-aligned and
//! every random choice is keyed by world coordinates or by a river's head
//! cell, so two regions that both contain a whole catchment agree on it.

use super::carve::{RiverSegment, RockObstacle, SegmentGrid, GRID_CELL};
use crate::constants::SEA_LEVEL;
use crate::noise::{NoiseField, base_height};
use crate::vegetation::ecology::{cell_key, noise as field_noise, random};
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
    /// Stable identity, from the head's world cell.
    pub seed: u64,
}

/// A boulder in or beside a channel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RiverRock {
    pub position: [f32; 2],
    /// Horizontal radius of the rock as placed, metres.
    pub radius: f32,
    /// Height of its top above the river bed, metres.
    pub height: f32,
    pub yaw: f32,
    /// Stable random in [0, 1): model choice, tint.
    pub seed: f32,
    /// Index of the river it sits in.
    pub river: u32,
    /// Height of the bed under its middle.
    pub bed: f32,
    /// Which `rock-N.glb` (N = model + 1), and its scale.
    pub model: u32,
    pub scale: f32,
}

impl RiverRock {
    /// Rocks settle into the bed by this share of their height.
    pub const EMBEDDED: f32 = 0.3;

    pub fn obstacle(&self) -> RockObstacle {
        RockObstacle {
            position: self.position,
            radius: self.radius,
            top: self.bed + self.height * (1.0 - Self::EMBEDDED),
        }
    }
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
    /// Carried scalars: contributing area (km²) and base-path position.
    area: f64,
    base: f64,
}

fn distance(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

fn lerp_point(a: &PathPoint, b: &PathPoint, t: f64) -> PathPoint {
    PathPoint {
        p: [a.p[0] + (b.p[0] - a.p[0]) * t, a.p[1] + (b.p[1] - a.p[1]) * t],
        area: a.area + (b.area - a.area) * t,
        base: a.base + (b.base - a.base) * t,
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

/// Wetted half width, metres.
pub fn half_width(area_km2: f32) -> f32 {
    0.5 * (1.0 + 3.2 * area_km2.max(0.0).sqrt())
}

/// Manning roughness: gravel-bed rivers to boulder-choked step-pools.
fn manning(slope: f32) -> f32 {
    0.034 + 0.05 * smoothstep(0.01, 0.12, slope)
}

/// (thalweg depth, mean speed) for a discharge in a channel of this half
/// width and slope.
pub fn depth_and_speed(discharge: f32, half_width: f32, slope: f32) -> (f32, f32) {
    let width = 2.0 * half_width.max(0.2);
    let slope = slope.max(4e-4);
    let mean_depth = (discharge * manning(slope) / (width * slope.sqrt())).powf(0.6);
    let thalweg = (mean_depth * 1.5).max(0.18);
    let speed = discharge / (width * mean_depth.max(0.05));
    (thalweg, speed.clamp(0.15, 6.0))
}

fn smoothstep(edge0: f32, edge1: f32, value: f32) -> f32 {
    let t = ((value - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
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

/// Half the width of the valley floor on each side of a point: how far the
/// ground stays within `rise` of the floor, out to 170 m.
fn valley_room(noise: &NoiseField, p: [f64; 2], normal: [f64; 2], rise: f64) -> (f64, f64) {
    const OFFSETS: [f64; 10] = [4.0, 8.0, 14.0, 22.0, 32.0, 46.0, 64.0, 90.0, 125.0, 170.0];
    let floor = base_height(noise, p[0] as f32, p[1] as f32) as f64;
    let side = |sign: f64| {
        let mut previous = (0.0, 0.0);
        for offset in OFFSETS {
            let q = [p[0] + normal[0] * offset * sign, p[1] + normal[1] * offset * sign];
            let above = base_height(noise, q[0] as f32, q[1] as f32) as f64 - floor;
            if above > rise {
                let (near, near_above) = previous;
                let t = ((rise - near_above) / (above - near_above).max(1e-6)).clamp(0.0, 1.0);
                return near + (offset - near) * t;
            }
            previous = (offset, above);
        }
        OFFSETS[OFFSETS.len() - 1]
    };
    (side(1.0), side(-1.0))
}

/// The meandering centreline over a smoothed valley-axis path.
#[allow(clippy::too_many_arguments)]
fn meander(
    noise: &NoiseField,
    base: &[PathPoint],
    seed: u64,
    taper_end: bool,
) -> Vec<PathPoint> {
    let n = base.len();
    if n < 4 {
        return base.to_vec();
    }
    let s = arc_lengths(base);
    let length = s[n - 1];
    let tangent = tangents(base);
    let normal: Vec<[f64; 2]> = tangent.iter().map(|t| [-t[1], t[0]]).collect();
    let heights: Vec<f64> = base.iter().map(|p| base_height(noise, p.p[0] as f32, p.p[1] as f32) as f64).collect();
    // Valley slope over ±60 m, the scale meanders respond to.
    let slope: Vec<f64> = (0..n)
        .map(|i| {
            let a = s.partition_point(|&v| v < s[i] - 60.0).min(n - 1);
            let b = s.partition_point(|&v| v <= s[i] + 60.0).saturating_sub(1).max(a);
            if b > a {
                ((heights[a] - heights[b]) / (s[b] - s[a]).max(1.0)).max(0.0)
            } else {
                0.0
            }
        })
        .collect();
    let slope = smooth_values(&slope, 4.0);
    // Valley-floor room every few points, interpolated.
    let stride = 4usize;
    let mut room_left = vec![0.0; n];
    let mut room_right = vec![0.0; n];
    let mut sampled = Vec::new();
    for i in (0..n).step_by(stride).chain(std::iter::once(n - 1)) {
        let half_width = half_width(base[i].area as f32) as f64;
        let (left, right) = valley_room(noise, base[i].p, normal[i], 1.2 + 0.5 * half_width);
        sampled.push((i, left, right));
    }
    for window in sampled.windows(2) {
        let (i0, l0, r0) = window[0];
        let (i1, l1, r1) = window[1];
        for i in i0..=i1 {
            let t = if i1 > i0 { (i - i0) as f64 / (i1 - i0) as f64 } else { 0.0 };
            room_left[i] = l0 + (l1 - l0) * t;
            room_right[i] = r0 + (r1 - r0) * t;
        }
    }
    let room_left = smooth_values(&room_left, 6.0);
    let room_right = smooth_values(&room_right, 6.0);

    // Integrate the Kinoshita curve along its own arc length, sampling the
    // valley's properties at the base position it has reached.
    let at = |x: f64| -> usize { s.partition_point(|&v| v < x).min(n - 1) };
    let river_offset = (seed % 100_000) as f64 * 3.17;
    let step = 2.0;
    let mut x = 0.0f64;
    let mut y = 0.0f64;
    let mut phase = random(seed, 7) as f64 * std::f64::consts::TAU;
    let mut sigma = 0.0;
    let mut curve: Vec<(f64, f64, f64)> = Vec::new(); // (x, y, wavelength)
    let mut guard = 0;
    while x < length && guard < 2_000_000 {
        guard += 1;
        let i = at(x);
        let width = 2.0 * half_width(base[i].area as f32) as f64;
        // Meander trains lengthen and shorten, tighten and relax, over a few
        // loops: no two bends alike.
        let nominal = 11.0 * width.max(2.6);
        let wobble = field_noise([sigma, river_offset], nominal * 2.6, 301) as f64;
        let wavelength = ((6.5 + 10.0 * wobble) * width.max(2.6)).clamp(26.0, 900.0);
        phase += std::f64::consts::TAU * step / wavelength;
        let lowland = 1.0 - 0.82 * smoothstep(0.004, 0.07, slope[i] as f32) as f64;
        let vigour = 0.25 + 0.95 * field_noise([sigma, river_offset + 77.0], nominal * 3.3, 302) as f64;
        let theta0 = (1.35 * lowland * vigour).min(1.55);
        let (js, jf) = (0.035, 0.02);
        let mut theta = theta0 * phase.sin()
            + theta0.powi(3) * (js * (3.0 * phase).cos() - jf * (3.0 * phase).sin());
        // Small irregularities: a gravel bar, a tree root, a harder bed.
        theta += 0.45 * (field_noise([sigma, river_offset + 191.0], width * 4.0 + 10.0, 303) as f64 - 0.5);
        x += theta.cos() * step;
        y += theta.sin() * step;
        sigma += step;
        curve.push((x, y, wavelength));
    }
    if curve.is_empty() {
        return base.to_vec();
    }
    // Remove the slow drift of the lateral offset, then the loop envelope.
    let ys: Vec<f64> = curve.iter().map(|c| c.1).collect();
    let mean_wavelength = curve.iter().map(|c| c.2).sum::<f64>() / curve.len() as f64;
    let drift = smooth_values(&ys, (mean_wavelength / step) * 0.6);
    let lateral: Vec<f64> = ys.iter().zip(&drift).map(|(y, d)| y - d).collect();
    let window = ((mean_wavelength / step) * 0.5).ceil() as usize;
    let mut envelope = vec![0.0; lateral.len()];
    {
        // Running maximum of |y| over ±half a wavelength.
        let mut deque: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
        let absolute: Vec<f64> = lateral.iter().map(|v| v.abs()).collect();
        let m = absolute.len();
        let mut right = 0;
        for (i, slot) in envelope.iter_mut().enumerate() {
            while right < m && right <= i + window {
                while deque.back().is_some_and(|&b| absolute[b] <= absolute[right]) {
                    deque.pop_back();
                }
                deque.push_back(right);
                right += 1;
            }
            while deque.front().is_some_and(|&f| f + window < i) {
                deque.pop_front();
            }
            *slot = absolute[*deque.front().unwrap()];
        }
    }
    let envelope = smooth_values(&envelope, window as f64 * 0.5);

    let mut out = Vec::with_capacity(curve.len() / 2 + 2);
    out.push(base[0]);
    for (k, &(cx, _, wavelength)) in curve.iter().enumerate() {
        if k % 2 == 1 {
            continue;
        }
        let x = cx.clamp(0.0, length);
        let i = at(x).max(1);
        let t = ((x - s[i - 1]) / (s[i] - s[i - 1]).max(1e-9)).clamp(0.0, 1.0);
        let p = lerp_point(&base[i - 1], &base[i], t);
        let nrm = [
            normal[i - 1][0] + (normal[i][0] - normal[i - 1][0]) * t,
            normal[i - 1][1] + (normal[i][1] - normal[i - 1][1]) * t,
        ];
        let nl = nrm[0].hypot(nrm[1]).max(1e-9);
        let nrm = [nrm[0] / nl, nrm[1] / nl];
        let half_width = half_width(p.area as f32) as f64;
        let left = room_left[i - 1] + (room_left[i] - room_left[i - 1]) * t;
        let right = room_right[i - 1] + (room_right[i] - room_right[i - 1]) * t;
        // Swing about the middle of the floor, within its width.
        let axis = ((left - right) * 0.5).clamp(-60.0, 60.0);
        let allowed = ((left + right) * 0.5 - 1.2 * half_width).max(0.0);
        let mut scale = (allowed / envelope[k].max(1e-3)).min(1.0);
        // Grow out of the head; settle onto the parent at a confluence.
        scale *= smoothstep(0.0, (wavelength * 0.8) as f32, x as f32) as f64;
        let mut axis_weight = smoothstep(0.0, 120.0, x as f32) as f64;
        if taper_end {
            let to_end = length - x;
            scale *= smoothstep((wavelength * 0.2) as f32, (wavelength * 1.2) as f32, to_end as f32) as f64;
            axis_weight *= smoothstep(10.0, 140.0, to_end as f32) as f64;
        }
        let offset = lateral[k] * scale + axis * axis_weight;
        out.push(PathPoint {
            p: [p.p[0] + nrm[0] * offset, p.p[1] + nrm[1] * offset],
            area: p.area,
            base: p.base,
        });
    }
    let last = *base.last().unwrap();
    if distance(out.last().unwrap().p, last.p) < 1.0 {
        out.pop();
    }
    out.push(last);
    out
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
    let mut water = vec![0.0f32; n];
    let mut slope = vec![0.01f32; n];
    // Two passes: the bank height depends on the depth, which depends on the
    // slope of the surface the first pass found.
    for pass in 0..2 {
        for i in 0..n {
            let bank = if pass == 0 {
                0.45
            } else {
                let (depth, _) = depth_and_speed(discharge(points[i].area as f32), half_width(points[i].area as f32), slope[i]);
                0.25 + 0.6 * depth
            };
            let raw = ground[i] - bank;
            water[i] = if i == 0 {
                raw
            } else {
                raw.min(water[i - 1] - 4e-4 * (s[i] - s[i - 1]) as f32)
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
        let maximum_drop = 1.2 + 6.5 * smoothstep(0.12, 0.6, slope) + 7.0 * smoothstep(0.6, 1.4, slope);
        let drop = (slope * 2.6 * width * jitter).clamp(0.22, maximum_drop);
        let spacing = ((drop / slope) as f64).max(1.6 * width as f64);
        let end = position + spacing;
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
        let (depth, _) = depth_and_speed(node.discharge, node.half_width, node.slope);
        // A plunge pool is scoured deep beneath the fall and shoals toward
        // the next lip.
        let scour = (0.35 * last_drop).min(2.5) * (1.0 - smoothstep(0.0, 3.0 * node.half_width + 2.0, since_fall));
        node.depth = depth + scour;
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
    pub rocks: Vec<RiverRock>,
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

    // Plan form, in parallel within each generation of the tree is not
    // needed: tributaries only read their parent's finished centreline.
    let mut rivers: Vec<Option<River>> = vec![None; paths.len()];
    // Which finished node of the parent each confluence lands on.
    for &index in &order {
        let path = &paths[index];
        let mut points: Vec<PathPoint> = path
            .cells
            .iter()
            .map(|&cell| PathPoint {
                p: routing.centre(cell as usize),
                area: (routing.area[cell as usize] * CELL_AREA_KM2) as f64,
                base: 0.0,
            })
            .collect();
        // A tributary's last cell is its parent's; it should not carry the
        // parent's catchment.
        if let RiverEnd::Confluence(..) = path.end {
            let n = points.len();
            if n >= 2 {
                points[n - 1].area = points[n - 2].area;
            }
        }
        if points.len() < 2 {
            continue;
        }
        let mut base = resample(&points, |_| 8.0);
        smooth(&mut base, 2.0);
        snap_to_floor(noise, &mut base, 40.0, 4.0);
        let mut base = resample(&base, |_| 8.0);
        smooth(&mut base, 1.5);
        snap_to_floor(noise, &mut base, 16.0, 2.0);
        // Smooth the valley axis enough that meander loops can ride on it.
        let typical_width = 2.0 * half_width(base[base.len() / 2].area as f32) as f64;
        smooth(&mut base, (typical_width * 1.2 / 8.0).clamp(1.5, 8.0));
        let tributary = matches!(path.end, RiverEnd::Confluence(..));
        if let RiverEnd::Confluence(parent, _) = path.end
            && let Some(parent_river) = rivers[parent].as_ref()
        {
            // A tributary that runs down the same valley floor beside its
            // parent joins it where it first comes close, rather than
            // threading through the parent's meanders.
            let own = half_width(base[base.len() / 2].area as f32) as f64;
            for i in 1..base.len() {
                let (k, point) = nearest_on_river(parent_river, base[i].p);
                let reach = (parent_river.nodes[k].half_width as f64 + own) * 2.0 + 14.0;
                if distance(point, base[i].p) < reach {
                    base.truncate(i + 1);
                    let last = base.len() - 1;
                    base[last].p = point;
                    break;
                }
            }
            if base.len() < 2 {
                continue;
            }
        }
        let mut centreline = meander(noise, &base, path.seed, tributary);
        let mut end_level = None;
        if let RiverEnd::Confluence(parent, _) = path.end {
            // Land on the parent's finished centreline, near where the grid
            // path met it.
            if let Some(parent_river) = rivers[parent].as_ref() {
                let target = centreline.last().unwrap().p;
                let (node_index, point) = nearest_on_river(parent_river, target);
                let delta = [point[0] - target[0], point[1] - target[1]];
                let s = arc_lengths(&centreline);
                let total = *s.last().unwrap();
                let blend = (4.0 * parent_river.nodes[node_index].half_width as f64).clamp(40.0, 160.0);
                for (p, &si) in centreline.iter_mut().zip(&s) {
                    let w = smoothstep((total - blend) as f32, total as f32, si as f32) as f64;
                    p.p[0] += delta[0] * w;
                    p.p[1] += delta[1] * w;
                }
                end_level = Some(parent_river.nodes[node_index].water);
            }
        }
        relax_tight_bends(&mut centreline);
        let mut nodes = build_nodes(noise, centreline, end_level, path.seed);
        if nodes.len() < 2 {
            continue;
        }
        let mut surface_end = nodes.len() - 1;
        match path.end {
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
            end: path.end,
            surface_end,
            seed: path.seed,
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
    let grid = SegmentGrid::build(grid_origin, resolution, &segments, &[]);
    let rocks = super::rocks::place_rocks(noise, &finished, &segments, &grid);
    let obstacles: Vec<RockObstacle> = rocks.iter().map(RiverRock::obstacle).collect();
    let grid = SegmentGrid::build(grid_origin, resolution, &segments, &obstacles);
    let surface = super::surface::build(&finished);
    RiverNetwork {
        region,
        rivers: finished,
        segments,
        grid,
        rocks,
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
                levee: smoothstep(SEA_LEVEL + 0.05, SEA_LEVEL + 0.6, a.water.min(b.water)),
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
