//! The water surfaces: a ribbon across each channel at the water level, and
//! a flat sheet over each lake.
//!
//! A ribbon's cross-sections sit at the river's nodes and reach a little
//! past each bank, so the waterline is wherever the carved bank rises through
//! the water and never a mesh edge. Every vertex carries the flow there (the
//! current is fastest over the thalweg and stalls at the banks), where across
//! the channel it lies and its whitewater, which is what the water shader
//! animates and foams.
//!
//! A lake is a flat sheet at its level over its basin and the shore cells
//! around it, so the shoreline is wherever the ground rises through it, but
//! never past its outlet, where the ground falls away. Where a river meets a
//! lake, its ribbon runs on just under the lake's sheet, a little deeper the
//! further inside, and stops only well inside. Where the sheet's edge
//! crosses the river's channel it slips just under the river's water
//! (`network::sheet_edge`), so the two surfaces cross inside the sheet's last
//! cell with no gap, step or shared plane, and the current runs on into the
//! sheet (`Lake::current`). At the sea, water at the sea's level is the sea's
//! and the ribbon sinks under it.
//!
//! The mesh is cut into 256 m chunks with their bounds, so the renderer only
//! draws what the camera can see.

use super::network::{FLOW_CELL, Lake, River, RiverEnd};
use crate::constants::SEA_LEVEL;
use std::collections::HashMap;

/// One water-surface vertex: 48 bytes, mirrored by the river vertex inputs
/// in water-surface.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SurfaceVertex {
    pub position: [f32; 3],
    /// Surface current, world XZ, m/s.
    pub velocity: [f32; 2],
    /// Across the channel in half widths: -1 at the left waterline, +1 at
    /// the right; beyond them the ribbon runs under the banks.
    pub across: f32,
    /// Whitewater, 0..1.
    pub turbulence: f32,
    /// 1 on a lake's still water, and on a river where it meets still water
    /// (a lake's sheet, or the sea past its mouth), fading to 0 some 30 m
    /// along the river from it. It is how much the shader holds the surface
    /// at its own height rather than lifting a distant river over coarse
    /// terrain.
    pub still: f32,
    /// Metres downstream from the river's head. With `across` times
    /// `half_width` it is the ribbon's own frame, in which the shader lays
    /// its ripples and standing waves so they follow every bend. 0 on a lake.
    pub along: f32,
    /// The channel's half width here, metres; 0 on a lake's sheet, which is
    /// how the shader tells the two apart.
    pub half_width: f32,
    /// Foam made by whitewater here or upstream and still drifting, 0..1.
    pub foam: f32,
    /// How far the water has turned to the sea's, 0..1: 1 where a river
    /// meets the sea at its mouth, fading to 0 some `SEA_BLEND` metres up it.
    /// The shader blends the river's tea-coloured water into the sea's
    /// green over that reach, so the two meet in one colour.
    pub sea: f32,
}

const _: () = assert!(std::mem::size_of::<SurfaceVertex>() == 48);

/// Side of a culling chunk, metres.
pub const CHUNK: f32 = 256.0;
/// Side of a lake surface's cells: the flow grid's.
const LAKE_CELL: f32 = FLOW_CELL as f32;

/// A river within this much of a lake's level counts as standing at it.
const LEVEL_TIE: f32 = 0.02;
/// A river further than this under the level of a lake's sheet over it is not
/// the lake's water (a creek well below a sill), and keeps to its own surface.
const SHEET_BAND: f32 = 0.3;
/// How far a ribbon lies under the sheet of the lake it runs into, out of or
/// through: just under it near the sheet's edge, where the two cross, and
/// deeper well inside. It sinks over `SINK_START..HIDDEN_DEPTH` metres inside
/// the edge, and past `HIDDEN_DEPTH` it is hidden for good and left out.
const UNDER_SHEET: f32 = 0.015;
const UNDER_SHEET_DEEP: f32 = 0.12;
const SINK_START: f32 = 3.0;
const HIDDEN_DEPTH: f32 = 9.0;
/// Metres along a river from still water over which its distant lift returns.
const LIFT_FADE: f32 = 30.0;
/// Seconds whitewater's foam lasts on the water as it drifts downstream.
const FOAM_LIFETIME: f32 = 15.0;
/// How far under the sea a river's surface tucks where it ends at the coast,
/// below the troughs of the waves running into its mouth: the sea's own
/// surface is drawn on from there, in the same colour.
const UNDER_SEA: f32 = 0.2;
/// Metres up a river from the coast over which its water turns to the sea's.
const SEA_BLEND: f32 = 90.0;
/// How far under its spring's ground a stream's surface begins.
const SPRING_SINK: f32 = 0.3;

fn smoothstep(edge0: f32, edge1: f32, value: f32) -> f32 {
    let t = ((value - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn under_sheet(depth: f32) -> f32 {
    UNDER_SHEET + (UNDER_SHEET_DEEP - UNDER_SHEET) * smoothstep(SINK_START, HIDDEN_DEPTH, depth)
}

/// Every lake sheet's cells, with the level of the water over each, and the
/// height of the sheets' corners (sunk at their edges), the highest where
/// two sheets meet.
struct Sheets {
    cells: HashMap<[i32; 2], f32>,
    corners: HashMap<[i32; 2], f32>,
}

impl Sheets {
    fn new(lakes: &[Lake]) -> Self {
        let mut cells: HashMap<[i32; 2], f32> = HashMap::new();
        let mut corners: HashMap<[i32; 2], f32> = HashMap::new();
        for lake in lakes {
            let sunk: HashMap<[i32; 2], f32> = lake.edge.iter().copied().collect();
            for &cell in lake.cells.iter().chain(&lake.shore) {
                let level = cells.entry(cell).or_insert(lake.level);
                *level = level.max(lake.level);
                for corner in [cell, [cell[0] + 1, cell[1]], [cell[0], cell[1] + 1], [cell[0] + 1, cell[1] + 1]] {
                    let height = sunk.get(&corner).map_or(lake.level, |&h| h.min(lake.level));
                    let entry = corners.entry(corner).or_insert(height);
                    *entry = entry.max(height);
                }
            }
        }
        Sheets { cells, corners }
    }

    /// The drawn sheet's height over `p`, over the two triangles each cell
    /// is drawn as (see `add_lake`), if a sheet is drawn there.
    fn height(&self, p: [f32; 2]) -> Option<f32> {
        let cell = Self::cell_of(p);
        self.cells.get(&cell)?;
        let corner = |dx: i32, dz: i32| self.corners.get(&[cell[0] + dx, cell[1] + dz]).copied();
        let (u, v) = (p[0] / LAKE_CELL - cell[0] as f32, p[1] / LAKE_CELL - cell[1] as f32);
        let (h00, h11) = (corner(0, 0)?, corner(1, 1)?);
        Some(if v >= u {
            let h01 = corner(0, 1)?;
            h00 + (h11 - h01) * u + (h01 - h00) * v
        } else {
            let h10 = corner(1, 0)?;
            h00 + (h10 - h00) * u + (h11 - h10) * v
        })
    }

    fn cell_of(p: [f32; 2]) -> [i32; 2] {
        [(p[0] / LAKE_CELL).floor() as i32, (p[1] / LAKE_CELL).floor() as i32]
    }

    fn level(&self, p: [f32; 2]) -> Option<f32> {
        self.cells.get(&Self::cell_of(p)).copied()
    }

    /// The level of the sheet over `p`, and how far inside the sheet's edge
    /// `p` lies (searched out a little past `HIDDEN_DEPTH`).
    fn under(&self, p: [f32; 2]) -> Option<(f32, f32)> {
        let level = self.level(p)?;
        let cell = Self::cell_of(p);
        let reach = (HIDDEN_DEPTH / LAKE_CELL).ceil() as i32 + 1;
        let mut depth = f32::INFINITY;
        for dz in -reach..=reach {
            for dx in -reach..=reach {
                let other = [cell[0] + dx, cell[1] + dz];
                if self.cells.contains_key(&other) {
                    continue;
                }
                let (x0, z0) = (other[0] as f32 * LAKE_CELL, other[1] as f32 * LAKE_CELL);
                let gap_x = (x0 - p[0]).max(p[0] - x0 - LAKE_CELL).max(0.0);
                let gap_z = (z0 - p[1]).max(p[1] - z0 - LAKE_CELL).max(0.0);
                depth = depth.min(gap_x.hypot(gap_z));
            }
        }
        Some((level, depth))
    }
}

/// A contiguous run of indices and its bounds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceChunk {
    pub minimum: [f32; 3],
    pub maximum: [f32; 3],
    pub first_index: u32,
    pub index_count: u32,
}

#[derive(Clone, Debug, Default)]
pub struct SurfaceMesh {
    pub vertices: Vec<SurfaceVertex>,
    pub indices: Vec<u32>,
    pub chunks: Vec<SurfaceChunk>,
}

fn normalize(v: [f32; 2]) -> [f32; 2] {
    let length = v[0].hypot(v[1]);
    if length < 1e-6 { [1.0, 0.0] } else { [v[0] / length, v[1] / length] }
}

/// Speed profile across the channel, as the carve uses it.
fn lateral(across: f32) -> f32 {
    (1.0 - (across * across).min(1.0)).powf(0.35) * 1.25
}

pub fn build(rivers: &[River], lakes: &[Lake]) -> SurfaceMesh {
    let mut vertices: Vec<SurfaceVertex> = Vec::new();
    // Triangles per chunk, gathered before they are laid out chunk by chunk.
    let mut chunk_triangles: std::collections::BTreeMap<(i32, i32), Vec<u32>> = Default::default();
    let add_quads = |triangles: &mut std::collections::BTreeMap<(i32, i32), Vec<u32>>,
                         vertices: &[SurfaceVertex],
                         row_a: u32,
                         row_b: u32,
                         columns: u32| {
        let a = vertices[row_a as usize].position;
        let b = vertices[(row_b + columns / 2) as usize].position;
        let key = (
            (((a[0] + b[0]) * 0.5) / CHUNK).floor() as i32,
            (((a[2] + b[2]) * 0.5) / CHUNK).floor() as i32,
        );
        let list = triangles.entry(key).or_default();
        for c in 0..columns - 1 {
            let (i0, i1, j0, j1) = (row_a + c, row_a + c + 1, row_b + c, row_b + c + 1);
            list.extend_from_slice(&[i0, j0, j1, i0, j1, i1]);
        }
    };
    let sheets = Sheets::new(lakes);
    let seas = sea_blends(rivers);
    for (river, &(sea_share, sea_along, sea_ramp)) in rivers.iter().zip(&seas) {
        let nodes = &river.nodes;
        // A river's surface runs on past where its own water ends, sunk
        // under the still water it meets, so it never ends in an edge: one
        // node into its parent's channel, and out past the shore under the
        // sea.
        let last = nodes.len().saturating_sub(1);
        // Where a river's water comes down to the sea's level: the coast,
        // from which the sea's own surface fills its mouth.
        let coast = coast_of(river);
        let (end, sink) = match (coast, river.end) {
            // One node on past the coast, tucked under the sea.
            (Some(coast), _) => ((coast + 1).min(last), 0.0),
            (None, RiverEnd::Confluence(..)) => ((river.surface_end + 1).min(last), 0.12),
            (None, _) => (river.surface_end.min(last), 0.0),
        };
        if end < 1 {
            continue;
        }
        let widest = nodes[..=end].iter().map(|n| n.half_width).fold(0.0f32, f32::max);
        let offsets: &[f32] = if widest > 1.6 {
            &[-1.12, -0.55, 0.0, 0.55, 1.12]
        } else {
            &[-1.15, 0.0, 1.15]
        };
        let columns = offsets.len() as u32;
        // Direction of the reach through each node.
        let direction = |i: usize| -> [f32; 2] {
            let pick = |a: usize, b: usize| {
                let d = [nodes[b].position[0] - nodes[a].position[0], nodes[b].position[1] - nodes[a].position[1]];
                (d[0].hypot(d[1]) > 0.5).then_some(d)
            };
            let before = if i > 0 { pick(i - 1, i) } else { None };
            let after = if i < end { pick(i, i + 1) } else { None };
            let wide = if i > 0 && i < end { pick(i - 1, i + 1) } else { None };
            normalize(wide.or(after).or(before).unwrap_or([1.0, 0.0]))
        };
        // A stream rising at a spring comes out of the ground: its first row
        // narrows to a point and lies under the ground there.
        let spring = !nodes[0].lake;
        // Each row's vertices across the channel.
        let row_positions = |i: usize| {
            let node = &nodes[i];
            let tangent = direction(i);
            let normal = [-tangent[1], tangent[0]];
            let half_width = if spring && i == 0 { 0.1 } else { node.half_width.max(0.25) };
            offsets
                .iter()
                .map(|&across| {
                    [node.position[0] + normal[0] * across * half_width, node.position[1] + normal[1] * across * half_width]
                })
                .collect()
        };
        let rows: Vec<Vec<[f32; 2]>> = (0..=end).map(row_positions).collect();
        // How far along the river each row lies from still water: a lake's
        // sheet, or the sea past the river's mouth. Neither is lifted in the
        // distance, so the ribbon must not be lifted where it meets them.
        let node_touches_still_water = |i: usize| {
            let water = nodes[i].water;
            water <= SEA_LEVEL + 0.03
                || (river.end == RiverEnd::Sea && i >= river.surface_end)
                || rows[i].iter().any(|&p| {
                    sheets.level(p).is_some_and(|lake| water <= lake + LEVEL_TIE && water >= lake - SHEET_BAND)
                })
        };
        let touches: Vec<bool> = (0..=end).map(node_touches_still_water).collect();
        let mut from_still = vec![f32::INFINITY; end + 1];
        let mut seen: Option<f32> = None;
        for ((node, &touch), near) in nodes[..=end].iter().zip(&touches).zip(from_still.iter_mut()) {
            if touch {
                seen = Some(node.along);
            }
            if let Some(at) = seen {
                *near = node.along - at;
            }
        }
        seen = None;
        let near_slots = nodes[..=end].iter().zip(&touches).zip(from_still.iter_mut());
        for ((node, &touch), near) in near_slots.rev() {
            if touch {
                seen = Some(node.along);
            }
            if let Some(at) = seen {
                *near = near.min(at - node.along);
            }
        }
        // Foam the whitewater makes drifts on downstream and thins as it
        // goes: a reach carries its own whitewater's or what is left of the
        // foam made above it, which draws foam lines down the pool below
        // every riffle and rapid.
        let mut foam = vec![0.0f32; end + 1];
        let mut carried = 0.0f32;
        for i in 0..=end {
            if i > 0 {
                let step = (nodes[i].along - nodes[i - 1].along).max(0.0);
                carried *= (-step / (FOAM_LIFETIME * nodes[i - 1].speed.max(0.3))).exp();
            }
            carried = carried.max(nodes[i].turbulence);
            foam[i] = carried;
        }
        let mut previous_row: Option<(u32, bool)> = None;
        for i in 0..=end {
            let node = &nodes[i];
            let tangent = direction(i);
            // Past where its own water ends, the river runs on sunk under the
            // still water it meets (the sea, its parent); and water standing
            // at the sea's level is the sea's, drawn by the sea itself.
            let mut water = if i > river.surface_end { node.water - sink } else { node.water };
            if spring && i == 0 {
                water -= SPRING_SINK;
            }
            if coast.is_some_and(|coast| i > coast) {
                water = water.min(SEA_LEVEL - UNDER_SEA);
            }
            let sea = sea_share * (1.0 - smoothstep(0.0, sea_ramp, sea_along - node.along));
            let still = 1.0 - smoothstep(0.0, LIFT_FADE, from_still[i]);
            let row = vertices.len() as u32;
            let mut hidden = true;
            for (&across, &p) in offsets.iter().zip(&rows[i]) {
                // Under the sheet of a lake whose level it stands at (running
                // into it, out of it or through it), the ribbon runs on just
                // beneath the sheet as drawn, its edge sunk and all, and
                // deeper the further inside. The two never share a plane,
                // the ribbon never shows over the lake's water, and its end
                // is never in sight. Well inside, where the sheet hides it for
                // good, it stops.
                let (level, deep) = match sheets.under(p) {
                    Some((lake, depth)) if water <= lake + LEVEL_TIE && water >= lake - SHEET_BAND => {
                        let sheet = sheets.height(p).unwrap_or(lake).min(lake);
                        (water.min(sheet) - under_sheet(depth), depth >= HIDDEN_DEPTH)
                    }
                    _ => (water, false),
                };
                hidden &= deep;
                let speed = node.speed * lateral(across);
                let vertex = SurfaceVertex {
                    position: [p[0], level, p[1]],
                    velocity: [tangent[0] * speed, tangent[1] * speed],
                    across,
                    turbulence: node.turbulence,
                    still,
                    along: node.along,
                    half_width: node.half_width.max(0.25),
                    foam: foam[i],
                    sea,
                };
                vertices.push(vertex);
            }
            if let Some((previous, previous_hidden)) = previous_row
                && !(hidden && previous_hidden)
            {
                add_quads(&mut chunk_triangles, &vertices, previous, row, columns);
            }
            previous_row = Some((row, hidden));
        }
    }

    for lake in lakes {
        add_lake(&mut vertices, &mut chunk_triangles, lake);
    }

    let mut indices = Vec::new();
    let mut chunks = Vec::new();
    for (_, triangles) in chunk_triangles {
        let mut minimum = [f32::INFINITY; 3];
        let mut maximum = [f32::NEG_INFINITY; 3];
        for &index in &triangles {
            let p = vertices[index as usize].position;
            for axis in 0..3 {
                minimum[axis] = minimum[axis].min(p[axis]);
                maximum[axis] = maximum[axis].max(p[axis]);
            }
        }
        let chunk = SurfaceChunk {
            minimum,
            maximum,
            first_index: indices.len() as u32,
            index_count: triangles.len() as u32,
        };
        chunks.push(chunk);
        indices.extend_from_slice(&triangles);
    }
    SurfaceMesh {
        vertices,
        indices,
        chunks,
    }
}

/// Where a river comes down to the sea's level, if it does: the coast, from
/// which the sea's own surface fills its channel. A river to the sea always
/// has one (its mouth at the latest); so does a tributary that joins a river
/// in its estuary, at the sea's level.
fn coast_of(river: &River) -> Option<usize> {
    let reached = river.nodes.iter().position(|n| n.water <= SEA_LEVEL + 0.03);
    match river.end {
        RiverEnd::Sea => Some(reached.unwrap_or(river.surface_end).min(river.surface_end).max(1)),
        _ => reached.filter(|&c| c <= river.surface_end).map(|c| c.max(1)),
    }
}

/// How far each river's water has turned to the sea's at each node: all of
/// it at a river's coast, fading over `SEA_BLEND` metres up it (or less, to
/// none where it leaves a coastal lagoon nearer the coast than that), and a
/// tributary joining it takes on what its parent has at the confluence,
/// fading the same way up the tributary, so no seam of colour crosses the
/// water where they meet. Per river: the share at its end (the coast, or
/// the confluence), how far along the river that is, and the ramp's length.
fn sea_blends(rivers: &[River]) -> Vec<(f32, f32, f32)> {
    let mut ends: Vec<Option<(f32, f32, f32)>> = vec![None; rivers.len()];
    fn ramp_to(river: &River, end: usize) -> f32 {
        // From the last lake above the end, where the water was still.
        let along = river.nodes[end].along;
        river.nodes[..=end]
            .iter()
            .rposition(|n| n.lake)
            .map_or(SEA_BLEND, |k| (along - river.nodes[k].along).clamp(1.0, SEA_BLEND))
    }
    fn resolve(rivers: &[River], ends: &mut Vec<Option<(f32, f32, f32)>>, index: usize, depth: usize) -> (f32, f32, f32) {
        if let Some(end) = ends[index] {
            return end;
        }
        let river = &rivers[index];
        let Some(last) = river.nodes.last() else {
            return (0.0, 0.0, SEA_BLEND);
        };
        let end = match (coast_of(river), river.end) {
            (Some(coast), _) => {
                let coast = coast.min(river.nodes.len() - 1);
                (1.0, river.nodes[coast].along, ramp_to(river, coast))
            }
            (None, RiverEnd::Confluence(parent, _)) if depth < 8 && parent < rivers.len() => {
                let (share, along, ramp) = resolve(rivers, ends, parent, depth + 1);
                // The parent's node nearest where the tributary ends.
                let closer_to_last = |a: &&super::network::RiverNode, b: &&super::network::RiverNode| {
                    let d = |n: &&super::network::RiverNode| {
                        (n.position[0] - last.position[0]).hypot(n.position[1] - last.position[1])
                    };
                    d(a).total_cmp(&d(b))
                };
                let here = rivers[parent]
                    .nodes
                    .iter()
                    .min_by(closer_to_last)
                    .map_or(f32::NEG_INFINITY, |n| n.along);
                let share = share * (1.0 - smoothstep(0.0, ramp, along - here));
                (share, last.along, ramp_to(river, river.nodes.len() - 1))
            }
            _ => (0.0, last.along, SEA_BLEND),
        };
        ends[index] = Some(end);
        end
    }
    (0..rivers.len()).map(|index| resolve(rivers, &mut ends, index, 0)).collect()
}

/// A lake's sheet: a quad over every flow-grid cell of its water and of its
/// shore, the cells around it where the ground rises through the water. Its
/// outer edge sinks under the ground wherever that lies lower than the
/// water, so the sheet never ends in the air.
fn add_lake(
    vertices: &mut Vec<SurfaceVertex>,
    chunk_triangles: &mut std::collections::BTreeMap<(i32, i32), Vec<u32>>,
    lake: &Lake,
) {
    let sunk: HashMap<[i32; 2], f32> = lake.edge.iter().copied().collect();
    let current: HashMap<[i32; 2], [f32; 2]> = lake.current.iter().copied().collect();
    let mut corner_index = HashMap::new();
    let mut corner = |vertices: &mut Vec<SurfaceVertex>, x: i32, z: i32| -> u32 {
        let push_vertex = || {
            let height = sunk.get(&[x, z]).map_or(lake.level, |&h| h.min(lake.level));
            vertices.push(SurfaceVertex {
                position: [x as f32 * LAKE_CELL, height, z as f32 * LAKE_CELL],
                // A river's current runs on a little way into the lake at its
                // mouth (network::sheet_current).
                velocity: current.get(&[x, z]).copied().unwrap_or([0.0, 0.0]),
                across: 0.0,
                turbulence: 0.0,
                still: 1.0,
                along: 0.0,
                half_width: 0.0,
                foam: 0.0,
                sea: 0.0,
            });
            vertices.len() as u32 - 1
        };
        *corner_index.entry((x, z)).or_insert_with(push_vertex)
    };
    for &[x, z] in lake.cells.iter().chain(&lake.shore) {
        let i00 = corner(vertices, x, z);
        let i10 = corner(vertices, x + 1, z);
        let i01 = corner(vertices, x, z + 1);
        let i11 = corner(vertices, x + 1, z + 1);
        let key = (
            (((x as f32 + 0.5) * LAKE_CELL) / CHUNK).floor() as i32,
            (((z as f32 + 0.5) * LAKE_CELL) / CHUNK).floor() as i32,
        );
        chunk_triangles
            .entry(key)
            .or_default()
            .extend_from_slice(&[i00, i01, i11, i00, i11, i10]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rivers::network::RiverNode;

    /// A river runs into a lake, through it and out of it. Under the lake's
    /// sheet its ribbon lies just beneath the sheet near the edge and deeper
    /// inside, and is left out only well inside. Outside the sheet it is at
    /// its own water. Where it meets the lake it is held from the distant
    /// lift.
    #[test]
    fn a_ribbon_runs_on_under_its_lake_and_out_of_it() {
        let level = 5.0;
        // The sheet covers x 0..48 m, z -16..16 m.
        let cells: Vec<[i32; 2]> = (0..12).flat_map(|x| (-4..4).map(move |z| [x, z])).collect();
        let lake = Lake { level, cells, shore: Vec::new(), edge: Vec::new(), current: Vec::new() };
        let water_at = |x: f32| if x < 48.0 { level } else { level - 0.02 * (x - 48.0) };
        let river_node = |k: i32| {
            let x = -20.0 + 2.5 * k as f32;
            RiverNode {
                position: [x, 1.0],
                water: water_at(x),
                half_width: 1.2,
                depth: 0.4,
                speed: 0.6,
                along: x + 20.0,
                lake: (0.0..48.0).contains(&x),
                ..Default::default()
            }
        };
        let nodes: Vec<RiverNode> = (0..=40).map(river_node).collect();
        let river = River { nodes, end: RiverEnd::Edge, surface_end: 40 };
        let mesh = build(&[river], &[lake]);
        // The ribbon's 41 rows of three vertices come first, then the sheet.
        let ribbon = 41 * 3;
        let depth = |p: [f32; 3]| -> Option<f32> {
            let inside = (0.0..48.0).contains(&p[0]) && (-16.0..16.0).contains(&p[2]);
            inside.then(|| p[0].min(48.0 - p[0]).min(16.0 - p[2]).min(p[2] + 16.0))
        };
        // The first row is the spring's, tucked under the ground at a point.
        let spring = &mesh.vertices[..3];
        assert!(spring.iter().all(|v| (v.position[1] - (water_at(-20.0) - SPRING_SINK)).abs() < 1e-4), "{spring:?}");
        assert!((spring[0].position[2] - spring[2].position[2]).abs() < 0.3, "{spring:?}");
        for v in &mesh.vertices[3..ribbon] {
            match depth(v.position) {
                Some(d) => {
                    let under = level - v.position[1];
                    assert!(under >= UNDER_SHEET - 1e-4 && under <= UNDER_SHEET_DEEP + 1e-4, "{v:?}");
                    if d < SINK_START {
                        assert!(under < UNDER_SHEET + 1e-4, "{v:?}");
                    }
                    assert_eq!(v.still, 1.0);
                }
                None => assert!((v.position[1] - water_at(v.position[0])).abs() < 1e-4, "{v:?}"),
            }
        }
        assert_eq!(mesh.vertices[ribbon - 1].still, 0.0);
        let triangles: Vec<&[u32]> = mesh.indices.chunks(3).filter(|t| t.iter().all(|&i| (i as usize) < ribbon)).collect();
        assert!(triangles.len() < 40 * 4, "nothing left out under the lake");
        for triangle in triangles {
            assert!(triangle.iter().any(|&i| depth(mesh.vertices[i as usize].position).is_none_or(|d| d < HIDDEN_DEPTH)));
        }
    }

    /// Foam made at a rapid drifts on downstream, thinning as it goes, and
    /// never runs upstream. A ribbon carries its own frame; a lake none.
    #[test]
    fn whitewater_foam_drifts_downstream_and_fades() {
        let river_node = |k: i32| RiverNode {
            position: [5.0 * k as f32, 0.0],
            water: 10.0 - 0.01 * k as f32,
            half_width: 2.0,
            depth: 0.5,
            speed: 1.0,
            along: 5.0 * k as f32,
            turbulence: if k == 5 { 0.8 } else { 0.0 },
            ..Default::default()
        };
        let nodes: Vec<RiverNode> = (0..=20).map(river_node).collect();
        let river = River { nodes, end: RiverEnd::Edge, surface_end: 20 };
        let cells: Vec<[i32; 2]> = vec![[100, 100]];
        let lake = Lake { level: 3.0, cells, shore: Vec::new(), edge: Vec::new(), current: Vec::new() };
        let mesh = build(&[river], &[lake]);
        // Half widths over 1.6 m give five columns.
        let row = |r: usize| &mesh.vertices[r * 5];
        assert_eq!(row(4).foam, 0.0);
        assert_eq!(row(5).foam, 0.8);
        for r in 6..=20 {
            assert!(row(r).foam < row(r - 1).foam && row(r).foam > 0.0, "{:?}", row(r));
        }
        assert!((row(6).foam - 0.8 * (-5.0f32 / 15.0).exp()).abs() < 1e-4);
        assert_eq!(row(7).along, 35.0);
        assert_eq!(row(7).half_width, 2.0);
        let sheet = mesh.vertices.last().unwrap();
        assert_eq!((sheet.along, sheet.half_width, sheet.foam), (0.0, 0.0, 0.0));
    }

    /// A river reaching the sea ends one row past the coast, tucked just
    /// under the sea's surface, and its water turns to the sea's over its
    /// last reach: none of it far upstream, all of it at the coast.
    #[test]
    fn a_river_ends_at_the_coast_in_the_seas_colour() {
        // Water falling a centimetre a metre, down to the sea's level at
        // x = 150 m, then on at the sea's level out to x = 200 m.
        let water_at = |x: f32| (SEA_LEVEL + 0.01 * (150.0 - x)).max(SEA_LEVEL);
        let river_node = |k: i32| RiverNode {
            position: [5.0 * k as f32, 0.0],
            water: water_at(5.0 * k as f32),
            half_width: 2.0,
            depth: 0.5,
            speed: 0.8,
            along: 5.0 * k as f32,
            ..Default::default()
        };
        let nodes: Vec<RiverNode> = (0..=40).map(river_node).collect();
        let river = River { nodes, end: RiverEnd::Sea, surface_end: 38 };
        let mesh = build(&[river], &[]);
        // Five columns a row; the coast is the first node at the sea's level.
        let coast = 30;
        assert_eq!(mesh.vertices.len(), (coast + 2) * 5, "one row past the coast");
        let row = |r: usize| &mesh.vertices[r * 5 + 2];
        assert!((row(coast + 1).position[1] - (SEA_LEVEL - UNDER_SEA)).abs() < 1e-5);
        assert!((row(coast).position[1] - water_at(150.0)).abs() < 1e-5);
        assert_eq!(row(coast).sea, 1.0);
        assert_eq!(row(coast + 1).sea, 1.0);
        assert_eq!(row(10).sea, 0.0, "far upstream the water is the river's own");
        for r in 13..=coast {
            assert!(row(r).sea >= row(r - 1).sea, "{:?} {:?}", row(r - 1), row(r));
        }
        assert!(row(24).sea > 0.2 && row(24).sea < 0.9, "{:?}", row(24));
    }
}
