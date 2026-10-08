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

/// One water-surface vertex: 64 bytes, mirrored by the river vertex inputs
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
    /// Metres downstream from the river's head. With `side` it is the
    /// ribbon's own frame, in which the shader lays its ripples and standing
    /// waves so they follow every bend. 0 on a lake.
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
    /// Metres across the channel in the ribbon's own frame: `across` times
    /// `half_width`. A tributary's frame (this and `along`) is shifted to
    /// line up with its parent's where it reaches it. 0 on a lake.
    pub side: f32,
    /// Where a tributary's water becomes its parent's: the parent's frame
    /// (`along`, `side`) here and how far the ripples and foam are laid in
    /// it rather than the tributary's own, 0 to 1. Two frames at an angle
    /// cannot be blended into one without squeezing the ripples between
    /// them, so the shader draws them in both and cross-fades. Elsewhere the
    /// ribbon's own frame and 0.
    pub joined: [f32; 3],
}

const _: () = assert!(std::mem::size_of::<SurfaceVertex>() == 64);

/// How far the ripples drawn change between two points of the water: the
/// change of each frame they are laid in, by its share of the cross-fade
/// (`SurfaceVertex::joined`), and of the cross-fade itself. A point with no
/// cross-fade shows the same ripples whatever its share, so it takes the
/// other's: a tributary that has all become its parent's water meets the
/// parent's own frame.
pub fn pattern_frame_change(a: &SurfaceVertex, b: &SurfaceVertex) -> f32 {
    let apart = |p: [f32; 2], q: [f32; 2]| (p[0] - q[0]).hypot(p[1] - q[1]);
    let own = |v: &SurfaceVertex| [v.along, v.side];
    let into = |v: &SurfaceVertex| if v.joined[2] > 0.0 { [v.joined[0], v.joined[1]] } else { own(v) };
    let (mut wa, mut wb) = (a.joined[2].clamp(0.0, 1.0), b.joined[2].clamp(0.0, 1.0));
    if wa <= 0.0 {
        wa = wb;
    } else if wb <= 0.0 {
        wb = wa;
    }
    let share = 0.5 * (wa + wb);
    let fade = (wa - wb).abs() * 0.5 * (apart(own(a), into(a)) + apart(own(b), into(b)));
    (1.0 - share) * apart(own(a), own(b)) + share * apart(into(a), into(b)) + fade
}

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
/// Extra hidden coverage for rounded confluences and tight bends. Terrain
/// still determines the waterline; this prevents the ribbon ending at it.
const JUNCTION_RIBBON_MARGIN: f32 = 0.8;
/// How far a tributary's ribbon lies under its parent's where the parent's
/// reaches over it: just under, so the parent's water is the one drawn and
/// the tributary's never shows through it as a second surface.
const UNDER_PARENT: f32 = 0.03;
/// How far a tributary's ribbon runs over its parent's across its mouth,
/// where the parent's ribbon reaches past its waterline into the open water.
const OVER_PARENT: f32 = 0.02;
/// Across its parent's channel, in the parent's half widths (its ribbon's
/// `across`), where a tributary's water starts to become the parent's, where
/// it has all become it, and where it has dipped under it.
const MIX_OUTER: f32 = 1.1;
const MIX_INNER: f32 = 0.45;
const DIP_INNER: f32 = 0.3;
/// Metres off its parent's ribbon over which a tributary's surface comes to
/// the level of the parent's edge.
const MERGE_APPROACH: f32 = 2.0;
/// Metres before its ribbon's end over which a tributary's water becomes
/// all its parent's and dips under it.
const END_MIX: f32 = 1.5;
/// Metres in from its ribbon's side edges, out past its waterline, over
/// which a tributary's water becomes its parent's where it reaches over it.
const SIDE_MIX: f32 = 1.0;
/// Most metres between a tributary's rows and columns where it runs into its
/// parent.
const MERGE_STEP: f32 = 0.5;
/// How far a tributary's water may stand under its parent's where it reaches
/// it and still run into it.
const JOIN_STEP: f32 = 0.15;

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

/// One row of a ribbon's vertices across its channel.
#[derive(Clone, Copy, Debug)]
struct Row {
    first: u32,
    columns: u32,
    /// Hidden for good under a lake's sheet.
    hidden: bool,
}

/// A river's ribbon, row by row down it.
#[derive(Clone, Debug, Default)]
struct Ribbon {
    vertices: Vec<SurfaceVertex>,
    rows: Vec<Row>,
}

impl Ribbon {
    /// Its triangles, as indices into its vertices, between every pair of
    /// rows but those hidden under a lake.
    fn triangles(&self) -> Vec<u32> {
        let mut triangles = Vec::new();
        for pair in self.rows.windows(2) {
            if !(pair[0].hidden && pair[1].hidden) {
                join_rows(pair[0], pair[1], &mut triangles);
            }
        }
        triangles
    }
}

/// The triangles between two rows: quads between rows of as many columns,
/// and fans where one row splits every gap of the other into as many.
fn join_rows(a: Row, b: Row, out: &mut Vec<u32>) {
    if a.columns == b.columns {
        for c in 0..a.columns - 1 {
            let (i0, i1, j0, j1) = (a.first + c, a.first + c + 1, b.first + c, b.first + c + 1);
            out.extend_from_slice(&[i0, j0, j1, i0, j1, i1]);
        }
    } else if a.columns < b.columns {
        let split = (b.columns - 1) / (a.columns - 1);
        let half = split / 2;
        for c in 0..a.columns - 1 {
            let (a0, a1) = (a.first + c, a.first + c + 1);
            let d = |s: u32| b.first + c * split + s;
            for s in 0..half {
                out.extend_from_slice(&[a0, d(s), d(s + 1)]);
            }
            out.extend_from_slice(&[a0, d(half), a1]);
            for s in half..split {
                out.extend_from_slice(&[a1, d(s), d(s + 1)]);
            }
        }
    } else {
        let split = (a.columns - 1) / (b.columns - 1);
        let half = split / 2;
        for c in 0..b.columns - 1 {
            let (b0, b1) = (b.first + c, b.first + c + 1);
            let d = |s: u32| a.first + c * split + s;
            for s in 0..half {
                out.extend_from_slice(&[d(s), b0, d(s + 1)]);
            }
            out.extend_from_slice(&[d(half), b0, b1]);
            for s in half..split {
                out.extend_from_slice(&[d(s), b1, d(s + 1)]);
            }
        }
    }
}

/// Vertices weighed together, every attribute alike.
fn weigh(parts: &[(SurfaceVertex, f32)]) -> SurfaceVertex {
    let mut sum = [0.0f32; 16];
    for &(vertex, weight) in parts {
        let fields: [f32; 16] = bytemuck::cast(vertex);
        for (total, field) in sum.iter_mut().zip(fields) {
            *total += field * weight;
        }
    }
    bytemuck::cast(sum)
}

fn mix_vertex(a: SurfaceVertex, b: SurfaceVertex, t: f32) -> SurfaceVertex {
    weigh(&[(a, 1.0 - t), (b, t)])
}

/// Side of the cells a drawn ribbon's triangles are found by.
const DRAWN_CELL: f32 = 8.0;

/// A river's ribbon as drawn, to find its surface over a point.
struct Drawn<'a> {
    vertices: &'a [SurfaceVertex],
    triangles: Vec<[u32; 3]>,
    cells: HashMap<[i32; 2], Vec<u32>>,
}

impl<'a> Drawn<'a> {
    fn new(ribbon: &'a Ribbon) -> Self {
        let triangles: Vec<[u32; 3]> = ribbon.triangles().chunks_exact(3).map(|t| [t[0], t[1], t[2]]).collect();
        let mut cells: HashMap<[i32; 2], Vec<u32>> = HashMap::new();
        for (k, triangle) in triangles.iter().enumerate() {
            let mut low = [f32::INFINITY; 2];
            let mut high = [f32::NEG_INFINITY; 2];
            for &i in triangle {
                let p = ribbon.vertices[i as usize].position;
                for (axis, value) in [p[0], p[2]].into_iter().enumerate() {
                    low[axis] = low[axis].min(value);
                    high[axis] = high[axis].max(value);
                }
            }
            let (x0, z0) = ((low[0] / DRAWN_CELL).floor() as i32, (low[1] / DRAWN_CELL).floor() as i32);
            let (x1, z1) = ((high[0] / DRAWN_CELL).floor() as i32, (high[1] / DRAWN_CELL).floor() as i32);
            for z in z0..=z1 {
                for x in x0..=x1 {
                    cells.entry([x, z]).or_default().push(k as u32);
                }
            }
        }
        Drawn { vertices: &ribbon.vertices, triangles, cells }
    }

    /// The drawn surface at `p`, the highest where the ribbon lies over it
    /// more than once; or, off the ribbon, the surface at its nearest point
    /// within `reach`, and how far that is.
    fn at(&self, p: [f32; 2], reach: f32) -> Option<(SurfaceVertex, f32)> {
        let cross = |a: [f32; 2], b: [f32; 2]| a[0] * b[1] - a[1] * b[0];
        let flat = |v: &SurfaceVertex| [v.position[0], v.position[2]];
        let (cx, cz) = ((p[0] / DRAWN_CELL).floor() as i32, (p[1] / DRAWN_CELL).floor() as i32);
        let span = (reach / DRAWN_CELL).ceil() as i32;
        let mut over: Option<SurfaceVertex> = None;
        let mut near: Option<(SurfaceVertex, f32)> = None;
        for z in cz - span..=cz + span {
            for x in cx - span..=cx + span {
                for &k in self.cells.get(&[x, z]).into_iter().flatten() {
                    let [a, b, c] = self.triangles[k as usize].map(|i| self.vertices[i as usize]);
                    let (pa, pb, pc) = (flat(&a), flat(&b), flat(&c));
                    let ab = [pb[0] - pa[0], pb[1] - pa[1]];
                    let ac = [pc[0] - pa[0], pc[1] - pa[1]];
                    let ap = [p[0] - pa[0], p[1] - pa[1]];
                    let det = cross(ab, ac);
                    if det.abs() > 1e-9 {
                        let wb = cross(ap, ac) / det;
                        let wc = cross(ab, ap) / det;
                        let wa = 1.0 - wb - wc;
                        if wa >= -1e-5 && wb >= -1e-5 && wc >= -1e-5 {
                            let here = weigh(&[(a, wa), (b, wb), (c, wc)]);
                            if over.is_none_or(|o| here.position[1] > o.position[1]) {
                                over = Some(here);
                            }
                            continue;
                        }
                    }
                    if over.is_some() {
                        continue;
                    }
                    for (u, w) in [(a, b), (b, c), (c, a)] {
                        let (pu, pw) = (flat(&u), flat(&w));
                        let uw = [pw[0] - pu[0], pw[1] - pu[1]];
                        let up = [p[0] - pu[0], p[1] - pu[1]];
                        let t = ((up[0] * uw[0] + up[1] * uw[1]) / (uw[0] * uw[0] + uw[1] * uw[1]).max(1e-12)).clamp(0.0, 1.0);
                        let gap = (up[0] - uw[0] * t).hypot(up[1] - uw[1] * t);
                        if gap <= reach && near.is_none_or(|n| gap < n.1) {
                            near = Some((mix_vertex(u, w, t), gap));
                        }
                    }
                }
            }
        }
        over.map(|o| (o, 0.0)).or(near)
    }
}

/// A tributary's surface where it runs into its parent's, `own` as it
/// would be alone and `drawn` the parent's drawn surface at it (or at the
/// nearest point of the parent's ribbon, and how far that is), `merge` how
/// far its water has become the parent's (`network::merge_weights`) and
/// `to_end` how far it is from its ribbon's end, metres, `side` how far it
/// lies out under its banks toward its ribbon's side edge, 0 to 1, and
/// `reached` whether it has reached the parent's channel
/// (`network::first_contact`).
///
/// Across the open water of its mouth it runs on just over the parent's
/// ribbon, whose edge (past the parent's waterline, under its banks) would
/// otherwise cross it, and comes to the parent's level as it nears it. Into
/// the parent's channel it takes on the parent's surface as drawn, its
/// flow, foam, ripple frame and all, and only once it is all the parent's
/// does it dip under: wherever one surface shows beside the other, the two
/// are one water. Out past its waterline toward its ribbon's side edges,
/// wherever its ribbon reaches over the parent's, it becomes the parent's
/// too.
///
/// Where it does not become the parent's water, it lies just under the
/// parent's ribbon wherever that reaches over it: from where it reaches the
/// parent's channel on, and before that never over it.
fn merge_into(
    own: SurfaceVertex,
    drawn: Option<(SurfaceVertex, f32)>,
    merge: f32,
    to_end: f32,
    side: f32,
    reached: bool,
) -> SurfaceVertex {
    let Some((parent, gap)) = drawn else {
        return own;
    };
    let end_mix = smoothstep(END_MIX, 0.5 * END_MIX, to_end);
    let end_dip = smoothstep(0.5 * END_MIX, 0.0, to_end);
    let height = own.position[1];
    let (share, dip, near, before) = if gap > 0.0 {
        let near = 1.0 - smoothstep(0.0, MERGE_APPROACH, gap);
        (side.max(end_mix), end_dip, near, height)
    } else {
        let into = parent.across.abs();
        let share = smoothstep(MIX_OUTER, MIX_INNER, into).max(side).max(end_mix);
        let dip = smoothstep(MIX_INNER, DIP_INNER, into).max(end_dip);
        let under = parent.position[1] - UNDER_PARENT;
        (share, dip, 1.0, if reached { under } else { height.min(under) })
    };
    let joined = parent.position[1] + OVER_PARENT - (OVER_PARENT + UNDER_PARENT) * dip;
    let level = before + (joined - before) * merge * near;
    let share = share * merge * near;
    let mut vertex = mix_vertex(own, parent, share);
    vertex.position = [own.position[0], level, own.position[2]];
    // Its ripples keep its own frame, cross-faded into the parent's as
    // drawn (as it reaches on out past the parent's edge, off its ribbon).
    vertex.along = own.along;
    vertex.side = own.side;
    let frame = if parent.joined[2] > 0.5 { [parent.joined[0], parent.joined[1]] } else { [parent.along, parent.side] };
    vertex.joined = if share > 0.0 { [frame[0], frame[1] + gap * frame[1].signum(), share] } else { own.joined };
    vertex
}

/// The rivers in an order that has every river's parent before it.
fn parents_first(rivers: &[River]) -> Vec<usize> {
    let mut order = Vec::with_capacity(rivers.len());
    let mut seen = vec![false; rivers.len()];
    for start in 0..rivers.len() {
        let mut chain = Vec::new();
        let mut at = start;
        while !seen[at] {
            seen[at] = true;
            chain.push(at);
            match rivers[at].end {
                RiverEnd::Confluence(parent, _) if parent < rivers.len() => at = parent,
                _ => break,
            }
        }
        order.extend(chain.into_iter().rev());
    }
    order
}

pub fn build(rivers: &[River], lakes: &[Lake]) -> SurfaceMesh {
    let sheets = Sheets::new(lakes);
    let seas = sea_blends(rivers);
    let mut junctions = vec![Vec::new(); rivers.len()];
    for (index, river) in rivers.iter().enumerate() {
        let RiverEnd::Confluence(parent, _) = river.end else {
            continue;
        };
        let Some(parent_river) = rivers.get(parent) else {
            continue;
        };
        let Some(node) = river.nodes.get(river.surface_end) else {
            continue;
        };
        junctions[index].push(node.position);
        let mut nearest_position = None;
        let mut nearest_distance = f32::INFINITY;
        for parent_node in &parent_river.nodes {
            let dx = parent_node.position[0] - node.position[0];
            let dz = parent_node.position[1] - node.position[1];
            let distance_squared = dx * dx + dz * dz;
            if distance_squared < nearest_distance {
                nearest_position = Some(parent_node.position);
                nearest_distance = distance_squared;
            }
        }
        if let Some(position) = nearest_position {
            junctions[parent].push(position);
        }
    }
    // Every parent's ribbon is built before its tributaries', which run
    // into it as drawn.
    let ribbons: Vec<std::cell::OnceCell<Ribbon>> = rivers.iter().map(|_| Default::default()).collect();
    let mut drawn: HashMap<usize, Drawn> = HashMap::new();
    for index in parents_first(rivers) {
        let river = &rivers[index];
        let parent = match river.end {
            RiverEnd::Confluence(parent, _) => rivers.get(parent).zip(ribbons[parent].get()).map(|(parent_river, ribbon)| {
                (parent_river, &*drawn.entry(parent).or_insert_with(|| Drawn::new(ribbon)))
            }),
            _ => None,
        };
        if let Some(ribbon) = ribbon(river, &junctions[index], &sheets, seas[index], parent) {
            let _ = ribbons[index].set(ribbon);
        }
    }

    let mut vertices: Vec<SurfaceVertex> = Vec::new();
    // Triangles per chunk, gathered before they are laid out chunk by chunk.
    let mut chunk_triangles: std::collections::BTreeMap<(i32, i32), Vec<u32>> = Default::default();
    for ribbon in ribbons.iter().filter_map(|r| r.get()) {
        let base = vertices.len() as u32;
        vertices.extend_from_slice(&ribbon.vertices);
        for pair in ribbon.rows.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            if a.hidden && b.hidden {
                continue;
            }
            let p = ribbon.vertices[a.first as usize].position;
            let q = ribbon.vertices[(b.first + b.columns / 2) as usize].position;
            let key = ((((p[0] + q[0]) * 0.5) / CHUNK).floor() as i32, (((p[2] + q[2]) * 0.5) / CHUNK).floor() as i32);
            let list = chunk_triangles.entry(key).or_default();
            let start = list.len();
            join_rows(a, b, list);
            for index in &mut list[start..] {
                *index += base;
            }
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

/// A row of a ribbon before it is laid: its vertices as the river alone
/// would have them, the water the row stands at, how far its water has
/// become its parent's and how far it lies from the ribbon's end.
#[derive(Clone, Debug)]
struct Draft {
    vertices: Vec<SurfaceVertex>,
    water: f32,
    merge: f32,
    to_end: f32,
    /// Per vertex, how far out it lies under the banks toward the row's
    /// side edge, 0 to 1 (`merge_into`).
    side: Vec<f32>,
    reached: bool,
    past_coast: bool,
}

impl Draft {
    fn mix(&self, other: &Draft, t: f32) -> Draft {
        let lerp = |a: f32, b: f32| a + (b - a) * t;
        Draft {
            vertices: self.vertices.iter().zip(&other.vertices).map(|(&a, &b)| mix_vertex(a, b, t)).collect(),
            water: lerp(self.water, other.water),
            merge: lerp(self.merge, other.merge),
            to_end: lerp(self.to_end, other.to_end),
            side: self.side.iter().zip(&other.side).map(|(&a, &b)| lerp(a, b)).collect(),
            reached: self.reached && other.reached,
            past_coast: self.past_coast && other.past_coast,
        }
    }
}

/// A river's ribbon, run into its parent's drawn ribbon if it has one.
fn ribbon(
    river: &River,
    junctions: &[[f32; 2]],
    sheets: &Sheets,
    (sea_share, sea_along, sea_ramp): (f32, f32, f32),
    parent: Option<(&River, &Drawn)>,
) -> Option<Ribbon> {
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
        return None;
    }
    let widest = nodes[..=end].iter().map(|n| n.half_width).fold(0.0f32, f32::max);
    let offsets: &[f32] = if widest > 1.6 {
        &[-1.12, -0.55, 0.0, 0.55, 1.12]
    } else {
        &[-1.15, 0.0, 1.15]
    };
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
    let ribbon_half_width = |i: usize| if spring && i == 0 { 0.1 } else { nodes[i].half_width.max(0.25) };
    // Each row's columns, metres across the channel from the centreline.
    let row_offsets = |i: usize| -> Vec<f32> {
        let node = &nodes[i];
        let half_width = ribbon_half_width(i);
        let mut junction_influence = 0.0f32;
        for position in junctions {
            let distance = (position[0] - node.position[0]).hypot(position[1] - node.position[1]);
            let influence = 1.0 - smoothstep(half_width + 4.0, half_width + 12.0, distance);
            junction_influence = junction_influence.max(influence);
        }
        if i > 0 && i < end {
            let before_direction = [
                node.position[0] - nodes[i - 1].position[0],
                node.position[1] - nodes[i - 1].position[1],
            ];
            let after_direction = [
                nodes[i + 1].position[0] - node.position[0],
                nodes[i + 1].position[1] - node.position[1],
            ];
            let before_length = before_direction[0].hypot(before_direction[1]);
            let after_length = after_direction[0].hypot(after_direction[1]);
            let length_product = before_length * after_length;
            let cross = before_direction[0] * after_direction[1] - before_direction[1] * after_direction[0];
            let turn = cross / length_product.max(1e-5);
            let bend_influence = smoothstep(0.05, 0.25, turn * turn);
            junction_influence = junction_influence.max(bend_influence);
        }
        let margin = if spring && i == 0 { 0.0 } else { JUNCTION_RIBBON_MARGIN * junction_influence };
        offsets
            .iter()
            .map(|&across| {
                let edge_margin = if across.abs() > 1.0 { margin * across.signum() } else { 0.0 };
                across * half_width + edge_margin
            })
            .collect()
    };
    let place = |i: usize, offset: f32| -> [f32; 2] {
        let tangent = direction(i);
        [nodes[i].position[0] - tangent[1] * offset, nodes[i].position[1] + tangent[0] * offset]
    };
    let base_offsets: Vec<Vec<f32>> = (0..=end).map(row_offsets).collect();
    // Where it reaches its parent, and how far its water has become the
    // parent's by each node.
    let (contact, mut merge) = match parent {
        Some((parent_river, _)) => super::network::merge_weights(&nodes[..=end], parent_river),
        None => (end, vec![0.0; end + 1]),
    };
    let reached = parent.and_then(|(_, drawn)| drawn.at(nodes[contact].position, 4.0 * widest + 16.0));
    // A tributary's ripples run on into its parent's: its frame is the
    // parent's where it reaches it (beyond the parent's ribbon, as if it ran
    // on out from its edge).
    let along_offset = reached.map_or(0.0, |(there, _)| there.along - nodes[contact].along);
    let side_offset = reached.map_or(0.0, |(there, gap)| there.side + gap * there.side.signum());
    // Only a tributary that comes in at its parent's level becomes its
    // water; one standing under it there (out of a lake at the foot of the
    // parent's fall into it) keeps its own.
    if reached.is_none_or(|(there, _)| there.position[1] - nodes[contact].water > JOIN_STEP) {
        merge.fill(0.0);
    }
    // How far along the river each row lies from still water: a lake's
    // sheet, or the sea past the river's mouth. Neither is lifted in the
    // distance, so the ribbon must not be lifted where it meets them.
    let node_touches_still_water = |i: usize| {
        let water = nodes[i].water;
        water <= SEA_LEVEL + 0.03
            || (river.end == RiverEnd::Sea && i >= river.surface_end)
            || base_offsets[i].iter().any(|&offset| {
                let p = place(i, offset);
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
    // A row as the river alone would have it, its columns `offsets`.
    let draft = |i: usize, offsets: &[f32]| -> Draft {
        let node = &nodes[i];
        let tangent = direction(i);
        // Past where its own water ends, the river runs on sunk under the
        // still water it meets (the sea, its parent); and water standing
        // at the sea's level is the sea's, drawn by the sea itself.
        let mut water = if i > river.surface_end { node.water - sink } else { node.water };
        if spring && i == 0 {
            water -= SPRING_SINK;
        }
        let past_coast = coast.is_some_and(|coast| i > coast);
        if past_coast {
            water = water.min(SEA_LEVEL - UNDER_SEA);
        }
        let sea = sea_share * (1.0 - smoothstep(0.0, sea_ramp, sea_along - node.along));
        let still = 1.0 - smoothstep(0.0, LIFT_FADE, from_still[i]);
        let vertices = offsets
            .iter()
            .map(|&offset| {
                let p = place(i, offset);
                let across = offset / ribbon_half_width(i);
                let speed = node.speed * lateral(across);
                SurfaceVertex {
                    position: [p[0], water, p[1]],
                    velocity: [tangent[0] * speed, tangent[1] * speed],
                    across,
                    turbulence: node.turbulence,
                    still,
                    along: node.along + along_offset,
                    half_width: node.half_width.max(0.25),
                    foam: foam[i],
                    sea,
                    side: offset + side_offset,
                    joined: [node.along + along_offset, offset + side_offset, 0.0],
                }
            })
            .collect();
        // Out past the waterline over the last metre to the side edges.
        let half_width = ribbon_half_width(i);
        let side = offsets
            .iter()
            .map(|&offset| {
                let edge = if offset < 0.0 { -offsets[0] } else { offsets[offsets.len() - 1] };
                let reach = (edge - half_width).clamp(1e-3, SIDE_MIX);
                1.0 - smoothstep(0.0, reach, edge - offset.abs())
            })
            .collect();
        Draft { vertices, water, merge: merge[i], to_end: nodes[end].along - node.along, side, reached: i >= contact, past_coast }
    };
    let mut ribbon = Ribbon::default();
    let mut lay = |draft: &Draft| {
        let first = ribbon.vertices.len() as u32;
        let mut hidden = true;
        for (k, &own) in draft.vertices.iter().enumerate() {
            let p = [own.position[0], own.position[2]];
            let mut vertex = match parent {
                Some((_, drawn)) => {
                    merge_into(own, drawn.at(p, MERGE_APPROACH), draft.merge, draft.to_end, draft.side[k], draft.reached)
                }
                None => own,
            };
            // Still tucked under the sea's waves past a coast.
            if draft.past_coast {
                vertex.position[1] = vertex.position[1].min(SEA_LEVEL - UNDER_SEA);
            }
            // Under the sheet of a lake whose level it stands at (running
            // into it, out of it or through it), the ribbon runs on just
            // beneath the sheet as drawn, its edge sunk and all, and
            // deeper the further inside. The two never share a plane,
            // the ribbon never shows over the lake's water, and its end
            // is never in sight. Well inside, where the sheet hides it for
            // good, it stops.
            let water = draft.water;
            if let Some((lake, depth)) = sheets.under(p)
                && water <= lake + LEVEL_TIE
                && water >= lake - SHEET_BAND
            {
                let sheet = sheets.height(p).unwrap_or(lake).min(lake);
                vertex.position[1] = vertex.position[1].min(sheet) - under_sheet(depth);
                hidden &= depth >= HIDDEN_DEPTH;
            } else {
                hidden = false;
            }
            ribbon.vertices.push(vertex);
        }
        ribbon.rows.push(Row { first, columns: draft.vertices.len() as u32, hidden });
    };
    // Where it runs into its parent, its rows and columns close up, so its
    // surface follows the parent's as drawn closely enough that the two
    // meet without a seam.
    let merging = parent.and_then(|_| merge.iter().position(|&m| m > 0.0)).unwrap_or(end + 1);
    let widest_gap = base_offsets[merging.min(end)..]
        .iter()
        .flat_map(|row| row.windows(2).map(|pair| pair[1] - pair[0]))
        .fold(0.0f32, f32::max);
    let split = ((widest_gap / MERGE_STEP).ceil() as usize).clamp(1, 8);
    // Its own surface across the closer columns is as its rows would draw
    // it alone.
    let dense = |row: Draft| -> Draft {
        let mut vertices: Vec<SurfaceVertex> = row
            .vertices
            .windows(2)
            .flat_map(|pair| (0..split).map(move |s| mix_vertex(pair[0], pair[1], s as f32 / split as f32)))
            .collect();
        vertices.extend(row.vertices.last());
        let mut side: Vec<f32> = row
            .side
            .windows(2)
            .flat_map(|pair| (0..split).map(move |s| pair[0] + (pair[1] - pair[0]) * s as f32 / split as f32))
            .collect();
        side.extend(row.side.last());
        Draft { vertices, side, ..row }
    };
    let mut previous: Option<Draft> = None;
    for (i, offsets) in base_offsets.iter().enumerate() {
        if i < merging {
            lay(&draft(i, offsets));
            continue;
        }
        let here = dense(draft(i, offsets));
        if let Some(before) = previous.as_ref() {
            let reach = before
                .vertices
                .iter()
                .zip(&here.vertices)
                .map(|(a, b)| (a.position[0] - b.position[0]).hypot(a.position[2] - b.position[2]))
                .fold(0.0f32, f32::max);
            let steps = ((reach / MERGE_STEP).ceil() as usize).max(1);
            for s in 1..steps {
                lay(&before.mix(&here, s as f32 / steps as f32));
            }
        }
        lay(&here);
        previous = Some(here);
    }
    Some(ribbon)
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
                side: 0.0,
                joined: [0.0; 3],
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

    #[test]
    fn confluence_ribbons_cover_the_rounded_shore_and_fade_back_to_normal_width() {
        let positions = [-20.0, 0.0, 20.0, 40.0];
        let node_at = |position, along| RiverNode {
            position,
            water: 5.0,
            half_width: 2.0,
            speed: 1.0,
            along,
            ..Default::default()
        };
        let trunk_nodes = positions.iter().map(|&x| node_at([x, 0.0], x + 20.0)).collect();
        let tributary_nodes = positions[..3].iter().map(|&z| node_at([0.0, z], z + 20.0)).collect();
        let trunk = River {
            nodes: trunk_nodes,
            end: RiverEnd::Edge,
            surface_end: 3,
        };
        let tributary = River {
            nodes: tributary_nodes,
            end: RiverEnd::Confluence(0, 1),
            surface_end: 1,
        };
        let rivers = [trunk, tributary];
        let mesh = build(&rivers, &[]);

        let junction_row = &mesh.vertices[5..10];
        assert!(junction_row[0].position[2].abs() > 2.6);
        assert!(junction_row[4].position[2].abs() > 2.6);
        let far_row = &mesh.vertices[15..20];
        assert!((far_row[0].position[2].abs() - 2.24).abs() < 1e-5);
        for vertex in junction_row {
            let actual_across = vertex.position[2] / vertex.half_width;
            assert!((actual_across - vertex.across).abs() < 1e-5);
        }
        // This point lies beyond both original ribbons but inside the
        // rounded confluence. At least one water triangle must cover it.
        let point = [2.5, 2.5];
        let cross = |a: [f32; 2], b: [f32; 2]| a[0] * b[1] - a[1] * b[0];
        let contains_point = |triangle: &[u32]| {
            let a = mesh.vertices[triangle[0] as usize].position;
            let b = mesh.vertices[triangle[1] as usize].position;
            let c = mesh.vertices[triangle[2] as usize].position;
            let sides = [
                cross([b[0] - a[0], b[2] - a[2]], [point[0] - a[0], point[1] - a[2]]),
                cross([c[0] - b[0], c[2] - b[2]], [point[0] - b[0], point[1] - b[2]]),
                cross([a[0] - c[0], a[2] - c[2]], [point[0] - c[0], point[1] - c[2]]),
            ];
            let same_sign = sides.iter().all(|&side| side >= 0.0) || sides.iter().all(|&side| side <= 0.0);
            same_sign && [a, b, c].iter().all(|vertex| vertex[1] >= 5.0 - 1e-5)
        };
        assert!(mesh.indices.chunks_exact(3).any(contains_point));
    }

    /// The surface drawn on top over `p`, the highest there, its attributes
    /// interpolated as the GPU interpolates them.
    fn top(mesh: &SurfaceMesh, p: [f32; 2]) -> Option<SurfaceVertex> {
        let cross = |a: [f32; 2], b: [f32; 2]| a[0] * b[1] - a[1] * b[0];
        let mut top: Option<SurfaceVertex> = None;
        for triangle in mesh.indices.chunks_exact(3) {
            let [a, b, c] = [0, 1, 2].map(|k| mesh.vertices[triangle[k] as usize]);
            let flat = |v: &SurfaceVertex| [v.position[0] - p[0], v.position[2] - p[1]];
            let (pa, pb, pc) = (flat(&a), flat(&b), flat(&c));
            let det = cross([pb[0] - pa[0], pb[1] - pa[1]], [pc[0] - pa[0], pc[1] - pa[1]]);
            if det.abs() < 1e-9 {
                continue;
            }
            let (wa, wb, wc) = (cross(pb, pc) / det, cross(pc, pa) / det, cross(pa, pb) / det);
            if wa < -1e-5 || wb < -1e-5 || wc < -1e-5 {
                continue;
            }
            let here = weigh(&[(a, wa), (b, wb), (c, wc)]);
            if top.is_none_or(|t| here.position[1] > t.position[1]) {
                top = Some(here);
            }
        }
        top
    }

    /// A creek runs in at an angle down the side of a steep, white river.
    /// Wherever their water meets, the surface drawn on top runs on from one
    /// to the other without a step in its height, its current, its
    /// whitewater and foam or its ripple frame: the creek runs on over the
    /// river's ribbon across its mouth, takes on the river's surface as drawn
    /// and only then dips under it.
    #[test]
    fn a_tributary_runs_into_its_parents_surface_without_a_seam() {
        let parent_water = |x: f32| 10.0 - 0.1 * x;
        let trunk_node = |k: i32| {
            let x = -30.0 + 2.5 * k as f32;
            RiverNode {
                position: [x, 0.0],
                water: parent_water(x),
                half_width: 2.0,
                depth: 0.6,
                speed: 2.0,
                turbulence: 0.8,
                along: x + 30.0,
                ..Default::default()
            }
        };
        let trunk = River { nodes: (0..=24).map(trunk_node).collect(), end: RiverEnd::Edge, surface_end: 24 };
        // In at 45 degrees from the trunk's right, ending on its centreline;
        // its water comes down to the trunk's beside it, as the network
        // grades it.
        let d = std::f32::consts::FRAC_1_SQRT_2;
        let tributary_node = |k: i32| {
            let back = 2.0 * (12 - k) as f32;
            let position = [-back * d, -back * d];
            RiverNode {
                position,
                water: parent_water(position[0]) + 0.02 * (back - 4.0).max(0.0),
                half_width: 1.0,
                depth: 0.4,
                speed: 1.0,
                along: 2.0 * k as f32,
                ..Default::default()
            }
        };
        let tributary = River { nodes: (0..=12).map(tributary_node).collect(), end: RiverEnd::Confluence(0, 0), surface_end: 11 };
        let mesh = build(&[trunk, tributary], &[]);

        // In its mouth, past the trunk's waterline, the creek's own water is
        // drawn, over the trunk's ribbon.
        let mouth = top(&mesh, [-2.6, -2.6]).unwrap();
        assert!(mouth.velocity[1] > 0.4 && mouth.turbulence < 0.1, "{mouth:?}");
        // Well inside the trunk's channel, the trunk's.
        let inside = top(&mesh, [0.5, -0.4]).unwrap();
        assert!(inside.turbulence > 0.79 && inside.velocity[1].abs() < 1e-3, "{inside:?}");

        let step = 0.125;
        let (columns, rows) = (112, 96);
        let samples: Vec<Option<SurfaceVertex>> = (0..rows)
            .flat_map(|zi| (0..columns).map(move |xi| [-8.0 + xi as f32 * step, -8.0 + zi as f32 * step]))
            .map(|p| top(&mesh, p))
            .collect();
        let mut worst = [0.0f32; 4];
        for zi in 0..rows {
            for xi in 0..columns {
                for (dx, dz) in [(1, 0), (0, 1)] {
                    if xi + dx >= columns || zi + dz >= rows {
                        continue;
                    }
                    let (Some(a), Some(b)) = (samples[zi * columns + xi], samples[(zi + dz) * columns + xi + dx]) else {
                        continue;
                    };
                    let changes = [
                        (a.position[1] - b.position[1]).abs(),
                        (a.velocity[0] - b.velocity[0]).hypot(a.velocity[1] - b.velocity[1]),
                        (a.turbulence - b.turbulence).abs().max((a.foam - b.foam).abs()),
                        pattern_frame_change(&a, &b),
                    ];
                    for (worst, change) in worst.iter_mut().zip(changes) {
                        *worst = worst.max(change);
                    }
                }
            }
        }
        // An eighth of a metre apart: the trunk's water falls 1.25 cm, its
        // current turns from its core to its banks and the frame moves on as
        // far again, a little more where the two waters blend and the creek's
        // ripples cross-fade into the trunk's over a metre or so. Without the
        // blend, the trunk's edge across the creek's mouth steps the frame by
        // tens of metres.
        let limits = [0.04, 0.5, 0.15, 0.75];
        for (k, name) in ["height", "current", "whitewater", "frame"].iter().enumerate() {
            assert!(worst[k] < limits[k], "{name} steps {} between neighbouring samples", worst[k]);
        }
    }

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
