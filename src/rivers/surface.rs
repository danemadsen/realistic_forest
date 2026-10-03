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
//! never past its outlet, where the ground falls away. A river's ribbon
//! stops where it enters a lake and starts again at its outlet.
//!
//! The mesh is cut into 256 m chunks with their bounds, so the renderer only
//! draws what the camera can see.

use super::network::{FLOW_CELL, Lake, River, RiverEnd};

/// One water-surface vertex: 32 bytes, mirrored by the river vertex inputs
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
    /// 1 on a lake's still water, 0 on a river's.
    pub still: f32,
}

const _: () = assert!(std::mem::size_of::<SurfaceVertex>() == 32);

/// Side of a culling chunk, metres.
pub const CHUNK: f32 = 256.0;
/// Side of a lake surface's cells: the flow grid's.
const LAKE_CELL: f32 = FLOW_CELL as f32;

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
    for river in rivers {
        let nodes = &river.nodes;
        // A river's surface runs on past where its own water ends, sunk
        // under the still water it meets, so it never ends in an edge: one
        // node into its parent's channel, and out past the shore under the
        // sea.
        let last = nodes.len().saturating_sub(1);
        let (end, sink) = match river.end {
            RiverEnd::Sea => (last, 0.3),
            RiverEnd::Confluence(..) => ((river.surface_end + 1).min(last), 0.12),
            RiverEnd::Edge => (river.surface_end.min(last), 0.0),
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
        let mut previous_row: Option<u32> = None;
        for i in 0..=end {
            let node = &nodes[i];
            let tangent = direction(i);
            let normal = [-tangent[1], tangent[0]];
            let half_width = node.half_width.max(0.25);
            let row = vertices.len() as u32;
            // Where the river runs into a lake its surface sinks under the
            // lake's, which then covers it.
            let level = if node.lake {
                node.water - 0.12
            } else if i > river.surface_end {
                node.water - sink
            } else {
                node.water
            };
            for &across in offsets {
                let speed = node.speed * lateral(across);
                vertices.push(SurfaceVertex {
                    position: [
                        node.position[0] + normal[0] * across * half_width,
                        level,
                        node.position[1] + normal[1] * across * half_width,
                    ],
                    velocity: [tangent[0] * speed, tangent[1] * speed],
                    across,
                    turbulence: node.turbulence,
                    still: 0.0,
                });
            }
            let under_lake = i > 0 && nodes[i - 1].lake && node.lake;
            if let Some(previous) = previous_row
                && !under_lake
            {
                add_quads(&mut chunk_triangles, &vertices, previous, row, columns);
            }
            previous_row = Some(row);
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
        chunks.push(SurfaceChunk {
            minimum,
            maximum,
            first_index: indices.len() as u32,
            index_count: triangles.len() as u32,
        });
        indices.extend_from_slice(&triangles);
    }
    SurfaceMesh {
        vertices,
        indices,
        chunks,
    }
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
    let sunk: std::collections::HashMap<[i32; 2], f32> = lake.edge.iter().copied().collect();
    let mut corner_index = std::collections::HashMap::new();
    let mut corner = |vertices: &mut Vec<SurfaceVertex>, x: i32, z: i32| -> u32 {
        *corner_index.entry((x, z)).or_insert_with(|| {
            let height = sunk.get(&[x, z]).map_or(lake.level, |&h| h.min(lake.level));
            vertices.push(SurfaceVertex {
                position: [x as f32 * LAKE_CELL, height, z as f32 * LAKE_CELL],
                velocity: [0.0, 0.0],
                across: 0.0,
                turbulence: 0.0,
                still: 1.0,
            });
            vertices.len() as u32 - 1
        })
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
