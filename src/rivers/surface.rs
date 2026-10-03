//! The water surfaces: a ribbon across each channel at the water level, and
//! a falling sheet over every waterfall.
//!
//! A ribbon's cross-sections sit at the river's nodes and reach a little
//! past each bank, so the waterline is wherever the carved bank rises through
//! the water and never a mesh edge. Every vertex carries the flow there (the
//! current is fastest over the thalweg and stalls at the banks), how far down
//! the river and across the channel it lies, the channel's depth and its
//! whitewater, which is what the water shader animates and foams.
//!
//! At a fall the water leaves the lip at the speed it arrived with and drops
//! under gravity, so the sheet follows the jet's parabola out from the lip
//! and lands in the plunge pool a little downstream. The pool's own surface
//! starts beneath it.
//!
//! The mesh is cut into 256 m chunks with their bounds, so the renderer only
//! draws what the camera can see.

use super::network::{GRAVITY, River};

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
    /// Metres down the river from its head.
    pub along: f32,
    /// Depth of the thalweg below the surface.
    pub depth: f32,
    /// Whitewater, 0..1.
    pub turbulence: f32,
    pub half_width: f32,
    /// On a falling sheet, how far down it the vertex is (0 at the lip, 1 at
    /// the pool); -1 on a flat surface.
    pub fall: f32,
    /// On a falling sheet, the height it falls; on a surface, the height of
    /// the last fall upstream within reach of its plunge, else 0.
    pub drop: f32,
}

const _: () = assert!(std::mem::size_of::<SurfaceVertex>() == 48);

/// Side of a culling chunk, metres.
pub const CHUNK: f32 = 256.0;

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

pub fn build(rivers: &[River]) -> SurfaceMesh {
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
        let end = river.surface_end.min(nodes.len().saturating_sub(1));
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
        // Direction of the reach through each node; a fall's two nodes take
        // the direction of the channel around them.
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
        // The plunge of the last fall reaches this far down the pool.
        let mut last_fall = (f32::NEG_INFINITY, 0.0f32);
        let mut previous_row: Option<u32> = None;
        for i in 0..=end {
            let node = &nodes[i];
            if i > 0 && nodes[i - 1].fall > 0.0 {
                last_fall = (nodes[i - 1].along, nodes[i - 1].fall);
            }
            let tangent = direction(i);
            let normal = [-tangent[1], tangent[0]];
            let half_width = node.half_width.max(0.25);
            // A plunge pool boils under its fall and settles downstream.
            let plunge_reach = 3.0 * half_width + 2.0 * last_fall.1 + 2.0;
            let since = node.along - last_fall.0;
            let drop = if since < plunge_reach { last_fall.1 * (1.0 - since / plunge_reach) } else { 0.0 };
            let row = vertices.len() as u32;
            for &across in offsets {
                let speed = node.speed * lateral(across);
                vertices.push(SurfaceVertex {
                    position: [
                        node.position[0] + normal[0] * across * half_width,
                        node.water,
                        node.position[1] + normal[1] * across * half_width,
                    ],
                    velocity: [tangent[0] * speed, tangent[1] * speed],
                    across,
                    along: node.along,
                    depth: node.depth,
                    turbulence: node.turbulence,
                    half_width,
                    fall: -1.0,
                    drop,
                });
            }
            // The ribbon steps down a fall through the sheet, not a ramp: no
            // quads join a lip to its foot.
            let crosses_fall = i > 0 && nodes[i - 1].fall > 0.0;
            if let Some(previous) = previous_row
                && !crosses_fall
            {
                add_quads(&mut chunk_triangles, &vertices, previous, row, columns);
            }
            previous_row = Some(row);

            if node.fall > 0.0 && i < end {
                // The sheet: out from the lip along the jet's parabola.
                let height = node.fall;
                let lip_speed = node.speed.max((GRAVITY * node.depth.max(0.1) * 0.5).sqrt()).max(0.6);
                let duration = (2.0 * height / GRAVITY).sqrt();
                let rows = ((height / 0.35).ceil() as usize).clamp(3, 12);
                let sheet_row = vertices.len() as u32;
                let mut sheet_previous: Option<u32> = None;
                for r in 0..=rows {
                    let t = r as f32 / rows as f32 * duration;
                    let out = lip_speed * t;
                    let y = node.water - 0.5 * GRAVITY * t * t;
                    let fall = r as f32 / rows as f32;
                    // A sheet gathers a little as it falls.
                    let gather = 0.95 - 0.12 * fall;
                    let base = vertices.len() as u32;
                    for &across in offsets {
                        let across = across.clamp(-1.0, 1.0) * gather;
                        vertices.push(SurfaceVertex {
                            position: [
                                node.position[0] + tangent[0] * out + normal[0] * across * half_width,
                                y,
                                node.position[1] + tangent[1] * out + normal[1] * across * half_width,
                            ],
                            velocity: [tangent[0] * lip_speed, tangent[1] * lip_speed],
                            across,
                            along: node.along + out,
                            depth: node.depth,
                            turbulence: 1.0,
                            half_width,
                            fall,
                            drop: height,
                        });
                    }
                    if let Some(previous) = sheet_previous {
                        add_quads(&mut chunk_triangles, &vertices, previous, base, columns);
                    }
                    sheet_previous = Some(base);
                }
                let _ = sheet_row;
            }
        }
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
