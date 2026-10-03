//! Stones in and along the channels.
//!
//! A channel's bed reflects the power of the water over it. Every stream
//! that runs over gravel is paved with cobbles, which pile along its
//! waterlines, half in the water and half out; boulders lie scattered among
//! them, sparse in a lowland river and crowding a steep creek's bed, packed
//! into the step lips that hold its pools and lodged in its banks; a
//! waterfall pours over a ledge of blocks with more tumbled into its plunge
//! pool. Only a slow, sandy reach has few, and a lake's silt bed none. Rocks
//! are placed by walking each river and drawing from densities that follow
//! its slope and speed, sized by the same stream power and kept from
//! overlapping; big ones break the surface, small ones lie under it and
//! make it boil.

use super::carve::{envelope_at, RiverSegment, SegmentGrid};
use super::network::{River, RiverRock};
use crate::noise::{NoiseField, base_height};
use crate::vegetation::ecology::{cell_key, random};

/// The rock models in assets/models, `rock-1.glb` to `rock-40.glb`: half
/// extents across X and Z and height, metres as authored (pivot at the base).
/// The river places them by these, so the water knows how far each boulder
/// stands out of it; a test holds the table to the files. (One extent
/// happens to read like 1/pi to the constant lint.)
#[allow(clippy::approx_constant)]
pub const ROCK_MODELS: [[f32; 3]; 40] = [
    [0.418, 0.244, 0.340],
    [0.323, 0.320, 0.599],
    [0.315, 0.369, 0.433],
    [0.429, 0.348, 0.541],
    [0.411, 0.284, 0.587],
    [0.299, 0.276, 0.649],
    [0.500, 0.309, 0.571],
    [0.395, 0.263, 0.687],
    [0.448, 0.492, 0.584],
    [0.320, 0.360, 0.555],
    [0.313, 0.333, 0.226],
    [0.347, 0.381, 0.492],
    [0.444, 0.274, 0.635],
    [0.362, 0.337, 0.461],
    [0.300, 0.318, 1.006],
    [0.398, 0.296, 0.688],
    [0.278, 0.269, 0.672],
    [0.431, 0.276, 0.596],
    [0.346, 0.389, 0.537],
    [0.281, 0.277, 0.323],
    [0.301, 0.372, 0.572],
    [0.370, 0.376, 0.601],
    [0.510, 0.304, 0.619],
    [0.497, 0.275, 0.909],
    [0.379, 0.377, 0.720],
    [0.487, 0.276, 0.590],
    [0.409, 0.321, 0.517],
    [0.380, 0.405, 0.530],
    [0.464, 0.331, 0.540],
    [0.475, 0.341, 0.518],
    [0.362, 0.324, 0.711],
    [0.471, 0.223, 0.437],
    [0.337, 0.269, 0.745],
    [0.479, 0.336, 0.572],
    [0.349, 0.272, 0.658],
    [0.316, 0.340, 0.552],
    [0.340, 0.332, 0.615],
    [0.321, 0.307, 0.583],
    [0.431, 0.349, 0.633],
    [0.393, 0.327, 0.675],
];

/// A rock model's mean horizontal radius.
pub fn model_radius(model: usize) -> f32 {
    0.5 * (ROCK_MODELS[model][0] + ROCK_MODELS[model][1])
}

/// Pick a model for a boulder: river-worn blocks are seldom taller than
/// wide, and a ledge wants flat-topped slabs.
fn choose_model(roll: f32, flat: bool) -> usize {
    let limit = if flat { 1.35 } else { 1.95 };
    let fitting: Vec<usize> = (0..ROCK_MODELS.len())
        .filter(|&m| ROCK_MODELS[m][2] / model_radius(m) <= limit)
        .collect();
    fitting[((roll * fitting.len() as f32) as usize).min(fitting.len() - 1)]
}


fn smoothstep(edge0: f32, edge1: f32, value: f32) -> f32 {
    let t = ((value - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Stones smaller than this (radius, metres) pave the bed but leave the
/// current alone: the water shader only sees the larger ones.
pub const OBSTACLE_RADIUS: f32 = 0.2;

/// Rocks already placed, bucketed by 4 m cell for overlap tests.
struct Placed {
    buckets: std::collections::HashMap<(i64, i64), Vec<usize>>,
}

impl Placed {
    fn key(p: [f32; 2]) -> (i64, i64) {
        ((p[0] / 4.0).floor() as i64, (p[1] / 4.0).floor() as i64)
    }

    fn clear(&self, rocks: &[RiverRock], p: [f32; 2], radius: f32) -> bool {
        let (cx, cz) = Self::key(p);
        let span = ((radius + 3.0) / 4.0).ceil() as i64;
        for z in cz - span..=cz + span {
            for x in cx - span..=cx + span {
                let Some(list) = self.buckets.get(&(x, z)) else {
                    continue;
                };
                for &index in list {
                    let other = &rocks[index];
                    let d = (other.position[0] - p[0]).hypot(other.position[1] - p[1]);
                    if d < 0.85 * (radius + other.radius) {
                        return false;
                    }
                }
            }
        }
        true
    }

    fn insert(&mut self, index: usize, p: [f32; 2]) {
        self.buckets.entry(Self::key(p)).or_default().push(index);
    }
}

pub fn place_rocks(
    noise: &NoiseField,
    rivers: &[River],
    segments: &[RiverSegment],
    grid: &SegmentGrid,
) -> Vec<RiverRock> {
    let mut rocks: Vec<RiverRock> = Vec::new();
    let mut placed = Placed {
        buckets: Default::default(),
    };
    let mut try_place = |rocks: &mut Vec<RiverRock>, mut rock: RiverRock, flat: bool| {
        // Never on dry ground far from the water, never inside another rock.
        let envelope = envelope_at(segments, grid, rock.position);
        if envelope.is_none() || envelope.bank_distance > rock.radius * 0.8 {
            return;
        }
        if !placed.clear(rocks, rock.position, rock.radius) {
            return;
        }
        rock.bed = envelope.clamp(base_height(noise, rock.position[0], rock.position[1]));
        let model = choose_model(rock.seed, flat);
        rock.model = model as u32;
        rock.scale = rock.radius / model_radius(model);
        rock.height = ROCK_MODELS[model][2] * rock.scale;
        placed.insert(rocks.len(), rock.position);
        rocks.push(rock);
    };
    for (river_index, river) in rivers.iter().enumerate() {
        let nodes = &river.nodes;
        if nodes.len() < 2 {
            continue;
        }
        let mut draw = 0u64;
        for i in 0..nodes.len() - 1 {
            let a = &nodes[i];
            let b = &nodes[i + 1];
            let d = [b.position[0] - a.position[0], b.position[1] - a.position[1]];
            let length = d[0].hypot(d[1]);
            // A lake's bed is silt, not boulders.
            if length < 1e-3 || a.lake || b.lake {
                continue;
            }
            let tangent = [d[0] / length, d[1] / length];
            let normal = [-tangent[1], tangent[0]];
            let key = cell_key(
                (a.position[0] * 2.0).floor() as i64,
                (a.position[1] * 2.0).floor() as i64,
                river.seed ^ 0x40C5,
            );
            let mut next = |channel: u64| {
                draw += 1;
                random(key, channel * 977 + draw)
            };
            // Stream power sets how rocky the bed is.
            let rocky = smoothstep(0.006, 0.07, a.slope).max(0.6 * smoothstep(0.3, 1.5, a.fall));
            let width = 2.0 * a.half_width;
            if a.fall > 0.35 {
                // A ledge of blocks across the lip, with gaps for the water,
                // and a few more tumbled into the plunge pool below.
                let blocks = (width / 1.6).clamp(2.0, 6.0) as usize;
                for k in 0..blocks {
                    if next(1) < 0.25 {
                        continue;
                    }
                    let u = (k as f32 + 0.5) / blocks as f32 * 2.0 - 1.0 + (next(2) - 0.5) * 0.2;
                    let radius = (0.35 + 0.35 * next(3)) * (0.5 * a.fall + 0.6).min(1.6) * (0.7 + 0.3 * a.half_width.min(3.0) / 3.0);
                    let back = -radius * 0.6;
                    let p = [
                        a.position[0] + normal[0] * u * a.half_width * 1.05 + tangent[0] * back,
                        a.position[1] + normal[1] * u * a.half_width * 1.05 + tangent[1] * back,
                    ];
                    try_place(&mut rocks, RiverRock {
                        position: p,
                        radius,
                        height: 0.0,
                        yaw: next(4) * std::f32::consts::TAU,
                        seed: next(5),
                        river: river_index as u32,
                        bed: 0.0,
                        model: 0,
                        scale: 1.0,
                    }, true);
                }
                for _ in 0..(1 + (next(6) * 3.0) as usize) {
                    let side = if next(7) < 0.5 { -1.0 } else { 1.0 };
                    let u = side * (0.55 + 0.45 * next(8));
                    let ahead = 1.0 + a.fall * 0.6 + next(9) * (2.0 + a.half_width);
                    let radius = (0.3 + 0.6 * next(10)) * (0.4 * a.fall + 0.5).min(1.4);
                    let p = [
                        a.position[0] + normal[0] * u * b.half_width + tangent[0] * ahead,
                        a.position[1] + normal[1] * u * b.half_width + tangent[1] * ahead,
                    ];
                    try_place(&mut rocks, RiverRock {
                        position: p,
                        radius,
                        height: 0.0,
                        yaw: next(11) * std::f32::consts::TAU,
                        seed: next(12),
                        river: river_index as u32,
                        bed: 0.0,
                        model: 0,
                        scale: 1.0,
                    }, false);
                }
            }
            // How much of the bed is gravel and cobble rather than sand: a
            // slow lowland reach settles sand over its stones.
            let stony = (0.3 + 0.7 * smoothstep(0.25, 0.8, a.speed)).max(rocky);
            // Cobbles pave the bed, from four per square metre of a
            // cascade's to one per eight of a sandy pool's.
            let cobbles = (0.12 + 0.5 * stony + 3.4 * rocky) * width * length;
            let count = cobbles.floor() as usize + usize::from(next(30) < cobbles.fract());
            for _ in 0..count {
                let along = next(31) * length;
                let u = (next(32) * 2.0 - 1.0) * 1.02;
                let size = next(33);
                let radius = 0.06 + (0.08 + 0.08 * rocky) * size * size + 0.05 * size;
                let p = [
                    a.position[0] + tangent[0] * along + normal[0] * u * a.half_width,
                    a.position[1] + tangent[1] * along + normal[1] * u * a.half_width,
                ];
                try_place(&mut rocks, RiverRock {
                    position: p,
                    radius,
                    height: 0.0,
                    yaw: next(34) * std::f32::consts::TAU,
                    seed: next(35),
                    river: river_index as u32,
                    bed: 0.0,
                    model: 0,
                    scale: 1.0,
                }, true);
            }
            // Stones piled along both waterlines, half out of the water.
            let edge = (0.6 + 1.2 * stony + 1.5 * rocky) * length;
            let count = edge.floor() as usize + usize::from(next(40) < edge.fract());
            for _ in 0..count {
                let side = if next(41) < 0.5 { -1.0 } else { 1.0 };
                let size = next(42);
                let radius = 0.08 + (0.12 + 0.25 * rocky) * size * size;
                let lateral = side * (a.half_width + radius * (next(43) * 1.2 - 0.6));
                let along = next(44) * length;
                let p = [
                    a.position[0] + tangent[0] * along + normal[0] * lateral,
                    a.position[1] + tangent[1] * along + normal[1] * lateral,
                ];
                try_place(&mut rocks, RiverRock {
                    position: p,
                    radius,
                    height: 0.0,
                    yaw: next(45) * std::f32::consts::TAU,
                    seed: next(46),
                    river: river_index as u32,
                    bed: 0.0,
                    model: 0,
                    scale: 1.0,
                }, next(47) < 0.6);
            }
            // Boulders among them: one per 60 m² of a lowland bed, one per
            // 6 m² of a cascade's.
            let density = 0.012 + 0.15 * rocky + 0.01 * stony;
            let expected = density * width * length;
            let count = expected.floor() as usize + usize::from(next(13) < expected.fract());
            for _ in 0..count {
                let along = next(14) * length;
                // Lowland stones gather toward the banks; a steep bed is
                // strewn right across.
                let raw = next(15) * 2.0 - 1.0;
                let u = raw.signum() * raw.abs().powf(0.6 - 0.35 * rocky) * 0.95;
                let size = next(16);
                let radius = (0.22 + (0.25 + 0.95 * rocky) * size * size) * (0.75 + 0.25 * (a.half_width / 3.0).min(1.5));
                let p = [
                    a.position[0] + tangent[0] * along + normal[0] * u * a.half_width,
                    a.position[1] + tangent[1] * along + normal[1] * u * a.half_width,
                ];
                try_place(&mut rocks, RiverRock {
                    position: p,
                    radius,
                    height: 0.0,
                    yaw: next(18) * std::f32::consts::TAU,
                    seed: next(19),
                    river: river_index as u32,
                    bed: 0.0,
                    model: 0,
                    scale: 1.0,
                }, false);
            }
            // Boulders lodged in the banks of steep reaches.
            let bank_rocks = 0.08 * rocky * length * 2.0;
            let count = bank_rocks.floor() as usize + usize::from(next(20) < bank_rocks.fract());
            for _ in 0..count {
                let side = if next(21) < 0.5 { -1.0 } else { 1.0 };
                let radius = 0.3 + 0.9 * rocky * next(22).powf(1.5);
                let lateral = side * (a.half_width + radius * (0.1 + 0.5 * next(23)));
                let along = next(24) * length;
                let p = [
                    a.position[0] + tangent[0] * along + normal[0] * lateral,
                    a.position[1] + tangent[1] * along + normal[1] * lateral,
                ];
                try_place(&mut rocks, RiverRock {
                    position: p,
                    radius,
                    height: 0.0,
                    yaw: next(25) * std::f32::consts::TAU,
                    seed: next(26),
                    river: river_index as u32,
                    bed: 0.0,
                    model: 0,
                    scale: 1.0,
                }, false);
            }
        }
    }
    rocks
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The placement table must describe the models the renderer draws, or
    /// the water would flow around boulders of the wrong size.
    #[test]
    fn rock_table_matches_the_models() {
        let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/models");
        for (index, expected) in ROCK_MODELS.iter().enumerate() {
            let path = directory.join(format!("rock-{}.glb", index + 1));
            let gltf = gltf::Gltf::open(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            let mut minimum = [f32::INFINITY; 3];
            let mut maximum = [f32::NEG_INFINITY; 3];
            for mesh in gltf.meshes() {
                for primitive in mesh.primitives() {
                    let bounds = primitive.bounding_box();
                    for axis in 0..3 {
                        minimum[axis] = minimum[axis].min(bounds.min[axis]);
                        maximum[axis] = maximum[axis].max(bounds.max[axis]);
                    }
                }
            }
            let actual = [
                (-minimum[0]).max(maximum[0]),
                (-minimum[2]).max(maximum[2]),
                maximum[1],
            ];
            for axis in 0..3 {
                assert!((actual[axis] - expected[axis]).abs() < 0.002, "rock-{}: {actual:?} vs {expected:?}", index + 1);
            }
            assert!(minimum[1].abs() < 0.01, "rock-{} must rest on its pivot", index + 1);
        }
    }
}
