//! Erosion input height fields and retained-footprint blend masks.

use forest_core::constants::*;
use forest_core::noise::{NoiseField, base_height};

use crate::{TileKey, tile_simulation_minimum};

/// The 480x480 base height map one erosion tile simulates, sampled at cell
/// centres (mirrors `CreateBaseHeightMap`), with the rivers carved into it.
/// The second map flags the cells a river's water covers: the simulation
/// holds them fixed and lets them drain what reaches them, since the river
/// carries it away.
///
/// Every cell is a pure function of its world position, so `workers` threads
/// splitting the rows between them write exactly the bits one thread does.
/// The rows are dealt out in turn rather than in blocks, which spreads the
/// costly cells beside the channels over every thread.
pub fn create_base_height_map(
    noise: &NoiseField,
    tile: TileKey,
    rivers: Option<&forest_water::rivers::network::RiverNetwork>,
    workers: usize,
) -> (Vec<f32>, Vec<f32>) {
    let world_minimum = tile_simulation_minimum(tile);
    let mut height = vec![0.0f32; EROSION_RESOLUTION * EROSION_RESOLUTION];
    let mut river = vec![0.0f32; EROSION_RESOLUTION * EROSION_RESOLUTION];
    let fill_row = |z: usize, height: &mut [f32], river: &mut [f32]| {
        let wz = world_minimum[1] + (z as f32 + 0.5) * EROSION_CELL_SIZE;
        for x in 0..EROSION_RESOLUTION {
            let wx = world_minimum[0] + (x as f32 + 0.5) * EROSION_CELL_SIZE;
            height[x] = base_height(noise, wx, wz);
            if let Some(network) = rivers {
                let envelope = network.envelope(wx, wz);
                height[x] = envelope.clamp(height[x]);
                // A cell whose middle the water covers belongs to the river,
                // or to a lake: the erosion drains both and leaves their beds.
                // A lake's shore up to 2.5 m over its surface stays as it
                // is too, so no gully cuts below the water there, nor down
                // through the surface its shore is measured from.
                let on_lake = envelope.lake > height[x] - 2.5;
                if on_lake
                    || envelope
                        .water_over(height[x])
                        .is_some_and(|water| height[x] < water)
                {
                    river[x] = 1.0;
                }
            }
        }
    };
    let rows = height
        .chunks_mut(EROSION_RESOLUTION)
        .zip(river.chunks_mut(EROSION_RESOLUTION))
        .enumerate();
    let workers = workers.clamp(1, EROSION_RESOLUTION);
    if workers == 1 {
        for (z, (height, river)) in rows {
            fill_row(z, height, river);
        }
    } else {
        let mut shares: Vec<Vec<_>> = (0..workers).map(|_| Vec::new()).collect();
        for (z, row) in rows {
            shares[z % workers].push((z, row));
        }
        std::thread::scope(|scope| {
            for share in shares {
                let fill_row = &fill_row;
                scope.spawn(move || {
                    for (z, (height, river)) in share {
                        fill_row(z, height, river);
                    }
                });
            }
        });
    }
    (height, river)
}

/// The separable Hermite blend mask applied to each tile's retained
/// footprint (mirrors `CreateErosionBlendMask`'s CPU half; the GPU samples
/// the uploaded texture version for bilinear consistency).
pub fn erosion_blend_mask_samples() -> Vec<f32> {
    let half_resolution = EROSION_OUTPUT_RESOLUTION / 2;
    let mut axis_weights = vec![0.0f32; EROSION_OUTPUT_RESOLUTION];
    for index in 0..half_resolution {
        let phase = index as f32 / (half_resolution - 1) as f32;
        let rising = smoothstep(0.0, 1.0, phase);
        axis_weights[index] = rising;
        axis_weights[index + half_resolution] = 1.0 - rising;
    }
    let mut cpu_samples = vec![0.0f32; EROSION_OUTPUT_RESOLUTION * EROSION_OUTPUT_RESOLUTION];
    for (z, weight_z) in axis_weights.iter().enumerate() {
        for (x, weight_x) in axis_weights.iter().enumerate() {
            let mask = (weight_x * weight_z).clamp(0.0, 1.0);
            cpu_samples[z * EROSION_OUTPUT_RESOLUTION + x] = mask;
        }
    }
    cpu_samples
}

fn smoothstep(edge0: f32, edge1: f32, value: f32) -> f32 {
    let t = ((value - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}
