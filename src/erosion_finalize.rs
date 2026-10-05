//! The CPU half of the C++ `FinalizeErosionTile`: the finalize math the
//! erosion worker runs on its readback, and the readback byte conversions.
//!
//! A plain module with no GPU and no bevy types, because two sides use it:
//! the worker thread finalizes each tile as its map callbacks land, and the
//! render world's slim node still needs the byte conversions for the lookup
//! and atlas uploads it performs into the shared world textures.

use crate::constants::*;
use crate::erosion::{FinalizedTile, TileDiagnostics, TileKey};

/// Crops the retained footprint out of the 480x480 readback and rebuilds the
/// C++'s atlas patch. Returns the tile's event payload (with a placeholder
/// slot; the main world publishes the slot through the lookup records) plus
/// the 260x260 height and flow patches.
pub fn finalize_erosion_tile(
    key: TileKey,
    terrain_rgba: &[f32],
    flow_rgba: &[f32],
    drainage_rgba: &[f32],
) -> (FinalizedTile, Vec<f32>, Vec<f32>) {
    let resolution = EROSION_RESOLUTION;
    let output_resolution = EROSION_OUTPUT_RESOLUTION;
    let output_pixels = output_resolution * output_resolution;

    let mut cpu_delta = vec![0.0f32; output_pixels];
    let mut output_height = vec![0.0f32; output_pixels * 4];
    let mut output_flow = vec![0.0f32; output_pixels * 4];

    // Preserve signed displacement in R for rendering and collision. The
    // remaining channels describe the final bed: G concavity (metres),
    // B drainage concentration (positive log2 ratio of contributing area
    // against the surrounding ring), A loose cover thickness (metres). The
    // flow patch keeps the solver's water depth and velocity and carries the
    // routed contributing area, in cells, in A.
    const MATERIAL_RING_OFFSET: usize = 3;
    const DRAINAGE_EPSILON: f32 = 1.0;

    let mut minimum = f32::INFINITY;
    let mut maximum = f32::NEG_INFINITY;
    let mut minimum_delta = f32::INFINITY;
    let mut maximum_delta = f32::NEG_INFINITY;
    let mut flow_axis_bias = 0.0f64;
    let mut moving_flow_cells = 0usize;
    let mut land_cells = 0usize;
    let mut maximum_drainage = 0.0f32;
    let mut bare_cells = 0usize;
    let mut loose_total = 0.0f64;
    for output_z in 0..output_resolution {
        for output_x in 0..output_resolution {
            let source_x = output_x + EROSION_OUTPUT_OFFSET;
            let source_z = output_z + EROSION_OUTPUT_OFFSET;
            let source = source_z * resolution + source_x;
            let height = terrain_rgba[source * 4];
            let base_height = terrain_rgba[source * 4 + 2];
            let delta = height - base_height;
            let destination = output_z * output_resolution + output_x;
            cpu_delta[destination] = delta;

            // Sample the simulation halo too, so the 12 m axial ring does not
            // flatten at the retained footprint's edges.
            let mut surrounding_height = 0.0f32;
            let mut surrounding_drainage = 0.0f32;
            for ring_z in -1i64..=1 {
                for ring_x in -1i64..=1 {
                    if ring_x == 0 && ring_z == 0 {
                        continue;
                    }
                    let neighbour_z =
                        (source_z as i64 + ring_z * MATERIAL_RING_OFFSET as i64) as usize;
                    let neighbour_x =
                        (source_x as i64 + ring_x * MATERIAL_RING_OFFSET as i64) as usize;
                    let neighbour = neighbour_z * resolution + neighbour_x;
                    surrounding_height += terrain_rgba[neighbour * 4];
                    surrounding_drainage += drainage_rgba[neighbour * 4].max(0.0);
                }
            }
            surrounding_height *= 0.125;
            surrounding_drainage *= 0.125;
            let drainage_area = drainage_rgba[source * 4].max(0.0);
            output_height[destination * 4] = delta;
            output_height[destination * 4 + 1] = surrounding_height - height;
            output_height[destination * 4 + 2] = ((drainage_area + DRAINAGE_EPSILON)
                / (surrounding_drainage + DRAINAGE_EPSILON))
                .log2()
                .max(0.0);
            output_height[destination * 4 + 3] = terrain_rgba[source * 4 + 3].max(0.0);
            for channel in 0..3 {
                output_flow[destination * 4 + channel] = flow_rgba[source * 4 + channel];
            }
            output_flow[destination * 4 + 3] = drainage_area;
            minimum = minimum.min(height);
            maximum = maximum.max(height);
            minimum_delta = minimum_delta.min(delta);
            maximum_delta = maximum_delta.max(delta);
            if height > SEA_LEVEL {
                land_cells += 1;
                let loose = terrain_rgba[source * 4 + 3].max(0.0);
                loose_total += loose as f64;
                if loose < 0.1 {
                    bare_cells += 1;
                }
            }
            maximum_drainage = maximum_drainage.max(drainage_area);

            let velocity_x = flow_rgba[source * 4 + 1];
            let velocity_z = flow_rgba[source * 4 + 2];
            let velocity_sum = velocity_x.abs() + velocity_z.abs();
            if velocity_sum > 0.01 {
                flow_axis_bias += (velocity_x.abs().max(velocity_z.abs()) / velocity_sum) as f64;
                moving_flow_cells += 1;
            }
        }
    }

    // The detail metric only looks at the retained footprint.
    let mut detail = 0.0f64;
    let mut detail_samples = 0usize;
    for z in 1..output_resolution - 1 {
        for x in 1..output_resolution - 1 {
            let center_delta = cpu_delta[z * output_resolution + x];
            if center_delta < -0.05 {
                let neighbour_mean = 0.25
                    * (cpu_delta[z * output_resolution + x - 1]
                        + cpu_delta[z * output_resolution + x + 1]
                        + cpu_delta[(z - 1) * output_resolution + x]
                        + cpu_delta[(z + 1) * output_resolution + x]);
                detail += ((center_delta - neighbour_mean).abs()) as f64;
                detail_samples += 1;
            }
        }
    }

    // Upload this pass exactly as simulated. The shader combines four distinct
    // atlas regions; no canonical overlap reconciliation is performed. The
    // gutter duplicates the border texels, so the 12 m ring does not clamp to
    // an atlas neighbour.
    let atlas_pitch = EROSION_ATLAS_PITCH;
    let gutter = EROSION_ATLAS_GUTTER as i64;
    let mut atlas_height = vec![0.0f32; atlas_pitch * atlas_pitch * 4];
    let mut atlas_flow = vec![0.0f32; atlas_pitch * atlas_pitch * 4];
    for atlas_z in 0..atlas_pitch {
        let output_z = (atlas_z as i64 - gutter).clamp(0, output_resolution as i64 - 1) as usize;
        for atlas_x in 0..atlas_pitch {
            let output_x =
                (atlas_x as i64 - gutter).clamp(0, output_resolution as i64 - 1) as usize;
            let source = output_z * output_resolution + output_x;
            let destination = atlas_z * atlas_pitch + atlas_x;
            for channel in 0..4 {
                atlas_height[destination * 4 + channel] = output_height[source * 4 + channel];
                atlas_flow[destination * 4 + channel] = output_flow[source * 4 + channel];
            }
        }
    }

    let stats = TileDiagnostics {
        minimum_height: minimum,
        maximum_height: maximum,
        land_coverage: 100.0 * land_cells as f32 / (output_resolution * output_resolution) as f32,
        maximum_incision: (-minimum_delta).max(0.0),
        maximum_deposition: maximum_delta.max(0.0),
        erosion_detail: if detail_samples > 0 {
            (detail / detail_samples as f64) as f32
        } else {
            0.0
        },
        flow_axis_bias: if moving_flow_cells > 0 {
            (flow_axis_bias / moving_flow_cells as f64) as f32
        } else {
            0.0
        },
        maximum_drainage,
        bedrock_exposure: 100.0 * bare_cells as f32 / land_cells.max(1) as f32,
        mean_loose_cover: (loose_total / land_cells.max(1) as f64) as f32,
    };

    let tile = FinalizedTile {
        key,
        cpu_delta,
        stats,
    };
    (tile, atlas_height, atlas_flow)
}

/// wgpu maps staging buffers as little-endian bytes and the host is LE too,
/// so a `cast_slice` round-trips floats bit-exactly without per-float work.
/// Truncates a trailing partial chunk exactly as a per-float
/// `chunks_exact(4)` walk did, instead of panicking on a misaligned map
/// range.
pub fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    let aligned = bytes.len() - bytes.len() % 4;
    bytemuck::cast_slice::<u8, f32>(&bytes[..aligned]).to_vec()
}

/// The inverse of [`bytes_to_f32`], for the lookup records and atlas patches
/// the render world uploads through `write_padded_at`.
pub fn f32_to_bytes(values: &[f32]) -> Vec<u8> {
    bytemuck::cast_slice::<f32, u8>(values).to_vec()
}

#[cfg(test)]
mod cast_slice_parity_tests {
    use super::*;

    /// The per-float implementations these replaced: kept so the bytemuck
    /// versions are pinned to them byte-for-byte.
    fn bytes_to_f32_per_float(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()
    }

    #[test]
    fn cast_slice_matches_per_float_conversions() {
        let mut values = Vec::new();
        for seed in 0..9999u32 {
            values.push(f32::from_bits(seed.wrapping_mul(2654435761)));
        }
        values.push(f32::NAN);
        values.push(f32::NEG_INFINITY);
        values.push(0.0);
        values.push(-0.0);
        let bytes = f32_to_bytes(&values);
        assert_eq!(bytes.len(), values.len() * 4);
        // Bit equality, not PartialEq: the vector contains a NaN, and NaN
        // != NaN even when the bits round-trip exactly.
        assert_eq!(
            bytes_to_f32(&bytes).iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
            values.iter().map(|f| f.to_bits()).collect::<Vec<_>>()
        );
        let reconverted = bytes_to_f32_per_float(&bytes);
        for (fast, slow) in values.iter().zip(reconverted.iter()) {
            assert_eq!(fast.to_bits(), slow.to_bits(), "NaN/zero sign payloads differ");
        }
        // A trailing partial chunk is dropped, matching chunks_exact.
        let truncated = bytes_to_f32(&bytes[..bytes.len() - 3]);
        assert_eq!(truncated.len(), values.len() - 1);
        assert_eq!(
            truncated.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
            values[..values.len() - 1].iter().map(|f| f.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(bytes_to_f32(&bytes[..3]), &[] as &[f32]);
    }

    /// A finalize over a zeroed readback must stay finite: the tile
    /// diagnostics divide by cell counts that are clamped, never by raw
    /// readings (the sea-hole NaN capture regression lives one level up, in
    /// the particle vertex — this pins the finalize side).
    #[test]
    fn finalize_over_zeroed_readback_stays_finite() {
        let pixels = EROSION_RESOLUTION * EROSION_RESOLUTION;
        let tile = finalize_erosion_tile(
            crate::erosion::tile(1, 2),
            &vec![0.0; pixels * 4],
            &vec![0.0; pixels * 4],
            &vec![0.0; pixels * 4],
        );
        let stats = tile.0.stats;
        for value in [
            stats.minimum_height,
            stats.maximum_height,
            stats.land_coverage,
            stats.maximum_incision,
            stats.maximum_deposition,
            stats.erosion_detail,
            stats.flow_axis_bias,
            stats.maximum_drainage,
            stats.bedrock_exposure,
            stats.mean_loose_cover,
        ] {
            assert!(value.is_finite(), "{value} is not finite");
        }
        assert_eq!(tile.1.len(), EROSION_ATLAS_PITCH * EROSION_ATLAS_PITCH * 4);
        assert_eq!(tile.2.len(), EROSION_ATLAS_PITCH * EROSION_ATLAS_PITCH * 4);
    }
}