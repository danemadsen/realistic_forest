use super::*;

/// The per-float implementations these replaced: kept so the bytemuck
/// versions are pinned to them byte-for-byte.
fn bytes_to_f32_per_float(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

#[test]
fn byte_conversion_accepts_unaligned_slices() {
    let storage = [0x0000_0080u32, 0x0000_003fu32, 0x0000_0080u32];
    let bytes = bytemuck::cast_slice::<u32, u8>(&storage);
    for offset in 1..4 {
        let slice = &bytes[offset..offset + 8];
        let expected = bytes_to_f32_per_float(slice);
        let actual = bytes_to_f32(slice);
        let expected_bits: Vec<u32> = expected.iter().map(|value| value.to_bits()).collect();
        let actual_bits: Vec<u32> = actual.iter().map(|value| value.to_bits()).collect();
        assert_eq!(actual_bits, expected_bits, "byte offset {offset}");
        assert!(bytes_to_f32(&slice[..3]).is_empty());
    }
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
        bytes_to_f32(&bytes)
            .iter()
            .map(|f| f.to_bits())
            .collect::<Vec<_>>(),
        values.iter().map(|f| f.to_bits()).collect::<Vec<_>>()
    );
    let reconverted = bytes_to_f32_per_float(&bytes);
    for (fast, slow) in values.iter().zip(reconverted.iter()) {
        assert_eq!(
            fast.to_bits(),
            slow.to_bits(),
            "NaN/zero sign payloads differ"
        );
    }
    // A trailing partial chunk is dropped, matching chunks_exact.
    let truncated = bytes_to_f32(&bytes[..bytes.len() - 3]);
    assert_eq!(truncated.len(), values.len() - 1);
    assert_eq!(
        truncated.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
        values[..values.len() - 1]
            .iter()
            .map(|f| f.to_bits())
            .collect::<Vec<_>>()
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
        crate::tile(1, 2),
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
/// The snow pack reads the eroded bed's concavity from the CPU copy under
/// the player and from the atlas everywhere else: they must be one value.
/// On a paraboloid h = k r^2 / 2 the 12 m ring (axial 12 m, diagonal
/// 17 m) averages 108 k of relief over the point.
#[test]
fn finalize_keeps_the_concavity_it_uploads() {
    let resolution = EROSION_RESOLUTION;
    let curvature = 0.01f32;
    let centre = resolution as f32 * 0.5 * EROSION_CELL_SIZE;
    let mut terrain = vec![0.0f32; resolution * resolution * 4];
    for z in 0..resolution {
        for x in 0..resolution {
            let dx = x as f32 * EROSION_CELL_SIZE - centre;
            let dz = z as f32 * EROSION_CELL_SIZE - centre;
            let index = (z * resolution + x) * 4;
            terrain[index] = 0.5 * curvature * (dx * dx + dz * dz);
            // The base under it is flat: only a ring over the eroded bed
            // (channel 0) finds the paraboloid's relief.
            terrain[index + 2] = 0.0;
        }
    }
    let zeros = vec![0.0; resolution * resolution * 4];
    let (tile, atlas_height, _) =
        finalize_erosion_tile(crate::tile(0, 0), &terrain, &zeros, &zeros);
    let output = EROSION_OUTPUT_RESOLUTION;
    assert_eq!(tile.cpu_concavity.len(), output * output);
    for z in 0..output {
        for x in 0..output {
            let atlas = (z + EROSION_ATLAS_GUTTER) * EROSION_ATLAS_PITCH + x + EROSION_ATLAS_GUTTER;
            let concavity = tile.cpu_concavity[z * output + x];
            assert_eq!(concavity.to_bits(), atlas_height[atlas * 4 + 1].to_bits());
            assert!(
                (concavity - 108.0 * curvature).abs() < 2.0e-3,
                "{concavity}"
            );
        }
    }
}
