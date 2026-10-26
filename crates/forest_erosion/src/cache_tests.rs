use super::*;

/// A frame of work for one tile, with no init (the common case: only the
/// frame that calls `BeginErosionTile` carries one).
fn frame_for(sim_min: [f32; 2], iterate: usize, finalize: bool) -> ErosionFrameCommands {
    ErosionFrameCommands {
        simulation_id: SimulationId::default(),
        sim_min,
        init: None,
        iterate: (iterate > 0).then(|| IterateCommand {
            count: iterate,
            settings: ErosionSettings::default(),
        }),
        finalize,
    }
}

/// The render world misses frames while `prepare_erosion_sim` builds its
/// GPU state, and the `init` `BeginErosionTile` sends is one-shot: losing
/// it leaves a tile `Simulating` with nothing to simulate and no erosion
/// ever finalizes.
#[test]
fn set_frame_carries_a_pending_init_to_the_next_frame() {
    let bridge = ErosionBridge::default();
    let with_pending_init = ErosionFrameCommands {
        simulation_id: SimulationId::default(),
        sim_min: [0.0, 0.0],
        init: Some(InitTileCommand {
            key: tile(0, 0),
            base_height: vec![1.0, 2.0],
            drainage_area: vec![1.0, 1.0],
            river: vec![0.0, 0.0],
        }),
        iterate: Some(IterateCommand {
            count: 6,
            settings: ErosionSettings::default(),
        }),
        finalize: false,
    };
    bridge.set_frame(with_pending_init, None);
    // Nothing consumed it, and the next frame carries no init of its own.
    bridge.set_frame(frame_for([0.0, 0.0], 6, false), None);

    let taken = bridge.take_commands().expect("commands survived");
    assert!(taken.init.is_some(), "the pending init must not be dropped");
    assert_eq!(taken.iterate.expect("iterate").count, 12);
}

/// Once the main world moves to another tile, anything still pending for
/// the old one is a redundant re-send and must not leak into the new
/// tile's frame — merging its `finalize` would end the new tile after a
/// single batch.
#[test]
fn set_frame_does_not_carry_work_across_tiles() {
    let bridge = ErosionBridge::default();
    bridge.set_frame(frame_for([0.0, 0.0], 6, true), None);
    bridge.set_frame(frame_for([512.0, 0.0], 6, false), None);

    let taken = bridge.take_commands().expect("commands");
    assert!(!taken.finalize);
    assert_eq!(taken.iterate.expect("iterate").count, 6);
}

/// A cache whose four passes around (300, 300) are Ready with constant
/// fields: delta 1, 2, 3, 4 m and concavity 10, 20, 30, 40 m.
fn ready_quartet(reveal: f32) -> ErosionCache {
    let mut cache = ErosionCache {
        blend_mask_samples: crate::erosion_blend_mask_samples(),
        ..Default::default()
    };
    let lower = erosion_candidate_minimum(300.0, 300.0);
    let pixels = EROSION_OUTPUT_RESOLUTION * EROSION_OUTPUT_RESOLUTION;
    for (index, key) in [
        lower,
        tile(lower.x + 1, lower.z),
        tile(lower.x, lower.z + 1),
        tile(lower.x + 1, lower.z + 1),
    ]
    .into_iter()
    .enumerate()
    {
        let mut found = ErosionTile::new(key, index);
        found.state = ErosionTileState::Ready;
        found.reveal = reveal;
        found.cpu_delta = vec![(index + 1) as f32; pixels];
        found.cpu_concavity = vec![10.0 * (index + 1) as f32; pixels];
        cache.tiles.insert(key, found);
    }
    cache
}

/// The snow reads the bed's concavity blended over the same four passes,
/// with the same partition, reveal and fade as the bed's change, so the
/// footing on snow matches the pack the shaders draw.
#[test]
fn concavity_blends_like_the_delta() {
    let noise = NoiseField::new();
    let cache = ready_quartet(0.5);
    let centre = [300.0 - 1300.0, 300.0];
    let fade = erosion_visibility(300.0, 300.0, centre);
    assert!(fade > 0.0 && fade < 1.0);
    let delta = sample_eroded_height(&cache, &noise, 300.0, 300.0, centre)
        - forest_core::noise::base_height(&noise, 300.0, 300.0);
    let concavity = sample_eroded_concavity(&cache, 300.0, 300.0, centre);
    assert!(delta > 0.5 * fade && delta < 2.0 * fade, "{delta}");
    assert!(
        (concavity - 10.0 * delta).abs() < 1.0e-3,
        "{concavity} against {delta}"
    );
    assert_eq!(
        sample_eroded_concavity(&ErosionCache::default(), 300.0, 300.0, centre),
        0.0
    );
}

/// Beyond the fade's zero radius no tile is sampled and the ground is
/// exactly the base landform, bit for bit, as the shaders draw it. The
/// tiles hold NaN there: sampling one at all would poison the result.
#[test]
fn eroded_height_beyond_the_fade_is_the_base_height() {
    let noise = NoiseField::new();
    let mut poisoned = ready_quartet(1.0);
    for found in poisoned.tiles.values_mut() {
        found.cpu_delta.fill(f32::NAN);
        found.cpu_concavity.fill(f32::NAN);
    }
    let centre = [300.0 + EROSION_VISIBILITY_ZERO_RADIUS + 1.0, 300.0];
    assert_eq!(erosion_visibility(300.0, 300.0, centre), 0.0);
    assert_eq!(
        sample_eroded_height(&poisoned, &noise, 300.0, 300.0, centre).to_bits(),
        forest_core::noise::base_height(&noise, 300.0, 300.0).to_bits()
    );
    assert_eq!(
        sample_eroded_concavity(&poisoned, 300.0, 300.0, centre),
        0.0
    );
    let cache = ready_quartet(1.0);
    let near = sample_eroded_height(&cache, &noise, 300.0, 300.0, [300.0, 300.0]);
    assert!(near > forest_core::noise::base_height(&noise, 300.0, 300.0) + 0.5);
}

/// An inclined plane drains down its slope: every row passes everything
/// it holds to the row below, so row z carries (z + 1) rows of cells, and
/// columns away from the closed side walls carry exactly z + 1 each.
#[test]
fn base_drainage_accumulates_down_a_plane() {
    let resolution = 8;
    let heights: Vec<f32> = (0..resolution * resolution)
        .map(|cell| 100.0 - (cell / resolution) as f32 * 2.0)
        .collect();
    let area = route_base_drainage(&heights, resolution, 4.0);
    for z in 0..resolution {
        let row: f32 = (0..resolution).map(|x| area[z * resolution + x]).sum();
        assert!(
            (row - ((z + 1) * resolution) as f32).abs() < 1e-3,
            "row {z} = {row}"
        );
        // The walls' influence spreads one column per row.
        for x in (z + 1).min(resolution)..resolution.saturating_sub(z + 1) {
            let value = area[z * resolution + x];
            assert!(
                (value - (z + 1) as f32).abs() < 1e-3,
                "({x}, {z}) = {value}"
            );
        }
    }
}

/// A valley gathers its sides into the thalweg, nothing is created or
/// lost on the way to the closed outlet row, and the sea swallows its share.
#[test]
fn base_drainage_conserves_area_into_the_valley_and_the_sea() {
    let resolution = 9;
    let valley_height = |cell| {
        let x = (cell % resolution) as f32;
        let z = (cell / resolution) as f32;
        // A V valley along x = 4 falling toward a flat strand at
        // z = 7 that only drains into the sea row at z = 8.
        match z as usize {
            8 => -5.0,
            7 => 1.0,
            _ => 40.0 + (x - 4.0).abs() * 3.0 - z,
        }
    };
    let heights: Vec<f32> = (0..resolution * resolution).map(valley_height).collect();
    let area = route_base_drainage(&heights, resolution, 4.0);
    let land_cells = resolution * (resolution - 1);
    // The strand cells exchange nothing among themselves, so their areas
    // partition every land cell exactly once.
    let reaching_sea: f32 = (0..resolution)
        .map(|x| area[(resolution - 2) * resolution + x])
        .sum();
    assert!(
        (reaching_sea - land_cells as f32).abs() < 1e-2,
        "{reaching_sea}"
    );
    let thalweg = area[(resolution - 3) * resolution + 4];
    assert!(
        thalweg > area[(resolution - 3) * resolution + 2] * 3.0,
        "{thalweg}"
    );
    assert!(area[(resolution - 1) * resolution + 4] == 0.0);
}

/// Rough ground with flats, pits, plateaus and a sea.
fn rough_ground(resolution: usize) -> Vec<f32> {
    let mut seed = 0x2545_f491u32;
    let quantized_height = |cell| {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        let x = (cell % resolution) as f32;
        let z = (cell / resolution) as f32;
        let relief = 30.0 * (x * 0.11).sin() * (z * 0.07).cos() + 0.4 * z - 12.0;
        // Quantised noise leaves exact ties (flats) between neighbours.
        ((relief + (seed % 3) as f32 * 0.1) * 4.0).round() * 0.25
    };
    (0..resolution * resolution).map(quantized_height).collect()
}

/// The donor-count ordering routes exactly what visiting cells highest
/// first did, on rough ground with flats, pits, plateaus and a sea.
#[test]
fn base_drainage_matches_highest_first_routing() {
    let resolution = 96;
    let heights = rough_ground(resolution);
    let routed = route_base_drainage(&heights, resolution, 4.0);
    let reference = highest_first_drainage(&heights, resolution, 4.0);
    for (cell, (a, b)) in routed.iter().zip(&reference).enumerate() {
        assert!(
            (a - b).abs() <= 1e-3 * b.max(1.0),
            "cell {cell}: {a} vs {b}"
        );
    }
    assert!(reference.iter().any(|area| *area > 100.0));
    assert!(heights.iter().any(|height| *height < SEA_LEVEL));
}

fn highest_first_drainage(heights: &[f32], resolution: usize, cell_size: f32) -> Vec<f32> {
    let count = resolution * resolution;
    let mut area = vec![1.0f32; count];
    let mut order: Vec<usize> = (0..count).collect();
    order.sort_unstable_by(|&a, &b| heights[b].total_cmp(&heights[a]));
    for cell in order {
        let height = heights[cell];
        if height < SEA_LEVEL {
            continue;
        }
        let mut shares = Vec::new();
        for_each_neighbour(cell, resolution, |neighbour, diagonal| {
            let distance = if diagonal {
                cell_size * std::f32::consts::SQRT_2
            } else {
                cell_size
            };
            let slope = (height - heights[neighbour]) / distance;
            if slope > 0.0 {
                shares.push((neighbour, slope * slope));
            }
        });
        let total: f32 = shares.iter().map(|share| share.1).sum();
        let passed = area[cell];
        for (neighbour, share) in shares {
            area[neighbour] += passed * share / total;
        }
    }
    for (cell, height) in heights.iter().enumerate() {
        if *height < SEA_LEVEL {
            area[cell] = 0.0;
        }
    }
    area
}

/// `route_base_drainage` as it was before each cell counted its own
/// donors: every donor added itself to its receivers' counts, and every
/// neighbour visit tested the grid's bounds.
fn scatter_reference_drainage(heights: &[f32], resolution: usize, cell_size: f32) -> Vec<f32> {
    let count = resolution * resolution;
    let mut area = vec![1.0f32; count.min(heights.len())];
    if area.len() != count {
        return area;
    }
    let mut waiting = vec![0u8; count];
    for cell in 0..count {
        let height = heights[cell];
        if height < SEA_LEVEL {
            continue;
        }
        for_each_neighbour(cell, resolution, |neighbour, _| {
            if heights[neighbour] < height {
                waiting[neighbour] += 1;
            }
        });
    }
    let mut ready: Vec<u32> = (0..count as u32)
        .filter(|&cell| waiting[cell as usize] == 0)
        .collect();
    let diagonal = cell_size * std::f32::consts::SQRT_2;
    while let Some(cell) = ready.pop() {
        let cell = cell as usize;
        let height = heights[cell];
        if height < SEA_LEVEL {
            continue;
        }
        let mut shares = [(0usize, 0.0f32); 8];
        let mut receivers = 0;
        let mut share_total = 0.0f32;
        for_each_neighbour(cell, resolution, |neighbour, diagonal_step| {
            let distance = if diagonal_step { diagonal } else { cell_size };
            let slope = (height - heights[neighbour]) / distance;
            if slope > 0.0 {
                shares[receivers] = (neighbour, slope * slope);
                receivers += 1;
                share_total += slope * slope;
            }
        });
        let passed = area[cell];
        for &(neighbour, share) in &shares[..receivers] {
            if share_total > 0.0 {
                area[neighbour] += passed * share / share_total;
            }
            waiting[neighbour] -= 1;
            if waiting[neighbour] == 0 {
                ready.push(neighbour as u32);
            }
        }
    }
    for (cell, height) in heights.iter().enumerate() {
        if *height < SEA_LEVEL {
            area[cell] = 0.0;
        }
    }
    area
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

/// Each cell counting its own donors, and the bounds tests skipped inside
/// the grid, leave the routing exactly as the donors' scatter had it: the
/// same release order and the same float sums, bit for bit. Checked on
/// small grids whose edges are most of the cells, on rough ground with
/// flats, pits and a sea, and on three real tiles.
#[test]
fn drainage_matches_the_scatter_reference_bit_for_bit() {
    let plane = |resolution: usize| -> Vec<f32> {
        (0..resolution * resolution)
            .map(|cell| 100.0 - (cell / resolution) as f32 * 2.0 + (cell % 3) as f32 * 0.1)
            .collect()
    };
    let mut grids = vec![
        (plane(1), 1),
        (plane(2), 2),
        (plane(3), 3),
        (plane(8), 8),
        (rough_ground(96), 96),
        (rough_ground(131), 131),
    ];
    let noise = NoiseField::new();
    for key in [tile(0, 0), tile(-1, 2), tile(3, -5)] {
        let (heights, _) = crate::create_base_height_map(&noise, key, None, 4);
        grids.push((heights, EROSION_RESOLUTION));
    }
    let mut largest = 0.0f32;
    for (heights, resolution) in &grids {
        let routed = route_base_drainage(heights, *resolution, EROSION_CELL_SIZE);
        let reference = scatter_reference_drainage(heights, *resolution, EROSION_CELL_SIZE);
        assert!(
            bits(&routed) == bits(&reference),
            "{resolution}-cell grid routes differently"
        );
        largest = routed.into_iter().fold(largest, f32::max);
    }
    assert!(
        largest > 1000.0,
        "real tiles gather catchments, largest {largest} cells"
    );
}

/// Rows dealt out to several threads change nothing: a tile's base
/// heights, river cells and routed drainage are the same bits from one
/// thread as from four, over the spawn region's rivers.
#[test]
fn prepared_tiles_are_bit_identical_across_worker_counts() {
    let (noise, network) = forest_water::rivers::spawn_network();
    let mut river_cells = 0;
    for key in [tile(0, 0), tile(-1, 2), tile(3, -5)] {
        let serial = prepare_tile(&noise, Some(&network), key, 1);
        let parallel = prepare_tile(&noise, Some(&network), key, 4);
        assert!(
            bits(&serial.base_height) == bits(&parallel.base_height),
            "{key:?} heights"
        );
        assert!(
            bits(&serial.river) == bits(&parallel.river),
            "{key:?} river cells"
        );
        assert!(
            bits(&serial.drainage_area) == bits(&parallel.drainage_area),
            "{key:?} drainage"
        );
        river_cells += serial.river.iter().filter(|cell| **cell > 0.5).count();
    }
    assert!(
        river_cells > 0,
        "the tiles cross the rivers, so the carving is compared too"
    );
}

/// One cell of the thermal pass's input: bed height (terrain R), loose
/// cover (terrain A), and the contributing area and bedrock resistance
/// the terrain pass left in drainage R and B.
#[derive(Clone, Copy)]
struct ThermalCell {
    bed: f32,
    loose: f32,
    area: f32,
    resistance: f32,
}

/// `weathering` in erosion_thermal.wgsl, line by line.
fn weathering_reference(
    loose: f32,
    resistance: f32,
    area: f32,
    steepest_rock: f32,
    rate: f32,
) -> f32 {
    let weatherability = 1.0 - smoothstep(0.25, 0.75, resistance);
    let shedding = smoothstep(
        EROSION_LOOSE_REPOSE - 0.05,
        EROSION_LOOSE_REPOSE + 0.20,
        steepest_rock,
    );
    let cover_scale = lerp(
        EROSION_WEATHERING_COVER,
        EROSION_WEATHERING_COVER_STEEP,
        shedding,
    );
    let shielding = (-loose / cover_scale.max(0.001)).exp();
    let exposed = smoothstep(
        EROSION_WEATHERING_SLOPE,
        EROSION_LOOSE_REPOSE,
        steepest_rock,
    );
    let hillslope = 1.0 - smoothstep(12.0, 60.0, area);
    rate.max(0.0) * weatherability * shielding * exposed * hillslope
}

/// One step of erosion_thermal.wgsl's `fs_main` over a `width` x `depth`
/// grid of land, line by line: the pairwise talus and rockfall exchange,
/// then weathering at `rate`, every cell reading only the step's input.
/// `wrap_z` joins the grid's z edges, so a strip stands for an endless
/// escarpment; the shader's edges are closed walls.
fn thermal_reference_step(
    cells: &[ThermalCell],
    width: usize,
    depth: usize,
    wrap_z: bool,
    rate: f32,
) -> Vec<ThermalCell> {
    let settings = ErosionSettings::default();
    let loose_rate = settings.talus_rate.clamp(0.0, 0.0625);
    let rock_rate = settings.rockfall_rate.clamp(0.0, 0.0625);
    let rock_limit = |resistance: f32| {
        lerp(
            EROSION_SOFT_ROCK_SLOPE,
            EROSION_HARD_ROCK_SLOPE,
            resistance.clamp(0.0, 1.0),
        )
    };
    let mut next = cells.to_vec();
    for z in 0..depth {
        for x in 0..width {
            let center = cells[z * width + x];
            let center_loose = center.loose.max(0.0);
            let center_rock_limit = rock_limit(center.resistance);
            let center_rock = center.bed - center_loose;
            let mut height_change = 0.0f32;
            let mut loose_change = 0.0f32;
            let mut steepest_rock = 0.0f32;
            for (dx, dz) in NEIGHBOURS {
                let nx = x as i64 + dx;
                let mut nz = z as i64 + dz;
                if wrap_z {
                    nz = nz.rem_euclid(depth as i64);
                }
                if nx < 0 || nz < 0 || nx >= width as i64 || nz >= depth as i64 {
                    continue;
                }
                let neighbour = cells[nz as usize * width + nx as usize];
                // SQRT_2 is the very f32 the shader's 1.41421356 rounds to.
                let diagonal = dx != 0 && dz != 0;
                let distance = EROSION_CELL_SIZE
                    * if diagonal {
                        std::f32::consts::SQRT_2
                    } else {
                        1.0
                    };
                steepest_rock = steepest_rock
                    .max((center_rock - (neighbour.bed - neighbour.loose.max(0.0))) / distance);
                let drop = center.bed - neighbour.bed;
                let fall = drop.abs();
                if fall <= EROSION_LOOSE_REPOSE * distance {
                    continue;
                }
                let center_is_source = drop > 0.0;
                let source_loose = if center_is_source {
                    center_loose
                } else {
                    neighbour.loose.max(0.0)
                };
                let source_rock_limit = if center_is_source {
                    center_rock_limit
                } else {
                    rock_limit(neighbour.resistance)
                };
                let loose_moved = (loose_rate * (fall - EROSION_LOOSE_REPOSE * distance))
                    .min(source_loose * 0.125);
                let rock_moved = rock_rate * (fall - source_rock_limit * distance).max(0.0);
                let moved = loose_moved.max(rock_moved);
                if center_is_source {
                    height_change -= moved;
                    loose_change -= loose_moved;
                } else {
                    height_change += moved;
                    loose_change += moved;
                }
            }
            let produced = weathering_reference(
                center_loose,
                center.resistance.clamp(0.0, 1.0),
                center.area,
                steepest_rock,
                rate,
            );
            next[z * width + x] = ThermalCell {
                bed: center.bed + height_change,
                loose: (center_loose + loose_change).max(0.0) + produced,
                ..center
            };
        }
    }
    next
}

/// The shared geology's resistance with both rock bodies at their mean
/// (0.5), so only the bedding varies: resistant beds every 19 m of
/// bedrock elevation, dipping along x as the shared block dips them.
fn bedding_resistance(bedrock: f32, x: f32) -> f32 {
    let bed_coordinate = (bedrock + x * 0.017) / 19.0;
    let bedding = smoothstep(
        0.35,
        0.85,
        0.5 + 0.5 * (bed_coordinate * std::f32::consts::TAU).sin(),
    );
    smoothstep(0.25, 0.82, 0.5 * 0.62 + 0.5 * 0.24 + bedding * 0.14)
}

/// Weathering only turns bedrock into cover where it stands, after the
/// exchange, so the thermal pass still moves bed material without
/// creating or losing any: the summed bed is unchanged while the cover
/// grows by what weathering produced.
#[test]
fn thermal_reference_conserves_bed_material_with_weathering() {
    let (width, depth) = (48, 40);
    let ground = rough_ground(width.max(depth));
    let mut cells: Vec<ThermalCell> = (0..width * depth)
        .map(|cell| {
            let (x, z) = (cell % width, cell / width);
            // Steep, rough relief: 3x the rough ground's relief, cells 4 m
            // apart, standing well above zero so the bed's sum is too.
            let bed = 200.0 + ground[z * width.max(depth) + x] * 3.0;
            ThermalCell {
                bed,
                loose: ((x * 7 + z * 3) % 5) as f32 * 0.08,
                area: ((x * 13 + z * 5) % 90) as f32,
                resistance: ((x * 11 + z * 17) % 10) as f32 * 0.1,
            }
        })
        .collect();
    let bed_total = |cells: &[ThermalCell]| cells.iter().map(|cell| cell.bed as f64).sum::<f64>();
    let loose_total =
        |cells: &[ThermalCell]| cells.iter().map(|cell| cell.loose as f64).sum::<f64>();
    let (bed_before, loose_before) = (bed_total(&cells), loose_total(&cells));
    let mut moved = 0.0f64;
    for _ in 0..60 {
        let next = thermal_reference_step(&cells, width, depth, false, 0.015);
        moved += cells
            .iter()
            .zip(&next)
            .map(|(a, b)| (a.bed - b.bed).abs() as f64)
            .sum::<f64>();
        cells = next;
    }
    let drift = (bed_total(&cells) - bed_before).abs() / bed_before;
    assert!(drift <= 1e-6, "the bed drifted by {drift:e} of its total");
    assert!(moved > 10.0, "the relief sheds material ({moved} m moved)");
    assert!(loose_total(&cells) > loose_before, "weathering adds cover");
}

/// Weathering stops under a thick enough cover, and never acts in
/// channels (from 60 cells of catchment), on the most resistant rock, or
/// on slopes gentler than the weathering slope; at rate 0 it is nothing.
#[test]
fn weathering_is_shielded_by_cover_and_absent_in_channels_and_hard_rock() {
    let rate = 0.015;
    for slope in [0.5, 0.67, 0.9, 1.2, 1.8] {
        let mut previous = f32::INFINITY;
        for step in 0..=40 {
            let produced = weathering_reference(step as f32 * 0.025, 0.2, 1.0, slope, rate);
            assert!(
                produced <= previous,
                "more cover weathers faster at slope {slope}"
            );
            previous = produced;
        }
        assert!(
            previous < rate * 0.04,
            "a metre of cover shields slope {slope}: {previous}"
        );
        assert!(
            weathering_reference(0.0, 0.2, 1.0, slope, rate) > 0.0,
            "bare weak rock at {slope}"
        );
        assert_eq!(
            weathering_reference(0.0, 0.75, 1.0, slope, rate),
            0.0,
            "resistant rock"
        );
        assert_eq!(
            weathering_reference(0.0, 0.2, 60.0, slope, rate),
            0.0,
            "a channel"
        );
        assert_eq!(
            weathering_reference(0.0, 0.2, 1.0, slope, 0.0),
            0.0,
            "rate 0"
        );
    }
    for slope in [0.0, 0.2, EROSION_WEATHERING_SLOPE] {
        assert_eq!(
            weathering_reference(0.0, 0.0, 1.0, slope, rate),
            0.0,
            "gentle slope {slope}"
        );
    }
    // Steep faces shed their debris, so a thinner skin shields them.
    let gentle = weathering_reference(0.1, 0.2, 1.0, 0.6, rate)
        / weathering_reference(0.0, 0.2, 1.0, 0.6, rate);
    let steep = weathering_reference(0.1, 0.2, 1.0, 1.2, rate)
        / weathering_reference(0.0, 0.2, 1.0, 1.2, rate);
    assert!(
        steep < gentle * 0.5,
        "10 cm shields a face more than a slope: {steep} vs {gentle}"
    );
}

/// An escarpment 60 m high at tan 1.2 (50 degrees) in weak rock, between a
/// gentle footslope and a plateau, never reaches its rockfall limit, so
/// without weathering it barely changes in 140 steps. Weathering lowers
/// its bedrock by decimetres, keeps the face nearly bare while the debris
/// sheds, and builds a talus apron of metres at its foot.
#[test]
fn weathered_escarpments_retreat_and_build_aprons() {
    let (width, depth) = (120, 8);
    let face_tan = 1.2f32;
    let height_at = |x: f32| {
        if x < 120.0 {
            0.1 * x
        } else if x <= 120.0 + 60.0 / face_tan {
            12.0 + face_tan * (x - 120.0)
        } else {
            72.0 + 0.05 * (x - 120.0 - 60.0 / face_tan)
        }
    };
    let centre = |x: usize| (x as f32 + 0.5) * EROSION_CELL_SIZE;
    let run = |rate: f32| {
        // The initial cover as erosion-init stamps it, at the mean soil patch.
        let heights: Vec<f32> = (0..width).map(|x| height_at(centre(x))).collect();
        let row: Vec<ThermalCell> = (0..width)
            .map(|x| {
                let (left, right) = (x.saturating_sub(1), (x + 1).min(width - 1));
                let steepness = (heights[right] - heights[left]).abs()
                    / ((right - left) as f32 * EROSION_CELL_SIZE);
                let resistance = bedding_resistance(heights[x], centre(x));
                ThermalCell {
                    bed: heights[x],
                    loose: 1.35
                        * (1.0 - smoothstep(0.42, 0.78, steepness))
                        * lerp(1.0, 0.45, resistance),
                    area: 1.0,
                    resistance,
                }
            })
            .collect();
        let start: Vec<ThermalCell> = (0..depth).flat_map(|_| row.iter().copied()).collect();
        let mut cells = start.clone();
        for _ in 0..ErosionSettings::default().iterations {
            // The terrain pass leaves the resistance of the bedrock it exposes.
            for (cell, state) in cells.iter_mut().enumerate() {
                state.resistance =
                    bedding_resistance(state.bed - state.loose, centre(cell % width));
            }
            cells = thermal_reference_step(&cells, width, depth, true, rate);
        }
        let middle = depth / 2 * width;
        let face: Vec<usize> = (0..width)
            .filter(|x| centre(*x) > 125.0 && centre(*x) < 120.0 + 60.0 / face_tan - 5.0)
            .collect();
        let lowering = face
            .iter()
            .map(|x| {
                let (before, after) = (start[middle + x], cells[middle + x]);
                (before.bed - before.loose) - (after.bed - after.loose)
            })
            .sum::<f32>()
            / face.len() as f32;
        let cover = face.iter().map(|x| cells[middle + x].loose).sum::<f32>() / face.len() as f32;
        let apron = (0..width)
            .filter(|x| centre(*x) > 90.0 && centre(*x) <= 125.0)
            .map(|x| cells[middle + x].bed - start[middle + x].bed)
            .fold(f32::NEG_INFINITY, f32::max);
        (lowering, cover, apron)
    };
    let (still_lowering, _, still_apron) = run(0.0);
    let (lowering, cover, apron) = run(0.015);
    eprintln!(
        "the face retreats {lowering:.3} m under {cover:.3} m of cover above a \
               {apron:.2} m apron; without weathering {still_lowering:.3} m and {still_apron:.2} m"
    );
    assert!(
        still_lowering.abs() < 0.05 && still_apron < 0.6,
        "without weathering the face stands: {still_lowering} m lower, {still_apron} m apron"
    );
    assert!(
        (0.15..=0.45).contains(&lowering),
        "the face retreats {lowering} m"
    );
    assert!(
        cover <= 0.2,
        "the face stays nearly bare, {cover} m of cover"
    );
    assert!(apron >= 1.5, "an apron of {apron} m builds at its foot");
}

/// Waits for the tile preparation's work in flight, if any, to finish.
fn wait_for_preparation(preparation: &TilePreparation) {
    let started = std::time::Instant::now();
    while preparation
        .pending
        .as_ref()
        .is_some_and(|pending| !pending.task.is_finished())
    {
        assert!(
            started.elapsed() < std::time::Duration::from_secs(300),
            "the preparation never finished"
        );
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
}

/// While one tile simulates, the next one's inputs build on the task
/// pool, so it begins the very frame the first one finalizes, with the
/// inputs an inline preparation gives.
#[test]
fn the_next_tile_is_prepared_while_one_simulates() {
    AsyncComputeTaskPool::get_or_init(bevy::tasks::TaskPool::new);
    let noise = Arc::new(NoiseField::new());
    let mut preparation = TilePreparation::new(noise.clone());
    let mut cache = ErosionCache::default();
    let bridge = ErosionBridge::default();
    let settings = ErosionSettings::default();
    let frame = |cache: &mut ErosionCache, preparation: &mut TilePreparation| {
        update_erosion_cache(
            cache,
            &bridge,
            &noise,
            Some(preparation),
            &settings,
            [10.0, 0.0, 10.0],
            EROSION_ITERATIONS_PER_FRAME,
            false,
            1.0 / 60.0,
        )
    };

    // Nothing simulates yet, so the first tile waits for its own inputs.
    while !frame(&mut cache, &mut preparation) {
        wait_for_preparation(&preparation);
    }
    let first = cache.active_tile;
    let next = preparation
        .pending
        .as_ref()
        .expect("the next tile is being prepared")
        .key;
    assert_ne!(next, first);
    assert_eq!(cache.tiles[&next].state, ErosionTileState::Queued);

    // Its inputs are ready long before the first tile finishes, and are
    // held until it does.
    wait_for_preparation(&preparation);
    assert!(!frame(&mut cache, &mut preparation));
    assert_eq!(cache.active_tile, first);

    // The first tile's finalize lands, as `apply_erosion_events` applies it.
    cache.tiles.get_mut(&first).expect("the first tile").state = ErosionTileState::Ready;
    cache.has_active_tile = false;
    bridge.take_commands();
    assert!(
        frame(&mut cache, &mut preparation),
        "the next tile begins in the same frame"
    );
    assert_eq!(cache.active_tile, next);
    let init = bridge
        .take_commands()
        .and_then(|commands| commands.init)
        .expect("an init");
    let inline = prepare_tile(&noise, None, next, 1);
    assert!(bits(&init.base_height) == bits(&inline.base_height));
    assert!(bits(&init.drainage_area) == bits(&inline.drainage_area));
    assert!(bits(&init.river) == bits(&inline.river));
}

/// A prefetch carves the rivers in force when it began. If the network
/// is replaced while it builds or waits (a region switch), the work is
/// dropped, never begun, and the tile's inputs are built again over the
/// network now in force.
#[test]
fn a_prefetch_built_for_other_rivers_is_dropped() {
    AsyncComputeTaskPool::get_or_init(bevy::tasks::TaskPool::new);
    let noise = Arc::new(NoiseField::new());
    let mut preparation = TilePreparation::new(noise);
    let mut cache = ErosionCache::default();
    ensure_erosion_cache(&mut cache, tile(0, 0));
    let next = tile(1, 0);
    preparation.start(&cache, next);
    wait_for_preparation(&preparation);
    let pending = preparation.pending.as_mut().expect("the prefetch");
    assert!(
        pending.rivers.is_none() && pending.carves(&None),
        "built without rivers"
    );

    // Stand in for a network since replaced and dropped: the finished
    // work no longer carves the rivers in force.
    pending.rivers = Some(Weak::new());
    assert!(!pending.carves(&cache.rivers));
    assert!(
        preparation.poll(&cache, next).is_none(),
        "stale inputs are never handed out"
    );
    let pending = preparation
        .pending
        .as_ref()
        .expect("the inputs are built again");
    assert!(pending.carves(&cache.rivers));
    wait_for_preparation(&preparation);
    let prepared = preparation.poll(&cache, next).expect("the rebuilt inputs");
    assert_eq!(prepared.key, next);
}

/// The prefetch follows the queue's best tile: when the player moves and
/// another tile becomes best, the work for the old one is dropped rather
/// than begun ahead of it, and work for the best tile is never restarted.
#[test]
fn a_prefetch_follows_the_best_tile() {
    AsyncComputeTaskPool::get_or_init(bevy::tasks::TaskPool::new);
    let noise = Arc::new(NoiseField::new());
    let mut preparation = TilePreparation::new(noise);
    let mut cache = ErosionCache::default();
    ensure_erosion_cache(&mut cache, tile(0, 0));
    let (chosen, best) = (tile(1, 0), tile(0, 1));
    assert_eq!(cache.tiles[&chosen].state, ErosionTileState::Queued);
    assert_eq!(cache.tiles[&best].state, ErosionTileState::Queued);

    // Built for the tile that was best when the active one began.
    preparation.start(&cache, chosen);
    wait_for_preparation(&preparation);
    preparation.start(&cache, chosen);
    let held = preparation
        .poll(&cache, chosen)
        .expect("finished work is kept, not restarted");
    assert_eq!(held.key, chosen);

    // The player moves on while it waits: the next frame's best tile
    // replaces it, whether the work is in flight or already built.
    preparation.start(&cache, chosen);
    preparation.start(&cache, best);
    assert_eq!(
        preparation.pending.as_ref().expect("the prefetch").key,
        best
    );
    preparation.start(&cache, chosen);
    wait_for_preparation(&preparation);
    assert!(
        preparation.poll(&cache, best).is_none(),
        "the old best tile is never begun"
    );
    assert_eq!(
        preparation
            .pending
            .as_ref()
            .expect("the best tile's inputs")
            .key,
        best
    );
    wait_for_preparation(&preparation);
    let prepared = preparation
        .poll(&cache, best)
        .expect("the best tile's inputs");
    assert_eq!(prepared.key, best);
    assert_eq!(
        cache.tiles[&chosen].state,
        ErosionTileState::Queued,
        "still queued for later"
    );
}

/// The ordinary path: a frame that is consumed leaves nothing behind.
#[test]
fn take_commands_empties_the_bridge() {
    let bridge = ErosionBridge::default();
    bridge.set_frame(frame_for([0.0, 0.0], 6, false), None);
    assert!(bridge.take_commands().is_some());
    assert!(bridge.take_commands().is_none());
}

/// A render side that stops draining must not grow the bridge unbounded:
/// the oldest patch is dropped past `EROSION_PENDING_PATCH_CAP`, keeping
/// the newest ones (which describe tiles the queue reached most recently).
#[test]
fn pending_patches_are_capped() {
    let bridge = ErosionBridge::default();
    let cap = forest_core::constants::EROSION_PENDING_PATCH_CAP;
    for index in 0..(cap as i64 + 8) {
        bridge.push_patches(
            tile(index, 0),
            SimulationId::default(),
            vec![index as f32],
            vec![index as f32],
        );
    }
    let drained = bridge.drain_patches();
    assert_eq!(drained.len(), cap, "the cap truncates the deque");
    // The front (oldest) side is what was dropped: survivors are the last
    // `cap` tiles pushed.
    for (slot, patch) in drained.iter().enumerate() {
        assert_eq!(patch.key.x, index_surviving(cap, slot));
    }
}

/// The key at drained position `slot` in the cap test above.
fn index_surviving(cap: usize, slot: usize) -> i64 {
    (cap as i64 + 8) - cap as i64 + slot as i64
}

/// Beginning a tile arms the watchdog: `apply_erosion_events` counts one
/// frame per call with an active tile and no delivery, and requeues the
/// tile at `EROSION_WORKER_WATCHDOG_FRAMES` (a stalled worker's tile
/// re-begins); the third consecutive strike gives up on it.
#[test]
fn watchdog_requeues_an_undelivered_tile_and_the_third_strike_fails_it() {
    let mut cache = ErosionCache::default();
    let center = tile(3, 7);
    ensure_erosion_cache(&mut cache, center);
    assert!(
        cache.tiles.contains_key(&center),
        "the centre tile is registered"
    );

    let bridge = ErosionBridge::default();
    let prepared = PreparedTile {
        key: center,
        base_height: vec![0.0; EROSION_RESOLUTION * EROSION_RESOLUTION],
        drainage_area: vec![0.0; EROSION_RESOLUTION * EROSION_RESOLUTION],
        river: vec![0.0; EROSION_RESOLUTION * EROSION_RESOLUTION * 4],
    };

    let begin = |cache: &mut ErosionCache| {
        let mut commands = ErosionFrameCommands {
            simulation_id: SimulationId::default(),
            sim_min: [0.0, 0.0],
            init: None,
            iterate: None,
            finalize: false,
        };
        let prepared_tile = PreparedTile {
            key: prepared.key,
            base_height: prepared.base_height.clone(),
            drainage_area: prepared.drainage_area.clone(),
            river: prepared.river.clone(),
        };
        begin_prepared_tile(cache, &mut commands, prepared_tile);
        assert!(commands.init.is_some(), "begin carries an init command");
    };

    let strike = |cache: &mut ErosionCache| {
        let mut frames = 0;
        while cache.has_active_tile {
            apply_erosion_events(cache, &bridge);
            frames += 1;
        }
        frames
    };

    begin(&mut cache);
    let frames = strike(&mut cache);
    assert_eq!(
        frames,
        forest_core::constants::EROSION_WORKER_WATCHDOG_FRAMES,
        "strike one fires after exactly the watchdog interval"
    );
    let entry = cache.tiles.get(&center).expect("tile entry");
    assert_eq!(entry.failed_readbacks, 1);
    assert_eq!(entry.state, ErosionTileState::Queued, "strike one requeues");

    begin(&mut cache);
    strike(&mut cache);
    begin(&mut cache);
    strike(&mut cache);
    let entry = cache.tiles.get(&center).expect("tile entry");
    assert_eq!(entry.failed_readbacks, 3);
    assert_eq!(
        entry.state,
        ErosionTileState::Failed,
        "three strikes retire the tile"
    );
    assert!(!cache.has_active_tile);
}

/// The failed tile clears a pending immediate reveal: `force_reveal` must
/// not carry to whichever tile finalizes next.
#[test]
fn a_failed_strike_clears_force_reveal() {
    let mut cache = ErosionCache::default();
    let key = tile(1, 2);
    ensure_erosion_cache(&mut cache, key);
    let mut commands = ErosionFrameCommands {
        simulation_id: SimulationId::default(),
        sim_min: [0.0, 0.0],
        init: None,
        iterate: None,
        finalize: false,
    };
    let prepared_tile = PreparedTile {
        key,
        base_height: vec![0.0],
        drainage_area: vec![0.0],
        river: vec![0.0],
    };
    begin_prepared_tile(&mut cache, &mut commands, prepared_tile);
    assert!(cache.has_active_tile);
    // force_reveal is set by the prewarm/advance path when a finalize is
    // requested; the failure below must consume it.
    cache.force_reveal = Some(key);

    let bridge = ErosionBridge::default();
    bridge.push_event(ErosionEvent::ReadbackFailed {
        key,
        simulation_id: cache.tiles[&key].simulation_id,
    });
    apply_erosion_events(&mut cache, &bridge);

    assert_eq!(
        cache.force_reveal, None,
        "failed strikes consume force_reveal"
    );
}

/// A readback failure requeues the tile for strikes one and two and
/// retires it on strike three, mirroring `mark_tile_failed`'s contract.
#[test]
fn map_failure_requeues_then_fails_a_tile() {
    let mut cache = ErosionCache::default();
    let key = tile(1, 2);
    ensure_erosion_cache(&mut cache, key);
    let mut commands = ErosionFrameCommands {
        simulation_id: SimulationId::default(),
        sim_min: [0.0, 0.0],
        init: None,
        iterate: None,
        finalize: false,
    };
    let prepared_tile = PreparedTile {
        key,
        base_height: vec![0.0],
        drainage_area: vec![0.0],
        river: vec![0.0],
    };
    begin_prepared_tile(&mut cache, &mut commands, prepared_tile);

    let bridge = ErosionBridge::default();
    for attempt in 1..=3 {
        // Re-begin only from strike two on: the first strike hits an
        // active tile.
        if attempt > 1 {
            let mut commands = ErosionFrameCommands {
                simulation_id: SimulationId::default(),
                sim_min: [0.0, 0.0],
                init: None,
                iterate: None,
                finalize: false,
            };
            let prepared_tile = PreparedTile {
                key,
                base_height: vec![0.0],
                drainage_area: vec![0.0],
                river: vec![0.0],
            };
            begin_prepared_tile(&mut cache, &mut commands, prepared_tile);
        }
        bridge.push_event(ErosionEvent::ReadbackFailed {
            key,
            simulation_id: cache.tiles[&key].simulation_id,
        });
        apply_erosion_events(&mut cache, &bridge);
        let entry = cache.tiles.get(&key).expect("tile entry");
        assert_eq!(entry.failed_readbacks, attempt);
        assert!(
            !cache.has_active_tile,
            "the failing slot releases immediately"
        );
    }
    let entry = cache.tiles.get(&key).expect("tile entry");
    assert_eq!(
        entry.state,
        ErosionTileState::Failed,
        "strike three retires"
    );
    // A retired tile no longer re-begin: begin_prepared_tile would still
    // flip it to Simulating (the scheduler never offers failed tiles), but
    // the state read here is what apply_erosion_events converged to.
}

/// A dead worker kills the simulation loudly (`apply_erosion_events`
/// panics on the bridge's death notice) — the render thread failed the
/// same way, so the port's failure contract matches.
#[test]
#[should_panic(expected = "worker thread died")]
fn worker_death_panics_apply_erosion_events() {
    let mut cache = ErosionCache::default();
    let bridge = ErosionBridge::default();
    bridge.set_worker_dead("device lost in the drill".to_string());
    apply_erosion_events(&mut cache, &bridge);
}

/// The WGSL sources publish once and read back exactly once, and the
/// readiness flag tracks them for the render node's spawn gate; a clone
/// of the bridge (the render app's copy) sees the same publication.
#[test]
fn shader_sources_publish_once_and_cross_clone() {
    let bridge = ErosionBridge::default();
    assert!(!bridge.shader_sources_ready());
    let sources = ErosionShaderSources {
        init: b"fn init() {}".to_vec(),
        flux: b"fn flux() {}".to_vec(),
        water: b"fn water() {}".to_vec(),
        terrain: b"fn terrain() {}".to_vec(),
        thermal: b"fn thermal() {}".to_vec(),
    };
    let published = sources.clone();
    // SAFETY: these test shaders perform no memory accesses.
    unsafe { bridge.publish_shader_sources(published) };
    assert!(
        bridge.shader_sources_ready(),
        "the render node's clone gate"
    );

    let taken = bridge
        .clone()
        .take_shader_sources()
        .expect("publish survives");
    for (taken_field, source_field) in [
        (&taken.init, &sources.init),
        (&taken.flux, &sources.flux),
        (&taken.water, &sources.water),
        (&taken.terrain, &sources.terrain),
        (&taken.thermal, &sources.thermal),
    ] {
        assert_eq!(taken_field, source_field, "sources arrive byte-identical");
    }
    assert!(!bridge.shader_sources_ready(), "consumed once");
    assert!(bridge.take_shader_sources().is_none(), "take-once");
}

fn begin_test_tile(cache: &mut ErosionCache, key: TileKey) -> ErosionFrameCommands {
    let mut commands = frame_for(tile_simulation_minimum(key), 0, false);
    let prepared = PreparedTile {
        key,
        base_height: Vec::new(),
        drainage_area: Vec::new(),
        river: Vec::new(),
    };
    begin_prepared_tile(cache, &mut commands, prepared);
    commands
}

fn finalized_test_event(key: TileKey, simulation_id: SimulationId, delta: f32) -> ErosionEvent {
    let tile = FinalizedTile {
        key,
        cpu_delta: vec![delta; EROSION_OUTPUT_RESOLUTION.pow(2)],
        cpu_concavity: vec![0.0; EROSION_OUTPUT_RESOLUTION.pow(2)],
        stats: TileDiagnostics::default(),
    };
    ErosionEvent::TileFinalized {
        simulation_id,
        tile,
    }
}

#[test]
fn pending_commands_do_not_merge_across_attempts_of_the_same_tile() {
    let bridge = ErosionBridge::default();
    let mut earlier = frame_for([0.0, 0.0], 12, true);
    earlier.simulation_id = SimulationId::next();
    let mut replacement = frame_for([0.0, 0.0], 6, false);
    replacement.simulation_id = SimulationId::next();
    bridge.set_frame(earlier, None);
    bridge.set_frame(replacement, None);

    let taken = bridge.take_commands().expect("the replacement is pending");
    assert!(!taken.finalize, "the earlier finalize must not carry over");
    assert_eq!(taken.iterate.expect("replacement iterations").count, 6);
}

#[test]
fn reset_rejects_results_from_an_earlier_attempt_of_the_same_tile() {
    let key = tile(0, 0);
    let mut cache = ErosionCache::default();
    let bridge = ErosionBridge::default();
    ensure_erosion_cache(&mut cache, key);
    let earlier = begin_test_tile(&mut cache, key).simulation_id;
    reset_erosion_cache(&mut cache);
    ensure_erosion_cache(&mut cache, key);
    let replacement = begin_test_tile(&mut cache, key).simulation_id;
    assert_ne!(earlier, replacement);

    bridge.push_event(finalized_test_event(key, earlier, 99.0));
    apply_erosion_events(&mut cache, &bridge);
    assert!(cache.has_active_tile, "the replacement must keep running");
    assert_eq!(cache.tiles[&key].state, ErosionTileState::Simulating);
    assert!(cache.tiles[&key].cpu_delta.is_empty());

    bridge.push_event(finalized_test_event(key, replacement, 2.0));
    apply_erosion_events(&mut cache, &bridge);
    assert!(!cache.has_active_tile);
    assert_eq!(cache.tiles[&key].state, ErosionTileState::Ready);
    assert!(
        cache.tiles[&key]
            .cpu_delta
            .iter()
            .all(|delta| *delta == 2.0)
    );
}

#[test]
fn stale_events_do_not_release_a_different_active_tile() {
    let earlier_key = tile(0, 0);
    let active_key = tile(1, 0);
    let mut cache = ErosionCache::default();
    let bridge = ErosionBridge::default();
    ensure_erosion_cache(&mut cache, earlier_key);
    let earlier_id = begin_test_tile(&mut cache, earlier_key).simulation_id;
    reset_erosion_cache(&mut cache);
    ensure_erosion_cache(&mut cache, active_key);
    begin_test_tile(&mut cache, active_key);

    bridge.push_event(finalized_test_event(earlier_key, earlier_id, 99.0));
    bridge.push_event(ErosionEvent::ReadbackFailed {
        key: earlier_key,
        simulation_id: earlier_id,
    });
    apply_erosion_events(&mut cache, &bridge);

    assert!(cache.has_active_tile);
    assert_eq!(cache.active_tile, active_key);
    assert_eq!(cache.tiles[&active_key].state, ErosionTileState::Simulating);
    assert_eq!(cache.tiles[&earlier_key].state, ErosionTileState::Queued);
    assert_eq!(cache.tiles[&earlier_key].failed_readbacks, 0);
}

#[test]
fn retry_rejects_late_completions_and_failures_from_its_previous_attempt() {
    let key = tile(0, 0);
    let mut cache = ErosionCache::default();
    let bridge = ErosionBridge::default();
    ensure_erosion_cache(&mut cache, key);
    let earlier_id = begin_test_tile(&mut cache, key).simulation_id;
    bridge.push_event(ErosionEvent::ReadbackFailed {
        key,
        simulation_id: earlier_id,
    });
    apply_erosion_events(&mut cache, &bridge);
    let retry_id = begin_test_tile(&mut cache, key).simulation_id;
    assert_ne!(earlier_id, retry_id);

    bridge.push_event(finalized_test_event(key, earlier_id, 99.0));
    bridge.push_event(ErosionEvent::ReadbackFailed {
        key,
        simulation_id: earlier_id,
    });
    apply_erosion_events(&mut cache, &bridge);
    assert!(cache.has_active_tile);
    assert_eq!(cache.tiles[&key].failed_readbacks, 1);
    assert_eq!(cache.tiles[&key].state, ErosionTileState::Simulating);

    bridge.push_event(finalized_test_event(key, retry_id, 3.0));
    apply_erosion_events(&mut cache, &bridge);
    assert_eq!(cache.tiles[&key].state, ErosionTileState::Ready);
}

#[test]
fn lookup_snapshot_keeps_attempts_with_their_allocations_and_origin() {
    let key = tile(4, -3);
    let mut cache = ErosionCache::default();
    ensure_erosion_cache(&mut cache, key);
    let commands = begin_test_tile(&mut cache, key);
    cache.tiles.get_mut(&key).expect("registered tile").state = ErosionTileState::Ready;

    let snapshot = erosion_lookup_snapshot(&cache);
    let relative_x = (key.x - snapshot.minimum.x) as usize;
    let relative_z = (key.z - snapshot.minimum.z) as usize;
    let cell = relative_z * EROSION_LOOKUP_DIAMETER + relative_x;
    assert_eq!(snapshot.minimum, cache.lookup_minimum);
    assert_eq!(snapshot.records[cell * 4 + 3], 1.0);
    assert_eq!(snapshot.simulation_ids[cell], commands.simulation_id);
}
