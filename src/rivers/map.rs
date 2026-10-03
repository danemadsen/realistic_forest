//! `--river-map out.png`: the river network from above, shaded relief and
//! all, with the statistics that matter for tuning it, without a window.

use super::carve::Envelope;
use super::network::{self, RiverEnd, RiverNetwork};
use crate::constants::SEA_LEVEL;
use crate::noise::{NoiseField, base_height};

/// Carved base height (no erosion) at a point.
pub fn carved_height(noise: &NoiseField, network: &RiverNetwork, x: f32, z: f32) -> (f32, Envelope) {
    let envelope = network.envelope(x, z);
    (envelope.clamp(base_height(noise, x, z)), envelope)
}

pub fn render(noise: &NoiseField, network: &RiverNetwork, centre: [f64; 2], extent: f64, pixels: usize) -> image::RgbImage {
    let metres_per_pixel = extent / pixels as f64;
    let origin = [centre[0] - extent * 0.5, centre[1] - extent * 0.5];
    let rows: Vec<Vec<[u8; 3]>> = {
        let next = std::sync::atomic::AtomicUsize::new(0);
        let results = std::sync::Mutex::new(vec![Vec::new(); pixels]);
        let workers = std::thread::available_parallelism().map_or(4, |n| n.get());
        std::thread::scope(|scope| {
            for _ in 0..workers {
                scope.spawn(|| loop {
                    let row = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if row >= pixels {
                        break;
                    }
                    let z = (origin[1] + (row as f64 + 0.5) * metres_per_pixel) as f32;
                    let mut out = Vec::with_capacity(pixels);
                    for column in 0..pixels {
                        let x = (origin[0] + (column as f64 + 0.5) * metres_per_pixel) as f32;
                        let step = metres_per_pixel as f32;
                        let (h, envelope) = carved_height(noise, network, x, z);
                        let (hx, _) = carved_height(noise, network, x + step, z);
                        let (hz, _) = carved_height(noise, network, x, z + step);
                        let normal = [-(hx - h) / step, 1.0, -(hz - h) / step];
                        let length = (normal[0] * normal[0] + 1.0 + normal[2] * normal[2]).sqrt();
                        let light = [-0.5f32, 0.7, -0.5];
                        let shade = ((normal[0] * light[0] + normal[1] * light[1] + normal[2] * light[2]) / length / 0.995)
                            .clamp(0.15, 1.0);
                        let colour: [f32; 3] = if h < SEA_LEVEL {
                            let depth = (-h / 60.0).min(1.0);
                            [30.0 - 20.0 * depth, 90.0 - 50.0 * depth, 150.0 - 60.0 * depth]
                        } else if h < envelope.lake {
                            let depth = ((envelope.lake - h) / 12.0).min(1.0);
                            [50.0 - 25.0 * depth, 120.0 - 50.0 * depth, 190.0 - 40.0 * depth]
                        } else if envelope.bank_distance < 0.0 && h < envelope.water {
                            if envelope.turbulence > 0.6 {
                                [200.0, 230.0, 245.0]
                            } else {
                                [40.0, 110.0, 200.0]
                            }
                        } else {
                            let t = (h / 160.0).clamp(0.0, 1.0);
                            let lowland = [110.0, 140.0, 80.0];
                            let upland = [150.0, 135.0, 110.0];
                            let peak = [235.0, 235.0, 235.0];
                            if t < 0.6 {
                                let u = t / 0.6;
                                [
                                    lowland[0] + (upland[0] - lowland[0]) * u,
                                    lowland[1] + (upland[1] - lowland[1]) * u,
                                    lowland[2] + (upland[2] - lowland[2]) * u,
                                ]
                            } else {
                                let u = (t - 0.6) / 0.4;
                                [
                                    upland[0] + (peak[0] - upland[0]) * u,
                                    upland[1] + (peak[1] - upland[1]) * u,
                                    upland[2] + (peak[2] - upland[2]) * u,
                                ]
                            }
                        };
                        let lit = if h < SEA_LEVEL { 1.0 } else { shade };
                        out.push([
                            (colour[0] * lit) as u8,
                            (colour[1] * lit) as u8,
                            (colour[2] * lit) as u8,
                        ]);
                    }
                    results.lock().unwrap()[row] = out;
                });
            }
        });
        results.into_inner().unwrap()
    };
    let mut image = image::RgbImage::new(pixels as u32, pixels as u32);
    for (row, values) in rows.iter().enumerate() {
        for (column, value) in values.iter().enumerate() {
            image.put_pixel(column as u32, row as u32, image::Rgb(*value));
        }
    }
    // Centrelines too narrow to show as water, so every creek is visible.
    let to_pixel = |p: [f32; 2]| {
        (
            (p[0] as f64 - origin[0]) / metres_per_pixel,
            (p[1] as f64 - origin[1]) / metres_per_pixel,
        )
    };
    let mut plot = |x: f64, y: f64, colour: [u8; 3]| {
        if x >= 0.0 && y >= 0.0 && (x as usize) < pixels && (y as usize) < pixels {
            image.put_pixel(x as u32, y as u32, image::Rgb(colour));
        }
    };
    for river in &network.rivers {
        for pair in river.nodes.windows(2) {
            let (ax, ay) = to_pixel(pair[0].position);
            let (bx, by) = to_pixel(pair[1].position);
            let steps = ((bx - ax).abs().max((by - ay).abs()).ceil() as usize).max(1);
            for k in 0..=steps {
                let t = k as f64 / steps as f64;
                plot(ax + (bx - ax) * t, ay + (by - ay) * t, [20, 70, 220]);
            }
        }
    }
    image
}

pub fn report(network: &RiverNetwork) -> Vec<String> {
    let mut lines = Vec::new();
    let rivers = &network.rivers;
    let sea = rivers.iter().filter(|r| r.end == RiverEnd::Sea).count();
    let confluent = rivers.iter().filter(|r| matches!(r.end, RiverEnd::Confluence(..))).count();
    lines.push(format!(
        "rivers: {} ({} reach the sea, {} join another, {} leave the domain), {:.1} km of channel",
        rivers.len(),
        sea,
        confluent,
        rivers.len() - sea - confluent,
        network.total_length_km()
    ));
    let nodes: Vec<&network::RiverNode> = rivers.iter().flat_map(|r| &r.nodes).collect();
    let largest = nodes.iter().map(|n| n.area).fold(0.0f32, f32::max);
    let widest = nodes.iter().map(|n| n.half_width * 2.0).fold(0.0f32, f32::max);
    let fastest = nodes.iter().map(|n| n.speed).fold(0.0f32, f32::max);
    lines.push(format!(
        "largest catchment {largest:.1} km², widest channel {widest:.1} m, fastest flow {fastest:.2} m/s, {} carve segments",
        network.segments.len()
    ));
    // Width and depth by slope: steep channels narrow and deep, lowland
    // ones wide and shallow.
    for (name, low, high) in [("lowland", 0.0f32, 0.006f32), ("moderate", 0.006, 0.03), ("steep", 0.03, 10.0)] {
        let band: Vec<&&network::RiverNode> = nodes
            .iter()
            .filter(|n| !n.lake && n.slope >= low && n.slope < high && n.area > 0.5)
            .collect();
        if band.is_empty() {
            continue;
        }
        let count = band.len() as f32;
        let width = band.iter().map(|n| n.half_width * 2.0).sum::<f32>() / count;
        let depth = band.iter().map(|n| n.depth).sum::<f32>() / count;
        let speed = band.iter().map(|n| n.speed).sum::<f32>() / count;
        lines.push(format!(
            "{name} reaches draining over 0.5 km²: {width:.1} m wide, {depth:.2} m deep, {speed:.2} m/s on average"
        ));
    }
    // Sinuosity of the larger rivers: channel length over straight distance
    // per 1 km window.
    let mut sinuosity = Vec::new();
    for river in rivers {
        let n = &river.nodes;
        let mut start = 0;
        for i in 1..n.len() {
            if n[i].along - n[start].along >= 1000.0 {
                let chord = (n[i].position[0] - n[start].position[0]).hypot(n[i].position[1] - n[start].position[1]);
                sinuosity.push(((n[i].along - n[start].along) / chord.max(1.0), n[i].slope));
                start = i;
            }
        }
    }
    let lowland: Vec<f32> = sinuosity.iter().filter(|s| s.1 < 0.006).map(|s| s.0).collect();
    let upland: Vec<f32> = sinuosity.iter().filter(|s| s.1 >= 0.02).map(|s| s.0).collect();
    let mean = |v: &[f32]| if v.is_empty() { 0.0 } else { v.iter().sum::<f32>() / v.len() as f32 };
    lines.push(format!(
        "sinuosity per km: lowland {:.2} ({} km), steep {:.2} ({} km)",
        mean(&lowland),
        lowland.len(),
        mean(&upland),
        upland.len()
    ));
    let cell_area = (network::FLOW_CELL * network::FLOW_CELL) as f32;
    let lake_areas: Vec<f32> = network.lakes.iter().map(|l| l.cells.len() as f32 * cell_area / 1.0e4).collect();
    lines.push(format!(
        "{} lakes, {:.1} ha of water, largest {:.1} ha",
        network.lakes.len(),
        lake_areas.iter().sum::<f32>(),
        lake_areas.iter().copied().fold(0.0f32, f32::max)
    ));
    lines.push(format!("built in {:.2} s", network.build_seconds));
    lines
}

/// How well the channels sit in the land they cross, over the natural
/// (uncarved, uneroded) ground: a river that follows the terrain runs on its
/// valley floor, its surface a little below the ground at its banks. Shares
/// of channel length (falls and the sea excluded) where
/// - on a gentle reach, the surface lies over 3 m under the natural ground:
///   a trench cut through a rise;
/// - the natural ground beside the channel lies under the surface, so the
///   water is held in by the carve's levee rather than by the land;
/// - lower ground lies within 25 m across: the river runs along a slope
///   above its valley's floor.
pub fn terrain_fit(noise: &NoiseField, network: &RiverNetwork) -> String {
    let mut total = 0.0f64;
    let (mut trench, mut perched, mut off_floor) = (0.0f64, 0.0f64, 0.0f64);
    // Mean depth of the surface under the natural ground: (sum, length)
    // over gentle and over steep reaches.
    let mut incision = [(0.0f64, 0.0f64); 2];
    for river in &network.rivers {
        let nodes = &river.nodes;
        let end = river.surface_end.min(nodes.len().saturating_sub(1));
        for i in 1..end {
            let node = &nodes[i];
            if node.water < SEA_LEVEL + 0.1 || node.lake {
                continue;
            }
            let (a, b) = (nodes[i - 1].position, nodes[i + 1].position);
            let length = (b[0] - a[0]).hypot(b[1] - a[1]).max(1e-3);
            let normal = [-(b[1] - a[1]) / length, (b[0] - a[0]) / length];
            let weight = (length * 0.5) as f64;
            total += weight;
            let at = |offset: f32| {
                base_height(noise, node.position[0] + normal[0] * offset, node.position[1] + normal[1] * offset)
            };
            // Steep creeks cut their pools into the slope by design.
            if node.slope < network::STEP_SLOPE && at(0.0) - node.water > 3.0 {
                trench += weight;
            }
            let band = usize::from(node.slope >= network::STEP_SLOPE);
            incision[band].0 += (at(0.0) - node.water).max(0.0) as f64 * weight;
            incision[band].1 += weight;
            let reach = [1.0f32, 2.5, 5.0];
            let spills = [-1.0f32, 1.0]
                .iter()
                .any(|&side| reach.iter().all(|&d| at(side * (node.half_width + d)) < node.water - 0.25));
            if spills {
                perched += weight;
            }
            let floor = at(0.0);
            let lower = [-25.0f32, -18.0, -12.0, -8.0, 8.0, 12.0, 18.0, 25.0]
                .iter()
                .any(|&d| d.abs() > node.half_width + 2.0 && at(d) < floor - 1.5);
            if lower {
                off_floor += weight;
            }
        }
    }
    let share = |v: f64| 100.0 * v / total.max(1.0);
    let mean = |(sum, length): (f64, f64)| sum / length.max(1.0);
    format!(
        "terrain fit over {:.1} km: {:.1}% trenched over 3 m (gentle reaches), {:.1}% held up by levees, {:.1}% above the valley floor; water {:.2} m under the ground on gentle reaches, {:.2} m on steep",
        total / 1000.0,
        share(trench),
        share(perched),
        share(off_floor),
        mean(incision[0]),
        mean(incision[1])
    )
}

/// Where a lake sheet's outer edge stands over ground (as the rivers carve
/// it) lower than the sheet there, so it would end in the air: of the
/// samples taken every half metre along every sheet's edge, how many, and
/// per lake that has any, how many and the worst gap and where.
pub struct LakeEdges {
    pub samples: usize,
    pub exposed: usize,
    pub lakes: Vec<(usize, [f32; 2], f32, f32)>,
}

pub fn lake_edges(noise: &NoiseField, network: &RiverNetwork) -> LakeEdges {
    let cell = network::FLOW_CELL as f32;
    let mut edges = LakeEdges { samples: 0, exposed: 0, lakes: Vec::new() };
    for lake in &network.lakes {
        let sheet: std::collections::HashSet<[i32; 2]> = lake.cells.iter().chain(&lake.shore).copied().collect();
        let sunk: std::collections::HashMap<[i32; 2], f32> = lake.edge.iter().copied().collect();
        let height = |corner: [i32; 2]| sunk.get(&corner).map_or(lake.level, |&h| h.min(lake.level));
        let (mut exposed, mut gap, mut at) = (0usize, 0.0f32, [0.0f32; 2]);
        for &[x, z] in &sheet {
            for (dx, dz) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                if sheet.contains(&[x + dx, z + dz]) {
                    continue;
                }
                let a = [x + i32::from(dx > 0), z + i32::from(dz > 0)];
                let b = [a[0] + dz.abs(), a[1] + dx.abs()];
                for k in 0..8 {
                    let t = (k as f32 + 0.5) / 8.0;
                    let p = [
                        (a[0] as f32 + (b[0] - a[0]) as f32 * t) * cell,
                        (a[1] as f32 + (b[1] - a[1]) as f32 * t) * cell,
                    ];
                    let water = height(a) + (height(b) - height(a)) * t;
                    let ground = network.envelope(p[0], p[1]).clamp(base_height(noise, p[0], p[1]));
                    edges.samples += 1;
                    if ground < water - 0.02 {
                        exposed += 1;
                        if water - ground > gap {
                            gap = water - ground;
                            at = p;
                        }
                    }
                }
            }
        }
        if exposed > 0 {
            edges.exposed += exposed;
            edges.lakes.push((exposed, at, gap, lake.level));
        }
    }
    edges.lakes.sort_by_key(|lake| std::cmp::Reverse(lake.0));
    edges
}

/// How well the lakes' sheets meet their shores, and the lakes worst so,
/// for framing a camera on them.
pub fn lake_fit(noise: &NoiseField, network: &RiverNetwork) -> Vec<String> {
    let edges = lake_edges(noise, network);
    let mut lines = vec![format!(
        "lake sheets: {:.2}% of their edges over lower ground; {} of {} lakes have such edges",
        100.0 * edges.exposed as f64 / edges.samples.max(1) as f64,
        edges.lakes.len(),
        network.lakes.len()
    )];
    for (count, at, gap, level) in edges.lakes.iter().take(6) {
        lines.push(format!(
            "  {count} edge samples exposed, worst {gap:.2} m at {:.1},{:.1} (water {level:.2} m)",
            at[0], at[1]
        ));
    }
    lines
}

pub fn run_map(noise: &NoiseField, path: &str, centre: [f64; 2], extent: f64) {
    let region = network::region_of(centre[0], centre[1]);
    let network = network::generate(noise, region);
    for line in report(&network) {
        println!("{line}");
    }
    println!("{}", terrain_fit(noise, &network));
    for line in lake_fit(noise, &network) {
        println!("{line}");
    }
    // The nodes nearest the map's centre, for framing a camera on them: the
    // heading is the `--camera` yaw that looks downstream.
    let mut nearest: Vec<(f32, &network::RiverNode, f32)> = network
        .rivers
        .iter()
        .flat_map(|r| {
            let nodes = &r.nodes;
            (0..nodes.len()).map(move |i| {
                let (a, b) = (&nodes[i.saturating_sub(1)], &nodes[(i + 1).min(nodes.len() - 1)]);
                let heading = (b.position[0] - a.position[0])
                    .atan2(a.position[1] - b.position[1])
                    .to_degrees()
                    .rem_euclid(360.0);
                (&nodes[i], heading)
            })
        })
        .map(|(n, heading)| ((n.position[0] - centre[0] as f32).hypot(n.position[1] - centre[1] as f32), n, heading))
        .collect();
    nearest.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut shown: Vec<[f32; 2]> = Vec::new();
    for (d, node, heading) in nearest {
        if shown.len() >= 8 {
            break;
        }
        if shown.iter().any(|p| (p[0] - node.position[0]).hypot(p[1] - node.position[1]) < 150.0) {
            continue;
        }
        shown.push(node.position);
        println!(
            "near centre ({d:.0} m): river at {:.1},{:.1} water {:.2} m, {:.1} m wide, {:.2} m deep, {:.2} m/s, slope {:.3}, heading {heading:.0}",
            node.position[0], node.position[1], node.water, node.half_width * 2.0, node.depth, node.speed, node.slope
        );
    }
    let cell = network::FLOW_CELL as f32;
    let mut lakes: Vec<(f32, [f32; 2], &network::Lake)> = network
        .lakes
        .iter()
        .map(|lake| {
            let count = lake.cells.len() as f32;
            let middle = lake.cells.iter().fold([0.0f32; 2], |sum, c| {
                [sum[0] + (c[0] as f32 + 0.5) * cell / count, sum[1] + (c[1] as f32 + 0.5) * cell / count]
            });
            ((middle[0] - centre[0] as f32).hypot(middle[1] - centre[1] as f32), middle, lake)
        })
        .collect();
    lakes.sort_by(|a, b| a.0.total_cmp(&b.0));
    for (d, middle, lake) in lakes.iter().take(3) {
        println!(
            "lake at {:.0},{:.0} ({d:.0} m away): water {:.2} m, {:.1} ha",
            middle[0],
            middle[1],
            lake.level,
            lake.cells.len() as f32 * cell * cell / 1.0e4
        );
        // Where rivers run into and out of it, and the yaw looking
        // downstream there.
        let inside: std::collections::HashSet<[i32; 2]> = lake.cells.iter().copied().collect();
        let holds = |p: [f32; 2]| inside.contains(&[(p[0] / cell).floor() as i32, (p[1] / cell).floor() as i32]);
        for river in &network.rivers {
            for pair in river.nodes.windows(2) {
                let (a, b) = (&pair[0], &pair[1]);
                if a.lake == b.lake || !(holds(a.position) || holds(b.position)) {
                    continue;
                }
                let heading = (b.position[0] - a.position[0])
                    .atan2(a.position[1] - b.position[1])
                    .to_degrees()
                    .rem_euclid(360.0);
                println!(
                    "  {} at {:.1},{:.1}, heading {heading:.0}",
                    if b.lake { "inlet" } else { "outlet" },
                    b.position[0],
                    b.position[1]
                );
            }
        }
    }
    let pixels = 2048usize.min((extent / 1.0) as usize).max(256);
    let image = render(noise, &network, centre, extent, pixels);
    match image.save(path) {
        Ok(()) => println!("river map written to {path}"),
        Err(error) => eprintln!("could not write {path}: {error}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The spawn region's rivers carve their channels: at a node the ground
    /// is held below the water, and the GPU payload finds the same segments.
    #[test]
    fn spawn_region_channels_are_carved() {
        let noise = NoiseField::new();
        let network = network::generate(&noise, [0, 0]);
        let node = network.rivers.iter().flat_map(|r| &r.nodes).find(|n| n.depth > 0.3 && n.water > 2.0 && !n.lake).unwrap();
        let (x, z) = (node.position[0], node.position[1]);
        let envelope = network.envelope(x, z);
        let carved = envelope.clamp(base_height(&noise, x, z));
        assert!(envelope.bank_distance < 0.0, "{envelope:?}");
        assert!(carved < node.water - 0.2, "carved {carved} water {}", node.water);
        let payload = network.gpu_payload();
        let words = &payload.grid_words;
        let origin = [f32::from_bits(words[0]), f32::from_bits(words[1])];
        let cell = f32::from_bits(words[2]);
        let resolution = words[3] as usize;
        let cx = ((x - origin[0]) / cell).floor() as usize;
        let cz = ((z - origin[1]) / cell).floor() as usize;
        let entry = 8 + (cz * resolution + cx) * super::super::carve::GRID_CELL_WORDS;
        let offset = words[entry] as usize;
        let count = words[entry + 1] as usize;
        assert!(count > 0);
        let mut total = super::super::carve::Envelope::NONE;
        for i in 0..count {
            let segment = &payload.segments[words[offset + i] as usize];
            super::super::carve::combine(&mut total, &super::super::carve::segment_envelope(segment, [x, z]));
        }
        assert_eq!(total.upper, envelope.upper);
    }

    /// No lake's sheet ends in the air: its outer edge runs into the ground
    /// all the way round.
    #[test]
    fn spawn_region_lake_sheets_meet_their_shores() {
        let noise = NoiseField::new();
        let network = network::generate(&noise, [0, 0]);
        assert!(!network.lakes.is_empty());
        let edges = lake_edges(&noise, &network);
        assert!(edges.samples > 1000);
        assert_eq!(edges.exposed, 0, "{:?}", &edges.lakes[..edges.lakes.len().min(4)]);
    }
}
