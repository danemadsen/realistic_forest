//! `--river-map out.png`: the river network from above, shaded relief and
//! all, with the statistics that matter for tuning it, without a window.

use super::carve::Envelope;
use super::network::{self, RiverEnd, RiverNetwork};
use super::surface::SurfaceVertex;
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
            let render_row = || loop {
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
            };
            for _ in 0..workers {
                scope.spawn(render_row);
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
    // Every gentle reach's incision with its length, for the median and the
    // deep tail.
    let mut cuts: Vec<(f32, f64)> = Vec::new();
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
            if band == 0 {
                cuts.push(((at(0.0) - node.water).max(0.0), weight));
            }
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
    cuts.sort_by(|a, b| a.0.total_cmp(&b.0));
    let cut_length: f64 = cuts.iter().map(|c| c.1).sum();
    let quantile = |q: f64| {
        let mut run = 0.0;
        cuts.iter().find(|c| {
            run += c.1;
            run >= q * cut_length
        })
        .map_or(0.0, |c| c.0)
    };
    format!(
        "terrain fit over {:.1} km: {:.1}% trenched over 3 m (gentle reaches), {:.1}% held up by levees, {:.1}% above the valley floor; water {:.2} m under the ground on gentle reaches (median {:.2}, 90th percentile {:.2}), {:.2} m on steep",
        total / 1000.0,
        share(trench),
        share(perched),
        share(off_floor),
        mean(incision[0]),
        quantile(0.5),
        quantile(0.9),
        mean(incision[1])
    )
}

/// Where a lake sheet's outer edge stands over ground (as the rivers carve
/// it), or over a river's water where its ribbon reaches over the edge,
/// lower than the sheet there, so it would end in the air: of the
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
                    // The ground, or a river's surface where its ribbon
                    // reaches over the edge (a channel meeting the lake).
                    let envelope = network.envelope(p[0], p[1]);
                    let mut cover = envelope.clamp(base_height(noise, p[0], p[1]));
                    if envelope.bank_distance < network::RIBBON_COVER * envelope.half_width {
                        cover = cover.max(envelope.water);
                    }
                    edges.samples += 1;
                    if cover < water - 0.02 {
                        exposed += 1;
                        if water - cover > gap {
                            gap = water - cover;
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

/// How cleanly the rivers meet still water:
/// - flicker: stretches shorter than 15 m of a river out of a lake between
///   two reaches in the same lake, where the river darts out and back;
/// - outlets: how far the water falls over the first 30 m below a lake, the
///   steepest such drop and where;
/// - mouths: rivers ending in the sea whose last node cannot see open sea
///   (ground a metre under the sea level) within 40 m beyond it across ground
///   no higher than the sea: a mouth into a hollow on the beach.
pub struct Junctions {
    pub flicker: usize,
    pub outlets: usize,
    pub steep_outlets: usize,
    pub steepest: (f32, [f32; 2]),
    pub mouths: usize,
    pub cut_off_mouths: Vec<[f32; 2]>,
}

pub fn junctions(noise: &NoiseField, network: &RiverNetwork) -> Junctions {
    let mut junctions = Junctions {
        flicker: 0,
        outlets: 0,
        steep_outlets: 0,
        steepest: (0.0, [0.0; 2]),
        mouths: 0,
        cut_off_mouths: Vec::new(),
    };
    for river in &network.rivers {
        let nodes = &river.nodes;
        let changes: Vec<usize> = (1..nodes.len()).filter(|&i| nodes[i].lake != nodes[i - 1].lake).collect();
        let mut previous: Option<usize> = None;
        for (i, node) in nodes.iter().enumerate().filter(|(_, n)| n.lake) {
            if let Some(p) = previous
                && p + 1 < i
                && (nodes[p].water - node.water).abs() < 0.01
                && node.along - nodes[p].along < 15.0
            {
                junctions.flicker += 1;
            }
            previous = Some(i);
        }
        for &i in changes.iter().filter(|&&i| nodes[i - 1].lake) {
            junctions.outlets += 1;
            let start = nodes[i - 1].water;
            let below = nodes[i..].iter().take_while(|n| n.along - nodes[i - 1].along <= 30.0).last();
            if let Some(below) = below {
                let drop = start - below.water;
                if drop > 1.5 {
                    junctions.steep_outlets += 1;
                }
                if drop > junctions.steepest.0 {
                    junctions.steepest = (drop, nodes[i].position);
                }
            }
        }
        if river.end == network::RiverEnd::Sea && nodes.len() >= 2 {
            junctions.mouths += 1;
            // Open sea in some direction within reach of the last node,
            // across ground (as carved) no higher than the sea.
            let b = nodes[nodes.len() - 1].position;
            let reach = 3.0 * nodes[nodes.len() - 1].half_width + 12.0;
            let ray_reaches_open_sea = |k: i32| {
                let angle = k as f32 / 24.0 * std::f32::consts::TAU;
                let direction = [angle.cos(), angle.sin()];
                let mut t = 0.0;
                while t <= reach {
                    let p = [b[0] + direction[0] * t, b[1] + direction[1] * t];
                    let ground = network.envelope(p[0], p[1]).clamp(base_height(noise, p[0], p[1]));
                    if ground > SEA_LEVEL + 0.02 {
                        return false;
                    }
                    if ground < SEA_LEVEL - 1.0 {
                        return true;
                    }
                    t += 0.5;
                }
                false
            };
            let open = (0..24).any(ray_reaches_open_sea);
            if !open {
                junctions.cut_off_mouths.push(b);
            }
        }
    }
    junctions
}

/// Where a lake's sheet meets a river's channel, how far its edge dips under
/// the river's surface there: (sunk corners in a channel, how many dip more
/// than 0.3 m, the worst dip).
pub fn mouth_dips(network: &RiverNetwork) -> (usize, usize, f32) {
    let cell = network::FLOW_CELL as f32;
    let (mut corners, mut deep, mut worst) = (0usize, 0usize, 0.0f32);
    for lake in &network.lakes {
        for &(corner, height) in &lake.edge {
            let p = [corner[0] as f32 * cell, corner[1] as f32 * cell];
            let envelope = network.envelope(p[0], p[1]);
            if envelope.bank_distance < 0.0 {
                corners += 1;
                let dip = envelope.water.min(lake.level) - height;
                deep += usize::from(dip > 0.3);
                worst = worst.max(dip);
            }
        }
    }
    (corners, deep, worst)
}

pub fn junction_fit(noise: &NoiseField, network: &RiverNetwork) -> Vec<String> {
    let j = junctions(noise, network);
    let mut lines = vec![format!(
        "junctions: {} lake flickers; {} outlets, {} falling over 1.5 m in their first 30 m (steepest {:.1} m at {:.1},{:.1}); {} of {} sea mouths cut off from the open sea",
        j.flicker,
        j.outlets,
        j.steep_outlets,
        j.steepest.0,
        j.steepest.1[0],
        j.steepest.1[1],
        j.cut_off_mouths.len(),
        j.mouths
    )];
    for p in j.cut_off_mouths.iter().take(4) {
        lines.push(format!("  cut-off mouth at {:.1},{:.1}", p[0], p[1]));
    }
    let (corners, deep, worst) = mouth_dips(network);
    lines.push(format!(
        "mouths: {corners} sunk sheet corners in channels, {deep} dipping over 0.3 m under the river (worst {worst:.2} m)"
    ));
    lines
}

/// The drawn water surfaces, bucketed for point queries: the highest
/// river ribbon and the highest lake sheet over a point.
pub struct SurfaceIndex<'a> {
    mesh: &'a super::surface::SurfaceMesh,
    /// Vertices from here on are the lakes' sheets.
    first_lake: usize,
    /// The first vertex of each lake's sheet after the first.
    lake_starts: Vec<usize>,
    buckets: std::collections::HashMap<[i32; 2], Vec<u32>>,
}

const SURFACE_BUCKET: f32 = 8.0;

impl<'a> SurfaceIndex<'a> {
    pub fn new(network: &'a RiverNetwork) -> Self {
        let mesh = &network.surface;
        let counts: Vec<usize> = network.lakes.iter().map(|lake| super::surface::build(&[], std::slice::from_ref(lake)).vertices.len()).collect();
        let first_lake = mesh.vertices.len() - counts.iter().sum::<usize>();
        let advance_start = |start: &mut usize, &count: &usize| {
            *start += count;
            Some(*start)
        };
        let lake_starts: Vec<usize> = counts.iter().scan(first_lake, advance_start).collect();
        let mut buckets: std::collections::HashMap<[i32; 2], Vec<u32>> = Default::default();
        for (t, triangle) in mesh.indices.chunks(3).enumerate() {
            let v: [[f32; 3]; 3] = [0, 1, 2].map(|k| mesh.vertices[triangle[k] as usize].position);
            let bucket = |value: f32| (value / SURFACE_BUCKET).floor() as i32;
            let (x0, x1) = (bucket(v.iter().map(|p| p[0]).fold(f32::INFINITY, f32::min)), bucket(v.iter().map(|p| p[0]).fold(f32::NEG_INFINITY, f32::max)));
            let (z0, z1) = (bucket(v.iter().map(|p| p[2]).fold(f32::INFINITY, f32::min)), bucket(v.iter().map(|p| p[2]).fold(f32::NEG_INFINITY, f32::max)));
            for z in z0..=z1 {
                for x in x0..=x1 {
                    buckets.entry([x, z]).or_default().push(t as u32);
                }
            }
        }
        Self { mesh, first_lake, lake_starts, buckets }
    }

    /// (ribbon, sheet): the highest of each over `p`, or -infinity.
    #[cfg(test)]
    pub fn at(&self, p: [f32; 2]) -> (f32, f32) {
        let (ribbon, sheet, _) = self.owned(p);
        (ribbon, sheet)
    }

    /// (ribbon, sheet, the sheet's lake): the highest of each over `p`.
    pub fn owned(&self, p: [f32; 2]) -> (f32, f32, usize) {
        let (mut ribbon, mut sheet, mut owner) = (f32::NEG_INFINITY, f32::NEG_INFINITY, usize::MAX);
        self.each_surface(p, |h, first_vertex| {
            if first_vertex >= self.first_lake {
                if h > sheet {
                    sheet = h;
                    owner = self.lake_starts.partition_point(|&start| start <= first_vertex);
                }
            } else {
                ribbon = ribbon.max(h);
            }
        });
        (ribbon, sheet, owner)
    }

    /// The lowest and highest river ribbon over `p`, or (+inf, -inf): two
    /// apart where one river's water shows under another's.
    pub fn ribbons(&self, p: [f32; 2]) -> (f32, f32) {
        let (mut lowest, mut highest) = (f32::INFINITY, f32::NEG_INFINITY);
        self.each_surface(p, |h, first_vertex| {
            if first_vertex < self.first_lake {
                lowest = lowest.min(h);
                highest = highest.max(h);
            }
        });
        (lowest, highest)
    }

    /// The river ribbon drawn on top over `p` (the highest), its vertex
    /// attributes interpolated there as the GPU interpolates them.
    pub fn top_ribbon(&self, p: [f32; 2]) -> Option<SurfaceVertex> {
        let mut top: Option<SurfaceVertex> = None;
        self.each_triangle(p, |triangle, weights| {
            if triangle[0] as usize >= self.first_lake {
                return;
            }
            let v = triangle.map(|k| self.mesh.vertices[k as usize]);
            let mix = |f: &dyn Fn(&SurfaceVertex) -> f32| weights[0] * f(&v[0]) + weights[1] * f(&v[1]) + weights[2] * f(&v[2]);
            let height = mix(&|v| v.position[1]);
            if top.is_some_and(|t| t.position[1] >= height) {
                return;
            }
            top = Some(SurfaceVertex {
                position: [p[0], height, p[1]],
                velocity: [mix(&|v| v.velocity[0]), mix(&|v| v.velocity[1])],
                across: mix(&|v| v.across),
                turbulence: mix(&|v| v.turbulence),
                still: mix(&|v| v.still),
                along: mix(&|v| v.along),
                half_width: mix(&|v| v.half_width),
                foam: mix(&|v| v.foam),
                sea: mix(&|v| v.sea),
                side: mix(&|v| v.side),
            });
        });
        top
    }

    /// Every drawn triangle over `p`: its height there and its first vertex.
    fn each_surface(&self, p: [f32; 2], mut visit: impl FnMut(f32, usize)) {
        self.each_triangle(p, |triangle, weights| {
            let height = (0..3).map(|k| weights[k] * self.mesh.vertices[triangle[k] as usize].position[1]).sum();
            visit(height, triangle[0] as usize);
        });
    }

    /// Every drawn triangle over `p`, with `p`'s barycentric weights in it.
    fn each_triangle(&self, p: [f32; 2], mut visit: impl FnMut([u32; 3], [f32; 3])) {
        let key = [(p[0] / SURFACE_BUCKET).floor() as i32, (p[1] / SURFACE_BUCKET).floor() as i32];
        for &t in self.buckets.get(&key).map_or(&[][..], |list| &list[..]) {
            let triangle = [0, 1, 2].map(|k| self.mesh.indices[t as usize * 3 + k]);
            let [a, b, c] = triangle.map(|k| self.mesh.vertices[k as usize].position);
            let d = (b[2] - c[2]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[2] - c[2]);
            if d.abs() < 1e-9 {
                continue;
            }
            let l1 = ((b[2] - c[2]) * (p[0] - c[0]) + (c[0] - b[0]) * (p[1] - c[2])) / d;
            let l2 = ((c[2] - a[2]) * (p[0] - c[0]) + (a[0] - c[0]) * (p[1] - c[2])) / d;
            let l3 = 1.0 - l1 - l2;
            if l1 < -1e-5 || l2 < -1e-5 || l3 < -1e-5 {
                continue;
            }
            visit(triangle, [l1, l2, l3]);
        }
    }
}

/// How smoothly the water changes where two rivers meet, sampled every
/// quarter metre over a square around each confluence. A seam is a step
/// between two neighbouring samples beyond the change on either side of
/// them, so a ramp or a kink is none, however steep:
/// - in the water drawn on top: its height, its current, its whitewater
///   (turbulence and drifting foam) and its ripple frame (`along` and
///   `side`, in metres), which the shader lays its
///   ripples, standing waves and foam in;
/// - in the bed the terrain paints under the water: the whitewater, speed
///   and water level of the channel the envelope hands the point to.
#[derive(Default)]
pub struct JunctionSeams {
    pub junctions: usize,
    pub samples: usize,
    /// Steps by kind, and the worst of each and where: height, current,
    /// whitewater, frame, bed whitewater, bed speed, bed water level.
    pub steps: [usize; SEAM_KINDS],
    pub worst: [(f32, [f32; 2]); SEAM_KINDS],
}

pub const SEAM_KINDS: usize = 7;
pub const SEAM_NAMES: [&str; SEAM_KINDS] = ["height", "current", "whitewater", "frame", "bed whitewater", "bed speed", "bed level"];
/// A step this far beyond the change either side of it is a seam.
pub const SEAM_LIMITS: [f32; SEAM_KINDS] = [0.03, 0.3, 0.1, 0.5, 0.1, 0.3, 0.03];

/// Seams within the square of side `extent` around `centre`.
pub fn seams_around(noise: &NoiseField, network: &RiverNetwork, index: &SurfaceIndex, centre: [f32; 2], extent: f32, seams: &mut JunctionSeams) {
    const STEP: f32 = 0.25;
    let n = (extent / STEP) as usize + 1;
    let origin = [centre[0] - 0.5 * extent, centre[1] - 0.5 * extent];
    // Per sample, its measures (several numbers for the current and the
    // frame), or None where no water shows.
    let mut grid: Vec<Option<[Vec<f32>; SEAM_KINDS]>> = vec![None; n * n];
    for zi in 0..n {
        for xi in 0..n {
            let p = [origin[0] + xi as f32 * STEP, origin[1] + zi as f32 * STEP];
            let (ground, envelope) = carved_height(noise, network, p[0], p[1]);
            let (_, sheet, _) = index.owned(p);
            let Some(top) = index.top_ribbon(p) else { continue };
            let visible = top.position[1] > ground + 0.01 && top.position[1] > sheet + 0.05 && top.position[1] > SEA_LEVEL + 0.05;
            if !visible {
                continue;
            }
            seams.samples += 1;
            // Clear of the banks, where a channel's current falls steeply
            // to nothing at its waterline.
            let bed = |v: f32| if envelope.bank_distance < -0.4 { vec![v] } else { vec![] };
            grid[zi * n + xi] = Some([
                vec![top.position[1]],
                vec![top.velocity[0], top.velocity[1]],
                vec![top.turbulence.max(top.foam)],
                vec![top.along, top.side],
                bed(envelope.turbulence),
                bed(envelope.velocity[0].hypot(envelope.velocity[1])),
                bed(envelope.water),
            ]);
        }
    }
    // The change between two samples of one kind: the length of the
    // difference of their measures, if both have them.
    let change = |a: &Option<[Vec<f32>; SEAM_KINDS]>, b: &Option<[Vec<f32>; SEAM_KINDS]>, kind: usize| -> Option<f32> {
        let (a, b) = (a.as_ref()?, b.as_ref()?);
        if a[kind].is_empty() || b[kind].is_empty() {
            return None;
        }
        Some(a[kind].iter().zip(&b[kind]).map(|(x, y)| (x - y) * (x - y)).sum::<f32>().sqrt())
    };
    for zi in 0..n {
        for xi in 0..n {
            for (dx, dz) in [(1usize, 0usize), (0, 1)] {
                // Four samples in a row: a, b, c, d; the step is b to c.
                if xi < dx || zi < dz || xi + 2 * dx >= n || zi + 2 * dz >= n {
                    continue;
                }
                let at = |k: usize| &grid[(zi + k * dz - dz) * n + xi + k * dx - dx];
                let (a, b, c, d) = (at(0), at(1), at(2), at(3));
                let middle = [origin[0] + (xi as f32 + 0.5 * dx as f32) * STEP, origin[1] + (zi as f32 + 0.5 * dz as f32) * STEP];
                for (kind, &limit) in SEAM_LIMITS.iter().enumerate() {
                    let Some(step) = change(b, c, kind) else { continue };
                    let before = change(a, b, kind).unwrap_or(0.0);
                    let after = change(c, d, kind).unwrap_or(0.0);
                    let excess = step - before.max(after);
                    if excess > limit {
                        seams.steps[kind] += 1;
                        if excess > seams.worst[kind].0 {
                            seams.worst[kind] = (excess, middle);
                        }
                    }
                }
            }
        }
    }
}

pub fn junction_seams(noise: &NoiseField, network: &RiverNetwork) -> JunctionSeams {
    let index = SurfaceIndex::new(network);
    let mut seams = JunctionSeams::default();
    for river in &network.rivers {
        let RiverEnd::Confluence(parent, _) = river.end else {
            continue;
        };
        let Some(parent) = network.rivers.get(parent) else {
            continue;
        };
        let Some(node) = river.nodes.get(river.surface_end) else {
            continue;
        };
        let widest = parent.nodes.iter().map(|n| n.half_width).fold(0.0f32, f32::max).min(8.0);
        seams.junctions += 1;
        seams_around(noise, network, &index, node.position, 2.0 * (widest + 10.0), &mut seams);
    }
    seams
}

pub fn seam_fit(noise: &NoiseField, network: &RiverNetwork) -> String {
    let seams = junction_seams(noise, network);
    let parts: Vec<String> = (0..SEAM_KINDS)
        .map(|k| {
            let (worst, at) = seams.worst[k];
            format!("{} {} (worst {worst:.2} at {:.1},{:.1})", seams.steps[k], SEAM_NAMES[k], at[0], at[1])
        })
        .collect();
    format!("confluences: {} junctions, {} water samples; seams: {}", seams.junctions, seams.samples, parts.join(", "))
}

/// How the water looks around the lakes, sampled every metre over and
/// around each:
/// - shaded: ground the terrain shades as under water (below a lake's
///   surface field, or a channel's water inside its banks) that no water
///   surface covers;
/// - tilted: a lake's sheet showing more than 10 cm under its level, where
///   it dives under the ground or a river's ribbon and its slope shows;
/// - walls: water showing over a sample more than 10 cm above the bare
///   ground of the next sample, so it ends in the air there;
/// - seams: neighbouring samples whose surface steps by more than 5 cm from
///   a ribbon to a sheet.
pub struct WaterHoles {
    pub samples: usize,
    pub shaded: usize,
    pub tilted: usize,
    pub walls: usize,
    pub seams: usize,
    /// (tilted and wall samples, the worst, where) by lake.
    pub worst: Vec<(usize, f32, [f32; 2])>,
}

pub fn water_holes(noise: &NoiseField, network: &RiverNetwork) -> WaterHoles {
    let index = SurfaceIndex::new(network);
    let cell = network::FLOW_CELL as f32;
    let mut holes = WaterHoles { samples: 0, shaded: 0, tilted: 0, walls: 0, seams: 0, worst: Vec::new() };
    for (number, lake) in network.lakes.iter().enumerate() {
        let (mut minimum, mut maximum) = ([f32::INFINITY; 2], [f32::NEG_INFINITY; 2]);
        for c in lake.cells.iter().chain(&lake.shore) {
            for axis in 0..2 {
                minimum[axis] = minimum[axis].min(c[axis] as f32 * cell - 16.0);
                maximum[axis] = maximum[axis].max((c[axis] + 1) as f32 * cell + 16.0);
            }
        }
        let width = (maximum[0] - minimum[0]).ceil() as usize;
        let depth = (maximum[1] - minimum[1]).ceil() as usize;
        // Per sample: the ground, and the water showing over it (or -inf),
        // and whether that is a sheet's.
        let mut ground = vec![0.0f32; width * depth];
        let mut top = vec![f32::NEG_INFINITY; width * depth];
        let mut sheet = vec![false; width * depth];
        let (mut count, mut worst, mut at) = (0usize, 0.0f32, [0.0f32; 2]);
        for zi in 0..depth {
            for xi in 0..width {
                let i = zi * width + xi;
                let p = [minimum[0] + xi as f32 + 0.5, minimum[1] + zi as f32 + 0.5];
                let (height, envelope) = carved_height(noise, network, p[0], p[1]);
                let (ribbon, sheet_height, owner) = index.owned(p);
                // The sea's surface lies over all ground below its level.
                let surface = ribbon.max(sheet_height).max(SEA_LEVEL);
                ground[i] = height;
                holes.samples += 1;
                if surface > height {
                    top[i] = surface;
                    sheet[i] = sheet_height >= ribbon && sheet_height > SEA_LEVEL;
                    let dip = lake.level - sheet_height;
                    if sheet[i] && owner == number && sheet_height > height + 0.01 && dip > 0.1 {
                        holes.tilted += 1;
                        count += 1;
                        if dip > worst {
                            worst = dip;
                            at = p;
                        }
                    }
                    continue;
                }
                let channel = envelope.bank_distance < 0.0;
                let under_river = if channel { envelope.water - height } else { f32::NEG_INFINITY };
                if (envelope.lake - height).max(under_river) > 0.03 {
                    holes.shaded += 1;
                }
            }
        }
        for zi in 0..depth {
            for xi in 0..width {
                let i = zi * width + xi;
                for j in [i + 1, i + width] {
                    if (j == i + 1 && xi + 1 >= width) || j >= width * depth {
                        continue;
                    }
                    let (wet, dry) = match (top[i] > f32::NEG_INFINITY, top[j] > f32::NEG_INFINITY) {
                        (true, true) => {
                            if sheet[i] != sheet[j] && (top[i] - top[j]).abs() > 0.05 {
                                holes.seams += 1;
                            }
                            continue;
                        }
                        (true, false) => (i, j),
                        (false, true) => (j, i),
                        (false, false) => continue,
                    };
                    let wall = top[wet] - ground[dry];
                    if wall > 0.1 {
                        holes.walls += 1;
                        count += 1;
                        if wall > worst {
                            worst = wall;
                            at = [minimum[0] + (dry % width) as f32 + 0.5, minimum[1] + (dry / width) as f32 + 0.5];
                        }
                    }
                }
            }
        }
        if count > 0 {
            holes.worst.push((count, worst, at));
        }
    }
    holes.worst.sort_by_key(|w| std::cmp::Reverse(w.0));
    holes
}

pub fn hole_fit(noise: &NoiseField, network: &RiverNetwork) -> Vec<String> {
    let holes = water_holes(noise, network);
    let mut lines = vec![format!(
        "water around lakes: of {} samples, {} shaded as under water but bare, {} of sheet tilted under its level, {} water walls, {} ribbon/sheet seams ({} lakes tilted or walled)",
        holes.samples,
        holes.shaded,
        holes.tilted,
        holes.walls,
        holes.seams,
        holes.worst.len()
    )];
    for (count, worst, at) in holes.worst.iter().take(8) {
        lines.push(format!("  {count} tilted or walled, worst {worst:.2} m at {:.1},{:.1}", at[0], at[1]));
    }
    lines
}

/// How the rivers' ribbons sit on the ground and on each other, sampled
/// across every drawn reach every half metre:
/// - perched: a ribbon standing more than 10 cm over dry ground past its
///   channel's waterline (not a lake's bed), so the water runs in the air
///   beside its bank;
/// - under: two ribbons over a point more than 20 cm apart with the lower
///   one above the ground, where one river's water shows under another's.
pub struct RibbonFit {
    pub samples: usize,
    pub perched: usize,
    pub under: usize,
    /// (how far, where): the worst of each.
    pub worst_perched: (f32, [f32; 2]),
    pub worst_under: (f32, [f32; 2]),
}

pub fn ribbon_checks(noise: &NoiseField, network: &RiverNetwork) -> RibbonFit {
    let index = SurfaceIndex::new(network);
    let mut fit = RibbonFit { samples: 0, perched: 0, under: 0, worst_perched: (0.0, [0.0; 2]), worst_under: (0.0, [0.0; 2]) };
    for river in &network.rivers {
        let nodes = &river.nodes;
        let end = (river.surface_end + 1).min(nodes.len().saturating_sub(1));
        for i in 0..end {
            let (a, b) = (&nodes[i], &nodes[i + 1]);
            let d = [b.position[0] - a.position[0], b.position[1] - a.position[1]];
            let length = d[0].hypot(d[1]).max(1e-3);
            let normal = [-d[1] / length, d[0] / length];
            let reach = 1.2 * a.half_width.max(b.half_width) + 1.5;
            for t in [0.0f32, 0.25, 0.5, 0.75] {
                let centre = [a.position[0] + d[0] * t, a.position[1] + d[1] * t];
                let mut offset = -reach;
                while offset <= reach {
                    let p = [centre[0] + normal[0] * offset, centre[1] + normal[1] * offset];
                    offset += 0.5;
                    let (lowest, highest) = index.ribbons(p);
                    if highest <= SEA_LEVEL + 0.05 {
                        continue;
                    }
                    let (_, sheet, _) = index.owned(p);
                    if sheet >= highest - 0.05 {
                        continue;
                    }
                    fit.samples += 1;
                    let (ground, envelope) = carved_height(noise, network, p[0], p[1]);
                    // Ground under a lake's or the sea's water is its bed,
                    // not a bank.
                    let perched = highest - ground.max(SEA_LEVEL);
                    let dry = envelope.lake < ground && sheet < ground;
                    if envelope.bank_distance > 0.0 && dry && perched > 0.1 {
                        fit.perched += 1;
                        if perched > fit.worst_perched.0 {
                            fit.worst_perched = (perched, p);
                        }
                    }
                    let gap = highest - lowest;
                    if gap > 0.2 && lowest > ground + 0.02 {
                        fit.under += 1;
                        if gap > fit.worst_under.0 {
                            fit.worst_under = (gap, p);
                        }
                    }
                }
            }
        }
    }
    fit
}

pub fn ribbon_fit(noise: &NoiseField, network: &RiverNetwork) -> String {
    let fit = ribbon_checks(noise, network);
    let (perched, p) = fit.worst_perched;
    let (under, u) = fit.worst_under;
    format!(
        "ribbons: of {} samples, {} perched over the ground beside their banks (worst {perched:.2} m at {:.1},{:.1}), {} showing one river under another (worst {under:.2} m at {:.1},{:.1})",
        fit.samples, fit.perched, p[0], p[1], fit.under, u[0], u[1]
    )
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
    for line in junction_fit(noise, &network) {
        println!("{line}");
    }
    for line in hole_fit(noise, &network) {
        println!("{line}");
    }
    println!("{}", ribbon_fit(noise, &network));
    println!("{}", seam_fit(noise, &network));
    // The nodes nearest the map's centre, for framing a camera on them: the
    // heading is the `--camera` yaw that looks downstream.
    let mut nearest: Vec<(f32, &network::RiverNode, f32)> = network
        .rivers
        .iter()
        .flat_map(|r| {
            let nodes = &r.nodes;
            let heading_at = move |i: usize| {
                let (a, b) = (&nodes[i.saturating_sub(1)], &nodes[(i + 1).min(nodes.len() - 1)]);
                let heading = (b.position[0] - a.position[0])
                    .atan2(a.position[1] - b.position[1])
                    .to_degrees()
                    .rem_euclid(360.0);
                (&nodes[i], heading)
            };
            (0..nodes.len()).map(heading_at)
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

/// `--river-probe x,z`: dump the carved base (no erosion) and the river
/// envelope on a fine grid around a point, plus the rivers and carve
/// segments there, for inspecting a reported spot without a window.
/// Writes `<prefix>.f32` (rows of base, carved, upper, lower, water,
/// bank_distance, lake, and the highest and lowest drawn ribbon and the
/// sheet over the point, per sample) and prints the header and nodes.
pub fn run_probe(noise: &NoiseField, centre: [f64; 2], extent: f64, step: f64, prefix: &str) {
    let region = network::region_of(centre[0], centre[1]);
    let network = network::generate(noise, region);
    let index = SurfaceIndex::new(&network);
    let n = (extent / step).round() as usize + 1;
    let origin = [centre[0] - extent * 0.5, centre[1] - extent * 0.5];
    println!("PROBE region {:?} origin {:.3},{:.3} step {} n {}", region, origin[0], origin[1], step, n);
    let mut data: Vec<f32> = Vec::with_capacity(n * n * 10);
    for row in 0..n {
        let z = (origin[1] + row as f64 * step) as f32;
        for column in 0..n {
            let x = (origin[0] + column as f64 * step) as f32;
            let base = base_height(noise, x, z);
            let (carved, e) = carved_height(noise, &network, x, z);
            let (_, sheet, _) = index.owned([x, z]);
            let (lowest, highest) = index.ribbons([x, z]);
            let clean = |v: f32| if v.is_finite() { v.clamp(-1.0e6, 1.0e6) } else if v > 0.0 { 1.0e6 } else { -1.0e6 };
            data.extend_from_slice(&[
                base,
                carved,
                clean(e.upper),
                clean(e.lower),
                clean(e.water),
                clean(e.bank_distance),
                clean(e.lake),
                clean(highest),
                clean(lowest.min(highest)),
                clean(sheet),
            ]);
        }
    }
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(format!("{prefix}.f32"), bytes).expect("write probe");
    let inside = |p: [f32; 2], margin: f64| {
        (p[0] as f64 - centre[0]).abs() <= extent * 0.5 + margin && (p[1] as f64 - centre[1]).abs() <= extent * 0.5 + margin
    };
    for (index, river) in network.rivers.iter().enumerate() {
        let near: Vec<usize> = (0..river.nodes.len()).filter(|&i| inside(river.nodes[i].position, 40.0)).collect();
        if near.is_empty() {
            continue;
        }
        println!(
            "RIVER {index} end {:?} surface_end {} nodes {} (near {}..={})",
            river.end,
            river.surface_end,
            river.nodes.len(),
            near[0],
            near[near.len() - 1]
        );
        for &i in &near {
            let node = &river.nodes[i];
            let ground = base_height(noise, node.position[0], node.position[1]);
            println!(
                "  node {i} pos {:.2},{:.2} along {:.1} water {:.3} ground {:.3} hw {:.2} depth {:.2} bank {:.3} slope {:.4} turb {:.2} skew {:.2} lake {} speed {:.2}",
                node.position[0], node.position[1], node.along, node.water, ground, node.half_width, node.depth, node.bank, node.slope,
                node.turbulence, node.skew, node.lake, node.speed
            );
        }
    }
    for (index, segment) in network.segments.iter().enumerate() {
        if inside(segment.a, 20.0) || inside(segment.b, 20.0) {
            println!(
                "SEG {index} a {:.2},{:.2} b {:.2},{:.2} water {:.3},{:.3} hw {:.2},{:.2} depth {:.2},{:.2} bank {:.3},{:.3} skew {:.2},{:.2} turb {:.2} levee {:.3},{:.3} caps {:.4},{:.4}",
                segment.a[0], segment.a[1], segment.b[0], segment.b[1], segment.water[0], segment.water[1],
                segment.half_width[0], segment.half_width[1], segment.depth[0], segment.depth[1], segment.bank[0], segment.bank[1],
                segment.skew[0], segment.skew[1], segment.turbulence[0].max(segment.turbulence[1]), segment.levee[0], segment.levee[1],
                segment.cap_slope[0], segment.cap_slope[1]
            );
        }
    }
    for (index, lake) in network.lakes.iter().enumerate() {
        let cell = network::FLOW_CELL as f32;
        if lake.cells.iter().any(|c| inside([(c[0] as f32 + 0.5) * cell, (c[1] as f32 + 0.5) * cell], 40.0)) {
            println!("LAKE {index} level {:.3} cells {} shore {} edge {}", lake.level, lake.cells.len(), lake.shore.len(), lake.edge.len());
        }
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

    /// Every river that reaches the sea runs out into open water rather than
    /// into a hollow on the beach, and no river darts out of its lake and
    /// straight back into it.
    #[test]
    fn spawn_region_rivers_meet_still_water_cleanly() {
        let noise = NoiseField::new();
        let network = network::generate(&noise, [0, 0]);
        let j = junctions(&noise, &network);
        assert!(j.mouths > 50);
        assert!(j.cut_off_mouths.len() * 20 <= j.mouths, "{:?}", j.cut_off_mouths);
        assert_eq!(j.flicker, 0);
    }

    /// The rivers' water sits in their channels: a ribbon stands over dry
    /// ground beyond its banks only here and there, and only a little, and
    /// a tributary's water rarely shows under its parent's where they meet.
    #[test]
    fn spawn_region_ribbons_sit_in_their_channels() {
        let noise = NoiseField::new();
        let network = network::generate(&noise, [0, 0]);
        let fit = ribbon_checks(&noise, &network);
        assert!(fit.samples > 500_000, "{}", fit.samples);
        assert!(fit.perched * 1000 <= fit.samples, "{} of {} perched", fit.perched, fit.samples);
        assert!(fit.worst_perched.0 < 1.5, "perched {:?}", fit.worst_perched);
        assert!(fit.under * 5000 <= fit.samples, "{} of {} under another", fit.under, fit.samples);
    }

    /// The highest gap between two ribbons over a point, with the lower over
    /// the ground, and the highest a ribbon stands over dry ground past its
    /// banks, over a square of side `extent` around `centre`.
    fn ribbons_around(noise: &NoiseField, network: &RiverNetwork, centre: [f32; 2], extent: f32) -> (usize, f32, f32) {
        let index = SurfaceIndex::new(network);
        let (mut samples, mut under, mut perched) = (0usize, 0.0f32, 0.0f32);
        let steps = (extent / 0.5) as i32;
        for zi in 0..=steps {
            for xi in 0..=steps {
                let p = [centre[0] - 0.5 * extent + xi as f32 * 0.5, centre[1] - 0.5 * extent + zi as f32 * 0.5];
                let (lowest, highest) = index.ribbons(p);
                if highest == f32::NEG_INFINITY {
                    continue;
                }
                samples += 1;
                let (_, sheet, _) = index.owned(p);
                let (ground, envelope) = carved_height(noise, network, p[0], p[1]);
                if lowest > ground {
                    under = under.max(highest - lowest);
                }
                if envelope.bank_distance > 0.0 && envelope.lake < ground && sheet < ground {
                    perched = perched.max(highest - ground.max(SEA_LEVEL));
                }
            }
        }
        (samples, under, perched)
    }

    /// Reported: down a steep reach a creek met its river at a narrow angle.
    /// Graded to the river's level where its course ended, on the river's
    /// thalweg, it ran in under the river's water, which stood most of a
    /// metre higher where the two channels first opened into each other.
    #[test]
    fn reported_steep_confluence_meets_its_parents_water() {
        let noise = NoiseField::new();
        let network = network::generate(&noise, network::region_of(496.1, 1516.9));
        let (samples, under, _) = ribbons_around(&noise, &network, [496.0, 1507.0], 16.0);
        assert!(samples > 200, "{samples}");
        assert!(under < 0.25, "the creek's water shows {under} m under the river's");
    }

    /// Where two rivers meet, the water drawn over the junction is one
    /// water: its height, its current, its whitewater and its ripple frame
    /// run on from the tributary's surface into the parent's without a step,
    /// and so do the whitewater, speed and level of the bed the terrain
    /// paints under it.
    #[test]
    fn spawn_region_confluences_join_without_a_seam() {
        let noise = NoiseField::new();
        let network = network::generate(&noise, [0, 0]);
        let seams = junction_seams(&noise, &network);
        assert!(seams.junctions >= 20 && seams.samples > 20_000, "{} junctions, {} samples", seams.junctions, seams.samples);
        // Seams per 10,000 samples of water: what is left lies mostly where
        // a tributary stands under its parent out of a lake, and keeps its
        // own water.
        let limits = [8, 8, 5, 6, 2, 18, 8];
        for kind in 0..SEAM_KINDS {
            let (worst, at) = seams.worst[kind];
            assert!(
                seams.steps[kind] * 10_000 <= limits[kind] * seams.samples,
                "{} {} seams in {} samples (worst {worst} at {at:?})",
                seams.steps[kind],
                SEAM_NAMES[kind],
                seams.samples
            );
        }
    }

    /// Reported: down a steep reach, where a creek ran into its river, the
    /// creek's water met the river's along a hard line across the creek's
    /// mouth: the river's ribbon reached past its waterline over the open
    /// water there, and every attribute of the water stepped at its edge;
    /// the bed under the merged water changed from the river's rock to the
    /// creek's gravel along another.
    #[test]
    fn reported_steep_confluence_joins_without_a_seam() {
        let noise = NoiseField::new();
        let network = network::generate(&noise, network::region_of(501.5, 1504.5));
        let index = SurfaceIndex::new(&network);
        let mut seams = JunctionSeams::default();
        seams_around(&noise, &network, &index, [497.0, 1506.0], 24.0, &mut seams);
        assert!(seams.samples > 1000, "{}", seams.samples);
        // A few steep changes are left right at the waterline, where the
        // creek's current falls to nothing against its bank.
        let limits = [0, 4, 0, 6, 0, 3, 3];
        for kind in 0..SEAM_KINDS {
            assert!(seams.steps[kind] <= limits[kind], "{} {} seams (worst {:?})", seams.steps[kind], SEAM_NAMES[kind], seams.worst[kind]);
        }
    }

    /// Reported: a creek cascading down a hillside into a lake ran in the
    /// air over the ground beside it. The levee that holds a channel's banks
    /// gives way near still water, its mouth flared wide and low-banked over
    /// the cascade, and its water stood a hand under the ground at its
    /// centre, over the hillside falling away across it.
    #[test]
    fn reported_cascade_into_a_lake_stays_in_its_channel() {
        let noise = NoiseField::new();
        let network = network::generate(&noise, network::region_of(2222.3, 5721.5));
        let (samples, _, perched) = ribbons_around(&noise, &network, [2232.0, 5721.0], 24.0);
        assert!(samples > 200, "{samples}");
        assert!(perched < 0.7, "the cascade's water stands {perched} m over the ground beside it");
    }

    /// The lakes' surface the terrain, the plants and the player measure
    /// the water by is the drawn sheet itself wherever a sheet is drawn, and
    /// where it gives out past a shore it has sunk far enough under the
    /// ground that the shore band measured from it has faded out: it never
    /// ends on an edge the terrain's shading would show.
    #[test]
    fn spawn_region_lake_surface_is_the_drawn_water() {
        let noise = NoiseField::new();
        let network = network::generate(&noise, [0, 0]);
        let index = SurfaceIndex::new(&network);
        let cell = network::FLOW_CELL as f32;
        let (mut samples, mut above) = (0usize, 0usize);
        for lake in &network.lakes {
            for c in lake.cells.iter().chain(&lake.shore).step_by(3) {
                for (u, v) in [(0.25, 0.75), (0.8, 0.3), (0.5, 0.5), (0.05, 0.9)] {
                    let p = [(c[0] as f32 + u) * cell, (c[1] as f32 + v) * cell];
                    let (_, sheet) = index.at(p);
                    let field = network.envelope(p[0], p[1]).lake;
                    samples += 1;
                    // The highest corners of any lakes meeting here: never
                    // under the sheet, over it only where two lakes meet.
                    assert!(field >= sheet - 2e-3, "field {field} under the sheet {sheet} at {p:?}");
                    above += usize::from(field > sheet + 2e-3);
                }
            }
        }
        assert!(samples > 1000);
        assert!(above * 200 <= samples, "{above} of {samples} samples over the sheet");
        // Every corner on the edge of a record's field (next to one no lake
        // reaches) lies far enough under the ground: a shore run of 11 m or
        // more, past the band of bare earth and the damp turf beyond it.
        let grid = &network.grid;
        let across = super::super::carve::LAKE_CORNERS_ACROSS;
        let fine = super::super::carve::GRID_CELL / super::super::carve::LAKE_CELLS_ACROSS as f32;
        let mut edges = 0;
        for (&index, record) in &grid.lakes {
            let origin = [
                grid.origin[0] + (index as usize % grid.resolution) as f32 * super::super::carve::GRID_CELL,
                grid.origin[1] + (index as usize / grid.resolution) as f32 * super::super::carve::GRID_CELL,
            ];
            for z in 0..across {
                for x in 0..across {
                    let height = record.corner(z * across + x);
                    if height <= super::super::carve::NO_LAKE {
                        continue;
                    }
                    let corner_is_open = |&(dx, dz): &(i32, i32)| {
                        let (nx, nz) = (x as i32 + dx, z as i32 + dz);
                        (0..across as i32).contains(&nx)
                            && (0..across as i32).contains(&nz)
                            && record.corner(nz as usize * across + nx as usize) <= super::super::carve::NO_LAKE
                    };
                    let open = [(1i32, 0i32), (-1, 0), (0, 1), (0, -1)].iter().any(corner_is_open);
                    if open {
                        edges += 1;
                        let p = [origin[0] + x as f32 * fine, origin[1] + z as f32 * fine];
                        let (ground, _) = carved_height(&noise, &network, p[0], p[1]);
                        assert!(ground - height >= 1.65, "the field ends {:.2} m under the ground at {p:?}", ground - height);
                    }
                }
            }
        }
        assert!(edges > 100, "{edges}");
    }
}
