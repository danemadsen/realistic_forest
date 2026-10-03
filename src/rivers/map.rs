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
        for node in &river.nodes {
            if node.fall > 1.0 {
                let (x, y) = to_pixel(node.position);
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        plot(x + dx as f64, y + dy as f64, [230, 30, 30]);
                    }
                }
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
    let falls: Vec<f32> = nodes.iter().filter(|n| n.fall > 0.0).map(|n| n.fall).collect();
    let tall = falls.iter().filter(|&&f| f > 1.5).count();
    let tallest = falls.iter().copied().fold(0.0f32, f32::max);
    lines.push(format!(
        "{} steps and falls ({} over 1.5 m, tallest {:.1} m), {} rocks",
        falls.len(),
        tall,
        tallest,
        network.rocks.len()
    ));
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
    lines.push(format!("built in {:.2} s", network.build_seconds));
    lines
}

pub fn run_map(noise: &NoiseField, path: &str, centre: [f64; 2], extent: f64) {
    let region = network::region_of(centre[0], centre[1]);
    let network = network::generate(noise, region);
    for line in report(&network) {
        println!("{line}");
    }
    // The nodes nearest the map's centre, for framing a camera on them.
    let mut nearest: Vec<(f32, &network::RiverNode)> = network
        .rivers
        .iter()
        .flat_map(|r| &r.nodes)
        .map(|n| ((n.position[0] - centre[0] as f32).hypot(n.position[1] - centre[1] as f32), n))
        .collect();
    nearest.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut shown: Vec<[f32; 2]> = Vec::new();
    for (d, node) in nearest {
        if shown.len() >= 8 {
            break;
        }
        if shown.iter().any(|p| (p[0] - node.position[0]).hypot(p[1] - node.position[1]) < 150.0) {
            continue;
        }
        shown.push(node.position);
        println!(
            "near centre ({d:.0} m): river at {:.1},{:.1} water {:.2} m, {:.1} m wide, {:.2} m deep, {:.2} m/s, slope {:.3}, fall {:.2} m",
            node.position[0], node.position[1], node.water, node.half_width * 2.0, node.depth, node.speed, node.slope, node.fall
        );
    }
    let mut falls: Vec<&network::RiverNode> = network
        .rivers
        .iter()
        .flat_map(|r| &r.nodes)
        .filter(|n| n.fall > 2.0)
        .collect();
    falls.sort_by(|a, b| {
        let da = (a.position[0] - centre[0] as f32).hypot(a.position[1] - centre[1] as f32);
        let db = (b.position[0] - centre[0] as f32).hypot(b.position[1] - centre[1] as f32);
        da.total_cmp(&db)
    });
    for node in falls.iter().take(4) {
        println!(
            "waterfall at {:.1},{:.1}: {:.1} m from {:.1} m, {:.1} m wide",
            node.position[0], node.position[1], node.fall, node.water, node.half_width * 2.0
        );
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
        let node = network.rivers.iter().flat_map(|r| &r.nodes).find(|n| n.depth > 0.3 && n.water > 2.0).unwrap();
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
        let entry = 8 + (cz * resolution + cx) * 2;
        let offset = words[entry] as usize;
        let count = (words[entry + 1] & 0xffff) as usize;
        assert!(count > 0);
        let mut total = super::super::carve::Envelope::NONE;
        for i in 0..count {
            let segment = &payload.segments[words[offset + i] as usize];
            super::super::carve::combine(&mut total, &super::super::carve::segment_envelope(segment, [x, z]));
        }
        assert_eq!(total.upper, envelope.upper);
    }
}
