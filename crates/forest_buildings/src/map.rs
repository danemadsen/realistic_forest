//! `--building-map out.png`: a top-down chart of the settlements over shaded
//! relief, with the sea, rivers and lakes, and a list of the towns and the
//! buildings nearest the chart's centre, without opening a window.

use super::assets::BuildingLibrary;
use super::placement::{Footprint, Placement, SettlementPlanner};
use bevy::math::Vec2;
use forest_core::constants::SEA_LEVEL;
use forest_core::noise::{NoiseField, base_height};
use std::sync::Arc;

/// Chart side in pixels.
const MAP_PIXELS: usize = 2048;

fn shade(height: f32, slope: Vec2) -> [u8; 3] {
    let normal = bevy::math::Vec3::new(-slope.x, 1.0, -slope.y).normalize();
    let light = bevy::math::Vec3::new(-0.5, 0.75, -0.45).normalize();
    let lit = normal.dot(light).clamp(0.0, 1.0) * 0.75 + 0.25;
    // Lowland green through upland brown to grey peaks.
    let t = ((height - SEA_LEVEL) / 120.0).clamp(0.0, 1.0);
    let low = [96.0, 128.0, 76.0];
    let high = [150.0, 140.0, 120.0];
    let colour: [f32; 3] = std::array::from_fn(|i| low[i] + (high[i] - low[i]) * t);
    colour.map(|channel| (channel * lit).clamp(0.0, 255.0) as u8)
}

fn draw_building(
    image: &mut image::RgbImage,
    placement: &Placement,
    footprint: &Footprint,
    colour: [u8; 3],
    to_pixel: impl Fn(Vec2) -> Vec2,
) {
    let pad = &placement.pad;
    let half = footprint.half;
    let samples = 24;
    for row in 0..=samples {
        for column in 0..=samples {
            let local = Vec2::new(
                -half.x + 2.0 * half.x * column as f32 / samples as f32,
                -half.y + 2.0 * half.y * row as f32 / samples as f32,
            );
            let world = pad.centre
                + Vec2::new(
                    local.x * pad.axis.x + local.y * pad.axis.y,
                    -local.x * pad.axis.y + local.y * pad.axis.x,
                );
            let pixel = to_pixel(world);
            if pixel.x >= 0.0
                && pixel.y >= 0.0
                && (pixel.x as u32) < image.width()
                && (pixel.y as u32) < image.height()
            {
                image.put_pixel(pixel.x as u32, pixel.y as u32, image::Rgb(colour));
            }
        }
    }
}

pub fn run_map(noise: &NoiseField, path: &str, centre: [f64; 2], extent: f64) {
    let start = std::time::Instant::now();
    let directory =
        std::path::PathBuf::from(forest_assets::resolveAssetRoot()).join("models/buildings");
    let library = match BuildingLibrary::load(&directory) {
        Ok(library) => library,
        Err(error) => {
            eprintln!("BUILDINGS: the kit could not be loaded: {error}");
            std::process::exit(1);
        }
    };
    let noise = Arc::new(noise.clone());
    let region = forest_water::rivers::network::region_of(centre[0], centre[1]);
    let network = Arc::new(forest_water::rivers::network::generate(&noise, region));
    let planner = SettlementPlanner::new(noise.clone(), Footprint::from_library(&library));
    let centre = Vec2::new(centre[0] as f32, centre[1] as f32);
    let half = extent as f32 * 0.5;
    let minimum = centre - Vec2::splat(half);
    let maximum = centre + Vec2::splat(half);
    let placements = planner.placements_in(Some(&network), minimum, maximum);
    let towns = planner.towns_in(Some(&network), minimum, maximum);
    let placed_seconds = start.elapsed().as_secs_f32();

    // Shaded relief, water and sea, a row of pixels per job.
    let metres_per_pixel = extent as f32 / MAP_PIXELS as f32;
    let rows: Vec<Vec<[u8; 3]>> = {
        let next = std::sync::atomic::AtomicUsize::new(0);
        let results = std::sync::Mutex::new(vec![Vec::new(); MAP_PIXELS]);
        let workers = std::thread::available_parallelism().map_or(4, |n| n.get());
        let worker = || loop {
            let row = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if row >= MAP_PIXELS {
                break;
            }
            let mut pixels = Vec::with_capacity(MAP_PIXELS);
            for column in 0..MAP_PIXELS {
                let point =
                    minimum + Vec2::new(column as f32 + 0.5, row as f32 + 0.5) * metres_per_pixel;
                let envelope = network.envelope(point.x, point.y);
                let ground = |p: Vec2| {
                    network
                        .envelope(p.x, p.y)
                        .clamp(base_height(&noise, p.x, p.y))
                };
                let height = envelope.clamp(base_height(&noise, point.x, point.y));
                let step = metres_per_pixel.max(2.0);
                let slope = Vec2::new(
                    ground(point + Vec2::new(step, 0.0)) - ground(point - Vec2::new(step, 0.0)),
                    ground(point + Vec2::new(0.0, step)) - ground(point - Vec2::new(0.0, step)),
                ) / (2.0 * step);
                let colour = if height < SEA_LEVEL {
                    [30, 60, 110]
                } else if envelope.bank_distance < 0.0 || height < envelope.lake {
                    [50, 110, 190]
                } else {
                    shade(height, slope)
                };
                pixels.push(colour);
            }
            results.lock().unwrap()[row] = pixels;
        };
        std::thread::scope(|scope| {
            for _ in 0..workers {
                scope.spawn(worker);
            }
        });
        results.into_inner().unwrap()
    };
    let mut image = image::RgbImage::new(MAP_PIXELS as u32, MAP_PIXELS as u32);
    for (row, pixels) in rows.iter().enumerate() {
        for (column, colour) in pixels.iter().enumerate() {
            image.put_pixel(column as u32, row as u32, image::Rgb(*colour));
        }
    }
    let to_pixel = |world: Vec2| (world - minimum) / metres_per_pixel;
    // Buildings drawn larger than life so they show at this scale: a red
    // block for a homestead, orange in a town, yellow for a jail.
    let footprints = planner.footprints();
    for placement in &placements {
        let footprint = &footprints[placement.model as usize];
        let colour = if footprint.name == "sheriff-office" {
            [255, 230, 40]
        } else if placement.town.is_some() {
            [255, 140, 30]
        } else {
            [220, 30, 30]
        };
        let mut swollen = footprint.clone();
        swollen.half = swollen.half.max(Vec2::splat(metres_per_pixel * 2.0));
        draw_building(&mut image, placement, &swollen, colour, to_pixel);
    }
    if let Err(error) = image.save(path) {
        eprintln!("BUILDINGS: could not write {path}: {error}");
        std::process::exit(1);
    }

    let homesteads = placements
        .iter()
        .filter(|placement| placement.town.is_none())
        .count();
    let area_km2 = extent as f32 * extent as f32 / 1.0e6;
    println!(
        "BUILDINGS: {} buildings over {:.0} km² ({} homesteads, {:.2} per km²; {} towns) placed in {:.2} s, charted to {path} in {:.1} s",
        placements.len(),
        area_km2,
        homesteads,
        homesteads as f32 / area_km2,
        towns.len(),
        placed_seconds,
        start.elapsed().as_secs_f32()
    );
    let mut by_model = std::collections::BTreeMap::new();
    for placement in &placements {
        *by_model
            .entry(footprints[placement.model as usize].name.as_str())
            .or_insert(0) += 1;
    }
    println!("BUILDINGS: by model {by_model:?}");
    let mut sorted_towns: Vec<_> = towns.iter().collect();
    sorted_towns.sort_by(|a, b| {
        a.centre
            .distance(centre)
            .total_cmp(&b.centre.distance(centre))
    });
    for town in sorted_towns {
        println!(
            "TOWN {:016x} at {:.0},{:.0} ({:.0} m away): {} buildings, ground {:.1} m",
            town.id,
            town.centre.x,
            town.centre.y,
            town.centre.distance(centre),
            town.placements.len(),
            town.placements
                .first()
                .map_or(0.0, |placement| placement.pad.height)
        );
    }
    let mut nearest: Vec<&Placement> = placements.iter().collect();
    nearest.sort_by(|a, b| {
        a.pad
            .centre
            .distance(centre)
            .total_cmp(&b.pad.centre.distance(centre))
    });
    for placement in nearest.iter().take(12) {
        println!(
            "BUILDING {} at {:.1},{:.1},{:.1} ({:.0} m away), origin {:.3},{:.3},{:.3}, yaw {:.5} rad, skirt {:.1} m{}",
            footprints[placement.model as usize].name,
            placement.pad.centre.x,
            placement.pad.height,
            placement.pad.centre.y,
            placement.pad.centre.distance(centre),
            placement.origin.x,
            placement.origin.y,
            placement.origin.z,
            placement.yaw,
            placement.pad.skirt,
            if placement.town.is_some() {
                ", in a town"
            } else {
                ""
            }
        );
    }
}
