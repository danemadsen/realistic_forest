//! `--vegetation-map out.png`: a top-down chart of the scatter, with the
//! statistics that matter for tuning it, without opening a window.

use super::assets::{Species, VegetationAssets};
use super::ecology::{self, SiteSampler};
use super::scatter::{self, Catalog, Layer, PlantInstance};
use crate::noise::NoiseField;

/// Generate every level of every chunk overlapping the square, in parallel.
pub fn scatter_region(noise: &NoiseField, catalog: &Catalog, minimum: [f64; 2], maximum: [f64; 2]) -> Vec<PlantInstance> {
    let low = scatter::chunk_of(minimum[0], minimum[1]);
    let high = scatter::chunk_of(maximum[0] - 1e-6, maximum[1] - 1e-6);
    let mut jobs = Vec::new();
    for cz in low[1]..=high[1] {
        for cx in low[0]..=high[0] {
            for level in 0..=scatter::LEVEL_GROUND {
                jobs.push(([cx, cz], level));
            }
        }
    }
    let next = std::sync::atomic::AtomicUsize::new(0);
    let results = std::sync::Mutex::new(Vec::new());
    let workers = std::thread::available_parallelism().map_or(4, |n| n.get());
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| loop {
                let index = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let Some(&(chunk, level)) = jobs.get(index) else {
                    break;
                };
                let plants = scatter::generate_level(noise, catalog, chunk, level);
                results.lock().unwrap().extend(plants);
            });
        }
    });
    let mut plants = results.into_inner().unwrap();
    plants.retain(|p| {
        (p.position[0] as f64) >= minimum[0]
            && (p.position[0] as f64) < maximum[0]
            && (p.position[2] as f64) >= minimum[1]
            && (p.position[2] as f64) < maximum[1]
    });
    // Chunk completion order varies; the set does not.
    plants.sort_by(|a, b| {
        (a.layer, a.position[0].to_bits(), a.position[2].to_bits())
            .cmp(&(b.layer, b.position[0].to_bits(), b.position[2].to_bits()))
    });
    plants
}

fn species_colour(species: Species) -> [u8; 3] {
    match species {
        Species::Fir => [18, 70, 38],
        Species::Pine => [150, 168, 52],
        Species::Oak => [120, 95, 40],
        Species::Maple => [210, 110, 30],
        Species::Lilac => [190, 120, 220],
        Species::Bush => [75, 120, 60],
        Species::Broadleaf => [40, 190, 170],
        Species::Lavender => [150, 70, 230],
    }
}

/// Counts and spatial statistics of a scatter, printed as a report. The map
/// prints `lines`; the tests hold the statistics to the brief.
#[cfg_attr(not(test), allow(dead_code))]
pub struct Report {
    pub lines: Vec<String>,
    pub species_counts: std::collections::BTreeMap<Species, usize>,
    pub fir_share_of_conifers: f32,
    pub same_species_neighbours: f32,
    pub expected_same_species: f32,
    pub forest_fraction: f32,
    pub closed_forest_stems_per_hectare: f32,
    pub lavender_in_clearings: f32,
}

pub fn report(noise: &NoiseField, catalog: &Catalog, plants: &[PlantInstance], minimum: [f64; 2], maximum: [f64; 2]) -> Report {
    use std::collections::BTreeMap;
    let mut lines = Vec::new();
    let mut species_counts: BTreeMap<Species, usize> = BTreeMap::new();
    let mut layer_counts: BTreeMap<u32, usize> = BTreeMap::new();
    for plant in plants {
        *species_counts.entry(catalog.models[plant.model as usize].species).or_default() += 1;
        *layer_counts.entry(plant.layer).or_default() += 1;
    }
    let area_m2 = (maximum[0] - minimum[0]) * (maximum[1] - minimum[1]);
    // Land and forest fractions from a 16 m sample lattice.
    let sampler = SiteSampler::new(noise, minimum, maximum);
    let (mut land, mut forest, mut closed) = (0usize, 0.0f64, 0usize);
    let step = 16.0;
    let mut z = minimum[1] + step * 0.5;
    while z < maximum[1] {
        let mut x = minimum[0] + step * 0.5;
        while x < maximum[0] {
            let habitat = ecology::habitat(&sampler, x, z);
            if habitat.site.height > 0.0 {
                land += 1;
                forest += habitat.forest as f64;
                if habitat.forest > 0.95 {
                    closed += 1;
                }
            }
            x += step;
        }
        z += step;
    }
    let land_m2 = land as f64 * step * step;
    let closed_m2 = closed as f64 * step * step;
    let forest_fraction = if land > 0 { (forest / land as f64) as f32 } else { 0.0 };

    // Trees standing in closed forest, and conifer neighbour statistics.
    let trees: Vec<&PlantInstance> = plants
        .iter()
        .filter(|p| p.layer == Layer::Canopy as u32 || p.layer == Layer::Regeneration as u32)
        .collect();
    let mut closed_stems = 0usize;
    for tree in &trees {
        let habitat = ecology::habitat(&sampler, tree.position[0] as f64, tree.position[2] as f64);
        if habitat.forest > 0.95 {
            closed_stems += 1;
        }
    }
    let conifers: Vec<(f32, f32, bool)> = trees
        .iter()
        .filter_map(|t| match catalog.models[t.model as usize].species {
            Species::Fir => Some((t.position[0], t.position[2], true)),
            Species::Pine => Some((t.position[0], t.position[2], false)),
            _ => None,
        })
        .collect();
    let firs = conifers.iter().filter(|c| c.2).count();
    let fir_share = firs as f32 / conifers.len().max(1) as f32;
    let mut buckets: std::collections::HashMap<(i32, i32), Vec<usize>> = Default::default();
    for (index, c) in conifers.iter().enumerate() {
        buckets.entry(((c.0 / 16.0).floor() as i32, (c.1 / 16.0).floor() as i32)).or_default().push(index);
    }
    let (mut same, mut pairs) = (0usize, 0usize);
    for (index, c) in conifers.iter().enumerate() {
        let (bx, bz) = ((c.0 / 16.0).floor() as i32, (c.1 / 16.0).floor() as i32);
        let mut best = (f32::INFINITY, None);
        for dz in -1..=1 {
            for dx in -1..=1 {
                for &other in buckets.get(&(bx + dx, bz + dz)).map(Vec::as_slice).unwrap_or(&[]) {
                    if other == index {
                        continue;
                    }
                    let o = conifers[other];
                    let d = (o.0 - c.0).hypot(o.1 - c.1);
                    if d < best.0 {
                        best = (d, Some(o.2));
                    }
                }
            }
        }
        if let Some(neighbour_is_fir) = best.1 {
            pairs += 1;
            same += (neighbour_is_fir == c.2) as usize;
        }
    }
    let same_fraction = same as f32 / pairs.max(1) as f32;
    let expected_same = fir_share * fir_share + (1.0 - fir_share) * (1.0 - fir_share);
    let lavender: Vec<&PlantInstance> = plants.iter().filter(|p| p.layer == Layer::Lavender as u32).collect();
    let lavender_open = lavender
        .iter()
        .filter(|p| ecology::habitat(&sampler, p.position[0] as f64, p.position[2] as f64).forest < 0.3)
        .count();
    let lavender_in_clearings = lavender_open as f32 / lavender.len().max(1) as f32;

    lines.push(format!(
        "VEGETATION: {:.2} km² ({:.2} km² land, {:.0}% forest, {:.2} km² closed)",
        area_m2 / 1e6,
        land_m2 / 1e6,
        forest_fraction * 100.0,
        closed_m2 / 1e6
    ));
    for (species, count) in &species_counts {
        lines.push(format!(
            "  {:16} {:8}  {:8.1} per hectare of land",
            species.name(),
            count,
            *count as f64 / (land_m2 / 1e4).max(1e-6)
        ));
    }
    for (layer, count) in &layer_counts {
        lines.push(format!("  layer {layer}: {count}"));
    }
    let closed_density = closed_stems as f64 / (closed_m2 / 1e4).max(1e-6);
    lines.push(format!(
        "  closed forest: {:.0} stems/ha (one per {:.0} m²)",
        closed_density,
        1e4 / closed_density.max(1e-6)
    ));
    lines.push(format!(
        "  conifers: {:.0}% fir; nearest conifer neighbour same species {:.0}% (random mixing {:.0}%)",
        fir_share * 100.0,
        same_fraction * 100.0,
        expected_same * 100.0
    ));
    lines.push(format!(
        "  lavender: {} clumps, {:.0}% in clearings",
        lavender.len(),
        lavender_in_clearings * 100.0
    ));
    Report {
        lines,
        species_counts,
        fir_share_of_conifers: fir_share,
        same_species_neighbours: same_fraction,
        expected_same_species: expected_same,
        forest_fraction,
        closed_forest_stems_per_hectare: closed_density as f32,
        lavender_in_clearings,
    }
}

/// Render the chart: shaded relief tinted by the forest field, plants drawn
/// as crown discs coloured by species.
pub fn render(
    noise: &NoiseField,
    catalog: &Catalog,
    plants: &[PlantInstance],
    minimum: [f64; 2],
    maximum: [f64; 2],
    metres_per_pixel: f64,
) -> image::RgbImage {
    let width = ((maximum[0] - minimum[0]) / metres_per_pixel).round() as u32;
    let height = ((maximum[1] - minimum[1]) / metres_per_pixel).round() as u32;
    let sampler = SiteSampler::new(noise, minimum, maximum);
    // Background at 4 m, upsampled by nearest neighbour.
    let step = 4.0f64;
    let cols = ((maximum[0] - minimum[0]) / step).ceil() as usize + 1;
    let rows = ((maximum[1] - minimum[1]) / step).ceil() as usize + 1;
    let mut background = vec![[0u8; 3]; cols * rows];
    for row in 0..rows {
        for col in 0..cols {
            let x = minimum[0] + col as f64 * step;
            let z = minimum[1] + row as f64 * step;
            let habitat = ecology::habitat(&sampler, x, z);
            let site = habitat.site;
            let colour = if site.height <= 0.0 {
                [40.0, 70.0, 120.0]
            } else {
                let shade = (0.75 + site.insolation * 1.4).clamp(0.35, 1.25);
                let meadow = [150.0, 165.0, 95.0];
                let floor = [70.0, 85.0, 55.0];
                let mut c = [0.0f32; 3];
                for i in 0..3 {
                    c[i] = (meadow[i] * (1.0 - habitat.forest) + floor[i] * habitat.forest) * shade;
                }
                if site.height < 4.0 {
                    c = [190.0, 180.0, 140.0];
                }
                if site.height > ecology::TREELINE + 18.0 {
                    c = [215.0, 215.0, 220.0];
                }
                c
            };
            background[row * cols + col] = colour.map(|v| v.clamp(0.0, 255.0) as u8);
        }
    }
    let mut image = image::RgbImage::from_fn(width, height, |px, py| {
        let col = ((px as f64 * metres_per_pixel) / step) as usize;
        let row = ((((height - 1 - py) as f64) * metres_per_pixel) / step) as usize;
        image::Rgb(background[row.min(rows - 1) * cols + col.min(cols - 1)])
    });
    // Ground layers first, canopy last, as seen from above.
    let mut order: Vec<&PlantInstance> = plants.iter().collect();
    order.sort_by_key(|p| std::cmp::Reverse(p.layer));
    for plant in order {
        let model = &catalog.models[plant.model as usize];
        let colour = species_colour(model.species);
        let radius = (model.crown_radius * plant.scale) as f64 / metres_per_pixel;
        let cx = (plant.position[0] as f64 - minimum[0]) / metres_per_pixel;
        let cy = height as f64 - 1.0 - (plant.position[2] as f64 - minimum[1]) / metres_per_pixel;
        let r = radius.max(0.6);
        let shade = 0.82 + 0.3 * plant.seed;
        for y in (cy - r).floor() as i64..=(cy + r).ceil() as i64 {
            for x in (cx - r).floor() as i64..=(cx + r).ceil() as i64 {
                if x < 0 || y < 0 || x >= width as i64 || y >= height as i64 {
                    continue;
                }
                let d = ((x as f64 - cx).powi(2) + (y as f64 - cy).powi(2)).sqrt();
                if d > r {
                    continue;
                }
                // Darker rim so overlapping crowns stay legible.
                let rim = if d > r - 1.0 && r > 2.0 { 0.65 } else { 1.0 };
                let pixel = colour.map(|v| (v as f64 * shade as f64 * rim).clamp(0.0, 255.0) as u8);
                image.put_pixel(x as u32, y as u32, image::Rgb(pixel));
            }
        }
    }
    image
}

/// The `--vegetation-map` entry point.
pub fn run_map(path: &str, centre: [f64; 2], extent: f64) {
    let start = std::time::Instant::now();
    let assets = match VegetationAssets::load_geometry(model_directory()) {
        Ok(assets) => assets,
        Err(error) => {
            eprintln!("VEGETATION: {error}");
            std::process::exit(1);
        }
    };
    let catalog = Catalog::from_assets(&assets);
    let noise = NoiseField::new();
    let minimum = [centre[0] - extent * 0.5, centre[1] - extent * 0.5];
    let maximum = [centre[0] + extent * 0.5, centre[1] + extent * 0.5];
    let plants = scatter_region(&noise, &catalog, minimum, maximum);
    let scattered = start.elapsed();
    for line in report(&noise, &catalog, &plants, minimum, maximum).lines {
        println!("{line}");
    }
    let metres_per_pixel = (extent / 2048.0).max(0.25);
    let image = render(&noise, &catalog, &plants, minimum, maximum, metres_per_pixel);
    match image.save(path) {
        Ok(()) => println!(
            "VEGETATION: {} plants scattered in {:.1?}; map ({:.2} m/px) saved to {path}",
            plants.len(),
            scattered,
            metres_per_pixel
        ),
        Err(error) => {
            eprintln!("VEGETATION: saving {path}: {error}");
            std::process::exit(1);
        }
    }
}

/// `assets/models`, resolved like the asset root (see `resolve_asset_root`).
pub fn model_directory() -> std::path::PathBuf {
    std::path::PathBuf::from(crate::resolve_asset_root()).join("models")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn catalog() -> Catalog {
        let assets = VegetationAssets::load_geometry(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/models"),
        )
        .expect("plant geometry loads");
        Catalog::from_assets(&assets)
    }

    #[test]
    fn chunks_are_deterministic_and_seamless() {
        let noise = NoiseField::new();
        let catalog = catalog();
        let a = scatter::generate_level(&noise, &catalog, [-1, 0], scatter::LEVEL_TREES);
        assert_eq!(a, scatter::generate_level(&noise, &catalog, [-1, 0], scatter::LEVEL_TREES));
        // A 2x2 block of independently generated chunks must still keep the
        // canopy's hard-core spacing across every shared border.
        let mut canopy = Vec::new();
        for chunk in [[-1, -1], [0, -1], [-1, 0], [0, 0]] {
            canopy.extend(
                scatter::generate_level(&noise, &catalog, chunk, scatter::LEVEL_TREES)
                    .into_iter()
                    .filter(|p| p.layer == Layer::Canopy as u32),
            );
        }
        assert!(canopy.len() > 1000, "{}", canopy.len());
        let reach = |p: &PlantInstance| catalog.models[p.model as usize].crown_radius * p.scale;
        let mut crossing = 0;
        for (i, a) in canopy.iter().enumerate() {
            for b in &canopy[i + 1..] {
                let d = (a.position[0] - b.position[0]).hypot(a.position[2] - b.position[2]);
                if d > 30.0 {
                    continue;
                }
                let limit = 0.66 * (reach(a) + reach(b));
                assert!(d >= limit - 1e-3, "trees {d:.2} m apart, limit {limit:.2}: {a:?} {b:?}");
                let border = |v: f32| v.signum() != (v - (a.position[0] - b.position[0])).signum();
                crossing += border(a.position[0]) as usize;
            }
        }
        assert!(crossing > 0);
    }

    #[test]
    fn composition_follows_the_brief() {
        let noise = NoiseField::new();
        let catalog = catalog();
        let (minimum, maximum) = ([-1536.0, -1536.0], [1536.0, 1536.0]);
        let plants = scatter_region(&noise, &catalog, minimum, maximum);
        let report = report(&noise, &catalog, &plants, minimum, maximum);
        for line in &report.lines {
            println!("{line}");
        }
        let count = |s: Species| *report.species_counts.get(&s).unwrap_or(&0);
        let (fir, pine) = (count(Species::Fir), count(Species::Pine));
        let shrubs = count(Species::Bush) + count(Species::Lilac);
        let broadleaf_trees = count(Species::Oak) + count(Species::Maple);
        // Fir dominates three to one, then pine; shrubs are commoner than
        // the scattered oaks and maples but far rarer than the conifers.
        assert!((0.70..0.80).contains(&report.fir_share_of_conifers), "{}", report.fir_share_of_conifers);
        assert!(fir > pine && pine > shrubs && shrubs > broadleaf_trees && broadleaf_trees > 0);
        assert!(shrubs * 3 < fir + pine);
        // Like grows near like: the nearest conifer is the same species far
        // more often than random mixing at this ratio would give.
        assert!(
            report.same_species_neighbours > report.expected_same_species + 0.1,
            "{} vs {}",
            report.same_species_neighbours,
            report.expected_same_species
        );
        // Closed forest a little sparser than one stem per 35 m².
        assert!(
            (150.0..260.0).contains(&report.closed_forest_stems_per_hectare),
            "{}",
            report.closed_forest_stems_per_hectare
        );
        assert!((0.35..0.80).contains(&report.forest_fraction), "{}", report.forest_fraction);
        assert!(report.lavender_in_clearings > 0.7, "{}", report.lavender_in_clearings);
        // Nothing grows in the sea or on the beach.
        assert!(plants.iter().all(|p| p.position[1] > 1.0), "plant below the shore");
        assert!(plants.iter().all(|p| p.scale.is_finite() && p.scale > 0.0));
    }
}
