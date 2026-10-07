//! Scattered vegetation: the plant library and where each plant stands.
//!
//! The main world owns the scatter. It loads the plant library on a
//! background task at startup, then streams 256 m chunks around the player
//! at three detail levels (trees out to ~2 km, shrubs to ~1 km, the shore's
//! broadleaf plants and the lavender to ~270 m), each level generated on the
//! async compute pool. The render world receives the library once and a flat
//! snapshot of every streamed plant whenever the set changes; it seats, culls
//! and draws them on the GPU (src/render/vegetation_node.rs).

pub mod assets;
pub mod collision;
pub mod ecology;
pub mod map;
pub mod scatter;

use crate::noise::NoiseField;
use crate::player::Player;
use assets::{Species, VegetationAssets};
use bevy::prelude::*;
use bevy::tasks::{AsyncComputeTaskPool, Task, block_on, poll_once};
use scatter::{Catalog, LEVEL_GROUND, LEVEL_SHRUBS, LEVEL_TREES, PlantInstance};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// How a model is drawn: its reach, LOD switch distances and root test.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RenderProfile {
    /// Beyond this many metres the plant is not drawn.
    pub max_distance: f32,
    /// LOD k ends this many plant heights from the eye (k < LOD count - 1);
    /// the last LOD runs to `max_distance`.
    pub lod_end: [f32; 3],
    /// Roots are seated on the exact terrain height and refused on steep or
    /// channelled ground (trees and tall shrubs), or read from the grass
    /// habitat capture and refused wherever grass would be (small plants).
    pub ground_habitat: bool,
    /// Light passing through leaves: none for bark, more for thin broad
    /// leaves than for needle clusters.
    pub translucency: f32,
    /// How many shadow cascades, nearest first, the plant is drawn into:
    /// trees shade the whole view, a herb only the ground at the eye's feet.
    pub shadow_cascades: u32,
}

/// Per-family drawing rules. LOD switches are in multiples of a plant's own
/// height, so a sapling simplifies at the same on-screen size as a 30 m
/// pine; the bands follow the earlier fir prototype's reach.
pub fn render_profile(species: Species, form: &str) -> RenderProfile {
    let tree = RenderProfile {
        max_distance: 2000.0,
        lod_end: [6.5, 13.0, 23.0],
        ground_habitat: false,
        translucency: 0.55,
        shadow_cascades: 4,
    };
    match species {
        Species::Fir | Species::Pine => tree,
        Species::Oak | Species::Maple => RenderProfile {
            translucency: 0.8,
            ..tree
        },
        Species::Lilac => RenderProfile {
            max_distance: if matches!(form, "sapling" | "small") { 520.0 } else { 900.0 },
            translucency: 0.8,
            shadow_cascades: if matches!(form, "sapling" | "small") { 2 } else { 3 },
            ..tree
        },
        Species::Bush => match form {
            "big" => RenderProfile {
                max_distance: 420.0,
                lod_end: [9.0, 18.0, 34.0],
                ground_habitat: false,
                translucency: 0.75,
                shadow_cascades: 2,
            },
            "medium" => RenderProfile {
                max_distance: 220.0,
                lod_end: [10.0, 20.0, 40.0],
                ground_habitat: true,
                translucency: 0.75,
                shadow_cascades: 1,
            },
            _ => RenderProfile {
                max_distance: 160.0,
                lod_end: [10.0, 20.0, 40.0],
                ground_habitat: true,
                translucency: 0.75,
                shadow_cascades: 1,
            },
        },
        // Shoreline colonies are seen along open beaches.
        Species::Broadleaf => RenderProfile {
            max_distance: 170.0,
            lod_end: [12.0, 24.0, 48.0],
            ground_habitat: true,
            translucency: 0.9,
            shadow_cascades: 1,
        },
        Species::Lavender => RenderProfile {
            max_distance: 230.0,
            lod_end: [f32::INFINITY; 3],
            ground_habitat: true,
            translucency: 0.65,
            shadow_cascades: 1,
        },
    }
}

/// A plant root is refused below this height above the sea, steeper than
/// this rise over run (about 42 degrees, where the terrain's faces turn to
/// bare rock), or this far below the ground around it (an incised channel).
/// The GPU cull enforces them to decide what is drawn, and the player's
/// collision to decide what is solid, so a trunk is never felt unseen.
pub const LOWEST_ROOT: f32 = 1.0;
pub const STEEPEST_ROOT: f32 = 0.9;
pub const DEEPEST_FURROW: f32 = 0.6;

/// Chunk detail levels are generated out to these distances from the
/// player (to the nearest point of the chunk) and dropped this much further
/// out, so walking back and forth across a boundary does not regenerate.
const LEVEL_RADIUS: [f64; 3] = [2060.0, 960.0, 270.0];
const LEVEL_HYSTERESIS: f64 = 220.0;
/// Concurrent chunk-level tasks. Each takes tens of milliseconds.
const MAX_IN_FLIGHT: usize = 6;
/// Arrivals are batched into one snapshot at most this often while the
/// stream is filling, since every snapshot is re-uploaded whole.
const PUBLISH_INTERVAL: f32 = 0.35;

/// Every streamed plant, flattened for the render world.
pub struct VegetationSnapshot {
    pub generation: u64,
    pub plants: Vec<PlantInstance>,
}

type ChunkLevels = [Option<Arc<Vec<PlantInstance>>>; 3];

#[derive(Resource)]
pub struct VegetationField {
    enabled: bool,
    noise: Arc<NoiseField>,
    loading: Option<Task<Result<VegetationAssets, String>>>,
    assets: Option<Arc<VegetationAssets>>,
    catalog: Option<Arc<Catalog>>,
    chunks: HashMap<[i64; 2], ChunkLevels>,
    tasks: HashMap<([i64; 2], u8), Task<Vec<PlantInstance>>>,
    snapshot: Arc<VegetationSnapshot>,
    dirty: bool,
    since_publish: f32,
    /// The snapshot generation the render world has on the GPU.
    pub uploaded: Arc<AtomicU64>,
    /// Whether every wanted level around the player was present at the last
    /// update, with nothing in flight.
    settled: bool,
}

impl VegetationField {
    pub fn new(noise: Arc<NoiseField>, enabled: bool) -> Self {
        let load_library = async move {
            let start = std::time::Instant::now();
            let assets = VegetationAssets::load(map::model_directory());
            if let Ok(assets) = &assets {
                log::info!(
                    "VEGETATION: {} models, {} textures ({:.0} MiB), {} triangles loaded in {:.1?}",
                    assets.models.len(),
                    assets.textures.len(),
                    assets.texture_bytes() as f64 / (1024.0 * 1024.0),
                    assets.indices.len() / 3,
                    start.elapsed()
                );
            }
            assets
        };
        let loading = enabled.then(|| AsyncComputeTaskPool::get().spawn(load_library));
        Self {
            enabled,
            noise,
            loading,
            assets: None,
            catalog: None,
            chunks: HashMap::new(),
            tasks: HashMap::new(),
            snapshot: Arc::new(VegetationSnapshot {
                generation: 0,
                plants: Vec::new(),
            }),
            dirty: false,
            since_publish: 0.0,
            uploaded: Arc::new(AtomicU64::new(0)),
            settled: !enabled,
        }
    }

    pub fn assets(&self) -> Option<&Arc<VegetationAssets>> {
        self.assets.as_ref()
    }

    pub fn snapshot(&self) -> &Arc<VegetationSnapshot> {
        &self.snapshot
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// The library is loaded, every level the player's position wants is
    /// streamed in, and the render world has the latest snapshot on the GPU.
    /// Screenshot runs wait for this. A disabled or failed field is ready.
    pub fn ready(&self) -> bool {
        if !self.enabled {
            return true;
        }
        self.settled
            && !self.dirty
            && self.uploaded.load(Ordering::Acquire) == self.snapshot.generation
    }

    /// Every solid plant (a tree or a lilac, see [`collision::is_solid`])
    /// streamed in with its stem within `reach` metres of `centre`, appended
    /// to `out`. Plants belong to the chunk their root stands in, so only the
    /// chunks the circle touches are read, and only their tree and shrub
    /// levels: the ground layers hold nothing solid.
    pub fn solids_near(&self, centre: Vec2, reach: f32, out: &mut Vec<collision::Solid>) {
        let Some(catalog) = self.catalog.as_deref() else {
            return;
        };
        let (low, high) = (centre - Vec2::splat(reach), centre + Vec2::splat(reach));
        let first = scatter::chunk_of(low.x as f64, low.y as f64);
        let last = scatter::chunk_of(high.x as f64, high.y as f64);
        for cz in first[1]..=last[1] {
            for cx in first[0]..=last[0] {
                let Some(levels) = self.chunks.get(&[cx, cz]) else {
                    continue;
                };
                for level in [LEVEL_TREES, LEVEL_SHRUBS] {
                    let Some(plants) = &levels[level as usize] else {
                        continue;
                    };
                    // A chunk holds thousands of plants and this runs every
                    // frame, so reject by position before the catalogue.
                    let near = reach + collision::MAX_TRUNK_RADIUS;
                    let candidates = plants.iter().filter(|plant| {
                        (plant.position[0] - centre.x).abs() <= near && (plant.position[2] - centre.y).abs() <= near
                    });
                    out.extend(
                        candidates
                            .filter_map(|plant| collision::Solid::of(catalog, plant))
                            .filter(|solid| solid.position.distance(centre) <= reach + solid.radius),
                    );
                }
            }
        }
    }

    /// A field holding exactly these plants, as if streamed into their chunks
    /// at the level their layer belongs to.
    #[cfg(test)]
    pub fn with_plants(catalog: Catalog, plants: Vec<PlantInstance>) -> Self {
        let mut field = Self::new(Arc::new(NoiseField { samples: Vec::new() }), false);
        field.catalog = Some(Arc::new(catalog));
        for plant in plants {
            let chunk = scatter::chunk_of(plant.position[0] as f64, plant.position[2] as f64);
            let level = match plant.layer {
                layer if layer <= scatter::Layer::Regeneration as u32 => LEVEL_TREES,
                layer if layer == scatter::Layer::Shrub as u32 => LEVEL_SHRUBS,
                _ => LEVEL_GROUND,
            };
            let levels = field.chunks.entry(chunk).or_default();
            let mut held = levels[level as usize].take().map_or_else(Vec::new, |held| (*held).clone());
            held.push(plant);
            levels[level as usize] = Some(Arc::new(held));
        }
        field
    }

    pub fn plant_count(&self) -> usize {
        self.snapshot.plants.len()
    }

    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    pub fn pending_count(&self) -> usize {
        self.tasks.len()
    }

    fn publish(&mut self) {
        let mut plants = Vec::with_capacity(self.snapshot.plants.len());
        for levels in self.chunks.values() {
            for level in levels.iter().flatten() {
                plants.extend_from_slice(level);
            }
        }
        self.snapshot = Arc::new(VegetationSnapshot {
            generation: self.snapshot.generation + 1,
            plants,
        });
        self.dirty = false;
        self.since_publish = 0.0;
    }
}

/// Distance from a point to the nearest point of a chunk.
fn chunk_distance(chunk: [i64; 2], x: f64, z: f64) -> f64 {
    let (minimum, maximum) = scatter::chunk_bounds(chunk);
    let dx = (minimum[0] - x).max(0.0).max(x - maximum[0]);
    let dz = (minimum[1] - z).max(0.0).max(z - maximum[1]);
    dx.hypot(dz)
}

/// Stream chunk levels around the player and publish snapshots.
pub fn stream_vegetation(
    mut field: ResMut<VegetationField>,
    rivers: Res<crate::rivers::RiverField>,
    players: Query<&Player>,
    time: Res<Time>,
) {
    if !field.enabled {
        return;
    }
    let field = &mut *field;
    if let Some(task) = field.loading.as_mut() {
        let Some(result) = block_on(poll_once(task)) else {
            return;
        };
        field.loading = None;
        match result {
            Ok(assets) => {
                field.catalog = Some(Arc::new(Catalog::from_assets(&assets)));
                field.assets = Some(Arc::new(assets));
            }
            Err(error) => {
                log::error!("VEGETATION: the plant library could not be loaded: {error}");
                field.enabled = false;
                field.settled = true;
                return;
            }
        }
    }
    let Some(catalog) = field.catalog.clone() else {
        return;
    };
    let Ok(player) = players.single() else {
        return;
    };
    let (px, pz) = (player.position.x as f64, player.position.z as f64);
    field.since_publish += time.delta_secs();

    // Collect finished levels.
    let store_level = |(key, plants): (([i64; 2], u8), Vec<PlantInstance>)| {
        field.chunks.entry(key.0).or_default()[key.1 as usize] = Some(Arc::new(plants));
        key
    };
    let finished: Vec<([i64; 2], u8)> = field
        .tasks
        .iter_mut()
        .filter_map(|(key, task)| block_on(poll_once(task)).map(|plants| (*key, plants)))
        .map(store_level)
        .collect();
    for key in &finished {
        field.tasks.remove(key);
    }
    field.dirty |= !finished.is_empty();

    // Drop what has fallen well behind; cancel work no longer wanted.
    let mut dropped = false;
    field.chunks.retain(|&chunk, levels| {
        let distance = chunk_distance(chunk, px, pz);
        for (level, slot) in levels.iter_mut().enumerate() {
            if slot.is_some() && distance > LEVEL_RADIUS[level] + LEVEL_HYSTERESIS {
                *slot = None;
                dropped = true;
            }
        }
        levels.iter().any(Option::is_some)
    });
    field.dirty |= dropped;
    field
        .tasks
        .retain(|&(chunk, level), _| chunk_distance(chunk, px, pz) <= LEVEL_RADIUS[level as usize] + LEVEL_HYSTERESIS);

    // Wanted levels, nearest first.
    let span = (LEVEL_RADIUS[0] / scatter::CHUNK_SIZE).ceil() as i64 + 1;
    let centre = scatter::chunk_of(px, pz);
    let mut wanted: Vec<(f64, [i64; 2], u8)> = Vec::new();
    for cz in centre[1] - span..=centre[1] + span {
        for cx in centre[0] - span..=centre[0] + span {
            let chunk = [cx, cz];
            let distance = chunk_distance(chunk, px, pz);
            for level in [LEVEL_TREES, LEVEL_SHRUBS, LEVEL_GROUND] {
                if distance > LEVEL_RADIUS[level as usize] {
                    continue;
                }
                let present = field
                    .chunks
                    .get(&chunk)
                    .is_some_and(|levels| levels[level as usize].is_some());
                if !present && !field.tasks.contains_key(&(chunk, level)) {
                    // Nearer detail first; trees before the ground layers.
                    wanted.push((distance + level as f64 * 40.0, chunk, level));
                }
            }
        }
    }
    wanted.sort_by(|a, b| a.0.total_cmp(&b.0));
    let pool = AsyncComputeTaskPool::get();
    for &(_, chunk, level) in wanted.iter().take(MAX_IN_FLIGHT.saturating_sub(field.tasks.len())) {
        let noise = field.noise.clone();
        let catalog = catalog.clone();
        let network = rivers.network().cloned();
        let task = pool.spawn(async move { scatter::generate_level(&noise, &catalog, network.as_deref(), chunk, level) });
        field.tasks.insert((chunk, level), task);
    }
    field.settled = wanted.is_empty() && field.tasks.is_empty();

    if field.dirty && (field.since_publish >= PUBLISH_INTERVAL || field.tasks.is_empty()) {
        field.publish();
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use scatter::{CatalogModel, Layer};

    /// Model 0 is an oak, 1 a bush and 2 a lilac.
    pub(crate) fn catalog() -> Catalog {
        let model = |species, trunk_radius| CatalogModel {
            species,
            form: String::new(),
            height: 8.0,
            crown_radius: 2.0,
            trunk_radius,
        };
        Catalog {
            models: vec![model(Species::Oak, 0.4), model(Species::Bush, 0.1), model(Species::Lilac, 0.2)],
        }
    }

    pub(crate) fn plant(x: f32, z: f32, model: u32, layer: Layer) -> PlantInstance {
        PlantInstance {
            position: [x, 0.0, z],
            scale: 1.0,
            model,
            layer: layer as u32,
            ..Default::default()
        }
    }

    #[test]
    fn solids_near_reads_only_trunks_in_reach_across_chunks() {
        let field = VegetationField::with_plants(
            catalog(),
            vec![
                plant(10.0, 0.0, 0, Layer::Canopy),
                plant(-1.5, 0.5, 0, Layer::Canopy),   // the chunk to the west
                plant(40.0, 0.0, 0, Layer::Canopy),   // out of reach
                plant(1.0, 1.0, 1, Layer::Shrub),     // a bush: walked through
                plant(0.0, -2.0, 2, Layer::Shrub),    // a lilac is solid
                plant(2.0, 2.0, 2, Layer::Herb),      // the ground layers hold nothing solid
            ],
        );
        let mut found = Vec::new();
        field.solids_near(Vec2::new(0.5, 0.0), 3.0, &mut found);
        let mut at: Vec<[f32; 2]> = found.iter().map(|solid| solid.position.to_array()).collect();
        at.sort_by(|a, b| a[0].total_cmp(&b[0]));
        assert_eq!(at, vec![[-1.5, 0.5], [0.0, -2.0]]);
        assert!(found.iter().any(|solid| solid.radius == 0.4));

        // Reach is to the stem's surface: a 0.4 m trunk 3.3 m away is touched.
        found.clear();
        field.solids_near(Vec2::new(10.0, 3.3), 3.0, &mut found);
        assert_eq!(found.len(), 1);
        found.clear();
        field.solids_near(Vec2::new(10.0, 3.5), 3.0, &mut found);
        assert!(found.is_empty());
    }

    /// The real library and scatter: every tree and lilac stands as a stem
    /// narrower than the gaps the scatter leaves between canopy trees, the
    /// chunk query finds exactly what a brute-force scan does, and the player
    /// can pass between any two neighbouring canopy trunks.
    #[test]
    fn the_scattered_forest_is_solid_and_walkable() {
        let assets = assets::VegetationAssets::load_geometry(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/models"),
        )
        .expect("plant geometry loads");
        let catalog = Catalog::from_assets(&assets);
        let noise = NoiseField::new();
        let mut plants = Vec::new();
        for chunk in [[-1, -1], [0, -1], [-1, 0], [0, 0]] {
            for level in [LEVEL_TREES, LEVEL_SHRUBS] {
                plants.extend(scatter::generate_level(&noise, &catalog, None, chunk, level));
            }
        }
        let solids: Vec<(collision::Solid, u32)> = plants
            .iter()
            .filter_map(|plant| collision::Solid::of(&catalog, plant).map(|solid| (solid, plant.layer)))
            .collect();
        assert!(solids.len() > 1000, "{}", solids.len());
        let widest = solids.iter().map(|(solid, _)| solid.radius).fold(0.0, f32::max);
        assert!(widest > 0.5 && widest < collision::MAX_TRUNK_RADIUS, "{widest}");
        assert!(solids.iter().any(|&(_, layer)| layer == Layer::Shrub as u32), "no lilacs among the solids");

        // Neighbouring canopy trees leave room between their bark.
        let canopy: Vec<&collision::Solid> = solids
            .iter()
            .filter(|&&(_, layer)| layer == Layer::Canopy as u32)
            .map(|(solid, _)| solid)
            .collect();
        for (i, a) in canopy.iter().enumerate() {
            for b in &canopy[i + 1..] {
                let gap = a.position.distance(b.position) - a.radius - b.radius;
                assert!(gap > 2.0 * collision::PLAYER_RADIUS, "trunks {gap:.2} m apart: {a:?} {b:?}");
            }
        }

        // The query agrees with a scan of everything, across chunk borders.
        let field = VegetationField::with_plants(catalog.clone(), plants);
        for centre in [Vec2::new(0.0, 0.0), Vec2::new(-3.0, 100.0), Vec2::new(-250.0, -255.0), Vec2::new(120.0, -40.0)] {
            let mut found = Vec::new();
            field.solids_near(centre, 12.0, &mut found);
            let expected = solids
                .iter()
                .filter(|(solid, _)| solid.position.distance(centre) <= 12.0 + solid.radius)
                .count();
            assert_eq!(found.len(), expected, "around {centre}");
        }
    }

    #[test]
    fn a_field_with_no_library_has_nothing_solid() {
        let field = VegetationField::new(Arc::new(NoiseField { samples: Vec::new() }), false);
        let mut found = Vec::new();
        field.solids_near(Vec2::ZERO, 100.0, &mut found);
        assert!(found.is_empty());
    }
}
