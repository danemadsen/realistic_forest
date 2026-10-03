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
        // Boulders break a river's surface well into the distance.
        Species::Rock => RenderProfile {
            max_distance: 520.0,
            lod_end: [f32::INFINITY; 3],
            ground_habitat: false,
            translucency: 0.0,
            shadow_cascades: 2,
        },
    }
}

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
    /// The river network's boulders, as plants of the rock models, and the
    /// network generation they came from.
    rocks: Arc<Vec<PlantInstance>>,
    rock_generation: u64,
}

impl VegetationField {
    pub fn new(noise: Arc<NoiseField>, enabled: bool) -> Self {
        let loading = enabled.then(|| {
            AsyncComputeTaskPool::get().spawn(async move {
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
            })
        });
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
            rocks: Arc::new(Vec::new()),
            rock_generation: 0,
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
        plants.extend_from_slice(&self.rocks);
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

/// The river network's boulders as instances of the rock models.
fn river_rocks(assets: &VegetationAssets, network: &crate::rivers::network::RiverNetwork) -> Vec<PlantInstance> {
    let models: Vec<Option<u32>> = (0..crate::rivers::rocks::ROCK_MODELS.len())
        .map(|k| {
            let name = format!("rock-{}", k + 1);
            assets.models.iter().position(|m| m.name == name).map(|i| i as u32)
        })
        .collect();
    network
        .rocks
        .iter()
        .filter_map(|rock| {
            Some(PlantInstance {
                position: [rock.position[0], rock.bed, rock.position[1]],
                scale: rock.scale,
                yaw: rock.yaw,
                seed: rock.seed,
                model: models.get(rock.model as usize).copied().flatten()?,
                layer: scatter::Layer::Rock as u32,
            })
        })
        .collect()
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
    if field.rock_generation != rivers.generation() {
        field.rock_generation = rivers.generation();
        field.rocks = Arc::new(match (field.assets.as_ref(), rivers.network()) {
            (Some(assets), Some(network)) => river_rocks(assets, network),
            _ => Vec::new(),
        });
        field.dirty = true;
    }
    let Ok(player) = players.single() else {
        return;
    };
    let (px, pz) = (player.position.x as f64, player.position.z as f64);
    field.since_publish += time.delta_secs();

    // Collect finished levels.
    let finished: Vec<([i64; 2], u8)> = field
        .tasks
        .iter_mut()
        .filter_map(|(key, task)| block_on(poll_once(task)).map(|plants| (*key, plants)))
        .map(|(key, plants)| {
            field.chunks.entry(key.0).or_default()[key.1 as usize] = Some(Arc::new(plants));
            key
        })
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
