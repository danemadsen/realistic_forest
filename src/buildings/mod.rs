//! Buildings: the wild-west kit from ~/buildings, placed over the world.
//!
//! The main world owns them. The kit's geometry and collision load at
//! startup (their footprints decide where they can stand) and its textures on
//! a background task. [`placement`] decides where every homestead and town
//! stands as a pure function of world position; the buildings within
//! [`STREAM_RADIUS`] of the player are kept here with their doors' states, and
//! the ground under them is levelled by their pads ([`pads`]), which the
//! erosion cache applies to every CPU height query. The render world receives
//! the library once, the placed buildings whenever they change, and the doors'
//! angles every frame (src/render/building_node.rs).

pub mod assets;
pub mod collision;
pub mod map;
pub mod pads;
pub mod placement;

use crate::noise::NoiseField;
use crate::player::Player;
use crate::vegetation::assets::PreparedTexture;
use assets::BuildingLibrary;
use bevy::math::Vec2;
use bevy::prelude::*;
use bevy::tasks::{AsyncComputeTaskPool, Task, block_on, poll_once};
use collision::{DoorState, NearBuilding};
use pads::PadIndex;
use placement::{Placement, SettlementPlanner};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// Buildings are kept, and their ground levelled, this far around the
/// player: past the farthest they are drawn.
pub const STREAM_RADIUS: f32 = 2400.0;
/// The kept set is refreshed when the player moves this far.
const STREAM_STEP: f32 = 256.0;

/// How far the eye has adapted to the light it stands in: 1 outdoors, more
/// inside a building, eased in and out over a second or so, as stepping in
/// from the sun does. The renderer multiplies its exposure by it.
#[derive(Resource)]
pub struct EyeAdaptation {
    pub exposure: f32,
}

impl Default for EyeAdaptation {
    fn default() -> Self {
        Self { exposure: 1.0 }
    }
}

/// The exposure the eye settles to fully indoors: rooms lit only through
/// their windows are several stops darker than the day outside.
const INDOOR_EXPOSURE: f32 = 3.2;
/// Seconds the eye takes to adapt, coming in and going out.
const ADAPT_IN_SECONDS: f32 = 1.4;
const ADAPT_OUT_SECONDS: f32 = 0.6;

/// What the menu says of the nearest building.
pub struct NearestBuilding {
    pub name: String,
    pub distance: f32,
    /// World XZ a few metres out from its front door, and the way to face it.
    pub approach: Vec2,
    pub facing: Vec2,
}

/// Every placed building the render world draws, with a generation that
/// moves whenever the set does.
pub struct BuildingSnapshot {
    pub generation: u64,
    pub placements: Vec<Placement>,
}

/// The settlements around a streaming centre, planned off the main thread:
/// the buildings and the towns' street pads.
type SettlementTask = Task<(Vec<Placement>, Vec<pads::Pad>)>;

#[derive(Resource)]
pub struct BuildingField {
    enabled: bool,
    library: Option<Arc<BuildingLibrary>>,
    planner: Option<Arc<SettlementPlanner>>,
    /// The kit's textures, waiting for the task pools to exist, then loading.
    texture_keys: Option<Vec<assets::TextureKey>>,
    loading_textures: Option<Task<Result<Vec<PreparedTexture>, String>>>,
    textures: Option<Arc<Vec<PreparedTexture>>>,
    /// The textures could not be loaded: the buildings stand, solid and on
    /// their pads, but are not drawn.
    textures_failed: bool,
    snapshot: Arc<BuildingSnapshot>,
    pads: Arc<PadIndex>,
    /// Door states of each placed building, in the snapshot's order, each
    /// in its model's door order.
    doors: Vec<Vec<DoorState>>,
    /// Buildings by 64 m cell of their pad centre, indices into the snapshot.
    cells: HashMap<[i32; 2], Vec<usize>>,
    streaming: Option<([i64; 2], SettlementTask)>,
    streamed: Option<[i64; 2]>,
    /// `--open-doors`: every door stands open and stays so.
    doors_held_open: bool,
    /// The snapshot generation the render world has on the GPU.
    pub uploaded: Arc<AtomicU64>,
}

fn stream_cell(position: Vec2) -> [i64; 2] {
    [(position.x / STREAM_STEP).floor() as i64, (position.y / STREAM_STEP).floor() as i64]
}

fn stream_area(cell: [i64; 2]) -> (Vec2, Vec2) {
    let centre = (Vec2::new(cell[0] as f32, cell[1] as f32) + Vec2::splat(0.5)) * STREAM_STEP;
    (centre - Vec2::splat(STREAM_RADIUS), centre + Vec2::splat(STREAM_RADIUS))
}

fn grid_cell(point: Vec2) -> [i32; 2] {
    [(point.x / 64.0).floor() as i32, (point.y / 64.0).floor() as i32]
}

impl BuildingField {
    /// Load the kit from `models_directory` and place the buildings around
    /// `spawn` before anything streams, so the first frames already stand on
    /// their pads. A missing or broken kit leaves the world without buildings.
    pub fn new(
        noise: Arc<NoiseField>,
        rivers: Option<&Arc<crate::rivers::network::RiverNetwork>>,
        models_directory: &std::path::Path,
        spawn: Vec2,
        enabled: bool,
    ) -> Self {
        let mut field = Self {
            enabled,
            library: None,
            planner: None,
            texture_keys: None,
            loading_textures: None,
            textures: None,
            textures_failed: false,
            snapshot: Arc::new(BuildingSnapshot { generation: 0, placements: Vec::new() }),
            pads: Arc::new(PadIndex::default()),
            doors: Vec::new(),
            cells: HashMap::new(),
            streaming: None,
            streamed: None,
            doors_held_open: false,
            uploaded: Arc::new(AtomicU64::new(0)),
        };
        if !enabled {
            return field;
        }
        let start = std::time::Instant::now();
        let library = match BuildingLibrary::load(models_directory) {
            Ok(library) => Arc::new(library),
            Err(error) => {
                println!("BUILDINGS: the kit could not be loaded: {error}");
                field.enabled = false;
                return field;
            }
        };
        let footprints = placement::Footprint::from_library(&library);
        let planner = Arc::new(SettlementPlanner::new(noise, footprints));
        let cell = stream_cell(spawn);
        let (minimum, maximum) = stream_area(cell);
        let placements = planner.placements_in(rivers, minimum, maximum);
        let streets = planner.streets_in(rivers, minimum, maximum);
        // Bevy's logger does not exist yet; the summary goes straight out.
        println!(
            "BUILDINGS: {} models, {} triangles; {} buildings within {:.0} m of spawn, placed in {:.2} s",
            library.models.len(),
            library.triangle_count(),
            placements.len(),
            STREAM_RADIUS,
            start.elapsed().as_secs_f32()
        );
        // The task pools do not exist before the app does: the textures start
        // loading on the first update.
        field.texture_keys = Some(library.texture_keys.clone());
        field.library = Some(library);
        field.planner = Some(planner);
        field.install(placements, streets);
        field.streamed = Some(cell);
        field
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// Open every door and keep it open, for captures that show the doors
    /// swung on their hinges.
    pub fn hold_doors_open(&mut self) {
        self.doors_held_open = true;
        let Some(library) = self.library.clone() else {
            return;
        };
        for (placement, states) in self.snapshot.placements.iter().zip(self.doors.iter_mut()) {
            let model = &library.models[placement.model as usize];
            for (door, state) in model.doors.iter().zip(states.iter_mut()) {
                *state = DoorState::held_open(door);
            }
        }
    }

    pub fn library(&self) -> Option<&Arc<BuildingLibrary>> {
        self.library.as_ref()
    }

    pub fn textures(&self) -> Option<&Arc<Vec<PreparedTexture>>> {
        self.textures.as_ref()
    }

    pub fn planner(&self) -> Option<&Arc<SettlementPlanner>> {
        self.planner.as_ref()
    }

    pub fn snapshot(&self) -> &Arc<BuildingSnapshot> {
        &self.snapshot
    }

    pub fn pads(&self) -> &Arc<PadIndex> {
        &self.pads
    }


    /// Whether everything the player could see is placed and on the GPU,
    /// textures included: screenshot runs wait for this.
    pub fn ready(&self) -> bool {
        if !self.enabled {
            return true;
        }
        if self.textures_failed {
            return self.streaming.is_none();
        }
        self.streaming.is_none()
            && self.texture_keys.is_none()
            && self.loading_textures.is_none()
            && self.uploaded.load(Ordering::Acquire) == self.snapshot.generation
    }

    fn install(&mut self, placements: Vec<Placement>, streets: Vec<pads::Pad>) {
        let library = self.library.clone();
        // Doors keep their swing across a restream.
        let old_placements = &self.snapshot.placements;
        let mut kept: HashMap<u64, Vec<DoorState>> = old_placements
            .iter()
            .map(|placement| placement.id)
            .zip(std::mem::take(&mut self.doors))
            .collect();
        let mut doors = Vec::with_capacity(placements.len());
        let mut cells: HashMap<[i32; 2], Vec<usize>> = HashMap::new();
        for (index, placement) in placements.iter().enumerate() {
            let model = library.as_ref().map(|library| &library.models[placement.model as usize]);
            let door_count = model.map_or(0, |model| model.doors.len());
            let mut states = kept.remove(&placement.id).unwrap_or_else(|| vec![DoorState::default(); door_count]);
            if self.doors_held_open
                && let Some(model) = model
            {
                for (door, state) in model.doors.iter().zip(states.iter_mut()) {
                    *state = DoorState::held_open(door);
                }
            }
            doors.push(states);
            cells.entry(grid_cell(placement.pad.centre)).or_default().push(index);
        }
        let mut all_pads: Vec<pads::Pad> = placements.iter().map(|placement| placement.pad).collect();
        all_pads.extend(streets);
        self.pads = Arc::new(PadIndex::new(all_pads));
        self.doors = doors;
        self.cells = cells;
        self.snapshot = Arc::new(BuildingSnapshot {
            generation: self.snapshot.generation + 1,
            placements,
        });
    }

    /// The placed buildings whose bounding circle comes within `reach` of a
    /// world point, with their doors' states, for collision.
    pub fn near_mut(&mut self, point: Vec2, reach: f32) -> Vec<NearBuilding<'_>> {
        let Some(library) = self.library.as_deref() else {
            return Vec::new();
        };
        let search = reach + 40.0;
        let first = grid_cell(point - Vec2::splat(search));
        let last = grid_cell(point + Vec2::splat(search));
        let mut indices = Vec::new();
        for cz in first[1]..=last[1] {
            for cx in first[0]..=last[0] {
                let Some(list) = self.cells.get(&[cx, cz]) else {
                    continue;
                };
                for &index in list {
                    let placement = &self.snapshot.placements[index];
                    let model = &library.models[placement.model as usize];
                    if placement.pad.centre.distance(point) <= reach + model.radius() {
                        indices.push(index);
                    }
                }
            }
        }
        indices.sort();
        let placements = &self.snapshot.placements;
        let mut near = Vec::with_capacity(indices.len());
        for (index, states) in self.doors.iter_mut().enumerate() {
            if indices.binary_search(&index).is_err() {
                continue;
            }
            let placement = &placements[index];
            near.push(NearBuilding {
                model: &library.models[placement.model as usize],
                frame: placement.frame(),
                doors: states.as_mut_slice(),
            });
        }
        near
    }

    /// Doors that are not shut, as (snapshot index, door index, angle).
    pub fn open_doors(&self) -> Vec<(u32, u16, f32)> {
        let mut open = Vec::new();
        for (building, states) in self.doors.iter().enumerate() {
            for (index, state) in states.iter().enumerate() {
                if !state.is_closed() {
                    open.push((building as u32, index as u16, state.angle));
                }
            }
        }
        open
    }

    /// The nearest building to a world point, for the menu: its model's
    /// name, how far away it is, and a spot a few metres out from its front
    /// door with the way to face it.
    /// The nearest placed building to `point` of those `keep` accepts (the
    /// ones drawn, say), with where to stand at its front door.
    pub fn nearest(&self, point: Vec2, keep: impl Fn(&Placement) -> bool) -> Option<NearestBuilding> {
        let library = self.library.as_deref()?;
        let distance_to = |placement: &&Placement| placement.pad.centre.distance(point);
        let nearest = self
            .snapshot
            .placements
            .iter()
            .filter(|placement| keep(placement))
            .min_by(|a, b| distance_to(a).total_cmp(&distance_to(b)))?;
        let model = &library.models[nearest.model as usize];
        let frame = nearest.frame();
        let door = model.doors.iter().find(|door| door.name == "front-door").or_else(|| model.doors.first())?;
        let hinge = frame.to_world(door.hinge);
        let outward = frame.turn(door.outward);
        let approach = Vec2::new(hinge.x, hinge.z) + outward * 6.0;
        Some(NearestBuilding {
            name: model.name.clone(),
            distance: nearest.pad.centre.distance(point),
            approach,
            facing: -outward,
        })
    }

    pub fn building_count(&self) -> usize {
        self.snapshot.placements.len()
    }
}

/// Stream the placed buildings around the player, finish loading the
/// textures, and let the doors swing.
pub fn stream_buildings(
    mut field: ResMut<BuildingField>,
    rivers: Res<crate::rivers::RiverField>,
    mut erosion: ResMut<crate::erosion::ErosionCache>,
    players: Query<&Player>,
    time: Res<Time>,
) {
    if !field.enabled {
        return;
    }
    let field = &mut *field;
    if let Some(keys) = field.texture_keys.take() {
        let load = async move { assets::prepare_textures(&keys) };
        field.loading_textures = Some(AsyncComputeTaskPool::get().spawn(load));
    }
    if let Some(task) = field.loading_textures.as_mut()
        && let Some(result) = block_on(poll_once(task))
    {
        field.loading_textures = None;
        match result {
            Ok(textures) => {
                let bytes: usize = textures.iter().map(PreparedTexture::byte_size).sum();
                log::info!("BUILDINGS: {} textures ({:.0} MiB) prepared", textures.len(), bytes as f64 / 1048576.0);
                field.textures = Some(Arc::new(textures));
            }
            Err(error) => {
                log::error!("BUILDINGS: the kit's textures could not be loaded: {error}");
                field.textures_failed = true;
            }
        }
    }
    if let Some((cell, task)) = field.streaming.as_mut()
        && let Some((placements, streets)) = block_on(poll_once(task))
    {
        let cell = *cell;
        field.streaming = None;
        field.install(placements, streets);
        field.streamed = Some(cell);
    }
    // The ground the erosion cache hands every CPU height query is levelled
    // by the pads in force.
    let current = erosion.buildings.as_ref().map(Arc::as_ptr);
    if current != Some(Arc::as_ptr(&field.pads)) {
        erosion.buildings = Some(field.pads.clone());
    }

    let dt = time.delta_secs().min(0.05);
    if let Some(library) = field.library.clone() {
        let placements = &field.snapshot.placements;
        for (placement, states) in placements.iter().zip(field.doors.iter_mut()) {
            let model = &library.models[placement.model as usize];
            for (door, state) in model.doors.iter().zip(states.iter_mut()) {
                state.advance(door, dt);
            }
        }
    }

    let Ok(player) = players.single() else {
        return;
    };
    if crate::planet::systems::streams_paused(player) {
        return;
    }
    let cell = stream_cell(Vec2::new(player.position.x, player.position.z));
    if field.streamed == Some(cell) || field.streaming.as_ref().is_some_and(|(pending, _)| *pending == cell) {
        return;
    }
    let Some(planner) = field.planner.clone() else {
        return;
    };
    let network = rivers.network().cloned();
    let (minimum, maximum) = stream_area(cell);
    let plan = async move {
        let placements = planner.placements_in(network.as_ref(), minimum, maximum);
        let streets = planner.streets_in(network.as_ref(), minimum, maximum);
        (placements, streets)
    };
    let task = AsyncComputeTaskPool::get().spawn(plan);
    field.streaming = Some((cell, task));
}

/// Let the eye adapt to the room the player stands in.
pub fn adapt_eye(
    mut field: ResMut<BuildingField>,
    mut adaptation: ResMut<EyeAdaptation>,
    players: Query<&Player>,
    time: Res<Time>,
    erosion: Option<Res<crate::erosion::ErosionCache>>,
) {
    let Ok(player) = players.single() else {
        return;
    };
    let point = Vec2::new(player.position.x, player.position.z);
    let feet = player.position.y - crate::constants::EYE_HEIGHT;
    let mut near = field.near_mut(point, 2.0);
    // Only buildings the GPU draws shade the eye.
    if let Some(erosion) = erosion.as_deref() {
        near.retain(|building| {
            crate::planet::height::features_whole(erosion, building.frame.origin.x, building.frame.origin.z)
        });
    }
    let enclosed = collision::enclosure(&near, point, feet);
    let target = 1.0 + (INDOOR_EXPOSURE - 1.0) * enclosed;
    let seconds = if target > adaptation.exposure { ADAPT_IN_SECONDS } else { ADAPT_OUT_SECONDS };
    let ease = 1.0 - (-time.delta_secs() / seconds).exp();
    adaptation.exposure += (target - adaptation.exposure) * ease;
}

/// The footprints of the buildings around a plant chunk, for the scatter
/// to keep clear of.
pub fn clearings_around(
    planner: &SettlementPlanner,
    rivers: Option<&Arc<crate::rivers::network::RiverNetwork>>,
    chunk: [i64; 2],
) -> Vec<crate::vegetation::scatter::Clearing> {
    let (minimum, maximum) = crate::vegetation::scatter::chunk_bounds(chunk);
    let margin = crate::vegetation::scatter::CLEARING_CONTEXT;
    let minimum = Vec2::new((minimum[0] - margin) as f32, (minimum[1] - margin) as f32);
    let maximum = Vec2::new((maximum[0] + margin) as f32, (maximum[1] + margin) as f32);
    let placements = planner.placements_in(rivers, minimum, maximum);
    let footprints = planner.footprints();
    let clearing_of = |placement: &Placement| crate::vegetation::scatter::Clearing {
        centre: placement.pad.centre.to_array(),
        axis: placement.pad.axis.to_array(),
        half: footprints[placement.model as usize].half.to_array(),
    };
    let mut clearings: Vec<crate::vegetation::scatter::Clearing> = placements.iter().map(clearing_of).collect();
    // Nothing grows in a town's main street either.
    let streets = planner.streets_in(rivers, minimum, maximum);
    let street_clearing = |street: &pads::Pad| crate::vegetation::scatter::Clearing {
        centre: street.centre.to_array(),
        axis: street.axis.to_array(),
        half: street.half.to_array(),
    };
    clearings.extend(streets.iter().map(street_clearing));
    clearings
}
