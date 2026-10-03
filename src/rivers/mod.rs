//! Rivers and creeks: where they run, how they shape the ground, and what
//! their water does.
//!
//! The main world owns the network. It is generated for a 16 km region
//! around the player (see [`network`]) on the async compute pool and
//! replaced when the player wanders far enough that a new region centre is
//! nearer; the region is regenerated whole, but every choice in it is keyed
//! by world position, so the rivers around the player come out the same.
//! The first region is built before anything streams, because the erosion
//! tiles, the vegetation and the player's footing all carve and avoid the
//! channels.
//!
//! The render world receives the finished network whenever it changes and
//! uploads its carve segments, lookup grid and rocks for the terrain, plant
//! and water shaders, plus the water-surface ribbons.

pub mod carve;
pub mod map;
pub mod network;
pub mod rocks;
pub mod surface;

use crate::noise::NoiseField;
use crate::player::Player;
use bevy::prelude::*;
use bevy::tasks::{AsyncComputeTaskPool, Task, block_on, poll_once};
use network::RiverNetwork;
use std::sync::Arc;

/// GPU capacities: the uploaded grid words, carve segments and rocks. A
/// network that needs more keeps what lies nearest its region's centre.
pub const GPU_GRID_WORDS: usize = 2_621_440;
pub const GPU_SEGMENTS: usize = 131_072;
pub const GPU_ROCKS: usize = 98_304;

/// What the GPU receives of a network.
pub struct GpuPayload {
    pub grid_words: Vec<u32>,
    pub segments: Vec<carve::RiverSegment>,
    pub rocks: Vec<carve::RockObstacle>,
}

impl RiverNetwork {
    pub fn gpu_payload(&self) -> GpuPayload {
        let centre = network::region_centre(self.region);
        let mut radius = f32::INFINITY;
        loop {
            let near = |p: [f32; 2]| {
                (p[0] - centre[0] as f32).hypot(p[1] - centre[1] as f32) <= radius
            };
            let segments: Vec<carve::RiverSegment> =
                self.segments.iter().filter(|s| near(s.a)).copied().collect();
            let rocks: Vec<carve::RockObstacle> = self
                .rocks
                .iter()
                .filter(|r| near(r.position))
                .map(|r| r.obstacle())
                .collect();
            let grid = if radius.is_finite() {
                let mut grid = carve::SegmentGrid::build(self.grid.origin, self.grid.resolution, &segments, &rocks);
                grid.lakes = self.grid.lakes.clone();
                grid
            } else {
                self.grid.clone()
            };
            let grid_words = grid.gpu_words(segments.len());
            if (segments.len() <= GPU_SEGMENTS && rocks.len() <= GPU_ROCKS && grid_words.len() <= GPU_GRID_WORDS)
                || radius < 1000.0
            {
                if radius.is_finite() {
                    log::warn!("RIVERS: the GPU holds the rivers within {radius:.0} m of the region centre");
                }
                let mut payload = GpuPayload { grid_words, segments, rocks };
                payload.segments.truncate(GPU_SEGMENTS);
                payload.rocks.truncate(GPU_ROCKS);
                payload.grid_words.truncate(GPU_GRID_WORDS);
                return payload;
            }
            radius = if radius.is_finite() { radius * 0.85 } else { (network::REGION_SIZE * 0.75) as f32 };
        }
    }
}

#[derive(Resource)]
pub struct RiverField {
    noise: Arc<NoiseField>,
    enabled: bool,
    current: Option<Arc<RiverNetwork>>,
    pending: Option<([i64; 2], Task<RiverNetwork>)>,
    generation: u64,
}

impl RiverField {
    pub fn new(noise: Arc<NoiseField>, enabled: bool) -> Self {
        Self {
            noise,
            enabled,
            current: None,
            pending: None,
            generation: 0,
        }
    }

    /// The network in force, if rivers are enabled and one is built.
    pub fn network(&self) -> Option<&Arc<RiverNetwork>> {
        self.current.as_ref()
    }

    /// Moves whenever the network is replaced.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Build the region around a point now, on this thread. Startup uses
    /// this so the first erosion tiles and plants already see the channels.
    pub fn build_blocking(&mut self, x: f32, z: f32) {
        if !self.enabled {
            return;
        }
        let region = network::region_of(x as f64, z as f64);
        if self.current.as_ref().is_some_and(|n| n.region == region) {
            return;
        }
        self.install(network::generate(&self.noise, region));
    }

    fn install(&mut self, network: RiverNetwork) {
        log::info!(
            "RIVERS: region ({}, {}) has {} rivers, {:.1} km of channel, {} waterfalls over 1.5 m and {} rocks, built in {:.2} s",
            network.region[0],
            network.region[1],
            network.rivers.len(),
            network.total_length_km(),
            network.waterfall_count(),
            network.rocks.len(),
            network.build_seconds
        );
        self.current = Some(Arc::new(network));
        self.generation += 1;
    }
}

/// Start building the next region once the player is nearer its centre, and
/// install it when done.
pub fn stream_rivers(mut field: ResMut<RiverField>, players: Query<&Player>) {
    if !field.enabled {
        return;
    }
    if let Some((_, task)) = field.pending.as_mut()
        && let Some(network) = block_on(poll_once(task))
    {
        field.pending = None;
        field.install(network);
    }
    let Ok(player) = players.single() else {
        return;
    };
    let wanted = network::region_of(player.position.x as f64, player.position.z as f64);
    let current = field.current.as_ref().map(|n| n.region);
    let pending = field.pending.as_ref().map(|p| p.0);
    if current == Some(wanted) || pending == Some(wanted) {
        return;
    }
    if current.is_none() {
        // Nothing to show yet: build it here rather than leave the world
        // without rivers for a frame.
        field.build_blocking(player.position.x, player.position.z);
        return;
    }
    let noise = field.noise.clone();
    let task = AsyncComputeTaskPool::get().spawn(async move { network::generate(&noise, wanted) });
    field.pending = Some((wanted, task));
}
