//! The rivers in the render world: the network extracted from the main
//! world, its carve segments, lookup grid and rocks uploaded for every
//! shader that shapes or seats on the ground, and the water surfaces.

use crate::render::gpu_textures::GpuWorldTexturesOption;
use crate::rivers::network::RiverNetwork;
use bevy::prelude::*;
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::render::{ExtractSchedule, MainWorld, Render, RenderSystems};
use std::sync::Arc;

/// The network in force, copied from the main world when it changes.
#[derive(Resource, Default)]
pub struct ExtractedRivers {
    pub network: Option<Arc<RiverNetwork>>,
    pub generation: u64,
    /// The generation whose data the GPU buffers hold.
    pub uploaded: u64,
    /// The water surfaces on the GPU.
    pub surface: Option<RiverSurfaceBuffers>,
}

pub struct RiverSurfaceBuffers {
    pub vertices: bevy::render::render_resource::Buffer,
    pub indices: bevy::render::render_resource::Buffer,
    pub chunks: Vec<crate::rivers::surface::SurfaceChunk>,
}

/// Rivers are drawn this far; beyond it a creek is a thread a pixel wide.
pub const RIVER_DRAW_DISTANCE: f32 = 3200.0;

impl RiverSurfaceBuffers {
    /// The chunks a view can see: near enough and not wholly outside one
    /// side of the frustum. `view_projection` is column-major.
    pub fn visible(&self, view_projection: &[f32; 16], eye: [f32; 3]) -> Vec<(u32, u32)> {
        let m = view_projection;
        let clip = |p: [f32; 3]| {
            [
                m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12],
                m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13],
                m[3] * p[0] + m[7] * p[1] + m[11] * p[2] + m[15],
            ]
        };
        let mut ranges: Vec<(u32, u32)> = Vec::new();
        for chunk in &self.chunks {
            let nearest = [
                eye[0].clamp(chunk.minimum[0], chunk.maximum[0]),
                eye[1].clamp(chunk.minimum[1], chunk.maximum[1]),
                eye[2].clamp(chunk.minimum[2], chunk.maximum[2]),
            ];
            let d = ((nearest[0] - eye[0]).powi(2) + (nearest[1] - eye[1]).powi(2) + (nearest[2] - eye[2]).powi(2)).sqrt();
            if d > RIVER_DRAW_DISTANCE {
                continue;
            }
            // Lifted distant surfaces stand a little above their bounds.
            let top = chunk.maximum[1] + 12.0;
            let mut outside = [true; 5];
            for corner in 0..8 {
                let p = [
                    if corner & 1 == 0 { chunk.minimum[0] } else { chunk.maximum[0] },
                    if corner & 2 == 0 { chunk.minimum[1] } else { top },
                    if corner & 4 == 0 { chunk.minimum[2] } else { chunk.maximum[2] },
                ];
                let [x, y, w] = clip(p);
                outside[0] &= x < -w;
                outside[1] &= x > w;
                outside[2] &= y < -w;
                outside[3] &= y > w;
                outside[4] &= w < 0.0;
            }
            if outside.iter().any(|&o| o) {
                continue;
            }
            // Neighbouring chunks are contiguous in the index buffer: merge.
            match ranges.last_mut() {
                Some(last) if last.0 + last.1 == chunk.first_index => last.1 += chunk.index_count,
                _ => ranges.push((chunk.first_index, chunk.index_count)),
            }
        }
        ranges
    }
}

fn extract_rivers(main_world: Res<MainWorld>, mut extracted: ResMut<ExtractedRivers>) {
    let Some(field) = main_world.get_resource::<crate::rivers::RiverField>() else {
        return;
    };
    if field.generation() != extracted.generation {
        extracted.generation = field.generation();
        extracted.network = field.network().cloned();
    }
}

/// Write a new network over the river buffers. The terrain revision moves
/// with it, so the grass habitat capture is redone over the new channels.
fn upload_rivers(
    textures: Res<GpuWorldTexturesOption>,
    mut rivers: ResMut<ExtractedRivers>,
    queue: Res<RenderQueue>,
    device: Res<RenderDevice>,
) {
    let Some(textures) = textures.0.as_deref() else {
        return;
    };
    if rivers.uploaded == rivers.generation {
        return;
    }
    rivers.uploaded = rivers.generation;
    rivers.surface = rivers.network.as_ref().and_then(|network| {
        let mesh = &network.surface;
        if mesh.indices.is_empty() {
            return None;
        }
        Some(RiverSurfaceBuffers {
            vertices: device.create_buffer_with_data(&wgpu::util::BufferInitDescriptor {
                label: Some("river_surface_vertices"),
                contents: bytemuck::cast_slice(&mesh.vertices),
                usage: wgpu::BufferUsages::VERTEX,
            }),
            indices: device.create_buffer_with_data(&wgpu::util::BufferInitDescriptor {
                label: Some("river_surface_indices"),
                contents: bytemuck::cast_slice(&mesh.indices),
                usage: wgpu::BufferUsages::INDEX,
            }),
            chunks: mesh.chunks.clone(),
        })
    });
    let Some(network) = rivers.network.as_ref() else {
        // No rivers: a zero resolution in the header switches them off.
        queue.write_buffer(&textures.river_grid, 0, bytemuck::cast_slice(&[0u32; 8]));
        textures.revision.note_atlas_upload();
        return;
    };
    let payload = network.gpu_payload();
    if !payload.segments.is_empty() {
        queue.write_buffer(&textures.river_segments, 0, bytemuck::cast_slice(&payload.segments));
    }
    if !payload.rocks.is_empty() {
        queue.write_buffer(&textures.river_rocks, 0, bytemuck::cast_slice(&payload.rocks));
    }
    queue.write_buffer(&textures.river_grid, 0, bytemuck::cast_slice(&payload.grid_words));
    textures.revision.note_atlas_upload();
    log::info!(
        "RIVERS: uploaded {} segments, {} rocks and {:.1} MiB of lookup grid",
        payload.segments.len(),
        payload.rocks.len(),
        payload.grid_words.len() as f64 * 4.0 / (1024.0 * 1024.0)
    );
}

pub fn register_river_systems(render_app: &mut bevy::app::SubApp) {
    render_app
        .init_resource::<ExtractedRivers>()
        .add_systems(ExtractSchedule, extract_rivers)
        .add_systems(
            Render,
            upload_rivers
                .in_set(RenderSystems::Prepare)
                .after(crate::render::gpu_textures::prepare_gpu_textures),
        );
}
