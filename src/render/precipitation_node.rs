//! Camera-near, world-anchored rain and snow rendered after the water passes.
//! The position G-buffer clips particles hidden by terrain and vegetation.

use super::{
    ExtractedForestView, ForestGlobals, ForestShaderHandles, globals_layout,
};
use crate::constants::SEA_LEVEL;
use bevy::prelude::*;
use bevy::render::render_resource::{
    BindGroup, BindGroupLayoutDescriptor, CachedRenderPipelineId, PipelineCache,
    RenderPipelineDescriptor,
};
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue, ViewQuery};
use bevy::render::view::ViewTarget;
use std::collections::HashMap;
use std::sync::Mutex;

// The smaller cells give heavy rain enough density while the shader culls
// particles before rasterization outside the local, depth-tested volume.
const RAIN_PARTICLES: u32 = 40 * 40 * 16;
const SNOW_PARTICLES: u32 = 24 * 24 * 12;

#[derive(Default)]
struct PrecipitationInner {
    globals_layout: Option<BindGroupLayoutDescriptor>,
    globals_group: Option<BindGroup>,
    screen_layout: Option<BindGroupLayoutDescriptor>,
    wind_layout: Option<BindGroupLayoutDescriptor>,
    wind_buffer: Option<wgpu::Buffer>,
    wind_group: Option<BindGroup>,
    pipelines: HashMap<wgpu::TextureFormat, (CachedRenderPipelineId, CachedRenderPipelineId)>,
}

#[derive(Resource, Default)]
pub struct PrecipitationNodeState {
    inner: Mutex<PrecipitationInner>,
}

fn prepare_precipitation(
    state: Res<PrecipitationNodeState>,
    globals: Res<ForestGlobals>,
    view: Res<ExtractedForestView>,
    water: Res<super::water_node::ExtractedWater>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    cache: Res<PipelineCache>,
) {
    let Some(globals_buffer) = globals.buffer.as_ref() else {
        return;
    };
    let Ok(mut inner) = state.inner.lock() else {
        return;
    };
    if inner.globals_layout.is_none() {
        let layout = globals_layout();
        let group = super::bind_group(
            &device,
            &cache,
            "forest_precipitation_globals",
            &layout,
            &[wgpu::BindGroupEntry {
                binding: 0,
                resource: globals_buffer.as_entire_binding(),
            }],
        );
        inner.globals_layout = Some(layout);
        inner.globals_group = Some(group);
    }
    if inner.screen_layout.is_none() {
        inner.screen_layout = Some(BindGroupLayoutDescriptor::new(
            "forest_precipitation_screen_layout",
            &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        ));
    }
    if inner.wind_layout.is_none() {
        let layout = BindGroupLayoutDescriptor::new(
            "forest_precipitation_wind_layout",
            &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(16),
                },
                count: None,
            }],
        );
        let buffer = device.wgpu_device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("forest_precipitation_wind"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let group = super::bind_group(
            &device,
            &cache,
            "forest_precipitation_wind_group",
            &layout,
            &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        );
        inner.wind_layout = Some(layout);
        inner.wind_buffer = Some(buffer);
        inner.wind_group = Some(group);
    }
    if let Some(buffer) = inner.wind_buffer.as_ref() {
        // Near-ground gusts are weaker than the cloud-level wind. A separate
        // uniform lets streaks lean with the current wind without adding an
        // unrelated field to every shader's shared GlobalUniforms ABI.
        let speed = view.settings.cloud_wind_speed * view.weather.current.wind_multiplier * 0.22;
        let direction = view.settings.cloud_wind_direction_degrees.to_radians();
        let wind = [
            speed * direction.cos(), speed * direction.sin(),
            water.draw as u32 as f32, 0.0,
        ];
        queue.write_buffer(buffer, 0, bytemuck::cast_slice(&wind));
    }
}

pub fn register_precipitation_systems(render_app: &mut bevy::app::SubApp) {
    render_app.init_resource::<PrecipitationNodeState>();
    render_app.add_systems(
        bevy::render::Render,
        prepare_precipitation
            .in_set(bevy::render::RenderSystems::Prepare)
            .after(super::prepare_forest_globals),
    );
}

pub fn forest_precipitation_pass(
    view: ViewQuery<&ViewTarget>,
    world: &World,
    mut ctx: RenderContext,
) {
    let view = view.into_inner();
    let (Some(state), Some(extracted), Some(shaders), Some(device), Some(cache), Some(terrain)) = (
        world.get_resource::<PrecipitationNodeState>(),
        world.get_resource::<ExtractedForestView>(),
        world.get_resource::<ForestShaderHandles>(),
        world.get_resource::<RenderDevice>(),
        world.get_resource::<PipelineCache>(),
        world.get_resource::<super::terrain_node::TerrainNodeState>(),
    ) else {
        return;
    };
    let ocean_visible = world
        .get_resource::<super::water_node::ExtractedWater>()
        .is_some_and(|water| water.draw);
    if ocean_visible && extracted.player_position[1] <= SEA_LEVEL + 0.2 {
        return;
    }
    // Rain bands can cross the near-camera particle volume before reaching
    // the player. Sample its corners as well as the local reading so their
    // leading edge remains visible and moves across the landscape smoothly.
    let nearby_precipitation = extracted.weather.local_precipitation.intensity() > 0.001
        || [-24.0, 0.0, 24.0].into_iter().any(|dx| {
            [-24.0, 0.0, 24.0].into_iter().any(|dz| {
                crate::weather::sample_precipitation(
                    [extracted.player_position[0] + dx, extracted.player_position[2] + dz],
                    extracted.player_position[1] - 8.0,
                    extracted.weather_offset,
                    extracted.weather.climate_bias,
                    extracted.weather.precipitation_bias,
                    extracted.weather.precipitation_override(),
                    extracted.settings.cloud_base_height,
                    extracted.weather.overrides.base,
                    extracted.settings.cloud_wind_direction_degrees.to_radians(),
                )
                .intensity()
                    > 0.001
            })
        });
    if !nearby_precipitation {
        return;
    }
    let Ok(mut inner) = state.inner.lock() else {
        return;
    };
    let format = view.main_texture_format();
    let (blit_id, particle_id) = if let Some(ids) = inner.pipelines.get(&format).copied() {
        ids
    } else {
        let (Some(globals_layout), Some(screen_layout), Some(wind_layout)) = (
            inner.globals_layout.clone(),
            inner.screen_layout.clone(),
            inner.wind_layout.clone(),
        ) else {
            return;
        };
        let layouts = vec![globals_layout, screen_layout, wind_layout];
        let blit = cache.queue_render_pipeline(pipeline_descriptor(
            "forest_precipitation_blit_pipeline",
            &shaders.precipitation,
            format,
            layouts.clone(),
            "vs_blit",
            "fs_blit",
            None,
            wgpu::PrimitiveTopology::TriangleList,
        ));
        let particles = cache.queue_render_pipeline(pipeline_descriptor(
            "forest_precipitation_particle_pipeline",
            &shaders.precipitation,
            format,
            layouts,
            "vs_particle",
            "fs_particle",
            Some(wgpu::BlendState::ALPHA_BLENDING),
            wgpu::PrimitiveTopology::TriangleStrip,
        ));
        inner.pipelines.insert(format, (blit, particles));
        (blit, particles)
    };
    let (Some(blit), Some(particles), Some(globals), Some(wind), Some(screen_layout)) = (
        cache.get_render_pipeline(blit_id),
        cache.get_render_pipeline(particle_id),
        inner.globals_group.as_ref(),
        inner.wind_group.as_ref(),
        inner.screen_layout.as_ref(),
    ) else {
        return;
    };

    let Ok(gbuffer_guard) = terrain.gbuffer.lock() else {
        return;
    };
    let Some(gbuffer) = gbuffer_guard.as_ref() else {
        return;
    };
    let post_process = view.post_process_write();
    let screen_group = super::bind_group(
        device,
        cache,
        "forest_precipitation_screen",
        screen_layout,
        &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(post_process.source),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&gbuffer.position_view),
            },
        ],
    );
    let mut pass = ctx.begin_tracked_render_pass(wgpu::RenderPassDescriptor {
        label: Some("forest_precipitation_pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: post_process.destination,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    pass.set_viewport(
        0.0,
        0.0,
        extracted.physical_width.max(1) as f32,
        extracted.physical_height.max(1) as f32,
        0.0,
        1.0,
    );
    pass.set_bind_group(0, globals, &[]);
    pass.set_bind_group(1, &screen_group, &[]);
    pass.set_bind_group(2, wind, &[]);
    pass.set_render_pipeline(blit);
    pass.draw(0..3, 0..1);
    pass.set_render_pipeline(particles);
    pass.draw(0..4, 0..RAIN_PARTICLES + SNOW_PARTICLES);
}

fn pipeline_descriptor(
    label: &'static str,
    shader: &Handle<Shader>,
    format: wgpu::TextureFormat,
    layout: Vec<BindGroupLayoutDescriptor>,
    vertex_entry: &'static str,
    fragment_entry: &'static str,
    blend: Option<wgpu::BlendState>,
    topology: wgpu::PrimitiveTopology,
) -> RenderPipelineDescriptor {
    RenderPipelineDescriptor {
        label: Some(label.into()),
        layout,
        immediate_size: 0,
        vertex: bevy::render::render_resource::VertexState {
            shader: shader.clone(),
            shader_defs: vec![],
            entry_point: Some(vertex_entry.into()),
            buffers: vec![],
        },
        primitive: wgpu::PrimitiveState {
            topology,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(bevy::render::render_resource::FragmentState {
            shader: shader.clone(),
            shader_defs: vec![],
            entry_point: Some(fragment_entry.into()),
            targets: vec![Some(wgpu::ColorTargetState {
                format,
                blend,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        zero_initialize_workgroup_memory: false,
    }
}
