//! Bounded, instanced vegetation in the terrain G-buffer. No entity per tuft,
//! CPU erosion readback, or alternative approximation of the biome rules.
use super::{ExtractedForestView, ForestGlobals, ForestShaderHandles, terrain_node};
use crate::grass::{self, GrassInstance, GrassTexture, SharedGrassAssets};
use bevy::mesh::VertexBufferLayout;
use bevy::prelude::*;
use bevy::render::render_resource::{
    BindGroup, BindGroupLayoutDescriptor, Buffer, CachedRenderPipelineId, FragmentState,
    PipelineCache, RenderPipelineDescriptor, VertexState,
};
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue};
use std::sync::Mutex;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GrassFrame {
    mapping: [f32; 4],
    wind: [f32; 4],
    range: [f32; 4],
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MaterialUniform {
    colour: [f32; 4],
    pbr: [f32; 4],
    shape: [f32; 4],
}
const _: () = assert!(std::mem::size_of::<GrassFrame>() == 48);
const _: () = assert!(std::mem::size_of::<MaterialUniform>() == 48);

struct GrassMesh {
    vertices: Buffer,
    indices: Buffer,
    index_count: u32,
    material: BindGroup,
    instances: Buffer,
    instance_count: u32,
}
struct GrassResources {
    pipeline: CachedRenderPipelineId,
    globals: BindGroup,
    frame_buffer: Buffer,
    frame_layout: BindGroupLayoutDescriptor,
    meshes: Vec<GrassMesh>,
    anchor: Option<[i64; 2]>,
}
#[derive(Resource, Default)]
pub struct GrassNodeState(Mutex<Option<GrassResources>>);

fn texture_entry(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}
fn sampler_entry(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}
fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: wgpu::BufferSize::new(48),
        },
        count: None,
    }
}
fn upload_texture(
    device: &RenderDevice,
    queue: &RenderQueue,
    source: &GrassTexture,
    srgb: bool,
    cutoff: Option<f32>,
) -> wgpu::TextureView {
    let levels = source.mip_chain(srgb, cutoff);
    let texture = device
        .wgpu_device()
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("grass_material_texture"),
            size: wgpu::Extent3d {
                width: source.width,
                height: source.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: levels.len() as u32,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: if srgb {
                wgpu::TextureFormat::Rgba8UnormSrgb
            } else {
                wgpu::TextureFormat::Rgba8Unorm
            },
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
    for (level, mip) in levels.iter().enumerate() {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: level as u32,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &mip.rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(mip.width * 4),
                rows_per_image: Some(mip.height),
            },
            wgpu::Extent3d {
                width: mip.width,
                height: mip.height,
                depth_or_array_layers: 1,
            },
        );
    }
    texture.create_view(&Default::default())
}

fn prepare_grass(
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    cache: Res<PipelineCache>,
    globals: Res<ForestGlobals>,
    shaders: Res<ForestShaderHandles>,
    assets: Option<Res<SharedGrassAssets>>,
    state: Res<GrassNodeState>,
) {
    let mut state = state.0.lock().unwrap_or_else(|e| e.into_inner());
    if state.is_some() {
        return;
    }
    let (Some(assets), Some(global_buffer)) = (assets, globals.buffer.as_ref()) else {
        return;
    };
    let frame_layout = BindGroupLayoutDescriptor::new(
        "grass_frame_layout",
        &[
            texture_entry(0, wgpu::ShaderStages::VERTEX),
            sampler_entry(1, wgpu::ShaderStages::VERTEX),
            uniform_entry(2),
            texture_entry(3, wgpu::ShaderStages::VERTEX),
        ],
    );
    let material_layout = BindGroupLayoutDescriptor::new(
        "grass_material_layout",
        &[
            texture_entry(0, wgpu::ShaderStages::FRAGMENT),
            texture_entry(1, wgpu::ShaderStages::FRAGMENT),
            texture_entry(2, wgpu::ShaderStages::FRAGMENT),
            sampler_entry(3, wgpu::ShaderStages::FRAGMENT),
            uniform_entry(4),
        ],
    );
    let global_layout = super::globals_layout();
    let global_group = super::bind_group(
        &device,
        &cache,
        "grass_globals",
        &global_layout,
        &super::globals_bind_group_entries(global_buffer),
    );
    let frame_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("grass_frame"),
        size: 48,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let sampler = device
        .wgpu_device()
        .create_sampler(&wgpu::SamplerDescriptor {
            label: Some("grass_material_sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            anisotropy_clamp: 8,
            ..Default::default()
        });
    // The nine short grass models supply this dense meadow.
    let active_models: Vec<_> = assets
        .0
        .models
        .iter()
        .filter(|model| model.name.starts_with("grass-"))
        .collect();
    let textures: Vec<_> = assets
        .0
        .materials
        .iter()
        .enumerate()
        .map(|(index, m)| {
            active_models
                .iter()
                .any(|model| model.material == index)
                .then(|| {
                    [
                        upload_texture(
                            &device,
                            &queue,
                            &m.base_color,
                            true,
                            Some(m.alpha_cutoff / m.base_color_factor[3].max(0.001)),
                        ),
                        upload_texture(&device, &queue, &m.normal, false, None),
                        upload_texture(&device, &queue, &m.orm, false, None),
                    ]
                })
        })
        .collect();
    let meshes = active_models
        .iter()
        .map(|model| {
            let m = &assets.0.materials[model.material];
            let params = MaterialUniform {
                colour: m.base_color_factor,
                pbr: [
                    m.alpha_cutoff,
                    m.normal_scale,
                    m.metallic_factor,
                    m.roughness_factor,
                ],
                shape: [
                    model.height,
                    model.radius,
                    m.occlusion_strength,
                    m.base_color
                        .visible_mean_luminance(m.base_color_factor, m.alpha_cutoff),
                ],
            };
            let buffer = device.create_buffer_with_data(&wgpu::util::BufferInitDescriptor {
                label: Some("grass_material_uniform"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let views = textures[model.material]
                .as_ref()
                .expect("active grass material uploaded");
            let material = super::bind_group(
                &device,
                &cache,
                "grass_material",
                &material_layout,
                &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&views[0]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&views[1]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&views[2]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: buffer.as_entire_binding(),
                    },
                ],
            );
            GrassMesh {
                vertices: device.create_buffer_with_data(&wgpu::util::BufferInitDescriptor {
                    label: Some(&model.name),
                    contents: bytemuck::cast_slice(&model.vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                }),
                indices: device.create_buffer_with_data(&wgpu::util::BufferInitDescriptor {
                    label: Some("grass_indices"),
                    contents: bytemuck::cast_slice(&model.indices),
                    usage: wgpu::BufferUsages::INDEX,
                }),
                index_count: model.indices.len() as u32,
                material,
                instances: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("grass_instances_empty"),
                    size: 32,
                    usage: wgpu::BufferUsages::VERTEX,
                    mapped_at_creation: false,
                }),
                instance_count: 0,
            }
        })
        .collect();
    let vertex_attributes =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x2, 3 => Float32x4];
    let instance_attributes = wgpu::vertex_attr_array![4 => Float32x2, 5 => Float32, 6 => Float32, 7 => Float32, 8 => Float32, 9 => Float32x2];
    let pipeline = cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("forest_grass_pipeline".into()),
        layout: vec![global_layout, frame_layout.clone(), material_layout],
        immediate_size: 0,
        vertex: VertexState {
            shader: shaders.grass.clone(),
            shader_defs: vec![],
            entry_point: Some("vs_main".into()),
            buffers: vec![
                VertexBufferLayout {
                    array_stride: 48,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: vertex_attributes.to_vec(),
                },
                VertexBufferLayout {
                    array_stride: std::mem::size_of::<GrassInstance>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: instance_attributes.to_vec(),
                },
            ],
        },
        primitive: wgpu::PrimitiveState {
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::GreaterEqual),
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: Default::default(),
        fragment: Some(FragmentState {
            shader: shaders.grass.clone(),
            shader_defs: vec![],
            entry_point: Some("fs_main".into()),
            targets: [
                wgpu::TextureFormat::Rgba32Float,
                wgpu::TextureFormat::Rgba16Float,
                wgpu::TextureFormat::Rgba8Unorm,
            ]
            .map(|format| {
                Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })
            })
            .to_vec(),
        }),
        zero_initialize_workgroup_memory: false,
    });
    *state = Some(GrassResources {
        pipeline,
        globals: global_group,
        frame_buffer,
        frame_layout,
        meshes,
        anchor: None,
    });
}

pub fn forest_grass_pass(world: &World, mut ctx: RenderContext) {
    let Some(state) = world.get_resource::<GrassNodeState>() else {
        return;
    };
    let Some(terrain) = world.get_resource::<terrain_node::TerrainNodeState>() else {
        return;
    };
    let Some(view) = world.get_resource::<ExtractedForestView>() else {
        return;
    };
    let Some(device) = world.get_resource::<RenderDevice>() else {
        return;
    };
    let Some(queue) = world.get_resource::<RenderQueue>() else {
        return;
    };
    let Some(cache) = world.get_resource::<PipelineCache>() else {
        return;
    };
    let gbuffer = terrain.gbuffer.lock().unwrap_or_else(|e| e.into_inner());
    let Some(gbuffer) = gbuffer.as_ref().filter(|g| g.grass_habitat_ready) else {
        return;
    };
    let mut guard = state.0.lock().unwrap_or_else(|e| e.into_inner());
    let Some(resources) = guard.as_mut() else {
        return;
    };
    let Some(pipeline) = cache.get_render_pipeline(resources.pipeline) else {
        return;
    };
    let center = [view.player_position[0], view.player_position[2]];
    let anchor = grass::scatter_anchor(center);
    if resources.anchor != Some(anchor) {
        let candidates = grass::scatter_grass(center, resources.meshes.len());
        for (mesh, instances) in resources.meshes.iter_mut().zip(candidates) {
            mesh.instance_count = instances.len() as u32;
            if !instances.is_empty() {
                mesh.instances =
                    device.create_buffer_with_data(&wgpu::util::BufferInitDescriptor {
                        label: Some("grass_instances"),
                        contents: bytemuck::cast_slice(&instances),
                        usage: wgpu::BufferUsages::VERTEX,
                    });
            }
        }
        resources.anchor = Some(anchor);
    }
    let elapsed = world
        .get_resource::<super::water_node::ExtractedWater>()
        .map_or(0.0, |w| w.elapsed);
    let direction = view.settings.cloud_wind_direction_degrees.to_radians();
    let conditions = view.weather.conditions(&view.settings);
    // Storm gusts thrash the grass well beyond the steady breeze.
    let gust = 1.0 + 1.6 * conditions.gust_strength;
    let frame = GrassFrame {
        mapping: terrain_node::grass_habitat_mapping(view.player_position),
        wind: [
            direction.cos(),
            direction.sin(),
            elapsed,
            (conditions.wind_speed / 18.0).clamp(0.0, 2.0) * 0.11 * gust,
        ],
        range: [100.0, grass::SCATTER_RADIUS, 0.0, 0.0],
    };
    queue.write_buffer(&resources.frame_buffer, 0, bytemuck::bytes_of(&frame));
    // Rebinding the capture view also handles window resize without stale views.
    let sampler = device
        .wgpu_device()
        .create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
    let frame_group = super::bind_group(
        device,
        cache,
        "grass_frame",
        &resources.frame_layout,
        &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&gbuffer.grass_habitat_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: resources.frame_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(&gbuffer.grass_ground_average_view),
            },
        ],
    );
    let attachments = [
        &gbuffer.position_view,
        &gbuffer.normal_view,
        &gbuffer.albedo_view,
    ]
    .map(|target| {
        Some(wgpu::RenderPassColorAttachment {
            view: target,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            },
        })
    });
    let mut pass = ctx.begin_tracked_render_pass(wgpu::RenderPassDescriptor {
        label: Some("forest_grass_gbuffer"),
        color_attachments: &attachments,
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: &gbuffer.depth_view,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    pass.set_render_pipeline(pipeline);
    pass.set_bind_group(0, &resources.globals, &[]);
    pass.set_bind_group(1, &frame_group, &[]);
    for mesh in &resources.meshes {
        if mesh.instance_count == 0 {
            continue;
        }
        pass.set_bind_group(2, &mesh.material, &[]);
        pass.set_vertex_buffer(0, mesh.vertices.slice(..));
        pass.set_vertex_buffer(1, mesh.instances.slice(..));
        pass.set_index_buffer(mesh.indices.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..mesh.index_count, 0, 0..mesh.instance_count);
    }
}

pub fn register_grass_systems(app: &mut bevy::app::SubApp) {
    app.init_resource::<GrassNodeState>();
    app.add_systems(
        bevy::render::Render,
        prepare_grass
            .in_set(bevy::render::RenderSystems::Prepare)
            .after(super::prepare_forest_globals),
    );
}
