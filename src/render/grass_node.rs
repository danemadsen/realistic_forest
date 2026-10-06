//! Bounded grass candidates, evaluated once per root in compute, then
//! compacted into prepared instance arenas and drawn indirectly in the G-buffer.
use super::{ExtractedForestView, ForestGlobals, ForestShaderHandles, terrain_node};
use crate::grass::{self, GrassInstance, GrassModel, GrassReadiness, GrassTexture, SharedGrassAssets};
use crate::grass_cull::{Footprint, LAYER_COUNT};
use crate::grass_stream::GrassStream;
use bevy::mesh::VertexBufferLayout;
use bevy::prelude::*;
use bevy::render::render_resource::{
    BindGroup, BindGroupLayoutDescriptor, Buffer, CachedComputePipelineId, CachedRenderPipelineId,
    ComputePipelineDescriptor, FragmentState, PipelineCache, RenderPipelineDescriptor, VertexState,
};
use bevy::render::renderer::{RenderAdapter, RenderContext, RenderDevice, RenderQueue};
use std::sync::Mutex;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GrassFrame {
    mapping: [f32; 4],
    wind: [f32; 4],
    range: [f32; 4],
    layer_end: [f32; 4],
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MaterialUniform {
    colour: [f32; 4],
    pbr: [f32; 4],
    shape: [f32; 4],
}
const _: () = assert!(std::mem::size_of::<GrassFrame>() == 64);
const _: () = assert!(std::mem::size_of::<MaterialUniform>() == 48);

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GrassDraw {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}
const _: () = assert!(std::mem::size_of::<GrassDraw>() == 20);

/// Rooted data consumed by the vertex stage, packed into three 16-byte rows.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GrassDrawInstance {
    xz: [f32; 2],
    rotation: f32,
    scale: f32,
    tint: f32,
    seed: f32,
    /// Seated height and the slope's X component.
    ground: [f32; 2],
    /// Linear ground RGB and the slope's Z component.
    ground_colour: [f32; 4],
}

/// One workgroup's contiguous candidates, sharing a bounded output region
/// and its indexed-indirect instance counter. Mirrors grass-cull.wgsl.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GrassCullJob {
    input_first: u32,
    count: u32,
    output_first: u32,
    draw_word: u32,
    radius: f32,
    _pad: [f32; 3],
}
const _: () = assert!(std::mem::size_of::<GrassDrawInstance>() == 48);
const _: () = assert!(std::mem::size_of::<GrassCullJob>() == 32);
const WORKGROUP_SIZE: u32 = 64;
const ARGS_WORDS: u32 = 5;

struct GrassLayer {
    candidates: Buffer,
    visible: Buffer,
    indirect: Buffer,
    jobs_buffer: Buffer,
    cull_group: BindGroup,
    /// Reused CPU descriptor capacity; GPU counts start at zero each frame.
    commands: Vec<GrassDraw>,
    output_starts: Vec<u32>,
    jobs: Vec<GrassCullJob>,
}

fn instances_per_slot(layer: usize) -> usize {
    grass::CHUNK_CELLS as usize * grass::CHUNK_CELLS as usize * grass::LAYER_CANDIDATES[layer]
}

struct GrassMesh {
    vertices: Buffer,
    indices: Buffer,
    index_count: u32,
    radius: f32,
    material: BindGroup,
    draws: [std::ops::Range<usize>; LAYER_COUNT],
}
struct GrassResources {
    pipeline: CachedRenderPipelineId,
    cull_pipeline: CachedComputePipelineId,
    globals: BindGroup,
    frame_buffer: Buffer,
    frame_layout: BindGroupLayoutDescriptor,
    meshes: Vec<GrassMesh>,
    stream: GrassStream,
    /// Permanent per-layer arenas shared by every model. Each chunk/model
    /// compacts inside its original candidate subrange, so it cannot overflow.
    layers: [GrassLayer; LAYER_COUNT],
    indirect_first_instance: bool,
    habitat_sampler: wgpu::Sampler,
    /// The widest root radius and the tallest blade of any model, unscaled.
    widest: f32,
    tallest: f32,
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
fn uniform_entry(binding: u32, size: u64) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: wgpu::BufferSize::new(size),
        },
        count: None,
    }
}
fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn push_cull_jobs(
    jobs: &mut Vec<GrassCullJob>,
    input_first: u32,
    count: u32,
    output_first: u32,
    draw_word: u32,
    radius: f32,
) {
    for offset in (0..count).step_by(WORKGROUP_SIZE as usize) {
        jobs.push(GrassCullJob {
            input_first: input_first + offset,
            count: (count - offset).min(WORKGROUP_SIZE),
            output_first,
            draw_word,
            radius,
            _pad: [0.0; 3],
        });
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
    let texture_descriptor = wgpu::TextureDescriptor {
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
    };
    let texture = device.wgpu_device().create_texture(&texture_descriptor);
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

/// Mesh attributes: positions, normals, texcoords and tangents of the imported
/// models, laid out as `grass::GrassVertex`.
fn mesh_vertex_attributes() -> [wgpu::VertexAttribute; 4] {
    wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x2, 3 => Float32x4]
}
/// Prepared root attributes laid out as the compute pass writes them, in
/// `GrassDrawInstance` order: identity, then seated height, slope and colour.
fn draw_instance_attributes() -> [wgpu::VertexAttribute; 7] {
    wgpu::vertex_attr_array![4 => Float32x2, 5 => Float32, 6 => Float32, 7 => Float32, 8 => Float32, 9 => Float32x2, 10 => Float32x4]
}

#[allow(clippy::too_many_arguments)]
fn prepare_grass(
    device: Res<RenderDevice>,
    adapter: Res<RenderAdapter>,
    queue: Res<RenderQueue>,
    cache: Res<PipelineCache>,
    globals: Res<ForestGlobals>,
    shaders: Res<ForestShaderHandles>,
    assets: Option<Res<SharedGrassAssets>>,
    state: Res<GrassNodeState>,
    readiness: Res<GrassReadiness>,
) {
    let mut state = state.0.lock().unwrap_or_else(|e| e.into_inner());
    if state.is_some() {
        return;
    }
    let (Some(assets), Some(global_buffer)) = (assets, globals.buffer.as_ref()) else {
        return;
    };
    if !adapter
        .get_downlevel_capabilities()
        .flags
        .contains(wgpu::DownlevelFlags::INDIRECT_EXECUTION)
    {
        bevy::log::warn_once!("Grass requires GPU indirect drawing on this adapter");
        readiness.disable();
        return;
    }
    let mut frame_uniform = uniform_entry(2, std::mem::size_of::<GrassFrame>() as u64);
    frame_uniform.visibility = wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::COMPUTE;
    let frame_layout = BindGroupLayoutDescriptor::new(
        "grass_frame_layout",
        &[
            texture_entry(0, wgpu::ShaderStages::COMPUTE),
            sampler_entry(1, wgpu::ShaderStages::COMPUTE),
            frame_uniform,
            texture_entry(3, wgpu::ShaderStages::COMPUTE),
        ],
    );
    let cull_layout = BindGroupLayoutDescriptor::new(
        "grass_cull_layout",
        &[
            storage_entry(0, true),
            storage_entry(1, true),
            storage_entry(2, false),
            storage_entry(3, false),
        ],
    );
    let cull_pipeline_descriptor = ComputePipelineDescriptor {
        label: Some("forest_grass_cull_pipeline".into()),
        layout: vec![cull_layout.clone(), frame_layout.clone()],
        immediate_size: 0,
        shader: shaders.grass_cull.clone(),
        shader_defs: vec![],
        entry_point: Some("cull_grass".into()),
        zero_initialize_workgroup_memory: false,
    };
    let cull_pipeline = cache.queue_compute_pipeline(cull_pipeline_descriptor);
    let material_layout = BindGroupLayoutDescriptor::new(
        "grass_material_layout",
        &[
            texture_entry(0, wgpu::ShaderStages::FRAGMENT),
            texture_entry(1, wgpu::ShaderStages::FRAGMENT),
            texture_entry(2, wgpu::ShaderStages::FRAGMENT),
            sampler_entry(3, wgpu::ShaderStages::FRAGMENT),
            uniform_entry(4, std::mem::size_of::<MaterialUniform>() as u64),
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
    let frame_buffer_descriptor = wgpu::BufferDescriptor {
        label: Some("grass_frame"),
        size: std::mem::size_of::<GrassFrame>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    };
    let frame_buffer = device.create_buffer(&frame_buffer_descriptor);
    let sampler_descriptor = wgpu::SamplerDescriptor {
        label: Some("grass_material_sampler"),
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Linear,
        anisotropy_clamp: 8,
        ..Default::default()
    };
    let sampler = device.wgpu_device().create_sampler(&sampler_descriptor);
    // The nine short grass models supply this dense meadow.
    let active_models: Vec<_> = assets
        .0
        .models
        .iter()
        .filter(|model| model.name.starts_with("grass-"))
        .collect();
    let stream = GrassStream::new(active_models.len());
    let slot_capacities = stream.slot_capacities();
    let indirect_first_instance = device
        .features()
        .contains(wgpu::Features::INDIRECT_FIRST_INSTANCE);
    let textures_with_index = assets.0.materials.iter().enumerate();
    let textures: Vec<_> = textures_with_index
        .map(|(index, m)| {
            let active = active_models
                .iter()
                .any(|model| model.material == index);
            active.then(|| {
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
    let build_mesh = |model: &GrassModel| {
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
        let material_buffer_descriptor = wgpu::util::BufferInitDescriptor {
            label: Some("grass_material_uniform"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        };
        let buffer = device.create_buffer_with_data(&material_buffer_descriptor);
        let views = textures[model.material]
            .as_ref()
            .expect("active grass material uploaded");
        let material_entries = [
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
        ];
        let material = super::bind_group(
            &device,
            &cache,
            "grass_material",
            &material_layout,
            &material_entries,
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
            radius: model.radius,
            material,
            draws: std::array::from_fn(|_| 0..0),
        }
    };
    let meshes = active_models
        .iter()
        .map(|model| build_mesh(model))
        .collect();
    let grass_pipeline_descriptor = RenderPipelineDescriptor {
        label: Some("forest_grass_pipeline".into()),
        layout: vec![global_layout, frame_layout.clone(), material_layout],
        immediate_size: 0,
        vertex: VertexState {
            shader: shaders.grass.clone(),
            shader_defs: vec![],
            entry_point: Some("vs_main".into()),
            buffers: vec![
                VertexBufferLayout {
                    array_stride: std::mem::size_of::<crate::grass::GrassVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: mesh_vertex_attributes().to_vec(),
                },
                VertexBufferLayout {
                    array_stride: std::mem::size_of::<GrassDrawInstance>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: draw_instance_attributes().to_vec(),
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
    };
    let pipeline = cache.queue_render_pipeline(grass_pipeline_descriptor);
    let widest = active_models.iter().map(|model| model.radius).fold(0.0, f32::max);
    let tallest = active_models.iter().map(|model| model.height).fold(0.0, f32::max);
    let grass_resources = GrassResources {
        pipeline,
        cull_pipeline,
        globals: global_group,
        frame_buffer,
        frame_layout,
        meshes,
        stream,
        layers: std::array::from_fn(|layer| {
            let instance_capacity = slot_capacities[layer] * instances_per_slot(layer);
            let draw_capacity = slot_capacities[layer] * active_models.len().max(1);
            // Splitting a slot across models adds at most one partial
            // workgroup per model beyond the first.
            let job_capacity = slot_capacities[layer]
                * (instances_per_slot(layer).div_ceil(WORKGROUP_SIZE as usize)
                    + active_models.len().saturating_sub(1));
            assert!(job_capacity <= device.limits().max_compute_workgroups_per_dimension as usize);
            let buffer = |label, size, usage| {
                let layer_buffer_descriptor = wgpu::BufferDescriptor {
                    label: Some(label),
                    size,
                    usage,
                    mapped_at_creation: false,
                };
                device.create_buffer(&layer_buffer_descriptor)
            };
            let candidates = buffer(
                "grass_chunk_candidates",
                (instance_capacity * std::mem::size_of::<GrassInstance>()) as u64,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            );
            let visible = buffer(
                "grass_prepared_instances",
                (instance_capacity * std::mem::size_of::<GrassDrawInstance>()) as u64,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX,
            );
            let indirect = buffer(
                "grass_chunk_draws",
                (draw_capacity * std::mem::size_of::<GrassDraw>()) as u64,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
            );
            let jobs_buffer = buffer(
                "grass_cull_jobs",
                (job_capacity * std::mem::size_of::<GrassCullJob>()) as u64,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            );
            let cull_entries = [
                wgpu::BindGroupEntry { binding: 0, resource: candidates.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: jobs_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: visible.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: indirect.as_entire_binding() },
            ];
            let cull_group = super::bind_group(
                &device,
                &cache,
                "grass_cull_buffers",
                &cull_layout,
                &cull_entries,
            );
            GrassLayer {
                candidates,
                visible,
                indirect,
                jobs_buffer,
                cull_group,
                commands: Vec::with_capacity(draw_capacity),
                output_starts: Vec::with_capacity(draw_capacity),
                jobs: Vec::with_capacity(job_capacity),
            }
        }),
        indirect_first_instance,
        habitat_sampler: device.wgpu_device().create_sampler(&wgpu::SamplerDescriptor {
            label: Some("grass_habitat_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        }),
        widest,
        tallest,
    };
    *state = Some(grass_resources);
}

/// Poll background generation once and upload only a bounded set of entering
/// chunk layers. The draw pass never generates candidates or allocates buffers.
fn prepare_grass_instances(
    queue: Res<RenderQueue>,
    view: Res<ExtractedForestView>,
    state: Res<GrassNodeState>,
    readiness: Res<GrassReadiness>,
) {
    let mut guard = state.0.lock().unwrap_or_else(|e| e.into_inner());
    let Some(resources) = guard.as_mut() else {
        return;
    };
    let center = [view.player_position[0], view.player_position[2]];
    for upload in resources.stream.update(center) {
        let offset = upload.slot
            * instances_per_slot(upload.key.layer)
            * std::mem::size_of::<GrassInstance>();
        queue.write_buffer(
            &resources.layers[upload.key.layer].candidates,
            offset as u64,
            bytemuck::cast_slice(&upload.instances),
        );
    }
    if !resources.stream.ready() {
        readiness.set_ready(None);
    }
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
    let Some(globals) = world.get_resource::<ForestGlobals>() else {
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
    let (Some(pipeline), Some(cull_pipeline)) = (
        cache.get_render_pipeline(resources.pipeline),
        cache.get_compute_pipeline(resources.cull_pipeline),
    ) else {
        return;
    };
    let center = [view.player_position[0], view.player_position[2]];
    let elapsed = world
        .get_resource::<super::water_node::ExtractedWater>()
        .map_or(0.0, |w| w.elapsed);
    let direction = view.settings.cloud_wind_direction_degrees.to_radians();
    let conditions = view.weather.conditions(&view.settings);
    // Storm gusts thrash the grass well beyond the steady breeze.
    let gust = 1.0 + 1.6 * conditions.gust_strength;
    let wind_strength = (conditions.wind_speed / 18.0).clamp(0.0, 2.0) * 0.11 * gust;
    let frame = GrassFrame {
        mapping: gbuffer.grass_habitat_mapping,
        wind: [direction.cos(), direction.sin(), elapsed, wind_strength],
        range: [100.0, grass::SCATTER_RADIUS, center[0], center[1]],
        layer_end: [
            grass::LAYER_END[1],
            grass::LAYER_END[2],
            grass::LAYER_END[3],
            0.0,
        ],
    };
    // Which chunks can reach the screen. A blade strays from its root by at
    // most its widest card, scaled, plus the wind's bend of its tallest tip
    // (grass.wgsl: `offset.x += wind.x * gust * bend * wind.w`, |gust| <= 1).
    let blade_extent =
        grass::MAX_SCALE * (resources.widest + resources.tallest * wind_strength.abs()) + 0.25;
    let footprint = Footprint::new(center, &globals.globals.view, &globals.globals.projection);
    for layer in &mut resources.layers {
        layer.commands.clear();
        layer.output_starts.clear();
        layer.jobs.clear();
    }
    // Commands for a model are contiguous in each layer, allowing one
    // multi-draw per model/layer. Only CPU-visible, fully uploaded slots enter
    // the job list; retained and recycled slots cannot leak old candidates.
    let chunks: Vec<_> = resources.stream.slots().filter(|chunk| {
        footprint.may_draw(
            chunk.key.centre(),
            chunk.key.root_radius(),
            grass::LAYER_END[chunk.key.layer],
            blade_extent,
        )
    }).collect();
    for (model, mesh) in resources.meshes.iter_mut().enumerate() {
        let first_draw = resources.layers.each_ref().map(|layer| layer.commands.len());
        for chunk in &chunks {
            let layer_index = chunk.key.layer;
            let layer = &mut resources.layers[layer_index];
            let start = chunk.model_starts[model];
            let count = chunk.model_starts[model + 1] - start;
            if count > 0 {
                let first_instance = (chunk.slot * instances_per_slot(layer_index)) as u32 + start;
                let draw_word = layer.commands.len() as u32 * ARGS_WORDS + 1;
                layer.commands.push(GrassDraw {
                    index_count: mesh.index_count,
                    instance_count: 0,
                    first_index: 0,
                    base_vertex: 0,
                    first_instance: if resources.indirect_first_instance { first_instance } else { 0 },
                });
                layer.output_starts.push(first_instance);
                push_cull_jobs(&mut layer.jobs, first_instance, count, first_instance, draw_word, mesh.radius);
            }
        }
        mesh.draws = std::array::from_fn(|layer| first_draw[layer]..resources.layers[layer].commands.len());
    }
    for layer in &resources.layers {
        if !layer.commands.is_empty() {
            // Every active draw starts empty, including one which rejected
            // all roots this frame after drawing survivors in the last frame.
            queue.write_buffer(&layer.indirect, 0, bytemuck::cast_slice(&layer.commands));
            queue.write_buffer(&layer.jobs_buffer, 0, bytemuck::cast_slice(&layer.jobs));
        }
    }
    queue.write_buffer(&resources.frame_buffer, 0, bytemuck::bytes_of(&frame));
    // Rebinding the capture view also handles window resize without stale views.
    let frame_entries = [
        wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(&gbuffer.grass_habitat_view),
        },
        wgpu::BindGroupEntry {
            binding: 1,
            resource: wgpu::BindingResource::Sampler(&resources.habitat_sampler),
        },
        wgpu::BindGroupEntry {
            binding: 2,
            resource: resources.frame_buffer.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 3,
            resource: wgpu::BindingResource::TextureView(&gbuffer.grass_ground_average_view),
        },
    ];
    let frame_group = super::bind_group(
        device,
        cache,
        "grass_frame",
        &resources.frame_layout,
        &frame_entries,
    );
    {
        // This pass follows the habitat and tree-canopy captures and precedes
        // the draw, so prepared roots and survivor counts share this frame's
        // terrain, shade and camera fades without a GPU readback.
        let mut pass = ctx.command_encoder().begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("forest_grass_cull"),
            timestamp_writes: None,
        });
        pass.set_pipeline(cull_pipeline);
        pass.set_bind_group(1, Some(&*frame_group), &[]);
        for layer in &resources.layers {
            if !layer.jobs.is_empty() {
                pass.set_bind_group(0, Some(&*layer.cull_group), &[]);
                pass.dispatch_workgroups(layer.jobs.len() as u32, 1, 1);
            }
        }
    }
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
    let gbuffer_pass_descriptor = wgpu::RenderPassDescriptor {
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
    };
    let mut pass = ctx.begin_tracked_render_pass(gbuffer_pass_descriptor);
    pass.set_render_pipeline(pipeline);
    pass.set_bind_group(0, &resources.globals, &[]);
    pass.set_bind_group(1, &frame_group, &[]);
    for mesh in &resources.meshes {
        pass.set_bind_group(2, &mesh.material, &[]);
        pass.set_vertex_buffer(0, mesh.vertices.slice(..));
        pass.set_index_buffer(mesh.indices.slice(..), wgpu::IndexFormat::Uint32);
        for (layer_index, draws) in mesh.draws.iter().enumerate() {
            if draws.is_empty() {
                continue;
            }
            let layer = &resources.layers[layer_index];
            let args_stride = std::mem::size_of::<GrassDraw>() as u64;
            if resources.indirect_first_instance {
                pass.set_vertex_buffer(1, layer.visible.slice(..));
                pass.multi_draw_indexed_indirect(&layer.indirect, draws.start as u64 * args_stride, draws.len() as u32);
            } else {
                for draw in draws.clone() {
                    // first_instance remains zero on adapters without that
                    // feature; the slice supplies the same compacted region.
                    let offset = layer.output_starts[draw] as u64 * std::mem::size_of::<GrassDrawInstance>() as u64;
                    pass.set_vertex_buffer(1, layer.visible.slice(offset..));
                    pass.draw_indexed_indirect(&layer.indirect, draw as u64 * args_stride);
                }
            }
        }
    }
    drop(pass);
    if resources.stream.ready()
        && let Some(readiness) = world.get_resource::<GrassReadiness>()
    {
        readiness.set_ready(resources.stream.anchor());
    }
}

#[cfg(test)]
#[path = "grass_node_tests.rs"]
mod tests;

pub fn register_grass_systems(app: &mut bevy::app::SubApp) {
    app.init_resource::<GrassNodeState>();
    let mut systems = (prepare_grass, prepare_grass_instances).chain();
    systems = systems.in_set(bevy::render::RenderSystems::Prepare);
    systems = systems.after(super::prepare_forest_globals);
    app.add_systems(bevy::render::Render, systems);
}
