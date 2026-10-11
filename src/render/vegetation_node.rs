//! GPU-driven vegetation in the terrain G-buffer.
//!
//! Every streamed plant lives in one storage buffer, re-uploaded only when the
//! main world publishes a new snapshot. Each frame a compute pass
//! (vegetation-cull.wgsl) seats every plant on the eroded terrain, culls it
//! against the view, picks its LOD and appends it to its model's region of
//! that LOD's output, counting it straight into the indirect draw arguments.
//! One indirect draw per (model, LOD, primitive) then renders the lists with a
//! shared shader (vegetation.wgsl), between the terrain and the grass so
//! the grass behind a trunk fails the depth test early. The CPU never reads
//! anything back; a draw whose list came out empty costs almost nothing.

use super::vegetation_shadows::{
    Cascade, LightBasis, SHADOW_CASCADES, SHADOW_FORMAT, SHADOW_FRONT_FACE, ShadowUniform, VegetationShadowMaps, fit_cascades,
    light_basis,
};
use super::{ExtractedForestView, ForestGlobals, ForestShaderHandles, TerrainStageUniforms, terrain_node};
use crate::constants::SEA_LEVEL;
use crate::vegetation::assets::{PreparedTexture, Species, Surface, VegetationAssets};
use crate::vegetation::scatter::PlantInstance;
use crate::vegetation::{
    DEEPEST_FURROW, LOWEST_ROOT, STEEPEST_ROOT, VegetationField, VegetationSnapshot, render_profile,
};
use bevy::mesh::VertexBufferLayout;
use bevy::prelude::*;
use bevy::render::render_resource::{
    BindGroup, BindGroupLayoutDescriptor, Buffer, CachedComputePipelineId, CachedRenderPipelineId,
    ComputePipelineDescriptor, FragmentState, PipelineCache, RenderPipelineDescriptor, VertexState,
};
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue};
use bevy::render::{ExtractSchedule, MainWorld};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------------------
// GPU layouts (each mirrored by a WGSL struct)
// ---------------------------------------------------------------------------

/// `ModelParams` in vegetation-cull.wgsl and vegetation.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct ModelParams {
    height: f32,
    crown_radius: f32,
    crown_base: f32,
    bound_radius: f32,
    lod_end: [f32; 4],
    max_distance: f32,
    lod_count: u32,
    habitat: u32,
    region: u32,
    /// Word offset of each LOD's first draw's `instance_count`.
    draw_word: [u32; 4],
    /// Consecutive draws (one per primitive) of each LOD.
    draw_count: [u32; 4],
    /// The same for each shadow cascade's draws; a zero count means the
    /// model casts no shadow into that cascade.
    shadow_word: [u32; 4],
    shadow_count: [u32; 4],
}

/// `DrawInstance` in both shaders: the culled, seated plant.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct DrawInstance {
    position: [f32; 3],
    scale: f32,
    yaw: f32,
    seed: f32,
    fade: f32,
    model: u32,
}

/// `CullUniform` in vegetation-cull.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct CullUniform {
    planes: [[f32; 4]; 4],
    camera: [f32; 4],
    habitat_mapping: [f32; 4],
    counts: [u32; 4],
    ground: [f32; 4],
    /// The shadowing light's frame: across, up and along its travel.
    light_right: [f32; 4],
    light_up: [f32; 4],
    light_forward: [f32; 4],
    /// Per cascade: light-frame centre across the light, half extent, far
    /// depth. Keep four slots to match the cull shader's vec4-based contract.
    cascades: [[f32; 4]; 4],
    /// Per cascade: light-frame near depth.
    cascade_near: [f32; 4],
}

/// `ShadowCascade` in vegetation.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct CascadeUniform {
    view_projection: [f32; 16],
    /// xyz the direction the light travels, w metres per texel.
    light: [f32; 4],
}

/// `VegetationFrame` in vegetation.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct FrameUniform {
    wind: [f32; 4],
    params: [f32; 4],
}

/// `PlantMaterial` in vegetation.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct MaterialUniform {
    base_color_factor: [f32; 4],
    params: [f32; 4],
    surface: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<ModelParams>() == 112);
const _: () = assert!(std::mem::size_of::<DrawInstance>() == 32);
const _: () = assert!(std::mem::size_of::<PlantInstance>() == 32);
const _: () = assert!(std::mem::size_of::<CullUniform>() == 256);
const _: () = assert!(std::mem::size_of::<CascadeUniform>() == 80);
const _: () = assert!(std::mem::size_of::<FrameUniform>() == 32);
const _: () = assert!(std::mem::size_of::<MaterialUniform>() == 48);

/// LOD regions in the output buffer; a model uses as many as it has LODs.
const LOD_SLOTS: u64 = 4;
/// Output regions: the LODs, then one per shadow cascade.
const OUTPUT_SLOTS: u64 = LOD_SLOTS + SHADOW_CASCADES as u64;
/// The LOD each shadow cascade draws a plant with (or its last, if it has
/// fewer): the near cascades need the crown's gaps, the far ones only its
/// outline.
const SHADOW_LOD: [usize; SHADOW_CASCADES] = [1, 2, 3];
/// Words of one `DrawIndexedIndirectArgs`.
const ARGS_WORDS: usize = 5;
const WORKGROUP_SIZE: u32 = 64;
const INSTANCE_STRIDE: u64 = std::mem::size_of::<DrawInstance>() as u64;

/// The slack pads the culling sphere for the scatter's height estimate,
/// which ignores erosion. The rooting limits it enforces are
/// `vegetation::{LOWEST_ROOT, STEEPEST_ROOT, DEEPEST_FURROW}`.
const CULLING_SLACK: f32 = 12.0;

// ---------------------------------------------------------------------------
// Extraction
// ---------------------------------------------------------------------------

/// The main world's library and latest snapshot, copied by pointer.
#[derive(Resource, Default)]
pub struct ExtractedVegetation {
    enabled: bool,
    assets: Option<Arc<VegetationAssets>>,
    snapshot: Option<Arc<VegetationSnapshot>>,
    uploaded: Option<Arc<AtomicU64>>,
}

impl ExtractedVegetation {
    /// Whether the scatter is running (not disabled on the command line).
    pub fn enabled(&self) -> bool {
        self.enabled
    }
}

fn extract_vegetation(main_world: Res<MainWorld>, mut extracted: ResMut<ExtractedVegetation>) {
    let Some(field) = main_world.get_resource::<VegetationField>() else {
        return;
    };
    extracted.enabled = field.enabled();
    if extracted.assets.is_none() {
        extracted.assets = field.assets().cloned();
    }
    if extracted.snapshot.as_ref().map(|s| s.generation) != Some(field.snapshot().generation) {
        extracted.snapshot = Some(field.snapshot().clone());
    }
    if extracted.uploaded.is_none() {
        extracted.uploaded = Some(field.uploaded.clone());
    }
}

// ---------------------------------------------------------------------------
// Resources
// ---------------------------------------------------------------------------

struct Draw {
    model: u32,
    /// The output region the draw reads: its LOD, or `LOD_SLOTS` plus its
    /// shadow cascade.
    slot: u32,
    material: usize,
    /// Byte offset of this draw's arguments.
    args_offset: u64,
}

/// Built once, when the library arrives.
struct MaterialResources {
    /// Keeps this material's textures alive.
    bind_group: BindGroup,
    double_sided: bool,
}

/// Built once, when the library arrives.
struct LibraryResources {
    vertices: Buffer,
    indices: Buffer,
    materials: Vec<MaterialResources>,
    /// In submission order: LOD 0 of every model first, so the nearest,
    /// largest plants lay down depth before the cheaper, farther ones.
    draws: Vec<Draw>,
    /// Shadow draws, by cascade then material.
    shadow_draws: Vec<Draw>,
    args_template: Buffer,
    args: Buffer,
    args_bytes: u64,
    params: Vec<ModelParams>,
    models: Buffer,
    cull_pipeline: CachedComputePipelineId,
    /// Single-sided, then double-sided, matching each material's flag.
    draw_pipelines: [CachedRenderPipelineId; 2],
    shadow_pipelines: [CachedRenderPipelineId; 2],
    canopy_pipeline: CachedRenderPipelineId,
    /// One uniform and bind group per cascade.
    cascade_buffers: Vec<Buffer>,
    cascade_groups: Vec<BindGroup>,
    cull_globals: BindGroup,
    draw_globals: BindGroup,
    terrain_layout: BindGroupLayoutDescriptor,
    cull_layout: BindGroupLayoutDescriptor,
    frame_layout: BindGroupLayoutDescriptor,
    frame_buffer: Buffer,
    cull_buffer: Buffer,
    stage_buffer: Buffer,
    noise_sampler: wgpu::Sampler,
    atlas_sampler: wgpu::Sampler,
    mask_sampler: wgpu::Sampler,
    habitat_sampler: wgpu::Sampler,
}

/// Rebuilt for every snapshot.
struct SnapshotResources {
    generation: u64,
    plant_count: u32,
    visible: Buffer,
    cull_group: BindGroup,
    frame_group: BindGroup,
    /// Draws whose model has plants, with their instance-buffer offsets.
    active: Vec<(usize, u64)>,
    /// The same for the shadow draws, by cascade.
    shadow_active: [Vec<(usize, u64)>; SHADOW_CASCADES],
    /// Bytes of one model's region per LOD, by model.
    region_bytes: Vec<u64>,
}

struct VegetationResources {
    library: LibraryResources,
    snapshot: Option<SnapshotResources>,
    /// Which (habitat capture, plant snapshot) the crowns' shade now on the
    /// grass's ground average was drawn from. The shade is a multiply into the
    /// texture's alpha, so it is applied once per pair, from the terrain pass's
    /// untouched copy, and left alone while the pair stands.
    canopy_stamp: Option<(u64, u64)>,
}

/// Take the crowns' shade back off the ground average the grass reads, if it
/// is on there: the plants are no longer drawn, so the grass must not keep
/// growing around them.
fn clear_canopy(
    resources: &mut Option<VegetationResources>,
    gbuffer: &terrain_node::GbufferTargets,
    ctx: &mut RenderContext,
) {
    if let Some(resources) = resources.as_mut()
        && resources
            .canopy_stamp
            .take()
            .is_some_and(|(capture, _)| capture == gbuffer.grass_habitat.generation)
    {
        terrain_node::copy_ground_average(ctx.command_encoder(), gbuffer);
    }
}

#[derive(Resource, Default)]
pub struct VegetationNodeState(Mutex<Option<VegetationResources>>);

// ---------------------------------------------------------------------------
// Layout helpers
// ---------------------------------------------------------------------------

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

fn uniform_entry(binding: u32, visibility: wgpu::ShaderStages, size: u64) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: wgpu::BufferSize::new(size),
        },
        count: None,
    }
}

fn storage_entry(binding: u32, visibility: wgpu::ShaderStages, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn sampler(device: &RenderDevice, label: &str, address: wgpu::AddressMode, anisotropy: u16) -> wgpu::Sampler {
    let sampler_descriptor = wgpu::SamplerDescriptor {
        label: Some(label),
        address_mode_u: address,
        address_mode_v: address,
        address_mode_w: address,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Linear,
        anisotropy_clamp: anisotropy,
        ..Default::default()
    };
    device
        .wgpu_device()
        .create_sampler(&sampler_descriptor)
}

/// Paired billboard cards already have separate front and back faces. Cull
/// their backfaces before rasterization so coplanar atlas sides cannot fight
/// for depth. Ordinary foliage and unpaired cards still draw both sides.
fn sided_pipelines(cache: &PipelineCache, descriptor: RenderPipelineDescriptor) -> [CachedRenderPipelineId; 2] {
    [Some(wgpu::Face::Back), None].map(|cull_mode| {
        let mut descriptor = descriptor.clone();
        descriptor.primitive.cull_mode = cull_mode;
        cache.queue_render_pipeline(descriptor)
    })
}

fn upload_texture(device: &RenderDevice, queue: &RenderQueue, source: &PreparedTexture) -> wgpu::TextureView {
    let texture_descriptor = wgpu::TextureDescriptor {
        label: Some(&source.label),
        size: wgpu::Extent3d {
            width: source.width,
            height: source.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: source.levels.len() as u32,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: if source.srgb {
            wgpu::TextureFormat::Rgba8UnormSrgb
        } else {
            wgpu::TextureFormat::Rgba8Unorm
        },
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    };
    let texture = device.wgpu_device().create_texture(&texture_descriptor);
    for (level, data) in source.levels.iter().enumerate() {
        let width = (source.width >> level).max(1);
        let height = (source.height >> level).max(1);
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: level as u32,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    }
    texture.create_view(&Default::default())
}

/// Species-wide surface treatment: leaf translucency, an albedo correction
/// that brings the packs' foliage atlases to a common exposure, and whether
/// the crown is a cone.
fn surface_terms(species: Species, surface: Surface) -> (f32, f32, f32) {
    let profile = render_profile(species, "");
    let translucency = if surface == Surface::Bark { 0.0 } else { profile.translucency };
    let albedo = match (species, surface) {
        (_, Surface::Bark) => 1.0,
        // The pine atlas is twice as bright as the fir one; real pines are
        // only a little lighter than firs.
        (Species::Pine, _) => 0.72,
        (Species::Lilac, _) => 0.9,
        _ => 1.0,
    };
    let conifer = matches!(species, Species::Fir | Species::Pine) as u32 as f32;
    (translucency, albedo, conifer)
}

// ---------------------------------------------------------------------------
// Preparation
// ---------------------------------------------------------------------------

fn build_library(
    device: &RenderDevice,
    queue: &RenderQueue,
    cache: &PipelineCache,
    shaders: &ForestShaderHandles,
    global_buffer: &wgpu::Buffer,
    assets: &VegetationAssets,
) -> LibraryResources {
    let start = std::time::Instant::now();
    let vertices = device.create_buffer_with_data(&wgpu::util::BufferInitDescriptor {
        label: Some("vegetation_vertices"),
        contents: bytemuck::cast_slice(&assets.vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let indices = device.create_buffer_with_data(&wgpu::util::BufferInitDescriptor {
        label: Some("vegetation_indices"),
        contents: bytemuck::cast_slice(&assets.indices),
        usage: wgpu::BufferUsages::INDEX,
    });
    let views: Vec<wgpu::TextureView> = assets
        .textures
        .iter()
        .map(|texture| upload_texture(device, queue, texture))
        .collect();
    let material_sampler = sampler(device, "vegetation_material_sampler", wgpu::AddressMode::Repeat, 8);

    // Which species each material belongs to, for its surface treatment.
    let mut material_species = vec![Species::Bush; assets.materials.len()];
    for model in assets.models.iter().rev() {
        for lod in &model.lods {
            for primitive in &lod.primitives {
                material_species[primitive.material] = model.species;
            }
        }
    }
    let material_layout = BindGroupLayoutDescriptor::new(
        "vegetation_material_layout",
        &[
            texture_entry(0, wgpu::ShaderStages::FRAGMENT),
            texture_entry(1, wgpu::ShaderStages::FRAGMENT),
            sampler_entry(2, wgpu::ShaderStages::FRAGMENT),
            uniform_entry(3, wgpu::ShaderStages::VERTEX_FRAGMENT, 48),
        ],
    );
    let materials = assets
        .materials
        .iter()
        .enumerate()
        .map(|(index, material)| {
            let (translucency, albedo, conifer) = surface_terms(material_species[index], material.surface);
            let uniform = MaterialUniform {
                base_color_factor: material.base_color_factor,
                params: [
                    material.alpha_cutoff,
                    material.normal_scale,
                    material.roughness_factor,
                    material.occlusion_strength,
                ],
                surface: [
                    match material.surface {
                        Surface::Bark => 0.0,
                        Surface::Foliage => 1.0,
                        Surface::Billboard => 2.0,
                    },
                    translucency,
                    albedo,
                    conifer,
                ],
            };
            let buffer = device.create_buffer_with_data(&wgpu::util::BufferInitDescriptor {
                label: Some("vegetation_material_uniform"),
                contents: bytemuck::bytes_of(&uniform),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let material_entries = [
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&views[material.base_color]),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&views[material.detail]),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&material_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffer.as_entire_binding(),
                },
            ];
            let bind_group = super::bind_group(
                device,
                cache,
                "vegetation_material",
                &material_layout,
                &material_entries,
            );
            MaterialResources { bind_group, double_sided: material.double_sided }
        })
        .collect();

    // Draws, their argument template and the per-model bookkeeping.
    let mut params = Vec::with_capacity(assets.models.len());
    let mut draws = Vec::new();
    let mut shadow_draws = Vec::new();
    let mut template: Vec<u32> = Vec::new();
    for (model_index, model) in assets.models.iter().enumerate() {
        if model.lods.len() > LOD_SLOTS as usize {
            log::warn!(
                "VEGETATION: {} has {} LODs; only the first {LOD_SLOTS} are drawn",
                model.name,
                model.lods.len()
            );
        }
        let profile = render_profile(model.species, &model.form);
        let mut model_params = ModelParams {
            height: model.height,
            crown_radius: model.crown_radius,
            crown_base: model.crown_base,
            bound_radius: model.bounding_radius,
            lod_end: [
                profile.lod_end[0],
                profile.lod_end[1],
                profile.lod_end[2],
                f32::INFINITY,
            ],
            max_distance: profile.max_distance,
            lod_count: (model.lods.len() as u32).min(LOD_SLOTS as u32),
            habitat: profile.ground_habitat as u32,
            region: 0,
            draw_word: [0; 4],
            draw_count: [0; 4],
            shadow_word: [0; 4],
            shadow_count: [0; 4],
        };
        // One indirect draw per primitive of every LOD, then of every
        // cascade's shadow LOD.
        let mut add_draws = |list: &mut Vec<Draw>, slot: usize, lod: &crate::vegetation::assets::PlantLod| {
            let first_draw = template.len() / ARGS_WORDS;
            for primitive in &lod.primitives {
                list.push(Draw {
                    model: model_index as u32,
                    slot: slot as u32,
                    material: primitive.material,
                    args_offset: (template.len() * 4) as u64,
                });
                template.extend_from_slice(&[
                    primitive.index_count,
                    0,
                    primitive.first_index,
                    lod.base_vertex,
                    0,
                ]);
            }
            ((first_draw * ARGS_WORDS + 1) as u32, lod.primitives.len() as u32)
        };
        for (lod_index, lod) in model.lods.iter().enumerate().take(LOD_SLOTS as usize) {
            (model_params.draw_word[lod_index], model_params.draw_count[lod_index]) =
                add_draws(&mut draws, lod_index, lod);
        }
        let last = model.lods.len().min(LOD_SLOTS as usize).saturating_sub(1);
        for (cascade, &shadow_lod) in SHADOW_LOD.iter().enumerate().take(profile.shadow_cascades as usize) {
            let Some(lod) = model.lods.get(shadow_lod.min(last)) else {
                break;
            };
            (model_params.shadow_word[cascade], model_params.shadow_count[cascade]) =
                add_draws(&mut shadow_draws, LOD_SLOTS as usize + cascade, lod);
        }
        params.push(model_params);
    }
    draws.sort_by_key(|draw| (draw.slot, draw.material, draw.model));
    shadow_draws.sort_by_key(|draw| (draw.slot, draw.material, draw.model));
    let args_bytes = (template.len() * 4).max(4) as u64;
    let args_template = device.create_buffer_with_data(&wgpu::util::BufferInitDescriptor {
        label: Some("vegetation_args_template"),
        contents: bytemuck::cast_slice(&template),
        usage: wgpu::BufferUsages::COPY_SRC,
    });
    let args = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vegetation_args"),
        size: args_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let models = device.create_buffer_with_data(&wgpu::util::BufferInitDescriptor {
        label: Some("vegetation_models"),
        contents: bytemuck::cast_slice(&params),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    // Pipelines.
    let compute = wgpu::ShaderStages::COMPUTE;
    let cull_globals_layout = BindGroupLayoutDescriptor::new(
        "vegetation_cull_globals_layout",
        &[uniform_entry(
            0,
            compute,
            std::mem::size_of::<super::GlobalUniformsGpu>() as u64,
        )],
    );
    let terrain_layout = BindGroupLayoutDescriptor::new(
        "vegetation_terrain_layout",
        &[
            texture_entry(0, compute),
            sampler_entry(1, compute),
            texture_entry(2, compute),
            sampler_entry(3, compute),
            texture_entry(4, compute),
            texture_entry(5, compute),
            sampler_entry(6, compute),
            uniform_entry(7, compute, std::mem::size_of::<TerrainStageUniforms>() as u64),
            texture_entry(8, compute),
            sampler_entry(9, compute),
            storage_entry(10, compute, true),
            storage_entry(11, compute, true),
        ],
    );
    let cull_layout = BindGroupLayoutDescriptor::new(
        "vegetation_cull_layout",
        &[
            uniform_entry(0, compute, std::mem::size_of::<CullUniform>() as u64),
            storage_entry(1, compute, true),
            storage_entry(2, compute, true),
            storage_entry(3, compute, false),
            storage_entry(4, compute, false),
        ],
    );
    let frame_layout = BindGroupLayoutDescriptor::new(
        "vegetation_frame_layout",
        &[
            uniform_entry(0, wgpu::ShaderStages::VERTEX_FRAGMENT, std::mem::size_of::<FrameUniform>() as u64),
            storage_entry(1, wgpu::ShaderStages::VERTEX, true),
            storage_entry(2, wgpu::ShaderStages::VERTEX, true),
        ],
    );
    let cull_pipeline_descriptor = ComputePipelineDescriptor {
        label: Some("forest_vegetation_cull_pipeline".into()),
        layout: vec![cull_globals_layout.clone(), terrain_layout.clone(), cull_layout.clone()],
        immediate_size: 0,
        shader: shaders.vegetation_cull.clone(),
        shader_defs: vec![],
        entry_point: Some("cull_plants".into()),
        zero_initialize_workgroup_memory: false,
    };
    let cull_pipeline = cache.queue_compute_pipeline(cull_pipeline_descriptor);
    let vertex_attributes =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x2, 3 => Float32x4];
    let instance_attributes = wgpu::vertex_attr_array![
        4 => Float32x3, 5 => Float32, 6 => Float32, 7 => Float32, 8 => Float32, 9 => Uint32
    ];
    let vertex_buffers = vec![
        VertexBufferLayout {
            array_stride: std::mem::size_of::<crate::vegetation::assets::PlantVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: vertex_attributes.to_vec(),
        },
        VertexBufferLayout {
            array_stride: INSTANCE_STRIDE,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: instance_attributes.to_vec(),
        },
    ];
    let draw_globals_layout = super::globals_layout();
    let draw_pipeline_descriptor = RenderPipelineDescriptor {
        label: Some("forest_vegetation_pipeline".into()),
        layout: vec![draw_globals_layout.clone(), frame_layout.clone(), material_layout.clone()],
        immediate_size: 0,
        vertex: VertexState {
            shader: shaders.vegetation.clone(),
            shader_defs: vec![],
            entry_point: Some("vs_main".into()),
            buffers: vertex_buffers.clone(),
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Some(true),
            // Reverse-Z, as the terrain and grass passes.
            depth_compare: Some(wgpu::CompareFunction::GreaterEqual),
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: Default::default(),
        fragment: Some(FragmentState {
            shader: shaders.vegetation.clone(),
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
    let draw_pipelines = sided_pipelines(cache, draw_pipeline_descriptor);
    // Shadows: depth only, from the light, with the same wind as the view.
    let cascade_layout = BindGroupLayoutDescriptor::new(
        "vegetation_cascade_layout",
        &[uniform_entry(
            1,
            wgpu::ShaderStages::VERTEX_FRAGMENT,
            std::mem::size_of::<CascadeUniform>() as u64,
        )],
    );
    let shadow_pipeline_descriptor = RenderPipelineDescriptor {
        label: Some("forest_vegetation_shadow_pipeline".into()),
        layout: vec![cascade_layout.clone(), frame_layout.clone(), material_layout],
        immediate_size: 0,
        vertex: VertexState {
            shader: shaders.vegetation.clone(),
            shader_defs: vec![],
            entry_point: Some("vs_shadow".into()),
            buffers: vertex_buffers,
        },
        primitive: wgpu::PrimitiveState {
            front_face: SHADOW_FRONT_FACE,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: SHADOW_FORMAT,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::LessEqual),
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: Default::default(),
        fragment: Some(FragmentState {
            shader: shaders.vegetation.clone(),
            shader_defs: vec![],
            entry_point: Some("fs_shadow".into()),
            targets: vec![],
        }),
        zero_initialize_workgroup_memory: false,
    };
    let shadow_pipelines = sided_pipelines(cache, shadow_pipeline_descriptor);
    // Crowns from above over the grass capture: multiplies the alpha of its
    // ground-colour texture by the light each crown lets through.
    let canopy_pipeline_descriptor = RenderPipelineDescriptor {
        label: Some("forest_vegetation_canopy_pipeline".into()),
        layout: vec![draw_globals_layout.clone(), frame_layout.clone()],
        immediate_size: 0,
        vertex: VertexState {
            shader: shaders.vegetation.clone(),
            shader_defs: vec![],
            entry_point: Some("vs_canopy".into()),
            buffers: vec![],
        },
        primitive: Default::default(),
        depth_stencil: None,
        multisample: Default::default(),
        fragment: Some(FragmentState {
            shader: shaders.vegetation.clone(),
            shader_defs: vec![],
            entry_point: Some("fs_canopy".into()),
            targets: vec![Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba16Float,
                blend: Some(wgpu::BlendState {
                    color: wgpu::BlendComponent::REPLACE,
                    alpha: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::Zero,
                        dst_factor: wgpu::BlendFactor::SrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    },
                }),
                write_mask: wgpu::ColorWrites::ALPHA,
            })],
        }),
        zero_initialize_workgroup_memory: false,
    };
    let canopy_pipeline = cache.queue_render_pipeline(canopy_pipeline_descriptor);
    let cascade_buffers: Vec<Buffer> = (0..SHADOW_CASCADES)
        .map(|_| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vegetation_cascade"),
                size: std::mem::size_of::<CascadeUniform>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        })
        .collect();
    let cascade_groups = cascade_buffers
        .iter()
        .map(|buffer| {
            super::bind_group(
                device,
                cache,
                "vegetation_cascade",
                &cascade_layout,
                &[wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffer.as_entire_binding(),
                }],
            )
        })
        .collect();
    let cull_globals = super::bind_group(
        device,
        cache,
        "vegetation_cull_globals",
        &cull_globals_layout,
        &super::globals_bind_group_entries(global_buffer),
    );
    let draw_globals = super::bind_group(
        device,
        cache,
        "vegetation_draw_globals",
        &draw_globals_layout,
        &super::globals_bind_group_entries(global_buffer),
    );
    let uniform = |label: &'static str, size: usize| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    };
    log::info!(
        "VEGETATION: {} draws ({} for shadows), {} materials and {} textures uploaded in {:.1?}",
        draws.len() + shadow_draws.len(),
        shadow_draws.len(),
        assets.materials.len(),
        views.len(),
        start.elapsed()
    );
    LibraryResources {
        vertices,
        indices,
        materials,
        draws,
        shadow_draws,
        args_template,
        args,
        args_bytes,
        params,
        models,
        cull_pipeline,
        draw_pipelines,
        shadow_pipelines,
        canopy_pipeline,
        cascade_buffers,
        cascade_groups,
        cull_globals,
        draw_globals,
        terrain_layout,
        cull_layout,
        frame_layout,
        frame_buffer: uniform("vegetation_frame", std::mem::size_of::<FrameUniform>()),
        cull_buffer: uniform("vegetation_cull", std::mem::size_of::<CullUniform>()),
        stage_buffer: uniform("vegetation_terrain_stage", std::mem::size_of::<TerrainStageUniforms>()),
        noise_sampler: sampler(device, "vegetation_noise_sampler", wgpu::AddressMode::Repeat, 1),
        atlas_sampler: sampler(device, "vegetation_atlas_sampler", wgpu::AddressMode::ClampToEdge, 1),
        mask_sampler: sampler(device, "vegetation_mask_sampler", wgpu::AddressMode::ClampToEdge, 1),
        habitat_sampler: sampler(device, "vegetation_habitat_sampler", wgpu::AddressMode::ClampToEdge, 1),
    }
}

fn build_snapshot(
    device: &RenderDevice,
    queue: &RenderQueue,
    cache: &PipelineCache,
    library: &mut LibraryResources,
    snapshot: &VegetationSnapshot,
) -> SnapshotResources {
    let plant_count = snapshot.plants.len() as u32;
    // Regions: every model gets as many slots per LOD as it has plants.
    let mut counts = vec![0u32; library.params.len()];
    for plant in &snapshot.plants {
        if let Some(count) = counts.get_mut(plant.model as usize) {
            *count += 1;
        }
    }
    let mut region = 0u32;
    for (params, count) in library.params.iter_mut().zip(&counts) {
        params.region = region;
        region += count;
    }
    queue.write_buffer(&library.models, 0, bytemuck::cast_slice(&library.params));
    let placeholder = [PlantInstance::default()];
    let plants_descriptor = wgpu::util::BufferInitDescriptor {
        label: Some("vegetation_plants"),
        contents: bytemuck::cast_slice(if snapshot.plants.is_empty() {
            &placeholder
        } else {
            &snapshot.plants
        }),
        usage: wgpu::BufferUsages::STORAGE,
    };
    let plants = device.create_buffer_with_data(&plants_descriptor);
    let visible = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vegetation_visible"),
        size: (OUTPUT_SLOTS * plant_count.max(1) as u64 * INSTANCE_STRIDE).max(INSTANCE_STRIDE),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX,
        mapped_at_creation: false,
    });
    let cull_entries = [
        wgpu::BindGroupEntry {
            binding: 0,
            resource: library.cull_buffer.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 1,
            resource: plants.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 2,
            resource: library.models.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 3,
            resource: visible.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 4,
            resource: library.args.as_entire_binding(),
        },
    ];
    let cull_group = super::bind_group(
        device,
        cache,
        "vegetation_cull",
        &library.cull_layout,
        &cull_entries,
    );
    let frame_entries = [
        wgpu::BindGroupEntry {
            binding: 0,
            resource: library.frame_buffer.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 1,
            resource: library.models.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 2,
            resource: plants.as_entire_binding(),
        },
    ];
    let frame_group = super::bind_group(
        device,
        cache,
        "vegetation_frame",
        &library.frame_layout,
        &frame_entries,
    );
    let offsets = |draws: &[Draw]| -> Vec<(usize, u64)> {
        let visible_draws = draws
            .iter()
            .enumerate()
            .filter(|(_, draw)| counts[draw.model as usize] > 0);
        visible_draws
            .map(|(index, draw)| {
                let model = &library.params[draw.model as usize];
                let offset = (draw.slot as u64 * plant_count as u64 + model.region as u64) * INSTANCE_STRIDE;
                (index, offset)
            })
            .collect()
    };
    let active = offsets(&library.draws);
    let shadow_offsets = |cascade: usize| -> Vec<(usize, u64)> {
        let slot = LOD_SLOTS as u32 + cascade as u32;
        offsets(&library.shadow_draws)
            .into_iter()
            .filter(|&(index, _)| library.shadow_draws[index].slot == slot)
            .collect()
    };
    let shadow_active = std::array::from_fn(shadow_offsets);
    let region_bytes = counts.iter().map(|&count| count as u64 * INSTANCE_STRIDE).collect();
    SnapshotResources {
        generation: snapshot.generation,
        plant_count,
        visible,
        cull_group,
        frame_group,
        active,
        shadow_active,
        region_bytes,
    }
}

fn prepare_vegetation(
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    cache: Res<PipelineCache>,
    globals: Res<ForestGlobals>,
    shaders: Res<ForestShaderHandles>,
    extracted: Res<ExtractedVegetation>,
    state: Res<VegetationNodeState>,
) {
    let mut state = state.0.lock().unwrap_or_else(|e| e.into_inner());
    if state.is_none() {
        let (Some(assets), Some(global_buffer)) = (extracted.assets.as_ref(), globals.buffer.as_ref()) else {
            return;
        };
        *state = Some(VegetationResources {
            library: build_library(&device, &queue, &cache, &shaders, global_buffer, assets),
            snapshot: None,
            canopy_stamp: None,
        });
    }
    let Some(resources) = state.as_mut() else {
        return;
    };
    let Some(snapshot) = extracted.snapshot.as_ref() else {
        return;
    };
    if resources.snapshot.as_ref().map(|s| s.generation) != Some(snapshot.generation) {
        resources.snapshot = Some(build_snapshot(&device, &queue, &cache, &mut resources.library, snapshot));
        if let Some(uploaded) = &extracted.uploaded {
            uploaded.store(snapshot.generation, Ordering::Release);
        }
    }
}

// ---------------------------------------------------------------------------
// The pass
// ---------------------------------------------------------------------------

/// The four side planes of the view frustum in world space, inward facing,
/// from the combined view-projection (Gribb-Hartmann).
fn frustum_planes(view: &[f32; 16], projection: &[f32; 16]) -> [[f32; 4]; 4] {
    let m = crate::matrices::mul_m4(projection, view);
    let row = |r: usize| [m[r], m[4 + r], m[8 + r], m[12 + r]];
    let (x, y, w) = (row(0), row(1), row(3));
    let mut planes = [
        std::array::from_fn(|i| w[i] + x[i]),
        std::array::from_fn(|i| w[i] - x[i]),
        std::array::from_fn(|i| w[i] + y[i]),
        std::array::from_fn(|i| w[i] - y[i]),
    ];
    for plane in &mut planes {
        let length = (plane[0] * plane[0] + plane[1] * plane[1] + plane[2] * plane[2]).sqrt();
        if length > 1e-12 {
            for value in plane.iter_mut() {
                *value /= length;
            }
        }
    }
    planes
}

/// The light whose shadows the plants cast this frame: the sun while it is
/// up, else the moon while it shines, as (direction of travel, is the moon).
fn shadowing_light(globals: &super::GlobalUniformsGpu) -> Option<([f32; 3], bool)> {
    let sun = [globals.sun_direction[0], globals.sun_direction[1], globals.sun_direction[2]];
    let moon = [globals.moon_direction[0], globals.moon_direction[1], globals.moon_direction[2]];
    if globals.settings_a[0] > 0.01 && sun[1] < -0.01 {
        Some((sun, false))
    } else if globals.atmosphere[1] > 0.001 && moon[1] < -0.01 {
        Some((moon, true))
    } else {
        None
    }
}

/// The cull pass's view of the cascades: the light frame and each box.
fn cascade_culling(cull: &mut CullUniform, light: &LightBasis, cascades: &[Cascade; SHADOW_CASCADES]) {
    let extend = |v: [f32; 3]| [v[0], v[1], v[2], 0.0];
    cull.light_right = extend(light.right);
    cull.light_up = extend(light.up);
    cull.light_forward = extend(light.forward);
    for (index, cascade) in cascades.iter().enumerate() {
        cull.cascades[index] = [cascade.centre[0], cascade.centre[1], cascade.half_extent, cascade.far];
        cull.cascade_near[index] = cascade.near;
    }
    cull.counts[3] = SHADOW_CASCADES as u32;
}

pub fn forest_vegetation_pass(world: &World, mut ctx: RenderContext) {
    // The composite reads plant shadows only once this frame has drawn them.
    let shadow_targets = world
        .get_resource::<VegetationShadowMaps>()
        .and_then(|maps| maps.targets.as_ref());
    if let (Some(targets), Some(queue)) = (shadow_targets, world.get_resource::<RenderQueue>()) {
        queue.write_buffer(&targets.uniform, 0, bytemuck::bytes_of(&ShadowUniform::disabled()));
    }
    let Some(state) = world.get_resource::<VegetationNodeState>() else {
        return;
    };
    let Some(view) = world.get_resource::<ExtractedForestView>() else {
        return;
    };
    if !view.settings.vegetation_enabled {
        if let Some(terrain) = world.get_resource::<terrain_node::TerrainNodeState>() {
            let gbuffer = terrain.gbuffer.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(gbuffer) = gbuffer.as_ref() {
                clear_canopy(&mut state.0.lock().unwrap_or_else(|e| e.into_inner()), gbuffer, &mut ctx);
            }
        }
        return;
    }
    let (Some(terrain), Some(globals), Some(textures), Some(device), Some(queue), Some(cache)) = (
        world.get_resource::<terrain_node::TerrainNodeState>(),
        world.get_resource::<ForestGlobals>(),
        world.get_resource::<super::gpu_textures::GpuWorldTexturesOption>(),
        world.get_resource::<RenderDevice>(),
        world.get_resource::<RenderQueue>(),
        world.get_resource::<PipelineCache>(),
    ) else {
        return;
    };
    let Some(textures) = textures.0.as_deref() else {
        return;
    };
    let gbuffer = terrain.gbuffer.lock().unwrap_or_else(|e| e.into_inner());
    let Some(gbuffer) = gbuffer.as_ref() else {
        return;
    };
    let mut guard = state.0.lock().unwrap_or_else(|e| e.into_inner());
    if guard
        .as_ref()
        .is_none_or(|r| r.snapshot.as_ref().is_none_or(|s| s.plant_count == 0))
    {
        clear_canopy(&mut guard, gbuffer, &mut ctx);
        return;
    }
    let Some(resources) = guard.as_mut() else {
        return;
    };
    let library = &resources.library;
    let Some(snapshot) = resources.snapshot.as_ref() else {
        return;
    };
    let (Some(cull_pipeline), [Some(single_sided), Some(double_sided)]) = (
        cache.get_compute_pipeline(library.cull_pipeline),
        library.draw_pipelines.map(|pipeline| cache.get_render_pipeline(pipeline)),
    ) else {
        return;
    };
    let draw_pipelines = [single_sided, double_sided];

    // Shadows, when wanted and possible this frame.
    let camera = [
        globals.globals.camera_position[0],
        globals.globals.camera_position[1],
        globals.globals.camera_position[2],
    ];
    let shadow_pipelines = library.shadow_pipelines.map(|pipeline| cache.get_render_pipeline(pipeline));
    let shadow_pipelines = shadow_pipelines[0].zip(shadow_pipelines[1]).map(|(single, double)| [single, double]);
    let wanted_shadow_targets = shadow_targets
        .filter(|targets| targets.resolutions[0] > 1 && view.settings.vegetation_shadows);
    let shadow_inputs = wanted_shadow_targets
        .zip(shadow_pipelines)
        .zip(shadowing_light(&globals.globals));
    let assemble_shadows = |((targets, pipeline), (direction, moon))| {
        let light = light_basis(direction);
        let cascades = fit_cascades(camera, &view.shadow_receiver_heights, &light);
        (targets, pipeline, light, cascades, moon)
    };
    let shadows = shadow_inputs.map(assemble_shadows);

    // Per-frame uniforms.
    let eye = view.player_position;
    let mut cull = CullUniform {
        planes: frustum_planes(&globals.globals.view, &globals.globals.projection),
        camera: [camera[0], camera[1], camera[2], view.settings.vegetation_detail.clamp(0.25, 4.0)],
        habitat_mapping: gbuffer.grass_habitat.mapping,
        counts: [
            snapshot.plant_count,
            snapshot.plant_count,
            gbuffer.grass_habitat.ready as u32,
            0,
        ],
        ground: [SEA_LEVEL + LOWEST_ROOT, STEEPEST_ROOT, DEEPEST_FURROW, CULLING_SLACK],
        ..Default::default()
    };
    if let Some((_, _, light, cascades, _)) = &shadows {
        cascade_culling(&mut cull, light, cascades);
        for (index, cascade) in cascades.iter().enumerate() {
            let uniform = CascadeUniform {
                view_projection: cascade.view_projection,
                light: [light.forward[0], light.forward[1], light.forward[2], cascade.texel],
            };
            queue.write_buffer(&library.cascade_buffers[index], 0, bytemuck::bytes_of(&uniform));
        }
    }
    queue.write_buffer(&library.cull_buffer, 0, bytemuck::bytes_of(&cull));
    let stage = TerrainStageUniforms::build(
        bevy::math::Mat4::IDENTITY.to_cols_array(),
        0,
        eye,
        view.lookup_minimum,
        [eye[0], eye[2]],
    );
    queue.write_buffer(&library.stage_buffer, 0, bytemuck::bytes_of(&stage));
    let elapsed = world
        .get_resource::<super::water_node::ExtractedWater>()
        .map_or(0.0, |water| water.elapsed);
    let direction = view.settings.cloud_wind_direction_degrees.to_radians();
    let conditions = view.weather.conditions(&view.settings);
    let frame = FrameUniform {
        wind: [
            direction.cos(),
            direction.sin(),
            elapsed,
            (conditions.wind_speed / 18.0).clamp(0.0, 2.0) * (1.0 + 1.4 * conditions.gust_strength),
        ],
        params: cull.habitat_mapping,
    };
    queue.write_buffer(&library.frame_buffer, 0, bytemuck::bytes_of(&frame));

    // The terrain inputs, rebound each frame: the habitat capture is
    // recreated with the G-buffer on resize.
    let terrain_entries = [
        wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(&textures.noise_view),
        },
        wgpu::BindGroupEntry {
            binding: 1,
            resource: wgpu::BindingResource::Sampler(&library.noise_sampler),
        },
        wgpu::BindGroupEntry {
            binding: 2,
            resource: wgpu::BindingResource::TextureView(&textures.height_atlas_view),
        },
        wgpu::BindGroupEntry {
            binding: 3,
            resource: wgpu::BindingResource::Sampler(&library.atlas_sampler),
        },
        wgpu::BindGroupEntry {
            binding: 4,
            resource: wgpu::BindingResource::TextureView(&textures.lookup_view),
        },
        wgpu::BindGroupEntry {
            binding: 5,
            resource: wgpu::BindingResource::TextureView(&textures.blend_mask_view),
        },
        wgpu::BindGroupEntry {
            binding: 6,
            resource: wgpu::BindingResource::Sampler(&library.mask_sampler),
        },
        wgpu::BindGroupEntry {
            binding: 7,
            resource: library.stage_buffer.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 8,
            resource: wgpu::BindingResource::TextureView(&gbuffer.grass_habitat_view),
        },
        wgpu::BindGroupEntry {
            binding: 9,
            resource: wgpu::BindingResource::Sampler(&library.habitat_sampler),
        },
        wgpu::BindGroupEntry {
            binding: 10,
            resource: textures.river_grid.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 11,
            resource: textures.river_segments.as_entire_binding(),
        },
    ];
    let terrain_group = super::bind_group(
        device,
        cache,
        "vegetation_terrain",
        &library.terrain_layout,
        &terrain_entries,
    );

    {
        let encoder = ctx.command_encoder();
        encoder.copy_buffer_to_buffer(&library.args_template, 0, &library.args, 0, library.args_bytes);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("forest_vegetation_cull"),
            timestamp_writes: None,
        });
        pass.set_pipeline(cull_pipeline);
        pass.set_bind_group(0, Some(&*library.cull_globals), &[]);
        pass.set_bind_group(1, Some(&*terrain_group), &[]);
        pass.set_bind_group(2, Some(&*snapshot.cull_group), &[]);
        pass.dispatch_workgroups(snapshot.plant_count.div_ceil(WORKGROUP_SIZE), 1, 1);
    }

    {
        let attachments = [&gbuffer.position_view, &gbuffer.normal_view, &gbuffer.albedo_view].map(|target| {
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
            label: Some("forest_vegetation_gbuffer"),
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
        pass.set_bind_group(0, &library.draw_globals, &[]);
        pass.set_bind_group(1, &snapshot.frame_group, &[]);
        pass.set_vertex_buffer(0, library.vertices.slice(..));
        pass.set_index_buffer(library.indices.slice(..), wgpu::IndexFormat::Uint32);
        for &(index, offset) in &snapshot.active {
            let draw = &library.draws[index];
            let region = snapshot.region_bytes[draw.model as usize];
            let material = &library.materials[draw.material];
            pass.set_render_pipeline(draw_pipelines[usize::from(material.double_sided)]);
            pass.set_bind_group(2, &material.bind_group, &[]);
            pass.set_vertex_buffer(1, snapshot.visible.slice(offset..offset + region));
            pass.draw_indexed_indirect(&library.args, draw.args_offset);
        }
    }

    // Shade under the crowns for the grass, which is drawn next. It only
    // changes with the capture it is drawn over and the plants it is drawn
    // from, so it is not redrawn every frame.
    let stamp = (gbuffer.grass_habitat.generation, snapshot.generation);
    if gbuffer.grass_habitat.ready
        && resources.canopy_stamp != Some(stamp)
        && let Some(canopy_pipeline) = cache.get_render_pipeline(library.canopy_pipeline)
    {
        terrain_node::copy_ground_average(ctx.command_encoder(), gbuffer);
        let canopy_pass_descriptor = wgpu::RenderPassDescriptor {
            label: Some("forest_vegetation_canopy"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &gbuffer.grass_ground_average_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        };
        let mut pass = ctx.begin_tracked_render_pass(canopy_pass_descriptor);
        pass.set_render_pipeline(canopy_pipeline);
        pass.set_bind_group(0, &library.draw_globals, &[]);
        pass.set_bind_group(1, &snapshot.frame_group, &[]);
        pass.draw(0..6, 0..snapshot.plant_count);
        resources.canopy_stamp = Some(stamp);
    }

    let Some((targets, shadow_pipelines, light, cascades, moon)) = shadows else {
        return;
    };
    for (index, layer) in targets.cascade_views.iter().enumerate() {
        let shadow_pass_descriptor = wgpu::RenderPassDescriptor {
            label: Some("forest_vegetation_shadow"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: layer,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        };
        let mut pass = ctx.begin_tracked_render_pass(shadow_pass_descriptor);
        pass.set_bind_group(0, &library.cascade_groups[index], &[]);
        pass.set_bind_group(1, &snapshot.frame_group, &[]);
        pass.set_vertex_buffer(0, library.vertices.slice(..));
        pass.set_index_buffer(library.indices.slice(..), wgpu::IndexFormat::Uint32);
        for &(draw_index, offset) in &snapshot.shadow_active[index] {
            let draw = &library.shadow_draws[draw_index];
            let region = snapshot.region_bytes[draw.model as usize];
            let material = &library.materials[draw.material];
            pass.set_render_pipeline(shadow_pipelines[usize::from(material.double_sided)]);
            pass.set_bind_group(2, &material.bind_group, &[]);
            pass.set_vertex_buffer(1, snapshot.visible.slice(offset..offset + region));
            pass.draw_indexed_indirect(&library.args, draw.args_offset);
        }
    }
    queue.write_buffer(&targets.uniform, 0, bytemuck::bytes_of(&ShadowUniform::new(&cascades, &light, moon)));
}

pub fn register_vegetation_systems(app: &mut bevy::app::SubApp) {
    app.init_resource::<VegetationNodeState>();
    app.init_resource::<ExtractedVegetation>();
    app.init_resource::<VegetationShadowMaps>();
    app.add_systems(ExtractSchedule, extract_vegetation);
    app.add_systems(
        bevy::render::Render,
        (
            prepare_vegetation
                .in_set(bevy::render::RenderSystems::Prepare)
                .after(super::prepare_forest_globals),
            super::vegetation_shadows::prepare_vegetation_shadow_maps.in_set(bevy::render::RenderSystems::Prepare),
        ),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::vegetation_shadows::ShadowUniform;

    /// Size and member offsets of a Rust mirror of a WGSL struct.
    macro_rules! layout {
        ($ty:ty, $($field:ident),+) => {
            (
                std::mem::size_of::<$ty>(),
                vec![$((stringify!($field).to_string(), std::mem::offset_of!($ty, $field))),+],
            )
        };
    }

    /// Size and member offsets of a struct as naga lays it out.
    fn wgsl_layout(source: &str, name: &str) -> (usize, Vec<(String, usize)>) {
        let module = naga::front::wgsl::parse_str(source).expect("the shader parses");
        for (_, ty) in module.types.iter() {
            if let (Some(name_here), naga::TypeInner::Struct { members, span }) = (ty.name.as_deref(), &ty.inner)
                && name_here == name
            {
                let members = members
                    .iter()
                    .map(|member| (member.name.clone().unwrap_or_default(), member.offset as usize))
                    .collect();
                return (*span as usize, members);
            }
        }
        panic!("no struct {name}");
    }

    /// Every buffer the plants share with the GPU is mirrored by hand on both
    /// sides; a field added to one alone would shift everything after it.
    #[test]
    fn gpu_structs_match_their_wgsl_mirrors() {
        let cull = include_str!("../../assets/shaders/vegetation-cull.wgsl");
        let draw = include_str!("../../assets/shaders/vegetation.wgsl");
        let composite = include_str!("../../assets/shaders/composite.wgsl");
        let water = include_str!("../../assets/shaders/water.wgsl");
        let model = layout!(
            ModelParams, height, crown_radius, crown_base, bound_radius, lod_end, max_distance, lod_count,
            habitat, region, draw_word, draw_count, shadow_word, shadow_count
        );
        let instance = layout!(DrawInstance, position, scale, yaw, seed, fade, model);
        let checks = [
            (cull, "PlantInstance", layout!(PlantInstance, position, scale, yaw, seed, model, layer)),
            (cull, "ModelParams", model.clone()),
            (draw, "ModelParams", model),
            (cull, "DrawInstance", instance),
            (
                cull,
                "CullUniform",
                layout!(
                    CullUniform, planes, camera, habitat_mapping, counts, ground, light_right, light_up,
                    light_forward, cascades, cascade_near
                ),
            ),
            (draw, "ShadowCascade", layout!(CascadeUniform, view_projection, light)),
            (draw, "VegetationFrame", layout!(FrameUniform, wind, params)),
            (draw, "PlantMaterial", layout!(MaterialUniform, base_color_factor, params, surface)),
            (composite, "VegetationShadows", layout!(ShadowUniform, view_projection, splits, texel, light, params)),
            (water, "VegetationShadows", layout!(ShadowUniform, view_projection, splits, texel, light, params)),
        ];
        for (source, name, rust) in checks {
            assert_eq!(wgsl_layout(source, name), rust, "{name}");
        }
    }

    /// The cull pass seats roots with the same height model the clipmap
    /// draws the ground with, or trees float and sink as erosion streams in.
    #[test]
    fn plants_root_on_the_terrain_the_clipmap_draws() {
        let common = include_str!("../../assets/shaders/terrain-height-functions.wgslinc").trim();
        for source in [
            include_str!("../../assets/shaders/terrain-vs.wgsl"),
            include_str!("../../assets/shaders/vegetation-cull.wgsl"),
        ] {
            assert!(source.contains(common), "terrain height model diverged");
        }
    }

    /// The rivers carve the clipmap, keep the plants out of the water, and
    /// shape the CPU's footing and erosion bases (src/rivers/carve.rs) from
    /// one envelope; the pasted copies of its shader must not drift apart.
    #[test]
    fn rivers_carve_the_same_ground_everywhere() {
        let common = include_str!("../../assets/shaders/river-functions.wgslinc").trim();
        for source in [
            include_str!("../../assets/shaders/terrain-vs.wgsl"),
            include_str!("../../assets/shaders/terrain-fs.wgsl"),
            include_str!("../../assets/shaders/vegetation-cull.wgsl"),
        ] {
            assert!(source.contains(common), "river carve diverged");
        }
    }

    #[test]
    fn frustum_planes_contain_what_the_camera_sees() {
        let eye = bevy::math::Vec3::new(10.0, 5.0, -3.0);
        let target = eye + bevy::math::Vec3::new(0.0, 0.0, -1.0);
        let view = crate::matrices::view_matrix(eye, target, bevy::math::Vec3::Y);
        let projection = crate::matrices::perspective(68.0, 16.0 / 9.0, 0.1, 5800.0);
        let planes = frustum_planes(&view, &projection);
        let inside = |p: [f32; 3]| planes.iter().all(|pl| pl[0] * p[0] + pl[1] * p[1] + pl[2] * p[2] + pl[3] >= 0.0);
        assert!(inside([10.0, 5.0, -50.0]));
        assert!(inside([30.0, 5.0, -50.0]));
        assert!(!inside([10.0, 5.0, 50.0]), "behind the camera");
        assert!(!inside([200.0, 5.0, -50.0]), "far to the right");
        assert!(!inside([10.0, 80.0, -50.0]), "far above");
    }
}
