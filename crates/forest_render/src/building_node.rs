//! The buildings in the render world.
//!
//! The kit's meshes, materials and textures are uploaded once; the pads are
//! written for the ground shaders whenever they or the grid around the player
//! move; and every frame the placed buildings and their doors are culled
//! against the view, gathered into one instance list per model (rooms and
//! stairs only near enough to be seen through a window or a door) and drawn
//! into the G-buffer between the terrain and the plants, then depth-only into
//! the shadow cascades the plants were drawn into. The roofs over the camera
//! are handed to the rain, which falls only outside them.

use super::vegetation_node::{
    frustum_planes, sampler, sampler_entry, shadowing_light, texture_entry, uniform_entry,
    upload_texture,
};
use super::vegetation_shadows::{
    Cascade, LightBasis, SHADOW_CASCADES, SHADOW_FORMAT, SHADOW_FRONT_FACE, SHADOW_SPLITS,
    ShadowUniform, VegetationShadowMaps, fit_cascades, light_basis,
};
use super::{ExtractedForestView, ForestGlobals, ForestShaderHandles, terrain_node};
use bevy::math::{Vec2, Vec3};
use bevy::mesh::VertexBufferLayout;
use bevy::prelude::*;
use bevy::render::render_resource::{
    BindGroup, BindGroupLayoutDescriptor, Buffer, CachedRenderPipelineId, FragmentState,
    PipelineCache, RenderPipelineDescriptor, VertexState,
};
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue, ViewQuery};
use bevy::render::view::ViewTarget;
use bevy::render::{ExtractSchedule, MainWorld};
use forest_buildings::assets::{BuildingLibrary, BuildingPrimitive, BuildingVertex};
use forest_buildings::collision::BuildingFrame;
use forest_buildings::pads::{GPU_GRID_SNAP, PadIndex};
use forest_buildings::{BuildingField, BuildingSnapshot};
use forest_vegetation::assets::PreparedTexture;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Buildings are drawn to this far: a house is a few pixels across there.
pub const DRAW_DISTANCE: f32 = 1800.0;
/// Rooms, stairs and ceiling joists are drawn only this near.
const INTERIOR_DISTANCE: f32 = 140.0;
/// Doors cast shadows only into the cascades this near.
const DOOR_SHADOW_CASCADES: usize = 2;
/// Nearby rooms and stairs cast shadows into the detailed cascades.
const INTERIOR_SHADOW_CASCADES: usize = 2;
/// Roofs within this of the camera keep the rain off what is under them.
const SHELTER_REACH: f32 = 70.0;
/// Roof boxes the rain is given at most.
pub const MAX_SHELTERS: usize = 32;

// ---------------------------------------------------------------------------
// GPU layouts (each mirrored by a WGSL struct)
// ---------------------------------------------------------------------------

/// One drawn building or door: its origin and turn about +Y.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct BuildingInstance {
    origin: [f32; 3],
    turn: [f32; 2],
    _pad: [f32; 3],
}

/// `ShadowCascade` in building.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct CascadeUniform {
    view_projection: [f32; 16],
    light: [f32; 4],
}

/// `BuildingMaterial` in building.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct MaterialUniform {
    base_color_factor: [f32; 4],
    params: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<BuildingInstance>() == 32);
const _: () = assert!(std::mem::size_of::<CascadeUniform>() == 80);
const _: () = assert!(std::mem::size_of::<MaterialUniform>() == 32);
const INSTANCE_STRIDE: u64 = std::mem::size_of::<BuildingInstance>() as u64;

// ---------------------------------------------------------------------------
// Extraction
// ---------------------------------------------------------------------------

/// The main world's library, placed buildings and pads, copied by pointer,
/// and the doors' angles, copied every frame.
#[derive(Resource, Default)]
pub struct ExtractedBuildings {
    enabled: bool,
    library: Option<Arc<BuildingLibrary>>,
    textures: Option<Arc<Vec<PreparedTexture>>>,
    snapshot: Option<Arc<BuildingSnapshot>>,
    pads: Option<Arc<PadIndex>>,
    uploaded: Option<Arc<AtomicU64>>,
    /// (snapshot index, door index, angle) of every door not shut.
    open_doors: Vec<(u32, u16, f32)>,
}

impl ExtractedBuildings {
    /// Whether the kit loaded and buildings are placed.
    pub fn enabled(&self) -> bool {
        self.enabled
    }
}

fn extract_buildings(main_world: Res<MainWorld>, mut extracted: ResMut<ExtractedBuildings>) {
    let Some(field) = main_world.get_resource::<BuildingField>() else {
        return;
    };
    extracted.enabled = field.enabled();
    if extracted.library.is_none() {
        extracted.library = field.library().cloned();
    }
    if extracted.textures.is_none() {
        extracted.textures = field.textures().cloned();
    }
    let generation = field.snapshot().generation;
    if extracted
        .snapshot
        .as_ref()
        .map(|snapshot| snapshot.generation)
        != Some(generation)
    {
        extracted.snapshot = Some(field.snapshot().clone());
    }
    if extracted.pads.as_ref().map(Arc::as_ptr) != Some(Arc::as_ptr(field.pads())) {
        extracted.pads = Some(field.pads().clone());
    }
    if extracted.uploaded.is_none() {
        extracted.uploaded = Some(field.uploaded.clone());
    }
    extracted.open_doors = field.open_doors();
}

// ---------------------------------------------------------------------------
// Pads
// ---------------------------------------------------------------------------

/// Which pads and which grid centre the GPU's pad buffer holds.
#[derive(Resource, Default)]
struct PadUpload {
    pads: usize,
    centre: Option<[i64; 2]>,
}

/// Write the pads over the GPU's buffer when they change or the player moves
/// to the next grid centre. The terrain revision moves with them, so the
/// heightfields and the grass capture are taken again over the levelled
/// ground.
fn upload_pads(
    textures: Res<super::gpu_textures::GpuWorldTexturesOption>,
    extracted: Res<ExtractedBuildings>,
    view: Res<ExtractedForestView>,
    queue: Res<RenderQueue>,
    mut upload: ResMut<PadUpload>,
) {
    let Some(textures) = textures.0.as_deref() else {
        return;
    };
    let Some(pads) = extracted.pads.as_ref() else {
        return;
    };
    let address = Arc::as_ptr(pads) as usize;
    let eye = view.player_position;
    let centre = [
        (eye[0] / GPU_GRID_SNAP).round() as i64,
        (eye[2] / GPU_GRID_SNAP).round() as i64,
    ];
    if upload.pads == address && upload.centre == Some(centre) {
        return;
    }
    let words = pads.gpu_words([eye[0], eye[2]]);
    queue.write_buffer(&textures.building_pads, 0, bytemuck::cast_slice(&words));
    textures.revision.note_atlas_upload();
    upload.pads = address;
    upload.centre = Some(centre);
}

// ---------------------------------------------------------------------------
// GPU resources
// ---------------------------------------------------------------------------

/// Built once, when the library and its textures arrive.
struct BuildingGpu {
    vertices: Buffer,
    indices: Buffer,
    /// A bind group per material; none for glass, which is not in the
    /// G-buffer but drawn over the lit scene.
    materials: Vec<Option<BindGroup>>,

    draw_pipeline: CachedRenderPipelineId,
    shadow_pipeline: CachedRenderPipelineId,
    /// The glass pipeline per view target format, queued the first time a
    /// view of that format draws glass.
    glass_pipelines: HashMap<wgpu::TextureFormat, CachedRenderPipelineId>,

    vertex_buffers: Vec<VertexBufferLayout>,
    globals_group: BindGroup,
    cascade_buffers: Vec<Buffer>,
    cascade_groups: Vec<BindGroup>,
    instances: Option<Buffer>,
    instance_capacity: u64,
}

/// One indexed draw over a run of the instance buffer.
#[derive(Clone, Copy, Debug)]
struct Draw {
    primitive: BuildingPrimitive,
    instance_offset: u64,
    instance_count: u32,
}

/// The cascades fitted this frame, as the plants fitted them.
struct ShadowFrame {
    cascades: [Cascade; SHADOW_CASCADES],
    light: LightBasis,
    moon: bool,
}

#[derive(Default)]
struct BuildingFrameDraws {
    gbuffer: Vec<Draw>,
    /// The window glass of the buildings drawn, over the lit scene.
    glass: Vec<Draw>,

    shadows: [Vec<Draw>; SHADOW_CASCADES],
    shadow: Option<ShadowFrame>,
}

#[derive(Default)]
struct BuildingState {
    gpu: Option<BuildingGpu>,
    frame: BuildingFrameDraws,
}

#[derive(Resource, Default)]
pub struct BuildingNodeState(Mutex<BuildingState>);

/// The roofs over and near the camera, for the rain: per box, its centre XZ
/// and turn (cosine, sine), then its half extents, the ground under it and
/// the top of its roof.
#[derive(Resource, Default)]
pub struct BuildingShelters {
    pub boxes: Vec<[[f32; 4]; 2]>,
}

fn build_gpu(
    device: &RenderDevice,
    queue: &RenderQueue,
    cache: &PipelineCache,
    shaders: &ForestShaderHandles,
    global_buffer: &wgpu::Buffer,
    library: &BuildingLibrary,
    textures: &[PreparedTexture],
) -> BuildingGpu {
    let start = std::time::Instant::now();
    let vertices = device.create_buffer_with_data(&wgpu::util::BufferInitDescriptor {
        label: Some("building_vertices"),
        contents: bytemuck::cast_slice(&library.vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let indices = device.create_buffer_with_data(&wgpu::util::BufferInitDescriptor {
        label: Some("building_indices"),
        contents: bytemuck::cast_slice(&library.indices),
        usage: wgpu::BufferUsages::INDEX,
    });
    let views: Vec<wgpu::TextureView> = textures
        .iter()
        .map(|texture| upload_texture(device, queue, texture))
        .collect();
    let material_sampler = sampler(
        device,
        "building_material_sampler",
        wgpu::AddressMode::Repeat,
        8,
    );
    let material_layout = BindGroupLayoutDescriptor::new(
        "building_material_layout",
        &[
            texture_entry(0, wgpu::ShaderStages::FRAGMENT),
            texture_entry(1, wgpu::ShaderStages::FRAGMENT),
            sampler_entry(2, wgpu::ShaderStages::FRAGMENT),
            uniform_entry(
                3,
                wgpu::ShaderStages::FRAGMENT,
                std::mem::size_of::<MaterialUniform>() as u64,
            ),
        ],
    );
    let materials = library
        .materials
        .iter()
        .map(|material| {
            if material.glass {
                return None;
            }
            let uniform = MaterialUniform {
                base_color_factor: material.base_color_factor,
                params: [material.roughness_factor, material.normal_scale, 0.0, 0.0],
            };
            let buffer = device.create_buffer_with_data(&wgpu::util::BufferInitDescriptor {
                label: Some("building_material_uniform"),
                contents: bytemuck::bytes_of(&uniform),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let entries = [
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
            Some(super::bind_group(
                device,
                cache,
                "building_material",
                &material_layout,
                &entries,
            ))
        })
        .collect();

    let vertex_attributes = wgpu::vertex_attr_array![
        0 => Float32x3, 1 => Float32x3, 2 => Float32x2, 3 => Float32x4, 4 => Float32x2
    ];
    let instance_attributes = wgpu::vertex_attr_array![5 => Float32x3, 6 => Float32x2];
    let vertex_buffers = vec![
        VertexBufferLayout {
            array_stride: std::mem::size_of::<BuildingVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: vertex_attributes.to_vec(),
        },
        VertexBufferLayout {
            array_stride: INSTANCE_STRIDE,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: instance_attributes.to_vec(),
        },
    ];
    let globals_layout = super::globals_layout();
    let glass_vertex_buffers = vertex_buffers.clone();
    let draw_descriptor = RenderPipelineDescriptor {
        label: Some("forest_building_pipeline".into()),
        layout: vec![globals_layout.clone(), material_layout],
        immediate_size: 0,
        vertex: VertexState {
            shader: shaders.building.clone(),
            shader_defs: vec![],
            entry_point: Some("vs_main".into()),
            buffers: vertex_buffers.clone(),
        },
        // The kit's walls are single skins, seen from either side.
        primitive: wgpu::PrimitiveState {
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Some(true),
            // Reverse-Z, as the terrain and plants.
            depth_compare: Some(wgpu::CompareFunction::GreaterEqual),
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: Default::default(),
        fragment: Some(FragmentState {
            shader: shaders.building.clone(),
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
    let draw_pipeline = cache.queue_render_pipeline(draw_descriptor);
    let cascade_layout = BindGroupLayoutDescriptor::new(
        "building_cascade_layout",
        &[uniform_entry(
            1,
            wgpu::ShaderStages::VERTEX_FRAGMENT,
            std::mem::size_of::<CascadeUniform>() as u64,
        )],
    );
    let shadow_descriptor = RenderPipelineDescriptor {
        label: Some("forest_building_shadow_pipeline".into()),
        layout: vec![cascade_layout.clone()],
        immediate_size: 0,
        vertex: VertexState {
            shader: shaders.building.clone(),
            shader_defs: vec![],
            entry_point: Some("vs_shadow".into()),
            buffers: vertex_buffers,
        },
        primitive: wgpu::PrimitiveState {
            front_face: SHADOW_FRONT_FACE,
            cull_mode: None,
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
            shader: shaders.building.clone(),
            shader_defs: vec![],
            entry_point: Some("fs_shadow".into()),
            targets: vec![],
        }),
        zero_initialize_workgroup_memory: false,
    };
    let shadow_pipeline = cache.queue_render_pipeline(shadow_descriptor);
    let cascade_buffers: Vec<Buffer> = (0..SHADOW_CASCADES)
        .map(|_| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("building_cascade"),
                size: std::mem::size_of::<CascadeUniform>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        })
        .collect();
    let cascade_groups = cascade_buffers
        .iter()
        .map(|buffer| {
            let entries = [wgpu::BindGroupEntry {
                binding: 1,
                resource: buffer.as_entire_binding(),
            }];
            super::bind_group(device, cache, "building_cascade", &cascade_layout, &entries)
        })
        .collect();
    let globals_group = super::bind_group(
        device,
        cache,
        "building_globals",
        &globals_layout,
        &super::globals_bind_group_entries(global_buffer),
    );
    log::info!(
        "BUILDINGS: {} models, {} materials, {} textures and {} triangles uploaded in {:.1?}",
        library.models.len(),
        library.materials.len(),
        views.len(),
        library.triangle_count(),
        start.elapsed()
    );
    BuildingGpu {
        vertices,
        indices,
        materials,
        draw_pipeline,
        shadow_pipeline,
        glass_pipelines: HashMap::new(),
        vertex_buffers: glass_vertex_buffers,
        globals_group,
        cascade_buffers,
        cascade_groups,
        instances: None,
        instance_capacity: 0,
    }
}

// ---------------------------------------------------------------------------
// Per-frame preparation
// ---------------------------------------------------------------------------

/// Instances of one draw group, gathered before they are laid out.
#[derive(Default)]
struct InstanceGroups {
    /// (model or door mesh, interior?) -> instances, for the G-buffer.
    gbuffer: HashMap<(Mesh, bool), Vec<BuildingInstance>>,

    /// The same per cascade, with nearby interiors in the detailed maps.
    shadows: [HashMap<(Mesh, bool), Vec<BuildingInstance>>; SHADOW_CASCADES],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum Mesh {
    Building(u32),
    Door(u32),
}

fn instance_of(origin: Vec3, yaw: f32) -> BuildingInstance {
    let (sine, cosine) = yaw.sin_cos();
    BuildingInstance {
        origin: origin.to_array(),
        turn: [cosine, sine],
        _pad: [0.0; 3],
    }
}

/// Whether a sphere is not wholly outside any side plane of the view.
fn in_frustum(planes: &[[f32; 4]; 4], centre: Vec3, radius: f32) -> bool {
    planes.iter().all(|plane| {
        plane[0] * centre.x + plane[1] * centre.y + plane[2] * centre.z + plane[3] >= -radius
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn prepare_buildings(
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    cache: Res<PipelineCache>,
    globals: Res<ForestGlobals>,
    shaders: Res<ForestShaderHandles>,
    view: Res<ExtractedForestView>,
    extracted: Res<ExtractedBuildings>,
    shadow_maps: Res<VegetationShadowMaps>,
    state: Res<BuildingNodeState>,
    mut shelters: ResMut<BuildingShelters>,
) {
    shelters.boxes.clear();
    let mut state = state.0.lock().unwrap_or_else(|e| e.into_inner());
    state.frame = BuildingFrameDraws::default();
    let (Some(library), Some(snapshot)) = (extracted.library.as_ref(), extracted.snapshot.as_ref())
    else {
        return;
    };
    // Hidden from the menu: nothing drawn, and no roofs over the rain.
    if !view.settings.buildings_visible {
        if let Some(uploaded) = &extracted.uploaded {
            uploaded.store(snapshot.generation, Ordering::Release);
        }
        return;
    }
    if state.gpu.is_none() {
        let (Some(textures), Some(global_buffer)) =
            (extracted.textures.as_ref(), globals.buffer.as_ref())
        else {
            return;
        };
        state.gpu = Some(build_gpu(
            &device,
            &queue,
            &cache,
            &shaders,
            global_buffer,
            library,
            textures,
        ));
    }
    if let Some(uploaded) = &extracted.uploaded {
        uploaded.store(snapshot.generation, Ordering::Release);
    }

    let planes = frustum_planes(&globals.globals.view, &globals.globals.projection);
    let open: HashMap<(u32, u16), f32> = extracted
        .open_doors
        .iter()
        .map(|&(building, door, angle)| ((building, door), angle))
        .collect();
    let wants_shadows = view.settings.vegetation_shadows
        && shadow_maps
            .targets
            .as_ref()
            .is_some_and(|targets| targets.resolutions[0] > 1);
    let shadow = shadowing_light(&globals.globals)
        .filter(|_| wants_shadows)
        .map(|(direction, moon)| {
            let light = light_basis(direction);
            let cascades =
                fit_cascades(view.player_position, &view.shadow_receiver_heights, &light);
            ShadowFrame {
                cascades,
                light,
                moon,
            }
        });
    let sight = Sight {
        camera: Vec3::from_array(view.player_position),
        planes,
        shadows: shadow.is_some(),
    };
    let groups = gather_instances(library, snapshot, &sight, &open, &mut shelters.boxes);

    let Some(gpu) = state.gpu.as_mut() else {
        return;
    };
    let (instances, mut frame_draws) = lay_out_draws(library, &groups);
    upload_instances(&device, &queue, gpu, &instances);
    if let Some(shadow) = &shadow {
        upload_cascades(&queue, gpu, shadow);
    }
    frame_draws.shadow = shadow;
    state.frame = frame_draws;
}

/// Where the camera is and what it and the light can see.
struct Sight {
    camera: Vec3,
    planes: [[f32; 4]; 4],
    shadows: bool,
}

/// Every placed building and door the view or the shadow cascades need this
/// frame, gathered per mesh, and the roofs near the camera the rain must keep
/// out from under.
fn gather_instances(
    library: &BuildingLibrary,
    snapshot: &BuildingSnapshot,
    sight: &Sight,
    open: &HashMap<(u32, u16), f32>,
    shelters: &mut Vec<[[f32; 4]; 2]>,
) -> InstanceGroups {
    let mut groups = InstanceGroups::default();
    for (index, placement) in snapshot.placements.iter().enumerate() {
        let model = &library.models[placement.model as usize];
        let frame = BuildingFrame::new(placement.origin, placement.yaw);
        let height = model.maximum.y;
        let centre = Vec3::new(
            placement.pad.centre.x,
            placement.origin.y + height * 0.5,
            placement.pad.centre.y,
        );
        let radius = Vec2::new(model.radius(), height * 0.5).length();
        let distance = Vec2::new(centre.x - sight.camera.x, centre.z - sight.camera.z).length();
        let nearby_interior = distance < INTERIOR_DISTANCE + radius;
        let drawn = distance < DRAW_DISTANCE + radius && in_frustum(&sight.planes, centre, radius);
        let shadow_cascades = if sight.shadows {
            SHADOW_SPLITS[1..]
                .iter()
                .filter(|&&split| distance < split + radius + 30.0)
                .count()
        } else {
            0
        };
        if !drawn && shadow_cascades == 0 {
            continue;
        }
        let building = instance_of(placement.origin, placement.yaw);
        let door_instance = |(door_index, door): (usize, &forest_buildings::assets::DoorModel)| {
            let angle = open
                .get(&(index as u32, door_index as u16))
                .copied()
                .unwrap_or(0.0);
            let hinge = frame.to_world(door.hinge);
            (
                door.mesh as u32,
                instance_of(hinge, placement.yaw + door.yaw + angle),
            )
        };
        let doors: Vec<(u32, BuildingInstance)> =
            model.doors.iter().enumerate().map(door_instance).collect();
        if drawn {
            groups
                .gbuffer
                .entry((Mesh::Building(placement.model), false))
                .or_default()
                .push(building);
            if nearby_interior {
                groups
                    .gbuffer
                    .entry((Mesh::Building(placement.model), true))
                    .or_default()
                    .push(building);
            }
            for (mesh, instance) in &doors {
                groups
                    .gbuffer
                    .entry((Mesh::Door(*mesh), false))
                    .or_default()
                    .push(*instance);
            }
        }
        // Cascades are nested from the camera: a building in reach of one is
        // in reach of every coarser one.
        let first_cascade = SHADOW_CASCADES - shadow_cascades;
        for cascade in first_cascade..SHADOW_CASCADES {
            groups.shadows[cascade]
                .entry((Mesh::Building(placement.model), false))
                .or_default()
                .push(building);
            // Interior floors and partitions must block the sun as well as
            // the shell, including when the camera cannot see their faces.
            if nearby_interior && cascade < INTERIOR_SHADOW_CASCADES {
                groups.shadows[cascade]
                    .entry((Mesh::Building(placement.model), true))
                    .or_default()
                    .push(building);
            }
            if cascade < DOOR_SHADOW_CASCADES {
                for (mesh, instance) in &doors {
                    groups.shadows[cascade]
                        .entry((Mesh::Door(*mesh), false))
                        .or_default()
                        .push(*instance);
                }
            }
        }
        // Roofs near the camera keep the rain off.
        if distance < SHELTER_REACH + radius {
            for shelter in &model.shelters {
                if shelters.len() >= MAX_SHELTERS {
                    break;
                }
                let middle = (shelter.minimum + shelter.maximum) * 0.5;
                let half = (shelter.maximum - shelter.minimum) * 0.5;
                let world = frame.to_world(Vec3::new(middle.x, 0.0, middle.y));
                let ground = placement.origin.y - 1.0;
                let top = placement.origin.y + shelter.top;
                shelters.push([
                    [world.x, world.z, frame.cosine, frame.sine],
                    [half.x, half.y, ground, top],
                ]);
            }
        }
    }
    groups
}

/// The instances laid out in one buffer, and the draws over it: one per
/// mesh, primitive and pass. Glass is kept out of the G-buffer, so a window
/// shows the room behind it and lets the sun in, and listed for the glass
/// pass over the lit scene instead; it casts no shadow.
fn lay_out_draws(
    library: &BuildingLibrary,
    groups: &InstanceGroups,
) -> (Vec<BuildingInstance>, BuildingFrameDraws) {
    let mut instances: Vec<BuildingInstance> = Vec::new();
    let mut frame_draws = BuildingFrameDraws::default();
    let primitives_of = |mesh: Mesh, interior: bool| -> &[BuildingPrimitive] {
        match mesh {
            Mesh::Building(model) if interior => &library.models[model as usize].interior,
            Mesh::Building(model) => &library.models[model as usize].exterior,
            Mesh::Door(door) => &library.door_meshes[door as usize].primitives,
        }
    };
    // Each run of instances is laid down once; its opaque primitives are
    // drawn into the pass's list and its glass, when wanted, into the glass
    // list.
    let push_draws = |instances: &mut Vec<BuildingInstance>,
                      draws: &mut Vec<Draw>,
                      mut glass: Option<&mut Vec<Draw>>,
                      primitives: &[BuildingPrimitive],
                      run: &[BuildingInstance]| {
        let offset = instances.len() as u64 * INSTANCE_STRIDE;
        instances.extend_from_slice(run);
        for primitive in primitives {
            let draw = Draw {
                primitive: *primitive,
                instance_offset: offset,
                instance_count: run.len() as u32,
            };
            let is_glass = library.materials[primitive.material].glass;
            if !is_glass {
                draws.push(draw);
            } else if let Some(glass) = glass.as_deref_mut() {
                glass.push(draw);
            }
        }
    };
    let mut keys: Vec<(Mesh, bool)> = groups.gbuffer.keys().copied().collect();
    keys.sort();
    for key in keys {
        let run = &groups.gbuffer[&key];
        let primitives = primitives_of(key.0, key.1);
        push_draws(
            &mut instances,
            &mut frame_draws.gbuffer,
            Some(&mut frame_draws.glass),
            primitives,
            run,
        );
    }
    for cascade in 0..SHADOW_CASCADES {
        let mut keys: Vec<(Mesh, bool)> = groups.shadows[cascade].keys().copied().collect();
        keys.sort();
        for key in keys {
            let run = &groups.shadows[cascade][&key];
            let primitives = primitives_of(key.0, key.1);
            push_draws(
                &mut instances,
                &mut frame_draws.shadows[cascade],
                None,
                primitives,
                run,
            );
        }
    }
    (instances, frame_draws)
}

/// Write the frame's instances, growing the buffer when they outgrow it.
fn upload_instances(
    device: &RenderDevice,
    queue: &RenderQueue,
    gpu: &mut BuildingGpu,
    instances: &[BuildingInstance],
) {
    if instances.is_empty() {
        return;
    }
    let bytes = instances.len() as u64 * INSTANCE_STRIDE;
    if gpu.instances.is_none() || gpu.instance_capacity < bytes {
        let capacity = bytes.next_power_of_two().max(64 * INSTANCE_STRIDE);
        let descriptor = wgpu::BufferDescriptor {
            label: Some("building_instances"),
            size: capacity,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        };
        gpu.instances = Some(device.create_buffer(&descriptor));
        gpu.instance_capacity = capacity;
    }
    if let Some(buffer) = &gpu.instances {
        queue.write_buffer(buffer, 0, bytemuck::cast_slice(instances));
    }
}

/// Write the cascades the plants were fitted to, for the shadow pass.
fn upload_cascades(queue: &RenderQueue, gpu: &BuildingGpu, shadow: &ShadowFrame) {
    let light = &shadow.light;
    for (index, cascade) in shadow.cascades.iter().enumerate() {
        let uniform = CascadeUniform {
            view_projection: cascade.view_projection,
            light: [
                light.forward[0],
                light.forward[1],
                light.forward[2],
                cascade.texel,
            ],
        };
        queue.write_buffer(&gpu.cascade_buffers[index], 0, bytemuck::bytes_of(&uniform));
    }
}

// ---------------------------------------------------------------------------
// The passes
// ---------------------------------------------------------------------------

fn record_draws<'a>(
    pass: &mut bevy::render::render_phase::TrackedRenderPass<'a>,
    gpu: &'a BuildingGpu,
    draws: &[Draw],
    with_materials: bool,
) {
    let Some(instances) = gpu.instances.as_ref() else {
        return;
    };
    pass.set_vertex_buffer(0, gpu.vertices.slice(..));
    pass.set_index_buffer(gpu.indices.slice(..), wgpu::IndexFormat::Uint32);
    for draw in draws {
        if with_materials {
            let Some(material) = &gpu.materials[draw.primitive.material] else {
                continue;
            };
            pass.set_bind_group(1, material, &[]);
        }
        let end = draw.instance_offset + draw.instance_count as u64 * INSTANCE_STRIDE;
        pass.set_vertex_buffer(1, instances.slice(draw.instance_offset..end));
        let first = draw.primitive.first_index;
        let indices = first..first + draw.primitive.index_count;
        pass.draw_indexed(
            indices,
            draw.primitive.base_vertex as i32,
            0..draw.instance_count,
        );
    }
}

/// The buildings and their doors into the G-buffer, after the terrain.
pub fn forest_building_pass(world: &World, mut ctx: RenderContext) {
    let (Some(state), Some(terrain), Some(cache)) = (
        world.get_resource::<BuildingNodeState>(),
        world.get_resource::<terrain_node::TerrainNodeState>(),
        world.get_resource::<PipelineCache>(),
    ) else {
        return;
    };
    let state = state.0.lock().unwrap_or_else(|e| e.into_inner());
    let Some(gpu) = state.gpu.as_ref() else {
        return;
    };
    if state.frame.gbuffer.is_empty() {
        return;
    }
    let Some(pipeline) = cache.get_render_pipeline(gpu.draw_pipeline) else {
        return;
    };
    let gbuffer = terrain.gbuffer.lock().unwrap_or_else(|e| e.into_inner());
    let Some(gbuffer) = gbuffer.as_ref() else {
        return;
    };
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
    let pass_descriptor = wgpu::RenderPassDescriptor {
        label: Some("forest_building_gbuffer"),
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
    let mut pass = ctx.begin_tracked_render_pass(pass_descriptor);
    pass.set_render_pipeline(pipeline);
    pass.set_bind_group(0, &gpu.globals_group, &[]);
    record_draws(&mut pass, gpu, &state.frame.gbuffer, true);
}

/// The buildings into the shadow cascades, after the plants: over the plants
/// when they drew the cascades this frame, into cleared ones (and the
/// composite told of them) when they did not.
pub fn forest_building_shadow_pass(world: &World, mut ctx: RenderContext) {
    let (Some(state), Some(cache), Some(maps), Some(view), Some(queue)) = (
        world.get_resource::<BuildingNodeState>(),
        world.get_resource::<PipelineCache>(),
        world.get_resource::<VegetationShadowMaps>(),
        world.get_resource::<ExtractedForestView>(),
        world.get_resource::<RenderQueue>(),
    ) else {
        return;
    };
    let state = state.0.lock().unwrap_or_else(|e| e.into_inner());
    let (Some(gpu), Some(shadow), Some(targets)) = (
        state.gpu.as_ref(),
        state.frame.shadow.as_ref(),
        maps.targets.as_ref(),
    ) else {
        return;
    };
    if state.frame.shadows.iter().all(Vec::is_empty) {
        return;
    }
    let Some(pipeline) = cache.get_render_pipeline(gpu.shadow_pipeline) else {
        return;
    };
    let plants_drew = targets.drawn_frame.load(Ordering::Acquire) == view.frame;
    for (index, layer) in targets.cascade_views.iter().enumerate() {
        let load = if plants_drew {
            wgpu::LoadOp::Load
        } else {
            wgpu::LoadOp::Clear(1.0)
        };
        let pass_descriptor = wgpu::RenderPassDescriptor {
            label: Some("forest_building_shadow"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: layer,
                depth_ops: Some(wgpu::Operations {
                    load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        };
        let mut pass = ctx.begin_tracked_render_pass(pass_descriptor);
        pass.set_render_pipeline(pipeline);
        pass.set_bind_group(0, &gpu.cascade_groups[index], &[]);
        record_draws(&mut pass, gpu, &state.frame.shadows[index], false);
    }
    if !plants_drew {
        let uniform = ShadowUniform::new(&shadow.cascades, &shadow.light, shadow.moon);
        queue.write_buffer(&targets.uniform, 0, bytemuck::bytes_of(&uniform));
    }
}

/// The glass pipeline for a view target format: the buildings' vertices,
/// tested against the G-buffer's depth without writing it, blended over the
/// lit scene.
fn glass_pipeline(
    gpu: &mut BuildingGpu,
    cache: &PipelineCache,
    shaders: &ForestShaderHandles,
    format: wgpu::TextureFormat,
) -> CachedRenderPipelineId {
    if let Some(&id) = gpu.glass_pipelines.get(&format) {
        return id;
    }
    let descriptor = RenderPipelineDescriptor {
        label: Some("forest_building_glass_pipeline".into()),
        layout: vec![super::globals_layout()],
        immediate_size: 0,
        vertex: VertexState {
            shader: shaders.building.clone(),
            shader_defs: vec![],
            entry_point: Some("vs_main".into()),
            buffers: gpu.vertex_buffers.clone(),
        },
        primitive: wgpu::PrimitiveState {
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Some(false),
            depth_compare: Some(wgpu::CompareFunction::GreaterEqual),
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: Default::default(),
        fragment: Some(FragmentState {
            shader: shaders.building.clone(),
            shader_defs: vec![],
            entry_point: Some("fs_glass".into()),
            targets: vec![Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::COLOR,
            })],
        }),
        zero_initialize_workgroup_memory: false,
    };
    let id = cache.queue_render_pipeline(descriptor);
    gpu.glass_pipelines.insert(format, id);
    id
}

/// The buildings' window glass over the lit scene, after the composite and
/// before the water, which copies the scene with the glass on it.
pub fn forest_building_glass_pass(
    view: ViewQuery<&ViewTarget>,
    world: &World,
    mut ctx: RenderContext,
) {
    let view = view.into_inner();
    let (Some(state), Some(terrain), Some(cache), Some(shaders)) = (
        world.get_resource::<BuildingNodeState>(),
        world.get_resource::<terrain_node::TerrainNodeState>(),
        world.get_resource::<PipelineCache>(),
        world.get_resource::<ForestShaderHandles>(),
    ) else {
        return;
    };
    let mut guard = state.0.lock().unwrap_or_else(|e| e.into_inner());
    // Through a plain reference, so the GPU state (whose pipeline list may
    // grow) and the frame's draws borrow apart.
    let state = &mut *guard;
    let Some(gpu) = state.gpu.as_mut() else {
        return;
    };
    if state.frame.glass.is_empty() {
        return;
    }
    let format = view.main_texture_format();
    let pipeline_id = glass_pipeline(gpu, cache, shaders, format);
    let Some(pipeline) = cache.get_render_pipeline(pipeline_id) else {
        return;
    };
    let gbuffer = terrain.gbuffer.lock().unwrap_or_else(|e| e.into_inner());
    let Some(gbuffer) = gbuffer.as_ref() else {
        return;
    };
    let pass_descriptor = wgpu::RenderPassDescriptor {
        label: Some("forest_building_glass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: view.main_texture_view(),
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            },
        })],
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
    let mut pass = ctx.begin_tracked_render_pass(pass_descriptor);
    pass.set_render_pipeline(pipeline);
    pass.set_bind_group(0, &gpu.globals_group, &[]);
    record_draws(&mut pass, gpu, &state.frame.glass, false);
}

pub fn register_building_systems(app: &mut bevy::app::SubApp) {
    app.init_resource::<BuildingNodeState>();
    app.init_resource::<ExtractedBuildings>();
    app.init_resource::<BuildingShelters>();
    app.init_resource::<PadUpload>();
    app.add_systems(ExtractSchedule, extract_buildings);
    app.add_systems(
        bevy::render::Render,
        (
            upload_pads
                .in_set(bevy::render::RenderSystems::Prepare)
                .after(super::gpu_textures::prepare_gpu_textures),
            prepare_buildings
                .in_set(bevy::render::RenderSystems::Prepare)
                .after(super::prepare_forest_globals),
        ),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use forest_buildings::collision::turn_by;

    /// Size and member offsets of a struct as naga lays it out.
    fn wgsl_layout(source: &str, name: &str) -> (usize, Vec<(String, usize)>) {
        let module = naga::front::wgsl::parse_str(source).expect("the shader parses");
        for (_, ty) in module.types.iter() {
            if let (Some(name_here), naga::TypeInner::Struct { members, span }) =
                (ty.name.as_deref(), &ty.inner)
                && name_here == name
            {
                let members = members
                    .iter()
                    .map(|member| {
                        (
                            member.name.clone().unwrap_or_default(),
                            member.offset as usize,
                        )
                    })
                    .collect();
                return (*span as usize, members);
            }
        }
        panic!("no struct {name}");
    }

    #[test]
    fn gpu_structs_match_their_wgsl_mirrors() {
        let shader = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../assets/shaders/building.wgsl"
        ));
        let cascade = (
            std::mem::size_of::<CascadeUniform>(),
            vec![
                (
                    "view_projection".to_string(),
                    std::mem::offset_of!(CascadeUniform, view_projection),
                ),
                (
                    "light".to_string(),
                    std::mem::offset_of!(CascadeUniform, light),
                ),
            ],
        );
        let material = (
            std::mem::size_of::<MaterialUniform>(),
            vec![
                (
                    "base_color_factor".to_string(),
                    std::mem::offset_of!(MaterialUniform, base_color_factor),
                ),
                (
                    "params".to_string(),
                    std::mem::offset_of!(MaterialUniform, params),
                ),
            ],
        );
        assert_eq!(wgsl_layout(shader, "ShadowCascade"), cascade);
        assert_eq!(wgsl_layout(shader, "BuildingMaterial"), material);
    }

    #[test]
    fn door_instances_turn_like_the_collision_does() {
        // A door's free edge, turned by the instance the shader draws, lands
        // where the collision puts the leaf.
        let yaw = 0.8f32 + 0.3;
        let instance = instance_of(Vec3::ZERO, yaw);
        let reach = Vec2::new(-1.0, 0.0);
        let shader_turn = Vec2::new(
            reach.x * instance.turn[0] + reach.y * instance.turn[1],
            -reach.x * instance.turn[1] + reach.y * instance.turn[0],
        );
        assert!(shader_turn.distance(turn_by(reach, yaw)) < 1e-6);
    }

    #[test]
    #[allow(non_snake_case)]
    fn nearbyInteriorMeshesCastShadowsWithoutWindowGlass() {
        use forest_buildings::pads::{Pad, PadKind};
        use forest_buildings::placement::Placement;

        let library = forest_buildings::assets::test_support::library();
        let open_doors = HashMap::new();
        let origin = Vec3::new(709.25, 10.0, 565.38);
        let mut glass_primitives = 0;

        for (model_index, model) in library.models.iter().enumerate() {
            for primitive in model.exterior.iter().chain(&model.interior) {
                if library.materials[primitive.material].glass {
                    glass_primitives += 1;
                }
            }
            let centre = Vec2::new(origin.x, origin.z) + model.footprint_centre();
            let pad = Pad {
                centre,
                axis: Vec2::X,
                half: model.footprint_half(),
                height: origin.y,
                skirt: 2.0,
                inset: 0.0,
                kind: PadKind::Building,
            };
            let placement = Placement {
                id: model_index as u64,
                model: model_index as u32,
                origin,
                yaw: 0.0,
                pad,
                town: None,
            };
            let snapshot = BuildingSnapshot {
                generation: 1,
                placements: vec![placement],
            };
            // Keep the model out of the view: its floors must still stop
            // light reaching another surface inside the building.
            let sight = Sight {
                camera: Vec3::new(centre.x, origin.y + 5.62, centre.y),
                planes: [[0.0, 0.0, 0.0, -10000.0]; 4],
                shadows: true,
            };
            let mut shelters = Vec::new();
            let groups = gather_instances(&library, &snapshot, &sight, &open_doors, &mut shelters);
            let (_, frame_draws) = lay_out_draws(&library, &groups);
            assert!(frame_draws.gbuffer.is_empty());

            for (cascade, draws) in frame_draws.shadows.iter().enumerate() {
                assert!(!draws.is_empty(), "{} cascade {cascade}", model.name);
                for primitive in &model.interior {
                    if library.materials[primitive.material].glass {
                        continue;
                    }
                    let drawn = draws.iter().any(|draw| {
                        draw.primitive.first_index == primitive.first_index
                            && draw.primitive.base_vertex == primitive.base_vertex
                    });
                    assert_eq!(
                        drawn,
                        cascade < INTERIOR_SHADOW_CASCADES,
                        "{} cascade {cascade}",
                        model.name
                    );
                }
                assert!(
                    draws
                        .iter()
                        .all(|draw| !library.materials[draw.primitive.material].glass)
                );
            }

            // Interior detail is unnecessary once the building is beyond
            // the distance where rooms could be seen through an opening.
            let radius = Vec2::new(model.radius(), model.maximum.y * 0.5).length();
            let distant_sight = Sight {
                camera: sight.camera + Vec3::X * (INTERIOR_DISTANCE + radius + 1.0),
                ..sight
            };
            let distant_groups = gather_instances(
                &library,
                &snapshot,
                &distant_sight,
                &open_doors,
                &mut shelters,
            );
            for cascade in &distant_groups.shadows {
                assert!(
                    cascade.keys().all(|(_, interior)| !interior),
                    "{}",
                    model.name
                );
            }
        }

        assert!(
            glass_primitives > 0,
            "the regression must exercise window glass"
        );
    }
}
