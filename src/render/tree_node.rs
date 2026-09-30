//! The instanced fir pass: one render-graph node between the terrain G-buffer
//! pass and SSAO, plus the prepare systems that upload the pack and bucket the
//! forest into LOD groups.
//!
//! # Why it sits exactly here
//!
//! Trees write the *same* three G-buffer targets the terrain pass writes, so
//! they have to run after it — otherwise the terrain's `LoadOp::Clear` would
//! erase them — and before SSAO, which reads the depth and normal targets for
//! the whole rest of the frame. Nothing downstream knows trees exist: the
//! composite lights whatever is in the G-buffer, the water pass occludes
//! against the depth in the position target, and the water's raymarched
//! reflection marches the same targets, so a tree standing at the shoreline is
//! reflected for free. This is the entire reason the pass is deferred rather
//! than a forward pass after the composite, which would be painted over by the
//! composite's own background and could never appear in a reflection.
//!
//! All three colour attachments use `LoadOp::Load`, and so does depth. The
//! terrain pass cleared them; this pass must not.
//!
//! # Draw structure
//!
//! One mesh per (size, variant, LOD) — 36 of them — with that mesh's primitives
//! merged into a single vertex buffer and a [`TreeMaterialRange`] per
//! primitive. One instanced draw per (group, material range), so a fir's bark
//! and its foliage cards are separate draws over one buffer, and the whole
//! forest is at most 63 draw calls however many trees there are.
//!
//! # LOD
//!
//! `TreePlacement`s cross from the main world with no LOD attached. This node
//! buckets them itself, from the camera's XZ, and only when the camera has
//! moved far enough for the buckets to change — so a stationary or slowly
//! turning camera uploads nothing, and rotation costs nothing because there is
//! no frustum culling to redo. That last part is deliberate: the LOD-3
//! billboard is 30 vertices, so submitting the several thousand far trees that
//! happen to be behind the camera costs less than the per-frame CPU pass that
//! would compact them, and it keeps rotation free of any upload at all.

use crate::render::glb::GlbVertex;
use crate::render::gpu_textures::write_padded_at;
use crate::render::terrain_node::TerrainNodeState;
use crate::render::{ExtractedForestView, ForestGlobals, ForestShaderHandles};
use crate::trees::{
    TreeAssetData, TreeField, TreeImage, TreeMaterialRange, TreeMeshData, TreePlacement,
    TreeScatter, TreeInstance, TREE_GROUP_COUNT, TREE_LOD_COUNT, TREE_SIZE_COUNT,
    TREE_VARIANT_COUNT,
};
use bevy::mesh::VertexBufferLayout;
use bevy::prelude::*;
use bevy::render::render_graph::{Node, NodeRunError, RenderGraphContext, RenderLabel};
use bevy::render::render_resource::{
    BindGroup, BindGroupLayout, Buffer, CachedRenderPipelineId, FragmentState, PipelineCache,
    RenderPipelineDescriptor, VertexState,
};
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue};
use bevy::render::{ExtractSchedule, MainWorld, Render, RenderSystems};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------------------
// Vertex + instance ABI
// ---------------------------------------------------------------------------

/// `GlbVertex`: position, normal, uv, tangent. Stride 48, asserted in glb.rs.
const TREE_VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; 4] = wgpu::vertex_attr_array![
    0 => Float32x3, // position
    1 => Float32x3, // normal
    2 => Float32x2, // uv
    3 => Float32x4, // tangent, w = bitangent sign
];

/// `TreeInstance`, which starts at location 4 because the vertex attributes
/// above already claimed 0..3. Stride 28, asserted in src/trees/mod.rs.
const TREE_INSTANCE_ATTRIBUTES: [wgpu::VertexAttribute; 6] = wgpu::vertex_attr_array![
    4 => Float32x2, // centre (world XZ)
    5 => Float32,   // ground (world Y under the centre)
    6 => Float32,   // scale
    7 => Float32x4, // rotation (unit quaternion, xyzw)
    8 => Float32,   // variation
    9 => Float32,   // sink
];

/// `TreeMaterialUniforms` in tree-fs.wgsl: 32 bytes.
///
/// The three trailing floats are WGSL's doing, not ours: a struct in the
/// `uniform` address space must be a multiple of 16 bytes, so the five real
/// fields round up from 20. They are declared in the shader and never read.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct TreeMaterialUniforms {
    is_branch: u32,
    roughness: f32,
    alpha_cutoff: f32,
    tint_strength: f32,
    pub(crate) specular_factor: f32,
    _padding: [f32; 3],
}

const _: () = assert!(std::mem::size_of::<TreeMaterialUniforms>() == 32);

// ---------------------------------------------------------------------------
// Tuning
// ---------------------------------------------------------------------------

/// LOD switch distances as multiples of a tree's authored LOD-0 height.
///
/// Choosing them by screen size rather than by a fixed distance is what keeps a
/// small fir from being drawn at full detail out to the range a large one
/// needs. The player camera's vertical field of view is 68 degrees
/// (src/player.rs), which puts a focal length of `0.7414 * frame_height` pixels
/// on a tree of height H at distance d, so a 1080-line frame resolves it at
/// `801*H/d` pixels. These limits therefore put the LOD-0/LOD-1 switch near
/// 89 px, LOD-1/LOD-2 near 50 px and LOD-2/LOD-3 near 33 px. A 900-line window
/// — the size these were last checked at — scales all three by 0.83.
///
/// The billboard has to take over far above the fourteen pixels at which it
/// stops being distinguishable, because a *closed canopy* is what these bands
/// have to fit in a frame. At one tree per 35 m² the band between LOD-2 and
/// LOD-3 holds hundreds of thousands of trees, and the pack's LOD-2 is still a
/// real mesh of about seventeen hundred vertices — the single largest item in
/// the pass. Bringing the handover in to 24 tree-heights puts everything past
/// about 220 m on thirty vertices, which is what makes the density affordable
/// at all; the cost of that choice is that a fir is a billboard by the time it
/// is 33 px tall, where the crossed cards still read as a canopy but no longer
/// as individual branches.
///
/// These are the numbers to reach for first if the tree pass ever shows up in a
/// frame-time budget, and LOD 2 is the one to shorten: its band covers more
/// ground than LOD 0 and LOD 1 together at a cost the billboard does not have.
const TREE_LOD_DISTANCE: [f32; TREE_LOD_COUNT] = [9.0, 16.0, 24.0, f32::INFINITY];

/// How far the camera must move (in XZ, in metres) before the forest is
/// re-bucketed into LOD groups and re-uploaded.
///
/// The buckets are a pure function of camera position, so any movement can in
/// principle change them. Re-bucketing is a pass over every placement in the
/// field plus an instance-buffer write: cheap, but not free, and four metres is
/// well under the distance at which a LOD switch becomes visible. Standing
/// still — the common case while looking around — therefore costs nothing.
const TREE_REBUCKET_DISTANCE: f32 = 4.0;

/// Per-tree tint spread handed to the fragment stage's `treeTint`.
///
/// A few percent of hue and value is all it takes for a stand of one mesh
/// repeated hundreds of times to stop reading as wallpaper; much more and the
/// trees stop looking like one species.
const TREE_TINT_STRENGTH: f32 = 0.35;

/// Instance-buffer capacity is rounded up to this many entries, so a group that
/// grows one tree at a time does not reallocate every frame.
const TREE_INSTANCE_GRANULARITY: u32 = 256;

// ---------------------------------------------------------------------------
// Asset + scatter extraction
// ---------------------------------------------------------------------------

/// The pack, uploaded once, plus the current scatter, replaced whenever the
/// main world's chunk stream republishes.
///
/// [`TreeAssetData`] is *removed* from the main world rather than copied: the
/// meshes are tens of megabytes of vertex data and the textures are decoded
/// RGBA with CPU mip chains, and the main world has no further use for any of
/// it once the GPU owns a copy. Same shape as `extract_terrain_layers`.
#[derive(Resource, Default)]
pub struct ExtractedTrees {
    pub assets: Option<TreeAssetData>,
    pub scatter: Option<Arc<TreeScatter>>,
}

fn extract_trees(mut main_world: ResMut<MainWorld>, mut extracted: ResMut<ExtractedTrees>) {
    let world: &mut World = &mut main_world;
    if extracted.assets.is_none() {
        extracted.assets = world.remove_resource::<TreeAssetData>();
    }
    if let Some(field) = world.get_resource::<TreeField>() {
        // An `Arc` clone, not a copy of the tree list: the render world holds
        // the same allocation the main world replaces with a fresh `Arc` on the
        // next republish, so extraction is O(1) whether the field holds ten
        // trees or ten thousand.
        extracted.scatter = Some(field.scatter().clone());
    }
}

// ---------------------------------------------------------------------------
// GPU state
// ---------------------------------------------------------------------------

/// One (size, variant, LOD) mesh, uploaded.
struct GpuTreeMesh {
    vertices: Buffer,
    indices: Buffer,
    /// `first_index`/`index_count` per primitive, in draw order.
    ranges: Vec<(u32, u32)>,
}

/// One group's instance buffer. Recreated when the group outgrows it.
struct InstanceBuffer {
    buffer: Buffer,
    /// Entries the buffer has room for.
    capacity: u32,
    /// Entries actually written this frame; zero groups are skipped entirely.
    count: u32,
}

impl InstanceBuffer {
    fn new(device: &RenderDevice, label: &str, capacity: u32) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (capacity as u64) * std::mem::size_of::<TreeInstance>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            buffer,
            capacity,
            count: 0,
        }
    }
}

struct TreeResources {
    /// group(0): the frame's shared globals.
    globals: BindGroup,
    /// Draws both faces: for the `doubleSided` bark and branch-card materials.
    pipeline: CachedRenderPipelineId,
    /// Culls back faces: for the single-sided LOD-3 billboard cards, whose
    /// crossed planes each carry a card per axis direction so that exactly one
    /// of the pair faces any given camera. Same shader, same layout, one
    /// `cull_mode` apart — see `TreeMaterialRange::double_sided`.
    pipeline_culled: CachedRenderPipelineId,
    meshes: Vec<GpuTreeMesh>,
    /// Material bind groups, flattened: group `g`'s materials start at
    /// `material_slot[g]` and run for `material_count[g]`.
    materials: Vec<BindGroup>,
    /// One per entry of `materials`: true when that material is single-sided
    /// and must be drawn with `pipeline_culled`.
    culled: Vec<bool>,
    material_slot: [u32; TREE_GROUP_COUNT],
    material_count: [u32; TREE_GROUP_COUNT],
    /// One per group, indexed by `tree_group`.
    instances: Vec<Option<InstanceBuffer>>,
    /// Scatter generation the instance buffers hold.
    uploaded_generation: u64,
    /// Camera XZ the buckets were computed from.
    bucketed_at: [f32; 2],
    /// False until the first upload, so frame one always uploads even though
    /// `bucketed_at` starts at infinity.
    bucketed: bool,
    /// Scratch: the per-group instance lists, rebuilt on each re-bucket.
    /// Kept here so re-bucketing does not allocate every four metres.
    staging: Vec<Vec<TreeInstance>>,
}

#[derive(Resource, Default)]
pub struct TreeNodeState {
    resources: Mutex<Option<TreeResources>>,
}

// ---------------------------------------------------------------------------
// Graph label + node
// ---------------------------------------------------------------------------

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub enum NodeTrees {
    TreePass,
}

pub struct ForestTreeNode;

impl Node for ForestTreeNode {
    fn run<'w>(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let Some(state) = world.get_resource::<TreeNodeState>() else {
            return Ok(());
        };
        // Lock order: terrain state, then tree state. `prepare_trees` takes them
        // in the same order (it needs the globals only, but `resize_gbuffer`
        // holds the terrain lock from the same schedule), so this cannot
        // deadlock.
        let terrain_state = world.get_resource::<TerrainNodeState>();
        let terrain_guard = terrain_state.map(|state| {
            state
                .gbuffer
                .lock()
                .unwrap_or_else(|error| error.into_inner())
        });
        let guard = state
            .resources
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let (Some(gbuffer), Some(resources)) = (
            terrain_guard.as_ref().and_then(|guard| guard.as_ref()),
            guard.as_ref(),
        ) else {
            return Ok(());
        };
        let Some(pipeline_cache) = world.get_resource::<PipelineCache>() else {
            return Ok(());
        };
        let Some(pipeline) = pipeline_cache.get_render_pipeline(resources.pipeline) else {
            // Still compiling on the first frames; drawing nothing for a frame
            // or two is correct, panicking is not.
            return Ok(());
        };
        let Some(pipeline_culled) = pipeline_cache.get_render_pipeline(resources.pipeline_culled)
        else {
            return Ok(());
        };

        // Load, never clear: the terrain pass has already written the G-buffer
        // and this pass is adding to it. Clearing any of these attachments
        // would erase the world the trees are standing in.
        let color_attachments = [
            Some(wgpu::RenderPassColorAttachment {
                view: &gbuffer.position_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            }),
            Some(wgpu::RenderPassColorAttachment {
                view: &gbuffer.normal_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            }),
            Some(wgpu::RenderPassColorAttachment {
                view: &gbuffer.albedo_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            }),
        ];
        let pass_descriptor = wgpu::RenderPassDescriptor {
            label: Some("forest_tree_gbuffer"),
            color_attachments: &color_attachments,
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &gbuffer.depth_view,
                depth_ops: Some(wgpu::Operations {
                    // Reverse-Z, and loaded: the terrain wrote real depths and
                    // the trees are tested against them. A clear here would
                    // erase that and, at 0.0, put every tree behind the far
                    // plane.
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        };
        let mut pass = render_context.begin_tracked_render_pass(pass_descriptor);
        // The C++ main loop sets rlViewport over the SSAO-sized target before
        // the 3D block; match the terrain pass exactly so both write the same
        // texels.
        pass.set_viewport(
            0.0,
            0.0,
            gbuffer.width as f32,
            gbuffer.height as f32,
            0.0,
            1.0,
        );
        pass.set_bind_group(0, &resources.globals, &[]);

        let mut draws = 0usize;
        // Which of the two pipelines is bound, so the loop does not re-set it
        // for a material that matches the previous draw's. `None` until the
        // first draw, and the first draw is always a culled or an unculled
        // group rather than neither.
        let mut bound_culled: Option<bool> = None;
        for group in 0..TREE_GROUP_COUNT {
            let Some(instance) = resources.instances[group].as_ref() else {
                continue;
            };
            if instance.count == 0 {
                continue;
            }
            let mesh = &resources.meshes[group];
            let first = resources.material_slot[group] as usize;
            let material_count = resources.material_count[group] as usize;
            pass.set_vertex_buffer(0, mesh.vertices.slice(..));
            pass.set_vertex_buffer(1, instance.buffer.slice(..));
            pass.set_index_buffer(mesh.indices.slice(..), 0, wgpu::IndexFormat::Uint32);
            // One draw per primitive, each with its own material — a fir has
            // two (bark, foliage cards) and a billboard has one. The pipeline
            // is re-set here rather than once outside the loop because
            // sidedness is a property of the material, and this pack mixes it:
            // every LOD-3 billboard is single-sided and everything nearer is
            // not. Most groups hold a single material, so in practice this is
            // one `set_render_pipeline` per group.
            for material in 0..material_count {
                let slot = first + material;
                let culled = resources.culled[slot];
                if bound_culled != Some(culled) {
                    pass.set_render_pipeline(if culled { pipeline_culled } else { pipeline });
                    bound_culled = Some(culled);
                }
                pass.set_bind_group(1, &resources.materials[slot], &[]);
                let (first_index, index_count) = mesh.ranges[material];
                pass.draw_indexed(first_index..first_index + index_count, 0, 0..instance.count);
                draws += 1;
            }
        }
        let _ = draws;

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Upload helpers
// ---------------------------------------------------------------------------

fn upload_mesh(device: &RenderDevice, label: &str, mesh: &TreeMeshData) -> GpuTreeMesh {
    let vertices = device.create_buffer_with_data(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(&mesh.vertices),
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
    });
    let indices = device.create_buffer_with_data(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(&mesh.indices),
        usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
    });
    GpuTreeMesh {
        vertices,
        indices,
        ranges: mesh
            .ranges
            .iter()
            .map(|range| (range.first_index, range.index_count))
            .collect(),
    }
}

/// Upload a decoded image and its CPU-built mip chain.
///
/// The chain is built on the CPU (in `src/trees/mod.rs`) rather than with a GPU
/// blit because the base-colour textures are sRGB: a GPU mip pass averages with
/// the transfer function applied, which darkens every level, and foliage is
/// exactly the case where that shows. The normal maps would be safe either way,
/// but one story for mip generation in this renderer is worth more than the
/// small saving from splitting them.
fn upload_image(
    device: &RenderDevice,
    queue: &RenderQueue,
    label: &str,
    image: &TreeImage,
) -> wgpu::TextureView {
    let format = if image.srgb {
        wgpu::TextureFormat::Rgba8UnormSrgb
    } else {
        wgpu::TextureFormat::Rgba8Unorm
    };
    let texture = device.wgpu_device().create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: image.width,
            height: image.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: image.levels.len() as u32,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let mut level_width = image.width;
    let mut level_height = image.height;
    for (level, data) in image.levels.iter().enumerate() {
        // `write_padded_at` pads each row to the 256-byte alignment
        // `write_texture` requires. It matters here: the chain runs down to
        // 1x1, and a 4-byte row is nowhere near aligned.
        write_padded_at(
            queue,
            &texture,
            data,
            level_width,
            level_height,
            4,
            [0, 0],
            level as u32,
        );
        level_width = (level_width / 2).max(1);
        level_height = (level_height / 2).max(1);
    }
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

/// A 1x1 texture, bound into whichever of the two material slots a draw does
/// not use.
///
/// A wgpu bind group must supply every binding its layout declares, including
/// the ones the shader's uniform branch will never sample. Pointing the unused
/// albedo slot at white makes an accidental read show up as white foliage
/// rather than black — and the unused normal slot at (128, 128, 255) decodes to
/// a straight-out-of-the-surface normal.
fn solid_texture(
    device: &RenderDevice,
    queue: &RenderQueue,
    label: &str,
    rgba: [u8; 4],
    srgb: bool,
) -> wgpu::TextureView {
    let format = if srgb {
        wgpu::TextureFormat::Rgba8UnormSrgb
    } else {
        wgpu::TextureFormat::Rgba8Unorm
    };
    let texture = device.wgpu_device().create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    write_padded_at(queue, &texture, &rgba, 1, 1, 4, [0, 0], 0);
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

/// group(1): the two texture pairs, the sampler and this draw's uniforms.
fn material_layout(device: &RenderDevice) -> BindGroupLayout {
    let texture = |binding: u32| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    };
    device.create_bind_group_layout(
        "forest_tree_material_layout",
        &[
            texture(0), // bark albedo
            texture(1), // bark normal
            texture(2), // branch albedo
            texture(3), // branch normal
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(
                        std::mem::size_of::<TreeMaterialUniforms>() as u64,
                    ),
                },
                count: None,
            },
        ],
    )
}

// ---------------------------------------------------------------------------
// Prepare
// ---------------------------------------------------------------------------

/// Everything that only has to happen once: upload the 36 meshes and their
/// shared textures, create the material bind groups, and queue the pipeline.
///
/// The globals guard is a hard requirement — the bind group wraps the shared
/// globals buffer — so this retries safely across the first frames while
/// `prepare_forest_globals` comes up.
fn prepare_trees(
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    pipeline_cache: Res<PipelineCache>,
    shaders: Res<ForestShaderHandles>,
    globals: Res<ForestGlobals>,
    extracted: Res<ExtractedTrees>,
    state: Res<TreeNodeState>,
) {
    let mut resources = state
        .resources
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    if resources.is_some() {
        return;
    }
    let (Some(globals_buffer), Some(assets)) = (globals.buffer.as_ref(), extracted.assets.as_ref())
    else {
        return;
    };
    let device = &*device;
    let queue = &*queue;

    let globals_layout = crate::render::globals_layout(device);
    let globals_entries = crate::render::globals_bind_group_entries(globals_buffer);
    let globals = device.create_bind_group(
        "forest_tree_globals_bind_group",
        &globals_layout,
        &globals_entries,
    );

    let layout = material_layout(device);

    // Two pipelines out of one descriptor, because the pack mixes sidedness
    // across LODs and the cull mode is the only thing that differs between
    // them. `TreeMaterialRange::double_sided` has the long version; the short
    // one is that a fir's branch cards are flat planes you are meant to see
    // from behind, while a billboard's cross carries a card per axis direction
    // so that only one of each pair ever faces the camera at all.
    //
    // The fragment stage still flips the shading normal on back faces. With
    // both pipelines in place that only ever fires for the double-sided LODs,
    // which is exactly where it is wanted.
    let build_tree_pipeline = |label: &'static str, cull_mode: Option<wgpu::Face>| {
        pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
            label: Some(label.into()),
            layout: vec![globals_layout.clone(), layout.clone()],
            push_constant_ranges: vec![],
            vertex: VertexState {
                shader: shaders.tree_vs.clone(),
                shader_defs: vec![],
                entry_point: Some("vs_main".into()),
                buffers: vec![
                    VertexBufferLayout {
                        array_stride: std::mem::size_of::<GlbVertex>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: TREE_VERTEX_ATTRIBUTES.to_vec(),
                    },
                    VertexBufferLayout {
                        array_stride: std::mem::size_of::<TreeInstance>() as u64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: TREE_INSTANCE_ATTRIBUTES.to_vec(),
                    },
                ],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                // Matches the terrain pass, which is what GL's glFrontFace(GL_CCW)
                // maps to (see the long note in terrain_node.rs). The winding
                // decides `@builtin(front_facing)` and therefore which foliage
                // normals get flipped, and it is what `cull_mode` tests.
                front_face: wgpu::FrontFace::Ccw,
                cull_mode,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                // Depth writes on, like the terrain. The alpha cutout discards
                // before the write, so the depth laid down is the tree's own
                // surface, and the water pass's texel-exact depth reconstruction
                // sees a crisp edge rather than a blend.
                depth_write_enabled: true,
                // Reverse-Z, same as every other G-buffer pass.
                depth_compare: wgpu::CompareFunction::GreaterEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(FragmentState {
                shader: shaders.tree_fs.clone(),
                shader_defs: vec![],
                entry_point: Some("fs_main".into()),
                targets: vec![
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba32Float,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
            }),
            zero_initialize_workgroup_memory: false,
        })
    };
    let pipeline = build_tree_pipeline("forest_tree_pipeline", None);
    let pipeline_culled =
        build_tree_pipeline("forest_tree_pipeline_culled", Some(wgpu::Face::Back));

    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("forest_tree_sampler"),
        // Repeat, not clamp: the bark texture tiles around the trunk and the
        // foliage cards' UVs run past 1 by design.
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        address_mode_w: wgpu::AddressMode::Repeat,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Linear,
        anisotropy_clamp: 4,
        ..Default::default()
    });
    let white = solid_texture(device, queue, "forest_tree_white", [255, 255, 255, 255], true);
    let flat_normal = solid_texture(
        device,
        queue,
        "forest_tree_flat_normal",
        [128, 128, 255, 255],
        false,
    );

    // Upload each distinct image once. The six shared textures are `Arc`-shared
    // across all 36 files by the loader's cache, so keying on the pointer
    // collapses them to six uploads and leaves the eighteen embedded LOD-3
    // billboard textures as their own.
    let mut images = ImageUploader {
        device,
        queue,
        uploaded: HashMap::new(),
    };

    let mut meshes = Vec::with_capacity(TREE_GROUP_COUNT);
    let mut materials = Vec::new();
    let mut culled = Vec::new();
    let mut material_slot = [0u32; TREE_GROUP_COUNT];
    let mut material_count = [0u32; TREE_GROUP_COUNT];
    for (group, mesh) in assets.meshes.iter().enumerate() {
        material_slot[group] = materials.len() as u32;
        material_count[group] = mesh.ranges.len() as u32;
        for material in &mesh.ranges {
            culled.push(!material.double_sided);
            materials.push(build_material_bind_group(
                device,
                &layout,
                &sampler,
                &mut images,
                material,
                &white,
                &flat_normal,
                group,
            ));
        }
        meshes.push(upload_mesh(
            device,
            &format!("forest_tree_mesh_{group}"),
            mesh,
        ));
    }

    let vertices: usize = assets.meshes.iter().map(|mesh| mesh.vertices.len()).sum();
    let triangles: usize = assets.meshes.iter().map(|mesh| mesh.indices.len() / 3).sum();
    log::info!(
        "TREES: uploaded {TREE_GROUP_COUNT} meshes ({vertices} vertices, {triangles} triangles), \
         {} textures, {} material bind groups",
        images.uploaded.len(),
        materials.len()
    );

    *resources = Some(TreeResources {
        globals,
        pipeline,
        pipeline_culled,
        meshes,
        materials,
        culled,
        material_slot,
        material_count,
        instances: (0..TREE_GROUP_COUNT).map(|_| None).collect(),
        uploaded_generation: u64::MAX,
        bucketed_at: [f32::INFINITY, f32::INFINITY],
        bucketed: false,
        staging: (0..TREE_GROUP_COUNT).map(|_| Vec::new()).collect(),
    });
}

/// Uploads each distinct image once and hands back the shared view.
///
/// Distinctness is `Arc` identity, not content: the loader already deduped the
/// six textures the 36 files share by caching them under their file path, so
/// two ranges referring to the same bark map hold the same `Arc` and the same
/// pointer.
struct ImageUploader<'a> {
    device: &'a RenderDevice,
    queue: &'a RenderQueue,
    uploaded: HashMap<usize, wgpu::TextureView>,
}

impl ImageUploader<'_> {
    fn get(&mut self, image: &Arc<TreeImage>, label: &str) -> wgpu::TextureView {
        let key = Arc::as_ptr(image) as usize;
        if let Some(view) = self.uploaded.get(&key) {
            return view.clone();
        }
        let view = upload_image(self.device, self.queue, label, image);
        self.uploaded.insert(key, view.clone());
        view
    }

    /// Upload a material's map if it has one, or hand back the dummy.
    fn or_else(
        &mut self,
        image: Option<&Arc<TreeImage>>,
        label: &str,
        dummy: &wgpu::TextureView,
    ) -> wgpu::TextureView {
        match image {
            Some(image) => self.get(image, label),
            None => dummy.clone(),
        }
    }
}

/// One group(1) bind group: this material's texture pair in the slots the
/// shader's uniform branch will read, and the dummy pair in the ones it will
/// not.
///
/// The shader branches on `is_branch`, and `is_branch` is the pack's own
/// `alphaMode` (`BLEND` for the foliage cards and the LOD-3 billboards,
/// `OPAQUE` for bark — see the loader). Slots 0/1 are the bark pair and 2/3 the
/// branch pair purely because that is the order tree-fs.wgsl declares them.
fn build_material_bind_group(
    device: &RenderDevice,
    layout: &BindGroupLayout,
    sampler: &wgpu::Sampler,
    images: &mut ImageUploader,
    material: &TreeMaterialRange,
    white: &wgpu::TextureView,
    flat_normal: &wgpu::TextureView,
    group: usize,
) -> BindGroup {
    let label = |kind: &str| format!("forest_tree_material_{group}_{kind}");
    // Views are cloned rather than borrowed: a bind group entry wants an owned
    // handle for its lifetime, and `TextureView` is a cheap `Arc` bump. This
    // runs 63 times at startup.
    let (bark_albedo, bark_normal, branch_albedo, branch_normal) = if material.is_branch {
        (
            white.clone(),
            flat_normal.clone(),
            images.or_else(material.base_colour.as_ref(), &label("branch_albedo"), white),
            images.or_else(material.normal.as_ref(), &label("branch_normal"), flat_normal),
        )
    } else {
        (
            images.or_else(material.base_colour.as_ref(), &label("bark_albedo"), white),
            images.or_else(material.normal.as_ref(), &label("bark_normal"), flat_normal),
            white.clone(),
            flat_normal.clone(),
        )
    };
    let uniform = device.create_buffer_with_data(&wgpu::util::BufferInitDescriptor {
        label: Some(&label("uniforms")),
        contents: bytemuck::bytes_of(&TreeMaterialUniforms {
            is_branch: u32::from(material.is_branch),
            roughness: material.roughness,
            alpha_cutoff: material.alpha_cutoff,
            tint_strength: TREE_TINT_STRENGTH,
            specular_factor: material.specular_factor,
            _padding: [0.0; 3],
        }),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    device.create_bind_group(
        label("bind_group").as_str(),
        layout,
        &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&bark_albedo),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&bark_normal),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&branch_albedo),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(&branch_normal),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: uniform.as_entire_binding(),
            },
        ],
    )
}

/// Re-bucket the forest into LOD groups when the camera has moved, uploading
/// one instance buffer per group.
///
/// Runs after [`prepare_trees`]. Does nothing until that has built the
/// resources and the main world has published a scatter.
fn prepare_tree_instances(
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    view: Res<ExtractedForestView>,
    extracted: Res<ExtractedTrees>,
    state: Res<TreeNodeState>,
) {
    let mut guard = state
        .resources
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let Some(resources) = guard.as_mut() else {
        return;
    };
    let Some(scatter) = extracted.scatter.as_ref() else {
        return;
    };

    let camera = [view.player_position[0], view.player_position[2]];
    let moved = (camera[0] - resources.bucketed_at[0]).hypot(camera[1] - resources.bucketed_at[1]);
    let fresh = resources.uploaded_generation != scatter.generation;
    if resources.bucketed && !fresh && moved < TREE_REBUCKET_DISTANCE {
        return;
    }

    for group in resources.staging.iter_mut() {
        group.clear();
    }
    for placement in &scatter.placements {
        let group = tree_group_of(scatter, placement, camera);
        resources.staging[group].push(instance_of(placement));
    }

    let device = &*device;
    let queue = &*queue;
    let mut total = 0usize;
    for group in 0..TREE_GROUP_COUNT {
        let count = resources.staging[group].len() as u32;
        total += count as usize;
        if count == 0 {
            if let Some(slot) = resources.instances[group].as_mut() {
                slot.count = 0;
            }
            continue;
        }
        let needed = count.next_multiple_of(TREE_INSTANCE_GRANULARITY);
        let slot = match resources.instances[group].as_mut() {
            Some(slot) if slot.capacity >= needed => slot,
            _ => {
                resources.instances[group] = Some(InstanceBuffer::new(
                    device,
                    &format!("forest_tree_instances_{group}"),
                    needed,
                ));
                resources.instances[group].as_mut().unwrap()
            }
        };
        queue.write_buffer(
            &slot.buffer,
            0,
            bytemuck::cast_slice(&resources.staging[group]),
        );
        slot.count = count;
    }

    // Per-LOD instance totals, which is the number the vertex budget is made
    // of: LOD 0-2 are thousands of vertices each and LOD 3 is thirty, so the
    // split between the first three bands and the last is nearly the whole
    // cost of the pass. Debug level because it changes every time the player
    // crosses a chunk boundary.
    let mut per_lod = [0usize; TREE_LOD_COUNT];
    for group in 0..TREE_GROUP_COUNT {
        per_lod[group % TREE_LOD_COUNT] += resources.instances[group]
            .as_ref()
            .map_or(0, |slot| slot.count as usize);
    }
    log::debug!("TREES: {total} instances across {TREE_GROUP_COUNT} LOD groups, per LOD {per_lod:?}");

    resources.uploaded_generation = scatter.generation;
    resources.bucketed_at = camera;
    resources.bucketed = true;
}

/// Which (size, variant, LOD) group one tree draws in.
///
/// The band is scaled by the tree's own LOD-0 height times its instance scale,
/// so a small fir becomes a billboard at the same apparent size a large one
/// does rather than at the same distance.
fn tree_group_of(scatter: &TreeScatter, placement: &TreePlacement, camera: [f32; 2]) -> usize {
    let size = placement.size as usize % TREE_SIZE_COUNT;
    let variant = placement.variant as usize % TREE_VARIANT_COUNT;
    let height = scatter.heights[size][variant] * placement.scale;
    let dx = placement.x - camera[0];
    let dz = placement.z - camera[1];
    let distance = (dx * dx + dz * dz).sqrt();
    let mut lod = TREE_LOD_COUNT - 1;
    for (index, limit) in TREE_LOD_DISTANCE.iter().enumerate() {
        if distance < height * limit {
            lod = index;
            break;
        }
    }
    crate::trees::tree_group(size, variant, lod)
}

fn instance_of(placement: &TreePlacement) -> TreeInstance {
    TreeInstance {
        centre: [placement.x, placement.z],
        ground: placement.ground,
        scale: placement.scale,
        rotation: placement.rotation,
        variation: placement.variation,
        sink: placement.sink,
    }
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Render-app registration: the extraction, then the two prepare systems in
/// order, both after the shared globals buffer exists.
pub fn register_tree_systems(render_app: &mut bevy::app::SubApp) {
    render_app.init_resource::<TreeNodeState>();
    render_app.init_resource::<ExtractedTrees>();
    render_app.add_systems(ExtractSchedule, extract_trees);
    render_app.add_systems(
        Render,
        (prepare_trees, prepare_tree_instances)
            .chain()
            .in_set(RenderSystems::Prepare)
            .after(crate::render::prepare_forest_globals),
    );
}
