//! Terrain G-buffer pass: ports `CreateClipGrid`, `CreateGBuffer`,
//! `DrawClipmap`, `DrawWorldGeometry` and the main loop's G-buffer section
//! (`rlEnableFramebuffer(ssao.gbuffer.fbo)` .. `rlDisableFramebuffer()`).
//!
//! One tracked render pass fills the three colour targets (view-space
//! position, encoded normal + roughness, albedo + masks) and the shared depth
//! buffer; `post_nodes` locks [`TerrainNodeState::gbuffer`] inside its own
//! `Node::run` to sample those same views for SSAO, blur and composite.

use crate::constants::*;
use crate::render::gpu_textures::{GpuWorldTextures, GpuWorldTexturesOption};
use crate::render::{
    BasicStageUniforms, ExtractedForestView, ForestGlobals, ForestShaderHandles,
    TerrainStageUniforms,
};
use bevy::asset::Handle;
use bevy::mesh::VertexBufferLayout;
use bevy::prelude::*;
use bevy::render::render_graph::{Node, NodeRunError, RenderGraphContext, RenderLabel};
use bevy::render::render_resource::{
    BindGroup, BindGroupLayout, Buffer, CachedRenderPipelineId, FragmentState, PipelineCache,
    RenderPipelineDescriptor, VertexState,
};
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue};
use bevy::shader::Shader;
use std::sync::Mutex;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// `DrawWorldGeometry`: `MatrixTranslate(oceanOrigin.x, kSeaLevel + 0.12,
/// oceanOrigin.y)`. The plane floats just above the waterline so the near-shore
/// terrain (lifted by the waterline clearance) stays clear of it instead of
/// fighting for the same depth.
const OCEAN_SURFACE_OFFSET: f32 = 0.12;
const OCEAN_SURFACE_HEIGHT: f32 = SEA_LEVEL + OCEAN_SURFACE_OFFSET;
/// `GenMeshPlane(16000.0f, 16000.0f, 1, 1)`.
const OCEAN_PLANE_SIZE: f32 = 16000.0;
/// `DrawWorldGeometry`: `oceanOrigin = floor(position / 256) * 256`, so the
/// patch follows the player in 256 m steps.
const OCEAN_ORIGIN_SNAP: f32 = 256.0;
/// Deep water is an extremely dark blue-green in linear space; the perceived
/// brightness comes from the sun glint the GGX lobe gives the smooth surface,
/// not from diffuse colour.
const OCEAN_COLOR: [f32; 4] = [0.012, 0.062, 0.088, 1.0];
const OCEAN_ROUGHNESS: f32 = 0.18;

/// The G-buffer clear: `rlClearColor(112, 173, 214, 0)` then
/// `rlClearScreenBuffers()`.
///
/// PORT NOTE: `rlClearColor` takes unsigned bytes and divides them by 255
/// before `glClearColor`, so the C++ really clears to 112/255, 173/255,
/// 214/255 with alpha 0. Only the alpha is load-bearing — composite.wgsl and
/// ssao.wgsl treat `alpha < 0.5` as sky and never read a cleared texel's RGB —
/// but matching the exact value keeps the port honest.
const GBUFFER_CLEAR_COLOR: wgpu::Color = wgpu::Color {
    r: 112.0 / 255.0,
    g: 173.0 / 255.0,
    b: 214.0 / 255.0,
    a: 0.0,
};

/// terrain-vs reads a single `vec3` position (planar grid coordinates).
const TERRAIN_VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; 1] = [wgpu::VertexAttribute {
    format: wgpu::VertexFormat::Float32x3,
    offset: 0,
    shader_location: 0,
}];
const TERRAIN_VERTEX_STRIDE: u64 = 12;

/// basic-gbuffer's raylib-style attributes, shader locations 0-3.
const BASIC_VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; 4] = [
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x3,
        offset: 0,
        shader_location: 0,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x2,
        offset: 12,
        shader_location: 1,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x3,
        offset: 20,
        shader_location: 2,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x4,
        offset: 32,
        shader_location: 3,
    },
];
const BASIC_VERTEX_STRIDE: u64 = 48;

// ---------------------------------------------------------------------------
// Public view/state types
// ---------------------------------------------------------------------------

/// The colour attachments + depth the terrain pass renders into. `post_nodes`
/// reads these views.
pub struct GbufferTargets {
    pub position_view: wgpu::TextureView, // Rgba32Float
    pub normal_view: wgpu::TextureView,   // Rgba16Float
    pub albedo_view: wgpu::TextureView,   // Rgba8Unorm
    pub depth_view: wgpu::TextureView,    // Depth32Float
    pub width: u32,
    pub height: u32,
}

/// Render-world state for the terrain pass. Interior mutability is required
/// because `Node::run` only ever sees `&World`.
#[derive(Resource)]
pub struct TerrainNodeState {
    /// The G-buffer colour targets + depth, recreated on window resize.
    /// `post_nodes` locks this inside its own `Node::run`.
    pub gbuffer: Mutex<Option<GbufferTargets>>,
    /// Pipelines, meshes, bind groups and uniform slots, built lazily by
    /// [`prepare_terrain`] once the globals buffer and the world textures
    /// exist.
    resources: Mutex<Option<TerrainResources>>,
}

impl Default for TerrainNodeState {
    fn default() -> Self {
        Self {
            gbuffer: Mutex::new(None),
            resources: Mutex::new(None),
        }
    }
}

/// One stage-uniform slot: the buffer the node rewrites every frame plus the
/// group(2) bind group wrapping it.
struct StageUniform {
    buffer: Buffer,
    bind_group: BindGroup,
}

impl StageUniform {
    fn new(device: &RenderDevice, layout: &BindGroupLayout, label: &str, size: u64) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(
            label,
            layout,
            &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        );
        Self { buffer, bind_group }
    }
}

/// An uploaded mesh: interleaved vertex data plus its u32 index buffer.
struct GpuMesh {
    vertices: Buffer,
    indices: Buffer,
    index_count: u32,
}

/// Everything [`prepare_terrain`] builds once the shared globals buffer and
/// the world textures exist. Every field is owned, so the node can hold the
/// state lock for the whole pass.
struct TerrainResources {
    /// group(0): the frame's shared globals.
    globals: BindGroup,
    /// group(1): the world textures.
    terrain_textures: BindGroup,
    /// group(1) for the ocean pipeline, whose layout has a deliberately empty
    /// slot there: `basic-gbuffer` samples no textures, but wgpu still
    /// validates that every group in the pipeline layout is bound at draw
    /// time, so the ocean draw binds this instead of inheriting the terrain
    /// textures group.
    empty_textures: BindGroup,
    terrain_pipeline: CachedRenderPipelineId,
    ocean_pipeline: CachedRenderPipelineId,
    center_mesh: GpuMesh,
    ring_mesh: GpuMesh,
    ocean_mesh: GpuMesh,
    /// One stage block per clipmap level, matching `DrawClipmap`'s per-level
    /// uniforms.
    levels: [StageUniform; CLIP_LEVELS],
    /// The ocean patch's `BasicStageUniforms` block.
    ocean: StageUniform,
}

// ---------------------------------------------------------------------------
// Graph label + node
// ---------------------------------------------------------------------------

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub enum NodeTerrain {
    TerrainPass,
}

pub struct ForestTerrainNode;

impl Node for ForestTerrainNode {
    fn run<'w>(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let Some(state) = world.get_resource::<TerrainNodeState>() else {
            return Ok(());
        };
        // Both slots are filled by the prepare systems above; an empty slot
        // just means "nothing to draw yet" (first frame, or before the world
        // textures exist). Never panic on it.
        //
        // Lock order: gbuffer first, then resources. No other system ever
        // holds both, so this cannot deadlock.
        let gbuffer_guard = state
            .gbuffer
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let resources_guard = state
            .resources
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let (Some(gbuffer), Some(resources)) = (gbuffer_guard.as_ref(), resources_guard.as_ref())
        else {
            return Ok(());
        };
        let Some(view) = world.get_resource::<ExtractedForestView>() else {
            return Ok(());
        };
        let Some(globals) = world.get_resource::<ForestGlobals>() else {
            return Ok(());
        };
        let Some(queue) = world.get_resource::<RenderQueue>() else {
            return Ok(());
        };
        let Some(pipeline_cache) = world.get_resource::<PipelineCache>() else {
            return Ok(());
        };

        // DrawClipmap: uErosionVisibilityCenter is the player's XZ and every
        // level shares it (SetTerrainSharedUniforms runs once per frame).
        let visibility_center = [view.player_position[0], view.player_position[2]];
        for (level, stage) in resources.levels.iter().enumerate() {
            let uniforms = TerrainStageUniforms::build(
                // The C++ hands MatrixIdentity() to DrawMesh for every level:
                // a level's world placement comes from uClipOrigin, not from
                // the model matrix.
                bevy::math::Mat4::IDENTITY.to_cols_array(),
                level as u32,
                view.player_position,
                view.lookup_minimum,
                visibility_center,
            );
            queue.write_buffer(&stage.buffer, 0, bytemuck::bytes_of(&uniforms));
        }

        if view.draw_ocean {
            let ocean_origin = [
                (view.player_position[0] / OCEAN_ORIGIN_SNAP).floor() * OCEAN_ORIGIN_SNAP,
                (view.player_position[2] / OCEAN_ORIGIN_SNAP).floor() * OCEAN_ORIGIN_SNAP,
            ];
            // MatrixTranslate(oceanOrigin.x, kSeaLevel + 0.12, oceanOrigin.y).
            let mat_model =
                crate::matrices::translation(ocean_origin[0], OCEAN_SURFACE_HEIGHT, ocean_origin[1]);
            // basic-gbuffer builds its normal matrix from
            // inverse(matView*matModel); the view half is the frame's shared
            // globals.view, written by prepare_forest_globals from the same
            // camera state DrawWorldGeometry used.
            let model_view = crate::matrices::mul_m4(&globals.globals.view, &mat_model);
            let uniforms = BasicStageUniforms {
                mat_model,
                inverse_model_view: crate::matrices::invert_affine(&model_view),
                color: OCEAN_COLOR,
                roughness: OCEAN_ROUGHNESS,
                // Trailing padding; the WGSL struct ends at roughness.
                _end_pad: [0.0; 3],
            };
            queue.write_buffer(&resources.ocean.buffer, 0, bytemuck::bytes_of(&uniforms));
        }

        // One pass, three colour targets + depth, cleared exactly like the
        // C++: rlClearColor(112, 173, 214, 0), rlClearScreenBuffers() (which
        // also clears depth to GL's default 1.0), then rlViewport over the
        // SSAO-sized target.
        let color_attachments = [
            Some(wgpu::RenderPassColorAttachment {
                view: &gbuffer.position_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(GBUFFER_CLEAR_COLOR),
                    store: wgpu::StoreOp::Store,
                },
            }),
            Some(wgpu::RenderPassColorAttachment {
                view: &gbuffer.normal_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(GBUFFER_CLEAR_COLOR),
                    store: wgpu::StoreOp::Store,
                },
            }),
            Some(wgpu::RenderPassColorAttachment {
                view: &gbuffer.albedo_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(GBUFFER_CLEAR_COLOR),
                    store: wgpu::StoreOp::Store,
                },
            }),
        ];
        let pass_descriptor = wgpu::RenderPassDescriptor {
            label: Some("forest_terrain_gbuffer"),
            color_attachments: &color_attachments,
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &gbuffer.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        };
        let mut pass = render_context.begin_tracked_render_pass(pass_descriptor);
        pass.set_viewport(
            0.0,
            0.0,
            gbuffer.width as f32,
            gbuffer.height as f32,
            0.0,
            1.0,
        );

        // Terrain first, then the ocean, both sharing this pass's depth
        // buffer. The C++ walks the levels in order and draws
        // `level == 0 ? center : ring` for each of them.
        if let Some(pipeline) = pipeline_cache.get_render_pipeline(resources.terrain_pipeline) {
            pass.set_render_pipeline(pipeline);
            pass.set_bind_group(0, &resources.globals, &[]);
            pass.set_bind_group(1, &resources.terrain_textures, &[]);
            for (level, stage) in resources.levels.iter().enumerate() {
                let mesh = if level == 0 {
                    &resources.center_mesh
                } else {
                    &resources.ring_mesh
                };
                pass.set_bind_group(2, &stage.bind_group, &[]);
                pass.set_vertex_buffer(0, mesh.vertices.slice(..));
                pass.set_index_buffer(mesh.indices.slice(..), 0, wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..mesh.index_count, 0, 0..1);
            }
        }

        if view.draw_ocean {
            if let Some(pipeline) = pipeline_cache.get_render_pipeline(resources.ocean_pipeline) {
                pass.set_render_pipeline(pipeline);
                pass.set_bind_group(0, &resources.globals, &[]);
                // Group 1 is unused by basic-gbuffer; its layout slot exists
                // only so the stage block stays at index 2.
                pass.set_bind_group(1, &resources.empty_textures, &[]);
                pass.set_bind_group(2, &resources.ocean.bind_group, &[]);
                pass.set_vertex_buffer(0, resources.ocean_mesh.vertices.slice(..));
                pass.set_index_buffer(
                    resources.ocean_mesh.indices.slice(..),
                    0,
                    wgpu::IndexFormat::Uint32,
                );
                pass.draw_indexed(0..resources.ocean_mesh.index_count, 0, 0..1);
            }
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Mesh construction (CreateClipGrid / GenMeshPlane)
// ---------------------------------------------------------------------------

/// Uploads one interleaved vertex buffer plus its u32 index buffer.
fn upload_mesh<V: bytemuck::Pod, I: bytemuck::Pod>(
    device: &RenderDevice,
    label: &str,
    vertices: &[V],
    indices: &[I],
) -> GpuMesh {
    let index_count = indices.len() as u32;
    let vertex_buffer = device.create_buffer_with_data(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(vertices),
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
    });
    let index_buffer = device.create_buffer_with_data(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(indices),
        usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
    });
    GpuMesh {
        vertices: vertex_buffer,
        indices: index_buffer,
        index_count,
    }
}

/// raylib's default 3D vertex layout, interleaved: position (3 f32),
/// texcoord (2 f32), normal (3 f32), colour (4 f32). raylib keeps these in
/// four separate VBOs; the port interleaves them into one buffer, which feeds
/// the vertex stage identical per-vertex values.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BasicVertex {
    position: [f32; 3],
    texcoord: [f32; 2],
    normal: [f32; 3],
    color: [f32; 4],
}
const _: () = assert!(std::mem::size_of::<BasicVertex>() == BASIC_VERTEX_STRIDE as usize);

/// `CreateClipGrid`: the clipmap grid in planar grid coordinates.
///
/// `ring` skips the central hole, so levels 1..CLIP_LEVELS draw only the
/// annulus around level 0's centre grid. Indexes are u32 here: CLIP_CELLS is
/// 224, so the grid has 50625 vertices and would overflow u16.
///
/// PORT NOTE: the C++ also fills a texcoord array with
/// `((x + kClipCells*0.5)/kClipCells, (z + kClipCells*0.5)/kClipCells)`.
/// Neither the GLSL terrain.vs nor terrain-vs.wgsl reads a texcoord attribute
/// (both consume vertexPosition only), so the port does not upload one.
fn build_clip_mesh(device: &RenderDevice, ring: bool) -> GpuMesh {
    let vertices_per_side = CLIP_CELLS + 1;
    let half = (CLIP_CELLS / 2) as i32;
    let hole = (CLIP_CELLS / 4) as i32;

    let mut vertices: Vec<f32> = Vec::with_capacity(vertices_per_side * vertices_per_side * 3);
    for z in -half..=half {
        for x in -half..=half {
            vertices.push(x as f32);
            vertices.push(0.0);
            vertices.push(z as f32);
        }
    }

    let mut indices: Vec<u32> = Vec::with_capacity(CLIP_CELLS * CLIP_CELLS * 6);
    for z in -half..half {
        for x in -half..half {
            if ring && x >= -hole && x < hole && z >= -hole && z < hole {
                continue;
            }
            let lower_left = ((z + half) * vertices_per_side as i32 + (x + half)) as u32;
            let upper_left = lower_left + vertices_per_side as u32;
            let lower_right = lower_left + 1;
            let upper_right = upper_left + 1;
            // Counter-clockwise seen from +Y, matching the
            // glFrontFace(GL_CCW) state raylib initialises.
            indices.extend_from_slice(&[
                lower_left,
                upper_left,
                upper_right,
                lower_left,
                upper_right,
                lower_right,
            ]);
        }
    }

    let label = if ring {
        "forest_clip_ring"
    } else {
        "forest_clip_center"
    };
    upload_mesh(device, label, &vertices, &indices)
}

/// `GenMeshPlane(16000.0f, 16000.0f, 1, 1)` from rmodels.c.
///
/// raylib bumps both resolutions to at least 2 first, so the plane is 4
/// vertices on the `(+-8000, 0, +-8000)` corners. The mesh ordering follows
/// raylib's loops exactly:
///   `vertices[x + z*resX] = ((x/(resX-1) - 0.5)*width, 0, (z/(resZ-1) - 0.5)*length)`
///   `texcoords[u + v*resX] = (u/(resX-1), v/(resZ-1))`, so index 0 is (0,0)
///   and index 3 is (1,1)
///   the single face `i = 0` emits `{i+resX, i+1, i}` then
///   `{i+resX, i+resX+1, i+1}` -> indices `{2, 1, 0, 2, 3, 1}`
///
/// GenMeshPlane leaves `mesh.colors` NULL, so raylib's DrawMesh supplies the
/// default vertex attribute (1, 1, 1, 1) — the white colour stored here.
fn build_ocean_mesh(device: &RenderDevice) -> GpuMesh {
    let half = OCEAN_PLANE_SIZE * 0.5;
    let vertices = [
        BasicVertex {
            position: [-half, 0.0, -half],
            texcoord: [0.0, 0.0],
            normal: [0.0, 1.0, 0.0],
            color: [1.0; 4],
        },
        BasicVertex {
            position: [half, 0.0, -half],
            texcoord: [1.0, 0.0],
            normal: [0.0, 1.0, 0.0],
            color: [1.0; 4],
        },
        BasicVertex {
            position: [-half, 0.0, half],
            texcoord: [0.0, 1.0],
            normal: [0.0, 1.0, 0.0],
            color: [1.0; 4],
        },
        BasicVertex {
            position: [half, 0.0, half],
            texcoord: [1.0, 1.0],
            normal: [0.0, 1.0, 0.0],
            color: [1.0; 4],
        },
    ];
    let indices: [u32; 6] = [2, 1, 0, 2, 3, 1];
    upload_mesh(device, "forest_ocean_mesh", &vertices, &indices)
}

// ---------------------------------------------------------------------------
// Layouts, samplers and bind groups
// ---------------------------------------------------------------------------

/// group(1) for the terrain pass: raylib's GL texture-unit numbering is
/// preserved, so texture unit N binds at binding N with its sampler at binding
/// N + 8. Unit 2 (the flow atlas, used by terrain-fs.wgsl) keeps its slot, and
/// the two PBR arrays added by the port live at units 5 and 6.
fn terrain_texture_layout(device: &RenderDevice) -> BindGroupLayout {
    device.create_bind_group_layout(
        "forest_terrain_textures_layout",
        &[
            wgpu::BindGroupLayoutEntry {
                binding: 0, // texture0: base noise (R32Float)
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    // Every float texture here is filtered by a linear or
                    // point sampler, exactly like the GL_LINEAR /
                    // GL_LINEAR_MIPMAP_LINEAR filters the C++ sets. R32Float
                    // and Rgba32Float only pass this check because the app
                    // enables TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES, which
                    // makes the adapter's own FILTERABLE flag authoritative.
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1, // texture1: eroded height atlas
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2, // texture2: flow atlas
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3, // texture3: erosion tile lookup (textureLoad only)
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4, // texture4: blend mask
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 5, // uAlbedoAO
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 6, // uNormalRough
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 8, // texture0_sampler: trilinear + repeat
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 9, // texture1_sampler: bilinear + clamp
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 10, // texture2_sampler: bilinear + clamp
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                // texture3_sampler: point + clamp. The lookup is only ever read
                // with textureLoad, and a non-filtering binding keeps the
                // layout honest about it.
                binding: 11,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 12, // texture4_sampler: bilinear + clamp
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 13, // uAlbedoAO sampler: trilinear + repeat
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 14, // uNormalRough sampler: trilinear + repeat
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    )
}

/// The group(2) stage-uniform layout both pipelines share.
fn stage_uniform_layout(device: &RenderDevice, label: &str, size: u64) -> BindGroupLayout {
    device.create_bind_group_layout(
        label,
        &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(size),
            },
            count: None,
        }],
    )
}

/// The seven group(1) samplers. Filters and wrap modes follow the C++
/// `SetTextureFilter` / `SetTextureWrap` calls: trilinear + REPEAT for the
/// base noise and the PBR arrays, bilinear + CLAMP for the erosion atlases and
/// the blend mask, point + CLAMP for the tile lookup. Anisotropy is a separate
/// axis and follows `CreateTerrainArray` alone: it is the reason the C++ turns
/// it on there — a grazing-angle pixel's footprint is many texels long and
/// about one wide, and an isotropic chain must blur along the long axis to fit
/// it, which combs distant hillsides into stretched streaks. The base noise
/// texture goes through raylib's `SetTextureFilter`/`SetTextureWrap`, which
/// never touch anisotropy, and the erosion targets are raw float textures, so
/// the PBR arrays are the only ones with it.
struct TerrainSamplers {
    noise: wgpu::Sampler,
    height_atlas: wgpu::Sampler,
    flow_atlas: wgpu::Sampler,
    lookup: wgpu::Sampler,
    blend_mask: wgpu::Sampler,
    albedo_array: wgpu::Sampler,
    normal_rough_array: wgpu::Sampler,
}

/// The C++ asks GL for `min(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, 16)`. wgpu has
/// no per-sampler query to mirror the minimum with, but it needs none: the
/// backend clamps the request to its own maximum and silently drops to 1 on
/// adapters without ANISOTROPIC_FILTERING, which is the same degrade-to-1 GL
/// does when the extension is missing.
const TERRAIN_ANISOTROPY_CLAMP: u16 = 16;

fn make_sampler(
    device: &RenderDevice,
    label: &str,
    filter: wgpu::FilterMode,
    address: wgpu::AddressMode,
) -> wgpu::Sampler {
    device
        .wgpu_device()
        .create_sampler(&wgpu::SamplerDescriptor {
            label: Some(label),
            address_mode_u: address,
            address_mode_v: address,
            address_mode_w: address,
            mag_filter: filter,
            min_filter: filter,
            mipmap_filter: filter,
            ..Default::default()
        })
}

/// `make_sampler` plus the anisotropic clamp the PBR arrays carry. wgpu
/// rejects anisotropy unless all three filters are linear, which every array
/// sampler satisfies.
fn make_anisotropic_sampler(
    device: &RenderDevice,
    label: &str,
    filter: wgpu::FilterMode,
    address: wgpu::AddressMode,
) -> wgpu::Sampler {
    device
        .wgpu_device()
        .create_sampler(&wgpu::SamplerDescriptor {
            label: Some(label),
            address_mode_u: address,
            address_mode_v: address,
            address_mode_w: address,
            mag_filter: filter,
            min_filter: filter,
            mipmap_filter: filter,
            anisotropy_clamp: TERRAIN_ANISOTROPY_CLAMP,
            ..Default::default()
        })
}

fn build_terrain_samplers(device: &RenderDevice) -> TerrainSamplers {
    let linear = wgpu::FilterMode::Linear;
    let nearest = wgpu::FilterMode::Nearest;
    let repeat = wgpu::AddressMode::Repeat;
    let clamp = wgpu::AddressMode::ClampToEdge;
    TerrainSamplers {
        noise: make_sampler(device, "forest_noise_sampler", linear, repeat),
        height_atlas: make_sampler(device, "forest_height_atlas_sampler", linear, clamp),
        flow_atlas: make_sampler(device, "forest_flow_atlas_sampler", linear, clamp),
        lookup: make_sampler(device, "forest_lookup_sampler", nearest, clamp),
        blend_mask: make_sampler(device, "forest_blend_mask_sampler", linear, clamp),
        albedo_array: make_anisotropic_sampler(
            device, "forest_albedo_array_sampler", linear, repeat),
        normal_rough_array: make_anisotropic_sampler(
            device, "forest_normal_rough_array_sampler", linear, repeat),
    }
}

/// The group(1) bind group over the shared world textures.
fn terrain_texture_bind_group(
    device: &RenderDevice,
    layout: &BindGroupLayout,
    textures: &GpuWorldTextures,
    samplers: &TerrainSamplers,
) -> BindGroup {
    let entries = [
        wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(&textures.noise_view),
        },
        wgpu::BindGroupEntry {
            binding: 1,
            resource: wgpu::BindingResource::TextureView(&textures.height_atlas_view),
        },
        wgpu::BindGroupEntry {
            binding: 2,
            resource: wgpu::BindingResource::TextureView(&textures.flow_atlas_view),
        },
        wgpu::BindGroupEntry {
            binding: 3,
            resource: wgpu::BindingResource::TextureView(&textures.lookup_view),
        },
        wgpu::BindGroupEntry {
            binding: 4,
            resource: wgpu::BindingResource::TextureView(&textures.blend_mask_view),
        },
        wgpu::BindGroupEntry {
            binding: 5,
            resource: wgpu::BindingResource::TextureView(&textures.albedo_array_view),
        },
        wgpu::BindGroupEntry {
            binding: 6,
            resource: wgpu::BindingResource::TextureView(&textures.normal_rough_array_view),
        },
        wgpu::BindGroupEntry {
            binding: 8,
            resource: wgpu::BindingResource::Sampler(&samplers.noise),
        },
        wgpu::BindGroupEntry {
            binding: 9,
            resource: wgpu::BindingResource::Sampler(&samplers.height_atlas),
        },
        wgpu::BindGroupEntry {
            binding: 10,
            resource: wgpu::BindingResource::Sampler(&samplers.flow_atlas),
        },
        wgpu::BindGroupEntry {
            binding: 11,
            resource: wgpu::BindingResource::Sampler(&samplers.lookup),
        },
        wgpu::BindGroupEntry {
            binding: 12,
            resource: wgpu::BindingResource::Sampler(&samplers.blend_mask),
        },
        wgpu::BindGroupEntry {
            binding: 13,
            resource: wgpu::BindingResource::Sampler(&samplers.albedo_array),
        },
        wgpu::BindGroupEntry {
            binding: 14,
            resource: wgpu::BindingResource::Sampler(&samplers.normal_rough_array),
        },
    ];
    device.create_bind_group("forest_terrain_textures", layout, &entries)
}

/// Builds one of the two G-buffer pipelines. Both write the same three targets
/// with blending off, depth writes on and back-face culling, so they share
/// this descriptor.
fn queue_gbuffer_pipeline(
    cache: &PipelineCache,
    label: &'static str,
    vertex_shader: Handle<Shader>,
    fragment_shader: Handle<Shader>,
    vertex_buffers: Vec<VertexBufferLayout>,
    layouts: Vec<BindGroupLayout>,
) -> CachedRenderPipelineId {
    cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some(label.into()),
        layout: layouts,
        push_constant_ranges: vec![],
        vertex: VertexState {
            shader: vertex_shader,
            shader_defs: vec![],
            entry_point: Some("vs_main".into()),
            buffers: vertex_buffers,
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            // PORT NOTE: raylib initialises GL with glFrontFace(GL_CCW) +
            // glCullFace(GL_BACK) in rlglInit and never changes it, and
            // crate::matrices::perspective keeps GL's y-up NDC, so the C++
            // draws both terrain meshes counter-clockwise-in-GL-window-space
            // front-facing. GL_CCW maps to wgpu's FrontFace::Ccw here:
            // verified by rendering the same scene under both settings —
            // Ccw + Back culls exactly what GL culls (the far side of the
            // clip rings), while Cw + Back erases the near terrain the C++
            // shows. The apparent y-flip argument for Cw is a red herring;
            // wgpu's front-face test is evaluated in the same y-up orientation
            // this projection produces, not in the y-down framebuffer space.
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back),
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            // CreateGBuffer uses rlLoadTextureDepth(w, h, true): a real depth
            // texture, not a renderbuffer.
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: true,
            // PORT NOTE: rlgl initialises the context with
            // glDepthFunc(GL_LEQUAL) and nothing in this path changes it, so
            // an exact tie goes to the later draw — the ocean plane wins
            // against terrain that lands on precisely the same depth.
            depth_compare: wgpu::CompareFunction::LessEqual,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(FragmentState {
            shader: fragment_shader,
            shader_defs: vec![],
            entry_point: Some("fs_main".into()),
            // The main loop calls rlDisableColorBlend() before the 3D block,
            // so each target takes the shader's raw value.
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
}

// ---------------------------------------------------------------------------
// Prepare systems
// ---------------------------------------------------------------------------

/// Builds the meshes, pipelines, uniform slots and bind groups once the shared
/// globals buffer (group 0) and the world textures (group 1) exist.
///
/// The globals guard is a hard requirement because the ocean's bind group
/// wraps that very buffer; the texture guard duplicates the explicit ordering
/// against `gpu_textures::prepare_gpu_textures` that `register_terrain_systems`
/// declares, so this system still retries safely if that ordering ever
/// changes.
fn prepare_terrain(
    device: Res<RenderDevice>,
    pipeline_cache: Res<PipelineCache>,
    shaders: Res<ForestShaderHandles>,
    globals: Res<ForestGlobals>,
    textures: Res<GpuWorldTexturesOption>,
    state: Res<TerrainNodeState>,
) {
    let mut resources = state
        .resources
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    if resources.is_some() {
        return;
    }
    let (Some(globals_buffer), Some(textures)) = (globals.buffer.as_ref(), textures.0.as_deref())
    else {
        return;
    };
    let device = &*device;

    // group(0): the frame's shared globals, created through the same layout
    // and entries every other pass uses.
    let globals_layout = crate::render::globals_layout(device);
    let globals_entries = crate::render::globals_bind_group_entries(globals_buffer);
    let globals_bind_group = device.create_bind_group(
        "forest_terrain_globals_bind_group",
        &globals_layout,
        &globals_entries,
    );

    // group(1): the world textures; group(2): the per-draw stage blocks.
    let terrain_textures_layout = terrain_texture_layout(device);
    let samplers = build_terrain_samplers(device);
    let terrain_textures =
        terrain_texture_bind_group(device, &terrain_textures_layout, textures, &samplers);
    let terrain_stage_layout = stage_uniform_layout(
        device,
        "forest_terrain_stage_layout",
        std::mem::size_of::<TerrainStageUniforms>() as u64,
    );
    let ocean_stage_layout = stage_uniform_layout(
        device,
        "forest_ocean_stage_layout",
        std::mem::size_of::<BasicStageUniforms>() as u64,
    );
    // basic-gbuffer samples no textures, but its pipeline layout still needs a
    // group-1 slot so the stage block stays at index 2.
    let empty_layout = device.create_bind_group_layout("forest_no_textures_layout", &[]);
    let empty_textures =
        device.create_bind_group("forest_no_textures_bind_group", &empty_layout, &[]);

    // Terrain: terrain-vs + terrain-fs over a plain vec3 position. Ocean:
    // basic-gbuffer over raylib's four attributes. Same three colour targets.
    let terrain_pipeline = queue_gbuffer_pipeline(
        &pipeline_cache,
        "forest_terrain_pipeline",
        shaders.terrain_vs.clone(),
        shaders.terrain_fs.clone(),
        vec![VertexBufferLayout {
            array_stride: TERRAIN_VERTEX_STRIDE,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: TERRAIN_VERTEX_ATTRIBUTES.to_vec(),
        }],
        vec![
            globals_layout.clone(),
            terrain_textures_layout,
            terrain_stage_layout.clone(),
        ],
    );
    let ocean_pipeline = queue_gbuffer_pipeline(
        &pipeline_cache,
        "forest_ocean_pipeline",
        shaders.basic.clone(),
        shaders.basic.clone(),
        vec![VertexBufferLayout {
            array_stride: BASIC_VERTEX_STRIDE,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: BASIC_VERTEX_ATTRIBUTES.to_vec(),
        }],
        vec![globals_layout, empty_layout, ocean_stage_layout.clone()],
    );

    let levels: [StageUniform; CLIP_LEVELS] = std::array::from_fn(|_| {
        StageUniform::new(
            device,
            &terrain_stage_layout,
            "forest_terrain_stage_uniform",
            std::mem::size_of::<TerrainStageUniforms>() as u64,
        )
    });
    let ocean = StageUniform::new(
        device,
        &ocean_stage_layout,
        "forest_ocean_stage_uniform",
        std::mem::size_of::<BasicStageUniforms>() as u64,
    );

    *resources = Some(TerrainResources {
        globals: globals_bind_group,
        terrain_textures,
        empty_textures,
        terrain_pipeline,
        ocean_pipeline,
        center_mesh: build_clip_mesh(device, false),
        ring_mesh: build_clip_mesh(device, true),
        ocean_mesh: build_ocean_mesh(device),
        levels,
        ocean,
    });

    // `samplers` and the standalone layouts are deliberately not stored: a
    // wgpu bind group keeps its bound resources alive, and the
    // RenderPipelineDescriptors queued above (retained by the PipelineCache)
    // keep the layouts alive for as long as the pipelines exist.
}

/// `CreateGBuffer` plus the main loop's `rlViewport(0, 0, ssao.width,
/// ssao.height)`: (re)creates the colour targets and the depth texture
/// whenever the physical window size changes, along with the three filter
/// samplers the C++ sets on them.
fn resize_gbuffer(
    device: Res<RenderDevice>,
    view: Res<ExtractedForestView>,
    state: Res<TerrainNodeState>,
) {
    let width = view.physical_width.max(1);
    let height = view.physical_height.max(1);
    {
        let mut gbuffer = state
            .gbuffer
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if let Some(existing) = gbuffer.as_ref() {
            if existing.width == width && existing.height == height {
                return;
            }
        }

        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        // CreateGBuffer: RGBA32F position, RGBA16F normal, RGBA8 albedo, then
        // rlLoadTextureDepth(w, h, true). Each view keeps its own texture
        // alive, so the wgpu texture handles are not stored.
        let position_texture = device.wgpu_device().create_texture(&wgpu::TextureDescriptor {
            label: Some("forest_gbuffer_position"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let normal_texture = device.wgpu_device().create_texture(&wgpu::TextureDescriptor {
            label: Some("forest_gbuffer_normal"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let albedo_texture = device.wgpu_device().create_texture(&wgpu::TextureDescriptor {
            label: Some("forest_gbuffer_albedo"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let depth_texture = device.wgpu_device().create_texture(&wgpu::TextureDescriptor {
            label: Some("forest_gbuffer_depth"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        *gbuffer = Some(GbufferTargets {
            position_view: position_texture.create_view(&wgpu::TextureViewDescriptor::default()),
            normal_view: normal_texture.create_view(&wgpu::TextureViewDescriptor::default()),
            albedo_view: albedo_texture.create_view(&wgpu::TextureViewDescriptor::default()),
            depth_view: depth_texture.create_view(&wgpu::TextureViewDescriptor::default()),
            width,
            height,
        });
    }

}

/// Render-app registration for the terrain pass: the resize check first, then
/// the one-time resource build, both in the `Render` schedule's `Prepare` set
/// and after the shared globals buffer exists. The world-texture dependency is
/// covered by the guard inside [`prepare_terrain`].
pub fn register_terrain_systems(render_app: &mut bevy::app::SubApp) {
    render_app.add_systems(
        bevy::render::Render,
        (resize_gbuffer, prepare_terrain)
            .chain()
            .in_set(bevy::render::RenderSystems::Prepare)
            .after(crate::render::prepare_forest_globals)
            // Textures before the first frame rather than the second; the
            // guard in `prepare_terrain` still covers a reordering.
            .after(crate::render::gpu_textures::prepare_gpu_textures),
    );
}
