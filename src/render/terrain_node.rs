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
    ExtractedForestView, ForestGlobals, ForestShaderHandles, GlobalUniformsGpu,
    TerrainStageUniforms,
};
use bevy::asset::Handle;
use bevy::mesh::VertexBufferLayout;
use bevy::prelude::*;
use bevy::render::render_resource::{
    BindGroup, BindGroupLayoutDescriptor, Buffer, CachedComputePipelineId,
    CachedRenderPipelineId, ComputePipelineDescriptor, FragmentState, PipelineCache,
    RenderPipelineDescriptor, VertexState,
};
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue};
use bevy::shader::Shader;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

// The ocean placeholder plane that used to live here (a 16 km `DrawWorldGeometry`
// quad tracking the player in 256 m steps, tinted OCEAN_COLOR and drawn through
// `basic-gbuffer`) is gone. The ocean is now the vendored bevy-aqua surface in
// `src/render/water_node.rs`, which shades itself and is drawn after the
// composite rather than into this G-buffer.

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

/// World-space shadow coverage extends beyond the 5800 m camera horizon.
/// Reusing terrainHeight in the vertex shader keeps streamed erosion and the
/// procedural landform identical between visible geometry and raymarching.
pub const LIGHTING_HEIGHTFIELD_SIZE: u32 = 1024;
pub const LIGHTING_HEIGHTFIELD_SPAN: f32 = 12288.0;

/// Near-shore bathymetry resolves one-metre terrain features independently of
/// the much wider lighting map. Whole-texel snapping keeps the sampled bed
/// fixed in world space as the camera moves.
pub const SHORE_HEIGHTFIELD_SIZE: u32 = 512;
pub const SHORE_HEIGHTFIELD_SPAN: f32 = 512.0;
/// The shore window moves eight texels at a time.
const SHORE_HEIGHTFIELD_SNAP: f32 = SHORE_HEIGHTFIELD_SPAN / SHORE_HEIGHTFIELD_SIZE as f32 * 8.0;

// The shore capture is kept while the player walks about inside its snap, so
// the erosion fade it was rendered with must be flat over every texel it
// holds: the window's far corner plus the snap, with room to spare, lies well
// inside the radius where erosion is at full strength. The lighting map
// reaches far past the fade, so its key holds the fade's centre instead.
const _: () = assert!(
    (SHORE_HEIGHTFIELD_SPAN / 2.0 + SHORE_HEIGHTFIELD_SNAP) * 1.5 < EROSION_VISIBILITY_FULL_RADIUS,
    "the shore heightfield window must sit inside the full-strength erosion radius"
);

/// Fixed world lattice shared with instanced vegetation. Sub-metre samples
/// retain narrow bare patches; root-footprint tests dilate their exclusion.
pub const GRASS_HABITAT_SIZE: u32 = 768;
pub const GRASS_HABITAT_SPAN: f32 = 512.0;
/// A two-by-two linear-light box average covers 1.33 m of terrain per texel.
/// Bilinear sampling adds a smooth local footprint without repeated vertex
/// fetches when the close grass density reaches dozens of clumps per m².
const GRASS_GROUND_AVERAGE_SIZE: u32 = GRASS_HABITAT_SIZE / 2;

pub fn grass_habitat_mapping(position: [f32; 3]) -> [f32; 4] {
    [
        (position[0] / 8.0).floor() * 8.0,
        (position[2] / 8.0).floor() * 8.0,
        GRASS_HABITAT_SPAN,
        GRASS_HABITAT_SPAN / GRASS_HABITAT_SIZE as f32,
    ]
}

// The capture is kept across frames and the player walks off its centre, so
// the erosion fade it was rendered with must be flat over every texel it
// holds: the window's far corner plus the 8 m snap, with room to spare, lies
// well inside the radius where erosion is at full strength.
const _: () = assert!(
    (GRASS_HABITAT_SPAN / 2.0 + 8.0) * 1.5 < EROSION_VISIBILITY_FULL_RADIUS,
    "the grass habitat window must sit inside the full-strength erosion radius"
);

/// The globals the habitat capture renders with: a top-down orthographic view
/// of the window `mapping` describes. The material shader's one use of the
/// camera is how far its slope samples reach (`materialStep` in terrain-vs),
/// so the "camera" is the middle of the window rather than the player. That
/// makes the capture a function of its window alone, and a capture taken
/// anywhere in the window is the one every other position in it would take.
/// `crest` is the ocean's wave-crest clearance when there is an ocean.
fn habitat_capture_globals(
    globals: &GlobalUniformsGpu,
    mapping: [f32; 4],
    crest: Option<f32>,
) -> GlobalUniformsGpu {
    let mut capture = *globals;
    capture.view = Mat4::IDENTITY.to_cols_array();
    let scale = 2.0 / mapping[2];
    capture.projection = [
        scale, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, -scale, 0.0, 0.0,
        -mapping[0] * scale, mapping[1] * scale, 0.5, 1.0,
    ];
    // The snap puts the player within 8 m of the window's corner; the middle
    // of that square is the best single stand-in for them.
    capture.camera_position = [
        mapping[0] + 4.0,
        0.0,
        mapping[1] + 4.0,
        crest.unwrap_or(globals.camera_position[3]),
    ];
    capture
}

/// Everything the habitat capture and the ground average derived from it
/// depend on. While this is unchanged the held capture is exactly what a new
/// one would be, so it is kept. Anything the capture reads belongs here: when
/// the shader gains an input, add it, or a stale capture will outlive it.
#[derive(Clone, Copy, Debug, PartialEq)]
struct HabitatKey {
    /// The snapped window (centre XZ), as bit patterns so equality is exact.
    mapping: [u32; 2],
    /// The clipmap's centre. Its triangle layout and LOD morph follow it, and
    /// the capture draws those triangles.
    clip_origin: [u32; 2],
    lookup_minimum: (i64, i64),
    /// Erosion lookup and atlas contents, see `TerrainRevision`.
    terrain_revision: u64,
    /// The globals the material shader reads in a capture, as shaded: texture
    /// scale, variant scale, normal strength, flow and erosion debug.
    shading: [u32; 5],
    /// The sea state behind the wave-crest clearance (amplitude, wind
    /// direction, flat), when there is an ocean.
    sea: Option<[u32; 3]>,
}

impl HabitatKey {
    fn new(
        player_position: [f32; 3],
        lookup_minimum: (i64, i64),
        globals: &GlobalUniformsGpu,
        water: Option<&super::water_node::ExtractedWater>,
        terrain_revision: u64,
    ) -> Self {
        let mapping = grass_habitat_mapping(player_position);
        Self {
            mapping: [mapping[0].to_bits(), mapping[1].to_bits()],
            clip_origin: TerrainStageUniforms::clip_origin(player_position).map(f32::to_bits),
            lookup_minimum,
            terrain_revision,
            shading: [
                globals.settings_a[1].to_bits(),
                globals.settings_a[3].to_bits(),
                globals.settings_b[0].to_bits(),
                globals.settings_b[2].to_bits(),
                globals.settings_b[3].to_bits(),
            ],
            sea: water.map(|water| {
                [
                    water.settings.sea_state_amplitude.to_bits(),
                    water.settings.wind_direction_degrees.to_bits(),
                    water.settings.flat_surface as u32,
                ]
            }),
        }
    }
}

/// Captures are numbered, never reusing a number, so a consumer can tell "the
/// same capture as last frame" from "a new one" even across a G-buffer rebuild.
static HABITAT_GENERATIONS: AtomicU64 = AtomicU64::new(1);

/// Which habitat capture the G-buffer's capture textures hold. The capture is
/// kept until its inputs change, and the player moves about inside its
/// window meanwhile, so everything that reads the capture must place it by
/// this record rather than by the player's position.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct HabitatCapture {
    /// The window the held capture covers, in `grass_habitat_mapping`
    /// layout; all zero until the first capture.
    pub mapping: [f32; 4],
    /// Which capture is held; changes whenever it is re-rendered.
    pub generation: u64,
    /// What the held capture was rendered from, `None` until there is one.
    key: Option<HabitatKey>,
    /// Whether the habitat texture holds the capture. False while a
    /// recapture is under way, when that texture has been cleared for it.
    pub ready: bool,
}

impl HabitatCapture {
    /// Whether the held capture is the one `key` asks for, and whole.
    fn holds(&self, key: HabitatKey) -> bool {
        self.ready && self.key == Some(key)
    }

    /// A recapture has begun: the habitat texture is cleared and redrawn, so
    /// the grass must not read it until [`Self::complete`]. The ground
    /// average, and the canopy the vegetation pass lays over it, still hold
    /// the previous capture until then, so its window stays as it is.
    fn begin(&mut self) {
        self.ready = false;
    }

    /// The capture of `mapping` is encoded into every capture texture.
    /// `upload` writes its window into the uniform the water places the
    /// canopy by, in the same pass as the texels: a window written anywhere
    /// else, before this pass in the frame (as the water's own block is) or
    /// from a capture still pending, would pair these texels with another
    /// capture's window for a frame, and shift every crown the water sees by
    /// a whole 8 m window.
    fn complete(&mut self, key: HabitatKey, mapping: [f32; 4], upload: impl FnOnce(&[f32; 4])) {
        self.key = Some(key);
        self.mapping = mapping;
        self.generation = HABITAT_GENERATIONS.fetch_add(1, Ordering::Relaxed);
        self.ready = true;
        upload(&self.mapping);
    }
}

/// Shared by the terrain capture and water sampling; xy = centre, z = span,
/// w = metres per texel. One source prevents the shoreline drifting between
/// the vertex and fragment passes or when crossing a snap boundary.
pub fn shore_heightfield_mapping(camera_position: [f32; 4]) -> [f32; 4] {
    let texel = SHORE_HEIGHTFIELD_SPAN / SHORE_HEIGHTFIELD_SIZE as f32;
    let snap = SHORE_HEIGHTFIELD_SNAP;
    [
        (camera_position[0] / snap).floor() * snap,
        (camera_position[2] / snap).floor() * snap,
        SHORE_HEIGHTFIELD_SPAN,
        texel,
    ]
}

/// Everything a heightfield capture depends on, and for the lighting map the
/// highest terrain reduced from it too. While this is unchanged the held
/// capture is exactly what a new one would be, so it is kept. A capture
/// evaluates `terrainHeight` alone, whose other inputs (the base noise, the
/// blend mask and the stage constants) never change after startup; when the
/// shader gains an input, add it here, or a stale capture will outlive it.
#[derive(Clone, Copy, Debug, PartialEq)]
struct HeightfieldKey {
    /// The captured window (`globals.heightfield` layout), as bit patterns so
    /// equality is exact.
    mapping: [u32; 4],
    /// The erosion fade's centre, the player's XZ, for a window the fade
    /// reaches into. `None` for a window wholly at full strength.
    fade_centre: Option<[u32; 2]>,
    lookup_minimum: (i64, i64),
    /// Erosion lookup, atlas and river contents, see `TerrainRevision`.
    terrain_revision: u64,
}

impl HeightfieldKey {
    /// The lighting map reaches far past the erosion fade, so every step the
    /// player takes moves the faded ring of erosion across it.
    fn lighting(
        mapping: [f32; 4],
        fade_centre: [f32; 2],
        lookup_minimum: (i64, i64),
        terrain_revision: u64,
    ) -> Self {
        Self {
            mapping: mapping.map(f32::to_bits),
            fade_centre: Some(fade_centre.map(f32::to_bits)),
            lookup_minimum,
            terrain_revision,
        }
    }

    /// The shore map lies wholly at full erosion strength (see
    /// SHORE_HEIGHTFIELD_SNAP), so the player can walk about inside its snap
    /// without retaking it.
    fn shore(mapping: [f32; 4], lookup_minimum: (i64, i64), terrain_revision: u64) -> Self {
        Self {
            mapping: mapping.map(f32::to_bits),
            fade_centre: None,
            lookup_minimum,
            terrain_revision,
        }
    }
}

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
    pub heightfield_view: wgpu::TextureView, // R32Float world-space terrain height
    pub shore_heightfield_view: wgpu::TextureView, // R32Float local seabed elevation
    /// What the held lighting heightfield was rendered from, `None` until
    /// there is one.
    lighting_heightfield_key: Option<HeightfieldKey>,
    /// Whether the highest-terrain buffer holds the held lighting
    /// heightfield's reduction. The buffer outlives a resize; this does not,
    /// because the resize discards the capture the reduction was taken from.
    lighting_highest_ready: bool,
    /// What the held shore heightfield was rendered from, `None` until there
    /// is one.
    shore_heightfield_key: Option<HeightfieldKey>,
    /// True while a shore capture matches this frame's inputs and the sea is
    /// drawn. Water must never sample an empty or stale map.
    pub shore_heightfield_ready: bool,
    pub grass_habitat_view: wgpu::TextureView,
    /// Linear terrain albedo at the same world-space texels as grass_habitat_view.
    pub grass_ground_albedo_view: wgpu::TextureView,
    /// Linear terrain albedo averaged over neighboring ground texels, as the
    /// terrain pass leaves it: before any tree crown shades it.
    pub grass_ground_base_view: wgpu::TextureView,
    /// `grass_ground_base_view` with the crowns' shade multiplied into its
    /// alpha, which is what the grass reads. The vegetation pass makes it from
    /// the base (the shade is a multiply, so it must start from the base each
    /// time); until then it is a plain copy.
    pub grass_ground_average_view: wgpu::TextureView,
    grass_ground_average_bind_group: BindGroup,
    /// Which capture the textures above hold.
    pub grass_habitat: HabitatCapture,
    /// `grass_habitat.mapping` as a 16-byte uniform, for the GPU passes that
    /// sample the capture after the frame's capture lands: the water, which
    /// reads the canopy over the ground average. Written by the capture's own
    /// pass ([`HabitatCapture::complete`]) and nowhere else; zero, no
    /// capture, until the first.
    pub grass_habitat_mapping_buffer: Buffer,
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
    fn new(
        device: &RenderDevice,
        cache: &PipelineCache,
        layout: &BindGroupLayoutDescriptor,
        label: &'static str,
        size: u64,
    ) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = super::bind_group(
            device,
            cache,
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
    snow_texture: wgpu::Texture,
    snow_uniform: Buffer,
    snow_revision: AtomicU64,
    terrain_pipeline: CachedRenderPipelineId,
    heightfield_pipeline: CachedRenderPipelineId,
    /// Reduces the lighting heightfield to its highest texel.
    highest_pipeline: CachedComputePipelineId,
    highest_layout: BindGroupLayoutDescriptor,
    /// The reduction's result, as the bits of a non-negative f32.
    highest_buffer: Buffer,
    habitat_pipeline: CachedRenderPipelineId,
    grass_ground_average_pipeline: CachedRenderPipelineId,
    habitat_globals: StageUniform,
    /// The same terrain-height shader rendered through a local map transform.
    shore_globals: StageUniform,
    center_mesh: GpuMesh,
    ring_mesh: GpuMesh,
    /// One stage block per clipmap level, matching `DrawClipmap`'s per-level
    /// uniforms.
    levels: [StageUniform; CLIP_LEVELS],
}

// ---------------------------------------------------------------------------
// The terrain G-buffer pass
// ---------------------------------------------------------------------------

/// Draws the clipmap G-buffer: view-space position, encoded normal +
/// roughness and albedo + masks, plus the shared depth buffer.
pub fn forest_terrain_pass(world: &World, mut ctx: RenderContext) {
    let Some(state) = world.get_resource::<TerrainNodeState>() else {
        return;
    };
    // Both slots are filled by the prepare systems above; an empty slot
    // just means "nothing to draw yet" (first frame, or before the world
    // textures exist). Never panic on it.
    //
    // Lock order: gbuffer first, then resources. No other system ever
    // holds both, so this cannot deadlock.
    let mut gbuffer_guard = state
        .gbuffer
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let resources_guard = state
        .resources
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let (Some(gbuffer), Some(resources)) = (gbuffer_guard.as_mut(), resources_guard.as_ref())
    else {
        return;
    };
    let Some(view) = world.get_resource::<ExtractedForestView>() else {
        return;
    };
    let Some(globals) = world.get_resource::<ForestGlobals>() else {
        return;
    };
    let Some(queue) = world.get_resource::<RenderQueue>() else {
        return;
    };
    let Some(pipeline_cache) = world.get_resource::<PipelineCache>() else {
        return;
    };

    // The texture object stays bound while its world window moves. Sparse CPU
    // storage restores old tracks when the player returns to a snowy area.
    if resources.snow_revision.load(Ordering::Relaxed) != view.snow_revision {
        queue.write_buffer(&resources.snow_uniform, 0, bytemuck::bytes_of(&view.snow_mapping));
        queue.write_texture(
            resources.snow_texture.as_image_copy(),
            &view.snow_pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(crate::snow::SNOW_MAP_SIZE as u32),
                rows_per_image: Some(crate::snow::SNOW_MAP_SIZE as u32),
            },
            wgpu::Extent3d {
                width: crate::snow::SNOW_MAP_SIZE as u32,
                height: crate::snow::SNOW_MAP_SIZE as u32,
                depth_or_array_layers: 1,
            },
        );
        resources.snow_revision.store(view.snow_revision, Ordering::Relaxed);
    }

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

    // Everything derived from the terrain (the two heightfields and the grass
    // habitat below) is kept until this moves or its own inputs change.
    let terrain_revision = world
        .get_resource::<GpuWorldTexturesOption>()
        .and_then(|option| option.0.as_deref())
        .map_or(0, |textures| textures.revision.current());
    let water = world.get_resource::<super::water_node::ExtractedWater>();

    // This is independent of camera visibility: hills behind the camera
    // still cast shadows and occlude light inside the fog. Rendering after
    // erosion and before lighting also follows tile reveals without a CPU
    // height readback. The capture is a function of the inputs in
    // `HeightfieldKey`; it is redone when one of them changes (the window
    // moving on, a tile streaming in or revealing, a new river network, the
    // player stepping, which moves the erosion fade across the map) and kept
    // otherwise, so standing still costs nothing.
    if view.settings.raymarched_shadows
        && let Some(pipeline) = pipeline_cache.get_render_pipeline(resources.heightfield_pipeline)
    {
        let key = HeightfieldKey::lighting(
            globals.globals.heightfield,
            visibility_center,
            view.lookup_minimum,
            terrain_revision,
        );
        if gbuffer.lighting_heightfield_key != Some(key) {
            let lighting_pass_descriptor = wgpu::RenderPassDescriptor {
                label: Some("forest_lighting_heightfield"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &gbuffer.heightfield_view,
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
            };
            let mut pass = ctx.begin_tracked_render_pass(lighting_pass_descriptor);
            pass.set_render_pipeline(pipeline);
            pass.set_bind_group(0, &resources.globals, &[]);
            pass.set_bind_group(1, &resources.terrain_textures, &[]);
            pass.set_bind_group(2, &resources.levels[0].bind_group, &[]);
            pass.draw(0..3, 0..1);
            drop(pass);
            gbuffer.lighting_heightfield_key = Some(key);
            gbuffer.lighting_highest_ready = false;
        }

        // The highest terrain in the map lets the sun-visibility marches stop
        // once they rise above it. Reduced on the GPU once per capture, so no
        // CPU readback and no stale bound, and copied into the globals every
        // frame, because `prepare_forest_globals` writes them afresh with
        // TERRAIN_HEIGHT_UNKNOWN; until there is a reduction the marches run
        // in full.
        if !gbuffer.lighting_highest_ready
            && let (Some(highest), Some(device)) = (
                pipeline_cache.get_compute_pipeline(resources.highest_pipeline),
                world.get_resource::<RenderDevice>(),
            )
        {
            let highest_entries = [
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&gbuffer.heightfield_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: resources.highest_buffer.as_entire_binding(),
                },
            ];
            let group = super::bind_group(
                device,
                pipeline_cache,
                "forest_heightfield_highest",
                &resources.highest_layout,
                &highest_entries,
            );
            let encoder = ctx.command_encoder();
            encoder.clear_buffer(&resources.highest_buffer, 0, None);
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("forest_heightfield_highest"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(highest);
                pass.set_bind_group(0, Some(&*group), &[]);
                let groups = LIGHTING_HEIGHTFIELD_SIZE.div_ceil(16);
                pass.dispatch_workgroups(groups, groups, 1);
            }
            gbuffer.lighting_highest_ready = true;
        }
        if gbuffer.lighting_highest_ready
            && let Some(globals_buffer) = globals.buffer.as_ref()
        {
            ctx.command_encoder().copy_buffer_to_buffer(
                &resources.highest_buffer,
                0,
                globals_buffer,
                super::TERRAIN_HEIGHT_OFFSET,
                4,
            );
        }
    }

    // A separate one-metre map gives the water a world-space seabed, including
    // off-screen shores. It follows the exact same terrain/erosion function as
    // the mesh, and remains available when raymarched shadows are disabled.
    // It is kept like the lighting map, but its window lies wholly inside the
    // full-strength erosion radius, so the erosion fade is flat over it.
    let draw_sea = water.is_some_and(|water| water.draw);
    let shore_mapping = shore_heightfield_mapping(globals.globals.camera_position);
    let shore_key = HeightfieldKey::shore(shore_mapping, view.lookup_minimum, terrain_revision);
    if draw_sea
        && gbuffer.shore_heightfield_key != Some(shore_key)
        && let Some(pipeline) = pipeline_cache.get_render_pipeline(resources.heightfield_pipeline)
    {
        let mut local_globals = globals.globals;
        local_globals.heightfield = shore_mapping;
        queue.write_buffer(&resources.shore_globals.buffer, 0, bytemuck::bytes_of(&local_globals));
        let shore_pass_descriptor = wgpu::RenderPassDescriptor {
            label: Some("forest_shore_heightfield"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &gbuffer.shore_heightfield_view,
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
        };
        let mut pass = ctx.begin_tracked_render_pass(shore_pass_descriptor);
        pass.set_render_pipeline(pipeline);
        pass.set_bind_group(0, &resources.shore_globals.bind_group, &[]);
        pass.set_bind_group(1, &resources.terrain_textures, &[]);
        pass.set_bind_group(2, &resources.levels[0].bind_group, &[]);
        pass.draw(0..3, 0..1);
        drop(pass);
        gbuffer.shore_heightfield_key = Some(shore_key);
    }
    gbuffer.shore_heightfield_ready = draw_sea && gbuffer.shore_heightfield_key == Some(shore_key);

    // Capture the *visible clipmap's* interpolated height and material contacts.
    // The capture is a function of the inputs in `HabitatKey`; it is redone when
    // one of them changes (the player crossing into the next 8 m window, a
    // freshly streamed erosion tile, a settings change) and kept otherwise, so
    // standing still costs nothing. Reveals and regeneration still reach the
    // grass: they move the key.
    let mapping = grass_habitat_mapping(view.player_position);
    let key = HabitatKey::new(
        view.player_position,
        view.lookup_minimum,
        &globals.globals,
        water,
        terrain_revision,
    );
    if !gbuffer.grass_habitat.holds(key)
        && let Some(pipeline) = pipeline_cache.get_render_pipeline(resources.habitat_pipeline)
    {
        gbuffer.grass_habitat.begin();
        // A crest bound protects roots even when the user increases the sea state.
        let crest_bound = |water: &super::water_node::ExtractedWater| {
            let spectrum = crate::water::waves::build(
                water.settings.sea_state_amplitude,
                water.settings.wind_direction_degrees.to_radians(),
            );
            crate::water::displacement_bounds(&spectrum, water.settings.flat_surface) + 0.10
        };
        let crest = water.map(crest_bound);
        let habitat_globals = habitat_capture_globals(&globals.globals, mapping, crest);
        queue.write_buffer(&resources.habitat_globals.buffer, 0, bytemuck::bytes_of(&habitat_globals));
        let habitat_attachments = [
            Some(wgpu::RenderPassColorAttachment {
                view: &gbuffer.grass_habitat_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            }),
            Some(wgpu::RenderPassColorAttachment {
                view: &gbuffer.grass_ground_albedo_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            }),
        ];
        let habitat_pass_descriptor = wgpu::RenderPassDescriptor {
            label: Some("forest_grass_habitat"),
            color_attachments: &habitat_attachments,
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        };
        let mut pass = ctx.begin_tracked_render_pass(habitat_pass_descriptor);
        pass.set_render_pipeline(pipeline);
        pass.set_bind_group(0, &resources.habitat_globals.bind_group, &[]);
        pass.set_bind_group(1, &resources.terrain_textures, &[]);
        // Five levels include the two new sub-metre grids and still cover
        // the complete 512 m capture. The habitat vertex entry reads the
        // underlying terrain so snow tracks do not invalidate the grass map.
        for (level, stage) in resources.levels.iter().take(5).enumerate() {
            let mesh = if level == 0 { &resources.center_mesh } else { &resources.ring_mesh };
            pass.set_bind_group(2, &stage.bind_group, &[]);
            pass.set_vertex_buffer(0, mesh.vertices.slice(..));
            pass.set_index_buffer(mesh.indices.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..mesh.index_count, 0, 0..1);
        }
        drop(pass);

        // Four full-resolution ground samples become one filterable texel in
        // linear RGB. This pass runs after the latest erosion/material capture
        // and before any grass is drawn, so every clump reads the local ground
        // colour with a single vertex texture lookup. It writes the base, which
        // is then copied to the texture the grass reads, and the vegetation pass
        // copies it again before shading it under the crowns.
        if let Some(average_pipeline) = pipeline_cache
            .get_render_pipeline(resources.grass_ground_average_pipeline)
        {
            let average_pass_descriptor = wgpu::RenderPassDescriptor {
                label: Some("forest_grass_ground_average"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &gbuffer.grass_ground_base_view,
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
            };
            let mut average_pass = ctx.begin_tracked_render_pass(average_pass_descriptor);
            average_pass.set_render_pipeline(average_pipeline);
            average_pass.set_bind_group(0, &gbuffer.grass_ground_average_bind_group, &[]);
            average_pass.draw(0..3, 0..1);
            drop(average_pass);
            copy_ground_average(ctx.command_encoder(), gbuffer);
            // Queued writes land ahead of the frame's command buffers, so the
            // window reaches the water in the same frame as the texels.
            gbuffer.grass_habitat.complete(key, mapping, |window| {
                queue.write_buffer(&gbuffer.grass_habitat_mapping_buffer, 0, bytemuck::bytes_of(window));
            });
        }
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
                // Reverse-Z: the far plane is 0, so the clear value is the
                // far distance rather than the near one.
                load: wgpu::LoadOp::Clear(0.0),
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    };
    let mut pass = ctx.begin_tracked_render_pass(pass_descriptor);
    pass.set_viewport(
        0.0,
        0.0,
        gbuffer.width as f32,
        gbuffer.height as f32,
        0.0,
        1.0,
    );

    // The C++ walks the levels in order and draws
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
            pass.set_index_buffer(mesh.indices.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..mesh.index_count, 0, 0..1);
        }
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

// ---------------------------------------------------------------------------
// Layouts, samplers and bind groups
// ---------------------------------------------------------------------------

/// group(1) for the terrain pass: raylib's GL texture-unit numbering is
/// preserved, so texture unit N binds at binding N with its sampler at binding
/// N + 8. Unit 2 (the flow atlas, used by terrain-fs.wgsl) keeps its slot, and
/// the two PBR arrays added by the port live at units 5 and 6.
fn terrain_texture_layout() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
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
            wgpu::BindGroupLayoutEntry {
                binding: 7,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 15,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 16,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: std::num::NonZeroU64::new(16),
                },
                count: None,
            },
            river_storage_entry(17), // river lookup grid
            river_storage_entry(18), // river carve segments
        ],
    )
}

/// A read-only river storage buffer, read where the ground height is.
fn river_storage_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Input to the local terrain-colour averaging pass.
fn grass_ground_average_layout() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        "forest_grass_ground_average_layout",
        &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        }],
    )
}

/// The group(2) stage-uniform layout shared by terrain pipelines.
fn stage_uniform_layout(label: &'static str, size: u64) -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
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
    snow: wgpu::Sampler,
}

/// The C++ asks GL for `min(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, 16)`. wgpu has
/// no per-sampler query to mirror the minimum with, but it needs none: the
/// backend clamps the request to its own maximum and silently drops to 1 on
/// adapters without ANISOTROPIC_FILTERING, which is the same degrade-to-1 GL
/// does when the extension is missing.
const TERRAIN_ANISOTROPY_CLAMP: u16 = 16;

/// wgpu 29 split the mip filter into its own type; map the mag/min filter
/// across so the three stay in step.
fn mipmap_mode(filter: wgpu::FilterMode) -> wgpu::MipmapFilterMode {
    match filter {
        wgpu::FilterMode::Nearest => wgpu::MipmapFilterMode::Nearest,
        wgpu::FilterMode::Linear => wgpu::MipmapFilterMode::Linear,
    }
}

fn make_sampler(
    device: &RenderDevice,
    label: &str,
    filter: wgpu::FilterMode,
    address: wgpu::AddressMode,
) -> wgpu::Sampler {
    let sampler_descriptor = wgpu::SamplerDescriptor {
        label: Some(label),
        address_mode_u: address,
        address_mode_v: address,
        address_mode_w: address,
        mag_filter: filter,
        min_filter: filter,
        mipmap_filter: mipmap_mode(filter),
        ..Default::default()
    };
    device
        .wgpu_device()
        .create_sampler(&sampler_descriptor)
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
    let sampler_descriptor = wgpu::SamplerDescriptor {
        label: Some(label),
        address_mode_u: address,
        address_mode_v: address,
        address_mode_w: address,
        mag_filter: filter,
        min_filter: filter,
        mipmap_filter: mipmap_mode(filter),
        anisotropy_clamp: TERRAIN_ANISOTROPY_CLAMP,
        ..Default::default()
    };
    device
        .wgpu_device()
        .create_sampler(&sampler_descriptor)
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
        snow: make_sampler(device, "forest_snow_sampler", linear, clamp),
    }
}

/// The group(1) bind group over the shared world textures.
fn terrain_texture_bind_group(
    device: &RenderDevice,
    cache: &PipelineCache,
    layout: &BindGroupLayoutDescriptor,
    textures: &GpuWorldTextures,
    samplers: &TerrainSamplers,
    snow_view: &wgpu::TextureView,
    snow_uniform: &Buffer,
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
        wgpu::BindGroupEntry {
            binding: 7,
            resource: wgpu::BindingResource::TextureView(snow_view),
        },
        wgpu::BindGroupEntry {
            binding: 15,
            resource: wgpu::BindingResource::Sampler(&samplers.snow),
        },
        wgpu::BindGroupEntry {
            binding: 16,
            resource: snow_uniform.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 17,
            resource: textures.river_grid.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 18,
            resource: textures.river_segments.as_entire_binding(),
        },
    ];
    super::bind_group(device, cache, "forest_terrain_textures", layout, &entries)
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
    layouts: Vec<BindGroupLayoutDescriptor>,
) -> CachedRenderPipelineId {
    let descriptor = RenderPipelineDescriptor {
        label: Some(label.into()),
        layout: layouts,
        immediate_size: 0,
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
            depth_write_enabled: Some(true),
            // DIVERGENCE FROM THE C++: rlgl runs glDepthFunc(GL_LEQUAL) against
            // a forward projection. The projection is reverse-Z now (see
            // crate::matrices::perspective), so the comparison flips with it:
            // GreaterEqual keeps the same "an exact tie goes to the later draw,
            // so the water surface wins against terrain at identical depth".
            depth_compare: Some(wgpu::CompareFunction::GreaterEqual),
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
    };
    cache.queue_render_pipeline(descriptor)
}

// ---------------------------------------------------------------------------
// Prepare systems
// ---------------------------------------------------------------------------

/// Builds the meshes, pipelines, uniform slots and bind groups once the shared
/// globals buffer (group 0) and the world textures (group 1) exist.
///
/// The globals guard is a hard requirement because the terrain's bind group
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
    let globals_layout = crate::render::globals_layout();
    let globals_entries = crate::render::globals_bind_group_entries(globals_buffer);
    let globals_bind_group = super::bind_group(
        device,
        &pipeline_cache,
        "forest_terrain_globals_bind_group",
        &globals_layout,
        &globals_entries,
    );

    // group(1): the world textures; group(2): the per-draw stage blocks.
    let terrain_textures_layout = terrain_texture_layout();
    let samplers = build_terrain_samplers(device);
    let snow_texture_descriptor = wgpu::TextureDescriptor {
        label: Some("forest_snow_compression"),
        size: wgpu::Extent3d {
            width: crate::snow::SNOW_MAP_SIZE as u32,
            height: crate::snow::SNOW_MAP_SIZE as u32,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    };
    let snow_texture = device.wgpu_device().create_texture(&snow_texture_descriptor);
    let snow_view = snow_texture.create_view(&Default::default());
    let snow_uniform = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("forest_snow_mapping"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let terrain_textures = terrain_texture_bind_group(
        device,
        &pipeline_cache,
        &terrain_textures_layout,
        textures,
        &samplers,
        &snow_view,
        &snow_uniform,
    );
    let terrain_stage_layout = stage_uniform_layout(
        "forest_terrain_stage_layout",
        std::mem::size_of::<TerrainStageUniforms>() as u64,
    );

    // terrain-vs + terrain-fs over a plain vec3 position, into the same three
    // colour targets the other G-buffer passes use.
    let terrain_vertex_buffers = vec![VertexBufferLayout {
        array_stride: TERRAIN_VERTEX_STRIDE,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: TERRAIN_VERTEX_ATTRIBUTES.to_vec(),
    }];
    let terrain_pipeline_layouts = vec![
        globals_layout.clone(),
        terrain_textures_layout.clone(),
        terrain_stage_layout.clone(),
    ];
    let terrain_pipeline = queue_gbuffer_pipeline(
        &pipeline_cache,
        "forest_terrain_pipeline",
        shaders.terrain_vs.clone(),
        shaders.terrain_fs.clone(),
        terrain_vertex_buffers,
        terrain_pipeline_layouts,
    );

    let habitat_pipeline_descriptor = RenderPipelineDescriptor {
        label: Some("forest_grass_habitat_pipeline".into()),
        layout: vec![globals_layout.clone(), terrain_textures_layout.clone(), terrain_stage_layout.clone()],
        immediate_size: 0,
        vertex: VertexState {
            shader: shaders.terrain_vs.clone(),
            shader_defs: vec![],
            entry_point: Some("vs_habitat".into()),
            buffers: vec![VertexBufferLayout {
                array_stride: TERRAIN_VERTEX_STRIDE,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: TERRAIN_VERTEX_ATTRIBUTES.to_vec(),
            }],
        },
        primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(FragmentState {
            shader: shaders.terrain_fs.clone(),
            shader_defs: vec![],
            entry_point: Some("fs_grass_habitat".into()),
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
            ],
        }),
        zero_initialize_workgroup_memory: false,
    };
    let habitat_pipeline = pipeline_cache.queue_render_pipeline(habitat_pipeline_descriptor);

    let ground_average_pipeline_descriptor = RenderPipelineDescriptor {
        label: Some("forest_grass_ground_average_pipeline".into()),
        layout: vec![grass_ground_average_layout()],
        immediate_size: 0,
        vertex: VertexState {
            shader: shaders.grass_ground_average.clone(),
            shader_defs: vec![],
            entry_point: Some("vs_main".into()),
            buffers: vec![],
        },
        primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(FragmentState {
            shader: shaders.grass_ground_average.clone(),
            shader_defs: vec![],
            entry_point: Some("fs_main".into()),
            targets: vec![Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba16Float,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        zero_initialize_workgroup_memory: false,
    };
    let grass_ground_average_pipeline =
        pipeline_cache.queue_render_pipeline(ground_average_pipeline_descriptor);

    let heightfield_pipeline_descriptor = RenderPipelineDescriptor {
        label: Some("forest_lighting_heightfield_pipeline".into()),
        layout: vec![globals_layout.clone(), terrain_textures_layout, terrain_stage_layout.clone()],
        immediate_size: 0,
        vertex: VertexState {
            shader: shaders.terrain_vs.clone(),
            shader_defs: vec![],
            entry_point: Some("vs_heightfield".into()),
            buffers: vec![],
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(FragmentState {
            shader: shaders.terrain_vs.clone(),
            shader_defs: vec![],
            entry_point: Some("fs_heightfield".into()),
            targets: vec![Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::R32Float,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        zero_initialize_workgroup_memory: false,
    };
    let heightfield_pipeline =
        pipeline_cache.queue_render_pipeline(heightfield_pipeline_descriptor);

    let highest_layout_entries = [
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(4),
            },
            count: None,
        },
    ];
    let highest_layout = BindGroupLayoutDescriptor::new(
        "forest_heightfield_highest_layout",
        &highest_layout_entries,
    );
    let compute_pipeline_descriptor = ComputePipelineDescriptor {
        label: Some("forest_heightfield_highest_pipeline".into()),
        layout: vec![highest_layout.clone()],
        immediate_size: 0,
        shader: shaders.heightfield_max.clone(),
        shader_defs: vec![],
        entry_point: Some("reduce_highest".into()),
        zero_initialize_workgroup_memory: false,
    };
    let highest_pipeline = pipeline_cache.queue_compute_pipeline(compute_pipeline_descriptor);
    let highest_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("forest_heightfield_highest"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let levels: [StageUniform; CLIP_LEVELS] = std::array::from_fn(|_| {
        StageUniform::new(
            device,
            &pipeline_cache,
            &terrain_stage_layout,
            "forest_terrain_stage_uniform",
            std::mem::size_of::<TerrainStageUniforms>() as u64,
        )
    });

    let terrain_resources = TerrainResources {
        globals: globals_bind_group,
        terrain_textures,
        snow_texture,
        snow_uniform,
        snow_revision: AtomicU64::new(u64::MAX),
        terrain_pipeline,
        heightfield_pipeline,
        highest_pipeline,
        highest_layout,
        highest_buffer,
        habitat_pipeline,
        grass_ground_average_pipeline,
        habitat_globals: StageUniform::new(
            device, &pipeline_cache, &globals_layout, "forest_habitat_globals",
            std::mem::size_of::<GlobalUniformsGpu>() as u64,
        ),
        shore_globals: StageUniform::new(
            device,
            &pipeline_cache,
            &globals_layout,
            "forest_shore_globals_uniform",
            std::mem::size_of::<GlobalUniformsGpu>() as u64,
        ),
        center_mesh: build_clip_mesh(device, false),
        ring_mesh: build_clip_mesh(device, true),
        levels,
    };
    *resources = Some(terrain_resources);

    // `samplers` and the standalone layouts are deliberately not stored: a
    // wgpu bind group keeps its bound resources alive, and the
    // RenderPipelineDescriptors queued above (retained by the PipelineCache)
    // keep the layouts alive for as long as the pipelines exist.
}

/// Reset the ground average the grass reads to the terrain pass's untouched
/// one, discarding any shade the vegetation pass multiplied into it.
pub fn copy_ground_average(encoder: &mut wgpu::CommandEncoder, gbuffer: &GbufferTargets) {
    fn whole(view: &wgpu::TextureView) -> wgpu::TexelCopyTextureInfo<'_> {
        wgpu::TexelCopyTextureInfo {
            texture: view.texture(),
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        }
    }
    encoder.copy_texture_to_texture(
        whole(&gbuffer.grass_ground_base_view),
        whole(&gbuffer.grass_ground_average_view),
        wgpu::Extent3d {
            width: GRASS_GROUND_AVERAGE_SIZE,
            height: GRASS_GROUND_AVERAGE_SIZE,
            depth_or_array_layers: 1,
        },
    );
}

/// `CreateGBuffer` plus the main loop's `rlViewport(0, 0, ssao.width,
/// ssao.height)`: (re)creates the colour targets and the depth texture
/// whenever the physical window size changes, along with the three filter
/// samplers the C++ sets on them.
pub(crate) fn resize_gbuffer(
    device: Res<RenderDevice>,
    pipeline_cache: Res<PipelineCache>,
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
        let position_texture_descriptor = wgpu::TextureDescriptor {
            label: Some("forest_gbuffer_position"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };
        let position_texture = device.wgpu_device().create_texture(&position_texture_descriptor);
        let normal_texture_descriptor = wgpu::TextureDescriptor {
            label: Some("forest_gbuffer_normal"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };
        let normal_texture = device.wgpu_device().create_texture(&normal_texture_descriptor);
        let albedo_texture_descriptor = wgpu::TextureDescriptor {
            label: Some("forest_gbuffer_albedo"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };
        let albedo_texture = device.wgpu_device().create_texture(&albedo_texture_descriptor);
        let depth_texture_descriptor = wgpu::TextureDescriptor {
            label: Some("forest_gbuffer_depth"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };
        let depth_texture = device.wgpu_device().create_texture(&depth_texture_descriptor);

        let grass_ground_albedo_descriptor = wgpu::TextureDescriptor {
            label: Some("forest_grass_ground_albedo"),
            size: wgpu::Extent3d {
                width: GRASS_HABITAT_SIZE, height: GRASS_HABITAT_SIZE, depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };
        let grass_ground_albedo_texture =
            device.wgpu_device().create_texture(&grass_ground_albedo_descriptor);
        let grass_ground_albedo_view = grass_ground_albedo_texture.create_view(&Default::default());
        let grass_ground_average_bind_group = super::bind_group(
            &device,
            &pipeline_cache,
            "forest_grass_ground_average_source",
            &grass_ground_average_layout(),
            &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&grass_ground_albedo_view),
            }],
        );
        let grass_ground_base_descriptor = wgpu::TextureDescriptor {
            label: Some("forest_grass_ground_base"),
            size: wgpu::Extent3d {
                width: GRASS_GROUND_AVERAGE_SIZE, height: GRASS_GROUND_AVERAGE_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        };
        let grass_ground_base_texture =
            device.wgpu_device().create_texture(&grass_ground_base_descriptor);
        let grass_ground_base_view = grass_ground_base_texture.create_view(&Default::default());
        let grass_ground_average_descriptor = wgpu::TextureDescriptor {
            label: Some("forest_grass_ground_average"),
            size: wgpu::Extent3d {
                width: GRASS_GROUND_AVERAGE_SIZE, height: GRASS_GROUND_AVERAGE_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        };
        let grass_ground_average_texture =
            device.wgpu_device().create_texture(&grass_ground_average_descriptor);
        let grass_ground_average_view =
            grass_ground_average_texture.create_view(&Default::default());

        let heightfield_texture_descriptor = wgpu::TextureDescriptor {
            label: Some("forest_lighting_heightfield"),
            size: wgpu::Extent3d {
                width: LIGHTING_HEIGHTFIELD_SIZE,
                height: LIGHTING_HEIGHTFIELD_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };
        let heightfield_texture =
            device.wgpu_device().create_texture(&heightfield_texture_descriptor);
        let heightfield_view =
            heightfield_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let shore_texture_descriptor = wgpu::TextureDescriptor {
            label: Some("forest_shore_heightfield"),
            size: wgpu::Extent3d {
                width: SHORE_HEIGHTFIELD_SIZE,
                height: SHORE_HEIGHTFIELD_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };
        let shore_texture = device.wgpu_device().create_texture(&shore_texture_descriptor);
        let shore_heightfield_view = shore_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let grass_habitat_texture_descriptor = wgpu::TextureDescriptor {
            label: Some("forest_grass_habitat"),
            size: wgpu::Extent3d {
                width: GRASS_HABITAT_SIZE, height: GRASS_HABITAT_SIZE, depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };
        let grass_habitat_texture =
            device.wgpu_device().create_texture(&grass_habitat_texture_descriptor);
        let grass_habitat_view = grass_habitat_texture.create_view(&Default::default());
        // Buffers are created zeroed: a span of zero, no capture yet, which
        // is what the new textures hold.
        let grass_habitat_mapping_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("forest_grass_habitat_mapping"),
            size: std::mem::size_of::<[f32; 4]>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let gbuffer_targets = GbufferTargets {
            position_view: position_texture.create_view(&wgpu::TextureViewDescriptor::default()),
            normal_view: normal_texture.create_view(&wgpu::TextureViewDescriptor::default()),
            albedo_view: albedo_texture.create_view(&wgpu::TextureViewDescriptor::default()),
            depth_view: depth_texture.create_view(&wgpu::TextureViewDescriptor::default()),
            heightfield_view,
            shore_heightfield_view,
            grass_habitat_view,
            grass_ground_albedo_view,
            grass_ground_base_view,
            grass_ground_average_view,
            grass_ground_average_bind_group,
            grass_habitat: HabitatCapture::default(),
            grass_habitat_mapping_buffer,
            lighting_heightfield_key: None,
            lighting_highest_ready: false,
            shore_heightfield_key: None,
            shore_heightfield_ready: false,
            width,
            height,
        };
        *gbuffer = Some(gbuffer_targets);
    }

}

/// Render-app registration for the terrain pass: the resize check first, then
/// the one-time resource build, both in the `Render` schedule's `Prepare` set
/// and after the shared globals buffer exists. The world-texture dependency is
/// covered by the guard inside [`prepare_terrain`].
pub fn register_terrain_systems(render_app: &mut bevy::app::SubApp) {
    let mut config = (resize_gbuffer, prepare_terrain).chain();
    config = config.in_set(bevy::render::RenderSystems::Prepare);
    config = config.after(crate::render::prepare_forest_globals);
    // Textures before the first frame rather than the second; the
    // guard in `prepare_terrain` still covers a reordering.
    config = config.after(crate::render::gpu_textures::prepare_gpu_textures);
    render_app.add_systems(bevy::render::Render, config);
}

#[cfg(test)]
mod habitat_cache_tests {
    use super::*;
    use crate::render::water_node::ExtractedWater;

    fn key_with(
        position: [f32; 3],
        edit: impl FnOnce(&mut GlobalUniformsGpu, &mut Option<ExtractedWater>, &mut u64, &mut (i64, i64)),
    ) -> HabitatKey {
        let mut globals = GlobalUniformsGpu::default();
        let mut water = Some(ExtractedWater::default());
        let (mut revision, mut lookup) = (0, (0, 0));
        edit(&mut globals, &mut water, &mut revision, &mut lookup);
        HabitatKey::new(position, lookup, &globals, water.as_ref(), revision)
    }

    fn key(position: [f32; 3]) -> HabitatKey {
        key_with(position, |_, _, _, _| {})
    }

    #[test]
    fn walking_inside_one_window_keeps_the_capture() {
        // The same 8 m window and the same 64 m clipmap cell, whatever the
        // player's height or the globals' camera say.
        assert_eq!(key([65.0, 20.0, 1.0]), key([71.9, 55.0, 7.9]));
        assert_eq!(key([0.0, 3.0, 0.0]), key([7.99, 3.0, 7.99]));
    }

    #[test]
    fn crossing_into_the_next_window_retakes_it() {
        assert_ne!(key([71.9, 20.0, 1.0]), key([72.1, 20.0, 1.0]));
        assert_ne!(key([65.0, 20.0, 7.9]), key([65.0, 20.0, 8.1]));
        // West and south of the origin too: the window floors, never truncates.
        assert_ne!(key([-0.1, 20.0, 0.0]), key([0.1, 20.0, 0.0]));
    }

    #[test]
    fn the_clipmap_origin_is_part_of_the_key() {
        // The clipmap recentres every 64 m, which moves the LOD morph the
        // capture's triangles carry.
        let before = key([31.9, 0.0, 0.0]);
        let after = key([32.1, 0.0, 0.0]);
        assert_ne!(before.clip_origin, after.clip_origin);
        assert_ne!(before, after);
    }

    #[test]
    fn terrain_and_settings_changes_retake_it() {
        let base = key([10.0, 0.0, 10.0]);
        assert_ne!(base, key_with([10.0, 0.0, 10.0], |_, _, revision, _| *revision = 1));
        assert_ne!(base, key_with([10.0, 0.0, 10.0], |_, _, _, lookup| *lookup = (1, 0)));
        for slot in [1usize, 3] {
            assert_ne!(
                base,
                key_with([10.0, 0.0, 10.0], |globals, _, _, _| globals.settings_a[slot] += 0.5),
                "settings_a[{slot}]"
            );
        }
        for slot in [0usize, 2, 3] {
            assert_ne!(
                base,
                key_with([10.0, 0.0, 10.0], |globals, _, _, _| globals.settings_b[slot] += 1.0),
                "settings_b[{slot}]"
            );
        }
        assert_ne!(base, key_with([10.0, 0.0, 10.0], |_, water, _, _| *water = None));
        assert_ne!(
            base,
            key_with([10.0, 0.0, 10.0], |_, water, _, _| {
                water.as_mut().unwrap().settings.sea_state_amplitude += 0.1;
            })
        );
        assert_ne!(
            base,
            key_with([10.0, 0.0, 10.0], |_, water, _, _| {
                water.as_mut().unwrap().settings.flat_surface = true;
            })
        );
    }

    #[test]
    fn settings_the_capture_never_reads_do_not_retake_it() {
        // Sun, fog and sparkle move every few frames; none of them is in a capture.
        let base = key([10.0, 0.0, 10.0]);
        let moved = key_with([10.0, 0.0, 10.0], |globals, _, _, _| {
            globals.settings_a[0] = 3.0;
            globals.settings_b[1] = 2.0;
            globals.sun_direction = [0.1, -0.9, 0.2, 0.0];
            globals.camera_position = [10.0, 99.0, 10.0, 0.0];
            globals.params[0] = 0.5;
        });
        assert_eq!(base, moved);
    }

    #[test]
    fn the_capture_does_not_see_where_the_player_stands() {
        let a = GlobalUniformsGpu {
            camera_position: [66.0, 20.0, 3.0, 0.0],
            ..Default::default()
        };
        let b = GlobalUniformsGpu {
            camera_position: [71.0, 55.0, 6.0, 0.0],
            ..a
        };
        let mapping = grass_habitat_mapping([66.0, 20.0, 3.0]);
        assert_eq!(mapping, grass_habitat_mapping([71.0, 55.0, 6.0]));
        let capture = |globals: &GlobalUniformsGpu| {
            bytemuck::bytes_of(&habitat_capture_globals(globals, mapping, Some(0.4))).to_vec()
        };
        assert_eq!(capture(&a), capture(&b));
    }

    /// The water samples the canopy laid over the capture and places it by
    /// the window uniform the capture's pass writes. In every frame of a
    /// flight, the frames it crosses into a new window among them, and while
    /// the first capture waits on its pipelines, that window must be the one
    /// of the capture whose texels the water samples: none before there is a
    /// capture, never a pending one's, never the last one's.
    #[test]
    fn the_water_places_the_canopy_by_the_window_of_the_capture_it_samples() {
        let mut capture = HabitatCapture::default();
        // The uniform as the GPU holds it: created zeroed, then whatever the
        // capture's pass queues.
        let mut window = [0.0f32; 4];
        // The window of the capture the canopy texture holds, if any.
        let mut texels: Option<[f32; 4]> = None;
        let mut generations = Vec::new();
        let mut copied_in_prepare_wrong = 0;
        // East over the forest pond at the flying speed, 60 frames a second.
        for frame in 0..240 {
            let position = [-760.0 + 38.0 * frame as f32 / 60.0, 51.0, 992.0];
            // What a window copied into the water's own block in Prepare,
            // before the frame's capture, would hold.
            let copied_in_prepare = if capture.ready { capture.mapping } else { [0.0; 4] };
            // The terrain pass. The ground average's pipeline compiles for
            // the first frames; until it has, a capture is begun and left.
            let key = key(position);
            if !capture.holds(key) {
                capture.begin();
                if frame >= 3 {
                    let mapping = grass_habitat_mapping(position);
                    texels = Some(mapping);
                    capture.complete(key, mapping, |queued| window = *queued);
                    generations.push(capture.generation);
                }
            }
            // The water pass.
            assert_eq!(window, texels.unwrap_or([0.0; 4]), "frame {frame}");
            copied_in_prepare_wrong += usize::from(copied_in_prepare != window);
        }
        // One capture per 8 m window flown through, each a new one.
        assert_eq!(generations.len(), 19);
        assert!(generations.windows(2).all(|pair| pair[1] > pair[0]));
        // The window copied before the capture was wrong in every frame that
        // took one: a crown edge flashing 8 m aside five times a second.
        assert_eq!(copied_in_prepare_wrong, generations.len());
    }

    #[test]
    fn the_capture_window_maps_world_to_texture_like_the_shader_expects() {
        let mapping = grass_habitat_mapping([66.0, 20.0, 3.0]);
        let globals = habitat_capture_globals(&GlobalUniformsGpu::default(), mapping, None);
        let p = globals.projection;
        // World (x, z) -> clip: x_clip = scale * (x - cx), y_clip = -scale * (z - cz).
        let clip = |x: f32, z: f32| (p[0] * x + p[12], p[9] * z + p[13]);
        let (cx, cz) = (mapping[0], mapping[1]);
        assert!((clip(cx, cz).0).abs() < 1e-5 && (clip(cx, cz).1).abs() < 1e-5);
        assert!((clip(cx + GRASS_HABITAT_SPAN / 2.0, cz).0 - 1.0).abs() < 1e-5);
        assert!((clip(cx, cz + GRASS_HABITAT_SPAN / 2.0).1 + 1.0).abs() < 1e-5);
        assert_eq!(globals.camera_position[3], 0.0);
        assert_eq!(
            habitat_capture_globals(&GlobalUniformsGpu::default(), mapping, Some(0.7)).camera_position[3],
            0.7
        );
    }

    /// The vertex stage hands each vertex's run up its shore to the fragment
    /// stage, which measures it again exactly near the water; the two must
    /// measure alike, or the bank band seams where the exact lookup hands
    /// over to the vertices.
    #[test]
    fn both_terrain_stages_measure_the_shore_alike() {
        let common = include_str!("../../assets/shaders/river-shore.wgslinc").trim();
        for source in [
            include_str!("../../assets/shaders/terrain-vs.wgsl"),
            include_str!("../../assets/shaders/terrain-fs.wgsl"),
        ] {
            assert!(source.contains(common), "shore run diverged");
        }
    }

    /// The capture is kept across frames, so it is only right while every input
    /// the shaders read is in `HabitatKey`. This lists the globals the terrain
    /// shaders read; a new one means the key (and this list) must be revisited.
    #[test]
    fn the_terrain_shaders_read_only_globals_the_key_accounts_for() {
        let fragment = include_str!("../../assets/shaders/terrain-fs.wgsl");
        let vertex = include_str!("../../assets/shaders/terrain-vs.wgsl");
        let uses = |source: &str, prefix: &str| -> std::collections::BTreeSet<String> {
            let mut found = std::collections::BTreeSet::new();
            for (at, _) in source.match_indices(prefix) {
                let rest = &source[at + prefix.len()..];
                let end = rest
                    .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                    .unwrap_or(rest.len());
                found.insert(rest[..end].to_string());
            }
            found
        };
        let set = |names: &[&str]| -> std::collections::BTreeSet<String> {
            names.iter().map(|name| name.to_string()).collect()
        };
        // Fragment: view-space outputs (identity in a capture), the camera's
        // w (crest clearance, in the key through the sea state), the sun and
        // camera position for the snow glints a capture skips, and the shading
        // settings the key holds.
        assert_eq!(
            uses(fragment, "globals."),
            set(&["camera_position", "settings_a", "settings_b", "sun_direction", "view"]),
            "terrain-fs.wgsl reads a global HabitatKey does not know about"
        );
        // Of settings_a only texture scale (y) and variant scale (w) are read.
        assert_eq!(uses(fragment, "globals.settings_a."), set(&["w", "y"]));
        // Vertex: the camera enters only through materialStep, which a capture
        // takes from the middle of its window; heightfield is the separate
        // lighting-map pass.
        assert_eq!(
            uses(vertex, "globals."),
            set(&["camera_position", "heightfield", "projection", "view"]),
            "terrain-vs.wgsl reads a global HabitatKey does not know about"
        );
    }
}

#[cfg(test)]
mod heightfield_cache_tests {
    use super::*;
    use std::collections::BTreeSet;

    fn shore_key(position: [f32; 3]) -> HeightfieldKey {
        let mapping = shore_heightfield_mapping([position[0], position[1], position[2], 0.0]);
        HeightfieldKey::shore(mapping, (0, 0), 0)
    }

    #[test]
    fn walking_inside_a_shore_snap_keeps_its_capture() {
        // The corners and middle of the snap square [64, 72) x [-8, 0), at
        // any height.
        let positions = [
            [64.0, 3.0, -8.0],
            [71.99, 40.0, -0.01],
            [64.0, 9.0, -0.01],
            [71.99, 3.0, -8.0],
            [68.0, 120.0, -4.0],
        ];
        for position in positions {
            assert_eq!(shore_key(position), shore_key(positions[0]), "{position:?}");
            // What lets the key leave the player out: wherever they stand in
            // the snap, erosion is at full strength out to the window's far
            // corners.
            let [x, z, span, _] =
                shore_heightfield_mapping([position[0], position[1], position[2], 0.0]);
            for (dx, dz) in [(-0.5, -0.5), (0.5, -0.5), (-0.5, 0.5), (0.5, 0.5)] {
                let fade = crate::erosion::erosion_visibility(
                    x + dx * span,
                    z + dz * span,
                    [position[0], position[2]],
                );
                assert_eq!(fade, 1.0, "{position:?}");
            }
        }
    }

    #[test]
    fn crossing_into_the_next_shore_snap_retakes_it() {
        assert_ne!(shore_key([71.9, 20.0, 1.0]), shore_key([72.1, 20.0, 1.0]));
        assert_ne!(shore_key([65.0, 20.0, 7.9]), shore_key([65.0, 20.0, 8.1]));
        // West and south of the origin too: the window floors, never truncates.
        assert_ne!(shore_key([-0.1, 20.0, 0.0]), shore_key([0.1, 20.0, 0.0]));
    }

    #[test]
    fn any_step_retakes_the_lighting_map_but_not_the_shore() {
        // A centimetre, inside one lighting window and one shore snap.
        let (from, to) = ([10.0, 20.0], [10.01, 20.0]);
        let mapping = [0.0, 0.0, LIGHTING_HEIGHTFIELD_SPAN, 12.0];
        assert_ne!(
            HeightfieldKey::lighting(mapping, from, (0, 0), 0),
            HeightfieldKey::lighting(mapping, to, (0, 0), 0)
        );
        // Because the step moves the erosion drawn far out in the map: 1.35 km
        // away, well inside the 12 km window, the fade is partway down its ramp.
        let far = [from[0] + 1350.0, from[1]];
        assert_ne!(
            crate::erosion::erosion_visibility(far[0], far[1], from),
            crate::erosion::erosion_visibility(far[0], far[1], to)
        );
        assert_eq!(shore_key([from[0], 5.0, from[1]]), shore_key([to[0], 5.0, to[1]]));
    }

    #[test]
    fn terrain_changes_retake_both_maps() {
        // A streamed tile, a reveal step or a new river network moves the
        // revision; the lookup window recentring moves its corner.
        let mapping = [0.0, 0.0, LIGHTING_HEIGHTFIELD_SPAN, 12.0];
        let lighting =
            |revision, lookup| HeightfieldKey::lighting(mapping, [10.0, 20.0], lookup, revision);
        let shore_mapping = shore_heightfield_mapping([10.0, 5.0, 20.0, 0.0]);
        let shore = |revision, lookup| HeightfieldKey::shore(shore_mapping, lookup, revision);
        assert_ne!(lighting(0, (0, 0)), lighting(1, (0, 0)));
        assert_ne!(lighting(0, (0, 0)), lighting(0, (1, 0)));
        assert_ne!(shore(0, (0, 0)), shore(1, (0, 0)));
        assert_ne!(shore(0, (0, 0)), shore(0, (0, -1)));
    }

    /// Every function a block calls, however deeply nested the call.
    fn calls(block: &naga::Block, found: &mut Vec<naga::Handle<naga::Function>>) {
        for statement in block.iter() {
            match statement {
                naga::Statement::Call { function, .. } => found.push(*function),
                naga::Statement::Block(inner) => calls(inner, found),
                naga::Statement::If { accept, reject, .. } => {
                    calls(accept, found);
                    calls(reject, found);
                }
                naga::Statement::Switch { cases, .. } => {
                    for case in cases {
                        calls(&case.body, found);
                    }
                }
                naga::Statement::Loop { body, continuing, .. } => {
                    calls(body, found);
                    calls(continuing, found);
                }
                _ => {}
            }
        }
    }

    /// A capture is kept across frames, so it is only right while every input
    /// its shader reads is in `HeightfieldKey` or fixed from startup. This
    /// follows the capture's entry points through every function they call and
    /// lists what is read; a new input means the key (and this list) must be
    /// revisited.
    #[test]
    fn the_heightfield_shader_reads_only_inputs_the_key_accounts_for() {
        let source = include_str!("../../assets/shaders/terrain-vs.wgsl");
        let module = naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|error| panic!("{}", error.emit_to_string(source)));
        let heightfield_entries = module
            .entry_points
            .iter()
            .filter(|entry| matches!(entry.name.as_str(), "vs_heightfield" | "fs_heightfield"));
        let mut pending: Vec<&naga::Function> = heightfield_entries
            .map(|entry| &entry.function)
            .collect();
        assert_eq!(pending.len(), 2, "the capture's entry points moved");
        let mut visited = std::collections::HashSet::new();
        // Bound resources by name, and uniform members as `block.member`.
        let mut read = BTreeSet::new();
        let mut members = BTreeSet::new();
        while let Some(function) = pending.pop() {
            let mut called = Vec::new();
            calls(&function.body, &mut called);
            for handle in called {
                if visited.insert(handle) {
                    pending.push(&module.functions[handle]);
                }
            }
            for (_, expression) in function.expressions.iter() {
                match *expression {
                    naga::Expression::GlobalVariable(variable) => {
                        read.insert(module.global_variables[variable].name.clone().unwrap());
                    }
                    naga::Expression::AccessIndex { base, index } => {
                        let naga::Expression::GlobalVariable(variable) = function.expressions[base]
                        else {
                            continue;
                        };
                        let variable = &module.global_variables[variable];
                        if let naga::TypeInner::Struct { members: fields, .. } =
                            &module.types[variable.ty].inner
                        {
                            members.insert(format!(
                                "{}.{}",
                                variable.name.as_deref().unwrap(),
                                fields[index as usize].name.as_deref().unwrap()
                            ));
                        }
                    }
                    _ => {}
                }
            }
        }
        let set = |names: &[&str]| -> BTreeSet<String> {
            names.iter().map(|name| name.to_string()).collect()
        };
        let uses = |block: &str| -> BTreeSet<String> {
            let prefix = format!("{block}.");
            let used_members = members
                .iter()
                .filter_map(|member| member.strip_prefix(&prefix));
            used_members.map(str::to_string).collect()
        };

        // The two uniform blocks below; the base noise and blend mask,
        // uploaded once; the surface atlas, tile lookup and river buffers,
        // which all move TerrainRevision; and their samplers. Not the snow,
        // so a fresh trail leaves both maps be.
        assert_eq!(
            read,
            set(&[
                "globals", "river_grid", "river_segments", "stage", "tex0", "tex0_sampler",
                "tex1", "tex1_sampler", "tex3", "tex4", "tex4_sampler",
            ]),
            "the heightfield capture reads a resource HeightfieldKey does not know about"
        );
        // Of the globals, only the window itself.
        assert_eq!(uses("globals"), set(&["heightfield"]));
        // Every stage field read is a constant of TerrainStageUniforms::build
        // except the lookup's corner and the fade's centre, which the key holds.
        assert_eq!(
            uses("stage"),
            set(&[
                "erosion_atlas_gutter", "erosion_atlas_pitch", "erosion_atlas_size",
                "erosion_footprint_size", "erosion_lookup_min_tile", "erosion_lookup_size",
                "erosion_output_resolution", "erosion_tile_stride", "erosion_visibility_center",
                "erosion_visibility_full_radius", "erosion_visibility_zero_radius",
                "land_profile_curve", "land_profile_peak", "land_profile_reference",
                "landform_horizontal_scale", "landform_vertical_scale", "noise_period",
                "ocean_profile_curve", "ocean_profile_depth", "ocean_profile_reference",
                "sea_level", "waterline_clearance", "waterline_clearance_decay",
                "waterline_clearance_scale", "waterline_push_land", "waterline_push_scale",
                "waterline_push_sea",
            ]),
            "the heightfield capture reads a stage field HeightfieldKey does not know about"
        );
    }
}
