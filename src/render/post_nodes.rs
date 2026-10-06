//! The deferred post passes: SSAO -> bilateral blur -> composite -> FXAA.
//!
//! Ports `RenderSSAO`, `CompositeScene` and `PresentFxaa` from the C++ main
//! loop, plus the half-res target bookkeeping `ResizeSSAO` performs. The
//! schedule order wired up in `render/mod.rs` (Ssao -> Blur -> Composite ->
//! Fxaa) is the C++ call order, and every pass is the same fullscreen
//! triangle drawn with blending disabled: these passes write numerical state
//! rather than translucent colour, so standard alpha blending would corrupt
//! the channels (see `RenderPass` in the C++).

use crate::render::gpu_textures::GpuWorldTexturesOption;
use crate::render::terrain_node::TerrainNodeState;
use crate::render::{
    globals_layout, BlurStageUniforms, CompositeStageUniforms, ExtractedForestView,
    ForestGlobals, ForestShaderHandles, GlobalUniformsGpu, SsaoStageUniforms,
};
use super::vegetation_shadows::{ShadowUniform, VegetationShadowMaps};
use bevy::app::SubApp;
use bevy::asset::Handle;
use bevy::ecs::schedule::IntoScheduleConfigs;
use bevy::prelude::{Res, Resource, World};
use bevy::render::render_phase::TrackedRenderPass;
use bevy::render::render_resource::{
    BindGroup, BindGroupEntry, BindGroupLayoutDescriptor, CachedRenderPipelineId, ColorTargetState,
    ColorWrites, FragmentState, MultisampleState, PipelineCache, PrimitiveState, RenderPipeline,
    RenderPipelineDescriptor, VertexState,
};
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue, ViewQuery};
use bevy::render::view::ViewTarget;
use bevy::render::{Render, RenderSystems};
use bevy::shader::Shader;
use std::collections::HashMap;
use std::sync::Mutex;

// ---------------------------------------------------------------------------
// Prepared GPU state
// ---------------------------------------------------------------------------

/// The half-res AO and blur targets `ResizeSSAO` recreates on every window
/// resize. Only the views are kept: a wgpu texture view holds its texture
/// alive, and both targets are only ever used through a view (as a render
/// attachment and as a sampled texture).
struct HalfResTargets {
    width: u32,
    height: u32,
    ao_view: wgpu::TextureView,
    blur_view: wgpu::TextureView,
    atmosphere_view: wgpu::TextureView,
}

/// The three samplers the C++ passes rely on: POINT/CLAMP for the G-buffer
/// position and normal (`CreateGBuffer` — their texel-exact values feed the
/// SSAO reconstruction), POINT/REPEAT for the 4x4 rotation noise
/// (`CreateSSAONoise`), and BILINEAR/CLAMP for every texture GL filtered
/// linearly (albedo, the AO/blur targets and the LDR target).
struct PostSamplers {
    point_clamp: wgpu::Sampler,
    point_repeat: wgpu::Sampler,
    linear_clamp: wgpu::Sampler,
}

/// The group-1 layouts of the four passes. Every pass binds its textures at
/// binding N with the matching sampler at binding N + 8, the convention the
/// post WGSL files use.
#[derive(Clone)]
struct PostLayouts {
    ssao: BindGroupLayoutDescriptor,
    blur: BindGroupLayoutDescriptor,
    composite: BindGroupLayoutDescriptor,
    atmosphere: BindGroupLayoutDescriptor,
    fxaa: BindGroupLayoutDescriptor,
}

/// One pass's group-2 stage uniforms: the layout, the buffer and the bind
/// group wrapping its whole range.
struct Stage {
    layout: BindGroupLayoutDescriptor,
    buffer: wgpu::Buffer,
    group: BindGroup,
}

/// A second copy of the shared globals whose viewport describes the half-res
/// AO target instead of the window.
struct HalfResGlobals {
    buffer: wgpu::Buffer,
    group: BindGroup,
}

#[derive(Default)]
struct SsaoNodeInner {
    targets: Option<HalfResTargets>,
    /// The G-buffer size the group-1 bind groups were built for; anything else
    /// (including a G-buffer rebuilt at a different size) rebuilds them.
    bound_size: Option<(u32, u32)>,
    globals_layout: Option<BindGroupLayoutDescriptor>,
    globals_group: Option<BindGroup>,
    half_res_globals: Option<HalfResGlobals>,
    samplers: Option<PostSamplers>,
    layouts: Option<PostLayouts>,
    ssao_stage: Option<Stage>,
    blur_stage: Option<Stage>,
    composite_stage: Option<Stage>,
    ssao_group: Option<BindGroup>,
    blur_group: Option<BindGroup>,
    composite_group: Option<BindGroup>,
    atmosphere_group: Option<BindGroup>,
    atmosphere_pipeline: Option<CachedRenderPipelineId>,
    ssao_pipeline: Option<CachedRenderPipelineId>,
    blur_pipeline: Option<CachedRenderPipelineId>,
    /// The composite and FXAA pipelines are keyed by the main texture's format
    /// (`ViewTarget::main_texture_format()`), which bevy picks per camera
    /// configuration: an LDR view target is an sRGB-format texture
    /// (Rgba8UnormSrgb in bevy's default configuration, while the swapchain is
    /// Bgra8UnormSrgb), so a pipeline built for one format is invalid for the
    /// other and the key must be the runtime format, never a constant.
    composite_pipelines: HashMap<wgpu::TextureFormat, CachedRenderPipelineId>,
    fxaa_pipelines: HashMap<wgpu::TextureFormat, CachedRenderPipelineId>,
}

/// Shared, interior-mutable state for the SSAO, blur, composite and FXAA
/// passes; the passes and the prepare systems run in different schedules, so
/// the CPU-side resources live behind one mutex.
#[derive(Resource)]
pub struct SsaoNodeState {
    inner: Mutex<SsaoNodeInner>,
}

impl Default for SsaoNodeState {
    fn default() -> Self {
        Self {
            inner: Mutex::new(SsaoNodeInner::default()),
        }
    }
}

// ---------------------------------------------------------------------------
// Prepare systems
// ---------------------------------------------------------------------------

/// `ResizeSSAO`: recreates the half-res AO and blur targets whenever the window
/// size changes, at `max(1, width / 2) x max(1, height / 2)` like the C++.
fn resize_ssao_targets(
    state: Res<SsaoNodeState>,
    view: Res<ExtractedForestView>,
    device: Res<RenderDevice>,
) {
    let Ok(mut inner) = state.inner.lock() else {
        return;
    };
    let width = view.physical_width.max(1);
    let height = view.physical_height.max(1);
    let ao_width = (width / 2).max(1);
    let ao_height = (height / 2).max(1);
    if inner
        .targets
        .as_ref()
        .is_some_and(|targets| targets.width == ao_width && targets.height == ao_height)
    {
        return;
    }
    let ao_targets = HalfResTargets {
        width: ao_width,
        height: ao_height,
        ao_view: create_half_res_target(&device, "forest_ssao_ao_target", ao_width, ao_height, wgpu::TextureFormat::Rgba8Unorm),
        blur_view: create_half_res_target(&device, "forest_ssao_blur_target", ao_width, ao_height, wgpu::TextureFormat::Rgba8Unorm),
        atmosphere_view: create_half_res_target(&device, "forest_atmosphere_target", ao_width, ao_height, wgpu::TextureFormat::Rgba16Float),
    };
    inner.targets = Some(ao_targets);
    // The group-1 bind groups read these views, so the next prepare rebuilds
    // them.
    inner.bound_size = None;
}

/// Builds the bind groups and uniform buffers the four post passes draw with,
/// and queues the two format-independent pipelines. Everything is skipped (and
/// retried next frame) until the G-buffer, the shared globals buffer and the
/// world textures exist.
fn prepare_post_pipelines(
    state: Res<SsaoNodeState>,
    view: Res<ExtractedForestView>,
    globals: Res<ForestGlobals>,
    terrain: Res<TerrainNodeState>,
    world_textures: Res<GpuWorldTexturesOption>,
    shaders: Res<ForestShaderHandles>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    pipeline_cache: Res<PipelineCache>,
    vegetation_shadows: Res<VegetationShadowMaps>,
) {
    let Ok(mut inner) = state.inner.lock() else {
        return;
    };
    // uScreenSize is the AO target size, NOT the window size.
    let Some((ao_width, ao_height)) = inner
        .targets
        .as_ref()
        .map(|targets| (targets.width as f32, targets.height as f32))
    else {
        return;
    };
    let Some(globals_buffer) = globals.buffer.as_ref() else {
        return;
    };

    if inner.globals_layout.is_none() {
        inner.globals_layout = Some(globals_layout());
    }
    let Some(globals_group_layout) = inner.globals_layout.clone() else {
        return;
    };

    // Group 0 for the full-resolution passes: the shared globals buffer.
    if inner.globals_group.is_none() {
        inner.globals_group = Some(super::bind_group(
            &device,
            &pipeline_cache,
            "forest_post_globals_group",
            &globals_group_layout,
            &[BindGroupEntry {
                binding: 0,
                resource: globals_buffer.as_entire_binding(),
            }],
        ));
    }

    // Group 0 for SSAO and blur, whose targets are half the window: the same
    // globals with the viewport rewritten to the AO target. Both shaders derive
    // their uv as `position.xy * globals.viewport.zw` while rendering into a
    // target half the window's size, and the blur steps neighbours by
    // 2.0 / full_size — one AO texel in AO-target uv space — so the shared
    // window viewport would confine both passes to the top-left quarter of the
    // G-buffer. The C++ read fragTexCoord across the whole AO target with
    // uScreenSize set to the AO size, and this copy reproduces exactly that.
    let mut ao_globals = globals.globals;
    ao_globals.viewport = [ao_width, ao_height, 1.0 / ao_width, 1.0 / ao_height];
    if inner.half_res_globals.is_none() {
        let buffer_descriptor = wgpu::BufferDescriptor {
            label: Some("forest_half_res_globals"),
            size: std::mem::size_of::<GlobalUniformsGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        };
        let buffer = device.wgpu_device().create_buffer(&buffer_descriptor);
        let group = super::bind_group(
            &device,
            &pipeline_cache,
            "forest_half_res_globals_group",
            &globals_group_layout,
            &[BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        );
        inner.half_res_globals = Some(HalfResGlobals { buffer, group });
    }
    if let Some(half_res) = inner.half_res_globals.as_ref() {
        queue.write_buffer(&half_res.buffer, 0, bytemuck::bytes_of(&ao_globals));
    }

    // The G-buffer and the world textures are created by their own prepare
    // systems; until they exist there is nothing to bind.
    let Ok(gbuffer_guard) = terrain.gbuffer.lock() else {
        return;
    };
    let Some(gbuffer) = gbuffer_guard.as_ref() else {
        return;
    };
    let Some(world_textures) = world_textures.0.as_ref() else {
        return;
    };
    let Some(shadows) = vegetation_shadows.targets.as_ref() else {
        return;
    };

    if inner.samplers.is_none() {
        let wgpu_device = device.wgpu_device();
        let post_samplers = PostSamplers {
            point_clamp: wgpu_device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("forest_point_clamp_sampler"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Nearest,
                min_filter: wgpu::FilterMode::Nearest,
                mipmap_filter: wgpu::MipmapFilterMode::Nearest,
                ..Default::default()
            }),
            // TEXTURE_WRAP_REPEAT for the 4x4 rotation noise, which is why the
            // ssao shader can tile it with noiseScale = screenSize / 4.
            point_repeat: wgpu_device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("forest_point_repeat_sampler"),
                address_mode_u: wgpu::AddressMode::Repeat,
                address_mode_v: wgpu::AddressMode::Repeat,
                address_mode_w: wgpu::AddressMode::Repeat,
                mag_filter: wgpu::FilterMode::Nearest,
                min_filter: wgpu::FilterMode::Nearest,
                mipmap_filter: wgpu::MipmapFilterMode::Nearest,
                ..Default::default()
            }),
            // FXAA blends taps at sub-texel offsets, which needs the hardware's
            // linear filtering; point sampling here would quantise its edge
            // walks.
            linear_clamp: wgpu_device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("forest_linear_clamp_sampler"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::MipmapFilterMode::Linear,
                ..Default::default()
            }),
        };
        inner.samplers = Some(post_samplers);
    }

    if inner.layouts.is_none() {
        let post_layouts = PostLayouts {
            // ssao: position, normal, rotation noise (all point-filtered).
            ssao: screen_group_layout("forest_ssao_group_layout", &[false, false, false]),
            // blur: raw SSAO (bilinear), position, normal.
            blur: screen_group_layout("forest_blur_group_layout", &[true, false, false]),
            // composite: position, normal, albedo, blurred SSAO.
            composite: with_vegetation_shadows(screen_group_layout(
                "forest_composite_group_layout",
                &[false, false, true, true, true, true],
            )),
            atmosphere: screen_group_layout(
                "forest_atmosphere_group_layout",
                &[false, false, true, true, true],
            ),
            // fxaa: the composited LDR frame (bilinear).
            fxaa: screen_group_layout("forest_fxaa_group_layout", &[true]),
        };
        inner.layouts = Some(post_layouts);
    }

    // The group-1 bind groups hold the G-buffer and target views, so they are
    // rebuilt whenever the G-buffer's size changes.
    let gbuffer_size = (gbuffer.width, gbuffer.height);
    if inner.bound_size != Some(gbuffer_size) {
        let (Some(layouts), Some(samplers), Some(targets)) = (
            inner.layouts.as_ref(),
            inner.samplers.as_ref(),
            inner.targets.as_ref(),
        ) else {
            return;
        };
        let position: &wgpu::TextureView = &gbuffer.position_view;
        let normal: &wgpu::TextureView = &gbuffer.normal_view;
        let albedo: &wgpu::TextureView = &gbuffer.albedo_view;
        let noise: &wgpu::TextureView = &world_textures.ssao_noise_view;
        let ao: &wgpu::TextureView = &targets.ao_view;
        let blurred: &wgpu::TextureView = &targets.blur_view;

        let ssao_group = super::bind_group(
            &device,
            &pipeline_cache,
            "forest_ssao_group",
            &layouts.ssao,
            &[
                texture_entry(0, position),
                texture_entry(1, normal),
                texture_entry(2, noise),
                sampler_entry(8, &samplers.point_clamp),
                sampler_entry(9, &samplers.point_clamp),
                sampler_entry(10, &samplers.point_repeat),
            ],
        );
        let blur_group = super::bind_group(
            &device,
            &pipeline_cache,
            "forest_blur_group",
            &layouts.blur,
            &[
                texture_entry(0, ao),
                texture_entry(1, position),
                texture_entry(2, normal),
                sampler_entry(8, &samplers.linear_clamp),
                sampler_entry(9, &samplers.point_clamp),
                sampler_entry(10, &samplers.point_clamp),
            ],
        );
        let composite_entries = [
            texture_entry(0, position),
            texture_entry(1, normal),
            texture_entry(2, albedo),
            texture_entry(3, blurred),
            texture_entry(4, &gbuffer.heightfield_view),
            texture_entry(5, &targets.atmosphere_view),
            texture_entry(6, &shadows.cascade_views[0]),
            BindGroupEntry {
                binding: 7,
                resource: shadows.uniform.as_entire_binding(),
            },
            sampler_entry(8, &samplers.point_clamp),
            sampler_entry(9, &samplers.point_clamp),
            sampler_entry(10, &samplers.linear_clamp),
            sampler_entry(11, &samplers.linear_clamp),
            sampler_entry(12, &samplers.linear_clamp),
            sampler_entry(13, &samplers.linear_clamp),
            sampler_entry(14, &shadows.sampler),
            texture_entry(15, &shadows.cascade_views[1]),
            texture_entry(16, &shadows.cascade_views[2]),
        ];
        let composite_group = super::bind_group(
            &device,
            &pipeline_cache,
            "forest_composite_group",
            &layouts.composite,
            &composite_entries,
        );

        // Keep the atmosphere output out of this bind group: a texture cannot
        // be sampled and used as an attachment in the same render pass.
        let atmosphere_group = super::bind_group(
            &device,
            &pipeline_cache,
            "forest_atmosphere_group",
            &layouts.atmosphere,
            &[
                texture_entry(0, position),
                texture_entry(1, normal),
                texture_entry(2, albedo),
                texture_entry(3, blurred),
                texture_entry(4, &gbuffer.heightfield_view),
                sampler_entry(8, &samplers.point_clamp),
                sampler_entry(9, &samplers.point_clamp),
                sampler_entry(10, &samplers.linear_clamp),
                sampler_entry(11, &samplers.linear_clamp),
                sampler_entry(12, &samplers.linear_clamp),
            ],
        );

        inner.ssao_group = Some(ssao_group);
        inner.blur_group = Some(blur_group);
        inner.composite_group = Some(composite_group);
        inner.atmosphere_group = Some(atmosphere_group);
        inner.bound_size = Some(gbuffer_size);
    }

    if inner.ssao_stage.is_none() {
        inner.ssao_stage = Some(create_stage(
            &device,
            &pipeline_cache,
            "forest_ssao_stage",
            std::mem::size_of::<SsaoStageUniforms>() as u64,
        ));
    }
    if inner.blur_stage.is_none() {
        inner.blur_stage = Some(create_stage(
            &device,
            &pipeline_cache,
            "forest_blur_stage",
            std::mem::size_of::<BlurStageUniforms>() as u64,
        ));
    }
    if inner.composite_stage.is_none() {
        inner.composite_stage = Some(create_stage(
            &device,
            &pipeline_cache,
            "forest_composite_stage",
            std::mem::size_of::<CompositeStageUniforms>() as u64,
        ));
    }

    if view.settings.ssao_enabled && let Some(stage) = inner.ssao_stage.as_ref() {
        let uniforms = SsaoStageUniforms {
            screen_size: [ao_width, ao_height],
            radius: view.settings.ao_radius,
            bias: view.settings.ao_bias,
            power: view.settings.ao_power,
            _end_pad: [0.0; 3],
        };
        queue.write_buffer(&stage.buffer, 0, bytemuck::bytes_of(&uniforms));
    }

    let full_width = view.physical_width.max(1) as f32;
    let full_height = view.physical_height.max(1) as f32;
    if view.settings.ssao_enabled && let Some(stage) = inner.blur_stage.as_ref() {
        // uTexelSize steps one AO texel in AO-target uv space.
        let uniforms = BlurStageUniforms {
            texel_size: [2.0 / full_width, 2.0 / full_height],
            depth_sharpness: 1.1,
            normal_sharpness: 12.0,
        };
        queue.write_buffer(&stage.buffer, 0, bytemuck::bytes_of(&uniforms));
    }

    if let Some(stage) = inner.composite_stage.as_ref() {
        // uLightDirectionView is the world-space sun rotated into view space by
        // matView (the C++ rotated it on the CPU with the same matrix); it is a
        // different value from globals.sun_direction, which stays world-space.
        let sun = globals.globals.sun_direction;
        let matrix = globals.globals.view;
        let uniforms = CompositeStageUniforms {
            light_direction_view: [
                matrix[0] * sun[0] + matrix[4] * sun[1] + matrix[8] * sun[2],
                matrix[1] * sun[0] + matrix[5] * sun[1] + matrix[9] * sun[2],
                matrix[2] * sun[0] + matrix[6] * sun[1] + matrix[10] * sun[2],
                0.0,
            ],
            // Disabled SSAO skips both passes; the composite uses neutral AO
            // without sampling the held blur target.
            ao_strength: if view.settings.ssao_enabled {
                view.settings.ao_strength
            } else {
                0.0
            },
            _end_pad: [0.0; 3],
        };
        queue.write_buffer(&stage.buffer, 0, bytemuck::bytes_of(&uniforms));
    }

    // The SSAO and blur pipelines render into the fixed Rgba8Unorm targets, so
    // they are format-independent; the composite and FXAA pipelines are queued
    // lazily from their nodes, which are the only place the main texture's
    // format is known.
    if inner.ssao_pipeline.is_none() {
        let (Some(globals_group_layout), Some(layouts), Some(stage)) = (
            inner.globals_layout.clone(),
            inner.layouts.clone(),
            inner.ssao_stage.as_ref(),
        ) else {
            return;
        };
        let descriptor = fullscreen_pipeline(
            "forest_ssao_pipeline",
            shaders.ssao.clone(),
            wgpu::TextureFormat::Rgba8Unorm,
            vec![globals_group_layout, layouts.ssao.clone(), stage.layout.clone()],
        );
        inner.ssao_pipeline = Some(pipeline_cache.queue_render_pipeline(descriptor));
    }
    if inner.blur_pipeline.is_none() {
        let (Some(globals_group_layout), Some(layouts), Some(stage)) = (
            inner.globals_layout.clone(),
            inner.layouts.clone(),
            inner.blur_stage.as_ref(),
        ) else {
            return;
        };
        let descriptor = fullscreen_pipeline(
            "forest_blur_pipeline",
            shaders.ssao_blur.clone(),
            wgpu::TextureFormat::Rgba8Unorm,
            vec![globals_group_layout, layouts.blur.clone(), stage.layout.clone()],
        );
        inner.blur_pipeline = Some(pipeline_cache.queue_render_pipeline(descriptor));
    }
}

/// Registers the two prepare systems of the post chain. They run in
/// `RenderSystems::Prepare`, after the shared globals buffer exists.
///
/// Resize the G-buffer first so the post passes bind the new position and
/// heightfield views in the same frame, including during HiDPI changes.
pub fn register_post_systems(render_app: &mut SubApp) {
    let mut pipelines = prepare_post_pipelines
        .after(crate::render::prepare_forest_globals);
    pipelines = pipelines.after(crate::render::terrain_node::resize_gbuffer);
    pipelines = pipelines
        .after(crate::render::vegetation_shadows::prepare_vegetation_shadow_maps);
    render_app.add_systems(
        Render,
        (resize_ssao_targets, pipelines)
            .chain()
            .in_set(RenderSystems::Prepare),
    );
}

// ---------------------------------------------------------------------------
// Graph nodes
// ---------------------------------------------------------------------------

/// `RenderSSAO`'s first half: the raw ambient occlusion term into the half-res
/// AO target.
pub fn forest_ssao_pass(world: &World, mut ctx: RenderContext) {
    if !world
        .get_resource::<ExtractedForestView>()
        .is_some_and(|view| view.settings.ssao_enabled)
    {
        return;
    }
    let Some(state) = world.get_resource::<SsaoNodeState>() else {
        return;
    };
    let Some(pipeline_cache) = world.get_resource::<PipelineCache>() else {
        return;
    };
    let Ok(inner) = state.inner.lock() else {
        return;
    };
    let (Some(targets), Some(globals), Some(group), Some(stage), Some(pipeline)) = (
        inner.targets.as_ref(),
        inner.half_res_globals.as_ref(),
        inner.ssao_group.as_ref(),
        inner.ssao_stage.as_ref(),
        inner
            .ssao_pipeline
            .and_then(|id| pipeline_cache.get_render_pipeline(id)),
    ) else {
        return;
    };

    let ssao_pass_descriptor = wgpu::RenderPassDescriptor {
        label: Some("forest_ssao_pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: &targets.ao_view,
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
    let mut render_pass = ctx.begin_tracked_render_pass(ssao_pass_descriptor);
    draw_fullscreen(
        &mut render_pass,
        pipeline,
        &[&globals.group, group, &stage.group],
        targets.width as f32,
        targets.height as f32,
    );
}

/// `RenderSSAO`'s second half: the bilateral, depth/normal-weighted blur of the
/// raw AO buffer into the second half-res target.
pub fn forest_blur_pass(world: &World, mut ctx: RenderContext) {
    if !world
        .get_resource::<ExtractedForestView>()
        .is_some_and(|view| view.settings.ssao_enabled)
    {
        return;
    }
    let Some(state) = world.get_resource::<SsaoNodeState>() else {
        return;
    };
    let Some(pipeline_cache) = world.get_resource::<PipelineCache>() else {
        return;
    };
    let Ok(inner) = state.inner.lock() else {
        return;
    };
    let (Some(targets), Some(globals), Some(group), Some(stage), Some(pipeline)) = (
        inner.targets.as_ref(),
        inner.half_res_globals.as_ref(),
        inner.blur_group.as_ref(),
        inner.blur_stage.as_ref(),
        inner
            .blur_pipeline
            .and_then(|id| pipeline_cache.get_render_pipeline(id)),
    ) else {
        return;
    };

    let blur_pass_descriptor = wgpu::RenderPassDescriptor {
        label: Some("forest_ssao_blur_pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: &targets.blur_view,
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
    let mut render_pass = ctx.begin_tracked_render_pass(blur_pass_descriptor);
    draw_fullscreen(
        &mut render_pass,
        pipeline,
        &[&globals.group, group, &stage.group],
        targets.width as f32,
        targets.height as f32,
    );
}

/// `CompositeScene`: lighting, fog, tonemap and gamma encode from the
/// G-buffers plus the blurred AO, written to the offscreen LDR target.
///
/// The finished image does not go straight to the screen: FXAA needs the
/// neighbouring final pixels, which only exist once the whole frame has been
/// composited. The next node blits it out.
pub fn forest_composite_pass(
    view: ViewQuery<&ViewTarget>,
    world: &World,
    mut ctx: RenderContext,
) {
    let view = view.into_inner();
    let Some(state) = world.get_resource::<SsaoNodeState>() else {
        return;
    };
    let Some(shaders) = world.get_resource::<ForestShaderHandles>() else {
        return;
    };
    let Some(extracted) = world.get_resource::<ExtractedForestView>() else {
        return;
    };
    let Some(pipeline_cache) = world.get_resource::<PipelineCache>() else {
        return;
    };
    let Some(clouds) = world.get_resource::<super::cloud_node::CloudRenderState>()
        .and_then(|state| state.resources.as_ref()) else {
        return;
    };
    let Ok(mut inner) = state.inner.lock() else {
        return;
    };

    // One pipeline per main-texture format (see `SsaoNodeInner`), queued
    // from here because only the view knows the format; a queued pipeline
    // becomes usable on the next frame.
    let format = view.main_texture_format();
    if inner.atmosphere_pipeline.is_none() {
        let (Some(globals_layout), Some(layouts), Some(stage)) = (
            inner.globals_layout.clone(), inner.layouts.as_ref(), inner.composite_stage.as_ref(),
        ) else {
            return;
        };
        let mut descriptor = fullscreen_pipeline(
            "forest_atmosphere_pipeline",
            shaders.composite.clone(),
            wgpu::TextureFormat::Rgba16Float,
            vec![globals_layout, layouts.atmosphere.clone(), stage.layout.clone(), clouds.layout.clone()],
        );
        descriptor.fragment.as_mut().unwrap().entry_point = Some("fs_atmosphere".into());
        inner.atmosphere_pipeline = Some(pipeline_cache.queue_render_pipeline(descriptor));
    }
    let pipeline_id = match inner.composite_pipelines.get(&format).copied() {
        Some(id) => id,
        None => {
            let (Some(globals_group_layout), Some(layouts), Some(stage)) = (
                inner.globals_layout.clone(),
                inner.layouts.clone(),
                inner.composite_stage.as_ref(),
            ) else {
                return;
            };
            let descriptor = fullscreen_pipeline(
                "forest_composite_pipeline",
                shaders.composite.clone(),
                format,
                vec![globals_group_layout, layouts.composite.clone(), stage.layout.clone(), clouds.layout.clone()],
            );
            let id = pipeline_cache.queue_render_pipeline(descriptor);
            inner.composite_pipelines.insert(format, id);
            id
        }
    };
    let (Some(pipeline), Some(globals_group), Some(group), Some(stage)) = (
        pipeline_cache.get_render_pipeline(pipeline_id),
        inner.globals_group.as_ref(),
        inner.composite_group.as_ref(),
        inner.composite_stage.as_ref(),
    ) else {
        return;
    };

    let (Some(atmosphere_pipeline), Some(atmosphere_group), Some(half_globals), Some(targets)) = (
        inner.atmosphere_pipeline.and_then(|id| pipeline_cache.get_render_pipeline(id)),
        inner.atmosphere_group.as_ref(),
        inner.half_res_globals.as_ref(),
        inner.targets.as_ref(),
    ) else {
        return;
    };
    {
        let atmosphere_pass_descriptor = wgpu::RenderPassDescriptor {
            label: Some("forest_atmosphere_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &targets.atmosphere_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 1.0 }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        };
        let mut pass = ctx.begin_tracked_render_pass(atmosphere_pass_descriptor);
        draw_fullscreen(
            &mut pass,
            atmosphere_pipeline,
            &[&half_globals.group, atmosphere_group, &stage.group, &clouds.group],
            targets.width as f32,
            targets.height as f32,
        );
    }

    let post_process = view.post_process_write();
    let destination: &wgpu::TextureView = post_process.destination;
    let width = extracted.physical_width.max(1) as f32;
    let height = extracted.physical_height.max(1) as f32;

    let composite_pass_descriptor = wgpu::RenderPassDescriptor {
        label: Some("forest_composite_pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: destination,
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
    let mut render_pass = ctx.begin_tracked_render_pass(composite_pass_descriptor);
    draw_fullscreen(
        &mut render_pass,
        pipeline,
        &[globals_group, group, &stage.group, &clouds.group],
        width,
        height,
    );
}

/// `PresentFxaa`: edge smoothing of the composited frame, blitted out to the
/// screen. The primary texture (unit 0) comes from the draw call itself
/// (`post_process.source`), matching how the composite receives its G-buffer
/// position texture.
pub fn forest_fxaa_pass(
    view: ViewQuery<&ViewTarget>,
    world: &World,
    mut ctx: RenderContext,
) {
    let view = view.into_inner();
    let Some(state) = world.get_resource::<SsaoNodeState>() else {
        return;
    };
    let Some(shaders) = world.get_resource::<ForestShaderHandles>() else {
        return;
    };
    let Some(extracted) = world.get_resource::<ExtractedForestView>() else {
        return;
    };
    let Some(device) = world.get_resource::<RenderDevice>() else {
        return;
    };
    let Some(pipeline_cache) = world.get_resource::<PipelineCache>() else {
        return;
    };
    let Ok(mut inner) = state.inner.lock() else {
        return;
    };

    let format = view.main_texture_format();
    let pipeline_id = match inner.fxaa_pipelines.get(&format).copied() {
        Some(id) => id,
        None => {
            let (Some(globals_group_layout), Some(layouts)) =
                (inner.globals_layout.clone(), inner.layouts.clone())
            else {
                return;
            };
            let descriptor = fullscreen_pipeline(
                "forest_fxaa_pipeline",
                shaders.fxaa.clone(),
                format,
                vec![globals_group_layout, layouts.fxaa.clone()],
            );
            let id = pipeline_cache.queue_render_pipeline(descriptor);
            inner.fxaa_pipelines.insert(format, id);
            id
        }
    };
    let (Some(pipeline), Some(globals_group), Some(layouts), Some(samplers)) = (
        pipeline_cache.get_render_pipeline(pipeline_id),
        inner.globals_group.as_ref(),
        inner.layouts.as_ref(),
        inner.samplers.as_ref(),
    ) else {
        return;
    };

    let post_process = view.post_process_write();
    let source: &wgpu::TextureView = post_process.source;
    let destination: &wgpu::TextureView = post_process.destination;
    // The composite writes into whichever main texture is not currently
    // being read, and the two alternate every frame, so this group is
    // rebuilt per frame from the source view handed back above.
    let group = super::bind_group(
        &device,
        &pipeline_cache,
        "forest_fxaa_group",
        &layouts.fxaa,
        &[
            texture_entry(0, source),
            sampler_entry(8, &samplers.linear_clamp),
        ],
    );

    let fxaa_pass_descriptor = wgpu::RenderPassDescriptor {
        label: Some("forest_fxaa_pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: destination,
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
    let mut render_pass = ctx.begin_tracked_render_pass(fxaa_pass_descriptor);
    draw_fullscreen(
        &mut render_pass,
        pipeline,
        &[globals_group, &group],
        extracted.physical_width.max(1) as f32,
        extracted.physical_height.max(1) as f32,
    );
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Binds one fullscreen pipeline and draws its three-vertex triangle over the
/// whole viewport. Blending stays off, as `RenderPass` does in the C++: these
/// passes write numerical state, so standard alpha blending would corrupt the
/// channels.
fn draw_fullscreen<'a>(
    render_pass: &mut TrackedRenderPass<'a>,
    pipeline: &'a RenderPipeline,
    bind_groups: &[&'a BindGroup],
    width: f32,
    height: f32,
) {
    render_pass.set_render_pipeline(pipeline);
    render_pass.set_viewport(0.0, 0.0, width, height, 0.0, 1.0);
    for (index, bind_group) in bind_groups.iter().enumerate() {
        // `*bind_group` copies the reference out of the slice element so the
        // bind group carries its own lifetime, not the iterator's borrow.
        render_pass.set_bind_group(index, *bind_group, &[]);
    }
    render_pass.draw(0..3, 0..1);
}

/// The fullscreen-triangle pipelines all share this shape: one vertex stage
/// with no vertex buffers, one fragment target, no depth state, no blending.
fn fullscreen_pipeline(
    label: &'static str,
    shader: Handle<Shader>,
    format: wgpu::TextureFormat,
    layout: Vec<BindGroupLayoutDescriptor>,
) -> RenderPipelineDescriptor {
    RenderPipelineDescriptor {
        label: Some(label.into()),
        layout,
        immediate_size: 0,
        vertex: VertexState {
            shader: shader.clone(),
            shader_defs: Vec::new(),
            entry_point: Some("vs_main".into()),
            buffers: Vec::new(),
        },
        primitive: PrimitiveState::default(),
        depth_stencil: None,
        multisample: MultisampleState::default(),
        fragment: Some(FragmentState {
            shader,
            shader_defs: Vec::new(),
            entry_point: Some("fs_main".into()),
            targets: vec![Some(ColorTargetState {
                format,
                // The C++ disabled colour blending for every one of these
                // passes; the shader's output is the final pixel state.
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
        }),
        zero_initialize_workgroup_memory: true,
    }
}

/// One group-1 layout: for every entry in `filterable`, a float texture at
/// binding N whose matching sampler sits at binding N + 8. Textures the C++
/// point-filtered take a non-filtering layout, the BILINEAR ones a filtering
/// layout; naga only distinguishes the image class for provided layouts, while
/// the sampler/texture pairing is what wgpu validates.
fn screen_group_layout(
    label: &'static str,
    filterable: &[bool],
) -> BindGroupLayoutDescriptor {
    let mut entries: Vec<wgpu::BindGroupLayoutEntry> = Vec::with_capacity(filterable.len() * 2);
    for (index, filterable) in filterable.iter().enumerate() {
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: index as u32,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float {
                    filterable: *filterable,
                },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        });
    }
    for (index, filterable) in filterable.iter().enumerate() {
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: index as u32 + 8,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(if *filterable {
                wgpu::SamplerBindingType::Filtering
            } else {
                wgpu::SamplerBindingType::NonFiltering
            }),
            count: None,
        });
    }
    BindGroupLayoutDescriptor::new(label, &entries)
}

/// Adds the plants' independently sized depth cascades at bindings 6 and
/// 15..=16, their uniform at 7 and the comparison sampler at 14.
fn with_vegetation_shadows(mut layout: BindGroupLayoutDescriptor) -> BindGroupLayoutDescriptor {
    layout.entries.extend(
        [6, 15, 16].map(|binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Depth,
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        }),
    );
    layout.entries.extend([
        wgpu::BindGroupLayoutEntry {
            binding: 7,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<ShadowUniform>() as u64),
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 14,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
            count: None,
        },
    ]);
    layout
}

/// Binds a sampled texture at `binding`.
fn texture_entry<'a>(binding: u32, view: &'a wgpu::TextureView) -> BindGroupEntry<'a> {
    BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::TextureView(view),
    }
}

/// Binds a sampler at `binding`.
fn sampler_entry<'a>(binding: u32, sampler: &'a wgpu::Sampler) -> BindGroupEntry<'a> {
    BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::Sampler(sampler),
    }
}

/// Creates one pass's group-2 uniform buffer, its single-entry layout and the
/// bind group wrapping the whole buffer.
fn create_stage(
    device: &RenderDevice,
    cache: &PipelineCache,
    label: &'static str,
    size: u64,
) -> Stage {
    let stage_entries = [wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: wgpu::BufferSize::new(size),
        },
        count: None,
    }];
    let layout = BindGroupLayoutDescriptor::new(label, &stage_entries);
    let buffer = device.wgpu_device().create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let group = super::bind_group(
        device,
        cache,
        label,
        &layout,
        &[BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    );
    Stage {
        layout,
        buffer,
        group,
    }
}

/// One half-res render target (`ResizeSSAO`'s `LoadRenderTexture` for the AO
/// and blur buffers): RGBA8, non-sRGB, usable both as an attachment and as a
/// sampled texture.
fn create_half_res_target(
    device: &RenderDevice,
    label: &str,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
) -> wgpu::TextureView {
    let texture = device.wgpu_device().create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}
