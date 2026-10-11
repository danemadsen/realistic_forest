//! The lightning channel, drawn after the water passes and before the rain.
//! [`crate::lightning`] grows the bolt; this pass uploads its segments with
//! the current stroke's luminance and adds the channel's glow to the scene.

use super::{ExtractedForestView, ForestGlobals, ForestShaderHandles, globals_layout};
use crate::lightning::{BoltKind, MAX_BOLT_SEGMENTS};
use bevy::prelude::*;
use bevy::render::render_resource::{
    BindGroup, BindGroupLayoutDescriptor, CachedRenderPipelineId, PipelineCache,
};
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue, ViewQuery};
use bevy::render::view::ViewTarget;
use std::collections::HashMap;
use std::sync::Mutex;

/// `BoltUniforms` in lightning.wgsl: header, cloud, then three vec4s per
/// segment. Written as a flat vec4 array, since bytemuck derives `Pod` for
/// arrays this long only behind an optional feature.
const BOLT_UNIFORM_VEC4S: usize = 2 + 3 * MAX_BOLT_SEGMENTS;
const BOLT_UNIFORM_BYTES: u64 = (BOLT_UNIFORM_VEC4S * 16) as u64;

/// Light added on top of the resolved scene; the alpha channel is untouched.
const ADDITIVE: wgpu::BlendState = wgpu::BlendState {
    color: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    },
    alpha: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::Zero,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    },
};

#[derive(Default)]
struct LightningInner {
    globals_group: Option<BindGroup>,
    globals_layout: Option<BindGroupLayoutDescriptor>,
    screen_layout: Option<BindGroupLayoutDescriptor>,
    bolt_layout: Option<BindGroupLayoutDescriptor>,
    bolt_buffer: Option<wgpu::Buffer>,
    bolt_group: Option<BindGroup>,
    pipelines: HashMap<wgpu::TextureFormat, (CachedRenderPipelineId, CachedRenderPipelineId)>,
}

#[derive(Resource, Default)]
pub struct LightningNodeState {
    inner: Mutex<LightningInner>,
}

/// Whether any part of the channel gives light this frame.
fn bolt_visible(view: &ExtractedForestView) -> bool {
    let lightning = view.weather.lightning;
    view.bolt.kind != BoltKind::Hidden
        && !view.bolt.segments.is_empty()
        && lightning.channel.max(lightning.branches) > 0.001
}

/// The uniform contents: the stroke's luminance and the channel geometry.
fn bolt_uniforms(view: &ExtractedForestView) -> Vec<[f32; 4]> {
    let lightning = view.weather.lightning;
    let segments = &view.bolt.segments[..view.bolt.segments.len().min(MAX_BOLT_SEGMENTS)];
    let above_deck = view.player_position[1] > lightning.cloud_base + 50.0;
    let mut data = vec![[0.0; 4]; BOLT_UNIFORM_VEC4S];
    data[0] = [segments.len() as f32, lightning.channel, lightning.branches, lightning.leader];
    data[1] = [lightning.cloud_base, 0.0, above_deck as u32 as f32, 0.0];
    for (index, segment) in segments.iter().enumerate() {
        let [x0, y0, z0] = segment.start;
        let [x1, y1, z1] = segment.end;
        data[2 + index * 3] = [x0, y0, z0, segment.width];
        data[3 + index * 3] = [x1, y1, z1, segment.brightness];
        data[4 + index * 3] = [segment.arrival[0], segment.arrival[1], segment.order as f32, 0.0];
    }
    data
}

fn prepare_lightning(
    state: Res<LightningNodeState>,
    globals: Res<ForestGlobals>,
    view: Res<ExtractedForestView>,
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
            "forest_lightning_globals",
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
        let texture = |binding: u32, filterable: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        inner.screen_layout = Some(BindGroupLayoutDescriptor::new(
            "forest_lightning_screen_layout",
            &[texture(0, true), texture(1, false)],
        ));
    }
    if inner.bolt_layout.is_none() {
        let bolt_entries = [wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(BOLT_UNIFORM_BYTES),
            },
            count: None,
        }];
        let layout = BindGroupLayoutDescriptor::new(
            "forest_lightning_bolt_layout",
            &bolt_entries,
        );
        let buffer_descriptor = wgpu::BufferDescriptor {
            label: Some("forest_lightning_bolt"),
            size: BOLT_UNIFORM_BYTES,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        };
        let buffer = device.wgpu_device().create_buffer(&buffer_descriptor);
        let group = super::bind_group(
            &device,
            &cache,
            "forest_lightning_bolt_group",
            &layout,
            &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        );
        inner.bolt_layout = Some(layout);
        inner.bolt_buffer = Some(buffer);
        inner.bolt_group = Some(group);
    }
    if bolt_visible(&view) {
        if let Some(buffer) = inner.bolt_buffer.as_ref() {
            let uniforms = bolt_uniforms(&view);
            queue.write_buffer(buffer, 0, bytemuck::cast_slice(&uniforms));
        }
    }
}

pub fn register_lightning_systems(render_app: &mut bevy::app::SubApp) {
    render_app.init_resource::<LightningNodeState>();
    render_app.add_systems(
        bevy::render::Render,
        prepare_lightning
            .in_set(bevy::render::RenderSystems::Prepare)
            .after(super::prepare_forest_globals),
    );
}

pub fn forest_lightning_pass(
    view: ViewQuery<&ViewTarget>,
    world: &World,
    mut ctx: RenderContext,
) {
    let view = view.into_inner();
    let (Some(state), Some(extracted), Some(shaders), Some(device), Some(cache), Some(terrain)) = (
        world.get_resource::<LightningNodeState>(),
        world.get_resource::<ExtractedForestView>(),
        world.get_resource::<ForestShaderHandles>(),
        world.get_resource::<RenderDevice>(),
        world.get_resource::<PipelineCache>(),
        world.get_resource::<super::terrain_node::TerrainNodeState>(),
    ) else {
        return;
    };
    let Ok(mut inner) = state.inner.lock() else {
        return;
    };
    // Queue the pipelines before the first strike, so its opening frames
    // are not lost to shader compilation.
    let format = view.main_texture_format();
    let (blit_id, bolt_id) = if let Some(ids) = inner.pipelines.get(&format).copied() {
        ids
    } else {
        let (Some(globals_layout), Some(screen_layout), Some(bolt_layout)) = (
            inner.globals_layout.clone(),
            inner.screen_layout.clone(),
            inner.bolt_layout.clone(),
        ) else {
            return;
        };
        let layouts = vec![globals_layout, screen_layout, bolt_layout];
        let blit = cache.queue_render_pipeline(super::precipitation_node::pipeline_descriptor(
            "forest_lightning_blit_pipeline",
            &shaders.lightning,
            format,
            layouts.clone(),
            "vs_blit",
            "fs_blit",
            None,
            wgpu::PrimitiveTopology::TriangleList,
        ));
        let bolt = cache.queue_render_pipeline(super::precipitation_node::pipeline_descriptor(
            "forest_lightning_bolt_pipeline",
            &shaders.lightning,
            format,
            layouts,
            "vs_bolt",
            "fs_bolt",
            Some(ADDITIVE),
            wgpu::PrimitiveTopology::TriangleStrip,
        ));
        inner.pipelines.insert(format, (blit, bolt));
        (blit, bolt)
    };
    if !bolt_visible(extracted) {
        return;
    }
    // Under the drawn water, the sea's or a river's or a lake's, the sky and
    // its bolts are the medium's to show.
    if world
        .get_resource::<super::water_node::ExtractedWater>()
        .is_some_and(|water| water.eye_submerged(0.2))
    {
        return;
    }
    let (Some(blit), Some(bolt), Some(globals), Some(bolt_group), Some(screen_layout)) = (
        cache.get_render_pipeline(blit_id),
        cache.get_render_pipeline(bolt_id),
        inner.globals_group.as_ref(),
        inner.bolt_group.as_ref(),
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
    let screen_entries = [
        wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(post_process.source),
        },
        wgpu::BindGroupEntry {
            binding: 1,
            resource: wgpu::BindingResource::TextureView(&gbuffer.position_view),
        },
    ];
    let screen_group = super::bind_group(
        device,
        cache,
        "forest_lightning_screen",
        screen_layout,
        &screen_entries,
    );
    let lightning_pass_descriptor = wgpu::RenderPassDescriptor {
        label: Some("forest_lightning_pass"),
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
    };
    let mut pass = ctx.begin_tracked_render_pass(lightning_pass_descriptor);
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
    pass.set_bind_group(2, bolt_group, &[]);
    pass.set_render_pipeline(blit);
    pass.draw(0..3, 0..1);
    pass.set_render_pipeline(bolt);
    let segments = extracted.bolt.segments.len().min(MAX_BOLT_SEGMENTS) as u32;
    pass.draw(0..4, 0..segments);
}

#[cfg(test)]
mod tests {
    /// The uniform the pass writes must match the WGSL struct byte for byte.
    #[test]
    fn bolt_uniform_matches_the_shader_layout() {
        let source = include_str!("../../assets/shaders/lightning.wgsl");
        let module = naga::front::wgsl::parse_str(source).unwrap();
        let (_, ty) = module
            .types
            .iter()
            .find(|(_, ty)| ty.name.as_deref() == Some("BoltUniforms"))
            .expect("BoltUniforms");
        let naga::TypeInner::Struct { span, .. } = &ty.inner else {
            panic!("BoltUniforms must be a struct");
        };
        assert_eq!(*span as u64, super::BOLT_UNIFORM_BYTES);
    }
}
