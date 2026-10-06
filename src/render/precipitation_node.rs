//! Camera-near, world-anchored rain and snow rendered after the water passes,
//! with rain layers that carry a downpour farther out and splash crowns where
//! drops land. The position G-buffer clips particles hidden by terrain and
//! vegetation and places the splashes on the ground.

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

/// A particle lattice around the camera: cells per side, vertical layers, and
/// cell size in metres. The shader culls cells outside the view frustum and
/// the depth-tested volume before it evaluates the weather field.
#[derive(Clone, Copy)]
struct Lattice {
    grid: u32,
    layers: u32,
    cell_size: f32,
}

impl Lattice {
    const fn count(self) -> u32 {
        self.grid * self.grid * self.layers
    }
}

/// Dense rain within about 16 m of the camera, about 3 drops per cubic metre
/// in a downpour.
const RAIN_NEAR_LATTICE: Lattice = Lattice { grid: 48, layers: 20, cell_size: 0.7 };
/// Sparser, longer streaks that carry the rain out to about 45 m.
const RAIN_FAR_LATTICE: Lattice = Lattice { grid: 48, layers: 16, cell_size: 2.0 };
const SNOW_LATTICE: Lattice = Lattice { grid: 24, layers: 12, cell_size: 2.0 };
/// Screen cells that each try to place one splash crown per cycle.
const SPLASH_CELLS: [u32; 2] = [96, 54];
const PARTICLE_INSTANCES: u32 = RAIN_NEAR_LATTICE.count()
    + RAIN_FAR_LATTICE.count()
    + SNOW_LATTICE.count()
    + SPLASH_CELLS[0] * SPLASH_CELLS[1];
/// Rain layers reach about 110 m; precipitation that close keeps the pass on.
const PRECIPITATION_REACH: f32 = 110.0;

/// `PrecipitationFrame` in precipitation.wgsl (group 2, binding 0).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PrecipitationFrameGpu {
    velocity: [f32; 4],
    rain_near: [f32; 4],
    rain_far: [f32; 4],
    snow: [f32; 4],
    splash: [f32; 4],
}
const _: () = assert!(std::mem::size_of::<PrecipitationFrameGpu>() == 80);

impl PrecipitationFrameGpu {
    /// Lays the instance range out as near rain, far rain, snow, then splashes.
    fn new(velocity: [f32; 4]) -> Self {
        let lattice = |lattice: Lattice, end: u32| {
            [lattice.grid as f32, lattice.layers as f32, lattice.cell_size, end as f32]
        };
        let near_end = RAIN_NEAR_LATTICE.count();
        let far_end = near_end + RAIN_FAR_LATTICE.count();
        let snow_end = far_end + SNOW_LATTICE.count();
        Self {
            velocity,
            rain_near: lattice(RAIN_NEAR_LATTICE, near_end),
            rain_far: lattice(RAIN_FAR_LATTICE, far_end),
            snow: lattice(SNOW_LATTICE, snow_end),
            splash: [
                SPLASH_CELLS[0] as f32,
                SPLASH_CELLS[1] as f32,
                0.0,
                PARTICLE_INSTANCES as f32,
            ],
        }
    }
}

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
        // Drops take their colour from the scene around them and splashes
        // read the G-buffer to find the ground, both in the vertex stage.
        let texture = |binding: u32, filterable: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        inner.screen_layout = Some(BindGroupLayoutDescriptor::new(
            "forest_precipitation_screen_layout",
            &[texture(0, true), texture(1, false), texture(2, true)],
        ));
    }
    if inner.wind_layout.is_none() {
        let size = std::mem::size_of::<PrecipitationFrameGpu>() as u64;
        let wind_entries = [wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(size),
            },
            count: None,
        }];
        let layout = BindGroupLayoutDescriptor::new(
            "forest_precipitation_wind_layout",
            &wind_entries,
        );
        let buffer_descriptor = wgpu::BufferDescriptor {
            label: Some("forest_precipitation_wind"),
            size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        };
        let buffer = device.wgpu_device().create_buffer(&buffer_descriptor);
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
        let frame = PrecipitationFrameGpu::new([
            speed * direction.cos(), speed * direction.sin(),
            water.draw as u32 as f32, view.weather.local_precipitation.gust,
        ]);
        queue.write_buffer(buffer, 0, bytemuck::bytes_of(&frame));
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
    // Rain bands can cross the particle volume and rain layers before
    // reaching the player. Sample their extent as well as the local reading
    // so a leading edge remains visible and moves across the landscape.
    let reach = [-PRECIPITATION_REACH, -24.0, 0.0, 24.0, PRECIPITATION_REACH];
    let sample_has_rain = |dx: f32, dz: f32| {
        let sample = crate::weather::sample_precipitation(
            [extracted.player_position[0] + dx, extracted.player_position[2] + dz],
            extracted.player_position[1] - 8.0,
            extracted.weather_offset,
            extracted.weather.climate_bias,
            extracted.weather.precipitation_bias,
            extracted.weather.precipitation_override(),
            extracted.settings.cloud_base_height,
            extracted.weather.overrides.base,
            extracted.settings.cloud_wind_direction_degrees.to_radians(),
        );
        sample.intensity() > 0.001
    };
    let row_has_rain = |dx: f32| reach.into_iter().any(|dz| sample_has_rain(dx, dz));
    let nearby_precipitation = extracted.weather.local_precipitation.intensity() > 0.001
        || reach.into_iter().any(row_has_rain);
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
            "fs_rain_layers",
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
    let screen_entries = [
        wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(post_process.source),
        },
        wgpu::BindGroupEntry {
            binding: 1,
            resource: wgpu::BindingResource::TextureView(&gbuffer.position_view),
        },
        wgpu::BindGroupEntry {
            binding: 2,
            resource: wgpu::BindingResource::TextureView(&gbuffer.normal_view),
        },
    ];
    let screen_group = super::bind_group(
        device,
        cache,
        "forest_precipitation_screen",
        screen_layout,
        &screen_entries,
    );
    let precipitation_pass_descriptor = wgpu::RenderPassDescriptor {
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
    };
    let mut pass = ctx.begin_tracked_render_pass(precipitation_pass_descriptor);
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
    pass.draw(0..4, 0..PARTICLE_INSTANCES);
}

pub(super) fn pipeline_descriptor(
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

#[cfg(test)]
mod tests {
    /// The frame uniform the pass writes must match the WGSL struct.
    #[test]
    fn precipitation_frame_matches_the_shader_layout() {
        let source = include_str!("../../assets/shaders/precipitation.wgsl");
        let module = naga::front::wgsl::parse_str(source).unwrap();
        let (_, ty) = module
            .types
            .iter()
            .find(|(_, ty)| ty.name.as_deref() == Some("PrecipitationFrame"))
            .expect("PrecipitationFrame");
        let naga::TypeInner::Struct { span, .. } = &ty.inner else {
            panic!("PrecipitationFrame must be a struct");
        };
        assert_eq!(*span as usize, std::mem::size_of::<super::PrecipitationFrameGpu>());
        // Instance indices travel through the uniform as floats.
        assert!(super::PARTICLE_INSTANCES < 1 << 24);
    }
}
