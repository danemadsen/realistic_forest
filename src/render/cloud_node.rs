//! Periodic density noise, the per-frame cloud shadow map and a raymarched,
//! all-direction cloud reflection probe. The main atmosphere pass uses the
//! same density and lighting model.

use super::{ForestGlobals, ForestShaderHandles, GlobalUniformsGpu, globals_layout};
use bevy::prelude::*;
use bevy::render::render_resource::{
    BindGroup, BindGroupLayoutDescriptor, CachedRenderPipelineId, FragmentState, PipelineCache,
    RenderPipelineDescriptor, VertexState,
};
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue};

const NOISE_SIZE: u32 = 64;
const NOISE_MIP_COUNT: u32 = 7;
pub const PROBE_WIDTH: u32 = 256;
pub const PROBE_HEIGHT: u32 = 128;
/// Texels per side of each cloud shadow map level (`CLOUD_SHADOW_MAP_TEXELS`
/// in cloud-functions.wgslinc). The two levels sit side by side.
pub const SHADOW_MAP_LEVEL_TEXELS: u32 = 1024;
const SHADOW_MAP_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;

#[derive(Resource, Default)]
pub struct CloudRenderState {
    pub resources: Option<CloudGpuResources>,
}

pub struct CloudGpuResources {
    /// Shared group 3: density noise, the completed sky reflection probe and
    /// the cloud shadow map.
    pub layout: BindGroupLayoutDescriptor,
    pub group: BindGroup,
    noise_group: BindGroup,
    globals_group: BindGroup,
    empty_group: BindGroup,
    probe: wgpu::TextureView,
    pipeline: CachedRenderPipelineId,
    /// Optical depth toward the sun per sun-ray entry point into the cloud
    /// layer, rebuilt each frame before the atmosphere reads it.
    shadow_map: wgpu::TextureView,
    shadow_pipeline: CachedRenderPipelineId,
}

/// How many frames the shadow map and sky probe reuse before
/// re-integrating. Both results are anchored in world space: camera rotation
/// never invalidates them, and camera translation during a reuse window
/// slides the sampling across cloud features the way a short walk does — the
/// default flight speed (38 m/s, [`crate::player`]) covers ~7 m per
/// two-frame window, under a quarter of one probe texel's 32-m arc at the
/// 1300 m default base. See [`ProbeSkyState`] for why the stepwise inputs
/// are latched separately and [`cloud_probe_pass`] for the reuse itself.
pub const PROBE_INTEGRATION_PERIOD: u32 = 2;

/// Per-frame decision to re-integrate the sky probe, shared between the
/// Prepare stage and the render pass.
#[derive(Resource, Default)]
pub struct CloudProbeSchedule {
    integrate: bool,
    frames_since_probe: u32,
    last_sky_state: Option<ProbeSkyState>,
}

/// The probe's stepwise inputs, latched against per-field tolerances so one
/// frame of drift never re-triggers an integration. Wind offsets
/// (`cloud_motion.xy`), the weather front position (`weather.xy`), the storm
/// clock (`storm.z`) and the camera are deliberately absent: they advect
/// continuously and their staleness is bounded by [`PROBE_INTEGRATION_PERIOD`]
/// instead. The flash is not latched either — it rises too fast for any
/// cadence window, so [`update_cloud_probe_schedule`] integrates at full rate
/// while it is lit.
#[derive(Clone, Copy)]
struct ProbeSkyState {
    sun_direction: [f32; 3],
    moon_direction: [f32; 3],
    sun_colour: [f32; 4],
    sun_intensity: f32,
    daylight: f32,
    moon_intensity: f32,
    clouds: [f32; 4],
    layer_thickness: f32,
    layer_scale: f32,
    layer_shadow: f32,
    layer_quality: f32,
    detail_strength: f32,
    march_range: f32,
    climate_bias: f32,
    overrides: u32,
    precip_bias: f32,
    precip_override: u32,
    wind_radians: f32,
    /// Whether the cloud shadow map's usage gate currently passes. The gate's
    /// conditions (raymarch toggle, sun intensity, layer opacity, sun height)
    /// ramp through thresholds narrower than the field tolerances above, so
    /// their crossings latch here: a flip forces the shadow map's first usable
    /// frame to rebuild instead of reading content from before the gate went
    /// dark.
    shadow_gate: bool,
}

impl ProbeSkyState {
    fn from_globals(g: &GlobalUniformsGpu) -> Self {
        Self {
            sun_direction: [g.sun_direction[0], g.sun_direction[1], g.sun_direction[2]],
            moon_direction: [
                g.moon_direction[0],
                g.moon_direction[1],
                g.moon_direction[2],
            ],
            sun_colour: g.sun_colour,
            sun_intensity: g.settings_a[0],
            daylight: g.atmosphere[0],
            moon_intensity: g.atmosphere[1],
            clouds: g.clouds,
            layer_thickness: g.cloud_layer[0],
            layer_scale: g.cloud_layer[1],
            layer_shadow: g.cloud_layer[2],
            layer_quality: g.cloud_layer[3],
            detail_strength: g.cloud_motion[2],
            march_range: g.cloud_motion[3],
            climate_bias: g.weather[2],
            overrides: g.weather[3] as u32,
            precip_bias: g.storm[0],
            precip_override: g.storm[1] as u32,
            wind_radians: g.storm[3],
            // Mirrors the shadow map's rebuild gate in cloud_probe_pass; the
            // two must stay in sync.
            shadow_gate: g.raymarch[1] >= 0.5
                && g.atmosphere[3] > 0.0
                && g.settings_a[0] > 0.01
                && g.clouds[0] >= 0.5
                && g.clouds[2] > 0.001
                && g.cloud_layer[2] > 0.001
                && g.sun_direction[1] < 0.0,
        }
    }

    /// Tolerances sit just above live drift and below one user step: a
    /// scrubbed slider or a teleported sun exceeds them, a 75 s weather
    /// easing and the sun's 0.0042°/s drift do not.
    fn changed_beyond(&self, next: &Self) -> bool {
        fn moved(a: [f32; 3], b: [f32; 3], limit: f32) -> bool {
            (a[0] - b[0]).abs() > limit
                || (a[1] - b[1]).abs() > limit
                || (a[2] - b[2]).abs() > limit
        }
        fn differs(a: f32, b: f32, limit: f32) -> bool {
            (a - b).abs() > limit
        }
        // One probe texel spans ~1.4° of direction.
        moved(self.sun_direction, next.sun_direction, 0.02)
            || moved(
                self.moon_direction,
                next.moon_direction,
                0.02,
            )
            || differs(self.sun_colour[0], next.sun_colour[0], 0.02)
            || differs(self.sun_colour[1], next.sun_colour[1], 0.02)
            || differs(self.sun_colour[2], next.sun_colour[2], 0.02)
            || differs(self.sun_colour[3], next.sun_colour[3], 0.02)
            || differs(self.sun_intensity, next.sun_intensity, 0.02)
            || differs(self.daylight, next.daylight, 0.02)
            || differs(self.moon_intensity, next.moon_intensity, 0.02)
            // Toggles and scrubbed sliders step, they never drift.
            || self.clouds[0] != next.clouds[0]
            || differs(self.clouds[1], next.clouds[1], 0.02)
            || differs(self.clouds[2], next.clouds[2], 0.02)
            || differs(self.clouds[3], next.clouds[3], 50.0)
            || differs(self.layer_thickness, next.layer_thickness, 20.0)
            || differs(self.layer_scale, next.layer_scale, 20.0)
            || differs(self.layer_shadow, next.layer_shadow, 0.02)
            // Quality steps are 1.0 apart (0/1/2).
            || differs(self.layer_quality, next.layer_quality, 0.5)
            || differs(self.detail_strength, next.detail_strength, 0.02)
            || differs(self.march_range, next.march_range, 500.0)
            || differs(self.climate_bias, next.climate_bias, 0.05)
            || self.overrides != next.overrides
            || differs(self.precip_bias, next.precip_bias, 0.05)
            || self.precip_override != next.precip_override
            || differs(self.wind_radians, next.wind_radians, 0.05)
            // The shadow gate crossed a threshold: rebuild the map on its
            // first usable frame.
            || self.shadow_gate != next.shadow_gate
    }
}

/// Runs in Prepare before [`cloud_probe_pass`]: decides whether this frame
/// re-integrates the cloud passes or lets consumers read their persistent
/// textures.
fn update_cloud_probe_schedule(
    mut schedule: ResMut<CloudProbeSchedule>,
    globals: Res<ForestGlobals>,
) {
    let state = ProbeSkyState::from_globals(&globals.globals);
    // While a strike lights the clouds the flash rises over ~20 ms, faster
    // than any cadence window; keep the probe at full rate until it clears.
    let changed = schedule
        .last_sky_state
        .map_or(true, |previous| previous.changed_beyond(&state));
    let integrate = globals.globals.lightning[3] > 0.001
        || changed
        || schedule.frames_since_probe >= PROBE_INTEGRATION_PERIOD;
    if integrate {
        schedule.frames_since_probe = 0;
        schedule.last_sky_state = Some(state);
    } else {
        schedule.frames_since_probe += 1;
    }
    schedule.integrate = integrate;
}

/// Renders the cloud shadow map, then the raymarched all-direction cloud
/// reflection probe. Both bind the noise group in slot 3 rather than the
/// resource group: sampling a texture while it is the active attachment is
/// invalid. Both passes skip on reuse frames ([`CloudProbeSchedule`]) and
/// their textures keep the previous integration: the map is world-anchored
/// like the LUT, and its consumers re-derive the lookup origin from the
/// current camera and sun each frame.
pub fn cloud_probe_pass(world: &World, mut ctx: RenderContext) {
    let Some(resources) = world
        .get_resource::<CloudRenderState>()
        .and_then(|state| state.resources.as_ref())
    else {
        return;
    };
    let cache = world.resource::<PipelineCache>();
    // Until its pipeline compiles the map keeps its -1 fill, which the
    // shaders read as "march the clouds instead".
    let lighting = &world.resource::<ForestGlobals>().globals;
    let reuse = !world
        .get_resource::<CloudProbeSchedule>()
        .is_some_and(|schedule| schedule.integrate);
    // Only the volumetric sunlight paths read this map. When those paths
    // become active again, rebuild it before their first lookup this frame:
    // the same chain is mirrored in ProbeSkyState::shadow_gate, and a gate
    // flip latches a re-integration. The map is a world-space entry-point
    // lattice whose lookups re-derive their origin from the current camera
    // and sun each frame, so reuse windows show only the sub-texel wind
    // drift, like the probe below.
    if lighting.raymarch[1] >= 0.5
        && lighting.atmosphere[3] > 0.0
        && lighting.settings_a[0] > 0.01
        && lighting.clouds[0] >= 0.5
        && lighting.clouds[2] > 0.001
        && lighting.cloud_layer[2] > 0.001
        && lighting.sun_direction[1] < 0.0
        && let Some(pipeline) = cache.get_render_pipeline(resources.shadow_pipeline)
        && !reuse
    {
        let mut pass = ctx.begin_tracked_render_pass(wgpu::RenderPassDescriptor {
            label: Some("forest_cloud_shadow_map"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &resources.shadow_map,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    // The fullscreen triangle writes every texel.
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_render_pipeline(pipeline);
        pass.set_bind_group(0, &resources.globals_group, &[]);
        pass.set_bind_group(1, &resources.empty_group, &[]);
        pass.set_bind_group(2, &resources.empty_group, &[]);
        pass.set_bind_group(3, &resources.noise_group, &[]);
        pass.draw(0..3, 0..1);
    }
    // Between re-integrations the probe keeps the previous frame's LUT —
    // world-anchored, changing a fraction of one texel per reuse window. The
    // shadow map above skips on the same cadence: `reuse` reads the same
    // schedule flag, so re-integration frames are the ones the shadow map
    // built on.
    let Some(pipeline) = cache.get_render_pipeline(resources.pipeline) else {
        return;
    };
    if !reuse {
        let mut pass = ctx.begin_tracked_render_pass(wgpu::RenderPassDescriptor {
            label: Some("forest_cloud_sky_probe"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &resources.probe,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.0,
                        g: 0.0,
                        b: 0.0,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_render_pipeline(pipeline);
        pass.set_bind_group(0, &resources.globals_group, &[]);
        pass.set_bind_group(1, &resources.empty_group, &[]);
        pass.set_bind_group(2, &resources.empty_group, &[]);
        // This group omits the probe: sampling an active attachment is invalid.
        pass.set_bind_group(3, &resources.noise_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

fn texture_layout(
    binding: u32,
    dimension: wgpu::TextureViewDimension,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: dimension,
            multisampled: false,
        },
        count: None,
    }
}

fn sampler_layout(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}

fn prepare_clouds(
    mut state: ResMut<CloudRenderState>,
    globals: Res<ForestGlobals>,
    shaders: Res<ForestShaderHandles>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    cache: Res<PipelineCache>,
) {
    if state.resources.is_some() {
        return;
    }
    let Some(globals_buffer) = globals.buffer.as_ref() else {
        return;
    };

    let noise = device
        .wgpu_device()
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("forest_cloud_density_noise"),
            size: wgpu::Extent3d {
                width: NOISE_SIZE,
                height: NOISE_SIZE,
                depth_or_array_layers: NOISE_SIZE,
            },
            mip_level_count: NOISE_MIP_COUNT,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
    // Average each 3D octave into a mip chain. Long, near-horizontal rays
    // otherwise alias the small Worley cells into visible depth slices.
    let mut noise_data = build_noise();
    let mut mip_size = NOISE_SIZE;
    for mip_level in 0..NOISE_MIP_COUNT {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &noise,
                mip_level,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &noise_data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(mip_size * 4),
                rows_per_image: Some(mip_size),
            },
            wgpu::Extent3d {
                width: mip_size,
                height: mip_size,
                depth_or_array_layers: mip_size,
            },
        );
        if mip_size > 1 {
            noise_data = downsample_noise(&noise_data, mip_size);
            mip_size /= 2;
        }
    }
    let noise_view = noise.create_view(&default());
    let repeat = device
        .wgpu_device()
        .create_sampler(&wgpu::SamplerDescriptor {
            label: Some("forest_cloud_noise_repeat"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..default()
        });
    let probe_texture = device
        .wgpu_device()
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("forest_cloud_reflection_probe"),
            size: wgpu::Extent3d {
                width: PROBE_WIDTH,
                height: PROBE_HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
    // Transparent cloud radiance before the first asynchronous pipeline compile.
    let initial: Vec<u16> = (0..PROBE_WIDTH * PROBE_HEIGHT)
        .flat_map(|_| [0, 0, 0, 0x3c00])
        .collect();
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &probe_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&initial),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(PROBE_WIDTH * 8),
            rows_per_image: Some(PROBE_HEIGHT),
        },
        probe_texture.size(),
    );
    let probe = probe_texture.create_view(&default());
    let probe_sampler = device
        .wgpu_device()
        .create_sampler(&wgpu::SamplerDescriptor {
            label: Some("forest_cloud_probe_sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..default()
        });
    let shadow_texture = device
        .wgpu_device()
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("forest_cloud_shadow_map"),
            size: wgpu::Extent3d {
                width: 2 * SHADOW_MAP_LEVEL_TEXELS,
                height: SHADOW_MAP_LEVEL_TEXELS,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: SHADOW_MAP_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
    // -1 marks "not built yet": the shaders march the clouds instead.
    let unbuilt: Vec<f32> = vec![-1.0; (2 * SHADOW_MAP_LEVEL_TEXELS * SHADOW_MAP_LEVEL_TEXELS) as usize];
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &shadow_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&unbuilt),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(2 * SHADOW_MAP_LEVEL_TEXELS * 4),
            rows_per_image: Some(SHADOW_MAP_LEVEL_TEXELS),
        },
        shadow_texture.size(),
    );
    let shadow_map = shadow_texture.create_view(&default());
    let shadow_sampler = device
        .wgpu_device()
        .create_sampler(&wgpu::SamplerDescriptor {
            label: Some("forest_cloud_shadow_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..default()
        });
    let noise_entries = [
        texture_layout(0, wgpu::TextureViewDimension::D3),
        sampler_layout(1),
    ];
    let noise_layout = BindGroupLayoutDescriptor::new("forest_cloud_noise_layout", &noise_entries);
    let layout = BindGroupLayoutDescriptor::new(
        "forest_cloud_resources_layout",
        &[
            noise_entries[0],
            noise_entries[1],
            texture_layout(2, wgpu::TextureViewDimension::D2),
            sampler_layout(3),
            texture_layout(4, wgpu::TextureViewDimension::D2),
            sampler_layout(5),
        ],
    );
    let noise_bindings = [
        wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(&noise_view),
        },
        wgpu::BindGroupEntry {
            binding: 1,
            resource: wgpu::BindingResource::Sampler(&repeat),
        },
    ];
    let noise_group = super::bind_group(
        &device,
        &cache,
        "forest_cloud_noise",
        &noise_layout,
        &noise_bindings,
    );
    let group = super::bind_group(
        &device,
        &cache,
        "forest_cloud_resources",
        &layout,
        &[
            noise_bindings[0].clone(),
            noise_bindings[1].clone(),
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&probe),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Sampler(&probe_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(&shadow_map),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::Sampler(&shadow_sampler),
            },
        ],
    );
    let global_layout = globals_layout();
    let globals_group = super::bind_group(
        &device,
        &cache,
        "forest_cloud_globals",
        &global_layout,
        &super::globals_bind_group_entries(globals_buffer),
    );
    let empty_layout = BindGroupLayoutDescriptor::new("forest_cloud_empty_layout", &[]);
    let empty_group = super::bind_group(
        &device,
        &cache,
        "forest_cloud_empty_group",
        &empty_layout,
        &[],
    );
    let shadow_pipeline = cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("forest_cloud_shadow_map_pipeline".into()),
        layout: vec![
            global_layout.clone(),
            empty_layout.clone(),
            empty_layout.clone(),
            noise_layout.clone(),
        ],
        immediate_size: 0,
        vertex: VertexState {
            shader: shaders.cloud_shadow_map.clone(),
            shader_defs: vec![],
            entry_point: Some("vs_main".into()),
            buffers: vec![],
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(FragmentState {
            shader: shaders.cloud_shadow_map.clone(),
            shader_defs: vec![],
            entry_point: Some("fs_main".into()),
            targets: vec![Some(wgpu::ColorTargetState {
                format: SHADOW_MAP_FORMAT,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        zero_initialize_workgroup_memory: false,
    });
    let pipeline = cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("forest_cloud_probe_pipeline".into()),
        layout: vec![
            global_layout,
            empty_layout.clone(),
            empty_layout,
            noise_layout,
        ],
        immediate_size: 0,
        vertex: VertexState {
            shader: shaders.cloud_probe.clone(),
            shader_defs: vec![],
            entry_point: Some("vs_main".into()),
            buffers: vec![],
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(FragmentState {
            shader: shaders.cloud_probe.clone(),
            shader_defs: vec![],
            entry_point: Some("fs_main".into()),
            targets: vec![Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba16Float,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        zero_initialize_workgroup_memory: false,
    });
    state.resources = Some(CloudGpuResources {
        layout,
        group,
        noise_group,
        globals_group,
        empty_group,
        probe,
        pipeline,
        shadow_map,
        shadow_pipeline,
    });
}

pub fn register_cloud_systems(render_app: &mut bevy::app::SubApp) {
    render_app.init_resource::<CloudRenderState>();
    render_app.init_resource::<CloudProbeSchedule>();
    render_app.add_systems(
        bevy::render::Render,
        (prepare_clouds, update_cloud_probe_schedule)
            .chain()
            .in_set(bevy::render::RenderSystems::Prepare)
            .after(super::prepare_forest_globals),
    );
}

fn hash(x: i32, y: i32, z: i32, seed: u32) -> f32 {
    let mut n = (x as u32).wrapping_mul(0x8da6b343)
        ^ (y as u32).wrapping_mul(0xd8163841)
        ^ (z as u32).wrapping_mul(0xcb1ab31f)
        ^ seed;
    n ^= n >> 16;
    n = n.wrapping_mul(0x7feb352d);
    n ^= n >> 15;
    n = n.wrapping_mul(0x846ca68b);
    n ^= n >> 16;
    (n >> 8) as f32 / 16777215.0
}

fn value_noise(p: [f32; 3], period: i32, seed: u32) -> f32 {
    let cell = p.map(|v| v.floor() as i32);
    let f = p.map(|v| {
        let f = v - v.floor();
        f * f * (3.0 - 2.0 * f)
    });
    let mut result = 0.0;
    for z in 0..2 {
        for y in 0..2 {
            for x in 0..2 {
                let weight = if x == 0 { 1.0 - f[0] } else { f[0] }
                    * if y == 0 { 1.0 - f[1] } else { f[1] }
                    * if z == 0 { 1.0 - f[2] } else { f[2] };
                result += weight
                    * hash(
                        (cell[0] + x).rem_euclid(period),
                        (cell[1] + y).rem_euclid(period),
                        (cell[2] + z).rem_euclid(period),
                        seed,
                    );
            }
        }
    }
    result
}

fn worley(p: [f32; 3], period: i32) -> f32 {
    let cell = p.map(|v| v.floor() as i32);
    let mut nearest = f32::MAX;
    for z in -1..=1 {
        for y in -1..=1 {
            for x in -1..=1 {
                let c = [cell[0] + x, cell[1] + y, cell[2] + z];
                let w = c.map(|v| v.rem_euclid(period));
                let feature = [
                    hash(w[0], w[1], w[2], 71),
                    hash(w[0], w[1], w[2], 113),
                    hash(w[0], w[1], w[2], 173),
                ];
                let distance: f32 = (0..3)
                    .map(|axis| (c[axis] as f32 + feature[axis] - p[axis]).powi(2))
                    .sum();
                nearest = nearest.min(distance);
            }
        }
    }
    (1.0 - nearest.sqrt() * 0.85).clamp(0.0, 1.0)
}

fn build_noise() -> Vec<u8> {
    let mut data = Vec::with_capacity((NOISE_SIZE.pow(3) * 4) as usize);
    for z in 0..NOISE_SIZE {
        for y in 0..NOISE_SIZE {
            for x in 0..NOISE_SIZE {
                let p = [x, y, z].map(|v| (v as f32 + 0.5) / NOISE_SIZE as f32);
                let sample = |frequency: i32, seed| {
                    value_noise(p.map(|v| v * frequency as f32), frequency, seed)
                };
                let shape = 0.58 * sample(4, 31) + 0.27 * sample(8, 37) + 0.15 * sample(16, 43);
                let shape = ((shape - 0.5) * 1.55 + 0.5).clamp(0.0, 1.0);
                let cellular = worley(p.map(|v| v * 8.0), 8);
                let detail = 0.7 * sample(16, 83) + 0.3 * sample(32, 97);
                let weather = ((0.7 * sample(2, 191) + 0.3 * sample(4, 193) - 0.5) * 1.8 + 0.5)
                    .clamp(0.0, 1.0);
                data.extend([shape, cellular, detail, weather].map(|v| (v * 255.0 + 0.5) as u8));
            }
        }
    }
    data
}

fn downsample_noise(source: &[u8], size: u32) -> Vec<u8> {
    let next = size / 2;
    let mut result = vec![0; (next * next * next * 4) as usize];
    for z in 0..next {
        for y in 0..next {
            for x in 0..next {
                for channel in 0..4 {
                    let mut sum = 0u32;
                    for dz in 0..2 {
                        for dy in 0..2 {
                            for dx in 0..2 {
                                let index =
                                    ((((2 * z + dz) * size + 2 * y + dy) * size + 2 * x + dx) * 4
                                        + channel) as usize;
                                sum += source[index] as u32;
                            }
                        }
                    }
                    let index = (((z * next + y) * next + x) * 4 + channel) as usize;
                    result[index] = ((sum + 4) / 8) as u8;
                }
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The map's texture is sized from the Rust constant and addressed from
    /// the WGSL one; they must agree or every lookup lands on the wrong texel.
    #[test]
    fn shadow_map_texel_count_matches_the_shaders() {
        let common = include_str!("../../assets/shaders/cloud-functions.wgslinc");
        let expected = format!(
            "const CLOUD_SHADOW_MAP_TEXELS: f32 = {}.0;",
            SHADOW_MAP_LEVEL_TEXELS
        );
        assert!(common.contains(&expected), "missing `{expected}`");
    }

    #[test]
    fn density_noise_tiles_without_a_seam() {
        for p in [[0.0, 0.0, 0.0], [1.23, -0.43, 7.8], [7.999, 3.2, 1.4]] {
            for axis in 0..3 {
                let mut repeated = p;
                repeated[axis] += 8.0;
                assert!((value_noise(p, 8, 31) - value_noise(repeated, 8, 31)).abs() < 1e-5);
                assert!((worley(p, 8) - worley(repeated, 8)).abs() < 1e-5);
            }
        }
    }

    /// Continuous advection must never wake the probe — it would cancel the
    /// cadence — while scrubbed settings and teleported lighting always do.
    #[test]
    fn probe_reuse_is_latched_only_by_stepwise_inputs() {
        let base = ProbeSkyState {
            sun_direction: [0.9, -0.4, 0.2],
            moon_direction: [0.0, 1.0, 0.1],
            sun_colour: [1.0, 0.95, 0.85, 1.1],
            sun_intensity: 1.0,
            daylight: 0.8,
            moon_intensity: 0.05,
            clouds: [1.0, 0.4, 0.55, 1300.0],
            layer_thickness: 700.0,
            layer_scale: 1.0,
            layer_shadow: 0.35,
            layer_quality: 1.0,
            detail_strength: 1.0,
            march_range: 40000.0,
            climate_bias: 0.3,
            overrides: 0,
            precip_bias: 0.2,
            precip_override: 0,
            wind_radians: 1.1,
            shadow_gate: true,
        };
        let drift = |delta: &dyn Fn(&mut ProbeSkyState)| {
            let mut next = base;
            delta(&mut next);
            base.changed_beyond(&next)
        };
        // Steady wind rotation and a weather easing stay under tolerance.
        assert!(!drift(&|s: &mut ProbeSkyState| s.wind_radians += 0.01));
        assert!(!drift(&|s: &mut ProbeSkyState| s.climate_bias += 0.01));
        assert!(!drift(&|s: &mut ProbeSkyState| s.sun_direction[1] -= 0.005));
        // One probe texel of direction, one user step of a setting: visible.
        assert!(drift(&|s: &mut ProbeSkyState| s.sun_direction[0] += 0.05));
        assert!(drift(&|s: &mut ProbeSkyState| s.clouds[1] += 0.05));
        assert!(drift(&|s: &mut ProbeSkyState| s.layer_quality = 2.0));
        assert!(drift(&|s: &mut ProbeSkyState| s.overrides = 4));
        assert!(drift(&|s: &mut ProbeSkyState| s.sun_colour = [1.0, 0.6, 0.3, 1.1]));
        // Toggles wake the latch exactly, so a re-enabled deck can't pop.
        assert!(drift(&|s: &mut ProbeSkyState| s.clouds[0] = 0.0));
        assert!(drift(&|s: &mut ProbeSkyState| s.detail_strength = 1.4));
        assert!(drift(&|s: &mut ProbeSkyState| s.precip_override = 2));
        // The shadow map's gate flipping rebuilds it on the first usable
        // frame instead of reading content from before the gate went dark.
        assert!(drift(&|s: &mut ProbeSkyState| s.shadow_gate = false));
    }
}
