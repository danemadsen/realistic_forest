//! Shadows cast by the scattered plants.
//!
//! The terrain and the clouds shadow the scene by ray marching in the
//! composite; a forest needs more than that. The plants are drawn depth-only,
//! from the sun (or, at night, the moon), into three separate depth
//! textures: each cascade covers a horizontal radius around the camera,
//! nearest the sharpest. The composite then looks each lit pixel up in its
//! cascade and darkens the direct light it receives, so trunks and crowns
//! shade the ground, the grass, each other and themselves.
//!
//! The cascades are stabilised: each is a metre-rounded square in light space
//! around the terrain and crowns in that radius, moved only in whole
//! texels, so shadows neither swim when the camera turns nor shimmer as it
//! walks or flies vertically. This module fits them on the CPU and owns the depth
//! textures, which the post passes bind when they build the composite's group.

use bevy::prelude::*;
use bevy::render::renderer::RenderDevice;

pub const SHADOW_CASCADES: usize = 3;
/// Texels along each side of each cascade. Only the distant forest uses a
/// smaller depth target.
pub const SHADOW_RESOLUTIONS: [u32; SHADOW_CASCADES] = [2048, 2048, 1536];
/// Horizontal-distance bounds of the cascades, metres. Keep detailed shadows
/// across the nearby forest before switching to the 2 km map; its larger
/// texels are only suitable for distant canopy silhouettes.
pub const SHADOW_SPLITS: [f32; SHADOW_CASCADES + 1] = [0.0, 50.0, 400.0, 2000.0];
/// How far up-light of a cascade's receivers a plant can stand and still be
/// drawn into it: a 35 m pine's shadow under a sun 3 degrees up.
pub const CASTER_REACH: f32 = 680.0;
/// Share of each cascade, at its far end, over which it blends into the next.
pub const CASCADE_BLEND: f32 = 0.25;
/// Keep the 1793.6..2000 m terminal fade independent of the cascade blends.
pub const SHADOW_FADE_BAND: f32 = 206.4;
/// Format of the depth targets.
pub const SHADOW_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
/// Room above the sampled ground for the tallest crowns, including wind.
const CANOPY_HEIGHT: f32 = 50.0;
/// Local terrain detail and differences while erosion tiles are revealing.
const TERRAIN_SLACK: f32 = 12.0;
/// The light basis projects along +Z, with right × up = forward. A face
/// pointing toward the light therefore winds clockwise in shadow clip XY.
pub const SHADOW_FRONT_FACE: wgpu::FrontFace = wgpu::FrontFace::Cw;

/// An orthonormal frame looking along the light: `forward` is the direction
/// the light travels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LightBasis {
    pub right: [f32; 3],
    pub up: [f32; 3],
    pub forward: [f32; 3],
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn normalize(v: [f64; 3]) -> [f64; 3] {
    let length = dot(v, v).sqrt().max(1e-12);
    v.map(|c| c / length)
}

fn widen(v: [f32; 3]) -> [f64; 3] {
    v.map(f64::from)
}

fn narrow(v: [f64; 3]) -> [f32; 3] {
    v.map(|c| c as f32)
}

/// The light frame for a light travelling along `direction`. Its sideways
/// axis stays level, so a slowly moving sun turns the cascades smoothly.
pub fn light_basis(direction: [f32; 3]) -> LightBasis {
    let forward = normalize(widen(direction));
    let reference = if forward[1].abs() > 0.995 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
    let right = normalize(cross(reference, forward));
    let up = cross(forward, right);
    LightBasis {
        right: narrow(right),
        up: narrow(up),
        forward: narrow(forward),
    }
}

/// One cascade's box in light space and its drawing transform.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cascade {
    /// Centre of the square across the light, in light-frame coordinates.
    pub centre: [f32; 2],
    /// Half the square's side, metres.
    pub half_extent: f32,
    /// Light-frame depth (along `forward`) of the box's near and far faces.
    pub near: f32,
    pub far: f32,
    /// World metres per texel.
    pub texel: f32,
    /// World to clip, column-major; depth 0 at the near face, 1 at the far.
    pub view_projection: [f32; 16],
}

/// Receiver heights for nested ground discs, including crowns and water.
/// Sample a world-anchored grid around each disc, with spacing-dependent
/// padding for terrain between samples. Quantise outwards so small height
/// changes do not continually resize the maps. Neither camera height nor
/// orientation participates in this fit.
pub fn receiver_height_bounds(
    eye: [f32; 3],
    mut ground_height: impl FnMut(f32, f32) -> f32,
) -> [[f32; 2]; SHADOW_CASCADES] {
    let mut inner = [f32::INFINITY, f32::NEG_INFINITY];
    let bounds_for_cascade = |index: usize| {
        let step = SHADOW_SPLITS[index + 1] / 4.0;
        let anchor = [(eye[0] / step).floor() * step, (eye[2] / step).floor() * step];
        let mut low = f32::INFINITY;
        let mut high = f32::NEG_INFINITY;
        // The extra positive row/column covers the disc even when the eye
        // lies just short of the next grid point.
        for z in -4..=5 {
            for x in -4..=5 {
                let height = ground_height(anchor[0] + x as f32 * step, anchor[1] + z as f32 * step)
                    .max(crate::constants::SEA_LEVEL);
                low = low.min(height);
                high = high.max(height);
            }
        }
        let padding = TERRAIN_SLACK + step;
        low = ((low - padding) / 8.0).floor() * 8.0;
        high = ((high + padding + CANOPY_HEIGHT) / 8.0).ceil() * 8.0;
        // A coarser grid must never exclude the inner cascade's receivers:
        // both maps are sampled throughout the blend band.
        inner = [low.min(inner[0]), high.max(inner[1])];
        inner
    };
    std::array::from_fn(bounds_for_cascade)
}

/// Fit nested vertical cylinders to the terrain below the camera. Their
/// horizontal radii match the shaders' distance selection, so even a view
/// straight down from high altitude keeps the nearby ground in the detailed
/// maps. The larger cylinders also cover every preceding blend band.
pub fn fit_cascades(
    eye: [f32; 3],
    heights: &[[f32; 2]; SHADOW_CASCADES],
    light: &LightBasis,
) -> [Cascade; SHADOW_CASCADES] {
    let (right, up, along) = (widen(light.right), widen(light.up), widen(light.forward));
    let cascade_for_index = |index: usize| {
        let [low, high] = heights[index].map(f64::from);
        let radius = SHADOW_SPLITS[index + 1] as f64;
        let half_height = (high - low) * 0.5;
        // Exact support of a vertical cylinder along a light-frame axis.
        let support = |axis: [f64; 3]| radius * axis[0].hypot(axis[2]) + half_height * axis[1].abs();
        // Leave room for texel snapping, PCF and the receiver's normal bias.
        let half_extent = (support(right).max(support(up)) * 1.01).ceil();
        let texel = 2.0 * half_extent / SHADOW_RESOLUTIONS[index] as f64;
        let centre = [eye[0] as f64, (low + high) * 0.5, eye[2] as f64];
        // Whole texels across the light, so the rasterised shadow edges stay
        // put while the camera moves.
        let x = (dot(centre, right) / texel).round() * texel;
        let y = (dot(centre, up) / texel).round() * texel;
        let z = dot(centre, along);
        let depth_extent = support(along) + texel * 4.0;
        let near = z - depth_extent - CASTER_REACH as f64;
        let far = z + depth_extent;
        let span = far - near;
        let s = 1.0 / half_extent;
        let view_projection = [
            right[0] * s, up[0] * s, along[0] / span, 0.0,
            right[1] * s, up[1] * s, along[1] / span, 0.0,
            right[2] * s, up[2] * s, along[2] / span, 0.0,
            -x * s, -y * s, -near / span, 1.0,
        ]
        .map(|value| value as f32);
        Cascade {
            centre: [x as f32, y as f32],
            half_extent: half_extent as f32,
            near: near as f32,
            far: far as f32,
            texel: texel as f32,
            view_projection,
        }
    };
    std::array::from_fn(cascade_for_index)
}

/// `VegetationShadows` in composite.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShadowUniform {
    pub view_projection: [[f32; 16]; SHADOW_CASCADES],
    /// Far horizontal distances; w repeats the last split for the shader cutoff.
    pub splits: [f32; 4],
    /// World metres per texel of each cascade; w is unused.
    pub texel: [f32; 4],
    /// xyz the direction the shadowing light travels; w 0 off, 1 sun, 2 moon.
    pub light: [f32; 4],
    /// x blend fraction, y strength, z terminal fade band in metres, w unused.
    pub params: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<ShadowUniform>() == 256);

impl ShadowUniform {
    pub fn disabled() -> Self {
        Self {
            params: [CASCADE_BLEND, 0.0, SHADOW_FADE_BAND, 0.0],
            ..Default::default()
        }
    }

    pub fn new(cascades: &[Cascade; SHADOW_CASCADES], light: &LightBasis, moon: bool) -> Self {
        Self {
            view_projection: cascades.map(|cascade| cascade.view_projection),
            splits: std::array::from_fn(|index| SHADOW_SPLITS[(index + 1).min(SHADOW_CASCADES)]),
            texel: std::array::from_fn(|index| cascades.get(index).map_or(0.0, |cascade| cascade.texel)),
            light: [
                light.forward[0],
                light.forward[1],
                light.forward[2],
                if moon { 2.0 } else { 1.0 },
            ],
            params: [CASCADE_BLEND, 1.0, SHADOW_FADE_BAND, 0.0],
        }
    }
}

/// The per-cascade depth views (both attachments and sampled textures), comparison sampler
/// and the composite's uniform.
pub struct ShadowTargets {
    pub resolutions: [u32; SHADOW_CASCADES],
    pub cascade_views: [wgpu::TextureView; SHADOW_CASCADES],
    pub sampler: wgpu::Sampler,
    pub uniform: wgpu::Buffer,
}

/// Created once, before the post passes build the composite's bind group.
#[derive(Resource, Default)]
pub struct VegetationShadowMaps {
    pub targets: Option<ShadowTargets>,
}

fn create_targets(device: &RenderDevice, resolutions: [u32; SHADOW_CASCADES]) -> ShadowTargets {
    let device = device.wgpu_device();
    // Texture-array layers must share an extent. Separate textures let the
    // distant cascade shrink without rescaling the near maps or their UVs.
    let build_cascade_view = |resolution: u32| {
        let texture_descriptor = wgpu::TextureDescriptor {
            label: Some("vegetation_shadow_cascade"),
            size: wgpu::Extent3d {
                width: resolution,
                height: resolution,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: SHADOW_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };
        let texture = device.create_texture(&texture_descriptor);
        texture.create_view(&wgpu::TextureViewDescriptor::default())
    };
    let cascade_views = resolutions.map(build_cascade_view);
    let sampler_descriptor = wgpu::SamplerDescriptor {
        label: Some("vegetation_shadow_sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        compare: Some(wgpu::CompareFunction::LessEqual),
        ..Default::default()
    };
    let sampler = device.create_sampler(&sampler_descriptor);
    let uniform_descriptor = wgpu::BufferDescriptor {
        label: Some("vegetation_shadow_uniform"),
        size: std::mem::size_of::<ShadowUniform>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    };
    let uniform = device.create_buffer(&uniform_descriptor);
    ShadowTargets {
        resolutions,
        cascade_views,
        sampler,
        uniform,
    }
}

/// Creates the targets on the first frame: full size when the scatter is on,
/// a single texel per cascade when it was disabled at startup.
pub fn prepare_vegetation_shadow_maps(
    device: Res<RenderDevice>,
    queue: Res<bevy::render::renderer::RenderQueue>,
    extracted: Res<super::vegetation_node::ExtractedVegetation>,
    mut maps: ResMut<VegetationShadowMaps>,
) {
    if maps.targets.is_some() {
        return;
    }
    let resolutions = if extracted.enabled() { SHADOW_RESOLUTIONS } else { [1; SHADOW_CASCADES] };
    let targets = create_targets(&device, resolutions);
    let initial_uniform = ShadowUniform::disabled();
    queue.write_buffer(&targets.uniform, 0, bytemuck::bytes_of(&initial_uniform));
    maps.targets = Some(targets);
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::math::Vec3;

    fn transform(m: &[f32; 16], p: [f32; 3]) -> [f32; 4] {
        std::array::from_fn(|row| m[row] * p[0] + m[4 + row] * p[1] + m[8 + row] * p[2] + m[12 + row])
    }

    fn assert_inside(cascade: &Cascade, point: [f32; 3]) {
        let clip = transform(&cascade.view_projection, point);
        assert!(clip[0].abs() <= 1.0 && clip[1].abs() <= 1.0, "{point:?}: {clip:?}");
        assert!((0.0..=1.0).contains(&clip[2]), "{point:?}: depth {}", clip[2]);
    }

    #[test]
    fn extended_detail_ranges_keep_the_terminal_fade_and_texture_budget() {
        let eye = [10.0, 30.0, 10.0];
        let heights = receiver_height_bounds(eye, |_, _| 14.0);
        let light = light_basis([0.4, -0.5, 0.3]);
        let cascades = fit_cascades(eye, &heights, &light);
        let uniform = ShadowUniform::new(&cascades, &light, false);

        // Wider crossfades must not bring the terminal fade closer as well.
        assert_eq!(uniform.splits, [50.0, 400.0, 2000.0, 2000.0]);
        assert_eq!(uniform.params[0], 0.25);
        assert_eq!(uniform.params[2], 206.4);
        assert_eq!(uniform.splits[3] - uniform.params[2], 1793.6);
        assert_eq!(uniform.texel[3], 0.0);
        assert_eq!(SHADOW_RESOLUTIONS, [2048, 2048, 1536]);
        for index in 0..SHADOW_CASCADES {
            assert_eq!(uniform.texel[index], cascades[index].texel);
        }
    }

    #[test]
    fn light_facing_triangles_keep_the_shadow_front_face() {
        let eye = [812.5, 64.0, -2210.25];
        let heights = receiver_height_bounds(eye, |_, _| 35.0);
        let projection = crate::matrices::perspective(68.0, 16.0 / 9.0, 0.1, 5800.0);
        let signed_area = |matrix: &[f32; 16], triangle: [Vec3; 3]| {
            let projected = triangle.map(|vertex| {
                let clip = transform(matrix, vertex.to_array());
                Vec2::new(clip[0], clip[1]) / clip[3]
            });
            (projected[1] - projected[0]).perp_dot(projected[2] - projected[0])
        };
        // Include the alternate reference axis used for overhead sunlight,
        // either horizon direction, and a light coming from below the view.
        for direction in [
            [-0.22, -0.62, 0.76],
            [0.4, -0.5, 0.3],
            [0.0, -1.0, 0.0],
            [0.001, -1.0, -0.001],
            [0.0, 1.0, 0.0],
            [0.6, 0.7, -0.2],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
        ] {
            let light = light_basis(direction);
            let toward_light = -Vec3::from_array(light.forward);
            let reference = if toward_light.y.abs() > 0.9 { Vec3::X } else { Vec3::Y };
            let tangent = reference.cross(toward_light).normalize();
            let bitangent = toward_light.cross(tangent);
            for cascade in fit_cascades(eye, &heights, &light) {
                let centre = Vec3::from_array(light.right) * cascade.centre[0]
                    + Vec3::from_array(light.up) * cascade.centre[1]
                    + Vec3::from_array(light.forward) * (0.5 * (cascade.near + cascade.far));
                let size = cascade.half_extent * 0.125;
                let triangle = [
                    centre - (tangent + bitangent) * size,
                    centre + (tangent - bitangent) * size,
                    centre + bitangent * size,
                ];
                // This is the authored front: its geometric normal faces
                // the light source, before either projection transforms it.
                assert!((triangle[1] - triangle[0]).cross(triangle[2] - triangle[0]).dot(toward_light) > 0.0);
                let area = signed_area(&cascade.view_projection, triangle);
                assert!(area.abs() > 1e-4, "degenerate projected triangle for {direction:?}");
                let front_face = if area > 0.0 { wgpu::FrontFace::Ccw } else { wgpu::FrontFace::Cw };
                assert_eq!(front_face, SHADOW_FRONT_FACE, "light {direction:?}");

                // The ordinary camera looks down -Z, so the same front
                // winds the other way when seen from the light's position.
                let camera_view = crate::matrices::view_matrix(
                    centre + toward_light * (size * 4.0), centre, reference,
                );
                let camera_clip = crate::matrices::mul_m4(&projection, &camera_view);
                assert!(signed_area(&camera_clip, triangle) > 0.0, "camera winding for {direction:?}");
            }
        }
    }

    #[test]
    fn ground_and_crowns_keep_the_same_maps_at_every_camera_height() {
        let ground = |x: f32, z: f32| 160.0 + 0.35 * x - 0.23 * z;
        let eye = [812.5, 64.0, -2210.25];
        let heights = receiver_height_bounds(eye, ground);
        let light = light_basis([-0.22, -0.62, 0.76]);
        let cascades = fit_cascades(eye, &heights, &light);
        for altitude in [-500.0, 1600.0, 20_000.0] {
            let elevated = [eye[0], altitude, eye[2]];
            let elevated_heights = receiver_height_bounds(elevated, ground);
            assert_eq!(elevated_heights, heights);
            assert_eq!(fit_cascades(elevated, &elevated_heights, &light), cascades);
        }
        for (index, cascade) in cascades.iter().enumerate() {
            let radius = SHADOW_SPLITS[index + 1];
            for spoke in 0..64 {
                let angle = std::f32::consts::TAU * spoke as f32 / 64.0;
                let x = eye[0] + radius * angle.cos();
                let z = eye[2] + radius * angle.sin();
                let floor = ground(x, z).max(crate::constants::SEA_LEVEL);
                assert!(heights[index][0] <= floor);
                assert!(heights[index][1] >= floor + CANOPY_HEIGHT);
                assert_inside(cascade, [x, floor, z]);
                assert_inside(cascade, [x, floor + CANOPY_HEIGHT, z]);
            }
        }
        // Directly below an airborne camera must use the first map even
        // when the receiver is kilometres below the original eye position.
        let floor = ground(eye[0], eye[2]);
        assert_inside(&cascades[0], [eye[0], floor, eye[2]]);
        assert_inside(&cascades[0], [eye[0], floor + CANOPY_HEIGHT, eye[2]]);
    }

    #[test]
    fn cylinder_boundaries_and_up_light_casters_fit_for_every_light_direction() {
        let eye = [812.5, 2000.0, -2210.25];
        let heights = [[-24.0, 96.0], [-80.0, 160.0], [-520.0, 700.0]];
        for direction in [
            [-0.22, -0.62, 0.76],
            [0.0, -1.0, 0.0],
            [0.001, -1.0, -0.001],
            [0.0, 1.0, 0.0],
            [0.6, 0.7, -0.2],
            [0.0, -0.0523, -0.9986],
            [0.0, 0.0, 1.0],
        ] {
            let light = light_basis(direction);
            let cascades = fit_cascades(eye, &heights, &light);
            for (index, cascade) in cascades.iter().enumerate() {
                let radius = SHADOW_SPLITS[index + 1];
                for height in heights[index] {
                    for spoke in 0..64 {
                        let angle = std::f32::consts::TAU * spoke as f32 / 64.0;
                        let receiver = [eye[0] + radius * angle.cos(), height, eye[2] + radius * angle.sin()];
                        assert_inside(cascade, receiver);
                        let caster = std::array::from_fn(|axis| receiver[axis] - light.forward[axis] * CASTER_REACH);
                        assert_inside(cascade, caster);
                    }
                }
            }
        }
    }

    #[test]
    fn coarse_height_samples_preserve_inner_hills_and_blend_coverage() {
        let eye = [10.0, 4000.0, 10.0];
        // Only the fine grid samples this narrow hill. Its crowns must
        // remain covered when transitioning into a coarser cascade.
        let heights = receiver_height_bounds(eye, |x, z| {
            if (x - 12.5).abs() < 0.5 && (z - 12.5).abs() < 0.5 { 900.0 } else { 24.0 }
        });
        assert!(heights[0][1] >= 900.0 + CANOPY_HEIGHT);
        for [low, high] in heights {
            assert_eq!(low % 8.0, 0.0);
            assert_eq!(high % 8.0, 0.0);
        }
        let light = light_basis([0.4, -0.5, 0.3]);
        let cascades = fit_cascades(eye, &heights, &light);
        for index in 0..SHADOW_CASCADES - 1 {
            assert!(heights[index + 1][0] <= heights[index][0]);
            assert!(heights[index + 1][1] >= heights[index][1]);
            let end = SHADOW_SPLITS[index + 1];
            let start = end - (end - SHADOW_SPLITS[index]) * CASCADE_BLEND;
            for radius in [start, (start + end) * 0.5, end] {
                for height in heights[index] {
                    for spoke in 0..32 {
                        let angle = std::f32::consts::TAU * spoke as f32 / 32.0;
                        let point = [eye[0] + radius * angle.cos(), height, eye[2] + radius * angle.sin()];
                        assert_inside(&cascades[index], point);
                        assert_inside(&cascades[index + 1], point);
                    }
                }
            }
        }
    }

    #[test]
    fn underwater_ground_keeps_water_receivers_and_heights_stable_between_grid_steps() {
        let eye_a = [10.1, 150.0, 10.2];
        let eye_b = [10.8, 150.0, 10.9];
        let ground = |x: f32, z: f32| -300.0 + x * 0.01 + z * 0.01;
        let a = receiver_height_bounds(eye_a, ground);
        let b = receiver_height_bounds(eye_b, ground);
        assert_eq!(a, b);
        for [low, high] in a {
            assert!(low <= crate::constants::SEA_LEVEL);
            assert!(high >= crate::constants::SEA_LEVEL + CANOPY_HEIGHT);
        }
    }

    #[test]
    fn cascades_move_in_whole_texels() {
        let light = light_basis([0.4, -0.5, 0.3]);
        let eye_a = [10.0, 30.0, 10.0];
        let eye_b = [10.37, 30.0, 9.81];
        let heights_a = receiver_height_bounds(eye_a, |_, _| 14.0);
        let heights_b = receiver_height_bounds(eye_b, |_, _| 14.0);
        let a = fit_cascades(eye_a, &heights_a, &light);
        let b = fit_cascades(eye_b, &heights_b, &light);
        for (a, b) in a.iter().zip(&b) {
            assert_eq!(a.half_extent, b.half_extent);
            for axis in 0..2 {
                let steps = (a.centre[axis] - b.centre[axis]) / a.texel;
                assert!((steps - steps.round()).abs() < 1e-2, "moved {steps} texels");
            }
        }
        // The light frame is orthonormal and its sideways axis level.
        assert!(light.right[1].abs() < 1e-6);
        let d = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        assert!(d(light.right, light.up).abs() < 1e-6 && d(light.up, light.forward).abs() < 1e-6);
        assert!((d(light.up, light.up) - 1.0).abs() < 1e-6);
    }
}
