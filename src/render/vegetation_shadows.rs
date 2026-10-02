//! Shadows cast by the scattered plants.
//!
//! The terrain and the clouds shadow the scene by ray marching in the
//! composite; a forest needs more than that. The plants are drawn depth-only,
//! from the sun (or, at night, the moon), into four cascades of one depth
//! array: each cascade covers a slice of the view frustum, nearest the
//! sharpest. The composite then looks each lit pixel up in its slice's
//! cascade and darkens the direct light it receives, so trunks and crowns
//! shade the ground, the grass, each other and themselves.
//!
//! The cascades are stabilised: each is a fixed-size square in light space
//! around the bounding sphere of its frustum slice, moved only in whole
//! texels, so shadows neither swim when the camera turns nor shimmer as it
//! walks. This module fits them on the CPU every frame and owns the depth
//! array, which the post passes bind when they build the composite's group.

use bevy::prelude::*;
use bevy::render::renderer::RenderDevice;

pub const SHADOW_CASCADES: usize = 4;
/// Texels along each side of a cascade.
pub const SHADOW_RESOLUTION: u32 = 2048;
/// View-depth bounds of the cascades, metres: about 2.7 cm, 9 cm, 38 cm and
/// 2.7 m per texel at the default field of view. The last reaches as far as
/// the trees are drawn, so no distant forest floor shows sunlit through the
/// canopy.
pub const SHADOW_SPLITS: [f32; SHADOW_CASCADES + 1] = [0.0, 20.0, 70.0, 280.0, 2000.0];
/// How far up-light of a cascade's slice a plant can stand and still be
/// drawn into it: a 35 m pine's shadow under a sun 3 degrees up.
pub const CASTER_REACH: f32 = 680.0;
/// Share of each cascade, at its far end, over which it blends into the next.
pub const CASCADE_BLEND: f32 = 0.12;
/// Format of the depth array.
pub const SHADOW_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

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

/// The smallest sphere around the frustum slice between view depths `near`
/// and `far`, as (distance of its centre along the view axis, radius), for a
/// frustum whose corner rays leave the axis at `corner` (the hypotenuse of the
/// two half-angle tangents).
pub fn slice_sphere(near: f32, far: f32, corner: f32) -> (f32, f32) {
    let (n, f, k2) = (near as f64, far as f64, (corner as f64).powi(2));
    let centre = 0.5 * (f + n) * (1.0 + k2);
    if centre >= f {
        (far, (f * k2.sqrt()) as f32)
    } else {
        let radius = ((f - centre).powi(2) + f * f * k2).sqrt();
        (centre as f32, radius as f32)
    }
}

/// Fits the four cascades for a camera (column-major world-to-view `view`
/// and the reverse-Z `projection` of `crate::matrices::perspective`) and a
/// light travelling along `light`.
pub fn fit_cascades(
    view: &[f32; 16],
    projection: &[f32; 16],
    eye: [f32; 3],
    light: &LightBasis,
) -> [Cascade; SHADOW_CASCADES] {
    let tan_x = 1.0 / projection[0].abs().max(1e-6);
    let tan_y = 1.0 / projection[5].abs().max(1e-6);
    let corner = tan_x.hypot(tan_y);
    // The camera looks down its view -Z: the third row of the rotation.
    let forward = normalize([-view[2] as f64, -view[6] as f64, -view[10] as f64]);
    let (right, up, along) = (widen(light.right), widen(light.up), widen(light.forward));
    std::array::from_fn(|index| {
        let (depth, radius) = slice_sphere(SHADOW_SPLITS[index], SHADOW_SPLITS[index + 1], corner);
        // A whole number of metres, so the box only changes size when the
        // field of view does.
        let half_extent = (radius as f64 * 1.01).ceil();
        let texel = 2.0 * half_extent / SHADOW_RESOLUTION as f64;
        let centre: [f64; 3] = std::array::from_fn(|axis| eye[axis] as f64 + forward[axis] * depth as f64);
        // Whole texels across the light, so the rasterised shadow edges stay
        // put while the camera moves.
        let x = (dot(centre, right) / texel).round() * texel;
        let y = (dot(centre, up) / texel).round() * texel;
        let z = dot(centre, along);
        let near = z - half_extent - CASTER_REACH as f64;
        let far = z + half_extent;
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
    })
}

/// `VegetationShadows` in composite.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShadowUniform {
    pub view_projection: [[f32; 16]; SHADOW_CASCADES],
    /// Far view depth of each cascade.
    pub splits: [f32; 4],
    /// World metres per texel of each cascade.
    pub texel: [f32; 4],
    /// xyz the direction the shadowing light travels; w 0 off, 1 sun, 2 moon.
    pub light: [f32; 4],
    /// x blend band, y strength, z texels per side, w unused.
    pub params: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<ShadowUniform>() == 320);

impl ShadowUniform {
    pub fn disabled() -> Self {
        Self {
            params: [CASCADE_BLEND, 0.0, SHADOW_RESOLUTION as f32, 0.0],
            ..Default::default()
        }
    }

    pub fn new(cascades: &[Cascade; SHADOW_CASCADES], light: &LightBasis, moon: bool) -> Self {
        Self {
            view_projection: cascades.map(|cascade| cascade.view_projection),
            splits: std::array::from_fn(|index| SHADOW_SPLITS[index + 1]),
            texel: cascades.map(|cascade| cascade.texel),
            light: [
                light.forward[0],
                light.forward[1],
                light.forward[2],
                if moon { 2.0 } else { 1.0 },
            ],
            params: [CASCADE_BLEND, 1.0, SHADOW_RESOLUTION as f32, 0.0],
        }
    }
}

/// The depth array, its per-cascade attachment views, the comparison sampler
/// and the composite's uniform.
pub struct ShadowTargets {
    pub resolution: u32,
    pub array_view: wgpu::TextureView,
    pub layer_views: Vec<wgpu::TextureView>,
    pub sampler: wgpu::Sampler,
    pub uniform: wgpu::Buffer,
}

/// Created once, before the post passes build the composite's bind group.
#[derive(Resource, Default)]
pub struct VegetationShadowMaps {
    pub targets: Option<ShadowTargets>,
}

fn create_targets(device: &RenderDevice, resolution: u32) -> ShadowTargets {
    let device = device.wgpu_device();
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("vegetation_shadow_cascades"),
        size: wgpu::Extent3d {
            width: resolution,
            height: resolution,
            depth_or_array_layers: SHADOW_CASCADES as u32,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: SHADOW_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let array_view = texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("vegetation_shadow_array"),
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    });
    let layer_views = (0..SHADOW_CASCADES as u32)
        .map(|layer| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("vegetation_shadow_layer"),
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: layer,
                array_layer_count: Some(1),
                ..Default::default()
            })
        })
        .collect();
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("vegetation_shadow_sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        compare: Some(wgpu::CompareFunction::LessEqual),
        ..Default::default()
    });
    let uniform = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vegetation_shadow_uniform"),
        size: std::mem::size_of::<ShadowUniform>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    ShadowTargets {
        resolution,
        array_view,
        layer_views,
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
    let resolution = if extracted.enabled() { SHADOW_RESOLUTION } else { 1 };
    let targets = create_targets(&device, resolution);
    queue.write_buffer(&targets.uniform, 0, bytemuck::bytes_of(&ShadowUniform::disabled()));
    maps.targets = Some(targets);
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::math::Vec3;

    fn transform(m: &[f32; 16], p: [f32; 3]) -> [f32; 4] {
        std::array::from_fn(|row| m[row] * p[0] + m[4 + row] * p[1] + m[8 + row] * p[2] + m[12 + row])
    }

    fn camera(eye: Vec3, look: Vec3) -> ([f32; 16], [f32; 16]) {
        (
            crate::matrices::view_matrix(eye, eye + look, Vec3::Y),
            crate::matrices::perspective(68.0, 16.0 / 9.0, 0.1, 5800.0),
        )
    }

    #[test]
    fn the_slice_sphere_holds_the_slice() {
        for (near, far) in [(0.0, 20.0), (20.0, 70.0), (250.0, 1200.0), (5.0, 6.0)] {
            let corner = 1.376;
            let (centre, radius) = slice_sphere(near, far, corner);
            for depth in [near, far] {
                let off_axis = depth * corner;
                let reach = ((depth - centre).powi(2) + off_axis * off_axis).sqrt();
                assert!(reach <= radius * 1.0001, "slice {near}..{far}: {reach} > {radius}");
            }
        }
    }

    #[test]
    fn every_slice_corner_lands_inside_its_cascade() {
        let eye = Vec3::new(812.5, 64.0, -2210.25);
        let look = Vec3::new(0.3, -0.25, -1.0).normalize();
        let (view, projection) = camera(eye, look);
        let light = light_basis([-0.22, -0.62, 0.76]);
        let cascades = fit_cascades(&view, &projection, eye.to_array(), &light);
        let inverse = crate::matrices::invert_affine(&view);
        let (tan_x, tan_y) = (1.0 / projection[0], 1.0 / projection[5]);
        for (index, cascade) in cascades.iter().enumerate() {
            for depth in [SHADOW_SPLITS[index].max(0.1), SHADOW_SPLITS[index + 1]] {
                for (sx, sy) in [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0), (0.0, 0.0)] {
                    let view_point = [sx * tan_x * depth, sy * tan_y * depth, -depth];
                    let world = transform(&inverse, view_point);
                    let clip = transform(&cascade.view_projection, [world[0], world[1], world[2]]);
                    assert!(clip[0].abs() <= 1.0 && clip[1].abs() <= 1.0, "cascade {index}: {clip:?}");
                    assert!((0.0..=1.0).contains(&clip[2]), "cascade {index} depth {}", clip[2]);
                    // Casters up to the reach toward the light still fit.
                    let caster: [f32; 3] =
                        std::array::from_fn(|axis| world[axis] - light.forward[axis] * (CASTER_REACH - 1.0));
                    let caster_clip = transform(&cascade.view_projection, caster);
                    assert!(caster_clip[2] >= 0.0, "cascade {index} caster depth {}", caster_clip[2]);
                }
            }
        }
    }

    #[test]
    fn cascades_move_in_whole_texels() {
        let look = Vec3::new(-0.6, -0.1, 0.8).normalize();
        let light = light_basis([0.4, -0.5, 0.3]);
        let (view_a, projection) = camera(Vec3::new(10.0, 30.0, 10.0), look);
        let (view_b, _) = camera(Vec3::new(10.37, 30.0, 9.81), look);
        let a = fit_cascades(&view_a, &projection, [10.0, 30.0, 10.0], &light);
        let b = fit_cascades(&view_b, &projection, [10.37, 30.0, 9.81], &light);
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
