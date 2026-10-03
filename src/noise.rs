//! CPU terrain generation, ported from the C++ `main.cpp` noise functions.
//!
//! The 1024x1024 noise field is generated once by quick-noise (standing in
//! for FastNoiseLite's OpenSimplex2S + FBm, evaluated at the same integer
//! texel coordinates), then every world-space query bilinearly resamples that
//! field with the exact mirror-fold semantics the GPU shader uses. The GPU
//! never evaluates the noise algorithm; it samples the uploaded field
//! texture, so both sides agree by construction.
//!
//! Simplex and OpenSimplex2S are different algorithms, so the two fields are
//! statistically equivalent realizations rather than the same realization.
//! Over the full field the marginals match closely (mean +0.00098 vs +0.00262,
//! sd 0.25672 vs 0.25045, range [-0.7158, 0.7023] vs [-0.7889, 0.7622]) and the
//! pointwise correlation is +0.014; coarse-grained sd agrees within 2 percent
//! for windows of 8 to 256 metres. Everything downstream is therefore a
//! different landscape, not a different formula: `base_height_cpu` fed the
//! C++'s own FastNoiseLite field reproduces the C++ probe grid to 0.01 m over
//! 4225 points, and fed this field it reproduces the port's grid to the same
//! 0.01 m. Across 40 FastNoiseLite seeds at identical parameters the height
//! field's sd spans 39-88 m and its cross-seed correlation spans -0.37 to
//! +1.0, a range that contains this port's terrain. See
//! `src/bin/noise-compare.rs` to dump the field for such a comparison.

use bevy::prelude::Resource;
use crate::constants::*;
use quick_noise::{BatchNoise, Fbm, Grid, Simplex};
use quick_noise::simd::StaticArch;

#[derive(Clone, Resource)]
pub struct NoiseField {
    /// R-channel texel data in [0, 1], row-major with x fastest
    /// (`z * NOISE_RESOLUTION + x`).
    pub samples: Vec<f32>,
}

impl NoiseField {
    /// The one field the whole app shares: the C++ built a single `NoiseField`
    /// in `main()` and handed it to the terrain setup, the erosion setup and
    /// the diagnostics window alike.
    pub fn new() -> Self {
        Self { samples: generate_noise_samples() }
    }
}

impl Default for NoiseField {
    fn default() -> Self {
        Self::new()
    }
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + t * (b - a)
}

fn smoothstep(edge0: f32, edge1: f32, value: f32) -> f32 {
    let t = ((value - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// CPU-side sample generation, shared by the noise texture upload and by the
/// windowless probe mode. Mirrors `GenerateNoiseSamples` exactly:
/// quasiperiodic FBm Simplex at integer texel coordinates, remapped to [0, 1].
pub fn generate_noise_samples() -> Vec<f32> {
    let mut samples = vec![0.0f32; NOISE_RESOLUTION * NOISE_RESOLUTION];
    let grid = Grid::<2, StaticArch>::new(NOISE_RESOLUTION, NOISE_RESOLUTION);
    BatchNoise::<2, Fbm, Simplex>::builder(grid.x_iter(), grid.y_iter())
        .seed(1337)
        .octaves(5)
        .frequency(0.0085)
        .lacunarity(2.02)
        .persistence(0.5)
        .fill(samples.as_mut_slice());
    for sample in samples.iter_mut() {
        *sample = *sample * 0.5 + 0.5;
    }
    samples
}

/// Bilinearly sample the periodic noise field at a world position,
/// mirror-folding the phase exactly like `signedNoise` in terrain.wgsl: the
/// R32 field is one non-periodic noise patch, so a plain repeat would tear
/// between unrelated edge texels at every period seam.
pub fn sample_periodic_noise(data: &[f32], world_x: f32, world_z: f32) -> f32 {
    let normalized_x = world_x / NOISE_PERIOD + 0.5;
    let normalized_z = world_z / NOISE_PERIOD + 0.5;
    let phase_x = normalized_x - normalized_x.floor();
    let phase_z = normalized_z - normalized_z.floor();
    let wrapped_x = 0.5 - (phase_x - 0.5).abs();
    let wrapped_z = 0.5 - (phase_z - 0.5).abs();
    // Keep the fold valley off the field's wrap edge, matching the halfTexel
    // clamp in signedNoise: texel 0 and texel 1023 are unrelated samples of
    // a non-periodic patch, so their bilinear blend at uv = 0 steps against
    // the local trend along every fold line.
    let half_texel = 0.5 / NOISE_RESOLUTION_F;
    let pixel_x = wrapped_x.max(half_texel) * NOISE_RESOLUTION_F - 0.5;
    let pixel_z = wrapped_z.max(half_texel) * NOISE_RESOLUTION_F - 0.5;
    let x0_raw = pixel_x.floor() as i32;
    let z0_raw = pixel_z.floor() as i32;
    let tx = pixel_x - pixel_x.floor();
    let tz = pixel_z - pixel_z.floor();
    let wrap = |v: i32| v.rem_euclid(NOISE_RESOLUTION as i32) as usize;
    let x0 = wrap(x0_raw);
    let z0 = wrap(z0_raw);
    let x1 = wrap(x0_raw + 1);
    let z1 = wrap(z0_raw + 1);
    let at = |x: usize, z: usize| data[z * NOISE_RESOLUTION + x];
    let a = lerp(at(x0, z0), at(x1, z0), tx);
    let b = lerp(at(x0, z1), at(x1, z1), tx);
    lerp(a, b, tz) * 2.0 - 1.0
}

/// Bounded-exponential altitude profile around sea level, applied to the
/// scaled macro landform. Convex on both sides of the waterline: near sea
/// level the gradient is gentler than the raw landform, and it accelerates
/// with elevation. Mirrored exactly by shapeElevation in terrain.wgsl.
pub fn shape_elevation(macro_height: f32) -> f32 {
    if macro_height >= 0.0 {
        // Clearance stretch near the waterline: sign-preserving, so the
        // coastline itself does not move, and smoothed to zero before the
        // foothills.
        let cleared = macro_height
            + WATERLINE_CLEARANCE
                * (macro_height / WATERLINE_CLEARANCE_SCALE).tanh()
                * (-(macro_height * macro_height)
                    / (WATERLINE_CLEARANCE_DECAY * WATERLINE_CLEARANCE_DECAY))
                    .exp();
        let t = cleared / LAND_PROFILE_REFERENCE;
        return LAND_PROFILE_PEAK
            * ((LAND_PROFILE_CURVE * t).exp() - 1.0)
            / ((LAND_PROFILE_CURVE).exp() - 1.0);
    }
    // Clamped at the reference: the deepest basins flatten into an abyssal
    // plain at the depth constant instead of diving without bound.
    let t = (macro_height / OCEAN_PROFILE_REFERENCE).min(1.0);
    OCEAN_PROFILE_DEPTH * ((OCEAN_PROFILE_CURVE * t).exp() - 1.0) / (OCEAN_PROFILE_CURVE.exp() - 1.0)
}

/// Push a finished height away from the waterline, preserving its sign.
///
/// `shape_elevation` shapes only the macro landform, and `base_height` adds
/// the fine detail octaves afterwards, so the terrain can land anywhere within
/// about +-2 m of sea level across the whole coastal plain. Terrain in that
/// band is indistinguishable from the water surface as far as the depth buffer
/// is concerned — and, once the sea has waves, as far as the eye is concerned
/// too, because every wave trough exposes it again.
///
/// This is the dead-zone removal that makes the coastline decisive. It is a
/// monotone, sign-preserving function that is exactly zero at sea level, so
/// the shoreline stays exactly where the landform put it; it only steepens the
/// gradient approaching it. `WATERLINE_PUSH_LAND` above and
/// `WATERLINE_PUSH_SEA` below are separate so the bed keeps its shelf.
///
/// Mirrored exactly by `pushFromWaterline` in terrain-vs.wgsl.
pub fn push_from_waterline(height: f32) -> f32 {
    let scale = WATERLINE_PUSH_SCALE.max(1e-4);
    let push = if height >= 0.0 {
        WATERLINE_PUSH_LAND
    } else {
        WATERLINE_PUSH_SEA
    };
    height + push * (height / scale).tanh()
}

/// The portable CPU base height, matching `baseHeight` in terrain.wgsl so
/// walking collision and spawn placement agree with the rendered surface.
pub fn base_height(noise: &NoiseField, x: f32, z: f32) -> f32 {
    let samples = &noise.samples;
    let noise = |sx: f32, sz: f32| sample_periodic_noise(samples, sx, sz);
    let rotate_a = |px: f32, pz: f32| [0.8 * px - 0.6 * pz, 0.6 * px + 0.8 * pz];
    let rotate_b = |px: f32, pz: f32| [0.6 * px + 0.8 * pz, -0.8 * px + 0.6 * pz];
    let detail_a = rotate_a(x, z);
    let detail_b = rotate_b(x, z);
    let macro_x = x / LANDFORM_HORIZONTAL_SCALE;
    let macro_z = z / LANDFORM_HORIZONTAL_SCALE;
    let source_a = rotate_a(macro_x, macro_z);
    let source_b = rotate_b(macro_x, macro_z);
    let warp_x = noise(source_a[0] * 0.74 + 1129.0, source_a[1] * 0.74 - 587.0);
    let warp_z = noise(source_b[0] * 0.69 - 977.0, source_b[1] * 0.69 + 1231.0);
    let warped_x = macro_x + warp_x * 210.0;
    let warped_z = macro_z + warp_z * 210.0;
    let a = rotate_a(warped_x, warped_z);
    let b = rotate_b(warped_x, warped_z);

    let continent = noise(warped_x * 0.38 + 137.0, warped_z * 0.38 - 311.0);
    let coast_detail = noise(a[0] * 0.83 - 743.0, a[1] * 0.83 + 521.0);
    let plains = noise(b[0] * 2.20 + 193.0, b[1] * 2.20 - 97.0);
    let mountain_field = noise(a[0] * 0.63 + 887.0, a[1] * 0.63 + 653.0);
    let ridge_noise_a = noise(b[0] * 2.15 - 419.0, b[1] * 2.15 - 811.0);
    let ridge_noise_b = noise(a[0] * 3.35 + 1601.0, a[1] * 3.35 - 1297.0);
    let ridge_a = smoothstep(0.0, 1.0, 1.0 - ridge_noise_a.abs());
    let ridge_b = smoothstep(0.0, 1.0, 1.0 - ridge_noise_b.abs());
    let ridge = ridge_a * 0.72 + ridge_b * 0.28;
    let detail = noise(detail_a[0] * 8.60 + 47.0, detail_a[1] * 8.60 + 101.0);
    let micro_detail = noise(detail_b[0] * 31.0 - 211.0, detail_b[1] * 31.0 + 379.0);
    let continental = continent * 0.70 + coast_detail * 0.30;
    let mountain_mask = smoothstep(-0.02, 0.62, mountain_field + continent * 0.38);
    let distance_from_spawn = (x * x + z * z).sqrt();
    let macro_height = SEA_LEVEL - 5.0 + continental * 32.0 + plains * 6.0
        + mountain_mask * (20.0 + 68.0 * ridge);
    let fine_height = detail * (1.4 + 2.8 * mountain_mask)
        + micro_detail * (0.55 + 0.55 * mountain_mask);
    let height = SEA_LEVEL
        + shape_elevation((macro_height - SEA_LEVEL) * LANDFORM_VERTICAL_SCALE)
        + fine_height * LANDFORM_VERTICAL_SCALE;
    let safe_height = SEA_LEVEL + 14.0 + plains * 1.8 + detail * 0.35;
    // The push goes on last, after the spawn blend, so the stabilised patch and
    // the open landscape are shaped by the same function and no artificial
    // ring appears at the blend's edge.
    push_from_waterline(lerp(safe_height, height, smoothstep(22.0, 90.0, distance_from_spawn)))
}

/// The terrain shader's grass line: `grassHeight` in terrain-fs.wgsl, the
/// point's height drifted up to a few metres by the noise field, so the
/// beach's sand gives way to turf along a wandering line instead of one
/// contour. Turf dominates the ground where it exceeds about 3.4 m above the
/// sea. Mirrors the shader's `mirrorTile(rotateUV(worldXZ * 0.0045, 0.85) +
/// (0.43, 0.67))` lookup, bilinear and repeating like its sampler.
pub fn grass_line_height(noise: &NoiseField, x: f32, z: f32, height: f32) -> f32 {
    let (sin, cos) = 0.85f32.sin_cos();
    let (u, v) = (x * 0.0045, z * 0.0045);
    let mirror = |p: f32| ((p - 2.0 * (p / 2.0).floor()) - 1.0).abs();
    let u_rotated = mirror(cos * u - sin * v + 0.43);
    let v_rotated = mirror(sin * u + cos * v + 0.67);
    let pixel_x = u_rotated * NOISE_RESOLUTION_F - 0.5;
    let pixel_z = v_rotated * NOISE_RESOLUTION_F - 0.5;
    let (x0, z0) = (pixel_x.floor(), pixel_z.floor());
    let (tx, tz) = (pixel_x - x0, pixel_z - z0);
    let wrap = |v: f32| (v as i32).rem_euclid(NOISE_RESOLUTION as i32) as usize;
    let at = |x: f32, z: f32| noise.samples[wrap(z) * NOISE_RESOLUTION + wrap(x)];
    let south = lerp(at(x0, z0), at(x0 + 1.0, z0), tx);
    let north = lerp(at(x0, z0 + 1.0), at(x0 + 1.0, z0 + 1.0), tx);
    height + (lerp(south, north, tz) - 0.5) * 7.0
}

/// The 480x480 base height map one erosion tile simulates, sampled at cell
/// centres (mirrors `CreateBaseHeightMap`), with the rivers carved into it.
/// The second map flags the cells a river's water covers: the simulation
/// holds them fixed and lets them drain what reaches them, since the river
/// carries it away.
pub fn create_base_height_map(
    noise: &NoiseField,
    tile: crate::erosion::TileKey,
    rivers: Option<&crate::rivers::network::RiverNetwork>,
) -> (Vec<f32>, Vec<f32>) {
    let world_minimum = crate::erosion::tile_simulation_minimum(tile);
    let mut height = vec![0.0f32; EROSION_RESOLUTION * EROSION_RESOLUTION];
    let mut river = vec![0.0f32; EROSION_RESOLUTION * EROSION_RESOLUTION];
    for z in 0..EROSION_RESOLUTION {
        let wz = world_minimum[1] + (z as f32 + 0.5) * EROSION_CELL_SIZE;
        for x in 0..EROSION_RESOLUTION {
            let wx = world_minimum[0] + (x as f32 + 0.5) * EROSION_CELL_SIZE;
            let index = z * EROSION_RESOLUTION + x;
            height[index] = base_height(noise, wx, wz);
            if let Some(network) = rivers {
                let envelope = network.envelope(wx, wz);
                height[index] = envelope.clamp(height[index]);
                // A cell whose middle the water covers belongs to the river.
                if envelope.bank_distance < 0.0 && height[index] < envelope.water {
                    river[index] = 1.0;
                }
            }
        }
    }
    (height, river)
}

/// The separable Hermite blend mask applied to each tile's retained
/// footprint (mirrors `CreateErosionBlendMask`'s CPU half; the GPU samples
/// the uploaded texture version for bilinear consistency).
pub fn erosion_blend_mask_samples() -> Vec<f32> {
    let half_resolution = EROSION_OUTPUT_RESOLUTION / 2;
    let mut axis_weights = vec![0.0f32; EROSION_OUTPUT_RESOLUTION];
    for index in 0..half_resolution {
        let phase = index as f32 / (half_resolution - 1) as f32;
        let rising = smoothstep(0.0, 1.0, phase);
        axis_weights[index] = rising;
        axis_weights[index + half_resolution] = 1.0 - rising;
    }
    let mut cpu_samples = vec![0.0f32; EROSION_OUTPUT_RESOLUTION * EROSION_OUTPUT_RESOLUTION];
    for (z, weight_z) in axis_weights.iter().enumerate() {
        for (x, weight_x) in axis_weights.iter().enumerate() {
            let mask = (weight_x * weight_z).clamp(0.0, 1.0);
            cpu_samples[z * EROSION_OUTPUT_RESOLUTION + x] = mask;
        }
    }
    cpu_samples
}

/// Print the un-eroded landform over a world grid as CSV so camera locations
/// (snowy peaks, grass plains) can be found from the shell. Mirrors `RunProbe`.
pub fn run_probe(probe_extent: f32, probe_step: f32) {
    let noise = NoiseField { samples: generate_noise_samples() };
    let mut z = -probe_extent;
    while z <= probe_extent {
        let mut x = -probe_extent;
        while x <= probe_extent {
            println!("{:.1},{:.1},{:.2}", x, z, base_height(&noise, x, z));
            x += probe_step;
        }
        z += probe_step;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::tasks::block_on;
    use wgpu::util::DeviceExt;

    /// Run explicitly on a GPU: cargo test terrain_profiles_match_gpu -- --ignored
    /// Shader validation alone cannot catch backend tanh overflow, which made
    /// elevated vertices and deep seabeds non-finite on Metal.
    #[test]
    #[ignore = "requires a GPU adapter"]
    fn terrain_profiles_match_gpu() {
        let instance = wgpu::Instance::default();
        let adapter = block_on(instance.request_adapter(&Default::default()))
            .expect("GPU adapter required for terrain regression test");
        let (device, queue) = block_on(adapter.request_device(&Default::default()))
            .expect("create terrain regression device");

        // Compile the production functions, with only the uniforms they read.
        // Keep inputs in a buffer so the backend evaluates them at runtime.
        let terrain = include_str!("../assets/shaders/terrain-vs.wgsl");
        let start = terrain.find("fn shapeElevation(").unwrap();
        let end = terrain.find("fn baseHeight(").unwrap();
        let parameters = [
            ("waterline_clearance", WATERLINE_CLEARANCE),
            ("waterline_clearance_scale", WATERLINE_CLEARANCE_SCALE),
            ("waterline_clearance_decay", WATERLINE_CLEARANCE_DECAY),
            ("land_profile_curve", LAND_PROFILE_CURVE),
            ("land_profile_reference", LAND_PROFILE_REFERENCE),
            ("land_profile_peak", LAND_PROFILE_PEAK),
            ("ocean_profile_curve", OCEAN_PROFILE_CURVE),
            ("ocean_profile_reference", OCEAN_PROFILE_REFERENCE),
            ("ocean_profile_depth", OCEAN_PROFILE_DEPTH),
            ("waterline_push_land", WATERLINE_PUSH_LAND),
            ("waterline_push_sea", WATERLINE_PUSH_SEA),
            ("waterline_push_scale", WATERLINE_PUSH_SCALE),
        ];
        let fields = parameters.iter()
            .map(|(name, _)| format!("{name}: f32,"))
            .collect::<String>();
        let source = format!(
            "struct StageUniforms {{ {fields} }};
             @group(0) @binding(0) var<uniform> stage: StageUniforms;
             @group(0) @binding(1) var<storage, read_write> samples: array<vec4<f32>>;
             {}
             @compute @workgroup_size(1)
             fn evaluate(@builtin(global_invocation_id) id: vec3<u32>) {{
                 let height = samples[id.x].x;
                 samples[id.x] = vec4<f32>(height, pushFromWaterline(height),
                                          shapeElevation(height), 0.0);
             }}",
            &terrain[start..end],
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("terrain altitude regression"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("terrain altitude regression"),
            layout: None,
            module: &shader,
            entry_point: Some("evaluate"),
            compilation_options: Default::default(),
            cache: None,
        });
        let parameters: Vec<f32> = parameters.iter().map(|(_, value)| *value).collect();
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("terrain profile parameters"),
            contents: bytemuck::cast_slice(&parameters),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let heights = [
            -1000.0, -500.0, -300.0, -180.0, -100.0, -90.0, -89.0, -88.0,
            -50.0, -20.0, -2.0, -0.1, -0.001, 0.0, 0.001, 0.1, 2.0, 20.0,
            50.0, 88.0, 89.0, 90.0, 100.0, 180.0, 300.0, 500.0, 1000.0,
        ];
        let samples: Vec<[f32; 4]> = heights.iter().map(|&h| [h, 0.0, 0.0, 0.0]).collect();
        let output = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("terrain profile samples"),
            contents: bytemuck::cast_slice(&samples),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("terrain profile readback"),
            size: output.size(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain altitude regression"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: output.as_entire_binding() },
            ],
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bindings, &[]);
            pass.dispatch_workgroups(heights.len() as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, output.size());
        queue.submit([encoder.finish()]);
        let (sender, receiver) = std::sync::mpsc::channel();
        readback.map_async(wgpu::MapMode::Read, .., move |result| {
            sender.send(result).unwrap();
        });
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        receiver.recv().unwrap().unwrap();
        let mapped = readback.get_mapped_range(..);
        let samples: &[[f32; 4]] = bytemuck::cast_slice(&mapped);
        for &[height, pushed, shaped, _] in samples {
            for (profile, actual, expected) in [
                ("pushFromWaterline", pushed, push_from_waterline(height)),
                ("shapeElevation", shaped, shape_elevation(height)),
            ] {
                let tolerance = 2e-5 * expected.abs().max(1.0);
                assert!(
                    actual.is_finite() && (actual - expected).abs() <= tolerance,
                    "{profile}({height}) on {:?}: GPU={actual}, CPU={expected}",
                    adapter.get_info(),
                );
            }
        }
    }
}
