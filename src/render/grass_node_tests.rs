use super::*;
use bevy::tasks::block_on;
use std::mem::{offset_of, size_of};
use wgpu::util::DeviceExt;

const CULL_SHADER: &str = include_str!("../../assets/shaders/grass-cull.wgsl");

fn shader_struct_layout(source: &str, name: &str) -> (u32, Vec<(String, u32)>) {
    let module = naga::front::wgsl::parse_str(source).expect("grass shader parses");
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .expect("grass shader validates");
    let ty = module.types.iter().find_map(|(_, ty)| {
        (ty.name.as_deref() == Some(name)).then_some(ty)
    }).unwrap_or_else(|| panic!("missing shader struct {name}"));
    let naga::TypeInner::Struct { members, span } = &ty.inner else {
        panic!("{name} is not a struct");
    };
    (*span, members.iter().map(|member| {
        (member.name.clone().expect("named member"), member.offset)
    }).collect())
}

fn assert_layout(source: &str, name: &str, size: usize, members: &[(&str, usize)]) {
    let (span, shader_members) = shader_struct_layout(source, name);
    assert_eq!(span as usize, size, "{name} array stride");
    for &(member, offset) in members {
        let shader_offset = shader_members.iter().find_map(|(name, offset)| {
            (name == member).then_some(*offset)
        }).unwrap_or_else(|| panic!("missing {name}.{member}"));
        assert_eq!(shader_offset as usize, offset, "{name}.{member}");
    }
}

#[test]
fn grass_compute_and_vertex_records_share_the_host_abi() {
    assert_layout(CULL_SHADER, "GrassCullJob", size_of::<GrassCullJob>(), &[
        ("input_first", offset_of!(GrassCullJob, input_first)),
        ("count", offset_of!(GrassCullJob, count)),
        ("output_first", offset_of!(GrassCullJob, output_first)),
        ("draw_word", offset_of!(GrassCullJob, draw_word)),
        ("radius", offset_of!(GrassCullJob, radius)),
        ("_pad0", offset_of!(GrassCullJob, _pad)),
        ("_pad1", offset_of!(GrassCullJob, _pad) + 4),
        ("_pad2", offset_of!(GrassCullJob, _pad) + 8),
    ]);
    assert_layout(CULL_SHADER, "GrassInstance", size_of::<GrassInstance>(), &[
        ("xz", offset_of!(GrassInstance, xz)),
        ("rotation", offset_of!(GrassInstance, rotation)),
        ("scale", offset_of!(GrassInstance, scale)),
        ("tint", offset_of!(GrassInstance, tint)),
        ("seed", offset_of!(GrassInstance, seed)),
        ("scatter_data", offset_of!(GrassInstance, _pad)),
    ]);
    assert_layout(CULL_SHADER, "GrassDrawInstance", size_of::<GrassDrawInstance>(), &[
        ("xz", offset_of!(GrassDrawInstance, xz)),
        ("rotation", offset_of!(GrassDrawInstance, rotation)),
        ("scale", offset_of!(GrassDrawInstance, scale)),
        ("tint", offset_of!(GrassDrawInstance, tint)),
        ("seed", offset_of!(GrassDrawInstance, seed)),
        ("ground_height", offset_of!(GrassDrawInstance, ground)),
        ("slope_x", offset_of!(GrassDrawInstance, ground) + 4),
        ("ground_average", offset_of!(GrassDrawInstance, ground_colour)),
        ("slope_z", offset_of!(GrassDrawInstance, ground_colour) + 12),
    ]);
    for source in [CULL_SHADER, include_str!("../../assets/shaders/grass.wgsl")] {
        assert_layout(source, "GrassFrame", size_of::<GrassFrame>(), &[
            ("mapping", offset_of!(GrassFrame, mapping)),
            ("wind", offset_of!(GrassFrame, wind)),
            ("range", offset_of!(GrassFrame, range)),
            ("layer_end", offset_of!(GrassFrame, layer_end)),
        ]);
    }
    assert_eq!(size_of::<GrassDraw>(), 20);
    assert_eq!(offset_of!(GrassDraw, instance_count), 4);
    assert_eq!(offset_of!(GrassDraw, first_instance), 16);
}

#[test]
fn render_pipeline_vertex_attributes_match_both_host_structs() {
    let mesh = mesh_vertex_attributes();
    let mesh_expect: [(u32, u64, wgpu::VertexFormat); 4] = [
        (0, offset_of!(grass::GrassVertex, position) as u64, wgpu::VertexFormat::Float32x3),
        (1, offset_of!(grass::GrassVertex, normal) as u64, wgpu::VertexFormat::Float32x3),
        (2, offset_of!(grass::GrassVertex, uv) as u64, wgpu::VertexFormat::Float32x2),
        (3, offset_of!(grass::GrassVertex, tangent) as u64, wgpu::VertexFormat::Float32x4),
    ];
    for ((location, offset, format), attribute) in mesh_expect.into_iter().zip(mesh) {
        assert_eq!(attribute.shader_location, location);
        assert_eq!(attribute.offset, offset);
        assert_eq!(attribute.format, format);
    }
    let instance = draw_instance_attributes();
    let instance_expect: [(u32, u64, wgpu::VertexFormat); 7] = [
        (4, offset_of!(GrassDrawInstance, xz) as u64, wgpu::VertexFormat::Float32x2),
        (5, offset_of!(GrassDrawInstance, rotation) as u64, wgpu::VertexFormat::Float32),
        (6, offset_of!(GrassDrawInstance, scale) as u64, wgpu::VertexFormat::Float32),
        (7, offset_of!(GrassDrawInstance, tint) as u64, wgpu::VertexFormat::Float32),
        (8, offset_of!(GrassDrawInstance, seed) as u64, wgpu::VertexFormat::Float32),
        (9, offset_of!(GrassDrawInstance, ground) as u64, wgpu::VertexFormat::Float32x2),
        (10, offset_of!(GrassDrawInstance, ground_colour) as u64, wgpu::VertexFormat::Float32x4),
    ];
    for ((location, offset, format), attribute) in instance_expect.into_iter().zip(instance) {
        assert_eq!(attribute.shader_location, location);
        assert_eq!(attribute.offset, offset);
        assert_eq!(attribute.format, format);
    }
    assert_eq!(size_of::<grass::GrassVertex>(), 48);
    assert_eq!(size_of::<GrassDrawInstance>(), 48);
}

#[test]
fn grass_cull_jobs_cover_each_candidate_once_and_share_a_bounded_output_region() {
    for count in [0, 1, 63, 64, 65, 129, 14_336] {
        let mut jobs = Vec::new();
        push_cull_jobs(&mut jobs, 117, count, 921, 16, 0.75);
        assert_eq!(jobs.len(), count.div_ceil(64) as usize);
        let mut next = 117;
        for job in &jobs {
            assert_eq!(job.input_first, next);
            assert!(job.count > 0 && job.count <= 64);
            assert_eq!(job.output_first, 921);
            assert_eq!(job.draw_word, 16);
            assert_eq!(job.radius, 0.75);
            next += job.count;
        }
        assert_eq!(next, 117 + count);
    }
    // Adjacent model subranges have independent survivor counters, while all
    // workgroups for one model append into that model's candidate-sized region.
    let mut jobs = Vec::new();
    push_cull_jobs(&mut jobs, 256, 65, 512, 1, 0.2);
    push_cull_jobs(&mut jobs, 321, 128, 577, 6, 3.0);
    assert_eq!(jobs.len(), 4);
    assert!(jobs[..2].iter().all(|job| job.output_first == 512 && job.draw_word == 1));
    assert!(jobs[2..].iter().all(|job| job.output_first == 577 && job.draw_word == 6));
    for layer in 0..LAYER_COUNT {
        let capacity = instances_per_slot(layer);
        assert_eq!(capacity, grass::CHUNK_CELLS as usize * grass::CHUNK_CELLS as usize * grass::LAYER_CANDIDATES[layer]);
        assert_eq!(capacity % 64, 0, "whole workgroups fill a slot");
    }
}

fn test_texture(device: &wgpu::Device, queue: &wgpu::Queue, bytes: &[u8]) -> wgpu::Texture {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("grass culling fixture"),
        size: wgpu::Extent3d { width: 64, height: 64, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    write_test_texture(queue, &texture, bytes);
    texture
}

fn write_test_texture(queue: &wgpu::Queue, texture: &wgpu::Texture, bytes: &[u8]) {
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture, mip_level: 0, origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0, bytes_per_row: Some(64 * 4), rows_per_image: Some(64),
        },
        wgpu::Extent3d { width: 64, height: 64, depth_or_array_layers: 1 },
    );
}

fn read_buffer(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Vec<u8> {
    let (sender, receiver) = std::sync::mpsc::channel();
    buffer.map_async(wgpu::MapMode::Read, .., move |result| sender.send(result).unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).expect("poll culling fixture");
    receiver.recv().expect("mapping callback").expect("map culling fixture");
    let bytes = buffer.get_mapped_range(..).to_vec();
    buffer.unmap();
    bytes
}

/// Exercises the production compute shader on Metal, Vulkan or a software
/// adapter. The fixture covers multiple workgroups and models without relying
/// on the nondeterministic order in which workgroups append survivors.
#[test]
fn grass_gpu_compacts_habitat_survivors_and_resets_indirect_counts() {
    let instance = wgpu::Instance::default();
    let Ok(adapter) = block_on(instance.request_adapter(&Default::default())) else {
        eprintln!("skipping grass compute regression: no GPU adapter");
        return;
    };
    let (device, queue) = block_on(adapter.request_device(&Default::default()))
        .expect("create grass compute regression device");
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("grass culling regression"),
        source: wgpu::ShaderSource::Wgsl(CULL_SHADER.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("grass culling regression"), layout: None, module: &shader,
        entry_point: Some("cull_grass"), compilation_options: Default::default(), cache: None,
    });
    let mut habitat_bytes = Vec::with_capacity(64 * 64 * 4);
    let mut colour_bytes = Vec::with_capacity(64 * 64 * 4);
    for _z in 0..64 {
        for x in 0..64 {
            // Uniform ground height/normal; a forbidden strip lies far enough
            // from root x=5 to admit a small clump but reject a wide footprint.
            habitat_bytes.extend_from_slice(&[153, if (39..=45).contains(&x) { 0 } else { 255 }, 51, 102]);
            colour_bytes.extend_from_slice(&[51, 102, 153, if (20..=25).contains(&x) { 0 } else { 255 }]);
        }
    }
    let habitat = test_texture(&device, &queue, &habitat_bytes);
    let ground_colour = test_texture(&device, &queue, &colour_bytes);
    let habitat_view = habitat.create_view(&Default::default());
    let colour_view = ground_colour.create_view(&Default::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        mag_filter: wgpu::FilterMode::Linear, min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });
    let frame = GrassFrame {
        mapping: [0.0, 0.0, 64.0, 1.0], wind: [0.0; 4],
        range: [100.0, 256.0, 0.0, 0.0], layer_end: [160.0, 80.0, 16.0, 0.0],
    };
    let frame_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("grass regression frame"), contents: bytemuck::bytes_of(&frame),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let mut candidates = Vec::new();
    let counts = [130u32, 70];
    let output_first = [3u32, 139];
    let radii = [0.2f32, 3.0];
    let mut expected = [Vec::new(), Vec::new()];
    for model in 0..2 {
        for index in 0..counts[model] {
            let kind = index % 7;
            let candidate = GrassInstance {
                xz: match kind {
                    0 => [300.0, 0.0], // distance
                    1 => [-10.0, 0.0], // closed canopy
                    2 => [10.0, 0.0], // unsuitable root
                    3 => [5.0, 0.0], // model-dependent footprint
                    4 => [20.0, 0.0], // carpet beyond its layer end
                    _ => [-20.0, 0.0],
                },
                rotation: index as f32 * 0.01,
                scale: 1.2 + index as f32 * 0.001,
                tint: model as f32 + index as f32 * 0.001,
                seed: 0.5,
                _pad: [match kind { 5 => 0.0, 6 => 0.5, _ => 1.0 }, if kind == 4 { 3.0 } else { 0.0 }],
            };
            if kind == 6 || (kind == 3 && model == 0) {
                expected[model].push((candidate, if kind == 6 { 0.5 } else { 1.0 }));
            }
            candidates.push(candidate);
        }
    }
    let mut jobs = Vec::new();
    push_cull_jobs(&mut jobs, 0, counts[0], output_first[0], 1, radii[0]);
    push_cull_jobs(&mut jobs, counts[0], counts[1], output_first[1], 6, radii[1]);
    let candidates_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("grass regression candidates"), contents: bytemuck::cast_slice(&candidates),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let jobs_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("grass regression jobs"), contents: bytemuck::cast_slice(&jobs),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let sentinel = GrassDrawInstance {
        xz: [-999.0; 2], rotation: -999.0, scale: -999.0, tint: -999.0,
        seed: -999.0, ground: [-999.0; 2], ground_colour: [-999.0; 4],
    };
    let prepared = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("grass regression prepared"),
        contents: bytemuck::cast_slice(&vec![sentinel; 214]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let commands = [
        GrassDraw { index_count: 12, instance_count: 0, first_index: 7, base_vertex: -3, first_instance: output_first[0] },
        GrassDraw { index_count: 36, instance_count: 0, first_index: 9, base_vertex: -5, first_instance: output_first[1] },
    ];
    let arguments = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("grass regression indirect"), contents: bytemuck::cast_slice(&commands),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    });
    let create_readback = |source: &wgpu::Buffer| device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("grass regression readback"), size: source.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false,
    });
    let prepared_readback = create_readback(&prepared);
    let arguments_readback = create_readback(&arguments);
    let inputs = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("grass regression inputs"), layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: candidates_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: jobs_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: prepared.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: arguments.as_entire_binding() },
        ],
    });
    let environment = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("grass regression environment"), layout: &pipeline.get_bind_group_layout(1),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&habitat_view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&sampler) },
            wgpu::BindGroupEntry { binding: 2, resource: frame_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&colour_view) },
        ],
    });
    let dispatch_and_read = || {
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &inputs, &[]);
            pass.set_bind_group(1, &environment, &[]);
            pass.dispatch_workgroups(jobs.len() as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&prepared, 0, &prepared_readback, 0, prepared.size());
        encoder.copy_buffer_to_buffer(&arguments, 0, &arguments_readback, 0, arguments.size());
        queue.submit([encoder.finish()]);
        (read_buffer(&device, &prepared_readback), read_buffer(&device, &arguments_readback))
    };
    let (prepared_bytes, arguments_bytes) = dispatch_and_read();
    let records: &[GrassDrawInstance] = bytemuck::cast_slice(&prepared_bytes);
    let draws: &[GrassDraw] = bytemuck::cast_slice(&arguments_bytes);
    let close = |actual: f32, expected: f32| {
        assert!((actual - expected).abs() < 2e-5, "GPU {actual}, expected {expected}");
    };
    for model in 0..2 {
        assert_eq!(draws[model].instance_count as usize, expected[model].len(), "model {model}");
        assert_eq!(draws[model].index_count, commands[model].index_count);
        assert_eq!(draws[model].first_index, commands[model].first_index);
        assert_eq!(draws[model].base_vertex, commands[model].base_vertex);
        assert_eq!(draws[model].first_instance, commands[model].first_instance);
        let start = output_first[model] as usize;
        let count = expected[model].len();
        let mut survivors: Vec<_> = records[start..start + count].iter().collect();
        survivors.sort_by(|a, b| a.tint.total_cmp(&b.tint));
        expected[model].sort_by(|a, b| a.0.tint.total_cmp(&b.0.tint));
        for (actual, (candidate, growth)) in survivors.into_iter().zip(&expected[model]) {
            assert_eq!(actual.xz, candidate.xz);
            close(actual.rotation, candidate.rotation);
            close(actual.scale, candidate.scale * growth);
            close(actual.tint, candidate.tint);
            close(actual.seed, candidate.seed);
            close(actual.ground[0], 0.6 - 0.015); // ground.r less the seat offset
            close(actual.ground[1], 0.2 / 0.8f32.sqrt());
            for (actual, expected) in actual.ground_colour[..3].iter().zip([0.2, 0.4, 0.6]) {
                close(*actual, expected);
            }
            close(actual.ground_colour[3], 0.4 / 0.8f32.sqrt());
        }
        for unused in &records[start + count..start + counts[model] as usize] {
            assert_eq!(bytemuck::bytes_of(unused), bytemuck::bytes_of(&sentinel), "output overflow for model {model}");
        }
    }
    for index in (0..3).chain(133..139).chain(209..214) {
        assert_eq!(bytemuck::bytes_of(&records[index]), bytemuck::bytes_of(&sentinel), "overwritten output guard {index}");
    }
    // A fresh frame resets the shared atomic words before compute. Rejecting
    // every root must leave zero indirect instances, even after a busy frame.
    queue.write_buffer(&arguments, 0, bytemuck::cast_slice(&commands));
    for texel in habitat_bytes.as_chunks_mut::<4>().0 {
        texel[1] = 0;
    }
    write_test_texture(&queue, &habitat, &habitat_bytes);
    let (next_prepared, next_arguments) = dispatch_and_read();
    assert_eq!(next_arguments, bytemuck::cast_slice::<GrassDraw, u8>(&commands));
    assert_eq!(next_prepared, prepared_bytes, "rejected clumps must not write prepared records");
}
