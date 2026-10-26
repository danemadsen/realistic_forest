use super::*;
use crate::{InitTileCommand, tile, tile_simulation_minimum};

/// The worker compiles the published WGSL verbatim; the sim shaders
/// declare the shared globals uniform but may never read it, because the
/// worker binds a zero-filled dummy instead of the render world's buffer.
#[test]
fn erosion_sim_shaders_never_read_the_globals_uniform() {
    let manifest = std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../.."));
    for name in ["init", "flux", "water", "terrain", "thermal"] {
        let path = manifest.join(format!("assets/shaders/erosion_{name}.wgsl"));
        let source = std::fs::read_to_string(&path)
            .unwrap_or_else(|err| panic!("cannot read {}: {err}", path.display()));
        assert!(
            !source.contains("globals."),
            "{} reads `globals.`; the worker's zeroed dummy uniform would be visible",
            path.display()
        );
    }
}

/// One sim shader, parsed as the worker compiles it.
fn sim_module(name: &str) -> naga::Module {
    let manifest = std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../.."));
    let path = manifest.join(format!("assets/shaders/erosion_{name}.wgsl"));
    let source = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("cannot read {}: {err}", path.display()));
    naga::front::wgsl::parse_str(&source)
        .unwrap_or_else(|error| panic!("{}: {}", path.display(), error.emit_to_string(&source)))
}

fn fragment_stage(module: &naga::Module) -> &naga::Function {
    let entry = module
        .entry_points
        .iter()
        .find(|entry| entry.name == "fs_main");
    &entry.expect("every sim shader has an fs_main").function
}

/// Every call a block makes, once per call site, however deeply nested.
fn call_sites(block: &naga::Block, found: &mut Vec<naga::Handle<naga::Function>>) {
    for statement in block.iter() {
        match statement {
            naga::Statement::Call { function, .. } => found.push(*function),
            naga::Statement::Block(inner) => call_sites(inner, found),
            naga::Statement::If { accept, reject, .. } => {
                call_sites(accept, found);
                call_sites(reject, found);
            }
            naga::Statement::Switch { cases, .. } => {
                for case in cases {
                    call_sites(&case.body, found);
                }
            }
            naga::Statement::Loop {
                body, continuing, ..
            } => {
                call_sites(body, found);
                call_sites(continuing, found);
            }
            _ => {}
        }
    }
}

/// The texel loads a function's text performs: its own, plus a callee's
/// for every site that calls it. A loop body counts once.
fn texel_loads(module: &naga::Module, function: &naga::Function) -> usize {
    let own = function
        .expressions
        .iter()
        .filter(|(_, expression)| matches!(expression, naga::Expression::ImageLoad { .. }))
        .count();
    let mut called = Vec::new();
    call_sites(&function.body, &mut called);
    own + called
        .into_iter()
        .map(|handle| texel_loads(module, &module.functions[handle]))
        .sum::<usize>()
}

/// The names of the functions `function` reaches (itself included) that
/// call `callee` directly.
fn callers_of(
    module: &naga::Module,
    function: &naga::Function,
    callee: &str,
) -> std::collections::BTreeSet<String> {
    let mut callers = std::collections::BTreeSet::new();
    let mut pending = vec![function];
    let mut seen = std::collections::BTreeSet::new();
    while let Some(function) = pending.pop() {
        let mut called = Vec::new();
        call_sites(&function.body, &mut called);
        for handle in called {
            let target = &module.functions[handle];
            let name = target.name.clone().unwrap_or_default();
            if name == callee {
                callers.insert(function.name.clone().unwrap_or_default());
            }
            if seen.insert(name) {
                pending.push(target);
            }
        }
    }
    callers
}

/// The terrain pass runs 140 times a tile, and the runoff's path noise
/// does not change between iterations: it reads it from the static field
/// the init pass stamps once, so the only world noise it still evaluates
/// is the geology's, which depends on the evolving bedrock. The stamp
/// builds the cell centre and cell coordinate exactly as the pass does.
#[test]
fn terrain_pass_reads_its_path_noise_from_the_static_field() {
    let module = sim_module("terrain");
    let noisy = callers_of(&module, fragment_stage(&module), "geologyNoise");
    assert_eq!(
        noisy.into_iter().collect::<Vec<_>>(),
        ["geologyResistance"],
        "only the geology evaluates noise in the terrain pass"
    );

    let manifest = std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../.."));
    let read = |name: &str| {
        let text =
            std::fs::read_to_string(manifest.join(format!("assets/shaders/erosion_{name}.wgsl")))
                .unwrap();
        text.split_whitespace().collect::<Vec<_>>().join(" ")
    };
    let (init, terrain) = (read("init"), read("terrain"));
    for shared in [
        "let cell_size = max(stage.cell_size, 0.0001);",
        "let world_position = stage.world_min + (vec2<f32>(coord) + vec2<f32>(0.5)) * cell_size;",
        "let world_cell = world_position / cell_size;",
    ] {
        assert!(
            init.contains(shared) && terrain.contains(shared),
            "`{shared}` differs"
        );
    }
    assert!(
        init.contains(
            "return vec4<f32>(geologyNoise(world_cell * 0.16 + vec2<f32>(17.13, -9.71)), \
         geologyNoise(world_cell * 0.43 + vec2<f32>(-31.7, 22.9)), 0.0, 0.0);"
        ),
        "the static field holds the broad and fine path noise"
    );
    assert!(
        terrain
            .contains("let broad_path_noise = path_noises.r; let fine_path_noise = path_noises.g;")
    );
}

/// Each pass loads every texel of its neighbourhood once. Counted from
/// the shader text, per call site: the terrain pass's eight donor calls
/// each hold the two loads a draining donor needs, and the thermal
/// pass's loop body counts once. The counts before every pass loaded
/// its neighbourhood once were 29, 35, 34 and 6.
#[test]
fn erosion_passes_respect_texel_load_budgets() {
    for (name, budget) in [("flux", 19), ("water", 23), ("terrain", 48), ("thermal", 4)] {
        let module = sim_module(name);
        let loads = texel_loads(&module, fragment_stage(&module));
        assert!(
            loads <= budget,
            "erosion-{name} loads {loads} texels, over its budget of {budget}"
        );
    }
}

#[test]
fn chunk_plan_preserves_the_iteration_total() {
    assert!(chunk_plan(0).is_empty());
    assert_eq!(chunk_plan(6), vec![6]);
    assert_eq!(chunk_plan(24), vec![24]);
    assert_eq!(chunk_plan(25), vec![24, 1]);
    assert_eq!(chunk_plan(140), vec![24, 24, 24, 24, 24, 20]);
    for count in [0usize, 1, 6, 23, 24, 25, 139, 140, 144, 1000] {
        let chunks = chunk_plan(count);
        assert_eq!(chunks.iter().sum::<usize>(), count);
        for chunk in chunks {
            assert!(chunk <= EROSION_WORKER_MAX_ITERATIONS_PER_SUBMIT);
        }
    }
}

/// One command set — the app's prewarm shape: init and finalize in the
/// same frame — driven through the real thread, its own device, the
/// full 140-iteration chunked submission and the readback maps. The tile,
/// a 20 % slope rippled by 3 m swells so that water runs and cuts, must
/// finalize with finite diagnostics, report the GPU time it took and land
/// one atlas patch. `--nocapture` prints the diagnostics.
#[ignore = "requires a GPU adapter"]
#[test]
fn worker_delivers_a_tile_end_to_end() {
    let manifest = std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../.."));
    let sources = ErosionShaderSources {
        init: std::fs::read(manifest.join("assets/shaders/erosion_init.wgsl")).unwrap(),
        flux: std::fs::read(manifest.join("assets/shaders/erosion_flux.wgsl")).unwrap(),
        water: std::fs::read(manifest.join("assets/shaders/erosion_water.wgsl")).unwrap(),
        terrain: std::fs::read(manifest.join("assets/shaders/erosion_terrain.wgsl")).unwrap(),
        thermal: std::fs::read(manifest.join("assets/shaders/erosion_thermal.wgsl")).unwrap(),
    };
    let instance = wgpu::Instance::default();
    let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .expect("no GPU adapter for the erosion worker test");
    let bridge = ErosionBridge::default();
    bridge.publish_shader_sources(sources);
    assert!(spawn_erosion_worker(
        &bridge,
        &adapter,
        adapter.features(),
        wgpu::Limits::default()
    ));

    let key = tile(1, 2);
    let base_height: Vec<f32> = (0..SIM_PIXELS)
        .map(|cell| {
            let x = (cell % EROSION_RESOLUTION) as f32;
            let z = (cell / EROSION_RESOLUTION) as f32;
            10.0 + 0.2 * EROSION_CELL_SIZE * z + 3.0 * (x * 0.21).sin() * (z * 0.17).cos()
        })
        .collect();
    let drainage_area =
        crate::route_base_drainage(&base_height, EROSION_RESOLUTION, EROSION_CELL_SIZE);
    let frame_commands = ErosionFrameCommands {
        sim_min: tile_simulation_minimum(key),
        init: Some(InitTileCommand {
            key,
            base_height,
            drainage_area,
            river: vec![0.0; SIM_PIXELS],
        }),
        iterate: None,
        finalize: true,
    };
    bridge.set_frame(frame_commands, None);

    for _ in 0..3000 {
        if let Some(why) = bridge.take_worker_dead() {
            panic!("worker died before finalizing: {why}");
        }
        let events = bridge.drain_events();
        let finalized = events.iter().find_map(|event| match event {
            ErosionEvent::TileFinalized(tile) => Some(tile),
            ErosionEvent::ReadbackFailed(key) => {
                panic!("the test tile's readback failed: ({}, {})", key.x, key.z)
            }
        });
        if let Some(tile) = finalized {
            assert_eq!(tile.key, key);
            assert_eq!(
                tile.cpu_delta.len(),
                EROSION_OUTPUT_RESOLUTION * EROSION_OUTPUT_RESOLUTION
            );
            for value in [
                tile.stats.minimum_height,
                tile.stats.maximum_height,
                tile.stats.land_coverage,
                tile.stats.maximum_incision,
                tile.stats.maximum_deposition,
                tile.stats.erosion_detail,
                tile.stats.flow_axis_bias,
                tile.stats.maximum_drainage,
                tile.stats.bedrock_exposure,
                tile.stats.mean_loose_cover,
                tile.stats.simulation_ms,
            ] {
                assert!(value.is_finite(), "{value} is not finite");
            }
            eprintln!("{:?}", tile.stats);
            assert!(
                tile.stats.simulation_ms > 0.0,
                "the prewarm-shaped tile is timed"
            );
            assert!(tile.stats.maximum_incision > 0.0, "the slope's runoff cuts");
            let patches = bridge.drain_patches();
            assert_eq!(patches.len(), 1);
            assert_eq!(patches[0].key, key);
            assert_eq!(
                patches[0].atlas_height.len(),
                EROSION_ATLAS_PITCH * EROSION_ATLAS_PITCH * 4
            );
            assert_eq!(
                patches[0].atlas_flow.len(),
                EROSION_ATLAS_PITCH * EROSION_ATLAS_PITCH * 4
            );
            return;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    panic!("the worker never finalized the test tile (30 s)");
}
