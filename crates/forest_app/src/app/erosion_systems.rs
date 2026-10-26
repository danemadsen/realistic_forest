use super::*;

/// One frame of erosion streaming around the player, mirroring the C++
/// ordering: apply finished readbacks, honour a UI rerun request, then feed
/// the cache with the PRE-update player position.
#[expect(
    clippy::too_many_arguments,
    reason = "keeps the domain inputs explicit at this call boundary"
)]
pub(super) fn erosion_stream_system(
    mut cache: ResMut<ErosionCache>,
    bridge: Res<ErosionBridge>,
    noise: Res<NoiseField>,
    erosion_settings: Res<ErosionSettings>,
    mut applied: ResMut<AppliedErosionSettings>,
    mut rerun: ResMut<RerunErosion>,
    mut prewarm: ResMut<PrewarmRemaining>,
    mut preparation: ResMut<erosion::TilePreparation>,
    automation: Res<AutomationSettings>,
    rivers: Res<rivers::RiverField>,
    player: Query<&Player>,
    time: Res<Time>,
) {
    let Ok(player) = player.single() else {
        return;
    };
    // Tiles begun from now on carve the network in force.
    if cache.rivers.as_ref().map(std::sync::Arc::as_ptr)
        != rivers.network().map(std::sync::Arc::as_ptr)
    {
        cache.rivers = rivers.network().cloned();
    }
    erosion::apply_erosion_events(&mut cache, &bridge);
    // The C++ `rerunErosion` is a per-frame local; the resource persists, so
    // read it and clear it rather than moving it out.
    let rerun_requested = rerun.0;
    rerun.0 = false;
    if rerun_requested {
        applied.0 = *erosion_settings;
        erosion::reset_erosion_cache(&mut cache);
    }
    let position = [player.position.x, player.position.y, player.position.z];
    // The prewarm passes run at the full iteration budget with immediate
    // reveal; the steady state streams at the per-frame budget. The C++ rerun
    // path uses the defaults (per-frame budget, animated reveal).
    let prewarming = prewarm.0 > 0;
    let budget = if prewarming {
        applied.0.iterations
    } else {
        EROSION_ITERATIONS_PER_FRAME
    };
    // Interactive streaming builds each tile's inputs off the main thread.
    // The prewarm and screenshot runs keep them inline, so a capture begins
    // the same tiles on the same frames however busy the task pool is.
    let background = !prewarming && automation.shot_path.is_none();
    let began = erosion::update_erosion_cache(
        &mut cache,
        &bridge,
        &noise,
        background.then_some(&mut *preparation),
        &applied.0,
        position,
        budget,
        prewarming,
        time.delta_secs(),
    );
    // Each pass simulates exactly one tile, mirroring the C++ loop.
    if prewarming && began {
        prewarm.0 -= 1;
    }
}

/// Publish the embedded sim shaders once. They are self-contained (no
/// `#import`s, asserted by `erosion_worker`'s tests), so the worker compiles
/// their original bytes directly on its own device.
pub(super) fn publish_erosion_shader_sources(
    mut published: Local<bool>,
    bridge: Res<ErosionBridge>,
) {
    if *published {
        return;
    }
    let sources = ErosionShaderSources {
        init: include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../assets/shaders/erosion_init.wgsl"
        ))
        .to_vec(),
        flux: include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../assets/shaders/erosion_flux.wgsl"
        ))
        .to_vec(),
        water: include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../assets/shaders/erosion_water.wgsl"
        ))
        .to_vec(),
        terrain: include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../assets/shaders/erosion_terrain.wgsl"
        ))
        .to_vec(),
        thermal: include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../assets/shaders/erosion_thermal.wgsl"
        ))
        .to_vec(),
    };
    // SAFETY: these embedded erosion shaders clamp texture coordinates and
    // bound every array access for the worker's full-screen triangle draws.
    unsafe { bridge.publish_shader_sources(sources) };
    *published = true;
}

/// `--measure-overlap` replaces the normal stream/measure pair, exactly as the
/// C++ `main` returns into `RunOverlapMeasurement` before its render loop.
pub(super) fn in_measure_mode(automation: Res<AutomationSettings>) -> bool {
    automation.measure_overlap
}

pub(super) fn not_in_measure_mode(automation: Res<AutomationSettings>) -> bool {
    !automation.measure_overlap
}

/// Unattended `--measure-overlap` driver: simulates the requested tile and
/// its +x neighbour, prints the seam CSV, and exits. Replaces normal
/// streaming, mirroring the C++ short-circuit in main().
pub(super) fn measure_overlap_system(
    mut cache: ResMut<ErosionCache>,
    bridge: Res<ErosionBridge>,
    noise: Res<NoiseField>,
    applied: Res<AppliedErosionSettings>,
    automation: Res<AutomationSettings>,
) {
    erosion::apply_erosion_events(&mut cache, &bridge);

    let first = automation.overlap_tile;
    let second = TileKey {
        x: first.x + 1,
        z: first.z,
    };
    let state_of = |cache: &ErosionCache, key: TileKey| {
        cache
            .tiles
            .get(&key)
            .map(|tile| tile.state)
            .unwrap_or(crate::erosion::ErosionTileState::Queued)
    };
    if state_of(&cache, first) == crate::erosion::ErosionTileState::Ready
        && state_of(&cache, second) == crate::erosion::ErosionTileState::Ready
    {
        let first_tile = cache.tiles.get(&first).unwrap();
        let second_tile = cache.tiles.get(&second).unwrap();
        erosion::measure_overlap(first_tile, second_tile);
        // Print once, then end the process, the way the C++ returns straight
        // out of `main`. A deferred `AppExit` would let this system run again
        // and print the whole CSV a second time, and would enter the same
        // shutdown path that can park forever after `--shot` (see
        // `shot_scheduling_system`). `measure_overlap` prints with `println!`,
        // whose line buffer is flushed on the trailing newline.
        std::process::exit(0);
    }

    let target = if state_of(&cache, first) != crate::erosion::ErosionTileState::Ready {
        first
    } else {
        second
    };
    // Seed the 9x9 cache entries (atlas slots) the begin requires, exactly as
    // `RunOverlapMeasurement`'s single `EnsureErosionCache(erosion, firstTile)`
    // does; it re-centres on the first tile, not the current target, so the
    // second tile is never evicted mid-measurement.
    erosion::ensure_erosion_cache(&mut cache, first);
    let mut commands = crate::erosion::ErosionFrameCommands {
        simulation_id: Default::default(),
        sim_min: crate::erosion::tile_simulation_minimum(target),
        init: None,
        iterate: None,
        finalize: false,
    };
    // The C++ measurement loop begins each tile once and then iterates it to
    // completion, so a tile already under simulation is never restarted
    // (restarting would reset `completedIterations` every frame).
    if !cache.has_active_tile || cache.active_tile != target {
        erosion::begin_tile(&mut cache, &mut commands, &noise, target);
    }
    // `RunOverlapMeasurement`'s inner loop feeds the full per-frame budget
    // without clamping to the remaining count, unlike `UpdateErosionCache`, so
    // the last batch can overshoot `settings.iterations`; `140` iterations at
    // six per frame ends on `144`. Reproduce that exactly — but only while the
    // budget is still outstanding: `FinalizeErosionTile` runs inside that loop
    // and clears `hasActiveTile`, which ends it, so past that point the only
    // work left is the readback. That repeats here until the async map lands
    // and `apply_erosion_events` clears `has_active_tile`; without the guard
    // the driver would keep feeding six more iterations per frame while the
    // readback was in flight and the tile would run well past the C++'s 144.
    if let Some(active) = cache.tiles.get_mut(&target) {
        commands.simulation_id = active.simulation_id;
        if active.completed_iterations < applied.0.iterations {
            active.completed_iterations += EROSION_ITERATIONS_PER_FRAME;
            commands.iterate = Some(IterateCommand {
                count: EROSION_ITERATIONS_PER_FRAME,
                settings: applied.0,
            });
        }
        if active.completed_iterations >= applied.0.iterations {
            commands.finalize = true;
        }
    }
    bridge.set_frame(commands, Some(erosion::erosion_lookup_records(&cache)));
}

#[cfg(test)]
mod embedded_shader_tests {
    use super::*;

    #[test]
    fn erosion_shaders_publish_once_without_asset_plugins() {
        let mut app = App::new();
        app.init_resource::<ErosionBridge>();
        app.add_systems(Update, publish_erosion_shader_sources);

        app.update();
        let bridge = app.world().resource::<ErosionBridge>();
        let sources = bridge
            .take_shader_sources()
            .expect("embedded shaders are ready");
        let shader_sources = [
            sources.init,
            sources.flux,
            sources.water,
            sources.terrain,
            sources.thermal,
        ];
        assert!(shader_sources.iter().all(|source| !source.is_empty()));

        app.update();
        let bridge = app.world().resource::<ErosionBridge>();
        assert!(!bridge.shader_sources_ready());
    }
}
