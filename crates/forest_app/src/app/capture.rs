use super::*;

/// F12 capture: save a timestamped screenshot next to the working directory.
pub(super) fn on_manual_screenshot(trigger: On<ScreenshotCaptured>) {
    let name = chrono::Local::now()
        .format("forest_%Y%m%d_%H%M%S.png")
        .to_string();
    write_png(&trigger.image, Path::new(&name));
    log::info!("SCREENSHOT: saved {name}");
}

/// Serializes a captured bevy Image as a PNG byte file. Returns whether the
/// file was written, so the `--shot` path can report failure in its exit code.
pub(super) fn write_png(image: &Image, path: &Path) -> bool {
    let bytes = match image_converter_png(image) {
        Ok(bytes) => bytes,
        Err(error) => {
            log::error!("SCREENSHOT: encode failed: {error}");
            return false;
        }
    };
    if let Err(error) = std::fs::write(path, bytes) {
        log::error!("SCREENSHOT: write failed: {error}");
        return false;
    }
    true
}

pub(super) fn image_converter_png(image: &Image) -> Result<Vec<u8>, String> {
    use bevy::render::render_resource::TextureFormat;
    match image.texture_descriptor.format {
        TextureFormat::Bgra8Unorm | TextureFormat::Bgra8UnormSrgb => {
            // The wgpu swapchain is BGRA; PNG wants RGBA.
            let mut rgba = image.data.clone().ok_or("capture had no data")?;
            for pixel in rgba.chunks_exact_mut(4) {
                pixel.swap(0, 2);
            }
            let decoded = image::RgbaImage::from_raw(image.width(), image.height(), rgba)
                .ok_or("capture buffer size mismatch")?;
            let mut encoded: Vec<u8> = Vec::new();
            decoded
                .write_to(
                    &mut std::io::Cursor::new(&mut encoded),
                    image::ImageFormat::Png,
                )
                .map_err(|error| error.to_string())?;
            Ok(encoded)
        }
        _ => Err("unsupported capture format".to_string()),
    }
}

/// Frame times a `--shot` run collects, split by whether erosion tiles were
/// still streaming in that frame.
#[derive(Default)]
pub(super) struct FrameTimes {
    streaming: Vec<f32>,
    settled: Vec<f32>,
    erosion_streaming: erosion::ErosionPhaseCosts,
    erosion_settled: erosion::ErosionPhaseCosts,
}

impl FrameTimes {
    fn record(&mut self, seconds: f32, streaming: bool, erosion: erosion::ErosionPhaseCosts) {
        if streaming {
            self.streaming.push(seconds);
            self.erosion_streaming += &erosion;
        } else {
            self.settled.push(seconds);
            self.erosion_settled += &erosion;
        }
    }

    fn log(&self) {
        for (phase, samples, erosion) in [
            ("streaming", &self.streaming, &self.erosion_streaming),
            ("settled", &self.settled, &self.erosion_settled),
        ] {
            if samples.is_empty() {
                continue;
            }
            let mut sorted = samples.clone();
            sorted.sort_by(f32::total_cmp);
            let average = sorted.iter().sum::<f32>() / sorted.len() as f32;
            let p95 = sorted[(sorted.len() * 95 / 100).min(sorted.len() - 1)];
            log::info!(
                "FRAME: {phase} {:.1} ms average, {:.1} ms p95, {:.1} ms max over {} frames",
                average * 1000.0,
                p95 * 1000.0,
                sorted[sorted.len() - 1] * 1000.0,
                sorted.len()
            );
            let frames = samples.len() as f32;
            log::info!(
                "EROSION-PHASE: {phase} lookup+atlas {:.3} ms, poll {:.3} ms, consume {:.3} ms, finalize {:.3} ms, sim-record {:.3} ms per frame over {} frames",
                erosion.lookup_atlas_us as f32 / frames / 1_000.0,
                erosion.poll_us as f32 / frames / 1_000.0,
                erosion.consume_us as f32 / frames / 1_000.0,
                erosion.finalize_us as f32 / frames / 1_000.0,
                erosion.sim_record_us as f32 / frames / 1_000.0,
                samples.len()
            );
        }
    }
}

/// Unattended capture: let the erosion cache stream in around the pinned
/// camera for `--wait` frames, then save one screenshot and exit.
pub(super) fn shot_scheduling_system(
    automation: Res<AutomationSettings>,
    cache: Res<ErosionCache>,
    vegetation: Res<vegetation::VegetationField>,
    buildings: Res<buildings::BuildingField>,
    grass: Res<grass::GrassReadiness>,
    bridge: Option<Res<erosion::ErosionBridge>>,
    players: Query<&Player>,
    mut counter: ResMut<FrameCounter>,
    mut requested: ResMut<ShotRequested>,
    mut commands: Commands,
    time: Res<Time<Real>>,
    mut frame_times: Local<FrameTimes>,
    offscreen: Option<Res<OffscreenCapture>>,
) {
    let Some(shot_path) = automation.shot_path.clone() else {
        return;
    };
    // `--wait` counts frames after the prewarmed tiles have settled, so
    // captures of the same pose compare the same erosion however slowly the
    // GPU (or a software rasteriser) works through the prewarm. A tile whose
    // readback failed for good counts as settled, or the capture would wait
    // on it forever.
    let settled_tiles = cache
        .tiles
        .values()
        .filter(|tile| {
            matches!(
                tile.state,
                erosion::ErosionTileState::Ready | erosion::ErosionTileState::Failed
            )
        })
        .count();
    if settled_tiles < automation.erosion_prewarm as usize {
        return;
    }
    // Likewise the plants: the library loaded, every chunk around the pinned
    // camera scattered, and the latest snapshot on the GPU.
    if !vegetation.ready() {
        return;
    }
    // And the buildings: placed around the pose, textured and uploaded.
    if !buildings.ready() {
        return;
    }
    // Grass arrives asynchronously too. Start counting only once every
    // wanted chunk around this pose has reached the renderer's GPU buffers.
    let Ok(player) = players.single() else {
        return;
    };
    if !grass.ready([player.position.x, player.position.z]) {
        return;
    }
    counter.0 += 1;
    // Real frame times across the `--wait` window, reported before the
    // capture so runs compare on cost as well as looks. Frames while erosion
    // tiles still stream are kept apart from settled ones; the first 30
    // counted frames (pipeline warm-up, the last prewarm uploads) are skipped.
    if !requested.0 && counter.0 > 30 {
        let streaming = cache.tiles.values().any(|tile| {
            !matches!(
                tile.state,
                erosion::ErosionTileState::Ready | erosion::ErosionTileState::Failed
            )
        });
        frame_times.record(
            time.delta_secs(),
            streaming,
            bridge.map_or_else(erosion::ErosionPhaseCosts::default, |b| {
                b.drain_phase_costs()
            }),
        );
    }
    if requested.0 || counter.0 < automation.wait_frames as u64 {
        // A request already made and not yet captured; AppExit is written by
        // the capture observer once the image lands.
        return;
    }
    frame_times.log();
    requested.0 = true;
    // The observer outlives this system, so it owns the path rather than
    // borrowing the resource.
    let save_and_exit = move |trigger: On<ScreenshotCaptured>| {
        let saved = write_png(&trigger.image, Path::new(&shot_path));
        if saved {
            log::info!("SCREENSHOT: saved {shot_path}");
        }
        // End the process here instead of letting an `AppExit` unwind the
        // app: Bevy 0.17.3's pipelined-rendering teardown can park forever,
        // and has been caught doing it. It is a race, not a plain deadlock
        // — a sampled hung run showed the main thread parked in
        // `World::clear_all` -> `RenderAppChannels::drop` ->
        // `render_to_app_receiver.recv_blocking()`, while the render thread
        // sat inside `RenderApp::update()` waiting on a
        // main-thread-executor task that only the main schedule runs, and
        // that schedule has already stopped. Neither a runner override
        // (WinitPlugin owns the runner and needs it for the event loop) nor
        // waiting longer helps, and nothing is left to tear down: the PNG
        // is closed by `std::fs::write`, and the log line above is already
        // on stderr. The C++'s automation path likewise just returns out of
        // `main`, and this way a failed capture reports through the exit
        // code rather than looking like a success.
        std::process::exit(if saved { 0 } else { 1 });
    };
    let screenshot = match offscreen
        .as_ref()
        .and_then(|offscreen| offscreen.image.clone())
    {
        Some(image) => Screenshot::image(image),
        None => Screenshot::primary_window(),
    };
    commands.spawn(screenshot).observe(save_and_exit);
}

#[cfg(test)]
mod shot_scheduling_tests {
    use super::*;
    use std::sync::Arc;

    fn capture_app(wait_frames: i32) -> (App, Entity) {
        let mut app = App::new();
        let automation = AutomationSettings {
            shot_path: Some("capture.png".into()),
            wait_frames,
            erosion_prewarm: 0,
            ..default()
        };
        app.insert_resource(automation);
        app.init_resource::<ErosionCache>();
        app.init_resource::<grass::GrassReadiness>();
        app.init_resource::<FrameCounter>();
        app.init_resource::<ShotRequested>();
        app.init_resource::<Time<Real>>();
        let empty_noise = NoiseField {
            samples: Vec::new(),
        };
        let empty_noise = Arc::new(empty_noise);
        app.insert_resource(vegetation::VegetationField::new(empty_noise.clone(), false));
        let no_buildings = buildings::BuildingField::new(
            empty_noise,
            None,
            std::path::Path::new("."),
            Vec2::ZERO,
            false,
        );
        app.insert_resource(no_buildings);
        app.add_systems(Update, shot_scheduling_system);
        let player = app.world_mut().spawn(Player::default()).id();
        (app, player)
    }

    #[test]
    fn screenshot_waits_for_uploaded_grass_at_the_current_pose() {
        let (mut app, player) = capture_app(2);
        app.update();
        app.update();
        assert_eq!(app.world().resource::<FrameCounter>().0, 0);
        assert!(!app.world().resource::<ShotRequested>().0);

        app.world()
            .resource::<grass::GrassReadiness>()
            .set_ready(Some([0, 0]));
        app.update();
        assert_eq!(app.world().resource::<FrameCounter>().0, 1);
        assert!(!app.world().resource::<ShotRequested>().0);

        // An upload for the old pose cannot release a capture after teleporting.
        app.world_mut()
            .get_mut::<Player>(player)
            .unwrap()
            .position
            .x = 16.0;
        app.update();
        assert_eq!(app.world().resource::<FrameCounter>().0, 1);
        assert!(!app.world().resource::<ShotRequested>().0);

        app.world()
            .resource::<grass::GrassReadiness>()
            .set_ready(Some([16, 0]));
        app.update();
        assert!(app.world().resource::<ShotRequested>().0);
        assert_eq!(
            app.world_mut()
                .query::<&Screenshot>()
                .iter(app.world())
                .count(),
            1
        );
    }

    #[test]
    fn unavailable_grass_assets_do_not_block_screenshots() {
        let (mut app, _) = capture_app(1);
        app.world().resource::<grass::GrassReadiness>().disable();
        app.update();
        assert!(app.world().resource::<ShotRequested>().0);
    }
}
