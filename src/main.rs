//! Application assembly for the Bevy + wgpu port of ~/forest (raylib/GL).
//!
//! Frame order mirrors the C++ main loop: keyed settings changes, the egui
//! input check, erosion streaming (which reads the pre-update player
//! position), the player update, and then the render graph (erosion sim ->
//! terrain G-buffer -> SSAO -> blur -> composite -> FXAA -> egui -> upscale).

mod automation;
mod constants;
mod erosion;
mod matrices;
mod noise;
mod player;
mod render;
mod ui;

use crate::automation::AutomationSettings;
use crate::constants::*;
use crate::erosion::{ErosionBridge, ErosionCache, IterateCommand, TileKey};
use crate::noise::NoiseField;
use crate::player::Player;
use bevy::app::AppExit;
use bevy::camera::primitives::Frustum;
use bevy::camera::visibility::VisibleEntities;
use bevy::camera::{
    Camera, ClearColorConfig, PerspectiveProjection, Projection, RenderTarget,
};
use bevy::prelude::*;
use bevy::render::camera::CameraRenderGraph;
use bevy::render::settings::{RenderCreation, WgpuSettings};
use bevy::render::view::window::screenshot::{Screenshot, ScreenshotCaptured};
use bevy::render::view::Msaa;
use bevy::render::RenderPlugin;
use bevy::window::{CursorOptions, PrimaryWindow, WindowRef, WindowResolution};
use bevy_egui::{EguiPlugin, EguiPrimaryContextPass};
use std::path::Path;

/// Erosion parameters in force for the currently streaming cache; edited by
/// the diagnostics panel, copied in when "Regenerate erosion cache" reruns.
#[derive(Resource)]
struct AppliedErosionSettings(pub ErosionSettings);

/// One-shot automation flags distilled from the command line so systems can
/// branch without re-reading argv.
#[derive(Resource)]
struct WorldOptions {
    /// --no-water: skip the ocean plane so tile seams stay measurable.
    pub draw_ocean: bool,
}

/// Rerun request from the diagnostics panel's "Regenerate erosion cache"
/// button; consumed exactly once by the erosion stream system.
#[derive(Resource, Default)]
struct RerunErosion(pub bool);

/// The C++ pre-loop prewarm: `for (int pass = 0; pass < 4; ++pass)
/// UpdateErosionCache(erosion, noise, appliedErosionSettings, {0,warm0,0},
/// appliedErosionSettings.iterations, true)`. ChooseNextErosionTile
/// prioritizes the quartet covering the player, so four full-budget
/// immediate-reveal passes leave the complete four-pass blend ready at spawn.
/// The port simulates one tile per pass across several frames, so this counts
/// the passes still owed.
#[derive(Resource)]
struct PrewarmRemaining(pub u32);

/// `for (int pass = 0; pass < 4; ++pass)` in the C++.
const PREWARM_PASSES: u32 = 4;

#[derive(Resource, Default)]
struct FrameCounter(pub u64);

#[derive(Resource, Default)]
struct ShotRequested(pub bool);

fn main() {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let automation = automation::parse_automation(arguments.into_iter());

    // --probe dumps the base landform as CSV and exits with no window.
    if automation.probe {
        noise::run_probe(automation.probe_extent, automation.probe_step);
        return;
    }

    let noise_field = NoiseField::new();

    let startup_settings = {
        let mut settings = AppSettings::default();
        if automation.lattice {
            settings.erosion_debug = true;
        }
        if automation.no_fog {
            settings.fog_density = 0.0;
        }
        settings
    };

    App::new()
        .add_plugins(
            bevy::DefaultPlugins
                .set(bevy::window::WindowPlugin {
                    primary_window: Some(Window {
                        title: "Forest - Infinite Procedural Terrain".into(),
                        resizable: true,
                        // InitWindow(automation.width, automation.height, ...).
                        // Those are the C++'s *screen* (logical) dimensions:
                        // it passes FLAG_WINDOW_HIGHDPI (main.cpp:2253), so
                        // rcore_desktop_glfw.c scales the render size by
                        // GetWindowScaleDPI() and ResizeSSAO/GetRenderWidth
                        // (main.cpp:2233-2234, 2332) then run every 3D pass
                        // and the screenshot at that doubled size. Leaving
                        // scale_factor_override unset reproduces the request:
                        // WindowResolution keeps the requested numbers as its
                        // physical fields while its scale factor stays 1.0, so
                        // width() is W and bevy_winit passes winit a
                        // LogicalSize(W, H) — exactly raylib's W x H screen
                        // points, which macOS backs with a 2W x 2H framebuffer
                        // on a Retina panel.
                        //
                        // Overriding to 1.0 instead takes bevy_winit's
                        // to_physical branch and asks winit for a
                        // PhysicalSize(W, H), i.e. a window W/2 by H/2 points
                        // on a 2x panel — half raylib's screen size, and a
                        // request raylib never makes. It does NOT change the
                        // captured pixels: bevy_winit calls
                        // set_scale_factor_and_apply_to_physical_size right
                        // after creating the window (bevy_window/src/
                        // window.rs:1017-1021 multiplies physical_width and
                        // physical_height by the OS scale factor), and the
                        // render targets and camera are sized from
                        // physical_size(), so a 2x panel gave 2W x 2H either
                        // way — measured with both binaries on this Retina
                        // panel: --size 400,300 wrote 800x600 PNGs and the
                        // default size 2940x1782 from both. The override was
                        // therefore a window-geometry error, not a capture
                        // error, and removing it is what makes the two
                        // binaries agree on the window they ask for.
                        resolution: WindowResolution::new(
                            automation.width as u32,
                            automation.height as u32,
                        ),
                        ..default()
                    }),
                    exit_condition: bevy::window::ExitCondition::OnPrimaryClosed,
                    close_when_requested: true,
                    ..default()
                })
                .set(bevy::asset::AssetPlugin {
                    file_path: resolve_asset_root(),
                    ..default()
                })
                // The port filters R32Float (the noise field) and Rgba32Float
                // (the erosion atlases and the G-buffer position target)
                // through linear samplers; wgpu only permits that when the
                // device advertises float32 filtering, which bevy does not
                // request by default. TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
                // is bevy's own default and must be kept alongside it.
                .set(RenderPlugin {
                    render_creation: RenderCreation::Automatic(WgpuSettings {
                        features: wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
                            | wgpu::Features::FLOAT32_FILTERABLE,
                        ..default()
                    }),
                    ..default()
                }),
        )
        .add_plugins(EguiPlugin::default())
        // Installs the render-world state the flow-atlas paint callback builds
        // its pipeline into (see FlowPreviewRenderState in src/ui.rs).
        .add_plugins(ui::UiRenderPlugin)
        .insert_resource(startup_settings)
        .init_resource::<ErosionCache>()
        .insert_resource(ErosionSettings::default())
        .insert_resource(AppliedErosionSettings(ErosionSettings::default()))
        .insert_resource(RerunErosion::default())
        .insert_resource(PrewarmRemaining(PREWARM_PASSES))
        .insert_resource(FrameCounter::default())
        .insert_resource(ShotRequested::default())
        .init_resource::<player::UiWantsInput>()
        .insert_resource(WorldOptions {
            draw_ocean: !automation.no_water,
        })
        .insert_resource(ErosionBridge::default())
        .insert_resource(noise_field.clone())
        .insert_resource(automation.clone())
        // The render plugin lifts the shared noise field and erosion bridge
        // into the render sub-app at build time, so they must exist first.
        .add_plugins(render::ForestRenderPlugin::new(noise_field.clone()))
        .add_systems(Startup, spawn_scene)
        .add_systems(
            Startup,
            (
                setup_blend_mask,
                setup_cursor_and_player,
            )
                // The player entity must exist before the pose is pinned.
                .chain()
                .after(spawn_scene),
        )
        .add_systems(
            Update,
            (
                handle_global_keys,
                ui::ui_wants_input_system,
                // --measure-overlap replaces normal streaming with its own
                // driver, mirroring the C++ short-circuit in main().
                erosion_stream_system.run_if(not_in_measure_mode),
                player::update_player_system,
                player::sync_player_camera_transform,
                measure_overlap_system.run_if(in_measure_mode),
                shot_scheduling_system,
            )
                .chain(),
        )
        .add_systems(EguiPrimaryContextPass, ui::draw_diagnostics_ui)
        .run();
}

/// The asset root the C++ used: `AssetPath` resolved `assets/` against the
/// working directory. Bevy instead uses `CARGO_MANIFEST_DIR` (under
/// `cargo run`) or the executable's directory (an installed binary), so accept
/// whichever candidate actually holds the shaders and fall back to bevy's
/// default. This keeps `./target/release/realistic_forest` working from the
/// project root the way `./build/forest` does.
fn resolve_asset_root() -> String {
    let candidates = [
        std::env::current_dir().ok().map(|dir| dir.join("assets")),
        std::env::var("CARGO_MANIFEST_DIR")
            .ok()
            .map(|dir| std::path::PathBuf::from(dir).join("assets")),
        std::env::current_exe()
            .ok()
            .and_then(|exe| exe.parent().map(|dir| dir.join("assets"))),
    ];
    for candidate in candidates.into_iter().flatten() {
        if candidate.join("shaders").join("terrain-vs.wgsl").is_file() {
            return candidate.to_string_lossy().into_owned();
        }
    }
    "assets".to_string()
}

/// The camera and the player. This is the only camera entity in the app, so
/// bevy_egui's auto-creation attaches the primary egui context to it.
fn spawn_scene(mut commands: Commands) {
    commands.spawn(Player::default());
    commands.spawn((
        Transform::default(),
        Projection::Perspective(PerspectiveProjection {
            fov: 68.0_f32.to_radians(),
            ..default()
        }),
        Camera {
            // The composite pass fully covers the view every frame, matching
            // the C++ render pass into the deferred G-buffer.
            clear_color: ClearColorConfig::None,
            target: RenderTarget::Window(WindowRef::Primary),
            ..default()
        },
        CameraRenderGraph::new(render::ForestSubGraph),
        Frustum::default(),
        VisibleEntities::default(),
        Msaa::Off,
    ));
}

/// The CPU half of `CreateErosionBlendMask`: the per-texel mask that erosion
/// tile borders share, sampled on the CPU for player collision and streamed
/// to the GPU once in the render world.
fn setup_blend_mask(mut cache: ResMut<ErosionCache>) {
    cache.blend_mask_samples = noise::erosion_blend_mask_samples();
}

/// Applies the automation startup bits the C++ main does before the loop:
/// the ground-level spawn height, the pinned pose for `--camera`, and the
/// cursor state for unattended runs (DisableCursor in the normal case).
fn setup_cursor_and_player(
    automation: Res<AutomationSettings>,
    mut local: Local<bool>,
    cache: Res<ErosionCache>,
    noise: Res<NoiseField>,
    mut player: Query<&mut Player>,
    mut windows: Query<(&mut bevy::window::CursorOptions,), With<PrimaryWindow>>,
) {
    if *local {
        return;
    }
    // The one-shot latch is set only once the queries actually resolve. Both
    // the window entity (WindowPlugin) and the player entity (spawn_scene)
    // are created in Startup, so an early run can legitimately find neither;
    // latching before the `?`s would then burn the one attempt — and the
    // pinned --camera pose, the whole point of an unattended capture, would
    // silently never be applied.
    let Ok((mut cursor,)) = windows.single_mut() else {
        return;
    };
    let Ok(mut player) = player.single_mut() else {
        return;
    };
    *local = true;
    // Ground-level spawn, matching the C++ pre-loop player placement.
    if !automation.has_camera {
        player.position.y = erosion::sample_eroded_height(
            &cache, &noise, 0.0, 0.0, [0.0, 0.0],
        ) + EYE_HEIGHT;
        // `if (automation.shotPath == nullptr) DisableCursor();` (main.cpp:2261)
        // is the whole of the C++'s cursor special-casing for automation, and
        // it touches the cursor only. mouseCaptured is left at its default
        // true (main.cpp:247), which is what keeps DrawCrosshair
        // (main.cpp:2414, `mouseCaptured && !showUi`, and showUi starts false
        // at main.cpp:256) drawing its two arms through the centre of a
        // --shot frame taken without --camera. Clearing mouseCaptured here
        // suppressed that crosshair, so those captures were missing it.
        // --camera is the case that clears it, and that happens below.
        if automation.shot_path.is_none() {
            cursor.grab_mode = bevy::window::CursorGrabMode::Locked;
            cursor.visible = false;
        }
        return;
    }
    // Fly so gravity never fights a pinned pose; leave the cursor alone so
    // the capture window does not fight mouse input.
    player.position = bevy::math::Vec3::from(automation.position);
    player.yaw = automation.yaw;
    player.pitch = automation.pitch;
    player.flying = true;
    player.mouse_captured = false;
}

/// F1 toggles the diagnostics panel (releasing the pointer when it opens,
/// matching the C++); F12 saves a timestamped screenshot.
fn handle_global_keys(
    mut settings: ResMut<AppSettings>,
    mut player: Query<&mut Player>,
    mut cursor_options: Query<(&mut CursorOptions,), With<PrimaryWindow>>,
    keyboard: Res<ButtonInput<KeyCode>>,
    mut commands: Commands,
) {
    if keyboard.just_pressed(KeyCode::F1) {
        settings.show_ui = !settings.show_ui;
        if settings.show_ui {
            if let Ok(mut player) = player.single_mut() {
                player.mouse_captured = false;
            }
            if let Ok((mut cursor,)) = cursor_options.single_mut() {
                cursor.grab_mode = bevy::window::CursorGrabMode::None;
                cursor.visible = true;
            }
        }
    }
    if keyboard.just_pressed(KeyCode::F12) {
        commands
            .spawn(Screenshot::primary_window())
            .observe(on_manual_screenshot);
    }
}

/// F12 capture: save a timestamped screenshot next to the working directory.
fn on_manual_screenshot(trigger: On<ScreenshotCaptured>) {
    let name = chrono::Local::now().format("forest_%Y%m%d_%H%M%S.png").to_string();
    write_png(&trigger.image, Path::new(&name));
    log::info!("SCREENSHOT: saved {name}");
}

/// Serializes a captured bevy Image as a PNG byte file.
fn write_png(image: &Image, path: &Path) {
    let bytes = match image_converter_png(image) {
        Ok(bytes) => bytes,
        Err(error) => {
            log::error!("SCREENSHOT: encode failed: {error}");
            return;
        }
    };
    if let Err(error) = std::fs::write(path, bytes) {
        log::error!("SCREENSHOT: write failed: {error}");
    }
}

fn image_converter_png(image: &Image) -> Result<Vec<u8>, String> {
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
                .write_to(&mut std::io::Cursor::new(&mut encoded), image::ImageFormat::Png)
                .map_err(|error| error.to_string())?;
            Ok(encoded)
        }
        _ => Err("unsupported capture format".to_string()),
    }
}

/// One frame of erosion streaming around the player, mirroring the C++
/// ordering: apply finished readbacks, honour a UI rerun request, then feed
/// the cache with the PRE-update player position.
#[allow(clippy::too_many_arguments)]
fn erosion_stream_system(
    mut cache: ResMut<ErosionCache>,
    bridge: Res<ErosionBridge>,
    noise: Res<NoiseField>,
    erosion_settings: Res<ErosionSettings>,
    mut applied: ResMut<AppliedErosionSettings>,
    mut rerun: ResMut<RerunErosion>,
    mut prewarm: ResMut<PrewarmRemaining>,
    player: Query<&Player>,
    time: Res<Time>,
) {
    let Ok(player) = player.single() else {
        return;
    };
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
    let budget = if prewarming { applied.0.iterations } else { EROSION_ITERATIONS_PER_FRAME };
    let began = erosion::update_erosion_cache(
        &mut cache,
        &bridge,
        &noise,
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

/// `--measure-overlap` replaces the normal stream/measure pair, exactly as the
/// C++ `main` returns into `RunOverlapMeasurement` before its render loop.
fn in_measure_mode(automation: Res<AutomationSettings>) -> bool {
    automation.measure_overlap
}

fn not_in_measure_mode(automation: Res<AutomationSettings>) -> bool {
    !automation.measure_overlap
}

/// Unattended `--measure-overlap` driver: simulates the requested tile and
/// its +x neighbour, prints the seam CSV, and exits. Replaces normal
/// streaming, mirroring the C++ short-circuit in main().
fn measure_overlap_system(
    mut cache: ResMut<ErosionCache>,
    bridge: Res<ErosionBridge>,
    noise: Res<NoiseField>,
    applied: Res<AppliedErosionSettings>,
    automation: Res<AutomationSettings>,
    mut exits: MessageWriter<AppExit>,
    // `AppExit` is applied after the schedule finishes, not the instant it is
    // written, so this system can still run again and print the whole CSV a
    // second time. The C++ returns straight out of `main` and prints once.
    mut done: Local<bool>,
) {
    if *done {
        return;
    }
    erosion::apply_erosion_events(&mut cache, &bridge);

    let first = automation.overlap_tile;
    let second = TileKey { x: first.x + 1, z: first.z };
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
        exits.write(AppExit::Success);
        *done = true;
        return;
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

/// Unattended capture: let the erosion cache stream in around the pinned
/// camera for `--wait` frames, then save one screenshot and exit.
fn shot_scheduling_system(
    automation: Res<AutomationSettings>,
    mut counter: ResMut<FrameCounter>,
    mut requested: ResMut<ShotRequested>,
    mut commands: Commands,
) {
    let Some(shot_path) = automation.shot_path.clone() else {
        return;
    };
    counter.0 += 1;
    if requested.0 || counter.0 < automation.wait_frames as u64 {
        // A request already made and not yet captured; AppExit is written by
        // the capture observer once the image lands.
        return;
    }
    requested.0 = true;
    // The observer outlives this system, so it owns the path rather than
    // borrowing the resource.
    commands
        .spawn(Screenshot::primary_window())
        .observe(move |trigger: On<ScreenshotCaptured>, mut exits: MessageWriter<AppExit>| {
            write_png(&trigger.image, Path::new(&shot_path));
            log::info!("SCREENSHOT: saved {shot_path}");
            exits.write(AppExit::Success);
        });
}