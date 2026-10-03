//! Application assembly for the Bevy + wgpu port of ~/forest (raylib/GL).
//!
//! Frame order mirrors the C++ main loop: keyed settings changes, the egui
//! input check, erosion streaming (which reads the pre-update player
//! position), the player update, and then the render schedule (erosion sim ->
//! terrain G-buffer -> SSAO -> blur -> composite -> FXAA -> egui -> upscale).

mod automation;
mod constants;
mod day_night;
mod erosion;
mod grass;
mod grass_cull;
mod lightning;
mod matrices;
mod noise;
mod player;
mod render;
mod thunder;
mod ui;
mod vegetation;
mod water;
mod weather;

use crate::automation::AutomationSettings;
use crate::constants::*;
use crate::erosion::{ErosionBridge, ErosionCache, IterateCommand, TileKey};
use crate::noise::NoiseField;
use crate::player::Player;
use bevy::camera::primitives::Frustum;
use bevy::camera::visibility::VisibleEntities;
use bevy::camera::{Camera, ClearColorConfig, PerspectiveProjection, Projection};
use bevy::audio::AddAudioSource;
use bevy::prelude::*;
use bevy::render::camera::CameraRenderGraph;
use bevy::render::settings::{RenderCreation, WgpuSettings};
use bevy::render::view::window::screenshot::{Screenshot, ScreenshotCaptured};
use bevy::render::view::Msaa;
use bevy::render::RenderPlugin;
use bevy::window::{CursorOptions, PrimaryWindow, WindowResolution};
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
/// the passes still owed; `--erosion-prewarm` raises it for captures.
#[derive(Resource)]
struct PrewarmRemaining(pub u32);

#[derive(Resource, Default)]
struct FrameCounter(pub u64);

#[derive(Resource, Default)]
struct ShotRequested(pub bool);

/// `--size` reproduces raylib's `InitWindow(w, h)`: those are *screen points*,
/// and raylib then renders at `screen x GetWindowScaleDPI()`. But raylib does
/// not hand the request straight to GLFW — it clamps it to the *primary*
/// monitor's work area first, in software, before the window exists:
/// `glfwGetMonitorWorkarea(glfwGetPrimaryMonitor(), ...)` followed by
/// `if (CORE.Window.screen.width > workWidth) CORE.Window.screen.width = workWidth;`
/// (rcore_desktop_glfw.c:1618, 1671-1679; raylib's own comment there explains
/// why — a GLFW window larger than the work area would not show up). On macOS
/// that work area is `[NSScreen visibleFrame]` (cocoa_monitor.m:487-499), the
/// space the menu bar and Dock leave over — and raylib applies it whichever
/// display the window actually opens on. Measured with the window on the
/// 1920x1080 external display, the reference still logged "Screen size:
/// 1470 x 923" for `--size 3000,2000`: the built-in panel's visibleFrame
/// exactly, and nothing the 1920x1080 display would have produced.
///
/// The port had no equivalent step, and that is the whole of the `--size`
/// divergence. winit exposes no work-area query — its macOS `MonitorHandle`
/// offers only `size()`, the full video mode — so nothing constrains the
/// request, and on a 1x display AppKit has no reason to either. Measured at
/// 1x: `--size 1600,900` wrote 1600x900 from the port against 1470x900 from
/// the reference. Adopting the backend's content size afterwards cannot close
/// that gap, because at 1x the backend genuinely is the unclamped 1600x900
/// (probed), so there is nothing to adopt.
///
/// Units are AppKit points, which is both what raylib clamps and what
/// `WindowResolution::new` is handed below.
#[cfg(target_os = "macos")]
fn clamp_to_primary_work_area(width: u32, height: u32) -> (u32, u32) {
    use objc2::MainThreadMarker;
    use objc2_app_kit::NSScreen;

    // NSScreen is main-thread-only and `main` is the main thread; the marker
    // is how objc2 proves it.
    let Some(mtm) = MainThreadMarker::new() else {
        eprintln!("WINDOW: not on the main thread, leaving the screen size unclamped");
        return (width, height);
    };
    let Some(primary) = NSScreen::screens(mtm).firstObject() else {
        eprintln!("WINDOW: no screens reported, leaving the screen size unclamped");
        return (width, height);
    };
    let work = primary.visibleFrame();
    let work_width = work.size.width as u32;
    let work_height = work.size.height as u32;
    // raylib floors a zero dimension at one pixel rather than clamping to zero
    // (rcore_desktop_glfw.c:1665-1667), so a screen reporting no work area
    // should not collapse the window either.
    if work_width == 0 || work_height == 0 {
        return (width, height);
    }
    let clamped_width = width.min(work_width);
    let clamped_height = height.min(work_height);
    // `println!` rather than `log::info!`: bevy installs its logger along with
    // DefaultPlugins, which this runs well before, so anything logged here is
    // silently dropped. The reference prints the same figure from InitWindow —
    // it is its "Screen size:" line — so this keeps the two capture logs
    // directly comparable.
    if clamped_width != width || clamped_height != height {
        println!(
            "WINDOW: screen size {clamped_width} x {clamped_height} (requested {width} x {height}, clamped to the primary monitor work area)"
        );
    } else {
        println!("WINDOW: screen size {clamped_width} x {clamped_height}");
    }
    (clamped_width, clamped_height)
}

/// Everywhere else the request is used as-is: winit has no work-area query to
/// make, and the port carries no per-platform one.
#[cfg(not(target_os = "macos"))]
fn clamp_to_primary_work_area(width: u32, height: u32) -> (u32, u32) {
    (width, height)
}

fn main() {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let automation = automation::parse_automation(arguments.into_iter());

    // --probe dumps the base landform as CSV and exits with no window.
    if automation.probe {
        noise::run_probe(automation.probe_extent, automation.probe_step);
        return;
    }
    // --vegetation-map charts the plant scatter and exits with no window.
    if let Some(path) = &automation.vegetation_map {
        vegetation::map::run_map(path, automation.map_centre, automation.map_extent);
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
        if automation.no_raymarch {
            settings.raymarched_shadows = false;
            settings.volumetric_lighting = false;
            settings.water_reflections = false;
        }
        settings.clouds_enabled = !automation.no_clouds && !automation.no_raymarch;
        settings.cloud_coverage = automation.cloud_coverage;
        settings.cloud_density = automation.cloud_density;
        settings.cloud_base_height = automation.cloud_base_height;
        settings.cloud_thickness = automation.cloud_thickness;
        settings.raymarch_quality = automation.raymarch_quality;
        settings
    };

    // The size raylib would have handed to glfwCreateWindow, clamped to the
    // primary monitor's work area the way raylib clamps it.
    let (screen_width, screen_height) =
        clamp_to_primary_work_area(automation.width as u32, automation.height as u32);

    App::new()
        .add_plugins(
            bevy::DefaultPlugins
                .set(bevy::window::WindowPlugin {
                    primary_window: Some(Window {
                        title: "Forest - Infinite Procedural Terrain".into(),
                        resizable: true,
                        // InitWindow(screen_width, screen_height, ...) — the
                        // clamped request, see clamp_to_primary_work_area.
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
                        resolution: WindowResolution::new(screen_width, screen_height),
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
                    render_creation: RenderCreation::Automatic(Box::new(WgpuSettings {
                        features: wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
                            | wgpu::Features::FLOAT32_FILTERABLE,
                        ..default()
                    })),
                    ..default()
                }),
        )
        .add_plugins(EguiPlugin::default())
        // Installs the render-world state the flow-atlas paint callback builds
        // its pipeline into (see FlowPreviewRenderState in src/ui.rs).
        .add_plugins(ui::UiRenderPlugin)
        .insert_resource(startup_settings)
        .insert_resource(day_night::DayNightCycle {
            time_hours: automation.time_of_day,
            paused: automation.pause_time,
            day_length_minutes: automation.day_length_minutes,
        })
        .init_resource::<weather::WeatherMotion>()
        .init_resource::<lightning::ActiveBolt>()
        .init_resource::<thunder::ThunderQueue>()
        .add_audio_source::<thunder::ThunderSound>()
        .insert_resource({
            let mut weather = weather::WeatherState::from_preset(
                automation.weather_preset,
                !automation.static_weather,
            );
            weather.overrides = weather::WeatherOverrides {
                coverage: automation.cloud_coverage_override,
                density: automation.cloud_density_override,
                base: automation.cloud_base_override,
                thickness: automation.cloud_thickness_override,
            };
            weather
        })
        .init_resource::<ErosionCache>()
        .insert_resource(ErosionSettings::default())
        .insert_resource(AppliedErosionSettings(ErosionSettings::default()))
        .insert_resource(RerunErosion::default())
        .insert_resource(PrewarmRemaining(automation.erosion_prewarm))
        .insert_resource(FrameCounter::default())
        .insert_resource(ShotRequested::default())
        .init_resource::<player::UiWantsInput>()
        .insert_resource(WorldOptions {
            draw_ocean: !automation.no_water,
        })
        .insert_resource(ErosionBridge::default())
        .insert_resource(erosion::TilePreparation::new(std::sync::Arc::new(noise_field.clone())))
        .insert_resource(vegetation::VegetationField::new(
            std::sync::Arc::new(noise_field.clone()),
            !automation.no_vegetation,
        ))
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
                day_night::advance_day_night,
                weather::advance_weather,
                thunder::queue_thunder,
                thunder::play_thunder,
                ui::ui_wants_input_system,
                // --measure-overlap replaces normal streaming with its own
                // driver, mirroring the C++ short-circuit in main().
                erosion_stream_system.run_if(not_in_measure_mode),
                player::update_player_system,
                player::sync_player_camera_transform,
                vegetation::stream_vegetation,
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
/// project root the way `./build/forest` does, and a packaged build working
/// from any directory. Every runtime file read under `assets/` goes through
/// here; a bare `"assets/..."` path only works from the project root.
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
            ..default()
        },
        CameraRenderGraph::new(render::ForestRender),
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
        log::warn!("AUTOMATION: no primary window yet; retrying the startup pose");
        return;
    };
    let Ok(mut player) = player.single_mut() else {
        log::warn!("AUTOMATION: no player entity yet; retrying the startup pose");
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
    log::info!(
        "SHOT: pose pinned at {:.1},{:.1},{:.1} yaw {:.1} pitch {:.1}",
        player.position.x,
        player.position.y,
        player.position.z,
        player.yaw.to_degrees(),
        player.pitch.to_degrees(),
    );
}

/// Tab opens the compact trainer, F1 opens the full diagnostics panel, and
/// either action releases the pointer so the controls can be used immediately.
/// F12 saves a timestamped screenshot.
fn handle_global_keys(
    mut settings: ResMut<AppSettings>,
    mut player: Query<&mut Player>,
    mut cursor_options: Query<(&mut CursorOptions,), With<PrimaryWindow>>,
    keyboard: Res<ButtonInput<KeyCode>>,
    mut commands: Commands,
) {
    let mut opened_panel = false;
    if keyboard.just_pressed(KeyCode::Tab) {
        settings.show_trainer = !settings.show_trainer;
        opened_panel |= settings.show_trainer;
    }
    if keyboard.just_pressed(KeyCode::F1) {
        settings.show_ui = !settings.show_ui;
        opened_panel |= settings.show_ui;
    }
    if opened_panel {
        if let Ok(mut player) = player.single_mut() {
            player.mouse_captured = false;
        }
        if let Ok((mut cursor,)) = cursor_options.single_mut() {
            cursor.grab_mode = bevy::window::CursorGrabMode::None;
            cursor.visible = true;
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

/// Serializes a captured bevy Image as a PNG byte file. Returns whether the
/// file was written, so the `--shot` path can report failure in its exit code.
fn write_png(image: &Image, path: &Path) -> bool {
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
    mut preparation: ResMut<erosion::TilePreparation>,
    automation: Res<AutomationSettings>,
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
) {
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

/// Frame times a `--shot` run collects, split by whether erosion tiles were
/// still streaming in that frame.
#[derive(Default)]
struct FrameTimes {
    streaming: Vec<f32>,
    settled: Vec<f32>,
}

impl FrameTimes {
    fn record(&mut self, seconds: f32, streaming: bool) {
        if streaming {
            self.streaming.push(seconds);
        } else {
            self.settled.push(seconds);
        }
    }

    fn log(&self) {
        for (phase, samples) in [("streaming", &self.streaming), ("settled", &self.settled)] {
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
        }
    }
}

/// Unattended capture: let the erosion cache stream in around the pinned
/// camera for `--wait` frames, then save one screenshot and exit.
fn shot_scheduling_system(
    automation: Res<AutomationSettings>,
    cache: Res<ErosionCache>,
    vegetation: Res<vegetation::VegetationField>,
    mut counter: ResMut<FrameCounter>,
    mut requested: ResMut<ShotRequested>,
    mut commands: Commands,
    time: Res<Time<Real>>,
    mut frame_times: Local<FrameTimes>,
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
        frame_times.record(time.delta_secs(), streaming);
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
    commands
        .spawn(Screenshot::primary_window())
        .observe(move |trigger: On<ScreenshotCaptured>| {
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
        });
}
