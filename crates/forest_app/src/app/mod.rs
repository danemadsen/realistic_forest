//! Application assembly and systems that connect domain crates.

mod capture;
mod erosion_systems;
mod input;
mod startup;
mod ui;
mod vegetation_systems;
mod window;

use crate::{
    OffscreenCapture, WorldOptions, asset_io, automation, buildings, day_night, erosion, grass,
    lightning, noise, player, render, resolve_asset_root, rivers, snow, thunder, vegetation,
    weather,
};
use capture::{on_manual_screenshot, shot_scheduling_system};
use erosion_systems::{
    erosion_stream_system, in_measure_mode, measure_overlap_system, not_in_measure_mode,
    publish_erosion_shader_sources,
};
use input::{CopyPose, copy_pose_to_clipboard, handle_global_keys};
use startup::{setup_blend_mask, setup_cursor_and_player, spawn_scene};
use vegetation_systems::stream_vegetation;
use window::clamp_to_primary_work_area;

use crate::automation::AutomationSettings;
use crate::constants::*;
use crate::erosion::{ErosionBridge, ErosionCache, ErosionShaderSources, IterateCommand, TileKey};
use crate::noise::NoiseField;
use crate::player::Player;
use bevy::audio::AddAudioSource;
use bevy::camera::primitives::Frustum;
use bevy::camera::visibility::VisibleEntities;
use bevy::camera::{Camera, ClearColorConfig, PerspectiveProjection, Projection};
use bevy::clipboard::Clipboard;
use bevy::prelude::*;
use bevy::render::RenderPlugin;
use bevy::render::camera::CameraRenderGraph;
use bevy::render::settings::{RenderCreation, WgpuSettings};
use bevy::render::view::Msaa;
use bevy::render::view::window::screenshot::{Screenshot, ScreenshotCaptured};
use bevy::window::{CursorOptions, PrimaryWindow, WindowResolution};
use bevy_egui::input::EguiWantsInput;
use bevy_egui::{EguiPlugin, EguiPostUpdateSet, EguiPrimaryContextPass};
use std::path::Path;

/// Erosion parameters in force for the currently streaming cache; edited by
/// the diagnostics panel, copied in when "Regenerate erosion cache" reruns.
#[derive(Resource)]
struct AppliedErosionSettings(pub ErosionSettings);

/// Rerun request from the diagnostics panel's "Regenerate erosion cache"
/// button; consumed exactly once by the erosion stream system.
#[derive(Resource, Default)]
pub(super) struct RerunErosion(pub bool);

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

pub(super) fn run() {
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

    // --river-probe samples the carved ground around a point and exits with
    // no window. `--probe-step` defaults to the landform probe's 128 m, far
    // too coarse for a channel, so anything that coarse means its own 0.5 m.
    if let Some(centre) = automation.river_probe {
        let extent = if automation.map_extent_set {
            automation.map_extent
        } else {
            128.0
        };
        let step = if automation.probe_step < 64.0 {
            automation.probe_step as f64
        } else {
            0.5
        };
        rivers::map::run_probe(&noise_field, centre, extent, step, &automation.probe_out);
        return;
    }
    // --river-map charts the river network and exits with no window.
    if let Some(path) = &automation.river_map {
        let extent = if automation.map_extent_set {
            automation.map_extent
        } else {
            8192.0
        };
        rivers::map::run_map(&noise_field, path, automation.map_centre, extent);
        return;
    }
    if let Err(error) = asset_io::initialize() {
        eprintln!("ASSETS: {error}");
        std::process::exit(1);
    }
    // --building-map charts the settlements and exits with no window.
    if let Some(path) = &automation.building_map {
        let extent = if automation.map_extent_set {
            automation.map_extent
        } else {
            8192.0
        };
        buildings::map::run_map(&noise_field, path, automation.map_centre, extent);
        return;
    }

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

    // The rivers of the region around spawn are built before anything
    // streams: the first erosion tiles, the plants and the player's footing
    // all carve and avoid their channels.
    let river_field = {
        let river_noise = std::sync::Arc::new(noise_field.clone());
        let mut field = rivers::RiverField::new(river_noise, !automation.no_rivers);
        let spawn = if automation.has_camera {
            automation.position
        } else {
            [0.0, 0.0, 0.0]
        };
        field.build_blocking(spawn[0], spawn[2]);
        // Bevy's logger does not exist yet; the summary goes straight out.
        if let Some(network) = field.network() {
            println!(
                "RIVERS: {} rivers, {:.1} km of channel, {} lakes, built in {:.2} s ({})",
                network.rivers.len(),
                network.total_length_km(),
                network.lakes.len(),
                network.build_seconds,
                network.timings
            );
        }
        field
    };

    // The buildings around spawn are placed before anything streams too: the
    // ground the player first stands on is already levelled under them.
    let building_field = {
        let spawn = if automation.has_camera {
            automation.position
        } else {
            [0.0, 0.0, 0.0]
        };
        let models = std::path::PathBuf::from(resolve_asset_root())
            .join("models")
            .join("buildings");
        let mut field = buildings::BuildingField::new(
            std::sync::Arc::new(noise_field.clone()),
            river_field.network(),
            &models,
            bevy::math::Vec2::new(spawn[0], spawn[2]),
            !automation.no_buildings,
        );
        if automation.open_doors {
            field.hold_doors_open();
        }
        field
    };

    // The size raylib would have handed to glfwCreateWindow, clamped to the
    // primary monitor's work area the way raylib clamps it. An offscreen
    // capture has no window to fit on a screen.
    let (screen_width, screen_height) = if automation.offscreen && automation.shot_path.is_some() {
        (automation.width as u32, automation.height as u32)
    } else {
        clamp_to_primary_work_area(automation.width as u32, automation.height as u32)
    };

    let window = Window {
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
    };
    // An offscreen capture opens no window at all: the camera renders into an
    // image, and the app runs on a plain loop instead of winit's.
    let offscreen = automation.offscreen && automation.shot_path.is_some();
    let window_plugin = if offscreen {
        bevy::window::WindowPlugin {
            primary_window: None,
            exit_condition: bevy::window::ExitCondition::DontExit,
            close_when_requested: false,
            ..default()
        }
    } else {
        bevy::window::WindowPlugin {
            primary_window: Some(window),
            exit_condition: bevy::window::ExitCondition::OnPrimaryClosed,
            close_when_requested: true,
            ..default()
        }
    };
    // The port filters R32Float (the noise field) and Rgba32Float
    // (the erosion atlases and the G-buffer position target)
    // through linear samplers; wgpu only permits that when the
    // device advertises float32 filtering, which bevy does not
    // request by default. TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
    // is bevy's own default and must be kept alongside it.
    let render_plugin = RenderPlugin {
        render_creation: RenderCreation::Automatic(Box::new(WgpuSettings {
            features: wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
                | wgpu::Features::FLOAT32_FILTERABLE,
            ..default()
        })),
        ..default()
    };
    let mut default_plugins = bevy::DefaultPlugins.build();
    default_plugins = default_plugins.set(window_plugin);
    default_plugins = default_plugins.set(bevy::asset::AssetPlugin {
        file_path: resolve_asset_root(),
        ..default()
    });
    default_plugins = default_plugins.set(render_plugin);
    if offscreen {
        default_plugins = default_plugins.disable::<bevy::winit::WinitPlugin>();
    }

    let mut app = App::new();
    app.add_plugins(default_plugins);
    if offscreen {
        app.add_plugins(bevy::app::ScheduleRunnerPlugin::run_loop(
            std::time::Duration::ZERO,
        ));
        app.insert_resource(OffscreenCapture {
            image: None,
            size: [
                automation.width.max(1) as u32,
                automation.height.max(1) as u32,
            ],
        });
    }
    app.add_plugins(EguiPlugin::default());
    // Developer teleports update the camera during the egui pass. Run
    // that pass before propagation so visibility sees the same pose.
    app.configure_sets(
        PostUpdate,
        EguiPostUpdateSet::EndPass.before(bevy::transform::TransformSystems::Propagate),
    );
    // Installs the render-world state the flow-atlas paint callback builds
    // its pipeline into (see FlowPreviewRenderState in crates/forest_app/src/app/ui.rs).
    app.add_plugins(ui::UiRenderPlugin);
    app.insert_resource(startup_settings);
    app.insert_resource(day_night::DayNightCycle {
        time_hours: automation.time_of_day,
        paused: automation.pause_time,
        day_length_minutes: automation.day_length_minutes,
    });
    app.init_resource::<weather::WeatherMotion>();
    app.init_resource::<lightning::ActiveBolt>();
    app.init_resource::<thunder::ThunderQueue>();
    app.add_audio_source::<thunder::ThunderSound>();
    let mut weather =
        weather::WeatherState::from_preset(automation.weather_preset, !automation.static_weather);
    weather.overrides = weather::WeatherOverrides {
        coverage: automation.cloud_coverage_override,
        density: automation.cloud_density_override,
        base: automation.cloud_base_override,
        thickness: automation.cloud_thickness_override,
    };
    app.insert_resource(weather);
    app.insert_resource(ErosionCache {
        rivers: river_field.network().cloned(),
        buildings: Some(building_field.pads().clone()),
        ..Default::default()
    });
    app.insert_resource(river_field);
    app.insert_resource(building_field);
    app.init_resource::<buildings::EyeAdaptation>();
    app.init_resource::<snow::SnowState>();
    app.insert_resource(ErosionSettings::default());
    app.insert_resource(AppliedErosionSettings(ErosionSettings::default()));
    app.insert_resource(RerunErosion::default());
    app.insert_resource(PrewarmRemaining(automation.erosion_prewarm));
    app.insert_resource(FrameCounter::default());
    app.insert_resource(ShotRequested::default());
    app.init_resource::<player::UiWantsInput>();
    app.add_message::<CopyPose>();
    app.insert_resource(WorldOptions {
        draw_ocean: !automation.no_water,
    });
    app.insert_resource(ErosionBridge::default());
    // The render world spawns the erosion worker thread from this bridge
    // (after the WGSL publisher's sources arrive). The thread is detached
    // and never joined: every exit path — normal shutdown or `--shot`'s
    // `std::process::exit`, which skips bevy teardown — simply drops the
    // Arc'd state it holds, so nothing at exit can block on a worker that
    // is mid-simulation.
    let worker_noise = std::sync::Arc::new(noise_field.clone());
    app.insert_resource(erosion::TilePreparation::new(worker_noise));
    let vegetation_noise = std::sync::Arc::new(noise_field.clone());
    app.insert_resource(vegetation::VegetationField::new(
        vegetation_noise,
        !automation.no_vegetation,
    ));
    app.insert_resource(noise_field.clone());
    app.insert_resource(automation.clone());
    // The render plugin lifts the shared noise field and erosion bridge
    // into the render sub-app at build time, so they must exist first.
    app.add_plugins(render::ForestRenderPlugin::new(noise_field.clone()));
    app.add_systems(Startup, spawn_scene);
    app.add_systems(
        Startup,
        (setup_blend_mask, setup_cursor_and_player)
            // The player entity must exist before the pose is pinned.
            .chain()
            .after(spawn_scene),
    );
    app.add_systems(
        Update,
        (
            // The worker takes its WGSL from here; publish before any
            // streaming system can begin a tile.
            publish_erosion_shader_sources,
            handle_global_keys,
            copy_pose_to_clipboard,
            day_night::advance_day_night,
            weather::advance_weather,
            thunder::queue_thunder,
            thunder::play_thunder,
            ui::ui_wants_input_system,
            rivers::stream_rivers,
            buildings::stream_buildings,
            // --measure-overlap replaces normal streaming with its own
            // driver, mirroring the C++ short-circuit in main().
            erosion_stream_system.run_if(not_in_measure_mode),
            player::update_player_system,
            buildings::adapt_eye,
            player::sync_player_camera_transform,
            stream_vegetation,
            measure_overlap_system.run_if(in_measure_mode),
            shot_scheduling_system,
        )
            .chain(),
    );
    app.add_systems(
        EguiPrimaryContextPass,
        (
            ui::draw_diagnostics_ui,
            player::sync_player_camera_transform,
        )
            .chain(),
    );
    app.run();
}
