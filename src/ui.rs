//! 3 toggles passive debug text; 2 opens developer controls.
//!
//! Debug text uses logical egui points and never captures input. The trainer
//! owns all interactive controls, including the advanced terrain/rendering
//! tools and the GPU erosion-flow preview. World overlays use the background
//! layer; their physical-pixel coordinates are converted to logical points.
//! Egui runs after the player update, and camera transforms are synchronized
//! once more after developer edits so teleports reach rendering immediately.

use crate::RerunErosion;
use crate::automation::AutomationSettings;
use crate::constants::*;
use crate::day_night::DayNightCycle;
use crate::erosion::{self, ErosionCache, ErosionTileState};
use crate::matrices;
use crate::noise::NoiseField;
use crate::player::{Player, PlayerCamera, UiWantsInput};
use crate::render::gpu_textures::GpuWorldTexturesOption;
use crate::snow::{self, SnowState};
use crate::water::{WaterOptics, WaterSettings};
use crate::weather::{WeatherMotion, WeatherPreset, WeatherState};
use bevy::math::Vec3;
use bevy::prelude::*;
use bevy::render::RenderApp;
use bevy::render::render_phase::TrackedRenderPass;
use bevy::render::render_resource::{BindGroup, BindGroupLayout, RenderPipeline};
use bevy::render::renderer::RenderDevice;
use bevy::render::sync_world::RenderEntity;
use bevy::window::{CursorOptions, PrimaryWindow};
use bevy_egui::render::{EguiBevyPaintCallback, EguiBevyPaintCallbackImpl, EguiPipelineKey};
use bevy_egui::{EguiContexts, egui};

/// `DrawErosionLattice`'s half-extent of the drawn lattice, in metres.
const LATTICE_EXTENT: f32 = 1600.0;
/// `DrawErosionLattice`'s maximum segment length, in metres.
const LATTICE_SEGMENT_STEP: f32 = 64.0;

/// Keep gameplay input out of the dev menu, including clicks outside it.
/// The text-only debug overlay never captures input.
pub fn ui_wants_input_system(settings: Res<AppSettings>, mut ui_wants_input: ResMut<UiWantsInput>) {
    ui_wants_input.0 = settings.show_trainer;
}

#[derive(Default)]
pub struct TrainerState {
    coordinates: [String; 3],
    initialized: bool,
    snap_to_ground: bool,
    status: Option<String>,
}

/// Draw the developer menu, passive diagnostics, and world overlays.
#[allow(clippy::too_many_arguments)]
pub fn draw_diagnostics_ui(
    mut contexts: EguiContexts,
    mut settings: ResMut<AppSettings>,
    mut erosion_settings: ResMut<ErosionSettings>,
    mut water_settings: ResMut<WaterSettings>,
    mut day_night: ResMut<DayNightCycle>,
    mut weather: ResMut<WeatherState>,
    weather_motion: Res<WeatherMotion>,
    cache: Res<ErosionCache>,
    noise: Res<NoiseField>,
    mut players: Query<&mut Player>,
    automation: Res<AutomationSettings>,
    mut rerun: ResMut<RerunErosion>,
    (vegetation, rivers): (
        Option<Res<crate::vegetation::VegetationField>>,
        Option<Res<crate::rivers::RiverField>>,
    ),
    mut snow: ResMut<SnowState>,
    mut trainer: Local<TrainerState>,
    mut cursors: Query<&mut CursorOptions, With<PrimaryWindow>>,
) -> Result {
    let Ok(mut player) = players.single_mut() else {
        return Ok(());
    };
    let Ok(ctx) = contexts.ctx_mut() else {
        return Ok(());
    };

    let debug_bottom = if settings.show_debug {
        draw_debug_overlay(
            ctx,
            &settings,
            &day_night,
            &weather,
            &cache,
            &noise,
            &snow,
            &player,
            vegetation.as_deref(),
        )
    } else {
        0.0
    };

    if settings.show_trainer {
        draw_trainer_window(
            ctx,
            debug_bottom,
            &mut settings,
            &mut erosion_settings,
            &mut water_settings,
            &mut day_night,
            &mut weather,
            weather_motion.offset,
            &cache,
            &noise,
            &mut snow,
            &mut player,
            &mut trainer,
            &mut rerun,
            rivers.as_deref(),
        );
        if !settings.show_trainer && automation.shot_path.is_none() {
            if let Ok(mut cursor) = cursors.single_mut() {
                player.set_mouse_capture(true, &mut cursor);
            }
        }
    }

    // Screen-space overlays. The C++ issued these straight to the default
    // framebuffer after PresentFxaa, so they are never anti-aliased; painting
    // them on the background layer keeps them under the diagnostics window,
    // matching the C++'s line-then-ImGui order.
    let pixels_per_point = ctx.pixels_per_point();
    let content_rect = ctx.content_rect();
    let viewport = (
        (content_rect.width() * pixels_per_point).round().max(1.0) as u32,
        (content_rect.height() * pixels_per_point).round().max(1.0) as u32,
    );
    let painter = ctx.layer_painter(egui::LayerId::background());
    if settings.erosion_debug {
        draw_erosion_lattice(
            &painter,
            &player,
            &cache,
            &noise,
            viewport,
            pixels_per_point,
        );
    }
    if player.mouse_captured && !settings.show_trainer {
        draw_crosshair(&painter, viewport, pixels_per_point);
    }
    if !player.mouse_captured && !settings.show_trainer && automation.shot_path.is_none() {
        painter.text(
            egui::pos2(20.0, viewport.1 as f32 - 38.0) / pixels_per_point,
            egui::Align2::LEFT_TOP,
            "Click the world to capture the mouse",
            egui::FontId::proportional(20.0 / pixels_per_point),
            egui::Color32::WHITE,
        );
    }
    Ok(())
}

/// Paint passive text directly over the world. Its bottom edge leaves room
/// for the trainer so both can be visible in the top-left corner.
#[allow(clippy::too_many_arguments)]
fn draw_debug_overlay(
    ctx: &egui::Context,
    settings: &AppSettings,
    day_night: &DayNightCycle,
    weather: &WeatherState,
    cache: &ErosionCache,
    noise: &NoiseField,
    snow: &SnowState,
    player: &Player,
    vegetation: Option<&crate::vegetation::VegetationField>,
) -> f32 {
    let dt = ctx.input(|input| input.stable_dt).max(f32::EPSILON);
    let forward = player.forward();
    let xz = Vec2::new(player.position.x, player.position.z);
    let ground = snow::sample_surface(cache, noise, xz, xz.to_array()).height(snow, xz);
    let ready = cache
        .tiles
        .values()
        .filter(|tile| tile.state == ErosionTileState::Ready)
        .count();
    let minutes = (day_night.time_hours * 60.0).round() as u32 % (24 * 60);
    let visibility = weather.visibility_metres(settings, player.position.y);
    let visibility = if visibility.is_finite() && visibility < 40_000.0 {
        format!("{visibility:.0} m")
    } else {
        ">40 km".to_owned()
    };
    let plants = vegetation.filter(|field| field.enabled()).map_or_else(
        || "Vegetation: scatter disabled".to_owned(),
        |field| {
            format!(
                "Vegetation: {} plants, {} chunks, {} generating",
                field.plant_count(),
                field.chunk_count(),
                field.pending_count()
            )
        },
    );
    let text = format!(
        "FPS: {:.0}  |  Frame: {:.2} ms\n\
         Player XYZ: {:.2}, {:.2}, {:.2}\n\
         View vector: {:.3}, {:.3}, {:.3}\n\
         Yaw / pitch: {:.1} / {:.1} deg\n\
         Movement: {}  |  Speed: {:.1}x  |  Vertical: {:.2} m/s\n\
         Ground: {:.2} m  |  Eye above ground: {:.2} m\n\
         Time: {:02}:{:02} {}  |  Weather: {}  |  Visibility: {}\n\
         Erosion tiles: {} ready, {} streaming  |  Clipmap: {:.0} m\n\
         Tile heights: {:.1}..{:.1} m  |  Land: {:.0}%\n\
         Incision / deposition: {:.2} / {:.2} m  |  Detail / axis: {:.3} / {:.3}\n\
         Drainage: {:.0} cells  |  Bare rock: {:.1}%  |  Cover: {:.2} m\n\
         {}",
        1.0 / dt,
        dt * 1000.0,
        player.position.x,
        player.position.y,
        player.position.z,
        forward.x,
        forward.y,
        forward.z,
        player.yaw.to_degrees(),
        player.pitch.to_degrees(),
        if player.flying { "FLYING" } else { "WALKING" },
        player.movement_speed_multiplier,
        player.vertical_velocity,
        ground,
        player.position.y - ground,
        minutes / 60,
        minutes % 60,
        if day_night.paused { "(paused)" } else { "" },
        weather.local_condition().label(),
        visibility,
        ready,
        cache.tiles.len() - ready,
        (CLIP_CELLS as f32 * 0.5) * CLIP_FINEST_SPACING * 2.0f32.powi(CLIP_LEVELS as i32 - 1),
        cache.stats.minimum_height,
        cache.stats.maximum_height,
        cache.stats.land_coverage,
        cache.stats.maximum_incision,
        cache.stats.maximum_deposition,
        cache.stats.erosion_detail,
        cache.stats.flow_axis_bias,
        cache.stats.maximum_drainage,
        cache.stats.bedrock_exposure,
        cache.stats.mean_loose_cover,
        plants,
    );
    let painter = ctx.layer_painter(egui::LayerId::background());
    let position = ctx.content_rect().min + egui::vec2(12.0, 12.0);
    let font = egui::FontId::monospace(13.0);
    // A one-point shadow keeps plain text readable over snow and bright sky.
    painter.text(
        position + egui::vec2(1.0, 1.0),
        egui::Align2::LEFT_TOP,
        &text,
        font.clone(),
        egui::Color32::BLACK,
    );
    painter
        .text(
            position,
            egui::Align2::LEFT_TOP,
            text,
            font,
            egui::Color32::WHITE,
        )
        .bottom()
}

fn parse_teleport_coordinates(coordinates: &[String; 3]) -> std::result::Result<Vec3, String> {
    let mut values = [0.0; 3];
    for (index, axis) in ["X", "Y", "Z"].into_iter().enumerate() {
        values[index] = coordinates[index]
            .trim()
            .parse::<f32>()
            .map_err(|_| format!("Enter a valid {axis} coordinate."))?;
        if !values[index].is_finite() || values[index].abs() > 1_000_000.0 {
            return Err(format!(
                "{axis} must be between -1,000,000 and 1,000,000 m."
            ));
        }
    }
    Ok(Vec3::from_array(values))
}

impl TrainerState {
    fn use_position(&mut self, position: Vec3) {
        self.coordinates = position.to_array().map(|value| format!("{value:.2}"));
        self.initialized = true;
        self.status = None;
    }
}

fn ground_position(
    position: Vec3,
    cache: &ErosionCache,
    noise: &NoiseField,
    snow: &SnowState,
) -> Vec3 {
    let xz = Vec2::new(position.x, position.z);
    let y = snow::sample_surface(cache, noise, xz, xz.to_array()).height(snow, xz) + EYE_HEIGHT;
    Vec3::new(position.x, y, position.z)
}

fn draw_player_tools(
    ui: &mut egui::Ui,
    player: &mut Player,
    trainer: &mut TrainerState,
    cache: &ErosionCache,
    noise: &NoiseField,
    snow: &SnowState,
) {
    if !trainer.initialized {
        trainer.use_position(player.position);
    }
    separator_text(ui, "Player and teleport");
    if ui
        .checkbox(&mut player.flying, "Fly / noclip [V]")
        .changed()
    {
        player.vertical_velocity = 0.0;
    }
    ui.add(
        egui::Slider::new(&mut player.movement_speed_multiplier, 0.1..=20.0)
            .text("Movement speed")
            .suffix("x")
            .logarithmic(true),
    );
    ui.horizontal(|ui| {
        if ui.button("Use current position").clicked() {
            trainer.use_position(player.position);
        }
        if ui.button("Copy XYZ").clicked() {
            ui.ctx().copy_text(format!(
                "{:.2}, {:.2}, {:.2}",
                player.position.x, player.position.y, player.position.z
            ));
        }
    });
    egui::Grid::new("teleport_coordinates")
        .num_columns(2)
        .show(ui, |ui| {
            for (index, axis) in ["X", "Y", "Z"].into_iter().enumerate() {
                ui.label(axis);
                ui.add_enabled(
                    !(index == 1 && trainer.snap_to_ground),
                    egui::TextEdit::singleline(&mut trainer.coordinates[index])
                        .desired_width(200.0),
                );
                ui.end_row();
            }
        });
    ui.checkbox(
        &mut trainer.snap_to_ground,
        "Place on ground at X/Z (ignore Y)",
    );
    ui.small("Exact XYZ teleport enables flight to hold the selected height.");
    ui.horizontal(|ui| {
        if ui.button("Teleport").clicked() {
            let mut coordinates = trainer.coordinates.clone();
            if trainer.snap_to_ground {
                coordinates[1] = "0".to_owned();
            }
            match parse_teleport_coordinates(&coordinates) {
                Ok(position) => {
                    let position = if trainer.snap_to_ground {
                        ground_position(position, cache, noise, snow)
                    } else {
                        position
                    };
                    if player.teleport(position) {
                        player.flying = !trainer.snap_to_ground;
                        trainer.status = Some(format!(
                            "Teleported to {:.2}, {:.2}, {:.2}",
                            position.x, position.y, position.z
                        ));
                    }
                }
                Err(message) => trainer.status = Some(message),
            }
        }
        if ui.button("Return to spawn").clicked() {
            player.teleport(ground_position(Vec3::ZERO, cache, noise, snow));
            player.flying = false;
            player.yaw = Player::default().yaw;
            player.pitch = Player::default().pitch;
            trainer.use_position(player.position);
            trainer.status = Some("Returned to spawn.".to_owned());
        }
    });
    ui.horizontal(|ui| {
        if ui.button("Ground here").clicked() {
            player.teleport(ground_position(player.position, cache, noise, snow));
            player.flying = false;
        }
        if ui.button("Rise 100 m").clicked() {
            player.teleport(player.position + Vec3::Y * 100.0);
            player.flying = true;
        }
        if ui.button("Reset speed").clicked() {
            player.movement_speed_multiplier = 1.0;
        }
    });
    if let Some(status) = &trainer.status {
        ui.small(status);
    }
}

/// One dev window owns all interactive controls; closing it hides everything.
#[allow(clippy::too_many_arguments)]
fn draw_trainer_window(
    ctx: &egui::Context,
    debug_bottom: f32,
    settings: &mut AppSettings,
    erosion_settings: &mut ErosionSettings,
    water_settings: &mut WaterSettings,
    day_night: &mut DayNightCycle,
    weather: &mut WeatherState,
    weather_offset: [f32; 2],
    cache: &ErosionCache,
    noise: &NoiseField,
    snow: &mut SnowState,
    player: &mut Player,
    trainer: &mut TrainerState,
    rerun: &mut RerunErosion,
    rivers: Option<&crate::rivers::RiverField>,
) {
    let mut open = settings.show_trainer;
    let previous_position = player.position;
    let top = if debug_bottom > 0.0 {
        debug_bottom + 12.0
    } else {
        ctx.content_rect().top() + 12.0
    };
    egui::Window::new("Developer menu [2]")
        .id(egui::Id::new("developer_menu"))
        .anchor(
            egui::Align2::LEFT_TOP,
            egui::vec2(12.0, top - ctx.content_rect().top()),
        )
        .default_width(360.0)
        .max_height((ctx.content_rect().bottom() - top - 12.0).max(120.0))
        .resizable(false)
        .vscroll(true)
        .open(&mut open)
        .show(ctx, |ui| {
            draw_player_tools(ui, player, trainer, cache, noise, snow);
            separator_text(ui, "Weather here");
            let selected = weather
                .trainer_preset
                .map(WeatherPreset::label)
                .unwrap_or("Natural weather");
            let mut requested = weather.trainer_preset;
            egui::ComboBox::from_id_salt("trainer_weather")
                .selected_text(selected)
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut requested, None, "Natural weather");
                    for preset in WeatherPreset::ALL {
                        ui.selectable_value(&mut requested, Some(preset), preset.label());
                    }
                });
            if requested != weather.trainer_preset {
                weather.set_trainer_preset(requested);
                weather.refresh_local(player.position.to_array(), weather_offset, settings);
            }
            ui.small(format!(
                "Current conditions: {}",
                weather.local_condition().label()
            ));
            if matches!(
                weather.trainer_preset,
                Some(WeatherPreset::Rain | WeatherPreset::Snow | WeatherPreset::Thunderstorm)
            ) && weather.local_precipitation.intensity() < 0.05
            {
                ui.small("Rain and snow fade above the clouds.");
            }
            ui.checkbox(&mut weather.automatic, "Move weather fronts");
            if weather.local_precipitation.thunderstorm > 0.12
                && ui.button("Strike lightning").clicked()
            {
                weather.request_strike();
            }
            ui.add(
                egui::Slider::new(&mut settings.thunder_volume, 0.0..=1.0)
                    .text("Thunder volume")
                    .fixed_decimals(2),
            );

            separator_text(ui, "Time of day");
            if ui
                .add(
                    egui::Slider::new(&mut day_night.time_hours, 0.0..=23.99).custom_formatter(
                        |hours, _| {
                            let minutes = (hours * 60.0).round() as u32 % (24 * 60);
                            format!("{:02}:{:02}", minutes / 60, minutes % 60)
                        },
                    ),
                )
                .changed()
            {
                day_night.paused = true;
            }
            ui.horizontal(|ui| {
                for (label, hour) in [
                    ("Dawn", 6.25),
                    ("Noon", 12.0),
                    ("Dusk", 17.75),
                    ("Night", 0.0),
                ] {
                    if ui.button(label).clicked() {
                        day_night.time_hours = hour;
                        day_night.paused = true;
                    }
                }
            });
            ui.checkbox(&mut day_night.paused, "Hold selected time");
            ui.collapsing("World and rendering shortcuts", |ui| {
                ui.checkbox(
                    &mut settings.vegetation_enabled,
                    "Trees, shrubs and flowers",
                );
                ui.checkbox(&mut settings.clouds_enabled, "Clouds");
                ui.checkbox(&mut settings.ssao_enabled, "SSAO");
                ui.checkbox(&mut settings.flow_debug, "Flow visualization");
                ui.checkbox(&mut settings.erosion_debug, "Erosion lattice / delta");
                if ui.button("Regenerate erosion cache").clicked() {
                    rerun.0 = true;
                }
            });
            ui.separator();
            ui.checkbox(&mut settings.show_ui, "Advanced controls");
            if settings.show_ui {
                draw_advanced_controls(
                    ui,
                    settings,
                    erosion_settings,
                    water_settings,
                    day_night,
                    weather,
                    player,
                    rerun,
                    rivers,
                );
            }
            ui.small("2 close menu  |  3 debug text  |  F12 screenshot");
        });
    settings.show_trainer = open;
    if player.position != previous_position {
        snow.recenter(Vec2::new(player.position.x, player.position.z));
        weather.refresh_local(player.position.to_array(), weather_offset, settings);
    }
}

/// Detailed controls live inside the trainer rather than a second window.
#[allow(clippy::too_many_arguments)]
fn draw_advanced_controls(
    ui: &mut egui::Ui,
    settings: &mut AppSettings,
    erosion_settings: &mut ErosionSettings,
    water_settings: &mut WaterSettings,
    day_night: &mut DayNightCycle,
    weather: &mut WeatherState,
    player: &Player,
    rerun: &mut RerunErosion,
    rivers: Option<&crate::rivers::RiverField>,
) {
    separator_text(ui, "Weather");
    ui.checkbox(&mut weather.automatic, "Moving weather fronts");
    egui::ComboBox::from_label("Weather trend")
        .selected_text(weather.target.label())
        .show_ui(ui, |ui| {
            for preset in WeatherPreset::ALL {
                if ui
                    .selectable_label(weather.target == preset, preset.label())
                    .clicked()
                {
                    weather.set_target(preset);
                }
            }
        });
    if weather.is_transitioning() {
        ui.label(format!("Changing to {}", weather.target.label()));
    }
    ui.small("Conditions vary by location; fronts move with the wind.");
    let visibility = weather.visibility_metres(settings, player.position.y);
    let visibility_label = if !visibility.is_finite() || visibility >= 40_000.0 {
        ">40 km".to_owned()
    } else if visibility >= 1000.0 {
        format!("~{:.1} km", visibility / 1000.0)
    } else {
        format!("~{:.0} m", visibility)
    };
    ui.label(format!(
        "Here: {} · visibility {}",
        weather.local_condition().label(),
        visibility_label,
    ));
    let local_precipitation = weather.conditions(settings);
    if local_precipitation.rain_intensity + local_precipitation.snow_intensity > 0.02 {
        ui.small(format!(
            "Rain {:.0}% · snow {:.0}% · thunder {:.0}% · gust {:.0}%",
            local_precipitation.rain_intensity * 100.0,
            local_precipitation.snow_intensity * 100.0,
            local_precipitation.thunder_intensity * 100.0,
            local_precipitation.gust_strength * 100.0,
        ));
    }
    ui.add(
        egui::Slider::new(&mut weather.transition_seconds, 10.0..=180.0)
            .text("Trend transition")
            .suffix(" s")
            .fixed_decimals(0),
    );
    ui.add(
        egui::Slider::new(&mut settings.thunder_volume, 0.0..=1.0)
            .text("Thunder volume")
            .fixed_decimals(2),
    );

    separator_text(ui, "Sun and time");
    ui.add(
        egui::Slider::new(&mut day_night.time_hours, 0.0..=23.99)
            .text("Time of day")
            .custom_formatter(|hours, _| {
                let minutes = (hours * 60.0).round() as u32 % (24 * 60);
                format!("{:02}:{:02}", minutes / 60, minutes % 60)
            }),
    );
    ui.checkbox(&mut day_night.paused, "Pause day/night cycle");
    ui.add(
        egui::Slider::new(&mut day_night.day_length_minutes, 1.0..=120.0)
            .text("Day duration")
            .suffix(" min")
            .logarithmic(true),
    );
    ui.horizontal(|ui| {
        for (label, hour) in [
            ("Dawn", 6.25),
            ("Noon", 12.0),
            ("Dusk", 17.75),
            ("Night", 0.0),
        ] {
            if ui.button(label).clicked() {
                day_night.time_hours = hour;
            }
        }
    });

    separator_text(ui, "Raymarched lighting");
    ui.checkbox(&mut settings.raymarched_shadows, "Terrain shadows");
    ui.checkbox(&mut settings.volumetric_lighting, "Volumetric sunlight");
    ui.checkbox(&mut settings.water_reflections, "Water reflections");
    let quality_name = match settings.raymarch_quality {
        0 => "Low",
        2 => "High",
        _ => "Balanced",
    };
    egui::ComboBox::from_label("Raymarch quality")
        .selected_text(quality_name)
        .show_ui(ui, |ui| {
            ui.selectable_value(&mut settings.raymarch_quality, 0, "Low");
            ui.selectable_value(&mut settings.raymarch_quality, 1, "Balanced");
            ui.selectable_value(&mut settings.raymarch_quality, 2, "High");
        });
    ui.add_enabled(
        settings.volumetric_lighting,
        egui::Slider::new(&mut settings.volumetric_strength, 0.0..=2.0)
            .text("Light shaft strength")
            .fixed_decimals(2),
    );

    separator_text(ui, "Volumetric clouds");
    ui.small(
        "Cloud controls set the cloudy baseline; weather adjusts coverage, height, and density.",
    );
    ui.checkbox(&mut settings.clouds_enabled, "Clouds");
    ui.add_enabled_ui(settings.clouds_enabled, |ui| {
        ui.add(
            egui::Slider::new(&mut settings.cloud_coverage, 0.0..=1.0)
                .text("Coverage")
                .fixed_decimals(2),
        );
        ui.add(
            egui::Slider::new(&mut settings.cloud_density, 0.0..=4.0)
                .text("Density")
                .fixed_decimals(2),
        );
        ui.add(
            egui::Slider::new(&mut settings.cloud_base_height, 100.0..=6000.0)
                .text("Cloud base")
                .suffix(" m")
                .fixed_decimals(0),
        );
        ui.add(
            egui::Slider::new(&mut settings.cloud_thickness, 100.0..=4000.0)
                .text("Layer thickness")
                .suffix(" m")
                .fixed_decimals(0),
        );
        ui.add(
            egui::Slider::new(&mut settings.cloud_wind_speed, 0.0..=80.0)
                .text("Cloud wind speed")
                .suffix(" m/s")
                .fixed_decimals(1),
        );
        ui.add(
            egui::Slider::new(&mut settings.cloud_wind_direction_degrees, 0.0..=360.0)
                .text("Cloud wind direction")
                .suffix("°")
                .fixed_decimals(0),
        );
        ui.add(
            egui::Slider::new(&mut settings.cloud_shadow_strength, 0.0..=1.0)
                .text("Cloud shadows")
                .fixed_decimals(2),
        );
        ui.collapsing("Cloud shape", |ui| {
            ui.add(
                egui::Slider::new(&mut settings.cloud_scale, 300.0..=6000.0)
                    .text("Formation scale")
                    .suffix(" m")
                    .fixed_decimals(0),
            );
            ui.add(
                egui::Slider::new(&mut settings.cloud_detail_strength, 0.0..=1.0)
                    .text("Edge detail")
                    .fixed_decimals(2),
            );
        });
    });

    separator_text(ui, "Rendering");
    ui.checkbox(&mut settings.ssao_enabled, "SSAO");
    ui.checkbox(&mut settings.flow_debug, "Flow visualization");
    ui.checkbox(&mut settings.erosion_debug, "Erosion delta debug");
    ui.add(egui::Slider::new(&mut settings.ao_radius, 0.2..=6.0).text("AO radius"));
    ui.add(egui::Slider::new(&mut settings.ao_bias, 0.005..=0.3).text("AO bias"));
    ui.add(egui::Slider::new(&mut settings.ao_power, 0.4..=3.0).text("AO power"));
    ui.add(egui::Slider::new(&mut settings.ao_strength, 0.0..=1.0).text("AO strength"));
    ui.add(egui::Slider::new(&mut settings.ao_tex_strength, 0.0..=1.0).text("Texture AO"));
    ui.add(
        egui::Slider::new(&mut settings.fog_density, 0.0..=0.001)
            .text("Fog")
            .fixed_decimals(5),
    );
    ui.add(
        egui::Slider::new(&mut settings.sun_intensity, 0.0..=8.0)
            .text("Sun intensity")
            .fixed_decimals(2),
    );
    ui.add(
        egui::Slider::new(&mut settings.exposure, 0.25..=4.0)
            .text("Exposure")
            .fixed_decimals(2),
    );
    ui.add(
        egui::Slider::new(&mut settings.sparkle_strength, 0.0..=2.0)
            .text("Snow sparkle")
            .fixed_decimals(2),
    );

    separator_text(ui, "Vegetation");
    ui.checkbox(
        &mut settings.vegetation_enabled,
        "Trees, shrubs and flowers",
    );
    ui.checkbox(&mut settings.vegetation_shadows, "Plant shadows");
    ui.add(
        egui::Slider::new(&mut settings.vegetation_detail, 0.4..=2.5)
            .text("Plant detail distance")
            .fixed_decimals(2),
    );

    separator_text(ui, "Rivers");
    ui.checkbox(&mut settings.rivers_visible, "River and lake surfaces");
    match rivers.and_then(|field| field.network()) {
        Some(network) => {
            ui.label(format!(
                "{} rivers, {:.1} km of channel, {} lakes (region built in {:.2} s)",
                network.rivers.len(),
                network.total_length_km(),
                network.lakes.len(),
                network.build_seconds
            ));
            let here = network.envelope(player.position.x, player.position.z);
            if here.is_none() {
                ui.label("No river within reach of the player");
            } else {
                ui.label(format!(
                    "Nearest channel: {:.1} m wide, water at {:.1} m, {:.1} m past its bank",
                    here.half_width * 2.0,
                    here.water,
                    here.bank_distance
                ));
            }
            if player.wading_depth > 0.0 {
                ui.label(format!(
                    "Wading {:.2} m deep in a {:.2} m/s current{}",
                    player.wading_depth,
                    player.current.length(),
                    if player.swept { ", swept off your feet" } else { "" }
                ));
            }
        }
        None => {
            ui.label("Rivers disabled (--no-rivers)");
        }
    }

    separator_text(ui, "Terrain textures");
    ui.add(
        egui::Slider::new(&mut settings.texture_scale, 0.005..=0.6)
            .text("Texture scale")
            .fixed_decimals(3),
    );
    ui.add(
        egui::Slider::new(&mut settings.normal_strength, 0.0..=2.0)
            .text("Normal strength")
            .fixed_decimals(2),
    );
    ui.add(
        egui::Slider::new(&mut settings.variant_scale, 0.001..=0.05)
            .text("Dirt/gravel variant scale")
            .fixed_decimals(3),
    );

    separator_text(ui, "Water");
    ui.checkbox(&mut water_settings.enabled, "Water surface");
    // The wave block is only rebuilt when one of these actually
    // changes, so graying them out while the surface is off keeps the
    // spectrum from being regenerated for a pass that will not run.
    ui.add_enabled_ui(water_settings.enabled, |ui| {
        ui.add(
            egui::Slider::new(&mut water_settings.sea_state_amplitude, 0.0..=1.5)
                .text("Sea state")
                .fixed_decimals(2),
        );
        ui.add(
            egui::Slider::new(&mut water_settings.wind_direction_degrees, 0.0..=360.0)
                .text("Wind direction")
                .fixed_decimals(0)
                .suffix("°"),
        );
        // A combo box rather than aqua's cycle-on-click button: the
        // presets are compared by value, so the selection survives
        // edits to the fields they do not carry.
        let selected = WaterOptics::PRESETS
            .iter()
            .position(|(_, preset)| *preset == water_settings.optics)
            .unwrap_or(0);
        let mut index = selected;
        egui::ComboBox::from_label("Optics")
            .selected_text(WaterOptics::PRESETS[selected].0)
            .show_ui(ui, |ui| {
                for (slot, (name, _)) in WaterOptics::PRESETS.iter().enumerate() {
                    ui.selectable_value(&mut index, slot, *name);
                }
            });
        if index != selected {
            water_settings.optics = WaterOptics::PRESETS[index].1;
        }
        ui.checkbox(&mut water_settings.underwater_effects, "Underwater effects");
        ui.checkbox(&mut water_settings.flat_surface, "Flat surface (debug)");
    });

    separator_text(ui, "Erosion");
    ui.add(egui::Slider::new(&mut erosion_settings.iterations, 20..=400).text("Iterations"));
    ui.add(egui::Slider::new(&mut erosion_settings.rain, 0.0..=0.08).text("Rain"));
    ui.add(egui::Slider::new(&mut erosion_settings.evaporation, 0.0..=0.4).text("Evaporation"));
    ui.add(egui::Slider::new(&mut erosion_settings.erosion_rate, 0.01..=1.2).text("Erosion rate"));
    ui.add(egui::Slider::new(&mut erosion_settings.deposition_rate, 0.01..=1.2).text("Deposition"));
    ui.add(egui::Slider::new(&mut erosion_settings.sediment_capacity, 0.5..=20.0).text("Capacity"));
    ui.add(egui::Slider::new(&mut erosion_settings.transport_rate, 0.05..=2.0).text("Transport"));
    ui.add(
        egui::Slider::new(&mut erosion_settings.maximum_erosion, 1.0..=40.0)
            .text("Max excavation")
            .fixed_decimals(1)
            .suffix(" m"),
    );
    ui.add(
        egui::Slider::new(&mut erosion_settings.fluvial_capacity, 0.0..=0.2).text("Stream power"),
    );
    ui.add(
        egui::Slider::new(&mut erosion_settings.fluvial_erosion, 0.0..=0.5)
            .text("Channel incision"),
    );
    ui.add(
        egui::Slider::new(&mut erosion_settings.fluvial_deposition, 0.0..=1.0)
            .text("Alluvial deposition"),
    );
    ui.add(
        egui::Slider::new(&mut erosion_settings.maximum_incision, 0.0..=30.0)
            .text("Max incision")
            .fixed_decimals(1)
            .suffix(" m"),
    );
    ui.add(egui::Slider::new(&mut erosion_settings.talus_rate, 0.0..=0.0625).text("Talus slide"));
    ui.add(egui::Slider::new(&mut erosion_settings.rockfall_rate, 0.0..=0.0625).text("Rockfall"));
    ui.horizontal(|ui| {
        if ui.button("Regenerate erosion cache").clicked() {
            rerun.0 = true;
        }
        ui.weak("(streams incrementally)");
    });
    ui.collapsing("Flow output", |ui| {
        ui.label("RGBA: water, velocity X, velocity Z, drainage area");
        let (response, painter) =
            ui.allocate_painter(egui::vec2(320.0, 320.0), egui::Sense::hover());
        painter.add(EguiBevyPaintCallback::new_paint_callback(
            response.rect,
            FlowAtlasPaintCallback,
        ));
    });

    separator_text(ui, "Controls");
    ui.label("WASD move  |  mouse look");
    ui.label("Space jump/up  |  Shift down");
    ui.label("V flight  |  Ctrl boost");
    ui.label("2 dev menu  |  3 debug text  |  Esc release cursor");
    // The C++ printed the noise texture's own dimensions, which are
    // NOISE_RESOLUTION square by construction.
    ui.weak(format!(
        "FastNoiseLite source texture: {NOISE_RESOLUTION}x{NOISE_RESOLUTION}"
    ));
}

/// `DrawErosionLattice`: the 512 m erosion-tile lattice drawn in screen space
/// so a visible cliff can be located relative to tile boundaries. Each line is
/// broken into short terrain-anchored segments: a single chord between two
/// endpoint heights would cut straight through hills, and a segment whose
/// endpoint passes behind the camera would project mirrored into the sky.
fn draw_erosion_lattice(
    painter: &egui::Painter,
    player: &Player,
    cache: &ErosionCache,
    noise: &NoiseField,
    viewport: (u32, u32),
    pixels_per_point: f32,
) {
    // The same matrices the render side builds for `GlobalUniforms`, so the
    // overlay lines up with the rasterized terrain.
    let camera = PlayerCamera::from_player(player);
    let aspect = viewport.0 as f32 / viewport.1 as f32;
    let view = matrices::view_matrix(camera.position, camera.target, camera.up);
    let projection = matrices::perspective(camera.fov_y, aspect, NEAR_PLANE, FAR_PLANE);

    let visibility_center = [player.position.x, player.position.z];
    let base_x = (player.position.x / EROSION_TILE_STRIDE).round() * EROSION_TILE_STRIDE;
    let base_z = (player.position.z / EROSION_TILE_STRIDE).round() * EROSION_TILE_STRIDE;
    let stroke = egui::Stroke::new(
        1.0 / pixels_per_point,
        egui::Color32::from_rgba_unmultiplied(255, 70, 70, 230),
    );
    let position = camera.position;
    // Horizontal forward, exactly as the C++ derived it.
    let forward = Vec3::new(
        camera.target.x - position.x,
        0.0,
        camera.target.z - position.z,
    )
    .normalize_or_zero();
    let point_in_front =
        |x: f32, z: f32| (x - position.x) * forward.x + (z - position.z) * forward.z > 1.0;
    // The C++ guarded only with pointInFront; world_to_screen's own
    // behind-the-near-plane rejection is the same predicate for a perspective
    // matrix, so the two agree on every point that reaches the painter.
    let project = |x: f32, y: f32, z: f32| -> Option<egui::Pos2> {
        let screen = matrices::world_to_screen(Vec3::new(x, y, z), &view, &projection, viewport)?;
        Some(egui::pos2(screen.x, screen.y) / pixels_per_point)
    };
    let line_at = |x0: f32, z0: f32, x1: f32, z1: f32| {
        let dx = x1 - x0;
        let dz = z1 - z0;
        let length = (dx * dx + dz * dz).sqrt().max(1.0);
        let segments = ((length / LATTICE_SEGMENT_STEP) as i32).max(1);
        let mut previous_x = x0;
        let mut previous_z = z0;
        let mut previous_visible = point_in_front(previous_x, previous_z);
        for step in 1..=segments {
            let t = step as f32 / segments as f32;
            let x = x0 + dx * t;
            let z = z0 + dz * t;
            let visible = point_in_front(x, z);
            if previous_visible && visible {
                let h0 = erosion::sample_eroded_height(
                    cache,
                    noise,
                    previous_x,
                    previous_z,
                    visibility_center,
                ) + 1.5;
                let h1 = erosion::sample_eroded_height(cache, noise, x, z, visibility_center) + 1.5;
                if let (Some(start), Some(end)) =
                    (project(previous_x, h0, previous_z), project(x, h1, z))
                {
                    painter.line_segment([start, end], stroke);
                }
            }
            previous_x = x;
            previous_z = z;
            previous_visible = visible;
        }
    };
    for n in -2..=2 {
        let x = base_x + n as f32 * EROSION_TILE_STRIDE;
        line_at(x, base_z - LATTICE_EXTENT, x, base_z + LATTICE_EXTENT);
        let z = base_z + n as f32 * EROSION_TILE_STRIDE;
        line_at(base_x - LATTICE_EXTENT, z, base_x + LATTICE_EXTENT, z);
    }
}

/// `DrawCrosshair`: two 6 px arms through the screen centre at 0.85 alpha.
fn draw_crosshair(painter: &egui::Painter, viewport: (u32, u32), pixels_per_point: f32) {
    // Integer halves, as the C++'s `GetScreenWidth() / 2` computed them.
    let centre = egui::pos2(
        (viewport.0 / 2) as f32 / pixels_per_point,
        (viewport.1 / 2) as f32 / pixels_per_point,
    );
    // The 6 is raylib *screen* (logical) points, not framebuffer pixels:
    // DrawLine rasterised through rlOrtho(0, GetScreenWidth(), ...), and
    // FLAG_WINDOW_HIGHDPI's screenScale then doubles the framebuffer under it.
    // Measured, that is what makes the reference's arm 12 framebuffer rows on
    // a 1x window and 24 on this 2x panel (/tmp/ab_cpp_1.png at 1470x900 vs
    // /tmp/cpp_shot.png at 2940x1782). egui already paints in logical points,
    // so the constant carries across unchanged — dividing it by
    // pixels_per_point, as this did, halved the crosshair on Retina (12 rows
    // instead of 24 at 2940x1782).
    let arm = 6.0;
    // The hairline is the one part that does NOT scale: raylib rasterised it
    // at one framebuffer pixel wide whatever the screenScale, and the
    // reference measures 1 column thick at both factors, so this stays a
    // single physical pixel expressed in logical points.
    let stroke = egui::Stroke::new(
        1.0 / pixels_per_point,
        egui::Color32::from_rgba_unmultiplied(255, 255, 255, 216),
    );
    painter.line_segment(
        [centre - egui::vec2(arm, 0.0), centre + egui::vec2(arm, 0.0)],
        stroke,
    );
    painter.line_segment(
        [centre - egui::vec2(0.0, arm), centre + egui::vec2(0.0, arm)],
        stroke,
    );
}

/// `ImGui::SeparatorText`: the label sits at the left with a rule filling the
/// rest of the row. egui's `Separator` carries no text, so both parts are
/// painted into one allocated row.
fn separator_text(ui: &mut egui::Ui, text: &str) {
    let spacing = ui.spacing().item_spacing.y;
    ui.add_space(spacing);
    let row_height = ui.text_style_height(&egui::TextStyle::Body);
    let (rect, _response) = ui.allocate_exact_size(
        egui::vec2(ui.available_width(), row_height),
        egui::Sense::hover(),
    );
    let text_colour = ui.visuals().text_color();
    let rule = ui.visuals().widgets.noninteractive.bg_stroke;
    let painter = ui.painter();
    let galley = painter.layout_no_wrap(
        text.to_owned(),
        egui::FontId::proportional(row_height),
        text_colour,
    );
    let rule_start = rect.left() + galley.size().x + 6.0;
    painter.galley(rect.min, galley, text_colour);
    if rule_start < rect.right() {
        painter.hline(rule_start..=rect.right(), rect.center().y, rule);
    }
    ui.add_space(spacing);
}

// ---------------------------------------------------------------------------
// Flow-atlas preview
//
// The C++ bound `erosion.flowAtlas` directly with `ImGui::Image(tex, {320,
// 320}, {0,1}, {1,0})`. bevy_egui's image path only takes bevy `Image` assets
// (it copies CPU pixel data into its own atlas) and the flow atlas exists
// solely on the GPU, so the port draws it with a raw wgpu pass issued from an
// egui paint callback; see assets/shaders/flow-preview.wgsl for the
// orientation and colour-space notes.
// ---------------------------------------------------------------------------

/// Render-world state the paint callback fills in on the first frame it runs
/// and reads on every draw. It lives in the render world (not behind a lock
/// in the callback) so `render` can borrow the pipeline straight out of the
/// world for the whole render pass.
#[derive(Resource, Default)]
pub struct FlowPreviewRenderState {
    /// The pipeline and the hdr flag it was built for.
    pipeline: Option<(EguiPipelineKey, RenderPipeline)>,
    bind_group: Option<BindGroup>,
}

/// Installs the flow-preview state above in the render world.
pub struct UiRenderPlugin;

impl Plugin for UiRenderPlugin {
    fn build(&self, app: &mut App) {
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.init_resource::<FlowPreviewRenderState>();
        }
    }
}

/// See the section comment above. The egui pass has already set the viewport
/// to the widget rect and clipped to it, so a fullscreen triangle covers
/// exactly the 320x320 image; it resets its own pipeline and viewport after
/// the callback returns.
struct FlowAtlasPaintCallback;

impl EguiBevyPaintCallbackImpl for FlowAtlasPaintCallback {
    fn update(
        &self,
        _info: egui::PaintCallbackInfo,
        _render_entity: RenderEntity,
        key: EguiPipelineKey,
        world: &mut World,
    ) {
        let built_for = world
            .resource::<FlowPreviewRenderState>()
            .pipeline
            .as_ref()
            .map(|(key, _)| *key);
        if built_for == Some(key) {
            return;
        }
        let Some((pipeline, bind_group)) = build_flow_preview_pipeline(world, key) else {
            return;
        };
        let mut state = world.resource_mut::<FlowPreviewRenderState>();
        state.pipeline = Some((key, pipeline));
        state.bind_group = Some(bind_group);
    }

    fn render<'pass>(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut TrackedRenderPass<'pass>,
        _render_entity: RenderEntity,
        _key: EguiPipelineKey,
        world: &'pass World,
    ) {
        let Some(state) = world.get_resource::<FlowPreviewRenderState>() else {
            return;
        };
        let (Some((_, pipeline)), Some(bind_group)) = (&state.pipeline, &state.bind_group) else {
            return;
        };
        render_pass.set_render_pipeline(pipeline);
        render_pass.set_bind_group(0, bind_group, &[]);
        render_pass.draw(0..3, 0..1);
    }
}

/// Builds the preview pipeline and its bind group (the flow atlas view plus a
/// linear, repeating sampler — raylib's default texture filter and wrap).
/// `layout: None` lets wgpu infer the bind group layout from the shader.
fn build_flow_preview_pipeline(
    world: &mut World,
    key: EguiPipelineKey,
) -> Option<(RenderPipeline, BindGroup)> {
    let device = world.resource::<RenderDevice>();
    let textures = world.resource::<GpuWorldTexturesOption>().0.as_ref()?;
    let shader = device
        .wgpu_device()
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("flow_preview_shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "../assets/shaders/flow-preview.wgsl"
            ))),
        });
    let target = key.target_format;
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("flow_preview_pipeline"),
        layout: None,
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: target,
                // ImGui's renderer blends with separate alpha factors
                // (GL_SRC_ALPHA/GL_ONE_MINUS_SRC_ALPHA for colour,
                // GL_ONE/GL_ONE_MINUS_SRC_ALPHA for alpha).
                blend: Some(wgpu::BlendState {
                    color: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::SrcAlpha,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    },
                    alpha: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    },
                }),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });
    let layout = BindGroupLayout::from(pipeline.get_bind_group_layout(0));
    let sampler = device
        .wgpu_device()
        .create_sampler(&wgpu::SamplerDescriptor {
            label: Some("flow_preview_sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..default()
        });
    let bind_group = device.create_bind_group(
        Some("flow_preview_bind_group"),
        &layout,
        &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&textures.flow_atlas_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
    );
    Some((pipeline, bind_group))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_overlay_is_passive_text_at_the_top_left() {
        let ctx = egui::Context::default();
        ctx.begin_pass(egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(960.0, 640.0),
            )),
            ..Default::default()
        });
        let noise = NoiseField {
            samples: vec![0.5; NOISE_RESOLUTION * NOISE_RESOLUTION],
        };
        let bottom = draw_debug_overlay(
            &ctx,
            &AppSettings::default(),
            &DayNightCycle::default(),
            &WeatherState::default(),
            &ErosionCache::default(),
            &noise,
            &SnowState::default(),
            &Player::default(),
            None,
        );
        let mut output = ctx.end_pass();
        output.textures_delta.clear(); // Headless checks do not upload font textures.
        assert!(bottom > 12.0 && bottom < 640.0);
        assert!(!ctx.egui_wants_pointer_input());
        assert!(!ctx.egui_wants_keyboard_input());
        assert_eq!(output.shapes.len(), 2); // Text plus its readability shadow.
        for shape in &output.shapes {
            let egui::Shape::Text(text) = &shape.shape else {
                panic!("The debug overlay must contain only text, without a window or frame.");
            };
            assert!(text.galley.job.text.contains("FPS:"));
            assert!(text.galley.job.text.contains("Player XYZ:"));
            assert!(text.galley.job.text.contains("View vector:"));
        }
        let egui::Shape::Text(text) = &output.shapes[1].shape else {
            unreachable!()
        };
        assert_eq!(text.pos, egui::pos2(12.0, 12.0));
    }

    #[test]
    fn trainer_stays_on_screen_below_debug_text_when_toggled() {
        let ctx = egui::Context::default();
        let mut settings = AppSettings {
            show_trainer: true,
            ..default()
        };
        let mut erosion_settings = ErosionSettings::default();
        let mut water_settings = WaterSettings::default();
        let mut day_night = DayNightCycle::default();
        let mut weather = WeatherState::default();
        let cache = ErosionCache::default();
        let noise = NoiseField {
            samples: vec![0.5; NOISE_RESOLUTION * NOISE_RESOLUTION],
        };
        let mut snow = SnowState::default();
        let mut player = Player::default();
        let mut trainer = TrainerState::default();
        let mut rerun = RerunErosion::default();
        for debug_visible in [false, true, false] {
            // Let egui settle its content size after each visibility change.
            for _ in 0..2 {
                ctx.begin_pass(egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(960.0, 640.0),
                    )),
                    ..Default::default()
                });
                let bottom = if debug_visible {
                    draw_debug_overlay(
                        &ctx, &settings, &day_night, &weather, &cache, &noise, &snow, &player, None,
                    )
                } else {
                    0.0
                };
                draw_trainer_window(
                    &ctx,
                    bottom,
                    &mut settings,
                    &mut erosion_settings,
                    &mut water_settings,
                    &mut day_night,
                    &mut weather,
                    [0.0; 2],
                    &cache,
                    &noise,
                    &mut snow,
                    &mut player,
                    &mut trainer,
                    &mut rerun,
                    None,
                );
                let mut output = ctx.end_pass();
                output.textures_delta.clear();
                let rect = ctx
                    .memory(|memory| memory.area_rect(egui::Id::new("developer_menu")))
                    .unwrap();
                assert!((rect.left() - 12.0).abs() <= 1.0, "{rect:?}");
                assert!(
                    rect.top() >= bottom + 11.0,
                    "{rect:?}, debug bottom {bottom}"
                );
                assert!(rect.bottom() <= 640.0, "{rect:?}");
            }
        }
    }

    #[test]
    fn teleport_accepts_signed_decimal_coordinates_with_whitespace() {
        let coordinates = [" -1234.5 ", "100.25", "+987.75"].map(str::to_owned);
        assert_eq!(
            parse_teleport_coordinates(&coordinates),
            Ok(Vec3::new(-1234.5, 100.25, 987.75))
        );
    }

    #[test]
    fn teleport_rejects_invalid_and_unsafe_coordinates_on_every_axis() {
        for axis in 0..3 {
            for invalid in [
                "", "oops", "NaN", "inf", "-inf", "1e30", "1000001", "-1000001",
            ] {
                let mut coordinates = ["0", "0", "0"].map(str::to_owned);
                coordinates[axis] = invalid.to_owned();
                let message = parse_teleport_coordinates(&coordinates).unwrap_err();
                assert!(message.contains(["X", "Y", "Z"][axis]));
            }
        }
    }
}
