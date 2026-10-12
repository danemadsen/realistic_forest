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
    /// The last sky jump's outcome.
    sky_status: Option<String>,
    /// Sunrise and sunset, seconds from the epoch, for the local midnight
    /// and the place (the eye's direction) they were worked out for.
    sun_times: Option<(f64, bevy::math::DVec3, Option<f64>, Option<f64>)>,
}

/// The streamed fields the diagnostics read when they exist, and the planet.
type WorldFields<'w> = (
    Option<Res<'w, crate::vegetation::VegetationField>>,
    Option<Res<'w, crate::rivers::RiverField>>,
    Option<Res<'w, crate::buildings::BuildingField>>,
    Option<Res<'w, crate::planet::view::PlanetTerrain>>,
    Res<'w, crate::planet::Planet>,
    Res<'w, crate::planet::view::PlanetView>,
);

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
    (vegetation, rivers, buildings, planet_terrain, planet, planet_view): WorldFields,
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
        let sky = sky_readout(&day_night, &planet, planet_view.eye);
        draw_debug_overlay(
            ctx,
            &settings,
            &day_night,
            &sky,
            &weather,
            &cache,
            &noise,
            &snow,
            &player,
            vegetation.as_deref(),
            buildings.as_deref(),
            planet_terrain.as_deref(),
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
            buildings.as_deref(),
            &planet,
            planet_view.eye,
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
            &planet,
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
    sky: &str,
    weather: &WeatherState,
    cache: &ErosionCache,
    noise: &NoiseField,
    snow: &SnowState,
    player: &Player,
    vegetation: Option<&crate::vegetation::VegetationField>,
    buildings: Option<&crate::buildings::BuildingField>,
    planet: Option<&crate::planet::view::PlanetTerrain>,
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
    let houses = buildings.filter(|field| field.enabled()).map_or_else(
        || "Buildings: none placed".to_owned(),
        |field| match field.nearest(xz, |placement| crate::planet::height::features_whole(cache, placement.origin.x, placement.origin.z)) {
            Some(nearest) => format!(
                "Buildings: {} placed  |  Nearest: {} at {:.0} m",
                field.building_count(),
                nearest.name,
                nearest.distance
            ),
            None => format!("Buildings: {} placed", field.building_count()),
        },
    );
    let plants = format!("{plants}\n{houses}");
    let text = format!(
        "FPS: {:.0}  |  Frame: {:.2} ms\n\
         Player XYZ: {:.2}, {:.2}, {:.2}\n\
         View vector: {:.3}, {:.3}, {:.3}\n\
         Yaw / pitch: {:.1} / {:.1} deg\n\
         Movement: {}  |  Speed: {:.1}x  |  Vertical: {:.2} m/s\n\
         Ground: {:.2} m  |  Eye above ground: {:.2} m\n\
         Time: {:02}:{:02} {}  |  Weather: {}  |  Visibility: {}\n\
         {}\n\
         Erosion tiles: {} ready, {} streaming  |  Planet: {:?} face, {} terrain patches\n\
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
        sky,
        ready,
        cache.tiles.len() - ready,
        player.face,
        planet.map_or(0, |terrain| terrain.stats.drawn),
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

/// The time warps the developer menu offers; at the default day's length the
/// fastest runs a day in under a second and a half (the clock is held to a
/// day a second whatever the warp, `day_night::MAXIMUM_DAYS_PER_SECOND`).
const TIME_WARPS: [f32; 4] = [1.0, 10.0, 100.0, 1_000.0];

fn clock_label(hours: f64) -> String {
    let minutes = (hours * 60.0).round() as i64 % (24 * 60);
    format!("{:02}:{:02}", minutes / 60, minutes % 60)
}

/// The sky for the readouts: the day, the season and the year, the moon's
/// phase, and any eclipse the eye at `eye` sees.
fn sky_readout(day_night: &DayNightCycle, planet: &crate::planet::Planet, eye: bevy::math::DVec3) -> String {
    use crate::celestial::{self, Orbits, SOLAR_DAY};
    let orbits = Orbits::new(planet);
    let seconds = day_night.seconds();
    let sky = orbits.at(seconds);
    let view = sky.seen_from(eye);
    let up = eye.normalize();
    let year_day = seconds.rem_euclid(orbits.year()) / SOLAR_DAY;
    let mut text = format!(
        "Day {}  |  {}, day {:.0} of the {:.0}-day year  |  Moon: {}, {:.0}% lit, round in {:.1} days, phases in {:.1}",
        day_night.day,
        orbits.season(&sky, eye),
        year_day.floor(),
        orbits.year() / SOLAR_DAY,
        celestial::phase_name(&sky),
        view.moon_lit * 100.0,
        orbits.month() / SOLAR_DAY,
        orbits.synodic_month() / SOLAR_DAY,
    );
    if view.solar_eclipse > 0.001 && view.to_sun.dot(up) > 0.0 {
        let separation = view.to_sun.dot(view.to_moon).clamp(-1.0, 1.0).acos();
        let kind = if view.solar_eclipse >= 0.999 {
            "Total"
        } else if view.moon_radius < view.sun_radius && separation <= view.sun_radius - view.moon_radius {
            "Annular"
        } else {
            "Partial"
        };
        text += &format!("\n{kind} solar eclipse: {:.0}% of the sun hidden", view.solar_eclipse * 100.0);
    } else if let Some(kind) = sky.lunar_eclipse().filter(|_| view.moon_sunlight < 0.98 && view.to_moon.dot(up) > 0.0) {
        text += &format!(
            "\n{kind:?} lunar eclipse: {:.0}% of the moon's sunlight shadowed",
            (1.0 - view.moon_sunlight) * 100.0
        );
    }
    text
}

/// The clock, the day, the time warp and the sky's readouts.
fn draw_sky_controls(
    ui: &mut egui::Ui,
    day_night: &mut DayNightCycle,
    trainer: &mut TrainerState,
    planet: &crate::planet::Planet,
    eye: bevy::math::DVec3,
) {
    use crate::celestial::{Orbits, SOLAR_DAY};
    separator_text(ui, "Sun, moon and time");
    // Edits only: clamping every frame would hold the running clock under
    // the slider's end and keep it from reaching midnight.
    if ui
        .add(
            egui::Slider::new(&mut day_night.time_hours, 0.0..=23.99)
                .clamping(egui::SliderClamping::Edits)
                .custom_formatter(|hours, _| clock_label(hours)),
        )
        .changed()
    {
        day_night.paused = true;
    }
    ui.horizontal(|ui| {
        ui.label("Day");
        if ui.add(egui::DragValue::new(&mut day_night.day).speed(0.25)).changed() {
            day_night.paused = true;
        }
        for (label, days) in [("-1", -1), ("+1", 1), ("+7", 7)] {
            if ui.button(label).clicked() {
                day_night.day += days;
                day_night.paused = true;
            }
        }
    });
    // The buttons pick hours of the player's own day: sunrise and sunset
    // there, and its local noon and midnight.
    let orbits = Orbits::new(planet);
    let midnight = orbits.local_midnight(day_night.seconds(), eye);
    let up = eye.normalize();
    // Walking moves local midnight a little; a new day moves it a whole one.
    let stale = trainer.sun_times.is_none_or(|(day, place, _, _)| {
        (day - midnight).abs() > 0.25 * SOLAR_DAY || place.dot(up) < 0.1f64.to_radians().cos()
    });
    if stale {
        let (rise, set) = orbits.sunrise_and_sunset(day_night.seconds(), eye);
        trainer.sun_times = Some((midnight, up, rise, set));
    }
    let (_, _, sunrise, sunset) = trainer.sun_times.unwrap_or_default();
    ui.horizontal(|ui| {
        // A quarter hour after sunrise and before sunset.
        for (label, seconds) in [
            ("Dawn", sunrise.map_or(midnight + 0.26 * SOLAR_DAY, |time| time + 900.0)),
            ("Noon", midnight + 0.5 * SOLAR_DAY),
            ("Dusk", sunset.map_or(midnight + 0.74 * SOLAR_DAY, |time| time - 900.0)),
            ("Night", midnight),
        ] {
            if ui.button(label).clicked() {
                day_night.set_seconds(seconds);
                day_night.paused = true;
            }
        }
    });
    let clock = |seconds: f64| clock_label((seconds / 3600.0).rem_euclid(24.0));
    ui.small(format!(
        "Sunrise {}  ·  sunset {}  (local noon {})",
        sunrise.map_or("none".to_owned(), clock),
        sunset.map_or("none".to_owned(), clock),
        clock(midnight + 0.5 * SOLAR_DAY),
    ));
    ui.checkbox(&mut day_night.paused, "Hold selected time");
    let warp_label = |warp: f32| format!("{warp:.0}x");
    egui::ComboBox::from_label("Time warp")
        .selected_text(warp_label(day_night.warp))
        .show_ui(ui, |ui| {
            for warp in TIME_WARPS {
                ui.selectable_value(&mut day_night.warp, warp, warp_label(warp));
            }
        });
    if ui.button("Next eclipse").clicked() {
        // Look ten years ahead.
        let days = 10.0 * orbits.year() / SOLAR_DAY;
        trainer.sky_status = Some(match orbits.next_eclipse(day_night.seconds(), eye, days) {
            Some((seconds, kind)) => {
                day_night.set_seconds(seconds);
                day_night.paused = true;
                format!("{kind:?} eclipse on day {}.", day_night.day)
            }
            None => format!("No eclipse seen from here in the next {days:.0} days."),
        });
    }
    if let Some(status) = &trainer.sky_status {
        ui.small(status);
    }
    ui.small(sky_readout(day_night, planet, eye));
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
    // Over water too deep to stand in, the player is set afloat on it, not
    // on its bed.
    let afloat = crate::player::water_at(cache, noise, position.x, position.z, xz.to_array())
        .and_then(|water| water.float_eye());
    Vec3::new(position.x, afloat.map_or(y, |eye| eye.max(y)), position.z)
}

/// Teleport the player, and give it the water at its new pose at once: the
/// renderer draws this pose before the next player update looks.
fn teleport(player: &mut Player, position: Vec3, cache: &ErosionCache, noise: &NoiseField) -> bool {
    let moved = player.teleport(position);
    if moved {
        player.water = crate::player::water_at(cache, noise, position.x, position.z, [position.x, position.z]);
    }
    moved
}

#[allow(clippy::too_many_arguments)]
fn draw_player_tools(
    ui: &mut egui::Ui,
    player: &mut Player,
    trainer: &mut TrainerState,
    cache: &ErosionCache,
    noise: &NoiseField,
    snow: &SnowState,
    buildings: Option<&crate::buildings::BuildingField>,
    planet: &crate::planet::Planet,
) {
    if !trainer.initialized {
        trainer.use_position(player.position);
    }
    separator_text(ui, "Player and teleport");
    let mut flying = player.flying;
    if ui.checkbox(&mut flying, "Fly / noclip [V]").changed() {
        player.set_flying(flying);
    }
    let mut movement_speed_slider = egui::Slider::new(&mut player.movement_speed_multiplier, 0.1..=20.0);
    movement_speed_slider = movement_speed_slider.text("Movement speed");
    movement_speed_slider = movement_speed_slider.suffix("x");
    movement_speed_slider = movement_speed_slider.logarithmic(true);
    ui.add(movement_speed_slider);
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
    ui.small(format!("Plane coordinates on the {} face", player.face.name()));
    let mut coordinate_grid = egui::Grid::new("teleport_coordinates");
    coordinate_grid = coordinate_grid.num_columns(2);
    coordinate_grid.show(ui, |ui| {
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
                    if teleport(player, position, cache, noise) {
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
            // Spawn is the centre of the spawn face's chart. Its ground is
            // measured there, not on the face the height queries answer for
            // until the frame ends.
            let spawn = crate::planet::Face::PosY;
            let ground = crate::planet::height::surface_height(planet, cache, noise, spawn, 0.0, 0.0, [0.0, 0.0]);
            let sea = crate::player::WaterHere::sea(ground);
            let eye = sea.and_then(|water| water.float_eye()).map_or(ground + EYE_HEIGHT, |eye| eye.max(ground + EYE_HEIGHT));
            player.face = spawn;
            player.teleport(Vec3::new(0.0, eye, 0.0));
            player.water = sea;
            player.flying = false;
            player.yaw = Player::default().yaw;
            player.pitch = Player::default().pitch;
            trainer.use_position(player.position);
            trainer.status = Some("Returned to spawn.".to_owned());
        }
    });
    ui.horizontal(|ui| {
        if ui.button("Ground here").clicked() {
            teleport(player, ground_position(player.position, cache, noise, snow), cache, noise);
            player.flying = false;
        }
        if ui.button("Rise 100 m").clicked() {
            teleport(player, player.position + Vec3::Y * 100.0, cache, noise);
            player.flying = true;
        }
        if ui.button("Reset speed").clicked() {
            player.movement_speed_multiplier = 1.0;
        }
    });
    let drawn = |placement: &crate::buildings::placement::Placement| {
        crate::planet::height::features_whole(cache, placement.origin.x, placement.origin.z)
    };
    let nearest = buildings.and_then(|field| field.nearest(Vec2::new(player.position.x, player.position.z), drawn));
    if let Some(nearest) = nearest
        && ui.button(format!("Go to the nearest building ({}, {:.0} m)", nearest.name, nearest.distance)).clicked()
    {
        let spot = Vec3::new(nearest.approach.x, 0.0, nearest.approach.y);
        teleport(player, ground_position(spot, cache, noise, snow), cache, noise);
        player.flying = false;
        player.yaw = nearest.facing.x.atan2(-nearest.facing.y);
        player.pitch = -0.05;
        trainer.use_position(player.position);
        trainer.status = Some(format!("At the front door of a {}.", nearest.name));
    }
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
    buildings: Option<&crate::buildings::BuildingField>,
    planet: &crate::planet::Planet,
    eye: bevy::math::DVec3,
) {
    let mut open = settings.show_trainer;
    let previous_position = player.position;
    let top = if debug_bottom > 0.0 {
        debug_bottom + 12.0
    } else {
        ctx.content_rect().top() + 12.0
    };
    let trainer_panels = |ui: &mut egui::Ui| {
        draw_player_tools(ui, player, trainer, cache, noise, snow, buildings, planet);
        separator_text(ui, "Weather here");
        let selected = weather
            .trainer_preset
            .map(WeatherPreset::label)
            .unwrap_or("Natural weather");
        let mut requested = weather.trainer_preset;
        let mut trainer_weather = egui::ComboBox::from_id_salt("trainer_weather");
        trainer_weather = trainer_weather.selected_text(selected);
        trainer_weather.show_ui(ui, |ui| {
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
        let mut thunder_volume_slider = egui::Slider::new(&mut settings.thunder_volume, 0.0..=1.0);
        thunder_volume_slider = thunder_volume_slider.text("Thunder volume");
        thunder_volume_slider = thunder_volume_slider.fixed_decimals(2);
        ui.add(thunder_volume_slider);

        draw_sky_controls(ui, day_night, trainer, planet, eye);
        ui.collapsing("World and rendering shortcuts", |ui| {
            ui.checkbox(
                &mut settings.vegetation_enabled,
                "Trees, shrubs and flowers",
            );
            ui.checkbox(&mut settings.buildings_visible, "Buildings");
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
                buildings,
                cache,
            );
        }
        ui.small("2 close menu  |  3 debug text  |  F12 screenshot");
    };
    let mut window = egui::Window::new("Developer menu [2]");
    window = window.id(egui::Id::new("developer_menu"));
    window = window.anchor(
        egui::Align2::LEFT_TOP,
        egui::vec2(12.0, top - ctx.content_rect().top()),
    );
    window = window.default_width(360.0);
    window = window.max_height((ctx.content_rect().bottom() - top - 12.0).max(120.0));
    window = window.resizable(false);
    window = window.vscroll(true);
    window = window.open(&mut open);
    window.show(ctx, trainer_panels);
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
    buildings: Option<&crate::buildings::BuildingField>,
    cache: &ErosionCache,
) {
    separator_text(ui, "Weather");
    ui.checkbox(&mut weather.automatic, "Moving weather fronts");
    let mut weather_trend = egui::ComboBox::from_label("Weather trend");
    weather_trend = weather_trend.selected_text(weather.target.label());
    weather_trend.show_ui(ui, |ui| {
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
    let mut transition_slider = egui::Slider::new(&mut weather.transition_seconds, 10.0..=180.0);
    transition_slider = transition_slider.text("Trend transition");
    transition_slider = transition_slider.suffix(" s");
    transition_slider = transition_slider.fixed_decimals(0);
    ui.add(transition_slider);
    let mut thunder_volume_slider = egui::Slider::new(&mut settings.thunder_volume, 0.0..=1.0);
    thunder_volume_slider = thunder_volume_slider.text("Thunder volume");
    thunder_volume_slider = thunder_volume_slider.fixed_decimals(2);
    ui.add(thunder_volume_slider);

    separator_text(ui, "Sun and time");
    let mut time_of_day_slider =
        egui::Slider::new(&mut day_night.time_hours, 0.0..=23.99).clamping(egui::SliderClamping::Edits);
    time_of_day_slider = time_of_day_slider.text("Time of day");
    time_of_day_slider = time_of_day_slider.custom_formatter(|hours, _| {
        let minutes = (hours * 60.0).round() as u32 % (24 * 60);
        format!("{:02}:{:02}", minutes / 60, minutes % 60)
    });
    ui.add(time_of_day_slider);
    ui.checkbox(&mut day_night.paused, "Pause day/night cycle");
    let mut day_duration_slider = egui::Slider::new(&mut day_night.day_length_minutes, 1.0..=120.0);
    day_duration_slider = day_duration_slider.text("Day duration");
    day_duration_slider = day_duration_slider.suffix(" min");
    day_duration_slider = day_duration_slider.logarithmic(true);
    ui.add(day_duration_slider);

    separator_text(ui, "Raymarched lighting");
    ui.checkbox(&mut settings.raymarched_shadows, "Terrain shadows");
    ui.checkbox(&mut settings.volumetric_lighting, "Volumetric sunlight");
    ui.checkbox(&mut settings.water_reflections, "Water reflections");
    let quality_name = match settings.raymarch_quality {
        0 => "Low",
        2 => "High",
        _ => "Balanced",
    };
    let mut quality_combo = egui::ComboBox::from_label("Raymarch quality");
    quality_combo = quality_combo.selected_text(quality_name);
    quality_combo.show_ui(ui, |ui| {
        ui.selectable_value(&mut settings.raymarch_quality, 0, "Low");
        ui.selectable_value(&mut settings.raymarch_quality, 1, "Balanced");
        ui.selectable_value(&mut settings.raymarch_quality, 2, "High");
    });
    let mut light_shaft_slider = egui::Slider::new(&mut settings.volumetric_strength, 0.0..=2.0);
    light_shaft_slider = light_shaft_slider.text("Light shaft strength");
    light_shaft_slider = light_shaft_slider.fixed_decimals(2);
    ui.add_enabled(settings.volumetric_lighting, light_shaft_slider);

    separator_text(ui, "Volumetric clouds");
    ui.small(
        "Cloud controls set the cloudy baseline; weather adjusts coverage, height, and density.",
    );
    ui.checkbox(&mut settings.clouds_enabled, "Clouds");
    ui.add_enabled_ui(settings.clouds_enabled, |ui| {
        let mut cloud_coverage_slider = egui::Slider::new(&mut settings.cloud_coverage, 0.0..=1.0);
        cloud_coverage_slider = cloud_coverage_slider.text("Coverage");
        cloud_coverage_slider = cloud_coverage_slider.fixed_decimals(2);
        ui.add(cloud_coverage_slider);
        let mut cloud_density_slider = egui::Slider::new(&mut settings.cloud_density, 0.0..=4.0);
        cloud_density_slider = cloud_density_slider.text("Density");
        cloud_density_slider = cloud_density_slider.fixed_decimals(2);
        ui.add(cloud_density_slider);
        let mut cloud_base_slider =
            egui::Slider::new(&mut settings.cloud_base_height, 100.0..=6000.0);
        cloud_base_slider = cloud_base_slider.text("Cloud base");
        cloud_base_slider = cloud_base_slider.suffix(" m");
        cloud_base_slider = cloud_base_slider.fixed_decimals(0);
        ui.add(cloud_base_slider);
        let mut layer_thickness_slider =
            egui::Slider::new(&mut settings.cloud_thickness, 100.0..=4000.0);
        layer_thickness_slider = layer_thickness_slider.text("Layer thickness");
        layer_thickness_slider = layer_thickness_slider.suffix(" m");
        layer_thickness_slider = layer_thickness_slider.fixed_decimals(0);
        ui.add(layer_thickness_slider);
        let mut cloud_wind_speed_slider =
            egui::Slider::new(&mut settings.cloud_wind_speed, 0.0..=80.0);
        cloud_wind_speed_slider = cloud_wind_speed_slider.text("Cloud wind speed");
        cloud_wind_speed_slider = cloud_wind_speed_slider.suffix(" m/s");
        cloud_wind_speed_slider = cloud_wind_speed_slider.fixed_decimals(1);
        ui.add(cloud_wind_speed_slider);
        let mut cloud_wind_direction_slider =
            egui::Slider::new(&mut settings.cloud_wind_direction_degrees, 0.0..=360.0);
        cloud_wind_direction_slider = cloud_wind_direction_slider.text("Cloud wind direction");
        cloud_wind_direction_slider = cloud_wind_direction_slider.suffix("°");
        cloud_wind_direction_slider = cloud_wind_direction_slider.fixed_decimals(0);
        ui.add(cloud_wind_direction_slider);
        let mut cloud_shadow_slider =
            egui::Slider::new(&mut settings.cloud_shadow_strength, 0.0..=1.0);
        cloud_shadow_slider = cloud_shadow_slider.text("Cloud shadows");
        cloud_shadow_slider = cloud_shadow_slider.fixed_decimals(2);
        ui.add(cloud_shadow_slider);
        ui.collapsing("Cloud shape", |ui| {
            let mut formation_scale_slider =
                egui::Slider::new(&mut settings.cloud_scale, 300.0..=6000.0);
            formation_scale_slider = formation_scale_slider.text("Formation scale");
            formation_scale_slider = formation_scale_slider.suffix(" m");
            formation_scale_slider = formation_scale_slider.fixed_decimals(0);
            ui.add(formation_scale_slider);
            let mut edge_detail_slider =
                egui::Slider::new(&mut settings.cloud_detail_strength, 0.0..=1.0);
            edge_detail_slider = edge_detail_slider.text("Edge detail");
            edge_detail_slider = edge_detail_slider.fixed_decimals(2);
            ui.add(edge_detail_slider);
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
    let mut fog_slider = egui::Slider::new(&mut settings.fog_density, 0.0..=0.001);
    fog_slider = fog_slider.text("Fog");
    fog_slider = fog_slider.fixed_decimals(5);
    ui.add(fog_slider);
    let mut sun_intensity_slider = egui::Slider::new(&mut settings.sun_intensity, 0.0..=8.0);
    sun_intensity_slider = sun_intensity_slider.text("Sun intensity");
    sun_intensity_slider = sun_intensity_slider.fixed_decimals(2);
    ui.add(sun_intensity_slider);
    let mut exposure_slider = egui::Slider::new(&mut settings.exposure, 0.25..=4.0);
    exposure_slider = exposure_slider.text("Exposure");
    exposure_slider = exposure_slider.fixed_decimals(2);
    ui.add(exposure_slider);
    let mut snow_sparkle_slider = egui::Slider::new(&mut settings.sparkle_strength, 0.0..=2.0);
    snow_sparkle_slider = snow_sparkle_slider.text("Snow sparkle");
    snow_sparkle_slider = snow_sparkle_slider.fixed_decimals(2);
    ui.add(snow_sparkle_slider);

    separator_text(ui, "Vegetation");
    ui.checkbox(
        &mut settings.vegetation_enabled,
        "Trees, shrubs and flowers",
    );
    ui.checkbox(&mut settings.vegetation_shadows, "Plant shadows");
    let mut plant_detail_slider = egui::Slider::new(&mut settings.vegetation_detail, 0.4..=2.5);
    plant_detail_slider = plant_detail_slider.text("Plant detail distance");
    plant_detail_slider = plant_detail_slider.fixed_decimals(2);
    ui.add(plant_detail_slider);

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
            if let Some(water) = player.water.filter(|water| player.position.y < water.surface) {
                let kind = match water.kind {
                    crate::player::WaterKind::River => "a river's",
                    crate::player::WaterKind::Lake => "a lake's",
                    crate::player::WaterKind::Sea => "the sea's",
                };
                ui.label(format!(
                    "Eye {:.1} m under {kind} water, {:.1} m deep",
                    water.surface - player.position.y,
                    water.depth()
                ));
            }
        }
        None => {
            ui.label("Rivers disabled (--no-rivers)");
        }
    }

    separator_text(ui, "Buildings");
    ui.checkbox(&mut settings.buildings_visible, "Draw buildings");
    match buildings.filter(|field| field.enabled()) {
        Some(field) => {
            ui.label(format!("{} buildings placed within 2.4 km", field.building_count()));
            let drawn = |placement: &crate::buildings::placement::Placement| {
                crate::planet::height::features_whole(cache, placement.origin.x, placement.origin.z)
            };
            if let Some(nearest) = field.nearest(Vec2::new(player.position.x, player.position.z), drawn) {
                ui.label(format!("Nearest: {} at {:.0} m", nearest.name, nearest.distance));
            }
        }
        None => {
            ui.label("Buildings disabled (--no-buildings) or the kit failed to load");
        }
    }

    separator_text(ui, "Terrain textures");
    let mut texture_scale_slider = egui::Slider::new(&mut settings.texture_scale, 0.005..=0.6);
    texture_scale_slider = texture_scale_slider.text("Texture scale");
    texture_scale_slider = texture_scale_slider.fixed_decimals(3);
    ui.add(texture_scale_slider);
    let mut normal_strength_slider = egui::Slider::new(&mut settings.normal_strength, 0.0..=2.0);
    normal_strength_slider = normal_strength_slider.text("Normal strength");
    normal_strength_slider = normal_strength_slider.fixed_decimals(2);
    ui.add(normal_strength_slider);
    let mut variant_scale_slider = egui::Slider::new(&mut settings.variant_scale, 0.001..=0.05);
    variant_scale_slider = variant_scale_slider.text("Dirt/gravel variant scale");
    variant_scale_slider = variant_scale_slider.fixed_decimals(3);
    ui.add(variant_scale_slider);

    separator_text(ui, "Water");
    ui.checkbox(&mut water_settings.enabled, "Water surface");
    // The wave block is only rebuilt when one of these actually
    // changes, so graying them out while the surface is off keeps the
    // spectrum from being regenerated for a pass that will not run.
    ui.add_enabled_ui(water_settings.enabled, |ui| {
        let mut sea_state_slider =
            egui::Slider::new(&mut water_settings.sea_state_amplitude, 0.0..=1.5);
        sea_state_slider = sea_state_slider.text("Sea state");
        sea_state_slider = sea_state_slider.fixed_decimals(2);
        ui.add(sea_state_slider);
        let mut wind_direction_slider =
            egui::Slider::new(&mut water_settings.wind_direction_degrees, 0.0..=360.0);
        wind_direction_slider = wind_direction_slider.text("Wind direction");
        wind_direction_slider = wind_direction_slider.fixed_decimals(0);
        wind_direction_slider = wind_direction_slider.suffix("°");
        ui.add(wind_direction_slider);
        // A combo box rather than aqua's cycle-on-click button: the
        // presets are compared by value, so the selection survives
        // edits to the fields they do not carry.
        let selected = WaterOptics::PRESETS
            .iter()
            .position(|(_, preset)| *preset == water_settings.optics)
            .unwrap_or(0);
        let mut index = selected;
        let mut optics_combo = egui::ComboBox::from_label("Optics");
        optics_combo = optics_combo.selected_text(WaterOptics::PRESETS[selected].0);
        optics_combo.show_ui(ui, |ui| {
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
    let mut max_excavation_slider =
        egui::Slider::new(&mut erosion_settings.maximum_erosion, 1.0..=40.0);
    max_excavation_slider = max_excavation_slider.text("Max excavation");
    max_excavation_slider = max_excavation_slider.fixed_decimals(1);
    max_excavation_slider = max_excavation_slider.suffix(" m");
    ui.add(max_excavation_slider);
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
    let mut max_incision_slider =
        egui::Slider::new(&mut erosion_settings.maximum_incision, 0.0..=30.0);
    max_incision_slider = max_incision_slider.text("Max incision");
    max_incision_slider = max_incision_slider.fixed_decimals(1);
    max_incision_slider = max_incision_slider.suffix(" m");
    ui.add(max_incision_slider);
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
#[allow(clippy::too_many_arguments)]
fn draw_erosion_lattice(
    painter: &egui::Painter,
    player: &Player,
    planet: &crate::planet::Planet,
    cache: &ErosionCache,
    noise: &NoiseField,
    viewport: (u32, u32),
    pixels_per_point: f32,
) {
    // The same matrices the render side builds for `GlobalUniforms`, so the
    // overlay lines up with the rasterized terrain, and plane points placed
    // on the planet as the terrain's are.
    let camera = PlayerCamera::from_player(player);
    let planet_view = crate::planet::view::PlanetView::new(planet, player.face, camera.position, player.ground_below as f64);
    let aspect = viewport.0 as f32 / viewport.1 as f32;
    let view = matrices::view_matrix(camera.position, camera.target, camera.up);
    let far = planet_view.far_plane(planet, FAR_PLANE as f64) as f32;
    let projection = matrices::perspective(camera.fov_y, aspect, NEAR_PLANE, far);
    let on_planet = |x: f32, y: f32, z: f32| -> Vec3 {
        let chart = crate::planet::Planet::chart_of_plane(player.face, bevy::math::DVec2::new(x as f64, z as f64));
        planet_view.world_of(planet.position(player.face, chart, y as f64)).as_vec3()
    };

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
    let point_in_front = |x: f32, z: f32| {
        let world = on_planet(x, position.y, z);
        (world.x - position.x) * forward.x + (world.z - position.z) * forward.z > 1.0
    };
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
                let (start, end) = (on_planet(previous_x, h0, previous_z), on_planet(x, h1, z));
                if let (Some(start), Some(end)) =
                    (project(start.x, start.y, start.z), project(end.x, end.y, end.z))
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
    let shader_descriptor = wgpu::ShaderModuleDescriptor {
        label: Some("flow_preview_shader"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
            "../assets/shaders/flow-preview.wgsl"
        ))),
    };
    let shader = device.wgpu_device().create_shader_module(shader_descriptor);
    let target = key.target_format;
    let pipeline_descriptor = wgpu::RenderPipelineDescriptor {
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
    };
    let pipeline = device.create_render_pipeline(&pipeline_descriptor);
    let layout = BindGroupLayout::from(pipeline.get_bind_group_layout(0));
    let sampler_descriptor = wgpu::SamplerDescriptor {
        label: Some("flow_preview_sampler"),
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        address_mode_w: wgpu::AddressMode::Repeat,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..default()
    };
    let sampler = device.wgpu_device().create_sampler(&sampler_descriptor);
    let bind_group_entries = [
        wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(&textures.flow_atlas_view),
        },
        wgpu::BindGroupEntry {
            binding: 1,
            resource: wgpu::BindingResource::Sampler(&sampler),
        },
    ];
    let bind_group = device.create_bind_group(
        Some("flow_preview_bind_group"),
        &layout,
        &bind_group_entries,
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
            "Day 0",
            &WeatherState::default(),
            &ErosionCache::default(),
            &noise,
            &SnowState::default(),
            &Player::default(),
            None,
            None,
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
            for pass in 0..2 {
                ctx.begin_pass(egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(960.0, 640.0),
                    )),
                    ..Default::default()
                });
                let bottom = if debug_visible {
                    draw_debug_overlay(
                        &ctx, &settings, &day_night, "Day 0", &weather, &cache, &noise, &snow, &player, None, None, None,
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
                    None,
                    &crate::planet::Planet::default(),
                    bevy::math::DVec3::new(0.0, crate::planet::Planet::default().radius, 0.0),
                );
                let mut output = ctx.end_pass();
                output.textures_delta.clear();
                let rect = ctx
                    .memory(|memory| memory.area_rect(egui::Id::new("developer_menu")))
                    .unwrap();
                // A window sized by its last frame may sit a frame off.
                if pass == 0 {
                    continue;
                }
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
