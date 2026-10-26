use super::*;
use std::time::Duration;

#[test]
fn presets_change_the_whole_environment_and_preserve_cloudy_baseline() {
    let settings = AppSettings::default();
    let cloudy = WeatherState::default().conditions(&settings);
    assert_eq!(cloudy.fog_density, settings.fog_density);
    assert_eq!(cloudy.wind_speed, settings.cloud_wind_speed);

    // Cloud and sun values resolve in the shaders; the profile table is
    // what carries a preset's cloud tendencies.
    let cloudy_profile = WeatherProfile::for_preset(WeatherPreset::Cloudy);
    assert_eq!(cloudy_profile.coverage_delta, 0.0);
    assert_eq!(cloudy_profile.density_multiplier, 1.0);
    assert_eq!(cloudy_profile.base_offset, 0.0);
    assert_eq!(cloudy_profile.thickness_multiplier, 1.0);
    assert_eq!(cloudy_profile.shadow_multiplier, 1.0);
    let clear_profile = WeatherProfile::for_preset(WeatherPreset::Clear);
    let overcast_profile = WeatherProfile::for_preset(WeatherPreset::Overcast);
    assert!(clear_profile.coverage_delta < cloudy_profile.coverage_delta);
    assert!(overcast_profile.coverage_delta > cloudy_profile.coverage_delta);
    assert!(overcast_profile.sun_multiplier < cloudy_profile.sun_multiplier);
    assert!(overcast_profile.shadow_multiplier < cloudy_profile.shadow_multiplier);

    let clear = WeatherState::from_preset(WeatherPreset::Clear, false).conditions(&settings);
    let overcast = WeatherState::from_preset(WeatherPreset::Overcast, false).conditions(&settings);
    let fog = WeatherState::from_preset(WeatherPreset::Fog, false).conditions(&settings);
    assert!(clear.fog_density < cloudy.fog_density);
    assert!(fog.fog_density > overcast.fog_density * 10.0);
    assert!(fog.sky_overcast > overcast.sky_overcast);
    assert!(fog.wind_speed < cloudy.wind_speed);
    let fog_state = WeatherState::from_preset(WeatherPreset::Fog, false);
    assert!((fog_state.visibility_metres(&settings, 20.0) - 100.0).abs() < 1.0);
    assert!(fog_state.visibility_metres(&settings, 1000.0) > 300.0);
}

#[test]
fn weather_blends_continuously_and_cli_overrides_remain_exact() {
    let mut weather = WeatherState::from_preset(WeatherPreset::Cloudy, false);
    weather.transition_seconds = 10.0;
    weather.set_target(WeatherPreset::Overcast);
    weather.advance(5.0, [0.0, 0.0, 0.0], [0.0, 0.0], &AppSettings::default());
    assert!(weather.is_transitioning());
    assert!((weather.climate_bias - 0.5).abs() < 1e-5);
    let before_retarget = weather.climate_bias;
    weather.set_target(WeatherPreset::Clear);
    assert_eq!(weather.climate_bias, before_retarget);
    weather.advance(10.0, [0.0, 0.0, 0.0], [0.0, 0.0], &AppSettings::default());
    assert_eq!(weather.climate_bias, WeatherPreset::Clear.climate_bias());
    assert_eq!(
        weather.current,
        WeatherProfile::from_severity(
            front_severity([0.0, 0.0], [0.0, 0.0], weather.climate_bias,)
        )
    );
    assert!(!weather.is_transitioning());

    weather.overrides = WeatherOverrides {
        coverage: true,
        density: true,
        base: true,
        thickness: true,
    };
    // The shader resolves cloud values from these bits (bit 0 coverage,
    // 1 density, 2 base, 3 thickness); they must map exactly.
    assert_eq!(weather.override_bits(), 0b1111);
    let settings = AppSettings {
        fog_density: 0.0,
        ..AppSettings::default()
    };
    // --no-fog stays fog-free whatever the weather profile is.
    let resolved = weather.conditions(&settings);
    assert_eq!(resolved.fog_density, 0.0);
}

#[test]
fn trainer_weather_follows_the_player_and_moving_fronts() {
    let settings = AppSettings::default();
    let locations = [
        ([0.0, 0.0, 0.0], [0.0, 0.0]),
        ([28_000.0, 80.0, -19_000.0], [1_500.0, -700.0]),
        ([-400_000.0, 0.0, -130_000.0], [0.0, 0.0]),
        ([-12_000.0, 25.0, 36_000.0], [-9_000.0, 3_000.0]),
    ];
    for preset in WeatherPreset::ALL {
        let mut weather = WeatherState::default();
        weather.set_trainer_preset(Some(preset));
        for (position, offset) in locations {
            weather.refresh_local(position, offset, &settings);
            assert_eq!(
                weather.local_condition(),
                preset,
                "{preset:?} at {position:?}"
            );
            let wanted_severity = match preset {
                WeatherPreset::Clear => 0.0,
                WeatherPreset::Cloudy => 1.0,
                WeatherPreset::Overcast => 2.0,
                WeatherPreset::Fog => 3.0,
                WeatherPreset::Rain | WeatherPreset::Snow | WeatherPreset::Thunderstorm => 2.0,
            };
            assert!((weather.local_severity - wanted_severity).abs() < 0.0001);
            let precipitation = weather.local_precipitation;
            match preset {
                WeatherPreset::Clear
                | WeatherPreset::Cloudy
                | WeatherPreset::Overcast
                | WeatherPreset::Fog => {
                    assert_eq!(precipitation.intensity(), 0.0);
                }
                WeatherPreset::Rain => {
                    assert!(precipitation.rain > 0.17);
                    assert_eq!(precipitation.snow, 0.0);
                    assert_eq!(precipitation.thunderstorm, 0.0);
                }
                WeatherPreset::Snow => {
                    assert!(precipitation.snow > 0.17);
                    assert_eq!(precipitation.rain, 0.0);
                }
                WeatherPreset::Thunderstorm => {
                    assert!(precipitation.rain > 0.17);
                    assert!(precipitation.thunderstorm > 0.9);
                }
            }
            weather.advance(0.25, position, offset, &settings);
            assert_eq!(weather.local_condition(), preset);
            assert!(!weather.is_transitioning());
        }
    }
}

#[test]
fn trainer_refresh_keeps_clock_and_lightning_untouched() {
    let settings = AppSettings::default();
    let mut weather = WeatherState::default();
    weather.set_trainer_preset(Some(WeatherPreset::Thunderstorm));
    weather.elapsed_seconds = 12.0;
    weather.lightning.flash = 3.0;
    weather.new_strike = true;
    weather.refresh_local([0.0; 3], [0.0; 2], &settings);
    assert_eq!(weather.elapsed_seconds, 12.0);
    assert_eq!(weather.lightning.flash, 3.0);
    assert!(weather.new_strike);
}

#[test]
fn leaving_trainer_restores_cloudy_and_normal_transitions() {
    let settings = AppSettings::default();
    let position = [28_000.0, 0.0, -19_000.0];
    let offset = [1_500.0, -700.0];
    let mut weather = WeatherState::default();
    weather.set_trainer_preset(Some(WeatherPreset::Fog));
    weather.refresh_local(position, offset, &settings);
    assert_eq!(weather.local_condition(), WeatherPreset::Fog);
    weather.set_trainer_preset(None);
    weather.refresh_local(position, offset, &settings);
    assert_eq!(weather.trainer_preset, None);
    assert_eq!(weather.target, WeatherPreset::Cloudy);
    assert_eq!(weather.climate_bias, 0.0);
    assert_eq!(weather.precipitation_bias, 0.0);
    assert_eq!(
        weather.local_severity,
        front_severity([position[0], position[2]], offset, 0.0)
    );
    assert!(!weather.is_transitioning());

    weather.set_trainer_preset(Some(WeatherPreset::Rain));
    weather.refresh_local(position, offset, &settings);
    assert_eq!(weather.precipitation_override(), 4);
    // A normal selection of the same named preset still releases the lock.
    weather.set_target(WeatherPreset::Rain);
    assert_eq!(weather.trainer_preset, None);
    assert!(weather.is_transitioning());
    assert_eq!(weather.precipitation_override(), 1);
    weather.advance(75.0, position, offset, &settings);
    assert_eq!(weather.climate_bias, WeatherPreset::Rain.climate_bias());
    assert_eq!(
        weather.precipitation_bias,
        WeatherPreset::Rain.precipitation_bias()
    );
    assert!(!weather.is_transitioning());
}

#[test]
fn moving_fronts_preserve_simultaneous_weather_regions() {
    let mut weather = WeatherState::default();
    let mut minimum = 3.0f32;
    let mut maximum = 0.0f32;
    for z in -10..=10 {
        for x in -10..=10 {
            let severity = front_severity([x as f32 * 4000.0, z as f32 * 4000.0], [0.0, 0.0], 0.0);
            minimum = minimum.min(severity);
            maximum = maximum.max(severity);
        }
    }
    assert!(minimum < 0.4 && maximum > 2.8);
    let location = [3000.0, -1200.0];
    weather.advance(
        0.0,
        [location[0], 0.0, location[1]],
        [0.0, 0.0],
        &AppSettings::default(),
    );
    let first = weather.local_severity;
    weather.advance(
        1.0,
        [location[0], 0.0, location[1]],
        [1500.0, 0.0],
        &AppSettings::default(),
    );
    assert_ne!(first, weather.local_severity);
    weather.advance(
        1.0,
        [location[0] + 0.1, 0.0, location[1]],
        [1500.0, 0.0],
        &AppSettings::default(),
    );
    assert!((weather.local_severity - front_severity(location, [1500.0, 0.0], 0.0)).abs() < 0.001);
}

#[test]
fn motion_integrates_elapsed_time_without_jumping_when_wind_changes() {
    let mut weather = WeatherMotion::default();
    weather.advance(2.0, 18.0, 0.0);
    assert_eq!(weather.offset, [36.0, 0.0]);
    weather.advance(1.0, 10.0, 90.0);
    assert!((weather.offset[0] - 36.0).abs() < 1e-5);
    assert!((weather.offset[1] - 10.0).abs() < 1e-5);
    weather.advance(300.0, 0.0, 180.0);
    assert_eq!(weather.offset, [36.0, 10.0]);
    weather.advance(1.0, 10.0, 180.0);
    assert_eq!(weather.offset, [26.0, 10.0]);
}

#[test]
fn invalid_motion_cannot_poison_shader_coordinates() {
    let mut weather = WeatherMotion {
        offset: [20.0, 30.0],
    };
    for delta in [-1.0, 0.0, f32::NAN, f32::INFINITY] {
        weather.advance(delta, 18.0, 70.0);
    }
    for speed in [-1.0, f32::NAN, f32::INFINITY] {
        weather.advance(1.0, speed, 70.0);
    }
    weather.advance(1.0, 18.0, f32::NAN);
    weather.advance(f32::MAX, f32::MAX, 0.0);
    assert_eq!(weather.offset, [20.0, 30.0]);
}

#[test]
fn capture_weather_is_frozen_unless_explicitly_advanced() {
    use bevy::prelude::{Schedule, World};

    for (arguments, expected_motion) in [
        (vec!["--shot", "test.png"], false),
        (vec!["--shot", "test.png", "--advance-time"], true),
        (vec!["--static-weather"], false),
        (vec![], true),
        (
            vec!["--shot", "test.png", "--advance-time", "--pause-time"],
            false,
        ),
    ] {
        let mut world = World::new();
        let mut time = Time::<Real>::default();
        time.advance_by(Duration::from_secs(1));
        world.insert_resource(time);
        world.insert_resource(AppSettings::default());
        let automation =
            crate::automation::parse_automation(arguments.into_iter().map(str::to_string));
        world.insert_resource(automation.clone());
        world.init_resource::<WeatherMotion>();
        world.init_resource::<ActiveBolt>();
        world.insert_resource(WeatherState::from_preset(
            automation.weather_preset,
            !automation.static_weather,
        ));
        let mut schedule = Schedule::default();
        schedule.add_systems(advance_weather);
        schedule.run(&mut world);
        let offset = world.resource::<WeatherMotion>().offset;
        assert_eq!(offset != [0.0, 0.0], expected_motion);
        let state = world.resource::<WeatherState>();
        assert_eq!(
            state.local_severity,
            front_severity([0.0, 0.0], offset, state.climate_bias)
        );
    }
}

#[test]
fn ordinary_weather_dominates_spatial_storm_cells() {
    let mut weather = WeatherState::default();
    let mut sea_level = [0usize; 7];
    let mut highland = [0usize; 7];
    for z in -30..=30 {
        for x in -30..=30 {
            let position = [x as f32 * 3000.0, 0.0, z as f32 * 3000.0];
            weather.advance(0.0, position, [0.0, 0.0], &AppSettings::default());
            sea_level[weather.local_condition() as usize] += 1;
            weather.advance(
                0.0,
                [position[0], 500.0, position[2]],
                [0.0, 0.0],
                &AppSettings::default(),
            );
            highland[weather.local_condition() as usize] += 1;
        }
    }
    let count = |preset: WeatherPreset, values: [usize; 7]| values[preset as usize];
    for common in [
        WeatherPreset::Clear,
        WeatherPreset::Cloudy,
        WeatherPreset::Overcast,
    ] {
        assert!(count(common, sea_level) > count(WeatherPreset::Fog, sea_level));
    }
    assert!(count(WeatherPreset::Fog, sea_level) > count(WeatherPreset::Rain, sea_level));
    assert!(count(WeatherPreset::Fog, highland) > count(WeatherPreset::Snow, highland));
    assert!(count(WeatherPreset::Rain, sea_level) > count(WeatherPreset::Thunderstorm, sea_level));
    assert_eq!(count(WeatherPreset::Snow, sea_level), 0);
}

#[test]
fn wind_advects_heavy_and_light_rain_sheets_through_a_fixed_location() {
    let angle = 70.0_f32.to_radians();
    let direction = [angle.cos(), angle.sin()];
    let location = [3200.0, -1700.0];
    let displacement = [direction[0] * 800.0, direction[1] * 800.0];
    let first = rain_band_multiplier(location, [0.0, 0.0], angle);
    let shifted = rain_band_multiplier(
        [location[0] + displacement[0], location[1] + displacement[1]],
        displacement,
        angle,
    );
    assert!((first - shifted).abs() < 0.0001);

    let mut minimum = 1.0_f32;
    let mut maximum = 0.0_f32;
    for second in 0..=120 {
        let offset = [
            direction[0] * second as f32 * 18.0,
            direction[1] * second as f32 * 18.0,
        ];
        let rain =
            sample_precipitation(location, 0.0, offset, 2.0, 1.0, 4, 1300.0, false, angle).rain;
        minimum = minimum.min(rain);
        maximum = maximum.max(rain);
    }
    // A lull is visibly lighter than a squall line, yet still steady rain.
    assert!(
        minimum > 0.40 && minimum < 0.55,
        "rain lull should stay wet: {minimum}"
    );
    assert!(maximum > 0.95, "heavy sheet should follow: {maximum}");
}

#[test]
fn lightning_requires_a_convective_cell_and_has_short_pulses() {
    let mut clear = WeatherState::from_preset(WeatherPreset::Clear, false);
    clear.advance(60.0, [0.0; 3], [0.0; 2], &AppSettings::default());
    assert!(!clear.new_strike);
    assert_eq!(clear.lightning.flash, 0.0);

    let mut storm = WeatherState::from_preset(WeatherPreset::Thunderstorm, false);
    storm.advance(5.0, [0.0; 3], [0.0; 2], &AppSettings::default());
    assert!(storm.local_precipitation.thunderstorm > 0.12);
    assert!(storm.new_strike);
    let strike = storm.lightning.position;
    assert!(strike[0].hypot(strike[2]) >= 120.0);
    let first = storm.lightning.first_stroke_seconds();
    storm.advance(first, [0.0; 3], [0.0; 2], &AppSettings::default());
    assert!(
        storm.lightning.flash > 0.65,
        "first stroke: {}",
        storm.lightning.flash
    );
    let duration = storm.lightning.duration_seconds();
    assert!(duration < 2.5);
    storm.advance(duration, [0.0; 3], [0.0; 2], &AppSettings::default());
    assert!(
        storm.lightning.flash < 0.001,
        "after: {}",
        storm.lightning.flash
    );
    assert!(storm.lightning.channel < 0.01);
}

#[test]
fn ground_bolts_restrike_down_the_main_channel_only() {
    let settings = AppSettings::default();
    let mut storm = WeatherState::from_preset(WeatherPreset::Thunderstorm, false);
    for strike in 0..40 {
        storm.force_strike([900.0, 0.0], BoltKind::Ground, -1.0, [0.0; 2], &settings);
        let event = storm.lightning;
        assert!(
            (2..=5).contains(&event.stroke_count),
            "strokes {}",
            event.stroke_count
        );
        let leader = event.pulse(event.first_stroke_seconds() * 0.5);
        assert!(leader.leader > 0.3 && leader.leader < 0.7);
        assert!(
            leader.channel < 0.2 && leader.flash < 0.05,
            "a leader is faint"
        );
        let first = event.pulse(event.first_stroke_seconds());
        assert_eq!(first.leader, 1.0);
        assert!(
            first.channel > 0.95 && first.branches > 0.95,
            "strike {strike}"
        );
        // Each restrike relights the channel while the branches stay dark.
        for stroke in &event.strokes[1..event.stroke_count] {
            let restrike = event.pulse(stroke.time);
            assert!(restrike.channel > stroke.peak, "restrike {restrike:?}");
            assert!(
                restrike.branches < 0.4 * restrike.channel,
                "branches {restrike:?}"
            );
        }
        for pair in event.strokes[..event.stroke_count].windows(2) {
            let gap = pair[1].time - pair[0].time;
            assert!((0.04..=0.12).contains(&gap), "gap {gap}");
        }
    }
}

#[test]
fn storms_discharge_every_few_seconds_with_occasional_bursts() {
    let settings = AppSettings::default();
    let mut storm = WeatherState::from_preset(WeatherPreset::Thunderstorm, false);
    let mut strikes = 0;
    let mut quick = 0;
    let mut previous = None;
    let mut seconds = 0.0;
    while seconds < 600.0 {
        storm.advance(0.05, [0.0; 3], [0.0; 2], &settings);
        seconds += 0.05;
        if storm.new_strike {
            strikes += 1;
            if let Some(previous) = previous
                && seconds - previous < 2.1
            {
                quick += 1;
            }
            previous = Some(seconds);
        }
    }
    // About one discharge every 4-12 seconds under a strong cell.
    assert!((50..150).contains(&strikes), "strikes {strikes}");
    assert!(quick > 5, "bursts {quick}");
}

#[test]
fn thunderstorms_produce_both_cloud_flashes_and_ground_bolts() {
    let mut storm = WeatherState::from_preset(WeatherPreset::Thunderstorm, false);
    let settings = AppSettings::default();
    let mut cloud_flashes = 0;
    let mut crawlers = 0;
    let mut ground_bolts = 0;
    for _ in 0..40 {
        storm.advance(40.0, [0.0; 3], [0.0; 2], &settings);
        assert!(storm.new_strike);
        assert_eq!(
            storm.lightning.seed < 0.0,
            storm.lightning.kind != BoltKind::Ground
        );
        match storm.lightning.kind {
            BoltKind::Hidden => cloud_flashes += 1,
            BoltKind::Crawler => crawlers += 1,
            BoltKind::Ground => ground_bolts += 1,
        }
        assert!(storm.lightning.top_height > storm.lightning.cloud_base);
    }
    assert!(cloud_flashes > 0 && crawlers > 0 && ground_bolts > 0);
}
