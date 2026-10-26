use super::*;

fn parse(arguments: &[&str]) -> AutomationSettings {
    parse_automation(arguments.iter().map(|value| (*value).to_string()))
}

#[test]
fn erosion_prewarm_is_bounded_and_keeps_following_flags() {
    assert_eq!(parse(&[]).erosion_prewarm, DEFAULT_EROSION_PREWARM);
    assert_eq!(parse(&["--erosion-prewarm", "16"]).erosion_prewarm, 16);
    assert_eq!(parse(&["--erosion-prewarm", "500"]).erosion_prewarm, 81);
    assert_eq!(parse(&["--erosion-prewarm", "0"]).erosion_prewarm, 1);
    for invalid in ["NaN", "inf", "bad"] {
        let settings = parse(&["--erosion-prewarm", invalid, "--pause-time"]);
        assert_eq!(settings.erosion_prewarm, DEFAULT_EROSION_PREWARM);
        assert!(settings.pause_time);
    }
}

#[test]
fn lighting_arguments_override_defaults() {
    let settings = parse(&[
        "--time-of-day",
        "18.5",
        "--day-length",
        "3",
        "--pause-time",
        "--no-raymarch",
        "--raymarch-quality",
        "high",
    ]);
    assert_eq!(settings.time_of_day, 18.5);
    assert_eq!(settings.day_length_minutes, 3.0);
    assert!(settings.pause_time && settings.no_raymarch);
    assert_eq!(settings.raymarch_quality, 2);
}

#[test]
fn invalid_lighting_values_preserve_defaults_and_following_flags() {
    for invalid in ["NaN", "inf", "-1", "24", "bad"] {
        let settings = parse(&["--time-of-day", invalid, "--pause-time"]);
        assert_eq!(settings.time_of_day, 10.0);
        assert!(settings.pause_time);
    }
    for invalid in ["NaN", "inf", "-1", "0", "bad"] {
        let settings = parse(&["--day-length", invalid]);
        assert_eq!(settings.day_length_minutes, 24.0);
    }
    let settings = parse(&[
        "--time-of-day",
        "--pause-time",
        "--raymarch-quality",
        "ultra",
    ]);
    assert_eq!(settings.time_of_day, 10.0);
    assert_eq!(settings.raymarch_quality, 1);
    assert!(settings.pause_time);
    assert_eq!(parse(&["--day-length"]).day_length_minutes, 24.0);
}

#[test]
fn captures_freeze_time_unless_explicitly_advanced() {
    assert!(!parse(&[]).pause_time);
    assert!(parse(&["--shot", "test.png"]).pause_time);
    assert!(!parse(&["--shot", "test.png", "--advance-time"]).pause_time);
    assert!(parse(&["--shot", "test.png", "--advance-time", "--pause-time"]).pause_time);
}

#[test]
fn weather_flags_select_starting_condition_and_preserve_following_flags() {
    for (name, preset) in [
        ("clear", WeatherPreset::Clear),
        ("cloudy", WeatherPreset::Cloudy),
        ("overcast", WeatherPreset::Overcast),
        ("fog", WeatherPreset::Fog),
        ("whiteout", WeatherPreset::Fog),
        ("rain", WeatherPreset::Rain),
        ("snow", WeatherPreset::Snow),
        ("thunderstorm", WeatherPreset::Thunderstorm),
        ("storm", WeatherPreset::Thunderstorm),
    ] {
        let settings = parse(&["--weather", name, "--static-weather"]);
        assert_eq!(settings.weather_preset, preset);
        assert!(settings.static_weather);
    }
    let settings = parse(&["--weather", "invalid", "--no-clouds"]);
    assert_eq!(settings.weather_preset, WeatherPreset::Cloudy);
    assert!(settings.no_clouds);
}

#[test]
fn lightning_flag_holds_a_discharge_at_a_chosen_moment() {
    let settings = parse(&["--lightning", "0.08", "--pause-time"]);
    assert_eq!(
        settings.lightning,
        Some(ForcedLightning {
            age_seconds: 0.08,
            distance: 900.0,
            bearing_degrees: 0.0,
            kind: BoltKind::Ground,
        })
    );
    assert!(settings.pause_time);
    let settings = parse(&["--lightning", "0.3,1800,-20,crawler"]);
    let forced = settings.lightning.unwrap();
    assert_eq!((forced.distance, forced.bearing_degrees), (1800.0, -20.0));
    assert_eq!(forced.kind, BoltKind::Crawler);
    assert_eq!(
        parse(&["--lightning", "0.1,500,0,cloud"])
            .lightning
            .unwrap()
            .kind,
        BoltKind::Hidden
    );
    assert_eq!(
        parse(&["--lightning", "-0.03"])
            .lightning
            .unwrap()
            .age_seconds,
        -0.03
    );
    for invalid in ["-2", "bad", "0.1,NaN", "0.1,900,0,spark", "0.1,-5"] {
        let settings = parse(&["--lightning", invalid, "--no-clouds"]);
        assert_eq!(settings.lightning, None, "{invalid}");
        assert!(settings.no_clouds);
    }
}

#[test]
fn cloud_arguments_accept_extremes_and_overrides() {
    let settings = parse(&[
        "--no-clouds",
        "--cloud-coverage",
        "1",
        "--cloud-density",
        "4",
        "--cloud-base",
        "6000",
        "--cloud-thickness",
        "4000",
    ]);
    assert!(settings.no_clouds);
    assert_eq!(settings.cloud_coverage, 1.0);
    assert_eq!(settings.cloud_density, 4.0);
    assert_eq!(settings.cloud_base_height, 6000.0);
    assert_eq!(settings.cloud_thickness, 4000.0);
    let settings = parse(&[
        "--cloud-coverage",
        "0",
        "--cloud-density",
        "0",
        "--cloud-base",
        "100",
        "--cloud-thickness",
        "100",
    ]);
    assert_eq!(settings.cloud_coverage, 0.0);
    assert_eq!(settings.cloud_density, 0.0);
    assert_eq!(settings.cloud_base_height, 100.0);
    assert_eq!(settings.cloud_thickness, 100.0);
}

#[test]
fn invalid_cloud_values_preserve_previous_value_and_following_flags() {
    for (flag, valid, invalid) in [
        ("--cloud-coverage", "0.7", "1.01"),
        ("--cloud-density", "2", "4.01"),
        ("--cloud-base", "1500", "6001"),
        ("--cloud-thickness", "900", "4001"),
    ] {
        let expected = parse(&[flag, valid]);
        for invalid in [invalid, "NaN", "inf", "-1", "bad", "--pause-time"] {
            let settings = parse(&[flag, valid, flag, invalid, "--no-clouds"]);
            assert_eq!(settings.cloud_coverage, expected.cloud_coverage);
            assert_eq!(settings.cloud_density, expected.cloud_density);
            assert_eq!(settings.cloud_base_height, expected.cloud_base_height);
            assert_eq!(settings.cloud_thickness, expected.cloud_thickness);
            assert!(settings.no_clouds);
            if invalid == "--pause-time" {
                assert!(settings.pause_time);
            }
        }
        let missing = parse(&[flag]);
        let defaults = AutomationSettings::default();
        assert_eq!(missing.cloud_coverage, defaults.cloud_coverage);
        assert_eq!(missing.cloud_density, defaults.cloud_density);
        assert_eq!(missing.cloud_base_height, defaults.cloud_base_height);
        assert_eq!(missing.cloud_thickness, defaults.cloud_thickness);
    }
    assert_eq!(parse(&["--cloud-base", "99"]).cloud_base_height, 1300.0);
    assert_eq!(parse(&["--cloud-thickness", "99"]).cloud_thickness, 1200.0);
}
