//! Command-line automation, ported from the C++ `AutomationSettings` and
//! `ParseAutomation`.
//!
//! `--time-of-day H` selects a celestial pose; `--day-length MIN` sets real
//! minutes per full day. `--pause-time` freezes it, and screenshot runs freeze
//! it automatically unless `--advance-time` is supplied. `--no-raymarch` turns
//! off terrain shadows, volume integration, clouds and water reflection tracing;
//! `--raymarch-quality low|balanced|high` selects their sampling budget.
//! `--no-clouds` disables clouds independently; `--cloud-coverage`,
//! `--cloud-density`, `--cloud-base` and `--cloud-thickness` set cloud weather.

use crate::constants::AppSettings;
use crate::erosion::TileKey;
use bevy::prelude::Resource;

/// `--camera x,y,z,yawDeg,pitchDeg` pins the player at a world pose (useful
/// for reproducible before/after comparisons); `--shot path` saves a window
/// screenshot after `--wait` frames and exits; `--probe` dumps the base
/// landform as x,z,height CSV to stdout so camera locations can be scouted
/// from the shell without opening a window; `--size W,H` (or `WxH`) overrides
/// the window dimensions so captures can match a physical display scale.
/// Shot runs leave the cursor enabled so an unattended window never grabs the
/// mouse. `--lattice` overlays the 512 m erosion-tile grid on the capture;
/// `--no-fog` zeroes the composite fog density and `--no-water` skips the
/// ocean plane so tile-boundary geometry stays measurable.
/// `--measure-overlap [--overlap-tile x,z]` simulates one adjacent tile pair,
/// prints the per-metre height disagreement across their shared overlap as
/// CSV, and exits — the seam regression check for the erosion boundaries.
#[derive(Clone, Debug, Resource)]
pub struct AutomationSettings {
    pub has_camera: bool,
    pub position: [f32; 3],
    /// radians, matching Player
    pub yaw: f32,
    /// radians, matching Player
    pub pitch: f32,
    pub shot_path: Option<String>,
    pub wait_frames: i32,
    pub probe: bool,
    pub width: i32,
    pub height: i32,
    pub probe_extent: f32,
    pub probe_step: f32,
    pub measure_overlap: bool,
    pub lattice: bool,
    pub no_fog: bool,
    pub no_water: bool,
    pub time_of_day: f32,
    pub day_length_minutes: f32,
    pub pause_time: bool,
    pub no_raymarch: bool,
    pub raymarch_quality: u32,
    pub no_clouds: bool,
    pub cloud_coverage: f32,
    pub cloud_density: f32,
    pub cloud_base_height: f32,
    pub cloud_thickness: f32,
    pub overlap_tile: TileKey,
}

impl Default for AutomationSettings {
    fn default() -> Self {
        let render_settings = AppSettings::default();
        Self {
            has_camera: false,
            position: [0.0, 0.0, 0.0],
            yaw: 0.0,
            pitch: 0.0,
            shot_path: None,
            wait_frames: 600,
            probe: false,
            width: 1600,
            height: 900,
            probe_extent: 4096.0,
            probe_step: 128.0,
            measure_overlap: false,
            lattice: false,
            no_fog: false,
            no_water: false,
            time_of_day: 10.0,
            day_length_minutes: 24.0,
            pause_time: false,
            no_raymarch: false,
            raymarch_quality: 1,
            no_clouds: false,
            cloud_coverage: render_settings.cloud_coverage,
            cloud_density: render_settings.cloud_density,
            cloud_base_height: render_settings.cloud_base_height,
            cloud_thickness: render_settings.cloud_thickness,
            overlap_tile: TileKey { x: 0, z: 0 },
        }
    }
}

const DEG_TO_RAD: f32 = std::f32::consts::PI / 180.0;

fn parse_floats(argument: &str, expected: usize) -> Option<Vec<f32>> {
    let values: Option<Vec<f32>> = argument
        .split(',')
        .map(|part| part.trim().parse::<f32>().ok())
        .collect();
    values.filter(|values| values.len() == expected)
}

fn parse_ints(argument: &str, expected: usize) -> Option<Vec<i64>> {
    parse_floats(argument, expected)
        .map(|values| values.into_iter().map(|value| value as i64).collect())
}

/// Parses `N` floats separated by commas; accepts a trailing-`x`/`,` variant
/// the C++ `sscanf` tolerated on malformed input only when all `N` parse.
pub fn parse_automation(arguments: impl Iterator<Item = String>) -> AutomationSettings {
    let arguments: Vec<String> = arguments.collect();
    let mut automation = AutomationSettings::default();
    let mut advance_capture_time = false;
    let mut index = 0;
    while index < arguments.len() {
        let flag = arguments[index].as_str();
        let next = arguments.get(index + 1);
        match flag {
            "--camera" => {
                if let Some(next) = next {
                    index += 1;
                    match parse_floats(next, 5) {
                        Some(v) => {
                            automation.has_camera = true;
                            automation.position = [v[0], v[1], v[2]];
                            automation.yaw = v[3] * DEG_TO_RAD;
                            automation.pitch = v[4] * DEG_TO_RAD;
                        }
                        None => println!("WARNING: --camera expects x,y,z,yawDeg,pitchDeg"),
                    }
                }
            }
            "--shot" => {
                if let Some(next) = next {
                    index += 1;
                    automation.shot_path = Some(next.clone());
                }
            }
            "--wait" => {
                if let Some(next) = next {
                    index += 1;
                    // atoi semantics: junk parses to zero, clamped up to 1.
                    automation.wait_frames = next
                        .parse::<f32>()
                        .map(|value| value as i32)
                        .unwrap_or(0)
                        .max(1);
                }
            }
            "--probe" => automation.probe = true,
            "--measure-overlap" => automation.measure_overlap = true,
            "--overlap-tile" => {
                if let Some(next) = next {
                    index += 1;
                    match parse_ints(next, 2) {
                        Some(v) => automation.overlap_tile = TileKey { x: v[0], z: v[1] },
                        None => println!("WARNING: --overlap-tile expects x,z tile keys"),
                    }
                }
            }
            "--lattice" => automation.lattice = true,
            "--no-fog" => automation.no_fog = true,
            "--no-water" => automation.no_water = true,
            "--pause-time" => automation.pause_time = true,
            "--advance-time" => advance_capture_time = true,
            "--no-raymarch" => automation.no_raymarch = true,
            "--no-clouds" => automation.no_clouds = true,
            "--cloud-coverage" | "--cloud-density" | "--cloud-base" | "--cloud-thickness" => {
                let bounds = match flag {
                    "--cloud-coverage" => 0.0..=1.0,
                    "--cloud-density" => 0.0..=4.0,
                    "--cloud-base" => 100.0..=6000.0,
                    _ => 100.0..=4000.0,
                };
                let parsed = next
                    .and_then(|value| value.parse::<f32>().ok())
                    .filter(|value| value.is_finite() && bounds.contains(value));
                if let Some(value) = parsed {
                    match flag {
                        "--cloud-coverage" => automation.cloud_coverage = value,
                        "--cloud-density" => automation.cloud_density = value,
                        "--cloud-base" => automation.cloud_base_height = value,
                        _ => automation.cloud_thickness = value,
                    }
                    index += 1;
                } else {
                    eprintln!(
                        "WARNING: {flag} expects a finite value in [{}, {}]; keeping the previous value",
                        bounds.start(), bounds.end()
                    );
                    if next.is_some_and(|value| !value.starts_with("--")) {
                        index += 1;
                    }
                }
            }
            "--time-of-day" | "--day-length" => {
                let parsed = next
                    .and_then(|value| value.parse::<f32>().ok())
                    .filter(|value| value.is_finite())
                    .filter(|value| {
                        if flag == "--time-of-day" {
                            (0.0..24.0).contains(value)
                        } else {
                            *value > 0.0
                        }
                    });
                if let Some(value) = parsed {
                    if flag == "--time-of-day" {
                        automation.time_of_day = value;
                    } else {
                        automation.day_length_minutes = value;
                    }
                    index += 1;
                } else {
                    let expected = if flag == "--time-of-day" {
                        "an hour in [0, 24)"
                    } else {
                        "positive real minutes per day"
                    };
                    eprintln!("WARNING: {flag} expects {expected}; keeping the previous value");
                    // Keep a following flag available to the parser.
                    if next.is_some_and(|value| !value.starts_with("--")) {
                        index += 1;
                    }
                }
            }
            "--raymarch-quality" => {
                let quality = next.and_then(|value| match value.as_str() {
                    "low" | "0" => Some(0),
                    "balanced" | "1" => Some(1),
                    "high" | "2" => Some(2),
                    _ => None,
                });
                if let Some(quality) = quality {
                    automation.raymarch_quality = quality;
                    index += 1;
                } else {
                    eprintln!("WARNING: --raymarch-quality expects low, balanced or high");
                    if next.is_some_and(|value| !value.starts_with("--")) {
                        index += 1;
                    }
                }
            }
            "--probe-extent" => {
                if let Some(next) = next {
                    index += 1;
                    if let Ok(value) = next.parse::<f32>() {
                        automation.probe_extent = value.max(1.0);
                    }
                }
            }
            "--probe-step" => {
                if let Some(next) = next {
                    index += 1;
                    if let Ok(value) = next.parse::<f32>() {
                        automation.probe_step = value.max(0.01);
                    }
                }
            }
            "--size" => {
                if let Some(next) = next {
                    index += 1;
                    // The C++ reads this with sscanf("%d,%d") (main.cpp:386),
                    // so only the comma form parses: `--size 1200x700` fails
                    // the scanf and falls back to 1600x900 with the warning
                    // below (whose text says "WxH" regardless). Accepting x/X
                    // here as well made one argument select a different window
                    // in each binary, so the two captures were not comparable.
                    let width_height = next
                        .split(',')
                        .filter_map(|part| part.trim().parse::<i32>().ok())
                        .collect::<Vec<_>>();
                    if width_height.len() == 2 && width_height[0] > 0 && width_height[1] > 0 {
                        automation.width = width_height[0];
                        automation.height = width_height[1];
                    } else {
                        automation.width = 1600;
                        automation.height = 900;
                        // The C++ prints this verbatim (main.cpp's sscanf on
                        // "%d,%d" falls through to the same default) and it
                        // is the reader's only clue that a malformed --size
                        // was ignored, so it must reach the terminal.
                        println!("WARNING: --size expects WxH, keeping 1600x900");
                    }
                }
            }
            _ => {}
        }
        index += 1;
    }
    // Screenshot warm-up duration depends on GPU speed. Hold celestial and
    // weather motion fixed so lighting is reproducible across machines.
    if automation.shot_path.is_some() && !advance_capture_time {
        automation.pause_time = true;
    }
    automation
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(arguments: &[&str]) -> AutomationSettings {
        parse_automation(arguments.iter().map(|value| (*value).to_string()))
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
    fn cloud_arguments_accept_extremes_and_overrides() {
        let settings = parse(&[
            "--no-clouds", "--cloud-coverage", "1", "--cloud-density", "4",
            "--cloud-base", "6000", "--cloud-thickness", "4000",
        ]);
        assert!(settings.no_clouds);
        assert_eq!(settings.cloud_coverage, 1.0);
        assert_eq!(settings.cloud_density, 4.0);
        assert_eq!(settings.cloud_base_height, 6000.0);
        assert_eq!(settings.cloud_thickness, 4000.0);
        let settings = parse(&[
            "--cloud-coverage", "0", "--cloud-density", "0",
            "--cloud-base", "100", "--cloud-thickness", "100",
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
}
