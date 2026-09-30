//! Command-line automation, ported from the C++ `AutomationSettings` and
//! `ParseAutomation`.

use bevy::prelude::Resource;
use crate::erosion::TileKey;

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
    pub overlap_tile: TileKey,
}

impl Default for AutomationSettings {
    fn default() -> Self {
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
    automation
}