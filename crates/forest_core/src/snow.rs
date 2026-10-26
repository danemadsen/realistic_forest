//! Shared snow coverage rule used by terrain and vegetation.

use crate::constants::{SEA_LEVEL, SNOWLINE_ALTITUDE};
use bevy::math::Vec2;

fn smooth_hermite(start: f32, end: f32, value: f32) -> f32 {
    let t = ((value - start) / (end - start)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Half-width of the snowline band, metres.
pub const SNOW_BAND: f32 = 13.0;
/// Metres of snowline per unit of spring-sun exposure.
pub const SNOW_MELT_ASPECT: f32 = 12.0;
/// Scale of the concavity ring, metres: Gaea's coarse cell, under which fine
/// relief is restored in proportion to the pack's depth.
pub const SNOW_BURIAL: f32 = 12.0;
/// 2A/D in band depths (Gaea's adhered snow mass): a full pack holds to tan
/// 0.80, exactly where the uniform pack began to thin.
pub const SNOW_HOLD: f32 = 1.45 / 0.65;
/// Tan of the slip-off angle a vanishing pack still holds (55.4 degrees).
pub const SNOW_SLIP_TAN: f32 = 1.45;
/// Rise per metre toward the lee: the slope test is tilted toward the
/// prevailing wind, so lee faces hold steep snow that windward faces shed.
pub const SNOW_WIND_TILT: f32 = 0.15;
/// The prevailing wind blows toward 70 degrees (+X is 0, +Z is 90), the
/// default cloud wind, as a climate constant rather than the live weather.
pub const SNOW_WIND_TO: Vec2 = Vec2::new(0.34202, 0.93969);
/// Width of the smooth minimum between the pack and what its slope holds,
/// band depths.
pub const SNOW_SMOOTH: f32 = 0.2;
/// Below this height above the sea no slope, aspect or hollow holds snow.
pub const SNOW_FLOOR: f32 = SNOWLINE_ALTITUDE - SNOW_BAND - 0.62 * SNOW_MELT_ASPECT;

/// Keep this cover rule identical to snowCoverage in the terrain shaders
/// (snow_functions.wgslinc): the late-season pack as Gaea's snowfall settles
/// it, in closed form. `concavity` is the 12 m ring's mean height minus the
/// point (+ in hollows), as the erosion cache reads it.
/// Trail slopes do not affect the material's snow cover.
pub fn snow_coverage(base_height: f32, gradient: Vec2, concavity: f32) -> f32 {
    let normal = bevy::math::Vec3::new(-gradient.x, 1.0, -gradient.y).normalize();
    let melt_sun = bevy::math::Vec3::new(-0.22, 0.62, -0.76).normalize();
    let aspect = (0.62 - normal.dot(melt_sun).max(0.0)) * SNOW_MELT_ASPECT;
    let pack =
        (base_height - SEA_LEVEL + aspect - SNOWLINE_ALTITUDE + SNOW_BAND) / (2.0 * SNOW_BAND);
    let depth = pack * (1.0 + concavity / SNOW_BURIAL).clamp(0.5, 1.5);
    let held =
        SNOW_HOLD * (1.0 - (gradient + SNOW_WIND_TILT * SNOW_WIND_TO).length() / SNOW_SLIP_TAN);
    let blend = (SNOW_SMOOTH - (depth - held).abs()).max(0.0) / SNOW_SMOOTH;
    smooth_hermite(
        0.0,
        1.0,
        depth.min(held) - blend * blend * SNOW_SMOOTH * 0.25,
    )
}
