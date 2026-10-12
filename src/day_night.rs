//! The celestial clock and the light the sun and the moon give.
//!
//! The clock counts days and the hours of the day (local mean solar time at
//! spawn); the sun and the moon stand where their orbits put them then
//! (`crate::celestial`).

use crate::celestial::{SkyView, SOLAR_DAY};
use bevy::math::DVec3;
use bevy::prelude::{Real, Res, ResMut, Resource, Time};

#[derive(Clone, Copy, Debug, Resource)]
pub struct DayNightCycle {
    pub time_hours: f32,
    /// Days since the epoch, midsummer's day with a full moon.
    pub day: i64,
    pub paused: bool,
    /// Real minutes for a complete 24-hour cycle.
    pub day_length_minutes: f32,
    /// How many times faster than that the clock runs: the developer menu's
    /// time warp, for watching the moon's phases and the seasons go by.
    pub warp: f32,
}

impl Default for DayNightCycle {
    fn default() -> Self {
        Self {
            time_hours: 10.0,
            day: 0,
            paused: false,
            day_length_minutes: 24.0,
            warp: 1.0,
        }
    }
}

impl DayNightCycle {
    pub fn advance(&mut self, delta_seconds: f32) {
        if self.paused
            || !delta_seconds.is_finite()
            || delta_seconds <= 0.0
            || !self.day_length_minutes.is_finite()
            || self.day_length_minutes <= 0.0
        {
            return;
        }
        // f64 intermediate keeps long frames and very short cycles well
        // defined. Midnight starts the next day.
        let warp = if self.warp.is_finite() && self.warp > 0.0 { self.warp as f64 } else { 1.0 };
        let elapsed_hours = delta_seconds as f64 * warp * 24.0 / (self.day_length_minutes as f64 * 60.0);
        let hours = self.time_hours as f64 + elapsed_hours;
        self.day += hours.div_euclid(24.0) as i64;
        self.time_hours = (hours.rem_euclid(24.0) as f32).min(23.999_998);
    }

    /// Celestial seconds since the epoch.
    pub fn seconds(&self) -> f64 {
        (self.day as f64 + self.time_hours as f64 / 24.0) * SOLAR_DAY
    }

    /// Set the clock to `seconds` since the epoch.
    pub fn set_seconds(&mut self, seconds: f64) {
        if !seconds.is_finite() {
            return;
        }
        let days = seconds / SOLAR_DAY;
        self.day = days.floor() as i64;
        self.time_hours = ((days - days.floor()) * 24.0).clamp(0.0, 23.999_998) as f32;
    }
}

/// Directions point along the light's travel, from the celestial body toward
/// the world. Negate xyz to obtain the direction toward the visible disk.
#[derive(Clone, Copy, Debug)]
pub struct CelestialLighting {
    pub sun_direction: [f32; 4],
    /// Linear RGB, warmed by the longer atmospheric path near the horizon.
    pub sun_colour: [f32; 4],
    /// Multiplies the user-selected daytime sun intensity.
    pub sun_strength: f32,
    /// Ambient sky blend; twilight continues after direct sunlight vanishes.
    pub daylight: f32,
    pub moon_direction: [f32; 4],
    pub moon_intensity: f32,
    /// The sun's and the moon's angular radii, radians.
    pub sun_radius: f32,
    pub moon_radius: f32,
    /// The share of the sun's disc the moon leaves showing.
    pub sun_visible: f32,
    /// The share of sunlight that reaches the moon past the planet's shadow.
    pub moon_sunlight: f32,
}

/// Moonlight over the ground under a full moon high in the sky, relative to
/// the sun's intensity setting.
pub const FULL_MOONLIGHT: f32 = 0.16;

fn smoothstep(edge0: f32, edge1: f32, value: f32) -> f32 {
    let t = ((value - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The light of a sun `elevation` (the sine of its angle) above the local
/// horizon. The cycle's directions are the planet's, fixed in the spawn's
/// frame; elsewhere on the planet the sun stands higher or lower, and the
/// renderer relights the camera's place from its own elevation (the shaders'
/// `planetSunlightAt` and `planetDaylightAt` repeat these curves for every
/// other place).
pub fn at_elevation(sun_direction: [f32; 4], moon_direction: [f32; 4], elevation: f32) -> CelestialLighting {
    let daylight = smoothstep(-0.18, 0.18, elevation);
    let colour_mix = smoothstep(0.0, 0.55, elevation);
    CelestialLighting {
        sun_direction,
        sun_colour: [
            1.0,
            0.29 + 0.66 * colour_mix,
            0.095 + 0.765 * colour_mix,
            1.0,
        ],
        sun_strength: smoothstep(0.0, 0.12, elevation),
        daylight,
        moon_direction,
        moon_intensity: FULL_MOONLIGHT * smoothstep(0.0, 0.18, -elevation),
        sun_radius: 0.004_65,
        moon_radius: 0.004_52,
        sun_visible: 1.0,
        moon_sunlight: 1.0,
    }
}

/// The light of the sun and the moon as an eye sees them (`view`, planet
/// axes), in the camera's world axes (`to_world`): the sun's by its height
/// over the horizon and the share of its disc the moon leaves, the moon's by
/// its height, its phase and the planet's shadow on it.
pub fn lighting(view: &SkyView, to_world: impl Fn(DVec3) -> DVec3) -> CelestialLighting {
    let sun = to_world(view.to_sun);
    let moon = to_world(view.to_moon);
    let travel = |toward: DVec3| [-toward.x as f32, -toward.y as f32, -toward.z as f32, 0.0];
    let mut light = at_elevation(travel(sun), travel(moon), sun.y as f32);
    let visible = (1.0 - view.solar_eclipse as f32).clamp(0.0, 1.0);
    light.sun_strength *= visible;
    // Near totality the sky darkens to a twilight with stars.
    light.daylight *= visible.sqrt();
    light.sun_visible = visible;
    light.moon_intensity = FULL_MOONLIGHT
        * (view.moon_phase_brightness * view.moon_sunlight) as f32
        * smoothstep(0.0, 0.18, moon.y as f32);
    light.sun_radius = view.sun_radius as f32;
    light.moon_radius = view.moon_radius as f32;
    light.moon_sunlight = view.moon_sunlight as f32;
    light
}

pub fn advance_day_night(time: Res<Time<Real>>, mut cycle: ResMut<DayNightCycle>) {
    cycle.advance(time.delta_secs());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::celestial::Orbits;
    use crate::planet::Planet;

    /// The light over spawn at `day` and `hours`.
    fn light_at(day: i64, hours: f32) -> CelestialLighting {
        let planet = Planet::default();
        let cycle = DayNightCycle { day, time_hours: hours, ..DayNightCycle::default() };
        let view = Orbits::new(&planet).at(cycle.seconds()).seen_from(DVec3::new(0.0, planet.radius, 0.0));
        // Spawn's own axes are the planet's.
        lighting(&view, |v| v)
    }

    #[test]
    fn midsummer_noon_is_bright_and_its_midnight_is_moonlit() {
        let noon = light_at(0, 12.0);
        assert!(noon.sun_direction[1] < -0.9, "{:?}", noon.sun_direction);
        assert_eq!(noon.sun_strength, 1.0);
        assert_eq!(noon.daylight, 1.0);
        assert_eq!(noon.sun_visible, 1.0);
        let midnight = light_at(0, 0.0);
        assert_eq!(midnight.sun_strength, 0.0);
        assert_eq!(midnight.daylight, 0.0);
        // The epoch's full moon is up.
        assert!(midnight.moon_intensity > 0.5 * FULL_MOONLIGHT, "{}", midnight.moon_intensity);
        assert!(midnight.moon_direction[1] < -0.2);
        for step in 0..96 {
            let light = light_at(0, step as f32 * 0.25);
            for direction in [light.sun_direction, light.moon_direction] {
                let length_squared: f32 = direction[..3].iter().map(|v| v * v).sum();
                assert!((length_squared - 1.0).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn a_new_moon_gives_no_light() {
        let planet = Planet::default();
        let orbits = Orbits::new(&planet);
        // Half a synodic month on: new, whatever the hour.
        let day = (0.5 * orbits.synodic_month() / SOLAR_DAY).round() as i64;
        for hour in 0..24 {
            assert!(light_at(day, hour as f32).moon_intensity < 0.01 * FULL_MOONLIGHT);
        }
    }

    #[test]
    fn twilight_and_midnight_are_continuous() {
        let before = light_at(0, 23.999);
        let after = light_at(1, 0.0);
        for axis in 0..3 {
            assert!((before.sun_direction[axis] - after.sun_direction[axis]).abs() < 0.001);
            assert!((before.moon_direction[axis] - after.moon_direction[axis]).abs() < 0.001);
        }
        // Dusk at midsummer comes late at 45° south.
        assert!(light_at(0, 19.0).sun_strength > 0.0);
        let late = light_at(0, 21.0);
        assert!(late.sun_strength == 0.0 && late.daylight < 0.5);
    }

    #[test]
    fn cycle_uses_elapsed_time_counts_days_and_pauses() {
        let mut cycle = DayNightCycle::default();
        cycle.advance(60.0);
        assert!((cycle.time_hours - 11.0).abs() < 1e-5);
        cycle.time_hours = 23.5;
        cycle.advance(60.0);
        assert!((cycle.time_hours - 0.5).abs() < 1e-5);
        assert_eq!(cycle.day, 1);
        cycle.advance(24.0 * 60.0);
        assert!((cycle.time_hours - 0.5).abs() < 1e-5);
        assert_eq!(cycle.day, 2);
        // Warp runs it faster.
        cycle.warp = 10.0;
        cycle.advance(24.0 * 60.0);
        assert_eq!(cycle.day, 12);
        cycle.warp = 1.0;
        cycle.paused = true;
        cycle.advance(60.0);
        assert_eq!(cycle.time_hours, 0.5);
        cycle.paused = false;
        cycle.advance(f32::INFINITY);
        cycle.advance(-1.0);
        cycle.day_length_minutes = 0.0;
        cycle.advance(1.0);
        assert_eq!(cycle.time_hours, 0.5);
        assert!((cycle.seconds() - (12.0 + 0.5 / 24.0) * SOLAR_DAY).abs() < 1e-6);
        cycle.set_seconds(-0.25 * SOLAR_DAY);
        assert_eq!(cycle.day, -1);
        assert!((cycle.time_hours - 18.0).abs() < 1e-4);
        cycle.set_seconds(f64::NAN);
        assert_eq!(cycle.day, -1);
    }
}
