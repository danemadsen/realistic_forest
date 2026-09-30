//! Continuous celestial motion shared by terrain, atmosphere and water.

use bevy::prelude::{Real, Res, ResMut, Resource, Time};

#[derive(Clone, Copy, Debug, Resource)]
pub struct DayNightCycle {
    pub time_hours: f32,
    pub paused: bool,
    /// Real minutes for a complete 24-hour cycle.
    pub day_length_minutes: f32,
}

impl Default for DayNightCycle {
    fn default() -> Self {
        Self {
            time_hours: 10.0,
            paused: false,
            day_length_minutes: 24.0,
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
        // defined. The modulo is periodic, including across midnight.
        let elapsed_hours = delta_seconds as f64 * 24.0 / (self.day_length_minutes as f64 * 60.0);
        self.time_hours = ((self.time_hours as f64 + elapsed_hours).rem_euclid(24.0)) as f32;
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
}

fn smoothstep(edge0: f32, edge1: f32, value: f32) -> f32 {
    let t = ((value - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

pub fn sample(time_hours: f32) -> CelestialLighting {
    let time_hours = if time_hours.is_finite() {
        time_hours.rem_euclid(24.0)
    } else {
        DayNightCycle::default().time_hours
    };
    // East (+X) at 06:00, highest at noon, west (-X) at 18:00.
    // Tilt the orbit so noon still casts readable directional shadows.
    let phase = (time_hours - 6.0) * std::f32::consts::TAU / 24.0;
    let (sin_phase, cos_phase) = phase.sin_cos();
    let elevation = sin_phase * 0.92;
    let sun_direction = [-cos_phase, -elevation, sin_phase * 0.39191836, 0.0];
    let moon_direction = [-sun_direction[0], -sun_direction[1], -sun_direction[2], 0.0];
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
        moon_intensity: 0.16 * smoothstep(0.0, 0.18, -elevation),
    }
}

pub fn advance_day_night(time: Res<Time<Real>>, mut cycle: ResMut<DayNightCycle>) {
    cycle.advance(time.delta_secs());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sun_rises_in_east_and_sets_in_west() {
        let sunrise = sample(6.0);
        let noon = sample(12.0);
        let sunset = sample(18.0);
        assert!(sunrise.sun_direction[0] < -0.999);
        assert!(sunset.sun_direction[0] > 0.999);
        assert!(sunrise.sun_direction[1].abs() < 1e-6);
        assert!(sunset.sun_direction[1].abs() < 1e-6);
        assert!(noon.sun_direction[1] < -0.9);
        assert_eq!(noon.sun_strength, 1.0);
        assert_eq!(noon.daylight, 1.0);
        assert_eq!(noon.moon_intensity, 0.0);
    }

    #[test]
    fn moon_lights_midnight_and_directions_are_unit_length() {
        let midnight = sample(0.0);
        assert_eq!(midnight.sun_strength, 0.0);
        assert_eq!(midnight.daylight, 0.0);
        assert!(midnight.moon_intensity > 0.0);
        assert!(midnight.moon_direction[1] < -0.9);
        for hour in 0..96 {
            let light = sample(hour as f32 * 0.25);
            let length_squared: f32 = light.sun_direction[..3].iter().map(|v| v * v).sum();
            assert!((length_squared - 1.0).abs() < 1e-5);
            for axis in 0..3 {
                assert_eq!(light.moon_direction[axis], -light.sun_direction[axis]);
            }
        }
    }

    #[test]
    fn twilight_and_midnight_are_continuous() {
        let dusk = sample(18.25);
        assert_eq!(dusk.sun_strength, 0.0);
        assert!(dusk.daylight > 0.0 && dusk.daylight < 0.5);
        let before = sample(23.999);
        let after = sample(0.001);
        for axis in 0..3 {
            assert!((before.sun_direction[axis] - after.sun_direction[axis]).abs() < 0.001);
        }
        assert_eq!(sample(24.0).sun_direction, sample(0.0).sun_direction);
    }

    #[test]
    fn cycle_uses_elapsed_time_wraps_and_pauses() {
        let mut cycle = DayNightCycle::default();
        cycle.advance(60.0);
        assert!((cycle.time_hours - 11.0).abs() < 1e-5);
        cycle.time_hours = 23.5;
        cycle.advance(60.0);
        assert!((cycle.time_hours - 0.5).abs() < 1e-5);
        cycle.advance(24.0 * 60.0);
        assert!((cycle.time_hours - 0.5).abs() < 1e-5);
        cycle.paused = true;
        cycle.advance(60.0);
        assert_eq!(cycle.time_hours, 0.5);
        cycle.paused = false;
        cycle.advance(f32::INFINITY);
        cycle.advance(-1.0);
        cycle.day_length_minutes = 0.0;
        cycle.advance(1.0);
        assert_eq!(cycle.time_hours, 0.5);
    }
}
