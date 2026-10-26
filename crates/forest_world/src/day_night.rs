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
#[path = "day_night_tests.rs"]
mod tests;
