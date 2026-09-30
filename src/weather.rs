//! Continuous wind advection shared by cloud rendering and cloud shadows.

use crate::automation::AutomationSettings;
use crate::constants::AppSettings;
use bevy::prelude::{Real, Res, ResMut, Resource, Time};

#[derive(Clone, Copy, Debug, Default, Resource)]
pub struct WeatherMotion {
    /// World-space displacement in X/Z metres. Accumulating wind motion keeps
    /// clouds continuous when the wind changes or the celestial clock wraps.
    pub offset: [f32; 2],
}

impl WeatherMotion {
    pub fn advance(&mut self, delta_seconds: f32, speed: f32, direction_degrees: f32) {
        if !delta_seconds.is_finite()
            || delta_seconds <= 0.0
            || !speed.is_finite()
            || speed < 0.0
            || !direction_degrees.is_finite()
        {
            return;
        }
        let (sin, cos) = (direction_degrees as f64).to_radians().sin_cos();
        let distance = delta_seconds as f64 * speed as f64;
        let next = [
            (self.offset[0] as f64 + cos * distance) as f32,
            (self.offset[1] as f64 + sin * distance) as f32,
        ];
        if next.iter().all(|value| value.is_finite()) {
            self.offset = next;
        }
    }
}

pub fn advance_weather(
    time: Res<Time<Real>>,
    settings: Res<AppSettings>,
    automation: Res<AutomationSettings>,
    mut weather: ResMut<WeatherMotion>,
) {
    // Automated captures freeze both weather and the celestial clock unless
    // --advance-time was requested. Live changes to time of day affect only
    // the lighting, so scrubbing the sun does not teleport cloud formations.
    if !automation.pause_time {
        weather.advance(
            time.delta_secs(),
            settings.cloud_wind_speed,
            settings.cloud_wind_direction_degrees,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

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
            world.insert_resource(crate::automation::parse_automation(
                arguments.into_iter().map(str::to_string),
            ));
            world.init_resource::<WeatherMotion>();
            let mut schedule = Schedule::default();
            schedule.add_systems(advance_weather);
            schedule.run(&mut world);
            let offset = world.resource::<WeatherMotion>().offset;
            assert_eq!(offset != [0.0, 0.0], expected_motion);
        }
    }
}
