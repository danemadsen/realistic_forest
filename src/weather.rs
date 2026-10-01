//! Continuous wind advection and smoothly changing large-scale weather.

use crate::automation::AutomationSettings;
use crate::constants::AppSettings;
use crate::player::Player;
use bevy::prelude::{Query, Real, Res, ResMut, Resource, Time};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum WeatherPreset {
    Clear,
    #[default]
    Cloudy,
    Overcast,
    Fog,
}

impl WeatherPreset {
    pub const ALL: [Self; 4] = [Self::Clear, Self::Cloudy, Self::Overcast, Self::Fog];

    pub const fn label(self) -> &'static str {
        match self {
            Self::Clear => "Clear",
            Self::Cloudy => "Cloudy",
            Self::Overcast => "Overcast",
            Self::Fog => "Fog / whiteout",
        }
    }

    pub const fn climate_bias(self) -> f32 {
        match self {
            Self::Clear => -0.9,
            Self::Cloudy => 0.0,
            Self::Overcast => 1.0,
            Self::Fog => 2.0,
        }
    }
}

/// Offsets relative to the existing cloudy settings. The cloudy preset is
/// exactly neutral, so cloud sliders and existing command-line captures keep
/// their previous meaning. These are meteorological tendencies rather than
/// complete replacements for the user's chosen cloud field.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WeatherProfile {
    pub coverage_delta: f32,
    pub density_multiplier: f32,
    pub base_offset: f32,
    pub thickness_multiplier: f32,
    pub fog_multiplier: f32,
    pub sun_multiplier: f32,
    /// 0 is the existing blue sky, 1 is fully diffuse gray sky.
    pub sky_overcast: f32,
    pub wind_multiplier: f32,
    pub shadow_multiplier: f32,
}

impl WeatherProfile {
    pub const fn for_preset(preset: WeatherPreset) -> Self {
        match preset {
            WeatherPreset::Clear => Self {
                coverage_delta: -0.43,
                density_multiplier: 0.72,
                base_offset: 150.0,
                thickness_multiplier: 0.8,
                fog_multiplier: 0.48,
                sun_multiplier: 1.08,
                sky_overcast: 0.0,
                wind_multiplier: 0.85,
                shadow_multiplier: 0.5,
            },
            WeatherPreset::Cloudy => Self {
                coverage_delta: 0.0,
                density_multiplier: 1.0,
                base_offset: 0.0,
                thickness_multiplier: 1.0,
                fog_multiplier: 1.0,
                sun_multiplier: 1.0,
                sky_overcast: 0.0,
                wind_multiplier: 1.0,
                shadow_multiplier: 1.0,
            },
            WeatherPreset::Overcast => Self {
                coverage_delta: 0.40,
                density_multiplier: 1.5,
                base_offset: -300.0,
                thickness_multiplier: 1.35,
                fog_multiplier: 2.4,
                sun_multiplier: 0.38,
                sky_overcast: 0.8,
                wind_multiplier: 1.25,
                shadow_multiplier: 0.22,
            },
            WeatherPreset::Fog => Self {
                coverage_delta: 0.38,
                density_multiplier: 1.25,
                base_offset: -450.0,
                thickness_multiplier: 1.45,
                // With the default 0.00012/m base, ground-level visibility
                // to 5% contrast is about 100 m. The shader's height term
                // softens the whiteout above the lower atmosphere.
                fog_multiplier: 250.0,
                sun_multiplier: 0.14,
                sky_overcast: 1.0,
                wind_multiplier: 0.45,
                shadow_multiplier: 0.05,
            },
        }
    }

    fn lerp(self, to: Self, t: f32) -> Self {
        let mix = |a: f32, b: f32| a + (b - a) * t;
        Self {
            coverage_delta: mix(self.coverage_delta, to.coverage_delta),
            density_multiplier: mix(self.density_multiplier, to.density_multiplier),
            base_offset: mix(self.base_offset, to.base_offset),
            thickness_multiplier: mix(self.thickness_multiplier, to.thickness_multiplier),
            fog_multiplier: mix(self.fog_multiplier, to.fog_multiplier),
            sun_multiplier: mix(self.sun_multiplier, to.sun_multiplier),
            sky_overcast: mix(self.sky_overcast, to.sky_overcast),
            wind_multiplier: mix(self.wind_multiplier, to.wind_multiplier),
            shadow_multiplier: mix(self.shadow_multiplier, to.shadow_multiplier),
        }
    }

    pub fn from_severity(severity: f32) -> Self {
        let severity = severity.clamp(0.0, 3.0);
        let lower = severity.floor() as usize;
        let upper = (lower + 1).min(3);
        Self::for_preset(WeatherPreset::ALL[lower]).lerp(
            Self::for_preset(WeatherPreset::ALL[upper]),
            severity - lower as f32,
        )
    }
}

/// Smooth world-space fronts, advected by the same offset as cloud detail.
/// The WGSL cloud and fog passes use this same small analytic field. Its wide
/// (8 km) scale allows different weather in neighbouring parts of one view.
pub fn front_severity(world_xz: [f32; 2], offset: [f32; 2], climate_bias: f32) -> f32 {
    let x = (world_xz[0] - offset[0]) / 8000.0;
    let z = (world_xz[1] - offset[1]) / 8000.0;
    let score = (0.523
        + 0.25 * (0.83 * x + 0.37 * z).sin()
        + 0.18 * (-0.49 * x + 0.91 * z + 0.3).sin()
        + 0.08 * (1.73 * x - 1.37 * z - 1.2).sin())
    .clamp(0.0, 1.0);
    let base = if score < 0.50 {
        ((score - 0.20) / 0.30).clamp(0.0, 1.0)
    } else if score < 0.70 {
        1.0 + (score - 0.50) / 0.20
    } else {
        2.0 + ((score - 0.70) / 0.18).clamp(0.0, 1.0)
    };
    (base + climate_bias).clamp(0.0, 3.0)
}

#[derive(Clone, Copy, Debug, Default)]
pub struct WeatherOverrides {
    pub coverage: bool,
    pub density: bool,
    pub base: bool,
    pub thickness: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct WeatherConditions {
    pub cloud_coverage: f32,
    pub cloud_density: f32,
    pub cloud_base_height: f32,
    pub cloud_thickness: f32,
    pub cloud_shadow_strength: f32,
    pub fog_density: f32,
    pub sun_intensity: f32,
    pub sky_overcast: f32,
    pub wind_speed: f32,
}

/// A spatial weather map sampled near the player for local sun/sky controls.
/// The render shaders sample the same map at each world point for cloud and fog
/// density; changing this state never changes every location at once.
#[derive(Clone, Copy, Debug, Resource)]
pub struct WeatherState {
    /// Whether wind advects the fronts. A frozen map is still spatial.
    pub automatic: bool,
    /// Broad climate bias; nearby places retain their own front conditions.
    pub target: WeatherPreset,
    pub current: WeatherProfile,
    pub transition_seconds: f32,
    pub overrides: WeatherOverrides,
    pub climate_bias: f32,
    pub local_severity: f32,
    source_bias: f32,
    transition_elapsed: f32,
    in_transition: bool,
}

impl Default for WeatherState {
    fn default() -> Self {
        Self::from_preset(WeatherPreset::Cloudy, true)
    }
}

impl WeatherState {
    pub fn from_preset(preset: WeatherPreset, automatic: bool) -> Self {
        let profile = WeatherProfile::for_preset(preset);
        Self {
            automatic,
            target: preset,
            current: profile,
            transition_seconds: 75.0,
            overrides: WeatherOverrides::default(),
            climate_bias: preset.climate_bias(),
            local_severity: preset as usize as f32,
            source_bias: preset.climate_bias(),
            transition_elapsed: 0.0,
            in_transition: false,
        }
    }

    pub fn set_target(&mut self, target: WeatherPreset) {
        if target == self.target {
            return;
        }
        self.source_bias = self.climate_bias;
        self.target = target;
        self.transition_elapsed = 0.0;
        self.in_transition = true;
    }

    pub fn is_transitioning(&self) -> bool {
        self.in_transition
    }

    pub fn advance(&mut self, delta_seconds: f32, world_xz: [f32; 2], offset: [f32; 2]) {
        if delta_seconds.is_finite() && delta_seconds > 0.0 && self.in_transition {
            let duration = self.transition_seconds.max(0.01);
            self.transition_elapsed = (self.transition_elapsed + delta_seconds).min(duration);
            let fraction = (self.transition_elapsed / duration).clamp(0.0, 1.0);
            let eased = fraction * fraction * (3.0 - 2.0 * fraction);
            self.climate_bias =
                self.source_bias + (self.target.climate_bias() - self.source_bias) * eased;
            if self.transition_elapsed >= duration {
                self.climate_bias = self.target.climate_bias();
                self.in_transition = false;
            }
        }
        self.local_severity = front_severity(world_xz, offset, self.climate_bias);
        self.current = WeatherProfile::from_severity(self.local_severity);
    }

    pub fn override_bits(&self) -> u32 {
        self.overrides.coverage as u32
            | (self.overrides.density as u32) << 1
            | (self.overrides.base as u32) << 2
            | (self.overrides.thickness as u32) << 3
    }

    pub fn local_condition(&self) -> WeatherPreset {
        let index = (self.local_severity + 0.5).floor().clamp(0.0, 3.0) as usize;
        WeatherPreset::ALL[index]
    }

    /// Approximate 5% contrast range through the current local air. The
    /// volumetric shader varies extinction along each ray, so this is a useful
    /// nearby readout rather than a guarantee of a distant sightline.
    pub fn visibility_metres(&self, settings: &AppSettings, altitude: f32) -> f32 {
        let base_density = self.conditions(settings).fog_density;
        if base_density <= 0.0 {
            return f32::INFINITY;
        }
        let height_density = (-(altitude - 20.0).max(0.0) / 450.0).exp();
        let extinction = base_density * (0.16 + 0.84 * height_density);
        2.995_732_3 / extinction
    }

    pub fn conditions(&self, settings: &AppSettings) -> WeatherConditions {
        let profile = self.current;
        let cloud_coverage = if self.overrides.coverage {
            settings.cloud_coverage
        } else {
            settings.cloud_coverage + profile.coverage_delta
        };
        let cloud_density = if self.overrides.density {
            settings.cloud_density
        } else {
            settings.cloud_density * profile.density_multiplier
        };
        let cloud_base_height = if self.overrides.base {
            settings.cloud_base_height
        } else {
            settings.cloud_base_height + profile.base_offset
        };
        let cloud_thickness = if self.overrides.thickness {
            settings.cloud_thickness
        } else {
            settings.cloud_thickness * profile.thickness_multiplier
        };
        WeatherConditions {
            cloud_coverage: cloud_coverage.clamp(0.0, 1.0),
            cloud_density: cloud_density.clamp(0.0, 4.0),
            cloud_base_height: cloud_base_height.max(100.0),
            cloud_thickness: cloud_thickness.max(100.0),
            cloud_shadow_strength: (settings.cloud_shadow_strength * profile.shadow_multiplier)
                .clamp(0.0, 1.0),
            // Zero remains zero for --no-fog and an explicitly disabled fog
            // slider, even during a fog/whiteout weather event.
            fog_density: settings.fog_density.max(0.0) * profile.fog_multiplier,
            sun_intensity: settings.sun_intensity.max(0.0) * profile.sun_multiplier,
            sky_overcast: profile.sky_overcast,
            wind_speed: settings.cloud_wind_speed.max(0.0) * profile.wind_multiplier,
        }
    }
}

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
    mut motion: ResMut<WeatherMotion>,
    mut weather: ResMut<WeatherState>,
    players: Query<&Player>,
) {
    // Automated captures freeze both weather and the celestial clock unless
    // --advance-time was requested. Live changes to time of day affect only
    // the lighting, so scrubbing the sun does not teleport cloud formations.
    let delta = if automation.pause_time {
        0.0
    } else {
        time.delta_secs()
    };
    if weather.automatic {
        motion.advance(
            delta,
            settings.cloud_wind_speed,
            settings.cloud_wind_direction_degrees,
        );
    }
    let world_xz = players
        .single()
        .map_or([0.0, 0.0], |player| [player.position.x, player.position.z]);
    weather.advance(delta, world_xz, motion.offset);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn presets_change_the_whole_environment_and_preserve_cloudy_baseline() {
        let settings = AppSettings::default();
        let cloudy = WeatherState::default().conditions(&settings);
        assert_eq!(cloudy.cloud_coverage, settings.cloud_coverage);
        assert_eq!(cloudy.cloud_density, settings.cloud_density);
        assert_eq!(cloudy.cloud_base_height, settings.cloud_base_height);
        assert_eq!(cloudy.cloud_thickness, settings.cloud_thickness);
        assert_eq!(cloudy.fog_density, settings.fog_density);
        assert_eq!(cloudy.sun_intensity, settings.sun_intensity);

        let clear = WeatherState::from_preset(WeatherPreset::Clear, false).conditions(&settings);
        let overcast =
            WeatherState::from_preset(WeatherPreset::Overcast, false).conditions(&settings);
        let fog = WeatherState::from_preset(WeatherPreset::Fog, false).conditions(&settings);
        assert!(clear.cloud_coverage < cloudy.cloud_coverage);
        assert!(clear.fog_density < cloudy.fog_density);
        assert!(overcast.cloud_coverage > cloudy.cloud_coverage);
        assert!(overcast.sun_intensity < cloudy.sun_intensity);
        assert!(overcast.cloud_shadow_strength < cloudy.cloud_shadow_strength);
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
        weather.advance(5.0, [0.0, 0.0], [0.0, 0.0]);
        assert!(weather.is_transitioning());
        assert!((weather.climate_bias - 0.5).abs() < 1e-5);
        let before_retarget = weather.climate_bias;
        weather.set_target(WeatherPreset::Clear);
        assert_eq!(weather.climate_bias, before_retarget);
        weather.advance(10.0, [0.0, 0.0], [0.0, 0.0]);
        assert_eq!(weather.climate_bias, WeatherPreset::Clear.climate_bias());
        assert_eq!(
            weather.current,
            WeatherProfile::from_severity(front_severity(
                [0.0, 0.0],
                [0.0, 0.0],
                weather.climate_bias,
            ))
        );
        assert!(!weather.is_transitioning());

        weather.overrides = WeatherOverrides {
            coverage: true,
            density: true,
            base: true,
            thickness: true,
        };
        let mut settings = AppSettings::default();
        settings.cloud_coverage = 0.11;
        settings.cloud_density = 0.4;
        settings.cloud_base_height = 2500.0;
        settings.cloud_thickness = 300.0;
        settings.fog_density = 0.0;
        let resolved = weather.conditions(&settings);
        assert_eq!(resolved.cloud_coverage, 0.11);
        assert_eq!(resolved.cloud_density, 0.4);
        assert_eq!(resolved.cloud_base_height, 2500.0);
        assert_eq!(resolved.cloud_thickness, 300.0);
        assert_eq!(resolved.fog_density, 0.0);
    }

    #[test]
    fn moving_fronts_preserve_simultaneous_weather_regions() {
        let mut weather = WeatherState::default();
        let mut minimum = 3.0f32;
        let mut maximum = 0.0f32;
        for z in -10..=10 {
            for x in -10..=10 {
                let severity =
                    front_severity([x as f32 * 4000.0, z as f32 * 4000.0], [0.0, 0.0], 0.0);
                minimum = minimum.min(severity);
                maximum = maximum.max(severity);
            }
        }
        assert!(minimum < 0.4 && maximum > 2.8);
        let location = [3000.0, -1200.0];
        weather.advance(0.0, location, [0.0, 0.0]);
        let first = weather.local_severity;
        weather.advance(1.0, location, [1500.0, 0.0]);
        assert_ne!(first, weather.local_severity);
        weather.advance(1.0, [location[0] + 0.1, location[1]], [1500.0, 0.0]);
        assert!(
            (weather.local_severity - front_severity(location, [1500.0, 0.0], 0.0)).abs() < 0.001
        );
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
}
