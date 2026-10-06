//! Continuous wind advection and smoothly changing large-scale weather.

use crate::automation::AutomationSettings;
use crate::constants::AppSettings;
use crate::lightning::{ActiveBolt, BoltKind};
use crate::player::Player;
use bevy::prelude::{Local, Query, Real, Res, ResMut, Resource, Time};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum WeatherPreset {
    Clear,
    #[default]
    Cloudy,
    Overcast,
    Fog,
    Rain,
    Snow,
    Thunderstorm,
}

impl WeatherPreset {
    pub const ALL: [Self; 7] = [
        Self::Clear, Self::Cloudy, Self::Overcast, Self::Fog,
        Self::Rain, Self::Snow, Self::Thunderstorm,
    ];
    pub const BASE: [Self; 4] = [Self::Clear, Self::Cloudy, Self::Overcast, Self::Fog];

    pub const fn label(self) -> &'static str {
        match self {
            Self::Clear => "Clear",
            Self::Cloudy => "Cloudy",
            Self::Overcast => "Overcast",
            Self::Fog => "Fog / whiteout",
            Self::Rain => "Rain",
            Self::Snow => "Snow",
            Self::Thunderstorm => "Thunderstorm",
        }
    }

    pub const fn climate_bias(self) -> f32 {
        match self {
            Self::Clear => -0.9,
            Self::Cloudy => 0.0,
            Self::Overcast => 1.0,
            Self::Fog => 2.0,
            Self::Rain | Self::Snow => 1.0,
            Self::Thunderstorm => 1.3,
        }
    }

    pub const fn precipitation_bias(self) -> f32 {
        match self {
            Self::Rain | Self::Snow => 0.25,
            Self::Thunderstorm => 0.37,
            _ => 0.0,
        }
    }

    /// 0 is temperature-driven phase; positive values force a demonstration
    /// trend while the precipitation amount remains spatial.
    pub const fn precipitation_override(self) -> u32 {
        match self {
            Self::Rain => 1,
            Self::Snow => 2,
            Self::Thunderstorm => 3,
            _ => 0,
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
            WeatherPreset::Overcast | WeatherPreset::Rain | WeatherPreset::Snow | WeatherPreset::Thunderstorm => Self {
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
        Self::for_preset(WeatherPreset::BASE[lower]).lerp(
            Self::for_preset(WeatherPreset::BASE[upper]),
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

pub fn storm_score(world_xz: [f32; 2], offset: [f32; 2]) -> f32 {
    let x = (world_xz[0] - offset[0]) / 5400.0;
    let z = (world_xz[1] - offset[1]) / 5400.0;
    (0.5
        + 0.28 * (1.07 * x + 0.31 * z + 0.53).sin()
        + 0.16 * (-0.61 * x + 1.27 * z - 1.1).sin()
        + 0.11 * (2.15 * x - 1.41 * z + 0.8).sin())
        .clamp(0.0, 1.0)
}

fn smoothstep(low: f32, high: f32, value: f32) -> f32 {
    let t = ((value - low) / (high - low)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The same bent, wind-aligned precipitation sheets as weatherRainBand in
/// precipitation-functions.wgslinc. Each sheet moves with WeatherMotion, so
/// heavy and light rain pass through a fixed location without following the
/// camera. Large storm cells continue to decide where rain is possible.
pub fn rain_band_multiplier(
    world_xz: [f32; 2],
    offset: [f32; 2],
    wind_direction_radians: f32,
) -> f32 {
    let (sin, cos) = wind_direction_radians.sin_cos();
    let p = [world_xz[0] - offset[0], world_xz[1] - offset[1]];
    let along = p[0] * cos + p[1] * sin;
    let across = -p[0] * sin + p[1] * cos;
    let bent = along
        + 135.0 * (across / 340.0 + along / 1450.0).sin()
        + 80.0 * (across / 125.0 - along / 870.0).sin();
    let broad = 0.5 + 0.5 * (bent / 150.0 + 1.2).sin();
    let fine = 0.5 + 0.5 * (bent / 51.0 + across / 190.0 + 0.6).sin();
    let sheet = smoothstep(0.28, 0.82, broad * 0.72 + fine * 0.28);
    // Lulls between squall lines remain steady rain rather than almost dry.
    0.45 + (1.16 - 0.45) * sheet
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PrecipitationSample {
    pub rain: f32,
    pub snow: f32,
    pub thunderstorm: f32,
    pub gust: f32,
}

impl PrecipitationSample {
    pub fn intensity(self) -> f32 {
        self.rain + self.snow
    }
}

/// The cloud base over a point: overcast and fog fronts lower the deck, as
/// cloudLocalLayer does in cloud-functions.wgslinc.
pub fn local_cloud_base(
    world_xz: [f32; 2],
    offset: [f32; 2],
    climate_bias: f32,
    cloud_base_height: f32,
    cloud_base_override: bool,
) -> f32 {
    let severity = front_severity(world_xz, offset, climate_bias);
    let base_offset = -300.0 * (severity - 1.0).clamp(0.0, 1.0)
        - 150.0 * (severity - 2.0).clamp(0.0, 1.0);
    (cloud_base_height + if cloud_base_override { 0.0 } else { base_offset }).max(100.0)
}

/// Matches weatherPrecipitation in precipitation-functions.wgslinc. Broad
/// overcast gates a narrower moving storm field; snow follows the low mountain
/// snowline unless a named showcase trend explicitly selects the phase.
pub fn sample_precipitation(
    world_xz: [f32; 2],
    altitude: f32,
    offset: [f32; 2],
    climate_bias: f32,
    precipitation_bias: f32,
    override_kind: u32,
    cloud_base_height: f32,
    cloud_base_override: bool,
    wind_direction_radians: f32,
) -> PrecipitationSample {
    let score = storm_score(world_xz, offset);
    let severity = front_severity(world_xz, offset, climate_bias);
    let wet_gate = smoothstep(1.25, 2.0, severity);
    let cloud_base = local_cloud_base(
        world_xz, offset, climate_bias, cloud_base_height, cloud_base_override,
    );
    let below_cloud = 1.0 - smoothstep(cloud_base + 100.0, cloud_base + 600.0, altitude);
    let storm_precipitation = smoothstep(0.74, 0.90, score + precipitation_bias)
        * wet_gate * below_cloud;
    let precipitation = (storm_precipitation
        * rain_band_multiplier(world_xz, offset, wind_direction_radians))
    .min(1.0);
    let cold_altitude = altitude
        + 70.0 * ((world_xz[0] - offset[0]) / 18000.0
            + (world_xz[1] - offset[1]) / 27000.0 + 0.7).sin();
    let snow_fraction = match override_kind {
        1 | 3 | 4 => 0.0,
        2 => 1.0,
        _ => smoothstep(80.0, 260.0, cold_altitude),
    };
    let rain = precipitation * (1.0 - snow_fraction);
    let snow = precipitation * snow_fraction;
    let convective_bias = if override_kind == 3 { precipitation_bias } else { 0.0 };
    let thunderstorm = if override_kind == 4 {
        0.0
    } else {
        storm_precipitation * (1.0 - snow_fraction)
            * smoothstep(0.91, 0.98, score + convective_bias)
    };
    let gust = (0.18 * precipitation + 0.75 * thunderstorm).clamp(0.0, 1.0);
    PrecipitationSample { rain, snow, thunderstorm, gust }
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
    pub rain_intensity: f32,
    pub snow_intensity: f32,
    pub thunder_intensity: f32,
    pub gust_strength: f32,
}

/// Each return stroke is microseconds long, but the eye and a camera hold its
/// glare for a few hundredths of a second.
const STROKE_DECAY_SECONDS: f32 = 0.022;
/// Continuing current keeps some strokes glowing for longer.
const CONTINUING_DECAY_SECONDS: f32 = 0.14;
/// The ionised channel fades out after the last stroke.
const AFTERGLOW_SECONDS: f32 = 0.25;
const MAX_STROKES: usize = 6;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct ReturnStroke {
    time: f32,
    peak: f32,
    continuing: f32,
}

/// One frame of a discharge: scene light and the channel's own luminance.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LightningPulse {
    /// Scene illumination relative to the first return stroke.
    pub flash: f32,
    /// Main-channel luminance relative to a return stroke.
    pub channel: f32,
    /// Branch luminance; branches burn only with the first stroke, since
    /// later strokes follow the main channel the first one opened.
    pub branches: f32,
    /// How far the leader has progressed, 0 at the cloud and 1 at contact.
    pub leader: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct LightningEvent {
    /// Bolt's contact point, resolved to eroded terrain or sea level on spawn.
    /// For a cloud discharge it is the point under the flash.
    pub position: [f32; 3],
    /// Linear HDR radiance of the flash. Peaks near 1: about a dim overcast
    /// day, so a strike flickers over a daytime storm and floods the night.
    pub flash: f32,
    /// Negative seeds denote a cloud discharge without a ground bolt.
    pub seed: f32,
    pub age_seconds: f32,
    /// Altitude of the channel's top, inside the cloud above the strike.
    pub top_height: f32,
    /// The cloud base over the discharge, which hides a channel above it.
    pub cloud_base: f32,
    pub kind: BoltKind,
    /// This frame's channel luminance, see [`LightningPulse`].
    pub channel: f32,
    pub branches: f32,
    pub leader: f32,
    peak_flash: f32,
    /// A ground bolt's stepped leader, or a crawler's spread across the base.
    leader_seconds: f32,
    strokes: [ReturnStroke; MAX_STROKES],
    stroke_count: usize,
}

impl Default for LightningEvent {
    fn default() -> Self {
        Self {
            position: [0.0; 3],
            flash: 0.0,
            seed: 0.0,
            age_seconds: 1000.0,
            top_height: 0.0,
            cloud_base: 0.0,
            kind: BoltKind::Hidden,
            channel: 0.0,
            branches: 0.0,
            leader: 0.0,
            peak_flash: 0.0,
            leader_seconds: 0.07,
            strokes: [ReturnStroke::default(); MAX_STROKES],
            stroke_count: 0,
        }
    }
}

impl LightningEvent {
    /// When the first return stroke fires, after the leader's descent.
    pub fn first_stroke_seconds(&self) -> f32 {
        if self.stroke_count == 0 { 0.0 } else { self.strokes[0].time }
    }

    /// When nothing of the discharge remains visible.
    #[cfg(test)]
    pub fn duration_seconds(&self) -> f32 {
        let last = if self.stroke_count == 0 { 0.0 } else { self.strokes[self.stroke_count - 1].time };
        last.max(self.leader_seconds) + 1.2
    }

    /// The light of the discharge `age` seconds after it began. A ground bolt
    /// shows a dim stepped leader feeling its way down, then a brilliant
    /// return stroke and two to four restrikes down the same channel, some
    /// held by continuing current, before the channel fades. A crawler
    /// flickers while it spreads; a hidden discharge only lights the cloud.
    pub fn pulse(&self, age: f32) -> LightningPulse {
        if !age.is_finite() || age < 0.0 || self.stroke_count == 0 {
            return LightningPulse::default();
        }
        let mut strokes = 0.0;
        let mut first = 0.0;
        for (index, stroke) in self.strokes[..self.stroke_count].iter().enumerate() {
            if age < stroke.time {
                break;
            }
            let since = age - stroke.time;
            let light = stroke.peak * (-since / STROKE_DECAY_SECONDS).exp()
                + stroke.continuing * (-since / CONTINUING_DECAY_SECONDS).exp();
            strokes += light;
            if index == 0 {
                first = stroke.peak * (-since / (STROKE_DECAY_SECONDS * 1.4)).exp();
            }
        }
        let leader = (age / self.leader_seconds.max(0.001)).clamp(0.0, 1.0);
        let mut reversed_strokes = self.strokes[..self.stroke_count].iter().rev();
        let last = reversed_strokes
            .find(|stroke| stroke.time <= age)
            .map(|stroke| stroke.time);
        let afterglow = last.map_or(0.0, |time| 0.08 * (-(age - time) / AFTERGLOW_SECONDS).exp());
        match self.kind {
            BoltKind::Hidden => LightningPulse { flash: strokes, ..LightningPulse::default() },
            BoltKind::Ground => {
                // The stepped leader is faint beside a return stroke, yet a
                // camera still catches it branching down in the dark.
                let stepping = if age < self.first_stroke_seconds() { 0.1 } else { 0.0 };
                LightningPulse {
                    flash: strokes + 0.03 * stepping,
                    channel: strokes + afterglow + stepping,
                    branches: first + stepping + 0.5 * afterglow,
                    leader,
                }
            }
            BoltKind::Crawler => LightningPulse {
                flash: strokes,
                channel: strokes + afterglow,
                branches: strokes + afterglow,
                leader,
            },
        }
    }

    fn apply_pulse(&mut self) {
        let pulse = self.pulse(self.age_seconds);
        self.flash = self.peak_flash * pulse.flash;
        self.channel = pulse.channel;
        self.branches = pulse.branches;
        self.leader = pulse.leader;
    }
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
    /// A trainer selection follows the player instead of waiting for a front.
    pub trainer_preset: Option<WeatherPreset>,
    pub current: WeatherProfile,
    pub transition_seconds: f32,
    pub overrides: WeatherOverrides,
    pub climate_bias: f32,
    pub precipitation_bias: f32,
    pub local_severity: f32,
    pub local_precipitation: PrecipitationSample,
    pub elapsed_seconds: f32,
    pub lightning: LightningEvent,
    /// Set for one update after a strike is scheduled, so terrain height can
    /// be resolved with the erosion cache outside this resource's pure model.
    pub new_strike: bool,
    source_bias: f32,
    source_precipitation_bias: f32,
    transition_elapsed: f32,
    in_transition: bool,
    lightning_wait: f32,
    random_state: u64,
}

impl Default for WeatherState {
    fn default() -> Self {
        Self::from_preset(WeatherPreset::Cloudy, true)
    }
}

impl WeatherState {
    pub fn from_preset(preset: WeatherPreset, automatic: bool) -> Self {
        let profile = WeatherProfile::for_preset(preset);
        let settings = AppSettings::default();
        let local_precipitation = sample_precipitation(
            [0.0, 0.0], 0.0, [0.0, 0.0], preset.climate_bias(),
            preset.precipitation_bias(), preset.precipitation_override(),
            settings.cloud_base_height, false,
            settings.cloud_wind_direction_degrees.to_radians(),
        );
        Self {
            automatic,
            target: preset,
            trainer_preset: None,
            current: profile,
            transition_seconds: 75.0,
            overrides: WeatherOverrides::default(),
            climate_bias: preset.climate_bias(),
            precipitation_bias: preset.precipitation_bias(),
            local_severity: preset as usize as f32,
            local_precipitation,
            elapsed_seconds: 0.0,
            lightning: LightningEvent::default(),
            new_strike: false,
            source_bias: preset.climate_bias(),
            source_precipitation_bias: preset.precipitation_bias(),
            transition_elapsed: 0.0,
            in_transition: false,
            lightning_wait: 4.0,
            random_state: 0xD6B5_EA92_5C31_0847,
        }
    }

    pub fn set_target(&mut self, target: WeatherPreset) {
        let had_trainer_preset = self.trainer_preset.take().is_some();
        if target == self.target && !had_trainer_preset {
            return;
        }
        self.source_bias = self.climate_bias;
        self.source_precipitation_bias = self.precipitation_bias;
        self.target = target;
        self.transition_elapsed = 0.0;
        self.in_transition = true;
    }

    /// Selects weather at the player's location without the trend transition.
    /// Call `refresh_local` with the current player pose and weather offset to
    /// apply it in the same frame. Clearing the selection restores the natural
    /// Cloudy trend immediately; front motion keeps its previous setting.
    pub fn set_trainer_preset(&mut self, preset: Option<WeatherPreset>) {
        if self.trainer_preset == preset {
            return;
        }
        self.trainer_preset = preset;
        self.in_transition = false;
        self.transition_elapsed = 0.0;
        self.target = preset.unwrap_or(WeatherPreset::Cloudy);
        if preset.is_some_and(|selected| selected != WeatherPreset::Thunderstorm) {
            // A previously spawned storm bolt must not continue flashing after
            // the trainer has selected dry weather, rain, or snow.
            self.lightning = LightningEvent::default();
            self.new_strike = false;
        }
        if preset.is_none() {
            self.climate_bias = WeatherPreset::Cloudy.climate_bias();
            self.precipitation_bias = WeatherPreset::Cloudy.precipitation_bias();
            self.source_bias = self.climate_bias;
            self.source_precipitation_bias = self.precipitation_bias;
        }
    }

    pub fn is_transitioning(&self) -> bool {
        self.in_transition
    }

    /// Kind 4 is trainer rain: force liquid precipitation without thunder.
    /// The render-world uniform must use this method rather than `target`'s
    /// ordinary override so the shader receives the same kind as the CPU.
    pub fn precipitation_override(&self) -> u32 {
        if self.trainer_preset == Some(WeatherPreset::Rain) {
            4
        } else {
            self.target.precipitation_override()
        }
    }

    pub fn advance(
        &mut self,
        delta_seconds: f32,
        world_position: [f32; 3],
        offset: [f32; 2],
        settings: &AppSettings,
    ) {
        if self.trainer_preset.is_none()
            && delta_seconds.is_finite()
            && delta_seconds > 0.0
            && self.in_transition
        {
            let duration = self.transition_seconds.max(0.01);
            self.transition_elapsed = (self.transition_elapsed + delta_seconds).min(duration);
            let fraction = (self.transition_elapsed / duration).clamp(0.0, 1.0);
            let eased = fraction * fraction * (3.0 - 2.0 * fraction);
            self.climate_bias =
                self.source_bias + (self.target.climate_bias() - self.source_bias) * eased;
            self.precipitation_bias = self.source_precipitation_bias
                + (self.target.precipitation_bias() - self.source_precipitation_bias) * eased;
            if self.transition_elapsed >= duration {
                self.climate_bias = self.target.climate_bias();
                self.precipitation_bias = self.target.precipitation_bias();
                self.in_transition = false;
            }
        }
        self.refresh_local(world_position, offset, settings);
        if delta_seconds.is_finite() && delta_seconds > 0.0 {
            self.elapsed_seconds += delta_seconds;
            self.advance_lightning(delta_seconds, world_position, offset, settings);
        } else {
            self.new_strike = false;
        }
    }

    /// Recomputes local conditions without changing the clock or lightning.
    /// The render shaders use these same climate and precipitation biases.
    pub fn refresh_local(
        &mut self,
        world_position: [f32; 3],
        offset: [f32; 2],
        settings: &AppSettings,
    ) {
        let world_xz = [world_position[0], world_position[2]];
        if let Some(preset) = self.trainer_preset {
            let severity = match preset {
                WeatherPreset::Clear => 0.0,
                WeatherPreset::Cloudy => 1.0,
                WeatherPreset::Overcast => 2.0,
                WeatherPreset::Fog => 3.0,
                // Severity 2 has a fully open wet gate and the overcast cloud
                // profile; values above 2 also blend toward whiteout fog.
                WeatherPreset::Rain | WeatherPreset::Snow | WeatherPreset::Thunderstorm => 2.0,
            };
            self.climate_bias = severity - front_severity(world_xz, offset, 0.0);
            self.precipitation_bias = match preset {
                WeatherPreset::Rain | WeatherPreset::Snow | WeatherPreset::Thunderstorm => {
                    1.0 - storm_score(world_xz, offset)
                }
                _ => -1.0,
            };
        }
        self.local_severity = front_severity(world_xz, offset, self.climate_bias);
        self.current = WeatherProfile::from_severity(self.local_severity);
        self.local_precipitation = sample_precipitation(
            world_xz,
            world_position[1],
            offset,
            self.climate_bias,
            self.precipitation_bias,
            self.precipitation_override(),
            settings.cloud_base_height,
            self.overrides.base,
            settings.cloud_wind_direction_degrees.to_radians(),
        );
    }

    pub fn override_bits(&self) -> u32 {
        self.overrides.coverage as u32
            | (self.overrides.density as u32) << 1
            | (self.overrides.base as u32) << 2
            | (self.overrides.thickness as u32) << 3
    }

    pub fn local_condition(&self) -> WeatherPreset {
        if let Some(preset) = self.trainer_preset {
            return preset;
        }
        if self.local_precipitation.thunderstorm > 0.25 {
            return WeatherPreset::Thunderstorm;
        }
        if self.local_precipitation.snow > 0.35 {
            return WeatherPreset::Snow;
        }
        if self.local_precipitation.rain > 0.35 {
            return WeatherPreset::Rain;
        }
        let index = (self.local_severity + 0.5).floor().clamp(0.0, 3.0) as usize;
        WeatherPreset::BASE[index]
    }

    /// Approximate 5% contrast range through the current local air. The
    /// volumetric shader varies extinction along each ray, so this is a useful
    /// nearby readout rather than a guarantee of a distant sightline.
    pub fn visibility_metres(&self, settings: &AppSettings, altitude: f32) -> f32 {
        let base_density = self.conditions(settings).fog_density;
        let precipitation = self.local_precipitation;
        if base_density <= 0.0 && precipitation.intensity() <= 0.0 {
            return f32::INFINITY;
        }
        let height_density = (-(altitude - 20.0).max(0.0) / 450.0).exp();
        let extinction = base_density * (0.16 + 0.84 * height_density)
            + precipitation.rain * 0.0021 + precipitation.snow * 0.0013
            + precipitation.thunderstorm * 0.00035;
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
            rain_intensity: self.local_precipitation.rain,
            snow_intensity: self.local_precipitation.snow,
            thunder_intensity: self.local_precipitation.thunderstorm,
            gust_strength: self.local_precipitation.gust,
        }
    }

    fn random(&mut self) -> f32 {
        // A local, reproducible stream keeps strikes independent of frame rate
        // and avoids adding a global random resource to the render schedule.
        self.random_state ^= self.random_state << 13;
        self.random_state ^= self.random_state >> 7;
        self.random_state ^= self.random_state << 17;
        (self.random_state as u32) as f32 / u32::MAX as f32
    }

    fn advance_lightning(&mut self, delta: f32, player: [f32; 3], offset: [f32; 2], settings: &AppSettings) {
        self.new_strike = false;
        self.lightning.age_seconds += delta;
        // The cloud, fog, terrain and water shaders receive this same
        // radiance each frame; the lightning pass draws the channel.
        self.lightning.apply_pulse();
        self.lightning_wait -= delta;
        if self.lightning_wait > 0.0 { return; }

        let local = self.local_precipitation.thunderstorm;
        if local < 0.12 {
            self.lightning_wait = 2.0;
            return;
        }
        // Storm cells determine where a bolt may appear. Scan several nearby
        // candidates so lightning stays under rain-bearing cloud, while the
        // local hazard controls how frequently a player experiences it.
        for _ in 0..10 {
            let angle = self.random() * std::f32::consts::TAU;
            // Most strikes land a few kilometres away; now and then one
            // comes down close enough to shake the ground.
            let radius = if self.random() < 0.1 {
                120.0 + 280.0 * self.random()
            } else {
                250.0 + self.random().sqrt() * 2500.0
            };
            let x = player[0] + radius * angle.cos();
            let z = player[2] + radius * angle.sin();
            let cell = sample_precipitation(
                [x, z], player[1], offset, self.climate_bias,
                self.precipitation_bias, self.precipitation_override(),
                settings.cloud_base_height, self.overrides.base,
                settings.cloud_wind_direction_degrees.to_radians(),
            );
            if cell.thunderstorm < 0.18 { continue; }
            // Many discharges stay in the cloud. Some only brighten the deck
            // and haze; others crawl visibly along its base.
            let roll = self.random();
            let kind = if roll < 0.2 {
                BoltKind::Hidden
            } else if roll < 0.4 {
                BoltKind::Crawler
            } else {
                BoltKind::Ground
            };
            self.spawn_strike([x, z], kind, offset, settings);
            break;
        }
        // A few seconds between discharges in a strong cell and longer in a
        // weaker one, with occasional rapid bursts. Weather outside the cell
        // produces no strikes nearby.
        self.lightning_wait = if self.random() < 0.25 {
            0.6 + 1.4 * self.random()
        } else {
            (2.5 + 12.0 * (1.0 - local) + 8.0 * self.random()).max(2.0)
        };
    }

    /// Begins a discharge over `world_xz` with its stroke sequence. The
    /// contact height is resolved afterwards, while `new_strike` is set.
    fn spawn_strike(&mut self, world_xz: [f32; 2], kind: BoltKind, offset: [f32; 2], settings: &AppSettings) {
        let cloud_base = local_cloud_base(
            world_xz, offset, self.climate_bias, settings.cloud_base_height, self.overrides.base,
        );
        let ground = kind == BoltKind::Ground;
        let seed = self.random().max(0.0001) * if ground { 1.0 } else { -1.0 };
        let mut event = LightningEvent {
            position: [world_xz[0], 0.0, world_xz[1]],
            seed,
            age_seconds: 0.0,
            // A ground channel starts a few hundred metres inside the cloud;
            // a cloud discharge lights the deck from just above its base.
            top_height: cloud_base + if ground { 250.0 + 200.0 * self.random() } else { 400.0 },
            cloud_base,
            kind,
            peak_flash: if ground {
                0.7 + 0.9 * self.random()
            } else {
                0.35 + 0.65 * self.random()
            },
            ..LightningEvent::default()
        };
        let count = match kind {
            BoltKind::Ground => match self.random() {
                roll if roll < 0.25 => 2,
                roll if roll < 0.6 => 3,
                roll if roll < 0.85 => 4,
                _ => 5,
            },
            BoltKind::Crawler => 3 + (self.random() * 3.99) as usize,
            BoltKind::Hidden => 2 + (self.random() * 2.99) as usize,
        };
        // A stepped leader takes tens of milliseconds to reach the ground;
        // a crawler spreads across the base for a few tenths of a second.
        event.leader_seconds = match kind {
            BoltKind::Ground => 0.05 + 0.04 * self.random(),
            BoltKind::Crawler => 0.25 + 0.3 * self.random(),
            BoltKind::Hidden => 0.05,
        };
        let mut time = match kind {
            BoltKind::Ground => event.leader_seconds,
            BoltKind::Crawler => 0.02 + 0.05 * self.random(),
            BoltKind::Hidden => 0.0,
        };
        for index in 0..count.min(MAX_STROKES) {
            if index > 0 {
                // Restrikes follow 40-120 ms apart down a ground channel.
                time += match kind {
                    BoltKind::Ground => 0.04 + 0.08 * self.random(),
                    _ => 0.05 + 0.12 * self.random(),
                };
            }
            let peak = if index == 0 { 1.0 } else { 0.35 + 0.45 * self.random() };
            let continuing = if self.random() < 0.35 { 0.12 + 0.18 * self.random() } else { 0.0 };
            event.strokes[index] = ReturnStroke { time, peak, continuing };
        }
        event.stroke_count = count.min(MAX_STROKES);
        event.apply_pulse();
        self.lightning = event;
        self.new_strike = true;
    }

    /// Fires a discharge at once over `world_xz` and holds it `age` seconds
    /// after its first return stroke (negative during the leader), so a
    /// frozen capture can show any moment of a strike.
    pub fn force_strike(
        &mut self,
        world_xz: [f32; 2],
        kind: BoltKind,
        age: f32,
        offset: [f32; 2],
        settings: &AppSettings,
    ) {
        self.spawn_strike(world_xz, kind, offset, settings);
        self.lightning.age_seconds = (self.lightning.first_stroke_seconds() + age).max(0.0);
        self.lightning.apply_pulse();
    }

    /// Lets the next update discharge if the local storm can produce one.
    pub fn request_strike(&mut self) {
        self.lightning_wait = 0.0;
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
    erosion: Option<Res<crate::erosion::ErosionCache>>,
    noise: Option<Res<crate::noise::NoiseField>>,
    mut bolt: ResMut<ActiveBolt>,
    mut forced_strike_fired: Local<bool>,
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
    let (world_position, yaw) = players.single().map_or(([0.0; 3], 0.0), |player| {
        ([player.position.x, player.position.y, player.position.z], player.yaw)
    });
    weather.advance(delta, world_position, motion.offset, &settings);
    if let Some(strike) = automation.lightning.filter(|_| !*forced_strike_fired) {
        *forced_strike_fired = true;
        // The bearing is measured clockwise from the camera's heading.
        let heading = yaw + strike.bearing_degrees.to_radians();
        let world_xz = [
            world_position[0] + heading.sin() * strike.distance,
            world_position[2] - heading.cos() * strike.distance,
        ];
        weather.force_strike(world_xz, strike.kind, strike.age_seconds, motion.offset, &settings);
    }
    if weather.new_strike {
        let [x, _, z] = weather.lightning.position;
        let terrain = if let (Some(cache), Some(field)) = (erosion, noise) {
            crate::erosion::sample_eroded_height(
                &cache, &field, x, z, [world_position[0], world_position[2]],
            )
        } else { crate::constants::SEA_LEVEL };
        let contact = terrain.max(crate::constants::SEA_LEVEL);
        let lightning = &mut weather.lightning;
        lightning.position[1] = contact;
        // A storm over a mountain still discharges from above its summit.
        lightning.top_height = lightning.top_height.max(contact + 300.0);
        lightning.cloud_base = lightning.cloud_base.max(contact + 150.0);
        let segments = match lightning.kind {
            BoltKind::Ground => {
                // Channels slant: the top sits a little way from the contact.
                let seed = lightning.seed;
                let angle = seed * 40.0;
                let reach = 100.0 + 400.0 * (seed * 13.7).fract();
                let top = [x + angle.cos() * reach, lightning.top_height, z + angle.sin() * reach];
                crate::lightning::ground_bolt(seed, lightning.position, top)
            }
            BoltKind::Crawler => crate::lightning::crawler_bolt(
                -lightning.seed, lightning.position, lightning.cloud_base,
            ),
            BoltKind::Hidden => Vec::new(),
        };
        let kind = lightning.kind;
        bolt.replace(kind, segments);
    }
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
                let severity =
                    front_severity([x as f32 * 4000.0, z as f32 * 4000.0], [0.0, 0.0], 0.0);
                minimum = minimum.min(severity);
                maximum = maximum.max(severity);
            }
        }
        assert!(minimum < 0.4 && maximum > 2.8);
        let location = [3000.0, -1200.0];
        weather.advance(0.0, [location[0], 0.0, location[1]], [0.0, 0.0], &AppSettings::default());
        let first = weather.local_severity;
        weather.advance(1.0, [location[0], 0.0, location[1]], [1500.0, 0.0], &AppSettings::default());
        assert_ne!(first, weather.local_severity);
        weather.advance(1.0, [location[0] + 0.1, 0.0, location[1]], [1500.0, 0.0], &AppSettings::default());
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
                weather.advance(0.0, [position[0], 500.0, position[2]], [0.0, 0.0], &AppSettings::default());
                highland[weather.local_condition() as usize] += 1;
            }
        }
        let count = |preset: WeatherPreset, values: [usize; 7]| values[preset as usize];
        for common in [WeatherPreset::Clear, WeatherPreset::Cloudy, WeatherPreset::Overcast] {
            assert!(count(common, sea_level) > count(WeatherPreset::Fog, sea_level));
        }
        assert!(count(WeatherPreset::Fog, sea_level) > count(WeatherPreset::Rain, sea_level));
        assert!(count(WeatherPreset::Fog, highland) > count(WeatherPreset::Snow, highland));
        assert!(count(WeatherPreset::Rain, sea_level)
                > count(WeatherPreset::Thunderstorm, sea_level));
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
            let offset = [direction[0] * second as f32 * 18.0,
                          direction[1] * second as f32 * 18.0];
            let rain = sample_precipitation(
                location, 0.0, offset, 2.0, 1.0, 4, 1300.0,
                false, angle,
            ).rain;
            minimum = minimum.min(rain);
            maximum = maximum.max(rain);
        }
        // A lull is visibly lighter than a squall line, yet still steady rain.
        assert!(minimum > 0.40 && minimum < 0.55, "rain lull should stay wet: {minimum}");
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
        assert!(storm.lightning.flash > 0.65, "first stroke: {}", storm.lightning.flash);
        let duration = storm.lightning.duration_seconds();
        assert!(duration < 2.5);
        storm.advance(duration, [0.0; 3], [0.0; 2], &AppSettings::default());
        assert!(storm.lightning.flash < 0.001, "after: {}", storm.lightning.flash);
        assert!(storm.lightning.channel < 0.01);
    }

    #[test]
    fn ground_bolts_restrike_down_the_main_channel_only() {
        let settings = AppSettings::default();
        let mut storm = WeatherState::from_preset(WeatherPreset::Thunderstorm, false);
        for strike in 0..40 {
            storm.force_strike([900.0, 0.0], BoltKind::Ground, -1.0, [0.0; 2], &settings);
            let event = storm.lightning;
            assert!((2..=5).contains(&event.stroke_count), "strokes {}", event.stroke_count);
            let leader = event.pulse(event.first_stroke_seconds() * 0.5);
            assert!(leader.leader > 0.3 && leader.leader < 0.7);
            assert!(leader.channel < 0.2 && leader.flash < 0.05, "a leader is faint");
            let first = event.pulse(event.first_stroke_seconds());
            assert_eq!(first.leader, 1.0);
            assert!(first.channel > 0.95 && first.branches > 0.95, "strike {strike}");
            // Each restrike relights the channel while the branches stay dark.
            for stroke in &event.strokes[1..event.stroke_count] {
                let restrike = event.pulse(stroke.time);
                assert!(restrike.channel > stroke.peak, "restrike {restrike:?}");
                assert!(restrike.branches < 0.4 * restrike.channel, "branches {restrike:?}");
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
                if let Some(previous) = previous {
                    if seconds - previous < 2.1 {
                        quick += 1;
                    }
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
            assert_eq!(storm.lightning.seed < 0.0, storm.lightning.kind != BoltKind::Ground);
            match storm.lightning.kind {
                BoltKind::Hidden => cloud_flashes += 1,
                BoltKind::Crawler => crawlers += 1,
                BoltKind::Ground => ground_bolts += 1,
            }
            assert!(storm.lightning.top_height > storm.lightning.cloud_base);
        }
        assert!(cloud_flashes > 0 && crawlers > 0 && ground_bolts > 0);
    }
}
