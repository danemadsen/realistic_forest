//! Water body optics: extinction, scatter and the subsurface tint.
//!
//! Ported from bevy-aqua's `bevy-aqua-core/src/lib.rs` `WaterOptics` and its
//! presets. See `src/water/ATTRIBUTION.md`.
//!
//! Aqua feeds these to the shader through exactly four `vec4`s
//! (`fog_density.rgb` / `.w`, `medium_scatter.rgb` / `.w`), which is the
//! contract kept here — the presets are Rust-side authoring, and the shader
//! only ever sees those two vectors plus `sun_roughness` and `sss_tint`.

use bevy::prelude::Resource;

/// One water body's optical profile.
///
/// The presets below are calibrated against a reference frame rather than
/// authored by eye. The target is a dark, cool, moderately desaturated blue —
/// hue 205-215 degrees, saturation 32-41%, value 30-48% — and the numbers that
/// produce it were solved, not guessed.
///
/// What the calibration changed, and why, is the *ratio* between the channels
/// rather than the overall level. The body colour of optically deep water
/// converges on `sigma_s / sigma_t`, so the red-to-blue contrast of the
/// extinction sets how saturated the water can ever be. The presets used to
/// carry an 8 : 2.1 : 1 red : green : blue extinction, which drives that ratio
/// to 0.107 and renders the sea at 77% saturation — roughly twice the target.
/// The reference implies 0.523, and reaching it needs a far flatter extinction
/// (about 1.8 : 1.3 : 1) together with more particle scatter, since a larger
/// white `sigma_s` dilutes the blue lean that dividing by a per-channel
/// `sigma_t` would otherwise give.
///
/// The scatter tint still stays white. That is the physical answer — scattering
/// off particles in water is close to wavelength-independent, so the sea's blue
/// belongs to absorption alone — and tinting the scatter as well would count
/// the same effect twice.
///
/// One honest caveat: the reference is a still mountain lake reflecting a dark
/// forested shore, so part of its low saturation is a dark environment rather
/// than a property of the water. These presets match the *material* colour, at
/// the view angles where the body dominates. Grazing water here still resolves
/// toward this renderer's own sky, which is brighter than that lake's.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterOptics {
    /// Beer-Lambert extinction per metre, per channel.
    pub extinction: [f32; 3],
    /// Multiplier on the particle scatter coefficient.
    pub scatter_scale: f32,
    /// Tint applied to particle scatter.
    pub scatter_tint: [f32; 3],
    /// Henyey-Greenstein asymmetry, clamped to +/- 0.99 in the shader.
    pub scattering_asymmetry: f32,
    /// Sun-lobe roughness override; negative means "use the wave roughness".
    pub sun_roughness: f32,
    /// Sunlit subsurface scattering tint through pinched crests.
    pub sss_tint: [f32; 3],
}

impl WaterOptics {
    /// Clear open ocean.
    ///
    /// The bluest of the three: its extinction still leans hardest on red, so
    /// the body settles at a 203-205 degree hue. It is also the least
    /// attenuating at every wavelength, which is what makes it read as clear
    /// rather than turbid — the colour arrives from a long path, not from
    /// anything suspended in it.
    pub const DEEP_OCEAN: Self = Self {
        extinction: [0.17, 0.0978, 0.0723],
        scatter_scale: 2.0,
        scatter_tint: [1.0, 1.0, 1.0],
        scattering_asymmetry: 0.8,
        sun_roughness: -1.0,
        sss_tint: [0.088_506_84, 0.497, 0.456_150_74],
    };

    /// Coastal water, and the default. This is the preset the reference
    /// calibration above was solved against: at the level that solve produced
    /// it measured hue 210, saturation 35, value 44 at the view angles where
    /// the body dominates.
    ///
    /// It attenuates slightly harder than the open ocean at every wavelength —
    /// more suspended matter, so a shorter path carries the colour — and its
    /// red-to-blue contrast is a touch softer, which is what keeps the shallows
    /// and the shore break reading green-teal over sand.
    ///
    /// The extinction below is that solved level scaled uniformly by 1.3,
    /// because the sea read a shade too pale for a coastal water. A common
    /// factor is the only edit that can darken the body without disturbing the
    /// calibration: optically deep water converges on `sigma_s / sigma_t`, and
    /// `sigma_s` does not move with `sigma_t` here — it is the white particle
    /// term, and it stays under `sigma_t` on every channel, so the `min()` in
    /// the shader never binds. Dividing by 1.3 therefore takes the level down
    /// by exactly 1.3 and leaves the ratio untouched, which means hue 213 and
    /// the 48.5% linear saturation are invariant and only the value falls;
    /// through ACES the measured 44 is expected to land near 37, still inside
    /// the 30-48 window above. That figure is a prediction from the model and
    /// has not been re-measured — treat the 210/35/44 as the recorded
    /// measurement and this as the expected consequence.
    ///
    /// Two side effects are intended, not incidental. Extinction is shared with
    /// the submerged-camera medium, so the view from below darkens by the same
    /// 1.3 and its 99% convergence depth shortens from 21.7/30.1/38.7 m to
    /// 16.7/23.2/29.8 m — the single-source optics contract means this cannot be
    /// split without letting the surface and the medium disagree. And the
    /// shallow shelf loses transmittance fastest in red (5 m falls R 0.346 to
    /// 0.251, B 0.552 to 0.461), so the green-teal band over sand narrows.
    pub const COASTAL: Self = Self {
        extinction: [0.27625, 0.1989, 0.1547],
        scatter_scale: 2.0,
        scatter_tint: [1.0, 1.0, 1.0],
        scattering_asymmetry: 0.8,
        sun_roughness: -1.0,
        sss_tint: [0.06, 0.55, 0.45],
    };

    /// Tropical lagoon: the only one that absorbs blue sooner than green, so
    /// its body settles well round the blue end of the circle — hue 166 to 193
    /// — and the deep water stays teal rather than going navy. It is the most
    /// attenuating of the three, which is what a lagoon full of suspended
    /// carbonate is.
    pub const TROPICAL: Self = Self {
        extinction: [0.238, 0.1275, 0.1615],
        scatter_scale: 2.0,
        scatter_tint: [1.0, 1.0, 1.0],
        scattering_asymmetry: 0.8,
        sun_roughness: -1.0,
        sss_tint: [0.025, 0.62, 0.38],
    };

    /// Named presets in runtime cycle order.
    pub const PRESETS: [(&'static str, Self); 3] = [
        ("deep-ocean", Self::DEEP_OCEAN),
        ("coastal", Self::COASTAL),
        ("tropical", Self::TROPICAL),
    ];
}

impl Default for WaterOptics {
    fn default() -> Self {
        Self::COASTAL
    }
}

/// Runtime-selectable sea state and optics, driven from the diagnostics
/// window the way aqua's `OceanWaves` and `WaterOptics` are authoring inputs.
///
/// `PartialEq` is load-bearing: the render node compares successive values to
/// decide whether the wave block needs rebuilding, so the 40-component spectrum
/// is only regenerated when a field it depends on actually changes.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct WaterSettings {
    pub enabled: bool,
    /// Global amplitude multiplier on the whole spectrum. 1.0 is the
    /// calibration the power table was authored against; aqua's default sea
    /// state is a little calmer than that.
    pub sea_state_amplitude: f32,
    /// Wind heading in degrees, which the component directions spread around.
    pub wind_direction_degrees: f32,
    pub optics: WaterOptics,
    /// Draw the water surface's underside and the submerged-camera medium
    /// when the eye is below the surface.
    pub underwater_effects: bool,
    /// Debug: skip the wave displacement and draw the flat plane, so the
    /// shoreline and depth tests can be read without the swell.
    pub flat_surface: bool,
}

impl Default for WaterSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            // Aqua's `SeaState::Calm` multiplier. The raw Crest table is a
            // 150 kph wind sea, which is far too rough for a lake shoreline.
            sea_state_amplitude: 0.28,
            wind_direction_degrees: 42.0,
            optics: WaterOptics::default(),
            underwater_effects: true,
            flat_surface: false,
        }
    }
}
