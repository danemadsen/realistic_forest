//! Analytic Gerstner wave spectrum.
//!
//! Ported from bevy-aqua's `bevy-aqua-waves/src/lib.rs`, which is itself a
//! reimplementation of Crest's `Scripts/Shapes/ShapeGerstnerBatched.cs` and
//! `Shaders/OceanInputs/GerstnerShared.hlsl`; the power table is Crest's
//! `OceanWaveSpectrum.cs` defaults. See `src/water/ATTRIBUTION.md`.
//!
//! DIVERGENCE: aqua bakes this sum into a five-layer `rgba16float` texture
//! array with a compute pass and samples it per fragment. This port evaluates
//! displacement per vertex with a continuous mesh limit, and normals and
//! compression per fragment with a directional pixel-footprint filter. That
//! keeps distant wave lighting independent of the coarse mesh and flat horizon
//! skirt. Unresolved components skip trigonometry and contribute slope variance
//! to specular roughness. The band partition is retained for the GPU contract.
//!
//! DIVERGENCE: [`RIPPLE_GAIN`] additionally scales the uploaded amplitude of
//! everything shorter than [`RIPPLE_BAND_METRES`]. Aqua carries no such gain;
//! it is this port's look tuning, applied on top of the verbatim Crest power
//! table rather than inside it, and its doc comment carries the calibration.

use bevy::math::Vec2;

/// Crest's spectrum resolution. 112 components, of which the 40 in the band
/// the mesh can actually resolve are uploaded.
pub const OCTAVE_COUNT: usize = 14;
pub const COMPONENTS_PER_OCTAVE: usize = 8;
pub const COMPONENT_COUNT: usize = OCTAVE_COUNT * COMPONENTS_PER_OCTAVE;

/// Wave slots in the uniform block. Aqua calls this `WAVE_SLOTS`; the band
/// partition below selects exactly this many components, so the array is
/// always full and the shader never reads an empty slot.
pub const WAVE_SLOTS: usize = 40;

const SMALLEST_WAVELENGTH_POWER: i32 = -4;
const DIRECTION_VARIANCE_DEGREES: f32 = 90.0;
const WIND_SPEED_KPH: f32 = 150.0;
const KPH_PER_MPS: f32 = 3.6;
const GRAVITY: f32 = 9.81;

/// Horizontal chop as a fraction of amplitude. Aqua's `ANALYTIC_CHOP`; above
/// about 1.7 the Gerstner sum starts folding over itself and crests invert.
const ANALYTIC_CHOP: f32 = 1.6;

/// Upper wavelength of the ripple band, in metres. 8.0 is an octave boundary
/// (octave 6 spans `[4, 8)`), so the gain's edge always falls between two
/// octaves and never half-scales an LOD band; `ripple_band_starts_on_an_octave`
/// below pins that.
///
/// The two octaves under it are where the surface's slope lives. At
/// `sea_state_amplitude` 0.28 the 40 uploaded components give an RMS surface
/// tilt of 4.154 degrees, and the 2-8 m octaves carry 53.9% of that slope
/// variance on 1.8% of the wave height. The 16-64 m swell owns the height, so
/// this is the only band a strength change can move without the sea going flat.
const RIPPLE_BAND_METRES: f32 = 8.0;

/// Amplitude gain for everything shorter than [`RIPPLE_BAND_METRES`].
///
/// This is layered on top of `POWER_LOG10`, not written into it. That table is
/// Crest's `OceanWaveSpectrum.cs` defaults ported verbatim, and it also could
/// not express this gain: a component's power is interpolated between adjacent
/// entries (see `spectrum_amplitude`), so editing an index tapers across its
/// octave instead of scaling a band.
///
/// Calibration, measured against the exact float32 generator at
/// `sea_state_amplitude` 0.28 (baseline: RMS tilt 4.154 degrees, H_s 1.0672 m,
/// far-field GGX alpha 0.07952, Gerstner fold metric 0.9116):
///   0.7 -> <8 m RMS slope -30.0%, whole-surface tilt 3.539 deg (-14.8%),
///          H_s -0.45%, alpha -12.2%, fold 0.7826, mean whitecap weight
///          0.0593 -> 0.0382.
///   0.6 -> -40.0%, tilt 3.364 deg, H_s -0.57%, alpha -15.6%.
///   0.5 -> -50.0%, tilt 3.209 deg, H_s -0.67%, alpha -18.6%.
/// The height cost is under 1% at every rung, because the swell is on the other
/// side of the band edge. 0.7 is the shipped value; the rungs below it are the
/// pre-computed next steps if the sea still reads too busy.
const RIPPLE_GAIN: f32 = 0.7;

const MIN_AMPLITUDE: f32 = 0.001;
const TAU: f32 = std::f32::consts::TAU;

/// Crest `OceanWaveSpectrum.cs` defaults, before its amplitude-neutral v1
/// calibration. Indexed by octave.
const POWER_LOG10: [f32; OCTAVE_COUNT] = [
    -5.71, -5.03, -4.54, -3.88, -3.28, -2.32, -1.78, -1.21, -0.54, 0.28, 0.54, 1.03, 1.44, -8.0,
];

/// Shortest wavelength band boundary. LOD `n` owns `[0.5*M(n), M(n))` where
/// `M(n) = LOD_BASE_WAVELENGTH * 2^n`.
///
/// These five bands must together select exactly `WAVE_SLOTS` components of
/// the 112, or `build` panics. Octave `o` spans `[2^(o-4), 2^(o-3))`, so
/// `M(n) = 4*2^n` splits the spectrum on octave boundaries: bands 0..4 are
/// octaves 5..9, eight components each, forty total.
pub const LOD_BASE_WAVELENGTH: f32 = 4.0;

/// One Gerstner component, in the shader's layout.
#[derive(Clone, Copy, Debug, Default)]
pub struct WaveComponent {
    /// Unit travel direction in world XZ.
    pub direction: [f32; 2],
    /// Vertical amplitude in metres.
    pub amplitude: f32,
    /// `TAU / wavelength`.
    pub wave_number: f32,
    /// Deep-water dispersion `sqrt(g*k)`.
    pub angular_frequency: f32,
    /// Phase offset in radians.
    pub phase: f32,
    /// Horizontal amplitude; negative by convention, so crests sharpen.
    pub chop_amplitude: f32,
    /// Uploaded for mesh and pixel-footprint filtering, and used by the band partition.
    pub wavelength: f32,
}

/// The uploaded spectrum plus the per-LOD band boundaries.
#[derive(Clone, Debug)]
pub struct WaveSpectrum {
    pub waves: [WaveComponent; WAVE_SLOTS],
    /// `[start, end)` index pairs into `waves`, one per LOD.
    pub ranges: [[u32; 2]; super::WATER_LOD_COUNT],
}

/// LOD `n`'s band boundary in metres.
pub fn lod_max_wavelength(lod: usize) -> f32 {
    LOD_BASE_WAVELENGTH * (1u32 << lod) as f32
}

/// Build the spectrum for one sea state.
///
/// `amplitude_multiplier` is aqua's `SeaState::amplitude_multiplier`; 1.0 is
/// the calibration the power table was authored against.
pub fn build(amplitude_multiplier: f32, wind_radians: f32) -> WaveSpectrum {
    let components = generate_components(amplitude_multiplier, wind_radians);
    let lowest = 0.5 * lod_max_wavelength(0);
    let highest = lod_max_wavelength(super::WATER_LOD_COUNT - 1);
    let mut selected: Vec<WaveComponent> = components
        .into_iter()
        .filter(|wave| wave.wavelength >= lowest && wave.wavelength < highest)
        .collect();
    // Ascending wavelength, so `partition_point` below is valid.
    selected.sort_by(|a, b| a.wavelength.total_cmp(&b.wavelength));
    assert_eq!(
        selected.len(),
        WAVE_SLOTS,
        "the five LOD bands must select exactly WAVE_SLOTS components",
    );
    let waves: [WaveComponent; WAVE_SLOTS] = selected
        .as_slice()
        .try_into()
        .expect("length checked above");

    let ranges = std::array::from_fn(|lod| {
        let maximum = lod_max_wavelength(lod);
        let minimum = 0.5 * maximum;
        let start = waves.partition_point(|wave| wave.wavelength < minimum);
        let end = waves.partition_point(|wave| wave.wavelength < maximum);
        [start as u32, end as u32]
    });
    WaveSpectrum { waves, ranges }
}

/// Crest's component distribution: per octave, one component per eighth of the
/// octave's wavelength span, with the direction spread evenly across the
/// wind's +/- 90 degree window.
fn generate_components(
    amplitude_multiplier: f32,
    wind_radians: f32,
) -> [WaveComponent; COMPONENT_COUNT] {
    let mut random = Random::new(0);
    let mut wavelengths = [0.0f32; COMPONENT_COUNT];
    let mut angles = [0.0f32; COMPONENT_COUNT];

    for octave in 0..OCTAVE_COUNT {
        let base = 2.0f32.powi(SMALLEST_WAVELENGTH_POWER + octave as i32);
        for component in 0..COMPONENTS_PER_OCTAVE {
            let index = octave * COMPONENTS_PER_OCTAVE + component;
            let fraction = component as f32 / COMPONENTS_PER_OCTAVE as f32;
            let minimum = base * (1.0 + fraction);
            let maximum =
                (minimum + base / COMPONENTS_PER_OCTAVE as f32).min(2.0 * base);
            wavelengths[index] = minimum + random.next() * (maximum - minimum);

            let direction_fraction =
                (component as f32 + random.next()) / COMPONENTS_PER_OCTAVE as f32;
            angles[index] = (2.0 * direction_fraction - 1.0) * DIRECTION_VARIANCE_DEGREES;
        }
    }

    let mut amplitudes = [0.0f32; COMPONENT_COUNT];
    for (amplitude, wavelength) in amplitudes.iter_mut().zip(wavelengths) {
        *amplitude = random.next() * spectrum_amplitude(wavelength);
        if *amplitude < MIN_AMPLITUDE {
            *amplitude = 0.0;
        }
        *amplitude *= amplitude_multiplier;
        // The ripple taper. Applied after the multiplier and after the
        // MIN_AMPLITUDE gate: this scales how much amplitude the sea carries,
        // never which components exist. The live set is therefore identical to
        // upstream's — a nonzero amplitude times 0.7 is still nonzero — so the
        // WAVE_SLOTS assert and the LOD partition below are untouched. One
        // component in the band does land under MIN_AMPLITUDE as a result
        // (3.33 m, 0.465 mm). That is harmless: the constant is only a
        // generation-time gate and nothing downstream treats it as a floor.
        //
        // `chop_amplitude` is derived from `amplitude` further down, so the
        // displacement, the analytic normals and the crest pinch all fall
        // together. That is the property a shader-side weight cut cannot have:
        // scaling amplitude at the upload point keeps the drawn geometry and
        // its shading the same surface.
        if wavelength < RIPPLE_BAND_METRES {
            *amplitude *= RIPPLE_GAIN;
        }
    }

    let mut phase_random = Random::new(0);
    std::array::from_fn(|index| {
        let wavelength = wavelengths[index];
        let direction =
            Vec2::from_angle(wind_radians + angles[index].to_radians());
        let amplitude = amplitudes[index];
        let wave_number = TAU / wavelength;
        let phase = TAU * ((index % COMPONENTS_PER_OCTAVE) as f32 + phase_random.next())
            / COMPONENTS_PER_OCTAVE as f32;
        WaveComponent {
            direction: [direction.x, direction.y],
            amplitude,
            wave_number,
            angular_frequency: (GRAVITY * wave_number).sqrt(),
            phase,
            chop_amplitude: -ANALYTIC_CHOP * amplitude,
            wavelength,
        }
    })
}

/// Crest's spectral power for one wavelength: the octave power table
/// interpolated in log-wavelength, windowed by the Pierson-Moskowitz wind
/// term, converted from a power density to an amplitude via `sqrt(2*S*dw)`.
fn spectrum_amplitude(wavelength: f32) -> f32 {
    let power = wavelength.log2().clamp(
        SMALLEST_WAVELENGTH_POWER as f32,
        (SMALLEST_WAVELENGTH_POWER + OCTAVE_COUNT as i32 - 1) as f32,
    );
    let lower_wavelength = 2.0f32.powf(power.floor());
    let octave = (power - SMALLEST_WAVELENGTH_POWER as f32) as usize;
    let alpha = (wavelength - lower_wavelength) / lower_wavelength;
    let next = (octave + 1).min(OCTAVE_COUNT - 1);
    let log_power = POWER_LOG10[octave] + alpha * (POWER_LOG10[next] - POWER_LOG10[octave]);
    let mut spectral_power = 10.0f32.powf(log_power);

    let omega_lower = (GRAVITY * TAU / lower_wavelength).sqrt();
    let omega_upper = (GRAVITY * TAU / (2.0 * lower_wavelength)).sqrt();
    let delta_omega = (omega_lower - omega_upper) / COMPONENTS_PER_OCTAVE as f32;
    let omega = (GRAVITY * TAU / wavelength).sqrt();
    let wind_frequency = 0.87 * GRAVITY / (WIND_SPEED_KPH / KPH_PER_MPS);
    spectral_power *= (-1.291 * (wind_frequency / omega).powi(4)).exp();

    (2.0 * spectral_power * delta_omega).sqrt()
}

/// Deterministic local substitute for Unity's `Random`. Aqua keeps the same
/// LCG so the spectrum is identical across runs and across ports; exact Unity
/// bit parity is not required and was never achieved upstream either.
#[derive(Clone, Copy)]
struct Random(u32);

impl Random {
    const fn new(seed: u32) -> Self {
        Self(seed)
    }

    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (self.0 >> 8) as f32 / 16_777_216.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_lod_band_is_populated() {
        let spectrum = build(1.0, 0.0);
        for (lod, range) in spectrum.ranges.iter().enumerate() {
            assert!(
                range[1] > range[0],
                "LOD {lod} owns an empty band {range:?}",
            );
        }
    }

    #[test]
    fn bands_are_contiguous_and_ascending() {
        let spectrum = build(1.0, 0.0);
        for pair in spectrum.ranges.windows(2) {
            assert_eq!(pair[0][1], pair[1][0], "bands must tile without a gap");
        }
        assert_eq!(spectrum.ranges[0][0], 0);
        assert_eq!(
            spectrum.ranges[super::super::WATER_LOD_COUNT - 1][1] as usize,
            WAVE_SLOTS,
        );
    }

    #[test]
    fn ripple_band_starts_on_an_octave_boundary() {
        // The gain's edge must fall between octaves. If it landed inside one,
        // part of that octave would be scaled and part not, and the band
        // partition (which splits on octave boundaries) would disagree with it.
        // `RIPPLE_BAND_METRES` is exactly octave 7's lower edge, so the scaled
        // set is whole octaves 5 and 6.
        assert_eq!(RIPPLE_BAND_METRES, lod_max_wavelength(1));
        let spectrum = build(1.0, 0.0);
        let shorter = spectrum
            .waves
            .iter()
            .filter(|wave| wave.wavelength < RIPPLE_BAND_METRES)
            .count();
        assert_eq!(
            shorter,
            2 * COMPONENTS_PER_OCTAVE,
            "the scaled set must be the two whole octaves under the edge",
        );
    }

    #[test]
    fn amplitudes_are_finite_and_bounded() {
        let spectrum = build(1.0, 0.0);
        let total: f32 = spectrum.waves.iter().map(|wave| wave.amplitude.abs()).sum();
        assert!(spectrum.waves.iter().all(|wave| wave.amplitude.is_finite()));
        // The whole sea state should stay well inside the water plan's bounds.
        assert!(total > 0.05 && total < 12.0, "suspicious total amplitude {total}");
    }
}
