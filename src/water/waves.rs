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
//! DIVERGENCE: three constants tune the sea this port draws without touching
//! the ported table. [`RIPPLE_GAIN`] and [`SWELL_GAIN`] scale the uploaded
//! amplitude of everything shorter than [`RIPPLE_BAND_METRES`] and of
//! everything at or above [`SWELL_BAND_METRES`]; [`DIRECTION_CONCENTRATION`]
//! shapes the per-component direction draw. Aqua carries no equivalent of any
//! of them. They are this port's look tuning, layered on top of the verbatim
//! Crest power table rather than written into it, and each doc comment carries
//! its own calibration.
//!
//! Together they scale Crest's ocean to a coastal sea under a moderate
//! breeze. Crest's table is a fully developed 150 kph wind sea: 82% of its
//! wave height sits in the 16-64 m octaves and the largest single component is
//! a 60 m, 6.2 s swell. The taper turns that swell down without taking it out,
//! the ripple gain takes the fine chop down, and the concentration organises
//! what is left into a travelling train instead of an isotropic fan. See the
//! constants below for the measured rungs. Waves shorter than the 2 m this
//! spectrum starts at are not Crest's: they are the wind sea every water body
//! shares (`windSea` in assets/shaders/water.wgsl), which on the open sea takes
//! the whole of the wind over unlimited fetch. Lakes and ponds have only that
//! wind sea, limited by their fetch and their shelter, and no swell at all,
//! which is what keeps them calmer than the sea.

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
/// Half-width of the direction window, in degrees, that Crest's generator
/// spreads components across. See [`DIRECTION_CONCENTRATION`] for the shape of
/// the draw inside it.
const DIRECTION_VARIANCE_DEGREES: f32 = 90.0;

/// Exponent concentrating the direction draw toward the wind heading.
///
/// Crest's generator spreads one component per eighth of each octave evenly
/// across the full `+/- DIRECTION_VARIANCE_DEGREES` window, so the 40 uploaded
/// components fan out with no preferred heading: the sum is directionally
/// random and reads as cross-hatched foil rather than as a wave train. A real
/// wind sea is narrow — the fetch axis *is* the wind axis, and every off-axis
/// wave is limited by a shorter cross-fetch — which a cosine power law
/// reproduces. The draw maps the same uniform variate `u` in `[-1, 1]`
/// through `u * |u|^(p-1)`, whose density peaks at the wind heading and whose
/// extremes still reach the window edge, so `1.0` is the **exact identity**
/// (bit-for-bit, `powf(x, 0.0) == 1.0`) and reproduces today's ship.
///
/// The number of LCG draws is unchanged — the exponent is applied to the value
/// the existing draw already produced — so the wavelengths, the amplitudes, the
/// sort order, the LOD partition and every other measured quantity are
/// untouched. Only the along/cross split of the slope covariance moves.
///
/// Calibration, the directional concentration `lambda_max/trace` of the slope
/// covariance `sum 0.5(ak)^2 dir dir^T` (0.5 is fully isotropic, 1.0 a perfect
/// train; measured with [`SWELL_GAIN`] at 0.20, [`RIPPLE_GAIN`] 0.5 and
/// `sea_state_amplitude` 0.28):
///   1.0 -> 0.552, along-wind tilt 0.037 deg, cross-wind 0.034 (shipped)
///   1.5 -> 0.690, along 0.042, cross 0.028
///   2.0 -> 0.770, along 0.044, cross 0.024 (shipped value)
///   2.5 -> 0.821, along 0.046, cross 0.021
/// Total tilt and H_s do not move at any rung (they are direction-independent),
/// and the breaking area moves by at most 0.11 points across the whole sweep:
/// this is pure shape. 2.5 is the next step if a capture still reads as
/// speckle; 1.5 if the surface reads as corduroy, which is the failure mode
/// above about 0.85.
const DIRECTION_CONCENTRATION: f32 = 2.0;

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
/// `sea_state_amplitude` 0.28 with [`SWELL_GAIN`] at 0.20 the 40 uploaded
/// components give an RMS surface tilt of 2.037 degrees; the 2-8 m octaves
/// carry 56.1% of that slope variance on 17.6% of the wave height, and the
/// 16-64 m octaves only 6.2% of the slope on 51.6% of the height. The 2-8 m
/// band is therefore both the near-field look and the band a strength change
/// moves.
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
/// `sea_state_amplitude` 0.28 with [`SWELL_GAIN`] at 0.20 and
/// [`DIRECTION_CONCENTRATION`] at 2.0 (baseline, gain 1.0: RMS tilt 3.335
/// degrees, H_s 0.3090 m, sum|a| 0.5835 m, far-field GGX alpha 0.06667, fold
/// 0.6240, breaking area 14.27%):
///   0.8 -> <8 m RMS slope -36.0%, tilt 2.789 deg, H_s -3.9%, alpha -12.3%
///   0.7 -> -51.0%, tilt 2.527 deg, H_s -5.6%, alpha -17.9%, fold 0.4950
///   0.6 -> -64.0%, tilt 2.275 deg, H_s -7.0%, alpha -23.1%
///   0.5 -> -75.0%, tilt 2.037 deg, H_s -8.3%, alpha -27.8%, fold 0.4090
/// 0.5 is the shipped value; the rungs above it are the pre-computed steps back
/// up if the lake goes too flat. Note the height cost is now several percent a
/// rung, not "under 1%": with the swell tapered away this band is 17.6% of the
/// wave height rather than 7.9%, so cutting it costs what it costs.
const RIPPLE_GAIN: f32 = 0.5;

/// Lower wavelength of the swell band, in metres. `lod_max_wavelength(2)`, an
/// octave boundary for the same reason [`RIPPLE_BAND_METRES`] is one: the gain
/// must not half-scale an LOD band. It is the boundary between octave 7
/// (`[8, 16)`) and octave 8, so the tapered set is whole octaves 8 and 9 — the
/// two longest uploaded bands, `[16, 64)`, sixteen components.
const SWELL_BAND_METRES: f32 = 16.0;

/// Amplitude gain for everything at or above [`SWELL_BAND_METRES`].
///
/// Built exactly like [`RIPPLE_GAIN`]: this port's look tuning, layered on top
/// of Crest's power table rather than written into it, applied after the
/// `MIN_AMPLITUDE` gate so the live component set, the `WAVE_SLOTS` assert and
/// the LOD partition are untouched.
///
/// Crest's table is a fully developed 150 kph wind sea. Because
/// `WIND_SPEED_KPH` puts the Pierson-Moskowitz peak at 1469 m — an order of
/// magnitude past the longest band the renderer draws — the wind window is
/// inert across the uploaded spectrum (its amplitude factor is 0.998 at 64 m)
/// and `POWER_LOG10` alone shapes the sea into 16-64 m swell: 79.6% of the wave
/// height, with a single 60 m, 6.2 s component as the largest wave in the
/// water. That is a storm sea's swell, so the gain tapers it. Moving
/// `WIND_SPEED_KPH` instead would drag the window's peak into the resolved
/// band and invalidate every gain rung documented above; editing
/// `POWER_LOG10` would break the "Crest defaults ported verbatim" contract and
/// smear across an octave through the interpolation in `spectrum_amplitude`.
///
/// The taper is also what changes the *shape* of the sea, not just its size:
/// below about 0.25 the largest single component stops being the 60 m swell and
/// becomes a 13.1 m, 2.90 s wave, which is the lake regime the shoreline
/// surveys report (H_s 0.10-0.40 m, T_p 1.4-2.5 s summary / 1.7-3.6 s
/// measured). The sea was held there (0.20) while it was the only water with
/// waves at all. Now that lakes raise their own fetch-limited waves, the sea
/// is an ocean again: at 0.35 the swell is still the largest wave and the sea
/// reads as an ocean with the amplitude turned down, H_s 0.41 m under the
/// default breeze, rougher than any lake and calm enough for a shoreline.
///
/// Calibration at `sea_state_amplitude` 0.28 with [`RIPPLE_GAIN`] at its
/// shipped 0.5 and [`DIRECTION_CONCENTRATION`] at 2.0, columns sum|a| m / H_s m
/// / RMS tilt deg / 2-8 m tilt deg / 16-64 m tilt deg / far-field alpha /
/// fold / breaking area (Jacobian < 0.90) / largest component:
///   1.00 -> 1.5204 / 1.0601 / 3.209 / 1.526 / 2.532 / 0.06475 / 0.6965 /
///           13.32% / 60 m, 6.19 s
///   0.50 -> 0.8802 / 0.5555 / 2.344 / 1.526 / 1.267 / 0.05221 / 0.5168 /
///            6.20% / 60 m, 6.19 s
///   0.35 -> 0.6882 / 0.4123 / 2.163 / 1.526 / 0.887 / 0.04977 / 0.4629 /
///            4.70% / 60 m, 6.19 s (shipped value)
///   0.25 -> 0.5601 / 0.3237 / 2.072 / 1.526 / 0.633 / 0.04857 / 0.4270 /
///            4.01% / 13 m, 2.90 s
///   0.20 -> 0.4961 / 0.2834 / 2.037 / 1.526 / 0.507 / 0.04812 / 0.4090 /
///            3.73% / 13 m, 2.90 s (the lake regime)
///   0.14 -> 0.4193 / 0.2412 / 2.005 / 1.526 / 0.355 / 0.04770 / 0.3874 /
///            3.48% / 13 m, 2.90 s
/// The 2-8 m tilt is 1.526 deg at every rung: the band edge is above it, so
/// this lever and [`RIPPLE_GAIN`] are orthogonal by construction. The taper
/// also *raises* the folding margin, from 1/sum(ak) 2.297 at 1.00 to 3.912 at
/// 0.20, rather than spending it; 0.35 still keeps it above 2. The far-field
/// alpha column is the swell's alone: the wind sea under 2 m adds its own
/// Cox-Munk roughness on top in the shader.
const SWELL_GAIN: f32 = 0.35;

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
/// octave's wavelength span, with the direction draw concentrated toward the
/// wind heading by [`DIRECTION_CONCENTRATION`] instead of spread evenly across
/// the wind's +/- 90 degree window.
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
            // The same draw as Crest, shaped rather than replaced: `u * |u|^(p-1)`
            // leaves `p = 1.0` bit-for-bit identical to the uniform fan (see
            // `DIRECTION_CONCENTRATION`) and consumes no extra LCG state, so the
            // wavelengths and amplitudes below are unchanged at every `p`.
            let direction = 2.0 * direction_fraction - 1.0;
            angles[index] = direction
                * direction.abs().powf(DIRECTION_CONCENTRATION - 1.0)
                * DIRECTION_VARIANCE_DEGREES;
        }
    }

    let mut amplitudes = [0.0f32; COMPONENT_COUNT];
    for (amplitude, wavelength) in amplitudes.iter_mut().zip(wavelengths) {
        *amplitude = random.next() * spectrum_amplitude(wavelength);
        if *amplitude < MIN_AMPLITUDE {
            *amplitude = 0.0;
        }
        *amplitude *= amplitude_multiplier;
        // The two tapers. Applied after the multiplier and after the
        // MIN_AMPLITUDE gate: they scale how much amplitude the sea carries,
        // never which components exist. The live set is therefore identical to
        // upstream's — a nonzero amplitude times 0.5 is still nonzero — so the
        // WAVE_SLOTS assert and the LOD partition below are untouched.
        //
        // Two components of the uploaded band sit under the nominal floor and
        // stay there: 3.33 m at 0.66 mm before the tapers, 0.33 mm after, and
        // 9.55 m at 0.68 mm, which no taper touches. That is harmless: the
        // constant is only a generation-time gate and nothing downstream treats
        // it as a floor.
        //
        // `chop_amplitude` is derived from `amplitude` further down, so the
        // displacement, the analytic normals and the crest pinch all fall
        // together. That is the property a shader-side weight cut cannot have:
        // scaling amplitude at the upload point keeps the drawn geometry and
        // its shading the same surface.
        if wavelength < RIPPLE_BAND_METRES {
            *amplitude *= RIPPLE_GAIN;
        }
        // The swell taper. Same construction and the same reasoning as the
        // ripple gain above: it scales how much amplitude the sea carries,
        // never which components exist, because `MIN_AMPLITUDE` has already
        // run. `SWELL_BAND_METRES` is octave 7's upper edge, so the taper
        // covers whole octaves 8 and 9 and no LOD band is half-scaled.
        if wavelength >= SWELL_BAND_METRES {
            *amplitude *= SWELL_GAIN;
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
    fn swell_band_starts_on_an_octave_boundary() {
        // Same contract as the ripple band's edge above: the taper must cover
        // whole octaves, or the LOD partition (which splits on octave
        // boundaries) would disagree with it. `SWELL_BAND_METRES` is octave 7's
        // upper edge, so the tapered set is whole octaves 8 and 9 — the two
        // longest uploaded bands.
        assert_eq!(SWELL_BAND_METRES, lod_max_wavelength(2));
        let spectrum = build(1.0, 0.0);
        let tapered = spectrum
            .waves
            .iter()
            .filter(|wave| wave.wavelength >= SWELL_BAND_METRES)
            .count();
        assert_eq!(
            tapered,
            2 * COMPONENTS_PER_OCTAVE,
            "the tapered set must be the two whole octaves over the edge",
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
