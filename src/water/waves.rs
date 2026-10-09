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
//! DIVERGENCE: two constants tune the sea this port draws without touching
//! the ported table. [`RIPPLE_GAIN`] and [`SWELL_GAIN`] scale the uploaded
//! amplitude of everything shorter than [`RIPPLE_BAND_METRES`] and of
//! everything at or above [`SWELL_BAND_METRES`]. Aqua carries no equivalent of
//! either. They are this port's look tuning, layered on top of the verbatim
//! Crest power table rather than written into it, and each doc comment carries
//! its own calibration.
//!
//! DIVERGENCE: the directions and phases are not Crest's. Crest gives each
//! component one index that picks its wavelength within its octave, its
//! direction stratum and its phase eighth all at once, and draws the phases
//! from the same seed as the wavelengths. Every octave then repeats the same
//! fan with the same phases, so the components fall into eight families of
//! octave harmonics running the same way in phase: a regular egg-crate of
//! dimples across the sea. Here the directions follow a measured directional
//! spectrum ([`spreading_beta`], drawn by [`spread_angle`]) through strata
//! shuffled independently in every octave, and the phases are uniform from a
//! seed of their own. The wavelengths and amplitudes are still Crest's draw,
//! bit for bit.
//!
//! Together they scale Crest's ocean to a coastal sea under a moderate
//! breeze. Crest's table is a fully developed 150 kph wind sea: 82% of its
//! wave height sits in the 16-64 m octaves and the largest single component is
//! a 60 m, 6.2 s swell. The taper turns that swell down without taking it out,
//! the ripple gain takes the fine chop down, and the directional spread makes
//! the swell long-crested and the shorter waves short-crested about the wind.
//! See the constants below for the measured rungs. Waves shorter than the 2 m
//! this spectrum starts at are not Crest's: they are the wind sea every water
//! body shares ([`wind_sea_components`], `windSea` in
//! assets/shaders/water.wgsl), which on the open sea takes the whole of the
//! wind over unlimited fetch. Lakes and ponds have only that wind sea, limited
//! by their fetch and their shelter, and no swell at all, which is what keeps
//! them calmer than the sea.

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

/// The broadest a sea is ever spread about the wind: the floor of
/// [`spreading_beta`], as the `beta` of a `sech^2(beta*theta)` spread.
///
/// Banner's fit for the short waves goes on widening the spread toward a
/// uniform circle (beta 0.7 at three times the peak frequency, 0.5 at ten).
/// Cox and Munk's sun-glitter photographs see more order than that in the
/// short waves that carry the slope: the wind-dependent part of their mean
/// square slope is 3.16e-3 per m/s upwind and 1.92e-3 crosswind, 1.65 to 1.
/// A sech^2 spread gives its waves' slopes that ratio at beta 0.955, so no
/// rung of either wave set is spread wider than that.
pub const SPREAD_FLOOR: f32 = 0.955;

/// The seeds of the swell's direction and phase draws. Each is a stream of
/// its own, so neither repeats the wavelength and amplitude draw (seed 0,
/// Crest's) nor the other. The direction seed was picked from the first two
/// hundred for a draw that leaves fewer octave harmonics running together
/// than chance does on average, heads within two degrees of the wind and
/// has the slope ratio its spread gives (see the tests below).
const DIRECTION_SEED: u32 = 112;
const PHASE_SEED: u32 = 2;

/// The wind sea's components, which the shader's `windSea` evaluates for
/// every water body: [`WIND_RUNGS`] half-octave rungs of wavelength from
/// [`WIND_SHORTEST_METRES`] up, [`WIND_PER_RUNG`] components to a rung.
/// Mirrored by `WIND_RUNGS`, `WIND_PER_RUNG` and `WIND_SHORTEST` in
/// assets/shaders/water.wgsl.
pub const WIND_RUNGS: usize = 14;
pub const WIND_PER_RUNG: usize = 4;
pub const WIND_COMPONENTS: usize = WIND_RUNGS * WIND_PER_RUNG;
pub const WIND_SHORTEST_METRES: f32 = 0.04;
/// The period, metres, every wind wave repeats over: its wave vector is a
/// whole number of cycles over it (see [`wind_sea_components`]). Mirrored by
/// `WIND_PERIOD` in the shader.
pub const WIND_PERIOD_METRES: f32 = 512.0;
/// The wind sea's own seed, picked like the swell's: no rung holds a mirrored
/// pair, and the set heads with the wind at its spread's slope ratio.
const WIND_SEED: u32 = 328;
/// The golden ratio's fractional part. Each rung turns its strata on by it
/// from the rung before, so no two rungs share their headings.
const GOLDEN_RATIO_FRACTION: f32 = 0.618_034;

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
/// `sea_state_amplitude` 0.28 with [`SWELL_GAIN`] at 0.20 (baseline, gain 1.0:
/// RMS tilt 3.335 degrees, H_s 0.3090 m, sum|a| 0.5835 m, far-field GGX alpha
/// 0.06667, fold 0.6240, breaking area 14.31% with the directions drawn by
/// [`spreading_beta`]; 14.27% with Crest's concentrated fan):
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
/// shipped 0.5, columns sum|a| m / H_s m / RMS tilt deg / 2-8 m tilt deg /
/// 16-64 m tilt deg / far-field alpha / fold (`ANALYTIC_CHOP * sum(ak)`) /
/// breaking area (Jacobian < 0.90, over 2 km square) / slope up:across the
/// wind / largest component:
///   1.00 -> 1.5204 / 1.0601 / 3.209 / 1.526 / 2.532 / 0.06475 / 0.6965 /
///           13.26% / 2.39 / 60 m, 6.19 s
///   0.50 -> 0.8802 / 0.5555 / 2.344 / 1.526 / 1.267 / 0.05221 / 0.5168 /
///            6.10% / 1.81 / 60 m, 6.19 s
///   0.35 -> 0.6882 / 0.4123 / 2.163 / 1.526 / 0.887 / 0.04977 / 0.4629 /
///            4.64% / 1.65 / 60 m, 6.19 s (shipped value)
///   0.25 -> 0.5601 / 0.3237 / 2.072 / 1.526 / 0.633 / 0.04857 / 0.4270 /
///            3.99% / 2.86 / 13 m, 2.90 s
///   0.20 -> 0.4961 / 0.2834 / 2.037 / 1.526 / 0.507 / 0.04812 / 0.4090 /
///            3.73% / 2.87 / 13 m, 2.90 s (the lake regime)
///   0.14 -> 0.4193 / 0.2412 / 2.005 / 1.526 / 0.355 / 0.04770 / 0.3874 /
///            3.49% / 2.89 / 13 m, 2.90 s
/// Of these, only the breaking area and the slope ratio depend on where the
/// waves head. The breaking area is within 0.1 point of what Crest's
/// concentrated fan gave (13.32%, 6.20%, 4.70%, 4.01%, 3.73%, 3.48%), whose
/// up:across was 2.7-3.3 at every rung. The spread follows the peak, so once
/// the 13 m wave is the largest the 2-8 m waves lie nearer the peak and run
/// closer to the wind.
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
/// octave's wavelength span at Crest's amplitudes, heading round the wind as a
/// measured sea does at that wavelength, at a random phase.
fn generate_components(
    amplitude_multiplier: f32,
    wind_radians: f32,
) -> [WaveComponent; COMPONENT_COUNT] {
    let mut random = Random::new(0);
    let mut wavelengths = [0.0f32; COMPONENT_COUNT];

    for octave in 0..OCTAVE_COUNT {
        let base = 2.0f32.powi(SMALLEST_WAVELENGTH_POWER + octave as i32);
        for component in 0..COMPONENTS_PER_OCTAVE {
            let index = octave * COMPONENTS_PER_OCTAVE + component;
            let fraction = component as f32 / COMPONENTS_PER_OCTAVE as f32;
            let minimum = base * (1.0 + fraction);
            let maximum =
                (minimum + base / COMPONENTS_PER_OCTAVE as f32).min(2.0 * base);
            wavelengths[index] = minimum + random.next() * (maximum - minimum);
            // Crest drew the component's direction here, from the same eighth
            // of the circle as its wavelength's eighth of the octave. The draw
            // is made and dropped, so the amplitudes below come out of the
            // stream exactly as they did: the wave heights, the tilt, the
            // folding margin, the live set and every gain calibration hold.
            random.next();
        }
    }

    let lowest = 0.5 * lod_max_wavelength(0);
    let highest = lod_max_wavelength(super::WATER_LOD_COUNT - 1);
    let mut amplitudes = [0.0f32; COMPONENT_COUNT];
    // The largest wave the sea draws, and its wavelength: the peak its
    // directional spread is measured from. Judged before the sea state scales
    // every wave alike, so it does not move with it.
    let mut peak = (0.0f32, highest);
    for (amplitude, wavelength) in amplitudes.iter_mut().zip(wavelengths) {
        *amplitude = random.next() * spectrum_amplitude(wavelength);
        if *amplitude < MIN_AMPLITUDE {
            *amplitude = 0.0;
        }
        let shape = *amplitude * band_gain(wavelength);
        if (lowest..highest).contains(&wavelength) && shape > peak.0 {
            peak = (shape, wavelength);
        }
        *amplitude *= amplitude_multiplier;
        *amplitude *= band_gain(wavelength);
    }

    // Each octave's eight components take the eight strata of the spread one
    // each, in an order shuffled afresh for every octave, so a wavelength's
    // place in its octave says nothing of its heading and no two octaves
    // repeat the same fan.
    let peak_frequency = (GRAVITY * TAU / peak.1).sqrt();
    let mut direction_random = Random::new(DIRECTION_SEED);
    let mut angles = [0.0f32; COMPONENT_COUNT];
    for octave in 0..OCTAVE_COUNT {
        let mut strata: [usize; COMPONENTS_PER_OCTAVE] = std::array::from_fn(|stratum| stratum);
        for last in (1..COMPONENTS_PER_OCTAVE).rev() {
            let pick = ((direction_random.next() * (last + 1) as f32) as usize).min(last);
            strata.swap(last, pick);
        }
        for (component, stratum) in strata.into_iter().enumerate() {
            let index = octave * COMPONENTS_PER_OCTAVE + component;
            let frequency = (GRAVITY * TAU / wavelengths[index]).sqrt();
            let u = (stratum as f32 + direction_random.next()) / COMPONENTS_PER_OCTAVE as f32;
            angles[index] = spread_angle(u, spreading_beta(frequency / peak_frequency));
        }
    }

    let mut phase_random = Random::new(PHASE_SEED);
    std::array::from_fn(|index| {
        let wavelength = wavelengths[index];
        let direction = Vec2::from_angle(wind_radians + angles[index]);
        let amplitude = amplitudes[index];
        let wave_number = TAU / wavelength;
        let phase = TAU * phase_random.next();
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

/// The two tapers, [`RIPPLE_GAIN`] under [`RIPPLE_BAND_METRES`] and
/// [`SWELL_GAIN`] from [`SWELL_BAND_METRES`] up. They are applied after the
/// multiplier and after the `MIN_AMPLITUDE` gate: they scale how much amplitude
/// the sea carries, never which components exist. The live set is therefore
/// identical to upstream's — a nonzero amplitude times 0.5 is still nonzero —
/// so the `WAVE_SLOTS` assert and the LOD partition are untouched. Both band
/// edges are octave boundaries, so the tapers cover whole octaves (5 and 6, 8
/// and 9) and no LOD band is half-scaled.
///
/// Two components of the uploaded band sit under the nominal floor and stay
/// there: 3.33 m at 0.66 mm before the tapers, 0.33 mm after, and 9.55 m at
/// 0.68 mm, which no taper touches. That is harmless: the constant is only a
/// generation-time gate and nothing downstream treats it as a floor.
///
/// `chop_amplitude` is derived from `amplitude` after the gain, so the
/// displacement, the analytic normals and the crest pinch all fall together.
/// That is the property a shader-side weight cut cannot have: scaling
/// amplitude at the upload point keeps the drawn geometry and its shading the
/// same surface.
fn band_gain(wavelength: f32) -> f32 {
    if wavelength < RIPPLE_BAND_METRES {
        RIPPLE_GAIN
    } else if wavelength >= SWELL_BAND_METRES {
        SWELL_GAIN
    } else {
        1.0
    }
}

/// How narrowly a wind sea's waves at `frequency_ratio` times its peak's
/// angular frequency are spread about the wind, as the `beta` of a
/// `sech^2(beta*theta)` spread.
///
/// Donelan, Hamilton and Hui (1985) measured it with a wave-staff array on
/// Lake Ontario: narrowest just past the peak (beta 2.4), wider to either
/// side, `2.61 r^1.3` below 0.95 times the peak frequency and `2.28 r^-0.65`
/// above it. Their data stopped at 1.6, past which they held it at 1.24;
/// Banner (1990) carried it on from stereo photographs of the shorter waves as
/// `10^(-0.4 + 0.8393 exp(-0.567 ln r^2))`, which starts from that 1.24. The
/// two fits step from 1.68 to 1.24 at 1.6; the step is eased over 1.6-2.0, so
/// a ripple field whose peak moves with the gusts does not change its spread
/// along a line. No wave is spread wider than [`SPREAD_FLOOR`].
///
/// Mirrored by `spreadingBeta` in assets/shaders/water.wgsl.
pub fn spreading_beta(frequency_ratio: f32) -> f32 {
    let r = frequency_ratio.max(1e-3);
    let donelan = if r < 0.95 { 2.61 * r.powf(1.3) } else { 2.28 * r.powf(-0.65) };
    let banner = 10.0f32.powf(-0.4 + 0.8393 * (-0.567 * (r * r).ln()).exp());
    let t = ((r - 1.6) / 0.4).clamp(0.0, 1.0);
    let eased = t * t * (3.0 - 2.0 * t);
    (donelan + (banner - donelan) * eased).max(SPREAD_FLOOR)
}

/// The heading, radians off the wind, under which a fraction `u` of the waves
/// of a `sech^2(beta*theta)` spread over the whole circle lie: the inverse of
/// that spread's cumulative distribution, `tanh(beta*theta)/tanh(beta*pi)`
/// mapped from `[-1, 1]` to `[0, 1]`.
pub fn spread_angle(u: f32, beta: f32) -> f32 {
    ((2.0 * u - 1.0) * (beta * std::f32::consts::PI).tanh()).atanh() / beta
}

/// The wind sea's components for a wind blowing toward `wind_radians`, as the
/// shader reads them (`stage.wind_waves`): each one's wave vector as whole
/// cycles over [`WIND_PERIOD_METRES`] along x and z, its phase in cycles, and
/// its heading off the wind in radians.
///
/// The set depends on the wind's heading and on nothing else. The shader
/// spreads a rung's slope over its components by weights that follow the
/// local wind and fetch, but where a crest lies and where it heads come only
/// from here: a gust, a shore further upwind or the camera's maps moving on
/// can change how high the ripples run, never re-phase them from one frame to
/// the next.
///
/// Every rung takes [`WIND_PER_RUNG`] headings, one from each equal share of
/// the broadest spread ([`SPREAD_FLOOR`]), turned on by the golden ratio from
/// rung to rung, so no rung pairs its waves mirrored about the wind to cross
/// in a lattice and no two rungs share their headings. The wavelengths are
/// spread across the rung's half octave and the phases are random, so the
/// rungs do not lock to each other either. A wave vector is a whole number of
/// cycles over the period so the shader can wrap the position into one period
/// first and keep the phase exact far from the origin; snapping moves a
/// heading by at most a third of a degree, and the period is far too long to
/// see repeat.
pub fn wind_sea_components(wind_radians: f32) -> [[f32; 4]; WIND_COMPONENTS] {
    let mut random = Random::new(WIND_SEED);
    let mut components = [[0.0f32; 4]; WIND_COMPONENTS];
    for (index, component) in components.iter_mut().enumerate() {
        let rung = index / WIND_PER_RUNG;
        let stratum = index % WIND_PER_RUNG;
        let jitter = random.next();
        let stretch = random.next();
        let phase = random.next();
        let u = ((stratum as f32 + jitter) / WIND_PER_RUNG as f32
            + rung as f32 * GOLDEN_RATIO_FRACTION)
            .fract();
        let angle = spread_angle(u, SPREAD_FLOOR);
        // Within a quarter octave either side of the rung's nominal length.
        let wavelength = f64::from(WIND_SHORTEST_METRES)
            * 2.0f64.powf(0.5 * rung as f64 + 0.5 * (f64::from(stretch) - 0.5));
        let cycles = f64::from(WIND_PERIOD_METRES) / wavelength;
        let heading = f64::from(wind_radians) + f64::from(angle);
        *component = [
            (heading.cos() * cycles).round() as f32,
            (heading.sin() * cycles).round() as f32,
            phase,
            angle,
        ];
    }
    components
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

    /// The heading off the wind, radians, of an uploaded swell component.
    fn heading_off(direction: [f32; 2], wind: f32) -> f32 {
        let turn = direction[1].atan2(direction[0]) - wind;
        (turn + std::f32::consts::PI).rem_euclid(TAU) - std::f32::consts::PI
    }

    /// The weighted mean heading off the wind, degrees, and the ratio of the
    /// mean square slope up the wind to across it, of waves at `(heading,
    /// slope variance)`.
    fn spread_of(waves: impl Iterator<Item = (f32, f32)>) -> (f32, f32) {
        let (mut sine, mut cosine, mut up, mut across) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        for (heading, variance) in waves {
            sine += variance * heading.sin();
            cosine += variance * heading.cos();
            up += variance * heading.cos().powi(2);
            across += variance * heading.sin().powi(2);
        }
        (sine.atan2(cosine).to_degrees(), up / across)
    }

    /// The share of a `sech^2(beta*theta)` spread under `theta`: what
    /// [`spread_angle`] inverts.
    fn spread_share(theta: f32, beta: f32) -> f32 {
        0.5 + 0.5 * (beta * theta).tanh() / (beta * std::f32::consts::PI).tanh()
    }

    /// The directions and phases are drawn from their own streams, so the
    /// wavelengths and amplitudes must still be Crest's draw exactly: the
    /// wave heights and the largest wave documented on [`SWELL_GAIN`].
    #[test]
    fn the_wavelengths_and_amplitudes_are_crests_draw() {
        let spectrum = build(0.28, 0.7);
        let total: f32 = spectrum.waves.iter().map(|wave| wave.amplitude.abs()).sum();
        let variance: f32 = spectrum.waves.iter().map(|wave| 0.5 * wave.amplitude * wave.amplitude).sum();
        assert!((total - 0.6882).abs() < 1e-4, "sum|a| {total}");
        assert!((4.0 * variance.sqrt() - 0.4123).abs() < 1e-4, "H_s {}", 4.0 * variance.sqrt());
        let largest = spectrum.waves.iter().max_by(|a, b| a.amplitude.total_cmp(&b.amplitude)).unwrap();
        assert!((59.0..61.0).contains(&largest.wavelength), "largest {}", largest.wavelength);
    }

    /// Two components an octave apart that run the same way are the first two
    /// harmonics of one sharpened wave. Crest's generator gave a component its
    /// heading by its place in its octave, so every octave repeated the same
    /// fan and 32 such pairs ran across the sea as eight families of
    /// egg-crate dimples. Drawn independently they pair by chance alone,
    /// about eight times (the mean over two hundred seeds, deviation 2.4).
    #[test]
    fn the_swell_has_no_families_of_octave_harmonics() {
        let spectrum = build(0.28, 0.7);
        let waves = &spectrum.waves;
        let mut pairs = 0;
        for a in waves {
            for b in waves {
                let ratio = b.wavelength / a.wavelength;
                let apart = heading_off(b.direction, 0.7) - heading_off(a.direction, 0.7);
                if (ratio - 2.0).abs() < 0.15 && apart.abs().to_degrees() < 10.0 {
                    pairs += 1;
                }
            }
        }
        assert!(pairs <= 10, "{pairs} octave-harmonic pairs run together");
    }

    /// The swell heads with the wind, spread as a measured sea is: the mean
    /// heading close to the wind's and the slope steeper up the wind than
    /// across it, as Cox and Munk's 1.65 to 1 for the short waves, less than a
    /// swell train's single direction. Crest's concentrated fan gave 3.2.
    #[test]
    fn the_swell_runs_with_the_wind_at_a_measured_spread() {
        for wind in [0.0f32, 0.7, 2.5] {
            let spectrum = build(0.28, wind);
            let (mean, ratio) = spread_of(spectrum.waves.iter().map(|wave| {
                let slope = wave.amplitude * wave.wave_number;
                (heading_off(wave.direction, wind), 0.5 * slope * slope)
            }));
            assert!(mean.abs() < 10.0, "wind {wind}: mean heading {mean} degrees off the wind");
            assert!((1.1..2.5).contains(&ratio), "wind {wind}: up/cross {ratio}");
        }
    }

    /// Crest drew each phase in the eighth of the circle its component's place
    /// in the octave picked, from the same seed as the wavelengths, so every
    /// octave started its waves at the same eight phases. Drawn uniformly from
    /// a stream of their own, no octave's phases follow its components'
    /// places.
    #[test]
    fn the_phases_are_not_tied_to_the_wavelengths() {
        let components = generate_components(1.0, 0.0);
        let tied = components
            .chunks(COMPONENTS_PER_OCTAVE)
            .filter(|octave| {
                octave.iter().enumerate().all(|(place, wave)| {
                    (wave.phase / TAU * COMPONENTS_PER_OCTAVE as f32) as usize == place
                })
            })
            .count();
        assert_eq!(tied, 0, "{tied} octaves start their waves at Crest's eight phases");
    }

    /// Cox and Munk measured the wind-dependent slope of the sea 3.16 up the
    /// wind to 1.92 across it; the floor of the spread gives its waves that
    /// ratio.
    #[test]
    fn the_spread_floor_gives_cox_and_munks_slope_ratio() {
        let steps = 20_000;
        let (mut up, mut across) = (0.0f64, 0.0f64);
        for step in 0..steps {
            let theta = std::f64::consts::PI * (2.0 * (step as f64 + 0.5) / steps as f64 - 1.0);
            let density = 1.0 / (f64::from(SPREAD_FLOOR) * theta).cosh().powi(2);
            up += density * theta.cos().powi(2);
            across += density * theta.sin().powi(2);
        }
        assert!((up / across - 3.16 / 1.92).abs() < 0.01, "up/cross {}", up / across);
    }

    /// The spread is narrowest just past the peak, broadens to either side
    /// down to the floor, and moves smoothly with the frequency, so a ripple
    /// field whose peak drifts with the wind never changes its spread along a
    /// line. Donelan's and Banner's fits step from 1.68 to 1.24 where they
    /// meet.
    #[test]
    fn the_spread_is_narrowest_past_the_peak_and_moves_smoothly() {
        assert!((spreading_beta(0.95 - 1e-4) - 2.44).abs() < 0.01);
        assert!((spreading_beta(1.3) - 1.92).abs() < 0.01);
        assert_eq!(spreading_beta(0.3), SPREAD_FLOOR);
        assert_eq!(spreading_beta(4.0), SPREAD_FLOOR);
        let mut previous = spreading_beta(0.2);
        for step in 1..=10_000 {
            let beta = spreading_beta(0.2 + step as f32 * 1e-3);
            assert!(beta >= SPREAD_FLOOR);
            assert!((beta - previous).abs() < 0.1, "a step of {} at {}", beta - previous, 0.2 + step as f32 * 1e-3);
            previous = beta;
        }
    }

    #[test]
    fn spread_angle_inverts_the_spread() {
        for beta in [SPREAD_FLOOR, 1.5, 2.4] {
            assert!(spread_angle(0.5, beta).abs() < 1e-6);
            for step in 1..100 {
                let u = step as f32 / 100.0;
                let theta = spread_angle(u, beta);
                assert!(theta.abs() < std::f32::consts::PI);
                assert!((spread_share(theta, beta) - u).abs() < 1e-4, "beta {beta}, u {u}: {theta}");
                assert!((spread_angle(1.0 - u, beta) + theta).abs() < 1e-4);
            }
        }
    }

    /// Where a ripple's crests lie must not change from one frame to the next,
    /// whatever the gusts, the fetch or the camera's maps do. The wind sea's
    /// components are a function of the wind's heading alone: the same heading
    /// gives the same set, bit for bit, and turning the wind turns every wave
    /// vector with it and leaves every phase and every heading off the wind
    /// as it was.
    #[test]
    fn the_wind_sea_depends_on_the_heading_alone() {
        let wind = 0.733f32;
        let set = wind_sea_components(wind);
        assert_eq!(set, wind_sea_components(wind));
        let turn = 1.1f32;
        let turned = wind_sea_components(wind + turn);
        for (component, after) in set.iter().zip(&turned) {
            assert_eq!(component[2], after[2]);
            assert_eq!(component[3], after[3]);
            let rotated = Vec2::from_angle(turn).rotate(Vec2::new(component[0], component[1]));
            let off = (rotated - Vec2::new(after[0], after[1])).abs().max_element();
            assert!(off <= 1.25, "{component:?} turned is {rotated}, not {after:?}");
            assert_eq!(after[0].fract(), 0.0);
            assert_eq!(after[1].fract(), 0.0);
        }
    }

    /// Two equal waves mirrored about the wind cross in a diamond lattice, the
    /// basket weave the wind sea used to be woven of, two crossing components
    /// a rung. No rung may hold such a pair.
    #[test]
    fn no_rung_of_the_wind_sea_holds_a_mirrored_pair() {
        let set = wind_sea_components(0.733);
        for (rung, components) in set.chunks(WIND_PER_RUNG).enumerate() {
            for (index, a) in components.iter().enumerate() {
                for b in &components[index + 1..] {
                    let mirror = (a[3] + b[3]).to_degrees().abs();
                    assert!(mirror >= 3.0, "rung {rung}: {} and {} degrees", a[3].to_degrees(), b[3].to_degrees());
                }
            }
        }
    }

    /// Every rung takes one heading from each quarter of the spread, the
    /// quarters turned on by the golden ratio from one rung to the next, and
    /// its wavelengths within a quarter octave of its own.
    #[test]
    fn every_rung_of_the_wind_sea_spans_its_spread() {
        let set = wind_sea_components(0.0);
        for (rung, components) in set.chunks(WIND_PER_RUNG).enumerate() {
            let mut quarters: Vec<usize> = components
                .iter()
                .map(|component| {
                    let share = spread_share(component[3], SPREAD_FLOOR) - rung as f32 * GOLDEN_RATIO_FRACTION;
                    ((share.rem_euclid(1.0) * WIND_PER_RUNG as f32) as usize).min(WIND_PER_RUNG - 1)
                })
                .collect();
            quarters.sort_unstable();
            assert_eq!(quarters, [0, 1, 2, 3], "rung {rung}");
            let nominal = WIND_SHORTEST_METRES * 2.0f32.powf(0.5 * rung as f32);
            for component in components {
                let wavelength = WIND_PERIOD_METRES / Vec2::new(component[0], component[1]).length();
                let octaves = (wavelength / nominal).log2();
                assert!(octaves.abs() < 0.26, "rung {rung}: {wavelength} m");
            }
        }
    }

    /// At the broadest spread every component of a rung carries the same
    /// slope: the set alone must head with the wind at Cox and Munk's ratio,
    /// a quarter or more of it within 20 degrees of the wind. The crossing
    /// pairs put none of it there and as much across the wind as up it.
    #[test]
    fn the_wind_sea_runs_with_the_wind() {
        let wind = 0.733f32;
        let set = wind_sea_components(wind);
        let headings: Vec<f32> = set.iter().map(|component| heading_off([component[0], component[1]], wind)).collect();
        let (mean, ratio) = spread_of(headings.iter().map(|&heading| (heading, 1.0)));
        assert!(mean.abs() < 10.0, "mean heading {mean} degrees off the wind");
        assert!((1.1..2.5).contains(&ratio), "up/cross {ratio}");
        let near = headings.iter().filter(|heading| heading.to_degrees().abs() < 20.0).count();
        assert!(near * 4 >= headings.len(), "{near} of {} within 20 degrees", headings.len());
    }
}
