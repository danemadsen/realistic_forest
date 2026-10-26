use super::*;

fn column(base_distance: f32, height: f32, pieces: usize) -> Vec<ThunderSource> {
    (0..pieces)
        .map(|index| {
            let up = height * index as f32 / pieces as f32;
            ThunderSource {
                distance: base_distance.hypot(up),
                energy: 12.0,
            }
        })
        .collect()
}

/// The share of the signal's energy in its sample-to-sample changes, a
/// simple measure of how much high frequency it carries.
fn brightness(samples: &[f32]) -> f32 {
    let total: f32 = samples.iter().map(|sample| sample * sample).sum();
    let changes: f32 = samples
        .windows(2)
        .map(|pair| (pair[1] - pair[0]).powi(2))
        .sum();
    changes / total.max(1e-9)
}

fn energy(samples: &[f32]) -> f64 {
    samples
        .iter()
        .map(|sample| f64::from(*sample).powi(2))
        .sum()
}

/// Two-pole analysis filters measure energy in a broad frequency range.
fn band_energy(samples: &[f32], low_frequency: f32, high_frequency: f32) -> f64 {
    let rate = THUNDER_SAMPLE_RATE as f32;
    let low_keep = (-std::f32::consts::TAU * high_frequency / rate).exp();
    let high_keep = (-std::f32::consts::TAU * low_frequency / rate).exp();
    let mut low_first = 0.0;
    let mut low_second = 0.0;
    let mut high_first = 0.0;
    let mut high_second = 0.0;
    let mut total = 0.0;

    for sample in samples {
        low_first = low_first * low_keep + sample * (1.0 - low_keep);
        low_second = low_second * low_keep + low_first * (1.0 - low_keep);
        high_first = high_first * high_keep + low_second * (1.0 - high_keep);
        let high_passed = low_second - high_first;
        high_second = high_second * high_keep + high_passed * (1.0 - high_keep);
        let middle = high_passed - high_second;
        total += f64::from(middle).powi(2);
    }

    total
}

fn sine_wave(frequency: f32) -> Vec<f32> {
    let rate = THUNDER_SAMPLE_RATE as f32;
    (0..THUNDER_SAMPLE_RATE)
        .map(|index| {
            let phase = std::f32::consts::TAU * frequency * index as f32 / rate;
            phase.sin() * 0.05
        })
        .collect()
}

#[test]
fn bass_shelf_adds_weight_without_dulling_the_crack() {
    for (frequency, minimum_db, maximum_db) in [
        (55.0, 7.5, 8.25),
        (170.0, 3.7, 4.3),
        (1000.0, -0.1, 0.1),
        (4000.0, -0.1, 0.1),
    ] {
        let original = sine_wave(frequency);
        let mut boosted = original.clone();
        boost_thunder_bass(&mut boosted, 8.0);

        // Exclude filter startup and the endpoint fade from steady-state gain.
        let from = THUNDER_SAMPLE_RATE as usize / 5;
        let to = THUNDER_SAMPLE_RATE as usize * 4 / 5;
        let gain_db = 10.0 * (energy(&boosted[from..to]) / energy(&original[from..to])).log10();
        assert!(
            (minimum_db..=maximum_db).contains(&gain_db),
            "{frequency} Hz must retain the intended bass/crack balance: {gain_db} dB"
        );
    }
}

#[test]
fn zero_and_invalid_bass_gains_preserve_the_original_samples() {
    let original = [0.0, 0.25, -0.4, 0.1, -0.05];
    for gain in [0.0, -8.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut samples = original;
        boost_thunder_bass(&mut samples, gain);
        assert_eq!(samples, original, "gain {gain} must bypass the bass filter");
    }

    let mut silence = Vec::new();
    boost_thunder_bass(&mut silence, 12.0);
    assert!(silence.is_empty(), "an empty source must remain silent");
}

#[test]
fn default_bass_adds_low_end_energy_to_different_procedural_claps() {
    let sources = column(250.0, 900.0, 30);
    let opening = THUNDER_SAMPLE_RATE as usize;
    for seed in [7, 19, 37] {
        let original = synthesize_thunder_with_bass(&sources, seed, false, 0.0);
        let boosted = synthesize_thunder(&sources, seed, false);
        let original_bass = band_energy(&original[..opening], 40.0, 160.0);
        let boosted_bass = band_energy(&boosted[..opening], 40.0, 160.0);
        let original_crack = band_energy(&original[..opening], 1000.0, 6000.0);
        let boosted_crack = band_energy(&boosted[..opening], 1000.0, 6000.0);

        assert!(
            boosted_bass > original_bass * 2.5,
            "seed {seed}: bass must gain substantial energy without peak normalization: original {original_bass}, boosted {boosted_bass}"
        );
        let crack_ratio = boosted_crack / original_crack;
        assert!(
            (0.75..=1.3).contains(&crack_ratio),
            "seed {seed}: the crack must keep its energy as bass grows: ratio {crack_ratio}"
        );
    }
}

#[test]
fn strong_bass_and_overlapping_restrikes_stay_finite_and_bounded() {
    let sources = column(100.0, 900.0, 30);
    let strokes = [
        (0.0, 1.0),
        (0.06, 0.7),
        (0.12, 0.6),
        (0.2, 0.5),
        (0.28, 0.4),
    ];
    for bass_db in [8.0, 12.0] {
        for seed in [3, 11] {
            let samples = synthesize_discharge(&sources, &strokes, seed, false, bass_db);
            assert!(
                samples
                    .iter()
                    .all(|sample| sample.is_finite() && sample.abs() < 1.0),
                "seed {seed}, boost {bass_db} dB: overlapping booms must not clip or become non-finite"
            );
            assert_eq!(
                samples[0], 0.0,
                "a boosted strike must start without a click"
            );
            assert_eq!(
                samples.last().copied(),
                Some(0.0),
                "a boosted strike must finish in silence"
            );
        }
    }
}

#[test]
fn sound_arrives_after_its_travel_time() {
    let sources = column(1715.0, 1200.0, 40);
    assert!((arrival_seconds(&sources) - 5.0).abs() < 1e-3);
    assert!(loudness(&column(300.0, 1000.0, 10)) > loudness(&sources));
}

#[test]
fn near_strikes_crack_and_far_strikes_rumble() {
    let near = synthesize_thunder(&column(300.0, 1200.0, 60), 7, false);
    let far = synthesize_thunder(&column(4000.0, 1200.0, 60), 7, false);
    let opening = (0.5 * THUNDER_SAMPLE_RATE as f32) as usize;
    assert!(
        brightness(&near[..opening]) > 3.0 * brightness(&far[..opening]),
        "near {} far {}",
        brightness(&near[..opening]),
        brightness(&far[..opening])
    );
    let muffled = synthesize_thunder(&column(300.0, 1200.0, 60), 7, true);
    assert!(brightness(&muffled[..opening]) < brightness(&near[..opening]));
}

#[test]
fn a_longer_channel_rolls_for_longer() {
    let (short_sources, long_sources) = (column(1500.0, 300.0, 40), column(1500.0, 2500.0, 40));
    let short = synthesize_thunder(&short_sources, 3, false);
    let long = synthesize_thunder(&long_sources, 3, false);
    let rate = THUNDER_SAMPLE_RATE as f32;
    let farthest = |sources: &[ThunderSource]| {
        sources
            .iter()
            .map(|source| source.distance)
            .fold(0.0, f32::max)
    };
    let spread = (farthest(&long_sources) - farthest(&short_sources)) / SPEED_OF_SOUND;
    let difference = (long.len() as f32 - short.len() as f32) / rate;
    assert!(
        (difference - spread).abs() < 0.05,
        "{difference} vs {spread}"
    );
}

#[test]
fn thunder_is_finite_bounded_and_click_free() {
    for muffled in [false, true] {
        let samples = synthesize_thunder(&column(800.0, 1200.0, 80), 11, muffled);
        assert!(!samples.is_empty());
        assert!(samples.iter().all(|sample| sample.is_finite()));
        let peak = samples
            .iter()
            .fold(0.0f32, |peak, sample| peak.max(sample.abs()));
        assert!(
            peak > 0.0 && peak < 1.0,
            "the original dynamics must remain audible without clipping: peak {peak}"
        );
        assert_eq!(samples[0], 0.0, "the onset starts without a click");
        assert_eq!(*samples.last().unwrap(), 0.0, "the tail fades to silence");

        let mean =
            samples.iter().map(|sample| f64::from(*sample)).sum::<f64>() / samples.len() as f64;
        let rms = (energy(&samples) / samples.len() as f64).sqrt();
        assert!(
            mean.abs() < rms * 0.01,
            "the waveform must not carry a DC offset: mean {mean}, rms {rms}"
        );
    }
    assert!(synthesize_thunder(&[], 1, false).is_empty());
}

#[test]
fn nearby_thunder_preserves_the_reference_clap() {
    let sources = [ThunderSource {
        distance: 250.0,
        energy: 12.0,
    }];
    let rendered = synthesize_thunder_with_bass(&sources, 7, false, 0.0);
    let reference_wav = thunder_synth::wav(thunder_synth::Thunder::Clap, 7);
    let (frames, _) = reference_wav[44..].as_chunks::<2>();
    let reference_samples: Vec<_> = frames
        .iter()
        .map(|bytes| f32::from(i16::from_le_bytes([bytes[0], bytes[1]])) / 32768.0)
        .collect();
    let preroll = (0.05 * THUNDER_SAMPLE_RATE as f32) as usize;
    let opening = (0.6 * THUNDER_SAMPLE_RATE as f32) as usize;
    let reference_opening = &reference_samples[preroll..preroll + opening];
    let rendered_opening = &rendered[..opening];
    let product: f64 = rendered_opening
        .iter()
        .zip(reference_opening)
        .map(|(actual, reference)| f64::from(*actual) * f64::from(*reference))
        .sum();
    let reference_energy = energy(reference_opening);
    let rendered_energy = energy(rendered_opening);
    let correlation = product / (reference_energy * rendered_energy).sqrt();

    assert!(
        correlation > 0.95,
        "the close crack must retain the reference's actual texture and decay: correlation {correlation}"
    );
    assert_eq!(
        THUNDER_SAMPLE_RATE,
        thunder_synth::SAMPLE_RATE,
        "play generated samples at the reference's native rate"
    );
}

#[test]
fn distant_rolls_keep_their_lower_playback_levels() {
    let near_sources = column(250.0, 100.0, 12);
    let far_sources = column(4500.0, 100.0, 12);
    for seed in [11, 23, 37] {
        let near = synthesize_thunder(&near_sources, seed, false);
        let far = synthesize_thunder(&far_sources, seed, false);
        let near_peak = near
            .iter()
            .fold(0.0_f32, |peak, sample| peak.max(sample.abs()));
        let far_peak = far
            .iter()
            .fold(0.0_f32, |peak, sample| peak.max(sample.abs()));

        let near_playback_peak = near_peak * loudness(&near_sources);
        let far_playback_peak = far_peak * loudness(&far_sources);
        assert!(
            near_playback_peak > far_playback_peak * 3.0,
            "seed {seed}: distant thunder must stay quieter after bass processing and distance attenuation: near {near_playback_peak}, far {far_playback_peak}"
        );
    }
}

#[test]
fn thunder_is_repeatable_without_reusing_the_same_sound() {
    let sources = column(900.0, 400.0, 24);
    let first = synthesize_thunder(&sources, 17, false);
    let repeated = synthesize_thunder(&sources, 17, false);
    let different = synthesize_thunder(&sources, 18, false);

    assert_eq!(
        first, repeated,
        "one seed must reproduce the same discharge"
    );
    assert_ne!(
        first, different,
        "another seed must create a fresh waveform"
    );
}

#[test]
fn silent_or_invalid_sources_do_not_create_a_normalised_noise_burst() {
    let invalid_sources = [
        ThunderSource {
            distance: 500.0,
            energy: 0.0,
        },
        ThunderSource {
            distance: f32::NAN,
            energy: 12.0,
        },
        ThunderSource {
            distance: f32::INFINITY,
            energy: 12.0,
        },
        ThunderSource {
            distance: 0.0,
            energy: 12.0,
        },
        ThunderSource {
            distance: -500.0,
            energy: 12.0,
        },
        ThunderSource {
            distance: 500.0,
            energy: f32::NAN,
        },
        ThunderSource {
            distance: 500.0,
            energy: f32::INFINITY,
        },
        ThunderSource {
            distance: 500.0,
            energy: -12.0,
        },
    ];

    for source in invalid_sources {
        let samples = synthesize_thunder(&[source], 5, false);
        assert!(
            samples.iter().all(|sample| *sample == 0.0),
            "an inactive source must stay silent: {source:?}"
        );
    }

    let mut mixed_sources = invalid_sources.to_vec();
    mixed_sources.push(ThunderSource {
        distance: 500.0,
        energy: 12.0,
    });
    let samples = synthesize_thunder(&mixed_sources, 5, false);
    assert!(
        samples.iter().all(|sample| sample.is_finite()),
        "invalid sources must not poison an otherwise audible discharge"
    );
    assert!(
        energy(&samples) > 0.0,
        "the valid source must remain audible"
    );
}

#[test]
fn nearby_thunder_has_audible_body_between_bass_and_crack() {
    let sources = column(450.0, 900.0, 60);
    let samples = synthesize_thunder(&sources, 23, false);
    let opening_length = THUNDER_SAMPLE_RATE as usize;
    let opening = &samples[..opening_length];
    let body_share = band_energy(opening, 100.0, 1200.0) / energy(opening);

    assert!(
        body_share > 0.08,
        "thunder needs midrange body as well as deep bass and high hiss: share {body_share}"
    );
}

#[test]
fn restrikes_extend_the_roll_and_add_a_subordinate_crack() {
    let sources = column(900.0, 100.0, 12);
    let single_stroke = [(0.0, 1.0)];
    let repeated_strokes = [(0.0, 1.0), (1.5, 0.8)];
    let single = synthesize_discharge(&sources, &single_stroke, 31, false, 0.0);
    let repeated = synthesize_discharge(&sources, &repeated_strokes, 31, false, 0.0);
    let extra_seconds = (repeated.len() - single.len()) as f32 / THUNDER_SAMPLE_RATE as f32;

    assert!(
        (extra_seconds - 1.5).abs() < 0.05,
        "the final stroke's travel time must extend the sound: {extra_seconds} seconds"
    );

    let window_start = (1.5 * THUNDER_SAMPLE_RATE as f32) as usize;
    let window_end = (2.2 * THUNDER_SAMPLE_RATE as f32) as usize;
    let single_window = &single[window_start..window_end];
    let repeated_window = &repeated[window_start..window_end];
    let difference: Vec<_> = repeated_window
        .iter()
        .zip(single_window)
        .map(|(repeated, single)| repeated - single)
        .collect();
    let added_energy = energy(&difference);
    let initial_end = (0.7 * THUNDER_SAMPLE_RATE as f32) as usize;
    let first_clap_energy = energy(&single[..initial_end]);
    assert!(
        added_energy > first_clap_energy * 0.001 && added_energy < first_clap_energy * 0.25,
        "a later stroke must add an audible crack while preserving the dominant first clap: added {added_energy}, first {first_clap_energy}"
    );
    assert!(
        synthesize_discharge(&sources, &[], 31, false, 0.0).is_empty(),
        "a discharge without return strokes must be silent"
    );
}

#[test]
fn channel_sources_follow_the_bolt() {
    let segments = crate::lightning::ground_bolt(0.4, [600.0, 10.0, 0.0], [700.0, 1300.0, 50.0]);
    let sources = channel_sources(&segments, [0.0, 20.0, 0.0]);
    assert_eq!(sources.len(), segments.len());
    assert!((arrival_seconds(&sources) - 600.0 / SPEED_OF_SOUND).abs() < 0.3);
    assert!(
        sources
            .iter()
            .all(|source| source.energy >= 0.0 && source.distance >= 1.0)
    );
}
