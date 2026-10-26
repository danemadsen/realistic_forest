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
fn thunder_is_finite_and_normalised() {
    for muffled in [false, true] {
        let samples = synthesize_thunder(&column(800.0, 1200.0, 80), 11, muffled);
        assert!(!samples.is_empty());
        assert!(samples.iter().all(|sample| sample.is_finite()));
        let peak = samples
            .iter()
            .fold(0.0f32, |peak, sample| peak.max(sample.abs()));
        assert!((peak - 0.9).abs() < 1e-3);
        assert_eq!(*samples.last().unwrap(), 0.0, "the tail fades to silence");
    }
    assert!(synthesize_thunder(&[], 1, false).is_empty());
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
