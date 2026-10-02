//! Thunder built from the lightning channel itself. Sound from each piece of
//! the channel reaches the listener after its own travel time at the speed
//! of sound, so a near strike opens with a ripping crack and rolls on as the
//! distant upper channel arrives, while a far one is a low grumble whose high
//! frequencies the air has absorbed. Echoes off the ground and the cloud deck
//! stretch the tail. The samples are synthesised off the main thread when a
//! discharge begins and played once its first sound reaches the listener.

use crate::automation::AutomationSettings;
use crate::constants::AppSettings;
use crate::lightning::{ActiveBolt, BoltKind, BoltSegment};
use crate::player::Player;
use crate::weather::WeatherState;
use bevy::audio::{AudioPlayer, ChannelCount, Decodable, PlaybackSettings, SampleRate, Source, Volume};
use bevy::math::Vec3;
use bevy::prelude::*;
use bevy::tasks::{AsyncComputeTaskPool, Task, block_on, poll_once};
use std::num::NonZero;
use std::sync::Arc;
use std::time::Duration;

pub const THUNDER_SAMPLE_RATE: u32 = 44_100;
pub const SPEED_OF_SOUND: f32 = 343.0;
/// Even a channel spanning kilometres of range stops rolling within this.
const MAX_THUNDER_SECONDS: f32 = 14.0;
/// Echoes keep a strike rumbling this long after its last direct sound.
const TAIL_SECONDS: f32 = 4.0;
/// Energy arrivals are gathered per millisecond.
const SAMPLES_PER_BIN: usize = 44;

/// A piece of the channel as the listener hears it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ThunderSource {
    /// Metres from the listener.
    pub distance: f32,
    /// Relative acoustic energy: the piece's length times its current.
    pub energy: f32,
}

/// Every segment of a visible channel is a source of sound.
pub fn channel_sources(segments: &[BoltSegment], listener: [f32; 3]) -> Vec<ThunderSource> {
    let listener = Vec3::from(listener);
    segments
        .iter()
        .map(|segment| {
            let start = Vec3::from(segment.start);
            let end = Vec3::from(segment.end);
            ThunderSource {
                distance: ((start + end) * 0.5).distance(listener).max(1.0),
                energy: start.distance(end) * segment.brightness * segment.width,
            }
        })
        .collect()
}

/// A discharge hidden in the cloud still thunders: its unseen channel is
/// spread through the lower cloud above the flash.
pub fn cloud_sources(position: [f32; 3], cloud_base: f32, listener: [f32; 3], seed: f32) -> Vec<ThunderSource> {
    let mut random = ThunderRandom::new(seed.to_bits() as u64 ^ 0xC10D);
    let listener = Vec3::from(listener);
    (0..64)
        .map(|_| {
            let point = Vec3::new(
                position[0] + random.range(-900.0, 900.0),
                cloud_base + random.range(0.0, 900.0),
                position[2] + random.range(-900.0, 900.0),
            );
            ThunderSource { distance: point.distance(listener).max(1.0), energy: 40.0 }
        })
        .collect()
}

/// Seconds from the flash until the first sound arrives.
pub fn arrival_seconds(sources: &[ThunderSource]) -> f32 {
    sources.iter().map(|source| source.distance).fold(f32::INFINITY, f32::min) / SPEED_OF_SOUND
}

/// How loud a strike sounds at its nearest point: about full volume within a
/// few hundred metres, falling off roughly inversely with range.
pub fn loudness(sources: &[ThunderSource]) -> f32 {
    let nearest = sources.iter().map(|source| source.distance).fold(f32::INFINITY, f32::min);
    (700.0 / (nearest + 200.0)).clamp(0.08, 1.0)
}

struct ThunderRandom(u64);

impl ThunderRandom {
    fn new(seed: u64) -> Self {
        let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
        state = (state ^ (state >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        state = (state ^ (state >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        Self((state ^ (state >> 31)) | 1)
    }

    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }

    fn range(&mut self, low: f32, high: f32) -> f32 {
        low + (high - low) * self.next()
    }
}

/// One-pole smoothing with time constant `seconds`, run over per-millisecond bins.
fn smooth_bins(bins: &mut [f32], seconds: f32) {
    let keep = (-0.001 / seconds).exp();
    let mut state = 0.0;
    for bin in bins.iter_mut() {
        state = state * keep + *bin * (1.0 - keep);
        *bin = state;
    }
}

fn root_mean_square(samples: &[f32]) -> f32 {
    (samples.iter().map(|sample| sample * sample).sum::<f32>() / samples.len().max(1) as f32)
        .sqrt()
        .max(1e-9)
}

/// The thunder of `sources` as mono samples at [`THUNDER_SAMPLE_RATE`],
/// starting when the nearest source is heard. `muffled` thunder comes from
/// inside the cloud and has lost its crack. Peaks are normalised to 0.9.
pub fn synthesize_thunder(sources: &[ThunderSource], seed: u64, muffled: bool) -> Vec<f32> {
    if sources.is_empty() {
        return Vec::new();
    }
    let rate = THUNDER_SAMPLE_RATE as f32;
    let nearest = sources.iter().map(|source| source.distance).fold(f32::INFINITY, f32::min);
    let farthest = sources.iter().map(|source| source.distance).fold(0.0, f32::max);
    let spread = ((farthest - nearest) / SPEED_OF_SOUND).min(MAX_THUNDER_SECONDS - TAIL_SECONDS);
    let length = ((spread + TAIL_SECONDS) * rate) as usize;
    let bins = length / SAMPLES_PER_BIN + 1;

    // Energy arriving each millisecond: the low rumble, and the high
    // frequencies only the nearer parts of the channel still carry, since
    // air absorbs several decibels of them per kilometre.
    let mut low = vec![0.0f32; bins];
    let mut high = vec![0.0f32; bins];
    for source in sources {
        let bin = ((source.distance - nearest) / SPEED_OF_SOUND * 1000.0) as usize;
        if bin >= bins {
            continue;
        }
        let arriving = source.energy / (source.distance * source.distance);
        low[bin] += arriving;
        high[bin] += arriving * (-source.distance / 400.0).exp() * if muffled { 0.05 } else { 1.0 };
    }
    // The rumble from each piece rings for tens of milliseconds; the crack
    // is sharp. Echoes off the ground and the cloud deck add a rolling tail.
    smooth_bins(&mut low, 0.045);
    smooth_bins(&mut high, 0.012);
    let mut echo = low.clone();
    smooth_bins(&mut echo, 1.6);
    for (bin, echoed) in low.iter_mut().zip(echo) {
        *bin += echoed * 6.0;
    }

    // Shape white noise into the two bands, then normalise each band so the
    // envelopes alone decide their balance.
    let mut random = ThunderRandom::new(seed);
    let low_cutoff: f32 = if muffled { 70.0 } else { 110.0 };
    let low_keep = (-std::f32::consts::TAU * low_cutoff / rate).exp();
    let crack_keep = (-std::f32::consts::TAU * 1500.0 / rate).exp();
    let mut rumble = Vec::with_capacity(length);
    let mut crack = Vec::with_capacity(length);
    let (mut first, mut second, mut smooth) = (0.0f32, 0.0f32, 0.0f32);
    for _ in 0..length {
        let noise = random.range(-1.0, 1.0);
        first = first * low_keep + noise * (1.0 - low_keep);
        second = second * low_keep + first * (1.0 - low_keep);
        smooth = smooth * crack_keep + noise * (1.0 - crack_keep);
        rumble.push(second);
        crack.push(noise - smooth);
    }
    let rumble_gain = 1.0 / root_mean_square(&rumble);
    let crack_gain = 1.0 / root_mean_square(&crack);

    // The rumble rolls: its loudness wanders a few times a second as
    // different stretches of the channel arrive in and out of step.
    let mut roll_phase = random.range(0.0, std::f32::consts::TAU);
    let mut roll_rate = random.range(3.0, 8.0);
    let mut samples = Vec::with_capacity(length);
    for index in 0..length {
        let position = index as f32 / SAMPLES_PER_BIN as f32;
        let bin = (position as usize).min(bins - 1);
        let next = (bin + 1).min(bins - 1);
        let fraction = position - bin as f32;
        let low_level = (low[bin] + (low[next] - low[bin]) * fraction).max(0.0).sqrt();
        let high_level = (high[bin] + (high[next] - high[bin]) * fraction).max(0.0).sqrt();
        if index % 4410 == 0 {
            roll_rate = (roll_rate + random.range(-1.5, 1.5)).clamp(2.5, 9.0);
        }
        roll_phase += std::f32::consts::TAU * roll_rate / rate;
        let roll = 0.7 + 0.3 * roll_phase.sin();
        samples.push(rumble[index] * rumble_gain * low_level * roll
            + crack[index] * crack_gain * high_level * 0.6);
    }

    // A few milliseconds of fade-in avoid a click; the end fades to silence.
    let fade_in = (0.003 * rate) as usize;
    let fade_out = ((0.4 * rate) as usize).min(length);
    for index in 0..fade_in.min(length) {
        samples[index] *= index as f32 / fade_in as f32;
    }
    for index in 0..fade_out {
        samples[length - 1 - index] *= index as f32 / fade_out as f32;
    }
    let peak = samples.iter().fold(0.0f32, |peak, sample| peak.max(sample.abs()));
    if peak > 0.0 {
        let gain = 0.9 / peak;
        samples.iter_mut().for_each(|sample| *sample *= gain);
    }
    samples
}

/// Synthesised thunder, played through bevy_audio as a custom source so no
/// audio file format is needed.
#[derive(Asset, TypePath, Clone)]
pub struct ThunderSound {
    samples: Arc<[f32]>,
}

pub struct ThunderDecoder {
    samples: Arc<[f32]>,
    position: usize,
}

impl Iterator for ThunderDecoder {
    type Item = f32;

    fn next(&mut self) -> Option<f32> {
        let sample = self.samples.get(self.position).copied();
        self.position += 1;
        sample
    }
}

impl Source for ThunderDecoder {
    fn current_span_len(&self) -> Option<usize> {
        if self.position >= self.samples.len() { Some(0) } else { Some(self.samples.len()) }
    }

    fn channels(&self) -> ChannelCount {
        NonZero::new(1).unwrap()
    }

    fn sample_rate(&self) -> SampleRate {
        NonZero::new(THUNDER_SAMPLE_RATE).unwrap()
    }

    fn total_duration(&self) -> Option<Duration> {
        Some(Duration::from_secs_f32(self.samples.len() as f32 / THUNDER_SAMPLE_RATE as f32))
    }
}

impl Decodable for ThunderSound {
    type Decoder = ThunderDecoder;

    fn decoder(&self) -> Self::Decoder {
        ThunderDecoder { samples: self.samples.clone(), position: 0 }
    }
}

struct PendingThunder {
    task: Task<Vec<f32>>,
    samples: Option<Vec<f32>>,
    /// Real-time seconds at which the first sound arrives.
    start_at: f64,
    loudness: f32,
}

/// Thunder being synthesised or waiting for its sound to arrive.
#[derive(Resource, Default)]
pub struct ThunderQueue {
    pending: Vec<PendingThunder>,
    generation: u64,
}

/// Starts synthesising the thunder of each new discharge. Screenshot runs
/// stay silent.
pub fn queue_thunder(
    time: Res<Time<Real>>,
    automation: Res<AutomationSettings>,
    bolt: Res<ActiveBolt>,
    weather: Res<WeatherState>,
    players: Query<&Player>,
    mut queue: ResMut<ThunderQueue>,
) {
    if bolt.generation == queue.generation {
        return;
    }
    queue.generation = bolt.generation;
    if automation.shot_path.is_some() {
        return;
    }
    let Ok(player) = players.single() else {
        return;
    };
    let listener = player.position.to_array();
    let lightning = weather.lightning;
    let (sources, muffled) = match bolt.kind {
        BoltKind::Hidden => (
            cloud_sources(lightning.position, lightning.cloud_base, listener, lightning.seed),
            true,
        ),
        BoltKind::Ground | BoltKind::Crawler => (channel_sources(&bolt.segments, listener), false),
    };
    if sources.is_empty() {
        return;
    }
    // The flash came this long ago; sound left the channel at the same time.
    let since_flash = (lightning.age_seconds - lightning.first_stroke_seconds()).max(0.0);
    let start_at = time.elapsed_secs_f64() + (arrival_seconds(&sources) - since_flash).max(0.0) as f64;
    let loudness = loudness(&sources);
    let seed = bolt.generation ^ (lightning.seed.to_bits() as u64) << 20;
    let task = AsyncComputeTaskPool::get().spawn(async move { synthesize_thunder(&sources, seed, muffled) });
    queue.pending.push(PendingThunder { task, samples: None, start_at, loudness });
}

/// Plays each synthesised thunder once its sound reaches the listener.
pub fn play_thunder(
    mut commands: Commands,
    time: Res<Time<Real>>,
    settings: Res<AppSettings>,
    mut queue: ResMut<ThunderQueue>,
    mut sounds: ResMut<Assets<ThunderSound>>,
) {
    let now = time.elapsed_secs_f64();
    let volume = settings.thunder_volume.clamp(0.0, 1.0);
    queue.pending.retain_mut(|pending| {
        if pending.samples.is_none() {
            pending.samples = block_on(poll_once(&mut pending.task));
        }
        if now < pending.start_at {
            return true;
        }
        let Some(samples) = pending.samples.take() else {
            // Still synthesising; play as soon as it is ready.
            return true;
        };
        if volume > 0.0 && !samples.is_empty() {
            let sound = sounds.add(ThunderSound { samples: samples.into() });
            commands.spawn((
                AudioPlayer::<ThunderSound>(sound),
                PlaybackSettings::DESPAWN.with_volume(Volume::Linear(volume * pending.loudness)),
            ));
        }
        false
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn column(base_distance: f32, height: f32, pieces: usize) -> Vec<ThunderSource> {
        (0..pieces)
            .map(|index| {
                let up = height * index as f32 / pieces as f32;
                ThunderSource { distance: base_distance.hypot(up), energy: 12.0 }
            })
            .collect()
    }

    /// The share of the signal's energy in its sample-to-sample changes, a
    /// simple measure of how much high frequency it carries.
    fn brightness(samples: &[f32]) -> f32 {
        let total: f32 = samples.iter().map(|sample| sample * sample).sum();
        let changes: f32 = samples.windows(2).map(|pair| (pair[1] - pair[0]).powi(2)).sum();
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
        assert!(brightness(&near[..opening]) > 3.0 * brightness(&far[..opening]),
                "near {} far {}", brightness(&near[..opening]), brightness(&far[..opening]));
        let muffled = synthesize_thunder(&column(300.0, 1200.0, 60), 7, true);
        assert!(brightness(&muffled[..opening]) < brightness(&near[..opening]));
    }

    #[test]
    fn a_longer_channel_rolls_for_longer() {
        let (short_sources, long_sources) = (column(1500.0, 300.0, 40), column(1500.0, 2500.0, 40));
        let short = synthesize_thunder(&short_sources, 3, false);
        let long = synthesize_thunder(&long_sources, 3, false);
        let rate = THUNDER_SAMPLE_RATE as f32;
        let farthest = |sources: &[ThunderSource]| sources.iter().map(|source| source.distance).fold(0.0, f32::max);
        let spread = (farthest(&long_sources) - farthest(&short_sources)) / SPEED_OF_SOUND;
        let difference = (long.len() as f32 - short.len() as f32) / rate;
        assert!((difference - spread).abs() < 0.05, "{difference} vs {spread}");
    }

    #[test]
    fn thunder_is_finite_and_normalised() {
        for muffled in [false, true] {
            let samples = synthesize_thunder(&column(800.0, 1200.0, 80), 11, muffled);
            assert!(!samples.is_empty());
            assert!(samples.iter().all(|sample| sample.is_finite()));
            let peak = samples.iter().fold(0.0f32, |peak, sample| peak.max(sample.abs()));
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
        assert!(sources.iter().all(|source| source.energy >= 0.0 && source.distance >= 1.0));
    }
}
