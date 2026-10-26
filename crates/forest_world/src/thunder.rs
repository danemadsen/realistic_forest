//! Procedural thunder using thunder-synth's statistically shaped claps and rolls.
//! Channel distance controls sound propagation and the balance of close cracks
//! and distant rolls; its range spread controls the rolling duration. Rendering
//! runs off the main thread. Every discharge is generated from fresh noise,
//! without recorded audio. Adjustable bass weight strengthens the boom while
//! preserving the reference synthesizer's crack and irregular rolling texture.

use crate::automation::AutomationSettings;
use crate::lightning::{ActiveBolt, BoltKind, BoltSegment};
use crate::player::Player;
use crate::weather::WeatherState;
use bevy::audio::{
    AudioPlayer, ChannelCount, Decodable, PlaybackSettings, SampleRate, Source, Volume,
};
use bevy::math::Vec3;
use bevy::prelude::*;
use bevy::tasks::{AsyncComputeTaskPool, Task, block_on, poll_once};
use forest_core::constants::{AppSettings, DEFAULT_THUNDER_BASS_DB, MAX_THUNDER_BASS_DB};
use std::num::NonZero;
use std::sync::Arc;
use std::time::Duration;

/// The reference generator's native rate; the audio backend resamples for playback.
pub const THUNDER_SAMPLE_RATE: u32 = thunder_synth::SAMPLE_RATE;
/// Approximate sound speed in metres per second in mild outdoor air.
pub const SPEED_OF_SOUND: f32 = 343.0;
/// Bound rendering work even for channels with an unusually large range spread.
const MAX_THUNDER_SECONDS: f32 = 18.0;
/// Leave the reference clap's complete three-second decay after the last arrival.
const TAIL_SECONDS: f32 = thunder_synth::CLAP_SECS;
/// Upstream claps have 50 ms of silence before their audible attack.
const CLAP_PREROLL_SECONDS: f32 = 0.05;
/// Later return strokes add a short crack, keeping the first clap's full tail.
const RESTRIKE_SECONDS: f32 = 0.18;
/// Restrikes stay beneath the main clap and taper further with each stroke.
const RESTRIKE_GAIN: f32 = 0.22;
/// Match the reference roll's fade when the final roll ends between clip boundaries.
const ROLL_FADE_SECONDS: f32 = 0.8;
/// Shelf midpoint: lifts deep bass and the low body without raising the crack.
const THUNDER_BASS_FREQUENCY: f64 = 170.0;

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
pub fn cloud_sources(
    position: [f32; 3],
    cloud_base: f32,
    listener: [f32; 3],
    seed: f32,
) -> Vec<ThunderSource> {
    let mut random = ThunderRandom::new(seed.to_bits() as u64 ^ 0xC10D);
    let listener = Vec3::from(listener);
    (0..64)
        .map(|_| {
            let point = Vec3::new(
                position[0] + random.range(-900.0, 900.0),
                cloud_base + random.range(0.0, 900.0),
                position[2] + random.range(-900.0, 900.0),
            );
            ThunderSource {
                distance: point.distance(listener).max(1.0),
                energy: 40.0,
            }
        })
        .collect()
}

/// Seconds from the flash until the first sound arrives.
pub fn arrival_seconds(sources: &[ThunderSource]) -> f32 {
    sources
        .iter()
        .map(|source| source.distance)
        .fold(f32::INFINITY, f32::min)
        / SPEED_OF_SOUND
}

/// How loud a strike sounds at its nearest point: about full volume within a
/// few hundred metres, falling off roughly inversely with range.
pub fn loudness(sources: &[ThunderSource]) -> f32 {
    let nearest = sources
        .iter()
        .map(|source| source.distance)
        .fold(f32::INFINITY, f32::min);
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

/// Render fresh noise through the unchanged upstream synthesizer. Its public API
/// guarantees a 44-byte header followed by mono, little-endian 16-bit PCM.
fn render_reference(kind: thunder_synth::Thunder, seed: u64) -> Vec<f32> {
    let generated_wav = thunder_synth::wav(kind, seed);
    let (frames, _) = generated_wav[44..].as_chunks::<2>();
    frames
        .iter()
        .map(|bytes| {
            let encoded = i16::from_le_bytes([bytes[0], bytes[1]]);
            f32::from(encoded) / 32768.0
        })
        .collect()
}

fn lowpass(samples: &mut [f32], cutoff: f32) {
    let keep = (-std::f32::consts::TAU * cutoff / THUNDER_SAMPLE_RATE as f32).exp();
    let (mut first, mut second) = (0.0, 0.0);
    for sample in samples {
        first = first * keep + *sample * (1.0 - keep);
        second = second * keep + first * (1.0 - keep);
        *sample = second;
    }
}

fn boost_thunder_bass(samples: &mut [f32], bass_db: f32) {
    if !bass_db.is_finite() || bass_db <= 0.0 || samples.is_empty() {
        return;
    }
    let bass_db = bass_db.min(MAX_THUNDER_BASS_DB);

    // RBJ low shelf, slope S = 1: https://www.w3.org/TR/audio-eq-cookbook/.
    // Double precision keeps the low-frequency poles accurate at audio rates.
    let amplitude = 10.0_f64.powf(f64::from(bass_db) / 40.0);
    let angle = std::f64::consts::TAU * THUNDER_BASS_FREQUENCY / f64::from(THUNDER_SAMPLE_RATE);
    let cosine = angle.cos();
    let alpha = angle.sin() / std::f64::consts::SQRT_2;
    let shelf_cross = 2.0 * amplitude.sqrt() * alpha;
    let amplitude_plus_one = amplitude + 1.0;
    let amplitude_minus_one = amplitude - 1.0;
    let normalizer = amplitude_plus_one + amplitude_minus_one * cosine + shelf_cross;

    let numerator = [
        amplitude * (amplitude_plus_one - amplitude_minus_one * cosine + shelf_cross),
        2.0 * amplitude * (amplitude_minus_one - amplitude_plus_one * cosine),
        amplitude * (amplitude_plus_one - amplitude_minus_one * cosine - shelf_cross),
    ];
    let denominator = [
        -2.0 * (amplitude_minus_one + amplitude_plus_one * cosine),
        amplitude_plus_one + amplitude_minus_one * cosine - shelf_cross,
    ];
    let feed = numerator.map(|coefficient| coefficient / normalizer);
    let feedback = denominator.map(|coefficient| coefficient / normalizer);

    let (mut delay_one, mut delay_two) = (0.0, 0.0);
    for sample in samples.iter_mut() {
        let input = f64::from(*sample);
        let output = feed[0] * input + delay_one;
        delay_one = feed[1] * input - feedback[0] * output + delay_two;
        delay_two = feed[2] * input - feedback[1] * output;
        *sample = output as f32;
    }

    // The shelf's filter state can outlast the source's final silent sample.
    let fade = (0.005 * THUNDER_SAMPLE_RATE as f32) as usize;
    let fade = fade.min(samples.len());
    for (index, sample) in samples.iter_mut().rev().take(fade).enumerate() {
        *sample *= index as f32 / fade as f32;
    }
}

fn add_rolls(samples: &mut [f32], seed: u64, proximity: f32, muffled: bool) {
    let rate = THUNDER_SAMPLE_RATE as f32;
    let overlap = (thunder_synth::ROLL_OVERLAP * rate) as usize;
    let fade = ROLL_FADE_SECONDS * rate;
    let roll_gain = 0.15 + 0.85 * (1.0 - proximity);
    let length = samples.len();
    let (mut start, mut index) = (0, 0_u64);
    while start < length {
        let roll_seed = seed.wrapping_add(index.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let mut roll = render_reference(thunder_synth::Thunder::Roll, roll_seed);
        if muffled {
            lowpass(&mut roll, 900.0);
        }
        for (offset, value) in roll.iter().take(length - start).enumerate() {
            let position = start + offset;
            let seconds = position as f32 / rate;
            // Give the initial clap room, as in the reference's clap-over-roll mix.
            let duck = 1.0 - 0.6 * proximity * (-seconds / 0.6).exp();
            let ending = ((length - 1 - position) as f32 / fade).min(1.0);
            samples[position] += value * roll_gain * duck * ending;
        }
        start += roll.len() - overlap;
        index += 1;
    }
}

fn add_claps(
    samples: &mut [f32],
    strokes: &[(f32, f32)],
    seed: u64,
    distance: f32,
    proximity: f32,
) {
    if proximity < 0.001 {
        return;
    }
    let rate = THUNDER_SAMPLE_RATE as f32;
    let preroll = (CLAP_PREROLL_SECONDS * rate) as usize;
    for (index, &(seconds, energy)) in strokes.iter().enumerate() {
        let clap_seed = seed.wrapping_add((index as u64).wrapping_mul(0xD1B5_4A32_D192_ED03));
        let mut clap = render_reference(thunder_synth::Thunder::Clap, clap_seed);
        if distance > 500.0 {
            let range = (distance - 500.0) / 1500.0;
            let cutoff = 12_000.0 / (1.0 + range * range);
            lowpass(&mut clap, cutoff);
        }
        let start = (seconds * rate) as usize;
        let available = samples.len().saturating_sub(start);
        let (duration, gain) = if index == 0 {
            (clap.len() - preroll, proximity * energy.sqrt())
        } else {
            let duration = (RESTRIKE_SECONDS * rate) as usize;
            let gain = proximity * energy.sqrt() * RESTRIKE_GAIN / (index as f32).sqrt();
            (duration, gain)
        };
        let count = duration.min(available);
        let attack = 0.001 * rate;
        let release = 0.05 * rate;
        for offset in 0..count {
            let onset = (offset as f32 / attack).min(1.0);
            let ending = if index == 0 {
                1.0
            } else {
                ((duration - 1 - offset) as f32 / release).min(1.0)
            };
            samples[start + offset] += clap[preroll + offset] * gain * onset * ending;
        }
    }
}

/// The thunder of one return stroke as mono samples at [`THUNDER_SAMPLE_RATE`].
/// Empty or inactive sources are silent. The default bass boost adds low-end
/// weight, with a soft limiter for overlapping peals.
pub fn synthesize_thunder(sources: &[ThunderSource], seed: u64, muffled: bool) -> Vec<f32> {
    synthesize_thunder_with_bass(sources, seed, muffled, DEFAULT_THUNDER_BASS_DB)
}

/// Render procedural thunder with a low-frequency boost in decibels.
/// Zero preserves the original bass balance; boosts are capped at 12 dB.
/// Negative and non-finite boosts bypass the equalizer.
pub fn synthesize_thunder_with_bass(
    sources: &[ThunderSource],
    seed: u64,
    muffled: bool,
    bass_db: f32,
) -> Vec<f32> {
    synthesize_discharge(sources, &[(0.0, 1.0)], seed, muffled, bass_db)
}

fn synthesize_discharge(
    sources: &[ThunderSource],
    strokes: &[(f32, f32)],
    seed: u64,
    muffled: bool,
    bass_db: f32,
) -> Vec<f32> {
    let sources: Vec<_> = sources
        .iter()
        .copied()
        .filter(|source| {
            source.distance.is_finite()
                && source.distance > 0.0
                && source.energy.is_finite()
                && source.energy > 0.0
        })
        .collect();
    let strokes: Vec<_> = strokes
        .iter()
        .copied()
        .filter(|(time, energy)| {
            time.is_finite()
                && *time >= 0.0
                && *time <= MAX_THUNDER_SECONDS - TAIL_SECONDS
                && energy.is_finite()
                && *energy > 0.0
        })
        .collect();
    if sources.is_empty() || strokes.is_empty() {
        return Vec::new();
    }
    let nearest = sources
        .iter()
        .map(|source| source.distance)
        .fold(f32::INFINITY, f32::min);
    let farthest = sources
        .iter()
        .map(|source| source.distance)
        .fold(0.0, f32::max);
    let last_stroke = strokes.iter().map(|(time, _)| *time).fold(0.0, f32::max);
    let spread = ((farthest - nearest) / SPEED_OF_SOUND + last_stroke)
        .min(MAX_THUNDER_SECONDS - TAIL_SECONDS);
    let rate = THUNDER_SAMPLE_RATE as f32;
    let length = ((spread + TAIL_SECONDS) * rate) as usize;
    let range = (nearest - 250.0).max(0.0) / 1800.0;
    let proximity = if muffled { 0.0 } else { (-range * range).exp() };
    let mut samples = vec![0.0; length];
    add_rolls(&mut samples, seed, proximity, muffled);
    add_claps(&mut samples, &strokes, seed, nearest, proximity);
    boost_thunder_bass(&mut samples, bass_db);

    // Preserve the reference's dynamic range: normalising every roll to a
    // clap's peak would turn quiet distant thunder into an equally loud roar.
    for sample in &mut samples {
        let magnitude = sample.abs();
        if magnitude > 0.9 {
            let compressed = 0.9 + 0.099 * ((magnitude - 0.9) / 0.099).tanh();
            *sample = sample.signum() * compressed;
        }
    }
    samples
}

/// Procedural PCM played through Bevy's custom audio source.
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
        if self.position >= self.samples.len() {
            Some(0)
        } else {
            Some(self.samples.len())
        }
    }

    fn channels(&self) -> ChannelCount {
        NonZero::new(1).unwrap()
    }

    fn sample_rate(&self) -> SampleRate {
        NonZero::new(THUNDER_SAMPLE_RATE).unwrap()
    }

    fn total_duration(&self) -> Option<Duration> {
        Some(Duration::from_secs_f32(
            self.samples.len() as f32 / THUNDER_SAMPLE_RATE as f32,
        ))
    }
}

impl Decodable for ThunderSound {
    type Decoder = ThunderDecoder;

    fn decoder(&self) -> Self::Decoder {
        ThunderDecoder {
            samples: self.samples.clone(),
            position: 0,
        }
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
    settings: Res<AppSettings>,
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
            cloud_sources(
                lightning.position,
                lightning.cloud_base,
                listener,
                lightning.seed,
            ),
            true,
        ),
        BoltKind::Ground | BoltKind::Crawler => (channel_sources(&bolt.segments, listener), false),
    };
    if sources.is_empty() {
        return;
    }
    // Keep the remaining leader time when the first return stroke has not
    // fired yet: sound leaves the channel with the stroke, not the leader.
    let since_flash = lightning.age_seconds - lightning.first_stroke_seconds();
    let start_at =
        time.elapsed_secs_f64() + (arrival_seconds(&sources) - since_flash).max(0.0) as f64;
    let loudness = loudness(&sources);
    let seed = bolt.generation ^ (lightning.seed.to_bits() as u64) << 20;
    let strokes: Vec<_> = lightning.thunder_strokes().collect();
    let bass_db = settings.thunder_bass_db;
    let task = AsyncComputeTaskPool::get()
        .spawn(async move { synthesize_discharge(&sources, &strokes, seed, muffled, bass_db) });
    queue.pending.push(PendingThunder {
        task,
        samples: None,
        start_at,
        loudness,
    });
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
            let sound = sounds.add(ThunderSound {
                samples: samples.into(),
            });
            commands.spawn((
                AudioPlayer::<ThunderSound>(sound),
                PlaybackSettings::DESPAWN.with_volume(Volume::Linear(volume * pending.loudness)),
            ));
        }
        false
    });
}

#[cfg(test)]
#[path = "thunder_tests.rs"]
mod tests;
