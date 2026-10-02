//! Lightning channel geometry. A cloud-to-ground bolt follows its stepped
//! leader: a tortuous random walk from inside the cloud base down to the
//! strike point, kinked again at finer scales on every step, with branches
//! that stall in mid-air. Crawlers spread along the underside of the deck.
//! The lightning pass draws these segments and the thunder is built from
//! their distances to the listener.

use bevy::math::Vec3;
use bevy::prelude::Resource;

/// The lightning pass uploads at most this many segments.
pub const MAX_BOLT_SEGMENTS: usize = 512;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BoltKind {
    /// A discharge inside the cloud: it lights the deck but shows no channel.
    #[default]
    Hidden,
    /// Cloud-to-ground.
    Ground,
    /// Horizontal "spider" lightning crawling along the cloud base.
    Crawler,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BoltSegment {
    pub start: [f32; 3],
    pub end: [f32; 3],
    /// Channel width relative to the main channel.
    pub width: f32,
    /// Relative current: 1 on the main channel, less on each branch.
    pub brightness: f32,
    /// 0 for the main channel, 1 for a branch, 2 for a branch of a branch.
    pub order: u32,
    /// When the leader reaches the segment's start and end, as a fraction of
    /// the time it takes to cross the main channel.
    pub arrival: [f32; 2],
}

/// The channel of the most recent discharge, kept beside the `Copy`
/// [`crate::weather::WeatherState`] so the render world can extract both.
#[derive(Clone, Debug, Default, Resource)]
pub struct ActiveBolt {
    pub kind: BoltKind,
    pub segments: Vec<BoltSegment>,
    /// Bumped for every new discharge so the renderer re-uploads on change.
    pub generation: u64,
}

impl ActiveBolt {
    pub fn replace(&mut self, kind: BoltKind, segments: Vec<BoltSegment>) {
        self.kind = kind;
        self.segments = segments;
        self.generation = self.generation.wrapping_add(1);
    }
}

/// A small deterministic stream, so one seed always grows the same bolt.
struct BoltRandom(u64);

impl BoltRandom {
    fn new(seed: f32) -> Self {
        // SplitMix64 spreads the seed's bits before the xorshift stream.
        let mut state = (seed.to_bits() as u64) ^ 0x9E37_79B9_7F4A_7C15;
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

    fn unit(&mut self) -> Vec3 {
        let y = self.range(-1.0, 1.0);
        let angle = self.range(0.0, std::f32::consts::TAU);
        let radius = (1.0 - y * y).max(0.0).sqrt();
        Vec3::new(radius * angle.cos(), y, radius * angle.sin())
    }

    fn horizontal(&mut self) -> Vec3 {
        let angle = self.range(0.0, std::f32::consts::TAU);
        Vec3::new(angle.cos(), 0.0, angle.sin())
    }
}

/// Turns `direction` by `angle` radians toward a random perpendicular.
fn deviate(random: &mut BoltRandom, direction: Vec3, angle: f32) -> Vec3 {
    let perpendicular = direction.cross(random.unit()).normalize_or(Vec3::X);
    (direction * angle.cos() + perpendicular * angle.sin()).normalize_or(direction)
}

/// Midpoint displacement: each level splits every piece and pushes its middle
/// sideways by a fraction of the piece's length. The fraction shrinks only to
/// 72% per level, which keeps lightning's rough, roughly 1.4-dimensional
/// fractal look at every scale instead of smoothing it out. Returns the
/// points after `start`, ending at `end`.
fn kink(random: &mut BoltRandom, start: Vec3, end: Vec3, levels: u32, roughness: f32) -> Vec<Vec3> {
    let mut points = vec![start, end];
    let mut scale = roughness;
    for _ in 0..levels {
        let mut finer = Vec::with_capacity(points.len() * 2);
        for pair in points.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            let along = b - a;
            let sideways = along.cross(random.unit()).normalize_or(Vec3::X);
            let push = along.length() * scale * random.range(0.4, 1.0);
            finer.push(a);
            finer.push((a + b) * 0.5 + sideways * push);
        }
        finer.push(end);
        points = finer;
        scale *= 0.72;
    }
    points.remove(0);
    points
}

/// Kinks every step of `path`. Returns the fine points, starting with
/// `path[0]`, and the length walked to reach each point of `path`.
fn kinked(random: &mut BoltRandom, path: &[Vec3], levels: u32, roughness: f32) -> (Vec<Vec3>, Vec<f32>) {
    let mut points = vec![path[0]];
    let mut lengths = vec![0.0];
    let mut walked = 0.0;
    for pair in path.windows(2) {
        for point in kink(random, pair[0], pair[1], levels, roughness) {
            walked += points.last().unwrap().distance(point);
            points.push(point);
        }
        lengths.push(walked);
    }
    (points, lengths)
}

/// Appends a polyline as segments. `start_length` is the leader path walked
/// before its first point; `arrival_scale` turns path length into arrival.
fn push_points(
    segments: &mut Vec<BoltSegment>,
    points: &[Vec3],
    style: (u32, f32, f32),
    start_length: f32,
    arrival_scale: f32,
) {
    let (order, width, brightness) = style;
    let mut walked = start_length;
    for pair in points.windows(2) {
        if segments.len() >= MAX_BOLT_SEGMENTS {
            return;
        }
        let length = pair[0].distance(pair[1]);
        segments.push(BoltSegment {
            start: pair[0].to_array(),
            end: pair[1].to_array(),
            width,
            brightness,
            order,
            arrival: [
                (walked * arrival_scale).min(1.0),
                ((walked + length) * arrival_scale).min(1.0),
            ],
        });
        walked += length;
    }
}

/// A cloud-to-ground bolt from `top`, inside the cloud, to the `strike`
/// contact point. The main channel always ends exactly on `strike`.
pub fn ground_bolt(seed: f32, strike: [f32; 3], top: [f32; 3]) -> Vec<BoltSegment> {
    let mut random = BoltRandom::new(seed);
    let strike = Vec3::from(strike);
    let top = Vec3::from(top);
    let height = (top.y - strike.y).max(60.0);
    // About two dozen leader steps between the cloud and the ground.
    let step = (height / 22.0).clamp(15.0, 60.0);
    let mut leader = vec![top];
    let mut position = top;
    for _ in 0..96 {
        let gap = strike - position;
        if gap.length() <= step * 1.6 || gap.y > -step * 0.5 {
            break;
        }
        let progress = (1.0 - -gap.y / height).clamp(0.0, 1.0);
        // The leader wanders while it is high and is drawn ever harder
        // toward the point it will strike as the gap to the ground closes,
        // and at once if the height left can no longer cover its drift.
        let drift = Vec3::new(gap.x, 0.0, gap.z).length() / -gap.y;
        let attraction = (0.3 + 0.65 * progress * progress).max((drift - 0.6).clamp(0.0, 0.95));
        let target = gap.normalize_or(Vec3::NEG_Y);
        let base = (Vec3::NEG_Y * (1.0 - attraction) + target * attraction).normalize_or(Vec3::NEG_Y);
        let angle = random.range(0.45, 0.8);
        let mut direction = deviate(&mut random, base, angle);
        direction.y = direction.y.min(-0.25);
        position += direction.normalize() * step * random.range(0.6, 1.4);
        leader.push(position);
    }
    leader.push(strike);

    // Three levels of kinks break the leader steps down to a few metres, so
    // even a close strike stays jagged all the way down.
    let (points, lengths) = kinked(&mut random, &leader, 3, 0.3);
    let main_length = lengths.last().copied().unwrap_or(1.0).max(1.0);
    let arrival_scale = 1.0 / main_length;
    let mut segments = Vec::with_capacity(MAX_BOLT_SEGMENTS);
    push_points(&mut segments, &points, (0, 1.0, 1.0), 0.0, arrival_scale);
    if let Some(last) = segments.last_mut() {
        // Rounding must not leave the contact a hair short of arrival 1.
        last.arrival[1] = 1.0;
    }

    // Branches fork from leader steps above the lowest sixth of the channel
    // and are likelier high up, where the leader was still searching.
    for index in 1..leader.len().saturating_sub(2) {
        let fork = leader[index];
        let progress = 1.0 - (fork.y - strike.y) / height;
        if progress > 0.84 || random.next() > 0.3 - 0.2 * progress {
            continue;
        }
        let parent = (leader[index + 1] - fork).normalize_or(Vec3::NEG_Y);
        let branch = Branch {
            fork,
            parent,
            reach: (fork.y - strike.y) * random.range(0.25, 0.6),
            step: step * 0.75,
            floor: strike.y + 15.0,
            order: 1,
            brightness: random.range(0.3, 0.6),
            start_length: lengths[index],
        };
        grow_branch(&mut segments, &mut random, branch, arrival_scale);
    }
    segments
}

/// A failed leader: a shorter walk away from its parent that ends in mid-air.
#[derive(Clone, Copy)]
struct Branch {
    fork: Vec3,
    parent: Vec3,
    reach: f32,
    step: f32,
    /// Branches stall before they come this low.
    floor: f32,
    order: u32,
    brightness: f32,
    start_length: f32,
}

fn grow_branch(segments: &mut Vec<BoltSegment>, random: &mut BoltRandom, branch: Branch, arrival_scale: f32) {
    if segments.len() >= MAX_BOLT_SEGMENTS || branch.reach < branch.step {
        return;
    }
    let outward = random.horizontal();
    let mut direction = (branch.parent * 0.45 + outward * 0.8 + Vec3::NEG_Y * 0.35)
        .normalize_or(Vec3::NEG_Y);
    let mut path = vec![branch.fork];
    let mut position = branch.fork;
    let mut length = 0.0;
    let mut forks = Vec::new();
    while length < branch.reach && path.len() < 16 {
        let angle = random.range(0.35, 0.75);
        direction = deviate(random, direction, angle);
        direction.y = direction.y.min(-0.15);
        direction = (direction + Vec3::NEG_Y * 0.15).normalize();
        let distance = branch.step * random.range(0.6, 1.3);
        let next = position + direction * distance;
        if next.y < branch.floor {
            break;
        }
        position = next;
        length += distance;
        path.push(position);
        if branch.order < 2 && random.next() < 0.16 {
            forks.push((path.len() - 1, direction));
        }
    }
    if path.len() < 2 {
        return;
    }
    let (points, lengths) = kinked(random, &path, 2, 0.2);
    let width = 0.55_f32.powi(branch.order as i32);
    push_points(
        segments,
        &points,
        (branch.order, width, branch.brightness),
        branch.start_length,
        arrival_scale,
    );
    for (index, direction) in forks {
        let reach = (branch.reach - lengths[index]) * random.range(0.3, 0.6);
        let brightness = branch.brightness * random.range(0.45, 0.7);
        let child = Branch {
            fork: path[index],
            parent: direction,
            reach,
            step: branch.step * 0.8,
            floor: branch.floor,
            order: branch.order + 1,
            brightness,
            start_length: branch.start_length + lengths[index],
        };
        grow_branch(segments, random, child, arrival_scale);
    }
}

fn turn(direction: Vec3, angle: f32) -> Vec3 {
    let (sin, cos) = angle.sin_cos();
    Vec3::new(direction.x * cos - direction.z * sin, 0.0, direction.x * sin + direction.z * cos)
}

/// One crawler arm: a mostly level walk that hugs the irregular cloud base,
/// dipping below it and rising back in.
fn crawl(random: &mut BoltRandom, start: Vec3, heading: Vec3, reach: f32, steps: (f32, f32), cloud_base: f32) -> Vec<Vec3> {
    let mut direction = heading;
    let mut position = start;
    let mut path = vec![position];
    let mut length = 0.0;
    while length < reach && path.len() < 32 {
        let angle = random.range(-0.4, 0.4);
        direction = turn(direction, angle);
        let distance = random.range(steps.0, steps.1);
        position += direction * distance;
        position.y = (position.y + random.range(-18.0, 18.0)).clamp(cloud_base - 190.0, cloud_base - 15.0);
        length += distance;
        path.push(position);
    }
    path
}

/// Keeps kinked crawler points out of the cloud they crawl beneath.
fn under_base(points: Vec<Vec3>, cloud_base: f32) -> Vec<Vec3> {
    points
        .into_iter()
        .map(|point| Vec3::new(point.x, point.y.clamp(cloud_base - 200.0, cloud_base - 8.0), point.z))
        .collect()
}

/// Spider lightning: several arms crawling outward just beneath the cloud
/// base from `centre`, forking as they go.
pub fn crawler_bolt(seed: f32, centre: [f32; 3], cloud_base: f32) -> Vec<BoltSegment> {
    let mut random = BoltRandom::new(seed);
    let centre = Vec3::new(centre[0], cloud_base - random.range(40.0, 120.0), centre[2]);
    let arms = 2 + (random.next() * 2.99) as usize;
    let plans: Vec<(Vec3, f32)> = (0..arms)
        .map(|_| (random.horizontal(), random.range(600.0, 2000.0)))
        .collect();
    let longest = plans.iter().map(|(_, length)| *length).fold(1.0, f32::max);
    let arrival_scale = 1.0 / longest;
    let mut segments = Vec::with_capacity(MAX_BOLT_SEGMENTS);
    let mut forks = Vec::new();
    for (heading, reach) in plans {
        let arm = crawl(&mut random, centre, heading, reach, (50.0, 95.0), cloud_base);
        let (points, lengths) = kinked(&mut random, &arm, 2, 0.18);
        let points = under_base(points, cloud_base);
        let brightness = random.range(0.75, 1.0);
        push_points(&mut segments, &points, (0, 1.0, brightness), 0.0, arrival_scale);
        for index in 1..arm.len() - 1 {
            if random.next() < 0.22 {
                let heading = (arm[index + 1] - arm[index]).normalize_or(heading);
                forks.push((arm[index], heading, lengths[index], reach - lengths[index]));
            }
        }
    }
    for (fork, heading, length, remaining) in forks {
        let side = if random.next() < 0.5 { -1.0 } else { 1.0 };
        let heading = turn(Vec3::new(heading.x, 0.0, heading.z).normalize_or(Vec3::X), side * random.range(0.5, 1.1));
        let reach = remaining.min(random.range(200.0, 600.0));
        let arm = crawl(&mut random, fork, heading, reach, (40.0, 80.0), cloud_base);
        if arm.len() >= 2 {
            let (points, _) = kinked(&mut random, &arm, 1, 0.18);
            let points = under_base(points, cloud_base);
            let brightness = random.range(0.35, 0.6);
            push_points(&mut segments, &points, (1, 0.55, brightness), length, arrival_scale);
        }
    }
    segments
}

#[cfg(test)]
mod tests {
    use super::*;

    const STRIKE: [f32; 3] = [1200.0, 35.0, -800.0];
    const TOP: [f32; 3] = [1400.0, 1350.0, -650.0];

    fn main_channel(segments: &[BoltSegment]) -> Vec<BoltSegment> {
        segments.iter().copied().filter(|segment| segment.order == 0).collect()
    }

    #[test]
    fn ground_channel_runs_from_the_cloud_to_exactly_the_strike_point() {
        for seed in 1..60 {
            let seed = seed as f32 / 61.0;
            let segments = ground_bolt(seed, STRIKE, TOP);
            assert!(segments.len() <= MAX_BOLT_SEGMENTS);
            let main = main_channel(&segments);
            assert!(main.len() > 20, "seed {seed}: {} main segments", main.len());
            assert_eq!(main[0].start, TOP);
            assert_eq!(main.last().unwrap().end, STRIKE);
            for pair in main.windows(2) {
                assert_eq!(pair[0].end, pair[1].start, "the channel must be continuous");
                assert!(pair[0].arrival[1] <= pair[1].arrival[1] + 1e-6);
            }
            assert!((main.last().unwrap().arrival[1] - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn bolts_are_deterministic_per_seed_and_differ_between_seeds() {
        assert_eq!(ground_bolt(0.37, STRIKE, TOP), ground_bolt(0.37, STRIKE, TOP));
        assert_ne!(ground_bolt(0.37, STRIKE, TOP), ground_bolt(0.38, STRIKE, TOP));
        assert_eq!(crawler_bolt(0.37, STRIKE, 1000.0), crawler_bolt(0.37, STRIKE, 1000.0));
    }

    #[test]
    fn ground_channel_is_tortuous_but_stays_near_its_column() {
        let height = TOP[1] - STRIKE[1];
        for seed in 1..60 {
            let main = main_channel(&ground_bolt(seed as f32 / 61.0, STRIKE, TOP));
            let path: f32 = main
                .iter()
                .map(|segment| Vec3::from(segment.start).distance(Vec3::from(segment.end)))
                .sum();
            let straight = Vec3::from(TOP).distance(Vec3::from(STRIKE));
            let tortuosity = path / straight;
            assert!((1.1..2.6).contains(&tortuosity), "tortuosity {tortuosity}");
            for segment in &main {
                let drift = Vec3::new(segment.end[0] - STRIKE[0], 0.0, segment.end[2] - STRIKE[2]).length();
                assert!(drift < 0.9 * height, "drift {drift}");
            }
        }
    }

    #[test]
    fn ground_bolts_branch_and_branches_stall_above_the_ground() {
        let mut branched = 0;
        for seed in 1..60 {
            let segments = ground_bolt(seed as f32 / 61.0, STRIKE, TOP);
            let branches: Vec<_> = segments.iter().filter(|segment| segment.order > 0).collect();
            if !branches.is_empty() {
                branched += 1;
            }
            for branch in branches {
                assert!(branch.end[1] >= STRIKE[1] + 4.0, "branch reaches the ground");
                assert!(branch.brightness < 1.0 && branch.width < 1.0);
                assert!(branch.arrival[0] > 0.0);
            }
        }
        assert!(branched > 45, "most bolts should fork: {branched}");
    }

    #[test]
    fn crawlers_stay_under_the_cloud_base() {
        for seed in 1..40 {
            let segments = crawler_bolt(seed as f32 / 41.0, STRIKE, 1000.0);
            assert!(segments.len() > 10 && segments.len() <= MAX_BOLT_SEGMENTS);
            for segment in segments {
                for point in [segment.start, segment.end] {
                    assert!(point[1] <= 1000.0 && point[1] >= 760.0, "crawler at {}", point[1]);
                }
            }
        }
    }
}
