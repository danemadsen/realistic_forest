//! How a river shapes the ground, and what the water is doing at a point.
//!
//! A river is drawn as a chain of short carve segments. Each segment bounds
//! the terrain from above and below near its centreline:
//!
//! - **Upper envelope.** Inside the wetted width the bed follows a skewed
//!   parabola below the water surface, deepest toward the outer bank of a
//!   bend. Beyond the waterline a bank cone rises at the bank slope, then
//!   ever more steeply, so ground that stands above it (a spur a meander
//!   swings into, the saddle a channel breaches) is cut back into a bank.
//! - **Lower envelope.** Just outside the waterline the ground is held a
//!   little above the water, so the channel always contains its river even
//!   where the floodplain dips; a short outer slope returns to the terrain.
//!
//! The terrain height is `min(max(h, lower), upper)`, with the upper bound
//! the minimum and the lower bound the maximum over every nearby segment, so
//! confluences open into each other and a channel always wins over a
//! neighbour's levee.
//!
//! Segments have a flat start and a round end: a point behind a segment's
//! start belongs to the segment before it. That keeps a waterfall's lip a
//! clean vertical face, since the plunge pool's segments never reach back
//! over it, while bends stay covered by the previous segment's round end.
//!
//! The same arithmetic runs in `river-functions.wgslinc` on the GPU, from the
//! same uploaded segments, so the drawn ground, the player's footing, the
//! seated plants and the erosion simulation all see one channel.

/// Beyond this many metres past the waterline a segment no longer shapes the
/// ground. The bank cone has risen some 30 m above the water by then.
pub const BANK_REACH: f32 = 12.0;
/// Curvature of the bank cone: its rise grows by this times the square of
/// the distance past the waterline.
pub const BANK_CURVE: f32 = 0.16;
/// Segments a lookup reads from one grid cell at most, here and on the GPU.
pub const MAX_CANDIDATES: usize = 64;
/// Slope of a levee's outer face back down to the terrain: a natural levee
/// is a broad, low ridge, not a dike.
pub const LEVEE_OUTER_SLOPE: f32 = 0.15;
/// Side of a cell of the segment lookup grid, metres.
pub const GRID_CELL: f32 = 32.0;

/// One carve segment, as the GPU reads it: 64 bytes, mirrored by
/// `RiverSegment` in river-functions.wgslinc.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RiverSegment {
    /// Centreline start and end, world XZ.
    pub a: [f32; 2],
    pub b: [f32; 2],
    /// Water surface height at the start and end.
    pub water: [f32; 2],
    /// Wetted half width at the start and end.
    pub half_width: [f32; 2],
    /// Depth of the thalweg below the water at the start and end.
    pub depth: [f32; 2],
    /// Mean flow speed at the start and end, m/s.
    pub speed: [f32; 2],
    /// Bank slope (rise over run) just past the waterline.
    pub bank: f32,
    /// Thalweg skew toward the left (+) or right (-) bank, -1..1: the outer
    /// bank of a bend, where the pool is deep and the bank is cut steep.
    pub skew: f32,
    /// Whitewater, 0 calm to 1 a cascade or waterfall: rough beds, bare
    /// rock and foam.
    pub turbulence: f32,
    /// How firmly the ground beside the channel is held above the water,
    /// 0..1: none where the river meets the sea, whose bed must not rise.
    pub levee: f32,
}

const _: () = assert!(std::mem::size_of::<RiverSegment>() == 64);

impl RiverSegment {
    /// Largest distance from the centreline at which the segment shapes the
    /// ground.
    pub fn reach(&self) -> f32 {
        self.half_width[0].max(self.half_width[1]) + BANK_REACH
    }
}

/// What one segment, or the whole river network, says about a point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Envelope {
    /// The ground may not stand above this.
    pub upper: f32,
    /// The ground may not lie below this.
    pub lower: f32,
    /// Distance past the nearest waterline, metres: negative inside a
    /// channel.
    pub bank_distance: f32,
    /// The water surface of the channel nearest its waterline here.
    pub water: f32,
    /// Flow velocity at the point, world XZ, m/s.
    pub velocity: [f32; 2],
    /// Wetted half width of that channel.
    pub half_width: f32,
    /// Its whitewater, 0..1.
    pub turbulence: f32,
    /// Which side of a bend the point lies on: toward +0.6 on the outside
    /// (a cut bank), toward -0.6 on the inside (a point bar).
    pub bend: f32,
    /// The level of a lake whose basin reaches here, or -infinity: ground
    /// below it lies under the lake.
    pub lake: f32,
}

/// Metres from a lake's shore per metre the ground stands above or below
/// its water, for a shore's typical slope; how near the shore a point is,
/// in the same terms as a river's bank distance.
pub const LAKE_SHORE_RUN: f32 = 6.0;

impl Envelope {
    pub const NONE: Envelope = Envelope {
        upper: f32::INFINITY,
        lower: f32::NEG_INFINITY,
        bank_distance: f32::INFINITY,
        water: f32::NEG_INFINITY,
        velocity: [0.0, 0.0],
        half_width: 0.0,
        turbulence: 0.0,
        bend: 0.0,
        lake: f32::NEG_INFINITY,
    };

    /// Distance past the nearest waterline, river or lake, for ground at
    /// `height`: negative under water.
    pub fn bank_at(&self, height: f32) -> f32 {
        self.bank_distance.min((height - self.lake) * LAKE_SHORE_RUN)
    }

    /// The water standing over ground at `height` here, if any: a river's
    /// surface in its channel, a lake's over its bed.
    pub fn water_over(&self, height: f32) -> Option<f32> {
        let river = (self.bank_distance < 0.0).then_some(self.water);
        let lake = (height < self.lake).then_some(self.lake);
        match (river, lake) {
            (Some(r), Some(l)) => Some(r.max(l)),
            (r, l) => r.or(l),
        }
    }

    /// Apply the envelope to a terrain height.
    pub fn clamp(&self, height: f32) -> f32 {
        height.max(self.lower).min(self.upper)
    }

    pub fn is_none(&self) -> bool {
        self.bank_distance == f32::INFINITY
    }
}

fn mix(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn smoothstep(edge0: f32, edge1: f32, value: f32) -> f32 {
    let t = ((value - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// One segment's envelope at `p`. Mirrored line for line by
/// `riverSegmentEnvelope` in river-functions.wgslinc.
pub fn segment_envelope(segment: &RiverSegment, p: [f32; 2]) -> Envelope {
    let ab = [segment.b[0] - segment.a[0], segment.b[1] - segment.a[1]];
    let ap = [p[0] - segment.a[0], p[1] - segment.a[1]];
    let length_squared = (ab[0] * ab[0] + ab[1] * ab[1]).max(1e-6);
    let along = (ap[0] * ab[0] + ap[1] * ab[1]) / length_squared;
    // Flat start: what lies behind the start belongs to the segment before.
    if along < 0.0 {
        return Envelope::NONE;
    }
    let t = along.min(1.0);
    let offset = [ap[0] - ab[0] * t, ap[1] - ab[1] * t];
    let distance = (offset[0] * offset[0] + offset[1] * offset[1]).sqrt();
    let half_width = mix(segment.half_width[0], segment.half_width[1], t).max(0.05);
    let past_bank = distance - half_width;
    if past_bank > BANK_REACH {
        return Envelope::NONE;
    }
    let water = mix(segment.water[0], segment.water[1], t);
    let depth = mix(segment.depth[0], segment.depth[1], t);
    let length = length_squared.sqrt();
    // Signed distance to the centreline, + on the left of the flow. The skew
    // fades out over the round end cap, where "left" stops meaning anything.
    let cross = ab[0] * ap[1] - ab[1] * ap[0];
    let side = if cross >= 0.0 { 1.0 } else { -1.0 };
    let beyond = (along - 1.0).max(0.0) * length;
    let skew = segment.skew * (1.0 - smoothstep(0.0, half_width, beyond));
    let speed = mix(segment.speed[0], segment.speed[1], t);
    let direction = [ab[0] / length, ab[1] / length];
    let mut envelope = Envelope {
        upper: f32::INFINITY,
        lower: f32::NEG_INFINITY,
        bank_distance: past_bank,
        water,
        velocity: [direction[0] * speed, direction[1] * speed],
        half_width,
        turbulence: segment.turbulence,
        bend: skew * side,
        lake: f32::NEG_INFINITY,
    };
    if past_bank < 0.0 {
        // Skewed parabola: (1 - u^2)(1 + skew u) is zero at both banks and
        // deepest toward the outer one.
        let u = side * distance / half_width;
        let profile = (1.0 - u * u) * (1.0 + skew * u);
        envelope.upper = water - depth * profile;
        // The current runs fastest over the thalweg and stalls at the banks.
        let lateral = (1.0 - u * u).max(0.0).powf(0.35);
        envelope.velocity = [envelope.velocity[0] * lateral * 1.25, envelope.velocity[1] * lateral * 1.25];
    } else {
        // Cut banks stand steep on the outside of a bend; point bars slope
        // gently into the water on the inside.
        let bank = segment.bank * (1.0 + 0.75 * skew * side).max(0.3);
        envelope.upper = water + bank * past_bank + BANK_CURVE * past_bank * past_bank;
        let freeboard = 0.1 + 0.25 * depth;
        let levee_width = 0.8 + 0.3 * half_width;
        envelope.lower = water + (bank * past_bank).min(freeboard)
            - (past_bank - levee_width).max(0.0) * LEVEE_OUTER_SLOPE
            - (1.0 - segment.levee) * 1.0e4;
        envelope.velocity = [0.0, 0.0];
    }
    envelope
}

/// Combine one segment's envelope into the running result. Mirrored by
/// `riverCombine` in river-functions.wgslinc.
pub fn combine(total: &mut Envelope, next: &Envelope) {
    if next.is_none() {
        return;
    }
    total.upper = total.upper.min(next.upper);
    total.lower = total.lower.max(next.lower);
    // The water, its motion and the bank distance come from the channel
    // whose waterline is nearest, measured in its own widths so a creek
    // meeting a river hands over to it inside the larger channel.
    let score = |e: &Envelope| e.bank_distance / e.half_width.clamp(0.5, 4.0);
    if total.is_none() || score(next) < score(total) {
        total.bank_distance = next.bank_distance;
        total.water = next.water;
        total.velocity = next.velocity;
        total.half_width = next.half_width;
        total.turbulence = next.turbulence;
        total.bend = next.bend;
    }
}

/// A uniform grid of segment lists over the network's whole domain, with
/// the lakes reaching into each cell. Uploaded to the GPU as one `u32` array:
/// an eight-word header, then `(offset, segment count, lake)` per cell, then
/// the index lists the offsets point into, then the lake records. A cell's
/// lake word is `NO_LAKE_RECORD` or the offset of its record: the lake's
/// level (f32 bits) and a 64-bit mask of which of the cell's 8 x 8 lake
/// cells (`GRID_CELL / 8`) lie under the lake or its shore.
#[derive(Clone, Debug, Default)]
pub struct SegmentGrid {
    pub origin: [f32; 2],
    pub resolution: usize,
    /// Per cell: offset into `indices` and segment count.
    pub cells: Vec<[u32; 2]>,
    pub indices: Vec<u32>,
    /// The cells lakes reach into: (level, mask of lake cells), by cell.
    pub lakes: std::collections::BTreeMap<u32, (f32, u64)>,
}

/// The "no lake" level on the GPU, where infinities are best avoided.
pub const NO_LAKE: f32 = -1.0e30;
/// A cell's lake word when no lake reaches it.
pub const NO_LAKE_RECORD: u32 = u32::MAX;
/// Lake cells across a grid cell.
pub const LAKE_CELLS_ACROSS: usize = 8;
/// Words per cell in the GPU grid.
pub const GRID_CELL_WORDS: usize = 3;

/// Words in the GPU grid header.
pub const GRID_HEADER_WORDS: usize = 8;

impl SegmentGrid {
    pub fn build(origin: [f32; 2], resolution: usize, segments: &[RiverSegment]) -> Self {
        let cell_range = |minimum: [f32; 2], maximum: [f32; 2]| {
            let cell = |value: f32, axis: usize| ((value - origin[axis]) / GRID_CELL).floor() as i64;
            (
                cell(minimum[0], 0).max(0),
                cell(maximum[0], 0).min(resolution as i64 - 1),
                cell(minimum[1], 1).max(0),
                cell(maximum[1], 1).min(resolution as i64 - 1),
            )
        };
        let mut segment_buckets: Vec<Vec<u32>> = vec![Vec::new(); resolution * resolution];
        for (index, segment) in segments.iter().enumerate() {
            let reach = segment.reach();
            let (x0, x1, z0, z1) = cell_range(
                [segment.a[0].min(segment.b[0]) - reach, segment.a[1].min(segment.b[1]) - reach],
                [segment.a[0].max(segment.b[0]) + reach, segment.a[1].max(segment.b[1]) + reach],
            );
            for z in z0..=z1 {
                for x in x0..=x1 {
                    segment_buckets[z as usize * resolution + x as usize].push(index as u32);
                }
            }
        }
        let mut cells = Vec::with_capacity(segment_buckets.len());
        let mut indices = Vec::new();
        for bucket in &segment_buckets {
            cells.push([indices.len() as u32, bucket.len() as u32]);
            indices.extend_from_slice(bucket);
        }
        Self {
            origin,
            resolution,
            cells,
            indices,
            lakes: Default::default(),
        }
    }

    fn cell_index(&self, p: [f32; 2]) -> Option<usize> {
        let x = ((p[0] - self.origin[0]) / GRID_CELL).floor();
        let z = ((p[1] - self.origin[1]) / GRID_CELL).floor();
        if x < 0.0 || z < 0.0 || x >= self.resolution as f32 || z >= self.resolution as f32 {
            return None;
        }
        Some(z as usize * self.resolution + x as usize)
    }

    fn cell(&self, p: [f32; 2]) -> Option<[u32; 2]> {
        self.cell_index(p).map(|index| self.cells[index])
    }

    /// Mark a lake's cells (`cell_size` = GRID_CELL / LAKE_CELLS_ACROSS
    /// metres, aligned with this grid) in the cells that hold them.
    pub fn add_lake(&mut self, level: f32, cells: &[[i32; 2]], cell_size: f32) {
        for cell in cells {
            let centre = [(cell[0] as f32 + 0.5) * cell_size, (cell[1] as f32 + 0.5) * cell_size];
            let Some(index) = self.cell_index(centre) else {
                continue;
            };
            let bit = self.lake_bit(centre);
            let entry = self.lakes.entry(index as u32).or_insert((level, 0));
            entry.0 = entry.0.max(level);
            entry.1 |= 1u64 << bit;
        }
    }

    /// Which of its cell's lake cells `p` lies in, 0..64.
    fn lake_bit(&self, p: [f32; 2]) -> u32 {
        let fine = GRID_CELL / LAKE_CELLS_ACROSS as f32;
        let x = ((p[0] - self.origin[0]) / fine).floor() as i64;
        let z = ((p[1] - self.origin[1]) / fine).floor() as i64;
        let across = LAKE_CELLS_ACROSS as i64;
        (z.rem_euclid(across) * across + x.rem_euclid(across)) as u32
    }

    /// The level of the lake `p` lies in or on the shore of, or `NO_LAKE`.
    pub fn lake(&self, p: [f32; 2]) -> f32 {
        let Some(index) = self.cell_index(p) else {
            return NO_LAKE;
        };
        match self.lakes.get(&(index as u32)) {
            Some(&(level, mask)) if (mask >> self.lake_bit(p)) & 1 == 1 => level,
            _ => NO_LAKE,
        }
    }

    /// The segment indices whose reach may cover `p`.
    pub fn candidates(&self, p: [f32; 2]) -> &[u32] {
        match self.cell(p) {
            Some([offset, count]) => &self.indices[offset as usize..(offset + count) as usize],
            None => &[],
        }
    }

    /// The GPU layout: header, cell table, index lists, lake records.
    pub fn gpu_words(&self, segment_count: usize) -> Vec<u32> {
        let mut words = Vec::with_capacity(GRID_HEADER_WORDS + self.cells.len() * GRID_CELL_WORDS + self.indices.len());
        words.push(self.origin[0].to_bits());
        words.push(self.origin[1].to_bits());
        words.push(GRID_CELL.to_bits());
        words.push(self.resolution as u32);
        words.push(segment_count as u32);
        words.extend_from_slice(&[0, 0, 0]);
        let base = (GRID_HEADER_WORDS + self.cells.len() * GRID_CELL_WORDS) as u32;
        let records = base + self.indices.len() as u32;
        let mut record = 0;
        for (index, cell) in self.cells.iter().enumerate() {
            // Offsets are into the whole word array.
            words.push(cell[0] + base);
            words.push(cell[1]);
            if self.lakes.contains_key(&(index as u32)) {
                words.push(records + 3 * record);
                record += 1;
            } else {
                words.push(NO_LAKE_RECORD);
            }
        }
        words.extend_from_slice(&self.indices);
        // In cell order, as the offsets above were handed out.
        for &(level, mask) in self.lakes.values() {
            words.push(level.to_bits());
            words.push(mask as u32);
            words.push((mask >> 32) as u32);
        }
        words
    }
}

/// The envelope of every segment near `p`.
pub fn envelope_at(segments: &[RiverSegment], grid: &SegmentGrid, p: [f32; 2]) -> Envelope {
    let mut total = Envelope::NONE;
    for &index in grid.candidates(p).iter().take(MAX_CANDIDATES) {
        let envelope = segment_envelope(&segments[index as usize], p);
        combine(&mut total, &envelope);
    }
    let lake = grid.lake(p);
    if lake > NO_LAKE {
        total.lake = lake;
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    fn straight(water: [f32; 2]) -> RiverSegment {
        RiverSegment {
            a: [0.0, 0.0],
            b: [10.0, 0.0],
            water,
            half_width: [2.0, 2.0],
            depth: [0.5, 0.5],
            speed: [1.0, 1.0],
            bank: 0.8,
            skew: 0.0,
            turbulence: 0.0,
            levee: 1.0,
        }
    }

    #[test]
    fn channel_bed_lies_below_the_water_and_meets_it_at_the_banks() {
        let segment = straight([10.0, 10.0]);
        let centre = segment_envelope(&segment, [5.0, 0.0]);
        assert!((centre.upper - 9.5).abs() < 1e-4, "{centre:?}");
        let bank = segment_envelope(&segment, [5.0, 2.0]);
        assert!((bank.upper - 10.0).abs() < 1e-4, "{bank:?}");
        assert!(bank.bank_distance.abs() < 1e-4);
        // Just past the waterline the ground is held above the water and
        // below the bank cone; further out the levee falls back away.
        let outside = segment_envelope(&segment, [5.0, 3.0]);
        assert!(outside.lower > 10.0 && outside.lower <= outside.upper, "{outside:?}");
        assert!((outside.upper - (10.0 + 0.8 + BANK_CURVE)).abs() < 1e-4);
        let beyond = segment_envelope(&segment, [5.0, 6.0]);
        assert!(beyond.lower < outside.lower);
        assert!(segment_envelope(&segment, [5.0, 2.0 + BANK_REACH + 0.5]).is_none());
        // Far away the segment says nothing.
        assert!(segment_envelope(&segment, [5.0, 2.0 + BANK_REACH + 0.5]).is_none());
    }

    #[test]
    fn a_segment_never_reaches_back_past_its_start() {
        let segment = straight([10.0, 9.0]);
        assert!(segment_envelope(&segment, [-0.5, 0.0]).is_none());
        // ...but its round end covers the outside of the next bend.
        let ahead = segment_envelope(&segment, [11.0, 0.5]);
        assert!(!ahead.is_none() && ahead.upper < 9.0);
    }

    #[test]
    fn the_outer_bank_of_a_bend_is_deep_and_steep() {
        let mut segment = straight([10.0, 10.0]);
        segment.skew = 0.6;
        let left = segment_envelope(&segment, [5.0, 1.0]);
        let right = segment_envelope(&segment, [5.0, -1.0]);
        assert!(left.upper < right.upper, "{left:?} {right:?}");
        let left_bank = segment_envelope(&segment, [5.0, 4.0]);
        let right_bank = segment_envelope(&segment, [5.0, -4.0]);
        assert!(left_bank.upper > right_bank.upper);
    }

    #[test]
    fn grid_finds_every_segment_that_reaches_a_point() {
        let segments: Vec<RiverSegment> = (0..20)
            .map(|i| {
                let mut s = straight([10.0, 10.0]);
                s.a = [i as f32 * 10.0, 3.0 * i as f32];
                s.b = [i as f32 * 10.0 + 10.0, 3.0 * i as f32 + 3.0];
                s
            })
            .collect();
        let grid = SegmentGrid::build([-100.0, -100.0], 16, &segments);
        for z in -60..120 {
            for x in -60..260 {
                let p = [x as f32 * 1.3, z as f32 * 1.1];
                let mut brute = Envelope::NONE;
                for segment in &segments {
                    combine(&mut brute, &segment_envelope(segment, p));
                }
                let fast = envelope_at(&segments, &grid, p);
                assert_eq!(brute.upper, fast.upper);
                assert_eq!(brute.lower, fast.lower);
            }
        }
        // The GPU words point at the same lists.
        let words = grid.gpu_words(segments.len());
        let cell = 5 * 16 + 7;
        let offset = words[GRID_HEADER_WORDS + cell * GRID_CELL_WORDS] as usize;
        let count = words[GRID_HEADER_WORDS + cell * GRID_CELL_WORDS + 1] as usize;
        assert_eq!(&words[offset..offset + count], &grid.indices[grid.cells[cell][0] as usize..][..count]);
    }
}

#[cfg(test)]
mod gpu_tests {
    use super::NO_LAKE;
    use bevy::tasks::block_on;
    use wgpu::util::DeviceExt;

    /// The WGSL river block must carve exactly what the CPU carves: the
    /// player walks on the CPU's ground and sees the GPU's.
    /// Run on a GPU (or lavapipe): cargo test river_block_matches_cpu -- --ignored
    #[test]
    #[ignore = "requires a GPU adapter"]
    fn river_block_matches_cpu() {
        let noise = crate::noise::NoiseField::new();
        let network = crate::rivers::network::generate(&noise, [0, 0]);
        let payload = network.gpu_payload();
        let instance = wgpu::Instance::default();
        let adapter = block_on(instance.request_adapter(&Default::default())).expect("adapter");
        let (device, queue) = block_on(adapter.request_device(&Default::default())).expect("device");
        let block = include_str!("../../assets/shaders/river-functions.wgslinc");
        let source = format!(
            "@group(0) @binding(0) var<storage, read> river_grid: array<u32>;
             @group(0) @binding(1) var<storage, read> river_segments: array<RiverSegment>;
             @group(0) @binding(2) var<storage, read_write> samples: array<vec4<f32>>;
             {block}
             @compute @workgroup_size(64)
             fn evaluate(@builtin(global_invocation_id) id: vec3<u32>) {{
                 if (id.x >= arrayLength(&samples)) {{ return; }}
                 let p = samples[id.x].xy;
                 let e = riverEnvelope(p);
                 samples[id.x] = vec4<f32>(e.upper, e.lower, e.bank_distance, e.lake);
             }}"
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("river block"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader,
            entry_point: Some("evaluate"),
            compilation_options: Default::default(),
            cache: None,
        });
        // Points along and across the channels near spawn.
        let mut points = Vec::new();
        for node in network.rivers.iter().flat_map(|r| &r.nodes).step_by(7).take(4000) {
            for offset in [-6.0f32, -2.0, -0.7, 0.0, 0.4, 1.3, 3.0, 9.0] {
                points.push([node.position[0] + offset, node.position[1] - offset * 0.5]);
            }
        }
        // ...and over the lakes and their shores.
        let cell = crate::rivers::network::FLOW_CELL as f32;
        for lake in &network.lakes {
            for c in lake.cells.iter().step_by(37).take(20) {
                points.push([(c[0] as f32 + 0.5) * cell, (c[1] as f32 + 0.5) * cell]);
            }
        }
        let samples: Vec<[f32; 4]> = points.iter().map(|p| [p[0], p[1], 0.0, 0.0]).collect();
        let grid = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&payload.grid_words),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let segments = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&payload.segments),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let output = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&samples),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: output.size(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: grid.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: segments.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: output.as_entire_binding() },
            ],
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups((samples.len() as u32).div_ceil(64), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, output.size());
        queue.submit([encoder.finish()]);
        let (sender, receiver) = std::sync::mpsc::channel();
        readback.map_async(wgpu::MapMode::Read, .., move |result| sender.send(result).unwrap());
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        receiver.recv().unwrap().unwrap();
        let mapped = readback.get_mapped_range(..);
        let results: &[[f32; 4]] = bytemuck::cast_slice(&mapped);
        let mut inside = 0;
        let mut lakes = 0;
        for (p, gpu) in points.iter().zip(results) {
            let cpu = network.envelope(p[0], p[1]);
            let close = |a: f32, b: f32| (a.min(1e29) - b.min(1e29)).abs() <= 2e-3 * a.abs().clamp(1.0, 1e29);
            assert!(close(gpu[0], cpu.upper.min(1e30)), "upper at {p:?}: GPU {gpu:?} CPU {cpu:?}");
            assert!(close(-gpu[1], -cpu.lower.max(-1e30)), "lower at {p:?}: GPU {gpu:?} CPU {cpu:?}");
            assert!(close(-gpu[3], -cpu.lake.max(NO_LAKE)), "lake at {p:?}: GPU {gpu:?} CPU {cpu:?}");
            if cpu.bank_distance < 0.0 {
                inside += 1;
            }
            if cpu.lake > NO_LAKE {
                lakes += 1;
            }
        }
        assert!(inside > 1000, "{inside}");
        assert!(lakes > 100, "{lakes}");
    }
}
