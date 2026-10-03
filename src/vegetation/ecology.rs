//! The environment a plant sees: terrain shape around a point and the
//! ecological fields derived from it.
//!
//! Everything here is a pure function of world position. The base landform
//! comes from `noise::base_height`, the CPU mirror of the shader's
//! `baseHeight`; erosion is left to the GPU, which re-seats every root on the
//! eroded surface and drops the few that land in a channel (see
//! vegetation-cull.wgsl). The fields are deliberately smooth at the scale of
//! a stand and only add small-scale noise where nature does (glades, thicket
//! patches, the lavender's abundance), so neighbouring plants see nearly the
//! same environment and plant communities form patches instead of confetti.

use crate::constants::SEA_LEVEL;
use crate::noise::{NoiseField, base_height, grass_line_height};

// ---------------------------------------------------------------------------
// Hashing and noise
// ---------------------------------------------------------------------------

pub fn mix64(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

/// A stable 64-bit identity for an integer lattice cell of one layer.
pub fn cell_key(x: i64, z: i64, salt: u64) -> u64 {
    mix64(
        (x as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
            ^ (z as u64).wrapping_mul(0xd1b5_4a32_d192_ed03)
            ^ salt.wrapping_mul(0x632b_e59b_d9b4_e019),
    )
}

/// Uniform [0, 1) from a key and a channel.
pub fn random(key: u64, channel: u64) -> f32 {
    (mix64(key ^ channel.wrapping_mul(0x9e37_79b9_7f4a_7c15)) >> 40) as f32 / 16_777_216.0
}

fn lattice_hash(x: i64, z: i64, seed: u32) -> u32 {
    (cell_key(x, z, seed as u64 ^ 0x5eed) >> 32) as u32
}

/// Eight unit gradients; enough for smooth, isotropic-looking landscape fields.
const GRADIENTS: [[f32; 2]; 8] = [
    [1.0, 0.0],
    [0.707_106_77, 0.707_106_77],
    [0.0, 1.0],
    [-0.707_106_77, 0.707_106_77],
    [-1.0, 0.0],
    [-0.707_106_77, -0.707_106_77],
    [0.0, -1.0],
    [0.707_106_77, -0.707_106_77],
];

fn quintic(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

/// Gradient noise remapped to roughly [0, 1] (mean 0.5, sd about 0.13).
/// Coordinates are in lattice units; f64 keeps distant worlds exact.
pub fn gradient_noise(x: f64, z: f64, seed: u32) -> f32 {
    let cell_x = x.floor();
    let cell_z = z.floor();
    let fx = (x - cell_x) as f32;
    let fz = (z - cell_z) as f32;
    let ix = cell_x as i64;
    let iz = cell_z as i64;
    let corner = |dx: i64, dz: i64| {
        let g = GRADIENTS[(lattice_hash(ix + dx, iz + dz, seed) >> 29) as usize];
        g[0] * (fx - dx as f32) + g[1] * (fz - dz as f32)
    };
    let u = quintic(fx);
    let v = quintic(fz);
    let south = corner(0, 0) + (corner(1, 0) - corner(0, 0)) * u;
    let north = corner(0, 1) + (corner(1, 1) - corner(0, 1)) * u;
    let value = south + (north - south) * v;
    (0.5 + value * 0.72).clamp(0.0, 1.0)
}

/// Noise over world metres at a given wavelength, rotated per seed so no two
/// fields share lattice axes.
pub fn noise(p: [f64; 2], wavelength: f64, seed: u32) -> f32 {
    let angle = (seed as f64 * 0.618_034) * std::f64::consts::TAU;
    let (s, c) = angle.sin_cos();
    let x = (p[0] * c - p[1] * s) / wavelength;
    let z = (p[0] * s + p[1] * c) / wavelength;
    gradient_noise(x + seed as f64 * 17.13, z - seed as f64 * 9.71, seed)
}

/// Two-octave noise, still in [0, 1].
pub fn noise2(p: [f64; 2], wavelength: f64, seed: u32) -> f32 {
    0.67 * noise(p, wavelength, seed) + 0.33 * noise(p, wavelength * 0.43, seed.wrapping_add(101))
}

pub fn smoothstep(edge0: f32, edge1: f32, value: f32) -> f32 {
    let t = ((value - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

// ---------------------------------------------------------------------------
// Terrain around a point
// ---------------------------------------------------------------------------

/// Direction toward the melt-season sun the terrain shader uses for snow
/// (terrain-fs.wgsl `meltSun`). Slopes facing it are the warm, dry ones.
const SUN: [f32; 3] = [-0.22, 0.62, -0.76];

/// Terrain shape at one point.
#[derive(Clone, Copy, Debug, Default)]
pub struct Site {
    /// Base landform height, metres above sea level.
    pub height: f32,
    /// Gradient magnitude (rise over run) at an 8 m interval.
    pub steepness: f32,
    /// Sun exposure relative to level ground: + facing the sun, - shaded.
    pub insolation: f32,
    /// Mean height of a 12 m ring minus the point: + in hollows, - on noses.
    pub hollow: f32,
    /// Mean height of a 90 m ring minus the point: + in valleys, - on ridges.
    pub valley: f32,
    /// The terrain shader's grass line height here, metres above sea level:
    /// turf takes over from the beach's sand where it passes `TURF_LINE`.
    pub turf: f32,
    /// Horizontal distance down the fall line to the sea, metres;
    /// `SHORE_SEARCH` where the sea is not that close (or the point is high).
    pub shore: f32,
}

/// Where the terrain shader's turf dominates the beach's sand: its grass
/// line height (`Site::turf`) at which the grass habitat admits roots.
pub const TURF_LINE: f32 = 3.4;
/// How far down the fall line a point looks for the sea, metres, and the
/// highest ground that looks at all.
pub const SHORE_SEARCH: f32 = 150.0;
const SHORE_HEIGHT: f32 = 20.0;

/// Bilinear base-height grids around one scatter region. The landform is
/// expensive (a dozen noise fetches per height); every candidate of every
/// layer reads it through these instead.
pub struct SiteSampler {
    fine: HeightGrid,
    coarse: HeightGrid,
    /// The terrain shader's grass line height on the fine lattice.
    turf: HeightGrid,
}

struct HeightGrid {
    origin: [f64; 2],
    spacing: f64,
    width: usize,
    depth: usize,
    heights: Vec<f32>,
}

impl HeightGrid {
    fn build(noise: &NoiseField, minimum: [f64; 2], maximum: [f64; 2], spacing: f64) -> Self {
        let origin = [
            (minimum[0] / spacing).floor() * spacing,
            (minimum[1] / spacing).floor() * spacing,
        ];
        let width = ((maximum[0] - origin[0]) / spacing).ceil() as usize + 2;
        let depth = ((maximum[1] - origin[1]) / spacing).ceil() as usize + 2;
        let mut heights = Vec::with_capacity(width * depth);
        for z in 0..depth {
            for x in 0..width {
                heights.push(base_height(
                    noise,
                    (origin[0] + x as f64 * spacing) as f32,
                    (origin[1] + z as f64 * spacing) as f32,
                ));
            }
        }
        Self {
            origin,
            spacing,
            width,
            depth,
            heights,
        }
    }

    fn sample(&self, x: f64, z: f64) -> f32 {
        let gx = ((x - self.origin[0]) / self.spacing).clamp(0.0, (self.width - 1) as f64 - 1e-6);
        let gz = ((z - self.origin[1]) / self.spacing).clamp(0.0, (self.depth - 1) as f64 - 1e-6);
        let ix = gx.floor() as usize;
        let iz = gz.floor() as usize;
        let tx = (gx - ix as f64) as f32;
        let tz = (gz - iz as f64) as f32;
        let at = |x: usize, z: usize| self.heights[z.min(self.depth - 1) * self.width + x.min(self.width - 1)];
        let south = lerp(at(ix, iz), at(ix + 1, iz), tx);
        let north = lerp(at(ix, iz + 1), at(ix + 1, iz + 1), tx);
        lerp(south, north, tz)
    }
}

/// Margin the coarse valley ring needs around a sampled region.
const VALLEY_RADIUS: f64 = 90.0;
const HOLLOW_RADIUS: f64 = 12.0;

impl SiteSampler {
    /// Grids covering every point a caller will ask about, plus the stencils.
    pub fn new(noise: &NoiseField, minimum: [f64; 2], maximum: [f64; 2]) -> Self {
        let fine_margin = HOLLOW_RADIUS + 8.0;
        let coarse_margin = VALLEY_RADIUS.max(SHORE_SEARCH as f64) + 16.0;
        let fine = HeightGrid::build(
            noise,
            [minimum[0] - fine_margin, minimum[1] - fine_margin],
            [maximum[0] + fine_margin, maximum[1] + fine_margin],
            4.0,
        );
        let mut turf = HeightGrid {
            heights: Vec::with_capacity(fine.heights.len()),
            ..fine
        };
        for (index, &height) in fine.heights.iter().enumerate() {
            let x = fine.origin[0] + (index % fine.width) as f64 * fine.spacing;
            let z = fine.origin[1] + (index / fine.width) as f64 * fine.spacing;
            turf.heights.push(grass_line_height(noise, x as f32, z as f32, height));
        }
        Self {
            fine,
            coarse: HeightGrid::build(
                noise,
                [minimum[0] - coarse_margin, minimum[1] - coarse_margin],
                [maximum[0] + coarse_margin, maximum[1] + coarse_margin],
                16.0,
            ),
            turf,
        }
    }

    pub fn height(&self, x: f64, z: f64) -> f32 {
        self.fine.sample(x, z)
    }

    pub fn site(&self, x: f64, z: f64) -> Site {
        let height = self.fine.sample(x, z);
        let step = 4.0;
        let gx = (self.fine.sample(x + step, z) - self.fine.sample(x - step, z)) / (2.0 * step as f32);
        let gz = (self.fine.sample(x, z + step) - self.fine.sample(x, z - step)) / (2.0 * step as f32);
        let steepness = gx.hypot(gz);
        let normal_length = (gx * gx + gz * gz + 1.0).sqrt();
        let normal = [-gx / normal_length, 1.0 / normal_length, -gz / normal_length];
        let sun_length = (SUN[0] * SUN[0] + SUN[1] * SUN[1] + SUN[2] * SUN[2]).sqrt();
        let exposure =
            (normal[0] * SUN[0] + normal[1] * SUN[1] + normal[2] * SUN[2]) / sun_length;
        let insolation = exposure - SUN[1] / sun_length;
        let ring = |grid: &HeightGrid, radius: f64| {
            let mut sum = 0.0;
            for k in 0..8 {
                let angle = k as f64 * std::f64::consts::FRAC_PI_4;
                sum += grid.sample(x + radius * angle.cos(), z + radius * angle.sin());
            }
            sum / 8.0
        };
        Site {
            height: height - SEA_LEVEL,
            steepness,
            insolation,
            hollow: ring(&self.fine, HOLLOW_RADIUS) - height,
            valley: ring(&self.coarse, VALLEY_RADIUS) - self.coarse.sample(x, z),
            turf: self.turf.sample(x, z) - SEA_LEVEL,
            shore: self.shore_distance(x, z, height - SEA_LEVEL),
        }
    }

    /// Walks down the coarse fall line, which ignores the small bumps that
    /// would turn a local gradient inland, until the ground drops below the
    /// sea; low ground that drains to a hollow instead is no shore.
    fn shore_distance(&self, x: f64, z: f64, height: f32) -> f32 {
        if height > SHORE_HEIGHT {
            return SHORE_SEARCH;
        }
        let step = 16.0;
        let gx = self.coarse.sample(x + step, z) - self.coarse.sample(x - step, z);
        let gz = self.coarse.sample(x, z + step) - self.coarse.sample(x, z - step);
        let fall = gx.hypot(gz);
        if fall < 1e-4 {
            return SHORE_SEARCH;
        }
        let (dx, dz) = ((-gx / fall) as f64, (-gz / fall) as f64);
        let mut previous = (0.0f32, height);
        for distance in [6.0f32, 12.0, 20.0, 30.0, 44.0, 62.0, 84.0, 110.0, SHORE_SEARCH] {
            let below = self.coarse.sample(x + dx * distance as f64, z + dz * distance as f64) - SEA_LEVEL;
            if below <= 0.0 {
                let (near, above) = previous;
                return near + (distance - near) * (above / (above - below).max(1e-3)).clamp(0.0, 1.0);
            }
            previous = (distance, below);
        }
        SHORE_SEARCH
    }
}

// ---------------------------------------------------------------------------
// Ecological fields
// ---------------------------------------------------------------------------

/// Field seeds. Changing one reshuffles that field everywhere and nothing
/// else.
mod seed {
    pub const WARP_X: u32 = 11;
    pub const WARP_Z: u32 = 12;
    pub const MOSAIC_BROAD: u32 = 21;
    pub const MOSAIC_STAND: u32 = 22;
    pub const MOSAIC_EDGE: u32 = 23;
    pub const GLADE: u32 = 24;
    pub const DENSITY: u32 = 25;
    pub const PINE_HISTORY: u32 = 31;
    pub const PINE_COHORT: u32 = 32;
    pub const AGE: u32 = 41;
    pub const TREELINE: u32 = 42;
    pub const BROADLEAF: u32 = 51;
    pub const OAK: u32 = 52;
    pub const THICKET: u32 = 61;
    pub const LILAC: u32 = 62;
    pub const HERB: u32 = 71;
    pub const SHORE: u32 = 72;
    pub const LAVENDER: u32 = 81;
    pub const LAVENDER_DETAIL: u32 = 82;
    pub const LAVENDER_PATCH: u32 = 83;
}

/// Everything the scatter needs to know about a point.
#[derive(Clone, Copy, Debug, Default)]
pub struct Habitat {
    pub site: Site,
    /// 0 in meadows, clearings, on the shore and above the treeline; 1 in
    /// closed forest. The transition is the forest edge.
    pub forest: f32,
    /// The ecotone: peaks along the forest margin, a few tens of metres wide.
    pub edge: f32,
    /// Fraction of the hard-core packing a stand reaches here (0..0.97).
    pub stocking: f32,
    /// Probability a conifer here is a pine rather than a fir.
    pub pine: f32,
    /// Probability a forest tree here is a broadleaf (oak or maple).
    pub broadleaf: f32,
    /// Probability a broadleaf here is an oak rather than a maple.
    pub oak: f32,
    /// Stand maturity, 0 young regrowth to 1 old growth.
    pub age: f32,
    /// Height multiplier from site quality and exposure (krummholz near the
    /// treeline, stunted on the coast and on thin soil).
    pub vigour: f32,
    /// Soil moisture proxy, 0 dry crest to 1 wet hollow.
    pub moisture: f32,
    /// Open ground that can carry parkland oaks and maples (meadow, not
    /// shore or alpine).
    pub parkland: f32,
    /// Shrubs per square metre.
    pub shrubs: f32,
    /// Probability a shrub is a lilac rather than a common bush.
    pub lilac: f32,
    /// Broadleaf plants per square metre: colonies lining the shore.
    pub herbs: f32,
    /// Lavender tufts per square metre, after the scatter's spacing.
    pub lavender: f32,
}

/// Centre of the pine stands' threshold on their history field. Raising it
/// shrinks and thins the pine stands; it sets the fir to pine ratio.
pub const PINE_THRESHOLD: f32 = 0.612;

/// Broadleaf plants per square metre at the heart of a shore colony, about
/// one every four square metres.
const SHORE_HERB_PEAK: f32 = 0.25;

/// Lavender tufts per square metre where the abundance is fullest (the
/// tufts' spacing caps it near one every two square metres); the abundance
/// at which it gets there; and how steeply the density falls away below it
/// (each 0.1 of abundance halves it).
const LAVENDER_PEAK: f32 = 0.6;
const LAVENDER_FULL: f32 = 0.58;
const LAVENDER_SPREAD: f32 = 7.0;

/// Rough limit of tree growth above sea level, before regional and aspect
/// variation. The terrain shader thins turf into alpine rubble from 80 m and
/// keeps snow from about 112 m; the forest gives out between the two.
pub const TREELINE: f32 = 96.0;

/// Warp the world point so every field's patches bend and stretch instead of
/// sitting on noise blobs of one size.
fn warped(p: [f64; 2]) -> [f64; 2] {
    let wx = noise(p, 520.0, seed::WARP_X) as f64 - 0.5;
    let wz = noise(p, 520.0, seed::WARP_Z) as f64 - 0.5;
    [p[0] + wx * 210.0, p[1] + wz * 210.0]
}

pub fn habitat(sampler: &SiteSampler, x: f64, z: f64) -> Habitat {
    let site = sampler.site(x, z);
    habitat_at(site, [x, z])
}

/// Ground lower than this above the sea carries nothing: the waterline and
/// the wave wash. Beaches are left to the grass habitat, which keeps every
/// small plant on turf, and to the coastal fields below.
const DRY_LAND: f32 = 1.5;

pub fn habitat_at(site: Site, p: [f64; 2]) -> Habitat {
    let h = site.height;
    if h < DRY_LAND {
        return Habitat {
            site,
            ..Habitat::default()
        };
    }
    let q = warped(p);

    // Moisture: valley floors and hollows are wet, crests and sun-facing
    // slopes dry out.
    let moisture = (0.45
        + 0.30 * smoothstep(-4.0, 10.0, site.valley)
        + 0.25 * smoothstep(-0.5, 1.2, site.hollow)
        - 0.35 * smoothstep(0.0, 0.25, site.insolation)
        - 0.25 * smoothstep(-3.0, -12.0, site.valley)
        - 0.20)
        .clamp(0.0, 1.0);

    // Hard limits: salt spray and dunes along the coast, the treeline, and
    // faces too steep to hold soil.
    let coast = smoothstep(4.0, 13.0, h + (noise(q, 160.0, seed::MOSAIC_EDGE) - 0.5) * 8.0);
    let treeline = TREELINE
        + (noise(q, 700.0, seed::TREELINE) - 0.5) * 30.0
        + site.insolation * 26.0
        - smoothstep(0.0, 8.0, -site.valley) * 6.0;
    let below_treeline = 1.0 - smoothstep(treeline - 30.0, treeline + 6.0, h);
    let soil = 1.0 - smoothstep(0.62, 1.05, site.steepness);

    // The stand mosaic. Broad and stand-scale noise decide where forest
    // stands; the edge octave roughens its outline; glades punch small
    // clearings into closed forest. Wet, flat valley floors become meadow, as
    // do exposed crests above the forest belt.
    let mosaic = 0.52 * noise(q, 1100.0, seed::MOSAIC_BROAD)
        + 0.33 * noise(q, 330.0, seed::MOSAIC_STAND)
        + 0.15 * noise(q, 95.0, seed::MOSAIC_EDGE);
    let flat = 1.0 - smoothstep(0.05, 0.16, site.steepness);
    let wet_meadow = smoothstep(2.0, 9.0, site.valley) * flat * smoothstep(0.55, 0.85, moisture);
    let crest = smoothstep(4.0, 14.0, -site.valley) * smoothstep(55.0, 85.0, h);
    let signal = mosaic - 0.16 * wet_meadow - 0.08 * crest
        - 0.10 * smoothstep(0.45, 0.9, site.steepness);
    const FOREST_THRESHOLD: f32 = 0.388;
    let forest_core = smoothstep(FOREST_THRESHOLD - 0.035, FOREST_THRESHOLD + 0.035, signal);
    let glade = smoothstep(0.70, 0.76, noise(q, 70.0, seed::GLADE)) * 0.9;
    let forest = (forest_core * (1.0 - glade) * coast * below_treeline * soil).clamp(0.0, 1.0);
    // Ecotone: strongest where the mosaic signal crosses its threshold, plus
    // the rims of glades, the coastal fringe and the treeline.
    let margin = 1.0 - smoothstep(0.0, 0.05, (signal - FOREST_THRESHOLD).abs());
    let edge = (margin.max(glade * (1.0 - glade) * 4.0 * forest_core)
        .max(coast * (1.0 - coast) * 4.0 * forest_core)
        * below_treeline
        * smoothstep(3.0, 7.0, h)
        * soil)
        .clamp(0.0, 1.0);

    // Stocking: most of the forest is closed; some stands are open woodland.
    let stocking = forest
        * lerp(0.62, 0.95, smoothstep(0.30, 0.62, noise(q, 260.0, seed::DENSITY)))
        * lerp(1.0, 0.55, edge)
        * lerp(1.0, 0.45, smoothstep(treeline - 30.0, treeline, h));

    // Fir dominates; pine takes the warm, dry, well-drained ground, the
    // sandy lowlands near the coast, and the stands a disturbance history
    // handed to it. The history field has a sharp threshold, so stands are
    // mostly one or the other and mix only along their borders, and a
    // cohort octave scatters groups of the other species through both.
    let dry_site = 0.45 * smoothstep(-0.05, 0.25, site.insolation)
        + 0.30 * smoothstep(1.0, 8.0, -site.valley)
        + 0.25 * smoothstep(-0.2, -1.0, site.hollow);
    let wet_site = 0.6 * moisture + 0.4 * smoothstep(40.0, 90.0, h);
    let history = 0.7 * noise(q, 380.0, seed::PINE_HISTORY) + 0.3 * noise(q, 115.0, seed::PINE_COHORT);
    let pine_signal = history + 0.18 * dry_site - 0.16 * wet_site
        + 0.06 * (1.0 - smoothstep(8.0, 25.0, h));
    // Calibrated for three firs to every pine (see the scatter tests).
    let pine = 0.05 + 0.87 * smoothstep(PINE_THRESHOLD - 0.04, PINE_THRESHOLD + 0.04, pine_signal);

    // Broadleaves: a scattering through the conifers, more along edges and
    // in warm, moist lowland valleys; none toward the treeline.
    let lowland = 1.0 - smoothstep(35.0, 70.0, h);
    let broadleaf = (0.012 + 0.035 * edge + 0.03 * lowland * moisture)
        * (0.5 + noise(q, 400.0, seed::BROADLEAF))
        * below_treeline;
    let oak = (0.55 + 0.35 * smoothstep(-0.05, 0.2, site.insolation) - 0.35 * moisture
        + 0.2 * (noise(q, 600.0, seed::OAK) - 0.5))
        .clamp(0.1, 0.9);

    // Maturity and growth: old stands of tall trees and younger regrowth,
    // stunted toward the treeline, on the coast and on steep, thin soil.
    let age = smoothstep(0.30, 0.70, noise(q, 380.0, seed::AGE));
    let vigour = (1.0
        - 0.55 * smoothstep(treeline - 40.0, treeline + 4.0, h)
        - 0.25 * (1.0 - smoothstep(6.0, 22.0, h))
        - 0.20 * smoothstep(0.35, 0.9, site.steepness)
        + 0.10 * (moisture - 0.5))
        .clamp(0.30, 1.05);

    // Open ground that is not shore or alpine: meadows and clearings.
    let open = (1.0 - forest) * coast * below_treeline * soil;
    let parkland = open * (1.0 - wet_meadow * 0.7) * lowland.max(0.35);

    // Shrubs: a thin understory, dense belts along the forest edge,
    // thickets scattered through clearings.
    let thicket = smoothstep(0.52, 0.72, noise(q, 85.0, seed::THICKET));
    let shrubs = (0.0009 * forest * (1.0 - 0.5 * stocking)
        + 0.0095 * edge * (0.4 + 0.6 * thicket)
        + 0.0026 * open * thicket)
        * soil
        * (1.0 - smoothstep(treeline - 10.0, treeline + 12.0, h));
    let lilac = (0.20 + 0.35 * edge + 0.25 * open * thicket
        - 0.15 * smoothstep(40.0, 80.0, h))
        * (0.4 + 1.2 * smoothstep(0.35, 0.75, noise(q, 220.0, seed::LILAC)));

    // Broadleaf plants: the herbs of the backshore, in the strip of turf the
    // beach gives way to. The strip follows the terrain shader's own grass
    // line, so it starts where the sand ends on wide beaches and in narrow
    // coves alike, and runs some 25 m inland on any slope (its height span
    // is the slope times that width), its inland side wandering. Across it,
    // a few pioneers stand at the sand's edge, the plants thicken a few
    // metres in and thin out into the meadow; along it they gather in loose
    // colonies with open turf between, and the scatter keeps them apart.
    let width = (site.steepness * 26.0).clamp(0.7, 8.0);
    let reach = width * (0.85 + 0.5 * (noise(q, 90.0, seed::SHORE) - 0.5));
    let across = (site.turf - TURF_LINE) / reach;
    let strip = smoothstep(-0.12, 0.35, across) * (1.0 - smoothstep(0.55, 1.0, across));
    let near_sea = 1.0 - smoothstep(0.6 * SHORE_SEARCH, SHORE_SEARCH, site.shore);
    let colony = smoothstep(0.38, 0.64, noise2(q, 40.0, seed::HERB));
    let herbs = SHORE_HERB_PEAK * strip * near_sea * (0.15 + 0.85 * colony) * (1.0 - 0.8 * forest);

    // Lavender: single tufts through sunny, dry, well-drained open ground
    // below the subalpine belt, kept apart by the scatter. How many varies
    // smoothly and widely, as wild lavender does: a regional, a stand and a
    // patch scale combine into a log-normal abundance, from a tuft every
    // hundred square metres or so to one every two or three, so no clearing
    // is quite like the next and none is a solid carpet.
    let sunny_open = (1.0 - forest).powi(2)
        * smoothstep(-0.20, 0.12, site.insolation)
        * (1.0 - smoothstep(0.55, 0.85, moisture))
        * (1.0 - smoothstep(55.0, 80.0, h))
        * smoothstep(5.0, 10.0, h)
        * (1.0 - smoothstep(0.30, 0.55, site.steepness));
    let abundance = 0.50 * noise(q, 260.0, seed::LAVENDER)
        + 0.32 * noise(q, 70.0, seed::LAVENDER_PATCH)
        + 0.18 * noise(q, 18.0, seed::LAVENDER_DETAIL);
    let lavender =
        LAVENDER_PEAK * sunny_open * (LAVENDER_SPREAD * (abundance - LAVENDER_FULL)).exp().min(1.0);

    Habitat {
        site,
        forest,
        edge,
        stocking,
        pine,
        broadleaf,
        oak,
        age,
        vigour,
        moisture,
        parkland,
        shrubs,
        lilac: lilac.clamp(0.0, 0.95),
        herbs,
        lavender,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noise_is_bounded_smooth_and_deterministic() {
        let mut sum = 0.0;
        let mut count = 0.0;
        for i in 0..2000 {
            let p = [i as f64 * 37.3 - 9000.0, (i * i % 977) as f64 * 11.1];
            let a = noise(p, 120.0, 7);
            assert!((0.0..=1.0).contains(&a));
            assert_eq!(a, noise(p, 120.0, 7));
            // One metre is a small fraction of the wavelength.
            let b = noise([p[0] + 1.0, p[1]], 120.0, 7);
            assert!((a - b).abs() < 0.05, "{a} {b}");
            sum += a;
            count += 1.0;
        }
        let mean = sum / count;
        assert!((0.42..0.58).contains(&mean), "{mean}");
    }

    #[test]
    fn far_coordinates_stay_finite() {
        for p in [[1.0e6, -2.5e6], [-7.3e5, 4.4e5]] {
            let value = noise(p, 95.0, 3);
            assert!(value.is_finite() && (0.0..=1.0).contains(&value));
        }
    }
}
