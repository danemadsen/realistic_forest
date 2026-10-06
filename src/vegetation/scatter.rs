//! Where every plant stands.
//!
//! The world is cut into fixed 256 m chunks, each generated independently and
//! deterministically from its integer coordinates, so streaming order never
//! changes the forest. Plants come from stacked layers, each a size-marked
//! hard-core point process (Matérn type II, the model forest ecology uses for
//! competition between trees):
//!
//! 1. **Canopy.** Jittered candidates are thinned by the local stocking; a
//!    survivor keeps its place only if no higher-priority candidate stands
//!    within the sum of their crown reaches. Priority rises with height, so
//!    tall trees win space and dense stands come out evenly spaced, while
//!    sparse woodland keeps the clumpy randomness of its thinning.
//! 2. **Regeneration.** Saplings and poles fill the gaps the canopy leaves,
//!    most of them along edges and in open stands.
//! 3. **Shrubs** grow under crown edges and in belts along the forest margin.
//! 4. **Broadleaf plants and lavender** take the ground the trees and shrubs
//!    leave: the broadleaf plants in colonies along the shore, the lavender
//!    as single tufts, kept apart, scattered through sunny clearings.
//!
//! Every test of a candidate reads only candidates within a bounded distance,
//! so a chunk computed alone matches the same ground computed with its
//! neighbours.

use super::assets::{Species, VegetationAssets};
use super::ecology::{self, Habitat, SiteSampler, cell_key, random, smoothstep};
use crate::noise::NoiseField;

/// Side of a scatter chunk, metres.
pub const CHUNK_SIZE: f64 = 256.0;

/// One plant as uploaded to the GPU. 32 bytes, mirrored by `PlantInstance`
/// in vegetation-cull.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PlantInstance {
    /// World X, base-landform height and world Z. The height is only a
    /// culling estimate; the GPU seats the root on the eroded surface.
    pub position: [f32; 3],
    pub scale: f32,
    pub yaw: f32,
    /// Stable per-plant random in [0, 1): tint, lean and wind phase.
    pub seed: f32,
    /// Index into the model table.
    pub model: u32,
    pub layer: u32,
}

const _: () = assert!(std::mem::size_of::<PlantInstance>() == 32);

/// Which layer produced a plant; also its streaming detail level.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Layer {
    Canopy = 0,
    Regeneration = 1,
    Shrub = 2,
    Herb = 3,
    Lavender = 4,
}

/// Detail levels a chunk is generated at: trees only, then shrubs, then the
/// ground layers, each wanted closer to the camera than the last.
pub const LEVEL_TREES: u8 = 0;
pub const LEVEL_SHRUBS: u8 = 1;
pub const LEVEL_GROUND: u8 = 2;

// ---------------------------------------------------------------------------
// Catalogue
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct CatalogModel {
    pub species: Species,
    pub form: String,
    pub height: f32,
    pub crown_radius: f32,
}

/// The model metrics the scatter needs, in model-table order.
#[derive(Clone, Debug, Default)]
pub struct Catalog {
    pub models: Vec<CatalogModel>,
}

/// Height bands (metres) a growth form is used for. Firs are scaled up: the
/// pack's firs are young, 9 m trees, but they dominate this forest, so its
/// "large" form stands in for 11-21 m firs, beside 12-34 m pines.
const FORMS: &[(Species, &str, f32, f32)] = &[
    (Species::Fir, "small", 0.9, 2.8),
    (Species::Fir, "medium", 2.8, 11.0),
    (Species::Fir, "large", 11.0, 21.5),
    (Species::Pine, "sapling", 0.8, 2.4),
    (Species::Pine, "small", 2.4, 7.0),
    (Species::Pine, "medium", 7.0, 14.5),
    (Species::Pine, "big", 14.5, 22.0),
    (Species::Pine, "large", 22.0, 34.0),
    (Species::Oak, "sapling", 0.6, 1.8),
    (Species::Oak, "small", 1.8, 8.0),
    (Species::Oak, "medium", 8.0, 15.0),
    (Species::Oak, "big", 15.0, 23.0),
    (Species::Oak, "large", 23.0, 30.0),
    (Species::Maple, "sapling", 1.5, 5.0),
    (Species::Maple, "small", 5.0, 9.0),
    (Species::Maple, "medium", 9.0, 16.0),
    (Species::Maple, "large", 16.0, 26.0),
    (Species::Lilac, "sapling", 1.4, 2.8),
    (Species::Lilac, "small", 2.0, 4.6),
    (Species::Lilac, "", 3.6, 5.6),
    (Species::Lilac, "single-tree", 5.6, 8.0),
    (Species::Bush, "small", 0.35, 0.75),
    (Species::Bush, "medium", 0.7, 1.3),
    (Species::Bush, "big", 1.2, 2.0),
    (Species::Broadleaf, "", 0.3, 0.7),
    (Species::Lavender, "", 0.6, 0.95),
];

/// No model is stretched or shrunk past these factors of its authored size,
/// except the lavender tuft: a plain star of cards, authored 2 m tall, that
/// takes any scale.
const MIN_SCALE: f32 = 0.5;
const MAX_SCALE: f32 = 2.6;
const MIN_LAVENDER_SCALE: f32 = 0.25;

impl Catalog {
    pub fn from_assets(assets: &VegetationAssets) -> Self {
        Self {
            models: assets
                .models
                .iter()
                .map(|model| CatalogModel {
                    species: model.species,
                    form: model.form.clone(),
                    height: model.height,
                    crown_radius: model.crown_radius,
                })
                .collect(),
        }
    }

    /// A model of `species` for a plant `height` metres tall, and the scale
    /// that gives it that height. Among the forms whose band holds the height
    /// one is chosen at random, then a variant.
    pub fn choose(&self, species: Species, height: f32, key: u64) -> Option<(u32, f32)> {
        let bands: Vec<&(Species, &str, f32, f32)> =
            FORMS.iter().filter(|band| band.0 == species).collect();
        if bands.is_empty() {
            return None;
        }
        let low = bands.iter().map(|band| band.2).fold(f32::INFINITY, f32::min);
        let high = bands.iter().map(|band| band.3).fold(0.0, f32::max);
        let height = height.clamp(low, high);
        let fitting: Vec<&&(Species, &str, f32, f32)> = bands
            .iter()
            .filter(|band| height >= band.2 && height <= band.3)
            .collect();
        let band = fitting[((random(key, 40) * fitting.len() as f32) as usize).min(fitting.len() - 1)];
        let matching_models = self
            .models
            .iter()
            .enumerate()
            .filter(|(_, model)| model.species == species && model.form == band.1);
        let variants: Vec<usize> = matching_models.map(|(index, _)| index).collect();
        if variants.is_empty() {
            return None;
        }
        let model = variants[((random(key, 41) * variants.len() as f32) as usize).min(variants.len() - 1)];
        let smallest = if species == Species::Lavender { MIN_LAVENDER_SCALE } else { MIN_SCALE };
        let scale = (height / self.models[model].height).clamp(smallest, MAX_SCALE);
        Some((model as u32, scale))
    }
}

// ---------------------------------------------------------------------------
// Habitat lattice
// ---------------------------------------------------------------------------

/// Ecological fields evaluated every 4 m and interpolated, so the dozen noise
/// fields cost one evaluation per 16 m² rather than one per candidate. The
/// finest field (the lavender's patch scale) has an 18 m wavelength.
struct HabitatGrid {
    origin: [f64; 2],
    width: usize,
    depth: usize,
    cells: Vec<Habitat>,
}

const HABITAT_SPACING: f64 = 4.0;

impl HabitatGrid {
    fn build(sampler: &SiteSampler, minimum: [f64; 2], maximum: [f64; 2]) -> Self {
        let origin = [
            (minimum[0] / HABITAT_SPACING).floor() * HABITAT_SPACING,
            (minimum[1] / HABITAT_SPACING).floor() * HABITAT_SPACING,
        ];
        let width = ((maximum[0] - origin[0]) / HABITAT_SPACING).ceil() as usize + 2;
        let depth = ((maximum[1] - origin[1]) / HABITAT_SPACING).ceil() as usize + 2;
        let mut cells = Vec::with_capacity(width * depth);
        for z in 0..depth {
            for x in 0..width {
                cells.push(ecology::habitat(
                    sampler,
                    origin[0] + x as f64 * HABITAT_SPACING,
                    origin[1] + z as f64 * HABITAT_SPACING,
                ));
            }
        }
        Self {
            origin,
            width,
            depth,
            cells,
        }
    }

    fn sample(&self, x: f64, z: f64) -> Habitat {
        let gx = ((x - self.origin[0]) / HABITAT_SPACING).clamp(0.0, (self.width - 1) as f64 - 1e-6);
        let gz = ((z - self.origin[1]) / HABITAT_SPACING).clamp(0.0, (self.depth - 1) as f64 - 1e-6);
        let ix = gx.floor() as usize;
        let iz = gz.floor() as usize;
        let tx = (gx - ix as f64) as f32;
        let tz = (gz - iz as f64) as f32;
        let at = |x: usize, z: usize| &self.cells[z.min(self.depth - 1) * self.width + x.min(self.width - 1)];
        let south = lerp_habitat(at(ix, iz), at(ix + 1, iz), tx);
        let north = lerp_habitat(at(ix, iz + 1), at(ix + 1, iz + 1), tx);
        lerp_habitat(&south, &north, tz)
    }
}

fn lerp_habitat(a: &Habitat, b: &Habitat, t: f32) -> Habitat {
    let l = |x: f32, y: f32| x + (y - x) * t;
    Habitat {
        site: ecology::Site {
            height: l(a.site.height, b.site.height),
            steepness: l(a.site.steepness, b.site.steepness),
            insolation: l(a.site.insolation, b.site.insolation),
            hollow: l(a.site.hollow, b.site.hollow),
            valley: l(a.site.valley, b.site.valley),
            turf: l(a.site.turf, b.site.turf),
            shore: l(a.site.shore, b.site.shore),
            river: l(a.site.river, b.site.river),
        },
        forest: l(a.forest, b.forest),
        edge: l(a.edge, b.edge),
        stocking: l(a.stocking, b.stocking),
        pine: l(a.pine, b.pine),
        broadleaf: l(a.broadleaf, b.broadleaf),
        oak: l(a.oak, b.oak),
        age: l(a.age, b.age),
        vigour: l(a.vigour, b.vigour),
        moisture: l(a.moisture, b.moisture),
        parkland: l(a.parkland, b.parkland),
        shrubs: l(a.shrubs, b.shrubs),
        lilac: l(a.lilac, b.lilac),
        herbs: l(a.herbs, b.herbs),
        lavender: l(a.lavender, b.lavender),
    }
}

// ---------------------------------------------------------------------------
// Candidates and the hard-core process
// ---------------------------------------------------------------------------

/// Layer parameters: candidate lattice, identity salt and how much of their
/// summed crown reach two plants of the layer may not share.
struct LayerSpec {
    spacing: f64,
    salt: u64,
    /// Two plants conflict when closer than `overlap * (reach_a + reach_b)`.
    overlap: f32,
}

const CANOPY: LayerSpec = LayerSpec { spacing: 3.0, salt: 0xC4_0001, overlap: 0.66 };
const REGENERATION: LayerSpec = LayerSpec { spacing: 3.5, salt: 0xC4_0002, overlap: 0.80 };
const SHRUB: LayerSpec = LayerSpec { spacing: 2.2, salt: 0xC4_0003, overlap: 0.75 };
const HERB: LayerSpec = LayerSpec { spacing: 1.3, salt: 0xC4_0004, overlap: 1.0 };
const LAVENDER: LayerSpec = LayerSpec { spacing: 0.7, salt: 0xC4_0005, overlap: LAVENDER_OVERLAP };
/// Two lavender tufts stand at least this many times their summed crown
/// reaches apart: never touching, however dense the patch.
pub const LAVENDER_OVERLAP: f32 = 1.15;

/// Widest lavender tuft (a 0.95 m tuft of the 2 m, 1 m-wide model): bounds
/// the tufts' conflict search.
const MAX_LAVENDER_REACH: f32 = 0.5;

/// Lowest ground above the sea a shore plant roots in; the grass habitat on
/// the GPU also keeps it above the wave crests.
const LOWEST_SHORE_ROOT: f32 = 1.2;

/// Widest broadleaf plant (0.65 m of the smallest model, scaled up):
/// bounds the shore plants' conflict search.
const MAX_BROADLEAF_REACH: f32 = 0.95;

/// Largest crown reach any canopy tree reaches (a 30 m oak at full scale),
/// which bounds every conflict search.
const MAX_CANOPY_REACH: f32 = 13.0;

#[derive(Clone, Copy, Debug)]
struct Candidate {
    x: f64,
    z: f64,
    eligible: bool,
    priority: f32,
    /// Crown reach (radius) at the chosen scale.
    reach: f32,
    model: u32,
    scale: f32,
    key: u64,
}

impl Candidate {
    fn instance(&self, sampler: &SiteSampler, layer: Layer) -> PlantInstance {
        PlantInstance {
            position: [self.x as f32, sampler.height(self.x, self.z), self.z as f32],
            scale: self.scale,
            yaw: random(self.key, 50) * std::f32::consts::TAU,
            seed: random(self.key, 51),
            model: self.model,
            layer: layer as u32,
        }
    }
}

/// A jittered lattice of candidates covering a rectangle.
struct CandidateGrid {
    min_cell: [i64; 2],
    width: usize,
    depth: usize,
    spacing: f64,
    cells: Vec<Candidate>,
}

impl CandidateGrid {
    fn build(
        spec: &LayerSpec,
        minimum: [f64; 2],
        maximum: [f64; 2],
        mut assign: impl FnMut(f64, f64, u64) -> Option<(f32, f32, u32, f32)>,
    ) -> Self {
        let min_cell = [
            (minimum[0] / spec.spacing).floor() as i64,
            (minimum[1] / spec.spacing).floor() as i64,
        ];
        let max_cell = [
            (maximum[0] / spec.spacing).ceil() as i64,
            (maximum[1] / spec.spacing).ceil() as i64,
        ];
        let width = (max_cell[0] - min_cell[0]).max(0) as usize;
        let depth = (max_cell[1] - min_cell[1]).max(0) as usize;
        let mut cells = Vec::with_capacity(width * depth);
        for iz in min_cell[1]..min_cell[1] + depth as i64 {
            for ix in min_cell[0]..min_cell[0] + width as i64 {
                let key = cell_key(ix, iz, spec.salt);
                let x = (ix as f64 + random(key, 0) as f64) * spec.spacing;
                let z = (iz as f64 + random(key, 1) as f64) * spec.spacing;
                let candidate = match assign(x, z, key) {
                    Some((priority, reach, model, scale)) => Candidate {
                        x,
                        z,
                        eligible: true,
                        priority,
                        reach,
                        model,
                        scale,
                        key,
                    },
                    None => Candidate {
                        x,
                        z,
                        eligible: false,
                        priority: 0.0,
                        reach: 0.0,
                        model: 0,
                        scale: 1.0,
                        key,
                    },
                };
                cells.push(candidate);
            }
        }
        Self {
            min_cell,
            width,
            depth,
            spacing: spec.spacing,
            cells,
        }
    }

    /// Matérn II test: no eligible candidate of higher priority within the
    /// conflict distance. Ties go to the larger key, which is unique.
    fn survives(&self, index: usize, overlap: f32, max_reach: f32) -> bool {
        let a = &self.cells[index];
        if !a.eligible {
            return false;
        }
        if overlap <= 0.0 {
            return true;
        }
        let search = (overlap * (a.reach + max_reach)) as f64;
        let cell_x = (a.x / self.spacing).floor() as i64;
        let cell_z = (a.z / self.spacing).floor() as i64;
        let span = (search / self.spacing).ceil() as i64 + 1;
        for iz in (cell_z - span).max(self.min_cell[1])..=(cell_z + span).min(self.min_cell[1] + self.depth as i64 - 1) {
            for ix in (cell_x - span).max(self.min_cell[0])..=(cell_x + span).min(self.min_cell[0] + self.width as i64 - 1) {
                let other = self.at(ix, iz);
                if !other.eligible || other.key == a.key {
                    continue;
                }
                if other.priority < a.priority || (other.priority == a.priority && other.key < a.key) {
                    continue;
                }
                let limit = (overlap * (a.reach + other.reach)) as f64;
                let dx = other.x - a.x;
                let dz = other.z - a.z;
                if dx * dx + dz * dz < limit * limit {
                    return false;
                }
            }
        }
        true
    }

    fn at(&self, ix: i64, iz: i64) -> &Candidate {
        let x = (ix - self.min_cell[0]) as usize;
        let z = (iz - self.min_cell[1]) as usize;
        &self.cells[z * self.width + x]
    }

    /// Surviving candidates whose position lies inside the rectangle.
    fn survivors(&self, overlap: f32, max_reach: f32, minimum: [f64; 2], maximum: [f64; 2]) -> Vec<Candidate> {
        (0..self.cells.len())
            .filter(|&index| {
                let c = &self.cells[index];
                c.eligible
                    && c.x >= minimum[0]
                    && c.x < maximum[0]
                    && c.z >= minimum[1]
                    && c.z < maximum[1]
                    && self.survives(index, overlap, max_reach)
            })
            .map(|index| self.cells[index])
            .collect()
    }
}

/// A placed plant's position and crown reach.
type Placed = (f64, f64, f32);

/// Plants already placed, bucketed for "is anything within r" queries.
struct Occupancy {
    bucket: f64,
    buckets: std::collections::HashMap<(i64, i64), Vec<Placed>>,
    max_reach: f32,
}

impl Occupancy {
    fn new(bucket: f64) -> Self {
        Self {
            bucket,
            buckets: Default::default(),
            max_reach: 0.0,
        }
    }

    fn insert(&mut self, x: f64, z: f64, reach: f32) {
        let key = ((x / self.bucket).floor() as i64, (z / self.bucket).floor() as i64);
        self.buckets.entry(key).or_default().push((x, z, reach));
        self.max_reach = self.max_reach.max(reach);
    }

    /// Whether any occupant is within `own + factor * occupant_reach`.
    fn blocks(&self, x: f64, z: f64, own: f32, factor: f32) -> bool {
        let search = (own + factor * self.max_reach) as f64;
        let span = (search / self.bucket).ceil() as i64;
        let cx = (x / self.bucket).floor() as i64;
        let cz = (z / self.bucket).floor() as i64;
        for bz in cz - span..=cz + span {
            for bx in cx - span..=cx + span {
                let Some(list) = self.buckets.get(&(bx, bz)) else {
                    continue;
                };
                for &(ox, oz, reach) in list {
                    let limit = (own + factor * reach) as f64;
                    let dx = ox - x;
                    let dz = oz - z;
                    if dx * dx + dz * dz < limit * limit {
                        return true;
                    }
                }
            }
        }
        false
    }
}

// ---------------------------------------------------------------------------
// Layer rules
// ---------------------------------------------------------------------------

/// Probability that a lattice cell of `area` m² holds an eligible candidate,
/// so that a hard-core process of conflict area `conflict` reaches
/// `fraction` of its packing limit (inverting `(1 - e^(-λA)) / A`).
fn eligibility(fraction: f32, conflict_area: f32, cell_area: f32) -> f32 {
    let fraction = fraction.clamp(0.0, 0.97);
    if fraction <= 0.0 {
        return 0.0;
    }
    let intensity = -(1.0 - fraction).ln() / conflict_area.max(0.5);
    (intensity * cell_area).clamp(0.0, 1.0)
}

/// Conifers' and broadleaves' canopy heights for a stand of this age and
/// site, with individual spread.
fn canopy_height(species: Species, habitat: &Habitat, key: u64) -> f32 {
    // Stands are even-aged only loosely: each tree's maturity scatters
    // widely about the stand's, and growth spreads it further.
    let maturity = (habitat.age * 0.55 + random(key, 20) * 0.45).clamp(0.0, 1.0);
    let spread = 0.80 + 0.40 * random(key, 21);
    let base = match species {
        Species::Fir => 10.0 + 11.0 * maturity,
        Species::Pine => 12.0 + 21.0 * maturity.powf(0.9),
        Species::Oak => 9.0 + 19.0 * maturity,
        Species::Maple => 9.0 + 15.0 * maturity,
        _ => 4.0,
    };
    base * habitat.vigour * spread
}

fn canopy_species(habitat: &Habitat, key: u64) -> Species {
    if random(key, 10) < habitat.broadleaf {
        if random(key, 11) < habitat.oak {
            Species::Oak
        } else {
            Species::Maple
        }
    } else if random(key, 12) < habitat.pine {
        Species::Pine
    } else {
        Species::Fir
    }
}

/// Crown reach of a candidate at its chosen scale.
fn reach(catalog: &Catalog, model: u32, scale: f32) -> f32 {
    catalog.models[model as usize].crown_radius * scale
}

/// Parkland trees per square metre of open meadow: a few solitary oaks and
/// maples per hectare, the landmarks of a clearing.
const PARKLAND_DENSITY: f32 = 1.0 / 2600.0;

fn canopy_candidate(
    catalog: &Catalog,
    habitats: &HabitatGrid,
    x: f64,
    z: f64,
    key: u64,
) -> Option<(f32, f32, u32, f32)> {
    let habitat = habitats.sample(x, z);
    let cell_area = (CANOPY.spacing * CANOPY.spacing) as f32;
    let roll = random(key, 2);
    // Typical conflict area of a closed stand of mature firs.
    let forest_probability = eligibility(habitat.stocking, 32.0, cell_area);
    let parkland_probability = habitat.parkland * PARKLAND_DENSITY * cell_area
        * (0.3 + 1.4 * smoothstep(0.45, 0.7, ecology::noise([x, z], 160.0, 91)));
    let (species, height) = if roll < forest_probability {
        let species = canopy_species(&habitat, key);
        (species, canopy_height(species, &habitat, key))
    } else if roll < forest_probability + parkland_probability {
        // Open-grown veterans: broad, tall oaks and maples.
        let species = if random(key, 11) < habitat.oak { Species::Oak } else { Species::Maple };
        let maturity = 0.55 + 0.45 * random(key, 20);
        let height = match species {
            Species::Oak => 14.0 + 15.0 * maturity,
            _ => 12.0 + 13.0 * maturity,
        } * habitat.vigour.max(0.75);
        (species, height)
    } else {
        return None;
    };
    let (model, scale) = catalog.choose(species, height, key)?;
    let reach = reach(catalog, model, scale);
    // Tall trees win: competition for light is asymmetric.
    let priority = 0.62 * (height / 34.0).min(1.0) + 0.38 * random(key, 3);
    Some((priority, reach, model, scale))
}

fn regeneration_candidate(
    catalog: &Catalog,
    habitats: &HabitatGrid,
    x: f64,
    z: f64,
    key: u64,
) -> Option<(f32, f32, u32, f32)> {
    let habitat = habitats.sample(x, z);
    // Young trees need light: gaps in open stands, edges, the treeline.
    let openness = 1.0 - habitat.stocking;
    let density = habitat.forest.max(habitat.edge * 0.8)
        * (0.003 + 0.012 * habitat.edge + 0.008 * openness)
        + habitat.parkland * 0.0004;
    if random(key, 2) >= density * (REGENERATION.spacing * REGENERATION.spacing) as f32 {
        return None;
    }
    // Fir regenerates in shade; pine and broadleaves only in the open.
    let shade_tolerance = 1.0 - openness;
    let species = if random(key, 10) < habitat.broadleaf * 1.5 + habitat.parkland * 0.05 {
        if random(key, 11) < habitat.oak { Species::Oak } else { Species::Maple }
    } else if random(key, 12) < habitat.pine * (1.0 - 0.6 * shade_tolerance) {
        Species::Pine
    } else {
        Species::Fir
    };
    let height = (0.9 + 7.5 * random(key, 20).powf(1.7)) * habitat.vigour.max(0.5);
    let (model, scale) = catalog.choose(species, height, key)?;
    let priority = 0.62 * (height / 34.0) + 0.38 * random(key, 3);
    Some((priority, reach(catalog, model, scale), model, scale))
}

fn shrub_candidate(
    catalog: &Catalog,
    habitats: &HabitatGrid,
    x: f64,
    z: f64,
    key: u64,
) -> Option<(f32, f32, u32, f32)> {
    let habitat = habitats.sample(x, z);
    if random(key, 2) >= habitat.shrubs * (SHRUB.spacing * SHRUB.spacing) as f32 {
        return None;
    }
    let (species, height) = if random(key, 10) < habitat.lilac {
        // Lilac: dense multi-stem bushes, with the odd small tree at an edge.
        let roll = random(key, 20);
        let height = if roll < 0.25 {
            1.5 + 1.2 * random(key, 21)
        } else if roll < 0.80 - 0.25 * habitat.edge {
            2.4 + 3.0 * random(key, 21)
        } else {
            5.6 + 2.2 * random(key, 21)
        };
        (Species::Lilac, height)
    } else {
        let roll = random(key, 20);
        let height = if roll < 0.35 {
            0.4 + 0.3 * random(key, 21)
        } else if roll < 0.75 {
            0.75 + 0.5 * random(key, 21)
        } else {
            1.25 + 0.7 * random(key, 21)
        } * (1.0 + 0.25 * habitat.edge);
        (Species::Bush, height)
    };
    let (model, scale) = catalog.choose(species, height, key)?;
    Some((random(key, 3), reach(catalog, model, scale), model, scale))
}

fn herb_candidate(
    catalog: &Catalog,
    habitats: &HabitatGrid,
    x: f64,
    z: f64,
    key: u64,
) -> Option<(f32, f32, u32, f32)> {
    let habitat = habitats.sample(x, z);
    // The strip's density is interpolated between lattice points; on a steep
    // shore that would carry it past the waterline, so test the root itself.
    if habitat.site.height < LOWEST_SHORE_ROOT || habitat.herbs <= 0.0 {
        return None;
    }
    // Knee-high, standing out of the turf around them.
    let height = 0.35 + 0.3 * random(key, 20).powf(0.7);
    let (model, scale) = catalog.choose(Species::Broadleaf, height, key)?;
    let reach = reach(catalog, model, scale);
    let conflict = std::f32::consts::PI * (HERB.overlap * 2.0 * reach).powi(2);
    let cell_area = (HERB.spacing * HERB.spacing) as f32;
    if random(key, 2) >= eligibility(habitat.herbs * conflict, conflict, cell_area) {
        return None;
    }
    Some((random(key, 3), reach, model, scale))
}

fn lavender_candidate(
    catalog: &Catalog,
    habitats: &HabitatGrid,
    x: f64,
    z: f64,
    key: u64,
) -> Option<(f32, f32, u32, f32)> {
    let habitat = habitats.sample(x, z);
    if habitat.lavender <= 0.0 {
        return None;
    }
    // Tufts stand clear of the grass, from 0.6 m to their flower spikes'
    // full 0.95 m.
    let height = 0.6 + 0.35 * random(key, 20).powf(0.8);
    let (model, scale) = catalog.choose(Species::Lavender, height, key)?;
    let reach = reach(catalog, model, scale);
    // Enough candidates that, once neighbours closer than the layer's
    // spacing have thinned them, the tufts left match the habitat's density.
    let conflict = std::f32::consts::PI * (LAVENDER.overlap * 2.0 * reach).powi(2);
    let cell_area = (LAVENDER.spacing * LAVENDER.spacing) as f32;
    if random(key, 2) >= eligibility(habitat.lavender * conflict, conflict, cell_area) {
        return None;
    }
    Some((random(key, 3), reach, model, scale))
}

// ---------------------------------------------------------------------------
// Chunks
// ---------------------------------------------------------------------------

/// Chunk containing a world point.
pub fn chunk_of(x: f64, z: f64) -> [i64; 2] {
    [(x / CHUNK_SIZE).floor() as i64, (z / CHUNK_SIZE).floor() as i64]
}

pub fn chunk_bounds(chunk: [i64; 2]) -> ([f64; 2], [f64; 2]) {
    let minimum = [chunk[0] as f64 * CHUNK_SIZE, chunk[1] as f64 * CHUNK_SIZE];
    (minimum, [minimum[0] + CHUNK_SIZE, minimum[1] + CHUNK_SIZE])
}

fn grow(minimum: [f64; 2], maximum: [f64; 2], margin: f64) -> ([f64; 2], [f64; 2]) {
    (
        [minimum[0] - margin, minimum[1] - margin],
        [maximum[0] + margin, maximum[1] + margin],
    )
}

/// How far around a chunk trees must be known for the layers that avoid
/// them. Lavender keeps out of a crown's shade up to ~11.5 m from a trunk,
/// and a sapling there depends on the canopy another ~10 m beyond it, so a
/// chunk evaluates the trees 24 m past its own edge.
const TREE_CONTEXT: f64 = 24.0;

/// Generate the plants one detail level adds to a chunk. Level 0 holds the
/// trees; 1 the shrubs; 2 the herbs and lavender. Each level recomputes the
/// trees around the chunk to stay clear of them, which keeps every level a
/// pure function of the chunk coordinates.
/// How far past a river's or lake's waterline each layer's stems stand at
/// least: the margin its floods scour stays open turf, trees and shrubs keep
/// a few metres back, the bank's broadleaf plants begin a metre and a half
/// up, and lavender, which wants dry ground, keeps well back. The GPU cull
/// holds the drawn plants to the same margin.
const RIVER_CLEARANCE: [f32; 5] = [4.5, 4.0, 3.5, 1.5, 3.0];

/// How far past the waterline a canopy tree of crown radius `reach` stands
/// at least: back by most of its crown, so a broad tree leans over the water
/// rather than spreading across a creek or a lake's outlet. The GPU cull
/// holds the drawn trees to the same.
pub fn crown_clearance(reach: f32) -> f32 {
    0.75 * reach + 1.5
}

/// How far past the waterline the GPU cull lets a plant of this model at
/// this scale stand (vegetation-cull.wgsl's footing): a tree or tall shrub
/// back by a few metres, more for a tall one, and by most of its crown; a
/// ground plant by its crown and a metre. The scatter holds every plant to
/// it as well, so none it places for the others to make room for is hidden;
/// a tree or tall shrub with `FOOTING_MARGIN` to spare, for the GPU measures
/// a lake's shore from the eroded ground, which the scatter does not see,
/// and a lake's bank grows six times as fast as the ground rises.
fn river_footing(catalog: &Catalog, model: u32, scale: f32) -> f32 {
    let model = &catalog.models[model as usize];
    let reach = model.crown_radius * scale;
    if super::render_profile(model.species, &model.form).ground_habitat {
        (reach + 1.0).max(1.5)
    } else {
        (4.0 + 0.06 * model.height * scale).max(crown_clearance(reach)) + FOOTING_MARGIN
    }
}

/// Spare bank the scatter leaves a tree or tall shrub over the GPU cull's
/// footing, metres.
const FOOTING_MARGIN: f32 = 1.0;

pub fn generate_level(
    noise: &NoiseField,
    catalog: &Catalog,
    rivers: Option<&crate::rivers::network::RiverNetwork>,
    chunk: [i64; 2],
    level: u8,
) -> Vec<PlantInstance> {
    let (minimum, maximum) = chunk_bounds(chunk);
    let canopy_margin = (CANOPY.overlap * 2.0 * MAX_CANOPY_REACH) as f64 + 2.0;
    let (outer_min, outer_max) = grow(minimum, maximum, TREE_CONTEXT + canopy_margin + 8.0);
    let sampler = SiteSampler::new(noise, outer_min, outer_max).with_rivers(rivers);
    let dry = |x: f64, z: f64, layer: Layer, model: u32, scale: f32| {
        sampler.river_bank(x, z) >= RIVER_CLEARANCE[layer as usize].max(river_footing(catalog, model, scale))
    };
    let habitats = HabitatGrid::build(&sampler, outer_min, outer_max);

    // Canopy, accepted out to the tree context around the chunk.
    let (canopy_min, canopy_max) = grow(minimum, maximum, TREE_CONTEXT + canopy_margin);
    let canopy = CandidateGrid::build(&CANOPY, canopy_min, canopy_max, |x, z, key| {
        canopy_candidate(catalog, &habitats, x, z, key)
            .filter(|&(_, _, model, scale)| dry(x, z, Layer::Canopy, model, scale))
    });
    let (context_min, context_max) = grow(minimum, maximum, TREE_CONTEXT);
    let canopy_trees = canopy.survivors(CANOPY.overlap, MAX_CANOPY_REACH, context_min, context_max);
    let mut trees = Occupancy::new(8.0);
    for tree in &canopy_trees {
        trees.insert(tree.x, tree.z, tree.reach);
    }

    // Regeneration: kept clear of the canopy trunks and crowns' cores.
    let (regen_min, regen_max) = grow(context_min, context_max, 8.0);
    let assign_regeneration = |x: f64, z: f64, key: u64| -> Option<(f32, f32, u32, f32)> {
        let candidate = regeneration_candidate(catalog, &habitats, x, z, key)?;
        (!trees.blocks(x, z, candidate.1 * 0.6, 0.62) && dry(x, z, Layer::Regeneration, candidate.2, candidate.3))
            .then_some(candidate)
    };
    let regeneration = CandidateGrid::build(&REGENERATION, regen_min, regen_max, assign_regeneration);
    let young_trees = regeneration.survivors(REGENERATION.overlap, 2.5, context_min, context_max);
    for tree in &young_trees {
        trees.insert(tree.x, tree.z, tree.reach);
    }

    let inside = |c: &Candidate| c.x >= minimum[0] && c.x < maximum[0] && c.z >= minimum[1] && c.z < maximum[1];
    let mut out = Vec::new();
    match level {
        LEVEL_TREES => {
            out.extend(canopy_trees.iter().filter(|c| inside(c)).map(|c| c.instance(&sampler, Layer::Canopy)));
            out.extend(
                young_trees
                    .iter()
                    .filter(|c| inside(c))
                    .map(|c| c.instance(&sampler, Layer::Regeneration)),
            );
        }
        LEVEL_SHRUBS => {
            let (shrub_min, shrub_max) = grow(minimum, maximum, 6.0);
            // Shrubs grow under crown edges but not against a trunk.
            let assign_shrub = |x: f64, z: f64, key: u64| -> Option<(f32, f32, u32, f32)> {
                let candidate = shrub_candidate(catalog, &habitats, x, z, key)?;
                (!trees.blocks(x, z, candidate.1 * 0.45, 0.30) && dry(x, z, Layer::Shrub, candidate.2, candidate.3))
                    .then_some(candidate)
            };
            let shrubs = CandidateGrid::build(&SHRUB, shrub_min, shrub_max, assign_shrub);
            out.extend(
                shrubs
                    .survivors(SHRUB.overlap, 2.4, minimum, maximum)
                    .iter()
                    .map(|c| c.instance(&sampler, Layer::Shrub)),
            );
        }
        _ => {
            let (shrub_min, shrub_max) = grow(minimum, maximum, 10.0);
            let assign_shrub = |x: f64, z: f64, key: u64| -> Option<(f32, f32, u32, f32)> {
                let candidate = shrub_candidate(catalog, &habitats, x, z, key)?;
                (!trees.blocks(x, z, candidate.1 * 0.45, 0.30) && dry(x, z, Layer::Shrub, candidate.2, candidate.3))
                    .then_some(candidate)
            };
            let shrubs = CandidateGrid::build(&SHRUB, shrub_min, shrub_max, assign_shrub);
            let mut cover = Occupancy::new(4.0);
            let (near_min, near_max) = grow(minimum, maximum, 4.0);
            for shrub in shrubs.survivors(SHRUB.overlap, 2.4, near_min, near_max) {
                cover.insert(shrub.x, shrub.z, shrub.reach);
            }
            // Shore plants stand between the trunks and shrubs and lavender
            // needs open sky; both keep room between their own plants.
            let herb_margin = (HERB.overlap * 2.0 * MAX_BROADLEAF_REACH) as f64 + 0.5;
            let (herb_min, herb_max) = grow(minimum, maximum, herb_margin);
            let assign_herb = |x: f64, z: f64, key: u64| -> Option<(f32, f32, u32, f32)> {
                let candidate = herb_candidate(catalog, &habitats, x, z, key)?;
                (!trees.blocks(x, z, 0.5, 0.12)
                    && !cover.blocks(x, z, 0.1, 0.6)
                    && dry(x, z, Layer::Herb, candidate.2, candidate.3)
                    && sampler.river_bank(x, z) >= RIVER_CLEARANCE[Layer::Herb as usize] + candidate.1 * 0.5)
                    .then_some(candidate)
            };
            let herbs = CandidateGrid::build(&HERB, herb_min, herb_max, assign_herb);
            out.extend(
                herbs
                    .survivors(HERB.overlap, MAX_BROADLEAF_REACH, minimum, maximum)
                    .iter()
                    .map(|c| c.instance(&sampler, Layer::Herb)),
            );
            let tuft_margin = (LAVENDER.overlap * 2.0 * MAX_LAVENDER_REACH) as f64 + 0.5;
            let (tuft_min, tuft_max) = grow(minimum, maximum, tuft_margin);
            let assign_lavender = |x: f64, z: f64, key: u64| -> Option<(f32, f32, u32, f32)> {
                let candidate = lavender_candidate(catalog, &habitats, x, z, key)?;
                (!trees.blocks(x, z, 0.4, 0.85) && !cover.blocks(x, z, 0.2, 0.9) && dry(x, z, Layer::Lavender, candidate.2, candidate.3))
                    .then_some(candidate)
            };
            let lavender = CandidateGrid::build(&LAVENDER, tuft_min, tuft_max, assign_lavender);
            out.extend(
                lavender
                    .survivors(LAVENDER.overlap, MAX_LAVENDER_REACH, minimum, maximum)
                    .iter()
                    .map(|c| c.instance(&sampler, Layer::Lavender)),
            );
        }
    }
    out
}
