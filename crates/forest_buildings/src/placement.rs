//! Where the buildings stand.
//!
//! Settlers took good, open, well-drained land near water, and so does the
//! placement here: every choice is a pure function of world position, made
//! from the same uneroded, river-carved ground the erosion starts from, so a
//! building is where it is however the world streams in.
//!
//! - **Homesteads.** Each 480 m cell may hold one house, more often where a
//!   broad settlement field runs high and near water. Up to six sites in the
//!   cell's middle are tried and the best kept: level over the house and its
//!   yard, not a terrace on a mountainside (the ground stays within a few
//!   metres of the floor 35 and 90 m around), well above the sea and away from
//!   any beach, never in or beside a river or a lake, and above the reach of
//!   their floods. A house near water faces it; one away from it faces down
//!   the land's fall.
//! - **Towns.** Each 3.2 km cell may hold a town where a broad flat stretch
//!   lies within reach of a river or a lake. A main street runs along the
//!   contour through its middle with a row of buildings on either side, their
//!   fronts on the street, the sheriff's office among them. Homesteads keep
//!   their distance from a town.
//!
//! A building's pad is levelled at the mean height of the ground under it,
//! and its skirt is as wide as the cut or fill at its edge needs to stay
//! gentle.

use super::assets::BuildingLibrary;
use super::collision::BuildingFrame;
use super::pads::{Pad, PadKind};
use bevy::math::{Vec2, Vec3};
use forest_core::constants::SEA_LEVEL;
use forest_core::noise::{NoiseField, base_height};
use forest_water::rivers::network::RiverNetwork;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, Weak};

/// Side of a homestead cell, metres: at most one house in each.
pub const HOMESTEAD_CELL: f32 = 480.0;
/// Side of a town cell, metres.
pub const TOWN_CELL: f32 = 3200.0;
/// Homesteads keep this far from a town's centre.
const TOWN_CLEARANCE: f32 = 420.0;
/// How far a town's buildings reach from its centre, at most.
pub const TOWN_REACH: f32 = 160.0;
/// The share of the most settled land's homestead cells that hold a house.
const HOMESTEAD_CHANCE: f32 = 0.6;
/// The share of town cells with a town site that hold a town.
const TOWN_CHANCE: f32 = 0.7;
/// Room levelled around a building's footprint for its yard.
pub const PAD_MARGIN: f32 = 2.0;
/// How far the foundation sinks into its pad, so it stands in the ground.
pub const FOUNDATION_SINK: f32 = 0.05;
/// Street half width plus the strip before a building's front.
const STREET_SETBACK: f32 = 8.0;
/// Spacing of the samples over a footprint.
const FOOTPRINT_SPACING: f32 = 2.5;

/// What a site must be.
#[derive(Clone, Copy, Debug)]
struct SiteRules {
    /// Height of the pad above the sea, metres.
    lowest: f32,
    highest: f32,
    /// Most the ground may vary under the pad, and its steepest fitted slope.
    spread: f32,
    slope: f32,
    /// Most the ground 35 m and 90 m around may stand above or below the pad.
    near_relief: f32,
    far_relief: f32,
    /// Nearest a river's or lake's water may come, and the least height the
    /// pad must stand above the water nearest it.
    water_clearance: f32,
    flood_freeboard: f32,
}

const HOMESTEAD_RULES: SiteRules = SiteRules {
    lowest: 4.0,
    highest: 75.0,
    spread: 1.4,
    slope: 0.075,
    near_relief: 5.0,
    far_relief: 11.0,
    water_clearance: 35.0,
    flood_freeboard: 2.5,
};

/// A town's buildings stand on ground already found flat; each still needs
/// its own lot to be sound.
const LOT_RULES: SiteRules = SiteRules {
    lowest: 3.5,
    highest: 70.0,
    spread: 1.8,
    slope: 0.08,
    near_relief: 6.0,
    far_relief: 16.0,
    water_clearance: 30.0,
    flood_freeboard: 2.0,
};

/// What placement needs of a model.
#[derive(Clone, Debug)]
pub struct Footprint {
    pub name: String,
    /// Centre and half extents of the footprint, building space XZ.
    pub centre: Vec2,

    pub half: Vec2,
    /// The way the front door faces, building space XZ.
    pub front: Vec2,

    /// Relative chance of standing alone as a homestead, and in a town.
    pub homestead_weight: f32,

    pub town_weight: f32,
}

impl Footprint {
    pub fn from_library(library: &BuildingLibrary) -> Vec<Footprint> {
        library
            .models
            .iter()
            .map(|model| {
                // Small, plain houses are the common ones; the big two-storey
                // places with granny flats are rarer, and the sheriff's office
                // belongs in a town.
                let (homestead_weight, town_weight) = match model.name.as_str() {
                    "house-a" | "house-b" => (3.0, 3.0),
                    "house-e" => (2.0, 1.5),
                    "house-c" | "house-d" => (1.6, 2.0),
                    "house-f" | "house-g" => (0.9, 0.6),
                    "sheriff-office" => (0.0, 0.0),
                    _ => (1.0, 1.0),
                };
                Footprint {
                    name: model.name.clone(),
                    centre: model.footprint_centre(),
                    half: model.footprint_half(),
                    front: model.front,
                    homestead_weight,
                    town_weight,
                }
            })
            .collect()
    }

    /// Half the footprint along the street and away from it, for a building
    /// whose front faces the street.
    fn frontage(&self) -> (f32, f32) {
        if self.front.x.abs() > 0.5 {
            (self.half.y, self.half.x)
        } else {
            (self.half.x, self.half.y)
        }
    }
}

/// One placed building.
#[derive(Clone, Debug, PartialEq)]
pub struct Placement {
    /// Stable across streaming: from the cell and the slot it was placed in.
    pub id: u64,

    pub model: u32,
    /// The building's origin, world space: its foundation's foot.
    pub origin: Vec3,

    pub yaw: f32,
    pub pad: Pad,
    /// The town it belongs to, if any.
    pub town: Option<u64>,
}

impl Placement {
    pub fn frame(&self) -> BuildingFrame {
        BuildingFrame::new(self.origin, self.yaw)
    }
}

/// A town's centre, its buildings and its main street.
#[derive(Clone, Debug)]
pub struct Town {
    pub id: u64,
    pub centre: Vec2,
    pub placements: Vec<Placement>,
    /// The dirt road between its rows.
    pub street: Pad,
}

// ---------------------------------------------------------------------------
// Randomness and fields
// ---------------------------------------------------------------------------

fn mix_bits(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn cell_key(cell: [i64; 2], salt: u64) -> u64 {
    let x = mix_bits(cell[0] as u64 ^ 0x9e37_79b9_7f4a_7c15);
    let z = mix_bits(cell[1] as u64 ^ 0xc2b2_ae3d_27d4_eb4f ^ x);
    mix_bits(z ^ salt.wrapping_mul(0x1656_67b1_9e37_79f9))
}

/// A number in [0, 1) for a key and a draw.
fn random(key: u64, draw: u64) -> f32 {
    let bits = mix_bits(key ^ draw.wrapping_mul(0x2545_f491_4f6c_dd1d));
    (bits >> 40) as f32 / (1u64 << 24) as f32
}

/// How settled the land is, 0 to 1, over a few kilometres: value noise on a
/// 2.6 km lattice, smoothly interpolated, so settled districts and empty
/// ones alternate.
fn settlement_field(point: Vec2) -> f32 {
    let scaled = point / 2600.0;
    let cell = scaled.floor();
    let fraction = scaled - cell;
    let smooth = fraction * fraction * (Vec2::splat(3.0) - 2.0 * fraction);
    let corner = |dx: i64, dz: i64| {
        random(
            cell_key([cell.x as i64 + dx, cell.y as i64 + dz], 0x5e77),
            0,
        )
    };
    let bottom = corner(0, 0) + (corner(1, 0) - corner(0, 0)) * smooth.x;
    let top = corner(0, 1) + (corner(1, 1) - corner(0, 1)) * smooth.x;
    let value = bottom + (top - bottom) * smooth.y;
    (value * 1.4 - 0.2).clamp(0.25, 1.0)
}

fn pick_weighted(
    footprints: &[Footprint],
    weight: impl Fn(&Footprint) -> f32,
    roll: f32,
) -> Option<usize> {
    let total: f32 = footprints.iter().map(&weight).sum();
    if total <= 0.0 {
        return None;
    }
    let mut remaining = roll * total;
    for (index, footprint) in footprints.iter().enumerate() {
        let share = weight(footprint);
        if share <= 0.0 {
            continue;
        }
        if remaining < share {
            return Some(index);
        }
        remaining -= share;
    }
    footprints
        .iter()
        .rposition(|footprint| weight(footprint) > 0.0)
}

/// The turn about +Y that makes a building whose front faces `front`
/// (building space) face `facing` (world).
pub fn yaw_facing(front: Vec2, facing: Vec2) -> f32 {
    front.y.atan2(front.x) - facing.y.atan2(facing.x)
}

// ---------------------------------------------------------------------------
// The ground and the water
// ---------------------------------------------------------------------------

/// Where water stands: points along every river's centreline and around
/// every lake's shore, by 64 m cell.
pub struct WaterIndex {
    cells: HashMap<[i32; 2], Vec<WaterPoint>>,
}

#[derive(Clone, Copy, Debug)]
struct WaterPoint {
    position: Vec2,
    /// How far from the point the water reaches.
    reach: f32,
    level: f32,
}

const WATER_CELL: f32 = 64.0;

fn water_cell(point: Vec2) -> [i32; 2] {
    [
        (point.x / WATER_CELL).floor() as i32,
        (point.y / WATER_CELL).floor() as i32,
    ]
}

impl WaterIndex {
    pub fn build(network: &RiverNetwork) -> Self {
        let mut cells: HashMap<[i32; 2], Vec<WaterPoint>> = HashMap::new();
        let mut insert = |point: WaterPoint| {
            cells
                .entry(water_cell(point.position))
                .or_default()
                .push(point)
        };
        for river in &network.rivers {
            for node in &river.nodes {
                insert(WaterPoint {
                    position: Vec2::from(node.position),
                    reach: node.half_width,
                    level: node.water,
                });
            }
        }
        let flow_cell = forest_water::rivers::network::FLOW_CELL as f32;
        for lake in &network.lakes {
            // One point per 8 m block of the shore is plenty to measure from.
            let mut seen: std::collections::HashSet<[i32; 2]> = std::collections::HashSet::new();
            for cell in lake.shore.iter().chain(&lake.cells) {
                if !seen.insert([cell[0].div_euclid(2), cell[1].div_euclid(2)]) {
                    continue;
                }
                let position = Vec2::new(
                    (cell[0] as f32 + 0.5) * flow_cell,
                    (cell[1] as f32 + 0.5) * flow_cell,
                );
                insert(WaterPoint {
                    position,
                    reach: flow_cell,
                    level: lake.level,
                });
            }
        }
        Self { cells }
    }

    /// The nearest water within `radius`: its distance from the water's edge
    /// and the level of that water.
    fn nearest(&self, point: Vec2, radius: f32) -> Option<(f32, f32, Vec2)> {
        let first = water_cell(point - Vec2::splat(radius));
        let last = water_cell(point + Vec2::splat(radius));
        let mut best: Option<(f32, f32, Vec2)> = None;
        for cz in first[1]..=last[1] {
            for cx in first[0]..=last[0] {
                let Some(points) = self.cells.get(&[cx, cz]) else {
                    continue;
                };
                for water in points {
                    let distance = (water.position.distance(point) - water.reach).max(0.0);
                    if distance <= radius && best.is_none_or(|known| distance < known.0) {
                        best = Some((distance, water.level, water.position));
                    }
                }
            }
        }
        best
    }
}

/// The ground placement reads: the base landform with the rivers carved.
struct Ground<'a> {
    noise: &'a NoiseField,
    rivers: Option<&'a RiverNetwork>,
    water: Option<&'a WaterIndex>,
}

impl Ground<'_> {
    fn height(&self, point: Vec2) -> f32 {
        let base = base_height(self.noise, point.x, point.y);
        match self.rivers {
            Some(network) => network.envelope(point.x, point.y).clamp(base),
            None => base,
        }
    }

    /// Whether a river's channel or a lake's water reaches within `clearance`
    /// of the point, by the carve envelope (exact near the channels).
    fn wet(&self, point: Vec2, height: f32, clearance: f32) -> bool {
        let Some(network) = self.rivers else {
            return false;
        };
        let envelope = network.envelope(point.x, point.y);
        envelope.bank_distance < clearance || height < envelope.lake + 1.5
    }

    /// The direction the land falls over tens of metres, world XZ.
    fn downhill(&self, point: Vec2) -> Vec2 {
        let step = 40.0;
        let dx =
            self.height(point + Vec2::new(step, 0.0)) - self.height(point - Vec2::new(step, 0.0));
        let dz =
            self.height(point + Vec2::new(0.0, step)) - self.height(point - Vec2::new(0.0, step));
        (-Vec2::new(dx, dz)).normalize_or_zero()
    }
}

/// What a sound site gives its building.
#[derive(Clone, Copy, Debug)]
struct Site {
    level: f32,
    skirt: f32,
    /// How good it is, higher better.
    score: f32,
}

/// Whether a pad of half extents `half` (its yard included) centred on
/// `centre` and turned by `axis` is a sound site, and how good.
fn examine_site(
    ground: &Ground<'_>,
    centre: Vec2,
    axis: Vec2,
    half: Vec2,
    rules: &SiteRules,
) -> Option<Site> {
    let middle = ground.height(centre);
    if middle < SEA_LEVEL + rules.lowest || middle > SEA_LEVEL + rules.highest {
        return None;
    }
    // Samples over the pad, in its own axes.
    let columns = (2.0 * half.x / FOOTPRINT_SPACING).ceil().max(2.0) as usize;
    let rows = (2.0 * half.y / FOOTPRINT_SPACING).ceil().max(2.0) as usize;
    let to_world = |local: Vec2| {
        centre
            + Vec2::new(
                local.x * axis.x + local.y * axis.y,
                -local.x * axis.y + local.y * axis.x,
            )
    };
    let mut samples: Vec<(Vec2, f32)> = Vec::with_capacity((columns + 1) * (rows + 1));
    for row in 0..=rows {
        for column in 0..=columns {
            let local = Vec2::new(
                -half.x + 2.0 * half.x * column as f32 / columns as f32,
                -half.y + 2.0 * half.y * row as f32 / rows as f32,
            );
            samples.push((local, ground.height(to_world(local))));
        }
    }
    let count = samples.len() as f32;
    let level = samples.iter().map(|sample| sample.1).sum::<f32>() / count;
    let lowest = samples
        .iter()
        .map(|sample| sample.1)
        .fold(f32::INFINITY, f32::min);
    let highest = samples
        .iter()
        .map(|sample| sample.1)
        .fold(f32::NEG_INFINITY, f32::max);
    if highest - lowest > rules.spread {
        return None;
    }
    // The plane through the samples, by least squares on the symmetric grid.
    let across: f32 = samples
        .iter()
        .map(|(local, height)| local.x * (height - level))
        .sum();
    let along: f32 = samples
        .iter()
        .map(|(local, height)| local.y * (height - level))
        .sum();
    let across_weight: f32 = samples
        .iter()
        .map(|(local, _)| local.x * local.x)
        .sum::<f32>()
        .max(1e-3);
    let along_weight: f32 = samples
        .iter()
        .map(|(local, _)| local.y * local.y)
        .sum::<f32>()
        .max(1e-3);
    let slope = Vec2::new(across / across_weight, along / along_weight).length();
    if slope > rules.slope {
        return None;
    }
    // Not in or beside water anywhere over it.
    let corners = [
        Vec2::ZERO,
        half,
        -half,
        Vec2::new(half.x, -half.y),
        Vec2::new(-half.x, half.y),
    ];
    for corner in corners {
        let point = to_world(corner);
        if ground.wet(point, level, rules.water_clearance * 0.5) {
            return None;
        }
    }
    // Not a terrace cut into a mountainside or a knoll by the sea: the land
    // around keeps near the pad's level and well above the water.
    let mut relief_near = 0.0f32;
    let mut relief_far = 0.0f32;
    let mut low_ground = f32::INFINITY;
    for (radius, count) in [(35.0f32, 12usize), (90.0, 16)] {
        for index in 0..count {
            let angle = std::f32::consts::TAU * (index as f32 + 0.5) / count as f32;
            let point = centre + Vec2::new(angle.cos(), angle.sin()) * radius;
            let height = ground.height(point);
            low_ground = low_ground.min(height);
            let relief = (height - level).abs();
            if radius < 50.0 {
                relief_near = relief_near.max(relief);
            } else {
                relief_far = relief_far.max(relief);
            }
        }
    }
    if relief_near > rules.near_relief || relief_far > rules.far_relief {
        return None;
    }
    if low_ground < SEA_LEVEL + 1.5 {
        return None;
    }
    // Clear of the water, and above its floods.
    let mut water_bonus = 0.0;
    if let Some(water) = ground.water
        && let Some((distance, water_level, _)) = water.nearest(centre, 450.0)
    {
        if distance < rules.water_clearance {
            return None;
        }
        if distance < 160.0 && level < water_level + rules.flood_freeboard {
            return None;
        }
        // Within a short walk of water is where people settled.
        water_bonus = 1.0 - ((distance - 60.0) / 390.0).clamp(0.0, 1.0);
    }
    // The skirt: wide enough for the cut or fill at the pad's edge to stay a
    // gentle slope.
    let edge_relief = samples
        .iter()
        .filter(|(local, _)| local.x.abs() >= half.x - 1e-3 || local.y.abs() >= half.y - 1e-3)
        .map(|(_, height)| (height - level).abs())
        .fold(0.0f32, f32::max);
    let skirt = (4.0 + 5.0 * edge_relief).clamp(5.0, 12.0);
    let score = water_bonus * 2.0 - slope * 12.0 - (highest - lowest) - relief_near * 0.15;
    Some(Site {
        level,
        skirt,
        score,
    })
}

fn placement_for(
    footprint: &Footprint,
    model: u32,
    centre: Vec2,
    yaw: f32,
    site: &Site,
    id: u64,
    town: Option<u64>,
) -> Placement {
    // The origin is where the footprint's centre lands at `centre`.
    let (sine, cosine) = yaw.sin_cos();
    let offset = Vec2::new(
        footprint.centre.x * cosine + footprint.centre.y * sine,
        -footprint.centre.x * sine + footprint.centre.y * cosine,
    );
    let origin_xz = centre - offset;
    let pad = Pad {
        centre,
        axis: Vec2::new(cosine, sine),
        half: footprint.half + Vec2::splat(PAD_MARGIN),
        height: site.level,
        skirt: site.skirt,
        inset: PAD_MARGIN,
        kind: PadKind::Building,
    };
    Placement {
        id,
        model,
        origin: Vec3::new(origin_xz.x, site.level - FOUNDATION_SINK, origin_xz.y),
        yaw,
        pad,
        town,
    }
}

// ---------------------------------------------------------------------------
// Towns
// ---------------------------------------------------------------------------

/// The best town site in a cell, if the cell holds a town.
fn plan_town(ground: &Ground<'_>, footprints: &[Footprint], cell: [i64; 2]) -> Option<Town> {
    let key = cell_key(cell, 0x70_77);
    if random(key, 0) >= TOWN_CHANCE {
        return None;
    }
    let minimum = Vec2::new(cell[0] as f32, cell[1] as f32) * TOWN_CELL;
    // The broad flat: a 150 m square sampled every 18.75 m.
    let flat = |centre: Vec2| -> Option<(f32, Vec2, f32)> {
        let middle = ground.height(centre);
        if !(SEA_LEVEL + 4.0..=SEA_LEVEL + 60.0).contains(&middle) {
            return None;
        }
        let mut samples = Vec::with_capacity(81);
        for row in 0..9 {
            for column in 0..9 {
                let local = Vec2::new(column as f32 - 4.0, row as f32 - 4.0) * 18.75;
                let point = centre + local;
                let height = ground.height(point);
                if height < SEA_LEVEL + 3.0 || ground.wet(point, height, 12.0) {
                    return None;
                }
                samples.push((local, height));
            }
        }
        let level = samples.iter().map(|sample| sample.1).sum::<f32>() / samples.len() as f32;
        let weight: f32 = samples.iter().map(|(local, _)| local.x * local.x).sum();
        let gradient = Vec2::new(
            samples
                .iter()
                .map(|(local, height)| local.x * (height - level))
                .sum::<f32>()
                / weight,
            samples
                .iter()
                .map(|(local, height)| local.y * (height - level))
                .sum::<f32>()
                / weight,
        );
        let residual = samples
            .iter()
            .map(|(local, height)| (height - level - gradient.dot(*local)).abs())
            .fold(0.0f32, f32::max);
        (gradient.length() < 0.05 && residual < 2.5).then_some((level, gradient, residual))
    };
    let mut best: Option<(f32, Vec2, Vec2)> = None;
    for index in 0..25u64 {
        let jitter = Vec2::new(random(key, 10 + index * 2), random(key, 11 + index * 2));
        let lattice = Vec2::new((index % 5) as f32, (index / 5) as f32);
        let centre = minimum + (Vec2::splat(0.2) + (lattice + jitter) * 0.12) * TOWN_CELL;
        let Some((_, gradient, residual)) = flat(centre) else {
            continue;
        };
        // A town grows where there is water to hand.
        let water = ground.water.and_then(|water| water.nearest(centre, 700.0));
        let Some((distance, _, water_position)) = water else {
            continue;
        };
        if distance < 90.0 {
            continue;
        }
        let score = -gradient.length() * 40.0 - residual - distance / 250.0;
        if best.is_none_or(|known| score > known.0) {
            best = Some((score, centre, water_position));
        }
    }
    let (_, centre, water_position) = best?;
    let id = key;
    // The main street runs along the contour, so its rows stand level; on
    // land too flat to have one, along the water.
    let fall = ground.downhill(centre);
    let towards_water = (water_position - centre).normalize_or_zero();
    let street = if fall != Vec2::ZERO {
        fall.perp()
    } else {
        towards_water.perp()
    };
    let street = if street == Vec2::ZERO {
        Vec2::X
    } else {
        street.normalize()
    };
    let across = street.perp();
    let per_side = 3 + (random(key, 3) * 2.99) as usize;
    let sheriff = footprints
        .iter()
        .position(|footprint| footprint.name == "sheriff-office");
    let mut placements = Vec::new();
    let mut street_length = 0.0f32;
    for (side_index, side) in [1.0f32, -1.0].into_iter().enumerate() {
        // Lay the row out first so it can be centred on the town.
        let mut lots: Vec<(usize, f32)> = Vec::new();
        let mut length = 0.0;
        for lot in 0..per_side {
            let lot_key = cell_key(
                [cell[0] * 8 + side_index as i64, cell[1] * 8 + lot as i64],
                key,
            );
            let model = match sheriff {
                Some(sheriff) if side_index == 0 && lot == per_side / 2 => sheriff,
                _ => pick_weighted(
                    footprints,
                    |footprint| footprint.town_weight,
                    random(lot_key, 1),
                )?,
            };
            let gap = 4.0 + random(lot_key, 2) * 5.0;
            let (half_width, _) = footprints[model].frontage();
            lots.push((model, gap));
            length += 2.0 * half_width + gap;
        }
        street_length = street_length.max(length);
        let mut cursor = -0.5 * length;
        for (lot, (model, gap)) in lots.into_iter().enumerate() {
            let footprint = &footprints[model];
            let (half_width, half_depth) = footprint.frontage();
            let along = cursor + half_width + 0.5 * gap;
            cursor += 2.0 * half_width + gap;
            let facing = -across * side;
            let lot_centre = centre
                + street * along
                + across * side * (STREET_SETBACK + half_depth + PAD_MARGIN);
            let yaw = yaw_facing(footprint.front, facing);
            let axis = Vec2::new(yaw.cos(), yaw.sin());
            let pad_half = footprint.half + Vec2::splat(PAD_MARGIN);
            let Some(site) = examine_site(ground, lot_centre, axis, pad_half, &LOT_RULES) else {
                continue;
            };
            let lot_id = cell_key(
                [cell[0], cell[1]],
                key ^ (side_index as u64 * 64 + lot as u64 + 1),
            );
            placements.push(placement_for(
                footprint,
                model as u32,
                lot_centre,
                yaw,
                &site,
                lot_id,
                Some(id),
            ));
        }
    }
    // A town is a handful of buildings and its jail; less is a hamlet that
    // never grew, and is not built.
    let has_sheriff = placements
        .iter()
        .any(|placement| Some(placement.model as usize) == sheriff);
    if placements.len() < 4 || !has_sheriff {
        return None;
    }
    // The main street, worn to a dirt road from end to end of the rows and
    // from porch to porch.
    let yaw = yaw_facing(Vec2::X, street);
    let street = Pad {
        centre,
        axis: Vec2::new(yaw.cos(), yaw.sin()),
        half: Vec2::new(0.5 * street_length + 6.0, STREET_SETBACK - 1.5),
        height: ground.height(centre),
        skirt: 2.5,
        inset: 0.0,
        kind: PadKind::Street,
    };
    Some(Town {
        id,
        centre,
        placements,
        street,
    })
}

// ---------------------------------------------------------------------------
// Homesteads
// ---------------------------------------------------------------------------

fn plan_homestead(
    ground: &Ground<'_>,
    footprints: &[Footprint],
    cell: [i64; 2],
    towns: &[Arc<Option<Town>>],
) -> Option<Placement> {
    let key = cell_key(cell, 0x40_3e);
    let minimum = Vec2::new(cell[0] as f32, cell[1] as f32) * HOMESTEAD_CELL;
    let middle = minimum + Vec2::splat(0.5 * HOMESTEAD_CELL);
    if random(key, 0) >= HOMESTEAD_CHANCE * settlement_field(middle) {
        return None;
    }
    let model = pick_weighted(
        footprints,
        |footprint| footprint.homestead_weight,
        random(key, 1),
    )?;
    let footprint = &footprints[model];
    let pad_half = footprint.half + Vec2::splat(PAD_MARGIN);
    let mut best: Option<(f32, Placement)> = None;
    for attempt in 0..6u64 {
        let jitter = Vec2::new(random(key, 10 + attempt * 2), random(key, 11 + attempt * 2));
        let centre = minimum + (Vec2::splat(0.15) + jitter * 0.7) * HOMESTEAD_CELL;
        let near_town = towns
            .iter()
            .filter_map(|town| town.as_ref().as_ref())
            .any(|town| town.centre.distance(centre) < TOWN_CLEARANCE);
        if near_town {
            continue;
        }
        // Face the water within sight, else down the land's fall, a few
        // degrees either way.
        let water = ground.water.and_then(|water| water.nearest(centre, 350.0));
        let facing = match water {
            Some((_, _, position)) => (position - centre).normalize_or_zero(),
            None => ground.downhill(centre),
        };
        let facing = if facing == Vec2::ZERO {
            Vec2::Y
        } else {
            facing
        };
        let wobble = (random(key, 30 + attempt) - 0.5) * 0.4;
        let yaw = yaw_facing(footprint.front, facing) + wobble;
        let axis = Vec2::new(yaw.cos(), yaw.sin());
        let Some(site) = examine_site(ground, centre, axis, pad_half, &HOMESTEAD_RULES) else {
            continue;
        };
        if best.as_ref().is_none_or(|known| site.score > known.0) {
            let placement = placement_for(footprint, model as u32, centre, yaw, &site, key, None);
            best = Some((site.score, placement));
        }
    }
    best.map(|(_, placement)| placement)
}

// ---------------------------------------------------------------------------
// The planner
// ---------------------------------------------------------------------------

/// Plans and remembers the settlements: towns by town cell, homesteads by
/// homestead cell. Shared by the building stream and the plant scatter,
/// which keeps its trees out of the buildings' yards.
pub struct SettlementPlanner {
    noise: Arc<NoiseField>,
    footprints: Vec<Footprint>,
    snapshots: Mutex<HashMap<Option<usize>, Arc<SettlementCache>>>,
}

/// Every cached decision shares one immutable river snapshot. A weak reference
/// keeps its address unique without retaining an obsolete river network.
struct SettlementCache {
    network: Option<Weak<RiverNetwork>>,
    water: Option<WaterIndex>,
    towns: Mutex<HashMap<[i64; 2], Arc<Option<Town>>>>,
    homesteads: Mutex<HashMap<[i64; 2], Arc<Option<Placement>>>>,
}

fn cell_range(minimum: Vec2, maximum: Vec2, size: f32) -> ([i64; 2], [i64; 2]) {
    let first = [
        (minimum.x / size).floor() as i64,
        (minimum.y / size).floor() as i64,
    ];
    let last = [
        (maximum.x / size).floor() as i64,
        (maximum.y / size).floor() as i64,
    ];
    (first, last)
}

impl SettlementPlanner {
    pub fn new(noise: Arc<NoiseField>, footprints: Vec<Footprint>) -> Self {
        Self {
            noise,
            footprints,
            snapshots: Mutex::new(HashMap::new()),
        }
    }

    pub fn footprints(&self) -> &[Footprint] {
        &self.footprints
    }

    fn cache_for(&self, rivers: Option<&Arc<RiverNetwork>>) -> Arc<SettlementCache> {
        let identity = rivers.map(|network| Arc::as_ptr(network) as usize);
        {
            let mut snapshots = self
                .snapshots
                .lock()
                .expect("settlement snapshots lock poisoned");
            snapshots.retain(|_, cache| {
                cache
                    .network
                    .as_ref()
                    .is_none_or(|network| network.strong_count() > 0)
            });
            if let Some(cache) = snapshots.get(&identity) {
                return Arc::clone(cache);
            }
        }
        // Build the index outside the registry lock; other snapshot queries
        // may still be planning towns or clearings concurrently.
        let cache = Arc::new(SettlementCache {
            network: rivers.map(Arc::downgrade),
            water: rivers.map(|network| WaterIndex::build(network)),
            towns: Mutex::new(HashMap::new()),
            homesteads: Mutex::new(HashMap::new()),
        });
        let mut snapshots = self
            .snapshots
            .lock()
            .expect("settlement snapshots lock poisoned");
        Arc::clone(snapshots.entry(identity).or_insert(cache))
    }

    fn town(
        &self,
        cache: &SettlementCache,
        ground: &Ground<'_>,
        cell: [i64; 2],
    ) -> Arc<Option<Town>> {
        if let Some(town) = cache
            .towns
            .lock()
            .expect("settlement cache lock poisoned")
            .get(&cell)
        {
            return town.clone();
        }
        let town = Arc::new(plan_town(ground, &self.footprints, cell));
        cache
            .towns
            .lock()
            .expect("settlement cache lock poisoned")
            .entry(cell)
            .or_insert(town)
            .clone()
    }

    fn homestead(
        &self,
        cache: &SettlementCache,
        ground: &Ground<'_>,
        cell: [i64; 2],
    ) -> Arc<Option<Placement>> {
        if let Some(homestead) = cache
            .homesteads
            .lock()
            .expect("settlement cache lock poisoned")
            .get(&cell)
        {
            return homestead.clone();
        }
        // The towns whose clearance could reach into the cell.
        let minimum = Vec2::new(cell[0] as f32, cell[1] as f32) * HOMESTEAD_CELL
            - Vec2::splat(TOWN_CLEARANCE);
        let maximum = minimum + Vec2::splat(HOMESTEAD_CELL + 2.0 * TOWN_CLEARANCE);
        let (first, last) = cell_range(minimum, maximum, TOWN_CELL);
        let mut towns = Vec::new();
        for cz in first[1]..=last[1] {
            for cx in first[0]..=last[0] {
                towns.push(self.town(cache, ground, [cx, cz]));
            }
        }
        let homestead = Arc::new(plan_homestead(ground, &self.footprints, cell, &towns));
        cache
            .homesteads
            .lock()
            .expect("settlement cache lock poisoned")
            .entry(cell)
            .or_insert(homestead)
            .clone()
    }

    /// Every building whose footprint's centre lies within the rectangle,
    /// homesteads and towns alike, in a stable order.
    pub fn placements_in(
        &self,
        rivers: Option<&Arc<RiverNetwork>>,
        minimum: Vec2,
        maximum: Vec2,
    ) -> Vec<Placement> {
        let cache = self.cache_for(rivers);
        let ground = Ground {
            noise: &self.noise,
            rivers: rivers.map(|network| network.as_ref()),
            water: cache.water.as_ref(),
        };
        let inside = |placement: &Placement| {
            let centre = placement.pad.centre;
            centre.x >= minimum.x
                && centre.y >= minimum.y
                && centre.x < maximum.x
                && centre.y < maximum.y
        };
        let mut found = Vec::new();
        let (first, last) = cell_range(
            minimum - Vec2::splat(TOWN_REACH),
            maximum + Vec2::splat(TOWN_REACH),
            TOWN_CELL,
        );
        for cz in first[1]..=last[1] {
            for cx in first[0]..=last[0] {
                if let Some(town) = self.town(&cache, &ground, [cx, cz]).as_ref() {
                    found.extend(
                        town.placements
                            .iter()
                            .filter(|placement| inside(placement))
                            .cloned(),
                    );
                }
            }
        }
        let (first, last) = cell_range(minimum, maximum, HOMESTEAD_CELL);
        for cz in first[1]..=last[1] {
            for cx in first[0]..=last[0] {
                let homestead = self.homestead(&cache, &ground, [cx, cz]);
                if let Some(placement) = homestead.as_ref()
                    && inside(placement)
                {
                    found.push(placement.clone());
                }
            }
        }
        found.sort_by_key(|placement| placement.id);
        found
    }

    /// The main streets of the towns whose centres lie within the rectangle,
    /// grown by the reach of a town's buildings.
    pub fn streets_in(
        &self,
        rivers: Option<&Arc<RiverNetwork>>,
        minimum: Vec2,
        maximum: Vec2,
    ) -> Vec<Pad> {
        let reach = Vec2::splat(TOWN_REACH);
        let towns = self.towns_in(rivers, minimum - reach, maximum + reach);
        let inside = |centre: Vec2| {
            centre.cmpge(minimum - reach).all() && centre.cmplt(maximum + reach).all()
        };
        let mut streets: Vec<(u64, Pad)> = towns
            .iter()
            .filter(|town| inside(town.centre))
            .map(|town| (town.id, town.street))
            .collect();
        streets.sort_by_key(|(id, _)| *id);
        streets.into_iter().map(|(_, street)| street).collect()
    }

    /// The towns whose cells overlap a rectangle, for maps and the menu.
    pub fn towns_in(
        &self,
        rivers: Option<&Arc<RiverNetwork>>,
        minimum: Vec2,
        maximum: Vec2,
    ) -> Vec<Town> {
        let cache = self.cache_for(rivers);
        let ground = Ground {
            noise: &self.noise,
            rivers: rivers.map(|network| network.as_ref()),
            water: cache.water.as_ref(),
        };
        let (first, last) = cell_range(minimum, maximum, TOWN_CELL);
        let mut towns = Vec::new();
        for cz in first[1]..=last[1] {
            for cx in first[0]..=last[0] {
                if let Some(town) = self.town(&cache, &ground, [cx, cz]).as_ref() {
                    towns.push(town.clone());
                }
            }
        }
        towns
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assets::test_support::library;

    fn planner() -> (SettlementPlanner, Arc<RiverNetwork>, Arc<NoiseField>) {
        let (noise, network) = forest_water::rivers::spawn_network();
        let library = library();
        let planner = SettlementPlanner::new(noise.clone(), Footprint::from_library(&library));
        (planner, network, noise)
    }

    #[test]
    fn settlement_queries_keep_each_river_snapshot_independent() {
        let (planner, network, noise) = planner();
        let minimum = Vec2::splat(-5000.0);
        let maximum = Vec2::splat(5000.0);
        let dry = planner.placements_in(None, minimum, maximum);
        let expected_planner = SettlementPlanner::new(noise, planner.footprints().to_vec());
        let expected = expected_planner.placements_in(Some(&network), minimum, maximum);
        assert_ne!(
            dry, expected,
            "the river constraints must affect the fixture"
        );

        std::thread::scope(|scope| {
            let wet_query = scope.spawn(|| planner.placements_in(Some(&network), minimum, maximum));
            let dry_again = planner.placements_in(None, minimum, maximum);
            let wet = wet_query.join().expect("settlement query worker panicked");
            assert_eq!(wet, expected, "dry cached cells must not hide the rivers");
            assert_eq!(
                dry_again, dry,
                "an old snapshot must retain its own decisions"
            );
        });
    }

    #[test]
    fn buildings_stand_on_sound_ground_and_the_same_wherever_asked_from() {
        let (planner, network, noise) = planner();
        let reach = 5000.0;
        let all = planner.placements_in(Some(&network), Vec2::splat(-reach), Vec2::splat(reach));
        assert!(
            all.len() > 20,
            "only {} buildings within 5 km of spawn",
            all.len()
        );
        let towns = planner.towns_in(Some(&network), Vec2::splat(-reach), Vec2::splat(reach));
        assert!(!towns.is_empty(), "no town within 5 km of spawn");
        let sheriff = planner
            .footprints()
            .iter()
            .position(|f| f.name == "sheriff-office")
            .unwrap() as u32;
        for town in &towns {
            assert_eq!(
                town.placements
                    .iter()
                    .filter(|p| p.model == sheriff)
                    .count(),
                1,
                "town {}",
                town.id
            );
        }
        let water = WaterIndex::build(&network);
        for placement in &all {
            // Above the sea, out of the water, on a level pad.
            assert!(placement.pad.height > SEA_LEVEL + 3.0, "{placement:?}");
            let envelope = network.envelope(placement.pad.centre.x, placement.pad.centre.y);
            assert!(
                envelope.bank_distance > 10.0 && placement.pad.height > envelope.lake + 1.5,
                "{placement:?}"
            );
            if let Some((distance, _, _)) = water.nearest(placement.pad.centre, 25.0) {
                panic!("{placement:?} stands {distance} m from water");
            }
            let base = base_height(&noise, placement.pad.centre.x, placement.pad.centre.y);
            assert!(
                (base - placement.pad.height).abs() < 1.5,
                "{placement:?} against ground at {base}"
            );
        }
        // No two buildings stand on each other's footprints: the turned
        // rectangles are apart along one of their four axes.
        let footprints = planner.footprints();
        let overlap = |a: &Placement, b: &Placement| {
            let half = |p: &Placement| footprints[p.model as usize].half;
            let axes = |p: &Placement| [p.pad.axis, Vec2::new(-p.pad.axis.y, p.pad.axis.x)];
            let reach = |p: &Placement, axis: Vec2| {
                let [u, v] = axes(p);
                half(p).x * u.dot(axis).abs() + half(p).y * v.dot(axis).abs()
            };
            axes(a).into_iter().chain(axes(b)).all(|axis| {
                let gap = (a.pad.centre - b.pad.centre).dot(axis).abs();
                gap < reach(a, axis) + reach(b, axis)
            })
        };
        for (index, a) in all.iter().enumerate() {
            for b in &all[index + 1..] {
                assert!(
                    !overlap(a, b),
                    "{:?} and {:?} overlap",
                    a.pad.centre,
                    b.pad.centre
                );
            }
        }
        // The same buildings, asked for a piece at a time.
        let mut pieces = Vec::new();
        for z in 0..4 {
            for x in 0..4 {
                let minimum = Vec2::new(-reach + x as f32 * 2500.0, -reach + z as f32 * 2500.0);
                pieces.extend(planner.placements_in(
                    Some(&network),
                    minimum,
                    minimum + Vec2::splat(2500.0),
                ));
            }
        }
        pieces.sort_by_key(|placement| placement.id);
        assert_eq!(pieces, all);
    }
}
