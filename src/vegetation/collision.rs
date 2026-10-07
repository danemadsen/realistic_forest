//! What the player cannot walk through: tree trunks and the stems of lilacs.
//!
//! The scatter and the GPU cull own the plants, so collision reads the same
//! streamed chunks the renderer draws from and asks, for each plant within
//! reach of the player, the question the cull asks: is it rooted here? A tree
//! the cull refuses (on a steep face, in an incised channel, at the waterline)
//! is not drawn, and must not be felt either, so [`Solid::stand`] repeats the
//! cull's rooting rules on the CPU with the same terrain height the player
//! walks on.
//!
//! A trunk is a vertical cylinder from the ground to the top of the plant,
//! its radius measured from the model's bark ([`super::assets::PlantModel`]).
//! The player is a circle of [`PLAYER_RADIUS`]: [`slide`] moves it in steps
//! too short to jump a trunk and pushes it out of whatever it enters, so it
//! slides around a trunk rather than stopping against it. Crowns, shrubs,
//! ground plants and the lilac's foliage are not solid; only the stems.

use super::assets::Species;
use super::scatter::{self, Catalog, PlantInstance};
use super::{DEEPEST_FURROW, LOWEST_ROOT, STEEPEST_ROOT};
use crate::constants::SEA_LEVEL;
use crate::erosion::{ErosionCache, sample_eroded_height};
use crate::noise::NoiseField;
use bevy::math::Vec2;

/// Radius of the circle the player's body stands in, metres. Roomy enough that
/// the eye, which has a near plane, stays clear of the bark.
pub const PLAYER_RADIUS: f32 = 0.3;
/// The widest stem collision reads: bounds how far from the player a plant
/// whose position is known can still touch it. The widest the library
/// produces is a good deal under this.
pub const MAX_TRUNK_RADIUS: f32 = 3.0;
/// A sapling's stem is a twig, but it is still a stem: nothing is thinner
/// than this to the player.
const MIN_TRUNK_RADIUS: f32 = 0.06;
/// A trunk whose top is no higher than this above the player's feet is
/// stepped over rather than walked around: a stump, a seedling.
const STEP_UP: f32 = 0.3;
/// The longest stretch of a walk resolved at once. It is under the narrowest
/// pair of trunk and body, so a fast player cannot step across a thin trunk
/// between two tests.
const MAX_STEP: f32 = 0.25;
/// How many times a step is pushed out of the trunks before it is judged
/// wedged: enough to settle a cluster of touching stems.
const SETTLE_PASSES: usize = 4;
/// A body pushed out of a trunk stands on its boundary, where rounding can
/// leave it a hair inside; closer than this to the bark is not touching it.
const CONTACT_SLACK: f32 = 1e-3;
/// Where the cull samples the ground around a root to judge its slope and
/// whether it stands in a channel (vegetation-cull.wgsl, `step`).
const SLOPE_STEP: f32 = 2.5;

/// Whether a plant of this species is solid to the player: the trees and the
/// lilac. Shrubs, shore plants and lavender are walked through.
pub fn is_solid(species: Species) -> bool {
    matches!(
        species,
        Species::Fir | Species::Pine | Species::Oak | Species::Maple | Species::Lilac
    )
}

/// A streamed solid plant, before the ground it stands on is known.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Solid {
    /// World XZ of the stem.
    pub position: Vec2,
    /// Radius of the stem at the plant's scale, metres.
    pub radius: f32,
    /// Height of the plant at its scale, metres.
    pub height: f32,
    /// Reach of the crown at its scale, which keeps trees back from water.
    pub crown_radius: f32,
}

impl Solid {
    /// The plant as a solid, or `None` for the species the player walks through.
    pub fn of(catalog: &Catalog, plant: &PlantInstance) -> Option<Self> {
        let model = catalog.models.get(plant.model as usize)?;
        is_solid(model.species).then(|| Self {
            position: Vec2::new(plant.position[0], plant.position[2]),
            radius: (model.trunk_radius * plant.scale).clamp(MIN_TRUNK_RADIUS, MAX_TRUNK_RADIUS),
            height: model.height * plant.scale,
            crown_radius: model.crown_radius * plant.scale,
        })
    }

    /// The trunk the plant stands as on the ground the terrain pass draws, or
    /// `None` where the GPU cull refuses its root and so never draws it.
    /// `visibility_center` is the player, whose surroundings the erosion is
    /// faded in around.
    pub fn stand(
        &self,
        erosion: &ErosionCache,
        noise: &NoiseField,
        visibility_center: Vec2,
    ) -> Option<Trunk> {
        let height_at = |offset: Vec2| {
            let at = self.position + offset;
            sample_eroded_height(erosion, noise, at.x, at.y, visibility_center.to_array())
        };
        let ground = height_at(Vec2::ZERO);
        // Nothing grows in a river or a lake, nor on the margin its floods
        // scour; a tall tree stands further back, and a broad one by most of
        // its crown (the cull's `footing`).
        let envelope = erosion.rivers.as_ref().map_or(
            crate::rivers::carve::Envelope::NONE,
            |network| network.envelope(self.position.x, self.position.y),
        );
        if envelope.bank_at(ground) < footing(self.height, self.crown_radius) {
            return None;
        }
        let around = [
            height_at(Vec2::new(SLOPE_STEP, 0.0)),
            height_at(Vec2::new(-SLOPE_STEP, 0.0)),
            height_at(Vec2::new(0.0, SLOPE_STEP)),
            height_at(Vec2::new(0.0, -SLOPE_STEP)),
        ];
        rooted(ground, around).then_some(Trunk {
            position: self.position,
            radius: self.radius,
            top: ground + self.height,
        })
    }
}

/// How far past a waterline a tree of this height and crown reach stands at
/// least: vegetation-cull.wgsl's `footing` for a tree.
fn footing(height: f32, crown_radius: f32) -> f32 {
    (4.0 + 0.06 * height).max(scatter::crown_clearance(crown_radius))
}

/// The cull's rooting rules for a tree: ground at `ground` above the sea with
/// the heights `[east, west, north, south]` a `SLOPE_STEP` away around it.
/// Refused too low, on a face steeper than `STEEPEST_ROOT`, or in a furrow
/// deeper than `DEEPEST_FURROW` below the ground around it.
fn rooted(ground: f32, around: [f32; 4]) -> bool {
    let [east, west, north, south] = around;
    if ground < SEA_LEVEL + LOWEST_ROOT {
        return false;
    }
    let steepness = Vec2::new(east - west, north - south).length() / (2.0 * SLOPE_STEP);
    if steepness > STEEPEST_ROOT {
        return false;
    }
    0.25 * (east + west + north + south) - ground <= DEEPEST_FURROW
}

/// A solid stem standing on the ground.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Trunk {
    pub position: Vec2,
    pub radius: f32,
    /// World height of the plant's top.
    pub top: f32,
}

impl Trunk {
    /// Whether it is in the way of feet at world height `feet`.
    fn blocks(&self, feet: f32) -> bool {
        feet < self.top - STEP_UP
    }

    fn touches(&self, at: Vec2) -> bool {
        let reach = self.radius + PLAYER_RADIUS - CONTACT_SLACK;
        at.distance_squared(self.position) < reach * reach
    }
}

/// Where a step through the trunks ends.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Slide {
    pub position: Vec2,
    /// Unit direction away from what the body pressed against, or zero if it
    /// touched nothing.
    pub normal: Vec2,
}

/// Move the player's circle from `from` toward `to` with its feet at world
/// height `feet`, around every trunk the feet are low enough to meet. Each
/// short stretch of the walk moves the body, then pushes it out of the trunks
/// it entered along the line from their axes, so a walk into a trunk turns
/// into a slide along it. A body that starts inside a trunk (one streamed in
/// beneath it) is pushed out the nearest way.
pub fn slide(trunks: &[Trunk], from: Vec2, to: Vec2, feet: f32) -> Slide {
    let travel = to - from;
    let steps = (travel.length() / MAX_STEP).ceil().max(1.0);
    let increment = travel / steps;
    let blocking = || trunks.iter().filter(|trunk| trunk.blocks(feet));
    let mut position = from;
    let mut normal = Vec2::ZERO;
    for _ in 0..steps as usize {
        position += increment;
        for _ in 0..SETTLE_PASSES {
            let mut pushed = false;
            for trunk in blocking() {
                if !trunk.touches(position) {
                    continue;
                }
                let away = position - trunk.position;
                let direction = away
                    .try_normalize()
                    .or_else(|| (-travel).try_normalize())
                    .unwrap_or(Vec2::X);
                position = trunk.position + direction * (trunk.radius + PLAYER_RADIUS);
                normal += direction;
                pushed = true;
            }
            if !pushed {
                break;
            }
        }
    }
    // Squeezed between trunks too close for the body: stay where it was
    // rather than end the step inside one.
    if blocking().any(|trunk| trunk.touches(position)) && !blocking().any(|trunk| trunk.touches(from)) {
        return Slide { position: from, normal: normal.normalize_or_zero() };
    }
    Slide { position, normal: normal.normalize_or_zero() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::{EYE_HEIGHT, NOISE_RESOLUTION};

    fn trunk(x: f32, z: f32, radius: f32) -> Trunk {
        Trunk { position: Vec2::new(x, z), radius, top: 20.0 }
    }

    #[test]
    fn walking_into_a_trunk_stops_at_its_bark() {
        let tree = trunk(0.0, 0.0, 0.5);
        let end = slide(&[tree], Vec2::new(-3.0, 0.0), Vec2::new(0.0, 0.0), 0.0);
        assert!((end.position - Vec2::new(-(0.5 + PLAYER_RADIUS), 0.0)).length() < 1e-4, "{end:?}");
        assert!((end.normal - Vec2::NEG_X).length() < 1e-4, "{end:?}");
    }

    #[test]
    fn a_glancing_walk_slides_around_the_trunk() {
        let tree = trunk(0.0, 0.0, 0.5);
        // Aimed a little off the axis: the body keeps moving along the bark.
        let end = slide(&[tree], Vec2::new(-3.0, 0.2), Vec2::new(0.4, 0.2), 0.0);
        assert!(end.position.x > -0.8, "made no progress: {end:?}");
        assert!(end.position.y > 0.2, "did not slide to the side it leaned: {end:?}");
        assert!((end.position.length() - (0.5 + PLAYER_RADIUS)).abs() < 1e-3, "{end:?}");
    }

    #[test]
    fn a_fast_step_cannot_jump_a_thin_trunk() {
        // A 1.25 m step (a boosted sprint on a slow frame) over a twig.
        let twig = trunk(0.0, 0.0, MIN_TRUNK_RADIUS);
        let end = slide(&[twig], Vec2::new(-0.6, 0.0), Vec2::new(0.65, 0.0), 0.0);
        assert!(end.position.x < 0.0, "tunnelled through: {end:?}");
    }

    #[test]
    fn a_body_inside_a_trunk_is_pushed_out() {
        let tree = trunk(1.0, 1.0, 0.4);
        let end = slide(&[tree], Vec2::new(1.1, 1.0), Vec2::new(1.1, 1.0), 0.0);
        assert!((end.position.distance(tree.position) - (0.4 + PLAYER_RADIUS)).abs() < 1e-4, "{end:?}");
        // Dead centre has no direction to leave by, but still leaves.
        let end = slide(&[tree], tree.position, tree.position, 0.0);
        assert!((end.position.distance(tree.position) - (0.4 + PLAYER_RADIUS)).abs() < 1e-4, "{end:?}");
    }

    #[test]
    fn a_gap_narrower_than_the_body_stays_shut() {
        // Two trunks 0.7 m apart leave 0.1 m between their bark.
        let gate = [trunk(0.0, -0.35, 0.3), trunk(0.0, 0.35, 0.3)];
        let end = slide(&gate, Vec2::new(-2.0, 0.0), Vec2::new(2.0, 0.0), 0.0);
        assert!(end.position.x < 0.0, "squeezed through: {end:?}");
        assert!(!gate.iter().any(|trunk| trunk.touches(end.position)), "{end:?}");
    }

    #[test]
    fn a_gap_wider_than_the_body_lets_it_through() {
        let gate = [trunk(0.0, -1.0, 0.3), trunk(0.0, 1.0, 0.3)];
        let end = slide(&gate, Vec2::new(-2.0, 0.0), Vec2::new(2.0, 0.0), 0.0);
        assert!((end.position - Vec2::new(2.0, 0.0)).length() < 1e-4, "{end:?}");
        assert_eq!(end.normal, Vec2::ZERO);
    }

    #[test]
    fn a_stump_is_stepped_over_and_a_trunk_is_cleared_from_above() {
        let stump = Trunk { top: 0.25, ..trunk(0.0, 0.0, 0.3) };
        let end = slide(&[stump], Vec2::new(-2.0, 0.0), Vec2::new(2.0, 0.0), 0.0);
        assert_eq!(end.position, Vec2::new(2.0, 0.0));

        // A bush-high stem blocks a walker but not a body over the top of it.
        let shrub = Trunk { top: 1.6, ..trunk(0.0, 0.0, 0.3) };
        let walking = slide(&[shrub], Vec2::new(-2.0, 0.0), Vec2::new(2.0, 0.0), 0.0);
        assert!(walking.position.x < 0.0, "{walking:?}");
        let leaping = slide(&[shrub], Vec2::new(-2.0, 0.0), Vec2::new(2.0, 0.0), 1.5);
        assert_eq!(leaping.position, Vec2::new(2.0, 0.0));
    }

    #[test]
    fn a_cluster_of_touching_trunks_leaves_the_body_clear_of_all() {
        let cluster = [trunk(0.0, 0.0, 0.4), trunk(0.5, 0.2, 0.4), trunk(0.2, 0.7, 0.4)];
        let end = slide(&cluster, Vec2::new(-2.0, 0.3), Vec2::new(0.3, 0.3), 0.0);
        assert!(!cluster.iter().any(|trunk| trunk.touches(end.position)), "{end:?}");
    }

    fn catalog_of(species: Species) -> Catalog {
        Catalog {
            models: vec![scatter::CatalogModel {
                species,
                form: String::new(),
                height: 10.0,
                crown_radius: 2.0,
                trunk_radius: 0.25,
            }],
        }
    }

    fn plant(scale: f32) -> PlantInstance {
        PlantInstance { position: [12.0, 3.0, -7.0], scale, model: 0, ..Default::default() }
    }

    #[test]
    fn trees_and_lilacs_are_solid_and_the_ground_layers_are_not() {
        for solid in [Species::Fir, Species::Pine, Species::Oak, Species::Maple, Species::Lilac] {
            assert!(is_solid(solid), "{solid:?}");
        }
        for open in [Species::Bush, Species::Broadleaf, Species::Lavender] {
            assert!(!is_solid(open), "{open:?}");
            assert_eq!(Solid::of(&catalog_of(open), &plant(1.0)), None);
        }
    }

    #[test]
    fn a_solid_takes_its_size_from_its_scale() {
        let solid = Solid::of(&catalog_of(Species::Oak), &plant(2.0)).unwrap();
        assert_eq!(solid.position, Vec2::new(12.0, -7.0));
        assert_eq!((solid.radius, solid.height, solid.crown_radius), (0.5, 20.0, 4.0));
        // A seedling still has a stem; a monster does not swallow the chunk.
        let seedling = Solid::of(&catalog_of(Species::Oak), &plant(0.01)).unwrap();
        assert_eq!(seedling.radius, MIN_TRUNK_RADIUS);
        let monster = Solid::of(&catalog_of(Species::Oak), &plant(1000.0)).unwrap();
        assert_eq!(monster.radius, MAX_TRUNK_RADIUS);
    }

    #[test]
    fn the_cull_roots_a_tree_on_open_ground_and_refuses_the_rest() {
        let level = |ground: f32| [ground; 4];
        assert!(rooted(40.0, level(40.0)));
        // Under the lowest root, and the line is the cull's.
        assert!(!rooted(SEA_LEVEL + LOWEST_ROOT - 0.01, level(SEA_LEVEL + LOWEST_ROOT)));
        assert!(rooted(SEA_LEVEL + LOWEST_ROOT, level(SEA_LEVEL + LOWEST_ROOT)));
        // A face rising 0.8 per metre is rooted; 1.0 per metre is bare rock.
        let face = |rise: f32| [40.0 + rise * SLOPE_STEP, 40.0 - rise * SLOPE_STEP, 40.0, 40.0];
        assert!(rooted(40.0, face(0.8)));
        assert!(!rooted(40.0, face(1.0)));
        // A furrow: the ground around is well above the root.
        assert!(rooted(40.0, level(40.0 + DEEPEST_FURROW)));
        assert!(!rooted(40.0, level(40.0 + DEEPEST_FURROW + 0.1)));
    }

    #[test]
    fn rooting_rules_match_the_cull_shader() {
        // The cull is WGSL; these are the literals this port repeats.
        let shader = include_str!("../../assets/shaders/vegetation-cull.wgsl");
        for line in [
            "let step = 2.5;",
            "max(4.0 + 0.06 * height, 0.75 * model.crown_radius * plant.scale + 1.5)",
            "if (centre_height < cull.ground.x)",
            "if (steepness > cull.ground.y)",
            "0.25 * (east + west + north + south) - centre_height > cull.ground.z",
            "bank = riverBankAt(river, terrainHeight(xz));",
        ] {
            assert!(shader.contains(line), "vegetation-cull.wgsl no longer says `{line}`");
        }
        assert_eq!(SLOPE_STEP, 2.5);
        // Tall trees keep back by height; broad ones by most of their crown.
        assert!((footing(10.0, 1.0) - 4.6).abs() < 1e-5);
        assert!((footing(10.0, 8.0) - 7.5).abs() < 1e-5);
        assert!((footing(40.0, 1.0) - 6.4).abs() < 1e-5);
    }

    fn flat_world(field: f32) -> (NoiseField, ErosionCache) {
        (
            NoiseField { samples: vec![field; NOISE_RESOLUTION * NOISE_RESOLUTION] },
            ErosionCache::default(),
        )
    }

    #[test]
    fn a_tree_on_dry_ground_stands_to_its_full_height_and_one_in_the_sea_is_not_there() {
        let at = Vec2::new(1000.0, 1000.0);
        let tree = Solid { position: at, radius: 0.4, height: 12.0, crown_radius: 3.0 };

        let (noise, erosion) = flat_world(0.70);
        let ground = sample_eroded_height(&erosion, &noise, at.x, at.y, at.to_array());
        assert!(ground > SEA_LEVEL + LOWEST_ROOT);
        let trunk = tree.stand(&erosion, &noise, at).expect("rooted on open ground");
        assert_eq!((trunk.position, trunk.radius), (at, 0.4));
        assert!((trunk.top - (ground + 12.0)).abs() < 1e-3, "{trunk:?}");
        // The player's feet at the ground meet it; their eye is not the test.
        assert!(trunk.blocks(ground));
        assert!(trunk.blocks(ground + EYE_HEIGHT - 0.5));

        let (noise, erosion) = flat_world(0.0);
        assert!(sample_eroded_height(&erosion, &noise, at.x, at.y, at.to_array()) < SEA_LEVEL);
        assert_eq!(tree.stand(&erosion, &noise, at), None);
    }
}
