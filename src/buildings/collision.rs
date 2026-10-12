//! What a walker feels of the buildings: walls, floors, stairs, ceilings and
//! doors.
//!
//! Each building's collision is the kit's own: closed convex pieces in
//! building space ([`super::assets::ConvexPiece`]). The player is the same
//! upright body the trees see, a circle of `PLAYER_RADIUS` from the feet to
//! the top of the head:
//!
//! - **Walls** are whatever reaches into the body above a step: each piece's
//!   cross-section between `BUILDING_STEP` over the feet and the head is a
//!   convex polygon, and the body is pushed out of it, so a walk into a wall
//!   slides along it. Floors, decks, ceilings and roofs never push sideways;
//!   they are stood on or bumped from below.
//! - **Floors** are the highest top of any piece under the body no more than a
//!   step over the feet, read at the centre and four points around it, so
//!   the gaps between deck boards hold nobody and a stair is climbed a tread
//!   at a time.
//! - **Ceilings** stop a jump: the lowest underside over the head.
//! - **Doors** are hinged boards. Walking into one turns it away from the
//!   body as far as it would have to move, within the swing its frame and the
//!   rooms around it allow ([`fit_door_swings`]); it swings on at the speed it
//!   was pushed, slows, and some seconds later eases shut, the way doors in
//!   open-world games do. Pushed at the hinge, or against its stop, it is a
//!   wall.

use super::assets::{BuildingModel, CollisionKind, ConvexPiece, DoorModel};
use bevy::math::{Vec2, Vec3};

/// The highest a walker steps up onto a building: a stair's riser, a porch's
/// deck from the ground. A railing or a window sill is higher.
pub const BUILDING_STEP: f32 = 0.5;
/// From the feet to the top of the head.
pub const BODY_HEIGHT: f32 = crate::constants::EYE_HEIGHT + 0.12;
/// The longest stretch of a walk resolved at once: under twice the thinnest
/// wall and the body's radius, so a fast walker cannot pass through a wall
/// between two tests.
pub const MAX_STEP: f32 = 0.1;
/// Settling passes per stretch, for corners where pieces meet.
const SETTLE_PASSES: usize = 3;
/// Where the floor is read around the body's centre.
const FOOTING_REACH: f32 = 0.2;
/// A door pushed nearer its hinge than this does not turn: the body is
/// pressing on the hinge.
const MIN_LEVER: f32 = 0.12;
/// Fastest a pushed door swings, radians a second.
const MAX_DOOR_SPEED: f32 = 5.0;
/// How quickly a swinging door slows, per second.
const DOOR_DAMPING: f32 = 2.4;
/// Seconds after the last push before a door starts to swing shut, and the
/// stiffness of its closer, per second squared.
const DOOR_HOLD: f32 = 1.6;
const DOOR_CLOSER: f32 = 2.2;
/// A door nearly shut and nearly still latches.
const DOOR_LATCH_ANGLE: f32 = 0.02;
/// The widest a door opens, radians, whatever room it has.
const DOOR_MAX_SWING: f32 = 1.85;
/// Swing analysis: the step it sweeps a door in, and the room it leaves short
/// of whatever it would touch.
const SWING_STEP: f32 = 0.035;
const SWING_SLACK: f32 = 0.06;

/// A building's place in the world: its origin and its turn about +Y.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BuildingFrame {
    pub origin: Vec3,
    pub cosine: f32,
    pub sine: f32,
}

impl BuildingFrame {
    pub fn new(origin: Vec3, yaw: f32) -> Self {
        let (sine, cosine) = yaw.sin_cos();
        Self { origin, cosine, sine }
    }

    /// A world point in building space.
    pub fn to_local(self, point: Vec3) -> Vec3 {
        let offset = point - self.origin;
        let x = offset.x * self.cosine - offset.z * self.sine;
        let z = offset.x * self.sine + offset.z * self.cosine;
        Vec3::new(x, offset.y, z)
    }

    /// A building-space point in the world.
    pub fn to_world(self, point: Vec3) -> Vec3 {
        let turned = self.turn(Vec2::new(point.x, point.z));
        self.origin + Vec3::new(turned.x, point.y, turned.y)
    }

    /// A building-space XZ direction in the world.
    pub fn turn(&self, vector: Vec2) -> Vec2 {
        Vec2::new(vector.x * self.cosine + vector.y * self.sine, -vector.x * self.sine + vector.y * self.cosine)
    }

    /// A world XZ direction in building space.
    #[cfg(test)]
    pub fn unturn(&self, vector: Vec2) -> Vec2 {
        Vec2::new(vector.x * self.cosine - vector.y * self.sine, vector.x * self.sine + vector.y * self.cosine)
    }
}

/// Turn a door-local XZ direction by a yaw about +Y.
pub fn turn_by(vector: Vec2, yaw: f32) -> Vec2 {
    let (sine, cosine) = yaw.sin_cos();
    Vec2::new(vector.x * cosine + vector.y * sine, -vector.x * sine + vector.y * cosine)
}

/// Whether a piece is stood on or bumped from below, never pushed against
/// from the side.
fn horizontal_surface(kind: CollisionKind) -> bool {
    matches!(kind, CollisionKind::Floor | CollisionKind::Ceiling | CollisionKind::Roof | CollisionKind::Deck)
}

// ---------------------------------------------------------------------------
// Pieces
// ---------------------------------------------------------------------------

/// The piece's cross-section between heights `low` and `high`, building space
/// XZ, as a counter-clockwise convex polygon: its corners inside the slab and
/// where its edges cross the slab's faces. Empty if it does not reach into it.
pub fn cross_section(piece: &ConvexPiece, low: f32, high: f32, out: &mut Vec<Vec2>) {
    out.clear();
    if piece.maximum.y <= low || piece.minimum.y >= high {
        return;
    }
    for vertex in &piece.vertices {
        if vertex.y >= low && vertex.y <= high {
            out.push(Vec2::new(vertex.x, vertex.z));
        }
    }
    for edge in &piece.edges {
        let a = piece.vertices[edge[0] as usize];
        let b = piece.vertices[edge[1] as usize];
        for level in [low, high] {
            if (a.y - level) * (b.y - level) < 0.0 {
                let t = (level - a.y) / (b.y - a.y);
                let crossing = a + (b - a) * t;
                out.push(Vec2::new(crossing.x, crossing.z));
            }
        }
    }
    convex_hull(out);
}

/// Andrew's monotone chain, in place, counter-clockwise.
fn convex_hull(points: &mut Vec<Vec2>) {
    if points.len() < 3 {
        return;
    }
    points.sort_by(|a, b| a.x.total_cmp(&b.x).then(a.y.total_cmp(&b.y)));
    points.dedup_by(|a, b| a.distance_squared(*b) < 1e-10);
    if points.len() < 3 {
        return;
    }
    let cross = |o: Vec2, a: Vec2, b: Vec2| (a - o).perp_dot(b - o);
    let mut hull: Vec<Vec2> = Vec::with_capacity(points.len() * 2);
    for &point in points.iter() {
        while hull.len() >= 2 && cross(hull[hull.len() - 2], hull[hull.len() - 1], point) <= 0.0 {
            hull.pop();
        }
        hull.push(point);
    }
    let lower = hull.len() + 1;
    for &point in points.iter().rev().skip(1) {
        while hull.len() >= lower && cross(hull[hull.len() - 2], hull[hull.len() - 1], point) <= 0.0 {
            hull.pop();
        }
        hull.push(point);
    }
    hull.pop();
    *points = hull;
}

/// The nearest point of a convex polygon's boundary to `point`, and whether
/// `point` is inside it. A polygon of one or two points is a point or a
/// segment.
fn nearest_on_polygon(polygon: &[Vec2], point: Vec2) -> (Vec2, bool) {
    let mut nearest = polygon[0];
    let mut nearest_distance = f32::INFINITY;
    let mut inside = polygon.len() >= 3;
    for index in 0..polygon.len() {
        let a = polygon[index];
        let b = polygon[(index + 1) % polygon.len()];
        let edge = b - a;
        if polygon.len() >= 3 && edge.perp_dot(point - a) < 0.0 {
            inside = false;
        }
        let length_squared = edge.length_squared();
        let t = if length_squared > 1e-12 { ((point - a).dot(edge) / length_squared).clamp(0.0, 1.0) } else { 0.0 };
        let candidate = a + edge * t;
        let distance = candidate.distance_squared(point);
        if distance < nearest_distance {
            nearest_distance = distance;
            nearest = candidate;
        }
    }
    (nearest, inside)
}

/// How far, and which way, a circle must move to leave a convex polygon.
fn push_out(polygon: &[Vec2], centre: Vec2, radius: f32) -> Option<Vec2> {
    if polygon.is_empty() {
        return None;
    }
    let (nearest, inside) = nearest_on_polygon(polygon, centre);
    let offset = centre - nearest;
    let distance = offset.length();
    if inside {
        // Out through the nearest edge, to a radius beyond it.
        let normal = if distance > 1e-6 { -offset / distance } else { Vec2::X };
        return Some(normal * (distance + radius));
    }
    if distance >= radius {
        return None;
    }
    let normal = if distance > 1e-6 { offset / distance } else { Vec2::X };
    Some(normal * (radius - distance))
}

/// Where a vertical line through `point` (building XZ) enters and leaves the
/// piece, if it passes through it.
pub fn vertical_span(piece: &ConvexPiece, point: Vec2) -> Option<(f32, f32)> {
    if point.x < piece.minimum.x || point.x > piece.maximum.x || point.y < piece.minimum.z || point.y > piece.maximum.z {
        return None;
    }
    let mut low = f32::NEG_INFINITY;
    let mut high = f32::INFINITY;
    for (normal, offset) in &piece.planes {
        let across = normal.x * point.x + normal.z * point.y;
        if normal.y.abs() < 1e-5 {
            if across > *offset + 1e-5 {
                return None;
            }
            continue;
        }
        let height = (offset - across) / normal.y;
        if normal.y > 0.0 {
            high = high.min(height);
        } else {
            low = low.max(height);
        }
    }
    (low <= high).then_some((low, high))
}

// ---------------------------------------------------------------------------
// Doors
// ---------------------------------------------------------------------------

/// Where a door is and how it moves.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DoorState {
    /// Radians from closed, within the door's swing.
    pub angle: f32,
    /// Radians a second.
    pub velocity: f32,
    /// Seconds since the body last pushed it.
    pub since_push: f32,
}

impl DoorState {
    pub fn is_closed(&self) -> bool {
        self.angle == 0.0 && self.velocity == 0.0
    }

    /// A door standing most of the way open, its closer never drawing it
    /// shut: for `--open-doors` captures.
    pub fn held_open(door: &DoorModel) -> Self {
        let wider = if door.swing[1] >= -door.swing[0] { door.swing[1] } else { door.swing[0] };
        DoorState {
            angle: wider * 0.8,
            velocity: 0.0,
            since_push: f32::NEG_INFINITY,
        }
    }

    /// One step of a door left to itself: it swings on, slows, and once it
    /// has stood a while its closer draws it shut and it latches.
    pub fn advance(&mut self, door: &DoorModel, dt: f32) {
        if self.is_closed() {
            return;
        }
        self.since_push += dt;
        if self.since_push > DOOR_HOLD {
            self.velocity -= DOOR_CLOSER * self.angle * dt;
        }
        self.velocity *= (-DOOR_DAMPING * dt).exp();
        self.angle += self.velocity * dt;
        if self.angle < door.swing[0] || self.angle > door.swing[1] {
            self.angle = self.angle.clamp(door.swing[0], door.swing[1]);
            // The leaf meets its stop and rebounds a little.
            self.velocity *= -0.2;
        }
        let latched = self.since_push > DOOR_HOLD && self.angle.abs() < DOOR_LATCH_ANGLE && self.velocity.abs() < 0.08;
        if latched {
            *self = DoorState::default();
        }
    }
}

/// The door's leaf as a segment in building XZ at its angle: hinge and free edge.
pub fn door_leaf(door: &DoorModel, angle: f32) -> (Vec2, Vec2) {
    let hinge = Vec2::new(door.hinge.x, door.hinge.z);
    let direction = turn_by(door.reach, door.yaw + angle);
    (hinge, hinge + direction * door.width)
}

fn nearest_on_segment(a: Vec2, b: Vec2, point: Vec2) -> Vec2 {
    let edge = b - a;
    let length_squared = edge.length_squared().max(1e-12);
    let t = ((point - a).dot(edge) / length_squared).clamp(0.0, 1.0);
    a + edge * t
}

/// Turn a door away from a body at `centre` (building XZ) that presses into
/// it over `dt` seconds, as far as the body would have to move to clear it;
/// returns how far the body must still be pushed out of the leaf, with the
/// leaf's state updated.
pub fn push_door(door: &DoorModel, state: &mut DoorState, centre: Vec2, radius: f32, dt: f32) -> Option<Vec2> {
    let reach = radius + door.thickness * 0.5;
    let (hinge, tip) = door_leaf(door, state.angle);
    let contact = nearest_on_segment(hinge, tip, centre);
    let offset = centre - contact;
    let distance = offset.length();
    if distance >= reach {
        return None;
    }
    let normal = if distance > 1e-6 { offset / distance } else { (tip - hinge).perp().normalize_or_zero() };
    let lever = contact.distance(hinge);
    let direction = (tip - hinge) / door.width.max(1e-3);
    // Turning the leaf by a small angle moves the contact along this.
    let sweep = Vec2::new(direction.y, -direction.x);
    let along = sweep.dot(normal);
    if lever > MIN_LEVER && along.abs() > 0.15 {
        let penetration = reach - distance;
        let wanted = -penetration / (along * lever) * 1.05;
        let target = (state.angle + wanted).clamp(door.swing[0], door.swing[1]);
        let turned = target - state.angle;
        if turned != 0.0 {
            state.angle = target;
            let speed = (turned / dt.max(1e-3)).clamp(-MAX_DOOR_SPEED, MAX_DOOR_SPEED);
            // Keep the faster of its own swing and the push, in the push's
            // direction: a body walking through keeps the door moving.
            state.velocity = if state.velocity * speed > 0.0 && state.velocity.abs() > speed.abs() { state.velocity } else { speed };
            state.since_push = 0.0;
        }
    }
    let (hinge, tip) = door_leaf(door, state.angle);
    let contact = nearest_on_segment(hinge, tip, centre);
    let offset = centre - contact;
    let distance = offset.length();
    if distance >= reach {
        return None;
    }
    let normal = if distance > 1e-6 { offset / distance } else { normal };
    Some(normal * (reach - distance))
}

/// How far each door can swing each way before its leaf would meet a wall, a
/// railing or anything else solid, short of `SWING_SLACK`: the room a hall
/// leaves it, a corner it opens into, the stop of a cell door that opens
/// outward only.
pub fn fit_door_swings(model: &mut BuildingModel) {
    let mut section = Vec::new();
    let mut swings = Vec::with_capacity(model.doors.len());
    for door in &model.doors {
        let low = door.bottom + 0.25;
        let high = door.top - 0.25;
        // The solid pieces the leaf could meet, as cross-sections at its height.
        let mut sections: Vec<Vec<Vec2>> = Vec::new();
        let hinge = Vec2::new(door.hinge.x, door.hinge.z);
        for piece in &model.pieces {
            if horizontal_surface(piece.kind) {
                continue;
            }
            let near_x = piece.minimum.x - door.width - 0.2 <= hinge.x && hinge.x <= piece.maximum.x + door.width + 0.2;
            let near_z = piece.minimum.z - door.width - 0.2 <= hinge.y && hinge.y <= piece.maximum.z + door.width + 0.2;
            if !near_x || !near_z {
                continue;
            }
            cross_section(piece, low, high, &mut section);
            if !section.is_empty() {
                sections.push(section.clone());
            }
        }
        let touches = |angle: f32, polygon: &[Vec2]| -> bool {
            let (hinge, tip) = door_leaf(door, angle);
            let samples = 12;
            (0..=samples).any(|index| {
                let along = 0.15 + (door.width - 0.15) * index as f32 / samples as f32;
                let point = hinge + (tip - hinge) * (along / door.width.max(1e-3));
                let (nearest, inside) = nearest_on_polygon(polygon, point);
                inside || nearest.distance(point) < door.thickness * 0.5 + SWING_SLACK
            })
        };
        // The jambs the shut leaf already stands against are what it swings
        // away from, not what stops it.
        sections.retain(|polygon| !touches(0.0, polygon));
        let blocked = |angle: f32| sections.iter().any(|polygon| touches(angle, polygon));
        let mut limits = [0.0f32; 2];
        for (side, sign) in [(0usize, -1.0f32), (1usize, 1.0f32)] {
            let mut angle = 0.0;
            while angle + SWING_STEP <= DOOR_MAX_SWING && !blocked(sign * (angle + SWING_STEP)) {
                angle += SWING_STEP;
            }
            limits[side] = sign * angle;
        }
        // A cell's barred door opens outward into the cell block only.
        if door.name.starts_with("cell-door") {
            limits[0] = 0.0;
        }
        swings.push(limits);
    }
    for (door, swing) in model.doors.iter_mut().zip(swings) {
        door.swing = swing;
    }
}

// ---------------------------------------------------------------------------
// The body
// ---------------------------------------------------------------------------

/// A building near the body, its doors' states alongside.
pub struct NearBuilding<'a> {
    pub model: &'a BuildingModel,
    pub frame: BuildingFrame,
    pub doors: &'a mut [DoorState],
}

/// Where a stretch of walk ended and what it pressed against.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Slide {
    /// World XZ.
    pub position: Vec2,
    /// The direction the body was last pushed back, world XZ, or zero.
    pub normal: Vec2,
}

/// Push a body at `position` (world XZ), its feet at `feet`, out of one
/// building's walls and doors, turning the doors it presses on over `dt`
/// seconds. Each piece sees the body where the last one left it. Returns the
/// direction it was last pushed, world XZ, if it was.
fn push_out_of_building(
    building: &mut NearBuilding<'_>,
    position: &mut Vec2,
    feet: f32,
    radius: f32,
    dt: f32,
    section: &mut Vec<Vec2>,
) -> Option<Vec2> {
    let mut pushed = None;
    for piece in &building.model.pieces {
        let local = building.frame.to_local(Vec3::new(position.x, feet, position.y));
        let centre = Vec2::new(local.x, local.z);
        let low = local.y + BUILDING_STEP;
        let high = local.y + BODY_HEIGHT;
        if horizontal_surface(piece.kind) || piece.maximum.y <= low || piece.minimum.y >= high {
            continue;
        }
        let clear_x = centre.x + radius < piece.minimum.x || centre.x - radius > piece.maximum.x;
        let clear_z = centre.y + radius < piece.minimum.z || centre.y - radius > piece.maximum.z;
        if clear_x || clear_z {
            continue;
        }
        cross_section(piece, low, high, section);
        let Some(push) = push_out(section, centre, radius) else {
            continue;
        };
        let world_push = building.frame.turn(push);
        *position += world_push;
        pushed = Some(world_push.normalize_or_zero());
    }
    for (door, state) in building.model.doors.iter().zip(building.doors.iter_mut()) {
        let local = building.frame.to_local(Vec3::new(position.x, feet, position.y));
        let overlaps = local.y + 0.05 < door.top && local.y + BODY_HEIGHT > door.bottom;
        if !overlaps {
            continue;
        }
        let centre = Vec2::new(local.x, local.z);
        let Some(push) = push_door(door, state, centre, radius, dt) else {
            continue;
        };
        let world_push = building.frame.turn(push);
        *position += world_push;
        pushed = Some(world_push.normalize_or_zero());
    }
    pushed
}

/// Walk a body with its feet at `feet` from `from` to `to` (world XZ) through
/// the buildings near it, sliding along their walls and pushing their doors
/// open over `dt` seconds.
pub fn slide(buildings: &mut [NearBuilding<'_>], from: Vec2, to: Vec2, feet: f32, radius: f32, dt: f32) -> Slide {
    let travel = to - from;
    let stretches = (travel.length() / MAX_STEP).ceil().max(1.0) as usize;
    let step = travel / stretches as f32;
    let stretch_dt = dt / stretches as f32;
    let mut position = from;
    let mut normal = Vec2::ZERO;
    let mut section = Vec::new();
    for _ in 0..stretches {
        position += step;
        // Settle: a corner where pieces meet takes a pass or two.
        for _ in 0..SETTLE_PASSES {
            let mut moved = false;
            for building in buildings.iter_mut() {
                let pushed = push_out_of_building(building, &mut position, feet, radius, stretch_dt, &mut section);
                if let Some(direction) = pushed {
                    normal = direction;
                    moved = true;
                }
            }
            if !moved {
                break;
            }
        }
    }
    Slide { position, normal }
}

/// The highest floor, deck, tread or anything else under a body at `point`
/// (world XZ) no more than a step above its feet, world height.
pub fn floor_under(buildings: &[NearBuilding<'_>], point: Vec2, feet: f32) -> Option<f32> {
    let mut best: Option<f32> = None;
    let offsets = [
        Vec2::ZERO,
        Vec2::new(FOOTING_REACH, 0.0),
        Vec2::new(-FOOTING_REACH, 0.0),
        Vec2::new(0.0, FOOTING_REACH),
        Vec2::new(0.0, -FOOTING_REACH),
    ];
    for building in buildings {
        let local = building.frame.to_local(Vec3::new(point.x, feet, point.y));
        let limit = local.y + BUILDING_STEP;
        for piece in &building.model.pieces {
            if piece.minimum.y > limit {
                continue;
            }
            for offset in offsets {
                let sample = Vec2::new(local.x, local.z) + offset;
                let Some((_, top)) = vertical_span(piece, sample) else {
                    continue;
                };
                if top <= limit {
                    let world_top = top + building.frame.origin.y;
                    best = Some(best.map_or(world_top, |known: f32| known.max(world_top)));
                }
            }
        }
    }
    best
}

/// How enclosed a body standing at `point` (world XZ) with its feet at `feet`
/// is: 1 on a building's floor or stair under its ceiling, a little on a deck
/// or a step under a roof (a porch), nothing in the open. The eye adapts to
/// it the way it does stepping indoors.
pub fn enclosure(buildings: &[NearBuilding<'_>], point: Vec2, feet: f32) -> f32 {
    let mut enclosed = 0.0f32;
    for building in buildings {
        let local = building.frame.to_local(Vec3::new(point.x, feet, point.y));
        let here = Vec2::new(local.x, local.z);
        let mut floor: Option<(f32, CollisionKind)> = None;
        let mut covered = false;
        for piece in &building.model.pieces {
            let Some((bottom, top)) = vertical_span(piece, here) else {
                continue;
            };
            if top <= local.y + 0.3 && floor.is_none_or(|(known, _)| top > known) {
                floor = Some((top, piece.kind));
            }
            if bottom > local.y + BUILDING_STEP && bottom < local.y + 4.5 {
                covered = true;
            }
        }
        let here_enclosed = match (floor, covered) {
            (Some((_, CollisionKind::Floor | CollisionKind::Stairs)), true) => 1.0,
            (Some((_, CollisionKind::Deck | CollisionKind::Steps)), true) => 0.35,
            _ => 0.0,
        };
        enclosed = enclosed.max(here_enclosed);
    }
    enclosed
}

/// The lowest underside over a body at `point` (world XZ) whose feet are at
/// `feet`, above a step over them, world height: what a jump bumps into.
pub fn ceiling_over(buildings: &[NearBuilding<'_>], point: Vec2, feet: f32) -> Option<f32> {
    let mut best: Option<f32> = None;
    for building in buildings {
        let local = building.frame.to_local(Vec3::new(point.x, feet, point.y));
        let floor = local.y + BUILDING_STEP;
        for piece in &building.model.pieces {
            if piece.maximum.y <= floor {
                continue;
            }
            let Some((bottom, _)) = vertical_span(piece, Vec2::new(local.x, local.z)) else {
                continue;
            };
            if bottom > floor {
                let world_bottom = bottom + building.frame.origin.y;
                best = Some(best.map_or(world_bottom, |known: f32| known.min(world_bottom)));
            }
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buildings::assets::tests::library;
    use crate::vegetation::collision::PLAYER_RADIUS;

    fn near<'a>(model: &'a BuildingModel, doors: &'a mut [DoorState]) -> Vec<NearBuilding<'a>> {
        vec![NearBuilding { model, frame: BuildingFrame::new(Vec3::new(100.0, 20.0, -50.0), 0.7), doors }]
    }

    #[test]
    fn frames_round_trip() {
        let frame = BuildingFrame::new(Vec3::new(3.0, 1.0, -2.0), 1.1);
        let point = Vec3::new(-4.0, 7.0, 9.0);
        assert!(frame.to_world(frame.to_local(point)).distance(point) < 1e-5);
        let direction = Vec2::new(0.6, -0.8);
        assert!(frame.unturn(frame.turn(direction)).distance(direction) < 1e-6);
        // The frame turns building space the way the door turn does.
        let turned = frame.turn(Vec2::X);
        assert!(turned.distance(turn_by(Vec2::X, 1.1)) < 1e-6);
    }

    #[test]
    fn a_wall_stops_a_walk_and_a_doorway_lets_it_through() {
        let library = library();
        let model = &library.models[library.model_index("house-c").unwrap()];
        let mut doors = vec![DoorState::default(); model.doors.len()];
        let mut buildings = near(model, &mut doors);
        let frame = buildings[0].frame;
        let floor = frame.origin.y + 0.47;
        // House-c's front wall stands at z = 3.57 (Blender's y = -3.57). From
        // the parlour out through it away from the door: the wall holds.
        let inside = frame.to_world(Vec3::new(-2.5, 0.0, 2.0));
        let outside = frame.to_world(Vec3::new(-2.5, 0.0, 6.0));
        let walk = slide(&mut buildings, Vec2::new(inside.x, inside.z), Vec2::new(outside.x, outside.z), floor, PLAYER_RADIUS, 0.5);
        let stopped = frame.to_local(Vec3::new(walk.position.x, floor, walk.position.y));
        assert!(stopped.z < 3.6, "walked through the front wall to {stopped}");
        // Through the front door, which swings open.
        let front = model.doors.iter().position(|door| door.name == "front-door").unwrap();
        let door = &model.doors[front];
        let (hinge, tip) = door_leaf(door, 0.0);
        let middle = (hinge + tip) * 0.5;
        // From the porch (outside is +Z) into the parlour.
        let start = frame.to_world(Vec3::new(middle.x, 0.0, middle.y + 1.0));
        let end = frame.to_world(Vec3::new(middle.x, 0.0, middle.y - 1.5));
        let walk = slide(&mut buildings, Vec2::new(start.x, start.z), Vec2::new(end.x, end.z), floor, PLAYER_RADIUS, 0.4);
        let through = frame.to_local(Vec3::new(walk.position.x, floor, walk.position.y));
        assert!(through.z < middle.y - 1.0, "stuck at the front door at {through}");
        assert!(buildings[0].doors[front].angle.abs() > 0.5, "the door did not swing: {:?}", buildings[0].doors[front]);
    }

    #[test]
    fn doors_swing_both_ways_or_one_and_never_through_walls() {
        let library = library();
        for model in &library.models {
            for door in &model.doors {
                let opening = door.swing[1] - door.swing[0];
                // A cell door opens against the next cell's bars.
                let least = if door.name.starts_with("cell-door") { 0.9 } else { 1.2 };
                assert!(opening > least, "{} {} swings only {:?}", model.name, door.name, door.swing);
                assert!(door.swing[0] <= 0.0 && door.swing[1] >= 0.0);
                if door.name.starts_with("cell-door") {
                    assert_eq!(door.swing[0], 0.0);
                }
            }
        }
    }

    #[test]
    fn a_door_left_open_swings_shut_and_latches() {
        let library = library();
        let model = &library.models[0];
        let door = &model.doors[0];
        let mut state = DoorState { angle: door.swing[1] * 0.8, velocity: 0.0, since_push: 0.0 };
        for _ in 0..(30 * 60) {
            state.advance(door, 1.0 / 60.0);
        }
        assert!(state.is_closed(), "{state:?}");
    }

    /// Walk a body in a straight line over `route` (building space XZ),
    /// sliding along walls and stepping onto whatever floor is a step or less
    /// above its feet, as the player does; returns where it ends, building
    /// space, and the height its feet end at.
    fn walk(buildings: &mut [NearBuilding<'_>], route: [Vec2; 2], feet: f32) -> (Vec2, f32) {
        let frame = buildings[0].frame;
        let start = frame.to_world(Vec3::new(route[0].x, 0.0, route[0].y));
        let end = frame.to_world(Vec3::new(route[1].x, 0.0, route[1].y));
        let mut position = Vec2::new(start.x, start.z);
        let mut feet = frame.origin.y + feet;
        let steps = (route[0].distance(route[1]) / 0.05).ceil() as usize;
        for step in 1..=steps {
            let along = Vec2::new(start.x, start.z).lerp(Vec2::new(end.x, end.z), step as f32 / steps as f32);
            let slid = slide(buildings, position, along, feet, PLAYER_RADIUS, 1.0 / 60.0);
            position = slid.position;
            if let Some(floor) = floor_under(buildings, position, feet) {
                feet = floor;
            }
        }
        let local = frame.to_local(Vec3::new(position.x, feet, position.y));
        (Vec2::new(local.x, local.z), local.y)
    }

    #[test]
    fn a_walker_climbs_the_stairs_inside_and_out() {
        let library = library();
        let model = &library.models[library.model_index("house-c").unwrap()];
        let mut doors = vec![DoorState::default(); model.doors.len()];
        let mut buildings = near(model, &mut doors);
        // The parlour's stair rises east between its stringers, from the
        // ground floor at 0.47 m to the upper floor at 3.0 m.
        let route = [Vec2::new(-4.2, 0.225), Vec2::new(0.6, 0.225)];
        let (end, feet) = walk(&mut buildings, route, 0.47);
        assert!((feet - 3.0).abs() < 0.05, "the stair left the feet at {feet} m, at {end}");
        assert!(end.x > 0.3, "the stair stopped the walker at {end}");
        // The outside stair climbs the east wall north from the ground to its
        // landing at 2.98 m.
        let route = [Vec2::new(5.05, 2.4), Vec2::new(5.05, -2.15)];
        let (end, feet) = walk(&mut buildings, route, 0.0);
        assert!((feet - 2.79).abs() < 0.25, "the outside stair left the feet at {feet} m, at {end}");
    }

    #[test]
    fn every_front_door_can_be_walked_through_from_the_yard() {
        let library = library();
        for model in &library.models {
            let mut doors = vec![DoorState::default(); model.doors.len()];
            let front = model.doors.iter().position(|door| door.name == "front-door").unwrap();
            let door = &model.doors[front];
            let (hinge, tip) = door_leaf(door, 0.0);
            let middle = (hinge + tip) * 0.5;
            let outward = door.outward;
            let mut buildings = near(model, &mut doors);
            let route = [middle + outward * 4.5, middle - outward * 1.5];
            let (end, feet) = walk(&mut buildings, route, 0.05);
            let past = (middle - end).dot(outward);
            assert!(past > 1.0, "{}: stopped {past:.2} m short of the doorway at {end}", model.name);
            assert!((feet - 0.47).abs() < 0.05, "{}: feet at {feet} m inside", model.name);
            assert!(buildings[0].doors[front].angle.abs() > 0.5, "{}: the front door did not open", model.name);
        }
    }

    #[test]
    fn floors_stairs_and_ceilings_are_where_the_house_has_them() {
        let library = library();
        let model = &library.models[library.model_index("house-c").unwrap()];
        let mut doors = vec![DoorState::default(); model.doors.len()];
        let buildings = near(model, &mut doors);
        let frame = buildings[0].frame;
        let parlour = frame.to_world(Vec3::new(-2.5, 0.0, 1.5));
        let point = Vec2::new(parlour.x, parlour.z);
        // Standing on the ground floor, the floor is under the feet.
        let floor = floor_under(&buildings, point, frame.origin.y + 0.47).expect("a floor in the parlour");
        assert!((floor - frame.origin.y - 0.47).abs() < 0.05, "floor at {}", floor - frame.origin.y);
        // Standing on the ground outside, the floor is a step up.
        let floor = floor_under(&buildings, point, frame.origin.y).expect("a step up onto the floor");
        assert!((floor - frame.origin.y - 0.47).abs() < 0.05);
        // The ceiling over the parlour is the upper floor, 2.47 m clear.
        let ceiling = ceiling_over(&buildings, point, frame.origin.y + 0.47).expect("a ceiling");
        assert!((ceiling - frame.origin.y - 2.94).abs() < 0.08, "ceiling at {}", ceiling - frame.origin.y);
    }
}
