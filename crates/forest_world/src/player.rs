//! First-person player controller, ported from the C++ `Player` and
//! `UpdatePlayer` / `PlayerCamera`.

use crate::snow::{self, SnowState};
use bevy::camera::Projection;
use bevy::input::{ButtonInput, keyboard::KeyCode, mouse::MouseButton};
use bevy::math::{Vec2, Vec3};
use bevy::prelude::*;
use bevy::transform::components::Transform;
use bevy::window::{CursorGrabMode, CursorOptions, PrimaryWindow};
use forest_buildings::BuildingField;
use forest_buildings::collision::{self as building_collision, NearBuilding};
use forest_core::constants::*;
use forest_core::noise::NoiseField;
use forest_erosion::{ErosionCache, erosion_fade_anchor};
use forest_vegetation::VegetationField;
use forest_vegetation::collision::{self, PLAYER_RADIUS};

pub use forest_core::player::*;

/// The water over the ground at `(x, z)`, if any: a river's (a little splash
/// past its waterline counts), a lake's still water, or the sea's; where two
/// overlap, the higher surface is the one standing there.
pub fn water_at(
    erosion: &ErosionCache,
    noise: &NoiseField,
    x: f32,
    z: f32,
    visibility_center: [f32; 2],
) -> Option<WaterHere> {
    let envelope = erosion
        .rivers
        .as_ref()
        .map_or(forest_water::rivers::carve::Envelope::NONE, |network| {
            network.envelope(x, z)
        });
    let ground = forest_erosion::sample_eroded_height(erosion, noise, x, z, visibility_center);
    let drawn = erosion
        .rivers
        .as_ref()
        .and_then(|network| network.surface.drawn_at([x, z]));
    let inland = |surface: f32, current: Vec2, turbulence: f32, kind: WaterKind| WaterHere {
        surface,
        ground,
        current,
        turbulence,
        ..drawn_as(drawn, surface, ground, kind)
    };
    let river = (envelope.bank_distance < 0.5).then(|| {
        inland(
            envelope.water,
            Vec2::from(envelope.velocity),
            envelope.turbulence,
            WaterKind::River,
        )
    });
    let lake =
        (ground < envelope.lake).then(|| inland(envelope.lake, Vec2::ZERO, 0.0, WaterKind::Lake));
    let sea = WaterHere::sea(ground);
    [river, lake, sea]
        .into_iter()
        .flatten()
        .filter(|water| water.depth() > 0.0)
        .reduce(|a, b| if b.surface > a.surface { b } else { a })
}

/// A river's or a lake's water at `surface` over `ground` as it is drawn:
/// the colour shares of the surface drawn over it (`drawn`), its own kind.
/// With nothing drawn there, it is the sea's where it lies at the sea's level
/// over the sea floor (a river's mouth past its ribbon, where the sea draws
/// the water), else a stream's own. Only the colour shares and the kind are
/// meant; the rest is the caller's.
fn drawn_as(
    drawn: Option<forest_water::rivers::surface::SurfaceVertex>,
    surface: f32,
    ground: f32,
    kind: WaterKind,
) -> WaterHere {
    let (clarity, sea, still, kind) = match drawn {
        Some(drawn) => (drawn.clarity, drawn.sea, drawn.still, kind),
        None if ground < SEA_LEVEL && surface <= SEA_LEVEL + 0.03 => {
            (0.0, 1.0, 1.0, WaterKind::Sea)
        }
        None => (
            forest_water::rivers::surface::stream_clarity(surface),
            0.0,
            f32::from(u8::from(kind == WaterKind::Lake)),
            kind,
        ),
    };
    WaterHere {
        surface,
        ground,
        current: Vec2::ZERO,
        turbulence: 0.0,
        clarity,
        sea,
        still,
        kind,
    }
}

/// Per-frame player update, mirroring `UpdatePlayer`. The open developer menu
/// suppresses gameplay input while the passive debug text leaves it active.
#[expect(
    clippy::too_many_arguments,
    reason = "keeps the domain inputs explicit at this call boundary"
)]
pub fn update_player_system(
    mut player: Query<&mut Player>,
    mut cursor_options: Query<(&mut CursorOptions,), With<PrimaryWindow>>,
    keys: Res<ButtonInput<KeyCode>>,
    mouse_button: Res<ButtonInput<MouseButton>>,
    // Mouse motion is a buffered message in this bevy version, hence
    // MessageReader rather than the old EventReader.
    mut mouse_motion: MessageReader<bevy::input::mouse::MouseMotion>,
    time: Res<bevy::time::Time>,
    erosion: Res<ErosionCache>,
    noise: Res<NoiseField>,
    mut snow: ResMut<SnowState>,
    vegetation: Res<VegetationField>,
    mut buildings: ResMut<BuildingField>,
    ui_wants_input: Res<UiWantsInput>,
    automation: Option<Res<crate::automation::AutomationSettings>>,
) {
    let Ok(mut player) = player.single_mut() else {
        return;
    };
    snow.recenter(Vec2::new(player.position.x, player.position.z));
    // The water at the eye, which the renderer needs whether or not the
    // player can move. The ground under it is the terrain pass's, its erosion
    // faded round the same anchor; within a kilometre of the player the fade
    // is whole either way.
    let here = Vec2::new(player.position.x, player.position.z);
    let fade_anchor = erosion_fade_anchor(here.to_array());
    let water = water_at(
        &erosion,
        &noise,
        player.position.x,
        player.position.z,
        fade_anchor,
    );
    player.water = water;
    // A pinned --camera capture holds its pose: a click or keystroke that
    // lands on the window while it renders must not move the shot.
    if automation.is_some_and(|automation| automation.has_camera && automation.shot_path.is_some())
    {
        mouse_motion.clear();
        return;
    }
    let dt = time.delta_secs().min(0.05);
    let ui_wants_input = ui_wants_input.0;

    if keys.just_pressed(KeyCode::Escape) && player.mouse_captured {
        player.mouse_captured = false;
        if let Ok((mut cursor,)) = cursor_options.single_mut() {
            cursor.grab_mode = CursorGrabMode::None;
            cursor.visible = true;
        }
    }
    if !player.mouse_captured && mouse_button.just_pressed(MouseButton::Left) && !ui_wants_input {
        player.mouse_captured = true;
        player.mouse_warmup_frames = 3;
        if let Ok((mut cursor,)) = cursor_options.single_mut() {
            cursor.grab_mode = CursorGrabMode::Locked;
            cursor.visible = false;
        }
    }
    if keys.just_pressed(KeyCode::KeyV) && !ui_wants_input {
        let flying = !player.flying;
        player.set_flying(flying);
    }
    if player.mouse_captured && !ui_wants_input {
        let delta = mouse_motion
            .read()
            .map(|m| m.delta)
            .fold(Vec2::ZERO, |a, b| a + b);
        if player.mouse_warmup_frames > 0 {
            player.mouse_warmup_frames -= 1;
        } else {
            player.yaw += delta.x * 0.00225;
            player.pitch = (player.pitch - delta.y * 0.00225).clamp(-1.53, 1.53);
        }
    } else {
        mouse_motion.clear();
    }

    let mut movement = Vec3::ZERO;
    if !ui_wants_input {
        let forward = Vec3::new(player.yaw.sin(), 0.0, -player.yaw.cos());
        let right = Vec3::new(player.yaw.cos(), 0.0, player.yaw.sin());
        let forward_input =
            (keys.pressed(KeyCode::KeyW) as i32 - keys.pressed(KeyCode::KeyS) as i32) as f32;
        let right_input =
            (keys.pressed(KeyCode::KeyD) as i32 - keys.pressed(KeyCode::KeyA) as i32) as f32;
        movement = forward * forward_input + right * right_input;
        if movement.length_squared() > 1.0 {
            movement = movement.normalize();
        }
    }
    let boosted = !ui_wants_input && keys.pressed(KeyCode::ControlLeft);
    let speed = if player.flying { 38.0 } else { 10.0 }
        * if boosted { 2.5 } else { 1.0 }
        * player.movement_speed_multiplier;
    let walk = Vec2::new(movement.x, movement.z) * speed;
    let mut candidate = player.position + movement * (speed * dt);
    // The water where the step ends, once a branch below has looked.
    let mut landed = None;
    player.wading_depth = 0.0;
    player.current = Vec2::ZERO;
    player.swept = false;

    if player.flying {
        let vertical = if !ui_wants_input {
            (keys.pressed(KeyCode::Space) as i32 - keys.pressed(KeyCode::ShiftLeft) as i32) as f32
        } else {
            0.0
        };
        candidate.y += vertical * speed * dt;
        player.velocity = walk;
        player.swimming = false;
    } else {
        // The water over the player's feet: a river's, a lake's or the
        // sea's. Flowing water moves the body at the velocity the current's
        // drag and the player's footing settle on, not at the walking speed.
        let (surface, depth, current, turbulence) = match water {
            Some(water) => (
                water.surface,
                water.depth(),
                water.current,
                water.turbulence,
            ),
            None => (f32::NEG_INFINITY, 0.0, Vec2::ZERO, 0.0),
        };
        // In the water only while the feet are: a leap over a creek or a fall
        // onto a lake carries the body on through the air.
        let in_water = player.position.y - EYE_HEIGHT < surface + 0.05;
        if depth > 0.02 && in_water {
            let wading = wade(player.velocity, walk, current, depth, turbulence, dt);
            player.velocity = wading.velocity;
            player.swept = wading.swept;
            player.wading_depth = depth;
            player.current = current;
        } else if depth <= 0.02 {
            player.velocity = walk;
        }
        // Airborne over the water the body keeps the velocity it left the
        // water with: a hop does not shake off the current, nor stop a body
        // the current was carrying dead in mid-air.
        candidate.x = player.position.x + player.velocity.x * dt;
        candidate.z = player.position.z + player.velocity.y * dt;
        // Trunks and lilac stems are solid. Flight passes through them, as it
        // does the ground.
        collide_with_trunks(&mut player, &mut candidate, &vegetation, &erosion, &noise);
        // So are the buildings' walls, and their doors swing open from the
        // body that walks into them.
        let travel = here.distance(Vec2::new(candidate.x, candidate.z));
        let mut near = buildings.near_mut(here, PLAYER_RADIUS + travel + 1.0);
        collide_with_buildings(&mut player, &mut candidate, &mut near, dt);
        // Water too deep to stand in is swum in: the body floats, head out,
        // until the player dives. A swimmer crouched under standing height
        // stands up. The ground is what walking stands on, snow and all, or
        // a building's floor.
        let feet = player.position.y - EYE_HEIGHT;
        let terrain = snow::sample_surface(&erosion, &noise, here, fade_anchor).height(&snow, here);
        let ground = footing(terrain, &near, here, feet);
        let swimming = swims(
            water,
            ground,
            player.position.y,
            player.vertical_velocity,
            player.swimming,
        );
        let jump = keys.just_pressed(KeyCode::Space) && !ui_wants_input;
        let at_surface = kicks_off(water, player.position.y);
        if swimming && !(jump && at_surface) {
            let stroke = SwimStroke {
                rise: keys.pressed(KeyCode::Space) && !ui_wants_input,
                dive: keys.pressed(KeyCode::ShiftLeft) && !ui_wants_input,
                // Swimming forward follows the eye up or down, past a glance
                // either way, so the usual slightly lowered eye keeps to the
                // surface.
                forward_pitch: movement.dot(Vec3::new(player.yaw.sin(), 0.0, -player.yaw.cos()))
                    * (player.pitch.sin().abs() - SWIM_PITCH_DEAD_ZONE)
                        .max(0.0)
                        .copysign(player.pitch)
                    / (1.0 - SWIM_PITCH_DEAD_ZONE),
            };
            let candidate_xz = Vec2::new(candidate.x, candidate.z);
            let there = snow::sample_surface(&erosion, &noise, candidate_xz, fade_anchor);
            let mut bed = there.height(&snow, candidate_xz);
            // A bank is as solid to a swimmer as to a walker.
            if too_high_to_step(bed, ground, player.position.y) {
                candidate.x = player.position.x;
                candidate.z = player.position.z;
                bed = ground;
            }
            // Crouched on dry ground, standing up, the feet tread the snow as
            // a walker's do.
            if water.is_none()
                && there.coverage > 0.0
                && candidate_xz.distance_squared(here) > 0.000001
                && candidate_xz == Vec2::new(candidate.x, candidate.z)
            {
                snow.stamp_segment(here, candidate_xz);
                bed = there.height(&snow, candidate_xz);
            }
            let (y, vertical_velocity) = swim_vertical(
                player.position.y,
                player.vertical_velocity,
                stroke,
                surface,
                ground,
                bed,
                dt,
            );
            candidate.y = y;
            player.vertical_velocity = vertical_velocity;
            player.swimming = true;
            player.leapt = false;
        } else {
            // A swimmer at the surface kicks up out of the water as a walker
            // jumps off the ground.
            if swimming && jump {
                player.vertical_velocity = JUMP_SPEED;
                player.leapt = true;
            }
            let before = (player.position.y, player.vertical_velocity);
            let afloat = water
                .filter(|water| water.float_eye().is_some())
                .map(|water| water.surface);
            player.swimming = update_walking_ground(
                &mut player,
                &mut candidate,
                &erosion,
                &noise,
                &mut snow,
                &near,
                jump,
                afloat,
                dt,
            );
            // Falling into deep water ends at its surface, not the bed: the
            // water the step ends over takes the fall there and the swim
            // takes over.
            let landing = if candidate.x != player.position.x || candidate.z != player.position.z {
                let fade_anchor = erosion_fade_anchor([candidate.x, candidate.z]);
                water_at(&erosion, &noise, candidate.x, candidate.z, fade_anchor)
            } else {
                water
            };
            if let Some(landing) = landing
                && landing.float_eye().is_some()
                && let Some((y, vertical_velocity)) = takes_the_fall(
                    before,
                    (candidate.y, player.vertical_velocity),
                    landing.surface,
                )
            {
                candidate.y = y;
                player.vertical_velocity = vertical_velocity;
            }
            landed = Some(landing);
        }
    }
    // The step moved through the water where it started; the renderer gets
    // the water where it ends.
    player.water = match landed {
        Some(landing) => landing,
        None if candidate.x != player.position.x || candidate.z != player.position.z => water_at(
            &erosion,
            &noise,
            candidate.x,
            candidate.z,
            erosion_fade_anchor([candidate.x, candidate.z]),
        ),
        None => water,
    };
    player.position = candidate;
    snow.recenter(Vec2::new(candidate.x, candidate.z));
}

/// Where a falling body's step ends, eye height and vertical velocity, if
/// the step takes it from `from`, over deep water at `surface`, to `to`. A
/// step that would carry it past the float line, even to the bed, ends there
/// instead, at the speed the fall reached it with, so however long the frame
/// the swim takes the body over at the same place and speed; and the water
/// takes three quarters of that speed. A fall then reaches the same depth at
/// any frame rate. Only a body coming down onto the water from over it is
/// taken: not one rising through it, nor one already in it.
fn takes_the_fall(from: (f32, f32), to: (f32, f32), surface: f32) -> Option<(f32, f32)> {
    let (eye, vertical_velocity) = to;
    let float = surface + SWIM_FREEBOARD;
    let coming_down = vertical_velocity < 0.0 || eye < float;
    if from.0 < float || eye > surface + SWIM_ENTRY || !coming_down {
        return None;
    }
    let (eye, vertical_velocity) = if eye < float {
        let drop = (from.0 - float).max(0.0);
        (
            float,
            -(from.1 * from.1 + 2.0 * FALL_ACCELERATION * drop).sqrt(),
        )
    } else {
        (eye, vertical_velocity)
    };
    Some((eye, vertical_velocity * 0.25))
}

/// Slide the step from the player's position to `candidate` around the
/// trunks and stems it would run into, and take from the player's velocity
/// what pressed into them. Only the ground-level circle matters: a trunk
/// blocks the feet while they are lower than its top (less a step), so a jump
/// clears a stem under a metre and a half and nothing taller.
fn collide_with_trunks(
    player: &mut Player,
    candidate: &mut Vec3,
    vegetation: &VegetationField,
    erosion: &ErosionCache,
    noise: &NoiseField,
) {
    let from = Vec2::new(player.position.x, player.position.z);
    let to = Vec2::new(candidate.x, candidate.z);
    let mut solids = Vec::new();
    vegetation.solids_near(from, PLAYER_RADIUS + from.distance(to), &mut solids);
    if solids.is_empty() {
        return;
    }
    // A tree the GPU would not draw is not there to walk into.
    let fade_anchor = Vec2::from(erosion_fade_anchor(from.to_array()));
    let trunks: Vec<collision::Trunk> = solids
        .iter()
        .filter_map(|solid| solid.stand(erosion, noise, fade_anchor))
        .collect();
    let feet = player.position.y - EYE_HEIGHT;
    let slid = collision::slide(&trunks, from, to, feet);
    candidate.x = slid.position.x;
    candidate.z = slid.position.y;
    let pressing = player.velocity.dot(slid.normal);
    if pressing < 0.0 {
        player.velocity -= slid.normal * pressing;
    }
}

/// Slide the step from the player's position to `candidate` along the walls
/// of the buildings near it, pushing their doors open, and take from the
/// player's velocity what pressed into a wall.
fn collide_with_buildings(
    player: &mut Player,
    candidate: &mut Vec3,
    near: &mut [NearBuilding<'_>],
    dt: f32,
) {
    if near.is_empty() {
        return;
    }
    let from = Vec2::new(player.position.x, player.position.z);
    let to = Vec2::new(candidate.x, candidate.z);
    let feet = player.position.y - EYE_HEIGHT;
    let slid = building_collision::slide(near, from, to, feet, PLAYER_RADIUS, dt);
    candidate.x = slid.position.x;
    candidate.z = slid.position.y;
    let pressing = player.velocity.dot(slid.normal);
    if pressing < 0.0 {
        player.velocity -= slid.normal * pressing;
    }
}

/// What a body whose feet are at `feet` stands on at `point`: the ground,
/// or a building's floor, deck or stair tread a step or less above its feet.
fn footing(terrain: f32, near: &[NearBuilding<'_>], point: Vec2, feet: f32) -> f32 {
    let floor = building_collision::floor_under(near, point, feet);
    floor.map_or(terrain, |floor| floor.max(terrain))
}

/// Resolve against the puffed surface first, then compact only the contact
/// segment and seat the player's feet on the newly flattened snow. Returns
/// whether the body is still a swimmer's (`Player::swimming`): kicked up out
/// of the water and still rising, or landed crouched on a bank, to stand up.
/// A building's floors are stood on as the ground is, and its ceilings stop
/// a jump.
#[expect(
    clippy::too_many_arguments,
    reason = "keeps the domain inputs explicit at this call boundary"
)]
fn update_walking_ground(
    player: &mut Player,
    candidate: &mut Vec3,
    erosion: &ErosionCache,
    noise: &NoiseField,
    snow: &mut SnowState,
    near: &[NearBuilding<'_>],
    jump_requested: bool,
    afloat: Option<f32>,
    dt: f32,
) -> bool {
    let previous_xz = Vec2::new(player.position.x, player.position.z);
    let fade_anchor = erosion_fade_anchor(previous_xz.to_array());
    let feet = player.position.y - EYE_HEIGHT;
    let old_surface = snow::sample_surface(erosion, noise, previous_xz, fade_anchor);
    let old_ground = footing(
        old_surface.height(snow, previous_xz),
        near,
        previous_xz,
        feet,
    );
    let mut next_xz = Vec2::new(candidate.x, candidate.z);
    let mut next_surface = snow::sample_surface(erosion, noise, next_xz, fade_anchor);
    let next_ground = footing(next_surface.height(snow, next_xz), near, next_xz, feet);
    if step_blocked(
        next_ground,
        old_ground,
        player.position.y,
        player.vertical_velocity,
        afloat,
        player.leapt,
    ) {
        candidate.x = player.position.x;
        candidate.z = player.position.z;
        next_xz = previous_xz;
        next_surface = old_surface;
    }
    let grounded_before =
        player.position.y <= old_ground + EYE_HEIGHT + 0.03 && player.vertical_velocity <= 0.0;
    let jumping = jump_requested && grounded_before;
    if jumping {
        player.vertical_velocity = JUMP_SPEED;
        player.leapt = true;
    }
    let swimmer = kicked(afloat, player.vertical_velocity, player.swimming);
    (candidate.y, player.vertical_velocity) = fall(
        candidate.y,
        player.vertical_velocity,
        grounded_before && !jumping,
        dt,
    );
    // A jump inside a building stops at its ceiling.
    if player.vertical_velocity > 0.0
        && let Some(ceiling) = building_collision::ceiling_over(near, next_xz, feet)
    {
        let highest_eye = ceiling - (building_collision::BODY_HEIGHT - EYE_HEIGHT);
        if candidate.y > highest_eye {
            candidate.y = highest_eye.max(player.position.y.min(highest_eye));
            player.vertical_velocity = 0.0;
        }
    }
    let floor = building_collision::floor_under(near, next_xz, feet);
    let on_floor = floor.is_some_and(|floor| floor >= next_surface.height(snow, next_xz));
    let ground_before_stamp =
        footing(next_surface.height(snow, next_xz), near, next_xz, feet) + EYE_HEIGHT;
    let touching = candidate.y <= ground_before_stamp && player.vertical_velocity <= 0.0;
    if touching {
        // Snow is trodden only where the feet are on it, not on a porch.
        if !jumping && next_surface.coverage > 0.0 && !on_floor {
            if grounded_before && next_xz.distance_squared(previous_xz) > 0.000001 {
                // Both endpoints are actual contacts in this update. There
                // is no stored segment to bridge jumps or camera teleports.
                snow.stamp_segment(previous_xz, next_xz);
            } else if !grounded_before {
                snow.stamp_segment(next_xz, next_xz);
            }
        }
        let crouched;
        let standing_on = floor.map_or(next_surface.height(snow, next_xz), |floor| {
            floor.max(next_surface.height(snow, next_xz))
        });
        (candidate.y, crouched) = lands(candidate.y, standing_on, swimmer);
        player.vertical_velocity = 0.0;
        player.leapt = false;
        return crouched;
    }
    swimmer
}

/// A body's eye height and vertical velocity after a step of falling. One in
/// the air moves exactly, so a jump or a fall goes as high or as far at any
/// frame rate. A walker on its feet (`on_feet`) is pulled down a whole step's
/// fall a step, as hard again, so walking down a slope it keeps its footing,
/// and Space finds it there to jump.
fn fall(eye: f32, vertical_velocity: f32, on_feet: bool, dt: f32) -> (f32, f32) {
    let after = vertical_velocity - FALL_ACCELERATION * dt;
    let moved = if on_feet {
        after
    } else {
        0.5 * (vertical_velocity + after)
    };
    (eye + moved * dt, after)
}

/// Where a body that comes down onto ground at `ground` this step, its eye
/// at `eye`, ends, and whether crouched: on its feet; or a swimmer kicked up
/// out of the water (`swimmer`) where it lands, crouched under standing
/// height, to stand up as a swimmer does rather than be lifted onto its feet
/// in a frame.
fn lands(eye: f32, ground: f32, swimmer: bool) -> (f32, bool) {
    let standing = ground + EYE_HEIGHT;
    if swimmer && eye < standing - STANDING_MARGIN {
        (eye.max(ground + SWIM_BED_CLEARANCE), true)
    } else {
        (standing, false)
    }
}

/// The highest a body steps up onto, metres.
const STEP_HEIGHT: f32 = 1.25;
/// How fast a jump, or a swimmer's kick up out of the water, leaves, m/s.
const JUMP_SPEED: f32 = 7.4;
/// How fast a body in the air gathers downward speed, m/s². Stronger than
/// gravity, so jumps and falls feel snappy rather than floaty.
const FALL_ACCELERATION: f32 = 23.0;
/// How far over the water the eye may be and swim: a body falling into deep
/// water is taken by it on the frame its eye comes within this.
const SWIM_ENTRY: f32 = SWIM_FREEBOARD + 0.03;
/// How far under standing height a body still counts as standing: walking
/// seats a body this close to it, and one crouched lower stands up.
const STANDING_MARGIN: f32 = 0.03;
/// How fast a swimmer climbs or dives, m/s.
const SWIM_VERTICAL_SPEED: f32 = 1.4;
/// How fast a swimmer who does nothing drifts back up: a body with a breath
/// in it floats, but only just.
const SWIM_BUOYANT_RISE: f32 = 0.35;
/// How closely the eye of a swimmer stretched out along the bed comes to it.
const SWIM_BED_CLEARANCE: f32 = 0.35;
/// How quickly the water's drag brings a swimmer's climb or dive to the speed
/// their stroke sets, per second.
const SWIM_RESPONSE: f32 = 3.0;
/// Sine of the pitch below which swimming forward does not climb or dive
/// (about 17 degrees).
const SWIM_PITCH_DEAD_ZONE: f32 = 0.3;

/// What the swimmer is doing this frame.
#[derive(Clone, Copy, Debug, Default)]
pub struct SwimStroke {
    /// Swimming up (Space held).
    pub rise: bool,

    /// Diving (Left Shift held).
    pub dive: bool,

    /// Forward input times the sine of the eye's pitch: swimming forward
    /// follows the eye down into the water or up out of it.
    pub forward_pitch: f32,
}

/// Whether ground at `next` is too high for a body to step onto from ground
/// at `ground`, its eye at `eye` and falling at `-vertical_velocity`, over
/// deep water at the surface `afloat` if it is. A walker steps up to
/// `STEP_HEIGHT`. A swimmer kicked up out of the water, or a body leaping
/// over it (`leapt`), steps up from its feet, not from the bed far under
/// them, so it reaches a bank; a body that walked or fell off something, or
/// falls faster than a jump does, lands on nothing over its feet.
fn step_blocked(
    next: f32,
    ground: f32,
    eye: f32,
    vertical_velocity: f32,
    afloat: Option<f32>,
    leapt: bool,
) -> bool {
    if kicking(afloat, vertical_velocity, leapt) {
        too_high_to_step(next, ground, eye)
    } else if afloat.is_some() {
        next > ground.max(eye - EYE_HEIGHT)
    } else {
        next - ground > STEP_HEIGHT
    }
}

/// Whether a body in the air over deep water (`afloat`, its surface), rising
/// at `vertical_velocity`, is a swimmer's kick up out of it or a leap over
/// it: it jumped or kicked (`leapt`), and comes down no faster than a jump
/// does. Such a body steps up onto a bank from its feet; one that walked off
/// a bank, or falls faster, from a cliff top say, lands on nothing over its
/// feet.
fn kicking(afloat: Option<f32>, vertical_velocity: f32, leapt: bool) -> bool {
    leapt && afloat.is_some() && vertical_velocity >= -JUMP_SPEED
}

/// Whether a body in the air this step, over deep water at the surface
/// `afloat` if it is, is still a swimmer's (`swam`): kicked up out of the
/// water and rising, out over a bank too, or coming back down over the
/// water no faster than a kick does. A walker's leap is a walker's, and
/// lands on its feet.
fn kicked(afloat: Option<f32>, vertical_velocity: f32, swam: bool) -> bool {
    swam && (vertical_velocity > 0.0 || (afloat.is_some() && vertical_velocity >= -JUMP_SPEED))
}

/// Whether Space kicks a swimmer whose eye is at `eye` up out of `water`
/// rather than swimming it up: at the surface, or crouched on dry ground
/// (come ashore), where it jumps as a walker does.
fn kicks_off(water: Option<WaterHere>, eye: f32) -> bool {
    water.is_none_or(|water| eye >= water.surface + SWIM_FREEBOARD - 0.05)
}

/// Whether ground at `bed` is too high for a swimmer whose eye is at `eye`,
/// over ground at `ground`, to move onto: more than a step over its feet,
/// or over the bed a diver lies along.
fn too_high_to_step(bed: f32, ground: f32, eye: f32) -> bool {
    bed - ground.max(eye - EYE_HEIGHT) > STEP_HEIGHT
}

/// Whether a body whose eye is at `eye`, over ground at `ground` and in
/// `water` if any, rising at `vertical_velocity`, swims this step, having
/// swum the last if `swam`. Water too deep to stand in is swum in once the
/// eye comes down within `SWIM_ENTRY` of its surface. A swimmer crouched
/// under standing height, a diver come up into the shallows or out onto the
/// bank, swims on until it has stood up (`swim_vertical`) rather than being
/// lifted to standing height in one step. A walker is set on its feet by
/// walking, wherever the ground puts them, and so is an eye under the ground
/// (flight set down inside a hill). A body rising faster than any stroke,
/// kicked up out of the water, is in the air until it falls back.
fn swims(
    water: Option<WaterHere>,
    ground: f32,
    eye: f32,
    vertical_velocity: f32,
    swam: bool,
) -> bool {
    let floating =
        water.is_some_and(|water| water.float_eye().is_some() && eye <= water.surface + SWIM_ENTRY);
    let crouched = swam && eye > ground && eye < ground + EYE_HEIGHT - STANDING_MARGIN;
    (floating || crouched) && vertical_velocity <= SWIM_VERTICAL_SPEED
}

/// The eye height and vertical velocity a swimmer comes to after one step
/// from over a bed at `previous_bed` to over one at `bed`: the stroke sets a
/// climb or a dive, the water's drag eases the body toward it, a body left
/// alone drifts up to float head out, and neither the surface (the swimmer
/// floats there) nor the bed stops being solid.
///
/// Where the water is too shallow to float in, the swimmer stands up out of
/// it instead, to standing height and no higher, at a stroke's pace from the
/// bed as it rises under them: left alone, and whatever the stroke where it
/// is too shallow to lie under or there is none (`surface` -inf, a body come
/// out onto the bank). A swimmer the bed falls away from under, off the edge
/// of a bank or a shelf, is left where it is: walking takes it on, and it
/// falls, onto the ground or into the water.
pub fn swim_vertical(
    eye: f32,
    vertical_velocity: f32,
    stroke: SwimStroke,
    surface: f32,
    previous_bed: f32,
    bed: f32,
    dt: f32,
) -> (f32, f32) {
    let standing = bed + EYE_HEIGHT;
    let shallow = standing > surface + SWIM_FREEBOARD;
    let aground = surface < bed + SWIM_BED_CLEARANCE;
    // A body crouched over the bed rises with it.
    let eye = if shallow && eye < previous_bed + EYE_HEIGHT {
        eye + (bed - previous_bed).max(0.0)
    } else {
        eye
    };
    let top = if shallow {
        standing
    } else {
        surface + SWIM_FREEBOARD
    };
    if eye > if shallow { top } else { surface + SWIM_ENTRY } {
        return (eye, vertical_velocity.min(0.0));
    }
    let mut target = SWIM_VERTICAL_SPEED
        * (stroke.forward_pitch.clamp(-1.0, 1.0) + f32::from(u8::from(stroke.rise))
            - f32::from(u8::from(stroke.dive)));
    if aground || (!stroke.rise && !stroke.dive && stroke.forward_pitch.abs() < 0.05) {
        target = if shallow {
            SWIM_VERTICAL_SPEED
        } else {
            SWIM_BUOYANT_RISE
        };
    }
    let target = target.clamp(-SWIM_VERTICAL_SPEED, SWIM_VERTICAL_SPEED);
    // The velocity eases to the target exponentially; the body moves by its
    // exact integral over the step, so a swim covers the same water at any
    // frame rate.
    let ease = 1.0 - (-SWIM_RESPONSE * dt).exp();
    let mut velocity = vertical_velocity + (target - vertical_velocity) * ease;
    let mut y = eye + target * dt + (vertical_velocity - target) * ease / SWIM_RESPONSE;
    let bottom = (bed + SWIM_BED_CLEARANCE).min(top);
    // Stood up as near as walking seats a body, the swimmer stands.
    if y >= top - if shallow { STANDING_MARGIN } else { 0.0 } {
        y = top;
        velocity = velocity.min(0.0);
    }
    if y <= bottom {
        y = bottom;
        velocity = velocity.max(0.0);
    }
    (y, velocity)
}

pub struct Wading {
    pub velocity: Vec2,
    /// The current overpowers the player's footing.
    pub swept: bool,
}

/// One step of a body standing (or swimming) in flowing water.
///
/// The water drags on what of the body it covers, `½ ρ Cd A u²`, with `A`
/// the legs below the knee and then the hips and torso as it deepens.
/// Against it the feet hold with at most the friction their weight allows,
/// and buoyancy takes weight off them, so the deeper the water the less the
/// footing and the more the drag. While the drag is a small part of the
/// footing the player stands firm; as it grows the current carries the body
/// more and more, and once it outweighs the footing it sweeps the player
/// off their feet. Wading itself is slow: pushing legs through water costs
/// more the deeper it is, and walking upstream is slower than walking down.
/// Water too deep to stand in floats the player with the current, swimming
/// weakly.
pub fn wade(body: Vec2, walk: Vec2, current: Vec2, depth: f32, turbulence: f32, dt: f32) -> Wading {
    const MASS: f32 = 75.0;
    const WATER_DENSITY: f32 = 1000.0;
    const DRAG_COEFFICIENT: f32 = 1.0;
    const BODY_VOLUME: f32 = 0.075;
    let gravity = forest_water::rivers::network::GRAVITY;
    let submerged = depth.clamp(0.0, EYE_HEIGHT + 0.1);
    let area = 0.30 * submerged.min(0.55) + 0.45 * (submerged - 0.55).max(0.0);
    let speed = current.length();
    let drag = 0.5 * WATER_DENSITY * DRAG_COEFFICIENT * area * speed * speed;
    let buoyancy =
        WATER_DENSITY * gravity * BODY_VOLUME * (submerged / (EYE_HEIGHT + 0.1)).powf(1.3);
    // Wet, rounded stones under whitewater grip less than a gravel bed.
    let grip = 0.6 - 0.2 * turbulence.clamp(0.0, 1.0);
    let footing = grip * (MASS * gravity - buoyancy).max(0.0);
    let floating = depth > EYE_HEIGHT - SWIM_FREEBOARD;
    let load = drag / footing.max(1.0);
    let swept = floating || load > 1.0;
    // How much of the current the body takes on.
    let carried = if floating {
        1.0
    } else {
        ((load - 0.3) / 1.2).clamp(0.0, 1.0)
    };
    // Walking through water: slower the deeper it is, and hardly at all once
    // the feet are gone.
    let stride = 1.0 / (1.0 + 2.5 * submerged);
    let control = if floating {
        0.12
    } else if swept {
        0.25 * stride
    } else {
        stride
    };
    let target = walk * control + current * carried;
    // The body follows over a fraction of a second: water is heavy.
    let response = if swept { 2.5 } else { 6.0 };
    let velocity = body + (target - body) * (1.0 - (-response * dt).exp());
    Wading { velocity, swept }
}

/// Set each frame before update_player runs: true when the diagnostics panel
/// is open and egui currently wants the pointer or keyboard.
#[derive(Resource, Default)]
pub struct UiWantsInput(pub bool);

/// Sync the player's Transform (used by bevy camera extraction, which only
/// needs the entity's placement; our renderer computes the exact matrices
/// itself from the Player state).
pub fn sync_player_camera_transform(
    mut player: Query<&Player>,
    mut transforms: Query<(&mut Transform, &mut Projection), bevy::ecs::query::Without<Player>>,
) {
    let Ok(player) = player.single_mut() else {
        return;
    };
    let camera = PlayerCamera::from_player(player);
    if let Ok((mut transform, mut projection)) = transforms.single_mut() {
        transform.translation = camera.position;
        transform.look_at(camera.target, Vec3::Y);
        if let Projection::Perspective(perspective) = &mut *projection {
            perspective.fov = camera.fov_y.to_radians();
        }
    }
}

#[cfg(test)]
#[path = "player_tests.rs"]
mod tests;
