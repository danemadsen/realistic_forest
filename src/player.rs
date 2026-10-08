//! First-person player controller, ported from the C++ `Player` and
//! `UpdatePlayer` / `PlayerCamera`.

use crate::constants::*;
use crate::erosion::ErosionCache;
use crate::noise::NoiseField;
use crate::snow::{self, SnowState};
use crate::vegetation::VegetationField;
use crate::vegetation::collision::{self, PLAYER_RADIUS};
use bevy::camera::Projection;
use bevy::input::{ButtonInput, keyboard::KeyCode, mouse::MouseButton};
use bevy::math::{Vec2, Vec3};
use bevy::prelude::*;
use bevy::transform::components::Transform;
use bevy::window::{CursorGrabMode, CursorOptions, PrimaryWindow};

#[derive(Component)]
pub struct Player {
    pub position: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub vertical_velocity: f32,
    pub flying: bool,
    pub movement_speed_multiplier: f32,
    pub mouse_captured: bool,
    pub mouse_warmup_frames: u32,
    /// Horizontal velocity of the body over the ground, m/s. On dry land it
    /// is whatever the player walks; in a river it is what the current and
    /// the player's footing leave of it.
    pub velocity: Vec2,
    /// How deep the water the player stands in is, metres (0 on dry land).
    pub wading_depth: f32,
    /// The river's current at the player, m/s, world XZ.
    pub current: Vec2,
    /// True while the current has taken the player's feet from under them.
    pub swept: bool,
    /// The water standing where the player (and so the camera) is, whether
    /// or not the player is in it: the renderer reads it to know when the
    /// eye has gone under a river, a lake or the sea.
    pub water: Option<WaterHere>,
}

/// What kind of water stands somewhere.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WaterKind {
    River,
    Lake,
    Sea,
}

/// The water standing at a point: its surface, the ground under it, its
/// current and what kind of water it is.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterHere {
    pub surface: f32,
    pub ground: f32,
    /// Surface current, world XZ, m/s; zero in a lake or the sea.
    pub current: Vec2,
    /// Whitewater, 0..1.
    pub turbulence: f32,
    pub kind: WaterKind,
}

impl WaterHere {
    pub fn depth(&self) -> f32 {
        self.surface - self.ground
    }
}

/// The water over the ground at `(x, z)`, if any: a river's (a little splash
/// past its waterline counts), a lake's still water, or the sea's; where two
/// overlap, the higher surface is the one standing there.
pub fn water_at(erosion: &ErosionCache, noise: &NoiseField, x: f32, z: f32, visibility_center: [f32; 2]) -> Option<WaterHere> {
    let envelope = erosion
        .rivers
        .as_ref()
        .map_or(crate::rivers::carve::Envelope::NONE, |network| network.envelope(x, z));
    let ground = crate::erosion::sample_eroded_height(erosion, noise, x, z, visibility_center);
    let river = (envelope.bank_distance < 0.5).then(|| WaterHere {
        surface: envelope.water,
        ground,
        current: Vec2::from(envelope.velocity),
        turbulence: envelope.turbulence,
        kind: WaterKind::River,
    });
    let lake = (ground < envelope.lake).then_some(WaterHere {
        surface: envelope.lake,
        ground,
        current: Vec2::ZERO,
        turbulence: 0.0,
        kind: WaterKind::Lake,
    });
    let sea = (ground < SEA_LEVEL).then_some(WaterHere {
        surface: SEA_LEVEL,
        ground,
        current: Vec2::ZERO,
        turbulence: 0.0,
        kind: WaterKind::Sea,
    });
    [river, lake, sea]
        .into_iter()
        .flatten()
        .filter(|water| water.depth() > 0.0)
        .reduce(|a, b| if b.surface > a.surface { b } else { a })
}

impl Default for Player {
    fn default() -> Self {
        Self {
            position: Vec3::new(0.0, 20.0, 0.0),
            yaw: 0.0,
            // A shallow downward glance at spawn keeps the horizon visible.
            pitch: -0.18,
            vertical_velocity: 0.0,
            flying: false,
            movement_speed_multiplier: 1.0,
            mouse_captured: true,
            mouse_warmup_frames: 3,
            velocity: Vec2::ZERO,
            wading_depth: 0.0,
            current: Vec2::ZERO,
            swept: false,
            water: None,
        }
    }
}

impl Player {
    /// Unit direction of the player's view, shared by the camera and debug UI.
    pub fn forward(&self) -> Vec3 {
        Vec3::new(
            self.yaw.sin() * self.pitch.cos(),
            self.pitch.sin(),
            -self.yaw.cos() * self.pitch.cos(),
        )
    }

    /// Move to finite world coordinates and discard momentum from the old pose.
    pub fn teleport(&mut self, position: Vec3) -> bool {
        if !position.is_finite() {
            return false;
        }
        self.position = position;
        self.vertical_velocity = 0.0;
        self.velocity = Vec2::ZERO;
        true
    }

    /// Keep cursor visibility, grab state and mouse-motion warmup consistent.
    pub fn set_mouse_capture(&mut self, captured: bool, cursor: &mut CursorOptions) {
        self.mouse_captured = captured;
        if captured {
            self.mouse_warmup_frames = 3;
            cursor.grab_mode = CursorGrabMode::Locked;
        } else {
            cursor.grab_mode = CursorGrabMode::None;
        }
        cursor.visible = !captured;
    }
}

pub struct PlayerCamera {
    pub position: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub fov_y: f32,
}

impl PlayerCamera {
    pub fn from_player(player: &Player) -> Self {
        Self {
            position: player.position,
            target: player.position + player.forward(),
            up: Vec3::Y,
            fov_y: 68.0,
        }
    }
}

/// Per-frame player update, mirroring `UpdatePlayer`. The open developer menu
/// suppresses gameplay input while the passive debug text leaves it active.
#[allow(clippy::too_many_arguments)]
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
    ui_wants_input: Res<UiWantsInput>,
    automation: Option<Res<crate::automation::AutomationSettings>>,
) {
    let Ok(mut player) = player.single_mut() else {
        return;
    };
    snow.recenter(Vec2::new(player.position.x, player.position.z));
    // The water at the eye, which the renderer needs whether or not the
    // player can move.
    let visibility_center = Vec2::new(player.position.x, player.position.z);
    let water = water_at(&erosion, &noise, player.position.x, player.position.z, visibility_center.to_array());
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
        player.flying = !player.flying;
        player.vertical_velocity = 0.0;
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
    } else {
        // The water over the player's feet: a river's, a lake's or the
        // sea's. Flowing water moves the body at the velocity the current's
        // drag and the player's footing settle on, not at the walking speed.
        let (surface, depth, current, turbulence) = match water {
            Some(water) => (water.surface, water.depth(), water.current, water.turbulence),
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
        // Water too deep to stand in is swum in: the body floats, head out,
        // until the player dives.
        let floating = depth > EYE_HEIGHT - SWIM_FREEBOARD;
        let swimming = floating && player.position.y <= surface + SWIM_FREEBOARD + 0.03;
        let jump = keys.just_pressed(KeyCode::Space) && !ui_wants_input;
        let at_surface = player.position.y >= surface + SWIM_FREEBOARD - 0.05;
        if swimming && !(jump && at_surface) {
            let stroke = SwimStroke {
                rise: keys.pressed(KeyCode::Space) && !ui_wants_input,
                dive: keys.pressed(KeyCode::ShiftLeft) && !ui_wants_input,
                // Swimming forward follows the eye up or down, past a glance
                // either way, so the usual slightly lowered eye keeps to the
                // surface.
                forward_pitch: movement.dot(Vec3::new(player.yaw.sin(), 0.0, -player.yaw.cos()))
                    * (player.pitch.sin().abs() - SWIM_PITCH_DEAD_ZONE).max(0.0).copysign(player.pitch)
                    / (1.0 - SWIM_PITCH_DEAD_ZONE),
            };
            let candidate_xz = Vec2::new(candidate.x, candidate.z);
            let bed = crate::erosion::sample_eroded_height(&erosion, &noise, candidate_xz.x, candidate_xz.y,
                                                           visibility_center.to_array());
            let (y, vertical_velocity) = swim_vertical(player.position.y, player.vertical_velocity, stroke,
                                                       surface, bed, dt);
            candidate.y = y;
            player.vertical_velocity = vertical_velocity;
        } else {
            // A swimmer at the surface kicks up out of the water as a walker
            // jumps off the ground.
            if swimming && jump {
                player.vertical_velocity = 7.4;
            }
            update_walking_ground(
                &mut player,
                &mut candidate,
                &erosion,
                &noise,
                &mut snow,
                jump,
                dt,
            );
            // Falling into deep water ends at its surface, not the bed: the
            // water takes the fall and the swim takes over from there.
            if floating && candidate.y < surface + SWIM_FREEBOARD && player.vertical_velocity < 0.0 {
                player.vertical_velocity *= 0.25;
            }
        }
    }
    player.position = candidate;
    snow.recenter(Vec2::new(candidate.x, candidate.z));
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
    let trunks: Vec<collision::Trunk> = solids
        .iter()
        .filter_map(|solid| solid.stand(erosion, noise, from))
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

/// Resolve against the puffed surface first, then compact only the contact
/// segment and seat the player's feet on the newly flattened snow.
#[allow(clippy::too_many_arguments)]
fn update_walking_ground(
    player: &mut Player,
    candidate: &mut Vec3,
    erosion: &ErosionCache,
    noise: &NoiseField,
    snow: &mut SnowState,
    jump_requested: bool,
    dt: f32,
) {
    let previous_xz = Vec2::new(player.position.x, player.position.z);
    let visibility_center = previous_xz.to_array();
    let old_surface = snow::sample_surface(erosion, noise, previous_xz, visibility_center);
    let old_ground = old_surface.height(snow, previous_xz);
    let mut next_xz = Vec2::new(candidate.x, candidate.z);
    let mut next_surface = snow::sample_surface(erosion, noise, next_xz, visibility_center);
    if next_surface.height(snow, next_xz) - old_ground > 1.25 {
        candidate.x = player.position.x;
        candidate.z = player.position.z;
        next_xz = previous_xz;
        next_surface = old_surface;
    }
    let grounded_before =
        player.position.y <= old_ground + EYE_HEIGHT + 0.03 && player.vertical_velocity <= 0.0;
    let jumping = jump_requested && grounded_before;
    if jumping {
        player.vertical_velocity = 7.4;
    }
    player.vertical_velocity -= 23.0 * dt;
    candidate.y += player.vertical_velocity * dt;
    let ground_before_stamp = next_surface.height(snow, next_xz) + EYE_HEIGHT;
    let touching = candidate.y <= ground_before_stamp && player.vertical_velocity <= 0.0;
    if touching {
        if !jumping && next_surface.coverage > 0.0 {
            if grounded_before && next_xz.distance_squared(previous_xz) > 0.000001 {
                // Both endpoints are actual contacts in this update. There
                // is no stored segment to bridge jumps or camera teleports.
                snow.stamp_segment(previous_xz, next_xz);
            } else if !grounded_before {
                snow.stamp_segment(next_xz, next_xz);
            }
        }
        candidate.y = next_surface.height(snow, next_xz) + EYE_HEIGHT;
        player.vertical_velocity = 0.0;
    }
}

/// How far the eye floats above the water when it is too deep to stand in.
const SWIM_FREEBOARD: f32 = 0.25;
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

/// The eye height and vertical velocity a swimmer comes to after one step:
/// the stroke sets a climb or a dive, the water's drag eases the body toward
/// it, a body left alone drifts up to float head out, and neither the
/// surface (the swimmer floats there) nor the bed stops being solid.
pub fn swim_vertical(eye: f32, vertical_velocity: f32, stroke: SwimStroke, surface: f32, bed: f32, dt: f32) -> (f32, f32) {
    let mut target = SWIM_VERTICAL_SPEED * (stroke.forward_pitch.clamp(-1.0, 1.0)
        + f32::from(u8::from(stroke.rise)) - f32::from(u8::from(stroke.dive)));
    if !stroke.rise && !stroke.dive && stroke.forward_pitch.abs() < 0.05 {
        target = SWIM_BUOYANT_RISE;
    }
    let target = target.clamp(-SWIM_VERTICAL_SPEED, SWIM_VERTICAL_SPEED);
    let mut velocity = vertical_velocity + (target - vertical_velocity) * (1.0 - (-SWIM_RESPONSE * dt).exp());
    let mut y = eye + velocity * dt;
    let top = surface + SWIM_FREEBOARD;
    let bottom = (bed + SWIM_BED_CLEARANCE).min(top);
    if y >= top {
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
    let gravity = crate::rivers::network::GRAVITY;
    let submerged = depth.clamp(0.0, EYE_HEIGHT + 0.1);
    let area = 0.30 * submerged.min(0.55) + 0.45 * (submerged - 0.55).max(0.0);
    let speed = current.length();
    let drag = 0.5 * WATER_DENSITY * DRAG_COEFFICIENT * area * speed * speed;
    let buoyancy = WATER_DENSITY * gravity * BODY_VOLUME * (submerged / (EYE_HEIGHT + 0.1)).powf(1.3);
    // Wet, rounded stones under whitewater grip less than a gravel bed.
    let grip = 0.6 - 0.2 * turbulence.clamp(0.0, 1.0);
    let footing = grip * (MASS * gravity - buoyancy).max(0.0);
    let floating = depth > EYE_HEIGHT - SWIM_FREEBOARD;
    let load = drag / footing.max(1.0);
    let swept = floating || load > 1.0;
    // How much of the current the body takes on.
    let carried = if floating { 1.0 } else { ((load - 0.3) / 1.2).clamp(0.0, 1.0) };
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
    let camera = PlayerCamera::from_player(&player);
    if let Ok((mut transform, mut projection)) = transforms.single_mut() {
        transform.translation = camera.position;
        transform.look_at(camera.target, Vec3::Y);
        if let Projection::Perspective(perspective) = &mut *projection {
            perspective.fov = camera.fov_y.to_radians();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn view_vector_is_unit_length_and_matches_camera_target() {
        let mut player = Player::default();
        player.yaw = std::f32::consts::FRAC_PI_2;
        player.pitch = 0.4;
        let forward = player.forward();
        assert!((forward.length() - 1.0).abs() < 0.000001);
        assert!((forward.x - player.pitch.cos()).abs() < 0.000001);
        assert!((forward.y - player.pitch.sin()).abs() < 0.000001);
        assert!(forward.z.abs() < 0.000001);
        let camera = PlayerCamera::from_player(&player);
        assert!(camera.target.abs_diff_eq(player.position + forward, 0.000001));
    }

    #[test]
    fn teleport_resets_momentum_and_rejects_invalid_coordinates() {
        let mut player = Player::default();
        player.vertical_velocity = -12.0;
        let destination = Vec3::new(120.0, 450.0, -870.0);
        assert!(player.teleport(destination));
        assert_eq!(player.position, destination);
        assert_eq!(player.vertical_velocity, 0.0);

        player.vertical_velocity = 4.0;
        for invalid in [
            Vec3::new(f32::NAN, 0.0, 0.0),
            Vec3::new(0.0, f32::INFINITY, 0.0),
            Vec3::new(0.0, 0.0, f32::NEG_INFINITY),
        ] {
            assert!(!player.teleport(invalid));
            assert_eq!(player.position, destination);
            assert_eq!(player.vertical_velocity, 4.0);
        }
    }

    fn snowy_world() -> (NoiseField, ErosionCache, SnowState, Player) {
        // A constant field gives flat mountain ground away from the spawn
        // clearing, making the contact behavior independent of terrain noise.
        let noise = NoiseField {
            samples: vec![0.70; NOISE_RESOLUTION * NOISE_RESOLUTION],
        };
        let erosion = ErosionCache::default();
        let mut snow = SnowState::default();
        let position = Vec2::new(1000.0, 1000.0);
        snow.recenter(position);
        let surface = snow::sample_surface(&erosion, &noise, position, position.to_array());
        assert_eq!(surface.coverage, 1.0);
        let mut player = Player::default();
        player.position = Vec3::new(
            position.x,
            surface.height(&snow, position) + EYE_HEIGHT,
            position.y,
        );
        (noise, erosion, snow, player)
    }

    #[test]
    fn grounded_walking_compacts_the_sweep_and_seats_feet_on_bare_terrain() {
        let (noise, erosion, mut snow, mut player) = snowy_world();
        let start = Vec2::new(player.position.x, player.position.z);
        let mut candidate = player.position + Vec3::new(0.55, 0.0, 0.19);
        update_walking_ground(
            &mut player,
            &mut candidate,
            &erosion,
            &noise,
            &mut snow,
            false,
            0.02,
        );
        let end = Vec2::new(candidate.x, candidate.z);
        assert_eq!(snow.compression(start.lerp(end, 0.5)), 1.0);
        let surface = snow::sample_surface(&erosion, &noise, end, start.to_array());
        assert!((candidate.y - (surface.base_height + EYE_HEIGHT)).abs() < 0.0001);
        assert_eq!(player.vertical_velocity, 0.0);
    }

    #[test]
    fn jump_and_airborne_motion_leave_no_trail() {
        let (noise, erosion, mut snow, mut player) = snowy_world();
        let revision = snow.revision();
        let mut candidate = player.position + Vec3::X * 0.5;
        update_walking_ground(
            &mut player,
            &mut candidate,
            &erosion,
            &noise,
            &mut snow,
            true,
            0.02,
        );
        assert!(player.vertical_velocity > 0.0);
        assert_eq!(snow.revision(), revision);
        player.position = candidate + Vec3::Y * 2.0;
        candidate = player.position + Vec3::X * 2.0;
        update_walking_ground(
            &mut player,
            &mut candidate,
            &erosion,
            &noise,
            &mut snow,
            false,
            0.02,
        );
        assert_eq!(snow.revision(), revision);
    }

    #[test]
    fn landing_compacts_only_the_contact_point() {
        let (noise, erosion, mut snow, mut player) = snowy_world();
        player.position.y += 0.5;
        player.vertical_velocity = -20.0;
        let old_xz = Vec2::new(player.position.x, player.position.z);
        let mut candidate = player.position + Vec3::X * 4.0;
        update_walking_ground(
            &mut player,
            &mut candidate,
            &erosion,
            &noise,
            &mut snow,
            false,
            0.05,
        );
        let end = Vec2::new(candidate.x, candidate.z);
        assert_eq!(snow.compression(end), 1.0);
        assert_eq!(snow.compression(old_xz.lerp(end, 0.5)), 0.0);
    }

    /// A flat mountain world with one 8 m oak, its trunk 0.4 m in radius, 3 m
    /// east of the player.
    fn forest_of_one(field: f32) -> (NoiseField, ErosionCache, VegetationField, Player) {
        let noise = NoiseField {
            samples: vec![field; NOISE_RESOLUTION * NOISE_RESOLUTION],
        };
        let erosion = ErosionCache::default();
        let tree = crate::vegetation::scatter::PlantInstance {
            position: [1003.0, 0.0, 1000.0],
            scale: 1.0,
            model: 0,
            layer: crate::vegetation::scatter::Layer::Canopy as u32,
            ..Default::default()
        };
        let vegetation = VegetationField::with_plants(crate::vegetation::tests::catalog(), vec![tree]);
        let mut player = Player::default();
        let ground = crate::erosion::sample_eroded_height(&erosion, &noise, 1000.0, 1000.0, [1000.0, 1000.0]);
        player.position = Vec3::new(1000.0, ground + EYE_HEIGHT, 1000.0);
        (noise, erosion, vegetation, player)
    }

    #[test]
    fn the_player_walks_up_to_a_trunk_and_no_further() {
        let (noise, erosion, vegetation, mut player) = forest_of_one(0.70);
        for _ in 0..200 {
            player.velocity = Vec2::new(10.0, 0.0);
            let mut candidate = player.position + Vec3::X * (10.0 * 0.016);
            collide_with_trunks(&mut player, &mut candidate, &vegetation, &erosion, &noise);
            player.position = candidate;
        }
        let stopped = 1003.0 - 0.4 - PLAYER_RADIUS;
        assert!((player.position.x - stopped).abs() < 0.01, "{}", player.position);
        assert!(player.velocity.x.abs() < 1e-4, "{:?}", player.velocity);
    }

    #[test]
    fn the_player_slides_along_a_trunk_it_meets_at_an_angle() {
        let (noise, erosion, vegetation, mut player) = forest_of_one(0.70);
        player.position.z += 0.3;
        for _ in 0..200 {
            let mut candidate = player.position + Vec3::X * (10.0 * 0.016);
            collide_with_trunks(&mut player, &mut candidate, &vegetation, &erosion, &noise);
            player.position = candidate;
        }
        // Past the tree, and round the side it was off-axis toward.
        assert!(player.position.x > 1003.5, "{}", player.position);
        assert!(player.position.z > 1000.3, "{}", player.position);
    }

    #[test]
    fn a_tree_the_gpu_would_not_draw_is_not_solid() {
        // The same tree on ground under the sea: the cull refuses its root.
        let (noise, erosion, vegetation, mut player) = forest_of_one(0.0);
        let mut candidate = player.position + Vec3::X * 6.0;
        collide_with_trunks(&mut player, &mut candidate, &vegetation, &erosion, &noise);
        assert_eq!(candidate.x, 1006.0);
    }

    #[test]
    fn a_jump_does_not_clear_a_tall_tree() {
        let (noise, erosion, vegetation, mut player) = forest_of_one(0.70);
        // The oak is 8 m tall: no jump clears it.
        player.position.y += 1.2;
        let mut candidate = player.position + Vec3::X * 6.0;
        collide_with_trunks(&mut player, &mut candidate, &vegetation, &erosion, &noise);
        assert!(candidate.x < 1003.0, "{}", candidate);
    }

    /// Run the wading model until it settles.
    fn settle(walk: Vec2, current: Vec2, depth: f32) -> Wading {
        let mut state = Wading { velocity: Vec2::ZERO, swept: false };
        for _ in 0..600 {
            state = wade(state.velocity, walk, current, depth, 0.0, 1.0 / 120.0);
        }
        state
    }

    #[test]
    fn ankle_deep_water_barely_slows_a_walk() {
        let state = settle(Vec2::new(10.0, 0.0), Vec2::new(0.0, 0.8), 0.12);
        assert!(state.velocity.x > 7.0, "{:?}", state.velocity);
        assert!(state.velocity.y.abs() < 0.05, "{:?}", state.velocity);
        assert!(!state.swept);
    }

    #[test]
    fn standing_in_a_gentle_current_holds_but_drifts_nothing() {
        let state = settle(Vec2::ZERO, Vec2::new(1.0, 0.0), 0.5);
        assert!(state.velocity.length() < 0.01, "{:?}", state.velocity);
        assert!(!state.swept);
    }

    #[test]
    fn wading_upstream_is_slower_than_downstream() {
        let current = Vec2::new(1.4, 0.0);
        let upstream = settle(Vec2::new(-4.0, 0.0), current, 0.7);
        let downstream = settle(Vec2::new(4.0, 0.0), current, 0.7);
        assert!(-upstream.velocity.x < downstream.velocity.x, "{:?} {:?}", upstream.velocity, downstream.velocity);
        assert!(-upstream.velocity.x > 0.5, "the current must not stop a wade upstream");
    }

    #[test]
    fn a_strong_waist_deep_current_sweeps_the_player_away() {
        let state = settle(Vec2::ZERO, Vec2::new(2.5, 0.0), 1.1);
        assert!(state.swept);
        assert!(state.velocity.x > 1.0, "{:?}", state.velocity);
    }

    #[test]
    fn deep_water_carries_a_swimmer_with_the_current() {
        let state = settle(Vec2::ZERO, Vec2::new(1.0, 0.5), 2.5);
        assert!((state.velocity - Vec2::new(1.0, 0.5)).length() < 0.1, "{:?}", state.velocity);
    }

    /// Swim with one stroke for `seconds`, from an eye height and vertical
    /// velocity, over a surface at 10 m and a bed at 4 m.
    fn swim(mut eye: f32, mut velocity: f32, stroke: SwimStroke, seconds: f32) -> (f32, f32) {
        let dt = 1.0 / 120.0;
        for _ in 0..(seconds / dt) as usize {
            (eye, velocity) = swim_vertical(eye, velocity, stroke, 10.0, 4.0, dt);
        }
        (eye, velocity)
    }

    #[test]
    fn a_swimmer_dives_under_and_floats_back_up() {
        let floating = 10.0 + SWIM_FREEBOARD;
        let dive = SwimStroke { dive: true, ..Default::default() };
        let (deep, _) = swim(floating, 0.0, dive, 2.0);
        assert!(deep < 9.0, "a dive takes the eye under: {deep}");
        // The bed stops the dive.
        let (bottom, velocity) = swim(floating, 0.0, dive, 10.0);
        assert!((bottom - (4.0 + SWIM_BED_CLEARANCE)).abs() < 1e-4 && velocity == 0.0, "{bottom} {velocity}");
        // Left alone the body drifts back up and floats, head out.
        let (rested, _) = swim(bottom, 0.0, SwimStroke::default(), 30.0);
        assert!((rested - floating).abs() < 1e-4, "{rested}");
        // Swimming up is quicker than drifting.
        let rise = SwimStroke { rise: true, ..Default::default() };
        let (risen, _) = swim(bottom, 0.0, rise, 2.0);
        let (drifted, _) = swim(bottom, 0.0, SwimStroke::default(), 2.0);
        assert!(risen > drifted + 1.0, "{risen} {drifted}");
    }

    #[test]
    fn swimming_forward_follows_the_eye_down() {
        let floating = 10.0 + SWIM_FREEBOARD;
        let looking_down = SwimStroke { forward_pitch: -0.7, ..Default::default() };
        let (eye, velocity) = swim(floating, 0.0, looking_down, 1.5);
        assert!(eye < floating - 0.8 && velocity < -0.5, "{eye} {velocity}");
        // Looking level, swimming forward keeps to the surface.
        let level = SwimStroke { forward_pitch: 0.0, ..Default::default() };
        let (eye, _) = swim(floating, 0.0, level, 1.5);
        assert!((eye - floating).abs() < 1e-4, "{eye}");
    }

    #[test]
    fn water_kinds_meet_at_the_higher_surface() {
        // Flat ground at the constant field's height with no rivers: no
        // water over it unless it lies under the sea.
        let noise = NoiseField { samples: vec![0.70; NOISE_RESOLUTION * NOISE_RESOLUTION] };
        let erosion = ErosionCache::default();
        assert!(water_at(&erosion, &noise, 1000.0, 1000.0, [1000.0, 1000.0]).is_none());
        let sea_floor = NoiseField { samples: vec![0.0; NOISE_RESOLUTION * NOISE_RESOLUTION] };
        let ground = crate::erosion::sample_eroded_height(&erosion, &sea_floor, 1000.0, 1000.0, [1000.0, 1000.0]);
        if ground < SEA_LEVEL {
            let water = water_at(&erosion, &sea_floor, 1000.0, 1000.0, [1000.0, 1000.0]).unwrap();
            assert_eq!(water.kind, WaterKind::Sea);
            assert_eq!(water.surface, SEA_LEVEL);
            assert!(water.depth() > 0.0);
        }
    }
}
