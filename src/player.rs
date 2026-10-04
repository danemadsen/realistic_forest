//! First-person player controller, ported from the C++ `Player` and
//! `UpdatePlayer` / `PlayerCamera`.

use crate::constants::*;
use crate::erosion::ErosionCache;
use crate::noise::NoiseField;
use crate::snow::{self, SnowState};
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
    ui_wants_input: Res<UiWantsInput>,
    automation: Option<Res<crate::automation::AutomationSettings>>,
) {
    let Ok(mut player) = player.single_mut() else {
        return;
    };
    snow.recenter(Vec2::new(player.position.x, player.position.z));
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
    } else {
        let visibility_center = Vec2::new(player.position.x, player.position.z);
        // Flowing water: the body moves at the velocity the current's drag
        // and the player's footing settle on, not at the walking speed.
        let envelope = erosion
            .rivers
            .as_ref()
            .map_or(crate::rivers::carve::Envelope::NONE, |network| {
                network.envelope(player.position.x, player.position.z)
            });
        let here = crate::erosion::sample_eroded_height(
            &erosion, &noise, player.position.x, player.position.z,
            visibility_center.to_array());
        // The water over the player's feet: a river's (a little splash past
        // its waterline counts) or a lake's still water.
        let river = (envelope.bank_distance < 0.5).then_some(envelope.water);
        let lake = (here < envelope.lake).then_some(envelope.lake);
        let (surface, flowing) = match (river, lake) {
            (Some(r), Some(l)) if l > r => (l, false),
            (Some(r), _) => (r, true),
            (None, Some(l)) => (l, false),
            (None, None) => (here, false),
        };
        let depth = surface - here;
        if depth > 0.02 {
            let current = if flowing { Vec2::from(envelope.velocity) } else { Vec2::ZERO };
            let wading = wade(player.velocity, walk, current, depth, envelope.turbulence, dt);
            player.velocity = wading.velocity;
            player.swept = wading.swept;
            player.wading_depth = depth;
            player.current = current;
        } else {
            player.velocity = walk;
        }
        candidate.x = player.position.x + player.velocity.x * dt;
        candidate.z = player.position.z + player.velocity.y * dt;
        update_walking_ground(
            &mut player,
            &mut candidate,
            &erosion,
            &noise,
            &mut snow,
            keys.just_pressed(KeyCode::Space) && !ui_wants_input,
            dt,
        );
        // Water too deep to stand in floats the player, head out.
        if depth > EYE_HEIGHT - SWIM_FREEBOARD && candidate.y < surface + SWIM_FREEBOARD {
            candidate.y = surface + SWIM_FREEBOARD;
            player.vertical_velocity = player.vertical_velocity.max(0.0);
        }
    }
    player.position = candidate;
    snow.recenter(Vec2::new(candidate.x, candidate.z));
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
}
