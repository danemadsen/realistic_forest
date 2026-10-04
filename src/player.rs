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
    pub mouse_captured: bool,
    pub mouse_warmup_frames: u32,
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
            mouse_captured: true,
            mouse_warmup_frames: 3,
        }
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
        let forward = Vec3::new(
            player.yaw.sin() * player.pitch.cos(),
            player.pitch.sin(),
            -player.yaw.cos() * player.pitch.cos(),
        );
        Self {
            position: player.position,
            target: player.position + forward,
            up: Vec3::Y,
            fov_y: 68.0,
        }
    }
}

/// Per-frame player update, mirroring `UpdatePlayer`. The diagnostics window
/// suppresses gameplay input whenever the pointer or keyboard focus is on it.
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
    let speed = if player.flying { 38.0 } else { 10.0 } * if boosted { 2.5 } else { 1.0 };
    let mut candidate = player.position + movement * (speed * dt);

    if player.flying {
        let vertical = if !ui_wants_input {
            (keys.pressed(KeyCode::Space) as i32 - keys.pressed(KeyCode::ShiftLeft) as i32) as f32
        } else {
            0.0
        };
        candidate.y += vertical * speed * dt;
    } else {
        update_walking_ground(
            &mut player,
            &mut candidate,
            &erosion,
            &noise,
            &mut snow,
            keys.just_pressed(KeyCode::Space) && !ui_wants_input,
            dt,
        );
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
}
