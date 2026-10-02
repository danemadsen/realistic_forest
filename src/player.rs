//! First-person player controller, ported from the C++ `Player` and
//! `UpdatePlayer` / `PlayerCamera`.

use crate::constants::*;
use crate::erosion::{self, ErosionCache};
use crate::noise::NoiseField;
use bevy::camera::Projection;
use bevy::input::{keyboard::KeyCode, mouse::MouseButton, ButtonInput};
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
    ui_wants_input: Res<UiWantsInput>,
    automation: Option<Res<crate::automation::AutomationSettings>>,
) {
    // A pinned --camera capture holds its pose: a click or keystroke that
    // lands on the window while it renders must not move the shot.
    if automation.is_some_and(|automation| automation.has_camera && automation.shot_path.is_some()) {
        mouse_motion.clear();
        return;
    }
    let Ok(mut player) = player.single_mut() else {
        return;
    };
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
        let delta = mouse_motion.read().map(|m| m.delta).fold(Vec2::ZERO, |a, b| a + b);
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
        let visibility_center = Vec2::new(player.position.x, player.position.z);
        let old_ground = erosion::sample_eroded_height(
            &erosion, &noise, player.position.x, player.position.z,
            visibility_center.to_array());
        let new_ground = erosion::sample_eroded_height(
            &erosion, &noise, candidate.x, candidate.z,
            visibility_center.to_array());
        if new_ground - old_ground > 1.25 {
            candidate.x = player.position.x;
            candidate.z = player.position.z;
        }
        let ground = erosion::sample_eroded_height(
            &erosion, &noise, candidate.x, candidate.z,
            visibility_center.to_array()) + EYE_HEIGHT;
        if keys.just_pressed(KeyCode::Space)
            && player.position.y <= ground + 0.03
            && !ui_wants_input
        {
            player.vertical_velocity = 7.4;
        }
        player.vertical_velocity -= 23.0 * dt;
        candidate.y += player.vertical_velocity * dt;
        if candidate.y < ground {
            candidate.y = ground;
            player.vertical_velocity = 0.0;
        }
    }
    player.position = candidate;
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