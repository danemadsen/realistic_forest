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
            mouse_captured: true,
            mouse_warmup_frames: 3,
            velocity: Vec2::ZERO,
            wading_depth: 0.0,
            current: Vec2::ZERO,
            swept: false,
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
        let here = erosion::sample_eroded_height(
            &erosion, &noise, player.position.x, player.position.z,
            visibility_center.to_array());
        let depth = if envelope.bank_distance < 0.5 { envelope.water - here } else { 0.0 };
        if depth > 0.02 {
            let current = Vec2::from(envelope.velocity);
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
        let mut ground = erosion::sample_eroded_height(
            &erosion, &noise, candidate.x, candidate.z,
            visibility_center.to_array()) + EYE_HEIGHT;
        // Water too deep to stand in floats the player, head out.
        if depth > EYE_HEIGHT - SWIM_FREEBOARD {
            ground = ground.max(envelope.water + SWIM_FREEBOARD);
        }
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
