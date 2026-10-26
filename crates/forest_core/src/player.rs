//! Shared player state read by world streaming and rendering.

use crate::constants::{EYE_HEIGHT, SEA_LEVEL};
use bevy::prelude::{Component, Vec2, Vec3};
use bevy::window::{CursorGrabMode, CursorOptions};

/// How far the eye floats above the water when it is too deep to stand in.
pub const SWIM_FREEBOARD: f32 = 0.25;

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

    /// Whether the body swam its last step: only a swimmer crouches under
    /// standing height, to stand up out of the shallows. A walker is set on
    /// its feet.
    pub swimming: bool,

    /// Whether the body is in the air from a jump or a kick, rather than
    /// having walked or fallen off something: only a leap over deep water
    /// steps up onto a bank from its feet.
    pub leapt: bool,
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

    /// How clear it is, 0 (stained by the forest) to 1 (clear mountain
    /// water), how far it is the sea's water and how still it is, each 0..1,
    /// as the surface drawn over it has them (`SurfaceMesh::drawn_at`): the
    /// medium around a submerged eye is the water drawn overhead, which
    /// blends into the sea toward a mouth and into a lake's still water
    /// toward its sheet, and carries a clear lake's clarity down its outlet.
    pub clarity: f32,

    pub sea: f32,
    pub still: f32,

    /// Which water's surface is drawn over it: a river's mouth past its
    /// ribbon is the sea's, though its level and current are the river's.
    pub kind: WaterKind,
}

impl WaterHere {
    pub fn depth(&self) -> f32 {
        self.surface - self.ground
    }

    /// The sea over ground at `ground`, if it covers it.
    pub fn sea(ground: f32) -> Option<WaterHere> {
        (ground < SEA_LEVEL).then_some(WaterHere {
            surface: SEA_LEVEL,
            ground,
            current: Vec2::ZERO,
            turbulence: 0.0,
            clarity: 0.0,
            sea: 1.0,
            still: 1.0,
            kind: WaterKind::Sea,
        })
    }

    /// Where the eye of a body floating head out in this water is, if it is
    /// too deep to stand in.
    pub fn float_eye(&self) -> Option<f32> {
        (self.depth() > EYE_HEIGHT - SWIM_FREEBOARD).then_some(self.surface + SWIM_FREEBOARD)
    }
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
            swimming: false,
            leapt: false,
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

    /// Move to finite world coordinates and discard momentum from the old
    /// pose, and the water there: the renderer may draw the new pose before
    /// the next update looks at the water under it, and must not put the eye
    /// in the old pose's water meanwhile.
    pub fn teleport(&mut self, position: Vec3) -> bool {
        if !position.is_finite() {
            return false;
        }
        self.position = position;
        self.vertical_velocity = 0.0;
        self.velocity = Vec2::ZERO;
        self.swimming = false;
        self.leapt = false;
        self.water = None;
        self.wading_depth = 0.0;
        self.current = Vec2::ZERO;
        self.swept = false;
        true
    }

    /// Start or stop flying. Flight's speed is not momentum: a body that
    /// stops flying drops from rest, rather than gliding on at flight speed
    /// over water it has not yet reached. It is set down as it is: one
    /// crouched under standing height, low in the shallows say, stands up as
    /// a swimmer does rather than being lifted onto its feet in a frame.
    pub fn set_flying(&mut self, flying: bool) {
        self.flying = flying;
        self.vertical_velocity = 0.0;
        self.velocity = Vec2::ZERO;
        self.swimming = !flying;
        self.leapt = false;
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
