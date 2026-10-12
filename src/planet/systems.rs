//! The planet's per-frame work in the main world: carrying the player from
//! one cube face's chart to the next, and placing the camera on the planet
//! and selecting the terrain patches it sees.

use super::quadtree::LodLadder;
use super::view::{PlanetTerrain, PlanetView};
use super::{Planet, PlanetPosition};
use crate::constants::{FAR_PLANE, NEAR_PLANE};
use crate::erosion::ErosionCache;
use crate::noise::NoiseField;
use crate::player::Player;
use bevy::math::DVec2;
use bevy::prelude::*;
use bevy::window::PrimaryWindow;

/// The terrain's LOD ladder, derived once from the planet.
#[derive(Resource, Clone, Debug)]
pub struct PlanetLadder(pub LodLadder);

/// Height above the ground below which flight keeps the planar build's
/// speeds; above it they grow in proportion, so the same keys that drift
/// over a meadow cross a continent from orbit.
pub const FLIGHT_SPEED_HEIGHT: f32 = 250.0;

/// Height above the ground over which the streams keyed on the player's
/// terrain plane position (erosion, rivers, buildings, plants) hold what they
/// have instead of following: from there their detail is too small to see,
/// and a camera crossing the planet would otherwise regenerate them every few
/// seconds.
pub const STREAM_PAUSE_HEIGHT: f32 = 20_000.0;

/// Whether the plane-keyed streams hold still for this player.
pub fn streams_paused(player: &Player) -> bool {
    player.position.y - player.ground_below > STREAM_PAUSE_HEIGHT
}

/// How much faster than at ground level a body flies `height` metres above
/// the ground under it.
pub fn flight_speed_scale(height: f32) -> f32 {
    (height / FLIGHT_SPEED_HEIGHT).max(1.0)
}

/// Move a player who has walked or flown past the edge of its face onto the
/// face it is now over. The point on the sphere, the heading, and every
/// horizontal vector are kept; only the chart they are written in changes,
/// which every plane-keyed stream (erosion, rivers, plants, buildings, snow)
/// sees as a move to new plane coordinates. Their features have faded out
/// before the seam band, so nothing on the ground changes as they re-stream.
/// The weather, keyed on plane coordinates too, is moved by as much, so the
/// sky and the rain stay where they were over the player (its field turns
/// about the player with the change of chart's axes).
pub fn carry_player_across_faces(
    planet: Res<Planet>,
    mut players: Query<&mut Player>,
    mut erosion: Option<ResMut<ErosionCache>>,
    noise: Option<Res<NoiseField>>,
    mut weather: Option<ResMut<crate::weather::WeatherMotion>>,
) {
    let Ok(mut player) = players.single_mut() else {
        return;
    };
    let before = Vec2::new(player.position.x, player.position.z);
    let position = PlanetPosition {
        face: player.face,
        plane: DVec2::new(player.position.x as f64, player.position.z as f64),
        altitude: player.position.y as f64,
    };
    // A teleport far past the edge can need a second step (past a cube
    // corner); a walk or a flight needs one.
    let mut position = position;
    let mut moved = false;
    for _ in 0..3 {
        let Some(transfer) = planet.transfer(position) else {
            break;
        };
        player.face = transfer.to.face;
        player.position.x = transfer.to.plane.x as f32;
        player.position.z = transfer.to.plane.y as f32;
        player.yaw += transfer.yaw_change as f32;
        player.velocity = transfer.rotate(player.velocity);
        player.current = transfer.rotate(player.current);
        // The lean's offset turns with the axes, close enough at a seam; the
        // rounding carry is the old chart's alone.
        player.lean_offset = transfer.rotate(player.lean_offset);
        player.plane_carry = Vec2::ZERO;
        position = transfer.to;
        moved = true;
    }
    if !moved {
        return;
    }
    if let Some(weather) = weather.as_deref_mut() {
        let shift = Vec2::new(player.position.x, player.position.z) - before;
        weather.offset = [weather.offset[0] + shift.x, weather.offset[1] + shift.y];
    }
    // The water under the old chart's coordinates is not this chart's: look
    // again on the new one, which the height queries answer for from now.
    player.water = None;
    if let (Some(erosion), Some(noise)) = (erosion.as_deref_mut(), noise.as_deref()) {
        erosion.planet = Some(super::height::PlanetGround { planet: *planet, face: player.face });
        let (x, z) = (player.position.x, player.position.z);
        player.water = crate::player::water_at(erosion, noise, x, z, [x, z]);
    }
    info!("Crossed onto the {:?} face at plane ({:.1}, {:.1})", player.face, player.position.x, player.position.z);
}

/// Place the camera on the planet and select the terrain patches for this
/// frame. Runs last in the frame, after the UI's teleports, so the patches
/// match the pose the renderer extracts.
#[allow(clippy::too_many_arguments)]
pub fn update_planet_view(
    planet: Res<Planet>,
    ladder: Res<PlanetLadder>,
    mut players: Query<&mut Player>,
    mut erosion: ResMut<ErosionCache>,
    noise: Res<NoiseField>,
    windows: Query<&Window, With<PrimaryWindow>>,
    offscreen: Option<Res<crate::OffscreenCapture>>,
    mut view: ResMut<PlanetView>,
    mut terrain: ResMut<PlanetTerrain>,
) {
    let Ok(mut player) = players.single_mut() else {
        return;
    };
    // Every CPU height query from here on stands on the camera's face.
    let ground = super::height::PlanetGround { planet: *planet, face: player.face };
    if erosion.planet != Some(ground) {
        erosion.planet = Some(ground);
    }
    let eye = player.eye();
    let ground = super::height::surface_height(
        &planet, &erosion, &noise, player.face, eye.x, eye.z, [eye.x, eye.z],
    );
    player.ground_below = ground;
    player.gravity_scale = (planet.gravity_at(eye.y as f64) / super::PLANET_SURFACE_GRAVITY) as f32;
    *view = PlanetView::new(&planet, player.face, eye, ground as f64);

    let (width, height) = if let Ok(window) = windows.single() {
        (window.physical_width(), window.physical_height())
    } else if let Some(offscreen) = offscreen {
        (offscreen.size[0], offscreen.size[1])
    } else {
        (16, 9)
    };
    let aspect = width.max(1) as f64 / height.max(1) as f64;
    let far = view.far_plane(&planet, FAR_PLANE as f64);
    terrain.select(
        &planet,
        &ladder.0,
        &view,
        player.forward(),
        player.up(),
        crate::render::CAMERA_FOV_Y_DEGREES.to_radians() as f64,
        aspect,
        NEAR_PLANE as f64,
        far,
    );
    let face_noise = &*noise;
    let planet = *planet;
    terrain.bounds.measure(&ladder.0, |face, chart| {
        super::height::bare_height(&planet, face_noise, face, chart)
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Horizontal plane vector of a heading.
    fn heading(yaw: f32) -> Vec2 {
        Vec2::new(yaw.sin(), -yaw.cos())
    }

    #[test]
    fn flight_keeps_its_speed_near_the_ground() {
        assert_eq!(flight_speed_scale(0.0), 1.0);
        assert_eq!(flight_speed_scale(FLIGHT_SPEED_HEIGHT), 1.0);
        assert_eq!(flight_speed_scale(400_000.0), 1_600.0);
    }

    #[test]
    fn crossing_a_face_keeps_the_heading_on_the_sphere() {
        let planet = Planet::default();
        let s = planet.half_width as f32;
        let mut app = App::new();
        app.insert_resource(planet);
        app.add_systems(Update, carry_player_across_faces);
        let player = Player {
            position: Vec3::new(s + 2.0, 50.0, 1000.0),
            yaw: 0.4,
            velocity: heading(0.4) * 10.0,
            ..Player::default()
        };
        let entity = app.world_mut().spawn(player).id();
        app.update();
        let after = app.world().get::<Player>(entity).unwrap();
        assert_ne!(after.face, super::super::Face::PosY);
        let chart = Planet::chart_of_plane(after.face, DVec2::new(after.position.x as f64, after.position.z as f64));
        assert!(Planet::default().edge_distance(chart) > -1e-2, "{chart}");
        assert_eq!(after.position.y, 50.0);
        // Velocity still points along the heading.
        let along = after.velocity.normalize().dot(heading(after.yaw));
        assert!(along > 0.9999, "{along}");
    }
}
