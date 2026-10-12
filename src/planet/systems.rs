//! The planet's per-frame work in the main world: carrying the player from
//! one cube face's chart to the next, and placing the camera on the planet
//! and selecting the terrain patches it sees.

use super::flight::{self, FreeFlight};
use super::quadtree::{LodLadder, TERRAIN_HEIGHT_CEILING};
use super::view::{PlanetTerrain, PlanetView};
use super::{Planet, PlanetPosition};
use crate::constants::{EYE_HEIGHT, FAR_PLANE, NEAR_PLANE};
use crate::erosion::ErosionCache;
use crate::noise::NoiseField;
use crate::player::{Player, UiWantsInput};
use crate::snow::{self, SnowState};
use bevy::math::{DVec2, DVec3};
#[cfg(test)]
use std::f32::consts::FRAC_PI_2;
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

/// Altitude below which a free-flying body looks for the ground and the
/// water under it: above the highest the terrain reaches, with the snow and
/// the streamed features on it.
const GROUND_CHECK_ALTITUDE: f64 = TERRAIN_HEIGHT_CEILING + 400.0;

/// Start free flight from wherever the player is, keeping the velocity it
/// walks, swims or falls at. The noclip flight's speed is not momentum
/// ([`Player::set_flying`]), so a flier starts at rest. A body on its feet
/// starts landed, on the floor it stands on, until its thrust lifts it.
pub fn enter_free_flight(player: &mut Player, planet: &Planet) {
    let (position, frame) = flight::planet_position(planet, player.face, player.position);
    let local = if player.flying {
        DVec3::ZERO
    } else {
        DVec3::new(player.velocity.x as f64, player.vertical_velocity as f64, player.velocity.y as f64)
    };
    let mut free = FreeFlight::new(position, frame.to_planet(local));
    free.landed = !player.flying && !player.swimming && player.vertical_velocity == 0.0;
    free.on_floor = free.landed;
    free.surface = player.ground_below as f64;
    if free.landed {
        free.velocity = DVec3::ZERO;
    }
    player.flying = false;
    player.free_flight = Some(free);
}

/// KSP's "set orbit": put the player in a circular orbit over where it is,
/// heading the way it faces, raised clear of the air if it is lower.
pub fn enter_circular_orbit(player: &mut Player, planet: &Planet) {
    let altitude = (player.position.y as f64).max(flight::CIRCULAR_ORBIT_MIN_ALTITUDE);
    let raised = Vec3::new(player.position.x, altitude as f32, player.position.z);
    if !player.teleport(raised) {
        return;
    }
    let (position, frame) = flight::planet_position(planet, player.face, raised);
    let heading = frame.to_planet(DVec3::new(player.yaw.sin() as f64, 0.0, -player.yaw.cos() as f64));
    let speed = (planet.gravitational_parameter() / position.length()).sqrt();
    player.flying = false;
    player.free_flight = Some(FreeFlight::new(position, heading * speed));
}

/// Free flight under the planet's gravity ([`flight`]): G starts and ends
/// it, W and S thrust along the view, A and D to the body's sides, Space and
/// Shift toward its head and feet (Q and E roll it,
/// [`crate::player::update_player_system`]), Left Ctrl boosts, and `,` and
/// `.` step the time warp down and up while coasting above the air. The body's state is written back into the
/// player's chart coordinates, which the face crossing that follows carries
/// on to the next face.
#[allow(clippy::too_many_arguments)]
pub fn free_flight_system(
    planet: Res<Planet>,
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    ui_wants_input: Res<UiWantsInput>,
    erosion: Res<ErosionCache>,
    noise: Res<NoiseField>,
    snow: Res<SnowState>,
    automation: Option<Res<crate::automation::AutomationSettings>>,
    mut players: Query<&mut Player>,
) {
    let Ok(mut player) = players.single_mut() else {
        return;
    };
    // A pinned --camera capture holds its pose.
    if automation.is_some_and(|automation| automation.has_camera && automation.shot_path.is_some()) {
        return;
    }
    let input = !ui_wants_input.0;
    if input && keys.just_pressed(KeyCode::KeyG) {
        if player.free_flight.is_some() {
            player.set_flying(false);
            return;
        }
        enter_free_flight(&mut player, &planet);
    }
    let Some(mut free) = player.free_flight else {
        return;
    };

    let here = flight::place_on(&planet, player.face, free.position);
    let mut thrust = Vec3::ZERO;
    if input {
        let axis = |plus: KeyCode, minus: KeyCode| (keys.pressed(plus) as i32 - keys.pressed(minus) as i32) as f32;
        // Along the body's own axes, as it is turned and rolled.
        thrust = player.forward() * axis(KeyCode::KeyW, KeyCode::KeyS)
            + player.right() * axis(KeyCode::KeyD, KeyCode::KeyA)
            + player.up() * axis(KeyCode::Space, KeyCode::ShiftLeft);
        if thrust.length_squared() > 1.0 {
            thrust = thrust.normalize();
        }
    }
    let boost = if input && keys.pressed(KeyCode::ControlLeft) { flight::THRUST_BOOST } else { 1.0 };
    let acceleration = flight::THRUST * boost * player.movement_speed_multiplier as f64;
    let thrust = here.frame.to_planet(thrust.as_dvec3() * acceleration);

    // The warp is for coasting above the air, as KSP's is: thrust or the air
    // ends it.
    if thrust != DVec3::ZERO || !free.may_warp(&planet) {
        free.warp = 0;
    } else if input && keys.just_pressed(KeyCode::Period) {
        free.warp = (free.warp + 1).min(flight::WARP_LEVELS.len() - 1);
    } else if input && keys.just_pressed(KeyCode::Comma) {
        free.warp = free.warp.saturating_sub(1);
    }

    let seconds = free.frame_seconds(time.delta_secs().min(0.05) as f64);
    let before = free.altitude(&planet);
    free.advance(&planet, thrust, seconds);

    let mut placed = flight::place_on(&planet, player.face, free.position);
    let mut water = None;
    if free.altitude(&planet) < GROUND_CHECK_ALTITUDE {
        let xz = Vec2::new(placed.position.x, placed.position.z);
        let ground = snow::sample_surface(&erosion, &noise, xz, xz.to_array()).height(&snow, xz);
        water = crate::player::water_at(&erosion, &noise, xz.x, xz.y, xz.to_array());
        free.surface = water.map_or(ground, |water| water.surface.max(ground)) as f64;
        // The water takes a body that comes down onto it from above; one
        // already under it (a diver who took off) sinks to the bed or rises.
        let float = water
            .and_then(|water| water.float_eye())
            .filter(|&eye| eye > ground + EYE_HEIGHT && before >= eye as f64 - 0.01);
        let touch = float.unwrap_or(ground + EYE_HEIGHT);
        let speed = free.velocity.length();
        if free.settle(&planet, touch as f64, float.is_some()) {
            info!("FLIGHT: touched down at {speed:.1} m/s");
        }
        placed = flight::place_on(&planet, player.face, free.position);
    } else {
        let (x, z) = (placed.position.x, placed.position.z);
        let ground = super::height::surface_height(&planet, &erosion, &noise, player.face, x, z, [x, z]);
        free.surface = ground.max(crate::constants::SEA_LEVEL) as f64;
    }
    player.position = placed.position;
    (player.velocity, player.vertical_velocity) = flight::local_velocity(placed.frame, free.velocity);
    player.water = water;
    player.flying = false;
    player.swimming = false;
    player.leapt = false;
    player.wading_depth = 0.0;
    player.current = Vec2::ZERO;
    player.swept = false;
    player.plane_carry = Vec2::ZERO;
    player.free_flight = Some(free);
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
        // Plane-axis offsets of the old chart: the next frame finds them anew.
        player.lean_offset = Vec2::ZERO;
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
    use super::super::Face;
    use crate::constants::NOISE_RESOLUTION;
    use std::time::Duration;

    /// An app running the free flight and the face crossing over a flat
    /// world (a constant noise field), with a hand-driven clock and keys.
    fn flight_app(player: Player) -> (App, Entity) {
        let mut app = App::new();
        app.insert_resource(Planet::default());
        app.insert_resource(Time::<()>::default());
        app.insert_resource(ButtonInput::<KeyCode>::default());
        app.insert_resource(UiWantsInput(false));
        app.insert_resource(ErosionCache::default());
        app.insert_resource(NoiseField { samples: vec![0.70; NOISE_RESOLUTION * NOISE_RESOLUTION] });
        app.insert_resource(SnowState::default());
        app.add_systems(Update, (free_flight_system, carry_player_across_faces).chain());
        let entity = app.world_mut().spawn(player).id();
        (app, entity)
    }

    /// One frame of `seconds`, with `key` pressed in it if any.
    fn frame(app: &mut App, seconds: f32, key: Option<KeyCode>) {
        app.world_mut().resource_mut::<Time>().advance_by(Duration::from_secs_f32(seconds));
        if let Some(key) = key {
            app.world_mut().resource_mut::<ButtonInput<KeyCode>>().press(key);
        }
        app.update();
        let mut keys = app.world_mut().resource_mut::<ButtonInput<KeyCode>>();
        keys.release_all();
        keys.clear();
    }

    #[test]
    fn an_orbit_carries_the_player_across_a_face_edge_and_on_round_the_planet() {
        let planet = Planet::default();
        let s = planet.half_width as f32;
        // 100 km up, 30 km short of the spawn face's east edge, heading east.
        let mut player = Player { position: Vec3::new(s - 30_000.0, 100_000.0, 2_000.0), yaw: FRAC_PI_2, ..Player::default() };
        enter_circular_orbit(&mut player, &planet);
        let start = player.free_flight.unwrap();
        let (mut app, entity) = flight_app(player);
        let mu = planet.gravitational_parameter();
        let energy = |flight: &FreeFlight| 0.5 * flight.velocity.length_squared() - mu / flight.position.length();
        let mut crossed = false;
        for _ in 0..1_200 {
            frame(&mut app, 1.0 / 60.0, None);
            let player = app.world().get::<Player>(entity).unwrap();
            crossed |= player.face != Face::PosY;
            let flight = player.free_flight.unwrap();
            // The written-back pose is the body's.
            let (position, _) = flight::planet_position(&planet, player.face, player.position);
            assert!(position.distance(flight.position) < 0.5, "{}", position.distance(flight.position));
            assert!((player.position.y - 100_000.0).abs() < 1.0, "{}", player.position.y);
            // The view still looks along the orbit.
            let along = player.velocity.normalize().dot(Vec2::new(player.yaw.sin(), -player.yaw.cos()));
            assert!(along > 0.999, "{along}");
        }
        assert!(crossed);
        let after = app.world().get::<Player>(entity).unwrap().free_flight.unwrap();
        assert!(((energy(&after) - energy(&start)) / energy(&start)).abs() < 1e-9);
        // 20 s at about 2.3 km/s.
        let travelled = after.position.distance(start.position);
        assert!((travelled - 20.0 * start.velocity.length()).abs() < 200.0, "{travelled}");
    }

    #[test]
    fn free_flight_thrusts_along_the_rolled_body() {
        let planet = Planet::default();
        // Rolled onto its right side a kilometre up, looking level north.
        let mut player = Player { position: Vec3::new(2_000.0, 1_000.0, -3_000.0), pitch: 0.0,
                                  roll: std::f32::consts::FRAC_PI_2, flying: true, ..Player::default() };
        enter_free_flight(&mut player, &planet);
        assert!(!player.free_flight.unwrap().landed);
        let (mut app, entity) = flight_app(player);
        for _ in 0..60 {
            frame(&mut app, 1.0 / 60.0, Some(KeyCode::Space));
        }
        let player = app.world().get::<Player>(entity).unwrap();
        // Space pushes toward the head, east; the planet still pulls down.
        assert!((player.velocity.x - flight::THRUST as f32).abs() < 0.5, "{}", player.velocity);
        assert!(player.velocity.y.abs() < 0.1, "{}", player.velocity);
        assert!((player.vertical_velocity + 9.8).abs() < 0.3, "{}", player.vertical_velocity);
    }

    #[test]
    fn g_lifts_off_into_free_flight_and_back_to_walking() {
        let mut player = Player { position: Vec3::new(1_000.0, 0.0, 1_000.0), ..Player::default() };
        let (mut app, entity) = flight_app(Player::default());
        // Stand on the flat ground.
        {
            let world = app.world();
            let erosion = world.resource::<ErosionCache>();
            let noise = world.resource::<NoiseField>();
            let snow = world.resource::<SnowState>();
            let xz = Vec2::new(player.position.x, player.position.z);
            player.position.y = snow::sample_surface(erosion, noise, xz, xz.to_array()).height(snow, xz) + EYE_HEIGHT;
        }
        let standing = player.position;
        *app.world_mut().get_mut::<Player>(entity).unwrap() = player;
        frame(&mut app, 1.0 / 60.0, Some(KeyCode::KeyG));
        let player = app.world().get::<Player>(entity).unwrap();
        let flight = player.free_flight.expect("G starts free flight");
        assert!(flight.landed);
        assert!(player.position.distance(standing) < 0.01);
        // Without thrust it stays on the ground.
        for _ in 0..60 {
            frame(&mut app, 1.0 / 60.0, None);
        }
        assert!(app.world().get::<Player>(entity).unwrap().position.distance(standing) < 0.01);
        // Space thrusts it up off the ground.
        for _ in 0..120 {
            frame(&mut app, 1.0 / 60.0, Some(KeyCode::Space));
        }
        let player = app.world().get::<Player>(entity).unwrap();
        assert!(!player.free_flight.unwrap().landed);
        assert!(player.position.y > standing.y + 2.0 && player.vertical_velocity > 0.0);
        // G again ends it, from rest.
        frame(&mut app, 1.0 / 60.0, Some(KeyCode::KeyG));
        let player = app.world().get::<Player>(entity).unwrap();
        assert!(player.free_flight.is_none() && !player.flying);
        assert_eq!(player.vertical_velocity, 0.0);
    }

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
