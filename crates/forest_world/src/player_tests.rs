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
    assert!(
        camera
            .target
            .abs_diff_eq(player.position + forward, 0.000001)
    );
}

#[test]
fn teleport_resets_momentum_and_rejects_invalid_coordinates() {
    let mut player = Player::default();
    player.vertical_velocity = -12.0;
    player.water = WaterHere::sea(-5.0);
    let destination = Vec3::new(120.0, 450.0, -870.0);
    assert!(player.teleport(destination));
    assert_eq!(player.position, destination);
    assert_eq!(player.vertical_velocity, 0.0);
    assert_eq!(
        player.water, None,
        "the old pose's water is not the new pose's"
    );
    assert_eq!(
        (player.wading_depth, player.current, player.swept),
        (0.0, Vec2::ZERO, false)
    );

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
        &[],
        false,
        None,
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
        &[],
        true,
        None,
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
        &[],
        false,
        None,
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
        &[],
        false,
        None,
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
    let tree = forest_vegetation::scatter::PlantInstance {
        position: [1003.0, 0.0, 1000.0],
        scale: 1.0,
        model: 0,
        layer: forest_vegetation::scatter::Layer::Canopy as u32,
        ..Default::default()
    };
    let vegetation =
        VegetationField::with_plants(forest_vegetation::test_support::catalog(), vec![tree]);
    let mut player = Player::default();
    let ground =
        forest_erosion::sample_eroded_height(&erosion, &noise, 1000.0, 1000.0, [1000.0, 1000.0]);
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
    assert!(
        (player.position.x - stopped).abs() < 0.01,
        "{}",
        player.position
    );
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
    let mut state = Wading {
        velocity: Vec2::ZERO,
        swept: false,
    };
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
    assert!(
        -upstream.velocity.x < downstream.velocity.x,
        "{:?} {:?}",
        upstream.velocity,
        downstream.velocity
    );
    assert!(
        -upstream.velocity.x > 0.5,
        "the current must not stop a wade upstream"
    );
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
    assert!(
        (state.velocity - Vec2::new(1.0, 0.5)).length() < 0.1,
        "{:?}",
        state.velocity
    );
}

/// Swim with one stroke for `seconds`, from an eye height and vertical
/// velocity, over a surface at 10 m and a bed at 4 m.
fn swim(mut eye: f32, mut velocity: f32, stroke: SwimStroke, seconds: f32) -> (f32, f32) {
    let dt = 1.0 / 120.0;
    for _ in 0..(seconds / dt) as usize {
        (eye, velocity) = swim_vertical(eye, velocity, stroke, 10.0, 4.0, 4.0, dt);
    }
    (eye, velocity)
}

/// Still water at `surface` over ground at `ground`, where it covers it.
fn still_water(surface: f32, ground: f32) -> Option<WaterHere> {
    (surface - ground > 0.02).then_some(WaterHere {
        surface,
        ground,
        current: Vec2::ZERO,
        turbulence: 0.0,
        clarity: 0.0,
        sea: 0.0,
        still: 1.0,
        kind: WaterKind::Lake,
    })
}

/// A body's vertical state: eye height, vertical velocity, and whether
/// it swam its last step.
type Body = (f32, f32, bool);

/// One step of update_player_system's vertical motion from over ground
/// at `from` to over ground at `to`, under still water at `surface` (-inf
/// for none): the swim, or update_walking_ground's fall and seating on
/// bare ground and then the water the step ends over taking the fall.
fn step(
    (eye, velocity, swam): Body,
    stroke: SwimStroke,
    surface: f32,
    from: f32,
    to: f32,
    dt: f32,
) -> Body {
    let water = still_water(surface, from);
    if swims(water, from, eye, velocity, swam) {
        let here = water.map_or(f32::NEG_INFINITY, |water| water.surface);
        let (eye, velocity) = swim_vertical(eye, velocity, stroke, here, from, to, dt);
        return (eye, velocity, true);
    }
    let afloat = water
        .filter(|water| water.float_eye().is_some())
        .map(|water| water.surface);
    let swimmer = kicked(afloat, velocity, swam);
    let on_feet = eye <= from + EYE_HEIGHT + 0.03 && velocity <= 0.0;
    let before = (eye, velocity);
    let (eye, velocity) = fall(eye, velocity, on_feet, dt);
    let (eye, velocity, crouched) = if eye <= to + EYE_HEIGHT && velocity <= 0.0 {
        let (eye, crouched) = lands(eye, to, swimmer);
        (eye, 0.0, crouched)
    } else {
        (eye, velocity, swimmer)
    };
    let (eye, velocity) = still_water(surface, to)
        .filter(|water| water.float_eye().is_some())
        .and_then(|water| takes_the_fall(before, (eye, velocity), water.surface))
        .unwrap_or((eye, velocity));
    (eye, velocity, crouched)
}

/// A swimmer floating in deep water at 10 m, `start` metres out from a
/// sheer bank `bank` metres over it, swimming into it at `speed` and
/// pressing Space on the first of `fps` frames a second, as
/// update_player_system steps it: the
/// swim or the kick and the walk, each step judged against the ground it
/// is over and the bank (`too_high_to_step`, `step_blocked`). The frames'
/// eye heights and where the body ends.
fn kick_out(bank: f32, start: f32, speed: f32, fps: f32) -> (Vec<f32>, f32, Body) {
    let dt = 1.0 / fps;
    let ground = |x: f32| if x < 0.0 { -10.0 } else { 10.0 + bank };
    let (mut x, mut body) = (-start, (10.0 + SWIM_FREEBOARD, 0.0f32, true));
    let mut leapt = false;
    let mut eyes = vec![body.0];
    for frame in 0..(4.0 * fps) as usize {
        let (eye, velocity, swam) = body;
        let water = still_water(10.0, ground(x));
        let jump = frame == 0;
        let swimming = swims(water, ground(x), eye, velocity, swam);
        let mut to = x + speed * dt;
        if swimming && !(jump && kicks_off(water, eye)) {
            if too_high_to_step(ground(to), ground(x), eye) {
                to = x;
            }
            let here = water.map_or(f32::NEG_INFINITY, |water| water.surface);
            let (eye, velocity) = swim_vertical(
                eye,
                velocity,
                SwimStroke::default(),
                here,
                ground(x),
                ground(to),
                dt,
            );
            body = (eye, velocity, true);
            leapt = false;
        } else {
            let velocity = if swimming && jump {
                JUMP_SPEED
            } else {
                velocity
            };
            leapt |= swimming && jump;
            let afloat = water
                .filter(|water| water.float_eye().is_some())
                .map(|water| water.surface);
            if step_blocked(ground(to), ground(x), eye, velocity, afloat, leapt) {
                to = x;
            }
            let swimmer = kicked(afloat, velocity, swam);
            let on_feet = eye <= ground(x) + EYE_HEIGHT + 0.03 && velocity <= 0.0;
            let before = (eye, velocity);
            let (eye, velocity) = fall(eye, velocity, on_feet, dt);
            body = if eye <= ground(to) + EYE_HEIGHT && velocity <= 0.0 {
                let (eye, crouched) = lands(eye, ground(to), swimmer);
                leapt = false;
                (eye, 0.0, crouched)
            } else {
                (eye, velocity, swimmer)
            };
            if let Some(water) =
                still_water(10.0, ground(to)).filter(|water| water.float_eye().is_some())
                && let Some((eye, velocity)) =
                    takes_the_fall(before, (body.0, body.1), water.surface)
            {
                body = (eye, velocity, body.2);
            }
        }
        x = to;
        eyes.push(body.0);
    }
    (eyes, x, body)
}

#[test]
fn a_swimmer_dives_under_and_floats_back_up() {
    let floating = 10.0 + SWIM_FREEBOARD;
    let dive = SwimStroke {
        dive: true,
        ..Default::default()
    };
    let (deep, _) = swim(floating, 0.0, dive, 2.0);
    assert!(deep < 9.0, "a dive takes the eye under: {deep}");
    // The bed stops the dive.
    let (bottom, velocity) = swim(floating, 0.0, dive, 10.0);
    assert!(
        (bottom - (4.0 + SWIM_BED_CLEARANCE)).abs() < 1e-4 && velocity == 0.0,
        "{bottom} {velocity}"
    );
    // Left alone the body drifts back up and floats, head out.
    let (rested, _) = swim(bottom, 0.0, SwimStroke::default(), 30.0);
    assert!((rested - floating).abs() < 1e-4, "{rested}");
    // Swimming up is quicker than drifting.
    let rise = SwimStroke {
        rise: true,
        ..Default::default()
    };
    let (risen, _) = swim(bottom, 0.0, rise, 2.0);
    let (drifted, _) = swim(bottom, 0.0, SwimStroke::default(), 2.0);
    assert!(risen > drifted + 1.0, "{risen} {drifted}");
}

#[test]
fn a_diver_in_the_shallows_stands_up_out_of_the_water() {
    let dt = 1.0 / 60.0;
    let dive = SwimStroke {
        dive: true,
        ..Default::default()
    };
    // A diver stretched out on a bed at 9 m under water this deep: left
    // alone, or still diving where it is too shallow to lie under.
    for (depth, stroke) in [
        (1.45, SwimStroke::default()),
        (1.0, SwimStroke::default()),
        (0.5, SwimStroke::default()),
        (0.3, dive),
    ] {
        let mut body = (9.0 + SWIM_BED_CLEARANCE, 0.0, true);
        for _ in 0..120 {
            let previous = body.0;
            body = step(body, stroke, 9.0 + depth, 9.0, 9.0, dt);
            // Up at a stroke's pace, without a step.
            assert!(
                body.0 - previous <= SWIM_VERTICAL_SPEED * dt + STANDING_MARGIN + 1e-5,
                "{depth}: {previous} {body:?}"
            );
        }
        // To standing height and no higher, within two seconds.
        assert!(
            (body.0 - (9.0 + EYE_HEIGHT)).abs() < 1e-4 && body.1 == 0.0,
            "{depth}: {body:?}"
        );
    }
    // Where it is deep enough to lie under, a diver stays down.
    let mut body = (9.0 + SWIM_BED_CLEARANCE, 0.0, true);
    for _ in 0..120 {
        body = step(body, dive, 10.0, 9.0, 9.0, dt);
    }
    assert!(
        (body.0 - (9.0 + SWIM_BED_CLEARANCE)).abs() < 1e-4,
        "{body:?}"
    );
    // A walker in the same water is standing, and an eye under the
    // ground (flight set down inside a hill) is seated by walking.
    assert!(!swims(
        still_water(10.0, 9.0),
        9.0,
        9.0 + EYE_HEIGHT,
        0.0,
        true
    ));
    assert!(!swims(None, 9.0, 8.0, 0.0, true));
    // Nor does a walker crouch where the ground rises under it, as it
    // does where the erosion settles: walking sets it on its feet, and a
    // wader looking down or diving stays on them.
    let mut body = (9.0 + EYE_HEIGHT, 0.0, false);
    for ground in [9.0, 9.3, 9.6, 9.6, 9.6] {
        body = step(body, dive, 10.0, ground, ground, dt);
        assert!(
            (body.0 - (ground + EYE_HEIGHT)).abs() < 1e-4 && !body.2,
            "{ground}: {body:?}"
        );
    }
}

#[test]
fn a_diver_comes_up_a_beach_onto_the_bank_without_a_jump() {
    let dive = SwimStroke {
        dive: true,
        ..Default::default()
    };
    let looking_down = SwimStroke {
        forward_pitch: -0.5,
        ..Default::default()
    };
    for slope in [0.1, 0.3, 1.0] {
        // From 3 m under a surface at 10 m up onto a bank a metre over it.
        let bed = |x: f32| (7.0 + slope * x).min(11.0);
        for speed in [1.5, 4.0] {
            for fps in [30.0, 60.0, 144.0] {
                for stroke in [SwimStroke::default(), dive, looking_down] {
                    let dt = 1.0 / fps;
                    let (mut x, mut body) = (0.0f32, (bed(0.0) + SWIM_BED_CLEARANCE, 0.0, true));
                    let (mut ashore, mut stood) = (None, None);
                    let seconds = 4.0 / slope / speed + 2.0;
                    for frame in 0..(seconds * fps) as usize {
                        let to = x + speed * dt;
                        let previous = body.0;
                        body = step(body, stroke, 10.0, bed(x), bed(to), dt);
                        let eye = body.0;
                        // Never up faster than the bed rises and a stroke
                        // climbs.
                        let climb = bed(to) - bed(x) + SWIM_VERTICAL_SPEED * dt + STANDING_MARGIN;
                        assert!(
                            eye - previous <= climb + 1e-4,
                            "{slope} {speed} {fps} {stroke:?}: {previous} -> {eye} at {x}"
                        );
                        x = to;
                        let t = frame as f32 * dt;
                        ashore = ashore.or((bed(x) >= 10.0).then_some(t));
                        stood = stood.or((eye >= bed(x) + EYE_HEIGHT - 1e-4).then_some(t));
                    }
                    let eye = body.0;
                    // Standing on the bank, having stood up with the bed
                    // rising under the body: within a stroke's climb of
                    // the waterline.
                    assert!(
                        (eye - (11.0 + EYE_HEIGHT)).abs() < 1e-3,
                        "{slope} {speed} {fps} {stroke:?}: {eye}"
                    );
                    let late = stood.unwrap() - ashore.unwrap();
                    assert!(
                        late < 1.5,
                        "{slope} {speed} {fps} {stroke:?}: stood {late} s after the waterline"
                    );
                }
            }
        }
    }
}

#[test]
fn a_swimmer_climbs_no_bank_a_walker_could_not() {
    // Floating head out on water at 10 m over a bed at -20 m: onto a
    // ledge under the surface, but not up a rock face out of it.
    let floating = 10.0 + SWIM_FREEBOARD;
    assert!(!too_high_to_step(9.5, -20.0, floating));
    assert!(too_high_to_step(12.0, -20.0, floating));
    // A bank at the water's edge is out of reach afloat, but a kick up
    // out of the water lifts the feet onto it, rising or coming back.
    assert!(too_high_to_step(10.0, -20.0, floating));
    assert!(!step_blocked(10.0, -20.0, 11.4, 2.0, Some(10.0), true));
    assert!(!step_blocked(10.5, -20.0, 11.2, -3.0, Some(10.0), true));
    // A walker steps up only so far from the ground it stands on.
    assert!(!step_blocked(5.2, 4.0, 5.75, 0.0, None, true));
    assert!(step_blocked(5.3, 4.0, 5.75, 0.0, None, true));
    // A body falling past a ledge over the pool from a cliff top lands
    // on nothing over its feet: it falls on into the water, though onto
    // a shore below its feet it lands.
    assert!(step_blocked(23.0, 0.0, 24.0, -18.0, Some(10.0), true));
    assert!(!step_blocked(15.0, 0.0, 24.0, -18.0, Some(10.0), true));
    // Near the water, a body coming down faster than a kick's own fall
    // lands on nothing over its feet either.
    assert!(!step_blocked(10.0, 0.0, 11.0, -3.0, Some(10.0), true));
    assert!(step_blocked(
        10.0,
        0.0,
        11.0,
        -JUMP_SPEED - 1.0,
        Some(10.0),
        true
    ));
    // A leap over a deep channel catches the far bank's lip over its feet,
    // but a walk off the near bank, or a slow fall past a ledge, does not:
    // it lands on nothing over its feet, and falls in.
    assert!(!step_blocked(
        11.5,
        -10.0,
        11.3 + EYE_HEIGHT,
        -5.0,
        Some(10.0),
        true
    ));
    assert!(step_blocked(
        11.5,
        -10.0,
        11.3 + EYE_HEIGHT,
        -5.0,
        Some(10.0),
        false
    ));
    // Only a swimmer's kick lands crouched; a walker's leap lands on its
    // feet, as it does over dry ground.
    assert!(kicked(Some(10.0), -3.0, true) && kicked(None, 2.0, true));
    assert!(!kicked(Some(10.0), -3.0, false) && !kicked(None, -3.0, true));
    // Stretched out along a bed at 4 m: up a slope, but not up a wall
    // without first swimming up it.
    assert!(!too_high_to_step(5.0, 4.0, 4.0 + SWIM_BED_CLEARANCE));
    assert!(too_high_to_step(6.0, 4.0, 4.0 + SWIM_BED_CLEARANCE));
    assert!(!too_high_to_step(6.0, 4.0, 7.0));
}

#[test]
fn a_swimmer_off_the_edge_of_a_bank_falls() {
    let dt = 1.0 / 60.0;
    // Crouched, come ashore on a bank at 10.3 m or still in a metre of
    // water at 10 m, off the edge: onto ground at 5 m, and into the lake
    // at 10 m, 5 m deep there.
    for (surface, bank, eye) in [
        (f32::NEG_INFINITY, 10.3, 11.3),
        (10.0, 10.3, 11.3),
        (10.0, 9.0, 10.5),
    ] {
        let below = 5.0;
        let mut body = (eye, 0.0, true);
        let mut ground = bank;
        let mut speed = 0.0f32;
        for frame in 0..600 {
            let next = if frame == 0 { bank } else { below };
            let previous = body;
            body = step(body, SwimStroke::default(), surface, ground, next, dt);
            // Down no faster than a fall (and the water's entry band,
            // which a swimmer settles out of onto the float line).
            speed = speed.max((previous.0 - body.0) / dt);
            let fall = (previous.1.min(0.0).abs() + FALL_ACCELERATION * dt) * dt;
            assert!(
                previous.0 - body.0 <= fall + SWIM_ENTRY - SWIM_FREEBOARD + 1e-4,
                "{surface} {frame}: {previous:?} -> {body:?}"
            );
            ground = next;
        }
        if surface.is_finite() {
            // Into the lake, taken at its surface, afloat in the end.
            assert!(
                (body.0 - (surface + SWIM_FREEBOARD)).abs() < 0.05,
                "{body:?}"
            );
        } else {
            assert!((body.0 - (below + EYE_HEIGHT)).abs() < 1e-4, "{body:?}");
        }
        assert!(speed > 1.0, "it fell: {speed}");
    }
}

#[test]
fn a_swimmer_over_a_drop_off_rises_at_a_strokes_pace() {
    let dt = 1.0 / 60.0;
    let dive = SwimStroke {
        dive: true,
        ..Default::default()
    };
    // Under water at 10 m over 20 m of it, onto a shelf just under the
    // surface: up at a stroke's pace from where the swimmer was, not
    // carried up the cliff with the bed.
    // (Lower down the shelf's face is a wall the swimmer cannot move
    // onto: `too_high_to_step`.)
    for (eye, shelf, stroke) in [
        (9.2, 8.6, dive),
        (9.5, 8.8, SwimStroke::default()),
        (9.6, 8.6, SwimStroke::default()),
        (10.0 + SWIM_FREEBOARD, 9.74, SwimStroke::default()),
    ] {
        assert!(!too_high_to_step(shelf, -10.0, eye));
        let mut body = (eye, 0.0, true);
        let mut ground = -10.0;
        for _ in 0..180 {
            let previous = body.0;
            body = step(body, stroke, 10.0, ground, shelf, dt);
            assert!(
                body.0 - previous <= SWIM_VERTICAL_SPEED * dt + STANDING_MARGIN + 1e-4,
                "{eye} {shelf}: {previous} -> {body:?}"
            );
            ground = shelf;
        }
        assert!(
            (body.0 - (shelf + EYE_HEIGHT)).abs() < 1e-4 || stroke.dive,
            "{eye} {shelf}: {body:?}"
        );
    }
}

#[test]
fn a_swimmer_kicks_up_out_of_the_water_onto_a_bank_alike_at_any_frame_rate() {
    // Kicking at the bank, it comes up over it rising; from half a
    // metre out, coming down onto it.
    for (bank, start) in [
        (0.0, 0.01),
        (0.4, 0.01),
        (0.8, 0.01),
        (0.9, 0.01),
        (0.0, 0.5),
        (0.4, 0.5),
    ] {
        for fps in [30.0, 60.0, 144.0, 360.0] {
            let (eyes, x, body) = kick_out(bank, start, 1.2, fps);
            // Out onto the bank, and standing on it.
            assert!(
                x > 0.0 && (body.0 - (10.0 + bank + EYE_HEIGHT)).abs() < 1e-4 && !body.2,
                "{bank} m from {start} at {fps}: {x} {body:?}"
            );
            // Up no faster than the kick rises or a stroke climbs: the
            // landing does not lift the body onto its feet.
            let climb = (JUMP_SPEED / fps).max(SWIM_VERTICAL_SPEED / fps + STANDING_MARGIN);
            for pair in eyes.windows(2) {
                assert!(
                    pair[1] - pair[0] <= climb + 1e-4,
                    "{bank} m from {start} at {fps}: {pair:?}"
                );
            }
        }
    }
    // Out of reach of the kick, the bank stays a wall.
    let (_, x, body) = kick_out(1.2, 0.01, 1.2, 60.0);
    assert!(x < 0.0 && body.0 < 10.5, "{x} {body:?}");
    // A body come ashore crouched jumps as a walker does; under the
    // water, Space swims it up.
    assert!(kicks_off(None, 11.0));
    assert!(kicks_off(still_water(10.0, 9.0), 10.2));
    assert!(!kicks_off(still_water(10.0, 9.0), 9.6));
}

#[test]
fn a_walker_keeps_its_footing_down_a_slope_and_jumps_alike_at_any_frame_rate() {
    // Walking down a slope at walking speed and boosted, as
    // update_walking_ground steps it: on its feet every frame, so Space
    // finds it there.
    for (grade, speed, fps) in [
        (0.13, 10.0, 60.0),
        (0.15, 10.0, 30.0),
        (0.05, 25.0, 60.0),
        (0.05, 25.0, 144.0),
    ] {
        {
            let dt = 1.0 / fps;
            let ground = |x: f32| -grade * x;
            let (mut x, mut eye, mut velocity) = (0.0f32, EYE_HEIGHT, 0.0f32);
            let mut footed = 0;
            let frames = (3.0 * fps) as usize;
            for _ in 0..frames {
                let on_feet = eye <= ground(x) + EYE_HEIGHT + 0.03 && velocity <= 0.0;
                footed += usize::from(on_feet);
                let to = x + speed * dt;
                (eye, velocity) = fall(eye, velocity, on_feet, dt);
                if eye <= ground(to) + EYE_HEIGHT && velocity <= 0.0 {
                    (eye, velocity) = (ground(to) + EYE_HEIGHT, 0.0);
                }
                x = to;
            }
            assert_eq!(footed, frames, "{grade} {speed} {fps}");
        }
    }
    // A jump from flat ground rises as high at any frame rate.
    let apex = |fps: f32| {
        let (mut eye, mut velocity, mut top) = (EYE_HEIGHT, JUMP_SPEED, EYE_HEIGHT);
        while velocity > -JUMP_SPEED {
            (eye, velocity) = fall(eye, velocity, false, 1.0 / fps);
            top = top.max(eye);
        }
        top - EYE_HEIGHT
    };
    let exact = JUMP_SPEED * JUMP_SPEED / (2.0 * FALL_ACCELERATION);
    for fps in [20.0, 30.0, 60.0, 144.0, 360.0] {
        assert!(
            (apex(fps) - exact).abs() < 0.01,
            "{fps}: {} vs {exact}",
            apex(fps)
        );
    }
}

#[test]
fn a_body_set_down_from_flight_is_taken_as_it_is() {
    let mut player = Player::default();
    player.set_flying(true);
    assert!(!player.swimming);
    // Low in the shallows it stands up as a swimmer does, rather than
    // being lifted onto its feet in a frame.
    player.set_flying(false);
    assert!(player.swimming && player.vertical_velocity == 0.0 && player.velocity == Vec2::ZERO);
    let dt = 1.0 / 60.0;
    let mut body = (9.0 + 0.5, 0.0, player.swimming);
    for _ in 0..120 {
        let previous = body.0;
        body = step(body, SwimStroke::default(), 9.6, 9.0, 9.0, dt);
        assert!(
            body.0 - previous <= SWIM_VERTICAL_SPEED * dt + STANDING_MARGIN + 1e-5,
            "{previous} {body:?}"
        );
    }
    assert!((body.0 - (9.0 + EYE_HEIGHT)).abs() < 1e-4, "{body:?}");
    // Over its feet in the air, it falls; under the ground, walking
    // seats it.
    assert!(!swims(None, 0.0, 3.0, 0.0, true));
    assert!(!swims(None, 0.0, -0.5, 0.0, true));
}

#[test]
fn deep_water_takes_a_fall_and_lets_a_kick_out_of_it() {
    let deep = WaterHere {
        surface: 10.0,
        ground: 0.0,
        current: Vec2::ZERO,
        turbulence: 0.0,
        clarity: 0.0,
        sea: 1.0,
        still: 1.0,
        kind: WaterKind::Sea,
    };
    assert_eq!(deep.float_eye(), Some(10.0 + SWIM_FREEBOARD));
    // A falling eye anywhere within the entry band is the water's.
    for eye in [10.0, 10.0 + SWIM_FREEBOARD, 10.0 + SWIM_ENTRY] {
        assert!(swims(Some(deep), 0.0, eye, -9.0, false), "{eye}");
    }
    assert!(!swims(
        Some(deep),
        0.0,
        10.0 + SWIM_ENTRY + 0.01,
        -9.0,
        true
    ));
    // A swimmer kicked up out of it is in the air while it rises, however
    // short a frame leaves it inside the band.
    assert!(!swims(
        Some(deep),
        0.0,
        10.0 + SWIM_FREEBOARD + 0.01,
        7.3,
        true
    ));
    assert!(swims(
        Some(deep),
        0.0,
        10.0 + SWIM_FREEBOARD,
        SWIM_VERTICAL_SPEED,
        true
    ));
}

#[test]
fn deep_water_takes_a_fall_alike_at_any_frame_rate() {
    // From rest `drop` metres over the float line on water `deep` metres
    // deep, the frames `fps` a second and the first of them `phase` of a
    // frame: the deepest the eye goes.
    let deepest = |deep: f32, drop: f32, fps: f32, phase: f32| {
        let mut body = (10.0 + SWIM_FREEBOARD + drop, 0.0f32, false);
        let mut deepest = body.0;
        let mut dt = phase / fps;
        for _ in 0..(20.0 * fps) as usize {
            body = step(
                body,
                SwimStroke::default(),
                10.0,
                10.0 - deep,
                10.0 - deep,
                dt,
            );
            deepest = deepest.min(body.0);
            dt = 1.0 / fps;
        }
        deepest
    };
    for deep in [30.0, 2.5, 1.6] {
        for drop in [0.3, 1.0, 10.0, 30.0] {
            let depths: Vec<f32> = [20.0, 30.0, 60.0, 144.0, 240.0, 360.0]
                .into_iter()
                .flat_map(|fps| (0..10).map(move |k| (fps, k as f32 / 10.0)))
                .map(|(fps, phase)| deepest(deep, drop, fps, phase))
                .collect();
            let (low, high) = depths
                .iter()
                .fold((f32::MAX, f32::MIN), |(l, h), &d| (l.min(d), h.max(d)));
            // The same plunge to within a few centimetres, from a hop to
            // a cliff, at 20 frames a second or 360, and never stood on
            // the bed by a frame that reached it.
            assert!(high - low < 0.08, "{deep} m deep, {drop} m: {low}..{high}");
            assert!(
                high < 10.0 + SWIM_FREEBOARD,
                "{deep} m deep, {drop} m: the water is entered"
            );
        }
    }
    // A wader stepping into deeper water whose surface stands higher is
    // not lifted onto it: only a fall from over the water is taken.
    assert_eq!(takes_the_fall((10.3, 0.0), (10.29, -0.38), 10.5), None);
}

#[test]
fn swimming_forward_follows_the_eye_down() {
    let floating = 10.0 + SWIM_FREEBOARD;
    let looking_down = SwimStroke {
        forward_pitch: -0.7,
        ..Default::default()
    };
    let (eye, velocity) = swim(floating, 0.0, looking_down, 1.5);
    assert!(eye < floating - 0.8 && velocity < -0.5, "{eye} {velocity}");
    // Looking level, swimming forward keeps to the surface.
    let level = SwimStroke {
        forward_pitch: 0.0,
        ..Default::default()
    };
    let (eye, _) = swim(floating, 0.0, level, 1.5);
    assert!((eye - floating).abs() < 1e-4, "{eye}");
}

#[test]
fn a_rivers_mouth_past_its_ribbon_is_the_seas_to_draw() {
    let mouth = drawn_as(None, SEA_LEVEL + 0.02, SEA_LEVEL - 2.0, WaterKind::River);
    assert_eq!(
        (mouth.kind, mouth.sea, mouth.still, mouth.clarity),
        (WaterKind::Sea, 1.0, 1.0, 0.0)
    );
    // Under its ribbon it is the river's, in the sea's colours.
    let ribbon = forest_water::rivers::surface::SurfaceVertex {
        sea: 1.0,
        still: 1.0,
        clarity: 0.2,
        ..Default::default()
    };
    let reach = drawn_as(
        Some(ribbon),
        SEA_LEVEL + 0.02,
        SEA_LEVEL - 2.0,
        WaterKind::River,
    );
    assert_eq!(
        (reach.kind, reach.sea, reach.clarity),
        (WaterKind::River, 1.0, 0.2)
    );
    // Upstream, over dry ground, with nothing drawn, a stream's own.
    let stream = drawn_as(None, SEA_LEVEL + 3.0, SEA_LEVEL + 2.0, WaterKind::River);
    assert_eq!((stream.kind, stream.sea), (WaterKind::River, 0.0));
}

#[test]
fn water_kinds_meet_at_the_higher_surface() {
    // Flat ground at the constant field's height with no rivers: no
    // water over it unless it lies under the sea.
    let noise = NoiseField {
        samples: vec![0.70; NOISE_RESOLUTION * NOISE_RESOLUTION],
    };
    let erosion = ErosionCache::default();
    assert!(water_at(&erosion, &noise, 1000.0, 1000.0, [1000.0, 1000.0]).is_none());
    let sea_floor = NoiseField {
        samples: vec![0.0; NOISE_RESOLUTION * NOISE_RESOLUTION],
    };
    let ground = forest_erosion::sample_eroded_height(
        &erosion,
        &sea_floor,
        1000.0,
        1000.0,
        [1000.0, 1000.0],
    );
    if ground < SEA_LEVEL {
        let water = water_at(&erosion, &sea_floor, 1000.0, 1000.0, [1000.0, 1000.0]).unwrap();
        assert_eq!(water.kind, WaterKind::Sea);
        assert_eq!(water.surface, SEA_LEVEL);
        assert!(water.depth() > 0.0);
    }
}
