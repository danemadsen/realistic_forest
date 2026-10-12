//! Free flight under the planet's gravity, in the manner of Kerbal Space
//! Program.
//!
//! Walking keeps its own tuned controller and the noclip flight its direct
//! speeds; free flight is the third way to move. The body's state is a
//! planet-centred position and velocity in f64, in axes that do not turn with
//! anything (the planet does not spin: its day is the sun's motion), so
//! momentum is conserved and an orbit is just a fall that keeps missing the
//! ground. Every substep the body feels:
//!
//! - the planet's pull, `μ / r²` toward its centre, `μ = g₀ R²`
//!   ([`Planet::gravitational_parameter`]);
//! - the air's drag, quadratic in the speed and in proportion to the air's
//!   density, which thins with the atmosphere's scale height and is gone at
//!   its top ([`air_density`]): a falling body settles to a terminal speed,
//!   and getting to orbit means climbing out of the air before turning
//!   sideways;
//! - its own thrust, along the view (W and S), sideways (A and D) and along
//!   the local vertical (Space and Shift).
//!
//! The integrator is velocity Verlet (kick, drift, kick) with the drag taken
//! implicitly in each half kick, so without air an orbit neither gains nor
//! loses energy however long it runs, and in the air no step can reverse the
//! velocity. Above the atmosphere, coasting, the body can be time-warped as
//! KSP warps "on rails"; the warp drops out at the top of the air.
//!
//! The body comes to rest on the ground or the water it meets, as a vessel
//! lands in KSP, and stays there until its thrust lifts it again. Its state
//! is written back into the player's chart coordinates every frame, so the
//! camera, the terrain streams and the face crossings see an ordinary
//! player.

use super::{Face, LocalFrame, Planet, AIR_SCALE_HEIGHT, ATMOSPHERE_TOP, PLANET_SURFACE_GRAVITY};
use bevy::math::{DVec2, DVec3, Vec2, Vec3};
use std::f64::consts::{PI, TAU};

/// Thrust, m/s²: a little more than the surface's pull, so the body can hover
/// and climb slowly; [`THRUST_BOOST`] times that with Left Ctrl held.
pub const THRUST: f64 = 12.0;
pub const THRUST_BOOST: f64 = 5.0;

/// Terminal speed of a falling body in sea-level air, m/s: a skydiver's. The
/// drag's strength follows from it; higher up the air is thinner and the
/// terminal speed greater.
pub const SEA_LEVEL_TERMINAL_SPEED: f64 = 60.0;

/// The time warp's rates, KSP's on-rails steps.
pub const WARP_LEVELS: [f64; 6] = [1.0, 5.0, 10.0, 50.0, 100.0, 1000.0];

/// Altitude the "circular orbit" shortcut raises a lower body to: clear of
/// the air, as KSP's low orbits are.
pub const CIRCULAR_ORBIT_MIN_ALTITUDE: f64 = 80_000.0;

/// The longest substep in real time and under warp, seconds.
const STEP: f64 = 1.0 / 120.0;
const WARP_STEP: f64 = 0.5;
/// The most substeps one frame may take.
const MAX_SUBSTEPS: usize = 4_000;
/// How far the ground may drop from under a resting body before it falls.
const SETTLE_TOLERANCE: f64 = 0.05;
/// The largest arc, radians of the planet, a warped frame may sweep: the
/// body's chart is extended past its face's edge for less than this, and
/// [`super::systems::carry_player_across_faces`] carries it from there.
const MAX_WARP_ARC: f64 = 0.1;

/// The air's density at `altitude`, as a fraction of its density at sea
/// level: exponential with the scale height, eased to nothing at the top of
/// the atmosphere so the drag has no step there.
pub fn air_density(altitude: f64) -> f64 {
    let scale_height = AIR_SCALE_HEIGHT as f64;
    let top = (-ATMOSPHERE_TOP / scale_height).exp();
    let density = (-altitude.max(0.0) / scale_height).exp();
    ((density - top) / (1.0 - top)).max(0.0)
}

/// Drag deceleration per (m/s)² of speed in sea-level air.
fn drag_per_speed_squared() -> f64 {
    PLANET_SURFACE_GRAVITY / (SEA_LEVEL_TERMINAL_SPEED * SEA_LEVEL_TERMINAL_SPEED)
}

impl Planet {
    /// The planet's gravitational parameter `GM`, m³/s²: what gives
    /// [`PLANET_SURFACE_GRAVITY`] at sea level.
    pub fn gravitational_parameter(&self) -> f64 {
        PLANET_SURFACE_GRAVITY * self.radius * self.radius
    }

    /// Gravitational acceleration at a planet-centred `position`, toward the
    /// centre and falling off with the square of the distance from it.
    pub fn gravity_vector(&self, position: DVec3) -> DVec3 {
        let distance = position.length().max(0.5 * self.radius);
        -position.normalize_or(DVec3::Y) * (self.gravitational_parameter() / (distance * distance))
    }
}

/// A body flying free: its eye's planet-centred position and its velocity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FreeFlight {
    /// Planet-centred metres.
    pub position: DVec3,
    /// Metres per second, planet axes.
    pub velocity: DVec3,
    /// At rest on the ground or the water, until its thrust lifts it.
    pub landed: bool,
    /// Whether what it rests on is water: KSP's "splashed down".
    pub on_water: bool,
    /// Whether it rests where it stood when free flight began, a building's
    /// floor say, rather than on the ground it came down on: it stays there
    /// while the ground under it changes.
    pub on_floor: bool,
    /// Altitude of the surface under it, the ground with its snow or the
    /// water, as last measured: what its height above the ground is from.
    pub surface: f64,
    /// Index into [`WARP_LEVELS`].
    pub warp: usize,
    /// The speed of its last touchdown, m/s.
    pub touchdown: Option<f64>,
}

impl FreeFlight {
    /// A body at `position` moving at `velocity`, in the air.
    pub fn new(position: DVec3, velocity: DVec3) -> Self {
        Self {
            position,
            velocity,
            landed: false,
            on_water: false,
            on_floor: false,
            surface: 0.0,
            warp: 0,
            touchdown: None,
        }
    }

    pub fn altitude(&self, planet: &Planet) -> f64 {
        self.position.length() - planet.radius
    }

    /// The local vertical at the body.
    pub fn up(&self) -> DVec3 {
        self.position.normalize_or(DVec3::Y)
    }

    /// The warp's rate.
    pub fn warp_rate(&self) -> f64 {
        WARP_LEVELS[self.warp.min(WARP_LEVELS.len() - 1)]
    }

    /// Whether the body may be warped: coasting above the air.
    pub fn may_warp(&self, planet: &Planet) -> bool {
        !self.landed && self.altitude(planet) >= ATMOSPHERE_TOP
    }

    /// The acceleration of a body at `position` moving at `velocity` under
    /// `thrust`, less the drag (which the integrator takes implicitly).
    fn pull(planet: &Planet, position: DVec3, thrust: DVec3) -> DVec3 {
        planet.gravity_vector(position) + thrust
    }

    /// Half a kick: the pull for `seconds`, then the drag for as long,
    /// implicitly, so it slows the body however long the step and never
    /// turns it round.
    fn kick(planet: &Planet, position: DVec3, velocity: DVec3, thrust: DVec3, seconds: f64) -> DVec3 {
        let velocity = velocity + Self::pull(planet, position, thrust) * seconds;
        let density = air_density(position.length() - planet.radius);
        if density <= 0.0 {
            return velocity;
        }
        velocity / (1.0 + drag_per_speed_squared() * density * velocity.length() * seconds)
    }

    /// Advance the body `seconds` (of its own time; a warped frame passes
    /// more) under a constant `thrust` in planet axes, m/s². A landed body
    /// stays put unless its thrust outweighs its weight. Returns the seconds
    /// actually flown: a warped step stops short where it enters the air.
    pub fn advance(&mut self, planet: &Planet, thrust: DVec3, seconds: f64) -> f64 {
        if seconds <= 0.0 {
            return 0.0;
        }
        if self.landed {
            let lift = Self::pull(planet, self.position, thrust).dot(self.up());
            if lift <= 0.0 {
                self.velocity = DVec3::ZERO;
                return seconds;
            }
            self.landed = false;
            self.on_water = false;
            self.on_floor = false;
        }
        let warped = self.warp_rate() > 1.0;
        let step = if warped { WARP_STEP } else { STEP };
        let steps = ((seconds / step).ceil() as usize).clamp(1, MAX_SUBSTEPS);
        let h = seconds / steps as f64;
        let mut flown = 0.0;
        for _ in 0..steps {
            let half = Self::kick(planet, self.position, self.velocity, thrust, 0.5 * h);
            self.position += half * h;
            self.velocity = Self::kick(planet, self.position, half, thrust, 0.5 * h);
            flown += h;
            if warped && self.altitude(planet) < ATMOSPHERE_TOP {
                self.warp = 0;
                break;
            }
        }
        flown
    }

    /// Bring the body to rest where its eye would sink below `touch` metres
    /// above sea level (the ground's or the water's surface plus the eye's
    /// height), as KSP lands a vessel that meets the ground. A body resting
    /// on ground that has since dropped away under it (a tile of erosion
    /// fading in, say) falls onto it again. Returns whether it touched down
    /// this call.
    pub fn settle(&mut self, planet: &Planet, touch: f64, water: bool) -> bool {
        let altitude = self.altitude(planet);
        if altitude > touch {
            if self.landed && !self.on_floor && altitude > touch + SETTLE_TOLERANCE {
                self.landed = false;
                self.on_water = false;
            }
            return false;
        }
        let touched_down = !self.landed;
        if touched_down {
            self.touchdown = Some(self.velocity.length());
            self.on_water = water;
        }
        self.landed = true;
        self.on_floor = false;
        self.warp = 0;
        self.position = self.up() * (planet.radius + touch);
        self.velocity = DVec3::ZERO;
        touched_down
    }

    /// The seconds of flight a frame of `frame_seconds` advances: the warp's
    /// rate times it, capped so a warped frame sweeps at most
    /// [`MAX_WARP_ARC`] of the planet.
    pub fn frame_seconds(&self, frame_seconds: f64) -> f64 {
        let rate = self.warp_rate();
        if rate <= 1.0 {
            return frame_seconds;
        }
        let speed = self.velocity.length().max(1.0);
        (frame_seconds * rate).min(MAX_WARP_ARC * self.position.length() / speed).max(frame_seconds)
    }
}

/// Where a free-flying body is in the game's coordinates on `face`: its
/// terrain plane position (on the face's chart, extended past the face's edge
/// where it has flown over it) and altitude as the player's `(x, y, z)`, and
/// the local frame there.
pub struct ChartPlacement {
    pub position: Vec3,
    pub frame: LocalFrame,
}

/// Place a planet-centred `position` on `face`'s chart. The chart is
/// undefined on the far side of the planet; a body never gets there from its
/// own face within a frame, and is put at the face's centre if it does.
pub fn place_on(planet: &Planet, face: Face, position: DVec3) -> ChartPlacement {
    let up = position.normalize_or(DVec3::Y);
    let chart = planet.chart_of(face, up).unwrap_or(DVec2::ZERO);
    let plane = Planet::plane_of(face, chart);
    let altitude = position.length() - planet.radius;
    ChartPlacement {
        position: Vec3::new(plane.x as f32, altitude as f32, plane.y as f32),
        frame: planet.local_frame(face, chart),
    }
}

/// The planet-centred position and local frame of the player's `(x, y, z)`
/// on `face`.
pub fn planet_position(planet: &Planet, face: Face, position: Vec3) -> (DVec3, LocalFrame) {
    let chart = Planet::chart_of_plane(face, DVec2::new(position.x as f64, position.z as f64));
    (planet.position(face, chart, position.y as f64), planet.local_frame(face, chart))
}

/// What KSP's flight readouts show of a body's orbit, from its state alone.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OrbitReadout {
    /// Altitudes above sea level, metres. No apoapsis on an escape path.
    pub apoapsis: Option<f64>,
    pub periapsis: f64,
    pub eccentricity: f64,
    /// Seconds; none on an escape path or a circle.
    pub period: Option<f64>,
    pub time_to_apoapsis: Option<f64>,
    pub time_to_periapsis: Option<f64>,
    /// Whether the body is moving fast enough to leave for good.
    pub escaping: bool,
}

impl OrbitReadout {
    /// The conic a body at `position` with `velocity` follows about the
    /// planet, from its energy and angular momentum.
    pub fn of(planet: &Planet, position: DVec3, velocity: DVec3) -> Self {
        let mu = planet.gravitational_parameter();
        let r = position.length().max(1.0);
        let energy = 0.5 * velocity.length_squared() - mu / r;
        let momentum = position.cross(velocity);
        let eccentricity_vector = velocity.cross(momentum) / mu - position / r;
        let eccentricity = eccentricity_vector.length();
        // The semi-latus rectum gives the periapsis of any conic.
        let semi_latus = momentum.length_squared() / mu;
        let periapsis = semi_latus / (1.0 + eccentricity) - planet.radius;
        if energy >= 0.0 {
            return Self {
                apoapsis: None,
                periapsis,
                eccentricity,
                period: None,
                time_to_apoapsis: None,
                time_to_periapsis: None,
                escaping: true,
            };
        }
        // Bound: an ellipse, or a straight fall up and back (eccentricity 1
        // to rounding), whose apoapsis is where it stops.
        let eccentricity = eccentricity.min(1.0);
        let semi_major = -mu / (2.0 * energy);
        let apoapsis = semi_major * (1.0 + eccentricity) - planet.radius;
        let periapsis = semi_major * (1.0 - eccentricity) - planet.radius;
        let mean_motion = (mu / (semi_major * semi_major * semi_major)).sqrt();
        let period = TAU / mean_motion;
        // Where along the ellipse the body is: its true anomaly, measured
        // from periapsis in the direction of motion, then the mean anomaly
        // that runs evenly in time. Undefined on a circle, and on the
        // degenerate ellipse of a straight fall.
        let (time_to_apoapsis, time_to_periapsis) = if eccentricity > 1e-6 && eccentricity < 1.0 - 1e-9 {
            let cosine = (eccentricity_vector.dot(position) / (eccentricity * r)).clamp(-1.0, 1.0);
            let mut true_anomaly = cosine.acos();
            if position.dot(velocity) < 0.0 {
                true_anomaly = TAU - true_anomaly;
            }
            let half = 0.5 * true_anomaly;
            let eccentric = 2.0
                * ((1.0 - eccentricity).sqrt() * half.sin()).atan2((1.0 + eccentricity).sqrt() * half.cos());
            let eccentric = eccentric.rem_euclid(TAU);
            let mean = eccentric - eccentricity * eccentric.sin();
            (
                Some((PI - mean).rem_euclid(TAU) / mean_motion),
                Some((TAU - mean).rem_euclid(TAU) / mean_motion),
            )
        } else {
            (None, None)
        };
        Self {
            apoapsis: Some(apoapsis),
            periapsis,
            eccentricity,
            period: Some(period),
            time_to_apoapsis,
            time_to_periapsis,
            escaping: false,
        }
    }
}

/// KSP's name for what a free-flying body is doing.
pub fn situation(planet: &Planet, flight: &FreeFlight, orbit: &OrbitReadout) -> &'static str {
    if flight.landed {
        if flight.on_water { "SPLASHED" } else { "LANDED" }
    } else if flight.altitude(planet) < ATMOSPHERE_TOP {
        "FLYING"
    } else if orbit.escaping {
        "ESCAPING"
    } else if orbit.periapsis > ATMOSPHERE_TOP {
        "ORBITING"
    } else {
        "SUB-ORBITAL"
    }
}

/// Velocity in the local frame's horizontal plane (x east, y north) and
/// along its vertical.
pub fn local_velocity(frame: LocalFrame, velocity: DVec3) -> (Vec2, f32) {
    let local = frame.to_local(velocity);
    (Vec2::new(local.x as f32, local.z as f32), local.y as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn circular(planet: &Planet, altitude: f64) -> FreeFlight {
        let r = planet.radius + altitude;
        let speed = (planet.gravitational_parameter() / r).sqrt();
        FreeFlight::new(DVec3::new(0.0, r, 0.0), DVec3::new(speed, 0.0, 0.0))
    }

    /// Fly `seconds` in frames of `frame` seconds, as the game does.
    fn fly(planet: &Planet, flight: &mut FreeFlight, thrust: DVec3, seconds: f64, frame: f64) {
        let mut left = seconds;
        while left > 1e-9 {
            let step = flight.frame_seconds(frame).min(left);
            left -= flight.advance(planet, thrust, step);
        }
    }

    #[test]
    fn the_pull_is_surface_gravity_at_sea_level_and_falls_with_the_square_of_the_distance() {
        let planet = Planet::default();
        let sea = planet.gravity_vector(DVec3::new(0.0, planet.radius, 0.0));
        assert!((sea.length() - PLANET_SURFACE_GRAVITY).abs() < 1e-9);
        assert!(sea.normalize().dot(DVec3::NEG_Y) > 0.999_999_999);
        let high = planet.gravity_vector(DVec3::new(2.0 * planet.radius, 0.0, 0.0));
        assert!((high.length() - PLANET_SURFACE_GRAVITY / 4.0).abs() < 1e-9);
        assert!(high.normalize().dot(DVec3::NEG_X) > 0.999_999_999);
        // The same pull the walking controller scales its falls by.
        assert!((planet.gravity_at(planet.radius) - high.length()).abs() < 1e-9);
    }

    #[test]
    fn the_air_thins_to_nothing_at_the_top_of_the_atmosphere() {
        assert!((air_density(0.0) - 1.0).abs() < 1e-12);
        assert!((air_density(-50.0) - 1.0).abs() < 1e-12);
        assert_eq!(air_density(ATMOSPHERE_TOP), 0.0);
        assert_eq!(air_density(2.0 * ATMOSPHERE_TOP), 0.0);
        let mut previous = 1.0;
        for step in 1..=600 {
            let density = air_density(step as f64 * 100.0);
            assert!(density < previous && density >= 0.0);
            previous = density;
        }
        // One scale height up, the air is about e times thinner.
        assert!((air_density(AIR_SCALE_HEIGHT as f64) - (-1.0f64).exp()).abs() < 1e-3);
    }

    #[test]
    fn a_circular_orbit_stays_circular_for_a_whole_orbit_in_real_time() {
        let planet = Planet::default();
        let mut flight = circular(&planet, 100_000.0);
        let start = flight.position;
        let r = start.length();
        let period = OrbitReadout::of(&planet, flight.position, flight.velocity).period.unwrap();
        let mut lowest = f64::MAX;
        let mut highest = f64::MIN;
        let mut left = period;
        while left > 1e-9 {
            left -= flight.advance(&planet, DVec3::ZERO, left.min(1.0 / 60.0));
            lowest = lowest.min(flight.position.length());
            highest = highest.max(flight.position.length());
        }
        assert!(highest - lowest < 0.05, "radius wandered {} m", highest - lowest);
        assert!((lowest - r).abs() < 0.05 && (highest - r).abs() < 0.05);
        assert!(flight.position.distance(start) < 1.0, "{} m from the start", flight.position.distance(start));
    }

    #[test]
    fn warped_orbits_keep_their_energy() {
        let planet = Planet::default();
        let mu = planet.gravitational_parameter();
        let energy = |flight: &FreeFlight| 0.5 * flight.velocity.length_squared() - mu / flight.position.length();
        let mut flight = circular(&planet, 100_000.0);
        // A little eccentric, to exercise the speed changing round the orbit.
        flight.velocity *= 1.02;
        flight.warp = WARP_LEVELS.len() - 1;
        let before = energy(&flight);
        let orbit = OrbitReadout::of(&planet, flight.position, flight.velocity);
        fly(&planet, &mut flight, DVec3::ZERO, 10.0 * orbit.period.unwrap(), 1.0 / 60.0);
        assert_eq!(flight.warp, WARP_LEVELS.len() - 1, "the orbit stays out of the air");
        let after = energy(&flight);
        assert!(((after - before) / before).abs() < 1e-5, "{before} -> {after}");
        let later = OrbitReadout::of(&planet, flight.position, flight.velocity);
        assert!((later.apoapsis.unwrap() - orbit.apoapsis.unwrap()).abs() < 50.0);
        assert!((later.periapsis - orbit.periapsis).abs() < 50.0);
    }

    #[test]
    fn a_warped_frame_sweeps_a_bounded_arc_and_the_warp_drops_out_in_the_air() {
        let planet = Planet::default();
        let mut flight = circular(&planet, 70_000.0);
        flight.warp = WARP_LEVELS.len() - 1;
        let seconds = flight.frame_seconds(0.05);
        let arc = seconds * flight.velocity.length() / flight.position.length();
        assert!(arc <= MAX_WARP_ARC + 1e-12 && seconds > 0.05);
        // Slowed to a fall that dips into the air, the warp stops there.
        flight.velocity *= 0.9;
        let mut left = 2_000.0;
        while flight.warp != 0 && left > 1e-9 {
            let step = flight.frame_seconds(1.0 / 60.0).min(left);
            left -= flight.advance(&planet, DVec3::ZERO, step);
        }
        let altitude = flight.altitude(&planet);
        assert_eq!(flight.warp, 0);
        // It drops within a half-second substep's fall of the top of the air.
        assert!(altitude < ATMOSPHERE_TOP && altitude > ATMOSPHERE_TOP - 500.0, "{altitude}");
        assert!(!flight.may_warp(&planet));
    }

    #[test]
    fn readouts_of_a_circle_an_ellipse_and_an_escape() {
        let planet = Planet::default();
        let mu = planet.gravitational_parameter();
        let flight = circular(&planet, 100_000.0);
        let circle = OrbitReadout::of(&planet, flight.position, flight.velocity);
        assert!(circle.eccentricity < 1e-9);
        assert!((circle.apoapsis.unwrap() - 100_000.0).abs() < 1e-3);
        assert!((circle.periapsis - 100_000.0).abs() < 1e-3);
        let r = planet.radius + 100_000.0;
        assert!((circle.period.unwrap() - TAU * (r * r * r / mu).sqrt()).abs() < 1e-6);
        assert!(!circle.escaping);

        // At periapsis 100 km of an orbit reaching 300 km.
        let rp = planet.radius + 100_000.0;
        let ra = planet.radius + 300_000.0;
        let a = 0.5 * (rp + ra);
        let speed = (mu * (2.0 / rp - 1.0 / a)).sqrt();
        let at_periapsis = OrbitReadout::of(&planet, DVec3::new(rp, 0.0, 0.0), DVec3::new(0.0, 0.0, speed));
        assert!((at_periapsis.apoapsis.unwrap() - 300_000.0).abs() < 1e-3);
        assert!((at_periapsis.periapsis - 100_000.0).abs() < 1e-3);
        let period = at_periapsis.period.unwrap();
        assert!((at_periapsis.time_to_apoapsis.unwrap() - 0.5 * period).abs() < 1e-3);
        let to_periapsis = at_periapsis.time_to_periapsis.unwrap();
        assert!(to_periapsis < 1e-3 || (to_periapsis - period).abs() < 1e-3);

        // A quarter of the way round in time, the times add up.
        let mut flight = FreeFlight::new(DVec3::new(rp, 0.0, 0.0), DVec3::new(0.0, 0.0, speed));
        let mut left = 0.25 * period;
        while left > 1e-9 {
            left -= flight.advance(&planet, DVec3::ZERO, left.min(0.5));
        }
        let later = OrbitReadout::of(&planet, flight.position, flight.velocity);
        assert!((later.time_to_periapsis.unwrap() - 0.75 * period).abs() < 1.0);
        assert!((later.time_to_apoapsis.unwrap() - 0.25 * period).abs() < 1.0);

        // Past escape speed there is no apoapsis.
        let escape = (2.0 * mu / r).sqrt() * 1.01;
        let leaving = OrbitReadout::of(&planet, DVec3::new(0.0, r, 0.0), DVec3::new(escape, 0.0, 0.0));
        assert!(leaving.escaping && leaving.apoapsis.is_none() && leaving.period.is_none());
        assert!((leaving.periapsis - 100_000.0).abs() < 1e-3);
    }

    #[test]
    fn a_hop_straight_up_is_sub_orbital_with_its_apoapsis_where_it_stops() {
        let planet = Planet::default();
        let mu = planet.gravitational_parameter();
        let r = planet.radius + 70_000.0;
        let speed = 1_000.0;
        let orbit = OrbitReadout::of(&planet, DVec3::new(0.0, 0.0, r), DVec3::new(0.0, 0.0, speed));
        let stops = 1.0 / (1.0 / r - speed * speed / (2.0 * mu)) - planet.radius;
        assert!((orbit.apoapsis.unwrap() - stops).abs() < 1e-3);
        assert!(orbit.periapsis < 0.0);
        let flight = FreeFlight::new(DVec3::new(0.0, 0.0, r), DVec3::new(0.0, 0.0, speed));
        assert_eq!(situation(&planet, &flight, &orbit), "SUB-ORBITAL");
    }

    #[test]
    fn a_fall_through_the_air_settles_to_its_terminal_speed() {
        let planet = Planet::default();
        // With no ground to stop it, from 500 m the body falls on through
        // sea-level air for most of a minute.
        let mut flight = FreeFlight::new(DVec3::new(0.0, planet.radius + 500.0, 0.0), DVec3::ZERO);
        fly(&planet, &mut flight, DVec3::ZERO, 60.0, 1.0 / 60.0);
        let terminal = SEA_LEVEL_TERMINAL_SPEED / air_density(flight.altitude(&planet)).sqrt();
        let falling = -flight.velocity.dot(flight.up());
        assert!((falling - terminal).abs() < 0.02 * terminal, "{falling} m/s, terminal {terminal}");
        // A slow frame rate falls alike.
        let mut slow = FreeFlight::new(DVec3::new(0.0, planet.radius + 500.0, 0.0), DVec3::ZERO);
        fly(&planet, &mut slow, DVec3::ZERO, 60.0, 0.05);
        assert!((slow.altitude(&planet) - flight.altitude(&planet)).abs() < 0.5);
    }

    #[test]
    fn a_body_lands_where_it_meets_the_ground_and_stays_until_its_thrust_lifts_it() {
        let planet = Planet::default();
        let ground = 120.0;
        let mut flight = FreeFlight::new(DVec3::new(0.0, planet.radius + ground + 40.0, 0.0), DVec3::ZERO);
        let mut touched = false;
        for _ in 0..600 {
            flight.advance(&planet, DVec3::ZERO, 1.0 / 60.0);
            touched |= flight.settle(&planet, ground, false);
        }
        assert!(touched && flight.landed && !flight.on_water);
        assert!((flight.altitude(&planet) - ground).abs() < 1e-6);
        let speed = flight.touchdown.unwrap();
        // About the speed of a 40 m drop, a little less for the air.
        assert!(speed < (2.0 * PLANET_SURFACE_GRAVITY * 40.0).sqrt() && speed > 20.0, "{speed}");
        // Thrust that does not outweigh the pull leaves it where it is.
        let up = flight.up();
        let rest = flight.position;
        flight.advance(&planet, up * 9.0 + DVec3::X * 5.0, 1.0);
        flight.settle(&planet, ground, false);
        assert!(flight.position.distance(rest) < 1e-6);
        assert!(flight.landed);
        // Ground that drops away under it lets it fall onto it again.
        flight.settle(&planet, ground - 3.0, false);
        assert!(!flight.landed);
        for _ in 0..120 {
            flight.advance(&planet, DVec3::ZERO, 1.0 / 60.0);
            flight.settle(&planet, ground - 3.0, false);
        }
        assert!(flight.landed && (flight.altitude(&planet) - (ground - 3.0)).abs() < 1e-6);
        // But a body that began on a floor stays on it.
        let mut on_floor = FreeFlight { landed: true, on_floor: true, ..flight };
        on_floor.settle(&planet, ground - 10.0, false);
        assert!(on_floor.landed);
        // Full thrust up lifts it off.
        let ground = ground - 3.0;
        for _ in 0..60 {
            flight.advance(&planet, up * THRUST, 1.0 / 60.0);
            flight.settle(&planet, ground, false);
        }
        assert!(!flight.landed && flight.altitude(&planet) > ground + 0.5);
    }

    #[test]
    fn hover_thrust_holds_a_body_still() {
        let planet = Planet::default();
        let start = DVec3::new(0.0, planet.radius + 500.0, 0.0);
        let mut flight = FreeFlight::new(start, DVec3::ZERO);
        let hover = -planet.gravity_vector(start);
        fly(&planet, &mut flight, hover, 10.0, 1.0 / 60.0);
        assert!(flight.position.distance(start) < 1e-3, "{}", flight.position.distance(start));
    }

    #[test]
    fn placing_a_body_on_its_face_round_trips_through_the_players_coordinates() {
        let planet = Planet::default();
        for (face, plane) in [
            (Face::PosY, Vec3::new(1_234.5, 812.0, -9_876.25)),
            (Face::NegX, Vec3::new(-400_000.0, 80_000.0, 300_000.0)),
        ] {
            let (position, frame) = planet_position(&planet, face, plane);
            let placed = place_on(&planet, face, position);
            assert!(placed.position.distance(plane) < 0.1, "{} -> {}", plane, placed.position);
            assert!(placed.frame.up.dot(frame.up) > 0.999_999_999);
        }
        // Past the face's edge the chart extends; the face crossing carries it on.
        let s = planet.half_width;
        let beyond = planet.position(Face::PosY, DVec2::new(s + 20_000.0, 0.0), 1_000.0);
        let placed = place_on(&planet, Face::PosY, beyond);
        let chart = Planet::chart_of_plane(Face::PosY, DVec2::new(placed.position.x as f64, placed.position.z as f64));
        assert!((chart.x - (s + 20_000.0)).abs() < 1.0 && chart.y.abs() < 1.0, "{chart}");
    }
}
