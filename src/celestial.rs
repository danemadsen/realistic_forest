//! The sun and the moon as bodies: the planet goes round the sun and the
//! moon round the planet, on Kepler orbits, in Earth's proportions.
//!
//! Every size, distance and mass is Earth's, the Moon's and the Sun's in
//! units of the planet's own radius and mass: the moon a 0.273-radius ball
//! 60.3 radii out with 1/81 of the planet's mass, the sun 109 radii across
//! and 23,481 radii away with 333,000 times its mass, so both discs span the
//! half degree they do from Earth and eclipses come out as Earth's do, total
//! and annular. The orbits' shapes are Earth's too (eccentricities 0.0167 and
//! 0.0549, the moon's tilted 5.145° to the ecliptic, its nodes and perigee
//! turning as the sun drags them), and the planet's axis leans 23.44° to its
//! orbit, so the seasons come and go. The periods follow from the masses by
//! Kepler's third law: with this planet's radius and its 23 m/s² of gravity
//! the year is about 77 days and the month about six, and the planet spins so
//! that its solar day is 24 hours.
//!
//! Time is celestial seconds from the epoch, local midnight at spawn on
//! midsummer's day with the moon full. Positions come in the planet's axes
//! (the cube faces' frame, `crate::planet`): at spawn +X is east and +Y up,
//! so in these right-handed axes geographic north is -Z. Spawn stands at 45°
//! south, where the noon sun stands toward -Z as the game's light always has
//! (and the snow's sunny slopes face): it rises at +X, culminates toward -Z
//! and sets at -X.

use crate::planet::Planet;
use bevy::math::{DMat3, DVec3};
use std::f64::consts::{PI, TAU};

/// The moon's radius, its orbit's semi-major axis and its mass, in the
/// planet's radius and mass: 1,737.4 km and 384,400 km over Earth's 6,371 km,
/// and 1/81.3 of Earth's mass.
pub const MOON_RADIUS: f64 = 0.2727;
pub const MOON_DISTANCE: f64 = 60.336;
pub const MOON_MASS: f64 = 0.0123;
pub const MOON_ECCENTRICITY: f64 = 0.0549;
pub const MOON_INCLINATION_DEGREES: f64 = 5.145;
/// The sun's radius, the planet's orbit's semi-major axis and the sun's mass,
/// likewise: 695,700 km, one astronomical unit and 332,946 Earths.
pub const SUN_RADIUS: f64 = 109.2;
pub const SUN_DISTANCE: f64 = 23_481.4;
pub const SUN_MASS: f64 = 332_946.0;
pub const ORBIT_ECCENTRICITY: f64 = 0.0167;
pub const AXIAL_TILT_DEGREES: f64 = 23.44;
/// Seconds in the planet's solar day: what the day/night cycle's 24 hours are.
pub const SOLAR_DAY: f64 = 86_400.0;
/// Spawn's latitude: 45° south of the equator.
pub const SPAWN_LATITUDE_DEGREES: f64 = -45.0;

/// Where the sun's perigee lies on the ecliptic, as seen from the planet:
/// Earth's perihelion longitude, 102.9°, half a turn round.
const SUN_PERIGEE_DEGREES: f64 = 282.9;
/// The epoch's mean longitude of the sun: midsummer in the south, the
/// December solstice.
const EPOCH_SUN_LONGITUDE_DEGREES: f64 = 270.0;
/// The moon at the epoch: full (opposite the mean sun), at its perigee, with
/// its ascending node a quarter turn from the line to the sun so the full
/// moon passes clear of the planet's shadow.
const EPOCH_MOON_NODE_DEGREES: f64 = 0.0;
const EPOCH_MOON_PERIGEE_DEGREES: f64 = 90.0;
/// The sun's pull turns the moon's orbit: its nodes regress once in about
/// (4/3) T_year² / T_month (Earth: 17.8 years by that rule, 18.61 in fact)
/// and its perigee advances once in 0.476 of that (8.85 years).
const NODE_PERIOD_FACTOR: f64 = 18.61 / 17.83;
const PERIGEE_PERIOD_FACTOR: f64 = 8.85 / 17.83;

/// The bodies' orbits and the planet's spin, worked out for a planet.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Orbits {
    /// The planet's radius, metres.
    pub radius: f64,
    sun_semi_major: f64,
    moon_semi_major: f64,
    /// Mean motions, radians per second.
    sun_motion: f64,
    moon_motion: f64,
    node_rate: f64,
    perigee_rate: f64,
    /// The planet's spin, radians per second, and its angle at the epoch.
    spin_rate: f64,
    spin_at_epoch: f64,
    tilt: f64,
    /// The spawn's east, up and geographic north in the planet's body axes
    /// (its pole is the body's z).
    spawn_east: DVec3,
    spawn_up: DVec3,
    spawn_north: DVec3,
}

impl Orbits {
    pub fn new(planet: &Planet) -> Self {
        let radius = planet.radius;
        let planet_gm = planet.gravitational_parameter();
        let sun_semi_major = SUN_DISTANCE * radius;
        let moon_semi_major = MOON_DISTANCE * radius;
        // Kepler's third law: n² a³ = G (M + m).
        let sun_motion = (planet_gm * (SUN_MASS + 1.0 + MOON_MASS) / sun_semi_major.powi(3)).sqrt();
        let moon_motion = (planet_gm * (1.0 + MOON_MASS) / moon_semi_major.powi(3)).sqrt();
        let year = TAU / sun_motion;
        let month = TAU / moon_motion;
        let precession = 4.0 / 3.0 * year * year / month;
        let latitude = SPAWN_LATITUDE_DEGREES.to_radians();
        let epoch_sun = EPOCH_SUN_LONGITUDE_DEGREES.to_radians();
        Self {
            radius,
            sun_semi_major,
            moon_semi_major,
            sun_motion,
            moon_motion,
            node_rate: -TAU / (NODE_PERIOD_FACTOR * precession),
            perigee_rate: TAU / (PERIGEE_PERIOD_FACTOR * precession),
            // A solar day of SOLAR_DAY: the planet turns once more than the
            // sun goes round it each year.
            spin_rate: TAU / SOLAR_DAY + sun_motion,
            // Local mean midnight at spawn (longitude 0) at the epoch: the
            // mean sun, which runs along the equator at the sun's mean
            // longitude, stands at the meridian's far side.
            spin_at_epoch: epoch_sun + PI,
            tilt: AXIAL_TILT_DEGREES.to_radians(),
            spawn_east: DVec3::new(0.0, 1.0, 0.0),
            spawn_up: DVec3::new(latitude.cos(), 0.0, latitude.sin()),
            spawn_north: DVec3::new(-latitude.sin(), 0.0, latitude.cos()),
        }
    }

    /// The sidereal year and month, seconds.
    pub fn year(&self) -> f64 {
        TAU / self.sun_motion
    }

    pub fn month(&self) -> f64 {
        TAU / self.moon_motion
    }

    /// New moon to new moon, seconds.
    pub fn synodic_month(&self) -> f64 {
        TAU / (self.moon_motion - self.sun_motion)
    }

    /// The planet's north pole in its own (the cube faces') axes.
    pub fn pole(&self) -> DVec3 {
        self.to_planet_axes(DVec3::Z)
    }

    /// A body-axes vector in the planet's (spawn-frame) axes: east to +X, up
    /// to +Y and, for a turn rather than a mirror, north to -Z.
    fn to_planet_axes(&self, body: DVec3) -> DVec3 {
        DVec3::new(body.dot(self.spawn_east), body.dot(self.spawn_up), -body.dot(self.spawn_north))
    }

    /// The rotation from ecliptic axes (x toward the vernal equinox, z to the
    /// ecliptic's north) to the planet's axes at `seconds`.
    fn ecliptic_to_planet(&self, seconds: f64) -> DMat3 {
        let (sin_tilt, cos_tilt) = self.tilt.sin_cos();
        // Ecliptic to equatorial: about the equinox line by the tilt.
        let equatorial = DMat3::from_cols(
            DVec3::new(1.0, 0.0, 0.0),
            DVec3::new(0.0, cos_tilt, sin_tilt),
            DVec3::new(0.0, -sin_tilt, cos_tilt),
        );
        // Equatorial to the turning body axes.
        let angle = (self.spin_at_epoch + self.spin_rate * seconds).rem_euclid(TAU);
        let (sin_spin, cos_spin) = angle.sin_cos();
        let spun = DMat3::from_cols(
            DVec3::new(cos_spin, -sin_spin, 0.0),
            DVec3::new(sin_spin, cos_spin, 0.0),
            DVec3::new(0.0, 0.0, 1.0),
        );
        // Body to the planet's (spawn-frame) axes.
        let planet = DMat3::from_cols(
            self.to_planet_axes(DVec3::X),
            self.to_planet_axes(DVec3::Y),
            self.to_planet_axes(DVec3::Z),
        );
        planet * spun * equatorial
    }

    /// Where the sun and the moon are at `seconds`, planet-centred metres in
    /// the planet's axes.
    pub fn at(&self, seconds: f64) -> Sky {
        let sun_anomaly = EPOCH_SUN_LONGITUDE_DEGREES.to_radians() - SUN_PERIGEE_DEGREES.to_radians()
            + self.sun_motion * seconds;
        let sun = orbit_position(
            self.sun_semi_major,
            ORBIT_ECCENTRICITY,
            0.0,
            0.0,
            SUN_PERIGEE_DEGREES.to_radians(),
            sun_anomaly,
        );
        let node = EPOCH_MOON_NODE_DEGREES.to_radians() + self.node_rate * seconds;
        let perigee = EPOCH_MOON_PERIGEE_DEGREES.to_radians() + self.perigee_rate * seconds;
        // Full at the epoch: the moon's mean longitude opposite the sun's.
        let moon_longitude = EPOCH_SUN_LONGITUDE_DEGREES.to_radians() + PI + self.moon_motion * seconds;
        let moon = orbit_position(
            self.moon_semi_major,
            MOON_ECCENTRICITY,
            MOON_INCLINATION_DEGREES.to_radians(),
            node,
            perigee - node,
            moon_longitude - perigee,
        );
        let to_planet = self.ecliptic_to_planet(seconds);
        Sky {
            sun: to_planet * sun,
            moon: to_planet * moon,
            planet_radius: self.radius,
            sun_radius: SUN_RADIUS * self.radius,
            moon_radius: MOON_RADIUS * self.radius,
            to_ecliptic: to_planet.transpose(),
        }
    }

    /// When the sun's centre rises and sets over the horizon of an eye at
    /// `eye` (planet-centred, planet axes) on day `day` of the clock, in
    /// hours: none for a crossing that day lacks, near the poles.
    pub fn sunrise_and_sunset(&self, day: i64, eye: DVec3) -> (Option<f64>, Option<f64>) {
        let up = eye.normalize();
        let elevation = |hours: f64| self.at((day as f64 + hours / 24.0) * SOLAR_DAY).seen_from(eye).to_sun.dot(up);
        let steps = 96;
        let step = 24.0 / steps as f64;
        let (mut rise, mut set) = (None, None);
        let mut previous = elevation(0.0);
        for index in 1..=steps {
            let hours = index as f64 * step;
            let current = elevation(hours);
            if (previous < 0.0) != (current < 0.0) {
                let (mut low, mut high) = (hours - step, hours);
                for _ in 0..30 {
                    let middle = 0.5 * (low + high);
                    if (elevation(middle) < 0.0) == (previous < 0.0) {
                        low = middle;
                    } else {
                        high = middle;
                    }
                }
                let crossing = 0.5 * (low + high);
                if previous < 0.0 {
                    rise.get_or_insert(crossing);
                } else {
                    set.get_or_insert(crossing);
                }
            }
            previous = current;
        }
        (rise, set)
    }

    /// The season for an eye at `eye` from the sun's place on its orbit:
    /// each a quarter of the year centred on its solstice or equinox, the
    /// hemispheres' opposite.
    pub fn season(&self, sky: &Sky, eye: DVec3) -> &'static str {
        let sun = sky.to_ecliptic * sky.sun;
        // From the northern spring equinox, at the ecliptic's x.
        let mut longitude = sun.y.atan2(sun.x).to_degrees();
        if eye.dot(self.pole()) < 0.0 {
            longitude += 180.0;
        }
        match longitude.rem_euclid(360.0) {
            l if l < 45.0 => "spring",
            l if l < 135.0 => "summer",
            l if l < 225.0 => "autumn",
            l if l < 315.0 => "winter",
            _ => "spring",
        }
    }

    /// The deepest moment of the next eclipse an eye at `eye` sees after
    /// `seconds`, within `days`: of the sun, with the sun up, or of the moon
    /// in the planet's shadow, with the moon up. One under way at `seconds`
    /// is passed over.
    pub fn next_eclipse(&self, seconds: f64, eye: DVec3, days: f64) -> Option<(f64, Eclipse)> {
        let up = eye.normalize();
        // How deep an eclipse the eye sees, over the depth that begins to
        // show: 2% of the sun hidden, or a fifth of the moon's sunlight taken
        // by the planet's shadow (a shallow pass through the penumbra hardly
        // dims it).
        let depth = |time: f64| {
            let sky = self.at(time);
            let view = sky.seen_from(eye);
            let solar = if view.to_sun.dot(up) > 0.0 { view.solar_eclipse / 0.02 } else { 0.0 };
            let lunar = if view.to_moon.dot(up) > 0.0 { (1.0 - view.moon_sunlight) / 0.2 } else { 0.0 };
            if solar >= lunar { (solar, Eclipse::Solar) } else { (lunar, Eclipse::Lunar) }
        };
        // Three degrees from both alignments the bodies cannot line up
        // within the coarse step: the moon gains under a degree on the sun in
        // twenty minutes, and the eclipses need it within a degree and two
        // thirds (its parallax and the discs, or the planet's penumbra).
        let far = |time: f64| {
            let sky = self.at(time);
            let sun = sky.sun.normalize();
            let moon = sky.moon.normalize();
            let threshold = 3f64.to_radians().cos();
            sun.dot(moon) < threshold && (-sun).dot(moon) < threshold
        };
        const VISIBLE: f64 = 1.0;
        let coarse = 1200.0;
        let fine = 60.0;
        let end = seconds + days * SOLAR_DAY;
        let mut time = seconds;
        // Pass over an eclipse already under way.
        while time < end && depth(time).0 > VISIBLE {
            time += fine;
        }
        while time < end {
            if far(time) {
                time += coarse;
                continue;
            }
            let (now, kind) = depth(time);
            if now > VISIBLE {
                // Follow it to its deepest moment.
                let (mut deepest, mut at) = (now, time);
                let mut probe = time + fine;
                while probe < time + SOLAR_DAY {
                    let (next, _) = depth(probe);
                    if next <= VISIBLE {
                        break;
                    }
                    if next > deepest {
                        (deepest, at) = (next, probe);
                    }
                    probe += fine;
                }
                return Some((at, kind));
            }
            time += fine;
        }
        None
    }
}

/// The two kinds of eclipse.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Eclipse {
    Solar,
    Lunar,
}

/// The true anomaly of a Kepler orbit of eccentricity `e` at `mean_anomaly`,
/// and the distance there over the semi-major axis.
pub fn kepler(mean_anomaly: f64, e: f64) -> (f64, f64) {
    let mean = mean_anomaly.rem_euclid(TAU);
    let mut eccentric = if e < 0.8 { mean } else { PI };
    for _ in 0..12 {
        let step = (eccentric - e * eccentric.sin() - mean) / (1.0 - e * eccentric.cos());
        eccentric -= step;
        if step.abs() < 1e-14 {
            break;
        }
    }
    let half = 0.5 * eccentric;
    let anomaly = 2.0 * ((1.0 + e).sqrt() * half.sin()).atan2((1.0 - e).sqrt() * half.cos());
    (anomaly, 1.0 - e * eccentric.cos())
}

/// Ecliptic position of a body on a Kepler orbit: semi-major axis `a`,
/// eccentricity `e`, inclination, longitude of the ascending node, argument
/// of the periapsis and mean anomaly.
fn orbit_position(a: f64, e: f64, inclination: f64, node: f64, periapsis: f64, mean_anomaly: f64) -> DVec3 {
    let (anomaly, distance) = kepler(mean_anomaly, e);
    let r = a * distance;
    let (sin_u, cos_u) = (periapsis + anomaly).sin_cos();
    let (sin_node, cos_node) = node.sin_cos();
    let (sin_i, cos_i) = inclination.sin_cos();
    DVec3::new(
        r * (cos_node * cos_u - sin_node * sin_u * cos_i),
        r * (sin_node * cos_u + cos_node * sin_u * cos_i),
        r * sin_u * sin_i,
    )
}

/// The sun and the moon at one moment, planet-centred metres in the
/// planet's axes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sky {
    pub sun: DVec3,
    pub moon: DVec3,
    pub planet_radius: f64,
    pub sun_radius: f64,
    pub moon_radius: f64,
    /// The planet's axes to the ecliptic's, which the stars are fixed in.
    pub to_ecliptic: DMat3,
}

/// What an eye somewhere on or over the planet sees of the sun and the moon.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SkyView {
    /// Unit directions toward the bodies' centres, planet axes.
    pub to_sun: DVec3,
    pub to_moon: DVec3,
    /// The discs' angular radii, radians.
    pub sun_radius: f64,
    pub moon_radius: f64,
    /// The share of the moon's disc the sun lights, as seen from the eye.
    pub moon_lit: f64,
    /// The moon's brightness for its phase over a full moon's: the lit share
    /// and the way a rough surface dims away from opposition (Allen's
    /// lunar phase law).
    pub moon_phase_brightness: f64,
    /// How much of the sun's disc the moon hides from the eye.
    pub solar_eclipse: f64,
    /// How much sunlight reaches the moon past the planet's shadow.
    pub moon_sunlight: f64,
}

impl Sky {
    /// The sky from an eye at `eye`, planet-centred metres in planet axes:
    /// the moon is near enough that where on the planet the eye is moves it
    /// by up to a degree, so an eclipse of the sun is seen only from a
    /// strip of the planet.
    pub fn seen_from(&self, eye: DVec3) -> SkyView {
        let to_sun = self.sun - eye;
        let to_moon = self.moon - eye;
        let sun_distance = to_sun.length();
        let moon_distance = to_moon.length();
        let to_sun = to_sun / sun_distance;
        let to_moon = to_moon / moon_distance;
        let sun_radius = (self.sun_radius / sun_distance).asin();
        let moon_radius = (self.moon_radius / moon_distance).asin();
        // The phase: the angle at the moon between the sun and the eye.
        let moon_to_sun = (self.sun - self.moon).normalize();
        let phase_angle = moon_to_sun.dot(-to_moon).clamp(-1.0, 1.0).acos();
        let moon_lit = 0.5 * (1.0 + phase_angle.cos());
        let degrees = phase_angle.to_degrees();
        let moon_phase_brightness = 10f64.powf(-0.4 * (0.026 * degrees + 4.0e-9 * degrees.powi(4)));
        let separation = to_sun.dot(to_moon).clamp(-1.0, 1.0).acos();
        SkyView {
            to_sun,
            to_moon,
            sun_radius,
            moon_radius,
            moon_lit,
            moon_phase_brightness,
            solar_eclipse: disc_overlap(sun_radius, moon_radius, separation),
            moon_sunlight: self.moon_sunlight(),
        }
    }

    /// The share of the sun's light that reaches the moon past the planet:
    /// one outside the planet's penumbra, none deep in its umbra, the cone
    /// widths following from the sun's and the planet's sizes.
    pub fn moon_sunlight(&self) -> f64 {
        let away = -self.sun.normalize();
        let along = self.moon.dot(away);
        if along <= 0.0 {
            return 1.0;
        }
        let off_axis = (self.moon - away * along).length();
        let sun_distance = self.sun.length();
        let umbra = self.planet_radius - along * (self.sun_radius - self.planet_radius) / sun_distance;
        let penumbra = self.planet_radius + along * (self.sun_radius + self.planet_radius) / sun_distance;
        // Sunlight at a point rises across the penumbra; the moon's disc
        // takes the average over its width.
        let light = |distance: f64| ((distance - umbra) / (penumbra - umbra)).clamp(0.0, 1.0);
        let m = self.moon_radius;
        0.25 * (light(off_axis - m) + 2.0 * light(off_axis) + light(off_axis + m))
    }
}

/// The share of a disc of angular radius `a` hidden by one of radius `b`
/// whose centre is `separation` away: the overlap of two circles, flat
/// enough at these sizes.
pub fn disc_overlap(a: f64, b: f64, separation: f64) -> f64 {
    if separation >= a + b {
        return 0.0;
    }
    if separation <= (a - b).abs() {
        return if b >= a { 1.0 } else { (b * b) / (a * a) };
    }
    let lens = |r: f64, other: f64| {
        let cosine = ((separation * separation + r * r - other * other) / (2.0 * separation * r)).clamp(-1.0, 1.0);
        r * r * cosine.acos()
    };
    let kite = 0.5
        * ((-separation + a + b) * (separation + a - b) * (separation - a + b) * (separation + a + b))
            .max(0.0)
            .sqrt();
    ((lens(a, b) + lens(b, a) - kite) / (PI * a * a)).clamp(0.0, 1.0)
}

/// A name for the moon's phase, from its elongation from the sun as the
/// planet sees it: waxing from new to full, waning back.
pub fn phase_name(sky: &Sky) -> &'static str {
    let sun = sky.sun.normalize();
    let moon = sky.moon.normalize();
    let pole = sun.cross(moon);
    // Waxing while the moon runs ahead of the sun eastward: the ecliptic's
    // north, in planet axes, is the ecliptic z.
    let north = sky.to_ecliptic.transpose() * DVec3::Z;
    let waxing = pole.dot(north) > 0.0;
    let lit = 0.5 * (1.0 - sun.dot(moon));
    match (lit, waxing) {
        (l, _) if l < 0.03 => "new",
        (l, _) if l > 0.97 => "full",
        (l, true) if l < 0.47 => "waxing crescent",
        (l, true) if l < 0.53 => "first quarter",
        (_, true) => "waxing gibbous",
        (l, false) if l < 0.47 => "waning crescent",
        (l, false) if l < 0.53 => "last quarter",
        (_, false) => "waning gibbous",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn orbits() -> Orbits {
        Orbits::new(&Planet::default())
    }

    fn spawn() -> DVec3 {
        DVec3::new(0.0, Planet::default().radius, 0.0)
    }

    /// Elevation of the sun over spawn, degrees.
    fn sun_elevation(orbits: &Orbits, seconds: f64) -> f64 {
        orbits.at(seconds).seen_from(spawn()).to_sun.dot(DVec3::Y).asin().to_degrees()
    }

    #[test]
    fn kepler_solves_its_equation() {
        for e in [0.0, 0.0167, 0.0549, 0.3, 0.9] {
            for step in 0..64 {
                let mean = step as f64 * TAU / 64.0;
                let (anomaly, distance) = kepler(mean, e);
                // Back from the true anomaly to the mean anomaly.
                let eccentric = 2.0 * ((1.0 - e).sqrt() * (0.5 * anomaly).sin()).atan2((1.0 + e).sqrt() * (0.5 * anomaly).cos());
                let back = (eccentric - e * eccentric.sin()).rem_euclid(TAU);
                let difference = (back - mean.rem_euclid(TAU)).abs();
                assert!(difference < 1e-9 || (TAU - difference) < 1e-9, "e {e} mean {mean}: {back}");
                assert!((distance - (1.0 - e * eccentric.cos())).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn the_periods_follow_from_the_masses_by_keplers_third_law() {
        let planet = Planet::default();
        let orbits = orbits();
        let gm = planet.gravitational_parameter();
        let moon = MOON_DISTANCE * planet.radius;
        let month = TAU * (moon.powi(3) / (gm * (1.0 + MOON_MASS))).sqrt();
        assert!((orbits.month() - month).abs() < 1e-6);
        // About a 77-day year and a six-day month with this small, heavy
        // planet; the moon's phases come round in a little over six days.
        let days = |seconds: f64| seconds / SOLAR_DAY;
        assert!((days(orbits.year()) - 77.2).abs() < 0.5, "{}", days(orbits.year()));
        assert!((days(orbits.month()) - 5.77).abs() < 0.05, "{}", days(orbits.month()));
        assert!(days(orbits.synodic_month()) > days(orbits.month()));
        // The bodies keep their distances.
        for step in 0..200 {
            let sky = orbits.at(step as f64 * 0.37 * SOLAR_DAY);
            let sun = sky.sun.length() / planet.radius;
            let moon = sky.moon.length() / planet.radius;
            assert!((sun - SUN_DISTANCE).abs() < SUN_DISTANCE * (ORBIT_ECCENTRICITY + 1e-9));
            assert!((moon - MOON_DISTANCE).abs() < MOON_DISTANCE * (MOON_ECCENTRICITY + 1e-9));
        }
    }

    #[test]
    fn both_discs_span_half_a_degree() {
        let sky = orbits().at(0.0).seen_from(spawn());
        let sun = 2.0 * sky.sun_radius.to_degrees();
        let moon = 2.0 * sky.moon_radius.to_degrees();
        assert!((sun - 0.533).abs() < 0.012, "{sun}");
        assert!((moon - 0.518).abs() < 0.04, "{moon}");
    }

    #[test]
    fn midsummer_at_spawn_has_a_low_midnight_sun_below_the_horizon_and_a_high_noon() {
        let orbits = orbits();
        // 45° south at its solstice: 21.6° below at midnight, 68.4° up at noon,
        // give or take the equation of time.
        assert!((sun_elevation(&orbits, 0.0) + 21.56).abs() < 0.5, "{}", sun_elevation(&orbits, 0.0));
        let noon = (0..240).map(|step| sun_elevation(&orbits, 0.5 * SOLAR_DAY + (step as f64 - 120.0) * 15.0))
            .fold(f64::MIN, f64::max);
        assert!((noon - 68.44).abs() < 0.3, "{noon}");
        // The sun rises in the east and sets in the west.
        let morning = orbits.at(0.25 * SOLAR_DAY).seen_from(spawn()).to_sun;
        let evening = orbits.at(0.75 * SOLAR_DAY).seen_from(spawn()).to_sun;
        assert!(morning.x > 0.5 && evening.x < -0.5, "{morning} {evening}");
        // At noon it stands due north, toward -Z.
        let north = orbits.at(0.5 * SOLAR_DAY).seen_from(spawn()).to_sun;
        assert!(north.z < -0.3 && north.x.abs() < 0.05, "{north}");
    }

    #[test]
    fn a_quarter_year_on_the_days_and_nights_are_even() {
        let orbits = orbits();
        let equinox = (0.25 * orbits.year() / SOLAR_DAY).round() * SOLAR_DAY;
        let noon = (0..240)
            .map(|step| sun_elevation(&orbits, equinox + 0.5 * SOLAR_DAY + (step as f64 - 120.0) * 15.0))
            .fold(f64::MIN, f64::max);
        // 45° from the equator at an equinox: the noon sun 45° up (within the
        // day's drift of the declination, which is quick in a short year).
        assert!((noon - 45.0).abs() < 2.5, "{noon}");
        let midwinter = (0.5 * orbits.year() / SOLAR_DAY).round() * SOLAR_DAY;
        let winter_noon = (0..240)
            .map(|step| sun_elevation(&orbits, midwinter + 0.5 * SOLAR_DAY + (step as f64 - 120.0) * 15.0))
            .fold(f64::MIN, f64::max);
        assert!((winter_noon - 21.56).abs() < 0.6, "{winter_noon}");
    }

    #[test]
    fn the_epoch_moon_is_full_and_up_at_midnight_and_wanes_to_new() {
        let orbits = orbits();
        let sky = orbits.at(0.0);
        let view = sky.seen_from(spawn());
        assert!(view.moon_lit > 0.98, "{}", view.moon_lit);
        assert!(view.to_moon.y > 0.2, "{}", view.to_moon);
        assert_eq!(phase_name(&sky), "full");
        assert!((view.moon_phase_brightness - 1.0).abs() < 0.2);
        assert!(view.moon_sunlight > 0.99, "no eclipse at the epoch");
        // Half a synodic month later the moon is new and dark.
        let new = orbits.at(0.5 * orbits.synodic_month());
        let new_view = new.seen_from(spawn());
        assert!(new_view.moon_lit < 0.02 && new_view.moon_phase_brightness < 0.01);
        assert_eq!(phase_name(&new), "new");
        // A quarter on it is at a quarter, waning.
        let quarter = orbits.at(0.25 * orbits.synodic_month());
        let lit = quarter.seen_from(spawn()).moon_lit;
        assert!((lit - 0.5).abs() < 0.08, "{lit}");
        assert!(phase_name(&quarter).starts_with("waning") || phase_name(&quarter) == "last quarter");
    }

    #[test]
    fn eclipses_happen_where_the_geometry_says() {
        // Discs: the same size centred hides all; apart, none; half-way, some.
        assert_eq!(disc_overlap(0.0046, 0.0047, 0.0), 1.0);
        assert!((disc_overlap(0.0046, 0.0040, 0.0) - (0.0040f64 / 0.0046).powi(2)).abs() < 1e-12);
        assert_eq!(disc_overlap(0.0046, 0.0047, 0.01), 0.0);
        let half = disc_overlap(0.0046, 0.0046, 0.0046);
        assert!(half > 0.3 && half < 0.5, "{half}");
        // Over a few years the moon passes through the planet's shadow and
        // in front of the sun now and then, never every month.
        let orbits = orbits();
        let months = 60;
        let step = orbits.synodic_month() / 400.0;
        let mut lunar = 0;
        let mut solar = 0;
        for month in 0..months {
            let full = month as f64 * orbits.synodic_month();
            let darkest = (0..400)
                .map(|i| orbits.at(full + (i as f64 - 200.0) * step * 0.25).moon_sunlight())
                .fold(1.0, f64::min);
            lunar += (darkest < 0.5) as usize;
            let new = full + 0.5 * orbits.synodic_month();
            // The planet's centre as the eye, so any eclipse anywhere counts.
            let deepest = (0..400)
                .map(|i| {
                    let sky = orbits.at(new + (i as f64 - 200.0) * step * 0.25);
                    let separation = sky.sun.normalize().dot(sky.moon.normalize()).clamp(-1.0, 1.0).acos();
                    // From somewhere on the planet the moon can stand a
                    // parallax's worth further toward the sun.
                    separation - sky.planet_radius / sky.moon.length()
                })
                .fold(f64::MAX, f64::min);
            solar += (deepest < 0.0093) as usize;
        }
        assert!(lunar > 0 && lunar < months / 2, "{lunar} lunar eclipses in {months} months");
        assert!(solar > 0 && solar < months / 2, "{solar} solar eclipses in {months} months");
    }

    #[test]
    fn the_stars_stay_put_as_the_planet_turns() {
        let orbits = orbits();
        let a = orbits.at(1000.0).to_ecliptic;
        let b = orbits.at(1000.0 + 0.25 * SOLAR_DAY).to_ecliptic;
        // An orthonormal turn.
        for matrix in [a, b] {
            let product = matrix * matrix.transpose();
            assert!((product - DMat3::IDENTITY).to_cols_array().iter().all(|v| v.abs() < 1e-12));
            assert!((matrix.determinant() - 1.0).abs() < 1e-12);
        }
        // The pole points at the same stars all the while, spawn's zenith not.
        let pole = orbits.pole();
        assert!(((a * pole) - (b * pole)).length() < 1e-9);
        assert!(((a * DVec3::Y) - (b * DVec3::Y)).length() > 0.5);
        // The south pole stands 45° up over spawn's southern (+Z) horizon.
        let south = -pole;
        assert!((south.y - 45f64.to_radians().sin()).abs() < 1e-12 && south.z > 0.7, "{south}");
    }

    #[test]
    fn sunrise_and_sunset_follow_the_seasons() {
        let orbits = orbits();
        // 45° south at the solstice: a day of 15.4 hours round noon, give or
        // take the equation of time.
        let (rise, set) = orbits.sunrise_and_sunset(0, spawn());
        let (rise, set) = (rise.unwrap(), set.unwrap());
        assert!((rise - 4.29).abs() < 0.3 && (set - 19.71).abs() < 0.3, "{rise} {set}");
        assert!((set - rise - 15.43).abs() < 0.1, "{}", set - rise);
        // Half a year on, 8.6 hours.
        let winter = (0.5 * orbits.year() / SOLAR_DAY).round() as i64;
        let (rise, set) = orbits.sunrise_and_sunset(winter, spawn());
        let length = set.unwrap() - rise.unwrap();
        assert!((length - 8.57).abs() < 0.15, "{length}");
        // The seasons are the hemispheres' opposites.
        let sky = orbits.at(0.0);
        assert_eq!(orbits.season(&sky, spawn()), "summer");
        assert_eq!(orbits.season(&sky, -spawn()), "winter");
        assert_eq!(orbits.season(&orbits.at(0.25 * orbits.year()), spawn()), "autumn");
        assert_eq!(orbits.season(&orbits.at(0.5 * orbits.year()), spawn()), "winter");
        assert_eq!(orbits.season(&orbits.at(0.75 * orbits.year()), spawn()), "spring");
    }

    #[test]
    fn the_next_eclipse_is_found_at_its_deepest_and_then_passed_over() {
        let orbits = orbits();
        let up = DVec3::Y;
        let (mut solar, mut lunar) = (0, 0);
        let mut time = 0.0;
        for _ in 0..8 {
            let (at, kind) = orbits.next_eclipse(time, spawn(), 2000.0).expect("eclipses come round");
            assert!(at > time);
            let view = orbits.at(at).seen_from(spawn());
            // Seen from spawn, with the body up, and deeper than a minute
            // either side.
            let depth = |view: &SkyView| match kind {
                Eclipse::Solar => view.solar_eclipse,
                Eclipse::Lunar => 1.0 - view.moon_sunlight,
            };
            match kind {
                Eclipse::Solar => {
                    solar += 1;
                    assert!(view.solar_eclipse > 0.02 && view.to_sun.dot(up) > 0.0, "{view:?}");
                }
                Eclipse::Lunar => {
                    lunar += 1;
                    assert!(view.moon_sunlight < 0.8 && view.to_moon.dot(up) > 0.0, "{view:?}");
                }
            }
            for side in [-60.0, 60.0] {
                let near = orbits.at(at + side).seen_from(spawn());
                let up_then = match kind {
                    Eclipse::Solar => near.to_sun.dot(up) > 0.0,
                    Eclipse::Lunar => near.to_moon.dot(up) > 0.0,
                };
                assert!(!up_then || depth(&near) <= depth(&view) + 1e-9);
            }
            time = at;
        }
        assert!(solar > 0 && lunar > 0, "{solar} solar, {lunar} lunar");
        assert_eq!(orbits.next_eclipse(0.0, spawn(), 0.5), None);
    }
}
