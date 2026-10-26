use super::*;

#[test]
fn sun_rises_in_east_and_sets_in_west() {
    let sunrise = sample(6.0);
    let noon = sample(12.0);
    let sunset = sample(18.0);
    assert!(sunrise.sun_direction[0] < -0.999);
    assert!(sunset.sun_direction[0] > 0.999);
    assert!(sunrise.sun_direction[1].abs() < 1e-6);
    assert!(sunset.sun_direction[1].abs() < 1e-6);
    assert!(noon.sun_direction[1] < -0.9);
    assert_eq!(noon.sun_strength, 1.0);
    assert_eq!(noon.daylight, 1.0);
    assert_eq!(noon.moon_intensity, 0.0);
}

#[test]
fn moon_lights_midnight_and_directions_are_unit_length() {
    let midnight = sample(0.0);
    assert_eq!(midnight.sun_strength, 0.0);
    assert_eq!(midnight.daylight, 0.0);
    assert!(midnight.moon_intensity > 0.0);
    assert!(midnight.moon_direction[1] < -0.9);
    for hour in 0..96 {
        let light = sample(hour as f32 * 0.25);
        let length_squared: f32 = light.sun_direction[..3].iter().map(|v| v * v).sum();
        assert!((length_squared - 1.0).abs() < 1e-5);
        for axis in 0..3 {
            assert_eq!(light.moon_direction[axis], -light.sun_direction[axis]);
        }
    }
}

#[test]
fn twilight_and_midnight_are_continuous() {
    let dusk = sample(18.25);
    assert_eq!(dusk.sun_strength, 0.0);
    assert!(dusk.daylight > 0.0 && dusk.daylight < 0.5);
    let before = sample(23.999);
    let after = sample(0.001);
    for axis in 0..3 {
        assert!((before.sun_direction[axis] - after.sun_direction[axis]).abs() < 0.001);
    }
    assert_eq!(sample(24.0).sun_direction, sample(0.0).sun_direction);
}

#[test]
fn cycle_uses_elapsed_time_wraps_and_pauses() {
    let mut cycle = DayNightCycle::default();
    cycle.advance(60.0);
    assert!((cycle.time_hours - 11.0).abs() < 1e-5);
    cycle.time_hours = 23.5;
    cycle.advance(60.0);
    assert!((cycle.time_hours - 0.5).abs() < 1e-5);
    cycle.advance(24.0 * 60.0);
    assert!((cycle.time_hours - 0.5).abs() < 1e-5);
    cycle.paused = true;
    cycle.advance(60.0);
    assert_eq!(cycle.time_hours, 0.5);
    cycle.paused = false;
    cycle.advance(f32::INFINITY);
    cycle.advance(-1.0);
    cycle.day_length_minutes = 0.0;
    cycle.advance(1.0);
    assert_eq!(cycle.time_hours, 0.5);
}
