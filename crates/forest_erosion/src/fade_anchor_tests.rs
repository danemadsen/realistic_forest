use super::*;

#[test]
fn the_fade_anchor_moves_the_fade_by_under_two_decimetres() {
    // SplitMix64, so the positions are the same on every run.
    let mut state = 0xfade_a9c4_u64;
    let mut unit = move || {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        ((z ^ (z >> 31)) >> 40) as f32 / (1u64 << 24) as f32
    };
    let farthest = (EROSION_FADE_SNAP * 0.5) * std::f32::consts::SQRT_2;
    let mut most = 0.0f32;
    for _ in 0..10_000 {
        let player = [(unit() - 0.5) * 40_000.0, (unit() - 0.5) * 40_000.0];
        let anchor = erosion_fade_anchor(player);
        // On the lattice, and the nearest lattice point to the player.
        assert_eq!(
            anchor.map(|value| value % EROSION_FADE_SNAP == 0.0),
            [true; 2],
            "{player:?}"
        );
        let offset = ((anchor[0] - player[0]).powi(2) + (anchor[1] - player[1]).powi(2)).sqrt();
        assert!(offset <= farthest + 0.01, "{player:?} -> {anchor:?}");
        // The fade around the anchor against the fade around the player,
        // on rings through the band where it falls, and inside it.
        for ring in 0..=50 {
            let radius = EROSION_VISIBILITY_FULL_RADIUS - 10.0 + ring as f32 * 10.4;
            let angle = unit() * std::f32::consts::TAU;
            let x = player[0] + radius * angle.cos();
            let z = player[1] + radius * angle.sin();
            let change =
                (erosion_visibility(x, z, anchor) - erosion_visibility(x, z, player)).abs();
            most = most.max(change);
            assert!(change <= 0.0171, "{player:?} at {radius} m: {change}");
        }
        // Nearer than the band less the offset, the fade is whole either way.
        let near = EROSION_VISIBILITY_FULL_RADIUS - farthest - 0.5;
        assert_eq!(erosion_visibility(player[0] + near, player[1], anchor), 1.0);
    }
    // 0.017 of the 11 m deltas the prewarm reports is 0.19 m.
    println!("the anchor changes the fade by {most:.4} at most");
}
