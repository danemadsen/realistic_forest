use super::*;

const STRIKE: [f32; 3] = [1200.0, 35.0, -800.0];
const TOP: [f32; 3] = [1400.0, 1350.0, -650.0];

fn main_channel(segments: &[BoltSegment]) -> Vec<BoltSegment> {
    let all_segments = segments.iter().copied();
    all_segments.filter(|segment| segment.order == 0).collect()
}

#[test]
fn ground_channel_runs_from_the_cloud_to_exactly_the_strike_point() {
    for seed in 1..60 {
        let seed = seed as f32 / 61.0;
        let segments = ground_bolt(seed, STRIKE, TOP);
        assert!(segments.len() <= MAX_BOLT_SEGMENTS);
        let main = main_channel(&segments);
        assert!(main.len() > 20, "seed {seed}: {} main segments", main.len());
        assert_eq!(main[0].start, TOP);
        assert_eq!(main.last().unwrap().end, STRIKE);
        for pair in main.windows(2) {
            assert_eq!(pair[0].end, pair[1].start, "the channel must be continuous");
            assert!(pair[0].arrival[1] <= pair[1].arrival[1] + 1e-6);
        }
        assert!((main.last().unwrap().arrival[1] - 1.0).abs() < 1e-6);
    }
}

#[test]
fn bolts_are_deterministic_per_seed_and_differ_between_seeds() {
    assert_eq!(
        ground_bolt(0.37, STRIKE, TOP),
        ground_bolt(0.37, STRIKE, TOP)
    );
    assert_ne!(
        ground_bolt(0.37, STRIKE, TOP),
        ground_bolt(0.38, STRIKE, TOP)
    );
    assert_eq!(
        crawler_bolt(0.37, STRIKE, 1000.0),
        crawler_bolt(0.37, STRIKE, 1000.0)
    );
}

#[test]
fn ground_channel_is_tortuous_but_stays_near_its_column() {
    let height = TOP[1] - STRIKE[1];
    for seed in 1..60 {
        let main = main_channel(&ground_bolt(seed as f32 / 61.0, STRIKE, TOP));
        let path: f32 = main
            .iter()
            .map(|segment| Vec3::from(segment.start).distance(Vec3::from(segment.end)))
            .sum();
        let straight = Vec3::from(TOP).distance(Vec3::from(STRIKE));
        let tortuosity = path / straight;
        assert!((1.1..2.6).contains(&tortuosity), "tortuosity {tortuosity}");
        for segment in &main {
            let drift =
                Vec3::new(segment.end[0] - STRIKE[0], 0.0, segment.end[2] - STRIKE[2]).length();
            assert!(drift < 0.9 * height, "drift {drift}");
        }
    }
}

#[test]
fn ground_bolts_branch_and_branches_stall_above_the_ground() {
    let mut branched = 0;
    for seed in 1..60 {
        let segments = ground_bolt(seed as f32 / 61.0, STRIKE, TOP);
        let branches: Vec<_> = segments
            .iter()
            .filter(|segment| segment.order > 0)
            .collect();
        if !branches.is_empty() {
            branched += 1;
        }
        for branch in branches {
            assert!(
                branch.end[1] >= STRIKE[1] + 4.0,
                "branch reaches the ground"
            );
            assert!(branch.brightness < 1.0 && branch.width < 1.0);
            assert!(branch.arrival[0] > 0.0);
        }
    }
    assert!(branched > 45, "most bolts should fork: {branched}");
}

#[test]
fn crawlers_stay_under_the_cloud_base() {
    for seed in 1..40 {
        let segments = crawler_bolt(seed as f32 / 41.0, STRIKE, 1000.0);
        assert!(segments.len() > 10 && segments.len() <= MAX_BOLT_SEGMENTS);
        for segment in segments {
            for point in [segment.start, segment.end] {
                assert!(
                    point[1] <= 1000.0 && point[1] >= 760.0,
                    "crawler at {}",
                    point[1]
                );
            }
        }
    }
}
