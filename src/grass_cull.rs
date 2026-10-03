//! Conservative visibility tests for the grass field's square chunks.
//!
//! The grass vertex shader already rejects instances one at a time (beyond the
//! draw radius, beyond their layer's fade, thinned out, unsuitable ground), but
//! every instance still costs a vertex-shader invocation per vertex to find
//! that out. These tests drop whole chunks on the CPU first, so the GPU only
//! sees instances that can reach the screen.
//!
//! Both tests are deliberately *looser* than what the shader decides, never
//! tighter: a chunk is dropped only when no instance rooted in it can produce a
//! pixel. The ground height is unknown on the CPU (it is read from the habitat
//! capture on the GPU), so the view test works on the frustum's footprint on
//! the ground plane, which holds for any terrain height.

use std::f32::consts::{PI, TAU};
use std::ops::Range;

/// Layers 0 (base) to 3 (foreground carpet): see `GrassInstance::_pad`.
pub const LAYER_COUNT: usize = 4;

/// A square grid of equally sized chunks, row-major (`z * per_side + x`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChunkGrid {
    /// World XZ of the grid's minimum corner.
    pub min: [f32; 2],
    pub chunk_size: f32,
    pub per_side: usize,
}

impl ChunkGrid {
    pub fn count(&self) -> usize {
        self.per_side * self.per_side
    }

    pub fn centre(&self, index: usize) -> [f32; 2] {
        let (x, z) = (index % self.per_side, index / self.per_side);
        [
            self.min[0] + (x as f32 + 0.5) * self.chunk_size,
            self.min[1] + (z as f32 + 0.5) * self.chunk_size,
        ]
    }

    /// Radius of a circle around `centre` holding every root of the chunk. The
    /// metre of slack covers an f32 root rounding onto the neighbouring cell
    /// boundary far from the origin.
    pub fn radius(&self) -> f32 {
        self.chunk_size * std::f32::consts::FRAC_1_SQRT_2 + 1.0
    }
}

/// The camera as the chunk test needs it: where it stands and which compass
/// directions its frustum can reach on the ground plane.
#[derive(Clone, Copy, Debug)]
pub struct Footprint {
    camera: [f32; 2],
    /// Compass angle (`atan2(z, x)`) of the middle of the footprint.
    heading: f32,
    /// Half the angular width of the footprint; `PI` means every direction.
    half_angle: f32,
}

impl Footprint {
    /// From the column-major view and projection matrices the frame renders
    /// with. Works for any orientation, roll included.
    pub fn new(camera: [f32; 2], view: &[f32; 16], projection: &[f32; 16]) -> Self {
        let everywhere = Self { camera, heading: 0.0, half_angle: PI };
        let (tan_x, tan_y) = (1.0 / projection[0], 1.0 / projection[5]);
        if !(tan_x.is_finite() && tan_y.is_finite() && tan_x > 0.0 && tan_y > 0.0) {
            return everywhere;
        }
        // The rows of the view rotation are the camera's axes in world space;
        // the camera looks down its own -Z.
        let axis = |row: usize| [view[row], view[4 + row], view[8 + row]];
        let (right, up, back) = (axis(0), axis(1), axis(2));
        // Straight up or down lies inside the frustum's cone exactly when the
        // world Y axis, in camera coordinates, points through the screen. Then
        // the footprint wraps around the camera and nothing can be dropped.
        let (x, y, z) = (view[4], view[5], view[6]);
        if x.abs() <= z.abs() * tan_x && y.abs() <= z.abs() * tan_y {
            return everywhere;
        }
        // Otherwise the footprint is the angular span of the cone's four edges.
        let ray = |sx: f32, sy: f32| -> [f32; 2] {
            let (cx, cy) = (sx * tan_x, sy * tan_y);
            [
                cx * right[0] + cy * up[0] - back[0],
                cx * right[2] + cy * up[2] - back[2],
            ]
        };
        let forward = [-back[0], -back[2]];
        if forward[0].hypot(forward[1]) < 1e-4 {
            return everywhere;
        }
        let reference = forward[1].atan2(forward[0]);
        let (mut low, mut high) = (0.0f32, 0.0f32);
        for (sx, sy) in [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)] {
            let edge = ray(sx, sy);
            if edge[0].hypot(edge[1]) < 1e-4 {
                return everywhere;
            }
            let offset = wrap(edge[1].atan2(edge[0]) - reference);
            low = low.min(offset);
            high = high.max(offset);
        }
        // A hair of slack for f32 error in the matrices and in `atan2`.
        const SLACK: f32 = 0.01;
        let half_angle = 0.5 * (high - low) + SLACK;
        if half_angle >= PI {
            return everywhere;
        }
        Self {
            camera,
            heading: reference + 0.5 * (high + low),
            half_angle,
        }
    }

    /// Could any point within `radius` of `centre`, at any height, be on screen?
    pub fn may_see(&self, centre: [f32; 2], radius: f32) -> bool {
        if self.half_angle >= PI {
            return true;
        }
        let offset = [centre[0] - self.camera[0], centre[1] - self.camera[1]];
        let distance = offset[0].hypot(offset[1]);
        if distance <= radius {
            return true;
        }
        let bearing = wrap(offset[1].atan2(offset[0]) - self.heading).abs();
        // The disk spans this much of the sky as seen from the camera.
        bearing - (radius / distance).asin() <= self.half_angle
    }
}

/// Fold an angle into (-PI, PI].
fn wrap(angle: f32) -> f32 {
    let folded = (angle + PI).rem_euclid(TAU) - PI;
    if folded <= -PI { PI } else { folded }
}

/// Distance from `camera` past which nothing of a layer is drawn, and the
/// reach of a blade away from its root in any direction. Both come from the
/// shader's own rules; see `grass.wgsl`.
pub struct Reach {
    pub layer_end: [f32; LAYER_COUNT],
    pub blade_extent: f32,
}

/// For every layer, the runs of consecutive chunk indices that can contribute a
/// pixel this frame. Consecutive chunks are consecutive in the instance
/// buffers too, so each run is one draw range.
pub fn visible_runs(
    grid: &ChunkGrid,
    footprint: &Footprint,
    reach: &Reach,
) -> [Vec<Range<u32>>; LAYER_COUNT] {
    let mut runs: [Vec<Range<u32>>; LAYER_COUNT] = Default::default();
    let radius = grid.radius();
    let blade_radius = radius + reach.blade_extent;
    let farthest = reach.layer_end.iter().copied().fold(0.0, f32::max);
    let mut open: [Option<u32>; LAYER_COUNT] = [None; LAYER_COUNT];
    for chunk in 0..=grid.count() {
        let mut visible = [false; LAYER_COUNT];
        if chunk < grid.count() {
            let centre = grid.centre(chunk);
            let distance =
                (centre[0] - footprint.camera[0]).hypot(centre[1] - footprint.camera[1]);
            // The shader measures to the root, so the chunk is out of a layer's
            // reach once its nearest possible root is.
            let nearest = distance - radius;
            if nearest < farthest && footprint.may_see(centre, blade_radius) {
                for (flag, end) in visible.iter_mut().zip(reach.layer_end) {
                    *flag = nearest < end;
                }
            }
        }
        for ((&seen, slot), layer_runs) in visible.iter().zip(&mut open).zip(&mut runs) {
            match (seen, *slot) {
                (true, None) => *slot = Some(chunk as u32),
                (false, Some(start)) => {
                    layer_runs.push(start..chunk as u32);
                    *slot = None;
                }
                _ => {}
            }
        }
    }
    runs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrices::{perspective, view_matrix};
    use bevy::math::Vec3;

    /// A small deterministic generator, so failures reproduce.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 40) as f32) / 16_777_216.0
        }
        fn range(&mut self, low: f32, high: f32) -> f32 {
            low + (high - low) * self.next()
        }
    }

    fn camera(yaw: f32, pitch: f32, roll: f32) -> ([f32; 16], [f32; 16]) {
        let forward = Vec3::new(
            yaw.sin() * pitch.cos(),
            pitch.sin(),
            -yaw.cos() * pitch.cos(),
        );
        let up = (Vec3::Y * roll.cos() + forward.cross(Vec3::Y).normalize_or_zero() * roll.sin())
            .normalize();
        let eye = Vec3::new(10.0, 3.0, -4.0);
        (
            view_matrix(eye, eye + forward, up),
            perspective(68.0, 16.0 / 9.0, 0.1, 5800.0),
        )
    }

    /// Point at camera-space `(x, y)` in units of `-z`, `depth` metres out.
    fn point_in_view(view: &[f32; 16], x: f32, y: f32, depth: f32, tan: (f32, f32)) -> [f32; 3] {
        let axis = |row: usize| Vec3::new(view[row], view[4 + row], view[8 + row]);
        let eye = Vec3::new(10.0, 3.0, -4.0);
        let p = eye + axis(0) * (x * tan.0 * depth) + axis(1) * (y * tan.1 * depth) - axis(2) * depth;
        [p.x, p.y, p.z]
    }

    #[test]
    fn nothing_the_camera_sees_is_ever_dropped() {
        let mut rng = Lcg(7);
        let projection = perspective(68.0, 16.0 / 9.0, 0.1, 5800.0);
        let tan = (1.0 / projection[0], 1.0 / projection[5]);
        for _ in 0..4000 {
            let (yaw, pitch, roll) = (
                rng.range(-PI, PI),
                rng.range(-1.55, 1.55),
                if rng.next() < 0.5 { 0.0 } else { rng.range(-0.6, 0.6) },
            );
            let (view, projection) = camera(yaw, pitch, roll);
            let footprint = Footprint::new([10.0, -4.0], &view, &projection);
            for _ in 0..40 {
                let p = point_in_view(
                    &view,
                    rng.range(-1.0, 1.0),
                    rng.range(-1.0, 1.0),
                    rng.range(0.1, 300.0),
                    tan,
                );
                assert!(
                    footprint.may_see([p[0], p[2]], 0.5),
                    "dropped a visible point: yaw {yaw} pitch {pitch} roll {roll} point {p:?}"
                );
            }
        }
    }

    #[test]
    fn a_level_camera_drops_what_is_behind_it() {
        let (view, projection) = camera(0.0, 0.0, 0.0); // looking along -Z
        let footprint = Footprint::new([10.0, -4.0], &view, &projection);
        assert!(footprint.may_see([10.0, -104.0], 5.0));
        assert!(!footprint.may_see([10.0, 96.0], 5.0));
        assert!(!footprint.may_see([210.0, -4.0], 5.0));
        // Looking straight down sees all around: nothing can be dropped.
        let (view, projection) = camera(0.0, -PI / 2.0 + 0.02, 0.0);
        let footprint = Footprint::new([10.0, -4.0], &view, &projection);
        assert!(footprint.may_see([10.0, 96.0], 5.0));
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn runs_cover_exactly_the_visible_chunks_in_reach() {
        let grid = ChunkGrid { min: [-240.0, -240.0], chunk_size: 16.0, per_side: 30 };
        let (view, projection) = camera(0.0, -0.18, 0.0);
        let footprint = Footprint::new([0.0, 0.0], &view, &projection);
        let reach = Reach { layer_end: [235.0, 200.0, 140.0, 33.0], blade_extent: 1.0 };
        let runs = visible_runs(&grid, &footprint, &reach);
        for layer in 0..LAYER_COUNT {
            let mut covered = vec![false; grid.count()];
            for run in &runs[layer] {
                assert!(run.start < run.end);
                for chunk in run.clone() {
                    covered[chunk as usize] = true;
                }
            }
            for chunk in 0..grid.count() {
                let centre = grid.centre(chunk);
                let distance = centre[0].hypot(centre[1]);
                // Anything that could hold a root inside the layer's reach and
                // that the view can see must be drawn.
                if distance - grid.radius() < reach.layer_end[layer]
                    && footprint.may_see(centre, grid.radius() + reach.blade_extent)
                {
                    assert!(covered[chunk], "layer {layer} chunk {chunk}");
                }
                // Nothing is drawn past the layer's reach.
                if distance - grid.radius() >= reach.layer_end[layer] {
                    assert!(!covered[chunk], "layer {layer} chunk {chunk}");
                }
            }
        }
        // The carpet is a small disc, the base layer a large one.
        let total = |layer: usize| runs[layer].iter().map(|r| r.len()).sum::<usize>();
        assert!(total(3) < total(2) && total(2) < total(1) && total(1) <= total(0));
        // A level 100-degree view keeps well under half of the chunks.
        assert!(total(0) < grid.count() * 45 / 100, "{}", total(0));
    }
}
