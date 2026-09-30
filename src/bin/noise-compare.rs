//! Dev tool: dumps the quick-noise field the port uses for the terrain, so it
//! can be compared against the C++'s FastNoiseLite OpenSimplex2S field.
//!
//! Mirrors `generate_noise_samples` in src/noise.rs exactly, including the
//! `*0.5 + 0.5` remap, so the printed values are the same `[0, 1]` texels
//! `NoiseField::samples` holds and the GPU texture is uploaded from. The
//! remap matters when comparing: the raw quick-noise output spans [-1, 1],
//! and feeding a reader the raw field makes every downstream height come out
//! wrong by hundreds of metres.
//!
//! 1024*1024 lines, row-major with x fastest (`z * NOISE_RESOLUTION + x`).
//!
//! This is the tool that established the substitution is faithful: the C++
//! height function fed this field reproduces the port's probe grid to 0.01 m.

use quick_noise::simd::StaticArch;
use quick_noise::{BatchNoise, Fbm, Grid, Simplex};

const NOISE_RESOLUTION: usize = 1024;

fn main() {
    let mut samples = vec![0.0f32; NOISE_RESOLUTION * NOISE_RESOLUTION];
    let grid = Grid::<2, StaticArch>::new(NOISE_RESOLUTION, NOISE_RESOLUTION);
    BatchNoise::<2, Fbm, Simplex>::builder(grid.x_iter(), grid.y_iter())
        .seed(1337)
        .octaves(5)
        .frequency(0.0085)
        .lacunarity(2.02)
        .persistence(0.5)
        .fill(samples.as_mut_slice());
    for sample in samples.iter_mut() {
        *sample = *sample * 0.5 + 0.5;
    }
    for sample in &samples {
        println!("{:.6}", sample);
    }
}
