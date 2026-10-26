//! Deterministic spatial hashes and noise shared by world generators.

pub fn mix64(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

/// A stable 64-bit identity for an integer lattice cell of one layer.
pub fn cell_key(x: i64, z: i64, salt: u64) -> u64 {
    mix64(
        (x as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
            ^ (z as u64).wrapping_mul(0xd1b5_4a32_d192_ed03)
            ^ salt.wrapping_mul(0x632b_e59b_d9b4_e019),
    )
}

/// Uniform [0, 1) from a key and a channel.
pub fn random(key: u64, channel: u64) -> f32 {
    (mix64(key ^ channel.wrapping_mul(0x9e37_79b9_7f4a_7c15)) >> 40) as f32 / 16_777_216.0
}

fn lattice_hash(x: i64, z: i64, seed: u32) -> u32 {
    (cell_key(x, z, seed as u64 ^ 0x5eed) >> 32) as u32
}

/// Eight unit gradients; enough for smooth, isotropic-looking landscape fields.
const GRADIENTS: [[f32; 2]; 8] = [
    [1.0, 0.0],
    [0.707_106_77, 0.707_106_77],
    [0.0, 1.0],
    [-0.707_106_77, 0.707_106_77],
    [-1.0, 0.0],
    [-0.707_106_77, -0.707_106_77],
    [0.0, -1.0],
    [0.707_106_77, -0.707_106_77],
];

fn quintic(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

/// Gradient noise remapped to roughly [0, 1] (mean 0.5, sd about 0.13).
/// Coordinates are in lattice units; f64 keeps distant worlds exact.
pub fn gradient_noise(x: f64, z: f64, seed: u32) -> f32 {
    let cell_x = x.floor();
    let cell_z = z.floor();
    let fx = (x - cell_x) as f32;
    let fz = (z - cell_z) as f32;
    let ix = cell_x as i64;
    let iz = cell_z as i64;
    let corner = |dx: i64, dz: i64| {
        let g = GRADIENTS[(lattice_hash(ix + dx, iz + dz, seed) >> 29) as usize];
        g[0] * (fx - dx as f32) + g[1] * (fz - dz as f32)
    };
    let u = quintic(fx);
    let v = quintic(fz);
    let south = corner(0, 0) + (corner(1, 0) - corner(0, 0)) * u;
    let north = corner(0, 1) + (corner(1, 1) - corner(0, 1)) * u;
    let value = south + (north - south) * v;
    (0.5 + value * 0.72).clamp(0.0, 1.0)
}

/// Noise over world metres at a given wavelength, rotated per seed so no two
/// fields share lattice axes.
pub fn noise(p: [f64; 2], wavelength: f64, seed: u32) -> f32 {
    let angle = (seed as f64 * 0.618_034) * std::f64::consts::TAU;
    let (s, c) = angle.sin_cos();
    let x = (p[0] * c - p[1] * s) / wavelength;
    let z = (p[0] * s + p[1] * c) / wavelength;
    gradient_noise(x + seed as f64 * 17.13, z - seed as f64 * 9.71, seed)
}

/// Two-octave noise, still in [0, 1].
pub fn noise2(p: [f64; 2], wavelength: f64, seed: u32) -> f32 {
    0.67 * noise(p, wavelength, seed) + 0.33 * noise(p, wavelength * 0.43, seed.wrapping_add(101))
}

pub fn smoothstep(edge0: f32, edge1: f32, value: f32) -> f32 {
    let t = ((value - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}
