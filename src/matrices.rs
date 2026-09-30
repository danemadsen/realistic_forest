//! Camera matrix construction, matching the C++ project's raylib camera with
//! the GL clip space rewritten for wgpu's [0, 1] depth range.

use bevy::math::{Vec2, Vec3};

/// Right-handed look-at view matrix; camera forward is -Z. Matches
/// raylib's MatrixLookAt for a camera built with position/target/up.
pub fn view_matrix(eye: Vec3, target: Vec3, up: Vec3) -> [f32; 16] {
    let z_axis = (eye - target).normalize_or_zero(); // forward is -z
    let x_axis = up.cross(z_axis).normalize_or_zero();
    let y_axis = z_axis.cross(x_axis);
    // Column-major storage, translation in columns 12..15 (WGSL mat4x4 layout).
    [
        x_axis.x, y_axis.x, z_axis.x, 0.0,
        x_axis.y, y_axis.y, z_axis.y, 0.0,
        x_axis.z, y_axis.z, z_axis.z, 0.0,
        -x_axis.dot(eye), -y_axis.dot(eye), -z_axis.dot(eye), 1.0,
    ]
}

/// Perspective projection matching raylib's `MatrixPerspective(fov_y, aspect,
/// near, far)` as fed by `BeginMode3D`, but with **reverse-Z** depth: the near
/// plane maps to depth 1 and the far plane to 0, which is what wgpu's [0, 1]
/// depth range wants when the buffer is `Depth32Float`.
///
/// DIVERGENCE FROM THE C++: raylib uses a GL-style forward projection, and the
/// port reproduced it exactly. That projection is the reason the coastal plains
/// tore apart. Forward [0, 1] depth spends its precision near the eye: with
/// `NEAR_PLANE` 0.1 and `FAR_PLANE` 5800, the resolvable step is already ~2.4 m
/// at 2 km and ~20 m at 5.8 km, so terrain within a few metres of the ocean
/// surface z-fights across its entire visible extent. Float depth is
/// distributed uniformly in *reciprocal* z, so reversing the mapping puts the
/// dense end of the float range at the far plane instead, giving a relative
/// resolution of about 1e-7 of the view distance — sub-millimetre at 2 km.
///
/// Column-major, eye looks down -Z. The -1 that turns w_clip into -z sits in
/// column 2 (index 11 = M[3][2]); M[3][3] stays 0. Putting -1 at index 15
/// makes w_clip a constant -1, which drives every vertex outside the clip
/// volume.
pub fn perspective(fov_y_degrees: f32, aspect: f32, near: f32, far: f32) -> [f32; 16] {
    let fov = fov_y_degrees.to_radians();
    let w = 1.0 / (0.5 * fov).tan();
    // Reverse-Z terms: z_clip = (near/(far - near))*z_eye + far*near/(far - near)
    // against w_clip = -z_eye, so z_ndc(-near) = 1 and z_ndc(-far) = 0.
    let span = far - near;
    [
        w / aspect, 0.0, 0.0, 0.0,
        0.0, w, 0.0, 0.0,
        0.0, 0.0, near / span, -1.0,
        0.0, 0.0, far * near / span, 0.0,
    ]
}

/// Inverse of an affine column-major 4x4 (bottom row 0, 0, 0, 1): the upper-
/// left 3x3 is inverted through its adjugate and the translation is rotated by
/// that inverse. The original GLSL computed these inverses in the shader
/// (`inverse(mat3(matView*matModel))` and `inverse(mat3(matModel))`, both 3x3
/// adjugate formulas), so this uses the same 3x3 adjugate rather than a 4x4
/// cofactor expansion — for an affine matrix the two agree, and the 3x3 form
/// reproduces the GLSL rounding exactly. A degenerate matrix (determinant 0)
/// yields the identity, matching raylib's `MatrixInvert` guard.
pub fn invert_affine(m: &[f32; 16]) -> [f32; 16] {
    // Column-major: m[col * 4 + row]; the upper-left 3x3 rotates/scales and
    // column 3 holds the translation.
    let m00 = m[0]; let m01 = m[4]; let m02 = m[8];
    let m10 = m[1]; let m11 = m[5]; let m12 = m[9];
    let m20 = m[2]; let m21 = m[6]; let m22 = m[10];
    let determinant = m00 * (m11 * m22 - m12 * m21)
        - m01 * (m10 * m22 - m12 * m20)
        + m02 * (m10 * m21 - m11 * m20);
    if determinant == 0.0 {
        return IDENTITY;
    }
    let inverse_determinant = 1.0 / determinant;
    // Adjugate of the upper-left 3x3, already divided by the determinant.
    let n00 = (m11 * m22 - m12 * m21) * inverse_determinant;
    let n01 = (m02 * m21 - m01 * m22) * inverse_determinant;
    let n02 = (m01 * m12 - m02 * m11) * inverse_determinant;
    let n10 = (m12 * m20 - m10 * m22) * inverse_determinant;
    let n11 = (m00 * m22 - m02 * m20) * inverse_determinant;
    let n12 = (m02 * m10 - m00 * m12) * inverse_determinant;
    let n20 = (m10 * m21 - m11 * m20) * inverse_determinant;
    let n21 = (m01 * m20 - m00 * m21) * inverse_determinant;
    let n22 = (m00 * m11 - m01 * m10) * inverse_determinant;
    let tx = m[12];
    let ty = m[13];
    let tz = m[14];
    [
        n00, n10, n20, 0.0,
        n01, n11, n21, 0.0,
        n02, n12, n22, 0.0,
        -(n00 * tx + n01 * ty + n02 * tz),
        -(n10 * tx + n11 * ty + n12 * tz),
        -(n20 * tx + n21 * ty + n22 * tz),
        1.0,
    ]
}

/// The identity matrix in the same column-major layout as the other helpers.
pub const IDENTITY: [f32; 16] = [
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
];

/// Multiply two column-major 4x4 matrices (a * b).
pub fn mul_m4(a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
    let mut out = [0.0f32; 16];
    for col in 0..4 {
        for row in 0..4 {
            let mut sum = 0.0;
            for k in 0..4 {
                sum += a[k * 4 + row] * b[col * 4 + k];
            }
            out[col * 4 + row] = sum;
        }
    }
    out
}

/// Project a world point through view/clip for CPU-side screen queries.
/// Returns (x, y) in framebuffer pixels (y down from top), plus the clip-space
/// z for behind-camera tests; `None` when the point is behind the near plane.
pub fn world_to_screen(
    point: Vec3,
    view: &[f32; 16],
    projection: &[f32; 16],
    viewport: (u32, u32),
) -> Option<Vec2> {
    let clip = mul_m4(projection, view);
    let x = point.x;
    let y = point.y;
    let z = point.z;
    let cx = clip[0] * x + clip[4] * y + clip[8] * z + clip[12];
    let cy = clip[1] * x + clip[5] * y + clip[9] * z + clip[13];
    let cw = clip[3] * x + clip[7] * y + clip[11] * z + clip[15];
    if cw <= 0.0 {
        return None;
    }
    let ndc_x = cx / cw;
    let ndc_y = cy / cw;
    // Raylib DrawLineV uses top-left origin screen coordinates.
    let screen_x = (ndc_x * 0.5 + 0.5) * viewport.0 as f32;
    let screen_y = (0.5 - ndc_y * 0.5) * viewport.1 as f32;
    Some(Vec2::new(screen_x, screen_y))
}