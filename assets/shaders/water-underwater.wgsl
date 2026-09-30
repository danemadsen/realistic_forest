// Submerged-camera medium. Derived from bevy-aqua (MIT OR Apache-2.0):
// bevy-aqua-volume/src/volume.wgsl and bevy-aqua-medium/src/medium.wgsl. See
// src/water/ATTRIBUTION.md.
//
// Aqua runs this as a fullscreen `ViewNode` writing a volume texture that its
// material shaders then sample; the reduction here applies the medium straight
// to the composited frame, which is equivalent for a single camera whose whole
// frustum sits in one homogeneous medium, and skips the intermediate target.
//
// Applied in display-referred linear space: the frame handed in has already
// been through the composite's ACES curve and gamma encode, and this pass
// decodes it, attenuates and re-encodes. The medium therefore sits after the
// tone curve rather than before it, which for an absorbing medium differs from
// aqua in the highlight roll-off only — extinction and in-scatter are the same
// expressions.
//
// Not ported: caustics, god rays, bubble spray, and the post-process distortion
// from aqua's `Effects`. None of them are part of the medium.

struct GlobalUniforms
{
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
    camera_position: vec4<f32>,
    sun_direction: vec4<f32>,     // xyz direction sunlight travels, WORLD space
    viewport: vec4<f32>,          // xy size in pixels, zw inverse size
    params: vec4<f32>,            // x fog_density, y z_far, z exposure, w ssao_enabled
    settings_a: vec4<f32>,        // x sun_intensity, y texture_scale, z ao_tex_strength, w variant_scale
    settings_b: vec4<f32>,        // x normal_strength, y sparkle_strength, z flow_debug, w erosion_debug
};
@group(0) @binding(0) var<uniform> globals: GlobalUniforms;

struct UnderwaterUniforms
{
    params: vec4<f32>,      // x sea level, y camera height above it, z elapsed seconds, w fade
    extinction: vec4<f32>,  // rgb extinction /m, w scatter scale
    scatter: vec4<f32>,     // rgb scatter tint, w asymmetry
    sun: vec4<f32>,         // rgb sun radiance at the surface, w unused
};
@group(2) @binding(0) var<uniform> medium: UnderwaterUniforms;

@group(1) @binding(0) var scene_texture: texture_2d<f32>;
@group(1) @binding(8) var scene_sampler: sampler;
@group(1) @binding(1) var gbuffer_position: texture_2d<f32>;
@group(1) @binding(9) var gbuffer_sampler: sampler;

const PI: f32 = 3.141592653589793;
const PATH_LENGTH_MAX: f32 = 256.0;
const PARTICLE_SCATTER: f32 = 0.02;
const RAYLEIGH: vec3<f32> = vec3<f32>(0.00095, 0.00193, 0.00456);
const N_WATER: f32 = 1.333;

fn smoothstepf(edge0: f32, edge1: f32, x: f32) -> f32
{
    let t = clamp((x - edge0)/(edge1 - edge0), 0.0, 1.0);
    return t*t*(3.0 - 2.0*t);
}

/// The gradient itself, as a function of its own parameter: 0 at the horizon,
/// 1 at the zenith. Byte-for-byte the copy composite.wgsl draws with.
fn skyColour(parameter: f32) -> vec3<f32>
{
    let horizon = vec3<f32>(0.72, 0.82, 0.90);
    let zenith = vec3<f32>(0.22, 0.46, 0.74);
    return mix(horizon, zenith, parameter);
}

/// Sky gradient parameter for a ray whose sine of elevation is `direction_y`,
/// matching composite.wgsl's and water-surface.wgsl's.
fn skyParameter(direction_y: f32) -> f32
{
    return smoothstepf(0.0, 0.85, direction_y);
}

/// `skyColour` decoded into the linear space this pass lights in.
///
/// The gradient is authored display-referred — composite.wgsl decodes it with
/// `pow(sky, 2.2)` before use — and this shader decodes the incoming frame the
/// same way, so the light pouring through Snell's window has to be decoded too
/// or it lands about 2.5x too bright.
fn skyLinear(parameter: f32) -> vec3<f32>
{
    return pow(max(skyColour(parameter), vec3<f32>(0.0)), vec3<f32>(2.2));
}

/// Elevation sine of the representative sky that pours through Snell's window.
///
/// The window is not a view direction: it is the whole sky dome refracted into
/// a cone, so it wants ONE colour and stays a constant under the elevation
/// parameterisation. `0.85 * 0.85` is the old screen-row value carried across —
/// under a ramp of `smoothstep(0, 1, x)` a value `x` and the elevation sine
/// `0.85x` land on the same gradient parameter — so the window light is
/// unchanged. (The same literal 0.85 in water-surface.wgsl's `skyParameter` is
/// a ramp EDGE; here it is a sample point. Unrelated meanings, kept apart by
/// naming this one for what it is.)
const WINDOW_SKY_ELEVATION: f32 = 0.85 * 0.85;

fn henyeyGreenstein(cos_theta: f32, g: f32) -> f32
{
    let g2 = g*g;
    let denominator = 1.0 + g2 - 2.0*g*cos_theta;
    return (1.0 - g2)/(4.0*PI*pow(max(denominator, 1e-4), 1.5));
}

fn phaseRayleigh(cos_theta: f32) -> f32
{
    return (3.0/(16.0*PI))*(1.0 + cos_theta*cos_theta);
}

struct VsOutput
{
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex: u32) -> VsOutput
{
    var corner = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    var output: VsOutput;
    output.position = vec4<f32>(corner[vertex], 0.0, 1.0);
    return output;
}

/// World-space direction of the view ray through `uv`. Built from the
/// projection's own scale terms rather than an inverse matrix: for a symmetric
/// perspective, view space x/y are `ndc / (projection[0][0], projection[1][1])`
/// at unit depth. uv.y is top-down and view space y is up, so it flips.
///
/// View -> world is the TRANSPOSE of `globals.view`, which is world -> view and
/// whose rotation is orthonormal. A transpose is the dot product of the vector
/// with each ROW, and WGSL stores the matrix column-major, so row `i` is
/// `vec3(view[0][i], view[1][i], view[2][i])` — NOT `view[i].xyz`, which is
/// column `i`. The three dot products below are that transpose.
///
/// This used to read `v.x*view[0] + v.y*view[1] + v.z*view[2]`, which is the
/// other contraction — `view * v`, world -> view applied to a view-space
/// vector. For an orthonormal basis the two differ only by which index is
/// summed, so the error is a function of orientation rather than a constant,
/// and it is exactly zero in two orientations reachable by accident: pitch 0,
/// and yaw 180, where this matrix is symmetric and `view * v` coincides with
/// its transpose. Everywhere else it is severe — at yaw 90 it returned
/// (-1, 0, 0) for a true forward of (0.906, -0.423, 0), 155 degrees away, and
/// at yaw 0 pitch -25 it put the horizon on row 381 of 450 where the truth is
/// row 69. `to_view.y` drives the Snell window, so the window sat where the
/// camera was not pointing: at yaw 0, 90 and 270 the contraction never reached
/// the window's 0.5412 threshold at all, so the window was simply absent, and
/// it appeared only at yaw 180 where the matrix is symmetric.
fn rayDirection(uv: vec2<f32>) -> vec3<f32>
{
    let ndc = vec2<f32>(uv.x*2.0 - 1.0, 1.0 - uv.y*2.0);
    let view_direction = normalize(vec3<f32>(ndc.x/max(globals.projection[0][0], 1e-6),
                                             ndc.y/max(globals.projection[1][1], 1e-6),
                                             -1.0));
    return vec3<f32>(dot(globals.view[0].xyz, view_direction),
                     dot(globals.view[1].xyz, view_direction),
                     dot(globals.view[2].xyz, view_direction));
}

@fragment
fn fs_main(in: VsOutput) -> @location(0) vec4<f32>
{
    let uv = in.position.xy*globals.viewport.zw;
    let fade = clamp(medium.params.w, 0.0, 1.0);

    // Every sample up front: this shader branches on sampled data below and
    // WGSL forbids implicit-derivative sampling in non-uniform control flow.
    let scene = textureSample(scene_texture, scene_sampler, uv);
    let packed = textureSample(gbuffer_position, gbuffer_sampler, uv);
    if (fade <= 0.0)
    {
        return scene;
    }

    // Optical path: the world distance from the eye to whatever the ray hits,
    // capped at the medium's maximum. A sky ray leaves the medium, so it takes
    // the cap — that is what aqua does there too.
    var optical_path = PATH_LENGTH_MAX;
    if (packed.a >= 0.5)
    {
        // packed.xyz is the G-buffer's view-space position; its length is
        // already the eye distance.
        optical_path = min(length(packed.xyz), PATH_LENGTH_MAX);
    }

    let scene_linear = pow(max(scene.rgb, vec3<f32>(0.0)), vec3<f32>(2.2));
    let sigma_t = max(medium.extinction.xyz, vec3<f32>(1e-4));
    let sigma_s = min(sigma_t,
                      PARTICLE_SCATTER*medium.extinction.w*medium.scatter.xyz + RAYLEIGH);
    let transmittance = exp(-sigma_t*optical_path);

    // In-scattered sunlight. The eye is inside the medium, so the whole path is
    // the water column — there is nothing above the shaded point to attenuate
    // first, unlike the surface pass.
    let to_view = rayDirection(uv);
    let to_sun = normalize(-globals.sun_direction.xyz);
    let cos_theta = dot(to_view, -to_sun);
    let phase = mix(henyeyGreenstein(cos_theta, clamp(medium.scatter.w, -0.99, 0.99)),
                    phaseRayleigh(cos_theta), 0.5);
    let scattered = sigma_s*medium.sun.rgb*phase*(1.0 - transmittance)/sigma_t;

    // Snell's window: seen from below, the surface transmits only within a cone
    // of asin(1/n) of the vertical; outside it the surface mirrors the dark
    // water back. Inside it, the refracted sky pours through. The cone's cosine
    // is sqrt(1 - 1/n^2) = 0.661 at n = 1.333.
    let critical_cosine = sqrt(max(1.0 - 1.0/(N_WATER*N_WATER), 0.0));
    let window = smoothstepf(critical_cosine - 0.12, critical_cosine + 0.03,
                             max(to_view.y, 0.0));
    let window_light = skyLinear(skyParameter(WINDOW_SKY_ELEVATION))*globals.params.z;

    let lit = scene_linear*transmittance + scattered + window_light*window*0.28;
    let medium_result = pow(max(lit, vec3<f32>(0.0)), vec3<f32>(1.0/2.2));
    return vec4<f32>(mix(scene.rgb, clamp(medium_result, vec3<f32>(0.0), vec3<f32>(1.0)), fade),
                     1.0);
}
