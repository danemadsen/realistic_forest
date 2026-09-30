// Submerged-camera medium. Derived from bevy-aqua (MIT OR Apache-2.0):
// bevy-aqua-volume/src/volume.wgsl and bevy-aqua-medium/src/medium.wgsl. See
// src/water/ATTRIBUTION.md.
//
// Aqua runs this as a fullscreen `ViewNode` writing a volume texture that its
// material shaders then sample; the reduction here applies the medium straight
// to the composited frame, which is equivalent for a single camera whose whole
// frustum sits in one homogeneous medium, and skips the intermediate target.
//
// The incoming composite has been exposed, tone-mapped and gamma encoded.
// Invert that transform before attenuation and apply it once to the final HDR
// medium result. The original radiance of clipped highlights is unrecoverable.
// Sunlight, moonlight and the sky through Snell's window follow the day clock.
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
    sun_colour: vec4<f32>,      // linear sunlight tint
    moon_direction: vec4<f32>,  // direction moonlight travels
    atmosphere: vec4<f32>,      // daylight, moon intensity, hours, volume strength
    raymarch: vec4<f32>,        // shadows, volumes, reflections, quality
    heightfield: vec4<f32>,     // world centre XZ, span, texel size
};
@group(0) @binding(0) var<uniform> globals: GlobalUniforms;

struct UnderwaterUniforms
{
    params: vec4<f32>,      // x sea level, y camera height above it, z elapsed seconds, w fade
    extinction: vec4<f32>,  // rgb extinction /m, w scatter scale
    scatter: vec4<f32>,     // rgb scatter tint, w asymmetry
    sun: vec4<f32>,         // rgb sun radiance at the surface, w moon radiance scale
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

// Shared linear atmosphere model. Kept byte-for-byte in the composite and
// water passes so reflected skies and the visible horizon agree.
fn skyRadiance(direction: vec3<f32>) -> vec3<f32> {
    let ray = normalize(direction);
    let to_sun = -normalize(globals.sun_direction.xyz);
    let to_moon = -normalize(globals.moon_direction.xyz);
    let daylight = clamp(globals.atmosphere.x, 0.0, 1.0);
    let elevation = smoothstep(0.0, 0.85, ray.y);
    let day_sky = mix(vec3<f32>(0.49, 0.64, 0.82),
                      vec3<f32>(0.045, 0.175, 0.43), elevation);
    let night_sky = mix(vec3<f32>(0.008, 0.013, 0.027),
                        vec3<f32>(0.0014, 0.0025, 0.008), elevation);
    var sky = mix(night_sky, day_sky, daylight);
    let sunset = exp(-pow((to_sun.y + 0.025)/0.17, 2.0));
    let horizon = exp(-abs(ray.y)*6.0);
    let sunward = pow(max(dot(normalize(vec3<f32>(ray.x, 0.001, ray.z)),
                              normalize(vec3<f32>(to_sun.x, 0.001, to_sun.z))), 0.0), 5.0);
    sky += vec3<f32>(0.65, 0.16, 0.025)*sunset*horizon*(0.15 + 0.85*sunward);

    let sun_mu = clamp(dot(ray, to_sun), -1.0, 1.0);
    let sun_visible = smoothstep(-0.018, 0.012, to_sun.y);
    let sun_disk = smoothstep(0.9999832, 0.9999905, sun_mu);
    let sun_halo = exp(-(1.0 - sun_mu)*750.0)*0.10
                   + pow(max(sun_mu, 0.0), 24.0)*0.035;
    // Direct ground irradiance fades before the visible solar disc sets.
    sky += globals.sun_colour.rgb*sun_visible
           *(sun_disk*14.0*globals.sun_colour.w + sun_halo*globals.settings_a.x);

    let moon_mu = clamp(dot(ray, to_moon), -1.0, 1.0);
    let moon_disk = smoothstep(0.999978, 0.999986, moon_mu);
    let moon_visible = smoothstep(-0.02, 0.03, to_moon.y)*(1.0 - daylight*0.9);
    sky += vec3<f32>(0.63, 0.74, 1.0)*moon_visible
           *(moon_disk*1.6 + exp(-(1.0 - moon_mu)*500.0)*0.025);

    // Stars occupy a world-oriented angular grid and remain stationary when
    // the camera rotates. Smooth small discs avoid binary sparkling.
    let star_uv = vec2<f32>(atan2(ray.z, ray.x), asin(clamp(ray.y, -1.0, 1.0)))*360.0;
    let star_cell = floor(star_uv);
    let star_hash = fract(sin(dot(star_cell, vec2<f32>(127.1, 311.7)))*43758.5453);
    let star_local = fract(star_uv) - vec2<f32>(0.5);
    let star_disc = 1.0 - smoothstep(0.035, 0.18, length(star_local));
    let star = step(0.997, star_hash)*star_disc*(0.25 + 0.75*star_hash);
    sky += vec3<f32>(0.6, 0.73, 1.0)*star*pow(1.0 - daylight, 4.0)
           *smoothstep(-0.03, 0.18, ray.y)*0.7;
    return max(sky, vec3<f32>(0.0));
}

// Hemisphere-integrated fill depends on the surface orientation and time of
// day, never its screen row or camera pitch. Excludes the celestial discs.
fn skyAmbient(normal_world: vec3<f32>) -> vec3<f32> {
    let daylight = clamp(globals.atmosphere.x, 0.0, 1.0);
    let upward = clamp(normal_world.y*0.5 + 0.5, 0.0, 1.0);
    let day_fill = mix(vec3<f32>(0.13, 0.17, 0.22),
                       vec3<f32>(0.32, 0.44, 0.62), upward);
    let night_fill = mix(vec3<f32>(0.007, 0.011, 0.022),
                         vec3<f32>(0.019, 0.029, 0.058), upward);
    return mix(night_fill, day_fill, daylight);
}

fn acesFilm(x: vec3<f32>) -> vec3<f32>
{
    const a: f32 = 2.51;
    const b: f32 = 0.03;
    const c: f32 = 2.43;
    const d: f32 = 0.59;
    const e: f32 = 0.14;
    return clamp((x*(a*x + b))/(x*(c*x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

// The composite has already applied exposure, ACES and gamma. Undo all three
// for both transmitted scene colours and reflected terrain, so neither is
// tone-mapped twice. Saturated pixels recover a finite radiance estimate.
fn sceneRadiance(encoded: vec3<f32>) -> vec3<f32>
{
    let y = clamp(pow(max(encoded, vec3<f32>(0.0)), vec3<f32>(2.2)),
                  vec3<f32>(0.0), vec3<f32>(0.999));
    let a = 2.51 - 2.43*y;
    let b = 0.03 - 0.59*y;
    let linear = (-b + sqrt(max(b*b + 0.56*a*y, vec3<f32>(0.0))))/(2.0*a);
    return linear/max(globals.params.z, 0.001);
}

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
    // capped at the medium's maximum and, for upward rays, at the sea surface.
    var optical_path = PATH_LENGTH_MAX;
    if (packed.a >= 0.5)
    {
        // packed.xyz is the G-buffer's view-space position; its length is
        // already the eye distance.
        optical_path = min(length(packed.xyz), PATH_LENGTH_MAX);
    }

    let to_view = rayDirection(uv);
    if (to_view.y > 0.001)
    {
        let surface_path = max(medium.params.x - globals.camera_position.y, 0.0)/to_view.y;
        optical_path = min(optical_path, surface_path);
    }
    let scene_linear = sceneRadiance(scene.rgb);
    let sigma_t = max(medium.extinction.xyz, vec3<f32>(1e-4));
    let sigma_s = min(sigma_t,
                      PARTICLE_SCATTER*medium.extinction.w*medium.scatter.xyz + RAYLEIGH);
    let transmittance = exp(-sigma_t*optical_path);

    // In-scattered sunlight. The eye is inside the medium, so the whole path is
    // the water column — there is nothing above the shaded point to attenuate
    // first, unlike the surface pass.
    let to_sun = normalize(-globals.sun_direction.xyz);
    // rayDirection points from the eye toward the scene, so forward
    // scattering peaks while looking toward the source.
    let cos_theta = dot(to_view, to_sun);
    let phase = mix(henyeyGreenstein(cos_theta, clamp(medium.scatter.w, -0.99, 0.99)),
                    phaseRayleigh(cos_theta), 0.5);
    let to_moon = normalize(-globals.moon_direction.xyz);
    let moon_phase = mix(henyeyGreenstein(dot(to_view, to_moon),
                        clamp(medium.scatter.w, -0.99, 0.99)),
                        phaseRayleigh(dot(to_view, to_moon)), 0.5);
    let moon_radiance = vec3<f32>(0.63, 0.74, 1.0)*medium.sun.w;
    let ambient = skyAmbient(vec3<f32>(0.0, 1.0, 0.0))*(0.35/(4.0*PI));
    let scattered = sigma_s*(medium.sun.rgb*phase + moon_radiance*moon_phase + ambient)
                   *(1.0 - transmittance)/sigma_t;

    // Snell's window: seen from below, the surface transmits only within a cone
    // of asin(1/n) of the vertical; outside it the surface mirrors the dark
    // water back. Inside it, the refracted sky pours through. The cone's cosine
    // is sqrt(1 - 1/n^2) = 0.661 at n = 1.333.
    let critical_cosine = sqrt(max(1.0 - 1.0/(N_WATER*N_WATER), 0.0));
    let window = smoothstepf(critical_cosine - 0.12, critical_cosine + 0.03,
                             max(to_view.y, 0.0));
    let window_direction = vec3<f32>(to_view.x*N_WATER,
                          sqrt(max(1.0 - N_WATER*N_WATER*(1.0 - to_view.y*to_view.y), 0.001)),
                          to_view.z*N_WATER);
    let window_light = skyRadiance(window_direction);

    let lit = scene_linear*transmittance + scattered + window_light*window*0.28;
    let medium_result = pow(acesFilm(lit*globals.params.z), vec3<f32>(1.0/2.2));
    return vec4<f32>(mix(scene.rgb, clamp(medium_result, vec3<f32>(0.0), vec3<f32>(1.0)), fade),
                     1.0);
}
