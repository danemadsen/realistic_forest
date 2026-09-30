// Ocean surface. Derived from bevy-aqua (MIT OR Apache-2.0): the Gerstner sum
// is bevy-aqua-waves/src/anim_waves.wgsl, the Fresnel is
// bevy-aqua-optics/src/optics.wgsl (godot_fresnel), the body radiance is
// bevy-aqua-medium/src/medium.wgsl, and the foam/SSS/composition is
// bevy-aqua-core/src/cascade/material.wgsl. See src/water/ATTRIBUTION.md.
//
// DIVERGENCE FROM AQUA, and why:
//
// - Aqua is a bevy_pbr material on Bevy 0.19 and imports bevy_pbr's mesh view
//   bindings, light probes, clustered lights, shadow maps, depth prepass and
//   transmission texture. This renderer has none of those. The water is drawn
//   as a forward pass between the composite and FXAA nodes, and it reads this
//   project's own deferred G-buffer instead of a depth prepass.
// - Aqua bakes the spectrum into a five-layer cascade texture array and
//   samples it, giving each cascade LOD only the wavelength band it newly
//   resolves (see `ranges` in src/water/waves.rs). Here the 40-component sum
//   is evaluated per vertex and every component is faded by the pixel's own
//   world footprint, so the visible wave content is a continuous function of
//   screen position. That is what removes the seam a discrete per-LOD band
//   would otherwise show at a ring boundary; `ranges` is uploaded for parity
//   but not read.
// - Aqua's shoaling reads a BedHeightMap texture. The optical path here comes
//   from the G-buffer: the distance from the water surface to whatever the ray
//   actually hits. That is the *real* path rather than a sampled approximation
//   of it, so the body tint and the transmitted background are both driven by
//   measured geometry. It cannot feed the vertex-stage shoaling term, which is
//   the one aqua feature this port gives up.
//   Foam deliberately does NOT use it. Depth arrives back out of the G-buffer
//   stair-stepped, so a threshold on it draws straight, constant-profile
//   boundaries through the water, and it cannot tell a calm shallow flat from
//   surf — which is exactly the mistake it made here. Foam is a property of the
//   surface's own compression instead, and needs no depth at all.
// - The reflected sky is this renderer's own skyColour() gradient, and the
//   fog/tonemap match composite.wgsl so water dissolves into the same haze as
//   the terrain behind it. Aqua uses an environment cubemap and an HDR target.
//   Two conventions have to be matched for that to be true rather than
//   approximate. The first is the gradient's *value*: it is display-encoded,
//   which the composite decodes with `pow(sky, 2.2)` before lighting, and
//   `skyLinear` below applies the same decode.
//   The second is its *parameter*, which used to be deliberately mismatched
//   and now is not. composite.wgsl once indexed the gradient by screen row —
//   `skyColour(uv.x, 1 - uv.y)` — which puts the zenith at the top of the
//   frame whatever the camera pitch and makes the horizon band unreachable.
//   Neither pass can use that: for the reflection the reflected ray leaves the
//   water pointing up, its screen projection clamps, and every water pixel
//   pinned to the zenith end of the ramp; for the drawn sky a camera pitched
//   down never shows the pale band at all. Both now index by the ray's own
//   elevation through `skyParameter`, the same function with the same 0.85
//   edge, so a water pixel mirrors exactly the sky the composite would have
//   painted in that direction.
//
// WGSL CONSTRAINT: `textureSample` uses implicit derivatives and is illegal in
// non-uniform control flow, and this shader branches on sampled data (the
// G-buffer depth test). Every texture read is therefore taken unconditionally
// up front and the branches only select between values already fetched — the
// same hoisting composite.wgsl documents.

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

struct GpuWave
{
    direction: vec2<f32>,
    amplitude: f32,
    wave_number: f32,
    angular_frequency: f32,
    phase: f32,
    chop_amplitude: f32,
    wavelength: f32,
};

const WAVE_SLOTS: u32 = 40u;
const LOD_COUNT: u32 = 5u;

struct WaterStageUniforms
{
    waves: array<GpuWave, WAVE_SLOTS>,
    ranges: array<vec4<u32>, LOD_COUNT>,   // per-LOD bands; see the note above
    params: vec4<f32>,        // x time, y amplitude multiplier, z sea level, w chop scale
    extinction: vec4<f32>,    // rgb extinction /m, w scatter scale
    scatter: vec4<f32>,       // rgb scatter tint, w asymmetry
    surface: vec4<f32>,       // x fresnel F0, y fresnel exponent, z sun roughness, w base scale
    sss_tint: vec4<f32>,      // rgb subsurface tint, w tile resolution
    misc: vec4<f32>,          // x refraction scale, y foam scale, z max amplitude, w unused
    flags: vec4<f32>,         // x flat-surface debug, y sea state amplitude, z wind radians, w wave fade end
};
@group(2) @binding(0) var<uniform> stage: WaterStageUniforms;

// The composited frame (refraction source) and the G-buffer's view-space
// position, whose alpha is 0 for sky and >= 1 for geometry.
@group(1) @binding(0) var scene_texture: texture_2d<f32>;
@group(1) @binding(8) var scene_sampler: sampler;
@group(1) @binding(1) var gbuffer_position: texture_2d<f32>;
@group(1) @binding(9) var gbuffer_sampler: sampler;

const PI: f32 = 3.141592653589793;
const PATH_LENGTH_MAX: f32 = 256.0;
const PARTICLE_SCATTER: f32 = 0.02;
const RAYLEIGH: vec3<f32> = vec3<f32>(0.00095, 0.00193, 0.00456);

// ---------------------------------------------------------------------------
// Waves
// ---------------------------------------------------------------------------

struct SurfaceSample
{
    /// World-space position after horizontal Gerstner displacement.
    displaced: vec3<f32>,
    normal: vec3<f32>,
    /// Vertical displacement, for crest foam.
    height: f32,
    /// Horizontal Jacobian determinant; below 1 means the surface is pinched.
    jacobian: f32,
};

fn smoothstepf(edge0: f32, edge1: f32, x: f32) -> f32
{
    let t = clamp((x - edge0)/(edge1 - edge0), 0.0, 1.0);
    return t*t*(3.0 - 2.0*t);
}

/// Fades a component out once its wavelength approaches the world distance one
/// sample spans. Aqua's cascade texture resolution does this by construction;
/// evaluating the sum per vertex needs it written down, or the coarse outer
/// tiles alias the short components into a shimmering sheet.
fn waveAttenuation(wavelength: f32, footprint: f32) -> f32
{
    return smoothstepf(0.5, 2.0, wavelength/max(footprint, 1e-4));
}

/// World distance one sample spans at `distance` from the camera.
///
/// The binding constraint is the *mesh*, not the pixel: LOD `n`'s tiles are
/// `WATER_BASE_SCALE * 2^n` metres across at 64 quads, so one quad is
/// `scale / 64`, and resolving a wave needs a wavelength of at least four
/// quads — `scale / 16`. Every ring covers distances from `scale` outward, so
/// writing that bound as `distance / 16` (which is `2 * distance / 32`, the
/// full-amplitude edge of `waveAttenuation`) makes it continuous across a ring
/// boundary *and* safe on both sides of it. That is what keeps the LOD
/// boundaries invisible here without aqua's vertex morphing: at the seam both
/// rings are at the same distance, so they carry an identical wave set.
fn sampleFootprint(distance: f32, pixel_angle: f32) -> f32
{
    let mesh_bound = distance*(1.0/32.0);
    return max(max(distance*pixel_angle, mesh_bound), 0.02);
}

fn sampleSurface(world_xz: vec2<f32>, footprint: f32, wave_weight: f32) -> SurfaceSample
{
    if (stage.flags.x > 0.5 || wave_weight <= 0.0)
    {
        return SurfaceSample(vec3<f32>(world_xz.x, 0.0, world_xz.y),
                             vec3<f32>(0.0, 1.0, 0.0), 0.0, 1.0);
    }

    let time = stage.params.x;
    var offset = vec3<f32>(0.0);
    var derivative_x = vec3<f32>(0.0);
    var derivative_z = vec3<f32>(0.0);

    // `waves` is sorted ascending by wavelength, so attenuation rises with the
    // index: the leading entries are the ones a distant tile cannot resolve
    // and the loop skips them without evaluating a sine.
    for (var index = 0u; index < WAVE_SLOTS; index += 1u)
    {
        let wave = stage.waves[index];
        let attenuation = waveAttenuation(wave.wavelength, footprint)*wave_weight;
        if (attenuation <= 0.0)
        {
            continue;
        }
        let amplitude = wave.amplitude*attenuation;
        let chop = wave.chop_amplitude*attenuation;
        let phase = wave.wave_number*dot(wave.direction, world_xz)
                  + wave.phase - wave.angular_frequency*time;
        let sin_phase = sin(phase);
        let cos_phase = cos(phase);

        offset += vec3<f32>(chop*wave.direction.x*sin_phase,
                            amplitude*cos_phase,
                            chop*wave.direction.y*sin_phase);

        // d(phase)/dx and d(phase)/dz, for the analytic tangents.
        let dphx = wave.wave_number*wave.direction.x;
        let dphz = wave.wave_number*wave.direction.y;
        derivative_x += vec3<f32>(-chop*wave.direction.x*dphx*cos_phase,
                                  -amplitude*dphx*sin_phase,
                                  -chop*wave.direction.y*dphx*cos_phase);
        derivative_z += vec3<f32>(-chop*wave.direction.x*dphz*cos_phase,
                                  -amplitude*dphz*sin_phase,
                                  -chop*wave.direction.y*dphz*cos_phase);
    }

    let tangent_x = vec3<f32>(1.0, 0.0, 0.0) + derivative_x;
    let tangent_z = vec3<f32>(0.0, 0.0, 1.0) + derivative_z;
    let normal = normalize(cross(tangent_z, tangent_x));
    // How much the horizontal displacement compresses the surface. Aqua uses
    // it to pinch subsurface scattering into crests.
    let jacobian = tangent_x.x*tangent_z.z - tangent_x.z*tangent_z.x;

    return SurfaceSample(
        vec3<f32>(world_xz.x + offset.x, offset.y, world_xz.y + offset.z),
        normal,
        offset.y,
        jacobian,
    );
}

// ---------------------------------------------------------------------------
// Sky and tone, matching composite.wgsl so water and terrain share one haze
// ---------------------------------------------------------------------------

/// The gradient itself, as a function of its own parameter: 0 at the horizon,
/// 1 at the zenith. Byte-for-byte the copy composite.wgsl draws with.
fn skyColour(parameter: f32) -> vec3<f32>
{
    let horizon = vec3<f32>(0.72, 0.82, 0.90);
    let zenith = vec3<f32>(0.22, 0.46, 0.74);
    return mix(horizon, zenith, parameter);
}

/// Sky gradient parameter for a ray whose sine of elevation is `direction_y`,
/// matching composite.wgsl's.
///
/// The 0.85 edge is the one this shader's reflection has always been authored
/// against; the sky now uses it as well, so the gradient a water pixel mirrors
/// and the gradient the sky pass paints are the same function of elevation and
/// agree at every angle, not only where both clamp to the horizon.
fn skyParameter(direction_y: f32) -> f32
{
    return smoothstepf(0.0, 0.85, direction_y);
}

/// `skyColour` in the linear space the rest of this shader lights in.
///
/// composite.wgsl authoring is display-referred: it decodes the gradient with
/// `pow(sky, 2.2)` before lighting. Multiplying by exposure and running ACES +
/// the gamma encode at the end then reproduces the sky pixel exactly, so a
/// fogged or fully reflective water fragment lands on the same value as the sky
/// behind it instead of banding at the horizon.
fn skyLinear(parameter: f32) -> vec3<f32>
{
    return pow(max(skyColour(parameter), vec3<f32>(0.0)), vec3<f32>(2.2));
}

/// Elevation sine of the representative sky that lights the foam.
///
/// Foam is neither a mirror nor a view direction: it is a diffuse surface lit
/// by the sky over the water, so it wants ONE colour rather than a ray, and it
/// stays a constant under the elevation parameterisation. `0.85 * 0.75` is the
/// old screen-row value carried across — under a ramp of
/// `smoothstep(0, 1, x)` a value `x` and the elevation sine `0.85x` land on the
/// same gradient parameter — so the foam is lit exactly as it was before.
const FOAM_SKY_ELEVATION: f32 = 0.85 * 0.75;

fn acesFilm(x: vec3<f32>) -> vec3<f32>
{
    const a: f32 = 2.51;
    const b: f32 = 0.03;
    const c: f32 = 2.43;
    const d: f32 = 0.59;
    const e: f32 = 0.14;
    return clamp((x*(a*x + b))/(x*(c*x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

// ---------------------------------------------------------------------------
// Medium optics (aqua::medium, single-sun reduction)
// ---------------------------------------------------------------------------

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

/// `godot_fresnel` from aqua::optics: F0 + (1 - F0)*(1 - cos)^exponent.
fn godotFresnel(view_alignment: f32, f0: f32, exponent: f32) -> f32
{
    let grazing = pow(clamp(1.0 - view_alignment, 0.0, 1.0), exponent);
    return mix(grazing, 1.0, f0);
}

fn dGGX(n_dot_h: f32, alpha: f32) -> f32
{
    let a2 = alpha*alpha;
    let d = n_dot_h*n_dot_h*(a2 - 1.0) + 1.0;
    return a2/max(d*d, 1e-7);
}

fn vSmithGGX(n_dot_l: f32, n_dot_v: f32, alpha: f32) -> f32
{
    let a2 = alpha*alpha;
    let ggx_l = n_dot_v*sqrt(n_dot_l*n_dot_l*(1.0 - a2) + a2);
    let ggx_v = n_dot_l*sqrt(n_dot_v*n_dot_v*(1.0 - a2) + a2);
    return 0.5/max(ggx_l + ggx_v, 1e-7);
}

/// Cheap value noise, used only to break up the shoreline foam edge. Aqua
/// simulates foam in a ping-pong texture; a shoreline band does not need a
/// simulation, just a non-uniform edge.
fn hash21(p: vec2<f32>) -> f32
{
    var q = fract(p*vec2<f32>(0.1031, 0.1030));
    q += dot(q, q.yx + 33.33);
    return fract((q.x + q.y)*q.x);
}

fn valueNoise(p: vec2<f32>) -> f32
{
    let i = floor(p);
    let f = fract(p);
    let u = f*f*(3.0 - 2.0*f);
    let a = hash21(i);
    let b = hash21(i + vec2<f32>(1.0, 0.0));
    let c = hash21(i + vec2<f32>(0.0, 1.0));
    let d = hash21(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

/// Two octaves is enough to break a contour line; this is not a detail layer.
fn foamBreakup(world_xz: vec2<f32>) -> f32
{
    let a = valueNoise(world_xz*0.35);
    let b = valueNoise(world_xz*1.10 + vec2<f32>(31.7, 11.3));
    return clamp(mix(0.55, 1.35, a*0.65 + b*0.35), 0.0, 1.4);
}

// ---------------------------------------------------------------------------
// Vertex stage
// ---------------------------------------------------------------------------

struct VertexOutput
{
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) water_height: f32,
    @location(3) jacobian: f32,
};

@vertex
fn vs_main(
    @location(0) local: vec2<f32>,
    @location(1) tile_centre: vec2<f32>,
    @location(2) tile_scale: f32,
    @location(3) tile_rotation: f32,
) -> VertexOutput
{
    let cos_r = cos(tile_rotation);
    let sin_r = sin(tile_rotation);
    let scaled = local*tile_scale;
    let world_xz = vec2<f32>(scaled.x*cos_r - scaled.y*sin_r,
                             scaled.x*sin_r + scaled.y*cos_r) + tile_centre;

    // World distance spanned by one pixel. `projection[1][1]` is cot(fov_y/2),
    // so the full vertical view at unit distance spans 2/projection[1][1] world
    // units and one pixel is that over the height. Deriving it beats a
    // constant: it follows the FOV and the window size.
    let pixel_angle = 2.0/(globals.projection[1][1]*max(globals.viewport.y, 1.0));
    let distance = length(world_xz - globals.camera_position.xz);

    // The outer skirt spans kilometres in one quad. Its inner and outer
    // vertices must both be flat: otherwise interpolation stretches the inner
    // vertex's wave slope all the way to the horizon. Fade the whole spectrum
    // across the last regular tiles, before the skirt starts. This distance is
    // shared by all LODs and includes the ring centre's camera-snap allowance.
    let wave_weight = 1.0 - smoothstepf(stage.flags.w*0.75, stage.flags.w, distance);
    let surface = sampleSurface(world_xz, sampleFootprint(distance, pixel_angle), wave_weight);
    let world_position = vec3<f32>(surface.displaced.x,
                                   stage.params.z + surface.displaced.y,
                                   surface.displaced.z);

    var out: VertexOutput;
    out.clip_position = globals.projection*globals.view*vec4<f32>(world_position, 1.0);
    out.world_position = world_position;
    out.world_normal = surface.normal;
    out.water_height = surface.height;
    out.jacobian = surface.jacobian;
    return out;
}

// ---------------------------------------------------------------------------
// Fragment stage
// ---------------------------------------------------------------------------

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32>
{
    let uv = in.clip_position.xy*globals.viewport.zw;
    let camera_position = globals.camera_position.xyz;
    let to_camera = camera_position - in.world_position;
    let view_distance = length(to_camera);
    let to_view = to_camera/max(view_distance, 1e-3);

    // View space looks down -Z, so a *greater* z is nearer. The water
    // surface's own view-space z, to compare against the G-buffer's.
    let water_view = (globals.view*vec4<f32>(in.world_position, 1.0)).xyz;

    // Every sample this shader will need, taken unconditionally: `normal` and
    // `uv` come from the vertex stage and are uniform, so these are legal, and
    // everything downstream only selects between values already in hand.
    // Two-sided: seen from below, the surface is the same sheet of water lit
    // from the other side, so the geometric normal is flipped toward the eye
    // rather than the pass culling backfaces.
    var normal = normalize(in.world_normal);
    if (dot(normal, to_view) < 0.0)
    {
        normal = -normal;
    }
    let packed = textureSample(gbuffer_position, gbuffer_sampler, uv);
    let scene_sky = packed.a < 0.5;
    var optical_path = PATH_LENGTH_MAX;
    if (!scene_sky)
    {
        optical_path = min(distance(packed.xyz, water_view), PATH_LENGTH_MAX);
    }

    // Refraction: perturb the screen uv by the surface slope, scaled by how
    // much water the ray crosses and by the inverse distance, since a metre of
    // slope covers fewer pixels far away.
    let refracted_uv = clamp(uv + normal.xz*(stage.misc.x*clamp(optical_path, 0.0, 1.0)
                           /max(view_distance, 1.0)), vec2<f32>(0.0), vec2<f32>(1.0));
    let flat_background = textureSample(scene_texture, scene_sampler, uv);
    let refracted_background = textureSample(scene_texture, scene_sampler, refracted_uv);
    let refracted_packed = textureSample(gbuffer_position, gbuffer_sampler, refracted_uv);

    // Hidden-surface test. The forward pass has no depth attachment, so this
    // stands in for one: the scene is in front when its view-space z is
    // greater. The epsilon absorbs the difference between a rasterized water
    // vertex and the same surface reconstructed from the G-buffer.
    if (!scene_sky && packed.z > water_view.z + 0.05)
    {
        discard;
    }

    // The refracted tap is only trustworthy when it is sky or genuinely behind
    // the water; otherwise it reached over a nearer surface and would smear
    // foreground colour into the water.
    let accept_refraction = refracted_packed.a < 0.5
                         || refracted_packed.z <= water_view.z;
    let background = select(flat_background.rgb, refracted_background.rgb, accept_refraction);
    let scene_linear = pow(max(background, vec3<f32>(0.0)), vec3<f32>(2.2));

    // --- Body: Beer-Lambert extinction plus in-scatter -----------------------
    let sea_level = stage.params.z;
    let sigma_t = max(stage.extinction.xyz, vec3<f32>(1e-4));
    let sigma_s = min(sigma_t,
                      PARTICLE_SCATTER*stage.extinction.w*stage.scatter.xyz + RAYLEIGH);
    let transmittance = exp(-sigma_t*optical_path);

    let to_sun = normalize(-globals.sun_direction.xyz);
    let sun_colour = vec3<f32>(1.0, 0.955, 0.90)*globals.settings_a.x;
    let cos_theta = dot(to_view, -to_sun);
    let phase = mix(henyeyGreenstein(cos_theta, clamp(stage.scatter.w, -0.99, 0.99)),
                    phaseRayleigh(cos_theta), 0.5);

    // Sunlight reaching the top of the column, attenuated over the water above
    // the shaded point. Aqua's closed-form in-scatter integral reduces to this
    // for a single direction and a homogeneous medium.
    let sun_attenuation = exp(-sigma_t*max(sea_level - in.world_position.y, 0.0));
    let scattered = sigma_s*sun_colour*sun_attenuation*phase*(1.0 - transmittance)/sigma_t;
    var body = scene_linear*transmittance + scattered;

    // --- Crest subsurface scattering ----------------------------------------
    // Where the horizontal displacement compresses the surface, light crosses
    // a thinner sheet of water and the crest glows.
    //
    // `jacobian` is 1 for an undisturbed surface, so measuring the pinch as
    // `1 - jacobian` puts the glow on the compressed crests and nowhere else.
    // Aqua's `1 - 0.5*jacobian` — which is what this port started from — reads
    // half strength *everywhere*, and since this term is a flat cyan-green
    // additive it then dominates the water body and washes the whole sea out
    // to a pale mint that no water ever is.
    let crest = clamp(1.0 - in.jacobian, 0.0, 1.0);
    let sss = pow(crest, 2.0)*stage.sss_tint.rgb*max(to_sun.y, 0.0);
    body += sss*sun_colour*0.35;

    // --- Whitecaps -----------------------------------------------------------
    // Foam is a wave breaking, so this is a local event on the surface rather
    // than a property of the sea as a whole.
    //
    // There used to be a second, depth-driven term alongside it — a ramp on
    // `water_depth` down to zero — and it was wrong twice over. It painted this
    // coast's whole shelf, which is shallow pretty much everywhere, so a ramp
    // meant to mark the waterline saturated across most of the visible ocean
    // and turned the sea into a milky sheet. And because `water_depth` is read
    // back out of the G-buffer it arrives stair-stepped, so a hard threshold on
    // it drew a straight, constant-profile boundary that cut the water into two
    // flat fields — dark on one side, milk on the other. Depth alone cannot
    // tell a calm shallow flat from surf, and here it labelled the former as
    // the latter. Foam now comes only from the surface itself, which needs no
    // depth and has no edge to fall off.
    //
    // `1 - jacobian` is the surface compression, so the top few per cent of it
    // is where the wave has actually pitched over. The threshold is set against
    // the measured distribution of that pinch (median 0.014, 99th percentile
    // 0.23), not against a guess.
    let whitecap = smoothstepf(0.10, 0.30, crest);
    let foam = clamp(whitecap*0.70*foamBreakup(in.world_position.xz)*stage.misc.y, 0.0, 1.0);
    // Foam is a bright diffuse surface, so its radiance sits a little above the
    // sky it is lit by — not the two-and-a-half times that `sun_colour*0.25`
    // produced, which clipped the shore break to flat white.
    let foam_colour = vec3<f32>(0.92, 0.95, 0.96)
                    * (sun_colour*0.05
                       + skyLinear(skyParameter(FOAM_SKY_ELEVATION))*0.75);
    body = mix(body, foam_colour, foam);

    // --- Reflection ----------------------------------------------------------
    // No environment cubemap here, so the reflected sky is the same analytic
    // gradient the sky pixels and the fog use. That is what the cubemap was
    // providing: a distant water surface converging on the sky above it.
    let reflection_direction = reflect(-to_view, normal);
    // Index the sky gradient by the reflected ray's own elevation.
    //
    // This used to project the reflected ray into screen space through the
    // camera and read the gradient there. A reflected ray leaves the water
    // pointing up, so that projection almost always landed off-screen and
    // clamped — the measured reflection was a single constant (43,128,198) at
    // every pixel, which is exactly the zenith end of the ramp. The water was
    // therefore mirroring the deepest blue in the sky at every angle,
    // including the grazing ones near the horizon that should be picking up
    // the pale band instead. Driving it from `reflection_direction.y` — the
    // sine of the reflected ray's elevation — lets grazing water converge on
    // the horizon colour and steep views reach up to the zenith, which is the
    // gradient real water shows.
    //
    // `skyParameter` is applied exactly once, here. It used to be applied
    // twice — once here and again inside `skyColour`, which then carried its
    // own `smoothstep(0, 1, ·)`. That was invisible while the drawn sky was
    // indexed by screen row, because the reflection was not matching it
    // anyway; now that both are functions of elevation, a second application
    // would mirror a sky that is not there — about 0.09 of gradient parameter
    // too dark from 10 to 30 degrees of elevation, a few luma of tone step
    // through the mid-field.
    let reflected = skyLinear(skyParameter(reflection_direction.y))*0.85;

    // Sun specular: GGX over the wave roughness, which is what produces a
    // glitter path rather than one broad highlight.
    let n_dot_l = max(dot(normal, to_sun), 0.0);
    let n_dot_v = max(dot(normal, to_view), 0.0);
    let half_vector = normalize(to_sun + to_view);
    let n_dot_h = max(dot(normal, half_vector), 0.0);
    var roughness = clamp(mix(0.18, 0.045, clamp(n_dot_l, 0.0, 1.0)), 0.02, 1.0);
    if (stage.surface.z > 0.0)
    {
        roughness = clamp(stage.surface.z, 0.01, 1.0);
    }
    let alpha = max(roughness*roughness, 1e-3);
    let specular = dGGX(n_dot_h, alpha)*vSmithGGX(n_dot_l, n_dot_v, alpha)*n_dot_l*0.35;

    // --- Fresnel composition -------------------------------------------------
    let fresnel = clamp(godotFresnel(clamp(dot(normal, to_view), 0.0, 1.0),
                                     stage.surface.x, stage.surface.y), 0.0, 1.0);
    var lit = mix(body, reflected, fresnel) + specular*sun_colour;

    // --- Aerial perspective, matching composite.wgsl -------------------------
    let fog = 1.0 - exp(-max(globals.params.x, 0.0)*view_distance);
    // The haze this water dissolves into is the sky along the same view ray,
    // which is `-to_view`: `to_view` points from the surface back to the eye,
    // so its negation is the eye's ray out to this fragment. Taking the
    // elevation from the geometry rather than rebuilding it from `uv` gives
    // exactly the direction composite.wgsl will use for the sky pixel at this
    // position — the fragment lies on that ray — so water haze and terrain
    // haze stay one colour where they meet instead of banding at the shoreline.
    // (A `y` below zero is a ray that ends on the water rather than in the sky;
    // `skyParameter` clamps it to the horizon end, the haze it is fading into.)
    let sky_fog_linear = skyLinear(skyParameter(-to_view.y));
    lit = mix(lit, sky_fog_linear, clamp(fog, 0.0, 1.0));

    return vec4<f32>(pow(acesFilm(lit*globals.params.z), vec3<f32>(1.0/2.2)), 1.0);
}
