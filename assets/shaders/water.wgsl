// All the water: the sea, rivers and creeks, ponds and lakes, and the medium
// the eye looks through once it goes under any of them. Derived from
// bevy-aqua (MIT OR Apache-2.0): the Gerstner sum is
// bevy-aqua-waves/src/anim_waves.wgsl, the Fresnel is
// bevy-aqua-optics/src/optics.wgsl (godot_fresnel), the body radiance is
// bevy-aqua-medium/src/medium.wgsl, the foam/SSS/composition is
// bevy-aqua-core/src/cascade/material.wgsl, and the submerged medium is
// bevy-aqua-volume/src/volume.wgsl. See src/water/ATTRIBUTION.md.
//
// ONE WATER. Every surface is shaded by `fs_water`, whichever mesh it comes
// from: the sea's camera-centred Crest tiles (`vs_sea`) or the rivers'
// ribbons and the lakes' sheets (`vs_inland`). Both hand it the same
// `WaterVertexOutput`, which describes the water there rather than naming a
// kind of it:
//
// - How much of it is the open sea's (1 on the sea, ramping up a river's last
//   reach to its mouth), how still it is (1 on a lake), its level, its
//   current, its whitewater and the foam drifting on it.
// - Its colour follows from those: the sea's optics preset, the tea of a
//   forest stream or a pond, or the clear cyan of a mountain lake above the
//   forest, where nothing stains it (`waterBody`).
// - Its roughness follows from the wind and from how much of it the water
//   feels (`windSea`): the open sea takes all of it over unlimited fetch, a
//   lake grows waves with the distance the wind has blown over it from its
//   upwind shore, and a pond or a creek under the trees lies sheltered and
//   nearly glassy. The sea adds its swell (the Gerstner sum) and its surf; a
//   river adds what its current does to it (`flowSurface`).
// - Seen from below, any of them is the same sheet: Snell's window and the
//   total internal reflection outside it.
//
// `fs_underwater` is the medium around a submerged eye: the same optics, for
// whichever water the eye is in. `fs_blit` copies the composited frame into
// the water pass's output target first.
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
//   displaces vertices with a continuous mesh limit, while fragments evaluate
//   normals and compression using the projected footprint of each wave.
//   Lighting therefore retains waves beyond the geometry's 744 m fade, even
//   on the flat horizon skirt. Unresolved waves become specular roughness.
//   Neither limit depends on the tile's LOD; `ranges` is uploaded for parity
//   but not read.
// - Waves shorter than the sea's 2 m spectrum are not aqua's: they are the
//   wind sea every water body shares (`windSea`), with a fetch-limited peak
//   and a saturated short-wave tail whose slope variance follows Cox and
//   Munk's measurements of the sea surface.
// - A local metre-resolution seabed map samples the same procedural terrain
//   and streamed erosion as the ground. Waves damp in shallow water and a
//   depth-limited, shoreward travelling wave train supplies small breakers and
//   receding foam. This is an analytic surf approximation, not a fluid solver.
//   The G-buffer still supplies the measured optical path for transmission;
//   it does not determine the surf band's width or direction.
// - Visible terrain reflections are raymarched against the view-position
//   buffer. Offscreen and disoccluded rays fade to the shared sky plus a
//   raymarched cloud probe, below the horizon the banks, hills and crowns
//   raise around the water. Scene taps are gamma-decoded and inverse-ACES
//   transformed before mixing with linear radiance; clipped highlights cannot
//   be recovered.
// - Aqua runs its medium as a fullscreen `ViewNode` writing a volume texture
//   that its material shaders then sample; `fs_underwater` applies the medium
//   straight to the frame, which is equivalent for a single camera whose
//   whole frustum sits in one homogeneous medium. Not ported: caustics, god
//   rays, bubble spray, and the post-process distortion from aqua's
//   `Effects`. None of them are part of the medium.
//
// WGSL CONSTRAINT: `textureSample` uses implicit derivatives and is illegal in
// non-uniform control flow. Screen derivatives are taken first; every texture
// read after them uses textureLoad or textureSampleLevel explicitly.

struct GlobalUniforms
{
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
    camera_position: vec4<f32>,
    sun_direction: vec4<f32>,     // xyz direction sunlight travels, WORLD space; w highest terrain in the lighting heightfield
    viewport: vec4<f32>,          // xy size in pixels, zw inverse size
    params: vec4<f32>,            // x fog_density, y z_far, z exposure, w ssao_enabled
    settings_a: vec4<f32>,        // x sun_intensity, y texture_scale, z ao_tex_strength, w variant_scale
    settings_b: vec4<f32>,        // x normal_strength, y sparkle_strength, z flow_debug, w erosion_debug
    sun_colour: vec4<f32>,      // linear sunlight tint
    moon_direction: vec4<f32>,  // direction moonlight travels
    atmosphere: vec4<f32>,      // daylight, moon intensity, local sky cover, volume strength
    raymarch: vec4<f32>,        // shadows, volumes, reflections, quality
    heightfield: vec4<f32>,     // world centre XZ, span, texel size
    clouds: vec4<f32>,          // enabled, coverage, density, base altitude
    cloud_layer: vec4<f32>,     // thickness, shape scale, shadow strength, quality
    cloud_motion: vec4<f32>,    // wind offset XZ, detail strength, maximum distance
    weather: vec4<f32>, // front offset XZ, climate bias, explicit cloud overrides
    storm: vec4<f32>, // precipitation bias, phase override, weather clock, wind direction radians
    lightning: vec4<f32>, // xyz terrain strike, w HDR flash radiance
    lightning_meta: vec4<f32>, // seed, age, bolt top altitude, front speed
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
const WIND_RUNGS: u32 = 14u;
const WIND_PER_RUNG: u32 = 4u;
const WIND_COMPONENTS: u32 = 56u;

struct WaterStageUniforms
{
    waves: array<GpuWave, WAVE_SLOTS>,
    ranges: array<vec4<u32>, LOD_COUNT>,   // per-LOD bands; see the note above
    params: vec4<f32>,        // x time, y authored amplitude, z sea level, w settled weather wave gain
    extinction: vec4<f32>,    // rgb extinction /m, w scatter scale
    scatter: vec4<f32>,       // rgb scatter tint, w asymmetry
    surface: vec4<f32>,       // x fresnel F0, y fresnel exponent, z sun roughness, w base scale
    sss_tint: vec4<f32>,      // rgb subsurface tint, w tile resolution
    misc: vec4<f32>,          // x refraction scale, y foam scale, z max amplitude, w significant wave height
    flags: vec4<f32>,         // x flat-surface debug, y sea state amplitude, z wind radians, w wave fade end
    shore_map: vec4<f32>,     // world centre XZ, span (0 while undrawn), texel size
    wind: vec4<f32>,          // x surface wind at 10 m, m/s; y gustiness 0..1; z viewer's exposure, s; w unused
    eye_water: vec4<f32>,     // x level of the water at the eye, y eye height above it, z submerged fade, w its sea share
    eye_body: vec4<f32>,      // x its stillness, y its whitewater, z its clarity; w the eye's share under it (its side)
    medium_sun: vec4<f32>,    // rgb sunlight at the surface for the medium, w moon radiance scale
    // The wind sea's components for the wind's heading (`windSea`): xy wave
    // vector in whole cycles over WIND_PERIOD, z phase in cycles, w heading
    // off the wind in radians.
    wind_waves: array<vec4<f32>, WIND_COMPONENTS>,
};
@group(2) @binding(0) var<uniform> stage: WaterStageUniforms;

// The composited frame (refraction source) and the G-buffer's view-space
// position, whose alpha is 0 for sky and >= 1 for geometry.
@group(1) @binding(0) var scene_texture: texture_2d<f32>;
@group(1) @binding(8) var scene_sampler: sampler;
@group(1) @binding(1) var gbuffer_position: texture_2d<f32>;
@group(1) @binding(9) var gbuffer_sampler: sampler;
@group(1) @binding(2) var terrain_heightfield: texture_2d<f32>;
@group(1) @binding(10) var terrain_heightfield_sampler: sampler;
@group(1) @binding(3) var shoreline_heightfield: texture_2d<f32>;
@group(1) @binding(11) var shoreline_sampler: sampler;
// The grass capture's ground colour, whose alpha is the share of open sky
// the tree crowns leave (vegetation.wgsl's canopy pass): the forest the
// water mirrors, and the shelter it gives from the wind.
@group(1) @binding(4) var canopy_texture: texture_2d<f32>;
@group(1) @binding(12) var canopy_sampler: sampler;
// Where that capture lies, as its own pass wrote it along with its texels:
// world centre XZ, span (0 until there is a capture), texel size. Never a
// window written before the frame's capture, which for the frame the camera
// crosses into the next 8 m window would shift every crown by 8 m.
@group(1) @binding(7) var<uniform> canopy_window: vec4<f32>;
// The sky probe is raymarched with a short reuse window (see
// PROBE_INTEGRATION_PERIOD in src/render/cloud_node.rs). Waves sample it in
// their reflected direction, so the complete cloud sky remains visible
// offscreen.
@group(3) @binding(2) var cloud_sky_probe: texture_2d<f32>;
@group(3) @binding(3) var cloud_sky_sampler: sampler;

const PI: f32 = 3.141592653589793;
const PATH_LENGTH_MAX: f32 = 256.0;
const PARTICLE_SCATTER: f32 = 0.02;
const RAYLEIGH: vec3<f32> = vec3<f32>(0.00095, 0.00193, 0.00456);
const N_WATER: f32 = 1.333;
const GRAVITY: f32 = 9.81;

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
    /// Mean squared slope of waves too small for the pixel to resolve.
    slope_variance: f32,
    /// Breaking/wash coverage before the bubble texture is applied.
    shore_foam: f32,
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

// The local map is world anchored, including when the camera translates.
// Fade the *effects*, not the sampled height, at its border so a moving map
// never sweeps a fictitious shallow contour across deep water.
struct ShoreSample {
    depth: f32,
    depth_gradient: vec2<f32>,
    weight: f32,
};

fn sampleShore(world_xz: vec2<f32>) -> ShoreSample {
    let uv = (world_xz - stage.shore_map.xy)/max(stage.shore_map.z, 1.0) + vec2<f32>(0.5);
    let edge = min(min(uv.x, uv.y), min(1.0 - uv.x, 1.0 - uv.y));
    let weight = smoothstepf(0.025, 0.16, edge);
    if (weight <= 0.0) { return ShoreSample(1000.0, vec2<f32>(0.0), 0.0); }
    let step = stage.shore_map.w*2.0;
    let du = vec2<f32>(step/stage.shore_map.z, 0.0);
    let bed = textureSampleLevel(shoreline_heightfield, shoreline_sampler, uv, 0.0).r;
    let left = textureSampleLevel(shoreline_heightfield, shoreline_sampler, uv - du, 0.0).r;
    let right = textureSampleLevel(shoreline_heightfield, shoreline_sampler, uv + du, 0.0).r;
    let back = textureSampleLevel(shoreline_heightfield, shoreline_sampler, uv - du.yx, 0.0).r;
    let front = textureSampleLevel(shoreline_heightfield, shoreline_sampler, uv + du.yx, 0.0).r;
    return ShoreSample(stage.params.z - bed,
                       vec2<f32>(left - right, back - front)/(2.0*step), weight);
}

struct ShoreMotion {
    height: f32,
    slope: vec2<f32>,
    foam: f32,
};

fn shoreMotion(world_xz: vec2<f32>, shore: ShoreSample, mesh_footprint: f32,
               pixel_dx: vec2<f32>, pixel_dy: vec2<f32>, local_gain: f32) -> ShoreMotion {
    let wave_height = stage.misc.w*local_gain;
    if (shore.weight <= 0.0 || wave_height < 0.001 || stage.flags.x > 0.5) {
        return ShoreMotion(0.0, vec2<f32>(0.0), 0.0);
    }
    let depth = max(shore.depth, 0.0);
    let slope = max(length(shore.depth_gradient), 0.035);
    let offshore = shore.depth_gradient/slope;
    let wind = vec2<f32>(cos(stage.flags.z), sin(stage.flags.z));
    // Offshore wind still leaves small lapping waves; exposed shores break more.
    let exposure = mix(0.28, 1.0, smoothstepf(-0.45, 0.65, dot(wind, -offshore)));
    let breaker_depth = max(wave_height/0.78, 0.08);
    let coast_distance = depth/slope;
    let band = (1.0 - smoothstepf(breaker_depth*1.5, breaker_depth*4.0, depth))
             * (1.0 - smoothstepf(5.0 + wave_height*8.0, 12.0 + wave_height*16.0, coast_distance))
             * smoothstepf(0.008, 0.07, length(shore.depth_gradient))*shore.weight;
    // Integration of c = sqrt(g*h) over a locally sloping bed gives arrival
    // time. The positive time sign moves these crests toward decreasing depth.
    // A small depth floor keeps the swash finite at the waterline.
    let period = 2.4 + sqrt(wave_height)*1.1;
    let omega = 2.0*PI/period;
    let phase = omega*(stage.params.x + 2.0*(sqrt(depth + 0.08) - sqrt(0.08))/(sqrt(9.81)*slope))
              + (valueNoise(world_xz*0.075) - 0.5)*1.4;
    let phase_gradient = omega*offshore/sqrt(9.81*(depth + 0.08));
    let phase_pixel = max(abs(dot(phase_gradient, pixel_dx)), abs(dot(phase_gradient, pixel_dy)));
    let resolved = (1.0 - smoothstepf(0.7, PI, phase_pixel))
                 * waveAttenuation(2.0*PI/max(length(phase_gradient), 0.01), mesh_footprint);
    let wash = smoothstepf(-0.08, 0.04, shore.depth);
    let amplitude = min(wave_height*0.24, depth*0.32 + wave_height*0.035)
                  * band*exposure*wash*resolved;
    let crest = cos(phase);
    let height = amplitude*(crest + 0.18*cos(2.0*phase));
    let gradient = -amplitude*(sin(phase) + 0.36*sin(2.0*phase))*phase_gradient;
    let breaking = (1.0 - smoothstepf(breaker_depth*0.65, breaker_depth*2.4, depth));
    let front = smoothstepf(0.25, 0.88, crest)*resolved;
    // After the crest passes, a weaker patch remains and decays between waves.
    let remnant = smoothstepf(-0.25, 0.85, cos(phase - 0.85))*resolved;
    let contact = 1.0 - smoothstepf(0.025, 0.10 + wave_height*0.32, depth);
    let foam = band*exposure*wash*breaking
             * (front*0.82 + remnant*0.24 + contact*(0.10 + remnant*0.25));
    return ShoreMotion(height, gradient, clamp(foam, 0.0, 1.0));
}

fn sampleSurface(world_xz: vec2<f32>, mesh_footprint: f32,
                 pixel_dx: vec2<f32>, pixel_dy: vec2<f32>, wave_weight: f32,
                 shore: ShoreSample) -> SurfaceSample
{
    if (stage.flags.x > 0.5 || wave_weight <= 0.0)
    {
        return SurfaceSample(vec3<f32>(world_xz.x, 0.0, world_xz.y),
                             vec3<f32>(0.0, 1.0, 0.0), 0.0, 1.0, 0.0, 0.0);
    }

    let time = stage.params.x;
    // Storm gusts roughen each water region under its own moving cell. The
    // authored spectrum and wave phase remain continuous across the map.
    let local_gust = weatherPrecipitation(vec3<f32>(world_xz.x, stage.params.z,
                                                   world_xz.y)).w;
    let local_gain = stage.params.w*(1.0 + 0.31*local_gust);
    var offset = vec3<f32>(0.0);
    var derivative_x = vec3<f32>(0.0);
    var derivative_z = vec3<f32>(0.0);
    var slope_variance = 0.0;
    let depth = max(shore.depth, 0.0);
    // Limit the entire train as the column becomes too thin to hold its waves.
    let depth_limit = smoothstepf(0.0, max(stage.misc.w*local_gain*1.5, 0.12), depth);

    // Skip unresolved components before evaluating trigonometry. Geometry
    // filters the shortest wavelengths first; pixels also account for each
    // wave's orientation inside their anisotropic footprint.
    for (var index = 0u; index < WAVE_SLOTS; index += 1u)
    {
        let wave = stage.waves[index];
        // Project the pixel footprint onto this wave's travel direction. A
        // grazing pixel stretches along the view ray, but can still resolve
        // waves running across it. Using only distance or the longest pixel
        // axis would erase that visible detail too early.
        let cycles_per_pixel = max(abs(dot(wave.direction, pixel_dx)),
                                   abs(dot(wave.direction, pixel_dy)))/wave.wavelength;
        // Full detail at eight pixels/cycle; gone before the Nyquist limit.
        let pixel_weight = 1.0 - smoothstepf(0.125, 0.5, cycles_per_pixel);
        let shallow = 1.0 - smoothstepf(0.0, 0.5, depth/wave.wavelength);
        let shoal = (1.0 + 0.22*shallow)*depth_limit;
        let bed_weight = mix(1.0, shoal, shore.weight);
        let slope = wave.amplitude*local_gain*wave.wave_number*bed_weight;
        slope_variance += 0.5*slope*slope*(1.0 - pixel_weight*pixel_weight);
        let attenuation = waveAttenuation(wave.wavelength, mesh_footprint)
                        * wave_weight*pixel_weight*bed_weight;
        if (attenuation <= 0.0)
        {
            continue;
        }
        let amplitude = wave.amplitude*local_gain*attenuation;
        let chop = wave.chop_amplitude*local_gain*attenuation;
        let phase = wave.wave_number*dot(wave.direction, world_xz)
                  + wave.phase - wave.angular_frequency*time;
        let sin_phase = sin(phase);
        let cos_phase = cos(phase);

        offset += vec3<f32>(chop*wave.direction.x*sin_phase,
                            amplitude*cos_phase,
                            chop*wave.direction.y*sin_phase);

        // d(phase)/dx and d(phase)/dz, for the analytic tangents. The horizontal
        // derivative keeps chop's sign: negative chop compresses the crest.
        let dphx = wave.wave_number*wave.direction.x;
        let dphz = wave.wave_number*wave.direction.y;
        derivative_x += vec3<f32>(chop*wave.direction.x*dphx*cos_phase,
                                  -amplitude*dphx*sin_phase,
                                  chop*wave.direction.y*dphx*cos_phase);
        derivative_z += vec3<f32>(chop*wave.direction.x*dphz*cos_phase,
                                  -amplitude*dphz*sin_phase,
                                  chop*wave.direction.y*dphz*cos_phase);
    }

    let surf = shoreMotion(world_xz, shore, mesh_footprint, pixel_dx, pixel_dy,
                           local_gain);
    offset.y += surf.height*wave_weight;
    derivative_x.y += surf.slope.x*wave_weight;
    derivative_z.y += surf.slope.y*wave_weight;
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
        slope_variance,
        surf.foam,
    );
}

// BEGIN SHARED SPATIAL PRECIPITATION
// This field is mirrored by weather.rs. Broad sky fronts and narrower storm
// cells share one advected world-space offset, so showers move with the wind.
fn weatherStormScore(world_xz: vec2<f32>) -> f32 {
    let p = (world_xz - globals.weather.xy)/5400.0;
    return clamp(0.5
        + 0.28*sin(1.07*p.x + 0.31*p.y + 0.53)
        + 0.16*sin(-0.61*p.x + 1.27*p.y - 1.1)
        + 0.11*sin(2.15*p.x - 1.41*p.y + 0.8), 0.0, 1.0);
}

fn weatherPrecipitationSeverity(world_xz: vec2<f32>) -> f32 {
    let p = (world_xz - globals.weather.xy)/8000.0;
    let score = clamp(0.523
        + 0.25*sin(0.83*p.x + 0.37*p.y)
        + 0.18*sin(-0.49*p.x + 0.91*p.y + 0.3)
        + 0.08*sin(1.73*p.x - 1.37*p.y - 1.2), 0.0, 1.0);
    var base = 0.0;
    if (score < 0.5) {
        base = clamp((score - 0.2)/0.3, 0.0, 1.0);
    } else if (score < 0.7) {
        base = 1.0 + (score - 0.5)/0.2;
    } else {
        base = 2.0 + clamp((score - 0.7)/0.18, 0.0, 1.0);
    }
    return clamp(base + globals.weather.z, 0.0, 3.0);
}

// Wind-carried rain curtains ride inside each broad storm cell. The two
// wavelengths and bent crosswind coordinate keep the passing sheets irregular
// while preserving a stationary world-space pattern as the camera moves.
fn weatherRainBand(world_xz: vec2<f32>) -> f32 {
    let p = world_xz - globals.weather.xy;
    let wind = vec2<f32>(cos(globals.storm.w), sin(globals.storm.w));
    let along = dot(p, wind);
    let across = dot(p, vec2<f32>(-wind.y, wind.x));
    let bent = along + 135.0*sin(across/340.0 + along/1450.0)
                      + 80.0*sin(across/125.0 - along/870.0);
    let broad = 0.5 + 0.5*sin(bent/150.0 + 1.2);
    let fine = 0.5 + 0.5*sin(bent/51.0 + across/190.0 + 0.6);
    let sheet = smoothstep(0.28, 0.82, broad*0.72 + fine*0.28);
    return mix(0.45, 1.16, sheet);
}

// A cheap core query for clouds and sky shading: avoid evaluating the finer
// moving rain sheets for every raymarch step through a thunderhead.
fn weatherStormBase(world_position: vec3<f32>) -> vec3<f32> {
    let score = weatherStormScore(world_position.xz);
    let severity = weatherPrecipitationSeverity(world_position.xz);
    let wet_gate = smoothstep(1.25, 2.0, severity);
    let base_offset = -300.0*clamp(severity - 1.0, 0.0, 1.0)
                        -150.0*clamp(severity - 2.0, 0.0, 1.0);
    let base_override = (u32(globals.weather.w) & 4u) != 0u;
    let cloud_base = max(globals.clouds.w
                         + select(base_offset, 0.0, base_override), 100.0);
    // Hydrometeors leave the lower cloud and fall toward the ground. Fade
    // them through its base so flying above a storm is dry.
    let below_cloud = 1.0 - smoothstep(cloud_base + 100.0,
                                       cloud_base + 600.0, world_position.y);
    let storm_precipitation = smoothstep(0.74, 0.90, score + globals.storm.x)
                        *wet_gate*below_cloud;
    if (storm_precipitation <= 0.0) { return vec3<f32>(0.0); }
    let cold_altitude = world_position.y
        + 70.0*sin((world_position.x - globals.weather.x)/18000.0
                   + (world_position.z - globals.weather.y)/27000.0 + 0.7);
    var snow_fraction = smoothstep(80.0, 260.0, cold_altitude);
    let override_kind = u32(globals.storm.y + 0.5);
    if (override_kind == 1u || override_kind == 3u || override_kind == 4u) { snow_fraction = 0.0; }
    if (override_kind == 2u) { snow_fraction = 1.0; }
    let convective_bias = select(0.0, globals.storm.x, override_kind == 3u);
    // A lull in a rain sheet does not instantly clear the thunderhead.
    let thunderstorm = select(storm_precipitation*(1.0 - snow_fraction)
                              *smoothstep(0.91, 0.98, score + convective_bias),
                              0.0, override_kind == 4u);
    return vec3<f32>(storm_precipitation, snow_fraction, thunderstorm);
}

fn weatherConvectiveCore(world_position: vec3<f32>) -> f32 {
    return weatherStormBase(world_position).z;
}

// Return rain, snow, thunderstorm, gust, each in [0, 1]. The altitude
// transition follows the terrain material's low mountain snowline. Explicit
// Snow/Rain/Thunderstorm trends select phase while retaining spatial cells.
fn weatherPrecipitation(world_position: vec3<f32>) -> vec4<f32> {
    let base = weatherStormBase(world_position);
    if (base.x <= 0.0) { return vec4<f32>(0.0); }
    let precipitation = min(base.x*weatherRainBand(world_position.xz), 1.0);
    let rain = precipitation*(1.0 - base.y);
    let snow = precipitation*base.y;
    let gust = clamp(0.18*precipitation + 0.75*base.z, 0.0, 1.0);
    return vec4<f32>(rain, snow, base.z, gust);
}
// END SHARED SPATIAL PRECIPITATION

// ---------------------------------------------------------------------------
// Sky and tone, matching composite.wgsl so water and terrain share one haze
// ---------------------------------------------------------------------------

// Shared linear atmosphere model. Kept byte-for-byte in the composite and
// water passes so reflected skies and the visible horizon agree.
// Broad moving fronts vary the conditions with world position. The CPU uses
// the same field for the player's local lighting and UI weather readout.
fn weatherSeverity(world_xz: vec2<f32>) -> f32 {
    let p = (world_xz - globals.weather.xy)/8000.0;
    let score = clamp(0.523
                      + 0.25*sin(0.83*p.x + 0.37*p.y)
                      + 0.18*sin(-0.49*p.x + 0.91*p.y + 0.3)
                      + 0.08*sin(1.73*p.x - 1.37*p.y - 1.2), 0.0, 1.0);
    var base = 0.0;
    if (score < 0.50) {
        base = clamp((score - 0.20)/0.30, 0.0, 1.0);
    } else if (score < 0.70) {
        base = 1.0 + (score - 0.50)/0.20;
    } else {
        base = 2.0 + clamp((score - 0.70)/0.18, 0.0, 1.0);
    }
    return clamp(base + globals.weather.z, 0.0, 3.0);
}

fn weatherFogMultiplier(severity: f32) -> f32 {
    if (severity < 1.0) { return mix(0.48, 1.0, severity); }
    if (severity < 2.0) { return mix(1.0, 2.4, severity - 1.0); }
    return mix(2.4, 250.0, severity - 2.0);
}

fn weatherSunMultiplier(severity: f32) -> f32 {
    if (severity < 1.0) { return mix(1.08, 1.0, severity); }
    if (severity < 2.0) { return mix(1.0, 0.38, severity - 1.0); }
    return mix(0.38, 0.14, severity - 2.0);
}

fn weatherSkyCover(severity: f32) -> f32 {
    if (severity <= 1.0) { return 0.0; }
    if (severity <= 2.0) { return 0.8*(severity - 1.0); }
    return mix(0.8, 1.0, severity - 2.0);
}

fn weatherFogAmbient(severity: f32, daylight: f32, thunder: f32) -> vec3<f32> {
    let sky_cover = weatherSkyCover(severity);
    let overcast = smoothstep(0.05, 0.9, sky_cover);
    let whiteout = smoothstep(0.78, 1.0, sky_cover);
    let storm = thunder;
    let night_ambient = mix(vec3<f32>(0.012, 0.019, 0.037),
                            vec3<f32>(0.021, 0.024, 0.028), overcast);
    let day_ambient = mix(mix(vec3<f32>(0.32, 0.44, 0.59),
                              vec3<f32>(0.40, 0.43, 0.43), overcast),
                          vec3<f32>(0.67, 0.69, 0.68), whiteout);
    let dark_night = mix(night_ambient, vec3<f32>(0.007, 0.010, 0.017), storm);
    let dark_day = mix(day_ambient, vec3<f32>(0.10, 0.125, 0.14), storm);
    return mix(dark_night, dark_day, clamp(daylight, 0.0, 1.0));
}

fn skyRadiance(direction: vec3<f32>) -> vec3<f32> {
    let ray = normalize(direction);
    let to_sun = -normalize(globals.sun_direction.xyz);
    let to_moon = -normalize(globals.moon_direction.xyz);
    let daylight = clamp(globals.atmosphere.x, 0.0, 1.0);
    // Weather transitions continuously from the clear blue dome to a diffuse
    // gray overcast, then to the nearly directionless sky of a whiteout.
    // A grazing view looks across weather several kilometres away, while a
    // steep view samples the front above the player. This lets blue gaps and
    // gray clouded regions share the same sky without a camera-wide switch.
    let cloud_distance = max((globals.clouds.w - globals.camera_position.y)
                             /max(ray.y, 0.08), 0.0);
    let horizon_distance = 12000.0*(1.0 - smoothstep(0.08, 0.8, ray.y));
    let sky_xz = globals.camera_position.xz + ray.xz
                 *clamp(max(cloud_distance, horizon_distance), 0.0, 16000.0);
    let storm = stormVisualStrength(sky_xz);
    let sky_cover = weatherSkyCover(weatherSeverity(sky_xz));
    let overcast = smoothstep(0.05, 0.9, sky_cover);
    let whiteout = smoothstep(0.78, 1.0, sky_cover);
    let elevation = smoothstep(0.0, 0.85, ray.y);
    let day_sky = mix(vec3<f32>(0.49, 0.64, 0.82),
                      vec3<f32>(0.045, 0.175, 0.43), elevation);
    let night_sky = mix(vec3<f32>(0.008, 0.013, 0.027),
                        vec3<f32>(0.0014, 0.0025, 0.008), elevation);
    let clouded_day_sky = mix(vec3<f32>(0.40, 0.43, 0.43),
                              vec3<f32>(0.28, 0.32, 0.34), elevation);
    let clouded_night_sky = mix(vec3<f32>(0.018, 0.022, 0.028),
                                vec3<f32>(0.009, 0.013, 0.021), elevation);
    let weather_day_sky = mix(mix(day_sky, clouded_day_sky, overcast),
                              vec3<f32>(0.64, 0.66, 0.65), whiteout);
    let weather_night_sky = mix(night_sky, clouded_night_sky, overcast);
    let storm_day_sky = mix(vec3<f32>(0.15, 0.17, 0.18),
                            vec3<f32>(0.06, 0.075, 0.09), elevation);
    let storm_night_sky = vec3<f32>(0.006, 0.009, 0.016);
    var sky = mix(mix(weather_night_sky, storm_night_sky, storm),
                  mix(weather_day_sky, storm_day_sky, storm), daylight);
    let sunset = exp(-pow((to_sun.y + 0.025)/0.17, 2.0));
    let horizon = exp(-abs(ray.y)*6.0);
    let sunward = pow(max(dot(normalize(vec3<f32>(ray.x, 0.001, ray.z)),
                              normalize(vec3<f32>(to_sun.x, 0.001, to_sun.z))), 0.0), 5.0);
    sky += vec3<f32>(0.65, 0.16, 0.025)*sunset*horizon*(0.15 + 0.85*sunward)
           *(1.0 - overcast)*(1.0 - storm);

    let sun_mu = clamp(dot(ray, to_sun), -1.0, 1.0);
    let sun_visible = smoothstep(-0.018, 0.012, to_sun.y);
    let sun_disk = smoothstep(0.9999832, 0.9999905, sun_mu);
    let sun_halo = exp(-(1.0 - sun_mu)*750.0)*0.10
                   + pow(max(sun_mu, 0.0), 24.0)*0.035;
    // Direct ground irradiance fades before the visible solar disc sets.
    sky += globals.sun_colour.rgb*sun_visible*(1.0 - overcast)*(1.0 - storm)
           *(sun_disk*14.0*globals.sun_colour.w + sun_halo*globals.settings_a.x);

    let moon_mu = clamp(dot(ray, to_moon), -1.0, 1.0);
    let moon_disk = smoothstep(0.999978, 0.999986, moon_mu);
    let moon_visible = smoothstep(-0.02, 0.03, to_moon.y)*(1.0 - daylight*0.9);
    sky += vec3<f32>(0.63, 0.74, 1.0)*moon_visible*(1.0 - overcast)*(1.0 - storm)
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
           *(1.0 - overcast)*(1.0 - storm)
           *smoothstep(-0.03, 0.18, ray.y)*0.7;
    return max(sky, vec3<f32>(0.0));
}

// Hemisphere-integrated fill depends on the surface orientation and time of
// day, never its screen row or camera pitch. Excludes the celestial discs.
fn skyAmbient(normal_world: vec3<f32>, world_position: vec3<f32>) -> vec3<f32> {
    let daylight = clamp(globals.atmosphere.x, 0.0, 1.0);
    let storm = stormLightAt(world_position);
    let sky_cover = weatherSkyCover(weatherSeverity(world_position.xz));
    let overcast = smoothstep(0.05, 0.9, sky_cover);
    let whiteout = smoothstep(0.78, 1.0, sky_cover);
    let upward = clamp(normal_world.y*0.5 + 0.5, 0.0, 1.0);
    let clear_day_fill = mix(vec3<f32>(0.13, 0.17, 0.22),
                             vec3<f32>(0.32, 0.44, 0.62), upward);
    let clouded_day_fill = mix(vec3<f32>(0.17, 0.19, 0.19),
                               vec3<f32>(0.30, 0.34, 0.35), upward);
    let day_fill = mix(mix(mix(clear_day_fill, clouded_day_fill, overcast),
                           vec3<f32>(0.52, 0.55, 0.54), whiteout),
                       mix(vec3<f32>(0.075, 0.095, 0.11),
                           vec3<f32>(0.19, 0.21, 0.22), upward), storm);
    let clear_night_fill = mix(vec3<f32>(0.007, 0.011, 0.022),
                               vec3<f32>(0.019, 0.029, 0.058), upward);
    let clouded_night_fill = mix(vec3<f32>(0.008, 0.010, 0.014),
                                 vec3<f32>(0.014, 0.018, 0.025), upward);
    let night_fill = mix(mix(clear_night_fill, clouded_night_fill, overcast),
                         vec3<f32>(0.006, 0.009, 0.016), storm);
    return mix(night_fill, day_fill, daylight);
}

// BEGIN SHARED VOLUMETRIC CLOUDS
// Periodic world-space noise keeps the volume stationary as the camera moves.
// RG: smooth fractal / inverted Worley shape; B: fine detail; A: broad weather.
@group(3) @binding(0) var cloud_noise: texture_3d<f32>;
@group(3) @binding(1) var cloud_noise_sampler: sampler;
// Optical depth toward the sun, per sun-ray entry point; see cloudSunShadow.
@group(3) @binding(4) var cloud_shadow_map: texture_2d<f32>;
@group(3) @binding(5) var cloud_shadow_sampler: sampler;
const CLOUD_PI: f32 = 3.14159265;
const CLOUD_EXTINCTION: f32 = 0.011;
struct CloudResult {
    scattering: vec3<f32>,
    transmittance: f32,
    distance: f32,
};

// Keep this field identical to WeatherState::front_severity in weather.rs.
// The offset advects coherent clear, cloudy, overcast and fog regions through
// the world instead of applying one condition to the entire camera view.
fn cloudFrontSeverity(world_xz: vec2<f32>) -> f32 {
    let p = (world_xz - globals.weather.xy)/8000.0;
    let score = clamp(0.523
        + 0.25*sin(0.83*p.x + 0.37*p.y)
        + 0.18*sin(-0.49*p.x + 0.91*p.y + 0.3)
        + 0.08*sin(1.73*p.x - 1.37*p.y - 1.2), 0.0, 1.0);
    var severity = 0.0;
    if (score < 0.5) {
        severity = clamp((score - 0.2)/0.3, 0.0, 1.0);
    } else if (score < 0.7) {
        severity = 1.0 + (score - 0.5)/0.2;
    } else {
        severity = 2.0 + clamp((score - 0.7)/0.18, 0.0, 1.0);
    }
    return clamp(severity + globals.weather.z, 0.0, 3.0);
}

// The cloud deck remains storm-dark through lulls between rain sheets.
// Sample below its base so an observer flying over the clouds can still see
// their dark upper billows while the open sky above stays bright.
fn stormCoreStrength(world_xz: vec2<f32>) -> f32 {
    let altitude = min(max(globals.camera_position.y, 0.0),
                       max(globals.clouds.w - 400.0, 0.0));
    return weatherConvectiveCore(vec3<f32>(world_xz.x, altitude, world_xz.y));
}

fn stormLightAt(world_position: vec3<f32>) -> f32 {
    let layer = cloudLocalLayer(cloudFrontSeverity(world_position.xz));
    let cloud_top = layer.x + layer.y;
    let below_deck = 1.0 - smoothstep(cloud_top - 150.0,
                                      cloud_top + 350.0, world_position.y);
    if (below_deck <= 0.0) { return 0.0; }
    return stormCoreStrength(world_position.xz)*below_deck;
}

fn stormVisualStrength(world_xz: vec2<f32>) -> f32 {
    return stormLightAt(vec3<f32>(world_xz.x, globals.camera_position.y,
                                  world_xz.y));
}

// x coverage offset, y extinction multiplier, z base offset in metres,
// w thickness multiplier. Adjacent states blend without moving cloud cells.
fn cloudFrontProfile(severity: f32) -> vec4<f32> {
    let clear = vec4<f32>(-0.43, 0.72, 150.0, 0.8);
    let cloudy = vec4<f32>(0.0, 1.0, 0.0, 1.0);
    let overcast = vec4<f32>(0.40, 1.5, -300.0, 1.35);
    let fog = vec4<f32>(0.38, 1.25, -450.0, 1.45);
    if (severity < 1.0) { return mix(clear, cloudy, severity); }
    if (severity < 2.0) { return mix(cloudy, overcast, severity - 1.0); }
    return mix(overcast, fog, severity - 2.0);
}

fn cloudLocalLayer(severity: f32) -> vec2<f32> {
    let profile = cloudFrontProfile(severity);
    let overrides = u32(globals.weather.w);
    let base_offset = select(profile.z, 0.0, (overrides & 4u) != 0u);
    let thickness_multiplier = select(profile.w, 1.0, (overrides & 8u) != 0u);
    return vec2<f32>(max(globals.clouds.w + base_offset, 100.0),
                     max(globals.cloud_layer.x*thickness_multiplier, 50.0));
}

fn cloudSunMultiplier(severity: f32) -> f32 {
    if (severity < 1.0) { return mix(1.08, 1.0, severity); }
    if (severity < 2.0) { return mix(1.0, 0.38, severity - 1.0); }
    return mix(0.38, 0.14, severity - 2.0);
}

fn cloudShadowMultiplier(severity: f32) -> f32 {
    if (severity < 1.0) { return mix(0.5, 1.0, severity); }
    if (severity < 2.0) { return mix(1.0, 0.22, severity - 1.0); }
    return mix(0.22, 0.05, severity - 2.0);
}

// The altitudes bounding every local cloud layer: x bottom, y top. Local
// bases and tops vary with the weather; this encloses all four profiles so an
// off-camera front cannot be clipped by the camera's own layer.
fn cloudSlab() -> vec2<f32> {
    let overrides = u32(globals.weather.w);
    let bottom = max(globals.clouds.w - select(450.0, 0.0, (overrides & 4u) != 0u), 100.0);
    let top = globals.clouds.w + select(150.0, 0.0, (overrides & 4u) != 0u)
              + max(globals.cloud_layer.x, 50.0)*select(1.45, 1.0, (overrides & 8u) != 0u);
    return vec2<f32>(bottom, top);
}

// Return a bounded interval even when the eye lies inside or above the layer.
// A horizontal ray is valid only if its origin already lies in the volume.
fn cloudInterval(origin: vec3<f32>, ray: vec3<f32>, maximum_distance: f32) -> vec2<f32> {
    let slab = cloudSlab();
    let bottom = slab.x;
    let top = slab.y;
    if (abs(ray.y) < 0.00001) {
        if (origin.y <= bottom || origin.y >= top) { return vec2<f32>(0.0); }
        return vec2<f32>(0.0, maximum_distance);
    }
    let a = (bottom - origin.y)/ray.y;
    let b = (top - origin.y)/ray.y;
    return vec2<f32>(max(min(a, b), 0.0), min(max(a, b), maximum_distance));
}

fn cloudNoiseLod(sample_width: f32, texel_world_size: f32) -> f32 {
    // A ray step can cross several noise voxels at grazing angles. Filtering
    // its footprint avoids treating one arbitrary voxel as an opaque sheet.
    return clamp(log2(max(sample_width/max(texel_world_size, 1.0), 1.0)), 0.0, 6.0);
}

fn cloudDensityFiltered(world_position: vec3<f32>, use_detail: bool, sample_width: f32) -> f32 {
    let severity = cloudFrontSeverity(world_position.xz);
    let profile = cloudFrontProfile(severity);
    let overrides = u32(globals.weather.w);
    let base_offset = select(profile.z, 0.0, (overrides & 4u) != 0u);
    let thickness_multiplier = select(profile.w, 1.0, (overrides & 8u) != 0u);
    let cloud_base = max(globals.clouds.w + base_offset, 100.0);
    let cloud_thickness = max(globals.cloud_layer.x*thickness_multiplier, 50.0);
    let height = (world_position.y - cloud_base)/cloud_thickness;
    if (height <= 0.0 || height >= 1.0) { return 0.0; }
    let scale = max(globals.cloud_layer.y, 100.0);
    let wind_position = world_position - vec3<f32>(globals.cloud_motion.x, 0.0, globals.cloud_motion.y);
    let weather_uv = vec3<f32>(wind_position.x, 0.37*scale, wind_position.z)/(scale*14.0);
    let weather_lod = cloudNoiseLod(sample_width, scale*14.0/64.0);
    let weather = textureSampleLevel(cloud_noise, cloud_noise_sampler, weather_uv, weather_lod).a;
    let coverage_offset = select(profile.x, 0.0, (overrides & 1u) != 0u);
    let mean_coverage = clamp(globals.clouds.y + coverage_offset, 0.0, 1.0);
    let patch_variation = min(mean_coverage*4.0, 1.0);
    let local_coverage = clamp(mean_coverage + (weather - 0.5)*0.30*patch_variation, 0.0, 1.0);
    if (local_coverage <= 0.001) { return 0.0; }
    // Sample several 3D cells through the layer. A metre-isotropic UV only
    // traverses a fraction of one cell vertically at the default thickness,
    // which makes cloud sides look like extruded horizontal silhouettes.
    let uv = vec3<f32>(wind_position.x/(scale*4.0) + height*0.055,
                       height*0.62 + 0.17,
                       wind_position.z/(scale*4.0) + height*0.0225);
    // Larger levels average away the occupied cells themselves; retain the
    // broad cloud bodies even when the distant fine structure is unresolved.
    let shape_lod = min(cloudNoiseLod(sample_width, scale*4.0/64.0), 3.0);
    let noise = textureSampleLevel(cloud_noise, cloud_noise_sampler, uv, shape_lod);
    let shape = noise.r*0.62 + noise.g*0.38;
    let threshold = mix(0.70, 0.23, local_coverage) + height*height*height*0.12;
    // Top heights follow the 3D shape, producing distinct billows when seen
    // from within or above the layer. The base varies slightly as well.
    let top = mix(0.72, 0.99, smoothstep(0.34, 0.70, shape));
    let base_profile = smoothstep(0.0, 0.08,
                                  height + (weather - 0.5)*0.05 + (noise.g - 0.5)*0.07);
    let height_profile = base_profile*(1.0 - smoothstep(top - 0.19, top, height));
    if (shape <= threshold) { return 0.0; }
    let detail_strength = clamp(globals.cloud_motion.z, 0.0, 1.0);
    // Erode the density field before the sharp remap: neighbouring Worley
    // cells carve rounded lobes out of the silhouette instead of merely
    // making a smooth blob more transparent. Two scales retain large billows
    // and fine wisps. Light rays use the average erosion for the same footprint.
    var erosion = detail_strength*0.20;
    if (use_detail) {
        let detail_lod_a = cloudNoiseLod(sample_width, scale*4.0/(64.0*2.7));
        let detail_lod_b = cloudNoiseLod(sample_width, scale*4.0/(64.0*6.1));
        let detail_a = textureSampleLevel(cloud_noise, cloud_noise_sampler,
                                          uv*2.7 + vec3<f32>(0.17, 0.41, 0.29), detail_lod_a);
        let detail_b = textureSampleLevel(cloud_noise, cloud_noise_sampler,
                                          uv*6.1 + vec3<f32>(0.53, 0.13, 0.71), detail_lod_b);
        let cellular_detail = detail_a.g*0.65 + detail_b.g*0.25 + detail_b.b*0.10;
        let detail = clamp((cellular_detail - 0.30)*2.0, 0.0, 1.0);
        let detail_weight = 1.0 - smoothstep(scale*0.12, scale*0.7, sample_width);
        erosion = mix(erosion, (1.0 - detail)*detail_strength*0.42, detail_weight);
    }
    let density = clamp((shape - threshold - erosion)/0.075, 0.0, 1.0)*height_profile;
    let density_multiplier = select(profile.y, 1.0, (overrides & 2u) != 0u);
    return density*max(globals.clouds.z*density_multiplier, 0.0);
}

fn cloudDensity(world_position: vec3<f32>, use_detail: bool) -> f32 {
    return cloudDensityFiltered(world_position, use_detail, 0.0);
}

fn cloudLightDepth(world_position: vec3<f32>, to_light: vec3<f32>, quality: f32) -> f32 {
    let interval = cloudInterval(world_position, to_light, 14000.0);
    let length = max(interval.y, 0.0);
    let count = 4u + u32(clamp(quality, 0.0, 2.0));
    var optical_depth = 0.0;
    // Quadratic spacing resolves the bright rim close to the sample and
    // increasingly broad taps estimate the shadow through the cloud body.
    for (var i = 0u; i < 6u; i += 1u) {
        if (i >= count) { break; }
        let a = f32(i)/f32(count);
        let b = (f32(i) + 1.0)/f32(count);
        let start = a*a*length;
        let end = b*b*length;
        let sample_position = world_position + to_light*((start + end)*0.5 + 4.0);
        optical_depth += cloudDensityFiltered(sample_position, false, end - start)
                         *(end - start)*CLOUD_EXTINCTION;
    }
    return optical_depth;
}

fn cloudPhase(mu: f32, g: f32) -> f32 {
    return (1.0 - g*g)/pow(max(1.0 + g*g - 2.0*g*mu, 0.025), 1.5);
}

fn marchClouds(origin: vec3<f32>, direction: vec3<f32>, maximum_distance: f32, quality: f32) -> CloudResult {
    var result: CloudResult;
    result.scattering = vec3<f32>(0.0);
    result.transmittance = 1.0;
    result.distance = maximum_distance;
    if (globals.clouds.x < 0.5 || globals.clouds.z <= 0.001) { return result; }
    let ray = normalize(direction);
    let cloud_range = max(globals.cloud_motion.w, 1000.0);
    let limit = min(maximum_distance, cloud_range);
    let interval = cloudInterval(origin, ray, limit);
    if (interval.y <= interval.x) { return result; }
    let count = select(select(32u, 48u, quality >= 1.0), 72u, quality >= 2.0);
    let span = interval.y - interval.x;
    // Quadratic spacing resolves nearby cloud surfaces even when a nearly
    // horizontal ray remains inside the layer for tens of kilometres.
    let near_bias = smoothstep(3000.0, 12000.0, span);
    let to_sun = -normalize(globals.sun_direction.xyz);
    let to_moon = -normalize(globals.moon_direction.xyz);
    let daylight = clamp(globals.atmosphere.x, 0.0, 1.0);
    let sun_phase = mix(cloudPhase(dot(ray, to_sun), 0.58), cloudPhase(dot(ray, to_sun), -0.22), 0.20);
    let moon_phase = mix(cloudPhase(dot(ray, to_moon), 0.48), 1.0, 0.28);
    var weighted_distance = 0.0;
    var distance_weight = 0.0;
    for (var i = 0u; i < 72u; i += 1u) {
        if (i >= count) { break; }
        let a = f32(i)/f32(count);
        let b = (f32(i) + 1.0)/f32(count);
        let step_start = mix(a, a*a, near_bias);
        let step_end = mix(b, b*b, near_bias);
        let step_length = (step_end - step_start)*span;
        let distance = interval.x + (step_start + step_end)*0.5*span;
        let world_position = origin + ray*distance;
        let density = cloudDensityFiltered(world_position, true, step_length)
                      *(1.0 - smoothstep(cloud_range*0.60, cloud_range, distance));
        if (density <= 0.001) { continue; }
        let local_severity = cloudFrontSeverity(world_position.xz);
        let storm = stormCoreStrength(world_position.xz);
        let local_layer = cloudLocalLayer(local_severity);
        let height = clamp((world_position.y - local_layer.x)/local_layer.y, 0.0, 1.0);
        let step_transmittance = exp(-density*CLOUD_EXTINCTION*step_length);
        let contribution = result.transmittance*(1.0 - step_transmittance);
        let clear_ambient = mix(vec3<f32>(0.065, 0.080, 0.105),
                                vec3<f32>(0.26, 0.31, 0.38), height);
        let overcast_ambient = mix(vec3<f32>(0.085, 0.095, 0.10),
                                   vec3<f32>(0.22, 0.24, 0.25), height);
        let day_ambient = mix(mix(mix(clear_ambient, overcast_ambient,
                                      smoothstep(1.0, 2.0, local_severity)),
                                  vec3<f32>(0.31, 0.33, 0.33),
                                  smoothstep(2.0, 3.0, local_severity)),
                              mix(vec3<f32>(0.027, 0.035, 0.044),
                                  vec3<f32>(0.13, 0.15, 0.17), height), storm);
        let night_ambient = mix(vec3<f32>(0.0015, 0.0025, 0.005), vec3<f32>(0.008, 0.012, 0.023), height);
        var lighting = mix(night_ambient, day_ambient, daylight);
        // Beer light transport gives dark rain-cloud cores; the two broader
        // scattering lobes approximate energy scattered repeatedly inside.
        // Strong forward scattering produces the silver lining toward the sun.
        let powder = mix(0.72, 1.18, 1.0 - exp(-density*3.0));
        if (globals.settings_a.x > 0.001) {
            let depth = cloudLightDepth(world_position, to_sun, quality);
            let transport = sun_phase*exp(-depth) + 0.28*exp(-depth*0.26) + 0.08*exp(-depth*0.055);
            lighting += globals.sun_colour.rgb*globals.settings_a.x
                        *cloudSunMultiplier(local_severity)*mix(1.0, 0.11, storm)
                        *0.18*transport*powder;
        }
        if (globals.atmosphere.y > 0.001) {
            let depth = cloudLightDepth(world_position, to_moon, max(quality - 1.0, 0.0));
            let transport = moon_phase*exp(-depth) + 0.32*exp(-depth*0.22);
            lighting += vec3<f32>(0.52, 0.65, 1.0)*globals.atmosphere.y
                        *cloudSunMultiplier(local_severity)*mix(1.0, 0.42, storm)
                        *0.22*transport;
        }
        if (globals.lightning.w > 0.001) {
            let source = vec3<f32>(globals.lightning.x,
                                   mix(globals.lightning.y, globals.lightning_meta.z, 0.65),
                                   globals.lightning.z);
            let range = length(world_position - source);
            let reach = 1.0/(1.0 + pow(range/1900.0, 2.0));
            lighting += vec3<f32>(0.72, 0.84, 1.0)*globals.lightning.w
                        *reach*0.55;
        }
        result.scattering += contribution*lighting;
        weighted_distance += distance*contribution;
        distance_weight += contribution;
        result.transmittance *= step_transmittance;
        if (result.transmittance < 0.008) { break; }
    }
    if (distance_weight > 0.0001) { result.distance = weighted_distance/distance_weight; }
    return result;
}

// The same density projects moving shadows onto land, water and atmospheric
// samples. Coarse taps deliberately omit fine erosion for a soft solar shadow.
// True when no cloud shadow can apply toward this light.
fn cloudShadowDisabled(to_light: vec3<f32>) -> bool {
    return globals.clouds.x < 0.5 || globals.clouds.z <= 0.001
        || globals.cloud_layer.z <= 0.001 || to_light.y <= 0.0;
}

// Optical depth through the layer from a point toward the light.
fn cloudShadowDepth(world_position: vec3<f32>, to_light: vec3<f32>,
                    interval: vec2<f32>) -> f32 {
    let count = 4u + u32(clamp(globals.cloud_layer.w, 0.0, 2.0));
    let step_length = (interval.y - interval.x)/f32(count);
    var optical_depth = 0.0;
    for (var i = 0u; i < 6u; i += 1u) {
        if (i >= count) { break; }
        let distance = interval.x + (f32(i) + 0.5)*step_length;
        optical_depth += cloudDensityFiltered(world_position + to_light*distance,
                                              false, step_length)*step_length*CLOUD_EXTINCTION;
    }
    return optical_depth;
}

fn cloudShadowFromDepth(world_position: vec3<f32>, optical_depth: f32) -> f32 {
    let strength = globals.cloud_layer.z*cloudShadowMultiplier(
        cloudFrontSeverity(world_position.xz));
    return mix(1.0, exp(-optical_depth), clamp(strength, 0.0, 1.0));
}

fn cloudShadow(world_position: vec3<f32>, to_light: vec3<f32>) -> f32 {
    if (cloudShadowDisabled(to_light)) { return 1.0; }
    let interval = cloudInterval(world_position, to_light, 65000.0);
    if (interval.y <= interval.x) { return 1.0; }
    return cloudShadowFromDepth(world_position,
                                cloudShadowDepth(world_position, to_light, interval));
}

// The cloud shadow map (src/render/cloud_node.rs) is rebuilt every frame.
// Below the layer, every point on one sun ray has the same interval through
// it, sampled at the same world positions, so cloudShadowDepth depends only on
// where that ray enters the layer's lower bound. The map stores that depth per
// entry point in two camera-centred levels, side by side in one texture: a
// fine one for the nearby air and a coarse one reaching past the 10 km fog
// range. Each level snaps to its own texel lattice.
const CLOUD_SHADOW_MAP_TEXELS: f32 = 1024.0;
fn cloudShadowMapHalfExtent(level: u32) -> f32 {
    return select(3072.0, 24576.0, level != 0u);
}

// The world XZ of texel (0, 0)'s lower corner in a level, for this sun.
fn cloudShadowMapOrigin(level: u32, to_sun: vec3<f32>) -> vec2<f32> {
    let slab = cloudSlab();
    let camera = globals.camera_position.xyz;
    let reach = (slab.x - min(camera.y, slab.x))/max(to_sun.y, 0.001);
    let centre = camera.xz + to_sun.xz*reach;
    let half_extent = cloudShadowMapHalfExtent(level);
    let texel = 2.0*half_extent/CLOUD_SHADOW_MAP_TEXELS;
    return floor(centre/texel + vec2<f32>(0.5))*texel - vec2<f32>(half_extent);
}

// Optical depth and an edge blend weight. Stay inside the texel centres so
// filtering cannot cross the packed levels; blend over 32 texels to avoid a
// seam when crossing from the fine level to the coarse one, or out of the map.
fn cloudShadowMapSample(entry: vec2<f32>, level: u32, to_sun: vec3<f32>) -> vec2<f32> {
    let texel = 2.0*cloudShadowMapHalfExtent(level)/CLOUD_SHADOW_MAP_TEXELS;
    let coordinate = (entry - cloudShadowMapOrigin(level, to_sun))/texel - vec2<f32>(0.5);
    let last = CLOUD_SHADOW_MAP_TEXELS - 1.0;
    if (any(coordinate < vec2<f32>(0.0)) || any(coordinate > vec2<f32>(last))) {
        return vec2<f32>(-1.0, 0.0);
    }
    let uv = (coordinate + vec2<f32>(0.5 + f32(level)*CLOUD_SHADOW_MAP_TEXELS, 0.5))
             /vec2<f32>(2.0*CLOUD_SHADOW_MAP_TEXELS, CLOUD_SHADOW_MAP_TEXELS);
    let edge = min(min(coordinate.x, coordinate.y), min(last - coordinate.x, last - coordinate.y));
    return vec2<f32>(textureSampleLevel(cloud_shadow_map, cloud_shadow_sampler, uv, 0.0).r,
                     smoothstep(0.0, 32.0, edge));
}

// The map's depth for a point, or -1 where it cannot stand in for the march:
// inside or above the layer, outside both levels, under a sun so low that the
// march's 65 km limit would clip the interval, or before the first build.
fn cloudShadowMapDepth(world_position: vec3<f32>, to_sun: vec3<f32>) -> f32 {
    let slab = cloudSlab();
    if (world_position.y >= slab.x || slab.y - world_position.y > 65000.0*to_sun.y) {
        return -1.0;
    }
    let entry = world_position.xz + to_sun.xz*((slab.x - world_position.y)/to_sun.y);
    let fine = cloudShadowMapSample(entry, 0u, to_sun);
    if (fine.x >= 0.0 && fine.y >= 1.0) { return fine.x; }
    let coarse = cloudShadowMapSample(entry, 1u, to_sun);
    var outer_depth = coarse.x;
    if (outer_depth >= 0.0 && coarse.y < 1.0) {
        let interval = cloudInterval(world_position, to_sun, 65000.0);
        let exact = cloudShadowDepth(world_position, to_sun, interval);
        outer_depth = mix(exact, outer_depth, coarse.y);
    }
    if (fine.x >= 0.0) {
        if (outer_depth >= 0.0) { return mix(outer_depth, fine.x, fine.y); }
        return fine.x;
    }
    return outer_depth;
}

// cloudShadow toward the sun, reading the shadow map where it applies.
fn cloudSunShadow(world_position: vec3<f32>, to_sun: vec3<f32>) -> f32 {
    if (cloudShadowDisabled(to_sun)) { return 1.0; }
    let mapped = cloudShadowMapDepth(world_position, to_sun);
    if (mapped >= 0.0) { return cloudShadowFromDepth(world_position, mapped); }
    return cloudShadow(world_position, to_sun);
}
// END SHARED VOLUMETRIC CLOUDS

fn cloudSkyRadiance(direction: vec3<f32>) -> vec3<f32> {
    let ray = normalize(direction);
    let clear_sky = skyRadiance(ray);
    if (globals.clouds.x < 0.5) { return clear_sky; }
    let uv = vec2<f32>(atan2(ray.z, ray.x)/(2.0*PI) + 0.5,
                       0.5 - asin(clamp(ray.y, -1.0, 1.0))/PI);
    let cloud = textureSampleLevel(cloud_sky_probe, cloud_sky_sampler, uv, 0.0);
    return clear_sky*cloud.a + cloud.rgb;
}

// BEGIN SHARED TERRAIN SHADOWS
// The map follows the camera, includes off-screen terrain, and contains the
// same procedural + erosion height used by the terrain vertex shader.
fn terrainHeightAt(world_xz: vec2<f32>) -> f32 {
    let uv = (world_xz - globals.heightfield.xy)/globals.heightfield.z + vec2<f32>(0.5);
    return textureSampleLevel(terrain_heightfield, terrain_heightfield_sampler, uv, 0.0).r;
}
fn terrainMapWeight(world_xz: vec2<f32>) -> f32 {
    let uv = (world_xz - globals.heightfield.xy)/globals.heightfield.z + vec2<f32>(0.5);
    let edge = min(min(uv.x, uv.y), min(1.0 - uv.x, 1.0 - uv.y));
    return smoothstep(0.0, 0.06, edge);
}
fn terrainShadow(world_position: vec3<f32>, normal_world: vec3<f32>, to_light: vec3<f32>) -> f32 {
    if (globals.raymarch.x < 0.5 || globals.heightfield.z < 1.0) { return 1.0; }
    if (to_light.y < -0.025) { return 0.0; }
    // The map has metre-scale texels. Lift the origin above the local map
    // mismatch and bias grazing slopes to avoid dark checkerboard acne.
    let texel = globals.heightfield.w;
    let bias = 1.5 + texel*(0.12 + 0.25*(1.0 - abs(normal_world.y)));
    let local_height = terrainHeightAt(world_position.xz);
    let lift = max(local_height - world_position.y, 0.0) + bias;
    let origin = world_position + vec3<f32>(0.0, lift, 0.0);
    let quality = clamp(globals.raymarch.w, 0.0, 2.0);
    let count = 20u + u32(quality)*12u;
    // Rising faster than the penumbra widens, a tap a full penumbra above the
    // highest terrain is fully lit, and so is every tap after it.
    let rising = to_light.y >= 0.0047;
    var visibility = 1.0;
    for (var i = 0u; i < 44u; i += 1u) {
        if (i >= count) { break; }
        let f = (f32(i) + 1.0)/f32(count);
        let distance = texel*1.2 + f*f*4800.0;
        let sample_position = origin + to_light*distance;
        // A finite solar disc makes the penumbra widen with occluder range.
        let penumbra = max(1.2, distance*0.0047);
        if (rising && sample_position.y - penumbra >= globals.sun_direction.w) { break; }
        let edge_weight = terrainMapWeight(sample_position.xz);
        if (edge_weight <= 0.0) { break; }
        let clearance = sample_position.y - terrainHeightAt(sample_position.xz);
        let sample_visibility = smoothstep(-penumbra, penumbra, clearance);
        visibility = min(visibility, mix(1.0, sample_visibility, edge_weight));
        if (visibility < 0.015) { break; }
    }
    return mix(1.0, visibility, terrainMapWeight(world_position.xz));
}
// END SHARED TERRAIN SHADOWS

// Match the composite's atmosphere to the actual water depth. Water is drawn
// after the half-resolution atmosphere target, whose depth is terrain/sky;
// using that target would integrate fog behind the water. This smaller march
// uses the same density, phase, shadow taps and source radiance at surface depth.
// Volumes need fewer samples per light ray than visible surfaces. All taps
// use explicit LOD because the ray loops have nonuniform termination.
fn volumeLightVisibility(world_position: vec3<f32>, to_light: vec3<f32>) -> f32 {
    if (globals.raymarch.x < 0.5 || globals.heightfield.z < 1.0) { return 1.0; }
    if (to_light.y < -0.025) { return 0.0; }
    var visibility = 1.0;
    let count = 6u + u32(clamp(globals.raymarch.w, 0.0, 2.0))*2u;
    // Rising toward the light, a tap 5 m above the highest terrain is fully
    // lit, and so is every tap after it.
    let rising = to_light.y > 0.0;
    for (var i = 0u; i < 10u; i += 1u) {
        if (i >= count) { break; }
        let f = (f32(i) + 0.4)/f32(count);
        let sample_position = world_position + to_light*(16.0 + f*f*3600.0);
        if (rising && sample_position.y - 5.0 >= globals.sun_direction.w) { break; }
        let edge_weight = terrainMapWeight(sample_position.xz);
        if (edge_weight <= 0.0) { break; }
        let clearance = sample_position.y - terrainHeightAt(sample_position.xz);
        visibility = min(visibility, mix(1.0, smoothstep(-5.0, 5.0, clearance), edge_weight));
        if (visibility < 0.02) { break; }
    }
    return visibility;
}

struct FogResult { scattering: vec3<f32>, transmittance: f32 };
fn precipitationExtinction(precipitation: vec4<f32>) -> f32 {
    return precipitation.x*0.0021 + precipitation.y*0.0013
           + precipitation.z*0.00035;
}
fn integrateAtmosphere(ray: vec3<f32>, distance_to_surface: f32) -> FogResult {
    var result: FogResult;
    result.scattering = vec3<f32>(0.0);
    result.transmittance = 1.0;
    let base_density = max(globals.params.x, 0.0);
    let ray_length = min(distance_to_surface, 10000.0);
    if (globals.raymarch.y < 0.5) {
        let world_position = globals.camera_position.xyz + ray*ray_length*0.5;
        let severity = weatherSeverity(world_position.xz);
        let precipitation = weatherPrecipitation(world_position);
        let density = base_density*weatherFogMultiplier(severity)
                      + precipitationExtinction(precipitation);
        result.transmittance = exp(-density*ray_length);
        var ambient = weatherFogAmbient(severity, globals.atmosphere.x,
                                        precipitation.z);
        if (globals.lightning.w > 0.001) {
            let source = vec3<f32>(globals.lightning.x,
                                   mix(globals.lightning.y, globals.lightning_meta.z, 0.60),
                                   globals.lightning.z);
            let range = length(world_position - source);
            ambient += vec3<f32>(0.72, 0.84, 1.0)*globals.lightning.w
                       *0.28/(1.0 + pow(range/2200.0, 2.0));
        }
        result.scattering = ambient*(1.0 - result.transmittance);
        return result;
    }
    let quality = clamp(globals.raymarch.w, 0.0, 2.0);
    let count = select(select(8u, 12u, quality >= 1.0), 16u, quality >= 2.0);
    let to_sun = -normalize(globals.sun_direction.xyz);
    let to_moon = -normalize(globals.moon_direction.xyz);
    let sun_scatter = globals.sun_colour.rgb*globals.settings_a.x
                      *henyeyGreenstein(dot(ray, to_sun), 0.68);
    let moon_scatter = vec3<f32>(0.52, 0.65, 1.0)*globals.atmosphere.y
                       *henyeyGreenstein(dot(ray, to_moon), 0.45);
    let strength = max(globals.atmosphere.w, 0.0);
    // Quadratic view steps keep close shafts detailed while reaching the
    // distant landscape. Beer-Lambert accumulation conserves transmittance.
    for (var i = 0u; i < 16u; i += 1u) {
        if (i >= count) { break; }
        let a = f32(i)/f32(count);
        let b = (f32(i) + 1.0)/f32(count);
        let start = a*a*ray_length;
        let end = b*b*ray_length;
        let midpoint = (start + end)*0.5;
        let world_position = globals.camera_position.xyz + ray*midpoint;
        let severity = weatherSeverity(world_position.xz);
        let precipitation = weatherPrecipitation(world_position);
        let sky_cover = weatherSkyCover(severity);
        var ambient = weatherFogAmbient(severity, globals.atmosphere.x,
                                        precipitation.z);
        if (globals.lightning.w > 0.001) {
            let source = vec3<f32>(globals.lightning.x,
                                   mix(globals.lightning.y, globals.lightning_meta.z, 0.60),
                                   globals.lightning.z);
            let range = length(world_position - source);
            ambient += vec3<f32>(0.72, 0.84, 1.0)*globals.lightning.w
                       *0.35/(1.0 + pow(range/2200.0, 2.0));
        }
        let height_density = exp(-max(world_position.y - 20.0, 0.0)/450.0);
        let extinction = base_density*weatherFogMultiplier(severity)
                         *(0.16 + 0.84*height_density)
                         + precipitationExtinction(precipitation);
        let step_transmittance = exp(-extinction*(end - start));
        var sunlight_visibility = 1.0;
        if (globals.settings_a.x > 0.01 && strength > 0.0) {
            // Where the terrain blocks the sun the cloud shadow cannot matter.
            sunlight_visibility = volumeLightVisibility(world_position, to_sun);
            if (sunlight_visibility > 0.0) {
                sunlight_visibility *= cloudSunShadow(world_position, to_sun);
            }
        }
        var moonlight_visibility = 1.0;
        if (globals.atmosphere.y > 0.001 && strength > 0.0) {
            moonlight_visibility = cloudShadow(world_position, to_moon);
        }
        let lighting = ambient + (sun_scatter*weatherSunMultiplier(severity)
                                  *(1.0 - sky_cover*0.85)
                                  *mix(1.0, 0.10, precipitation.z)
                                  *sunlight_visibility
                                 + moon_scatter*weatherSunMultiplier(severity)
                                 *mix(1.0, 0.42, precipitation.z)
                                 *moonlight_visibility)*strength;
        result.scattering += result.transmittance*(1.0 - step_transmittance)*lighting;
        result.transmittance *= step_transmittance;
        if (result.transmittance < 0.01) { break; }
    }
    return result;
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

fn projectView(position: vec3<f32>) -> vec2<f32>
{
    let clip = globals.projection*vec4<f32>(position, 1.0);
    let ndc = clip.xy/max(clip.w, 0.001);
    return vec2<f32>(ndc.x*0.5 + 0.5, 0.5 - ndc.y*0.5);
}

fn reflectionDepth(uv: vec2<f32>) -> vec4<f32>
{
    let size = vec2<i32>(textureDimensions(gbuffer_position));
    return textureLoad(gbuffer_position, clamp(vec2<i32>(uv*vec2<f32>(size)),
                        vec2<i32>(0), size - vec2<i32>(1)), 0);
}

// RGB is incident radiance and A confidence. A miss never stretches an edge
// texel across the sea: it leaves the analytic sky visible instead. Exponential
// world-distance steps retain nearby detail while reaching distant headlands.
fn marchWaterReflection(world_origin: vec3<f32>, world_normal: vec3<f32>,
                         direction: vec3<f32>, roughness: f32, surface_level: f32,
                         from_below: bool) -> vec4<f32>
{
    if (globals.raymarch.z < 0.5)
    {
        return vec4<f32>(0.0);
    }
    let origin = (globals.view*vec4<f32>(world_origin + world_normal*0.12, 1.0)).xyz;
    let ray = normalize((globals.view*vec4<f32>(direction, 0.0)).xyz);
    var max_distance = min(globals.params.y*0.75, 2200.0);
    if (ray.z > 0.0)
    {
        max_distance = min(max_distance, (-0.15 - origin.z)/ray.z);
    }
    if (max_distance <= 0.75 || origin.z > -0.1)
    {
        return vec4<f32>(0.0);
    }

    let quality = u32(clamp(globals.raymarch.w, 0.0, 2.0));
    var steps = 24u;
    if (quality == 1u) { steps = 40u; }
    if (quality == 2u) { steps = 64u; }
    let near_step = max(0.35, -origin.z*0.001);
    let logarithmic_range = log(1.0 + max_distance/near_step);
    var previous_distance = 0.0;
    var previous_front = true;

    for (var step = 0u; step < 64u; step += 1u)
    {
        if (step >= steps) { break; }
        let progress = f32(step + 1u)/f32(steps);
        let distance = near_step*(exp(logarithmic_range*progress) - 1.0);
        let sample_position = origin + ray*distance;
        let uv = projectView(sample_position);
        if (any(uv <= vec2<f32>(0.001)) || any(uv >= vec2<f32>(0.999)))
        {
            break;
        }
        let packed = reflectionDepth(uv);
        let behind = packed.a >= 0.5 && packed.z - sample_position.z > 0.0;
        if (behind && previous_front)
        {
            var low = previous_distance;
            var high = distance;
            for (var refine = 0u; refine < 7u; refine += 1u)
            {
                if (refine >= 5u + quality) { break; }
                let middle = (low + high)*0.5;
                let candidate = origin + ray*middle;
                let depth = reflectionDepth(projectView(candidate));
                if (depth.a >= 0.5 && depth.z - candidate.z > 0.0)
                {
                    high = middle;
                }
                else
                {
                    low = middle;
                }
            }
            let hit_position = origin + ray*high;
            let hit_uv = projectView(hit_position);
            let hit = reflectionDepth(hit_uv);
            let thickness = clamp(0.4 - hit.z*0.0015, 0.4, 4.0);
            let depth_error = hit.z - hit_position.z;
            // Reject self hits, depth discontinuities and submerged geometry
            // seen through the water's reflective side. Seen from above
            // (`from_below` false, the side fs_water sees the surface from),
            // the reflected ray stays in the air however high the surface
            // stands over the eye, and what lies under its level is the bed
            // seen through it, not its reflection.
            let world_hit = globals.camera_position.xyz
                          + vec3<f32>(dot(globals.view[0].xyz, hit.xyz),
                                      dot(globals.view[1].xyz, hit.xyz),
                                      dot(globals.view[2].xyz, hit.xyz));
            let above_surface = from_below || world_hit.y >= surface_level - 0.25;
            if (hit.a >= 0.5 && high > max(0.5, near_step)
                && depth_error >= 0.0 && depth_error < thickness && above_surface)
            {
                let edge = min(min(hit_uv.x, hit_uv.y), min(1.0 - hit_uv.x, 1.0 - hit_uv.y));
                let edge_fade = smoothstepf(0.005, 0.07, edge);
                let distance_fade = 1.0 - smoothstepf(max_distance*0.65, max_distance, high);
                let thickness_fade = 1.0 - smoothstepf(thickness*0.35, thickness, depth_error);
                let confidence = edge_fade*distance_fade*thickness_fade
                               * (1.0 - smoothstepf(0.15, 0.6, roughness));
                let colour = textureSampleLevel(scene_texture, scene_sampler, hit_uv, 0.0).rgb;
                return vec4<f32>(sceneRadiance(colour), confidence);
            }
        }
        previous_front = !behind;
        previous_distance = distance;
    }
    return vec4<f32>(0.0);
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

// Variance, in pixels squared, of the pixel filter the shading is judged
// over: the Gaussian whose peak is one (Tokuyoshi and Kaplanyan 2019).
const PIXEL_FILTER_VARIANCE: f32 = 0.15915494;

// The surface's roughness `alpha` as the pixel sees it, with `normal_dx` and
// `normal_dy` the change of the normal to the next pixel across and down.
// What is resolved can still turn faster across a pixel than its glint is
// wide: the crest of a riffle, the kink of a standing wave, a ripple seen
// edge-on far off. One normal for the pixel would make a glint that falls
// between pixels and flickers as the camera moves; the pixel sees every
// normal its footprint spans, so the normal's spread over the pixel filter
// widens the lobe (geometric specular antialiasing: Kaplanyan et al. 2016,
// Tokuyoshi and Kaplanyan 2019). A ripple's glint two pixels across or more
// is resolved and widens by under five percent; one narrower than a pixel
// spreads over the pixel, as dim as it is wide. The widening is capped where
// the normal jumps rather than turns, at an edge-on facet.
fn pixelRoughness(alpha: f32, normal_dx: vec3<f32>, normal_dy: vec3<f32>) -> f32
{
    let spread = PIXEL_FILTER_VARIANCE*(dot(normal_dx, normal_dx) + dot(normal_dy, normal_dy));
    return sqrt(min(alpha*alpha + min(2.0*spread, 0.18), 1.0));
}

/// Sun- or moonlight a water surface reflects toward the eye per unit of
/// irradiance: f_r*n.l of a GGX microfacet dielectric. D carries its 1/pi
/// (irradiance here follows the composite's albedo/pi convention), V is the
/// height-correlated Smith term, and the Fresnel is the surface's own fit
/// taken at the microfacet, v.h: a facet that faces the eye returns 2% of
/// the sun and only a grazing one most of it. Nothing else scales the glint.
fn waterSpecular(normal: vec3<f32>, to_view: vec3<f32>, to_light: vec3<f32>, alpha: f32) -> f32
{
    let sum = to_light + to_view;
    let half_vector = sum/max(length(sum), 1e-4);
    let n_dot_l = max(dot(normal, to_light), 0.0);
    let n_dot_v = max(dot(normal, to_view), 1e-4);
    let n_dot_h = max(dot(normal, half_vector), 0.0);
    let v_dot_h = clamp(dot(to_view, half_vector), 0.0, 1.0);
    let fresnel = godotFresnel(v_dot_h, stage.surface.x, stage.surface.y);
    return dGGX(n_dot_h, alpha)*(1.0/PI)*vSmithGGX(n_dot_l, n_dot_v, alpha)*fresnel*n_dot_l;
}

/// Cheap value noise to break up whitecaps.
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

// A random unit gradient at every lattice point.
fn latticeGradient(cell: vec2<f32>) -> vec2<f32>
{
    let angle = 6.2831853*hash21(cell);
    return vec2<f32>(cos(angle), sin(angle));
}

// Gradient noise, blended quintically: about 0 on average, within +-0.7,
// with a deviation of 0.215.
fn gradientNoise(p: vec2<f32>) -> f32
{
    let i = floor(p);
    let f = p - i;
    let u = f*f*f*(f*(f*6.0 - 15.0) + 10.0);
    let a = dot(latticeGradient(i), f);
    let b = dot(latticeGradient(i + vec2<f32>(1.0, 0.0)), f - vec2<f32>(1.0, 0.0));
    let c = dot(latticeGradient(i + vec2<f32>(0.0, 1.0)), f - vec2<f32>(0.0, 1.0));
    let d = dot(latticeGradient(i + vec2<f32>(1.0, 1.0)), f - vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

/// Two octaves is enough to break a contour line; this is not a detail layer.
fn foamBreakup(world_xz: vec2<f32>, footprint: f32) -> f32
{
    // Average subpixel noise away instead of letting distant caps sparkle.
    let a = mix(valueNoise(world_xz*0.35), 0.5, smoothstepf(0.25, 0.75, footprint*0.35));
    let b = mix(valueNoise(world_xz*1.10 + vec2<f32>(31.7, 11.3)), 0.5,
                smoothstepf(0.25, 0.75, footprint*1.10));
    return clamp(mix(0.55, 1.35, a*0.65 + b*0.35), 0.0, 1.4);
}

// Two scales of analytic rain impacts: larger drops raise a short crown and
// expanding capillary rings, while finer drops break up the gaps between them.
// The grid, phase and impact rate stay in world space as the camera moves. The
// rain field gates cells continuously as fronts pass, and distant subpixel
// impacts become slope variance rather than flashing dots. Snow melts on this
// unfrozen water and leaves only faint flecks, without rain-like rings.
struct WaterImpacts {
    slope: vec2<f32>,
    splash: f32,
    snow_fleck: f32,
    slope_variance: f32,
};

fn waterImpacts(world_xz: vec2<f32>, time: f32, footprint: f32,
                view_distance: f32, rain: f32, snow: f32, gust: f32) -> WaterImpacts {
    var result = WaterImpacts(vec2<f32>(0.0), 0.0, 0.0,
                              rain*rain*0.006 + snow*0.00025 + gust*gust*0.008);
    if (stage.flags.x > 0.5) { return result; }
    let visibility = 1.0 - smoothstepf(28.0, 95.0, view_distance);
    let resolved = 1.0 - smoothstepf(0.065, 0.22, footprint);
    if (visibility*resolved < 0.001) { return result; }

    // Sample the sheet that crossed this point half a second ago so an impact
    // ring completes its lifetime after the band has moved on. Current rain
    // still controls newly appearing splash crowns.
    var recent_rain = rain;
    if (globals.lightning_meta.w > 0.05) {
        let wind_direction = vec2<f32>(cos(globals.storm.w), sin(globals.storm.w));
        let earlier_position = vec3<f32>(world_xz.x + wind_direction.x*globals.lightning_meta.w*0.5,
                                          stage.params.z,
                                          world_xz.y + wind_direction.y*globals.lightning_meta.w*0.5);
        recent_rain = max(rain, weatherPrecipitation(earlier_position).x);
    }
    if (max(recent_rain, snow) < 0.005) { return result; }

    for (var layer = 0u; layer < 2u; layer += 1u) {
        let fine = layer == 1u;
        if (fine && recent_rain < 0.005) { continue; }
        let spacing = select(0.56, 0.32, fine);
        let base = floor(world_xz/spacing - vec2<f32>(0.5));
        let layer_seed = vec2<f32>(f32(layer)*81.17, f32(layer)*137.41);
        for (var z = 0i; z < 2i; z += 1i) {
            for (var x = 0i; x < 2i; x += 1i) {
                let cell = base + vec2<f32>(f32(x), f32(z));
                let key = cell + layer_seed;
                let seed = hash21(key + vec2<f32>(7.13, 41.9));
                let jitter = vec2<f32>(hash21(key + vec2<f32>(23.7, 18.2)),
                                        hash21(key + vec2<f32>(49.1, 71.4)));
                let centre = (cell + vec2<f32>(0.5) + (jitter - vec2<f32>(0.5))*0.32)*spacing;
                let delta = world_xz - centre;
                let distance_to_drop = length(delta);

                if (recent_rain > 0.005) {
                    let age = fract(time*select(1.7, 3.0, fine)
                                    + hash21(key + vec2<f32>(92.3, 12.7)));
                    let ring_rain = mix(rain, recent_rain,
                                        smoothstepf(0.05, 0.55, age));
                    let rain_active = smoothstepf(seed - 0.11, seed + 0.11, ring_rain);
                    if (rain_active > 0.001) {
                        let life = 1.0 - smoothstepf(0.52, 0.85, age);
                        let radius = select(0.021 + age*0.27, 0.012 + age*0.14, fine);
                        let width = max(select(0.019, 0.012, fine), footprint*0.55);
                        let ring_distance = (distance_to_drop - radius)/width;
                        let ring = exp(-1.4*ring_distance*ring_distance);
                        let echo_distance = (distance_to_drop - radius*0.58)/(width*1.4);
                        let echo = exp(-1.4*echo_distance*echo_distance);
                        let height = select(0.0026, 0.0011, fine)*rain_active*ring_rain*life;
                        let gradient = -height/width
                                       *(2.8*ring_distance*ring
                                         + 0.60*echo_distance*echo);
                        result.slope += delta/max(distance_to_drop, 0.001)*gradient;

                        let first_splash = 1.0 - smoothstepf(0.025, 0.16, age);
                        let core = 1.0 - smoothstepf(0.008,
                                                     select(0.050, 0.026, fine) + footprint*0.35,
                                                     distance_to_drop);
                        let splash_active = smoothstepf(seed - 0.11, seed + 0.11, rain);
                        result.splash += first_splash*core*splash_active*rain
                                         *select(0.70, 0.26, fine);
                    }
                }

                if (!fine && snow > 0.005) {
                    let snow_active = smoothstepf(seed - 0.05, seed + 0.28, snow*0.38);
                    if (snow_active > 0.001) {
                        let age = fract(time*0.62 + hash21(key + vec2<f32>(39.1, 63.7)));
                        let brief = 1.0 - smoothstepf(0.03, 0.23, age);
                        let soft_core = 1.0 - smoothstepf(0.015, 0.085 + footprint*0.5,
                                                          distance_to_drop);
                        result.snow_fleck += brief*soft_core*snow_active*snow;
                    }
                }
            }
        }
    }
    result.slope *= visibility*resolved;
    result.splash *= visibility*resolved;
    result.snow_fleck *= visibility*resolved;
    return result;
}

// ---------------------------------------------------------------------------
// The water body: what it is, and so what colour it is
// ---------------------------------------------------------------------------
//
// The sea's colour is its optics preset (src/water/optics.rs, uploaded as
// `stage.extinction` and `stage.scatter`). Inland water takes one of three
// natural colours, blended by what the water is and where it lies:
//
// - Forest stream water: clear, but stained the colour of weak tea by the
//   tannins and humic acids the rain leaches from the forest floor.
//   Dissolved organic matter absorbs steeply toward the blue (about 1.5/m at
//   440 nm, a fifth of that in the green), pure water takes the red, and a
//   little silt adds to all three. Through a hand's depth the bed shows
//   golden; through a metre its light is amber and a third of what it was; a
//   pool two metres deep is a dark brown. The silt's backscatter is nearly
//   grey, so a deep body keeps that dull amber-brown and never the sea's blue
//   or a swimming pool's green.
// - A pond gathers more of the forest's tannin and grows its own plankton:
//   browner and darker, a mirror over its depths with a golden margin.
// - Above the forest nothing stains the water. A mountain lake or stream is
//   snowmelt and spring water over bare rock: pure water's own absorption,
//   which takes the red (about 0.3/m at 620 nm) and leaves the blue and the
//   green, and a fine flour of ground rock held in suspension whose
//   backscatter leans a little toward the short wavelengths. Its shallows are
//   glass over the stones; a few metres down it turns turquoise, and its
//   depths a deep blue-cyan.
//
// Which a piece of water is, its clarity, is decided on the CPU per water
// body and handed over with it (`lake_clarity` in src/rivers/surface.rs):
// above the treeline every tarn is clear; among the thinning trees below it
// a lake large enough to hold its snowmelt clear is too, where a pond the
// same height stays humic; a stream is clear above the forest and carries
// the clarity of the lake it leaves down into the forest, where the stain
// takes over again.
const RIVER_EXTINCTION: vec3<f32> = vec3<f32>(0.40, 0.38, 1.05);
const RIVER_SCATTER: vec3<f32> = vec3<f32>(0.012, 0.012, 0.010);
const POND_EXTINCTION: vec3<f32> = vec3<f32>(0.55, 0.58, 1.60);
const POND_SCATTER: vec3<f32> = vec3<f32>(0.010, 0.010, 0.008);
const ALPINE_EXTINCTION: vec3<f32> = vec3<f32>(0.48, 0.13, 0.12);
const ALPINE_SCATTER: vec3<f32> = vec3<f32>(0.026, 0.031, 0.031);
// Bubbles beaten into whitewater scatter every colour alike and absorb none
// (per metre at full aeration); half of it is counted as leaving the eye ray.
const BUBBLE_SCATTER: f32 = 2.4;

// What a piece of water is. Every input is a share, so a river's last reach
// blends into the sea and a stream leaving a tarn carries its colour down.
struct WaterBody
{
    // How much of it is the open sea's water.
    sea: f32,
    // Still water (a lake, a pond or the sea) rather than a running stream.
    still: f32,
    // Clear mountain water rather than the forest's stained water: its
    // clarity.
    alpine: f32,
    // Bubbles beaten in by whitewater, 0..1.
    aeration: f32,
};

// Extinction and scattering, per metre.
struct WaterMedium
{
    sigma_t: vec3<f32>,
    sigma_s: vec3<f32>,
};

fn seaMedium() -> WaterMedium
{
    let sigma_t = max(stage.extinction.xyz, vec3<f32>(1e-4));
    let sigma_s = min(sigma_t, PARTICLE_SCATTER*stage.extinction.w*stage.scatter.xyz + RAYLEIGH);
    return WaterMedium(sigma_t, sigma_s);
}

fn inlandMedium(body: WaterBody) -> WaterMedium
{
    let stained_t = mix(RIVER_EXTINCTION, POND_EXTINCTION, body.still);
    let stained_s = mix(RIVER_SCATTER, POND_SCATTER, body.still);
    let sigma_t = mix(stained_t, ALPINE_EXTINCTION, body.alpine) + vec3<f32>(BUBBLE_SCATTER*body.aeration);
    let sigma_s = mix(stained_s, ALPINE_SCATTER, body.alpine) + vec3<f32>(0.5*BUBBLE_SCATTER*body.aeration);
    return WaterMedium(sigma_t, sigma_s);
}

// The medium around a submerged eye: its water's, the sea's blended in by
// its share.
fn waterMedium(body: WaterBody) -> WaterMedium
{
    let inland = inlandMedium(body);
    let sea = seaMedium();
    return WaterMedium(mix(inland.sigma_t, sea.sigma_t, body.sea), mix(inland.sigma_s, sea.sigma_s, body.sea));
}

// ---------------------------------------------------------------------------
// The wind sea: the waves the wind raises on any water
// ---------------------------------------------------------------------------
//
// Wind roughens every water body the same way; what differs is how much wind
// reaches the water and how far it has blown over it. Three measured
// relations set the waves:
//
// - Cox and Munk's sun-glitter surveys give the mean square slope of a wind
//   sea, 0.00512 per m/s of wind at mast height (their clean-surface fit,
//   less its 0.003 intercept, which is swell). Almost all of it lives in the
//   short waves, whose slope spectrum is saturated: each octave of
//   wavenumber between the peak and the capillary cutoff (1.7 cm, where
//   surface tension takes over) holds the same slope variance. That fixes the
//   saturation level from the wind alone.
// - JONSWAP gives the peak of a fetch-limited sea: the further the wind has
//   blown over the water, the longer its dominant waves, until the sea is
//   fully developed (Pierson-Moskowitz). Below the peak the spectrum falls
//   away, and at it a fetch-limited sea is peaked (JONSWAP's enhancement).
// - Under about 1 m/s no capillary ripples form at all, and the water is a
//   mirror.
//
// So the open sea takes all of the wind over unlimited fetch; a lake's waves
// grow from its upwind shore across it; a pond sheltered by the trees round
// it, or a creek in its cut, sees little wind and few waves; and a gust's
// cat's paw crosses any of them as a patch of ripples. The sea's longer waves
// are its Gerstner spectrum, so on the sea these waves stop at the 2 m that
// spectrum starts at.
//
// The waves are laid in half-octave rungs of four components, each moving
// with the capillary-gravity dispersion `w^2 = g k + (sigma/rho) k^3`. They
// head round the wind as a measured wind sea's do: a sech^2 spread, narrowest
// just past the peak and broader for the shorter waves (Donelan, Hamilton and
// Hui; Banner), but never broader than Cox and Munk found the short waves,
// whose slope runs 1.65 times as steep up the wind as across it. Where each
// component heads, its wavelength and its phase are fixed for the wind's
// heading (`stage.wind_waves`, drawn on the CPU from that broadest spread by
// `wind_sea_components` in src/water/waves.rs); the local wind and fetch only
// weight them, so where the spread is narrower the waves across the wind
// carry less. A gust or a nearer shore then changes how high the ripples run,
// never where their crests lie, and nothing re-phases them from one frame to
// the next. On running water they ride the current (`drift`). What a pixel
// cannot resolve, or a frame cannot follow, becomes roughness at the slope
// variance it has.
const WIND_SHORTEST: f32 = 0.04;
const CAPILLARY_CUTOFF: f32 = 0.017;
// Surface tension over density, m^3/s^2.
const SURFACE_TENSION: f32 = 7.28e-5;
// Mean square slope of a wind sea per m/s of wind (Cox and Munk).
const COX_MUNK_SLOPE: f32 = 0.00512;
// Half an octave of wavenumber, as a natural log: one rung's share.
const RUNG_LOG_SPAN: f32 = 0.34657359;
// The period, metres, every wind wave repeats over (see `windSea`): far too
// long to see repeat.
const WIND_PERIOD: f32 = 512.0;
// The broadest the wind sea is spread, the spread its components are drawn
// from (`SPREAD_FLOOR` in src/water/waves.rs).
const SPREAD_FLOOR: f32 = 0.955;
// A component's wavelength lies within a quarter octave of its rung's.
const WIND_RUNG_STRETCH: f32 = 1.1892071;

struct WindSea
{
    slope: vec2<f32>,
    variance: f32,
    height: f32,
    // Variance of the resolved height, to judge a crest by.
    height_variance: f32,
};

// The saturation level of the short-wave slope spectrum per unit of log
// wavenumber: Cox and Munk's mean square slope shared over the decades of
// wavenumber between a developed sea's peak and the capillary cutoff.
fn windSaturation(wind: f32) -> f32
{
    let u = max(wind, 0.5);
    let developed_peak = pow(2.0*PI*0.13*GRAVITY/u, 2.0)/GRAVITY;
    let decades = log((2.0*PI/CAPILLARY_CUTOFF)/developed_peak);
    return COX_MUNK_SLOPE*u/max(decades, 1.0)*smoothstepf(0.7, 2.6, wind);
}

// The peak wavenumber of a sea the wind has raised over `fetch` metres:
// JONSWAP's fetch-limited peak frequency, no lower than a fully developed
// sea's.
fn windPeakWavenumber(wind: f32, fetch: f32) -> f32
{
    let u = max(wind, 0.5);
    let reduced_fetch = GRAVITY*max(fetch, 1.0)/(u*u);
    let peak_frequency = max(3.5*GRAVITY/u*pow(reduced_fetch, -0.33), 0.13*GRAVITY/u);
    let omega = 2.0*PI*peak_frequency;
    return omega*omega/GRAVITY;
}

// How narrowly the waves at `frequency_ratio` times the peak's angular
// frequency are spread about the wind, as the `beta` of a sech^2(beta*theta)
// spread: Donelan, Hamilton and Hui's fit eased into Banner's, floored at
// SPREAD_FLOOR. Mirrors `spreading_beta` in src/water/waves.rs, which
// carries the measurements.
fn spreadingBeta(frequency_ratio: f32) -> f32
{
    let r = max(frequency_ratio, 1e-3);
    let donelan = select(2.28*pow(r, -0.65), 2.61*pow(r, 1.3), r < 0.95);
    let banner = pow(10.0, -0.4 + 0.8393*exp(-0.567*log(r*r)));
    return max(mix(donelan, banner, smoothstepf(1.6, 2.0, r)), SPREAD_FLOOR);
}

fn windSea(xz: vec2<f32>, drift: vec2<f32>, wind: f32, fetch: f32,
           longest: f32, time: f32, pixel_dx: vec2<f32>, pixel_dy: vec2<f32>, current: f32,
           frame_seconds: f32) -> WindSea
{
    var result = WindSea(vec2<f32>(0.0), 1e-5, 0.0, 0.0);
    let saturation = windSaturation(wind);
    if (saturation <= 0.0 || stage.flags.x > 0.5)
    {
        return result;
    }
    let k_peak = windPeakWavenumber(wind, fetch);
    let omega_peak = sqrt(GRAVITY*k_peak);
    // The capillary ripples under the shortest rung are always unresolved.
    result.variance += saturation*log(WIND_SHORTEST/CAPILLARY_CUTOFF);
    // Every wave vector is a whole number of cycles over WIND_PERIOD metres,
    // and the position is wrapped into one period first, so the phase stays
    // exact however far out the water lies: k*x itself passes a million
    // radians a few kilometres from the origin, beyond what a 32-bit float
    // can resolve.
    let p = xz - drift;
    let wrapped = p - WIND_PERIOD*floor(p/WIND_PERIOD);
    // The pixel's narrowest width in any direction: it resolves no wave
    // shorter than twice that, whichever way the wave runs.
    let finest = abs(pixel_dx.x*pixel_dy.y - pixel_dx.y*pixel_dy.x)
               / max(length(pixel_dx) + length(pixel_dy), 1e-6);
    for (var rung = 0u; rung < WIND_RUNGS; rung += 1u)
    {
        let nominal = WIND_SHORTEST*exp2(f32(rung)*0.5);
        if (nominal/WIND_RUNG_STRETCH > longest)
        {
            break;
        }
        let k = 2.0*PI/nominal;
        let below_peak = k_peak/k;
        let from_peak = log(k/k_peak);
        // Pierson-Moskowitz roll-off below the peak, JONSWAP's enhancement
        // at it (gamma 3.3 over a sigma of 0.08 in frequency, 0.16 in log
        // wavenumber).
        let shape = exp(-1.25*below_peak*below_peak)
                  * (1.0 + 2.3*exp(-from_peak*from_peak/(2.0*0.16*0.16)));
        let rung_variance = saturation*RUNG_LOG_SPAN*shape;
        if (rung_variance < 1e-7)
        {
            if (below_peak > 1.0) { break; }
            continue;
        }
        // Every wave of the rung is too short for the pixel: all of it is
        // roughness, and none of it needs evaluating.
        let rung_longest = nominal*WIND_RUNG_STRETCH*1.01;
        if (finest >= 0.5*rung_longest && rung_longest <= 0.85*longest)
        {
            result.variance += rung_variance;
            continue;
        }
        // The components were drawn from the broadest spread. Each carries
        // the ratio of the rung's own spread to that one at its heading, and
        // the rung's slope variance is shared out by those weights; at the
        // broadest spread itself, which is every rung well short of the
        // peak, they share it equally.
        let omega = sqrt(GRAVITY*k + SURFACE_TENSION*k*k*k);
        let beta = spreadingBeta(omega/omega_peak);
        var weights: array<f32, WIND_PER_RUNG>;
        var total = 0.0;
        for (var j = 0u; j < WIND_PER_RUNG; j += 1u)
        {
            weights[j] = 1.0;
            if (beta > SPREAD_FLOOR)
            {
                let heading = stage.wind_waves[rung*WIND_PER_RUNG + j].w;
                let ratio = cosh(SPREAD_FLOOR*heading)/cosh(beta*heading);
                weights[j] = ratio*ratio;
            }
            total += weights[j];
        }
        for (var j = 0u; j < WIND_PER_RUNG; j += 1u)
        {
            let wave = stage.wind_waves[rung*WIND_PER_RUNG + j];
            let cycles = max(length(wave.xy), 1.0);
            let direction = wave.xy/cycles;
            let wavelength = WIND_PERIOD/cycles;
            let k_wave = 2.0*PI/wavelength;
            let omega_wave = sqrt(GRAVITY*k_wave + SURFACE_TENSION*k_wave*k_wave*k_wave);
            // A frame must not carry a crest more than a fraction of its
            // wavelength, or it seems to run backwards.
            let per_frame = (omega_wave/k_wave + current)*frame_seconds/wavelength;
            let steady = 1.0 - smoothstepf(0.22, 0.42, per_frame);
            // Up to the longest wave this water raises, they fade out rather
            // than stop.
            let kept = 1.0 - smoothstepf(0.85*longest, longest, wavelength);
            let cycles_per_pixel = max(abs(dot(direction, pixel_dx)), abs(dot(direction, pixel_dy)))/wavelength;
            let resolved = (1.0 - smoothstepf(0.125, 0.5, cycles_per_pixel))*steady;
            let component_variance = rung_variance*kept*weights[j]/total;
            result.variance += component_variance*(1.0 - resolved*resolved);
            if (resolved <= 0.0 || component_variance <= 0.0)
            {
                continue;
            }
            // Slope variance (a k)^2 / 2 per component.
            let amplitude = sqrt(2.0*component_variance)/k_wave;
            let phase = 6.2831853*(fract(dot(wave.xy, wrapped)/WIND_PERIOD) + wave.z) - omega_wave*time;
            result.slope -= direction*(amplitude*k_wave*sin(phase)*resolved);
            result.height += amplitude*cos(phase)*resolved;
            result.height_variance += 0.5*amplitude*amplitude*resolved*resolved;
        }
    }
    return result;
}

// Gusts cross the water as cat's paws: patches of ripples blown along at
// about the wind's speed and drawn out down it, some three times as long as
// they are wide, as the streaks a gust leaves on water are. The field lives
// in the wind's own frame, so no patch lines up with the world's axes, and
// is gradient noise, whose blobs have no corners. Its cells are as wide
// across the wind as the value noise's it replaced and three times as long,
// which keeps the paws as large and the gusts as strong. About 0.5 on
// average, and within 0..1 over most of the water.
const GUST_STRETCH: f32 = 3.0;

fn windGust(xz: vec2<f32>, wind_direction: vec2<f32>, wind: f32, time: f32) -> f32
{
    let speed = max(wind, 1.0);
    let frame = vec2<f32>(dot(xz, wind_direction), dot(xz, vec2<f32>(-wind_direction.y, wind_direction.x)));
    let broad = (10.0 + 2.5*speed)*vec2<f32>(GUST_STRETCH, 1.0);
    let fine = (4.0 + speed)*vec2<f32>(GUST_STRETCH, 1.0);
    return 0.5 + 0.65*gradientNoise((frame - vec2<f32>(time*speed*0.8, 0.0))/broad)
               + 0.35*gradientNoise((frame - vec2<f32>(time*speed, 0.0))/fine + vec2<f32>(7.3, 1.9));
}

// How much of the wind reaches the water, and over what fetch. Walk upwind
// from the water to its upwind shore: the fetch is the distance to it, and
// whatever stands there, a bank or a hill and the crowns on it, casts a wind
// shadow some ten times its height downwind, out of which the wind comes
// back down to the water. A channel's banks bound the fetch across it even
// beyond the one-metre map, which does not see a creek's own cut. Crowns
// over the water itself shelter it as well.
struct WindExposure
{
    fetch: f32,
    shelter: f32,
};

fn windExposure(xz: vec2<f32>, level: f32, upwind: vec2<f32>, open_sky: f32, sea: f32,
                across: f32, half_width: f32, cross_stream: vec2<f32>) -> WindExposure
{
    // No shore within reach: the open sea, whose wind sea is fully developed
    // a hundred kilometres out, or a lake wider than the walk, which in this
    // landscape is a few hundred metres at most.
    var fetch = mix(400.0, 1.0e5, sea);
    var obstacle = 0.0;
    var previous = 0.0;
    for (var i = 0u; i < 8u; i += 1u)
    {
        // 3 m out to 235 m.
        let distance = 3.0*exp2(f32(i)*0.9);
        let p = xz + upwind*distance;
        let ground = riverGround(p).x;
        if (ground > level + 0.05)
        {
            fetch = 0.5*(previous + distance);
            let canopy = riverCanopy(p);
            obstacle = ground - level + (1.0 - canopy.a)*CANOPY_HEIGHT;
            break;
        }
        previous = distance;
    }
    if (half_width > 0.01)
    {
        let sideways = dot(upwind, cross_stream);
        let to_bank = half_width*(1.0 - across*sign(sideways))/max(abs(sideways), 0.05);
        let bank_fetch = min(max(to_bank, 0.5), 40.0*half_width);
        if (bank_fetch < fetch)
        {
            fetch = bank_fetch;
            let canopy = riverCanopy(xz + upwind*(bank_fetch + 2.0));
            obstacle = max(obstacle, 0.6 + 0.15*half_width + (1.0 - canopy.a)*CANOPY_HEIGHT);
        }
    }
    // Gusts still find their way down into the lee, so even a pond ringed by
    // trees feels a little of the wind.
    let shadow = max(smoothstepf(3.0*obstacle, 14.0*obstacle + 2.0, fetch), 0.15);
    return WindExposure(fetch, shadow*mix(0.25, 1.0, open_sky));
}

// ---------------------------------------------------------------------------
// The current's own surface
// ---------------------------------------------------------------------------

// Value noise with its analytic gradient (x value, yz d/dp), quintic.
fn riverNoise(p: vec2<f32>) -> vec3<f32>
{
    let i = floor(p);
    let f = p - i;
    let u = f*f*f*(f*(f*6.0 - 15.0) + 10.0);
    let du = 30.0*f*f*(f*(f - 2.0) + 1.0);
    let a = hash21(i);
    let b = hash21(i + vec2<f32>(1.0, 0.0));
    let c = hash21(i + vec2<f32>(0.0, 1.0));
    let d = hash21(i + vec2<f32>(1.0, 1.0));
    let k = a - b - c + d;
    let value = a + (b - a)*u.x + (c - a)*u.y + k*u.x*u.y;
    let gradient = du*vec2<f32>(b - a + k*u.y, c - a + k*u.x);
    return vec3<f32>(value, gradient);
}

// Mean square of riverNoise's gradient along one axis, per unit of its
// domain (quintic value noise over uniform hashes; measured). The slope a
// pixel cannot resolve is moved into roughness at exactly this variance.
const RIVER_NOISE_SLOPE_VARIANCE: f32 = 0.187;
// Metres between a rapid's steps, and over which the current's tongue
// wanders across its channel. Constant, not in the channel's widths: the
// half width varies along the ribbon, and dividing the distance from the
// river's head by it would scramble the pattern's scale far downstream.
const RIVER_STEP_SPACING: f32 = 4.0;
const RIVER_TONGUE_WANDER: f32 = 18.0;

// Standard deviation of valueNoise about its mean of 0.5 (cubic-smoothed
// value noise over uniform hashes; measured).
const VALUE_NOISE_DEVIATION: f32 = 0.214;

// How much of a value-noise octave a pixel resolves, from its footprint in
// the octave's cells along each axis: all of it under a quarter of a cell,
// none from half a cell, where the octave's finest detail passes a pixel.
fn octaveResolved(cells: vec2<f32>) -> f32
{
    return 1.0 - smoothstepf(0.25, 0.5, max(cells.x, cells.y));
}

// Where a rapid's bed steps, at metres `along` its channel and `across` it in
// half widths: ledges a few metres apart drawn out downstream, broken by
// boulders at a third of their scale. x is the pattern as the pixel, of
// `footprint` (metres along, half widths across), resolves it, about 0.5;
// y is the variance of the detail it cannot, which is not dropped but
// averaged over (see riverStepShares). Unfiltered, the boulders' cells, a
// quarter of a half width across, fall under a pixel some tens of metres off
// and the whitewater over them crawls and sparkles as the camera moves.
fn riverSteps(along: f32, across: f32, footprint: vec2<f32>) -> vec2<f32>
{
    let ledge_resolved = octaveResolved(footprint*vec2<f32>(1.0/(1.6*RIVER_STEP_SPACING), 1.7));
    let boulder_resolved = octaveResolved(footprint*vec2<f32>(1.0/(0.45*RIVER_STEP_SPACING), 4.1));
    var ledges = 0.5;
    var boulders = 0.5;
    if (ledge_resolved > 0.0)
    {
        ledges = valueNoise(vec2<f32>(along/(1.6*RIVER_STEP_SPACING), across*1.7 + 3.1));
    }
    if (boulder_resolved > 0.0)
    {
        boulders = valueNoise(vec2<f32>(along/(0.45*RIVER_STEP_SPACING) + 7.3, across*4.1 + 1.7));
    }
    let value = 0.5 + (ledges - 0.5)*1.15*ledge_resolved + (boulders - 0.5)*0.55*boulder_resolved;
    let lost = VALUE_NOISE_DEVIATION*VALUE_NOISE_DEVIATION
             * (1.15*1.15*(1.0 - ledge_resolved*ledge_resolved)
                + 0.55*0.55*(1.0 - boulder_resolved*boulder_resolved));
    return vec2<f32>(value, lost);
}

// How much of each of three equally likely thirds of a pixel lies below a
// step, 0..1: from the steps as the pixel resolves them (`steps`,
// riverSteps' x) and the variance of the detail it does not (`lost`), which
// spreads the thirds about it. Each third sits at the middle of its third of
// value noise's distribution, 1.05 standard deviations out (measured; a
// normal distribution's would be 0.97). Whatever is made of the steps is
// made of each third and averaged, never of their mean: the whitewater over
// the steps is a threshold, which the mean of a pixel half below steps and
// half over tongues would never reach, and the white would go out of a
// rapid far off. A resolved pixel's thirds are one and the same.
fn riverStepShares(steps: f32, lost: f32) -> vec3<f32>
{
    let spread = 1.05*sqrt(lost);
    return vec3<f32>(smoothstepf(0.25, 0.75, steps - spread),
                     smoothstepf(0.25, 0.75, steps),
                     smoothstepf(0.25, 0.75, steps + spread));
}

// One cycle of a flow-mapped layer: a texture carried by the current for
// `period` seconds and then laid down afresh, two such layers half a cycle
// apart so one is always at its height while the other is renewed. Their
// weights are sin and cos of the cycle, whose squares sum to one: two
// independent patterns blended so keep the contrast of one, and the water
// does not pulse as they hand over.
struct FlowLayer
{
    offset: vec2<f32>,
    jitter: vec2<f32>,
    weight: f32,
};

fn flowLayer(time: f32, period: f32, layer: u32, seed: f32, flow: vec2<f32>) -> FlowLayer
{
    let cycle_time = time/period + f32(layer)*0.5 + seed;
    let phase = fract(cycle_time);
    let cycle = floor(cycle_time);
    let jitter = vec2<f32>(hash21(vec2<f32>(cycle, f32(layer)*17.3 + seed*31.0)),
                           hash21(vec2<f32>(f32(layer)*5.1 + seed*13.0, cycle)))*31.0;
    return FlowLayer(flow*(phase*period), jitter, sin(PI*phase));
}

struct FlowSurface
{
    // Height gradient in the channel's frame: (downstream, across).
    slope: vec2<f32>,
    // Slope variance the pixel (or the frame) cannot resolve.
    variance: f32,
    // 0..1 on the faces of standing waves, where a riffle's flecks sit.
    crest: f32,
};

// What the current does to the surface, in the channel's own frame `st`
// (metres downstream and across; a lake's is world XZ), with `flow` the
// current in that frame.
//
// - Turbulence welling up from the bed breaks the surface into small
//   ripples the current carries. Their slope grows with the bed's roughness
//   and the current (RMS about 0.02 on a run, 0.07 on a riffle, 0.17 in
//   whitewater), and fast water draws them out into streaks along the flow.
//   Each scale is carried on its own cycle, so the hand-overs between its
//   layers never line up across scales.
// - Boils: in runs and pools the turbulence of the reach above wells up as
//   glassy domes a couple of metres across, ringed by ripples, drifting
//   down.
// - Standing waves over riffles, fixed to the bed: a gravity wave holds
//   still against a current running at its own phase speed, so its
//   wavelength is 2*pi*U^2/g (0.6 m at 1 m/s, 1.4 m at 1.5 m/s). Crests lie
//   across the flow, kinked and broken by the bed, and swell and ebb in
//   place: their envelope morphs between two fixed patterns rather than
//   travelling, since anything travelling over a riffle reads as the water's
//   own motion, and a swell that drifted upstream made the river look as if
//   it ran backwards.
//
// A texture the current carries further in one frame than about a third of
// its own scale can no longer be followed by the eye (it strobes, and seems
// to run backwards); it goes to roughness instead, as an unresolved one does.
fn flowSurface(st: vec2<f32>, flow: vec2<f32>, time: f32, roughness: f32, riffle: f32,
               river: bool, footprint: vec2<f32>, frame_seconds: f32) -> FlowSurface
{
    var result = FlowSurface(vec2<f32>(0.0), 0.0, 0.0);
    let speed = length(flow);
    // Only a channel has a frame to stretch its ripples along.
    let stretch = select(1.0, 1.0 + 1.8*smoothstepf(0.3, 1.8, speed)*(1.0 - 0.6*roughness), river);
    let steepness = 0.018 + 0.20*roughness + 0.012*min(speed, 3.0);
    let travel = speed*frame_seconds;

    // Boils, and how they stir the ripples: glassy on their domes, stirred
    // at their rims.
    let boils = select(0.0, smoothstepf(0.3, 1.0, speed)*(1.0 - smoothstepf(0.25, 0.6, roughness)), river);
    var stir = 1.0;
    if (boils > 0.001)
    {
        let boil_resolved = (1.0 - smoothstepf(0.4, 1.2, max(footprint.x, footprint.y)))
                          * (1.0 - smoothstepf(0.25, 0.45, travel/2.2));
        let boil_slope = 0.035/2.2;
        var dome = 0.0;
        var rim = 0.0;
        for (var layer = 0u; layer < 2u; layer += 1u)
        {
            let cycle = flowLayer(time, 2.6, layer, 0.61, flow);
            let boil = riverNoise((st - cycle.offset)*(1.0/2.2) + cycle.jitter*0.37 + vec2<f32>(3.1, 8.9));
            let share = cycle.weight*cycle.weight;
            dome += smoothstepf(0.58, 0.85, boil.x)*share;
            rim += (1.0 - abs(2.0*smoothstepf(0.45, 0.75, boil.x) - 1.0))*share;
            result.slope += boil.yz*(boil_slope*boils*cycle.weight*boil_resolved);
        }
        result.variance += boil_slope*boil_slope*RIVER_NOISE_SLOPE_VARIANCE*2.0*boils*boils
                         *(1.0 - boil_resolved*boil_resolved);
        stir = 1.0 - 0.75*dome*boils + 0.6*rim*boils;
    }

    for (var octave = 0u; octave < 3u; octave += 1u)
    {
        let wavelength = select(select(0.85, 0.32, octave == 1u), 0.12, octave == 2u);
        let period = select(select(2.3, 1.45, octave == 1u), 0.95, octave == 2u);
        let c = steepness*stir*select(1.0, 0.8, octave == 2u);
        let along_scale = wavelength*stretch;
        let resolved = (1.0 - smoothstepf(0.18, 0.6, footprint.x/along_scale))
                     * (1.0 - smoothstepf(0.18, 0.6, footprint.y/wavelength))
                     * (1.0 - smoothstepf(0.22, 0.4, travel/along_scale));
        result.variance += c*c*RIVER_NOISE_SLOPE_VARIANCE*(1.0/(stretch*stretch) + 1.0)
                         *(1.0 - resolved*resolved);
        if (resolved <= 0.0)
        {
            continue;
        }
        for (var layer = 0u; layer < 2u; layer += 1u)
        {
            let cycle = flowLayer(time, period, layer, f32(octave)*0.37, flow);
            let advected = st - cycle.offset;
            let q = vec2<f32>(advected.x/along_scale, advected.y/wavelength)
                  + cycle.jitter + vec2<f32>(f32(octave)*7.7);
            let noise = riverNoise(q);
            // Height c*wavelength*noise: d/ds = c*noise.y/stretch, d/dt = c*noise.z.
            result.slope += vec2<f32>(noise.y/stretch, noise.z)*(c*cycle.weight*resolved);
        }
    }
    if (riffle > 0.001)
    {
        // Wavelengths are taken between fixed half-octave rungs, so a crest
        // stays continuous where the current slows toward the banks.
        let level = 2.0*log2(clamp(0.64*speed*speed, 0.35, 2.5)/0.35);
        let base = floor(level);
        // The two fixed envelopes the swell morphs between, in place.
        let ebb = 0.5 + 0.5*sin(time*0.9);
        for (var k = 0u; k < 2u; k += 1u)
        {
            let rung = base + f32(k);
            let share = select(1.0 - (level - base), level - base, k == 1u);
            let wavelength = 0.35*exp2(rung*0.5);
            let seed = vec2<f32>(rung*13.7, rung*7.1);
            let kink = valueNoise(vec2<f32>(st.x/(4.0*wavelength), st.y/(1.3*wavelength)) + seed);
            let lengths = smoothstepf(0.3, 0.7, valueNoise(vec2<f32>(st.x/(3.0*wavelength),
                                                                     st.y/(0.9*wavelength)) + seed.yx));
            let swell_a = valueNoise(vec2<f32>(st.x/(2.0*wavelength), st.y/(1.1*wavelength)) + seed*1.9);
            let swell_b = valueNoise(vec2<f32>(st.x/(2.0*wavelength), st.y/(1.1*wavelength)) + seed*2.7 + 5.3);
            let pulse = 0.72 + 0.28*mix(swell_a, swell_b, ebb);
            let wave = 6.2831853*fract(st.x/wavelength + 0.9*kink);
            let amplitude = 0.20*riffle*lengths*pulse*share;
            let resolved = 1.0 - smoothstepf(0.2, 0.6, footprint.x/wavelength);
            result.slope.x += amplitude*cos(wave)*resolved;
            result.variance += 0.5*amplitude*amplitude*(1.0 - resolved*resolved);
            result.crest += smoothstepf(0.55, 0.95, sin(wave))*lengths*share*resolved;
        }
    }
    return result;
}

// Foam carried downstream in the channel's frame and drawn out into streaks
// along the current, 0..1 about a mean of 0.5. Its two layers hand over as
// the ripples' do, so the foam keeps its contrast through the hand-over, and
// its fine bubbles average away once the current carries them too far a
// frame to follow.
fn riverFoamPattern(st: vec2<f32>, flow: vec2<f32>, stretch: f32, time: f32,
                    footprint: vec2<f32>, frame_seconds: f32) -> f32
{
    let squeeze = vec2<f32>(1.0/stretch, 1.0);
    let extent = max(footprint.x/stretch, footprint.y);
    let travel = length(flow)*frame_seconds/stretch;
    let coarse_blur = max(smoothstepf(0.25, 0.75, extent*2.1), smoothstepf(0.1, 0.2, travel*2.1));
    let fine_blur = max(smoothstepf(0.25, 0.75, extent*7.3), smoothstepf(0.1, 0.2, travel*7.3));
    var pattern = 0.5;
    for (var layer = 0u; layer < 2u; layer += 1u)
    {
        let cycle = flowLayer(time, 2.2, layer, 0.13, flow);
        let q = (st - cycle.offset)*squeeze + cycle.jitter*0.74;
        let coarse = mix(valueNoise(q*2.1), 0.5, coarse_blur);
        let bubbles = mix(valueNoise(q*7.3 + vec2<f32>(13.1, 4.7)), 0.5, fine_blur);
        pattern += (coarse*0.62 + bubbles*0.38 - 0.5)*cycle.weight;
    }
    return clamp(pattern, 0.0, 1.0);
}

// The foam on a stream where `amount` of it is made or carried, as the
// pattern of its lace (riverFoamPattern) covers it, and far off, where that
// pattern is unresolved, the share of the water it covers.
fn streamFoam(amount: f32, pattern: f32, resolved: f32) -> f32
{
    let made = clamp(amount, 0.0, 1.0);
    return mix(smoothstepf(1.05 - made, 1.3 - made*0.6, 0.58),
               smoothstepf(1.05 - made, 1.3 - made*0.6, pattern), resolved)*made;
}

// ---------------------------------------------------------------------------
// What the water sees and what shades it
// ---------------------------------------------------------------------------

// The plants' shadow cascades, as the composite reads them, so water under
// the forest lies in the same shade as its banks. Mirrors
// `VegetationShadows` in composite.wgsl.
struct VegetationShadows {
    view_projection: array<mat4x4<f32>, 3>,
    splits: vec4<f32>, // horizontal radius of each cascade; w far radius, metres
    texel: vec4<f32>,
    light: vec4<f32>,
    params: vec4<f32>,
};
@group(2) @binding(1) var vegetation_shadow_map_0: texture_depth_2d;
@group(2) @binding(4) var vegetation_shadow_map_1: texture_depth_2d;
@group(2) @binding(5) var vegetation_shadow_map_2: texture_depth_2d;
@group(2) @binding(2) var<uniform> vegetation_shadows: VegetationShadows;
@group(2) @binding(3) var vegetation_shadow_sampler: sampler_comparison;

fn plantShadowPcf(map: texture_depth_2d, uv: vec2<f32>, depth: f32) -> f32 {
    let step = 1.0/f32(textureDimensions(map).x);
    var lit = 0.0;
    for (var y = 0; y < 2; y++) {
        for (var x = 0; x < 2; x++) {
            let offset = vec2<f32>(f32(x) - 0.5, f32(y) - 0.5)*step*1.5;
            lit += textureSampleCompareLevel(map, vegetation_shadow_sampler,
                                             uv + offset, depth);
        }
    }
    return lit;
}

fn plantShadowCascade(world_position: vec3<f32>, cascade: u32) -> f32 {
    let texel = vegetation_shadows.texel[cascade];
    let lookup = world_position + vec3<f32>(0.0, texel*1.5, 0.0) - vegetation_shadows.light.xyz*texel*1.5;
    let clip = vegetation_shadows.view_projection[cascade]*vec4<f32>(lookup, 1.0);
    let uv = vec2<f32>(0.5 + 0.5*clip.x, 0.5 - 0.5*clip.y);
    if (any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0)) || clip.z > 1.0) {
        return 1.0;
    }
    switch cascade {
        case 0u: { return plantShadowPcf(vegetation_shadow_map_0, uv, clip.z)*0.25; }
        case 1u: { return plantShadowPcf(vegetation_shadow_map_1, uv, clip.z)*0.25; }
        default: { return plantShadowPcf(vegetation_shadow_map_2, uv, clip.z)*0.25; }
    }
}

// Light reaching a water point through the plants: the cascade covering its
// horizontal distance from the camera, blended into the next as on the banks,
// with a 2x2 grid of bilinear comparisons (the composite's 4x4 tent is more
// than a rippling surface needs).
fn plantShadow(world_position: vec3<f32>) -> f32 {
    let horizontal_distance = length(world_position.xz - globals.camera_position.xz);
    if (vegetation_shadows.light.w < 0.5 || horizontal_distance >= vegetation_shadows.splits.w) {
        return 1.0;
    }
    var cascade = 2u;
    for (var c = 0u; c < 2u; c++) {
        if (horizontal_distance < vegetation_shadows.splits[c]) {
            cascade = c;
            break;
        }
    }
    var lit = plantShadowCascade(world_position, cascade);
    if (cascade < 2u) {
        let start = select(0.0, vegetation_shadows.splits[max(cascade, 1u) - 1u], cascade > 0u);
        let end = vegetation_shadows.splits[cascade];
        let band = (end - start)*vegetation_shadows.params.x;
        let blend = smoothstep(0.0, 1.0, (horizontal_distance - (end - band))/max(band, 1e-3));
        if (blend > 0.0) {
            lit = mix(lit, plantShadowCascade(world_position, cascade + 1u), blend);
        }
    }
    let far_fade = smoothstepf(vegetation_shadows.splits.w*0.85, vegetation_shadows.splits.w, horizontal_distance);
    return mix(1.0, mix(lit, 1.0, far_fade), vegetation_shadows.params.y);
}

// What a reflection meets where it does not meet the sky: foliage seen from
// the water, and how tall a closed canopy stands over its ground.
const CANOPY_ALBEDO: vec3<f32> = vec3<f32>(0.05, 0.085, 0.035);
const CANOPY_HEIGHT: f32 = 18.0;
// Open ground of no particular colour, outside the canopy capture.
const OPEN_GROUND: vec4<f32> = vec4<f32>(0.10, 0.12, 0.06, 1.0);

// The ground at `world_xz` (x) and how much of it the camera's one-metre map
// gave (y). That map holds the carved channels and their banks; beyond it
// only the twelve-metre lighting map remains, which sees valleys but not a
// creek's own cut.
fn riverGround(world_xz: vec2<f32>) -> vec2<f32>
{
    var coarse = -1.0e4;
    if (globals.heightfield.z >= 1.0)
    {
        coarse = terrainHeightAt(world_xz);
    }
    if (stage.shore_map.z < 1.0)
    {
        return vec2<f32>(coarse, 0.0);
    }
    let uv = (world_xz - stage.shore_map.xy)/stage.shore_map.z + vec2<f32>(0.5);
    let edge = min(min(uv.x, uv.y), min(1.0 - uv.x, 1.0 - uv.y));
    let weight = smoothstepf(0.0, 0.03, edge);
    let fine = textureSampleLevel(shoreline_heightfield, shoreline_sampler,
                                  clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)), 0.0).r;
    return vec2<f32>(mix(coarse, fine, weight), weight);
}

// The ground's colour (rgb) and the share of open sky the tree crowns leave
// over it (a): the grass capture, whose alpha vegetation.wgsl's canopy pass
// darkens under every crown. Open ground outside it.
fn riverCanopy(world_xz: vec2<f32>) -> vec4<f32>
{
    if (canopy_window.z < 1.0)
    {
        return OPEN_GROUND;
    }
    let uv = (world_xz - canopy_window.xy)/canopy_window.z + vec2<f32>(0.5);
    let edge = min(min(uv.x, uv.y), min(1.0 - uv.x, 1.0 - uv.y));
    let texel = textureSampleLevel(canopy_texture, canopy_sampler,
                                   clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)), 0.0);
    return mix(OPEN_GROUND, vec4<f32>(texel.rgb, clamp(texel.a, 0.0, 1.0)),
               smoothstepf(0.0, 0.03, edge));
}

struct Surroundings
{
    // Share of the reflected lobe that reaches open sky.
    sky: f32,
    // Radiance of the banks and crowns that hide the rest.
    occluder: vec3<f32>,
    // Open sky over the water itself, for its ambient light.
    sky_view: f32,
};

// What a reflection sees where the screen cannot show it. Walk out from the
// water along the reflected ray's bearing and find the highest elevation the
// banks and the crowns on them reach: the ground, plus CANOPY_HEIGHT where
// the canopy capture is closed. A ray below that elevation meets the bank or
// the forest, not the sky. Beyond the one-metre map a river's own cut stands
// in: banks rising from about 0.35 m to 1.35 m over the two metres past its
// waterline. So a creek in its cut under the trees mirrors dark banks and
// foliage everywhere but up its own corridor, a pond in a clearing keeps its
// sky, and the sea mirrors its wooded headlands near the shore and the open
// sky beyond them.
fn surroundings(world_xz: vec2<f32>, level: f32, reflection: vec3<f32>, alpha: f32,
                here: vec4<f32>, across: f32, half_width: f32, cross_stream: vec2<f32>,
                sun_open: vec3<f32>, to_sun: vec3<f32>,
                world_position: vec3<f32>) -> Surroundings
{
    let run = length(reflection.xz);
    let bearing = select(vec2<f32>(1.0, 0.0), reflection.xz/max(run, 1e-4), run > 1e-4);
    // Tangent of the reflected ray's elevation.
    let rise = reflection.y/max(run, 1e-3);
    var horizon = 0.0;
    var crown_share = 0.0;
    var bank_albedo = here.rgb;
    var fine = 1.0;
    for (var i = 0u; i < 6u; i += 1u)
    {
        // 1.5 m out to 81 m.
        let distance = 1.5*exp2(f32(i)*1.15);
        let p = world_xz + bearing*distance;
        let ground = riverGround(p);
        let canopy = riverCanopy(p);
        let crown = (1.0 - canopy.a)*CANOPY_HEIGHT;
        let top = ground.x + crown - level;
        if (top > horizon*distance)
        {
            horizon = top/distance;
            crown_share = clamp(crown/max(top, 0.5), 0.0, 1.0);
            bank_albedo = canopy.rgb;
        }
        if (i == 0u)
        {
            fine = ground.y;
        }
    }
    if (half_width > 0.01)
    {
        let sideways = dot(bearing, cross_stream);
        let to_bank = half_width*(1.0 - across*sign(sideways))/max(abs(sideways), 0.05);
        horizon = max(horizon, 1.35/(max(to_bank, 0.0) + 2.0)*(1.0 - fine));
    }
    // The lobe's spread, in the same tangent measure, widened by how little
    // six taps and a canopy map can say about where the horizon really is:
    // a crown's silhouette is not a line, and on glassy water a hard edge
    // would cut the trees' reflection out of the sky as one flat blot.
    let spread = (0.04 + 1.5*alpha)*(1.0 + rise*rise) + 0.3*horizon;
    var sky = smoothstepf(horizon - spread, horizon + spread, rise);
    // Crowns over the water itself hide even the steep rays.
    sky *= mix(1.0, here.a, smoothstepf(0.3, 1.5, rise));
    // The banks and crowns face the water: lit by the sky's fill, and by
    // the sun where it falls on the side the ray meets.
    let facing = normalize(vec3<f32>(-bearing.x, 0.35, -bearing.y));
    let albedo = mix(bank_albedo, CANOPY_ALBEDO, crown_share);
    let occluder = albedo*(sun_open*(max(dot(facing, to_sun), 0.0)*0.45/PI)
                           + skyAmbient(facing, world_position)*0.35);
    let sky_view = here.a*(0.65 + 0.35/(1.0 + horizon*horizon));
    return Surroundings(sky, occluder, sky_view);
}

// ---------------------------------------------------------------------------
// Vertex stages
// ---------------------------------------------------------------------------

// What every surface hands the fragment stage, whichever mesh it came from.
struct WaterVertexOutput
{
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    // Where the waves are evaluated: on the sea its Gerstner parameter
    // position, before the horizontal chop, so fragment crests and their
    // displaced geometry have the same phase; elsewhere the plain world XZ.
    @location(1) wave_position: vec2<f32>,
    // The surface current, world XZ, m/s.
    @location(2) velocity: vec2<f32>,
    // x across the channel (-1..1 between the waterlines, 0 off a channel),
    // y whitewater, z the water's own level, w still water (1 on a lake's
    // sheet and on the sea, and on a river where it meets them).
    @location(3) channel: vec4<f32>,
    // x metres downstream from the river's head, y its half width (0 on a
    // lake's sheet and on the sea), z foam drifting down from whitewater
    // upstream, w how far the water is the sea's (1 on the sea and where a
    // river meets it).
    @location(4) stream: vec4<f32>,
    // Metres across the channel in its own frame.
    @location(5) side: f32,
    // Where a tributary's water becomes its parent's: the parent's frame
    // (xy) and how far its ripples and foam are laid in it (z).
    @location(6) joined: vec3<f32>,
    // How clear the water is: 0 stained by the forest, 1 clear mountain
    // water. The sea's colour is its own, so 0 there.
    @location(7) clarity: f32,
};

// The sea: one of Crest's concentric tiles, displaced by the Gerstner sum.
@vertex
fn vs_sea(
    @location(0) local: vec2<f32>,
    @location(1) tile_centre: vec2<f32>,
    @location(2) tile_scale: f32,
    @location(3) tile_rotation: f32,
) -> WaterVertexOutput
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
    // vertex's displacement all the way to the horizon. Fade geometry across
    // the last regular tiles, before the skirt starts. Fragment lighting has
    // its own pixel limit and continues smoothly over this flat geometry.
    // The shared distance includes the ring centre's camera-snap allowance.
    let wave_weight = 1.0 - smoothstepf(stage.flags.w*0.75, stage.flags.w, distance);
    let surface = sampleSurface(world_xz, sampleFootprint(distance, pixel_angle),
                                vec2<f32>(0.0), vec2<f32>(0.0), wave_weight, sampleShore(world_xz));
    let world_position = vec3<f32>(surface.displaced.x,
                                   stage.params.z + surface.displaced.y,
                                   surface.displaced.z);

    var out: WaterVertexOutput;
    out.clip_position = globals.projection*globals.view*vec4<f32>(world_position, 1.0);
    out.world_position = world_position;
    out.wave_position = world_xz;
    out.velocity = vec2<f32>(0.0);
    out.channel = vec4<f32>(0.0, 0.0, stage.params.z, 1.0);
    out.stream = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    out.side = 0.0;
    out.joined = vec3<f32>(0.0);
    out.clarity = 0.0;
    return out;
}

// A river's ribbon or a lake's sheet: `SurfaceVertex` in
// src/rivers/surface.rs.
@vertex
fn vs_inland(
    @location(0) position: vec3<f32>,
    @location(1) velocity: vec2<f32>,
    @location(2) across: f32,
    @location(3) turbulence: f32,
    @location(4) still: f32,
    @location(5) along: f32,
    @location(6) half_width: f32,
    @location(7) foam: f32,
    @location(8) sea: f32,
    @location(9) side: f32,
    @location(10) joined: vec3<f32>,
    @location(11) clarity: f32,
) -> WaterVertexOutput
{
    // A distant channel is narrower than the clipmap's triangles there,
    // which round its banks off above the water. Lift a distant surface by
    // about the bank those triangles leave (their spacing grows with
    // distance, a metre per 110 m), so a river still shows from a ridge.
    let flat_distance = length(position.xz - globals.camera_position.xz);
    let spacing = max(1.0, flat_distance/110.0);
    // A lake is wide enough for any triangles and stays at its level. So does
    // a river where it meets a lake or the sea (`still` is 1 there, fading to
    // 0 some 30 m along it), so the two surfaces still meet from afar.
    let lift = smoothstepf(150.0, 650.0, flat_distance)*min(spacing*0.45, 12.0)*(1.0 - clamp(still, 0.0, 1.0));
    let world = vec3<f32>(position.x, position.y + lift, position.z);
    var out: WaterVertexOutput;
    out.clip_position = globals.projection*globals.view*vec4<f32>(world, 1.0);
    out.world_position = world;
    out.wave_position = world.xz;
    out.velocity = velocity;
    out.channel = vec4<f32>(across, turbulence, position.y, still);
    out.stream = vec4<f32>(along, half_width, foam, sea);
    out.side = side;
    out.joined = joined;
    out.clarity = clarity;
    return out;
}

// ---------------------------------------------------------------------------
// The surface
// ---------------------------------------------------------------------------

// The unit normal of the triangle whose world position changes by `dx` and
// `dy` to the next pixel across and down, turned up out of the water: the
// sea's displaced wave face, a ribbon's tilted reach, a sheet's level, the
// same whichever way the triangle is wound. A triangle seen so nearly edge
// on that the two lie along one line keeps the level's.
fn facetUp(dx: vec3<f32>, dy: vec3<f32>) -> vec3<f32>
{
    let normal = cross(dx, dy);
    let area = dot(normal, normal);
    let up = normal*inverseSqrt(max(area, 1e-30))*select(1.0, -1.0, normal.y < 0.0);
    return select(vec3<f32>(0.0, 1.0, 0.0), up, area > 1e-6*dot(dx, dx)*dot(dy, dy));
}

// Which side of a surface the eye sees it from.
struct SurfaceSide
{
    // Seen from under the water.
    below: bool,
    // How far its upper side faces the eye, 0..1: foam on it fades out over
    // the last centimetres as the eye goes under it.
    upper: f32,
};

// The side of a surface the eye sees, with `eye_under` how far the eye is
// under its own water (`stage.eye_body.w`) and `eye_over_facet` the eye's
// height over the surface's own plane, along its upturned normal.
//
// It is the side of the water the eye is on, not how high the eye stands: a
// ray from the air meets every surface from the air, even one standing above
// the eye, such as a rapid climbing away up its channel, a river upstream of
// a swimmer or a swell's crest over a swimmer's head; a ray under the water
// meets every surface from below. Only while the eye crosses its own water's
// surface (`eye_under` between 0 and 1) can it see some surfaces from above
// and others from below. Then each is seen from the side of its own plane the
// eye is on, as a ray that reaches a surface through one medium must meet
// it: on a level sheet that is the eye's height over it, on the sea the
// wave's own face.
fn surfaceSide(eye_under: f32, eye_over_facet: f32) -> SurfaceSide
{
    let under = clamp(eye_under, 0.0, 1.0);
    let crossing = under > 0.0 && under < 1.0;
    return SurfaceSide(select(under >= 1.0, eye_over_facet < 0.0, crossing),
                       select(1.0 - under, smoothstepf(-0.08, 0.12, eye_over_facet), crossing));
}

// The deepest bed that shows through inland water. The bed's light crosses
// the column twice, down to it and back up to the eye, and even the clearest
// water, a tarn's in the blue, brings back only a thousandth of it from
// ln(1000)/(2 x 0.12/m) = 29 m down: a deeper bed looks the same as none.
const BED_DEPTH_MAX: f32 = 0.5*log(1000.0)
                         /min(min(ALPINE_EXTINCTION.x, ALPINE_EXTINCTION.y), ALPINE_EXTINCTION.z);

// How deep the bed lies under the surface, with `path` the straight line's
// length from the surface to the bed drawn behind it and `rise` how steeply
// the eye looks down along it (the sine of its elevation). The bed lies as
// deep whatever the angle it is seen at; only a bed deeper than any light
// comes back from is taken for one BED_DEPTH_MAX deep.
fn bedDepth(path: f32, rise: f32) -> f32
{
    return min(path*max(rise, 0.0), BED_DEPTH_MAX);
}

@fragment
fn fs_water(in: WaterVertexOutput) -> @location(0) vec4<f32>
{
    let uv = in.clip_position.xy*globals.viewport.zw;
    let camera_position = globals.camera_position.xyz;
    let to_camera = camera_position - in.world_position;
    let view_distance = length(to_camera);
    let to_view = to_camera/max(view_distance, 1e-3);
    let time = stage.params.x;
    // The frame the viewer sees: the frame time the eye has settled into,
    // which one slow frame barely moves (see EXPOSURE_SETTLE_SECONDS in
    // water_node.rs), so a hitch neither strips the ripples nor flashes the
    // glints.
    let frame_seconds = stage.wind.z;
    // Derivatives first, in uniform control flow.
    let pixel_dx = dpdx(in.wave_position);
    let pixel_dy = dpdy(in.wave_position);
    let pixel_footprint = max(length(pixel_dx), length(pixel_dy));
    // The plane of the triangle drawn here, for the side the eye sees it from.
    let facet = facetUp(dpdx(in.world_position), dpdy(in.world_position));
    let xz = in.world_position.xz;

    // --- What water this is -----------------------------------------------
    let sea = clamp(in.stream.w, 0.0, 1.0);
    let stillness = clamp(in.channel.w, 0.0, 1.0);
    let level = in.channel.z;
    let across = in.channel.x;
    let half_width = in.stream.y;
    let river = half_width > 0.01;
    let along = in.stream.x;
    // Down its last reach a river spreads and slows into the sea's: its
    // rapids and its foam give out as its water turns to the sea's.
    let turbulence = in.channel.y*(1.0 - sea);
    // Foam made by whitewater upstream and still drifting here.
    let supply = clamp(in.stream.z*3.0, 0.0, 1.0)*(1.0 - sea);
    // Seen from below, the surface is the same sheet of water from its other
    // side; which side the eye sees is the side of the water it is on
    // (`surfaceSide`), not how high it stands.
    let side = surfaceSide(stage.eye_body.w, dot(facet, to_camera));
    let underside = side.below;
    let above_water = side.upper;

    let precipitation = weatherPrecipitation(vec3<f32>(xz.x, level, xz.y));
    // View space looks down -Z, so a *greater* z is nearer. The water
    // surface's own view-space z, to compare against the G-buffer's.
    let water_view = (globals.view*vec4<f32>(in.world_position, 1.0)).xyz;
    let packed = reflectionDepth(uv);
    let scene_sky = packed.a < 0.5;
    var optical_path = PATH_LENGTH_MAX;
    if (!scene_sky)
    {
        optical_path = min(distance(packed.xyz, water_view), PATH_LENGTH_MAX);
    }
    let to_sun = normalize(-globals.sun_direction.xyz);
    let to_moon = normalize(-globals.moon_direction.xyz);

    // --- The current --------------------------------------------------------
    // A river's surface lives in its own frame, metres downstream and across,
    // so its ripples and standing waves follow every bend. Still water has no
    // frame and keeps world axes; a current running on into a lake only
    // carries its ripples.
    let velocity = in.velocity;
    let speed = length(velocity);
    let downstream = select(vec2<f32>(1.0, 0.0), velocity/max(speed, 1e-4), river && speed > 1e-4);
    let cross_stream = vec2<f32>(-downstream.y, downstream.x);
    let st = select(xz, vec2<f32>(along, in.side), river);
    // Where a tributary's water becomes its parent's, its ripples, steps and
    // foam cross-fade into those laid in the parent's own frame: two frames
    // at an angle cannot be blended into one without squeezing the ripples
    // between them. Two patterns mixed lose contrast, which `restore` puts
    // back.
    let joined_st = in.joined.xy;
    let joining = select(0.0, clamp(in.joined.z, 0.0, 1.0), river);
    let restore = inverseSqrt((1.0 - joining)*(1.0 - joining) + joining*joining);
    let flow = select(velocity, vec2<f32>(speed, 0.0), river);
    // The pixel's footprint along each axis of that frame: a grazing view
    // stretches it along the view, not across the flow.
    let footprint = vec2<f32>(max(abs(dot(pixel_dx, downstream)), abs(dot(pixel_dy, downstream))),
                              max(abs(dot(pixel_dx, cross_stream)), abs(dot(pixel_dy, cross_stream))));
    let roughness = clamp(turbulence*0.8 + smoothstepf(0.6, 3.0, speed)*0.25*(1.0 - sea), 0.0, 1.0);
    let riffle = select(0.0, smoothstepf(0.03, 0.2, turbulence)*smoothstepf(0.45, 1.2, speed), river);

    // --- The surface: swell, wind sea, current, rain ---------------------------
    var slope = vec2<f32>(0.0);
    var variance = 0.0;
    var crest = 0.0;
    var shore_foam = 0.0;
    // The sea's swell and surf, dying out up a river's mouth.
    if (sea > 0.0)
    {
        let shore = sampleShore(in.wave_position);
        let swell = sampleSurface(in.wave_position, 0.0, pixel_dx, pixel_dy, sea, shore);
        slope += -swell.normal.xz/max(swell.normal.y, 0.05);
        variance += swell.slope_variance;
        crest = clamp(1.0 - swell.jacobian, 0.0, 1.0);
        shore_foam = swell.shore_foam;
    }
    // The wind sea, as much as the wind and the fetch allow.
    let wind_direction = vec2<f32>(cos(stage.flags.z), sin(stage.flags.z));
    let here = riverCanopy(xz);
    let exposure = windExposure(xz, level, -wind_direction, here.a, sea, across, half_width, cross_stream);
    let gust = windGust(xz, wind_direction, stage.wind.x, time);
    let gust_factor = 1.0 + stage.wind.y*(gust - 0.5)*1.6;
    let local_wind = stage.wind.x*exposure.shelter*max(gust_factor, 0.0)*(1.0 + 0.31*precipitation.w);
    // On the sea the wind sea stops where the swell's spectrum begins.
    let longest = mix(3.7, 2.0, sea);
    var wind_sea = WindSea(vec2<f32>(0.0), 0.0, 0.0, 0.0);
    if (speed > 0.02)
    {
        // Running water carries its wind waves along with it.
        for (var layer = 0u; layer < 2u; layer += 1u)
        {
            let cycle = flowLayer(time, 3.0, layer, 0.29, velocity);
            let waves = windSea(xz, cycle.offset, local_wind, exposure.fetch, longest,
                                time, pixel_dx, pixel_dy, speed, frame_seconds);
            wind_sea.slope += waves.slope*cycle.weight;
            wind_sea.height += waves.height*cycle.weight;
            wind_sea.height_variance += waves.height_variance*cycle.weight*cycle.weight;
            wind_sea.variance += waves.variance*cycle.weight*cycle.weight;
        }
    }
    else
    {
        wind_sea = windSea(xz, vec2<f32>(0.0), local_wind, exposure.fetch, longest,
                           time, pixel_dx, pixel_dy, 0.0, frame_seconds);
    }
    slope += wind_sea.slope;
    variance += wind_sea.variance;
    // The current's own surface: turbulence, boils, standing waves.
    var water_surface = FlowSurface(vec2<f32>(0.0), 0.0, 0.0);
    if (river || speed > 0.02)
    {
        water_surface = flowSurface(st, flow, time, roughness, riffle, river, footprint, frame_seconds);
        if (joining > 0.001)
        {
            let other = flowSurface(joined_st, flow, time, roughness, riffle, river, footprint, frame_seconds);
            water_surface.slope = mix(water_surface.slope, other.slope, joining)*restore;
            water_surface.variance = mix(water_surface.variance, other.variance, joining);
            water_surface.crest = mix(water_surface.crest, other.crest, joining);
        }
        slope += downstream*water_surface.slope.x + cross_stream*water_surface.slope.y;
        variance += water_surface.variance;
    }
    let impacts = waterImpacts(xz, globals.storm.z, pixel_footprint,
                               view_distance, precipitation.x, precipitation.y, precipitation.w);
    if (!underside)
    {
        slope += impacts.slope;
    }
    var normal = normalize(vec3<f32>(-slope.x, 1.0, -slope.y));
    // Seen from above, a facet tipped away from the eye is met edge-on, not
    // from behind: bend it back just into view. From below, the surface is
    // the other side of the same sheet.
    let facing = dot(normal, to_view);
    if (!underside)
    {
        normal = normalize(normal + to_view*max(0.02 - facing, 0.0));
    }
    else if (facing < 0.0)
    {
        normal = -normal;
    }

    // --- Light ----------------------------------------------------------------
    // Trees shade the water as they shade its banks: by the sun's cascades
    // by day, the moon's by night.
    let plants = plantShadow(in.world_position);
    let sun_plants = select(1.0, plants, vegetation_shadows.light.w < 1.5);
    let moon_plants = select(1.0, plants, vegetation_shadows.light.w > 1.5);
    let severity = weatherSeverity(xz);
    // Sunlight over the banks and crowns, for what the water mirrors of them.
    let sun_open = globals.sun_colour.rgb*globals.settings_a.x*weatherSunMultiplier(severity)
                   *mix(1.0, 0.11, precipitation.z)*cloudShadow(in.world_position, to_sun);
    let sun_colour = sun_open*terrainShadow(in.world_position, normal, to_sun)*sun_plants;
    var moon_visibility = 1.0;
    if (globals.atmosphere.y > 0.001)
    {
        moon_visibility = terrainShadow(in.world_position, normal, to_moon)
                        *cloudShadow(in.world_position, to_moon);
    }
    let moon_colour = vec3<f32>(0.63, 0.74, 1.0)*globals.atmosphere.y
                      *weatherSunMultiplier(severity)*mix(1.0, 0.42, precipitation.z)
                      *moon_visibility*moon_plants;
    let sky_light = skyAmbient(vec3<f32>(0.0, 1.0, 0.0), in.world_position);

    // --- Roughness ------------------------------------------------------------
    // Everything the pixel could not resolve, from the swell to the capillary
    // ripples, widens the glints by the slope variance it really has. A
    // preset's fixed sun roughness overrides it on the sea.
    let base_roughness = mix(0.02, 0.10, roughness);
    var alpha = clamp(sqrt(base_roughness*base_roughness + variance + impacts.slope_variance), 0.02, 1.0);
    if (stage.surface.z > 0.0)
    {
        alpha = mix(alpha, clamp(stage.surface.z, 0.01, 1.0), sea);
    }
    // The normals the pixel spans, from the final normal's screen
    // derivatives, taken here at the top level where every pixel of the
    // quad runs.
    alpha = pixelRoughness(alpha, dpdx(normal), dpdy(normal));
    let n_dot_v = max(dot(normal, to_view), 0.0);
    let reflection_direction = reflect(-to_view, normal);
    let seen = surroundings(xz, level, reflection_direction, alpha, here, across, half_width,
                            cross_stream, sun_open, to_sun, in.world_position);

    // --- Refraction and the body of the water -------------------------------
    // Perturb the screen uv by the surface slope, scaled by how much water the
    // ray crosses and by the inverse distance, since a metre of slope covers
    // fewer pixels far away.
    let refraction_scale = mix(0.35, stage.misc.x, sea);
    let refracted_uv = clamp(uv + normal.xz*(refraction_scale*clamp(optical_path, 0.0, 1.0)
                             /max(view_distance, 1.0)), vec2<f32>(0.0), vec2<f32>(1.0));
    let flat_background = textureSampleLevel(scene_texture, scene_sampler, uv, 0.0).rgb;
    let refracted_background = textureSampleLevel(scene_texture, scene_sampler, refracted_uv, 0.0).rgb;
    let refracted_packed = reflectionDepth(refracted_uv);
    // The refracted tap is only trustworthy when it is sky or genuinely behind
    // the water; otherwise it reached over a nearer surface and would smear
    // foreground colour into the water.
    let accept_refraction = refracted_packed.a < 0.5 || refracted_packed.z <= water_view.z;
    let scene_linear = sceneRadiance(select(flat_background, refracted_background, accept_refraction));

    // Whitewater gathers where the bed steps: over the ledges and boulders
    // of a rapid, with dark glassy tongues of water running between them.
    // The steps stay put as the water runs over them. A ledge throws its
    // white water out in a train downstream of it, and the boulders between
    // break it up, so the steps are drawn out along the flow and broken by a
    // finer scale, not laid as round patches.
    let step_footprint = vec2<f32>(footprint.x, footprint.y/max(half_width, 0.01));
    let own_steps = riverSteps(along, across, step_footprint);
    let joined_steps = riverSteps(joined_st.x, across, step_footprint);
    let step_noise = mix(own_steps.x, joined_steps.x, joining);
    let step_shares = riverStepShares(0.5 + (step_noise - 0.5)*restore, own_steps.y);
    let steps = (step_shares.x + step_shares.y + step_shares.z)/3.0;
    // Rapids turn milky below each step, where the water plunges and fills
    // with bubbles, and run clear over the smooth tongues between.
    let aeration = select(0.0, smoothstepf(0.3, 0.9, turbulence)*mix(0.15, 0.7, steps), river);
    let body = WaterBody(sea, stillness, clamp(in.clarity, 0.0, 1.0)*(1.0 - sea), aeration);

    let cos_refracted = sqrt(1.0 - (1.0 - to_view.y*to_view.y)/(N_WATER*N_WATER));
    var water_body = vec3<f32>(0.0);
    // Inland water: the eye's ray bends down into the water, so it crosses
    // the column far more steeply than the straight line to the bed drawn
    // behind it: at a glance from the bank a shallow bed still shows through.
    // The bed lies as deep under the surface whatever the angle it is seen
    // at (`bedDepth`): a deep bed seen at a glance is no clearer than seen
    // from above.
    // The bed was lit as if in air: its light also crossed the water on the
    // way down, a path of its depth over the sun's height. And the light
    // scattered back out of the column is lit less the deeper it lies: each
    // metre of the eye's path is also a metre further down.
    if (sea < 1.0)
    {
        let medium = inlandMedium(body);
        let bed_depth = bedDepth(optical_path, to_view.y);
        let water_path = select(optical_path, bed_depth/max(cos_refracted, 0.05), !scene_sky && !underside);
        let transmittance = exp(-medium.sigma_t*water_path);
        let downwelling = exp(-medium.sigma_t*bed_depth/max(to_sun.y, 0.25));
        let sigma_column = medium.sigma_t*(1.0 + cos_refracted/max(to_sun.y, 0.25));
        let source = sun_colour*0.08 + moon_colour*0.08 + sky_light*0.35*seen.sky_view;
        let in_scatter = medium.sigma_s*source*(1.0 - exp(-sigma_column*water_path))
                       /max(sigma_column, vec3<f32>(1e-3));
        water_body = scene_linear*transmittance*downwelling + in_scatter;
    }
    // The sea, as its optics preset was calibrated: Beer-Lambert along the
    // straight path to what lies behind, in-scatter of the sunlight reaching
    // the top of the column through the water above the shaded point, and the
    // glow of light crossing the thinner sheet where crests are pinched.
    if (sea > 0.0)
    {
        let medium = seaMedium();
        let transmittance = exp(-medium.sigma_t*optical_path);
        let asymmetry = clamp(stage.scatter.w, -0.99, 0.99);
        let sun_phase = mix(henyeyGreenstein(dot(to_view, -to_sun), asymmetry),
                            phaseRayleigh(dot(to_view, -to_sun)), 0.5);
        let moon_phase = mix(henyeyGreenstein(dot(to_view, -to_moon), asymmetry),
                             phaseRayleigh(dot(to_view, -to_moon)), 0.5);
        let sun_attenuation = exp(-medium.sigma_t*max(stage.params.z - in.world_position.y, 0.0));
        let sea_source = sun_colour*sun_phase + moon_colour*moon_phase + sky_light*(0.35/(4.0*PI));
        var sea_body = scene_linear*transmittance
                     + medium.sigma_s*sun_attenuation*sea_source*(1.0 - transmittance)/medium.sigma_t;
        // `1 - jacobian` is 0 on undisturbed water, so the glow sits on the
        // compressed crests and nowhere else (aqua's `1 - 0.5*jacobian` read
        // half strength everywhere and washed the sea out to a pale mint).
        sea_body += pow(crest, 2.0)*stage.sss_tint.rgb*max(to_sun.y, 0.0)*sun_colour*0.35;
        water_body = mix(water_body, sea_body, sea);
    }

    // --- Reflection and glitter -----------------------------------------------
    // The sky's authored gain stands in for the shores a grazing ray would
    // find past the reach of the surroundings walk (see optics.rs): 0.75 on
    // the open sea, 0.8 on the narrower inland water, a little less under a
    // thunder cell, whose roughness and darkness deepen the reflection.
    let reflected_sky = cloudSkyRadiance(reflection_direction)
                      *mix(mix(0.8, 0.68, precipitation.z), mix(0.75, 0.66, precipitation.z), sea);
    // Where the screen cannot show the reflection, the banks and the crowns
    // on them hide the sky below their horizon.
    let unseen = mix(seen.occluder, reflected_sky, seen.sky);
    let terrain_reflection = marchWaterReflection(in.world_position, normal,
                                                  reflection_direction, alpha, level, underside);
    let reflected = mix(unseen, terrain_reflection.rgb, terrain_reflection.a);
    // Fresnel-weighted and normalised (waterSpecular): a low sun's glitter
    // path keeps its clipped core and loses only its halo.
    let specular = waterSpecular(normal, to_view, to_sun, alpha);
    let moon_specular = waterSpecular(normal, to_view, to_moon, alpha);

    // A lightning strike is a short-lived local source. Its cool reflection
    // follows the wave normal, while a narrow corridor approximates the image
    // of the bright vertical bolt in the water. Both fade with strike range;
    // the ordinary sky flash still comes from the shared atmosphere model.
    var flash_specular = vec3<f32>(0.0);
    var bolt_reflection = vec3<f32>(0.0);
    if (globals.lightning.w > 0.001 && above_water > 0.0) {
        let strike_base = globals.lightning.xyz;
        let strike_top = vec3<f32>(strike_base.x,
                                  max(globals.lightning_meta.z, strike_base.y + 20.0),
                                  strike_base.z);
        let source = (strike_base + strike_top)*0.5;
        let source_vector = source - in.world_position;
        let source_range = max(length(source_vector), 1.0);
        let flash_direction = source_vector/source_range;
        let range_fade = 1.0/(1.0 + pow(source_range/3000.0, 2.0));
        let flash_colour = vec3<f32>(0.73, 0.84, 1.0)
                           *globals.lightning.w*range_fade;
        let flash_n_dot_l = max(dot(normal, flash_direction), 0.0);
        let flash_half = normalize(flash_direction + to_view);
        let flash_alpha = max(alpha, 0.045);
        let flash_lobe = dGGX(max(dot(normal, flash_half), 0.0), flash_alpha)
                          *vSmithGGX(flash_n_dot_l, n_dot_v, flash_alpha)
                          *flash_n_dot_l*0.35;
        flash_specular = flash_colour*flash_lobe;

        if (globals.lightning_meta.x >= 0.0) {
            let horizontal_ray = reflection_direction.xz;
            let ray_span = max(dot(horizontal_ray, horizontal_ray), 0.005);
            let along_ray = max(dot(strike_base.xz - in.world_position.xz,
                                    horizontal_ray)/ray_span, 0.0);
            let reflected_point = in.world_position + reflection_direction*along_ray;
            let nearest_bolt = vec3<f32>(strike_base.x,
                                         clamp(reflected_point.y, strike_base.y, strike_top.y),
                                         strike_base.z);
            let miss = distance(reflected_point, nearest_bolt);
            let glint_width = 2.5 + along_ray*(0.0025 + 0.012*flash_alpha);
            let glint = exp(-pow(miss/max(glint_width, 0.1), 2.0))
                        *smoothstepf(0.0, 50.0, along_ray);
            // The channel itself is far brighter than the light it sheds.
            bolt_reflection = flash_colour*glint*6.5;
        }
    }

    var lit = vec3<f32>(0.0);
    if (!underside)
    {
        let fresnel = clamp(godotFresnel(n_dot_v, stage.surface.x, stage.surface.y), 0.0, 1.0);
        lit = mix(water_body, reflected, fresnel) + specular*sun_colour + moon_specular*moon_colour
            + flash_specular + bolt_reflection*fresnel;
    }
    else
    {
        // From below, light leaving the water bends away from the normal and
        // past the critical angle cannot leave at all: inside Snell's window
        // the sky and the banks pour through, refracted; outside it the
        // surface is a mirror of the water's own glow. The medium between
        // the eye and the surface is `fs_underwater`'s.
        let sin2_air = N_WATER*N_WATER*(1.0 - n_dot_v*n_dot_v);
        let medium = waterMedium(body);
        let deep = medium.sigma_s*(sun_colour*0.08 + sky_light*0.35)/max(medium.sigma_t*2.0, vec3<f32>(1e-3));
        if (sin2_air >= 1.0)
        {
            lit = deep;
        }
        else
        {
            let cos_air = sqrt(1.0 - sin2_air);
            let fresnel = godotFresnel(cos_air, stage.surface.x, stage.surface.y);
            let leaving = refract(-to_view, normal, N_WATER);
            let window = select(scene_linear, cloudSkyRadiance(leaving), scene_sky);
            lit = mix(window, deep, fresnel);
        }
    }

    // --- Foam -----------------------------------------------------------------
    // Foam is a diffuse layer over the reflecting surface: it replaces both
    // transmission and reflection, including at grazing view angles. Where it
    // comes from depends on the water:
    //
    // - On the sea, the breaking crests and the shore's wash.
    // - On any water a strong enough wind breaks the crests of its own waves:
    //   whitecaps cover 3.84e-6 U^3.41 of a wind sea (Monahan), from about a
    //   fresh breeze, less where the fetch has not let the sea develop.
    // - On a river, whitewater below the bed's steps, flecks on a riffle's
    //   standing waves, and the foam that drifts down from them.
    var foam = 0.0;
    let foam_uv = in.wave_position - wind_direction*stage.params.x*0.075;
    if (sea > 0.0)
    {
        let whitecap = smoothstepf(0.10, 0.30, crest);
        let coarse = mix(valueNoise(foam_uv*1.3), 0.5,
                         smoothstepf(0.25, 0.75, pixel_footprint*1.3));
        let bubbles = mix(valueNoise(foam_uv*5.7 + vec2<f32>(23.4, 7.1)), 0.5,
                          smoothstepf(0.25, 0.75, pixel_footprint*5.7));
        let lace = smoothstepf(0.22, 0.72, coarse*0.68 + bubbles*0.32);
        let caps = whitecap*0.70*foamBreakup(foam_uv, pixel_footprint);
        foam = (1.0 - (1.0 - caps)*(1.0 - shore_foam*mix(0.22, 1.15, lace)))*stage.misc.y*sea;
    }
    let coverage = 3.84e-6*pow(max(local_wind, 0.0), 3.41)
                 * smoothstepf(0.05, 0.5, windPeakWavenumber(local_wind, 1.0e6)/windPeakWavenumber(local_wind, exposure.fetch));
    if (coverage > 1e-5)
    {
        // A crest breaks once it stands as high as the share of the sea that
        // is white would put it.
        let sigma = sqrt(max(wind_sea.height_variance, 1e-8));
        let threshold = 0.85*sqrt(-2.0*log(clamp(coverage, 1e-5, 0.5)));
        let breaking = smoothstepf(threshold, threshold + 0.7, wind_sea.height/sigma)
                     * foamBreakup(foam_uv*2.3, pixel_footprint*2.3);
        let unresolved = smoothstepf(0.5, 2.0, pixel_footprint*4.0);
        foam = max(foam, mix(breaking, coverage, unresolved)*0.8);
    }
    if (river || speed > 0.02)
    {
        // Foam on a run is drawn out along the current; churned in a rapid it
        // breaks into short ragged patches.
        let foam_stretch = select(1.0, 1.0 + 2.5*smoothstepf(0.3, 1.5, speed)*(1.0 - 0.75*turbulence), river);
        var pattern = riverFoamPattern(st, flow, foam_stretch, time, footprint, frame_seconds);
        if (joining > 0.001)
        {
            let other = riverFoamPattern(joined_st, flow, foam_stretch, time, footprint, frame_seconds);
            pattern = 0.5 + (mix(pattern, other, joining) - 0.5)*restore;
        }
        // Whitewater where the bed breaks the surface, below the steps, in
        // each third of the pixel (riverStepShares).
        let whitewater = smoothstepf(0.35, 0.85, turbulence)*mix(vec3<f32>(0.15), vec3<f32>(0.8), step_shares);
        // Flecks on the faces of a riffle's standing waves.
        let flecks = water_surface.crest*riffle*0.5;
        // Foam the whitewater upstream made drifts down as a thin lace,
        // gathered into lines where the surface currents converge: the seams
        // between the fast core and the slack water by the banks, a tongue
        // down the current that wanders across the channel, and a scum in the
        // slack edges of pools.
        let wander = (mix(valueNoise(vec2<f32>(along/RIVER_TONGUE_WANDER, 5.3)),
                          valueNoise(vec2<f32>(joined_st.x/RIVER_TONGUE_WANDER, 5.3)), joining) - 0.5)*0.8;
        // The tongue is a line a sixth of a half width wide; where the pixel
        // is wider the line is spread over it, dimmer by as much, so it
        // neither breaks into dashes nor brightens far off.
        let tongue_width = sqrt(1.0/36.0 + 2.0*PIXEL_FILTER_VARIANCE*step_footprint.y*step_footprint.y);
        let off_tongue = (across - wander)/tongue_width;
        let tongue = exp(-off_tongue*off_tongue)*(1.0/6.0)/tongue_width;
        let seam = smoothstepf(0.5, 0.75, abs(across))*(1.0 - smoothstepf(0.82, 0.97, abs(across)));
        let slack = smoothstepf(0.8, 0.98, abs(across))*(1.0 - smoothstepf(0.25, 0.7, speed));
        let drift = select(0.0, supply*(0.05 + 0.25*max(tongue, seam) + 0.2*slack), river);
        // A creek's jet carries its foam out into the pond it feeds.
        let mouth = select(smoothstepf(0.08, 0.5, speed)*0.3, 0.0, river)*(1.0 - sea);
        // Far off the texture averages away; keep the share of water it covers.
        let foam_resolved = 1.0 - smoothstepf(0.15, 0.5, max(footprint.x/foam_stretch, footprint.y)*2.1);
        let carried = max(flecks, max(drift, mouth));
        let stream_foam = (streamFoam(max(whitewater.x, carried), pattern, foam_resolved)
                         + streamFoam(max(whitewater.y, carried), pattern, foam_resolved)
                         + streamFoam(max(whitewater.z, carried), pattern, foam_resolved))/3.0;
        foam = max(foam, stream_foam);
    }
    foam = clamp(foam + impacts.splash*0.2 + impacts.snow_fleck*0.06, 0.0, 1.0);
    // Bubbles sit on the upper surface. Avoid a bright foam sheet underwater.
    foam *= above_water;
    // Sea foam is white; stream foam is cream with the same tannins that
    // stain the water, and under the trees it sees only the sky the crowns
    // leave. It is a bright diffuse surface, so its radiance sits a little
    // above the sky it is lit by.
    let stained = (1.0 - sea)*(1.0 - body.alpine);
    let foam_albedo = mix(vec3<f32>(0.92, 0.95, 0.96), vec3<f32>(0.86, 0.84, 0.76), stained);
    let foam_colour = foam_albedo
                    * ((sun_colour*max(dot(normal, to_sun), 0.0)
                        + moon_colour*max(dot(normal, to_moon), 0.0))*mix(0.05, 0.06, stained)
                       + skyAmbient(normal, in.world_position)*mix(0.75, 0.8*seen.sky_view, stained));
    lit = mix(lit, foam_colour, foam);
    // Resolve the last few centimetres into the actual bank instead of a hard
    // cutout. The measured ray length makes this agree with visible geometry.
    lit = mix(scene_linear, lit, smoothstepf(0.0, mix(0.05, 0.12, sea), optical_path));

    // Raymarch aerial scattering only up to this water surface. Height fog,
    // sunset shafts and the volumetric switch agree with the terrain pass.
    let fog = integrateAtmosphere(-to_view, view_distance);
    lit = lit*fog.transmittance + fog.scattering;
    return vec4<f32>(pow(acesFilm(lit*globals.params.z), vec3<f32>(1.0/2.2)), 1.0);
}

// ---------------------------------------------------------------------------
// Fullscreen passes: the copy under the surfaces, and the submerged medium
// ---------------------------------------------------------------------------

struct FullscreenOutput
{
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) vertex: u32) -> FullscreenOutput
{
    var corner = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    var output: FullscreenOutput;
    output.position = vec4<f32>(corner[vertex], 0.0, 1.0);
    return output;
}

// Copies the composited frame into the water pass's output target. Not from
// bevy-aqua: `ViewTarget::post_process_write()` ping-pongs between two
// textures, and the water surfaces only cover the pixels they draw, so the
// pass writes the composite's output for every pixel first and the surfaces
// over it. Same top-down uv convention as the composite and FXAA: `position`
// is in the target's pixel space, so `position.xy * viewport.zw` is a plain
// same-frame blit with no flip.
@fragment
fn fs_blit(in: FullscreenOutput) -> @location(0) vec4<f32>
{
    return textureSample(scene_texture, scene_sampler, in.position.xy*globals.viewport.zw);
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
/// row 69. `to_view.y` decides which rays reach the surface overhead (and
/// drove the Snell window this pass used to draw, which therefore sat where
/// the camera was not pointing: at yaw 0, 90 and 270 the contraction never
/// reached the window's 0.5412 threshold at all, so the window was simply
/// absent, and it appeared only at yaw 180 where the matrix is symmetric).
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

// Trace cloud occlusion from where light enters the water. The homogeneous
// medium uses its optical midpoint as a representative scattering position.
fn underwaterCloudShadow(position: vec3<f32>, to_light: vec3<f32>) -> f32 {
    let depth = max(stage.eye_water.x - position.y, 0.0);
    let water_light = normalize(vec3<f32>(to_light.x/N_WATER,
        sqrt(max(1.0 - (1.0 - to_light.y*to_light.y)/(N_WATER*N_WATER), 0.001)),
        to_light.z/N_WATER));
    let entry = position + water_light*(depth/max(water_light.y, 0.1));
    return cloudShadow(entry, to_light);
}

// The medium around a submerged eye, whichever water it is in: the sea's, a
// river's or a lake's (`stage.eye_water`, `stage.eye_body`), with that
// water's own optics. The incoming composite has been exposed, tone-mapped
// and gamma encoded; the transform is inverted before attenuation and
// applied once to the final HDR result. Sunlight, moonlight and the sky
// through Snell's window follow the day clock.
@fragment
fn fs_underwater(in: FullscreenOutput) -> @location(0) vec4<f32>
{
    let uv = in.position.xy*globals.viewport.zw;
    let fade = clamp(stage.eye_water.z, 0.0, 1.0);

    // Every sample up front: this shader branches on sampled data below and
    // WGSL forbids implicit-derivative sampling in non-uniform control flow.
    let scene = textureSample(scene_texture, scene_sampler, uv);
    let packed = textureSample(gbuffer_position, gbuffer_sampler, uv);
    if (fade <= 0.0)
    {
        return scene;
    }
    let level = stage.eye_water.x;

    // Optical path: the world distance from the eye to whatever the ray hits,
    // capped at the medium's maximum and, for upward rays, at the surface.
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
        let surface_path = max(level - globals.camera_position.y, 0.0)/to_view.y;
        optical_path = min(optical_path, surface_path);
    }
    let scene_linear = sceneRadiance(scene.rgb);
    let body = WaterBody(stage.eye_water.w, stage.eye_body.x, clamp(stage.eye_body.z, 0.0, 1.0)*(1.0 - stage.eye_water.w),
                         smoothstepf(0.3, 0.9, stage.eye_body.y)*0.4);
    let medium = waterMedium(body);
    let transmittance = exp(-medium.sigma_t*optical_path);

    // In-scattered sunlight. The eye is inside the medium, so the whole path is
    // the water column — there is nothing above the shaded point to attenuate
    // first, unlike the surface pass.
    let to_sun = normalize(-globals.sun_direction.xyz);
    // rayDirection points from the eye toward the scene, so forward
    // scattering peaks while looking toward the source.
    let asymmetry = clamp(mix(0.8, stage.scatter.w, body.sea), -0.99, 0.99);
    let cos_theta = dot(to_view, to_sun);
    let phase = mix(henyeyGreenstein(cos_theta, asymmetry), phaseRayleigh(cos_theta), 0.5);
    let to_moon = normalize(-globals.moon_direction.xyz);
    let scatter_position = globals.camera_position.xyz + to_view*optical_path*0.5;
    var sun_visibility = 1.0;
    if (globals.settings_a.x > 0.001) {
        sun_visibility = underwaterCloudShadow(scatter_position, to_sun);
    }
    var moon_visibility = 1.0;
    if (globals.atmosphere.y > 0.001) {
        moon_visibility = underwaterCloudShadow(scatter_position, to_moon);
    }
    let moon_phase = mix(henyeyGreenstein(dot(to_view, to_moon), asymmetry),
                         phaseRayleigh(dot(to_view, to_moon)), 0.5);
    let local_weather = weatherSeverity(scatter_position.xz);
    let convective_core = weatherConvectiveCore(scatter_position);
    let moon_radiance = vec3<f32>(0.63, 0.74, 1.0)*stage.medium_sun.w
                        *weatherSunMultiplier(local_weather)
                        *mix(1.0, 0.42, convective_core)*moon_visibility;
    let ambient = skyAmbient(vec3<f32>(0.0, 1.0, 0.0), scatter_position)
                  *(0.35/(4.0*PI));
    // Light reaching the scattering point has crossed the water above it.
    let overhead = exp(-medium.sigma_t*max(level - scatter_position.y, 0.0));
    let scattered = medium.sigma_s*overhead*(stage.medium_sun.rgb*weatherSunMultiplier(local_weather)
                             *mix(1.0, 0.11, convective_core)*phase*sun_visibility
                             + moon_radiance*moon_phase + ambient)
                   *(1.0 - transmittance)/medium.sigma_t;

    // Snell's window is the surface pass's: wherever a water surface lies
    // over the eye it draws the sky refracted through it, and the medium here
    // attenuates it on its way down to the eye like everything else.
    let lit = scene_linear*transmittance + scattered;
    let medium_result = pow(acesFilm(lit*globals.params.z), vec3<f32>(1.0/2.2));
    return vec4<f32>(mix(scene.rgb, clamp(medium_result, vec3<f32>(0.0), vec3<f32>(1.0)), fade),
                     1.0);
}
