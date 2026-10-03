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
//   displaces vertices with a continuous mesh limit, while fragments evaluate
//   normals and compression using the projected footprint of each wave.
//   Lighting therefore retains waves beyond the geometry's 744 m fade, even
//   on the flat horizon skirt. Unresolved waves become specular roughness.
//   Neither limit depends on the tile's LOD; `ranges` is uploaded for parity
//   but not read.
// - A local metre-resolution seabed map samples the same procedural terrain
//   and streamed erosion as the ground. Waves damp in shallow water and a
//   depth-limited, shoreward travelling wave train supplies small breakers and
//   receding foam. This is an analytic surf approximation, not a fluid solver.
//   The G-buffer still supplies the measured optical path for transmission;
//   it does not determine the surf band's width or direction.
// - Visible terrain reflections are raymarched against the view-position
//   buffer. Offscreen and disoccluded rays fade to the shared sky plus a
//   raymarched cloud probe. Scene taps are gamma-decoded and inverse-ACES transformed
//   before mixing with linear radiance; clipped highlights cannot be recovered.
//
// WGSL CONSTRAINT: `textureSample` uses implicit derivatives and is illegal in
// non-uniform control flow. Refraction samples and wave derivatives are taken
// before discard; raymarching uses textureLoad or textureSampleLevel explicitly.

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
    shore_map: vec4<f32>,     // world centre XZ, span, texel size
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
// The sky probe is raymarched once per frame. Waves sample it in their
// reflected direction, so the complete cloud sky remains visible offscreen.
@group(3) @binding(2) var cloud_sky_probe: texture_2d<f32>;
@group(3) @binding(3) var cloud_sky_sampler: sampler;

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

// Return a bounded interval even when the eye lies inside or above the layer.
// A horizontal ray is valid only if its origin already lies in the volume.
fn cloudInterval(origin: vec3<f32>, ray: vec3<f32>, maximum_distance: f32) -> vec2<f32> {
    // Local bases and tops vary along the ray. Enclose all four weather
    // profiles so an off-camera front cannot be clipped by the camera layer.
    let overrides = u32(globals.weather.w);
    let bottom = max(globals.clouds.w - select(450.0, 0.0, (overrides & 4u) != 0u), 100.0);
    let top = globals.clouds.w + select(150.0, 0.0, (overrides & 4u) != 0u)
              + max(globals.cloud_layer.x, 50.0)*select(1.45, 1.0, (overrides & 8u) != 0u);
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
fn cloudShadow(world_position: vec3<f32>, to_light: vec3<f32>) -> f32 {
    if (globals.clouds.x < 0.5 || globals.clouds.z <= 0.001
        || globals.cloud_layer.z <= 0.001 || to_light.y <= 0.0) { return 1.0; }
    let interval = cloudInterval(world_position, to_light, 65000.0);
    if (interval.y <= interval.x) { return 1.0; }
    let count = 4u + u32(clamp(globals.cloud_layer.w, 0.0, 2.0));
    let step_length = (interval.y - interval.x)/f32(count);
    var optical_depth = 0.0;
    for (var i = 0u; i < 6u; i += 1u) {
        if (i >= count) { break; }
        let distance = interval.x + (f32(i) + 0.5)*step_length;
        optical_depth += cloudDensityFiltered(world_position + to_light*distance,
                                              false, step_length)*step_length*CLOUD_EXTINCTION;
    }
    let strength = globals.cloud_layer.z*cloudShadowMultiplier(
        cloudFrontSeverity(world_position.xz));
    return mix(1.0, exp(-optical_depth), clamp(strength, 0.0, 1.0));
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
    var visibility = 1.0;
    for (var i = 0u; i < 44u; i += 1u) {
        if (i >= count) { break; }
        let f = (f32(i) + 1.0)/f32(count);
        let distance = texel*1.2 + f*f*4800.0;
        let sample_position = origin + to_light*distance;
        let edge_weight = terrainMapWeight(sample_position.xz);
        if (edge_weight <= 0.0) { break; }
        let clearance = sample_position.y - terrainHeightAt(sample_position.xz);
        // A finite solar disc makes the penumbra widen with occluder range.
        let penumbra = max(1.2, distance*0.0047);
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
    for (var i = 0u; i < 10u; i += 1u) {
        if (i >= count) { break; }
        let f = (f32(i) + 0.4)/f32(count);
        let sample_position = world_position + to_light*(16.0 + f*f*3600.0);
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
            sunlight_visibility = volumeLightVisibility(world_position, to_sun)
                                *cloudShadow(world_position, to_sun);
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
                         direction: vec3<f32>, roughness: f32, surface_level: f32) -> vec4<f32>
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
            // seen through the water's reflective side.
            let world_hit = globals.camera_position.xyz
                          + vec3<f32>(dot(globals.view[0].xyz, hit.xyz),
                                      dot(globals.view[1].xyz, hit.xyz),
                                      dot(globals.view[2].xyz, hit.xyz));
            let above_surface = globals.camera_position.y < surface_level
                             || world_hit.y >= surface_level - 0.25;
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
// Vertex stage
// ---------------------------------------------------------------------------

struct VertexOutput
{
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) wave_position: vec2<f32>,
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

    var out: VertexOutput;
    out.clip_position = globals.projection*globals.view*vec4<f32>(world_position, 1.0);
    out.world_position = world_position;
    // Preserve the Gerstner parameter position, before horizontal chop, so
    // fragment crests and their displaced geometry have the same phase.
    out.wave_position = world_xz;
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
    // Derivatives must be evaluated before any data-dependent branch/discard.
    let pixel_dx = dpdx(in.wave_position);
    let pixel_dy = dpdy(in.wave_position);
    let pixel_footprint = max(length(pixel_dx), length(pixel_dy));
    let shore = sampleShore(in.wave_position);
    let surface = sampleSurface(in.wave_position, 0.0, pixel_dx, pixel_dy, 1.0, shore);
    let precipitation = weatherPrecipitation(vec3<f32>(in.world_position.x,
                                                       stage.params.z,
                                                       in.world_position.z));
    let impacts = waterImpacts(in.world_position.xz, globals.storm.z, pixel_footprint,
                               view_distance, precipitation.x, precipitation.y,
                               precipitation.w);

    // View space looks down -Z, so a *greater* z is nearer. The water
    // surface's own view-space z, to compare against the G-buffer's.
    let water_view = (globals.view*vec4<f32>(in.world_position, 1.0)).xyz;

    // Every texture sample is taken in uniform control flow, before discard.
    // Two-sided: seen from below, the surface is the same sheet of water lit
    // from the other side, so the geometric normal is flipped toward the eye
    // rather than the pass culling backfaces.
    var normal = surface.normal;
    if (camera_position.y >= in.world_position.y) {
        normal = normalize(normal + vec3<f32>(-impacts.slope.x, 0.0, -impacts.slope.y));
    }
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

    // Hidden surfaces never get here: the pass depth-tests against the
    // G-buffer's depth in hardware, ahead of this shader, so there is no
    // discard to defeat the early test.

    // The refracted tap is only trustworthy when it is sky or genuinely behind
    // the water; otherwise it reached over a nearer surface and would smear
    // foreground colour into the water.
    let accept_refraction = refracted_packed.a < 0.5
                         || refracted_packed.z <= water_view.z;
    let background = select(flat_background.rgb, refracted_background.rgb, accept_refraction);
    let scene_linear = sceneRadiance(background);

    // --- Body: Beer-Lambert extinction plus in-scatter -----------------------
    let sea_level = stage.params.z;
    let sigma_t = max(stage.extinction.xyz, vec3<f32>(1e-4));
    let sigma_s = min(sigma_t,
                      PARTICLE_SCATTER*stage.extinction.w*stage.scatter.xyz + RAYLEIGH);
    let transmittance = exp(-sigma_t*optical_path);

    let to_sun = normalize(-globals.sun_direction.xyz);
    let sun_visibility = terrainShadow(in.world_position, normal, to_sun)
                       *cloudShadow(in.world_position, to_sun);
    let sun_colour = globals.sun_colour.rgb*globals.settings_a.x
                     *weatherSunMultiplier(weatherSeverity(in.world_position.xz))
                     *mix(1.0, 0.11, precipitation.z)*sun_visibility;
    let to_moon = normalize(-globals.moon_direction.xyz);
    var moon_visibility = 1.0;
    if (globals.atmosphere.y > 0.001)
    {
        moon_visibility = terrainShadow(in.world_position, normal, to_moon)
                         *cloudShadow(in.world_position, to_moon);
    }
    let moon_colour = vec3<f32>(0.63, 0.74, 1.0)*globals.atmosphere.y
                      *weatherSunMultiplier(weatherSeverity(in.world_position.xz))
                      *mix(1.0, 0.42, precipitation.z)
                      *moon_visibility;
    let cos_theta = dot(to_view, -to_sun);
    let phase = mix(henyeyGreenstein(cos_theta, clamp(stage.scatter.w, -0.99, 0.99)),
                    phaseRayleigh(cos_theta), 0.5);

    // Sunlight reaching the top of the column, attenuated over the water above
    // the shaded point. Aqua's closed-form in-scatter integral reduces to this
    // for a single direction and a homogeneous medium.
    let sun_attenuation = exp(-sigma_t*max(sea_level - in.world_position.y, 0.0));
    let moon_phase = mix(henyeyGreenstein(dot(to_view, -to_moon),
                         clamp(stage.scatter.w, -0.99, 0.99)),
                         phaseRayleigh(dot(to_view, -to_moon)), 0.5);
    let ambient_source = skyAmbient(vec3<f32>(0.0, 1.0, 0.0), in.world_position)
                         *(0.35/(4.0*PI));
    let scattered = sigma_s*sun_attenuation*(sun_colour*phase + moon_colour*moon_phase
                    + ambient_source)*(1.0 - transmittance)/sigma_t;
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
    let crest = clamp(1.0 - surface.jacobian, 0.0, 1.0);
    let sss = pow(crest, 2.0)*stage.sss_tint.rgb*max(to_sun.y, 0.0);
    body += sss*sun_colour*0.35;

    // --- Whitecaps and shoreline wash ----------------------------------------
    // Only an energetic, breaking wave generates foam. The local seabed
    // confines the shore term to a narrow band; a calm shallow flat stays clear.
    let whitecap = smoothstepf(0.10, 0.30, crest);
    let drift = vec2<f32>(cos(stage.flags.z), sin(stage.flags.z))*stage.params.x*0.075;
    let foam_uv = in.wave_position - drift;
    let coarse = mix(valueNoise(foam_uv*1.3), 0.5,
                     smoothstepf(0.25, 0.75, pixel_footprint*1.3));
    let bubbles = mix(valueNoise(foam_uv*5.7 + vec2<f32>(23.4, 7.1)), 0.5,
                      smoothstepf(0.25, 0.75, pixel_footprint*5.7));
    let lace = smoothstepf(0.22, 0.72, coarse*0.68 + bubbles*0.32);
    let shore_foam = surface.shore_foam*mix(0.22, 1.15, lace);
    let caps = whitecap*0.70*foamBreakup(foam_uv, pixel_footprint);
    // Bubbles sit on the upper surface. Avoid a bright foam sheet underwater.
    let above_water = smoothstepf(-0.08, 0.12, camera_position.y - in.world_position.y);
    let foam = clamp((1.0 - (1.0 - caps)*(1.0 - shore_foam))*stage.misc.y
                     + impacts.splash*0.22 + impacts.snow_fleck*0.06, 0.0, 1.0)
             * above_water;
    // Foam is a bright diffuse surface, so its radiance sits a little above the
    // sky it is lit by — not the two-and-a-half times that `sun_colour*0.25`
    // produced, which clipped the shore break to flat white.
    let foam_colour = vec3<f32>(0.92, 0.95, 0.96)
                    * ((sun_colour*max(dot(normal, to_sun), 0.0)
                        + moon_colour*max(dot(normal, to_moon), 0.0))*0.05
                       + skyAmbient(normal, in.world_position)*0.75);

    // --- Reflection ----------------------------------------------------------
    //
    // The 0.75 gain is authored, and it is the one number in this block that is
    // not trying to be physical. Fresnel pins the far sea at roughly full
    // reflectance — a grazing ray reflects 0.92 of its radiance into the eye,
    // measured on this scene at a 10 m eye height — so the far water can only
    // ever be as dark as the sky it mirrors. That is correct optics and it is
    // also why the horizon reads as a continuation of the sky here: this sea is
    // kilometres wide, so a grazing ray finds sky, where the reference lake
    // found a dark forested shore. The optics doc already names that difference
    // as the reason the reference's water looks darker than this renderer's.
    //
    // Dimming the mirrored copy stands in for it. It is deliberately applied
    // here, at the call site, and not to `cloudSkyRadiance` itself: the sky the
    // camera sees directly must stay exactly as the composite pass draws it, or
    // the waterline gains a step in the wrong direction. Note the asymmetry
    // that follows — the raymarched terrain reflection below is mixed in
    // unscaled, so a dark headland mirrored in the water now sits against a
    // dimmed sky. Over open water the march returns no hit and the two never
    // meet; near shore it is the one visible cost of this constant.
    //
    // 0.85 was the previous value. The step to 0.75 is a 12% cut in the
    // reflection, which moves the far band about 10 luma against an unchanged
    // sky; the Fresnel exponent handles 60-85 degrees incidence and this
    // handles the 85-90 degree band where the curve is pinned flat. Under a
    // thunder cell, rain roughness and a small local reduction deepen the
    // reflection further while the shared atmosphere darkens the cloud sky.
    let reflection_direction = reflect(-to_view, normal);
    let reflected_sky = cloudSkyRadiance(reflection_direction)
                        *mix(0.75, 0.66, precipitation.z);

    // Sun specular: GGX over the wave roughness, which is what produces a
    // glitter path rather than one broad highlight.
    let n_dot_l = max(dot(normal, to_sun), 0.0);
    let n_dot_v = max(dot(normal, to_view), 0.0);
    let half_vector = (to_sun + to_view)/max(length(to_sun + to_view), 1e-4);
    let n_dot_h = max(dot(normal, half_vector), 0.0);
    var roughness = clamp(mix(0.18, 0.045, clamp(n_dot_l, 0.0, 1.0)), 0.02, 1.0);
    if (stage.surface.z > 0.0)
    {
        roughness = clamp(stage.surface.z, 0.01, 1.0);
    }
    // Filtered-out slopes broaden the glitter instead of leaving a perfectly
    // smooth mirror or aliasing into isolated bright pixels at the horizon.
    let alpha = clamp(sqrt(pow(roughness, 4.0) + surface.slope_variance
                           + impacts.slope_variance), 1e-3, 1.0);
    let specular = dGGX(n_dot_h, alpha)*vSmithGGX(n_dot_l, n_dot_v, alpha)*n_dot_l*0.35;
    let moon_half = (to_moon + to_view)/max(length(to_moon + to_view), 1e-4);
    let moon_dot_l = max(dot(normal, to_moon), 0.0);
    let moon_specular = dGGX(max(dot(normal, moon_half), 0.0), alpha)
                      * vSmithGGX(moon_dot_l, n_dot_v, alpha)*moon_dot_l*0.35;
    let terrain_reflection = marchWaterReflection(in.world_position, normal,
                                                   reflection_direction, alpha, stage.params.z);
    let reflected = mix(reflected_sky, terrain_reflection.rgb, terrain_reflection.a);

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

    // --- Fresnel composition -------------------------------------------------
    let fresnel = clamp(godotFresnel(clamp(dot(normal, to_view), 0.0, 1.0),
                                     stage.surface.x, stage.surface.y), 0.0, 1.0);
    var lit = mix(body, reflected, fresnel) + specular*sun_colour + moon_specular*moon_colour
              + flash_specular + bolt_reflection*fresnel;

    // Foam is a diffuse layer above the reflecting surface: it replaces both
    // transmission and reflection, including at grazing view angles.
    lit = mix(lit, foam_colour, foam);
    // Resolve the last few centimetres into the actual bank instead of a hard
    // cutout. The measured ray length makes this agree with visible geometry.
    let contact_coverage = smoothstepf(0.0, 0.12, optical_path);
    lit = mix(scene_linear, lit, contact_coverage);

    // Raymarch aerial scattering only up to this water surface. Height fog,
    // sunset shafts and the volumetric switch now agree with the terrain pass.
    let fog = integrateAtmosphere(-to_view, view_distance);
    lit = lit*fog.transmittance + fog.scattering;

    return vec4<f32>(pow(acesFilm(lit*globals.params.z), vec3<f32>(1.0/2.2)), 1.0);
}

// ===========================================================================
// Rivers and creeks
// ===========================================================================
//
// The same water as the sea, flowing. Each river's surface is a ribbon at
// its water level (src/rivers/surface.rs) whose vertices carry the current;
// waterfalls are sheets falling along the jet's parabola. The surface is
// shaded with the sea's optics (Fresnel, sky and screen-space reflections,
// Beer-Lambert body and refraction, sun glitter, rain rings, fog) and three
// things the sea does not have:
//
// - Flow-mapped ripples. Two phases of procedural ripple noise are carried
//   downstream by the local current and cross-faded, so the texture streams
//   with the water at its own speed, stretched along the flow where it runs
//   fast and choppy where the bed is rough.
// - Boulders. Every rock near a point is looked up through the river grid.
//   Water parts around one that breaks the surface as potential flow around
//   a cylinder, heaps into a pillow against its upstream face (the
//   stagnation head, v^2/2g), and leaves a slow, churning, foamy wake. A
//   rock just under the surface lifts a standing hump, and where the water
//   pours over it fast enough it breaks into a foaming hole downstream.
// - Whitewater. Cascades, plunge pools below falls, eddy lines along fast
//   banks and the wakes above all carry advected foam; the falling sheets
//   are aerated water, streaked and torn at their edges.

// A boulder as the water sees it. Mirrors `RockObstacle` in
// src/rivers/carve.rs (16 bytes).
struct RiverRock
{
    position: vec2<f32>,
    radius: f32,
    top: f32,
};

// The river lookup grid (see river-functions.wgslinc for its layout): each
// cell lists, after its carve segments, the rocks whose wakes reach it.
@group(2) @binding(1) var<storage, read> river_rocks: array<RiverRock>;
@group(2) @binding(2) var<storage, read> river_grid: array<u32>;

// The plants' shadow cascades, as the composite reads them, so a creek under
// the forest lies in the same shade as its banks. Mirrors
// `VegetationShadows` in composite.wgsl.
struct VegetationShadows {
    view_projection: array<mat4x4<f32>, 4>,
    splits: vec4<f32>,
    texel: vec4<f32>,
    light: vec4<f32>,
    params: vec4<f32>,
};
@group(2) @binding(3) var vegetation_shadow_map: texture_depth_2d_array;
@group(2) @binding(4) var<uniform> vegetation_shadows: VegetationShadows;
@group(2) @binding(5) var vegetation_shadow_sampler: sampler_comparison;

// Light reaching a water point through the plants: the cascade covering its
// view depth, with a 2x2 grid of bilinear comparisons (the composite's
// 4x4 tent is more than a rippling surface needs).
fn riverPlantShadow(world_position: vec3<f32>, view_depth: f32) -> f32 {
    if (vegetation_shadows.light.w < 0.5 || view_depth >= vegetation_shadows.splits.w) {
        return 1.0;
    }
    var cascade = 3u;
    for (var c = 0u; c < 3u; c++) {
        if (view_depth < vegetation_shadows.splits[c]) {
            cascade = c;
            break;
        }
    }
    let texel = vegetation_shadows.texel[cascade];
    let lookup = world_position + vec3<f32>(0.0, texel*1.5, 0.0) - vegetation_shadows.light.xyz*texel*1.5;
    let clip = vegetation_shadows.view_projection[cascade]*vec4<f32>(lookup, 1.0);
    let uv = vec2<f32>(0.5 + 0.5*clip.x, 0.5 - 0.5*clip.y);
    if (any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0)) || clip.z > 1.0) {
        return 1.0;
    }
    let step = 1.0/vegetation_shadows.params.z;
    var lit = 0.0;
    for (var y = 0; y < 2; y++) {
        for (var x = 0; x < 2; x++) {
            let offset = vec2<f32>(f32(x) - 0.5, f32(y) - 0.5)*step*1.5;
            lit += textureSampleCompareLevel(vegetation_shadow_map, vegetation_shadow_sampler,
                                             uv + offset, i32(cascade), clip.z);
        }
    }
    let far_fade = smoothstepf(vegetation_shadows.splits.w*0.85, vegetation_shadows.splits.w, view_depth);
    return mix(1.0, mix(lit*0.25, 1.0, far_fade), vegetation_shadows.params.y);
}

const RIVER_GRAVITY: f32 = 9.81;
// Clear stream water, faintly stained by the tannins of the forest floor:
// red is absorbed within a couple of metres and the dissolved organic matter
// takes some blue, so the bed shows through a shallow riffle, a pool goes
// deep green-brown and a lake dark. Fresh water scatters little (it carries
// none of the sea's plankton), so its body is dark and the bed and the
// reflections give it its colour.
const RIVER_EXTINCTION: vec3<f32> = vec3<f32>(0.40, 0.14, 0.21);
const RIVER_SCATTER: vec3<f32> = vec3<f32>(0.006, 0.010, 0.006);
// Rocks a fragment considers at most.
const RIVER_MAX_ROCKS: u32 = 24u;

struct RiverVertexOutput
{
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) velocity: vec2<f32>,
    // x across the channel (-1..1 between the waterlines), y metres along
    // the river, z thalweg depth, w whitewater.
    @location(2) channel: vec4<f32>,
    // x half width, y fall progress (-1 on a flat surface), z fall height
    // (a plunge's strength on a surface), w the surface's own height.
    @location(3) sheet: vec4<f32>,
};

@vertex
fn vs_river(
    @location(0) position: vec3<f32>,
    @location(1) velocity: vec2<f32>,
    @location(2) across: f32,
    @location(3) along: f32,
    @location(4) depth: f32,
    @location(5) turbulence: f32,
    @location(6) half_width: f32,
    @location(7) fall: f32,
    @location(8) drop: f32,
) -> RiverVertexOutput
{
    // A distant channel is narrower than the clipmap's triangles there,
    // which round its banks off above the water. Lift a distant surface by
    // about the bank those triangles leave (their spacing grows with
    // distance, a metre per 110 m), so a river still shows from a ridge.
    let flat_distance = length(position.xz - globals.camera_position.xz);
    let spacing = max(1.0, flat_distance/110.0);
    // A lake is wide enough for any triangles and stays at its level.
    let lake = fall < -1.5;
    let lift = select(smoothstepf(150.0, 650.0, flat_distance)*min(spacing*0.45, 12.0), 0.0, lake);
    let world = vec3<f32>(position.x, position.y + lift, position.z);
    var out: RiverVertexOutput;
    out.clip_position = globals.projection*globals.view*vec4<f32>(world, 1.0);
    out.world_position = world;
    out.velocity = velocity;
    out.channel = vec4<f32>(across, along, depth, turbulence);
    out.sheet = vec4<f32>(half_width, fall, drop, position.y);
    return out;
}

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

struct RiverFlow
{
    velocity: vec2<f32>,
    // Surface gradient from the humps the rocks raise.
    slope: vec2<f32>,
    foam: f32,
    // Extra roughness: wakes and holes churn the surface.
    churn: f32,
};

// What the boulders near a point do to the water there.
fn riverRockFlow(p: vec2<f32>, base: vec2<f32>, surface: f32) -> RiverFlow
{
    var result = RiverFlow(base, vec2<f32>(0.0), 0.0, 0.0);
    let resolution = river_grid[3];
    if (resolution == 0u) { return result; }
    let origin = vec2<f32>(bitcast<f32>(river_grid[0]), bitcast<f32>(river_grid[1]));
    let cell = floor((p - origin)/bitcast<f32>(river_grid[2]));
    if (any(cell < vec2<f32>(0.0)) || any(cell >= vec2<f32>(f32(resolution)))) { return result; }
    let speed = length(base);
    if (speed < 0.03) { return result; }
    // Three words per cell: see river-functions.wgslinc.
    let entry = 8u + (u32(cell.y)*resolution + u32(cell.x))*3u;
    let offset = river_grid[entry];
    let packed = river_grid[entry + 1u];
    let first_rock = offset + (packed & 0xffffu);
    let rock_count = min(packed >> 16u, RIVER_MAX_ROCKS);
    let e = base/speed;
    let n = vec2<f32>(-e.y, e.x);
    let head = speed*speed/(2.0*RIVER_GRAVITY);
    var deflection = vec2<f32>(0.0);
    for (var k = 0u; k < rock_count; k += 1u)
    {
        let rock = river_rocks[river_grid[first_rock + k]];
        let r = p - rock.position;
        let reach = rock.radius*7.0;
        if (dot(r, r) > reach*reach) { continue; }
        let x = dot(r, e);
        let y = dot(r, n);
        let emergence = rock.top - surface;
        if (emergence > 0.0)
        {
            // Radius where the rounded block meets the water.
            let radius = rock.radius*clamp(0.6 + emergence/(rock.radius*0.9), 0.6, 1.0);
            let r2 = max(x*x + y*y, radius*radius);
            let ratio = radius*radius/(r2*r2);
            // Potential flow past a cylinder: the stream parts around the
            // rock, slows ahead of it and speeds past its flanks.
            let around = e*(-speed*ratio*(x*x - y*y)) + n*(-speed*2.0*ratio*x*y);
            // Behind it the flow separates: a slack, churning wake.
            let wake_width = radius*(1.0 + 0.2*max(x, 0.0)/radius);
            let wake = select(0.0,
                              exp(-(y*y)/(wake_width*wake_width))
                              *exp(-max(x - 0.5*radius, 0.0)/(5.0*radius)),
                              x > -0.2*radius);
            deflection += around*(1.0 - wake) - base*wake*0.7;
            // The pillow against the upstream face, and its gradient.
            let pillow_centre = vec2<f32>(-1.05*radius, 0.0);
            let q = vec2<f32>(x, y) - pillow_centre;
            let spread = 0.45*radius + 0.06;
            let pillow = head*exp(-dot(q, q)/(spread*spread));
            let local_slope = -2.0*pillow*q/(spread*spread);
            result.slope += e*local_slope.x + n*local_slope.y;
            // Foam where the water piles on the rock, and down its wake.
            let rim = exp(-pow((sqrt(x*x + y*y) - radius)/(0.15*radius + 0.05), 2.0));
            result.foam = max(result.foam, rim*smoothstep(0.15, 1.0, speed)*0.85);
            result.foam = max(result.foam, wake*smoothstep(0.3, 1.4, speed)*0.75);
            result.churn += wake*smoothstep(0.2, 1.0, speed);
        }
        else
        {
            // Under the surface: the water rises over the rock in a standing
            // hump, highest when it only just covers it, and breaks into a
            // hole behind it once it pours over fast (a Froude number over
            // about one).
            let cover = -emergence;
            let froude = speed*speed/(RIVER_GRAVITY*max(cover, 0.02));
            let height = min(head, 0.3*rock.radius)*exp(-cover/(0.5*rock.radius + 0.08));
            let q = vec2<f32>(x - 0.35*rock.radius, y);
            let spread = 1.1*rock.radius + 0.05;
            let hump = height*exp(-dot(q, q)/(spread*spread));
            let local_slope = -2.0*hump*q/(spread*spread);
            result.slope += e*local_slope.x + n*local_slope.y;
            let hole_q = vec2<f32>((x - 1.4*rock.radius)/(0.7*rock.radius + 0.05),
                                   y/(0.9*rock.radius + 0.05));
            let hole = exp(-dot(hole_q, hole_q))*smoothstep(0.5, 2.5, froude)
                     *exp(-cover/(rock.radius + 0.1));
            result.foam = max(result.foam, hole*0.9);
            result.churn += hole + hump*4.0;
            // The current speeds over the crest and slackens behind it.
            deflection += base*(hump/max(cover + 0.1, 0.1) - 0.4*hole);
        }
    }
    result.velocity = base + deflection;
    return result;
}

// Ripples carried by the current: (height gradient xz, subpixel slope
// variance). Two phases of noise advect downstream and cross-fade, each
// restarting from a fresh offset while invisible.
fn riverRipples(p: vec2<f32>, velocity: vec2<f32>, time: f32, roughness: f32,
                footprint: f32) -> vec3<f32>
{
    let speed = length(velocity);
    let e = select(vec2<f32>(1.0, 0.0), velocity/max(speed, 1e-4), speed > 1e-4);
    let n = vec2<f32>(-e.y, e.x);
    let period = 1.6;
    var slope = vec2<f32>(0.0);
    var unresolved = 0.0;
    for (var layer = 0u; layer < 2u; layer += 1u)
    {
        let cycle_time = time/period + f32(layer)*0.5;
        let phase = fract(cycle_time);
        let cycle = floor(cycle_time);
        let weight = 1.0 - abs(2.0*phase - 1.0);
        let jitter = vec2<f32>(hash21(vec2<f32>(cycle, f32(layer)*17.3)),
                               hash21(vec2<f32>(f32(layer)*5.1, cycle)))*31.0;
        let advected = p - velocity*(phase*period);
        // Fast water draws its ripples out into streaks along the flow;
        // riffles and wakes chop them short.
        let stretch = 1.0 + 1.6*smoothstep(0.4, 2.0, speed)*(1.0 - 0.6*roughness);
        for (var octave = 0u; octave < 3u; octave += 1u)
        {
            let wavelength = select(select(0.85, 0.32, octave == 1u), 0.12, octave == 2u);
            let amplitude = wavelength*(0.010 + 0.035*roughness + 0.006*min(speed, 3.0))
                          *select(1.0, 0.8, octave == 2u);
            let along_scale = wavelength*stretch;
            let q = vec2<f32>(dot(advected, e)/along_scale, dot(advected, n)/wavelength)
                  + jitter + vec2<f32>(f32(octave)*7.7);
            let noise = riverNoise(q);
            // Filter what the pixel cannot resolve into roughness.
            let resolved = 1.0 - smoothstepf(wavelength*0.18, wavelength*0.6, footprint);
            let gradient = e*(noise.y/along_scale) + n*(noise.z/wavelength);
            slope += gradient*amplitude*weight*resolved;
            unresolved += pow(amplitude*6.2831853/wavelength, 2.0)*weight*(1.0 - resolved)*0.5;
        }
    }
    return vec3<f32>(slope, unresolved);
}

// Foam texture streaming with the current, 0..1.
fn riverFoamPattern(p: vec2<f32>, velocity: vec2<f32>, time: f32, footprint: f32) -> f32
{
    let period = 2.2;
    var pattern = 0.0;
    for (var layer = 0u; layer < 2u; layer += 1u)
    {
        let cycle_time = time/period + f32(layer)*0.5;
        let phase = fract(cycle_time);
        let weight = 1.0 - abs(2.0*phase - 1.0);
        let jitter = vec2<f32>(hash21(vec2<f32>(floor(cycle_time), f32(layer) + 3.0)))*23.0;
        let q = p - velocity*(phase*period) + jitter;
        let coarse = mix(valueNoise(q*2.1), 0.5, smoothstepf(0.25, 0.75, footprint*2.1));
        let bubbles = mix(valueNoise(q*7.3 + vec2<f32>(13.1, 4.7)), 0.5,
                          smoothstepf(0.25, 0.75, footprint*7.3));
        pattern += (coarse*0.62 + bubbles*0.38)*weight;
    }
    return pattern;
}

@fragment
fn fs_river(in: RiverVertexOutput) -> @location(0) vec4<f32>
{
    let uv = in.clip_position.xy*globals.viewport.zw;
    let camera_position = globals.camera_position.xyz;
    let to_camera = camera_position - in.world_position;
    let view_distance = length(to_camera);
    let to_view = to_camera/max(view_distance, 1e-3);
    let time = stage.params.x;
    let across = in.channel.x;
    let turbulence = in.channel.w;
    let half_width = in.sheet.x;
    let fall = in.sheet.y;
    let drop = in.sheet.z;
    let falling = fall >= 0.0;
    let surface_level = in.sheet.w;
    // Derivatives first, in uniform control flow.
    let pixel_dx = dpdx(in.world_position);
    let pixel_dy = dpdy(in.world_position);
    let footprint = max(length(pixel_dx.xz), length(pixel_dy.xz));
    let geometric = normalize(cross(pixel_dx, pixel_dy));

    let precipitation = weatherPrecipitation(in.world_position);
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
    let above_water = smoothstepf(-0.08, 0.12, camera_position.y - in.world_position.y);

    // --- The current, the rocks and the surface ------------------------------
    let flow = riverRockFlow(in.world_position.xz, in.velocity, surface_level);
    let speed = length(flow.velocity);
    let roughness = clamp(turbulence*0.8 + flow.churn*0.6 + smoothstepf(0.6, 3.0, speed)*0.25
                          + drop*0.15, 0.0, 1.0);
    let ripples = riverRipples(in.world_position.xz, flow.velocity, time, roughness, footprint);
    let impacts = waterImpacts(in.world_position.xz, globals.storm.z, footprint,
                               view_distance, precipitation.x, precipitation.y, precipitation.w);
    let slope = ripples.xy + flow.slope + impacts.slope;
    var normal = normalize(vec3<f32>(-slope.x, 1.0, -slope.y));
    if (falling)
    {
        // A falling sheet: its own facing, rippled by the streaks.
        let streak = riverNoise(vec2<f32>(across*half_width*5.0, (in.channel.y - time*3.0)*0.7));
        normal = normalize(geometric + vec3<f32>(streak.y, 0.0, streak.z)*0.08);
    }
    if (dot(normal, to_view) < 0.0)
    {
        normal = -normal;
    }

    // --- Refraction and the body of the water ---------------------------------
    let refracted_uv = clamp(uv + normal.xz*(0.35*clamp(optical_path, 0.0, 1.0)
                             /max(view_distance, 1.0)), vec2<f32>(0.0), vec2<f32>(1.0));
    let flat_background = textureSampleLevel(scene_texture, scene_sampler, uv, 0.0).rgb;
    let refracted_background = textureSampleLevel(scene_texture, scene_sampler, refracted_uv, 0.0).rgb;
    let refracted_packed = reflectionDepth(refracted_uv);
    let accept_refraction = refracted_packed.a < 0.5 || refracted_packed.z <= water_view.z;
    let scene_linear = sceneRadiance(select(flat_background, refracted_background, accept_refraction));

    // Trees shade the water as they shade its banks: by the sun's cascades
    // by day, the moon's by night.
    let plant_shadow = riverPlantShadow(in.world_position, -water_view.z);
    let sun_plants = select(1.0, plant_shadow, vegetation_shadows.light.w < 1.5);
    let moon_plants = select(1.0, plant_shadow, vegetation_shadows.light.w > 1.5);
    let sun_visibility = terrainShadow(in.world_position, normal, to_sun)
                       *cloudShadow(in.world_position, to_sun)*sun_plants;
    let severity = weatherSeverity(in.world_position.xz);
    let sun_colour = globals.sun_colour.rgb*globals.settings_a.x
                     *weatherSunMultiplier(severity)*mix(1.0, 0.11, precipitation.z)*sun_visibility;
    let moon_colour = vec3<f32>(0.63, 0.74, 1.0)*globals.atmosphere.y
                      *weatherSunMultiplier(severity)*mix(1.0, 0.42, precipitation.z)*moon_plants;
    let sky_light = skyAmbient(vec3<f32>(0.0, 1.0, 0.0), in.world_position);
    // Aerated water scatters far more: whitewater is milky even unfoamed.
    let aeration = clamp(turbulence*0.6 + flow.churn*0.3 + drop*0.2, 0.0, 1.0);
    let sigma_t = RIVER_EXTINCTION*(1.0 + 2.0*aeration);
    let transmittance = exp(-sigma_t*optical_path);
    // The bed was lit as if in air: its light also crossed the water on the
    // way down, a path of its depth over the sun's height.
    let bed_depth = min(optical_path, 8.0)*max(to_view.y, 0.05);
    let downwelling = exp(-sigma_t*bed_depth/max(to_sun.y, 0.25));
    let in_scatter = (RIVER_SCATTER*(1.0 + 4.0*aeration))
                   *(sun_colour*0.08 + moon_colour*0.08 + sky_light*0.35)
                   *(1.0 - transmittance)/max(sigma_t, vec3<f32>(1e-3));
    var body = scene_linear*transmittance*downwelling + in_scatter;

    // --- Reflection and glitter -------------------------------------------
    let n_dot_v = max(dot(normal, to_view), 0.0);
    let alpha = clamp(sqrt(pow(mix(0.03, 0.16, roughness), 2.0) + ripples.z
                           + impacts.slope_variance), 0.02, 1.0);
    let reflection_direction = reflect(-to_view, normal);
    let reflected_sky = cloudSkyRadiance(reflection_direction)*mix(0.8, 0.68, precipitation.z);
    let terrain_reflection = marchWaterReflection(in.world_position, normal,
                                                  reflection_direction, alpha, surface_level);
    let reflected = mix(reflected_sky, terrain_reflection.rgb, terrain_reflection.a);
    let half_vector = normalize(to_sun + to_view);
    let n_dot_l = max(dot(normal, to_sun), 0.0);
    let specular = dGGX(max(dot(normal, half_vector), 0.0), alpha)
                 *vSmithGGX(n_dot_l, n_dot_v, alpha)*n_dot_l*0.35;
    let moon_half = normalize(to_moon + to_view);
    let moon_dot_l = max(dot(normal, to_moon), 0.0);
    let moon_specular = dGGX(max(dot(normal, moon_half), 0.0), alpha)
                      *vSmithGGX(moon_dot_l, n_dot_v, alpha)*moon_dot_l*0.35;
    let fresnel = clamp(godotFresnel(n_dot_v, stage.surface.x, stage.surface.y), 0.0, 1.0);
    var lit = mix(body, reflected, fresnel) + specular*sun_colour + moon_specular*moon_colour;

    // --- Whitewater ---------------------------------------------------------
    let foam_colour = vec3<f32>(0.90, 0.93, 0.94)
                    *((sun_colour*max(dot(normal, to_sun), 0.0)
                       + moon_colour*max(dot(normal, to_moon), 0.0))*0.06
                      + skyAmbient(normal, in.world_position)*0.8);
    let pattern = riverFoamPattern(in.world_position.xz, flow.velocity, time, footprint);
    // Eddy lines along the banks of fast water; cascades; plunge pools.
    let eddy_line = smoothstepf(0.62, 0.98, abs(across))*smoothstepf(0.7, 2.2, speed)*0.45;
    let cascade = smoothstepf(0.25, 0.85, turbulence)*0.85;
    let plunge = smoothstepf(0.05, 0.9, drop)*0.95;
    let amount = clamp(max(max(cascade, plunge), max(flow.foam, eddy_line))
                       + 0.06*smoothstepf(0.2, 1.0, speed), 0.0, 1.0);
    // Foam gathers in patches and streaks the pattern carries downstream.
    var foam = smoothstepf(1.05 - amount, 1.3 - amount*0.6, pattern)*amount
             + impacts.splash*0.2 + impacts.snow_fleck*0.06;
    if (falling)
    {
        // The sheet is aerated water falling in strands: they stream down
        // at the speed of the fall, the sheet is thin and glassy at the lip
        // and turns white as it breaks up, torn at its edges, and a thin
        // creek's veil shows the wet rock through it.
        let world_y = in.world_position.y;
        let fall_speed = 2.0 + sqrt(2.0*RIVER_GRAVITY*max(drop*fall, 0.05));
        let strands = valueNoise(vec2<f32>(across*half_width*6.0, world_y*1.4 + time*fall_speed*1.4));
        let fine = valueNoise(vec2<f32>(across*half_width*17.0 + 3.7, world_y*3.1 + time*fall_speed*3.1));
        let strand = smoothstepf(0.30, 0.85, strands*0.65 + fine*0.35);
        let edge = 1.0 - smoothstepf(0.5, 1.0, abs(across));
        let body = 0.2 + 0.45*smoothstepf(0.1, 0.6, in.channel.z);
        let aerated = smoothstepf(0.0, 0.7, fall);
        foam = clamp(strand*(0.45 + 0.55*aerated) + 0.25*aerated, 0.0, 1.0);
        let veil = clamp((body + 0.5*strand)*mix(0.3, 1.0, edge) + 0.3*aerated*edge, 0.0, 0.96);
        lit = mix(scene_linear*0.85, mix(lit, foam_colour, foam), veil);
    }
    else
    {
        lit = mix(lit, foam_colour, clamp(foam, 0.0, 1.0)*above_water);
        // Resolve the last centimetres into the bank instead of a hard edge.
        lit = mix(scene_linear, lit, smoothstepf(0.0, 0.05, optical_path));
    }

    let fog = integrateAtmosphere(-to_view, view_distance);
    lit = lit*fog.transmittance + fog.scattering;
    return vec4<f32>(pow(acesFilm(lit*globals.params.z), vec3<f32>(1.0/2.2)), 1.0);
}
