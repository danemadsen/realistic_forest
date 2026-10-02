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
    storm: vec4<f32>, // precipitation bias, phase override, weather clock, local gust
    lightning: vec4<f32>, // xyz terrain strike, w HDR flash radiance
    lightning_meta: vec4<f32>, // seed, age, bolt top altitude, local thunder
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
    let local_gain = stage.params.w*(1.0 + 0.14*local_gust);
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

// Return rain, snow, thunderstorm, gust, each in [0, 1]. The altitude
// transition follows the terrain material's low mountain snowline. Explicit
// Snow/Rain/Thunderstorm trends select phase while retaining spatial cells.
fn weatherPrecipitation(world_position: vec3<f32>) -> vec4<f32> {
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
    let precipitation = smoothstep(0.74, 0.90, score + globals.storm.x)
                        *wet_gate*below_cloud;
    let cold_altitude = world_position.y
        + 70.0*sin((world_position.x - globals.weather.x)/18000.0
                   + (world_position.z - globals.weather.y)/27000.0 + 0.7);
    var snow_fraction = smoothstep(80.0, 260.0, cold_altitude);
    let override_kind = u32(globals.storm.y + 0.5);
    if (override_kind == 1u || override_kind == 3u || override_kind == 4u) { snow_fraction = 0.0; }
    if (override_kind == 2u) { snow_fraction = 1.0; }
    let rain = precipitation*(1.0 - snow_fraction);
    let snow = precipitation*snow_fraction;
    let convective_bias = select(0.0, globals.storm.x, override_kind == 3u);
    let thunderstorm = select(rain*smoothstep(0.91, 0.98, score + convective_bias),
                              0.0, override_kind == 4u);
    let gust = clamp(0.18*precipitation + 0.75*thunderstorm, 0.0, 1.0);
    return vec4<f32>(rain, snow, thunderstorm, gust);
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

fn weatherFogAmbient(severity: f32, daylight: f32) -> vec3<f32> {
    let sky_cover = weatherSkyCover(severity);
    let overcast = smoothstep(0.05, 0.9, sky_cover);
    let whiteout = smoothstep(0.78, 1.0, sky_cover);
    let night_ambient = mix(vec3<f32>(0.012, 0.019, 0.037),
                            vec3<f32>(0.021, 0.024, 0.028), overcast);
    let day_ambient = mix(mix(vec3<f32>(0.32, 0.44, 0.59),
                              vec3<f32>(0.40, 0.43, 0.43), overcast),
                          vec3<f32>(0.67, 0.69, 0.68), whiteout);
    return mix(night_ambient, day_ambient, clamp(daylight, 0.0, 1.0));
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
    var sky = mix(weather_night_sky, weather_day_sky, daylight);
    let sunset = exp(-pow((to_sun.y + 0.025)/0.17, 2.0));
    let horizon = exp(-abs(ray.y)*6.0);
    let sunward = pow(max(dot(normalize(vec3<f32>(ray.x, 0.001, ray.z)),
                              normalize(vec3<f32>(to_sun.x, 0.001, to_sun.z))), 0.0), 5.0);
    sky += vec3<f32>(0.65, 0.16, 0.025)*sunset*horizon*(0.15 + 0.85*sunward)
           *(1.0 - overcast);

    let sun_mu = clamp(dot(ray, to_sun), -1.0, 1.0);
    let sun_visible = smoothstep(-0.018, 0.012, to_sun.y);
    let sun_disk = smoothstep(0.9999832, 0.9999905, sun_mu);
    let sun_halo = exp(-(1.0 - sun_mu)*750.0)*0.10
                   + pow(max(sun_mu, 0.0), 24.0)*0.035;
    // Direct ground irradiance fades before the visible solar disc sets.
    sky += globals.sun_colour.rgb*sun_visible*(1.0 - overcast)
           *(sun_disk*14.0*globals.sun_colour.w + sun_halo*globals.settings_a.x);

    let moon_mu = clamp(dot(ray, to_moon), -1.0, 1.0);
    let moon_disk = smoothstep(0.999978, 0.999986, moon_mu);
    let moon_visible = smoothstep(-0.02, 0.03, to_moon.y)*(1.0 - daylight*0.9);
    sky += vec3<f32>(0.63, 0.74, 1.0)*moon_visible*(1.0 - overcast)
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
           *(1.0 - overcast)
           *smoothstep(-0.03, 0.18, ray.y)*0.7;
    return max(sky, vec3<f32>(0.0));
}

// Hemisphere-integrated fill depends on the surface orientation and time of
// day, never its screen row or camera pitch. Excludes the celestial discs.
fn skyAmbient(normal_world: vec3<f32>, world_xz: vec2<f32>) -> vec3<f32> {
    let daylight = clamp(globals.atmosphere.x, 0.0, 1.0);
    let sky_cover = weatherSkyCover(weatherSeverity(world_xz));
    let overcast = smoothstep(0.05, 0.9, sky_cover);
    let whiteout = smoothstep(0.78, 1.0, sky_cover);
    let upward = clamp(normal_world.y*0.5 + 0.5, 0.0, 1.0);
    let clear_day_fill = mix(vec3<f32>(0.13, 0.17, 0.22),
                             vec3<f32>(0.32, 0.44, 0.62), upward);
    let clouded_day_fill = mix(vec3<f32>(0.17, 0.19, 0.19),
                               vec3<f32>(0.30, 0.34, 0.35), upward);
    let day_fill = mix(mix(clear_day_fill, clouded_day_fill, overcast),
                       vec3<f32>(0.52, 0.55, 0.54), whiteout);
    let clear_night_fill = mix(vec3<f32>(0.007, 0.011, 0.022),
                               vec3<f32>(0.019, 0.029, 0.058), upward);
    let clouded_night_fill = mix(vec3<f32>(0.008, 0.010, 0.014),
                                 vec3<f32>(0.014, 0.018, 0.025), upward);
    let night_fill = mix(clear_night_fill, clouded_night_fill, overcast);
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
        let local_layer = cloudLocalLayer(local_severity);
        let height = clamp((world_position.y - local_layer.x)/local_layer.y, 0.0, 1.0);
        let step_transmittance = exp(-density*CLOUD_EXTINCTION*step_length);
        let contribution = result.transmittance*(1.0 - step_transmittance);
        let clear_ambient = mix(vec3<f32>(0.065, 0.080, 0.105),
                                vec3<f32>(0.26, 0.31, 0.38), height);
        let overcast_ambient = mix(vec3<f32>(0.085, 0.095, 0.10),
                                   vec3<f32>(0.22, 0.24, 0.25), height);
        let day_ambient = mix(mix(clear_ambient, overcast_ambient,
                                  smoothstep(1.0, 2.0, local_severity)),
                              vec3<f32>(0.31, 0.33, 0.33),
                              smoothstep(2.0, 3.0, local_severity));
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
                        *cloudSunMultiplier(local_severity)*0.18*transport*powder;
        }
        if (globals.atmosphere.y > 0.001) {
            let depth = cloudLightDepth(world_position, to_moon, max(quality - 1.0, 0.0));
            let transport = moon_phase*exp(-depth) + 0.32*exp(-depth*0.22);
            lighting += vec3<f32>(0.52, 0.65, 1.0)*globals.atmosphere.y
                        *cloudSunMultiplier(local_severity)*0.22*transport;
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
fn integrateAtmosphere(ray: vec3<f32>, distance_to_surface: f32) -> FogResult {
    var result: FogResult;
    result.scattering = vec3<f32>(0.0);
    result.transmittance = 1.0;
    let base_density = max(globals.params.x, 0.0);
    if (base_density <= 0.0) { return result; }
    let ray_length = min(distance_to_surface, 10000.0);
    if (globals.raymarch.y < 0.5) {
        let severity = weatherSeverity((globals.camera_position.xyz + ray*ray_length*0.5).xz);
        let density = base_density*weatherFogMultiplier(severity);
        result.transmittance = exp(-density*ray_length);
        result.scattering = weatherFogAmbient(severity, globals.atmosphere.x)
                            *(1.0 - result.transmittance);
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
        let sky_cover = weatherSkyCover(severity);
        let ambient = weatherFogAmbient(severity, globals.atmosphere.x);
        let height_density = exp(-max(world_position.y - 20.0, 0.0)/450.0);
        let extinction = base_density*weatherFogMultiplier(severity)
                         *(0.16 + 0.84*height_density);
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
                                  *(1.0 - sky_cover*0.85)*sunlight_visibility
                                 + moon_scatter*weatherSunMultiplier(severity)
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
                         direction: vec3<f32>, roughness: f32) -> vec4<f32>
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
            let above_surface = globals.camera_position.y < stage.params.z
                             || world_hit.y >= stage.params.z - 0.25;
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

// Rain impacts are analytic, short-lived circular capillary waves in world
// space. Cell seeds and phase never depend on the camera or the front value,
// so a moving rain band smoothly reveals impacts already in progress. At a
// large pixel footprint, the rings fade into a small slope variance instead
// of becoming flashing subpixel dots. Snow melts on this unfrozen surface:
// it leaves only a few soft, brief flecks, without rain-like rings.
struct WaterImpacts {
    slope: vec2<f32>,
    splash: f32,
    snow_fleck: f32,
    slope_variance: f32,
};

fn waterImpacts(world_xz: vec2<f32>, time: f32, footprint: f32,
                view_distance: f32, rain: f32, snow: f32, gust: f32) -> WaterImpacts {
    var result = WaterImpacts(vec2<f32>(0.0), 0.0, 0.0,
                              rain*0.0025 + snow*0.00025 + gust*0.0008);
    if (stage.flags.x > 0.5 || max(rain, snow) < 0.005) { return result; }
    let visibility = 1.0 - smoothstepf(25.0, 115.0, view_distance);
    let resolved = 1.0 - smoothstepf(0.055, 0.24, footprint);
    if (visibility*resolved < 0.001) { return result; }

    let spacing = 0.78;
    let coordinate = world_xz/spacing;
    let base = floor(coordinate - vec2<f32>(0.5));
    for (var z = 0i; z < 2i; z += 1i) {
        for (var x = 0i; x < 2i; x += 1i) {
            let cell = base + vec2<f32>(f32(x), f32(z));
            let seed = hash21(cell + vec2<f32>(7.13, 41.9));
            let jitter = vec2<f32>(hash21(cell + vec2<f32>(23.7, 18.2)),
                                    hash21(cell + vec2<f32>(49.1, 71.4)));
            let centre = (cell + vec2<f32>(0.5) + (jitter - vec2<f32>(0.5))*0.25)*spacing;
            let delta = world_xz - centre;
            let distance_to_drop = length(delta);

            let rain_active = smoothstepf(seed - 0.10, seed + 0.10, rain);
            if (rain_active > 0.001) {
                let age = fract(time*1.15 + hash21(cell + vec2<f32>(92.3, 12.7)));
                let life = 1.0 - smoothstepf(0.45, 0.80, age);
                let radius = 0.025 + age*0.36;
                let width = max(0.025, footprint*0.50);
                let ring_distance = (distance_to_drop - radius)/width;
                let ring = exp(-1.25*ring_distance*ring_distance)*life;
                let height = 0.0045*rain_active*rain;
                let gradient = ring*(-2.5*ring_distance/width)*height;
                result.slope += delta/max(distance_to_drop, 0.001)*gradient;
                let first_splash = 1.0 - smoothstepf(0.015, 0.12, age);
                let core = 1.0 - smoothstepf(0.01, 0.065 + footprint*0.4,
                                             distance_to_drop);
                result.splash += first_splash*core*rain_active*rain;
            }

            let snow_active = smoothstepf(seed - 0.05, seed + 0.28, snow*0.38);
            if (snow_active > 0.001) {
                let age = fract(time*0.62 + hash21(cell + vec2<f32>(39.1, 63.7)));
                let brief = 1.0 - smoothstepf(0.03, 0.23, age);
                let soft_core = 1.0 - smoothstepf(0.015, 0.085 + footprint*0.5,
                                                  distance_to_drop);
                result.snow_fleck += brief*soft_core*snow_active;
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
                     *weatherSunMultiplier(weatherSeverity(in.world_position.xz))*sun_visibility;
    let to_moon = normalize(-globals.moon_direction.xyz);
    var moon_visibility = 1.0;
    if (globals.atmosphere.y > 0.001)
    {
        moon_visibility = terrainShadow(in.world_position, normal, to_moon)
                         *cloudShadow(in.world_position, to_moon);
    }
    let moon_colour = vec3<f32>(0.63, 0.74, 1.0)*globals.atmosphere.y
                      *weatherSunMultiplier(weatherSeverity(in.world_position.xz))
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
    let ambient_source = skyAmbient(vec3<f32>(0.0, 1.0, 0.0), in.world_position.xz)
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
                       + skyAmbient(normal, in.world_position.xz)*0.75);

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
    // handles the 85-90 degree band where the curve is pinned flat.
    let reflection_direction = reflect(-to_view, normal);
    let reflected_sky = cloudSkyRadiance(reflection_direction)*0.75;

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
                                                   reflection_direction, alpha);
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
        bolt_reflection = flash_colour*glint*0.65;
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
