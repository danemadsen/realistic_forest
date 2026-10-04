// Deferred HDR lighting, heightfield shadows, participating fog and volumetric clouds.
// G-buffer positions/normals are in view space. All atmosphere and terrain
// marching is in world space. ACES + gamma remain the final output contract
// consumed by FXAA and the water compositor.

struct GlobalUniforms {
    view: mat4x4<f32>,            // matView: column-major world->view
    projection: mat4x4<f32>,      // matProjection (standard GL shape, z converted to [0,1] NDC)
    camera_position: vec4<f32>,   // xyz world-space eye; w unused
    sun_direction: vec4<f32>,     // xyz direction sunlight travels, WORLD space; w highest terrain in the lighting heightfield
    viewport: vec4<f32>,          // xy = width, height in px; zw = 1/width, 1/height
    params: vec4<f32>,            // x fog_density, y z_far, z exposure, w ssao_enabled (1=on, 0=off)
    settings_a: vec4<f32>,        // x sun_intensity, y texture_scale, z ao_tex_strength, w variant_scale
    settings_b: vec4<f32>,        // x normal_strength, y sparkle_strength, z flow_debug(1/0), w erosion_debug(1/0)
    sun_colour: vec4<f32>,       // RGB solar tint, w unattenuated sun intensity
    moon_direction: vec4<f32>,   // xyz direction moonlight travels
    atmosphere: vec4<f32>,       // daylight, moon intensity, local sky cover, volumetric strength
    raymarch: vec4<f32>,         // shadows, volumetrics, reflections, quality (0/1/2)
    heightfield: vec4<f32>,      // world centre XZ, world span, texel size in metres
    clouds: vec4<f32>,          // enabled, coverage, density, base altitude
    cloud_layer: vec4<f32>,     // thickness, shape scale, shadow strength, quality
    cloud_motion: vec4<f32>,    // wind offset XZ, detail strength, maximum distance
    weather: vec4<f32>, // front offset XZ, climate bias, explicit cloud overrides
    storm: vec4<f32>, // precipitation bias, phase override, elapsed seconds, wind direction radians
    lightning: vec4<f32>, // strike world position xyz, flash radiance
    lightning_meta: vec4<f32>, // seed, strike age, bolt top altitude, front speed
};
@group(0) @binding(0) var<uniform> globals: GlobalUniforms;


struct StageUniforms {
    light_direction_view: vec4<f32>,
    ao_strength: f32,
};
@group(2) @binding(0) var<uniform> stage: StageUniforms;
@group(1) @binding(0) var texture0: texture_2d<f32>;
@group(1) @binding(8) var texture0_sampler: sampler;
@group(1) @binding(1) var texture1: texture_2d<f32>;
@group(1) @binding(9) var texture1_sampler: sampler;
@group(1) @binding(2) var texture2: texture_2d<f32>;
@group(1) @binding(10) var texture2_sampler: sampler;
@group(1) @binding(3) var texture3: texture_2d<f32>;
@group(1) @binding(11) var texture3_sampler: sampler;
@group(1) @binding(4) var terrain_heightfield: texture_2d<f32>;
@group(1) @binding(12) var terrain_heightfield_sampler: sampler;
@group(1) @binding(5) var atmosphere_texture: texture_2d<f32>;
@group(1) @binding(13) var atmosphere_sampler: sampler;

// The plants' shadow cascades (src/render/vegetation_shadows.rs): depth from
// the sun, or from the moon at night, one array layer per slice of the view.
struct VegetationShadows {
    view_projection: array<mat4x4<f32>, 4>, // world -> cascade clip, depth 0 nearest the light
    splits: vec4<f32>,  // far view depth of each cascade, metres
    texel: vec4<f32>,   // world metres per texel of each cascade
    light: vec4<f32>,   // xyz direction the light travels; w 0 off, 1 sun, 2 moon
    params: vec4<f32>,  // blend band, strength, texels per side, unused
};
@group(1) @binding(6) var vegetation_shadow_map: texture_depth_2d_array;
@group(1) @binding(7) var<uniform> vegetation_shadows: VegetationShadows;
@group(1) @binding(14) var vegetation_shadow_sampler: sampler_comparison;

struct VsOutput { @builtin(position) position: vec4<f32> };
@vertex
fn vs_main(@builtin(vertex_index) vertex: u32) -> VsOutput {
    var corners = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    var out: VsOutput;
    out.position = vec4<f32>(corners[vertex], 0.0, 1.0);
    return out;
}

const PI: f32 = 3.14159265;

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
fn viewToWorld(direction: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(dot(globals.view[0].xyz, direction),
                     dot(globals.view[1].xyz, direction),
                     dot(globals.view[2].xyz, direction));
}
fn worldViewRay(uv: vec2<f32>) -> vec3<f32> {
    let ndc = vec2<f32>(uv.x*2.0 - 1.0, 1.0 - uv.y*2.0);
    return normalize(viewToWorld(vec3<f32>(ndc.x/globals.projection[0][0],
                                           ndc.y/globals.projection[1][1], -1.0)));
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
// Light reaching a point through the plants, in one cascade. The lookup is
// pushed off the surface, along its normal and toward the light, by a
// little more than a texel, so a lit trunk or crown does not shadow itself
// through the depth map's own quantisation.
fn vegetationShadowCascade(world_position: vec3<f32>, normal_world: vec3<f32>, cascade: u32) -> f32 {
    let texel = vegetation_shadows.texel[cascade];
    let to_light = -vegetation_shadows.light.xyz;
    let grazing = 1.0 - abs(dot(normal_world, to_light));
    let lookup = world_position + normal_world*texel*(1.0 + 1.5*grazing) + to_light*texel*1.5;
    let clip = vegetation_shadows.view_projection[cascade]*vec4<f32>(lookup, 1.0);
    let uv = vec2<f32>(0.5 + 0.5*clip.x, 0.5 - 0.5*clip.y);
    if (any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0)) || clip.z > 1.0) {
        return 1.0;
    }
    // A 4x4 grid of bilinear comparisons a texel apart: a smooth tent five
    // texels wide, about the penumbra the sun's disc gives a branch ten
    // metres up in the nearest cascade. Sparser taps would stamp every small
    // sunfleck into a visible grid of copies.
    let step = 1.0/vegetation_shadows.params.z;
    var lit = 0.0;
    for (var y = 0; y < 4; y++) {
        for (var x = 0; x < 4; x++) {
            let offset = vec2<f32>(f32(x) - 1.5, f32(y) - 1.5)*step;
            lit += textureSampleCompareLevel(vegetation_shadow_map, vegetation_shadow_sampler,
                                             uv + offset, i32(cascade), clip.z);
        }
    }
    return lit/16.0;
}

// Light reaching a point through the plants: the cascade covering its view
// depth, blended into the next over the last stretch of each, and fading out
// past the last.
fn vegetationShadow(world_position: vec3<f32>, normal_world: vec3<f32>, view_depth: f32) -> f32 {
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
    var lit = vegetationShadowCascade(world_position, normal_world, cascade);
    let start = select(0.0, vegetation_shadows.splits[max(cascade, 1u) - 1u], cascade > 0u);
    let end = vegetation_shadows.splits[cascade];
    let band = (end - start)*vegetation_shadows.params.x;
    let blend = clamp((view_depth - (end - band))/max(band, 1e-3), 0.0, 1.0);
    if (blend > 0.0) {
        var next = 1.0;
        if (cascade < 3u) {
            next = vegetationShadowCascade(world_position, normal_world, cascade + 1u);
        }
        lit = mix(lit, next, blend);
    }
    return mix(1.0, lit, vegetation_shadows.params.y);
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

fn henyeyGreenstein(cos_angle: f32, anisotropy: f32) -> f32 {
    let g2 = anisotropy*anisotropy;
    return (1.0 - g2)/(4.0*PI*pow(max(1.0 + g2 - 2.0*anisotropy*cos_angle, 0.01), 1.5));
}
struct FogResult { scattering: vec3<f32>, transmittance: f32 };
fn precipitationExtinction(precipitation: vec4<f32>) -> f32 {
    // Rain bands visibly sweep through the view, while a convective core
    // retains a dim humid veil in the gaps between heavy sheets.
    return precipitation.x*0.0021 + precipitation.y*0.0013
           + precipitation.z*0.00035;
}
fn integrateAtmosphereSegment(ray: vec3<f32>, start_distance: f32, end_distance: f32) -> FogResult {
    var result: FogResult;
    result.scattering = vec3<f32>(0.0);
    result.transmittance = 1.0;
    let base_density = max(globals.params.x, 0.0);
    let segment_start = clamp(start_distance, 0.0, 10000.0);
    let ray_length = max(min(end_distance, 10000.0) - segment_start, 0.0);
    if (ray_length <= 0.001) { return result; }
    if (globals.raymarch.y < 0.5) {
        let midpoint = segment_start + ray_length*0.5;
        let world_position = globals.camera_position.xyz + ray*midpoint;
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
    let count = select(select(12u, 16u, quality >= 1.0), 24u, quality >= 2.0);
    let to_sun = -normalize(globals.sun_direction.xyz);
    let to_moon = -normalize(globals.moon_direction.xyz);
    let sun_scatter = globals.sun_colour.rgb*globals.settings_a.x
                      *henyeyGreenstein(dot(ray, to_sun), 0.68);
    let moon_scatter = vec3<f32>(0.52, 0.65, 1.0)*globals.atmosphere.y
                       *henyeyGreenstein(dot(ray, to_moon), 0.45);
    let strength = max(globals.atmosphere.w, 0.0);
    // Quadratic view steps keep close shafts detailed while reaching the
    // distant landscape. Beer-Lambert accumulation conserves transmittance.
    for (var i = 0u; i < 24u; i += 1u) {
        if (i >= count) { break; }
        let a = f32(i)/f32(count);
        let b = (f32(i) + 1.0)/f32(count);
        let start = a*a*ray_length;
        let end = b*b*ray_length;
        let midpoint = segment_start + (start + end)*0.5;
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

fn D_GGX(n_dot_h: f32, alpha: f32) -> f32 {
    let a2 = alpha*alpha;
    let d = n_dot_h*n_dot_h*(a2 - 1.0) + 1.0;
    return a2/max(d*d, 1e-7);
}
fn V_SmithGGX(n_dot_l: f32, n_dot_v: f32, alpha: f32) -> f32 {
    let a2 = alpha*alpha;
    let ggx_l = n_dot_v*sqrt(n_dot_l*n_dot_l*(1.0 - a2) + a2);
    let ggx_v = n_dot_l*sqrt(n_dot_v*n_dot_v*(1.0 - a2) + a2);
    return 0.5/max(ggx_l + ggx_v, 1e-7);
}
fn acesFilm(x: vec3<f32>) -> vec3<f32> {
    return clamp((x*(2.51*x + 0.03))/(x*(2.43*x + 0.59) + 0.14),
                  vec3<f32>(0.0), vec3<f32>(1.0));
}
fn encodeOutput(radiance: vec3<f32>) -> vec4<f32> {
    return vec4<f32>(pow(acesFilm(max(radiance, vec3<f32>(0.0))*globals.params.z),
                              vec3<f32>(1.0/2.2)), 1.0);
}

fn groundHash(cell: vec2<i32>) -> f32 {
    var value = bitcast<u32>(cell.x)*0x8da6b343u ^ bitcast<u32>(cell.y)*0xd8163841u;
    value ^= value >> 16u;
    value *= 0x7feb352du;
    value ^= value >> 15u;
    value *= 0x846ca68bu;
    value ^= value >> 16u;
    return f32(value & 0x00ffffffu)/16777215.0;
}

fn groundNoise(p: vec2<f32>) -> f32 {
    let cell = vec2<i32>(floor(p));
    let f = fract(p);
    let u = f*f*(3.0 - 2.0*f);
    return mix(mix(groundHash(cell), groundHash(cell + vec2<i32>(1, 0)), u.x),
               mix(groundHash(cell + vec2<i32>(0, 1)), groundHash(cell + vec2<i32>(1, 1)), u.x),
               u.y);
}

// Rings spreading from raindrops that land in a puddle tilt its surface.
// Each 35 cm cell holds one impact repeating at a rate set by the rain.
fn puddleRipples(world_xz: vec2<f32>, rain: f32) -> vec2<f32> {
    let spacing = 0.35;
    let base = floor(world_xz/spacing - vec2<f32>(0.5));
    let period = mix(1.4, 0.45, rain);
    var tilt = vec2<f32>(0.0);
    for (var z = 0i; z < 2i; z += 1i) {
        for (var x = 0i; x < 2i; x += 1i) {
            let cell = base + vec2<f32>(f32(x), f32(z));
            let id = vec2<i32>(cell);
            if (groundHash(id + vec2<i32>(53, 7)) > rain + 0.15) { continue; }
            let centre = (cell + vec2<f32>(0.2 + 0.6*groundHash(id + vec2<i32>(91, 17)),
                                           0.2 + 0.6*groundHash(id + vec2<i32>(13, 71))))*spacing;
            let age = fract(globals.storm.z/period + groundHash(id));
            let offset = world_xz - centre;
            let distance = max(length(offset), 0.001);
            let ring = exp(-pow((distance - age*0.3)/0.03, 2.0))*(1.0 - age);
            tilt += offset/distance*sin((distance - age*0.3)*80.0)*ring;
        }
    }
    return tilt*0.3;
}

// Runs at half resolution before the composite. This entry point intentionally
// does not access atmosphere_texture, which is its own render attachment.
@fragment
fn fs_atmosphere(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = position.xy*globals.viewport.zw;
    let packed_position = textureSampleLevel(texture0, texture0_sampler, uv, 0.0);
    let distance_to_surface = select(length(packed_position.xyz),
                                     max(globals.params.y, globals.cloud_motion.w),
                                     packed_position.a < 0.5);
    let ray = worldViewRay(uv);
    let clouds = marchClouds(globals.camera_position.xyz, ray, distance_to_surface, globals.cloud_layer.w);
    if (clouds.transmittance > 0.9999) {
        let fog = integrateAtmosphereSegment(ray, 0.0, distance_to_surface);
        return vec4<f32>(fog.scattering, fog.transmittance);
    }
    // Place the integrated cloud at its extinction-weighted distance, then
    // transport air in front and behind separately. This preserves foreground
    // haze and mountain silhouettes without drawing clouds over opaque terrain.
    let front = integrateAtmosphereSegment(ray, 0.0, clouds.distance);
    let back = integrateAtmosphereSegment(ray, clouds.distance, distance_to_surface);
    let scattering = front.scattering + front.transmittance
                     *(clouds.scattering + clouds.transmittance*back.scattering);
    let transmittance = front.transmittance*clouds.transmittance*back.transmittance;
    return vec4<f32>(scattering, transmittance);
}

fn upsampleAtmosphere(uv: vec2<f32>, packed_position: vec4<f32>) -> vec4<f32> {
    let size = vec2<i32>(textureDimensions(atmosphere_texture));
    let texel = uv*vec2<f32>(size) - vec2<f32>(0.5);
    let base = vec2<i32>(floor(texel));
    let fraction = fract(texel);
    let target_sky = packed_position.a < 0.5;
    let target_depth = length(packed_position.xyz);
    var fog = vec4<f32>(0.0);
    var total_weight = 0.0;
    for (var y = 0i; y < 2i; y += 1i) {
        for (var x = 0i; x < 2i; x += 1i) {
            let coord = clamp(base + vec2<i32>(x, y), vec2<i32>(0), size - vec2<i32>(1));
            let tap_uv = (vec2<f32>(coord) + vec2<f32>(0.5))/vec2<f32>(size);
            let tap_position = textureSampleLevel(texture0, texture0_sampler, tap_uv, 0.0);
            let tap_sky = tap_position.a < 0.5;
            let spatial = select(1.0 - fraction.x, fraction.x, x == 1i)
                          *select(1.0 - fraction.y, fraction.y, y == 1i);
            let depth_weight = exp(-abs(length(tap_position.xyz) - target_depth)
                                   /max(target_depth*0.025, 5.0));
            let weight = spatial*select(depth_weight, 1.0, target_sky && tap_sky)
                         *select(0.0, 1.0, target_sky == tap_sky);
            fog += textureLoad(atmosphere_texture, coord, 0)*weight;
            total_weight += weight;
        }
    }
    if (total_weight > 0.0001) { return fog/total_weight; }
    // A subpixel ridge can have no matching half-resolution tap. Evaluate
    // local extinction instead of borrowing sky fog across the silhouette.
    let distance_to_surface = select(target_depth, globals.params.y, target_sky);
    let fallback_midpoint = globals.camera_position.xyz
                            + worldViewRay(uv)*min(distance_to_surface, 10000.0)*0.5;
    let severity = weatherSeverity(fallback_midpoint.xz);
    let precipitation = weatherPrecipitation(fallback_midpoint);
    let extinction = max(globals.params.x, 0.0)*weatherFogMultiplier(severity)
                       + precipitationExtinction(precipitation);
    let transmission = exp(-extinction
                             *min(distance_to_surface, 10000.0));
    let ambient = weatherFogAmbient(severity, globals.atmosphere.x,
                                    precipitation.z);
    return vec4<f32>(ambient*(1.0 - transmission), transmission);
}

@fragment
fn fs_main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = position.xy*globals.viewport.zw;
    // Hoist derivative-dependent reads before the geometry/sky branch.
    let packed_position = textureSample(texture0, texture0_sampler, uv);
    let normal_sample = textureSample(texture1, texture1_sampler, uv);
    let albedo_sample = textureSample(texture2, texture2_sampler, uv);
    let ssao = textureSample(texture3, texture3_sampler, uv).r;
    let world_ray = worldViewRay(uv);
    let is_sky = packed_position.a < 0.5;
    let view_distance = select(length(packed_position.xyz), globals.params.y, is_sky);
    let fog = upsampleAtmosphere(uv, packed_position);
    // The lightning pass draws the channel itself; this pass lights the
    // scene, clouds and air with its flash.
    if (is_sky) {
        return encodeOutput(skyRadiance(world_ray)*fog.a + fog.rgb);
    }
    // G-buffer alpha stores snow to hundredths and grass in the residue.
    let snow_mask = clamp(floor((packed_position.a - 1.0)*100.0 + 0.5)*0.01, 0.0, 1.0);
    let grass_mask = clamp((packed_position.a - 1.0 - snow_mask)*10000.0, 0.0, 1.0);
    let normal_view = normalize(normal_sample.xyz*2.0 - 1.0);
    let normal_world = normalize(viewToWorld(normal_view));
    let world_position = globals.camera_position.xyz + viewToWorld(packed_position.xyz);
    let local_precipitation = weatherPrecipitation(world_position);
    let upward = smoothstep(0.15, 0.75, normal_world.y);
    // Fronts translate with the wind. Looking upwind samples the rain that
    // crossed this ground during the previous minute, so soil and rock stay
    // damp between rain sheets and dry gradually behind a departing cell.
    var lingering_rain = 0.0;
    if (globals.lightning_meta.w > 0.05) {
        let wind_direction = vec2<f32>(cos(globals.storm.w), sin(globals.storm.w));
        let previous_offset = wind_direction*globals.lightning_meta.w*70.0;
        let earlier_ground = vec3<f32>(world_position.x + previous_offset.x,
                                       world_position.y,
                                       world_position.z + previous_offset.y);
        let earlier_storm = weatherStormBase(earlier_ground);
        lingering_rain = earlier_storm.x*(1.0 - earlier_storm.y)*0.62;
    }
    let soak = max(local_precipitation.x, lingering_rain);
    let wetness = soak*upward*mix(1.0, 0.32, grass_mask);
    let falling_snow = local_precipitation.y*upward*mix(1.0, 0.25, grass_mask)
                       *(1.0 - snow_mask);
    // Rain pools on level ground outside the grass: a broad pattern of
    // hollows fills as the ground stays wet, so puddles spread through a
    // downpour and shrink again once the cell has passed. Water ignores the
    // small bumps of the surface texture, so levelness comes from the terrain
    // heightfield's slope rather than the shading normal.
    let texel = globals.heightfield.w;
    let slope = vec2<f32>(
        terrainHeightAt(world_position.xz + vec2<f32>(texel, 0.0))
            - terrainHeightAt(world_position.xz - vec2<f32>(texel, 0.0)),
        terrainHeightAt(world_position.xz + vec2<f32>(0.0, texel))
            - terrainHeightAt(world_position.xz - vec2<f32>(0.0, texel)))/(2.0*texel);
    let level = (1.0 - smoothstep(0.03, 0.09, length(slope)))
                *smoothstep(0.55, 0.8, normal_world.y)*terrainMapWeight(world_position.xz);
    let hollows = groundNoise(world_position.xz/4.5)*0.7
                  + groundNoise(world_position.xz/1.6 + vec2<f32>(17.0, 5.0))*0.3;
    let puddle = smoothstep(0.66 - 0.14*soak, 0.72 - 0.14*soak, hollows)
                 *level*smoothstep(0.25, 0.6, soak)
                 *(1.0 - grass_mask)*(1.0 - snow_mask)
                 *smoothstep(0.2, 0.6, world_position.y);
    let roughness = mix(mix(clamp(normal_sample.a, 0.06, 1.0), 0.22, wetness*0.55),
                        0.02, puddle);
    var albedo = albedo_sample.rgb*albedo_sample.rgb*2.5;
    albedo *= mix(1.0 - wetness*0.35, 0.3, puddle);
    albedo = mix(albedo, vec3<f32>(0.48, 0.53, 0.58), falling_snow*0.18);
    let ao_factor = mix(1.0, ssao, clamp(stage.ao_strength, 0.0, 1.0))
                    *mix(1.0, albedo_sample.a, clamp(globals.settings_a.z, 0.0, 1.0));
    let to_light = -normalize(stage.light_direction_view.xyz);
    let to_camera = normalize(-packed_position.xyz);
    let half_vector = normalize(to_light + to_camera);
    let signed_n_dot_l = dot(normal_view, to_light);
    let n_dot_l = max(signed_n_dot_l, 0.0);
    let n_dot_h = max(dot(normal_view, half_vector), 0.0);
    let n_dot_v = max(dot(normal_view, to_camera), 0.0);
    let v_dot_h = max(dot(to_camera, half_vector), 0.0);
    // Plants shade whichever light their cascades were drawn from.
    let plant_shadow = vegetationShadow(world_position, normal_world, -packed_position.z);
    let plant_light = vegetation_shadows.light.w;
    var sun_visibility = 1.0;
    if (globals.settings_a.x > 0.01) {
        sun_visibility = terrainShadow(world_position, normal_world, -normalize(globals.sun_direction.xyz))
                         *cloudShadow(world_position, -normalize(globals.sun_direction.xyz))
                         *select(1.0, plant_shadow, plant_light > 0.5 && plant_light < 1.5);
    }
    let local_weather = weatherSeverity(world_position.xz);
    let sun_colour = globals.sun_colour.rgb*globals.settings_a.x
                     *weatherSunMultiplier(local_weather)
                     *mix(1.0, 0.11, local_precipitation.z)*sun_visibility;
    let alpha = roughness*roughness;
    let fresnel = vec3<f32>(0.04) + vec3<f32>(0.96)*pow(1.0 - v_dot_h, 5.0);
    let specular = D_GGX(n_dot_h, alpha)*V_SmithGGX(n_dot_l, n_dot_v, alpha)*fresnel
                   *mix(1.0, 0.10, grass_mask);
    let wrap = 0.22*snow_mask;
    // Thin grass blades transmit some sunlight through their back faces.
    // Their transmitted and direct light use the same raymarched terrain/cloud
    // visibility above, so a tuft in mountain or cloud shadow still darkens.
    let opaque_diffuse = clamp((n_dot_l + wrap)/(1.0 + wrap), 0.0, 1.0);
    let blade_diffuse = clamp((signed_n_dot_l + 0.35)/1.35, 0.0, 1.0)
                        + 0.28*max(-signed_n_dot_l, 0.0);
    let wrapped_diffuse = mix(opaque_diffuse, blade_diffuse, grass_mask);
    var lit = ((albedo/PI)*wrapped_diffuse + specular*n_dot_l)*sun_colour;
    let hotspot = pow(max(dot(to_camera, to_light), 0.0), 3.0)
                  *0.048*smoothstep(15.0, 45.0, view_distance)*grass_mask;
    lit += albedo*sun_colour*hotspot*max(n_dot_l, 0.15);
    lit *= mix(ao_factor, 1.0, n_dot_l*0.35);

    // Moonlight follows its own direction and terrain occlusion at night.
    let to_moon = -normalize(globals.moon_direction.xyz);
    var moon_visibility = 1.0;
    if (globals.atmosphere.y > 0.001) {
        moon_visibility = terrainShadow(world_position, normal_world, to_moon)*cloudShadow(world_position, to_moon)
                          *select(1.0, plant_shadow, plant_light > 1.5);
    }
    lit += albedo/PI*vec3<f32>(0.52, 0.65, 1.0)*globals.atmosphere.y
           *weatherSunMultiplier(local_weather)
           *mix(1.0, 0.42, local_precipitation.z)
           *max(dot(normal_world, to_moon), 0.0)*moon_visibility*ao_factor;
    let ambient = skyAmbient(normal_world, world_position);
    let ambient_colour = mix(ambient*vec3<f32>(0.80, 1.00, 0.68),
                             ambient*vec3<f32>(0.865, 0.955, 1.189), snow_mask);
    lit += albedo*ambient_colour*mix(0.22, 0.42, snow_mask)*ao_factor;
    if (globals.lightning.w > 0.001) {
        let source = vec3<f32>(globals.lightning.x,
                               mix(globals.lightning.y, globals.lightning_meta.z, 0.58),
                               globals.lightning.z);
        let to_flash = source - world_position;
        let range = max(length(to_flash), 1.0);
        let direction = to_flash/range;
        let attenuation = 1.0/(1.0 + pow(range/2800.0, 2.0));
        let flash_colour = vec3<f32>(0.73, 0.85, 1.0)
                           *globals.lightning.w*attenuation;
        let incidence = max(dot(normal_world, direction), 0.0);
        let water = max(wetness, puddle);
        let wet_glint = pow(max(dot(reflect(-direction, normal_world), -world_ray), 0.0),
                            mix(28.0, 120.0, water));
        lit += flash_colour*((albedo/PI)*incidence*ao_factor
                             + wet_glint*water*1.6);
    }
    // Wet films and puddles mirror the sky. Ripples from falling drops break
    // up the image in a puddle; a film on rough ground only adds a sheen.
    let sheen = max(wetness*0.3, puddle);
    if (sheen > 0.01) {
        var mirror_normal = normalize(mix(normal_world, vec3<f32>(0.0, 1.0, 0.0), puddle));
        if (puddle > 0.01 && view_distance < 40.0 && local_precipitation.x > 0.01) {
            let tilt = puddleRipples(world_position.xz, local_precipitation.x)
                       *(1.0 - smoothstep(15.0, 40.0, view_distance));
            mirror_normal = normalize(mirror_normal + vec3<f32>(tilt.x, 0.0, tilt.y)*puddle);
        }
        var mirror_direction = reflect(world_ray, mirror_normal);
        mirror_direction = normalize(vec3<f32>(mirror_direction.x, max(mirror_direction.y, 0.01),
                                               mirror_direction.z));
        let facing = clamp(dot(mirror_normal, -world_ray), 0.0, 1.0);
        // A rough film loses most of the grazing-angle boost a mirror gets.
        let grazing = max(1.0 - roughness, 0.02);
        let mirror_fresnel = 0.02 + (grazing - 0.02)*pow(1.0 - facing, 5.0);
        let reflected_sky = skyRadiance(mirror_direction)*mix(1.0, ao_factor, 0.6);
        lit = mix(lit, reflected_sky, mirror_fresnel*sheen);
    }
    return encodeOutput(lit*fog.a + fog.rgb);
}
