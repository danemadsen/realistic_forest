// Deferred HDR lighting, heightfield raymarched shadows and participating fog.
// G-buffer positions/normals are in view space. All atmosphere and terrain
// marching is in world space. ACES + gamma remain the final output contract
// consumed by FXAA and the water compositor.

struct GlobalUniforms {
    view: mat4x4<f32>,            // matView: column-major world->view
    projection: mat4x4<f32>,      // matProjection (standard GL shape, z converted to [0,1] NDC)
    camera_position: vec4<f32>,   // xyz world-space eye; w unused
    sun_direction: vec4<f32>,     // xyz direction sunlight travels, WORLD space
    viewport: vec4<f32>,          // xy = width, height in px; zw = 1/width, 1/height
    params: vec4<f32>,            // x fog_density, y z_far, z exposure, w ssao_enabled (1=on, 0=off)
    settings_a: vec4<f32>,        // x sun_intensity, y texture_scale, z ao_tex_strength, w variant_scale
    settings_b: vec4<f32>,        // x normal_strength, y sparkle_strength, z flow_debug(1/0), w erosion_debug(1/0)
    sun_colour: vec4<f32>,       // RGB solar tint, w unattenuated sun intensity
    moon_direction: vec4<f32>,   // xyz direction moonlight travels
    atmosphere: vec4<f32>,       // daylight, moon intensity, hours, volumetric strength
    raymarch: vec4<f32>,         // shadows, volumetrics, reflections, quality (0/1/2)
    heightfield: vec4<f32>,      // world centre XZ, world span, texel size in metres
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

fn henyeyGreenstein(cos_angle: f32, anisotropy: f32) -> f32 {
    let g2 = anisotropy*anisotropy;
    return (1.0 - g2)/(4.0*PI*pow(max(1.0 + g2 - 2.0*anisotropy*cos_angle, 0.01), 1.5));
}
struct FogResult { scattering: vec3<f32>, transmittance: f32 };
fn integrateAtmosphere(ray: vec3<f32>, distance_to_surface: f32) -> FogResult {
    var result: FogResult;
    result.scattering = vec3<f32>(0.0);
    result.transmittance = 1.0;
    let density = max(globals.params.x, 0.0);
    if (density <= 0.0) { return result; }
    let ray_length = min(distance_to_surface, 10000.0);
    let ambient = mix(vec3<f32>(0.012, 0.019, 0.037),
                      vec3<f32>(0.32, 0.44, 0.59), clamp(globals.atmosphere.x, 0.0, 1.0));
    if (globals.raymarch.y < 0.5) {
        result.transmittance = exp(-density*ray_length);
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
        let midpoint = (start + end)*0.5;
        let world_position = globals.camera_position.xyz + ray*midpoint;
        let height_density = exp(-max(world_position.y - 20.0, 0.0)/450.0);
        let extinction = density*(0.16 + 0.84*height_density);
        let step_transmittance = exp(-extinction*(end - start));
        var sunlight_visibility = 1.0;
        if (globals.settings_a.x > 0.01 && strength > 0.0) {
            sunlight_visibility = volumeLightVisibility(world_position, to_sun);
        }
        let lighting = ambient + (sun_scatter*sunlight_visibility + moon_scatter)*strength;
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

// Runs at half resolution before the composite. This entry point intentionally
// does not access atmosphere_texture, which is its own render attachment.
@fragment
fn fs_atmosphere(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = position.xy*globals.viewport.zw;
    let packed_position = textureSampleLevel(texture0, texture0_sampler, uv, 0.0);
    let distance_to_surface = select(length(packed_position.xyz), globals.params.y,
                                     packed_position.a < 0.5);
    let fog = integrateAtmosphere(worldViewRay(uv), distance_to_surface);
    return vec4<f32>(fog.scattering, fog.transmittance);
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
    let transmission = exp(-max(globals.params.x, 0.0)*min(distance_to_surface, 10000.0));
    let ambient = mix(vec3<f32>(0.012, 0.019, 0.037),
                      vec3<f32>(0.32, 0.44, 0.59), clamp(globals.atmosphere.x, 0.0, 1.0));
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
    if (is_sky) {
        return encodeOutput(skyRadiance(world_ray)*fog.a + fog.rgb);
    }
    // G-buffer alpha stores snow to hundredths and grass in the residue.
    let snow_mask = clamp(floor((packed_position.a - 1.0)*100.0 + 0.5)*0.01, 0.0, 1.0);
    let grass_mask = clamp((packed_position.a - 1.0 - snow_mask)*10000.0, 0.0, 1.0);
    let normal_view = normalize(normal_sample.xyz*2.0 - 1.0);
    let normal_world = normalize(viewToWorld(normal_view));
    let world_position = globals.camera_position.xyz + viewToWorld(packed_position.xyz);
    let roughness = clamp(normal_sample.a, 0.06, 1.0);
    let albedo = albedo_sample.rgb*albedo_sample.rgb*2.5;
    let ao_factor = mix(1.0, ssao, clamp(stage.ao_strength, 0.0, 1.0))
                    *mix(1.0, albedo_sample.a, clamp(globals.settings_a.z, 0.0, 1.0));
    let to_light = -normalize(stage.light_direction_view.xyz);
    let to_camera = normalize(-packed_position.xyz);
    let half_vector = normalize(to_light + to_camera);
    let n_dot_l = max(dot(normal_view, to_light), 0.0);
    let n_dot_h = max(dot(normal_view, half_vector), 0.0);
    let n_dot_v = max(dot(normal_view, to_camera), 0.0);
    let v_dot_h = max(dot(to_camera, half_vector), 0.0);
    var sun_visibility = 1.0;
    if (globals.settings_a.x > 0.01) {
        sun_visibility = terrainShadow(world_position, normal_world, -normalize(globals.sun_direction.xyz));
    }
    let sun_colour = globals.sun_colour.rgb*globals.settings_a.x*sun_visibility;
    let alpha = roughness*roughness;
    let fresnel = vec3<f32>(0.04) + vec3<f32>(0.96)*pow(1.0 - v_dot_h, 5.0);
    let specular = D_GGX(n_dot_h, alpha)*V_SmithGGX(n_dot_l, n_dot_v, alpha)*fresnel
                   *mix(1.0, 0.10, grass_mask);
    let wrap = 0.22*snow_mask;
    let wrapped_diffuse = clamp((n_dot_l + wrap)/(1.0 + wrap), 0.0, 1.0);
    var lit = ((albedo/PI)*wrapped_diffuse + specular*n_dot_l)*sun_colour;
    let hotspot = pow(max(dot(to_camera, to_light), 0.0), 3.0)
                  *0.048*smoothstep(15.0, 45.0, view_distance)*grass_mask;
    lit += albedo*sun_colour*hotspot*max(n_dot_l, 0.15);
    lit *= mix(ao_factor, 1.0, n_dot_l*0.35);

    // Moonlight follows its own direction and terrain occlusion at night.
    let to_moon = -normalize(globals.moon_direction.xyz);
    var moon_visibility = 1.0;
    if (globals.atmosphere.y > 0.001) {
        moon_visibility = terrainShadow(world_position, normal_world, to_moon);
    }
    lit += albedo/PI*vec3<f32>(0.52, 0.65, 1.0)*globals.atmosphere.y
           *max(dot(normal_world, to_moon), 0.0)*moon_visibility*ao_factor;
    let ambient = skyAmbient(normal_world);
    let ambient_colour = mix(ambient*vec3<f32>(0.80, 1.00, 0.68),
                             ambient*vec3<f32>(0.865, 0.955, 1.189), snow_mask);
    lit += albedo*ambient_colour*mix(0.22, 0.42, snow_mask)*ao_factor;
    return encodeOutput(lit*fog.a + fog.rgb);
}
