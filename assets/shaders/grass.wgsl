// Instanced, alpha-tested imported grass. The compute pass evaluates habitat
// once per clump and compacts the prepared roots used by this pass.
struct GlobalUniforms {
    view: mat4x4<f32>, projection: mat4x4<f32>, camera_position: vec4<f32>,
    sun_direction: vec4<f32>, viewport: vec4<f32>, params: vec4<f32>,
    settings_a: vec4<f32>, settings_b: vec4<f32>, sun_colour: vec4<f32>,
    moon_direction: vec4<f32>, atmosphere: vec4<f32>, raymarch: vec4<f32>,
    heightfield: vec4<f32>, clouds: vec4<f32>, cloud_layer: vec4<f32>, cloud_motion: vec4<f32>,
    weather: vec4<f32>, // front offset XZ, climate bias, explicit cloud overrides
    storm: vec4<f32>, // precipitation bias, type override, elapsed time, wind direction radians
    lightning: vec4<f32>, // strike world xyz, HDR flash
    lightning_meta: vec4<f32>, // seed, age, bolt top, front speed
};
@group(0) @binding(0) var<uniform> globals: GlobalUniforms;
struct GrassFrame {
    mapping: vec4<f32>, // centre XZ, span, metres per texel
    wind: vec4<f32>,    // wind direction XZ, time, strength
    range: vec4<f32>,   // full density distance, draw radius, camera XZ
    // Distance at which layers 1..3 (far middle, close middle, carpet) fade to
    // nothing. The CPU culls whole chunks by the same numbers, so they come
    // from one place: grass.rs LAYER_END.
    layer_end: vec4<f32>,
};
@group(1) @binding(0) var habitat: texture_2d<f32>;
@group(1) @binding(1) var habitat_sampler: sampler;
@group(1) @binding(2) var<uniform> frame: GrassFrame;
@group(1) @binding(3) var grass_ground_albedo: texture_2d<f32>;
struct GrassMaterial {
    colour: vec4<f32>,
    pbr: vec4<f32>, // alpha cutoff, normal strength, metal factor, rough factor
    shape: vec4<f32>, // blade height, [1] unused (the cull pass carries radius), AO strength, visible atlas mean luminance
};
@group(2) @binding(0) var base_colour: texture_2d<f32>;
@group(2) @binding(1) var normal_map: texture_2d<f32>;
@group(2) @binding(2) var orm_map: texture_2d<f32>;
@group(2) @binding(3) var material_sampler: sampler;
@group(2) @binding(4) var<uniform> material: GrassMaterial;
struct GrassOut {
    @builtin(position) position: vec4<f32>,
    @location(0) view_position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tangent: vec4<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) tint: f32,
    @location(5) blade_height: f32,
    @location(6) ground_average: vec3<f32>,
};
fn yawRotate(v: vec3<f32>, c: f32, s: f32) -> vec3<f32> {
    return vec3<f32>(v.x * c + v.z * s, v.y, -v.x * s + v.z * c);
}
@vertex
fn vs_main(
    @location(0) position: vec3<f32>, @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>, @location(3) tangent: vec4<f32>,
    @location(4) root: vec2<f32>, @location(5) rotation: f32, @location(6) scale: f32,
    @location(7) tint: f32, @location(8) seed: f32,
    @location(9) ground_height_slope_x: vec2<f32>,
    @location(10) ground_average_slope_z: vec4<f32>,
) -> GrassOut {
    var out: GrassOut;
    let c = cos(rotation); let s = sin(rotation);
    let blade = clamp(position.y / max(material.shape.x, 0.01), 0.0, 1.0);
    var offset = yawRotate(position * scale, c, s);
    // Set the wide card roots into the local slope while letting tips grow up.
    let slope = vec2<f32>(ground_height_slope_x.y, ground_average_slope_z.w);
    offset.y -= dot(slope, offset.xz) * (1.0 - blade);
    let phase = dot(root, vec2<f32>(0.37, 0.21)) - frame.wind.z * 1.8;
    let gust = sin(phase) * 0.65 + sin(phase * 0.47 + seed * 6.283) * 0.35;
    let bend = blade * blade * material.shape.x * scale;
    offset.x += frame.wind.x * gust * bend * frame.wind.w;
    offset.z += frame.wind.y * gust * bend * frame.wind.w;
    let world = vec3<f32>(root.x, ground_height_slope_x.x, root.y) + offset;
    let view = globals.view * vec4<f32>(world, 1.0);
    out.position = globals.projection * view;
    out.view_position = view.xyz;
    // A slight upward bias gives thin leaves a canopy normal, keeping
    // two-sided cards from turning into dark vertical rectangles at noon.
    out.normal = normalize(yawRotate(normal, c, s) + vec3<f32>(0.0, 0.20, 0.0));
    out.tangent = vec4<f32>(yawRotate(tangent.xyz, c, s), tangent.w);
    out.uv = uv;
    out.tint = tint;
    out.blade_height = blade;
    // One averaged terrain colour per rooted clump, before lighting and shadow.
    out.ground_average = ground_average_slope_z.rgb;
    return out;
}
struct Gbuffer {
    @location(0) position: vec4<f32>,
    @location(1) normal: vec4<f32>,
    @location(2) albedo: vec4<f32>,
};
@fragment
fn fs_main(input: GrassOut, @builtin(front_facing) front: bool) -> Gbuffer {
    let colour = textureSample(base_colour, material_sampler, input.uv) * material.colour;
    if (colour.a < material.pbr.x) { discard; }
    let orm = textureSample(orm_map, material_sampler, input.uv);
    let sampledNormal = textureSample(normal_map, material_sampler, input.uv).xyz * 2.0 - 1.0;
    let n = normalize(input.normal);
    let t = normalize(input.tangent.xyz - n * dot(n, input.tangent.xyz));
    let b = cross(n, t) * input.tangent.w;
    var worldNormal = normalize(t * sampledNormal.x * material.pbr.y
                             + b * sampledNormal.y * material.pbr.y + n * sampledNormal.z);
    worldNormal = select(-worldNormal, worldNormal, front);
    // A two-sided leaf still faces the sky on its back. Negating all three
    // components would aim half the cards into the soil and shade them black.
    worldNormal = normalize(vec3<f32>(worldNormal.x, abs(worldNormal.y), worldNormal.z));
    // The existing composite provides sunlight, sky, terrain/cloud shadows,
    // fog and grass transmission; foliage shares SSAO and water occlusion.
    let viewNormal = normalize((globals.view * vec4<f32>(worldNormal, 0.0)).xyz);
    // The grayscale atlas supplies light/dark blade detail; the neighborhood
    // averaged ground albedo supplies RGB tint and a brightness baseline. Normalise
    // by the atlas's visible-texel average so a white atlas does not make the
    // plants white. Both textures are linear before G-buffer encoding below.
    let luminance = vec3<f32>(0.2126, 0.7152, 0.0722);
    let blade_gray = dot(colour.rgb, luminance);
    let blade_detail = clamp(blade_gray / max(material.shape.w, 0.05), 0.45, 1.4);
    let value_variation = mix(0.86, 1.12, clamp((input.tint - 0.88) / 0.20, 0.0, 1.0));
    // Upright leaf cards receive less diffuse/ambient light than the flat
    // ground they cover. Lift reflectance while retaining the sampled RGB
    // ratios, so a clump stays olive over olive turf rather than going black.
    let albedo = input.ground_average * blade_detail * value_variation * 3.5;
    let rootAO = mix(0.64, 1.0, smoothstep(0.0, 0.65, input.blade_height));
    var out: Gbuffer;
    out.position = vec4<f32>(input.view_position, 1.0001);
    out.normal = vec4<f32>(viewNormal * 0.5 + 0.5, max(0.76, orm.g * material.pbr.w));
    out.albedo = vec4<f32>(sqrt(max(albedo, vec3<f32>(0.0)) * 0.4),
                          mix(1.0, orm.r, material.shape.z) * rootAO);
    return out;
}
