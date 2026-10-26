// The buildings in the terrain G-buffer and the plants' shadow cascades, and
// their window glass over the lit scene.
//
// Every placed building and door is an instance of its model's mesh, turned
// about +Y and set at its origin. The kit's materials are textured (base
// colour, and a detail map packing the normal's XY, roughness and ambient
// occlusion), and every vertex carries the share of the sky it sees past the
// building and the share of the rain that reaches it, baked when the kit was
// exported (scripts/export_buildings.py).
//
// The G-buffer position's alpha tells the composite what a surface is: under
// 0.5 the sky, 1.0 and up the ground and plants (snow in the hundredths, grass
// and leaf translucency in the ten-thousandths), and 3.0 and up a building,
// its sky and rain shares packed in 1/4096ths: the composite dims the sky's
// light indoors by the first and keeps sheltered floors dry by the second.
//
// `vs_shadow`/`fs_shadow` draw the same instances depth-only from the light
// into the cascades the plants were drawn into, so roofs shade the floors
// under them and houses shade the ground, the plants and each other.
//
// `fs_glass` draws the window panes, which the G-buffer leaves out, blended
// over the composite's output.

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
    atmosphere: vec4<f32>,       // daylight, moon intensity, local sky cover, volumetric strength
    raymarch: vec4<f32>,         // shadows, volumetrics, reflections, quality (0/1/2)
    heightfield: vec4<f32>,      // world centre XZ, world span, texel size in metres
    clouds: vec4<f32>,          // enabled, coverage, density, base altitude
    cloud_layer: vec4<f32>,     // thickness, shape scale, shadow strength, quality
    cloud_motion: vec4<f32>,    // wind offset XZ, detail strength, maximum distance
    weather: vec4<f32>, // front offset XZ, climate bias, explicit cloud overrides
    storm: vec4<f32>, // precipitation bias, type override, elapsed time, wind direction radians
    lightning: vec4<f32>, // strike world xyz, HDR flash
    lightning_meta: vec4<f32>, // seed, age, bolt top, front speed
};
@group(0) @binding(0) var<uniform> globals: GlobalUniforms;

// One shadow cascade being drawn; the shadow pipeline binds it in place of
// the globals. Mirrors `CascadeUniform` in crates/forest_render/src/building_node.rs.
struct ShadowCascade {
    view_projection: mat4x4<f32>, // world -> cascade clip, depth 0 nearest the light
    light: vec4<f32>,             // direction the light travels, metres per texel
};
@group(0) @binding(1) var<uniform> cascade: ShadowCascade;

// Mirrors `MaterialUniform` in crates/forest_render/src/building_node.rs.
struct BuildingMaterial {
    base_color_factor: vec4<f32>,
    params: vec4<f32>,   // roughness factor, normal scale, unused, unused
};
@group(1) @binding(0) var base_color: texture_2d<f32>;
@group(1) @binding(1) var detail_map: texture_2d<f32>;   // normal XY, roughness, occlusion
@group(1) @binding(2) var material_sampler: sampler;
@group(1) @binding(3) var<uniform> material: BuildingMaterial;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tangent: vec4<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) sky: vec2<f32>,
};

// A building-space direction turned about +Y by the instance's cosine and
// sine (crates/forest_buildings/src/collision.rs, BuildingFrame::turn).
fn turnAboutY(v: vec3<f32>, turn: vec2<f32>) -> vec3<f32> {
    return vec3<f32>(v.x*turn.x + v.z*turn.y, v.y, -v.x*turn.y + v.z*turn.x);
}

@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) tangent: vec4<f32>,
    @location(4) sky: vec2<f32>,
    @location(5) origin: vec3<f32>,
    @location(6) turn: vec2<f32>,
) -> VsOut {
    let world = origin + turnAboutY(position, turn);
    var out: VsOut;
    out.clip = globals.projection*globals.view*vec4<f32>(world, 1.0);
    out.world = world;
    out.normal = turnAboutY(normal, turn);
    out.tangent = vec4<f32>(turnAboutY(tangent.xyz, turn), tangent.w);
    out.uv = uv;
    out.sky = sky;
    return out;
}

struct Gbuffer {
    @location(0) position: vec4<f32>,
    @location(1) normal: vec4<f32>,
    @location(2) albedo: vec4<f32>,
};

// The sky and rain shares in 1/4096ths over 3.0, 63 steps each.
fn packShelter(sky: vec2<f32>) -> f32 {
    let steps = floor(clamp(sky, vec2<f32>(0.0), vec2<f32>(1.0))*63.0 + vec2<f32>(0.5));
    return 3.0 + (steps.x*64.0 + steps.y + 0.5)/4096.0;
}

@fragment
fn fs_main(in: VsOut, @builtin(front_facing) front: bool) -> Gbuffer {
    let colour = textureSample(base_color, material_sampler, in.uv)*material.base_color_factor;
    let detail = textureSample(detail_map, material_sampler, in.uv);

    var tangent_normal = vec3<f32>((detail.xy*2.0 - 1.0)*material.params.y, 0.0);
    tangent_normal.z = sqrt(max(1.0 - dot(tangent_normal.xy, tangent_normal.xy), 0.0));
    let n = normalize(in.normal);
    let raw_tangent = in.tangent.xyz - n*dot(n, in.tangent.xyz);
    let t = raw_tangent/max(length(raw_tangent), 1e-5);
    let b = cross(n, t)*in.tangent.w;
    var world_normal = normalize(t*tangent_normal.x + b*tangent_normal.y + n*tangent_normal.z);
    // The kit's walls are skins: a face seen from behind shades as its front.
    world_normal = select(-world_normal, world_normal, front);
    let roughness = clamp(detail.b*material.params.x, 0.35, 1.0);

    // As the terrain does, rebuild the view-space position from the world
    // position rather than interpolating it.
    let view_position = (globals.view*vec4<f32>(in.world, 1.0)).xyz;
    let view_normal = normalize((globals.view*vec4<f32>(world_normal, 0.0)).xyz);
    var out: Gbuffer;
    out.position = vec4<f32>(view_position, packShelter(in.sky));
    out.normal = vec4<f32>(view_normal*0.5 + 0.5, roughness);
    out.albedo = vec4<f32>(sqrt(max(colour.rgb, vec3<f32>(0.0))*0.4), clamp(detail.a, 0.0, 1.0));
    return out;
}

// Window glass, drawn over the lit scene after the composite (it is not in
// the G-buffer, so a window shows the room behind it and lets the sun in): a
// pane mirrors the sky at a glance and hardly at all head on, as Fresnel has
// it, and seen from inside the room it mirrors the dim room instead. Output
// in the composite's display encoding, blended over it.
fn acesFilm(x: vec3<f32>) -> vec3<f32> {
    return clamp((x*(2.51*x + 0.03))/(x*(2.43*x + 0.59) + 0.14), vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs_glass(in: VsOut, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    let n = normalize(select(-in.normal, in.normal, front));
    let to_eye = normalize(globals.camera_position.xyz - in.world);
    let facing = clamp(dot(n, to_eye), 0.0, 1.0);
    let fresnel = 0.04 + 0.96*pow(1.0 - facing, 5.0);
    let reflected = reflect(-to_eye, n);
    let daylight = clamp(globals.atmosphere.x, 0.0, 1.0);
    let overcast = clamp(globals.atmosphere.z, 0.0, 1.0);
    let up = clamp(reflected.y*0.5 + 0.5, 0.0, 1.0);
    let clear_sky = mix(vec3<f32>(0.42, 0.48, 0.52), vec3<f32>(0.55, 0.68, 0.88), up);
    let grey_sky = vec3<f32>(0.55, 0.58, 0.6);
    let day_sky = mix(clear_sky, grey_sky, overcast);
    let sky = mix(vec3<f32>(0.01, 0.014, 0.025), day_sky, daylight);
    // Outside, how much of the yard and sky the pane faces; inside, the room.
    let seen = select(0.08, mix(0.35, 1.0, smoothstep(0.1, 0.5, in.sky.x)), front);
    let radiance = sky*seen;
    let colour = pow(acesFilm(radiance*globals.params.z), vec3<f32>(1.0/2.2));
    let alpha = clamp(0.05 + 0.85*fresnel, 0.0, 0.85);
    return vec4<f32>(colour, alpha);
}

@vertex
fn vs_shadow(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) tangent: vec4<f32>,
    @location(4) sky: vec2<f32>,
    @location(5) origin: vec3<f32>,
    @location(6) turn: vec2<f32>,
) -> @builtin(position) vec4<f32> {
    let world = origin + turnAboutY(position, turn);
    return cascade.view_projection*vec4<f32>(world, 1.0);
}

// Depth only.
@fragment
fn fs_shadow() {
}
