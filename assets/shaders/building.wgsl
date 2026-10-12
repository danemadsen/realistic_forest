// The buildings in the terrain G-buffer and the plants' shadow cascades, and
// their window glass over the lit scene.
//
// Every placed building and door is an instance of its model's mesh, turned
// about +Y and set at its origin. The kit's materials are textured (base
// colour, and a detail map packing the normal's XY, roughness and ambient
// occlusion), and every vertex carries the share of the sky it sees past the
// building and the share of the rain that reaches it, baked when the kit was
// exported (scripts/export-buildings.py).
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
    planet: vec4<f32>,          // radius, face half width, seam band, camera face (planet-functions.wgslinc)
    planet_camera: vec4<f32>,   // camera chart xy on its face, altitude, LOD reference altitude
    planet_east: vec4<f32>,     // camera east in planet axes, feature fade width
    planet_up: vec4<f32>,       // camera up in planet axes, atmosphere top altitude
    planet_north: vec4<f32>,    // camera north in planet axes
};
@group(0) @binding(0) var<uniform> globals: GlobalUniforms;

// BEGIN PLANET FUNCTIONS
// The cube-sphere planet (src/planet/mod.rs) for the shaders. Every pass that
// places geometry on the ground or reads altitude pastes this block; a test
// keeps the copies identical and its constants equal to the Rust ones.
//
// "World" space in every shader is the planar world the passes were written
// for, bent onto the planet around the camera: a Euclidean frame whose axes
// are the camera's local east, up and north, placed so the camera sits at
// globals.camera_position (its terrain plane coordinates and altitude). Near
// the camera it agrees with the terrain plane to second order; the planet's
// centre sits at (camera x, -radius, camera z). Plane points (x, altitude, z)
// on the camera's face go to world through planetPlaneToWorld, which is exact
// on the sphere and computed relative to the camera, so precision follows the
// distance to the camera rather than the planet's radius.
//
// Requires GlobalUniforms as `globals` with the planet fields:
//   planet        x sea-level radius, y face half width S, z seam band, w camera face
//   planet_camera xy camera chart on its face, z camera altitude, w LOD reference altitude
//   planet_east   xyz camera east in planet axes, w feature fade width
//   planet_up     xyz camera up in planet axes, w highest altitude of the atmosphere
//   planet_north  xyz camera north in planet axes, w unused
const PLANET_QUARTER_PI: f32 = 0.78539816339;
const PLANET_FACE_AXIS = array<vec3<f32>, 6>(
    vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(-1.0, 0.0, 0.0),
    vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, -1.0, 0.0),
    vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 0.0, -1.0));
const PLANET_FACE_EAST = array<vec3<f32>, 6>(
    vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 0.0, -1.0),
    vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(1.0, 0.0, 0.0),
    vec3<f32>(-1.0, 0.0, 0.0), vec3<f32>(1.0, 0.0, 0.0));
const PLANET_FACE_NORTH = array<vec3<f32>, 6>(
    vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 1.0, 0.0),
    vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 0.0, -1.0),
    vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 1.0, 0.0));
// Where each face's chart sits on the terrain plane (FACE_PLANE_OFFSETS).
const PLANET_FACE_PLANE_OFFSET = array<vec2<f32>, 6>(
    vec2<f32>(37123.0, -21457.0), vec2<f32>(-45311.0, 29873.0),
    vec2<f32>(0.0, 0.0), vec2<f32>(61207.0, 53119.0),
    vec2<f32>(-27449.0, -58231.0), vec2<f32>(13877.0, 71363.0));

fn planetCameraFace() -> u32
{
    return u32(globals.planet.w + 0.5);
}

// Equi-angular warp, odd by construction and exactly 1 on a face edge, so two
// faces build bit-identical cube points along the edge they share.
fn planetWarp(u: f32) -> f32
{
    let magnitude = abs(u);
    let warped = select(tan(PLANET_QUARTER_PI*magnitude), 1.0, magnitude == 1.0);
    return select(warped, -warped, u < 0.0);
}

// warp(u) - warp(u - du), accurate to the size of du rather than of u:
// tan(a) - tan(b) = sin(a - b)/(cos a cos b).
fn planetWarpDelta(u: f32, du: f32) -> f32
{
    return sin(PLANET_QUARTER_PI*du)/(cos(PLANET_QUARTER_PI*u)*cos(PLANET_QUARTER_PI*(u - du)));
}

fn planetCube(face: u32, chart: vec2<f32>) -> vec3<f32>
{
    let s = globals.planet.y;
    return PLANET_FACE_AXIS[face] + PLANET_FACE_EAST[face]*planetWarp(chart.x/s)
        + PLANET_FACE_NORTH[face]*planetWarp(chart.y/s);
}

// Planet-axis displacement from chart point `reference` at `referenceAltitude`
// to `reference + delta` at `altitude`, both on `face`. Formed from the
// difference of the two directions directly, never from two planet-centred
// positions, so it is as precise as its own length.
fn planetChartDelta(face: u32, reference: vec2<f32>, delta: vec2<f32>,
                    referenceAltitude: f32, altitude: f32) -> vec3<f32>
{
    let s = globals.planet.y;
    let chart = reference + delta;
    let cubeReference = planetCube(face, reference);
    let cube = planetCube(face, chart);
    let du = delta/s;
    let cubeDelta = PLANET_FACE_EAST[face]*planetWarpDelta(chart.x/s, du.x)
        + PLANET_FACE_NORTH[face]*planetWarpDelta(chart.y/s, du.y);
    let lengthReference = length(cubeReference);
    let lengthPoint = length(cube);
    // d - d_ref = dc/|c| + c_ref (1/|c| - 1/|c_ref|), with the reciprocal
    // difference rewritten through |c|^2 - |c_ref|^2 = dc . (c + c_ref).
    let directionDelta = cubeDelta/lengthPoint
        - cubeReference*(dot(cubeDelta, cube + cubeReference)
                         /(lengthPoint*lengthReference*(lengthPoint + lengthReference)));
    return (globals.planet.x + referenceAltitude)*directionDelta
        + (altitude - referenceAltitude)*(cube/lengthPoint);
}

// Planet axes to world axes: the camera's east, up and north.
fn planetToWorldAxes(v: vec3<f32>) -> vec3<f32>
{
    return vec3<f32>(dot(v, globals.planet_east.xyz), dot(v, globals.planet_up.xyz),
                     dot(v, globals.planet_north.xyz));
}

fn planetFromWorldAxes(v: vec3<f32>) -> vec3<f32>
{
    return globals.planet_east.xyz*v.x + globals.planet_up.xyz*v.y + globals.planet_north.xyz*v.z;
}

// World offset from the camera of a terrain plane point (x, altitude, z) on
// the camera's face.
fn planetPlaneOffset(plane: vec3<f32>) -> vec3<f32>
{
    let delta = plane.xz - globals.camera_position.xz;
    return planetToWorldAxes(planetChartDelta(planetCameraFace(), globals.planet_camera.xy, delta,
                                              globals.planet_camera.z, plane.y));
}

// A terrain plane point (x, altitude, z) on the camera's face in world space.
fn planetPlaneToWorld(plane: vec3<f32>) -> vec3<f32>
{
    return globals.camera_position.xyz + planetPlaneOffset(plane);
}

// The world direction of a plane-axis vector at a plane point: the local
// frame there, which the camera's own frame matches only at the camera.
fn planetPlaneVectorToWorld(plane: vec3<f32>, v: vec3<f32>) -> vec3<f32>
{
    let face = planetCameraFace();
    let chart = globals.planet_camera.xy + (plane.xz - globals.camera_position.xz);
    let up = normalize(planetCube(face, chart));
    let eastAxis = PLANET_FACE_EAST[face];
    let east = normalize(eastAxis - up*dot(up, eastAxis));
    let north = cross(east, up);
    return planetToWorldAxes(east*v.x + up*v.y + north*v.z);
}

// How much of the streamed features (erosion, river carving, building pads)
// the ground keeps at a terrain plane point of the camera's face: 1 across
// the face, fading out before the seam band at its edges
// (Planet::feature_weight). Things seated on those features (plants, grass,
// buildings, river surfaces) stand only where it is whole.
fn planetFeatureShare(plane_xz: vec2<f32>) -> f32
{
    let chart = plane_xz - PLANET_FACE_PLANE_OFFSET[planetCameraFace()];
    let edge = globals.planet.y - max(abs(chart.x), abs(chart.y));
    return smoothstep(globals.planet.z, globals.planet.z + globals.planet_east.w, edge);
}

// The planet's centre in world space.
fn planetCentreWorld() -> vec3<f32>
{
    return vec3<f32>(globals.camera_position.x, -globals.planet.x, globals.camera_position.z);
}

// Altitude above sea level of a world point. Formed as
// (|v|^2 - r^2)/(|v| + r) from the camera-relative offsets so it does not
// lose the radius's worth of precision a plain length(v) - r would.
fn planetAltitude(world: vec3<f32>) -> f32
{
    let r = globals.planet.x;
    let horizontal = world.xz - globals.camera_position.xz;
    let y = world.y;
    let lifted = dot(horizontal, horizontal) + y*(2.0*r + y);
    let distance = sqrt(dot(horizontal, horizontal) + (y + r)*(y + r));
    return lifted/(distance + r);
}

// The local up (away from the centre) at a world point.
fn planetUpAt(world: vec3<f32>) -> vec3<f32>
{
    return normalize(world - planetCentreWorld());
}

// The near and far distances along a world ray from `origin` at which it
// crosses the sphere `altitude` metres above sea level, or (-1, -1) when it
// misses it. A ray starting inside the sphere has a negative near distance.
fn planetRaySphere(origin: vec3<f32>, direction: vec3<f32>, altitude: f32) -> vec2<f32>
{
    let r = globals.planet.x + altitude;
    let fromCentre = origin - planetCentreWorld();
    let b = dot(fromCentre, direction);
    // |o|^2 - r^2 from the precise altitude of the origin.
    let originAltitude = planetAltitude(origin) - altitude;
    let c = originAltitude*(2.0*r + originAltitude);
    let discriminant = b*b - c;
    if (discriminant < 0.0) {
        return vec2<f32>(-1.0, -1.0);
    }
    let root = sqrt(discriminant);
    return vec2<f32>(-b - root, -b + root);
}

// The sun's elevation where a world point lies: the sine of its angle above
// the local horizon there, which the camera's own frame gives only for the
// camera's place.
fn planetSunElevationAt(world: vec3<f32>) -> f32
{
    return dot(-normalize(globals.sun_direction.xyz), planetUpAt(world));
}

// The day/night cycle's daylight curve (day_night.rs sample) at a world point,
// so the night side of the planet is dark seen from the day side. At the
// camera it is globals.atmosphere.x.
fn planetDaylightAt(world: vec3<f32>) -> f32
{
    return smoothstep(-0.18, 0.18, planetSunElevationAt(world));
}

// The direct sunlight reaching a world point: the day/night cycle's colour
// and strength curves (day_night.rs sample) for the sun's elevation there,
// times the sun's intensity. At the camera it is the uniform
// globals.sun_colour.rgb*globals.settings_a.x.
fn planetSunlightAt(world: vec3<f32>) -> vec3<f32>
{
    let elevation = planetSunElevationAt(world);
    let colourMix = smoothstep(0.0, 0.55, elevation);
    let colour = vec3<f32>(1.0, 0.29 + 0.66*colourMix, 0.095 + 0.765*colourMix);
    return colour*globals.sun_colour.w*smoothstep(0.0, 0.12, elevation);
}

// The air's density scale height. Haze thins with it above the ground fog's
// own 450 m scale, so the column over any point holds a finite amount of air
// and the planet seen from orbit wears a thin, bright limb.
const PLANET_AIR_SCALE_HEIGHT: f32 = 8000.0;

// Metres of sea-level air along a world ray from `origin`, up to `limit`
// metres or the ground: about the scale height straight up from sea level,
// nothing in space, and most along the limb.
fn planetAirMass(origin: vec3<f32>, ray: vec3<f32>, limit: f32) -> f32
{
    let shell = planetRaySphere(origin, ray, globals.planet_up.w);
    if (shell.y <= 0.0) { return 0.0; }
    var end = min(shell.y, limit);
    let ground = planetRaySphere(origin, ray, 0.0);
    if (ground.x > 0.0) { end = min(end, ground.x); }
    let start = max(shell.x, 0.0);
    if (end <= start) { return 0.0; }
    // Quadratic steps from the near end, where the air is densest when the
    // ray starts on the ground and rises.
    var mass = 0.0;
    let span = end - start;
    for (var i = 0u; i < 12u; i = i + 1u) {
        let a = f32(i)/12.0;
        let b = (f32(i) + 1.0)/12.0;
        let t = start + span*(a*a + b*b)*0.5;
        let altitude = max(planetAltitude(origin + ray*t), 0.0);
        mass += exp(-altitude/PLANET_AIR_SCALE_HEIGHT)*span*(b*b - a*a);
    }
    return mass;
}
// END PLANET FUNCTIONS

// One shadow cascade being drawn; the shadow pipeline binds it in place of
// the globals. Mirrors `CascadeUniform` in src/render/building_node.rs.
struct ShadowCascade {
    view_projection: mat4x4<f32>, // world -> cascade clip, depth 0 nearest the light
    light: vec4<f32>,             // direction the light travels, metres per texel
};
@group(0) @binding(1) var<uniform> cascade: ShadowCascade;

// Mirrors `MaterialUniform` in src/render/building_node.rs.
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
// sine (src/buildings/collision.rs, BuildingFrame::turn).
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
    // The building stands on the curved ground at its origin, upright in the
    // camera's frame (the local up tilts by under a degree in drawing range).
    let world = planetPlaneToWorld(origin) + turnAboutY(position, turn);
    var out: VsOut;
    out.clip = globals.projection*globals.view*vec4<f32>(world, 1.0);
    // A building stands on its pad, which the ground near a cube edge fades
    // out of; it stands only where the pad is whole.
    if (planetFeatureShare(origin.xz) < 0.999) { out.clip = vec4<f32>(0.0, 0.0, -1.0, 1.0); }
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
    let world = planetPlaneToWorld(origin) + turnAboutY(position, turn);
    if (planetFeatureShare(origin.xz) < 0.999) { return vec4<f32>(0.0, 0.0, -1.0, 1.0); }
    return cascade.view_projection*vec4<f32>(world, 1.0);
}

// Depth only.
@fragment
fn fs_shadow() {
}
