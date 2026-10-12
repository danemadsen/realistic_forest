// Instanced plants in the terrain G-buffer: bark, foliage cards and LOD-3
// billboards through one pipeline, fed by vegetation-cull.wgsl's compacted
// per-model, per-LOD lists. From the composite on, a plant is lit, shadowed
// by the terrain and cloud marches, fogged and reflected exactly as the
// ground is; foliage carries a translucency mask in the position alpha that
// the composite's canopy lighting reads, as the grass does.
//
// `vs_shadow`/`fs_shadow` draw the same plants into the shadow cascades:
// depth only, alpha-tested, from the light. Mesh LODs share the view's wind
// deformation; LOD-3 billboards stay static in both passes.
// `vs_canopy`/`fs_canopy` lay every tree's crown, seen from above, over the
// grass capture so the grass thins in the shade under the trees.

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
// the globals. Mirrors `CascadeUniform` in src/render/vegetation_node.rs.
struct ShadowCascade {
    view_projection: mat4x4<f32>, // world -> cascade clip, depth 0 nearest the light
    light: vec4<f32>,             // direction the light travels, metres per texel
};
@group(0) @binding(1) var<uniform> cascade: ShadowCascade;

// Mirrors `ModelParams` in src/render/vegetation_node.rs (112 bytes).
struct ModelParams {
    height: f32,
    crown_radius: f32,
    crown_base: f32,
    bound_radius: f32,
    lod_end: vec4<f32>,
    max_distance: f32,
    lod_count: u32,
    habitat: u32,
    region: u32,
    draw_word: vec4<u32>,
    draw_count: vec4<u32>,
    shadow_word: vec4<u32>,
    shadow_count: vec4<u32>,
};

struct VegetationFrame {
    wind: vec4<f32>,    // downwind direction XZ, seconds, strength (1 = an 18 m/s breeze)
    params: vec4<f32>,  // grass capture centre XZ, span, metres per texel
};
@group(1) @binding(0) var<uniform> frame: VegetationFrame;
@group(1) @binding(1) var<storage, read> models: array<ModelParams>;

// One streamed plant, as the scatter wrote it. Mirrors `PlantInstance` in
// src/vegetation/scatter.rs (32 bytes).
struct PlantInstance {
    position: vec3<f32>,
    scale: f32,
    yaw: f32,
    seed: f32,
    model: u32,
    layer: u32,   // 0 canopy, 1 regeneration: the trees
};
@group(1) @binding(2) var<storage, read> plants: array<PlantInstance>;

struct PlantMaterial {
    base_color_factor: vec4<f32>,
    params: vec4<f32>,   // alpha cutoff (0 opaque), normal scale, roughness factor, occlusion strength
    surface: vec4<f32>,  // kind (0 bark, 1 foliage, 2 billboard), translucency, albedo scale, conifer crown (1/0)
};
@group(2) @binding(0) var base_color: texture_2d<f32>;
@group(2) @binding(1) var detail_map: texture_2d<f32>;   // normal XY, roughness, occlusion
@group(2) @binding(2) var material_sampler: sampler;
@group(2) @binding(3) var<uniform> material: PlantMaterial;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tangent: vec4<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) crown_normal: vec3<f32>,
    // Crown interior (0 at the rim, 1 at the trunk), height fraction, metres
    // above the root.
    @location(5) shade: vec3<f32>,
    @location(6) @interpolate(flat) fade: f32,
    @location(7) @interpolate(flat) seed: f32,
};

fn yawRotate(v: vec3<f32>, c: f32, s: f32) -> vec3<f32> {
    return vec3<f32>(v.x * c + v.z * s, v.y, -v.x * s + v.z * c);
}

// Crown radius at a height, as a fraction of the widest: a cone narrowing to
// the leader for conifers, an ellipsoid for broadleaves and shrubs.
fn crownProfile(model: ModelParams, y: f32) -> f32 {
    let span = max(model.height - model.crown_base, 0.05);
    let along = clamp((y - model.crown_base) / span, 0.0, 1.0);
    if (material.surface.w > 0.5) {
        return clamp(1.0 - along, 0.12, 1.0);
    }
    let centred = along * 2.0 - 1.0;
    return sqrt(clamp(1.0 - centred * centred, 0.12, 1.0));
}

// Where a vertex of a placed plant ends up in the world: the plant's own
// lean, its yaw and scale, then the wind for mesh LODs only.
fn placeVertex(model: ModelParams, position: vec3<f32>, normal: vec3<f32>, root: vec3<f32>,
               scale: f32, yaw: f32, seed: f32) -> vec3<f32> {
    let kind = material.surface.x;
    let height01 = clamp(position.y / max(model.height, 0.01), 0.0, 1.0);
    let tall = model.height * scale;

    // Every plant leans a little its own way, more with height, as trees
    // that grew toward light or settled on soft ground do.
    let lean_direction = fract(seed * 91.7) * 6.2831853;
    let lean = 0.006 + 0.03 * fract(seed * 37.1);
    var local = position;
    local.x += cos(lean_direction) * lean * max(position.y, 0.0);
    local.z += sin(lean_direction) * lean * max(position.y, 0.0);
    let c = cos(yaw);
    let s = sin(yaw);
    var offset = yawRotate(local, c, s) * scale;

    // LOD-3 billboards keep their baked silhouette. Bending or fluttering
    // their crossed cards makes the alpha-tested vegetation flicker.
    // Both the view and shadow passes use this static placement.
    if (kind > 1.5) {
        return root + offset;
    }

    // Wind. Gusts sweep downwind across the canopy, so neighbours bow
    // together; each stem then sways at a rate set by its height (tall trees
    // are slow) and bends from the root, roughly quadratically with height.
    let t = frame.wind.z;
    let strength = frame.wind.w;
    let downwind = vec3<f32>(frame.wind.x, 0.0, frame.wind.y);
    let gust_phase = dot(root.xz, frame.wind.xy) * 0.021 - t * 0.85;
    let gust = 0.6 + 0.4 * sin(gust_phase) * sin(gust_phase * 0.41 + 1.3);
    let phase = seed * 6.2831853 + dot(root.xz, vec2<f32>(0.013, 0.021));
    let sway_rate = 2.6 / sqrt(max(tall, 0.6));
    let sway = (0.55 + 0.45 * sin(t * sway_rate + phase)) * gust;
    let bend = height01 * height01 * min(tall, 30.0) * 0.011 * strength * sway;
    offset += downwind * bend;
    if (kind > 0.5) {
        // Twigs and leaves flutter around their branch, the rim the most.
        let rim = clamp(length(position.xz) / max(model.crown_radius, 0.05), 0.0, 1.4);
        let flutter_rate = 4.5 + 3.0 * fract(seed * 5.3);
        let flutter = sin(t * flutter_rate + dot(position, vec3<f32>(3.1, 1.7, 2.3)))
                    * 0.018 * strength * gust * (0.25 + rim) * min(tall, 12.0) * 0.12;
        offset += yawRotate(normal, c, s) * flutter;
    }
    return root + offset;
}

@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) tangent: vec4<f32>,
    @location(4) root: vec3<f32>,
    @location(5) scale: f32,
    @location(6) yaw: f32,
    @location(7) seed: f32,
    @location(8) fade: f32,
    @location(9) model_index: u32,
) -> VsOut {
    let model = models[model_index];
    let height01 = clamp(position.y / max(model.height, 0.01), 0.0, 1.0);
    let c = cos(yaw);
    let s = sin(yaw);
    // On the planet a plant stands where its root lies on the curved ground,
    // upright in the camera's frame: within a plant's drawing distance the
    // local up tilts from the camera's by well under a degree.
    let world = planetPlaneToWorld(root) + (placeVertex(model, position, normal, root, scale, yaw, seed) - root);

    var out: VsOut;
    out.clip = globals.projection * globals.view * vec4<f32>(world, 1.0);
    // Near a cube edge the ground fades from the plane's into the seam blend
    // the plant was not seated on; it stands only where the plane's is whole.
    if (planetFeatureShare(root.xz) < 0.999) { out.clip = vec4<f32>(0.0, 0.0, -1.0, 1.0); }
    out.world = world;
    out.normal = yawRotate(normal, c, s);
    out.tangent = vec4<f32>(yawRotate(tangent.xyz, c, s), tangent.w);
    out.uv = uv;
    // The crown's own outward normal: radial from the trunk axis, tipped up
    // toward the top of the crown.
    let crown_mid = mix(model.crown_base, model.height, 0.45);
    let radial = vec3<f32>(position.x, (position.y - crown_mid) * 0.55, position.z);
    let radial_length = max(length(radial), 1e-3);
    out.crown_normal = yawRotate(normalize(radial / radial_length + vec3<f32>(0.0, 0.35, 0.0)), c, s);
    let reach = max(model.crown_radius * crownProfile(model, position.y), 0.08);
    let interior = 1.0 - clamp(length(position.xz) / reach, 0.0, 1.0);
    out.shade = vec3<f32>(interior, height01, max(position.y, 0.0) * scale);
    out.fade = fade;
    out.seed = seed;
    return out;
}

struct Gbuffer {
    @location(0) position: vec4<f32>,
    @location(1) normal: vec4<f32>,
    @location(2) albedo: vec4<f32>,
};

// Interleaved gradient noise: a stable screen-space dither for the LOD
// cross-fades and the far-edge dissolve.
fn ditherValue(pixel: vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(dot(floor(pixel), vec2<f32>(0.06711056, 0.00583715))));
}

@fragment
fn fs_main(in: VsOut, @builtin(front_facing) front: bool) -> Gbuffer {
    let colour = textureSample(base_color, material_sampler, in.uv) * material.base_color_factor;
    let detail = textureSample(detail_map, material_sampler, in.uv);
    let dither = ditherValue(in.clip.xy);
    if (in.fade > 0.0) {
        if (dither >= in.fade) {
            discard;
        }
    } else if (dither < -in.fade) {
        discard;
    }
    if (material.params.x > 0.0 && colour.a < material.params.x) {
        discard;
    }
    let kind = material.surface.x;

    var tangent_normal = vec3<f32>((detail.xy * 2.0 - 1.0) * material.params.y, 0.0);
    tangent_normal.z = sqrt(max(1.0 - dot(tangent_normal.xy, tangent_normal.xy), 0.0));
    let n = normalize(in.normal);
    let raw_tangent = in.tangent.xyz - n * dot(n, in.tangent.xyz);
    let t = raw_tangent / max(length(raw_tangent), 1e-5);
    let b = cross(n, t) * in.tangent.w;
    var world_normal = normalize(t * tangent_normal.x + b * tangent_normal.y + n * tangent_normal.z);
    world_normal = select(-world_normal, world_normal, front);

    var ao = mix(1.0, detail.a, material.params.w);
    var albedo = colour.rgb * material.surface.z;
    let variation = fract(in.seed * 4.17);
    if (kind > 0.5) {
        // Cards shade as one crown volume rather than a stack of flat
        // planes; billboards already carry a baked crown normal map.
        let crown_blend = select(0.6, 0.3, kind > 1.5);
        world_normal = normalize(mix(world_normal, in.crown_normal, crown_blend));
        // Foliage deep inside the crown and low in it sees little sky.
        ao *= mix(1.0, 0.42, in.shade.x * (0.85 - 0.45 * in.shade.y));
        ao *= mix(0.70, 1.0, smoothstep(0.0, 0.75, in.shade.y));
        // No two crowns are quite the same shade.
        let warmth = fract(in.seed * 7.31) - 0.5;
        albedo *= mix(0.84, 1.12, variation) * vec3<f32>(1.0 + warmth * 0.14, 1.0, 1.0 - warmth * 0.12);
    } else {
        // Bark darkens where the trunk meets the ground and under the crown.
        ao *= mix(0.55, 1.0, smoothstep(0.0, 1.6, in.shade.z));
        ao *= mix(0.78, 1.0, in.shade.y);
        albedo *= mix(0.9, 1.08, variation);
    }
    let roughness = clamp(detail.b * material.params.z, select(0.5, 0.62, kind > 0.5), 1.0);

    // As the terrain does, rebuild the view-space position from the world
    // position rather than interpolating it.
    let view_position = (globals.view * vec4<f32>(in.world, 1.0)).xyz;
    let view_normal = normalize((globals.view * vec4<f32>(world_normal, 0.0)).xyz);
    var out: Gbuffer;
    // The alpha's ten-thousandths carry the leaf translucency the composite
    // reads as its canopy mask (bark carries none: exactly 1.0).
    out.position = vec4<f32>(view_position, 1.0 + 0.0001 * material.surface.y);
    out.normal = vec4<f32>(view_normal * 0.5 + 0.5, roughness);
    out.albedo = vec4<f32>(sqrt(max(albedo, vec3<f32>(0.0)) * 0.4), clamp(ao, 0.0, 1.0));
    return out;
}

struct ShadowOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_shadow(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) tangent: vec4<f32>,
    @location(4) root: vec3<f32>,
    @location(5) scale: f32,
    @location(6) yaw: f32,
    @location(7) seed: f32,
    @location(8) fade: f32,
    @location(9) model_index: u32,
) -> ShadowOut {
    let world = planetPlaneToWorld(root)
        + (placeVertex(models[model_index], position, normal, root, scale, yaw, seed) - root);
    var out: ShadowOut;
    out.clip = cascade.view_projection * vec4<f32>(world, 1.0);
    if (planetFeatureShare(root.xz) < 0.999) { out.clip = vec4<f32>(0.0, 0.0, -1.0, 1.0); }
    out.uv = uv;
    return out;
}

// Depth only: leaves and needles cut out of their cards as they are in view.
@fragment
fn fs_shadow(in: ShadowOut) {
    if (material.params.x > 0.0) {
        let alpha = textureSample(base_color, material_sampler, in.uv).a * material.base_color_factor.a;
        if (alpha < material.params.x) {
            discard;
        }
    }
}

struct CanopyOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) local: vec2<f32>,
    @location(1) @interpolate(flat) density: f32,
};

// One quad per tree over the 512 m grass capture, drawn into the alpha of
// its ground-colour texture with multiplicative blending: the alpha left is
// the share of open sky over each texel, crowns that overlap shading more.
// Every streamed plant is an instance; all but the trees inside the capture
// collapse to a point off screen.
@vertex
fn vs_canopy(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> CanopyOut {
    var out: CanopyOut;
    out.clip = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    let plant = plants[instance];
    if (plant.layer > 1u) {
        return out;
    }
    let model = models[plant.model];
    let radius = max(model.crown_radius * plant.scale, 0.4);
    let half_span = 0.5 * frame.params.z;
    let offset = plant.position.xz - frame.params.xy;
    if (any(abs(offset) > vec2<f32>(half_span + radius))) {
        return out;
    }
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
        vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0));
    let corner = corners[vertex];
    let ground = offset + corner * radius;
    // The capture's projection: +X right, +Z up the texture's rows.
    out.clip = vec4<f32>(ground.x / half_span, -ground.y / half_span, 0.5, 1.0);
    out.local = corner;
    // A mature crown stops most of the light; saplings and poles less.
    out.density = select(0.82, 0.6, plant.layer == 1u);
    return out;
}

@fragment
fn fs_canopy(in: CanopyOut) -> @location(0) vec4<f32> {
    let r = length(in.local);
    if (r >= 1.0) {
        discard;
    }
    let cover = in.density * (1.0 - smoothstep(0.45, 1.0, r));
    return vec4<f32>(0.0, 0.0, 0.0, 1.0 - cover);
}
