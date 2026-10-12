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
    planet: vec4<f32>,          // radius, face half width, seam band, camera face (planet-functions.wgslinc)
    planet_camera: vec4<f32>,   // camera chart xy on its face, altitude, LOD reference altitude
    planet_east: vec4<f32>,     // camera east in planet axes, feature fade width
    planet_up: vec4<f32>,       // camera up in planet axes, atmosphere top altitude
    planet_north: vec4<f32>,    // camera north in planet axes
    celestial: vec4<f32>,       // sun and moon angular radii, sun's unhidden share, moon's sunlight
    stars: vec4<f32>,           // world to ecliptic axes as a quaternion: the stars' frame
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
// globals.camera_position (its terrain plane coordinates and altitude). At
// the camera it is the terrain plane mapped through the chart's derivatives
// (the plane itself at a face centre, stretched and sheared toward its
// edges); the planet's centre sits at (camera x, -radius, camera z). Plane points (x, altitude, z)
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

// The terrain plane point (x, z) under a world position, on the camera's
// face: where planetPlaneToWorld put it, read back through the chart. Maps
// keyed on plane coordinates (the lighting heightfield) are read through it:
// away from a face's centre a world offset is not the plane offset, the
// chart's stretch and shear lie between them.
fn planetWorldToPlane(world: vec3<f32>) -> vec2<f32>
{
    let face = planetCameraFace();
    let d = planetFromWorldAxes(world - planetCentreWorld());
    let along = dot(d, PLANET_FACE_AXIS[face]);
    let chart = vec2<f32>(atan2(dot(d, PLANET_FACE_EAST[face]), along),
                          atan2(dot(d, PLANET_FACE_NORTH[face]), along))
        *(globals.planet.y/PLANET_QUARTER_PI);
    return chart + PLANET_FACE_PLANE_OFFSET[face];
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

// A vector given as if `up` were the camera's up (a surface normal built
// from its slopes, say), turned the least way that takes the camera's up
// onto `up`: the same vector at a point of the planet away from the camera.
fn planetTiltUp(v: vec3<f32>, up: vec3<f32>) -> vec3<f32>
{
    let c = up.y;
    if (c < -0.999) { return vec3<f32>(v.x, -v.y, -v.z); }
    // (0, 1, 0) x up
    let k = vec3<f32>(up.z, 0.0, -up.x);
    return v*c + cross(k, v) + k*(dot(k, v)/(1.0 + c));
}

// The sun's elevation where a world point lies: the sine of its angle above
// the local horizon there, which the camera's own frame gives only for the
// camera's place.
fn planetSunElevationAt(world: vec3<f32>) -> f32
{
    return dot(-normalize(globals.sun_direction.xyz), planetUpAt(world));
}

// The day/night cycle's daylight curve (day_night.rs at_elevation) at a world
// point, so the night side of the planet is dark seen from the day side,
// darkened as the moon hides the sun (globals.celestial.z, the share it leaves).
// At the camera it is globals.atmosphere.x.
fn planetDaylightAt(world: vec3<f32>) -> f32
{
    return smoothstep(-0.18, 0.18, planetSunElevationAt(world))*sqrt(globals.celestial.z);
}

// The direct sunlight reaching a world point: the day/night cycle's colour
// and strength curves (day_night.rs at_elevation) for the sun's elevation
// there, times the sun's intensity and the share of its disc the moon leaves.
// At the camera it is the uniform globals.sun_colour.rgb*globals.settings_a.x.
fn planetSunlightAt(world: vec3<f32>) -> vec3<f32>
{
    let elevation = planetSunElevationAt(world);
    let colourMix = smoothstep(0.0, 0.55, elevation);
    let colour = vec3<f32>(1.0, 0.29 + 0.66*colourMix, 0.095 + 0.765*colourMix);
    return colour*globals.sun_colour.w*smoothstep(0.0, 0.12, elevation)*globals.celestial.z;
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
    // The clump's root on the curved ground; the blades stand upright in the
    // camera's frame, which within grass range is the ground's own.
    let world = planetPlaneToWorld(vec3<f32>(root.x, ground_height_slope_x.x, root.y)) + offset;
    let view = globals.view * vec4<f32>(world, 1.0);
    out.position = globals.projection * view;
    // Only where the plane's ground, which the habitat capture seated the
    // clump on, is drawn whole: not near a cube edge.
    if (planetFeatureShare(root) < 0.999) { out.position = vec4<f32>(0.0, 0.0, -1.0, 1.0); }
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
