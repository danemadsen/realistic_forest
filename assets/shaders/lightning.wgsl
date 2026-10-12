// The lightning channel: a cloud-to-ground bolt or a crawler drawn as glowing
// segments over the resolved scene. It runs after the water passes and before
// the rain, so falling streaks pass in front of it. The scene is tonemapped
// already, so the channel's light is added in display space: a white core
// that still saturates through heavy rain, wrapped in a blue-violet halo that
// stands in for the bloom of light scattered by the wet air around it.
struct GlobalUniforms {
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
    camera_position: vec4<f32>,
    sun_direction: vec4<f32>,
    viewport: vec4<f32>,
    params: vec4<f32>,
    settings_a: vec4<f32>,
    settings_b: vec4<f32>,
    sun_colour: vec4<f32>,
    moon_direction: vec4<f32>,
    atmosphere: vec4<f32>,
    raymarch: vec4<f32>,
    heightfield: vec4<f32>,
    clouds: vec4<f32>,
    cloud_layer: vec4<f32>,
    cloud_motion: vec4<f32>,
    weather: vec4<f32>,
    storm: vec4<f32>,
    lightning: vec4<f32>,
    lightning_meta: vec4<f32>,
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
@group(1) @binding(0) var scene_texture: texture_2d<f32>;
@group(1) @binding(1) var scene_position: texture_2d<f32>;

struct BoltUniforms {
    // Segment count, main-channel luminance, branch luminance, leader front.
    header: vec4<f32>,
    // Cloud base over the discharge, unused, 1 when the eye is above the
    // cloud deck, unused.
    cloud: vec4<f32>,
    // Three per segment: start xyz and width; end xyz and brightness; leader
    // arrival at start and end, branch order, unused.
    segments: array<vec4<f32>, 1536>,
};
@group(2) @binding(0) var<uniform> bolt: BoltUniforms;

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
                                       cloud_base + 600.0, planetAltitude(world_position));
    let storm_precipitation = smoothstep(0.74, 0.90, score + globals.storm.x)
                        *wet_gate*below_cloud;
    if (storm_precipitation <= 0.0) { return vec3<f32>(0.0); }
    let cold_altitude = planetAltitude(world_position)
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

fn viewToWorld(direction: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(dot(globals.view[0].xyz, direction),
                     dot(globals.view[1].xyz, direction),
                     dot(globals.view[2].xyz, direction));
}

// The angle one pixel row subtends at the centre of the view.
fn pixelAngle() -> f32 {
    return 2.0/(abs(globals.projection[1][1])*max(globals.viewport.y, 1.0));
}

// Mirrors weatherFogMultiplier in composite.wgsl.
fn boltFogMultiplier(severity: f32) -> f32 {
    if (severity < 1.0) { return mix(0.48, 1.0, severity); }
    if (severity < 2.0) { return mix(1.0, 2.4, severity - 1.0); }
    return mix(2.4, 250.0, severity - 2.0);
}

// Air and precipitation extinction near a point, as composite integrates it.
fn localExtinction(world_position: vec3<f32>) -> f32 {
    let precipitation = weatherPrecipitation(world_position);
    let severity = weatherPrecipitationSeverity(world_position.xz);
    let height_density = exp(-max(world_position.y - 20.0, 0.0)/450.0);
    return max(globals.params.x, 0.0)*boltFogMultiplier(severity)
           *(0.16 + 0.84*height_density)
           + precipitation.x*0.0021 + precipitation.y*0.0013 + precipitation.z*0.00035;
}

struct BlitVertex {
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vs_blit(@builtin(vertex_index) vertex: u32) -> BlitVertex {
    let corners = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    var out: BlitVertex;
    out.position = vec4<f32>(corners[vertex], 0.0, 1.0);
    return out;
}

@fragment
fn fs_blit(in: BlitVertex) -> @location(0) vec4<f32> {
    return textureLoad(scene_texture, vec2<i32>(in.position.xy), 0);
}

struct BoltVertex {
    @builtin(position) position: vec4<f32>,
    // Metres across the segment and along it from its start.
    @location(0) ribbon: vec2<f32>,
    @location(1) view_depth: f32,
    // Segment length, core radius and halo radius in metres, and luminance.
    @location(2) @interpolate(flat) shape: vec4<f32>,
    // Core and halo strength after the air, rain and cloud they cross.
    @location(3) @interpolate(flat) strength: vec2<f32>,
};

fn hiddenBolt() -> BoltVertex {
    var out: BoltVertex;
    out.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    out.ribbon = vec2<f32>(0.0);
    out.view_depth = 0.0;
    out.shape = vec4<f32>(1.0, 1.0, 1.0, 0.0);
    out.strength = vec2<f32>(0.0);
    return out;
}

@vertex
fn vs_bolt(@builtin(vertex_index) vertex: u32,
           @builtin(instance_index) instance: u32) -> BoltVertex {
    if (instance >= u32(bolt.header.x)) { return hiddenBolt(); }
    let head = bolt.segments[instance*3u];
    let tail = bolt.segments[instance*3u + 1u];
    let timing = bolt.segments[instance*3u + 2u];
    // The main channel and the branches burn at different times.
    let luminance = select(bolt.header.z, bolt.header.y, timing.z < 0.5)*tail.w;
    let leader = bolt.header.w;
    if (luminance <= 0.001 || timing.x >= leader) { return hiddenBolt(); }
    // The channel is built in terrain plane coordinates; bend its ends onto
    // the planet around the camera like the ground it strikes.
    let start = planetPlaneToWorld(head.xyz);
    var end = planetPlaneToWorld(tail.xyz);
    // The leader's tip is partway down this segment.
    if (timing.y > leader) {
        end = mix(start, end, clamp((leader - timing.x)/max(timing.y - timing.x, 0.00001), 0.0, 1.0));
    }
    let camera = globals.camera_position.xyz;
    let middle = (start + end)*0.5;
    let distance = max(length(middle - camera), 1.0);
    let pixel_metres = distance*pixelAngle();

    // Inside the cloud the channel is only a diffuse glow; from above the
    // deck, whatever lies beneath the base is hidden.
    let base = bolt.cloud.x;
    let middle_altitude = planetAltitude(middle);
    let inside = smoothstep(base - 30.0, base + 150.0, middle_altitude);
    let under_deck = bolt.cloud.z*(1.0 - smoothstep(base - 60.0, base + 60.0, middle_altitude));
    // Air and rain between the eye and the channel. The channel is some ten
    // thousand times brighter than a storm sky, so its core still saturates
    // through a downpour, while forward scattering blurs it into a broader,
    // softer line wrapped in a wider glow.
    let ground_level = vec3<f32>(middle.x, max(camera.y, 0.0), middle.z);
    let optical_depth = 0.5*(localExtinction(camera) + localExtinction(ground_level))*distance;
    let transmittance = exp(-optical_depth);
    let visible = (1.0 - under_deck*0.95);
    let core_strength = clamp(transmittance*3000.0, 0.0, 1.0)*(1.0 - inside)*visible;
    let halo_strength = (0.45*clamp(transmittance*8.0, 0.0, 1.0) + 0.3*(1.0 - transmittance))
                        *mix(1.0, 0.6, inside)*visible;
    let width = head.w;
    // A channel a few centimetres across glows about a metre wide; the core
    // keeps at least a pixel and a half so a distant bolt stays a crisp line.
    let core = max(0.6*width, 0.75*pixel_metres)*(1.0 + 0.5*min(optical_depth, 8.0));
    let halo = max(6.0*width, 8.0*pixel_metres)
               *(1.0 + 0.5*min(optical_depth, 3.0))*(1.0 + 2.0*inside);

    let axis = end - start;
    let segment_length = length(axis);
    let along_axis = axis/max(segment_length, 0.0001);
    var side = cross(along_axis, normalize(camera - middle));
    if (length(side) < 0.001) {
        side = vec3<f32>(globals.view[0].x, globals.view[1].x, globals.view[2].x);
    }
    side = normalize(side);
    let half_width = halo*1.6;
    let corners = array<vec2<f32>, 4>(
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 1.0));
    let corner = corners[vertex];
    let across = (corner.x*2.0 - 1.0)*half_width;
    let along = mix(-half_width, segment_length + half_width, corner.y);
    let world_position = start + along_axis*along + side*across;
    let view_position = globals.view*vec4<f32>(world_position, 1.0);
    if (view_position.z >= -0.5) { return hiddenBolt(); }
    let clip = globals.projection*view_position;
    if (!all(abs(clip) < vec4<f32>(1.0e30))) { return hiddenBolt(); }
    var out: BoltVertex;
    out.position = clip;
    out.ribbon = vec2<f32>(across, along);
    out.view_depth = -view_position.z;
    out.shape = vec4<f32>(segment_length, core, halo, luminance);
    // Neighbouring segments overlap their halos; scale each so a continuous
    // channel glows as one line rather than a string of beads.
    out.strength = vec2<f32>(core_strength,
                             halo_strength*clamp(segment_length/(1.77*halo), 0.05, 1.0));
    return out;
}

@fragment
fn fs_bolt(in: BoltVertex) -> @location(0) vec4<f32> {
    let surface = textureLoad(scene_position, vec2<i32>(in.position.xy), 0);
    // Terrain in front of the channel hides it.
    if (surface.a >= 0.5 && in.view_depth > -surface.z + 1.0) { discard; }
    let beyond = max(max(-in.ribbon.y, in.ribbon.y - in.shape.x), 0.0);
    let radial = length(vec2<f32>(in.ribbon.x, beyond));
    let core = exp(-2.0*pow(radial/in.shape.y, 2.0));
    let halo = exp(-1.5*pow(radial/in.shape.z, 2.0));
    let core_light = core*in.strength.x*in.shape.w;
    let halo_light = halo*in.strength.y*in.shape.w;
    let colour = vec3<f32>(1.0, 0.97, 1.0)*min(core_light*1.6, 1.0)
                 + vec3<f32>(0.5, 0.55, 1.0)*halo_light*0.55;
    return vec4<f32>(colour, 0.0);
}
