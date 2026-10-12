// Full-sphere cloud-only reflection probe, rendered before the atmosphere.
// Keep the canonical helper in cloud-functions.wgslinc synchronized; ABI and
// shared-source regression tests validate all copies.
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
                                      cloud_top + 350.0, planetAltitude(world_position));
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
// The layer is the shell between two spheres about the planet's centre
// (planet-functions.wgslinc), so a deck seen from below curves down to the
// horizon with the ground, and one seen from above wraps round the planet.
// The first pass through the shell along the ray is returned: from below,
// from where the ray leaves the inner sphere to where it leaves the outer;
// from inside or above, until it leaves the shell or dips into the inner
// sphere. An empty interval has its start past its end.
fn cloudInterval(origin: vec3<f32>, ray: vec3<f32>, maximum_distance: f32) -> vec2<f32> {
    let slab = cloudSlab();
    let outer = planetRaySphere(origin, ray, slab.y);
    if (outer.y <= 0.0) { return vec2<f32>(0.0); }
    var start = max(outer.x, 0.0);
    var end = outer.y;
    let inner = planetRaySphere(origin, ray, slab.x);
    if (inner.y > 0.0) {
        if (inner.x <= 0.0) {
            start = max(start, inner.y);
        } else {
            end = min(end, inner.x);
        }
    }
    return vec2<f32>(start, min(end, maximum_distance));
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
    let height = (planetAltitude(world_position) - cloud_base)/cloud_thickness;
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
        let height = clamp((planetAltitude(world_position) - local_layer.x)/local_layer.y, 0.0, 1.0);
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

@vertex
fn vs_main(@builtin(vertex_index) vertex: u32) -> @builtin(position) vec4<f32> {
    let corners = array<vec2<f32>, 3>(vec2<f32>(-1.0,-1.0),
        vec2<f32>(3.0,-1.0), vec2<f32>(-1.0,3.0));
    return vec4<f32>(corners[vertex], 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = position.xy / vec2<f32>(256.0, 128.0);
    let azimuth = (uv.x - 0.5)*2.0*CLOUD_PI;
    let elevation = (0.5 - uv.y)*CLOUD_PI;
    let ray = vec3<f32>(cos(azimuth)*cos(elevation), sin(elevation),
                        sin(azimuth)*cos(elevation));
    // Cloud parallax from a water pixel is small relative to the cloud layer;
    // capturing at sea level also works while the player flies above the clouds.
    let origin = vec3<f32>(globals.camera_position.x, 0.0, globals.camera_position.z);
    let clouds = marchClouds(origin, ray, globals.cloud_motion.w, min(globals.cloud_layer.w, 1.0));
    return vec4<f32>(clouds.scattering, clouds.transmittance);
}
