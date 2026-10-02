// Rain and snow around the camera: world-anchored streaks and flakes, rain
// layers that carry a downpour out past a hundred metres, and the crowns
// thrown up where drops land. Drawn after the water pass so precipitation
// appears in front of the resolved scene and before FXAA so subpixel streaks
// receive the same edge treatment as terrain. The scene is already tonemapped
// here, so a drop takes its colour from the encoded scene around it.
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
};
@group(0) @binding(0) var<uniform> globals: GlobalUniforms;
@group(1) @binding(0) var scene_texture: texture_2d<f32>;
@group(1) @binding(1) var scene_position: texture_2d<f32>;
@group(1) @binding(2) var scene_normal: texture_2d<f32>;

struct PrecipitationFrame {
    velocity: vec4<f32>, // xy near-ground XZ wind, z ocean enabled, w local storm gust
    // Particle lattices: cells per side, vertical layers, cell size in metres,
    // and one past the lattice's last instance index.
    rain_near: vec4<f32>,
    rain_far: vec4<f32>,
    snow: vec4<f32>,
    // Ground splashes: screen cells across and down, unused, last instance + 1.
    splash: vec4<f32>,
};
@group(2) @binding(0) var<uniform> frame: PrecipitationFrame;

const PI: f32 = 3.14159265;
// A cinematic 1/30 s exposure draws each falling drop as a streak.
const SHUTTER_SECONDS: f32 = 0.0333;
// Crowns visible at one instant per square metre of ground in a downpour, a
// small fraction of the ~1600 impacts each second.
const SPLASHES_PER_SQUARE_METRE: f32 = 14.0;
const SPLASH_CYCLE_SECONDS: f32 = 0.16;
// Rain layers double in radius from the first; streaks on them fall at the
// typical speed of a 2.5 mm drop.
const RAIN_LAYER_RADIUS: f32 = 14.0;
const RAIN_LAYER_FALL_SPEED: f32 = 7.3;

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

fn hash32(input: u32) -> u32 {
    var value = input;
    value ^= value >> 16u;
    value *= 0x7feb352du;
    value ^= value >> 15u;
    value *= 0x846ca68bu;
    value ^= value >> 16u;
    return value;
}

fn cellRandom(cell: vec3<i32>, salt: u32) -> f32 {
    let bits = vec3<u32>(cell);
    let value = hash32(bits.x*0x8da6b343u ^ bits.y*0xd8163841u
                        ^ bits.z*0xcb1ab31fu ^ salt);
    return f32(value & 0x00ffffffu)/16777215.0;
}

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

// The angle one pixel row subtends at the centre of the view.
fn pixelAngle() -> f32 {
    return 2.0/(abs(globals.projection[1][1])*max(globals.viewport.y, 1.0));
}

fn acesFilm(x: vec3<f32>) -> vec3<f32> {
    return clamp((x*(2.51*x + 0.03))/(x*(2.43*x + 0.59) + 0.14),
                  vec3<f32>(0.0), vec3<f32>(1.0));
}

// The composite's display encoding: exposure, ACES, then gamma.
fn encodeRadiance(radiance: vec3<f32>) -> vec3<f32> {
    return pow(acesFilm(max(radiance, vec3<f32>(0.0))*globals.params.z),
               vec3<f32>(1.0/2.2));
}

// Mirrors weatherFogMultiplier in composite.wgsl.
fn rainFogMultiplier(severity: f32) -> f32 {
    if (severity < 1.0) { return mix(0.48, 1.0, severity); }
    if (severity < 2.0) { return mix(1.0, 2.4, severity - 1.0); }
    return mix(2.4, 250.0, severity - 2.0);
}

// Air and precipitation extinction near a point, as composite integrates it.
fn localExtinction(world_position: vec3<f32>, precipitation: vec4<f32>) -> f32 {
    let severity = weatherPrecipitationSeverity(world_position.xz);
    let height_density = exp(-max(world_position.y - 20.0, 0.0)/450.0);
    return max(globals.params.x, 0.0)*rainFogMultiplier(severity)
           *(0.16 + 0.84*height_density)
           + precipitation.x*0.0021 + precipitation.y*0.0013 + precipitation.z*0.00035;
}

fn lightningIllumination(world_position: vec3<f32>) -> vec3<f32> {
    if (globals.lightning.w <= 0.001) { return vec3<f32>(0.0); }
    let source = vec3<f32>(globals.lightning.x,
                           mix(globals.lightning.y, globals.lightning_meta.z, 0.6),
                           globals.lightning.z);
    let range = length(world_position - source);
    return vec3<f32>(0.73, 0.85, 1.0)*globals.lightning.w/(1.0 + pow(range/2200.0, 2.0));
}

fn clampedScene(pixel: vec2<i32>) -> vec3<f32> {
    let size = vec2<i32>(textureDimensions(scene_texture, 0u));
    return textureLoad(scene_texture, clamp(pixel, vec2<i32>(0), size - vec2<i32>(1)), 0).rgb;
}

// Mirrors weatherSkyCover in composite.wgsl.
fn rainSkyCover(severity: f32) -> f32 {
    if (severity <= 1.0) { return 0.0; }
    if (severity <= 2.0) { return 0.8*(severity - 1.0); }
    return mix(0.8, 1.0, severity - 2.0);
}

// Mirrors weatherFogAmbient in composite.wgsl: the diffuse sky light that a
// drop always sees, even when the camera looks down at the ground.
fn rainSkyAmbient(world_position: vec3<f32>, thunder: f32) -> vec3<f32> {
    let sky_cover = rainSkyCover(weatherPrecipitationSeverity(world_position.xz));
    let overcast = smoothstep(0.05, 0.9, sky_cover);
    let whiteout = smoothstep(0.78, 1.0, sky_cover);
    let night_ambient = mix(vec3<f32>(0.012, 0.019, 0.037),
                            vec3<f32>(0.021, 0.024, 0.028), overcast);
    let day_ambient = mix(mix(vec3<f32>(0.32, 0.44, 0.59),
                              vec3<f32>(0.40, 0.43, 0.43), overcast),
                          vec3<f32>(0.67, 0.69, 0.68), whiteout);
    let dark_night = mix(night_ambient, vec3<f32>(0.007, 0.010, 0.017), thunder);
    let dark_day = mix(day_ambient, vec3<f32>(0.10, 0.125, 0.14), thunder);
    return mix(dark_night, dark_day, clamp(globals.atmosphere.x, 0.0, 1.0));
}

// What a raindrop at this pixel shows. A drop is a wide-angle lens with a
// field of about 164 degrees around the view direction, so it always holds
// both sky and ground: the share of sky follows the view elevation. That
// reads lighter than dark forest and a little darker than open sky. The top
// screen row is usually the real sky; the diffuse sky light is its floor
// when the camera looks down. A faint daylight reflection keeps streaks
// visible against a uniform background, and a flash makes drops glitter.
fn dropColour(pixel: vec2<i32>, world_position: vec3<f32>, thunder: f32) -> vec3<f32> {
    let reach = i32(f32(textureDimensions(scene_texture, 0u).y)*0.3);
    let sky = max(max(clampedScene(vec2<i32>(pixel.x, 0)),
                      clampedScene(pixel - vec2<i32>(0, reach))),
                  encodeRadiance(rainSkyAmbient(world_position, thunder))*0.85);
    let ground = clampedScene(pixel + vec2<i32>(0, reach));
    let view = normalize(world_position - globals.camera_position.xyz);
    let elevation = asin(clamp(view.y, -1.0, 1.0));
    let sky_share = clamp((elevation + 1.43)/2.86, 0.0, 1.0);
    let daylight = clamp(globals.atmosphere.x, 0.0, 1.0);
    return mix(ground, sky, sky_share)*0.97
           + vec3<f32>(0.05*daylight)
           + encodeRadiance(lightningIllumination(world_position)*0.1);
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

// One rain layer: a cylinder of streaks centred on the camera. Each cell
// holds at most one streak and keeps the same size on screen at every radius,
// so farther layers read as finer, slower rain behind the nearer ones.
// Returns the streak coverage and the distance to the layer along the ray.
fn rainLayer(ray: vec3<f32>, layer: u32, surface_distance: f32,
             wind_velocity: vec2<f32>) -> vec2<f32> {
    let horizontal = length(ray.xz);
    if (horizontal < 0.15) { return vec2<f32>(0.0); }
    let radius = RAIN_LAYER_RADIUS*exp2(f32(layer));
    let distance = radius/horizontal;
    // Terrain and foliage in front of the layer hide it.
    let visible = clamp((surface_distance - distance)/(0.12*radius), 0.0, 1.0);
    if (visible <= 0.0) { return vec2<f32>(0.0, distance); }
    let pixel_angle = pixelAngle();
    let pixel_metres = distance*pixel_angle;
    let cell_width = radius*pixel_angle*22.0;
    let streak_length = max(RAIN_LAYER_FALL_SPEED*SHUTTER_SECONDS,
                            15.0*radius*pixel_angle);
    let cell_height = streak_length*2.4;
    let theta = atan2(ray.z, ray.x);
    let around = max(round(2.0*PI*radius/cell_width), 1.0);
    let tangent = vec2<f32>(-sin(theta), cos(theta));
    // A drop keeps this crosswind coordinate along its wind-slanted path, so
    // the streaks lean with the wind while they fall. The shear varies around
    // the cylinder, so it acts on height above the eye: on absolute height it
    // would smear cells sideways wherever the crosswind changes.
    let rise = ray.y*distance;
    let shear = dot(wind_velocity, tangent)/RAIN_LAYER_FALL_SPEED;
    let a = theta/(2.0*PI)*around + shear*rise/cell_width;
    let b = (globals.camera_position.y + rise
             + globals.storm.z*RAIN_LAYER_FALL_SPEED)/cell_height;
    let cell_a = floor(a);
    // Wrap the column index so the cylinder closes without a seam.
    let column = cell_a - around*floor(cell_a/around);
    let cell = vec3<i32>(i32(column), i32(floor(b)), i32(layer));
    let presence = cellRandom(cell, 223u);
    if (presence > 0.72) { return vec2<f32>(0.0, distance); }
    let half_length = 0.5*streak_length/cell_height;
    let centre = vec2<f32>(0.15 + 0.7*cellRandom(cell, 227u),
                           mix(half_length, 1.0 - half_length, cellRandom(cell, 229u)));
    let across_pixels = abs(fract(a) - centre.x)*cell_width/max(pixel_metres, 0.0001);
    let half_width = mix(1.0, 0.7, f32(layer)/3.0);
    let line = 1.0 - smoothstep(0.35*half_width, half_width, across_pixels);
    let along = abs(fract(b) - centre.y)/half_length;
    let ends = 1.0 - smoothstep(0.55, 1.0, along);
    if (line*ends <= 0.0) { return vec2<f32>(0.0, distance); }
    // Only pixels on a candidate streak pay for the weather field.
    let rain = weatherPrecipitation(globals.camera_position.xyz + ray*distance).x;
    let present = step(presence, 0.72*pow(rain, 0.7));
    return vec2<f32>(line*ends*visible*present, distance);
}

@fragment
fn fs_rain_layers(in: BlitVertex) -> @location(0) vec4<f32> {
    let pixel = vec2<i32>(in.position.xy);
    var colour = textureLoad(scene_texture, pixel, 0).rgb;
    let camera = globals.camera_position.xyz;
    let surface = textureLoad(scene_position, pixel, 0);
    let surface_distance = select(length(surface.xyz), 1.0e7, surface.a < 0.5);
    let ray = worldViewRay(in.position.xy*globals.viewport.zw);
    // Seen from directly below, streaks foreshorten into dots.
    let elevation_fade = 1.0 - smoothstep(0.55, 0.85, abs(ray.y));
    if (elevation_fade <= 0.0) { return vec4<f32>(colour, 1.0); }
    let gust = clamp(frame.velocity.w, 0.0, 1.0);
    let wind_velocity = frame.velocity.xy*(1.0 + 0.55*gust);
    let eye_weather = weatherPrecipitation(camera);
    let extinction = localExtinction(camera, eye_weather);
    let streak_colour = dropColour(pixel, camera + ray*RAIN_LAYER_RADIUS, eye_weather.z);
    // Farthest first, so nearer layers composite over farther ones.
    for (var i = 0u; i < 4u; i += 1u) {
        let layer = 3u - i;
        let streak = rainLayer(ray, layer, surface_distance, wind_velocity);
        if (streak.x <= 0.0) { continue; }
        let alpha = streak.x*exp(-extinction*streak.y)*elevation_fade
                    *mix(0.55, 0.32, f32(layer)/3.0);
        colour = mix(colour, streak_colour, alpha);
    }
    return vec4<f32>(colour, 1.0);
}

struct ParticleVertex {
    @builtin(position) position: vec4<f32>,
    @location(0) local: vec2<f32>,
    @location(1) view_depth: f32,
    @location(2) opacity: f32,
    // 0 rain streak, 1 snow flake, 2 ground splash.
    @location(3) @interpolate(flat) kind: u32,
    // Rain: screen-space normal to the streak, specular seed, drop variation.
    // Splash: ground facing, age fraction, shape seed, half size in metres.
    @location(4) @interpolate(flat) shading: vec4<f32>,
    @location(5) @interpolate(flat) colour: vec3<f32>,
};

// A vertex that is not finite can stall the rasteriser badly enough to lose
// the whole frame, so any such particle is hidden instead.
fn finiteClip(clip: vec4<f32>) -> bool {
    return all(abs(clip) < vec4<f32>(1.0e30));
}

fn hiddenParticle(kind: u32) -> ParticleVertex {
    var out: ParticleVertex;
    out.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    out.local = vec2<f32>(0.0);
    out.view_depth = 0.0;
    out.opacity = 0.0;
    out.kind = kind;
    out.shading = vec4<f32>(0.0);
    out.colour = vec3<f32>(0.0);
    return out;
}

// The fraction of a lattice's volume a camera-relative point lies inside,
// fading over the outer fifth so recycled cells never pop at the boundary.
fn latticeFade(lattice: vec4<f32>, offset: vec3<f32>) -> f32 {
    let half_width = lattice.x*lattice.z*0.5;
    let half_height = lattice.y*lattice.z*0.5;
    return (1.0 - smoothstep(0.74*half_width, 0.94*half_width,
                             max(abs(offset.x), abs(offset.z))))
           *(1.0 - smoothstep(0.74*half_height, 0.94*half_height, abs(offset.y)));
}

fn screenPixel(clip: vec4<f32>) -> vec2<i32> {
    let ndc = clip.xy/clip.w;
    return vec2<i32>(vec2<f32>((ndc.x + 1.0)*0.5, (1.0 - ndc.y)*0.5)*globals.viewport.xy);
}

// A crown sprite where a drop lands. Each screen cell picks one random pixel
// per cycle; the chance of a crown there follows the ground area the cell
// covers, so splashes keep a constant density per square metre.
fn splashParticle(vertex: u32, index: u32) -> ParticleVertex {
    let cells = vec2<u32>(u32(frame.splash.x), u32(frame.splash.y));
    let cell = vec2<i32>(i32(index % cells.x), i32(index/cells.x));
    let cycles = globals.storm.z/SPLASH_CYCLE_SECONDS
                 + cellRandom(vec3<i32>(cell, -7), 401u);
    let age = fract(cycles);
    let seed = vec3<i32>(cell, i32(floor(cycles)));
    let size = vec2<i32>(textureDimensions(scene_position, 0u));
    let uv = (vec2<f32>(cell) + vec2<f32>(cellRandom(seed, 409u), cellRandom(seed, 419u)))
             /vec2<f32>(cells);
    let pixel = clamp(vec2<i32>(uv*vec2<f32>(size)), vec2<i32>(0), size - vec2<i32>(1));
    let surface = textureLoad(scene_position, pixel, 0);
    if (surface.a < 0.5) { return hiddenParticle(2u); }
    let view_distance = length(surface.xyz);
    if (!(view_distance >= 0.5 && view_distance <= 14.0)) { return hiddenParticle(2u); }
    // G-buffer alpha stores snow to hundredths and grass in the residue.
    let snow_mask = clamp(floor((surface.a - 1.0)*100.0 + 0.5)*0.01, 0.0, 1.0);
    let grass_mask = clamp((surface.a - 1.0 - snow_mask)*10000.0, 0.0, 1.0);
    // Drops also burst on grass blades, whose own normals point anywhere;
    // other surfaces must face the sky.
    var normal_world = vec3<f32>(0.0, 1.0, 0.0);
    if (grass_mask < 0.5) {
        let packed_normal = textureLoad(scene_normal, pixel, 0).xyz*2.0 - 1.0;
        // An unwritten texel decodes to a zero vector; normalising it would
        // give NaN, which passes every comparison below.
        if (dot(packed_normal, packed_normal) < 0.25) { return hiddenParticle(2u); }
        normal_world = normalize(viewToWorld(normalize(packed_normal)));
        if (!(normal_world.y >= 0.55)) { return hiddenParticle(2u); }
    }
    let camera = globals.camera_position.xyz;
    let ground = camera + viewToWorld(surface.xyz);
    // The ocean draws its own impact rings.
    if (frame.velocity.z > 0.5 && ground.y < 0.05) { return hiddenParticle(2u); }
    let precipitation = weatherPrecipitation(ground);
    let rain = precipitation.x;
    if (rain <= 0.01) { return hiddenParticle(2u); }
    let ray = normalize(ground - camera);
    let facing = clamp(abs(dot(normal_world, ray)), 0.12, 1.0);
    let pixel_metres = view_distance*pixelAngle();
    let cell_area = f32(size.x*size.y)/f32(cells.x*cells.y)
                    *pixel_metres*pixel_metres/facing;
    let chance = cell_area*SPLASHES_PER_SQUARE_METRE*rain*mix(1.0, 0.5, grass_mask);
    let half_size = 0.1;
    if (cellRandom(seed, 431u) >= chance || pixel_metres > half_size) {
        return hiddenParticle(2u);
    }
    let corners = array<vec2<f32>, 4>(
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 1.0));
    let local = corners[vertex];
    let signed = local*2.0 - vec2<f32>(1.0);
    let camera_right = normalize(vec3<f32>(globals.view[0].x, globals.view[1].x, globals.view[2].x));
    let camera_up = normalize(vec3<f32>(globals.view[0].y, globals.view[1].y, globals.view[2].y));
    let world_position = ground + normal_world*0.003
                         + (camera_right*signed.x + camera_up*signed.y)*half_size;
    let view_position = globals.view*vec4<f32>(world_position, 1.0);
    let clip = globals.projection*view_position;
    if (view_position.z >= -0.1 || !finiteClip(clip)) { return hiddenParticle(2u); }
    var out: ParticleVertex;
    out.position = clip;
    out.local = local;
    out.view_depth = -view_position.z;
    // Beyond about ten metres a real splash is too small and brief to see.
    out.opacity = (1.0 - smoothstep(6.0, 14.0, view_distance))*0.6;
    out.kind = 2u;
    out.shading = vec4<f32>(facing, age, cellRandom(seed, 433u), half_size);
    out.colour = dropColour(pixel, ground, precipitation.z);
    return out;
}

@vertex
fn vs_particle(@builtin(vertex_index) vertex: u32,
               @builtin(instance_index) instance: u32) -> ParticleVertex {
    let near_end = u32(frame.rain_near.w);
    let far_end = u32(frame.rain_far.w);
    let snow_end = u32(frame.snow.w);
    if (instance >= snow_end) {
        return splashParticle(vertex, instance - snow_end);
    }
    let snow = instance >= far_end;
    let far = !snow && instance >= near_end;
    var lattice = frame.rain_near;
    var start = 0u;
    if (snow) {
        lattice = frame.snow;
        start = far_end;
    } else if (far) {
        lattice = frame.rain_far;
        start = near_end;
    }
    let lattice_salt = select(select(0u, 0x632be5abu, far), 0x9e3779b9u, snow);
    let kind = select(0u, 1u, snow);
    let index = instance - start;
    let grid = u32(lattice.x);
    let layers = u32(lattice.y);
    let cell_size = lattice.z;
    let x = index % grid;
    let z = (index / grid) % grid;
    let y = index / (grid*grid);
    let camera = globals.camera_position.xyz;
    let wind_shift = globals.weather.xy*select(0.22, 0.35, snow);
    let base_x = i32(floor((camera.x - wind_shift.x)/cell_size)) - i32(grid/2u);
    let base_z = i32(floor((camera.z - wind_shift.y)/cell_size)) - i32(grid/2u);
    let base_y = i32(floor(camera.y/cell_size)) - i32(layers/2u);
    let cell_x = base_x + i32(x);
    let cell_z = base_z + i32(z);
    let cell_y = base_y + i32(y);
    // Variation belongs to a world column, so adjacent vertical cells keep
    // the same velocity when the falling lattice recycles.
    let column = vec3<i32>(cell_x, 0, cell_z);
    // The visible tail of a heavy-rain drop spectrum, 1.5-4.5 mm. Atlas et
    // al.'s fit gives each its terminal velocity of roughly 6-9 m/s.
    let diameter_mm = 1.5 + 3.0*pow(cellRandom(column, 181u ^ lattice_salt), 1.4);
    let fall_speed = select(9.65 - 10.3*exp(-0.6*diameter_mm), 1.8, snow);
    let time_cells = globals.storm.z*fall_speed/cell_size;
    let cycle = floor(time_cells);
    let phase = fract(time_cells);
    // As the phase wraps, the particle in the next lattice cell takes this
    // instance's place. The *set* of world particles therefore falls without
    // a visible periodic jump, even when the camera moves between cells.
    let source_cell = vec3<i32>(cell_x, cell_y + i32(cycle), cell_z);
    let jitter = vec3<f32>(cellRandom(source_cell, 19u ^ lattice_salt),
                           cellRandom(source_cell, 43u ^ lattice_salt),
                           cellRandom(source_cell, 71u ^ lattice_salt));
    let gust = clamp(frame.velocity.w, 0.0, 1.0);
    // Storm gusts sweep through the lattice as moving pulses of wind.
    let gust_pulse = 1.0 + gust*(0.55 + 0.45*sin(globals.storm.z*0.9
        + f32(cell_x)*cell_size*0.045 + f32(cell_z)*cell_size*0.03));
    let wind_velocity = frame.velocity.xy*gust_pulse;
    let gust_amplitude = clamp(length(wind_velocity)/5.0, 0.15, 1.6);
    let gust_sway = vec2<f32>(
        sin(globals.storm.z*1.7 + f32(cell_x)*0.7 + f32(cell_z)*1.1),
        cos(globals.storm.z*1.4 + f32(cell_x)*1.2 - f32(cell_z)*0.6))
        *gust_amplitude*select(0.055, 0.12, snow);
    let center = vec3<f32>((f32(cell_x) + jitter.x)*cell_size + wind_shift.x + gust_sway.x,
                           (f32(cell_y) + jitter.y - phase)*cell_size,
                           (f32(cell_z) + jitter.z)*cell_size + wind_shift.y + gust_sway.y);
    let offset = center - camera;
    var coverage = latticeFade(lattice, offset);
    // The far lattice fills in only where the dense near lattice fades out.
    if (far) { coverage *= 1.0 - latticeFade(frame.rain_near, offset); }
    if (coverage <= 0.001 || (frame.velocity.z > 0.5 && center.y < 0.0)) {
        return hiddenParticle(kind);
    }
    let center_view = globals.view*vec4<f32>(center, 1.0);
    if (center_view.z >= -0.15) { return hiddenParticle(kind); }
    let distance = length(offset);
    let center_clip = globals.projection*center_view;
    // Most cells lie outside the view frustum. Cull them before the more
    // expensive world-space weather field is evaluated.
    if (distance > 3.0 && any(abs(center_clip.xy/center_clip.w) > vec2<f32>(1.15))) {
        return hiddenParticle(kind);
    }
    let precipitation = weatherPrecipitation(center);
    let intensity = select(precipitation.x, precipitation.y, snow);
    // Moderate rain shows proportionally more streaks than its rate implies,
    // since the eye picks out the largest drops first.
    let spawn = select(0.96*pow(intensity, 0.7), 0.96*intensity, snow);
    if (intensity <= 0.002 || cellRandom(source_cell, 109u ^ lattice_salt) >= spawn) {
        return hiddenParticle(kind);
    }
    let near_fade = select(smoothstep(0.35, 1.2, distance),
                           smoothstep(0.4, 1.7, distance), snow);

    let corners = array<vec2<f32>, 4>(
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 1.0));
    let local = corners[vertex];
    let signed = local*2.0 - vec2<f32>(1.0);
    let camera_right = normalize(vec3<f32>(globals.view[0].x, globals.view[1].x, globals.view[2].x));
    let camera_up = normalize(vec3<f32>(globals.view[0].y, globals.view[1].y, globals.view[2].y));
    var world_position = center;
    var shading = vec4<f32>(0.0);
    var colour = vec3<f32>(0.0);
    var opacity = coverage*near_fade*sqrt(intensity);
    if (snow) {
        let size = 0.023 + 0.027*cellRandom(source_cell, 137u ^ lattice_salt);
        world_position += camera_right*signed.x*size + camera_up*signed.y*size;
    } else {
        let velocity_variation = 0.78 + 0.44*cellRandom(column, 197u ^ lattice_salt);
        let velocity = vec3<f32>(wind_velocity.x*velocity_variation,
                                 -fall_speed,
                                 wind_velocity.y*velocity_variation);
        let speed = length(velocity);
        let fall_direction = velocity/speed;
        let to_eye = normalize(camera - center);
        var side = cross(fall_direction, to_eye);
        if (length(side) < 0.05) { side = camera_right; }
        side = normalize(side);
        let size_seed = cellRandom(source_cell, 149u ^ lattice_salt);
        let pixels_per_metre = abs(globals.projection[1].y)*globals.viewport.y
                               /max(-center_view.z*2.0, 0.2);
        // Millimetre drops keep a pixel-wide footprint so thin streaks fade
        // smoothly instead of flickering at middle distance.
        let width = max(diameter_mm*0.001*mix(0.8, 1.2, size_seed),
                        select(0.9, 1.15, far)/max(pixels_per_metre, 1.0));
        // The drop's path during the exposure. Distant streaks keep a
        // readable minimum length instead of collapsing to shimmering dots.
        let streak = max(speed*SHUTTER_SECONDS
                         *mix(0.75, 1.15, cellRandom(source_cell, 163u ^ lattice_salt)),
                         select(6.0, 14.0, far)/max(pixels_per_metre, 1.0));
        world_position += side*signed.x*width*0.5 + fall_direction*signed.y*streak*0.5;

        let side_clip = globals.projection*(globals.view*vec4<f32>(center + side*0.1, 1.0));
        let screen_delta = (side_clip.xy/side_clip.w - center_clip.xy/center_clip.w)
                           *vec2<f32>(globals.viewport.x, -globals.viewport.y);
        var screen_side = vec2<f32>(1.0, 0.0);
        if (length(screen_delta) > 0.0001) { screen_side = normalize(screen_delta); }
        let sun_towards_drop = normalize(-globals.sun_direction.xyz);
        var half_direction = sun_towards_drop + to_eye;
        if (length(half_direction) < 0.01) { half_direction = camera_right; }
        let half_vector = normalize(half_direction);
        let alignment = dot(half_vector, fall_direction);
        let specular_angle = sqrt(max(0.0, 1.0 - alignment*alignment));
        let glint_seed = pow(cellRandom(source_cell, 211u ^ lattice_salt), 20.0)*specular_angle;
        shading = vec4<f32>(screen_side, glint_seed, size_seed);
        colour = dropColour(screenPixel(center_clip), center, precipitation.z);
        opacity = coverage*near_fade*mix(0.6, 1.0, intensity)*select(0.85, 0.55, far)
                  *mix(1.0, 0.65, smoothstep(4.0, 40.0, distance));
    }
    let view_position = globals.view*vec4<f32>(world_position, 1.0);
    let clip = globals.projection*view_position;
    if (view_position.z >= -0.1 || !finiteClip(clip)) { return hiddenParticle(kind); }
    var out: ParticleVertex;
    out.position = clip;
    out.local = local;
    out.view_depth = -view_position.z;
    out.opacity = opacity;
    out.kind = kind;
    out.shading = shading;
    out.colour = colour;
    return out;
}

@fragment
fn fs_particle(in: ParticleVertex) -> @location(0) vec4<f32> {
    if (in.opacity <= 0.001) { discard; }
    let pixel = vec2<i32>(in.position.xy);
    let surface = textureLoad(scene_position, pixel, 0);
    // The G-buffer stores view-space position. Sky has alpha < 0.5; for an
    // opaque surface, a particle farther along the view axis is hidden.
    if (surface.a >= 0.5 && in.view_depth > -surface.z - 0.08) { discard; }
    let uv = in.local;
    let day = clamp(globals.atmosphere.x, 0.0, 1.0);
    let strike_distance = distance(globals.lightning.xz, globals.camera_position.xz);
    let flash = clamp(globals.lightning.w
                      /(1.0 + strike_distance*strike_distance/9000000.0), 0.0, 1.0);
    if (in.kind == 0u) {
        let background = textureLoad(scene_texture, pixel, 0).rgb;
        let shift = select(1.0, 2.0, in.view_depth < 5.0);
        let refracted = clampedScene(pixel + vec2<i32>(round(in.shading.xy*shift)));

        // A drop shows the scene around it (dropColour) and slightly
        // displaces what lies directly behind. Only a small stochastic
        // fraction catches a sharp sun or lightning highlight.
        let sky_cover = clamp(globals.atmosphere.z, 0.0, 1.0);
        let sun_visibility = smoothstep(-0.04, 0.22, -globals.sun_direction.y)
                             *(1.0 - sky_cover)*(1.0 - sky_cover);
        // Uniform motion blur along the streak with soft ends; across it a
        // smooth line profile that stays antialiased at one pixel wide.
        let along = smoothstep(0.0, 0.16, uv.y)*(1.0 - smoothstep(0.84, 1.0, uv.y));
        let across = 1.0 - smoothstep(0.2, 1.0, abs(uv.x - 0.5)*2.0);
        let head = exp(-pow((uv.y - 0.8)/0.12, 2.0));
        let specular = in.shading.z*(0.10 + 0.64*sun_visibility + 0.85*flash)
                       *pow(head, 3.0);
        let colour = mix(mix(background, refracted, 0.7), in.colour, 0.8)
                     + vec3<f32>(0.55, 0.65, 0.78)*specular;
        let alpha = in.opacity*across*along*mix(0.75, 1.0, in.shading.w);
        return vec4<f32>(colour, alpha);
    }
    if (in.kind == 2u) {
        let facing = in.shading.x;
        let age = in.shading.y;
        let shape = in.shading.z;
        // Sprite coordinates in metres: x along the camera's right, y up.
        let q = (uv - vec2<f32>(0.5))*2.0*in.shading.w;
        let pixel_metres = max(in.view_depth*pixelAngle(), 0.0015);
        // The crown's rim spreads across the ground, foreshortened by the
        // viewing angle, and fades as its sheet breaks up.
        let radius = mix(0.015, 0.065, sqrt(age))*mix(0.7, 1.25, shape);
        let rim = length(vec2<f32>(q.x, q.y/facing));
        let ring = (1.0 - smoothstep(0.5*pixel_metres, max(0.004, 1.5*pixel_metres),
                                     abs(rim - radius)))
                   *(1.0 - age)*smoothstep(0.0, 0.08, age);
        // Secondary droplets leave the rim and fall back under gravity.
        let flight = age*SPLASH_CYCLE_SECONDS;
        var droplets = 0.0;
        for (var i = 0u; i < 6u; i += 1u) {
            let angle = (f32(i) + shape)*PI/3.0;
            let launch = 0.75 + 0.6*fract(shape*7.3 + f32(i)*0.37);
            let rise = max(launch*flight - 4.9*flight*flight, 0.0)*(1.0 - 0.8*facing);
            let reach = radius*1.1 + 0.22*flight;
            let droplet = vec2<f32>(cos(angle)*reach, sin(angle)*reach*facing + rise);
            let droplet_radius = max(0.003, 0.8*pixel_metres);
            droplets = max(droplets, 1.0 - smoothstep(droplet_radius,
                                                      droplet_radius + pixel_metres,
                                                      length(q - droplet)));
        }
        let alpha = in.opacity*max(ring, droplets*(1.0 - 0.6*age));
        return vec4<f32>(in.colour*1.1, alpha);
    }
    let radius = length((uv - vec2<f32>(0.5))*2.0);
    let flake = exp(-radius*radius*2.8)*(1.0 - smoothstep(0.78, 1.0, radius));
    let colour = vec3<f32>(0.85, 0.91, 1.0)*mix(0.42, 1.0, max(day, flash));
    return vec4<f32>(colour, in.opacity*flake*0.72);
}
