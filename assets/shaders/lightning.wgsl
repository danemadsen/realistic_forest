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
};
@group(0) @binding(0) var<uniform> globals: GlobalUniforms;
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
    let start = head.xyz;
    var end = tail.xyz;
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
    let inside = smoothstep(base - 30.0, base + 150.0, middle.y);
    let under_deck = bolt.cloud.z*(1.0 - smoothstep(base - 60.0, base + 60.0, middle.y));
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
