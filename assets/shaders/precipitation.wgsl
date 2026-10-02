// Camera-near world-space rain streaks and snow flakes. Drawn after the water
// pass so particles appear in front of the resolved scene and before FXAA so
// subpixel streaks receive the same edge treatment as terrain.
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

struct WindUniforms {
    velocity: vec4<f32>, // xy near-ground XZ wind, z ocean enabled
};
@group(2) @binding(0) var<uniform> wind: WindUniforms;

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

// Return rain, snow, thunderstorm, gust, each in [0, 1]. The altitude
// transition follows the terrain material's low mountain snowline. Explicit
// Snow/Rain/Thunderstorm trends select phase while retaining spatial cells.
fn weatherPrecipitation(world_position: vec3<f32>) -> vec4<f32> {
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
    let precipitation = smoothstep(0.74, 0.90, score + globals.storm.x)
                        *wet_gate*below_cloud;
    let cold_altitude = world_position.y
        + 70.0*sin((world_position.x - globals.weather.x)/18000.0
                   + (world_position.z - globals.weather.y)/27000.0 + 0.7);
    var snow_fraction = smoothstep(80.0, 260.0, cold_altitude);
    let override_kind = u32(globals.storm.y + 0.5);
    if (override_kind == 1u || override_kind == 3u) { snow_fraction = 0.0; }
    if (override_kind == 2u) { snow_fraction = 1.0; }
    let rain = precipitation*(1.0 - snow_fraction);
    let snow = precipitation*snow_fraction;
    let convective_bias = select(0.0, globals.storm.x, override_kind == 3u);
    let thunderstorm = rain*smoothstep(0.91, 0.98, score + convective_bias);
    let gust = clamp(0.18*precipitation + 0.75*thunderstorm, 0.0, 1.0);
    return vec4<f32>(rain, snow, thunderstorm, gust);
}
// END SHARED SPATIAL PRECIPITATION

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

struct ParticleVertex {
    @builtin(position) position: vec4<f32>,
    @location(0) local: vec2<f32>,
    @location(1) view_depth: f32,
    @location(2) opacity: f32,
    @location(3) @interpolate(flat) kind: u32,
};

fn hiddenParticle(kind: u32) -> ParticleVertex {
    var out: ParticleVertex;
    out.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    out.local = vec2<f32>(0.0);
    out.view_depth = 0.0;
    out.opacity = 0.0;
    out.kind = kind;
    return out;
}

@vertex
fn vs_particle(@builtin(vertex_index) vertex: u32,
               @builtin(instance_index) instance: u32) -> ParticleVertex {
    let snow = instance >= 16384u;
    let index = select(instance, instance - 16384u, snow);
    let grid = select(32u, 24u, snow);
    let layers = select(16u, 12u, snow);
    let cell_size = select(1.5, 2.0, snow);
    let fall_speed = select(12.0, 1.8, snow);
    let kind = select(0u, 1u, snow);
    let x = index % grid;
    let z = (index / grid) % grid;
    let y = index / (grid*grid);
    let camera = globals.camera_position.xyz;
    let wind_shift = globals.weather.xy*select(0.22, 0.35, snow);
    let base_x = i32(floor((camera.x - wind_shift.x)/cell_size)) - i32(grid/2u);
    let base_z = i32(floor((camera.z - wind_shift.y)/cell_size)) - i32(grid/2u);
    let base_y = i32(floor(camera.y/cell_size)) - i32(layers/2u);
    let time_cells = globals.storm.z*fall_speed/cell_size;
    let cycle = floor(time_cells);
    let phase = fract(time_cells);
    let cell_x = base_x + i32(x);
    let cell_z = base_z + i32(z);
    let cell_y = base_y + i32(y);
    // As the phase wraps, the particle in the next lattice cell takes this
    // instance's place. The *set* of world particles therefore falls without
    // a visible periodic jump, even when the camera moves between cells.
    let source_cell = vec3<i32>(cell_x, cell_y + i32(cycle), cell_z);
    let jitter = vec3<f32>(cellRandom(source_cell, 19u),
                            cellRandom(source_cell, 43u),
                            cellRandom(source_cell, 71u));
    let center = vec3<f32>((f32(cell_x) + jitter.x)*cell_size + wind_shift.x,
                           (f32(cell_y) + jitter.y - phase)*cell_size,
                           (f32(cell_z) + jitter.z)*cell_size + wind_shift.y);
    let precipitation = weatherPrecipitation(center);
    let intensity = select(precipitation.x, precipitation.y, snow);
    let visible_drop = cellRandom(source_cell, 109u) < intensity*0.96;
    let xz_edge = max(abs(center.x - camera.x), abs(center.z - camera.z));
    let y_edge = abs(center.y - camera.y);
    let edge_fade = (1.0 - smoothstep(18.0, 24.0, xz_edge))
                    *(1.0 - smoothstep(9.0, 12.0, y_edge));
    let distance = length(center - camera);
    let near_fade = smoothstep(0.4, 1.7, distance);
    if (!visible_drop || intensity <= 0.002 || edge_fade <= 0.001
        || (wind.velocity.z > 0.5 && center.y < 0.0)) {
        return hiddenParticle(kind);
    }

    let corners = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0),
        vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0));
    let local = corners[vertex];
    let signed = local*2.0 - vec2<f32>(1.0);
    let camera_right = normalize(vec3<f32>(globals.view[0].x, globals.view[1].x, globals.view[2].x));
    let camera_up = normalize(vec3<f32>(globals.view[0].y, globals.view[1].y, globals.view[2].y));
    var world_position = center;
    if (snow) {
        let size = 0.023 + 0.027*cellRandom(source_cell, 137u);
        world_position += camera_right*signed.x*size + camera_up*signed.y*size;
    } else {
        let fall_direction = normalize(vec3<f32>(wind.velocity.x, -fall_speed, wind.velocity.y));
        let to_eye = normalize(camera - center);
        var side = cross(fall_direction, to_eye);
        if (length(side) < 0.05) { side = camera_right; }
        side = normalize(side);
        let width = 0.007 + 0.004*cellRandom(source_cell, 149u);
        let streak = (0.17 + 0.18*cellRandom(source_cell, 163u))
                     *(1.0 + precipitation.w*0.35);
        world_position += side*signed.x*width + fall_direction*signed.y*streak;
    }
    let view_position = globals.view*vec4<f32>(world_position, 1.0);
    if (view_position.z >= -0.1) { return hiddenParticle(kind); }
    var out: ParticleVertex;
    out.position = globals.projection*view_position;
    out.local = local;
    out.view_depth = -view_position.z;
    out.opacity = edge_fade*near_fade*sqrt(intensity);
    out.kind = kind;
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
    let flash = clamp(globals.lightning.w*0.12, 0.0, 1.0);
    if (in.kind == 0u) {
        let edge = smoothstep(0.0, 0.25, uv.x)
                   *(1.0 - smoothstep(0.75, 1.0, uv.x));
        let tail = smoothstep(0.0, 0.18, uv.y)
                   *(1.0 - smoothstep(0.70, 1.0, uv.y));
        let background = textureLoad(scene_texture, pixel, 0).rgb;
        let brightness = dot(background, vec3<f32>(0.2126, 0.7152, 0.0722));
        let bright_streak = vec3<f32>(0.78, 0.85, 0.91);
        let dark_streak = vec3<f32>(0.28, 0.34, 0.39);
        let colour = mix(bright_streak, dark_streak,
                         smoothstep(0.58, 0.86, brightness))
                     *mix(0.45, 1.0, max(day, flash));
        return vec4<f32>(colour, in.opacity*edge*tail*0.34);
    }
    let radius = length((uv - vec2<f32>(0.5))*2.0);
    let flake = exp(-radius*radius*2.8)*(1.0 - smoothstep(0.78, 1.0, radius));
    let colour = vec3<f32>(0.85, 0.91, 1.0)*mix(0.42, 1.0, max(day, flash));
    return vec4<f32>(colour, in.opacity*flake*0.72);
}
