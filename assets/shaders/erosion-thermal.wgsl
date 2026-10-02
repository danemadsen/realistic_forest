// Hydraulic erosion pass 4: thermal (talus) relaxation.
// texture0: terrain (bed height, suspended sediment, immutable base height,
//           loose cover thickness)
//
// Weathered material cannot stand on any slope steeper than its stable
// angle. Loose cover (talus, colluvium and alluvium, tracked in terrain A)
// sheds at its angle of repose; bare bedrock only fails once a face exceeds
// a much steeper, hardness-dependent angle, and then slowly, as rockfall.
// Everything that moves arrives as loose cover. Repeated with the hydraulic
// passes this widens fresh incisions into V-shaped valleys, rounds weak
// crests, keeps resistant rock standing as cliffs and accumulates scree
// aprons at the foot of steep faces.
//
// The exchange is pairwise and antisymmetric: both cells of a pair evaluate
// the same transfer from the same data (the higher cell is the source), so
// the pass conserves material exactly without a scatter step. Each pair moves
// at most 1/16 of its excess per step, which cannot overshoot even when a
// spike sheds to all eight neighbours at once. Inputs are texel fetches and
// world-keyed geology only, so overlapping tiles stay deterministic.

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

struct StageUniforms {
    resolution: vec2<f32>,      // simulation grid size in texels
    cell_size: f32,             // metres per texel
    sea_level: f32,             // ocean beds below this are immutable
    world_min: vec2<f32>,       // world XZ corner of texel (0, 0)
    loose_rate: f32,            // fraction of a pair's loose excess moved per step
    rock_rate: f32,             // fraction of a pair's bedrock excess moved per step
    loose_repose: f32,          // tan of the loose cover's angle of repose
    soft_rock_slope: f32,       // tan of the steepest stable face in weak bedrock
    hard_rock_slope: f32,       // tan of the steepest stable face in resistant bedrock
    _pad: f32,
};
@group(2) @binding(0) var<uniform> stage: StageUniforms;

@group(1) @binding(0) var texture0: texture_2d<f32>;
@group(1) @binding(8) var tex0_sampler: sampler;   // declared for layout parity; texel loads only

struct VsOutput {
    @builtin(position) position: vec4<f32>,
};
@vertex
fn vs_main(@builtin(vertex_index) vertex: u32) -> VsOutput {
    var corner = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    var output: VsOutput;
    output.position = vec4<f32>(corner[vertex], 0.0, 1.0);
    return output;
}

// BEGIN SHARED GEOLOGY
// World-keyed substrate resistance. The erosion passes and the terrain
// material shader paste this block verbatim (a test keeps the copies equal),
// so the rock that resists incision and holds steep faces in the simulation
// is the same rock the material shader exposes. Every input is a world
// coordinate, which keeps overlapping erosion tiles deterministic.
//
// Broad (~92 m) and fine (~27 m) rock bodies vary hardness without following
// contours. A weaker bedding term adds gently dipping, warped resistant beds
// every ~19 m of bedrock elevation: where incision or talus relaxation cuts
// through them they hold short cliff bands and benches instead of one
// uniform slope. The bedding is keyed to the bedrock surface, so a bench
// stays put as loose cover accumulates above it.
const GEOLOGY_HARDNESS_SCALE: f32 = 0.56;

fn geologyHash(position: vec2<f32>) -> f32
{
    var p = fract(vec3<f32>(position.xyx) * 0.1031);
    p += dot(p, p.yzx + 33.33);
    return fract((p.x + p.y) * p.z);
}

fn geologyNoise(position: vec2<f32>) -> f32
{
    let cell = floor(position);
    var fraction = fract(position);
    fraction = fraction * fraction * (3.0 - 2.0 * fraction);
    let a = geologyHash(cell);
    let b = geologyHash(cell + vec2<f32>(1.0, 0.0));
    let c = geologyHash(cell + vec2<f32>(0.0, 1.0));
    let d = geologyHash(cell + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, fraction.x), mix(c, d, fraction.x), fraction.y);
}

// 0 = weak, readily weathered substrate; 1 = the most resistant rock.
fn geologyResistance(world_xz: vec2<f32>, bedrock_height: f32) -> f32
{
    let broadRock = geologyNoise(world_xz / 92.0 + vec2<f32>(31.7, -18.2));
    let fineRock = geologyNoise(world_xz / 27.0 + vec2<f32>(-73.1, 46.4));
    let bedWarp = geologyNoise(world_xz / 310.0 + vec2<f32>(-12.9, 57.3)) - 0.5;
    let bedCoordinate = (bedrock_height + dot(world_xz, vec2<f32>(0.017, -0.011))
                         + bedWarp * 26.0) / 19.0;
    let bedding = smoothstep(0.35, 0.85, 0.5 + 0.5 * sin(bedCoordinate * 6.2831853));
    let geology = broadRock * 0.62 + fineRock * 0.24 + bedding * 0.14;
    return smoothstep(0.25, 0.82, geology);
}

// The small spawn footprint resists destructive excavation.
fn erosionSpawnProtection(world_xz: vec2<f32>) -> f32
{
    return 1.0 - smoothstep(90.0, 150.0, length(world_xz));
}

// Hardness as the erosion solver uses it: scaled resistance, raised to the
// spawn protection inside its footprint.
fn erosionHardness(world_xz: vec2<f32>, bedrock_height: f32) -> f32
{
    return max(geologyResistance(world_xz, bedrock_height) * GEOLOGY_HARDNESS_SCALE,
               erosionSpawnProtection(world_xz) * 0.96);
}
// END SHARED GEOLOGY

fn gridSize() -> vec2<i32>
{
    let texture_resolution = vec2<i32>(textureDimensions(texture0, 0));
    if (stage.resolution.x < 1.0 || stage.resolution.y < 1.0)
    {
        return texture_resolution;
    }
    return min(texture_resolution, max(vec2<i32>(stage.resolution + vec2<f32>(0.5)), vec2<i32>(1)));
}

fn clampCoord(coord: vec2<i32>) -> vec2<i32>
{
    return clamp(coord, vec2<i32>(0), gridSize() - vec2<i32>(1));
}

fn fragmentCoord(position: vec4<f32>) -> vec2<i32>
{
    let resolution = select(vec2<f32>(gridSize()), stage.resolution,
        stage.resolution.x >= 1.0 && stage.resolution.y >= 1.0);
    let uv = position.xy / resolution;
    return clampCoord(vec2<i32>(floor(uv * vec2<f32>(gridSize()))));
}

fn cellWorld(coord: vec2<i32>) -> vec2<f32>
{
    return stage.world_min + (vec2<f32>(coord) + vec2<f32>(0.5)) * max(stage.cell_size, 0.0001);
}

// Steepest stable face for the bedrock exposed in a cell.
fn rockSlopeLimit(coord: vec2<i32>, terrain: vec4<f32>) -> f32
{
    let bedrock = terrain.r - max(terrain.a, 0.0);
    let resistance = clamp(erosionHardness(cellWorld(coord), bedrock) / GEOLOGY_HARDNESS_SCALE,
                           0.0, 1.0);
    return mix(stage.soft_rock_slope, stage.hard_rock_slope, resistance);
}

@fragment
fn fs_main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32>
{
    let coord = fragmentCoord(position);
    let size = gridSize();
    let center = textureLoad(texture0, coord, 0);
    // Ocean beds are immutable; the sea carries off anything shed into it,
    // so coastal faces exchange nothing with them.
    if (center.b < stage.sea_level)
    {
        return center;
    }

    let cell_size = max(stage.cell_size, 0.0001);
    let center_loose = max(center.a, 0.0);
    let center_rock_limit = rockSlopeLimit(coord, center);
    let loose_rate = clamp(stage.loose_rate, 0.0, 0.0625);
    let rock_rate = clamp(stage.rock_rate, 0.0, 0.0625);

    var height_change = 0.0;
    var loose_change = 0.0;
    for (var index = 0; index < 9; index++)
    {
        if (index == 4) { continue; }
        let offset = vec2<i32>(index % 3 - 1, index / 3 - 1);
        let neighbour_coord = coord + offset;
        // Off-grid neighbours do not exist: a clamped fetch would pair the
        // edge cell with itself or double-count a real neighbour.
        if (any(neighbour_coord < vec2<i32>(0)) || any(neighbour_coord >= size)) { continue; }
        let neighbour = textureLoad(texture0, neighbour_coord, 0);
        if (neighbour.b < stage.sea_level) { continue; }

        let diagonal = offset.x != 0 && offset.y != 0;
        let distance = cell_size * select(1.0, 1.41421356, diagonal);
        let drop = center.r - neighbour.r;
        let fall = abs(drop);
        if (fall <= stage.loose_repose * distance) { continue; }

        // Both cells of the pair pick the same source: the higher one.
        let center_is_source = drop > 0.0;
        let source_loose = select(max(neighbour.a, 0.0), center_loose, center_is_source);
        let source_rock_limit = select(rockSlopeLimit(neighbour_coord, neighbour),
                                       center_rock_limit, center_is_source);

        // Loose cover sheds first and can only give what it has; a face
        // steeper than its rock limit also sheds its weathered bedrock.
        let loose_moved = min(loose_rate * (fall - stage.loose_repose * distance),
                              source_loose * 0.125);
        let rock_moved = rock_rate * max(fall - source_rock_limit * distance, 0.0);
        let moved = max(loose_moved, rock_moved);
        if (center_is_source)
        {
            height_change -= moved;
            loose_change -= loose_moved;
        }
        else
        {
            height_change += moved;
            loose_change += moved;
        }
    }

    return vec4<f32>(center.r + height_change, center.g, center.b,
                     max(center_loose + loose_change, 0.0));
}

// STAGE UNIFORMS (group 2, binding 0), 48 bytes:
//   offset  0 resolution: vec2<f32>  offset  8 cell_size        offset 12 sea_level
//   offset 16 world_min: vec2<f32>   offset 24 loose_rate       offset 28 rock_rate
//   offset 32 loose_repose           offset 36 soft_rock_slope  offset 40 hard_rock_slope
//   offset 44 pad
// Textures (group 1): binding 0 = terrain state RGBA32F (texel loads only;
// non-filtering sampler at binding 8 for layout parity).
// Output: the relaxed terrain state (bed, suspended sediment, base, loose cover).
