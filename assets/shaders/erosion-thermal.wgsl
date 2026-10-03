// Hydraulic erosion pass 4: thermal (talus) relaxation.
// texture0: terrain (bed height, suspended sediment, immutable base height,
//           loose cover thickness)
// uDrainage: the terrain pass's drainage state; B is the resistance (0-1) of
//           the bedrock under each cell's cover, evaluated by that pass from
//           the shared geology at the surface this pass reads
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
// spike sheds to all eight neighbours at once. Inputs are texel fetches of
// world-keyed state only, so overlapping tiles stay deterministic.

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
@group(1) @binding(1) var uDrainage: texture_2d<f32>;
@group(1) @binding(8) var tex0_sampler: sampler;   // declared for layout parity; texel loads only
@group(1) @binding(9) var uDrainage_sampler: sampler;

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

// Steepest stable face for the bedrock exposed in a cell. The terrain pass
// evaluated the shared geology at this very bedrock surface and left the
// resistance in drainage B, so no cell re-derives it per neighbour.
fn rockSlopeLimit(coord: vec2<i32>) -> f32
{
    let resistance = clamp(textureLoad(uDrainage, coord, 0).b, 0.0, 1.0);
    return mix(stage.soft_rock_slope, stage.hard_rock_slope, resistance);
}

@fragment
fn fs_main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32>
{
    let coord = fragmentCoord(position);
    let size = gridSize();
    let center = textureLoad(texture0, coord, 0);
    // Ocean beds are immutable; the sea carries off anything shed into it,
    // so coastal faces exchange nothing with them. River beds likewise.
    if (center.b < stage.sea_level || textureLoad(uDrainage, coord, 0).a > 0.5)
    {
        return center;
    }

    let cell_size = max(stage.cell_size, 0.0001);
    let center_loose = max(center.a, 0.0);
    let center_rock_limit = rockSlopeLimit(coord);
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
        if (neighbour.b < stage.sea_level || textureLoad(uDrainage, neighbour_coord, 0).a > 0.5) { continue; }

        let diagonal = offset.x != 0 && offset.y != 0;
        let distance = cell_size * select(1.0, 1.41421356, diagonal);
        let drop = center.r - neighbour.r;
        let fall = abs(drop);
        if (fall <= stage.loose_repose * distance) { continue; }

        // Both cells of the pair pick the same source: the higher one.
        let center_is_source = drop > 0.0;
        let source_loose = select(max(neighbour.a, 0.0), center_loose, center_is_source);
        let source_rock_limit = select(rockSlopeLimit(neighbour_coord),
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
//   offset 16 world_min: vec2<f32> (unused here; kept for the shared fill)
//   offset 24 loose_rate             offset 28 rock_rate
//   offset 32 loose_repose           offset 36 soft_rock_slope  offset 40 hard_rock_slope
//   offset 44 pad
// Textures (group 1): binding 0 = terrain state RGBA32F, binding 1 = drainage
// state RGBA32F (texel loads only; non-filtering samplers at 8 and 9 for
// layout parity).
// Output: the relaxed terrain state (bed, suspended sediment, base, loose cover).
