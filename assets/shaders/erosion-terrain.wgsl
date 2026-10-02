// erosion-terrain.wgsl — 1:1 port of erosion_terrain.fs (GLSL #version 330)
// Source: /Users/danemadsen/forest/assets/shaders/erosion_terrain.fs
//
// Hydraulic erosion pass 3: advect sediment, then erode or deposit terrain.
// texture0:       terrain (bed height, sediment, immutable base height, loose cover)
// uWaterState:    water (depth, velocity x, velocity z, accumulated discharge)
// uDrainage:      drainage (contributing area in cells, routed sediment flux,
//                 bedrock resistance for the thermal pass, 0)
//
// Two processes share the pass. The runoff solver below (the original
// erosion_terrain.fs) carves the fine rills and gullies that the transient
// rain sheet can reach in a few seconds. Fluvial transport adds what that
// water never lives long enough to do: drainage area is routed between bed
// cells (multiple flow direction, slope-squared shares), so every channel
// knows its whole catchment, and a stream-power capacity, k*sqrt(A)*S,
// detaches bed where the routed sediment flux is below capacity and drops it
// where flux exceeds capacity. Steep, well-fed channels incise; slope breaks
// build fans; valley floors and closed hollows fill with alluvium.
//
// Terrain A tracks the loose cover above bedrock. Deposits add to it, loose
// material is entrained first, and only then is bedrock cut at the
// resistance of the world-keyed geology it exposes. The thermal pass reads
// the same cover to decide which slopes stand and which shed.
//
// PORT NOTES (GLSL -> WGSL deviations):
//  - No inverse_* stage-uniform fields added: the source performs no matrix
//    inverse.
//  - No vertical flips dropped: the source performs no render-texture flips,
//    and per the porting spec the wgpu port shares one top-left row order for
//    every target/readback, so fragment_coord's GL gl_FragCoord math ports
//    unchanged (this screen->texel mapping is the only coordinate-handedness
//    use in the file; texel indices come out identical).
//  - No samples hoisted: every texture read is texelFetch(..., 0), ported to
//    textureLoad, which carries no uniform-control-flow restriction, so all
//    reads stay exactly where the GLSL placed them (including inside branches
//    and ternaries on per-pixel data).
//  - GLSL ternaries became WGSL select(...), which evaluates BOTH operands:
//    an untaken operand can yield inf/NaN (e.g. -gradient / slope when
//    slope == 0.0); select discards it, so the produced value matches the
//    GLSL taken branch. No other branching was restructured.
//  - Samplers for texture0 (binding 8) and uWaterState (binding 9) are
//    declared because the spec maps every sampler2D to a texture + sampler
//    pair; this pass never filters, so the Rust side can use non-filtering
//    samplers there.
//  - fragTexCoord/fragColor (raylib interpolators) were declared but unused
//    in the GLSL — nothing to port for them.
//  - fs_main reads the previous terrain state from texture0 and writes the
//    updated state to @location(0), plus the routed drainage state to
//    @location(1); exactly as with the GL FBO feedback rules, the Rust side
//    must ping-pong both instead of sampling the attachments it renders to.

// ---------------------------------------------------------------------------
// Shared global uniforms (porting spec section 1) — pasted verbatim.
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Per-stage uniforms (porting spec section 2)
// ---------------------------------------------------------------------------

// StageUniforms provenance (GLSL uFooBar -> foo_bar): every remaining uniform
// of erosion_terrain.fs — uResolution, uDeltaTime, uCellSize, uSeaLevel,
// uErosionRate, uDepositionRate, uSedimentCapacity, uMinimumSlope,
// uMaximumErosion, uTransportRate, uGuardBandPixels, uBrushStrength,
// uWorldMin.
struct StageUniforms {
    resolution: vec2<f32>,          // uResolution
    delta_time: f32,                // uDeltaTime
    cell_size: f32,                 // uCellSize
    sea_level: f32,                 // uSeaLevel
    erosion_rate: f32,              // uErosionRate
    deposition_rate: f32,           // uDepositionRate
    sediment_capacity: f32,         // uSedimentCapacity
    minimum_slope: f32,             // uMinimumSlope
    maximum_erosion: f32,           // uMaximumErosion
    transport_rate: f32,            // uTransportRate
    guard_band_pixels: f32,         // uGuardBandPixels
    brush_strength: f32,            // uBrushStrength
    // Required for the world-stable sub-cell variation below. The value is the
    // world-space XZ corner of texel (0, 0), just like erosion_init.fs.
    world_min: vec2<f32>,           // uWorldMin
    fluvial_capacity: f32,          // stream-power capacity per sqrt(cell) of catchment
    fluvial_erosion: f32,           // fraction of a capacity deficit detached per step
    fluvial_deposition: f32,        // fraction of a capacity excess deposited per step
    maximum_incision: f32,          // deepest fluvial cut below the base surface, metres
    drainage_saturation: f32,       // catchment (cells) at which stream power saturates
};
@group(2) @binding(0) var<uniform> stage: StageUniforms;

// ---------------------------------------------------------------------------
// Textures (porting spec section 3 — GL unit numbers preserved)
// ---------------------------------------------------------------------------

// texture0: terrain (bed height, sediment, immutable base height, loose cover)
@group(1) @binding(0) var texture0: texture_2d<f32>;
// uWaterState: water (depth, velocity x, velocity z, accumulated discharge)
@group(1) @binding(1) var uWaterState: texture_2d<f32>;
// uDrainage: drainage (contributing area in cells, routed sediment flux,
// bedrock resistance, 0)
@group(1) @binding(2) var uDrainage: texture_2d<f32>;
// uRouting: each cell's routing normaliser (R), written by this iteration's
// water pass from the same beds this pass reads
@group(1) @binding(3) var uRouting: texture_2d<f32>;

// Samplers sit at GL unit + 8 per the spec; this pass only texel-fetches.
@group(1) @binding(8) var tex0_sampler: sampler;
@group(1) @binding(9) var uWaterState_sampler: sampler;
@group(1) @binding(10) var uDrainage_sampler: sampler;
@group(1) @binding(11) var uRouting_sampler: sampler;

// ---------------------------------------------------------------------------
// Fullscreen-triangle vertex shader (porting spec section 4)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Helper functions (decomposition preserved 1:1)
// ---------------------------------------------------------------------------

fn gridSize() -> vec2<i32>
{
    let texture_resolution = vec2<i32>(textureDimensions(texture0, 0u));
    if (stage.resolution.x < 1.0 || stage.resolution.y < 1.0)
    {
        return texture_resolution;
    }

    return min(texture_resolution,
               max(vec2<i32>(stage.resolution + vec2<f32>(0.5)), vec2<i32>(1)));
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

fn fetchTerrain(coord: vec2<i32>) -> vec4<f32>
{
    return textureLoad(texture0, clampCoord(coord), 0);
}

fn fetchWater(coord: vec2<i32>) -> vec4<f32>
{
    return textureLoad(uWaterState, clampCoord(coord), 0);
}

fn fetchDrainage(coord: vec2<i32>) -> vec4<f32>
{
    return textureLoad(uDrainage, clampCoord(coord), 0);
}

fn sampleTerrainBilinear(texel_position: vec2<f32>) -> vec4<f32>
{
    let max_position = vec2<f32>(gridSize() - vec2<i32>(1));
    let position = clamp(texel_position, vec2<f32>(0.0), max_position);
    let lower = vec2<i32>(floor(position));
    let upper = clampCoord(lower + vec2<i32>(1));
    let fraction = fract(position);

    let sample00 = fetchTerrain(lower);
    let sample10 = fetchTerrain(vec2<i32>(upper.x, lower.y));
    let sample01 = fetchTerrain(vec2<i32>(lower.x, upper.y));
    let sample11 = fetchTerrain(upper);
    return mix(mix(sample00, sample10, fraction.x),
               mix(sample01, sample11, fraction.x), fraction.y);
}

fn sampleWaterBilinear(texel_position: vec2<f32>) -> vec4<f32>
{
    let max_position = vec2<f32>(gridSize() - vec2<i32>(1));
    let position = clamp(texel_position, vec2<f32>(0.0), max_position);
    let lower = vec2<i32>(floor(position));
    let upper = clampCoord(lower + vec2<i32>(1));
    let fraction = fract(position);

    let sample00 = fetchWater(lower);
    let sample10 = fetchWater(vec2<i32>(upper.x, lower.y));
    let sample01 = fetchWater(vec2<i32>(lower.x, upper.y));
    let sample11 = fetchWater(upper);
    return mix(mix(sample00, sample10, fraction.x),
               mix(sample01, sample11, fraction.x), fraction.y);
}

// A small, smooth, world-space variation takes the place of the random initial
// direction of Nerthus's droplets. It breaks up perfectly parallel Eulerian
// flow without changing from iteration to iteration or at tile overlaps.
fn hash12(position: vec2<f32>) -> f32
{
    var p = fract(vec3<f32>(position.xyx) * 0.1031);
    p += dot(p, p.yzx + 33.33);
    return fract((p.x + p.y) * p.z);
}

fn valueNoise(position: vec2<f32>) -> f32
{
    let cell = floor(position);
    var fraction = fract(position);
    fraction = fraction * fraction * (3.0 - 2.0 * fraction);
    let n00 = hash12(cell);
    let n10 = hash12(cell + vec2<f32>(1.0, 0.0));
    let n01 = hash12(cell + vec2<f32>(0.0, 1.0));
    let n11 = hash12(cell + vec2<f32>(1.0, 1.0));
    return mix(mix(n00, n10, fraction.x), mix(n01, n11, fraction.x), fraction.y);
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

fn similarityWeight(neighbour_height: f32, center_height: f32,
                    spatial_weight: f32, height_scale: f32) -> f32
{
    let difference = (neighbour_height - center_height) / max(height_scale, 0.0001);
    return spatial_weight / (1.0 + difference * difference);
}

// ---------------------------------------------------------------------------
// Fragment stage
// ---------------------------------------------------------------------------

struct FragmentOutput {
    @location(0) final_color: vec4<f32>,   // finalColor: terrain state
    @location(1) drainage: vec4<f32>,      // routed drainage state
};

@fragment
fn fs_main(@builtin(position) position: vec4<f32>) -> FragmentOutput
{
    let coord = fragmentCoord(position);
    let size = gridSize();

    // Simulate the complete padded domain. The retained footprint is only
    // a readback crop, never a hydraulic or erosion boundary.
    let terrain = fetchTerrain(coord);
    let water = fetchWater(coord);

    var bed_height = terrain.r;
    let base_height = terrain.b;
    var loose = max(terrain.a, 0.0);

    // Ocean terrain is immutable; ocean water is rendered/supplied separately.
    // The sea also swallows any drainage and sediment routed into it.
    if (base_height < stage.sea_level)
    {
        return FragmentOutput(vec4<f32>(base_height, 0.0, base_height, loose), vec4<f32>(0.0));
    }

    let dt = max(stage.delta_time, 0.0);
    let cell_size = max(stage.cell_size, 0.0001);
    let velocity = water.gb;

    // Resistance belongs to the bedrock surface under the loose cover, so a
    // cut that reaches a resistant bed slows there and leaves a bench.
    let world_position = stage.world_min + (vec2<f32>(coord) + vec2<f32>(0.5)) * cell_size;
    let hardness = clamp(erosionHardness(world_position, bed_height - loose), 0.0, 1.0);
    let rock_erodibility = pow(max(1.0 - hardness, 0.0), 1.35);
    let loose_erodibility = 1.0 - 0.96 * erosionSpawnProtection(world_position);

    // Drainage routing. Every bed cell passes its contributing area, and the
    // sediment flux it carries, to its lower neighbours in proportion to
    // their squared slope. The water pass just before this one left each
    // donor's own total of squared downhill slopes in uRouting, so the gather
    // reads exactly what each donor gives away without re-deriving it. Bed
    // heights route the flow (not the transient water surface), which keeps
    // the routing acyclic from one iteration to the next. Cells beyond the
    // grid neither give nor receive flow, keeping the simulation wall closed.
    var drainage_area = 1.0;
    var sediment_in = 0.0;
    var steepest_descent = 0.0;
    var lowest_neighbour = bed_height;
    for (var donor = 0; donor < 9; donor++)
    {
        if (donor == 4) { continue; }
        let donor_offset = vec2<i32>(donor % 3 - 1, donor / 3 - 1);
        let donor_coord = coord + donor_offset;
        if (any(donor_coord < vec2<i32>(0)) || any(donor_coord >= size)) { continue; }
        let donor_bed = fetchTerrain(donor_coord).r;
        let donor_distance = cell_size
            * select(1.0, 1.41421356, donor_offset.x != 0 && donor_offset.y != 0);
        let fall = (bed_height - donor_bed) / donor_distance;
        steepest_descent = max(steepest_descent, fall);
        lowest_neighbour = min(lowest_neighbour, donor_bed);
        // The same pair seen from the donor: its slope toward this cell.
        let rise = -fall;
        if (rise <= 0.0) { continue; }
        let share = rise * rise / max(textureLoad(uRouting, donor_coord, 0).r, 1.0e-12);
        let donor_drainage = fetchDrainage(donor_coord);
        drainage_area += max(donor_drainage.r, 0.0) * share;
        sediment_in += max(donor_drainage.g, 0.0) * share;
    }

    // Semi-Lagrangian backtracing asks which upstream cell supplied this cell.
    // Nerthus advances droplets by a bounded step. Applying the same bound here
    // prevents a numerically large shallow-water velocity from drawing long,
    // ruler-straight sediment streaks across the grid.
    var backtrace = velocity * (dt / cell_size);
    let backtrace_length = length(backtrace);
    if (backtrace_length > 1.25)
    {
        backtrace = backtrace * (1.25 / backtrace_length);
    }
    let upstream_position = vec2<f32>(coord) - backtrace;
    let upstream_sediment = max(sampleTerrainBilinear(upstream_position).g, 0.0);
    let transport_blend = clamp(max(stage.transport_rate, 0.0) * dt, 0.0, 1.0);
    var sediment = mix(max(terrain.g, 0.0), upstream_sediment, transport_blend);

    // A Sobel gradient is less axis-biased than the old cardinal difference.
    // The same samples are reused by the detail-preserving footprint below.
    let terrain_left  = fetchTerrain(coord + vec2<i32>(-1, 0));
    let terrain_right = fetchTerrain(coord + vec2<i32>( 1, 0));
    let terrain_down  = fetchTerrain(coord + vec2<i32>( 0,-1));
    let terrain_up    = fetchTerrain(coord + vec2<i32>( 0, 1));
    let terrain_left_down  = fetchTerrain(coord + vec2<i32>(-1,-1));
    let terrain_right_down = fetchTerrain(coord + vec2<i32>( 1,-1));
    let terrain_left_up    = fetchTerrain(coord + vec2<i32>(-1, 1));
    let terrain_right_up   = fetchTerrain(coord + vec2<i32>( 1, 1));

    let water_left  = fetchWater(coord + vec2<i32>(-1, 0));
    let water_right = fetchWater(coord + vec2<i32>( 1, 0));
    let water_down  = fetchWater(coord + vec2<i32>( 0,-1));
    let water_up    = fetchWater(coord + vec2<i32>( 0, 1));
    let water_left_down  = fetchWater(coord + vec2<i32>(-1,-1));
    let water_right_down = fetchWater(coord + vec2<i32>( 1,-1));
    let water_left_up    = fetchWater(coord + vec2<i32>(-1, 1));
    let water_right_up   = fetchWater(coord + vec2<i32>( 1, 1));

    let left_height  = terrain_left.r  + max(water_left.r, 0.0);
    let right_height = terrain_right.r + max(water_right.r, 0.0);
    let down_height  = terrain_down.r  + max(water_down.r, 0.0);
    let up_height    = terrain_up.r    + max(water_up.r, 0.0);
    let left_down_height  = terrain_left_down.r  + max(water_left_down.r, 0.0);
    let right_down_height = terrain_right_down.r + max(water_right_down.r, 0.0);
    let left_up_height    = terrain_left_up.r    + max(water_left_up.r, 0.0);
    let right_up_height   = terrain_right_up.r   + max(water_right_up.r, 0.0);
    let gradient = vec2<f32>(
        right_down_height + 2.0 * right_height + right_up_height
            - left_down_height - 2.0 * left_height - left_up_height,
        left_up_height + 2.0 * up_height + right_up_height
            - left_down_height - 2.0 * down_height - right_down_height
    ) / (8.0 * cell_size);
    let slope = length(gradient);
    let speed = length(velocity);
    let water_depth = max(water.r, 0.0);

    let world_cell = world_position / cell_size;
    let broad_path_noise = valueNoise(world_cell * 0.16 + vec2<f32>(17.13, -9.71));
    let fine_path_noise = valueNoise(world_cell * 0.43 + vec2<f32>(-31.7, 22.9));
    let path_noise = broad_path_noise * 0.45 + fine_path_noise * 0.55;

    // Nerthus gives a droplet a random direction on perfectly flat ground.
    // Use a deterministic world-cell angle here so an overlap evaluates the
    // same way in every tile and flat cells do not all inherit an eastward bias.
    let flat_angle = hash12(floor(world_cell) + vec2<f32>(91.7, -54.3)) * 6.28318530718;
    let flat_direction = vec2<f32>(cos(flat_angle), sin(flat_angle));

    // Treat the water velocity as the droplet's inertial direction and bend it
    // back toward the local downhill gradient. This is the fragment-pass form
    // of Nerthus's `dir = dir*inertia - grad*(1-inertia)` update.
    let downhill_direction = select(
        select(flat_direction, velocity / speed, speed > 0.000001),
        -gradient / slope,
        slope > 0.000001);
    let velocity_direction = select(downhill_direction, velocity / speed,
                                    speed > 0.000001);
    let inertia = clamp(0.18 + 0.54 * speed / (speed + 4.0), 0.18, 0.72);
    var travel_direction = mix(downhill_direction, velocity_direction, inertia);
    let travel_direction_length = length(travel_direction);
    travel_direction = select(downhill_direction,
                              travel_direction / travel_direction_length,
                              travel_direction_length > 0.000001);

    let bend_angle = ((broad_path_noise - 0.5) * 0.68
        + (fine_path_noise - 0.5) * 0.24)
        * (1.0 - smoothstep(0.035, 0.55, slope));
    let bend_cos = cos(bend_angle);
    let bend_sin = sin(bend_angle);
    // mat2(bendCos, bendSin, -bendSin, bendCos) takes GLSL column-major
    // arguments, so the WGSL columns are built identically.
    travel_direction = mat2x2<f32>(vec2<f32>(bend_cos, bend_sin),
                                   vec2<f32>(-bend_sin, bend_cos)) * travel_direction;

    let travel_texels = mix(0.65, 1.15, speed / (speed + 3.0));
    let forward_position = vec2<f32>(coord) + travel_direction * travel_texels;
    let forward_terrain = sampleTerrainBilinear(forward_position);
    let forward_water = sampleWaterBilinear(forward_position);
    let center_surface = bed_height + water_depth;
    let forward_surface = forward_terrain.r + max(forward_water.r, 0.0);
    let height_change_along_path = forward_surface - center_surface;
    let directional_slope = max(-height_change_along_path / (travel_texels * cell_size), 0.0);

    // Prefer cells which already collect a little more discharge. This modest
    // positive feedback creates fine tributaries, while the depth resistance
    // below prevents those tributaries becoming deep trenches.
    let neighbour_discharge = (
        max(water_left.a, 0.0) + max(water_right.a, 0.0)
        + max(water_down.a, 0.0) + max(water_up.a, 0.0)
        + 0.5 * (max(water_left_down.a, 0.0) + max(water_right_down.a, 0.0)
                 + max(water_left_up.a, 0.0) + max(water_right_up.a, 0.0))
    ) / 6.0;
    let discharge_ratio = (max(water.a, 0.0) + 0.002)
        / (neighbour_discharge + 0.002);
    let channel_focus = mix(0.45, 1.55,
        smoothstep(0.82, 1.18, clamp(discharge_ratio, 0.0, 2.0)));
    let detail_variation = mix(0.82, 1.18, path_noise);

    // Very shallow cells can report extreme velocities because velocity is
    // reconstructed by dividing flux by depth. Nerthus has a bounded particle
    // velocity; capping only the capacity term gives this solver the same
    // stability without discarding the direction stored in the flow output.
    let capacity_speed = min(speed, 8.0);
    let capacity = max(stage.sediment_capacity, 0.0) * capacity_speed
        * max(directional_slope, max(stage.minimum_slope, 0.0)) * water_depth
        * channel_focus * detail_variation;
    var eroded_this_step = 0.0;

    // Nerthus immediately drops enough load when a particle tries to climb.
    // Doing this before the normal capacity response fills pits instead of
    // letting a cell-centred solver dig through their uphill side.
    if (height_change_along_path > 0.0 && sediment > 0.0)
    {
        let deposit = min(sediment, height_change_along_path);
        bed_height += deposit;
        loose += deposit;
        sediment -= deposit;
    }
    else if (sediment > capacity)
    {
        let deposit = min(sediment,
            (sediment - capacity) * max(stage.deposition_rate, 0.0) * dt);
        bed_height += deposit;
        loose += deposit;
        sediment -= deposit;
    }
    else
    {
        // Fine noise also stands in for sub-cell lithology. It produces narrow
        // resistant ribs and tributaries instead of uniformly widening every
        // wet slope, but remains deterministic in world space.
        let material_variation = mix(0.58, 1.42, fine_path_noise);
        // Loose cover is entrained readily; bare bedrock at its resistance.
        let erodibility = mix(rock_erodibility, loose_erodibility,
                              smoothstep(0.0, 0.15, loose));
        var erode = (capacity - sediment) * max(stage.erosion_rate, 0.0)
            * erodibility * material_variation * dt;

        // Excavation rapidly becomes more expensive with depth. The configured
        // maximum remains a hard limit, but is no longer the depth every active
        // channel naturally converges to.
        let excavation_depth = max(base_height - bed_height, 0.0);
        var depth_resistance = 1.0 / (1.0 + excavation_depth / (0.90 * cell_size));
        if (stage.maximum_erosion > 0.0)
        {
            let depth_fraction = clamp(excavation_depth / stage.maximum_erosion, 0.0, 1.0);
            depth_resistance *= pow(1.0 - depth_fraction, 2.5);
        }
        erode *= depth_resistance;

        let remaining_excavation = select(
            erode,
            max(0.0, bed_height - (base_height - stage.maximum_erosion)),
            stage.maximum_erosion > 0.0);
        erode = min(erode, remaining_excavation);
        erode = min(erode, max(capacity - sediment, 0.0));

        // Cap the cut to the actual, continuous downhill bed step, exactly as
        // Nerthus caps a droplet by `-dh`. A small per-pass ceiling is also
        // necessary because one fragment represents many concurrent droplets.
        let downhill_bed_step = max(0.0, bed_height - forward_terrain.r);
        let step_ceiling = cell_size * mix(0.003, 0.012,
            smoothstep(max(stage.minimum_slope, 0.0), 0.85, directional_slope));
        erode = min(erode, min(downhill_bed_step, step_ceiling));

        bed_height -= erode;
        loose = max(loose - erode, 0.0);
        sediment += erode;
        eroded_this_step = erode;
    }

    // Continuously settle slow or nearly dry loads. This is the streaming-grid
    // equivalent of Nerthus settling a droplet's sediment on evaporation, edge
    // exit, or TTL expiry; otherwise the last frame's suspended load vanishes.
    let dry_signal = 1.0 - smoothstep(0.003, 0.045, water_depth);
    let slow_signal = 1.0 - smoothstep(0.15, 1.25, capacity_speed);
    let settle_signal = max(dry_signal, 0.7 * slow_signal);
    let settle_fraction = clamp(max(stage.deposition_rate, 0.0) * dt
        * (0.15 + 1.85 * settle_signal), 0.0, 1.0);
    let settled = sediment * settle_fraction;
    bed_height += settled;
    loose += settled;
    sediment -= settled;

    // The old binomial average blurred every hydraulically active 3x3 region.
    // This bilateral cone ignores neighbours across real height discontinuities,
    // preserves a small detail dead-zone, and only fills a newly-created isolated
    // cut. It cannot erode an otherwise untouched detail feature.
    let bilateral_scale = max(0.35, 0.20 * cell_size);
    let weight_left = similarityWeight(terrain_left.r, terrain.r, 1.0, bilateral_scale);
    let weight_right = similarityWeight(terrain_right.r, terrain.r, 1.0, bilateral_scale);
    let weight_down = similarityWeight(terrain_down.r, terrain.r, 1.0, bilateral_scale);
    let weight_up = similarityWeight(terrain_up.r, terrain.r, 1.0, bilateral_scale);
    let weight_left_down = similarityWeight(terrain_left_down.r, terrain.r, 0.55, bilateral_scale);
    let weight_right_down = similarityWeight(terrain_right_down.r, terrain.r, 0.55, bilateral_scale);
    let weight_left_up = similarityWeight(terrain_left_up.r, terrain.r, 0.55, bilateral_scale);
    let weight_right_up = similarityWeight(terrain_right_up.r, terrain.r, 0.55, bilateral_scale);
    let total_weight = weight_left + weight_right + weight_down + weight_up
        + weight_left_down + weight_right_down + weight_left_up + weight_right_up;
    let bilateral_average = (
        terrain_left.r * weight_left + terrain_right.r * weight_right
        + terrain_down.r * weight_down + terrain_up.r * weight_up
        + terrain_left_down.r * weight_left_down + terrain_right_down.r * weight_right_down
        + terrain_left_up.r * weight_left_up + terrain_right_up.r * weight_right_up
    ) / max(total_weight, 0.0001);
    let detail_dead_zone = 0.04 * cell_size;
    let isolated_cut = max(0.0, bilateral_average - bed_height - detail_dead_zone);
    var brush_correction = min(isolated_cut, eroded_this_step
        * clamp(stage.brush_strength, 0.0, 1.0) * 0.45);
    brush_correction = min(brush_correction, sediment);
    bed_height += brush_correction;
    loose += brush_correction;
    sediment -= brush_correction;

    // Fluvial transport along the routed drainage. Capacity grows with the
    // square root of the catchment and the steepest bed descent, saturating
    // for catchments larger than the tile can see in full (so overlapping
    // tiles, whose domains truncate big rivers differently, still agree).
    // Sheet wash on unchannelled hillslopes is left to the runoff solver.
    // Below a few hundred square metres of catchment, overland flow spreads
    // as a sheet that vegetation holds; channels initiate above it.
    let catchment = drainage_area * max(stage.drainage_saturation, 1.0)
                  / (drainage_area + max(stage.drainage_saturation, 1.0));
    let channelised = smoothstep(12.0, 60.0, drainage_area);
    let fluvial_capacity = max(stage.fluvial_capacity, 0.0) * sqrt(catchment)
                         * steepest_descent * channelised;
    var fluvial_change = 0.0;
    if (sediment_in > fluvial_capacity)
    {
        // Slack water drops what it can no longer carry: fans at slope
        // breaks, alluvium on valley floors, fill in closed hollows.
        fluvial_change = (sediment_in - fluvial_capacity)
                       * clamp(stage.fluvial_deposition, 0.0, 1.0);
    }
    else
    {
        // Loose cover meets the deficit first; whatever remains is cut from
        // bedrock at the geology's resistance.
        let deficit = (fluvial_capacity - sediment_in) * max(stage.fluvial_erosion, 0.0);
        let from_loose = min(deficit, loose);
        var detach = from_loose * loose_erodibility + (deficit - from_loose) * rock_erodibility;
        // Incision slows as it deepens and stops at the configured depth, so
        // valleys deepen into V forms (the thermal pass opens their walls)
        // rather than slots.
        let excavation = max(base_height - bed_height, 0.0);
        let depth_room = clamp(1.0 - excavation / max(stage.maximum_incision, 0.01), 0.0, 1.0);
        detach = detach * depth_room * depth_room;
        // A bed never cuts below its lowest neighbour, so no pits open, nor
        // below the sea, so river mouths do not drown into inlets; one step
        // stays small against the 4 m cell.
        detach = min(detach, max(bed_height - lowest_neighbour, 0.0) * 0.5);
        detach = min(detach, max(bed_height - stage.sea_level - 0.5, 0.0));
        detach = min(detach, cell_size * 0.012);
        fluvial_change = -detach;
    }
    bed_height += fluvial_change;
    loose = max(loose + fluvial_change, 0.0);
    var sediment_out = max(sediment_in - fluvial_change, 0.0);

    // Fade the simulation into its immutable source terrain. The outermost
    // texel is exactly baseHeight, producing a stable collision/render guard.
    let edge_distance = f32(min(min(coord.x, size.x - 1 - coord.x),
                                min(coord.y, size.y - 1 - coord.y)));
    let guard_width = max(stage.guard_band_pixels, 1.0);
    let simulation_weight = smoothstep(0.0, guard_width, edge_distance);
    bed_height = mix(base_height, bed_height, simulation_weight);
    sediment *= simulation_weight;
    sediment_out *= simulation_weight;

    // The thermal pass reads every cell's bedrock resistance from drainage B
    // rather than evaluating the geology for each of its steep neighbours:
    // this is the same function of the same bedrock surface it writes here.
    let bedrock_resistance = clamp(erosionHardness(world_position, bed_height - loose)
                                   / GEOLOGY_HARDNESS_SCALE, 0.0, 1.0);

    return FragmentOutput(vec4<f32>(bed_height, max(sediment, 0.0), base_height, loose),
                          vec4<f32>(drainage_area, sediment_out, bedrock_resistance, 0.0));
}

// STAGE UNIFORMS:
//   Bind as ONE uniform buffer at @group(2) @binding(0), 88 bytes total:
//     offset  0  resolution:        vec2<f32>  // uResolution — simulation-domain
//                                              //   texel resolution; a component
//                                              //   < 1.0 falls back to the texture
//                                              //   dimensions (see gridSize)
//     offset  8  delta_time:        f32        // uDeltaTime
//     offset 12  cell_size:         f32        // uCellSize
//     offset 16  sea_level:         f32        // uSeaLevel
//     offset 20  erosion_rate:      f32        // uErosionRate
//     offset 24  deposition_rate:   f32        // uDepositionRate
//     offset 28  sediment_capacity: f32        // uSedimentCapacity
//     offset 32  minimum_slope:     f32        // uMinimumSlope
//     offset 36  maximum_erosion:   f32        // uMaximumErosion (<= 0 disables the
//                                              //   depth-fraction limit)
//     offset 40  transport_rate:    f32        // uTransportRate
//     offset 44  guard_band_pixels: f32        // uGuardBandPixels (floored at 1.0)
//     offset 48  brush_strength:    f32        // uBrushStrength (0..1)
//     offset 55  (pad 4 bytes for vec2 alignment)
//     offset 56  world_min:         vec2<f32>  // uWorldMin — world-space XZ corner
//                                              //   of texel (0, 0) of the padded
//                                              //   simulation domain
//     offset 64  fluvial_capacity:  f32
//     offset 68  fluvial_erosion:   f32
//     offset 72  fluvial_deposition: f32
//     offset 76  maximum_incision:  f32
//     offset 80  drainage_saturation: f32
//     offset 84  (struct rounds to 88 bytes)
//   Texture bindings (RGBA32F float textures, texel loads only, no filtering):
//     texture0    -> @group(1) @binding(0) texture_2d<f32>; sampler @binding(8) (unused)
//     uWaterState -> @group(1) @binding(1) texture_2d<f32>; sampler @binding(9) (unused)
//     uDrainage   -> @group(1) @binding(2) texture_2d<f32>; sampler @binding(10) (unused)
//     uRouting    -> @group(1) @binding(3) texture_2d<f32>; sampler @binding(11) (unused)
//   Outputs: @location(0) = updated terrain
//     (bed height, sediment, immutable base height, loose cover);
//     @location(1) = drainage (contributing area, sediment flux out,
//     resistance of the bedrock under the updated cover, 0).
//     Neither input may be an attachment fs_main renders to — ping-pong both
//     states as in the GL FBO version.