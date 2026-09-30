// Hydraulic erosion pass 1: update the four directional outflow fluxes.
// texture0:       previous flux (left, right, down, up)
// uTerrainState:  bed height, sediment, immutable base height, hardness
// uWaterState:    water depth, velocity x, velocity z, accumulated discharge
//
// PORT NOTES (GLSL erosion_flux.fs -> WGSL, see PORTING-SPEC.md):
// - Bindings follow spec section 3 (GL unit numbers preserved): texture0 ->
//   group 1/binding 0, uTerrainState -> binding 1, uWaterState -> binding 2;
//   samplers at unit + 8 (bindings 8, 9, 10). The three samplers are declared
//   but unused: every fetch in this pass is texelFetch/textureLoad, so the
//   WGSL contains no textureSample call; the Rust bind group still supplies
//   them because the declared entries appear in the reflected layout.
// - No vertical flip added or dropped: fragmentCoord() derives from
//   @builtin(position) (top-left pixel origin) instead of gl_FragCoord
//   (bottom-left origin). Per the spec's one-row-convention rule, every
//   texture, attachment and readback in the WGSL port is top-left origin, so
//   the outflow channel semantics (left/right/down/up = -x/+x/-y/+y in
//   texture space) stay internally consistent across all four erosion passes.
// - texelFetch -> textureLoad, textureSize(t, 0) -> textureDimensions(t).
// - No samples hoisted (this pass samples nothing inside per-pixel branches),
//   no inverse_* uniform fields added, no output channel dropped.
// - GLSL ternaries became select(false_value, true_value, condition); the
//   unused raylib interpolators fragTexCoord/fragColor are not carried over.

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
    atmosphere: vec4<f32>,       // daylight, moon intensity, hours, volumetric strength
    raymarch: vec4<f32>,         // shadows, volumetrics, reflections, quality (0/1/2)
    heightfield: vec4<f32>,      // world centre XZ, world span, texel size in metres
};
@group(0) @binding(0) var<uniform> globals: GlobalUniforms;

// Uniforms: uResolution, uDeltaTime, uCellSize, uGravity
struct StageUniforms {
    resolution: vec2<f32>,      // uResolution
    delta_time: f32,            // uDeltaTime
    cell_size: f32,             // uCellSize
    gravity: f32,               // uGravity
};
@group(2) @binding(0) var<uniform> stage: StageUniforms;

@group(1) @binding(0) var texture0: texture_2d<f32>;       // texture0: previous flux (left, right, down, up)
@group(1) @binding(1) var terrain_state: texture_2d<f32>;  // uTerrainState
@group(1) @binding(2) var water_state: texture_2d<f32>;    // uWaterState
@group(1) @binding(8) var tex0_sampler: sampler;
@group(1) @binding(9) var terrain_state_sampler: sampler;
@group(1) @binding(10) var water_state_sampler: sampler;

struct FsOutput {
    @location(0) final_color: vec4<f32>,   // out vec4 finalColor
};

fn gridSize() -> vec2<i32>
{
    let texture_resolution = vec2<i32>(textureDimensions(texture0));
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

fn fragmentCoord(frag_position: vec4<f32>) -> vec2<i32>
{
    let resolution = select(vec2<f32>(gridSize()), stage.resolution,
        stage.resolution.x >= 1.0 && stage.resolution.y >= 1.0);
    let uv = frag_position.xy / resolution;
    return clampCoord(vec2<i32>(floor(uv * vec2<f32>(gridSize()))));
}

fn fetchFlux(coord: vec2<i32>) -> vec4<f32>
{
    return textureLoad(texture0, clampCoord(coord), 0);
}

fn fetchTerrain(coord: vec2<i32>) -> vec4<f32>
{
    return textureLoad(terrain_state, clampCoord(coord), 0);
}

fn fetchWater(coord: vec2<i32>) -> vec4<f32>
{
    return textureLoad(water_state, clampCoord(coord), 0);
}

fn totalHeight(coord: vec2<i32>) -> f32
{
    let safe_coord = clampCoord(coord);
    return fetchTerrain(safe_coord).r + max(fetchWater(safe_coord).r, 0.0);
}

fn hydraulicGradient(coord: vec2<i32>, cell_size: f32) -> vec2<f32>
{
    // A Scharr derivative is much more rotationally symmetric than the four
    // cardinal differences used by the pipe model.  In particular, a valley
    // crossing a texel diagonally produces a diagonal downhill vector instead
    // of alternately choosing horizontal and vertical faces.
    let lower_left  = totalHeight(coord + vec2<i32>(-1,-1));
    let left        = totalHeight(coord + vec2<i32>(-1, 0));
    let upper_left  = totalHeight(coord + vec2<i32>(-1, 1));
    let lower       = totalHeight(coord + vec2<i32>( 0,-1));
    let upper       = totalHeight(coord + vec2<i32>( 0, 1));
    let lower_right = totalHeight(coord + vec2<i32>( 1,-1));
    let right       = totalHeight(coord + vec2<i32>( 1, 0));
    let upper_right = totalHeight(coord + vec2<i32>( 1, 1));

    let derivative = vec2<f32>(
        3.0*(upper_right - upper_left) + 10.0*(right - left)
            + 3.0*(lower_right - lower_left),
        3.0*(upper_left - lower_left) + 10.0*(upper - lower)
            + 3.0*(upper_right - lower_right)
    );
    return derivative/(32.0*cell_size);
}

fn safeDirection(vector: vec2<f32>) -> vec2<f32>
{
    let magnitude = length(vector);
    return select(vec2<f32>(0.0), vector/magnitude, magnitude > 0.000001);
}

@fragment
fn fs_main(@builtin(position) frag_position: vec4<f32>) -> FsOutput
{
    let coord = fragmentCoord(frag_position);
    let size = gridSize();

    let dt = max(stage.delta_time, 0.0);
    let cell_size = max(stage.cell_size, 0.0001);
    let gravity = max(stage.gravity, 0.0);
    let water_depth = max(fetchWater(coord).r, 0.0);
    let center_height = totalHeight(coord);

    let previous_flux = max(fetchFlux(coord), vec4<f32>(0.0));
    let cardinal_difference = vec4<f32>(
        center_height - totalHeight(coord + vec2<i32>(-1, 0)),
        center_height - totalHeight(coord + vec2<i32>( 1, 0)),
        center_height - totalHeight(coord + vec2<i32>( 0,-1)),
        center_height - totalHeight(coord + vec2<i32>( 0, 1))
    );

    // Nerthus droplets follow a continuous terrain gradient while retaining a
    // small amount of their previous direction.  Keep the conservative four-
    // pipe storage, but project that continuous direction onto its faces.  A
    // diagonal direction therefore feeds two faces rather than stair-stepping
    // down whichever cardinal neighbour happens to be lowest.
    let gradient = hydraulicGradient(coord, cell_size);
    let downhill_direction = safeDirection(-gradient);
    let previous_direction = safeDirection(fetchWater(coord).gb);
    const DIRECTION_INERTIA = 0.12;
    let steered_direction = safeDirection(
        downhill_direction*(1.0 - DIRECTION_INERTIA) +
        previous_direction*DIRECTION_INERTIA
    );

    let continuous_drop = length(gradient)*cell_size;
    let projected_difference = continuous_drop*vec4<f32>(
        -steered_direction.x, steered_direction.x,
        -steered_direction.y, steered_direction.y
    );
    // The direct face head remains the majority term, preserving the pressure
    // solver and allowing opposing flux to brake.  The projected term removes
    // its cardinal directional bias without introducing stochastic forcing.
    let height_difference = mix(cardinal_difference, projected_difference, 0.42);

    // Each channel is a water-depth flux (metres per second) through one face.
    var flux = max(vec4<f32>(0.0), previous_flux + (dt * gravity / cell_size) * height_difference);

    // The edge is a closed wall. This also removes stale outward flux at the edge.
    if (coord.x == 0)          { flux.r = 0.0; }
    if (coord.x == size.x - 1) { flux.g = 0.0; }
    if (coord.y == 0)          { flux.b = 0.0; }
    if (coord.y == size.y - 1) { flux.a = 0.0; }

    // Never allow this step to send more water than the cell currently contains.
    let total_outflow = flux.r + flux.g + flux.b + flux.a;
    if (dt > 0.0 && total_outflow > 0.0)
    {
        let available_rate = water_depth / dt;
        flux *= min(1.0, available_rate / total_outflow);
    }
    else if (water_depth <= 0.0)
    {
        flux = vec4<f32>(0.0);
    }

    var output: FsOutput;
    output.final_color = flux;
    return output;
}

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

// STAGE UNIFORMS:
//   resolution: vec2<f32> - uResolution: pixel resolution of the erosion grid;
//       either component < 1.0 means "fall back to texture0's dimensions".
//       Struct offsets: resolution 0, delta_time 8, cell_size 12, gravity 16
//       (padded to 24 bytes by the vec2 alignment).
//   delta_time: f32 - uDeltaTime: simulation time step in seconds.
//   cell_size:  f32 - uCellSize: cell spacing in metres.
//   gravity:    f32 - uGravity: gravitational acceleration for the pipe model.