// PORT NOTES (erosion_water.fs -> erosion-water.wgsl):
// - gl_FragCoord.xy -> @builtin(position).xy, coordinates ported 1:1. Every
//   texture, attachment and readback in this port shares one top-left row
//   convention, and fragmentCoord here is pure cell-index math on those
//   same-orientation textures, so GL's bottom-left origin never surfaces as a
//   behavioral difference; no screen-pattern handedness use exists here.
// - No render-texture uv flips existed in this file to drop.
// - textureSize(t, 0) -> textureDimensions(t, 0); the u32 result is converted
//   to i32 immediately (values identical).
// - GLSL ternaries -> WGSL select(): select evaluates both operands, but every
//   discarded operand is pure float math with no trap semantics, so there is
//   no behavioral change. No samples sit inside branches, so no hoisting was
//   needed (the depth<0.0001 and sea-level branches load nothing).
// - The bound samplers (GL unit + 8) are declared per the binding layout but
//   are unused: the GLSL only ever texelFetch-es these textures, so the Rust
//   side may bind point or linear samplers for them.
// - fragTexCoord / fragColor vertex inputs were declared-but-unused in the
//   GLSL; the fullscreen-triangle stage supplies no vertex attributes.
// - No matrix inverse is needed; no inverse_* stage fields were added.
// - Local variables renamed snake_case per repo convention; function names,
//   constants, epsilon values, math and comments are verbatim 1:1.

// Hydraulic erosion pass 2: conserve water and derive its horizontal velocity.
// texture0:       previous water (depth, velocity x, velocity z, discharge)
// uFluxState:     outflow flux (left, right, down, up)
// uTerrainState:  bed height, sediment, immutable base height, hardness

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
    storm: vec4<f32>, // precipitation bias, type override, elapsed time, local gust
    lightning: vec4<f32>, // strike world xyz, HDR flash
    lightning_meta: vec4<f32>, // seed, age, bolt top, local thunder
};
@group(0) @binding(0) var<uniform> globals: GlobalUniforms;

// StageUniforms: uResolution, uDeltaTime, uCellSize, uRain, uEvaporation, uSeaLevel
struct StageUniforms {
    resolution: vec2<f32>,   // uResolution: retained footprint in texels (readback crop)
    delta_time: f32,         // uDeltaTime: simulation timestep
    cell_size: f32,          // uCellSize: world size of one grid cell
    rain: f32,               // uRain: rain volume per second
    evaporation: f32,        // uEvaporation: evaporation rate per second
    sea_level: f32,          // uSeaLevel: ocean level, sink for simulation water below it
};
@group(2) @binding(0) var<uniform> stage: StageUniforms;

// GL texture units preserved: texture0 -> 0, uFluxState -> 1, uTerrainState -> 2
// (samplers sit at unit + 8; all three are unused by this pass, see PORT NOTES)
@group(1) @binding(0) var texture0: texture_2d<f32>;
@group(1) @binding(8) var texture0_sampler: sampler;
@group(1) @binding(1) var uFluxState: texture_2d<f32>;
@group(1) @binding(9) var uFluxState_sampler: sampler;
@group(1) @binding(2) var uTerrainState: texture_2d<f32>;
@group(1) @binding(10) var uTerrainState_sampler: sampler;

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

fn fragmentCoord(frag_coord: vec2<f32>) -> vec2<i32>
{
    let resolution = select(
        vec2<f32>(gridSize()),
        stage.resolution,
        (stage.resolution.x >= 1.0 && stage.resolution.y >= 1.0)
    );
    let uv = frag_coord / resolution;
    return clampCoord(vec2<i32>(floor(uv * vec2<f32>(gridSize()))));
}

fn fetchWater(coord: vec2<i32>) -> vec4<f32>
{
    return textureLoad(texture0, clampCoord(coord), 0);
}

fn fetchFlux(coord: vec2<i32>) -> vec4<f32>
{
    return max(textureLoad(uFluxState, clampCoord(coord), 0), vec4<f32>(0.0));
}

fn fetchTerrain(coord: vec2<i32>) -> vec4<f32>
{
    return textureLoad(uTerrainState, clampCoord(coord), 0);
}

fn surfaceHeight(coord: vec2<i32>) -> f32
{
    let safe_coord = clampCoord(coord);
    return fetchTerrain(safe_coord).r + max(fetchWater(safe_coord).r, 0.0);
}

fn sampleSurface(texel_position: vec2<f32>) -> f32
{
    let position = clamp(texel_position, vec2<f32>(0.0), vec2<f32>(gridSize() - vec2<i32>(1)));
    let lower = vec2<i32>(floor(position));
    let upper = clampCoord(lower + vec2<i32>(1));
    let fraction = fract(position);

    let height00 = surfaceHeight(lower);
    let height10 = surfaceHeight(vec2<i32>(upper.x, lower.y));
    let height01 = surfaceHeight(vec2<i32>(lower.x, upper.y));
    let height11 = surfaceHeight(upper);
    return mix(mix(height00, height10, fraction.x),
               mix(height01, height11, fraction.x), fraction.y);
}

fn hydraulicGradient(coord: vec2<i32>, cellSize: f32) -> vec2<f32>
{
    let lower_left  = surfaceHeight(coord + vec2<i32>(-1,-1));
    let left        = surfaceHeight(coord + vec2<i32>(-1, 0));
    let upper_left  = surfaceHeight(coord + vec2<i32>(-1, 1));
    let lower       = surfaceHeight(coord + vec2<i32>( 0,-1));
    let upper       = surfaceHeight(coord + vec2<i32>( 0, 1));
    let lower_right = surfaceHeight(coord + vec2<i32>( 1,-1));
    let right       = surfaceHeight(coord + vec2<i32>( 1, 0));
    let upper_right = surfaceHeight(coord + vec2<i32>( 1, 1));

    let derivative = vec2<f32>(
        3.0*(upper_right - upper_left) + 10.0*(right - left)
            + 3.0*(lower_right - lower_left),
        3.0*(upper_left - lower_left) + 10.0*(upper - lower)
            + 3.0*(upper_right - lower_right)
    );
    return derivative/(32.0*cellSize);
}

fn safeDirection(vector: vec2<f32>) -> vec2<f32>
{
    let magnitude = length(vector);
    return select(vec2<f32>(0.0), vector/magnitude, magnitude > 0.000001);
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

@fragment
fn fs_main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let coord = fragmentCoord(position.xy);
    let size = gridSize();

    // Simulate the complete padded domain. The retained footprint is only
    // a readback crop, never a hydraulic or erosion boundary.
    let old_water = fetchWater(coord);
    let outgoing = fetchFlux(coord);

    let left_flux  = fetchFlux(coord + vec2<i32>(-1, 0));
    let right_flux = fetchFlux(coord + vec2<i32>( 1, 0));
    let down_flux  = fetchFlux(coord + vec2<i32>( 0,-1));
    let up_flux    = fetchFlux(coord + vec2<i32>( 0, 1));

    // The neighbor channel facing this cell is its contribution to our inflow.
    let from_left  = select(0.0, left_flux.g,  coord.x > 0);
    let from_right = select(0.0, right_flux.r, coord.x < size.x - 1);
    let from_down  = select(0.0, down_flux.a,  coord.y > 0);
    let from_up    = select(0.0, up_flux.b,    coord.y < size.y - 1);
    let total_inflow = from_left + from_right + from_down + from_up;
    let total_outflow = outgoing.r + outgoing.g + outgoing.b + outgoing.a;

    let dt = max(stage.delta_time, 0.0);
    var depth = max(0.0, old_water.r + max(stage.rain, 0.0) * dt
        + (total_inflow - total_outflow) * dt);
    depth *= max(0.0, 1.0 - max(stage.evaporation, 0.0) * dt);

    // Face fluxes determine the transported volume and speed.  They do not
    // determine direction on their own: doing that locks sediment advection to
    // the four pipe axes even when both faces approximate one diagonal stream.
    let signed_flux = 0.5 * vec2<f32>(
        outgoing.g + from_left - outgoing.r - from_right,
        outgoing.a + from_down - outgoing.b - from_up
    );
    let cell_size = max(stage.cell_size, 0.0001);
    let mean_depth = max(0.5 * (max(old_water.r, 0.0) + depth), 0.0001);
    let flux_velocity = signed_flux*cell_size/mean_depth;

    // Follow Nerthus's low-inertia continuous-gradient steering.  Flux remains
    // exactly conservative, while this arbitrary-angle velocity is retained in
    // the water state and used by the sediment backtrace in the next pass.
    let downhill_direction = safeDirection(-hydraulicGradient(coord, cell_size));
    let flux_direction = safeDirection(flux_velocity);
    let previous_direction = safeDirection(old_water.gb);
    var target_direction = safeDirection(
        downhill_direction*0.66 + flux_direction*0.22 + previous_direction*0.12
    );
    if (length(target_direction) < 0.000001)
    {
        target_direction = flux_direction;
    }

    // Nerthus updates droplet velocity from the elevation change it actually
    // traverses.  Apply the same energy bound to the pipe-derived speed.  This
    // prevents vanishingly shallow water from reporting enormous velocities,
    // which otherwise causes narrow channels to excavate far too deeply.
    let previous_speed = length(old_water.gb);
    let next_surface = sampleSurface(vec2<f32>(coord) + target_direction);
    let height_change = next_surface - surfaceHeight(coord);
    const gravity: f32 = 9.81;
    let energy_speed = sqrt(max(0.0,
        previous_speed*previous_speed - 2.0*gravity*height_change));
    let energy_limit = max(0.75, energy_speed + 0.5);
    let target_speed = min(length(flux_velocity), energy_limit);
    let speed_blend = clamp(dt*7.0, 0.0, 1.0);
    let speed = mix(previous_speed, target_speed, speed_blend);
    var velocity = target_direction*speed;
    if (depth < 0.0001)
    {
        velocity = vec2<f32>(0.0);
    }

    let step_discharge = 0.5 * (total_inflow + total_outflow) * dt;
    let accumulated_discharge = max(old_water.a, 0.0) + max(step_discharge, 0.0);

    // The visible ocean is supplied separately at sea level. Simulation water
    // reaching an ocean-bed cell is therefore a sink instead of accumulating.
    let base_height = fetchTerrain(coord).b;
    if (base_height < stage.sea_level)
    {
        depth = 0.0;
        velocity = vec2<f32>(0.0);
    }

    return vec4<f32>(depth, velocity, accumulated_discharge);  // finalColor
}

// STAGE UNIFORMS:
//   resolution:  vec2<f32> — uResolution: retained footprint in texels (readback crop);
//                            values < 1.0 on either axis fall back to the texture size
//   delta_time:  f32       — uDeltaTime: simulation timestep in seconds (shader clamps to >= 0)
//   cell_size:   f32       — uCellSize: world size of one grid cell (shader clamps to >= 0.0001)
//   rain:        f32       — uRain: rain volume per second (shader clamps to >= 0)
//   evaporation: f32       — uEvaporation: evaporation rate per second (shader clamps to >= 0)
//   sea_level:   f32       — uSeaLevel: ocean level; bed cells below it zero depth and velocity
// Pack in this field order for bytemuck (vec2 then 5 x f32).