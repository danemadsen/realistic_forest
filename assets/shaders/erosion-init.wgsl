// PORT NOTES (erosion_init.fs -> erosion-init.wgsl):
// - Single RGBA32F output at @location(0), exactly as the GLSL wrote it
//   (`finalColor`); every uMode branch writes that one output slot (mode 0
//   stamps terrain state, mode 2 stamps the CPU-routed drainage area, any
//   other mode clears to 0.0). Terrain A now carries the initial loose cover;
//   hardness is evaluated from the shared world-keyed geology by each pass.
// - No hoisted samples: the GLSL only texelFetches, and textureLoad carries no
//   WGSL uniform-control-flow restriction. No dropped blit flips (the source
//   has none). No `inverse_*` fields needed (the source inverts no matrices).
// - texture0 is never filtered in the GLSL (texelFetch only); the sampler at
//   binding 8 is declared per spec's texture+sampler-pair rule but unused.
//   Bind a non-filtering sampler (or leave it unbound if the Rust layout
//   drops unused bindings).
// - fragTexCoord / fragColor interpolators were declared but unused in the
//   GLSL; the fullscreen-tri vertex stage supplies no interpolators.
// - Handedness: GL counted rows from the bottom (gl_FragCoord.y); WGSL @builtin
//   position counts from the top. Texel (0, 0) of the state textures is now
//   the first stored row, so uWorldMin must be given as the world corner of
//   THAT texel — if the CPU derived it for GL's bottom-origin target it needs
//   the opposite Z edge. The shader math itself is unchanged.

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

struct StageUniforms {            // uMode, uWorldMin, uCellSize
    mode: i32,                    // uMode: 0 = stamp terrain state (initial loose cover),
                                  // 2 = stamp drainage area, any other value = clear
    world_min: vec2<f32>,         // uWorldMin: world-space XZ corner of texel (0, 0)
    cell_size: f32,               // uCellSize: world units per erosion texel
};
@group(2) @binding(0) var<uniform> stage: StageUniforms;

@group(1) @binding(0) var tex0: texture_2d<f32>;  // texture0: base height (R) and CPU drainage area (G), RGBA32F, texelFetch only
@group(1) @binding(8) var tex0_sampler: sampler;  // declared per spec; the GLSL never filters this texture

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

fn baseHeightAt(coord: vec2<i32>, size: vec2<i32>) -> f32
{
    return textureLoad(tex0, clamp(coord, vec2<i32>(0), size - vec2<i32>(1)), 0).r;
}

@fragment
fn fs_main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32>
{
    let size = vec2<i32>(textureDimensions(tex0, 0));
    let coord = clamp(vec2<i32>(position.xy), vec2<i32>(0), size - vec2<i32>(1));
    let base = textureLoad(tex0, coord, 0);
    let height = base.r;
    if (stage.mode == 0)
    {
        // Spawn protection belongs to world spawn, not to the centre of every
        // erosion scratch tile. uWorldMin is the world-space XZ corner of texel
        // (0, 0); the half-cell offset addresses texel centres.
        let worldPosition = stage.world_min + (vec2<f32>(coord) + vec2<f32>(0.5)) * stage.cell_size;
        // Hardness is no longer stamped: every pass evaluates the shared,
        // world-keyed geology at the bedrock surface it actually exposes.
        // Terrain A instead carries the loose cover (soil, regolith, talus
        // and alluvium) above that bedrock. Soil mantles gentle ground,
        // thins with slope and on resistant rock, and is absent where a face
        // is already steeper than loose material can stand (~38 degrees).
        let cell = max(stage.cell_size, 0.0001);
        let gradient = vec2<f32>(
            baseHeightAt(coord + vec2<i32>(1, 0), size) - baseHeightAt(coord - vec2<i32>(1, 0), size),
            baseHeightAt(coord + vec2<i32>(0, 1), size) - baseHeightAt(coord - vec2<i32>(0, 1), size))
            / (2.0 * cell);
        let steepness = length(gradient);
        let resistance = geologyResistance(worldPosition, height);
        let soilPatch = geologyNoise(worldPosition / 41.0 + vec2<f32>(5.3, -27.7));
        let looseCover = 1.35 * (1.0 - smoothstep(0.42, 0.78, steepness))
                       * mix(1.0, 0.45, resistance) * mix(0.70, 1.30, soilPatch);
        return vec4<f32>(height, 0.0, height, looseCover);
    }
    else if (stage.mode == 2)
    {
        // The CPU routes the base surface's drainage before the tile starts,
        // so stream power acts on whole catchments from the first iteration.
        return vec4<f32>(max(base.g, 0.0), 0.0, 0.0, 0.0);
    }
    else
    {
        return vec4<f32>(0.0);
    }
}

// STAGE UNIFORMS:
//   mode      : i32        — uMode: 0 = stamp terrain state (height, 0, height, loose cover),
//                              2 = stamp drainage (area, 0, 0, 0) from texture0.g,
//                              any other value clears the tile to vec4(0.0)
//   world_min : vec2<f32>  — uWorldMin: world-space XZ corner of texel (0, 0) of the state texture
//   cell_size : f32        — uCellSize: world units per erosion texel
// Rust fill (uniform, 32 bytes, WGSL uniform-address alignment):
//   [0] mode: i32, [4] pad: f32, [8] world_min: [f32; 2], [16] cell_size: f32, [20] pad: [f32; 3]
// Textures (group 1): binding 0 = base height (R) + drainage area (G) RGBA32F (texelFetch-only; bind a
// non-filtering sampler at binding 8 for layout parity), group 0 binding 0 =
// shared GlobalUniforms.