// PORT NOTES (erosion_init.fs -> erosion-init.wgsl):
// - Single RGBA32F output at @location(0), exactly as the GLSL wrote it
//   (`finalColor`); both uMode branches write that one output slot (mode 0
//   stamps terrain state, any other mode clears to 0.0).
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
    mode: i32,                    // uMode: 0 = stamp terrain state (geology hardness), any other value = clear
    world_min: vec2<f32>,         // uWorldMin: world-space XZ corner of texel (0, 0)
    cell_size: f32,               // uCellSize: world units per erosion texel
};
@group(2) @binding(0) var<uniform> stage: StageUniforms;

@group(1) @binding(0) var tex0: texture_2d<f32>;  // texture0: base height RGBA32F (texelFetch only)
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

fn hash12(position: vec2<f32>) -> f32
{
    var p: vec3<f32> = fract(vec3<f32>(position.xyx) * 0.1031);
    p += dot(p, p.yzx + 33.33);
    return fract((p.x + p.y) * p.z);
}

fn valueNoise(position: vec2<f32>) -> f32
{
    let cell = floor(position);
    var fraction: vec2<f32> = fract(position);
    fraction = fraction * fraction * (3.0 - 2.0 * fraction);
    let a = hash12(cell);
    let b = hash12(cell + vec2<f32>(1.0, 0.0));
    let c = hash12(cell + vec2<f32>(0.0, 1.0));
    let d = hash12(cell + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, fraction.x), mix(c, d, fraction.x), fraction.y);
}

@fragment
fn fs_main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32>
{
    let size = vec2<i32>(textureDimensions(tex0, 0));
    let coord = clamp(vec2<i32>(position.xy), vec2<i32>(0), size - vec2<i32>(1));
    let height = textureLoad(tex0, coord, 0).r;
    if (stage.mode == 0)
    {
        // Spawn protection belongs to world spawn, not to the centre of every
        // erosion scratch tile. uWorldMin is the world-space XZ corner of texel
        // (0, 0); the half-cell offset addresses texel centres.
        let worldPosition = stage.world_min + (vec2<f32>(coord) + vec2<f32>(0.5)) * stage.cell_size;
        // World-keyed geology avoids the horizontal contour bands produced by
        // deriving hardness from height alone. Its two scales are deterministic
        // in every overlap and expose smaller resistant ribs for runoff to turn
        // around, which helps branch channels without random per-tile seams.
        let broadRock = valueNoise(worldPosition / 92.0 + vec2<f32>(31.7, -18.2));
        let fineRock = valueNoise(worldPosition / 27.0 + vec2<f32>(-73.1, 46.4));
        let geology = broadRock * 0.72 + fineRock * 0.28;
        let hardness = smoothstep(0.25, 0.82, geology) * 0.56;
        let spawnProtection = 1.0 - smoothstep(90.0, 150.0, length(worldPosition));
        let materialHardness = max(hardness, spawnProtection * 0.96);
        return vec4<f32>(height, 0.0, height, materialHardness);
    }
    else
    {
        return vec4<f32>(0.0);
    }
}

// STAGE UNIFORMS:
//   mode      : i32        — uMode: 0 = stamp terrain state (height, height, geology hardness),
//                              any other value clears the tile to vec4(0.0)
//   world_min : vec2<f32>  — uWorldMin: world-space XZ corner of texel (0, 0) of the state texture
//   cell_size : f32        — uCellSize: world units per erosion texel
// Rust fill (uniform, 32 bytes, WGSL uniform-address alignment):
//   [0] mode: i32, [4] pad: f32, [8] world_min: [f32; 2], [16] cell_size: f32, [20] pad: [f32; 3]
// Textures (group 1): binding 0 = base height RGBA32F (texelFetch-only; bind a
// non-filtering sampler at binding 8 for layout parity), group 0 binding 0 =
// shared GlobalUniforms.