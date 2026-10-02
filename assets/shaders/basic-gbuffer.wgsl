// PORT NOTES (basic_gbuffer.vs + basic_gbuffer.fs -> basic-gbuffer.wgsl):
// - Combined pair per the file map: vs_main (geometry: ocean plane, debug
//   shapes) and fs_main (basic G-buffer fill, 3 colour outputs) live in this
//   one file.
// - WGSL has no inverse(): the GLSL `transpose(inverse(mat3(matView*matModel)))`
//   now reads stage.inverse_model_view (mat4x4<f32>), which the Rust side fills
//   with inverse(matView*matModel). For an affine matrix (last row 0,0,0,1)
//   mat3(inverse(M)) == inverse(mat3(M)), so transposing the upper-left 3x3 of
//   the supplied inverse is exactly the GLSL normal matrix. matModel itself is
//   stage.model.
// - matView/matProjection moved into GlobalUniforms (globals.view /
//   globals.projection). globals.projection already maps GL z to the [0,1]
//   NDC range on the CPU; this stage multiplies it exactly as the GLSL did.
// - Deviations beyond the inverse field: none. No textures are bound (the GLSL
//   samples nothing), so there are no hoisted samples; there are no blit flips
//   to drop and no gl_FragCoord-style origin math (neither stage reads
//   screen-space coordinates).
// - Vertex input vertexTexCoord (@location(1)) is declared but unused, exactly
//   as the GLSL declared it; kept so the vertex layout matches raylib's default
//   attribute order (0 position, 1 texcoord, 2 normal, 3 color). Drop it here
//   AND from the pipeline vertex layout together if the basic mesh carries no
//   UVs. The Rust BasicStageUniforms in src/render/mod.rs predates this file
//   and must gain inverse_model_view to match the WGSL layout (see the STAGE
//   UNIFORMS block for offsets).

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

// basic_gbuffer.vs uniforms: matModel alone (matView/matProjection live in
// GlobalUniforms). basic_gbuffer.fs uniforms: uColor, uRoughness.
// inverse_model_view carries inverse(matView*matModel) because the GLSL called
// inverse() and WGSL has none; see PORT NOTES.
struct StageUniforms {                 // matModel, inverse(matView*matModel), uColor, uRoughness
    model: mat4x4<f32>,                // matModel
    inverse_model_view: mat4x4<f32>,   // inverse(matView*matModel), Rust-supplied
    // Base colour for simple deferred meshes such as the ocean and border walls.
    color: vec4<f32>,                  // uColor
    // Surface roughness written to the g-buffer; low
    // values give the ocean a calm sun glint.
    roughness: f32,                    // uRoughness
};
@group(2) @binding(0) var<uniform> stage: StageUniforms;

struct VsOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) fragPositionView: vec3<f32>,  // fragPositionView
    @location(1) fragNormalView: vec3<f32>,    // fragNormalView
    @location(2) fragColor: vec4<f32>,         // fragColor
};

@vertex
fn vs_main(
    @location(0) vertexPosition: vec3<f32>,    // raylib vertexPosition
    @location(1) vertexTexCoord: vec2<f32>,    // raylib vertexTexCoord (unused, kept for layout parity)
    @location(2) vertexNormal: vec3<f32>,      // raylib vertexNormal
    @location(3) vertexColor: vec4<f32>,       // raylib vertexColor
) -> VsOutput
{
    let worldPosition4 = stage.model*vec4<f32>(vertexPosition, 1.0);
    let viewPosition4 = globals.view*worldPosition4;
    // GLSL computed transpose(inverse(mat3(matView*matModel)));
    // stage.inverse_model_view is inverse(matView*matModel) supplied by the CPU
    // because WGSL has no inverse(). For an affine matrix (last row 0,0,0,1)
    // mat3(inverse(M)) == inverse(mat3(M)), so the transposed upper-left 3x3 is
    // exactly the GLSL normal matrix.
    let normalModelView = transpose(mat3x3<f32>(stage.inverse_model_view[0].xyz,
                                                stage.inverse_model_view[1].xyz,
                                                stage.inverse_model_view[2].xyz));

    var output: VsOutput;
    output.fragPositionView = viewPosition4.xyz;
    output.fragNormalView = normalize(normalModelView*vertexNormal);
    output.fragColor = vertexColor;
    output.position = globals.projection*viewPosition4;
    return output;
}

struct FsInput {
    @location(0) fragPositionView: vec3<f32>,  // fragPositionView
    @location(1) fragNormalView: vec3<f32>,    // fragNormalView
    @location(2) fragColor: vec4<f32>,         // fragColor
};

// G-buffer channels: position (view space), normal (encoded), albedo + alpha.
struct FsOutput {
    @location(0) gPosition: vec4<f32>,  // gPosition
    @location(1) gNormal: vec4<f32>,    // gNormal
    @location(2) gAlbedo: vec4<f32>,    // gAlbedo
};

@fragment
fn fs_main(frag: FsInput) -> FsOutput
{
    let normalView = normalize(frag.fragNormalView);
    var output: FsOutput;
    output.gPosition = vec4<f32>(frag.fragPositionView, 1.0);
    output.gNormal = vec4<f32>(normalView*0.5 + 0.5, clamp(stage.roughness, 0.0, 1.0));
    output.gAlbedo = vec4<f32>(sqrt(max(stage.color.rgb*frag.fragColor.rgb, vec3<f32>(0.0))*0.4),
                               stage.color.a*frag.fragColor.a);
    return output;
}

// STAGE UNIFORMS:
//   model              (mat4x4<f32>) — matModel: model transform of the basic
//                                     mesh (ocean plane, debug shapes)
//   inverse_model_view (mat4x4<f32>) — inverse(matView*matModel), CPU-computed
//                                     (WGSL has no inverse()); the shader
//                                     transposes its upper-left 3x3
//   color              (vec4<f32>)   — uColor: base colour for simple deferred
//                                     meshes such as the ocean and border walls
//   roughness          (f32)         — uRoughness: surface roughness written to
//                                     gNormal.a (clamped to [0,1] in the shader)
// Rust fill (uniform, 160 bytes, WGSL uniform-address alignment):
//   [0]   model: [f32; 16] (column-major)
//   [64]  inverse_model_view: [f32; 16] (column-major inverse of matView*matModel)
//   [128] color: [f32; 4]
//   [144] roughness
//   [148..160] pad (struct size rounds up to 160)
//   NOTE: BasicStageUniforms in src/render/mod.rs (96 bytes, no inverse field)
//   predates this layout and must be updated to match. Compute inverse_model_view
//   per draw as inverse(globals.view * model).
// Textures (group 1): none — the GLSL samples nothing, so no texture/sampler
//   bindings exist in this file. Vertex input:
//   @location(0) vertexPosition (vec3), @location(1) vertexTexCoord (vec2,
//   unused, declared for raylib parity), @location(2) vertexNormal (vec3),
//   @location(3) vertexColor (vec4). vs_main outputs: @location(0)
//   fragPositionView (view-space xyz), @location(1) fragNormalView (view-space
//   normal), @location(2) fragColor (vertex colour). Fragment outputs:
//   @location(0) gPosition = vec4(view-space position, 1.0), @location(1)
//   gNormal = vec4(normal*0.5+0.5, clamp(uRoughness, 0, 1)), @location(2)
//   gAlbedo = vec4(sqrt(max(uColor.rgb*fragColor.rgb, 0.0)*0.4) [sRGB-style
//   gamma encode as in the GLSL], uColor.a*fragColor.a).
// group 0 binding 0 = shared GlobalUniforms.