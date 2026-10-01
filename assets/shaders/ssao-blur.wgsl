// SSAO bilateral blur: 5x5 spatial/depth/normal weighted average of the raw
// SSAO buffer; sky texels (position alpha < 0.5) pass through as white.
// texture0: Raw SSAO.
// texture1: View-space position, alpha zero for sky.
// texture2: Encoded view-space normal.
//
// PORT NOTES (GLSL ssao_blur.fs -> WGSL, see PORTING-SPEC.md):
// - Bindings follow spec section 3: texture0 -> group 1/binding 0, texture1 ->
//   binding 1, texture2 -> binding 2; samplers at unit + 8 (bindings 8, 9, 10).
// - Samples hoisted per spec sections 5/6 (WGSL uniform control flow forbids
//   textureSample under per-pixel branches; naga's analysis keeps the
//   disruption alive after a conditional early return):
//   * The centre texture2 (normal) and texture0 (AO) samples, which the GLSL
//     fetched after the sky branch / in the trailing ternary, are pre-sampled
//     unconditionally before the sky branch.
//   * Inside the tap loop, the texture2 and texture0 samples the GLSL fetched
//     after `if (samplePosition.a < 0.5) continue;` are pre-sampled before
//     that branch and the selection kept inside.
//   All extra unconditional samples have no side effects and their results are
//   either used exactly as before or discarded, so output is identical.
// - The GLSL returned white for sky texels BEFORE running the blur; here the
//   sky early return is placed after the blur computation (nothing with
//   textureSample may follow it), so sky texels run the loop against discarded
//   values and then return vec4(1.0) with the same early return. The GLSL
//   ternary likewise became select(), which evaluates both operands. Math and
//   output ordering are unchanged; only discarded work was added.
// - fragTexCoord -> position.xy * globals.viewport.zw (spec section 4). No
//   vertical flip added or dropped: every texture and attachment in the WGSL
//   port shares one top-left origin, and the 5x5 neighbour window is symmetric
//   in y, so no GL vs WGSL handedness asymmetry arises on this pass.
// - No inverse_* uniform fields added, no output channel dropped; constants
//   (0.28 spatial falloff, 0.5 sky threshold, 0.00001 weight fallback)
//   verbatim.

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
};
@group(0) @binding(0) var<uniform> globals: GlobalUniforms;

// Uniforms: uTexelSize, uDepthSharpness, uNormalSharpness
struct StageUniforms {
    texel_size: vec2<f32>,      // uTexelSize
    depth_sharpness: f32,       // uDepthSharpness
    normal_sharpness: f32,      // uNormalSharpness
};
@group(2) @binding(0) var<uniform> stage: StageUniforms;

@group(1) @binding(0) var texture0: texture_2d<f32>;       // Raw SSAO.
@group(1) @binding(1) var texture1: texture_2d<f32>;       // View-space position, alpha zero for sky.
@group(1) @binding(2) var texture2: texture_2d<f32>;       // Encoded view-space normal.
@group(1) @binding(8) var tex0_sampler: sampler;
@group(1) @binding(9) var tex1_sampler: sampler;
@group(1) @binding(10) var tex2_sampler: sampler;

struct FsOutput {
    @location(0) final_color: vec4<f32>,   // out vec4 finalColor
};

@fragment
fn fs_main(@builtin(position) frag_position: vec4<f32>) -> FsOutput
{
    let frag_tex_coord = frag_position.xy*globals.viewport.zw;   // fragTexCoord

    let centre_position = textureSample(texture1, tex1_sampler, frag_tex_coord);
    // Pre-sampled unconditionally: the sky branch below is per-pixel, and WGSL
    // forbids textureSample inside non-uniform control flow.
    let centre_normal = normalize(textureSample(texture2, tex2_sampler, frag_tex_coord).xyz*2.0 - 1.0);
    let centre_ao = textureSample(texture0, tex0_sampler, frag_tex_coord).r;

    var weighted_ao = 0.0;
    var total_weight = 0.0;

    for (var y: i32 = -2; y <= 2; y = y + 1)
    {
        for (var x: i32 = -2; x <= 2; x = x + 1)
        {
            let offset = vec2<f32>(f32(x), f32(y));
            let sample_uv = clamp(frag_tex_coord + offset*stage.texel_size, vec2<f32>(0.0), vec2<f32>(1.0));
            let sample_position = textureSample(texture1, tex1_sampler, sample_uv);
            // Pre-sampled before the sky `continue`: WGSL forbids textureSample
            // after a per-pixel `continue` inside a loop; the selection stays
            // inside and the GLSL never used these samples for skipped taps.
            let sample_normal_texel = textureSample(texture2, tex2_sampler, sample_uv);
            let sample_ao = textureSample(texture0, tex0_sampler, sample_uv).r;
            if (sample_position.a < 0.5) { continue; }

            let sample_normal = normalize(sample_normal_texel.xyz*2.0 - 1.0);
            let spatial_weight = exp(-dot(offset, offset)*0.28);
            let depth_weight = exp(-abs(sample_position.z - centre_position.z)*
                                    max(stage.depth_sharpness, 0.0));
            let normal_weight = pow(max(dot(centre_normal, sample_normal), 0.0),
                                     max(stage.normal_sharpness, 0.0));
            let weight = spatial_weight*depth_weight*normal_weight;

            weighted_ao += sample_ao*weight;
            total_weight += weight;
        }
    }

    let ao = select(centre_ao, weighted_ao/total_weight, total_weight > 0.00001);

    if (centre_position.a < 0.5)
    {
        var output: FsOutput;
        output.final_color = vec4<f32>(1.0);
        return output;
    }

    var output: FsOutput;
    output.final_color = vec4<f32>(vec3<f32>(ao), 1.0);
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
//   texel_size:       vec2<f32> - uTexelSize: 1/width, 1/height in UV units of
//       the SSAO/position/normal targets; each tap offset is offset*texel_size.
//       Struct offsets: texel_size 0, depth_sharpness 8, normal_sharpness 12
//       (total size 16 bytes).
//   depth_sharpness:  f32 - uDepthSharpness: weight exponent scale on the
//       view-space z difference; max(depth_sharpness, 0.0) applied in the
//       shader.
//   normal_sharpness: f32 - uNormalSharpness: exponent on the dot of the two
//       decoded view-space normals; max(normal_sharpness, 0.0) applied in the
//       shader.