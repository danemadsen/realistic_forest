// PORT NOTES (ssao.fs -> ssao.wgsl):
// - uProjection is replaced by globals.projection from the shared GlobalUniforms
//   buffer. The ported projection keeps the GL rows 0, 1 and 3 (x, y and w)
//   unchanged; only row 2 (z) was converted to the [0, 1] NDC range, and this
//   shader reads only projected.xy and projected.w, so the view->clip sample
//   math is ported as-is. The GLSL never inverts a matrix, so no inverse_*
//   stage-uniform field was needed.
// - Every texture() call became textureSampleLevel(.., 0.0): the per-pixel sky
//   early-out and the per-pixel sampleUV inside the loop put all later samples
//   in non-uniform control flow, where WGSL forbids implicit-derivative
//   textureSample, and the in-loop neighbour lookup cannot be hoisted at all
//   (its UV does not exist before the loop). All three inputs are
//   full-resolution, mip-free render targets sampled 1:1 (texture0/texture1)
//   or point-sampled with repeating addressing (texture2), so GL's implicit
//   derivatives already selected level 0 — results are unchanged.
// - No blit flips existed to drop and the GLSL never reads gl_FragCoord, so
//   fragTexCoord becomes position.xy * globals.viewport.zw per spec. sampleUV
//   maps NDC*0.5+0.5 to a UV and NEEDS the y flip the GLSL did not: clip-space
//   y counts up while the WGSL port's framebuffer rows count down from the top,
//   so it is written as (0.5 + 0.5*x, 0.5 - 0.5*y). See the note at the
//   sampleUV line.
// - The remaining scalar/vector uniforms (uScreenSize, uRadius, uBias, uPower)
//   moved into the group-2 StageUniforms below; the ?: in the tangent fallback
//   became select(). Everything else is a 1:1 translation with comments kept.

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

struct StageUniforms {            // uScreenSize, uRadius, uBias, uPower
    screen_size: vec2<f32>,       // uScreenSize: pixel size of the SSAO target this pass renders into
    radius: f32,                  // uRadius: hemispherical search radius, view-space units
    bias: f32,                    // uBias: depth bias added to the sample z in the step() test
    power: f32,                   // uPower: exponent applied to the clamped AO result
};
@group(2) @binding(0) var<uniform> stage: StageUniforms;

@group(1) @binding(0) var tex0: texture_2d<f32>;  // View-space position, alpha zero for sky.
@group(1) @binding(8) var tex0_sampler: sampler;
@group(1) @binding(1) var tex1: texture_2d<f32>;  // Encoded view-space normal.
@group(1) @binding(9) var tex1_sampler: sampler;
@group(1) @binding(2) var tex2: texture_2d<f32>;  // Repeating 4x4 RGB tangent-rotation noise.
@group(1) @binding(10) var tex2_sampler: sampler;

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

const SAMPLE_COUNT: i32 = 24;
const GOLDEN_ANGLE: f32 = 2.39996322972865332;

fn hemisphereSample(index: i32) -> vec3<f32>
{
    let f = (f32(index) + 0.5)/f32(SAMPLE_COUNT);
    let phi = f32(index)*GOLDEN_ANGLE;
    let z = f;
    let radial = sqrt(max(0.0, 1.0 - z*z));
    let sampleDirection = vec3<f32>(cos(phi)*radial, sin(phi)*radial, z);

    // Concentrate samples near the shaded point while retaining enough long
    // rays to reveal large creases.
    let scale = mix(0.12, 1.0, f*f);
    return sampleDirection*scale;
}

@fragment
fn fs_main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32>
{
    let uv = position.xy * globals.viewport.zw;   // stand-in for fragTexCoord

    let packedPosition = textureSampleLevel(tex0, tex0_sampler, uv, 0.0);
    if (packedPosition.a < 0.5)
    {
        return vec4<f32>(1.0);
    }

    let positionView = packedPosition.xyz;
    let normalView = normalize(textureSampleLevel(tex1, tex1_sampler, uv, 0.0).xyz*2.0 - 1.0);

    let noiseScale = stage.screen_size/4.0;
    var randomVector = textureSampleLevel(tex2, tex2_sampler, uv*noiseScale, 0.0).xyz*2.0 - 1.0;
    randomVector.z = 0.0;
    if (dot(randomVector, randomVector) < 0.0001) { randomVector = vec3<f32>(1.0, 0.0, 0.0); }
    randomVector = normalize(randomVector);

    var tangent = randomVector - normalView*dot(randomVector, normalView);
    if (dot(tangent, tangent) < 0.0001)
    {
        let fallback = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 0.0, 1.0),
                              abs(normalView.z) < 0.9);
        tangent = fallback - normalView*dot(fallback, normalView);
    }
    tangent = normalize(tangent);
    let bitangent = normalize(cross(normalView, tangent));
    let tangentToView = mat3x3<f32>(tangent, bitangent, normalView);

    var occlusion = 0.0;
    var validSamples = 0.0;
    let radius = max(stage.radius, 0.001);

    for (var i = 0; i < SAMPLE_COUNT; i++)
    {
        let samplePosition = positionView + tangentToView*hemisphereSample(i)*radius;
        let projected = globals.projection*vec4<f32>(samplePosition, 1.0);
        if (projected.w <= 0.0) { continue; }

        // NDC -> target uv. Clip-space y counts UP while @builtin(position).y
        // (and therefore the uv derived from it) counts DOWN from the top of
        // the target, so the y half of the mapping is flipped; x is not. The
        // GLSL this is ported from had no flip to make: its framebuffer origin
        // was bottom-left, so clip +y and texture row +v already agreed. With
        // the sign dropped the sample lands on the mirrored row — the whole
        // screen's depth is wrong by the vertical distance to its own mirror,
        // rangeWeight collapses to zero and the SSAO silently returns 1.0
        // everywhere except the one screen row where the mirror is the
        // identity, which paints a single dark band across the centre.
        let sampleUV = vec2<f32>(projected.x/projected.w*0.5 + 0.5,
                                 0.5 - projected.y/projected.w*0.5);
        if (any(sampleUV < vec2<f32>(0.0)) || any(sampleUV > vec2<f32>(1.0)))
        {
            continue;
        }

        let neighbour = textureSampleLevel(tex0, tex0_sampler, sampleUV, 0.0);
        if (neighbour.a < 0.5) { continue; }

        let separation = abs(positionView.z - neighbour.z);
        let rangeWeight = smoothstep(0.0, 1.0, radius/max(separation, 0.0001));

        // OpenGL view space looks down -Z. A larger Z is closer to the camera
        // and therefore lies in front of the tested hemisphere sample.
        let blocked = step(samplePosition.z + stage.bias, neighbour.z);
        occlusion += blocked*rangeWeight;
        validSamples += 1.0;
    }

    var ao = 1.0;
    if (validSamples > 0.0) { ao = 1.0 - occlusion/validSamples; }
    ao = pow(clamp(ao, 0.0, 1.0), max(stage.power, 0.001));
    return vec4<f32>(vec3<f32>(ao), 1.0);
}

// STAGE UNIFORMS:
//   screen_size : vec2<f32> — uScreenSize: pixel dimensions of the SSAO render
//                 target; the C++ fed {ao texture width, height}, which is NOT
//                 necessarily globals.viewport.xy — keep feeding the AO target size
//   radius      : f32       — uRadius: hemispherical search radius, view-space units
//   bias        : f32       — uBias: depth bias added to the sample z before step()
//   power       : f32       — uPower: gamma exponent applied to the clamped AO
// Rust fill (uniform, 32 bytes, WGSL uniform-address alignment):
//   [0] screen_size: [f32; 2], [8] radius: f32, [12] bias: f32, [16] power: f32, [20] pad: [f32; 3]
// Textures (group 1): binding 0 = view-space position RGBA (linear sampler at 8),
// binding 1 = encoded view-space normal (linear sampler at 9), binding 2 = 4x4
// tangent-rotation noise (GL made it POINT-filtered, REPEAT wrap, mip-free; use
// a non-filtering repeating sampler at 10). Group 0 binding 0 = shared
// GlobalUniforms; globals.projection stands in for uProjection (only the x/y/w
// rows are read, so the [0,1] z conversion is irrelevant here).