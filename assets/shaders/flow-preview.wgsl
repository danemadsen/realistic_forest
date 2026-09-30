// PORT NOTES (no GLSL source file; new in the Rust port):
// - The C++ diagnostics window displayed `erosion.flowAtlas` with
//   `ImGui::Image(tex, {320,320}, {0,1}, {1,0})`, i.e. it bound the raw GL
//   texture. bevy_egui's image path only accepts bevy Image assets (it copies
//   CPU-side pixel data into its own atlas), and the flow atlas exists solely
//   on the GPU, so the port draws it with a raw wgpu pass issued from an egui
//   paint callback (see `FlowAtlasPaintCallback` in src/ui.rs). The callback
//   runs inside the egui render pass with the viewport already clipped to the
//   widget rect, so a fullscreen triangle fills exactly the 320x320 image.
// - The C++ passed uv0 = (0, 1), uv1 = (1, 0) because a GL FBO's rows run
//   bottom-up in texture memory. The wgpu atlas is filled top-down by
//   `gpu_textures::write_padded_at` (its row 0 lands at texel y = 0, the top),
//   so leaving uv.y = 0 at the top of the viewport reproduces the exact
//   orientation the C++ displayed. No flip is applied.
// - Blending stays on: ImGui's pipeline always blends, and the atlas's alpha
//   channel is discharge, so the C++ image was already composited that way.
//   The blend is the same separate-alpha pair ImGui's GL backend used
//   (GL_SRC_ALPHA/GL_ONE_MINUS_SRC_ALPHA for colour,
//   GL_ONE/GL_ONE_MINUS_SRC_ALPHA for alpha); the Rust pipeline sets it.
// - sRGB TARGET (wgpu adaptation, no GLSL counterpart): ImGui drew into a
//   non-sRGB framebuffer, so each float texel reached the byte verbatim —
//   clamped to [0, 1] by the fixed-function RGBA8 conversion, then blended in
//   that encoded space. This pass renders into the sRGB-format main texture
//   (see the sRGB note in fxaa.wgsl), which encodes on store, so the fragment
//   clamps first and then applies the inverse encode: byte out = clamp(texel),
//   the byte the C++ showed. Blending still lands in a different space than
//   GL's did, the same unavoidable shift the egui widgets carry.
// - No uniforms: the rect comes from the render pass viewport and the atlas
//   sample is the whole texture, so the vertex stage derives uv from its own
//   clip position.

@group(0) @binding(0) var flow_atlas: texture_2d<f32>;
@group(0) @binding(1) var flow_atlas_sampler: sampler;

struct VsOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex: u32) -> VsOutput {
    var corner = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    let position = corner[vertex];
    var output: VsOutput;
    output.position = vec4<f32>(position, 0.0, 1.0);
    // NDC y = +1 is the top of the viewport; the atlas's row 0 (its top) is at
    // uv.y = 0, so the port negates y here to keep the orientation.
    output.uv = vec2<f32>(position.x * 0.5 + 0.5, 0.5 - position.y * 0.5);
    return output;
}

// Inverse of the sRGB transfer function the render target applies on store
// (see the sRGB TARGET note above).
fn toLinearForSrgbTarget(colour: vec3<f32>) -> vec3<f32>
{
    let linear = colour/vec3<f32>(12.92);
    let curved = pow((colour + vec3<f32>(0.055))/vec3<f32>(1.055), vec3<f32>(2.4));
    return select(curved, linear, colour <= vec3<f32>(0.04045));
}

@fragment
fn fs_main(input: VsOutput) -> @location(0) vec4<f32> {
    let texel = textureSample(flow_atlas, flow_atlas_sampler, input.uv);
    return vec4<f32>(toLinearForSrgbTarget(clamp(texel.rgb, vec3<f32>(0.0), vec3<f32>(1.0))),
                     clamp(texel.a, 0.0, 1.0));
}
