// PORT NOTES (fxaa.fs -> fxaa.wgsl):
// - HOISTED SAMPLES: the four rgbA/rgbB edge taps (offsets direction*(1/3-0.5),
//   direction*(2/3-0.5), direction*(-0.5), direction*0.5) were taken AFTER the
//   per-pixel threshold early return in the GLSL, and WGSL forbids
//   textureSample in non-uniform control flow, so they are pre-sampled
//   unconditionally right after the direction block and the threshold branch
//   keeps its original shape and place. The direction math between them is
//   pure arithmetic on lumas already fetched (no derivative ops), so it moved
//   up with the taps — identical values, same expression order. rgbA/rgbB
//   assemble below the branch from the hoisted tap values exactly as the GLSL
//   wrote them. Below-threshold pixels now additionally take four samples the
//   GLSL skipped (results discarded); every pixel returns the same colour the
//   GLSL returned.
// - No blit flip was dropped and none added: the GL/raylib composite->screen
//   blit this pass replaces carried an implicit Y flip (GL render targets are
//   bottom-up), but the WGSL port shares ONE top-left-origin row convention
//   for the composite target and this pass, so fragTexCoord derives from
//   position.xy * globals.viewport.zw with no flip (spec section 4). Same
//   convention change flips the y component of `direction` relative to the GL
//   read: every tap offset (1/3, 2/3 and +/-0.5 of the edge direction) lies
//   symmetric about M and mirrors with the texture content itself, so the
//   blended image matches the GL pass pixel-for-pixel under the one shared
//   orientation (spec no-flip rule; handedness note per spec section 5).
// - textureSize(texture0, 0) -> textureDimensions(texture0, 0u): texel comes
//   from the composite target's own dimensions, which equal
//   globals.viewport.zw when this pass renders at native size (the way the
//   Rust pipeline pairs them). No inverse_* stage-uniform field was needed:
//   nothing inverts a matrix in this pass.
// - sampler2D texture0 -> texture_2d<f32> + sampler pair (group 1, bindings
//   0 and 8); the sampler stays linear-filtered — the clamp-to-interior
//   behaviour is tapInterior's uv clamp, exactly as the GLSL wrote it, not a
//   sampler mode. tapInterior gained a `sam` argument because WGSL pairs
//   texture and sampler in the function signature.
// - GL's scalar-extent clamp overload `clamp(vec2, float, float)` does not
//   exist in WGSL: the two limits are splatted to
//   vec2<f32>(-kSpanMax)/vec2<f32>(kSpanMax) — identical math. The final GLSL
//   ternary became select(), which evaluates both operands (both are pure
//   arithmetic over already-taken samples, no derivative work in either).
// - fxaa.fs declares NO uniform beyond `sampler2D texture0`, so this file has
//   no StageUniforms and no @group(2) binding — the Rust bind group for
//   group 2 can be omitted for this pipeline. The four FXAA tuning values
//   (kEdgeThreshold, kSpanMax, kReduceMul, kReduceMin) are compile-time GLSL
//   consts and stay compile-time consts here; Rust does not fill any uniform.
// - Single output: finalColor -> @location(0), alpha forced to 1.0 exactly as
//   the GLSL wrote it in both paths; the gamma-encoded composite bytes are
//   consumed and re-emitted unchanged in encoding (see composite.wgsl), so no
//   channel is dropped or re-ordered.
// - sRGB RENDER TARGET (wgpu adaptation, no GLSL counterpart): the C++ drew
//   into a default framebuffer with no sRGB conversion, so its shader output
//   landed in the byte verbatim. Bevy renders into sRGB-FORMATTED targets —
//   ViewTarget::main_texture_format() is Rgba8UnormSrgb for an LDR camera and
//   the window surface is Bgra8UnormSrgb — and wgpu applies the sRGB encode on
//   every store to one. This pass is the last one that authors pixels (the
//   upscaling blit only resamples them), so its store is where that single
//   extra encode would double-gamma the frame. The fix keeps the whole
//   pipeline in the C++'s display-encoded domain — FXAA filters exactly the
//   bytes the composite wrote, as its luma thresholds require — and converts
//   only the final stored value back to the linear form an sRGB target
//   expects, which is the same last-step convention bevy's own tonemapping
//   uses. Byte out = encode_srgb(decode_srgb(C)) = C, i.e. the frame the C++
//   put on screen. The composite target and this target share one format, so
//   the composite's store/this pass's fetch round-trips the display-encoded
//   value untouched in between.

// FXAA over the tonemapped composite. The deferred G-buffer passes render
// into raylib-managed FBOs, which cannot be multisampled here, so distant
// terrain crests stair-step against the sky. FXAA runs on the final LDR
// image, where the gamma-space luminance edges it needs already exist: the
// composite renders into an offscreen target and this pass blits it to the
// screen, smoothing the 1-2 px staircase jaggies on silhouette boundaries.
// Terrain keeps most of its detail because the subpixel blend only engages
// where local contrast crosses the edge threshold.

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

@group(1) @binding(0) var texture0: texture_2d<f32>;   // The tonemapped, gamma-encoded composite (see composite.wgsl).
@group(1) @binding(8) var texture0_sampler: sampler;

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

// Edge contrast below this passes through untouched, so flat texture and
// the snowfield's micro-contrast keep their exact pixels.
const kEdgeThreshold: f32 = 0.06;
const kSpanMax: f32 = 8.0;
const kReduceMul: f32 = 1.0/8.0;
const kReduceMin: f32 = 1.0/128.0;

fn luminance(colour: vec3<f32>) -> f32
{
    return dot(colour, vec3<f32>(0.299, 0.587, 0.114));
}

// Inverse of the sRGB transfer function the render target applies on store
// (see the sRGB RENDER TARGET note in PORT NOTES). This is the exact analytic
// inverse of the sRGB EOTF, so the target's encode reproduces the
// display-encoded value the filter produced.
fn toLinearForSrgbTarget(colour: vec3<f32>) -> vec3<f32>
{
    let linear = colour/vec3<f32>(12.92);
    let curved = pow((colour + vec3<f32>(0.055))/vec3<f32>(1.055), vec3<f32>(2.4));
    return select(curved, linear, colour <= vec3<f32>(0.04045));
}

// Clamp sample coordinates into the texture interior: the render targets
// default to REPEAT wrap addressing, and this pass runs over the full
// screen quad, so pixels on a frame edge sample the wrapped opposite edge —
// writing a 1-px sky-coloured sliver along the top and bottom rows and
// wrapping distant shoreline into the nearest ground (measured: row 899
// rendered 99.6% identical to the sky row). Clamping makes edge texels
// replicate, the standard clamp-addressing FXAA behaviour.
fn tapInterior(tex: texture_2d<f32>, sam: sampler, uv: vec2<f32>, texel: vec2<f32>) -> vec3<f32>
{
    return textureSample(tex, sam, clamp(uv, texel*0.5, vec2<f32>(1.0) - texel*0.5)).rgb;
}

@fragment
fn fs_main(@builtin(position) frag_position: vec4<f32>) -> @location(0) vec4<f32>
{
    let frag_tex_coord = frag_position.xy*globals.viewport.zw;   // fragTexCoord

    let texel = 1.0/vec2<f32>(textureDimensions(texture0, 0u));  // 1.0/vec2(textureSize(texture0, 0))
    let rgb_nw = tapInterior(texture0, texture0_sampler, frag_tex_coord + vec2<f32>(-1.0, -1.0)*texel, texel);
    let rgb_ne = tapInterior(texture0, texture0_sampler, frag_tex_coord + vec2<f32>( 1.0, -1.0)*texel, texel);
    let rgb_sw = tapInterior(texture0, texture0_sampler, frag_tex_coord + vec2<f32>(-1.0,  1.0)*texel, texel);
    let rgb_se = tapInterior(texture0, texture0_sampler, frag_tex_coord + vec2<f32>( 1.0,  1.0)*texel, texel);
    let rgb_m  = tapInterior(texture0, texture0_sampler, frag_tex_coord, texel);

    let luma_nw = luminance(rgb_nw);
    let luma_ne = luminance(rgb_ne);
    let luma_sw = luminance(rgb_sw);
    let luma_se = luminance(rgb_se);
    let luma_m  = luminance(rgb_m);

    let luma_min = min(luma_m, min(min(luma_nw, luma_ne), min(luma_sw, luma_se)));
    let luma_max = max(luma_m, max(max(luma_nw, luma_ne), max(luma_sw, luma_se)));

    // Edge normal: the luminance gradient, so the taps below walk ACROSS the
    // edge line. (The along-edge orientation — comparing top-half vs
    // bottom-half sums for x and left vs right for y — degenerates to
    // passthrough on axis-aligned edges: every tap lands on M's own side.)
    // The imbalance is inverse-scaled by the neighbourhood mean so bright
    // skies do not produce runaway offsets.
    // HOISTED with the taps below the threshold return: pure arithmetic on
    // the lumas already fetched, so its value is unchanged (see PORT NOTES).
    var direction = vec2<f32>(-((luma_nw + luma_sw) - (luma_ne + luma_se)),
                               ((luma_nw + luma_ne) - (luma_sw + luma_se)));
    let direction_reduce = max((luma_nw + luma_ne + luma_sw + luma_se)*0.25*kReduceMul,
                               kReduceMin);
    let inverse_scale = 1.0/(min(abs(direction.x), abs(direction.y)) + direction_reduce);
    direction = clamp(direction*inverse_scale, vec2<f32>(-kSpanMax), vec2<f32>(kSpanMax))*texel;

    // HOISTED SAMPLES: the GLSL fetched these four taps after the threshold
    // early return below, where WGSL forbids textureSample (non-uniform
    // control flow). They are pre-sampled unconditionally here; the
    // passthrough pixels ignore them, and rgbA/rgbB keep the GLSL
    // expressions, offsets and order below.
    let tap_a1 = tapInterior(texture0, texture0_sampler, frag_tex_coord + direction*(1.0/3.0 - 0.5), texel);
    let tap_a2 = tapInterior(texture0, texture0_sampler, frag_tex_coord + direction*(2.0/3.0 - 0.5), texel);
    let tap_b1 = tapInterior(texture0, texture0_sampler, frag_tex_coord + direction*(-0.5), texel);
    let tap_b2 = tapInterior(texture0, texture0_sampler, frag_tex_coord + direction*0.5, texel);

    if (luma_max - luma_min < kEdgeThreshold)
    {
        return vec4<f32>(toLinearForSrgbTarget(rgb_m), 1.0);
    }

    // Walk the edge in both directions: the 1/3 and 2/3 taps blend along the
    // edge line, the second pair straddles it; if the straddled blend leaves
    // the local luminance range the edge is longer than the span and the
    // inner pair alone is the safer estimate.
    let rgb_a = 0.5*(tap_a1 + tap_a2);
    let rgb_b = rgb_a*0.5 + 0.25*(tap_b1 + tap_b2);
    let luma_b = luminance(rgb_b);
    // finalColor = (lumaB < lumaMin || lumaB > lumaMax)
    //            ? vec4(rgbA, 1.0) : vec4(rgbB, 1.0);
    return select(vec4<f32>(toLinearForSrgbTarget(rgb_b), 1.0),
                  vec4<f32>(toLinearForSrgbTarget(rgb_a), 1.0),
                  luma_b < luma_min || luma_b > luma_max);
}

// STAGE UNIFORMS:
//   None: fxaa.fs declares no uniforms beyond `sampler2D texture0`, so this
//   file has no StageUniforms struct and no @group(2) uniform buffer; the
//   Rust bind group for group 2 can be omitted entirely for this pipeline.
//   The four FXAA tuning values are compile-time GLSL consts and stay
//   compile-time WGSL consts:
//     kEdgeThreshold = 0.06, kSpanMax = 8.0,
//     kReduceMul = 1.0/8.0, kReduceMin = 1.0/128.0
// Shared uniforms carried by GlobalUniforms (group 0/binding 0):
//   viewport.zw supplies fragTexCoord; nothing else is read. texel derives
//   from textureDimensions(texture0, 0u), exactly as the GLSL derived it from
//   textureSize(texture0, 0).
// Textures (group 1, linear-filtered, sampler at binding 0 + 8):
//   0 = the tonemapped, gamma-encoded composite written by composite.wgsl;
//       consumed raw (REPEAT wrap addressing is emulated by tapInterior's
//       clamp-to-interior uv clamp, matching the GL behaviour it fixed).
// Output: location 0 = the anti-aliased gamma-encoded RGBA8 — FXAA consumes
//   the composite bytes as-is and re-emits the same encoding; nothing
//   re-linearizes in between (see composite.wgsl PORT NOTES).