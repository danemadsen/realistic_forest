// Copies the composited frame into the water pass's output target.
//
// Not from bevy-aqua: this exists because `ViewTarget::post_process_write()`
// ping-pongs between two textures, and the water surface only covers the
// pixels it draws. The pass therefore has to write the *whole* target — the
// composite's output for every pixel the water does not touch, and the water
// shading for the ones it does. Drawing this first, then the patches over it,
// is that fill.
//
// It is deliberately paired with the water surface's group-1 layout so the
// same bind group serves both: binding 0 is the composited frame and binding 8
// its sampler, exactly as the surface reads its refraction source. The unused
// G-buffer position slot (1/9) is simply not referenced.

struct GlobalUniforms
{
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
    camera_position: vec4<f32>,
    sun_direction: vec4<f32>,
    viewport: vec4<f32>,          // xy size in pixels, zw inverse size
    params: vec4<f32>,
    settings_a: vec4<f32>,
    settings_b: vec4<f32>,
    sun_colour: vec4<f32>,
    moon_direction: vec4<f32>,
    atmosphere: vec4<f32>,
    raymarch: vec4<f32>,
    heightfield: vec4<f32>,
};
@group(0) @binding(0) var<uniform> globals: GlobalUniforms;

@group(1) @binding(0) var source_texture: texture_2d<f32>;
@group(1) @binding(8) var source_sampler: sampler;

struct VsOutput
{
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex: u32) -> VsOutput
{
    var corner = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    var output: VsOutput;
    output.position = vec4<f32>(corner[vertex], 0.0, 1.0);
    return output;
}

@fragment
fn fs_main(in: VsOutput) -> @location(0) vec4<f32>
{
    // Same top-down uv convention as the composite and FXAA: `position` is in
    // the target's pixel space, so `position.xy * viewport.zw` is a plain
    // same-frame blit with no flip.
    return textureSample(source_texture, source_sampler, in.position.xy*globals.viewport.zw);
}
