// Instanced fir trees: the vertex stage.
//
// Runs between the terrain pass and SSAO, drawing every tree of one
// (size, variant, LOD) group in a single instanced draw. It writes the same
// three G-buffer targets the terrain pass writes, so from the composite pass
// onward a tree is indistinguishable from terrain: it is lit, shadowed by the
// raymarched terrain and cloud marches, occluded by the water surface, and
// visible in the raymarched water reflection, all without the composite or the
// water shader knowing trees exist.
//
// WHERE THE GROUND HEIGHT COMES FROM — and why it is *not* `terrainHeight`.
//
// The scatter picked each tree's XZ by evaluating the full height model on the
// CPU (src/trees/placement.rs), and it kept the height it computed. That value
// arrives here as `instance.ground`. Calling `terrainHeight` per vertex
// instead would be the obvious symmetry with the clipmap, and it is wrong:
// the height model costs about twenty noise texture samples and four erosion
// atlas fetches, and this pass shades on the order of a million vertices a
// frame once the near LODs and their instance counts are counted — the trees
// would cost several times the entire terrain. Per-instance, the same value is
// free.
//
// The two agree because they are the same model: the CPU port reads the same
// erosion cache the atlas is filled from, and the player's own walking
// collision already snaps to it (src/player.rs), which is a far more sensitive
// test of "does the CPU height match the rendered surface" than a tree's foot
// two hundred metres away. Two divergences remain, both bounded and both
// documented in src/trees/placement.rs: the scatter judges a chunk against the
// *chunk's* visibility centre rather than the player's (a centimetre of error
// at the scatter radius, where `erosionVisibility` is still 0.90), and a tree
// does not re-seat itself while a newly-finalised erosion tile fades in around
// it.
//
// The instance's XZ is the only horizontal input. Everything else — scale,
// yaw, the sink that seats the trunk on a slope, the ground height itself —
// arrives per instance.

struct GlobalUniforms {
    view: mat4x4<f32>,            // matView: column-major world->view
    projection: mat4x4<f32>,      // matProjection (standard GL shape, z converted to [0,1] NDC)
    camera_position: vec4<f32>,   // xyz world-space eye; w unused
    sun_direction: vec4<f32>,     // xyz direction sunlight travels, WORLD space
    viewport: vec4<f32>,          // xy = width, height in px; zw = 1/width, 1/height
    params: vec4<f32>,            // x fog_density, y z_far, z exposure, w ssao_enabled (1=on, 0=off)
    settings_a: vec4<f32>,        // x sun_intensity, y texture_scale, z ao_tex_strength, w variant_scale
    settings_b: vec4<f32>,        // x normal_strength, y sparkle_strength, z flow_debug(1/0), w erosion_debug(1/0)
    sun_colour: vec4<f32>,        // RGB solar tint, w unattenuated sun intensity
    moon_direction: vec4<f32>,    // xyz direction moonlight travels
    atmosphere: vec4<f32>,        // daylight, moon intensity, hours, volumetric strength
    raymarch: vec4<f32>,          // shadows, volumetrics, reflections, quality (0/1/2)
    heightfield: vec4<f32>,       // world centre XZ, world span, texel size in metres
    clouds: vec4<f32>,            // enabled, coverage, density, base altitude
    cloud_layer: vec4<f32>,       // thickness, shape scale, shadow strength, quality
    cloud_motion: vec4<f32>,      // wind offset XZ, detail strength, maximum distance
};
@group(0) @binding(0) var<uniform> globals: GlobalUniforms;

// ---------------------------------------------------------------------------
// Per-instance placement
// ---------------------------------------------------------------------------

// One scattered tree. 28 bytes, mirrored by `TreeInstance` in
// src/trees/mod.rs, which pins the size with a const assert.
struct TreeInstance {
    @location(4) centre: vec2<f32>,     // world XZ
    @location(5) ground: f32,           // world Y of the ground under `centre`
    @location(6) scale: f32,            // uniform object-space scale
    @location(7) rotation: f32,         // yaw, radians
    @location(8) variation: f32,        // 0..1 stable per-tree random
    @location(9) sink: f32,             // metres the base is pushed below ground
};

// Yaw only. A tree grows upright whatever the ground under it does, so the
// trunk is never tilted onto the surface normal — it is sunk into it instead.
fn rotateY(p: vec3<f32>, angle: f32) -> vec3<f32>
{
    let c = cos(angle);
    let s = sin(angle);
    return vec3<f32>(c*p.x + s*p.z, p.y, -s*p.x + c*p.z);
}

struct VsOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) fragPositionView: vec3<f32>,
    @location(1) fragNormalView: vec3<f32>,
    @location(2) fragUv: vec2<f32>,
    @location(3) fragTangentView: vec4<f32>,
    @location(4) fragVariation: f32,
};

@vertex
fn vs_main(@location(0) vertexPosition: vec3<f32>,
           @location(1) vertexNormal: vec3<f32>,
           @location(2) vertexUv: vec2<f32>,
           @location(3) vertexTangent: vec4<f32>,
           instance: TreeInstance) -> VsOutput
{
    // `sink` is the only vertical authoring input beyond the ground itself: it
    // seats the trunk's foot below the surface so a tree standing on a slope
    // does not balance on the downhill edge of its own base.
    let foot = vec3<f32>(instance.centre.x, instance.ground - instance.sink, instance.centre.y);

    let localPosition = rotateY(vertexPosition*instance.scale, instance.rotation);
    let worldPosition = foot + localPosition;
    let worldNormal = rotateY(vertexNormal, instance.rotation);
    let worldTangent = rotateY(vertexTangent.xyz, instance.rotation);

    var output: VsOutput;
    output.position = globals.projection*globals.view*vec4<f32>(worldPosition, 1.0);
    output.fragPositionView = (globals.view*vec4<f32>(worldPosition, 1.0)).xyz;
    output.fragNormalView = (globals.view*vec4<f32>(worldNormal, 0.0)).xyz;
    // The tangent stays in view space alongside the normal so the fragment
    // stage builds the normal-map frame with one matrix, not two.
    output.fragTangentView = vec4<f32>(
        (globals.view*vec4<f32>(worldTangent, 0.0)).xyz, vertexTangent.w);
    output.fragUv = vertexUv;
    output.fragVariation = instance.variation;
    return output;
}
