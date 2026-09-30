// Instanced fir trees: the fragment stage.
//
// It writes the deferred G-buffer and nothing else. Everything after this —
// sun and moon lighting, the raymarched terrain and cloud shadows, the
// volumetric fog, SSAO, the water's occlusion test and the water's raymarched
// reflection — is the existing post chain operating on a G-buffer texel that
// happens to belong to a tree. That is the whole reason trees are drawn here
// rather than forward after the composite: a forward pass would be covered by
// the composite's own background, would not occlude the water (the water has
// no depth attachment; it reconstructs depth from the position target), and
// could never appear in a reflection.
//
// WHAT TREES DO NOT DO: cast raymarched shadows. `terrainShadow` in
// composite.wgsl and water-surface.wgsl march the R32Float terrain heightfield
// only, and adding a tree's canopy to that field would make every fragment
// below its own crown self-shadow, because the march's `lift` term starts at
// the fragment's own height. Trees are shadow *receivers* here.

// group(1): this draw's material. Only the fragment stage reads it, so this is
// its single declaration — the Rust side that fills it is `TreeMaterialUniforms`
// in src/render/tree_node.rs, which carries a const assert on the size. The
// other passes in this renderer repeat GlobalUniforms in each of their modules
// because WGSL has no #include; this struct avoids that by being used once.
struct TreeMaterialUniforms {
    is_branch: u32,
    roughness: f32,
    alpha_cutoff: f32,
    tint_strength: f32,
    specular_factor: f32,
    // WGSL rounds a `uniform` struct up to a multiple of 16 bytes; the five
    // fields above come to 20. Three plain floats rather than a `vec3` on
    // purpose — a vec3 aligns to 16, which would pad this out to 48 bytes and
    // stop it matching the Rust `TreeMaterialUniforms`. Never read.
    _padding0: f32,
    _padding1: f32,
    _padding2: f32,
};
@group(1) @binding(0) var bark_albedo: texture_2d<f32>;
@group(1) @binding(1) var bark_normal_map: texture_2d<f32>;
@group(1) @binding(2) var branch_albedo: texture_2d<f32>;
@group(1) @binding(3) var branch_normal_map: texture_2d<f32>;
@group(1) @binding(4) var tree_sampler: sampler;
@group(1) @binding(5) var<uniform> material: TreeMaterialUniforms;

struct FsInput {
    @location(0) fragPositionView: vec3<f32>,
    @location(1) fragNormalView: vec3<f32>,
    @location(2) fragUv: vec2<f32>,
    @location(3) fragTangentView: vec4<f32>,
    @location(4) fragVariation: f32,
    @builtin(front_facing) frontFacing: bool,
};

struct FsOutput {
    // Rgba32Float: view-space position, a >= 0.5 = covered. Above that the
    // alpha is a material tag, not a number the geometry pass has any use for:
    // terrain packs its biome masks into [1.0, 1.01] and trees have no biome
    // masks, so a tree marks itself at TREE_POSITION_ALPHA and carries its
    // glTF specular factor in the fraction. composite.wgsl decodes it.
    @location(0) position: vec4<f32>,
    @location(1) normal: vec4<f32>,     // Rgba16Float: view normal*0.5+0.5, a = roughness
    @location(2) albedo: vec4<f32>,     // Rgba8Unorm:  sqrt(albedo*0.4), a = AO
};

/// Marks a fragment as foliage on the position target's alpha and carries the
/// material's `KHR_materials_specular` factor above it. The same constant is
/// spelled out in composite.wgsl — WGSL has no #include, and the two files are
/// separate shader assets, so a unit test in src/render/mod.rs asserts the two
/// spellings still agree rather than letting them drift apart silently.
const TREE_POSITION_ALPHA: f32 = 1.5;

// A per-tree tint, keyed on the instance's stable random. Stands of one mesh
// repeated hundreds of times otherwise read as wallpaper; a few percent of
// hue and value spread is enough for the eye to stop counting.
//
// The two ends are a cooler, darker tree and a warmer, lighter one, so the
// spread reads as age and light rather than as noise.
fn treeTint(variation: f32, strength: f32) -> vec3<f32> {
    let cool = vec3<f32>(0.90, 0.98, 0.92);
    let warm = vec3<f32>(1.08, 1.00, 0.86);
    return mix(vec3<f32>(1.0), mix(cool, warm, variation), strength);
}

@fragment
fn fs_main(input: FsInput) -> FsOutput {
    // Uniform branch on the material uniform: legal for textureSample, and it
    // keeps the bark path from paying for the branch textures (and the reverse
    // on the LOD-3 billboard, which has neither a branch material nor bark).
    var base: vec4<f32>;
    var normal_sample: vec3<f32>;
    if (material.is_branch == 1u) {
        base = textureSample(branch_albedo, tree_sampler, input.fragUv);
        normal_sample = textureSample(branch_normal_map, tree_sampler, input.fragUv).xyz;
    } else {
        base = textureSample(bark_albedo, tree_sampler, input.fragUv);
        normal_sample = textureSample(bark_normal_map, tree_sampler, input.fragUv).xyz;
    }

    // Alpha cutout, not blending. The branch albedo's alpha is effectively
    // binary (72.9% of texels are zero, 26.1% are one), and the water pass
    // reconstructs depth from texel-exact point samples of the position
    // target — a blended fragment writes depth and coverage that disagree and
    // streaks the raymarched reflection.
    if (base.a < material.alpha_cutoff) {
        discard;
    }

    // The base colour textures are sampled through an sRGB view, so `base.rgb`
    // is already linear here; the terrain's albedo atlas instead decodes with
    // an explicit pow(2.2) because its own upload path is Rgba8Unorm. Both
    // arrive in the same space at the G-buffer.
    let albedo = base.rgb*treeTint(input.fragVariation, material.tint_strength);

    // The pack ships per-vertex tangents and its normal maps are OpenGL
    // green-up, so the frame is built directly rather than from screen-space
    // derivatives: a foliage card's UV gradient is undefined across the alpha
    // cut, which is exactly where derivative frames break down.
    //
    // Every tree material is doubleSided (see the cull_mode note in
    // src/render/glb.rs), so a camera inside a canopy sees the back of the
    // cards that face away from it. Flipping the shading normal there is what
    // keeps those leaves lit: a back face shaded with its own outward normal
    // faces away from the sun and renders black, which reads as a hole in the
    // tree rather than as the inside of one. Only the normal flips — the
    // tangent frame's handedness is a property of the UV layout, not of which
    // side is being drawn.
    let normal = normalize(input.fragNormalView)*select(-1.0, 1.0, input.frontFacing);
    let tangent = normalize(input.fragTangentView.xyz
                            - normal*dot(normal, input.fragTangentView.xyz));
    let bitangent = cross(normal, tangent)*input.fragTangentView.w;
    let mapped = normal_sample*2.0 - vec3<f32>(1.0);
    let shaded_normal = normalize(tangent*mapped.x + bitangent*mapped.y + normal*mapped.z);

    var output: FsOutput;
    output.position = vec4<f32>(input.fragPositionView,
                                TREE_POSITION_ALPHA + material.specular_factor);
    output.normal = vec4<f32>(shaded_normal*0.5 + vec3<f32>(0.5), material.roughness);
    // sqrt(albedo*0.4) is the terrain G-buffer's own encode; composite decodes
    // with rgb*rgb*2.5. Writing linear albedo here would darken every tree.
    // The pack ships no ambient-occlusion map, so the AO channel is neutral.
    output.albedo = vec4<f32>(sqrt(max(albedo, vec3<f32>(0.0))*0.4), 1.0);
    return output;
}
