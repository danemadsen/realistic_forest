// G-buffer terrain shading — WGSL port of forest/assets/shaders/terrain.fs
// (GLSL 330, raylib/GL 4.3 deferred renderer). 1:1 behavioral port: identical
// math, identical constants, preserved comments. Function decomposition kept;
// GLSL out/inout parameters became ptr<function> parameters; GLSL ternaries
// became select() (false-value first — both arms evaluate, discarded arms are
// side-effect free, see PORT NOTES).

// PORT NOTES (deviations from a blind mechanical translation):
// 1. textureGrad -> textureSampleGrad, NOT the spec table's textureSample.
//    Every original textureGrad call supplies analytic gradients that are
//    behaviorally load-bearing: in sampleMaterial the per-cell quarter turns,
//    hash-jittered offsets and cell jumps would spike implicit derivatives at
//    cell boundaries ("Analytic gradients keep cell boundaries out of mip
//    selection"), and all texture0 noise fetches cross mirrorTile folds whose
//    abs() cusp doubles the derivative ("auto-mip ... spike at every
//    mirrorTile fold"). textureSampleGrad lowers to naga SampleLevel::Gradient
//    which, exactly like GLSL textureGrad, carries NO uniform-control-flow
//    requirement, so every sample keeps its original placement — no hoisting,
//    no restructuring anywhere. Comments below keep the GL spelling
//    "textureGrad" where they describe the history; the ported calls read
//    textureSampleGrad.
// 2. Hoisted samples: the three implicit texture() calls inside the erosion
//    helpers (erosionTileMask -> texture4; accumulateFlowTile -> texture2 and
//    texture1) sit behind early-return guards in the GLSL. WGSL forbids
//    textureSample in non-uniform control flow and naga's analysis poisons
//    statements after a conditional return, so each sample is pre-sampled
//    unconditionally (uniform flow) ahead of the guards and the guards select
//    the identical value afterwards. Values used are unchanged; the skipped
//    fetches merely execute and are discarded.
// 3. uFlowDebug / uErosionDebug are GL ints; they travel in
//    globals.settings_b.z / globals.settings_b.w as f32 (1/0) and are tested
//    `!= 0.0` per the spec's bool-condition rule.
// 4. No texture flips were dropped: this fragment stage never blits a
//    render-target and never reads gl_FragCoord; there is nothing to flip.
// 5. No inverse_* uniform fields were needed: this stage has no matrix
//    inversion (the vertex stage's inverse() is that file's business).
// 6. Varying contract: @location(0..4) = fragPositionView, fragNormalView,
//    fragWorldPosition, fragWorldNormal, fragErosionDelta — terrain.vs's
//    `out` declaration order. The wgsl-check validator requires a vs_main
//    entry in every file it validates, so a never-bound dummy vs_main sits at
//    the bottom of this fragment-only file.
// 7. WGSL has no int uniforms here: everything else routes into the
//    spec-fixed GlobalUniforms (settings_a.y = uTexScale, settings_a.w =
//    uVariantScale, settings_b.x = uNormalStrength, settings_b.y =
//    uSparkleStrength, sun_direction = uSunDirectionWorld, camera_position =
//    uCameraPosition, view = matView) or into StageUniforms (see its
//    provenance comment and the STAGE UNIFORMS block at the file end).
// 8. Material placement now uses world-anchored terrain exposure for an
//    ordered grass -> soil -> rock transition. Snow combines regional climate
//    variation with terrain retention instead of a fixed contour on rock.

// Shared global uniforms — the spec preamble, verbatim.
struct GlobalUniforms {
    view: mat4x4<f32>,            // matView: column-major world->view
    projection: mat4x4<f32>,      // matProjection (standard GL shape, z converted to [0,1] NDC)
    camera_position: vec4<f32>,   // xyz world-space eye; w unused
    sun_direction: vec4<f32>,     // xyz direction sunlight travels, WORLD space
    viewport: vec4<f32>,          // xy = width, height in px; zw = 1/width, 1/height
    params: vec4<f32>,            // x fog_density, y z_far, z exposure, w ssao_enabled (1=on, 0=off)
    settings_a: vec4<f32>,        // x sun_intensity, y texture_scale, z ao_tex_strength, w variant_scale
    settings_b: vec4<f32>,        // x normal_strength, y sparkle_strength, z flow_debug(1/0), w erosion_debug(1/0)
};
@group(0) @binding(0) var<uniform> globals: GlobalUniforms;

// Per-stage uniforms — every remaining terrain.fs uniform, uName -> name.
// terrain.vs and terrain.fs are one program in the GLSL original, so both
// stages share a single StageUniforms uniform block. WGSL splits the stages
// into separate modules but the bind group is still one object at group 2
// binding 0, so this declaration must reproduce terrain.vs's 272-byte layout
// byte for byte — the fields the fragment stage reads (sea_level and the
// erosion lookup block) must sit at their vertex-stage offsets or the
// fragment stage would decode the vertex stage's clipmap fields as its own.
struct StageUniforms {
    model: mat4x4<f32>,                    // matModel (vertex stage)
    inverse_model: mat4x4<f32>,            // inverse(matModel), Rust-supplied (vertex stage)
    clip_origin: vec2<f32>,                // uClipOrigin (vertex stage)
    spacing: f32,                          // uSpacing (vertex stage)
    next_spacing: f32,                     // uNextSpacing (vertex stage)
    morph_start: f32,                      // uMorphStart (vertex stage)
    morph_end: f32,                        // uMorphEnd (vertex stage)
    sea_level: f32,                        // uSeaLevel
    noise_period: f32,                     // uNoisePeriod (vertex stage)
    landform_horizontal_scale: f32,        // uLandformHorizontalScale (vertex stage)
    landform_vertical_scale: f32,          // uLandformVerticalScale (vertex stage)
    land_profile_curve: f32,               // uLandProfileCurve (vertex stage)
    land_profile_reference: f32,           // uLandProfileReference (vertex stage)
    land_profile_peak: f32,                // uLandProfilePeak (vertex stage)
    waterline_clearance: f32,              // uWaterlineClearance (vertex stage)
    waterline_clearance_scale: f32,        // uWaterlineClearanceScale (vertex stage)
    waterline_clearance_decay: f32,        // uWaterlineClearanceDecay (vertex stage)
    ocean_profile_curve: f32,              // uOceanProfileCurve (vertex stage)
    ocean_profile_reference: f32,          // uOceanProfileReference (vertex stage)
    ocean_profile_depth: f32,              // uOceanProfileDepth (vertex stage)
    erosion_tile_stride: f32,              // uErosionTileStride
    erosion_footprint_size: f32,           // uErosionFootprintSize
    erosion_output_resolution: f32,        // uErosionOutputResolution
    erosion_atlas_pitch: f32,              // uErosionAtlasPitch
    erosion_atlas_size: f32,               // uErosionAtlasSize
    erosion_atlas_gutter: f32,             // uErosionAtlasGutter
    erosion_lookup_min_tile: vec2<f32>,    // uErosionLookupMinTile
    erosion_lookup_size: f32,              // uErosionLookupSize
    erosion_visibility_center: vec2<f32>,  // uErosionVisibilityCenter
    erosion_visibility_full_radius: f32,   // uErosionVisibilityFullRadius
    erosion_visibility_zero_radius: f32,   // uErosionVisibilityZeroRadius
    waterline_push_land: f32,              // uWaterlinePushLand (vertex stage)
    waterline_push_sea: f32,               // uWaterlinePushSea (vertex stage)
    waterline_push_scale: f32,             // uWaterlinePushScale (vertex stage)
};
@group(2) @binding(0) var<uniform> stage: StageUniforms;

// GL texture unit numbers preserved (see PORTING-SPEC section 3): texture at
// binding N, sampler at binding N + 8.
// Native hydraulic result packing:
//   R = water depth, G = signed velocity X, B = signed velocity Z,
//   A = accumulated discharge.
// Surface atlas: signed bed delta (m), local concavity (m), discharge
// concentration (positive log2 ratio), and simulated substrate hardness.
@group(1) @binding(1) var texture1: texture_2d<f32>;
@group(1) @binding(2) var texture2: texture_2d<f32>;
@group(1) @binding(3) var texture3: texture_2d<f32>;        // Point-filtered RGBA32F tile lookup.
@group(1) @binding(4) var texture4: texture_2d<f32>;        // Raylib square-gradient tile blend mask.
// PBR terrain texture atlases (see LoadTerrainTextures in main.cpp).
// texture0 is the shared base-noise field for biome boundaries and surface detail.
@group(1) @binding(0) var texture0: texture_2d<f32>;        // R32 base noise (also used by the vertex stage).
@group(1) @binding(5) var albedo_ao: texture_2d_array<f32>;   // uAlbedoAO: 7 layers: rgb albedo (sRGB), a ambient occlusion.
@group(1) @binding(6) var normal_rough: texture_2d_array<f32>; // uNormalRough: 7 layers: rgb tangent normal (OpenGL green-up), a roughness.

@group(1) @binding(8) var texture0_sampler: sampler;
@group(1) @binding(9) var texture1_sampler: sampler;
@group(1) @binding(10) var texture2_sampler: sampler;
@group(1) @binding(11) var texture3_sampler: sampler;
@group(1) @binding(12) var texture4_sampler: sampler;
@group(1) @binding(13) var albedo_ao_sampler: sampler;
@group(1) @binding(14) var normal_rough_sampler: sampler;

// Fragment-stage inputs: terrain.vs's five `out` varyings, declaration order
// = @location order (see PORT NOTES 6). Naming: fragPositionView etc.
struct FsInput {
    @location(0) frag_position_view: vec3<f32>,   // fragPositionView
    @location(1) frag_normal_view: vec3<f32>,     // fragNormalView
    @location(2) frag_world_position: vec3<f32>,  // fragWorldPosition
    @location(3) frag_world_normal: vec3<f32>,    // fragWorldNormal
    @location(4) frag_erosion_delta: f32,         // fragErosionDelta
};

// G-buffer outputs, locations preserved from the GLSL layout qualifiers.
struct FsOutput {
    @location(0) g_position: vec4<f32>,   // gPosition
    @location(1) g_normal: vec4<f32>,     // gNormal
    @location(2) g_albedo: vec4<f32>,     // gAlbedo
};

fn smoothHermite(edge0: f32, edge1: f32, value: f32) -> f32
{
    let t = clamp((value - edge0) / max(edge1 - edge0, 0.0001), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

// Mirror-tile a coordinate pair so a non-periodic field can be re-tiled by
// reflection: C0-continuous across every wrap (period 2 in p units), unlike a
// plain fract() which steps at each tile border.
fn mirrorTile(p: vec2<f32>) -> vec2<f32>
{
    // GLSL mod(p, 2.0) == p - 2.0 * floor(p / 2.0)
    let m = p - 2.0 * floor(p / 2.0);
    return abs(m - vec2<f32>(1.0));
}

fn lookupErosionTile(tile_coordinate: vec2<f32>, record: ptr<function, vec4<f32>>) -> bool
{
    let relative_tile = tile_coordinate - stage.erosion_lookup_min_tile;
    let lookup_size = max(stage.erosion_lookup_size, 1.0);
    if (any(relative_tile < vec2<f32>(0.0)) ||
        any(relative_tile >= vec2<f32>(lookup_size)))
    {
        (*record) = vec4<f32>(0.0);
        return false;
    }

    // GLSL texelFetch(texture3, ivec2(relativeTile), 0)
    (*record) = textureLoad(texture3, vec2<i32>(relative_tile), 0);
    return (*record).a > 0.0;
}

fn erosionSupportUV(worldXZ: vec2<f32>, tile_coordinate: vec2<f32>) -> vec2<f32>
{
    let stride = max(stage.erosion_tile_stride, 0.0001);
    let footprint = max(stage.erosion_footprint_size, 0.0001);
    let support_minimum = tile_coordinate * stride - vec2<f32>(0.5 * footprint);
    return (worldXZ - support_minimum) / footprint;
}

fn erosionTileMask(worldXZ: vec2<f32>, tile_coordinate: vec2<f32>) -> f32
{
    let tileUV = erosionSupportUV(worldXZ, tile_coordinate);
    // PORT: the GLSL samples texture4 only after the range guard below; the
    // sample is hoisted here (uniform control flow, see PORT NOTES 2) and the
    // guard selects the identical value.
    let mask_sample = textureSample(texture4, texture4_sampler, tileUV).r;
    if (any(tileUV < vec2<f32>(0.0)) ||
        any(tileUV > vec2<f32>(1.0))) { return 0.0; }
    return max(mask_sample, 0.0);
}

fn erosionAtlasUV(worldXZ: vec2<f32>, tile_coordinate: vec2<f32>, atlas_slot: vec2<f32>) -> vec2<f32>
{
    let output_resolution = max(stage.erosion_output_resolution, 1.0);
    let tileUV = clamp(erosionSupportUV(worldXZ, tile_coordinate),
                       vec2<f32>(0.0), vec2<f32>(1.0));
    let atlas_pixel = atlas_slot * max(stage.erosion_atlas_pitch, output_resolution)
                    + vec2<f32>(max(stage.erosion_atlas_gutter, 0.0))
                    + tileUV * output_resolution;
    return atlas_pixel / max(stage.erosion_atlas_size, 1.0);
}

fn accumulateFlowTile(worldXZ: vec2<f32>, tile_coordinate: vec2<f32>, spatial_weight: f32,
                      depth: ptr<function, f32>, momentum: ptr<function, vec2<f32>>,
                      discharge: ptr<function, f32>, revealed_weight: ptr<function, f32>,
                      surface: ptr<function, vec4<f32>>)
{
    // PORT: the GLSL samples the two atlas textures only after the spatial
    // weight and lookup-record guards below; both samples are hoisted here
    // (uniform control flow, see PORT NOTES 2) and the guards select the
    // original contributions verbatim.
    var record = vec4<f32>(0.0);
    let record_valid = lookupErosionTile(tile_coordinate, &record);
    let atlas_uv = erosionAtlasUV(worldXZ, tile_coordinate, record.xy);
    let tile_flow = textureSample(texture2, texture2_sampler, atlas_uv);
    let surface_sample = textureSample(texture1, texture1_sampler, atlas_uv);

    if (spatial_weight <= 0.0) { return; }
    if (!record_valid) { return; }
    let reveal = clamp(record.z * record.a, 0.0, 1.0);
    let contribution = spatial_weight * reveal;
    (*surface) += contribution * surface_sample;
    let tile_depth = max(tile_flow.r, 0.0);
    (*depth) += contribution * tile_depth;
    (*momentum) += contribution * tile_depth * tile_flow.gb;
    (*discharge) += contribution * max(tile_flow.a, 0.0);
    (*revealed_weight) += contribution;
}

fn blendedFlow(worldXZ: vec2<f32>, coverage: ptr<function, f32>,
               surface: ptr<function, vec4<f32>>) -> vec4<f32>
{
    (*surface) = vec4<f32>(0.0);
    let stride = max(stage.erosion_tile_stride, 0.0001);
    let minimum_tile = floor(worldXZ / stride);
    let tile00 = minimum_tile;
    let tile10 = minimum_tile + vec2<f32>(1.0, 0.0);
    let tile01 = minimum_tile + vec2<f32>(0.0, 1.0);
    let tile11 = minimum_tile + vec2<f32>(1.0, 1.0);

    var spatial_weights = vec4<f32>(erosionTileMask(worldXZ, tile00),
                                    erosionTileMask(worldXZ, tile10),
                                    erosionTileMask(worldXZ, tile01),
                                    erosionTileMask(worldXZ, tile11));
    let geometric_sum = dot(spatial_weights, vec4<f32>(1.0));
    if (geometric_sum <= 0.000001)
    {
        (*coverage) = 0.0;
        return vec4<f32>(0.0);
    }
    spatial_weights = spatial_weights / geometric_sum;

    var depth = 0.0;
    var momentum = vec2<f32>(0.0);
    var discharge = 0.0;
    var revealed_weight = 0.0;
    accumulateFlowTile(worldXZ, tile00, spatial_weights.x,
                       &depth, &momentum, &discharge, &revealed_weight, surface);
    accumulateFlowTile(worldXZ, tile10, spatial_weights.y,
                       &depth, &momentum, &discharge, &revealed_weight, surface);
    accumulateFlowTile(worldXZ, tile01, spatial_weights.z,
                       &depth, &momentum, &discharge, &revealed_weight, surface);
    accumulateFlowTile(worldXZ, tile11, spatial_weights.w,
                       &depth, &momentum, &discharge, &revealed_weight, surface);

    // Momentum is blended alongside depth, so opposing independent currents
    // settle smoothly instead of averaging signed velocity without regard for
    // the amount of water carrying it.
    let velocity = select(vec2<f32>(0.0), momentum / depth, depth > 0.00001);

    let visibility = 1.0 - smoothHermite(stage.erosion_visibility_full_radius,
                                         stage.erosion_visibility_zero_radius,
                                         length(worldXZ - stage.erosion_visibility_center));
    depth = depth * visibility;
    discharge = discharge * visibility;
    (*surface) = (*surface) * visibility;
    let reveal_coverage = revealed_weight * visibility;
    (*coverage) = clamp(reveal_coverage, 0.0, 1.0);
    return vec4<f32>(depth, velocity, discharge);
}

// Material indices match kTerrainMaterials in main.cpp. Grass and dirt form
// one ground cover; dirt and gravel retain two consecutive scan variants,
// while rock carries a single scan (see accumulateGroup).
const SNOW_BASE: i32 = 0;     // kSnowBase
const GRASS_BASE: i32 = 1;    // kGrassBase
const SAND_BASE: i32 = 2;     // kSandBase
const DIRT_BASE: i32 = 3;     // kDirtBase
const GRAVEL_BASE: i32 = 5;   // kGravelBase
const ROCK_BASE: i32 = 7;     // kRockBase

fn rotateUV(uv: vec2<f32>, angle: f32) -> vec2<f32>
{
    let c = cos(angle);
    let s = sin(angle);
    return vec2<f32>(c * uv.x - s * uv.y, s * uv.x + c * uv.y);
}

// Continuous tile coordinates anchor each material's random quarter turns
// in world space. Analytic gradients keep cell boundaries out of mip selection.
// Dirt and gravel alone use a secondary variant at an incommensurate scale.

// Single source of truth for the per-material/per-octave UV rotation. The
// tangent frames below derive from the same angle so normal-map detail always
// aligns with the albedo it belongs to.
fn uvAngle(material_index: i32, octave: i32) -> f32
{
    return f32(material_index) * 0.6 + select(0.9, 0.0, octave == 0);
}

// Keep the selected scans at their existing feature sizes now that they are
// sampled directly: ~1 m powder and ~2.1 m moss/litter tiles at uTexScale 0.16.
fn materialDensity(material_index: i32) -> f32
{
    if (material_index == SNOW_BASE) { return 2.6 * 2.41421356; }
    if (material_index == GRASS_BASE) { return 1.25 * 2.41421356; }
    // Soil stones belong at the same scale as the surrounding grass blades.
    if (material_index == DIRT_BASE || material_index == DIRT_BASE + 1) { return 2.0; }
    // Rock shares the gravel-scale tile: the Rock016 scan's chip scars and
    // crack stains span ~0.1-1.3 m inside one 6 m repeat, so a face reads as
    // weathered cliff detail at close and mid range without duplicating any
    // single feature across the whole mountainside.
    return 1.0;
}

fn tileCoordContinuous(material_index: i32, worldXZ: vec2<f32>, octave: i32) -> vec2<f32>
{
    let scale_mul = select(2.41421356, 1.0, octave == 0);
    let angle = uvAngle(material_index, octave);
    let base = vec2<f32>(fract(f32(material_index) * 0.37 + 0.13),
                         fract(f32(material_index) * 0.71 + 0.29));
    let offset = base + select(vec2<f32>(0.31, 0.17), vec2<f32>(0.0), octave == 0);
    return rotateUV(worldXZ * globals.settings_a.y * scale_mul * materialDensity(material_index), angle) + offset;
}

// Screen-space gradient of the continuous coordinate. The rotation and scale
// in tileCoordContinuous are constant per material/octave, so the gradient of
// p is just the rotated, scaled gradient of the plane coordinate. Taking it
// analytically (instead of dpdx(p)) lets callers precompute derivatives once,
// outside the weight-gated branches below, where dpdx would be undefined.
fn gradContinuous(material_index: i32, octave: i32, d_plane: vec2<f32>) -> vec2<f32>
{
    let scale_mul = select(2.41421356, 1.0, octave == 0);
    return rotateUV(d_plane * globals.settings_a.y * scale_mul * materialDensity(material_index),
                    uvAngle(material_index, octave));
}

// Shear a flat-plane UV direction into the surface tangent plane along the
// projection axis. Subtracting the normal component along the axis leaves the
// direction's projection onto the plane untouched, so the vector still follows
// the actual UV derivative of the mapped texture.
fn shearToSurface(d: vec3<f32>, axis: vec3<f32>, N: vec3<f32>) -> vec3<f32>
{
    let dn = dot(N, axis);
    if (abs(dn) < 0.25) { return normalize(d); }   // projection unused; caller gates
    return normalize(d - axis * (dot(d, N) / dn));
}

// Tangent/bitangent matching tileCoordContinuous for this material/octave.
// eP and eQ are the world axes along which the projection's u and v increase,
// so the normal map's green channel maps onto B directly — B is never derived
// from cross(N, T), which would hand back a left-handed frame for XZ UVs.
// Both directions are sheared into the surface plane, then B is re-orthogonal-
// ised against T because the shear is a single-axis distortion.
fn tangentFrame(eP: vec3<f32>, eQ: vec3<f32>, axis: vec3<f32>, angle: f32, N: vec3<f32>,
                T: ptr<function, vec3<f32>>, B: ptr<function, vec3<f32>>)
{
    let c = cos(angle);
    let s = sin(angle);
    (*T) = shearToSurface(c * eP - s * eQ, axis, N);
    (*B) = shearToSurface(s * eP + c * eQ, axis, N);
    (*B) = normalize((*B) - (*T) * dot((*B), (*T)));
}

// Integer hashing keeps cell variation stable, including across negative
// world coordinates. Each material/octave has an independent seed.
fn surfaceCellHash(cell: vec2<f32>, material_index: i32, octave: i32) -> u32
{
    let key = vec2<u32>(vec2<i32>(cell));
    var h = key.x * 0x9e3779b9u + key.y * 0x85ebca6bu
          + u32(material_index + 1) * 0xc2b2ae35u + u32(octave) * 0x27d4eb2fu;
    h = h ^ (h >> 16u);
    h = h * 0x7feb352du;
    h = h ^ (h >> 15u);
    h = h * 0x846ca68bu;
    h = h ^ (h >> 16u);
    return h;
}

// An unbounded world-space field for ground cover. Unlike the mirrored
// noise texture, this has no short repeat period or reflected patch pairs.
// Quintic interpolation keeps the warped patch boundaries smooth.
fn groundNoise(p: vec2<f32>) -> f32
{
    let cell = floor(p);
    let f = fract(p);
    let w = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let inv_hash_range = 1.0 / 16777215.0;
    let a = f32(surfaceCellHash(cell, 17, 0) & 0xffffffu) * inv_hash_range;
    let b = f32(surfaceCellHash(cell + vec2<f32>(1.0, 0.0), 17, 0) & 0xffffffu) * inv_hash_range;
    let c = f32(surfaceCellHash(cell + vec2<f32>(0.0, 1.0), 17, 0) & 0xffffffu) * inv_hash_range;
    let d = f32(surfaceCellHash(cell + vec2<f32>(1.0), 17, 0) & 0xffffffu) * inv_hash_range;
    return mix(mix(a, b, w.x), mix(c, d, w.x), w.y);
}

// Only call before material branches. Derivatives include the domain warp;
// unresolved octaves settle to their mean instead of sparkling at distance.
fn filteredGroundNoise(p: vec2<f32>) -> f32
{
    let pixel_width = max(length(dpdx(p)), length(dpdy(p)));
    return mix(groundNoise(p), 0.5, smoothHermite(0.35, 1.1, pixel_width));
}

fn rotateQuarterTurn(p: vec2<f32>, turn: i32) -> vec2<f32>
{
    if (turn == 1) { return vec2<f32>(-p.y, p.x); }
    if (turn == 2) { return -p; }
    if (turn == 3) { return vec2<f32>(p.y, -p.x); }
    return p;
}

// Reuse one scan with 0/90/180/270-degree rotations. Ground-cover cells also
// sample independent offsets so the same tuft or stone cannot mark a grid.
// Most of each cell uses
// just its own orientation; the outer 15% blends with the neighbouring cell
// so differently rotated edges never meet at a hard seam. The weights have
// zero slope at their endpoints and form a partition of unity.
// All channels share coordinates, gradients and weights. Colour blends in
// linear space; normals enter world space through each rotated tangent frame
// before blending, keeping lighting aligned with the rotated surface detail.
fn sampleMaterial(material_index: i32, octave: i32, plane_coord: vec2<f32>,
                  d_plane_x: vec2<f32>, d_plane_y: vec2<f32>,
                  eP: vec3<f32>, eQ: vec3<f32>, axis: vec3<f32>, N: vec3<f32>, strength: f32,
                  albedo_ao_out: ptr<function, vec4<f32>>,
                  detail_rough_out: ptr<function, vec4<f32>>)
{
    let p = tileCoordContinuous(material_index, plane_coord, octave);
    let dpdx_p = gradContinuous(material_index, octave, d_plane_x);
    let dpdy_p = gradContinuous(material_index, octave, d_plane_y);
    let cell_base = floor(p - vec2<f32>(0.5));
    let blend = smoothstep(vec2<f32>(0.35), vec2<f32>(0.65), fract(p - vec2<f32>(0.5)));
    (*albedo_ao_out) = vec4<f32>(0.0);
    (*detail_rough_out) = vec4<f32>(0.0);
    for (var y: i32 = 0; y < 2; y++)
    {
        for (var x: i32 = 0; x < 2; x++)
        {
            let weight = select(blend.x, 1.0 - blend.x, x == 0)
                       * select(blend.y, 1.0 - blend.y, y == 0);
            if (weight <= 0.0) { continue; }

            let cell = cell_base + vec2<f32>(f32(x), f32(y));
            let cell_hash = surfaceCellHash(cell, material_index, octave);
            let turn = i32(cell_hash & 3u);
            var uv = rotateQuarterTurn(p - cell - vec2<f32>(0.5), turn) + vec2<f32>(0.5);
            if (material_index != SNOW_BASE) {
                uv += vec2<f32>(f32((cell_hash >> 2u) & 1023u),
                                f32((cell_hash >> 12u) & 1023u)) * (1.0 / 1024.0);
            }
            let dx = rotateQuarterTurn(dpdx_p, turn);
            let dy = rotateQuarterTurn(dpdy_p, turn);
            // GLSL textureGrad(uAlbedoAO, vec3(uv, float(materialIndex)), dx, dy)
            let colour = textureSampleGrad(albedo_ao, albedo_ao_sampler, uv, material_index, dx, dy);
            let nr = textureSampleGrad(normal_rough, normal_rough_sampler, uv, material_index, dx, dy);
            var tangent: vec3<f32>;
            var bitangent: vec3<f32>;
            tangentFrame(eP, eQ, axis, uvAngle(material_index, octave)
                         + f32(turn) * 1.57079632679, N, &tangent, &bitangent);
            let tn = nr.rgb * 2.0 - 1.0;
            let detail = tangent * (tn.x * strength) + bitangent * (tn.y * strength) + N * tn.z;
            (*albedo_ao_out) += weight * vec4<f32>(pow(colour.rgb, vec3<f32>(2.2)), colour.a);
            (*detail_rough_out) += weight * vec4<f32>(detail, nr.a);
        }
    }
}

// Cheap 2D hash for world-anchored random values (snow sparkle cells).
fn hash12(p: vec2<f32>) -> f32
{
    var p3 = fract(vec3<f32>(p.xyx) * 0.1031);
    p3 += vec3<f32>(dot(p3, p3.yzx + 33.33));
    return fract((p3.x + p3.y) * p3.z);
}

// Snow sparkle tuning: hash cells per metre (facets ~4 cm), the share of
// cells that can glint, their facet tilt distribution (maximum tilt in
// radians, concentration exponent), and how sharply alignment with the
// half-vector must hold before a cell lights up. Which tilt population a
// pose can draw on is set by H's tilt from vertical, and the two pose
// families sit at opposite ends of that range: INTO the sun the horizontal
// parts of the sun ray and the camera ray cancel, so H stands nearly
// upright — 6-30 degrees of tilt from standing height out to a far ridge —
// while AWAY from the sun they reinforce and H leans over to 48-62
// degrees. Round 7 built facets from square jitter, whose tilt piles up
// around atan(slope) ~ 55-60 degrees: backwards against real snow, whose
// facet crystals lie mostly near-flat — tilts concentrated at 0-30 degrees
// with a thin steep tail — and that shallow population is exactly what the
// into-sun glitter path draws on. The jitter starved it (round 7 measured
// the into-sun path at 0.01% bright pixels against the photographed
// 0.3-1%, while the anti-solar field rode the jitter's steep peak at 40x
// the intended density). Facets are now polar: tilt = TiltMax*pow(hash,
// TiltShape) concentrates them shallow, and TiltMax's tail keeps a sparse
// anti-solar shimmer alive at every range — none at all reads as a dead
// matte field. Round 8's numeric pass exposed two further couplings that
// this retune fixes. First, acceptance versus lobe width: the applied window
// takes facets aligned to ~13-16 degrees, but the roughness write buys a GGX
// mirror only ~6 degrees wide — every winner in the 6-13 degree band
// rotated fully to a lobe too tight to catch it, reading as the faint
// diffuse chip the rotation comment warns about, which is why the glitter
// bands plateaued at 246-250 instead of blowing past. The write rises to
// 0.13 (lobe ~9 degrees) and Power drops so the acceptance band and the
// lobe agree. Second, disc size: the radius clamp's half-cell ceiling made
// every disc sub-pixel once cells-per-pixel fell under ~2 — exactly the
// 7-15 m midfield, measured 10x sparser than the near band — so the
// ceilings now let discs track ~1 px through each octave's fade instead of
// shrinking under FXAA before the next octave takes over.
const SNOW_SPARKLE_CELLS: f32 = 26.0;          // kSnowSparkleCells
const SNOW_SPARKLE_SHARE: f32 = 0.92;          // kSnowSparkleShare
const SNOW_SPARKLE_TILT_MAX: f32 = 1.19;       // kSnowSparkleTiltMax: radians, ~68 degrees
const SNOW_SPARKLE_TILT_SHAPE: f32 = 2.2;      // kSnowSparkleTiltShape: median tilt ~15 degrees: mass
                                               // on the into-sun H band (16-31)
const SNOW_SPARKLE_POWER: f32 = 21.0;          // kSnowSparklePower
// A second, coarser octave: sastrugi-scale facets stay resolvable — and so
// keep glinting — out to mid distance where the near octave has long since
// faded, their few survivors favoured in winner selection as cells shrink
// toward a pixel (Bowles & Wang, "Sparkly but not too sparkly!", SIGGRAPH
// 2015).
const SNOW_SPARKLE_CELLS_FAR: f32 = 7.0;       // kSnowSparkleCellsFar
const SNOW_SPARKLE_SHARE_FAR: f32 = 0.70;      // kSnowSparkleShareFar
const SNOW_SPARKLE_TILT_MAX_FAR: f32 = 1.19;   // kSnowSparkleTiltMaxFar
const SNOW_SPARKLE_TILT_SHAPE_FAR: f32 = 2.2;  // kSnowSparkleTiltShapeFar
const SNOW_SPARKLE_POWER_FAR: f32 = 24.0;      // kSnowSparklePowerFar
// A third, coarse octave: ~55 cm wind-crust facets stay resolvable to
// several hundred metres, so mid-distance and distant snowfields keep a
// sparse aggregate glint after both finer octaves have faded below the
// pixel footprint. It fades in only as the far octave thins out, then back
// out (hash-jittered, like the others) once the survival gate has already
// thinned the sub-pixel survivors to near zero. Round 9 densifies the
// cells (70 -> 55 cm): the coarse octave carries the midfield, whose
// winner count is azimuth-limited — more cells is the one lever that
// scales that band, which round 8 measured at a tenth of the near band.
const SNOW_SPARKLE_CELLS_COARSE: f32 = 1.8;    // kSnowSparkleCellsCoarse
const SNOW_SPARKLE_SHARE_COARSE: f32 = 0.60;   // kSnowSparkleShareCoarse
const SNOW_SPARKLE_TILT_MAX_COARSE: f32 = 1.22;   // kSnowSparkleTiltMaxCoarse: wind-crust facets lean
const SNOW_SPARKLE_TILT_SHAPE_COARSE: f32 = 1.8;  // kSnowSparkleTiltShapeCoarse: harder than crystals
const SNOW_SPARKLE_POWER_COARSE: f32 = 24.0;   // kSnowSparklePowerCoarse

// Round 9: the application window's graded zone was the dotted-chain
// generator. The old (0.20, 0.40) window fully rotated only winners
// inside ~16 degrees of the half-vector; the 16-21 degree graded annulus
// rotated them PART way — enough to tilt N.L down and pick up the blue
// shadow ambient, never enough for the lobe to catch — and because the
// macro crest tilt systematically raises alignment, those half-rotated
// cells strung along the crest lines at hash-cell spacing, which vision
// measured as dotted chains of dark blue-grey chips (snow_steep's high
// verdict; the same family as snow_ground's scratch hairlines). The
// window narrows to (0.28, 0.38): the dense 16-17 degree population now
// rotates fully and flashes instead of chipping dark, the 18-21 degree
// cells never touch the normal at all, and Power eases to 21 so the
// widened acceptance still matches the lobe. The radius floors rise to
// ~1 px on all octaves for the same reason the ceilings rose in round 8:
// at the closest metres the near discs clamped to 0.2 px, fired, and
// then averaged with the surrounding diffuse under FXAA to 242-249 —
// present in the field, invisible as sparkle — so the closest band read
// dim while the mid band blew (snow_sunfacing: closest ge250 at 0.018%
// against the near band's 0.12%, the glitter gradient upside down).

// Accumulate the ground-cover surfaces, with a second variant for dirt and
// gravel. Albedo, AO, roughness and normal detail all use the same
// material selection. Each normal is transformed through its own UV frame.
// Negligible biome weights skip texture fetches on every projection.
fn accumulateGroup(base: i32, group_weight: f32, plane_coord: vec2<f32>,
                   d_plane_x: vec2<f32>, d_plane_y: vec2<f32>, crossfade: f32,
                   eP: vec3<f32>, eQ: vec3<f32>, axis: vec3<f32>, N: vec3<f32>, strength: f32,
                   albedo_tint: vec3<f32>, albedo_desat: f32, rough_floor: f32,
                   ao_retain: f32, normal_mul: f32,
                   albedo: ptr<function, vec3<f32>>, world_detail: ptr<function, vec3<f32>>,
                   rough: ptr<function, f32>, ao: ptr<function, f32>, weight_sum: ptr<function, f32>)
{
    if (group_weight <= 0.01) { return; }

    var albedo_ao_sample: vec4<f32>;
    var detail_rough_sample: vec4<f32>;
    let detail_strength = strength * normal_mul;
    sampleMaterial(base, 0, plane_coord, d_plane_x, d_plane_y,
                   eP, eQ, axis, N, detail_strength, &albedo_ao_sample, &detail_rough_sample);

    if (base == DIRT_BASE || base == GRAVEL_BASE)
    {
        var variant_albedo_ao: vec4<f32>;
        var variant_detail_rough: vec4<f32>;
        sampleMaterial(base + 1, 1, plane_coord, d_plane_x, d_plane_y,
                       eP, eQ, axis, N, detail_strength, &variant_albedo_ao, &variant_detail_rough);
        albedo_ao_sample = mix(albedo_ao_sample, variant_albedo_ao, crossfade);
        detail_rough_sample = mix(detail_rough_sample, variant_detail_rough, crossfade);
    }

    var sampled_albedo = albedo_ao_sample.rgb;
    sampled_albedo = mix(sampled_albedo,
                         vec3<f32>(dot(sampled_albedo, vec3<f32>(0.299, 0.587, 0.114))), albedo_desat);
    if (base == ROCK_BASE)
    {
        // Rock032 is a brighter, more contrasty scan than the Rock016 this
        // stretch was first cut for (mean 0.099 vs 0.068 linear, native
        // p1->p50 span already 40 sRGB levels vs Rock016's 27), so the
        // slope eases 1.35 -> 1.20 around the new scan's measured mean.
        // The darker rock tint below compresses the span back in sRGB, and
        // 1.20 restores the same ~36-level rendered crack depth the Rock016
        // round was tuned to, while the gentler slope holds the zero clamp
        // at 0.5% of texels (1.35 would clamp 2.3% into pinprick black
        // speckle). Clamped at zero because the deepest pores land
        // fractionally negative after the stretch.
        sampled_albedo = max((sampled_albedo - vec3<f32>(0.099)) * 1.20 + vec3<f32>(0.099),
                             vec3<f32>(0.0));
    }
    (*albedo) += group_weight * sampled_albedo * albedo_tint;
    // AO retention scales cavity contrast, with zero giving an unoccluded surface.
    (*ao) += group_weight * (1.0 - (1.0 - albedo_ao_sample.a) * ao_retain);
    (*rough) += group_weight * max(detail_rough_sample.a, rough_floor);
    (*world_detail) += group_weight * detail_rough_sample.xyz;
    (*weight_sum) += group_weight;
}

@fragment
fn fs_main(input: FsInput) -> FsOutput
{
    var output: FsOutput;

    let frag_position_view = input.frag_position_view;
    let frag_world_position = input.frag_world_position;

    var normalWorld = normalize(input.frag_world_normal);
    let height = frag_world_position.y;
    let slope = 1.0 - clamp(normalWorld.y, 0.0, 1.0);
    let worldXZ = frag_world_position.xz;
    // Evaluate derivatives before any material gates. Explicit gradients keep
    // the mirrored patch fields filtered without seams at their folds.
    let worldStepX = dpdx(worldXZ);
    let worldStepY = dpdy(worldXZ);
    // Horizontal bearing toward the sun, for the mottle-to-drift coupling
    // in the near-field detail block below.
    let toSunXZ = normalize(vec2<f32>(-globals.sun_direction.x, -globals.sun_direction.z));

    // Use the same four-tile reveal and distance fade as the actual landform.
    // Discharge concentration distinguishes channels from the thin sheet of
    // rainwater present across the simulation; speed alone is not evidence
    // of a river, especially in nearly dry cells.
    var flowDomain: f32;
    var surface: vec4<f32>;
    let flow = blendedFlow(worldXZ, &flowDomain, &surface);
    let dischargeAmount = 1.0 - exp(-max(flow.a, 0.0) * 0.035);
    // Centimetres of fresh sediment can cover vegetation; stripping the
    // rooted soil takes a deeper cut. Calibrate against the actual solver's
    // centimetre-to-metre output rather than forcing vegetation to survive.
    let incision = 1.0 - exp(-max(-surface.r - 0.012, 0.0) * 3.0);
    let deposition = 1.0 - exp(-max(surface.r - 0.004, 0.0) * 12.0);
    // Ring concavity/convexity re-anchored to the field's real scales: the
    // 12 m-ring relief of a common swale is 0.1-0.5 m, so the old (0.15, 2.5)
    // ramp passed only 0.001-0.06 — every systematic snow hold landed under
    // half a metre against the +/-15 m wander octaves. (0.05, 1.2) puts a
    // 0.3 m swale at ~0.12 and saturates at 1.2 m bowls. These gates feed only
    // the snow hold path (dryHollow -> snowHeight and grassDryness); the
    // channel, incision and deposition gates below are untouched.
    let hollow = smoothHermite(0.05, 1.2, surface.g);
    let ridge = smoothHermite(0.05, 1.2, -surface.g);
    let channel = smoothHermite(0.12, 1.25, surface.b)
                * smoothHermite(0.04, 0.35, dischargeAmount);
    let transport = smoothHermite(0.6, 4.0, min(length(flow.gb), 8.0))
                  * smoothHermite(0.002, 0.025, flow.r) * channel;

    // Low-frequency drift lets the grass boundary with the beach breathe a
    // few metres instead of tracing one deterministic contour. Frequency
    // calibration: the CPU noise generator evaluates FastNoiseLite at
    // INTEGER texel coordinates over the 1024-texel texture (base
    // wavelength ~118 texels), so the texture spans ~8.7 base-octave
    // wavelengths and each octave's coherent blobs are ~0.06-0.12 UV
    // across. A worldXZ multiplier of m therefore puts real features at
    // ~(0.06-0.12)/m metres — NOT 1/m. The 0.0045 multiplier is chosen for
    // a ~20 m feature scale (the earlier 0.011 was authored against the
    // wrong scale and landed at 0.9 m).
    let grassDrift = textureSample(texture0, texture0_sampler,
                                   mirrorTile(rotateUV(worldXZ * 0.0045, 0.85)
                                              + vec2<f32>(0.43, 0.67))).r;
    let grassHeight = height + (grassDrift - 0.5) * 7.0;

    // Sand gives way to grass, then soil as cover thins, then exposed rock.
    // Eroded drainage and retained snow overlay that ground progression.
    // A wider band here survives grazing sightlines but reads as one
    // painted shore stripe at distance. The exponential altitude profile
    // halves the waterline gradient, so 0.6-5 m spans the same shore strip
    // while the drift noise above breaks its contour.
    let aboveBeach = smoothHermite(stage.sea_level + 0.6, stage.sea_level + 5.0, grassHeight);
    let fGrass = aboveBeach;
    // Regional growth varies over tens to hundreds of metres. A slow warp
    // bends the smaller fields so openings bunch together, stretch and vary
    // in size instead of making evenly spaced, same-size islands of dirt.
    // Keep this slow warp derivative-free: the filtered fields below must
    // differentiate the domain once, without undefined higher derivatives.
    let groundWarp = vec2<f32>(
        groundNoise(rotateUV(worldXZ * 0.011, 0.37) + vec2<f32>(17.3, -9.1)),
        groundNoise(rotateUV(worldXZ * 0.009, 1.21) + vec2<f32>(-31.7, 23.4)));
    let groundDomain = worldXZ + (groundWarp - 0.5) * 48.0;
    let groundRegion = filteredGroundNoise(
        rotateUV(groundDomain * 0.007, 0.81) + vec2<f32>(51.8, 7.2));
    let groundGrowth = filteredGroundNoise(
        rotateUV(groundDomain * 0.021, 1.73) + vec2<f32>(-12.5, 39.6));
    let soilLarge = filteredGroundNoise(
        rotateUV(groundDomain * 0.043, 0.57) + vec2<f32>(6.2, 81.3));
    let soilSmall = filteredGroundNoise(
        rotateUV(groundDomain * 0.137, 1.13) + vec2<f32>(73.1, -24.8));
    let soilEdge = filteredGroundNoise(
        rotateUV(groundDomain * 0.83, 2.04) + vec2<f32>(-41.6, -63.2));
    let soilPattern = mix(soilLarge, soilSmall, mix(0.18, 0.62, groundGrowth))
                    + (soilEdge - 0.5) * 0.16;
    let soilThreshold = mix(0.66, 0.43, groundRegion);
    let soilPatches = smoothHermite(soilThreshold, soilThreshold + 0.22, soilPattern);
    // Fine sediment favours flats. Scouring can also expose soil on banks,
    // with retention fading as steep ground gives way to rock.
    let soilFlatness = 1.0 - smoothHermite(0.02, 0.25, slope);
    let soilRetention = 1.0 - smoothHermite(0.12, 0.48, slope);
    let backgroundSoil = 0.015 + 0.18 * soilPatches * mix(0.55, 1.0, groundRegion);
    let depositedSoil = deposition * soilFlatness;
    let scouredSoil = incision * soilRetention;
    let channelSoil = channel * (1.0 - transport) * soilRetention;
    // Bounded coverage union: any strong disturbance can displace grass.
    // Noise varies the transition edges without weakening fully bare ground.
    var disturbedSoil = 1.0 - (1.0 - depositedSoil) * (1.0 - scouredSoil)
                              * (1.0 - channelSoil);
    disturbedSoil += (soilEdge - 0.5) * 0.6 * disturbedSoil * (1.0 - disturbedSoil);
    // Soil depth follows the landform, with no altitude gate for stone.
    // Broad geology and smaller weathering patches persist outside the
    // erosion cache. Incised, convex ground exposes resistant beds; hollows
    // and deposited fines retain cover. Hardness only amplifies incision so
    // the protected spawn's high hardness cannot paint a ring of rock.
    let rockRegion = filteredGroundNoise(
        rotateUV(groundDomain * 0.0031, 0.26) + vec2<f32>(-57.4, 18.9));
    let resistantBed = clamp(surface.a / 0.56, 0.0, 1.0);
    let terrainExposure = slope
                        + (rockRegion - 0.5) * 0.22
                        + (groundGrowth - 0.5) * 0.10
                        + (soilLarge - 0.5) * 0.04
                        + incision * mix(0.10, 0.24, resistantBed)
                        + ridge * 0.10 - hollow * 0.08
                        - deposition * soilRetention * (1.0 - transport) * 0.24;
    // Both transitions share this field: grass has completely yielded to
    // dirt before any rock appears, including at erosion-cut boundaries.
    // This preserves a soil shoulder instead of blending grass into stone.
    let exposedSoil = smoothHermite(0.085, 0.22, terrainExposure);
    let grassSoilBlend = 1.0 - (1.0 - backgroundSoil * soilFlatness)
                               * (1.0 - disturbedSoil) * (1.0 - exposedSoil);
    let fRock = smoothHermite(0.24, 0.52, terrainExposure);
    // Gravel is the eroded mountain: energetic drainage cuts through the
    // rock faces and leaves coarse debris along its gullies. Substrate
    // hardness keeps the coarsest material on the hardest beds.
    let scouredGravel = incision * mix(0.35, 1.0, transport)
                      * mix(0.75, 1.0, clamp(surface.a / 0.56, 0.0, 1.0));
    let fGravel = scouredGravel;
    // Fine sediment bars occur where concentrated runoff slows and deposits
    // material. Shore sand remains the coastal base, including underwater.
    let sedimentSand = deposition * channel * (1.0 - transport) * soilFlatness;
    let fSand = max(1.0 - fGrass, sedimentSand);

    // Regional climate and local drifts use repeatable world-space fields
    // on EVERY substrate, including rock. The broad ~600 m variation keeps
    // entire slopes from sharing a snowline; ~77 / 20 / 7 m detail breaks
    // up its edge and filters away only when smaller than a pixel.
    let snowRegion = filteredGroundNoise(
        rotateUV(groundDomain * 0.0017, 0.63) + vec2<f32>(91.7, -53.2));
    let snowDrift = filteredGroundNoise(
        rotateUV(groundDomain * 0.013, 1.19) + vec2<f32>(-23.8, 67.4));
    let snowDriftFine = filteredGroundNoise(
        rotateUV(groundDomain * 0.05, 0.41) + vec2<f32>(37.1, 12.6));
    let snowDriftMicro = filteredGroundNoise(
        rotateUV(groundDomain * 0.15, 1.87) + vec2<f32>(-61.3, -42.9));
    // Snow settles like sediment: closed hollows retain deeper beds lower
    // down, while eroded ridges and sun-facing slopes expose the substrate
    // earlier. Active watercourses (channel) flush what falls into them,
    // so only dry hollows hold the pack. Hydraulic water is a landforming
    // signal here, not a temperature or snowmelt simulation.
    // Weak trickles do not scour a pack: the dry gate keys to actual
    // discharge, so intermittently flushed gullies keep part of their hold
    // while continuously running beds lose all of it.
    let flushSnow = smoothHermite(0.08, 0.30, dischargeAmount);
    let dryHollow = hollow * (1.0 - channel * mix(0.45, 1.0, flushSnow));
    let sunExposure = max(dot(normalWorld, -normalize(globals.sun_direction.xyz)), 0.0);
    // Height supplies a broad climate bias, not a shared material cutoff.
    // Curvature and solar aspect shift local retention by comparable amounts
    // to the drift fields. Aspect remains active beyond the erosion cache.
    let snowHeight = height + (snowRegion - 0.5) * 100.0
                            + (snowDrift - 0.5) * 42.0
                            + (snowDriftFine - 0.5) * 16.0
                            + (snowDriftMicro - 0.5) * 6.0
                            + dryHollow * 28.0 - ridge * 18.0
                            - (sunExposure - 0.65) * 22.0;
    let snowLine = smoothHermite(stage.sea_level + 92.0, stage.sea_level + 126.0, snowHeight);
    let snowHold = smoothHermite(0.12, 0.72, normalWorld.y);
    var fSnow = snowLine * snowHold;

    // Wind shapes the pack the way altitude noise alone cannot: prevailing
    // wind scours exposed windward aspects back toward bare ground while lee
    // slopes hold their cover, so the snowline follows terrain aspect as well
    // as height. The aspect term is slope-gated so flat snowfields (degenerate
    // normalWorld.xz) are left alone.
    let aspectN = normalize(normalWorld.xz + vec2<f32>(1e-4, 0.0));
    let windAlignment = dot(aspectN, normalize(vec2<f32>(0.70, 0.42)));
    let scour = smoothHermite(0.15, 0.60, -windAlignment)
              * smoothHermite(0.08, 0.30, slope);
    let leeDrift = smoothHermite(0.20, 0.70, windAlignment) * snowLine;
    fSnow = clamp(fSnow * (1.0 - 0.70 * scour) + 0.12 * leeDrift, 0.0, 1.0);

    // Thin pack opens over convex noses and active cuts at every viewing
    // distance. Camera movement must not change where a surface holds snow.
    let marginBand = snowLine * (1.0 - snowLine) * 4.0;
    let marginShed = clamp(0.9 * ridge + smoothHermite(0.12, 0.50, slope)
                         + 0.5 * incision + 0.5 * channel, 0.0, 1.0);
    fSnow = fSnow * (1.0 - 0.70 * marginBand * marginShed);

    // Sediment pack vs active drainage: the dry hollows that thicken the
    // pack above also let it settle lower down, but the watercourses that
    // cut the mountain keep their beds clear — a snowpack settles, it does
    // not coat. Discharge concentration drives the cut through a dedicated
    // low-edged gate: tributaries sit at 1.1-1.9x the ring concentration
    // (surface.b 0.15-0.9), inside the old (0.12, 1.25) gate's dead low end
    // where the cut parked at <=0.17 of fSnow and read as nothing at all,
    // so (0.04, 0.70) enters at the first measurable concentration. The
    // transport floor rises 0.50 -> 0.70 so idle tributaries still read as
    // cleared ribbon (a partial strip at landscape range reads as nothing);
    // incision extends the cut to the discharge-starved gorges high on the
    // mountain, and deposition keeps slack-water flats snowy. The hardness
    // factor moves from inside the cut to the ceiling, mirroring
    // scouredGravel from the other side: hard beds melt through to 8% snow
    // retention (the old 25% film read as pale snow, never as a bed), while
    // soft beds keep the same thin skin as before.
    let channelCut = smoothHermite(0.04, 0.70, surface.b)
                   * smoothHermite(0.04, 0.35, dischargeAmount);
    var snowCut = channelCut * mix(0.70, 1.0, transport)
                + 0.60 * incision * (1.0 - deposition);
    snowCut = clamp(snowCut, 0.0, 1.0);
    fSnow = fSnow * (1.0 - snowCut * mix(0.92, 0.55, 1.0 - clamp(surface.a / 0.56, 0.0, 1.0)));

    // Inactive, shaded gully cores retain fingers below their local climate
    // margin. The cubed hollow gate concentrates these in real troughs;
    // the shared snowHeight field keeps their lower edge irregular too.
    let gullyShade = 1.0 - 0.85 * smoothHermite(0.30, 0.85, sunExposure);
    let fingerBand = smoothHermite(stage.sea_level + 70.0, stage.sea_level + 110.0,
                                   snowHeight + dryHollow * 8.0);
    let gullyFinger = 0.92 * flowDomain * (dryHollow * dryHollow * dryHollow) * gullyShade
                    * (1.0 - 0.45 * incision)
                    * (0.30 + 0.70 * snowHold)
                    * fingerBand;
    fSnow = 1.0 - (1.0 - fSnow) * (1.0 - gullyFinger);

    // Steep-face shed: a settling pack coats slabs and ledges but cannot
    // cling to walls, so real mountains stay bare rock above roughly the
    // 50-60 degree shedding angle no matter the altitude — the high face
    // reads as dark rock cut by drainage ribbons instead of one pale
    // snow-flooded continuum. The gate starts at ~51 degrees and saturates
    // at ~59; below it the tuned margin, belt and snowfield reads are
    // untouched, and like snowHold it is a smooth per-pixel hermite on a
    // continuous field with no distance gating.
    fSnow = fSnow * (1.0 - 0.85 * smoothHermite(0.36, 0.52, slope));

    let groundCover = (1.0 - fSand) * (1.0 - fGravel) * (1.0 - fRock) * (1.0 - fSnow);
    let wSand = fSand * (1.0 - fGravel) * (1.0 - fRock) * (1.0 - fSnow);
    let wGrass = groundCover * (1.0 - grassSoilBlend);
    let wDirt = groundCover * grassSoilBlend;
    // The eroded cuts win over the intact faces they carve, so a gully
    // through rock reads as gravel in its bed with rock on either wall.
    let wRock = fRock * (1.0 - fGravel) * (1.0 - fSnow);
    let wGravel = fGravel * (1.0 - fSnow);
    let wSnow = fSnow;

    // Surface weights blend between biomes at shores, slopes and snowlines.
    let gSnow = wSnow;
    let gGrass = wGrass;
    let gSand = wSand;
    let gDirt = wDirt;
    let gGravel = wGravel;
    let gRock = wRock;

    // Ground103 leads the soil, with a little Ground106 variation.
    let nDirt = 0.25 * filteredGroundNoise(
        rotateUV(groundDomain * globals.settings_a.w * 8.0, 0.73) + vec2<f32>(19.4, 63.1));
    let nGravel = filteredGroundNoise(
        rotateUV(groundDomain * globals.settings_a.w * 8.0, 1.61) + vec2<f32>(-37.2, 11.8));

    // Broad green-to-olive growth variation survives when individual blades
    // mip away. Its scale differs from the soil openings to avoid outlining
    // every dirt patch with the same coloured halo.
    var grassDryness = smoothHermite(0.18, 0.84,
                                     groundRegion * 0.65 + groundGrowth * 0.35);
    grassDryness = clamp(grassDryness - dryHollow * 0.22
                         - dischargeAmount * (1.0 - transport) * 0.12, 0.0, 1.0);
    let grassTint = mix(vec3<f32>(0.22, 0.33, 0.24), vec3<f32>(0.28, 0.33, 0.20), grassDryness)
                  * mix(0.94, 1.06, groundGrowth);
    // Neutralise the powder scan's blue cast while leaving headroom for its
    // grain and wind relief in sunlight. Shadow colour comes from sky lighting.
    let snowTint = vec3<f32>(1.08, 1.03, 0.99);

    var albedo = vec3<f32>(0.0);
    var worldDetail = vec3<f32>(0.0);
    var rough = 0.0;
    var ao = 0.0;
    var weightSum = 0.0;

    // Triplanar atlas mapping. Plane coordinates and their screen derivatives
    // are taken up front, outside every weight gate below: derivative calls
    // are undefined in non-uniform control flow. Sampling inside the gates
    // remains well-defined because sampleMaterial feeds textureSampleGrad
    // explicit gradients.
    let coordY = frag_world_position.xz;   // top:  u along +x, v along +z
    let coordX = frag_world_position.zy;   // side X: u along +z, v along +y
    let coordZ = frag_world_position.xy;   // side Z: u along +x, v along +y
    let dYx = dpdx(coordY); let dYy = dpdy(coordY);
    let dXx = dpdx(coordX); let dXy = dpdy(coordX);
    let dZx = dpdx(coordZ); let dZy = dpdy(coordZ);

    // Largest screen-space world footprint across the three projections: the
    // patch of ground one pixel covers, used by the scan detail fade below,
    // the normal prefilter, the near-field detail fades and the sparkle
    // distance fades. Computed before the weight-gated groups because every
    // gated branch below needs it.
    let footprint = max(max(length(dYx), length(dYy)),
                        max(max(length(dXx), length(dXy)),
                            max(length(dZx), length(dZy))));
    // Ease scan normal detail at sub-millimetre footprints to keep magnified
    // texels from reading as hard facets. Albedo retains its filtered detail.
    let scanDetailFade = mix(0.35, 1.0, smoothHermite(0.0008, 0.0045, footprint));

    // Sharp slope weights keep flat ground on the top projection alone (both
    // side gates fall below the 0.02 threshold until the surface tilts
    // well past 30 degrees), so open terrain costs no extra fetches and only
    // cliff pixels pay for the side projections.
    var slopeWeights = pow(abs(normalWorld), vec3<f32>(6.0));
    slopeWeights = slopeWeights / (slopeWeights.x + slopeWeights.y + slopeWeights.z);

    var groupBases = array<i32, 6>(SNOW_BASE, GRASS_BASE, SAND_BASE,
                                   DIRT_BASE, GRAVEL_BASE, ROCK_BASE);
    var groupWeights = array<f32, 6>(gSnow, gGrass, gSand,
                                     gDirt, gGravel, gRock);
    var crossfades = array<f32, 6>(0.0, 0.0, 0.0,
                                   nDirt, nGravel, 0.0);
    var albedoTints = array<vec3<f32>, 6>(
        snowTint, grassTint,
        // Muted mineral sand suits a cold coastline; dampness still follows flow.
        vec3<f32>(0.34, 0.36, 0.38),
        vec3<f32>(0.24, 0.22, 0.18),
        // Weathered gravel should sit within the same exposure as the turf,
        // including the small patches newly exposed in drainage channels.
        vec3<f32>(0.52, 0.53, 0.50),
        // Rock032 is a brighter scan than the Rock016 the committed rock round
        // was measured on (mean 0.099 vs 0.068 linear, with a faint warm
        // cast), so the tint re-solves to land the SAME rendered per-channel
        // means that round calibrated: ~2/3 of gravel's effective brightness
        // (Gravel040 at tint 0.52 lands ~0.066 linear; this lands ~0.045) with
        // the same faint cool cast, so intact faces read darker than both the
        // loose debris cutting them and the snowpack above the snowline —
        // fresh scree's angular facets catching light against shaded bedrock
        // — while the composite's ambient floor keeps the scan's darks well
        // clear of silhouette. Without this re-anchor the brighter scan
        // lands rock AT parity with the gravel beds, inverting the ordering.
        vec3<f32>(0.449, 0.442, 0.482));
    var albedoDesats = array<f32, 6>(0.0, 0.20, 0.28,
                                     0.12, 0.0, 0.0);
    var roughFloors = array<f32, 6>(0.72, 0.72, 0.70,
                                    0.88, 0.80, 0.65);
    // Rock's own roughness map (Rock032) runs mean 0.70 with p5 0.57, so
    // its floor sits below the map's mean instead of at it: clamping at
    // 0.70 would cut away half the map and erase the scar pockets that
    // catch the light. 0.65 preserves the same below-mean share (~21%)
    // the Rock016 round kept with its 0.70 floor.
    // aoRetain (cavity contrast, see accumulateGroup): the AO map's darks
    // are what make scanned surfaces read as relief — blade gaps, thatch
    // shadows, snow pore shadows. Flattening them past ~25% turns a meadow
    // into felt and powder into smudge, so both biomes keep most of their
    // cavity structure and rely on SSAO for the landscape-scale darkening
    // instead (the maps' deep soles still ease so folds never double-darken
    // to mud).
    var aoRetains = array<f32, 6>(0.75, 0.85, 1.0,
                                  0.90, 1.0, 1.0);
    var normalMuls = array<f32, 6>(1.0, 1.0, 1.0,
                                   0.85, 1.0, 0.34);
    // Rock032's normal map runs ~2.1x stronger than Rock016's (mean
    // deviation from flat 0.156 vs 0.076), so the rock multiplier halves
    // to 0.34 to hold the relief strength the committed round tuned.

    // World-space detail accumulates across projections and biome groups and
    // is normalised once at the end, so cliffs pick up side-projected texture
    // with no vertical smear and the flat-to-cliff crossover stays continuous.
    if (slopeWeights.y >= 0.02) {
        for (var g: i32 = 0; g < 6; g++) {
            accumulateGroup(groupBases[g], groupWeights[g] * slopeWeights.y, coordY,
                            dYx, dYy, crossfades[g],
                            vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 1.0, 0.0),
                            normalWorld, globals.settings_b.x * scanDetailFade,
                            albedoTints[g], albedoDesats[g], roughFloors[g],
                            aoRetains[g], normalMuls[g],
                            &albedo, &worldDetail, &rough, &ao, &weightSum);
        }
    }
    if (slopeWeights.x >= 0.02) {
        for (var g: i32 = 0; g < 6; g++) {
            accumulateGroup(groupBases[g], groupWeights[g] * slopeWeights.x, coordX,
                            dXx, dXy, crossfades[g],
                            vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0),
                            normalWorld, globals.settings_b.x * scanDetailFade,
                            albedoTints[g], albedoDesats[g], roughFloors[g],
                            aoRetains[g], normalMuls[g],
                            &albedo, &worldDetail, &rough, &ao, &weightSum);
        }
    }
    if (slopeWeights.z >= 0.02) {
        for (var g: i32 = 0; g < 6; g++) {
            accumulateGroup(groupBases[g], groupWeights[g] * slopeWeights.z, coordZ,
                            dZx, dZy, crossfades[g],
                            vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 0.0, 1.0),
                            normalWorld, globals.settings_b.x * scanDetailFade,
                            albedoTints[g], albedoDesats[g], roughFloors[g],
                            aoRetains[g], normalMuls[g],
                            &albedo, &worldDetail, &rough, &ao, &weightSum);
        }
    }

    let denom = max(weightSum, 1e-4);
    albedo = albedo / denom;
    rough = rough / denom;
    ao = ao / denom;

    // The blended world-space detail vector perturbs the geometric normal
    // before a single rotation into view space. Because every tangent frame is
    // anchored in world space (and matched to its material's UV rotation), the
    // normal detail stays pinned to the terrain as the camera moves instead of
    // swimming with the view, and it stays aligned with its own albedo.
    var perturbedWorld = normalize(worldDetail / max(weightSum, 1e-4));

    // Gentle wind relief over the single powder scan, at ~22 m and ~8 m
    // wavelengths. This perturbs shading only, keeping soft sun/lee variation
    // as the scan mips out without changing the terrain silhouette.
    var snowGrainValue = 0.5;
    var snowCrystalValue = 0.5;
    // The macro gradient is also consumed by the mottle coupling in the
    // near-field detail block below, so it outlives this gate.
    var macroGradient = vec2<f32>(0.0);
    if (gSnow > 0.02)
    {
        let kMacroDriftFreq = 0.004;    // ~22 m wavelength
        let kMacroDriftHeight = 9.0;    // noise-height multiplier
        let kMacroDriftStep = 2.5;      // gradient tap separation, m
        let kMacroSastrugiFreq = 0.011; // ~8 m wavelength
        let kMacroSastrugiHeight = 3.5;
        let kMacroSastrugiStep = 1.0;
        // Domains rotated, and mutually decorrelated: the base noise field
        // is axis-aligned, so unrotated drift/sastrugi contours ran parallel
        // to the world axes and read at grazing distance as long combed
        // windrow wisps. The taps run inside the rotated frame, so the
        // measured gradient is rotated back to world axes here — the normal
        // tilt and the mottle's sun coupling both consume world-space slope.
        let driftUV = rotateUV(worldXZ * kMacroDriftFreq, 0.40) + vec2<f32>(0.37, 0.91);
        let driftUVdx = rotateUV(worldStepX * kMacroDriftFreq, 0.40);
        let driftUVdy = rotateUV(worldStepY * kMacroDriftFreq, 0.40);
        let driftTapX = vec2<f32>(kMacroDriftStep * kMacroDriftFreq, 0.0);
        let driftTapZ = vec2<f32>(0.0, kMacroDriftStep * kMacroDriftFreq);
        // Screen-derivative sampling, like the near-field octaves below.
        // These fetches were raw texture() reads: the macro domains are so
        // low-frequency that auto-mip never engaged, so every range sampled
        // LOD 0, and the bilinear texel steps leaked into the
        // finite-difference gradient as 1-2 px scratch hairlines and beaded
        // dash chains along the crest contours at mid/far range — the
        // dotted-line artifact family round 9's vision pass flagged on
        // three snow poses (snow_ground's 20-150 px scratches sit exactly
        // on the grain octave's fold spacing; snow_steep's dash chains ride
        // the crest lines). textureSampleGrad (GL textureGrad) hands the
        // sampler the true footprint so the field mip-filters coherently
        // with distance.
        let driftCentre = textureSampleGrad(texture0, texture0_sampler, mirrorTile(driftUV),
                                            driftUVdx, driftUVdy).r;
        let driftGradRot = vec2<f32>(
            textureSampleGrad(texture0, texture0_sampler, mirrorTile(driftUV + driftTapX),
                              driftUVdx, driftUVdy).r - driftCentre,
            textureSampleGrad(texture0, texture0_sampler, mirrorTile(driftUV + driftTapZ),
                              driftUVdx, driftUVdy).r - driftCentre)
          * (kMacroDriftHeight / kMacroDriftStep);
        let sastrugiUV = rotateUV(worldXZ * kMacroSastrugiFreq, 1.05) + vec2<f32>(0.83, 0.47);
        let sastrugiUVdx = rotateUV(worldStepX * kMacroSastrugiFreq, 1.05);
        let sastrugiUVdy = rotateUV(worldStepY * kMacroSastrugiFreq, 1.05);
        let sastrugiTapX = vec2<f32>(kMacroSastrugiStep * kMacroSastrugiFreq, 0.0);
        let sastrugiTapZ = vec2<f32>(0.0, kMacroSastrugiStep * kMacroSastrugiFreq);
        let sastrugiCentre = textureSampleGrad(texture0, texture0_sampler, mirrorTile(sastrugiUV),
                                               sastrugiUVdx, sastrugiUVdy).r;
        let sastrugiGradRot = vec2<f32>(
            textureSampleGrad(texture0, texture0_sampler, mirrorTile(sastrugiUV + sastrugiTapX),
                              sastrugiUVdx, sastrugiUVdy).r - sastrugiCentre,
            textureSampleGrad(texture0, texture0_sampler, mirrorTile(sastrugiUV + sastrugiTapZ),
                              sastrugiUVdx, sastrugiUVdy).r - sastrugiCentre)
          * (kMacroSastrugiHeight / kMacroSastrugiStep);
        let driftGrad = rotateUV(driftGradRot, -0.40);
        let sastrugiGrad = rotateUV(sastrugiGradRot, -1.05);
        // mirrorTile folds at every odd integer of the pre-reflection
        // coordinate: the field value is C0 across the cusp but its slope
        // flips sign, so a gradient tap straddling the fold doubles and
        // paints a 1-2 px dark hairline along it. cos zeroes exactly on
        // those cusps, so easing the tilt to zero within a few pixels of
        // every fold kills the line without touching the field between
        // folds.
        let driftFold = smoothHermite(0.0, 0.05,
            min(abs(cos(1.5707963 * driftUV.x)), abs(cos(1.5707963 * driftUV.y))));
        let sastrugiFold = smoothHermite(0.0, 0.05,
            min(abs(cos(1.5707963 * sastrugiUV.x)), abs(cos(1.5707963 * sastrugiUV.y))));
        macroGradient = driftGrad * driftFold + sastrugiGrad * sastrugiFold;
        perturbedWorld = normalize(perturbedWorld
                                 + vec3<f32>(-macroGradient.x, 0.0, -macroGradient.y) * gSnow);

        // Near-field wind grain. The scans' own normal detail magnifies to
        // soft nothing at the camera's feet, yet photographic snow is at its
        // most textured up close: centimetre-scale crust facets tilt 15-30
        // degrees and read as granular micro-shadow. A ~3 cm noise octave
        // tilts the normal where the pixel footprint can still resolve it,
        // fading out as the grain approaches pixel size (the noise texture's
        // mip chain keeps the octave itself from aliasing on the way out).
        // Fade window widened (0.03 -> 0.08): with the 3 px tap below the
        // octave's pixel-scale salt is gone, so the ~19 cm crust texture
        // stays resolvable through the mid band instead of handing the
        // whole 3-8 cm/px range to the mottle alone.
        let grainFade = 1.0 - smoothHermite(0.003, 0.08, footprint);
        if (grainFade > 0.001)
        {
            let kGrainFreq = 0.48;   // ~19 cm wind-crust undulation
                                     // (the noise texture spans ~8.7
                                     // base-octave wavelengths, so
                                     // features sit at ~0.09/m metres)
            let kGrainSlope = 0.9;
            let grainUV = rotateUV(worldXZ * kGrainFreq, 0.44) + vec2<f32>(0.11, 0.71);
            // Screen-derivative sampling, not LOD 0 (see the crystal block
            // below for the aliasing that forced the change).
            let grainUVdx = rotateUV(worldStepX * kGrainFreq, 0.44);
            let grainUVdy = rotateUV(worldStepY * kGrainFreq, 0.44);
            // 3 px tap, capped at a third of the ~0.09 UV feature so it never
            // steps past the undulation at the fade's outer edge: the 1 px
            // tap differenced the fetched field at the pixel scale, where it
            // still carries per-pixel jitter, and that read out as the 2-3 px
            // checkerboard salt the artifact hunter measured on the nearest
            // snow crops (negative lag-2 autocorrelation, period 2-3 px
            // against the intended 4-8).
            let grainTap = min(max(length(grainUVdx), length(grainUVdy)) * 3.0,
                               0.03);
            let grainCentre = textureSampleGrad(texture0, texture0_sampler, mirrorTile(grainUV),
                                                grainUVdx, grainUVdy).r;
            // Expand around the mean to retain soft grain contrast up close.
            snowGrainValue = clamp((grainCentre - 0.5) * 1.8 + 0.5, 0.0, 1.0);
            let grainX = textureSampleGrad(texture0, texture0_sampler,
                                           mirrorTile(grainUV + vec2<f32>(grainTap, 0.0)),
                                           grainUVdx, grainUVdy).r
                       - grainCentre;
            let grainZ = textureSampleGrad(texture0, texture0_sampler,
                                           mirrorTile(grainUV + vec2<f32>(0.0, grainTap)),
                                           grainUVdx, grainUVdy).r
                       - grainCentre;
            // Fold mask (see the macro block): a tap straddling a mirrorTile
            // cusp doubles the measured slope and paints a hairline along the
            // fold — here at 4.2 m spacing, which projects into the 20-150 px
            // scratch family vision measured on snow_ground.
            let grainFold = smoothHermite(0.0, 0.05,
                min(abs(cos(1.5707963 * grainUV.x)), abs(cos(1.5707963 * grainUV.y))));
            // Rise ramp: the enclosing gSnow > 0.02 gate admits nearly-bare
            // pixels along erosion-cut ribbon edges, where a full-amplitude
            // tilt would step 1-2 px of shading from full grain to nothing.
            let grainSnowRise = smoothHermite(0.02, 0.10, gSnow);
            perturbedWorld = normalize(perturbedWorld
                                     + vec3<f32>(-grainX, 0.0, -grainZ)
                                        * kGrainSlope * grainFade * grainFold
                                        * grainSnowRise);
        }

        // Crystal-scale fur. The 19 cm crust grain reads as gentle
        // undulation at the camera's feet: at 1-3 mm/px each grain facet
        // spans 60-200 px, so the closest metres carry no texture at all —
        // vision measured near-field high-frequency grain at 0.4 luma
        // against a photographed pack's 4-8. Real snow at boot range is
        // surface hoar and broken crystal clusters at 1-3 cm: features NEAR
        // the pixel scale. This octave tilts the normal and swings albedo
        // at ~2 cm, fading out as the footprint approaches feature size
        // (its fade window overlaps the grain's, so the two hand off
        // without a smooth gap between them).
        // Fade ceiling eases 0.014 -> 0.02: the coarser clusters below stay
        // resolvable one band farther out, so the boot-range albedo hands off
        // to the grain octave without a smooth gap.
        let crystalFade = 1.0 - smoothHermite(0.0015, 0.02, footprint);
        if (crystalFade > 0.001)
        {
            // 5.0 -> 3.2 (~2.8 cm clusters): the finer domain parked its
            // texel content at the boot-range pixel scale after the mip
            // chain, feeding the nearest-crop salt; coarser clusters also
            // sit closer to the 4-8 px boot-range grain photography shows.
            let kCrystalFreq = 3.2;     // ~2.8 cm crystal clusters
            let kCrystalSlope = 1.4;
            let crystalUV = rotateUV(worldXZ * kCrystalFreq, 0.97) + vec2<f32>(0.53, 0.09);
            // Screen-derivative sampling. LOD 0 served these near-field
            // octaves badly: at boot range one pixel covers 10-30 texels
            // of the octave's domain, and sampling the top mip there is
            // 10-30x undersampling — the "texture" round 7 measured was
            // texel-alias salt (2 px autocorrelation 0.15 on snow crystal,
            // 0.006 on the grass blade octave, negative axis lags =
            // checkerboard alternation), which FXAA then mashed into the
            // felt/dither the critics flagged on both biomes. Plain
            // auto-mip cannot replace it either: mirrorTile's wrap spikes
            // the GPU-computed gradient into a one-pixel blur line at
            // every fold. textureSampleGrad (GL textureGrad) with the
            // analytic domain derivative gives the sampler the true
            // footprint — the minified fetch mip-filters, so the noise
            // reads as coherent 4-8 px crystal clusters instead of salt —
            // and the gradient taps run at one-pixel separation through
            // the same filtered field, so the tilt is a real slope, not
            // per-pixel noise.
            let crystalUVdx = rotateUV(worldStepX * kCrystalFreq, 0.97);
            let crystalUVdy = rotateUV(worldStepY * kCrystalFreq, 0.97);
            // 3 px tap capped at a third of the feature (see the grain block):
            // kills the 2-3 px nearest-crop salt.
            let crystalTap = min(max(length(crystalUVdx), length(crystalUVdy)) * 3.0,
                                 0.03);
            let crystalCentre = textureSampleGrad(texture0, texture0_sampler, mirrorTile(crystalUV),
                                                  crystalUVdx, crystalUVdy).r;
            snowCrystalValue = clamp((crystalCentre - 0.5) * 2.8 + 0.5, 0.0, 1.0);
            let crystalX = textureSampleGrad(texture0, texture0_sampler,
                                             mirrorTile(crystalUV + vec2<f32>(crystalTap, 0.0)),
                                             crystalUVdx, crystalUVdy).r
                         - crystalCentre;
            let crystalZ = textureSampleGrad(texture0, texture0_sampler,
                                             mirrorTile(crystalUV + vec2<f32>(0.0, crystalTap)),
                                             crystalUVdx, crystalUVdy).r
                         - crystalCentre;
            // Fold mask (see the macro block): the crystal domain folds every
            // 0.4 m — a straddled tap doubles the slope along the fold line.
            let crystalFold = smoothHermite(0.0, 0.05,
                min(abs(cos(1.5707963 * crystalUV.x)), abs(cos(1.5707963 * crystalUV.y))));
            // Rise ramp, as the grain block above: erosion-cut edges fade
            // their tilt in with the first 0.10 of snow cover.
            let crystalSnowRise = smoothHermite(0.02, 0.10, gSnow);
            perturbedWorld = normalize(perturbedWorld
                                     + vec3<f32>(-crystalX, 0.0, -crystalZ)
                                        * kCrystalSlope * crystalFade * crystalFold
                                        * crystalSnowRise);
        }
    }
    let perturbedView = normalize((globals.view * vec4<f32>(perturbedWorld, 0.0)).xyz);

    // Filtering distributions of normals (Kaplanyan et al., HPG 2016):
    // sub-pixel normal variance widens the effective roughness by a GGX
    // proxy, so distant tiling detail stops aliasing into specular shimmer
    // — which the eye reads as shine at range. Runs before the sparkle write
    // below so surviving glints keep their tight mirror roughness.
    let texelFootprint = footprint * globals.settings_a.y;
    var roughVariance = 0.15 * texelFootprint * texelFootprint;
    // Snow is the exception again: its micro-facet field integrates into a
    // narrow forward sheen (that is what thousands of sub-pixel glints add
    // up to at range), so letting pixel variance widen the lobe like rough
    // rock's would fog the distant crystalline band away entirely.
    roughVariance = roughVariance * mix(1.0, 0.3, gSnow);
    rough = clamp(sqrt(rough * rough + roughVariance), 0.0, 1.0);

    // Snow glints: fresh snow is a dense field of ice facets, and a small
    // share of them catches the sun from any one viewpoint, reading as
    // sparkle. Hash cells are pinned to world space, so glints stay fixed to
    // the surface and pop in and out as the view moves. Alignment is
    // evaluated here every frame because the G-buffer is rewritten every
    // frame; a winning facet rotates the stored normal to itself and drops
    // the stored roughness to a tight mirror, letting the composite's GGX
    // lobe turn it into a sun glint. The hash-jittered fades thin the three
    // octaves out by distance instead of dissolving in one uniform ring.
    var outNormalView = perturbedView;
    var outRough = rough;
    if (globals.settings_b.y > 0.0 && gSnow > 0.02)
    {
        let up = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0),
                        abs(perturbedWorld.y) < 0.98);
        let sparkleTangent = normalize(cross(up, perturbedWorld));
        let sparkleBitangent = cross(perturbedWorld, sparkleTangent);
        let toCamera = normalize(globals.camera_position.xyz - frag_world_position);
        let toSun = -normalize(globals.sun_direction.xyz);
        let halfVector = normalize(toCamera + toSun);

        // Near octave: ~4 cm crystal facets.
        let cellNear = floor(worldXZ * SNOW_SPARKLE_CELLS);
        let h1 = hash12(cellNear);
        let h2 = hash12(cellNear + vec2<f32>(37.0, 81.0));
        let gateNear = step(1.0 - SNOW_SPARKLE_SHARE,
                            hash12(cellNear + vec2<f32>(11.0, 53.0)));
        let tiltNear = SNOW_SPARKLE_TILT_MAX * pow(h1, SNOW_SPARKLE_TILT_SHAPE);
        let facetNear = normalize(perturbedWorld * cos(tiltNear)
                       + (sparkleTangent * cos(h2 * 6.2831853)
                          + sparkleBitangent * sin(h2 * 6.2831853)) * sin(tiltNear));
        let fadeNear = 1.0
                     - smoothHermite(0.5 / SNOW_SPARKLE_CELLS, 2.2 / SNOW_SPARKLE_CELLS,
                                     footprint + hash12(cellNear + vec2<f32>(7.0, 3.0))
                                                * (0.5 / SNOW_SPARKLE_CELLS));
        // Real crystals are points, not cell-sized patches: without a
        // sub-cell mask a winning facet saturates its whole cell, which up
        // close spans dozens of pixels and reads as a white shard. Each
        // cell's glint lives at a hash-jittered point sized to ~1 px. The
        // near floor rose in round 8 (0.12 -> 0.17 cell) so close winners
        // paint ~3 px discs; round 8's measurement then showed the CLAMP's
        // half-cell ceiling was the real midfield killer — wherever
        // cells-per-pixel fell under ~2 the disc shrank to 0.2-0.7 px and
        // vanished under FXAA, leaving the 7-15 m band 10x sparser than the
        // near band. The ceiling now rises past a full cell so the disc
        // tracks ~1 px all the way to the fade (the cell partition still
        // bounds it), and the fade hands off to the far octave still holding
        // pixel-sized glints. The size jitter's low end is 0.8x rather than
        // 0.5x for the same reason: a 0.5 px disc fires, but its flash
        // averages with the surrounding diffuse to ~242-245 — present in
        // the field yet invisible to any glitter band, which is where the
        // retune's first measurement left the midfield.
        let localNear = worldXZ * SNOW_SPARKLE_CELLS - cellNear;
        let pointNear = vec2<f32>(hash12(cellNear + vec2<f32>(3.0, 71.0)),
                                  hash12(cellNear + vec2<f32>(43.0, 17.0)));
        let radiusNear = clamp(footprint * SNOW_SPARKLE_CELLS
                             * (0.8 + 0.7 * hash12(cellNear + vec2<f32>(29.0, 5.0))),
                               0.90, 2.5);
        let maskNear = 1.0 - smoothHermite(radiusNear * 0.5, radiusNear,
                                           length(localNear - pointNear));
        let mirrorNear = pow(max(dot(facetNear, halfVector), 0.0), SNOW_SPARKLE_POWER);
        let alignNear = mirrorNear * gateNear * fadeNear * maskNear;
        let glintNear = alignNear;

        // Far octave: ~14 cm sastrugi facets that persist to mid distance,
        // with amplitude compensation on the survivors.
        let cellFar = floor(worldXZ * SNOW_SPARKLE_CELLS_FAR);
        let h3 = hash12(cellFar + vec2<f32>(97.0, 13.0));
        let h4 = hash12(cellFar + vec2<f32>(59.0, 41.0));
        let gateFar = step(1.0 - SNOW_SPARKLE_SHARE_FAR,
                           hash12(cellFar + vec2<f32>(73.0, 29.0)));
        let tiltFar = SNOW_SPARKLE_TILT_MAX_FAR * pow(h3, SNOW_SPARKLE_TILT_SHAPE_FAR);
        let facetFar = normalize(perturbedWorld * cos(tiltFar)
                      + (sparkleTangent * cos(h4 * 6.2831853)
                         + sparkleBitangent * sin(h4 * 6.2831853)) * sin(tiltFar));
        let fadeFar = 1.0
                    - smoothHermite(0.5 / SNOW_SPARKLE_CELLS_FAR, 1.5 / SNOW_SPARKLE_CELLS_FAR,
                                    footprint + hash12(cellFar + vec2<f32>(7.0, 3.0))
                                               * (0.5 / SNOW_SPARKLE_CELLS_FAR));
        let localFar = worldXZ * SNOW_SPARKLE_CELLS_FAR - cellFar;
        let pointFar = vec2<f32>(hash12(cellFar + vec2<f32>(61.0, 7.0)),
                                 hash12(cellFar + vec2<f32>(13.0, 29.0)));
        let radiusFar = clamp(footprint * SNOW_SPARKLE_CELLS_FAR
                            * (0.8 + 0.7 * hash12(cellFar + vec2<f32>(19.0, 47.0))),
                              0.55, 1.5);
        let maskFar = 1.0 - smoothHermite(radiusFar * 0.5, radiusFar,
                                          length(localFar - pointFar));
        let mirrorFar = pow(max(dot(facetFar, halfVector), 0.0), SNOW_SPARKLE_POWER_FAR);
        let alignFar = mirrorFar * gateFar * fadeFar * maskFar;
        let glintFar = alignFar
                     * (1.0 + smoothHermite(0.5 / SNOW_SPARKLE_CELLS_FAR,
                                            1.5 / SNOW_SPARKLE_CELLS_FAR, footprint));

        // Coarse octave: wind-crust facets for mid and far snowfields. The
        // fade-in window matches the far octave's fade-out, so the handoff
        // never leaves a sparkle-free gap. The fade-out ceiling sits well
        // above the facet's own pixel size because sub-pixel facets are
        // thinned by the survival gate below rather than cut off here — a
        // hard fade at one-metre footprints is what left far snowfields
        // completely sparkle-free.
        let cellCoarse = floor(worldXZ * SNOW_SPARKLE_CELLS_COARSE);
        let h5 = hash12(cellCoarse + vec2<f32>(23.0, 101.0));
        let h6 = hash12(cellCoarse + vec2<f32>(149.0, 31.0));
        let gateCoarse = step(1.0 - SNOW_SPARKLE_SHARE_COARSE,
                              hash12(cellCoarse + vec2<f32>(5.0, 89.0)));
        let tiltCoarse = SNOW_SPARKLE_TILT_MAX_COARSE * pow(h5, SNOW_SPARKLE_TILT_SHAPE_COARSE);
        let facetCoarse = normalize(perturbedWorld * cos(tiltCoarse)
                        + (sparkleTangent * cos(h6 * 6.2831853)
                           + sparkleBitangent * sin(h6 * 6.2831853)) * sin(tiltCoarse));
        let fadeCoarse = smoothHermite(0.5 / SNOW_SPARKLE_CELLS_FAR,
                                       1.5 / SNOW_SPARKLE_CELLS_FAR, footprint)
                       * (1.0 - smoothHermite(2.0, 6.0,
                                              footprint + hash12(cellCoarse + vec2<f32>(7.0, 3.0)) * 1.5));
        let localCoarse = worldXZ * SNOW_SPARKLE_CELLS_COARSE - cellCoarse;
        let pointCoarse = vec2<f32>(hash12(cellCoarse + vec2<f32>(31.0, 3.0)),
                                    hash12(cellCoarse + vec2<f32>(7.0, 59.0)));
        let radiusCoarse = clamp(footprint * SNOW_SPARKLE_CELLS_COARSE
                               * (0.8 + 0.7 * hash12(cellCoarse + vec2<f32>(23.0, 41.0))),
                                 0.60, 1.5);
        let maskCoarse = 1.0 - smoothHermite(radiusCoarse * 0.5, radiusCoarse,
                                             length(localCoarse - pointCoarse));
        let mirrorCoarse = pow(max(dot(facetCoarse, halfVector), 0.0), SNOW_SPARKLE_POWER_COARSE);
        let alignCoarse = mirrorCoarse * gateCoarse * fadeCoarse * maskCoarse;
        let glintCoarse = alignCoarse;

        let farWins = glintFar > glintNear;
        var facet = select(facetNear, facetFar, farWins);
        var bestGlint = select(glintNear, glintFar, farWins);
        var bestAlign = select(alignNear, alignFar, farWins);
        var bestCells = select(SNOW_SPARKLE_CELLS, SNOW_SPARKLE_CELLS_FAR, farWins);
        if (glintCoarse > bestGlint)
        {
            facet = facetCoarse;
            bestGlint = glintCoarse;
            bestAlign = alignCoarse;
            bestCells = SNOW_SPARKLE_CELLS_COARSE;
        }
        // Rotation authority is alignment quality, not sparkle amplitude: a
        // winning facet turns the stored normal ALL the way to itself so the
        // composite's GGX lobe can centre on the mirror flash.
        // (Proportional rotation parks the lobe between the shaded normal
        // and the facet — close enough to tilt N.L, too far to fire any
        // specular — which read as faint diffuse chips, not glints.) Weak
        // alignments stay below the window and never touch the normal, so
        // a facet reflecting sky cannot render as a dark chip. The window
        // matches the lobe the rotation earns: the stored roughness drops
        // to a mirror whose GGX half-width is a few degrees, so every
        // fully-rotated winner lands inside its own flash.
        // Facets smaller than a pixel cannot fill it with a mirror flash,
        // so they survive only with probability proportional to their
        // pixel coverage: the twinkle field thins out with distance the
        // way photographed sparkle does, instead of every sub-pixel facet
        // detonating a full white pixel.
        // (0.28, 0.38), not the old (0.20, 0.40): the graded annulus of the
        // wide window was the dotted-chain generator — see the constants
        // block above. Winners inside ~17 degrees rotate ALL the way (the
        // lobe catches them: flash), the 17-20 degree near-misses never touch
        // the normal, and the graded zone between is one degree thin.
        var applied = smoothHermite(0.28, 0.38, bestAlign * gSnow * globals.settings_b.y);
        let cellsPerPixel = max(footprint * bestCells, 0.5);
        // The thinning keeps a floor: a small constant fraction of facets
        // survives at any range, so distant snowfields hold a sparse
        // shimmer instead of the sparkle decaying to zero past a few
        // hundred metres — photographed glint fields thin with distance,
        // they do not vanish while the field is still resolved. The 3.2
        // numerator (raised from 1.6 with round-6's verdicts: the
        // into-sun glitter path measured 0.03% bright pixels against a
        // photographed glitter path's ~0.5-2%) thins at half the old rate,
        // so the 10-60 m band where facets are 1-10 px keeps a visible
        // glint field instead of decaying to the floor before midfield.
        applied = applied * step(hash12(floor(worldXZ * bestCells) + vec2<f32>(51.0, 83.0)),
                                 clamp(3.2 / cellsPerPixel, 0.08, 1.0));
        let facetView = normalize((globals.view * vec4<f32>(facet, 0.0)).xyz);
        outNormalView = normalize(mix(perturbedView, facetView, applied));
        // 0.13, not the old 0.10: the applied window accepts facets aligned
        // to ~13-16 degrees, but a 0.10 mirror lobe is only ~6 degrees wide —
        // accepted winners in the 6-13 degree band rotated fully into a lobe
        // that could not catch them (the 246-250 plateau round 8 measured
        // instead of blown glints). 0.13 widens the lobe to ~9 degrees so the
        // acceptance band and the flash agree; composite.fs's fog-relief
        // window moved with it.
        outRough = mix(outRough, 0.13, applied);
    }

    // Snow grain follows the same footprint fades as its normal detail.
    // Grass and soil already carry their blended scan detail.
    {
        // Snow: ~25 cm mottling that survives to mid distance, plus the
        // albedo face of the grain octave computed with the normals above.
        let mottleFade = 1.0 - smoothHermite(0.08, 0.30, footprint);
        // textureSampleGrad (GL textureGrad; see the crystal block): the
        // plain fetch's GPU-computed derivatives spike at every mirrorTile
        // fold once auto-mip engages past ~8 cm/px, and those spikes draw
        // 1 px blur/dark lines at the fold's 5.6 m spacing through the
        // whole mid band — the same hairline family the macro block's taps
        // produced.
        let mottle = textureSampleGrad(texture0, texture0_sampler,
                                       mirrorTile(rotateUV(worldXZ * 0.36, 0.29)
                                                  + vec2<f32>(0.23, 0.67)),
                                       rotateUV(worldStepX * 0.36, 0.29),
                                       rotateUV(worldStepY * 0.36, 0.29)).r;
        // Half the mottling is tied to the macro relief: faces tilted away
        // from the sun take the shade end of the drift and sun-side crests
        // the bright end, so the metre-scale blobs read as sastrugi form
        // shading instead of painted albedo smudges (which vision measured
        // as isotropic airbrushed blobs with symmetric flanks).
        let driftSun = -dot(normalize(macroGradient + vec2<f32>(1e-5, 0.0)), toSunXZ);
        let driftForm = smoothHermite(0.02, 0.10, length(macroGradient));
        let mottleForm = clamp(0.5 + 0.5 * driftSun * driftForm, 0.0, 1.0);
        albedo = albedo * (1.0 + (mix(mottle, mottleForm, 0.6) - 0.5) * 0.30 * gSnow * mottleFade);
        let grainFade = 1.0 - smoothHermite(0.003, 0.08, footprint);
        // Inner fade: the grain domain's texel is ~2 mm, so below ~6 mm/px
        // the fetch sits near LOD 0 and the octave's own texel noise rides
        // straight through as 1-2 px speckle — the "nearest snow grain
        // salt" the artifact hunter measured on the steepest-crop checks.
        // The crystal octave (texel 0.3 mm, deep in its mip chain at that
        // range) owns the boot-range albedo instead.
        let grainInner = smoothHermite(0.0015, 0.006, footprint);
        albedo = albedo * (1.0 + (snowGrainValue - 0.5) * 0.32 * gSnow * grainFade * grainInner);
        // Crystal albedo: boot-range snow varies at the cluster scale —
        // shadowed interstices between hoar clusters read darker than the
        // cluster faces, so the crystal octave swings albedo as well as
        // tilting the normal (its fade is recomputed rather than reused
        // because it lives in this block's scope).
        let crystalFade = 1.0 - smoothHermite(0.0015, 0.02, footprint);
        albedo = albedo * (1.0 + (snowCrystalValue - 0.5) * 0.26 * gSnow * crystalFade);

        // The snow-to-grass biome crossfade interpolates the two albedos
        // straight through an olive half-mix; a real snowline melts through
        // dirty grey-brown snow. Desaturate the overlap band toward its own
        // luminance with a faintly earthen tint.
        let fringe = clamp(min(gSnow, groundCover) * 1.8, 0.0, 1.0);
        let fringeLum = dot(albedo, vec3<f32>(0.299, 0.587, 0.114));
        albedo = mix(albedo, vec3<f32>(fringeLum) * vec3<f32>(1.04, 0.99, 0.92), fringe * 0.5);

        // Silt stains the thinning pack around its local climate margin.
        // Use coverage rather than another altitude band so the stain tracks
        // drift, hollows and exposure. Deep pack and bare ground stay clean.
        let sedimentClearance = smoothHermite(0.35, 0.95, snowLine);
        let meltBand = snowLine * (1.0 - snowLine) * 4.0;
        var sedimentLoad = flowDomain
                         * (0.85 * smoothHermite(0.55, 0.95, snowLine)
                            + 0.50 * meltBand
                            + 0.60 * incision * (1.0 - deposition)
                            + 0.40 * transport)
                         * smoothHermite(0.15, 0.60, gSnow)
                         * (1.0 - sedimentClearance);
        sedimentLoad = clamp(sedimentLoad, 0.0, 1.0);
        albedo = mix(albedo, albedo * vec3<f32>(0.52, 0.37, 0.29), 0.85 * sedimentLoad);
    }

    let waterAmount = smoothHermite(0.04, 0.35, max(flow.r, 0.0));

    if (globals.settings_b.z != 0.0)   // uFlowDebug
    {
        let velocity = flow.gb;
        let speed = length(velocity);
        let direction = select(vec2<f32>(0.0), velocity / speed, speed > 0.00001);

        // Red/green encode signed X/Z direction and blue encodes accumulated
        // discharge. Brightness makes both standing water and velocity visible.
        let directionColour = vec3<f32>(direction * 0.5 + vec2<f32>(0.5), dischargeAmount);
        let speedAmount = 1.0 - exp(-speed * 0.18);
        let signal = clamp(max(waterAmount, max(dischargeAmount, speedAmount)), 0.0, 1.0);
        let debugColour = mix(vec3<f32>(0.015, 0.018, 0.025), directionColour, signal);
        albedo = mix(albedo, debugColour, flowDomain);
    }
    else
    {
        let moisture = clamp(waterAmount * 0.52 + channel * 0.25, 0.0, 0.72)
                     * (1.0 - gSnow);
        let damp = albedo * vec3<f32>(0.56, 0.69, 0.63);
        albedo = mix(albedo, damp, moisture);
        // Damp earth stays rough; only pooled water approaches a low
        // roughness. Write the final value used by the G-buffer (snow glints
        // have already modified outRough above).
        let wetness = moisture / 0.72;
        let wetRoughness = mix(0.70, 0.42, waterAmount);
        outRough = mix(outRough, wetRoughness, wetness);
    }

    // Diagnostic: replace the biome albedo with a false-colour of the blended
    // erosion contribution (blue = incision, orange = deposition, dark = base).
    // A hard colour step at a tile edge is a true discontinuity; a smooth grade
    // that breaks a channel is independent-simulation divergence.
    if (globals.settings_b.w != 0.0)   // uErosionDebug
    {
        let d = clamp(input.frag_erosion_delta / 6.0, -1.0, 1.0);
        let base = vec3<f32>(0.02, 0.05, 0.02);
        let col = select(mix(base, vec3<f32>(1.00, 0.40, 0.10), d),   // d >= 0
                         mix(base, vec3<f32>(0.10, 0.45, 1.00), -d),  // d < 0
                         d < 0.0);
        albedo = col;
    }

    // Only surviving grass contributes to canopy lighting in the composite;
    // the soil portion keeps its own rough mineral response.
    // gPosition alpha packs two biome masks on top of the sky flag: snow
    // quantized to hundredths, grass at a ten-thousandth scale, which the
    // 32F channel resolves exactly. 0 is sky, 1 is geometry carrying
    // neither biome. Snow must be QUANTIZED at the write: the composite
    // decodes snow as a clamped remainder of a - 1, and any continuous
    // grass term below snow's own precision is absorbed by that clamp,
    // making the grass decode (a - 1 - snowMask) identically zero. That is
    // exactly what the original hundredth-scale packing did: grassMask was
    // stuck at 0, the specular damper never engaged (full-strength Fresnel
    // GGX on every meadow — the pale sunward wash every vision round
    // measured) and the anti-solar warm-up stayed dead. With snow on the
    // hundredth lattice and grass in the 1e-4 slot, rounding (a - 1) to
    // hundredths recovers snow exactly and the residue is grass.
    output.g_position = vec4<f32>(frag_position_view,
                                  1.0 + floor(gSnow * 100.0 + 0.5) * 0.01 + gGrass * 0.0001);
    output.g_normal = vec4<f32>(outNormalView * 0.5 + 0.5, clamp(outRough, 0.0, 1.0));
    // The albedo target is 8-bit, and snow's tinted albedo rides above 1.0
    // (sunlit noon snow is rendered super-white on purpose) — a linear write
    // would clamp it to 1.0 there, silently deleting the mottle and grain
    // albedo texture and capping the pack's brightness no matter the tint.
    // sqrt-encode against a 2.5 headroom constant instead: highlights keep
    // 8-bit headroom while dark albedos (grass, ocean) gain precision, and
    // composite.fs squares the read back. The alpha (texture AO) is already
    // 0-1 and stays linear.
    output.g_albedo = vec4<f32>(sqrt(max(albedo, vec3<f32>(0.0)) * 0.4), clamp(ao, 0.0, 1.0));
    return output;
}

// Validator-only dummy: wgsl-check requires a vs_main entry in every file it
// validates; this fragment-only file's real vertex stage is terrain-vs.wgsl.
// Never bound by the pipeline (see PORT NOTES 6).
@vertex
fn vs_main() -> @builtin(position) vec4<f32> { return vec4<f32>(); }

// STAGE UNIFORMS (group 2, binding 0) — StageUniforms, 272 bytes, shared with
// terrain.vs (the GLSL original linked both stages into one program with one
// uniform block). Fields the fragment stage reads, at their byte offsets:
//   sea_level                       f32        @152   uSeaLevel
//   erosion_tile_stride             f32        @204   uErosionTileStride
//   erosion_footprint_size          f32        @208   uErosionFootprintSize
//   erosion_output_resolution       f32        @212   uErosionOutputResolution
//   erosion_atlas_pitch             f32        @216   uErosionAtlasPitch
//   erosion_atlas_size              f32        @220   uErosionAtlasSize
//   erosion_atlas_gutter            f32        @224   uErosionAtlasGutter
//   erosion_lookup_min_tile         vec2<f32>  @232   uErosionLookupMinTile
//   erosion_lookup_size             f32        @240   uErosionLookupSize
//   erosion_visibility_center       vec2<f32>  @248   uErosionVisibilityCenter
//   erosion_visibility_full_radius  f32        @256   uErosionVisibilityFullRadius
//   erosion_visibility_zero_radius  f32        @260   uErosionVisibilityZeroRadius
// GlobalUniforms (group 0, binding 0) fields this stage reads:
//   view (matView), camera_position.xyz (uCameraPosition),
//   sun_direction.xyz (uSunDirectionWorld), settings_a.y (uTexScale),
//   settings_a.w (uVariantScale), settings_b.x (uNormalStrength),
//   settings_b.y (uSparkleStrength), settings_b.z (uFlowDebug, f32 1/0),
//   settings_b.w (uErosionDebug, f32 1/0).
// Group 1 bindings (texture N / sampler N+8, GL unit numbers preserved):
//   texture0 f32-2D      binding 0  (R32 base noise; sampler binding 8)
//   texture1 f32-2D      binding 1  (surface atlas RGBA32F; sampler 9)
//   texture2 f32-2D      binding 2  (flow atlas RGBA32F; sampler 10)
//   texture3 f32-2D      binding 3  (RGBA32F tile lookup, POINT-filtered; sampler 11)
//   texture4 f32-2D      binding 4  (blend mask; sampler 12)
//   albedo_ao  2D array  binding 5  (uAlbedoAO: rgb albedo sRGB, a AO; sampler 13)
//   normal_rough 2D arr  binding 6  (uNormalRough: rgb tangent normal, a roughness; sampler 14)
// Fragment outputs: @location(0) gPosition, @location(1) gNormal,
// @location(2) gAlbedo. Fragment inputs @location(0..4): fragPositionView,
// fragNormalView, fragWorldPosition, fragWorldNormal, fragErosionDelta
// (terrain.vs out order).
