// G-buffer terrain shading — WGSL port of forest/assets/shaders/terrain.fs
// (GLSL 330, raylib/GL 4.3 deferred renderer), extended with terrain-driven
// PBR coverage, cavity-aware material contacts and filtered weathering relief.
// The port notes below describe the original resource and sampling contracts.

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
//    `out` declaration order, plus location 5 for fixed-scale material slope.
//    The wgsl-check validator requires a vs_main
//    entry in every file it validates, so a never-bound dummy vs_main sits at
//    the bottom of this fragment-only file.
// 7. WGSL has no int uniforms here: everything else routes into the
//    spec-fixed GlobalUniforms (settings_a.y = uTexScale, settings_a.w =
//    uVariantScale, settings_b.x = uNormalStrength, settings_b.y =
//    uSparkleStrength, sun_direction = uSunDirectionWorld, camera_position =
//    uCameraPosition, view = matView) or into StageUniforms (see its
//    provenance comment and the STAGE UNIFORMS block at the file end).
// 8. Material placement now uses world-anchored terrain exposure for an
//    overlapping ground cover and rock exposure. Snow combines regional climate
//    variation with terrain retention instead of a fixed contour on rock.

// Shared global uniforms — the spec preamble, verbatim.
struct GlobalUniforms {
    view: mat4x4<f32>,            // matView: column-major world->view
    projection: mat4x4<f32>,      // matProjection (standard GL shape, z converted to [0,1] NDC)
    camera_position: vec4<f32>,   // xyz world-space eye; w habitat-only ocean crest clearance
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
    snowline_altitude: f32,                // centre of the persistent snowline above sea level
};
@group(2) @binding(0) var<uniform> stage: StageUniforms;

// GL texture unit numbers preserved (see PORTING-SPEC section 3): texture at
// binding N, sampler at binding N + 8.
// Flow atlas packing:
//   R = water depth, G = signed velocity X, B = signed velocity Z,
//   A = routed contributing area in 4x4 m cells.
// Surface atlas: signed bed delta (m), local concavity (m), drainage
// concentration (positive log2 ratio), and the loose cover left above
// bedrock (m).
@group(1) @binding(1) var texture1: texture_2d<f32>;
@group(1) @binding(2) var texture2: texture_2d<f32>;
@group(1) @binding(3) var texture3: texture_2d<f32>;        // Point-filtered RGBA32F tile lookup.
@group(1) @binding(4) var texture4: texture_2d<f32>;        // Raylib square-gradient tile blend mask.
// PBR terrain texture atlases (see LoadTerrainTextures in main.cpp).
// texture0 is the shared base-noise field for biome boundaries and surface detail.
@group(1) @binding(0) var texture0: texture_2d<f32>;        // R32 base noise (also used by the vertex stage).
@group(1) @binding(5) var albedo_ao: texture_2d_array<f32>;   // uAlbedoAO: 8 layers: rgb albedo (sRGB), a ambient occlusion.
@group(1) @binding(6) var normal_rough: texture_2d_array<f32>; // uNormalRough: 8 layers: rgb tangent normal (OpenGL green-up), a roughness.

@group(1) @binding(8) var texture0_sampler: sampler;
@group(1) @binding(9) var texture1_sampler: sampler;
@group(1) @binding(10) var texture2_sampler: sampler;
@group(1) @binding(11) var texture3_sampler: sampler;
@group(1) @binding(12) var texture4_sampler: sampler;
@group(1) @binding(13) var albedo_ao_sampler: sampler;
@group(1) @binding(14) var normal_rough_sampler: sampler;
@group(1) @binding(15) var<storage, read> river_grid: array<u32>;      // river lookup grid
@group(1) @binding(16) var<storage, read> river_segments: array<RiverSegment>; // river carve segments

// Fragment-stage inputs: terrain.vs's five `out` varyings, declaration order
// = @location order (see PORT NOTES 6). Naming: fragPositionView etc.
struct FsInput {
    @location(0) frag_position_view: vec3<f32>,   // fragPositionView
    @location(1) frag_normal_view: vec3<f32>,     // fragNormalView
    @location(2) frag_world_position: vec3<f32>,  // fragWorldPosition
    @location(3) frag_world_normal: vec3<f32>,    // fragWorldNormal
    @location(4) frag_erosion_delta: f32,         // fragErosionDelta
    @location(5) frag_material_normal: vec3<f32>, // fixed-scale world slope
    // The nearest river or lake: metres past its waterline (negative under
    // water), side of the bend (+ outside, - inside), whitewater, flow speed.
    @location(6) frag_river: vec4<f32>,
};

// G-buffer outputs, locations preserved from the GLSL layout qualifiers.
struct FsOutput {
    @location(0) g_position: vec4<f32>,   // gPosition
    @location(1) g_normal: vec4<f32>,     // gNormal
    @location(2) g_albedo: vec4<f32>,     // gAlbedo
};

struct GrassHabitatOutput {
    @location(0) habitat: vec4<f32>,
    @location(1) ground_albedo: vec4<f32>,
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

// BEGIN SHARED GEOLOGY
// World-keyed substrate resistance. The erosion passes and the terrain
// material shader paste this block verbatim (a test keeps the copies equal),
// so the rock that resists incision and holds steep faces in the simulation
// is the same rock the material shader exposes. Every input is a world
// coordinate, which keeps overlapping erosion tiles deterministic.
//
// Broad (~92 m) and fine (~27 m) rock bodies vary hardness without following
// contours. A weaker bedding term adds gently dipping, warped resistant beds
// every ~19 m of bedrock elevation: where incision or talus relaxation cuts
// through them they hold short cliff bands and benches instead of one
// uniform slope. The bedding is keyed to the bedrock surface, so a bench
// stays put as loose cover accumulates above it.
const GEOLOGY_HARDNESS_SCALE: f32 = 0.56;

fn geologyHash(position: vec2<f32>) -> f32
{
    var p = fract(vec3<f32>(position.xyx) * 0.1031);
    p += dot(p, p.yzx + 33.33);
    return fract((p.x + p.y) * p.z);
}

fn geologyNoise(position: vec2<f32>) -> f32
{
    let cell = floor(position);
    var fraction = fract(position);
    fraction = fraction * fraction * (3.0 - 2.0 * fraction);
    let a = geologyHash(cell);
    let b = geologyHash(cell + vec2<f32>(1.0, 0.0));
    let c = geologyHash(cell + vec2<f32>(0.0, 1.0));
    let d = geologyHash(cell + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, fraction.x), mix(c, d, fraction.x), fraction.y);
}

// 0 = weak, readily weathered substrate; 1 = the most resistant rock.
fn geologyResistance(world_xz: vec2<f32>, bedrock_height: f32) -> f32
{
    let broadRock = geologyNoise(world_xz / 92.0 + vec2<f32>(31.7, -18.2));
    let fineRock = geologyNoise(world_xz / 27.0 + vec2<f32>(-73.1, 46.4));
    let bedWarp = geologyNoise(world_xz / 310.0 + vec2<f32>(-12.9, 57.3)) - 0.5;
    let bedCoordinate = (bedrock_height + dot(world_xz, vec2<f32>(0.017, -0.011))
                         + bedWarp * 26.0) / 19.0;
    let bedding = smoothstep(0.35, 0.85, 0.5 + 0.5 * sin(bedCoordinate * 6.2831853));
    let geology = broadRock * 0.62 + fineRock * 0.24 + bedding * 0.14;
    return smoothstep(0.25, 0.82, geology);
}

// The small spawn footprint resists destructive excavation.
fn erosionSpawnProtection(world_xz: vec2<f32>) -> f32
{
    return 1.0 - smoothstep(90.0, 150.0, length(world_xz));
}

// Hardness as the erosion solver uses it: scaled resistance, raised to the
// spawn protection inside its footprint.
fn erosionHardness(world_xz: vec2<f32>, bedrock_height: f32) -> f32
{
    return max(geologyResistance(world_xz, bedrock_height) * GEOLOGY_HARDNESS_SCALE,
               erosionSpawnProtection(world_xz) * 0.96);
}
// END SHARED GEOLOGY

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
    // Rock shares the gravel-scale tile: the Rock032 scan's chip scars and
    // crack stains span ~0.1-1.3 m inside one 6 m repeat, so a face reads as
    // weathered cliff detail at close and mid range without duplicating any
    // single feature across the whole mountainside.
    return 1.0;
}

fn tileCoordContinuous(material_index: i32, worldXZ: vec2<f32>, octave: i32) -> vec2<f32>
{
    let scale_mul = select(1.17, 1.0, octave == 0);
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
    let scale_mul = select(1.17, 1.0, octave == 0);
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
// Value and analytic spatial derivatives. These derivatives describe the
// world field, so relief remains stable when screen resolution/LOD changes.
fn groundNoiseGradient(p: vec2<f32>) -> vec3<f32>
{
    let cell = floor(p);
    let f = fract(p);
    let w = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let dw = 30.0 * f * f * (f - 1.0) * (f - 1.0);
    let inv_hash_range = 1.0 / 16777215.0;
    let a = f32(surfaceCellHash(cell, 17, 0) & 0xffffffu) * inv_hash_range;
    let b = f32(surfaceCellHash(cell + vec2<f32>(1.0, 0.0), 17, 0) & 0xffffffu) * inv_hash_range;
    let c = f32(surfaceCellHash(cell + vec2<f32>(0.0, 1.0), 17, 0) & 0xffffffu) * inv_hash_range;
    let d = f32(surfaceCellHash(cell + vec2<f32>(1.0), 17, 0) & 0xffffffu) * inv_hash_range;
    return vec3<f32>(mix(mix(a, b, w.x), mix(c, d, w.x), w.y),
                     mix(b - a, d - c, w.y) * dw.x,
                     mix(c - a, d - b, w.x) * dw.y);
}

fn groundNoise(p: vec2<f32>) -> f32
{
    return groundNoiseGradient(p).x;
}

// Two independent values in [0, 1) per lattice cell.
fn jointHash2(cell: vec2<f32>, seed: i32) -> vec2<f32>
{
    let h = surfaceCellHash(cell, seed, 1);
    return vec2<f32>(f32(h & 0xffffu), f32(h >> 16u)) * (1.0 / 65536.0);
}

// Joint blocks: a jittered Voronoi tessellation of a projection plane, in
// cell units. Returns the distance to the nearest block border (x, as the gap
// to the second-nearest site), the block's facet tilt in [-1, 1] (yz), and a
// per-block brightness hash (w). Exposed rock breaks along joint sets into
// blocks whose faces catch the light at slightly different angles; the scan
// alone resolves its grain but not that metre-scale fracture pattern.
fn rockJoints(p: vec2<f32>, seed: i32) -> vec4<f32>
{
    let base = floor(p);
    var nearest = 8.0;
    var second = 8.0;
    var nearestCell = base;
    for (var y = -1; y <= 1; y++)
    {
        for (var x = -1; x <= 1; x++)
        {
            let cell = base + vec2<f32>(f32(x), f32(y));
            let site = cell + vec2<f32>(0.15) + 0.70 * jointHash2(cell, seed);
            let distance = length(p - site);
            if (distance < nearest)
            {
                second = nearest;
                nearest = distance;
                nearestCell = cell;
            }
            else if (distance < second)
            {
                second = distance;
            }
        }
    }
    let facet = jointHash2(nearestCell + vec2<f32>(17.0, -31.0), seed) * 2.0 - vec2<f32>(1.0);
    let shade = jointHash2(nearestCell + vec2<f32>(-53.0, 11.0), seed).x;
    return vec4<f32>(second - nearest, facet, shade);
}

// A block's facet tilt eases to nothing at its joints, so neighbouring
// blocks meet on a shared normal and round into the joint instead of
// stepping there; a one-pixel normal step along every joint sparkles as the
// view moves.
fn facetTaper(gap: f32) -> f32
{
    return smoothHermite(0.0, 0.2, gap);
}

// Metre-scale weathered relief bridges the gap between scanned grains and
// the terrain mesh. Both octaves fade before becoming subpixel; explicit
// footprint inputs keep this callable inside material branches.
fn rockWeathering(p: vec2<f32>, dx: vec2<f32>, dy: vec2<f32>) -> vec3<f32>
{
    let pixel = max(length(dx), length(dy));
    let broad = groundNoiseGradient(rotateUV(p * 0.11, 0.37) + vec2<f32>(13.7, -41.2));
    let fine = groundNoiseGradient(rotateUV(p * 0.37, 1.19) + vec2<f32>(-31.6, 7.3));
    let broadFade = 1.0 - smoothHermite(0.35, 1.1, pixel * 0.11);
    let fineFade = 1.0 - smoothHermite(0.35, 1.1, pixel * 0.37);
    let gradient = rotateUV(broad.yz, -0.37) * (0.32 * 0.11 * broadFade)
                 + rotateUV(fine.yz, -1.19) * (0.07 * 0.37 * fineFade);
    return vec3<f32>((broad.x - 0.5) * broadFade * 0.7
                    + (fine.x - 0.5) * fineFade * 0.3, gradient);
}

// Only call before material branches. Derivatives include the domain warp;
// unresolved octaves settle to their mean instead of sparkling at distance.
// BEGIN SHARED RIVERS
// The river network's carve segments and the lookup grid over them, exactly
// as src/rivers/carve.rs evaluates them on the CPU, so the drawn ground, the
// player's footing, the seated plants and the erosion simulation all see one
// channel. Every pass that pastes this block (a test keeps the copies
// identical) declares `river_grid: array<u32>` and
// `river_segments: array<RiverSegment>` as read-only storage.
//
// river_grid holds an eight-word header (origin XZ and cell size as f32
// bits, resolution, segment count), then three words per cell, then the
// index lists the cells point into: a cell's segments, then the rocks whose
// wakes reach it. A cell's words are the offset of its lists, the segment
// count in the low and the rock count in the high 16 bits, and the level of
// any lake reaching into the cell as f32 bits (RIVER_NO_LAKE if none). A
// zero resolution means there are no rivers.
//
// A lake has no channel: ground below its level near it lies under its
// water, and riverBankAt measures the shore in a river bank's terms.
//
// Each segment bounds the ground from above (the channel bed, then a bank
// cone that steepens away from the water) and from below (a low levee that
// keeps the water in its channel). The upper bounds combine by minimum and
// the lower ones by maximum, so confluences open into each other; a segment
// has a flat start and a round end, which keeps a waterfall's lip vertical.
struct RiverSegment {
    a: vec2<f32>,
    b: vec2<f32>,
    water: vec2<f32>,
    half_width: vec2<f32>,
    depth: vec2<f32>,
    speed: vec2<f32>,
    bank: f32,
    skew: f32,
    turbulence: f32,
    levee: f32,
};

struct RiverEnvelope {
    upper: f32,
    lower: f32,
    // Metres past the nearest waterline; negative inside a channel.
    bank_distance: f32,
    water: f32,
    velocity: vec2<f32>,
    half_width: f32,
    turbulence: f32,
    // Which side of a bend the point lies on: toward +0.6 on the outside,
    // where the current cuts a steep bank, toward -0.6 on the inside, where
    // it drops its point bar.
    bend: f32,
    // The level of a lake reaching here, or RIVER_NO_LAKE.
    lake: f32,
};

const RIVER_NONE: f32 = 1.0e30;
const RIVER_NO_LAKE: f32 = -1.0e30;
const RIVER_GRID_CELL_WORDS: u32 = 3u;
// Metres of shore per metre of rise above a lake's water.
const RIVER_LAKE_SHORE_RUN: f32 = 6.0;
const RIVER_BANK_REACH: f32 = 12.0;
const RIVER_BANK_CURVE: f32 = 0.16;
const RIVER_LEVEE_OUTER_SLOPE: f32 = 0.6;
const RIVER_MAX_CANDIDATES: u32 = 64u;

fn riverNone() -> RiverEnvelope
{
    return RiverEnvelope(RIVER_NONE, -RIVER_NONE, RIVER_NONE, -RIVER_NONE,
                         vec2<f32>(0.0), 0.0, 0.0, 0.0, RIVER_NO_LAKE);
}

fn riverSegmentEnvelope(segment: RiverSegment, p: vec2<f32>) -> RiverEnvelope
{
    let ab = segment.b - segment.a;
    let ap = p - segment.a;
    let lengthSquared = max(dot(ab, ab), 1e-6);
    let along = dot(ap, ab)/lengthSquared;
    // Flat start: what lies behind the start belongs to the segment before.
    if (along < 0.0) { return riverNone(); }
    let t = min(along, 1.0);
    let offset = ap - ab*t;
    let centreDistance = length(offset);
    let halfWidth = max(mix(segment.half_width.x, segment.half_width.y, t), 0.05);
    let pastBank = centreDistance - halfWidth;
    if (pastBank > RIVER_BANK_REACH) { return riverNone(); }
    let water = mix(segment.water.x, segment.water.y, t);
    let depth = mix(segment.depth.x, segment.depth.y, t);
    let segmentLength = sqrt(lengthSquared);
    // Signed distance to the centreline, + on the left of the flow. The skew
    // fades out over the round end cap, where "left" stops meaning anything.
    let crossValue = ab.x*ap.y - ab.y*ap.x;
    let side = select(-1.0, 1.0, crossValue >= 0.0);
    let beyond = max(along - 1.0, 0.0)*segmentLength;
    let skew = segment.skew*(1.0 - smoothstep(0.0, halfWidth, beyond));
    let speed = mix(segment.speed.x, segment.speed.y, t);
    let direction = ab/segmentLength;
    var envelope = RiverEnvelope(RIVER_NONE, -RIVER_NONE, pastBank, water,
                                 direction*speed, halfWidth, segment.turbulence, skew*side,
                                 RIVER_NO_LAKE);
    if (pastBank < 0.0)
    {
        // Skewed parabola: zero at both banks, deepest toward the outer one.
        let u = side*centreDistance/halfWidth;
        let profile = (1.0 - u*u)*(1.0 + skew*u);
        envelope.upper = water - depth*profile;
        // The current runs fastest over the thalweg and stalls at the banks.
        let lateral = pow(max(1.0 - u*u, 0.0), 0.35);
        envelope.velocity = envelope.velocity*(lateral*1.25);
    }
    else
    {
        // Cut banks stand steep on the outside of a bend; point bars slope
        // gently into the water on the inside.
        let bank = segment.bank*max(1.0 + 0.75*skew*side, 0.3);
        envelope.upper = water + bank*pastBank + RIVER_BANK_CURVE*pastBank*pastBank;
        let freeboard = 0.1 + 0.25*depth;
        let leveeWidth = 0.8 + 0.3*halfWidth;
        envelope.lower = water + min(bank*pastBank, freeboard)
                       - max(pastBank - leveeWidth, 0.0)*RIVER_LEVEE_OUTER_SLOPE
                       - (1.0 - segment.levee)*1.0e4;
        envelope.velocity = vec2<f32>(0.0);
    }
    return envelope;
}

fn riverScore(envelope: RiverEnvelope) -> f32
{
    return envelope.bank_distance/min(max(envelope.half_width, 0.5), 4.0);
}

fn riverCombine(total: ptr<function, RiverEnvelope>, next: RiverEnvelope)
{
    if (next.bank_distance >= RIVER_NONE) { return; }
    (*total).upper = min((*total).upper, next.upper);
    (*total).lower = max((*total).lower, next.lower);
    // The water and its motion come from the channel whose waterline is
    // nearest, in its own widths, so a creek hands over to the river it
    // joins inside the larger channel.
    if ((*total).bank_distance >= RIVER_NONE || riverScore(next) < riverScore(*total))
    {
        (*total).bank_distance = next.bank_distance;
        (*total).water = next.water;
        (*total).velocity = next.velocity;
        (*total).half_width = next.half_width;
        (*total).turbulence = next.turbulence;
        (*total).bend = next.bend;
    }
}

fn riverEnvelope(p: vec2<f32>) -> RiverEnvelope
{
    var total = riverNone();
    let resolution = river_grid[3];
    if (resolution == 0u) { return total; }
    let origin = vec2<f32>(bitcast<f32>(river_grid[0]), bitcast<f32>(river_grid[1]));
    let cell = floor((p - origin)/bitcast<f32>(river_grid[2]));
    if (any(cell < vec2<f32>(0.0)) || any(cell >= vec2<f32>(f32(resolution)))) { return total; }
    let entry = 8u + (u32(cell.y)*resolution + u32(cell.x))*RIVER_GRID_CELL_WORDS;
    let offset = river_grid[entry];
    let count = min(river_grid[entry + 1u] & 0xffffu, RIVER_MAX_CANDIDATES);
    for (var i = 0u; i < count; i += 1u)
    {
        riverCombine(&total, riverSegmentEnvelope(river_segments[river_grid[offset + i]], p));
    }
    total.lake = bitcast<f32>(river_grid[entry + 2u]);
    return total;
}

// Metres past the nearest waterline, a river's or a lake's, for ground at
// `height`: negative under water.
fn riverBankAt(envelope: RiverEnvelope, height: f32) -> f32
{
    return min(envelope.bank_distance, (height - envelope.lake)*RIVER_LAKE_SHORE_RUN);
}

// The ground with the channels cut into it and their banks held up.
fn riverClamp(envelope: RiverEnvelope, height: f32) -> f32
{
    return min(max(height, envelope.lower), envelope.upper);
}
// END SHARED RIVERS

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
            (*albedo_ao_out) += weight * colour;
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
                   rough_lift: f32, ao_retain: f32, normal_mul: f32,
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
        // A restrained contrast stretch keeps scanned recesses legible
        // without clipping broad areas of the weathered face to black.
        sampled_albedo = max((sampled_albedo - vec3<f32>(0.099)) * 1.20 + vec3<f32>(0.099),
                             vec3<f32>(0.0));
    }
    (*albedo) += group_weight * sampled_albedo * albedo_tint;
    // AO retention scales cavity contrast, with zero giving an unoccluded surface.
    (*ao) += group_weight * (1.0 - (1.0 - albedo_ao_sample.a) * ao_retain);
    // A lift remaps the scan's roughness into [lift, 1] before the floor,
    // keeping its relative variation; zero leaves the scan as it is.
    (*rough) += group_weight * max(mix(rough_lift, 1.0, detail_rough_sample.a), rough_floor);
    (*world_detail) += group_weight * detail_rough_sample.xyz;
    (*weight_sum) += group_weight;
}

// Both terrain shading and the vegetation habitat capture resolve precisely
// the same biome/contact weights. A separate approximation for scatter would
// drift from the visible snowline, soil patches and exposed erosion beds.
fn shadeTerrain(input: FsInput, habitat: bool,
                habitat_data: ptr<function, vec4<f32>>) -> FsOutput
{
    var output: FsOutput;

    let frag_world_position = input.frag_world_position;
    // The G-buffer position is rebuilt from the world-space varying instead
    // of read from fragPositionView. On Metal that interpolant arrives as
    // zero in scattered fragments (1 px runs along ridges and silhouettes,
    // thousands per frame) while the world position beside it is intact. A
    // zero position hands the composite a degenerate view vector - Fresnel
    // at 1, no fog - so the pixel flashes white, and the water's
    // hidden-surface test discards over it, showing the seabed through the
    // sea. Grass-damped specular used to hide most of them.
    let frag_position_view = (globals.view * vec4<f32>(frag_world_position, 1.0)).xyz;

    var normalWorld = normalize(input.frag_world_normal);
    let height = frag_world_position.y;
    let materialNormal = normalize(input.frag_material_normal);
    let slope = 1.0 - clamp(materialNormal.y, 0.0, 1.0);
    let worldXZ = frag_world_position.xz;
    // Evaluate derivatives before any material gates. Explicit gradients keep
    // the mirrored patch fields filtered without seams at their folds.
    let worldStepX = dpdx(worldXZ);
    let worldStepY = dpdy(worldXZ);
    // Snow placement records long-term climate, not the instantaneous sun.
    // Keep drifts and melt margins fixed while the daily lighting moves.
    let climateSun = normalize(vec3<f32>(-0.35, 0.87, -0.32));
    let toSunXZ = normalize(climateSun.xz);

    // Use the same four-tile reveal and distance fade as the actual landform.
    // The surface atlas carries the simulation's own record: signed bed change
    // (R), ring concavity (G), drainage concentration (B) and the loose cover
    // left above bedrock (A); the flow atlas adds the routed contributing area
    // (A). Every simulated signal fades with reveal and the visibility radius,
    // where slope, geology and the base cover estimate carry on alone.
    var flowDomain: f32;
    var surface: vec4<f32>;
    let flow = blendedFlow(worldXZ, &flowDomain, &surface);
    // Contributing area separates rills from rivers: ~20 cells (320 m2) of
    // catchment is still a grassed hollow, ~2000 cells (3.2 ha) a permanent
    // stream. The ring concentration finds each channel's thalweg inside its
    // valley.
    let catchment = log2(1.0 + max(flow.a, 0.0));
    let dischargeAmount = smoothHermite(4.5, 11.0, catchment);
    let thalweg = smoothHermite(0.5, 2.8, surface.b);
    let channel = dischargeAmount * thalweg;
    let transport = smoothHermite(0.6, 4.0, min(length(flow.gb), 8.0))
                  * smoothHermite(0.002, 0.025, flow.r) * channel;
    // Signed bed change from fluvial transport, runoff and talus relaxation.
    // Centimetres of fresh sediment can bury turf; stripping rooted soil
    // takes a deeper cut.
    let incision = 1.0 - exp(-max(-surface.r - 0.012, 0.0) * 3.0);
    let deposition = 1.0 - exp(-max(surface.r - 0.004, 0.0) * 12.0);
    // Ring concavity re-anchored to the field's real scales: the 12 m-ring
    // relief of a common swale is 0.1-0.5 m, so (0.05, 1.2) puts a 0.3 m swale
    // at ~0.12 and saturates at 1.2 m bowls.
    let hollow = smoothHermite(0.05, 1.2, surface.g);
    let ridge = smoothHermite(0.05, 1.2, -surface.g);

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
    let rockRegion = filteredGroundNoise(
        rotateUV(groundDomain * 0.0031, 0.26) + vec2<f32>(-57.4, 18.9));
    let soilPattern = mix(soilLarge, soilSmall, mix(0.18, 0.62, groundGrowth))
                    + (soilEdge - 0.5) * 0.16;
    let soilThreshold = mix(0.64, 0.42, groundRegion);
    let soilPatches = smoothHermite(soilThreshold, soilThreshold + 0.18, soilPattern);

    // Loose cover above bedrock: soil, colluvium, talus and alluvium. The
    // simulation tracks it; beyond its radius (and before a tile reveals) the
    // estimate the simulation itself starts from stands in, from the same
    // shared geology: soil mantles gentle ground, thins with slope and on
    // resistant rock, and is gone where a face exceeds ~38 degrees.
    let steepness = sqrt(max(1.0 - materialNormal.y * materialNormal.y, 0.0))
                  / max(materialNormal.y, 0.05);
    let bedrockResistance = geologyResistance(worldXZ, height - max(surface.a, 0.0));
    let soilBody = geologyNoise(worldXZ / 41.0 + vec2<f32>(5.3, -27.7));
    let baseCover = 1.35 * (1.0 - smoothHermite(0.42, 0.78, steepness))
                  * mix(1.0, 0.45, bedrockResistance) * mix(0.70, 1.30, soilBody);
    let looseCover = max(surface.a, 0.0) + (1.0 - flowDomain) * baseCover;
    // Above roughly the treeline altitude soils stay thin and stony: alpine
    // turf thins out and frost-shattered rubble covers more of the ground.
    let alpine = smoothHermite(stage.sea_level + 80.0, stage.sea_level + 165.0,
                               height + (groundRegion - 0.5) * 30.0);

    // Weathered mantle. The talus relaxation strips loose cover from every
    // slope steeper than repose, yet weak rock keeps renewing a mantle of
    // shattered rubble and thin turf on faces up to ~50 degrees; only
    // resistant rock stands clean. Because the shared bedding keys hardness
    // to bedrock elevation, a steep face reads as ledges of stone between
    // bands of rubble and turf instead of one uniform slab.
    let weakRock = 1.0 - smoothHermite(0.30, 0.70,
                                       bedrockResistance + (soilSmall - 0.5) * 0.25);
    let steepMantle = weakRock * smoothHermite(0.50, 0.80, steepness)
                    * (1.0 - smoothHermite(0.95, 1.30, steepness));

    // Bedrock shows where a face sheds faster than it weathers: steepness
    // exposes it, resistant beds stand bare where weak ones keep a broken
    // mantle, and any loose cover the simulation left hides it. Convex noses
    // hold their faces, sheltered hollows keep soil, and patch noise at
    // several scales turns a 30-45 degree slope into a mosaic of outcrop,
    // rubble and turf instead of one slab. Past ~46 degrees little but rock
    // remains; scoured channel beds expose it on any slope.
    let steepFace = smoothHermite(0.55, 1.05, steepness);
    let hardBed = smoothHermite(0.30, 0.75, bedrockResistance);
    let outcropBreakup = (soilSmall - 0.5) * 0.35 + (soilEdge - 0.5) * 0.12
                       + (rockRegion - 0.5) * 0.30;
    let exposure = steepFace * mix(0.55, 1.15, hardBed)
                 - looseCover * 0.9
                 + 0.20 * ridge - 0.20 * hollow
                 + outcropBreakup + 0.12 * alpine
                 + channel * incision * 0.35 * hardBed;
    let fRock = smoothHermite(0.30, 0.62, exposure);

    // Footslopes: below an exposure, stone rarely meets closed turf. What a
    // face sheds - frost-shattered debris, soil washed off it - comes to rest
    // at its foot and in the hollows between its spurs as stony colluvium the
    // turf has not closed over, so a band of dirt and rubble separates bare
    // rock from the grass beneath it. Above a face the crest only loses
    // material, so the turf thins to a narrow, faint rim of stony soil before
    // the stone. The simulation records which side is which: the talus
    // relaxation and slope wash leave deposits and concave footslopes below
    // a face, and lower the convex crest above it. The band is the stretch of
    // the rock's own exposure field just short of bare rock, so it widens
    // where a face wanes gradually into its base; thick footslope deposits
    // (not channel fill) extend it as colluvial aprons. Beyond the simulated
    // radius, where neither side is known, only the faint rim remains.
    // The band's outer edge is ragged at two scales, so grass islands
    // survive in the debris and dirt bays reach into the turf over a few
    // metres instead of meeting along one line.
    let fringeNoise = (soilSmall - 0.5) * 0.22 + (soilEdge - 0.5) * 0.16
                    + (groundGrowth - 0.5) * 0.06;
    let footslope = clamp(hollow * 1.6 + smoothHermite(0.02, 0.35, surface.r)
                          - ridge * 1.5 - smoothHermite(0.05, 0.60, -surface.r) * 0.6,
                          0.0, 1.0);
    let colluvium = smoothHermite(0.08, 0.90, surface.r)
                  * smoothHermite(0.20, 0.55, steepness)
                  * (1.0 - thalweg * dischargeAmount);
    let fringeBelow = smoothHermite(-0.12, 0.42, exposure + fringeNoise) * footslope;
    let fringeRim = smoothHermite(0.12, 0.40, exposure + fringeNoise) * 0.30;
    let rockFringe = clamp(max(max(fringeBelow, fringeRim), colluvium * 0.7), 0.0, 1.0)
                   * (1.0 - fRock);
    // Turf thins and dries before it gives way to the debris: stressed
    // grass on stony ground turns olive a little way out from the band.
    let fringeApproach = smoothHermite(-0.30, 0.25, exposure + fringeNoise)
                       * max(footslope, 0.25) * (1.0 - fRock);

    // Talus: debris the relaxation dropped at the foot of steep faces, lying
    // close to its angle of repose (~25-35 degrees). Rubble also mantles weak
    // steep ground, increasingly above the treeline where turf gives out.
    // Deposits steeper than repose have already slid, and gentle aprons grade
    // into soil.
    let talusSlope = smoothHermite(0.36, 0.58, steepness)
                   * (1.0 - smoothHermite(0.85, 1.15, steepness));
    let talusDeposit = smoothHermite(0.04, 0.50, surface.r) * smoothHermite(0.08, 0.45, looseCover);
    let rubblePatch = smoothHermite(0.38, 0.62, soilLarge + (soilEdge - 0.5) * 0.30);
    let alpineRubble = alpine * (1.0 - smoothHermite(0.45, 1.10, looseCover)) * 0.45 * rubblePatch;
    let mantleRubble = steepMantle * mix(0.30, 0.80, alpine) * rubblePatch;
    let scree = talusSlope * clamp(talusDeposit + alpineRubble + mantleRubble, 0.0, 1.0);
    // Bed load: coarse gravel lines scoured, active channels and the bars
    // they leave; where the same water slows over its own deposits, sand and
    // silt settle instead. Fans keep a gravelly apex near their feeder.
    let channelScour = channel * max(incision, transport);
    let channelBar = channel * deposition;
    let debrisHold = 1.0 - smoothHermite(0.80, 1.10, steepness);
    // Stony colluvium carries scattered rubble patches of its own.
    let fringeRubble = rockFringe * rubblePatch * 0.35;
    let fGravel = clamp(max(max(scree, fringeRubble),
                            channelScour * mix(0.55, 1.0, transport)
                            + channelBar * mix(0.25, 0.85, transport)), 0.0, 1.0)
                * debrisHold;
    let soilFlatness = 1.0 - smoothHermite(0.02, 0.25, slope);
    let sedimentSand = channelBar * (1.0 - transport) * soilFlatness;
    let fSand = max(1.0 - fGrass, sedimentSand);

    // Soil between the turf: broad background patches, thin stony cover on
    // steeper ground, cut banks and fresh fans beside active channels. A
    // bounded union lets any strong disturbance displace grass while noise
    // varies the transition edges without weakening fully bare ground.
    let soilRetention = 1.0 - smoothHermite(0.25, 0.75, steepness);
    let backgroundSoil = (0.015 + 0.52 * soilPatches * mix(0.45, 1.0, groundRegion)) * soilFlatness;
    // Turf roots in a few decimetres of soil; only stony, nearly bare ground
    // shows through it. Convex shoulders above a face keep their turf to the
    // stone, so their thin soil shows less.
    let thinSoil = (1.0 - smoothHermite(0.06, 0.30, looseCover + steepMantle * 0.5
                                                    + (soilEdge - 0.5) * 0.10))
                 * smoothHermite(0.15, 0.40, steepness) * (1.0 - 0.7 * ridge);
    let scouredSoil = incision * soilRetention * smoothHermite(0.15, 0.45, dischargeAmount);
    // Older fill on floodplains and fan surfaces revegetates; only deposits
    // beside the active thread stay raw.
    let freshFan = deposition * dischargeAmount * smoothHermite(0.25, 0.75, thalweg) * soilFlatness;
    let channelSoil = channel * (1.0 - transport) * soilRetention;
    var disturbedSoil = 1.0 - (1.0 - scouredSoil) * (1.0 - freshFan) * (1.0 - channelSoil);
    disturbedSoil += (soilEdge - 0.5) * 0.6 * disturbedSoil * (1.0 - disturbedSoil);
    let grassSoilBlend = clamp(1.0 - (1.0 - backgroundSoil) * (1.0 - disturbedSoil)
                                   * (1.0 - thinSoil * 0.55) * (1.0 - alpine * 0.30)
                                   * (1.0 - rockFringe * 0.90), 0.0, 1.0);
    let resistantBed = bedrockResistance;

    // Snow lies where the ground stays cold through the melt season. The
    // regional snowline wanders with broad climate cells; shaded (poleward)
    // slopes hold snow tens of metres lower than flat ground and sun-facing
    // slopes lose it higher; sheltered hollows and lee slopes keep a deeper
    // pack and wind-scoured crests and windward faces a thinner one. Snow does
    // not flow downhill: a gully below the line melts out like any other low
    // ground, so cover never reaches warm valleys. Only shaded avalanche
    // gullies just under the line keep a short tongue of old debris snow.
    // Every term is a world-space field; only the slope they read is
    // prefiltered with distance (see the vertex stage), so cover stays put as
    // the camera moves and only sub-vertex detail softens far away.
    let snowRegion = filteredGroundNoise(
        rotateUV(groundDomain * 0.0017, 0.63) + vec2<f32>(91.7, -53.2));
    let snowDrift = filteredGroundNoise(
        rotateUV(groundDomain * 0.013, 1.19) + vec2<f32>(-23.8, 67.4));
    let snowDriftFine = filteredGroundNoise(
        rotateUV(groundDomain * 0.05, 0.41) + vec2<f32>(37.1, 12.6));
    let snowDriftMicro = filteredGroundNoise(
        rotateUV(groundDomain * 0.15, 1.87) + vec2<f32>(-61.3, -42.9));
    // Melt-season insolation relative to level ground. The spring sun stands
    // lower than the summer climate sun, so aspect matters more: a steep
    // shaded face holds snow ~20 m lower than a meadow at the same height,
    // and a sun-facing slope loses it ~9 m higher. The melt sun stands only a
    // little west of south: poleward against sunward aspect dominates, and
    // the east and west flanks of one spur stay nearly alike instead of
    // alternating white and bare down every rib of a mountainside.
    let meltSun = normalize(vec3<f32>(-0.22, 0.62, -0.76));
    let meltExposure = max(dot(materialNormal, meltSun), 0.0);
    let aspectShift = (meltSun.y - meltExposure) * 34.0;
    // Wind: prevailing south-westerlies scour windward aspects and crests
    // and load lee slopes; the aspect term is slope-gated so flat snowfields
    // (degenerate aspect) are left alone.
    let aspectN = normalize(materialNormal.xz + vec2<f32>(1e-4, 0.0));
    let windAlignment = dot(aspectN, normalize(vec2<f32>(0.70, 0.42)));
    let windGate = smoothHermite(0.08, 0.30, slope);
    let scour = smoothHermite(0.15, 0.60, -windAlignment) * windGate;
    let leeLoad = smoothHermite(0.20, 0.70, windAlignment) * windGate;
    let snowHeight = height + (snowRegion - 0.5) * 48.0
                            + (snowDrift - 0.5) * 16.0
                            + (snowDriftFine - 0.5) * 6.0
                            + (snowDriftMicro - 0.5) * 2.5
                            + aspectShift
                            + hollow * 7.0 - ridge * 9.0
                            + leeLoad * 6.0 - scour * 8.0;
    let snowLine = smoothHermite(stage.snowline_altitude - 13.0, stage.snowline_altitude + 13.0,
                                 snowHeight - stage.sea_level);
    // Snow cannot cling to walls: it thins past ~39 degrees, where sluffs keep
    // clearing it, and sheds by ~55, so steep faces stay dark rock cut
    // through the white.
    let snowHold = 1.0 - smoothHermite(0.80, 1.45, steepness);
    var fSnow = snowLine * snowHold;

    // Thin pack opens first over convex noses, steep, sun-facing and freshly
    // cut ground, at every viewing distance.
    let marginBand = snowLine * (1.0 - snowLine) * 4.0;
    let marginShed = clamp(0.9 * ridge + smoothHermite(0.45, 0.95, steepness)
                         + 0.8 * smoothHermite(0.75, 0.95, meltExposure)
                         + 0.5 * incision, 0.0, 1.0);
    fSnow = fSnow * (1.0 - 0.70 * marginBand * marginShed);

    // Meltwater runs under and through the pack: permanent streams open dark
    // ribbons, while small rills stay buried. Resistant beds melt clear;
    // soft, debris-choked beds keep a thin skin.
    let meltChannel = channel * smoothHermite(0.25, 0.75, dischargeAmount);
    fSnow = fSnow * (1.0 - meltChannel * mix(0.55, 0.90, resistantBed));

    // Avalanche debris: shaded gullies collect snow sliding off the steep
    // ground above them and keep it a little below the line, as long as the
    // ground stays cold. The tongue is short (it ends ~18 m under the local
    // line) and sun-facing gullies get none.
    let shade = 1.0 - smoothHermite(0.30, 0.65, meltExposure);
    let avalancheTongue = smoothHermite(stage.snowline_altitude - 18.0, stage.snowline_altitude - 4.0,
                                        snowHeight - stage.sea_level)
                        * hollow * smoothHermite(0.15, 0.60, dischargeAmount) * shade
                        * snowHold * flowDomain * 0.85;
    fSnow = 1.0 - (1.0 - fSnow) * (1.0 - avalancheTongue);

    // Rivers, creeks and lakes (src/rivers). The bed under the water sorts
    // by the power of the current over it: slow pools, lake beds and point
    // bars settle sand, runs and riffles keep a gravel bed, and cascades
    // and the faces of falls are scoured to bedrock. Out of the water the
    // turf reaches the waterline, except where the bank tells otherwise:
    // on the outside of a bend the current undercuts a steep bank of bare
    // soil and roots, on the inside it leaves a bar of sand and gravel
    // standing out of the water, and beside whitewater the banks are stone.
    // Every edge wanders, so no waterline runs parallel to its channel.
    let riverBank = input.frag_river.x + (soilEdge - 0.5)*0.35 + (soilSmall - 0.5)*0.25;
    let riverBend = input.frag_river.y;
    let riverTurbulence = input.frag_river.z;
    let riverPower = input.frag_river.w*(1.0 + 1.5*riverTurbulence);
    let inRiver = 1.0 - smoothHermite(-0.30, 0.05, riverBank);
    let riverRock = smoothHermite(0.45, 0.85, riverTurbulence);
    let riverSand = (1.0 - smoothHermite(0.12, 0.45, riverPower + (soilSmall - 0.5)*0.3))*(1.0 - riverRock);
    let riverGravel = clamp(1.0 - riverRock - riverSand, 0.0, 1.0);
    // The local bank: how steeply this very ground rises from the water.
    let bankSteep = smoothHermite(0.10, 0.35, 1.0 - normalWorld.y);
    let cutBank = bankSteep*(0.45 + 0.55*smoothHermite(0.0, 0.35, riverBend))
                * (1.0 - smoothHermite(0.25, 0.75 + 0.6*soilSmall, riverBank))*(1.0 - inRiver);
    let pointBar = smoothHermite(0.04, 0.30, -riverBend)*(1.0 - bankSteep)
                 * (1.0 - smoothHermite(0.20, 0.55 + 1.0*soilLarge, riverBank))*(1.0 - inRiver)
                 * smoothHermite(0.30, 0.60, soilSmall + (soilEdge - 0.5)*0.5);
    let stoneBank = riverRock*(1.0 - smoothHermite(0.15, 0.9 + 0.6*soilEdge, riverBank))*(1.0 - inRiver);
    // A calm stream's banks are cohesive earth bound by roots, not the
    // scree and bedrock a hillside of the same steepness sheds, nor the bare
    // gravel and sand the erosion's working water leaves in its furrows:
    // within a few metres of the water those rules give way to soil and a
    // lush riparian turf, and where the bank is steep the earth shows
    // through it in patches. Whitewater keeps its stony banks.
    let riverside = (1.0 - smoothHermite(1.5, 5.0 + 4.0*soilLarge, riverBank))
                  * (1.0 - riverRock)*(1.0 - inRiver);
    let bankEarth = riverside*bankSteep
                  * smoothHermite(0.40, 0.70, soilSmall + (soilEdge - 0.5)*0.5);
    let fSandRiver = max(max(1.0 - fGrass, sedimentSand*(1.0 - riverside)),
                         inRiver*riverSand + pointBar*(0.35 + 0.5*riverSand));
    let fGravelRiver = max(fGravel*(1.0 - 0.9*riverside),
                           inRiver*riverGravel + pointBar*(0.65 - 0.5*riverSand) + stoneBank*0.5);
    let fRockRiver = max(fRock*(1.0 - 0.9*riverside), inRiver*riverRock + stoneBank*0.6);
    let fSnowRiver = fSnow*(1.0 - max(inRiver, max(pointBar, stoneBank)*0.7));
    let grassSoilRiver = max(grassSoilBlend*(1.0 - 0.6*riverside), max(cutBank, bankEarth));

    let groundCover = (1.0 - fSandRiver) * (1.0 - fGravelRiver) * (1.0 - fRockRiver) * (1.0 - fSnowRiver);
    let wSand = fSandRiver * (1.0 - fGravelRiver) * (1.0 - fRockRiver) * (1.0 - fSnowRiver);
    let wGrass = groundCover * (1.0 - grassSoilRiver);
    let wDirt = groundCover * grassSoilRiver;
    // Loose debris and channel beds lie on top of the bedrock they came from,
    // so talus aprons and gravel bars cover rock rather than the reverse.
    let wRock = fRockRiver * (1.0 - fGravelRiver) * (1.0 - fSnowRiver);
    let wGravel = fGravelRiver * (1.0 - fSnowRiver);
    let wSnow = fSnowRiver;

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
    grassDryness = clamp(grassDryness - hollow * (1.0 - channel) * 0.22
                         - dischargeAmount * (1.0 - transport) * 0.12
                         + ridge * 0.12
                         + smoothHermite(0.25, 0.60, steepness) * 0.10
                         + alpine * 0.22
                         + fringeApproach * 0.28
                         // Riparian turf stays lush and dark green.
                         - (1.0 - smoothHermite(1.0, 9.0 + 6.0 * soilLarge, riverBank)) * 0.45,
                         0.0, 1.0);
    let grassTint = mix(vec3<f32>(0.31, 0.39, 0.27), vec3<f32>(0.48, 0.40, 0.28), grassDryness)
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
    var groupWeights = array<f32, 6>(wSnow, wGrass, wSand,
                                     wDirt, wGravel, wRock);
    var crossfades = array<f32, 6>(0.0, 0.0, 0.0,
                                   nDirt, nGravel, 0.0);
    var albedoTints = array<vec3<f32>, 6>(
        snowTint, grassTint,
        // Muted mineral sand suits a cold coastline; dampness still follows flow.
        vec3<f32>(0.49, 0.46, 0.39),
        // Soil: humic brown in the lowlands, greyer and stonier where it is
        // colluvium shed from rock or frost-worked ground above the treeline.
        mix(vec3<f32>(0.40, 0.33, 0.25), vec3<f32>(0.35, 0.33, 0.30),
            clamp(rockFringe + alpine * 0.6, 0.0, 1.0)),
        // Weathered gravel and talus should sit within the same exposure as
        // the turf, including the small patches newly exposed in drainage
        // channels; a lighter tint reads as lingering snow from afar.
        vec3<f32>(0.46, 0.46, 0.43),
        // Neutral weathered bedrock, with colour variation added below at
        // geological scales after the scan detail has been resolved. Kept
        // well below the snow and a step below the scree so faces read
        // against both, as weathered granite and gneiss do.
        vec3<f32>(0.48, 0.46, 0.43));
    // Stony colluvium and frost-worked alpine soil lose the humic colour of
    // lowland soil, so the dirt desaturates with them.
    var albedoDesats = array<f32, 6>(0.0, 0.20, 0.28,
                                     mix(0.12, 0.45, clamp(rockFringe + alpine * 0.6, 0.0, 1.0)),
                                     0.0, 0.0);
    var roughFloors = array<f32, 6>(0.72, 0.72, 0.70,
                                    0.88, 0.80, 0.65);
    // Dry stone is among the roughest ground there is, but the scans are
    // smooth: Rock032 runs 0.58-0.78 (mean 0.70) and both gravels 0.47-0.66,
    // so gravel sat flat on its floor and rock read polished, sheening at
    // every grazing sun angle. Lifting each map into [lift, 1] keeps its
    // pockets and scars while placing rock at ~0.89-0.94 and gravel at
    // ~0.89-0.93, alongside the soil's 0.88.
    var roughLifts = array<f32, 6>(0.0, 0.0, 0.0,
                                   0.0, 0.80, 0.73);
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
                                   0.85, 1.0, 0.48);
    // Resolve each material's three projections first, then blend whole
    // surfaces. Scan cavities approximate local relief at the contact edge:
    // exposed grains/tufts survive while the adjacent material fills gaps.
    // This is a cavity proxy, not a displacement map. All PBR channels and
    // the lighting masks use the resulting coverage, avoiding pale ghosted
    // mixtures of snow, turf and stone across the full slope transition.
    var surfaceAlbedo: array<vec3<f32>, 6>;
    var surfaceDetail: array<vec3<f32>, 6>;
    var surfaceRough: array<f32, 6>;
    var surfaceAO: array<f32, 6>;
    var scores: array<f32, 6>;
    var edgeWidths: array<f32, 6>;
    // Derivatives must run before the nonuniform texture-sampling branches.
    for (var g = 0; g < 6; g++) {
        edgeWidths[g] = min(fwidth(groupWeights[g]), 0.30);
    }
    let contactDetail = 1.0 - smoothHermite(0.08, 0.65, footprint);
    var highestScore = -1.0;
    var highestGroup = -1;
    for (var g = 0; g < 6; g++) {
        scores[g] = -1.0;
        if (groupWeights[g] <= 0.01) { continue; }
        var colour = vec3<f32>(0.0);
        var detail = vec3<f32>(0.0);
        var roughness = 0.0;
        var cavity = 0.0;
        var projectionSum = 0.0;
        if (slopeWeights.y >= 0.02) {
            accumulateGroup(groupBases[g], slopeWeights.y, coordY, dYx, dYy, crossfades[g],
                            vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 1.0, 0.0),
                            normalWorld, globals.settings_b.x * scanDetailFade,
                            albedoTints[g], albedoDesats[g], roughFloors[g], roughLifts[g],
                            aoRetains[g], normalMuls[g],
                            &colour, &detail, &roughness, &cavity, &projectionSum);
        }
        if (slopeWeights.x >= 0.02) {
            accumulateGroup(groupBases[g], slopeWeights.x, coordX, dXx, dXy, crossfades[g],
                            vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0),
                            normalWorld, globals.settings_b.x * scanDetailFade,
                            albedoTints[g], albedoDesats[g], roughFloors[g], roughLifts[g],
                            aoRetains[g], normalMuls[g],
                            &colour, &detail, &roughness, &cavity, &projectionSum);
        }
        if (slopeWeights.z >= 0.02) {
            accumulateGroup(groupBases[g], slopeWeights.z, coordZ, dZx, dZy, crossfades[g],
                            vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 0.0, 1.0),
                            normalWorld, globals.settings_b.x * scanDetailFade,
                            albedoTints[g], albedoDesats[g], roughFloors[g], roughLifts[g],
                            aoRetains[g], normalMuls[g],
                            &colour, &detail, &roughness, &cavity, &projectionSum);
        }
        let invProjection = 1.0 / max(projectionSum, 0.0001);
        surfaceAlbedo[g] = colour * invProjection;
        surfaceDetail[g] = detail * invProjection;
        surfaceRough[g] = roughness * invProjection;
        surfaceAO[g] = cavity * invProjection;
        let relief = (surfaceAO[g] - 0.65) * 0.22 * contactDetail;
        scores[g] = groupWeights[g] + relief * 4.0 * groupWeights[g] * (1.0 - groupWeights[g]);
        if (scores[g] > highestScore) {
            highestScore = scores[g];
            highestGroup = g;
        }
    }
    for (var g = 0; g < 6; g++) {
        // Turf and soil are both low, matte ground: where they meet each
        // other, tufts grow through the soil and soil shows between tufts
        // over metres, so their contact opens much wider than any other
        // pair's and the two mix in proportion across the transition.
        let groundPair = (g == 1 && highestGroup == 3) || (g == 3 && highestGroup == 1);
        let width = 0.34 + edgeWidths[g] + select(0.0, 0.60, groundPair);
        let contact = smoothHermite(highestScore - width, highestScore, scores[g]);
        // Keep a small mineral contribution below the contact threshold;
        // thin silt and sparse grains should not vanish from distant turf.
        let weight = select(0.0, groupWeights[g] * mix(0.08, 1.0, contact), scores[g] > -0.5);
        groupWeights[g] = weight;
        albedo += surfaceAlbedo[g] * weight;
        worldDetail += surfaceDetail[g] * weight;
        rough += surfaceRough[g] * weight;
        ao += surfaceAO[g] * weight;
        weightSum += weight;
    }
    let denom = max(weightSum, 0.0001);
    albedo /= denom;
    rough /= denom;
    ao /= denom;
    let gSnow = groupWeights[0] / denom;
    let gGrass = groupWeights[1] / denom;
    let gSand = groupWeights[2] / denom;
    let gDirt = groupWeights[3] / denom;
    let gGravel = groupWeights[4] / denom;
    let gRock = groupWeights[5] / denom;

    if (habitat)
    {
        // Dirt and sand may show between roots where this very same
        // material blend still reads primarily as grass. Snow, rock and
        // loose gravel do not hold these plants, even in a mixed pixel.
        // Erode their suitability before the candidate's whole root
        // footprint is tested in the grass vertex stage.
        let grassDominance = smoothHermite(0.55, 0.82, gGrass);
        let forbiddenCover = gSnow + gRock + gGravel;
        let turf = grassDominance
                 * (1.0 - smoothHermite(0.004, 0.028, forbiddenCover));

        // The capture supplies the rendered ocean's maximum crest clearance
        // in the otherwise unused camera_position.w. The extra dry fringe
        // prevents wave wash and interpolation at the shoreline from planting
        // roots underwater. Hydraulic flow.r is the *solver's* working water,
        // which is not drawn as a lake or river in the current game; healthy
        // valley turf can have a large temporary depth after erosion prewarm.
        // Active drainage is excluded below using cuts, concentration and
        // channel evidence rather than that unrendered simulation depth.
        let dryShore = smoothHermite(stage.sea_level + max(globals.camera_position.w, 0.35),
                                    stage.sea_level + max(globals.camera_position.w, 0.35) + 0.30,
                                    height);

        // A grassy valley can be concave and drain a large catchment. A true
        // eroded furrow needs signed bed loss as well as focused concavity or
        // discharge, so broad swales and simulated rain flow remain planted.
        // The cut persists after runoff stops, keeping dry furrows clear.
        let cutDepth = max(-surface.r, 0.0);
        let incisedRoots = smoothHermite(0.030, 0.120, cutDepth);
        let focusedCut = smoothHermite(0.06, 0.30, surface.g)
                         * smoothHermite(0.012, 0.080, cutDepth);
        let channelCut = channel
                         * smoothHermite(0.045, 0.25, surface.g)
                         * smoothHermite(0.010, 0.060, cutDepth);
        let furrow = max(incisedRoots, max(focusedCut, channelCut));
        let intactRoots = 1.0 - smoothHermite(0.10, 0.50, furrow);
        // Do not root upright clumps on cliff triangles even if a small
        // interpolated material patch happens to meet the turf threshold.
        let stableSlope = smoothHermite(0.70, 0.86, normalWorld.y);
        // Nothing roots in a river or a lake, or on the margin its water
        // laps. The capture is only redrawn when its inputs change, so it
        // looks the water up exactly at every texel instead of trusting the
        // clipmap's vertices, which farther out stand wider apart than a
        // creek is wide.
        let water = riverEnvelope(worldXZ);
        let dryBank = smoothHermite(0.25, 0.75, riverBankAt(water, height));
        let suitability = clamp(turf * dryShore * intactRoots * stableSlope * dryBank, 0.0, 1.0);
        (*habitat_data) = vec4<f32>(height, suitability, normalWorld.x, normalWorld.z);
    }

    // The blended world-space detail vector perturbs the geometric normal
    // before a single rotation into view space. Because every tangent frame is
    // anchored in world space (and matched to its material's UV rotation), the
    // normal detail stays pinned to the terrain as the camera moves instead of
    // swimming with the view, and it stays aligned with its own albedo.
    var perturbedWorld = normalize(worldDetail / max(weightSum, 1e-4));

    // Mineral colour varies through the geological mass as well as across
    // individual scan tiles. Restrained tilted bedding gives cliff faces a
    // common structure without forcing every face into horizontal stripes.
    let mineralMottle = (rockRegion - 0.5) * 0.24 + (soilLarge - 0.5) * 0.16;
    albedo *= 1.0 + gGrass * ((groundGrowth - 0.5) * 0.22 + (soilLarge - 0.5) * 0.16)
                  + (gDirt + gGravel + gSand) * (soilLarge - 0.5) * 0.22
                  + gRock * mineralMottle;
    if (gRock > 0.001) {
        let wy = rockWeathering(coordY, dYx, dYy);
        let wx = rockWeathering(coordX, dXx, dXy);
        let wz = rockWeathering(coordZ, dZx, dZy);
        let reliefGradient = vec3<f32>(wy.y, 0.0, wy.z) * slopeWeights.y
                           + vec3<f32>(0.0, wx.z, wx.y) * slopeWeights.x
                           + vec3<f32>(wz.y, wz.z, 0.0) * slopeWeights.z;
        let tangentGradient = reliefGradient - normalWorld * dot(reliefGradient, normalWorld);
        perturbedWorld = normalize(perturbedWorld - tangentGradient * gRock * globals.settings_b.x);
        let weather = dot(vec3<f32>(wx.x, wy.x, wz.x), slopeWeights);
        let bedCoordinate = height * 0.095 + dot(worldXZ, vec2<f32>(0.014, -0.009))
                          + (rockRegion - 0.5) * 1.6;
        let bedding = sin(bedCoordinate * 6.2831853)
                    * (1.0 - smoothHermite(1.5, 6.0, footprint));
        albedo *= 1.0 + gRock * (weather * 0.38 + bedding * 0.045);
        rough = clamp(rough + gRock * weather * 0.10, 0.0, 1.0);

        // Jointed blocks. Faces break into blocks ~3 m across, with a finer
        // ~1 m set inside them: each block's face tilts a little its own way,
        // weathers a little lighter or darker, and dark joints open between
        // blocks. On the side projections the blocks are flattened into slabs
        // (bedding planes crossed by vertical joints); seen from above they
        // are equant. Facets fade before a block is a few pixels across, and
        // the joints, being narrow, earlier still: the ~20 cm coarse joints
        // by ~3 px, the ~5 cm fine ones by ~2 px. Blocks under a few percent
        // of the cover cannot be seen, and, as with the scans, only the
        // projections carrying at least 2% of the slope weight are searched,
        // so a face pays for one or two Voronoi pairs rather than three.
        let coarseFade = 1.0 - smoothHermite(0.16, 0.40, footprint);
        let fineFade = 1.0 - smoothHermite(0.06, 0.16, footprint);
        let jointFade = 1.0 - smoothHermite(0.025, 0.07, footprint);
        let fineJointFade = 1.0 - smoothHermite(0.012, 0.03, footprint);
        if (coarseFade > 0.001 && gRock > 0.04)
        {
            // Facet tilts enter world space through each projection's axes,
            // as the weathering relief does.
            var tiltCoarse = vec3<f32>(0.0);
            var tiltFine = vec3<f32>(0.0);
            var border = 0.0;
            var fineBorder = 0.0;
            var blockShade = 0.0;
            var jointWeight = 0.0;
            if (slopeWeights.y >= 0.02)
            {
                let coarse = rockJoints(coordY / 2.8, 31);
                let fine = rockJoints(coordY / 0.95, 43);
                tiltCoarse += vec3<f32>(coarse.y, 0.0, coarse.z) * (slopeWeights.y * facetTaper(coarse.x));
                tiltFine += vec3<f32>(fine.y, 0.0, fine.z) * (slopeWeights.y * facetTaper(fine.x));
                border += coarse.x * slopeWeights.y;
                fineBorder += fine.x * slopeWeights.y;
                blockShade += coarse.w * slopeWeights.y;
                jointWeight += slopeWeights.y;
            }
            if (slopeWeights.x >= 0.02)
            {
                let coarse = rockJoints(coordX / vec2<f32>(3.2, 1.7), 37);
                let fine = rockJoints(coordX / vec2<f32>(1.1, 0.6), 47);
                tiltCoarse += vec3<f32>(0.0, coarse.z, coarse.y) * (slopeWeights.x * facetTaper(coarse.x));
                tiltFine += vec3<f32>(0.0, fine.z, fine.y) * (slopeWeights.x * facetTaper(fine.x));
                border += coarse.x * slopeWeights.x;
                fineBorder += fine.x * slopeWeights.x;
                blockShade += coarse.w * slopeWeights.x;
                jointWeight += slopeWeights.x;
            }
            if (slopeWeights.z >= 0.02)
            {
                let coarse = rockJoints(coordZ / vec2<f32>(3.2, 1.7), 41);
                let fine = rockJoints(coordZ / vec2<f32>(1.1, 0.6), 53);
                tiltCoarse += vec3<f32>(coarse.y, coarse.z, 0.0) * (slopeWeights.z * facetTaper(coarse.x));
                tiltFine += vec3<f32>(fine.y, fine.z, 0.0) * (slopeWeights.z * facetTaper(fine.x));
                border += coarse.x * slopeWeights.z;
                fineBorder += fine.x * slopeWeights.z;
                blockShade += coarse.w * slopeWeights.z;
                jointWeight += slopeWeights.z;
            }
            let projectionScale = 1.0 / max(jointWeight, 0.0001);
            let facetTilt = tiltCoarse * (0.22 * coarseFade) + tiltFine * (0.10 * fineFade);
            let facetTangent = facetTilt - normalWorld * dot(facetTilt, normalWorld);
            perturbedWorld = normalize(perturbedWorld
                                     - facetTangent * gRock * globals.settings_b.x);
            let jointCoarse = 1.0 - smoothHermite(0.0, 0.07, border * projectionScale);
            let jointFine = 1.0 - smoothHermite(0.0, 0.05, fineBorder * projectionScale);
            albedo *= 1.0 + gRock * ((blockShade * projectionScale - 0.5) * 0.16 * coarseFade
                                     - jointCoarse * 0.42 * jointFade
                                     - jointFine * 0.22 * fineJointFade * fineFade);
            ao = mix(ao, ao * (1.0 - 0.45 * jointCoarse), gRock * jointFade);
        }

        // Runoff stains: water routed over steep rock gathers along the
        // simulated rills and leaves dark streaks down the face, lined up with
        // the gullies the same drainage cut below. They fade with the erosion
        // cache like the rest of the flow evidence. A stain is a dry film of
        // oxide and algae, so it darkens the stone without polishing it.
        let runoffStain = smoothHermite(2.5, 6.5, catchment)
                        * smoothHermite(0.45, 0.90, steepness);
        albedo *= 1.0 - gRock * runoffStain * 0.32;
        // Lichen: grey-green crusts on shaded, damp faces, ochre on sunny
        // ones, in patches that thin toward the summits.
        let lichen = smoothHermite(0.55, 0.80, soilSmall * 0.6 + soilEdge * 0.4)
                   * (1.0 - 0.5 * alpine);
        let dampFace = 1.0 - smoothHermite(0.35, 0.80, meltExposure);
        albedo = mix(albedo, albedo * vec3<f32>(0.90, 0.98, 0.84), gRock * lichen * dampFace * 0.55);
        albedo = mix(albedo, albedo * vec3<f32>(1.10, 0.97, 0.80),
                     gRock * lichen * (1.0 - dampFace) * 0.40);
    }

    // Talus and rubble sort their debris along the fall line: rockfall and
    // small debris flows leave long streaks of coarser and finer, lighter and
    // darker scree running straight downslope, which keeps a smooth repose
    // slope from reading as one uniform sheet. The frame follows the 4 m
    // aspect, so streaks bend with the apron. Each octave fades out while
    // its period still spans four or more pixels: closer to a pixel, the
    // noise aliases into light and dark specks that crawl as the view moves.
    let screeStreaks = gGravel * smoothHermite(0.40, 0.65, steepness);
    let broadStreakFade = 1.0 - smoothHermite(0.42, 0.83, footprint);   // 3.3 m across
    let fineStreakFade = 1.0 - smoothHermite(0.13, 0.26, footprint);    // 1.05 m across
    if (screeStreaks > 0.01 && broadStreakFade > 0.001)
    {
        let aspect = materialNormal.xz;
        let aspectLength = dot(aspect, aspect);
        let fallLine = select(vec2<f32>(1.0, 0.0), aspect * inverseSqrt(aspectLength),
                              aspectLength > 1e-8);
        let across = dot(worldXZ, vec2<f32>(-fallLine.y, fallLine.x));
        let along = dot(worldXZ, fallLine);
        let streak = (groundNoise(vec2<f32>(across * 0.30, along * 0.035) + vec2<f32>(11.3, -7.7))
                      - 0.5) * 0.7 * broadStreakFade
                   + (groundNoise(vec2<f32>(across * 0.95, along * 0.11) + vec2<f32>(-23.1, 5.9))
                      - 0.5) * 0.3 * fineStreakFade;
        albedo *= 1.0 + screeStreaks * streak * 0.45;
    }

    // Block-scale rubble. The gravel scans resolve pebbles, which are
    // sub-pixel beyond a few tens of metres, while a real talus slope stays
    // lumpy with metre and half-metre blocks: lit block tops, shaded gaps.
    // Two relief octaves tilt the normal and shade the albedo, each fading
    // out while its period still spans four or more pixels, before its
    // normals can alias into sparkle.
    let rubbleWeight = gGravel * smoothHermite(0.30, 0.60, steepness);
    let blockFade = 1.0 - smoothHermite(0.16, 0.31, footprint);   // 1.25 m blocks
    let chipFade = 1.0 - smoothHermite(0.054, 0.109, footprint);  // 0.43 m chips
    if (rubbleWeight > 0.01 && blockFade > 0.001)
    {
        let blocks = groundNoiseGradient(worldXZ * 0.8 + vec2<f32>(3.7, -9.2));
        let chips = groundNoiseGradient(worldXZ * 2.3 + vec2<f32>(-14.1, 6.6));
        albedo *= 1.0 + rubbleWeight * ((blocks.x - 0.5) * 0.34 * blockFade
                                        + (chips.x - 0.5) * 0.20 * chipFade);
        // Block heights ~0.35 m and chip heights ~0.12 m, as world slopes.
        let rubbleGradient = blocks.yz * (0.35 * 0.8) * blockFade
                           + chips.yz * (0.12 * 2.3) * chipFade;
        let rubbleTilt = vec3<f32>(rubbleGradient.x, 0.0, rubbleGradient.y);
        let rubbleTangent = rubbleTilt - normalWorld * dot(rubbleTilt, normalWorld);
        perturbedWorld = normalize(perturbedWorld
                                 - rubbleTangent * rubbleWeight * globals.settings_b.x);
    }

    // Gentle wind relief over the single powder scan, at ~22 m and ~8 m
    // wavelengths. This perturbs shading only, keeping soft sun/lee variation
    // as the scan mips out without changing the terrain silhouette.
    var snowGrainValue = 0.5;
    var snowCrystalValue = 0.5;
    // The macro gradient is also consumed by the mottle coupling in the
    // near-field detail block below, so it outlives this gate.
    var macroGradient = vec2<f32>(0.0);
    // Habitat colour is only consumed where snow coverage is effectively
    // zero. Skip the costly snow relief on that offscreen capture.
    if (!habitat && gSnow > 0.02)
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
    if (!habitat && globals.settings_b.y > 0.0 && gSnow > 0.02)
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
        let fringe = clamp(min(gSnow, gGrass + gDirt) * 1.8, 0.0, 1.0);
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

        // Red/green encode signed X/Z direction and blue encodes the routed
        // drainage. Brightness makes both standing water and velocity visible.
        let directionColour = vec3<f32>(direction * 0.5 + vec2<f32>(0.5), dischargeAmount);
        let speedAmount = 1.0 - exp(-speed * 0.18);
        let signal = clamp(max(waterAmount, max(dischargeAmount, speedAmount)), 0.0, 1.0);
        let debugColour = mix(vec3<f32>(0.015, 0.018, 0.025), directionColour, signal);
        albedo = mix(albedo, debugColour, flowDomain);
    }
    else
    {
        let waterline = 1.0 - smoothHermite(stage.sea_level + 0.05,
                                            stage.sea_level + 1.15 + (soilLarge - 0.5) * 0.45,
                                            height);
        let shoreWet = waterline * (gSand + gGravel + gRock);
        // A river wets its bed and the bank a hand's breadth above the
        // water, splashed higher beside whitewater.
        let riverWet = 1.0 - smoothHermite(0.0, 0.35 + 0.9*input.frag_river.z,
                                           input.frag_river.x);
        // A bar the river last covered, and the stones beside whitewater,
        // stay damp and stained.
        let moisture = clamp(waterAmount * 0.52 + channel * 0.25 + shoreWet * 0.65
                             + riverWet * 0.72 + pointBar * 0.6 + stoneBank * 0.5, 0.0, 0.72)
                     * (1.0 - gSnow);
        let damp = albedo * vec3<f32>(0.55, 0.57, 0.56);
        albedo = mix(albedo, damp, moisture);
        // Under the water the bed is filmed with algae and settled silt.
        let submergedBed = (1.0 - smoothHermite(-0.35, 0.0, input.frag_river.x)) * (1.0 - gSnow);
        albedo = mix(albedo, albedo * vec3<f32>(0.50, 0.55, 0.36), submergedBed);
        // Damp earth stays rough; only pooled water approaches a low
        // roughness. Write the final value used by the G-buffer (snow glints
        // have already modified outRough above). Stone darkens when damp but
        // glazes only at the waterline: the solver's working water in the
        // channels is never drawn, so a wet sheen on channel gravel and rock
        // reads as polish rather than as water.
        let wetness = moisture / 0.72;
        let stony = clamp(gGravel + gRock, 0.0, 1.0);
        let glaze = wetness * mix(1.0, max(waterline, riverWet), stony);
        let wetRoughness = mix(0.70, 0.42, waterAmount);
        outRough = mix(outRough, min(outRough, wetRoughness), glaze);
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

@fragment
fn fs_main(input: FsInput) -> FsOutput
{
    var unused_habitat = vec4<f32>(0.0);
    return shadeTerrain(input, false, &unused_habitat);
}

// MRT capture: RGBA32F terrain world height, grass suitability and geometric
// normal XZ; RGBA16F final terrain albedo in linear light. Both values come
// from one evaluation of the same material blend as the visible G-buffer.
// The consumer reconstructs upward normal Y. Capturing the actual terrain
// meshes preserves their erosion, reveal and morph state.
// The capture is kept across frames until its inputs change (HabitatKey in
// terrain_node.rs). Before reading another global or stage value here, make
// sure the key covers it, or a stale capture will outlive a change to it.
@fragment
fn fs_grass_habitat(input: FsInput) -> GrassHabitatOutput
{
    var habitat_data = vec4<f32>(0.0);
    let shaded = shadeTerrain(input, true, &habitat_data);
    // g_albedo is sqrt-encoded into the Rgba8Unorm G-buffer with a 2.5
    // headroom constant; invert that encoding before the linear 16-bit write.
    let linear_albedo = shaded.g_albedo.rgb * shaded.g_albedo.rgb * 2.5;
    return GrassHabitatOutput(habitat_data, vec4<f32>(linear_albedo, 1.0));
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
//   snowline_altitude               f32        @276   persistent snowline above sea level
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
