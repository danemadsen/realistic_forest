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
//    overlapping ground cover and rock exposure. Snow forms a continuous pack
//    above its climate line and sheds from steep rock faces.

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
@group(1) @binding(17) var<storage, read> river_grid: array<u32>;      // river lookup grid
@group(1) @binding(18) var<storage, read> river_segments: array<RiverSegment>; // river carve segments

// Fragment-stage inputs: terrain-vs's eight varyings, declaration order
// = @location order (see PORT NOTES 6). Naming: fragPositionView etc.
struct FsInput {
    @location(0) frag_position_view: vec3<f32>,   // fragPositionView
    @location(1) frag_normal_view: vec3<f32>,     // fragNormalView
    @location(2) frag_world_position: vec3<f32>,  // fragWorldPosition
    @location(3) frag_world_normal: vec3<f32>,    // fragWorldNormal
    @location(4) frag_erosion_delta: f32,         // fragErosionDelta
    @location(5) frag_material_normal: vec3<f32>, // fixed-scale world slope
    @location(6) frag_base_height: f32, // placement stays fixed when snow compacts
    @location(7) frag_snow_compaction: f32,
    // The nearest river or lake: metres past its waterline (negative under
    // water), side of the bend (+ outside, - inside), whitewater, flow speed.
    @location(8) frag_river: vec4<f32>,
    // How far up the nearest river's or lake's shore the ground lies
    // (riverShoreRun, terrain-vs): negative under water, 40 near none.
    @location(9) frag_river_shore: f32,
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

// BEGIN SHARED SNOW COVERAGE
// A continuous pack above the climate snowline. Aspect moves only the broad
// transition, while steep walls shed snow. Geometry and material use this
// same field so the raised snow surface always matches its white coverage.
fn snowCoverage(height: f32, normal: vec3<f32>) -> f32
{
    let meltSun = normalize(vec3<f32>(-0.22, 0.62, -0.76));
    let aspectShift = (0.62 - max(dot(normal, meltSun), 0.0)) * 12.0;
    let snowLine = smoothHermite(stage.snowline_altitude - 13.0,
                                 stage.snowline_altitude + 13.0,
                                 height - stage.sea_level + aspectShift);
    let steepness = length(normal.xz) / max(normal.y, 0.001);
    return snowLine * (1.0 - smoothHermite(0.80, 1.45, steepness));
}
// END SHARED SNOW COVERAGE

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
// segment lists the cells point into, then the lake records. A cell's words
// are the offset of its list, the list's length, and the offset of its lake
// record (RIVER_NO_LAKE_RECORD if no lake reaches it). A record holds the
// lakes' surface at the corners of the cell's 8 x 8 lake cells: the highest
// corner's height as f32 bits, then each corner's depth under it in 2 mm
// steps, two u16 to a word (0xffff where no lake reaches), 9 corners
// to a row. Between corners the surface is interpolated over the two
// triangles a lake's sheet draws each cell as, split from corner (0, 0) to
// corner (1, 1) (carve.rs LakeRecord). A zero resolution means there are
// no rivers.
//
// A lake has no channel: under its sheet the surface is the drawn water
// itself, so ground below it lies under the water; past the sheet it runs on
// under the ground and sinks away (network.rs lake_surface), and
// riverBankAt measures the shore from it in a river bank's terms.
//
// Each segment bounds the ground from above (the channel bed, then a bank
// cone that steepens away from the water) and from below (a low levee that
// keeps the water in its channel). The upper bounds combine by minimum and
// the lower ones by maximum, so confluences open into each other. Everything
// that shapes the channel is given at both ends and interpolated along the
// segment, so neighbours carve the same ground where they meet, and a segment
// between two others has a round cap at either end whose water is the reach
// beside's (cap_slope: how fast the water rises up the river behind its start
// and falls down the reach after its end), so nothing switches on along a line and
// a cap never cuts below the bed or holds a levee over the bank beside it. A
// segment no reach comes before (cap_slope.x below zero) starts flat.
// Crossing banks and their shallow shoreline strip round together once;
// a bounded shelf opens at junctions and deep beds keep their exact profile.
struct RiverSegment {
    a: vec2<f32>,
    b: vec2<f32>,
    water: vec2<f32>,
    half_width: vec2<f32>,
    depth: vec2<f32>,
    speed: vec2<f32>,
    bank: vec2<f32>,
    skew: vec2<f32>,
    levee: vec2<f32>,
    cap_slope: vec2<f32>,
    // Whitewater at the start and end, interpolated like the rest.
    turbulence: vec2<f32>,
    // The level of the still water nearest along the river from each end,
    // or RIVER_NO_LAKE: where the channel's water stands at or over it, its
    // levee never builds ground out of the lake's water (riverHeld).
    still: vec2<f32>,
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
    // The surface of a lake reaching here, or RIVER_NO_LAKE.
    lake: f32,
    // How much of the levee here holds water standing at or over the still
    // water of its lake, 0 to 1, blended over the segments (RiverLeveeBlend).
    perched: f32,
    // The level of the still water those perched levees' channels run into
    // or out of, blended as perched is, or RIVER_NO_LAKE.
    still: f32,
};

const RIVER_NONE: f32 = 1.0e30;
const RIVER_NO_LAKE: f32 = -1.0e30;
const RIVER_GRID_CELL_WORDS: u32 = 3u;
const RIVER_NO_LAKE_RECORD: u32 = 0xffffffffu;
const RIVER_LAKE_CELLS_ACROSS: u32 = 8u;
const RIVER_LAKE_CORNERS_ACROSS: u32 = 9u;
const RIVER_LAKE_DEPTH_UNIT: f32 = 0.002;
const RIVER_LAKE_NO_CORNER: u32 = 0xffffu;
// Metres of shore per metre of rise above a lake's water.
const RIVER_LAKE_SHORE_RUN: f32 = 6.0;
// The steepest face, rise over run, a levee meets a lake's water in
// (LEVEE_LAKE_FACE), and how far under a lake's level a river's water may
// stand and still be the lake's own, and is for certain (UNDER_LAKE).
const RIVER_LEVEE_LAKE_FACE: f32 = 1.0;
const RIVER_UNDER_LAKE: vec2<f32> = vec2<f32>(0.2, 0.05);
const RIVER_BANK_REACH: f32 = 12.0;
const RIVER_BANK_CURVE: f32 = 0.16;
const RIVER_LEVEE_OUTER_SLOPE: f32 = 0.15;
// Height over which the carve's creases are rounded (CARVE_ROUNDING).
const RIVER_CARVE_ROUNDING: f32 = 0.8;
// Height difference over which intersecting bank cones are rounded.
const RIVER_BANK_UNION_ROUNDING: f32 = 2.8;
const RIVER_BANK_UNION_SHORE_BLEND: f32 = 0.75;
const RIVER_BANK_UNION_INTRUSION: f32 = 0.6;
const RIVER_MAX_CANDIDATES: u32 = 64u;
// How far behind the nearest channel, in riverScore, another channel's water
// still counts: each weighs exp(-behind/RIVER_OWNER_BLEND) (OWNER_BLEND).
const RIVER_OWNER_BLEND: f32 = 0.25;
// How far under the highest levee at a point, metres, another levee still
// counts toward whether the levee there holds water over a lake (LEVEE_BLEND).
const RIVER_LEVEE_BLEND: f32 = 0.1;

fn riverNone() -> RiverEnvelope
{
    return RiverEnvelope(RIVER_NONE, -RIVER_NONE, RIVER_NONE, -RIVER_NONE,
                         vec2<f32>(0.0), 0.0, 0.0, 0.0, RIVER_NO_LAKE, 0.0, RIVER_NO_LAKE);
}

// How far `water` stands at or over the still water of its lake, 0 to 1
// (carve.rs perched, UNDER_LAKE).
fn riverPerched(water: f32, still: vec2<f32>, t: f32) -> f32
{
    if (min(still.x, still.y) <= RIVER_NO_LAKE) { return 0.0; }
    let level = mix(still.x, still.y, t);
    return smoothstep(level - RIVER_UNDER_LAKE.x, level - RIVER_UNDER_LAKE.y, water);
}

fn riverSegmentEnvelope(segment: RiverSegment, p: vec2<f32>) -> RiverEnvelope
{
    let ab = segment.b - segment.a;
    let ap = p - segment.a;
    let lengthSquared = max(dot(ab, ab), 1e-6);
    let segmentLength = sqrt(lengthSquared);
    let along = dot(ap, ab)/lengthSquared;
    // Round caps carry the channel on round each node, their water the
    // reach beside's; a segment no reach comes before starts flat.
    let behind = max(-along, 0.0)*segmentLength;
    if (behind > 0.0 && segment.cap_slope.x < 0.0) { return riverNone(); }
    let t = clamp(along, 0.0, 1.0);
    let beyond = max(along - 1.0, 0.0)*segmentLength;
    let offset = ap - ab*t;
    let centreDistance = length(offset);
    let halfWidth = max(mix(segment.half_width.x, segment.half_width.y, t), 0.05);
    let pastBank = centreDistance - halfWidth;
    // Low junction banks spread their shoulders over a broader run.
    let bankSlope = mix(segment.bank.x, segment.bank.y, t);
    let bankRun = 2.0 - smoothstep(0.2, 0.55, bankSlope);
    let bankReach = RIVER_BANK_REACH*bankRun;
    if (pastBank >= bankReach) { return riverNone(); }
    let water = mix(segment.water.x, segment.water.y, t) + max(segment.cap_slope.x, 0.0)*behind;
    let depth = mix(segment.depth.x, segment.depth.y, t);
    let direction = ab/segmentLength;
    // Signed distance from the centreline's line, + on the left of the flow,
    // and which way across the flow the point lies: a side beside the
    // segment, turning smoothly from one to the other round a cap.
    let lateral = direction.x*ap.y - direction.y*ap.x;
    let across = lateral/max(centreDistance, 1e-6);
    let skew = mix(segment.skew.x, segment.skew.y, t);
    let speed = mix(segment.speed.x, segment.speed.y, t);
    var still = RIVER_NO_LAKE;
    if (min(segment.still.x, segment.still.y) > RIVER_NO_LAKE) { still = mix(segment.still.x, segment.still.y, t); }
    var envelope = RiverEnvelope(RIVER_NONE, -RIVER_NONE, pastBank, water,
                                 direction*speed, halfWidth, mix(segment.turbulence.x, segment.turbulence.y, t), skew*across,
                                 RIVER_NO_LAKE, riverPerched(water, segment.still, t), still);
    if (pastBank < 0.0)
    {
        // A flat-bottomed bowl, skewed: zero at both banks, deep right up to
        // them, and deepest toward the outer one.
        let u = centreDistance/halfWidth;
        let profile = (1.0 - u*u*u*u)*(1.0 + skew*lateral/halfWidth);
        envelope.upper = water - depth*profile;
        // The current runs fastest over the thalweg and stalls at the banks.
        let lateralSpeed = pow(max(1.0 - u*u, 0.0), 0.35);
        envelope.velocity = envelope.velocity*(lateralSpeed*1.25);
    }
    else
    {
        // Cut banks stand steep on the outside of a bend; point bars slope
        // gently into the water on the inside.
        let bank = bankSlope*max(1.0 + 0.75*skew*across, 0.3);
        let shoulder = pastBank/bankRun;
        // Fade both constraints before their finite support ends: a high
        // terrace or low hollow must not jump back to its uncarved height.
        let edge = smoothstep(0.5*bankReach, bankReach, pastBank);
        // Saturate the released constraint beyond the terrain's height range
        // before edge-distance precision differences become greatly amplified.
        let release = min(bankReach*edge*edge/max(1.0 - edge, 1e-5), 1.0e4);
        envelope.upper = water + bank*pastBank + RIVER_BANK_CURVE*shoulder*shoulder + release;
        let freeboard = 0.1 + 0.25*depth;
        let leveeWidth = 0.8 + 0.3*halfWidth;
        // Past the segment's end its levee falls on as the water after it
        // does, or down a rapid each end would hold a ledge up beside the next.
        let fall = segment.cap_slope.y*beyond;
        // A fading levee retreats gradually instead of switching off as
        // soon as its strength is a little below one.
        let levee = mix(segment.levee.x, segment.levee.y, t);
        if (levee > 0.0)
        {
            let retreat = 1.0 - levee;
            let leveeRelease = (depth + 0.25)*retreat*retreat/max(levee, 1e-5);
            envelope.lower = water - fall + min(bank*pastBank, freeboard)
                           - max(pastBank - leveeWidth, 0.0)*RIVER_LEVEE_OUTER_SLOPE
                           - leveeRelease - release;
        }
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

// The water a point belongs to, blended over every channel near it by how far
// behind the nearest each lies (carve.rs OwnerBlend): where two channels'
// waters meet, the water level, current, width, whitewater and bend hand over
// smoothly instead of switching along a line. An online softmax: one pass.
struct RiverOwnerBlend {
    best: f32,
    weight: f32,
    water: f32,
    velocity: vec2<f32>,
    half_width: f32,
    turbulence: f32,
    bend: f32,
};

fn riverOwnerBlendAdd(blend: ptr<function, RiverOwnerBlend>, next: RiverEnvelope)
{
    if (next.bank_distance >= RIVER_NONE) { return; }
    let score = riverScore(next);
    var k = 1.0;
    if ((*blend).weight == 0.0)
    {
        (*blend).best = score;
    }
    else if (score < (*blend).best)
    {
        // A nearer channel: what was gathered so far falls behind it.
        let fade = exp(-((*blend).best - score)/RIVER_OWNER_BLEND);
        (*blend).weight *= fade;
        (*blend).water *= fade;
        (*blend).velocity *= fade;
        (*blend).half_width *= fade;
        (*blend).turbulence *= fade;
        (*blend).bend *= fade;
        (*blend).best = score;
    }
    else
    {
        k = exp(-(score - (*blend).best)/RIVER_OWNER_BLEND);
    }
    (*blend).weight += k;
    (*blend).water += k*next.water;
    (*blend).velocity += k*next.velocity;
    (*blend).half_width += k*next.half_width;
    (*blend).turbulence += k*next.turbulence;
    (*blend).bend += k*next.bend;
}

fn riverOwnerBlendApply(blend: RiverOwnerBlend, total: ptr<function, RiverEnvelope>)
{
    if (blend.weight <= 0.0) { return; }
    let inverse = 1.0/blend.weight;
    (*total).water = blend.water*inverse;
    (*total).velocity = blend.velocity*inverse;
    (*total).half_width = blend.half_width*inverse;
    (*total).turbulence = blend.turbulence*inverse;
    (*total).bend = blend.bend*inverse;
}

// Whether the levee holding the ground up at a point holds water at or over
// a lake's, blended over the segments by how near each one's levee comes to
// the highest (carve.rs LeveeBlend): an online softmax, so it never switches
// where two channels' levees meet.
struct RiverLeveeBlend {
    best: f32,
    weight: f32,
    perched: f32,
    // The perched levees' still water, summed by weight times perched.
    still: f32,
    still_weight: f32,
};

fn riverLeveeBlendAdd(blend: ptr<function, RiverLeveeBlend>, next: RiverEnvelope)
{
    if (next.lower <= -RIVER_NONE) { return; }
    var k = 1.0;
    if ((*blend).weight == 0.0)
    {
        (*blend).best = next.lower;
    }
    else if (next.lower > (*blend).best)
    {
        // A higher levee: what was gathered so far falls under it.
        let fade = exp(-(next.lower - (*blend).best)/RIVER_LEVEE_BLEND);
        (*blend).weight *= fade;
        (*blend).perched *= fade;
        (*blend).still *= fade;
        (*blend).still_weight *= fade;
        (*blend).best = next.lower;
    }
    else
    {
        k = exp(-((*blend).best - next.lower)/RIVER_LEVEE_BLEND);
    }
    (*blend).weight += k;
    (*blend).perched += k*next.perched;
    if (next.still > RIVER_NO_LAKE)
    {
        (*blend).still += k*next.perched*next.still;
        (*blend).still_weight += k*next.perched;
    }
}

// Round a pair against the unchanged raw upper minimum. The caller takes
// the lowest pair once, so segment count cannot accumulate excavation.
fn riverBankSurfaceSlope(segment: RiverSegment) -> f32
{
    // Fixed conservative waterline slope avoids introducing another cusp
    // as the nearer bank changes at the junction's bisector.
    let start = max(segment.bank.x*max(1.0 - 0.75*abs(segment.skew.x), 0.3), 0.001);
    let end = max(segment.bank.y*max(1.0 - 0.75*abs(segment.skew.y), 0.3), 0.001);
    return min(start, end);
}

fn riverBankUnionUpper(primary: RiverSegment, first: RiverEnvelope,
                      next: RiverSegment, second: RiverEnvelope) -> f32
{
    let nearest_bank = min(first.bank_distance, second.bank_distance);
    if (nearest_bank <= -RIVER_BANK_UNION_SHORE_BLEND ||
        second.bank_distance >= RIVER_NONE) { return first.upper; }
    let primary_direction = primary.b - primary.a;
    let other_direction = next.b - next.a;
    let cross_value = primary_direction.x*other_direction.y - primary_direction.y*other_direction.x;
    let length_product = max(dot(primary_direction, primary_direction)*dot(other_direction, other_direction), 1e-12);
    // Squared sine: independent of tangent orientation, zero for straight
    // overlapping reaches and progressively stronger at a joining branch.
    let turn = clamp(cross_value*cross_value/length_product, 0.0, 1.0);
    let bank_slope = min(riverBankSurfaceSlope(primary), riverBankSurfaceSlope(next));
    let shelf_rounding = 4.0*RIVER_BANK_UNION_INTRUSION*bank_slope;
    let shelf_clearance = first.upper - max(first.water, second.water)
                        + bank_slope*(RIVER_BANK_UNION_INTRUSION - nearest_bank);
    let bed_fade = smoothstep(-RIVER_BANK_UNION_SHORE_BLEND, 0.0, nearest_bank);
    let rounding = min(min(RIVER_BANK_UNION_ROUNDING, shelf_rounding), 4.0*max(shelf_clearance, 0.0))*turn*bed_fade;
    if (rounding <= 0.0) { return first.upper; }
    return riverSmoothMin(first.upper, second.upper, rounding);
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
    let count = min(river_grid[entry + 1u], RIVER_MAX_CANDIDATES);
    var lake = RIVER_NO_LAKE;
    let record = river_grid[entry + 2u];
    if (record != RIVER_NO_LAKE_RECORD)
    {
        let local = (p - origin)/(bitcast<f32>(river_grid[2])/f32(RIVER_LAKE_CELLS_ACROSS))
                  - cell*f32(RIVER_LAKE_CELLS_ACROSS);
        lake = riverLakeSurface(record, local);
    }
    var primary = 0xffffffffu;
    var blend = RiverOwnerBlend(0.0, 0.0, 0.0, vec2<f32>(0.0), 0.0, 0.0, 0.0);
    var levees = RiverLeveeBlend(0.0, 0.0, 0.0, 0.0, 0.0);
    for (var i = 0u; i < count; i += 1u)
    {
        let index = river_grid[offset + i];
        let envelope = riverSegmentEnvelope(river_segments[index], p);
        if (envelope.upper < total.upper) { primary = index; }
        riverCombine(&total, envelope);
        riverOwnerBlendAdd(&blend, envelope);
        riverLeveeBlendAdd(&levees, envelope);
    }
    riverOwnerBlendApply(blend, &total);
    if (levees.weight > 0.0) { total.perched = levees.perched/levees.weight; }
    if (levees.still_weight > 0.0) { total.still = levees.still/levees.still_weight; }
    // Continue the bank rounding through its shallow shoreline strip, then
    // fade it out smoothly before the undisturbed channel bed.
    if (primary != 0xffffffffu && total.bank_distance > -RIVER_BANK_UNION_SHORE_BLEND)
    {
        let raw_bank_distance = total.bank_distance;
        let first_segment = river_segments[primary];
        let first = riverSegmentEnvelope(first_segment, p);
        if (first.bank_distance > -RIVER_BANK_UNION_SHORE_BLEND)
        {
            for (var i = 0u; i < count; i += 1u)
            {
                let next = river_segments[river_grid[offset + i]];
                let second = riverSegmentEnvelope(next, p);
                let rounded_upper = riverBankUnionUpper(first_segment, first, next, second);
                if (rounded_upper < total.upper)
                {
                    total.upper = rounded_upper;
                    let bank_slope = min(riverBankSurfaceSlope(first_segment), riverBankSurfaceSlope(next));
                    let shelf_water = max(first.water, second.water);
                    if (raw_bank_distance >= 0.0 && first.bank_distance >= 0.0 && second.bank_distance >= 0.0 && rounded_upper < shelf_water)
                    {
                        total.water = max(total.water, shelf_water);
                    }
                    let shelf_distance = max((rounded_upper - total.water)/bank_slope, -RIVER_BANK_UNION_INTRUSION);
                    total.bank_distance = min(total.bank_distance, shelf_distance);
                }
            }
        }
    }
    total.lake = lake;
    return total;
}

// The surface at corner `k` of a lake record, or RIVER_NO_LAKE.
fn riverLakeCorner(record: u32, k: u32) -> f32
{
    let depth = (river_grid[record + 1u + k/2u] >> (16u*(k % 2u))) & 0xffffu;
    if (depth == RIVER_LAKE_NO_CORNER) { return RIVER_NO_LAKE; }
    return bitcast<f32>(river_grid[record]) - f32(depth)*RIVER_LAKE_DEPTH_UNIT;
}

// A lake record's surface at `local`, in lake cells from the record's corner.
fn riverLakeSurface(record: u32, local: vec2<f32>) -> f32
{
    let i = clamp(floor(local), vec2<f32>(0.0), vec2<f32>(f32(RIVER_LAKE_CELLS_ACROSS - 1u)));
    let uv = local - i;
    let k = u32(i.y)*RIVER_LAKE_CORNERS_ACROSS + u32(i.x);
    let h00 = riverLakeCorner(record, k);
    let h11 = riverLakeCorner(record, k + RIVER_LAKE_CORNERS_ACROSS + 1u);
    var h = 0.0;
    var other = 0.0;
    if (uv.y >= uv.x)
    {
        other = riverLakeCorner(record, k + RIVER_LAKE_CORNERS_ACROSS);
        h = h00 + (h11 - other)*uv.x + (other - h00)*uv.y;
    }
    else
    {
        other = riverLakeCorner(record, k + 1u);
        h = h00 + (other - h00)*uv.x + (h11 - other)*uv.y;
    }
    if (min(min(h00, h11), other) <= RIVER_NO_LAKE) { return RIVER_NO_LAKE; }
    return h;
}

// Metres past the nearest waterline, a river's or a lake's, for ground at
// `height`: negative under water.
fn riverBankAt(envelope: RiverEnvelope, height: f32) -> f32
{
    return min(envelope.bank_distance, (height - envelope.lake)*RIVER_LAKE_SHORE_RUN);
}

// Polynomial smooth minimum and maximum (smooth_min and smooth_max in
// carve.rs): the corner rounded over a difference of `k`.
fn riverSmoothMin(a: f32, b: f32, k: f32) -> f32
{
    let h = max(k - abs(a - b), 0.0)/k;
    return min(a, b) - h*h*k*0.25;
}

fn riverSmoothMax(a: f32, b: f32, k: f32) -> f32
{
    let h = max(k - abs(a - b), 0.0)/k;
    return max(a, b) + h*h*k*0.25;
}

// How high the levees hold up ground standing at `height` (Envelope::held):
// a channel whose water stands at or over its lake's never builds ground out
// of the lake's water (its surface, or its level where the sheet's edge is
// drawn sunk lower), and its levee comes back up the shore in a face no
// steeper than RIVER_LEVEE_LAKE_FACE over the run from the waterline.
fn riverHeld(envelope: RiverEnvelope, height: f32) -> f32
{
    if (envelope.lake <= RIVER_NO_LAKE || envelope.perched <= 0.0) { return envelope.lower; }
    let lake = max(envelope.lake, envelope.still);
    let face = lake - RIVER_CARVE_ROUNDING
             + (height - lake)*RIVER_LAKE_SHORE_RUN*RIVER_LEVEE_LAKE_FACE;
    return envelope.lower - envelope.perched*max(envelope.lower - face, 0.0);
}

// The ground with the channels cut into it and their banks held up, the
// creases where the carve meets the natural ground rounded off: a bank's top
// curves over into the land above it, and a levee's foot into the land below.
fn riverClamp(envelope: RiverEnvelope, height: f32) -> f32
{
    return riverSmoothMin(riverSmoothMax(height, riverHeld(envelope, height), RIVER_CARVE_ROUNDING),
                          envelope.upper, RIVER_CARVE_ROUNDING);
}
// END SHARED RIVERS

// BEGIN SHORE RUN
// How far up the nearest river's or lake's shore ground at `height` lies:
// the coordinate the terrain's bank band is measured in (terrain-fs.wgsl).
// terrain-vs.wgsl hands each vertex's value on and terrain-fs.wgsl measures
// it again exactly near the water, so both paste this block verbatim (a
// test keeps the copies identical).
//
// Where the shore is gentle the run is the horizontal distance from the
// waterline; where it climbs steeply, six metres per metre of rise
// (RIVER_LAKE_SHORE_RUN, a lake's own shore unit), so a band measured in it
// spreads over a floodplain or a flat pond margin and narrows on a cut bank
// or a valley wall. A river's bank distance is its horizontal run. A lake
// records only its level, so a lake's run is the rise over the slope of the
// ground (`groundNormal`, the prefiltered material normal), as if its shore
// were a plane, never taken gentler than 1 in 100. The run counts in the
// water's own size: a creek's narrow margin reads as farther (its band is
// narrower), a river's or a pond's broad one as nearer. Negative under
// water; ground near no water reads 40, as frag_river's bank does.
const RIVER_SHORE_LAKE_SCALE: f32 = 0.9;

// Smooth sediment handover where a channel crosses a pond's margin. Both
// stages use the same blend so the silt/gravel boundary does not move with LOD.
fn riverLakeShare(envelope: RiverEnvelope, height: f32) -> f32
{
    let lakeBank = (height - envelope.lake)*RIVER_LAKE_SHORE_RUN;
    return smoothstep(-2.0, 2.0, envelope.bank_distance - lakeBank);
}

fn riverShoreRun(envelope: RiverEnvelope, height: f32, groundNormal: vec3<f32>) -> f32
{
    let tangent = sqrt(max(1.0 - groundNormal.y*groundNormal.y, 0.0))/max(groundNormal.y, 0.05);
    let riverScale = clamp(0.3 + 0.28*envelope.half_width, 0.35, 1.3);
    let riverRun = max(envelope.bank_distance,
                       (height - envelope.water)*RIVER_LAKE_SHORE_RUN)/riverScale;
    let lakeRise = height - envelope.lake;
    let lakeRun = max(lakeRise/max(tangent, 0.01), lakeRise*RIVER_LAKE_SHORE_RUN)
                / RIVER_SHORE_LAKE_SCALE;
    // Round the meeting of the pond margin and channel bank. A hard minimum
    // leaves a crease through the material bands at a flared inlet/outlet.
    // Bound both distances first so absent water fields stay numerically safe.
    return clamp(riverSmoothMin(clamp(riverRun, -40.0, 40.0),
                                clamp(lakeRun, -40.0, 40.0), 1.5), -40.0, 40.0);
}

// Flow belongs to one channel at a confluence, but the shore belongs to all
// of them. Measure each bank before combining: choosing the flow owner's
// width and level first left triangular material wedges where ownership
// changed between a narrow tributary and a wide receiving river.
fn riverShoreRunAt(envelope: RiverEnvelope, p: vec2<f32>, height: f32,
                   groundNormal: vec3<f32>) -> f32
{
    var run = riverShoreRun(envelope, height, groundNormal);
    let resolution = river_grid[3];
    if (resolution == 0u) { return run; }
    let origin = vec2<f32>(bitcast<f32>(river_grid[0]), bitcast<f32>(river_grid[1]));
    let cell = floor((p - origin)/bitcast<f32>(river_grid[2]));
    if (any(cell < vec2<f32>(0.0)) || any(cell >= vec2<f32>(f32(resolution)))) { return run; }
    let entry = 8u + (u32(cell.y)*resolution + u32(cell.x))*RIVER_GRID_CELL_WORDS;
    let offset = river_grid[entry];
    let count = min(river_grid[entry + 1u], RIVER_MAX_CANDIDATES);
    for (var i = 0u; i < count; i += 1u)
    {
        let bank = riverSegmentEnvelope(river_segments[river_grid[offset + i]], p);
        run = min(run, riverShoreRun(bank, height, groundNormal));
    }
    return run;
}
// END SHORE RUN

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
    if (base == SNOW_BASE)
    {
        // Fresh powder has a consistent white body. Keep a little scan grain
        // for close lighting without repeating its broad grey patches.
        sampled_albedo = mix(vec3<f32>(0.67), sampled_albedo, 0.12);
    }
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
    let height = input.frag_base_height;
    let materialNormal = normalize(input.frag_material_normal);
    let slope = 1.0 - clamp(materialNormal.y, 0.0, 1.0);
    let worldXZ = frag_world_position.xz;
    // The river or lake here. The vertex stage's value is interpolated
    // across the clipmap's triangles, which beyond the first ring are wider
    // than a creek and would smear its bed and banks across them; wherever
    // that value says water may lie within a triangle's span, look it up
    // exactly. Elsewhere (nearly everywhere) the cheap value stands.
    var river = input.frag_river;
    // How far up its shore (riverShoreRun), for the bank band below.
    var shoreRun = input.frag_river_shore;
    // The surface of the water nearest, where it is looked up exactly.
    var waterLevel = 1.0e6;
    // Keep the entire material fringe inside the exact lookup. Handing back
    // at 4 m cut through the soil/turf blend and made a different outline for
    // every clipmap ring, most visibly beside a broad outlet.
    let exactReach = 16.0 + 1.5*max(stage.spacing, stage.next_spacing);
    if (river.x < exactReach)
    {
        let exact = riverEnvelope(worldXZ);
        // Measured from the ground under any snow, as the vertex stage does.
        let bank = riverBankAt(exact, height);
        // Sediment grades from a pond's settled silt into its outlet's gravel.
        // Choosing either lake or channel at one distance boundary produced a
        // hard material seam even where their water surfaces met smoothly.
        let lakeShare = riverLakeShare(exact, height);
        waterLevel = max(exact.water, exact.lake);
        let looked = vec4<f32>(clamp(bank, -40.0, 40.0),
                               exact.bend*(1.0 - lakeShare),
                               exact.turbulence*(1.0 - lakeShare),
                               length(exact.velocity)*(1.0 - lakeShare));
        // Hand over to the vertices' values without a seam.
        let handover = smoothHermite(exactReach - 3.0, exactReach, river.x);
        // Past the first ring a triangle is wider than a creek and its height
        // is a chord across the carved groove; the carve's bounds put it back
        // on the bank (a no-op at the already-clamped vertices).
        let shoreGround = riverClamp(exact, height);
        shoreRun = mix(riverShoreRunAt(exact, worldXZ, shoreGround, materialNormal), shoreRun, handover);
        river = mix(looked, river, handover);
    }
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
    // Keep the pack continuous across cold ground. Coverage follows the same
    // climate and slope field as the raised snow geometry; erosion, drift
    // noise and melt channels no longer punch isolated holes into it.
    let fSnow = snowCoverage(height, materialNormal);
    let snowLine = fSnow;

    // Rivers, creeks and lakes (src/rivers). The bed under the water sorts
    // by the power of the current over it: lake beds, pools and the slack
    // water along the banks settle silt and mud, runs and riffles keep a
    // gravel bed, and whitewater scours it to cobble and bedrock.
    //
    // Out of the water every river and lake is edged with bare earth, as the
    // sea is with its beach: dark mud where the water laps, damp soil above
    // it that floods, frost and trampling keep open, then turf closing in
    // over a few metres through tufts, bays and tongues. The band is
    // measured in shoreRun (riverShoreRun), so it spreads over a floodplain
    // or a gentle pond margin and narrows on a steep bank, and its edge
    // wanders at three scales as the beach's does with its drift: stretches
    // tens of metres long where it runs wider or narrower (the beach's own
    // drift field), bays and tongues of turf a few metres across, and tufts
    // a metre across. The bank adds its own story: on the outside of a bend
    // the current undercuts a face of bare soil and roots up to its lip; on
    // the inside it leaves a bar, gravel where the stream runs quick and silt
    // where it is slow; beside whitewater the margin is stone. By the sea the
    // beach's sand still takes the ground, beds and banks alike.
    let riverBank = river.x + (soilEdge - 0.5)*0.35 + (soilSmall - 0.5)*0.25;
    let riverBend = river.y;
    let riverTurbulence = river.z;
    let riverPower = river.w*(1.0 + 1.5*riverTurbulence);
    // The carve's horizontal footprint can continue under receiving water.
    // Its footprint alone must not paint exposed ground as a river bed.
    let submerged = smoothHermite(-0.08, 0.12, waterLevel - height);
    let inRiver = (1.0 - smoothHermite(-0.30, 0.05, riverBank))*submerged;
    let riverRock = smoothHermite(0.45, 0.85, riverTurbulence);
    // The speeds follow Manning's equation (flow_speed, src/rivers/network.rs):
    // ~0.5 m/s through a lowland reach, ~1.2 through a moderate one and ~1.7
    // down a steep one, a quarter faster over the thalweg and slack toward
    // the banks. Silt and organic mud settle where the current over the bed
    // stays under ~0.3 m/s; above ~0.8 only gravel stays put.
    let riverSilt = (1.0 - smoothHermite(0.30, 0.80, riverPower + (soilSmall - 0.5)*0.45))
                  * (1.0 - riverRock);
    let riverGravel = clamp(1.0 - riverRock - riverSilt, 0.0, 1.0);
    // The bank band: bare from the water up, with turf closing over it
    // between runs of ~2 and ~6, and never on the margin the water laps.
    let bandWidth = mix(0.80, 1.15, grassDrift);
    let bandRun = shoreRun/bandWidth + (soilSmall - 0.5)*2.4 + (soilEdge - 0.5)*1.6;
    let lappedMargin = 1.0 - smoothHermite(0.4, 1.2, shoreRun);
    let shoreBare = max(1.0 - smoothHermite(2.0, 6.0, bandRun), lappedMargin)*(1.0 - inRiver);
    // The ground the shore governs: its band and the lush turf beyond it,
    // ending well inside the 12 m a river's bank field reaches and mostly
    // inside the 8-16 m a lake's shore record does.
    let shoreZone = 1.0 - smoothHermite(5.0, 9.0, shoreRun + (soilSmall - 0.5)*2.0);
    // The local bank: how steeply this very ground rises from the water.
    let bankSteep = smoothHermite(0.10, 0.35, 1.0 - normalWorld.y);
    let cutBank = bankSteep*(0.45 + 0.55*smoothHermite(0.0, 0.35, riverBend))
                * (1.0 - smoothHermite(2.5, 4.5 + 2.0*soilSmall, shoreRun))*(1.0 - inRiver);
    let pointBar = smoothHermite(0.04, 0.30, -riverBend)*(1.0 - bankSteep)
                 * (1.0 - smoothHermite(0.8, 1.6 + 1.6*soilLarge, shoreRun))*(1.0 - inRiver)
                 * smoothHermite(0.30, 0.60, soilSmall + (soilEdge - 0.5)*0.5);
    // A quick stream's bar is gravel, a slow lowland one's silt.
    let barGravel = smoothHermite(0.04, 0.20, riverTurbulence);
    // Measured across the ground, not up it: a whitewater creek's banks
    // climb steeply, and its stones line them a metre or so either way.
    let stoneBank = riverRock*(1.0 - smoothHermite(0.15, 0.9 + 0.6*soilEdge, riverBank))*(1.0 - inRiver);
    // Sand is the sea's: inland, beds and banks are silt, earth and stone.
    // Where a river crosses the beach to the sea, its water down at the sea's
    // level, its bed is the beach's sand as the sea's is, so the two waters
    // meet over one bed; a low river or pond inland keeps its own bed.
    let estuary = 1.0 - smoothHermite(stage.sea_level + 0.1, stage.sea_level + 4.0, waterLevel);
    // Carry the same sediment mixture across the waterline and up the shore.
    // Inland silt fades through the bank's soil before it meets coastal sand;
    // it must not replace that sand only inside the narrow wetted channel.
    // The broad damp fringe keeps its existing coastal sand and turf. Only
    // the bed and the actual bare margin receive fresh river sediment.
    let freshwaterSediment = (1.0 - estuary)*max(inRiver, shoreBare*0.35);
    let fSandRiver = (1.0 - fGrass)*(1.0 - freshwaterSediment);
    let coastalSediment = (1.0 - fGrass)*estuary*max(inRiver, shoreBare);
    let riverBed = inRiver*(1.0 - fSandRiver);
    let shoreSoil = max(max(shoreBare, cutBank), pointBar*(1.0 - barGravel));
    let calmShore = shoreSoil*(1.0 - riverRock);
    // Along calm water the erosion's gravel and rock give way to the band (a
    // steep face still stands as rock); whitewater keeps its stony banks.
    let fGravelRiver = max(fGravel*(1.0 - 0.9*calmShore)*(1.0 - inRiver*riverSilt),
                           riverBed*riverGravel + pointBar*barGravel*0.75 + stoneBank*0.5)
                      * (1.0 - 0.85*coastalSediment);
    let fRockRiver = max(fRock*(1.0 - 0.9*calmShore*(1.0 - steepFace)),
                         riverBed*riverRock + stoneBank*0.6);
    let fSnowRiver = fSnow*(1.0 - max(inRiver, max(pointBar, stoneBank)*0.7));
    // Under the water whatever is not gravel or rock is silt (never turf); on
    // the shore the band's bare earth, a cut bank's face and a slow stream's
    // bar displace the turf, and the background soil patches thin near it.
    // Add alluvium to the local ground mixture. Suppressing background soil
    // throughout shoreZone made a green halo around a separate brown strip.
    let grassSoilRiver = 1.0 - (1.0 - grassSoilBlend)*(1.0 - shoreSoil)*(1.0 - inRiver);

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
                         // The turf just beyond a river's or lake's bare
                         // margin grows lush and dark green on the moist soil.
                         - shoreZone * (1.0 - shoreBare) * 0.40,
                         0.0, 1.0);
    let grassTint = mix(vec3<f32>(0.31, 0.39, 0.27), vec3<f32>(0.48, 0.40, 0.28), grassDryness)
                  * mix(0.94, 1.06, groundGrowth);
    // Neutralise the powder scan's blue cast while leaving headroom for its
    // grain and wind relief in sunlight. Shadow colour comes from sky lighting.
    let snowTint = vec3<f32>(1.03, 1.02, 1.00);

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
        // A river's or lake's dry upper margin is paler alluvial silt, so the
        // band grades from dark wet mud to pale dry silt to turf, as the
        // beach grades from wet sand to dry.
        mix(mix(vec3<f32>(0.40, 0.33, 0.25), vec3<f32>(0.35, 0.33, 0.30),
                clamp(rockFringe + alpine * 0.6, 0.0, 1.0)),
            vec3<f32>(0.44, 0.38, 0.30), shoreZone * shoreBare * 0.65),
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
    // Grass keeps its scanned blade gaps and thatch shadows. Snow retains
    // only shallow pore contrast so undisturbed powder stays uniformly white;
    // mesh relief and SSAO supply the shadows around compressed trails.
    var aoRetains = array<f32, 6>(0.12, 0.85, 1.0,
                                  0.90, 1.0, 1.0);
    var normalMuls = array<f32, 6>(0.35, 1.0, 1.0,
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
        let relief = (surfaceAO[g] - 0.65) * select(0.22, 0.04, g == 0) * contactDetail;
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
        // Sand, alluvial soil and gravel interleave across bars and outlets.
        // Preserve their weighted mixture instead of sharpening it into a
        // different texture patch at the pond/river or river/beach boundary.
        let sedimentPair = g >= 2 && g <= 4 && highestGroup >= 2 && highestGroup <= 4;
        let width = 0.34 + edgeWidths[g]
                  + select(0.0, 0.60, groundPair)
                  + select(0.0, 0.50*shoreZone, sedimentPair);
        let sharpened_contact = smoothHermite(highestScore - width, highestScore, scores[g]);
        // Littoral sand and river alluvium share a continuous mixture at the
        // mouth. Picking a dominant scan here made sand/soil/turf boundaries
        // snap even when their placement weights already changed gradually.
        let shoreline_mix = select(0.0, shoreZone*(1.0 - fSnow), g > 0);
        let contact = mix(sharpened_contact, 1.0, shoreline_mix);
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

    // Packed snow has less airy scattering and a smoother surface. A small
    // tonal change helps the geometric depression read in diffuse light.
    let packedSnow = clamp(input.frag_snow_compaction, 0.0, 1.0) * gSnow;
    albedo *= mix(1.0, 0.92, packedSnow);
    rough = mix(rough, max(0.55, rough - 0.10), packedSnow);

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
        // Nothing roots in a river or a lake, on the mud its water laps or on
        // the bare earth just above it; farther up the bank band the turf
        // factor above keeps blades to its grassy tufts and bays. The capture
        // is only redrawn when its inputs change, so it looks the water up
        // exactly at every texel instead of trusting the clipmap's vertices,
        // which farther out stand wider apart than a creek is wide.
        let water = riverEnvelope(worldXZ);
        let dryBank = smoothHermite(0.25, 0.75, riverBankAt(water, height))
                    * smoothHermite(0.7, 1.6, riverShoreRunAt(water, worldXZ, riverClamp(water, height), materialNormal));
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
        let meltExposure = max(dot(materialNormal, normalize(vec3<f32>(-0.22, 0.62, -0.76))), 0.0);
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
        let kMacroDriftHeight = 0.8;    // noise-height multiplier
        let kMacroDriftStep = 2.5;      // gradient tap separation, m
        let kMacroSastrugiFreq = 0.011; // ~8 m wavelength
        let kMacroSastrugiHeight = 0.3;
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

        // Soft near-field wind grain adds shallow crust lighting while the
        // white body remains even. It fades as its features approach pixel
        // size, with the mip chain filtering the field along the way.
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
            let kGrainSlope = 0.25;
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
            let kCrystalSlope = 0.4;
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
        albedo = albedo * (1.0 + (mix(mottle, mottleForm, 0.6) - 0.5) * 0.035 * gSnow * mottleFade);
        let grainFade = 1.0 - smoothHermite(0.003, 0.08, footprint);
        // Inner fade: the grain domain's texel is ~2 mm, so below ~6 mm/px
        // the fetch sits near LOD 0 and the octave's own texel noise rides
        // straight through as 1-2 px speckle — the "nearest snow grain
        // salt" the artifact hunter measured on the steepest-crop checks.
        // The crystal octave (texel 0.3 mm, deep in its mip chain at that
        // range) owns the boot-range albedo instead.
        let grainInner = smoothHermite(0.0015, 0.006, footprint);
        albedo = albedo * (1.0 + (snowGrainValue - 0.5) * 0.055 * gSnow * grainFade * grainInner);
        // Crystal albedo: boot-range snow varies at the cluster scale —
        // shadowed interstices between hoar clusters read darker than the
        // cluster faces, so the crystal octave swings albedo as well as
        // tilting the normal (its fade is recomputed rather than reused
        // because it lives in this block's scope).
        let crystalFade = 1.0 - smoothHermite(0.0015, 0.02, footprint);
        albedo = albedo * (1.0 + (snowCrystalValue - 0.5) * 0.05 * gSnow * crystalFade);

        // The snow-to-grass biome crossfade interpolates the two albedos
        // straight through an olive half-mix; a real snowline melts through
        // dirty grey-brown snow. Desaturate the overlap band toward its own
        // luminance with a faintly earthen tint.
        let fringe = clamp(min(gSnow, gGrass + gDirt) * 1.8, 0.0, 1.0);
        let fringeLum = dot(albedo, vec3<f32>(0.299, 0.587, 0.114));
        albedo = mix(albedo, vec3<f32>(fringeLum) * vec3<f32>(1.04, 0.99, 0.92), fringe * 0.12);

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
        albedo = mix(albedo, albedo * vec3<f32>(0.52, 0.37, 0.29), 0.12 * sedimentLoad);
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
        let under_sea = 1.0 - smoothHermite(stage.sea_level - 0.12, stage.sea_level + 0.04, height);
        let shoreWet = waterline * (gSand + gGravel + gRock);
        // A river or lake saturates the margin its water laps, splashed
        // higher beside whitewater, and keeps the earth for a metre or two
        // above it damp (the turf there only a little: its blades stand clear
        // of the wet soil). A bar the river last covered, and the stones
        // beside whitewater, stay damp and stained.
        // Spray wets the stones a metre or so up beside whitewater (horizontal
        // metres); elsewhere the lapped margin is measured up the shore.
        let splash = 1.0 - smoothHermite(0.0, 0.05 + 0.9*river.z, river.x);
        let lapped = max(1.0 - smoothHermite(0.0, 0.45 + 0.3*soilEdge, shoreRun), splash);
        let dampShore = (1.0 - smoothHermite(0.8, 2.8 + 1.2*soilSmall, shoreRun))
                      * (1.0 - 0.7*gGrass);
        // Saturated coastal sediment has the same finish on both sides of a
        // mouth. Otherwise the river lookup alone darkens an offshore strip.
        let marine_saturation = under_sea*(gSand + gGravel + gRock)*0.72;
        let moisture = clamp(max(waterAmount * 0.52 + channel * 0.25 + shoreWet * 0.65
                                 + lapped * 0.72 + dampShore * 0.32
                                 + pointBar * 0.5 + stoneBank * 0.5, marine_saturation), 0.0, 0.72)
                     * (1.0 - gSnow);
        let damp = albedo * vec3<f32>(0.55, 0.57, 0.56);
        albedo = mix(albedo, damp, moisture);
        // Under the water the bed is filmed with algae and settled silt; the
        // beach's sand under a river's mouth is scoured clean, as the sea's
        // is, though just as matte under its water.
        let submerged_river = (1.0 - smoothHermite(-0.35, 0.0, river.x))*submerged;
        let submergedBed = max(submerged_river, under_sea)*(1.0 - gSnow);
        let algae = submerged_river*(1.0 - estuary)*(1.0 - 0.85*gSand)*(1.0 - gSnow);
        // Saturated mud at the waterline is darker again than damp earth, and
        // greyer: water fills its pores and its iron is reduced.
        let lappedAbove = lapped * (1.0 - submergedBed) * (1.0 - gSnow);
        albedo = mix(albedo, albedo * vec3<f32>(0.66, 0.70, 0.78), lappedAbove * gDirt);
        albedo = mix(albedo, albedo * vec3<f32>(0.50, 0.55, 0.36), algae);
        // Damp earth stays rough; only pooled water, and the film on mud and
        // stone where the water laps, approach a low roughness. Write the
        // final value used by the G-buffer (snow glints have already modified
        // outRough above). Stone darkens when damp but glazes only at a
        // waterline: the solver's working water in the channels is never
        // drawn, so a wet sheen on channel gravel and rock reads as polish
        // rather than as water.
        let wetness = moisture / 0.72;
        let stony = clamp(gGravel + gRock, 0.0, 1.0);
        let glaze = wetness * mix(1.0, max(waterline, lappedAbove), stony);
        let wetRoughness = mix(0.70, 0.42, waterAmount);
        outRough = mix(outRough, min(outRough, wetRoughness), glaze);
        // Wet mud glistens, but the strip is decimetres wide: fade its gloss
        // before it is a pixel across, or every waterline sparkles.
        let mudGloss = lappedAbove * (1.0 - gGrass)
                     * (1.0 - smoothHermite(0.03, 0.15, footprint));
        outRough = mix(outRough, min(outRough, 0.58), mudGloss);
        // A bed under a river or lake glints at nothing: its water's surface
        // carries the reflections, and the grains beneath scatter as matte as
        // the water lets them. Glazed, the bed caught the sun's lobe as if it
        // lay dry and the river passed it up as a pale sheen.
        outRough = mix(outRough, max(outRough, 0.92), submergedBed);
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
// @location(2) gAlbedo. Fragment inputs @location(0..7): fragPositionView,
// fragNormalView, fragWorldPosition, fragWorldNormal, fragErosionDelta,
// frag_material_normal, frag_river, frag_river_shore (terrain-vs out order).
