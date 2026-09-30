// PORT NOTES (terrain.vs -> terrain-vs.wgsl):
// - Clipmap terrain VERTEX stage. Every GLSL texture() is a vertex-stage
//   sample without implicit derivatives, so they all become
//   textureSampleLevel(..., 0.0): tex0 (base noise), tex1 (erosion atlas) and
//   tex4 (blend mask). GL vertex texture() also reads lod 0, so the filtering
//   result is identical; no samples were hoisted out of branches (explicit
//   levels carry no WGSL uniform-control-flow restriction).
// - texelFetch(texture3, ...) -> textureLoad(tex3, ...). tex3 is the
//   point-filtered RGBA32F tile lookup; bind a non-filtering sampler at
//   binding 11 (the sampler is declared for binding parity but textureLoad
//   does not use it).
// - WGSL has no inverse(): the GLSL `transpose(inverse(mat3(matModel)))` now
//   reads stage.inverse_model (mat4x4<f32>), which the Rust side fills with
//   inverse(matModel). For an affine model matrix (last row 0,0,0,1)
//   mat3(inverse(matModel)) == inverse(mat3(matModel)), so the math is
//   identical. matModel itself is stage.model.
// - matView/matProjection moved into GlobalUniforms (globals.view /
//   globals.projection). globals.projection already maps GL z to the [0,1]
//   NDC range on the CPU; this stage multiplies it exactly as the GLSL did.
// - GLSL out/inout parameters become ptr<function> parameters; the `record`
//   scratch variables are zero-initialised (the GLSL left them uninitialized
//   but every path wrote them before reading, so behaviour is unchanged).
// - The shapeElevation parameter `macro` was renamed macroIn/macroValue
//   because `macro` is a reserved word in WGSL. Single-statement GLSL ifs
//   gained braces. Nothing else moved.
// - No vertical flips were dropped (the source has none) and no
//   gl_FragCoord-style origin fixup applies: this stage reads no screen-space
//   coordinates.

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

// Independent fullscreen heightfield pass. Sharing terrainHeight below
// avoids a second approximation of the terrain for off-screen sun occlusion.
@vertex
fn vs_heightfield(@builtin(vertex_index) vertex: u32) -> @builtin(position) vec4<f32> {
    let corners = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    return vec4<f32>(corners[vertex], 0.0, 1.0);
}

@fragment
fn fs_heightfield(@builtin(position) position: vec4<f32>) -> @location(0) f32 {
    let world_xz = globals.heightfield.xy
        + position.xy * globals.heightfield.w - vec2<f32>(globals.heightfield.z * 0.5);
    return terrainHeight(world_xz);
}

// Every terrain.vs uniform except matView/matProjection. model/inverse_model
// carry matModel (see PORT NOTES for the inverse() substitution); the rest
// are the GLSL uniforms renamed uFooBar -> foo_bar, in declaration order.
struct StageUniforms {
    model: mat4x4<f32>,                    // matModel
    inverse_model: mat4x4<f32>,            // inverse(matModel), Rust-supplied
    clip_origin: vec2<f32>,                // uClipOrigin: world-space centre of the current clipmap level
    spacing: f32,                          // uSpacing: distance between two vertices at this level
    next_spacing: f32,                     // uNextSpacing: spacing of the next coarser level
    morph_start: f32,                      // uMorphStart
    morph_end: f32,                        // uMorphEnd
    sea_level: f32,                        // uSeaLevel
    noise_period: f32,                     // uNoisePeriod
    landform_horizontal_scale: f32,        // uLandformHorizontalScale
    landform_vertical_scale: f32,          // uLandformVerticalScale
    land_profile_curve: f32,               // uLandProfileCurve
    land_profile_reference: f32,           // uLandProfileReference
    land_profile_peak: f32,                // uLandProfilePeak
    waterline_clearance: f32,              // uWaterlineClearance
    waterline_clearance_scale: f32,        // uWaterlineClearanceScale
    waterline_clearance_decay: f32,        // uWaterlineClearanceDecay
    ocean_profile_curve: f32,              // uOceanProfileCurve
    ocean_profile_reference: f32,          // uOceanProfileReference
    ocean_profile_depth: f32,              // uOceanProfileDepth
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
    waterline_push_land: f32,              // uWaterlinePushLand
    waterline_push_sea: f32,               // uWaterlinePushSea
    waterline_push_scale: f32,             // uWaterlinePushScale
};
@group(2) @binding(0) var<uniform> stage: StageUniforms;

@group(1) @binding(0) var tex0: texture_2d<f32>;   // texture0: Tileable, bilinear R32 base-noise texture.
@group(1) @binding(8) var tex0_sampler: sampler;   // linear on the Rust side
@group(1) @binding(1) var tex1: texture_2d<f32>;   // texture1: Surface atlas; R is the eroded-height delta.
@group(1) @binding(9) var tex1_sampler: sampler;   // linear on the Rust side
@group(1) @binding(3) var tex3: texture_2d<f32>;   // texture3: Point-filtered RGBA32F tile lookup.
@group(1) @binding(11) var tex3_sampler: sampler;  // declared for binding parity; textureLoad only, non-filtering on the Rust side
@group(1) @binding(4) var tex4: texture_2d<f32>;   // texture4: Raylib square-gradient tile blend mask.
@group(1) @binding(12) var tex4_sampler: sampler;  // linear on the Rust side

fn smoothHermite(edge0: f32, edge1: f32, value: f32) -> f32
{
    let t = clamp((value - edge0)/max(edge1 - edge0, 0.0001), 0.0, 1.0);
    return t*t*(3.0 - 2.0*t);
}

// This is deliberately a single, simple sampling rule so the CPU erosion
// initialisation can reproduce it with a bilinear texture lookup. The R32
// field is NOT periodic (it is one FastNoiseLite patch, not a wrapped one),
// so the phase is mirror-folded back over the texture instead of wrapped:
// a plain fract() repeat would tear between unrelated edge texels at every
// period seam, leaving cliff lines through the landform. The first half of
// each period samples exactly as before; the second half reads the first
// half reversed, so the fold is continuous at every crease.
//
// The fold valley must not reach uv = 0: that is the field's wrap edge, where
// bilinear filtering blends the unrelated first and last texels of the
// non-periodic patch, and the blend steps against the local trend along the
// fold line. The valley is clamped to the first texel centre instead, so the
// fold runs along the edge texel without ever blending across it. The
// reverse fold at uv = 0.5 lands on a genuine interior texel pair and needs
// no clamp.
fn signedNoise(p: vec2<f32>) -> f32
{
    let phase = fract(p/max(stage.noise_period, 0.0001) + vec2<f32>(0.5));
    let uv = vec2<f32>(0.5) - abs(phase - vec2<f32>(0.5));
    let halfTexel = 0.5/vec2<f32>(textureDimensions(tex0, 0));
    return textureSampleLevel(tex0, tex0_sampler, max(uv, halfTexel), 0.0).r*2.0 - 1.0;
}

fn rotateA(p: vec2<f32>) -> vec2<f32>
{
    return vec2<f32>(0.8*p.x - 0.6*p.y, 0.6*p.x + 0.8*p.y);
}

fn rotateB(p: vec2<f32>) -> vec2<f32>
{
    return vec2<f32>(0.6*p.x + 0.8*p.y, -0.8*p.x + 0.6*p.y);
}

// Bounded-exponential altitude profile around sea level, mirrored exactly by
// ShapeElevation in main.cpp (the constants arrive as uniforms). Convex on
// both sides of the waterline: the shore keeps gentle slopes for wide plains
// and a littoral shelf, and the gradient accelerates with elevation into
// steep mountain crowns and a progressively deepening seabed.
// (GLSL named the parameter `macro`; that word is reserved in WGSL.)
fn shapeElevation(macroIn: f32) -> f32
{
    var macroValue = macroIn;
    if (macroValue >= 0.0)
    {
        // tanh is saturated to +/-1 at |x| = 10 in f32. Bound its input to
        // avoid overflowing the GPU backend's exponential implementation.
        macroValue += stage.waterline_clearance * tanh(clamp(
                         macroValue/stage.waterline_clearance_scale, -10.0, 10.0))
                    * exp(-(macroValue*macroValue)
                          / (stage.waterline_clearance_decay*stage.waterline_clearance_decay));
        let t = macroValue/max(stage.land_profile_reference, 0.0001);
        return stage.land_profile_peak*(exp(stage.land_profile_curve*t) - 1.0)
             / (exp(stage.land_profile_curve) - 1.0);
    }
    // Clamped at the reference: the deepest basins flatten into an abyssal
    // plain at the depth constant instead of diving without bound.
    let t = min(macroValue/min(stage.ocean_profile_reference, -0.0001), 1.0);
    return stage.ocean_profile_depth*(exp(stage.ocean_profile_curve*t) - 1.0)
         / (exp(stage.ocean_profile_curve) - 1.0);
}

// Push a finished height away from the waterline, preserving its sign.
// Mirrored exactly by `push_from_waterline` in src/noise.rs, which is what the
// player's walking collision samples — the two must stay identical or the
// player floats over or sinks into the rendered ground.
//
// shapeElevation shapes only the macro landform, and the fine detail octaves
// are added after it, so the surface can land anywhere within about +-2 m of
// sea level across the whole coastal plain. Terrain in that band is
// indistinguishable from the water surface to the depth buffer, and once the
// sea has waves it is indistinguishable to the eye as well, because every
// trough exposes it again. This is the dead-zone removal that makes the
// coastline decisive: monotone, exactly zero at sea level so the shoreline
// does not move, and steepening the approach to it by 1 + push/scale.
fn pushFromWaterline(height: f32) -> f32
{
    let scale = max(stage.waterline_push_scale, 0.0001);
    let push = select(stage.waterline_push_sea, stage.waterline_push_land,
                      height >= 0.0);
    // Unbounded tanh overflows on Metal at mountain elevations, producing
    // NaN heights and normals. Saturation preserves the CPU profile while
    // keeping the shader finite on summits and in deep ocean basins.
    return height + push*tanh(clamp(height/scale, -10.0, 10.0));
}

fn baseHeight(p: vec2<f32>) -> f32
{
    // Compress only the macro domain. The final detail octaves still use
    // physical coordinates, retaining nearby surface definition even though
    // ranges, plains, islands, and bays are much broader.
    let detailA = rotateA(p);
    let detailB = rotateB(p);
    let macroP = p/max(stage.landform_horizontal_scale, 0.0001);
    let sourceA = rotateA(macroP);
    let sourceB = rotateB(macroP);
    let warp = vec2<f32>(
        signedNoise(sourceA*0.74 + vec2<f32>(1129.0, -587.0)),
        signedNoise(sourceB*0.69 + vec2<f32>(-977.0, 1231.0)));
    let warped = macroP + warp*210.0;
    let a = rotateA(warped);
    let b = rotateB(warped);

    let continent = signedNoise(warped*0.38 + vec2<f32>(137.0, -311.0));
    let coastDetail = signedNoise(a*0.83 + vec2<f32>(-743.0, 521.0));
    let plains = signedNoise(b*2.20 + vec2<f32>(193.0, -97.0));
    let mountainField = signedNoise(a*0.63 + vec2<f32>(887.0, 653.0));
    let ridgeNoiseA = signedNoise(b*2.15 + vec2<f32>(-419.0, -811.0));
    let ridgeNoiseB = signedNoise(a*3.35 + vec2<f32>(1601.0, -1297.0));
    let ridgeA = smoothHermite(0.0, 1.0, 1.0 - abs(ridgeNoiseA));
    let ridgeB = smoothHermite(0.0, 1.0, 1.0 - abs(ridgeNoiseB));
    let ridge = ridgeA*0.72 + ridgeB*0.28;
    let detail = signedNoise(detailA*8.60 + vec2<f32>(47.0, 101.0));
    let microDetail = signedNoise(detailB*31.0 + vec2<f32>(-211.0, 379.0));

    let continental = continent*0.70 + coastDetail*0.30;
    let mountainMask = smoothHermite(-0.02, 0.62,
                                     mountainField + continent*0.38);

    var macroHeight = stage.sea_level - 5.0;
    macroHeight += continental*32.0;
    macroHeight += plains*6.0;
    macroHeight += mountainMask*(20.0 + 68.0*ridge);
    let fineHeight = detail*(1.4 + 2.8*mountainMask)
                   + microDetail*(0.55 + 0.55*mountainMask);
    // The macro landform goes through the exponential altitude profile; fine
    // detail octaves are added after it so plains keep their rolling surface
    // and the steep crowns do not shimmer.
    let verticalScale = max(stage.landform_vertical_scale, 0.0001);
    let height = stage.sea_level
        + shapeElevation((macroHeight - stage.sea_level)*verticalScale)
        + fineHeight*verticalScale;

    // Only a small patch at spawn is stabilised for the first player frame;
    // every landscape-scale feature comes from the warped noise field.
    let distanceFromSpawn = length(p);
    let safeHeight = stage.sea_level + 14.0 + plains*1.8 + detail*0.35;
    let spawnBlend = smoothHermite(22.0, 90.0, distanceFromSpawn);
    // The push goes on last, after the spawn blend, so the stabilised patch and
    // the open landscape are shaped by the same function and no artificial
    // ring appears at the blend's edge.
    return pushFromWaterline(mix(safeHeight, height, spawnBlend));
}

fn lookupErosionTile(tileCoordinate: vec2<f32>, record: ptr<function, vec4<f32>>) -> bool
{
    let relativeTile = tileCoordinate - stage.erosion_lookup_min_tile;
    let lookupSize = max(stage.erosion_lookup_size, 1.0);
    if (any(relativeTile < vec2<f32>(0.0)) ||
        any(relativeTile >= vec2<f32>(lookupSize)))
    {
        *record = vec4<f32>(0.0);
        return false;
    }

    *record = textureLoad(tex3, vec2<i32>(relativeTile), 0);
    return (*record).a > 0.0;
}

fn erosionSupportUV(worldXZ: vec2<f32>, tileCoordinate: vec2<f32>) -> vec2<f32>
{
    let stride = max(stage.erosion_tile_stride, 0.0001);
    let footprint = max(stage.erosion_footprint_size, 0.0001);
    let supportMinimum = tileCoordinate*stride - vec2<f32>(0.5*footprint);
    return (worldXZ - supportMinimum)/footprint;
}

fn erosionTileMask(worldXZ: vec2<f32>, tileCoordinate: vec2<f32>) -> f32
{
    let tileUV = erosionSupportUV(worldXZ, tileCoordinate);
    if (any(tileUV < vec2<f32>(0.0)) ||
        any(tileUV > vec2<f32>(1.0))) { return 0.0; }
    return max(textureSampleLevel(tex4, tex4_sampler, tileUV, 0.0).r, 0.0);
}

fn erosionAtlasUV(worldXZ: vec2<f32>, tileCoordinate: vec2<f32>, atlasSlot: vec2<f32>) -> vec2<f32>
{
    let outputResolution = max(stage.erosion_output_resolution, 1.0);
    let tileUV = clamp(erosionSupportUV(worldXZ, tileCoordinate),
                       vec2<f32>(0.0), vec2<f32>(1.0));
    let atlasPixel = atlasSlot*max(stage.erosion_atlas_pitch, outputResolution)
                   + vec2<f32>(max(stage.erosion_atlas_gutter, 0.0))
                   + tileUV*outputResolution;
    return atlasPixel/max(stage.erosion_atlas_size, 1.0);
}

fn accumulateErosionTile(worldXZ: vec2<f32>, tileCoordinate: vec2<f32>, spatialWeight: f32,
                         weightedDelta: ptr<function, f32>)
{
    if (spatialWeight <= 0.0) { return; }

    var record = vec4<f32>(0.0);
    if (!lookupErosionTile(tileCoordinate, &record)) { return; }
    let reveal = clamp(record.z*record.a, 0.0, 1.0);
    let heightDelta = textureSampleLevel(tex1, tex1_sampler,
        erosionAtlasUV(worldXZ, tileCoordinate, record.xy), 0.0).r;
    *weightedDelta += spatialWeight*reveal*heightDelta;
}

fn erosionDelta(worldXZ: vec2<f32>) -> f32
{
    // Pass centres lie on a half-footprint lattice. Between two adjacent
    // centres, the raylib-derived Hermite tents are complementary; their
    // separable products therefore form one smooth four-pass partition.
    let stride = max(stage.erosion_tile_stride, 0.0001);
    let minimumTile = floor(worldXZ/stride);
    let tile00 = minimumTile;
    let tile10 = minimumTile + vec2<f32>(1.0, 0.0);
    let tile01 = minimumTile + vec2<f32>(0.0, 1.0);
    let tile11 = minimumTile + vec2<f32>(1.0, 1.0);

    var spatialWeights = vec4<f32>(erosionTileMask(worldXZ, tile00),
                                   erosionTileMask(worldXZ, tile10),
                                   erosionTileMask(worldXZ, tile01),
                                   erosionTileMask(worldXZ, tile11));
    let geometricSum = dot(spatialWeights, vec4<f32>(1.0));
    if (geometricSum <= 0.000001) { return 0.0; }
    spatialWeights /= geometricSum;

    // Only this fixed geometric partition is normalized. Readiness never
    // changes its denominator: a missing pass contributes zero erosion (the
    // base surface) and its reveal brings it in without amplifying neighbours.
    var weightedDelta = 0.0;
    accumulateErosionTile(worldXZ, tile00, spatialWeights.x, &weightedDelta);
    accumulateErosionTile(worldXZ, tile10, spatialWeights.y, &weightedDelta);
    accumulateErosionTile(worldXZ, tile01, spatialWeights.z, &weightedDelta);
    accumulateErosionTile(worldXZ, tile11, spatialWeights.w, &weightedDelta);
    return weightedDelta;
}

fn erosionVisibility(worldXZ: vec2<f32>) -> f32
{
    return 1.0 - smoothHermite(stage.erosion_visibility_full_radius,
                               stage.erosion_visibility_zero_radius,
                               length(worldXZ - stage.erosion_visibility_center));
}

fn terrainHeight(worldXZ: vec2<f32>) -> f32
{
    // The four-pass result is already spatially normalized. The radial fade
    // stays well inside the region where every required lookup record exists.
    return baseHeight(worldXZ) + erosionDelta(worldXZ)*erosionVisibility(worldXZ);
}

// The clipmap mesh stores planar grid coordinates in vertexPosition.xz.
// uClipOrigin is the world-space centre of the current clipmap level and
// uSpacing is the distance between two vertices at this level.
struct VsOutput {
    @builtin(position) position: vec4<f32>,     // gl_Position
    @location(0) fragPositionView: vec3<f32>,   // fragPositionView
    @location(1) fragNormalView: vec3<f32>,     // fragNormalView
    @location(2) fragWorldPosition: vec3<f32>,  // fragWorldPosition
    @location(3) fragWorldNormal: vec3<f32>,    // fragWorldNormal
    @location(4) fragErosionDelta: f32,         // fragErosionDelta
};

@vertex
fn vs_main(@location(0) vertexPosition: vec3<f32>) -> VsOutput
{
    let localXZ = vertexPosition.xz*stage.spacing;
    let fineXZ = stage.clip_origin + localXZ;

    // Geomorph the outer portion of every level onto the next coarser global
    // lattice. Because both levels meet on the same lattice, ring seams remain
    // closed as the camera and clip origins move.
    let ringDistance = max(abs(localXZ.x), abs(localXZ.y));
    let morph = smoothHermite(stage.morph_start, stage.morph_end, ringDistance);
    let coarseXZ =
        floor(fineXZ/max(stage.next_spacing, 0.0001) + vec2<f32>(0.5))*stage.next_spacing;
    let worldXZ = mix(fineXZ, coarseXZ, morph);

    // Expose the blended erosion contribution for the fragment-stage debug
    // overlay. Compute it once here instead of calling terrainHeight() so the
    // delta is not evaluated twice for the primary vertex.
    let erosionContribution = erosionDelta(worldXZ)*erosionVisibility(worldXZ);
    let height = baseHeight(worldXZ) + erosionContribution;
    var output: VsOutput;
    output.fragErosionDelta = erosionContribution;

    // Evaluate normals at a world-space interval appropriate to this LOD. This
    // avoids the high-frequency shimmer produced by differentiating the mesh.
    // Follow the same LOD blend as position so lighting does not reveal a seam
    // where the fine geometry has already morphed onto the coarse lattice.
    let normalStep = max(1.0, mix(stage.spacing, stage.next_spacing, morph));
    let heightLeft = terrainHeight(worldXZ - vec2<f32>(normalStep, 0.0));
    let heightRight = terrainHeight(worldXZ + vec2<f32>(normalStep, 0.0));
    let heightBack = terrainHeight(worldXZ - vec2<f32>(0.0, normalStep));
    let heightFront = terrainHeight(worldXZ + vec2<f32>(0.0, normalStep));
    let localNormal = normalize(vec3<f32>(heightLeft - heightRight,
                                          2.0*normalStep,
                                          heightBack - heightFront));

    let worldPosition4 = stage.model*vec4<f32>(worldXZ.x, height, worldXZ.y, 1.0);
    // GLSL computed transpose(inverse(mat3(matModel))); stage.inverse_model is
    // inverse(matModel) supplied by the CPU because WGSL has no inverse().
    let normalModel = transpose(mat3x3<f32>(stage.inverse_model[0].xyz,
                                            stage.inverse_model[1].xyz,
                                            stage.inverse_model[2].xyz));
    let worldNormal = normalize(normalModel*localNormal);
    let viewPosition4 = globals.view*worldPosition4;
    let viewNormal = normalize(mat3x3<f32>(globals.view[0].xyz, globals.view[1].xyz, globals.view[2].xyz)*worldNormal);

    output.fragPositionView = viewPosition4.xyz;
    output.fragNormalView = viewNormal;
    output.fragWorldPosition = worldPosition4.xyz;
    output.fragWorldNormal = worldNormal;

    output.position = globals.projection*viewPosition4;
    return output;
}

// STAGE UNIFORMS:
//   model (mat4x4<f32>)                    — matModel: model transform of the clipmap mesh
//   inverse_model (mat4x4<f32>)            — inverse(matModel), CPU-computed (WGSL has no inverse())
//   clip_origin (vec2<f32>)                — uClipOrigin: world-space centre of the current clipmap level
//   spacing (f32)                          — uSpacing
//   next_spacing (f32)                     — uNextSpacing
//   morph_start (f32)                      — uMorphStart
//   morph_end (f32)                        — uMorphEnd
//   sea_level (f32)                        — uSeaLevel
//   noise_period (f32)                     — uNoisePeriod
//   landform_horizontal_scale (f32)        — uLandformHorizontalScale
//   landform_vertical_scale (f32)          — uLandformVerticalScale
//   land_profile_curve (f32)               — uLandProfileCurve
//   land_profile_reference (f32)           — uLandProfileReference
//   land_profile_peak (f32)                — uLandProfilePeak
//   waterline_clearance (f32)              — uWaterlineClearance
//   waterline_clearance_scale (f32)        — uWaterlineClearanceScale
//   waterline_clearance_decay (f32)        — uWaterlineClearanceDecay
//   ocean_profile_curve (f32)              — uOceanProfileCurve
//   ocean_profile_reference (f32)          — uOceanProfileReference
//   ocean_profile_depth (f32)              — uOceanProfileDepth
//   erosion_tile_stride (f32)              — uErosionTileStride
//   erosion_footprint_size (f32)           — uErosionFootprintSize
//   erosion_output_resolution (f32)        — uErosionOutputResolution
//   erosion_atlas_pitch (f32)              — uErosionAtlasPitch
//   erosion_atlas_size (f32)               — uErosionAtlasSize
//   erosion_atlas_gutter (f32)             — uErosionAtlasGutter
//   erosion_lookup_min_tile (vec2<f32>)    — uErosionLookupMinTile
//   erosion_lookup_size (f32)              — uErosionLookupSize
//   erosion_visibility_center (vec2<f32>)  — uErosionVisibilityCenter
//   erosion_visibility_full_radius (f32)   — uErosionVisibilityFullRadius
//   erosion_visibility_zero_radius (f32)   — uErosionVisibilityZeroRadius
// Rust fill (uniform, 272 bytes, WGSL uniform-address alignment):
//   [0]   model: [f32; 16] (column-major)
//   [64]  inverse_model: [f32; 16] (column-major inverse of model)
//   [128] clip_origin: [f32; 2]
//   [136..228] the 23 f32 fields in the order spacing .. erosion_atlas_gutter
//         (4 bytes each), 4 bytes pad at [228] for the next vec2's alignment
//   [232] erosion_lookup_min_tile: [f32; 2]
//   [240] erosion_lookup_size
//   [248] erosion_visibility_center: [f32; 2]
//   [256] erosion_visibility_full_radius
//   [260] erosion_visibility_zero_radius (struct rounds to 272 bytes)
// Textures (group 1): binding 0 = R32 base noise (linear sampler at 8),
// 1 = surface atlas (linear sampler at 9), 3 = RGBA32F tile lookup
// (TEXTURE-LOAD ONLY; non-filtering sampler at 11), 4 = blend mask (linear
// sampler at 12). Vertex input: @location(0) = vec3 planar grid coordinates
// (raylib vertexPosition; .xz scaled by spacing). Outputs:
// @location(0) fragPositionView (view-space xyz),
// @location(1) fragNormalView (view-space normal),
// @location(2) fragWorldPosition (world-space xyz),
// @location(3) fragWorldNormal (world-space normal),
// @location(4) fragErosionDelta (f32, blended erosion contribution for the
// fragment-stage debug overlay).
// group 0 binding 0 = shared GlobalUniforms.
