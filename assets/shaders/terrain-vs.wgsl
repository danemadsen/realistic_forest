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
    snowline_altitude: f32,                // fragment stage: persistent snowline above sea level
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

@group(1) @binding(17) var<storage, read> river_grid: array<u32>;      // river lookup grid
@group(1) @binding(18) var<storage, read> river_segments: array<RiverSegment>; // river carve segments

// BEGIN SHARED TERRAIN HEIGHT
// The streamed landform: base noise relief plus the four-pass erosion delta,
// faded with distance from the player. terrain-vs.wgsl draws the clipmap and
// the lighting heightfield from it, and vegetation-cull.wgsl seats every
// plant's root on it; a test keeps the pasted copies identical, so a trunk
// stands exactly on the ground the terrain pass draws. Requires StageUniforms
// as `stage` and the base noise, surface atlas, tile lookup and blend mask as
// tex0, tex1, tex3 and tex4 with their samplers, and the shared river block
// (river-functions.wgslinc) with its two storage buffers.
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
    // Erosion was simulated over the river-carved base, so its delta goes on
    // top of the carved base, and the channels are held once more over the
    // result: a river keeps its bed and banks whatever the water did there.
    let river = riverEnvelope(worldXZ);
    return riverClamp(river, riverClamp(river, baseHeight(worldXZ))
                             + erosionDelta(worldXZ)*erosionVisibility(worldXZ));
}
// END SHARED TERRAIN HEIGHT

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

@group(1) @binding(7) var snow_compression: texture_2d<f32>;
@group(1) @binding(15) var snow_sampler: sampler;
struct SnowUniforms {
    mapping: vec4<f32>, // world minimum XZ, texel size, untouched depth
};
@group(1) @binding(16) var<uniform> snow: SnowUniforms;

fn snowCompression(worldXZ: vec2<f32>) -> f32 {
    let span = vec2<f32>(textureDimensions(snow_compression)) * snow.mapping.z;
    let uv = (worldXZ - snow.mapping.xy) / max(span, vec2<f32>(0.001));
    if (any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0))) { return 0.0; }
    return textureSampleLevel(snow_compression, snow_sampler, uv, 0.0).r;
}

fn snowLift(worldXZ: vec2<f32>, height: f32, materialNormal: vec3<f32>) -> f32 {
    return snowCoverage(height, materialNormal) * snow.mapping.w * (1.0 - snowCompression(worldXZ));
}

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
// record (RIVER_NO_LAKE_RECORD if no lake reaches it): the lake's level as
// f32 bits and a 64-bit mask of which of the cell's 8 x 8 lake cells lie
// under the lake or its shore. A zero resolution means there are no rivers.
//
// A lake has no channel: ground below its level near it lies under its
// water, and riverBankAt measures the shore in a river bank's terms.
//
// Each segment bounds the ground from above (the channel bed, then a bank
// cone that steepens away from the water) and from below (a low levee that
// keeps the water in its channel). The upper bounds combine by minimum and
// the lower ones by maximum, so confluences open into each other; a segment
// has a flat start and a round end, so it never reaches back up the channel.
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
const RIVER_NO_LAKE_RECORD: u32 = 0xffffffffu;
const RIVER_LAKE_CELLS_ACROSS: u32 = 8u;
// Metres of shore per metre of rise above a lake's water.
const RIVER_LAKE_SHORE_RUN: f32 = 6.0;
const RIVER_BANK_REACH: f32 = 12.0;
const RIVER_BANK_CURVE: f32 = 0.16;
const RIVER_LEVEE_OUTER_SLOPE: f32 = 0.15;
// Height over which the carve's creases are rounded (CARVE_ROUNDING).
const RIVER_CARVE_ROUNDING: f32 = 0.8;
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
        // A flat-bottomed bowl, skewed: zero at both banks, deep right up to
        // them, and deepest toward the outer one.
        let u = side*centreDistance/halfWidth;
        let profile = (1.0 - u*u*u*u)*(1.0 + skew*u);
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
    let count = min(river_grid[entry + 1u], RIVER_MAX_CANDIDATES);
    for (var i = 0u; i < count; i += 1u)
    {
        riverCombine(&total, riverSegmentEnvelope(river_segments[river_grid[offset + i]], p));
    }
    let record = river_grid[entry + 2u];
    if (record != RIVER_NO_LAKE_RECORD)
    {
        let fine = floor((p - origin)/(bitcast<f32>(river_grid[2])/f32(RIVER_LAKE_CELLS_ACROSS)));
        let local = vec2<u32>(fine - cell*f32(RIVER_LAKE_CELLS_ACROSS));
        let bit = local.y*RIVER_LAKE_CELLS_ACROSS + local.x;
        if (((river_grid[record + 1u + bit/32u] >> (bit % 32u)) & 1u) == 1u)
        {
            total.lake = bitcast<f32>(river_grid[record]);
        }
    }
    return total;
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

// The ground with the channels cut into it and their banks held up, the
// creases where the carve meets the natural ground rounded off: a bank's top
// curves over into the land above it, and a levee's foot into the land below.
fn riverClamp(envelope: RiverEnvelope, height: f32) -> f32
{
    return riverSmoothMin(riverSmoothMax(height, envelope.lower, RIVER_CARVE_ROUNDING),
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

fn riverShoreRun(envelope: RiverEnvelope, height: f32, groundNormal: vec3<f32>) -> f32
{
    let tangent = sqrt(max(1.0 - groundNormal.y*groundNormal.y, 0.0))/max(groundNormal.y, 0.05);
    let riverScale = clamp(0.3 + 0.28*envelope.half_width, 0.35, 1.3);
    let riverRun = max(envelope.bank_distance,
                       (height - envelope.water)*RIVER_LAKE_SHORE_RUN)/riverScale;
    let lakeRise = height - envelope.lake;
    let lakeRun = max(lakeRise/max(tangent, 0.01), lakeRise*RIVER_LAKE_SHORE_RUN)
                / RIVER_SHORE_LAKE_SCALE;
    return clamp(min(riverRun, lakeRun), -40.0, 40.0);
}
// END SHORE RUN

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
    @location(5) frag_material_normal: vec3<f32>, // Distance-prefiltered slope/aspect for material placement.
    @location(6) frag_base_height: f32,
    @location(7) frag_snow_compaction: f32,
    // The nearest river or lake: metres past its waterline (negative under
    // water), side of the bend (+ outside, - inside), whitewater, flow speed.
    @location(8) frag_river: vec4<f32>,
    // How far up the nearest river's or lake's shore the ground lies, the
    // coordinate of the terrain's bank band (riverShoreRun): negative under
    // water, 40 near none.
    @location(9) frag_river_shore: f32,
};

fn terrainVertex(vertexPosition: vec3<f32>, deformSnow: bool) -> VsOutput
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
    // The same carve terrainHeight applies, with the river looked up once.
    let river = riverEnvelope(worldXZ);
    var height = riverClamp(river, riverClamp(river, baseHeight(worldXZ)) + erosionContribution);
    var output: VsOutput;
    output.fragErosionDelta = erosionContribution;
    output.frag_base_height = height;
    output.frag_snow_compaction = 0.0;
    // Bounded, so a triangle straddling a river's reach interpolates sanely.
    // A lake's still water and silted bed take over where its shore is
    // nearer than any river's bank.
    let bank = riverBankAt(river, height);
    let still = bank < river.bank_distance;
    output.frag_river = vec4<f32>(clamp(bank, -40.0, 40.0),
                                  select(river.bend, 0.0, still),
                                  select(river.turbulence, 0.0, still),
                                  select(length(river.velocity), 0.0, still));

    // Evaluate normals at a world-space interval appropriate to this LOD. This
    // avoids the high-frequency shimmer produced by differentiating the mesh.
    // Follow the same LOD blend as position so lighting does not reveal a seam
    // where the fine geometry has already morphed onto the coarse lattice.
    let normalStep = max(0.25, mix(stage.spacing, stage.next_spacing, morph));
    let heightLeft = terrainHeight(worldXZ - vec2<f32>(normalStep, 0.0));
    let heightRight = terrainHeight(worldXZ + vec2<f32>(normalStep, 0.0));
    let heightBack = terrainHeight(worldXZ - vec2<f32>(0.0, normalStep));
    let heightFront = terrainHeight(worldXZ + vec2<f32>(0.0, normalStep));
    var localNormal = normalize(vec3<f32>(heightLeft - heightRight,
                                          2.0*normalStep,
                                          heightBack - heightFront));

    // Material slope and aspect use a four-metre terrain interval near the
    // camera. Farther out the clipmap's vertices are 8-64 m apart, and a 4 m
    // slope sampled once per vertex aliases: neighbouring vertices pick up
    // unrelated micro-relief, and the interpolated snow, rock and soil
    // decisions turn into rows of triangles along every material margin.
    // The interval therefore widens with horizontal camera distance to about
    // 1.3 vertex spacings, which prefilters distant coverage the way mips
    // prefilter a texture. It grows continuously with distance rather than
    // following the LOD rings, so coverage never pops when the clipmap
    // origin snaps (the lighting normal above does follow the rings).
    // terrainHeight includes the same erosion reveal and fade as the mesh.
    // Reuse the lighting samples where that LOD already has this interval.
    let cameraDistance = length(worldXZ - globals.camera_position.xz);
    let materialStep = clamp(cameraDistance / 48.0, 4.0, 96.0);
    var materialNormal = localNormal;
    if (abs(normalStep - materialStep) > 0.0001) {
        let materialLeft = terrainHeight(worldXZ - vec2<f32>(materialStep, 0.0));
        let materialRight = terrainHeight(worldXZ + vec2<f32>(materialStep, 0.0));
        let materialBack = terrainHeight(worldXZ - vec2<f32>(0.0, materialStep));
        let materialFront = terrainHeight(worldXZ + vec2<f32>(0.0, materialStep));
        materialNormal = normalize(vec3<f32>(materialLeft - materialRight,
                                             2.0*materialStep,
                                             materialBack - materialFront));
    }

    if (deformSnow) {
        // Placement reads the underlying terrain, so the sides of a trail
        // remain snow-covered. Only the geometry and lighting normals deform.
        let compression = snowCompression(worldXZ);
        output.frag_snow_compaction = compression;
        height += snowCoverage(height, materialNormal) * snow.mapping.w * (1.0 - compression);
        let left = heightLeft + snowLift(worldXZ - vec2<f32>(normalStep, 0.0), heightLeft, materialNormal);
        let right = heightRight + snowLift(worldXZ + vec2<f32>(normalStep, 0.0), heightRight, materialNormal);
        let back = heightBack + snowLift(worldXZ - vec2<f32>(0.0, normalStep), heightBack, materialNormal);
        let front = heightFront + snowLift(worldXZ + vec2<f32>(0.0, normalStep), heightFront, materialNormal);
        localNormal = normalize(vec3<f32>(left - right, 2.0 * normalStep, back - front));
    }

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
    output.frag_material_normal = normalize(normalModel*materialNormal);
    // How far up its shore this ground lies, over the same prefiltered
    // slope the fragment stage reads when it measures the shore exactly.
    output.frag_river_shore = riverShoreRun(river, output.frag_base_height, output.frag_material_normal);

    output.position = globals.projection*viewPosition4;
    return output;
}

// Grass roots use baseline terrain. That capture is cached independently of
// moving snow tracks, while the visible terrain uses the displaced surface.
@vertex
fn vs_habitat(@location(0) vertexPosition: vec3<f32>) -> VsOutput {
    return terrainVertex(vertexPosition, false);
}

@vertex
fn vs_main(@location(0) vertexPosition: vec3<f32>) -> VsOutput {
    return terrainVertex(vertexPosition, true);
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
//   [260] erosion_visibility_zero_radius
//   [264] waterline_push_land, [268] waterline_push_sea, [272] waterline_push_scale
//   [276] snowline_altitude (struct rounds to 288 bytes)
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
// @location(5) frag_material_normal (world-space normal for material
// slope/aspect: a 4 m interval near the camera, widening with horizontal
// camera distance to track the vertex spacing, independent of the LOD rings).
// @location(8) frag_river (bank distance, bend side, whitewater, flow speed),
// @location(9) frag_river_shore (metres up the nearest shore, riverShoreRun).
// group 0 binding 0 = shared GlobalUniforms.
