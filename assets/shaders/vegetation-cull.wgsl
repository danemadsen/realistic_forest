// Vegetation culling and LOD selection: one thread per streamed plant.
//
// The CPU scatter places every plant on the base landform and knows nothing
// of erosion, the camera or the frame. This pass does the rest, every frame,
// so nothing it decides can go stale:
//
// 1. Distance and frustum culling against a bounding sphere around the
//    scatter's height estimate, padded by the largest erosion change.
// 2. Seating the root on the ground the terrain pass draws. Trees and tall
//    shrubs evaluate the shared terrain height model (base relief plus the
//    streamed erosion, faded with distance exactly as the clipmap fades it)
//    and refuse roots on steep faces and in incised channels. Small plants
//    read the grass habitat capture instead and grow only where grass would.
// 3. LOD selection in multiples of the plant's own height, with a
//    screen-door cross-fade over the last stretch of every band, and a
//    dissolve at the far edge.
// 4. Compaction: each surviving plant is appended to its model's region of
//    the LOD's output, and the matching indirect draws' instance counts are
//    incremented in place. The CPU never reads anything back.
// 5. Shadows: a plant whose bounds fall inside a shadow cascade's light-space
//    box is appended once more, to that cascade's region, at the cascade's
//    fixed LOD, whether or not the camera sees it: a tree behind the eye
//    still shades the ground in front of it.

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

// The terrain stage block, byte for byte the layout terrain-vs.wgsl reads;
// the Rust side fills it with the clipmap's level-0 values every frame.
struct StageUniforms {
    model: mat4x4<f32>,
    inverse_model: mat4x4<f32>,
    clip_origin: vec2<f32>,
    spacing: f32,
    next_spacing: f32,
    morph_start: f32,
    morph_end: f32,
    sea_level: f32,
    noise_period: f32,
    landform_horizontal_scale: f32,
    landform_vertical_scale: f32,
    land_profile_curve: f32,
    land_profile_reference: f32,
    land_profile_peak: f32,
    waterline_clearance: f32,
    waterline_clearance_scale: f32,
    waterline_clearance_decay: f32,
    ocean_profile_curve: f32,
    ocean_profile_reference: f32,
    ocean_profile_depth: f32,
    erosion_tile_stride: f32,
    erosion_footprint_size: f32,
    erosion_output_resolution: f32,
    erosion_atlas_pitch: f32,
    erosion_atlas_size: f32,
    erosion_atlas_gutter: f32,
    erosion_lookup_min_tile: vec2<f32>,
    erosion_lookup_size: f32,
    erosion_visibility_center: vec2<f32>,
    erosion_visibility_full_radius: f32,
    erosion_visibility_zero_radius: f32,
    waterline_push_land: f32,
    waterline_push_sea: f32,
    waterline_push_scale: f32,
    snowline_altitude: f32,
};

@group(1) @binding(0) var tex0: texture_2d<f32>;      // R32 base noise
@group(1) @binding(1) var tex0_sampler: sampler;      // linear, repeat
@group(1) @binding(2) var tex1: texture_2d<f32>;      // surface atlas; R is the eroded-height delta
@group(1) @binding(3) var tex1_sampler: sampler;      // linear, clamp
@group(1) @binding(4) var tex3: texture_2d<f32>;      // point tile lookup (textureLoad only)
@group(1) @binding(5) var tex4: texture_2d<f32>;      // tile blend mask
@group(1) @binding(6) var tex4_sampler: sampler;      // linear, clamp
@group(1) @binding(7) var<uniform> stage: StageUniforms;
@group(1) @binding(8) var habitat: texture_2d<f32>;   // grass habitat: height, suitability, normal XZ
@group(1) @binding(9) var habitat_sampler: sampler;   // linear, clamp

@group(1) @binding(10) var<storage, read> river_grid: array<u32>;      // river lookup grid
@group(1) @binding(11) var<storage, read> river_segments: array<RiverSegment>; // river carve segments

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
    // The surface of a lake reaching here, or RIVER_NO_LAKE.
    lake: f32,
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
        // Past the segment's end (beyond a half width, the outside of a
        // bend's waterline) its levee falls on as its water does, or down a
        // rapid each end would hold a ledge up beside the next.
        let fall = max(segment.water.x - segment.water.y, 0.0)/segmentLength*max(beyond - halfWidth, 0.0);
        envelope.lower = water - fall + min(bank*pastBank, freeboard)
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
        let local = (p - origin)/(bitcast<f32>(river_grid[2])/f32(RIVER_LAKE_CELLS_ACROSS))
                  - cell*f32(RIVER_LAKE_CELLS_ACROSS);
        total.lake = riverLakeSurface(record, local);
    }
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

// The ground with the channels cut into it and their banks held up, the
// creases where the carve meets the natural ground rounded off: a bank's top
// curves over into the land above it, and a levee's foot into the land below.
fn riverClamp(envelope: RiverEnvelope, height: f32) -> f32
{
    return riverSmoothMin(riverSmoothMax(height, envelope.lower, RIVER_CARVE_ROUNDING),
                          envelope.upper, RIVER_CARVE_ROUNDING);
}
// END SHARED RIVERS

// One streamed plant, as the scatter wrote it. Mirrors `PlantInstance` in
// src/vegetation/scatter.rs (32 bytes).
struct PlantInstance {
    position: vec3<f32>,   // world XZ and the base-landform height estimate
    scale: f32,
    yaw: f32,
    seed: f32,
    model: u32,
    layer: u32,
};

// Per-model drawing rules and output bookkeeping. Mirrors `ModelParams` in
// src/render/vegetation_node.rs (112 bytes).
struct ModelParams {
    height: f32,           // authored height of the LOD-0 mesh, metres
    crown_radius: f32,
    crown_base: f32,
    bound_radius: f32,     // widest reach of any LOD
    lod_end: vec4<f32>,    // LOD k ends lod_end[k] plant heights from the eye
    max_distance: f32,
    lod_count: u32,
    habitat: u32,          // 1: small plant rooted through the grass habitat
    region: u32,           // this model's first slot in every LOD's output
    draw_word: vec4<u32>,  // args word of each LOD's first draw's instance count
    draw_count: vec4<u32>, // consecutive draws (one per primitive) per LOD
    shadow_word: vec4<u32>,  // the same for each shadow cascade's draws
    shadow_count: vec4<u32>, // 0: the model casts nothing into that cascade
};

// A plant to draw, as the vertex stage reads it. Mirrors `DrawInstance`.
struct DrawInstance {
    position: vec3<f32>,   // world root, seated on the terrain
    scale: f32,
    yaw: f32,
    seed: f32,
    // Screen-door coverage: (0, 1] keeps pixels whose dither is below it,
    // [-1, 0) keeps those at or above its magnitude. Complementary pairs
    // cross-fade two LODs without a gap or a doubled pixel.
    fade: f32,
    model: u32,
};

struct CullUniform {
    planes: array<vec4<f32>, 4>, // frustum sides, world space, normals inward
    camera: vec4<f32>,           // eye XYZ, LOD distance scale
    habitat_mapping: vec4<f32>,  // grass capture centre XZ, span, metres per texel
    counts: vec4<u32>,           // plants, region stride, habitat ready, shadow cascades
    ground: vec4<f32>,           // lowest root, steepest root, deepest furrow, culling slack
    light_right: vec4<f32>,      // the shadowing light's frame: across,
    light_up: vec4<f32>,         // up,
    light_forward: vec4<f32>,    // and along its travel
    cascades: array<vec4<f32>, 4>, // light-frame centre XY, half extent, far depth
    cascade_near: vec4<f32>,     // light-frame near depth of each cascade
};

@group(2) @binding(0) var<uniform> cull: CullUniform;
@group(2) @binding(1) var<storage, read> plants: array<PlantInstance>;
@group(2) @binding(2) var<storage, read> models: array<ModelParams>;
@group(2) @binding(3) var<storage, read_write> visible: array<DrawInstance>;
@group(2) @binding(4) var<storage, read_write> draw_args: array<atomic<u32>>;

/// Share of a LOD band, at its far end, over which it cross-fades into the
/// next one.
const LOD_FADE_BAND: f32 = 0.14;
/// Share of the maximum distance over which plants dissolve away.
const EDGE_FADE_BAND: f32 = 0.12;
/// Words of one `DrawIndexedIndirectArgs`.
const ARGS_WORDS: u32 = 5u;
/// Output regions before the first shadow cascade's: one per LOD.
const LOD_SLOTS: u32 = 4u;
/// Least grass suitability a small plant roots in, from the habitat capture;
/// the grass's own clumps take root from 0.02.
const SMALL_PLANT_ROOT: f32 = 0.06;

fn sphereVisible(centre: vec3<f32>, radius: f32) -> bool {
    for (var i = 0; i < 4; i++) {
        let plane = cull.planes[i];
        if (dot(plane.xyz, centre) + plane.w < -radius) {
            return false;
        }
    }
    return true;
}

fn habitatUV(xz: vec2<f32>) -> vec2<f32> {
    return (xz - cull.habitat_mapping.xy) / cull.habitat_mapping.z + 0.5;
}

// The least suitable of the four capture texels around a point, as the grass
// reads it, so a forbidden surface is never blurred into a root.
fn habitatAllowed(xz: vec2<f32>) -> f32 {
    let size = vec2<i32>(textureDimensions(habitat));
    let cell = vec2<i32>(floor(habitatUV(xz) * vec2<f32>(size) - 0.5));
    if (any(cell < vec2<i32>(0)) || any(cell + 1 >= size)) {
        return 0.0;
    }
    let four = textureGather(1, habitat, habitat_sampler, habitatUV(xz));
    return min(min(four.x, four.y), min(four.z, four.w));
}

// Appends a plant to one output region: counts it into each of the region's
// draws (one per primitive) and writes it at the slot the first returned.
fn append(plant: PlantInstance, model: ModelParams, region: u32, first: u32, draws: u32,
          root: vec3<f32>, fade: f32) {
    let slot = atomicAdd(&draw_args[first], 1u);
    for (var primitive = 1u; primitive < draws; primitive++) {
        atomicAdd(&draw_args[first + primitive * ARGS_WORDS], 1u);
    }
    visible[region * cull.counts.y + model.region + slot] =
        DrawInstance(root, plant.scale, plant.yaw, plant.seed, fade, plant.model);
}

fn emit(plant: PlantInstance, model: ModelParams, lod: u32, root: vec3<f32>, fade: f32) {
    append(plant, model, lod, model.draw_word[lod], model.draw_count[lod], root, fade);
}

// The shadow cascades whose light-space boxes a bounding sphere touches, as
// a bit mask, among those the model casts into.
fn shadowCascades(centre: vec3<f32>, radius: f32, model: ModelParams) -> u32 {
    let cascades = min(cull.counts.w, 4u);
    if (cascades == 0u) {
        return 0u;
    }
    let light = vec3<f32>(dot(centre, cull.light_right.xyz),
                          dot(centre, cull.light_up.xyz),
                          dot(centre, cull.light_forward.xyz));
    var mask = 0u;
    for (var c = 0u; c < cascades; c++) {
        if (model.shadow_count[c] == 0u) {
            continue;
        }
        let box = cull.cascades[c];
        let reach = box.z + radius;
        if (abs(light.x - box.x) > reach || abs(light.y - box.y) > reach) {
            continue;
        }
        // Beyond the far face it shades nothing in the box; nearer than the
        // near face it would be clipped away.
        if (light.z - radius > box.w || light.z + radius < cull.cascade_near[c]) {
            continue;
        }
        mask |= 1u << c;
    }
    return mask;
}

@compute @workgroup_size(64)
fn cull_plants(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= cull.counts.x) {
        return;
    }
    let plant = plants[id.x];
    let model = models[plant.model];
    if (model.draw_count[0] == 0u) {
        return;
    }
    let height = model.height * plant.scale;
    let eye = cull.camera.xyz;
    let flat_distance = length(plant.position.xz - eye.xz);
    // The far edge breathes a little per plant, so the forest does not end
    // on a circle.
    let reach = model.max_distance * (0.93 + 0.14 * fract(plant.seed * 7.31));
    if (flat_distance > reach) {
        return;
    }
    let radius = max(height * 0.5, model.bound_radius * plant.scale) + cull.ground.w;
    let centre = vec3<f32>(plant.position.x, plant.position.y + height * 0.5, plant.position.z);
    let seen = sphereVisible(centre, radius);
    let shadows = shadowCascades(centre, radius, model);
    if (!seen && shadows == 0u) {
        return;
    }

    // Seat the root.
    let xz = plant.position.xz;
    var ground = 0.0;
    // Nothing grows in a river or a lake, nor on the margin its floods
    // scour: a tree or shrub stands back from the waterline by a few metres
    // and its crown, a ground plant by a metre and a half.
    // A broad tree stands back by most of its crown as well (crown_clearance
    // in scatter.rs), so it leans over the water rather than across it.
    let footing = select(max(4.0 + 0.06 * height, 0.75 * model.crown_radius * plant.scale + 1.5),
                         max(1.5, model.crown_radius * plant.scale + 1.0),
                         model.habitat == 1u);
    let river = riverEnvelope(xz);
    var bank = river.bank_distance;
    if (river.lake > RIVER_NO_LAKE) {
        bank = riverBankAt(river, terrainHeight(xz));
    }
    if (bank < footing) {
        return;
    }
    if (model.habitat == 1u) {
        if (cull.counts.z == 0u) {
            return;
        }
        let uv = habitatUV(xz);
        if (any(uv < vec2<f32>(0.02)) || any(uv > vec2<f32>(0.98))) {
            return;
        }
        // Wherever the grass itself would root: turf, even with soil showing
        // through it, never sand, rock, gravel, snow, a furrow or a cliff.
        if (habitatAllowed(xz) < SMALL_PLANT_ROOT) {
            return;
        }
        let footprint = max(0.12, model.crown_radius * plant.scale * 0.5);
        let around = min(min(habitatAllowed(xz + vec2<f32>(footprint, 0.0)),
                             habitatAllowed(xz - vec2<f32>(footprint, 0.0))),
                         min(habitatAllowed(xz + vec2<f32>(0.0, footprint)),
                             habitatAllowed(xz - vec2<f32>(0.0, footprint))));
        if (around < 0.5 * SMALL_PLANT_ROOT) {
            return;
        }
        ground = textureSampleLevel(habitat, habitat_sampler, uv, 0.0).r - 0.02;
    } else {
        let centre_height = terrainHeight(xz);
        if (centre_height < cull.ground.x) {
            return;
        }
        let step = 2.5;
        let east = terrainHeight(xz + vec2<f32>(step, 0.0));
        let west = terrainHeight(xz - vec2<f32>(step, 0.0));
        let north = terrainHeight(xz + vec2<f32>(0.0, step));
        let south = terrainHeight(xz - vec2<f32>(0.0, step));
        let steepness = length(vec2<f32>(east - west, north - south)) / (2.0 * step);
        if (steepness > cull.ground.y) {
            return;
        }
        // An incised channel: the root sits well below the ground around it.
        if (0.25 * (east + west + north + south) - centre_height > cull.ground.z) {
            return;
        }
        // Sink the trunk into a slope so its downhill side does not float.
        let trunk = 0.05 + 0.014 * height;
        ground = centre_height - 0.06 - steepness * trunk * 1.5;
    }
    let root = vec3<f32>(xz.x, ground, xz.y);

    for (var c = 0u; c < 4u; c++) {
        if ((shadows & (1u << c)) != 0u) {
            append(plant, model, LOD_SLOTS + c, model.shadow_word[c], model.shadow_count[c], root, 1.0);
        }
    }
    if (!seen) {
        return;
    }

    // Visibility at the far edge.
    let edge_start = reach * (1.0 - EDGE_FADE_BAND);
    let visibility = 1.0 - clamp((flat_distance - edge_start) / (reach - edge_start), 0.0, 0.999);

    // LOD by distance, in plant heights, from the eye to the crown. Each
    // plant switches at its own slightly different distance, so a band edge
    // is never a visible ring.
    let crown = vec3<f32>(root.x, root.y + height * 0.55, root.z);
    let units = distance(eye, crown) / max(height * cull.camera.w, 0.05)
              / (0.92 + 0.16 * fract(plant.seed * 13.37));
    let last = model.lod_count - 1u;
    var lod = last;
    for (var k = 0u; k < last; k++) {
        if (units < model.lod_end[k]) {
            lod = k;
            break;
        }
    }
    if (lod < last) {
        let end = model.lod_end[lod];
        let t = clamp((units - end * (1.0 - LOD_FADE_BAND)) / (end * LOD_FADE_BAND), 0.0, 1.0);
        if (t > 0.0) {
            emit(plant, model, lod, root, -t);
            emit(plant, model, lod + 1u, root, t);
            return;
        }
    }
    emit(plant, model, lod, root, visibility);
}
