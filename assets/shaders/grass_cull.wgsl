// Evaluate captured terrain and canopy once per candidate root, then compact
// survivors into their model's vertex buffer and increment its indirect count.
struct GrassInstance {
    xz: vec2<f32>,
    rotation: f32,
    scale: f32,
    tint: f32,
    seed: f32,
    scatter_data: vec2<f32>,
};
struct GrassCullJob {
    input_first: u32,
    count: u32,
    output_first: u32,
    draw_word: u32,
    radius: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};
struct GrassDrawInstance {
    xz: vec2<f32>,
    rotation: f32,
    scale: f32,
    tint: f32,
    seed: f32,
    ground_height: f32,
    slope_x: f32,
    ground_average: vec3<f32>,
    slope_z: f32,
};
@group(0) @binding(0) var<storage, read> candidates: array<GrassInstance>;
@group(0) @binding(1) var<storage, read> jobs: array<GrassCullJob>;
@group(0) @binding(2) var<storage, read_write> visible: array<GrassDrawInstance>;
@group(0) @binding(3) var<storage, read_write> draw_args: array<atomic<u32>>;
struct GrassFrame {
    mapping: vec4<f32>, // centre XZ, span, metres per texel
    wind: vec4<f32>,    // wind direction XZ, time, strength
    range: vec4<f32>,   // full density distance, draw radius, camera XZ
    layer_end: vec4<f32>,
};
@group(1) @binding(0) var habitat: texture_2d<f32>;
@group(1) @binding(1) var habitat_sampler: sampler;
@group(1) @binding(2) var<uniform> frame: GrassFrame;
@group(1) @binding(3) var grass_ground_albedo: texture_2d<f32>;

fn habitatUV(xz: vec2<f32>) -> vec2<f32> {
    return (xz - frame.mapping.xy) / frame.mapping.z + 0.5;
}
fn groundAt(xz: vec2<f32>) -> vec4<f32> {
    return textureSampleLevel(habitat, habitat_sampler, habitatUV(xz), 0.0);
}
// Read the least suitable texel, never an average across a forbidden boundary.
fn allowedAt(xz: vec2<f32>) -> f32 {
    let size = vec2<i32>(textureDimensions(habitat));
    let cell = vec2<i32>(floor(habitatUV(xz) * vec2<f32>(size) - 0.5));
    if (any(cell < vec2<i32>(0)) || any(cell + 1 >= size)) { return 0.0; }
    // Taking the minimum of the four neighbouring texels keeps forbidden
    // surfaces from being blurred into eligible roots at boundaries.
    let four = textureGather(1u, habitat, habitat_sampler, habitatUV(xz));
    return min(min(four.x, four.y), min(four.z, four.w));
}

@compute @workgroup_size(64)
fn cull_grass(
    @builtin(workgroup_id) workgroup: vec3<u32>,
    @builtin(local_invocation_index) local: u32,
) {
    let job = jobs[workgroup.x];
    if (local >= job.count) { return; }
    let instance = candidates[job.input_first + local];
    let root = instance.xz;
    let seed = instance.seed;
    let scatter_data = instance.scatter_data;
    let distance = length(root - frame.range.zw);
    if (distance >= frame.range.y) { return; }
    // Continuous, nested density reduction: every independent imported model
    // remains itself. Seed and position do not change on camera-cell crossings.
    // Keep a living outer meadow: distant clumps become sparser gradually,
    // but retain enough full-size silhouettes to read against the terrain.
    let density = mix(1.0, 0.78, smoothstep(frame.range.x, frame.range.y, distance)) * scatter_data.x;
    let survival = smoothstep(seed - 0.10, seed + 0.10, density);
    let fade = 1.0 - smoothstep(frame.range.y - 22.0, frame.range.y, distance);
    // Two gradually thinning middle tiers keep the valley planted without
    // sending foreground-level card counts into the far field. Candidate
    // rings extend past their zero-growth edges across camera-anchor moves.
    let far_medium = 1.0 - smoothstep(frame.layer_end.x - 30.0, frame.layer_end.x, distance);
    let near_medium = 1.0 - smoothstep(frame.layer_end.y - 30.0, frame.layer_end.y, distance);
    let carpet = 1.0 - smoothstep(frame.layer_end.z - 8.0, frame.layer_end.z, distance);
    let layer = select(1.0,
                       select(far_medium,
                              select(near_medium, carpet, scatter_data.y > 2.5),
                              scatter_data.y > 1.5),
                       scatter_data.y > 0.5);
    // The ground-colour texture's alpha is the share of open sky the tree
    // crowns leave over the root (vegetation.wgsl's canopy pass): deep shade
    // under a closed canopy keeps only a few, smaller clumps.
    let ground_colour = textureSampleLevel(grass_ground_albedo, habitat_sampler, habitatUV(root), 0.0);
    let sky = ground_colour.a;
    let shade_survival = smoothstep(seed - 0.12, seed + 0.12, 1.25 * sky - 0.12);
    let potential = survival * fade * layer * shade_survival;
    if (potential < 0.02) { return; }
    let ground = groundAt(root);
    let radius = max(0.12, job.radius * instance.scale);
    var suitability = allowedAt(root);
    if (suitability < 0.02) { return; }
    suitability = min(suitability, allowedAt(root + vec2<f32>(radius, 0.0)));
    suitability = min(suitability, allowedAt(root - vec2<f32>(radius, 0.0)));
    suitability = min(suitability, allowedAt(root + vec2<f32>(0.0, radius)));
    suitability = min(suitability, allowedAt(root - vec2<f32>(0.0, radius)));
    let growth = suitability * potential * mix(0.6, 1.0, sky);
    if (growth < 0.02) { return; }
    let up = normalize(vec3<f32>(ground.b, sqrt(max(0.01, 1.0 - dot(ground.ba, ground.ba))), ground.a));
    let slope = up.xz / max(up.y, 0.5);
    let output = job.output_first + atomicAdd(&draw_args[job.draw_word], 1u);
    visible[output] = GrassDrawInstance(
        root, instance.rotation, instance.scale * growth, instance.tint, seed,
        ground.r - 0.015, slope.x, ground_colour.rgb, slope.y,
    );
}
