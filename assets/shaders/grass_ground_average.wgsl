// Downsample the terrain's unlit linear albedo once per frame. A rooted grass
// clump then samples this prefiltered colour only once, avoiding a neighborhood
// of terrain fetches for every vertex in a dense field.
@group(0) @binding(0) var ground_albedo: texture_2d<f32>;

@vertex
fn vs_main(@builtin(vertex_index) vertex: u32) -> @builtin(position) vec4<f32> {
    let xy = vec2<f32>(f32((vertex << 1u) & 2u), f32(vertex & 2u));
    return vec4<f32>(xy.x * 2.0 - 1.0, 1.0 - xy.y * 2.0, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let source_xy = vec2<i32>(position.xy) * 2;
    var colour = vec3<f32>(0.0);
    var coverage = 0.0;
    for (var y = 0; y < 2; y++) {
        for (var x = 0; x < 2; x++) {
            let sample = textureLoad(ground_albedo, source_xy + vec2<i32>(x, y), 0);
            // The terrain capture clears to transparent beyond its triangles.
            // Do not pull that black into grass at the edge of a valid texel.
            colour += sample.rgb * sample.a;
            coverage += sample.a;
        }
    }
    return vec4<f32>(colour / max(coverage, 1e-6), clamp(coverage * 0.25, 0.0, 1.0));
}
