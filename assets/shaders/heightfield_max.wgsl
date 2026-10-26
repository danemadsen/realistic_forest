// The highest terrain in the lighting heightfield, for exact early exits
// from the sun-visibility marches (terrainShadow and volumeLightVisibility in
// the composite and water passes): once a march toward the light has risen
// above every height in the map, no later tap can be occluded.
//
// Heights below sea level are raised to zero first. Non-negative floats order
// exactly like their bit patterns, so the maximum can be reduced with integer
// atomics, and the raised bound stays an upper bound for every texel. The
// node copies the result straight into globals.sun_direction.w.
@group(0) @binding(0) var heightfield: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> highest: atomic<u32>;

var<workgroup> workgroup_highest: atomic<u32>;

@compute @workgroup_size(16, 16)
fn reduce_highest(@builtin(global_invocation_id) id: vec3<u32>,
                  @builtin(local_invocation_index) index: u32) {
    if (index == 0u) {
        atomicStore(&workgroup_highest, 0u);
    }
    workgroupBarrier();
    let size = textureDimensions(heightfield);
    if (all(id.xy < size)) {
        let raw = textureLoad(heightfield, vec2<i32>(id.xy), 0).r;
        // Clamp sea level, -0.0 and negative heights to +0.0 before padding,
        // so all resulting bit patterns stay ordered.
        // A centimetre of slack also covers floating-point rounding in the
        // filtered height and the receiver's penumbra/clearance calculations.
        let height = select(0.0, raw, raw > 0.0) + 0.01;
        atomicMax(&workgroup_highest, bitcast<u32>(height));
    }
    workgroupBarrier();
    if (index == 0u) {
        atomicMax(&highest, atomicLoad(&workgroup_highest));
    }
}
