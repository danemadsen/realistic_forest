# GLSL → WGSL porting conventions (Forest renderer port)

Port a raylib/GL 4.3 deferred terrain renderer to WGSL. Every result must be a
**1:1 behavioral port**: identical math, identical constant values, identical
structure. Do not "improve" anything. Preserve the original comments.

## File map

| GLSL source | WGSL output | Entry points |
|---|---|---|
| terrain.vs | terrain-vs.wgsl | `vs_main` (`@vertex`) |
| terrain.fs | terrain-fs.wgsl | `fs_main` (`@fragment`), 3 color outputs |
| basic_gbuffer.vs + .fs | basic-gbuffer.wgsl | `vs_main`, `fs_main` |
| erosion_init.fs | erosion-init.wgsl | `vs_main` (fullscreen tri), `fs_main` |
| erosion_flux.fs | erosion-flux.wgsl | `vs_main`, `fs_main` |
| erosion_water.fs | erosion-water.wgsl | `vs_main`, `fs_main` |
| erosion_terrain.fs | erosion-terrain.wgsl | `vs_main`, `fs_main` |
| ssao.fs | ssao.wgsl | `vs_main`, `fs_main` |
| ssao_blur.fs | ssao-blur.wgsl | `vs_main`, `fs_main` |
| composite.fs | composite.wgsl | `vs_main`, `fs_main` |
| fxaa.fs | fxaa.wgsl | `vs_main`, `fs_main` |

GLSL sources: `/Users/danemadsen/forest/assets/shaders/`. WGSL outputs go to
`/Users/danemadsen/realistic_forest/assets/shaders/`.

After writing each file, validate it:

```
cargo run -q --bin wgsl-check <path>     # run in /Users/danemadsen/realistic_forest
```

Note: this validator requires a `vs_main` entry point even on files that only
have a fragment stage when paired separately — terrain-vs.wgsl and
terrain-fs.wgsl each have only one entry; add a no-op dummy entry of the
missing kind at the bottom if needed, e.g.:

```wgsl
@vertex fn fs_main_dummy() -> @builtin(position) vec4<f32> { return vec4<f32>(); }
```

Only add dummies when the real file is otherwise one-stage; keep real files
self-contained (post passes include the fullscreen-tri vertex below).

### Sim shaders run on their own device (hot reload does not reach them)

Since the erosion tile simulation moved to a dedicated worker thread
(`src/erosion_worker.rs`), the four sim WGSLs above (plus `erosion-thermal.wgsl`)
are compiled by that worker with raw `wgpu` from the source TEXT the main world
publishes through the erosion bridge — not through bevy's `Shader` asset +
`RenderPipeline` path. The consequences:

- The worker compiles once at startup, when it opens its own device on the
  renderer's adapter. Bevy's shader-asset hot reload never runs naga over these
  files again, so editing an erosion sim shader needs a restart to take effect.
- The files must stay fully self-contained: no `#import` preprocessing, because
  the worker takes the raw file bytes (`create_shader_module` on WGSL text).
- The main world still validates them: `cargo test erosion_sim_shaders` parses
  all four from `assets/shaders/`, and an end-to-end worker test compiles and
  runs them on the real GPU.

## 1. Shared global uniforms — paste this preamble into EVERY file

The vertex-data and the camera matrices come from one uniform buffer bound at
group 0. All shaders declare the same struct; use only the fields the original
GLSL pulled from raylib's built-ins (`matView`, `matProjection`,
`uCameraPosition`...).

```wgsl
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
```

`matModel` and every other non-camera uniform go into a per-pipeline stage
uniform (below). The projection matrix that arrives in `globals.projection`
already maps the near plane to NDC z = 0 and the far plane to z = 1 (the GL
matrix has been halved/shifted on the CPU). Any linearization math that the
GLSL performs on a depth value or on `uProjection`-style matrices must account
for this ONLY if it inverts depth itself; reconstruct-from-g-buffer position
math ports unchanged.

## 2. Per-stage uniforms — group 2

Collect every remaining scalar/vector uniform of the original into one struct
named `StageUniforms`, bound at:

```wgsl
@group(2) @binding(0) var<uniform> stage: StageUniforms;
```

Name fields after the original uniforms, lowercased with GLSL `uFooBar` →
`foo_bar`. Example:

```wgsl
struct StageUniforms {           // uSeaLevel, uTexScale, uFlowDebug
    sea_level: f32,              // uSeaLevel
    tex_scale: f32,              // uTexScale
};
```

Start the file's StageUniforms declaration with a comment listing the
provenance of every field, and end the file with a `// STAGE UNIFORMS:` block
summarizing name → meaning + type, so the Rust pipeline code can fill it.

If a uniform's natural value is known at compile time (e.g. a constant equal
to an engine constant such as sea level = 0.0), still keep it as a uniform
field — the Rust side owns the constants.

## 3. Texture bindings — group 1 (GL unit numbers preserved)

Every GLSL `sampler2D`/`sampler2DArray` becomes a texture + sampler pair in
group 1. Binding numbers EQUAL the GL texture unit the C++ assigned:

- `textureN` samplers → texture at binding `N`, sampler at binding `N + 8`.
- Named samplers (`uAlbedoAO`, `uNormalRough`, `uFluxState`, `uTerrainState`,
  `uWaterState`) get the unit listed below, sampler at unit + 8:

| File | texture | binding |
|---|---|---|
| terrain-fs | texture0 (R32 base noise) | 0 |
| terrain-fs | texture1 | 1 |
| terrain-fs | texture2 | 2 |
| terrain-fs | texture3 (RGBA32F tile lookup, point) | 3 |
| terrain-fs | texture4 (blend mask) | 4 |
| terrain-fs | uAlbedoAO (texture_2d_array) | 5 |
| terrain-fs | uNormalRough (texture_2d_array) | 6 |
| ssao | texture0/1/2 | 0/1/2 |
| ssao-blur | texture0/1/2 | 0/1/2 |
| composite | texture0/1/2/3 | 0/1/2/3 |
| fxaa | texture0 | 0 |
| basic-gbuffer | (none) | — |
| erosion-init | texture0 (base height RGBA32F) | 0 |
| erosion-flux | texture0 (flux being updated) 0, uTerrainState 1, uWaterState 2 | — |
| erosion-water | texture0 (water being updated) 0, uFluxState 1, uTerrainState 2 | — |
| erosion-terrain | texture0 (terrain being updated) 0, uWaterState 1 | — |

Sampler declarations: `@group(1) @binding(8) var tex0_sampler: sampler;` etc.
Filtering mode comes from the GL comments in the shader (point-filtered lookups
get a non-filtering sampler in the Rust side; all others linear). In WGSL you
write `sampler` for both — the Rust bind group picks the mode.

`texture2DArray` → `texture_2d_array<f32>`; `layer()` selection stays the
integer layer index.

## 4. Fullscreen-triangle vertex shader (post passes + erosion passes)

The GLSL post passes rely on raylib blits; in WGSL every fullscreen pass
vertex stage is:

```wgsl
struct VsOutput {
    @builtin(position) position: vec4<f32>,
};
@vertex
fn vs_main(@builtin(vertex_index) vertex: u32) -> VsOutput {
    var corner = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    var output: VsOutput;
    output.position = vec4<f32>(corner[vertex], 0.0, 1.0);
    return output;
}
```

In the fragment stage, derive what GLSL got from raylib's UV interpolators as:

```wgsl
let uv = position.xy * globals.viewport.zw;            // position is @builtin(position)
let texel = vec2<i32>(position.xy);                    // integer texel == GL (int)gl_FragCoord.xy
```

Because every texture, attachment, and readback in the WGSL port shares ONE
top-left-origin row convention (wgpu), do NOT port GL/raylib vertical flips.
If the GLSL flips a uv for a render-texture blit (`vec2(uv.x, 1.0 - uv.y)` on
a texture that was rendered in the same frame), DROP the flip and note it in
`// PORT NOTES:` at the top of the file.

## 5. GLSL → WGSL translation table

- `vec2/3/4` → `vec2<f32>`/`vec3<f32>`/`vec4<f32>`; `ivec*` → `vec*<i32>`;
  `uvec*` → `vec*<u32>`; `mat4` → `mat4x4<f32>`; `mat3` → `mat3x3<f32>`;
  `mat3(mat4)` → `mat3x3<f32>(m[0].xyz, m[1].xyz, m[2].xyz)`.
- `texture()` → `textureSample(tex, sam, uv)` in fragment stages;
  `textureSampleLevel(tex, sam, uv, 0.0)` in vertex stages (no implicit
  derivatives there). `textureLod()` → `textureSampleLevel(tex, sam, uv, lod)`.
- `textureGrad(tex, sam, uv, dPdx, dPdy)` → `textureSample(tex, sam, uv)`
  (implicit derivatives carry the same anisotropic filtering); ONLY hoist
  samples out of non-uniform branches when the WGSL uniform-control-flow rule
  forbids the original placement. When you must hoist, pre-sample
  unconditionally before the branch and keep selection inside.
  `textureSample`/`dpdx`/`dpdy` are illegal inside non-uniform control flow;
  WGSL's rule is stricter than GLSL's. If you need `textureGrad`'s derivative
  parameters for a mathematically different mapping, describe the deviation in
  PORT NOTES instead of silently changing it.
- `texelFetch(t, iv, 0)` → `textureLoad(t, vec2<i32>(iv), 0)`.
- `gl_FragCoord.xy` → `position.xy`; y originates TOP-LEFT in WGSL (GL is
  bottom-left). Every use in these shaders is screen-pattern math on
  same-orientation textures, so port coordinates unchanged (see the no-flip
  rule above) and note any place where the handedness actually matters.
- `gl_FragData[k]`/`out vecN` → `@location(k)` outputs.
- `discard` → `discard;`.
- `mod(a, b)` → `a - b * floor(a / b)`.
- `inversesqrt` → `inverseSqrt`. `fract` → `fract`. `mix` → `mix`.
  `smoothstep` → `smoothstep`. `clamp` → `clamp`. `atan(y, x)` → `atan2(y, x)`.
  `mat2/3` constructors → `matNxM<f32>` column constructor.
- `int(x)` → `i32(x)`, `float(x)` → `f32(x)`, `uint(x)` → `u32(x)`.
- There is NO `inverse()` builtin in WGSL. If the GLSL needs an inverted
  matrix, keep instead a stage-uniform field named `inverse_<thing>` of type
  mat4x4/ mat4x4<f32> (Rust supplies it) and note it in PORT NOTES.
- Integer division/remainder `%` on floats: replace with the `mod` rule above;
  on integers `%` is fine.
- `const` arrays of floats/vectors: use module `const` or `var<private>` —
  both are legal; keep the original's initialization order (deterministic).
- WGSL `if` conditions must be bool: `if (x != 0)` not `if (x)`.

## 6. Semantics that MUST be preserved

- All constants, magic numbers, and epsilon values copied verbatim.
- Comments copied (converted to `//`, translated terms like "GL" kept).
- Function decomposition preserved (helper functions become WGSL `fn`s).
- Uniform control-flow restrictions: if the GLSL samples textures inside
  branches that depend on per-pixel data, restructure to pre-sample before the
  branch; verify the branch's condition truly needs the restructure.
- Loop bounds and iteration ORDER (e.g. erosion neighbor accumulation order)
  must be identical — floating-point sums are order-sensitive.
- Output channels: preserve which output location holds which G-buffer channel
  (position/normal/albedo order), including alpha conventions (sky=0 flags).
- sRGB: the GLSL writes gamma-encoded byte values through linear-space
  shaders; port the encoding steps as-is.

## 7. What each agent returns (structured output)

- `file`: output wgsl path
- `stage_uniforms`: field list with types and their GLSL provenance
- `deviations`: any deviation from 1:1 (hoisting, dropped flips, inverse
  uniforms added), each with a one-line reason
- `notes`: anything the Rust pipeline code must know (e.g. "texture2 is
  point-sampled", "second output is octahedral-encoded")