// PORT NOTES (composite.fs -> composite.wgsl):
// - Deferred composite pass: lighting from the g-buffers + blurred SSAO +
//   fog + ACES tonemap + gamma encode. The output stays raw gamma-encoded
//   bytes (the pow(1.0/2.2) is the final write of BOTH paths) because FXAA
//   consumes them; nothing re-linearizes the result after this pass.
// - HOISTED SAMPLES: the GLSL read the G-buffers after the early sky return
//   (`if (packedPosition.a < 0.5) return;`), which leaves every statement
//   after it in non-uniform control flow — WGSL forbids implicit-derivative
//   sampling (textureSample) there. All four reads moved unconditionally
//   above the branch; the sky path simply ignores them, and every decode
//   computation keeps its original place, order and value. No flips were
//   dropped (the source has none): fragTexCoord derives from
//   @builtin(position) as position.xy * globals.viewport.zw. The G-buffer
//   reads are a same-frame blit, so GL's bottom-left gl_FragCoord origin
//   cancels out there; the analytic sky gradient is not, and takes a
//   GL-oriented v = 1 - uv.y (see sky_uv in fs_main). The sky/terrain
//   branch keys off texture DATA (packedPosition.a), not screen
//   orientation.
// - texture1 was sampled twice in the GLSL (xyz for the normal, .a for
//   roughness); one sample of the same texture at the same uv now feeds
//   both reads — bit-identical values, one fewer fetch.
// - Uniform mapping (the shared GlobalUniforms preamble owns these four):
//   uFogDensity -> globals.params.x, uExposure -> globals.params.z,
//   uSunIntensity -> globals.settings_a.x, uAoTexStrength ->
//   globals.settings_a.z. uLightDirectionView and uAoStrength exist in no
//   globals field, so they stay in StageUniforms (group 2). No inverse_*
//   uniform fields were added (nothing inverts a matrix in this pass).
// - uLightDirectionView is VIEW-space (the raylib build rotated the world
//   sun direction by matView on the CPU); it is a different value from
//   globals.sun_direction, which is WORLD-space. Rust can reproduce it with
//   (globals.view * vec4<f32>(globals.sun_direction.xyz, 0.0)).xyz.
// - fragColor was declared but never used in the GLSL; it is not carried
//   over. Single output: @location(0) vec4<f32> (finalColor), no channel
//   dropped or re-ordered.

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

struct StageUniforms {                // uLightDirectionView, uAoStrength
    // vec4, NOT vec3. WGSL packs the following f32 into the vec3's 12-byte
    // slot, so `ao_strength` lands at offset 12; the Rust writer's
    // #[repr(C, align(16))] CompositeStageUniforms declares
    // light_direction_view as [f32; 4] (w unused) and writes ao_strength at
    // offset 16. As a vec3 the shader read the pad float instead — a constant
    // 0.0 — so `mix(1.0, ssao, 0.0)` dropped the blurred SSAO from every lit
    // pixel while still paying for the pass. naga's layouter reports the two
    // layouts; `cargo run --bin wgsl-layout -- assets/shaders/composite.wgsl`
    // is how this was caught. The C++ has no packing to get wrong: GLSL
    // uLightDirectionView and uAoStrength are separate uniforms with separate
    // locations.
    light_direction_view: vec4<f32>,  // uLightDirectionView: view-space direction travelled by sunlight (w unused)
    ao_strength: f32,                 // uAoStrength: SSAO blend strength (clamped to [0,1] in-shader)
};
@group(2) @binding(0) var<uniform> stage: StageUniforms;

@group(1) @binding(0) var texture0: texture_2d<f32>;   // View-space position, alpha: 0 sky, 1 geometry,
                                                       // 1 + snow mask (see terrain.fs gPosition).
@group(1) @binding(8) var texture0_sampler: sampler;
@group(1) @binding(1) var texture1: texture_2d<f32>;   // Encoded view-space normal (.a = roughness).
@group(1) @binding(9) var texture1_sampler: sampler;
@group(1) @binding(2) var texture2: texture_2d<f32>;   // sqrt-encoded albedo, 2.5x headroom (.a = texture AO).
@group(1) @binding(10) var texture2_sampler: sampler;
@group(1) @binding(3) var texture3: texture_2d<f32>;   // Bilaterally blurred SSAO.
@group(1) @binding(11) var texture3_sampler: sampler;

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

const PI: f32 = 3.14159265;

// GGX distribution and the height-correlated Smith visibility term. Together
// with the Schlick fresnel below they form a physically based specular lobe
// that stays energy-conserving across the whole roughness range.
fn D_GGX(n_dot_h: f32, alpha: f32) -> f32
{
    let a2 = alpha*alpha;
    let d = n_dot_h*n_dot_h*(a2 - 1.0) + 1.0;
    return a2/max(d*d, 1e-7);
}

fn V_SmithGGX(n_dot_l: f32, n_dot_v: f32, alpha: f32) -> f32
{
    let a2 = alpha*alpha;
    let ggxL = n_dot_v*sqrt(n_dot_l*n_dot_l*(1.0 - a2) + a2);
    let ggxV = n_dot_l*sqrt(n_dot_v*n_dot_v*(1.0 - a2) + a2);
    return 0.5/max(ggxL + ggxV, 1e-7);
}

fn skyColour(uv: vec2<f32>) -> vec3<f32>
{
    let elevation = smoothstep(0.0, 1.0, uv.y);
    let horizon = vec3<f32>(0.72, 0.82, 0.90);
    let zenith = vec3<f32>(0.22, 0.46, 0.74);
    return mix(horizon, zenith, elevation);
}

// Narkowicz's fitted ACES filmic curve. A tonemap replaces the raw gamma
// encode: sun-facing snow and sun glints carry HDR values above 1.0 that a
// plain gamma curve would clip into flat white patches, while the filmic
// shoulder rolls them off with the desaturating compress real cameras show.
// Sky pixels run through the same curve so fogged terrain converges to
// exactly the sky behind it instead of banding at the horizon.
fn acesFilm(x: vec3<f32>) -> vec3<f32>
{
    const a: f32 = 2.51;
    const b: f32 = 0.03;
    const c: f32 = 2.43;
    const d: f32 = 0.59;
    const e: f32 = 0.14;
    // WGSL's clamp has no scalar-bounds vector overload, so the GLSL
    // clamp(v, 0.0, 1.0) needs explicitly splatted bounds.
    return clamp((x*(a*x + b))/(x*(c*x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs_main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32>
{
    let uv = position.xy * globals.viewport.zw;
    // The G-buffer reads below are a same-frame blit and share wgpu's
    // top-down origin with the passes that wrote them, so they use `uv`
    // unchanged. The sky gradient is NOT a blit: the GLSL evaluated
    // skyColour(gl_FragCoord.xy * uViewport.zw) with GL's bottom-up
    // fragment origin, so v = 0 was the horizon and v = 1 the zenith.
    // @builtin(position).y counts down from the top, so the gradient is
    // fed the GL-oriented v to keep zenith up.
    let sky_uv = vec2<f32>(uv.x, 1.0 - uv.y);

    // Uniform mapping (see PORT NOTES): uFogDensity -> globals.params.x,
    // uExposure -> globals.params.z, uSunIntensity -> globals.settings_a.x,
    // uAoTexStrength -> globals.settings_a.z; stage.light_direction_view and
    // stage.ao_strength carry uLightDirectionView and uAoStrength.
    let packed_position = textureSample(texture0, texture0_sampler, uv);
    let sky = skyColour(sky_uv);
    // The sky constants are authored in display space; decode to linear once
    // so ambient, fog and sky pixels all share one space.
    let sky_lin = pow(sky, vec3<f32>(2.2));

    // HOISTED SAMPLES: the GLSL read these below the early sky return, where
    // WGSL forbids implicit-derivative sampling (non-uniform control flow).
    // They are taken unconditionally here; the sky path ignores them.
    let normal_sample = textureSample(texture1, texture1_sampler, uv);
    let albedo_sample = textureSample(texture2, texture2_sampler, uv);
    let ssao_sample = textureSample(texture3, texture3_sampler, uv);

    if (packed_position.a < 0.5)
    {
        return vec4<f32>(pow(acesFilm(sky_lin*globals.params.z), vec3<f32>(1.0/2.2)), 1.0);
    }

    // The terrain G-buffer packs biome weights above the sky flag: snow
    // quantized to hundredths, grass in the 1e-4 residue (see terrain.fs).
    // Snow is decoded by rounding, NOT by clamping the raw remainder: snow
    // is a continuous field, and with grass packed anywhere inside its own
    // precision a clamped decode absorbs the grass term and the grass
    // decode below returns identically zero — which is what the original
    // packing did. grassMask stuck at 0 meant the specular damper never
    // engaged (full Fresnel GGX on every meadow: the pale sunward wash)
    // and the anti-solar warm-up stayed dead. Snow's mask drives its
    // multiply-scattered ambient; grass's damps the grazing Fresnel ramp
    // that would otherwise paint a plastic sheen over meadows.
    let snow_mask = clamp(floor((packed_position.a - 1.0)*100.0 + 0.5)*0.01,
                           0.0, 1.0);
    let grass_mask = clamp((packed_position.a - 1.0 - snow_mask)*10000.0,
                            0.0, 1.0);

    let normal_view = normalize(normal_sample.xyz*2.0 - 1.0);
    let roughness = clamp(normal_sample.a, 0.0, 1.0);
    // Albedo is sqrt-encoded against 2.5 of headroom by the G-buffer
    // writers (see terrain.fs): sunlit snow's tinted albedo rides above
    // 1.0, which a linear 8-bit write would clamp away — deleting the
    // snow mottle/grain albedo texture and pinning the pack's brightness.
    // Square the read back out of the encoded range.
    let albedo = albedo_sample.rgb*albedo_sample.rgb*2.5;
    let texture_ao = clamp(albedo_sample.a, 0.0, 1.0);
    let ssao = ssao_sample.r;
    let ao_factor = mix(1.0, ssao, clamp(stage.ao_strength, 0.0, 1.0))
                   * mix(1.0, texture_ao, clamp(globals.settings_a.z, 0.0, 1.0));

    let light_direction = normalize(stage.light_direction_view.xyz);
    let to_light = -light_direction;
    let to_camera = normalize(-packed_position.xyz);
    let half_vector = normalize(to_light + to_camera);

    let n_dot_l = max(dot(normal_view, to_light), 0.0);
    let n_dot_h = max(dot(normal_view, half_vector), 0.0);
    let n_dot_v = max(dot(normal_view, to_camera), 0.0);
    let v_dot_h = max(dot(to_camera, half_vector), 0.0);

    // Roughness floor tames sub-pixel highlight aliasing on the smoothest
    // surfaces (wet ground, the ocean, snow glints); alpha = roughness^2
    // by convention.
    var alpha = clamp(roughness, 0.06, 1.0);
    alpha *= alpha;

    // Warm noon sun. Lambert diffuse plus GGX specular, dielectric fresnel
    // (F0 0.04) — terrain, water and rock are all non-metals. With
    // uSunIntensity at pi the diffuse at normal incidence equals the albedo,
    // so sun-facing slopes hold the brightness of the previous half-Lambert.
    let sun_colour = vec3<f32>(1.0, 0.955, 0.90)*globals.settings_a.x;
    let fresnel = vec3<f32>(0.04) + vec3<f32>(0.96)*pow(1.0 - v_dot_h, 5.0);
    // A grass canopy is a volume of rough blades that mutually shadow each
    // other's highlights: measured meadow BRDFs are diffuse-dominant, and
    // the GGX lobe is only kept at all for the wet blades near waterlines.
    // The grazing-angle Fresnel ramp paints a broad desaturated sheen band
    // across every near hillside otherwise — the "too shiny" plastic wash
    // — so vegetation specular runs near zero.
    let specular = D_GGX(n_dot_h, alpha)*V_SmithGGX(n_dot_l, n_dot_v, alpha)*fresnel
                   *mix(1.0, 0.10, grass_mask);

    // Snow is a translucent pack: light diffuses through the top centimetres,
    // so its sun-to-shadow terminator stays soft. The wrap touches diffuse
    // only — snow's glints come from the sparkle path's mirror facets.
    let wrap = 0.22*snow_mask;
    let n_dot_lw = clamp((n_dot_l + wrap)/(1.0 + wrap), 0.0, 1.0);
    var lit = (albedo/PI)*n_dot_lw*sun_colour + specular*n_dot_l*sun_colour;

    let view_distance = length(packed_position.xyz);

    // Vegetation backscatter: looking sunward across a meadow shows a bright
    // retroreflective wash — the anti-solar hotspot from blade geometry and
    // translucency. This is the photographic grass signature the suppressed
    // GGX lobe used to fake as sheen. The exponent is softer than the
    // literature's tight hotspot because this sun sits high: at 60 degrees
    // elevation the phase angle never approaches zero, so a broad, gentle
    // directional warm-up is all the geometry can express. Two brakes keep
    // it honest: the amplitude stays low — at full strength the additive
    // lift washes ground out to a milky pastel that reads as fog lying on
    // the field — and it is distance-gated, because the hotspot belongs to
    // receding canopy while the nearest ground must keep its full
    // saturation (vision rounds measured the ungated term as a pale
    // desaturated wash across the sunward side of the closest metres, and
    // as chartreuse drift where it stacked with warm-tinted clumps).
    let v_dot_l = max(dot(to_camera, to_light), 0.0);
    let hotspot_gate = smoothstep(15.0, 45.0, view_distance);
    lit += albedo*sun_colour*pow(v_dot_l, 3.0)*0.048*hotspot_gate*grass_mask*max(n_dot_l, 0.15);

    // AO primarily modulates indirect light; retaining a little direct light
    // prevents deep terrain folds from becoming featureless black regions.
    lit *= mix(ao_factor, 1.0, n_dot_l*0.35);

    // Sky ambient fill. Snow is the exception every other material is not:
    // a snowpack is strongly multiply-scattering, so its shaded side stays
    // bright and — because ice preferentially absorbs red — sky-blue
    // instead of falling toward black; that blue shadow tint is what makes
    // snow read as snow. The mix keeps the cast mostly on the shaded side:
    // sunlit faces are dominated by the warm direct term and read near
    // neutral, as photographed snow does. Ambient stays a minority of the
    // sun term so drifts and sastrugi keep visible sun-side/lee-side
    // contrast instead of washing into a flat field. Grass ambient is
    // canopy-filtered skylight, not open-sky azure: a woodland floor's
    // fill light has already scattered through leaves, so the blue is
    // tempered toward green and the fill runs slightly weaker — open-sky
    // ambient painted a pale blue wash over near grass and, stacked with
    // the warm-up, pushed the nearest field toward chartreuse. The snow
    // path is unchanged (same 0.45 blue mix as before).
    let ambient_strength = mix(0.12, 0.28, snow_mask);
    let ambient_colour = mix(sky_lin*vec3<f32>(0.80, 1.00, 0.68),
                             mix(sky_lin, sky_lin*vec3<f32>(0.70, 0.90, 1.42), 0.45),
                             snow_mask);
    lit += albedo*ambient_colour*ambient_strength*ao_factor;

    // Vegetation gains aerial perspective faster than bare ground: a
    // canopy scatters the light a second time on the way out, so distant
    // woodland washes toward haze sooner than rock or snow does. Round 7
    // measured the spawn grass plain flat from horizon to feet (sat 0.581
    // far vs 0.570 near, far luma no brighter) — no recession at all. The
    // fog rate rises ~45% for grass only; snow keeps its rate (its
    // verdicts pass, and its fog-relieved glints must not dim), and the
    // nearest metres stay untouched either way.
    var fog = 1.0 - exp(-max(globals.params.x, 0.0)*(1.0 + 0.45*grass_mask)
                          *view_distance);
    // Aerial perspective fades terrain into the same linear sky the sky
    // pixels display, so distant land dissolves into the haze behind it.
    // Surviving sun glints are the exception: a mirror facet's specular
    // spike rides orders of magnitude above the diffuse field, and light
    // haze attenuates such a spike without dissolving it into sky — the
    // sun path on distant water stays visible through haze in real
    // photographs. The sparkle path writes roughness 0.13 (the flash lobe
    // widened to match its acceptance window, see terrain.fs) and wet
    // ground/water writes 0.10-0.12; the window starts above both so those
    // tight-mirror pixels halve their fog pull and distant sunlit
    // snowfields keep their sparse shimmer instead of fogging flat (round
    // 6 measured the far ridge at 2 px of surviving glint over a 650x340
    // lit-snow region).
    fog *= 1.0 - 0.55*(1.0 - smoothstep(0.13, 0.22, roughness));
    lit = mix(lit, sky_lin, clamp(fog, 0.0, 1.0));

    return vec4<f32>(pow(max(acesFilm(lit*globals.params.z), vec3<f32>(0.0)), vec3<f32>(1.0/2.2)), 1.0);
}

// STAGE UNIFORMS:
//   light_direction_view : vec4<f32> - uLightDirectionView: view-space direction
//       travelled by sunlight (already rotated by the view matrix; NOT the
//       world-space globals.sun_direction). Rust can fill it with
//       (globals.view * vec4<f32>(globals.sun_direction.xyz, 0.0)); only .xyz
//       is read, w exists to keep the next member at offset 16.
//   ao_strength : f32 - uAoStrength: SSAO blend strength, clamped to [0,1]
//       in-shader (1.0 = full blurred-SSAO weighting, 0.0 = disabled).
// Rust fill (uniform, 32 bytes, matching CompositeStageUniforms exactly):
//   [0]  light_direction_view: [f32; 4] (w = 0.0),
//   [16] ao_strength: f32, [20] pad: [f32; 3]
// Composite uniforms carried by the SHARED GlobalUniforms (group 0/binding 0):
//   params.x = uFogDensity, params.z = uExposure,
//   settings_a.x = uSunIntensity, settings_a.z = uAoTexStrength
// Textures (group 1, all linear-filtered, samplers at binding + 8):
//   0 = gPosition (a = sky flag + packed biome masks),
//   1 = gNormal (rgb view-space*0.5+0.5, a = roughness),
//   2 = gAlbedo (sqrt-encoded rgb, a = texture AO),
//   3 = bilaterally blurred SSAO (r channel)
// Output: location 0 = gamma-encoded RGBA8 — the exact bytes FXAA consumes.