# Forest

Forest is a Rust 2024 prototype for an infinite procedural landscape. It renders
mountains, plains, beaches, ocean, and biome colours with an indexed geometry
clipmap. Procedural terrain can be evaluated at any world coordinate, and
overlapping GPU hydraulic-erosion tiles are generated and cached around the
player as the world streams. Walking and flight are not clamped to a simulation
domain: there is no world border.

It is a port of the C++17/raylib/OpenGL project at `~/forest`, keeping the same
math, constants and structure. The renderer is now wgpu through Bevy, the source
noise comes from quick-noise, and the diagnostics panel is egui. Ported shaders
in `assets/shaders` carry `PORT NOTES` headers naming their GLSL sources and
recording deliberate deviations; the atmospheric rendering extends that base.

## Build and run

Requirements:

- A recent stable Rust toolchain (developed against 1.98.1; the crate uses the
  2024 edition)
- A Vulkan, Metal or D3D12 capable GPU

From a fresh checkout:

```sh
cargo run --release --bin realistic_forest
```

A debug build also works and is much faster to compile; the shaders are compiled
by naga at runtime either way.

## Controls

| Input | Action |
| --- | --- |
| Mouse | Look |
| W/A/S/D | Move |
| Space | Jump, or rise while flying |
| Left Shift | Descend while flying |
| Left Control | Movement boost |
| V | Toggle flight |
| F1 | Toggle the diagnostics panel and release the cursor |
| F12 | Save a timestamped screenshot |
| Escape | Release the cursor |
| Left click | Capture the cursor again |

Movement is unbounded in both walking and flight modes.

The F1 panel's **Sun and time** section sets the time of day, pauses the cycle,
and changes its duration in real minutes. Dawn, Noon, Dusk and Night buttons
quickly select a lighting setup. The default starts at 10:00 and completes a
full day in 24 real minutes. **Raymarched lighting** exposes terrain shadows,
volumetric sunlight, water reflections, quality and light shaft strength;
**Rendering** includes fog density, sun intensity and exposure.

**Weather** has moving fronts, so Clear, Cloudy, Overcast, and Fog / Whiteout
can occur in different parts of the world at the same time. The starting area
is Cloudy; local weather changes as fronts pass or the player travels. The F1
condition selector biases the map toward a chosen condition while retaining
local variation, and its transition control blends that bias over 75 seconds
by default. Weather changes cloud amount and altitude, moving shadows, sun
strength, sky color, and atmospheric visibility together. Fog / Whiteout
produces short-range visibility and a diffuse, nearly colorless sky by day.

**Volumetric clouds** controls cloud coverage, density, base altitude, layer
thickness, wind speed and direction, and shadow strength. **Cloud shape** expands
formation scale and edge detail controls. Clouds start enabled with coverage
`0.55`, density `1.0`, a base at `1300 m`, and a `1200 m` thick layer. The default
wind is `18 m/s` toward `70°` (`0°` is world +X, `90°` is +Z), with shadow strength
`0.65`, formation scale `1500 m`, and edge detail `0.35`. Wind accumulates in real
time independently of the sun, so changing the hour or crossing midnight keeps
cloud formations continuous. Set wind speed to zero to hold them in place.

## Command line

| Flag | Effect |
| --- | --- |
| `--camera x,y,z,yawDeg,pitchDeg` | Pin the camera pose and fly, for reproducible shots |
| `--shot path.png` | Render, save a screenshot, then exit |
| `--wait n` | Frames to render before the screenshot |
| `--size W,H` | Window size in points (comma-separated, as the C++'s `sscanf`) |
| `--probe` | Print terrain and erosion statistics, then exit |
| `--probe-extent`, `--probe-step` | Probe sampling window |
| `--measure-overlap` | Compare erosion tiles across a shared lattice edge |
| `--overlap-tile x,z` | Select the tile pair to compare |
| `--lattice` | Draw the erosion lattice overlay |
| `--no-fog`, `--no-water` | Disable those stages |
| `--time-of-day H` | Start at hour `H` in `[0, 24)`; default `10` |
| `--day-length MIN` | Positive real minutes per complete day; default `24` |
| `--pause-time` | Hold the selected time of day and freeze weather fronts and cloud wind |
| `--advance-time` | Allow the day/night cycle, weather, and cloud wind to advance during a screenshot run |
| `--weather clear\|cloudy\|overcast\|fog` | Bias the spatial weather map toward a chosen condition; default `cloudy` (`whiteout` also selects fog) |
| `--static-weather` | Freeze weather fronts in place; conditions still vary by location |
| `--raymarch-quality low\|balanced\|high` | Choose the ray sampling budget; default `balanced` (also accepts `0`, `1`, `2`) |
| `--no-raymarch` | Disable raymarched terrain shadows, volumetric integration, clouds and terrain reflections on water |
| `--no-clouds` | Disable cloud rendering and cloud shadows independently |
| `--cloud-coverage N` | Coverage in `[0, 1]`; default `0.55` |
| `--cloud-density N` | Density multiplier in `[0, 4]`; default `1` |
| `--cloud-base M` | Cloud base altitude in `[100, 6000]` metres; default `1300` |
| `--cloud-thickness M` | Layer thickness in `[100, 4000]` metres; default `1200` |

Screenshot runs freeze the celestial clock, weather fronts, and cloud wind automatically, so
erosion warm-up does not change the selected weather or lighting. Keep the same
camera and `--wait` value when comparing settings. For example, these capture noon, dusk and night from
the same elevated position above spawn:

```sh
cargo run --release --bin realistic_forest -- --camera 0,80,0,45,-8 --time-of-day 12 --shot noon.png --wait 600
cargo run --release --bin realistic_forest -- --camera 0,80,0,45,-8 --time-of-day 17.75 --shot dusk.png --wait 600
cargo run --release --bin realistic_forest -- --camera 0,80,0,45,-8 --time-of-day 0 --shot night.png --wait 600
```

`--no-raymarch` retains the moving sun, moon, sky, analytic water reflection and
simple atmospheric fog. Add `--no-fog` to remove atmospheric scattering and
extinction as well; this leaves the water's underwater optics controls separate.

## Terrain and clipmap

quick-noise creates a deterministic, tileable 1024×1024 R32 source field.
Both CPU collision/erosion setup and the terrain vertex shader sample that same
field with matching bilinear rules at any world coordinate. Domain warping,
independent continental and plain fields, and two blended ridge scales keep the
large landforms irregular. `LANDFORM_HORIZONTAL_SCALE` stretches their macro
coordinate domain 2.5×, producing broader ranges, plains, coastlines, and
islands while retaining the original fine surface frequencies. The macro
landform is then shaped through a bounded-exponential altitude profile around
sea level: land compresses toward the waterline into broad plains and rises
through an accelerating gradient into steeper, taller alpine crowns, while
the seabed keeps a gentle littoral shelf and then descends progressively
into a deep basin. Fine surface octaves bypass the profile so plains keep
their rolling detail. `LANDFORM_VERTICAL_SCALE` independently scales the
macro input and fine relief 1.5×. A decaying clearance stretch lifts the
lowest land a few metres clear of the sea plane so wide near-shore plains do
not z-fight with it, and the deepest basins flatten onto an abyssal plain.
Only a small area immediately around spawn
is gently stabilised.

The terrain uses one indexed 224×224-quad centre mesh and one reusable ring
mesh. Seven levels use vertex spacing from 1 to 64 metres, reaching 7168 metres
from the clipmap anchor. All levels share a stable 64-metre world lattice. The
outer part of each level morphs onto the next coarser global lattice, including
its normal sampling interval, which prevents cracks and greatly reduces LOD
shimmer. A 5800-metre far plane bounds the finite render horizon while
generation itself remains unbounded.

Terrain uses triplanar PBR textures with a muted, earthy palette. Grass004
covers stable ground; erosion and fresh deposition can replace it with
Ground103 soil and subtle Ground106 variation, without a minimum grass share.
Deposited fines favour flats, while scouring also exposes soil on banks.
Grass, shallow soil and Rock032 outcrops overlap through a shared exposure
field, allowing grass to meet exposed stone without a continuous dirt belt.
Material contacts use the scans' cavity information as a local relief proxy;
colour, normals, roughness, AO and lighting masks share the resulting coverage.
Pixel filtering softens unresolved edges, and a small residual mix preserves
thin sediment instead of discarding every minor material.
Slope, incision, substrate hardness and convex ridges expose stone; sheltered
hollows and deposited sediment retain cover. Broad, deterministic geology and
smaller weathering patches vary those boundaries in world space, with no rock
altitude cutoff, so high gentle benches can keep meadows and low cliffs can
expose stone.
Gravel fills scoured drainage and concave footslopes where loose debris can
remain. It sheds between roughly 34 and 46 degrees, exposing the bedrock on
steep gully walls. Material slope and aspect use a fixed four-metre terrain
sampling interval; distant mesh interpolation still limits their detail.
Ground093C sand follows the coast and slower depositional channels. Snow006
combines a broad altitude climate bias with world-anchored regional variation and local drifts,
including on exposed rock, so slopes do not share a fixed snow contour.
It settles like sediment: it holds deeper and reaches
lower inside dry sheltered hollows, drains down inactive gullies as fingers
below the regional line, sheds steep walls to bare rock, melts earlier from
sun-facing slopes, eroded ridges and scoured faces, and its melt margin
picks up a grey-brown sediment stain where active drainage works the
thinning pack. Retention and shedding use terrain evidence at every viewing
distance; only unresolved fine noise is filtered away. These are
material-placement rules derived from the erosion
results, not a climate or snowmelt simulation.

World-anchored, warped fields vary patch size and density across broad regions;
fine breakup filters away with distance. Grass shifts between green and dry
olive with moisture, exposure and alpine climate. Metre-scale rock relief,
restrained tilted bedding and broad mineral variation keep exposed faces from going uniform when the scan detail mips out.
Each material supplies matching colour, normal, roughness and optional
ambient-occlusion maps. The scans retain 1024² texels per layer; albedo is resized
and mip-filtered in linear light and decoded by the GPU's sRGB sampler, while
AO, normals and roughness remain linear data. The two eight-layer arrays use
about 85 MiB including mipmaps (64 MiB more than the former 512² arrays). Texture
cells use deterministic quarter turns and narrow edge blends; grass, soil,
gravel and sand also use independent sample offsets to avoid repeating the same tufts and stones. All PBR channels share
the mapping, including normal orientation. Damp drainage and the shoreline
darken and become smoother, while snow retains its wind relief and glints.
A camera-centred sea-level plane creates ocean and shorelines. The
**Dirt/gravel variant scale** control affects scan variation within soil and gravel.

Ocean wave lighting is evaluated per pixel, so distant waves stay visible
beyond the mesh's displacement fade and across the flat horizon skirt.
Each wave filters against its projected pixel footprint; unresolved ripples
broaden the sun reflection instead of shimmering. Crest compression drives
whitecaps and subsurface light, with foam detail also filtered at distance.

## Imported grass

The nine separate grass pack 2 models live in `assets/models/`, with one
shared material set in `assets/textures/grass/`. They retain their authored
metre scale and individual shapes. The unused pack 1 models and textures are
archived in `tmp/unused/`; its human reference and display stand were never
extracted. `assets/models/grass-manifest.json` records each active source mesh
and material; `assets/models/GRASS-ATTRIBUTION.md` preserves the original
licence and author credit. Run
`python3 scripts/extract_grass.py --validate` to verify the extracted assets.

Grass is scattered deterministically in world cells, so camera movement and
streaming do not reshuffle plants. Density and size vary in broad patches:
the nearby carpet places up to 64 candidates per square metre, thinning to
eight through 110 metres, then three through 170 metres, and finally a sparse
field of larger clumps. Each tier fades over distance rather than ending at a
hard ring. Grass fades beyond 213 metres and reaches its draw limit at 235 metres.
Pack 2's medium and small clumps dominate the carpet. The GPU captures
a 512-metre local map of terrain height, grass suitability and ground colour
every frame, using the terrain's actual snow, sand, soil, rock, gravel and
erosion material weights. Roots are admitted where grass dominates the visible
blend, even if some dirt or sand shows through, above the ocean crest
and away from snow, rock, gravel, and incised drainage furrows.
The erosion solver's water is working runoff rather than a visible inland
water body; it does not remove otherwise healthy valley grass. Each
clump checks its root footprint, rather than only its centre. The capture
retains sub-metre root precision while covering the distant field.

Pack 2's grayscale blade texture supplies light and dark detail. Each frame the
GPU averages nearby unlit terrain colours, and every clump samples that local
average at its root for colour. The atlas's visible-pixel average normalises
its grayscale detail; a canopy brightness factor compensates for the lower
light received by upright cards. Small per-clump brightness variation keeps
the carpet from looking uniform.

The imported alpha-tested meshes join the terrain G-buffer before SSAO, so the
lighting composite applies the same raymarched terrain and cloud shadows,
volumetric atmosphere, moonlight and fog to the grass. A small transmitted-light
term keeps thin leaf cards readable when backlit while respecting those
raymarched shadows. Wind bends the blade tips and leaves roots anchored.

## Fidelity of the noise substitution

The C++ samples FastNoiseLite's OpenSimplex2S; the port samples quick-noise's
Simplex with identical parameters (seed 1337, 5 octaves, frequency 0.0085,
lacunarity 2.02, gain 0.5) and the identical `*0.5 + 0.5` remap. These are
different algorithms, so they produce different fields — statistically
equivalent realizations, not the same realization. Measured over the full
1024×1024 field:

| | FastNoiseLite | quick-noise |
| --- | --- | --- |
| mean | +0.00262 | +0.00098 |
| sd | 0.25045 | 0.25672 |
| range | [-0.7889, 0.7622] | [-0.7158, 0.7023] |

Coarse-grained sd agrees within 2 percent for windows from 8 to 256 metres, and
the pointwise correlation between the two fields is +0.014.

That difference propagates through the domain warp, so the port renders a
*different landscape*, not a different formula. The evidence: feeding the C++'s
own height function the quick-noise field reproduces the port's 4225-point probe
grid to 0.01 metres (r = 1.000000), while feeding it the FastNoiseLite field
reproduces the C++'s grid to the same 0.01 metres. And the C++'s generator
disagrees with itself this much from one seed to the next — across 40
FastNoiseLite seeds at identical parameters the height field's sd spans
39–88 metres and its cross-seed correlation spans -0.37 to +1.0, a range that
contains the port's terrain (sd 45 metres, the 15th percentile). Run
`cargo run --bin noise-compare` to dump the field for such a comparison.

## Hydraulic erosion and flow output

Independent 1024×1024-metre erosion passes are centred on a globally aligned
512-metre lattice, so neighbouring footprints overlap by exactly 50 percent.
Every steady-state world point is therefore a gradient blend of four genuinely
independent hydraulic simulations. Each pass uses 4-metre cells plus a
448-metre scratch halo; only the central 1024 metres are retained. Water and
sediment evolve throughout that halo, putting the closed simulation edge well
outside the visible footprint. Each pass starts from the same world-aligned
base terrain and geology, so its result does not depend on which neighbours
finished first. There are no hard overlap thresholds or frozen internal
boundaries; only the distant outer guard band fades erosion back to base.

Each tile runs as three fragment-shader passes with float ping-pong targets,
which avoids compute shaders and keeps every pass a plain render target:

1. Conservative pipe flux steered by a rotationally symmetric 3×3 gradient.
2. Rain, evaporation, water transport, continuous-angle velocity, and discharge.
3. Inertial sediment advection, erosion, settling, and deposition.

The flow and sediment passes adapt the useful physics from the hydraulic
erosion shader in `~/nerthus` without depending on its WebGPU compute atomics.
Runoff retains inertia and follows a bilinearly sampled downhill direction;
excavation is capped by the actual forward drop and becomes progressively more
resistant with channel depth. Uphill travel and slow or evaporating water settle
their carried sediment. World-stable multi-scale geology bends and branches
channels consistently through tile overlaps, while a bilateral cone footprint
removes isolated cuts without smoothing away small tributaries. Ocean cells
remain drains, and the small spawn footprint resists destructive excavation.

One scratch simulation is advanced incrementally while completed passes are
packed into surface and flow atlases. A lookup texture maps
nearby lattice coordinates to cache slots and reveal weights. The shared
blend-mask image is a square gradient profile remapped to a separable Hermite
tent and stored as R32, so opposing pass weights form a smooth partition
without 8-bit bands. The four fixed geometric weights are normalized together,
while each pass fades in independently; a pass that is not ready contributes
procedural base terrain rather than amplifying the others. The initial four-pass
quartet is prewarmed at spawn. Beyond 1100 metres from the player, one shared C1
radial visibility field fades height and flow and material evidence back to the
procedural base terrain, reaching zero by 1600 metres — safely before the
nearest exact-four support edge of the 9×9 pass cache. CPU collision and both
terrain shaders use the same lattice, weights, centre, and radii.

The retained RGBA flow atlas is available to the terrain shader and the
diagnostics view with this per-texel contract:

| Channel | Value |
| --- | --- |
| R | Water depth |
| G | Signed world-X velocity |
| B | Signed world-Z velocity |
| A | Accumulated discharge |

The RGBA surface atlas retains signed height displacement in R for the vertex
shader and collision. G stores local concavity in
metres; B stores positive log2 discharge concentration relative to a surrounding
12-metre ring; A stores substrate hardness. These fields are derived at tile
completion using the simulation halo, then share the same gutters, overlap
weights, reveal and visibility as the terrain. Positive displacement is net
deposition; suspended sediment is not treated as deposited soil. Concentrated
discharge distinguishes drainage channels from general rainfall, and shallow
water velocities are gated when estimating transport strength.

The flow output
remains available for later river geometry. Press F1 and enable **Flow visualization** to inspect it on the
terrain, or expand **Flow output** to inspect the cached target. Erosion
parameters can be edited from the same panel and applied with **Regenerate
erosion cache**. The panel also reports maximum incision, deposited height,
small-scale detail within actively eroded ground, and flow-axis bias for the
most recently completed tile.

## Day/night lighting and atmosphere

The atmosphere takes inspiration from the coordinated sky, fog and lighting
approach presented in Rockstar's
[Creating the Atmospheric World of Red Dead Redemption 2](https://advances.realtimerendering.com/s2019/index.htm)
at SIGGRAPH 2019. This prototype implements a smaller, independent version of
those ideas with procedural volumetric clouds, fog and shared celestial lighting.

The sun rises in the east at 06:00, reaches its highest point at noon, and sets
in the west at 18:00. Its colour warms toward the horizon and its direct light
fades out at sunset, while twilight continues to light the sky. The moon follows
the opposite orbit and provides subdued blue light at night, when stars become
visible. Terrain, ocean highlights and reflected sky all use the same celestial
state. Snow placement retains its fixed climate aspect; only its lighting and
glints move with the sun.

Terrain shadows march toward the light through a camera-centred world heightfield.
The 1024×1024 map spans 12,288 metres and samples the same procedural terrain and
streamed erosion as the visible geometry. Hills outside the camera view can
therefore cast shadows and occlude sunlight in the fog. Shadow coverage and fine
occluder detail remain limited by the map's extent and 12-metre texels, with a
fade at its outer boundary.

Volumetric lighting integrates scattering and Beer–Lambert transmittance along
view rays at half resolution. Height-dependent density concentrates haze near
the ground, and samples toward the sun through the heightfield produce shafts
where terrain blocks light. A depth-aware bilateral upscale preserves terrain
silhouettes when combining the fog with the full-resolution scene. Fog density
controls the atmosphere's thickness; **Light shaft strength** controls direct
light scattered into it. Turning off **Volumetric sunlight** keeps a cheaper
uniform fog approximation.
Water integrates the same atmosphere up to its own surface with a smaller
sample budget, so haze and shafts follow the shoreline rather than the seabed.

Clouds occupy a world-space volume whose base, thickness and coverage follow
the local weather front. Periodic 3D fractal and Worley noise build rounded
cumulus formations with irregular coverage, varied tops and eroded edges.
Progressive view-ray steps resolve nearby cloud surfaces when flying through
the layer, and filtered 3D noise limits distant sampling artifacts. View rays
integrate density and transmittance through the volume; secondary rays toward the sun or moon
produce shadowed interiors, warm sunset lighting and bright edges around the
sun. Multiple scattering is approximated to retain light inside dense clouds.
The same wind-driven density casts moving shadows over terrain, water and fog.
Cloud rays stop at opaque geometry, and the volume remains visible when flying
inside or above it. This is one procedural cloud layer, with a `40 km` maximum
view-ray distance and a gradual fade near that limit.

Water traces reflected rays against the terrain view-position buffer, refines
depth crossings, and samples the lit scene at each valid hit. This adds visible
coastlines and mountains to the wave-distorted reflections. Invalid hits and
screen edges fade into the shared day/night sky with clouds. A separate
`256×128` all-direction sky probe raymarches the same cloud volume each frame,
so water can reflect clouds outside the camera view. Its angular resolution
and camera-centred origin make cloud reflections an approximation, especially
near or inside the cloud layer. These screen-space terrain reflections
cannot recover terrain hidden behind other surfaces or outside the camera view;
the world-heightfield shadows have different coverage from the reflections.

| Quality | Cloud view steps | Cloud sunlight/shadow taps | Cost and appearance |
| --- | --- | --- | --- |
| Low | Up to 32 | 4 | Fewer shadow, fog and reflection samples; useful at high display resolutions or on slower GPUs |
| Balanced | Up to 48 | 5 | Default compromise between sampling detail and GPU cost |
| High | Up to 72 | 6 | More samples to resolve clouds, shadows, shafts and reflection intersections; higher GPU cost |

All three presets keep the same effects enabled. The F1 switches can disable
individual effects independently, and `--no-raymarch` disables the traced
effects together. Fog and clouds share a half-resolution integration target
with a depth-aware upscale; the scene keeps its full output resolution. Cloud
rays stop early once sufficiently opaque and skip lighting in empty regions.
Clouds still add substantial GPU work, especially with high quality, large
display resolutions or many partially transparent formations. Lower raymarch
quality or disable clouds independently if the frame rate becomes too low.

## Render pipeline

The renderer writes view position, view normal, and albedo to a three-target
G-buffer. SSAO is evaluated at half resolution with a 24-sample rotated
hemisphere kernel, followed by a depth/normal-aware bilateral blur. A separate
half-resolution atmosphere pass supplies combined fog/cloud scattering and
transmittance to the lighting composite, which also evaluates terrain and cloud
shadows and the celestial sky. A cloud sky probe supplies reflection directions
outside the current view.
The water surface then samples the opaque scene for refraction and raymarched
reflections, followed by underwater effects where applicable. FXAA smooths the
completed scene, and the diagnostics panel draws on top. The G-buffer, AO and
atmosphere targets follow window resize and HiDPI render-size changes.

The composite, water and FXAA stages use the view target's two main textures,
with an opaque scene copy available to water. The composite reads the G-buffer;
water adds its surface and medium effects; FXAA reads the completed scene and
writes the texture the upscale node presents. The output passes write into an
sRGB-format target, where the C++ wrote into a plain framebuffer with no
conversion, so the composite and FXAA split the encode between them: the composite
writes its display-encoded value and lets the target encode it, FXAA's fetch of
that same target decodes it straight back — which is what keeps FXAA filtering
in the C++'s display-encoded domain, where its luma thresholds mean what the
GLSL meant. FXAA is the last pass that authors pixels, so its store inverts the
encode analytically and the target's encode cancels it: the byte written is the
value the filter produced, matching the C++'s byte pixel for pixel wherever FXAA
passes a pixel through untouched. The `PORT NOTES` in `assets/shaders/fxaa.wgsl`
explains this in full.

This is currently a desktop prototype with a fixed terrain seed and a bounded
in-memory erosion cache over an unbounded procedural world.
