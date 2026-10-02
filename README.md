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
| Tab | Toggle the debug trainer and release the cursor |
| F1 | Toggle the diagnostics panel and release the cursor |
| F12 | Save a timestamped screenshot |
| Escape | Release the cursor |
| Left click | Capture the cursor again |

Movement is unbounded in both walking and flight modes.

The **Trainer [Tab]** button at the top right also opens a compact debug trainer.
Choose a weather preset to apply it immediately around the player, or choose
**Natural weather** to return to varied conditions. The **Move weather fronts**
checkbox controls whether those conditions drift. Under a thunderstorm,
**Strike lightning** fires the next discharge at once, and **Thunder volume**
sets the loudness of thunder. Set the hour with
the slider or Dawn, Noon, Dusk and Night buttons; changing it holds the selected
time until **Hold selected time** is unchecked. **Advanced diagnostics** opens
the full panel without using F1.

The diagnostics panel's **Sun and time** section sets the time of day, pauses the cycle,
and changes its duration in real minutes. Dawn, Noon, Dusk and Night buttons
quickly select a lighting setup. The default starts at 10:00 and completes a
full day in 24 real minutes. **Raymarched lighting** exposes terrain shadows,
volumetric sunlight, water reflections, quality and light shaft strength;
**Rendering** includes fog density, sun intensity and exposure.

**Weather** has moving fronts and smaller storm cells, so Clear, Cloudy,
Overcast, Fog / Whiteout, Rain, Snow, and Thunderstorm can occur in different
parts of the world at the same time. Clear, Cloudy, and Overcast cover most of
the map. Fog is less common; rain and snow occupy smaller areas; thunderstorms
are the rarest. The starting area is Cloudy. The F1 condition selector biases
the map toward a chosen condition while retaining local variation, and its
transition control blends that bias over 75 seconds by default. Weather
changes cloud amount and altitude, moving shadows, sun strength, sky color,
and atmospheric visibility together. Fog / Whiteout produces short-range
visibility and a diffuse, nearly colorless sky by day.

Rain travels in wind-driven sheets inside the larger moving storm cells. Broad
and fine bands bend across the landscape, so heavy squall lines alternate with
steadier rain at a fixed location. The raymarched atmosphere integrates rain,
snow, and fog density along each view ray, so a downpour cuts visibility to a
kilometre or two and distant curtains of rain stand out against the landscape.
Two depth-tested lattices of drops surround the camera: a dense one within about
16 m and a sparser one out to about 45 m. Each drop falls at the terminal
velocity of its size and is drawn as the streak it traces during a 1/30 s
exposure, leaning with the wind and storm gusts. A drop takes its colour from
the scene around it, as the small lens it is, so rain reads light against dark
forest and slightly darker against the sky. Camera-centred layers of finer
streaks carry the rain out past 100 m, and splash crowns burst where drops land
on nearby ground. Snow falls more slowly as flakes. Wet ground darkens and
mirrors the sky, puddles collect on level ground outside the grass and ripple
under the drops, and the ground stays damp between passing sheets, while
snowfall gives upward surfaces a light dusting. Grass thrashes in storm gusts.
The F1 panel reports local intensity and visibility. Rain makes short-lived
ripples and splashes on the water; snowflakes briefly fleck its surface. Local
storm gusts roughen the existing wave spectrum without resetting it.
Thunderheads dim the sky, clouds, terrain, and fog even between heavy rain
sheets. Snow phase normally follows altitude and the mountain snowline;
selecting Rain, Snow, or Thunderstorm forces that phase for previews.

Thunderstorm cells discharge every few seconds, sometimes in rapid bursts. A
cloud-to-ground bolt follows its stepped leader: a tortuous, fractal channel
that branches as it searches down from inside the cloud base, drawn faintly as
it descends. A brilliant return stroke then lights the channel and its branches,
and two to four restrikes flicker down the main channel alone, some held by
continuing current, before it fades. Some discharges stay in the cloud: they
only light the deck, or crawl visibly along its base as branching spider
lightning. The channel is drawn with a white core and a blue-violet glow that
heavy rain softens and widens, and cloud hides any part above the base. Each
flash lights nearby ground, clouds, fog, rain, and water, and the water mirrors
the bolt. Thunder is synthesised from the bolt itself: each part of the channel
is heard after its own travel time at the speed of sound, so a close strike
cracks and then rolls, a distant one only rumbles, and a flash inside the cloud
is muffled.

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
| `--camera x,y,z,yawDeg,pitchDeg` | Pin the camera pose and fly, for reproducible shots; with `--shot`, mouse and keyboard input cannot move it |
| `--shot path.png` | Render, save a screenshot, then exit; the log reports average, p95 and maximum frame time over the `--wait` window, split into frames while erosion tiles stream and settled frames |
| `--wait n` | Frames to render before the screenshot, counted once the prewarmed erosion tiles are ready |
| `--erosion-prewarm N` | Simulate the `N` nearest erosion tiles at full budget before streaming (default `4`, the quartet around the player); raise it so a capture shows erosion beyond the player's own lattice cell |
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
| `--weather clear\|cloudy\|overcast\|fog\|rain\|snow\|thunderstorm` | Bias the spatial weather map toward a chosen condition; default `cloudy` (`whiteout` and `storm` are aliases) |
| `--static-weather` | Freeze weather fronts in place; conditions still vary by location |
| `--lightning AGE[,DIST[,BEARING[,ground\|crawler\|cloud]]]` | Fire one discharge `DIST` metres away (default `900`), `BEARING` degrees clockwise from the camera heading (default `0`), and hold it `AGE` seconds after its first return stroke (negative to show the leader); for screenshots |
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

Terrain uses triplanar PBR textures with a muted, earthy palette, placed from
the erosion simulation's own record of the ground. The simulation tracks the
loose cover (soil, colluvium, talus and alluvium) left above bedrock, and the
material shader reads it together with the shared geology. Rock032 bedrock
shows where a face sheds faster than it weathers: steepness exposes it,
resistant beds stand bare where weak ones keep a broken mantle of rubble and
thin turf, and any loose cover hides it. Patch noise at several scales turns a
30-45 degree slope into a mosaic of outcrop, rubble and turf, the dipping
resistant beds give mountainsides ledges rather than one uniform slab, and
little but rock remains past about 46 degrees. Convex noses hold their faces,
sheltered hollows keep their soil, and scoured channel beds expose rock on any
slope. Below an exposure stone rarely meets closed turf: what a face sheds
comes to rest at its foot and in the hollows between its spurs as stony
colluvium, so a ragged band of grey-brown soil and rubble separates bare rock
from the grass beneath it. Above a face the crest only loses material, so the
turf thins to just a narrow, faint rim of stony soil before the stone. The
simulation supplies the distinction - the
talus relaxation and slope wash leave deposits and concave footslopes below a
face and lower the convex crest above it - and the band follows the rock's own
exposure field, widening where a face wanes gradually into its base; thick
footslope deposits extend it as colluvial aprons. Its outer edge is ragged
over a few metres rather than a line: grass islands survive in the debris,
dirt bays reach into the turf, the turf thins and dries to olive before it
gives way, and where turf and soil meet they mix in proportion, as tufts
growing through soil do, instead of switching at a contact edge. Beyond the erosion radius
the shader estimates the cover the simulation starts from, so material
boundaries stay put with distance.

Grass004 covers ground with a few decimetres of soil. Ground103 soil, with
subtle Ground106 variation, shows through thin, stony cover, in cut banks and
fresh deposits beside active channels, and in broad background patches; older
fill on floodplains and fan surfaces revegetates. Gravel lines scoured, active
channels and their bars, and forms talus aprons at the foot of steep faces
near its angle of repose; rubble mantles more of the steep ground above the
treeline, where alpine turf thins and dries. Contributing area separates
permanent streams from grassed hollows, so only real channels open the turf.
Ground093C sand follows the coast and settles where channel water slows over
its own deposits. Water routed over steep rock leaves dark streaks lined up
with the gullies below, and lichen tints shaded faces grey-green and sunny
ones ochre. Material contacts use the scans' cavity information as a local
relief proxy; colour, normals, roughness, AO and lighting masks share the
resulting coverage. Pixel filtering softens unresolved edges, and a small
residual mix preserves thin sediment instead of discarding every minor
material. Material slope and aspect use a four-metre terrain sampling interval
near the camera; farther out the interval widens continuously with distance,
tracking the clipmap's vertex spacing, so distant coverage is prefiltered
instead of aliasing into rows of triangles along snow and rock margins.

Snow006 lies where the ground stays cold through the melt season. A regional
snowline (`SNOWLINE_ALTITUDE`, 112 metres) wanders with broad climate cells.
Against the low spring sun, shaded poleward slopes hold snow about 20 metres
lower than level ground at the same height and sun-facing slopes lose it about
9 metres higher; sheltered hollows and lee slopes keep a deeper pack, while
wind-scoured crests and windward faces lose theirs. Snow thins past about 39
degrees and sheds by 55, so steep faces stay dark rock. It does not flow
downhill: a gully below the line melts out like any other low ground, and only
shaded avalanche gullies just under the line keep a short tongue of debris
snow. Permanent streams open dark meltwater ribbons through the pack, and its
thinning margin picks up a grey-brown sediment stain. Every term is a
world-space field; only the slope it reads is prefiltered with distance, so
cover stays put as the camera moves. These are
material-placement rules derived from the erosion results and terrain aspect,
not a snowmelt simulation.

World-anchored, warped fields vary patch size and density across broad regions;
fine breakup filters away with distance. Grass shifts between green and dry
olive with moisture, exposure and alpine climate. Exposed faces break into
jointed blocks about three metres across, with a finer metre-scale set inside
them: each block's face tilts and weathers a little differently and dark joints
open between them, flattened into slabs on steep faces where bedding planes
cross vertical joints. Together with metre-scale relief, restrained tilted
bedding and broad mineral variation this keeps exposed faces from going uniform
when the scan detail mips out, and every octave fades before it shrinks below a
few pixels. Talus carries fall-line streaks and block-scale rubble relief for
the same reason.
Each material supplies matching colour, normal, roughness and optional
ambient-occlusion maps. The scans retain 1024² texels per layer; albedo is resized
and mip-filtered in linear light and decoded by the GPU's sRGB sampler, while
AO, normals and roughness remain linear data. The two eight-layer arrays use
about 85 MiB including mipmaps (64 MiB more than the former 512² arrays). Texture
cells use deterministic quarter turns and narrow edge blends; grass, soil,
gravel and sand also use independent sample offsets to avoid repeating the same tufts and stones. All PBR channels share
the mapping, including normal orientation. Dry rock and gravel are matte: their
scans' roughness is lifted toward fully rough while keeping its own variation,
and runoff stains darken stone without polishing it. Damp drainage and the
shoreline darken, but only wet ground at the waterline turns smoother, so
channel gravel and rock never read as polished. Snow retains its wind relief and
glints. The G-buffer position is rebuilt from each terrain fragment's world
position: the interpolated view-space position arrives as zero in scattered
fragments on Metal, which flashed white specks across the terrain and opened
holes in the sea's hidden-surface test.
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

## Erosion and flow output

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

Each tile iteration runs four fragment-shader passes with float ping-pong
targets, which avoids compute shaders and keeps every pass a plain render
target:

1. Conservative pipe flux steered by a rotationally symmetric 3×3 gradient.
2. Rain, evaporation, water transport, continuous-angle velocity, and discharge.
   This pass also writes each cell's drainage-routing normaliser (the sum of
   its squared downhill slopes) for the next pass, since no bed moves between
   them.
3. Inertial runoff sediment advection, erosion, settling and deposition,
   together with fluvial transport along routed drainage. This pass writes the
   terrain and drainage states as two targets; drainage carries each cell's
   bedrock resistance for the thermal pass.
4. Thermal (talus) relaxation of over-steepened slopes.

The runoff solver adapts the useful physics from the hydraulic erosion shader
in `~/nerthus` without depending on its WebGPU compute atomics. Runoff retains
inertia and follows a bilinearly sampled downhill direction; excavation is
capped by the actual forward drop and becomes progressively more resistant with
channel depth. Uphill travel and slow or evaporating water settle their carried
sediment. World-stable multi-scale noise bends and branches channels
consistently through tile overlaps, while a bilateral cone footprint removes
isolated cuts without smoothing away small tributaries. Ocean cells remain
drains, and the small spawn footprint resists destructive excavation.

That rain sheet only lives a few simulated seconds, so on its own it carves
rills and small gullies but never organises a valley. Fluvial transport
supplies what it cannot. Contributing area is routed between bed cells with
multiple-flow-direction shares proportional to the squared slope, so every
channel knows its whole catchment within the tile's 1920-metre domain; the CPU
routes the base surface before a tile starts, so this holds from the first
iteration. That routing releases each cell once every higher neighbour draining
into it has passed its area on, which keeps it linear in the tile size, and in
interactive play the base heights and routing are built on a background task so
a starting tile never stalls a frame (prewarm and screenshot runs build them
inline, keeping captures deterministic). A stream-power capacity proportional to √A·S then detaches bed where
the routed sediment flux is below capacity and deposits where the flux exceeds
it: steep, well-fed channels incise, slope breaks build fans, and valley floors
and closed hollows fill with alluvium. Channels initiate above a few hundred
square metres of catchment, leaving unchannelled hillslopes to the runoff
solver; stream power saturates beyond about 6.4 hectares, because overlapping
tiles truncate large rivers at different domain edges; and incision stops at
**Max incision** below the base surface, never below the sea or a neighbouring
bed.

The terrain state tracks the loose cover above bedrock: deposits add to it,
erosion entrains it before cutting bedrock, and the thermal pass uses it to
decide which slopes stand. Loose material sheds at its ~34-degree angle of
repose; bare bedrock only fails above a much steeper, hardness-dependent limit
(about 48 degrees in weak rock to 68 in the most resistant beds), and then
slowly, as rockfall. The exchange is pairwise and antisymmetric, so it conserves
material exactly without a scatter step. Together with incision this opens
fresh cuts into V-shaped valleys, rounds weak crests, keeps resistant rock
standing as cliffs and gathers talus aprons below steep faces. Substrate
resistance comes from one shared, world-keyed geology
(`assets/shaders/geology-functions.wgslinc`): broad and fine rock bodies plus
gently dipping, warped resistant beds every ~19 metres of elevation, evaluated
at the bedrock surface a cut actually exposes, so incision slows on a resistant
bed and leaves a bench. The terrain material shader reads the same geology.

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
| A | Routed contributing area, in 4×4-metre cells |

The RGBA surface atlas retains signed height displacement in R for the vertex
shader and collision. G stores local concavity in metres; B stores the positive
log2 concentration of contributing area relative to a surrounding 12-metre
ring, which finds each channel's thalweg; A stores the loose cover left above
bedrock, in metres. These fields are derived at tile completion using the
simulation halo, then share the same gutters, overlap weights, reveal and
visibility as the terrain. Positive displacement is net deposition; suspended
sediment is not treated as deposited soil. Contributing area separates
permanent channels from rills and hillslopes, and shallow water velocities are
gated when estimating transport strength.

The flow output
remains available for later river geometry. Press F1 and enable **Flow visualization** to inspect it on the
terrain, or expand **Flow output** to inspect the cached target. Erosion
parameters can be edited from the same panel and applied with **Regenerate
erosion cache**; besides the runoff controls they include stream power,
channel incision, alluvial deposition, maximum incision, talus slide and
rockfall rates. The panel also reports maximum incision, deposited height,
small-scale detail within actively eroded ground, flow-axis bias, the largest
contributing area, the share of bare bedrock and the mean loose cover for the
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
