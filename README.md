# Forest

Forest is a Rust 2024 prototype for an infinite procedural landscape. It renders
mountains, plains, beaches, ocean, rivers and creeks, and biome colours with an
indexed geometry clipmap. Procedural terrain can be evaluated at any world coordinate, and
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
| 2 | Open or close the developer menu; release the cursor while open, capture it on close |
| 3 | Toggle the debug text overlay |
| F12 | Save a timestamped screenshot |
| Escape | Release the cursor |
| Left click | Capture the cursor again |

Movement is unbounded in both walking and flight modes.

The developer menu opens at the top left with 2 and leaves no launcher or
overlay when closed. Its movement tools teleport to exact XYZ coordinates
(automatically enabling flight), place the player on the ground at the entered
XZ coordinates (ignoring Y), return to spawn, place the player on the ground
here, or rise 100 m. Movement speed is adjustable from 0.1× to 20×, and **Copy
XYZ** copies the current position. Rendering shortcuts sit alongside the weather
and time controls; click **Advanced controls** to expand them inside the same menu.

3 independently toggles a passive, plain text overlay at the top left, outside
the developer menu. It reports FPS and frame timing, player XYZ, view direction,
yaw and pitch, movement and ground state, time, weather, erosion, and vegetation.

Choose a weather preset to apply it immediately around the player, or choose
**Natural weather** to return to varied conditions. The **Move weather fronts**
checkbox controls whether those conditions drift. Under a thunderstorm,
**Strike lightning** fires the next discharge at once, and **Thunder volume**
sets the loudness of thunder. Set the hour with
the slider or Dawn, Noon, Dusk and Night buttons; changing it holds the selected
time until **Hold selected time** is unchecked. **Advanced controls** expands
the full controls.

The developer menu's advanced **Sun and time** section sets the time of day,
pauses the cycle, and changes its duration in real minutes. Dawn, Noon, Dusk and Night buttons
quickly select a lighting setup. The default starts at 10:00 and completes a
full day in 24 real minutes. **Raymarched lighting** exposes terrain shadows,
volumetric sunlight, water reflections, quality and light shaft strength;
**Rendering** includes fog density, sun intensity and exposure.

**Weather** has moving fronts and smaller storm cells, so Clear, Cloudy,
Overcast, Fog / Whiteout, Rain, Snow, and Thunderstorm can occur in different
parts of the world at the same time. Clear, Cloudy, and Overcast cover most of
the map. Fog is less common; rain and snow occupy smaller areas; thunderstorms
are the rarest. The starting area is Cloudy. The advanced condition selector
biases the map toward a chosen condition while retaining local variation, and its
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
The advanced weather controls report local intensity and visibility. Rain makes
short-lived ripples and splashes on the water; snowflakes briefly fleck its surface. Local
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
| `--snow-trail x,z,x,z` | Seed one compressed snow segment between the two world XZ positions for reproducible screenshots |
| `--wait n` | Frames to render before the screenshot, counted once the prewarmed erosion tiles, plants and grass are ready |
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
| `--no-vegetation` | Skip loading and scattering the trees, shrubs and flowers |
| `--no-rivers` | Generate, carve and draw no rivers |
| `--river-map path.png` | Render the river network from above, print its statistics and the rivers and lakes nearest `--map-centre`, then exit |
| `--vegetation-map path.png` | Render the plant scatter from above, print its statistics, then exit |
| `--map-centre x,z`, `--map-extent M` | Area of the vegetation or river map: its centre (default `0,0`) and side in metres (default `1024`, or `8192` for the river map) |

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
mesh. Nine levels use vertex spacing from 0.25 to 64 metres, reaching 7168 metres
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

Snow006 forms a uniform white pack above the climate snowline
(`SNOWLINE_ALTITUDE`, 112 metres). A broad altitude transition and modest
spring-sun aspect shift keep the edge smooth. Snow thins past about 39 degrees
and sheds by 55, leaving steep rock faces exposed. Scan contrast and wind
relief are restrained, while fine grain, roughness and glints still respond
to the light; only the thinning fringe keeps a faint sediment stain.

Fully covered snow rises 32 centimetres above the underlying terrain. Walking
compresses that raised layer back to the terrain surface, leaving a continuous
trail with an 80-centimetre fully compressed core and smooth shoulders over a
total width of 1.3 metres. Jumping and flying do not create walking trails.
The player stands on the remaining snow depth, so the surface underfoot lowers
as it compresses. A high-resolution, 128-metre compression map moves with the
player, while sparse world storage preserves tracks outside that window and
restores them when the player returns. Trails persist for the current session
and are cleared when the game restarts.

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

The nine separate grass models (`grass-{large,medium,small}-{1,2,3}.glb`) live
in `assets/models/`, with one shared material set in `assets/textures/grass/`.
They retain their authored metre scale and individual shapes.
`assets/models/grass-manifest.json` records each source mesh and material;
`assets/models/GRASS-ATTRIBUTION.md` preserves the original licence and author
credit.

Grass is scattered deterministically in world cells, so camera movement and
streaming do not reshuffle plants. Density and size vary in broad patches:
the nearby carpet places up to 64 candidates per square metre, thinning to
eight through 110 metres, then three through 170 metres, and finally a sparse
field of larger clumps. Each tier fades over distance rather than ending at a
hard ring. Grass fades beyond 213 metres and reaches its draw limit at 235 metres.
The medium and small clumps dominate the carpet.

Grass streams on a permanent 16-metre world grid, with each density tier
loaded independently. When the 8-metre planning anchor moves, only entering
chunk tiers are generated and grouped by model on background workers. Nearby
chunks stay resident through a retention margin, including when the player
turns back. Eight persistent GPU instance arenas recycle vacated chunk slots:
four 32-byte candidate arenas and, beside them, four 48-byte prepared arenas
holding each clump's seated, habitat-approved records. Each frame uploads at
most 16 chunk tiers and 2 MiB of instance data. Their bounded capacity
reserves about 160 MiB — roughly 65 MiB of candidates and 97 MiB of prepared
copies, with the near-carpet tier owning about half — including prefetch and
retention space.
Screenshot runs wait until the current grass stream has been uploaded and drawn.

Each frame draws only the chunks that can reach the screen: those within a tier's
fade distance (33, 140, 200 and 235 metres) and within the footprint the view
frustum covers on the ground, which holds at any terrain height. A chunk is
dropped only when no clump rooted in it can produce a pixel. A compute pass
then thins what remains: each surviving candidate's habitat is evaluated once,
not once per mesh vertex, and the accepted clumps compact into the arena the
vertex stage reads. Draws are batched indirect draws; adapters without
indirect execution render no grass, as for the plant library.

The GPU captures a 512-metre local map of terrain height, grass suitability and
ground colour, using the terrain's actual snow, sand, soil, rock, gravel and
erosion material weights. The map depends only on its 8-metre window, the
clipmap's centre, the streamed erosion data and a few material settings, so it
is kept until one of those changes (the player entering the next window, an
erosion tile arriving or revealing) and standing still costs nothing. It is
rendered as though the camera stood in the middle of its window, so the
player's exact position does not enter it. Roots are admitted where grass
dominates the visible
blend, even if some dirt or sand shows through, above the ocean crest
and away from snow, rock, gravel, and incised drainage furrows.
The erosion solver's water is working runoff rather than a visible inland
water body; it does not remove otherwise healthy valley grass. Each
clump checks its root footprint, rather than only its centre. The capture
retains sub-metre root precision while covering the distant field.

The grass's grayscale blade texture supplies light and dark detail. With each
new capture the GPU averages nearby unlit terrain colours, and every clump
samples that local average at its root for colour. The atlas's visible-pixel
average normalises its grayscale detail; a canopy brightness factor
compensates for the lower light received by upright cards. Small per-clump
brightness variation keeps the carpet from looking uniform.

The imported alpha-tested meshes join the terrain G-buffer before SSAO, so the
lighting composite applies the same raymarched terrain and cloud shadows,
volumetric atmosphere, moonlight and fog to the grass. A small transmitted-light
term keeps thin leaf cards readable when backlit while respecting those
raymarched shadows. Wind bends the blade tips and leaves roots anchored.

## Trees, shrubs and flowers

The plant library in `assets/models/` holds 86 models in eight families: fir,
pine, oak, maple, lilac bush, bush, broadleaf plant and lavender. Each model is
a set of `name-lod-N.glb` files (`fir-large-2-lod-0.glb` … `-lod-3.glb`): three
mesh LODs of bark and foliage cards and a billboard, or a single LOD for
lavender. Textures live in `assets/textures/`. A background task loads the
library at startup, in about two seconds. Foliage, bark and billboard textures
are capped in resolution to keep about 230 MiB of video memory. Mip levels of
alpha-tested cards keep the coverage of the full-size image, so distant
crowns stay full, and transparent texels take the colour of the nearest leaves
so filtering never draws dark fringes.

### Where plants grow

Placement is ecological and deterministic: every plant is a pure function of
its world position, generated in 256-metre chunks that stream around the
player and match exactly where they meet. The same base landform the terrain
uses supplies height, slope, sun exposure, hollows and valleys, from which
smooth fields describe the forest:

- **Forest stands** follow a broad mosaic, with ragged edges and small
  glades, and stop at the coast, on faces too steep to hold soil, and at a
  treeline that rises on sunny slopes. Wet valley floors and exposed crests
  stay meadow.
- **Fir and pine** grow in single-species stands, about three firs to every
  pine. Pine takes warm, dry, well-drained ground and sandy lowlands, and a
  sharp disturbance-history field makes the stands mostly one species,
  mixing only along their borders, with groups of the other scattered through.
  A conifer's nearest conifer neighbour is the same species about 80 % of the
  time, against about 64 % if they were mixed at random.
- **Oak and maple** are sparse, scattered through the conifers, more along
  edges, in warm moist lowlands and standing alone in clearings. Oaks prefer
  sunny, dry ground.
- **Lilac and common bushes** form a thin understory, dense belts along the
  forest edge and thickets in clearings. They outnumber oak and maple and are
  far fewer than the conifers.
- **Broadleaf plants** grow along the shore, in the strip of turf where the
  beach's sand gives way to vegetation: knee-high plants, never touching, in
  loose colonies with open turf between. A few pioneers stand at the sand's
  edge; the plants are thickest a few metres in and thin out into the
  meadow. The strip follows the terrain shader's own wandering grass line
  and runs some 25 metres inland on gentle and steep coasts alike, wherever
  the sea lies close down the fall line. They line the banks of rivers and
  creeks the same way, on the damp ground just above the water.
- **Lavender** is scattered as single tufts, 0.6 to 0.95 m tall, through
  sunny, dry, well-drained open ground, never touching, at a density that
  varies smoothly and widely across a regional, a stand and a patch scale:
  from a tuft every hundred square metres or so to one every two or three
  (about one every five in a typical lowland clearing), so clearings range
  from a few flowers to purple swathes without any becoming a solid carpet.

Plants are placed in layers. The trees, shrubs, shore plants and lavender
each form a Matérn hard-core point process, the model forest ecology uses for
competition between trees: candidates are thinned by the local density, and
a survivor keeps its place only if no stronger candidate stands within the
sum of their crown reaches. Closed forest comes out evenly spaced at about
190 stems per hectare (one tree per 52 m²), while open woodland keeps a
clumpier pattern. Saplings and poles fill the gaps along edges and in open
stands, then shrubs grow around the trees, and the broadleaf plants and
lavender take what remains of the ground. Height grows with the stand's age
and vigour, so old stands are tall and the trees shrink toward the treeline,
the coast and thin soil. Each tree picks its size class and model from that
height and gets its own scale, rotation, lean and tint.

Chunks are generated at three detail levels: trees out to about 2 km, shrubs
to 1 km, and the broadleaf plants and lavender to 270 m. Each level runs on
the async compute pool, nearest first, so the forest fills in around the
player within a few seconds and is not regenerated when the player walks back
and forth across a boundary.

`--vegetation-map path.png` renders the scatter from above without opening a
window and prints its statistics. `--map-centre x,z` and `--map-extent metres`
(default `1024`) choose the area:

```sh
cargo run --release --bin realistic_forest -- --vegetation-map forest.png --map-centre 0,0 --map-extent 1600
```

### Drawing them

The scatter knows only the base landform. Every frame, a compute pass on the
GPU (`vegetation-cull.wgsl`) handles every streamed plant. It seats each root
on the eroded terrain, using the same height function as the clipmap, and
sinks trunks a little into slopes. It drops plants that land on steep faces,
in incised channels or at the waterline, and grows small plants only where the
grass would grow. It culls against the view, picks a LOD and compacts the
survivors into one list per model and LOD, counting them straight into the
indirect draw arguments. One indirect draw per model, LOD and material then
renders them into the G-buffer between the terrain and the grass; the CPU
never reads anything back.

LODs switch at a fixed multiple of each plant's own height, so a sapling
simplifies at the same size on screen as a 30 m pine: trees use their full
meshes to about 6.5 heights and billboards beyond 23. Every switch
cross-fades through a screen-door dither, each plant at a slightly different
distance, so no ring of popping trees follows the camera. Trees are drawn to
2 km, bushes and lilacs to between 160 and 900 metres depending on their size,
broadleaf plants to 170 m and lavender to 230 m, each dissolving out at its
edge. Wind from the weather bends stems in the mesh LODs from their roots,
with gusts sweeping across the canopy and leaves fluttering at the crown's
rim. LOD3 billboards stay static in both the view and shadow passes.
Billboards with separate front and back cards cull their backfaces in both
passes, preventing overlapping atlas faces from flashing dark as the camera
moves. Billboards made of single crossed cards remain two-sided.

Plants are lit by the same composite as the ground, with terrain, cloud and
volumetric lighting. Foliage shades as one rounded crown rather than a stack
of flat cards: it darkens toward the trunk and the crown's base, and thin
leaves let some light through. The plants also cast their own shadows. Three
cascaded shadow maps (2048², 2048² and 1536²) are drawn from the sun, or from
the moon at night, depth-only and alpha-tested. They cover horizontal radii
of 50, 400 and 2000 metres around the camera, fitted vertically to the terrain
and crowns. Cascades blend smoothly over the last quarter of each detail
band, keeping the coarsest map out of the nearby forest. Shadow detail
depends on distance across the ground, so flying
up and looking down preserves detailed shadows on the plants below. Each
cascade moves in whole texels, keeping shadows stable as the camera moves.
Trees shade the ground, the grass, each other and their own
crowns, out to the far cascade. The cull pass gathers shadow casters even
behind the camera, at a cheaper LOD per cascade. The tree crowns are also
laid over the grass's 512-metre capture from above, so the grass thins out and
stays short in the shade under a closed canopy and grows back in its gaps.

The diagnostics panel's **Vegetation** section turns the plants and their
shadows on and off, scales every LOD distance with **Plant detail distance**,
and reports how many plants and chunks are streamed. `--no-vegetation` skips
loading the library entirely.

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

The rivers (see below) are routed separately over the uneroded heights, and
the erosion treats their channels as fixed drains. Open the developer menu
with 2, click **Advanced controls**, and enable **Flow visualization** to
inspect the erosion flow output on the terrain, or expand **Flow output** to
inspect the cached target. Erosion
parameters can be edited from the same panel and applied with **Regenerate
erosion cache**; besides the runoff controls they include stream power,
channel incision, alluvial deposition, maximum incision, talus slide and
rockfall rates. The panel also reports maximum incision, deposited height,
small-scale detail within actively eroded ground, flow-axis bias, the largest
contributing area, the share of bare bedrock and the mean loose cover for the
most recently completed tile.

## Rivers and creeks

Rivers and creeks run from the mountains to the sea, through lakes where
their water fills a basin. They are generated in `src/rivers` for a 16 km
region around the player, follow the land the way its water would, are cut
into the terrain, lined with bank plants, and drawn with their
own flowing-water shader; their current pushes the player about.

### Where they run

The base landform is sampled on a world-aligned 32 m grid over the region and
a 2 km margin. A priority flood from the sea and the grid's edge gives every
cell a downhill receiver, so closed hollows drain over their lowest saddle the
way a lake would spill, and contributing area accumulates down that tree.
Each cell counts by its own rainfall: the uplands wring up to three and a
half times as much water from the weather as the coast, so mountain-fed
streams carry more than their catchment alone would. A channel begins where
about a third of a square kilometre of that weighted catchment gathers on
level ground, and with a little less on steeper ground; once begun it runs on
to the sea. At a confluence the larger branch keeps its course and the
smaller one ends on it. A creek on a mountainside is drawn only from where
the land along it first eases below a slope of 0.22: the steep slope above is
seeps and sheet wash, not a channel.

The coarse grid decides the network, which catchments drain where and how
much water each river carries, but not the course: a 32 m cell is wider than
many valleys. Each river's course is traced the way its water would find it,
over a 4 m grid of the natural ground in a corridor 72 m either side of the
coarse path. A priority flood from where the river leaves the corridor (the
sea, the channel of the river it joins, or the edge of the domain) fills
every hollow to the level it spills at, and the water runs down the steepest
descent of that filled surface: along the valley floor, around every spur
and knoll, across a filled hollow by the shortest way to its spill. The grid's
stair-steps are smoothed out and the path settled back onto the lowest ground
across it. Its bends are the land's own, so a lowland creek wanders over its
floodplain while a mountain stream keeps to the bottom of its V. A tributary
runs until it reaches its parent's channel, wherever the ground brings it
there; its corridor reaches the parent's actual course. A course that comes
down beside a river already there (two streams settling onto one valley
floor) joins it where they first meet, rather than running on alongside it
in a channel of its own. A tributary arrives at its parent's level, as real
confluences do, its mouth drowned in the parent's water and calm: over its
last metres it stands at that level and above them rises at most 3 cm per
metre, its channel cut down for it (a gorge where it comes down a steep
valley side, and no more than 2.5 m farther up, fading to nothing 60 m up
it, where any steeper water stays as rapids). A lake's water keeps its level:
where a tributary leaves a pond too close above the confluence to fall at
3 cm per metre, it runs down from the pond's level as one even rapid to where
it meets its parent, rather than dropping off the pond's edge. Where water seems held
in a basin it could in fact leave, the corridor was too narrow to show its
way: a basin that drains to the sea through a gap the coarse grid missed
sends its river out through that gap, and any other is routed again through
a wider corridor.

The water surface follows the channel's ground a hand or two below its banks
(0.1 m plus a seventh of its depth) and only ever falls downstream: the
surface is the falling profile that best fits the ground along the river (a
least-squares, never-rising fit), so where the ground rises over a bump and
falls again the river neither cuts the whole bump away nor floods the hollow
behind it, but meets them halfway, never standing above the natural ground.
Down a steep reach that fit falls in steps, flat behind each little rise and
steep between, which the channel and its banks would terrace into the
hillside; there the surface is smoothed over a few channel widths (never
lifted above the lowest ground it has passed in the last few widths, so it
cannot trace the rises, though below a cliff the fit drops off it is lifted
to round the drop into a rapid),
and across a bench it keeps falling at no less than a third of the reach's
slope, so a mountain stream runs evenly down its rapids.
A hollow
the water would stand less than 2 m deep in at its deepest, or less than
0.6 m on average (a flooded flat), is crossed in a cut through its rim, as a
river incises the sill it spills over. A deeper basin holds a lake or pond:
the whole basin (found by flood fill beyond the corridor) fills to just under
the rim it spills over (the fill sees each 4 m cell at its centre, so the rim
is checked again metre by metre, and where it dips lower between centres the
water stands no higher than that gap), every river that reaches it shares it (a basin
holding a smaller lake drowns it), the reaches above it are backed up to its
level, and the river leaves it at its outlet over a sill: for its first 25 m
the water draws down gently from the lake's level and always stands a little
under the ground beside it, so the outlet has banks from the start. A river
crossing a lake is in it from where its course enters the lake's cells to
where it leaves them; a short stretch where its course strays over a bar or
a ragged edge of the lake counts as the lake's, and a brush with its shore
opens no mouth. Water runs down every slope, however steep, as rapids; there
are no waterfalls.

A river to the sea runs out into open water: the sea is only what joins
water a metre deep (on the coarse routing grid and again on the fine one,
through water at least 0.3 m deep), so a hollow on a beach that dips just
below the sea's level fills and spills like any basin instead of swallowing
the river. Where the land drops to the sea down a beach's face the river is
graded down to it along a chord from its surface 120 m above the shore,
cutting a notch at most 3 m deep through the beach and the dune behind it;
from where its water reaches the sea's level, the sea fills its channel out
to the open water. Nothing holds the ground up beside a river's last stretch
to the sea, or beside it where it runs into or out of a lake, so no levee
raises a bar across its own mouth between its water and the still water it
meets.

Hydraulics follow from the catchment and the slope. Bankfull discharge grows
with the catchment, and the channel follows downstream hydraulic geometry,
the power laws real rivers are fitted with, plus the slope's own terms: width
grows as the square root of the discharge and shrinks with the slope of the
valley over some 40 m (as S^-0.25), and depth grows as Q^0.4 (Leopold &
Maddock) and a little with the slope, so a creek carrying a cubic metre a
second is about 0.95 m deep over its thalweg and seven to ten times as wide
as it is deep. Its bed is a flat-bottomed bowl, deep right up to the banks
rather than shoaling over a broad parabola. The water runs at Manning's speed
for a gravel and boulder bed (Jarrett's roughness): around 0.7 m/s on the
lowland, up to about 2 m/s down rapids. A mountain stream is held in a
narrower, deeper channel in boulders and bedrock; on the flat the same water
spreads wide and shallow over its own gravel and silt. Around spawn, reaches
draining over half a square kilometre average 9.3 m wide and 1.06 m deep on
the lowland, 6.7 m and 1.16 m on moderate slopes and 5.0 m and 1.0 m on steep
ones, and no creek is narrower than 1.6 m. A river widens below a confluence
and narrows into a steeper reach gradually, never by more than 2.5 cm per
metre along it; it swells a little and narrows again, its bed runs through a
pool every five to seven widths and at every tight bend (a third deeper and
slow) with a shallow riffle between (a quarter shallower, quick and broken
into small standing waves), and it churns white down its rapids. A stream
rising at a spring starts as a trickle a few hands wide, sunk in the ground
rather than walled at its head, its little channel fading up the slope above
it; over its first 40 m it gathers into a runnel and only then into broken
water. A spring in a hollow lower than the lake, sea or river its water runs
to rises where its water first lies under the ground, and the hollow above
stays dry rather than holding a trickle over it. A river that rises in a
lake is its outlet, full from the start. Where it runs into a lake
or the sea it spreads and slows into it, as a mouth does, and an estuary is
scoured a little deeper; where it leaves a lake over its sill it widens only
a little and keeps its pace. The thalweg hugs the outside of every bend.

Every choice is keyed by world position or by a river's head cell, so two
regions that both see a whole catchment agree on it; when the player moves
on, the next region is built on the async compute pool (about a second) and
swapped in. The first region is built before anything streams.

`--river-map path.png` charts a region from above (shaded relief, channels
and lakes) and prints its statistics, how well the channels fit the land (how
much of their length is trenched through a rise, held in by an embankment,
or running along a slope above its valley's floor), how much of the lakes'
sheet edges would stand over lower ground, how cleanly the rivers meet still
water (rivers darting out of a lake and back, how far outlets fall below
their lakes, river mouths cut off from the open sea, how far sheets dip
under a river where they meet), and the rivers and lakes nearest
`--map-centre`, with each lake's inlets and outlets; `--map-extent` defaults
to 8 km here. The spawn region holds some 160 rivers, 86 km of channel and
170 lakes and ponds covering 280 ha; on gentle reaches the water stands about
0.8 m under the natural ground on average (half of it under 0.56 m), under
1 % of their length is trenched more than 3 m into it, and one of 127 river
mouths does not reach open sea (a creek down a sea cliff).

### How they shape the ground

A river is a chain of short carve segments, each bounding the ground above
and below near its centreline. Inside the wetted width the bed follows a
parabola below the water, skewed toward the outer bank of a bend; past the
waterline a bank cone rises, steep on a cut bank and gentle on a point bar,
and ground standing above it is cut back into a bank; just beyond the
waterline the ground is held a little above the water and falls gently away
from it, the broad, low natural levee a river builds, so the channel always
contains its river. Bounds combine across segments (minimum above, maximum
below), so confluences open into each other, and where the carve meets the
natural ground its creases are rounded (a smooth minimum and maximum over
0.8 m of height), so a bank's top curves over into the land above it rather
than breaking at an edge the terrain's triangles would draw as a saw-tooth.
A bank's slope varies only over many widths along the river, so its top does
not jog in and out from one segment to the next. A segment has a flat start and a
round end, so it never reaches back up the channel over ground the segments
above it shape, and past its end its levee falls on as its water does: down
a rapid steeper than the levee's outer slope, an end held up at its own water
would stand over the next segment's lower levee, a ledge at every segment and
a flight of steps down the bank.

The same arithmetic runs on the CPU (`src/rivers/carve.rs`) and in the shared
`assets/shaders/river-functions.wgslinc`, over the same uploaded segments and
a 32 m lookup grid, so the drawn ground, the player's footing, the seated
plants, the lighting heightfield and the erosion all see one
channel; tests hold the shader copies identical and the GPU result to the
CPU's. The erosion tiles simulate over the carved base, and cells under a
river's water are fixed drains there, like the sea: the river carries away
what reaches it, so tributary gullies grade to it and nothing fills or
trenches the channel.

Lakes carve nothing. Each lookup-grid cell a lake reaches carries the lake's
surface at the corners of its 4 m cells (in 2 mm steps under the cell's highest
corner), drawn between them over the same two triangles as the lake's sheet.
Under the sheet it is the sheet, sunk edges and all, so ground below it is
exactly the ground the drawn water covers: plants keep out of it, the terrain
shades its bed and shore, and the player wades and swims in it. Past the
sheet the surface runs on under the ground, at the lake's level beneath a
rising shore, so the shore is measured as if the water went on, and two
metres under lower ground (beyond a rim, down an outlet), sinking away from
the water as it goes, so that where it ends, 24 m out, any shore band
measured from it has long faded: no band stops on a cell's edge. Where two
lakes reach the same corner the higher surface counts. The erosion leaves
the lake and its shore up to 2.5 m above it as they are, a drain for what
runs into it, so no gully cuts down below the water at its edge.

The terrain material shader reads the river or lake at every vertex, and
exactly at every pixel where water may lie within a triangle's span (beyond
the first ring the clipmap's triangles are wider than a creek, and would
smear its bed across them). The bed sorts by the power of the water over it
(Hjulström's thresholds at the water's Manning speed): silt and mud where the
current stays under about 0.3 m/s (lake beds, pools and the slack water along
the banks), gravel through runs and riffles, bedrock under rapids. Under the
water it is darkened by its film of algae and settled silt, and it is matte:
the water's surface carries the reflections.

Out of the water every river and lake is edged with bare earth, as the sea is
with its beach. The band is measured up the shore (`river-shore.wgslinc`,
shared by both terrain stages): horizontally on a gentle shore and six metres
per metre of rise on a steep one, in the water's own size, so it spreads over
a floodplain or a flat pond margin, narrows to a strip on a cut bank, and is
narrower beside a creek than beside a river or a pond. From the water up it
is dark, glistening mud where the water laps, damp earth, then pale dry silt,
and turf closes over it within a few metres through tufts, bays and tongues
that wander at the beach's own scales; past it the turf grows lush and dark
on the moist soil. The bank adds its own story: the outside of a bend is
undercut into a face of bare soil to its lip, the inside keeps a bar (gravel
where the stream runs quick, silt where it is slow), and beside whitewater
the banks are stone, splashed wet. Along calm water the rules that bare a
hillside of the same steepness to scree, or leave the erosion's furrows
gravelly, give way to the band; inland, sand is the sea's alone. The grass
habitat looks the water up exactly at every texel of its capture, so no
blade roots in a creek however narrow or far off, nor on the bare band. Every
edge wanders, so no waterline runs parallel to its channel, and running water
never holds snow.

### Plants

No plant stands in a river or a lake, nor on the margin its floods scour:
trees and shrubs keep at least four metres back from the waterline (more for
a tall tree, and a broad one by three quarters of its crown's radius as well,
so it leans over a creek or a pond's outlet rather than spreading across
it), ground plants a metre and a half, and lavender, which wants dry
ground, three. The scatter keeps each layer's stems to its margin, the GPU
cull refuses any root it would seat closer to a channel or a lake's
shoreline, and the grass habitat refuses roots in the water and on the wet
margin, as well as on bare cut banks and gravel bars. Riverbanks are damp
ground in the ecology, and broadleaf plants line them the way they line the
shore: in colonies along the strip of bank behind the open margin, thickest a
few metres back from the water and thinning beyond, tolerant of a gallery
forest's shade.

### The water

Each river's surface is a ribbon across its channel at the water level,
reaching under both banks so the waterline is wherever the carved bank rises
through the water. A lake is a flat sheet at its level over its basin, in
4 m cells: over the basin, every metre of ground below its level joined to
it that the cell-by-cell fill stepped past, the closed hollows beside it
(which fill to its level, as a waterlogged hollow by a pond does) and a few
cells up the shore around them, so its shoreline is wherever the ground
meets the water. It never reaches past its outlet or over a narrow rim,
where the ground falls away below its level (nor along an outlet's banks
below the sill), nor over a river's channel, and wherever the ground under
its outer edge still lies lower than the water, that edge sinks just under
the ground, so the sheet never ends in the air. Where a river meets a
lake, coming in or going out, the two surfaces cross inside the sheet's last
cell: the sheet's edge slips just under the river's water, and the river's
ribbon runs on just under the sheet, a little deeper the further in, until it
is hidden for good. They never share a plane, so nothing flickers, and the
river's current runs on into the sheet where they meet. At the sea, water
standing at the sea's level is the sea's: the ribbon ends a node past the
coast, tucked just under the sea's surface, and over its last 90 m the
river's water turns to the sea's, its colour and clarity and the light in
it, as its rapids and foam give out, so the two meet in one water.
Water is drawn to 3.2 km in 256 m chunks culled against the view; beyond a
few hundred metres, where the clipmap's triangles are wider than a creek, a
river's surface is lifted by about the bank those triangles leave so it still
shows from a ridge (a lake, wide enough for any triangles, never is, and nor
is a river near the still water it meets, so the two still meet from afar).

The river shader (`vs_river`/`fs_river` in `water-surface.wgsl`) shades the
water with the sea's own optics, sky and screen-space reflections, sun
glitter, refraction, rain rings and fog, and the plants' shadow cascades. Sun
and moon glints are a normalised GGX lobe weighted by the water's own Fresnel
at each facet (the sea's glints use the same), so water mirrors the sun only
where its facets line up, as sparkles and a soft path, never as a white
sheet. On top of that:

- **Brown, tannin-stained water.** The forest's dissolved organic matter
  absorbs blue most, so a riffle shows a golden bed, a pool an amber one and
  deep water goes dark brown; lakes are darker still, and rapids are milky
  with bubbles. The eye's path is bent down into the water and the light
  scattered back out of it is dimmed with depth, so depth reads.
- **What the water mirrors.** Where the screen cannot show a reflection, the
  shader walks out along the reflected ray's bearing over the ground and the
  canopy (the shore map near the camera, the lighting map beyond, and the
  share of open sky the tree crowns leave) and finds the horizon the banks
  and crowns raise: below it the water mirrors dark banks and foliage, above
  it the sky. A creek in its cut under the trees is dark but for a strip of
  sky down its corridor; a pond in a clearing keeps its sky.
- **The stream's own frame.** Ripples live in metres down and across the
  channel, so they follow every bend. Two advected phases stream at the
  water's own speed, drawn out along fast water; standing waves stand still
  over riffles, fixed to the bed, while ripples and foam stream through
  them; boils swell glassy on runs and pools; gusts dull open water in cat's
  paws. What a pixel cannot resolve becomes roughness by its true slope
  variance, per axis of the flow, so distant water does not turn white.
- **Foam that follows the flow.** Rapids churn white over the steps and
  boulders of their bed, with dark glassy tongues of water between; foam they make drifts
  downstream for some fifteen seconds as lace gathered on the seams by the
  banks, on a tongue down the current and as scum in slack water, and an
  inlet carries it out into the pond. It is cream, stained like the water,
  and in the shade of the crowns it is lit only by the sky they leave.

### Wading

Flowing water pushes the player. The current's drag on the submerged body,
½ρC_dAu² over the legs and then the torso as the water deepens, is set
against the friction the feet can hold with, which buoyancy and a whitewater
bed reduce. Knee-deep water slows a wade, more so walking upstream; a gentle
current is stood against; a strong one carries the body along more and more,
and once its drag outweighs the footing it sweeps the player off their feet
and away downstream, down any rapids coming. Water too deep to stand in
floats the player with the current; a lake's still water only floats them.
Space kicks a swimmer up out of the water as it jumps a walker off the
ground, onto a bank or over a ledge; a body in the air over the water keeps
the motion it left the water with, so a hop neither shakes off the current
nor stops a body it was carrying dead.
The F1 panel's **Rivers** section
reports the network, the nearest channel and the current the player stands
in, and can hide the water surfaces.

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
The GPU also measures the highest terrain in this map. Light rays stop once
they rise far enough above that bound that every remaining tap is fully lit,
including the widening penumbra on surface shadows. This skips work without
changing the visibility computed by the original terrain shadow march.

The lighting heightfield, its maximum-height reduction, and the 512×512 shoreline
heightfield are cached until their terrain inputs or world mappings change.
Erosion tile uploads and reveals, lookup shifts, and river updates invalidate
both maps. Moving the player also invalidates the lighting map because erosion
fades with distance from the player. The shoreline map stays entirely inside
the full-strength erosion radius, so it can be reused within its eight-metre
mapping snap. Stationary frames after streaming settles reuse both captures and
copy the cached maximum into the frame's lighting uniforms.

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

Sunlight in the fog, including the fog over water, reuses a cloud shadow map
rebuilt every frame. It stores optical depth through the cloud layer along
sun rays in two camera-centred `1024×1024` levels: nearby rays use 6-metre
texels and distant rays use 48-metre texels, with a smooth transition between
the levels. Shadow strength still follows the weather at each fog sample.
Samples inside or above the cloud layer, beyond the map, or under a sun low
enough to clip the original shadow ray's distance limit use the original
cloud march. Direct cloud shadows on terrain and water, underwater lighting,
and moonlight retain their original marches. These optimizations keep the
existing quality presets, sample budgets, scene resolution, half-resolution
fog/cloud target and reflection probe resolution.

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

All three presets keep the same effects enabled. The advanced rendering switches
can disable individual effects independently, and `--no-raymarch` disables the traced
effects together. Fog and clouds share a half-resolution integration target
with a depth-aware upscale; the scene keeps its full output resolution. Cloud
rays stop early once sufficiently opaque and skip lighting in empty regions.
Clouds still add substantial GPU work, especially with high quality, large
display resolutions or many partially transparent formations. Lower raymarch
quality or disable clouds independently if the frame rate becomes too low.

## Render pipeline

The renderer writes view position, view normal, and albedo to a three-target
G-buffer: the terrain first, then the plants (after their cull pass), then the
grass. The plants also draw their three shadow cascades. SSAO is evaluated at half resolution with a 24-sample rotated
hemisphere kernel, followed by a depth/normal-aware bilateral blur. A separate
half-resolution atmosphere pass supplies combined fog/cloud scattering and
transmittance to the lighting composite, which also evaluates terrain, cloud
and plant shadows and the celestial sky. A cloud sky probe supplies reflection directions
outside the current view, and the cloud shadow map supplies sunlight
occlusion for fog in the atmosphere and water passes.
The water surface then samples the opaque scene for refraction and raymarched
reflections, followed by underwater effects where applicable. The sea and the
rivers depth-test against the G-buffer's own depth buffer in hardware, ahead
of their shaders, so water hidden behind terrain or plants is never shaded;
they write that depth too, so a river never paints over a nearer wave. FXAA smooths the
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
