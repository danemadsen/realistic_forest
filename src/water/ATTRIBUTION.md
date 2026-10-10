# Attribution

The code in `src/water/` and the water shader `assets/shaders/water.wgsl` are
derived from **bevy-aqua** (<https://github.com/sayhisam1/bevy-aqua>),
licensed **MIT OR Apache-2.0**. Aqua is itself a port of **Crest Ocean System**
for Unity (MIT) and of the Godot ocean shader, and its underwater medium,
fullscreen medium pass and water-surface underside design came from
**PR #9 by @wellscrosby** (commit `8f236ee`, MIT OR Apache-2.0).

Aqua's `ATTRIBUTION.md` asks that this attribution be preserved in derived
distributions. That is what this file is. Keep it if any of the files below
are copied out of this repository.

## Per-file provenance

| This repository | Upstream |
| --- | --- |
| `src/water/waves.rs` | `bevy-aqua-waves/src/lib.rs` (spectrum, band partition); Crest `Scripts/Shapes/ShapeGerstnerBatched.cs`, `Shaders/OceanInputs/GerstnerShared.hlsl`, `OceanWaveSpectrum.cs` |
| `src/water/rings.rs` | `bevy-aqua-geom/src/rings.rs`; Crest `Scripts/OceanBuilder.cs` (`BuildOceanPatch`, `CreateLOD`) |
| `src/water/optics.rs` | `bevy-aqua-core/src/lib.rs` (`WaterOptics`, `WaterOptics::PRESETS`) |
| `assets/shaders/water.wgsl` (`vs_sea`, `fs_water`) | `bevy-aqua-waves/src/anim_waves.wgsl` (shoaling, Gerstner sum), `bevy-aqua-core/src/cascade/material.wgsl` (foam, SSS, Fresnel composition), `bevy-aqua-optics/src/optics.wgsl` (`godot_fresnel`, deep-water weight), `bevy-aqua-medium/src/medium.wgsl` (Beer-Lambert) |
| `assets/shaders/water.wgsl` (`fs_underwater`) | `bevy-aqua-volume/src/volume.wgsl` and `bevy-aqua-medium/src/medium.wgsl` |

## What was changed

This is a port, not a copy. The upstream water is a Bevy 0.19 `bevy_pbr`
material; this renderer is a bespoke Bevy 0.17.3 `RenderGraph` with no
`bevy_pbr`. The waves are evaluated directly in the vertex shader rather than
baked to a cascade texture array, the tile placement is an instance buffer
rather than an entity hierarchy, and every `bevy_pbr` dependency (light
probes, clustered lights, shadow maps, the depth prepass, the transmission
texture, the atmosphere LUT) has been replaced by this renderer's own single
sun, deferred G-buffer and composited frame.

Not ported: the FFT spectral wave model, planar reflections, screen-space
reflections, caustics, spray, motion vectors, bounded water bodies, and the
GPU wave-query readback. See the module comment in `src/water/mod.rs`.

The Gerstner set keeps Crest's wavelengths and amplitudes, drawn from the
same stream, but not Crest's directions or phases. Crest's generator gives
each component one index within its octave that sets its wavelength, its
direction stratum and its phase eighth together, and draws the phases from
the wavelengths' own seed, so every octave repeats the same fan in step and
the sea shows rows of dimples. `src/water/waves.rs` draws the directions from
a measured directional spectrum (Donelan, Hamilton and Hui's spread with
Banner's extension, floored at Cox and Munk's slope ratio) through strata
shuffled independently in every octave, and the phases uniformly, each from
a seed of its own.

Not from aqua: the rivers and lakes the same surface shader draws, their
current, their inland optics and the wind sea every water body shares
(`vs_inland`, `windSea`, `flowSurface` and `inlandMedium` in
`assets/shaders/water.wgsl`) are this project's own.
