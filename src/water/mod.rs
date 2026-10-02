//! Vendored ocean renderer.
//!
//! Derived from bevy-aqua (<https://github.com/sayhisam1/bevy-aqua>), MIT OR
//! Apache-2.0, which is itself a port of Crest Ocean System for Unity and of
//! Godot's ocean shader. The per-file provenance and the upstream attribution
//! requirements are recorded in `src/water/ATTRIBUTION.md`; read it before
//! redistributing.
//!
//! # Why this is a port and not a dependency
//!
//! Aqua draws its water as a `bevy_pbr::MaterialPlugin<CascadeMaterial>` mesh
//! on Bevy 0.19: the material's pipeline layout is built from bevy_pbr's
//! `MeshPipeline` view layout, its binding-array layout, and its mesh bind
//! group, and the shader imports `bevy_pbr::mesh_view_bindings` and bevy's
//! depth prepass and transmission textures. This renderer is a bespoke Bevy
//! 0.17.3 `RenderGraph` with no bevy_pbr at all — it has its own deferred
//! G-buffer, its own single-sun composite, and its own clipmap terrain. A
//! verbatim vendor of aqua's crates cannot compile here.
//!
//! What is vendored is therefore aqua's *model*, restructured into this
//! renderer's idiom:
//!
//! - [`waves`] — the 40-component Crest Gerstner spectrum and its LOD band
//!   partition, ported essentially verbatim.
//! - [`rings`] — Crest's concentric tile topology ("patches"), ported
//!   verbatim; the tiles become an instance buffer instead of entities.
//! - [`optics`] — the `WaterOptics` presets and their four-`vec4` shader
//!   contract, ported verbatim.
//! - `assets/shaders/water-surface.wgsl` — the surface shading: shoaling,
//!   chop, Fresnel, Beer-Lambert body, refraction of the composited frame,
//!   foam. This is a consolidation of aqua's `cascade/material.wgsl`,
//!   `waves/displace.wgsl`, `optics/optics.wgsl` and `medium/medium.wgsl`
//!   onto one sun and one opaque buffer.
//! - `assets/shaders/water-underwater.wgsl` — the submerged-camera medium
//!   pass, from aqua's `volume.wgsl` + `medium.wgsl`.
//!
//! Deliberately not ported: the FFT spectral wave model, planar reflections,
//! screen-space reflections, the environment cubemap probe, clustered local
//! lights, caustics, Hanabi spray, motion vectors, bounded water bodies and
//! rivers, and the GPU wave-query readback. Each needs renderer facilities
//! this project does not have (a light probe, a motion-vector prepass, a
//! second camera, an HDR target). The analytic Gerstner model, the cascade
//! rings, the optics and the underwater medium are the parts that make the
//! ocean read as water, and those are all here.

pub mod optics;
pub mod rings;
pub mod waves;

pub use optics::{WaterOptics, WaterSettings};

/// Concentric tile LODs, coarse to fine. Matches aqua's `LOD_COUNT`.
pub const WATER_LOD_COUNT: usize = 5;

/// Quads across one tile edge. Matches aqua's `TILE_RESOLUTION`.
pub const WATER_TILE_RESOLUTION: usize = 64;

/// Width of an LOD 0 tile in metres. Matches aqua's `BASE_SCALE`.
pub const WATER_BASE_SCALE: f32 = 24.0;

/// Exponent of the surface's Fresnel mix, `F0 + (1 - F0)*(1 - cos)^e`: the
/// shape of how reflectance climbs from the near-normal floor to the grazing
/// mirror. Aqua exposes it as a free authoring value rather than deriving it,
/// so this is the named home for that freedom.
///
/// It is not a free parameter in practice, because the exact Fresnel curve for
/// water fixes it. For n = 1.333, a least-squares fit of this curve gives
/// e = 5.58 over 0-89 degrees, 5.70 over 70-88, and 5.97 over 80-89. The old
/// value of 5.0 is Schlick's fit for a *dielectric in general*, and over water
/// it runs four to six points of reflectance high through exactly the band a
/// standing eye reads as the mirror — +5.0 points at 80 degrees incidence,
/// +5.9 at 84, +3.7 at 88. That is the sea looking glossier than water does,
/// and it is what this constant corrects.
///
/// 6.0 is the fit weighted to that band rather than the uniform one. It lands
/// within 0.8 points of the exact curve from 84 to 88 degrees (84 is +0.02),
/// and pays for it with 2.4 to 3.4 points of *under*-reflectance at 60-70
/// degrees, where the surface is mostly body colour anyway. Nothing here can
/// soften the last degree or two of horizon: `(1 - cos)^e` still reaches 1 at
/// cos = 0 for every exponent, so the true horizon stays a full mirror.
///
/// It applies to all three optics presets, to the surface seen from below, and
/// it cannot be reached from the diagnostics window — it is not a `WaterOptics`
/// field, by the same argument that keeps `F0` out of one: the underwater pass
/// has no Fresnel at all, so a per-preset value would let the two views drift.
pub const FRESNEL_EXPONENT: f32 = 6.0;

/// How far the tile layout is snapped, in metres, before the ring centre
/// moves. Coarser than one LOD 0 tile, so the innermost ring never crawls.
/// Aqua snaps and geomorphs instead; without vertex morphing a plain snap is
/// exact, because the Gerstner sum is evaluated from the world position, not
/// from the tile's local coordinates — a tile that jumps by a whole number of
/// its own quads displaces its vertices to exactly the same world heights.
pub const WATER_SNAP: f32 = WATER_BASE_SCALE;

// The medium's own constants — aqua's `PATH_LENGTH_MAX` (256 m), `N_WATER`
// (1.333), `PARTICLE_SCATTER` and `RAYLEIGH` — live in
// `assets/shaders/water-underwater.wgsl`, which is the only place that
// evaluates them; the port does not mirror them here, so there is no second
// copy to drift. Aqua's `TRANSMISSION_OPAQUE_OPTICAL_DEPTH` clamp is not
// ported: the underwater pass lets `exp(-sigma_t * path)` fall to zero on its
// own.

/// One Gerstner component as the shader sees it. Seven floats padded to eight
/// so the array stride is 32 bytes in both Rust and WGSL.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuWave {
    pub direction: [f32; 2],
    pub amplitude: f32,
    pub wave_number: f32,
    pub angular_frequency: f32,
    pub phase: f32,
    pub chop_amplitude: f32,
    pub wavelength: f32,
}

/// The water surface's material block, group 2 binding 0.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct WaterStageUniforms {
    pub waves: [GpuWave; waves::WAVE_SLOTS], // 0    1280
    pub ranges: [[u32; 4]; WATER_LOD_COUNT], // 1280 80
    /// x elapsed seconds, y authored amplitude, z sea level, w slowly varying
    /// weather wave gain. Wave components stay fixed while fronts pass, so
    /// their phases remain continuous.
    pub params: [f32; 4], // 1360
    /// rgb extinction per metre, w scatter scale.
    pub extinction: [f32; 4], // 1376
    /// rgb scatter tint, w Henyey-Greenstein asymmetry.
    pub scatter: [f32; 4], // 1392
    /// x Fresnel F0, y Fresnel exponent, z sun roughness, w base scale.
    pub surface: [f32; 4], // 1408
    /// rgb subsurface tint, w tile resolution.
    pub sss_tint: [f32; 4], // 1424
    /// x refraction scale, y foam scale, z max wave amplitude,
    /// w significant wave height (4 * sqrt(sum(amplitude²) / 2)), metres.
    pub misc: [f32; 4], // 1440
    /// x flat-surface debug, y sea state amplitude, z wind radians, w wave fade end.
    pub flags: [f32; 4], // 1456
    /// xy local seabed-map centre, z world span, w metres per texel.
    pub shore_map: [f32; 4], // 1472
}

const _: () = assert!(std::mem::size_of::<GpuWave>() == 32);
const _: () = assert!(std::mem::size_of::<WaterStageUniforms>() == 1488);

/// The submerged-camera medium block, group 2 binding 0 of the underwater
/// pass.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct UnderwaterUniforms {
    /// x sea level, y camera height above it, z elapsed seconds, w fade.
    pub params: [f32; 4],
    /// rgb extinction per metre, w scatter scale.
    pub extinction: [f32; 4],
    /// rgb scatter tint, w asymmetry.
    pub scatter: [f32; 4],
    /// rgb sun radiance reaching the surface, w moon radiance scale.
    pub sun: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<UnderwaterUniforms>() == 64);

impl Default for WaterStageUniforms {
    fn default() -> Self {
        Self {
            waves: [GpuWave::default(); waves::WAVE_SLOTS],
            ranges: [[0; 4]; WATER_LOD_COUNT],
            params: [0.0, 1.0, crate::constants::SEA_LEVEL, 1.0],
            extinction: [0.86, 0.24, 0.39, 1.0],
            scatter: [1.0, 1.0, 1.0, 0.8],
            surface: [0.02, FRESNEL_EXPONENT, -1.0, WATER_BASE_SCALE],
            sss_tint: [0.06, 0.55, 0.45, WATER_TILE_RESOLUTION as f32],
            misc: [0.5, 1.0, 1.0, 0.0],
            flags: [0.0, 1.0, 0.0, rings::horizon_wave_fade_end()],
            shore_map: [0.0; 4],
        }
    }
}

impl Default for UnderwaterUniforms {
    fn default() -> Self {
        Self {
            params: [crate::constants::SEA_LEVEL, 0.0, 0.0, 1.0],
            extinction: [0.86, 0.24, 0.39, 1.0],
            scatter: [1.0, 1.0, 1.0, 0.8],
            sun: [0.0, 0.0, 0.0, 0.0],
        }
    }
}

/// Total vertical displacement bound of the current spectrum, used to reject
/// water fragments early and to size the underwater test.
pub fn displacement_bounds(spectrum: &waves::WaveSpectrum, flat: bool) -> f32 {
    if flat {
        return 0.0;
    }
    spectrum.waves.iter().map(|wave| wave.amplitude.abs()).sum()
}
