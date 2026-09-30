//! Diagnostics panel and screen-space overlays, ported from the C++
//! `DrawDiagnostics`, `DrawErosionLattice` and `DrawCrosshair`, plus the
//! `ImGui::Image` preview of the erosion flow atlas.
//!
//! PORT NOTES (egui vs ImGui/rlImGui):
//! - UNITS: ImGui measures in logical pixels; raylib's 2D drawing measures in
//!   physical ones. egui's points are logical pixels, so the window's
//!   position, size and every widget transfer 1:1, while each physical-pixel
//!   figure the C++ used for the overlay (screen size, the 1 px line widths,
//!   the 20 px hint font, the 6 px crosshair arms) is divided by
//!   `pixels_per_point` on the way in.
//! - WINDOW BACKGROUND: `ImGui::SetNextWindowBgAlpha` *multiplies* the theme's
//!   window background alpha rather than replacing it, so the port scales the
//!   egui theme's own `window_fill` alpha by 0.88.
//! - LAYER ORDER: the C++ drew the overlays with raylib and then let ImGui
//!   render over them. The port paints them on `LayerId::background()`, which
//!   egui keeps below the window layer, reproducing the same stacking.
//! - GAMMA: raylib blended the overlays straight into a non-sRGB default
//!   framebuffer, i.e. in the display-encoded domain. The egui pass blends in
//!   linear space into the sRGB-format main texture (bevy's convention — see
//!   the sRGB note in fxaa.wgsl), so the overlay colours are exact but their
//!   alpha ramps mix in a different space than the C++'s. The egui widgets
//!   themselves carry the same, unavoidable shift.
//! - FRAME ORDER: the C++ computed `uiWantsInput` and drew the panel before
//!   `UpdatePlayer`; egui's pass runs in `PostUpdate`, after the player
//!   update. The panel therefore reports the current frame's player state
//!   (the C++ reported the previous frame's) and `ui_wants_input_system`
//!   reads what the previous frame's pass produced — which is exactly the
//!   one-frame relationship ImGui's `WantCaptureMouse`/`WantCaptureKeyboard`
//!   have, since those also resolve against the previous frame's layout.
//! - TEXT: raylib's built-in bitmap font has no egui counterpart; labels and
//!   values use egui's proportional font at the same logical size. ImGui's
//!   `%.Nf` formats are reproduced with `Slider::fixed_decimals`, and
//!   `ImGui::SeparatorText` (a label with a rule filling the row) is painted
//!   by hand because egui's `Separator` carries no text.

use crate::RerunErosion;
use crate::automation::AutomationSettings;
use crate::constants::*;
use crate::erosion::{self, ErosionCache, ErosionTileState};
use crate::matrices;
use crate::noise::NoiseField;
use crate::player::{Player, PlayerCamera, UiWantsInput};
use crate::render::gpu_textures::GpuWorldTexturesOption;
use bevy::image::BevyDefault;
use bevy::math::Vec3;
use bevy::prelude::*;
use bevy::render::render_phase::TrackedRenderPass;
use bevy::render::render_resource::{BindGroup, BindGroupLayout, RenderPipeline};
use bevy::render::renderer::RenderDevice;
use bevy::render::sync_world::RenderEntity;
use bevy::render::view::ViewTarget;
use bevy::render::RenderApp;
use bevy_egui::render::{EguiBevyPaintCallback, EguiBevyPaintCallbackImpl, EguiPipelineKey};
use bevy_egui::input::EguiWantsInput;
use bevy_egui::{egui, EguiContexts};

/// The 0.88 alpha `ImGui::SetNextWindowBgAlpha(0.88f)` requests.
const WINDOW_BACKGROUND_ALPHA: f32 = 0.88;
/// `DrawErosionLattice`'s half-extent of the drawn lattice, in metres.
const LATTICE_EXTENT: f32 = 1600.0;
/// `DrawErosionLattice`'s maximum segment length, in metres.
const LATTICE_SEGMENT_STEP: f32 = 64.0;

/// Mirrors the C++'s `uiWantsInput`: the panel is open and egui currently
/// claims the pointer or keyboard. Read before the frame's player update so
/// gameplay input yields to the panel, exactly as the C++ did.
pub fn ui_wants_input_system(
    settings: Res<AppSettings>,
    egui_wants_input: Res<EguiWantsInput>,
    mut ui_wants_input: ResMut<UiWantsInput>,
) {
    ui_wants_input.0 = settings.show_ui
        && (egui_wants_input.wants_pointer_input() || egui_wants_input.wants_keyboard_input());
}

/// `DrawDiagnostics` plus the three screen-space overlays the C++ drew after
/// `PresentFxaa` and before `imgui.Render()`.
#[allow(clippy::too_many_arguments)]
pub fn draw_diagnostics_ui(
    mut contexts: EguiContexts,
    mut settings: ResMut<AppSettings>,
    mut erosion_settings: ResMut<ErosionSettings>,
    cache: Res<ErosionCache>,
    noise: Res<NoiseField>,
    players: Query<&Player>,
    automation: Res<AutomationSettings>,
    mut rerun: ResMut<RerunErosion>,
) -> Result {
    let Ok(player) = players.single() else {
        return Ok(());
    };
    let Ok(ctx) = contexts.ctx_mut() else {
        return Ok(());
    };

    if settings.show_ui {
        draw_diagnostics_window(
            ctx,
            &mut settings,
            &mut erosion_settings,
            &cache,
            player,
            &mut rerun,
        );
    }

    // Screen-space overlays. The C++ issued these straight to the default
    // framebuffer after PresentFxaa, so they are never anti-aliased; painting
    // them on the background layer keeps them under the diagnostics window,
    // matching the C++'s line-then-ImGui order.
    let pixels_per_point = ctx.pixels_per_point();
    let content_rect = ctx.content_rect();
    let viewport = (
        (content_rect.width() * pixels_per_point).round().max(1.0) as u32,
        (content_rect.height() * pixels_per_point).round().max(1.0) as u32,
    );
    let painter = ctx.layer_painter(egui::LayerId::background());
    if settings.erosion_debug {
        draw_erosion_lattice(
            &painter,
            player,
            &cache,
            &noise,
            viewport,
            pixels_per_point,
        );
    }
    if player.mouse_captured && !settings.show_ui {
        draw_crosshair(&painter, viewport, pixels_per_point);
    }
    if !player.mouse_captured && automation.shot_path.is_none() {
        painter.text(
            egui::pos2(20.0, viewport.1 as f32 - 38.0) / pixels_per_point,
            egui::Align2::LEFT_TOP,
            "Click the world to capture the mouse",
            egui::FontId::proportional(20.0 / pixels_per_point),
            egui::Color32::WHITE,
        );
    }
    Ok(())
}

/// The "Infinite Terrain Lab" window. Mirrors `DrawDiagnostics` widget for
/// widget, including the ImGui formatting strings.
fn draw_diagnostics_window(
    ctx: &mut egui::Context,
    settings: &mut AppSettings,
    erosion_settings: &mut ErosionSettings,
    cache: &ErosionCache,
    player: &Player,
    rerun: &mut RerunErosion,
) {
    // ImGui's background alpha is a multiplier on the theme's window colour.
    let style = ctx.style();
    let base_fill = style.visuals.window_fill();
    let window_fill = egui::Color32::from_rgba_unmultiplied(
        base_fill.r(),
        base_fill.g(),
        base_fill.b(),
        (base_fill.a() as f32 * WINDOW_BACKGROUND_ALPHA).round() as u8,
    );
    let frame = egui::Frame::window(&style).fill(window_fill);

    // The close button writes through a local so the borrow of the window
    // builder does not alias the settings resource.
    let mut open = settings.show_ui;
    egui::Window::new("Infinite Terrain Lab")
        .frame(frame)
        .default_pos(egui::pos2(16.0, 16.0))
        .default_size(egui::vec2(380.0, 520.0))
        .open(&mut open)
        .show(ctx, |ui| {
            ui.label("GPU clipmap terrain + hydraulic erosion");
            ui.separator();
            // ImGui::GetIO().Framerate is a smoothed frame rate; egui's
            // stable_dt is smoothed the same way.
            let fps = 1.0 / ctx.input(|input| input.stable_dt).max(f32::EPSILON);
            ui.label(format!("FPS: {fps:.0}"));
            ui.label(format!(
                "Position: {:.1}, {:.1}, {:.1}",
                player.position.x, player.position.y, player.position.z
            ));
            ui.label(format!(
                "Movement: {}",
                if player.flying { "FLYING" } else { "WALKING" }
            ));
            ui.label(format!(
                "Clipmap reach: {:.0} m",
                (CLIP_CELLS as f32 * 0.5) * 2.0f32.powi(CLIP_LEVELS as i32 - 1)
            ));
            let ready_tiles = cache
                .tiles
                .values()
                .filter(|tile| tile.state == ErosionTileState::Ready)
                .count();
            let streaming_tiles = cache.tiles.len() - ready_tiles;
            ui.label(format!(
                "Erosion tiles: {ready_tiles} ready, {streaming_tiles} streaming"
            ));
            ui.label(format!(
                "Latest tile: {:.1}..{:.1} m ({:.0}% land)",
                cache.stats.minimum_height, cache.stats.maximum_height, cache.stats.land_coverage
            ));
            ui.label(format!(
                "Incision: {:.2} m  deposition: {:.2} m",
                cache.stats.maximum_incision, cache.stats.maximum_deposition
            ));
            ui.label(format!(
                "Detail: {:.3}  flow axis bias: {:.3}",
                cache.stats.erosion_detail, cache.stats.flow_axis_bias
            ));

            separator_text(ui, "Rendering");
            ui.checkbox(&mut settings.ssao_enabled, "SSAO");
            ui.checkbox(&mut settings.flow_debug, "Flow visualization");
            ui.checkbox(&mut settings.erosion_debug, "Erosion delta debug");
            ui.add(egui::Slider::new(&mut settings.ao_radius, 0.2..=6.0).text("AO radius"));
            ui.add(egui::Slider::new(&mut settings.ao_bias, 0.005..=0.3).text("AO bias"));
            ui.add(egui::Slider::new(&mut settings.ao_power, 0.4..=3.0).text("AO power"));
            ui.add(egui::Slider::new(&mut settings.ao_strength, 0.0..=1.0).text("AO strength"));
            ui.add(egui::Slider::new(&mut settings.ao_tex_strength, 0.0..=1.0).text("Texture AO"));
            ui.add(
                egui::Slider::new(&mut settings.fog_density, 0.0..=0.001)
                    .text("Fog")
                    .fixed_decimals(5),
            );
            ui.add(
                egui::Slider::new(&mut settings.sun_intensity, 0.0..=8.0)
                    .text("Sun intensity")
                    .fixed_decimals(2),
            );
            ui.add(
                egui::Slider::new(&mut settings.exposure, 0.25..=4.0)
                    .text("Exposure")
                    .fixed_decimals(2),
            );
            ui.add(
                egui::Slider::new(&mut settings.sparkle_strength, 0.0..=2.0)
                    .text("Snow sparkle")
                    .fixed_decimals(2),
            );

            separator_text(ui, "Terrain textures");
            ui.add(
                egui::Slider::new(&mut settings.texture_scale, 0.005..=0.6)
                    .text("Texture scale")
                    .fixed_decimals(3),
            );
            ui.add(
                egui::Slider::new(&mut settings.normal_strength, 0.0..=2.0)
                    .text("Normal strength")
                    .fixed_decimals(2),
            );
            ui.add(
                egui::Slider::new(&mut settings.variant_scale, 0.001..=0.05)
                    .text("Dirt/gravel variant scale")
                    .fixed_decimals(3),
            );

            separator_text(ui, "Hydraulic erosion");
            ui.add(egui::Slider::new(&mut erosion_settings.iterations, 20..=400).text("Iterations"));
            ui.add(egui::Slider::new(&mut erosion_settings.rain, 0.0..=0.08).text("Rain"));
            ui.add(
                egui::Slider::new(&mut erosion_settings.evaporation, 0.0..=0.4).text("Evaporation"),
            );
            ui.add(egui::Slider::new(&mut erosion_settings.erosion_rate, 0.01..=1.2).text("Erosion rate"));
            ui.add(
                egui::Slider::new(&mut erosion_settings.deposition_rate, 0.01..=1.2)
                    .text("Deposition"),
            );
            ui.add(
                egui::Slider::new(&mut erosion_settings.sediment_capacity, 0.5..=20.0)
                    .text("Capacity"),
            );
            ui.add(egui::Slider::new(&mut erosion_settings.transport_rate, 0.05..=2.0).text("Transport"));
            ui.add(
                egui::Slider::new(&mut erosion_settings.maximum_erosion, 1.0..=40.0)
                    .text("Max excavation")
                    .fixed_decimals(1)
                    .suffix(" m"),
            );
            ui.horizontal(|ui| {
                if ui.button("Regenerate erosion cache").clicked() {
                    rerun.0 = true;
                }
                ui.weak("(streams incrementally)");
            });
            ui.collapsing("Flow output", |ui| {
                ui.label("RGBA: water, velocity X, velocity Z, discharge");
                let (response, painter) =
                    ui.allocate_painter(egui::vec2(320.0, 320.0), egui::Sense::hover());
                painter.add(EguiBevyPaintCallback::new_paint_callback(
                    response.rect,
                    FlowAtlasPaintCallback,
                ));
            });

            separator_text(ui, "Controls");
            ui.label("WASD move  |  mouse look");
            ui.label("Space jump/up  |  Shift down");
            ui.label("V flight  |  Ctrl boost");
            ui.label("F1 UI  |  Escape release cursor");
            // The C++ printed the noise texture's own dimensions, which are
            // NOISE_RESOLUTION square by construction.
            ui.weak(format!(
                "FastNoiseLite source texture: {NOISE_RESOLUTION}x{NOISE_RESOLUTION}"
            ));
        });
    settings.show_ui = open;
}

/// `DrawErosionLattice`: the 512 m erosion-tile lattice drawn in screen space
/// so a visible cliff can be located relative to tile boundaries. Each line is
/// broken into short terrain-anchored segments: a single chord between two
/// endpoint heights would cut straight through hills, and a segment whose
/// endpoint passes behind the camera would project mirrored into the sky.
fn draw_erosion_lattice(
    painter: &egui::Painter,
    player: &Player,
    cache: &ErosionCache,
    noise: &NoiseField,
    viewport: (u32, u32),
    pixels_per_point: f32,
) {
    // The same matrices the render side builds for `GlobalUniforms`, so the
    // overlay lines up with the rasterized terrain.
    let camera = PlayerCamera::from_player(player);
    let aspect = viewport.0 as f32 / viewport.1 as f32;
    let view = matrices::view_matrix(camera.position, camera.target, camera.up);
    let projection = matrices::perspective(camera.fov_y, aspect, NEAR_PLANE, FAR_PLANE);

    let visibility_center = [player.position.x, player.position.z];
    let base_x = (player.position.x / EROSION_TILE_STRIDE).round() * EROSION_TILE_STRIDE;
    let base_z = (player.position.z / EROSION_TILE_STRIDE).round() * EROSION_TILE_STRIDE;
    let stroke = egui::Stroke::new(
        1.0 / pixels_per_point,
        egui::Color32::from_rgba_unmultiplied(255, 70, 70, 230),
    );
    let position = camera.position;
    // Horizontal forward, exactly as the C++ derived it.
    let forward = Vec3::new(camera.target.x - position.x, 0.0, camera.target.z - position.z)
        .normalize_or_zero();
    let point_in_front = |x: f32, z: f32| {
        (x - position.x) * forward.x + (z - position.z) * forward.z > 1.0
    };
    // The C++ guarded only with pointInFront; world_to_screen's own
    // behind-the-near-plane rejection is the same predicate for a perspective
    // matrix, so the two agree on every point that reaches the painter.
    let project = |x: f32, y: f32, z: f32| -> Option<egui::Pos2> {
        let screen = matrices::world_to_screen(Vec3::new(x, y, z), &view, &projection, viewport)?;
        Some(egui::pos2(screen.x, screen.y) / pixels_per_point)
    };
    let line_at = |x0: f32, z0: f32, x1: f32, z1: f32| {
        let dx = x1 - x0;
        let dz = z1 - z0;
        let length = (dx * dx + dz * dz).sqrt().max(1.0);
        let segments = ((length / LATTICE_SEGMENT_STEP) as i32).max(1);
        let mut previous_x = x0;
        let mut previous_z = z0;
        let mut previous_visible = point_in_front(previous_x, previous_z);
        for step in 1..=segments {
            let t = step as f32 / segments as f32;
            let x = x0 + dx * t;
            let z = z0 + dz * t;
            let visible = point_in_front(x, z);
            if previous_visible && visible {
                let h0 = erosion::sample_eroded_height(
                    cache, noise, previous_x, previous_z, visibility_center,
                ) + 1.5;
                let h1 =
                    erosion::sample_eroded_height(cache, noise, x, z, visibility_center) + 1.5;
                if let (Some(start), Some(end)) = (
                    project(previous_x, h0, previous_z),
                    project(x, h1, z),
                ) {
                    painter.line_segment([start, end], stroke);
                }
            }
            previous_x = x;
            previous_z = z;
            previous_visible = visible;
        }
    };
    for n in -2..=2 {
        let x = base_x + n as f32 * EROSION_TILE_STRIDE;
        line_at(x, base_z - LATTICE_EXTENT, x, base_z + LATTICE_EXTENT);
        let z = base_z + n as f32 * EROSION_TILE_STRIDE;
        line_at(base_x - LATTICE_EXTENT, z, base_x + LATTICE_EXTENT, z);
    }
}

/// `DrawCrosshair`: two 6 px arms through the screen centre at 0.85 alpha.
fn draw_crosshair(painter: &egui::Painter, viewport: (u32, u32), pixels_per_point: f32) {
    // Integer halves, as the C++'s `GetScreenWidth() / 2` computed them.
    let centre = egui::pos2(
        (viewport.0 / 2) as f32 / pixels_per_point,
        (viewport.1 / 2) as f32 / pixels_per_point,
    );
    let arm = 6.0 / pixels_per_point;
    let stroke = egui::Stroke::new(
        1.0 / pixels_per_point,
        egui::Color32::from_rgba_unmultiplied(255, 255, 255, 216),
    );
    painter.line_segment(
        [centre - egui::vec2(arm, 0.0), centre + egui::vec2(arm, 0.0)],
        stroke,
    );
    painter.line_segment(
        [centre - egui::vec2(0.0, arm), centre + egui::vec2(0.0, arm)],
        stroke,
    );
}

/// `ImGui::SeparatorText`: the label sits at the left with a rule filling the
/// rest of the row. egui's `Separator` carries no text, so both parts are
/// painted into one allocated row.
fn separator_text(ui: &mut egui::Ui, text: &str) {
    let spacing = ui.spacing().item_spacing.y;
    ui.add_space(spacing);
    let row_height = ui.text_style_height(&egui::TextStyle::Body);
    let (rect, _response) =
        ui.allocate_exact_size(egui::vec2(ui.available_width(), row_height), egui::Sense::hover());
    let text_colour = ui.visuals().text_color();
    let rule = ui.visuals().widgets.noninteractive.bg_stroke;
    let painter = ui.painter();
    let galley = painter.layout_no_wrap(
        text.to_owned(),
        egui::FontId::proportional(row_height),
        text_colour,
    );
    let rule_start = rect.left() + galley.size().x + 6.0;
    painter.galley(rect.min, galley, text_colour);
    if rule_start < rect.right() {
        painter.hline(rule_start..=rect.right(), rect.center().y, rule);
    }
    ui.add_space(spacing);
}

// ---------------------------------------------------------------------------
// Flow-atlas preview
//
// The C++ bound `erosion.flowAtlas` directly with `ImGui::Image(tex, {320,
// 320}, {0,1}, {1,0})`. bevy_egui's image path only takes bevy `Image` assets
// (it copies CPU pixel data into its own atlas) and the flow atlas exists
// solely on the GPU, so the port draws it with a raw wgpu pass issued from an
// egui paint callback; see assets/shaders/flow-preview.wgsl for the
// orientation and colour-space notes.
// ---------------------------------------------------------------------------

/// Render-world state the paint callback fills in on the first frame it runs
/// and reads on every draw. It lives in the render world (not behind a lock
/// in the callback) so `render` can borrow the pipeline straight out of the
/// world for the whole render pass.
#[derive(Resource, Default)]
pub struct FlowPreviewRenderState {
    /// The pipeline and the hdr flag it was built for.
    pipeline: Option<(EguiPipelineKey, RenderPipeline)>,
    bind_group: Option<BindGroup>,
}

/// Installs the flow-preview state above in the render world.
pub struct UiRenderPlugin;

impl Plugin for UiRenderPlugin {
    fn build(&self, app: &mut App) {
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.init_resource::<FlowPreviewRenderState>();
        }
    }
}

/// See the section comment above. The egui pass has already set the viewport
/// to the widget rect and clipped to it, so a fullscreen triangle covers
/// exactly the 320x320 image; it resets its own pipeline and viewport after
/// the callback returns.
struct FlowAtlasPaintCallback;

impl EguiBevyPaintCallbackImpl for FlowAtlasPaintCallback {
    fn update(
        &self,
        _info: egui::PaintCallbackInfo,
        _render_entity: RenderEntity,
        key: EguiPipelineKey,
        world: &mut World,
    ) {
        let built_for = world
            .resource::<FlowPreviewRenderState>()
            .pipeline
            .as_ref()
            .map(|(key, _)| *key);
        if built_for == Some(key) {
            return;
        }
        let Some((pipeline, bind_group)) = build_flow_preview_pipeline(world, key) else {
            return;
        };
        let mut state = world.resource_mut::<FlowPreviewRenderState>();
        state.pipeline = Some((key, pipeline));
        state.bind_group = Some(bind_group);
    }

    fn render<'pass>(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut TrackedRenderPass<'pass>,
        _render_entity: RenderEntity,
        _key: EguiPipelineKey,
        world: &'pass World,
    ) {
        let Some(state) = world.get_resource::<FlowPreviewRenderState>() else {
            return;
        };
        let (Some((_, pipeline)), Some(bind_group)) = (&state.pipeline, &state.bind_group) else {
            return;
        };
        render_pass.set_render_pipeline(pipeline);
        render_pass.set_bind_group(0, bind_group, &[]);
        render_pass.draw(0..3, 0..1);
    }
}

/// Builds the preview pipeline and its bind group (the flow atlas view plus a
/// linear, repeating sampler — raylib's default texture filter and wrap).
/// `layout: None` lets wgpu infer the bind group layout from the shader.
fn build_flow_preview_pipeline(
    world: &mut World,
    key: EguiPipelineKey,
) -> Option<(RenderPipeline, BindGroup)> {
    let device = world.resource::<RenderDevice>();
    let textures = world.resource::<GpuWorldTexturesOption>().0.as_ref()?;
    let shader = device
        .wgpu_device()
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("flow_preview_shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "../assets/shaders/flow-preview.wgsl"
            ))),
        });
    let target = if key.hdr {
        ViewTarget::TEXTURE_FORMAT_HDR
    } else {
        wgpu::TextureFormat::bevy_default()
    };
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("flow_preview_pipeline"),
        layout: None,
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: target,
                // ImGui's renderer blends with separate alpha factors
                // (GL_SRC_ALPHA/GL_ONE_MINUS_SRC_ALPHA for colour,
                // GL_ONE/GL_ONE_MINUS_SRC_ALPHA for alpha).
                blend: Some(wgpu::BlendState {
                    color: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::SrcAlpha,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    },
                    alpha: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    },
                }),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    });
    let layout = BindGroupLayout::from(pipeline.get_bind_group_layout(0));
    let sampler = device.wgpu_device().create_sampler(&wgpu::SamplerDescriptor {
        label: Some("flow_preview_sampler"),
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        address_mode_w: wgpu::AddressMode::Repeat,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..default()
    });
    let bind_group = device.create_bind_group(
        Some("flow_preview_bind_group"),
        &layout,
        &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&textures.flow_atlas_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
    );
    Some((pipeline, bind_group))
}
