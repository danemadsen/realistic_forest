use super::*;

/// The camera and the player. This is the only camera entity in the app, so
/// bevy_egui's auto-creation attaches the primary egui context to it. An
/// offscreen capture's camera renders into its image instead of the window.
pub(super) fn spawn_scene(
    mut commands: Commands,
    offscreen: Option<ResMut<OffscreenCapture>>,
    mut images: ResMut<Assets<Image>>,
) {
    commands.spawn(Player::default());
    let camera_bundle = (
        Transform::default(),
        Projection::Perspective(PerspectiveProjection {
            fov: 68.0_f32.to_radians(),
            ..default()
        }),
        Camera {
            // The composite pass fully covers the view every frame, matching
            // the C++ render pass into the deferred G-buffer.
            clear_color: ClearColorConfig::None,
            ..default()
        },
        CameraRenderGraph::new(render::ForestRender),
        Frustum::default(),
        VisibleEntities::default(),
        Msaa::Off,
    );
    let camera = commands.spawn(camera_bundle).id();
    if let Some(mut offscreen) = offscreen {
        // The swapchain's own format, so every pass builds the pipelines it
        // would for the window, and the PNG writer reads it as it does a
        // window capture.
        let format = bevy::render::render_resource::TextureFormat::Bgra8UnormSrgb;
        let mut target =
            Image::new_target_texture(offscreen.size[0], offscreen.size[1], format, None);
        target.texture_descriptor.usage |= bevy::render::render_resource::TextureUsages::COPY_SRC;
        let handle = images.add(target);
        commands
            .entity(camera)
            .insert(bevy::camera::RenderTarget::Image(handle.clone().into()));
        offscreen.image = Some(handle);
    }
}

/// The CPU half of `CreateErosionBlendMask`: the per-texel mask that erosion
/// tile borders share, sampled on the CPU for player collision and streamed
/// to the GPU once in the render world.
pub(super) fn setup_blend_mask(mut cache: ResMut<ErosionCache>) {
    cache.blend_mask_samples = erosion::erosion_blend_mask_samples();
}

/// Applies the automation startup bits the C++ main does before the loop:
/// the ground-level spawn height, the pinned pose for `--camera`, and the
/// cursor state for unattended runs (DisableCursor in the normal case).
pub(super) fn setup_cursor_and_player(
    automation: Res<AutomationSettings>,
    mut local: Local<bool>,
    cache: Res<ErosionCache>,
    noise: Res<NoiseField>,
    mut player: Query<&mut Player>,
    mut windows: Query<(&mut bevy::window::CursorOptions,), With<PrimaryWindow>>,
    mut snow: ResMut<snow::SnowState>,
    offscreen: Option<Res<OffscreenCapture>>,
) {
    if *local {
        return;
    }
    // An offscreen capture has no window, and no cursor to set.
    if offscreen.is_some() {
        let Ok(mut player) = player.single_mut() else {
            return;
        };
        *local = true;
        let spawn = if automation.has_camera {
            automation.position
        } else {
            [0.0, 0.0, 0.0]
        };
        snow.recenter(bevy::math::Vec2::new(spawn[0], spawn[2]));
        if automation.has_camera {
            player.position = bevy::math::Vec3::from(automation.position);
            player.yaw = automation.yaw;
            player.pitch = automation.pitch;
            player.flying = true;
        } else {
            let fade_anchor = erosion::erosion_fade_anchor([0.0, 0.0]);
            player.position.y =
                erosion::sample_eroded_height(&cache, &noise, 0.0, 0.0, fade_anchor) + EYE_HEIGHT;
        }
        player.mouse_captured = false;
        log::info!(
            "SHOT: pose pinned at {:.1},{:.1},{:.1} yaw {:.1} pitch {:.1} (offscreen)",
            player.position.x,
            player.position.y,
            player.position.z,
            player.yaw.to_degrees(),
            player.pitch.to_degrees(),
        );
        return;
    }
    // The one-shot latch is set only once the queries actually resolve. Both
    // the window entity (WindowPlugin) and the player entity (spawn_scene)
    // are created in Startup, so an early run can legitimately find neither;
    // latching before the `?`s would then burn the one attempt — and the
    // pinned --camera pose, the whole point of an unattended capture, would
    // silently never be applied.
    let Ok((mut cursor,)) = windows.single_mut() else {
        log::warn!("AUTOMATION: no primary window yet; retrying the startup pose");
        return;
    };
    let Ok(mut player) = player.single_mut() else {
        log::warn!("AUTOMATION: no player entity yet; retrying the startup pose");
        return;
    };
    *local = true;
    let snow_center = if automation.has_camera {
        bevy::math::Vec2::new(automation.position[0], automation.position[2])
    } else {
        bevy::math::Vec2::ZERO
    };
    snow.recenter(snow_center);
    if let Some([x0, z0, x1, z1]) = automation.snow_trail {
        snow.stamp_segment(bevy::math::Vec2::new(x0, z0), bevy::math::Vec2::new(x1, z1));
    }
    // Ground-level spawn, matching the C++ pre-loop player placement.
    if !automation.has_camera {
        player.position.y = erosion::sample_eroded_height(
            &cache,
            &noise,
            0.0,
            0.0,
            erosion::erosion_fade_anchor([0.0, 0.0]),
        ) + EYE_HEIGHT;
        // `if (automation.shotPath == nullptr) DisableCursor();` (main.cpp:2261)
        // is the whole of the C++'s cursor special-casing for automation, and
        // it touches the cursor only. mouseCaptured is left at its default
        // true (main.cpp:247), which is what keeps DrawCrosshair
        // (main.cpp:2414, `mouseCaptured && !showUi`, and showUi starts false
        // at main.cpp:256) drawing its two arms through the centre of a
        // --shot frame taken without --camera. Clearing mouseCaptured here
        // suppressed that crosshair, so those captures were missing it.
        // --camera is the case that clears it, and that happens below.
        if automation.shot_path.is_none() {
            cursor.grab_mode = bevy::window::CursorGrabMode::Locked;
            cursor.visible = false;
        }
        return;
    }
    // Fly so gravity never fights a pinned pose; leave the cursor alone so
    // the capture window does not fight mouse input.
    player.position = bevy::math::Vec3::from(automation.position);
    player.yaw = automation.yaw;
    player.pitch = automation.pitch;
    player.flying = true;
    player.mouse_captured = false;
    log::info!(
        "SHOT: pose pinned at {:.1},{:.1},{:.1} yaw {:.1} pitch {:.1}",
        player.position.x,
        player.position.y,
        player.position.z,
        player.yaw.to_degrees(),
        player.pitch.to_degrees(),
    );
}
