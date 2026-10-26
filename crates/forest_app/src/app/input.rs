use super::*;

/// The pose P copies, on its way to the OS clipboard: handle_global_keys
/// writes it and copy_pose_to_clipboard, chained right after, copies it in the
/// same frame. The split lets the key handling be tested without an OS
/// clipboard, which headless CI runners do not have.
#[derive(Message)]
pub(super) struct CopyPose(String);

/// 2 toggles the trainer and pointer capture; 3 toggles passive debug text.
/// F12 saves a timestamped screenshot.
#[allow(clippy::too_many_arguments)]
pub(super) fn handle_global_keys(
    mut settings: ResMut<AppSettings>,
    mut player: Query<&mut Player>,
    mut cursor_options: Query<(&mut CursorOptions,), With<PrimaryWindow>>,
    keyboard: Res<ButtonInput<KeyCode>>,
    mut pose_copies: MessageWriter<CopyPose>,
    mut commands: Commands,
    automation: Option<Res<AutomationSettings>>,
    egui_wants_input: Option<Res<EguiWantsInput>>,
) {
    // Number keys must remain available when entering teleport coordinates.
    let editing_menu = settings.show_trainer
        && egui_wants_input
            .as_ref()
            .is_some_and(|input| input.wants_any_keyboard_input());
    let trainer_was_open = settings.show_trainer;
    if !editing_menu
        && (keyboard.just_pressed(KeyCode::Digit2) || keyboard.just_pressed(KeyCode::Numpad2))
    {
        settings.show_trainer = !settings.show_trainer;
    }
    if !editing_menu
        && (keyboard.just_pressed(KeyCode::Digit3) || keyboard.just_pressed(KeyCode::Numpad3))
    {
        settings.show_debug = !settings.show_debug;
    }
    if settings.show_trainer != trainer_was_open {
        // A pinned screenshot pose must keep ignoring gameplay input even if
        // a function key happens to land on the unattended capture window.
        let pinned_capture = automation
            .as_ref()
            .is_some_and(|automation| automation.has_camera && automation.shot_path.is_some());
        let capture_mouse = !settings.show_trainer && !pinned_capture;
        if let Ok(mut player) = player.single_mut() {
            if let Ok((mut cursor,)) = cursor_options.single_mut() {
                player.set_mouse_capture(capture_mouse, &mut cursor);
            } else {
                player.mouse_captured = capture_mouse;
                if capture_mouse {
                    player.mouse_warmup_frames = 3;
                }
            }
        }
    }
    // P copies the current pose to the clipboard for debugging.
    if !editing_menu && keyboard.just_pressed(KeyCode::KeyP) {
        if let Ok(player) = player.single() {
            let forward = player.forward();
            let pose = format!(
                "Position: {:.2}, {:.2}, {:.2}\nView: {:.3}, {:.3}, {:.3}",
                player.position.x,
                player.position.y,
                player.position.z,
                forward.x,
                forward.y,
                forward.z,
            );
            pose_copies.write(CopyPose(pose));
        }
    }
    if keyboard.just_pressed(KeyCode::F12) {
        commands
            .spawn(Screenshot::primary_window())
            .observe(on_manual_screenshot);
    }
}

/// Puts each pose P copied on the OS clipboard and logs the outcome.
pub(super) fn copy_pose_to_clipboard(
    mut copies: MessageReader<CopyPose>,
    mut clipboard: ResMut<Clipboard>,
) {
    for CopyPose(pose) in copies.read() {
        match clipboard.set_text(pose.as_str()) {
            Ok(()) => log::info!("POSE: copied to clipboard: {pose}"),
            Err(error) => log::error!("POSE: clipboard write failed: {error}"),
        }
    }
}

#[cfg(test)]
mod global_key_tests {
    use super::*;
    use bevy::window::CursorGrabMode;

    fn input_app() -> (App, Entity, Entity) {
        let mut app = App::new();
        app.insert_resource(AppSettings::default());
        app.insert_resource(ButtonInput::<KeyCode>::default());
        // main() registers this; the bare test app must too, as
        // handle_global_keys writes it.
        app.add_message::<CopyPose>();
        app.add_systems(Update, handle_global_keys);
        let player = app.world_mut().spawn(Player::default()).id();
        let cursor_bundle = (
            PrimaryWindow,
            CursorOptions {
                grab_mode: CursorGrabMode::Locked,
                visible: false,
                ..default()
            },
        );
        let cursor = app.world_mut().spawn(cursor_bundle).id();
        (app, player, cursor)
    }

    fn press(app: &mut App, key: KeyCode) {
        let mut keys = app.world_mut().resource_mut::<ButtonInput<KeyCode>>();
        keys.reset_all();
        keys.press(key);
        drop(keys);
        app.update();
    }

    #[test]
    fn digit2_releases_cursor_and_closing_recaptures_with_warmup() {
        let (mut app, player, cursor) = input_app();
        press(&mut app, KeyCode::Digit2);
        assert!(app.world().resource::<AppSettings>().show_trainer);
        assert!(!app.world().get::<Player>(player).unwrap().mouse_captured);
        let cursor_options = app.world().get::<CursorOptions>(cursor).unwrap();
        assert_eq!(cursor_options.grab_mode, CursorGrabMode::None);
        assert!(cursor_options.visible);

        app.world_mut()
            .get_mut::<Player>(player)
            .unwrap()
            .mouse_warmup_frames = 0;
        press(&mut app, KeyCode::Digit2);
        assert!(!app.world().resource::<AppSettings>().show_trainer);
        let player = app.world().get::<Player>(player).unwrap();
        assert!(player.mouse_captured);
        assert_eq!(player.mouse_warmup_frames, 3);
        let cursor_options = app.world().get::<CursorOptions>(cursor).unwrap();
        assert_eq!(cursor_options.grab_mode, CursorGrabMode::Locked);
        assert!(!cursor_options.visible);
    }

    #[test]
    fn digit3_toggles_only_debug_text_and_tab_no_longer_opens_trainer() {
        let (mut app, player, cursor) = input_app();
        press(&mut app, KeyCode::Digit3);
        let settings = app.world().resource::<AppSettings>();
        assert!(settings.show_debug);
        assert!(!settings.show_trainer);
        assert!(!settings.show_ui);
        assert!(app.world().get::<Player>(player).unwrap().mouse_captured);
        assert_eq!(
            app.world().get::<CursorOptions>(cursor).unwrap().grab_mode,
            CursorGrabMode::Locked,
        );
        press(&mut app, KeyCode::Digit3);
        assert!(!app.world().resource::<AppSettings>().show_debug);
        press(&mut app, KeyCode::Tab);
        assert!(!app.world().resource::<AppSettings>().show_trainer);
    }

    #[test]
    fn f1_leaves_trainer_and_advanced_controls_unchanged() {
        let (mut app, player, _) = input_app();
        press(&mut app, KeyCode::F1);
        let settings = app.world().resource::<AppSettings>();
        assert!(!settings.show_ui);
        assert!(!settings.show_trainer);
        assert!(app.world().get::<Player>(player).unwrap().mouse_captured);
        press(&mut app, KeyCode::Digit2);
        app.world_mut().resource_mut::<AppSettings>().show_ui = true;
        press(&mut app, KeyCode::F1);
        let settings = app.world().resource::<AppSettings>();
        assert!(settings.show_ui);
        assert!(settings.show_trainer);
        assert!(!app.world().get::<Player>(player).unwrap().mouse_captured);
    }

    #[test]
    fn numpad2_and_numpad3_toggle_the_same_controls() {
        let (mut app, _, _) = input_app();
        press(&mut app, KeyCode::Numpad2);
        assert!(app.world().resource::<AppSettings>().show_trainer);
        press(&mut app, KeyCode::Numpad3);
        assert!(app.world().resource::<AppSettings>().show_debug);
        press(&mut app, KeyCode::Numpad2);
        assert!(!app.world().resource::<AppSettings>().show_trainer);
        press(&mut app, KeyCode::Numpad3);
        assert!(!app.world().resource::<AppSettings>().show_debug);
    }

    #[test]
    fn typing_numbers_in_menu_fields_does_not_toggle_shortcuts() {
        use bevy_egui::{EguiContext, egui};
        let (mut app, player, _) = input_app();
        app.init_resource::<EguiWantsInput>()
            .add_systems(PreUpdate, bevy_egui::input::write_egui_wants_input_system);
        let mut context = EguiContext::default();
        let ctx = context.get_mut();
        ctx.begin_pass(egui::RawInput::default());
        let mut coordinate = String::new();
        let text_edit_field = |ui: &mut egui::Ui| {
            let response = ui.text_edit_singleline(&mut coordinate);
            response.request_focus();
            response.id
        };
        let field = egui::Area::new(egui::Id::new("coordinate_input_test"))
            .show(ctx, text_edit_field)
            .inner;
        let mut output = ctx.end_pass();
        output.textures_delta.clear();
        let context_entity = app.world_mut().spawn(context).id();
        app.world_mut().resource_mut::<AppSettings>().show_trainer = true;
        for key in [
            KeyCode::Digit2,
            KeyCode::Digit3,
            KeyCode::Numpad2,
            KeyCode::Numpad3,
        ] {
            press(&mut app, key);
            let settings = app.world().resource::<AppSettings>();
            assert!(settings.show_trainer);
            assert!(!settings.show_debug);
        }
        let mut stored_context = app
            .world_mut()
            .get_mut::<EguiContext>(context_entity)
            .unwrap();
        stored_context
            .get_mut()
            .memory_mut(|memory| memory.surrender_focus(field));
        press(&mut app, KeyCode::Digit2);
        assert!(!app.world().resource::<AppSettings>().show_trainer);
        assert!(app.world().get::<Player>(player).unwrap().mouse_captured);
        press(&mut app, KeyCode::Digit3);
        assert!(app.world().resource::<AppSettings>().show_debug);
    }

    #[test]
    fn p_copies_position_and_view_vector_to_the_clipboard() {
        let (mut app, player, _) = input_app();
        app.world_mut()
            .get_mut::<Player>(player)
            .unwrap()
            .teleport(Vec3::new(100.5, 20.0, -50.25));
        let mut pose_player = app.world_mut().get_mut::<Player>(player).unwrap();
        pose_player.yaw = std::f32::consts::FRAC_PI_2;
        pose_player.pitch = 0.0;
        press(&mut app, KeyCode::KeyP);
        let expected = "Position: 100.50, 20.00, -50.25\nView: 1.000, 0.000, 0.000";
        let copies = app.world().resource::<Messages<CopyPose>>();
        let copied: Vec<&str> = copies
            .iter_current_update_messages()
            .map(|CopyPose(pose)| pose.as_str())
            .collect();
        assert_eq!(copied, [expected]);
    }

    #[test]
    fn closing_trainer_does_not_capture_mouse_during_pinned_screenshot() {
        let (mut app, player, cursor) = input_app();
        let automation = AutomationSettings {
            has_camera: true,
            shot_path: Some("capture.png".into()),
            ..default()
        };
        app.insert_resource(automation);
        press(&mut app, KeyCode::Digit2);
        press(&mut app, KeyCode::Digit2);
        assert!(!app.world().get::<Player>(player).unwrap().mouse_captured);
        assert_eq!(
            app.world().get::<CursorOptions>(cursor).unwrap().grab_mode,
            CursorGrabMode::None,
        );
    }
}
