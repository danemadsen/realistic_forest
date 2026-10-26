/// `--size` reproduces raylib's `InitWindow(w, h)`: those are *screen points*,
/// and raylib then renders at `screen x GetWindowScaleDPI()`. But raylib does
/// not hand the request straight to GLFW — it clamps it to the *primary*
/// monitor's work area first, in software, before the window exists:
/// `glfwGetMonitorWorkarea(glfwGetPrimaryMonitor(), ...)` followed by
/// `if (CORE.Window.screen.width > workWidth) CORE.Window.screen.width = workWidth;`
/// (rcore_desktop_glfw.c:1618, 1671-1679; raylib's own comment there explains
/// why — a GLFW window larger than the work area would not show up). On macOS
/// that work area is `[NSScreen visibleFrame]` (cocoa_monitor.m:487-499), the
/// space the menu bar and Dock leave over — and raylib applies it whichever
/// display the window actually opens on. Measured with the window on the
/// 1920x1080 external display, the reference still logged "Screen size:
/// 1470 x 923" for `--size 3000,2000`: the built-in panel's visibleFrame
/// exactly, and nothing the 1920x1080 display would have produced.
///
/// The port had no equivalent step, and that is the whole of the `--size`
/// divergence. winit exposes no work-area query — its macOS `MonitorHandle`
/// offers only `size()`, the full video mode — so nothing constrains the
/// request, and on a 1x display AppKit has no reason to either. Measured at
/// 1x: `--size 1600,900` wrote 1600x900 from the port against 1470x900 from
/// the reference. Adopting the backend's content size afterwards cannot close
/// that gap, because at 1x the backend genuinely is the unclamped 1600x900
/// (probed), so there is nothing to adopt.
///
/// Units are AppKit points, which is both what raylib clamps and what
/// `WindowResolution::new` is handed below.
#[cfg(target_os = "macos")]
pub(super) fn clamp_to_primary_work_area(width: u32, height: u32) -> (u32, u32) {
    use objc2::MainThreadMarker;
    use objc2_app_kit::NSScreen;

    // NSScreen is main-thread-only and `main` is the main thread; the marker
    // is how objc2 proves it.
    let Some(mtm) = MainThreadMarker::new() else {
        eprintln!("WINDOW: not on the main thread, leaving the screen size unclamped");
        return (width, height);
    };
    let Some(primary) = NSScreen::screens(mtm).firstObject() else {
        eprintln!("WINDOW: no screens reported, leaving the screen size unclamped");
        return (width, height);
    };
    let work = primary.visibleFrame();
    let work_width = work.size.width as u32;
    let work_height = work.size.height as u32;
    // raylib floors a zero dimension at one pixel rather than clamping to zero
    // (rcore_desktop_glfw.c:1665-1667), so a screen reporting no work area
    // should not collapse the window either.
    if work_width == 0 || work_height == 0 {
        return (width, height);
    }
    let clamped_width = width.min(work_width);
    let clamped_height = height.min(work_height);
    // `println!` rather than `log::info!`: bevy installs its logger along with
    // DefaultPlugins, which this runs well before, so anything logged here is
    // silently dropped. The reference prints the same figure from InitWindow —
    // it is its "Screen size:" line — so this keeps the two capture logs
    // directly comparable.
    if clamped_width != width || clamped_height != height {
        println!(
            "WINDOW: screen size {clamped_width} x {clamped_height} (requested {width} x {height}, clamped to the primary monitor work area)"
        );
    } else {
        println!("WINDOW: screen size {clamped_width} x {clamped_height}");
    }
    (clamped_width, clamped_height)
}

/// Everywhere else the request is used as-is: winit has no work-area query to
/// make, and the port carries no per-platform one.
#[cfg(not(target_os = "macos"))]
pub(super) fn clamp_to_primary_work_area(width: u32, height: u32) -> (u32, u32) {
    (width, height)
}
