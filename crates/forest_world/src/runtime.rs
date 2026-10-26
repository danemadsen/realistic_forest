//! Runtime options shared by application setup and render extraction.

use bevy::prelude::{Handle, Image, Resource};

/// Whether the ocean plane is drawn.
#[derive(Resource)]
pub struct WorldOptions {
    pub draw_ocean: bool,
}

/// The image and pixel dimensions used for a capture without a window.
#[derive(Resource, Clone)]
pub struct OffscreenCapture {
    pub image: Option<Handle<Image>>,
    pub size: [u32; 2],
}
