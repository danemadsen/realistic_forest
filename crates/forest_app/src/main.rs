//! Executable entry point; domain code lives in the workspace crates.

mod app;

use forest_assets::{asset_io, resolveAssetRoot as resolve_asset_root};
use forest_buildings as buildings;
use forest_core::{constants, matrices, noise};
use forest_erosion as erosion;
use forest_render as render;
use forest_vegetation as vegetation;
use forest_vegetation::grass;
use forest_water as water;
use forest_water::rivers;
use forest_world::{OffscreenCapture, WorldOptions};
use forest_world::{automation, day_night, lightning, player, snow, thunder, weather};

fn main() {
    app::run();
}
