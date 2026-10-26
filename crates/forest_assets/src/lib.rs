//! Asset discovery and seekable compressed archives.

pub mod asset_io;
pub mod asset_pak;

/// Resolve the common asset root for Bevy and domain asset loaders.
#[expect(non_snake_case, reason = "Preserve the existing camelCase asset API")]
pub fn resolveAssetRoot() -> String {
    asset_io::rootPath().to_string_lossy().into_owned()
}
