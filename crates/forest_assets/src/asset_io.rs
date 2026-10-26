//! Shared virtual asset paths for loose development files and release paks.

#![expect(non_snake_case, reason = "Preserve the existing camelCase asset API")]

use crate::asset_pak::{MAX_ASSET_BYTES, PACK_GROUPS, PakArchive};
use std::collections::HashMap;
use std::fs;
use std::io::Read;
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

enum AssetStorage {
    Loose,
    Packed([PakArchive; 2]),
}

pub struct AssetStore {
    root: PathBuf,
    storage: AssetStorage,
}

impl AssetStore {
    /// Open a loose asset directory or its complete pair of pak archives.
    ///
    /// # Errors
    /// Returns an error if the asset layout is incomplete or either pak is invalid.
    pub fn open(root: &Path) -> Result<Self, String> {
        let root = absolutePath(root)?;
        let storage = if hasPaks(&root) {
            // A partial or damaged package is an error. Never mix it with loose files.
            let models = PakArchive::open(&root.join("models.pak"), "models")?;
            let textures = PakArchive::open(&root.join("textures.pak"), "textures")?;
            AssetStorage::Packed([models, textures])
        } else {
            for group in PACK_GROUPS {
                if !root.join(group).is_dir() {
                    return Err(format!(
                        "missing asset directory: {}",
                        root.join(group).display()
                    ));
                }
            }
            AssetStorage::Loose
        };
        Ok(Self { root, storage })
    }

    fn assetPath(&self, path: &Path) -> Result<(PathBuf, String, usize), String> {
        let absolute = absolutePath(path)?;
        let relative = absolute.strip_prefix(&self.root).map_err(|_| {
            format!(
                "asset path escapes {}: {}",
                self.root.display(),
                path.display()
            )
        })?;
        let mut components = Vec::new();
        for component in relative.components() {
            let Component::Normal(value) = component else {
                return Err(format!("invalid asset path: {}", path.display()));
            };
            let value = value.to_str().ok_or("asset paths must be UTF-8")?;
            if value.contains(['\\', ':', '\0']) {
                return Err(format!("invalid asset path: {}", path.display()));
            }
            components.push(value);
        }
        let group = components.first().ok_or("asset path needs a group")?;
        let group_index = PACK_GROUPS
            .iter()
            .position(|known| known == group)
            .ok_or_else(|| format!("unknown asset group: {group}"))?;
        let name = components.join("/");
        Ok((absolute, name, group_index))
    }

    fn loosePath(&self, path: &Path) -> Result<PathBuf, String> {
        let resolved =
            fs::canonicalize(path).map_err(|error| format!("{}: {error}", path.display()))?;
        let resolved_root = fs::canonicalize(&self.root).map_err(|error| error.to_string())?;
        if !resolved.starts_with(&resolved_root) {
            return Err(format!(
                "asset symlink escapes {}: {}",
                self.root.display(),
                path.display()
            ));
        }
        Ok(resolved)
    }

    /// Read one asset without allowing paths or symlinks outside the store.
    ///
    /// # Errors
    /// Returns an error for invalid paths, missing files, oversized assets,
    /// failed I/O, or corrupt archive data.
    pub fn read(&self, path: &Path) -> Result<Vec<u8>, String> {
        let (absolute, name, group_index) = self.assetPath(path)?;
        match &self.storage {
            AssetStorage::Packed(archives) => archives[group_index].read(&name),
            AssetStorage::Loose => {
                let resolved = self.loosePath(&absolute)?;
                let file = fs::File::open(&resolved).map_err(|error| error.to_string())?;
                if file.metadata().map_err(|error| error.to_string())?.len() > MAX_ASSET_BYTES {
                    return Err(format!("asset exceeds size limit: {}", path.display()));
                }
                let mut bytes = Vec::new();
                file.take(MAX_ASSET_BYTES + 1)
                    .read_to_end(&mut bytes)
                    .map_err(|error| format!("{}: {error}", path.display()))?;
                if bytes.len() as u64 > MAX_ASSET_BYTES {
                    return Err(format!("asset exceeds size limit: {}", path.display()));
                }
                Ok(bytes)
            }
        }
    }

    /// Validate an asset path and return its absolute logical name.
    ///
    /// # Errors
    /// Returns an error if the asset is missing or its path escapes the store.
    pub fn canonicalize(&self, path: &Path) -> Result<PathBuf, String> {
        let (absolute, name, group_index) = self.assetPath(path)?;
        match &self.storage {
            AssetStorage::Packed(archives) => {
                if !archives[group_index].files().contains(&name) {
                    return Err(format!("missing packed asset: {name}"));
                }
            }
            AssetStorage::Loose => {
                self.loosePath(&absolute)?;
            }
        }
        Ok(absolute)
    }

    /// Return files immediately inside a logical directory in stable order.
    ///
    /// # Errors
    /// Returns an error for invalid or missing directories and failed I/O.
    pub fn readDirectory(&self, directory: &Path) -> Result<Vec<PathBuf>, String> {
        let (absolute, name, group_index) = self.assetPath(directory)?;
        let mut paths = Vec::new();
        match &self.storage {
            AssetStorage::Packed(archives) => {
                let prefix = format!("{name}/");
                let mut directory_exists = false;
                for entry in archives[group_index].files().range(prefix.clone()..) {
                    let Some(relative) = entry.strip_prefix(&prefix) else {
                        break;
                    };
                    directory_exists = true;
                    if !relative.contains('/') {
                        paths.push(absolute.join(relative));
                    }
                }
                if !directory_exists {
                    return Err(format!("missing packed directory: {name}"));
                }
            }
            AssetStorage::Loose => {
                let resolved = self.loosePath(&absolute)?;
                let entries = fs::read_dir(&resolved).map_err(|error| error.to_string())?;
                for entry in entries {
                    let entry = entry.map_err(|error| error.to_string())?;
                    let file_type = entry.file_type().map_err(|error| error.to_string())?;
                    if file_type.is_file() {
                        paths.push(absolute.join(entry.file_name()));
                    }
                }
                paths.sort();
            }
        }
        Ok(paths)
    }
}

fn hasPaks(root: &Path) -> bool {
    PACK_GROUPS
        .iter()
        .any(|group| fs::symlink_metadata(root.join(format!("{group}.pak"))).is_ok())
}

fn hasAssets(root: &Path) -> bool {
    hasPaks(root) || PACK_GROUPS.iter().all(|group| root.join(group).is_dir())
}

/// Prefer an installed package beside the executable over the working directory.
pub fn rootPath() -> &'static Path {
    static ROOT: OnceLock<PathBuf> = OnceLock::new();
    ROOT.get_or_init(findRoot)
}

fn findRoot() -> PathBuf {
    let executable_root = std::env::current_exe()
        .ok()
        .and_then(|executable| executable.parent().map(|parent| parent.join("assets")));
    let working_root = std::env::current_dir()
        .ok()
        .map(|directory| directory.join("assets"));
    // Cargo runs binaries with the application member's manifest directory.
    let cargo_root = std::env::var_os("CARGO_MANIFEST_DIR")
        .map(|directory| PathBuf::from(directory).join("../../assets"));
    let mut candidates: Vec<PathBuf> = [executable_root, working_root, cargo_root]
        .into_iter()
        .flatten()
        .collect();
    if cfg!(debug_assertions) {
        candidates.push(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../assets"));
    }
    chooseRoot(&candidates)
}

fn chooseRoot(candidates: &[PathBuf]) -> PathBuf {
    for candidate in candidates {
        if hasAssets(candidate) {
            return candidate.clone();
        }
    }
    candidates
        .first()
        .cloned()
        .unwrap_or_else(|| PathBuf::from("assets"))
}

fn absolutePath(path: &Path) -> Result<PathBuf, String> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|error| error.to_string())?
            .join(path)
    };
    let mut normalized = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
            other => normalized.push(other.as_os_str()),
        }
    }
    Ok(normalized)
}

fn storeFor(path: &Path) -> Result<Arc<AssetStore>, String> {
    static STORES: OnceLock<Mutex<HashMap<PathBuf, Arc<AssetStore>>>> = OnceLock::new();
    // Retain parent components until the root is identified, so a texture URI
    // escaping that root is rejected rather than resolved against another store.
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|error| error.to_string())?
            .join(path)
    };
    let root = if absolute.starts_with(rootPath()) {
        rootPath().to_path_buf()
    } else {
        let mut root = None;
        let ancestors: Vec<_> = absolute.ancestors().collect();
        for ancestor in ancestors.into_iter().rev() {
            let candidate = absolutePath(ancestor)?;
            if hasAssets(&candidate) {
                root = Some(candidate);
                break;
            }
        }
        root.ok_or_else(|| format!("no asset root for {}", path.display()))?
    };
    let stores = STORES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut stores = stores.lock().map_err(|error| error.to_string())?;
    if let Some(store) = stores.get(&root) {
        return Ok(Arc::clone(store));
    }
    let store = Arc::new(AssetStore::open(&root)?);
    stores.insert(root, Arc::clone(&store));
    Ok(store)
}

/// Read an asset from the store containing its logical path.
///
/// # Errors
/// Returns an error if no valid store is found or [`AssetStore::read`] fails.
pub fn read(path: &Path) -> Result<Vec<u8>, String> {
    storeFor(path)?.read(path)
}

/// List the files immediately inside a directory in its containing store.
///
/// # Errors
/// Returns an error if no valid store is found or [`AssetStore::readDirectory`] fails.
pub fn readDirectory(directory: &Path) -> Result<Vec<PathBuf>, String> {
    storeFor(directory)?.readDirectory(directory)
}

/// Validate an asset path within its containing store.
///
/// # Errors
/// Returns an error if no valid store is found or [`AssetStore::canonicalize`] fails.
pub fn canonicalize(path: &Path) -> Result<PathBuf, String> {
    storeFor(path)?.canonicalize(path)
}

/// Validate the selected asset layout before the application starts loading it.
///
/// # Errors
/// Returns an error if the selected root does not contain a complete valid store.
pub fn initialize() -> Result<(), String> {
    storeFor(&rootPath().join("models"))?;
    Ok(())
}

#[cfg(test)]
#[path = "asset_io_tests.rs"]
mod tests;
