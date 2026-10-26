//! Seekable ZIP paks with independently compressed Zstandard entries.
//! No archive is extracted, and parallel readers use separate file cursors.

#![expect(non_snake_case, reason = "Preserve the existing camelCase asset API")]

use std::collections::BTreeSet;
use std::fs::{self, File};
use std::io::{self, BufReader, Read};
use std::path::{Component, Path, PathBuf};
use std::sync::Mutex;
use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipArchive, ZipWriter};

pub const PACK_GROUPS: [&str; 2] = ["models", "textures"];
// Bound allocations even if an archive's metadata or compressed stream is damaged.
pub const MAX_ASSET_BYTES: u64 = 256 * 1024 * 1024;
const MAX_PACK_FILES: usize = 100_000;
const COMPRESSION_LEVEL: i64 = 9;

type PakReader = ZipArchive<BufReader<File>>;

pub struct PakArchive {
    path: PathBuf,
    files: BTreeSet<String>,
    readers: Mutex<Vec<PakReader>>,
}

impl PakArchive {
    /// Open a pak and validate its file names, compression, and size metadata.
    ///
    /// # Errors
    /// Returns an error for inaccessible archives, invalid groups, unsupported
    /// entries, duplicate names, or entries exceeding the size limit.
    pub fn open(path: &Path, group: &str) -> Result<Self, String> {
        validateGroup(group)?;
        let mut archive = openReader(path)?;
        if archive.is_empty() || archive.len() > MAX_PACK_FILES {
            return Err(format!("invalid file count in {}", path.display()));
        }
        if archive.has_duplicate_names() {
            return Err(format!("duplicate filenames in {}", path.display()));
        }
        let mut files = BTreeSet::new();
        for index in 0..archive.len() {
            let entry = archive
                .by_index_raw(index)
                .map_err(|error| error.to_string())?;
            let name = entry.name().map_err(|error| error.to_string())?;
            validateEntryName(&name, group)?;
            if !entry.is_file() || entry.encrypted() {
                return Err(format!(
                    "pak entry must be a regular unencrypted file: {name}"
                ));
            }
            if entry.compression() != CompressionMethod::Zstd {
                return Err(format!("pak entry is not Zstandard compressed: {name}"));
            }
            if entry.size() > MAX_ASSET_BYTES {
                return Err(format!("pak entry exceeds the asset size limit: {name}"));
            }
            files.insert(name.into_owned());
        }
        Ok(Self {
            path: path.to_path_buf(),
            files,
            readers: Mutex::new(vec![archive]),
        })
    }

    pub fn files(&self) -> &BTreeSet<String> {
        &self.files
    }

    /// Read and checksum one named asset with a bounded decompression buffer.
    ///
    /// # Errors
    /// Returns an error if the asset is missing, exceeds its declared size,
    /// has a bad checksum, or cannot be read or decompressed.
    pub fn read(&self, name: &str) -> Result<Vec<u8>, String> {
        if !self.files.contains(name) {
            return Err(format!("{}: missing asset {name}", self.path.display()));
        }
        let available_reader = self
            .readers
            .lock()
            .map_err(|error| error.to_string())?
            .pop();
        let mut archive = match available_reader {
            Some(reader) => reader,
            None => openReader(&self.path)?,
        };
        let result = readEntry(&mut archive, name);
        // Never hold the pool lock during I/O or decompression.
        self.readers
            .lock()
            .map_err(|error| error.to_string())?
            .push(archive);
        result.map_err(|error| format!("{}: {name}: {error}", self.path.display()))
    }
}

fn openReader(path: &Path) -> Result<PakReader, String> {
    let file = File::open(path).map_err(|error| format!("opening {}: {error}", path.display()))?;
    ZipArchive::new(BufReader::new(file))
        .map_err(|error| format!("reading pak {}: {error}", path.display()))
}

fn readEntry(archive: &mut PakReader, name: &str) -> Result<Vec<u8>, String> {
    let entry = archive.by_name(name).map_err(|error| error.to_string())?;
    let expected_size = entry.size();
    if expected_size > MAX_ASSET_BYTES {
        return Err("asset exceeds the size limit".into());
    }
    let mut bytes = Vec::new();
    let mut bounded_entry = entry.take(expected_size + 1);
    bounded_entry
        .read_to_end(&mut bytes)
        .map_err(|error| error.to_string())?;
    // Reading to EOF also runs the ZIP reader's CRC validation.
    if bytes.len() as u64 != expected_size {
        return Err("asset length differs from archive metadata".into());
    }
    Ok(bytes)
}

fn validateGroup(group: &str) -> Result<(), String> {
    if !PACK_GROUPS.contains(&group) {
        return Err(format!("unknown asset group: {group}"));
    }
    Ok(())
}

fn validateEntryName(name: &str, group: &str) -> Result<(), String> {
    let prefix = format!("{group}/");
    let invalid_component = name.split('/').any(|component| {
        component.is_empty() || component == "." || component == ".." || component.starts_with('.')
    });
    if !name.starts_with(&prefix) || invalid_component || name.contains(['\\', ':', '\0']) {
        return Err(format!("invalid pak asset path: {name:?}"));
    }
    Ok(())
}

/// List source files without following symlinks or including OS metadata.
///
/// # Errors
/// Returns an error for invalid groups, unsupported filesystem entries,
/// failed I/O, empty groups, or groups exceeding the file-count limit.
pub fn sourceFiles(source_root: &Path, group: &str) -> Result<Vec<PathBuf>, String> {
    validateGroup(group)?;
    let directory = source_root.join(group);
    let mut files = Vec::new();
    collectFiles(&directory, &mut files)?;
    files.sort();
    if files.is_empty() || files.len() > MAX_PACK_FILES {
        return Err(format!("invalid file count in {}", directory.display()));
    }
    Ok(files)
}

fn collectFiles(directory: &Path, files: &mut Vec<PathBuf>) -> Result<(), String> {
    let metadata = fs::symlink_metadata(directory).map_err(|error| error.to_string())?;
    if !metadata.is_dir() || metadata.is_symlink() {
        return Err(format!(
            "asset directory must be a real directory: {}",
            directory.display()
        ));
    }
    let entries =
        fs::read_dir(directory).map_err(|error| format!("{}: {error}", directory.display()))?;
    for entry in entries {
        let entry = entry.map_err(|error| error.to_string())?;
        let file_name = entry.file_name();
        let file_name = file_name.to_str().ok_or("asset paths must be UTF-8")?;
        if file_name.starts_with('.') {
            continue;
        }
        let path = entry.path();
        let file_type = entry.file_type().map_err(|error| error.to_string())?;
        if file_type.is_dir() {
            collectFiles(&path, files)?;
        } else if file_type.is_file() {
            files.push(path);
        } else {
            return Err(format!("asset must be a regular file: {}", path.display()));
        }
    }
    Ok(())
}

fn entryName(path: &Path, root: &Path, group: &str) -> Result<String, String> {
    let relative = path.strip_prefix(root).map_err(|error| error.to_string())?;
    let mut components = Vec::new();
    for component in relative.components() {
        let Component::Normal(value) = component else {
            return Err(format!("invalid source path: {}", path.display()));
        };
        components.push(value.to_str().ok_or("asset paths must be UTF-8")?);
    }
    let name = components.join("/");
    validateEntryName(&name, group)?;
    Ok(name)
}

pub struct PackStats {
    pub file_count: usize,
    pub source_bytes: u64,
    pub packed_bytes: u64,
}

/// Build and verify a reproducible pak before replacing the destination atomically.
///
/// # Errors
/// Returns an error if source validation, compression, verification, or filesystem
/// operations fail. Source files that change while packing are rejected.
pub fn writePak(source_root: &Path, output_path: &Path, group: &str) -> Result<PackStats, String> {
    let files = sourceFiles(source_root, group)?;
    let output_directory = output_path
        .parent()
        .ok_or("pak output needs a parent directory")?;
    fs::create_dir_all(output_directory).map_err(|error| error.to_string())?;
    let mut staging_file =
        tempfile::NamedTempFile::new_in(output_directory).map_err(|error| error.to_string())?;
    let mut options = SimpleFileOptions::default();
    options = options.compression_method(CompressionMethod::Zstd);
    options = options.compression_level(Some(COMPRESSION_LEVEL));
    options = options.last_modified_time(zip::DateTime::default());
    options = options.unix_permissions(0o644);
    let mut archive = ZipWriter::new(staging_file.as_file_mut());
    let mut source_bytes = 0;
    for path in &files {
        let name = entryName(path, source_root, group)?;
        let mut source = File::open(path).map_err(|error| error.to_string())?;
        let file_size = source.metadata().map_err(|error| error.to_string())?.len();
        if file_size > MAX_ASSET_BYTES {
            return Err(format!(
                "source asset exceeds the size limit: {}",
                path.display()
            ));
        }
        archive
            .start_file(&name, options)
            .map_err(|error| error.to_string())?;
        let copied = io::copy(&mut source, &mut archive).map_err(|error| error.to_string())?;
        if copied != file_size {
            return Err(format!(
                "source asset changed while packing: {}",
                path.display()
            ));
        }
        source_bytes += copied;
    }
    archive.finish().map_err(|error| error.to_string())?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let permissions = fs::Permissions::from_mode(0o644);
        staging_file
            .as_file()
            .set_permissions(permissions)
            .map_err(|error| error.to_string())?;
    }
    staging_file
        .as_file()
        .sync_all()
        .map_err(|error| error.to_string())?;
    // Validate every decompressed byte before atomically replacing an old pak.
    verifyPak(source_root, staging_file.path(), group)?;
    let packed_bytes = staging_file
        .as_file()
        .metadata()
        .map_err(|error| error.to_string())?
        .len();
    staging_file
        .persist(output_path)
        .map_err(|error| error.to_string())?;
    Ok(PackStats {
        file_count: files.len(),
        source_bytes,
        packed_bytes,
    })
}

/// Verify that a pak contains exactly the source group's files and bytes.
///
/// # Errors
/// Returns an error for invalid archives, inaccessible sources, missing or extra
/// entries, or byte differences between the archive and source files.
pub fn verifyPak(source_root: &Path, pak_path: &Path, group: &str) -> Result<PackStats, String> {
    let files = sourceFiles(source_root, group)?;
    let archive = PakArchive::open(pak_path, group)?;
    let mut expected_names = BTreeSet::new();
    let mut source_bytes = 0;
    for path in &files {
        let name = entryName(path, source_root, group)?;
        let bytes = archive.read(&name)?;
        let mut source = File::open(path).map_err(|error| error.to_string())?;
        let mut offset = 0;
        let mut buffer = [0u8; 64 * 1024];
        loop {
            let count = source
                .read(&mut buffer)
                .map_err(|error| error.to_string())?;
            if count == 0 {
                break;
            }
            let end = offset + count;
            if bytes.get(offset..end) != Some(&buffer[..count]) {
                return Err(format!("pak asset differs from source: {name}"));
            }
            offset = end;
        }
        if offset != bytes.len() {
            return Err(format!("pak asset differs from source: {name}"));
        }
        source_bytes += offset as u64;
        expected_names.insert(name);
    }
    if archive.files() != &expected_names {
        return Err(format!("{}: unexpected pak entries", pak_path.display()));
    }
    let packed_bytes = fs::metadata(pak_path)
        .map_err(|error| error.to_string())?
        .len();
    Ok(PackStats {
        file_count: files.len(),
        source_bytes,
        packed_bytes,
    })
}

#[cfg(test)]
#[path = "asset_pak_tests.rs"]
mod tests;
