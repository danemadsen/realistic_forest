//! Build or verify the compressed model and texture archives distributed with Forest.

#![allow(non_snake_case)]

use asset_pak::{PACK_GROUPS, PackStats};
use realistic_forest::asset_pak;
use std::ffi::OsString;
use std::path::{Component, Path, PathBuf};
use std::process::ExitCode;

const USAGE: &str = "Usage: pack_assets [--verify] SOURCE_ROOT OUTPUT_ROOT\n\
    Pack models and textures, or verify existing packs against their source files.\n\
    Example: pack_assets assets target/release/assets";
const BYTES_PER_MIB: f64 = 1024.0 * 1024.0;

struct Options {
    source_root: PathBuf,
    output_root: PathBuf,
    verify: bool,
}

fn main() -> ExitCode {
    let args: Vec<OsString> = std::env::args_os().skip(1).collect();
    if args.len() == 1 && (args[0] == "--help" || args[0] == "-h") {
        println!("{USAGE}");
        return ExitCode::SUCCESS;
    }

    let result = parseArgs(&args).and_then(run);
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("pack_assets: {error}");
            ExitCode::FAILURE
        }
    }
}

fn parseArgs(args: &[OsString]) -> Result<Options, String> {
    let verify = args.first().is_some_and(|arg| arg == "--verify");
    let positional_args = if verify { &args[1..] } else { args };
    if positional_args.len() != 2 {
        return Err(USAGE.to_string());
    }
    if positional_args
        .iter()
        .any(|arg| arg.to_string_lossy().starts_with('-'))
    {
        return Err(format!("Unexpected option.\n{USAGE}"));
    }
    if positional_args.iter().any(|arg| arg.is_empty()) {
        return Err(format!("Asset directory paths must not be empty.\n{USAGE}"));
    }

    Ok(Options {
        source_root: PathBuf::from(&positional_args[0]),
        output_root: PathBuf::from(&positional_args[1]),
        verify,
    })
}

/// Resolve existing path components, including symlinks, before creating any
/// missing output directories. This prevents a destination alias from placing
/// generated archives inside the source tree being packed.
fn resolveOutputRoot(path: &Path) -> Result<PathBuf, String> {
    let absolute_path = std::path::absolute(path)
        .map_err(|error| format!("Resolving output directory {}: {error}", path.display()))?;
    let mut resolved = PathBuf::new();
    for component in absolute_path.components() {
        match component {
            Component::Prefix(_) | Component::RootDir => resolved.push(component.as_os_str()),
            Component::CurDir => {}
            Component::ParentDir => {
                resolved.pop();
            }
            Component::Normal(name) => {
                resolved.push(name);
                match std::fs::symlink_metadata(&resolved) {
                    Ok(_) => {
                        resolved = std::fs::canonicalize(&resolved).map_err(|error| {
                            format!("Resolving output directory {}: {error}", resolved.display())
                        })?;
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                    Err(error) => {
                        return Err(format!(
                            "Resolving output directory {}: {error}",
                            resolved.display()
                        ));
                    }
                }
            }
        }
    }
    Ok(resolved)
}

fn validateOutputRoot(source_root: &Path, output_root: &Path) -> Result<(), String> {
    if output_root == source_root {
        return Err("Source and output directories must be different".to_string());
    }
    for group in PACK_GROUPS {
        let source_group = source_root.join(group);
        if output_root.starts_with(&source_group) {
            return Err(format!(
                "Output directory {} must be outside source {}",
                output_root.display(),
                source_group.display()
            ));
        }
    }
    Ok(())
}

fn reportStats(action: &str, group: &str, stats: &PackStats) {
    let source_mib = stats.source_bytes as f64 / BYTES_PER_MIB;
    let packed_mib = stats.packed_bytes as f64 / BYTES_PER_MIB;
    let size_summary = if stats.source_bytes == 0 {
        "empty source files".to_string()
    } else {
        let percentage = stats.packed_bytes as f64 / stats.source_bytes as f64 * 100.0;
        format!("{percentage:.1}% of source")
    };
    println!(
        "{action} {group}.pak: {} files, {source_mib:.2} MiB -> {packed_mib:.2} MiB ({size_summary})",
        stats.file_count
    );
}

fn run(options: Options) -> Result<(), String> {
    let source_root = std::fs::canonicalize(&options.source_root).map_err(|error| {
        format!(
            "Reading source directory {}: {error}",
            options.source_root.display()
        )
    })?;
    if !source_root.is_dir() {
        return Err(format!(
            "Source {} is not a directory",
            source_root.display()
        ));
    }

    let output_root = resolveOutputRoot(&options.output_root)?;
    validateOutputRoot(&source_root, &output_root)?;
    if !options.verify {
        std::fs::create_dir_all(&output_root).map_err(|error| {
            format!(
                "Creating output directory {}: {error}",
                output_root.display()
            )
        })?;
    }

    for group in PACK_GROUPS {
        let pak_path = output_root.join(format!("{group}.pak"));
        let stats = if options.verify {
            asset_pak::verifyPak(&source_root, &pak_path, group)?
        } else {
            asset_pak::writePak(&source_root, &pak_path, group)?
        };
        let action = if options.verify { "Verified" } else { "Packed" };
        reportStats(action, group, &stats);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parsesPackingAndVerification() {
        let source_root = OsString::from("source assets");
        let output_root = OsString::from("output assets");
        let packing_args = [source_root.clone(), output_root.clone()];
        let packing = parseArgs(&packing_args).expect("packing arguments parse");
        assert!(!packing.verify);
        assert_eq!(packing.source_root, Path::new("source assets"));
        assert_eq!(packing.output_root, Path::new("output assets"));

        let verify_args = [OsString::from("--verify"), source_root, output_root];
        let verification = parseArgs(&verify_args).expect("verification arguments parse");
        assert!(verification.verify);
    }

    #[test]
    fn rejectsMissingExtraAndUnexpectedArguments() {
        let no_args = [];
        assert!(parseArgs(&no_args).is_err());
        let incomplete_args = [OsString::from("--verify"), OsString::from("assets")];
        assert!(parseArgs(&incomplete_args).is_err());
        let extra_args = [
            OsString::from("assets"),
            OsString::from("output"),
            OsString::from("extra"),
        ];
        assert!(parseArgs(&extra_args).is_err());
        let invalid_args = [OsString::from("--invalid"), OsString::from("output")];
        assert!(parseArgs(&invalid_args).is_err());
    }

    #[test]
    fn rejectsOutputInsideEitherSourceGroupBeforeCreatingDirectories() {
        let directory = tempfile::tempdir().expect("temporary directory exists");
        let source_root = std::fs::canonicalize(directory.path()).unwrap();
        for group in PACK_GROUPS {
            let output_path = source_root.join(group).join("generated/nested");
            let output_root = resolveOutputRoot(&output_path).unwrap();
            assert!(validateOutputRoot(&source_root, &output_root).is_err());
            assert!(!output_path.exists());
        }
        assert!(validateOutputRoot(&source_root, &source_root).is_err());
        assert!(validateOutputRoot(&source_root, &source_root.join("packs")).is_ok());
    }

    #[cfg(unix)]
    #[test]
    fn rejectsOutputSymlinksIntoSourceGroups() {
        let directory = tempfile::tempdir().expect("temporary directory exists");
        let source_path = directory.path().join("source");
        let source_models = source_path.join("models");
        std::fs::create_dir_all(&source_models).unwrap();
        let source_root = std::fs::canonicalize(source_path).unwrap();
        let alias = directory.path().join("output");
        std::os::unix::fs::symlink(&source_models, &alias).unwrap();
        let output_root = resolveOutputRoot(&alias.join("nested")).unwrap();
        assert!(validateOutputRoot(&source_root, &output_root).is_err());
    }
}
