//! Validates WGSL files with naga (bevy's actual shader front end), so shader
//! ports can be checked without launching the full renderer.
//!
//! Usage: `cargo run --bin wgsl-check [paths...]` — no paths validates every
//! `assets/shaders/*.wgsl`.

#[derive(Clone, Debug)]
struct EntryPoint {
    stage: naga::ShaderStage,
    name: String,
}

fn validate_file(path: &std::path::Path, entry_points: &[EntryPoint]) -> Result<(), String> {
    let source = std::fs::read_to_string(path).map_err(|e| format!("read {path:?}: {e}"))?;
    let module = naga::front::wgsl::parse_str(&source)
        .map_err(|error| format!("{path:?} parse: {}", error.emit_to_string(&source)))?;
    let mut validator =
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all());
    validator
        .validate(&module)
        .map_err(|errors| format!("{path:?} validate: {errors:?}"))?;
    for entry in entry_points {
        if module
            .entry_points
            .iter()
            .any(|point| point.stage == entry.stage && point.name == entry.name)
        {
            continue;
        }
        return Err(format!(
            "{path:?}: missing entry point {:?} ({:?})",
            entry.name, entry.stage
        ));
    }
    println!("{} OK", path.display());
    Ok(())
}

fn main() {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let paths: Vec<std::path::PathBuf> = if arguments.is_empty() {
        let mut glob: Vec<std::path::PathBuf> = std::fs::read_dir("assets/shaders")
            .expect("assets/shaders exists")
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry.path().extension().is_some_and(|extension| extension == "wgsl")
            })
            .map(|entry| entry.path())
            .collect();
        glob.sort();
        glob
    } else {
        arguments.iter().map(std::path::PathBuf::from).collect()
    };
    let mut failures = false;
    for path in paths {
        // Fullscreen passes (SSAO/blur/composite/fxaa) use `vs_main` +
        // `fs_main`; clipmap/geometry passes name their vs entry `vs_main` too.
        let entry_points = if path.file_name().is_some_and(|name| name == "precipitation.wgsl") {
            vec![
                EntryPoint { stage: naga::ShaderStage::Vertex, name: "vs_blit".into() },
                EntryPoint { stage: naga::ShaderStage::Fragment, name: "fs_blit".into() },
                EntryPoint { stage: naga::ShaderStage::Vertex, name: "vs_particle".into() },
                EntryPoint { stage: naga::ShaderStage::Fragment, name: "fs_particle".into() },
            ]
        } else {
            vec![EntryPoint { stage: naga::ShaderStage::Vertex, name: "vs_main".into() }]
        };
        match validate_file(&path, &entry_points) {
            Ok(()) => {}
            Err(error) => {
                eprintln!("{error}");
                failures = true;
            }
        }
    }
    if failures {
        std::process::exit(1);
    }
}
