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

fn validate_file(path: &std::path::Path) -> Result<(), String> {
    let source = std::fs::read_to_string(path).map_err(|e| format!("read {path:?}: {e}"))?;
    let module = naga::front::wgsl::parse_str(&source)
        .map_err(|error| format!("{path:?} parse: {}", error.emit_to_string(&source)))?;
    let mut validator =
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all());
    validator
        .validate(&module)
        .map_err(|errors| format!("{path:?} validate: {errors:?}"))?;
    if module.entry_points.is_empty() {
        return Err(format!(
            "{path:?}: declares no entry point; shared helper text belongs in a .wgslinc"
        ));
    }
    let entry_points = required_entry_points(&module);
    // naga only type-checks the entry points a module actually declares, so an
    // entry point that the pipelines expect but the file no longer defines is
    // exactly the mistake this tool exists to catch.
    for entry in &entry_points {
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

/// The entry points a file must define, derived from the stages it declares.
///
/// A module that ships a vertex stage has to call it `vs_main` — that is the
/// name the pipelines look up, so a rename is exactly the break this tool
/// exists to catch. A file with no vertex stage at all is not asked for one:
/// `tree-fs.wgsl` is a fragment-only asset paired with `tree-vs.wgsl`, and the
/// dummy `vs_main` stub it would otherwise need is dead code that only ever
/// confuses the next reader. Extra entry points beyond these are fine —
/// `terrain-vs.wgsl` also carries the erosion heightfield capture's
/// `fs_heightfield` in the same module.
fn required_entry_points(module: &naga::Module) -> Vec<EntryPoint> {
    let mut required = Vec::new();
    if module.entry_points.iter().any(|point| point.stage == naga::ShaderStage::Vertex) {
        required.push(EntryPoint { stage: naga::ShaderStage::Vertex, name: "vs_main".into() });
    }
    required
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
        match validate_file(&path) {
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