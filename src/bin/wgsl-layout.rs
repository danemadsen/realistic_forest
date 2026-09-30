//! Dev tool: parses a WGSL file with naga and prints every resource binding's
//! min_binding_size, plus struct member offsets/sizes under the uniform /
//! storage rules. Used to reconcile the WGSL StageUniforms with the Rust
//! bytemuck structs.

fn main() {
    let paths: Vec<String> = std::env::args().skip(1).collect();
    for path in paths {
        let Ok(source) = std::fs::read_to_string(&path) else {
            println!("{path}: unreadable");
            continue;
        };
        let module = match naga::front::wgsl::Frontend::new().parse(&source) {
            Ok(module) => module,
            Err(error) => {
                println!("{path}: parse error: {error}");
                continue;
            }
        };
        let mut layouter = naga::proc::Layouter::default();
        let gctx = naga::proc::GlobalCtx {
            types: &module.types,
            constants: &module.constants,
            overrides: &module.overrides,
            global_expressions: &module.global_expressions,
        };
        if let Err(error) = layouter.update(gctx) {
            println!("{path}: layout error: {error}");
            continue;
        }
        println!("=== {path}");
        for (type_handle, ty) in module.types.iter() {
            if let naga::TypeInner::Struct { ref members, .. } = ty.inner {
                let name = ty.name.clone().unwrap_or_else(|| String::from("<anon>"));
                // Uniform address space raises the struct's own alignment to
                // 16, rounding the total size up to a 16 multiple.
                let base = layouter[type_handle].size;
                println!("  struct {name:?} size={base} uniform_size={}", base.next_multiple_of(16));
                for member in members.iter() {
                    let member_size = layouter[member.ty].size;
                    println!(
                        "    {:?}@{} size={}",
                        member.name.clone().unwrap_or_default(),
                        member.offset,
                        member_size
                    );
                }
            }
        }
    }
}