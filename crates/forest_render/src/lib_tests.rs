use super::GlobalUniformsGpu;

#[test]
fn snow_geometry_and_material_share_the_same_coverage() {
    let common = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../assets/shaders/snow_functions.wgslinc"
    ))
    .trim();
    for source in [
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../assets/shaders/terrain_vs.wgsl"
        )),
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../assets/shaders/terrain_fs.wgsl"
        )),
    ] {
        assert!(
            source.contains(common),
            "snow geometry and colour placement diverged"
        );
    }
}

/// The drawn pack must read the concavity the player's footing reads
/// (snow.rs sample_surface: the atlas G, faded) near the camera, handing
/// over to the material ring's 216/r^2 term far out, at the lifted
/// height and at all four neighbours that shape its normal, and hand the
/// same value to the fragment stage's white.
#[test]
fn the_drawn_pack_reads_the_footings_concavity() {
    let vertex = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../assets/shaders/terrain_vs.wgsl"
    ));
    for line in [
        "let erosionHere = fadedErosionSurface(worldXZ);",
        "let ringConcavity = 216.0*(0.25*ringSum - height)/(materialStep*materialStep);",
        "let farRelief = smoothHermite(12.0, 20.0, materialStep);",
        "let snowConcavity = mix(erosionHere.y, ringConcavity, farRelief);",
        "output.frag_concavity = vec2<f32>(snowConcavity, ringConcavity*farRelief);",
        "height += dry * snowCoverage(height, materialNormal, snowConcavity)",
    ] {
        assert!(vertex.contains(line), "terrain_vs.wgsl lost `{line}`");
    }
    assert_eq!(
        vertex.matches("materialNormal, snowConcavity);").count(),
        4,
        "all four lift neighbours must read the footing's concavity"
    );
    let fragment = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../assets/shaders/terrain_fs.wgsl"
    ));
    assert!(
        fragment
            .contains("let fSnow = snowCoverage(height, materialNormal, input.frag_concavity.x);")
    );
    assert!(fragment.contains("@location(10) frag_concavity: vec2<f32>,"));
    assert!(vertex.contains("@location(10) frag_concavity: vec2<f32>,"));
}

#[test]
fn snow_detail_preserves_the_clipmap_horizon_and_anchor() {
    use super::TerrainStageUniforms;
    use forest_core::constants::*;
    let stage = |level| {
        TerrainStageUniforms::build(
            bevy::math::Mat4::IDENTITY.to_cols_array(),
            level,
            [0.0; 3],
            (0, 0),
            [0.0; 2],
        )
    };
    assert_eq!(stage(0).spacing, 0.25);
    assert_eq!(stage(1).spacing, 0.5);
    assert_eq!(stage(CLIP_LEVELS as u32 - 1).spacing, 64.0);
    assert_eq!(CLIP_ANCHOR_SPACING, 64.0);
    assert_eq!(
        stage(CLIP_LEVELS as u32 - 1).spacing * CLIP_CELLS as f32 / 2.0,
        7168.0
    );
}

#[test]
fn water_and_visible_sky_share_the_same_atmosphere() {
    let common = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../assets/shaders/atmosphere_functions.wgslinc"
    ))
    .trim();
    for source in [
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../assets/shaders/composite.wgsl"
        )),
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../assets/shaders/water.wgsl"
        )),
    ] {
        assert!(
            source.contains(common),
            "visible and reflected sky helpers diverged"
        );
    }
}

#[test]
fn visible_clouds_reflections_and_shadows_use_one_density_model() {
    let common = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../assets/shaders/cloud_functions.wgslinc"
    ))
    .trim();
    for source in [
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../assets/shaders/composite.wgsl"
        )),
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../assets/shaders/cloud_probe.wgsl"
        )),
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../assets/shaders/water.wgsl"
        )),
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../assets/shaders/cloud_shadow_map.wgsl"
        )),
    ] {
        assert!(
            source.contains(common),
            "cloud shape or lighting differs between passes"
        );
    }
}

/// Every draw that fades the erosion centres the fade on the player's
/// eight-metre lattice point, as the CPU's footing does: a stage built
/// round the raw player position would shift the 1.1-1.6 km band against
/// the collision and make the lighting map's key stale.
#[test]
fn every_terrain_stage_fades_round_the_anchor() {
    for (name, source) in [
        ("terrain_node.rs", include_str!("terrain_node.rs")),
        ("vegetation_node.rs", include_str!("vegetation_node.rs")),
    ] {
        let code = source.split("#[cfg(test)]").next().unwrap();
        let mut calls = 0;
        for (at, _) in code.match_indices("TerrainStageUniforms::build(") {
            calls += 1;
            let window = &code[at.saturating_sub(600)..(at + 600).min(code.len())];
            assert!(
                window.contains("erosion_fade_anchor("),
                "{name}: a terrain stage near byte {at} is built without the fade anchor"
            );
        }
        assert!(
            calls > 0,
            "{name} builds no terrain stage any more; update this test"
        );
    }
}

#[test]
fn erosion_and_materials_share_one_geology() {
    let common = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../assets/shaders/geology_functions.wgslinc"
    ))
    .trim();
    for source in [
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../assets/shaders/erosion_init.wgsl"
        )),
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../assets/shaders/erosion_terrain.wgsl"
        )),
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../assets/shaders/terrain_fs.wgsl"
        )),
    ] {
        assert!(
            source.contains(common),
            "erosion and material geology diverged"
        );
    }
}

/// The erosion stage uniforms are written from Rust structs and read
/// by the WGSL; a member out of place reads its neighbour's value. Check
/// every named member's offset against the struct the worker writes, in
/// both directions, and that each struct fits the 128-byte stage buffer.
#[test]
fn erosion_stage_uniforms_match_wgsl_layout() {
    use super::{
        ErosionFluxStageUniforms, ErosionInitStageUniforms, ErosionTerrainStageUniforms,
        ErosionThermalStageUniforms, ErosionWaterStageUniforms,
    };
    use std::mem::{offset_of, size_of};
    // A shader, the size of the Rust struct it is fed, and that struct's
    // named fields with their offsets, in declaration order.
    type Layout = (&'static str, usize, Vec<(&'static str, usize)>);
    let layouts: [Layout; 5] = [
        (
            "erosion_init",
            size_of::<ErosionInitStageUniforms>(),
            vec![
                ("mode", offset_of!(ErosionInitStageUniforms, mode)),
                ("world_min", offset_of!(ErosionInitStageUniforms, world_min)),
                ("cell_size", offset_of!(ErosionInitStageUniforms, cell_size)),
            ],
        ),
        (
            "erosion_flux",
            size_of::<ErosionFluxStageUniforms>(),
            vec![
                (
                    "resolution",
                    offset_of!(ErosionFluxStageUniforms, resolution),
                ),
                (
                    "delta_time",
                    offset_of!(ErosionFluxStageUniforms, delta_time),
                ),
                ("cell_size", offset_of!(ErosionFluxStageUniforms, cell_size)),
                ("gravity", offset_of!(ErosionFluxStageUniforms, gravity)),
            ],
        ),
        (
            "erosion_water",
            size_of::<ErosionWaterStageUniforms>(),
            vec![
                (
                    "resolution",
                    offset_of!(ErosionWaterStageUniforms, resolution),
                ),
                (
                    "delta_time",
                    offset_of!(ErosionWaterStageUniforms, delta_time),
                ),
                (
                    "cell_size",
                    offset_of!(ErosionWaterStageUniforms, cell_size),
                ),
                ("rain", offset_of!(ErosionWaterStageUniforms, rain)),
                (
                    "evaporation",
                    offset_of!(ErosionWaterStageUniforms, evaporation),
                ),
                (
                    "sea_level",
                    offset_of!(ErosionWaterStageUniforms, sea_level),
                ),
            ],
        ),
        (
            "erosion_terrain",
            size_of::<ErosionTerrainStageUniforms>(),
            vec![
                (
                    "resolution",
                    offset_of!(ErosionTerrainStageUniforms, resolution),
                ),
                (
                    "delta_time",
                    offset_of!(ErosionTerrainStageUniforms, delta_time),
                ),
                (
                    "cell_size",
                    offset_of!(ErosionTerrainStageUniforms, cell_size),
                ),
                (
                    "sea_level",
                    offset_of!(ErosionTerrainStageUniforms, sea_level),
                ),
                (
                    "erosion_rate",
                    offset_of!(ErosionTerrainStageUniforms, erosion_rate),
                ),
                (
                    "deposition_rate",
                    offset_of!(ErosionTerrainStageUniforms, deposition_rate),
                ),
                (
                    "sediment_capacity",
                    offset_of!(ErosionTerrainStageUniforms, sediment_capacity),
                ),
                (
                    "minimum_slope",
                    offset_of!(ErosionTerrainStageUniforms, minimum_slope),
                ),
                (
                    "maximum_erosion",
                    offset_of!(ErosionTerrainStageUniforms, maximum_erosion),
                ),
                (
                    "transport_rate",
                    offset_of!(ErosionTerrainStageUniforms, transport_rate),
                ),
                (
                    "guard_band_pixels",
                    offset_of!(ErosionTerrainStageUniforms, guard_band_pixels),
                ),
                (
                    "brush_strength",
                    offset_of!(ErosionTerrainStageUniforms, brush_strength),
                ),
                (
                    "world_min",
                    offset_of!(ErosionTerrainStageUniforms, world_min),
                ),
                (
                    "fluvial_capacity",
                    offset_of!(ErosionTerrainStageUniforms, fluvial_capacity),
                ),
                (
                    "fluvial_erosion",
                    offset_of!(ErosionTerrainStageUniforms, fluvial_erosion),
                ),
                (
                    "fluvial_deposition",
                    offset_of!(ErosionTerrainStageUniforms, fluvial_deposition),
                ),
                (
                    "maximum_incision",
                    offset_of!(ErosionTerrainStageUniforms, maximum_incision),
                ),
                (
                    "drainage_saturation",
                    offset_of!(ErosionTerrainStageUniforms, drainage_saturation),
                ),
            ],
        ),
        (
            "erosion_thermal",
            size_of::<ErosionThermalStageUniforms>(),
            vec![
                (
                    "resolution",
                    offset_of!(ErosionThermalStageUniforms, resolution),
                ),
                (
                    "cell_size",
                    offset_of!(ErosionThermalStageUniforms, cell_size),
                ),
                (
                    "sea_level",
                    offset_of!(ErosionThermalStageUniforms, sea_level),
                ),
                (
                    "world_min",
                    offset_of!(ErosionThermalStageUniforms, world_min),
                ),
                (
                    "loose_rate",
                    offset_of!(ErosionThermalStageUniforms, loose_rate),
                ),
                (
                    "rock_rate",
                    offset_of!(ErosionThermalStageUniforms, rock_rate),
                ),
                (
                    "loose_repose",
                    offset_of!(ErosionThermalStageUniforms, loose_repose),
                ),
                (
                    "soft_rock_slope",
                    offset_of!(ErosionThermalStageUniforms, soft_rock_slope),
                ),
                (
                    "hard_rock_slope",
                    offset_of!(ErosionThermalStageUniforms, hard_rock_slope),
                ),
                (
                    "weathering_rate",
                    offset_of!(ErosionThermalStageUniforms, weathering_rate),
                ),
                (
                    "weathering_cover",
                    offset_of!(ErosionThermalStageUniforms, weathering_cover),
                ),
                (
                    "weathering_cover_steep",
                    offset_of!(ErosionThermalStageUniforms, weathering_cover_steep),
                ),
                (
                    "weathering_slope",
                    offset_of!(ErosionThermalStageUniforms, weathering_slope),
                ),
            ],
        ),
    ];
    let directory =
        std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../..")).join("assets/shaders");
    for (name, rust_size, rust) in layouts {
        let path = directory.join(format!("{name}.wgsl"));
        let source = std::fs::read_to_string(&path).unwrap();
        let module = naga::front::wgsl::parse_str(&source).unwrap_or_else(|error| {
            panic!("{}: {}", path.display(), error.emit_to_string(&source))
        });
        let stage = module
            .types
            .iter()
            .find(|(_, ty)| ty.name.as_deref() == Some("StageUniforms"));
        let Some((_, stage)) = stage else {
            panic!("{name} declares no StageUniforms")
        };
        let naga::TypeInner::Struct { members, span } = &stage.inner else {
            panic!("{name}: StageUniforms must be a struct");
        };
        // Padding members carry a leading underscore on either side.
        let wgsl: Vec<(&str, usize)> = members
            .iter()
            .filter_map(|member| {
                member
                    .name
                    .as_deref()
                    .map(|field| (field, member.offset as usize))
            })
            .filter(|(field, _)| !field.starts_with('_'))
            .collect();
        assert_eq!(
            wgsl, rust,
            "{name}: the WGSL members and the Rust fields disagree"
        );
        assert!(
            rust_size >= *span as usize,
            "{name}: Rust writes {rust_size} B of a {span} B struct"
        );
        assert!(
            rust_size <= 128,
            "{name}: {rust_size} B overflows the 128 B stage buffer"
        );
    }
}

#[test]
fn rain_snow_and_water_share_one_spatial_storm_field() {
    let common = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../assets/shaders/precipitation_functions.wgslinc"
    ))
    .trim();
    for source in [
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../assets/shaders/composite.wgsl"
        )),
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../assets/shaders/water.wgsl"
        )),
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../assets/shaders/precipitation.wgsl"
        )),
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../assets/shaders/lightning.wgsl"
        )),
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../assets/shaders/cloud_probe.wgsl"
        )),
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../assets/shaders/cloud_shadow_map.wgsl"
        )),
    ] {
        assert!(
            source.contains(common),
            "precipitation field differs between passes"
        );
    }
}

/// A stale layout in even an unrelated pass can reinterpret daylight as
/// a matrix or bind too small a uniform range. Check the actual WGSL ABI
/// against Rust, including offsets, rather than just matching byte sizes.
#[test]
fn every_shader_uses_the_same_global_uniform_layout() {
    let expected = [
        ("view", std::mem::offset_of!(GlobalUniformsGpu, view)),
        (
            "projection",
            std::mem::offset_of!(GlobalUniformsGpu, projection),
        ),
        (
            "camera_position",
            std::mem::offset_of!(GlobalUniformsGpu, camera_position),
        ),
        (
            "sun_direction",
            std::mem::offset_of!(GlobalUniformsGpu, sun_direction),
        ),
        (
            "viewport",
            std::mem::offset_of!(GlobalUniformsGpu, viewport),
        ),
        ("params", std::mem::offset_of!(GlobalUniformsGpu, params)),
        (
            "settings_a",
            std::mem::offset_of!(GlobalUniformsGpu, settings_a),
        ),
        (
            "settings_b",
            std::mem::offset_of!(GlobalUniformsGpu, settings_b),
        ),
        (
            "sun_colour",
            std::mem::offset_of!(GlobalUniformsGpu, sun_colour),
        ),
        (
            "moon_direction",
            std::mem::offset_of!(GlobalUniformsGpu, moon_direction),
        ),
        (
            "atmosphere",
            std::mem::offset_of!(GlobalUniformsGpu, atmosphere),
        ),
        (
            "raymarch",
            std::mem::offset_of!(GlobalUniformsGpu, raymarch),
        ),
        (
            "heightfield",
            std::mem::offset_of!(GlobalUniformsGpu, heightfield),
        ),
        ("clouds", std::mem::offset_of!(GlobalUniformsGpu, clouds)),
        (
            "cloud_layer",
            std::mem::offset_of!(GlobalUniformsGpu, cloud_layer),
        ),
        (
            "cloud_motion",
            std::mem::offset_of!(GlobalUniformsGpu, cloud_motion),
        ),
        ("weather", std::mem::offset_of!(GlobalUniformsGpu, weather)),
        ("storm", std::mem::offset_of!(GlobalUniformsGpu, storm)),
        (
            "lightning",
            std::mem::offset_of!(GlobalUniformsGpu, lightning),
        ),
        (
            "lightning_meta",
            std::mem::offset_of!(GlobalUniformsGpu, lightning_meta),
        ),
    ];
    let directory =
        std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../..")).join("assets/shaders");
    let mut checked = 0;
    for entry in std::fs::read_dir(directory).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().and_then(|v| v.to_str()) != Some("wgsl") {
            continue;
        }
        let source = std::fs::read_to_string(&path).unwrap();
        let module = naga::front::wgsl::parse_str(&source).unwrap_or_else(|error| {
            panic!("{}: {}", path.display(), error.emit_to_string(&source))
        });
        for (_, ty) in module.types.iter() {
            if ty.name.as_deref() != Some("GlobalUniforms") {
                continue;
            }
            let naga::TypeInner::Struct { members, span } = &ty.inner else {
                panic!("GlobalUniforms must be a struct");
            };
            assert_eq!(
                *span as usize,
                std::mem::size_of::<GlobalUniformsGpu>(),
                "{}",
                path.display()
            );
            assert_eq!(members.len(), expected.len(), "{}", path.display());
            for (member, (name, offset)) in members.iter().zip(expected) {
                assert_eq!(member.name.as_deref(), Some(name), "{}", path.display());
                assert_eq!(member.offset as usize, offset, "{}: {name}", path.display());
            }
            checked += 1;
        }
    }
    assert!(checked >= 10, "expected the complete render pipeline");
}
