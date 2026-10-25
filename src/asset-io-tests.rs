use super::*;
use crate::asset_pak::writePak;

const FIXTURE_FILES: [(&str, &[u8]); 7] = [
    ("models/zebra.glb", b"glTF\0\xff\x80zebra"),
    ("models/grass.glb", b"glTF\0\xff\x80grass"),
    ("models/buildings/house.glb", b"glTF\0house"),
    ("models/buildings/annex/shed.glb", b"glTF\0shed"),
    ("textures/grass/albedo.png", b"\x89PNG\0\xffgrass"),
    ("textures/buildings/wall.png", b"\x89PNG\0\xffwall"),
    ("textures/empty.bin", b""),
];

fn writeFixture(root: &Path) {
    for (name, bytes) in FIXTURE_FILES {
        let path = root.join(name);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, bytes).unwrap();
    }
}

fn packFixture(source_root: &Path, packed_root: &Path) {
    for group in PACK_GROUPS {
        let output = packed_root.join(format!("{group}.pak"));
        writePak(source_root, &output, group).unwrap();
    }
}

#[test]
fn looseAndPackedStoresReadTheSameBytesAndResolveTextureReferences() {
    let workspace = tempfile::tempdir().unwrap();
    let source_root = workspace.path().join("source");
    let packed_root = workspace.path().join("packed");
    writeFixture(&source_root);
    packFixture(&source_root, &packed_root);

    assert!(!packed_root.join("models").exists());
    assert!(!packed_root.join("textures").exists());
    for root in [&source_root, &packed_root] {
        let store = AssetStore::open(root).unwrap();
        for (name, expected) in FIXTURE_FILES {
            let path = root.join(name);
            assert_eq!(store.read(&path).unwrap(), expected, "{name}");
            assert_eq!(store.canonicalize(&path).unwrap(), path);
        }

        let grass_texture = root.join("models/../textures/grass/albedo.png");
        let resolved_grass = root.join("textures/grass/albedo.png");
        assert_eq!(store.canonicalize(&grass_texture).unwrap(), resolved_grass);
        assert_eq!(store.read(&grass_texture).unwrap(), b"\x89PNG\0\xffgrass");

        let building_texture = root.join("models/buildings/../../textures/buildings/wall.png");
        let resolved_building = root.join("textures/buildings/wall.png");
        assert_eq!(
            store.canonicalize(&building_texture).unwrap(),
            resolved_building
        );
        assert_eq!(store.read(&building_texture).unwrap(), b"\x89PNG\0\xffwall");

        let missing_texture = root.join("textures/missing.png");
        assert!(store.read(&missing_texture).is_err());
        assert!(store.canonicalize(&missing_texture).is_err());
    }
}

#[test]
fn directoryListingIsSortedAndIncludesOnlyImmediateFiles() {
    let workspace = tempfile::tempdir().unwrap();
    let source_root = workspace.path().join("source");
    let packed_root = workspace.path().join("packed");
    writeFixture(&source_root);
    packFixture(&source_root, &packed_root);

    for root in [&source_root, &packed_root] {
        let store = AssetStore::open(root).unwrap();
        let expected_models = vec![root.join("models/grass.glb"), root.join("models/zebra.glb")];
        let models = store.readDirectory(&root.join("models")).unwrap();
        assert_eq!(models, expected_models);

        let expected_buildings = vec![root.join("models/buildings/house.glb")];
        let buildings = store.readDirectory(&root.join("models/buildings")).unwrap();
        assert_eq!(buildings, expected_buildings);

        let expected_nested = vec![root.join("models/buildings/annex/shed.glb")];
        let nested = store
            .readDirectory(&root.join("models/buildings/annex"))
            .unwrap();
        assert_eq!(nested, expected_nested);
        assert!(store.readDirectory(&root.join("models/missing")).is_err());
        assert!(store.readDirectory(&root.join("models/grass.glb")).is_err());
    }
}

#[test]
fn partialAndCorruptPackagesCannotFallBackToValidLooseAssets() {
    let workspace = tempfile::tempdir().unwrap();
    let source_root = workspace.path().join("source");
    writeFixture(&source_root);

    let models_pak = source_root.join("models.pak");
    writePak(&source_root, &models_pak, "models").unwrap();
    assert!(AssetStore::open(&source_root).is_err());

    let textures_pak = source_root.join("textures.pak");
    writePak(&source_root, &textures_pak, "textures").unwrap();
    fs::write(&models_pak, b"damaged package").unwrap();
    assert!(AssetStore::open(&source_root).is_err());

    writePak(&source_root, &models_pak, "models").unwrap();
    fs::write(&textures_pak, b"damaged package").unwrap();
    assert!(AssetStore::open(&source_root).is_err());
}

#[test]
fn packedStoreDoesNotReadLooseFilesMissingFromItsArchives() {
    let workspace = tempfile::tempdir().unwrap();
    let root = workspace.path().join("assets");
    writeFixture(&root);
    packFixture(&root, &root);

    let loose_only = root.join("textures/loose-only.png");
    fs::write(&loose_only, b"unpacked texture").unwrap();
    let store = AssetStore::open(&root).unwrap();
    assert!(store.read(&loose_only).is_err());
    assert!(store.canonicalize(&loose_only).is_err());

    let textures = store.readDirectory(&root.join("textures")).unwrap();
    assert!(!textures.contains(&loose_only));
}

#[test]
fn readsCanonicalizationAndListingRejectPathsOutsideTheirRoot() {
    let workspace = tempfile::tempdir().unwrap();
    let source_root = workspace.path().join("source");
    let packed_root = workspace.path().join("packed");
    let outside_root = workspace.path().join("outside");
    writeFixture(&source_root);
    writeFixture(&outside_root);
    packFixture(&source_root, &packed_root);

    for root in [&source_root, &packed_root] {
        let store = AssetStore::open(root).unwrap();
        let escaped_file = root.join("models/../../outside/textures/grass/albedo.png");
        let escaped_directory = root.join("models/../../outside/textures");
        assert!(store.read(&escaped_file).is_err());
        assert!(store.canonicalize(&escaped_file).is_err());
        assert!(store.readDirectory(&escaped_directory).is_err());
        assert!(
            store
                .read(&outside_root.join("textures/grass/albedo.png"))
                .is_err()
        );
        assert!(store.read(&root.join("shaders/terrain-vs.wgsl")).is_err());
    }
}

#[test]
fn sharedHelpersKeepTheOriginalRootWhenAPathTriesToEnterAnotherStore() {
    let workspace = tempfile::tempdir().unwrap();
    let original_root = workspace.path().join("original");
    let other_root = workspace.path().join("other");
    writeFixture(&original_root);
    writeFixture(&other_root);

    let escaped_file = original_root.join("models/../../other/textures/grass/albedo.png");
    let escaped_directory = original_root.join("models/../../other/textures");
    assert!(read(&escaped_file).is_err());
    assert!(canonicalize(&escaped_file).is_err());
    assert!(readDirectory(&escaped_directory).is_err());

    let valid_texture = original_root.join("models/../textures/grass/albedo.png");
    assert_eq!(read(&valid_texture).unwrap(), b"\x89PNG\0\xffgrass");
}

#[test]
fn rootSelectionPrefersExecutableAssetsIncludingIncompletePackages() {
    let workspace = tempfile::tempdir().unwrap();
    let executable_root = workspace.path().join("installed/assets");
    let working_root = workspace.path().join("working/assets");
    let cargo_root = workspace.path().join("checkout/assets");
    writeFixture(&working_root);
    writeFixture(&cargo_root);
    let candidates = vec![executable_root.clone(), working_root.clone(), cargo_root];

    assert_eq!(chooseRoot(&candidates), working_root);

    fs::create_dir_all(&executable_root).unwrap();
    fs::write(executable_root.join("models.pak"), b"incomplete package").unwrap();
    assert_eq!(chooseRoot(&candidates), executable_root);
    assert!(AssetStore::open(&executable_root).is_err());

    packFixture(&working_root, &executable_root);
    assert_eq!(chooseRoot(&candidates), executable_root);
    assert!(AssetStore::open(&executable_root).is_ok());
}

#[cfg(unix)]
#[test]
fn looseReadsAndListingsRejectSymlinksEscapingTheAssetRoot() {
    use std::os::unix::fs::symlink;

    let workspace = tempfile::tempdir().unwrap();
    let root = workspace.path().join("assets");
    let outside_root = workspace.path().join("outside");
    writeFixture(&root);
    writeFixture(&outside_root);
    let outside_texture = outside_root.join("textures/grass/albedo.png");
    let linked_file = root.join("textures/escaped.png");
    let linked_directory = root.join("textures/escaped");
    symlink(outside_texture, &linked_file).unwrap();
    symlink(outside_root.join("textures"), &linked_directory).unwrap();

    let store = AssetStore::open(&root).unwrap();
    assert!(store.read(&linked_file).is_err());
    assert!(store.canonicalize(&linked_file).is_err());
    assert!(store.readDirectory(&linked_directory).is_err());
    assert!(
        store
            .read(&linked_directory.join("grass/albedo.png"))
            .is_err()
    );
}

#[cfg(unix)]
#[test]
fn danglingPakSymlinkPreventsLooseFallback() {
    use std::os::unix::fs::symlink;

    let workspace = tempfile::tempdir().unwrap();
    let root = workspace.path().join("assets");
    writeFixture(&root);
    symlink(root.join("missing.pak"), root.join("models.pak")).unwrap();

    assert!(hasPaks(&root));
    assert!(AssetStore::open(&root).is_err());
}
