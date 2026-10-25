use super::*;
use std::io::Write;
use std::sync::Barrier;
use std::time::{Duration, SystemTime};

#[test]
fn roundtripPreservesNestedBinaryAndEmptyAssetsDeterministically() {
    let directory = tempfile::tempdir().unwrap();
    let source_root = directory.path().join("assets");
    let nested_directory = source_root.join("models/buildings");
    fs::create_dir_all(&nested_directory).unwrap();
    let binary_path = nested_directory.join("house.glb");
    let binary_bytes: Vec<u8> = (0..=255).collect();
    fs::write(&binary_path, &binary_bytes).unwrap();
    fs::write(source_root.join("models/empty.glb"), []).unwrap();
    fs::write(source_root.join("models/.DS_Store"), b"ignored metadata").unwrap();
    let first_path = directory.path().join("first.pak");
    let second_path = directory.path().join("second.pak");

    let first_stats = writePak(&source_root, &first_path, "models").unwrap();
    let archive = PakArchive::open(&first_path, "models").unwrap();
    let expected_names = BTreeSet::from([
        "models/buildings/house.glb".to_string(),
        "models/empty.glb".to_string(),
    ]);
    assert_eq!(archive.files(), &expected_names);
    assert_eq!(
        archive.read("models/buildings/house.glb").unwrap(),
        binary_bytes
    );
    assert!(archive.read("models/empty.glb").unwrap().is_empty());
    assert!(archive.read("models/missing.glb").is_err());
    assert_eq!(first_stats.file_count, 2);
    assert_eq!(first_stats.source_bytes, 256);
    assert_eq!(
        first_stats.packed_bytes,
        fs::metadata(&first_path).unwrap().len()
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = fs::metadata(&first_path).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o644);
    }

    // Filesystem timestamps never enter the archive, including on a rebuild.
    let source_file = fs::OpenOptions::new()
        .write(true)
        .open(&binary_path)
        .unwrap();
    let changed_time = SystemTime::UNIX_EPOCH + Duration::from_secs(500_000);
    source_file.set_modified(changed_time).unwrap();
    writePak(&source_root, &second_path, "models").unwrap();
    assert_eq!(
        fs::read(first_path).unwrap(),
        fs::read(second_path).unwrap()
    );
}

#[test]
fn verificationRejectsChangedMissingAndExtraAssets() {
    let directory = tempfile::tempdir().unwrap();
    let source_root = directory.path().join("assets");
    fs::create_dir_all(source_root.join("textures")).unwrap();
    let texture_path = source_root.join("textures/albedo.png");
    fs::write(&texture_path, b"original texture bytes").unwrap();
    let pak_path = directory.path().join("textures.pak");
    writePak(&source_root, &pak_path, "textures").unwrap();
    verifyPak(&source_root, &pak_path, "textures").unwrap();

    fs::write(&texture_path, b"different texture data").unwrap();
    let changed_error = verifyPak(&source_root, &pak_path, "textures")
        .err()
        .unwrap();
    assert!(
        changed_error.contains("differs from source"),
        "{changed_error}"
    );

    fs::write(&texture_path, b"original texture bytes").unwrap();
    let extra_path = source_root.join("textures/normal.png");
    fs::write(&extra_path, b"new source asset").unwrap();
    let missing_error = verifyPak(&source_root, &pak_path, "textures")
        .err()
        .unwrap();
    assert!(missing_error.contains("missing asset"), "{missing_error}");

    fs::write(&extra_path, b"second packed asset").unwrap();
    writePak(&source_root, &pak_path, "textures").unwrap();
    fs::remove_file(&extra_path).unwrap();
    let extra_error = verifyPak(&source_root, &pak_path, "textures")
        .err()
        .unwrap();
    assert!(
        extra_error.contains("unexpected pak entries"),
        "{extra_error}"
    );
}

#[test]
fn openingRejectsUnsafePathsAndUnsupportedEntries() {
    let directory = tempfile::tempdir().unwrap();
    let pak_path = directory.path().join("invalid.pak");
    let unsafe_names = [
        "models/../outside.glb",
        "models/./house.glb",
        "models//house.glb",
        "models/.hidden.glb",
        "models/C:/house.glb",
        "models/house\\door.glb",
        "textures/albedo.png",
        "/models/house.glb",
    ];
    for name in unsafe_names {
        writeSingleEntry(&pak_path, name, CompressionMethod::Zstd);
        let error = PakArchive::open(&pak_path, "models").err().unwrap();
        assert!(error.contains("invalid pak asset path"), "{name}: {error}");
    }

    writeSingleEntry(&pak_path, "models/house.glb", CompressionMethod::Stored);
    let compression_error = PakArchive::open(&pak_path, "models").err().unwrap();
    assert!(
        compression_error.contains("not Zstandard"),
        "{compression_error}"
    );

    let file = File::create(&pak_path).unwrap();
    let mut archive = ZipWriter::new(file);
    let options = SimpleFileOptions::default();
    archive
        .add_symlink("models/link.glb", "house.glb", options)
        .unwrap();
    archive.finish().unwrap();
    let symlink_error = PakArchive::open(&pak_path, "models").err().unwrap();
    assert!(
        symlink_error.contains("regular unencrypted file"),
        "{symlink_error}"
    );
}

#[test]
fn oversizedSourceCannotReplaceAnExistingPak() {
    let directory = tempfile::tempdir().unwrap();
    let source_root = directory.path().join("assets");
    fs::create_dir_all(source_root.join("models")).unwrap();
    let model_path = source_root.join("models/house.glb");
    fs::write(&model_path, b"valid original asset").unwrap();
    let pak_path = directory.path().join("models.pak");
    writePak(&source_root, &pak_path, "models").unwrap();
    let original_pak = fs::read(&pak_path).unwrap();

    let source_file = File::create(&model_path).unwrap();
    source_file.set_len(MAX_ASSET_BYTES + 1).unwrap();
    let error = writePak(&source_root, &pak_path, "models").err().unwrap();
    assert!(error.contains("size limit"), "{error}");
    assert_eq!(fs::read(&pak_path).unwrap(), original_pak);
}

#[test]
fn oversizedMetadataAndCorruptedChecksumsAreRejected() {
    let directory = tempfile::tempdir().unwrap();
    let pak_path = directory.path().join("models.pak");
    writeSingleEntry(&pak_path, "models/house.glb", CompressionMethod::Zstd);
    let original_bytes = fs::read(&pak_path).unwrap();
    let central_signature = [0x50, 0x4b, 0x01, 0x02];
    let central_offset = original_bytes
        .windows(4)
        .position(|window| window == central_signature)
        .unwrap();

    // The uncompressed-size field starts 24 bytes into a ZIP central header.
    let mut oversized_bytes = original_bytes.clone();
    let oversized_length = u32::try_from(MAX_ASSET_BYTES + 1).unwrap();
    let length_offset = central_offset + 24;
    oversized_bytes[length_offset..length_offset + 4]
        .copy_from_slice(&oversized_length.to_le_bytes());
    fs::write(&pak_path, oversized_bytes).unwrap();
    let size_error = PakArchive::open(&pak_path, "models").err().unwrap();
    assert!(size_error.contains("size limit"), "{size_error}");

    // Alter only the expected CRC, keeping the Zstandard payload decodable.
    let mut corrupt_bytes = original_bytes;
    corrupt_bytes[central_offset + 16] ^= 0xff;
    fs::write(&pak_path, corrupt_bytes).unwrap();
    let archive = PakArchive::open(&pak_path, "models").unwrap();
    assert!(archive.read("models/house.glb").is_err());
}

#[test]
fn parallelReadsKeepIndependentCursors() {
    let directory = tempfile::tempdir().unwrap();
    let source_root = directory.path().join("assets");
    fs::create_dir_all(source_root.join("models")).unwrap();
    let expected_assets: Vec<Vec<u8>> = (0u8..4).map(|value| vec![value; 128 * 1024]).collect();
    for (index, bytes) in expected_assets.iter().enumerate() {
        fs::write(source_root.join(format!("models/model-{index}.glb")), bytes).unwrap();
    }
    let pak_path = directory.path().join("models.pak");
    writePak(&source_root, &pak_path, "models").unwrap();
    let archive = PakArchive::open(&pak_path, "models").unwrap();
    let barrier = Barrier::new(expected_assets.len());

    std::thread::scope(|scope| {
        for (index, expected_bytes) in expected_assets.iter().enumerate() {
            let archive = &archive;
            let barrier = &barrier;
            scope.spawn(move || {
                barrier.wait();
                let name = format!("models/model-{index}.glb");
                for _ in 0..8 {
                    assert_eq!(&archive.read(&name).unwrap(), expected_bytes);
                }
            });
        }
    });
}

#[cfg(unix)]
#[test]
fn sourceDiscoveryRefusesSymlinkedFilesAndDirectories() {
    use std::os::unix::fs::symlink;

    let directory = tempfile::tempdir().unwrap();
    let source_root = directory.path().join("assets");
    let model_directory = source_root.join("models");
    fs::create_dir_all(&model_directory).unwrap();
    let target_path = directory.path().join("outside.glb");
    fs::write(&target_path, b"outside the asset root").unwrap();
    let link_path = model_directory.join("link.glb");
    symlink(&target_path, &link_path).unwrap();
    assert!(sourceFiles(&source_root, "models").is_err());

    fs::remove_file(&link_path).unwrap();
    symlink(directory.path(), &link_path).unwrap();
    assert!(sourceFiles(&source_root, "models").is_err());

    fs::remove_file(&link_path).unwrap();
    fs::remove_dir(&model_directory).unwrap();
    symlink(directory.path(), &model_directory).unwrap();
    assert!(sourceFiles(&source_root, "models").is_err());
}

fn writeSingleEntry(path: &Path, name: &str, compression: CompressionMethod) {
    let file = File::create(path).unwrap();
    let mut archive = ZipWriter::new(file);
    let options = SimpleFileOptions::default().compression_method(compression);
    archive.start_file(name, options).unwrap();
    archive.write_all(b"test asset payload").unwrap();
    archive.finish().unwrap();
}
