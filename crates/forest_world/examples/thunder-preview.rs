//! Render deterministic procedural thunder previews without opening the application.

use forest_world::lightning;
use forest_world::thunder::{self, THUNDER_SAMPLE_RATE};
use std::env;
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use thunder_synth::Thunder;

fn write_wav(path: &Path, samples: &[f32]) -> io::Result<()> {
    let data_bytes = samples.len().checked_mul(2);
    let Some(data_bytes) = data_bytes.and_then(|length| u32::try_from(length).ok()) else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "audio is too long for a WAV file",
        ));
    };
    let riff_bytes = data_bytes.checked_add(36).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "audio is too long for a WAV file",
        )
    })?;
    let file = File::create(path)?;
    let mut output = BufWriter::new(file);
    output.write_all(b"RIFF")?;
    output.write_all(&riff_bytes.to_le_bytes())?;
    output.write_all(b"WAVEfmt ")?;
    output.write_all(&16_u32.to_le_bytes())?;
    output.write_all(&1_u16.to_le_bytes())?;
    output.write_all(&1_u16.to_le_bytes())?;
    output.write_all(&THUNDER_SAMPLE_RATE.to_le_bytes())?;
    output.write_all(&(THUNDER_SAMPLE_RATE * 2).to_le_bytes())?;
    output.write_all(&2_u16.to_le_bytes())?;
    output.write_all(&16_u16.to_le_bytes())?;
    output.write_all(b"data")?;
    output.write_all(&data_bytes.to_le_bytes())?;

    for sample in samples {
        let limited = sample.clamp(-1.0, 1.0);
        let encoded = (limited * f32::from(i16::MAX)).round() as i16;
        output.write_all(&encoded.to_le_bytes())?;
    }
    output.flush()
}

fn main() -> io::Result<()> {
    let output_directory = env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("tmp/thunder-preview"));
    fs::create_dir_all(&output_directory)?;

    let references = [
        ("reference-clap.wav", Thunder::Clap),
        ("reference-roll.wav", Thunder::Roll),
    ];
    for (name, kind) in references {
        let path = output_directory.join(name);
        let wav = thunder_synth::wav(kind, 7);
        fs::write(&path, wav)?;
        println!(
            "{} ({:.2} seconds, unchanged upstream generator)",
            path.display(),
            thunder_synth::secs(kind)
        );
    }

    let segments = lightning::ground_bolt(0.41, [0.0, 0.0, 0.0], [100.0, 1600.0, 50.0]);
    let previews = [
        ("very-near-100m.wav", 100.0, 7),
        ("near-250m.wav", 250.0, 7),
        ("near-250m-seed-2.wav", 250.0, 2),
        ("near-250m-seed-19.wav", 250.0, 19),
        ("medium-1200m.wav", 1200.0, 7),
        ("distant-4500m.wav", 4500.0, 7),
    ];
    for (name, distance, seed) in previews {
        let listener = [distance, 20.0, 0.0];
        let sources = thunder::channel_sources(&segments, listener);
        let samples = thunder::synthesize_thunder(&sources, seed, false);
        let path = output_directory.join(name);
        write_wav(&path, &samples)?;
        println!(
            "{} ({:.2} seconds)",
            path.display(),
            samples.len() as f32 / THUNDER_SAMPLE_RATE as f32
        );
    }

    let sources = thunder::cloud_sources([1200.0, 0.0, 0.0], 1300.0, [0.0, 20.0, 0.0], 0.41);
    let samples = thunder::synthesize_thunder(&sources, 7, true);
    let path = output_directory.join("cloud.wav");
    write_wav(&path, &samples)?;
    println!(
        "{} ({:.2} seconds)",
        path.display(),
        samples.len() as f32 / THUNDER_SAMPLE_RATE as f32
    );

    Ok(())
}
