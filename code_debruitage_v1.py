
"""
Batch processes an entire folder of audio files:
- Copies originals to 'original/'
- Applies denoising (optional) and saves to 'denoised/'
- Applies VAD and saves speech segments to 'vad/'
"""

import argparse
import librosa
import soundfile as sf
import noisereduce as nr
import numpy as np
import os
import shutil
import sys
from pathlib import Path


def load_audio(file_path, sr_target=16000):
    """Load audio and resample to 16kHz (mono)."""
    audio, sr_orig = librosa.load(file_path, sr=None, mono=True)
    if sr_orig != sr_target:
        audio = librosa.resample(audio, orig_sr=sr_orig, target_sr=sr_target)
        sr = sr_target
    else:
        sr = sr_orig
    return audio, sr

def denoise(audio, sr, stationary=True):
    """Spectral noise reduction using noisereduce."""
    noise_len = int(0.5 * sr)
    if len(audio) > noise_len:
        noise_sample = audio[:noise_len]
    else:
        noise_sample = audio
    denoised = nr.reduce_noise(y=audio, sr=sr, y_noise=noise_sample, stationary=stationary)
    return denoised

def vad_librosa_split(audio, sr, top_db=30, min_silence_dur=0.3,
                      min_speech_dur=0.5, pad_dur=0.2):
    """
    Voice activity detection using librosa.effects.split.
    Returns list of (start_sample, end_sample) indices.
    """
    intervals = librosa.effects.split(audio, top_db=top_db,
                                      frame_length=int(sr*0.025),
                                      hop_length=int(sr*0.010))
    min_speech_samples = int(min_speech_dur * sr)
    pad_samples = int(pad_dur * sr)
    padded_segments = []
    for start, end in intervals:
        if end - start >= min_speech_samples:
            start_pad = max(0, start - pad_samples)
            end_pad = min(len(audio), end + pad_samples)
            padded_segments.append((start_pad, end_pad))
    return padded_segments

def apply_vad(audio, segments):
    """Extract speech segments and concatenate."""
    if not segments:
        return None
    speech_parts = [audio[s:e] for s, e in segments]
    return np.concatenate(speech_parts)


def main():
    parser = argparse.ArgumentParser(
        description="Batch audio cleaner: denoise + VAD on all files in a folder."
    )
    parser.add_argument("input_dir",
                        help="Path to the folder containing audio files to process")
    parser.add_argument("-o", "--output-dir", default="processed_audio",
                        help="Base output directory (default: 'processed_audio')")
    parser.add_argument("--no-denoise", action="store_true",
                        help="Skip denoising step (only VAD will be applied)")
    parser.add_argument("--top-db", type=float, default=30,
                        help="VAD sensitivity (lower = more sensitive, default 30)")
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    if not input_path.is_dir():
        print(f"Error: '{args.input_dir}' is not a valid directory.")
        sys.exit(1)

    # Create output structure
    output_base = Path(args.output_dir)
    original_dir = output_base / "original"
    denoised_dir = output_base / "denoised"
    vad_dir = output_base / "vad"

    for d in [original_dir, denoised_dir, vad_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Supported audio extensions (librosa can read many, but we filter)
    audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aiff', '.aif'}

    # Gather all audio files in input_dir (non‑recursive)
    files = [f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() in audio_extensions]

    if not files:
        print(f"No supported audio files found in '{args.input_dir}'. Exiting.")
        sys.exit(1)

    print(f"Found {len(files)} audio file(s) to process.")
    print(f"Output base: {output_base.resolve()}")

    for idx, file_path in enumerate(files, 1):
        base_name = file_path.stem  # without extension
        print(f"\n[{idx}/{len(files)}] Processing: {file_path.name}")

        try:
            # 1. Copy original file to 'original/' folder (preserve format)
            original_out = original_dir / file_path.name
            shutil.copy2(file_path, original_out)
            print(f"Copied original to {original_out}")

            # 2. Load audio (resampled to 16kHz mono)
            audio, sr = load_audio(str(file_path))
            print(f"Duration: {len(audio)/sr:.2f} sec, SR: {sr} Hz")

            # 3. Denoising (unless skipped)
            if not args.no_denoise:
                print("Applying denoising...")
                audio_denoised = denoise(audio, sr)
                denoised_out = denoised_dir / f"{base_name}.wav"
                sf.write(denoised_out, audio_denoised, sr)
                print(f"Denoised saved to {denoised_out}")
                # Use denoised audio for VAD
                audio_for_vad = audio_denoised
            else:
                print("Skipping denoising.")
                audio_for_vad = audio  # use original for VAD

            # 4. VAD
            print("Performing VAD...")
            segments = vad_librosa_split(audio_for_vad, sr, top_db=args.top_db)
            if segments:
                speech_audio = apply_vad(audio_for_vad, segments)
                vad_out = vad_dir / f"{base_name}_vad.wav"
                sf.write(vad_out, speech_audio, sr)
                print(f"VAD output saved to {vad_out} ({len(segments)} segment(s))")
            else:
                print("No speech detected; VAD file not saved.")

        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")

    print("\n All done! Processed files are in:")
    print(f"   - Originals:   {original_dir.resolve()}")
    if not args.no_denoise:
        print(f"   - Denoised:    {denoised_dir.resolve()}")
    else:
        print("   - Denoising was skipped.")
    print(f"   - VAD outputs: {vad_dir.resolve()}")

if __name__ == "__main__":
    main()
