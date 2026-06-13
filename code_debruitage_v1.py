#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Audio Cleaner – Project 3 (Speech preprocessing)
Performs denoising and Voice Activity Detection (VAD) using librosa (no compilation required)
"""

import argparse
import librosa
import soundfile as sf
import noisereduce as nr
import numpy as np
import sys

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

def vad_librosa_split(audio, sr, top_db=30, min_silence_dur=0.3, min_speech_dur=0.5, pad_dur=0.2):
    """
    Voice activity detection using librosa.effects.split.
    top_db: threshold in dB below the reference (lower = more sensitive)
    min_silence_dur: minimum silence duration between speech segments (seconds)
    min_speech_dur: minimum speech segment duration (seconds)
    pad_dur: padding added before/after each segment (seconds)
    Returns: list of (start_sample, end_sample) indices
    """
    # split returns intervals in samples
    intervals = librosa.effects.split(audio, top_db=top_db,
                                      frame_length=int(sr*0.025),
                                      hop_length=int(sr*0.010))
    # filter short speech segments
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
        return np.zeros(0, dtype=audio.dtype)
    speech_parts = [audio[s:e] for s, e in segments]
    return np.concatenate(speech_parts)

def main():
    parser = argparse.ArgumentParser(description="Clean audio: denoise + VAD (no compilation)")
    parser.add_argument("input", help="Input audio file (wav, mp3, etc.)")
    parser.add_argument("-o", "--output", default="cleaned_output.wav",
                        help="Output file for denoised audio")
    parser.add_argument("--vad-output", default="vad_output.wav",
                        help="Output file after VAD (silence removed)")
    parser.add_argument("--no-denoise", action="store_true",
                        help="Skip denoising step")
    parser.add_argument("--top-db", type=float, default=30,
                        help="VAD sensitivity (lower = more sensitive, default 30)")
    args = parser.parse_args()

    print(f"Loading: {args.input}")
    audio, sr = load_audio(args.input)
    print(f"Sample rate: {sr} Hz, duration: {len(audio)/sr:.2f} sec")

    # Denoising
    if not args.no_denoise:
        print("Applying denoising...")
        audio = denoise(audio, sr)
        sf.write(args.output, audio, sr)
        print(f"Denoised audio saved to: {args.output}")
    else:
        print("Skipping denoising.")

    # VAD
    print("Performing Voice Activity Detection (librosa.split)...")
    segments = vad_librosa_split(audio, sr, top_db=args.top_db)
    print(f"Found {len(segments)} speech segment(s)")
    if segments:
        speech_audio = apply_vad(audio, segments)
        sf.write(args.vad_output, speech_audio, sr)
        print(f"VAD output (only speech) saved to: {args.vad_output}")
    else:
        print("No speech detected. VAD output not saved.")
        sys.exit(1)

if __name__ == "__main__":
    main()