#!/usr/bin/env python3
"""
Interactive Musical Chord Recognition CLI
=========================================
Classifies audio clips into 24 standard musical triad chords
(12 Major + 12 Minor) using Chroma Feature Extraction and ChordNet.

Usage:
    python recognize_chord.py --audio outputs/sample_C_major.wav
    python recognize_chord.py --audio outputs/sample_A_minor.wav --top-k 5
"""

import os
import sys
import argparse
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

from main import (
    ChordNet,
    ChromaExtractor,
    CHORDS,
    PITCH_NAMES,
    SAMPLE_RATE
)


def load_audio(audio_path: str, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Loads audio from disk and ensures mono at target sample rate."""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    data, sr = sf.read(audio_path, dtype='float32')
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        tensor_data = torch.from_numpy(data).unsqueeze(0)
        data = resampler(tensor_data).squeeze(0).numpy()
    return data


def display_chroma_bars(chroma_vec):
    """Prints a visual ASCII representation of the 12-bin Pitch Class Profile."""
    print("\n12-Bin Pitch Class Energy Profile (Chroma):")
    print("-" * 52)
    for p_name, val in zip(PITCH_NAMES, chroma_vec):
        bar_len = int(val * 35)
        bar = "█" * bar_len + "░" * (35 - bar_len)
        print(f" {p_name:<3} |{bar}| {val:.3f}")
    print("-" * 52)


def recognize_audio_chord(audio_path: str, model_path: str, top_k: int = 3, device_str: str = 'cpu'):
    device = torch.device(device_str)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}. Run main.py first!")

    ckpt = torch.load(model_path, map_location=device)
    chords = ckpt.get('chords', CHORDS)

    model = ChordNet(in_features=12, num_classes=len(chords)).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    waveform = load_audio(audio_path, target_sr=SAMPLE_RATE)
    duration_sec = len(waveform) / SAMPLE_RATE

    extractor = ChromaExtractor(sample_rate=SAMPLE_RATE)
    wav_t = torch.from_numpy(waveform)
    chroma_vec = extractor(wav_t).to(device)

    with torch.no_grad():
        logits = model(chroma_vec.unsqueeze(0))
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    top_indices = np.argsort(probs)[::-1][:top_k]
    top1_chord = chords[top_indices[0]]
    top1_conf = probs[top_indices[0]] * 100.0

    print("\n" + "=" * 58)
    print("      CHORDNET AUTOMATIC CHORD RECOGNITION REPORT")
    print("=" * 58)
    print(f"File Path        : {audio_path}")
    print(f"Duration         : {duration_sec:.2f} seconds ({len(waveform):,} samples)")
    print(f"Model Checkpoint : {model_path}")
    print("-" * 58)

    print(f"\n>>> IDENTIFIED CHORD: {top1_chord} ({top1_conf:.2f}% confidence)\n")

    print(f"{'Rank':<6}{'Chord':<12}{'Confidence':<12}{'Distribution Bar'}")
    print("-" * 58)
    for rank, idx in enumerate(top_indices, 1):
        c = chords[idx]
        conf = probs[idx] * 100.0
        bar_len = int(conf / 2.5)
        bar = "█" * bar_len + "░" * (40 - bar_len)
        print(f"#{rank:<5}{c:<12}{conf:>6.2f}%     |{bar}|")

    display_chroma_bars(chroma_vec.cpu().numpy())
    print("=" * 58 + "\n")

    return {
        'top_chord': top1_chord,
        'confidence': float(top1_conf),
        'top_k': [(chords[i], float(probs[i])) for i in top_indices]
    }


def main():
    parser = argparse.ArgumentParser(description="Recognize musical chords from audio")
    parser.add_argument('--audio', type=str, required=True, help='Path to audio file (.wav, .mp3)')
    parser.add_argument('--model', type=str, default='outputs/chordnet_best.pth', help='Path to trained model')
    parser.add_argument('--top-k', type=int, default=3, help='Top K candidate chords to display')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu or cuda)')
    args = parser.parse_args()

    recognize_audio_chord(args.audio, args.model, args.top_k, args.device)


if __name__ == '__main__':
    main()

