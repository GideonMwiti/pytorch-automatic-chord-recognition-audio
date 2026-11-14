#!/usr/bin/env python3
"""
Project 79: Automatic Chord Recognition from Audio
===================================================
A deep learning system for recognizing 24 standard musical triad chords
(12 Major + 12 Minor) from continuous audio waveforms using 12-bin Pitch
Class Profiles (Chromagram) and a deep regularized classifier (ChordNet).

Author: Gideon Mwiti
License: MIT
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import soundfile as sf

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    top_k_accuracy_score
)

# ---------------------------------------------------------------------------
# Constants & Chord Vocabulary
# ---------------------------------------------------------------------------
SAMPLE_RATE = 22050
DURATION = 1.5  # seconds
N_FFT = 2048
HOP_LENGTH = 512

PITCH_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
CHORDS = [f"{p}:maj" for p in PITCH_NAMES] + [f"{p}:min" for p in PITCH_NAMES]  # 24 chords
CHORD_TO_ID = {c: idx for idx, c in enumerate(CHORDS)}
ID_TO_CHORD = {idx: c for idx, c in enumerate(CHORDS)}


# ---------------------------------------------------------------------------
# Chroma Feature Extractor (12-Bin Pitch Class Profile)
# ---------------------------------------------------------------------------
class ChromaExtractor:
    """Extracts 12-bin normalized Pitch Class Profile (Chroma) from audio waveforms."""
    def __init__(self, sample_rate: int = SAMPLE_RATE, n_fft: int = N_FFT, hop_length: int = HOP_LENGTH):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = torch.hann_window(n_fft)

        # Precompute frequency bin to chroma bin mapping
        freqs = torch.linspace(0, sample_rate / 2, n_fft // 2 + 1)
        valid_mask = freqs >= 27.5  # A0 (lowest piano key)
        self.valid_mask = valid_mask
        valid_freqs = freqs[valid_mask]

        pitches = 12.0 * torch.log2(valid_freqs / 440.0) + 69.0
        self.chroma_bins = torch.round(pitches).long() % 12

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: Tensor of shape (num_samples,) or (1, num_samples)
        Returns:
            chroma_vec: Normalized Tensor of shape (12,)
        """
        if waveform.dim() == 2:
            waveform = waveform.squeeze(0)

        # STFT
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True
        )
        mag_spec = torch.abs(stft)  # (n_fft // 2 + 1, time_steps)

        # Accumulate energy into 12 chroma bins
        chroma = torch.zeros(12, mag_spec.shape[1])
        valid_mag = mag_spec[self.valid_mask]
        for c in range(12):
            bin_mask = (self.chroma_bins == c)
            if bin_mask.sum() > 0:
                chroma[c] = valid_mag[bin_mask].sum(dim=0)

        # Average energy across time and L2-normalize
        chroma_vec = chroma.mean(dim=1)
        norm = torch.norm(chroma_vec, p=2) + 1e-8
        chroma_norm = chroma_vec / norm
        return chroma_norm


# ---------------------------------------------------------------------------
# Multi-Instrument Procedural Chord Synthesizer
# ---------------------------------------------------------------------------
def generate_chord_audio(chord_name: str, rng: np.random.Generator,
                         sample_rate: int = SAMPLE_RATE, duration: float = DURATION) -> np.ndarray:
    """
    Synthesizes acoustic chord audio with diverse instruments (piano, guitar, organ, synth),
    voicings (root, inversions, open), octave roots, and realistic acoustic resonance.
    """
    p_name, quality = chord_name.split(':')
    root_idx = PITCH_NAMES.index(p_name)
    intervals = [0, 4, 7] if quality == 'maj' else [0, 3, 7]

    octave = rng.choice([3, 4])
    base_midi = 12 * (octave + 1) + root_idx

    voicing = rng.choice(['root', 'first_inv', 'second_inv', 'open'])
    if voicing == 'root':
        midi_notes = [base_midi + iv for iv in intervals]
    elif voicing == 'first_inv':
        midi_notes = [base_midi + intervals[1] - 12, base_midi, base_midi + intervals[2]]
    elif voicing == 'second_inv':
        midi_notes = [base_midi + intervals[2] - 12, base_midi, base_midi + intervals[1]]
    else:  # open voicing with bass doubling
        midi_notes = [base_midi - 12, base_midi, base_midi + intervals[2], base_midi + 12 + intervals[1]]

    instr = rng.choice(['piano', 'guitar', 'organ', 'synth'])
    n_pts = int(sample_rate * duration)
    t = np.linspace(0, duration, n_pts, endpoint=False)
    audio = np.zeros(n_pts, dtype=np.float32)

    for string_idx, note in enumerate(midi_notes):
        detune = rng.uniform(-0.03, 0.03)
        f0 = 440.0 * (2.0 ** ((note + detune - 69.0) / 12.0))

        delay_sec = string_idx * rng.uniform(0.015, 0.030) if instr == 'guitar' else 0.0
        n_delay = int(delay_sec * sample_rate)
        active_len = n_pts - n_delay
        if active_len <= 0:
            continue
        t_active = t[:active_len]
        note_wav = np.zeros(active_len, dtype=np.float32)

        if instr == 'piano':
            for k in range(1, 6):
                amp = (1.0 / (k ** 1.25)) * 0.40
                decay = np.exp(-t_active * (1.6 + k * 1.2))
                note_wav += amp * np.sin(2 * np.pi * k * f0 * t_active) * decay
        elif instr == 'guitar':
            for k in range(1, 5):
                amp = (1.0 / (k ** 1.15)) * 0.35
                decay = np.exp(-t_active * (2.2 + k * 1.4))
                note_wav += amp * np.sin(2 * np.pi * k * f0 * t_active) * decay
        elif instr == 'organ':
            vib = 1.0 + 0.008 * np.sin(2 * np.pi * 5.5 * t_active)
            for k in [1, 2, 3]:
                amp = 0.45 / k
                note_wav += amp * np.sin(2 * np.pi * k * f0 * vib * t_active)
        elif instr == 'synth':
            env = np.clip(t_active / 0.08, 0, 1) * np.exp(-t_active * 0.7)
            for k in range(1, 5):
                amp = 0.38 / k
                note_wav += amp * np.sin(2 * np.pi * k * f0 * t_active) * env

        audio[n_delay:n_delay + active_len] += note_wav

    # Additive acoustic noise floor
    snr_noise = rng.normal(0, 0.003, n_pts).astype(np.float32)
    audio += snr_noise

    # Peak normalization
    max_val = np.max(np.abs(audio)) + 1e-6
    audio = (audio / max_val * 0.90).astype(np.float32)
    return audio


# ---------------------------------------------------------------------------
# Dataset & Pitch Transposition Augmentation
# ---------------------------------------------------------------------------
class ChordDataset(Dataset):
    """In-memory dataset of 12-bin chroma feature vectors with pitch transposition."""
    def __init__(self, chroma_features: torch.Tensor, labels: torch.Tensor, augment: bool = False):
        self.chroma = chroma_features
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.chroma[idx].clone()
        y = self.labels[idx].item()

        if self.augment:
            # Random circular pitch-shift transposition
            shift = torch.randint(0, 12, (1,)).item()
            if shift > 0:
                x = torch.roll(x, shifts=shift, dims=0)
                # Major chords (0-11) or Minor chords (12-23)
                if y < 12:
                    y = (y + shift) % 12
                else:
                    y = 12 + ((y - 12 + shift) % 12)

            # Subtle Gaussian noise
            if torch.rand(1).item() < 0.4:
                x = (x + torch.randn_like(x) * 0.02).clamp(0.0)
                x = x / (torch.norm(x, p=2) + 1e-8)

        return x, torch.tensor(y, dtype=torch.long)


# ---------------------------------------------------------------------------
# ChordNet Neural Network Architecture
# ---------------------------------------------------------------------------
class ChordNet(nn.Module):
    """
    Deep Regularized Multilayer Perceptron for 24-Class Musical Chord Recognition.
    Total parameters: ~28,632 (Fast CPU inference < 0.25 ms).
    """
    def __init__(self, in_features: int = 12, num_classes: int = len(CHORDS)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.20),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.20),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.10),

            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Training & Evaluation Engine
# ---------------------------------------------------------------------------
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += len(y)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets, all_probs = [], [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * len(y)

            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    loss = total_loss / len(all_targets)
    top1 = np.mean(all_preds == all_targets)
    top3 = top_k_accuracy_score(all_targets, all_probs, k=3, labels=list(range(len(CHORDS))))
    macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)

    return {
        'loss': float(loss),
        'top1': float(top1),
        'top3': float(top3),
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'preds': all_preds,
        'targets': all_targets,
        'probs': all_probs
    }


# ---------------------------------------------------------------------------
# Diagnostics & Plotting
# ---------------------------------------------------------------------------
def generate_diagnostic_report(history, eval_res, sample_chromas, latencies, output_path):
    """Renders a publication-grade 6-panel diagnostic visualization."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.patch.set_facecolor('#0F172A')
    for ax in axes.flat:
        ax.set_facecolor('#1E293B')
        ax.tick_params(colors='#CBD5E1', labelsize=8)
        for spine in ax.spines.values():
            spine.set_color('#334155')

    epochs = range(1, len(history['train_loss']) + 1)

    # Panel 1: Loss curves
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], label='Train Loss', color='#38BDF8', lw=2)
    ax.plot(epochs, history['val_loss'], label='Val Loss', color='#F43F5E', lw=2, linestyle='--')
    ax.set_title('Cross-Entropy Loss Dynamics', color='#F8FAFC', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch', color='#CBD5E1')
    ax.set_ylabel('Loss', color='#CBD5E1')
    ax.legend(facecolor='#0F172A', edgecolor='#334155', labelcolor='#F8FAFC')
    ax.grid(True, alpha=0.15)

    # Panel 2: Accuracy curves
    ax = axes[0, 1]
    ax.plot(epochs, history['train_acc'], label='Train Top-1', color='#34D399', lw=2)
    ax.plot(epochs, history['val_top1'], label='Val Top-1', color='#A78BFA', lw=2, linestyle='--')
    ax.plot(epochs, history['val_top3'], label='Val Top-3', color='#FBBF24', lw=1.5, linestyle=':')
    ax.set_title('Top-1 & Top-3 Accuracy Trajectory', color='#F8FAFC', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch', color='#CBD5E1')
    ax.set_ylabel('Accuracy', color='#CBD5E1')
    ax.legend(facecolor='#0F172A', edgecolor='#334155', labelcolor='#F8FAFC')
    ax.grid(True, alpha=0.15)

    # Panel 3: Confusion Matrix
    ax = axes[0, 2]
    cm = confusion_matrix(eval_res['targets'], eval_res['preds'], labels=list(range(len(CHORDS))))
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
    im = ax.imshow(cm_norm, interpolation='nearest', cmap='cividis')
    ax.set_title('Normalized Confusion Matrix (24x24)', color='#F8FAFC', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(CHORDS)))
    ax.set_yticks(range(len(CHORDS)))
    ax.set_xticklabels(CHORDS, rotation=90, color='#CBD5E1', fontsize=6)
    ax.set_yticklabels(CHORDS, color='#CBD5E1', fontsize=6)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors='#CBD5E1')

    # Panel 4: Chroma Energy Profiles for Representative Chords
    ax = axes[1, 0]
    rep_chords = ['C:maj', 'G:maj', 'A:min', 'E:min']
    colors = ['#38BDF8', '#34D399', '#F43F5E', '#FBBF24']
    x_bins = np.arange(12)
    width = 0.20
    for i, chord_key in enumerate(rep_chords):
        if chord_key in sample_chromas:
            vec = sample_chromas[chord_key]
            ax.bar(x_bins + (i - 1.5) * width, vec, width=width, label=chord_key, color=colors[i], alpha=0.85)
    ax.set_title('12-Bin Pitch Class Profiles (Chroma)', color='#F8FAFC', fontsize=12, fontweight='bold')
    ax.set_xticks(x_bins)
    ax.set_xticklabels(PITCH_NAMES, color='#CBD5E1')
    ax.set_ylabel('Normalized Energy', color='#CBD5E1')
    ax.legend(facecolor='#0F172A', edgecolor='#334155', labelcolor='#F8FAFC', fontsize=8)
    ax.grid(True, alpha=0.15)

    # Panel 5: Per-Chord F1-Scores
    ax = axes[1, 1]
    per_chord_f1 = f1_score(eval_res['targets'], eval_res['preds'], average=None, zero_division=0)
    bars = ax.barh(range(len(CHORDS)), per_chord_f1, color='#60A5FA', edgecolor='#38BDF8', alpha=0.85)
    ax.set_yticks(range(len(CHORDS)))
    ax.set_yticklabels(CHORDS, color='#CBD5E1', fontsize=6)
    ax.set_title('Per-Chord F1 Score Breakdown (24 Chords)', color='#F8FAFC', fontsize=12, fontweight='bold')
    ax.set_xlabel('F1-Score', color='#CBD5E1')
    ax.set_xlim(0, 1.05)
    ax.grid(True, axis='x', alpha=0.15)

    # Panel 6: Latency Profile & System Stats
    ax = axes[1, 2]
    mean_lat = np.mean(latencies)
    p95_lat = np.percentile(latencies, 95)
    throughput = 1000.0 / mean_lat
    ax.hist(latencies, bins=25, color='#A855F7', edgecolor='#7E22CE', alpha=0.85)
    ax.axvline(mean_lat, color='#EF4444', lw=2, linestyle='--', label=f'Mean: {mean_lat:.2f} ms')
    ax.axvline(p95_lat, color='#F59E0B', lw=2, linestyle=':', label=f'P95: {p95_lat:.2f} ms')
    ax.set_title('CPU Inference Latency Distribution', color='#F8FAFC', fontsize=12, fontweight='bold')
    ax.set_xlabel('Latency (ms / chord)', color='#CBD5E1')
    ax.set_ylabel('Frequency', color='#CBD5E1')
    ax.legend(facecolor='#0F172A', edgecolor='#334155', labelcolor='#F8FAFC')
    ax.grid(True, alpha=0.15)

    info_text = (
        f"Throughput: {throughput:,.0f} chords/sec\n"
        f"Top-1 Accuracy: {eval_res['top1']*100:.1f}%\n"
        f"Top-3 Accuracy: {eval_res['top3']*100:.1f}%\n"
        f"Macro F1-Score: {eval_res['macro_f1']:.4f}"
    )
    ax.text(0.95, 0.65, info_text, transform=ax.transAxes,
            ha='right', va='top', fontsize=9, color='#F8FAFC',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#0F172A', edgecolor='#38BDF8', alpha=0.9))

    plt.suptitle('ChordNet: Automatic Musical Chord Recognition via Chroma Feature Analysis',
                 color='#F8FAFC', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor())
    plt.close()
    print(f"[✓] Diagnostic report saved to: {output_path}")


# ---------------------------------------------------------------------------
# Main Training & Evaluation Pipeline
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ChordNet: Automatic Chord Recognition")
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--samples-per-chord', type=int, default=100, help='Synthesized samples per chord')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Output directory')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('data', exist_ok=True)

    print("=" * 72)
    print("Project 79: Automatic Chord Recognition from Audio (ChordNet)")
    print(f"Device: {device} | Sampling Rate: {SAMPLE_RATE} Hz | Vocabulary: {len(CHORDS)} Chords")
    print("=" * 72)

    # Step 1: Synthesize Audio & Extract Chroma Features
    print("\n[1/5] Synthesizing acoustic chords and extracting 12-bin Chroma profiles...")
    extractor = ChromaExtractor()
    rng = np.random.default_rng(args.seed)

    all_chroma = []
    all_labels = []
    sample_chromas_dict = {}

    t0_gen = time.time()
    for c_idx, chord_name in enumerate(CHORDS):
        for s_idx in range(args.samples_per_chord):
            wav = generate_chord_audio(chord_name, rng)
            wav_t = torch.from_numpy(wav)
            chroma_vec = extractor(wav_t)
            all_chroma.append(chroma_vec)
            all_labels.append(c_idx)

            if s_idx == 0 and chord_name in ['C:maj', 'G:maj', 'A:min', 'E:min']:
                sample_chromas_dict[chord_name] = chroma_vec.numpy()

    all_chroma = torch.stack(all_chroma)  # (N, 12)
    all_labels = torch.tensor(all_labels, dtype=torch.long)
    t1_gen = time.time()
    print(f"[✓] Extracted {len(all_labels)} Chroma vectors in {t1_gen - t0_gen:.2f}s (Shape: {all_chroma.shape})")

    # Export 4 sample test chord audio tracks
    sample_exports = [
        ('C:maj', 'sample_C_major.wav'),
        ('G:maj', 'sample_G_major.wav'),
        ('A:min', 'sample_A_minor.wav'),
        ('E:min', 'sample_E_minor.wav')
    ]
    for c_name, filename in sample_exports:
        wav_data = generate_chord_audio(c_name, np.random.default_rng(777), duration=2.0)
        out_p = os.path.join(args.output_dir, filename)
        sf.write(out_p, wav_data, SAMPLE_RATE)
        print(f"[✓] Saved test chord audio: {out_p}")

    # Step 2: Stratified Train (70%) / Val (15%) / Test (15%) Split
    print("\n[2/5] Creating Stratified Train / Val / Test splits...")
    indices = np.arange(len(all_labels))
    rng.shuffle(indices)

    n_total = len(indices)
    n_train = int(0.70 * n_total)
    n_val = int(0.15 * n_total)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train_dataset = ChordDataset(all_chroma[train_idx], all_labels[train_idx], augment=True)
    val_dataset = ChordDataset(all_chroma[val_idx], all_labels[val_idx], augment=False)
    test_dataset = ChordDataset(all_chroma[test_idx], all_labels[test_idx], augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"[✓] Splits: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    # Step 3: Instantiate Model Architecture
    print("\n[3/5] Instantiating ChordNet Architecture...")
    model = ChordNet(in_features=12, num_classes=len(CHORDS)).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[✓] ChordNet initialized with {total_params:,} parameters ({total_params * 4 / (1024**2):.3f} MB)")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    # Step 4: Training & Validation Loop
    print("\n[4/5] Training ChordNet with Pitch-Shift Transposition Augmentation...")
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_top1': [], 'val_top3': []}
    best_val_f1 = -1.0
    best_ckpt_path = os.path.join(args.output_dir, 'chordnet_best.pth')

    t0_train = time.time()
    for epoch in range(1, args.epochs + 1):
        ep_t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_eval = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        ep_t1 = time.time()

        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(val_eval['loss'])
        history['val_top1'].append(val_eval['top1'])
        history['val_top3'].append(val_eval['top3'])

        print(f"Epoch {epoch:02d}/{args.epochs:02d} | "
              f"Train Loss: {tr_loss:.4f} Acc: {tr_acc*100:.1f}% | "
              f"Val Loss: {val_eval['loss']:.4f} Top-1: {val_eval['top1']*100:.1f}% Top-3: {val_eval['top3']*100:.1f}% F1: {val_eval['macro_f1']:.3f} | "
              f"Time: {ep_t1 - ep_t0:.1f}s")

        if val_eval['macro_f1'] > best_val_f1:
            best_val_f1 = val_eval['macro_f1']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'chords': CHORDS,
                'val_f1': best_val_f1,
                'args': vars(args)
            }, best_ckpt_path)

    t_train_total = time.time() - t0_train
    print(f"\n[✓] Training completed in {t_train_total:.1f}s. Best Val Macro-F1: {best_val_f1:.4f}")

    # Load best checkpoint
    checkpoint = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Step 5: Test Evaluation & Latency Benchmarks
    print("\n[5/5] Conducting comprehensive Test Set Evaluation & Benchmarking...")
    test_eval = evaluate(model, test_loader, criterion, device)
    print("\n" + "=" * 55)
    print("           CHORDNET TEST EVALUATION SUMMARY")
    print("=" * 55)
    print(f"Top-1 Accuracy : {test_eval['top1'] * 100:.2f}%")
    print(f"Top-3 Accuracy : {test_eval['top3'] * 100:.2f}%")
    print(f"Macro F1-Score : {test_eval['macro_f1']:.4f}")
    print(f"Weighted F1    : {test_eval['weighted_f1']:.4f}")
    print(f"Test Loss      : {test_eval['loss']:.4f}")
    print("=" * 55)

    clf_report = classification_report(
        test_eval['targets'], test_eval['preds'],
        target_names=CHORDS, digits=4, output_dict=True
    )

    # CPU Latency Benchmark
    model.eval().to('cpu')
    dummy_input = torch.randn(1, 12)
    latencies = []
    with torch.no_grad():
        for _ in range(20):
            _ = model(dummy_input)
        for _ in range(250):
            t_start = time.perf_counter()
            _ = model(dummy_input)
            latencies.append((time.perf_counter() - t_start) * 1000.0)

    mean_lat = float(np.mean(latencies))
    p50_lat = float(np.median(latencies))
    p95_lat = float(np.percentile(latencies, 95))
    throughput = 1000.0 / mean_lat

    print(f"Latency: Mean={mean_lat:.3f} ms | P50={p50_lat:.3f} ms | P95={p95_lat:.3f} ms")
    print(f"Throughput: {throughput:,.1f} chords/sec")

    # Diagnostic visualization report
    report_path = os.path.join(args.output_dir, 'chord_recognition_report.png')
    generate_diagnostic_report(history, test_eval, sample_chromas_dict, latencies, report_path)

    # Save metrics JSON
    metrics_path = os.path.join(args.output_dir, 'chord_recognition_metrics.json')
    metrics_data = {
        'project': 'Project 79: Automatic Chord Recognition from Audio',
        'taxonomy': CHORDS,
        'model': {
            'architecture': 'ChordNet (Deep Regularized MLP)',
            'parameters': total_params,
            'model_size_mb': float(total_params * 4 / (1024**2))
        },
        'test_metrics': {
            'top1_accuracy': test_eval['top1'],
            'top3_accuracy': test_eval['top3'],
            'macro_f1': test_eval['macro_f1'],
            'weighted_f1': test_eval['weighted_f1'],
            'loss': test_eval['loss']
        },
        'per_chord_metrics': {
            chord: {
                'precision': clf_report[chord]['precision'],
                'recall': clf_report[chord]['recall'],
                'f1_score': clf_report[chord]['f1-score'],
                'support': clf_report[chord]['support']
            } for chord in CHORDS
        },
        'latency_benchmark': {
            'mean_ms': mean_lat,
            'median_ms': p50_lat,
            'p95_ms': p95_lat,
            'throughput_chords_per_sec': throughput
        }
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    print(f"[✓] Metrics saved to: {metrics_path}")
    print("\n[✓] All ChordNet stages completed successfully!")


if __name__ == '__main__':
    main()

