import argparse
from pathlib import Path

import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt


def load_mono_resampled(path: str, target_sr: int = 22050) -> tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr
    return wav.squeeze(0), sr


def compute_log_mel(wav: torch.Tensor, sr: int = 22050) -> np.ndarray:
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=1024,
        win_length=1024,
        hop_length=256,
        f_min=0,
        f_max=11025 if sr == 22050 else sr // 2,
        n_mels=80,
        power=2.0,
    )(wav.unsqueeze(0))
    mel = mel.clamp_min(1e-5).log().squeeze(0).cpu().numpy()
    return mel


def plot_quality_bars(out_dir: Path):
    # Твои агрегированные оценки
    # Для "роботизированности" и "артефактов" это штрафы (меньше лучше)
    labels = [
        "Разборчивость",
        "Натуральность",
        "Роботизированность (штраф)",
        "Артефакты (штраф)",
        "Интонация",
    ]
    values = [2.0, 1.0, 2.0, 0.5, 2.0]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values)
    ax.set_ylim(0, 2.2)
    ax.set_ylabel("Оценка (0..2)")
    ax.set_title("Качественная оценка baseline-вокодера (агрегировано, 15 файлов)")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=15, ha="right")

    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:g}", ha="center", va="bottom")

    out_path = out_dir / "quality_bars_aggregate.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"[OK] Saved: {out_path}")


def plot_quality_radar(out_dir: Path):
    # Для radar инвертируем штрафы, чтобы "больше = лучше"
    labels = [
        "Разборчивость",
        "Натуральность",
        "Анти-роботизация",
        "Чистота (мало артефактов)",
        "Интонация",
    ]
    # robot_penalty=2.0 -> anti_robot=0.0 ; artifacts_penalty=0.5 -> clean=1.5
    values = [2.0, 1.0, 0.0, 1.5, 2.0]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values_closed = values + values[:1]
    angles_closed = angles + angles[:1]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles_closed, values_closed)
    ax.fill(angles_closed, values_closed, alpha=0.2)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 2.0)
    ax.set_yticks([0.5, 1.0, 1.5, 2.0])
    ax.set_title("Сводный профиль качества (0..2, больше = лучше)")

    out_path = out_dir / "quality_radar_aggregate.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"[OK] Saved: {out_path}")


def plot_audio_comparison(orig_path: str, gen_path: str, out_dir: Path, prefix: str = "sample_compare"):
    orig_wav, sr = load_mono_resampled(orig_path, 22050)
    gen_wav, _ = load_mono_resampled(gen_path, 22050)

    min_len = min(orig_wav.numel(), gen_wav.numel())
    orig = orig_wav[:min_len].cpu().numpy()
    gen = gen_wav[:min_len].cpu().numpy()

    t = np.arange(min_len) / sr

    # Waveform figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(t, orig)
    axes[0].set_title("Original waveform")
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, gen)
    axes[1].set_title("Generated waveform (vocoder)")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(alpha=0.3)

    wave_out = out_dir / f"{prefix}_waveforms.png"
    plt.tight_layout()
    plt.savefig(wave_out, dpi=160)
    plt.close(fig)
    print(f"[OK] Saved: {wave_out}")

    # Mel comparison figure
    orig_mel = compute_log_mel(torch.tensor(orig), sr)
    gen_mel = compute_log_mel(torch.tensor(gen), sr)

    vmin = min(orig_mel.min(), gen_mel.min())
    vmax = max(orig_mel.max(), gen_mel.max())

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    im0 = axes[0].imshow(orig_mel, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
    axes[0].set_title("Original log-mel")
    axes[0].set_ylabel("Mel bin")

    im1 = axes[1].imshow(gen_mel, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
    axes[1].set_title("Generated log-mel")
    axes[1].set_ylabel("Mel bin")
    axes[1].set_xlabel("Frame")

    fig.colorbar(im1, ax=axes, fraction=0.02, pad=0.01)
    mel_out = out_dir / f"{prefix}_mels.png"
    plt.tight_layout()
    plt.savefig(mel_out, dpi=160)
    plt.close(fig)
    print(f"[OK] Saved: {mel_out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="report/figures")
    parser.add_argument("--orig_wav", type=str, default="")
    parser.add_argument("--gen_wav", type=str, default="")
    parser.add_argument("--prefix", type=str, default="sample_compare")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_quality_bars(out_dir)
    plot_quality_radar(out_dir)

    if args.orig_wav and args.gen_wav:
        plot_audio_comparison(args.orig_wav, args.gen_wav, out_dir, args.prefix)
    else:
        print("[INFO] orig/gen wav not provided -> skipping waveform/mel comparison")


if __name__ == "__main__":
    main()