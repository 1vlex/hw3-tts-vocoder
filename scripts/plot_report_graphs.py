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


def plot_quality_bars(
    out_dir: Path,
    title: str = "Качественная оценка baseline-вокодера (агрегировано, 15 файлов)",
    ylabel: str = "Оценка (0..2)",
):
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
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=15, ha="right")

    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:g}", ha="center", va="bottom")

    out_path = out_dir / "quality_bars_aggregate.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"[OK] Saved: {out_path}")


def plot_quality_radar(
    out_dir: Path,
    title: str = "Сводный профиль качества (0..2, больше = лучше)",
):
    labels = [
        "Разборчивость",
        "Натуральность",
        "Анти-роботизация",
        "Чистота (мало артефактов)",
        "Интонация",
    ]
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
    ax.set_title(title)

    out_path = out_dir / "quality_radar_aggregate.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"[OK] Saved: {out_path}")


def _resolve_title(custom: str, label: str, suffix: str) -> str:
    if custom:
        return custom
    return f"{label} {suffix}"


def plot_audio_comparison(
    orig_path: str,
    gen_path: str,
    out_dir: Path,
    prefix: str = "sample_compare",
    orig_label: str = "Original",
    gen_label: str = "Generated (vocoder)",
    orig_wave_title: str = "",
    gen_wave_title: str = "",
    orig_mel_title: str = "",
    gen_mel_title: str = "",
    time_xlabel: str = "Time (s)",
    frame_xlabel: str = "Frame",
    wave_ylabel: str = "Amplitude",
    mel_ylabel: str = "Mel bin",
    colorbar_label: str = "Log-mel value",
):
    orig_wav, sr = load_mono_resampled(orig_path, 22050)
    gen_wav, _ = load_mono_resampled(gen_path, 22050)

    min_len = min(orig_wav.numel(), gen_wav.numel())
    orig = orig_wav[:min_len].cpu().numpy()
    gen = gen_wav[:min_len].cpu().numpy()

    t = np.arange(min_len) / sr

    # Waveform figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(t, orig)
    axes[0].set_title(_resolve_title(orig_wave_title, orig_label, "waveform"))
    axes[0].set_ylabel(wave_ylabel)
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, gen)
    axes[1].set_title(_resolve_title(gen_wave_title, gen_label, "waveform"))
    axes[1].set_ylabel(wave_ylabel)
    axes[1].set_xlabel(time_xlabel)
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
    axes[0].set_title(_resolve_title(orig_mel_title, orig_label, "log-mel"))
    axes[0].set_ylabel(mel_ylabel)

    im1 = axes[1].imshow(gen_mel, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
    axes[1].set_title(_resolve_title(gen_mel_title, gen_label, "log-mel"))
    axes[1].set_ylabel(mel_ylabel)
    axes[1].set_xlabel(frame_xlabel)

    cbar = fig.colorbar(im1, ax=axes, fraction=0.02, pad=0.01)
    if colorbar_label:
        cbar.set_label(colorbar_label)

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

    parser.add_argument("--skip_aggregate", action="store_true")

    # Generic labels (used to build titles if explicit titles are not passed)
    parser.add_argument("--orig_label", type=str, default="Original")
    parser.add_argument("--gen_label", type=str, default="Generated (vocoder)")

    # Explicit plot titles
    parser.add_argument("--orig_wave_title", type=str, default="")
    parser.add_argument("--gen_wave_title", type=str, default="")
    parser.add_argument("--orig_mel_title", type=str, default="")
    parser.add_argument("--gen_mel_title", type=str, default="")

    # Axis names
    parser.add_argument("--time_xlabel", type=str, default="Time (s)")
    parser.add_argument("--frame_xlabel", type=str, default="Frame")
    parser.add_argument("--wave_ylabel", type=str, default="Amplitude")
    parser.add_argument("--mel_ylabel", type=str, default="Mel bin")
    parser.add_argument("--colorbar_label", type=str, default="Log-mel value")

    # Aggregate plot titles
    parser.add_argument(
        "--bars_title",
        type=str,
        default="Качественная оценка baseline-вокодера (агрегировано, 15 файлов)",
    )
    parser.add_argument(
        "--bars_ylabel",
        type=str,
        default="Оценка (0..2)",
    )
    parser.add_argument(
        "--radar_title",
        type=str,
        default="Сводный профиль качества (0..2, больше = лучше)",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_aggregate:
        plot_quality_bars(out_dir, title=args.bars_title, ylabel=args.bars_ylabel)
        plot_quality_radar(out_dir, title=args.radar_title)
    else:
        print("[INFO] --skip_aggregate enabled -> skipping aggregate bar/radar plots")

    if args.orig_wav and args.gen_wav:
        plot_audio_comparison(
            args.orig_wav,
            args.gen_wav,
            out_dir,
            args.prefix,
            orig_label=args.orig_label,
            gen_label=args.gen_label,
            orig_wave_title=args.orig_wave_title,
            gen_wave_title=args.gen_wave_title,
            orig_mel_title=args.orig_mel_title,
            gen_mel_title=args.gen_mel_title,
            time_xlabel=args.time_xlabel,
            frame_xlabel=args.frame_xlabel,
            wave_ylabel=args.wave_ylabel,
            mel_ylabel=args.mel_ylabel,
            colorbar_label=args.colorbar_label,
        )
    else:
        print("[INFO] orig/gen wav not provided -> skipping waveform/mel comparison")


if __name__ == "__main__":
    main()