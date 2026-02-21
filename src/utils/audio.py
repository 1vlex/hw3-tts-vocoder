from pathlib import Path

import soundfile as sf
import torch
import torchaudio


def load_wav(path: str, target_sr: int) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.squeeze(0).clamp(-1.0, 1.0)


def save_wav(path: str, audio: torch.Tensor, sr: int):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(p), audio.detach().cpu().float().clamp(-1.0, 1.0).numpy(), sr)
