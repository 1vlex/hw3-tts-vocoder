from dataclasses import dataclass

import librosa
import torch
from torch import nn
import torchaudio


@dataclass
class MelSpectrogramConfig:
    sr: int = 22050
    win_length: int = 1024
    hop_length: int = 256
    n_fft: int = 1024
    f_min: int = 0
    f_max: int = 11025
    n_mels: int = 80
    power: float = 2.0
    pad_value: float = -11.5129251


class MelSpectrogram(nn.Module):
    def __init__(self, config: MelSpectrogramConfig):
        super().__init__()
        self.config = config
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sr,
            win_length=config.win_length,
            hop_length=config.hop_length,
            n_fft=config.n_fft,
            f_min=config.f_min,
            f_max=config.f_max,
            n_mels=config.n_mels,
        )
        try:
            self.mel_spectrogram.spectrogram.power = config.power
        except Exception:
            pass
        mel_basis = librosa.filters.mel(
            sr=config.sr,
            n_fft=config.n_fft,
            n_mels=config.n_mels,
            fmin=config.f_min,
            fmax=config.f_max,
        ).T
        with torch.no_grad():
            self.mel_spectrogram.mel_scale.fb.copy_(
                torch.tensor(mel_basis, dtype=self.mel_spectrogram.mel_scale.fb.dtype)
            )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return self.mel_spectrogram(audio).clamp_(min=1e-5).log_()
