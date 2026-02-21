from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.utils.audio import load_wav
from src.utils.mel import MelSpectrogram, MelSpectrogramConfig


class RuslanVocoderDataset(Dataset):
    def __init__(self, audio_dir, sample_rate=22050, segment_size=16384, file_pattern='**/*.wav', random_crop=True, mel_cfg=None):
        self.audio_dir = Path(audio_dir)
        self.sample_rate = sample_rate
        self.segment_size = segment_size
        self.random_crop = random_crop
        self.paths = sorted(self.audio_dir.glob(file_pattern))
        if len(self.paths) == 0:
            raise RuntimeError(f'No wav files found in {audio_dir}')
        self.mel_extractor = MelSpectrogram(mel_cfg or MelSpectrogramConfig(sr=sample_rate))

    def __len__(self):
        return len(self.paths)

    def _crop_or_pad(self, wav):
        T = wav.numel()
        if T >= self.segment_size:
            start = torch.randint(0, T - self.segment_size + 1, (1,)).item() if self.random_crop else 0
            wav = wav[start:start+self.segment_size]
        else:
            wav = torch.nn.functional.pad(wav, (0, self.segment_size - T))
        return wav

    def __getitem__(self, idx):
        path = self.paths[idx]
        wav = load_wav(str(path), self.sample_rate)
        wav = self._crop_or_pad(wav)
        mel = self.mel_extractor(wav.unsqueeze(0)).squeeze(0)
        return {'audio': wav, 'mel': mel, 'path': str(path)}
