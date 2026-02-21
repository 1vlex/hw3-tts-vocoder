from pathlib import Path

from torch.utils.data import Dataset

from src.utils.audio import load_wav
from src.utils.mel import MelSpectrogram, MelSpectrogramConfig


class CustomDirDataset(Dataset):
    """Обязательный датасет по условию ДЗ."""

    def __init__(self, root_dir, sample_rate=22050, mel_cfg=None, load_audio=True, load_text=True):
        self.root_dir = Path(root_dir)
        self.audio_dir = self.root_dir / 'audio'
        self.text_dir = self.root_dir / 'transcriptions'
        self.sample_rate = sample_rate
        self.load_audio = load_audio
        self.load_text = load_text
        self.mel_extractor = MelSpectrogram(mel_cfg or MelSpectrogramConfig(sr=sample_rate))

        if not self.audio_dir.exists():
            raise RuntimeError(f'Audio dir not found: {self.audio_dir}')
        self.audio_files = sorted(self.audio_dir.glob('*.wav'))
        if len(self.audio_files) == 0:
            raise RuntimeError(f'No wav files in {self.audio_dir}')

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        wav_path = self.audio_files[idx]
        stem = wav_path.stem
        item = {'id': stem, 'wav_path': str(wav_path)}

        if self.load_audio:
            wav = load_wav(str(wav_path), self.sample_rate)
            mel = self.mel_extractor(wav.unsqueeze(0)).squeeze(0)
            item['audio'] = wav
            item['mel'] = mel

        if self.load_text:
            txt_path = self.text_dir / f'{stem}.txt'
            item['text'] = txt_path.read_text(encoding='utf-8').strip() if txt_path.exists() else None
        return item
