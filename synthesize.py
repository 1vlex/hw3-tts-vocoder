from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader

from src.datasets.custom_dir_dataset import CustomDirDataset
from src.models.hifigan_generator import HiFiGANGenerator
from src.utils.audio import save_wav
from src.utils.checkpoint import load_checkpoint
from src.utils.mel import MelSpectrogramConfig


@hydra.main(version_base=None, config_path="src/configs", config_name="synthesize")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mel_cfg = MelSpectrogramConfig(
        sr=cfg.audio.sample_rate,
        win_length=cfg.audio.win_length,
        hop_length=cfg.audio.hop_length,
        n_fft=cfg.audio.n_fft,
        f_min=cfg.audio.f_min,
        f_max=cfg.audio.f_max,
        n_mels=cfg.audio.n_mels,
        power=cfg.audio.power,
    )

    dataset = CustomDirDataset(
        root_dir=to_absolute_path(cfg.infer.input_dir),
        sample_rate=cfg.audio.sample_rate,
        mel_cfg=mel_cfg,
        load_audio=True,
        load_text=True,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    G = HiFiGANGenerator(
        in_channels=cfg.audio.n_mels,
        upsample_rates=tuple(cfg.model.generator.upsample_rates),
        upsample_kernel_sizes=tuple(cfg.model.generator.upsample_kernel_sizes),
        upsample_initial_channel=cfg.model.generator.upsample_initial_channel,
    ).to(device)

    ckpt = load_checkpoint(to_absolute_path(cfg.infer.checkpoint_path), map_location=device)
    state = ckpt["generator"] if isinstance(ckpt, dict) and "generator" in ckpt else ckpt
    G.load_state_dict(state, strict=True)
    G.eval()

    out_dir = Path(to_absolute_path(cfg.infer.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch in loader:
            mel = batch["mel"].to(device)   # [1, 80, Tm]
            audio_pred = G(mel).squeeze(0).squeeze(0)
            stem = batch["id"][0]
            out_path = out_dir / f"{stem}_gen.wav"
            save_wav(str(out_path), audio_pred, cfg.audio.sample_rate)
            print(f"[OK] {out_path}")

    print("[DONE] Inference completed")


if __name__ == "__main__":
    main()
