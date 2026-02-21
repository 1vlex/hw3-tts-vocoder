from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.ruslan_dataset import RuslanVocoderDataset
from src.losses.hifigan_losses import discriminator_loss, generator_adv_loss, feature_matching_loss
from src.models.hifigan_discriminators import MultiScaleDiscriminator, MultiPeriodDiscriminator
from src.models.hifigan_generator import HiFiGANGenerator
from src.utils.checkpoint import save_checkpoint
from src.utils.common import seed_everything
from src.utils.logging_comet import init_comet, log_metrics, log_audio
from src.utils.mel import MelSpectrogram, MelSpectrogramConfig


@hydra.main(version_base=None, config_path="src/configs", config_name="train")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    seed_everything(int(cfg.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

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

    train_ds = RuslanVocoderDataset(
        audio_dir=to_absolute_path(cfg.data.train_audio_dir),
        sample_rate=cfg.audio.sample_rate,
        segment_size=cfg.train.segment_size,
        random_crop=True,
        mel_cfg=mel_cfg,
    )
    val_ds = RuslanVocoderDataset(
        audio_dir=to_absolute_path(cfg.data.val_audio_dir),
        sample_rate=cfg.audio.sample_rate,
        segment_size=cfg.train.segment_size,
        random_crop=False,
        mel_cfg=mel_cfg,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=max(0, int(cfg.train.num_workers) // 2),
        pin_memory=True,
    )

    G = HiFiGANGenerator(
        in_channels=cfg.audio.n_mels,
        upsample_rates=tuple(cfg.model.generator.upsample_rates),
        upsample_kernel_sizes=tuple(cfg.model.generator.upsample_kernel_sizes),
        upsample_initial_channel=cfg.model.generator.upsample_initial_channel,
    ).to(device)
    MSD = MultiScaleDiscriminator(num_scales=cfg.model.discriminator.num_scales).to(device)
    MPD = MultiPeriodDiscriminator(periods=tuple(cfg.model.discriminator.periods)).to(device)

    mel_extractor = MelSpectrogram(mel_cfg).to(device)

    opt_g = torch.optim.AdamW(G.parameters(), lr=cfg.train.lr_g, betas=(cfg.train.beta1, cfg.train.beta2))
    opt_d = torch.optim.AdamW(
        list(MSD.parameters()) + list(MPD.parameters()),
        lr=cfg.train.lr_d,
        betas=(cfg.train.beta1, cfg.train.beta2),
    )

    use_amp = bool(cfg.train.use_amp and device.type == "cuda")
    scaler_g = torch.cuda.amp.GradScaler(enabled=use_amp)
    scaler_d = torch.cuda.amp.GradScaler(enabled=use_amp)

    logger = init_comet(
        bool(cfg.logging.use_comet),
        str(cfg.logging.project),
        str(cfg.logging.run_name),
        OmegaConf.to_container(cfg, resolve=True),
    )

    ckpt_dir = Path(to_absolute_path(cfg.checkpoint.dir))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    global_step = 0

    for epoch in range(int(cfg.train.epochs)):
        G.train()
        MSD.train()
        MPD.train()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.train.epochs}")
        for batch in pbar:
            audio_real = batch["audio"].to(device)  # [B, T]
            mel = batch["mel"].to(device)           # [B, n_mels, Tm]

            with torch.cuda.amp.autocast(enabled=use_amp):
                audio_fake = G(mel).squeeze(1)      # [B, T']
                min_len = min(audio_real.shape[-1], audio_fake.shape[-1])
                audio_real_cut = audio_real[:, :min_len]
                audio_fake_cut = audio_fake[:, :min_len]

            # D step
            opt_d.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                real_msd = MSD(audio_real_cut.unsqueeze(1))
                fake_msd = MSD(audio_fake_cut.detach().unsqueeze(1))
                real_mpd = MPD(audio_real_cut.unsqueeze(1))
                fake_mpd = MPD(audio_fake_cut.detach().unsqueeze(1))
                d_loss = discriminator_loss(real_msd, fake_msd) + discriminator_loss(real_mpd, fake_mpd)

            scaler_d.scale(d_loss).backward()
            if float(cfg.train.grad_clip) > 0:
                scaler_d.unscale_(opt_d)
                torch.nn.utils.clip_grad_norm_(list(MSD.parameters()) + list(MPD.parameters()), float(cfg.train.grad_clip))
            scaler_d.step(opt_d)
            scaler_d.update()

            # G step
            opt_g.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                fake_msd = MSD(audio_fake_cut.unsqueeze(1))
                fake_mpd = MPD(audio_fake_cut.unsqueeze(1))

                # Recompute real outputs for feature matching (simplest, not most efficient)
                real_msd_fm = MSD(audio_real_cut.unsqueeze(1))
                real_mpd_fm = MPD(audio_real_cut.unsqueeze(1))

                adv_loss = generator_adv_loss(fake_msd) + generator_adv_loss(fake_mpd)
                fm_loss = feature_matching_loss(real_msd_fm, fake_msd) + feature_matching_loss(real_mpd_fm, fake_mpd)

                mel_fake = mel_extractor(audio_fake_cut)
                mel_real = mel_extractor(audio_real_cut)
                mel_loss = F.l1_loss(mel_fake, mel_real)

                g_loss = (
                    float(cfg.train.lambda_adv) * adv_loss
                    + float(cfg.train.lambda_fm) * fm_loss
                    + float(cfg.train.lambda_mel) * mel_loss
                )

            scaler_g.scale(g_loss).backward()
            if float(cfg.train.grad_clip) > 0:
                scaler_g.unscale_(opt_g)
                torch.nn.utils.clip_grad_norm_(G.parameters(), float(cfg.train.grad_clip))
            scaler_g.step(opt_g)
            scaler_g.update()

            pbar.set_postfix(g=float(g_loss.item()), d=float(d_loss.item()), mel=float(mel_loss.item()))

            if global_step % int(cfg.logging.log_every_steps) == 0:
                log_metrics(logger, {
                    "train/loss_g": float(g_loss.item()),
                    "train/loss_d": float(d_loss.item()),
                    "train/loss_adv": float(adv_loss.item()),
                    "train/loss_fm": float(fm_loss.item()),
                    "train/loss_mel": float(mel_loss.item()),
                    "train/epoch": epoch,
                }, step=global_step)

            if global_step % int(cfg.logging.log_audio_every_steps) == 0:
                log_audio(logger, "audio/fake", audio_fake_cut[0].detach().cpu().numpy(), cfg.audio.sample_rate, step=global_step)
                log_audio(logger, "audio/real", audio_real_cut[0].detach().cpu().numpy(), cfg.audio.sample_rate, step=global_step)

            if global_step > 0 and global_step % int(cfg.checkpoint.save_every_steps) == 0:
                save_checkpoint(str(ckpt_dir / "last.pt"), {
                    "generator": G.state_dict(),
                    "msd": MSD.state_dict(),
                    "mpd": MPD.state_dict(),
                    "opt_g": opt_g.state_dict(),
                    "opt_d": opt_d.state_dict(),
                    "global_step": global_step,
                    "epoch": epoch,
                    "config": OmegaConf.to_container(cfg, resolve=True),
                })
                save_checkpoint(str(ckpt_dir / "last_generator.pt"), {"generator": G.state_dict()})

            global_step += 1

        # validation
        G.eval()
        val_losses = []
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= int(cfg.train.val_num_batches):
                    break
                audio_real = batch["audio"].to(device)
                mel = batch["mel"].to(device)
                audio_fake = G(mel).squeeze(1)
                min_len = min(audio_real.shape[-1], audio_fake.shape[-1])
                mel_fake = mel_extractor(audio_fake[:, :min_len])
                mel_real = mel_extractor(audio_real[:, :min_len])
                val_losses.append(F.l1_loss(mel_fake, mel_real).item())
        val_mel = sum(val_losses) / max(1, len(val_losses))
        log_metrics(logger, {"val/mel_l1": val_mel, "val/epoch": epoch}, step=global_step)

        save_checkpoint(str(ckpt_dir / "epoch_last_generator.pt"), {"generator": G.state_dict(), "val_mel_l1": val_mel})
        if val_mel < best_val:
            best_val = val_mel
            save_checkpoint(str(ckpt_dir / "best_generator.pt"), {
                "generator": G.state_dict(),
                "best_val_mel_l1": best_val,
                "global_step": global_step,
                "config": OmegaConf.to_container(cfg, resolve=True),
            })
            print(f"[INFO] New best checkpoint saved: val_mel_l1={best_val:.6f}")

    if logger is not None:
        logger.finish()


if __name__ == "__main__":
    main()
