import torch
import torchaudio
import soundfile as sf
from pathlib import Path
from transformers import VitsModel, AutoTokenizer

from src.models.hifigan_generator import HiFiGANGenerator
from src.utils.checkpoint import load_checkpoint
from src.utils.mel import MelSpectrogram, MelSpectrogramConfig


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SR = 22050

ruslan_csv = Path("data/metadata_RUSLAN_22200.csv")

ruslan_texts = []
with open(ruslan_csv, encoding="utf-8") as f:
    for _ in range(3):
        line = f.readline().strip()
        if not line:
            continue
        parts = line.split("|", maxsplit=1)
        text = parts[1] if len(parts) == 2 else line
        ruslan_texts.append(text)

mos_texts = [
    "Ожидал увидеть тут толпы игроков, жаждущих меня убить, но никому моя голова почему-то не потребовалась",
    "Уверяю вас, сэр, большую часть своей жизни я был глубоко убежден, что любой мир состоит из чего угодно, только не из добрых людей",
    "Наш самый суровый священник был абсолютно никакой, в хлам, как распоследний питерский сантехник в канун дня праздника Парижской коммуны",
]

custom_texts = [
    "Привет мир",
    "Это тест полной системы синтеза речи",
    "Проверка работы вокодера",
]

texts = (
    [("ruslan", t) for t in ruslan_texts] +
    [("mos", t) for t in mos_texts] +
    [("custom", t) for t in custom_texts]
)

print("\nLoaded texts:")
for tag, t in texts:
    print(f"{tag}: {t}")

print("\nLoading MMS Russian TTS model...")
mms_model = VitsModel.from_pretrained("facebook/mms-tts-rus").to(DEVICE).eval()
mms_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-rus")
mms_sr = int(mms_model.config.sampling_rate)
print(f"MMS sampling rate: {mms_sr}")


mel_cfg = MelSpectrogramConfig(
    sr=TARGET_SR,
    win_length=1024,
    hop_length=256,
    n_fft=1024,
    f_min=0,
    f_max=11025,
    n_mels=80,
    power=2.0,
)
mel_extractor = MelSpectrogram(mel_cfg).to(DEVICE).eval()


ckpt = load_checkpoint("checkpoints/best_generator.pt", map_location=DEVICE)

gen = HiFiGANGenerator(
    in_channels=80,
    upsample_rates=(8, 8, 2, 2),
    upsample_kernel_sizes=(16, 16, 4, 4),
    upsample_initial_channel=512,
).to(DEVICE)

gen.load_state_dict(ckpt["generator"])
gen.eval()


out_dir = Path("outputs/full_tts_mms_proxy")
out_dir.mkdir(parents=True, exist_ok=True)

with torch.no_grad():
    for i, (tag, text) in enumerate(texts):
        # --- text -> wav (MMS) ---
        inputs = mms_tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        wav_mms = mms_model(**inputs).waveform  # [B, T] 
        if wav_mms.dim() == 1:
            wav_mms = wav_mms.unsqueeze(0)

        wav_mms = wav_mms.float()

        if mms_sr != TARGET_SR:
            wav_22 = torchaudio.functional.resample(wav_mms, orig_freq=mms_sr, new_freq=TARGET_SR)
        else:
            wav_22 = wav_mms

        wav_22 = wav_22.clamp(-1.0, 1.0)

        mel = mel_extractor(wav_22)

        wav_gen = gen(mel).squeeze(1)  # [B, T]
        wav_gen = wav_gen.clamp(-1.0, 1.0).cpu()

        wav_mms_save = wav_22.squeeze(0).detach().cpu().numpy()
        wav_gen_save = wav_gen.squeeze(0).detach().numpy()

        base = f"{i:02d}_{tag}"

        path_mms = out_dir / f"{base}_mms.wav"
        path_gen = out_dir / f"{base}_mms_to_ourvocoder.wav"

        sf.write(path_mms, wav_mms_save, TARGET_SR)
        sf.write(path_gen, wav_gen_save, TARGET_SR)

        print("saved:", path_mms)
        print("saved:", path_gen)

print("\nDONE")