from pathlib import Path
import argparse
import math

import numpy as np
import soundfile as sf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="examples/custom_dir_dataset_example")
    ap.add_argument("--sr", type=int, default=22050)
    args = ap.parse_args()

    root = Path(args.out)
    (root / "audio").mkdir(parents=True, exist_ok=True)
    (root / "transcriptions").mkdir(parents=True, exist_ok=True)

    texts = {
        "file1": "Привет, это тестовый файл.",
        "file2": "Это заглушка для проверки synthesize.py.",
    }
    for i, (stem, text) in enumerate(texts.items()):
        dur = 1.5 + i * 0.5
        t = np.linspace(0, dur, int(args.sr * dur), endpoint=False)
        freq = 220 + i * 110
        wav = 0.1 * np.sin(2 * math.pi * freq * t).astype(np.float32)
        sf.write(str(root / "audio" / f"{stem}.wav"), wav, args.sr)
        (root / "transcriptions" / f"{stem}.txt").write_text(text, encoding="utf-8")

    print(f"[OK] Dummy dataset created at: {root}")


if __name__ == "__main__":
    main()
