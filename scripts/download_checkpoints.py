import argparse
from pathlib import Path
import os

import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default=os.getenv("VOCODER_CKPT_URL", ""))
    parser.add_argument("--out", type=str, default="checkpoints/best_generator.pt")
    args = parser.parse_args()

    if not args.url:
        raise SystemExit(
            "Укажи публичную ссылку на чекпоинт: "
            "python scripts/download_checkpoints.py --url <DIRECT_URL>"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(args.url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    print(f"[OK] Downloaded checkpoint to {out_path}")


if __name__ == "__main__":
    main()
