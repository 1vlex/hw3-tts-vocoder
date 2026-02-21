import argparse
import os
from pathlib import Path
from urllib.parse import urlparse, unquote

import requests


def _filename_from_url(url: str) -> str:
    """
    Пытаемся взять имя файла из URL.
    Если не получилось - используем best_generator.pt
    """
    try:
        parsed = urlparse(url)
        name = Path(unquote(parsed.path)).name
        if not name:
            return "best_generator.pt"
        return name
    except Exception:
        return "best_generator.pt"


def _resolve_output_path(out_arg: str, url: str) -> Path:
    """
    Поддерживает 2 режима:
    1) --out checkpoints_test                -> это директория, файл будет checkpoints_test/best_generator.pt
    2) --out checkpoints/best_generator.pt   -> это полный путь к файлу
    """
    out = Path(out_arg)

    # Если путь уже существует и это директория
    if out.exists() and out.is_dir():
        out.mkdir(parents=True, exist_ok=True)
        return out / _filename_from_url(url)

    # Если нет суффикса (например "checkpoints_test") - считаем, что это директория
    if out.suffix == "":
        out.mkdir(parents=True, exist_ok=True)
        return out / _filename_from_url(url)

    # Иначе считаем, что это путь к файлу
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def main():
    parser = argparse.ArgumentParser(description="Скачивание чекпоинта вокодера")
    parser.add_argument(
        "--url",
        type=str,
        default=os.getenv("VOCODER_CKPT_URL", ""),
        help="Публичная прямая ссылка на .pt файл",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="checkpoints/best_generator.pt",
        help=(
            "Либо путь к файлу (например checkpoints/best_generator.pt), "
            "либо директория (например checkpoints)"
        ),
    )
    args = parser.parse_args()

    if not args.url:
        raise SystemExit(
            "Укажи публичную ссылку на чекпоинт: "
            "python scripts/download_checkpoints.py --url <DIRECT_URL>"
        )

    out_path = _resolve_output_path(args.out, args.url)

    with requests.get(args.url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"[OK] Downloaded checkpoint to {out_path} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()