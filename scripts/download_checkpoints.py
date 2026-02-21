import argparse
import os
from pathlib import Path
from urllib.parse import urlparse, unquote

import requests


def _filename_from_url(url: str) -> str:
    """
    Пытаемся взять имя файла из URL.
    Если не получилось - используем best_generator.pt
    Для Google Drive ссылок с /view имя тоже может быть 'view',
    поэтому ниже есть доп. защита.
    """
    try:
        parsed = urlparse(url)
        name = Path(unquote(parsed.path)).name
        if not name or name.lower() == "view":
            return "best_generator.pt"
        return name
    except Exception:
        return "best_generator.pt"


def _resolve_output_path(out_arg: str, url: str) -> Path:
    """
    Поддерживает 2 режима:
    1) --out checkpoints_test                -> директория, файл будет checkpoints_test/best_generator.pt
    2) --out checkpoints/best_generator.pt   -> полный путь к файлу
    """
    out = Path(out_arg)

    # Если путь уже существует и это директория
    if out.exists() and out.is_dir():
        out.mkdir(parents=True, exist_ok=True)
        return out / _filename_from_url(url)

    # Если нет суффикса - считаем директорией
    if out.suffix == "":
        out.mkdir(parents=True, exist_ok=True)
        return out / _filename_from_url(url)

    # Иначе считаем, что это путь к файлу
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def _is_google_drive_url(url: str) -> bool:
    u = url.lower()
    return "drive.google.com" in u or "docs.google.com" in u


def _download_via_requests(url: str, out_path: Path) -> None:
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def _download_via_gdown(url: str, out_path: Path) -> None:
    try:
        import gdown
    except ImportError as e:
        raise SystemExit(
            "Для скачивания с Google Drive нужен gdown.\n"
            "Установи: py -3.10 -m pip install gdown"
        ) from e

    # fuzzy=True позволяет принимать обычную ссылку вида /file/d/.../view
    result = gdown.download(url, str(out_path), quiet=False, fuzzy=True)
    if result is None:
        raise RuntimeError("gdown не смог скачать файл (result=None)")


def main():
    parser = argparse.ArgumentParser(description="Скачивание чекпоинта вокодера")
    parser.add_argument(
        "--url",
        type=str,
        default=os.getenv("VOCODER_CKPT_URL", ""),
        help="Публичная ссылка на файл (Google Drive или прямой HTTP URL)",
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
            "Укажи ссылку на чекпоинт:\n"
            "python scripts/download_checkpoints.py --url <URL>"
        )

    out_path = _resolve_output_path(args.out, args.url)

    if _is_google_drive_url(args.url):
        _download_via_gdown(args.url, out_path)
    else:
        _download_via_requests(args.url, out_path)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    if size_mb < 1.0:
        raise SystemExit(
            f"Скачанный файл слишком маленький ({size_mb:.2f} MB). "
            "Скорее всего это не чекпоинт, а HTML-страница."
        )

    print(f"[OK] Downloaded checkpoint to {out_path} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()