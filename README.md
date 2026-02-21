# HW3 TTS Vocoder Starter (HiFi-GAN baseline, simplest version)

Это **минимальный рабочий стартовый проект** для домашнего задания по вокодерам.
Он сделан так, чтобы быстро начать и не получить 0 за неработающий инференс.

## Что уже есть
- `synthesize.py` (обязательный скрипт)
- `src/datasets/custom_dir_dataset.py` (обязательный датасет)
- базовый HiFi-GAN-like generator + discriminators
- обучение с W&B логированием
- Hydra-конфиги (`src/configs`)
- `scripts/download_checkpoints.py`
- шаблон отчета и demo notebook

## Важно для сдачи
Даже если сейчас работаешь через Google Drive/Colab, **для сдачи** по условию нужен публичный GitHub/GitLab репозиторий.

## Быстрый старт (Windows)
```bash
cd E:\hww3
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Формат для CustomDirDataset
```text
RootDir
├── audio/
│   ├── file1.wav
│   └── file2.wav
└── transcriptions/
    ├── file1.txt
    └── file2.txt
```

## Обязательная проверка (иначе риск 0)
Перед сдачей обязательно **запусти `synthesize.py`** на своей финальной модели и проверь, что wav-файлы сохраняются без ошибок.
