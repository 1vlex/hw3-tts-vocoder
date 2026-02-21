# HW3 TTS Vocoder (HiFi-GAN baseline)

В этом репозитории реализован вокодер типа HiFi-GAN, обученный с нуля на датасете [RUSLAN](https://ruslan-corpus.github.io/).

Модель преобразует mel-спектрограммы в аудиосигнал.

Важно:
- предобученные вокодеры не использовались
- обучение выполняется с нуля

## Структура проекта

- [`train.py`](train.py) - обучение модели
- [`synthesize.py`](synthesize.py) - инференс (resynthesis)
- [`full_tts.py`](full_tts.py) - Full TTS анализ (Windows-friendly proxy через внешнюю TTS)
- [`src/`](src)
  - [`datasets/`](src/datasets)
    - [`ruslan_dataset.py`](src/datasets/ruslan_dataset.py) - датасет для обучения
    - [`custom_dir_dataset.py`](src/datasets/custom_dir_dataset.py) - датасет для инференса на директории
  - [`models/`](src/models) - архитектуры генератора и дискриминаторов
  - [`utils/`](src/utils) - функции аудио, mel и логирования
- [`src/configs/`](src/configs) - конфиги Hydra
- [`scripts/`](scripts)
  - [`download_checkpoints.py`](scripts/download_checkpoints.py) - скачивание чекпоинтов
  - [`make_dummy_custom_dir_dataset.py`](scripts/make_dummy_custom_dir_dataset.py) - генерация тестового датасета
  - [`plot_report_graphs.py`](scripts/plot_report_graphs.py) - графики для отчета
- [`demo/demo_colab.ipynb`](demo/demo_colab.ipynb) - Colab demo
- [`REPORT_TEMPLATE.md`](REPORT_TEMPLATE.md) - отчет

## Требования

Протестировано на:
- Windows
- Python 3.10
- GPU RTX 5070 Ti
- PyTorch + torchaudio

## Установка

```powershell
cd E:\hww3
py -3.10 -m pip install -r requirements.txt
```

## Датасет

### Обучающий датасет RUSLAN

Поместите `.wav` файлы в:

```text
data/ruslan_wavs/
```

Если в архиве есть файлы вида `._*.wav`, удалите их:

```powershell
Get-ChildItem .\data\ruslan_wavs -Recurse -Filter '._*.wav' | Remove-Item -Force
```

### MOS датасет для оценки

Структура:

```text
data/mos_eval/
  audio/
    1.wav
    2.wav
    3.wav
  transcriptions/
    1.txt
    2.txt
    3.txt
```

## Логирование CometML

Задайте API ключ:

```powershell
setx COMET_API_KEY "ВАШ_КЛЮЧ"
```

Проверка в PowerShell:

```powershell
echo $env:COMET_API_KEY
```

## Обучение

### Быстрый тест

```powershell
py -3.10 train.py `
  data.train_audio_dir=.\data\ruslan_wavs `
  data.val_audio_dir=.\data\ruslan_wavs `
  train.epochs=1 `
  train.batch_size=2 `
  train.num_workers=0 `
  logging.use_comet=false
```

### Основное обучение

```powershell
py -3.10 train.py `
  data.train_audio_dir=.\data\ruslan_wavs `
  data.val_audio_dir=.\data\ruslan_wavs `
  train.epochs=10 `
  train.batch_size=12 `
  train.num_workers=4 `
  train.segment_size=16384 `
  train.val_num_batches=2 `
  logging.use_comet=true `
  logging.run_name=hifigan-baseline `
  logging.log_every_steps=100 `
  logging.log_audio_every_steps=2000 `
  checkpoint.save_every_steps=2000
```

Чекпоинты сохраняются в:

```text
checkpoints/
```

## Скачивание чекпоинта (для инференса)

Пример через скрипт:

```powershell
py -3.10 scripts\download_checkpoints.py `
  --out_dir checkpoints `
  --gdrive_file_id YOUR_FILE_ID `
  --filename best_generator.pt
```

Если используете прямую ссылку:

```powershell
py -3.10 scripts\download_checkpoints.py `
  --out_dir checkpoints `
  --url "https://your-host/path/to/best_generator.pt" `
  --filename best_generator.pt
```

## Инференс (resynthesis)

### Проверка пайплайна на dummy dataset

```powershell
py -3.10 scripts\make_dummy_custom_dir_dataset.py --out examples\custom_dir_dataset_example
```

```powershell
py -3.10 synthesize.py `
  infer.checkpoint_path=.\checkpoints\best_generator.pt `
  infer.input_dir=.\examples\custom_dir_dataset_example `
  infer.output_dir=.\outputs\test `
  infer.mode=resynthesis
```

### MOS инференс

```powershell
py -3.10 synthesize.py `
  infer.checkpoint_path=.\checkpoints\best_generator.pt `
  infer.input_dir=.\data\mos_eval `
  infer.output_dir=.\outputs\mos_eval `
  infer.mode=resynthesis
```

Результат:

```text
outputs/mos_eval/
```

## Full TTS анализ (Windows-friendly proxy)

Файл [`full_tts.py`](full_tts.py) запускает эксперимент для анализа поведения вокодера на синтетических признаках.

Что делает `full_tts.py`:
- берет первые 3 текста из `data/metadata_RUSLAN_22200.csv`
- использует 3 MOS-предложения
- добавляет 3 пользовательские фразы
- генерирует внешнее русское TTS-аудио (MMS через Hugging Face / Transformers)
- извлекает mel тем же кодом проекта
- прогоняет mel через наш HiFi-GAN вокодер

### Запуск

```powershell
py -3.10 full_tts.py
```

Ожидаемые выходы (пример):

```text
outputs/full_tts_mms_proxy/
  00_ruslan_mms.wav
  00_ruslan_mms_to_ourvocoder.wav
  ...
```

Примечание:
- это Windows-friendly proxy для Full TTS анализа
- используется внешний end-to-end TTS, после которого выполняется повторная вокодеризация через mel

## Построение графиков для отчета

Скрипт [`scripts/plot_report_graphs.py`](scripts/plot_report_graphs.py) умеет:
- строить waveform / log-mel сравнения
- настраивать заголовки и подписи осей
- пропускать агрегированные bar/radar графики (`--skip_aggregate`)

### Пример: RUSLAN (MMS vs наш вокодер)

```powershell
py -3.10 scripts\plot_report_graphs.py `
  --skip_aggregate `
  --out_dir report\figures\full_tts_mms `
  --orig_wav outputs\full_tts_mms_proxy\00_ruslan_mms.wav `
  --gen_wav outputs\full_tts_mms_proxy\00_ruslan_mms_to_ourvocoder.wav `
  --prefix ruslan_fulltts_mms_00 `
  --orig_label "MMS external TTS" `
  --gen_label "Our vocoder (from MMS-derived mel)"
```

### Пример: MOS (MMS vs наш вокодер)

```powershell
py -3.10 scripts\plot_report_graphs.py `
  --skip_aggregate `
  --out_dir report\figures\full_tts_mms `
  --orig_wav outputs\full_tts_mms_proxy\03_mos_mms.wav `
  --gen_wav outputs\full_tts_mms_proxy\03_mos_mms_to_ourvocoder.wav `
  --prefix mos_fulltts_mms_03 `
  --orig_label "MMS external TTS" `
  --gen_label "Our vocoder (from MMS-derived mel)"
```

### Пример: Custom (MMS vs наш вокодер)

```powershell
py -3.10 scripts\plot_report_graphs.py `
  --skip_aggregate `
  --out_dir report\figures\full_tts_mms `
  --orig_wav outputs\full_tts_mms_proxy\06_custom_mms.wav `
  --gen_wav outputs\full_tts_mms_proxy\06_custom_mms_to_ourvocoder.wav `
  --prefix custom_fulltts_mms_06 `
  --orig_label "MMS external TTS" `
  --gen_label "Our vocoder (from MMS-derived mel)"
```

## Dемо (Colab)

В репозитории приложен ноутбук:
- [`demo/demo_colab.ipynb`](demo/demo_colab.ipynb)

В нем показаны:
- установка зависимостей
- скачивание чекпоинта
- запуск `synthesize.py`
- прослушивание результатов

## Отчет

Подробный отчет и анализ:
- [`REPORT_TEMPLATE.md`](REPORT_TEMPLATE.md)
