# HW3 TTS Vocoder (HiFi-GAN baseline)

В этом репозитории реализован вокодер типа HiFi-GAN, обученный с нуля на датасете RUSLAN.  
Модель преобразует mel-спектрограммы в аудиосигнал.

Важно: предобученные вокодеры не использовались. Обучение выполняется с нуля.

---

## Структура проекта

- train.py - обучение модели
- synthesize.py - инференс
- src/
  - datasets/
    - ruslan_dataset.py - датасет для обучения
    - custom_dir_dataset.py - датасет для инференса на директории
  - models/ - архитектуры генератора и дискриминаторов
  - utils/ - функции аудио и логирования
- src/configs/ - конфиги Hydra
- scripts/
  - download_checkpoints.py - скачивание чекпоинтов
  - make_dummy_custom_dir_dataset.py - генерация тестового датасета

---

## Требования

Протестировано на:

- Windows
- Python 3.10
- GPU RTX 5070 Ti
- PyTorch + torchaudio

---

## Установка
```
cd E:\hww3
py -3.10 -m pip install -r requirements.txt
```

---

## Датасет

### Обучающий датасет RUSLAN

Поместите wav файлы в:

data/ruslan_wavs/

Если в архиве есть файлы вида ._*.wav, удалите их:
```
Get-ChildItem .\data\ruslan_wavs -Recurse -Filter '._*.wav' | Remove-Item -Force
```
---

### MOS датасет для оценки

Структура:

data/mos_eval/
  audio/
    1.wav
    2.wav
    3.wav
  transcriptions/
    1.txt
    2.txt
    3.txt

---

## Логирование CometML

Задайте API ключ:
```
setx COMET_API_KEY "ВАШ_КЛЮЧ"
```
Проверка:
```
echo $env:COMET_API_KEY
```
---

## Обучение

### Быстрый тест
```
py -3.10 train.py `
  data.train_audio_dir=.\data\ruslan_wavs `
  data.val_audio_dir=.\data\ruslan_wavs `
  train.epochs=1 `
  train.batch_size=2 `
  train.num_workers=0 `
  logging.use_comet=false
```
---

### Основное обучение
```
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

checkpoints/

---

## Инференс

### Проверка пайплайна
```
py -3.10 scripts\make_dummy_custom_dir_dataset.py --out examples\custom_dir_dataset_example
```
```
py -3.10 synthesize.py `
 infer.checkpoint_path=.\checkpoints\best_generator.pt `
 infer.input_dir=.\examples\custom_dir_dataset_example `
 infer.output_dir=.\outputs\test `
 infer.mode=resynthesis
```
---

### MOS инференс
```
py -3.10 synthesize.py `
 infer.checkpoint_path=.\checkpoints\best_generator.pt `
 infer.input_dir=.\data\mos_eval `
 infer.output_dir=.\outputs\mos_eval `
 infer.mode=resynthesis
```
Результат:

outputs/mos_eval/

---

## Скачивание чекпоинта
```
py -3.10 scripts\download_checkpoints.py --url "https://drive.google.com/uc?export=download&id=197YvUjKbh2pW9jAg6XH4PDw4Lu1gRS3R" --out checkpoints\best_generator.pt
```
[тык](https://drive.google.com/file/d/197YvUjKbh2pW9jAg6XH4PDw4Lu1gRS3R/view?usp=sharing) - ссылка на best_generator.pt
---

## Важно

В git не добавляются:

- датасеты
- wav файлы
- чекпоинты

---

## Назначение проекта

Учебная реализация вокодера для задания по синтезу речи.

