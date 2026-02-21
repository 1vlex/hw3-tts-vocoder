# HiFi-GAN Vocoder (HW3)

Реализация нейровокодера HiFi-GAN, обученного с нуля на датасете RUSLAN.

## Возможности
- обучение вокодера
- resynthesis инференс
- Full TTS анализ
- графики сравнения сигналов

## Установка
pip install -r requirements.txt

## Обучение
python train.py

## Инференс
python synthesize.py

## Full TTS анализ
python full_tts.py

## Построение графиков
python scripts/plot_report_graphs.py --orig_wav A.wav --gen_wav B.wav

## Структура
src/ - код модели  
scripts/ - вспомогательные скрипты  
report/ - отчёт и графики  

## Датасет
Используется RUSLAN corpus

## Логи
Используется CometML

## Colab demo
см. demo/demo_colab.ipynb
