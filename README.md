# Talc Analysis

Единый inference-модуль для анализа микроскопических изображений руды:

1. сегментация грубой области нейросетевой моделью;
2. CV-уточнение тёмных талькоподобных областей;
3. расчёт площадей, контуров, confidence и времени этапов;
4. классификация изображения;
5. экспорт результата для приложения.

Модуль принимает бинарные checkpoint DeepLabV3 и SegFormer в его текущем формате.

## Установка

```bash
python -m pip install -e .
```

Checkpoint не входит в репозиторий и передаётся отдельным путём.

## Python API

```python
from talc_analysis import SegmentationMode, TalcAnalyzer

analyzer = TalcAnalyzer.from_files(
    checkpoint_path=".../checkpoint.pt",
    config_path="talc_analysis/default_config.yaml",  # можно не указывать
)

result = analyzer.analyze_path(
    "examples/sample.jpg",
    mode=SegmentationMode.OVERLAP,
)
result.save("outputs/sample")
```

Для интеграции с интерфейсом можно передавать RGB-массив:

```python
result = analyzer.analyze(image_rgb, mode="no_overlap")

coarse_mask = result.segmentation_mask
refined_mask = result.refined_talc_mask
statistics = result.statistics
```

Экземпляр `TalcAnalyzer` следует создавать один раз: модель загружается при
`from_files`, а затем используется для любого числа изображений.

## CLI

Один файл:

```bash
python solution.py \
  --checkpoint .../checkpoint.pt \
  --input /path/to/image.jpg \
  --output outputs \
  --mode overlap
```

Каталог:

```bash
python solution.py \
  --checkpoint /path/to/checkponit.pt \
  --input /path/to/images \
  --output outputs \
  --mode no_overlap
```

Повторная запись существующего результата разрешается только с `--overwrite`.
Поддерживаются JPG, PNG, BMP и TIFF.

## Режимы сегментации

- `overlap`: тайлы перекрываются, каждый прогноз даёт один голос на пиксель.
  Используется строгое большинство; ничья разрешается средней вероятностью.
- `no_overlap`: шаг равен размеру тайла, поэтому каждый пиксель получает ровно
  один голос. Неполные крайние тайлы дополняются до размера модели.

Размер тайла, размер входа модели и threshold читаются из checkpoint. Доля
перекрытия и CV-параметры задаются в `default_config.yaml`.

## Классификация

Доля талька считается по уточнённой маске относительно всех пикселей:

- строго больше 10% — `talc_bearing`;
- 10% и меньше — `non_talc_bearing`.

## Результат

```text
output/
  run.json
  image_name/
    segmentation_mask.png
    refined_talc_mask.png
    overlay.png
    confidence_maps.npz
    result.json
```

`confidence_maps.npz` содержит:

- `segmentation_confidence` — среднюю вероятность модели;
- `cv_confidence` — карту уверенности CV-уточнения;
- `positive_votes` — число голосов за грубую область;
- `vote_count` — общее число голосов.

`result.json` содержит контуры и компоненты обеих масок, площади, проценты,
агрегаты confidence, итоговый класс и время этапов. Все координаты и площади
заданы в пикселях.