# Контекст проекта для Codex

## Назначение

Проект посвящён автоматической классификации руды по микроскопическим OM-изображениям полированных шлифов.

Важное ограничение: исходное описание пайплайна ниже является проектным контекстом, а не текущим заданием на немедленную реализацию. Реализацию отдельных этапов начинать только по явной просьбе пользователя.

## Текущее состояние репозитория

- Рабочая папка: `/Users/macbook/Documents/talk`.
- На старте был только пустой файл `solution.py`.
- В `data/img` лежат 2 исходных OM-изображения `.JPG`; в `data/labels` лежат соответствующие `.txt`-разметки полигонов грубой зоны оталькования в YOLO-segmentation формате. `data/classes.txt` содержит класс талька. Обученные модели в репозитории пока не обнаружены.
- Созданное conda-окружение: `talk`.
- Воспроизводимый spec окружения: `environment.yml`.
- Машинно-читаемый проектный контекст и конфиг: `project.toml`.
- Реализован пакет `cv_analysis` для CV-алгоритмов текущей ветки.
- Основной модуль текущей ветки: `cv_analysis.post_segformer`.
- Дефолтная YAML-конфигурация CV post-processing: `cv_analysis/post_segformer.yaml`.
- Реализован модуль `cv_analysis.sulfide_candidates` для простой быстрой бинарной CV-сегментации светлых сульфидоподобных включений без обучаемых моделей; дефолтная конфигурация: `cv_analysis/sulfide_candidates.yaml`.
- Добавлена папка `test` с конфигом `test/photometric_tests.yaml` и раннером `test/run_photometric_tests.py` для визуальной проверки устойчивости post-processing к фотометрическим преобразованиям и деградациям съёмки. Сгенерированные изображения сохраняются в `test/outputs` и не предназначены для коммита.

## Проектный контекст

Целевой пайплайн должен поддерживать:

1. Загрузку крупных OM-изображений и обработку тайлами.
2. Сегментацию сульфидных включений классическими CV-методами, потому что отдельной разметки сульфидов пока нет.
3. Классификацию найденных сульфидных областей на обычные и тонкие срастания по геометрическим, текстурным и embedding-признакам. При отсутствии object-level разметки допустимы weakly supervised и MIL-подходы по классам целых изображений.
4. Сегментацию грубых зон оталькования с помощью SegFormer. Разметка трактуется как грубые обведённые области, а не точные маски отдельных зёрен.
5. Поиск локально тёмных рассеянных областей внутри зоны оталькования как кандидатов в отдельные тальковые зёрна.
6. Классификацию всего изображения как оталькованной, рядовой или труднообогатимой руды на основе масок, статистик и признаков RGB-изображения.
7. Интерпретируемый результат: маски сульфидов, обычных и тонких срастаний, зоны оталькования, тальковых кандидатов, итоговый класс и overlay.

## Ограничения данных

- Ожидаемый объём размеченных изображений малый: около 43 изображений.
- Нужны pretrained-модели, patch-based обучение и аккуратная аугментация.
- Split должен выполняться по исходным образцам, а не по тайлам, чтобы не было утечки между train/val/test.
- Грубую разметку оталькования нельзя трактовать как идеальные pixel-perfect маски.
- Следует минимизировать зависимость от ручной object-level аннотации.

## Фокус текущей ветки

Данная ветка направлена на разработку CV-алгоритмов, в первую очередь post-processing предсказанной SegFormer зоны оталькования.

SegFormer используется только как источник грубой области поиска. Внутри этой области CV-методами нужно локализовать более мелкие тёмные талькоподобные участки и сформировать отдельную маску кандидатов в тальковую фазу.

Ожидаемые идеи для CV post-processing:

- локальная темнота относительно окружения;
- multi-scale black-hat;
- multi-scale DoG;
- морфологические операции;
- connected components analysis.

Критерий качества для этого этапа смещён в сторону высокого precision. Лучше пропустить сомнительные участки, чем добавлять шум. Нужно отбрасывать шум, тени, трещины, царапины и слишком вытянутые компоненты.

Реализованный публичный интерфейс:

```python
from cv_analysis.post_segformer import TalcCVPipeline

pipeline = TalcCVPipeline.from_yaml("cv_analysis/post_segformer.yaml")
mask = pipeline(image_rgb, segformer_mask)
```

`image_rgb` должен быть массивом `H x W x 3`. `segformer_mask` должен быть маской грубой зоны оталькования формы `H x W`, полученной от SegFormer; поддерживаются bool, целочисленные маски `0/1` и float-маски, где порог применяется как `> 0.5`. Модуль не реализует SegFormer и не связан с конкретной моделью; SegFormer используется только как источник области поиска. Возвращаемая маска имеет форму `H x W` и dtype `uint8` со значениями `0/1`.

В `cv_analysis.post_segformer` реализованы последовательные обработчики: подготовка изображения, локальная темнота, multi-scale black-hat, robust-нормализация признаков, fusion confidence, hysteresis через connected components, консервативная морфологическая очистка и фильтрация компонент.

В YAML добавлены параметры robust-нормализации для `multiscale_blackhat`, потому что инструкция требует нормализовать каждый scale до объединения и держать численные параметры в YAML.

Текущая логика fusion/hysteresis/filtering: `blackhat_persistence` используется как уже нормированный признак и не проходит повторную `normalize_features`; `confidence_fusion` объединяет weighted mean и strongest primary response. `HysteresisSegmentation` может создавать seed не только по общей confidence, но и по одному сильному primary-признаку через `strong_response_threshold`, а также сохраняет `grow_mask` в state. `ComponentFilter` имеет `shape_filter_min_area` и не применяет elongation/solidity-фильтры к компонентам меньше этого порога; финальная фильтрация настроена мягко, чтобы не срезать мелкие кандидаты после hysteresis.

Фотометрический test harness запускается командой `conda run -n talk python test/run_photometric_tests.py`. Он берёт изображения из `data/img`, строит грубую `segformer_mask` из YOLO-segmentation разметки `data/labels`, применяет конфигурируемые преобразования, прогоняет `TalcCVPipeline` и для каждого кейса сохраняет 5 PNG: оригинал, преобразованное изображение, overlay SegFormer-маски, overlay результата фильтрации и triptych из преобразованного изображения, SegFormer overlay и результата.

Модуль `cv_analysis.sulfide_candidates` теперь минимален: публичный интерфейс `segment_sulfides(image, config, sam_refiner=None)` принимает одно RGB/BGR-изображение и возвращает бинарную `uint8`-маску сульфидоподобных светлых включений со значениями `0/255`. Базовый CV-алгоритм использует Lab L-channel, небольшой Gaussian blur, Otsu threshold с настраиваемым `thresholds.otsu_offset` и простую морфологическую очистку. Опциональный `MobileSamRefiner` с MobileSAM `vit_t` может уточнять границы нескольких крупнейших CV-компонент, вызывая `set_image()` один раз на изображение; текущий preset умеренно ограничивает `box_padding_ratio`, positive points и `max_area_ratio`, а подозрительные SAM-результаты отклоняются и оставляют исходную CV-компоненту. Визуальный прогон выполняется командой `conda run -n talk python -m cv_analysis.sulfide_candidates --input data/img --output test/outputs/sulfide_candidates` и сохраняет `cv_mask.png`, `refined_mask.png`, `cv_overlay.png`, `refined_overlay.png`.

## Рабочие правила

- Обновлять этот файл при появлении новых устойчивых фактов о структуре проекта, данных, окружении, соглашениях или пайплайне.
- Параллельно обновлять `project.toml`, когда меняются зависимости, пути, стадии пайплайна, статус данных, модели или принятые решения.
- Не придумывать структуру проекта заранее. Фиксировать только реально существующие файлы/папки или явно заданные пользователем соглашения.
- Не коммитить сырые микроскопические изображения, веса моделей, большие артефакты и приватные данные без явного указания пользователя.
- Для запуска Python-команд использовать:

```bash
conda run -n talk python ...
```

- Для установки новых зависимостей сначала обновлять `environment.yml` и `project.toml`, затем устанавливать пакет в окружение.
- Предпочитать воспроизводимые скрипты и конфиги вместо одноразовых notebook-only шагов.
- Не добавлять предполагаемые директории или архитектуру проекта без фактической необходимости или явного указания пользователя.

## Зависимости окружения

Базовый стек выбран под CV/MVP и дальнейшее обучение:

- Python 3.11
- NumPy, SciPy, pandas
- OpenCV через pip-пакет `opencv-python-headless`, scikit-image, Pillow, tifffile
- scikit-learn
- PyTorch, torchvision
- PyTorch CUDA runtime through `pytorch-cuda=12.1` when using the Windows/NVIDIA setup
- transformers, timm, segmentation-models-pytorch
- albumentations
- matplotlib, seaborn
- tqdm, rich
- tensorboard
- pytest, JupyterLab, ipykernel

Проверка импортов в окружении `talk` прошла успешно для NumPy, OpenCV, scikit-image, scikit-learn, PyTorch, torchvision, Transformers, albumentations, tifffile и segmentation-models-pytorch.

Текущие замечания по окружению:

- `torch.backends.mps.is_available()` возвращает `False`, то есть текущий conda PyTorch работает без MPS-ускорения.
- При импорте `torchvision` есть предупреждение о недоступной `torchvision.io` image extension из-за `libjpeg.9.dylib`. Базовые библиотеки чтения изображений (`cv2`, Pillow, tifffile) импортируются; не полагаться на `torchvision.io` без отдельной проверки.
- Для текущего conda PyTorch 2.3.x зависимость `transformers` зафиксирована как `transformers>=4.44,<5`, потому что `transformers` 5.x требует PyTorch >=2.4 и отключает PyTorch-интеграцию при импорте.
- `tensorboard` установлен в окружение `talk`; smoke-проверка импорта вернула версию `2.20.0`.
- На Windows-машине обнаружена NVIDIA GeForce RTX 3070 Laptop GPU через `nvidia-smi`; для окружения `talk` используется CUDA-сборка PyTorch через `pytorch-cuda=12.1`.

## Ore classifier MVP

- The binary ore-classifier dataset root is `data/dataset/dataset`.
- Actual top-level dataset folders are `set1`, `set2`, and `Панорамы`.
- Binary training uses only `set1` and `set2` class folders mapped to `ordinary` and `difficult`.
- Talc folders (`Оталькованные руды`, `оталькованные`) and `Панорамы` are excluded from train and validation.
- Observed source/class counts before exclusion: `set1/difficult=68`, `set1/ordinary=68`, `set1/Оталькованные руды=84`, `set2/difficult=418`, `set2/ordinary=497`, `set2/оталькованные=87`, `Панорамы/Панорамы=14`.
- Implemented standalone package: `ore_classifier`.
- Keep the ore classifier separate from `cv_analysis`: `cv_analysis` is for classical CV post-processing modules, while `ore_classifier` is a self-contained supervised image-classification pipeline.
- Default classifier config: `configs/classifier.yaml`.
- Dataset inspection CLI: `conda run -n talk python inspect_dataset.py --config configs/classifier.yaml`.
- Training CLI: `conda run -n talk python train_classifier.py --config configs/classifier.yaml`.
- Inference CLI: `conda run -n talk python predict.py --config configs/classifier.yaml --checkpoint path/to/best.ckpt --input path/to/image_or_directory`.
- The classifier indexes images with Unicode-safe IO, stores `dataset_source`, extracts configurable `group_id`, and assigns shared group IDs to exact duplicate file hashes to reduce grouped-CV leakage.
- The model uses one shared `timm` encoder for one global resized view plus local crops, with `head_only`, `last_stage`, `last_two_stages`, and `full_finetune` modes.
- Ablations are config-driven: `data.color_mode`, `data.num_local_crops`, `model.pooling`, `model.finetune_mode`, `model.backbone`, and `data.include_sources`.
- Training writes TensorBoard event logs when `tensorboard` is installed and `reports.tensorboard: true`.
- Best checkpoints are saved as full `best.ckpt` files and weights-only `best.pt` files when `reports.save_pt_weights: true`.
- Train/validation loops use a tqdm progress bar when `reports.progress_bar: true`.
- Inference reports detailed latency in milliseconds: image read/decode, view preprocessing, tensor transfer to device, model forward, postprocess, and total time.
- Training supports experiment names: `--experiment-name NAME` writes to `runs/ore_classifier/NAME/train_YYYYMMDD_HHMMSS` by default.
- Default classifier training uses a simple group-aware stratified train/val split: `data.split_strategy=train_val`, `data.val_fraction=0.3`. It saves explicit `train_samples.csv` and `val_samples.csv` lists. Cross-validation is opt-in via `data.split_strategy=cross_val`.
- Classifier leakage protection groups exact SHA duplicates and same-dHash perceptual duplicate buckets together before splitting. Every training run writes `split_audit.json` and aborts if train/val overlap exists by `file_path`, `rel_path`, `group_id`, `sha256`, or `dhash`.
- Generated training/inspection outputs are written under `runs/ore_classifier` and ignored by git.
