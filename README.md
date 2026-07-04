# PyTorchi: Ore analyzer

Локальное web-приложение для анализа микроскопических изображений руды. Один
pipeline объединяет сегментацию и CV-уточнение талька с классификатором
`ordinary` / `difficult` для изображений, где талька не больше 10%.

## Возможности

- загрузка одного изображения, нескольких файлов или папки через drag-and-drop;
- два режима сегментации: с перекрытием и без перекрытия;
- progress bar загрузки, сегментации, CV, классификации и экспорта;
- итоговые классы `talc_bearing`, `ordinary`, `difficult`;
- просмотр большого снимка с zoom/pan;
- независимые слои грубого контура, уточнённого талька и сульфидных CV/SAM-масок
  с настройкой opacity;
- каждый снимок хранит собственные пороги сегментации, CV, доли талька и
  ordinary/difficult;
- статистика площадей, количества сульфидных компонент, confidence карт
  сегментации/CV и времени этапов;
- последовательная очередь для пакетной обработки без дублирования моделей в RAM;
- адаптивный интерфейс и режим `prefers-reduced-motion`.

## Запуск на другом ПК

Требования:

- Docker Desktop 4.35+ или Docker Engine 27+ с Compose v2;
- минимум 8 ГБ RAM, для больших изображений рекомендуется 16 ГБ;
- два обязательных совместимых checkpoint-файла: для талька и классификатора
  сульфидного класса; опционально checkpoint MobileSAM для уточнения
  сульфидных CV-масок.

Inference-модули `talc_analysis` и `ore_classifier` включены в backend как
исходный код. Соседние рабочие копии `talk_combined` и `talk_sulfid` для
сборки не требуются.

Веса моделей и пользовательские изображения намеренно не входят в Docker image.

```bash
cp .env.example .env
```

Укажите в `.env` абсолютный путь к каталогу с checkpoint-файлами и их имена.
На Windows с Docker Desktop используйте путь вида `C:/models`. Приложение
может запуститься с пустым каталогом, но анализ будет недоступен до появления
совместимых весов. Затем:

```bash
docker compose config --quiet
docker compose up --build -d
```

Откройте [http://localhost:8080](http://localhost:8080). Проверка состояния:

```bash
curl http://localhost:8080/health
docker compose ps
docker compose logs -f
```

Остановка:

```bash
docker compose down
```

Обработанные задания находятся в именованном volume `talk-app_jobs_data` и
сохраняются между перезапусками и `docker compose down`. Для удаления данных:

```bash
docker compose down -v
```

### История и дозагрузка

- Backend хранит последние 50 завершённых снимков вместе с manifest,
  загруженными файлами и артефактами; активные снимки не удаляются лимитом
  истории.
- История восстанавливается из `jobs_data` после перезапуска контейнеров.
  Незавершённое при аварийной остановке задание помечается как failed, а уже
  записанные артефакты остаются доступными.
- Дозагрузка добавляет изображения в существующее задание. В очередь ставятся
  только новые изображения; готовые результаты не пересчитываются.
- Изменение параметров отдельного изображения запускает пересчёт только
  затронутых и последующих стадий этого изображения.
- `docker compose down -v` безвозвратно удаляет историю, исходные загрузки и
  результаты. Обычный `docker compose down` данные сохраняет.

## Режим разработки

Dev Compose монтирует исходники backend, запускает Uvicorn с reload и Vite HMR,
сохраняя единый адрес приложения:

```bash
docker compose -f docker-compose.dev.yml up --build
```

Изменения `frontend/` и `backend/` подхватываются без production-сборки.

## Инфраструктурная проверка

Без сборки тяжёлых ML images можно проверить production, dev и GPU Compose,
обязательные vendored-модули и whitespace:

```bash
make check
```

Полная проверка сборки требует запущенного Docker daemon и доступа к npm/PyPI:

```bash
docker compose build
docker compose up -d
curl --fail http://localhost:8080/health
```

## GPU (необязательно)

Production по умолчанию использует CPU wheels PyTorch, поэтому запускается на
ПК без NVIDIA GPU. Для Linux с NVIDIA Container Toolkit и совместимым
драйвером доступна CUDA 12.1 сборка:

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.gpu.yml \
  up --build -d
```

GPU override не предназначен для macOS/Windows без корректно настроенного
NVIDIA passthrough.
