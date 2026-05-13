
# Face Age Estimation

Модульная платформа для оценки возраста человека по фотографии лица.
Поддерживает произвольные комбинации backbone-архитектур и типов голов модели,
полностью конфигурируется через YAML без изменения кода. Результат доступен
через Telegram-бот или командную строку.

---

## Содержание

- [Возможности](#возможности)
- [Требования](#требования)
- [Установка](#установка)
- [Структура проекта](#структура-проекта)
- [Конфигурация](#конфигурация)
- [Подготовка данных](#подготовка-данных)
- [Обучение](#обучение)
- [Инференс](#инференс)
- [Telegram-бот](#telegram-бот)
- [Мониторинг экспериментов](#мониторинг-экспериментов)
- [Воспроизведение экспериментов](#воспроизведение-экспериментов)
- [Контроль качества кода](#контроль-качества-кода)
- [Частые проблемы](#частые-проблемы)

---

## Возможности

| Компонент          | Варианты                                                             |
| ------------------ | -------------------------------------------------------------------- |
| **Backbone**       | ResNet18, ResNet50, EfficientNet-B0                                  |
| **Голова модели**  | regression, dldl, coral, classification (Mean-Variance)              |
| **Оптимизатор**    | Adam, AdamW, SGD                                                     |
| **Планировщик LR** | cosine, step, reduce_on_plateau, warmup_cosine                       |
| **Fine-tuning**    | заморозка, полная разморозка, последние N блоков, произвольные блоки |
| **Интерфейс**      | Telegram-бот, пакетный CLI-инференс                                  |
| **Трекинг**        | MLflow — метрики, гиперпараметры, git-хеш коммита                    |
| **Данные**         | DVC + Google Drive, воспроизводимое разбиение без data leakage       |

---

## Требования

- Python 3.12
- [Poetry](https://python-poetry.org/)
- [Conda](https://docs.conda.io/)
- Git
- DVC
- GPU с CUDA или Apple Silicon (MPS) — опционально

---

## Установка

**1. Клонировать репозиторий**

```bash
git clone https://github.com/Max-Arkhipov/face-to-age.git
cd face-to-age
```

**2. Создать виртуальное окружение**

```bash
conda create -n face-to-age python=3.12
conda activate face-to-age
```

**3. Установить зависимости**

```bash
poetry install
```

**4. Установить pre-commit хуки**

```bash
pre-commit install
```

**5. Загрузить данные и чекпоинты**

```bash
dvc pull
```

Скачивает из Google Drive предобработанный датасет UTKFace (`data/train/`,
`data/val/`, `data/test/`) и сохранённые чекпоинты. Если данные уже загружены —
не перезапишет их.

**6. Запустить MLflow**

```bash
mlflow ui --host 127.0.0.1 --port 8080
```

Интерфейс доступен по адресу: http://127.0.0.1:8080

---

## Структура проекта

```
face-to-age/
├── bot/                          # Telegram-бот
│   ├── main.py                   # Точка входа, инициализация и запуск бота
│   ├── handlers.py               # Обработчики сообщений и фотографий
│   ├── keyboards.py              # Inline-клавиатуры (выбор модели и др.)
│   ├── state.py                  # FSM-состояния диалога
│   └── storage.py                # Хранилище сессий пользователей
├── configs/                      # Конфигурационные файлы Hydra
│   ├── config.yaml               # Основной конфиг (собирает все модули)
│   ├── prepare.yaml              # Конфиг подготовки датасета
│   ├── dataloader/
│   │   └── default.yaml          # Размер батча, num_workers
│   ├── dataset/
│   │   └── utkface.yaml          # Пути к выборкам UTKFace
│   ├── infer/
│   │   └── default.yaml          # Параметры инференса
│   ├── logger/
│   │   └── mlflow.yaml           # Tracking URI, experiment name
│   ├── model/                    # Конфиги экспериментов (backbone + head)
│   │   ├── resnet_18_reg_pt_full.yaml
│   │   ├── resnet_18_dldl_pt_ft.yaml
│   │   ├── resnet_18_coral_pt_full.yaml
│   │   ├── resnet_18_cemv_pt_ft.yaml
│   │   ├── resnet_50_reg_pt_ft_full.yaml
│   │   ├── resnet_50_dldl_pt_ft_full.yaml
│   │   ├── resnet_50_coral_pt_ft.yaml
│   │   ├── resnet_50_cemv_pt_ft_full.yaml
│   │   ├── efficientnet_b0_reg_pt_78.yaml
│   │   ├── efficientnet_b0_dldl_pt_78.yaml
│   │   ├── efficientnet_b0_coral_pt_78.yaml
│   │   ├── efficientnet_b0_cemv_pt_78.yaml
│   │   └── ...                   # Остальные конфиги экспериментов
│   ├── paths/
│   │   └── default.yaml          # Пути к данным и чекпоинтам
│   ├── preprocessing/
│   │   └── image.yaml            # Размер, нормализация, аугментации
│   └── training/
│       └── lightning.yaml        # max_epochs, accelerator
├── checkpoints/                  # Чекпоинты обученных моделей (DVC)
│   ├── resnet_18_reg_pt_full.ckpt
│   ├── resnet_18_dldl_pt_ft.ckpt
│   ├── resnet_18_coral_pt_full.ckpt
│   ├── resnet_18_cemv_pt_ft.ckpt
│   ├── resnet_50_reg_pt_ft_full.ckpt
│   ├── resnet_50_dldl_pt_ft_full.ckpt
│   ├── resnet_50_coral_pt_ft.ckpt
│   ├── resnet_50_cemv_pt_ft_full.ckpt
│   ├── efficientnet_b0_reg_pt_78.ckpt
│   ├── efficientnet_b0_dldl_pt_78.ckpt
│   ├── efficientnet_b0_coral_pt_78.ckpt
│   └── ...                       # Остальные чекпоинты
├── face_to_age/                  # Основной пакет
│   ├── model.py                  # AgeModel, SimpleRegressor, ConvRegressor
│   ├── lightning.py              # AgeRegressionModule — логика обучения
│   ├── data.py                   # UTKFaceDataModule, UTKFaceDataset
│   ├── finetuning.py             # BackboneFinetuning — поэтапная разморозка
│   ├── logger.py                 # Построение MLflow-логгера
│   ├── train.py                  # Скрипт обучения
│   ├── infer.py                  # Пакетный инференс → preds/predictions.csv
│   ├── predict.py                # Инференс одиночного изображения (CLI)
│   ├── prepare.py                # Предобработка датасета (детекция + выравнивание)
│   ├── telegram_predictor.py     # TelegramFacePredictor — инференс для бота
│   ├── commands.py               # CLI-команды (точки входа пакета)
│   └── utils.py                  # Вспомогательные функции (crop, alignment)
├── utils/
│   ├── dvc_utils.py              # Автозагрузка данных через DVC перед обучением
│   ├── predictions.py            # Постобработка и сохранение предсказаний в CSV
│   └── split_utkface.py          # Разбиение UTKFace на train/val/test
├── data/                         # Данные (управляются через DVC)
│   ├── train/                    # Обучающая выборка (14 465 изображений)
│   ├── val/                      # Валидационная выборка (4 822 изображения)
│   ├── test/                     # Тестовая выборка (4 821 изображение)
│   └── predict/                  # Изображения для инференса
├── preds/
│   └── predictions.csv           # Результаты пакетного инференса
├── .dvcignore                    # Настройка dvcignore
├── .gitignore                    # Настройка gitignore
├── .pre-commit-config            # Настройка прекоммитов
├── .env                          # Секреты
├── pyproject.toml                # Зависимости и метаданные пакета (Poetry)
├── poetry.lock                   # Зафиксированные версии зависимостей
└── README.md                     # Документация проекта
```

---

## Конфигурация

Все параметры задаются в `configs/config.yaml`. Смена модели, функции потерь или
стратегии обучения не требует изменения кода.

### Модель и голова

```yaml
model:
  name: age_model # age_model | simple_regressor | conv_regressor | conv_regressor_256
  backbone: resnet50 # resnet18 | resnet50 | efficientnet_b0
  head: coral # regression | dldl | coral | classification
  num_classes: 117
  pretrained: true
```

### Функция потерь

```yaml
loss:
  name: coral # mse | mae | dldl | coral | classification
  sigma: 2.0 # для dldl — ширина целевого распределения
  lambda_mean: 0.2 # для classification (Mean-Variance)
  lambda_var: 0.05 # для classification (Mean-Variance)
```

### Оптимизатор

```yaml
optimizer:
  name: adamw # adam | adamw | sgd
  lr: 1e-4
  weight_decay: 1e-4
```

### Планировщик LR

```yaml
scheduler:
  name: warmup_cosine # cosine | step | reduce_on_plateau | warmup_cosine
  warmup_epochs: 3
  warmup_start_factor: 0.1
  eta_min: 1e-6
  # для step:
  # step_size: 10
  # gamma: 0.1
  # для reduce_on_plateau:
  # factor: 0.5
  # patience: 5
```

### Fine-tuning — поэтапная разморозка backbone

```yaml
finetune:
  enabled: true
  unfreeze_epoch: 5 # эпоха разморозки
  backbone_lr: 5e-5 # отдельный LR для backbone

  # Варианты unfreeze_layers:
  unfreeze_layers: null # разморозить весь backbone
  # unfreeze_layers: 2   # последние 2 блока согласно архитектурному маппингу
  # unfreeze_layers:     # произвольные блоки
  #   - layer4
  #   - layer3
```

Поддерживаемые блоки по архитектурам:

| Backbone            | Блоки (от конца к началу)                      |
| ------------------- | ---------------------------------------------- |
| resnet18 / resnet50 | `layer4`, `layer3`, `layer2`, `layer1`         |
| efficientnet_b0     | `features.8` ... `features.1`                  |
| vit_b_16            | `encoder.layers.11` ... `encoder.layers.6`     |
| swin_t              | `layers.3`, `layers.2`, `layers.1`, `layers.0` |

### Данные и загрузчик

```yaml
dataset:
  name: utk_face
  train_data_dir: data/train
  val_data_dir: data/val
  test_data_dir: data/test
  predict_data_dir: data/predict

dataloader:
  train_batch_size: 64
  predict_batch_size: 64
  num_workers: 4
  persistent_workers: true

training:
  max_epochs: 30

preprocessing:
  image:
    size: [224, 224]
    input_size: [224, 224]
    input_extension: [0.1, 0.1]
    bbox_extension: [0.05, 0.05]
    normalize:
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
    train:
      horizontal_flip_p: 0.5
```

---

## Подготовка данных

Если требуется пересоздать предобработанные изображения из исходного датасета:

```bash
python -m face_to_age.prepare
```

Конвейер для каждого изображения:

1. Детекция лица через RetinaFace → 106 ключевых точек
2. Усреднение координат левого глаза (точки 35–41), правого глаза (89–95) и рта
   (52–71)
3. Аффинное выравнивание по ключевым точкам
4. Обрезка и масштабирование до 224×224
5. Сохранение в `data/{train,val,test}/`

Разбиение фиксировано: используется split 0 протокола [Paplham & Franc, 2023].
Один субъект не появляется одновременно в обучающей и тестовой выборках.

---

## Обучение

```bash
python -m face_to_age.train
```

При запуске:

- автоматически загружает данные через DVC, если они отсутствуют локально
- обучает модель согласно `configs/config.yaml`
- сохраняет лучший чекпоинт по `val_mae` в `checkpoints/`
- запускает тестирование на тестовой выборке по завершении
- логирует всё в MLflow: параметры, метрики по эпохам, git-хеш

Имя запуска в MLflow формируется автоматически: `{backbone}-{head}-lr:{lr}`.

**Примеры конфигураций:**

<details>
<summary>ResNet50 + CORAL + warmup (лучший результат)</summary>

```yaml
model:
  backbone: resnet50
  head: coral
  loss:
    name: coral
  finetune:
    enabled: true
    unfreeze_epoch: 5
    backbone_lr: 5e-5
    unfreeze_layers: null
  scheduler:
    name: warmup_cosine
    warmup_epochs: 3
```

</details>

<details>
<summary>ResNet18 + DLDL</summary>

```yaml
model:
  backbone: resnet18
  head: dldl
  loss:
    name: dldl
    sigma: 2.0
  finetune:
    enabled: true
    unfreeze_epoch: 5
    backbone_lr: 5e-5
    unfreeze_layers: 2
  scheduler:
    name: warmup_cosine
    warmup_epochs: 3
```

</details>

<details>
<summary>EfficientNet-B0 + Mean-Variance</summary>

```yaml
model:
  backbone: efficientnet_b0
  head: classification
  loss:
    name: classification
    lambda_mean: 0.2
    lambda_var: 0.05
  finetune:
    enabled: true
    unfreeze_epoch: 5
    backbone_lr: 5e-5
    unfreeze_layers: 2
  scheduler:
    name: warmup_cosine
    warmup_epochs: 3
```

</details>

---

## Инференс

**Пакетная обработка директории:**

```bash
python -m face_to_age.infer
```

Обрабатывает все изображения из `data/predict/`, сохраняет результаты в
`preds/predictions.csv`. Для каждого изображения возвращает предсказанный
возраст и оценку неопределённости σ (для вероятностных голов).

**Путь к чекпоинту задаётся в конфигурации:**

```yaml
paths:
  checkpoints_dir: checkpoints/
  checkpoint_name: best_model.ckpt
```

---

## Telegram-бот

**Запуск:**

```bash
python -m face_to_age.bot
```

**Использование:**

1. Найти бота в Telegram и запустить командой `/start`
2. Отправить фотографию лица
3. Получить ответ: выровненное изображение +
   `Предсказанный возраст: 34 ± 4 года`

**Рекомендации для лучшего результата:**

- фронтальное фото с хорошим освещением
- лицо занимает значительную часть кадра
- угол поворота головы не более 45°
- фотографии в очках и головных уборах обрабатываются, но точность может быть
  снижена

**Поведение при ошибках:**

| Ситуация                      | Ответ бота                                                 |
| ----------------------------- | ---------------------------------------------------------- |
| Лицо не найдено               | `Лицо не найдено. Попробуйте другое изображение.`          |
| Файл не является изображением | `Не удалось прочитать изображение. Отправьте JPG или PNG.` |
| Поворот головы > 45°          | `Лицо не найдено. Попробуйте фронтальную фотографию.`      |

---

## Мониторинг экспериментов

Все запуски доступны в MLflow UI: http://127.0.0.1:8080

Для каждого запуска сохраняются:

- конфигурация: backbone, head, loss, optimizer, scheduler, fine-tuning
  параметры
- метрики по эпохам: `train_loss`, `val_loss`, `val_mae`, `test_mae`
- git-хеш коммита

---

## Воспроизведение экспериментов

Любой эксперимент из истории MLflow воспроизводится точно:

```bash
# 1. Переключиться на коммит из MLflow
git checkout <git_hash>

# 2. Восстановить данные соответствующей версии
dvc checkout

# 3. Запустить обучение с сохранённой конфигурацией
python -m face_to_age.train
```

---

## Контроль качества кода

При каждом коммите автоматически запускаются проверки:

```bash
pre-commit run --all-files   # запустить вручную
```

| Инструмент              | Назначение                                 |
| ----------------------- | ------------------------------------------ |
| Ruff                    | статический анализ и форматирование Python |
| Prettier                | форматирование YAML, JSON, TOML, Markdown  |
| check-yaml / check-json | валидация синтаксиса конфигов              |
| end-of-file-fixer       | контроль окончания файлов                  |
| trailing-whitespace     | удаление лишних пробелов                   |

---

## Частые проблемы

**`dvc pull` завершается с ошибкой авторизации**

```bash
dvc remote modify myremote --local gdrive_use_service_account false
dvc pull   # пройти авторизацию через браузер
```

**Лицо не найдено при подготовке датасета**

Проверьте разрешение изображений — область лица должна быть не менее 80×80
пикселей. Порог уверенности детектора задаётся в конфигурации:

```yaml
preprocessing:
  detector:
    det_thresh: 0.5 # снизить до 0.3 для сложных изображений
```

**`CUDA out of memory`**

Уменьшите размер батча в конфигурации:

```yaml
dataloader:
  train_batch_size: 32 # или 16
```

**MLflow не отображает новые запуски**

Убедитесь, что `tracking_uri` в конфигурации совпадает с адресом запущенного
сервера:

```yaml
logger:
  tracking_uri: http://127.0.0.1:8080
```

---

## Метрика качества

Основная метрика — **MAE (Mean Absolute Error)**, средняя абсолютная ошибка в
годах. Чем меньше значение — тем точнее модель. Используется как критерий
сохранения лучшего чекпоинта (`val_mae`).

Результаты на тестовой выборке UTKFace (split 0):

| Backbone        | Regression | DLDL | CORAL    | Mean-Variance |
| --------------- | ---------- | ---- | -------- | ------------- |
| ResNet18        | 4.75       | 4.82 | 4.68     | 4.91          |
| ResNet50        | 4.81       | 4.78 | **4.58** | 5.07          |
| EfficientNet-B0 | 8.10       | 6.43 | 6.42     | 6.33          |
