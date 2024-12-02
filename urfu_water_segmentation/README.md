# Гайд

urfu_water_segmentation - измененная версия urfu_project. Советуем так же ознакомиться с urfu_project/README.md, тут будут только ключевые изменения.

## Окружение
Зависимости зафиксированы при помощи poetry. Для полной установки окруения достаточно написать
```sh
sh init_venv.sh
```
находясь в urfu_water_segmentation

Далее для активации окружения можно использовать
```sh
source .venv/bin/activate
```

## Датасет landcover.ai

Готовый датасет лежит в `/misc/home1/m_imm_freedata/Segmentation/Projects/mmseg_water/landcover.ai_512`, его уже можно использовать в mmsegmentation.

Для запуска обучения на этом датасете:
```sh
cd landcover
sh dist_train.sh
```

Для мониторинга обучения:
```sh
tensorboard --logdir logs
```

# Валидация на нескольких датасетах
Для того, чтобы посчитать метрики обученной модели на нескольких датасетах, можно использовать два файла `evaluate.sh` и `evaluate.py`.
Если ничего менять не нужно, то можно просто запустить валидацию
```sh
sh evaluate.sh
```

Результаты будут писатьсяв логи джобы, но так же они будут сохранены в файле `eval_<id батч джобы>.csv`.

> В `evaluate.sh` настраиваются параметры запуска батч джобы, а так же папка с экспериментом, откуда будет взят конфиг модели и последний чекпоинт.

> В `evaluate.py` можно изменить список датасетов, на которых будет проходить валидация.

# Хард аугментации 
Чтобы запустить обучение с хард аугментациями надо 
1) в configе в train_pipeline их добавить, вот пример как это можно сделать
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(
        type='RandomResize',
        scale=(2048, 512),
        ratio_range=(0.5, 2.0),
        keep_ratio=True
    ),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(
        type='Resize',
        scale=(512, 512),
        keep_ratio=False  # Force exact 512×512 resizing
    ),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='HardAugmentationsTransform',
        tile_size=crop_size[0],
        cloud_shadows_dir=clouds_shadows_path,
        for_3ch_img=True,
        mask_dropout=True,
        make_blue=True
    ),
    dict(type='PackSegInputs')
]

2) установить в .venv albumentations
```sh
source .venv/bin/activate
pip install albumentations
```
