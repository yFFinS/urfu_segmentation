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