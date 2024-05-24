# Инструкция по развёртыванию проекта

Для установки необходимых пакетов при работе с проектом необходимо выполнить следующие команды:

```sh
# Выполняем клонирования репозитория с проектом
git clone https://github.com/Electrotubbie/mmsegmentation.git

# Выполняем создание окружения и установку необходимых пакетов
# при помощи скрипта create_venv.sh
bash mmsegmentation/urfu_project/create_venv.sh
```

# Требования к наборам данных

## Требование к структуре

Датасет иметь следующую структуру:

```
/dataset
    /images
        img1.tif
        img2.tif
        img3.tif
        img4.tif
    /gt
        img1.png
        img2.png
        img3.png
        img4.png
    /splits
        train.txt
        val.txt
```

Файлы `train.txt` и `val.txt` содержат списки изображений для train и val наборов. Например:

`img1.tif` и `img3.tif` принадлежат train набору, тогда `train.txt` будет содержать следующее (только названия файлов без типов):

```
img1
img3
```

А `val.txt` в таком случае будет таким:

```
img2
img4
```

## Требования к изображениям масок в директории gt

Наши текущие наборы при чтении их через библиотеку `PIL` и преобразовании их в `np.array` передают цвет пикселя в формате 0/255.

При работе с mmsegmentation необходимо, чтоб изображения при преобразовании их в массив `np.array` передавали индекс класса из палитры, которую необходимо настраивать в файле [dataset.py](./dataset.py) через словарь `DATASET_COLORMAP`, а не цвет в ч/б формате. Поэтому преобразовать такие файлы можно через функцию `add_palette_for_1ch_img` из файла [aux_tools.py](./aux_tools.py).

Пример:

```python
>>> import numpy as np
>>> from PIL import Image
>>> np.array(Image.open('./M-33-20-D-c-4-2_16.tif')) # файл без палитры
array([[  0,   0,   0, ...,   0,   0,   0],
       [  0,   0,   0, ...,   0,   0,   0],
       [  0,   0,   0, ...,   0,   0,   0],
       ...,
       [128, 128, 128, ..., 128, 128, 128],
       [128, 128, 128, ..., 128, 128, 128],
       [128, 128, 128, ..., 128, 128, 128]], dtype=uint8)
>>> Image.open('./data/gt/M-33-20-D-c-4-2_16.tif').palette # ничего не возвращает, т.к. палитры нет
>>> np.array(Image.open('./M-33-20-D-c-4-2_16.png')) # файл с палитрой
array([[0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       ...,
       [2, 2, 2, ..., 2, 2, 2],
       [2, 2, 2, ..., 2, 2, 2],
       [2, 2, 2, ..., 2, 2, 2]], dtype=uint8)
>>> Image.open('./data/gt/M-33-20-D-c-4-2_16.png').palette.colors # возвращает палитру цветов
{(0, 0, 0): 0, (64, 64, 64): 1, (128, 128, 128): 2, (192, 192, 192): 3, (255, 255, 255): 4}
```

## Преобразование изображений без палитры в изображения с палитрой

Из директории с файлом [aux_tools.py](./aux_tools.py) и вашим набором данных (в случае примера директория data - это набор данных) выполните скрипт из примера ниже для того, чтобы сгенерировать изображения с палитрой.

Пример преобразования:

```python
>>> from aux_tools import add_palette_for_1ch_img
>>> add_palette_for_1ch_img('./data/gt') # в функцию передаём путь к папке с разметкой снимка
100%|███████████████████████████████████████████████████| 10675/10675 [04:18<00:00, 41.33it/s]
```

# Инструкция по работе с файлом конфигурации обучения config.py

Исходный файл [config.py](./configs/config.py) расположен в директории `urfu_project/configs/`.

## Изменение гиперпараметров

Изменение гиперпараметров осуществляется путем редактирования соответсвующих переменных в файле [config.py](./configs/config.py). Гиперпараметры для редактирования представлены ниже.

### Изменение модели

Изменение осуществляется в списке `_base_`.

Доступный список моделей можно посмотреть в папке `configs/_base_/models`.

Например, чтобы изменить модель на `pspnet`, необходимо изменить первый элемент списка `_base_` :

```python
_base_ = [
    '../../configs/_base_/models/pspnet_r50-d8.py', # Изменили на pspnet_r50-d8.py
    '../../configs/_base_/default_runtime.py',
    '../../configs/_base_/schedules/schedule_40k.py'
]
```

### Изменение энкодера (backbone)

Backbone изменяется в 57 строке файла [config.py](./configs/config.py).

В нашем проекте предусмотрено применение следующих backbones:

* resnet18_v1c

* resnet50_v1c

* resnet101_v1c

По умолчанию установлен resnet18_v1c. Чтобы изменить backbone на resnet50_v1c или resnet101_v1c необходимо внести следующие корректировки в файл [config.py](./configs/config.py):

1) Удалить параметры:

```python
    decode_head=dict(
        c1_in_channels=64,
        c1_channels=12,
        in_channels=512,
        channels=128,
    ),
    auxiliary_head=dict(in_channels=256, channels=64))
```

2. Изменить глубину `depth` на соответствующую выбранному backbone. Для resnet50_v1c код будет иметь следующий вид:

```python
    backbone=dict(
        depth=50, # глубина
        norm_cfg=norm_cfg
    )
```

Подводя итог, если мы изменим backbone на resnet50_v1c блок кода с параметрами `model` примет следующий вид:

```python
model = dict(
    data_preprocessor=data_preprocessor,
    pretrained='open-mmlab://resnet50_v1c',  # ссылка для загрузки предобученного совместимого энкодера
    # параметры энкодера
    backbone=dict(
        depth=50, # глубина
        norm_cfg=norm_cfg
    ),
    decode_head=dict(
        num_classes=num_classes,
        loss_decode=dict(type=loss),
        norm_cfg=norm_cfg
    ),
    auxiliary_head=dict(
        num_classes=num_classes,
        loss_decode=dict(type=loss),
        norm_cfg=norm_cfg,
    )
)
```

### Изменение размера изображений,  функции потерь, оптимизатора, числа эпох

Чтобы изменить какой-то из данных параметров: размер изображения, функция потерь, оптимизатор, число эпох, необходимо присвоить желаемое значение соответствующей переменной в [config.py](./configs/config.py). 

Например, меняем оптимизатор на Adam:

```python
optimizer = dict(_delete_=True, type='Adam', lr=0.001, weight_decay=0.0005)
```

## Изменение параметров логирования

Изменение параметров логирования осуществляется в соответствующем блоке в [config.py](./configs/config.py):

```python
# Параметры логирования
experiment_name = f'{dataset_type}_{crop_size[0]}_{loss}_{optimizer["type"]}_bsize_{batch_size}'
logs_dir = 'logs'
work_dir = f'{logs_dir}/{experiment_name}'  # директория для сохранения логов
log_interval = 10  # интервал в итерациях для печати логов
```

### Инструкция по запуску обучения

Из директории с проектом `urfu_project` выполните следующую команду:

```sh
python3 ./train.py ./configs/config.py
```

После запуска эксперимента в `urfu_project/logs` будет создана папка с именем эксперимента.

# Просмотр графиков

Для просмотра графиков в директории `urfu_project` выполняем команду:

```sh
tensorboard --logdir logs
```