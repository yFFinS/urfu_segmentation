import os
import numpy as np
from PIL import Image
from tqdm import tqdm


def add_palette_for_1ch_img(path,
                            palette=((0, 0, 0),  # 0
                                     (64, 64, 64),  # 1
                                     (128, 128, 128),  # 2
                                     (192, 192, 192),  # 3
                                     (255, 255, 255)),  # 4
                            pic_suffix='.tif'):
    '''
    !!! Функция предназначена для подготовки масок снимков к обучению с помощью MMSegmentation.
    Функция для создания палтиры у изображения с последующей перезаписью изображения в формате PNG. 

    path: путь к директории с изображениями масок
    palette: палитра цветов у масок
        Пример палитры:
        [[0, 0, 0], [64, 64, 64], [128, 128, 128], [192, 192, 192], [255, 255, 255]]
        для классов:
        ('background', 'water', 'tree', 'roads', 'buildings')
    pic_suffix='.tif': тип изображения
    '''
    palette_dict = {int(color[0]):i for i, color in enumerate(palette)}
    for img in tqdm(list(os.scandir(path))):
        im = Image.open(img.path)
        ar = np.array(im)
        ar = np.vectorize(palette_dict.get)(ar)
        im = Image.fromarray(ar).convert('P')
        im.putpalette(np.array(palette, dtype=np.uint8))
        im.save(img.path.rstrip(pic_suffix) + '.png')


def create_splits(dataset_path,
                  splits_dir_name,
                  imgs_dir_name,
                  pic_count=None,
                  train_size=0.7,
                  pic_suffix=''):
    '''
    !!! В директории с изображениями не должно быть иных файлов. Возможно добавление в train val чего-то лишнего. 
    Функция для создания папки splits с случайным выбором изображений из общей папки imgs_dir_name, 
    находящейся на пути dataset_path.

    В папку splits сохраняет train.txt и val.txt с списком картинок.
    Количество случайно взятых изображений из общего набора можно регулировать через pic_count.
    Размер тренировочного набора можно регулировать через train_size.

    dataset_path: папка с набором данных
    splits_dir_name: путь к папке с наборами train.txt и val.txt в папке с набором данных
    imgs_dir_name: путь к папке с набором изображений
    pic_count: количество случайно выбираемых изображений из общего набора (опционально)
    train_size: размер тренировочной выборки из pic_count, по умолчанию 0.7
    pic_suffix: тип файла изображений
    '''
    splits = os.path.join(dataset_path, splits_dir_name)
    images = os.path.join(dataset_path, imgs_dir_name)

    if not os.path.isdir(splits):
        os.mkdir(splits)
        print(f'Папка {splits} создана.')

    filename_list = np.array(list(os.scandir(images)))
    if pic_count:
        filename_list = np.random.choice(filename_list, size=pic_count)

    n_train_pics = int(train_size*len(filename_list))
    train_idx = np.random.choice(np.arange(len(filename_list)), size=n_train_pics)
    with open(os.path.join(splits, 'train.txt'), 'w') as f:
        f.writelines(line.name.strip(pic_suffix) + '\n' for line in filename_list[train_idx])
        print(f'Набор train.txt создан, {len(train_idx)} изображений.')
    with open(os.path.join(splits, 'val.txt'), 'w') as f:
        f.writelines(line.name.strip(pic_suffix) + '\n' for line in filename_list[~train_idx])
        print(f'Набор val.txt создан, {len(filename_list[~train_idx])} изображений.')
