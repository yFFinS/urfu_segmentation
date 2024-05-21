_base_ = [
    '../../configs/_base_/models/deeplabv3plus_r50-d8.py',
    '../../configs/_base_/default_runtime.py',
    '../../configs/_base_/schedules/schedule_40k.py'
]

# ----------------------------------------------------------------
# Изменение гиперпараметров

# Поскольку мы используем только один графический процессор, вместо SyncBN используется BN
norm_cfg = dict(type='BN', requires_grad=True)

# Название датасета из файла urfu_project/dataset.py
dataset_type = 'Dataset'

# Путь к папке с преобразованным набором данных
data_root = 'data'

# Количество классов для сегментации
num_classes = 5

# Размер изображения, который принимает на вход сеть
crop_size = (128, 128)

# Количичество эпох для обучения
max_epochs = 10

# Функция потерь
loss = 'CrossEntropyLoss'

# Размер батча
batch_size = 2

# num_workers
num_workers = 1

# Оптимизатор
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)

# Параметры логирования
experiment_name = f'{dataset_type}_{crop_size[0]}_{loss}_{optimizer["type"]}_bsize_{batch_size}'
logs_dir = 'logs'
work_dir = f'{logs_dir}/{experiment_name}'  # директория для сохранения логов
log_interval = 10  # интервал для печати логов

# Splits directory
splits = 'debug_split'

# Имя эксперимента
# ----------------------------------------------------------------
# Параметры модели
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    pretrained='open-mmlab://resnet18_v1c',  # https://github.com/open-mmlab/mmcv/blob/master/mmcv/model_zoo/open_mmlab.json
    backbone=dict(
        depth=18,
        norm_cfg=norm_cfg
    ),                 # https://mmdetection.readthedocs.io/en/latest/model_zoo.html
    decode_head=dict(
        num_classes=num_classes,
        loss_decode=dict(type=loss),
        norm_cfg=norm_cfg,
        c1_in_channels=64,
        c1_channels=12,
        in_channels=512,
        channels=128,
    ),
    auxiliary_head=dict(
        num_classes=num_classes,
        loss_decode=dict(type=loss),
        norm_cfg=norm_cfg,
        in_channels=256,
        channels=64
    )
)
# ----------------------------------------------------------------
# Параметры default_runtime
log_processor = dict(by_epoch=True)
# ----------------------------------------------------------------
# Параметры scheduler
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=40000,
        by_epoch=True)
]
train_cfg = dict(_delete_=True, max_epochs=max_epochs, type='EpochBasedTrainLoop', val_interval=1)
# В default_hooks draw=False для того, чтобы не выводить изображения из val с результатами модели после эпохи в логи
# Если поставить True, то увеличьте интервал до более высокого, чтоб не сохранял часто много изображений
default_hooks = dict(logger=dict(type='LoggerHook', log_metric_by_epoch=True, interval=log_interval),
                     checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=1, max_keep_ckpts=1),
                     visualization=dict(type='SegVisualizationHook', draw=False, interval=500))
# ----------------------------------------------------------------
# Параметры датасета
train_pipeline = [
    dict(type='LoadImageFromFile'),  # загружаем изображения
    dict(type='LoadAnnotations'),  # загружаем аннотации
    dict(
        type='RandomResize',  # задаем аугментации изображений
        scale=crop_size,
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=crop_size, keep_ratio=True),
    # добавьте загрузочную аннотацию после `Resize`, 
    # потому что val не нуждается в преобразовании данных с изменением размера
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images',
            seg_map_path='gt'),
        pipeline=train_pipeline,
        ann_file=f'{splits}/train.txt')
    )

val_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images',
            seg_map_path='gt'),
        pipeline=test_pipeline,
        ann_file=f'{splits}/train.txt')
    )

test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [  # Test Time Augmentation (TTA)
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
        ])
]
# ----------------------------------------------------------------
# Tensorboard visualization
vis_backends = [dict(type='LocalVisBackend'),  # save_dir=f'{work_dir}/{experiment_name}'),
                dict(type='TensorboardVisBackend')]  #, save_dir=f'{work_dir}/{experiment_name}')]
visualizer = dict(type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
# ----------------------------------------------------------------
