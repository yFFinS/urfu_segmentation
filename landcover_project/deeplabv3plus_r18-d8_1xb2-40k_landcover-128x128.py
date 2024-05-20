_base_ = [
    '../configs/_base_/models/deeplabv3plus_r50-d8.py',
    '../configs/_base_/default_runtime.py',
    '../configs/_base_/schedules/schedule_40k.py'
]

# Изменяем параметры под датасет
norm_cfg = dict(type='BN', requires_grad=True)
dataset_type = 'Landcover'  # название датасета
data_root = 'landcover_project/data'  # путь к папке с изображениями
num_classes = 5
crop_size = (128, 128)  # задаем размер изображений
max_epochs = 5
loss = 'CrossEntropyLoss'

train_pipeline = [
    dict(type='LoadImageFromFile'),  # загружаем изображения
    dict(type='LoadAnnotations'),  # загружаем аннотации
    dict(
        type='RandomResize',  # задаем аугментации изображений
        scale=(256, 256),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(256, 256), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=2,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images',
            seg_map_path='gt'),
        pipeline=train_pipeline,
        ann_file='splits/train.txt')
    )

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images',
            seg_map_path='gt'),
        pipeline=test_pipeline,
        ann_file='splits/train.txt')
    )

test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    pretrained='open-mmlab://resnet18_v1c',  # https://github.com/open-mmlab/mmcv/blob/master/mmcv/model_zoo/open_mmlab.json
    backbone=dict(depth=18),                 # https://mmdetection.readthedocs.io/en/latest/model_zoo.html
    decode_head=dict(
        num_classes=num_classes,
        loss_decode=dict(type=loss)
    ),
    auxiliary_head=dict(
        num_classes=num_classes,
        loss_decode=dict(type=loss)
    )
)

# Изменяем параметры default_runtime
log_processor = dict(by_epoch=True)

# Изменяем параметры scheduler (т.к. обучаем по эпохам)
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=40000,
        by_epoch=True)
]
train_cfg = dict(_delete_=True, max_epochs=5, type='EpochBasedTrainLoop', val_interval=1)
default_hooks = dict(logger=dict(type='LoggerHook', log_metric_by_epoch=True),
                     checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=1))
