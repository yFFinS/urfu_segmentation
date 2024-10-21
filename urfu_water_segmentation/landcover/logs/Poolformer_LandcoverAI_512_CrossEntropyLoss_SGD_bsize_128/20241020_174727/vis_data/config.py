actual_batch_size = 128
batch_size = 32
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/poolformer/poolformer-s36_3rdparty_32xb128_in1k_20220414-d78ff3e8.pth'
crop_size = (
    512,
    512,
)
custom_imports = dict(
    allow_failed_imports=False, imports=[
        'mmpretrain.models',
    ])
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    pad_val=0,
    seg_pad_val=255,
    size=(
        512,
        512,
    ),
    std=[
        58.395,
        57.12,
        57.375,
    ],
    type='SegDataPreProcessor')
data_root = '/misc/home1/m_imm_freedata/Segmentation/Projects/mmseg_water/landcover.ai_512'
dataset_type = 'LandcoverAI'
default_hooks = dict(
    checkpoint=dict(
        by_epoch=True, interval=1, max_keep_ckpts=5, type='CheckpointHook'),
    logger=dict(interval=10, log_metric_by_epoch=True, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(draw=False, interval=500, type='SegVisualizationHook'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
experiment_name = 'Poolformer_LandcoverAI_512_CrossEntropyLoss_SGD_bsize_128'
gradient_accumulation_steps = 4
img_norm_cfg = dict(
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    std=[
        58.395,
        57.12,
        57.375,
    ],
    to_rgb=True)
launcher = 'none'
load_from = None
log_interval = 10
log_level = 'INFO'
log_processor = dict(by_epoch=True)
logs_dir = 'logs'
loss = 'CrossEntropyLoss'
max_epochs = 50
model = dict(
    backbone=dict(
        arch='s36',
        down_pad=1,
        down_patch_size=3,
        down_stride=2,
        drop_path_rate=0.0,
        drop_rate=0.0,
        frozen_stages=0,
        in_pad=2,
        in_patch_size=7,
        in_stride=4,
        init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmclassification/v0/poolformer/poolformer-s36_3rdparty_32xb128_in1k_20220414-d78ff3e8.pth',
            prefix='backbone.',
            type='Pretrained'),
        out_indices=(
            0,
            2,
            4,
            6,
        ),
        type='mmpretrain.PoolFormer'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            87.68234143910625,
            101.27625783429993,
            94.20303444919348,
        ],
        pad_val=0,
        seg_pad_val=255,
        std=[
            20.67239945318879,
            24.662403479389187,
            28.111042386290034,
        ],
        type='SegDataPreProcessor'),
    decode_head=dict(
        align_corners=False,
        channels=128,
        dropout_ratio=0.1,
        feature_strides=[
            4,
            8,
            16,
            32,
        ],
        in_channels=[
            256,
            256,
            256,
            256,
        ],
        in_index=[
            0,
            1,
            2,
            3,
        ],
        loss_decode=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=19,
        type='FPNHead'),
    neck=dict(
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        num_outs=4,
        out_channels=256,
        type='FPN'),
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='EncoderDecoder')
norm_cfg = dict(requires_grad=True, type='SyncBN')
num_classes = 2
num_workers = 8
optim_wrapper = dict(
    accumulative_counts=4,
    clip_grad=None,
    optimizer=dict(lr=0.001, momentum=0.9, type='SGD', weight_decay=0.0005),
    type='OptimWrapper')
optimizer = dict(lr=0.001, momentum=0.9, type='SGD', weight_decay=0.0005)
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        end=40000,
        eta_min=0.0001,
        power=0.9,
        type='PolyLR'),
]
resume = False
splits = 'splits'
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=32,
    dataset=dict(
        ann_file='splits/val.txt',
        data_prefix=dict(img_path='images', seg_map_path='gt'),
        data_root=
        '/misc/home1/m_imm_freedata/Segmentation/Projects/mmseg_water/landcover.ai_512',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                2048,
                512,
            ), type='Resize'),
            dict(size_divisor=32, type='ResizeToMultiple'),
            dict(reduce_zero_label=True, type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='LandcoverAI'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ], type='IoUMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        2048,
        512,
    ), type='Resize'),
    dict(size_divisor=32, type='ResizeToMultiple'),
    dict(reduce_zero_label=True, type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
train_cfg = dict(max_epochs=50, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_size=32,
    dataset=dict(
        ann_file='splits/train.txt',
        data_prefix=dict(img_path='images', seg_map_path='gt'),
        data_root=
        '/misc/home1/m_imm_freedata/Segmentation/Projects/mmseg_water/landcover.ai_512',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(reduce_zero_label=True, type='LoadAnnotations'),
            dict(
                keep_ratio=True,
                ratio_range=(
                    0.5,
                    2.0,
                ),
                scale=(
                    2048,
                    512,
                ),
                type='RandomResize'),
            dict(
                cat_max_ratio=0.75, crop_size=(
                    512,
                    512,
                ), type='RandomCrop'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PhotoMetricDistortion'),
            dict(type='PackSegInputs'),
        ],
        type='LandcoverAI'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(reduce_zero_label=True, type='LoadAnnotations'),
    dict(
        keep_ratio=True,
        ratio_range=(
            0.5,
            2.0,
        ),
        scale=(
            2048,
            512,
        ),
        type='RandomResize'),
    dict(cat_max_ratio=0.75, crop_size=(
        512,
        512,
    ), type='RandomCrop'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]
tta_model = dict(type='SegTTAModel')
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=32,
    dataset=dict(
        ann_file='splits/val.txt',
        data_prefix=dict(img_path='images', seg_map_path='gt'),
        data_root=
        '/misc/home1/m_imm_freedata/Segmentation/Projects/mmseg_water/landcover.ai_512',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                2048,
                512,
            ), type='Resize'),
            dict(size_divisor=32, type='ResizeToMultiple'),
            dict(reduce_zero_label=True, type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='LandcoverAI'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ], type='IoUMetric')
vis_backends = [
    dict(scalar_save_file='../../scalars.json', type='LocalVisBackend'),
    dict(
        save_dir=
        'logs/Poolformer_LandcoverAI_512_CrossEntropyLoss_SGD_bsize_128',
        type='TensorboardVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(scalar_save_file='../../scalars.json', type='LocalVisBackend'),
        dict(
            save_dir=
            'logs/Poolformer_LandcoverAI_512_CrossEntropyLoss_SGD_bsize_128',
            type='TensorboardVisBackend'),
    ])
work_dir = 'logs/Poolformer_LandcoverAI_512_CrossEntropyLoss_SGD_bsize_128'
