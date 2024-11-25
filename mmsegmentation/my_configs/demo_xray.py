

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b5_20220624-658746d9.pth'  # noqa



_base_ = [
    "../configs/_base_/models/segformer_mit-b0.py",    
    "../configs/_base_/default_runtime.py",
]


data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[0., 0., 0.],
    std=[255., 255., 255.],
    bgr_to_rgb=True,
    size=(1024, 1024),
    pad_val=0,
    seg_pad_val=255,
)

model = dict(
    type='EncoderDecoderWithoutArgmax',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        embed_dims=64,
        num_heads=[1, 2, 5, 8],
        num_layers=[3, 6, 40, 3]),
    decode_head=dict(
        type='SegformerHeadWithoutAccuracy',
        in_channels=[64, 128, 320, 512],
        num_classes=29,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
        ),
    ),
)

# optimizer
optimizer = dict(type='AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=0.01)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)

# mixed precision
fp16 = dict(loss_scale='dynamic')

# learning policy
param_scheduler = [
    # dict(
    #     type='LinearLR', start_factor=1e-4, by_epoch=False, begin=0, end=1500
    # ),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=0,
        end=20000,
        by_epoch=False,
    )
]

# training schedule for 20k
train_cfg = dict(type='IterBasedTrainLoop', max_iters=20000, val_interval=2000)
# dict(
#     type='EpochBasedTrainLoop',
#     max_epochs=2,  # 총 epoch 수
#     val_interval=1  # 검증 주기 (매 epoch마다)
#     )
#dict(type='IterBasedTrainLoop', max_iters=20000, val_interval=2000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=80, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook')
)


# dataset settings
dataset_train_type = 'XRayDataset'
dataset_test_type = 'XRayDatasetTest'
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadXRayAnnotations'),
    dict(type='Resize', scale=(1024, 1024)),
    dict(type='TransposeAnnotations'),
    dict(type='PackSegInputs')
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1024, 1024)),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadXRayAnnotations'),
    dict(type='TransposeAnnotations'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1024, 1024)),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_train_type,
        is_train=True,
        pipeline=train_pipeline
    )
)
val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_train_type,
        is_train=False,
        pipeline=val_pipeline
    )
)
test_dataloader = dict(
    batch_size=16,
    num_workers=8,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_test_type,
        pipeline=test_pipeline
    )
)

val_evaluator = dict(type='DiceMetric')
test_evaluator = dict(type='NoneMetric')



# model = dict(
#     backbone=dict(
#         init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
#         embed_dims=64,
#         num_layers=[3, 6, 40, 3]),
#     decode_head=dict(in_channels=[64, 128, 320, 512]))


# Train Segformer Mit B3
# _base_ = ['./segformer_mit-b0_8xb1-160k_cityscapes-1024x1024.py']

# checkpoint="https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b3_20220624-13b1141c.pth"
# model = dict(
#     type='EncoderDecoderWithoutArgmax',
#     data_preprocessor=data_preprocessor,
#     backbone=dict(
#         init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
#         embed_dims=64,
#         num_heads=[1, 2, 5, 8],
#         num_layers=[3, 4, 18, 3]),
#     decode_head=dict(
#         type='SegformerHeadWithoutAccuracy',
#         in_channels=[64, 128, 320, 512],
#         num_classes=29,
#         loss_decode=dict(
#             type='CrossEntropyLoss',
#             use_sigmoid=True,
#             loss_weight=1.0,
#         ),
#     ),
# )

