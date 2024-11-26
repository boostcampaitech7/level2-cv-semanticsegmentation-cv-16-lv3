
_base_ = [
    '../configs/hrnet/fcn_hr18_4xb4-80k_isaid-896x896.py',
 #  '../configs/_base_/schedules/schedule_80k.py',
  #  "../configs/_base_/default_runtime.py",
]


data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    size=(1024, 1024),
    pad_val=0,
    seg_pad_val=255,
)


# model = dict(
#     pretrained='open-mmlab://msra/hrnetv2_w48',
#     data_preprocessor=data_preprocessor,
#     backbone=dict(
#         extra=dict(
#             stage2=dict(num_channels=(48, 96)),
#             stage3=dict(num_channels=(48, 96, 192)),
#             stage4=dict(num_channels=(48, 96, 192, 384)))),
#     decode_head=dict(
#         type='FCNHeadWithoutAccuracy',
#         in_channels=[48, 96, 192, 384],
#         channels=sum([48, 96, 192, 384]),
#         num_classes=29,
#         loss_decode=dict(
#             type='CrossEntropyLoss',
#             use_sigmoid=True,
#             loss_weight=1.0,
#         ),
# ))
model = dict(
    pretrained='open-mmlab://msra/hrnetv2_w48',  # HRNetV2-W48 모델의 pre-trained weights
    data_preprocessor=data_preprocessor,
    type='EncoderDecoderWithoutArgmax',
    backbone=dict(
        extra=dict(
            stage2=dict(num_channels=(48, 96)),
            stage3=dict(num_channels=(48, 96, 192)),
            stage4=dict(num_channels=(48, 96, 192, 384))
        )
    ),
    decode_head=dict(
        type='FCNHeadWithoutAccuracy',  # FCNHead 유형을 사용
        in_channels=[48, 96, 192, 384],  # 각 stage에서의 입력 채널 수
        channels=sum([48, 96, 192, 384]),  # 전체 채널 수 (모든 stage 채널의 합)
        num_classes=29,  # 출력 클래스 수
        input_transform='resize_concat', 
        loss_decode=dict(
            type='CrossEntropyLoss',  # CrossEntropyLoss 사용
            use_sigmoid=True,  # Sigmoid 활성화 사용
            loss_weight=1.0,  # 손실 가중치
        ),
    ),
    test_cfg=dict(
        mode='whole',  # Whole image inference during testing
        crop_size=(1024, 1024)  # Ensures the output is of size 1024x1024
    ),
    # val_cfg=dict(  # 이 부분은 MMsegmentation에 기본적으로 지원되지는 않지만, 원하는 경우 커스터마이징 가능
    #     mode='whole',
    #     crop_size=(1024, 1024),
    # ),
)
# optimizer
optimizer = dict(type='AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=0.01)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)

# mixed precision
fp16 = dict(loss_scale='dynamic')

# learning policy
param_scheduler = [
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
test_dataloader = val_dataloader

val_evaluator = dict(type='DiceMetric')
test_evaluator = val_evaluator
