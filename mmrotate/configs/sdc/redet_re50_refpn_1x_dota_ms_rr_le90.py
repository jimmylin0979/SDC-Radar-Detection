_base_ = ['../redet/redet_re50_refpn_1x_dota_le90.py']

angle_version = 'le90'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
    dict(
        type='PolyRandomRotate',
        rotate_ratio=0.5,
        angles_range=180,
        auto_bound=False,
        rect_classes=[9, 11],
        version=angle_version),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]


data_root = '/home/jimmylin0979/Desktop/Disk1/SDC-Final/data/mini_train_dota/'
# classes=('group_of_pedestrians', 'truck', 'pedestrian', 'van', 'bus', 'car', 'bicycle')
classes=('car', )
dataset_type = 'DOTADataset'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        pipeline=train_pipeline,
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'train/annotations/',
        img_prefix=data_root + 'train/images_preprocessed/'),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'test/annotations/',
        img_prefix=data_root + 'test/images_preprocessed/'),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'test/images/',
        img_prefix=data_root + 'test/images_preprocessed/'))


model = dict(
    train_cfg=dict(rpn=dict(assigner=dict(gpu_assign_thr=200))), 
    roi_head=dict(
        bbox_head=[
            dict(
                type='RotatedShared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=len(classes),
                bbox_coder=dict(
                    type='DeltaXYWHAHBBoxCoder',
                    angle_range=angle_version,
                    norm_factor=2,
                    edge_swap=True,
                    target_means=[0., 0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2, 1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='RotatedShared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=len(classes),
                bbox_coder=dict(
                    type='DeltaXYWHAOBBoxCoder',
                    angle_range=angle_version,
                    norm_factor=None,
                    edge_swap=True,
                    proj_xy=True,
                    target_means=[0., 0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1, 0.5]),
                reg_class_agnostic=False,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ]
    ))

workflow = [('train', 1), ('val', 1)]
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', save_best='auto'))

custom_hooks = [
    dict(type='EMAHook')
] 