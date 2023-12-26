_base_ = ['../rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le90.py']

data_root = '/home/jimmylin0979/Desktop/Disk1/SDC-Final/data/mini_train_dota/'
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

dataset_type = 'DOTADataset'
classes=('group_of_pedestrians', 'truck', 'pedestrian', 'van', 'bus', 'car', 'bicycle')
data = dict(
    train=dict(
        type=dataset_type,
        pipeline=train_pipeline,
        classes=classes,
        ann_file=data_root + 'train/annotations/',
        img_prefix=data_root + 'train/images/'),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'valid/annotations/',
        img_prefix=data_root + 'valid/images/'),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'test/annotations/',
        img_prefix=data_root + 'test/images/'))
