
# the new config inherits the base configs to highlight the necessary modification
_base_ = '../rotated_retinanet/rotated_retinanet_hbb_r50_fpn_1x_dota_oc.py'

# 1. dataset settings
# explicitly add your class names to the field `classes`
classes=('car', )

dataset_type = 'DOTADataset'
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file='/home/jimmylin0979/Desktop/Disk1/SDC-Final/data/mini_train_dota/train/annotations',
        img_prefix='/home/jimmylin0979/Desktop/Disk1/SDC-Final/data/mini_train_dota/train/images'),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file='/home/jimmylin0979/Desktop/Disk1/SDC-Final/data/mini_train_dota/test/annotations',
        img_prefix='/home/jimmylin0979/Desktop/Disk1/SDC-Final/data/mini_train_dota/test/images'),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file='/home/jimmylin0979/Desktop/Disk1/SDC-Final/data/mini_train_dota/test/annotations',
        img_prefix='/home/jimmylin0979/Desktop/Disk1/SDC-Final/data/mini_train_dota/test/images'))

# 2. model settings
model = dict(
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet101')),
    bbox_head=dict(
        type='RotatedRetinaHead',
        # explicitly over-write all the `num_classes` field from default 15 to 5.
        num_classes=len(classes)))

workflow = [('train', 1), ('val', 1)]
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', save_best='auto'))

# custom_hooks = [
#     dict(type='EMAHook')
# ] 

# evaluation
evaluation = dict(interval=1, metric='mAP')
# optimizer
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[24, 33])
runner = dict(type='EpochBasedRunner', max_epochs=36)
checkpoint_config = dict(interval=1)
