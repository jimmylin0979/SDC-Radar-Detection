_base_ = ['../oriented_reppoints/oriented_reppoints_r50_fpn_1x_dota_le135.py']

# 1. dataset settings
# explicitly add your class names to the field `classes`
classes=('group_of_pedestrians', 'truck', 'pedestrian', 'van', 'bus', 'car', 'bicycle')

dataset_type = 'DOTADataset'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file='/home/jimmylin0979/Desktop/Disk1/SDC-Final/data/mini_train_dota/train/annotations',
        img_prefix='/home/jimmylin0979/Desktop/Disk1/SDC-Final/data/mini_train_dota/train/images'),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file='/home/jimmylin0979/Desktop/Disk1/SDC-Final/data/mini_train_dota/valid/annotations',
        img_prefix='/home/jimmylin0979/Desktop/Disk1/SDC-Final/data/mini_train_dota/valid/images'),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file='/home/jimmylin0979/Desktop/Disk1/SDC-Final/data/mini_train_dota/test/annotations',
        img_prefix='/home/jimmylin0979/Desktop/Disk1/SDC-Final/data/mini_train_dota/test/images'))

# 2. model settings
model = dict(
    bbox_head=dict(
        type='OrientedRepPointsHead',
        # explicitly over-write all the `num_classes` field from default 15 to 5.
        num_classes=len(classes)))