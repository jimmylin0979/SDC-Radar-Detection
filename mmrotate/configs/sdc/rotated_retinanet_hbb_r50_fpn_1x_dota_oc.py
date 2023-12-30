
# the new config inherits the base configs to highlight the necessary modification
_base_ = '../rotated_retinanet/rotated_retinanet_hbb_r50_fpn_1x_dota_oc.py'

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
        ann_file='/home/jimmylin0979/Desktop/Disk1/SDC-Final/data/mini_train_dota/test/annotations',
        img_prefix='/home/jimmylin0979/Desktop/Disk1/SDC-Final/data/mini_train_dota/test/images'),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file='/home/jimmylin0979/Desktop/Disk1/SDC-Final/data/mini_train_dota/test/annotations',
        img_prefix='/home/jimmylin0979/Desktop/Disk1/SDC-Final/data/mini_train_dota/test/images'))

# 2. model settings
model = dict(
    bbox_head=dict(
        type='RotatedRetinaHead',
        # explicitly over-write all the `num_classes` field from default 15 to 5.
        num_classes=len(classes)))