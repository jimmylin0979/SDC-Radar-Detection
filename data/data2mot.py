import os
import cv2
import numpy as np
import json
import shutil
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser

def gen_boundingbox_rot(bbox, angle):
        """
        generate a list of 2D points from bbox and angle 
        """
        theta = np.deg2rad(-angle)
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        points = np.array([[bbox[0], bbox[1]],
                           [bbox[0] + bbox[2], bbox[1]],
                           [bbox[0] + bbox[2], bbox[1] + bbox[3]],
                           [bbox[0], bbox[1] + bbox[3]]]).T

        cx = bbox[0] + bbox[2] / 2
        cy = bbox[1] + bbox[3] / 2
        T = np.array([[cx], [cy]])

        points = points - T
        points = np.matmul(R, points) + T
        points = points.astype(int)

        # [[x1, x2, x3, x4], [y1, y2, y3, y4]]
        min_x = np.min(points[0, :])
        min_y = np.min(points[1, :])
        max_x = np.max(points[0, :])
        max_y = np.max(points[1, :])

        return [min_x, min_y, max_x, max_y]

def get_radar_dicts(root_src: str, root_dst: str):
    
    dataset_dicts = []
    idd = 0
    folders = os.listdir(root_src)
    folders.sort()
    seen_class_name = set()
    # folder_size = len(folders)
    for folder in folders:

        if not os.path.isdir(os.path.join(root_src, folder)):
            continue

        radar_folder = os.path.join(root_src, folder, 'Navtech_Cartesian')
        annotation_path = os.path.join(root_src, folder, 'annotations', 'annotations.json')
        with open(annotation_path, 'r') as f_annotation:
            annotations = json.load(f_annotation)

        # transfer tp jpg
        for file in tqdm(sorted(os.listdir(radar_folder))):
            if not file.endswith(".png"):
                continue
            img = cv2.imread(os.path.join(radar_folder, file))
            os.makedirs(os.path.join(root_src, folder, 'img1'), exist_ok=True)
            cv2.imwrite(os.path.join(os.path.join(root_src, folder, 'img1'), f"{file[:-4]}.jpg"), img)

        objs = []
        for annotation in tqdm(annotations):

            object_id = annotation['id']
            class_name = annotation['class_name']   # ['cars', ]
            seen_class_name.add(class_name)
            visibility = 1.0

            # Only care for class car
            # classes=('group_of_pedestrians', 'truck', 'pedestrian', 'van', 'bus', 'car', 'bicycle')
            if class_name in ['group_of_pedestrians', 'pedestrian']:
                continue

            obj = []

            radar_files = os.listdir(radar_folder)
            radar_files.sort()
            for frame_number in range(len(radar_files)):
                record = {}
                bb_created = False
                idd += 1
                filename = os.path.join(radar_folder, radar_files[frame_number])

                if radar_files[frame_number] == 'desktop.ini':
                    continue
                if (not os.path.isfile(filename)):
                    print(filename)
                    continue
                # record["file_name"] = filename
                # record["image_id"] = idd
                # record["height"] = 1152
                # record["width"] = 1152

                if (annotation['bboxes'][frame_number]):
                    # example {"position": [563.4757032731811, 490.9060756081957, 15.766302833034956, 24.05435500075953], "rotation": 0},
                    position = annotation['bboxes'][frame_number]['position']
                    angle = annotation['bboxes'][frame_number]['rotation']

                    x, y, w, h = position
                    # position = get_rotated_rect_corners(x, y, w, h, angle)
                    position_2d = gen_boundingbox_rot([x, y, w, h], angle)
                    # position_2d = position_2d.tolist()
                    x = position_2d[0]
                    y = position_2d[1]
                    w = position_2d[2] - position_2d[0]
                    h = position_2d[3] - position_2d[1]

                    obj_in_frame = [frame_number + 1]
                    obj_in_frame.append(object_id)
                    obj_in_frame.extend([x, y, w, h])
                    obj_in_frame.append(1)
                    obj_in_frame.append(1)
                    obj_in_frame.append(visibility)
                    obj.append(obj_in_frame)

            objs.append(obj)
            
        # os.path.join(root_src, folder, 'Navtech_Cartesian')
        os.makedirs(f'{root_dst}/{folder}/gt', exist_ok=True)
        os.makedirs(f'{root_dst}/{folder}/det', exist_ok=True)
        # for mode in ['train', 'valid']:
        #     os.makedirs(f'{root_dst}/{mode}/images', exist_ok=True)
        #     os.makedirs(f'{root_dst}/{mode}/GTs', exist_ok=True)
        #     os.makedirs(f'{root_dst}/{mode}/annotations', exist_ok=True)
        
        # # split the files into train/valid depend on the train_test_spilt result
        # mode = 'train' # if frame_number in train_indices else 'valid'
        # shutil.copyfile(filename, os.path.join(f'{root_dst}/{mode}/images', f'{folder}_{radar_files[frame_number]}'))

        # print(objs)
        with open(f'{root_dst}/{folder}/gt/gt.txt', 'w') as fw:
            # image = cv2.imread(os.path.join(os.path.join(f'{root_dst}/{mode}/images', f'{folder}_{radar_files[frame_number]}')))
            for obj in tqdm(objs):
                # 
                # image = draw_rotated_bbox(image, obj[:8], colors[obj[8]])
                # image = draw_boundingbox_rot(image, obj[:8])

                for obj_in_frame in obj:
                    # 
                    obj_in_frame = [str(i) for i in obj_in_frame]
                    # obj[-2] = 'car'
                    annotation = ','.join(obj_in_frame)
                    fw.write(annotation + '\n')

        # print(f'{root_dst}/{folder}')
        with open(f'{root_dst}/{folder}/det/det.txt', 'w') as fw:
            # image = cv2.imread(os.path.join(os.path.join(f'{root_dst}/{mode}/images', f'{folder}_{radar_files[frame_number]}')))
            for obj in tqdm(objs):
                # 
                # image = draw_rotated_bbox(image, obj[:8], colors[obj[8]])
                # image = draw_boundingbox_rot(image, obj[:8])

                for obj_in_frame in obj:
                    # 
                    det_in_frame = [obj_in_frame[0], -1, obj_in_frame[2], obj_in_frame[3], obj_in_frame[4], obj_in_frame[5], 1]
                    det_in_frame = [str(i) for i in det_in_frame]
                    # obj[-2] = 'car'
                    annotation = ','.join(det_in_frame)
                    fw.write(annotation + '\n')
            
        #     cv2.imwrite(os.path.join(f'{root_dst}/{mode}/GTs', f'{folder}_{radar_files[frame_number]}'), image)

        # TODO
        # break
    
    print(seen_class_name)
    return dataset_dicts

if __name__ == '__main__':

    # python3 visualization.py --root mini_train

    parser = ArgumentParser()
    parser.add_argument("--root_src", help="root of the radar dataset folder", type=str, default='mini_train')
    parser.add_argument("--root_dst", help="root of the destination folder", type=str, default='mini_train')
    args = parser.parse_args()

    dicts = get_radar_dicts(args.root_src, args.root_dst)