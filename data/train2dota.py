import os
import cv2
import numpy as np
import json
import shutil
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser

def get_rotated_rect_corners(x, y, w, h, theta):
    
    center_x = x + w / 2
    center_y = y + h / 2
    cos_theta = np.cos(np.radians(theta))
    sin_theta = np.sin(np.radians(theta))

    x1 = int(center_x + w / 2 * cos_theta - h / 2 * sin_theta)
    y1 = int(center_y + w / 2 * sin_theta + h / 2 * cos_theta)
    x2 = int(center_x - w / 2 * cos_theta - h / 2 * sin_theta)
    y2 = int(center_y - w / 2 * sin_theta + h / 2 * cos_theta)
    x3 = int(center_x - w / 2 * cos_theta + h / 2 * sin_theta)
    y3 = int(center_y - w / 2 * sin_theta - h / 2 * cos_theta)
    x4 = int(center_x + w / 2 * cos_theta + h / 2 * sin_theta)
    y4 = int(center_y + w / 2 * sin_theta - h / 2 * cos_theta)

    return [x1, y1, x2, y2, x3, y3, x4, y4]

def draw_rotated_bbox(image, position):

    x1, y1, x2, y2, x3, y3, x4, y4 = position
    corners = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.int32)
    corners = corners.reshape((-1, 1, 2))
    cv2.polylines(image, [corners], isClosed=True, color=(0, 255, 0), thickness=2)
    
    return image


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

        radar_files = os.listdir(radar_folder)
        radar_files.sort()
        for frame_number in tqdm(range(len(radar_files))):
            record = {}
            objs = []
            bb_created = False
            idd += 1
            filename = os.path.join(radar_folder, radar_files[frame_number])

            indices = list(range(len(radar_files) - 1))
            test_size = 0.2
            random_state = 42
            train_indices, valid_indices = train_test_split(
                indices,
                test_size=test_size,
                # stratify=dataset.targets,
                random_state=random_state,
            )

            if radar_files[frame_number] == 'desktop.ini':
                continue
            if (not os.path.isfile(filename)):
                print(filename)
                continue
            record["file_name"] = filename
            record["image_id"] = idd
            record["height"] = 1152
            record["width"] = 1152

            for annotation in annotations:

                object_id = annotation['id']
                class_name = annotation['class_name']   # ['cars', ]
                seen_class_name.add(class_name)
                difficulty = 0

                if (annotation['bboxes'][frame_number]):
                    # example {"position": [563.4757032731811, 490.9060756081957, 15.766302833034956, 24.05435500075953], "rotation": 0},
                    position = annotation['bboxes'][frame_number]['position']
                    theta = annotation['bboxes'][frame_number]['rotation']

                    x, y, w, h = position
                    position = get_rotated_rect_corners(x, y, w, h, theta)
                    
                    obj = position
                    obj.append(class_name)
                    obj.append(difficulty)
                    objs.append(obj)

            # 
            os.makedirs(f'{root_dst}', exist_ok=True)
            for mode in ['train', 'valid']:
                os.makedirs(f'{root_dst}/{mode}/images', exist_ok=True)
                os.makedirs(f'{root_dst}/{mode}/GTs', exist_ok=True)
                os.makedirs(f'{root_dst}/{mode}/annotations', exist_ok=True)
            
            # split the files into train/valid depend on the train_test_spilt result
            mode = 'train' if frame_number in train_indices else 'valid'
            shutil.copyfile(filename, os.path.join(f'{root_dst}/{mode}/images', f'{folder}_{radar_files[frame_number]}'))
            with open(os.path.join(f'{root_dst}/{mode}/annotations', f'{folder}_{radar_files[frame_number][:-4]}.txt'), 'w') as fw:
                image = cv2.imread(os.path.join(os.path.join(f'{root_dst}/{mode}/images', f'{folder}_{radar_files[frame_number]}')))
                for obj in objs:
                    # 
                    image = draw_rotated_bbox(image, obj[:8])

                    # 
                    obj = [str(i) for i in obj]
                    annotation = ' '.join(obj)
                    fw.write(annotation + '\n')
                
                cv2.imwrite(os.path.join(f'{root_dst}/{mode}/GTs', f'{folder}_{radar_files[frame_number]}'), image)

            # if bb_created:
            #     record["annotations"] = objs
            #     dataset_dicts.append(record)

        # TODO
        # break
    
    print(seen_class_name)
    return dataset_dicts

if __name__ == '__main__':

    # python3 visualization.py --root mini_train

    parser = ArgumentParser()
    parser.add_argument("--root_src", help="root of the radar dataset folder", type=str, default='mini_train')
    parser.add_argument("--root_dst", help="root of the destination folder", type=str, default='mini_train_dota')
    args = parser.parse_args()

    dicts = get_radar_dicts(args.root_src, args.root_dst)