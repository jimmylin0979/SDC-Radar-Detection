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

        return points

def draw_boundingbox_rot(im, points, color=(0, 1, 0)):
    # points = gen_boundingbox_rot(bbox, angle)

    color = (np.array(color) * 255).tolist()

    cv2.line(im, tuple(points[:, 0]), tuple(points[:, 1]), color, 3)
    cv2.line(im, tuple(points[:, 1]), tuple(points[:, 2]), color, 3)
    cv2.line(im, tuple(points[:, 2]), tuple(points[:, 3]), color, 3)
    cv2.line(im, tuple(points[:, 3]), tuple(points[:, 0]), color, 3)

    return im

def draw_rotated_bbox(image, position, color=(0, 1, 0)):

    x1, y1, x2, y2, x3, y3, x4, y4 = position
    corners = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.int32)
    corners = corners.reshape((-1, 1, 2))
    color = (np.array(color) * 255).tolist()
    cv2.polylines(image, [corners], isClosed=True, color=color, thickness=2)
    
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

            # indices = list(range(len(radar_files) - 1))
            # test_size = 0.2
            # random_state = 42
            # train_indices, valid_indices = train_test_split(
            #     indices,
            #     test_size=test_size,
            #     # stratify=dataset.targets,
            #     random_state=random_state,
            # )

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
                    angle = annotation['bboxes'][frame_number]['rotation']

                    x, y, w, h = position
                    position_2d = gen_boundingbox_rot([x, y, w, h], angle)
                    position_2d = position_2d.tolist()
                    
                    position = []
                    for i in range(4):
                        position.append(position_2d[0][i])
                        position.append(position_2d[1][i])
                    
                    obj = position
                    obj.append(class_name)
                    obj.append(difficulty)
                    objs.append(obj)

            # 
            os.makedirs(f'{root_dst}', exist_ok=True)
            for mode in ['test']:
                os.makedirs(f'{root_dst}/{mode}/images', exist_ok=True)
                os.makedirs(f'{root_dst}/{mode}/GTs', exist_ok=True)
                os.makedirs(f'{root_dst}/{mode}/annotations', exist_ok=True)
            
            # split the files into train/valid depend on the train_test_spilt result
            mode = 'test'
            shutil.copyfile(filename, os.path.join(f'{root_dst}/{mode}/images', f'{folder}_{radar_files[frame_number]}'))
            
            colors = {'car': (1, 1, 1),
                       'bus': (0, 1, 0),
                       'truck': (0, 0, 1),
                       'pedestrian': (1.0, 1.0, 0.0),
                       'van': (1.0, 0.3, 0.0),
                       'group_of_pedestrians': (1.0, 1.0, 0.3),
                       'motorbike': (0.0, 1.0, 1.0),
                       'bicycle': (0.3, 1.0, 1.0),
                       'vehicle': (1.0, 0.0, 0.0)
            }
            with open(os.path.join(f'{root_dst}/{mode}/annotations', f'{folder}_{radar_files[frame_number][:-4]}.txt'), 'w') as fw:
                image = cv2.imread(os.path.join(os.path.join(f'{root_dst}/{mode}/images', f'{folder}_{radar_files[frame_number]}')))
                for obj in objs:
                    # 
                    image = draw_rotated_bbox(image, obj[:8], colors[obj[8]])

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
    parser.add_argument("--root_src", help="root of the radar dataset folder", type=str, default='mini_test')
    parser.add_argument("--root_dst", help="root of the destination folder", type=str, default='mini_train_dota')
    args = parser.parse_args()

    dicts = get_radar_dicts(args.root_src, args.root_dst)