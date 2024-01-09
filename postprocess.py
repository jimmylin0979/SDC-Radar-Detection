import os
import cv2
import numpy as np
import json
import math
from tqdm import tqdm
from argparse import ArgumentParser

def load_annotations(path: str):    
    bboxes = None
    with open(path, "r") as fr:
        bboxes = json.load(fr)
    return bboxes

def postprocess_eraseGhost(bboxes_in_frame):
    
    def getCenterOfCircle(bbox):
        # [[517, 748], [518, 719], [536, 720], [535, 749]]
        x1, y1 = bbox[0]
        x2, y2 = bbox[2]
        return int((x1 + x2) / 2), int((y1 + y2) / 2)
    
    def isCollision(cc_prev_bbox, cc_bbox, circle_diameter = 30.0):
        distance = (cc_prev_bbox[0] - cc_bbox[0]) ** 2 + (cc_prev_bbox[1] - cc_bbox[1]) ** 2
        distance = math.sqrt(distance)
        if distance < circle_diameter:
            return True
        return False

    def isInherit(bboxes_in_next_frame, bbox):
        for prev_bbox_i, prev_bbox in enumerate(bboxes_in_next_frame):
            cc_prev_bbox = getCenterOfCircle(prev_bbox["points"])
            cc_bbox = getCenterOfCircle(bbox["points"])
            if isCollision(cc_prev_bbox, cc_bbox):
                return prev_bbox_i
        return -2 
    
    def getCloserBBox(parent_bbox, bbox1, bbox2):

        def dist(cc1, cc2):
            return (cc1[0] - cc2[0]) ** 2 + (cc1[1] - cc2[1]) ** 2

        cc_box1 = getCenterOfCircle(bbox1["points"])
        cc_box2 = getCenterOfCircle(bbox2["points"])
        cc_parent_box = getCenterOfCircle(parent_bbox["points"])

        if dist(cc_parent_box, cc_box1) < dist(cc_parent_box, cc_box2):
            return bbox1
        return bbox2

    # 
    num_frames = len(bboxes_in_frame)
    for frame in tqdm(range(num_frames - 1)):

        filtered_bboxs = []
        filtered_bboxs_parent = []
        for bbox in bboxes_in_frame[frame]:
            # TODO which frames ?
            # print(bboxes_in_frame[frame - 1])
            isInherit_prev = -1 if frame == num_frames - 1 else isInherit(bboxes_in_frame[frame + 1], bbox)
            isInherit_next = -1 if frame == 0 else isInherit(bboxes_in_frame[frame - 1], bbox)
            
            if isInherit_next != -2 or isInherit_prev != -2:

                # if isInherit_prev in filtered_bboxs_parent:
                #     for filtered_bbox_i, filtered_bbox in enumerate(filtered_bboxs):
                #         parent_bbox_i = filtered_bboxs_parent[filtered_bbox_i]
                #         if parent_bbox_i == isInherit_prev:
                #             bbox = getCloserBBox(bboxes_in_frame[frame + 1][parent_bbox_i], filtered_bbox, bbox)
                #             filtered_bboxs[filtered_bbox_i] = bbox
                #             break
                # else:
                filtered_bboxs.append(bbox)
                filtered_bboxs_parent.append(isInherit_prev if isInherit_prev != -2 else isInherit_next)
        
        # print(bboxes_in_frame[frame], filtered_bbox)
        bboxes_in_frame[frame] = filtered_bboxs
    
    return bboxes_in_frame

# # global
# num_objects = 0

class TrackingItem():
    def __init__(self, ):
        self.p_x = 0
        self.p_y = 0
        self.v_x = 0
        self.v_y = 0
        self.v_theta = 0
    
    def update(self, p_x, p_y):
        self.v_x = p_x - self.p_x
        self.v_y = p_y - self.p_y
        self.p_x = p_x
        self.p_y = p_y
    
    def predict(self):
        return self.p_x + self.v_x, self.p_y + self._v_y

def postprocess_interpolation(bboxes_in_frame):

    def interpolation(bbox_in_prev_frame, bbox_in_cur_frame):
        pass

    num_frames = len(bboxes_in_frame)
    for frame in tqdm(range(num_frames)):
        for bbox in bboxes_in_frame[frame]:
            pass

def draw_rotated_bbox(image, position, color=(0, 1, 0)):

    x1, y1, x2, y2, x3, y3, x4, y4 = position
    corners = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.int32)
    corners = corners.reshape((-1, 1, 2))
    color = (np.array(color) * 255).tolist()
    cv2.polylines(image, [corners], isClosed=True, color=color, thickness=2)
    
    return image

def to_json(bboxes, save_path: str, root_image: str = ""):

    """
    [
        {
            "sample_token": "000001",
            "points": [
                [
                    787,
                    1052
                ],
                [
                    787,
                    1045
                ],
                [
                    804,
                    1045
                ],
                [
                    804,
                    1052
                ]
            ],
            "name": "scooter",
            "scores": 0.842
        },
        {
            "sample_token": "000002",
            "points": [
                [
                    771,
                    1037
                ],
                [
                    771,
                    1025
                ],
                [
                    792,
                    1025
                ],
                [
                    792,
                    1037
                ]
            ],
            "name": "car",
            "scores": 0.783
        },
    ]
    """
    
    # class_name = ('group_of_pedestrians', 'truck', 'pedestrian', 'van', 'bus', 'car',
    #        'bicycle')
    class_name = ('car', )

    colors = {
        'car': (1, 1, 1),
        'bus': (0, 1, 0),
        'truck': (0, 0, 1),
        'pedestrian': (1.0, 1.0, 0.0),
        'van': (1.0, 0.3, 0.0),
        'group_of_pedestrians': (1.0, 1.0, 0.3),
        'motorbike': (0.0, 1.0, 1.0),
        'bicycle': (0.3, 1.0, 1.0),
        'vehicle': (1.0, 0.0, 0.0)
    }
        

    res = []
    for image_id, bboxes_in_image in tqdm(enumerate(bboxes)):
        
        image = cv2.imread(os.path.join(root_image, f"{image_id + 1:06d}.png"))
        for bbox_id, bbox in enumerate(bboxes_in_image):

            # bbox = [672, 482, 685, 497, 661, 517, 648, 502, 3, 0.44206986]
            # points = bbox["points"]
            points = []
            for i in bbox["points"]:
                points.append(i[0])
                points.append(i[1])
            bbox["sample_token"] = f"{bbox['sample_token']:06d}"
            res.append(bbox)

            # 
            image = draw_rotated_bbox(image, points, colors[bbox["name"]])

        os.makedirs("viz", exist_ok=True)
        cv2.imwrite(os.path.join("viz", f"{image_id + 1:06d}.png"), image)

    res = json.dumps(res, indent=4)
    # with open('predictions.json', 'w', encoding='utf-8') as fw:
    #     json.dump(data, f, ensure_ascii=False, indent=4)
    with open(save_path, "w") as fw:
        fw.write(res)

if __name__ == "__main__":

    #
    parser = ArgumentParser()
    parser.add_argument("--json-path", help="", type=str, required=True)
    parser.add_argument("--save-path", help="", type=str, required=True)
    parser.add_argument("--root-image", help="", type=str, required=True)
    # parser.add_argument("--save-path", help="", type=str, default="predictions.json")
    args = parser.parse_args()    
    
    # 
    # path = "./results/redet_re50_refpn_1x_dota_ms_rr_le90_batch2/20240103_rotated_redet_re50_refpn_1x_dota_ms_rr_le90.json"
    path = args.json_path
    bboxes = load_annotations(path)
    # print(bboxes)

    # 
    num_frames = 722
    bboxes_in_frame = [[] for _ in range(num_frames)]
    for bbox in bboxes:
        # 
        sample_token = int(bbox["sample_token"])
        bbox["sample_token"] = int(bbox["sample_token"])
        points = bbox["points"]
        name = bbox["name"]
        scores = bbox["scores"]

        # 
        bboxes_in_frame[sample_token - 1].append(bbox)
    # print(bboxes_in_frame)

    # erase ghost
    bboxes_in_frame = postprocess_eraseGhost(bboxes_in_frame)
    # print(bboxes_in_frame)

    # interpolation
    # bboxes_in_frame = postprocess_interpolation(bboxes_in_frame)
    # print(bboxes_in_frame)
    
    # 
    # save_path = "predictionsspetition_Image_preprocessed"
    save_path = args.save_path
    root_image = args.root_image
    to_json(bboxes_in_frame, save_path=save_path, root_image=root_image)
