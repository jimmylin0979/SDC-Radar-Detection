#
import mmcv
from mmcv.runner import load_checkpoint
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
import mmdet
from mmdet.apis import inference_detector, show_result_pyplot
import mmrotate
from mmrotate.models import build_detector

#
import os
import cv2
import json
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser


def rbbox2corner(bboxes):
    """Draw oriented bounding boxes on the axes.

    Args:
        ax (matplotlib.Axes): The input axes.
        bboxes (ndarray): The input bounding boxes with the shape
            of (n, 5).
        color (list[tuple] | matplotlib.color): the colors for each
            bounding boxes.
        alpha (float): Transparency of bounding boxes. Default: 0.8.
        thickness (int): Thickness of lines. Default: 2.

    Returns:
        matplotlib.Axes: The result axes.
    """
    polygons = []
    for i, bbox in enumerate(bboxes):
        xc, yc, w, h, ag = bbox[:5]
        wx, wy = w / 2 * np.cos(ag), w / 2 * np.sin(ag)
        hx, hy = -h / 2 * np.sin(ag), h / 2 * np.cos(ag)
        p1 = (xc - wx - hx, yc - wy - hy)
        p2 = (xc + wx - hx, yc + wy - hy)
        p3 = (xc + wx + hx, yc + wy + hy)
        p4 = (xc - wx + hx, yc - wy + hy)
        poly = np.int0(np.array([p1, p2, p3, p4]))
        poly = poly.tolist()
        flatted_poly = []
        for i in range(4):
            flatted_poly.append(poly[i][0])
            flatted_poly.append(poly[i][1])
        polygons.append(flatted_poly)
    # p = PatchCollection(
    #     polygons,
    #     facecolor='none',
    #     edgecolors=color,
    #     linewidths=thickness,
    #     alpha=alpha)
    # ax.add_collection(p)

    return polygons


def postprocess(bboxes, score_thr=0.3):

    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bboxes)
    ]
    labels = np.concatenate(labels)

    bboxes = np.vstack(bboxes)  # (113, 6)

    if score_thr > 0:
        assert bboxes is not None and bboxes.shape[1] == 6
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        scores = bboxes[:, 5] if bboxes.shape[1] == 6 else None
        labels = labels[inds]
    
    bboxes = rbbox2corner(bboxes)
    for i, bbox in enumerate(bboxes):
        bbox.append(labels[i])
        bbox.append(scores[i])
    return bboxes


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
    
    class_name = ('group_of_pedestrians', 'truck', 'pedestrian', 'van', 'bus', 'car',
           'bicycle')

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
            points = [[max(0, bbox[2 * i]), max(0, bbox[2 *i + 1])] for i in range(4)]
            d = {
                "sample_token": f"{image_id + 1:06d}",
                "points": points, 
                "name": class_name[bbox[-2]],
                "scores": float(bbox[-1]),
            }
            res.append(d)

            # 
            image = draw_rotated_bbox(image, bbox[:8], colors[class_name[bbox[-2]]])
        
        os.makedirs("viz", exist_ok=True)
        cv2.imwrite(os.path.join("viz", f"{image_id + 1:06d}.png"), image)

    res = json.dumps(res, indent=4)
    # with open('predictions.json', 'w', encoding='utf-8') as fw:
    #     json.dump(data, f, ensure_ascii=False, indent=4)
    with open(save_path, "w") as fw:
        fw.write(res)


if __name__ == '__main__':

    """
    python inference.py \
        --config mmrotate/work_dirs/sdc_oriented_reppoints_r50_fpn_40e_dota_ms_le135/sdc_oriented_reppoints_r50_fpn_40e_dota_ms_le135.py \
        --ckpt mmrotate/work_dirs/sdc_oriented_reppoints_r50_fpn_40e_dota_ms_le135/epoch_12.pth \
        --root data/mini_train_dota/test/images
    """

    parser = ArgumentParser()
    parser.add_argument("--config", help="", type=str, required=True)
    parser.add_argument("--ckpt", help="", type=str, required=True)
    parser.add_argument("--root", help="", type=str, required=True)
    # parser.add_argument("--save-path", help="", type=str, default="predictions.json")
    args = parser.parse_args()    

    # build model from loaded config file
    config = mmcv.Config.fromfile(args.config)
    config.model.pretrained = None
    model = build_detector(config.model)

    # load model with certain checkpoint
    device='cuda:0'
    checkpoint = load_checkpoint(model, args.ckpt, map_location=device)

    # set the classes of models for inference
    model.CLASSES = checkpoint['meta']['CLASSES']
    model.cfg = config
    model.to(device)
    model.eval()

    results = []
    listdir = sorted(os.listdir(args.root))
    for file in tqdm(listdir):
        if not file.endswith(".png"):
            continue
        # img = './data/mini_train_dota/test/images/city_7_0_000001.png'
        img = os.path.join(args.root, file)
        result = inference_detector(model, img)
        # show_result_pyplot(model, img, result, score_thr=0.3, palette='dota')

        # [[672, 482, 685, 497, 661, 517, 648, 502, 0.44206986, 3], ...]
        result = postprocess(result)
        results.append(result)

    # 
    save_path = args.config.split(r"/")[-1][:-3]
    save_path = save_path + ".json"
    to_json(results, save_path=save_path, root_image=args.root)