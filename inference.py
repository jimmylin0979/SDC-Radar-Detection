#
import mmcv
from mmcv.runner import load_checkpoint
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
import mmdet
from mmdet.apis import inference_detector, show_result_pyplot
import mmrotate
from mmrotate.models import build_detector

#
import numpy as np
from matplotlib.patches import Polygon
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

    bboxes = np.vstack(bboxes)  # (113, 6)

    if score_thr > 0:
        assert bboxes is not None and bboxes.shape[1] == 6
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        scores = bboxes[:, 5] if bboxes.shape[1] == 6 else None
    
    bboxes = rbbox2corner(bboxes)
    return bboxes

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("--config", help="", type=str, required=True)
    parser.add_argument("--ckpt", help="", type=str, required=True)
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

    img = './data/mini_train_dota/test/images/city_7_0_000001.png'
    result = inference_detector(model, img)
    result = postprocess(result)
    print(result)
    # show_result_pyplot(model, img, result, score_thr=0.3, palette='dota')