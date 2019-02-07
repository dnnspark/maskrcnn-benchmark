import argparse
import glob
import os
import time

import imageio
# (dnnspark)
import numpy as np
import torch

from torchvision import transforms as T
from matplotlib.pyplot import clf, close, figure, imshow, title, show

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.utils.visualization import (COCO_CATEGORIES,
                                                    DetectionVisualizer)
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.model_serialization import load_state_dict

LOG = setup_logger(__name__, None, 0)

def build_transforms(cfg):

    transforms = [T.ToTensor()] # [0-1]

    if cfg.INPUT.TO_BGR255:
        transforms.append(T.Lambda(lambda x: x[[2, 1, 0]] * 255.))
    else:
        transforms.append(T.Lambda(lambda x: x * 255.))

    mean, std = cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD
    transforms.append(T.Normalize(mean, std))

    return T.Compose(transforms)



def run_on_images(
    cfg,
    image_path_iterator,
    checkpoint_file,
    confidence_threshold,
    device,
    show_mask_heatmaps,
    ):

    # set up model
    model = build_detection_model(cfg)

    # loading weights from checkpoint.
    LOG.info('Loading weights from %s' % checkpoint_file)
    loaded_state_dict = torch.load(checkpoint_file)
    load_state_dict(model, loaded_state_dict['model'])

    model.eval()
    model.to(device)

    transforms = build_transforms(cfg)

    # masker
    mask_threshold = -1 if show_mask_heatmaps else 0.5
    masker = Masker(threshold=mask_threshold, padding=1)

    # run on images one-by-one.
    for image_path in image_path_iterator:
        LOG.info('Processing image: %s' % image_path)
        image = np.array(imageio.imread(image_path)) # RGB255
        transformed_image = transforms(image)
        transformed_image = transformed_image.to(device)

        # convert to an ImageList, padded so that it is divisible by
        # cfg.DATALOADER.SIZE_DIVISIBILITY
        image_list = to_image_list(transformed_image, cfg.DATALOADER.SIZE_DIVISIBILITY)

        start_time = time.time()
        with torch.no_grad():
            prediction = model(image_list)[0]
        elapsed = time.time() - start_time

        # filter and sort using scores.
        keep = torch.nonzero(prediction.get_field("scores") > confidence_threshold).squeeze()
        prediction = prediction[keep]
        _, order = prediction.get_field("scores").sort()
        prediction = prediction[order]

        # reshape prediction (a BoxList) into the original image size
        height, width = image.shape[:-1]
        prediction = prediction.resize((width, height))

        if prediction.has_field("mask"):
            masks = prediction.get_field("mask")
            expanded_masks = masker([masks], [prediction])[0]

            prediction.add_field("mask", expanded_masks)

        yield image, prediction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--checkpoint-file",
        metavar="FILE",
        help="path to checkpoint file.",
    )
    parser.add_argument(
        "--image-folder",
        metavar="DIR",
        help="path to image folder",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.7,
        help="Minimum score for the prediction to be shown",
    )
    parser.add_argument(
        "--min-image-size",
        type=int,
        default=224,
        help="Smallest size of the image to feed to the model. "
            "Model was trained with 800, which gives best results",
    )
    parser.add_argument(
        "--show-mask-heatmaps",
        dest="show_mask_heatmaps",
        help="Show a heatmap probability for the top masks-per-dim masks",
        action="store_true",
    )
    parser.add_argument(
        "--masks-per-dim",
        type=int,
        default=2,
        help="Number of heatmaps per dimension to show",
    )
    parser.add_argument(
        "opts",
        help="Modify model config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    # load config from file and command-line arguments
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    image_folder = args.image_folder
    image_paths = glob.glob(os.path.join(image_folder, '*.jpg'))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    prediction_gen = run_on_images(
        cfg,
        image_paths,
        args.checkpoint_file,
        args.confidence_threshold,
        device,
        args.show_mask_heatmaps
    )

    mask_is_heatmap = args.show_mask_heatmaps

    viz = DetectionVisualizer(num_categories=80)

    for image, prediction in prediction_gen:
        masks, labels = prediction.get_field("mask"), prediction.get_field("labels")
        bboxes = prediction.bbox.cpu().numpy()
        masks = masks.cpu().numpy()
        labels = labels.cpu().numpy()

        close('all')
        for mask_idx, (bbox, mask, label) in enumerate(zip(bboxes, masks, labels)):
            overlaid = viz.overlay_mask(image, mask[0], label, mask_is_heatmap)
            overlaid = viz.overlay_box(overlaid, bbox, label)

            figure(mask_idx+1), clf(), imshow(overlaid), title(COCO_CATEGORIES[label])

        show(block=False)
        input("Press Enter to continue...")

if __name__ == "__main__":
    main()
