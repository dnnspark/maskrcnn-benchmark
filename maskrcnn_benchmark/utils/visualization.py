# (dnnspark)
import cv2
import numpy as np
import seaborn as sns
from skimage.color import rgb2gray


def normalize_image(image, divide_255=True):
    if image.dtype in [np.float32, np.float64]:
        return image.astype(np.float32)
    elif image.dtype == np.uint8:
        image = image.astype(np.float32)
        if divide_255:
            image /= 255.
        return image
    else:
        raise ValueError("Invalid dtype.")


class DetectionVisualizer():

    def __init__(self, num_categories, alpha=0.4):
        self.colors = np.array(sns.color_palette("husl", num_categories))
        self.alpha = alpha

    def overlay_mask(self, image, mask, label, mask_is_heatmap):
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        alpha = self.alpha
        image = rgb2gray(image)
        # bump the brightness
        image = np.clip(image + 0.1, 0., 1.)
        mask = normalize_image(mask, mask_is_heatmap) # [0-255] if mask_is_heatmap, else [0-1]
        blended = image[...,None] * (1. - alpha) + np.stack([mask,mask,mask], axis=-1) * self.colors[label] * alpha
        if not mask_is_heatmap:
            blended = cv2.drawContours(blended, contours, -1, self.colors[label], 2)
        return blended

    def overlay_box(self, image, bbox, label):
        top_left, bot_right = bbox[:2], bbox[2:]
        overlaid = cv2.rectangle(image, tuple(top_left), tuple(bot_right), self.colors[label], 2)
        return overlaid



COCO_CATEGORIES = [
    "__background",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]
