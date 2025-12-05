from PIL import Image
import torch
import numpy as np

def crop_image(img: Image.Image, bbox):
    """
    Crop the image using [x_min, y_min, x_max, y_max] in pixel coords.
    """
    x_min, y_min, x_max, y_max = bbox
    return img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
