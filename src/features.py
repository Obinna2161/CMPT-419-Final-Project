from PIL import Image
import torch
import numpy as np

def crop_image(img: Image.Image, bbox):
    """
    Crop the image using [x_min, y_min, x_max, y_max] in pixel coords.
    """
    x_min, y_min, x_max, y_max = bbox
    return img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

@torch.no_grad()
def encode_pair(img_crop, phrase, model, processor, device):
    """
    Given a cropped image and a phrase, return a feature vector by
    concatenating CLIP image and text embeddings.
    """
    inputs = processor(
        text=[phrase],
        images=[img_crop],
        return_tensors="pt",
        padding=True
    ).to(device)

    outputs = model(**inputs)
    img_emb = outputs.image_embeds      # (1, d)
    txt_emb = outputs.text_embeds       # (1, d)

    feat = torch.cat([img_emb, txt_emb], dim=-1)  # (1, 2d)
    return feat.cpu().numpy().squeeze(0)          # (2d,)