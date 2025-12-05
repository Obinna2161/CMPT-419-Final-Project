# src/build_binary_dataset.py

import numpy as np
from tqdm import tqdm

from src.refcoco_dataset import RefCOCODataset
from src.features import crop_image, encode_pair


def build_binary_dataset(
    hf_split,
    model,
    processor,
    device,
    max_examples=1000,
    neg_per_pos=2,
    seed=42,
):
    """
    hf_split: HuggingFace split (train_split or val_split)

    Returns:
        X         : (N, d) numpy array of features
        y         : (N,)   numpy array of labels (1=pos, 0=neg)
        metadata  : list of dicts, one per row (for later analysis)
    """
    rng = np.random.default_rng(seed)
    X_list = []
    y_list = []
    metadata = []

    ref_ds = RefCOCODataset(hf_split)

    n = min(len(ref_ds), max_examples)
    for i in tqdm(range(n), desc="Building binary dataset"):
        ex = ref_ds[i]
        img = ex["image"]
        phrase = ex["phrase"]
        bbox = ex["bbox"]
        image_id = ex["image_id"]

        # ----- Positive example -----
        pos_crop = crop_image(img, bbox)
        pos_feat = encode_pair(pos_crop, phrase, model, processor, device)

        X_list.append(pos_feat)
        y_list.append(1)
        metadata.append(
            {
                "image_id": image_id,
                "phrase": phrase,
                "example_type": "pos",
                "bbox": bbox,
            }
        )

        # Negative examples
        w, h = img.size
        bw = max(10, int(bbox[2] - bbox[0]))
        bh = max(10, int(bbox[3] - bbox[1]))

        for _ in range(neg_per_pos):
            x_min = rng.integers(0, max(1, w - bw))
            y_min = rng.integers(0, max(1, h - bh))
            x_max = x_min + bw
            y_max = y_min + bh
            neg_bbox = [int(x_min), int(y_min), int(x_max), int(y_max)]

            neg_crop = crop_image(img, neg_bbox)
            neg_feat = encode_pair(neg_crop, phrase, model, processor, device)

            X_list.append(neg_feat)
            y_list.append(0)
            metadata.append(
                {
                    "image_id": image_id,
                    "phrase": phrase,
                    "example_type": "neg",
                    "bbox": neg_bbox,
                }
            )

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)
    return X, y, metadata
