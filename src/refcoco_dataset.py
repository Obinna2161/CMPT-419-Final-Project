from torch.utils.data import Dataset
from PIL import Image

class RefCOCODataset(Dataset):
    """
    Flattens moondream/refcoco-m:
    Each item = one (image, phrase, bbox, image_id).
    """

    def __init__(self, hf_split):
        """
        hf_split: a HuggingFace Dataset split (train_split or val_split).
        """
        self.hf_split = hf_split
        self.index = []  # list of (row_idx, sample_idx, sent_idx)

        # Build a flat index
        for row_idx, ex in enumerate(hf_split):
            samples = ex["samples"]  # list
            for sample_idx, s in enumerate(samples):
                sentences = s["sentences"]
                for sent_idx, sent in enumerate(sentences):
                    self.index.append((row_idx, sample_idx, sent_idx))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        row_idx, sample_idx, sent_idx = self.index[idx]
        ex = self.hf_split[row_idx]

        image = ex["image"]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        image = image.convert("RGB")
        image_id = ex["image_id"]

        sample = ex["samples"][sample_idx]
        bbox_coco = sample["bbox"]  # [x, y, w, h]
        
        sent_entry = sample["sentences"][sent_idx]
        if isinstance(sent_entry, str):
            sentence = sent_entry
        else:
            sentence = sent_entry.get("raw", str(sent_entry))


        # Convert COCO bbox -> [x_min, y_min, x_max, y_max]
        x, y, w, h = bbox_coco
        x_min, y_min = x, y
        x_max, y_max = x + w, y + h
        bbox = [x_min, y_min, x_max, y_max]

        return {
            "image": image,
            "image_id": image_id,
            "phrase": sentence,
            "bbox": bbox,
        }
