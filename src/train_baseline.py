import os
import json
import numpy as np
import torch

from datasets import load_dataset
from transformers import CLIPModel, CLIPProcessor

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# --- Local imports from src/ ---
from src.refcoco_dataset import RefCOCODataset
from src.build_binary_dataset import build_binary_dataset


def main():
    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_dir = os.path.join(os.getcwd(), "data")
    os.makedirs(data_dir, exist_ok=True)

    # 2. Load CLIP model + processor
    print("Loading CLIP model...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()

    # 3. Load RefCOCO-m splits
    print("Loading RefCOCO-m dataset...")
    ds = load_dataset("moondream/refcoco-m")
    base = ds["validation"]
    split = base.train_test_split(test_size=0.2, seed=42)
    train_split = split["train"]
    val_split = split["test"]

    print(f"Train split size: {len(train_split)}")
    print(f"Val split size:   {len(val_split)}")

    # 4. Build binary datasets (image+phrase → CLIP features → X, y, meta)
    print("Building binary training dataset...")
    X_train, y_train, meta_train = build_binary_dataset(
        train_split,
        clip_model,
        clip_processor,
        device,
        max_examples=800,
        neg_per_pos=2,
        seed=42,
    )

    print("Building binary validation dataset...")
    X_val, y_val, meta_val = build_binary_dataset(
        val_split,
        clip_model,
        clip_processor,
        device,
        max_examples=200,
        neg_per_pos=2,
        seed=123,
    )

    print("X_train:", X_train.shape, " y_train:", y_train.shape)
    print("X_val:  ", X_val.shape,   " y_val:  ", y_val.shape)

    # 5. Save arrays + metadata
    print("Saving arrays and metadata to 'data/'...")

    np.save(os.path.join(data_dir, "X_train.npy"), X_train)
    np.save(os.path.join(data_dir, "y_train.npy"), y_train)
    np.save(os.path.join(data_dir, "X_val.npy"),   X_val)
    np.save(os.path.join(data_dir, "y_val.npy"),   y_val)

    with open(os.path.join(data_dir, "meta_train.json"), "w") as f:
        json.dump(meta_train, f)

    with open(os.path.join(data_dir, "meta_val.json"), "w") as f:
        json.dump(meta_val, f)

    # 6. Train baseline classifier (logistic regression)
    print("Training logistic regression classifier...")
    clf = LogisticRegression(max_iter=1000, n_jobs=-1)
    clf.fit(X_train, y_train)

    # 7. Evaluate and print accuracies
    train_pred = clf.predict(X_train)
    val_pred   = clf.predict(X_val)

    train_acc = accuracy_score(y_train, train_pred)
    val_acc   = accuracy_score(y_val,   val_pred)

    print(f"Train accuracy: {train_acc:.3f}")
    print(f"Val accuracy:   {val_acc:.3f}")

    # 8. Save classifier
    clf_path = os.path.join(data_dir, "logistic_regression_clip_refcoco.pkl")
    joblib.dump(clf, clf_path)
    print(f"Saved classifier to: {clf_path}")


if __name__ == "__main__":
    main()
