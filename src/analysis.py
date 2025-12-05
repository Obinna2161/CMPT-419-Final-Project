# src/analysis.py

from pathlib import Path
import json

import numpy as np
import pandas as pd
import joblib

from src.phrase_tags import classify_phrase_type


def main():
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data"

    # Load arrays & metadata 
    X_val = np.load(data_dir / "X_val.npy")
    y_val = np.load(data_dir / "y_val.npy")
    meta_val = json.loads((data_dir / "meta_val.json").read_text())

    clf = joblib.load(data_dir / "logistic_regression_clip_refcoco.pkl")

    # Build DataFrame
    df = pd.DataFrame(meta_val)

    # Only evaluate on positive examples (true box for the phrase)
    df_pos = df[df["example_type"] == "pos"].copy()

    # Indices in the original val arrays corresponding to positives
    pos_indices = df_pos.index.to_numpy()

    y_val_pos = y_val[pos_indices]
    X_val_pos = X_val[pos_indices]

    print("Unique labels in positive subset:", np.unique(y_val_pos))

    # Predictions
    y_pred_pos = clf.predict(X_val_pos)

    df_pos["pred"] = y_pred_pos
    df_pos["correct"] = (df_pos["pred"] == y_val_pos)

    # Phrase types
    df_pos["phrase_type"] = df_pos["phrase"].apply(classify_phrase_type)

    print("\nCounts by phrase_type:")
    print(df_pos["phrase_type"].value_counts())

    print("\nAccuracy by phrase_type:")
    acc_by_type = (
        df_pos.groupby("phrase_type")["correct"].mean().sort_values(ascending=False)
    )
    print(acc_by_type)

    # save to CSV for inspection
    df_pos.to_csv(data_dir / "val_results_with_phrase_type.csv", index=False)


if __name__ == "__main__":
    main()
