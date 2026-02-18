"""
Training + evaluation pipeline with stratified k-fold CV.
Uses same feature extraction as app.py (via features.py); saves model.pkl for the API.
"""
import argparse
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import preprocess, extract_features


def load_image(path):
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def load_dataset(data_dir, benign_subdir="benign", malignant_subdir="malignant"):
    """Load images from data_dir/benign and data_dir/malignant; return X, y."""
    data_dir = Path(data_dir)
    X_list, y_list = [], []
    for label, subdir in enumerate([benign_subdir, malignant_subdir]):
        folder = data_dir / subdir
        if not folder.is_dir():
            continue
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            for path in folder.glob(ext):
                try:
                    img = load_image(path)
                    gray = preprocess(img)
                    feats = extract_features(gray)
                    X_list.append(feats)
                    y_list.append(label)
                except Exception as e:
                    print(f"Skip {path}: {e}")
    if not X_list:
        raise FileNotFoundError(
            f"No images found under {data_dir / benign_subdir} or {data_dir / malignant_subdir}"
        )
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32)


def main():
    parser = argparse.ArgumentParser(description="Train breast cancer classifier with stratified CV")
    parser.add_argument("--data-dir", type=str, default="data", help="Root directory with benign/ and malignant/")
    parser.add_argument("--out", type=str, default="model.pkl", help="Output model path")
    parser.add_argument("--n-folds", type=int, default=5, help="Stratified k-fold splits")
    parser.add_argument("--n-estimators", type=int, default=100, help="RandomForest n_estimators")
    args = parser.parse_args()

    print("Loading dataset...")
    X, y = load_dataset(args.data_dir)
    print(f"Loaded {len(X)} samples, {X.shape[1]} features.")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=args.n_estimators, random_state=42)),
    ])
    cv = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)

    print(f"Running stratified {args.n_folds}-fold CV...")
    scores = cross_validate(
        pipeline, X, y, cv=cv, scoring=("accuracy", "balanced_accuracy", "roc_auc"), n_jobs=-1
    )
    for name, vals in scores.items():
        if name.startswith("test_"):
            print(f"  {name}: {vals.mean():.4f} (+/- {vals.std() * 2:.4f})")

    print("Fitting on full dataset and saving model...")
    pipeline.fit(X, y)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
