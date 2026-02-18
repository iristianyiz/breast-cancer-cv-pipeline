"""
Training + evaluation pipeline with stratified k-fold CV.
Uses same feature extraction as app.py (via features.py); saves model.pkl for the API.
Benchmarks: LogisticRegression, RandomForest, XGBoost.
"""
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from features import preprocess, extract_features


@dataclass
class Sample:
    path: str
    y: int  # 0 = benign, 1 = malignant
    id_group: str  # e.g., patient_id or case_id to prevent leakage (if available)


# -------------------------
# Data loading
# -------------------------
def load_image(path: str) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def load_dataset(samples: List[Sample]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load images from a list of Sample; return X, y, paths."""
    X, y, paths = [], [], []
    for s in samples:
        try:
            img = load_image(s.path)
            gray = preprocess(img)
            feats = extract_features(gray)
            X.append(feats)
            y.append(s.y)
            paths.append(s.path)
        except Exception as e:
            print(f"Skip {s.path}: {e}")
    if not X:
        raise FileNotFoundError("No valid samples loaded.")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), paths


def load_dataset_from_dir(
    data_dir: str,
    benign_subdir: str = "benign",
    malignant_subdir: str = "malignant",
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Build Sample list from data_dir/benign and data_dir/malignant; return X, y, paths."""
    data_path = Path(data_dir)
    samples: List[Sample] = []
    for label, subdir in enumerate([benign_subdir, malignant_subdir]):
        folder = data_path / subdir
        if not folder.is_dir():
            continue
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            for path in folder.glob(ext):
                samples.append(Sample(path=str(path), y=label, id_group=path.stem))
    return load_dataset(samples)


# -------------------------
# Model definitions
# -------------------------
def get_models() -> Dict[str, Pipeline]:
    """
    Returns a dict of named pipelines to benchmark.
    - LogisticRegression: linear baseline
    - RandomForest: handles non-linear feature relationships
    - XGBoost: gradient boosted trees, often strong on tabular feature vectors
    """
    return {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )),
        ]),
        "XGBoost": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                scale_pos_weight=1,  # set to (num_benign / num_malignant) for imbalance
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
            )),
        ]),
    }


# -------------------------
# Evaluation
# -------------------------
def evaluate_stratified_cv(
    X: np.ndarray,
    y: np.ndarray,
    paths: List[str],
    k: int = 5,
) -> Dict[str, Dict[str, float]]:
    """Run stratified k-fold CV for each model; print per-fold accuracy, confusion matrix, report, and comparison."""
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    models = get_models()
    results: Dict[str, Dict[str, float]] = {}

    for model_name, pipe in models.items():
        print(f"\n{'='*40}")
        print(f"Model: {model_name}")
        print(f"{'='*40}")

        accs = []
        all_true: List[int] = []
        all_pred: List[int] = []
        hard_cases: List[str] = []
        va_list = list(skf.split(X, y))

        for fold, (tr, va) in enumerate(va_list, start=1):
            pipe = get_models()[model_name]  # fresh pipeline per fold
            pipe.fit(X[tr], y[tr])
            pred = pipe.predict(X[va])

            acc = accuracy_score(y[va], pred)
            accs.append(acc)
            all_true.extend(y[va].tolist())
            all_pred.extend(pred.tolist())

            for local_i, global_i in enumerate(va):
                if pred[local_i] != y[global_i]:
                    hard_cases.append(paths[global_i])

            print(f"  Fold {fold} accuracy: {acc:.4f}")

        print(f"\nConfusion matrix:\n{confusion_matrix(all_true, all_pred)}")
        print(f"\nClassification report:\n{classification_report(all_true, all_pred, digits=4)}")

        results[model_name] = {
            "cv_mean_acc": float(np.mean(accs)),
            "cv_std_acc": float(np.std(accs)),
            "num_misclassified": float(sum(t != p for t, p in zip(all_true, all_pred))),
        }

    # -------------------------
    # Summary comparison
    # -------------------------
    print(f"\n{'='*40}")
    print("Model comparison summary:")
    print(f"{'='*40}")
    for name, stats in results.items():
        print(
            f"  {name:25s}  mean_acc={stats['cv_mean_acc']:.4f}  "
            f"std={stats['cv_std_acc']:.4f}  misclassified={int(stats['num_misclassified'])}"
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train breast cancer classifier with stratified CV")
    parser.add_argument("--data-dir", type=str, default="data", help="Root directory with benign/ and malignant/")
    parser.add_argument("--out", type=str, default="model.pkl", help="Output model path")
    parser.add_argument("--n-folds", type=int, default=5, help="Stratified k-fold splits")
    parser.add_argument(
        "--model",
        type=str,
        choices=["LogisticRegression", "RandomForest", "XGBoost"],
        default="RandomForest",
        help="Which model to fit on full data and save",
    )
    args = parser.parse_args()

    print("Loading dataset...")
    X, y, paths = load_dataset_from_dir(args.data_dir)
    print(f"Loaded {len(X)} samples, {X.shape[1]} features.")

    print(f"Running stratified {args.n_folds}-fold CV for all models...")
    results = evaluate_stratified_cv(X, y, paths, k=args.n_folds)

    # Fit chosen model on full data and save
    pipe = get_models()[args.model]
    print(f"\nFitting {args.model} on full dataset and saving...")
    pipe.fit(X, y)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
