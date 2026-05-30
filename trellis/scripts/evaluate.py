"""Evaluate trained model on train and test splits using DocumentClassifier."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.evaluation import get_confusion_matrix, compute_precision_recall
from model import DocumentClassifier
from config import OUTPUT_DIR, CLASS_NAMES

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
EPISODES_CSV = PROJECT_ROOT / "data" / "episodes.csv"


def load_split_ids(split_name):
    return [int(line.strip())
            for line in open(SPLITS_DIR / f"{split_name}.txt")
            if line.strip()]


def evaluate(split_name, model, ids, episodes, n_true_classes, is_train=False):
    """Evaluate model on a set of episode IDs and print metrics."""
    mask = episodes["episode_id"].isin(ids)
    subset = episodes[mask]

    y_true, y_pred = [], []
    for _, row in subset.iterrows():
        filepath = PROJECT_ROOT / row["file_path"]
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        result = model.classify(text)
        pred_id = CLASS_NAMES.index("other") if result["category"] == "other" else CLASS_NAMES.index(result["category"])
        y_true.append(row["class_id"])
        y_pred.append(pred_id)

    n_classes = len(CLASS_NAMES)
    cm = get_confusion_matrix(y_true, y_pred, n_classes=n_classes)
    metrics = compute_precision_recall(cm, balanced=True)

    # Accuracy
    if is_train:
        correct = int(sum(cm[i, i] for i in range(n_true_classes)))
        total = int(sum(cm[i, :] for i in range(n_true_classes)).sum())
    else:
        correct = int(sum(cm[i, i] for i in range(n_classes)))
        total = int(sum(cm[i, :] for i in range(n_classes)).sum())
    accuracy = correct / total if total > 0 else 0.0

    print(f"\n=== {split_name.upper()} Evaluation ({n_true_classes}-class) ===")
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")

    # Confusion matrix
    longest_name = max(len(c) for c in CLASS_NAMES)
    longest_val = max(f"{cm[i, j]}".__len__() for i in range(n_true_classes) for j in range(n_classes))
    col_width = max(longest_name, longest_val) + 2

    row_width = max(longest_name, len("Confusion Matrix")) + 2

    print("\nConfusion Matrix (true x predicted):")
    header = f"{'':>{row_width}s}"
    for j in range(n_classes):
        header += f"{CLASS_NAMES[j]:>{col_width}s}"
    print(header)

    for i in range(n_true_classes):
        row = f"{CLASS_NAMES[i]:>{row_width}s}"
        for j in range(n_classes):
            row += f"{cm[i, j]:>{col_width}d}"
        print(row)

    # Per-class metrics
    header = f"{'Class':<15s} {'Precision':>9s} {'Recall':>7s}"
    print(f"\n{header}")
    print("-" * len(header))
    for cls_idx in range(n_true_classes):
        pre = metrics['pre'][cls_idx]
        rec = metrics['rec'][cls_idx]
        print(f"{CLASS_NAMES[cls_idx]:<15s} {pre:>9.3f} {rec:>7.3f}")


def main():
    model = DocumentClassifier()
    episodes = pd.read_csv(EPISODES_CSV)

    train_ids = load_split_ids("train")
    evaluate("train", model, train_ids, episodes, len(CLASS_NAMES[:-1]), is_train=True)  # classes 0-9, excluding "other"

    test_ids = load_split_ids("test")
    evaluate("test", model, test_ids, episodes, len(CLASS_NAMES), is_train=False)  # all classes 0-10, including "other"


if __name__ == "__main__":
    main()
