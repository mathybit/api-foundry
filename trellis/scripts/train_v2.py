"""Train TF-IDF + LinearSVC using split train.txt file."""

import csv
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from preprocessing import InputPreprocessor
from config import CLASS_NAMES, OUTPUT_DIR, ARTIFACT_METADATA, ARTIFACT_VECTORIZER, ARTIFACT_CLASSIFIER, ARTIFACT_RAW_SVC, ARTIFACT_CLASSES, SVC_C, SVC_MAX_ITER, SVC_CLASS_WEIGHT, TFIDF_MAX_FEATURES, TFIDF_MIN_DF, TFIDF_MAX_DF, TFIDF_SUBLINEAR_TF, CALIBRATION_METHOD, CALIBRATION_CV_FOLDS

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
EPISODES_CSV = PROJECT_ROOT / "data" / "episodes.csv"
OUTPUT_DIR = PROJECT_ROOT / OUTPUT_DIR
TRAIN_CATS = [c for c in CLASS_NAMES if c != "other"]


def load_train_dataset():
    """Load training data from the train.txt split file."""
    train_ids = [int(line.strip()) for line in open(SPLITS_DIR / "train.txt") if line.strip()]
    episodes = pd.read_csv(EPISODES_CSV)
    mask = episodes["episode_id"].isin(train_ids)
    texts, labels = [], []
    for _, row in episodes[mask].iterrows():
        filepath = PROJECT_ROOT / row["file_path"]
        with open(filepath, "r", encoding="utf-8") as f:
            texts.append(f.read())
        labels.append(row["label"])
    return texts, labels


def load_other_texts():
    """Load 'other' folder texts for evaluation only."""
    other_ids = [int(line.strip()) for line in open(SPLITS_DIR / "test.txt") if line.strip()]
    episodes = pd.read_csv(EPISODES_CSV)
    mask = episodes["episode_id"].isin(other_ids)
    other_episodes = episodes[mask]
    texts = []
    for _, row in other_episodes.iterrows():
        filepath = PROJECT_ROOT / row["file_path"]
        with open(filepath, "r", encoding="utf-8") as f:
            texts.append(f.read())
    return texts


def train_model():
    log.info("Loading training data from split...")
    texts, labels = load_train_dataset()
    n_per_class = {c: labels.count(c) for c in TRAIN_CATS}
    log.info("Training samples per class: %s", n_per_class)
    log.info("Total training samples: %d", len(texts))

    preprocessor = InputPreprocessor()
    processed = [preprocessor.preprocess(t) for t in texts]
    corpus = [" ".join(tokens) for tokens in processed]

    log.info("Building TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF,
        sublinear_tf=TFIDF_SUBLINEAR_TF,
    )
    X = vectorizer.fit_transform(corpus)
    vocab_size = len(vectorizer.vocabulary_)
    log.info("Vocabulary size: %d", vocab_size)

    log.info("Training LinearSVC with calibration...")
    base_svc = LinearSVC(C=SVC_C, max_iter=SVC_MAX_ITER, class_weight=SVC_CLASS_WEIGHT)
    raw_svc = LinearSVC(C=SVC_C, max_iter=SVC_MAX_ITER, class_weight=SVC_CLASS_WEIGHT)
    calibrated_svc = CalibratedClassifierCV(
        estimator=base_svc,
        method=CALIBRATION_METHOD,
        cv=CALIBRATION_CV_FOLDS,
    )
    calibrated_svc.fit(X, labels)
    raw_svc.fit(X, labels)
    log.info("Training complete.")

    log.info("Running 5-fold stratified cross-validation...")
    cv_scores = cross_val_score(raw_svc, X, labels, cv=StratifiedKFold(5), scoring="accuracy")
    log.info("CV accuracy: %.4f (+/- %.4f)", cv_scores.mean(), cv_scores.std() * 2)

    train_preds = calibrated_svc.predict(X)
    log.info("\nFull training set metrics:")
    log.info(classification_report(labels, train_preds))

    other_texts = load_other_texts()
    if other_texts:
        log.info("Evaluating on 'other' (%d samples)...", len(other_texts))
        other_processed = [preprocessor.preprocess(t) for t in other_texts]
        other_corpus = [" ".join(tokens) for tokens in other_processed]
        other_X = vectorizer.transform(other_corpus)
        other_preds = calibrated_svc.predict(other_X)
        other_probs = calibrated_svc.predict_proba(other_X)
        other_max_probs = np.max(other_probs, axis=1)
        other_classified_as_other = sum(1 for p in other_preds if p == "other")
        log.info("Classified as 'other': %d / %d", other_classified_as_other, len(other_texts))

    return {
        "vectorizer": vectorizer,
        "classifier": calibrated_svc,
        "raw_svc": raw_svc,
        "preprocessor": preprocessor,
        "classes": calibrated_svc.classes_,
        "vocab_size": vocab_size,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
    }


def save_artifacts(results):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    vectorizer = results["vectorizer"]
    classifier = results["classifier"]
    raw_svc = results["raw_svc"]
    classes = results["classes"]
    vocab_size = results["vocab_size"]

    joblib.dump(vectorizer, OUTPUT_DIR / ARTIFACT_VECTORIZER)
    joblib.dump(classifier, OUTPUT_DIR / ARTIFACT_CLASSIFIER)
    joblib.dump(raw_svc, OUTPUT_DIR / ARTIFACT_RAW_SVC)
    joblib.dump(classes, OUTPUT_DIR / ARTIFACT_CLASSES)

    vocab_list = sorted(vectorizer.vocabulary_.items(), key=lambda x: x[1])
    with open(OUTPUT_DIR / "vocab.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "word"])
        for word, idx in vocab_list:
            writer.writerow([idx, word])
    log.info("Vocabulary: %d rows -> vocab.csv", len(vocab_list))

    feature_names = vectorizer.get_feature_names_out()
    with open(OUTPUT_DIR / "idf.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "word", "idf"])
        for i, name in enumerate(feature_names):
            writer.writerow([i, name, float(vectorizer.idf_[i])])
    log.info("IDF values: %d rows -> idf.csv", len(feature_names))

    with open(OUTPUT_DIR / "svm_coef.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["class"] + [f"w{j}" for j in range(vocab_size)]
        writer.writerow(header)
        for ci, cclass in enumerate(classes):
            row = [cclass] + [float(v) for v in raw_svc.coef_[ci]]
            writer.writerow(row)
    log.info("SVC coefficients: %d rows x %d cols -> svm_coef.csv", len(classes), vocab_size)

    with open(OUTPUT_DIR / "svm_intercept.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "intercept"])
        for ci, cclass in enumerate(classes):
            writer.writerow([cclass, float(raw_svc.intercept_[ci])])
    log.info("SVC intercepts: %d rows -> svm_intercept.csv")

    metadata = {
        "vocab_size": vocab_size,
        "n_classes": len(classes),
        "classes": list(classes),
        "cv_accuracy": float(results["cv_mean"]),
        "cv_std": float(results["cv_std"]),
        "SVC_C": SVC_C,
        "SVC_MAX_ITER": SVC_MAX_ITER,
        "SVC_CLASS_WEIGHT": SVC_CLASS_WEIGHT,
        "TFIDF_MAX_FEATURES": TFIDF_MAX_FEATURES,
        "TFIDF_MIN_DF": TFIDF_MIN_DF,
        "TFIDF_MAX_DF": TFIDF_MAX_DF,
        "TFIDF_SUBLINEAR_TF": TFIDF_SUBLINEAR_TF,
        "CALIBRATION_METHOD": CALIBRATION_METHOD,
        "CALIBRATION_CV_FOLDS": CALIBRATION_CV_FOLDS,
    }
    with open(OUTPUT_DIR / ARTIFACT_METADATA, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    log.info("Metadata saved -> metadata.json")
    log.info("All artifacts saved to %s", OUTPUT_DIR)


def main():
    results = train_model()
    save_artifacts(results)
    log.info("=== SUMMARY ===")
    log.info("CV accuracy: %.1f%%", results["cv_mean"] * 100)
    log.info("Training completed from split file (data/splits/train.txt)")


if __name__ == "__main__":
    main()
