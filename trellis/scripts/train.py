"""Train TF-IDF + LinearSVC document classifier and export all artifacts."""

import os
import csv
import json
import logging
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report
import joblib

import sys

# Ensure trellis package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from trellis.preprocessing import InputPreprocessor
from trellis import config

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "data", "dataset")
OTHER_DIR = os.path.join(DATASET_DIR, "other")
OUTPUT_DIR = os.path.join(BASE_DIR, config.OUTPUT_DIR)

TRAIN_CATS = config.CLASS_NAMES[:-1]  # omits 'other'


def load_dataset():
    """Load all training data (excludes 'other')."""
    texts, labels = [], []
    for cat in TRAIN_CATS:
        cat_dir = os.path.join(DATASET_DIR, cat)
        for fname in sorted(os.listdir(cat_dir)):
            if not fname.endswith(".txt"):
                continue
            with open(os.path.join(cat_dir, fname), "r", encoding="utf-8") as f:
                texts.append(f.read())
            labels.append(cat)
    return texts, labels


def load_other_texts():
    """Load 'other' folder texts for evaluation only (never training)."""
    texts = []
    if not os.path.isdir(OTHER_DIR):
        return texts
    for fname in sorted(os.listdir(OTHER_DIR)):
        if fname.endswith(".txt"):
            with open(os.path.join(OTHER_DIR, fname), "r", encoding="utf-8") as f:
                texts.append(f.read())
    return texts


def train_model():
    log.info("Loading training data...")
    texts, labels = load_dataset()
    n_per_class = {c: labels.count(c) for c in TRAIN_CATS}
    log.info("Training samples per class: %s", n_per_class)
    log.info("Total training samples: %d", len(texts))

    # Preprocess + join tokens into space-separated strings for TfidfVectorizer
    preprocessor = InputPreprocessor()
    processed = [preprocessor.preprocess(t) for t in texts]
    corpus = [" ".join(tokens) for tokens in processed]

    # Build TF-IDF features
    log.info("Building TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=config.TFIDF_MAX_FEATURES,
        min_df=config.TFIDF_MIN_DF,
        max_df=config.TFIDF_MAX_DF,
        sublinear_tf=config.TFIDF_SUBLINEAR_TF,
    )
    X = vectorizer.fit_transform(corpus)
    vocab_size = len(vectorizer.vocabulary_)
    log.info("Vocabulary size: %d", vocab_size)

    # Train raw SVC (for coefficients) and calibrated SVC (for probabilities)
    log.info("Training LinearSVC with calibration...")
    base_svc = LinearSVC(
        C=config.SVC_C,
        max_iter=config.SVC_MAX_ITER,
        class_weight=config.SVC_CLASS_WEIGHT,
    )
    raw_svc = LinearSVC(
        C=config.SVC_C,
        max_iter=config.SVC_MAX_ITER,
        class_weight=config.SVC_CLASS_WEIGHT,
    )
    calibrated_svc = CalibratedClassifierCV(
        estimator=base_svc,
        method=config.CALIBRATION_METHOD,
        cv=config.CALIBRATION_CV_FOLDS,
    )
    calibrated_svc.fit(X, labels)
    raw_svc.fit(X, labels)
    log.info("Training complete.")

    # Cross-validation accuracy
    log.info("Running 5-fold stratified cross-validation...")
    cv_scores = cross_val_score(raw_svc, X, labels, cv=StratifiedKFold(5), scoring="accuracy")
    log.info("CV accuracy: %.4f (+/- %.4f)", cv_scores.mean(), cv_scores.std() * 2)

    # Full training set metrics (for inspection only)
    train_preds = calibrated_svc.predict(X)
    log.info("\nFull training set metrics:")
    log.info(classification_report(labels, train_preds))

    # Evaluate on 'other' folder
    other_texts = load_other_texts()
    if other_texts:
        log.info("Evaluating on 'other' folder (%d samples)...", len(other_texts))
        other_processed = [preprocessor.preprocess(t) for t in other_texts]
        other_corpus = [" ".join(tokens) for tokens in other_processed]
        other_X = vectorizer.transform(other_corpus)
        other_preds = calibrated_svc.predict(other_X)
        other_probs = calibrated_svc.predict_proba(other_X)
        other_max_probs = np.max(other_probs, axis=1)

        log.info("Other predictions: %s", other_preds)
        log.info("Other max confidences: %s", [f"{p:.4f}" for p in other_max_probs])
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
        "other_texts": other_texts,
        "other_preds": other_preds if other_texts else None,
        "other_max_probs": other_max_probs if other_texts else None,
    }


def save_artifacts(results):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    vectorizer = results["vectorizer"]
    classifier = results["classifier"]
    raw_svc = results["raw_svc"]
    classes = results["classes"]
    vocab_size = results["vocab_size"]

    # Save via joblib (full objects, numpy arrays included)
    joblib.dump(vectorizer, os.path.join(OUTPUT_DIR, config.ARTIFACT_VECTORIZER))
    joblib.dump(classifier, os.path.join(OUTPUT_DIR, config.ARTIFACT_CLASSIFIER))
    joblib.dump(raw_svc, os.path.join(OUTPUT_DIR, config.ARTIFACT_RAW_SVC))
    joblib.dump(classes, os.path.join(OUTPUT_DIR, config.ARTIFACT_CLASSES))

    # Export vocabulary to CSV (word -> column index)
    vocab_list = sorted(vectorizer.vocabulary_.items(), key=lambda x: x[1])
    with open(os.path.join(OUTPUT_DIR, config.ARTIFACT_VOCAB), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "word"])
        for word, idx in vocab_list:
            writer.writerow([idx, word])
    log.info("Vocabulary: %d rows -> vocab.csv", len(vocab_list))

    # Export IDF to CSV
    feature_names = vectorizer.get_feature_names_out()
    with open(os.path.join(OUTPUT_DIR, config.ARTIFACT_IDF), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "word", "idf"])
        for i, name in enumerate(feature_names):
            writer.writerow([i, name, float(vectorizer.idf_[i])])
    log.info("IDF values: %d rows -> idf.csv", len(feature_names))

    # Export SVC coefficients to CSV (coef_ = (n_classes, vocab_size))
    with open(os.path.join(OUTPUT_DIR, config.ARTIFACT_SVM_COEF), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["class"] + [f"w{j}" for j in range(vocab_size)]
        writer.writerow(header)
        for ci, cclass in enumerate(classes):
            row = [cclass] + [float(v) for v in raw_svc.coef_[ci]]
            writer.writerow(row)
    log.info("SVC coefficients: %d rows x %d cols -> svm_coef.csv", len(classes), vocab_size)

    # Export SVC intercept to CSV
    with open(os.path.join(OUTPUT_DIR, config.ARTIFACT_SVM_INTERCEPT), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "intercept"])
        for ci, cclass in enumerate(classes):
            writer.writerow([cclass, float(raw_svc.intercept_[ci])])
    log.info("SVC intercepts: %d rows -> svm_intercept.csv", len(classes))

    # Save training metadata
    metadata = {
        "vocab_size": vocab_size,
        "n_classes": len(classes),
        "classes": list(classes),
        "cv_accuracy": float(results["cv_mean"]),
        "cv_std": float(results["cv_std"]),
        "SVC_C": config.SVC_C,
        "SVC_MAX_ITER": config.SVC_MAX_ITER,
        "SVC_CLASS_WEIGHT": config.SVC_CLASS_WEIGHT,
        "TFIDF_MAX_FEATURES": config.TFIDF_MAX_FEATURES,
        "TFIDF_MIN_DF": config.TFIDF_MIN_DF,
        "TFIDF_MAX_DF": config.TFIDF_MAX_DF,
        "TFIDF_SUBLINEAR_TF": config.TFIDF_SUBLINEAR_TF,
        "CALIBRATION_METHOD": config.CALIBRATION_METHOD,
        "CALIBRATION_CV_FOLDS": config.CALIBRATION_CV_FOLDS,
    }
    with open(os.path.join(OUTPUT_DIR, config.ARTIFACT_METADATA), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    log.info("Metadata saved -> metadata.json")

    log.info("All artifacts saved to %s", OUTPUT_DIR)


def main():
    results = train_model()
    save_artifacts(results)

    # Final summary
    other_texts = results["other_texts"]
    if other_texts and results["other_preds"] is not None:
        other_classified_as_other = sum(1 for p in results["other_preds"] if p == "other")
        log.info("=== SUMMARY ===")
        log.info("CV accuracy: %.1f%%", results["cv_mean"] * 100)
        log.info("Other folder: %d / %d classified as 'other'",
                 other_classified_as_other, len(other_texts))
        log.info(f"Threshold used: {config.DEFAULT_THRESHOLD} (see model.py for configurable threshold)")


if __name__ == "__main__":
    main()
