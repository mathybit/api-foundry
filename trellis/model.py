"""Document classifier for inference. Loads trained artifacts and predicts categories."""

import os
import sys
import json
import logging
import numpy as np
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessing import InputPreprocessor
from config import DEFAULT_THRESHOLD, OUTPUT_DIR, ARTIFACT_VECTORIZER, ARTIFACT_CLASSIFIER, ARTIFACT_CLASSES, ARTIFACT_METADATA

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

DEFAULT_THRESHOLD = DEFAULT_THRESHOLD
ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_DIR)


class DocumentClassifier:
    """Load and run a trained TF-IDF + LinearSVC document classifier."""

    def __init__(self, artifacts_dir=None, threshold=None):
        self.threshold = threshold or DEFAULT_THRESHOLD
        self.artifacts_dir = artifacts_dir or ARTIFACTS_DIR
        self._load()

    def _load(self):
        log.info("Loading artifacts from %s", self.artifacts_dir)

        # Load vectorizer + classifier
        self.vectorizer = joblib.load(os.path.join(self.artifacts_dir, ARTIFACT_VECTORIZER))
        self.classifier = joblib.load(os.path.join(self.artifacts_dir, ARTIFACT_CLASSIFIER))
        self.classes = joblib.load(os.path.join(self.artifacts_dir, ARTIFACT_CLASSES))

        # Metadata
        meta_path = os.path.join(self.artifacts_dir, ARTIFACT_METADATA)
        with open(meta_path, "r") as f:
            self.metadata = json.load(f)

        self.preprocessor = InputPreprocessor()
        log.info("Loaded classifier: %d classes, vocab=%d, cv_accuracy=%.4f",
                 len(self.classes), self.metadata["vocab_size"],
                 self.metadata["cv_accuracy"])

    def classify(self, text: str) -> dict:
        """Classify a single document. Returns prediction dict."""
        tokens = self.preprocessor.preprocess(text)
        token_str = " ".join(tokens)

        vector = self.vectorizer.transform([token_str])
        probs = self.classifier.predict_proba(vector).ravel()

        top_idx = int(np.argmax(probs))
        top_class = str(self.classes[top_idx])
        top_prob = float(probs[top_idx])

        is_other = top_prob < self.threshold

        return {
            "category": top_class if not is_other else "other",
            "confidence": top_prob,
            "threshold": self.threshold,
            "is_other": is_other,
            "probabilities": {
                str(c): float(probs[i])
                for i, c in enumerate(self.classes)
            },
        }

    def classify_batch(self, texts: list[str]) -> list[dict]:
        """Classify multiple documents. Returns list of prediction dicts."""
        results = []
        for text in texts:
            results.append(self.classify(text))
        return results
