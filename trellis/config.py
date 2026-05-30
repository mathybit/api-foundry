# Trellis document classifier configuration

# --- Model hyperparameters ---
SVC_C = 1.0
SVC_MAX_ITER = 5000
SVC_CLASS_WEIGHT = "balanced"

TFIDF_MAX_FEATURES = 2000  # vocab size
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.95
TFIDF_SUBLINEAR_TF = True

CALIBRATION_METHOD = "sigmoid"
CALIBRATION_CV_FOLDS = 5

DEFAULT_THRESHOLD = 0.60

# Model artifacts
OUTPUT_DIR = "output"
ARTIFACT_VECTORIZER = "vectorizer.joblib"
ARTIFACT_CLASSIFIER = "classifier.joblib"
ARTIFACT_RAW_SVC = "raw_svc.joblib"
ARTIFACT_CLASSES = "classes.joblib"
ARTIFACT_METADATA = "metadata.json"

# Other CSV exports
ARTIFACT_VOCAB = "vocab.csv"
ARTIFACT_IDF = "idf.csv"
ARTIFACT_SVM_COEF = "svm_coef.csv"
ARTIFACT_SVM_INTERCEPT = "svm_intercept.csv"

# --- Dataset / classification config ---
CLASS_NAMES = [
    "business", "entertainment", "food", "graphics", "historical",
    "medical", "politics", "space", "sport", "technologie", "other",
]
CLASS_IDS = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# --- Split config ---
TEST_SPLIT_RATIO = 0.20
TRAIN_SPLIT_RATIO = 1.0 - TEST_SPLIT_RATIO
SPLIT_RANDOM_SEED = 42
