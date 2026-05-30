# Trellis Document Classifier Model - Training Code

TF-IDF + LinearSVC document classifier for categorizing text into 11 classes.

## Data folder structure

```
data/
  dataset/                    # Raw text files
    business/                 # 100 files (business_1.txt ... business_100.txt)
    entertainment/            # 100 files
    food/                     # 100 files
    graphics/                 # 100 files
    historical/               # 100 files
    medical/                  # 100 files
    other/                    # 6 files
    politics/                 # 100 files
    space/                    # 100 files
    sport/                    # 100 files
    technologie/              # 100 files
  episodes.csv                # Generated metadata (1006 rows, one per file)
  splits/
    train.txt                 # 800 episode IDs
    test.txt                  # 206 episode IDs (includes 6 "other")
```

Each class directory contains `.txt` files with the text samples for that category. `episodes.csv` is generated from the raw files and tracks `episode_id`, `file_name`, `file_path`, `class_id`, and `label` for every sample. `data/splits/` holds the train/test split (80/20 stratified split on the 10 real classes; "other" is forced to test).

## How to run

All commands are run from the `trellis/` directory.

### 1. Generate episodes

```bash
python scripts/generate_episodes.py
```

Scans `data/dataset/`, writes `data/episodes.csv` with one row per text file.

### 2. Generate train/test split

```bash
python scripts/generate_splits.py
```

Stratified 80/20 split on the 10 real classes (seed=42). "Other" is always placed in the test set. Writes `data/splits/train.txt` and `data/splits/test.txt`.

### 3. Train the model

```bash
python scripts/train_v2.py
```

Trains TF-IDF + LinearSVC with probability calibration using only the train split. Saves artifacts to `output/`:

| File               | Content                                        |
|--------------------|------------------------------------------------|
| `vectorizer.joblib`| Fitted TfidfVectorizer                       |
| `classifier.joblib`| CalibratedClassifierCV (probability outputs)   |
| `raw_svc.joblib`   | Uncalibrated LinearSVC (coefficients + intercept) |
| `classes.joblib`   | List of class names                            |
| `vocab.csv`        | Vocabulary with index                          |
| `idf.csv`          | IDF values for each term                       |
| `svm_coef.csv`     | SVM weight vector per class                    |
| `svm_intercept.csv`| SVM bias per class                             |
| `metadata.json`    | Vocab size, CV accuracy, hyperparameters       |

### 4. Evaluate

```bash
python scripts/evaluate.py
```

Runs inference on both splits and prints confusion matrix + per-class precision/recall.

- **Train** (10-class): 11x11 confusion matrix (rows 0-9, cols 0-10). Accuracy computed over 10 real classes.
- **Test** (11-class): 11x11 confusion matrix (rows/cols 0-10 including "other"). Accuracy over all 11 classes.

## How the model works

### Text preprocessing

Every input passes through `InputPreprocessor.preprocess()` which applies six steps in order:

1. **Lowercase** - normalize case
2. **Tokenize** - split into tokens via NLTK's `word_tokenize`
3. **Strip non-alpha** - remove punctuation, digits, special chars (keep only alphabetic tokens)
4. **Remove stopwords** - filter out common English words (the, is, at, etc.)
5. **Lemmatize** - reduce to base form (running -> run, cats -> cat)
6. **Drop short tokens** - remove any remaining tokens with one character or fewer

**Example:**

Input: `The company reported strong quarterly earnings exceeding analyst expectations.`

Processing:
- Lowercase: `the company reported strong quarterly earnings exceeding analyst expectations.`
- Tokenize: `['The', 'company', 'reported', 'strong', 'quarterly', 'earnings', 'exceeding', 'analyst', 'expectations', '.']`
- Strip non-alpha: `['The', 'company', 'reported', 'strong', 'quarterly', 'earnings', 'exceeding', 'analyst', 'expectations']`
- Remove stopwords: `['company', 'reported', 'strong', 'quarterly', 'earnings', 'exceeding', 'analyst', 'expectations']`
- Lemmatize: `['company', 'report', 'strong', 'quarterly', 'earning', 'exceed', 'analyst', 'expect']`
- Drop <=1 char: `['company', 'report', 'strong', 'quarterly', 'earning', 'exceed', 'analyst', 'expect']`

### Feature extraction: TF-IDF

The preprocessed tokens are joined into a single string and passed to a `TfidfVectorizer`:

- **Max features**: 2000 terms (frequency-filtered by the training set)
- **min_df=2**: ignore terms appearing in fewer than 2 documents
- **max_df=0.95**: ignore terms appearing in more than 95% of documents (likely noise)
- **sublinear_tf=True**: apply log(1 + tf) to raw term frequencies, reducing the impact of very frequent terms

This converts text into a sparse 1000-dimensional vector where each dimension represents how important a term is in this document relative to the training corpus.

### Classifier: LinearSVC with calibration

A `LinearSVC` (C=1.0, balanced class weights) fits a linear decision boundary in this 1000D space. The balanced weight parameter automatically compensates for the "other" class having only 6 samples vs 100 for the real classes.

A `CalibratedClassifierCV` wraps the SVC with Platt scaling (sigmoid, 5-fold CV) to convert raw SVM scores into well-calibrated probabilities. Without calibration, SVM outputs are distances from the decision boundary and are not directly comparable probabilities.

**Example classification:**

For input `A detailed analysis of medical imaging techniques for early cancer detection`:

1. Preprocessing yields: `['detailed', 'analysis', 'medical', 'imaging', 'technique', 'early', 'cancer', 'detection']`
2. TF-IDF vector maps this to the 1000D space
3. Calibration produces probabilities across all 11 classes
4. If the top prediction ("medical") has probability < 0.60 (the confidence threshold), the result is relabeled as "other"
5. Final output:

```json
{
  "category": "medical",
  "confidence": 0.87,
  "threshold": 0.60,
  "is_other": false,
  "probabilities": {
    "business": 0.02,
    "medical": 0.87,
    ...
  }
}
```

The model is non-neural, fast, and deterministic (no randomness at inference time). It categorizes based on keyword patterns learned from the training corpus, not semantic understanding.
