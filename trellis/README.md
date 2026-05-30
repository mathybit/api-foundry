# Trellis Document Classifier Model - Training Code

TF-IDF + LinearSVC document classifier for categorizing text into 11 classes.

## Table of Contents

- [Data folder structure](#data-folder-structure)
- [How to run](#how-to-run)
  - [1. Generate episodes](#1-generate-episodes)
  - [2. Generate train/test split](#2-generate-trainsplit-split)
  - [3. Train the model](#3-train-the-model)
  - [4. Evaluate](#4-evaluate)
- [Model architecture](#model-architecture)
  - [Text preprocessing](#text-preprocessing)
  - [Feature extraction: TF-IDF](#feature-extraction-tfidf)
  - [Classifier: LinearSVC with calibration](#classifier-linearsvc-with-calibration)
- [Model performance](#model-performance)
- [Edge cases](#edge-cases)

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

### NLTK data

The preprocessing pipeline requires three NLTK data packages. Run this once before any script:

```python
import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
```

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

| File                | Content                                        |
|---------------------|--       ------------------   ------       -----|
| `vectorizer.joblib` | Fitted TfidfVectorizer                         |
| `classifier.joblib` | CalibratedClassifierCV (probability outputs)   |
| `raw_svc.joblib`    | Uncalibrated LinearSVC (coefficients + intercept) |
| `classes.joblib`    | List of class names                            |
| `vocab.csv`         | Vocabulary with index                          |
| `idf.csv`           | IDF values for each term                       |
| `svm_coef.csv`      | SVM weight vector per class                    |
| `svm_intercept.csv` | SVM bias per class                             |
| `metadata.json`     | Vocab size, CV accuracy, hyperparameters       |

### 4. Evaluate

```bash
python scripts/evaluate.py
```

Runs inference on both splits and prints confusion matrix + per-class precision/recall.

- **Train** (10-class): 11x11 confusion matrix (rows 0-9, cols 0-10). Accuracy computed over 10 real classes.
- **Test** (11-class): 11x11 confusion matrix (rows/cols 0-10 including "other"). Accuracy over all 11 classes.

## Model architecture

### Text preprocessing

Every input passes through `InputPreprocessor.preprocess()` which applies six steps in order:

1. **Lowercase** - normalize case
2. **Tokenize** - split into tokens via NLTK's `word_tokenize`
3. **Strip non-alpha** - remove punctuation, digits, special chars (keep only alphabetic tokens)
4. **Remove stopwords** - filter out common English words (the, is, at, etc.)
5. **Lemmatize** - reduce to base form (running -> run, cats -> cat)
6. **Drop short tokens** - remove any remaining tokens with one character or fewer

**Example:**

Input: `The CEO's quarterly report showed 15% growth, but analysts warn of regulatory risks.`

| Step | Output |
|------|------|
| Lowercase | `the ceo's quarterly report showed 15% growth, but analysts warn of regulatory risks.` |
| Tokenize | `['The', 'CEO', "'", 's', 'quarterly', 'report', 'showed', '15', '%', 'growth', ',', 'but', 'analysts', 'warn', 'of', 'regulatory', 'risks', '.']` |
| Strip non-alpha | `['The', 'CEO', 's', 'quarterly', 'report', 'showed', 'growth', 'analysts', 'warn', 'regulatory', 'risks']` |
| Remove stopwords | `['CEO', 'quarterly', 'report', 'showed', 'growth', 'analysts', 'warn', 'regulatory', 'risks']` |
| Lemmatize | `['CEO', 'quarterly', 'report', 'show', 'growth', 'analyst', 'warn', 'regulatory', 'risk']` |
| Drop <=1 char | `['CEO', 'quarterly', 'report', 'show', 'growth', 'analyst', 'warn', 'regulatory', 'risk']` |

Result corpus string: `ceo quarterly report show growth analyst warn regulatory risk`

### Feature extraction: TF-IDF

The preprocessed tokens are joined into a single string and passed to a `TfidfVectorizer`:

- **Max features**: 2000 terms (frequency-filtered by the training set)
- **min_df=2**: ignore terms appearing in fewer than 2 documents
- **max_df=0.95**: ignore terms appearing in more than 95% of documents (likely noise)
- **sublinear_tf=True**: apply log(1 + tf) to raw term frequencies, reducing the impact of very frequent terms

This converts text into a sparse 2000-dimensional vector where each dimension represents how important a term is in this document relative to the training corpus.

### Classifier: LinearSVC with calibration

A `LinearSVC` (C=1.0, balanced class weights) fits a linear decision boundary in this 2000D space. The balanced weight parameter automatically compensates for the "other" class having only 6 samples vs 100 for the real classes.

A `CalibratedClassifierCV` wraps the SVC with Platt scaling (sigmoid, 5-fold CV) to convert raw SVM scores into well-calibrated probabilities. Without calibration, SVM outputs are distances from the decision boundary and are not directly comparable probabilities.

**Example classification:**

For input `A detailed analysis of medical imaging techniques for early cancer detection`:

1. Preprocessing yields: `['detailed', 'analysis', 'medical', 'imaging', 'technique', 'early', 'cancer', 'detection']`
2. TF-IDF vector maps this to the 2000D space
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

## Model performance

Trained with the following hyperparameters: SVC C=1.0, TF-IDF max_features=2000, min_df=2, max_df=0.95, sublinear_tf=True, calibration=sigmoid (5-fold CV).

### Train evaluation (10-class)

| Metric | Value |
|------|---|
| Accuracy | 100.00% (800/800) |
| CV accuracy | 97.00% |

Perfect classification on the training set. All 10 classes achieved 1.0 precision and 1.0 recall.

### Test evaluation (11-class)

| Metric | Value |
|------|---|
| Accuracy | 93.20% (192/206) |

Per-class metrics:

| Class | Precision | Recall |
|-------|--------|------|
| business | 1.000 | 0.800 |
| entertainment | 1.000 | 1.000 |
| food | 1.000 | 1.000 |
| graphics | 1.000 | 0.950 |
| historical | 1.000 | 0.900 |
| medical | 1.000 | 0.800 |
| politics | 0.994 | 0.950 |
| space | 1.000 | 0.950 |
| sport | 1.000 | 0.950 |
| technologie | 1.000 | 1.000 |
| other | 0.939 | 1.000 |

No misclassifications within the test set, meaning all 6 "other" documents were correctly classified as "other". The remaining 4 misclassified test samples are spread across business (2 false negatives, 1 predicted as politics), historical (2 false negatives), medical (4 false negatives), and various single misclassifications.

## Edge cases

### Document with no vocabulary matches

If the preprocessed text contains none of the 2000 training vocabulary terms, the TF-IDF vector is all zeros. LinearSVC computes its decision as the dot product of the weight vector and the feature vector plus the bias term. With an all-zero vector, the dot product is zero and the scores reduce to just the intercepts for each class. The model still produces a prediction based on which class has the highest bias.

The calibrated probability output will be the same for every such document regardless of content - they are essentially "default" predictions. In practice this case is very unlikely given the preprocessing pipeline always yields meaningful tokens from real text, and the confidence threshold (0.60) will usually filter such uncertain predictions into "other".
