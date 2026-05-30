import numpy as np


def get_confusion_matrix(y_true1d, y_pred1d, n_classes=2):
    """
    Calculates the confusion matrix given ground truths and predictions (lists or 1D arrays)
    """
    assert len(y_true1d) == len(y_pred1d), f'Inputs y_true1d and y_pred1d must have the same length (got {len(y_true1d)} and {len(y_pred1d)})'

    cmatrix = np.zeros((n_classes, n_classes), dtype=int)
    for i in range(len(y_true1d)):
        true_label = y_true1d[i]  # ground truth (row index)
        pred_label = y_pred1d[i]  # predicted truth (col index)
        cmatrix[true_label, pred_label] += 1

    assert cmatrix.sum() == len(y_true1d), 'Check that class indices are in [0, n_classes).'
    return cmatrix


def compute_precision_recall(cmatrix, balanced=False, verbose=False):
    assert (cmatrix.ndim == 2) and (cmatrix.shape[0] == cmatrix.shape[1]), f'cmatrix must be square 2-D (got shape {cmatrix.shape})'
    cmatrix = cmatrix.copy()  # we will be altering cmatrix - make a copy to avoid changing the original

    n_classes = cmatrix.shape[0]

    result = {
        'pre': {class_idx: 0.0 for class_idx in range(n_classes)},
        'rec': {class_idx: 0.0 for class_idx in range(n_classes)},
    }

    idxarr = np.arange(n_classes)
    for class_idx in range(n_classes):
        nonidx = idxarr[idxarr != class_idx]

        # True positives
        class_TP = cmatrix[class_idx, class_idx]

        # False negatives
        fn_submatrix = cmatrix[class_idx, nonidx]
        class_FN = fn_submatrix.sum()

        # False positives
        fp_submatrix = cmatrix[nonidx, class_idx]
        class_FP = fp_submatrix.sum()

        # True negatives
        tn_submatrix = cmatrix[nonidx, :][:, nonidx]
        class_TN = tn_submatrix.sum()

        # Build 2x2 confusion matrix (binary classification relative to class_idx)
        binary_cmatrix = np.array([
            [class_TN, class_FP],
            [class_FN, class_TP]
        ]).astype(np.float32)

        # Optional balancing
        if balanced:
            row_sums = binary_cmatrix.sum(axis=1)
            max_row_idx = int(np.argmax(row_sums))
            max_row_sum = row_sums[max_row_idx]

            for i, row_sum in enumerate(row_sums):
                if row_sum > 0:
                    scaling_factor = max_row_sum / row_sum
                    binary_cmatrix[i, :] *= scaling_factor

            class_TN = binary_cmatrix[0, 0]
            class_FP = binary_cmatrix[0, 1]
            class_FN = binary_cmatrix[1, 0]
            class_TP = binary_cmatrix[1, 1]

        pre = class_TP / (class_TP + class_FP) if class_TP + class_FP > 0 else 0.0
        rec = class_TP / (class_TP + class_FN) if class_TP + class_FN > 0 else 0.0

        result['pre'][class_idx] = pre
        result['rec'][class_idx] = rec

    if verbose:
        for class_idx in range(n_classes):
            pre = result['pre'][class_idx]
            rec = result['rec'][class_idx]
            print(f'Class {class_idx} metrics: pre={pre:.4f} | rec={rec:.4f}')

    return result
