from numba import vectorize
import numpy as np


@vectorize
def generate_confusion_matrix(x, y):
    """
    NumPy ufunc implemented with Numba that generates a confusion matrix as follows:
    1 = True Positive, 2 = False Positive, 3 = False Negative, 4 = True Negative.
    """
    if x and y:
        return 1

    elif not x and y:
        return 2

    elif x and not y:
        return 3

    else:
        return 4


@vectorize
def generate_intersection(x, y):
    """
    NumPy ufunc implemented with Numba that generates the intersection of two arrays.
    Serves as a helper method for fast_jaccard_score().
    """
    if x & y:
        return 1

    else:
        return 0


@vectorize
def generate_union(x, y):
    """
    NumPy ufunc implemented with Numba that generates the union of two arrays.
    Serves as a helper method for fast_jaccard_score().
    """
    if x | y:
        return 1

    else:
        return 0


def count_confusion_matrix(y_true, y_pred):
    matrix = generate_confusion_matrix(y_true, y_pred)
    tp = np.count_nonzero(matrix == 1)  # True Positive
    fp = np.count_nonzero(matrix == 2)  # False Positive
    fn = np.count_nonzero(matrix == 3)  # False Negative
    tn = np.count_nonzero(matrix == 4)  # True Negative
    return tp, fp, fn, tn


def fast_true_positive_rate(y_true, y_pred):
    tp, fp, fn, tn = count_confusion_matrix(y_true, y_pred)
    return 0 if (tp + fn) == 0 else tp / (tp + fn)


def fast_false_positive_rate(y_true, y_pred):
    tp, fp, fn, tn = count_confusion_matrix(y_true, y_pred)
    return 0 if (fp + tn) == 0 else fp / (fp + tn)


def fast_precision_score(y_true, y_pred):
    tp, fp, fn, tn = count_confusion_matrix(y_true, y_pred)
    return 0 if (tp + fp) == 0 else tp / (tp + fp)


def fast_recall_score(y_true, y_pred):
    tp, fp, fn, tn = count_confusion_matrix(y_true, y_pred)
    return 0 if (tp + fn) == 0 else tp / (tp + fn)


def fast_sensitivity_score(y_true, y_pred):
    tp, fp, fn, tn = count_confusion_matrix(y_true, y_pred)
    return 0 if (tp + fn) == 0 else tp / (tp + fn)


def fast_specificity_score(y_true, y_pred):
    tp, fp, fn, tn = count_confusion_matrix(y_true, y_pred)
    return 0 if (fp + tn) == 0 else tn / (fp + tn)


def fast_accuracy_score(y_true, y_pred):
    tp, fp, fn, tn = count_confusion_matrix(y_true, y_pred)
    return 0 if (tp + fp + fn + tn) == 0 else (tp + tn) / (tp + fp + fn + tn)


def fast_balanced_accuracy_score(y_true, y_pred):
    sensitivity = fast_sensitivity_score(y_true, y_pred)
    specificity = fast_specificity_score(y_true, y_pred)
    return (sensitivity + specificity) / 2


def fast_f1_score(y_true, y_pred):
    precision = fast_precision_score(y_true, y_pred)
    recall = fast_recall_score(y_true, y_pred)
    return 0 if (precision + recall) == 0 else 2 * (precision * recall) / (precision + recall)


def fast_roc_auc_score(y_true, y_pred):
    true_positive_rate = fast_true_positive_rate(y_true, y_pred)
    false_positive_rate = fast_false_positive_rate(y_true, y_pred)
    return (1 + true_positive_rate - false_positive_rate) / 2


def fast_jaccard_score(y_true, y_pred):
    intersection = generate_intersection(y_true, y_pred)
    union = generate_union(y_true, y_pred)
    numer = np.count_nonzero(intersection == 1)
    denom = np.count_nonzero(union == 1)
    return 0 if denom == 0 else numer / denom
