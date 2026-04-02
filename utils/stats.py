import numpy as np
from scipy.stats import mannwhitneyu


def roc_auc(labels: np.ndarray, scores: np.ndarray):
    """area under the ROC curve w mann whitney U"""
    labels = np.asarray(labels, dtype=bool)
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    u, _ = mannwhitneyu(pos, neg, alternative='two-sided')
    return u / (len(pos) * len(neg))


def cosine_similarity(a: np.ndarray, b: np.ndarray):
    """cosine similarity between two vectors, clipped to [-1, 1]"""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return np.nan
    return np.clip(np.dot(a, b) / (na * nb), -1, 1)
