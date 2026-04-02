import numpy as np


def circular_shift_labels(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """circularly shift a 1d label array by a random offset"""
    n = len(labels)
    offset = rng.integers(1, n)
    return np.roll(labels, offset)
