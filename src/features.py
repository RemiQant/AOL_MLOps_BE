import numpy as np


def make_windows(values, n):
    """X[i] = values[i:i+n] and y[i] = values[i+n]: the target never appears in its own window."""
    values = np.asarray(values, dtype="float32")
    X = np.lib.stride_tricks.sliding_window_view(values[:-1], n)
    return X, values[n:]
