import numpy as np

from src.features import make_windows


def test_windows_shape_and_alignment():
    X, y = make_windows(np.arange(10), 3)
    assert X.shape == (7, 3) and y.shape == (7,)
    assert X[0].tolist() == [0, 1, 2] and y[0] == 3
    assert X[-1].tolist() == [6, 7, 8] and y[-1] == 9


def test_target_never_in_its_window():
    v = np.arange(100)  # unique values, so membership is unambiguous
    X, y = make_windows(v, 14)
    assert not any(t in row for row, t in zip(X, y))


def test_future_change_does_not_alter_earlier_windows():
    v = np.arange(50, dtype=float)
    w = v.copy()
    w[30:] += 999
    X1, _ = make_windows(v, 5)
    X2, _ = make_windows(w, 5)
    # window i holds days i..i+4, so windows 0..25 end before day 30
    assert np.array_equal(X1[:26], X2[:26])
