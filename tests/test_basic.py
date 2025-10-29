import numpy as np


def test_basic(snapshot):
    RESULT = 135  # TODO: delete.

    assert snapshot == RESULT


def test_tolerance(snapshot):
    RESULT = 159  # TODO: delete.

    assert snapshot(rtol=1e-02, atol=1e-04) == RESULT


def test_match(snapshot):
    RESULT = 321  # TODO: delete.

    assert snapshot.match(RESULT, rtol=1e-02, atol=1e-04)


def test_numpy(snapshot):
    RESULT = np.array([1, 2, 3], dtype=int)

    snapshot.assert_array_equal(RESULT)
