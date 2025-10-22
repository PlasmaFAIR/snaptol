def test_basic(snapshot):
    RESULT = 135  # TODO: delete.

    assert snapshot == RESULT


def test_tolerance(snapshot):
    RESULT = 159  # TODO: delete.

    assert snapshot(rtol=1e-02, atol=1e-04) == RESULT


def test_match(snapshot):
    RESULT = 321  # TODO: delete.

    assert snapshot.match(RESULT, rtol=1e-02, atol=1e-04)
