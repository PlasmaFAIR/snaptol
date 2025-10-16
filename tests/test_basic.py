def test_basic(snapshot):
    RESULT = 123  # TODO: delete.

    assert RESULT == snapshot  # noqa: SIM300


def test_tolerance(snapshot):
    RESULT = 123  # TODO: delete.

    assert RESULT == snapshot(rtol=1e-02, atol=1e-04)  # noqa: SIM300


def test_match(snapshot):
    RESULT = 123  # TODO: delete.

    assert snapshot.match(RESULT, rtol=1e-02, atol=1e-04)
