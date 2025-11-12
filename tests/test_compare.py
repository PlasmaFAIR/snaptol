import pytest
from numpy import array

from snaptol import compare_intelligent


@pytest.mark.parametrize(
    ("a", "b", "kwargs"),
    [
        (1, 1, {}),
        (1.0, 1.0, {}),
        (1, 1.0, {}),
        (1.0, 1, {}),
        (1 + 2j, 1 + 2j, {}),
        (1e-9, 2e-9, {"rtol": 0, "atol": 1e-8}),
        (1e6, 1.000001e6, {"rtol": 1e-5, "atol": 0}),
        ("string", "string", {}),
        (b"string", b"string", {}),
        (bytearray(b"string"), bytearray(b"string"), {}),
        (memoryview(b"string"), memoryview(b"string"), {}),
        (True, True, {}),
        (False, False, {}),
        (None, None, {}),
        ([1, 2, 3], [1, 2, 3], {}),
        ({"a": 1, "b": 2}, {"a": 1, "b": 2}, {}),
        (map(str, range(10)), map(str, range(10)), {}),
        (set(range(10)), set(range(10)), {}),
        (frozenset(range(10)), frozenset(range(10)), {}),
        (array([1, 2, 3]), array([1, 2, 3]), {}),
        (array([1, 2, 3]), array([1, 2, 3], dtype=float), {}),
    ],
)
def test_compare_equal(a, b, kwargs):
    assert compare_intelligent(a, b, **kwargs)


@pytest.mark.parametrize(
    ("a", "b", "kwargs"),
    [
        (1, 2, {}),
        (1.0, 2.0, {}),
        (1, 2.0, {}),
        (1.0, 2, {}),
        (1 + 2j, 2 + 1j, {}),
        (1e-9, 5e-8, {"rtol": 0, "atol": 1e-9}),
        (1e6, 1.1e6, {"rtol": 1e-3, "atol": 0}),
        ("string1", "string2", {}),
        (b"string1", b"string2", {}),
        (bytearray(b"string1"), bytearray(b"string2"), {}),
        (memoryview(b"string1"), memoryview(b"string2"), {}),
        (True, False, {}),
        ([1, 2, 3], [3, 2, 1], {}),
        ({"a": 1, "b": 2}, {"c": 1, "d": 2}, {}),
        ({"a": 1, "b": 2}, {"a": 3, "b": 4}, {}),
        (map(str, range(10)), map(str, range(11)), {}),
        (set(range(10)), set(range(11)), {}),
        (frozenset(range(10)), frozenset(range(11)), {}),
        (array([1, 2, 3]), array([3, 2, 1]), {}),
        (array([1, 2, 3]), array([3, 2, 1], dtype=float), {}),
    ],
)
def test_compare_not_equal(a, b, kwargs):
    assert not compare_intelligent(a, b, **kwargs)
