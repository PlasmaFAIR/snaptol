from collections.abc import Collection, Iterable, Mapping
from itertools import zip_longest
from typing import Any

import numpy as np

DEFAULT_RTOL = 1e-05
DEFAULT_ATOL = 1e-08


def compare_intelligent(  # noqa: PLR0911, PLR0912
    actual: Any,
    expected: Any,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    equal_nan: bool = False,
) -> bool:
    if isinstance(actual, float) or isinstance(expected, float):
        return np.isclose(actual, expected, rtol=rtol, atol=atol, equal_nan=equal_nan)

    if isinstance(actual, int) and isinstance(expected, int):
        return actual == expected

    if isinstance(actual, complex) and isinstance(expected, complex):
        return actual == expected

    if isinstance(actual, str) and isinstance(expected, str):
        return actual == expected

    if isinstance(actual, bool) and isinstance(expected, bool):
        return actual == expected

    if isinstance(actual, bytes | bytearray | memoryview) and isinstance(
        expected, bytes | bytearray | memoryview
    ):
        return bytes(actual) == bytes(expected)

    if actual is None and expected is None:
        return True

    if isinstance(actual, np.ndarray) and isinstance(expected, np.ndarray):
        return np.allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=equal_nan)

    if isinstance(actual, Mapping) and isinstance(expected, Mapping):
        ak, ek = set(actual.keys()), set(expected.keys())

        if ak != ek:
            return False

        for k in sorted(ak, key=str):
            if not compare_intelligent(actual[k], expected[k], rtol, atol, equal_nan):
                return False

        return True

    if isinstance(actual, Iterable) and isinstance(expected, Iterable):
        sentinel = object()

        for a, e in zip_longest(actual, expected, fillvalue=sentinel):
            if a is sentinel or e is sentinel:
                return False  # Length mismatch.

            if not compare_intelligent(a, e, rtol, atol, equal_nan):
                return False

        return True

    if isinstance(actual, Collection) and isinstance(expected, Collection):
        if len(actual) != len(expected):
            return False

        for a, e in zip(actual, expected):
            if not compare_intelligent(a, e, rtol, atol, equal_nan):
                return False

        return True

    if isinstance(actual, set) and isinstance(expected, set):
        a = sorted(map(repr, actual))
        b = sorted(map(repr, expected))

        return a == b

    if type(actual) is type(expected):
        return actual == expected

    return False
