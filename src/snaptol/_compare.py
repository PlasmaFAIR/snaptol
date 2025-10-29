from collections.abc import Collection, Mapping
from typing import Any

import numpy as np


def compare_intelligent(  # noqa: PLR0911, PLR0912
    actual: Any, expected: Any, rtol: float, atol: float, equal_nan: bool
) -> bool:
    if isinstance(actual, float) or isinstance(expected, float):
        return np.isclose(actual, expected)

    if isinstance(actual, int) and isinstance(expected, int):
        return actual == expected

    if isinstance(actual, str) and isinstance(expected, str):
        return actual == expected

    if isinstance(actual, Collection) and isinstance(expected, Collection):
        if len(actual) != len(expected):
            return False

        for a, e in zip(actual, expected):
            if not compare_intelligent(a, e, rtol, atol, equal_nan):
                return False

        return True

    if isinstance(actual, Mapping) and isinstance(expected, Mapping):
        ak, ek = set(actual.keys()), set(expected.keys())

        if ak != ek:
            return False

        for k in sorted(ak, key=str):
            if not compare_intelligent(actual[k], expected[k]):
                return False
        return True

    if isinstance(actual, set) and isinstance(expected, set):
        a = sorted(map(repr, actual))
        b = sorted(map(repr, expected))

        return a == b

    if isinstance(actual, np.ndarray) and isinstance(expected, np.ndarray):
        return np.allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=equal_nan)

    if type(actual) is type(expected):
        return actual == expected

    return False
