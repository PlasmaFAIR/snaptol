from typing import Any


def compare_intelligent(a: Any, b: Any) -> bool:
    # TODO: extend to support other types and consider numerical tolerance.
    if not isinstance(a, int) and not isinstance(b, int):
        raise NotImplementedError("Not implemented for types other than int yet")
    return a == b
