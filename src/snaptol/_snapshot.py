from __future__ import annotations

import dataclasses
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any, TypeVar

import numpy.testing as npt

from ._compare import DEFAULT_ATOL, DEFAULT_RTOL, compare_intelligent
from ._io import read_snapshot, snapshot_filename, write_snapshot

F = TypeVar("F", bound=Callable[..., Any])


def auto_update(method: F) -> F:
    @wraps(method)
    def wrapper(self: Snapshot, value: Any, *args, **kwargs):
        if self.snapshot_update:
            write_snapshot(self.snapshot_file, value)
            return True

        if not self.snapshot_found:
            raise AssertionError("Snapshot not found.")

        return method(value, self.expected, *args, **kwargs)

    return wrapper


@dataclasses.dataclass
class Snapshot:
    test_name: str
    test_file: Path
    snapshot_file: Path
    snapshot_update: bool
    snapshot_found: bool = False
    rtol: float = DEFAULT_RTOL
    atol: float = DEFAULT_ATOL
    equal_nan: bool = False
    expected: Any = dataclasses.field(init=False, repr=False)

    @classmethod
    def from_request(cls, request) -> Snapshot:
        test_name = request.node.name
        test_file = Path(request.fspath)
        snapshot_file = snapshot_filename(test_name, test_file)
        snapshot_update = request.config.getoption("--snapshot-update")

        return cls(
            test_name=test_name,
            test_file=test_file,
            snapshot_file=snapshot_file,
            snapshot_update=snapshot_update,
        )

    def __post_init__(self) -> None:
        if not self.snapshot_update:
            try:
                self.expected = read_snapshot(self.snapshot_file)
                self.snapshot_found = True
            except FileNotFoundError:
                self.expected = None
                self.snapshot_found = False

    def __eq__(self, value: Any) -> bool:
        if self.snapshot_update:
            write_snapshot(self.snapshot_file, value)
            return True

        if not self.snapshot_found:
            return False

        return compare_intelligent(
            self.expected, value, self.rtol, self.atol, self.equal_nan
        )

    def __call__(
        self, *, rtol: float = DEFAULT_RTOL, atol: float = DEFAULT_ATOL
    ) -> Snapshot:
        return dataclasses.replace(self, rtol=rtol, atol=atol)

    def match(
        self, value, *, rtol: float = DEFAULT_RTOL, atol: float = DEFAULT_ATOL
    ) -> bool:
        return self(rtol=rtol, atol=atol) == value

    def matches(self, *args, **kwargs) -> bool:
        return self.match(*args, **kwargs)

    assert_allclose = auto_update(npt.assert_allclose)
    assert_array_almost_equal_nulp = auto_update(npt.assert_array_almost_equal_nulp)
    assert_array_max_ulp = auto_update(npt.assert_array_max_ulp)
    assert_array_equal = auto_update(npt.assert_array_equal)
    assert_equal = auto_update(npt.assert_equal)
    assert_string_equal = auto_update(npt.assert_string_equal)
