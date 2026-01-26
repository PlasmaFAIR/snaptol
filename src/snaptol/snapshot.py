from __future__ import annotations

import dataclasses
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any, TypeVar

import numpy.testing as npt
import pytest

from .compare import DEFAULT_ATOL, DEFAULT_RTOL, compare_intelligent
from .io import (
    _cache_failed_test,
    _uncache_test,
    read_snapshot,
    snapshot_filename,
    write_snapshot,
)

F = TypeVar("F", bound=Callable[..., Any])


def auto_update(method: F) -> F:
    """
    Decorator that handles snapshot updates and comparisons for testing functions.

    Parameters
    ----------
    method
        The testing function to wrap.

    Raises
    ------
    AssertionError
        If snapshot not found and ``snaptol_update`` is ``False``.
    """

    @wraps(method)
    def wrapper(self: Snapshot, value: Any, *args, **kwargs):
        if self.snaptol_update:
            write_snapshot(self.snapshot_file, value)
            _uncache_test(self.cache, self.nodeid)
            return True

        try:
            if not snapshot.snapshot_found:
                raise FileNotFoundError("Snapshot file not found.")

            method(value, snapshot.expected, *args, **kwargs)

        except (AssertionError, FileNotFoundError):
            _cache_failed_test(
                snapshot.cache, snapshot.nodeid, snapshot.snapshot_file, value
            )
            raise

        return True

    return wrapper


@dataclasses.dataclass
class Snapshot:
    test_name: str
    test_file: Path
    nodeid: str
    snapshot_file: Path
    snaptol_update: bool
    snapshot_found: bool = False
    rtol: float = DEFAULT_RTOL
    atol: float = DEFAULT_ATOL
    equal_nan: bool = False
    expected: Any = dataclasses.field(init=False, repr=False)
    cache: pytest.Cache = None

    @classmethod
    def from_request(cls, request) -> Snapshot:
        """
        Create a ``Snapshot`` instance from a pytest request object. Returns
        the instansiated ``Snapshot`` object.

        Parameters
        ----------
        request
            The pytest request fixture containing test information.
        """

        test_name = request.node.name
        test_file = Path(request.fspath)
        nodeid = request.node.nodeid
        snapshot_file = snapshot_filename(test_name, test_file)
        snaptol_update = request.config.getoption(
            "--snaptol-update"
        ) or request.config.getoption("--snaptol-update-all")
        cache = request.config.cache

        return cls(
            test_name=test_name,
            test_file=test_file,
            nodeid=nodeid,
            snapshot_file=snapshot_file,
            snaptol_update=snaptol_update,
            cache=cache,
        )

    def __post_init__(self) -> None:
        if not self.snaptol_update:
            try:
                self.expected = read_snapshot(self.snapshot_file)
                self.snapshot_found = True
            except FileNotFoundError:
                self.expected = None
                self.snapshot_found = False

    def __eq__(self, value: Any) -> bool:
        if self.snaptol_update:
            write_snapshot(self.snapshot_file, value)
            _uncache_test(self.cache, self.nodeid)
            return True

        success = (
            compare_intelligent(
                self.expected, value, self.rtol, self.atol, self.equal_nan
            )
            if self.snapshot_found
            else False
        )

        if not success:
            _cache_failed_test(self.cache, self.nodeid, self.snapshot_file, value)

            if not self.snapshot_found:
                raise FileNotFoundError("Snapshot file not found.")

            return False

        return True

    def __hash__(self):
        return hash((self.test_file, self.test_name))

    def __call__(
        self, *, rtol: float = DEFAULT_RTOL, atol: float = DEFAULT_ATOL
    ) -> Snapshot:
        return dataclasses.replace(self, rtol=rtol, atol=atol)

    def match(
        self, value, *, rtol: float = DEFAULT_RTOL, atol: float = DEFAULT_ATOL
    ) -> bool:
        """
        Compare a value with the stored snapshot. Returns ``True`` if the values match, ``False`` otherwise.

        Parameters
        ----------
        value
            The value to compare with the snapshot.
        rtol
            Relative tolerance for comparison.
        atol
            Absolute tolerance for comparison.
        """

        return self(rtol=rtol, atol=atol) == value

    def matches(self, *args, **kwargs) -> bool:
        """
        Alias for match() method. Compare a value with the stored snapshot.
        """

        return self.match(*args, **kwargs)

    assert_allclose = auto_update(npt.assert_allclose)
    assert_array_almost_equal_nulp = auto_update(npt.assert_array_almost_equal_nulp)
    assert_array_max_ulp = auto_update(npt.assert_array_max_ulp)
    assert_array_equal = auto_update(npt.assert_array_equal)
    assert_equal = auto_update(npt.assert_equal)
    assert_string_equal = auto_update(npt.assert_string_equal)
