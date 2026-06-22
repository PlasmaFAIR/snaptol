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
    SENTINEL,
    _cache_failed_test,
    _store_test_diff,
    _uncache_test,
    read_snapshot,
    snapshot_directory,
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
    def wrapper(snapshot: Snapshot, value: Any, *args, **kwargs):
        # Do the comparison and store any exceptions for later.
        comparison_matched = False
        caught_exception = None
        problem_found = False

        if snapshot.snapshot_found:
            try:
                method(value, snapshot.expected, *args, **kwargs)
                comparison_matched = True
            except AssertionError as exc:
                # The comparison has not matched, this is only a problem if we are NOT in update mode.
                caught_exception = exc
                problem_found = not snapshot.snaptol_update
            except TypeError as exc:
                caught_exception = exc
                problem_found = not snapshot.snaptol_update
        elif not snapshot.snaptol_update:
            # If we are in update mode, we don't care that the snapshot is missing.
            caught_exception = FileNotFoundError("Snapshot file not found.")
            problem_found = True

        if snapshot.snaptol_update:
            write_snapshot(snapshot.snapshot_file, value)
            _uncache_test(snapshot.cache, snapshot.nodeid)

        # Show a diff if requested and if a difference exists.
        if snapshot.show_diff and not comparison_matched:
            _store_test_diff(
                snapshot.config,
                snapshot.snapshot_file,
                before=snapshot.expected if snapshot.snapshot_found else SENTINEL,
                after=value,
            )

        if problem_found:
            _cache_failed_test(
                snapshot.cache, snapshot.nodeid, snapshot.snapshot_file, value
            )
            raise caught_exception from None

        return True

    return wrapper


@dataclasses.dataclass
class Snapshot:
    nodeid: str
    snapshot_file: Path
    snapshot_dir: Path
    snaptol_update: bool = False
    snapshot_found: bool = False
    show_diff: bool = False
    rtol: float = DEFAULT_RTOL
    atol: float = DEFAULT_ATOL
    equal_nan: bool = False
    expected: Any = dataclasses.field(init=False, repr=False)
    cache: pytest.Cache = None
    config: pytest.Config = None

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

        nodeid = request.node.nodeid
        snapshot_file = snapshot_filename(
            request.node.nodeid, test_dir=Path(request.fspath).parent
        )
        snapshot_dir = snapshot_directory(test_dir=Path(request.fspath).parent)
        snaptol_update = request.config.getoption(
            "--snaptol-update"
        ) or request.config.getoption("--snaptol-update-all")
        show_diff = request.config.getoption("--snaptol-show-diff")
        cache = request.config.cache
        config = request.config

        return cls(
            nodeid=nodeid,
            snapshot_file=snapshot_file,
            snapshot_dir=snapshot_dir,
            snaptol_update=snaptol_update,
            show_diff=show_diff,
            cache=cache,
            config=config,
        )

    def __post_init__(self) -> None:
        try:
            self.expected = read_snapshot(self.snapshot_file)
            self.snapshot_found = True
        except FileNotFoundError:
            self.expected = None
            self.snapshot_found = False

    def __eq__(self, value: Any) -> bool:
        # Do the comparison and store any exceptions for later.
        comparison_matched = False
        caught_exception = None
        problem_found = False

        if self.snapshot_found:
            try:
                comparison_matched = compare_intelligent(
                    value, self.expected, self.rtol, self.atol, self.equal_nan
                )
                # If the comparison has not matched, this is only a problem if we are NOT in update mode.
                if not comparison_matched:
                    problem_found = not self.snaptol_update
            except TypeError as exc:
                caught_exception = exc
                problem_found = not self.snaptol_update
        elif not self.snaptol_update:
            # If we are in update mode, we don't care that the snapshot is missing.
            caught_exception = FileNotFoundError("Snapshot file not found.")
            problem_found = True

        if self.snaptol_update:
            write_snapshot(self.snapshot_file, value)
            _uncache_test(self.cache, self.nodeid)

        # Show a diff if requested and if a difference exists.
        if self.show_diff and not comparison_matched:
            _store_test_diff(
                self.config,
                self.snapshot_file,
                before=self.expected if self.snapshot_found else SENTINEL,
                after=value,
            )

        if problem_found:
            _cache_failed_test(self.cache, self.nodeid, self.snapshot_file, value)
            if caught_exception is not None:
                raise caught_exception from None
            return False

        return True

    def __hash__(self):
        return hash(self.nodeid)

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

    def __repr__(self):
        return f"Snapshot({self.expected})"

    assert_allclose = auto_update(npt.assert_allclose)
    assert_array_almost_equal_nulp = auto_update(npt.assert_array_almost_equal_nulp)
    assert_array_max_ulp = auto_update(npt.assert_array_max_ulp)
    assert_array_equal = auto_update(npt.assert_array_equal)
    assert_equal = auto_update(npt.assert_equal)
    assert_string_equal = auto_update(npt.assert_string_equal)
