from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, Self, TypeVar

import numpy.testing as npt
import pytest

from .compare import DEFAULT_ATOL, DEFAULT_RTOL, compare_intelligent
from .io import (
    SENTINEL,
    _cache_failed_test,
    _store_test_diff,
    _uncache_test,
    read_snapshot,
    snapshot_filename,
    write_snapshot,
)
from .session import SnaptolSession

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
    def wrapper(self: "Snapshot", value: Any, *args, **kwargs):  # ruff: ignore[UP037]
        __tracebackhide__ = True  # Hide traceback for py.test
        return self._match_with_method(method, value, *args, **kwargs)

    return wrapper


@dataclass
class SnaptolResult:
    nodeid: str
    index: str | int
    filename: Path
    success: bool


@dataclass
class Snapshot:
    session: SnaptolSession
    nodeid: str
    snapshot_file: Path
    snapshot_dir: Path
    snaptol_update: bool = False
    snapshot_found: bool = False
    show_diff: bool = False
    rtol: float = DEFAULT_RTOL
    atol: float = DEFAULT_ATOL
    equal_nan: bool = False
    expected: Any = field(init=False, repr=False)
    cache: pytest.Cache = None
    config: pytest.Config = None

    # How many times has this fixture been called/asserted
    _executions: int = field(init=False, default=0)
    # Some metadata on the result of each execution
    _execution_results: dict[str, SnaptolResult] = field(
        init=False, default_factory=dict
    )
    # The current index name
    _index: str | int | None = None
    # List of actions to take after each execution
    _post_execution_actions: list[Callable[..., None]] = field(
        init=False,
        default_factory=list,
    )

    @classmethod
    def from_request(cls, request: pytest.FixtureRequest) -> Self:
        """
        Create a ``Snapshot`` instance from a pytest request object. Returns
        the instansiated ``Snapshot`` object.

        Parameters
        ----------
        request
            The pytest request fixture containing test information.
        """

        nodeid = request.node.nodeid
        base_dir = Path(request.fspath).parent
        snapshot_file = snapshot_filename(request.node, test_dir=base_dir)
        snapshot_dir = snapshot_file.parent
        snaptol_update = request.config.getoption(
            "--snaptol-update"
        ) or request.config.getoption("--snaptol-update-all")
        show_diff = request.config.getoption("--snaptol-show-diff")
        cache = request.config.cache
        config = request.config

        return cls(
            session=request.session.config._snaptol,  # ty: ignore[unresolved-attribute]
            nodeid=nodeid,
            snapshot_file=snapshot_file,
            snapshot_dir=snapshot_dir,
            snaptol_update=snaptol_update,
            show_diff=show_diff,
            cache=cache,
            config=config,
        )

    def __post_init__(self) -> None:
        self.session.register_request(self)

    @property
    def index(self) -> str | int:
        if self._index is not None:
            return self._index
        return self._executions

    @property
    def filename(self) -> Path:
        if self.index != 0:
            filestem = self.snapshot_file.stem
            return self.snapshot_file.with_stem(f"{filestem}[{self.index}]")

        return self.snapshot_file

    def _read_snapshot(self) -> None:
        try:
            self.expected = read_snapshot(self.filename)
            self.snapshot_found = True
        except FileNotFoundError:
            self.expected = None
            self.snapshot_found = False

    def _match_with_method(self, method: F, value: Any, *args, **kwargs) -> bool:
        __tracebackhide__ = True  # Hide traceback for py.test
        # Do the comparison and store any exceptions for later.
        comparison_matched = False
        caught_exception = None
        problem_found = False

        self._read_snapshot()

        if self.snapshot_found:
            try:
                _matched = method(value, self.expected, *args, **kwargs)
                # If the comparison has not matched, this is only a problem if
                # we are NOT in update mode.
                # _matched might be an array (e.g. np.testing.assert_array_max_ulp),
                # and numpy hates `array == True`, so need identity comparisons here
                if _matched is True or _matched is None:
                    comparison_matched = True
                else:
                    problem_found = not self.snaptol_update
            except (AssertionError, TypeError) as exc:
                caught_exception = exc
                problem_found = not self.snaptol_update
        elif not self.snaptol_update:
            # If we are in update mode, we don't care that the snapshot is missing.
            caught_exception = FileNotFoundError(
                f"Snapshot file '{self.filename}' not found."
            )
            problem_found = True

        if self.snaptol_update:
            write_snapshot(self.filename, value)
            _uncache_test(self.cache, self.nodeid)

        # Show a diff if requested and if a difference exists.
        if self.show_diff and not comparison_matched:
            _store_test_diff(
                self.config,
                self.filename,
                before=self.expected if self.snapshot_found else SENTINEL,
                after=value,
            )

        # Store result for later checking for unused snapshots
        result = SnaptolResult(
            nodeid=self.nodeid,
            index=self.index,
            filename=self.filename,
            success=not problem_found,
        )
        self._execution_results[str(self.index)] = result
        self._executions += 1
        self._post_execution()

        if problem_found:
            _cache_failed_test(self.cache, result, value)
            if caught_exception is not None:
                raise caught_exception from None
            return False

        return True

    def __eq__(self, value: object) -> bool:
        __tracebackhide__ = True  # Hide traceback for py.test
        return self._match_with_method(
            compare_intelligent,
            value,
            rtol=self.rtol,
            atol=self.atol,
            equal_nan=self.equal_nan,
        )

    def __hash__(self):
        return hash(self.nodeid)

    def __call__(
        self,
        *,
        rtol: float | None = None,
        atol: float | None = None,
        equal_nan: bool | None = None,
    ) -> Self:
        if rtol is not None:
            self.__with_prop("rtol", rtol)
        if atol is not None:
            self.__with_prop("atol", atol)
        if equal_nan is not None:
            self.__with_prop("equal_nan", equal_nan)
        return self

    def __getitem__(self, index: str | int) -> Self:
        self.__with_prop("_index", index)
        return self

    def __with_prop(self, prop_name: str, prop_value: Any) -> None:
        _value = getattr(self, prop_name, None)
        setattr(self, prop_name, prop_value)
        self._post_execution_actions.append(lambda: setattr(self, prop_name, _value))

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

    def _post_execution(self) -> None:
        """
        Restore instance attributes
        """
        while self._post_execution_actions:
            self._post_execution_actions.pop()()

    assert_allclose = auto_update(npt.assert_allclose)
    assert_array_almost_equal_nulp = auto_update(npt.assert_array_almost_equal_nulp)
    assert_array_max_ulp = auto_update(npt.assert_array_max_ulp)
    assert_array_equal = auto_update(npt.assert_array_equal)
    assert_equal = auto_update(npt.assert_equal)
    assert_string_equal = auto_update(npt.assert_string_equal)
