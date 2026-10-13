from pathlib import Path
from typing import Any

import pytest

from snaptol.session import SnaptolSession

from .io import (
    CACHE_STASH_KEY,
    DELETABLE_STASH_KEY,
    DELETED_STASH_KEY,
    DIFFS_STASH_KEY,
    _get_cache,
    _show_test_diff,
    _store_test_diff,
    _uncache_test,
    deserialise_snapshot,
    nodeid_to_key,
    read_snapshot,
    write_snapshot,
)
from .snapshot import Snapshot

# Global so we can have access in pytest hooks that don't take the session
_snaptol: SnaptolSession | None = None


@pytest.fixture
def snaptolshot(request: pytest.FixtureRequest) -> Snapshot:
    """
    A pytest fixture that provides a `Snapshot` object tied to the current test request.
    Returns the instanciated Snapshot object.

    Parameters
    ----------
    request
        The pytest request object containing test context information.
    """

    return Snapshot.from_request(request)


def pytest_addoption(parser: pytest.Parser):
    """
    Adds the ``--snaptol-update`` command line option to pytest.
    This option enables updating or cleaning up snapshot files during test execution.

    Parameters
    ----------
    parser
        The pytest command line parser to which the option will be added.
    """

    group = parser.getgroup("snaptol")
    group.addoption(
        "--snaptol-update",
        action="store_true",
        default=False,
        help="Update snaptol snapshot files of previously failed tests",
    )

    group.addoption(
        "--snaptol-update-all",
        action="store_true",
        default=False,
        help="Update all snaptol snapshot files",
    )

    group.addoption(
        "--snaptol-use-cache",
        action="store_true",
        default=False,
        help="In update mode, use cached snaptol snapshot data if available",
    )

    group.addoption(
        "--snaptol-show-cache",
        action="store_true",
        default=False,
        help="Show cached snaptol snapshot data",
    )

    group.addoption(
        "--snaptol-clear-cache",
        action="store_true",
        default=False,
        help="Clear cached snaptol snapshot data",
    )

    group.addoption(
        "--snaptol-show-diff",
        action="store_true",
        default=False,
        help="Show diff in update mode when snapshot data does not match data on file",
    )

    group.addoption(
        "--snaptol-dirname",
        dest="snaptol_dirname",
        default="__snapshots__",
        help="Name of directory for storing snapshots",
    )


def pytest_configure(config: pytest.Config):
    """
    Validates command line option combinations for snaptol snapshot management.
    This hook is called during pytest configuration to ensure that incompatible
    options are not used together.

    Parameters
    ----------
    config
        The pytest configuration object containing command line options and settings.

    Raises
    ------
    ValueError
        If incompatible command line options are used together.
    """

    snaptol_update = config.getoption("--snaptol-update")
    snaptol_update_all = config.getoption("--snaptol-update-all")
    snaptol_use_cache = config.getoption("--snaptol-use-cache")
    last_failed = config.getoption("--last-failed") or config.getoption("--lf")

    if snaptol_update and snaptol_update_all:
        raise ValueError(
            "Cannot use both --snaptol-update and --snaptol-update-all options"
        )

    if not snaptol_update and not snaptol_update_all and snaptol_use_cache:
        raise ValueError(
            "Cannot use --snaptol-use-cache option without --snaptol-update or --snaptol-update-all"
        )

    if snaptol_update_all and last_failed:
        raise ValueError("Cannot use --snaptol-update-all with --last-failed or --lf")


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config: Any, items: list[pytest.Item]):
    """
    Modifies the collection of test items based on snapshot update options.

    This hook is called after test collection to potentially filter which tests
    should be executed. When ``--snaptol-update`` is used, only tests that failed
    in the previous run are kept for execution. When ``--snaptol-use-cache`` is
    enabled, tests with cached snapshot data are deselected and their snapshots
    are written directly from cache without re-running the tests.

    Parameters
    ----------
    config
        The pytest configuration object containing command line options and cache.
    items
        List of collected pytest test items that can be modified in-place.
    """

    config._snaptol.collect_items(items)

    snaptol_update = config.getoption("--snaptol-update")
    snaptol_update_all = config.getoption("--snaptol-update-all")
    snaptol_use_cache = config.getoption("--snaptol-use-cache")
    snaptol_show_diff = config.getoption("--snaptol-show-diff")

    if not snaptol_update and not snaptol_update_all:
        return

    if not items:
        return

    # If normal update then we only update the tests that previously failed.
    if snaptol_update:
        lastfailed = _get_cache(config.cache, "cache/lastfailed")

        # If none failed last then we don't need to update anything.
        if not lastfailed:
            config.hook.pytest_deselected(items=items)
            items[:] = []
            return

        # We have some failed tests. Remove any that passed.
        to_keep = [item for item in items if item.nodeid in lastfailed]
        to_deselect = [item for item in items if item.nodeid not in lastfailed]

        if to_deselect:
            config.hook.pytest_deselected(items=to_deselect)

        items[:] = to_keep

    if snaptol_use_cache:
        all_cache = {
            item.nodeid: _get_cache(config.cache, nodeid_to_key(item.nodeid))
            for item in items
        }

        to_keep = []
        to_deselect = []

        for item in items:
            entry = all_cache.get(item.nodeid)

            if entry is None:
                to_keep.append(item)
                continue

            snapshot_file = Path(entry["snapshot_file"])
            data = deserialise_snapshot(entry["data"])

            if snaptol_show_diff:
                _store_test_diff(
                    config,
                    snapshot_file,
                    before=read_snapshot(snapshot_file),
                    after=data,
                )
            write_snapshot(snapshot_file, data)

            to_deselect.append(item)

            # Stash away the node IDs of the tests that were updated from cache.
            config.stash.setdefault(CACHE_STASH_KEY, []).append(item.nodeid)

        if to_deselect:
            config.hook.pytest_deselected(items=to_deselect)

            for nodeid in [item.nodeid for item in to_deselect]:
                _uncache_test(config.cache, nodeid)

        items[:] = to_keep


def pytest_deselected(items: list[pytest.Item]):
    """
    Stores deselected test items for later processing during the test session cleanup.
    This hook is called when tests are deselected (e.g., by using test markers or keywords).

    Parameters
    ----------
    items
        List of pytest test items that were deselected during test collection.
    """

    if _snaptol:
        _snaptol.add_deselected(items)


def pytest_sessionstart(session: Any) -> None:
    """
    Initialize snapshot session before tests are collected and ran.
    https://docs.pytest.org/en/latest/reference.html#_pytest.hookspec.pytest_sessionstart
    """

    session.config._snaptol = SnaptolSession(
        pytest_session=session,
        default_snapshot_dirname=session.config.option.snaptol_dirname,
    )
    global _snaptol  # ruff: ignore[PLW0603]
    _snaptol = session.config._snaptol


def pytest_sessionfinish(session: Any):
    """
    Runs after all tests are completed. When the ``--snaptol-update`` option
    is enabled, it scans through all test items (including deselected ones) to
    identify relevant snapshot files. Any snapshot file that is not associated
    with an existing test using the `snaptolshot` fixture will be deleted,
    ensuring only active snapshots are maintained.

    Parameters
    ----------
    session
        The pytest session object containing test execution information.
    """

    session.config._snaptol.finish()


def pytest_terminal_summary(
    terminalreporter: pytest.TerminalReporter, config: pytest.Config
):
    """
    Reports to the terminal information regarding the tests performed and any information
    requested by the user during the test session, such as snapshot differences or cache usage.

    Parameters
    ----------
    terminalreporter
        The terminal reporter object used for writing to the terminal.

    config
        The pytest configuration object containing the state and options of the test session.
    """
    if diffs := config.stash.get(DIFFS_STASH_KEY, []):
        terminalreporter.ensure_newline()
        terminalreporter.write_line(" -+- snaptol diffs -+-", bold=True)

        for diff in diffs:
            _show_test_diff(
                terminalreporter, diff.snapshot_file, diff.before, diff.after
            )

    if nodeids_used_cache := config.stash.get(CACHE_STASH_KEY, []):
        terminalreporter.ensure_newline()
        terminalreporter.write_line(
            " Used snaptol cache data to update the following test(s):", bold=True
        )
        terminalreporter.write_line(" - " + "\n - ".join(nodeids_used_cache))

    if deleted_snapshot_files := config.stash.get(DELETED_STASH_KEY, []):
        terminalreporter.ensure_newline()
        terminalreporter.write_line(
            " Removed the following snapshot file(s) because they were not used by any test:",
            bold=True,
        )
        terminalreporter.write_line(
            " - " + "\n - ".join(map(str, deleted_snapshot_files))
        )

    if deletable_snapshot_files := config.stash.get(DELETABLE_STASH_KEY, []):
        terminalreporter.ensure_newline()
        terminalreporter.write_line(
            " The following snapshot file(s) could be deleted in update mode because they are not used by any test:",
            bold=True,
        )
        terminalreporter.write_line(
            " - " + "\n - ".join(map(str, deletable_snapshot_files))
        )
