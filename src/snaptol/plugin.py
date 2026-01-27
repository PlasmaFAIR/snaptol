from pathlib import Path

import pytest

from .io import (
    CACHE_STASH_KEY,
    DELETABLE_STASH_KEY,
    DELETED_STASH_KEY,
    DIFFS_STASH_KEY,
    _get_cache,
    _show_test_diff,
    _store_test_diff,
    _uncache_test,
    nodeid_to_key,
    read_snapshot,
    snapshot_filename,
    write_snapshot,
)
from .snapshot import Snapshot

_deselected_items = []


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

    parser.addoption(
        "--snaptol-update",
        action="store_true",
        default=False,
        help="Update snaptol snapshot files of previously failed tests",
    )

    parser.addoption(
        "--snaptol-update-all",
        action="store_true",
        default=False,
        help="Update all snaptol snapshot files",
    )

    parser.addoption(
        "--snaptol-use-cache",
        action="store_true",
        default=False,
        help="In update mode, use cached snaptol snapshot data if available",
    )

    parser.addoption(
        "--snaptol-show-cache",
        action="store_true",
        default=False,
        help="Show cached snaptol snapshot data",
    )

    parser.addoption(
        "--snaptol-clear-cache",
        action="store_true",
        default=False,
        help="Clear cached snaptol snapshot data",
    )

    parser.addoption(
        "--snaptol-show-diff",
        action="store_true",
        default=False,
        help="Show diff in update mode when snapshot data does not match data on file",
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


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]):
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
            entry = all_cache.get(item.nodeid, None)

            if entry is None:
                to_keep.append(item)
                continue

            snapshot_file = Path(entry["snapshot_file"])
            data = entry["data"]

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

    global _deselected_items  # noqa: PLW0603

    _deselected_items += items


def pytest_sessionfinish(session: pytest.Session):
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

    snaptol_update = session.config.getoption("--snaptol-update")
    snaptol_update_all = session.config.getoption("--snaptol-update-all")

    # The items (tests) that are in the session are relevant and thus their snapshot files musn't be deleted.
    relevant_snapshot_files = []
    snapshot_dirs = set()

    # We loop through the session items and items that were deselected (e.g by keyword).
    for item in session.items + _deselected_items:
        snapshot_file = snapshot_filename(item.originalname, item.path)
        snapshot_dirs.add(snapshot_file.parent)

        if not snapshot_file.exists():
            continue

        # A test may still exist that used to have a snapshot file but no longer does -> if so, it's not relevant.
        if "snaptolshot" not in item.fixturenames:
            continue

        relevant_snapshot_files.append(snapshot_file)

    # We now have all the relevant snapshot files -> delete snapshots that are not included in the list.
    for snapshot_dir in snapshot_dirs:
        for path in snapshot_dir.glob("*.json"):
            if path not in relevant_snapshot_files:
                # Delete the snapshotfile if we are in an update mode.
                if snaptol_update or snaptol_update_all:
                    path.unlink(missing_ok=True)

                    # Stash away the deleted snapshot file paths for later reporting.
                    session.config.stash.setdefault(DELETED_STASH_KEY, []).append(path)

                else:
                    # Otherwise, stash away the file name to alert the user that it could be deleted.
                    session.config.stash.setdefault(DELETABLE_STASH_KEY, []).append(
                        path
                    )


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
