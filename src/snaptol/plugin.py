from pathlib import Path

import pytest

from .io import _get_cache, _set_cache, snapshot_filename, write_snapshot
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

    if not snaptol_update and not snaptol_update_all:
        return

    if not items:
        return

    # If normal update then we only update the tests that previously failed.
    if snaptol_update:
        lastfailed = _get_cache(config.cache, cache_key="cache/lastfailed")

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
        cached = _get_cache(config.cache)

        to_keep = []
        to_deselect = []

        for item in items:
            entry = cached.get(item.nodeid, None)

            if entry is None:
                to_keep.append(item)
                continue

            snapshot_file = Path(entry["snapshot_file"])
            data = entry["data"]

            write_snapshot(snapshot_file, data)

            to_deselect.append(item)

        if to_deselect:
            config.hook.pytest_deselected(items=to_deselect)

            for nodeid in [item.nodeid for item in to_deselect]:
                cached.pop(nodeid, None)

            _set_cache(config.cache, cached)

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

    _deselected_items = items


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

    # Don't need to clean up files if we are not running a full snapshot update.
    if not session.config.getoption("--snaptol-update-all"):
        return

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
                path.unlink(missing_ok=True)
