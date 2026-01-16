import pytest

from .io import snapshot_filename
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
        "--use-snaptol-cache",
        action="store_true",
        default=False,
        help="In update mode, use cached snaptol snapshot data if available",
    )

    parser.addoption(
        "--show-snaptol-cache",
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

    if config.getoption("--snaptol-update") and config.getoption(
        "--snaptol-update-all"
    ):
        raise ValueError(
            "Cannot use both --snaptol-update and --snaptol-update-all options"
        )


def pytest_deselected(items):
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

    # Don't need to clean up files if we are not running a snapshot update.
    if not session.config.getoption("--snaptol-update"):
        return

    # Don't need to clean up files if we are only running tests that previously failed.
    if session.config.getoption("--lf") or session.config.getoption("--last-failed"):
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
