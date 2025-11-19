import pytest

from ._io import snapshot_filename
from ._snapshot import Snapshot

_deselected_items = []


@pytest.fixture
def snapshot(request) -> Snapshot:
    """
    A pytest fixture that provides a Snapshot object tied to the current test request.
    Returns the instanciated Snapshot object.

    Parameters
    ----------
        request
            The pytest request object containing test context information.
    """

    return Snapshot.from_request(request)


def pytest_addoption(parser):
    """
    Adds the '--snapshot-update' command line option to pytest.
    This option enables updating or cleaning up snapshot files during test execution.

    Parameters
    ----------
        parser: The pytest command line parser to which the option will be added.
    """

    parser.addoption(
        "--snapshot-update",
        action="store_true",
        default=False,
        help="Update snaptol snapshot files",
    )


def pytest_deselected(items):
    """
    Stores deselected test items for later processing during the test session cleanup.
    This hook is called when tests are deselected (e.g., by using test markers or keywords).

    Parameters
    ----------
        items: List of pytest test items that were deselected during test collection.
    """

    global _deselected_items  # noqa: PLW0603

    _deselected_items = items


def pytest_sessionfinish(session):
    """
    Runs after all tests are completed. When '--snapshot-update' option is enabled, it scans through
    all test items (including deselected ones) to identify relevant snapshot files. Any snapshot file
    that is not associated with an existing test using the 'snapshot' fixture will be deleted, ensuring
    only active snapshots are maintained.

    Parameters
    ----------
        session: The pytest session object containing test execution information.
    """

    if not session.config.getoption("--snapshot-update"):
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
        if "snapshot" not in item.fixturenames:
            continue

        relevant_snapshot_files.append(snapshot_file)

    # We now have all the relevant snapshot files -> delete snapshots that are not included in the list.
    for snapshot_dir in snapshot_dirs:
        for path in snapshot_dir.glob("*.json"):
            if path not in relevant_snapshot_files:
                path.unlink(missing_ok=True)
