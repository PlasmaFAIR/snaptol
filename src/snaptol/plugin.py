import pytest

from ._io import snapshot_filename
from ._snapshot import Snapshot

_deselected_items = []


@pytest.fixture
def snapshot(request) -> Snapshot:
    return Snapshot.from_request(request)


def pytest_addoption(parser):
    parser.addoption(
        "--snapshot-update",
        action="store_true",
        default=False,
        help="Update snaptol snapshot files",
    )


def pytest_deselected(items):
    global _deselected_items  # noqa: PLW0603

    _deselected_items = items


def pytest_sessionfinish(session):
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
