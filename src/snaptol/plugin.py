import pytest

from ._snapshot import Snapshot
from ._tracker import SnapshotTracker


def pytest_addoption(parser):
    parser.addoption(
        "--snapshot-update",
        action="store_true",
        default=False,
        help="Update snaptol snapshot files",
    )


def pytest_configure(config):
    config.snaptol_tracker = SnapshotTracker()


def pytest_sessionfinish(session):
    if not session.config.getoption("--snapshot-update"):
        return

    # Skip cleanup if a -k filter was passed.
    if session.config.getoption("keyword"):
        return

    tracker: SnapshotTracker = session.config.snaptol_tracker

    for snapshot_dir in tracker.snapshot_dirs:
        for path in snapshot_dir.glob("*.json"):
            if path not in tracker.touched_snapshot_files:
                path.unlink(missing_ok=True)


@pytest.fixture
def snapshot(request) -> Snapshot:
    snap = Snapshot.from_request(request)

    request.config.snaptol_tracker.touch(snap.snapshot_file)

    return snap
