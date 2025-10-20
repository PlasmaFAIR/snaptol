import pytest

from ._snapshot import Snapshot


def pytest_addoption(parser):
    parser.addoption(
        "--snapshot-update",
        action="store_true",
        default=False,
        help="Update snaptol snapshot files",
    )


@pytest.fixture
def snapshot(request) -> Snapshot:
    return Snapshot.from_request(request)
