import pytest

from ._snapshot import Snapshot


@pytest.fixture
def snapshot(request) -> Snapshot:
    return Snapshot.from_request(request)
