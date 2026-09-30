from itertools import chain
from typing import Any
from enum import Enum
from collections.abc import Iterable
from pathlib import Path
from dataclasses import field, dataclass


import pytest

from .io import snapshot_filename, DELETED_STASH_KEY, DELETABLE_STASH_KEY
from .snapshot import Snapshot


class ItemStatus(Enum):
    NOT_RUN = False
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class SnaptolSession:
    pytest_session: pytest.Session

    # All the collected test items, keyed by nodeid to preserve collection order
    _collected_items: dict[str, pytest.Item] = field(default_factory=dict)
    _deselected_items: list[pytest.Item] = field(default_factory=list)

    @staticmethod
    def filter_valid_items(items: list["pytest.Item"]) -> Iterable["pytest.Item"]:
        return (item for item in items if isinstance(item, pytest.Function))

    def collect_items(self, items: list["pytest.Item"]) -> None:
        for item in self.filter_valid_items(items):
            self._collected_items[item.nodeid] = item

    def add_deselected(self, items: list[pytest.Item]) -> None:
        for item in self.filter_valid_items(items):
            self._deselected_items.append(item)

    def finish(self):
        config = self.pytest_session.config
        last_failed = config.getoption("--last-failed") or config.getoption("--lf")
        if last_failed:
            # Can't accurately determine unused snapshots when running with last-failed, so bail
            return

        snaptol_update = config.getoption("--snaptol-update")
        snaptol_update_all = config.getoption("--snaptol-update-all")

        # The items (tests) that are in the session are relevant and thus their snapshot files musn't be deleted.
        relevant_snapshot_files = []
        snapshot_dirs = set()

        # We loop through the session items and items that were deselected (e.g by keyword).
        for item in chain(self._collected_items.values(), self._deselected_items):
            snapshot_file = snapshot_filename(item, test_dir=Path(item.fspath).parent)
            snapshot_dirs.add(snapshot_file.parent)

            if not snapshot_file.exists():
                continue

            # A test may still exist that used to have a snapshot file but no longer does -> if so, it's not relevant.
            if "snaptolshot" not in getattr(item, "fixturenames", ()):
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
                        self.pytest_session.config.stash.setdefault(
                            DELETED_STASH_KEY, []
                        ).append(path)

                    else:
                        # Otherwise, stash away the file name to alert the user that it could be deleted.
                        self.pytest_session.config.stash.setdefault(
                            DELETABLE_STASH_KEY, []
                        ).append(path)
