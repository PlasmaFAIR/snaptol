from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from .io import DELETABLE_STASH_KEY, DELETED_STASH_KEY

if TYPE_CHECKING:
    from .snapshot import Snapshot


class ItemStatus(Enum):
    NOT_RUN = False
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class SnaptolSession:
    pytest_session: pytest.Session
    default_snapshot_dirname: Path | str = "__snapshots__"

    _snapshots: list["Snapshot"] = field(default_factory=list)
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

    def register_request(self, snapshot: "Snapshot"):
        self._snapshots.append(snapshot)

    def snapshot_directory(self, base_dir: Path) -> Path:
        """Directory where snapshot files will be stored, relative to `base_dir`"""
        return base_dir / self.default_snapshot_dirname

    def snapshot_filename(self, base_dir: Path, nodeid: str) -> Path:
        """Full path to a snapshot file for a given test nodeid"""
        return (
            self.snapshot_directory(base_dir)
            / f"{Path(nodeid.replace(':', '_')).name}.json"
        )

    @staticmethod
    def _snapshot_file_matches_test(test_file: Path, path: Path) -> bool:
        """Does `path` match `test_file` or a parameterisation/multiple assert version?"""
        if test_file == path:
            return True
        if test_file.parent != path.parent:
            return False

        return path.stem.startswith(f"{test_file.stem}[")

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

        # Go over tests we know definitely ran:
        for snapshot in self._snapshots:
            snapshot_dirs.add(snapshot.snapshot_dir)

            # Mark all the files from assertions that ran
            for result in snapshot._execution_results.values():
                relevant_snapshot_files.append(result.filename)

        # We loop through the session items that were deselected (e.g by keyword).
        for item in self._deselected_items:
            snapshot_file = self.snapshot_filename(item.path.parent, item.nodeid)
            snapshot_dir = self.snapshot_directory(item.path.parent)
            snapshot_dirs.add(snapshot_dir)

            # A test may still exist that used to have a snapshot file but no
            # longer does -> if so, it's not relevant.
            if "snaptolshot" not in getattr(item, "fixturenames", ()):
                continue

            # Mark partial matching files as being relevant
            for path in snapshot_dir.glob("*.json"):
                if self._snapshot_file_matches_test(snapshot_file, path):
                    relevant_snapshot_files.append(path)

        # Now go through collected items that _don't_ use our fixture:
        for item in self._collected_items.values():
            if "snaptolshot" in getattr(item, "fixturenames", ()):
                continue
            snapshot_dirs.add(self.snapshot_directory(item.path.parent))

        # We now have all the relevant snapshot files -> delete snapshots that are not included in the list.
        for snapshot_dir in snapshot_dirs:
            if not snapshot_dir.exists():
                continue

            for path in snapshot_dir.glob("*.json"):
                if path in relevant_snapshot_files:
                    continue
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

            # If there's no more snapshots left, remove the directory
            if len(list(snapshot_dir.iterdir())) == 0:
                snapshot_dir.rmdir()
