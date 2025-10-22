from pathlib import Path


class SnapshotTracker:
    """Tracks snapshot files touched during a test session."""

    def __init__(self):
        self.snapshot_dirs: set[Path] = set()
        self.touched_snapshot_files: set[Path] = set()

    def touch(self, snapshot_file: Path):
        self.snapshot_dirs.add(snapshot_file.parent)
        self.touched_snapshot_files.add(snapshot_file)
