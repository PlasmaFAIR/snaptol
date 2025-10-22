from json import dumps, loads
from pathlib import Path
from typing import Any

from numpy import ndarray


def snapshot_dir(test_file: Path) -> Path:
    return test_file.parent / "__snapshots__"


def snapshot_filename(test_name: str, test_file: Path) -> Path:
    return snapshot_dir(test_file) / f"{test_file.stem}.{test_name}.json"


def write_snapshot(test_name: str, test_file: Path, value: Any):
    jsoned = dumps(value, indent=2, default=_json_fallback)

    path = snapshot_filename(test_name, test_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(jsoned, encoding="utf-8")


def read_snapshot(test_name: str, test_file: Path) -> Any:
    path = snapshot_filename(test_name, test_file)
    return loads(path.read_text(encoding="utf-8"))


def _json_fallback(value: Any) -> Any:
    try:
        if isinstance(value, ndarray):
            return value.tolist()
    except Exception:
        pass

    return repr(value)
