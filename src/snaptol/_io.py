from json import dumps, loads
from pathlib import Path
from typing import Any

from numpy import ndarray


def snapshot_filename(test_name: str, test_file: Path) -> Path:
    return test_file.parent / "__snapshots__" / f"{test_file.stem}.{test_name}.json"


def write_snapshot(snapshot_file: Path, value: Any):
    jsoned = dumps(value, indent=2, default=_json_fallback)

    snapshot_file.parent.mkdir(parents=True, exist_ok=True)
    snapshot_file.write_text(jsoned, encoding="utf-8")


def read_snapshot(snapshot_file: Path) -> Any:
    return loads(snapshot_file.read_text(encoding="utf-8"))


def _json_fallback(value: Any) -> Any:
    try:
        if isinstance(value, ndarray):
            return value.tolist()
    except Exception:
        pass

    return repr(value)
