from json import dumps, loads
from pathlib import Path
from typing import Any

from numpy import ndarray


def snapshot_filename(test_name: str, test_file: Path) -> Path:
    """
    Generates a snapshot filename based on the test name and file path. Returns a Path object
    with a '.json' extension.

    Parameters
    ----------
    test_name
        The name of the test for which the snapshot is being created.
    test_file
        The path to the test file containing the test.
    """

    return test_file.parent / "__snapshots__" / f"{test_file.stem}.{test_name}.json"


def write_snapshot(snapshot_file: Path, value: Any):
    """
    Writes a snapshot to its file in JSON format.

    Parameters
    ----------
    snapshot_file
        The path where the snapshot file will be written.
    value
        The value to be serialised and written to the snapshot file.
    """

    jsoned = dumps(value, indent=2, default=_json_fallback)

    snapshot_file.parent.mkdir(parents=True, exist_ok=True)
    snapshot_file.write_text(jsoned, encoding="utf-8")


def read_snapshot(snapshot_file: Path) -> Any:
    """
    Reads and deserialises a snapshot from a JSON file. Returns the deserialised content.

    Parameters
    ----------
    snapshot_file
        The path to the snapshot file to be read.
    """

    return loads(snapshot_file.read_text(encoding="utf-8"))


def _json_fallback(value: Any) -> Any:
    """
    A fallback function for JSON serialisation that handles special data types.
    Converts numpy arrays to lists and other non-serialisable objects to their string representation.
    Returns the serialised value.

    Parameters
    ----------
    value
        The value to be serialised to JSON format.
    """

    try:
        if isinstance(value, ndarray):
            return value.tolist()
    except Exception:
        pass

    return repr(value)
