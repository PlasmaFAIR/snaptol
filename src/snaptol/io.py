from json import dumps, loads
from pathlib import Path
from typing import Any

import pytest
from numpy import ndarray

CACHE_KEY = "snaptol"


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


def _cache_failed_test(
    cache: pytest.Cache, nodeid: str, snapshot_file: Path, data: Any
):
    """
    Caches the snapshot data from a failed test to enable later regeneration without re-running the test.
    This allows the ``--use-snaptol-cache`` option to update snapshots using cached data.
    Serialises the data to JSON format if possible, falling back to string representation if needed.

    Parameters
    ----------
    cache
        The pytest cache object used to store and retrieve test data.
    nodeid
        The unique identifier of the test node whose snapshot data is being cached.
    snapshot_file
        The path to the snapshot file associated with the test.
    data
        The snapshot data to be cached, which will be serialised if possible.
    """

    try:
        dumps(data)
    except (TypeError, OverflowError):
        data = _json_fallback(data)

    cached = cache.get(CACHE_KEY, {})

    cached[nodeid] = {"snapshot_file": str(snapshot_file), "data": data}

    cache.set(CACHE_KEY, cached)


def _uncache_test(cache: pytest.Cache, nodeid: str):
    """
    Removes a test entry from the snaptol cache after it has been successfully updated.
    This is typically called when a snapshot has been regenerated normally without using the cache.

    Parameters
    ----------
    cache
        The pytest cache object used to store and retrieve test data.
    nodeid
        The unique identifier of the test node to be removed from the cache.
    """

    cached = cache.get(CACHE_KEY, {})

    cached.pop(nodeid, None)

    cache.set(CACHE_KEY, cached)
