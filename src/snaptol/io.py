import dataclasses
import difflib
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from numpy import ndarray

CACHE_KEY = "snaptol"
CACHE_STASH_KEY = pytest.StashKey[list[str]]()
DIFFS_STASH_KEY = pytest.StashKey[list["SnapshotDiff"]]()
DELETED_STASH_KEY = pytest.StashKey[list[Path]]()
DELETABLE_STASH_KEY = pytest.StashKey[list[Path]]()
SENTINEL = object()


def snapshot_filename(nodeid: str, test_dir: Path) -> Path:
    """
    Generates a snapshot filename based on the test nodeid. Returns a Path object
    with a '.json' extension.

    Parameters
    ----------
    nodeid
        The nodeid of the test.
    test_dir
        The directory where the test lives.
    """

    return snapshot_directory(test_dir) / f"{Path(nodeid.replace(':', '_')).name}.json"


def snapshot_directory(test_dir: Path) -> Path:
    """
    Generates the directory where snapshot files will be stored. Returns a Path object.

    Parameters
    ----------
    test_dir
        The directory where the test lives.
    """

    return test_dir / "__snapshots__"


def json_dump(*args, **kwargs) -> str:
    """
    Serialises Python objects to a JSON formatted string with indentation.

    Wraps the `json.dumps` method and adds a default indentation of 2 spaces
    as well as a fallback function for unsupported types during serialisation.

    Parameters
    ----------
    *args
        Positional arguments to be passed to `json.dumps`.
    **kwargs
        Keyword arguments to be passed to `json.dumps`.
    """

    return json.dumps(*args, indent=2, default=_json_fallback, **kwargs)


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

    jsoned = json_dump(value)

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

    return json.loads(snapshot_file.read_text(encoding="utf-8"))


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


def nodeid_to_key(nodeid: str) -> str:
    """
    Get the unique cache key based on the nodeid.

    Parameters
    ----------
    nodeid
        The node ID of a test.
    """
    digest = hashlib.sha1(nodeid.encode("utf-8")).hexdigest()

    return f"{CACHE_KEY}/{digest}"


def _get_cache(cache: pytest.Cache, cache_key: str) -> dict:
    """
    Gets the snaptol cache from the pytest cache object. Returns an empty dictionary if no cache exists.

    Parameters
    ----------
    cache
        The pytest cache object used to store and retrieve test data.
    cache_key
        The unique key used to identify the cache entry.
    """
    return cache.get(cache_key, None)


def _set_cache(cache: pytest.Cache, data: Any, cache_key: str) -> None:
    """
    Sets the snaptol cache for a test in the pytest cache object.

    Parameters
    ----------
    cache
        The pytest cache object used to store and retrieve test data.
    data
        The data to be stored in the snaptol cache.
    cache_key
        The unique key used to identify the cache entry.
    """
    cache.set(cache_key, data)


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
        json.dumps(data)
    except (TypeError, OverflowError):
        data = _json_fallback(data)

    data = {"snapshot_file": str(snapshot_file), "data": data}

    _set_cache(cache, data, nodeid_to_key(nodeid))


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

    # Pytest stores values under <cachedir>/v/<key> - remove the file entirely. If we fail, set cache to nothing.
    try:
        path = cache._cachedir / "v" / nodeid_to_key(nodeid)
        path.unlink(missing_ok=True)
    except Exception:
        _set_cache(cache, None, nodeid_to_key(nodeid))


def _store_test_diff(
    config: pytest.Config,
    snapshot_file: Path,
    before: Any,
    after: Any,
):
    """
    Stores the before and after diff of a snapshot for later printing in the terminal reporter.

    Parameters
    ----------
    config
        The pytest configuration object.
    snapshot_file
        The path to the snapshot file associated with the test.
    before
        The current snapshot data on file.
    after
        The snapshot data generated by the test.
    """

    before = json_dump(before).splitlines() if before is not SENTINEL else []
    after = json_dump(after).splitlines() if after is not SENTINEL else []

    # Stash away the before and after diff of this test snapshot for later printing.
    config.stash.setdefault(DIFFS_STASH_KEY, []).append(
        SnapshotDiff(snapshot_file=snapshot_file, before=before, after=after)
    )


def _show_test_diff(
    terminalreporter: pytest.TerminalReporter,
    snapshot_file: Path,
    before: Any,
    after: Any,
):
    """
    Uses the given terminal reporter to pretty print the before and after diff of a snapshot.

    Parameters
    ----------
    terminalreporter
        The pytest terminal reporter object.
    snapshot_file
        The path to the snapshot file associated with the test.
    before
        The current snapshot data on file.
    after
        The snapshot data generated by the test.
    """
    terminalreporter.ensure_newline()
    terminalreporter.write_line("-" * 80, bold=True)
    terminalreporter.write_line(f" Snapshot: {snapshot_file}", bold=True)
    terminalreporter.write_line("")

    diff = difflib.unified_diff(
        before,
        after,
        fromfile="before",
        tofile="after",
        lineterm="",
    )

    for line in diff:
        if line.startswith("+") and not line.startswith("+++"):
            terminalreporter.write_line(line, green=True)
        elif line.startswith("-") and not line.startswith("---"):
            terminalreporter.write_line(line, red=True)
        else:
            terminalreporter.write_line(line)


@dataclasses.dataclass(frozen=True)
class SnapshotDiff:
    snapshot_file: Path
    before: Any
    after: Any
