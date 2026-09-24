import dataclasses
import difflib
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

CACHE_KEY = "snaptol"
CACHE_STASH_KEY = pytest.StashKey[list[str]]()
DIFFS_STASH_KEY = pytest.StashKey[list["SnapshotDiff"]]()
DELETED_STASH_KEY = pytest.StashKey[list[Path]]()
DELETABLE_STASH_KEY = pytest.StashKey[list[Path]]()
SENTINEL = object()

COMPLEX128_VIEW_DTYPE = np.dtype([("real", np.float64), ("imag", np.float64)])
COMPLEX64_VIEW_DTYPE = np.dtype([("real", np.float32), ("imag", np.float32)])


def format_compact(json_str: str, max_line_len: int = 90) -> str:
    """Convert a JSON string in Python's indented format to a more compact one.

    Try to keep lists on one line, wrapping at `max_line_len`, everything else
    remains newline separated:

    Before:
    ```json
    {
      "a": [
        1,
        2,
        3
      ],
      "b": [
        [
          4,
          5,
          6
        ],
        [
          7,
          8,
          9
        ]
      ]
    }
    ```

    After:
    ```json
    {
      "a": [1, 2, 3],
      "b": [
        [4, 5, 6],
        [7, 8, 9]
      ]
    }
    ```

    """

    def _iter(o):
        depth = 0
        in_list = False
        just_eaten = False
        eaten_count = 0
        line_length = 0
        current_list_indent = ""

        for ch in o:
            match ch:
                case "{" if in_list and just_eaten:
                    # Dict inside a list: stop eating new lines and add one back.
                    # This is dumb but it works.
                    in_list = False
                    depth += 1
                    indent = (eaten_count - 1) * "  " + "{"
                    line_length = len(indent)
                    yield f"\n{indent}"
                case "{":
                    depth += 1
                    yield ch
                case "[":
                    # Nested list, but we can't work this out till we've already
                    # consumed the newline, so add it back
                    current_list_indent = depth * "  "
                    if depth > 0 and just_eaten:
                        yield f"\n{current_list_indent}"
                    in_list = True
                    depth += 1
                    yield ch
                case "]" | "}":
                    in_list = False
                    depth -= 1
                    yield ch
                case "\n" if in_list:
                    # Skip any newlines inside lists, along with all the indent
                    # whitespace
                    just_eaten = True
                    continue
                case " " if just_eaten:
                    eaten_count += 1
                    continue
                case "," if in_list and line_length > max_line_len:
                    # Wrap line
                    line_length = 0
                    yield f",\n{current_list_indent} "
                case "," if in_list:
                    # We'll now need to add some whitespace back as a separator
                    line_length += 2
                    yield ch + " "
                case "\n":
                    just_eaten = False
                    line_length = 0
                    yield ch
                case _:
                    just_eaten = False
                    eaten_count = 0
                    line_length += 1
                    yield ch

    return "".join(list(_iter(json_str)))


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that can handle numpy objects.

    Numpy arrays are written as JSON objects with some metadata for their dtype
    and shape.

    Complex numbers are zero-cost converted to an array of two elements first.

    The final JSON string is formatted to try and condense arrays.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def default(self, o):
        def numpy_to_dict(data, dtype):
            # We may need to keep track of more info here, such as C or F
            # ordering
            return {
                "__numpy__": True,
                "dtype": dtype,
                "data": data.tolist(),
                "shape": data.shape,
            }

        match o:
            case np.ndarray() | np.number():
                # For complex numbers, we need to view as a 2-element struct,
                # which should be cheap/free. This will then be written as a
                # list, which we'll need to convert back
                if o.dtype == "c8":
                    data = o.view(COMPLEX64_VIEW_DTYPE)
                elif o.dtype == "c16":
                    data = o.view(COMPLEX128_VIEW_DTYPE)
                else:
                    data = o

                return numpy_to_dict(data, dtype=np.lib.format.dtype_to_descr(o.dtype))
            case complex():
                # This is a Python native complex, which is always a double
                view = np.complex128(o).view(COMPLEX128_VIEW_DTYPE)
                return numpy_to_dict(view, dtype="c16")
            case _:
                pass

        return super().default(o)

    def encode(self, o):
        # We don't really have much control over the low-level formatting, so we
        return format_compact(super().encode(o))


class NumpyDecoder(json.JSONDecoder):
    """Custom JSON decoder for `NumpyEncoder` encoded JSON strings."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, object_hook=self.object_hook)

    def object_hook(self, dct):
        if "__numpy__" in dct:
            dtype = dct["dtype"]
            view_type = None
            match dtype:
                case "c8" | "<c8" | ">c8" | "=c8":
                    base_dtype = np.float32
                    view_type = np.complex64
                case "c16" | "<c16" | ">c16" | "=c16":
                    base_dtype = np.float64
                    view_type = np.complex128
                case list():
                    # Numpy requires list of 2- or 3-tuple NOT a list, JSON only
                    # has lists
                    base_dtype = np.dtype([tuple(field) for field in dtype])
                case _:
                    base_dtype = dtype

            # This is a cast: json will read everything back as `float`, so we
            # need to know the base type otherwise any view will be wrong
            data = np.array(dct["data"], dtype=base_dtype)

            if view_type is not None:
                # Cheap/free cast back to a complex type
                data = data.view(view_type)

            if dct["shape"] == []:
                # Convert to scalar -- can't just use `.item()`, because that
                # gives a Python scalar, which might be the wrong dtype, so
                # reshape to a 0D array, then scalar-index
                return data.reshape([])[()]

            # We might have an extra useless dimension, especially for complex
            return data.reshape(dct["shape"])

        return dct


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

    return json.dumps(*args, indent=2, cls=NumpyEncoder, **kwargs)


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

    return deserialise_snapshot(snapshot_file.read_text(encoding="utf-8"))


def deserialise_snapshot(data: str) -> Any:
    """
    Deserialise a snapshot from a JSON string
    """
    return json.loads(data, cls=NumpyDecoder)


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

    data = {
        "snapshot_file": str(snapshot_file),
        "data": json.dumps(data, cls=NumpyEncoder),
    }

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
