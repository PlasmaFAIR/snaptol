from importlib.metadata import (
    PackageNotFoundError as _PackageNotFoundError,
)
from importlib.metadata import (
    version as _version,
)

from .snapshot import compare_intelligent

try:
    __version__ = _version(__name__)
except _PackageNotFoundError:
    __version__ = "0.0.0"


__all__ = ["compare_intelligent"]
