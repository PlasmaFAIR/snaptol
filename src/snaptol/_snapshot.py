from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from ._compare import compare_intelligent
from ._io import read_snapshot, write_snapshot

DEFAULT_RTOL = 1e-05
DEFAULT_ATOL = 1e-08


@dataclasses.dataclass(frozen=True)
class Snapshot:  # noqa: PLW1641
    test_name: str
    test_file: Path
    update: bool
    rtol: float = DEFAULT_RTOL
    atol: float = DEFAULT_ATOL

    @classmethod
    def from_request(cls, request) -> Snapshot:
        test_name = request.node.name
        test_file = Path(request.fspath)
        update = request.config.getoption("--snapshot-update")

        return cls(test_name=test_name, test_file=test_file, update=update)

    def __eq__(self, value: Any) -> bool:
        if self.update:
            write_snapshot(self.test_name, self.test_file, value)
            return True

        expected = read_snapshot(self.test_name, self.test_file)

        return compare_intelligent(value, expected)

    def __call__(
        self, *, rtol: float = DEFAULT_RTOL, atol: float = DEFAULT_ATOL
    ) -> Snapshot:
        return dataclasses.replace(self, rtol=rtol, atol=atol)

    def match(
        self, value, *, rtol: float = DEFAULT_RTOL, atol: float = DEFAULT_ATOL
    ) -> bool:
        return value == self(rtol=rtol, atol=atol)
