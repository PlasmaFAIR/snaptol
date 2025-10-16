from __future__ import annotations

import dataclasses

DEFAULT_RTOL = 1e-05
DEFAULT_ATOL = 1e-08


@dataclasses.dataclass(frozen=True)
class Snapshot:  # noqa: PLW1641
    name: str
    TEMP_EXPECTED: int = 123  # TODO: delete.
    rtol: float = DEFAULT_RTOL
    atol: float = DEFAULT_ATOL

    @classmethod
    def from_request(cls, request) -> Snapshot:
        name = request.node.name

        return cls(name=name)

    def __eq__(self, value) -> bool:
        # TODO: extend to support other types and consider numerical tolerance.
        if not isinstance(value, int):
            raise NotImplementedError("{type(value)} is not supported.")

        return value == self.TEMP_EXPECTED

    def __call__(
        self, *, rtol: float = DEFAULT_RTOL, atol: float = DEFAULT_ATOL
    ) -> Snapshot:
        return dataclasses.replace(self, rtol=rtol, atol=atol)

    def match(
        self, value, *, rtol: float = DEFAULT_RTOL, atol: float = DEFAULT_ATOL
    ) -> bool:
        return value == self(rtol=rtol, atol=atol)
