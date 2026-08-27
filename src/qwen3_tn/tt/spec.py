"""Shape and rank definitions for TT-matrices."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import prod
from typing import Any, Iterable, Mapping, Sequence


def _positive_tuple(values: Iterable[int], name: str) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if not result or any(value <= 0 for value in result):
        raise ValueError(f"{name} must contain positive integers")
    return result


@dataclass(frozen=True)
class TTMatrixSpec:
    """Canonical TT-matrix core layout is ``[r_left, out_mode, in_mode, r_right]``."""

    out_modes: tuple[int, ...]
    in_modes: tuple[int, ...]
    ranks: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "out_modes", _positive_tuple(self.out_modes, "out_modes")
        )
        object.__setattr__(self, "in_modes", _positive_tuple(self.in_modes, "in_modes"))
        object.__setattr__(self, "ranks", _positive_tuple(self.ranks, "ranks"))
        if len(self.out_modes) != len(self.in_modes):
            raise ValueError("out_modes and in_modes must have the same length")
        if len(self.ranks) != self.order + 1:
            raise ValueError("ranks must contain order + 1 entries")
        if self.ranks[0] != 1 or self.ranks[-1] != 1:
            raise ValueError("TT boundary ranks must both be 1")
        for index, (rank, maximum) in enumerate(
            zip(self.ranks[1:-1], self.maximum_internal_ranks, strict=True), start=1
        ):
            if rank > maximum:
                raise ValueError(f"rank r{index}={rank} exceeds maximum {maximum}")

    @property
    def order(self) -> int:
        return len(self.out_modes)

    @property
    def out_features(self) -> int:
        return prod(self.out_modes)

    @property
    def in_features(self) -> int:
        return prod(self.in_modes)

    @property
    def physical_modes(self) -> tuple[int, ...]:
        return tuple(m * n for m, n in zip(self.out_modes, self.in_modes, strict=True))

    @property
    def maximum_internal_ranks(self) -> tuple[int, ...]:
        modes = self.physical_modes
        return tuple(
            min(prod(modes[:i]), prod(modes[i:])) for i in range(1, self.order)
        )

    @property
    def core_shapes(self) -> tuple[tuple[int, int, int, int], ...]:
        return tuple(
            (self.ranks[i], self.out_modes[i], self.in_modes[i], self.ranks[i + 1])
            for i in range(self.order)
        )

    @property
    def num_parameters(self) -> int:
        return sum(prod(shape) for shape in self.core_shapes)

    @property
    def dense_num_parameters(self) -> int:
        return self.out_features * self.in_features

    @property
    def compression_ratio(self) -> float:
        return self.dense_num_parameters / self.num_parameters

    def with_bond_rank(self, bond_index: int, rank: int) -> "TTMatrixSpec":
        if not 1 <= bond_index < self.order:
            raise ValueError(f"bond_index must be in [1, {self.order - 1}]")
        values = list(self.ranks)
        values[bond_index] = int(rank)
        return replace(self, ranks=tuple(values))

    @classmethod
    def full_rank(
        cls, out_modes: Sequence[int], in_modes: Sequence[int]
    ) -> "TTMatrixSpec":
        out_values = _positive_tuple(out_modes, "out_modes")
        in_values = _positive_tuple(in_modes, "in_modes")
        if len(out_values) != len(in_values):
            raise ValueError("out_modes and in_modes must have the same length")
        physical = tuple(m * n for m, n in zip(out_values, in_values, strict=True))
        internal = tuple(
            min(prod(physical[:i]), prod(physical[i:])) for i in range(1, len(physical))
        )
        return cls(out_values, in_values, (1, *internal, 1))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TTMatrixSpec":
        try:
            return cls(
                tuple(value["out_modes"]),
                tuple(value["in_modes"]),
                tuple(value["ranks"]),
            )
        except (KeyError, TypeError) as error:
            raise ValueError("invalid TTMatrixSpec mapping") from error

    def to_dict(self) -> dict[str, Any]:
        return {
            "out_modes": list(self.out_modes),
            "in_modes": list(self.in_modes),
            "ranks": list(self.ranks),
            "out_features": self.out_features,
            "in_features": self.in_features,
            "num_parameters": self.num_parameters,
            "dense_num_parameters": self.dense_num_parameters,
            "compression_ratio": self.compression_ratio,
        }
