"""Dataset-neutral causal-LM data contracts."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Protocol, runtime_checkable
from torch.utils.data import DataLoader


@runtime_checkable
class DocumentSource(Protocol):
    def documents(self) -> Iterable[str]: ...
    def fingerprint(self) -> str: ...
    def metadata(self) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class PreparedCausalLMData:
    dataloader: DataLoader[Any]
    fingerprint: str
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.fingerprint:
            raise ValueError("prepared data fingerprint must not be empty")
        if len(self.dataloader) == 0:
            raise ValueError("prepared dataloader must not be empty")


def prepared_data_from_dataloader(
    dataloader: DataLoader[Any], *, fingerprint: str, metadata: Mapping[str, Any]
) -> PreparedCausalLMData:
    """Wrap custom/prebuilt data; callers must supply provenance explicitly."""
    return PreparedCausalLMData(dataloader, fingerprint, dict(metadata))
