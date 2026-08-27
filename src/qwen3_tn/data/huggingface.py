"""Generic Hugging Face Dataset document source."""

from __future__ import annotations
from typing import Any, Iterable, Mapping
from ..provenance import canonical_json_sha256


class HFDatasetDocumentSource:
    def __init__(
        self,
        dataset: str,
        *,
        config: str | None = None,
        split: str = "train",
        text_field: str = "text",
        load_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if not dataset or not split or not text_field:
            raise ValueError("dataset, split, and text_field must not be empty")
        self.dataset_name = dataset
        self.config = config
        self.split = split
        self.text_field = text_field
        self.load_kwargs = dict(load_kwargs or {})
        self._dataset: Any | None = None

    def _load(self) -> Any:
        if self._dataset is None:
            from datasets import load_dataset

            self._dataset = load_dataset(
                self.dataset_name, self.config, split=self.split, **self.load_kwargs
            )
            if len(self._dataset) == 0:
                raise ValueError("Hugging Face dataset split is empty")
        return self._dataset

    def documents(self) -> Iterable[str]:
        for index, record in enumerate(self._load()):
            text = record.get(self.text_field) if isinstance(record, Mapping) else None
            if not isinstance(text, str):
                raise ValueError(
                    f"dataset record {index} has no string field {self.text_field!r}"
                )
            yield text

    def fingerprint(self) -> str:
        dataset_fingerprint = str(getattr(self._load(), "_fingerprint", ""))
        descriptor = {**self.metadata(), "dataset_fingerprint": dataset_fingerprint}
        return canonical_json_sha256(descriptor)

    def metadata(self) -> Mapping[str, Any]:
        return {
            "type": "huggingface",
            "dataset": self.dataset_name,
            "config": self.config,
            "split": self.split,
            "text_field": self.text_field,
            "load_kwargs": self.load_kwargs,
        }
