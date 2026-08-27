"""Strict local JSONL text source."""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Iterable, Mapping
from ..provenance import canonical_json_sha256, file_sha256


class JsonlDocumentSource:
    def __init__(self, path: str | Path, *, text_field: str = "text") -> None:
        self.path = Path(path).expanduser().resolve()
        self.text_field = text_field
        if not self.path.is_file():
            raise FileNotFoundError(f"JSONL file does not exist: {self.path}")
        if not text_field:
            raise ValueError("text_field must not be empty")

    def documents(self) -> Iterable[str]:
        found = False
        with self.path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"{self.path}:{line_number} is not valid JSON"
                    ) from error
                text = (
                    record.get(self.text_field) if isinstance(record, Mapping) else None
                )
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(
                        f"{self.path}:{line_number} must contain non-empty string field {self.text_field!r}"
                    )
                found = True
                yield text
        if not found:
            raise ValueError(f"JSONL source contains no documents: {self.path}")

    def fingerprint(self) -> str:
        return canonical_json_sha256(
            {
                "type": "jsonl",
                "file_sha256": file_sha256(self.path),
                "text_field": self.text_field,
            }
        )

    def metadata(self) -> Mapping[str, Any]:
        return {
            "type": "jsonl",
            "path": str(self.path),
            "text_field": self.text_field,
            "file_sha256": file_sha256(self.path),
        }
