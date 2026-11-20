"""JSON Lines dataset reader."""

import json
from pathlib import Path
from typing import Iterator

from pydantic import ValidationError

from assay.models import DatasetExample


def load_dataset(path: str | Path) -> list[DatasetExample]:
    """Read and validate all examples from a UTF-8 JSONL file."""
    dataset_path = Path(path)
    examples: list[DatasetExample] = []
    seen_ids: set[str] = set()
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                example = DatasetExample.model_validate(json.loads(line))
            except (json.JSONDecodeError, ValidationError) as exc:
                raise ValueError(f"Invalid dataset row at {dataset_path}:{line_number}: {exc}") from exc
            if example.id in seen_ids:
                raise ValueError(f"Duplicate example id {example.id!r} at line {line_number}")
            seen_ids.add(example.id)
            examples.append(example)
    return examples
