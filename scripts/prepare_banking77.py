#!/usr/bin/env python3
"""Download and normalize the relevant BANKING77 intents.

This produces an external benchmark, not the MoMo golden set. The source
examples remain English single-turn utterances; the normalized records make
the source intent and the project intent mapping explicit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Iterable
from urllib.request import Request, urlopen

from momo_ops_agent.contracts import IntentName


SOURCE_REPOSITORY = "https://github.com/PolyAI-LDN/task-specific-datasets"
SOURCE_RAW_BASE = (
    "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data"
)
SOURCE_LICENSE = "CC BY 4.0"

SOURCE_TO_PROJECT_INTENT = {
    "Refund_not_showing_up": IntentName.MISSING_REFUND.value,
    "pending_transfer": IntentName.TRANSACTION_PENDING.value,
    "failed_transfer": IntentName.TRANSACTION_FAILED.value,
}


def download_source(filename: str, destination: Path) -> str:
    """Download one source file and return its SHA-256 digest."""

    request = Request(
        f"{SOURCE_RAW_BASE}/{filename}",
        headers={"User-Agent": "momo-ops-agent-dataset-preparer/0.1"},
    )
    digest = hashlib.sha256()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(request, timeout=30) as response, destination.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            digest.update(chunk)
            output.write(chunk)
    return digest.hexdigest()


def normalize_rows(rows: Iterable[dict[str, str]], split: str) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        source_intent = row["category"]
        project_intent = SOURCE_TO_PROJECT_INTENT.get(source_intent)
        if project_intent is None:
            continue
        text = row["text"].strip()
        if not text:
            raise ValueError(f"empty utterance at {split}:{index}")
        normalized_index = len(normalized)
        normalized.append(
            {
                "id": f"banking77-{split}-{normalized_index:05d}",
                "text": text,
                "language": "en",
                "source": "PolyAI/banking77",
                "source_intent": source_intent,
                "project_intent": project_intent,
                "single_turn": True,
            }
        )
    return normalized


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def prepare(output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="momo-ops-agent-banking77-") as temporary:
        cache_dir = Path(temporary)
        source_digests = {
            filename: download_source(filename, cache_dir / filename)
            for filename in ("train.csv", "test.csv", "categories.json")
        }
        train = normalize_rows(read_csv(cache_dir / "train.csv"), "train")
        test = normalize_rows(read_csv(cache_dir / "test.csv"), "test")

    train_count = write_jsonl(output_dir / "train.jsonl", train)
    test_count = write_jsonl(output_dir / "test.jsonl", test)
    counts = {
        "train": train_count,
        "test": test_count,
        "total": train_count + test_count,
    }
    metadata = {
        "dataset": "BANKING77",
        "source_repository": SOURCE_REPOSITORY,
        "source_license": SOURCE_LICENSE,
        "source_files_sha256": source_digests,
        "project_intent_mapping": SOURCE_TO_PROJECT_INTENT,
        "counts": counts,
        "notes": [
            "External benchmark only; not the MoMo golden set.",
            "Source examples are English single-turn utterances.",
            "Labels are remapped to the initial MoMo Ops Agent intent contract.",
        ],
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "README.md").write_text(
        f"""# BANKING77 benchmark subset

This directory contains the filtered BANKING77 subset used as an external
intent-classification benchmark. It is **not** the MoMo golden set: examples
are English, single-turn utterances, and do not include MoMo policy, state, or
tool outcomes.

## Contents

- `train.jsonl`: {train_count} records
- `test.jsonl`: {test_count} records
- `metadata.json`: source hashes, label mapping, and counts

## Mapping

| BANKING77 label | Project intent |
| --- | --- |
| `Refund_not_showing_up` | `missing_refund` |
| `pending_transfer` | `transaction_pending` |
| `failed_transfer` | `transaction_failed` |

## Attribution and license

This is a filtered and normalized subset of [PolyAI's BANKING77 dataset]({SOURCE_REPOSITORY}),
licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The original dataset is described in [Efficient Intent Detection with Dual
Sentence Encoders](https://arxiv.org/abs/2003.04807). The label mapping and
JSONL normalization are modifications made by this project.
""",
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/external/banking77"),
        help="Directory for the normalized benchmark output.",
    )
    args = parser.parse_args()
    metadata = prepare(args.output_dir)
    print(json.dumps(metadata["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
