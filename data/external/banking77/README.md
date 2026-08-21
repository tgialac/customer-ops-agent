# BANKING77 benchmark subset

This directory contains the filtered BANKING77 subset used as an external
intent-classification benchmark. It is **not** the MoMo golden set: examples
are English, single-turn utterances, and do not include MoMo policy, state, or
tool outcomes.

## Contents

- `train.jsonl`: 447 records
- `test.jsonl`: 120 records
- `metadata.json`: source hashes, label mapping, and counts

## Mapping

| BANKING77 label | Project intent |
| --- | --- |
| `Refund_not_showing_up` | `missing_refund` |
| `pending_transfer` | `transaction_pending` |
| `failed_transfer` | `transaction_failed` |

## Attribution and license

This is a filtered and normalized subset of [PolyAI's BANKING77 dataset](https://github.com/PolyAI-LDN/task-specific-datasets),
licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The original dataset is described in [Efficient Intent Detection with Dual
Sentence Encoders](https://arxiv.org/abs/2003.04807). The label mapping and
JSONL normalization are modifications made by this project.
