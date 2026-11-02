# LoRAPrune on LLaMA-7B (huggyllama/llama-7b), 20k LaMini samples, cosine LR, bf16

## Perplexity (2048-token segments) and 5-shot MMLU

| Ratio | Params | WikiText2 ↓ | PTB ↓ | MMLU 5-shot ↑ | paper WikiText2 | paper PTB | paper MMLU |
|---|---|---|---|---|---|---|---|
| 0% (unpruned) | 6.76B | 5.68 | 29.63 | 35.19 | 5.69 | 8.93 | 37.1 |
| 20% | 5.65B | 8.50 | 52.40 | 30.77 | 7.63 | 11.87 | 36.81 |
| 30% | 5.09B | 12.18 | 62.60 | 25.12 | 8.79 | 13.33 | 33.6 |
| 50% | 3.98B | 21.38 | 123.52 | 25.09 | 11.6 | 17.39 | 27.84 |

## Zero-shot accuracy (%), metric = `acc`

| Ratio | BoolQ | PIQA | HellaSwag | WinoGrande | ARC-e | ARC-c | OBQA | Avg (7) |
|---|---|---|---|---|---|---|---|---|
| 0% (unpruned) | 75.11 | 78.67 | 56.95 | 69.85 | 75.25 | 41.89 | 34.40 | 61.73 |
| 20% | 71.80 | 75.73 | 51.54 | 64.33 | 68.18 | 38.05 | 26.60 | 56.61 |
| 30% | 59.72 | 74.16 | 47.42 | 60.54 | 62.42 | 32.51 | 24.20 | 51.57 |
| 50% | 55.02 | 66.16 | 38.37 | 54.62 | 52.02 | 27.65 | 19.40 | 44.75 |

## Zero-shot accuracy (%), metric = `acc_norm`

| Ratio | BoolQ | PIQA | HellaSwag | WinoGrande | ARC-e | ARC-c | OBQA | Avg (7) |
|---|---|---|---|---|---|---|---|---|
| 0% (unpruned) | 75.11 | 79.16 | 76.20 | 69.85 | 72.73 | 44.62 | 44.60 | 66.04 |
| 20% | 71.80 | 75.24 | 69.36 | 64.33 | 65.32 | 39.59 | 39.40 | 60.72 |
| 30% | 59.72 | 73.34 | 64.05 | 60.54 | 59.34 | 35.58 | 35.60 | 55.45 |
| 50% | 55.02 | 66.38 | 49.00 | 54.62 | 49.79 | 30.63 | 30.60 | 48.00 |

## Paper (Table 2, LoRAPrune on LaMini)

| Ratio | PIQA | HellaSwag | WinoGrande | ARC-e | ARC-c | OBQA |
|---|---|---|---|---|---|---|
| 0% (unpruned) | 78.35 | 72.99 | 67.01 | 67.45 | 41.38 | 42.4 |
| 20% | 77.48 | 68.64 | 66.77 | 70.2 | 41.89 | 38.6 |
| 30% | 77.48 | 66.68 | 62.75 | 69.61 | 41.21 | 36.2 |
| 50% | 75.39 | 60.21 | 56.97 | 60.38 | 40.12 | 35.8 |

Notes: `acc_norm` table falls back to `acc` for BoolQ and WinoGrande (no acc_norm). 'Params' counts all weights incl. embeddings; layers 0-2 and the last layer are never pruned (loraprune.utils.freeze), so a 50% ratio removes less than 50% of the total.
