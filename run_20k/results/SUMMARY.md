# LoRAPrune on LLaMA-7B (huggyllama/llama-7b), 20k LaMini samples

## Perplexity (2048-token segments) and 5-shot MMLU

| Ratio | Params | WikiText2 ↓ | PTB ↓ | MMLU 5-shot ↑ | paper WikiText2 | paper PTB | paper MMLU |
|---|---|---|---|---|---|---|---|
| 0% (unpruned) | 6.76B | 5.68 | 29.63 | 35.19 | 5.69 | 8.93 | 37.1 |
| 20% | 5.65B | 8.21 | 45.26 | 30.40 | 7.63 | 11.87 | 36.81 |
| 30% | 5.09B | 11.18 | 67.95 | 28.54 | 8.79 | 13.33 | 33.6 |
| 50% | 3.98B | 20.36 | 111.59 | 26.53 | 11.6 | 17.39 | 27.84 |

## Zero-shot accuracy (%), metric = `acc`

| Ratio | BoolQ | PIQA | HellaSwag | WinoGrande | ARC-e | ARC-c | OBQA | Avg (7) |
|---|---|---|---|---|---|---|---|---|
| 0% (unpruned) | 75.11 | 78.67 | 56.95 | 69.85 | 75.25 | 41.89 | 34.40 | 61.73 |
| 20% | 69.42 | 75.41 | 51.34 | 65.90 | 67.97 | 37.63 | 26.80 | 56.35 |
| 30% | 66.21 | 72.96 | 47.19 | 59.51 | 62.42 | 31.83 | 24.00 | 52.02 |
| 50% | 51.59 | 65.51 | 37.97 | 54.30 | 51.85 | 27.99 | 19.60 | 44.12 |

## Zero-shot accuracy (%), metric = `acc_norm`

| Ratio | BoolQ | PIQA | HellaSwag | WinoGrande | ARC-e | ARC-c | OBQA | Avg (7) |
|---|---|---|---|---|---|---|---|---|
| 0% (unpruned) | 75.11 | 79.16 | 76.20 | 69.85 | 72.73 | 44.62 | 44.60 | 66.04 |
| 20% | 69.42 | 75.73 | 69.07 | 65.90 | 65.19 | 39.42 | 39.40 | 60.59 |
| 30% | 66.21 | 73.94 | 63.09 | 59.51 | 59.64 | 34.47 | 36.60 | 56.21 |
| 50% | 51.59 | 65.89 | 48.60 | 54.30 | 47.64 | 29.86 | 30.20 | 46.87 |

## Paper (Table 2, LoRAPrune on LaMini)

| Ratio | PIQA | HellaSwag | WinoGrande | ARC-e | ARC-c | OBQA |
|---|---|---|---|---|---|---|
| 0% (unpruned) | 78.35 | 72.99 | 67.01 | 67.45 | 41.38 | 42.4 |
| 20% | 77.48 | 68.64 | 66.77 | 70.2 | 41.89 | 38.6 |
| 30% | 77.48 | 66.68 | 62.75 | 69.61 | 41.21 | 36.2 |
| 50% | 75.39 | 60.21 | 56.97 | 60.38 | 40.12 | 35.8 |

Notes: `acc_norm` table falls back to `acc` for BoolQ and WinoGrande (no acc_norm). 'Params' counts all weights incl. embeddings; layers 0-2 and the last layer are never pruned (loraprune.utils.freeze), so a 50% ratio removes less than 50% of the total.
