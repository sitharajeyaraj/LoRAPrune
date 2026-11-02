# LoRAPrune on LLaMA-7B (huggyllama/llama-7b), 40k LaMini samples, cosine LR

## Perplexity (2048-token segments) and 5-shot MMLU

| Ratio | Params | WikiText2 ↓ | PTB ↓ | MMLU 5-shot ↑ | paper WikiText2 | paper PTB | paper MMLU |
|---|---|---|---|---|---|---|---|
| 0% (unpruned) | 6.76B | 5.68 | 29.63 | 35.19 | 5.69 | 8.93 | 37.1 |
| 20% | 5.65B | 7.99 | 48.94 | 30.10 | 7.63 | 11.87 | 36.81 |
| 30% | 5.09B | 9.71 | 52.61 | 28.56 | 8.79 | 13.33 | 33.6 |
| 50% | 3.98B | 18.69 | 90.37 | 26.15 | 11.6 | 17.39 | 27.84 |

## Zero-shot accuracy (%), metric = `acc`

| Ratio | BoolQ | PIQA | HellaSwag | WinoGrande | ARC-e | ARC-c | OBQA | Avg (7) |
|---|---|---|---|---|---|---|---|---|
| 0% (unpruned) | 75.11 | 78.67 | 56.95 | 69.85 | 75.25 | 41.89 | 34.40 | 61.73 |
| 20% | 72.17 | 75.08 | 51.06 | 65.19 | 67.51 | 36.86 | 27.20 | 56.44 |
| 30% | 65.60 | 72.63 | 46.95 | 61.96 | 61.78 | 32.68 | 23.40 | 52.14 |
| 50% | 55.63 | 67.41 | 38.99 | 55.96 | 53.16 | 27.90 | 21.00 | 45.72 |

## Zero-shot accuracy (%), metric = `acc_norm`

| Ratio | BoolQ | PIQA | HellaSwag | WinoGrande | ARC-e | ARC-c | OBQA | Avg (7) |
|---|---|---|---|---|---|---|---|---|
| 0% (unpruned) | 75.11 | 79.16 | 76.20 | 69.85 | 72.73 | 44.62 | 44.60 | 66.04 |
| 20% | 72.17 | 76.28 | 68.99 | 65.19 | 66.92 | 38.74 | 36.80 | 60.73 |
| 30% | 65.60 | 73.61 | 62.95 | 61.96 | 58.63 | 34.73 | 35.20 | 56.10 |
| 50% | 55.63 | 66.87 | 50.31 | 55.96 | 49.54 | 30.97 | 32.00 | 48.75 |

## Paper (Table 2, LoRAPrune on LaMini)

| Ratio | PIQA | HellaSwag | WinoGrande | ARC-e | ARC-c | OBQA |
|---|---|---|---|---|---|---|
| 0% (unpruned) | 78.35 | 72.99 | 67.01 | 67.45 | 41.38 | 42.4 |
| 20% | 77.48 | 68.64 | 66.77 | 70.2 | 41.89 | 38.6 |
| 30% | 77.48 | 66.68 | 62.75 | 69.61 | 41.21 | 36.2 |
| 50% | 75.39 | 60.21 | 56.97 | 60.38 | 40.12 | 35.8 |

Notes: `acc_norm` table falls back to `acc` for BoolQ and WinoGrande (no acc_norm). 'Params' counts all weights incl. embeddings; layers 0-2 and the last layer are never pruned (loraprune.utils.freeze), so a 50% ratio removes less than 50% of the total.
