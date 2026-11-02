"""Collect results/* into results/SUMMARY.md, next to the paper's Table 2 numbers."""
import json
import os
import re

R = os.path.expanduser("~/LoRAPrune/results")
RUNS = [("base", "0% (unpruned)"), ("r20", "20%"), ("r30", "30%"), ("r50", "50%")]
ZS = ["boolq", "piqa", "hellaswag", "winogrande", "arc_easy", "arc_challenge", "openbookqa"]
NAMES = {"boolq": "BoolQ", "piqa": "PIQA", "hellaswag": "HellaSwag", "winogrande": "WinoGrande",
         "arc_easy": "ARC-e", "arc_challenge": "ARC-c", "openbookqa": "OBQA"}

# LoRAPrune paper, Table 2 (LaMini). BoolQ is not in Table 2. ARC-c at 0% is taken from
# Table 5 (41.38) because Table 2 repeats the ARC-e value there (67.45, a typo).
PAPER = {
    "base": dict(wikitext2=5.69, ptb=8.93, mmlu=37.10, openbookqa=42.40, arc_easy=67.45,
                 winogrande=67.01, arc_challenge=41.38, piqa=78.35, hellaswag=72.99),
    "r20": dict(wikitext2=7.63, ptb=11.87, mmlu=36.81, openbookqa=38.6, arc_easy=70.20,
                winogrande=66.77, arc_challenge=41.89, piqa=77.48, hellaswag=68.64),
    "r30": dict(wikitext2=8.79, ptb=13.33, mmlu=33.60, openbookqa=36.20, arc_easy=69.61,
                winogrande=62.75, arc_challenge=41.21, piqa=77.48, hellaswag=66.68),
    "r50": dict(wikitext2=11.60, ptb=17.39, mmlu=27.84, openbookqa=35.80, arc_easy=60.38,
                winogrande=56.97, arc_challenge=40.12, piqa=75.39, hellaswag=60.21),
}


def load(tag):
    out = {}
    p = f"{R}/{tag}_ppl.txt"
    if os.path.exists(p):
        txt = open(p).read()
        m = re.search(r"wikitext2 ppl:([\d.]+)", txt)
        out["wikitext2"] = float(m.group(1)) if m else None
        m = re.search(r"PTB ppl:([\d.]+)", txt)
        out["ptb"] = float(m.group(1)) if m else None
    p = f"{R}/{tag}_zeroshot.json"
    if os.path.exists(p):
        d = json.load(open(p))
        out["n_params"] = d.get("n_params")
        for t in ZS:
            r = d["results"].get(t, {})
            out[t] = (r.get("acc,none"), r.get("acc_norm,none"))
    p = f"{R}/{tag}_mmlu.json"
    if os.path.exists(p):
        d = json.load(open(p))
        out["mmlu"] = d["results"].get("mmlu", {}).get("acc,none")
    return out


def f(x, pct=False):
    if x is None:
        return "–"
    return f"{x * 100:.2f}" if pct else f"{x:.2f}"


res = {tag: load(tag) for tag, _ in RUNS}
lines = [f"# LoRAPrune on LLaMA-7B (huggyllama/llama-7b), {os.environ.get('RUN_NAME', '20k LaMini samples')}\n"]

lines.append("## Perplexity (2048-token segments) and 5-shot MMLU\n")
lines.append("| Ratio | Params | WikiText2 ↓ | PTB ↓ | MMLU 5-shot ↑ | paper WikiText2 | paper PTB | paper MMLU |")
lines.append("|---|---|---|---|---|---|---|---|")
for tag, name in RUNS:
    r, p = res[tag], PAPER[tag]
    n = f"{r['n_params'] / 1e9:.2f}B" if r.get("n_params") else "–"
    lines.append(f"| {name} | {n} | {f(r.get('wikitext2'))} | {f(r.get('ptb'))} | {f(r.get('mmlu'), True)} "
                 f"| {p['wikitext2']} | {p['ptb']} | {p['mmlu']} |")

for metric, idx in [("acc", 0), ("acc_norm", 1)]:
    lines.append(f"\n## Zero-shot accuracy (%), metric = `{metric}`\n")
    lines.append("| Ratio | " + " | ".join(NAMES[t] for t in ZS) + " | Avg (7) |")
    lines.append("|---|" + "---|" * (len(ZS) + 1))
    for tag, name in RUNS:
        r = res[tag]
        vals = [r.get(t, (None, None))[idx] for t in ZS]
        # boolq / winogrande only report acc
        vals = [v if v is not None else r.get(t, (None, None))[0] for v, t in zip(vals, ZS)]
        avg = sum(vals) / len(vals) if all(v is not None for v in vals) else None
        lines.append(f"| {name} | " + " | ".join(f(v, True) for v in vals) + f" | {f(avg, True)} |")

lines.append("\n## Paper (Table 2, LoRAPrune on LaMini)\n")
lines.append("| Ratio | " + " | ".join(NAMES[t] for t in ZS if t != "boolq") + " |")
lines.append("|---|" + "---|" * (len(ZS) - 1))
for tag, name in RUNS:
    lines.append(f"| {name} | " + " | ".join(str(PAPER[tag][t]) for t in ZS if t != "boolq") + " |")

lines.append("\nNotes: `acc_norm` table falls back to `acc` for BoolQ and WinoGrande (no acc_norm). "
             "'Params' counts all weights incl. embeddings; layers 0-2 and the last layer are never pruned "
             "(loraprune.utils.freeze), so a 50% ratio removes less than 50% of the total.")
open(f"{R}/SUMMARY.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
