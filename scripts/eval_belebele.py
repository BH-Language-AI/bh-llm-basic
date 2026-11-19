"""Belebele-bs zero-shot baseline (CPU, scoring po PPL opcija).

Metodologija (gruba, dokumentirana): za svako pitanje racuna PPL cijele
sekvence passage + pitanje + opcija; bira najmanji. Nije isto sto i
vjerojatnost opcije uz maskiranje, ali je deterministicki orijentir
istom metodom prije/poslije treninga.
Koristenje: python scripts/eval_belebele.py [--cap 120] [--out docs/BELEBELE_BASELINE_2B.json]
"""
from __future__ import annotations

import argparse
import json
import math


def ppl_of(model, tok, text: str) -> float:
    import torch
    ids = tok(text, return_tensors="pt").input_ids
    with torch.no_grad():
        loss = model(ids, labels=ids).loss.item()
    return math.exp(loss)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=120)
    ap.add_argument("--out", default="docs/BELEBELE_BASELINE_2B.json")
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    m = "BSC-LT/salamandra-2b"
    tok = AutoTokenizer.from_pretrained(m, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        m, dtype=torch.float32, device_map="cpu", trust_remote_code=False)
    model.eval()

    ds = load_dataset("facebook/belebele", "hrv_Latn", split="test",
                      streaming=True)
    total = correct = 0
    details = []
    for i, row in enumerate(ds):
        if i >= args.cap:
            break
        ctx = f"{row['flores_passage']}\nPitanje: {row['question']}\nOdgovor:"
        opts = [row[f"mc_answer{k}"] for k in (1, 2, 3, 4)]
        try:
            ppls = [ppl_of(model, tok, f"{ctx} {o}") for o in opts]
        except Exception as e:  # jedan red ne smije srusiti baseline
            print(f"red {i}: SKIP {e}", flush=True)
            continue
        pred = int(min(range(4), key=lambda k: ppls[k])) + 1
        gold = int(row["correct_answer_num"])
        total += 1
        ok = pred == gold
        correct += ok
        details.append({"i": i, "pred": pred, "gold": gold, "ok": ok,
                        "ppls": [round(p, 1) for p in ppls]})
        if total % 20 == 0:
            print(f"{total}: acc={correct/total:.3f}", flush=True)
    rep = {"model": m, "method": "full-seq-PPL-argmin",
           "n": total, "correct": correct,
           "acc": round(correct / max(total, 1), 4)}
    json.dump({**rep, "details": details}, open(args.out, "w"),
              ensure_ascii=False, indent=1)
    print(f"BELEBELE-bs: {correct}/{total} = {rep['acc']:.3f} (slucajno 0.25)")
    return 0


if __name__ == "__main__":
    main()
