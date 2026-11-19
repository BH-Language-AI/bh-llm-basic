"""Dokument-level PPL baseline Salamandra-2B (CPU, sliding window).

Koristenje: python scripts/ppl_baseline.py [--docs 5] [--out docs/PPL_BASELINE_2B.json]
Single-sentence PPL je sumljiv (prvi tokeni dominiraju); ovo vrti prozor
od 512 tokena preko stvarnih dokumenata iz kandidatskih izvora.
Sporo na CPU (~1 min/prozor) — zato mali N; pravi baseline ide na LUMI.
"""
from __future__ import annotations

import argparse
import json
import math
import sys

TEXTS = [
    ("klix", "Seferovic/bosnian-news-articles-dataset-from-klixba",
     None, "train", lambda r: r.get("text") or ""),
    ("wiki", "ffatty/plaintext-wikipedia-full-bosnian",
     None, "train", lambda r: r.get("text") or ""),
]


def doc_ppl(model, tok, text: str, window: int = 512, stride: int = 256) -> float:
    import torch
    ids = tok(text, return_tensors="pt").input_ids[0]
    if len(ids) < 32:
        return float("nan")
    ids = ids[: window * 4]  # cap: prvih ~2k tokena po doku (CPU trosak)
    nll, ntok = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(ids) - 16, stride):
            seg = ids[i: i + window].unsqueeze(0)
            out = model(seg, labels=seg)
            n = seg.numel()
            nll += out.loss.item() * n
            ntok += n
            if i + stride >= len(ids) - 16:
                break
    return math.exp(nll / ntok)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=5)
    ap.add_argument("--out", default="docs/PPL_BASELINE_2B.json")
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    m = "BSC-LT/salamandra-2b"
    tok = AutoTokenizer.from_pretrained(m, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        m, dtype=torch.float32, device_map="cpu", trust_remote_code=False)
    model.eval()

    rep: dict = {"model": m, "window": 512, "stride": 256, "sources": {}}
    for name, repo, cfg, split, pick in TEXTS:
        ds = load_dataset(repo, cfg, split=split, streaming=True)
        ppls = []
        for i, row in enumerate(ds):
            if len(ppls) >= args.docs:
                break
            t = pick(row)
            if t and len(t.split()) > 100:
                p = doc_ppl(model, tok, t)
                ppls.append(round(p, 1))
                print(f"{name} doc{len(ppls)}: PPL={p:.1f}", flush=True)
        rep["sources"][name] = ppls
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print(f"upisano: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
