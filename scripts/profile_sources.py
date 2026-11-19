"""Profiliranje kandidatskih datasetova za BH-LLM-Basic 2B (streaming, CAP redova).

Koristenje: python scripts/profile_sources.py [--cap 2000] [--out docs/SOURCE_PROFILES_2B.json]
Heuristike su grube (shortcut: ijekavica-probe rjecnik, ne full classifier;
upgrade na fasttext lang+script ID kad miks udje u pipeline).
"""
from __future__ import annotations

import argparse
import json
import re
import sys

IJE_MARKERS = re.compile(
    r"(svjetlo|svijet|lijep|djevoj|vrijeme|mjesto|dijete|čovjek|žena[^e]|"
    r"pjesma|pjeva|sjećanje|ocjena|primijen|vijest|riječ|rješenje|"
    r"djelov|mjesec|snijeg|blijed|smijeh|dijel|cijena|liječ)",
    re.IGNORECASE,
)
EK_MARKERS = re.compile(
    r"(svetlo|svelt|lepo|devoj|vreme|mesto|dete|čovek|pesma|peva|"
    r"sećanje|primen|vest|reč|rešenje|delov|mesec|sneg|bled|smeh|"
    r"deo|cena|leč)",
    re.IGNORECASE,
)
CYRILLIC = re.compile(r"[\u0400-\u04FF]")

SOURCES = [
    ("klix-news", "Seferovic/bosnian-news-articles-dataset-from-klixba",
     None, "train", lambda r: r.get("text") or ""),
    ("wikipedia-bs", "ffatty/plaintext-wikipedia-full-bosnian",
     None, "train", lambda r: r.get("text") or ""),
    ("finetome-bs", "TimesLast/FineTome-Alpaca-Bosnian",
     None, "train",
     lambda r: f"{r.get('instruction','')}\n{r.get('input','')}\n{r.get('output','')}"),
    ("parliament-700", "zastuck/bosnian-parliament-700",
     None, "train", lambda r: " ".join(str(v) for v in r.values())),
]


def profile_texts(texts: list[str]) -> dict:
    n = len(texts)
    chars = sum(len(t) for t in texts)
    words = sum(len(t.split()) for t in texts)
    cyr = sum(1 for t in texts if CYRILLIC.search(t))
    ije = sum(len(IJE_MARKERS.findall(t)) for t in texts)
    ek = sum(len(EK_MARKERS.findall(t)) for t in texts)
    tot = ije + ek
    return {
        "n": n,
        "avg_chars": round(chars / max(n, 1), 1),
        "avg_words": round(words / max(n, 1), 1),
        "cyrillic_share": round(cyr / max(n, 1), 4),
        "ije_hits": ije,
        "ek_hits": ek,
        "ije_share": round(ije / tot, 4) if tot else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=2000)
    ap.add_argument("--out", default="docs/SOURCE_PROFILES_2B.json")
    args = ap.parse_args()

    from datasets import load_dataset  # kasni import (tezak)

    report: dict = {}
    for name, repo, cfg, split, pick in SOURCES:
        try:
            ds = load_dataset(repo, cfg, split=split, streaming=True)
            texts = []
            for i, row in enumerate(ds):
                if i >= args.cap:
                    break
                t = pick(row)
                if t and t.strip():
                    texts.append(t)
            report[name] = {"repo": repo, "cap": args.cap,
                            **profile_texts(texts)}
            print(f"{name}: n={len(texts)} {report[name]}", flush=True)
        except Exception as e:  # jedan izvor ne smije srusiti ostale
            report[name] = {"repo": repo, "error": str(e)[:300]}
            print(f"{name}: ERROR {e}", flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"upisano: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
