"""Gradnja BH-LLM 2B miks v0.1 (lokalno, streaming; izlazi VAN gita).

Ulazi: lokalni canonical JSONL + HF streaming izvori.
Filteri v0.1 (heuristic; PPL-filter ide na LUMI ingestu, pravilo PPL>100 drop):
  - duzina 50-4000 rijeci (CPT) / 10-2000 (SFT redovi vec kurirani)
  - ijekavica udio >= 0.5 na CPT tekstu (ista probe-lista kao profile_sources)
  - dedup: sha1(normalizirano: lowercase, bez interpunkcije/viska space)
  - held-out: 2% hash-split (sha1 % 50 == 0) -> heldout.jsonl, NIKAD train
Izlaz: /tmp/opencode/bh-mix-v01/{cpt,sft,heldout}.jsonl + MANIFEST.json
(samo MANIFEST + leak izvjestaj se commita; podaci idu na HF tek uz odobrenje).

Koristenje: python scripts/build_mix_v01.py [--klix-cap 50000]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

from profile_sources import IJE_MARKERS, EK_MARKERS  # noqa: E402

HOME = Path.home()
CANON = HOME / "agentmujo-training/datasets/canonical"
OUT = Path("/tmp/opencode/bh-mix-v01")

CPT_LOCAL = [("bc-v02", CANON / "bosnian_core_v0.2.jsonl", 98224),
             ("bc-v03", CANON / "bosnian_core_v0.3.jsonl", 300000)]
SFT_LOCAL = ["bosnian_instructions_v0.1.jsonl", "knowledge_wikipedia_bs_v0.4.jsonl",
             "knowledge_wikipedia_bs_v0.5.jsonl", "function_calling_v0.5.jsonl",
             "agentic_terminal_v0.5.jsonl", "safety_v0.5.jsonl",
             "coding_repo_v0.5.jsonl", "linux_terminal_v0.1.jsonl",
             "thinking_v0.4.jsonl", "confirmation_v0.4.jsonl"]
HF_CPT = [
    ("klix", "Seferovic/bosnian-news-articles-dataset-from-klixba",
     None, "train", lambda r: r.get("text") or ""),
    ("wiki", "ffatty/plaintext-wikipedia-full-bosnian",
     None, "train", lambda r: r.get("text") or ""),
    ("parl", "zastuck/bosnian-parliament-700",
     None, "train", lambda r: " ".join(
         str(v) for k, v in r.items() if k not in ("link", "picture_path"))),
    ("finetome", "TimesLast/FineTome-Alpaca-Bosnian", None, "train",
     lambda r: f"Instrukcija: {r.get('instruction','')}\n{r.get('input','')}\n"
               f"Odgovor: {r.get('output','')}"),
]


def norm(t: str) -> str:
    t = t.lower()
    t = re.sub(r"[^\w\s]", "", t)
    return re.sub(r"\s+", " ", t).strip()


def h(t: str) -> str:
    return hashlib.sha1(norm(t).encode("utf-8")).hexdigest()


def ije_ok(t: str) -> bool:
    i, e = len(IJE_MARKERS.findall(t)), len(EK_MARKERS.findall(t))
    return (i + e) == 0 or i / (i + e) >= 0.5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--klix-cap", type=int, default=50000)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    stats: dict = {"cpt": {}, "sft": 0, "heldout": 0, "drop_dup": 0,
                   "drop_filter": 0}

    def emit(fh_h, fh_t, tag: str, text: str, meta: dict) -> None:
        hh = h(text)
        if hh in seen:
            stats["drop_dup"] += 1
            return
        seen.add(hh)
        rec = {"id": f"mix01-{tag}-{hh[:12]}", "text": text, "meta": meta}
        (fh_h if int(hh[:4], 16) % 50 == 0 else fh_t).write(
            json.dumps(rec, ensure_ascii=False) + "\n")
        if int(hh[:4], 16) % 50 == 0:
            stats["heldout"] += 1

    from datasets import load_dataset  # kasni import

    with open(OUT / "cpt.jsonl", "w", encoding="utf-8") as fc, \
         open(OUT / "heldout.jsonl", "w", encoding="utf-8") as fh:
        # 1) lokalni canonical CPT
        for tag, path, _ in CPT_LOCAL:
            n = 0
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except ValueError:
                        continue
                    msgs = r.get("messages", [])
                    t = "\n\n".join(m.get("content", "") for m in msgs)
                    w = len(t.split())
                    if not (50 <= w <= 4000) or not ije_ok(t):
                        stats["drop_filter"] += 1
                        continue
                    emit(fh, fc, tag, t, {"src": tag, "license":
                         r.get("metadata", {}).get("license", "?")})
                    n += 1
            stats["cpt"][tag] = n
            print(f"{tag}: {n}", flush=True)
        # 2) HF izvori
        for tag, repo, cfg, split, pick in HF_CPT:
            cap = args.klix_cap if tag == "klix" else 10 ** 9
            n = 0
            ds = load_dataset(repo, cfg, split=split, streaming=True)
            for i, row in enumerate(ds):
                if n >= cap:
                    break
                t = (pick(row) or "").strip()
                w = len(t.split())
                if not (50 <= w <= 4000) or not ije_ok(t):
                    stats["drop_filter"] += 1
                    continue
                emit(fh, fc, tag, t, {"src": repo})
                n += 1
            stats["cpt"][tag] = n
            print(f"{tag}: {n}", flush=True)

    # 3) SFT lokalni (vec kurirani; samo dedup + held-out split)
    with open(OUT / "sft.jsonl", "w", encoding="utf-8") as fs, \
         open(OUT / "heldout.jsonl", "a", encoding="utf-8") as fh:
        for fn in SFT_LOCAL:
            p = CANON / fn
            if not p.exists():
                print(f"nema {fn}, preskacem", flush=True)
                continue
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        emit(fh, fs, fn.replace(".jsonl", ""), line, {})
                        stats["sft"] += 1

    (OUT / "MANIFEST.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"gotovo: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
