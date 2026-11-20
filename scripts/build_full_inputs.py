"""Gradnja full-run ulaza iz miksa v0.1.

1) sft_full_rendered.jsonl: 122k kanonskih redova -> sample_to_text
   (train_sft.py ocekuje gotov tekst, ne kanonski JSON).
2) cpt_focus.jsonl: wiki + klix + finetome + parl cijeli + random 50k bc-v03
   (PPL-top selekcija ide na LUMI s GPU-om; ovdje stratificirani random, seed 42).
Izlaz: /tmp/opencode/bh-llm-job/data/ (isti dir koji ide na LUMI).
"""
from __future__ import annotations

import json
import random
import sys

sys.path.insert(0, "scripts")
from render_salamandra import sample_to_text  # noqa: E402

MIX = "/tmp/opencode/bh-mix-v01"
OUT = "/tmp/opencode/bh-llm-job/data"


def main() -> None:
    # 1) SFT render
    n = 0
    with open(f"{MIX}/sft.jsonl", encoding="utf-8") as f, \
            open(f"{OUT}/sft_full_rendered.jsonl", "w",
                 encoding="utf-8") as o:
        for line in f:
            line = line.strip()
            if not line:
                continue
            canon = json.loads(json.loads(line)["text"])
            o.write(json.dumps({"text": sample_to_text(canon)},
                               ensure_ascii=False) + "\n")
            n += 1
    print(f"sft_full_rendered: {n}")

    # 2) CPT fokus (meta.src nosi puna HF imena — mapirano ovdje)
    SRC2TAG = {
        "ffatty/plaintext-wikipedia-full-bosnian": "wiki",
        "Seferovic/bosnian-news-articles-dataset-from-klixba": "klix",
        "TimesLast/FineTome-Alpaca-Bosnian": "finetome",
        "zastuck/bosnian-parliament-700": "parl",
        "bc-v02": "bc-v02",
    }
    random.seed(42)
    bc3, rest = [], []
    kept = {"wiki": 0, "klix": 0, "finetome": 0, "parl": 0, "bc-v02": 0}
    with open(f"{MIX}/cpt.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            src = r.get("meta", {}).get("src", "")
            if src == "bc-v03":
                bc3.append(line)
            elif src in SRC2TAG:
                rest.append(line)
                kept[SRC2TAG[src]] += 1
    random.shuffle(bc3)
    focus = rest + bc3[:50000]
    random.shuffle(focus)
    with open(f"{OUT}/cpt_focus.jsonl", "w", encoding="utf-8") as o:
        o.writelines(focus)
    kept["bc-v03-sample50k"] = min(50000, len(bc3))
    kept["UKUPNO"] = len(focus)
    print(f"cpt_focus: {kept} (bc-v03 pool: {len(bc3)})")


if __name__ == "__main__":
    main()
