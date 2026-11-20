"""Kratki SFT ulaz (~21k redova, ~2 h na MI250X).

Sastav iz sft.jsonl (redoslijed = SFT_LOCAL + bc-v02 na kraju):
  0-17499      bosnian_instructions (1500) + knowledge v0.4/v0.5 (16000)
  17500-19499  function_calling v0.5 (2000)
  20700-22199  safety v0.5 (1500)
CPT se preskace (odluka 2026-11-20: predugo); gate je Belebele-hrv + BiH set.
"""
from __future__ import annotations

import json
import sys

sys.path.insert(0, "scripts")
from render_salamandra import sample_to_text  # noqa: E402

RANGES = [(0, 17499), (17500, 19499), (20700, 22199)]
SRC = "/tmp/opencode/bh-mix-v01/sft.jsonl"
OUT = "/tmp/opencode/bh-llm-job/data/sft_short21k.jsonl"


def main() -> None:
    rows = [json.loads(l) for l in open(SRC, encoding="utf-8") if l.strip()]
    print(f"sft pool: {len(rows)}")
    keep = []
    for a, b in RANGES:
        keep.extend(rows[a:b + 1])
    print(f"short: {len(keep)}")
    with open(OUT, "w", encoding="utf-8") as o:
        for line in keep:
            canon = json.loads(line["text"])
            o.write(json.dumps({"text": sample_to_text(canon)},
                               ensure_ascii=False) + "\n")
    print(f"upisano: {OUT}")


if __name__ == "__main__":
    main()
