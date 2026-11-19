"""Template adapter: kanonski red -> Salamandra-2B SFT tekst.

Salamandra-2B je BASE model, tokenizer NEMA chat_template (provjereno
2026-11-19: chat_template=None). Zato NE koristimo ChatML (<|im_start|>)
niti Llama-3 headere — nisu u 256k SentencePiece vokabularu i trazili bi
operaciju nad tokenizatorom. Umjesto toga: obicne bosanske oznake koje se
uredno tokeniziraju + EOS `</s>` na kraju asistenta.

Mapiranje (sadrzaj 1:1 iz kanonskog reda, samo ovojnica se mijenja):
  system   -> ### Sistem:
  user     -> ### Korisnik:
  assistant-> ### Asistent: (+ <tool_call> XML inline, kao i dosad)
  tool     -> ### Alat: (sadrzaj = tool_response tekst)

CPT rezim: sample_to_cpt() vraca cisti tekst prompt+completion za
prefix-completion redove (bosnian_core v0.3 stil).

Kompatibilnost: ulaz je isti kanonski JSONL kao za
agentmujo-training/training/format.py (messages[] s role/content,
opcionalno tool_calls[]). Texture provjeriti:
  python scripts/render_salamandra.py --check
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

EOS = "</s>"

HDR = {
    "system": "### Sistem:",
    "user": "### Korisnik:",
    "assistant": "### Asistent:",
    "tool": "### Alat:",
}


def render_tool_calls(tool_calls: list[dict[str, Any]]) -> str:
    # shortcut: ista XML semantika kao agentmujo format.py; drzati uskladjeno
    # (upgrade: uvesti zajednicki paket kad transfer zavrsi).
    parts = []
    for tc in tool_calls or []:
        lines = [f"<tool_call>\n<function={tc['name']}>"]
        for k, v in (tc.get("arguments") or {}).items():
            lines.append(f"<parameter={k}>\n{v}\n</parameter>")
        lines.append("</function>\n</tool_call>")
        parts.append("\n".join(lines))
    return "\n".join(parts)


def sample_to_text(sample: dict[str, Any]) -> str:
    chunks = []
    for m in sample.get("messages", []):
        role, content = m.get("role", ""), m.get("content", "")
        if role == "assistant":
            if m.get("tool_calls"):
                tc = render_tool_calls(m["tool_calls"])
                content = f"{content}\n\n{tc}" if content.strip() else tc
            chunks.append(f"{HDR['assistant']}\n{content}\n{EOS}")
        elif role == "tool":
            chunks.append(f"{HDR['tool']}\n{content}")
        elif role in HDR:
            chunks.append(f"{HDR[role]}\n{content}")
    return "\n\n".join(chunks)


def sample_to_cpt(sample: dict[str, Any]) -> str:
    """CPT: user+assistant spojeno kao tekuci tekst (bez oznaka gubitka)."""
    texts = [m.get("content", "") for m in sample.get("messages", [])
             if m.get("content", "").strip()]
    return "\n\n".join(texts)


def check(n: int = 3) -> int:
    """Brzi self-check: render nekoliko kanonskih redova + EOS invarijanta."""
    import pathlib
    src = pathlib.Path.home() / "agentmujo-training/datasets/canonical/bosnian_instructions_v0.1.jsonl"
    if not src.exists():
        print(f"preskoceno: nema {src}")
        return 0
    ok = 0
    with open(src, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            out = sample_to_text(json.loads(line))
            assert "### Korisnik:" in out and "### Asistent:" in out, out[:200]
            assert out.rstrip().endswith(EOS), out[-50:]
            ok += 1
    print(f"OK: {ok} uzorka, EOS+zaglavlja prisutni; primjer (600c):")
    print(out[:600])
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--in", dest="inp", default=None)
    ap.add_argument("--mode", choices=["sft", "cpt"], default="sft")
    a = ap.parse_args()
    if a.check or not a.inp:
        return check()
    fn = sample_to_text if a.mode == "sft" else sample_to_cpt
    with open(a.inp, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                sys.stdout.write(fn(json.loads(line)) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
