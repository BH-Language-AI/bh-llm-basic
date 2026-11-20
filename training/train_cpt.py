#!/usr/bin/env python3
"""Salamandra-2B CPT: LoRA bf16 nad cistim tekstom, plain Trainer.

Ulaz: JSONL s {"text": ...} (cpt.jsonl iz build_mix_v01).
Chunkiranje na 2048 tokena s preklapanjem 0 (smoke), packing je v0.2.
"""
from __future__ import annotations

import argparse
import json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="BSC-LT/salamandra-2b")
    ap.add_argument("--rev", default=None)
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--cpu", action="store_true",
                    help="lokalna CPU validacija (use_cpu, bez bf16)")
    a = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import Dataset as TorchDataset
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              Trainer, TrainingArguments)

    texts = [json.loads(l)["text"] for l in open(a.train, encoding="utf-8")
             if l.strip()]
    print(f"dokumenata: {len(texts)}", flush=True)
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.rev,
                                        trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, revision=a.rev, dtype=torch.bfloat16,
        trust_remote_code=False)
    # shortcut: chunk po ~2048 tokena bez packinga (v0.2: packing + documentske granice).
    ids_all: list[list[int]] = []
    for t in texts:
        ids = tok(t, truncation=False)["input_ids"] + [tok.eos_token_id]
        for i in range(0, len(ids), 2048):
            c = ids[i:i + 2048]
            if len(c) > 128:
                ids_all.append(c)
    print(f"chunkova: {len(ids_all)}", flush=True)
    # shortcut: obican torch Dataset (datasets 5.2 + pyarrow 26 puca na dillu).
    class ListDS(TorchDataset):
        def __init__(self, ids):
            self.ids = ids
        def __len__(self):
            return len(self.ids)
        def __getitem__(self, i):
            return {"input_ids": self.ids[i]}
    ds = ListDS(ids_all)

    def collate(b):
        import torch as T
        ids = [x["input_ids"] for x in b]
        m = max(map(len, ids))
        pad = tok.eos_token_id
        ib = T.tensor([x + [pad] * (m - len(x)) for x in ids])
        return {"input_ids": ib, "attention_mask": ib.ne(pad),
                "labels": ib}

    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                      target_modules=["q_proj", "k_proj", "v_proj",
                                      "o_proj", "gate_proj", "up_proj",
                                      "down_proj"],
                      task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    # transformers>=5: nema warmup_ratio, samo warmup_steps.
    args = TrainingArguments(
        output_dir=a.out, per_device_train_batch_size=1,
        gradient_accumulation_steps=8, learning_rate=2e-5,
        max_steps=a.max_steps, warmup_steps=max(1, a.max_steps // 20),
        logging_steps=10,
        save_steps=100, save_total_limit=1, bf16=not a.cpu, use_cpu=a.cpu,
        gradient_checkpointing=True, seed=42, report_to="none")
    Trainer(model=model, args=args, train_dataset=ds,
            data_collator=collate).train()
    model.save_pretrained(a.out + "-adapter")
    print("CPT_TRAIN_OK", flush=True)


if __name__ == "__main__":
    main()
