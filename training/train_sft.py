#!/usr/bin/env python3
"""Salamandra-2B SFT smoke: LoRA bf16, plain Trainer, bez trl/bitsandbytes.

Ulaz: JSONL s {"text": <render_salamandra.sample_to_text output>}.
Tokenizer nema chat_template (base model) pa se tekst tokenizira direktno;
labels = input_ids (maskiranje user-dijela je v0.2 nadogradnja, smoke uci sve).
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
    ap.add_argument("--max-steps", type=int, default=100)
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
    print(f"uzoraka: {len(texts)}", flush=True)
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.rev,
                                        trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, revision=a.rev, dtype=torch.bfloat16,
        trust_remote_code=False)
    enc = tok(texts, truncation=True, max_length=2048)
    # shortcut: obican torch Dataset umjesto datasets.from_dict
    # (datasets 5.2 + pyarrow 26 puca na dill fingerprintu: MonthDayNano).
    class ListDS(TorchDataset):
        def __init__(self, ids, mask):
            self.ids, self.mask = ids, mask
        def __len__(self):
            return len(self.ids)
        def __getitem__(self, i):
            return {"input_ids": self.ids[i],
                    "attention_mask": self.mask[i]}
    ds = ListDS(enc["input_ids"], enc["attention_mask"])

    def collate(b):
        import torch as T
        ids = [x["input_ids"] for x in b]
        m = max(map(len, ids))
        pad = tok.eos_token_id
        ib = T.tensor([x + [pad] * (m - len(x)) for x in ids])
        ab = T.tensor([[1] * len(x) + [0] * (m - len(x)) for x in b
                       for x in [x["attention_mask"]]])
        return {"input_ids": ib, "attention_mask": ab, "labels": ib}

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
        save_steps=50, save_total_limit=1, bf16=not a.cpu, use_cpu=a.cpu,
        gradient_checkpointing=True, seed=42, report_to="none")
    Trainer(model=model, args=args, train_dataset=ds,
            data_collator=collate).train()
    model.save_pretrained(a.out + "-adapter")
    print("SMOKE_TRAIN_OK", flush=True)


if __name__ == "__main__":
    main()
