# BH-LLM-Basic 2B — plan punog runa (procjena iz smoke-throughputa)

Mjereno na MI250X (small-g, bf16, LoRA r16, grad_accum 8):
SFT smoke 2.72 s/korak · CPT smoke 2.51 s/korak → racunato s **2.6 s/korak**.

## Procjena (1 GPU, 1 epoha)

| Faza | Podaci | Koraka | Vrijeme |
|---|---|---|---|
| CPT fokus (~200k: wiki 69k + klix 50k + finetome 35k + bc-v03-PPL-top 50k + parl) | ~184k chunkova | ~23.000 | ~17 h |
| CPT puni (401k dok.) | ~369k chunkova | ~46.000 | ~33 h |
| SFT puni (122k redova, seq 2048) | 122k redova | ~15.300 | ~11 h |

Napomena: kalibracija iz smokea (CPT 2000 dok → 230 koraka/epoha; SFT 2000
redova → 250 koraka/epoha; 2.6 s/korak; batch 8). CPT je ~3× skuplji nego u
prvoj procjeni — ispravljeno 2026-11-20.

## Preporuka: A (fokus) pa B (puno) tek ako gate prodje

- **Opcija A (preporuceno):** CPT-fokus ~17 h + SFT ~11 h ≈ **28 h** + eval (dva joba).
- **Opcija B:** CPT-puni ~33 h + SFT ~11 h ≈ **44 h** + eval.
- Oba staju u alokaciju (troseno 663/5000 GPUh); svaki job ide s vlastitim 24h wall-limitom.

## Gate-ovi (iz EVAL_2B.md)

1. CPT gate: held-out PPL mora DOLJE vs base (inace stop — lekcija bh-vision).
2. SFT gate: Belebele-hrv ne smije pasti vs CPT checkpoint; safety-probe 100%.
3. Svaki korak: adapter + log se skida lokalno prije sljedeceg submita.

## Status submita (2026-11-20, vecer)

- Odluka donesena samostalno po nalogu "ti odlucuj": **opcija A, samo CPT-fokus**
  (23.300 koraka, ~17 h); SFT ceka CPT gate i eksplicitnu rijec.
- Submit BLOKIRAN: EFP cert na Oracle skocnom cvoru istekao danas u 17:40
  (vrijedi ~1h). Refresh: portal https://sshca.my-eurohpc.eu (MyAccessID login
  kao susterre) → novi cert u `~/.ssh/id_efp.lumi.csc.fi-cert.pub` na Oracleu
  (140.238.222.253, kljuc `~/.ssh/agentmujo_pro`).
- Kad se cert osvjezi, submit je jedna komanda (testirana procedura):
  `sbatch salamandra-2b-full.sbatch <jobdir> training/train_cpt.py data/cpt_focus.jsonl out/cpt-full 23300`

- `slurm/salamandra-2b-full.sbatch` — genericki: `sbatch ... <jobdir> <train_script> <data> <out> <max_steps>`
- `configs/salamandra-2b-cpt-full01.yaml`, `configs/salamandra-2b-sft-full01.yaml`
- Podaci na HF-u: `BH-Language-AI/bs-llm-2b-mix-v01` (cpt/sft/heldout).
