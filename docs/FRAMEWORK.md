# AgentMujo Training — compatibility report (BH-LLM-Basic Faza 0)

Datum: 2026-11-11. Repo: `/home/sha/agentmujo-training` (remote `xnet-ba`), čitano bez izmjena. Riječ "salamandra" se nigdje ne pojavljuje — podrška ne postoji.

## Šta postoji (generičko, radi za Salamandru bez izmjena)
- Dataset pipeline: validator (GOLD/SILVER/BRONZE/REJECT), factory (ingest/clean/dedup/build/export), release (cross-dataset dedup, held-out) — `src/agentmujo_training/datasets/`
- Eval scorer + agent loop parser (`benchmark/runner.py`, `agent/loop.py` — parsira tekst, ne zavisi od tokenizera)
- CLI `amj`, job/manifest sistem, DPO `{prompt,chosen,rejected}` format, LoRA target moduli (standardni nazivi — provjeriti na Salamandri)
- Licenca frameworka: Apache-2.0; core lagan (pyyaml/jsonschema/hf-hub/datasets), train extra (torch/transformers/trl/peft)
- Testovi: 14 fajlova, CI pytest

## Qwen-vezano (mora se dirati)
1. `training/format.py` — hardkodiran ChatML + `<tool_call>` XML trening tekst; treba per-model renderer
2. `workers/kaggle/job_run.py` — oslanja se na `tok.apply_chat_template` baznog modela; neprovjereno da li Salamandra template podržava `tools`
3. `smoke_test.py:54-60` — assert-a `"tool_call"` u templateu; pada za druge formate
4. `configs/models.yaml` — cijeli registar Qwen3.5; Salamandra se dodaje kao novi unos
5. `configs/training/*.yaml` (~76) + `training/jobs/qwen35-*` — svi pinuju Qwen baze; treba `salamandra-*.yaml` serija
6. `quantization/` (`patch_gguf.py` block_count, llama patch) + `deployment/Modelfile.*` (`RENDERER qwen3.5`, `enable_thinking`) — cijeli GGUF/Ollama put Qwen-specifičan
7. `factory.py` export format `qwen`, benchmark SYSTEM promptovi (Qwen stil — baseline se resnimava)
8. `project.yaml:5`, `release.py:179`, README/CITATION — kanon URL `xnet-ba` (README čak navodi zastarjeli `gamai-io`)

## Šta fali za Salamandru
Template adapter, registry unos (revizija/vocab/kontekst/licenca/tool-format), training configovi, GGUF/Ollama protokol, parametrizirani smoke testovi, provjera Salamandra chat template tool-capability (prvi verifikacijski korak).

## Migracija xnet-ba → BH-Language-AI/agentmujo-training
Preporuka: transfer (čuva historiju/issues/PR-ove), ne fork. Zatim: kanon URL-ovi (koraci 2–8 iz punog izvještaja), branch protection + CI, odluka o HF namespace-u (`shaban2024/*` vs org) jer configovi pinuju datasete. Bez brisanja originala, bez nepovratnih koraka bez odobrenja.
