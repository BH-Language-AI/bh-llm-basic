# BH-LLM-Basic

Familija otvorenih jezičkih modela za **bosanski jezik i Balkan**: 2B, 4B, 7B.
Baza: **Salamandra** (Barcelona Supercomputing Center, Apache-2.0).
Framework: **AgentMujo Training** (migracija u toku).

**Status: Faza 0 — istraživanje i audit.** Težine se ne objavljuju bez odobrenja.

- Kolekcija: https://huggingface.co/collections/BH-Language-AI/bh-llm-basic
- Kontakt: xnetba@gmail.com

## Dokumenti
- `docs/PROJECT_CHARTER.md` — misija, opseg, pravila
- `docs/SALAMANDRA.md` — audit baza 2B/7B + opcije za 4B
- `docs/FRAMEWORK.md` — AgentMujo Training compatibility + migracija
- `docs/PLAN.md` — faze, eval plan, compute procjena, odluke koje trebaju odobrenje

## Modeli (ciljevi, ne postojeći artefakti)
| Model | Baza | Status |
|---|---|---|
| BH-LLM-Basic-2B | Salamandra 2B (`BSC-LT/salamandra-2b` @ `1812f382…`) | kandidat potvrđen |
| BH-LLM-Basic-4B | arh. prijedlog u SALAMANDRA.md (opcije A/B/C) | NEMA checkpointa |
| BH-LLM-Basic-7B | Salamandra 7B (`BSC-LT/salamandra-7b` @ `110c55b5…`) | kandidat potvrđen |
