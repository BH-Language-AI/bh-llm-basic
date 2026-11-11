# BH-LLM-Basic — Plan (Faza 0 zaključak)

## Gotovo (Faza 0)
- Salamandra 2B/7B audit: arh, tokenizer, licenca, revizije pinovane; bs pokrivenost tanka/neevaluirana → obavezna vlastita evaluacija (fertility, perplexity, bench na bs)
- 4B opcije: (A) nova arh + CPT, (B) distilacija 7B→4B, (C) 2B odmah. Preporuka: (C) za brz rezultat, (A)+(B) za pravi 4B
- Framework audit: generički dataset/eval/CLI/DPO dijelovi OK; Qwen-vezano izolirano (format.py, models.yaml, 76 configova, GGUF/Ollama put); fali template adapter + registry + configovi + GGUF protokol
- Migracijski plan: transfer (ne fork), URL zamjene, branch protection, odluka HF namespace
- Kolekcija: BH-LLM-Basic (Salamandra 2B/7B kao eksterni kandidati)
- LUMI: ključ postoji, ali pristup s ove mašine odbijen (korisnik placeholder) — alokacija 5000 GPUh / 250GB / project_465003647 NEPROVJERENA

## Sljedeće (traži odobrenje / odluku)
1. Migracija agentmujo-training (transfer — nepovratan korak, treba klik/odobrenje)
2. Odluka 4B: (A)/(B)/(C) + HF namespace za datasete
3. LUMI pristup: provjera alokacije (treba Shaov cert/konekcija)
4. Baseline eval Salamandra na bosanskom (mali GPU/CPU smoke — dozvoljeno bez odobrenja kad se definira)

## Compute (grubo, NEPROVJERENO)
- 2B CPT/instruct: 1–2 GPU, dani
- 4B (A): ~50–200B tokena, 8×H100 dani–sedmice; (B): ×1.5–3 od A
- 7B: više od 4B; tek nakon 2B/4B rezultata
