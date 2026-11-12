# BH-LLM-Basic — Roadmap

## Gotovo
- [x] Faza 0: Salamandra audit (2B/7B pinovano, Apache-2.0), framework audit, 4B opcije (A/B/C), kolekcija, repo temelji
- [x] v05 Qwen-4B merge + eval (odvojen AgentMujo LLMs trag — dokaz da merge pipeline radi)

## Sljedeće (redoslijed)
- [ ] F0-verifikacija: Salamandra tokenizer fertility na bosanskom (u toku) + perplexity baseline (treba GPU)
- [ ] Odluka: transfer `agentmujo-training` → org (čeka odobrenje)
- [ ] Odluka: 4B smjer A/B/C (preporuka: C za brz rezultat, A+B za pravi 4B)
- [ ] Faza 1: registracija Salamandre u framework (model unos, template adapter, configovi)
- [ ] Faza 2: baseline eval 2B na bs (fertility ✅ → perplexity → bench)
- [ ] Faza 3: dataset registar + bs held-out set
- [ ] Faza 4: LUMI compute plan (čeka provjeru alokacije)
- [ ] Faza 5+: trening — samo uz odobrenje i plan

## Pravila
Bez treninga/objava/migracija bez odobrenja. Svaki skor reproducibilan.
