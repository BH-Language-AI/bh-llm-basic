# BH-LLM-Basic 2B — miks v0.1 (DRAFT, ceka PPL filter + LUMI smoke)

Baza: `BSC-LT/salamandra-2b` @ `1812f382`. Render: `scripts/render_salamandra.py`.
Mjerenja: fertility **1.57 tok/rijec** (256k vocab, CPU, 8 bs recenica);
izvorni profili u `docs/SOURCE_PROFILES_2B.json` (cap 2000, heuristike grube).

## CPT blok (tekuci tekst, `sample_to_cpt`)

| Izvor | Kolicina | Licenca | Uslov ulaska |
|---|---|---|---|
| bosnian_core v0.2 (Sag69) | 98.224 | MIT | cijeli — visok kvalitet |
| bosnian_core v0.3 (corpus-v1) | ~100k od 300k | CC-BY-SA-4.0 | PPL/dedup filter (baseline u toku); ostatak otpada |
| wikipedia-plaintext (`ffatty/...`) | cijeli | CC-BY-SA-3.0 | dedup vs knowledge v0.4/0.5 |
| klix-news (`Seferovic/...`) | cijeli (100K–1M) | provjeriti karticu | ije 0.76, cir 0.7% — direktno |
| parliament-700 | 563 | MIT | cijeli — formalan registar |

Pravila: ijekavica primarna (filter na ulazu, prag TBD iz profila);
latinica primarna, cirilica ≤5% (samo razumijevanje); EN ≤15% (tehnika/kod).

## SFT blok (`sample_to_text`, bosanske oznake + `</s>`)

| Izvor | Kolicina | Napomena |
|---|---|---|
| bosnian_instructions v0.1 | 1.500 | sjeme — prosiriti 10× u v0.2 |
| knowledge_wikipedia_bs v0.4+v0.5 | 16.000 | atributivni QA |
| FineTome-Alpaca-Bosnian | filtrirano (ije>0.5 redovi) | Apache-2.0; EK redovi ODBACENI (ije_share 0.57 sirovo) |
| alpaca-bosnian-cleaned | dedup vs FineTome | rezerva ako fali volumen |
| function_calling + agentic_terminal | ~16k | Charter funkcije/tokovi |
| safety v0.3–v0.5 | 4.500 | obavezno (bias baze!) |
| coding_repo + swe_gym + linux_terminal | ~6.8k | kod/EN/sysadmin |
| thinking / confirmation | ~1.3k | format disciplina |

Orijentir: ~70% bs / ~15% hr-sr-cnr / ~15% en.

## Held-out (NIKAD u train)

Safety-probe + bs bench (poseban split); `bs-vision-ocr-heldout-v01` prazan —
ne dirati. Leakage: cross-dataset dedup + `leakage_report.json` (reuse
`factory.py`/`release.py` iz agentmujo-traininga) prije svakog runa.

## Status

- [x] fertility 1.57 (CPU)
- [x] single-sentence PPL orijentir (CPU, fp32, sa dijakritikom — sumljivo,
      prvi tokeni dominiraju, ali pattern jasan):
      bs-opce 16.0 / bs-admin 31.9 / en 25.8 / bs-kolok 68.2 /
      bs-lokal (Jelah/Tešanj) 174.6 / bs-knjiz 599.4.
      Zakljucak: opci jezik OK, lokalni toponimi i kolokacije rupe -> CPT ima sta popraviti.
- [x] dokument-level PPL (`scripts/ppl_baseline.py`, klix+wiki po 5 dokova):
      klix 19.6/17.5/11.9/27.9/32.5 (sredina ~21.9);
      wiki 15.4/19.5/19.1/24.1/15.2 (sredina ~18.7).
      Standardni news/enciklopedijski bs baza drzi (10-30 ocekivano za base);
      rupe ostaju lokalno/kolokvijalno/knjizevno (single-sentence test).
      Filter-pravilo v0.1: PPL>100 na LUMI ingestu -> drop (sum); lokalno v0.1
      samo heuristic filteri (duzina, ijekavica udio, dedup hash).
- [x] izvorni profili (4 nova izvora)
- [x] template adapter + self-check
- [x] SFT config (`configs/salamandra-2b-sft01.yaml`)
- [ ] PPL filter prag + zamrznuti manifest + leakage izvjestaj
- [ ] LUMI smoke LoRA (ceka cert refresh + GPU odobrenje — vec trazeno)
