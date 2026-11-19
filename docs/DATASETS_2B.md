# BH-LLM-Basic 2B — dataset plan (Faza 0 → priprema)

Datum: 2026-11-19. Baza: `BSC-LT/salamandra-2b` @ `1812f382` (Apache-2.0, base, vocab 256k, ctx 8k).
Framework: agentmujo-training (transfer u BH-Language-AI u toku — BEZ forka, čeka odobrenje).
Cilj: CPT + SFT miks za bosanski (ijekavica, latinica; razumijevanje ćirilice) + Balkan/hr/sr + EN za tehniku.

## 1. Šta već imamo (reuse, ne izmišljati novo)

Lokalno `~/agentmujo-training/datasets/canonical/` = **481.820 redova**, svi sa ID/licenca/source/tier:

| Blok | Redovi | HF kopija (`BH-Language-AI/*`) | Napomena za 2B |
|---|---|---|---|
| bosnian_core v0.2 (Sag69 MIT, rječnici/književnost) | 98.224 | bs-lang-core-v01 | GOLD za CPT-jezgro, ijekavica OK |
| bosnian_core v0.3 (bosnian-corpus-v1, CC-BY-SA-4.0, prefix-completion) | 300.000 | bs-lang-core-v01 | najveći CPT blok; kvalitet SILVER — treba PPL filter |
| bosnian_instructions v0.1 | 1.500 | bs-lang-instructions-v01 | premalo — proširiti 10× |
| knowledge_wikipedia_bs v0.4+v0.5 | 16.000 | bs-agent-knowledge-v01 | činjenični QA, atribucija URL u metapodaci |
| rag_wikipedia_bs v0.4–v0.6 | 11.000 | (isti) | RAG/memory format, ne CPT |
| function_calling (sve verzije) | ~10.889 | bs-agent-function-calling-v01 | Charter traži function-calling — reuse uz re-template |
| agentic_terminal (sve verzije) | ~5.649 | bs-agent-terminal-v01 | agentski tokovi — reuse |
| safety v0.3–v0.5 | 4.500 | bs-agent-safety-v01 | obavezno u SFT miksu (BBQ/bias baze!) |
| coding_repo + swe_gym | ~5.280 | bs-agent-coding-v01 | kod + EN tehnika |
| linux_terminal (manpages) | 1.500 | bs-agent-linux-terminal-v01 | sysadmin domen |
| thinking / confirmation / dpo | ~500+800+~8.000 | bs-agent-thinking/-confirmation/-dpo-v01 | DPO tek nakon SFT-a |
| joint_train v0.40–v0.45 ogledi | ~10k | bs-agent-joint-v01 | receptura rebalansa, ne final |

Format napomena: svi redovi su trenutno u **Qwen ChatML + `<tool_call>` XML** ovojnici
(`training/format.py`). Za Salamandru treba **template adapter** (base model nema
instruct template — mi definiramo svoj; `smoke_test.py:54-60` assert pada → parametrizirati).
Tekstualni sadržaj se reusa, ovojnica se re-rendera. Bez adaptera nema treninga.

## 2. Vanjski HF dataseti — shortlist s licencom (provjereno 2026-11-19)

CPT (sirovi/čisti bosanski tekst):
- `hyper-efficient-system-llc/bosnian-corpus-v1` — 10M–100M kat., CC-BY-SA-4.0, zip 6.1GB — VEĆ ingestirano (v0.3). Atribucija obavezna.
- `Sag69/bosnian-dataset` — MIT, `ultimate_bosnian_dataset.jsonl` — VEĆ ingestirano (v0.2). Rječnici/književnost = visok kvalitet.
- `Seferovic/bosnian-news-articles-dataset-from-klixba` — 100K–1M, savremeni news jezik — NOVO, predloženo za v0.4 (uzorak u toku).
- `ffatty/plaintext-wikipedia-full-bosnian` — CC-BY-SA-3.0, 10K–100K — NOVO, dedup protiv knowledge v0.4/0.5.
- `zastuck/bosnian-parliament-700` — MIT, formalni/admin registar — NOVO, mali ali vrijedan (zakoni/uprava iz Chartera).
- MaCoCu-bs / CLASSLA-web.bs / bsWaC — NISU na HF pod tim imenima; izvorno: CLARIN.SI / opus.nlpl.eu. Status: pending, ne skidati cijelo.
- `uonlp/CulturaX` bs podkorpus — ODC-BY + CC0 + Common Crawl ToU, **gated** (nalog shaban2024 nema pristup, skripta `ingest_culturax_bc_v04.py` spremna) — samo 1237 dokumenata, niski prioritet.
- `Helsinki-NLP/opus-100` bs-en — licenca **unknown**, ingestija STOJI dok se ne razjasni.

SFT/instruct (gotovi instruction parovi):
- `TimesLast/FineTome-Alpaca-Bosnian` — **Apache-2.0**, 10K–100K, bs/hr/sr — TOP kandidat za instruct bazu.
- `saillab/alpaca-bosnian-cleaned` + `saillab/alpaca_bosnian_taco` — 10K–100K — uzeti cleaned, dedup vs FineTome.
- `datatab/alpaca-cleaned-serbian-full` (+30000/10000) i `saillab/alpaca-croatian-cleaned` — cross-transfer hr/sr, manji udio (≤10%).
- `saillab/alpaca_serbian_taco` / `aleksandard/alpaca-gpt4-serbian` — rezerva.

## 3. Miks prijedlog za 2B (podložno baseline evaluaciji)

- **CPT faza:** v0.2 (98k, visok kvalitet) + v0.3-filtrirano (PPL/dedup, ne svih 300k) + wikipedia-plaintext + klix-news + parliament. Cilj: dominacija ijekavice, latinica primarna, ćirilica ≤5% (razumijevanje, ne proizvodnja).
- **SFT faza:** bosnian_instructions-prošireno + FineTome-bs (Apache-2.0) + alpaca-bs-cleaned + knowledge-QA + function_calling + agentic + safety + coding. Omjer jezika orijentir: ~70% bs / ~15% hr-sr-cnr / ~15% en.
- **Held-out:** benchmark i safety-probe POTPUNO odvojeni od traina (pravilo iz datasets.yaml). `bs-vision-ocr-heldout-v01` je prazan placeholder — NIKAD u train.
- **Licence za objavu miksa:** MIT (Sag69, parliament) + Apache-2.0 (FineTome, Salamandra) + CC-BY-SA (corpus-v1, wikipedia) — miks je objavljiv uz atribuciju; OPUS-100 i CulturaX-gated NE ulaze dok se status ne riješi.

## 4. Prepreke / odluke koje trebaju odobrenje

1. **LUMI:** EFP cert istekao (`Permission denied (publickey)` 2026-11-19). Treba refresh preko https://sshca.my-eurohpc.eu (vrijedi ~1h) + provjera alokacije project_465003647 / 5000 GPUh. Bez toga nema provjere compute plana.
2. **Transfer** `xnet-ba/agentmujo-training` → `BH-Language-AI/agentmujo-training`: nepovratan klik, čeka odobrenje (iz PLAN.md).
3. **Template adapter + `salamandra-*.yaml` configovi + registry unos** — prvi tehnički korak, mali CPU posao, ne treba GPU odobrenje.
4. **Baseline eval** (fertility + perplexity + bs bench, CPU/mali GPU smoke) — dozvoljeno bez odobrenja kad se definira; `scripts/fertility_bs.py` spreman.
5. Push ovog dokumenta i sljedećih skripti u `BH-Language-AI/bh-llm-basic` — predloženo, čeka potvrdu (pravilo: objave samo uz odobrenje).

## 5. Sljedeći koraci (redom)

1. Fertility + PPL baseline Salamandra-2B na bs uzorcima (CPU).
2. Uzorak + profiliranje: klix-news, wikipedia-plaintext, FineTome-bs, parliament (veličina, ijekavica %, ćirilica %, dužine, licence).
3. Template adapter (render postojećih redova u Salamandra ovojnicu) + 1 parametrizirani smoke test.
4. CPT+SFT miks v0.1 manifest + leakage/dedup izvještaj (reuse `factory.py`/`release.py`).
5. LUMI cert refresh + alokacija check + 1 smoke LoRA run tek uz GPU odobrenje.
