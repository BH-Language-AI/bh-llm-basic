# BH-LLM-Basic 2B — eval protokol (DRAFT v0.1)

Pravilo iz Chartera: svaki rezultat mjerljiv, baseline prije treninga, test
odvojen od traina. Ovaj protokol se zamrzava PRIJE prvog GPU runa.

## 1. Sta se mjeri (obosmjereno na bosanski)

| Sloj | Benchmark | Zasto | Status |
|---|---|---|---|
| Razumijevanje | Belebele `hrv_Latn` (proxy) | bs NE POSTOJI u Belebele (provjereno 2026-11-19); hrv najblizi srodnik | hrv proxy |
| Prijevod | Flores-200 `bos_Latn` | GATED dataset (treba pristup) — baseline tek nakon odobrenja | BLOKIRANO |
| Zdrav razum | ARC/hr-sr-bs prijevodi + PIQA-bs (sinteticki?) | NEMA gotovog bs — treba prevesti/provjeriti | GAP |
| Znanje BiH | vlastiti set (50-100 pitanja: ustav, kantoni, historija) | niko drugi ne pokriva | NAPRAVITI, held-out |
| Sigurnost/bias | BBQ-primacy probe + safety-probe (postojeci) | BBQ bias potvrdjen u SALAMANDRA.md | postoji safety |
| Instrukcije | AlpacaEval-bs (prevedeni subset, ljudska ocjena uzorka) | skupa, radi se na kraju | kasnije |
| Perpleksnost | klix/wiki held-out (2% split iz miksa) | gate za CPT kvalitet | SPREMAN (miks v0.1) |
| Tokenizacija | fertility (scripts/fertility_bs.py) | 1.57 izmjereno | GOTOVO |

## 2. Baseline red (prije GPU-a)

1. fertility 1.57 ✅ (CPU)
2. single-sentence PPL ✅ (16-599, pattern dokumentiran)
3. doc-PPL klix ~21.9 / wiki ~18.7 ✅
4. Belebele-bs + Flores-bs zero-shot na base modelu — MOZE NA CPU (mali setovi),
   radi se lokalno cim se stigne; rezultat = sluzbeni baseline u ovaj dokument.
5. BiH-znanje set (50 pitanja) — napisati rucno, drzati VAN traina.

## 3. Gate-ovi (bez prolaza nema sljedece faze)

- CPT gate: held-out PPL mora DOLJE vs base (inace CPT steti — vidi bh-vision lekciju 37→12%).
- SFT gate: Belebele/Flores ne smiju pasti vs CPT checkpoint; safety-probe 100% bez degradacije.
- Publish gate: ljudska ocjena uzorka (2 govornika) + model card + licence/atribucije.

## 4. Sta NIKAD ne ulazi u train

heldout.jsonl iz miksa, BiH-znanje set, Belebele/Flores test splitovi,
`bs-vision-ocr-heldout-v01`, safety-probe. Krsenje = kontaminacija = ponistava run.
