# Salamandra audit (BH-LLM-Basic Faza 0)

Datum: 2026-11-11. Izvori uz svaku tvrdnju; NEPROVJERENO gdje nije potvrđeno.

## Salamandra-2B (`BSC-LT/salamandra-2b`)
Decoder-only Llama-tip: 24 sloja, hidden 2048, FFN 5440, 16/16 glava (MHA), 2.25B ukupno, vocab 256k, context 8192, RoPE theta 10k, SwiGLU/RMSNorm, BF16, FlashAttention. Tokenizer: SentencePiece BPE 256k (`LlamaTokenizer`, bos 1, eos 2), byte_fallback, split_digits, NFC. Trening: 12.875T tokena, NeMo/MN5 (256 GPU). Težine: 1× safetensors 4.51GB. Revizija težina v1.1: `1812f3829e12308541b3ea2fede5366f29d3ee92`.
Izvori: model card + config.json + arXiv:2502.08489.

## Salamandra-7B (`BSC-LT/salamandra-7b`)
32 sloja, hidden 4096, FFN 11008, 32Q/8KV (GQA), 7.77B ukupno; ostalo isto kao 2B (vocab 256k, ctx 8192, BF16). Težine: 4 sharda, 15.6GB. Revizija v1.1: `110c55b5f40310a0f806837234292bf341bac909`.

## Licenca (oba): Apache-2.0
Komercijala dozvoljena; atribucija: citat `gonzalezagirre2025salamandratechnicalreport, arXiv:2502.08489` + Copyright(c) 2024 LTU-BSC.

## Jezici — VAŽNO
35 jezika + kod; sr/sh potvrđeni u korpusu, `bs,hr,srWaC` kao izvor. Zaseban `bs` udio NEPROVJEREN; bs/hr/sr NISU u gold benchmarku (samo es/ca/eu/gl/en) → kvalitet na našim jezicima nepoznat. Prije odluke: izmjeriti tokenizer fertility + perplexity na bosanskom.

## 4B — nema prečice
Naivno rezanje 7B lomi GQA/RoPE/SwiGLU — zabranjeno. Opcije:
- (A) Nova 4B arh. (28×3072, GQA 24/8, isti vocab) + continued-pretraining (~50–200B tokena; ~8×H100 dani–sedmice — grubo, NEPROVJERENO)
- (B) Distilacija 7B→4B na bs/hr/sr (×1.5–3 compute od A; teacher prenosi i rupe na bs)
- (C) Odmah na 2B bazi (najjeftinije, 1–2 GPU, dani) — bez pravog 4B
Preporuka: (C) za brz bosanski rezultat + evaluacija; (A)+(B) ako se ide na pravi 4B.

## Ograničenja baze
Base (ne instruct), kontekst 8k, biasi (BBQ, primacy, majority-class), reprodukcija ±1.5%.
