# BH-LLM-Basic — Project Charter

Misija: otvorena familija modela za bosanski (ijekavica, latinica; razumijevanje ćirilice) i Balkan (hr/sr/cnr/sl/mk/sq) + engleski za tehniku. Namjene: QA, komunikacija, dokumenti, sumarizacija, kod, JSON, function-calling, agentski tokovi, lokalni self-hosted rad.

Pravila:
- Tri veličine, bez zamjene: 2B, 4B, 7B.
- 4B se ne "reže" iz 7B — samo opravdana arhitektura/distilacija.
- Svaki rezultat mjerljiv: baseline prije treninga, bez izmišljenih skorova.
- Test podaci odvojeni od treninga; bez privatnih/nelicenciranih podataka.
- GPU trening, objave, migracije: samo uz odobrenje.
- Vanjski resursi se referenciraju; licence i atribucije se poštuju (Salamandra: Apache-2.0 + citat arXiv:2502.08489).
