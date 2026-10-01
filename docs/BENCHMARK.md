# Benchmark (LLM = SimLLM determinístico)

Gerado por `python scripts/bench.py` em 2026-09-30; seeds 1..8, 8 rounds máx. por run.

**Leitura correta:** SimLLM valida o MECANISMO (propor→testar→promover/reverter). Os números abaixo NÃO medem capacidade de LLM real.

| variante | evolve: seed→campeão | lift evolve (média±dp) | lift OOD-teste (média±dp) | OOD lift>0 | promoções/run | descartes/run | saboteur admitido |
|---|---|---|---|---|---|---|---|
| completo (GEPA+RRSI) | 0.42→0.98 | +0.562 ± 0.024 | +0.096 ± 0.055 | 7/8 | 3.0 | 1.2 | 0 |
| sem GEPA | 0.42→0.99 | +0.573 ± 0.017 | +0.010 ± 0.086 | 5/8 | 3.0 | 0.9 | 0 |
| sem estrutural (só GEPA) | 0.42→0.97 | +0.549 ± 0.018 | +0.144 ± 0.080 | 7/8 | 2.2 | 0.4 | 0 |
| completo + saboteur | 0.42→0.98 | +0.562 ± 0.024 | +0.096 ± 0.055 | 7/8 | 3.0 | 1.2 | 1 |
