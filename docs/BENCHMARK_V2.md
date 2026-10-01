# Benchmark v0.2 (SimLLM; 50% das tarefas parafraseadas)

Gerado por `python scripts/bench2.py 5` em 2026-10-01. **Valida o mecanismo, não capacidade de LLM real.**

`dilution` é uma SUPOSIÇÃO do simulador (interferência de instruções em prompts longos). Com dilution=0, sub-agentes não têm vantagem de acerto por construção.

| variante | seed→campeão (evolve) | lift evolve | lift OOD-teste | OOD>0 | promoções/run | router≠regex no campeão | campeão com sub-agente |
|---|---|---|---|---|---|---|---|
| v0 lexical + regex, sem swarm | 0.42→0.88 | +0.453 ± 0.074 | +0.142 ± 0.186 | 4/5 | 3.0 | 0/5 | 0/5 |
| +A memória vetorial | 0.42→0.93 | +0.507 ± 0.034 | +0.024 ± 0.055 | 4/5 | 3.6 | 0/5 | 0/5 |
| +A+B router function-calling | 0.42→0.94 | +0.517 ± 0.050 | +0.040 ± 0.061 | 3/5 | 3.6 | 3/5 | 0/5 |
| +A+B+D swarm (completo) | 0.42→0.94 | +0.521 ± 0.031 | +0.184 ± 0.295 | 4/5 | 3.8 | 3/5 | 3/5 |
| +A+B sem swarm, dilution=0 | 0.42→0.95 | +0.531 ± 0.010 | +0.104 ± 0.088 | 4/5 | 2.4 | 1/5 | 0/5 |
| +A+B+D swarm, dilution=0 | 0.42→0.97 | +0.542 ± 0.017 | +0.144 ± 0.088 | 4/5 | 2.4 | 0/5 | 3/5 |
