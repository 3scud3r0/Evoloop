"""Ablação v0.2 (A memória semântica, B router por function calling, D swarm). LLM=SimLLM; backend de sandbox = o auto-detectado.
Uso: python scripts/bench2.py [n_seeds]  -> docs/BENCHMARK_V2.md"""
import datetime
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from evoloop.loop import EvoLoop, LoopConfig  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 5
ONLY = int(sys.argv[2]) if len(sys.argv) > 2 else None      # índice da variante (permite rodar em partes)
PART = Path("/tmp/b2_rows"); PART.mkdir(exist_ok=True)
BASE = dict(rounds=8, para_rate=0.5)
V = [("v0 lexical + regex, sem swarm", dict(memory="lexical", use_router=False, use_swarm=False, dilution=0.06)),
     ("+A memória vetorial", dict(memory="concept", use_router=False, use_swarm=False, dilution=0.06)),
     ("+A+B router function-calling", dict(memory="concept", use_router=True, use_swarm=False, dilution=0.06)),
     ("+A+B+D swarm (completo)", dict(memory="concept", use_router=True, use_swarm=True, dilution=0.06)),
     ("+A+B sem swarm, dilution=0", dict(memory="concept", use_router=True, use_swarm=False, dilution=0.0)),
     ("+A+B+D swarm, dilution=0", dict(memory="concept", use_router=True, use_swarm=True, dilution=0.0))]


def ms(x):
    return f"{st.mean(x):+.3f} ± {st.stdev(x):.3f}" if len(x) > 1 else f"{x[0]:+.3f}"


rows = []
for vi, (name, kw) in enumerate(V):
    if ONLY is not None and vi != ONLY:
        rows.append((PART / f"{vi}.txt").read_text().strip() if (PART / f"{vi}.txt").exists() else "")
        continue
    fe, fo, ch, pr, so = [], [], [], [], []
    ro = sw = 0
    for sd in range(1, N + 1):
        rep = EvoLoop(LoopConfig(seed=sd, out=f"/tmp/b2/{abs(hash(name)) % 9999}_{sd}", **BASE, **kw), quiet=True).run()
        f, o = rep["final"]["evolve_fresh"], rep["final"]["ood_test"]
        fe.append(f["lift"]); fo.append(o["lift"]); ch.append(f["champion_S"]); so.append(f["seed_S"]); pr.append(len(rep["promotions"]))
        sw += bool(rep["champion_subagents"]); ro += rep["champion_config"]["router_mode"] != "regex"
    rows.append(f"| {name} | {st.mean(so):.2f}→{st.mean(ch):.2f} | {ms(fe)} | {ms(fo)} | {sum(x > 0 for x in fo)}/{N} | {st.mean(pr):.1f} | {ro}/{N} | {sw}/{N} |")
    print(rows[-1], flush=True)
    (PART / f"{vi}.txt").write_text(rows[-1])
hdr = ["# Benchmark v0.2 (SimLLM; 50% das tarefas parafraseadas)", "",
       f"Gerado por `python scripts/bench2.py {N}` em {datetime.date.today()}. **Valida o mecanismo, não capacidade de LLM real.**", "",
       "`dilution` é uma SUPOSIÇÃO do simulador (interferência de instruções em prompts longos). Com dilution=0, sub-agentes não têm vantagem de acerto por construção.", "",
       "| variante | seed→campeão (evolve) | lift evolve | lift OOD-teste | OOD>0 | promoções/run | router≠regex no campeão | campeão com sub-agente |", "|---|---|---|---|---|---|---|---|"]
if ONLY is None or all(rows):
    Path(ROOT, "docs", "BENCHMARK_V2.md").write_text("\n".join(hdr + rows) + "\n")
