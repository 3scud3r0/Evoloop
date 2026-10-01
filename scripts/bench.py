"""Benchmark multi-seed com ablação. Gera docs/BENCHMARK.md com números REAIS desta máquina (LLM=sim)."""
import datetime
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from evoloop.loop import EvoLoop, LoopConfig  # noqa: E402

SEEDS = range(1, 9)
VARIANTS = {"completo (GEPA+RRSI)": {}, "sem GEPA": {"use_gepa": False}, "sem estrutural (só GEPA)": {"use_structural": False}, "completo + saboteur": {"chaos": True}}


def ms(xs):
    return f"{st.mean(xs):+.3f} ± {st.stdev(xs):.3f}"


def main():
    lines = ["# Benchmark (LLM = SimLLM determinístico)", "",
             f"Gerado por `python scripts/bench.py` em {datetime.date.today()}; seeds {min(SEEDS)}..{max(SEEDS)}, 8 rounds máx. por run.", "",
             "**Leitura correta:** SimLLM valida o MECANISMO (propor→testar→promover/reverter). Os números abaixo NÃO medem capacidade de LLM real.", "",
             "| variante | evolve: seed→campeão | lift evolve (média±dp) | lift OOD-teste (média±dp) | OOD lift>0 | promoções/run | descartes/run | saboteur admitido |",
             "|---|---|---|---|---|---|---|---|"]
    for name, kw in VARIANTS.items():
        le, lo, s0, s1, pro, dis, sab, pos = [], [], [], [], [], [], 0, 0
        for sd in SEEDS:
            cfg = LoopConfig(rounds=8, seed=sd, out=f"/tmp/bench/{abs(hash(name)) % 9999}_{sd}", **kw)
            rep = EvoLoop(cfg, quiet=True).run()
            f, o = rep["final"]["evolve_fresh"], rep["final"]["ood_test"]
            le.append(f["lift"]); lo.append(o["lift"]); s0.append(f["seed_S"]); s1.append(f["champion_S"]); pos += o["lift"] > 0
            pro.append(len(rep["promotions"]))
            dis.append(sum(1 for r in rep["rounds"] if r["winner"] and not r["promoted"]))
            sab += sum(1 for r in rep["rounds"] for c in r["candidates"] if c["source"] == "chaos" and c["admissible"])
        lines.append(f"| {name} | {st.mean(s0):.2f}→{st.mean(s1):.2f} | {ms(le)} | {ms(lo)} | {pos}/{len(SEEDS)} | {st.mean(pro):.1f} | {st.mean(dis):.1f} | {sab} |")
        print(lines[-1], flush=True)
    Path(ROOT, "docs", "BENCHMARK.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
