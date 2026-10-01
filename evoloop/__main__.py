import argparse
import json
import subprocess
import sys
from pathlib import Path

from . import ui
from .loop import EvoLoop, LoopConfig


def _general(a) -> int:
    import os, secrets, threading, webbrowser
    from .server import AppState, serve
    from .env import make_env, NamespaceEnv
    ws, state = os.path.expanduser(getattr(a, "workspace", "~/Workspace")), os.path.expanduser(a.state)
    if a.llm == "anthropic":
        from .llm import AnthropicLLM
        llm = AnthropicLLM()
    elif a.llm == "local":
        from .llm import OpenAICompatibleLLM
        llm = OpenAICompatibleLLM(model=a.model, base_url=a.base_url, api_key=a.model_api_key)
    else:
        from .testing import RuleLLM
        llm = RuleLLM()
    if a.cmd in ("serve", "app"):
        env = make_env(a.env, ws, **({"image": a.image, "network": a.docker_network, "runtime": a.runtime} if a.env in ("docker", "auto") else {}))
        key = a.api_key or os.environ.get("EVOLOOP_API_KEY") or secrets.token_urlsafe(18)
        win = None if a.window == "any" else tuple(int(x) for x in a.window.split("-"))
        app = AppState(state, ws, env, llm, key, memory_kind=a.memory, auto_promote=a.auto_promote, idle_s=a.idle_min * 60, window=win, gui=a.gui, use_gepa=a.gepa)
        print(ui.cyan("EVOLOOP agente geral") + f"  http://{a.host}:{a.port}/v1   chave: {key}")
        if a.cmd == "app":
            url = f"http://{a.host}:{a.port}/?token={key}"
            print(ui.cyan("interface") + f"  {url}")
            threading.Timer(0.8, lambda: webbrowser.open(url)).start()
        print(ui.dim(f"ambiente: {env.describe()}  | consolidação: {'automática' if a.auto_promote else 'com aprovação'}, janela {a.window}"))
        if a.host not in ("127.0.0.1", "localhost"):
            print(ui.amber("⚠ escutando fora de localhost: qualquer um com a chave executa comandos no ambiente. Use firewall."))
        serve(app, a.host, a.port)
        return 0
    app = AppState(state, ws, NamespaceEnv(ws), llm, "-", daemon=False)
    if a.cmd == "pending":
        print(json.dumps(app.registry.pending(), ensure_ascii=False, indent=1, default=str))
    elif a.cmd == "consolidate":
        print(json.dumps(app.daemon.run_once().as_dict(), ensure_ascii=False, indent=1, default=str))
    else:
        print("ok" if (app.consolidator.approve if a.cmd == "approve" else app.consolidator.decline)(a.v) else "versão pendente não encontrada")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="evoloop", description="Harness evolutivo em ciclo fechado (LLM congelado)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="executa o ciclo completo")
    r.add_argument("--rounds", type=int, default=6)
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--out", default="runs/demo")
    r.add_argument("--llm", choices=("sim", "anthropic"), default="sim")
    r.add_argument("--chaos", action="store_true", help="injeta um candidato saboteur a cada round (deve ser barrado)")
    r.add_argument("--memory", default="lexical", choices=("lexical", "hash", "concept", "http", "st"))
    r.add_argument("--vstore", default="brute", choices=("brute", "chroma"))
    r.add_argument("--sandbox", default="auto", choices=("auto", "rlimit", "namespaces", "nsjail", "gvisor"))
    r.add_argument("--require-isolation", action="store_true", help="falha fechada se só houver rlimit")
    r.add_argument("--para-rate", type=float, default=0.5)
    r.add_argument("--dilution", type=float, default=0.0)
    r.add_argument("--no-swarm", action="store_true")
    r.add_argument("--no-gepa", action="store_true")
    r.add_argument("--no-structural", action="store_true")
    r.add_argument("--gepa-budget", type=int, default=160)
    def general_args(parser):
        parser.add_argument("--state", default="~/.evoloop")
        parser.add_argument("--workspace", default="~/Workspace")
        parser.add_argument("--env", default="auto", choices=("auto", "docker", "namespaces"))
        parser.add_argument("--image", default="evoloop-agent:latest")
        parser.add_argument("--docker-network", default="none", help="none (padrão) | bridge (internet aumenta o risco de exfiltração)")
        parser.add_argument("--runtime", default=None, help="ex.: runsc (gVisor) para o contêiner")
        parser.add_argument("--llm", default="local", choices=("local", "anthropic", "rule"))
        parser.add_argument("--model", default=None, help="modelo local (padrão: EVOLOOP_MODEL ou qwen3:4b)")
        parser.add_argument("--base-url", default=None, help="API OpenAI-compatible (padrão: http://127.0.0.1:11434/v1)")
        parser.add_argument("--model-api-key", default=None, help="chave do servidor de modelo; use variável de ambiente quando possível")
        parser.add_argument("--host", default="127.0.0.1")
        parser.add_argument("--port", type=int, default=8000)
        parser.add_argument("--api-key", default=None)
        parser.add_argument("--memory", default="lexical", choices=("lexical", "hash", "concept", "http", "st"))
        parser.add_argument("--gui", action="store_true", help="habilita GUI somente no display virtual isolado")
        parser.add_argument("--auto-promote", action="store_true", help="PERIGOSO: promove mudanças automaticamente")
        parser.add_argument("--gepa", action="store_true", help="inclui reescrita de prompt via GEPA na consolidação")
        parser.add_argument("--idle-min", type=float, default=30)
        parser.add_argument("--window", default="2-6", help="janela horária, ex. 2-6; 'any' = sempre")
    sv = sub.add_parser("serve", help="servidor OpenAI-compatible para integrações")
    general_args(sv)
    app = sub.add_parser("app", help="abre a interface local do Evoloop no navegador")
    general_args(app)
    for name, hp in (("pending", "lista mudanças pendentes"), ("consolidate", "roda a consolidação agora"), ("approve", "aprova a versão N"), ("decline", "recusa a versão N")):
        q = sub.add_parser(name, help=hp)
        q.add_argument("--state", default="~/.evoloop")
        q.add_argument("--workspace", default="~/Workspace")
        q.add_argument("--llm", default="anthropic", choices=("anthropic", "rule"))
        if name in ("approve", "decline"):
            q.add_argument("v", type=int)
    sub.add_parser("selftest", help="roda a suíte de testes")
    s = sub.add_parser("show", help="resume um relatório")
    s.add_argument("path")
    a = ap.parse_args()
    if a.cmd == "selftest":
        return subprocess.call([sys.executable, "-m", "unittest", "discover", "-s", str(Path(__file__).resolve().parent.parent / "tests"), "-v"])
    if a.cmd == "show":
        d = json.loads(Path(a.path, "report.json").read_text())
        print(json.dumps({k: d[k] for k in ("final", "llm_calls", "episodes", "forge_rejections", "honestidade")}, indent=1, ensure_ascii=False))
        return 0
    if a.cmd in ("serve", "app", "pending", "consolidate", "approve", "decline"):
        return _general(a)
    cfg = LoopConfig(rounds=a.rounds, seed=a.seed, out=a.out, llm=a.llm, chaos=a.chaos, gepa_budget=a.gepa_budget, use_gepa=not a.no_gepa, use_structural=not a.no_structural,
                     memory=a.memory, vstore=a.vstore, sandbox=a.sandbox, require_isolation=a.require_isolation, para_rate=a.para_rate,
                     dilution=a.dilution, use_swarm=not a.no_swarm)
    rep = EvoLoop(cfg).run()
    for name, f in rep["final"].items():
        print(ui.magenta(f"FINAL {name:<13}") + f" seed={f['seed_S']:.3f} → campeão={f['champion_S']:.3f}  lift={f['lift']:+.3f} z={f['z']}  verif-pass={f['verifier_pass_champion']:.2f} gap-Goodhart={f['goodhart_gap']:+.2f}")
    print(ui.dim(f"chamadas LLM {rep['llm_calls']} · episódios {rep['episodes']} · rejeições forge {rep['forge_rejections']}"))
    print(ui.amber("⚠ " + rep["honestidade"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
