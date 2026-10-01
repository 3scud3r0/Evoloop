# Evoloop

**Um assistente local, seguro e evolutivo para modelos executados no seu computador.**

O Evoloop oferece uma interface de chat local, conecta-se a Ollama/llama.cpp/vLLM por uma API OpenAI-compatible e permite que o modelo trabalhe em um workspace isolado. Prompts, ferramentas, memória e políticas do agente podem ser avaliados, versionados, promovidos ou revertidos; a fronteira de segurança permanece fora da evolução.

> **Estado do projeto:** alpha para pesquisa. Use apenas em workspaces sem segredos e mantenha a rede desativada. Nenhum sandbox de software torna código hostil perfeitamente seguro.

## Instalação rápida (Linux)

Pré-requisitos: Python 3.10+ e um servidor local como Ollama.

```bash
git clone <URL-DO-REPOSITORIO>/Evoloop.git
cd Evoloop
./install.sh
ollama pull qwen3:4b
evoloop app
```

O comando `evoloop app` inicia o serviço somente em `127.0.0.1`, gera uma chave efêmera, abre a interface no navegador e usa por padrão `http://127.0.0.1:11434/v1`. Para outro servidor/modelo:

```bash
evoloop app --base-url http://127.0.0.1:8080/v1 --model meu-modelo
```

O workspace padrão é `~/Workspace`. O agente não recebe deliberadamente acesso direto ao desktop real; GUI, quando habilitada, deve executar em display virtual isolado.

## Componentes

- **Interface local:** chat responsivo incluído no pacote, sem CDN ou telemetria.
- **Modelos:** Ollama e qualquer endpoint OpenAI-compatible; Anthropic continua opcional.
- **Ferramentas:** arquivos, shell e macros dentro do ambiente configurado.
- **Segurança:** localhost e autenticação por padrão, rede desligada no sandbox, snapshots e `/undo`.
- **Evolução:** GEPA + seleção estrutural, confirmação, promoção humana e rollback.

## Desenvolvimento

```bash
python -m evoloop selftest
python -m evoloop app --llm rule --env namespaces
```

Leia [`docs/GENERAL_MODE.md`](docs/GENERAL_MODE.md) e [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) antes de expor o serviço ou confiar em resultados experimentais.

---

## EVOLOOP v0.3 — harness evolutivo em ciclo fechado (LLM congelado) + modo agente geral

Fusão **funcional** (não cosmética) de 6 repositórios reais numa cadeia executável:

```
LLM congelado → núcleo cognitivo (memória · planner · router) → tool forging → sandbox → verificador → avaliador
   → experience DB → evolution engine (GEPA + RRSI) → novo harness → teste → { regressão: rollback | melhora: promover } → ciclo
```

## Uso (Python ≥ 3.10, zero dependências para rodar)
```
python -m evoloop run --rounds 8 --seed 1            # ciclo completo, LLM simulado
python -m evoloop run --chaos                        # injeta saboteur a cada round
python -m evoloop run --llm anthropic                # LLM real (ANTHROPIC_API_KEY; NÃO testado aqui)
python -m evoloop run --memory concept --vstore brute|chroma   # A: memória vetorial (chroma: pip install chromadb; usa o ChromaVectorStore do GEPA)
python -m evoloop run --sandbox auto|rlimit|namespaces|nsjail|gvisor --require-isolation   # C: isolamento; --require-isolation = falha fechada
python -m evoloop run --no-swarm / --para-rate 0.5 / --dilution 0.06                          # D e controles do simulador
python -m evoloop selftest                           # 44 testes; integrações opcionais são ignoradas quando indisponíveis
python -m evoloop serve --env auto ...               # v0.3: agente geral para Open WebUI/LibreChat (ver docs/GENERAL_MODE.md)
python scripts/bench2.py 5 [i]                       # ablação v0.2 (A/B/D) -> docs/BENCHMARK_V2.md   (bench.py = v0.1, não regenerado)
```
Saídas em `runs/<nome>/`: `report.json`, `champion_harness.json`, `experience.db`, `registry.db`, `memory.db`.

## Mapa diagrama → código
| caixa | arquivo | origem |
|---|---|---|
| GEPA | `third_party/gepa/` (inteiro, MIT) | **vendorizado** gepa-ai/gepa; `evolution.GepaBridge` é o adaptador |
| RRSI | `third_party/rrsi_core/` (Apache-2.0) | **vendorizado** google-research/rrsi: seleção, banda δ, orçamento L0 anelado, taxonomia K |
| rollback/promoção estatística | `third_party/raven_gates/` (Apache-2.0) | **vendorizado** EverMind-AI/Raven: Fisher exato + lift pareado 2σ |
| MEMORY semântica (A) | `evoloop/embed.py`, `vstore.py`, `memory.py` | próprio; `ChromaStore` reusa `third_party/gepa/adapters/generic_rag_adapter` |
| ROUTER por function calling (B) | `agent._route`, `llm.complete_tools`, `swarm.tool_schema` | próprio; formato `tools`/`tool_choice=auto` da Messages API |
| ORQUESTRADOR + SUB-AGENTES + QA (D) | `evoloop/swarm.py`, `agent._delegate/_qa`, `evolution.structural_candidates` (tag `subagent` do RRSI) | próprio |
| ISOLAMENTO (C) | `evoloop/isolation.py` | próprio: namespaces+chroot ro+seccomp (verificado); nsjail/gVisor/Firecracker (não executados) |
| TOOL FORGING | `evoloop/forge.py` | **reescrito em Python** a partir do desenho do framerslab/agentos (TS) |
| EXECUTION/SANDBOX | `evoloop/sandbox.py` | próprio (AST heurístico + backend de isolamento) |
| PLANNER/ROUTER/VERIFIER | `evoloop/agent.py` | próprio; mecanismos inspirados no jinglv/agent-evolve |
| EVALUATOR | `evoloop/evaluator.py` | próprio |
| EXPERIENCE DB | `evoloop/experience.py` | próprio (SQLite) |
| EVOLUTION ENGINE / NEW HARNESS | `evoloop/evolution.py` | próprio, orquestra GEPA + proposer estrutural RRSI + crítico |
| TEST/EVAL → rollback/promover | `evoloop/loop.py`, `registry.py` | próprio, usa os gates vendorizados |

Ver `UPSTREAM.md` (o que foi copiado, o que não, e por quê) e `docs/LIMITATIONS.md` (leia antes de confiar em qualquer número).

## Novidades da v0.2 (pedidos A–D) — resumo honesto
| | o que mudou | verificado aqui? |
|---|---|---|
| A | memória lexical → vetorial (store exato Python puro ou `ChromaStore` via GEPA) | store Chroma e plumbing: sim. **Embedder neural real: não** (`ConceptEmbedder` é um substituto com léxico de sinônimos escrito por mim) |
| B | router regex → function calling (`router_mode`: `regex`/`llm`/`hybrid`, evolutivo via RRSI) | lógica e validação de esquema: sim. **API nativa ao vivo: não** (SimLLM decide) |
| C | `Sandbox` com backends: rlimit(0) · namespaces+seccomp(1) · nsjail(2) · gVisor(3) · Firecracker(4) | níveis 0 e 1: executados e atacados. 2/3: só linha de comando testada por unidade. 4: só contrato/config |
| D | orquestrador + sub-agentes proponíveis pelo RRSI + revisor QA | mecanismo: sim. **Ganho mensurável: não** (ver BENCHMARK_V2) |

## v0.3 — agente geral (computer use) sobre o harness evolutivo
Novos módulos: `env.py` (Docker/namespaces), `tools_os.py` (bash/arquivos/tela/mouse), `general.py` (ReAct com function calling), `server.py` (API OpenAI-compatível),
`store.py` (turnos, feedback, snapshots/undo), `consolidate.py` + `daemon.py` (sono: replay, juiz, gates do Raven, candidatos PENDENTES), `macro_forge.py`, `testing.py` (duplo de teste).
**Leia `docs/GENERAL_MODE.md`**: lista o que foi testado, o que NÃO foi (LLM real, Open WebUI, Docker) e onde o blueprint original estava errado.
