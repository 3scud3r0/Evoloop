# Procedência e licenças

Commits clonados (shallow) em 2026-09-30 — ver `docs/upstream_commits.txt`.

| repo | commit | uso | licença |
|---|---|---|---|
| gepa-ai/gepa | 3f160c2 | **cópia integral** em `third_party/gepa` (rodando de verdade no ciclo) | MIT |
| google-research/rrsi | be50316 | **cópia sem modificação** de `components,selection,evaluate,schedule,config` | Apache-2.0 |
| EverMind-AI/Raven | e6c0344 | `evolver/orchestrator/gates/{fisher,paired}.py` (1 import trocado) + dataclass `TaskEval` extraída de `scoring.py` (modificada: sem imports do Raven) | Apache-2.0 |
| framerslab/agentos | bdbe2ae | **só desenho** (ForgeShapeValidator, SandboxedToolForge, rejeição por estágio) reescrito em Python. Nada copiado. | Apache-2.0 |
| AetherLabsAI/RSIAgent | a9e5626 | **só ideias**: verificador isolado do ator, memória congelada por hash, exploração ampla→profunda. Nada copiado. | Apache-2.0 |
| jinglv/agent-evolve ("HarnessAgent") | d1b1fec | **só ideias** (loop único, verificação, checkpoint, orçamento). **Sem arquivo LICENSE → nada copiado.** | não declarada |

## Escolhas de identificação (ambíguas — confira)
"AgentOS", "Raven" e "HarnessAgent" têm vários repositórios homônimos. Escolhi: AgentOS = framerslab/agentos (único com *runtime tool forging*, batendo com o diagrama); Raven = EverMind-AI/Raven ("harness of harnesses" para RSI); HarnessAgent = jinglv/agent-evolve (classe `HarnessAgent`, Python). Se você quis outros, diga.

## Por que não vendorizei mais
* **RSIAgent** (16k linhas): acoplado a ambientes (OSWorld/ALE), provisionamento e cliente LLM próprio; não separável sem reescrever.
* **Raven** (2349 .py + 683 .ts): runtime completo com dependências (litellm, pydantic, loguru…). Extraí só os gates estatísticos, que são puros. O roteador do Raven escolhe *modelo* por benchmark, não *ferramenta* por tarefa — não serve ao nosso router.
* **AgentOS**: TypeScript/Node; misturar runtimes quebraria o "zero dependências".
Notas de atribuição: cada pasta de `third_party/` mantém seu LICENSE.

## v0.2
* `third_party/gepa/adapters/generic_rag_adapter/vector_stores/chroma_store.py` (MIT) é usado de fato por `evoloop/vstore.ChromaStore` (requer `pip install chromadb`; testado com 1.5.9 em venv).
* Nenhum código novo foi copiado de outros upstreams. Orquestrador/sub-agentes/QA são implementação própria (Magentic-One/LangGraph foram apenas referência conceitual, não consultados).
