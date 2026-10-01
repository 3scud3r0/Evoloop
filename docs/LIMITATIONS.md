# Limitações (v0.2) — leia antes de citar qualquer número

## Geral
1. **Nenhum LLM nem embedder neurais reais foram usados.** Sem chaves/pesos neste ambiente. `AnthropicLLM` (inclui `complete_tools`), `HTTPEmbedder`
   e `SentenceTransformerEmbedder` existem e **nunca foram executados**. Todos os números usam `SimLLM` + `ConceptEmbedder`, cujo modelo de mundo
   está escrito em `sim.py`/`embed.py`. Os resultados validam o MECANISMO do ciclo, não capacidade real.
2. Não é RSI: pesos congelados; só o harness evolui. Teto = LLM base + qualidade do avaliador.
3. Tarefas sintéticas de string (8 famílias). Transferência OOD continua ruidosa (lift OOD-teste de −0,3 a +0,5 por seed; dp 0,06–0,30).

## A — memória semântica
* `recall@1` em instruções 100% parafraseadas: léxica 0,48 · char-3gram 0,50 · `ConceptEmbedder` 0,88 (`recall@2` 1,00). **O substituto é um léxico
  de sinônimos feito à mão, quase um oráculo para este domínio**: prova que paráfrase passa a ser recuperada, não mede um embedder de verdade.
* `ChromaStore` roda de verdade (chromadb 1.5.9, via `ChromaVectorStore` do GEPA vendorizado, com vetores nossos; o embedder padrão do Chroma baixa
  pesos e não funciona offline). Qdrant/LanceDB/Milvus/Weaviate do GEPA **não** foram ligados.
* No ablation, +A rendeu lift evolve +0,507 vs +0,453 (n=5, dp 0,03–0,07): sugestivo, **não significativo**.

## B — router por function calling
* No SimLLM, "decidir chamar a ferramenta" = detectar a família com oráculo + 6% de omissão + 4% de ferramenta errada. Micro-teste (ferramentas pré-instaladas,
  100% paráfrases): regex 0,475 → function calling 0,942. **Dentro do ciclo evolutivo o ganho extra foi indistinguível de ruído** (+0,507 → +0,517; o campeão
  adotou router≠regex em 3/5 seeds). A acurácia de function calling de um LLM real com esses esquemas é desconhecida.
* Custo: +1 chamada de LLM por tarefa roteada (demo: 1605 chamadas `router`).

## C — isolamento
| nível | backend | estado |
|---|---|---|
| 0 | rlimit + seccomp | executado; **não isola o FS do host** (teste documenta: lê /etc/passwd) |
| 1 | user/pid/net/mount/ipc/uts ns + raiz tmpfs com /usr,/lib ro + /tmp noexec + caps zeradas + no_new_privs + seccomp | executado e atacado: sem rede, sem /etc /home, escrita em /usr = EROFS, fork/exec/socket/mount negados, só PID 1 visível |
| 2 | nsjail | comando gerado, **nunca executado** |
| 3 | gVisor (`docker --runtime=runsc --network=none --read-only --cap-drop=ALL`) | comando gerado, **nunca executado** |
| 4 | Firecracker | **só contrato + gerador de config**; falta o agente-convidado (init que lê o payload por vsock). `available()` é False de propósito |

* Nível 1 ainda compartilha o **kernel do host** (superfície de syscalls não filtradas pelo seccomp). Seccomp é denylist x86_64 (em outras arquiteturas não filtra).
  Sem cgroups: memória/CPU só por RLIMIT_AS/CPU. Para código realmente hostil, use nível ≥3.
* **Achado:** o AST da v0.1 era escapável. `operator.attrgetter("__class__.__base__")(())` passou no filtro e alcançou `object.__subclasses__()` e o módulo
  `sys` (reproduzido na v0.1; a falha final foi só `KeyError: 'os'`). Corrigido removendo `operator`, banindo `attrgetter/format/Formatter/gi_frame/f_*`, nomes
  `type/object` e strings com `__`. **O AST continua heurístico**: strings montadas dinamicamente sempre o contornam em tese — por isso os testes de contenção
  atacam o backend SEM passar pelo AST.
* Overhead: 42 ms/exec (rlimit) vs 72 ms/exec (namespaces).
* `Sandbox(require_isolation=True)` / `--require-isolation` falha fechado se o melhor backend for o nível 0.

## D — multi-agente
* Implementado: orquestrador que delega por function calling, sub-agentes especialistas propostos por um "arquiteto" (validados, checados pelo crítico anti-vazamento,
  tag `subagent` do RRSI confirmada por evidência no diff), revisor QA independente (re-derivação; reduz o gap Goodhart ao custo de tokens).
* **Sem ganho mensurável:** +A+B 0,517 vs +A+B+D 0,521 (dilution 0,06); 0,531 vs 0,542 (dilution 0). `dilution` é uma **suposição minha** (interferência em prompts
  longos), não um fato medido; mesmo assim o efeito ficou dentro do ruído (n=5).
* O arquiteto do SimLLM é trivial (agrupa as 2 famílias mais fracas). Sub-agentes não têm memória/forja próprias e só existem os tipos especialista e QA
  (não há agente de web/código como no Magentic-One). O valor de verdade dependeria de um arquiteto LLM real.

## Herdados da v0.1 (ainda valem)
* Gap de Goodhart: verificador fraco passa ~99% das respostas OOD com ~40–55% corretas (gap ≈ +0,45–0,59).
* A regra shaped do RRSI admitia regressão paga por economia de tokens; guard ΔS ≥ −δ/4 é decisão minha.
* `docs/BENCHMARK.md` é da v0.1 e **não foi regenerado** (defaults mudaram: para_rate 0,5). Use `docs/BENCHMARK_V2.md`.
* Identificação de repositórios homônimos é minha melhor leitura (UPSTREAM.md).

## v0.3 — agente geral
Ver `docs/GENERAL_MODE.md` (limites completos). Resumo: nenhum LLM, juiz ou Open WebUI reais foram usados; o ciclo noturno foi validado só com um duplo de teste
(`RuleLLM`) e juiz programático; feedback e juiz são sinais fracos (por isso aprovação humana por padrão); replay exclui tarefas com rede/GUI; isolamento nível 1 sem seccomp para o shell.
