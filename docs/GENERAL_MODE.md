# Modo agente geral (v0.3) — o que foi implementado do blueprint e o que foi CORRIGIDO

## Uso rápido
```bash
./install.sh
ollama pull qwen3:4b
evoloop app                                      # UI local + Ollama em http://127.0.0.1:11434/v1
evoloop app --model outro --base-url http://127.0.0.1:8080/v1

# Integração opcional com interface externa/API remota:
export ANTHROPIC_API_KEY=...                      # NÃO testado ao vivo aqui (sem chave)
docker build -f docker/Dockerfile.agent -t evoloop-agent docker/      # imagem do ambiente (não testada: sem docker na máquina de dev)
python -m evoloop serve --env docker --workspace ~/Workspace --host 0.0.0.0 --api-key MINHA_CHAVE
EVOLOOP_API_KEY=MINHA_CHAVE docker compose -f docker/docker-compose.yml up      # Open WebUI em http://localhost:3000
python -m evoloop serve --env namespaces ...      # sem Docker (Linux): shell isolado por namespaces (verificado)
python -m evoloop pending | approve N | decline N | consolidate
```
Chat: `/bad motivo` · `/good` · `/undo` · `/status` · `/pending` · `/approve N` · `/decline N` · `/consolidate`.

## Mapa do blueprint → código
| Passo | Implementação | Estado |
|---|---|---|
| 1. "Falsa API OpenAI" | `server.py` (stdlib `http.server`, SSE, `/v1/models`, `/v1/chat/completions`, Bearer obrigatório) | Testado com `urllib` e com o cliente `openai` 3.22.1 (normal + streaming). **Open WebUI real: não testado (sem docker)** |
| 2. Olhos e mãos | `tools_os.py`: `bash`, `read_file`, `write_file`, `screenshot`, `computer` (xdotool) | bash/arquivos: testados no shell isolado. **GUI: testada com Xvfb+xdotool+scrot reais** (PNG válido, ações, validação). Não testada dentro do contêiner Docker |
| 2b. Forja | `macro_forge.py` (macros bash/python testadas em workspace descartável) | Testado com LLM falso; só na consolidação (ver abaixo) |
| 3. Ator + sono | `daemon.py` (ocioso + janela horária + feedback novo) e `consolidate.py` | Lógica testada com `RuleLLM`; **nunca com LLM/juiz reais** |
| 4. Fronteira | `env.py`: `DockerEnv` (blueprint) e `NamespaceEnv` | Namespaces: executado e atacado. Docker: só linhas de comando testadas |
| Fitness = feedback | `store.py` (feedback explícito/implícito) → replay + juiz → gates do Raven | ver limites |

## Onde o texto do blueprint estava errado ou incompleto (corrigido)
1. **O botão 👎 do Open WebUI NÃO chega a uma API OpenAI-compatível** (só mensagens trafegam). Canais reais implementados: `/bad`/`/good` no chat, `POST /evoloop/feedback` (para um Filter/Pipeline do Open WebUI chamar — **não escrito nem testado**) e detecção implícita por regex PT/EN (peso 0,5–0,6; ruidosa).
2. **`OPENAI_API_BASE_URL=http://localhost:8000` dentro do contêiner aponta para o próprio contêiner.** O compose usa `host.docker.internal` + `extra_hosts: host-gateway`, e o EVOLOOP tem de escutar em `0.0.0.0` (⇒ chave obrigatória + firewall).
3. **Requisições auxiliares:** o Open WebUI chama o mesmo "modelo" para título/tags/sugestões (prompts que começam com `### Task:`). Sem tratamento, cada uma rodaria o agente (com ferramentas!) e viraria "episódio". Agora respondem sem agente e sem gravar.
4. **Promover sozinho à noite com feedback ruidoso é perigoso.** Padrão: candidatos ficam **PENDENTES** até `/approve N`. `--auto-promote` existe, mas não é recomendado.
5. **Forjar ferramentas durante o pedido do usuário é um vetor de prompt injection** (uma página web poderia induzir a criação de uma macro persistente). Macros só são forjadas offline, na consolidação, com teste + lint + juiz + aprovação.
6. **Memória poisoning:** o agente NÃO tem ferramenta para gravar memória. Lições só entram via consolidação, com crítico (bloqueia "ignore previous", sandbox, sudo, URLs, tokens…).
7. **`rm -rf` no workspace montado destrói arquivos reais** (o contêiner é descartável, o volume não). Há snapshot antes de cada turno e `/undo` (limite 500 MB; acima disso, sem snapshot e o usuário é avisado).
8. **O SAFETY_PREAMBLE** (tratar saída de ferramenta como dado não confiável etc.) é prefixado fora do harness: a evolução não pode editá-lo.
9. "Evolução brutal do OpenHands/Cline", "estado da arte": **não há evidência**. Nada aqui foi comparado a eles nem rodou com um LLM real.

## Limites que você deve conhecer
* **Zero LLM real testado.** `AnthropicLLM.chat` (tool use multi-turno com imagens) foi escrito conforme a API, nunca executado. Os testes usam `RuleLLM`, um duplo com um bug fixo ("afirma sucesso sem criar o arquivo") que uma lição conserta. Isso valida encanamento, não aprendizado.
* **Feedback/juiz são sinais fracos.** O `LLMJudge` pode estar errado ou ser enganado; o usuário pode reclamar do que não é culpa do agente. Por isso: gates pareados do Raven (lift ≥ 0,2, z ≥ 1,28), guarda de não regressão em turnos positivos, mínimo de 3 falhas reproduzíveis, aprovação humana. Com n pequeno (3–14 turnos) o z tem pouco poder estatístico.
* **Reprodução é parcial:** o replay roda num workspace restaurado do snapshot do turno, **sem rede** e sem GUI. Turnos que usaram rede/GUI são excluídos da evidência — justamente as tarefas "de mundo real" mais interessantes ficam sem evolução automática.
* **Isolamento:** `NamespaceEnv` (nível 1) compartilha o kernel e **não tem seccomp** (bash precisa de fork/exec); sem cgroups, fork-bomb só é contida pelo timeout + pid namespace. Docker comum é nível 2; gVisor (`--runtime runsc`) nível 3; nada disso foi executado aqui. Com `--docker-network bridge` o agente tem internet: exfiltração e prompt injection viram o risco principal — não há allowlist de domínio.
* **GUI no host não é suportada de propósito** (PyAutoGUI no seu desktop real daria ao modelo controle total do seu computador). A GUI roda num display virtual dentro do ambiente isolado.
* **Concorrência:** um turno por vez (lock global). Conversas são identificadas pelo hash da 1ª mensagem do usuário (duas conversas que começam igual colidem).
* **Servidor stdlib** (não FastAPI) por coerência com "zero dependências": sem HTTP/2, sem backpressure; adequado para uso local/pessoal.
* Memória lexical por padrão; com ≤ 12 lições todas entram no prompt (regras de feedback raramente casam por palavra com o pedido). Para muitas lições, use embeddings reais (`--memory http|st`, não testados).
