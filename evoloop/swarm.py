"""Multi-agente: orquestrador + sub-agentes especialistas (evoluíveis, tag `subagent` do RRSI) + revisor QA independente.
Esquemas de ferramenta/delegação no formato de function calling nativo (name/description/input_schema)."""
from __future__ import annotations
import re

from .llm import LLM, extract_json

ARCHITECT_SYSTEM = ("You design ONE specialist sub-agent for an orchestrator. Reply with ONE JSON object: name (snake_case), description "
                    "(what it handles; quote the task definitions verbatim), system_prompt (narrow, <=1200 chars, definitions of ONLY the "
                    "tasks it handles), tools (list of existing tool names it may use, may be empty).")
ORCH_SYSTEM = "You are the orchestrator. Delegate the task to the specialist whose description matches it; if none matches, do not delegate."
QA_SYSTEM = "You are an independent reviewer. Re-derive the answer yourself, then reply OK if it matches the given ANSWER, else 'REJECT: <why>'."


def tool_schema(name: str, tool: dict) -> dict:
    """Esquema JSON completo (type/properties/required) a partir do spec que o Forger já produz."""
    props = (tool["spec"].get("input_schema") or {}).get("properties") or {"text": {"type": "string"}}
    return {"name": name, "description": tool["spec"].get("description", ""),
            "input_schema": {"type": "object", "properties": props, "required": sorted(props)}}


def validate_args(schema: dict, args) -> list[str]:
    if not isinstance(args, dict):
        return ["argumentos não são objeto"]
    sch, errs = schema["input_schema"], []
    for k in sch.get("required", []):
        if k not in args:
            errs.append(f"falta argumento obrigatório: {k}")
    for k, v in args.items():
        t = sch["properties"].get(k, {}).get("type")
        if k not in sch["properties"]:
            errs.append(f"argumento desconhecido: {k}")
        elif t == "string" and not isinstance(v, str):
            errs.append(f"{k} deveria ser string")
    return errs


def delegate_schemas(subagents: dict) -> list[dict]:
    return [{"name": "delegate_" + n, "description": a["description"],
             "input_schema": {"type": "object", "properties": {"task": {"type": "string"}}, "required": ["task"]}}
            for n, a in sorted(subagents.items())]


def validate_subagent(spec: dict, tools: dict, existing: dict) -> list[str]:
    errs = []
    name = str(spec.get("name", ""))
    if not re.fullmatch(r"[a-z][a-z0-9_]{2,40}", name):
        errs.append("nome inválido (snake_case 3-41)")
    if name in existing:
        errs.append("nome já existe")
    if not str(spec.get("description", "")).strip():
        errs.append("descrição vazia")
    sp = str(spec.get("system_prompt", ""))
    if not sp.strip() or len(sp) > 1200:
        errs.append("system_prompt vazio ou > 1200 chars")
    bad = [t for t in spec.get("tools", []) if t not in tools]
    if bad:
        errs.append(f"ferramentas inexistentes: {bad}")
    return errs


def propose_subagent(llm: LLM, weak_instructions: list[str], tools: dict, existing: dict) -> tuple[dict | None, list[str]]:
    import json
    try:
        txt = llm.complete("architect", ARCHITECT_SYSTEM, "### REQUEST\n" + json.dumps({"weak": weak_instructions, "tools": sorted(tools)})).text
        spec = extract_json(txt)
    except Exception as e:                              # noqa: BLE001
        return None, [f"arquiteto falhou: {str(e)[:80]}"]
    errs = validate_subagent(spec, tools, existing)
    return (None, errs) if errs else ({k: spec[k] for k in ("description", "system_prompt", "tools")} | {"name": spec["name"]}, [])
