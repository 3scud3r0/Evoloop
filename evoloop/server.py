"""Servidor compatível com a API de chat da OpenAI (stdlib, zero dependências) + endpoints /evoloop/* de administração.
Open WebUI / LibreChat / qualquer cliente OpenAI apontam para http://HOST:PORT/v1. Escuta em 127.0.0.1 por padrão e EXIGE Authorization: Bearer <chave>:
este servidor executa comandos — nunca exponha sem chave/firewall."""
from __future__ import annotations
import hashlib
import hmac
import json
import queue
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .consolidate import Consolidator, LLMJudge
from .daemon import SleepDaemon
from .embed import CachedEmbedder, make_embedder
from .env import ExecEnv
from .general import GeneralAgent, general_seed_harness
from .memory import Memory
from .registry import Registry
from .store import TurnStore, Workspace

NEG = re.compile(r"\b(errado|errou|erro|incorreto|n[ãa]o era isso|n[ãa]o funcionou|n[ãa]o deu certo|de novo|wrong|incorrect|that'?s not|didn'?t work|not what i)\b", re.I)
POS = re.compile(r"\b(obrigad[oa]|valeu|perfeito|[óo]timo|funcionou|isso mesmo|thanks|thank you|perfect|great|works)\b", re.I)
HELP = ("Comandos: /bad [motivo] · /good · /undo · /status · /pending · /approve N · /decline N · /consolidate")


def classify_feedback(text: str) -> tuple[float | None, float]:
    """Heurística grosseira (PT/EN) -> (score, peso). Peso baixo: é sinal implícito e ruidoso; só mensagens curtas."""
    t = text.strip()
    if len(t) > 280:
        return None, 0.0
    neg, pos = bool(NEG.search(t)), bool(POS.search(t))
    if neg and not pos:
        return 0.0, 0.6
    if pos and not neg:
        return 1.0, 0.5
    return None, 0.0


class AppState:
    def __init__(self, state_dir: str, workspace: str, env: ExecEnv, llm, api_key: str, *, memory_kind: str = "lexical", judge=None, auto_promote=False,
                 idle_s=1800, window=(2, 6), gui=False, use_gepa=False, daemon=True):
        self.dir = Path(state_dir); self.dir.mkdir(parents=True, exist_ok=True)
        self.api_key, self.llm, self.env = api_key, llm, env
        self.store = TurnStore(str(self.dir / "turns.db"))
        self.registry = Registry(str(self.dir / "registry.db"))
        emb = make_embedder(memory_kind)
        self.memory = Memory(str(self.dir / "memory.db"), embedder=CachedEmbedder(emb) if emb else None)
        self.workspace = Workspace(workspace, self.dir)
        if self.registry.champion() is None:
            seed = general_seed_harness(); seed.config["gui"] = gui
            self.registry.promote(self.registry.add(seed, None, "seed"), "seed")
        self.agent_lock = threading.Lock()
        self.busy = False
        self.last_activity = time.time()
        self.consolidator = Consolidator(self.store, self.registry, self.memory, llm, judge or LLMJudge(llm), self.dir / "snapshots",
                                         auto_promote=auto_promote, use_gepa=use_gepa, apply_lessons=self.apply_lessons)
        self.daemon = SleepDaemon(self.consolidator, lambda: self.last_activity, lambda: self.busy, idle_s=idle_s, window=window)
        self._daemon_on = daemon

    def apply_lessons(self, lessons):
        for l in lessons:
            self.memory.remember("lesson", hashlib.sha256(l.encode()).hexdigest()[:12], l)

    def champion(self):
        v = self.registry.champion()
        return self.registry.get(v), v

    def status(self) -> dict:
        h, v = self.champion()
        return {"champion_version": v, "champion_note": h.note, "tools": sorted(h.tools), "subagents": sorted(h.subagents), "lessons": len(self.memory),
                "turns": self.store.count(), "pending": [{"v": p["v"], "evidence": p["evidence"].get("candidate")} for p in self.registry.pending()],
                "environment": self.env.describe(), "last_nights": [r.as_dict() for r in self.daemon.reports[-3:]]}


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return ""


def conv_id(messages: list[dict]) -> str:
    first = next((_text_of(m.get("content")) for m in messages if m.get("role") == "user"), "")
    return hashlib.sha256(first.strip().encode()).hexdigest()[:16]


def handle_command(app: AppState, cmd: str, conv: str) -> str | None:
    parts = cmd.strip().split(None, 1)
    c, arg = parts[0].lower(), (parts[1] if len(parts) > 1 else "")
    last = (app.store.query("conv=?", (conv,), 1) or [None])[0]
    if c in ("/bad", "/good", "/👎", "/👍"):
        if not last:
            return "Não há resposta anterior nesta conversa para avaliar."
        good = c in ("/good", "/👍")
        app.store.set_feedback(last["id"], 1.0 if good else 0.0, arg, "explicit", 1.0)
        return ("Obrigado — registrado como bom." if good else "Registrado como ruim. Vou usar isso para aprender (sob sua aprovação).")
    if c == "/undo":
        if not last or not last.get("snapshot"):
            return "Não há snapshot para desfazer (workspace grande demais ou sem turno anterior)."
        return "Workspace restaurado ao estado anterior à última resposta." if app.workspace.restore(last["snapshot"]) else "Falha ao restaurar o snapshot."
    if c == "/status":
        return "```json\n" + json.dumps(app.status(), ensure_ascii=False, indent=1, default=str) + "\n```"
    if c == "/pending":
        p = app.registry.pending()
        return "\n".join(f"v{x['v']}: {x['evidence'].get('candidate', {}).get('what')} (lift {x['evidence'].get('candidate', {}).get('lift')})" for x in p) or "Nenhuma mudança pendente."
    if c in ("/approve", "/decline"):
        ok = (app.consolidator.approve if c == "/approve" else app.consolidator.decline)(int(arg or 0)) if arg.strip().isdigit() else False
        return ("Feito." if ok else "Versão pendente não encontrada.")
    if c == "/consolidate":
        return "```json\n" + json.dumps(app.daemon.run_once().as_dict(), ensure_ascii=False, indent=1, default=str) + "\n```"
    if c == "/help":
        return HELP
    return None


def run_turn(app: AppState, messages: list[dict], on_event=None) -> tuple[str, dict]:
    """Executa um turno. Devolve (texto, meta). Comandos / e requisições auxiliares não rodam o agente."""
    app.last_activity = time.time()
    user_text = next((_text_of(m.get("content")) for m in reversed(messages) if m.get("role") == "user"), "")
    conv = conv_id(messages)
    if user_text.lstrip().startswith("/"):
        out = handle_command(app, user_text.lstrip(), conv)
        if out is not None:
            return out, {"tokens": 0, "command": True}
    last = (app.store.query("conv=?", (conv,), 1) or [None])[0]
    if last and last["score"] is None:                       # a nova mensagem pode ser feedback implícito sobre a resposta anterior
        s, w = classify_feedback(user_text)
        if s is not None:
            app.store.set_feedback(last["id"], s, user_text[:300], "implicit", w)
    with app.agent_lock:
        app.busy = True
        try:
            champ, v = app.champion()
            snap = app.workspace.snapshot("turn")
            hist = [{"role": m["role"], "content": _text_of(m.get("content"))} for m in messages if m.get("role") in ("user", "assistant")]
            res = GeneralAgent(champ, app.llm, app.env, app.memory).run(hist, on_event)
            text = res.text + ("" if snap else "\n\n⚠ Workspace grande demais: sem snapshot para /undo neste turno.")
            tid = app.store.add_turn(conv, champ.digest(), v, user_text, text, res.transcript, res.steps, res.tokens, res.status, snap, hist)
        finally:
            app.busy = False
            app.last_activity = time.time()
    return text, {"tokens": res.tokens, "turn_id": tid, "status": res.status, "steps": res.steps}


def _event_md(kind: str, d) -> str:
    if kind == "tool_use":
        arg = d["input"].get("command") or d["input"].get("path") or d["input"].get("action") or d["input"].get("task") or ""
        return f"\n> 🔧 `{d['name']}` {str(arg)[:120]!r}\n"
    if kind == "tool_result":
        return f"> {'❌' if d['error'] else '↳'} {d['output'][:160].strip()!r}\n\n"
    return ""


def make_handler(app: AppState):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _send(self, code: int, obj, ctype="application/json"):
            body = json.dumps(obj, ensure_ascii=False).encode() if not isinstance(obj, bytes) else obj
            self.send_response(code)
            self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body)))
            self.end_headers(); self.wfile.write(body)

        def _err(self, code, msg, typ="invalid_request_error"):
            self._send(code, {"error": {"message": msg, "type": typ, "code": code}})

        def _auth(self) -> bool:
            tok = self.headers.get("Authorization", "").removeprefix("Bearer ").strip()
            if hmac.compare_digest(tok.encode(), app.api_key.encode()):
                return True
            self._err(401, "chave de API inválida", "authentication_error")
            return False

        def _body(self):
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}")

        def do_GET(self):
            if self.path.split("?", 1)[0] == "/":
                body = (Path(__file__).parent / "web" / "index.html").read_bytes()
                self._send(200, body, "text/html; charset=utf-8")
                return
            if self.path == "/healthz":
                self._send(200, {"ok": True, "service": "evoloop"})
                return
            if not self._auth():
                return
            if self.path == "/v1/models":
                self._send(200, {"object": "list", "data": [{"id": "evoloop", "object": "model", "created": 0, "owned_by": "evoloop"}]})
            elif self.path == "/evoloop/status":
                self._send(200, app.status())
            elif self.path == "/evoloop/pending":
                self._send(200, app.registry.pending())
            else:
                self._err(404, "não encontrado")

        def do_POST(self):
            if not self._auth():
                return
            try:
                b = self._body()
            except Exception:                           # noqa: BLE001
                return self._err(400, "JSON inválido")
            if self.path == "/v1/chat/completions":
                return self._chat(b)
            if self.path == "/evoloop/feedback":
                tid = b.get("turn_id")
                if tid is None and b.get("conv"):
                    last = (app.store.query("conv=?", (b["conv"],), 1) or [None])[0]
                    tid = last and last["id"]
                if tid is None or app.store.get(int(tid)) is None:
                    return self._err(404, "turno não encontrado")
                app.store.set_feedback(int(tid), float(b["score"]), str(b.get("text", ""))[:500], "explicit", 1.0)
                return self._send(200, {"ok": True})
            if self.path in ("/evoloop/approve", "/evoloop/decline"):
                f = app.consolidator.approve if self.path.endswith("approve") else app.consolidator.decline
                return self._send(200, {"ok": bool(f(int(b.get("v", 0))))})
            if self.path == "/evoloop/consolidate":
                return self._send(200, app.daemon.run_once().as_dict())
            self._err(404, "não encontrado")

        def _chat(self, b):
            msgs = b.get("messages")
            if not isinstance(msgs, list) or not msgs:
                return self._err(400, "messages ausente")
            model, stream = b.get("model", "evoloop"), bool(b.get("stream"))
            cid, created = "chatcmpl-" + uuid.uuid4().hex[:16], int(time.time())
            # pedidos auxiliares do Open WebUI (título/tags/sugestões) usam o mesmo modelo: respondem sem agente, sem gravar, sem snapshot
            if any(_text_of(m.get("content")).lstrip().startswith("### Task:") for m in msgs):
                prompt = "\n".join(_text_of(m.get("content")) for m in msgs)
                text = app.llm.complete("aux", "Follow the task exactly and reply concisely.", prompt).text
                return self._reply(cid, created, model, stream, text, 0, None)
            if stream:
                q: queue.Queue = queue.Queue()
                out: dict = {}
                def work():
                    try:
                        t, meta = run_turn(app, msgs, lambda k, d: q.put(("ev", _event_md(k, d))))
                        out.update(t=t, m=meta)
                    except Exception as e:              # noqa: BLE001
                        out.update(t=f"Erro interno: {str(e)[:200]}", m={"tokens": 0})
                    q.put(("done", None))
                threading.Thread(target=work, daemon=True).start()
                self._sse_start(cid, created, model)
                while True:
                    k, d = q.get()
                    if k == "done":
                        break
                    if d:
                        self._sse_chunk(cid, created, model, d)
                self._sse_chunk(cid, created, model, out["t"])
                self._sse_end(cid, created, model)
                return
            text, meta = run_turn(app, msgs)
            self._reply(cid, created, model, False, text, meta.get("tokens", 0), meta)

        def _reply(self, cid, created, model, stream, text, tokens, meta):
            if stream:
                self._sse_start(cid, created, model); self._sse_chunk(cid, created, model, text); return self._sse_end(cid, created, model)
            self._send(200, {"id": cid, "object": "chat.completion", "created": created, "model": model,
                             "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                             "usage": {"prompt_tokens": 0, "completion_tokens": tokens, "total_tokens": tokens}, "evoloop": meta or {}})

        def _sse_start(self, cid, created, model):
            self.send_response(200)
            for k, v in (("Content-Type", "text/event-stream"), ("Cache-Control", "no-cache"), ("Connection", "close")):
                self.send_header(k, v)
            self.end_headers()
            self.close_connection = True
            self._w({"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
                     "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]})

        def _w(self, obj):
            self.wfile.write(b"data: " + json.dumps(obj, ensure_ascii=False).encode() + b"\n\n"); self.wfile.flush()

        def _sse_chunk(self, cid, created, model, text):
            self._w({"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
                     "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]})

        def _sse_end(self, cid, created, model):
            self._w({"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
                     "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
            self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()
    return H


def serve(app: AppState, host: str = "127.0.0.1", port: int = 8000, block: bool = True) -> ThreadingHTTPServer:
    srv = ThreadingHTTPServer((host, port), make_handler(app))
    srv.daemon_threads = True
    if app._daemon_on:
        app.daemon.start()
    if block:
        try:
            srv.serve_forever()
        finally:
            app.daemon.stop()
    return srv
