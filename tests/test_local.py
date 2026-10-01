import json
import tempfile
import threading
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from evoloop.llm import OpenAICompatibleLLM
from evoloop.server import AppState, serve
from evoloop.testing import RuleLLM


class LocalProvider(unittest.TestCase):
    def test_openai_tool_call_round_trip(self):
        seen = {}

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                n = int(self.headers["Content-Length"])
                seen.update(json.loads(self.rfile.read(n)))
                body = json.dumps({
                    "choices": [{"message": {"content": "", "tool_calls": [{
                        "id": "c1", "type": "function",
                        "function": {"name": "read_file", "arguments": '{"path":"README.md"}'},
                    }]}, "finish_reason": "tool_calls"}],
                    "usage": {"total_tokens": 12},
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            llm = OpenAICompatibleLLM("test", f"http://127.0.0.1:{server.server_port}/v1")
            result = llm.chat("safe", [{"role": "user", "content": "leia"}], [{
                "name": "read_file", "description": "read", "input_schema": {"type": "object"},
            }])
            self.assertEqual(result.tool_uses[0]["input"], {"path": "README.md"})
            self.assertEqual(result.tokens, 12)
            self.assertEqual(seen["tools"][0]["type"], "function")
        finally:
            server.shutdown()
            server.server_close()


class BundledInterface(unittest.TestCase):
    def test_root_and_health_are_local_bootstrap_endpoints(self):
        class FakeEnv:
            def describe(self):
                return {"name": "test", "level": 0}

        with tempfile.TemporaryDirectory() as directory:
            app = AppState(directory + "/state", directory + "/workspace", FakeEnv(),
                           RuleLLM(), "secret", daemon=False)
            server = serve(app, port=0, block=False)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            try:
                root = urllib.request.urlopen(f"http://127.0.0.1:{server.server_port}/").read().decode()
                health = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{server.server_port}/healthz").read())
                self.assertIn("Assistente local e evolutivo", root)
                self.assertEqual(health, {"ok": True, "service": "evoloop"})
            finally:
                server.shutdown()
                server.server_close()


if __name__ == "__main__":
    unittest.main()
