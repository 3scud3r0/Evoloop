"""Ator + sono. O ator (servidor) lê o campeão vigente a cada pedido; a thread de sono só ESCREVE candidatos pendentes (ou promove, se
auto_promote). Dispara quando ocioso há `idle_s` E dentro da janela horária (se definida) E há feedback novo suficiente."""
from __future__ import annotations
import threading
import time
from datetime import datetime


class SleepDaemon:
    def __init__(self, consolidator, get_last_activity, busy, idle_s: float = 1800, window: tuple[int, int] | None = (2, 6),
                 min_new: int = 2, poll_s: float = 30):
        self.c, self.last, self.busy = consolidator, get_last_activity, busy
        self.idle_s, self.window, self.min_new, self.poll_s = idle_s, window, min_new, poll_s
        self.last_run = 0.0
        self.reports: list = []
        self._stop = threading.Event()
        self.thread: threading.Thread | None = None

    def in_window(self, now: datetime | None = None) -> bool:
        if not self.window:
            return True
        h = (now or datetime.now()).hour
        a, b = self.window
        return a <= h < b if a <= b else (h >= a or h < b)

    def due(self, now: float | None = None, hour_now: datetime | None = None) -> bool:
        now = now or time.time()
        if self.busy() or now - self.last() < self.idle_s or not self.in_window(hour_now):
            return False
        new = self.c.store.query("score IS NOT NULL AND ts>=?", (self.last_run,), 500)
        return len(new) >= self.min_new

    def run_once(self):
        rep = self.c.run(self.last_run)
        self.last_run = time.time()
        self.reports.append(rep)
        return rep

    def _loop(self):
        while not self._stop.wait(self.poll_s):
            try:
                if self.due():
                    self.run_once()
            except Exception as e:                      # noqa: BLE001
                self.reports.append(type("R", (), {"note": f"erro: {e}", "as_dict": lambda s: {"note": s.note}})())

    def start(self):
        self.thread = threading.Thread(target=self._loop, daemon=True, name="evoloop-sleep")
        self.thread.start()

    def stop(self):
        self._stop.set()
