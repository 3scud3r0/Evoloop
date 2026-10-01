"""Saída de terminal neon (ciano/magenta/âmbar/lima). Desliga com NO_COLOR=1 ou quando não é TTY."""
import os
import sys

_ON = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
def _c(code):
    return (lambda s: f"\x1b[{code}m{s}\x1b[0m") if _ON else (lambda s: str(s))
cyan, magenta, amber, lime, dim = _c("96"), _c("95"), _c("93"), _c("92"), _c("2")
