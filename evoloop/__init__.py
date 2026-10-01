"""evoloop — harness evolutivo em ciclo fechado com LLM congelado.

Integra (código vendorizado, ver UPSTREAM.md): GEPA, núcleo RRSI, gates estatísticos do Raven;
porta para Python a lógica de forge do AgentOS; reimplementa (sem copiar) os mecanismos de
harness do agent-evolve (sem licença) e a disciplina memória-congelada/verificador-isolado do RSIAgent.
"""
from . import _vendor  # noqa: F401  (side effect: sys.path)
__version__ = "0.4.0"
