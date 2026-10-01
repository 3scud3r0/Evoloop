"""Coloca third_party/ no sys.path (gepa, rrsi_core, raven_gates são importados como pacotes de topo)."""
import sys
from pathlib import Path
_TP = str(Path(__file__).resolve().parent.parent / "third_party")
if _TP not in sys.path:
    sys.path.insert(0, _TP)
