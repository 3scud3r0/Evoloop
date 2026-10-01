"""Núcleo puro do RRSI (google-research/rrsi, Apache-2.0): seleção regularizada, orçamento L0 anelado, componentes K."""
from .config import RRSIConfig
from .components import K, K_STR, normalize, novelty
from .evaluate import EvalResult, TaskResult, aggregate, relative_cost_change
from .schedule import edit_budget, budget_table
from .selection import Candidate, Decision, judge, select_round
