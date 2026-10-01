"""Gates estatísticos do Raven (Fisher exato unilateral + lift pareado 2σ). Apache-2.0."""
from .scoring import TaskEval
from .fisher import fisher_one_sided, focused_counts, train_mean
from .paired import PairedResult, paired_lift
__all__ = ["TaskEval", "fisher_one_sided", "focused_counts", "train_mean", "PairedResult", "paired_lift"]
