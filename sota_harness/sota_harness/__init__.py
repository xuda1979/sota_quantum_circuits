"""sota_harness: an evolutionary quantum-circuit discovery harness.

Improves on the framework of arXiv:2610.22367 ("Discovering state-of-the-art
quantum circuits with large language models"):

- Same core loop: propose circuit-generating programs -> independent
  evaluator verifies and scores -> archive selects (the evaluator is ground
  truth).
- Plus: deterministic algebraic proposers (ANF/span synthesis) alongside LLM
  proposers; Toffoli lower-bound certificates (multiplicative complexity);
  Pareto-frontier + MAP-Elites archiving with lineage and stagnation-driven
  cross-family migration; categorized failure diagnostics; reproducible
  seeded runs with full JSONL logging.
"""

__version__ = "0.1.0"

from .circuit import Program, run_program  # noqa: F401
from .task import Task, Register  # noqa: F401
from .evaluator import Evaluator, Verdict  # noqa: F401
from .archive import Archive  # noqa: F401
from .analysis import MonoSpace, multiplicative_complexity_lb  # noqa: F401
from .proposers import (Proposer, TemplateProposer, LLMProposer,  # noqa: F401
                        CompositeProposer)
from .mutators import (MbuRewriter, AncillaPackRewriter,  # noqa: F401
                       RedundantPairElimRewriter, MutationPipeline)
from .evolution import Evolution  # noqa: F401
from .tasks_f24 import f24_task  # noqa: F401
from .templates_f24 import TEMPLATES  # noqa: F401
