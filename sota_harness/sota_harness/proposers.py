"""Proposers: sources of candidate programs.

The paper's only proposer is the LLM (ideation + evolutionary rewrite).
This harness adds deterministic proposers so the loop can run LLM-free:

  - TemplateProposer: small hand-written architectural seeds (the paper's
    "founding directions" — Fermat chain, lookup, tower field — encoded
    as starting programs the rewriters then optimize).
  - AlgebraicProposer: ANF/span synthesis (analysis.py + synthesis.py) —
    proposes the multiplicative-complexity-optimal construction when the
    span search closes, with an optimality certificate attached.
  - LLMProposer: calls an LLM (paper-faithful ideation/rewrite); optional,
    enabled by providing a callable that maps prompt -> program source.

Every proposer honors the same interface: propose(archive, feedback) -> Program.
"""

from __future__ import annotations

from typing import Callable, Optional

from .circuit import Program


class Proposer:
    """Interface: propose(archive, feedback) -> Program (or raise)."""

    family: str = "generic"

    def propose(self, archive=None, feedback=None) -> Program:
        raise NotImplementedError


class TemplateProposer(Proposer):
    """Yields seeded architectural templates in order (round-robin after
    exhaustion), mirroring the paper's ideation step."""

    def __init__(self, templates: dict[str, str], family: str = "template"):
        # templates: {name: program source}
        self.templates = dict(templates)
        self.family = family
        self._order = list(templates)
        self._i = 0

    def propose(self, archive=None, feedback=None) -> Program:
        name = self._order[self._i % len(self._order)]
        self._i += 1
        return Program(
            source=self.templates[name],
            family=self.family,
            notes=[f"TEMPLATE: {name}"],
        )


class LLMProposer(Proposer):
    """Paper-faithful proposer: an LLM writes the whole program.

    ``llm_fn(prompt: str) -> str`` maps the composed prompt (task spec +
    parent + feedback + archive context) to program source. The harness
    itself stays model-agnostic: plug in any callable.
    """

    def __init__(self, llm_fn: Callable[[str], str], prompt_builder,
                 family: str = "llm"):
        self.llm_fn = llm_fn
        self.prompt_builder = prompt_builder  # (task, archive, feedback) -> str
        self.family = family

    def propose(self, archive=None, feedback=None) -> Program:
        prompt = self.prompt_builder(archive=archive, feedback=feedback)
        src = self.llm_fn(prompt)
        parent = None
        if feedback and "parent_source" in feedback:
            parent = feedback.get("parent_id")
        return Program(source=src, family=self.family, parent=parent)


class CompositeProposer(Proposer):
    """Round-robin over several proposers (the paper merges families; here
    each proposer family keeps its own lineage in the archive)."""

    def __init__(self, proposers: list[Proposer]):
        self.proposers = proposers
        self._i = 0

    def propose(self, archive=None, feedback=None) -> Program:
        p = self.proposers[self._i % len(self.proposers)]
        self._i += 1
        return p.propose(archive, feedback)
