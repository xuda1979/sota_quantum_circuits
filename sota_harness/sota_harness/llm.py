"""LLM agent wiring for the evolutionary loop.

``default_llm_fn`` POSTs the composed prompt to the configured
deepseek_v4-compatible proxy (env: HUANXIN_DP4_BASE_URL/API_KEY/
UPSTREAM_MODEL) and returns the candidate program source. Model-agnostic:
any callable that maps prompt -> source works with LLMProposer.

``make_prompt_builder(task)`` returns the (archive, feedback) -> prompt
callable LLMProposer expects; it lays out the task spec, the harness
circuit-model contract, and the current archive context (best programs,
discoveries) plus any repair feedback on an invalid parent.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
import urllib.error
from typing import Callable  # noqa: F401  (re-exported for convenience)


def default_llm_fn(model: str | None = None) -> Callable[[str], str]:
    base = os.environ.get("HUANXIN_DP4_BASE_URL", "")
    key = os.environ.get("HUANXIN_DP4_API_KEY", "")
    upstream = model or os.environ.get("HUANXIN_DP4_UPSTREAM_MODEL", "")
    max_chars = int(os.environ.get("HUANXIN_DP4_MAX_INPUT_CHARS", "120000"))

    max_tokens = int(os.environ.get("HUANXIN_DP4_MAX_TOKENS", "4096"))

    def llm_fn(prompt: str) -> str:
        body = {
            "model": upstream,
            "messages": [{"role": "user", "content": prompt[:max_chars]}],
            "temperature": 0.4,
            "max_tokens": max_tokens,
            # deepseek_v4 on this proxy burns its whole budget on hidden
            # reasoning otherwise; disable it so content is emitted directly.
            "chat_template_kwargs": {"enable_thinking": False},
        }
        req = urllib.request.Request(
            base,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {key}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            payload = json.loads(resp.read().decode())
        msg = payload["choices"][0]["message"]
        content = msg.get("content")
        if not content:
            raise ValueError(
                "LLM reply carried no content (reasoning-only response): "
                f"finish={payload['choices'][0].get('finish_reason')}"
            )
        return _extract_build_source(content)

    return llm_fn


def _extract_build_source(text: str) -> str:
    """Pull the ``def build(ctx):`` block out of a possibly-verbose reply."""
    if "def build" not in text:
        raise ValueError("no def build(ctx) in LLM reply")
    # drop markdown fence if present, keep everything from def build on
    m = re.search(r"def build\(ctx\):", text)
    src = text[m.start():]
    src = src.replace("```python", "").replace("```", "")
    # truncate at the first blank-line-followed-by-non-indented line that
    # is clearly past the program (heuristic: next top-level statement)
    lines = src.split("\n")
    out, in_def = [], False
    for ln in lines:
        if ln.startswith("def build"):
            in_def = True
        if in_def:
            if (ln.strip() and not ln.startswith((" ", "\t"))
                    and not ln.startswith("def build")):
                break
            out.append(ln)
    return "\n".join(out).rstrip() + "\n"


def make_prompt_builder(task):
    """(archive, feedback) -> prompt, closing over the task."""
    MODEL_CONTRACT = """You synthesize python source that defines ONE function:

def build(ctx):
    # ctx API (the harness wraps an exact reversible-circuit simulator):
    #   a = ctx.alloc(3)          # allocate n qubits (list); ctx.alloc() -> 1
    #   ctx.reg("a", a)           # declare interface register (name must match)
    #   ctx.cx(a[i], b[j])        # CNOT (free Clifford)
    #   ctx.ccx(a[i], b[j], t)    # Toffoli (1 unit of cost each)
    #   ctx.x(q) / ctx.free(q)    # NOT / recycle
    # The evaluator checks the EXACT truth table over every input. Any
    # temporary qubit must return to |0> before free; global phase must be 0.
    # Reference addition: |a>|b> -> |a+b mod 2^n>|b>."""

    def pb(archive=None, feedback=None) -> str:
        parts = [
            "TASK SPEC (ground truth):",
            task.spec_text,
            "",
            "INTERFACE: registers = "
            + ", ".join(f"{r.name}[{r.n}]" for r in task.registers),
            "",
            MODEL_CONTRACT,
            "",
        ]
        if archive is not None and archive.best is not None:
            parts.append("ARCHIVE CONTEXT (best so far; beat it to win):")
            ctx = archive.context_for(archive.best.family, max_programs=3)
            for eid, line in ctx["elite"]:
                parts.append(f"  elite {eid}: {line}")
            if ctx["discoveries"]:
                parts.append("  discoveries: " + "; ".join(ctx["discoveries"][-4:]))
            parts.append(
                f"  best overall: {ctx['best_overall'][0]} "
                f"{ctx['best_overall'][1]}"
            )
            parts.append("")
        if feedback is not None and feedback.get("parent_verdict"):
            parts.append("PREVIOUS ATTEMPT — REPAIR THIS (fix the defect, "
                         "then improve):")
            parts.append(f"verdict: {feedback['parent_verdict']}")
            parts.append("source was:")
            parts.append(feedback.get("parent_source", ""))
            parts.append("")
        parts.append(
            "Return ONLY the python source defining build(ctx). "
            "Do not explain. Do not use measurement (hmr) unless needed."
        )
        return "\n".join(parts)

    return pb
