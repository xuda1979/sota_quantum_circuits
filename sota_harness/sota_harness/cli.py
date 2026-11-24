"""Command-line interface for the sota_harness evolutionary search.

Usage:
  python -m sota_harness run-f24 [--iters N] [--log PATH] [--no-lb]
      Run the F24-inversion evolutionary loop (paper's toy demonstration).
  python -m sota_harness eval-f24 <program.py>
      Evaluate one candidate program against the F24 task.
  python -m sota_harness run-adder --n {3,4} [--iters N] [--log PATH]
      [--snapshot PATH] [--no-llm] [--no-template]
      Evolutionary loop on the in-place adder task: LLM agent
      (deepseek_v4 proxy, repair-feedback loop) plus CDKM seed templates;
      --no-template isolates the LLM, --no-llm isolates the templates.
  python -m sota_harness eval-adder --n {3,4} <program.py>
      Evaluate one candidate circuit against the adder task.
  python -m sota_harness certify
      Print the recorded F22/F23/F24 optimality certificates.
"""

from __future__ import annotations

import argparse
import sys

from .circuit import Program
from .evaluator import Evaluator
from .evolution import Evolution
from .mutators import (AncillaPackRewriter, MbuRewriter, MutationPipeline,
                       RedundantPairElimRewriter)
from .tasks_adder import adder_task
from .tasks_f24 import f24_task, f24_task_no_lb
from .templates_f24 import TEMPLATES
from .llm import default_llm_fn, make_prompt_builder
from .proposers import (CompositeProposer, LLMProposer,
                        TemplateProposer)


def cmd_run_f24(args) -> int:
    task = f24_task_no_lb() if args.no_lb else f24_task()
    ev = Evaluator(task)
    proposer = TemplateProposer(TEMPLATES, family="f24")
    pipe = MutationPipeline([MbuRewriter(), AncillaPackRewriter()])
    evo = Evolution(task, ev, proposer, mutators=pipe,
                    max_iters=args.iters, log_path=args.log)
    arch = evo.run(verbose=True)
    if arch.best is None:
        print("no valid circuit found")
        return 1
    v = arch.best.verdict
    print(f"\nbest: {arch.best.id} {v.one_line()}")
    if "toffoli_lower_bound" in v.meta:
        print(f"lower bound: {v.meta['toffoli_lower_bound']} "
              f"(certified optimal: {v.meta['certified_optimal']})")
    return 0


def cmd_eval_f24(args) -> int:
    task = f24_task()
    ev = Evaluator(task)
    with open(args.program) as f:
        src = f.read()
    v = ev.evaluate(src)
    print(v.one_line())
    return 0 if v.valid else 1


def cmd_eval_adder(args) -> int:
    task = adder_task(args.n)
    ev = Evaluator(task)
    with open(args.program) as f:
        src = f.read()
    v = ev.evaluate(src)
    print(v.one_line())
    return 0 if v.valid else 1


def cmd_run_adder(args) -> int:
    """Evolutionary loop on the in-place adder task (n in {3,4}).

    Proposer mix: LLMProposer (deepseek_v4 proxy via default_llm_fn) plus
    CDKM seed templates, so the loop can ideate from scratch AND refine.
    """
    from .archive import Archive
    from .templates_adder import cdkm_templates

    task = adder_task(args.n)
    ev = Evaluator(task)

    proposers: list = []
    if not args.no_llm:
        llm = LLMProposer(
            llm_fn=default_llm_fn(),
            prompt_builder=make_prompt_builder(task),
            family="llm",
        )
        proposers.append(llm)
        if not args.no_template:
            proposers.append(TemplateProposer(cdkm_templates(args.n),
                                              family="cdkm"))
    else:
        proposers.append(TemplateProposer(cdkm_templates(args.n),
                                          family="cdkm"))
    proposer = (proposers[0] if len(proposers) == 1
                else CompositeProposer(proposers))
    if args.mutators == "structural":
        pipe = MutationPipeline([MbuRewriter(), AncillaPackRewriter(),
                                 RedundantPairElimRewriter()])
    else:
        pipe = MutationPipeline([MbuRewriter(), AncillaPackRewriter()])

    arch = Archive()
    evo = Evolution(task, ev, proposer, mutators=pipe,
                    max_iters=args.iters, log_path=args.log,
                    archive=arch, dedup=not args.no_dedup)
    arch = evo.run(verbose=not args.quiet)
    if args.snapshot:
        arch.snapshot(args.snapshot)
    if arch.best is None:
        print("no valid circuit found")
        return 1
    v = arch.best.verdict
    print(f"\nbest: {arch.best.id} {v.one_line()}")
    if "toffoli_lower_bound" in v.meta:
        print(f"lower bound: {v.meta['toffoli_lower_bound']} "
              f"(certified optimal: {v.meta['certified_optimal']})")
    return 0


def cmd_certify(_args) -> int:
    """Print the F22/F23/F24 optimality certificates (cheap legs only;
    exhaustive legs' recorded node counts shown; re-run run_all.sh for
    the full searches)."""
    from .certificates import (f22_certificate, f23_certificate,
                               f24_certificate)
    for build, name in ((f22_certificate, "F22"), (f23_certificate, "F23"),
                        (f24_certificate, "F24")):
        cert = build() if build is not f24_certificate else build({}, {})
        print(f"== {name} ==")
        print(cert.conclusion())
        print(f"verify (cheap legs): {'OK' if cert.verify() else 'FAIL'}")
        print()
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="sota_harness")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run-f24", help="run the F24 evolutionary loop")
    r.add_argument("--iters", type=int, default=20)
    r.add_argument("--log", type=str, default=None)
    r.add_argument("--no-lb", action="store_true",
                   help="disable the Toffoli lower-bound certificate")
    r.set_defaults(fn=cmd_run_f24)

    e = sub.add_parser("eval-f24", help="evaluate one candidate program")
    e.add_argument("program")
    e.set_defaults(fn=cmd_eval_f24)

    ea = sub.add_parser("eval-adder",
                        help="evaluate one candidate against the in-place "
                             "adder task")
    ea.add_argument("--n", type=int, default=3, choices=[3, 4])
    ea.add_argument("program")
    ea.set_defaults(fn=cmd_eval_adder)

    ra = sub.add_parser("run-adder",
                        help="evolutionary loop on the in-place adder task")
    ra.add_argument("--n", type=int, default=3, choices=[3, 4])
    ra.add_argument("--iters", type=int, default=20)
    ra.add_argument("--log", type=str, default=None)
    ra.add_argument("--snapshot", type=str, default=None)
    ra.add_argument("--no-llm", action="store_true",
                    help="seed templates only (no LLM proposals)")
    ra.add_argument("--no-template", action="store_true",
                    help="LLM proposes only (no CDKM template seeds)")
    ra.add_argument("--mutators", choices=["base", "structural"],
                    default="base",
                    help="mutation operators: base (mbu + ancilla_pack) or "
                         "structural (adds pair_elim ripple-carry rewrite)")
    ra.add_argument("--no-dedup", action="store_true",
                    help="disable canonical-source dedup (re-evaluate "
                         "identical sources every iteration)")
    ra.add_argument("--quiet", action="store_true")
    ra.set_defaults(fn=cmd_run_adder)

    c = sub.add_parser("certify",
                       help="print the optimality certificates")
    c.set_defaults(fn=cmd_certify)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
