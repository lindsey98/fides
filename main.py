# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""FIDES prompt-injection defense baseline -- AgentDojo / AgentDyn entry point.

Run FIDES (information-flow-control planner with dynamic taint-tracking + policies) as a
defense over the AgentDojo benchmark, with the same CLI as the sibling baselines:

    # FIDES defense, under attack (reports utility + security):
    python main.py MODEL --run-attack --attack important_instructions \
        --suite banking slack travel workspace --defense fides

    # Undefended original model (baseline):
    python main.py MODEL --run-attack --attack important_instructions \
        --suite banking slack travel workspace --defense none

    # Benign utility only (no attack):
    python main.py MODEL --suite banking slack travel workspace --defense fides

MODEL is positional: a bare cloud name (gpt-*, gemini-*), ``anthropic:<name>`` / ``claude*``,
an ``org/model`` OpenRouter name, or ``local:<name>`` / a bare served name for a local
OpenAI-compatible server. The AgentDyn suites (shopping / github / dailylife) require the
AgentDyn / lindsey98 agentdojo fork to be installed (see README).
"""

import argparse
import logging as pylogging
import os
from pathlib import Path

from fides_defense.models import build_llm
from fides_defense.pipeline import make_fides_pipeline
from fides_defense.policies import POLICIES, make_policy

DEFAULT_SUITES = ["banking", "slack", "travel", "workspace"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FIDES IFC defense over AgentDojo / AgentDyn.")
    p.add_argument("model", type=str, help="Model name, e.g. gpt-4o-2024-08-06, anthropic:claude-sonnet-4-5-20250929, local:Qwen3-30B-A3B-Instruct-2507.")
    p.add_argument("--suite", "-s", dest="suites", nargs="+", default=DEFAULT_SUITES,
                   help="Suites to run (space-separated). Available: banking slack travel workspace shopping github dailylife.")
    p.add_argument("--benchmark-version", dest="benchmark_version", type=str, default="v1.2", help="AgentDojo benchmark version.")
    p.add_argument("--run-attack", dest="run_attack", action="store_true", help="Run under attack (omit for the benign/no-attack utility setting).")
    p.add_argument("--attack", type=str, default="important_instructions",
                   help="Attack name (any registered AgentDojo attack), e.g. important_instructions, tool_knowledge, ignore_previous, direct, dos, chat_inject_qwen3.")
    p.add_argument("--defense", type=str, default="fides", choices=["fides", "none"],
                   help="'fides' (IFC taint-tracking + policy) or 'none' (undefended original model).")
    p.add_argument("--policy", type=str, default="no_untrusted_sink", choices=list(POLICIES),
                   help="FIDES policy: no_untrusted_sink (default; block any sink once context is untrusted), "
                        "no_untrusted_url (block a sink only when untrusted context carries a URL -- tutorial's policy_no_untrusted_url), "
                        "or none (taint-track only, never block).")
    p.add_argument("--user-task", "-ut", dest="user_tasks", type=str, nargs="*", default=None, help="User task ids/numbers to run (default: all).")
    p.add_argument("--injection-task", "-it", dest="injection_tasks", type=str, nargs="*", default=None, help="Injection task ids/numbers to run (default: all).")
    p.add_argument("--force-rerun", "-f", dest="force_rerun", action="store_true", help="Re-run tasks even if a cached result exists.")
    p.add_argument("--max-iters", dest="max_iters", type=int, default=15, help="Max tool-loop iterations.")
    p.add_argument("--logdir", type=str, default="./logs", help="Directory for AgentDojo result logs.")
    return p.parse_args()


def _write_summary(logdir, args, suite_name, utility, security):
    """Write a compact per-(model,defense,policy,attack,suite) summary JSON so
    print_results.py can aggregate without re-scoring (AgentDojo's own per-task logs
    do not record utility/security)."""
    import json

    attack = args.attack if args.run_attack else "none"
    model_key = args.model.replace("/", "_").replace(":", "_")
    out_dir = logdir / "_fides_summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": args.model,
        "defense": args.defense,
        "policy": args.policy if args.defense == "fides" else None,
        "attack": attack,
        "suite": suite_name,
        "n_tasks": len(utility),
        "utility": (sum(utility) / len(utility)) if utility else None,
        "security_asr": (sum(security) / len(security)) if security else None,
    }
    fname = f"{model_key}__{args.defense}__{args.policy}__{attack}__{suite_name}.json"
    with open(out_dir / fname, "w") as f:
        json.dump(payload, f, indent=2)


def _norm_ids(ids, prefix):
    if ids is None:
        return None
    out = []
    for t in ids:
        t = str(t)
        out.append(t if t.startswith(prefix) else f"{prefix}{t}")
    return out


def main() -> None:
    args = parse_args()

    # AgentDojo's OutputLogger emits the chat trace via logging.info but installs no handler
    # when the benchmark is driven directly (as here); add one so the trace is visible.
    if not pylogging.getLogger().handlers:
        try:
            from rich.logging import RichHandler

            pylogging.basicConfig(level=os.getenv("FIDES_LOG_LEVEL", "INFO").upper(), format="%(message)s",
                                  datefmt="[%X]", handlers=[RichHandler(markup=True, show_path=False, rich_tracebacks=True)])
        except Exception:
            pylogging.basicConfig(level=os.getenv("FIDES_LOG_LEVEL", "INFO").upper())

    from agentdojo import attacks, benchmark, logging
    from agentdojo.task_suite import get_suite
    import agentdojo.attacks  # noqa: F401 -- ensure the attack registry (incl. fork attacks) is populated

    llm = build_llm(args.model)
    logdir = Path(args.logdir)
    user_tasks = _norm_ids(args.user_tasks, "user_task_")
    injection_tasks = _norm_ids(args.injection_tasks, "injection_task_")

    total_utility, total_security = [], []
    for suite_name in args.suites:
        policy = make_policy(args.policy, suite_name)
        pipeline = make_fides_pipeline(
            llm, args.model, suite_name, policy, args.defense, max_iters=args.max_iters
        )
        # The pipeline keeps the real model name, so result logs are separated per model.
        # AgentDojo's attacks address the victim model when the id is recognized and fall
        # back to a generic name otherwise.
        suite = get_suite(args.benchmark_version, suite_name)

        with logging.OutputLogger(str(logdir)):
            if args.run_attack:
                attack = attacks.load_attack(args.attack, suite, pipeline)
                results = benchmark.benchmark_suite_with_injections(
                    pipeline, suite, attack, logdir, force_rerun=args.force_rerun,
                    user_tasks=user_tasks, injection_tasks=injection_tasks,
                    benchmark_version=args.benchmark_version,
                )
            else:
                results = benchmark.benchmark_suite_without_injections(
                    pipeline, suite, logdir, force_rerun=args.force_rerun,
                    user_tasks=user_tasks, benchmark_version=args.benchmark_version,
                )

        utility = list(results["utility_results"].values())
        total_utility += utility
        u = sum(utility) / len(utility) if utility else float("nan")
        print(f"[{suite_name}] defense={args.defense} policy={args.policy} utility={u:.3f}", flush=True)
        security = None
        if args.run_attack:
            security = list(results["security_results"].values())
            total_security += security
            s = sum(security) / len(security) if security else float("nan")
            print(f"[{suite_name}] attack={args.attack} security(ASR)={s:.3f}", flush=True)

        _write_summary(logdir, args, suite_name, utility, security)

    if total_utility:
        print(f"\noverall - utility: {sum(total_utility) / len(total_utility):.3f}")
    if args.run_attack and total_security:
        print(f"overall - security (ASR): {sum(total_security) / len(total_security):.3f}")


if __name__ == "__main__":
    main()
