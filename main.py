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

Results are written DRIFT-style, one JSON per task, under
``logs/<model>+fides/<suite>/user_task_<i>/<attack>/injection_task_<j>.json`` (and
``.../none/none.json`` for benign runs). The top-level dir is ``<model>+fides`` when the
FIDES defense is on, and the bare ``<model>`` for the undefended model -- mirroring DRIFT's
``<model>+drift`` / ``<model>`` convention. Aggregate with ``print_results.py``.

MODEL is positional: a bare cloud name (gpt-*, gemini-*), ``openai:<name>`` (custom OpenAI
endpoint), ``anthropic:<name>`` / ``claude*``, an ``org/model`` OpenRouter name, or
``local:<name>`` / a bare served name for a local OpenAI-compatible server.
"""

import argparse
import json
import logging as pylogging
import os
import re
import time
from pathlib import Path

from fides_defense.models import build_llm
from fides_defense.pipeline import make_fides_pipeline
from fides_defense.policies import POLICIES, make_policy

DEFAULT_SUITES = ["banking", "slack", "travel", "workspace"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FIDES IFC defense over AgentDojo / AgentDyn.")
    p.add_argument("model", type=str, help="Model name, e.g. gpt-4o-2024-08-06, openai:my-model, anthropic:claude-sonnet-4-5-20250929, local:Qwen3-30B-A3B-Instruct-2507.")
    p.add_argument("--suite", "-s", dest="suites", nargs="+", default=DEFAULT_SUITES,
                   help="Suites to run (space-separated). Available: banking slack travel workspace shopping github dailylife.")
    p.add_argument("--benchmark-version", dest="benchmark_version", type=str, default="v1.2", help="AgentDojo benchmark version.")
    p.add_argument("--run-attack", dest="run_attack", action="store_true", help="Run under attack (omit for the benign/no-attack utility setting).")
    p.add_argument("--attack", type=str, default="important_instructions",
                   help="Attack name (any registered AgentDojo attack), e.g. important_instructions, tool_knowledge, ignore_previous, direct, dos, chat_inject_qwen3.")
    p.add_argument("--defense", type=str, default="fides", choices=["fides", "none"],
                   help="'fides' (IFC taint-tracking + policy) or 'none' (undefended original model).")
    p.add_argument("--policy", type=str, default="no_untrusted_sink", choices=list(POLICIES),
                   help="FIDES policy: no_untrusted_sink (default), no_untrusted_url (tutorial's policy_no_untrusted_url), or none (taint-track only).")
    p.add_argument("--planner", type=str, default="full", choices=["full", "labeled_basic"],
                   help="FIDES planner (with --defense fides): 'full' = variable hiding + quarantined LLM + per-argument policy "
                        "(the paper's planner; needs an OpenAI-compatible endpoint); 'labeled_basic' = the tutorial's "
                        "LabeledBasicPlanner (context-level taint + policy; works with every provider).")
    p.add_argument("--user-task", "-ut", dest="user_tasks", type=str, nargs="*", default=None, help="User task ids/numbers to run (default: all).")
    p.add_argument("--injection-task", "-it", dest="injection_tasks", type=str, nargs="*", default=None, help="Injection task ids/numbers to run (default: all).")
    p.add_argument("--force-rerun", "-f", dest="force_rerun", action="store_true", help="Re-run tasks even if a cached result JSON exists.")
    p.add_argument("--max-iters", dest="max_iters", type=int, default=15, help="Max tool-loop iterations.")
    p.add_argument("--logdir", type=str, default="./logs", help="Directory for result logs.")
    return p.parse_args()


def model_dir_name(args) -> str:
    """Output sub-directory for a run: '<model>+fides' when the FIDES defense is on, the
    bare '<model>' otherwise (the undefended original model). Mirrors DRIFT's get_model_dir.

    The provider prefix (local:/openai:/anthropic:) is stripped and path-unsafe characters
    are sanitized, so `local:Qwen3.6-35B-A3B` -> `Qwen3.6-35B-A3B[+fides]`."""
    base = args.model
    for pre in ("local:", "openai:", "anthropic:"):
        if base.startswith(pre):
            base = base[len(pre):]
            break
    base = base.replace("/", "_").replace(":", "_")
    if args.defense != "fides":
        return base
    # The full (paper) planner is the default and keeps the plain '+fides' dir; the
    # tutorial's labeled_basic planner gets its own dir so the two don't collide.
    return f"{base}+fides" if args.planner == "full" else f"{base}+fides-basic"


def _norm_id(t, prefix):
    t = str(t)
    return t if t.startswith(prefix) else f"{prefix}{t}"


def _task_num(task_id: str) -> int:
    m = re.fullmatch(r"\w+_task_(\d+)", task_id)
    return int(m.group(1)) if m else -1


def _json_default(o):
    from pydantic import BaseModel

    if isinstance(o, BaseModel):
        return o.model_dump(mode="json")
    return str(o)


def _dump_record(record: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(record, f, indent=4, default=_json_default)


def _load_cached(path: Path):
    """Return (utility, security) from an existing result JSON, or None if absent/invalid."""
    if not path.exists():
        return None
    try:
        with open(path) as f:
            d = json.load(f)
        if "utility" in d and "security" in d:
            return d["utility"], d["security"]
    except Exception:
        return None
    return None


class _CaptureMessages:
    """Trailing pipeline element that records the final messages + FIDES events per task.

    AgentDojo's ``run_task_with_pipeline`` returns utility/security but not the message
    trace, so we snapshot it here (the pipeline runs elements in order, and this is last)."""

    name: str | None = None

    def __init__(self):
        self.messages = []
        self.fides_events = []

    def query(self, query, runtime, env, messages, extra_args={}):
        self.messages = list(messages)
        self.fides_events = list(extra_args.get("fides_events", []))
        return query, runtime, env, messages, extra_args


def main() -> None:
    args = parse_args()

    if not pylogging.getLogger().handlers:
        pylogging.basicConfig(level=os.getenv("FIDES_LOG_LEVEL", "INFO").upper(), format="%(message)s")

    from agentdojo import attacks
    from agentdojo.agent_pipeline import AgentPipeline
    from agentdojo.task_suite import get_suite
    import agentdojo.attacks  # noqa: F401 -- populate the attack registry (incl. fork attacks)

    use_full = args.defense == "fides" and args.planner == "full"
    if use_full:
        from fides_defense.models import build_raw_client
        from fides_defense.planner_full import FidesFullPipeline
        from agentdojo.agent_pipeline.agent_pipeline import load_system_message

        raw_client, raw_model = build_raw_client(args.model)
        base_sys = load_system_message(None)
        llm = None
    else:
        llm = build_llm(args.model)

    logdir = Path(args.logdir)
    mdir = model_dir_name(args)
    run_label = args.attack if args.run_attack else "none"

    want_user = set(_norm_id(t, "user_task_") for t in args.user_tasks) if args.user_tasks else None
    want_inj = set(_norm_id(t, "injection_task_") for t in args.injection_tasks) if args.injection_tasks else None

    total_utility, total_security = [], []

    for suite_name in args.suites:
        capture = _CaptureMessages()
        if use_full:
            inner = FidesFullPipeline(
                raw_client, raw_model, suite_name, policy_name=args.policy,
                max_iters=args.max_iters, base_system_message=base_sys,
            )
            pipeline = AgentPipeline([inner, capture])
        else:
            policy = make_policy(args.policy, suite_name)
            base_pipeline = make_fides_pipeline(llm, args.model, suite_name, policy, args.defense, max_iters=args.max_iters)
            pipeline = AgentPipeline([*base_pipeline.elements, capture])
        # Keep the pipeline name = real model so the attack addresses the right victim
        # (the fork degrades gracefully for unknown names).
        pipeline.name = args.model

        suite = get_suite(args.benchmark_version, suite_name)
        attack = attacks.load_attack(args.attack, suite, pipeline) if args.run_attack else None

        user_tasks = [t for tid, t in suite.user_tasks.items() if want_user is None or tid in want_user]
        if args.run_attack:
            injection_tasks = [t for tid, t in suite.injection_tasks.items() if want_inj is None or tid in want_inj]
        else:
            injection_tasks = [None]

        suite_u, suite_s = [], []
        for user_task in user_tasks:
            ut_n = _task_num(user_task.ID)
            for injection_task in injection_tasks:
                if injection_task is not None:
                    it_n = _task_num(injection_task.ID)
                    result_path = logdir / mdir / suite_name / f"user_task_{ut_n}" / run_label / f"injection_task_{it_n}.json"
                else:
                    result_path = logdir / mdir / suite_name / f"user_task_{ut_n}" / "none" / "none.json"

                cached = None if args.force_rerun else _load_cached(result_path)
                if cached is not None:
                    u, s = cached
                    suite_u.append(u); suite_s.append(s)
                    print(f"[{suite_name}] user_task_{ut_n}" + (f" x injection_task_{it_n}" if injection_task is not None else "") + f" cached: utility={u} security={s}", flush=True)
                    continue

                injections = attack.attack(user_task, injection_task) if (attack and injection_task is not None) else {}
                capture.messages, capture.fides_events = [], []
                t0 = time.time()
                err = None
                try:
                    utility, security, injection_happened, _, _ = suite.run_task_with_pipeline(
                        pipeline, user_task, injection_task, injections
                    )
                except Exception as e:
                    # A task-level failure (e.g. a model/API error) must not crash the whole
                    # run or leave empty logs: record it as a failed task and continue.
                    import traceback

                    utility, security, injection_happened = False, False, False
                    err = f"{type(e).__name__}: {e}"
                    print(f"[{suite_name}] user_task_{ut_n} ERROR: {err}", flush=True)
                    if os.getenv("FIDES_LOG_LEVEL", "").upper() == "DEBUG":
                        traceback.print_exc()
                duration = time.time() - t0
                suite_u.append(utility); suite_s.append(security)

                record = {
                    "suite_name": suite_name,
                    "pipeline_name": args.model,
                    "user_task_id": f"user_task_{ut_n}",
                    "injection_task_id": (f"injection_task_{it_n}" if injection_task is not None else None),
                    "attack_type": (args.attack if args.run_attack else None),
                    "injections": injections,
                    "defense": args.defense,
                    "planner": (args.planner if args.defense == "fides" else None),
                    "policy": (args.policy if args.defense == "fides" else None),
                    "fides_events": capture.fides_events,
                    "conversations": capture.messages,
                    "benchmark_version": args.benchmark_version,
                    "utility": utility,
                    "security": security,
                    "injection_happened": injection_happened,
                    "error": err,
                    "duration": duration,
                }
                _dump_record(record, result_path)
                tag = f" x injection_task_{it_n}" if injection_task is not None else ""
                print(f"[{suite_name}] user_task_{ut_n}{tag}: utility={utility} security={security}", flush=True)

        total_utility += suite_u
        total_security += suite_s
        u = sum(suite_u) / len(suite_u) if suite_u else float("nan")
        print(f"[{suite_name}] defense={args.defense} policy={args.policy} utility={u:.3f}", flush=True)
        if args.run_attack:
            s = sum(suite_s) / len(suite_s) if suite_s else float("nan")
            print(f"[{suite_name}] attack={args.attack} security(ASR)={s:.3f}", flush=True)

    if total_utility:
        print(f"\noverall - utility: {sum(total_utility) / len(total_utility):.3f}")
    if args.run_attack and total_security:
        print(f"overall - security (ASR): {sum(total_security) / len(total_security):.3f}")


if __name__ == "__main__":
    main()
