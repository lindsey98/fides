# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Summarize FIDES AgentDojo / AgentDyn results (utility / security per suite).

Reads the per-task result JSONs written by ``main.py`` under
``logs/<model>[+fides]/<suite>/user_task_*/<attack|none>/*.json`` and prints a per-suite
utility / security (ASR) table with averages.

Like the sibling baselines, this is paired with ``threat_model.py``: under attack it counts
only the tasks the trust-separation threat model admits (``--filter inscope``, the default) --
i.e. injectable tasks whose payload is separable from the task's trusted content -- so the
reported ASR is over the in-scope task set, not every AgentDojo pair. ``--filter none`` counts
every task. Utility (no attack) is always over the full task set (the filter applies only
under attack).

    python print_results.py --model local:Qwen3.6-35B-A3B --defense fides
    python print_results.py --name Qwen3.6-35B-A3B+fides --suites banking slack travel workspace
    python print_results.py --discover Qwen3.6-35B-A3B          # compare fides vs none side by side
"""

import argparse
import glob
import json
import os
import re

import threat_model as tm

DEFAULT_SUITES = ["banking", "slack", "travel", "workspace"]


def model_dir_name(model: str, defense: str) -> str:
    """Same mapping main.py uses: strip provider prefix, sanitize, add '+fides' when defended."""
    base = model
    for pre in ("local:", "openai:", "anthropic:"):
        if base.startswith(pre):
            base = base[len(pre):]
            break
    base = base.replace("/", "_").replace(":", "_")
    return f"{base}+fides" if defense == "fides" else base


def _task_of(path: str) -> str:
    m = re.search(r"(user_task_\d+)", path)
    return m.group(1) if m else ""


def summarize(log_dir: str, name: str, attack, suites, task_filter: str):
    """task_filter: 'inscope' counts only tasks threat_model.qualifying_tasks admits; 'none' counts all."""
    results = {}
    for suite in suites:
        if attack is None:
            pattern = f"{log_dir}/{name}/{suite}/user_task_*/none/none.json"
        else:
            pattern = f"{log_dir}/{name}/{suite}/user_task_*/{attack}/injection_task_*.json"
        files = glob.glob(pattern)
        if not files:
            continue
        allow = None if task_filter == "none" else tm.qualifying_tasks(suite)
        utility_scores, security_scores, kept_tasks = [], [], set()
        for f in files:
            if allow is not None and _task_of(f) not in allow:
                continue
            try:
                with open(f) as fp:
                    data = json.load(fp)
            except Exception:
                continue
            kept_tasks.add(_task_of(f))
            if data.get("utility") is not None:  # skip unscored / errored records
                utility_scores.append(bool(data["utility"]))
            if attack is not None and data.get("security") is not None:
                security_scores.append(bool(data["security"]))
        if allow is not None and not kept_tasks:
            continue
        results[suite] = {
            "utility": sum(utility_scores) / len(utility_scores) if utility_scores else "N/A",
            "n_tasks": len(kept_tasks),
            "n_total": len({_task_of(f) for f in files}),
        }
        if attack is not None:
            results[suite]["security"] = (
                sum(security_scores) / len(security_scores) if security_scores else "N/A"
            )
    return results


def discover_runs(log_dir: str, substr: str) -> list:
    return sorted(
        os.path.basename(p.rstrip("/"))
        for p in glob.glob(f"{log_dir}/*")
        if os.path.isdir(p) and not os.path.basename(p.rstrip("/")).startswith("_") and substr in os.path.basename(p.rstrip("/"))
    )


def print_compare(title: str, log_dir: str, runs, suites, attack, task_filter: str):
    metric = "security" if attack is not None else "utility"
    rows = []
    for run in runs:
        res = summarize(log_dir, run, attack, suites, task_filter)
        if not res:
            continue
        vals = {s: res[s].get(metric) for s in suites if s in res}
        nums = [v for v in vals.values() if isinstance(v, float)]
        rows.append((run, vals, sum(nums) / len(nums) if nums else None))
    print(f"\n=== {title} :: {metric}{f'  [filter: {task_filter}]' if task_filter != 'none' else ''} ===")
    if not rows:
        print("(no logs found)")
        return
    wlab = max(len(r[0]) for r in rows) + 1
    header = f"{'solution':<{wlab}}" + "".join(f"{s[:8]:>9}" for s in suites) + f"{'avg':>9}"
    print(header)
    print("-" * len(header))
    for run, vals, avg in rows:
        line = f"{run:<{wlab}}"
        for s in suites:
            v = vals.get(s)
            line += f"{(f'{v:.3f}' if isinstance(v, float) else '-'):>9}"
        line += f"{(f'{avg:.3f}' if isinstance(avg, float) else '-'):>9}"
        print(line)


def print_table(title: str, results: dict, task_filter: str):
    print(f"\n=== {title}{f'  [filter: {task_filter}]' if task_filter != 'none' else ''} ===")
    if not results:
        print("(no logs found)")
        return
    has_sec = any("security" in v for v in results.values())
    header = f"{'Suite':<12} {'Tasks':>7} {'Utility':>8}"
    if has_sec:
        header += f" {'Security':>10}"
    print(header)
    print("-" * len(header))
    for suite, scores in results.items():
        tasks = f"{scores['n_tasks']}/{scores['n_total']}"
        line = f"{suite:<12} {tasks:>7} {str(scores['utility'])[:6]:>8}"
        if has_sec:
            line += f" {str(scores.get('security', '-'))[:6]:>10}"
        print(line)
    u = [v["utility"] for v in results.values() if isinstance(v["utility"], float)]
    s = [v["security"] for v in results.values() if isinstance(v.get("security"), float)]
    print("-" * len(header))
    line = f"{'avg':<12} {'':>7}"
    line += f" {sum(u) / len(u):>8.3f}" if u else f" {'-':>8}"
    if has_sec:
        line += f" {sum(s) / len(s):>10.3f}" if s else f" {'-':>10}"
    print(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize FIDES AgentDojo results (utility/security per suite).")
    parser.add_argument("--model", default=None, help="Model name (same as passed to main.py); mapped to its run dir with --defense.")
    parser.add_argument("--defense", choices=("fides", "none"), default="fides", help="Which run dir of --model to read (fides -> <model>+fides, none -> <model>).")
    parser.add_argument("--name", default=None, help="Exact run-directory name under --log-dir (overrides --model/--defense).")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--suites", nargs="+", default=DEFAULT_SUITES, help="Suites to summarize (add shopping github dailylife for AgentDyn).")
    parser.add_argument("--attack", nargs="+", default=["important_instructions"], help="Attack label(s) to report (dir names). Use e.g. tool_knowledge.")
    parser.add_argument("--filter", choices=("inscope", "none"), default="inscope",
                        help="inscope: count only tasks the trust-separation threat model admits (threat_model.qualifying_tasks); none: all tasks.")
    parser.add_argument("--discover", default=None, help="Compare mode: substring of run dirs to collect and print side by side.")
    parser.add_argument("--runs", nargs="+", default=None, help="Compare mode: explicit run-directory names to compare.")
    args = parser.parse_args()

    # (attack_label, title): None = benign utility, then each requested attack.
    attacks = [(None, "No attack (utility)")] + [(a, f"Attack: {a}") for a in args.attack]

    # The threat-model filter applies only under attack; utility is always over the full set.
    def eff_filter(attack):
        return args.filter if attack is not None else "none"

    if args.discover or args.runs:
        runs = args.runs or discover_runs(args.log_dir, args.discover)
        print(f"Comparing {len(runs)} run(s) under {args.log_dir}: " + ", ".join(runs))
        for attack, title in attacks:
            print_compare(title, args.log_dir, runs, args.suites, attack, eff_filter(attack))
    else:
        if not args.name and not args.model:
            parser.error("provide --name <run dir>, or --model <name> [--defense fides|none]")
        name = args.name or model_dir_name(args.model, args.defense)
        print(f"Results for: {name}")
        for attack, title in attacks:
            print_table(title, summarize(args.log_dir, name, attack, args.suites, eff_filter(attack)), eff_filter(attack))
