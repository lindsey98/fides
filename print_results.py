# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Aggregate and print FIDES benchmark results.

Reads the compact summary JSONs that ``main.py`` writes under
``<logdir>/_fides_summary/`` and prints a per-suite utility / security (ASR) table with
averages. Group by model + defense + policy + attack.

    python print_results.py                       # all runs under ./logs
    python print_results.py --logdir ./logs --model gpt-4o-2024-08-06 --attack important_instructions
"""

import argparse
import glob
import json
import os
from collections import defaultdict


def main() -> None:
    p = argparse.ArgumentParser(description="Aggregate FIDES benchmark summaries.")
    p.add_argument("--logdir", default="./logs")
    p.add_argument("--model", default=None, help="Filter by model.")
    p.add_argument("--defense", default=None, help="Filter by defense (fides/none).")
    p.add_argument("--attack", default=None, help="Filter by attack name (or 'none').")
    args = p.parse_args()

    files = glob.glob(os.path.join(args.logdir, "_fides_summary", "*.json"))
    if not files:
        print(f"No summary files under {os.path.join(args.logdir, '_fides_summary')}. "
              "Run main.py first.")
        return

    # group key -> {suite: (utility, asr)}
    groups: dict[tuple, dict[str, tuple]] = defaultdict(dict)
    for f in files:
        with open(f) as fp:
            d = json.load(fp)
        if args.model and d["model"] != args.model:
            continue
        if args.defense and d["defense"] != args.defense:
            continue
        if args.attack and d["attack"] != args.attack:
            continue
        key = (d["model"], d["defense"], d.get("policy"), d["attack"])
        groups[key][d["suite"]] = (d.get("utility"), d.get("security_asr"))

    for key, suites in sorted(groups.items()):
        model, defense, policy, attack = key
        header = f"Model: {model}   Defense: {defense}"
        if defense == "fides":
            header += f" ({policy})"
        header += f"   Attack: {attack}"
        print("\n" + header)
        print(f"{'Suite':<12} {'Utility':>8} {'ASR':>8}")
        print("-" * 30)
        us, ss = [], []
        for suite in sorted(suites):
            u, s = suites[suite]
            if u is not None:
                us.append(u)
            if s is not None:
                ss.append(s)
            u_str = f"{u:.3f}" if u is not None else "  N/A"
            s_str = f"{s:.3f}" if s is not None else "  N/A"
            print(f"{suite:<12} {u_str:>8} {s_str:>8}")
        print("-" * 30)
        avg_u = f"{sum(us)/len(us):.3f}" if us else "  N/A"
        avg_s = f"{sum(ss)/len(ss):.3f}" if ss else "  N/A"
        print(f"{'avg':<12} {avg_u:>8} {avg_s:>8}")


if __name__ == "__main__":
    main()
