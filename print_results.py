# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Aggregate and print FIDES benchmark results (DRIFT-style per-task JSON logs).

Reads the per-task result JSONs written by ``main.py`` under
``logs/<model>[+fides]/<suite>/user_task_*/<attack|none>/*.json`` and prints a per-suite
utility / security (ASR) table with averages.

    python print_results.py <model_dir> --suite banking slack travel workspace
    python print_results.py gpt-4o-2024-08-06+fides --attack important_instructions
    python print_results.py gpt-4o-2024-08-06            # undefended (no attack -> benign)

``<model_dir>`` is the directory name under ``logs/`` (i.e. ``<model>+fides`` or the bare
``<model>``); if omitted, every model dir found is aggregated.
"""

import argparse
import glob
import json
import os


def main() -> None:
    p = argparse.ArgumentParser(description="Aggregate FIDES per-task result JSONs.")
    p.add_argument("model_dir", nargs="?", default="*", help="Model dir under logs/ (e.g. 'gpt-4o-2024-08-06+fides'). Default: all.")
    p.add_argument("--logdir", default="./logs")
    p.add_argument("--suite", dest="suites", nargs="+", default=None, help="Restrict to these suites.")
    p.add_argument("--attack", default=None, help="Attack label (dir name), or 'none' for benign. Default: all.")
    args = p.parse_args()

    attack_glob = args.attack if args.attack else "*"

    model_dirs = sorted(
        d for d in glob.glob(os.path.join(args.logdir, args.model_dir))
        if os.path.isdir(d) and not os.path.basename(d).startswith("_")
    )
    if not model_dirs:
        print(f"No model dirs matching {os.path.join(args.logdir, args.model_dir)}. Run main.py first.")
        return

    for mdir in model_dirs:
        model = os.path.basename(mdir)
        suites = args.suites or sorted(
            os.path.basename(d) for d in glob.glob(os.path.join(mdir, "*")) if os.path.isdir(d)
        )
        # suite -> attack -> (utilities, securities)
        rows: dict[tuple[str, str], tuple[list, list]] = {}
        for suite in suites:
            pattern = os.path.join(mdir, suite, "user_task_*", attack_glob, "*.json")
            for f in glob.glob(pattern):
                try:
                    with open(f) as fp:
                        d = json.load(fp)
                except Exception:
                    continue
                atk = d.get("attack_type") or "none"
                key = (suite, atk)
                us, ss = rows.setdefault(key, ([], []))
                if d.get("utility") is not None:
                    us.append(bool(d["utility"]))
                if d.get("security") is not None and atk != "none":
                    ss.append(bool(d["security"]))

        if not rows:
            continue

        attacks_seen = sorted({a for (_, a) in rows})
        for atk in attacks_seen:
            print(f"\nModel dir: {model}    Attack: {atk}")
            print(f"{'Suite':<12} {'Utility':>8} {'ASR':>8} {'N':>6}")
            print("-" * 38)
            all_u, all_s = [], []
            for suite in suites:
                if (suite, atk) not in rows:
                    continue
                us, ss = rows[(suite, atk)]
                all_u += us; all_s += ss
                u = f"{sum(us)/len(us):.3f}" if us else "  N/A"
                s = f"{sum(ss)/len(ss):.3f}" if ss else "  N/A"
                print(f"{suite:<12} {u:>8} {s:>8} {len(us):>6}")
            print("-" * 38)
            avg_u = f"{sum(all_u)/len(all_u):.3f}" if all_u else "  N/A"
            avg_s = f"{sum(all_s)/len(all_s):.3f}" if all_s else "  N/A"
            print(f"{'avg':<12} {avg_u:>8} {avg_s:>8} {len(all_u):>6}")


if __name__ == "__main__":
    main()
