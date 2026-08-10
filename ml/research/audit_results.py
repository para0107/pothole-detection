
"""
ml/research/audit_results.py
----------------------------
Verify a rescued `runs/research` tree is complete and reportable.

WHY THIS EXISTS
    The research weekends run on a temporary AWS account that is destroyed with no
    recovery, so results are copied off in a hurry. "I think I downloaded everything"
    is not a claim anyone can check by eye across 15 run directories, and the failure
    is silent: a missing `metrics.csv` looks identical to a run that simply had none,
    until months later when the per-epoch curve is needed for a paper figure and the
    GPU hours are gone.

    This walks a rescued tree and answers three questions:
      1. Which runs are COMPLETE (trained, evaluated on test, per-class AP recorded)?
      2. Which are REPORTABLE (complete AND produced from a clean git tree)?
      3. What is MISSING that cannot be regenerated?

USAGE
    python ml/research/audit_results.py "C:/path/to/Weekend2"
    python ml/research/audit_results.py ~/rdds/runs/research --verbose

EXIT CODES
    0  every run directory is complete
    1  at least one run is missing an irreplaceable artefact
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Per-run artefacts. `critical` means it cannot be regenerated without re-running the
# GPU hours: weights retrain from (seed, config), a lost metrics.csv does not.
ARTEFACTS = {
    "run.json":          ("provenance: seed, git SHA, dataset hash, timings", True),
    "config.json":       ("every resolved hyperparameter", True),
    "metrics.csv":       ("per-epoch train/val curve", True),
    "test_metrics.json": ("held-out test headline numbers", True),
    "per_class_ap.json": ("per-class AP@50 on test", True),
}

# Tree-level artefacts that are not per-run.
TREE_LEVEL = {
    "_comparison": "leaderboard + statistical comparison",
    "_figures": "publication figures",
    "_weekend_log.json": "queue execution log (what ran, how long, exit codes)",
    "_kaggle_nrdd2024.json": "dataset inspection: class order, country composition",
    "E1_anisotropy": "weekend-1 shape analysis (the refuted hypothesis)",
}


def audit_run(d: Path) -> dict:
    """Inspect one run directory."""
    present = {name: (d / name).exists() for name in ARTEFACTS}
    missing_critical = [n for n, ok in present.items() if not ok and ARTEFACTS[n][1]]

    info: dict = {
        "name": d.name,
        "present": present,
        "missing_critical": missing_critical,
        "complete": not missing_critical,
        "dirty": None,
        "seed": None,
        "experiment": None,
        "status": None,
        "map": None,
        "n_epochs": 0,
        "has_weights": (d / "weights").is_dir(),
    }

    if present["run.json"]:
        try:
            r = json.loads((d / "run.json").read_text(encoding="utf-8"))
            info["dirty"] = r.get("git_dirty")
            info["seed"] = r.get("seed")
            info["experiment"] = r.get("experiment")
            info["status"] = r.get("status")
        except (json.JSONDecodeError, OSError):
            info["status"] = "run.json unreadable"

    if present["test_metrics.json"]:
        try:
            t = json.loads((d / "test_metrics.json").read_text(encoding="utf-8"))
            info["map"] = t.get("test", {}).get("mAP50-95")
        except (json.JSONDecodeError, OSError):
            pass

    if present["metrics.csv"]:
        try:
            info["n_epochs"] = max(
                0, len((d / "metrics.csv").read_text(encoding="utf-8").splitlines()) - 1
            )
        except OSError:
            pass

    return info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("root", help="the rescued runs/research directory")
    ap.add_argument("--verbose", action="store_true",
                    help="list every artefact, present or not")
    args = ap.parse_args()

    root = Path(args.root).expanduser()
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 1

    # A rescued tree is sometimes nested one level deep (tar extracted into a folder).
    runs = sorted(d for d in root.iterdir()
                  if d.is_dir() and not d.name.startswith(("_", ".")) and d.name != "E1_anisotropy")
    if not runs:
        nested = [d for d in root.iterdir() if d.is_dir()]
        for n in nested:
            cand = sorted(x for x in n.iterdir()
                          if x.is_dir() and not x.name.startswith(("_", ".")))
            if cand:
                print(f"[note] runs found nested under {n.name}/\n")
                root, runs = n, cand
                break

    print(f"auditing {root}\n")
    audits = [audit_run(d) for d in runs]

    print(f"{'run':44s} {'exp':22s} {'seed':>5s} {'ep':>3s} {'mAP':>7s} {'dirty':>6s}  state")
    print("-" * 104)
    complete = reportable = 0
    for a in audits:
        state = "COMPLETE" if a["complete"] else f"missing {','.join(a['missing_critical'])}"
        if a["complete"]:
            complete += 1
            if a["dirty"] is False:
                reportable += 1
        dirty = {True: "yes", False: "no", None: "?"}[a["dirty"]]
        mp = f"{a['map']:.4f}" if a["map"] is not None else "-"
        print(f"{a['name'][:44]:44s} {str(a['experiment'])[:22]:22s} "
              f"{str(a['seed']):>5s} {a['n_epochs']:>3d} {mp:>7s} {dirty:>6s}  {state}")

    print(f"\n{len(audits)} run(s): {complete} complete, "
          f"{reportable} reportable (complete AND clean git tree)")

    print("\ntree-level artefacts:")
    for name, why in TREE_LEVEL.items():
        ok = (root / name).exists()
        print(f"  {'ok  ' if ok else 'MISS'} {name:24s} {why}")

    n_weights = sum(1 for a in audits if a["has_weights"])
    print(f"\ncheckpoints: {n_weights}/{len(audits)} runs have a weights/ directory")
    if n_weights < len(audits):
        print("  Not a problem by itself - weights retrain from the recorded seed and")
        print("  config. The metrics are the irreplaceable half, and they are above.")

    broken = [a for a in audits if not a["complete"]]
    if broken:
        print(f"\n{len(broken)} incomplete run(s):")
        for a in broken:
            print(f"  {a['name']}: missing {', '.join(a['missing_critical'])}")
        print("\n  An incomplete run is expected if it was killed mid-training or only")
        print("  finished phase 1. Check its status field before assuming data loss.")
        return 1

    print("\nAll runs complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
