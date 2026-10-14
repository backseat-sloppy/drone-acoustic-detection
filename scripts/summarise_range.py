"""Tables for the report from one or more run_range.py result files.

    uv run python -I scripts/summarise_range.py it3_screen [more run names]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from dad.rangerun import _get  # noqa: E402


def f(v, d=2) -> str:
    return "-" if v is None else f"{v:.{d}f}"


def ci(v) -> str:
    return "-" if not v else f"{v[0]:.2f}-{v[1]:.2f}"


def main(names: list[str]) -> None:
    rows = []
    for n in names:
        rows += json.loads((REPO / "artifacts" / "runs" / f"{n}.json").read_text())["results"]
    print("| Arm | Recipe | Model | Seed | Window x k | Recall 100-200 m at 5 / 10 / 20% FA | 95% interval (10%) | "
          "Recall 0-50 m (10%) | Control >300 m AUC | Alarm within 5 s at 100-200 m (no drone) (10%) | Deploy: recall 100-200 m / FA |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        t = r["test"]
        rec = [_get(t, "own", k, "by_range", "100-200 m", "recall") for k in ("0.05", "0.10", "0.20")]
        print(f"| {r['arm']} | {r['recipe']} | {r['scorer']} | {r.get('seed', 0)} | {r.get('window_s', 1.0):g} s x {r.get('integrate', 1)} | "
              f"{' / '.join(f(v) for v in rec)} | {ci(_get(t, 'own', '0.10', 'recall_100_200_ci95'))} | "
              f"{f(_get(t, 'own', '0.10', 'by_range', '0-50 m', 'recall'))} | "
              f"{f(_get(t, 'auc_session', '300-inf m', 'auc'))} | "
              f"{f(_get(t, 'own', '0.10', 'latency_100_200', 'within_5s'))} ({f(_get(t, 'own', '0.10', 'latency_background', 'within_5s'))}) | "
              f"{f(_get(t, 'deploy', 'by_range', '100-200 m', 'recall'))} / {f(t.get('deploy_fa'))} |")
    print("\nAUC against the same sessions' background, by range (held-out rig)")
    bins = ["0-50 m", "50-100 m", "100-150 m", "150-200 m", "200-300 m", "300-inf m"]
    print("| Arm | Recipe | Model | " + " | ".join(bins) + " |")
    print("|---|---|---|" + "---|" * len(bins))
    for r in rows:
        a = r["test"].get("auc_session", {})
        print(f"| {r['arm']} | {r['recipe']} | {r['scorer']} | " + " | ".join(f(_get(a, b, "auc")) for b in bins) + " |")
    print("\nFurther rigs and the mower session")
    print("| Arm | Recipe | Model | Lund recall 100-200 m (deploy) | Glasgow recall 1-5 m at own 10% FA | "
          "Mower: FA at own 10% threshold | Mower: recall 100-200 m |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        u = r.get("unseen", {})
        c = r.get("confuser", {})
        print(f"| {r['arm']} | {r['recipe']} | {r['scorer']} | "
              f"{f(_get(u, 'lund', 'deploy', 'by_range', '100-200 m', 'recall'))} | "
              f"{f(_get(u, 'glasgow', 'own', '0.10', 'by_range', '0-50 m', 'recall'))} | "
              f"{f(c.get('fa_own10'))} | {f(_get(c, 'recall_own10', '100-200 m', 'recall'))} |")


if __name__ == "__main__":
    main(sys.argv[1:])
