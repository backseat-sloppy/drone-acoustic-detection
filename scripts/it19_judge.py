"""Iteration 19 (report entry 169): the box's four readouts, it19_colour against it16_600k.

    .venv/bin/python scripts/it19_judge.py      # after scripts/it19_colour_check.py

Per network, means over seeds 0-2 at the deployment threshold (5% FA on validation): validation
recall, Lund recall, mower-only false alarms (from the run files) and pink / brown alarms at -30 dBFS
(from artifacts/runs/it19_colour_check.json).
"""
import json
import statistics
from pathlib import Path

RUNS = Path(__file__).resolve().parents[1] / "artifacts" / "runs"
ARMS = {"600k": "it16_600k", "colour": "it19_colour"}


def get(d, *keys):
    for k in keys:
        d = d.get(k) if isinstance(d, dict) else None
    return d


colour = json.loads((RUNS / "it19_colour_check.json").read_text())
m = {}
for arm, prefix in ARMS.items():
    for net in ("comb", "lgbm", "dscnn", "crnn"):
        rows = []
        for s in (0, 1, 2):
            run = f"{prefix}_s{s}"
            for r in json.loads((RUNS / f"{run}.json").read_text())["results"]:
                if r.get("integrate", 1) != 1 or r["scorer"].split(" ")[0] != net:
                    continue
                c = {x["colour"]: x["alarm"] for x in colour if x["run"] == run and x["arch"] == net and x["dbfs"] == -30}
                rows.append(dict(val=get(r, "val", "deploy", "by_range", "all", "recall"),
                                 lund=get(r, "unseen", "lund", "deploy", "by_range", "all", "recall"),
                                 mower=get(r, "confuser", "fa_deploy"), pink=c.get("pink"), brown=c.get("brown")))
        if rows:
            m[(arm, net)] = {k: statistics.mean(v) if (v := [r[k] for r in rows if r[k] is not None]) else None
                             for k in rows[0]} | {"seeds": len(rows)}

fmt = lambda v: "   -  " if v is None else f"{v:6.3f}"   # noqa: E731
print(f"{'arm':7} {'net':6} seeds    val   lund  mower   pink  brown")
for (arm, net), v in m.items():
    print(f"{arm:7} {net:6} {v['seeds']:5} " + " ".join(fmt(v[k]) for k in ("val", "lund", "mower", "pink", "brown")))

print("\nBox: pink and brown <= 0.05, Lund >= 600k - 0.03, validation >= 600k - 0.02, mower <= 600k + 0.05")
for net in ("dscnn", "crnn"):
    a, b = m.get(("600k", net)), m.get(("colour", net))
    if not a or not b:
        continue
    checks = dict(colour=max(b["pink"], b["brown"]) <= 0.05, lund=b["lund"] - a["lund"] >= -0.03,
                  val=b["val"] - a["val"] >= -0.02, mower=b["mower"] - a["mower"] <= 0.05)
    print(f"  {net:6} Lund {b['lund'] - a['lund']:+.3f}  val {b['val'] - a['val']:+.3f}  mower {b['mower'] - a['mower']:+.3f}  "
          f"pink {b['pink']:.3f}  brown {b['brown']:.3f}  -> " + ("PASSES" if all(checks.values()) else
                                                                   "fails: " + ", ".join(k for k, ok in checks.items() if not ok)))
