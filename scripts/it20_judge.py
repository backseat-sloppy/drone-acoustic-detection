"""Iteration 20 (report entries 172-173): the bandpass arm against it16_600k, by the box's rules.

    .venv/bin/python scripts/it19_colour_check.py it20_bp_s0 it20_bp_s1 it20_bp_s2
    .venv/bin/python scripts/it20_judge.py

Per network, means over seeds 0-2 at the deployment threshold (5% FA on validation): validation recall,
Lund recall, mower-only false alarms (run files) and pink / brown alarms at -30 dBFS (it16_600k's rows from
it19_colour_check.json, the arm's from it20_bp_colour_check.json).
Pass: Lund >= 600k + 0.07 for at least one network, and for both networks validation >= 600k - 0.02,
mower <= 600k + 0.05, pink and brown <= 600k's.
"""
import json
import statistics
from pathlib import Path

RUNS = Path(__file__).resolve().parents[1] / "artifacts" / "runs"
ARMS = {"600k": "it16_600k", "bandpass": "it20_bp"}
colour = [row for f in ("it19_colour_check.json", "it20_bp_colour_check.json")
          for row in json.loads((RUNS / f).read_text())]


def get(d, *keys):
    for k in keys:
        d = d.get(k) if isinstance(d, dict) else None
    return d


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
            v = {k: statistics.mean(x) if (x := [r[k] for r in rows if r[k] is not None]) else None
                 for k in ("val", "lund", "mower", "pink", "brown")}
            m[(arm, net)] = v | {"seeds": len(rows), "lund_seeds": [round(r["lund"], 3) for r in rows]}

fmt = lambda v: "   -  " if v is None else f"{v:6.3f}"   # noqa: E731
print(f"{'arm':8} {'net':6} seeds    val   lund  mower   pink  brown   Lund per seed")
for (arm, net), v in m.items():
    print(f"{arm:8} {net:6} {v['seeds']:5} " + " ".join(fmt(v[k]) for k in ("val", "lund", "mower", "pink", "brown"))
          + f"   {v['lund_seeds']}")

print("\nBox: Lund >= 600k + 0.07 for one network; validation >= 600k - 0.02, mower <= 600k + 0.05, "
      "pink and brown <= 600k's for both")
lund_gain, guards = False, True
for net in ("dscnn", "crnn"):
    a, b = m[("600k", net)], m[("bandpass", net)]
    ok = dict(val=b["val"] - a["val"] >= -0.02, mower=b["mower"] - a["mower"] <= 0.05,
              colour=b["pink"] <= a["pink"] and b["brown"] <= a["brown"])
    lund_gain |= b["lund"] - a["lund"] >= 0.07
    guards &= all(ok.values())
    print(f"  {net:6} Lund {b['lund'] - a['lund']:+.3f}  val {b['val'] - a['val']:+.3f}  mower {b['mower'] - a['mower']:+.3f}  "
          f"pink {b['pink'] - a['pink']:+.3f}  brown {b['brown'] - a['brown']:+.3f}  guards: "
          + ("ok" if all(ok.values()) else "broken: " + ", ".join(k for k, g in ok.items() if not g)))
print("Verdict:", "PASSES" if lund_gain and guards else
      "fails: " + ", ".join(x for x, bad in (("no Lund gain of 0.07", not lund_gain), ("guards", not guards)) if bad))
