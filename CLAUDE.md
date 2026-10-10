# Working agreement for this repo

Acoustic drone detection from ground-based microphones. Mads owns the goals;
Claude plans, builds, tests and iterates on its own between check-ins. These
rules apply in every session, on every machine.

Reply to Mads in Danish. Everything written into the repo is in English.

## 1. Log every step in the report

`docs/REPORT.md` is the lab notebook, and it becomes the project report.
**Append an entry for every completed step, before moving to the next one.**
Nothing is too small: a new model, a new algorithm, a change to how data is
prepared before training, a tuned hyperparameter, a changed split, a bug fix
that moves a number, a dead end.

- Append only. Never rewrite or delete an old entry; if it turned out wrong,
  add a new entry that says so and points back to it.
- Follow the entry template at the top of `docs/REPORT.md`.
- Record negative results with the same care as positive ones.
- Every number in an entry must come from a run that actually happened, with
  the command and commit that produced it. If something was not run or not
  verified, write that.

## 2. Read the requirements before each iteration

`REQUIREMENTS.md` is Mads's file. It holds the targets and the knobs that are
allowed to move (metric targets, window, hop, model limits and so on).

- Read it at the start of every iteration and judge results against it.
- Do not change a target or a locked setting in it. If a requirement looks
  unreachable or mis-specified, say so in the report and ask.
- When Mads changes it, log the change as a report entry.

## 3. Iterate autonomously until the requirements are met

After every test: evaluate the result against `REQUIREMENTS.md`, decide the
next step, do it, log it. Do not stop to ask between ordinary iterations.

Loop: hypothesis -> change one thing -> run -> evaluate -> report entry ->
next hypothesis.

Stop and check in when:
- the requirements are met on the held-out test sources,
- a requirement looks unreachable with the available data,
- the next step needs something only Mads can provide or decide (new data,
  spending money, changing a requirement, anything outside this repo).

## 4. Convene the council before big decisions

Run the `council` skill (`.claude/skills/council/`) before:
- trying a new model family or architecture,
- replacing the feature frontend or the detection algorithm,
- changing the evaluation protocol, the split or the primary metric,
- abandoning a line of work that took more than one iteration.

Not for hyperparameter tuning, bug fixes or reruns. Weigh the council's
feedback, decide, and record the brief, the verdicts, any dissent and the
decision in the report. The council advises; it does not decide.

## 5. Evaluation hygiene (non-negotiable)

These exist because an inflated number is worse than a low one.

- **Split by session or source, never by clip.** Neighbouring clips from one
  recording leak.
- **Tune on validation, not on test.** Model selection, thresholds and early
  stopping use a validation split carved from the training sources. The
  held-out test sources are scored only to report a finished iteration, and
  every such scoring is logged. If test results start steering design
  choices, say so in the report.
- **Report the metric with its conditions:** split, sources, class balance,
  window and hop. Accuracy alone on a balanced set is not a result.
- **Always report the baselines** (`comb`, `lgbm`) next to any new model.
- **Synthetic data is scaffolding.** Numbers from `src/dad/synth.py` are never
  reported as detection performance.

## 6. Data rules

- `data/` is git-ignored and holds the raw downloads exactly as published.
  Never modify, rename or delete anything under the raw dataset folders.
  Derived data goes in a separate folder (for example `data/extracted/`,
  `data/processed/`).
- Provenance and licences are in `docs/datasets.md`. Re-create `data/` on a
  new machine with `scripts/download_datasets.ps1`.
- Respect per-clip licences (FSD50K) and unresolved ones (Halmstad).

## Project commands

```powershell
uv sync --group dev --extra dl   # environment, with torch
uv run dad info                  # the frozen frontend contract
uv run dad bench                 # per-stage latency
uv run dad compare               # every model on the same windows
uv run pytest                    # tests
```

Commit or push only when Mads asks.
