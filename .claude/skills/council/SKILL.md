---
name: council
description: Convene three independent reviewers before a big decision in this repo - a new model family, a new frontend or detection algorithm, a changed evaluation protocol or metric, or abandoning a line of work. Use it on your own initiative whenever CLAUDE.md section 4 applies, or when Mads types /council.
---

# Council

Three reviewers judge a proposed decision independently, each through a
different lens. You then weigh what they said and decide. The council
advises; you remain responsible for the decision.

## 1. Write the brief

One self-contained brief, because the reviewers know nothing about this
conversation. Include:

- **The decision**, as a single question with the options on the table
  (always include "do nothing / keep the current approach").
- **Evidence so far**: the relevant entries from `docs/REPORT.md`, with the
  actual numbers, split and conditions.
- **Requirements**: the relevant rows of `REQUIREMENTS.md`.
- **Constraints**: data available (`docs/datasets.md`), hardware, time.
- **What would change your mind.**

State the evidence neutrally. Do not argue for the option you prefer.

## 2. Ask three reviewers, in parallel, independently

Spawn three subagents in one message. Each gets the same brief plus one lens.
They must not see each other's answers.

1. **Methodology sceptic.** Is the evidence sound? Look for leakage, test-set
   tuning, confounds between source and label, too few sessions, a metric
   that flatters. Would the proposed change be measurable at all with this
   data?
2. **Signal and model specialist.** Does the proposal fit the physics of the
   problem (harmonic rotor noise, distance, wind, confusers such as
   helicopters and engines) and the amount of data? Is there a simpler
   change that would capture most of the gain?
3. **Requirements and deployment advocate.** Does it move a number in
   `REQUIREMENTS.md`? What does it cost in latency, size, complexity and
   iteration time? Is it the cheapest next step toward the targets?

Ask each for exactly this:

- Verdict: **try it**, **try a modified version**, or **do not try it**.
- The two or three reasons that carry the verdict.
- The biggest risk if the verdict is followed.
- The cheapest experiment that would settle the question.
- Confidence: low, medium or high, and what it rests on.

## 3. Weigh and decide

- Where all three agree, that is a strong signal. Follow it unless you have
  evidence they lacked.
- Where they disagree, the disagreement is the finding. Identify which
  factual question it turns on and prefer the cheapest experiment that
  answers it.
- A reviewer can be wrong. Check any factual claim that the decision depends
  on against the repo or the data before relying on it.

## 4. Record it

Append a `docs/REPORT.md` entry of type **Council**, containing the brief's
question, each reviewer's verdict and main reason, the points of dissent, the
decision, and why. Then carry on with the iteration.
