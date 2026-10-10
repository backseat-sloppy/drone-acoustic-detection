---
name: council
description: Convene the three-member council (Devil's Advocate, Scholar, Problem Solver) before a big decision in this repo - a new model family, a new frontend or detection algorithm, a changed evaluation protocol or metric, or abandoning a line of work. Use it on your own initiative whenever CLAUDE.md section 4 applies, or when Mads types /council.
---

# Council

Three members judge a proposed decision independently. Each has a fixed role,
and the roles are deliberately unequal in what they are told:

| Member | Sees the project context? | Job |
|---|---|---|
| **Devil's Advocate** | Yes | Argue against the proposal, as strongly as honestly possible |
| **Scholar** | **No** | Say what established best practice is for this kind of problem |
| **Problem Solver** | Yes | Find the most practical way to reach the goal from where we are |

You then weigh what they said and decide. The council advises; you remain
responsible for the decision.

## 1. Write two briefs

The members know nothing about this conversation, so each brief must stand on
its own.

**The full brief** (for the Devil's Advocate and the Problem Solver):

- **The decision**, as a single question, naming the proposal and the
  alternatives (always including "keep the current approach").
- **Evidence so far**: the relevant entries from `docs/REPORT.md`, with the
  actual numbers, split and conditions.
- **Requirements**: the relevant rows of `REQUIREMENTS.md`.
- **Constraints**: data available (`docs/datasets.md`), hardware, time.

State the evidence neutrally. Do not argue for the option you prefer.

**The context-free brief** (for the Scholar):

- The same decision restated as a general question about the class of
  problem, for example "binary acoustic event detection from a few hours of
  labelled audio, evaluated on unseen recording conditions".
- Only the generic facts needed to make the question answerable: task type,
  rough data scale, signal type, deployment class.
- **Leave out** our results, our current approach, which option we lean
  toward, and anything else that would let the Scholar tell what answer we
  hope for. The Scholar is the outside reference point; context would pull
  the answer toward what we already believe.

## 2. Ask the three members, in parallel, independently

Spawn three subagents in one message. They must not see each other's answers.

### Devil's Advocate (full brief)

Role: argue against the proposal. Not balanced, not constructive: make the
best case that it is the wrong thing to do. Critical thinking is the point,
so do not soften the case or end by agreeing.

Ask for:
- The strongest case against the proposal.
- How it is most likely to fail, or to look like it succeeded when it did
  not (leakage, a flattering metric, source confounded with label, tuning on
  test data, too few sessions).
- What it costs, and what else that effort could have bought.
- What would have to be true for the proposal to be right after all.
- Its own ranking of its objections: which are decisive, which are serious,
  and which are weak points raised for completeness.

### Scholar (context-free brief)

Role: state best practice for this class of problem, from the literature and
from established methodology. The Scholar judges against the standard, not
against our situation.

Ask for:
- The standard, well-supported approach to this kind of problem.
- The known pitfalls and the accepted way of evaluating it.
- Where each option in the question sits relative to that practice:
  standard, defensible variation, or against practice.
- Sources. Every reference must be one the Scholar actually checked; anything
  from memory is marked "unverified". An invented citation is worse than
  none.
- Where best practice is genuinely unsettled or disputed.

### Problem Solver (full brief)

Role: get to the goal. Looks at the actual numbers, data and constraints with
practical eyes and asks what the most effective next move is.

Ask for:
- Verdict: **try it**, **try a modified version**, or **do not try it**.
- If not this, what instead. A concrete alternative, not just a no.
- The cheapest experiment that would settle the question, and what result
  would count as success or failure.
- A short plan: steps, and what is needed for each.
- Confidence: low, medium or high, and what it rests on.

## 3. Weigh and decide

- **The Devil's Advocate's opposition is not a vote.** It argues against by
  design. Judge the arguments, not the stance: every objection it ranks as
  decisive or serious must be answered, either with evidence already in the
  report or by building the answer into the experiment. An objection you
  cannot answer stands.
- **The Scholar lacks our context on purpose.** If the proposal departs from
  what the Scholar calls best practice, that is allowed only with a specific
  reason rooted in our situation, and the reason goes in the report. "It is
  easier" is not a reason.
- **Scholar against Problem Solver** is the most informative disagreement:
  it marks exactly where our situation is pulling us away from the standard.
  Work out which fact the disagreement turns on.
- Any member can be wrong. Check a factual claim against the repo, the data
  or a source before a decision rests on it.

## 4. Record it

Append a `docs/REPORT.md` entry of type **Council** with:

- the question,
- the Devil's Advocate's decisive and serious objections, and how each was
  answered or left standing,
- the Scholar's statement of best practice, and any departure from it with
  its justification,
- the Problem Solver's verdict and proposed experiment,
- the decision and why.

Then carry on with the iteration.
