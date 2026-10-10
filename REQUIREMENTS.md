# Requirements

This file is Mads's. It says what the iterations aim for and which settings
are allowed to move. Claude reads it before every iteration and does not
change targets or locked settings without asking.

How to use it: edit a value, and change its status. Add rows freely.

- **LOCKED**: must not change.
- **TARGET**: a goal to reach.
- **FREE**: Claude may vary this while iterating.
- **PROPOSED**: a default Claude suggested. Not yet confirmed by Mads; treat
  as a working assumption and confirm or replace it.

## Goal

Supervised detection of a drone from a ground-based microphone: "drone
present" versus "no drone". Localisation is not a goal.

## Performance targets

| Requirement | Value | Status |
|---|---|---|
| Primary metric | False-alarm rate at 95% drone recall, on held-out test sources | PROPOSED |
| Primary target | at most 5% false alarms at 95% recall | PROPOSED |
| Accuracy (balanced, held-out sources) | at least 90% | PROPOSED |
| ROC AUC (held-out sources) | at least 0.95 | PROPOSED |
| False alarms on helicopter clips (Halmstad) | reported separately, no target yet | PROPOSED |
| Must beat baselines `comb` and `lgbm` | yes | PROPOSED |

Accuracy is listed because it is easy to read, but it is the secondary
number: on a balanced laboratory split it looks high for any model. The
held-out-source false-alarm rate is the one that predicts field behaviour.

## Signal settings

| Setting | Value | Status |
|---|---|---|
| Sample rate | 16 kHz mono | PROPOSED (currently fixed in `src/dad/config.py`) |
| Window length | 1.0 s | FREE within 0.25 to 2.0 s |
| Hop | 0.125 s | FREE within 0.05 to 0.5 s |
| Features | 64 log-mel bands, 50 to 8000 Hz | FREE |

## Latency and deployment

| Requirement | Value | Status |
|---|---|---|
| Deployment hardware | not decided (laptop, Raspberry Pi or microcontroller) | OPEN |
| Training hardware | Mads's PC. The AMD GPU may be used through DirectML, in a separate environment. | Decided by Mads, 2026-10-10 |
| Cold-onset latency | no target yet | OPEN |
| Model size or parameter limit | no limit yet | OPEN |

## Data

| Requirement | Value | Status |
|---|---|---|
| Training sources | DDL (real), FSD50K-dev (licence-filtered) | PROPOSED |
| Held-out test sources | DroneNoise, Halmstad, FSD50K-eval | PROPOSED |
| Only real recordings, no synthetic training data | yes. Summing two real recordings (a drone window plus a real background) is allowed as training augmentation. Generated or rendered audio is not. | LOCKED (decided by Mads, 2026-10-10) |
| Licences allowed | CC0 and CC BY | PROPOSED |

## Stop condition

Stop iterating and report when every TARGET (and confirmed PROPOSED target)
above is met on the held-out test sources, or when the report shows it cannot
be met with the available data.

## Notes and wishes

Write anything else here: things to try, things to avoid, deadlines.

-
