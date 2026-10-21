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

## Priorities

Set by Mads on 2026-10-14 (report entry 021), in his words: "Falsealarms are
not a pirority, high recall rate matters more, time is also important, and the
most relevant distance is around 150 meters distance from drones, meaning we
should somehow weigh the data."

| Requirement | Value | Status |
|---|---|---|
| Order of priority | 1. recall, 2. time to detection, 3. false alarms | Decided by Mads, 2026-10-14 |
| Range that matters most | about 150 m from the microphone to the drone; data and metrics weighted toward it | WITHDRAWN for now by Mads, 2026-10-20 (entry 038); was decided by Mads, 2026-10-14 |
| Meaning of "time" | time to detection (how soon an alarm follows the drone coming into range), not processing speed | Confirmed by Mads, 2026-10-14 |
| Operational scenario | small drones, flying at up to about 50 m/s toward a soldier, in heavy noise; the alarm must come at least 3 s before the drone arrives | Decided by Mads, 2026-10-14 (entry 032). The clause "i.e. while it is still at least about 150 m away" is WITHDRAWN for now by Mads, 2026-10-20 (entry 038) |
| Drone class | small drones | Decided by Mads, 2026-10-14 |
| Recall target | as high as can be reached; no fixed number | Decided by Mads, 2026-10-14 |

## Performance targets

| Requirement | Value | Status |
|---|---|---|
| Primary metric (range-free) | Recall on all drone windows of a rig never trained on, at 10% window false alarms on that rig's own no-drone audio; worse rig decides; interval from resampling flights. Recall by range is still reported where range is known, without a target | PROPOSED (entry 038) |
| Primary metric (at 100-200 m) | Recall on drone windows at 100-200 m slant range, on a rig never trained on, at 10% window false alarms on that rig's own no-drone audio; worse rig decides; interval from resampling flights | WITHDRAWN for now by Mads, 2026-10-20 (entry 038); was PROPOSED (council, entry 025) |
| Deployment figure (new) | Held-out recall on all drone windows, and the false alarms it costs, at a threshold frozen at 10% false alarms on validation | PROPOSED (entry 025; range clause withdrawn in entry 038) |
| Time to detection (new) | Share of drone moments followed by an alarm within 5 s (window length included); no target yet | PROPOSED (entry 025; range clause withdrawn in entry 038) |
| Primary metric | False-alarm rate at 95% drone recall, on held-out test sources | SUPERSEDED by the priorities above (2026-10-14); still reported |
| Primary target | at most 5% false alarms at 95% recall | SUPERSEDED by the priorities above (2026-10-14); still reported |
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
| Deployment microphones | an array of 2 omnidirectional microphones | Decided by Mads, 2026-10-14 |
| Training hardware | Mads's PC. The AMD GPU may be used through DirectML, in a separate environment. | Decided by Mads, 2026-10-10 |
| Cold-onset latency | no target yet | OPEN |
| Model size or parameter limit | no limit yet | OPEN |

## Data

| Requirement | Value | Status |
|---|---|---|
| Training sources | DDL (real), FSD50K-dev (licence-filtered) | PROPOSED |
| Held-out test sources | DroneNoise, Halmstad, FSD50K-eval | PROPOSED |
| Only real recordings, no synthetic training data | yes. Summing two real recordings (a drone window plus a real background) is allowed as training augmentation. Generated or rendered audio is not. | LOCKED (decided by Mads, 2026-10-10) |
| Pitch or speed perturbation | allowed: a real drone recording may be resampled (played faster or slower, about +-20%) to imitate the Doppler shift and higher motor speed of a fast drone | Decided by Mads, 2026-10-14 (amends the row above) |
| Acoustic UAV Field Recordings (FPV, Zenodo) | may be fetched | Decided by Mads, 2026-10-14 |
| Licences allowed | CC0 and CC BY | PROPOSED |
| d3set (GPL-3.0) and Lund (no licence stated) | allowed for internal use; not for publishing results or audio without a further decision (Lund: ask the authors) | Decided by Mads, 2026-10-14 |
| Training and test rigs from iteration 3 | leave one rig out between Nightjar and d3set; Lund and Glasgow scored as further unseen rigs (entry 025) | PROPOSED |
| Sealed test for recall by range | ITU ARIS and RODENT (fetched 2026-10-14 on Mads's go-ahead), scored only for a finished candidate | PROPOSED |

## Stop condition

Stop iterating and report when every TARGET (and confirmed PROPOSED target)
above is met on the held-out test sources, or when the report shows it cannot
be met with the available data.

## Notes and wishes

Write anything else here: things to try, things to avoid, deadlines.

-
