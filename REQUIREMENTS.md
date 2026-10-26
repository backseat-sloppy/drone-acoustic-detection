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

## Requirements (MoSCoW)

Set by Mads on 2026-10-21 (report entry 039) for the coming real test. These
are the governing requirements: where a row further down disagrees with
them, these win. Distance to the drone is no longer a main criterion.

| ID | Functional | Non-functional | MoSCoW | Status |
|---|---|---|---|---|
| R1 | The solution should detect drone | The program should return true value when it detects a drone | Must have | TARGET, decided by Mads, 2026-10-21 |
| R2 | The solution should minimize false negatives | The program should have a recall rate of 95% | Should have | TARGET, decided by Mads, 2026-10-21 |
| R3 | The solution should minimize false positives | The program should introduce a maximum of 5% false positives | Should have | TARGET, decided by Mads, 2026-10-21 |
| R4 | The solution should be precise | The program should have a precision rate of 95% | Could have | TARGET, decided by Mads, 2026-10-21 |
| R5 | The solution should give soldiers time to react to an alert | The program should detect drones 3 seconds before impact | Must have | TARGET, decided by Mads, 2026-10-21 |

How each requirement is measured. The wording above is Mads's; the
measurement below is Claude's reading of it and is PROPOSED until Mads
confirms it.

| ID | Measured as | Status |
|---|---|---|
| R1 | For every hop the program returns a boolean: `True` when the model's score is at or above the threshold, else `False` | PROPOSED (entry 039) |
| R2 | Recall: share of drone windows returned `True`, on sources or rigs never trained on, at one threshold frozen on validation. Windows within 1 s of a drone's start or end are left out | PROPOSED (entry 039) |
| R3 | False-positive rate: share of no-drone windows returned `True`, on the same data and at the same threshold as R2 | PROPOSED (entry 039) |
| R4 | Precision: share of `True` windows that are drone windows, same data and threshold. Precision depends on how much of the test audio holds a drone, so that share is always stated with it | PROPOSED (entry 039) |
| R5 | On test audio where the drone's approach is known (for example generated test files): time from the first `True` to impact, per approach, counting only windows that are at least half drone (an alarm from a window that is mostly pre-drone audio is not credited, entry 041); met when it is at least 3 s. Share of approaches that must meet it: not set | PROPOSED (entry 039); share OPEN |
| All | R2, R3 and R4 hold together at one threshold, not each at its own | PROPOSED (entry 039) |

## Priorities

Set by Mads on 2026-10-14 (report entry 021), in his words: "Falsealarms are
not a pirority, high recall rate matters more, time is also important, and the
most relevant distance is around 150 meters distance from drones, meaning we
should somehow weigh the data." Partly superseded on 2026-10-21 by the
MoSCoW requirements above.

| Requirement | Value | Status |
|---|---|---|
| Order of priority | 1. recall, 2. time to detection, 3. false alarms | SUPERSEDED by the MoSCoW requirements (2026-10-21): detection and 3 s before impact are Must; recall and false positives are both Should; precision is Could |
| Range that matters most | about 150 m from the microphone to the drone; data and metrics weighted toward it | WITHDRAWN by Mads, 2026-10-20 (entry 038); on 2026-10-21 Mads added that distance is no longer as valuable. Recall by range is still reported where range is known, without a target |
| Meaning of "time" | time to detection (how soon an alarm follows the drone coming into range), not processing speed | Confirmed by Mads, 2026-10-14; made concrete by R5 |
| Operational scenario | small drones, flying at up to about 50 m/s toward a soldier, in heavy noise; the alarm must come at least 3 s before the drone arrives | Decided by Mads, 2026-10-14 (entry 032); the 3 s is now R5. The clause "i.e. while it is still at least about 150 m away" is WITHDRAWN by Mads, 2026-10-20 (entry 038) |
| Drone class | small drones | Decided by Mads, 2026-10-14 |
| Recall target | as high as can be reached; no fixed number | SUPERSEDED by R2 (95%), 2026-10-21 |

## Performance targets

| Requirement | Value | Status |
|---|---|---|
| Primary figures | Recall (R2), false-positive rate (R3) and precision (R4) at one threshold frozen on validation, on sources or rigs never trained on, with the drone share of the test audio | PROPOSED (entry 039) |
| Primary metric (range-free) | Recall on all drone windows of a rig never trained on, at 10% window false alarms on that rig's own no-drone audio; worse rig decides; interval from resampling flights | SUPERSEDED by R2 and R3, 2026-10-21; still reported so iterations 3 to 5 stay comparable |
| Primary metric (at 100-200 m) | Recall on drone windows at 100-200 m slant range, on a rig never trained on, at 10% window false alarms on that rig's own no-drone audio; worse rig decides; interval from resampling flights | WITHDRAWN by Mads, 2026-10-20 (entry 038); was PROPOSED (council, entry 025) |
| Deployment figure | Held-out recall on all drone windows, and the false alarms it costs, at a threshold frozen at 10% false alarms on validation | SUPERSEDED by the primary figures above, 2026-10-21 |
| Time to detection | Share of drone moments followed by an alarm within 5 s (window length included); no target yet | SUPERSEDED by R5, 2026-10-21; still reported |
| Primary metric (2026-10-10) | False-alarm rate at 95% drone recall, on held-out test sources | SUPERSEDED (2026-10-14); in substance R2 and R3 restore it |
| Primary target (2026-10-10) | at most 5% false alarms at 95% recall | SUPERSEDED (2026-10-14); in substance R2 and R3 restore it |
| Accuracy (balanced, held-out sources) | at least 90% | PROPOSED; reported, secondary |
| ROC AUC (held-out sources) | at least 0.95 | PROPOSED; reported, secondary |
| False alarms on helicopter clips (Halmstad) | reported separately, no target yet | PROPOSED |
| Must beat baselines `comb` and `lgbm` | yes | PROPOSED |

Accuracy is listed because it is easy to read, but it is the secondary
number: on a balanced laboratory split it looks high for any model. The
held-out-source false-alarm rate is the one that predicts field behaviour.

## Signal settings

| Setting | Value | Status |
|---|---|---|
| Sample rate | 16 kHz mono | PROPOSED (currently fixed in `src/dad/config.py`) |
| Test WAV files | mono, "decent quality" | Decided by Mads, 2026-10-21; exact format PROPOSED in entry 039 |
| Window length | 1.0 s | FREE within 0.25 to 2.0 s |
| Hop | 0.125 s | FREE within 0.05 to 0.5 s |
| Features | 64 log-mel bands, 50 to 8000 Hz | FREE |

## Latency and deployment

| Requirement | Value | Status |
|---|---|---|
| Deployment hardware | not decided (laptop, Raspberry Pi or microcontroller) | OPEN |
| Deployment on a microcontroller | out of scope | Decided by Mads, 2026-10-26 (entry 060) |
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
| Lund in a cloud tool | Lund may be used for research, including upload to the private Edge Impulse project; publishing results or audio still needs the authors' permission | Decided by Mads, 2026-10-26 (entry 058) |
| Training and test rigs from iteration 3 | leave one rig out between Nightjar and d3set; Lund and Glasgow scored as further unseen rigs (entry 025) | PROPOSED |
| Sealed test for recall by range | ITU ARIS and RODENT (fetched 2026-10-14 on Mads's go-ahead), scored only for a finished candidate | PROPOSED |

## Stop condition

Stop iterating and report when every Must and Should requirement (R1, R2,
R3, R5) is met on the held-out test sources, or when the report shows it
cannot be met with the available data.

## Notes and wishes

Write anything else here: things to try, things to avoid, deadlines.

- 2026-10-21: the real test is coming up (Mads). Test audio will be WAV
  files, mono, decent quality.
