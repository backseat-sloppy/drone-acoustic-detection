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

## Requirements

Stated by Nicolai on 2026-11-09, for Mads and himself (report entry 149). They
replace everything deleted on the same day (entry 148; the previous version is
in git at commit e4c78806ec10574b8071f065f166659aa03b24d0).

**When anyone on the project asks about the requirements, first establish which
of the three tests below they mean, and explain this section to them.**

Performance is judged in confidence, under specific, methodical conditions:

- With a drone, confidence must be as high as possible.
- Without a drone, confidence must be as low as possible.
- First priority: this holds for drone against the war soundscape.
- After that: this holds with distractors as well.

There are no numeric targets. "High confidence" is deliberately not defined.
The same model is used in all three tests.

| Test | What it tests | What matters |
|---|---|---|
| 1. Real drone test: Parameters | The 35 real drones from ITU ARIS | Confidence when there is a drone, and how little confidence when there is none; then the same with distractors mixed in (confidence stays low even with distractors). **The most important requirements of the whole project.** |
| 2. Modified drone: Time before impact | How long before a drone hits you confidence is high | Confidence as high as possible 3 to 7 s before impact |
| 3. Natural environment simulation | The product as a demo in the surround sound speaker array | How it behaves outside a virtual test |

This section is the project's Final Problem Statement.

Details, answered by Nicolai on 2026-11-09 (report entry 150):

| Topic | Decision |
|---|---|
| Ranking models | First: confidence with a drone. Second: confidence without a drone, with distractors |
| Order of the backgrounds | Main parameter: war soundscape. Secondary: wind. Tertiary: cars, helicopter, insects |
| ITU ARIS | Only for scoring finished models, never for training or tuning. It was read during development (entry 129), and results on it say so |
| Test 1 levels | War soundscape fixed at 90 dB SPL; drone at 100, 90, 80, 70 and 60 dB SPL (+10 to -30 dB) |
| Test 2 speeds | Chosen by the tester, between 50 and 200 km/h |
| Test 2 figure | Mean confidence over 3 to 7 s before impact, every second counted equally, with the curve over time beside it |
| Test 3 | Not specified yet; waiting for the specifications from Mathias |
| Model | The it9 CRNN is the default. More models are being trained: when one arrives, propose integrating it in `Complete_Test_Gui` |

## Notes and wishes

Write anything else here: things to try, things to avoid, deadlines.

- 2026-10-21: the real test is coming up (Mads). Test audio will be WAV
  files, mono, decent quality.
