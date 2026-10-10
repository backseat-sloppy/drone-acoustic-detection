# Project report log

The running record of everything done in this project, in order. It is
append-only: entries are never rewritten. If an entry turns out to be wrong, a
later entry corrects it and refers back.

## Entry template

```
## NNN - Short title
- Date: YYYY-MM-DD
- Type: Data | Preprocessing | Model | Algorithm | Tuning | Evaluation |
        Council | Requirements | Infrastructure | Correction
- Commit: <hash, or "uncommitted">

**What changed.** What was done, concretely.
**Why.** The hypothesis or the reason.
**How.** Commands, files and settings needed to repeat it.
**Result.** Numbers with their conditions: split, sources, class balance,
window, hop. Baselines alongside. "Not run" or "not verified" where true.
**Assessment.** What the result means against REQUIREMENTS.md.
**Next.** The decision taken and the next step.
```

---

## 001 - Dataset selection

- Date: 2026-10-09
- Type: Data
- Commit: uncommitted

**What changed.** Public drone-audio datasets were surveyed and four were
selected: DDL (University of Surrey, real recordings only), DroneNoise
(University of Salford), the Halmstad multi-sensor dataset, and FSD50K as a
source of negatives.

**Why.** The pipeline had only been run on synthetic audio from
`src/dad/synth.py`, whose accuracy numbers are meaningless by construction.
Supervised training needs real, labelled recordings. A dataset was accepted
only if it has real rather than synthetic or noise-augmented audio,
documented provenance, per-sample labels or metadata, and a clear licence.
Localisation was ruled out as a goal, so datasets whose value was position
ground truth were dropped.

**How.** Web survey. The accepted and rejected datasets, with reasons, are
listed in `docs/datasets.md`.

**Result.** Four datasets accepted, nine considered and left out. No audio had
been inspected at this point; all properties were taken from the hosts'
descriptions and the papers.

**Assessment.** The known weakness is negatives. DDL has only two noise-only
sessions recorded through the same microphone chain as its drone clips, and
no dataset was recorded with a deployment microphone at a deployment-like
site.

**Next.** Download the four datasets unmodified.

## 002 - Dataset acquisition

- Date: 2026-10-10
- Type: Data
- Commit: uncommitted

**What changed.** The four datasets were downloaded into `data/`, 38.4 GB in
total, and a provenance document was written.

**Why.** To have the raw data available, exactly as published, before any
processing decision is made.

**How.** `scripts/download_datasets.ps1`, which fetches byte ranges over
parallel connections with `curl.exe`, resumes interrupted runs, and gives a
file its final name only after its MD5 matches the checksum published by the
host. The host's API record is saved in each dataset folder as
`_source_record.json`.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\download_datasets.ps1
```

**Result.**

| Folder | Files | Size | Checksums |
|---|---|---|---|
| `data/DDL_Surrey/` | `MLSP_2022_Real_Data.zip` | 12.63 GB | MD5 verified |
| `data/DroneNoise_Salford/` | 173 WAV, 1 XLSX | 0.74 GB | MD5 verified |
| `data/Halmstad/` | 1 zip | 0.31 GB | MD5 verified |
| `data/FSD50K/` | 11 files (split zip, metadata, labels, docs) | 24.68 GB | MD5 verified |

Nothing was unzipped or modified. The DDL synthesized zip was deliberately
not downloaded. figshare lists 175 DroneNoise entries because one WAV was
uploaded twice with an identical MD5; 174 unique files exist. A copy of
`data/` was also placed on Google Drive, verified by file count and byte
size only.

**Assessment.** The data is in place and its integrity is established. Its
contents are not: sample rates, clip lengths and the Halmstad licence are
still unverified (see "Check before use" in `docs/datasets.md`).

**Next.** Inspect the archives before changing anything, then build a single
manifest across the four sources.

## 003 - Working agreement established

- Date: 2026-10-10
- Type: Infrastructure
- Commit: uncommitted

**What changed.** Added `CLAUDE.md` (the working rules), `REQUIREMENTS.md`
(targets and adjustable settings, owned by Mads), this report log, and a
project-local `council` skill in `.claude/skills/council/`.

**Why.** So the same process applies in every session and on every machine:
every step is logged here, big decisions go through a council review, and
iterations are judged against written requirements.

**How.** See the four files. They are plain text in the repo and travel with
it through git.

**Result.** Not applicable.

**Assessment.** Most values in `REQUIREMENTS.md` are marked PROPOSED: they
are defaults taken from the existing pipeline and from the dataset survey,
not yet confirmed. Deployment hardware and latency targets are open. The
council skill is a new stand-in written for this repo; it is not the
`/council` command used earlier on the other machine, which was not
available here.

**Next.** Mads reviews `REQUIREMENTS.md`. Install the Python toolchain
(`uv`), then step 1 of data preparation: inspect the archives.

## 004 - Council roles redefined

- Date: 2026-10-10
- Type: Infrastructure
- Commit: uncommitted

**What changed.** The three council members in
`.claude/skills/council/SKILL.md` were replaced. Entry 003 set them up as
three reviewers with different subject lenses (methodology, signal and model,
requirements). They are now three roles specified by Mads:

- **Devil's Advocate**: sees the project context and argues against the
  proposal.
- **Scholar**: sees no project context and states best practice for the
  class of problem.
- **Problem Solver**: sees the project context and looks for the most
  practical way to reach the goal.

**Why.** Mads's reasoning: critical thinking is what produces the best
outcome, so one member must be dedicated to arguing against; one must hold to
best practice without being coloured by the project; and one must work from
the actual situation with a problem-solving view.

**How.** The skill now writes two briefs, a full one and a context-free one
for the Scholar. The weighing rules changed accordingly: the Devil's
Advocate's opposition is treated as arguments to be answered rather than a
vote, and any departure from what the Scholar calls best practice must be
justified in the report.

**Result.** Not applicable. The council has not yet been convened.

**Assessment.** "Without context" was implemented as: the Scholar is given
only a generic statement of the question, and none of our results, current
approach or preferred option. This is an interpretation of the instruction
and can be relaxed if the Scholar's answers turn out too general to use.

**Next.** Unchanged from entry 003.

## 005 - Python environment on the second machine

- Date: 2026-10-10
- Type: Infrastructure
- Commit: uncommitted

**What changed.** Installed `uv` 0.13.0 (via winget) and created the project
environment: CPython 3.12.15 and the locked dependencies.

**Why.** The machine had no working Python, and the repo is managed with `uv`.

**How.**

```powershell
winget install --id astral-sh.uv -e
uv sync --group dev
uv run pytest
```

Machine: AMD Ryzen 5 3600 (6 cores, 12 threads), 16 GB RAM, AMD Radeon
RX 5700 (8 GB), Windows 11.

**Result.** 12 tests passed, 1 skipped (the torch-dependent test; torch was
not yet installed at this point).

**Assessment.** Two machine facts that affect later steps:

- The GPU is AMD, so PyTorch's CUDA backend is not available. Whether training
  can use the GPU at all is open and is tested separately.
- When `uv` is run from inside the Claude desktop app on Windows, its default
  Python directory under `%APPDATA%` is redirected by the app's sandbox and
  the install fails ("Missing expected target directory for Python minor
  version link"). Setting `UV_PYTHON_INSTALL_DIR` and `UV_CACHE_DIR` to a
  folder outside `AppData` fixes it. A normal terminal is not affected.

**Next.** Inspect the raw archives.

## 006 - Inspection of the raw archives

- Date: 2026-10-10
- Type: Data
- Commit: uncommitted

**What changed.** Nothing in the data. Every archive was read in place to
establish what it really contains, before any processing decision.

**Why.** Entries 001 and 002 relied on the hosts' descriptions. Those had
known inconsistencies (clip length, sample rates, the Halmstad licence).

**How.** `uv run python -I scripts/inspect_raw.py`, plus one-off scans of
every DDL file (size, MD5 of the audio payload, per-channel level) and of the
FSD50K metadata.

**Result.**

*DDL (Surrey).* 62,103 WAV files, 96 kHz, 8 channels, 32-bit PCM, 19.08 GB
uncompressed, in 36 session folders. Several things differ from the host's
description:

| Host says | Files show |
|---|---|
| Classes MINI, PRO4 and XXXX (no drone) | MINI 37,249 and PRO4 24,854. **No XXXX file exists.** |
| 13-character timestamp, separate flight and recording session IDs | 14-character timestamp, one session ID (`DDMMYY-Tnnn`) |
| Recorded 2022 (file name `MLSP_2022`) | Timestamps are 2021-03-29 and 2021-03-31 |
| 100 ms clips | 58,126 files are exactly 100 ms. 3,945 are header-only with no audio. 14 hold between 0.2 s and 79.8 s. 7 are shorter than 100 ms. |

- Sessions: 2021-03-29 has 5 sessions, all DJI Mini 2. 2021-03-31 has 13
  DJI Phantom 4 Pro sessions and 18 DJI Mini 2 sessions.
- **Duplicates.** 2,629 files are byte-identical in audio to an earlier file
  in the same session, but carry different labels. In session `290321-T004`
  the second half (2,009 files) repeats the first half, labelled 48 m range
  where the first half says 24 m. In `290321-T005`, 620 files repeat.
- **Long files and empties.** In 10 sessions one or two files hold many
  seconds of audio under a single label, and the session ends with a run of
  empty files of about the same total length. The audio is all there and in
  order, but from the long file onward the timestamp, range and bearing in
  the file names no longer line up with the audio.
- Consecutive files are continuous audio: the sample jump across a file
  boundary (median 3.4e-5 of full scale) is far below the signal level
  (5.1e-3).
- 11 files encode a negative altitude as `0-1` or `0-0`.
- Channels: between 200 Hz and 2 kHz all eight agree within about 3 dB. Below
  80 Hz they differ by up to 12 dB on 2021-03-29 and by under 2 dB on
  2021-03-31, which points to wind on the capsules on the first day.

*DroneNoise (Salford).* 173 WAV files, 50 kHz, mono, 32-bit float. 9 are
calibration signals. The other 164 are 20 distinct flight events, each
recorded by up to 9 microphones: 14 flyovers at 15 m/s and 6 hovers, all at
10 m above ground. Durations 20 s (137 files) or 30 s (27 files), a few
shorter. The file-name codes are explained by the spreadsheet in the dataset:
drone, height, operation and speed, payload, start side, wind direction,
event, microphone. The unexplained code `Fp` has a gross weight of 795 g in
the spreadsheet; the model is still not named.

*Halmstad.* 90 WAV files, 44.1 kHz, stereo, 16-bit, 10 s each: 30 drone,
30 helicopter, 30 background. The `LICENSE` file inside the zip is CC0 1.0,
which settles the open licence question. The zip also holds 652 video files
that are not used.

*FSD50K.* 51,197 clips. Licences in dev / eval: CC BY 20,017 / 3,489,
CC0 14,959 / 4,914, CC BY-NC 4,616 / 1,425, Sampling+ 1,374 / 403. The dev
set comes with its own train/val division. 199 clips mention "drone" or a
related word in title, description or tags; those inspected were the musical
sense of the word (harp, didgeridoo). The audio is a multi-disk zip64
archive, which the Python standard library refuses to open.

**Assessment.** Two findings change the plan.

1. **DDL contains no negatives.** Entry 001 assumed two noise-only sessions
   from the same microphone chain. They are not in the archive. Every
   positive for training therefore comes from one recording setup, and every
   negative from a different one. A model can then separate the classes by
   recording chain instead of by drone. This is now the main risk to any
   accuracy number.
2. **The range labels in DDL are only partly trustworthy.** Detection labels
   are unaffected (every file is a drone), but any detection-versus-distance
   analysis must exclude the duplicated and post-long-file stretches.

**Next.** Process all four sources into one format, removing the empty and
duplicated DDL files and flagging the unreliable range labels.

## 007 - Preprocessing into a uniform store

- Date: 2026-10-10
- Type: Preprocessing
- Commit: uncommitted

**What changed.** All four sources were converted to 16 kHz mono and written
to `data/processed/` with one manifest. The raw downloads are untouched.

**Why.** The frontend contract of the pipeline is 16 kHz mono
(`src/dad/config.py`), and training needs every source addressable the same
way, with its label, group and licence.

**How.**

```powershell
7z x data\FSD50K\FSD50K.dev_audio.zip  -odata\extracted\FSD50K
7z x data\FSD50K\FSD50K.eval_audio.zip -odata\extracted\FSD50K
uv run python -I scripts/prepare_data.py
```

Decisions, all in `scripts/prepare_data.py`:

- **Format only.** Channel selection, resampling, and joining. No gain
  change, normalisation, filtering or mixing; those belong in training where
  they can be varied and logged.
- **Resampling:** polyphase (`scipy.signal.resample_poly`) from 96, 50 and
  44.1 kHz to 16 kHz.
- **Mono:** channel 0 everywhere. For DDL, channel 0 is the one with the
  least wind noise (entry 006). For Halmstad it is the left channel.
- **DDL joined back into recordings.** Files are concatenated per session in
  sequence order before resampling, so there are no filter edge effects
  every 100 ms and windows of any length can be cut later. Empty files and
  exact duplicates (first copy kept) are dropped.
- **DDL label reliability.** `ddl_clips.csv` has one row per original file
  with where it landed and a `range_reliable` flag, which is off for the kept
  copy of every duplicated file and for everything from a long file onward.
- **DroneNoise:** calibration files skipped. All microphones kept; the
  manifest group is the flight event, so the nine views of one event can
  never be split apart.
- **FSD50K:** only CC0 and CC BY clips. Every clip matching the drone word
  list is left out. Manifest group is the Freesound uploader.
- **Storage:** float32 for DDL and DroneNoise (their originals are 32-bit and
  DDL is very quiet), int16 for Halmstad and FSD50K (16-bit originals).

**Result.**

| Source | Recordings | Audio | Label | Notes |
|---|---|---|---|---|
| DDL | 36 sessions | 99.1 min | drone | 55,529 of 62,103 files kept; 3,945 empty and 2,629 duplicate dropped |
| DroneNoise | 164 (20 events) | 59.2 min | drone | about 6.6 min per microphone |
| Halmstad | 90 | 15.0 min | 30 drone, 30 helicopter, 30 background | |
| FSD50K | 43,200 | 89.3 h | not drone | train 31,415, val 3,400, eval 8,385; 7,818 excluded by licence, 179 by keyword |

Range labels are flagged reliable for 46,937 of the 55,529 kept DDL files.
No discontinuity was detected at any of the DDL file junctions.

**Assessment.** Positives total about 2.9 hours, of which only 99 minutes are
available for training if DroneNoise and Halmstad are held out. Negatives
outnumber positives by more than 30 to 1 in duration.

**Next.** Quality checks on the processed audio.

## 008 - Quality checks on the processed data

- Date: 2026-10-10
- Type: Data
- Commit: uncommitted

**What changed.** Nothing. Checks only.

**Why.** To find properties that would let a model separate the sources
without hearing a drone, and to confirm there is no shared audio across
groups that will later be split.

**How.** `uv run python scripts/qa_processed.py`

**Result.**

*Level.* Recording RMS in dBFS, median (5th to 95th percentile):

| Source | Level |
|---|---|
| DDL 2021-03-29 | -37.0 (-37.4 to -31.3) |
| DDL 2021-03-31 | -56.5 (-60.4 to -51.0) |
| DroneNoise | -30.3 (-38.5 to -21.0) |
| FSD50K | -26.7 (-47.7 to -13.3) |
| Halmstad background / drone / helicopter | -31.7 / -25.0 / -19.2 |

*Shared audio between DDL sessions.* 24 session pairs lie close in time. Each
was searched in both directions with 2 s probes every 20 s. The highest
normalised cross-correlation was 0.67, against 1.0 for a self-test of the
method and a 0.9 threshold. High-passing at 1.5 kHz brought the three highest
pairs to 0.20 to 0.49. No shared audio was found. The remaining correlation
is attributed to stationary tones common to the sessions and was not
investigated further.

*DroneNoise, how long the drone is loud.* Per-second level relative to the
loudest second of the same file, microphone M5: in flyovers 56% of seconds
are within 10 dB of the peak and 81% within 20 dB; in hovers 69% and 99%.

*Tonal content.* In the files examined, narrow spectral peaks stand 5 to
17 dB above the broadband level in DDL, against 10 to 25 dB in DroneNoise
and Halmstad drone clips. Some DDL sessions show lines at 100, 200 and
300 Hz, and three Mini 2 sessions from 2021-03-31 share a line near 1176 Hz.
The origin of these lines was not established.

**Assessment.**

- **Level is a shortcut.** Two thirds of the DDL audio sits about 30 dB below
  typical FSD50K clips, and the log-mel features of the frontend keep
  absolute level. Unless level is removed or randomised in training, "quiet"
  predicts "drone".
- **DDL drones are faint.** At 20 to 250 m the drone is often only a few dB
  above the background, so some DDL windows labelled drone may contain almost
  no audible drone. This limits reachable recall on DDL and is a reason to
  report results by range.
- **DroneNoise flyover edges are weak positives.** About a fifth of flyover
  seconds are more than 20 dB below the pass-by peak.
- The mains-related lines in DDL are a possible recording-chain fingerprint.

**Next.** Decide the training and evaluation design in light of entry 006,
finding 1. This is a change to the evaluation protocol, so it goes to the
council first.

## 009 - Training throughput, data reader, and corrections

- Date: 2026-10-10
- Type: Infrastructure
- Commit: uncommitted

**What changed.**

- Installed PyTorch into the project environment (`uv sync --group dev
  --extra dl`): torch 2.14.1, CPU build.
- Measured training speed on the CPU and, in a separate throwaway
  environment, on the AMD GPU through DirectML (`torch-directml` 0.2.5 with
  torch 2.4.1).
- Added `src/dad/realdata.py`, which reads `data/processed/` and returns
  recordings unchanged, and `tests/test_realdata.py`.
- Updated `docs/datasets.md` with what entries 006 to 008 established.

**Why.** To know before designing training whether the GPU can be used and
how long an epoch takes, and to have one tested way into the processed data.

**How.** One optimiser step (AdamW, batch 64, input 64 mel bands by 98
frames, random data) timed for each architecture in `src/dad/models/zoo.py`.

**Result.**

| Model | Parameters | CPU, windows/s | GPU via DirectML, windows/s |
|---|---|---|---|
| dscnn | 22,530 | 76 | 722 |
| resnet | 695,010 | 26 | 341 |
| crnn | 739,747 | 51 | fails: GRU not supported by the backend |
| transformer | 826,626 | 283 | 712 |

Log-mel extraction takes 0.68 ms per one-second window on one core. The test
suite passes: 33 tests.

FSD50K facts needed for splitting: the train/val division shipped with the
dev set shares 774 uploaders, so it is not leak-free; dev and eval share
none. 5,304 of the 43,200 processed clips are shorter than one second.

**Assessment.** The GPU is usable for three of the four neural models and is
about 10 times faster than the CPU for the two CNNs. DirectML needs an older
torch (2.4.1) than the project lock (2.14.1), so using it means a second
environment or a change to the lock. The CRNN would have to train on the CPU
at about 50 windows per second. These are synthetic-input timings; a real
epoch adds data loading and feature extraction.

**Corrections to earlier entries.**

- Entry 006 gave the 2021-03-31 sessions as 13 Phantom 4 Pro and 18 Mini 2.
  The manifest shows 12 Phantom 4 Pro (41.4 min) and 19 Mini 2 (25.6 min).
  The 2021-03-29 figure (5 Mini 2 sessions, 32.1 min) stands.
- Entry 002 said the Halmstad licence was unverified. It is CC0 1.0
  (entry 006).

**Next.** Await the council on the training and evaluation design.

## 010 - Council: training and evaluation design

- Date: 2026-10-10
- Type: Council
- Commit: uncommitted

**Question.** How should training and evaluation be designed, given that
every training positive comes from one recording chain (DDL) and every
negative from other chains (FSD50K)?

The proposal put to the council: remove level as a cue; mix real drone
windows with real FSD50K backgrounds at random SNR; move DroneNoise into
training as a second positive chain; validate on held-out DDL sessions; test
on Halmstad and FSD50K-eval at a threshold frozen on validation; report
same-chain AUC on Halmstad next to cross-chain AUC every run. Alternatives:
A, the original plan with only level normalisation; B, the proposal without
DroneNoise in training; D, no training until same-chain negatives exist.

**Devil's Advocate** (saw the project; argued against). Its objections, by
its own ranking, and what became of each:

| Objection | Rank | Outcome |
|---|---|---|
| The only test with positives would be 30 Halmstad clips, read every run, under a stop rule tied to test results. Best-of-ten scorings would inflate AUC by about 0.045. | Decisive | **Accepted.** DroneNoise stays out of training. Halmstad and FSD50K-eval are sealed and scored only for a finished iteration, every scoring counted. |
| Mixing does not remove the confound: every positive still carries the DDL chain and no negative does. After waveform RMS normalisation the lowest mel band alone separates DDL from FSD50K at AUC 0.79 and temporal flatness at 0.76. | Decisive | **Stands as a risk.** It cannot be argued away without same-chain negatives. It is made testable: see the gate below. |
| Level is not removed by RMS normalisation, because the DDL level is dominated by wind below 50 Hz that the mel bank never sees. | part of the above | **Accepted and verified.** In four of five day-1 sessions the total level is 15 to 16 dB above the level over 50 Hz. Level is normalised in-band instead. |
| Validation is the confounded contrast, and selecting the best epoch by validation AUC rewards the shortcut. | Serious | **Accepted.** Design choices are judged on transfer to DroneNoise, not on validation AUC. |
| A threshold set for 95% recall on faint DDL does not transfer to loud Halmstad clips. | Serious | **Partly answered.** The threshold is set on validation positives with reliable range up to 100 m, and full curves are reported next to the frozen point. |
| Moving DroneNoise into training destroys the better independent probe. | Serious | **Accepted.** |
| Halmstad is easy and its single-chain premise unverified; two background clips are identical. | Serious | **Verified and accepted.** `BACKGROUND_019` is a copy of `BACKGROUND_010` and is now dropped (29 background clips). Halmstad is reported at clip level with intervals and without claims it cannot support. |
| The proposal changes four PROPOSED requirements without Mads. | Serious | **Accepted.** The changes are put to Mads before any training. |
| FSD50K-eval is the same contrast as training; padding and digital silence occur only in negatives. | Weak | False alarms are reported per kind; clips under 1 s are dropped rather than padded; silent windows are skipped. |

**Scholar** (saw no project context). Best practice as stated, with 21
sources it reported locating in this session:

- A label aligned with recording source is the textbook condition for
  shortcut learning; the documented analogue is COVID X-ray classifiers that
  learned the hospital (DeGrave et al. 2021, Nature Machine Intelligence).
- The preferred remedy is data that breaks the confound. Failing that:
  mixing targets into varied backgrounds at controlled SNR, device and
  filter augmentation, and a level-robust frontend such as PCEN
  (Lostanlen et al. 2019; Stowell 2022).
- Evaluation: split by setup or session; choose model and threshold on
  non-test data; report precision-recall next to ROC; use Wilson intervals;
  treat adjacent windows as dependent, so the effective sample size is
  nearer the number of recordings; consult the test set once after freezing.
- Verdicts: A is against practice. B is the standard remedy. C is defensible
  but leaves 5 minutes of positives as the sole evidence. D is the cleanest
  remedy, but refusing to train meanwhile is stricter than field practice.
- From 30 clips per class one can claim "no gross failure in one further
  setup", not an error rate and not a ranking of similar models.

Departures from the Scholar's practice: one. In-band level normalisation is
used in place of PCEN for the first iteration, because it leaves the frozen
frontend otherwise unchanged and its effect can be measured on its own.
PCEN is the next step if the gate fails.

**Problem Solver** (saw the project). Verdict: try a modified version.

- A is ruled out by the data: recording level alone separates DDL from
  FSD50K-dev at AUC 0.956, and gives 0.58 on the other sources (verified).
- Run the "DroneNoise held out" comparison first, as a gate.
- Do not score Halmstad every run.
- Mix DDL only at +5 to +20 dB, since a DDL window is already mostly site
  background.
- Proposed experiment: the small CNN trained twice on DDL versus
  FSD50K-dev, once with level normalisation only and once with full shortcut
  suppression, each scored on DroneNoise against uploader-disjoint FSD50K.

**Agreement and dissent.** All three reject A. All three keep DroneNoise out
of training for now. The Scholar and the Problem Solver judge B worth
running; the Devil's Advocate holds that mixing cannot deconfound. That
disagreement turns on one fact nobody has measured: whether a model trained
this way fires on drones recorded by a different chain. The gate measures
exactly that.

**Decision.** Option B, run as a gated experiment.

- *Training:* DDL training sessions (about 79 min) against FSD50K-dev,
  sampled by uploader. In-band level normalisation. Mixing: DDL windows with
  FSD50K backgrounds at +5 to +20 dB; half the negatives are sums of two
  FSD50K windows.
- *Validation (used freely):* 8 held-out DDL sessions (about 20 min, covering
  both days, both models, near and far) and the 1,514 FSD50K-dev clips from
  1,098 uploaders absent from training (3.9 h).
- *Development probe (used freely, and therefore not a test):* DroneNoise,
  as transfer to an unseen chain and four unseen drone models; and DDL
  validation score against range.
- *Sealed test:* Halmstad (30 drone, 30 helicopter, 29 background clips) and
  FSD50K-eval (22.3 h). Scored only for a finished iteration, at clip level
  with Wilson intervals, each scoring counted in this log.
- *Threshold:* 95% recall on validation positives with reliable range up to
  100 m, then frozen. The requirement's "false alarms at 95% recall" is also
  reported per test source, labelled as using that source's own threshold.
- *Gate:* level rule, comb, LightGBM, and the small CNN twice (level
  normalisation only; full suppression). Passed if the full-suppression CNN
  reaches DroneNoise transfer AUC of at least 0.90, clearly above the
  level-only run and LightGBM, within 0.05 of its DDL validation AUC, and
  its DDL scores fall with range. Failed if validation AUC is near 0.99 with
  transfer under 0.8, or if suppression does not beat level-only. On
  failure: spectral whitening or PCEN in the frontend, or option D.

**Consequences for the requirements,** for Mads to decide:

1. "Only real recordings, no synthetic training data" must be read to allow
   sums of two real recordings, or the gate cannot run as designed.
2. The primary target, at most 5% false alarms at 95% recall on held-out
   sources, cannot be demonstrated on Halmstad: zero false alarms in 59
   clips bounds the rate only at about 5 to 6%. FSD50K-eval can support a
   false-alarm rate; DroneNoise and Halmstad can support recall. No
   available source supports both from one chain at that precision.
3. Whether DroneNoise later joins the training positives is decided after
   the gate, and would move it out of the held-out list.

**Next.** Present the pipeline to Mads and get the two decisions needed to
start (mixing; using the GPU through DirectML).
