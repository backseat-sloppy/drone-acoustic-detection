# Project report log

The running record of everything done in this project, in order. It is
append-only: entries are never rewritten. If an entry turns out to be wrong, a
later entry corrects it and refers back.

## How this log is organised

The work is divided into **iterations**. An iteration is one attempt at the
problem: a stated change from the attempt before, the reason for it, the
experiments that test it, and a verdict. Each iteration has its own heading
(`# Iteration N`) and opens with a box that says what changed from the
previous iteration and why. Numbered entries sit inside iterations.

## Iterations at a glance

| | Entries | What was tried | Why it differed from the one before | Outcome |
|---|---|---|---|---|
| Phase 0 | 001 to 009 | Select, download, inspect and prepare the data | (start) | Data ready. DDL has no no-drone recordings, so label and recording chain are confounded. |
| Iteration 1 | 010 to 013 | Train on DDL against FSD50K with level removal, background mixing and random EQ | First design: the standard remedy for a confounded label | **Failed.** The model learned the DDL recording chain. Its score rose with distance to the drone. |
| Iteration 2 | 014 onward | Train only on drones that are loud in their own recording, always buried in a background; judge with a same-background probe | Iteration 1 showed the faint, distant drones teach the background, and that validation AUC cannot tell | In progress |

## Templates

```
# Iteration N - Short name

> **Changed from iteration N-1:** what is different.
> **Why:** the evidence from iteration N-1 that motivates it.
> **Kept the same:** what is deliberately not changed, so the comparison holds.
> **Hypothesis:** what should happen if the change is right.
> **Judged by:** the measurement and the numbers that decide it, fixed in advance.

## NNN - Short title
- Date: YYYY-MM-DD
- Iteration: N
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

The last entry of an iteration states its verdict and what the next iteration
will change because of it.

---

# Phase 0 - Setup and data

> Everything before the first model: choosing and downloading the datasets,
> finding out what is really in them, and converting them to one format.
> Entries 001 to 009. (This heading and the two iteration headings below
> were added on 2026-10-10 at Mads's request, to make the iterations visible.
> The entries themselves are unchanged.)

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

---

# Iteration 1 - Break the recording-chain shortcut by mixing

> **Changed from Phase 0:** nothing was trained before; this is the first
> design.
> **Why this design:** Phase 0 found that every training drone comes from one
> recording chain (DDL) and every non-drone from others (FSD50K), so a model
> can score well without hearing a drone. The standard remedy is to remove
> level as a cue and to mix each drone window with a real background.
> **Hypothesis:** with level removed and backgrounds mixed in, the model
> must rely on the drone, and will then also find drones recorded by another
> chain (DroneNoise).
> **Judged by:** the gate fixed in entry 010: transfer AUC to DroneNoise of at
> least 0.90, clearly above the same model without mixing, within 0.05 of its
> validation AUC, and a score that falls with distance to the drone.
> Entries 010 to 013.

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

## 011 - Requirements: mixing allowed, GPU through DirectML

- Date: 2026-10-10
- Type: Requirements
- Commit: uncommitted

**What changed.** Mads decided the two open points from entry 010:

1. Summing two real recordings is allowed as training augmentation. The row
   in `REQUIREMENTS.md` now reads: real recordings only; a drone window plus
   a real background is allowed; generated or rendered audio is not. Status
   LOCKED.
2. The AMD GPU may be used through DirectML, in a separate environment.

**Why.** Without mixing the gate experiment of entry 010 cannot run. Without
the GPU an iteration takes hours on the CPU (entry 009).

**How.** `REQUIREMENTS.md` edited as above. The GPU environment is created by
`scripts/setup_dml_env.ps1` into `.venv-dml` (git-ignored): torch 2.4.1 with
torch-directml 0.2.5. The main environment and `uv.lock` are unchanged.

**Result.** The DirectML environment reports the device as "AMD Radeon
RX 5700".

**Assessment.** The other consequences listed in entry 010 are still open:
the primary target cannot be demonstrated on Halmstad alone, and whether
DroneNoise later joins training.

**Next.** Implement training on the real data and run the gate.

## 012 - Training and evaluation code for real data

- Date: 2026-10-10
- Type: Infrastructure
- Commit: uncommitted

**What changed.** Added `src/dad/realrun.py`, `scripts/run_real.py` and
`tests/test_realrun.py`. The existing synthetic pipeline (`dad compare`) is
untouched.

**Why.** `src/dad/compare.py` is wired to synthetic clips, holds every
window in memory, and takes the 95%-recall threshold from the set being
scored. None of that fits the design of entry 010.

**How.** What the new code fixes in place, so that later entries only need
to name what they vary:

- **Splits** (`make_splits`). DDL: 27 sessions train (79.2 min), 9 sessions
  validation (19.9 min): `290321-T005`; `310321-T004`, `T009`, `T013`,
  `T015` (Phantom 4 Pro); `310321-T018`, `T023`, `T026`, `T032` (Mini 2).
  FSD50K: 27,140 training clips (58.0 h); 1,397 validation clips (3.9 h)
  from uploaders with no clip in training. Clips shorter than the window are
  dropped, not padded. Probe: DroneNoise microphone M5, 20 files. Sealed:
  Halmstad and FSD50K-eval.
- **Level** (`level_normalise`). Each window's log-mel has the log of its
  own mean mel power subtracted, which equals scaling the waveform to a fixed
  level inside the 50 Hz to 8 kHz band. Values are floored 50 dB below that
  mean, so digital silence (present only in the 16-bit sources) is not a cue.
- **Training windows** (`WindowSampler`). Drawn at random on the fly, half
  drone and half not. Drones uniformly over DDL training time. Backgrounds
  with each clip weighted by one over the square root of its uploader's clip
  count. Silent windows are redrawn.
- **Variants.** `level`: level normalisation only. `full`: in addition, 75%
  of drone windows are summed with a background at an in-band
  signal-to-noise ratio drawn from +5 to +20 dB; 50% of negatives are a sum
  of two backgrounds at -10 to +10 dB; every window gets a random spectral
  tilt of up to 6 dB across the band plus a smooth random shape of about
  2 dB.
- **Training loop** (`NetScorer`). AdamW, learning rate 2e-3 on a one-cycle
  schedule, weight decay 1e-2, batch 64, SpecAugment, a fixed budget of
  windows, final weights kept. No best-epoch selection, because the
  validation contrast is the confounded one (entry 010).
- **Evaluation** (`evaluate`). Validation and probe windows are fixed,
  non-overlapping and unmixed. The threshold is the 95%-recall point on
  validation drones with reliable range up to 100 m. Reported: validation
  AUC (the confounded contrast), transfer AUC and recall on DroneNoise
  against the same validation negatives, score by range, and false alarms
  by sound class. DroneNoise windows more than 20 dB below their file's
  loudest window are left out.
- **Sealed test** (`evaluate_sealed`). Only run with `--sealed`. Clip-level,
  with Wilson intervals; a clip counts as flagged when at least half its
  windows exceed the frozen threshold.

**Result.** A smoke run with a tiny budget (6,400 windows) completed end to
end on the GPU at 621 windows per second including feature extraction. Its
numbers are not results.

**Assessment.** One correction to entry 010: the DDL validation set is 9
sessions, not 8. The durations given there (about 79 and 20 minutes) were
right.

**Next.** Run the gate.

## 013 - Gate experiment: failed

- Date: 2026-10-10
- Type: Evaluation
- Commit: uncommitted

**What changed.** First training on real data: the gate defined in entry
010.

**Why.** To find out whether mixing and equalisation stop a model from
learning the DDL recording chain instead of the drone.

**How.**

```powershell
.venv-dml\Scripts\python.exe scripts\run_real.py --name gate `
    --models level comb lgbm dscnn --variants level full --device dml --windows 200000
```

Seed 0. Small CNN: 200,000 training windows, about 5 minutes each on the GPU
at 700 windows per second. LightGBM: 40,000 windows. Development windows,
all 1 s: 1,194 DDL validation (780 with reliable range up to 100 m), 3,679
FSD50K validation, 335 DroneNoise. Results in `artifacts/runs/gate.json`.
The sealed test sources were not scored.

**Result.**

| Scorer | Validation AUC (DDL vs FSD50K) | Transfer AUC (DroneNoise vs FSD50K) | DroneNoise recall at frozen threshold | Validation false alarms |
|---|---|---|---|---|
| Level rule (quieter = drone) | 0.900 | 0.692 | 0.44 | 0.248 |
| Comb, no ML | 0.460 | 0.587 | threshold collapses to zero | 1.000 |
| LightGBM, level only | 0.992 | 0.850 | 0.40 | 0.037 |
| Small CNN, level only | 0.995 | 0.874 | 0.31 | 0.022 |
| LightGBM, full suppression | 0.989 | 0.848 | 0.40 | 0.051 |
| Small CNN, full suppression | 0.991 | 0.850 | 0.44 | 0.064 |

Mean score of the small CNN on DDL validation windows, by range to the drone:

| Range | Windows | Level only | Full suppression | Comb score |
|---|---|---|---|---|
| 0 to 50 m | 256 | 0.97 | 0.76 | 16.7 |
| 50 to 100 m | 514 | 0.93 | 0.95 | 2.8 |
| 100 to 150 m | 26 | 0.94 | 0.97 | 0.0 |
| over 150 m | 196 | 0.96 | 0.97 | 0.5 |

DroneNoise recall at the frozen threshold, by drone model, small CNN with
full suppression: Matrice 300 RTK 0.01, Mini 3 Pro 0.48, Yuneec H520E 0.60,
the unnamed 795 g model 0.59.

During training the transfer AUC of the full-suppression CNN moved between
0.74 and 0.87 from one check to the next, while validation AUC stayed at
0.98 to 0.99.

**Assessment.** The gate failed on every criterion set in advance.

- Transfer AUC is 0.850, under the 0.90 required.
- Full suppression is no better than level normalisation alone (0.850
  against 0.874), and no better than LightGBM (0.848).
- Transfer is 0.14 below validation, against an allowed 0.05.
- **The score does not fall with range. With full suppression it rises:**
  the model is least confident at 0 to 50 m, where the drone is loudest and
  where the comb detector, which only responds to harmonic structure,
  responds strongly; and most confident beyond 150 m, where the comb finds
  nothing. That is the behaviour of a model recognising the DDL site and
  recording chain, not the drone. A nearby drone makes the window look
  *less* like the background it has learned.

This confirms the Devil's Advocate's second decisive objection from entry
010: mixing a faint drone recording with a background does not remove the
recording's own background, and that background is the consistent thing in
every positive.

Two further observations.

- The validation numbers (AUC 0.99, 2 to 6% false alarms) are exactly the
  flattering result the council warned about. Reported alone they would have
  met the secondary targets in `REQUIREMENTS.md`.
- The transfer AUC is itself not a clean measure. DroneNoise windows and
  FSD50K windows also come from different chains, so some of the 0.85 may be
  "continuous outdoor recording against short event clip" and not drone
  detection. A probe is needed in which positive and negative share the same
  background.
- The comb baseline is not a usable floor here. It needs a clearly audible
  harmonic stack, which DDL has only at short range, and FSD50K is full of
  harmonic sounds (speech, music).

**Next.** Do not scale up to larger models; the problem is the training
data, not capacity. Build a probe where the background is identical in both
classes (a drone added to a real background at a known signal-to-noise
ratio, against that same background alone), apply it to the gate models,
then decide how the training positives should change.

**Verdict on iteration 1: failed.** Mixing a faint drone recording into a
background does not hide the recording's own background. Iteration 2 changes
which drone audio is used for training, and how success is measured.

---

# Iteration 2 - Train on audible drones, judge on identical backgrounds

> **Changed from iteration 1:**
> 1. *Training drones.* Iteration 1 used every DDL window, out to 250 m.
>    Iteration 2 uses only windows where the drone is loud in its own
>    recording: DDL within 50 m, and DroneNoise near the pass-by.
> 2. *Mixing.* Iteration 1 mixed 75% of drone windows at +5 to +20 dB.
>    Iteration 2 buries every drone window in a background, down to 5 dB
>    below it.
> 3. *How a model is judged.* Iteration 1 relied on validation AUC and on
>    transfer AUC to DroneNoise. Iteration 2 adds a paired probe: the same
>    background window scored with and without a drone added.
>
> **Why:**
> 1. In iteration 1 the score rose with distance. Beyond about 100 m the
>    harmonic detector finds nothing in DDL, so those windows are site and
>    recording chain with a drone label. They taught the model the
>    background. A window at 20 m is mostly drone.
> 2. A drone that is always under a different background cannot be found by
>    its own background.
> 3. Validation AUC was 0.99 for a model that had learned the wrong thing,
>    and transfer AUC compares two different chains as well. A probe whose
>    two sides share the background cannot be passed by recognising a chain.
>
> **Kept the same:** the frontend (16 kHz, 1 s window, 64 log-mel bands),
> in-band level normalisation, the FSD50K negatives and their split, the
> small CNN and LightGBM, the training budget (200,000 windows), the seed,
> the validation sessions, and the sealed test sources (still unscored).
>
> **Hypothesis:** a model trained this way raises its score when a drone is
> added to a background, does not raise it when DDL's own background or
> another ordinary sound is added, and scores DDL windows lower the farther
> the drone is.
>
> **Judged by:** the paired probe on drones from a chain the model was not
> trained on, against two controls (see entry 014).

## 014 - A probe the recording chain cannot pass

- Date: 2026-10-10
- Iteration: 2
- Type: Evaluation
- Commit: uncommitted

**What changed.** Added `paired_probe` to `src/dad/realrun.py` and to every
evaluation. No model was retrained for this entry; the iteration 1 models
were scored again.

**Why.** Entry 013 ended with two measures that cannot be trusted alone.
Validation AUC is the confounded contrast. Transfer AUC to DroneNoise also
compares two chains. Neither can say how much of a score is the drone.

**How.** 1,500 FSD50K validation windows are each scored twice: alone, and
with a real drone window added at an in-band signal-to-noise ratio of +10,
0 and -10 dB. The two members of a pair share the background, its recording
chain and its level. The figure reported is the AUC of "with drone" against
"alone": 0.5 means the added sound made no difference, 1.0 means it always
raised the score.

Four things are added in turn:

| Added sound | What a rise means |
|---|---|
| DroneNoise drone | The model responds to a drone from a chain it never trained on. The measure that counts. |
| DDL at up to 50 m | Drone plus the DDL chain. Cannot separate the two. |
| DDL at 150 m and beyond (control) | Mostly DDL site and chain. A rise here is the chain. |
| Another FSD50K window (control) | An ordinary sound. Should be about 0.5. |

```powershell
.venv-dml\Scripts\python.exe scripts\run_real.py --name gate_paired --load-from gate `
    --models level comb lgbm dscnn --variants level full --device dml
```

**Result.** Paired AUC at +10 / 0 / -10 dB, iteration 1 models:

| Model | DroneNoise drone | DDL up to 50 m | DDL beyond 150 m (control) | Other background (control) |
|---|---|---|---|---|
| Level rule | 0.31 / 0.44 / 0.49 | 0.31 / 0.44 / 0.49 | not run | not run |
| Comb, no ML | 0.57 / 0.56 / 0.55 | 0.59 / 0.58 / 0.53 | not run | not run |
| LightGBM, level only | 0.85 / 0.81 / 0.73 | 0.98 / 0.93 / 0.80 | not run | not run |
| Small CNN, level only | 0.87 / 0.84 / 0.74 | 0.99 / 0.97 / 0.85 | 0.99 / 0.95 / 0.82 | 0.60 / 0.61 / 0.61 |
| Small CNN, full suppression | 0.86 / 0.84 / 0.75 | 0.98 / 0.95 / 0.82 | 0.99 / 0.96 / 0.80 | 0.60 / 0.61 / 0.61 |

(The two controls were added after the first pass and only the CNNs were
rescored with them.)

**Assessment.**

- **The chain cue is now measured directly.** Adding DDL audio recorded
  beyond 150 m raises the score exactly as much as adding DDL audio recorded
  within 50 m (0.96 against 0.95 at 0 dB). The iteration 1 models respond to
  DDL, with or without an audible drone in it.
- **They are not deaf to drones.** A DroneNoise drone, from a chain and of
  models they never saw, raises the score in most pairs (0.84 at 0 dB). So
  some real drone evidence was learned next to the shortcut.
- **The second control is not at 0.5.** Adding any ordinary sound gives 0.61,
  and it does not depend on how loud the added sound is. The likely reason
  is that a second sound fills the silent gaps of a short event clip, and
  the models learned "no gaps" as a sign of drone, because DDL is continuous
  and FSD50K clips are not. Part of the 0.84 above is therefore not the
  drone. The honest reading is the margin over this control.
- The level rule falls below 0.5, as it must: adding a drone makes a window
  louder, and it calls quiet windows drones.

**Next.** Train on audible drones only, each one buried in a background, and
see whether the far-DDL control falls while the DroneNoise response holds.

## 015 - Training on audible drones only

- Date: 2026-10-10
- Iteration: 2
- Type: Data
- Commit: uncommitted

**What changed.** The choice of training drones, and the mixing. The model,
frontend, negatives, budget and seed are those of iteration 1.

- *Drone pools* (`ddl_pool`, `dronenoise_pool` in `src/dad/realrun.py`).
  "DDL near": windows of the DDL training sessions whose range label is
  reliable and at most 50 m at both ends of the window, 14.9 minutes.
  "DroneNoise": windows within 10 dB of the loudest window of the same file,
  27.6 minutes over all nine microphones, which is roughly 3 minutes of
  distinct audio.
- *Variant `buried`.* Every drone window is summed with a background at an
  in-band signal-to-noise ratio of -5 to +15 dB. Negatives and equalisation
  as in the `full` variant of entry 012.

**Why.** See the box at the top of this iteration. In short: a window in
which the drone is faint is a labelled sample of the recording site.

**How.** Four runs, small CNN and LightGBM, 200,000 and 40,000 training
windows, seed 0:

```powershell
powershell -File artifacts\run_it2_diagnostics.ps1
```

1. Train on DDL near. DroneNoise is then an unseen chain with unseen drones.
2. Train on DroneNoise. DDL is then an unseen chain with unseen drones.
3. Train on DroneNoise without the Matrice 300; probe with the Matrice 300.
4. Train on DDL near plus DroneNoise without the Matrice 300.

**Result.** Small CNN. "Paired" is the AUC of entry 014 at 0 dB; the figure
in brackets is the mean score of the background alone and with the sound
added.

| Trained on | Validation AUC | Transfer AUC | Paired: unseen-chain drone | Paired: DDL beyond 150 m | Paired: other background |
|---|---|---|---|---|---|
| Iteration 1: all DDL, `full` | 0.991 | 0.850 to DroneNoise | 0.84 | 0.96 | 0.61 |
| 1. DDL near | 0.977 | 0.939 to DroneNoise | 0.91 (0.02 to 0.21) | 0.95 (0.02 to 0.30) | 0.65 (0.02 to 0.05) |
| 2. DroneNoise | 0.911 on DDL | (DroneNoise is training data) | 0.95 on DDL near (0.06 to 0.60) | 0.77 (0.06 to 0.09) | 0.63 (0.06 to 0.09) |
| 3. DroneNoise, no Matrice | 0.930 on DDL | 0.924 to Matrice 300 | 0.87 on Matrice (0.05 to 0.34) | 0.81 (0.05 to 0.11) | 0.63 (0.05 to 0.08) |
| 4. DDL near + DroneNoise, no Matrice | 0.964 | 0.912 to Matrice 300 | 0.88 on Matrice (0.05 to 0.26) | 0.90 (0.05 to 0.26) | 0.64 (0.05 to 0.09) |

Mean score on DDL validation windows by range to the drone:

| Trained on | 0 to 50 m | 50 to 100 m | 100 to 150 m | over 150 m |
|---|---|---|---|---|
| Iteration 1: all DDL, `full` | 0.76 | 0.95 | 0.97 | 0.97 |
| 1. DDL near | 0.91 | 0.70 | 0.14 | 0.47 |
| 2. DroneNoise (never saw DDL) | 0.77 | 0.59 | 0.37 | 0.06 |
| 3. DroneNoise, no Matrice | 0.84 | 0.69 | 0.34 | 0.10 |
| 4. DDL near + DroneNoise, no Matrice | 0.95 | 0.84 | 0.40 | 0.29 |

(256, 514, 26 and 196 windows per column; the third column is thin.)

Run 2 in more detail, because it never saw the DDL chain: against FSD50K
validation, AUC on DDL windows is 0.97 at 0 to 50 m, 0.95 at 50 to 100 m,
0.92 at 100 to 150 m and 0.77 beyond 150 m. Recall at its threshold is 0.98,
0.94, 0.88 and 0.27.

Run 1 on DroneNoise by drone model, recall at the frozen threshold: Matrice
300 0.76 (iteration 1: 0.01), Mini 3 Pro 0.87, Yuneec H520E 0.93, the
unnamed 795 g model 0.76.

False alarms on FSD50K validation at each run's threshold: 12.3%, 17.7%,
14.4% and 8.2%, against 6.4% for the iteration 1 model.

LightGBM, same four runs: transfer AUC 0.883, not independent, not run, and
0.966. In run 4 it is ahead of the CNN (0.966 against 0.912 on the Matrice).

**Assessment.**

- **The hypothesis holds in its main part.** Trained on audible drones, the
  model's score now falls with distance instead of rising, and it finds
  drones on a chain it never trained on in both directions: DDL to
  DroneNoise 0.94, DroneNoise to DDL 0.91. The Matrice 300, which iteration
  1 missed entirely, is found three times in four.
- **Run 2 is the cleanest evidence in the project so far.** A model that
  has heard only four drone types at one Scottish airfield detects two other
  drone types at two English sites, well at short range and progressively
  worse with distance, and barely at all beyond 150 m. That is what hearing
  a drone looks like. It also says what DDL beyond 150 m mostly is:
  background.
- **A chain cue remains whenever the chain is in the training drones.**
  Adding DDL audio from beyond 150 m lifts the run 1 model's score from 0.02
  to 0.30. The same audio lifts the run 2 model, which never saw DDL, only
  from 0.06 to 0.09. So most of that lift is the DDL chain. It is much
  smaller than in iteration 1, where far and near were indistinguishable,
  but it is not gone.
- **The cost is visible and honest.** Validation AUC dropped from 0.99 to
  0.91 to 0.98, and false alarms at 95% recall rose to between 8% and 18%.
  The iteration 1 figures were inflated by the shortcut. The primary target
  (5%) is not met.
- **The small CNN does not clearly beat LightGBM.** It is ahead in run 1 and
  behind in run 4. `REQUIREMENTS.md` requires beating the baselines.
- The "other background" control stays near 0.63 everywhere, with a very
  small change in score (0.05 to 0.09). AUC counts any consistent rise,
  however small; the change in mean score is the more useful figure here.
- All of this is one seed. Two more seeds of runs 1 and 2 are being trained.

**Next.** A council on how to conclude iteration 2 and what iteration 3
should be: which chains to train on, what to do about the remaining chain
cue, and whether to move on to the larger models.

## 016 - Council: how to set up iteration 3

- Date: 2026-10-10
- Iteration: 2
- Type: Council
- Commit: uncommitted

**Question.** How should iteration 3 be set up?

The proposal put to the council: (1) train on both chains, DDL within 50 m
and DroneNoise, each window buried; (2) for every candidate model, run the
two leave-one-chain-out trainings with three seeds as the evidence of
generalisation, then train the final model on both chains; (3) cancel the
remaining DDL chain cue by using DDL windows from beyond 150 m as negatives;
(4) then try the larger models and a larger training budget to bring false
alarms toward 5%; (5) score the sealed test once at the end; (6) move
DroneNoise from held-out to training in the requirements. Alternatives: B,
train on DroneNoise only and keep all of DDL for evaluation; C, the proposal
without step 3; D, no new model families yet, work on the data first.

**Devil's Advocate** (saw the project and rescored the saved models).

| Objection | Rank | Outcome |
|---|---|---|
| Step 3 has no data. The 27 training sessions hold 16 s of reliable audio at 150 m or more, all in one session. Taking it from validation would turn the far control into a training target. And the far windows are not drone-free: the model that never saw DDL flags 62% of the far windows in one session. | Decisive | **Accepted. Step 3 is dropped.** |
| The false-alarm figure is one session's tail when DDL is in training. In run 1 of entry 015, 36 of the 39 windows that set the threshold come from `290321-T005`, the only day-1 validation session. Without it false alarms would read 2.4% instead of 12.3%, and in run 4 4.5% instead of 8.2%. On unseen drones at their own 95% point the figure is 22 to 25%. | Decisive | **Accepted.** The in-chain validation false-alarm rate is no longer a headline figure. The headline is false alarms at the 95%-recall point of a chain the model never trained on. |
| Training on both chains brought no gain on unseen drones (run 3 to run 4: Matrice AUC 0.924 to 0.912), and a larger budget made it worse (unseen-drone paired AUC fell from 0.906 to 0.878 between 50,000 and 200,000 windows). | Serious | **Accepted as a caution.** No larger budget. Whether both chains help is measured, not assumed. |
| The final both-chain model can only be tested on 30 Halmstad clips, which cannot show 5% at 95% nor rank a network against LightGBM. | Serious | **Stands.** It is a limit of the data, stated wherever the sealed test is reported. |
| Three seeds do not add flight events; the Matrice probe is 7 events. | Serious | **Stands.** The Matrice probe is not used to rank models. |
| The full plan costs 10 to 50 hours, most of it the CRNN on the CPU. | Serious | **Accepted.** The model screen is bounded (below) and the CRNN runs only if the ResNet wins. |
| A requirement that looks unreachable calls for a check-in with Mads, not another iteration. | Weak | **Partly accepted.** A stop rule is set below. |

**Scholar** (saw no project context; reported 31 sources located in this
session).

- Leave-one-domain-out is the accepted protocol, and training the final
  model on all domains afterwards is standard (Gulrajani and Lopez-Paz,
  2020). With two domains it supports "transfers between these two setups in
  each direction", not a prediction for a third. Three seeds is the floor.
  Anything tuned on the held-out domain makes the figure optimistic.
- Relabelling weak positives as negatives runs against practice, which is to
  mask, exclude or soft-label them. The noise would be concentrated at low
  signal-to-noise ratio and would teach "faint means absent". Closer to
  practice: leave them out of training and use them as a negative control in
  evaluation.
- Mixing a real target into a real background at a set ratio is
  long-established in sound event detection (Scaper; DCASE 2017 rare events),
  and the paired design is sound, as a diagnostic. It needs a sham-addition
  control, and it does not replace real recordings.
- With minutes of positives, capacity is rarely the constraint. The usual
  order is more diverse sources, then pretrained embeddings and
  augmentation, then capacity. A transformer trained from scratch is against
  the grain. The accepted way to decide is a pre-specified comparison under
  the same protocol and seeds, judged on the worse direction.
- For a false-alarm target across setups: fix the threshold by a rule
  declared in advance on development data, report recall and false alarms
  per held-out setup, worst case rather than pooled.

Departures from the Scholar's practice: none in what is decided below. One
recommendation is not taken up yet: a frozen pretrained audio embedding with
a linear classifier as a baseline. It needs a model download and a new
dependency, and is noted as a candidate for a later iteration.

**Problem Solver** (saw the project and rescored the saved models). Verdict:
try a modified version, the proposal's data without step 3, in the order of
alternative D.

- The far-DDL negatives cannot be built (same finding as the Devil's
  Advocate, reached independently).
- Read a different figure from the leave-one-chain-out runs: false alarms at
  the unseen chain's own 95% recall. Today: DDL-near to DroneNoise 22% and
  26% (two seeds); DroneNoise to DDL within 50 m 15%.
- The false alarms are not concentrated in a few sound classes. The
  threshold is low because of the weakest 5% of positives, so coverage of
  the positives is the lever.
- Averaging the score over three windows helps in-chain and does nothing
  across chains, so the cross-chain misses are systematic.
- Proposed levers: widen the DDL range limit to 75 or 100 m (33 or 51
  minutes instead of 15, and it brings in the first recording day); weight
  steady-noise negatives; lower the mixing ratio; pitch perturbation.

**Agreement and dissent.** All three reject relabelling far DDL as negative.
All three put data before capacity. They differ on the final model: the
Scholar calls training it on both chains standard, the Devil's Advocate
points out it can then barely be tested. Both are right, and it is not
resolved by more analysis: it is a limit of having two chains.

**Decision.**

1. **Step 3 is dropped.** Far DDL stays a control in evaluation only.
2. **Headline figure from now on:** in each leave-one-chain-out arm, false
   alarms on FSD50K validation at the unseen chain's own 95% recall, with
   recall at 5% false alarms next to it. Both arms are reported and a model
   is judged on the worse one. These are development figures: recipe choices
   are made on them, so they are optimistic as estimates.
3. **Data before models.** One-seed screens of four changes on the small CNN
   and LightGBM, both arms: DDL range limit 75 m and 100 m; steady-noise
   negatives weighted three times; mixing down to -10 dB. A change is kept
   only if it improves the headline figure by more than the seed spread
   without hurting the other arm.
4. **Not done:** pitch or speed perturbation. It alters a recording, which
   goes beyond the mixing Mads allowed in entry 011. It is listed for him.
5. **Then** the kept recipe with three seeds on both arms, and a bounded
   model screen: ResNet and transformer, one seed, both arms; three seeds
   only if they fall outside the small CNN's seed range on both arms; CRNN
   only if the ResNet wins.
6. **Final model** on both chains, three seeds, threshold rule written down
   beforehand, sealed test scored once.
7. **Stop rule.** If the kept recipe leaves both arms above about 10% false
   alarms, stop and tell Mads that 5% looks out of reach with two recording
   chains, before spending on capacity.

**Next.** Rescore the existing models with the headline figure, including
the DroneNoise-trained models on all 36 DDL sessions, and measure the seed
spread.

## 017 - Baseline for the headline figure, three seeds

- Date: 2026-10-10
- Iteration: 2
- Type: Evaluation
- Commit: uncommitted

**What changed.** The evaluation now reports the headline figure decided in
entry 016 (`unseen_chain` in `src/dad/realrun.py`), and a model that was not
trained on DDL is scored on all 36 DDL sessions instead of the 9 validation
ones. The two leave-one-chain-out runs of entry 015 were repeated with seeds
1 and 2, and all six models rescored. No change to training.

**Why.** To have a baseline and a seed spread before judging any change.

**How.**

```powershell
powershell -File artifacts\run_it2_seeds.ps1
powershell -File artifacts\run_it2_rescore.ps1
```

Small CNN, `buried` variant, 200,000 windows. Negatives in every figure are
the 3,679 FSD50K validation windows.

**Result.** False alarms at the unseen chain's own 95% recall, and recall at
5% false alarms:

| Arm | Unseen drones | Seed 0 | Seed 1 | Seed 2 | Mean |
|---|---|---|---|---|---|
| X: trained on DDL within 50 m | DroneNoise, 335 windows, 20 events | 22.1% / 0.61 | 25.6% / 0.56 | 21.7% / 0.60 | **23.1% / 0.59** |
| Y: trained on DroneNoise | DDL 0 to 100 m, all sessions | 24.7% / 0.63 | 21.3% / 0.68 | 24.5% / 0.60 | **23.5% / 0.64** |

Arm Y by range (mean of three seeds; false alarms / recall at 5%):

| Range | False alarms at 95% recall | Recall at 5% false alarms |
|---|---|---|
| 0 to 50 m | 12.0% | 0.85 |
| 50 to 100 m | 25.8% | 0.55 |
| 100 to 150 m | 15.0% | 0.76 |
| beyond 150 m | 37.7% | 0.02 |

Other figures were stable across seeds: arm X transfer AUC 0.939, 0.929,
0.939; arm Y AUC on DDL validation 0.911, 0.916, 0.908; the paired figures
moved by 0.01 at most.

**Assessment.**

- **The honest distance to the target is about 23% false alarms at 95%
  recall, in both directions.** The target is 5%. Put the other way: at 5%
  false alarms, about 60% of the drone windows from an unseen chain are
  found.
- The seed spread of the headline figure is about 4 points (range) in both
  arms. A change has to move it by more than that to count.
- Arm Y at short range is much closer: 12% false alarms, or 85% of windows
  found at 5%. The requirement does not say at what distance it should hold.
  That is a question for Mads.
- The 100 to 150 m bin does better than 50 to 100 m. The likely reason is
  which sessions fall in which bin (the 50 to 100 m bin holds most of the
  windy first day) and not distance as such. This was not checked per
  session.
- Beyond 150 m the drone is effectively not found (2% at 5% false alarms).

**Next.** The screens of entry 016, step 3.
