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
| Iteration 2 | 014 to 020 | Train only on drones that are loud in their own recording, always buried in a background; judge with a same-background probe and on a recording chain the model never trained on | Iteration 1 showed the faint, distant drones teach the background, and that validation AUC cannot tell | **Hypothesis confirmed, target not met.** The model now detects drones across recording chains and its score falls with distance. False alarms at 95% recall on an unseen chain are about 23% against a target of 5%. Data recipe changes and larger models did not close the gap. Stopped on Mads's instruction: more data is needed. |
| Phase 1 | 021 to 024 | New priorities (recall first, then time to detection, at about 150 m) and four new datasets, two with no-drone audio from the drone's own rig | Mads's new priorities; the new data answers iteration 2's request | Data ready, with range at every moment. At 100-200 m the drone is at the edge of audibility for an independent model, and below it for the DJI Mini 2 at Nightjar. |
| Iteration 3 | 025 to 029 | Leave one rig out (Nightjar, d3set); weight positives toward 150 m with real far drones or with near drones summed into the rig's background at a lower level (simulated distance) | Same-rig negatives remove the reason iteration 2 excluded far drones; Mads's 150 m priority | **Hypothesis rejected.** Recall at 100-200 m at 10% false alarms on an unseen rig: about 0.04 (Mini 2) and 0.21 (X500), best recipe, three seeds. Real far drones did not help; simulated distance helped a little and never hurt. The limit is how much drone the audio holds at 150 m. |
| Iteration 4 | 030 to 031 | Average the scores of several consecutive windows, and a 2 s window | Per-window evidence at 150 m is weak; Mads ranks recall above time | **Hypothesis rejected by its rule, small consistent gain.** Averaging 8 windows (4.5 s) lifts X500 recall at 100-200 m from 0.21 to 0.28 (three seeds, +7 against +10 required); a 2 s window adds nothing; the Mini 2 stays at chance. Stopped for Mads: the next lever (a microphone array) and the drone class that matters are his decisions. |
| Iteration 5 | 032 to 037 | Pool every rig but the held-out one (Nightjar, d3set, three FPV campaigns); speed perturbation; two-microphone rescore | Mads's scenario (small drones at 50 m/s, alarm before 150 m, 2 omni mics) and his question about one big pool | **Pooling alone rejected; pooling with speed perturbation is the new best recipe.** Three seeds: X500 at 100-200 m 0.33 (was 0.21), FPV on an unseen campaign 0.66, Mini 2 still at chance. Two microphones: no consistent change. |
| Iteration 6 | 043 | Iteration 5's best recipe trained once on every rig, threshold frozen at 5% false alarms on validation; CNN, LightGBM and comb saved for the real test | Mads asked for a final model for the real test | **Checkpoints, not evidence.** Validation (in-distribution): CNN recall 0.83 at 5% false alarms. Unseen Lund rig: CNN 0.50, baselines near 0. CNN alarms on 55% of mower-only audio. R2 and R3 not met. |
| Iteration 7 | 055 to 058 | Edge Impulse model (MFE + small CNN) on upload-licensed data, scored locally next to model 6 | Mads asked to try Edge Impulse; deployment hardware still open | **Negative result.** On the shared windows the EI model detects nothing off its training rig (Halmstad recall 0 at the frozen threshold, test AUC 0.36); model 6 0.55 at 4.8% false positives on Halmstad. Small on Cortex-M4F (12 ms, 20 KB RAM), but its MFE is too slow at FFT 1024. Model 6 stays the candidate. |
| Iteration 8 | 059 to 061 | Edge Impulse on the pool, d3set held out, clips scaled to -30 dBFS | Mads asked for leave-one-dataset-out in Edge Impulse; iteration 7 failed on a 50 dB level gap between rigs | **Hypothesis rejected.** In-distribution AUC rose to 0.92, but on unseen d3set AUC 0.12 (background ranked above drones), worse than the untrained comb (0.53). Microcontroller out of scope (entry 060); Edge Impulse stopped. |

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

## 018 - Screens of four data changes: none kept

- Date: 2026-10-10
- Iteration: 2
- Type: Tuning
- Commit: uncommitted

**What changed.** Four changes to the training data were each tried alone,
seed 0, small CNN and LightGBM, 200,000 and 40,000 windows:

1. DDL range limit 75 m (33.2 min of drone audio instead of 14.9).
2. DDL range limit 100 m (50.7 min).
3. Steady-noise backgrounds (engines, vehicles, aircraft, wind, rain, water,
   fans, tools; `STEADY_NOISE` in `src/dad/realrun.py`) sampled three times
   as often.
4. Mixing ratio down to -10 dB instead of -5 dB.

Changes 1 and 2 only exist in arm X, since arm Y does not train on DDL.

**Why.** Entry 016, step 3. The Problem Solver's reading was that the
weakest positives set the threshold, so wider coverage of positives and
harder negatives were the cheapest levers.

**How.** `powershell -File artifacts\run_it2_screens.ps1`. Rule fixed in
advance: keep a change only if it improves the headline figure by more than
the seed spread of about 4 points without hurting the other arm.

**Result.** Small CNN, false alarms at the unseen chain's own 95% recall
(lower is better):

| Recipe | Arm X: DDL to DroneNoise | Arm Y: DroneNoise to DDL 0 to 100 m |
|---|---|---|
| Baseline, three seeds | 22.1%, 25.6%, 21.7% | 24.7%, 21.3%, 24.5% |
| DDL limit 75 m | 28.0% | not applicable |
| DDL limit 100 m | 30.5% | not applicable |
| Steady noise three times | 23.1% | 25.4% |
| Mixing down to -10 dB | 25.1% | 29.5% |

Widening the DDL limit also brought the chain cue back. Mean score on DDL
validation by range (0 to 50, 50 to 100, 100 to 150, over 150 m): limit 50 m
0.91 / 0.70 / 0.14 / 0.47; limit 75 m 0.98 / 0.95 / 0.66 / 0.85; limit 100 m
0.94 / 0.96 / 0.92 / 0.83. Transfer AUC to DroneNoise: 0.939, 0.922, 0.908,
and 0.850 with no limit (iteration 1).

LightGBM on the baseline recipe: 32.0% in arm X and 21.2% in arm Y. At 5%
false alarms it finds 30% and 56% of the unseen-chain windows, against 59%
and 64% for the small CNN.

**Assessment.**

- **None of the four changes is kept.** Two made the headline figure worse
  by more than the seed spread (DDL limit 100 m; mixing to -10 dB in arm Y),
  the others did not move it.
- **The range limit shows a dose-response.** Each step that lets fainter DDL
  windows into training (50 m, 75 m, 100 m, no limit) lowers transfer to the
  unseen chain and flattens the score against distance. That is the
  hypothesis of this iteration, confirmed a second way: faint positives
  teach the background.
- **The small CNN beats LightGBM on the worse arm** (about 23.5% against
  32.0%), which is the comparison rule of entry 016. In arm Y they are level.
- **The stop rule of entry 016 is met.** The recipe leaves both arms near
  23%, far above the 10% line. With these two recording chains, changing how
  the data is sampled and mixed does not bring 5% within reach.

**Next.** By the stop rule this goes to Mads before any spending on
capacity. Mads is away and the machine is otherwise idle, so the bounded
model screen that entry 016 had already specified (ResNet and transformer,
one seed, both arms, about 45 minutes) is run now, as information for that
conversation and nothing more: no CRNN, no larger budgets, no follow-up
unless a model falls outside the small CNN's seed range on both arms. One
frontend setting that `REQUIREMENTS.md` marks as free, a 2 s window, is
screened in the same batch. The sealed test stays sealed.

## 019 - Larger models and a longer window

- Date: 2026-10-10
- Iteration: 2
- Type: Model
- Commit: uncommitted

**What changed.** Two larger architectures and one frontend setting were
each tried on the unchanged iteration 2 recipe, seed 0, 200,000 windows,
both leave-one-chain-out arms:

- ResNet, 695,010 parameters (the small CNN has 22,530).
- Spectrogram transformer, 826,626 parameters.
- The small CNN and LightGBM with a 2 s window instead of 1 s.

**Why.** To test the remaining explanations for the gap to the target before
concluding that it is the data: that the small CNN lacks capacity, or that
one second is too short to integrate over. Both were specified in advance
(entries 016 and 018). The council had advised that capacity is rarely the
constraint with minutes of positive data.

**How.** `powershell -File artifacts\run_it2_models.ps1`. Training took
about 10 minutes per ResNet run, 6 per transformer run and 10 per 2 s run on
the GPU. Rule fixed in advance: a model is followed up with three seeds only
if it falls outside the small CNN's seed range on both arms, and models are
compared on their worse arm.

**Result.** False alarms on FSD50K validation at the unseen chain's own 95%
recall (lower is better):

| Model | Arm X: DDL to DroneNoise | Arm Y: DroneNoise to DDL 0 to 100 m | Worse arm |
|---|---|---|---|
| Small CNN, 1 s, three seeds | 21.7 to 25.6% (mean 23.1%) | 21.3 to 24.7% (mean 23.5%) | 23.5% |
| LightGBM, 1 s | 32.0% | 21.2% | 32.0% |
| ResNet, 1 s | 36.1% | 20.5% | 36.1% |
| Transformer, 1 s | 27.1% | 15.9% | 27.1% |
| Small CNN, 2 s | 19.6% | 19.2% | 19.6% |
| LightGBM, 2 s | 30.7% | 16.7% | 30.7% |

Recall at 5% false alarms, same order (arm X / arm Y): 0.59 / 0.64 (mean of
three seeds), 0.30 / 0.56, 0.44 / 0.61, 0.33 / 0.61, 0.64 / 0.69,
0.32 / 0.62.

The 2 s runs are scored on 2 s windows, so their evaluation sets are not the
1 s ones: 180 DroneNoise windows instead of 335, 1,847 DDL windows instead
of 3,702, and 3,118 FSD50K validation windows from the 1,275 clips that are
at least 2 s long.

**Assessment.**

- **Capacity is not what is missing.** A network 30 times larger is clearly
  worse in arm X and no better than the seed range in arm Y. The transformer
  is better in arm Y and worse in arm X. On the worse arm, which is the
  agreed comparison, the small CNN is the best model tried. Neither larger
  model meets the rule for a follow-up, and none was run.
- **A 2 s window is the only change in this iteration that helped in both
  directions,** by about 4 points. That is the size of the seed spread, it
  is one seed, and the evaluation windows differ, so it is a lead and not a
  result. It is what integration time predicts, and it costs a second of
  latency, which this project treats as the scarce quantity. Still 19%.
- Every model reaches 0.92 to 0.99 AUC on the confounded validation
  contrast. On that number alone the ResNet would have looked as good as the
  small CNN.

**Next.** Conclude the iteration.

## 020 - Iteration 2 verdict, and why the work stops here

- Date: 2026-10-10
- Iteration: 2
- Type: Evaluation
- Commit: uncommitted

**What changed.** Nothing. This entry closes iteration 2. On reading the
intermediate results Mads asked whether the conclusion is that more data is
needed, and to stop if that is the council's consensus. It is, and the
experiments agree, so no further training is started.

**Verdict on the hypothesis: confirmed.** Training only on drones that are
loud in their own recording, each buried in a real background, produces a
model that detects drones and not a recording site:

- It finds drones recorded by a chain, and of models, it never trained on,
  in both directions (transfer AUC 0.93 to 0.95).
- Its score falls with distance to the drone, where iteration 1's rose.
- Letting fainter DDL windows back in removes the effect step by step
  (entry 018).

**Verdict against the requirements: not met.**

| Requirement (all PROPOSED) | Target | Best honest figure | Met |
|---|---|---|---|
| False alarms at 95% recall, held-out sources | at most 5% | about 23% on an unseen chain (small CNN, 1 s, three seeds) | No |
| ROC AUC, held-out sources | at least 0.95 | 0.93 to 0.95 on an unseen chain | Borderline |
| Balanced accuracy | at least 90% | not computed: it needs one threshold that carries across chains, which has not been established | Unknown |
| Beat the comb and LightGBM baselines | yes | small CNN ahead on the worse arm (23.5% against 32.0%), level with LightGBM on the other | Yes, narrowly |

The sealed test sources (Halmstad, FSD50K-eval) have been scored **zero
times**. They were kept for a finished candidate, and there is none.

**Why more data, and not more modelling.**

- The council said so three times over, independently. The Scholar: with
  minutes of positive audio the order is more diverse sources, then
  pretrained features, then capacity. The Devil's Advocate: the networks are
  limited by "3 minutes of DroneNoise and one afternoon of DDL". The Problem
  Solver: coverage of the positives is the lever, and low confidence that 5%
  is reachable with two chains.
- The experiments then tested the alternatives. Four changes to sampling and
  mixing did not help (entry 018). Two models about 30 times larger did not
  help (entry 019). A longer window helped by about 4 points.
- What the training drones amount to: about 15 minutes of two drone models
  within 50 m on one afternoon at one site, and about 3 minutes of distinct
  audio of four other models at one airfield. The other 60 minutes of DDL
  make the model worse when used.

**What data would change the picture,** in order of expected value:

1. **Non-drone recordings from a setup that also recorded drones.** This is
   the only thing that removes the confound at its source and that allows an
   honest test with both classes from one chain. The DDL host describes
   no-drone sessions that are not in the archive; asking the authors is the
   cheapest route. Any own recording of a drone and of the same spot without
   it would do the same.
2. **Drone recordings from more recording setups, close enough to be clearly
   audible.** Number of distinct setups and drone types matters more than
   minutes. Candidates already surveyed in entry 001, each with an obstacle
   that is Mads's to weigh: RWDA (needs an IEEE DataPort subscription),
   MMAUD (non-commercial licence), Ljungbyhed (no licence stated for the
   audio).
3. **Not more distant or faint drone audio.** It teaches the background.

**Decisions that are Mads's,** none of which were taken here:

1. Which of the data routes above to pursue.
2. Whether to try pretrained audio features (for example YAMNet or PANNs
   embeddings with a small classifier on top). The Scholar recommends it as
   the standard small-data baseline. It needs a model download and a new
   dependency.
3. At what distance the false-alarm target must hold. Within 50 m the unseen
   -chain figure is about 12%; at 50 to 100 m about 26%; beyond 150 m the
   drone is not found.
4. Whether a 2 s window is acceptable. It is the one change that helped, at
   the cost of latency.
5. Whether pitch or speed perturbation of drone recordings counts as real
   audio under the rule set in entry 011. It was not used.
6. The rows in `REQUIREMENTS.md` on training and held-out sources no longer
   describe what was done: DroneNoise was used both as a training source and
   as an unseen chain, in different runs. They are still marked PROPOSED and
   were not edited.

**State of the repository.** Processed data in `data/processed/`. Code for
training and evaluation on real data in `src/dad/realrun.py` and
`scripts/run_real.py`; 44 tests. Every run of this iteration has its result
file in `artifacts/runs/` and its checkpoint in `artifacts/models/`. That
folder is git-ignored, so results and checkpoints stay on this machine. The
batch scripts that produced them were first written under `artifacts/`,
which is how entries 015 to 019 refer to them; copies that travel with the
repository are in `scripts/experiments/`, under the same file names.

**Next.** Wait for Mads.

---

# Phase 1 - New priorities and new data

> Between iteration 2 and iteration 3. Mads changed what the work optimises
> for, and twelve more datasets were found, four of which were downloaded.
> This phase records the new priorities, what the new data really contains,
> and how it was brought into the processed store. Entries 021 onward, up to
> the council that opens iteration 3.

## 021 - Requirements: recall first, at about 150 m

- Date: 2026-10-14
- Type: Requirements
- Commit: uncommitted

**What changed.** Mads set new priorities, in his words: "Falsealarms are not
a pirority, high recall rate matters more, time is also important, and the
most relevant distance is around 150 meters distance from drones, meaning we
should somehow weigh the data. this has to be done in a smart way though,
consult the council." `REQUIREMENTS.md` has a new "Priorities" section:

| Requirement | Value | Status |
|---|---|---|
| Order of priority | 1. recall, 2. time to detection, 3. false alarms | Decided by Mads |
| Range that matters most | about 150 m; data and metrics weighted toward it | Decided by Mads |
| Meaning of "time" | time to detection, not processing speed | PROPOSED, to confirm |

The old primary metric and target (at most 5% false alarms at 95% recall,
entry 003) are marked SUPERSEDED and are still reported.

**Why.** Mads's decision. Iteration 2 had stopped on the false-alarm target
(entry 020); under the new order that target is no longer the one that
decides.

**How.** Edit to `REQUIREMENTS.md`. Nothing was run.

**Result.** Not applicable.

**Assessment.** Three things are left open by the wording and are put to the
council (entry 025) and to Mads:

- "Time" is read as time to detection. It could also mean processing latency
  or project time. Under the first reading it trades against recall: a longer
  window or several windows of evidence raise recall and delay the alarm.
- "False alarms are not a priority" cannot mean "unbounded": a detector that
  always says "drone" has full recall. Some false-alarm budget has to be
  fixed for recall to mean anything.
- "Weigh the data" can mean weighting the training data, weighting the
  evaluation, or both.

Iteration 2's finding that faint, far drones teach the recording site
(entries 013, 018) was made with no no-drone audio from those sites. Whether
it still holds with same-site negatives is the central question for the new
weighting.

**Next.** Bring in the new datasets and convene the council.

## 022 - New datasets: what arrived and what is in them

- Date: 2026-10-14
- Type: Data
- Commit: uncommitted

**What changed.** Twelve datasets were found after iteration 2 and are
documented in `data/README_new_datasets.md` (written when they were fetched,
2026-10-12). Four were downloaded and are on Google Drive; the other eight,
all on Zenodo (28.24 GB), are held back at Mads's request and have only their
`_source_record.json`. The four downloaded ones were copied unchanged from
Google Drive into `data/` (`robocopy`, file sizes checked) and inspected in
place.

**Why.** Entry 020 asked for no-drone audio from setups that also recorded
drones. Three of the four downloaded sets have that.

**How.** Read-only inspection of the headers, the drone logs, ground-truth
files and dataset cards. Scratch scripts, not kept: per-session DJI log
parsing for Nightjar, per-flight telemetry for d3set, ground truth for Lund.

**Result.**

| Set | Rig, site | Drone | Audio | Same-rig no-drone audio | Licence |
|---|---|---|---|---|---|
| Nightjar | miniDSP UMA-16 (16 ch, 48 kHz, 16-bit); one rural US site; 4-6 Aug 2026 | DJI Mini 2 | 10 sessions, 159 min | Two deliberate drone-free night sessions (20 min) and the stretches before, between and after flights | CC BY 4.0 |
| d3set | 16-mic MEMS ground array (16 kHz, 16-bit); two Canadian sites; Oct 2024 to May 2025 | Holybro X500 | 22 flights, 140 min | Before the motors start (with hand claps for synchronisation), one 14 s clip | GPL-3.0 |
| Lund (Ljungbyhed) | 12-mic array (96 kHz, 24-bit); Swedish airfield; Nov 2024 and Mar 2025 | small quad, big quad, fixed wing | 13 recordings, 143 min | Unknown (below) | none stated: internal use only |
| Glasgow | one microphone (44.1 kHz), probably indoors | 24 drones hovering at 1 m and 5 m | 256 clips of 2 min | One 60 min noise recording | CC BY 4.0 |

Facts that decide how each can be used:

- **Nightjar.** The DJI logs give position at 10 Hz and the motor state; the
  rig position is in the series' own analysis files (41.534765, -75.595254,
  surveyed 2026-07-27; the author notes it moved by about 10 m on 08-05).
  Airborne audio by slant range, 0-50 / 50-100 / 100-150 / 150-200 /
  200-300 / 300+ m: 2,538 / 1,007 / 328 / 77 / 289 / 644 s. The dataset
  author's own analysis: "Neither array demonstrably hears the drone past
  ~200 m". A fixed machinery tone at about 140-170 Hz and 245 Hz is present
  with and without the drone. The deliberate negatives are at night
  (crickets), most flights by day; one flight is at night.
- **d3set.** Telemetry gives position, the `isflying` flag and the
  blade-pass frequency of each motor (`actuator.csv`, in milliseconds from
  the start of the audio, unlike the other files, which use Unix time). The
  motors start within a second of take-off in every flight. Most flying is
  within 50 m; four "distance test" flights reach 150-190 m. One flight
  (`Training/25 oct`) was recorded at 8 kHz.
- **Lund.** Ground truth (x, y, z at 5 Hz) covers the flight only. Its
  `drone_live` flag is True on every row, so it marks no negatives. Before
  and after the ground truth (1,010 s and 2,010 s in all) the drone stands at
  its take-off point 75-375 m from the array; whether its motors run then is
  not recorded. Seconds of ground truth by slant range, same bins as above:
  578 / 1,080 / 409 / 188 / 526 / 1,403. `info.md` names the drone only for
  recordings 01 to 05 (small quad).
- **Glasgow.** Very close range only, so of little use at 150 m. The 60 min
  noise file is the same recording chain's no-drone audio.
- **Wind** (asked by Mads). None of the drone logs records wind speed: not
  the DJI logs (Nightjar), not the PX4 telemetry (d3set), not Lund's ground
  truth. Of the older sets, DroneNoise records only whether a flyover was
  upwind or downwind (`uw`/`dw` in the file name; the spreadsheet has no wind
  column) and DDL only temperature. Among the held-back Zenodo sets,
  `DJI_Mini2_Wojcik` describes the weather in its `metadata.csv`. Hourly
  wind could be taken from a weather archive by date and place; that was not
  done.

**Assessment.** With Nightjar and d3set, positives and negatives from the
same rig exist for the first time, and with range ground truth. Seconds of
drone audio at 100-200 m across Nightjar, d3set and Lund: about 410, 650 and
600. That is enough to measure recall near 150 m, and thin enough that it
has to be counted per flight, not per window.

**Next.** Convert to the processed store, with range at every moment.

## 023 - New datasets into the processed store

- Date: 2026-10-14
- Type: Preprocessing
- Commit: uncommitted (on top of 7d4f08a)

**What changed.** Nightjar, d3set, Lund and Glasgow were converted to 16 kHz
mono and added to `data/processed/`, with the drone's state and range at
every moment. The raw copies in `data/` are untouched.

**Why.** Training and evaluation by range need, for every window, whether a
drone is flying and how far away it is.

**How.**

```powershell
7z x data\Glasgow_Drone_Authentication\ACSAC_2022_Drone_Authentication.7z -odata\extracted\Glasgow
uv run python -I scripts/prepare_new_data.py
```

Decisions, all in `scripts/prepare_new_data.py`:

- **Format only,** as in entry 007: one channel, polyphase resampling to
  16 kHz, no gain or filtering. Channel 0 of each array (Nightjar UMA-16,
  d3set matrix, Lund array). The second Nightjar array (SB-POLARIS, 3 live
  capsules) and d3set's on-board microphone are not used.
- **One stored recording, several manifest rows.** Each source file is
  stored whole; the manifest gets one row per stretch of one state:
  `drone` (label 1: motors on by the drone's own log), `background`
  (label 0: before the motors start or after they stop), `uncertain`
  (label -1: 10 s either side of every change of state, and stretches the
  logs do not settle). Windows never cross from one row into the next.
- **Range.** `range_track.csv` gives the slant range at 10 Hz for every drone
  row. Nightjar: DJI log position (`OSD.height` is in feet) against the rig
  position from the series' own analysis; timing anchored on the first valid
  UTC stamp and counted forward with `OSD.flyTime`. d3set: GPS against the
  array's reference position (`additional_data.json`; the authors warn GPS
  altitude is unreliable). Lund: ground truth against the centre of the 12
  microphones. Glasgow: the distance in the file name (1 m or 5 m).
- **Lund before and after its ground truth** is `uncertain`: nothing says
  whether the motors run.
- **d3set `Training/25 oct`** is left out: it was recorded at 8 kHz, and
  upsampling would leave the 4-8 kHz band empty for that flight only.
- Nightjar UMA-16 and d3set are 16-bit and stored as int16; Lund (32-bit
  integer, see entry 024) as float32; Glasgow as int16.

**Result.**

| Source | Audio | Drone | Background | Uncertain | Drone by slant range, 0-50 / 50-100 / 100-150 / 150-200 / 200-300 / 300+ m (s) |
|---|---|---|---|---|---|
| Nightjar | 159.1 min (39 files, 10 sessions) | 79.0 min | 75.3 min | 4.8 min | 2,389 / 1,008 / 331 / 77 / 289 / 644 |
| d3set | 137.7 min (20 flights, 2 clips) | 110.6 min | 14.1 min | 13.0 min | 4,870 / 1,098 / 481 / 173 / 0 / 0 |
| Lund | 90.5 min (13 recordings) | 69.9 min | none | 20.6 min | 584 / 1,061 / 413 / 190 / 532 / 1,412 |
| Glasgow | 582.3 min (257 files) | 521.8 min | 60.5 min | none | all at 1 m or 5 m |

**Assessment.** For the first time, two rigs (Nightjar, d3set) supply both
classes with range ground truth. The band that now matters is thin: Nightjar
has 6.8 min and d3set 10.9 min at 100-200 m, each from 7 sessions. Lund adds
10.1 min at 100-200 m but no negatives.

**Next.** Check the converted data before using it.

## 024 - Checks on the new data, and what an independent model hears by range

- Date: 2026-10-14
- Type: Data
- Commit: uncommitted (on top of 7d4f08a)

**What changed.** Nothing in the data. Integrity, clock and level checks,
and the iteration 2 models scored on the new rigs as listeners that have
never heard them.

**Why.** Before weighting toward 150 m it has to be known whether a drone at
150 m is audible on these rigs at all, and whether the labels' clocks agree
with the audio.

**How.**

```powershell
uv run python -I scripts/verify_new_data.py
uv run python -I scripts/qa_new_data.py
.venv-dml\Scripts\python.exe scripts\probe_chains.py --name probe_it2_models --models it2_dronenoise_dscnn_buried it2_dronenoise_s1_dscnn_buried it2_dronenoise_s2_dscnn_buried it2_ddl_near_dscnn_buried --chains nightjar d3set glasgow --device dml
```

Figures in `artifacts/qa/`, probe results in
`artifacts/runs/probe_it2_models.json`.

**Result.**

*Integrity.* The copy from Google Drive hit transient read errors (Windows
error 121) on 8 Nightjar files, which `robocopy` retried. All 381 local files
were re-hashed against the checksums in the download log: 0 problems.

*Clocks.* Spectrograms around the logged motor start: in Nightjar
`flight-20260804` the harmonic stack appears at the logged start, and in
d3set's first distance flight within 1 s of the first non-zero blade-pass
frequency. In `flight-20260804` the rotor lines end about 10 s before the
logged stop (landing), inside the 10 s guard. In the 08-06 sessions the drone
takes off about 43 m from the rig and its onset is faint. The automatic
level-step detector was not reliable (claps, voices, birds), so this check
rests on the spectrograms that were looked at, not on every flight.

*Lund.*

- **Correction to entry 022:** the Lund WAVs are 32-bit, not 24-bit. The
  durations in entry 022 were computed for 24-bit and are a third too long:
  the 13 recordings hold 90.5 min, not 143 min, and the audio after the
  ground truth is 226 s in all, not 2,010 s. The council's Devil's Advocate
  found the same independently (entry 025).
- In `10_flower3` the ground truth runs 29 s past the end of the audio. Its
  clock cannot be trusted, so it is left out of every evaluation.
- In-band level (150-4,000 Hz) follows -20 log10(range) only weakly
  (correlation -0.14 to 0.38 at zero shift, no consistent best shift), so the
  level-based clock check is inconclusive. The airfield recordings are loud
  (median -23 dBFS) and the background dominates the level.
- In the spectrograms of `01_flower` and `06_manual_close`, rotor noise
  starts at the ground-truth start; before it there are voices and beeps but
  no rotor lines. Two of 13 recordings checked by eye is not enough to use
  the stretches before take-off as negatives.

*Level by state* (dBFS, median): Nightjar day background -80.2, day drone
-76.7, night background -80.8, night drone -82.5; in the mower session the
drone stretches are at -55.8 against -71.1 for its one minute of mower
alone. d3set background -45.1, drone -47.9.

*An independent listener by range.* The four iteration 2 small CNNs never
heard these rigs. AUC of drone windows against background windows of the
same sessions (1 s windows, 1 s apart; the three DroneNoise-trained seeds as
a range, then the DDL-trained one):

| Range | Nightjar | d3set |
|---|---|---|
| 0-50 m | 0.69-0.74; 0.88 | 0.75-0.81; 0.69 |
| 50-100 m | 0.51-0.54; 0.68 | 0.67-0.75; 0.57 |
| 100-150 m | 0.41-0.45; 0.57 | 0.64-0.68; 0.59 |
| 150-200 m | 0.55-0.58; 0.54 | 0.65-0.69; 0.58 |
| 200-300 m | 0.46-0.52; 0.61 | no audio |
| 300+ m | 0.48-0.52; 0.47 | no audio |

Windows: Nightjar 331 at 100-150 m and 77 at 150-200 m; d3set 481 and 172.
Recall at 20% false alarms on the rig's own background, 100-150 m: Nightjar
0.09-0.18, d3set 0.16-0.25. Glasgow (1-5 m) against its own noise: AUC 0.66
to 0.92 for the DroneNoise models, 0.51 for the DDL model.

**Assessment.**

- **At 100-200 m the drone is at the edge of audibility for these models,**
  and for the DJI Mini 2 at Nightjar mostly below it. d3set's larger Holybro
  X500 is heard weakly (AUC about 0.65). Beyond 200 m neither model hears
  anything, which agrees with the Nightjar author's own analysis. At
  100-150 m an iteration 2 model finds about 1 drone window in 5 if 20% false
  alarms are accepted.
- The listeners are weak (trained on 18 minutes of near drones), so this is
  a floor on what is audible, not a ceiling.
- Lund's labels are usable as positives with a caveat on their clock; Lund
  supplies no negatives yet.

**Next.** The council on how to weight the data (entry 025).

## 025 - Council: how to weight toward 150 m, and how to judge it

- Date: 2026-10-14
- Iteration: 3
- Type: Council
- Commit: uncommitted (on top of 7d4f08a)

**Question.** How should iteration 3 use the new rigs, weight the data toward
about 150 m, and be evaluated under Mads's new priorities (entry 021)?

The proposal put to the council (C): train on the rigs that have their own
no-drone audio (Nightjar, d3set, Lund), balanced per rig; bring back real
positives out to 250 m; sample positives with a weight centred on 150 m;
leave one rig out; headline figure recall at 100-200 m at a threshold frozen
on validation; time to first detection per flight. Alternatives: A, keep
training on near drones only and just report at 150 m; B, weight only the
evaluation; D, simulate distance by summing near drones into real
backgrounds at a lower level; E, weight each positive by an estimated
signal-to-noise ratio instead of by range.

The brief contained two errors, found by the members: Lund's audio after the
ground truth is about 226 s, not "~2,010 s" (entry 024), and Lund's
negatives were described as "unverified" where in fact there is no evidence
either way.

**Devil's Advocate** (saw the project; rescored data read-only).

| Objection | Rank | Outcome |
|---|---|---|
| Far positives are mostly inaudible. In Nightjar's cleanest flight the 500-4,000 Hz level against the same session's no-drone audio is +9.8 dB at 0-50 m, +1.1 dB at 50-100 m, -0.5 dB at 100-150 m (AUC 0.38) and -0.7 dB at 150-200 m (AUC 0.31). In Lund `09_range_test` the level does not step up at take-off 87-127 m away. Weighting toward 150 m weights label noise. | Decisive | **Partly accepted.** Positives stop at 200 m, not 250 m; 200-300 m is left out of training; beyond 300 m is kept only as a control. Band level is a crude measure: the independent CNN listener of entry 024 still separates d3set at 100-200 m (AUC about 0.65) though barely Nightjar. Whether real 100-200 m positives help is the experiment itself, against a recipe that has none. |
| Same-rig negatives are not matched in time: they are recorded before and after flights (handling, claps, a calibration beacon). In the lawn-mower session the mower runs through 531 s of flight and only about a minute without the drone; that session is 41% of Nightjar's 100-250 m positives. | Decisive | **Accepted in part.** The mower session is taken out of training and scored separately as a confuser. Negatives are drawn from the training rig's own sessions, and night recordings no more often than night flights. The rest stands: this data has no in-flight negatives. |
| Under a recall-first metric no shortcut detector remains: a flat score out to 250 m is the goal, so failure and success look alike. | Decisive | **Accepted.** Three tripwires are built in: drone windows beyond 300 m (airborne, inaudible) must score no higher than the same sessions' background (AUC at most 0.60); recall within 50 m must not fall; the mower session is reported every run. |
| Too few independent units: each fold is one site, one rig and one drone type; 150-200 m holds 77 s (Nightjar) and 173 s (d3set). | Serious | **Stands.** Results are reported per rig with intervals from resampling flights, and the worse arm decides. |
| Time to detection has almost no events: 5 inbound crossings of 150 m in Nightjar, none in d3set. | Serious | **Answered by the Problem Solver's measure:** time to the first alarm from every second the drone spends at 100-200 m, not only from crossings. |
| Lund and d3set negatives are thin or contaminated (claps, zero padding, a parked drone); Lund `10_flower3` ground truth runs past the audio. | Serious | **Accepted.** Lund gives no negatives and is scored for recall only; `10_flower3` is left out; digital silence is skipped. d3set's claps stay in its background. |
| Recipe choices made on the same three folds amount to tuning on test. | Serious | **Accepted.** The leave-one-rig-out figures are development figures. A sealed test that can measure recall by range is asked of Mads (below). |
| Drone-only and no-drone-only sources remain in training; licences (d3set GPL, Lund none); the 08-06 evening is 33% of Nightjar negatives but 4% of its 100-200 m positives. | Weak | Noted. DDL and DroneNoise stay buried in FSD50K backgrounds as in iteration 2. Licences go to Mads. |

**Scholar** (saw no project context; 27 sources it reports checking).

- Weighting the *evaluation* by the operating range is standard and needed
  whatever is done in training (importance-weighted validation, Sugiyama et
  al. 2007).
- Weighting *training* by distance is a defensible variation: resample rather
  than reweight the loss (An et al. 2021), weight toward a band such as
  100-250 m and not monotonically toward far, because far examples carry the
  label noise. Its effect on large separable networks fades (Byrd and Lipton
  2019); for a small network on noisy audio it needs an ablation.
- Distance simulation by attenuating near recordings into real backgrounds
  is standard in form (DCASE 2017 task 2; Reuter et al. 2026 report 1 s to
  2 s windows raising recall at 100-150 m from 0.52 to 0.68), but ignores air
  absorption and ground effect; validate against real 150 m audio and never
  report on simulated data.
- Known-inaudible windows as hard positives are against practice; mask an
  ambiguous band as "don't care" (Fonseca et al. 2020). Near-range training
  keeps a detector's score tied to distance (Knight and Bayne 2019). Define
  audibility independently of the detector being trained.
- SNR weighting and curricula: defensible but weakly supported; focal loss
  would amplify the inaudible far positives.
- Evaluation: fix a false-alarm rate, report detection probability at the
  operating range with the threshold frozen on validation; count both
  windows and passes; the unit of independence is the flight (cluster
  bootstrap); time to detection as delay against a false-alarm constraint,
  with censoring; leave one setup out and report each setup separately.
- Every setup that contributes positives should contribute time-matched
  negatives; probe for shortcuts by scoring drone-absent periods inside
  positive sessions and by checking that the score falls with distance.

Departures from the Scholar's practice: one. The Scholar fixes the operating
point as false alarms *per hour* on long negative recordings. Each rig here
has 14 to 75 minutes of background, too little for a per-hour rate with a
useful interval, so the operating point is a 10% *window* false-alarm rate.

**Problem Solver** (saw the project; scored iteration 2 models on Nightjar
`flight-20260804` and d3set's distance flights). Verdict: **try a modified
version.**

- At 100-200 m the iteration 2 CNNs separate Nightjar `flight-20260804` from
  its own motors-off audio at AUC 0.65-0.80, and d3set at 0.64-0.73; beyond
  200 m at Nightjar AUC 0.29-0.65, i.e. not heard. So 100-200 m holds weak
  but real evidence; beyond 200 m is label noise. (Entry 024, with all
  sessions and same-session negatives, finds less for Nightjar.)
- Option E would discard exactly the windows that matter: at 150 m a
  per-window SNR estimate drowns in the background's own fluctuation while
  the CNN still separates them.
- Positives up to 200 m; target shares of positive draws 0.20 / 0.25 / 0.30 /
  0.25 for 0-50 / 50-100 / 100-150 / 150-200 m, split equally over the
  sources with audio in each bin; no cell drawn more than 8 times its natural
  rate, the excess moving to the next nearer bin.
- Cheapest settling experiment: small CNN, 200k windows, one seed, two arms
  (hold out Nightjar; hold out d3set), three recipes: A' near only, C' plus
  real 50-200 m by the shares above, D' near drones summed into the training
  rig's background at -10 to +5 dB. C' counts if it beats A' at 100-200 m by
  at least 10 points on the worse arm (three seeds), loses at most 5 points
  within 50 m, and keeps the beyond-300 m control at AUC 0.60 or below.
- Asks: what "time" means; release ITU ARIS and RODENT (both CC BY, both with
  same-session negatives) as a sealed test now; iteration 4 candidates: a
  multi-microphone frontend (on Nightjar the plain channel mean raised recall
  at 100-200 m by 0.14-0.23 for all three models), pretrained embeddings.
- Confidence: medium.

**Agreement and dissent.** All three agree that positives beyond about
200 m must not be trained on, that E is not the way, and that a simulated
distance arm belongs next to the real one. The Devil's Advocate and the
Problem Solver disagree on whether 100-200 m holds usable evidence. Their
measurements differ in what they compare (band level against a CNN score;
one session against all) and both are weak at Nightjar. The disagreement
turns on a fact that only training can settle: does a model trained with
same-rig negatives and real 100-200 m positives find more drones at
100-200 m on a rig it never heard? That is the experiment.

**Decision.** Iteration 3 runs the Problem Solver's experiment with the
Devil's Advocate's tripwires.

1. *Labels.* Positive: motors on and slant range at most 200 m. 200-300 m:
   not used. Beyond 300 m: not trained on; scored as a control.
2. *Rigs.* Two leave-one-rig-out arms, Nightjar and d3set (the only rigs with
   negatives). Validation: Nightjar `flight-20260804-3`, d3set's own
   `Validation` flights. Lund (minus `10_flower3`) and Glasgow are scored in
   both arms as further unseen rigs, Lund for recall only. The mower session
   is never trained on and is reported separately.
3. *Sampling.* Per arm, the training rig plus DDL within 50 m and DroneNoise
   near the pass-by (both buried in FSD50K backgrounds as in iteration 2).
   Bin shares as above, cap 8. The rig supplies the same share of negatives
   as of positives; FSD50K the rest; night recordings no more often than
   night flights. Recipes A' (`near`), C' (`range`), D' (`simdist`: 20% real
   near drones, 80% near drones summed into the training rig's background at
   -10 to +5 dB, gain only).
4. *Figures* (proposed to Mads as the new primary metric in
   `REQUIREMENTS.md`). Ranking: recall at 100-200 m on the held-out rig at
   10% window false alarms on that rig's own background, with a 95% interval
   from resampling flights; the worse arm decides. Deployment: threshold
   frozen at 10% false alarms on the training rig's validation background;
   held-out recall at 100-200 m and the false alarms it produces. Time: from
   every second at 100-200 m, the share of starts with an alarm within 5 s,
   counting the window's own length. Recall in every range bin always.
5. *Rule, fixed now.* A recipe beats A' if, on the worse arm, recall at
   100-200 m at 10% false alarms is at least 10 points higher, confirmed with
   three seeds; recall within 50 m drops by no more than 5 points; and the
   beyond-300 m control stays at AUC 0.60 or below. Baselines `comb` and
   `lgbm` alongside.
6. *For Mads:* what "time" means; whether ITU ARIS and RODENT may be fetched
   as a sealed test; the licence row (d3set GPL-3.0, Lund none stated).

**Next.** Open iteration 3 and run the screen.

---

# Iteration 3 - Same-rig negatives, positives weighted toward 150 m

> **Changed from iteration 2:**
> 1. *Data.* Two new rigs that recorded both drones and no drones (Nightjar,
>    d3set) become the training and test rigs; each supplies as many
>    negatives as positives. DDL and DroneNoise stay as extra near drones.
> 2. *Positives.* Out to 200 m instead of 50 m, drawn by fixed shares per
>    range bin (0-50 / 50-100 / 100-150 / 150-200 m: 0.20 / 0.25 / 0.30 /
>    0.25, at most 8 times a bin's natural rate). A recipe with simulated
>    distance runs alongside.
> 3. *Judgement.* The primary figure follows Mads's new priorities: recall at
>    100-200 m on a rig never trained on, at 10% false alarms on that rig's
>    own background; time to detection; false alarms last.
>
> **Why:** Mads's priorities (entry 021). New same-rig negatives (entries
> 022-023) remove the reason iteration 2 excluded far drones: with no
> negatives from the drone's own rig, faint positives taught the rig
> (entries 013, 018). Entry 024 shows that 100-200 m is at the edge of
> audibility, so whether far positives now help has to be measured.
>
> **Kept the same:** frontend (16 kHz, 1 s window, 64 log-mel bands, in-band
> level normalisation), the small CNN and LightGBM, 200,000 training
> windows, seed 0 for the screen, burying near drones in FSD50K
> backgrounds, the random equalisation, the sealed test sources (unscored).
>
> **Hypothesis:** with negatives from the same rig, real drones at 50-200 m
> in training raise recall at 100-200 m on an unseen rig by at least 10
> points over training on near drones only, without costing recall within
> 50 m and without the beyond-300 m control firing.
>
> **Judged by:** the rule in entry 025, point 5.

## 026 - Training and evaluation code for iteration 3

- Date: 2026-10-14
- Iteration: 3
- Type: Infrastructure
- Commit: uncommitted (on top of 7d4f08a)

**What changed.** Added `src/dad/chains.py` (windows of the new rigs with the
range at each window's centre), `src/dad/rangerun.py` (the sampler and the
evaluation of entry 025), `scripts/run_range.py`, `scripts/probe_chains.py`
and `tests/test_rangerun.py`. The iteration 2 code (`realrun.py`,
`run_real.py`) is unchanged and is reused for the network, LightGBM, the
FSD50K backgrounds and the mixing.

**Why.** Entry 025 changes what is sampled and what is reported; nothing in
the iteration 2 code knew about range bins, same-rig negatives or time to
detection.

**How.** What the code fixes in place:

- *Arms* (`make_arm`). Hold out Nightjar: train on d3set except its
  `Validation` flights, which set the deployment threshold. Hold out d3set:
  train on Nightjar except `flight-20260804-3` (validation) and the mower
  session (never trained on; scored as a confuser in both arms). Lund
  without `10_flower3`, and Glasgow, are scored in both arms.
- *Positive cells* (`RangeSampler`, `set_weights`). One cell per source and
  range bin, on a 0.1 s grid with +-50 ms jitter. Range is the slant range at
  the window's centre. DDL within 50 m and DroneNoise near the pass-by are
  cells in the 0-50 m bin. Bin shares are split equally over the sources
  with audio in the bin; a cell above 8 times its natural share is cut back
  and the excess moves to the same source's next nearer bin.
- *Mixing.* Near (0-50 m) positives are buried in an FSD50K background at
  -5 to +15 dB, as in iteration 2. Positives at 50-200 m are used as
  recorded. `simdist` positives: a near drone summed into a background window
  of the training rig at -10 to +5 dB. Half of all negatives get a second
  FSD50K background at -10 to +10 dB, and every window the random
  equalisation, as in iteration 2.
- *Negatives.* The training rig supplies the share of negatives equal to its
  share of positives; FSD50K the rest. Within Nightjar, each condition (day,
  night) is drawn in the proportion it has among positives within 200 m.
- *Evaluation* (`evaluate_chain`). Held-out windows 1 s long every 0.5 s
  (Glasgow every 2 s), digital silence skipped. At 5, 10 and 20% false
  alarms on the held-out rig's own background: recall per range bin and at
  100-200 m, a 95% interval from resampling sessions, and time to detection
  (`ring_latency`): from every second at 100-200 m, the wait until the end of
  the first window that starts then or later and crosses the threshold;
  censored as a miss once the drone is beyond 200 m. AUC per range bin
  against the same sessions' background; the beyond-300 m row is the
  control. The same at the deployment threshold.

A smoke run (3,200 training windows, one arm, two recipes) ran end to end on
the GPU at about 600 windows per second; its numbers are not results. The
test suite passes (4 new tests).

**Result.** Not applicable.

**Assessment.** One weakness to keep in mind: the recipe shares are applied
before the cap, and the training rigs' far bins are small. With Nightjar as
the training rig the `range` recipe actually draws 0-50 m 6.7% per source
(20% in all), 50-100 m 46%, 100-150 m 27%, 150-200 m 6% (the cap moves most
of the 150-200 m share to 50-100 m).

**Next.** Run the screen of entry 025.

## 027 - Requirements: Mads's answers on time, sealed test and licences

- Date: 2026-10-14
- Iteration: 3
- Type: Requirements
- Commit: uncommitted (on top of 7d4f08a)

**What changed.** Mads answered the three questions of entry 025:

1. "Time" means time to detection. `REQUIREMENTS.md`: confirmed.
2. ITU ARIS and RODENT may be fetched if it is quick. Both were fetched from
   Zenodo in 8 minutes with `download_new_datasets.ps1 -Manifest
   manifest_zenodo.tsv -Only ITU_ARIS_Lab,RODENT_ICAERUS`; MD5s match the
   host (`90eb3f25...` and `f6411d50...`), and the local copies in `data/`
   match too. They are now the sealed test and have not been scored.
3. d3set (GPL-3.0) and Lund (no licence) are allowed for internal use.
   `REQUIREMENTS.md` has a row for it.

**Why.** Mads's decisions.

**How.** Edits to `REQUIREMENTS.md`; the download as above.

**Result.** What the sealed sets hold, from their documentation only (no
audio was scored): ITU ARIS is 4,491 one-second drone segments and 1,000
background segments at 22.05 kHz, one 3.1 h session, 35 hand-annotated
intervals out to about 2 km. It gives drone presence but **no distance per
segment**, and its background comes from the first 25 minutes of the
session, not from between flights. RODENT is four WAVs (one "sound without
drone") with DJI and Litchi logs; how the logs align with the audio is not
yet established.

**Assessment.** Neither sealed set can measure recall at 150 m directly.
ITU ARIS can measure recall on audible, mostly distant drones and false
alarms on a new rig; RODENT may give range if its logs can be aligned.

**Next.** Score the screen of entry 025.

## 028 - Screen: real far drones, simulated distance and near only

- Date: 2026-10-14
- Iteration: 3
- Type: Evaluation
- Commit: uncommitted (on top of 7d4f08a)

**What changed.** The three recipes of entry 025, both leave-one-rig-out
arms, small CNN and LightGBM (one seed), and the comb.

**Why.** Entry 025, point 3.

**How.**

```powershell
powershell -File scripts\experiments\run_it3_screen.ps1
```

Small CNN 200,000 training windows (about 5 min each on the GPU), LightGBM
40,000, seed 0. Held-out rig scored with 1 s windows every 0.5 s. Held out
Nightjar: 8,398 drone and 8,874 background windows, of which 590 drone
windows at 100-200 m from 6 sessions. Held out d3set: 13,239 drone and 1,356
background windows, 1,307 drone windows at 100-200 m from 7 flights. The
deployment threshold rests on 83 validation background windows when d3set
trains (its `Validation` flights) and 1,482 when Nightjar trains. Results in `artifacts/runs/it3_screen.json`; tables by
`scripts/summarise_range.py it3_screen`.

**Result.** Recall at 100-200 m at 10% window false alarms on the held-out
rig's own background (95% interval from resampling flights), recall within
50 m at the same threshold, and the beyond-300 m control (AUC against the
same sessions' background; Nightjar only, d3set has no audio there):

| Held out | Recipe | Model | Recall 100-200 m | Interval | Recall 0-50 m | Control >300 m |
|---|---|---|---|---|---|---|
| Nightjar | near | small CNN | 0.02 | 0.00-0.03 | 0.46 | 0.41 |
| Nightjar | range | small CNN | 0.06 | 0.01-0.16 | 0.45 | 0.46 |
| Nightjar | simdist | small CNN | 0.04 | 0.02-0.07 | 0.52 | 0.44 |
| Nightjar | near | LightGBM | 0.05 | 0.01-0.07 | 0.11 | 0.41 |
| Nightjar | range | LightGBM | 0.06 | 0.00-0.12 | 0.13 | 0.41 |
| Nightjar | simdist | LightGBM | 0.02 | 0.00-0.04 | 0.16 | 0.39 |
| Nightjar | - | comb | 0.11 | 0.02-0.28 | 0.21 | 0.54 |
| d3set | near | small CNN | 0.14 | 0.09-0.19 | 0.33 | - |
| d3set | range | small CNN | 0.13 | 0.09-0.20 | 0.05 | - |
| d3set | simdist | small CNN | **0.25** | 0.12-0.41 | 0.32 | - |
| d3set | near | LightGBM | 0.14 | 0.07-0.27 | 0.32 | - |
| d3set | range | LightGBM | **0.33** | 0.21-0.43 | 0.14 | - |
| d3set | simdist | LightGBM | 0.24 | 0.10-0.33 | 0.18 | - |
| d3set | - | comb | 0.00 | 0.00-0.01 | 0.20 | - |

AUC of drone windows against the same sessions' background, small CNN,
100-150 / 150-200 m: Nightjar near 0.52 / 0.51, range 0.52 / 0.40, simdist
0.43 / 0.48; d3set near 0.67 / 0.68, range 0.68 / 0.74, simdist 0.73 / 0.74.
LightGBM on d3set: range 0.83 / 0.83, simdist 0.82 / 0.81.

Further figures, in `it3_screen.json`:

- *Deployment threshold* (10% false alarms on the training rig's
  validation sessions) does not carry across rigs. On the held-out rig it
  gives false-alarm rates from 0.1% to 85% depending on model and arm (small
  CNN, simdist: Nightjar 56%, d3set 17%).
- *Mower session.* With Nightjar as the training rig (and the mower session
  never seen), the mower alone triggers 53-70% of windows at the d3set arm's
  own 10% threshold.
- *Comb.* Its scores are exactly 0 on most windows; the 20% column of the
  first pass was meaningless (every tied window counted as an alarm). The
  thresholds were made tie-safe (`fa_threshold`) and the comb rescored
  (entry 029). The comb figures above are at 10%, where the tie did not bind
  on the held-out rig.
- *Time to detection* at 10% window false alarms: the share of moments at
  100-200 m followed by an alarm within 5 s was 0.15-0.26 (Nightjar) and
  0.45-0.81 (d3set). **These are not interpretable as they stand:** at 10%
  false alarms per window and a decision every 0.5 s, the background alone
  raises an alarm within 5 s much of the time. A no-drone reference for the
  same figure was added to the evaluation after this screen (entry 029).

**Assessment.**

- **Against the rule of entry 025, no recipe passes.** The worse arm is
  always Nightjar, where every recipe and model stays at 0.02-0.11. The best
  gain there is 4 points (range, small CNN), against 10 required.
- **Nightjar at 100-200 m is at the floor for everything,** including the
  comb. Its drone (DJI Mini 2, 249 g) is not separable from the site's own
  background at 100-200 m by any model here (AUC 0.40-0.58). This repeats
  entry 024's finding with models trained on same-rig negatives. With one
  microphone, a Mini 2 at 150 m at this site is below what these detectors
  hear.
- **On d3set the weighting does something.** Simulated distance lifts the
  small CNN from 0.14 to 0.25 with near recall unchanged (0.33 to 0.32);
  real far positives lift LightGBM to 0.33 but cost it 18 points within
  50 m, and cost the small CNN 28 points within 50 m with no gain at
  100-200 m. One seed each; whether the simulated-distance gain is larger
  than the seed spread is tested next.
- **The control did not fire** (0.39-0.54), so no recipe learned a session
  cue that marks inaudible flight.
- **Near recall is low on both rigs** (0.32-0.52 within 50 m at 10% false
  alarms), which is far below what iteration 2 suggested for near drones.
  The held-out rigs' backgrounds (voices, claps, birds, machinery) are
  harder negatives than FSD50K validation was.

**Next.** Seeds 1 and 2 of `near` and `simdist`, and the comb with tie-safe
thresholds.

## 029 - Three seeds, the comb rescored, and iteration 3's verdict

- Date: 2026-10-14
- Iteration: 3
- Type: Evaluation
- Commit: uncommitted (on top of 7d4f08a)

**What changed.** Seeds 1 and 2 of the small CNN on the `near` and
`simdist` recipes, both arms. The comb rescored with tie-safe thresholds
(`fa_threshold`: the lowest threshold whose false-alarm rate does not exceed
the budget). From seed 1 on, the evaluation also reports how often the
background alone raises an alarm within 5 s (`latency_background`), the
reference without which a detection time means little (entry 028).

**Why.** Entry 028: the one promising gain (simulated distance on d3set,
+11 points) rested on one seed.

**How.**

```powershell
powershell -File scripts\experiments\run_it3_seeds.ps1
```

Results in `artifacts/runs/it3_comb.json`, `it3_seed1.json`,
`it3_seed2.json`.

**Result.** Small CNN, recall at 100-200 m at 10% window false alarms on the
held-out rig, seeds 0 / 1 / 2 and mean:

| Held out | near | simdist | Paired gain |
|---|---|---|---|
| Nightjar | 0.017 / 0.019 / 0.010, mean **0.015** | 0.037 / 0.039 / 0.036, mean **0.037** | +2.0, +2.0, +2.6 |
| d3set | 0.145 / 0.176 / 0.091, mean **0.137** | 0.253 / 0.183 / 0.181, mean **0.206** | +10.8, +0.7, +9.0 |

Recall within 50 m at the same threshold, mean of three seeds: Nightjar
near 0.44, simdist 0.52; d3set near 0.34, simdist 0.34. Beyond-300 m
control (Nightjar): 0.39-0.47 for all six runs.

Alarm within 5 s from a moment at 100-200 m, against the same from a moment
of background (seeds 1 and 2, 10% false alarms per window):

| Held out | near | simdist |
|---|---|---|
| Nightjar | 0.15 vs 0.30; 0.09 vs 0.28 | 0.23 vs 0.29; 0.18 vs 0.29 |
| d3set | 0.51 vs 0.33; 0.35 vs 0.39 | 0.52 vs 0.33; 0.51 vs 0.36 |

The comb with tie-safe thresholds: recall at 100-200 m 0.11 (Nightjar) and
0.00 (d3set) at 10% false alarms; at the deployment threshold its false
alarms are 14% and 15%, so its deployment figures in entry 028 (100%) were
an artefact of the ties.

**Assessment.**

- **The hypothesis of iteration 3 is rejected.** Real drones at 50-200 m in
  training did not raise recall at 100-200 m by 10 points on either arm
  (entry 028: +4 and -2 points, and on d3set they cost 28 points within
  50 m). Simulated distance did better but also misses the rule: +2.2 points
  on the worse arm (Nightjar), +6.9 on d3set with a seed spread of about
  10 points. It was never worse than `near` in any of the six pairs, and on
  Nightjar it raised recall within 50 m by 8 points.
- **On Nightjar, at 100-200 m, the detector is worse than no detector:** an
  alarm follows a moment at 100-200 m less often than it follows a moment of
  background (0.09-0.23 against 0.28-0.30). The DJI Mini 2 at that range is
  not in the single-microphone audio in a form any model here can use.
- **On d3set, a drone at 100-200 m is caught within 5 s about half the
  time,** against about a third from background alone. That margin, not the
  0.5, is what the detector adds.
- **Thresholds do not carry from one rig to another** (deployment false
  alarms 17% to 74% for the small CNN, three seeds, at a threshold set for
  10%).

**Verdict on iteration 3: hypothesis rejected; target not set.** Weighting
the training data toward 150 m, whether with real far drones or with
simulated distance, does not bring a 150 m drone within reach of a
single-microphone detector on an unseen rig. Recall at 100-200 m at 10%
false alarms is about 0.04 (DJI Mini 2, Nightjar) and 0.21 (Holybro X500,
d3set), mean of three seeds, best recipe. No numeric target exists for the
new primary metric; even so, these numbers are far from anything usable.
The limit is how much drone the audio holds at 150 m, not how the training
data is weighted.

**What iteration 4 changes because of it.** Per-window evidence at
100-200 m is weak (AUC 0.6-0.75 on d3set), and Mads ranks recall above time.
The cheapest way to gather more evidence per decision is to listen for
longer: average the scores of several consecutive windows, or use a longer
window. That trades detection time for recall, exactly the trade the new
priorities name. `simdist` is kept as the training recipe alongside `near`
because it was never worse in six pairs; it was not adopted by the rule.

---

# Iteration 4 - Listen longer: trade detection time for recall at 150 m

> **Changed from iteration 3:** the decision. Iteration 3 decided on one
> 1 s window every 0.5 s. Iteration 4 decides on the average score of k
> consecutive windows (k = 1, 2, 4, 8, spanning 1, 1.5, 2.5 and 4.5 s of
> audio), and tries a model trained and scored on 2 s windows.
>
> **Why:** entry 029. At 100-200 m the evidence in one 1 s window is weak
> (AUC 0.6-0.75 on d3set, chance on Nightjar), and changing what the model
> is trained on did not fix it. Averaging over time is the standard way to
> raise the signal-to-noise ratio of a weak, persistent source, and it costs
> exactly the quantity Mads ranks second: time. Both levers were in the plan
> the council reviewed (entry 025: k-of-n smoothing chosen on validation;
> 1 s against 2 s windows judged on time to detection), so no new council.
> The window length is FREE within 0.25-2.0 s in `REQUIREMENTS.md`.
>
> **Kept the same:** data, arms, validation sessions, the `near` and
> `simdist` recipes, the small CNN, 200,000 training windows, seeds 0-2 (the
> iteration 3 checkpoints are rescored, not retrained), frontend except the
> window length in the 2 s arm, the sealed sets (unscored).
>
> **Hypothesis:** averaging over 2.5-4.5 s raises recall at 100-200 m at 10%
> false alarms per decision by at least 10 points on d3set over a single
> 1 s window (mean of three seeds), and widens the margin between "alarm
> within 5 s at 100-200 m" and the same figure on background.
>
> **Judged by:** on the d3set arm, mean of three seeds: recall at 100-200 m
> at 10% decision false alarms, at least +10 points over k = 1; margin of
> alarm within 5 s (and 10 s) over the no-drone rate, not smaller than at
> k = 1; recall within 50 m not lower. The Nightjar arm is reported but
> cannot decide: no model there beats chance at 100-200 m (entry 029), so a
> worse-arm rule would fail every option by construction. Nightjar must not
> lose recall within 50 m.

## 030 - Averaging over consecutive windows

- Date: 2026-10-14
- Iteration: 4
- Type: Algorithm
- Commit: uncommitted (on top of 7d4f08a)

**What changed.** The decision is the mean score of k consecutive 1 s windows
(`integrate` in `src/dad/rangerun.py`; windows every 0.5 s, runs broken by a
skipped silent window are dropped, the range is taken at the middle window).
The threshold is set on the averaged background scores, so the false-alarm
budget is 10% of decisions at every k. The alarm time counts the whole
averaged span (`window + (k - 1) * hop`). No model was retrained: the twelve
small-CNN checkpoints of entries 028-029 (two recipes, two arms, three seeds)
were rescored.

**Why.** The box of iteration 4.

**How.**

```powershell
powershell -File scripts\experiments\run_it4_integrate.ps1
uv run python -I scripts/aggregate_range.py it4_int_it3_screen it4_int_it3_seed1 it4_int_it3_seed2
```

**Result.** Mean of three seeds (range over seeds in brackets), 10% false
alarms per decision on the held-out rig's own background:

| Held out | Recipe | Averaged span | Recall 100-200 m | Recall 0-50 m | Alarm within 5 s at 100-200 m / on background | Within 10 s / on background |
|---|---|---|---|---|---|---|
| d3set | simdist | 1 s (k = 1) | 0.206 (0.18-0.25) | 0.340 | 0.55 / 0.34 | 0.72 / 0.47 |
| d3set | simdist | 1.5 s (k = 2) | 0.225 (0.19-0.28) | 0.356 | 0.50 / 0.28 | 0.67 / 0.41 |
| d3set | simdist | 2.5 s (k = 4) | 0.247 (0.21-0.32) | 0.383 | 0.39 / 0.19 | 0.57 / 0.29 |
| d3set | simdist | 4.5 s (k = 8) | **0.276** (0.22-0.35) | 0.410 | 0.30 / 0.11 | 0.46 / 0.21 |
| d3set | near | 1 s (k = 1) | 0.137 (0.09-0.18) | 0.341 | 0.45 / 0.35 | 0.63 / 0.49 |
| d3set | near | 4.5 s (k = 8) | 0.181 (0.10-0.25) | 0.411 | 0.20 / 0.12 | 0.33 / 0.23 |
| Nightjar | simdist | 1 s (k = 1) | 0.037 (0.04-0.04) | 0.523 | 0.22 / 0.30 | 0.37 / 0.41 |
| Nightjar | simdist | 4.5 s (k = 8) | 0.022 (0.01-0.04) | 0.602 | 0.03 / 0.11 | 0.08 / 0.19 |
| Nightjar | near | 1 s (k = 1) | 0.015 (0.01-0.02) | 0.439 | 0.13 / 0.28 | 0.29 / 0.38 |
| Nightjar | near | 4.5 s (k = 8) | 0.002 (0.00-0.01) | 0.487 | 0.00 / 0.11 | 0.03 / 0.18 |

AUC at 100-200 m against the same sessions' background, d3set simdist:
0.68 at k = 1, 0.73 at k = 8. Beyond-300 m control (Nightjar): 0.39-0.46 at
every k.

**Assessment.**

- **Averaging helps on d3set, steadily but not enough for the rule.**
  Recall at 100-200 m rises with every doubling of k and in every seed, to
  +7.0 points at 4.5 s (simdist) and +4.4 (near). The rule asked for +10.
  Recall within 50 m rises too (+7 points).
- **What it costs in time is visible in the alarm figures.** At the same
  10% of decisions, an averaged detector raises fewer separate alarms on
  background (within 5 s: 0.34 at 1 s, 0.11 at 4.5 s), because consecutive
  averaged decisions are correlated. Its lead over background within 10 s is
  about the same at every k (0.25, 0.26, 0.28, 0.25). So at a fixed
  false-alarm budget per decision, averaging does not buy faster detection;
  it buys a better recall per decision and fewer false-alarm episodes.
- **On Nightjar averaging makes 100-200 m worse,** because the score there
  is slightly lower with a drone than without one (AUC 0.44-0.49), and
  averaging sharpens whatever the score does. This is the same finding as
  entry 029 from another angle: there is no drone evidence to integrate.

**Next.** The 2 s window, trained and scored as such (the iteration 4 box).

## 031 - A 2 s window, and iteration 4's verdict

- Date: 2026-10-14
- Iteration: 4
- Type: Model
- Commit: uncommitted (on top of 7d4f08a)

**What changed.** The small CNN trained and scored on 2 s windows instead of
1 s (the frontend's FREE window setting), both recipes, both arms, seed 0;
scored with 1, 2 and 4 consecutive 2 s windows averaged.

**Why.** The second lever of the iteration 4 box. Entry 019 found a 2 s
window the one change that helped in iteration 2 (about 4 points, one seed),
and the Scholar cites a published gain at 100-150 m from 1 s to 2 s
(entry 025).

**How.**

```powershell
powershell -File scripts\experiments\run_it4_win2.ps1
uv run python -I scripts/aggregate_range.py it4_win2 it4_int_it3_screen
```

About 10 minutes of training per model. Result in
`artifacts/runs/it4_win2.json`.

**Result.** Seed 0, recall at 100-200 m at 10% false alarms per decision on
the held-out rig, against the 1 s model of the same seed averaged over a
similar span:

| Held out | Recipe | 1 s, 1 window | 1 s, 4 windows (2.5 s) | 1 s, 8 windows (4.5 s) | 2 s, 1 window | 2 s, 4 windows (3.5 s) |
|---|---|---|---|---|---|---|
| d3set | simdist | 0.253 | 0.316 | 0.352 | 0.246 | 0.291 |
| d3set | near | 0.145 | 0.190 | 0.195 | 0.090 | 0.096 |
| Nightjar | simdist | 0.037 | 0.020 | 0.005 | 0.015 | 0.012 |
| Nightjar | near | 0.017 | 0.003 | 0.000 | 0.009 | 0.007 |

Recall within 50 m, 2 s against 1 s single windows: d3set simdist 0.31 vs
0.32, near 0.38 vs 0.33; Nightjar simdist 0.41 vs 0.52, near 0.49 vs 0.46.
Beyond-300 m control with 2 s windows: 0.33-0.48.

**Assessment.** A model that sees 2 s at once is no better than one that
sees 1 s, and worse than averaging the scores of 1 s windows over the same
span. One seed; the difference to averaged 1 s windows (0.25 against 0.32 at
about 2.5 s on d3set simdist) is inside the seed spread of entry 030, so the
safe reading is "no gain", not "a loss". This differs from iteration 2's
small 2 s gain (entry 019), which was measured against FSD50K negatives.

**Verdict on iteration 4: hypothesis rejected by its rule, with a small,
consistent gain.** Averaging 8 consecutive 1 s windows (4.5 s) raises recall
at 100-200 m on d3set from 0.21 to 0.28 (three seeds; +7 points against +10
required) and recall within 50 m from 0.34 to 0.41, in every seed, at no cost
except decision span. A 2 s window adds nothing. On Nightjar nothing helps
at 100-200 m, because there is nothing to integrate.

Where the work stands against the new priorities:

| Priority | Best honest figure (held-out rig, 10% false alarms per decision) |
|---|---|
| Recall at 100-200 m | Holybro X500 (d3set): 0.28 with 4.5 s averaging, 0.21 without. DJI Mini 2 (Nightjar): 0.04, not better than chance. |
| Time to detection | X500 at 100-200 m: alarm within 10 s from 46-72% of moments, against 21-47% from background alone; the lead is about 25 points at every averaging span. |
| False alarms | 10% of decisions by construction; a threshold set on one rig gives 17-74% on another. |

**What the next iteration would need, and why it stops here.** Two
iterations changed the training data and the decision and moved recall at
150 m by a few points. CLAUDE.md asks for a check-in when a requirement looks
unreachable with the available data and when the next step needs a decision
only Mads can take. Both apply:

1. **The most promising lever is more microphones.** The Problem Solver
   measured on Nightjar that simply averaging the 15 live UMA-16 channels
   raised recall at 100-200 m by 0.14-0.23 for three iteration 2 models
   (entry 025). All three new rigs are arrays (16, 16 and 12 microphones),
   so this can be tested on the data at hand. It presupposes a deployed
   array, and deployment hardware is Mads's open decision. It also changes
   the frontend, which needs the council.
2. **Which drone matters.** At 150 m a 249 g DJI Mini 2 is not separable
   from the site's background by any model tried; a 500 mm Holybro X500 (its
   weight is not given in the dataset) is,
   weakly. If the drones Mads cares about are Mini-class, 150 m with one
   microphone is out of reach on this evidence.
3. **A number for "high recall".** The new primary metric has no target yet.
4. **The sealed test.** ITU ARIS has drone presence but no range; scoring it
   is worthwhile only for a candidate Mads wants to keep.

**Next.** Check in with Mads.

## 032 - Requirements: the operational scenario, the deployment array, Doppler

- Date: 2026-10-14
- Type: Requirements
- Commit: uncommitted (on top of 7d4f08a)

**What changed.** Mads answered the questions of entry 031 and two more:

| Question | Mads's answer | In `REQUIREMENTS.md` |
|---|---|---|
| Which drones | small drones | Drone class: small drones |
| Deployment microphones | "an array of 2 omnidirectional microphones" | Deployment microphones |
| Target | as high as can be reached; the most important thing is to recognise drones at a distance in heavy noise, fairly quickly: at least 3 s before the drone reaches the soldier; at about 50 m/s that is 150 m | Operational scenario; Recall target |
| Fetch Acoustic UAV Field Recordings (FPV, 4.2 GB, Zenodo) | yes | row added; download started |
| Resample real drone recordings by about +-20% to imitate Doppler and higher motor speed | yes | Pitch or speed perturbation: allowed (amends the LOCKED real-recordings row) |

Mads also asked why the project tests on specific sets instead of pooling all
data and trying different splits. The answer given: a random split of pooled
data puts neighbouring windows of one recording (same microphone, site,
background and day) on both sides, so it measures recognition of the
recording, not of the drone (iteration 1: validation AUC 0.99 while the score
rose with distance); and choosing the split by its result picks the most
optimistic number. But pooling for training is right: iteration 3 trained each
arm on one new rig only. Pooling every rig but the one held out, rotating the
held-out rig, is the proposal for iteration 5, with the three kinds of split
(random, by session, by rig) run side by side on the same pool to show the
difference in numbers.

**Why.** Mads's decisions.

**How.** Edits to `REQUIREMENTS.md`.

**Result.** Not applicable.

**Assessment.** The scenario changes what "150 m" and "time" mean:

- The alarm must be raised while the drone is still at least 150 m away. At
  50 m/s the drone covers 300 m to 150 m in the last 3 s before that line,
  so the windows that matter are those at 150-300 m, not 100-200 m.
- Time is now a hard budget, not a second priority to trade: every second of
  listening is 50 m of approach. Iteration 4's 4.5 s averaging spans 225 m
  and does not fit. Decisions must be short.
- A drone at 50 m/s is heard about 17% higher in pitch while it approaches
  (343 / (343 - 50) = 1.17), and small fast drones (FPV) run their motors
  much faster than the hovering and cruising drones in the data so far. The
  data has almost none of either; the FPV set and speed perturbation address
  this.
- Two omnidirectional microphones can give at most about 3 dB against noise
  that differs between them (wind, self-noise), less against noise that
  reaches both alike. The 15-channel gain measured in entry 025 does not
  carry over.
- "Heavy noise" makes the negatives matter again: false alarms are the last
  priority, but a detector that alarms on every engine is not usable.

**Next.** Council on the design of iteration 5.

## 033 - Council: pooling, splits and the metric for the operational scenario

- Date: 2026-10-14
- Iteration: 5
- Type: Council
- Commit: uncommitted (on top of 7d4f08a)

**Question.** How should iteration 5 be designed after entry 032: the
pooling and split protocol (including Mads's question "why not one big pool
and different splits?"), the metric for "alarm before a ~50 m/s small drone
reaches 150 m in heavy noise", and the order of changes (pooling all rigs,
speed perturbation, the FPV set, a 2-microphone frontend)?

Proposal P: train on all rigs but the held-out one, rotating; report random,
by-session and by-rig splits side by side; primary metric per-window recall
at 150-300 m plus an event estimate; decisions of at most about 1 s; speed
perturbation of drone windows; 2-mic frontend later. Alternatives: A, keep
one training rig per arm; B, Mads's literal proposal (pool, random splits,
pick the best); C, P without the split comparison; D, 2-mic or FPV first.
Raw notes of the members' answers: `artifacts/council5_notes.md`.

**Devil's Advocate** (saw the project; checked range tracks and `Uav.zip`).

| Objection | Rank | Outcome |
|---|---|---|
| The 150-300 m metric has almost no data: Nightjar 361 s, of which 167 s is the mower session and the rest is at chance for every model; d3set 174 s, all at 150-190 m, hovering (median range rate 0.1 m/s); no inbound 300-150 m pass faster than 9 m/s anywhere. | Decisive | **Accepted.** The range metric is kept as a diagnostic on d3set, and the scenario is carried mainly by recall on held-out FPV campaigns (small fast drones), which have no range. Stated wherever the figures appear. |
| P pulls the training-mix lever a third time although entry 031 says the limit is the audio; the 2-mic frontend the deployment fixes comes last. | Decisive | **Partly accepted.** Pooling is still run, because it is the first test of data *diversity* (iteration 3 trained on one new rig per arm), and it answers Mads's question. The 2-mic question is measured in this iteration by rescoring on channel pairs, not left to the end. |
| FPV devices recorded the same flights at the same time (labels shifted 8-12 s between phones); a duplicate background file; the shipped splits leak; labels mark activity in long blocks. | Serious | **Accepted.** FPV is held out by campaign, never by device; the shipped `ProcessedAudioSplits` are not used; windows are cut from `labeled_data`. Label quality stands as a limit. |
| The event metric built from hovering windows, and an "own-background" threshold a deployed system never has, flatter the result. | Serious | **Accepted.** The event figure is always paired with the same figure on background (the lead). The own-background threshold is named for what it is: a site calibration (listening to the site for some minutes before use), reported next to the transferred-threshold false-alarm rate. |
| Slowing a 16 kHz recording leaves the top band empty, in positives only. | Serious | **Accepted.** The same random resampling is applied to negatives, so an empty top band says nothing about the label. |
| No sealed set has range. | Serious | **Stands.** Mads is asked for flight logs of the FPV campaign `ariel_25.02` and for a recording with his own 2-microphone array. |
| The random-split row will be the largest number in the report. | Weak | Kept as a labelled answer to Mads's question; it is never used to choose anything. |

**Scholar** (no project context; sources in the notes file).

- Random splits over pooled windows: against practice (adjacent windows
  leak; Roberts et al. 2017). Grouped split by session: standard, answers
  "new recordings on a known setup". Leave-one-setup-out: standard for "an
  unseen device or site" (DomainBed; DCASE 2020 task 1A). **Choosing the
  split by its result is against practice** (Cawley and Talbot 2010;
  DomainBed calls it an oracle). Reporting several labelled splits is fine.
- The operational metric is event-level: cumulative probability of an alarm
  before the target reaches 150 m, after all smoothing, against false alarms
  per hour on long deployment-like background. Extrapolating from slow to
  fast targets has no validated method; bracket it and label it.
- Speed perturbation is standard (Ko et al. 2015; Salamon and Bello 2017);
  for drones the measured gain is small (Reuter et al. 2026). Resample the
  drone recording before mixing, and give every class the same bandwidth.
- Two microphones: at most 3 dB against noise that differs between them
  (wind), close to 0 dB against diffuse noise at low frequency when they are
  close together.
- More devices and sites help generalisation (DCASE 2020; Vilhurin et al.
  2026), unsettled against pretraining.

Departures from the Scholar's practice: false alarms stay a per-window rate
plus the background-alarm reference, not alarms per hour; the backgrounds
per rig are minutes to an hour long, too short for a per-hour rate with a
useful interval.

**Problem Solver** (saw the project; inspected the complete FPV set; ran a
CPU split experiment). Verdict: **try a modified version.**

- The FPV set is three campaigns (`ariel_25.02`, `palmahaim_10.02`,
  `PALMAHIM27_5`), phones and a DJI Mic 2, 44.1-48 kHz, drone and background
  files cut from Audacity labels. About 70 min of unique drone audio and
  60 min of background. `ariel_25.02` flew 150 m legs in four directions
  (per its experiment notes). Four iPhone files named `.wav` are M4A.
  `Electric Uavs.zip` (8 kHz roadside sensors) and `Uav.zip` (airbase
  departures and landings) do not fit the scenario.
- **Mads's question answered with numbers** (FPV `labeled_data`, 1 s
  windows, LightGBM on log-mel statistics, five repeats):

  | Split | AUC | Recall at 10% false alarms |
  |---|---|---|
  | Random 1 s windows | 0.92 | 0.80 |
  | By file | 0.84 | 0.66 |
  | By event, all devices | 0.85 | 0.70 |
  | Leave one device out, same events in training | 0.84 | 0.67 |
  | Leave one campaign out | 0.83 (0.79-0.90) | 0.63 (0.57-0.73) |

  A random split adds about 0.09 AUC and 17 points of recall that do not
  survive a new site.
- Protocol: pool everything but the held-out unit (Nightjar, d3set, each FPV
  campaign); `simdist`, 1 s windows, at most 2 averaged. Primary: probability
  of at least one alarm within any 3 s at 150-300 m, minus the same on
  background; plus recall per held-out FPV campaign. 2-mic: rescore existing
  models on 2-channel averages from the arrays now.
- Success: d3set recall at 150-200 m up by at least 5 points (three seeds)
  with the lead over background not smaller; FPV campaign recall at 10%
  false alarms at least 0.63 (the LightGBM figure above); no rig loses recall
  within 50 m.
- Confidence: medium.

**Agreement and dissent.** All three reject random splits and choosing a
split by its result. The Scholar and the Problem Solver support pooling with
a held-out setup; the Devil's Advocate holds that the binding limit is the
audio and the microphones, not the mix. That disagreement is settled by the
pooled runs against iteration 3's single-rig runs on the same held-out rigs.

**Decision.**

1. **Mads's question** is answered in the report with the Problem Solver's
   table; the random-split figure is labelled as recognition of known
   recordings and never used for a choice.
2. **Iteration 5 pools for training** and holds out one unit at a time:
   Nightjar, d3set, and each FPV campaign. Lund is scored for recall only.
   FPV is read from `labeled_data`, grouped by campaign.
3. **Two arms:** `simdist` on the pool, and the same with speed
   perturbation (factor 0.8-1.25, drawn per window, applied to positives and
   negatives alike).
4. **Figures:** recall at 10% window false alarms per held-out unit (FPV
   campaigns; d3set at 100-200 m as the range diagnostic; Nightjar reported);
   the probability of an alarm within any 3 s at 150-300 m and its lead over
   background; recall within 50 m. Thresholds set on the held-out unit's own
   background are reported as site calibration, next to the false alarms a
   transferred threshold gives.
5. **Rule, fixed now:** pooling is kept if, against iteration 3's single-rig
   `simdist` (three seeds), d3set recall at 100-200 m rises by at least
   5 points and no rig loses more than 5 points within 50 m; the FPV
   campaign mean must beat the LightGBM 0.63. Speed perturbation is kept if
   it adds at least 5 points of FPV campaign recall without costing d3set.
6. **The 2-mic rescore** follows in this iteration.
7. **For Mads:** flight logs or GPS for the FPV campaign `ariel_25.02`, and
   whether a recording with his own 2-microphone array (drone and no drone)
   is possible.

**Next.** Load the FPV set; open iteration 5.

---

# Iteration 5 - One pool, one rig held out at a time

> **Changed from iteration 4:** training uses every rig but the held-out
> one (Nightjar, d3set, Glasgow, the three FPV campaigns, DDL and DroneNoise
> near drones, FSD50K), instead of one new rig per arm; FPV campaigns become
> held-out units; speed perturbation is tried as a second arm.
>
> **Why:** Mads's question (entry 032) and the council (entry 033): data
> diversity was never tested, since iteration 3 trained each arm on a single
> new rig. The scenario's small fast drones are only in the FPV set.
>
> **Kept the same:** small CNN, 1 s window, 0.5 s hop, 200,000 training
> windows, the `simdist` recipe, the evaluation code and false-alarm budget,
> the sealed sets (unscored).
>
> **Hypothesis:** a pooled model detects drones on an unseen rig better than
> a model trained on one rig: at least +5 points at 100-200 m on d3set, and
> FPV campaign recall above the LightGBM 0.63.
>
> **Judged by:** entry 033, point 5.

## 034 - The FPV set into the store; pooled training and speed perturbation in code

- Date: 2026-10-14
- Iteration: 5
- Type: Data
- Commit: uncommitted (on top of 7d4f08a)

**What changed.**

- *Data.* `Acoustic_UAV_Field_Recordings` was fetched from Zenodo in 35 min
  (`Uav.zip`, `Electric Uavs.zip`, `Fpv.zip`; MD5s match the host and the
  local copies). `prepare_fpv` in `scripts/prepare_new_data.py` reads only
  `Fpv.zip`'s `labeled_data` drone and background files (one manifest row
  per file, channel 0, 16 kHz, group = campaign, subset = device). The
  shipped `ProcessedAudioSplits` are not read (they reuse one source segment
  across train, validate and test, entry 033). Four iPhone files named
  `.wav` are M4A and were skipped; no exact duplicates were found in
  `labeled_data`. `Electric Uavs.zip` (8 kHz sensors) and `Uav.zip` (airbase
  departures and landings) are not used.
- *Code.* `make_pool_arm` in `src/dad/rangerun.py`: train on Nightjar,
  d3set, Glasgow and the FPV campaigns except the held-out unit; the
  sampler now handles several rigs, each supplying the share of negatives
  equal to its share of positives; FPV drone audio, which has no range,
  counts as near. `speed_change` and the sampler's `speed` option resample
  every training window, drone or not, by a factor drawn per window.
  `run_range.py --pool --speed LO HI`. The evaluation adds recall over all
  drone windows (for units without range) and the share of moments at
  150-300 m followed by an alarm within 3 s, next to the same on background.

**Why.** Entry 033, decisions 2-4.

**How.**

```powershell
uv run python -I scripts/prepare_new_data.py fpv
```

**Result.** FPV in the store, unique audio per campaign is lower, because
the phones of one campaign recorded the same flights:

| Campaign | Drone | Background | Devices |
|---|---|---|---|
| `ariel_25.02` | 60.8 min | 40.3 min | DJI Mic 2, Galaxy J5 |
| `palmahim_10.02` | 27.9 min | 4.6 min | DJI Mic 2 (drone only), iPhone |
| `palmahim_27.05` | 44.2 min | 64.7 min | Galaxy J5, Huawei Mate 20 lite, Redmi Note 5, Redmi 12, two DJI Mic 2 |

A smoke run of the pooled sampler (one arm, 3,200 windows) ran end to end;
its numbers are not results. Tests pass (51).

**Assessment.** `palmahim_10.02` has 4.6 min of background, so its held-out
false-alarm thresholds rest on about 550 windows.

**Next.** The pooled screen of entry 033.

## 035 - Pooled screen, with and without speed perturbation; iteration 5 paused

- Date: 2026-10-14
- Iteration: 5
- Type: Evaluation
- Commit: uncommitted (on top of 7d4f08a)

**What changed.** The screen of entry 033: small CNN (200,000 windows) and
LightGBM (40,000), `simdist` recipe, trained on the pool with one unit held
out at a time (Nightjar, d3set, the three FPV campaigns), seed 0; the small
CNN again with every training window resampled by a factor drawn from
0.8-1.25 (`--speed 0.8 1.25`).

**Why.** Entry 033, decisions 2-3.

**How.**

```powershell
powershell -File scripts\experiments\run_it5_screen.ps1
```

34 min for the pooled runs, 40 min for the speed arm (resampling slows
training from about 700 to about 430 windows per second). Results in
`artifacts/runs/it5_pool.json` and `it5_pool_speed.json`.

**Result.** Recall at 10% false alarms per decision on the held-out unit's
own background (1 s windows, no averaging), seed 0. For comparison, the
iteration 3 single-rig `simdist` small CNN (mean of three seeds, entry 029):

| Held out | Figure | Single rig (it. 3) | Pool, LightGBM | Pool, small CNN | Pool + speed, small CNN |
|---|---|---|---|---|---|
| d3set | recall 100-200 m | 0.206 | 0.519 | 0.244 | **0.371** |
| d3set | recall 0-50 m | 0.340 | 0.337 | 0.439 | **0.567** |
| d3set | alarm within 3 s at 150-300 m / on background | - | 0.74 / 0.23 | 0.36 / 0.29 | 0.45 / 0.28 |
| Nightjar | recall 100-200 m | 0.037 | 0.017 | 0.010 | 0.036 |
| Nightjar | recall 0-50 m | 0.523 | 0.338 | 0.576 | 0.484 |
| FPV `ariel_25.02` | recall, all drone windows | - | 0.353 | 0.634 | **0.678** |
| FPV `palmahim_10.02` | recall, all drone windows | - | 0.743 | 0.793 | 0.777 |
| FPV `palmahim_27.05` | recall, all drone windows | - | 0.292 | 0.389 | **0.481** |
| FPV mean | | - | 0.463 | 0.605 | **0.645** |

AUC of d3set drone windows at 100-200 m against the same sessions'
background: pooled LightGBM 0.82, pooled small CNN 0.71. Beyond-300 m
control on Nightjar: 0.34-0.39 in all pooled runs. Averaging two windows
(k = 2) adds 1-4 points everywhere. Reference from entry 033: LightGBM on
simple features trained on FPV campaigns only, leave one campaign out, 0.63.

**Assessment.**

- **Pooling with speed perturbation is the best configuration so far**, on
  one seed: d3set recall at 100-200 m 0.37 against 0.21 for a single rig
  (+16 points), recall within 50 m 0.57 against 0.34, and FPV campaign mean
  0.645, above the 0.63 reference. Pooling alone gives +4 points on d3set at
  100-200 m (inside the seed spread) and 0.605 on FPV. Speed perturbation
  adds 13 points on d3set and 4 points on the FPV mean.
- **Against the rule of entry 033 this is not yet decided.** The rule asks
  for three seeds; only seed 0 was run, and the seed spread at 100-200 m on
  d3set is about 10 points (entry 029). Speed perturbation was to be kept for
  +5 points on the FPV mean; it gave +4 there and +13 on d3set.
- **Pooled LightGBM is the strongest d3set result (0.52 at 100-200 m) but
  the weakest on FPV (0.46).** It is a baseline; it is reported, not chosen.
- **Nightjar does not move:** the DJI Mini 2 at 100-200 m stays at chance
  in every configuration, and pooling costs it recall within 50 m in the
  speed arm (0.48 against 0.52).
- No run raised the beyond-300 m control.

**Iteration 5 is paused here at Mads's request**, with its verdict open.
What remains to finish it: seeds 1 and 2 of pooled and pooled-with-speed
(about 2.5 hours); the 2-microphone rescore on channel pairs (entry 033,
point 6); and the two questions to Mads (flight logs for `ariel_25.02`; a
recording with his own 2-microphone array).

**Next.** Wait for Mads.

## 036 - What a second microphone adds, by rescoring

- Date: 2026-10-14
- Iteration: 5
- Type: Evaluation
- Commit: uncommitted (on top of 7d4f08a)

**What changed.** A second store for Nightjar and d3set in which every
sample is the mean of two channels of the same array (`prepare_new_data.py
pairs`; Nightjar UMA-16 channels 0 and 8, d3set the first and the last
remaining channel), laid out exactly like the one-channel store (byte sizes
checked equal; the manifest is unchanged). `run_range.py --pair` reads these
stores. The seed 0 pooled checkpoints of entry 035 were rescored on them, no
retraining.

**Why.** Entry 033, point 6: Mads's deployment has two omnidirectional
microphones. The council expected at most about 3 dB from averaging two.

**How.**

```powershell
uv run python -I scripts/prepare_new_data.py pairs
powershell -File scripts\experiments\run_it5_pair.ps1
```

The spacing of the chosen pairs is not known: the array geometry files are
not in the downloaded data. For UMA-16 both channels lie within its roughly
13 cm aperture.

**Result.** Small CNN, seed 0, 10% false alarms per decision, one channel
against the two-channel mean, same model:

| Held out | Model | Recall 100-200 m | Recall 0-50 m | All drone windows |
|---|---|---|---|---|
| d3set | pool | 0.244 to 0.314 | 0.439 to 0.501 | 0.412 to 0.467 |
| d3set | pool + speed | 0.371 to 0.338 | 0.567 to 0.530 | 0.534 to 0.497 |
| Nightjar | pool | 0.010 to 0.019 | 0.576 to 0.513 | 0.347 to 0.312 |
| Nightjar | pool + speed | 0.036 to 0.012 | 0.484 to 0.506 | 0.295 to 0.312 |

**Assessment.** Averaging two microphones of one array moves recall by
-7 to +7 points with no consistent direction, which is inside the seed
spread. This fits the expectation: two closely spaced microphones gain
little against background that reaches both alike, and the models were
trained on single channels. A gain from two microphones, if any, would have
to come from training on two-channel input or from spacing them further
apart; neither can be tested well without knowing the deployment spacing or
recordings made with it.

**Next.** Seeds 1 and 2 of entry 035.

## 037 - Three seeds of the pooled runs, and iteration 5's verdict

- Date: 2026-10-14
- Iteration: 5
- Type: Evaluation
- Commit: uncommitted (on top of 7d4f08a)

**What changed.** Seeds 1 and 2 of the two pooled configurations of entry
035 (small CNN, `simdist`, pooled; and the same with speed perturbation
0.8-1.25), all five held-out units.

**Why.** Entry 033, point 5: the rule asks for three seeds.

**How.**

```powershell
powershell -File scripts\experiments\run_it5_seeds.ps1
```

About 32 min per pooled run and 45-48 min per run with speed perturbation.
Results in `artifacts/runs/it5_pool_s1.json`, `it5_pool_s2.json`,
`it5_pool_speed_s1.json`, `it5_pool_speed_s2.json`.

**Result.** Mean of three seeds (seeds 0 / 1 / 2 in brackets), recall at 10%
false alarms per decision on the held-out unit's own background, 1 s
windows. Single-rig figures are iteration 3's `simdist` (entry 029):

| Held out | Figure | Single rig | Pool | Pool + speed |
|---|---|---|---|---|
| d3set | recall 100-200 m | 0.206 | 0.205 (0.24 / 0.19 / 0.18) | **0.333** (0.37 / 0.37 / 0.26) |
| d3set | recall 0-50 m | 0.340 | 0.439 | **0.555** |
| d3set | alarm within 3 s at 150-300 m / on background | - | 0.33 / 0.27 | **0.48 / 0.28** |
| Nightjar | recall 100-200 m | 0.037 | 0.020 | 0.029 |
| Nightjar | recall 0-50 m | 0.523 | 0.607 | 0.491 |
| FPV `ariel_25.02` | recall, all drone windows | - | 0.619 | 0.677 |
| FPV `palmahim_10.02` | recall, all drone windows | - | 0.789 | 0.789 |
| FPV `palmahim_27.05` | recall, all drone windows | - | 0.435 | 0.498 |
| FPV mean | | - | 0.614 (0.61 / 0.63 / 0.61) | **0.655** (0.65 / 0.66 / 0.66) |

Beyond-300 m control (Nightjar): pool 0.33-0.38, pool + speed 0.39-0.44.

**Assessment against the rule of entry 033.**

- **Pooling alone: rejected.** d3set recall at 100-200 m is unchanged
  (0.205 against 0.206; +5 required), and the FPV mean 0.614 is below the
  0.63 reference. Pooling does raise recall within 50 m on both rigs (+10 on
  d3set, +8 on Nightjar).
- **Speed perturbation: kept, with one criterion missed narrowly.** On top of
  pooling it adds 12.8 points at 100-200 m on d3set (all three seeds above
  every pooled seed but one), 11.6 within 50 m, and 4.1 on the FPV mean,
  against the 5 asked. It costs Nightjar 12 points within 50 m relative to
  pooling alone, 3 relative to the single rig (inside the allowed 5).
- **Pool + speed against the single rig meets all three conditions:** +12.7
  at 100-200 m on d3set, FPV mean 0.655 above 0.63, no rig more than 5 points
  down within 50 m. Its lead in the scenario figure on d3set is 20 points: an
  alarm within 3 s from a moment at 150-300 m 48% of the time, against 28%
  from background alone.
- **Two microphones** (entry 036) moved recall by -7 to +7 points with no
  consistent direction.
- **The DJI Mini 2 at 100-200 m stays at chance** in every configuration.

**Verdict on iteration 5: hypothesis rejected for pooling alone; pooling with
speed perturbation is the new best recipe.** Training on every rig but the
held-out one did not by itself help at 150 m; it helped close to the
microphone. The gain at 150 m came from resampling every training window by
0.8-1.25, which imitates faster motors and Doppler shift. Best honest
figures, unseen rig, 10% false alarms per decision, three seeds: Holybro X500
at 100-200 m 0.33 (single rig 0.21); FPV drones on an unseen campaign 0.66
(no range known); DJI Mini 2 at 100-200 m about 0.03, not above chance.

**What the next iteration would change.** With speed perturbation working,
the obvious next steps are a wider range of factors and resampling done
before mixing (the Scholar's recommendation, entry 033); a final model
trained on the whole pool; and, as before, data that only Mads can provide:
flight logs for `ariel_25.02` and a recording with his own two-microphone
array.

**Next.** Report to Mads.

## 038 - Requirements: the distance requirement withdrawn for now

- Date: 2026-10-20
- Type: Requirements
- Commit: uncommitted (on top of 3fd36b4)

**What changed.** Mads withdrew the distance requirement for now, in his
words: "Lad os trække kravet om distancen tilbage for nu" ("let us withdraw
the distance requirement for now"). In `REQUIREMENTS.md`:

| Row | Before | Now |
|---|---|---|
| Range that matters most | about 150 m, data and metrics weighted toward it | withdrawn for now |
| Operational scenario | alarm at least 3 s before arrival, "i.e. while it is still at least about 150 m away" | the 3 s budget stays; the 150 m clause is withdrawn |
| Primary metric | recall at 100-200 m on an unseen rig at 10% window false alarms | withdrawn; replaced by recall on all drone windows of an unseen rig at 10% window false alarms (PROPOSED) |
| Deployment figure | held-out recall at 100-200 m | held-out recall on all drone windows |
| Time to detection | share of moments at 100-200 m followed by an alarm within 5 s | share of drone moments followed by an alarm within 5 s |

The withdrawn rows are kept and marked, not deleted. Recall by range is
still reported wherever range is known, without a target.

**Why.** Mads's decision. It followed a count of the data with known range
(this conversation, figures from entries 023 and 034): 0.46 h of drone audio
at 100-200 m in all, from Nightjar, d3set and Lund.

**How.** Edits to `REQUIREMENTS.md`.

**Result.** Not applicable.

**Assessment.**

- The replacement primary metric is Claude's proposal, not Mads's. Changing
  the primary metric is a council matter (CLAUDE.md section 4), so the
  council should see it before an iteration is judged by it.
- Without range, "at least 3 s before arrival" cannot be measured on the
  real recordings, only on files where the approach is known, such as the
  generated test files under discussion. Whether the 3 s budget stays a hard
  requirement is open.
- Two rows still refer to range and were not changed: the sealed test "for
  recall by range" (ITU ARIS and RODENT) and the 2026-10-14 priorities text
  quoted from Mads.
- Iterations 3 to 5 were judged by the withdrawn metric. Their verdicts
  stand as records; their recall over all drone windows can be recomputed
  from the saved run files without retraining (not done).

**Next.** Ask Mads whether the 3 s budget stays, and convene the council on
the replacement metric before the next iteration.

## 039 - Requirements: MoSCoW requirements for the real test

- Date: 2026-10-21
- Type: Requirements
- Commit: uncommitted (on top of 8e0b42b)

**What changed.** Mads gave a new requirements table for the coming real
test and said that distance is not as valuable anymore. It is now the
governing section of `REQUIREMENTS.md`, word for word:

| ID | Functional | Non-functional | MoSCoW |
|---|---|---|---|
| R1 | The solution should detect drone | The program should return true value when it detects a drone | Must have |
| R2 | The solution should minimize false negatives | The program should have a recall rate of 95% | Should have |
| R3 | The solution should minimize false positives | The program should introduce a maximum of 5% false positives | Should have |
| R4 | The solution should be precise | The program should have a precision rate of 95% | Could have |
| R5 | The solution should give soldiers time to react to an alert | The program should detect drones 3 seconds before impact | Must have |

He also decided that the test audio will be WAV files, mono, "decent
quality".

Rows they replace were marked, not deleted: the priority order of entry 021
(false alarms were last, now they are a Should like recall), the open recall
target (now 95%), the range-free primary metric of entry 038, the deployment
figure and the time-to-detection row (now R5). The stop condition now names
the Must and Should requirements.

**Why.** Mads's decision.

**How.** Edits to `REQUIREMENTS.md`. A second table there says how Claude
reads each requirement as a measurement; it is marked PROPOSED:

- R1: a boolean per hop, `True` at or above the threshold.
- R2, R3, R4: recall, false-positive rate and precision per window, on
  sources or rigs never trained on, all at one threshold frozen on
  validation; windows within 1 s of a drone's start or end left out.
- R5: on test audio where the approach is known, time from the first `True`
  to impact, at least 3 s. The share of approaches that must meet it was not
  given and is open.

**Result.** Not applicable.

**Assessment.**

- **R2 and R3 together are far from the best figures so far.** Iteration 5's
  best recipe, at 10% false alarms on an unseen rig, reached recall 0.66 on
  FPV drones and 0.33 on the Holybro X500 at 100-200 m; the DJI Mini 2 at
  100-200 m was at chance. The requirement asks for 0.95 at 5%. With the data
  available this looks out of reach for distant drones; how close it comes
  over all drone windows is not known, because the iteration 5 runs were not
  scored that way (the run files are on the training PC).
- **R4 depends on the drone share.** In a test file with a drone in 5 s of
  20 s (25%), recall 0.95 and 5% false positives give precision
  0.2375 / (0.2375 + 0.0375) = 0.86. Precision 0.95 at that share needs at
  most 1.7% false positives.
- **R5 brings distance back through speed.** 3 s before impact is 150 m at
  50 m/s and 30 m at 10 m/s. Distance is no longer a target, but R5 cannot be
  checked without knowing when the drone arrives, which real recordings
  mostly do not give.
- The measurement readings are Claude's. A change of primary metric is a
  council matter (CLAUDE.md section 4); the council has not seen this one.

**Next.** Mads's next task from the scrum board: "Make ML Algorithm
understand WAV", with mono WAV files of decent quality.

## 040 - The detector reads WAV files ("Make ML Algorithm understand WAV")

- Date: 2026-10-21
- Type: Infrastructure
- Commit: uncommitted (on top of 8e0b42b)

**What changed.** A task from Mads's scrum board: make the detector run on
WAV files, since the real test will hand it WAV files, mono, decent quality.

- `src/dad/wavio.py`: reads any file soundfile can read, at any sample rate
  and bit depth; keeps channel 0 of a multi-channel file (as the processed
  store does, entries 007 and 023) and says so; resamples to 16 kHz with the
  same polyphase filter as `prepare_data.py`. It then cuts one window per
  hop and scores it with the same scorer objects the training runs use, so
  a WAV is seen exactly as the evaluation windows were. Each decision is
  stamped with the time its window ends, the earliest moment it could exist
  live, and is `True` at or above the threshold (R1).
- An optional sidecar `<name>.json` with `{"drones": [{"start_s", "end_s",
  "impact_s"}]}` turns the decisions into R2 to R5: recall, false-positive
  rate and precision per window (windows straddling a drone's start or end
  left out, the reading of entry 039), the drone share of the audio, and for
  each drone with an impact time the first alarm and whether it came at
  least 3 s before impact. Any other key in the sidecar is carried into the
  summary, so the generated test files can record wind, explosion level and
  speed there.
- `uv run dad wav`: files or folders; `--detector comb` or a `.pt`
  checkpoint; `--threshold`, or `--threshold-from` a run's JSON, which looks
  up the single-window threshold that run froze on validation for that
  checkpoint; `--out` a CSV with every hop, `--summary` a JSON with totals
  and per-file results.
- README and the command list in CLAUDE.md describe it.

**Why.** Mads's task. Until now the detector only read the processed store.

**How.**

```powershell
uv run pytest
uv run dad wav <folder> --detector comb --threshold 0.5
```

**Result.** 62 tests pass, 11 of them new (`tests/test_wavio.py`): a 48 kHz
16-bit stereo file comes out 16 kHz mono from channel 0; one decision per
hop, the first at 1.0 s; straddling windows left out; recall, false
positives, precision and the lead before impact on a known file; a late or
missing alarm fails R5, and an alarm before the drone appears does not
count for it; a malformed sidecar is refused; the threshold lookup from a
run file; the command end to end; an (untrained) network checkpoint saved
and run over a WAV.

A synthetic demo file (a harmonic tone in noise, 44.1 kHz) was run through
the command with the comb at an arbitrary threshold to check the output. It
is a plumbing check and says nothing about detection.

**Assessment.**

- The tool is ready for the generated test files once they exist, and for
  any trained checkpoint. No trained checkpoint is on this machine; the
  iteration 5 models and their run files are on the training PC.
- The LightGBM baseline cannot yet be run this way: it is trained inside a
  run and not saved. The comb can, without training.
- Two-microphone input is reduced to channel 0. Mads chose mono for the
  test.

**Next.** Copy the iteration 5 checkpoints and run files from the training
PC, agree the sidecar format with whoever generates the test files, and
decide which checkpoint goes to the real test.

## 041 - Review of the WAV tool, its fixes, and its speed

- Date: 2026-10-21
- Type: Correction
- Commit: uncommitted (on top of 8e0b42b)

**What changed.** Mads ran a code review (`/ponytail-review`) on entry 040's
WAV tool and asked how it could be optimised. Six findings, all fixed:

1. **R5 could be passed by an alarm the drone did not cause.** A decision
   counted for R5 as soon as its window ended after the drone appeared, even
   if almost all of the window was earlier audio. An explosion 0.5 s before
   the drone would then count as detecting it. Now a decision counts only
   once at least half its window is drone (`lead_times`). This moves the
   earliest creditable alarm half a window later, so R5 is judged more
   strictly than in entry 040.
2. **A checkpoint could be run with the wrong window length without an
   error.** Checkpoints do not record their window. With `--threshold-from`,
   the run file's window is now compared with `--window` and a mismatch
   stops the command.
3. **Every hop of every file was kept in memory,** even without `--out`;
   about 1.7 million rows for 11,250 test files. The CSV is now written file
   by file, only when asked for.
4. A sidecar with `impact_s` before `start_s` is refused instead of counting
   as a missed drone.
5. `--threshold-from` with the comb now says how to get the comb's
   threshold (it depends on the held-out arm) instead of a misleading error.
6. A WAV is opened once instead of twice; an import moved out of a loop.

The R5 measurement row in `REQUIREMENTS.md` (PROPOSED, Claude's reading)
now states the half-window rule.

**Why.** Findings 1 and 2 would have produced wrong test results without any
sign of it; finding 1 bears on a Must requirement.

**How.** `uv run pytest`; timing with a scratch script on the synthetic
44.1 kHz demo file of entry 040 (not part of the repository).

**Result.** 63 tests pass (12 in `tests/test_wavio.py`: the R5 test now has
an explosion-like alarm just before the drone, the sidecar test a reversed
impact, the threshold test a wrong window and the comb).

Speed on this machine (12 cores, CPU only), one 20 s file, 153 decisions:

| Step | Time |
|---|---|
| Read and resample 44.1 to 16 kHz | 16 ms |
| Comb | 54 ms (0.35 ms per decision) |
| Small CNN (`dscnn`, untrained weights; same cost as trained) | 565 ms (3.7 ms per decision), of which about 114 ms log-mel |

**Assessment.**

- Live, 3.7 ms per decision against a new decision every 125 ms leaves a
  wide margin, so speed is not a constraint for the detector itself.
- For the generated test set the cost is about 0.6 s per file with the CNN:
  roughly 2 hours for 11,250 files on one process. Running files in
  parallel would cut this several times; it was not built, since nobody has
  asked for it and the file count is not settled.
- Computing log-mel once per file instead of once per window would save at
  most about a fifth of the time, and the 0.125 s hop does not fall on the
  10 ms frame grid, so the features would differ slightly from those the
  model was trained on. Not done.
- Not checked: how the models score digital silence. The training
  evaluation dropped silent windows; `dad wav` scores them.

**Next.** As entry 040: the trained checkpoints and run files from the
training PC, and the sidecar format with whoever generates the test files.

## 042 - The LightGBM baseline can be saved and run over WAV files

- Date: 2026-10-21
- Type: Infrastructure
- Commit: uncommitted (on top of 8e0b42b)

**What changed.** Mads decided that the LightGBM baseline (`lgbm`) goes into
the real test next to the comb and the CNN. Until now it was trained inside
each run and thrown away.

- `LgbmScorer.save` and `LgbmScorer.load` in `src/dad/realrun.py` (pickle,
  with the variant name, as `NetScorer` stores its own).
- `scripts/run_range.py` saves every LightGBM model it trains as
  `artifacts/models/<run>_<held-out unit>_<recipe>_lgbm.pkl`, and with
  `--load-from` rescores saved ones, as it already did for networks.
- `dad wav --detector <file>.pkl` runs it; `--threshold-from` finds its
  frozen threshold in the run file by the same naming rule.

**Why.** The working agreement requires the baselines next to any model, and
Mads wants all three in the real test.

**How.** `uv run pytest`. To produce LightGBM models with iteration 5's
recipe, on the training PC:

```powershell
.venv-dml\Scripts\python.exe -u scripts\run_range.py --name it5_pool_speed_baselines_s0 --pool --arms nightjar d3set fpv:ariel_25.02 fpv:palmahim_10.02 fpv:palmahim_27.05 --recipes simdist --models comb lgbm --speed 0.8 1.25 --seed 0
```

This also records the comb's frozen threshold for each held-out unit.

**Result.** 64 tests pass; the new one saves a small LightGBM model, loads it
through `dad wav`'s loader, checks the scores are identical, and finds its
threshold in a run file. No real model was trained: the data is on the
training PC.

**Assessment.** Iteration 5 trained only the CNN (`--models dscnn`), so no
LightGBM model with that recipe exists yet; the command above has to run
first. Loading a pickle runs code from the file, so only models trained here
should be loaded; the same already holds for the `.pt` checkpoints.

**Next.** Run the command above on the training PC, then choose the
checkpoints for the real test.

# Iteration 6 - The final model for the real test

> **Changed from iteration 5:** nothing is held out. Iteration 5's best
> recipe (pool + speed perturbation 0.8-1.25, `simdist`) is trained once on
> every unit (Nightjar, d3set, the three FPV campaigns, Glasgow, DDL and
> DroneNoise near drones, FSD50K), with LightGBM and the comb next to it,
> and the threshold is frozen at 5% false alarms on validation (R3) instead
> of 10%.
>
> **Why:** Mads asked for a fresh model 6 and chose (this conversation) a
> final model for the real test over a further experiment. Iteration 5
> showed the recipe; a deployed model should see all the data.
>
> **Kept the same:** small CNN, 1 s window, 200,000 training windows, the
> recipe, speed range, seed 0, validation sessions (Nightjar flight 3, d3set
> Validation), the sealed sets (unscored).
>
> **Hypothesis:** none to test. This iteration produces checkpoints, not
> evidence: with nothing held out there is no unseen rig with no-drone
> audio left to judge it on.
>
> **Judged by:** not judged against R2-R4 here. The honest unseen-rig figures
> remain iteration 5's (entry 037). The model is judged by the real test.

## 043 - The final model trained on every rig

- Date: 2026-10-22
- Iteration: 6
- Type: Model
- Commit: uncommitted (on top of dd09c77)

**What changed.**

- `make_pool_arm` accepts `held="none"`: train on every unit, empty test
  list; an unknown unit name is now refused instead of silently training on
  everything.
- `scripts/run_range.py --arms none`: with no test list, the run scores
  validation only and stores it under `val` in the run file, marked
  `VALIDATION` in the log. The lawn-mower figures use validation's 10%
  threshold in that case. The first launch crashed on that lookup; fixed and
  rerun.
- Checkpoints: `artifacts/models/it6_final_none_simdist_dscnn.pt` and
  `it6_final_none_simdist_lgbm.pkl`; run file `artifacts/runs/it6_final.json`;
  log `artifacts/it6_final.log`. `dad wav --threshold-from` finds both
  thresholds (checked: CNN 0.2968, LightGBM 0.6115).

**Why.** Entry 037's "final model trained on the whole pool", and Mads's
choice of it as model 6.

**How.**

```powershell
.venv-dml\Scripts\python.exe -u scripts\run_range.py --name it6_final --pool --arms none --recipes simdist --models comb lgbm dscnn --device dml --windows 200000 --speed 0.8 1.25 --deploy-fa 0.05 --seed 0
```

459 s in all on the AMD GPU through DirectML (CNN about 600 windows/s).
64 tests pass.

**Result.** 1 s window, 0.5 s evaluation hop, threshold frozen at 5% false
alarms on the validation sessions' own no-drone audio (4 sessions, 2,648
drone and 1,565 no-drone windows; 92% of the drone windows within 50 m).
Validation also chose the threshold, so its recall is in-distribution and
optimistic. Lund is a rig never trained on, drone audio only (12 flights,
7,871 windows), so only recall can be measured there. The lawn-mower
session's mower-only audio (125 windows) is a confuser never trained on.

| Scorer | Threshold | Validation recall, all | 0-50 m | 50-100 m | 100-200 m (97 windows, 1 session) | Lund recall, all | Mower-only false alarms |
|---|---|---|---|---|---|---|---|
| comb | 1.914 | 0.305 | 0.323 | 0.087 | 0.124 | 0.022 | 0.248 |
| lgbm | 0.612 | 0.735 | 0.768 | 0.635 | 0.000 | 0.035 | 0.344 |
| dscnn | 0.297 | **0.825** | 0.856 | 0.852 | 0.021 | **0.502** | 0.552 |

**Assessment.**

- Against R2 (recall 95%) and R3 (5% false positives) at one threshold: not
  met even in-distribution. At exactly 5% false alarms on validation the CNN
  finds 83% of drone windows, nearly all of them within 100 m.
- On an unseen rig (Lund) the CNN finds half the drone windows at that
  threshold; both baselines fall to near zero there. The CNN is the only
  candidate worth taking to the test.
- The mower: the CNN alarms on 55% of mower-only windows. A loud motor
  without a drone is the weakest point found so far, and the real test may
  hold similar noise (generators, vehicles).
- Nothing here is comparable to iteration 5's figures, which were measured on
  held-out rigs at 10% false alarms.
- One seed. Not checked: digital silence (entry 041), and the sealed sets.

**Next.** Report to Mads. Run the three models over the real-test WAV files
with `dad wav` once they exist.

## 044 - Ten drone-only WAVs from the Pure Data patch
- Date: 2026-10-26
- Iteration: 6
- Type: Data
- Commit: uncommitted (on top of e924a30)

**What changed.** New script
`Pure-data+Script/EnviromentTest/drone_only.py`. It renders 10 flights of
the Pure Data drone with no background, wind or events, one per seed (1-10).
Each seed draws its own start point (distance 300-500 m, elevation 20-60 deg,
azimuth 0-360 deg) and speed (28-45 m/s) through the existing `make_scene`.
The flight is a straight line to the microphone with delay, Doppler and 1/r.
Two fixes came with it, both in `render.py`:
- Pd is now also found at `C:/Program Files*/Pd*/bin/pd.com` (Windows).
  Before this, the script only looked under macOS `/Applications`.
- Pd quits right after `wr stop`, and sometimes does so before `writesf~`
  has written the WAV header. The header then says 0 samples, although the
  samples are in the file. This always happened for seed 6 and gave a NaN
  scene. `render_dry` now reads the raw 16-bit data after the 44-byte header
  when the file reports 0 samples.

**Why.** Mads asked for drone sound only, at different start points and
speeds.

**How.** `cd Pure-data+Script/EnviromentTest; uv run python drone_only.py`
(Pd 0.55.2). It writes these files to `Pure-data+Script/EnviromentTest/drone_only/`:
- `drone_NN.wav`: calibrated, full scale = 110 dB SPL, 32-bit float.
- `drone_NN_listen.wav`: peak-normalised.
- `drone_NN.json`: every drawn parameter.

The WAVs are 32-bit float because the start of the calibrated files sits
near -84 dBFS, which is too close to the 16-bit floor.

**Result.**

| Seed | Speed (m/s) | Start (m) | Length (s) |
|---|---|---|---|
| 1 | 36.7 | 490 | 13.4 |
| 2 | 32.4 | 360 | 11.1 |
| 3 | 29.5 | 347 | 11.8 |
| 4 | 44.0 | 402 | 9.1 |
| 5 | 41.7 | 462 | 11.1 |
| 6 | 37.1 | 369 | 9.9 |
| 7 | 38.6 | 479 | 12.4 |
| 8 | 33.6 | 497 | 14.8 |
| 9 | 42.8 | 357 | 8.4 |
| 10 | 44.3 | 342 | 7.7 |

All 10 files were checked: none holds NaN, and each gets louder towards
impact (RMS of the last second is about 100x the RMS of the first half).
This is synthetic data and is not detection performance (section 5).

**Assessment.** Scaffolding only; no requirement is affected.

**Next.** Wait for Mads.

## 045 - Drone-only WAVs: rising speed instead of random draws
- Date: 2026-10-26
- Iteration: 6
- Type: Data
- Commit: uncommitted (on top of e924a30)

**What changed.** Mads asked for the 10 files from entry 044 to step through
rising speed instead of using random draws.
- `make_scene` in `drone_sim.py` now takes `**fixed`, which overrides any
  drawn parameter. The override is applied after the draws, so the random
  sequence, and with it every existing scene, is unchanged.
- `drone_only.py` now uses one seed (1) for all 10 files, so the drone's
  pitch and throttle are the same everywhere. Only speed and start point
  differ:
  - Speed rises in even steps from 28 to 45 m/s.
  - Start distance is fixed at 400 m and elevation at 40 deg.
  - Azimuth steps 0, 36, ..., 324 deg.

The files from entry 044 were overwritten under the same names.

**How.** `cd Pure-data+Script/EnviromentTest; uv run python drone_only.py`.

**Result.**

| File | Speed (m/s) | Azimuth (deg) | Length (s) |
|---|---|---|---|
| 01 | 28.0 | 0 | 14.3 |
| 02 | 29.9 | 36 | 13.4 |
| 03 | 31.8 | 72 | 12.6 |
| 04 | 33.7 | 108 | 11.9 |
| 05 | 35.6 | 144 | 11.2 |
| 06 | 37.4 | 180 | 10.7 |
| 07 | 39.3 | 216 | 10.2 |
| 08 | 41.2 | 252 | 9.7 |
| 09 | 43.1 | 288 | 9.3 |
| 10 | 45.0 | 324 | 8.9 |

All 10 files were checked: they are 32-bit float, none holds NaN, and each
gets louder towards impact. Synthetic data, not detection performance.

**Assessment.** Scaffolding only; no requirement is affected.

**Next.** Wait for Mads.

## 046 - Generated WAV files are git-ignored
- Date: 2026-10-26
- Iteration: 6
- Type: Infrastructure
- Commit: uncommitted (on top of e924a30)

**What changed.** Mads asked for all generated WAV files to be ignored by git.
- `.gitignore` now has `*.wav`.
- The 16 WAV files that were already committed under
  `Pure-data+Script/EnviromentTest/` (scenes, live runs, `out.wav`, dry and
  approach tests) were removed from the index with `git rm --cached`. They
  remain on disk.

The scripts in that folder re-create them. The JSON sidecars in
`drone_only/` are still tracked candidates.

**Result.** No numbers. **Next.** Wait for Mads.

## 047 - Drone-only WAVs are all 10 s long
- Date: 2026-10-26
- Iteration: 6
- Type: Data
- Commit: uncommitted (on top of e924a30)

**What changed.** Mads asked for every drone file to be 10 s long.
`drone_only.py` now sets the start distance to speed x 10 s instead of a
fixed 400 m (entry 045). Every flight therefore reaches the microphone after
exactly 10 s, and no file is cut or padded. Speed (28-45 m/s, even steps),
azimuth (36 deg steps), elevation (40 deg) and the drone sound (seed 1) are
unchanged. The start distance now rises with the speed: 280, 299, 318, 337,
356, 374, 393, 412, 431, 450 m.

**How.** `cd Pure-data+Script/EnviromentTest; uv run python drone_only.py`.

**Result.** All 20 WAVs (10 calibrated, 10 listen) are 10.0000 s at 44.1 kHz,
none holds NaN. Synthetic data, not detection performance.

**Next.** Wait for Mads.

## 048 - Drone-only WAVs: a drone count and a 100-200 km/h speed range
- Date: 2026-10-26
- Iteration: 6
- Type: Data
- Commit: uncommitted (on top of e924a30)

**What changed.** At Mads's request, `drone_only.py` now has an integer
`N_DRONES` that sets how many files are generated (default 10). The speeds
always run from 27.778 m/s (100 km/h) to 55.556 m/s (200 km/h) in
`N_DRONES` equal steps (`np.linspace`). Before this the range was 28-45 m/s
(entry 045). The start azimuths are spread evenly at 360/`N_DRONES` deg. The
other settings are unchanged:
- Start distance = speed x 10 s, so every file lasts 10 s (entry 047).
- Elevation 40 deg.
- Seed 1.

**How.** `cd Pure-data+Script/EnviromentTest; uv run python drone_only.py`,
with `N_DRONES = 10`.

**Result.** Speeds were 27.8, 30.9, 34.0, 37.0, 40.1, 43.2, 46.3, 49.4, 52.5
and 55.6 m/s (step 3.09 m/s). Start distances ran from 278 m to 556 m. All 20
WAVs are 10 s long and none holds NaN. Synthetic data, not detection
performance.

Not handled: files from an earlier run with a larger `N_DRONES` stay in
`drone_only/` until they are deleted by hand.

**Next.** Wait for Mads.

## 049 - drone_only JSON sidecars are git-ignored for now
- Date: 2026-10-26
- Iteration: 6
- Type: Infrastructure
- Commit: uncommitted (on top of e924a30)

**What changed.** At Mads's request, `.gitignore` now has
`Pure-data+Script/EnviromentTest/drone_only/*.json`, so the parameter
sidecars of the drone-only files are not tracked for now. They were never
committed. With entry 046 the whole `drone_only/` folder is now ignored.

**Next.** Wait for Mads.

## 050 - Council: a deterministic synthetic soundscape test of model 6
- Date: 2026-10-26
- Iteration: 6
- Type: Council
- Commit: uncommitted (on top of e924a30)

**Question.** Should we build and run Mads's soundscape test? Each 20 s
noise clip (from a colleague's generator, about 10,000 clips) is scored:
- once alone,
- once with each Pure Data drone from entry 048 added, at a deterministic
  onset that rotates per clip.

That is about 110k runs. Each run writes its WAVs and a sidecar, scores
them, appends one result line and deletes the WAVs. The alternatives were a
modified version, or no synthetic test (wait for the real-test WAVs).

**Devil's Advocate.**
- *Decisive: the absolute level mapping is arbitrary.* The frontend is raw
  log-mel (`src/dad/features.py` `frames`, checked) and the model has no
  input normalisation (first layer conv + BatchNorm, `models/dscnn.py`,
  checked). Moving the full-scale mapping moves the scores.
  *Answer:* measured in the gate (entry 051). A ±20 dB gain on the mix
  leaves comb and lgbm unchanged. dscnn false alarms on noise go from 45% to
  37% at −20 dB and do not change at +20 dB. So level matters moderately.
  The objection partly stands, and it applies equally to the real test
  hardware.
- *Decisive: one drone timbre.* All runs use seed 1, so the effective n is
  10 drones. *Left standing.* More seeds or patch settings are cheap
  (deterministic), but it is Mads's choice.
- *Decisive: the result cannot be reported under §5.* Accepted. The test is
  a diagnostic tool, not a performance figure. The gate already found a
  failure mode (entry 051). The real-test WAVs remain the only reportable
  numbers.
- *Serious: R5 is flattered.* Confirmed in the gate. `lead_times` credits
  the first alarm after the drone starts, so false alarms in the noise count
  as early detections. R5 must be read together with the false alarms on
  the same noise clip alone.
- *Serious: re-thresholding from the stored traces is tuning on test.* The
  rule is that thresholds stay frozen at it6 validation. The traces are for
  diagnosis only. This is written into the script docstring.
- *Serious: the placeholder noise is not the real scenario.* True. It is a
  placeholder until the colleague's generator arrives.
- *Weak: speed is tied to start range.* This follows from Mads's
  10-s-per-file rule, so range is recorded per hop.

**Scholar.**
- Best practice (Scaper / DCASE Task 4) uses synthetic soundscapes for
  diagnosis with controlled factors. Claims about the target are made on
  real recordings.
- A deterministic factorial grid with rotation is a "defensible variation".
- Store per-window scores.
- Confidence intervals must resample by background clip and target variant,
  not by mixture.
- Record commit, checkpoint hash, frontend, every mixture parameter, label
  rule and threshold source.
- Our design follows this. The CI rule is adopted for any later summary.

**Problem Solver.** *Try a modified version.*
- Add baselines (comb, lgbm) to every run.
- Record range per hop.
- Run a clean-drone sanity check and a ~130-run gate before scaling.
- Use a factorial over noise types rather than 10k clips.
- Keep the real-test path going in parallel.

**Decision.** Build the harness with the Problem Solver's changes:
- three scorers at frozen thresholds,
- per-hop score, label and range,
- checkpoint hashes and commit,
- resumable by test id,
- WAVs deleted after each test.

Run the gate on placeholder noise and do not scale to 110k runs. Before
that, Mads must decide on drone variety, and the colleague's noise must
exist with the same level convention.

## 051 - Soundscape test harness and its gate run
- Date: 2026-10-26
- Iteration: 6
- Type: Evaluation
- Commit: uncommitted (on top of e924a30)

**What changed.** New `Pure-data+Script/EnviromentTest/soundscape_test.py`,
following entry 050. Per noise clip k it runs:
- the noise alone,
- noise + drone i at onset ((i + k) mod 10) s.

Each run writes a noise WAV, then a mix WAV with sidecar
`{drones: [{start_s = onset + start_dist/343, end_s = impact_s = onset + 10 s}]}`.
It scores the mix with `dad.wavio.evaluate_file` for dscnn, lgbm and comb at
the frozen it6 thresholds (0.2968, 0.6115, 1.9137) and appends one line to
`artifacts/soundscape/results.jsonl`, which is git-ignored. The line holds:
- every parameter, commit, dirty flag and checkpoint hashes,
- per-hop time, label, drone range, and each scorer's score, counts and
  leads.

The WAVs are then deleted, and tests already in the file are skipped. Noise
is `--noise placeholder` (scene.py's 40 dB white background + 45 dB wind +
bangs and passing vehicles, seed 10000+k) or `--noise silence`. `--gain-db`
scales the whole mix. Asserts check the range formula.

**How.** In `Pure-data+Script/EnviromentTest`:
- `uv run python soundscape_test.py --clips 1 --noise silence`
- `uv run python soundscape_test.py --clips 10 --gain-db G` for G in 0, −20, +20

The drones are entry 048's 10 files. Results: 341 lines, 2.4 MB. Run time
is about 1.3 s per test with all three scorers on CPU, so 110k tests take
about 40 h in one process.

**Result.** Synthetic scaffolding; not detection performance. Window
counts are pooled over 10 clips (110 tests) per row; the silence row is one
clip (11 tests). Window 1 s, hop 0.125 s, 16 kHz after resampling.

| Noise, gain | Scorer | False alarms, noise alone | Recall in mixes | R5 met |
|---|---|---|---|---|
| silence, 0 dB | dscnn | 0.0% | 94.7% | 10/10 |
| silence, 0 dB | lgbm | 0.0% | 1.9% | 2/10 |
| silence, 0 dB | comb | 0.0% | 77.5% | 10/10 |
| placeholder, 0 dB | dscnn | 45.2% | 74.0% | 100% |
| placeholder, 0 dB | lgbm | 61.8% | 80.0% | 100% |
| placeholder, 0 dB | comb | 15.3% | 32.3% | 75% |
| placeholder, −20 dB | dscnn | 36.6% | 62.9% | 99% |
| placeholder, +20 dB | dscnn | 45.4% | 74.1% | 100% |

lgbm and comb change by less than 3 points across gain.

- **dscnn false alarms on the noise alone, at 0 dB**, by what is in the
  window:
  - windows with a passing vehicle (harmonic engine, f0 70–130 Hz): 86.4%
    of 575;
  - bangs: 19.0% of 205;
  - background and wind only: 20.7% of 750.
- **dscnn recall in the mixes by drone range**, at 0 dB:

  | Range | Recall | Windows |
  |---|---|---|
  | 0–100 m | 0.98 | 1830 |
  | 100–200 m | 0.77 | 1780 |
  | 200–300 m | 0.58 | 1570 |
  | 300–400 m | 0.53 | 840 |
  | 400–500 m | 0.44 | 260 |

  Above 200 m these figures are close to the false-alarm rate, so they are
  mostly false alarms, not detections.

**Assessment.**
- The harness works end to end, and model 6 does respond to the Pd drone:
  in silence it alarms 7.5–8.3 s before impact on all 10.
- The test cannot judge R2–R5 while the noise itself triggers 45% of
  windows. R5 "100% met" is then an artefact of false alarms (entry 050).
- The vehicle tone is the same weakness as the mower in entry 043: a
  steady harmonic motor is taken for a drone.
- lgbm barely fires on the clean synth drone (1.9%). Its 80% recall in
  the mixes therefore comes from the noise, not the drone.

**Next.** Report to Mads. Do not scale up yet. Open points:
1. the colleague's noise generator and its level convention,
2. more drone timbres,
3. whether the test should keep "first alarm after drone start" for R5 or
   also require no alarm in the noise before it.

## 052 - Five drone sounds in the soundscape test
- Date: 2026-10-26
- Iteration: 6
- Type: Data
- Commit: uncommitted (on top of e924a30)

**What changed.** Mads asked for more drone sounds with fixed seeds. This
answers the council's "one timbre" objection (entry 050).

`drone_only.py` now renders `N_DRONES` = 10 flights for each of
`SEEDS = [1, 2, 3, 4, 5]`, as files `drone_s<seed>_<i>.*`, 50 in all. The
seed sets the throttle curve and the level `L_1m`. The pitch scale is fixed
per seed, evenly from 0.85 to 1.25 (`PSCALES`), because the drawn values
bunched together: seeds 1, 3, 4 and 5 drew 1.00–1.02. Everything else is as
in entry 048: speeds 27.8–55.6 m/s, 10 s each, start distance = speed x 10 s.

`soundscape_test.py` rotates the drone sound with the noise clip:
clip k uses seed number (k mod 5). There are still 11 tests per clip, and
the test id now holds the seed. The old single-seed drone files were
deleted. The gate results of entry 051 were kept as
`artifacts/soundscape/results_gate051.jsonl`.

**How.** In `Pure-data+Script/EnviromentTest`:
- `uv run python drone_only.py` (50 files, all 10.0 s, no NaN, checked)
- `uv run python soundscape_test.py --clips 5 --noise silence`
- `uv run python soundscape_test.py --clips 10`

**Result.** Synthetic scaffolding; not detection performance. Thresholds
are frozen at it6 values. Each row is recall in mixes, then R5 met
(approaches). In silence there is 1 clip per seed (10 approaches each); with
placeholder noise there are 2 clips per seed (20 approaches each).

| Drone sound | dscnn, silence | dscnn, placeholder | comb, silence | lgbm, silence |
|---|---|---|---|---|
| seed 1, pitch 0.85 | 26.4%, 4/10 | 63.3%, 100% | 89.2%, 10/10 | 0.3%, 0/10 |
| seed 2, pitch 0.95 | 63.2%, 7/10 | 77.1%, 100% | 86.1%, 10/10 | 1.0%, 0/10 |
| seed 3, pitch 1.05 | 98.9%, 10/10 | 81.7%, 100% | 78.3%, 10/10 | 3.5%, 5/10 |
| seed 4, pitch 1.15 | 95.2%, 10/10 | 75.6%, 100% | 5.9%, 1/10 | 11.5%, 9/10 |
| seed 5, pitch 1.25 | 79.1%, 10/10 | 73.9%, 100% | 1.4%, 2/10 | 10.4%, 10/10 |
| all | 72.6%, 82% | 74.3%, 100% | 52.2%, 66% | 5.3%, 48% |

False alarms on the placeholder noise alone are unchanged from entry 051:
dscnn 45.2%, lgbm 61.8%, comb 15.3%. It is the same noise, seeds 10000–10009.

**Assessment.**
- The one-timbre objection was real. In silence, model 6 finds the
  higher-pitched drones (pitch 1.05–1.25) almost every time. The lowest
  pitch (0.85) is found in 26% of windows, with an alarm 3 s before impact
  on only 4 of 10 approaches.
- With the old single drone sound (pitch 1.02, entry 051) this weakness was
  invisible.
- The comb is the mirror image: it finds low-pitched drones and misses
  high-pitched ones. That suggests its harmonic search range is limited.
- In the placeholder noise these differences are masked by false alarms.
  There, R5 "100%" still says nothing (entry 051).

**Next.** Report to Mads. The soundscape test now covers 5 drone sounds.
Scaling up still waits for the colleague's noise.

## 053 - Council: a pretrained audio embedding as frontend
- Date: 2026-10-26
- Iteration: 6 (prepares the next iteration)
- Type: Council
- Commit: uncommitted (on top of bf8e1ac)

**Question.** Should the next iteration add a frozen pretrained audio
embedding (PANNs, YAMNet, BEATs or similar) with a small classifier, next to
or instead of the log-mel + small CNN trained from scratch? The alternatives:
- A. keep the current approach and spend the effort on hard negatives or data;
- B. frozen embedding + linear probe as an extra candidate;
- C. fine-tune a pretrained network;
- D. finer, time-stamped labels instead.

Mads asked for the council after asking whether we had "transcribed" the
sound for the CNN (labels by time, or pretrained audio models). No code was
written. This idea was first raised in entry 016 and listed as Mads's
decision in entry 020.

**Devil's Advocate** (full brief). Against B and C.
- *Decisive:* the embeddings aim at the wrong failure. The requirement that
  fails is false alarms on engines: 55% of mower-only windows (entry 043).
  AudioSet training puts propellers, light engines, lawn mowers and vehicles
  close together, and a linear probe cannot separate what the frozen
  encoder has merged.
- *Serious:*
  - Embeddings encode the recording chain strongly, and with about 6 rigs
    the protocol is too weak to tell a 0.05-0.10 gain from seed and fold
    spread.
  - Opportunity cost just before the field test.
  - C overfits on tens of minutes of drone audio (entry 019: 30x capacity
    did not help).
- *Weak:* 32/48 kHz encoders on 16 kHz audio; Pi compute; FSD50K possibly in
  some encoders' pretraining (not verified).
- *What would make B right:* a cheap check that the embedding separates
  drone from mower within one rig better than dscnn, a gain larger than the
  seed spread on two held-out rigs at a fixed false-alarm rate, a 16 kHz
  native encoder, and about a day of work.

**Scholar** (context-free brief). Best practice for small labelled data
that must transfer to unseen setups is to start from an AudioSet-scale
pretrained network:
- First a frozen encoder with a linear/MLP head (the HEAR protocol), then
  fine-tuning a compact pretrained CNN with augmentation, preferably
  linear-probe-then-fine-tune (LP-FT).
- The closest drone study (Reuter, Ohlenbusch, Rollwage, arXiv 2605.31329,
  2026) found pretraining "the dominant factor" for generalising to unseen
  recording setups. The Scholar read the abstract only.
- Pitfalls named:
  - Sample rate: PANNs' default is 32 kHz, but an official `Cnn14_16k`
    exists. YAMNet's 0.96 s patches set a floor on the window.
  - AudioSet's "Drone" class is the musical drone, not a UAV.
  - Several AudioSet classes are our confusers.
  - Embeddings encode site and microphone: run a "predict the rig from the
    embedding" sanity probe.
  - Full fine-tuning can lose out-of-distribution accuracy against a linear
    probe (Kumar et al., ICLR 2022, shown in vision).
- Verdicts:
  - A: defensible as a control, below current practice as the main bet.
  - B: the standard first step.
  - C: standard and best supported for the end system.
  - D: a complement, against practice as a substitute.
- Unsettled: which encoder keeps the fine harmonic structure that separates
  a drone from a mower.

**Problem Solver** (full brief, read the code). Verdict: **try a modified
B**.
- Encoder: PANNs `Cnn14_16k`, frozen. It is native 16 kHz like the store,
  MIT licence, plain PyTorch, and one embedding per 1 s window.
- Classifier: logistic regression on cached embeddings of the same sampled
  windows dscnn trains on (`RangeSampler.batch`, pool + simdist + speed
  0.8-1.25).
- New code: about one module plus a `--models` entry.
- Not C before B; D later.
- Pre-declared rule, with comb, lgbm and dscnn alongside:
  - *Success:* mower-only false alarms ≤ 25% at the threshold frozen at 5%
    on validation (dscnn 55%), and Lund recall ≥ 0.45 (dscnn 0.50), and mean
    held-out recall at 10% false alarms within 0.05 of dscnn on the
    iteration-5 arms, no rig worse by more than 0.10, three seeds.
  - *Failure:* mower false alarms above 40%, or Lund recall below 0.35.
  - *In between:* try a probe on the embedding plus the dscnn score.
- If B fails: real hard negatives (licence-filtered FSD50K engine and
  vehicle clips) mixed in as negatives.
- Confidence: medium that it cuts mower false alarms, low that it reaches
  R2/R3.

**Weighing.**
- *Decisive objection (engines clustered).* It cannot be settled by
  argument. The Scholar raises the same doubt as unsettled, and the Problem
  Solver argues the opposite: AudioSet has seen thousands of hours of
  mowers. The experiment is built around it:
  - The mower false-alarm figure is the first gate.
  - Before any full run, a within-rig check: a probe on cached embeddings,
    drone against mower-only windows of the lawn-mower session, compared
    with the same probe on dscnn's penultimate features.
  - The objection stands until that check is run.
- *Weak protocol.* Partly answered. The success margin on the mower figure
  (55% → 25%) is far larger than the seed spread seen so far, and three
  seeds are required. It stays a limit: the mower and Lund figures rest on
  one session each, and that is written next to every result.
- *Chain encoding.* Answered by adopting the Scholar's sanity probe: report
  how well the rig can be predicted from the embedding, against the same
  probe on dscnn features.
- *Opportunity cost.* Answered by design. The cache-based arm is about one
  module, and its rule is fixed now, so it ends in a verdict, not a
  tour of encoders. Only one encoder is tried, and Lund is never used to
  choose between encoders. The timing against the field test is Mads's
  call.
- *C overfits.* Accepted for now.
- *Sample rate.* Answered by `Cnn14_16k`. Its 8 kHz top matches the 16 kHz
  store and the 50-8000 Hz band.
- *FSD50K in pretraining.* PANNs are trained on AudioSet (YouTube), not
  Freesound. The figures that decide are on Lund and the mower session,
  which are not from either source.

**Departures from the Scholar's practice.**
- Fine-tuning (C), which the Scholar calls the best-supported end system,
  is deferred, not rejected. The reason is our situation: about 6 rigs,
  tens of minutes of distinct drone audio, and entry 019, where 30x
  capacity did not help. If B shows the encoder helps, LP-FT on
  `Cnn14_16k` is the next step.
- D (strong labels) is kept as a complement. The Scholar is right that the
  long FPV blocks and per-file DDL labels are label noise. It is not the
  lever on the mower false alarms, which is the failure that blocks the
  field test.

**Decision.** Modified B, as the Problem Solver specified, with two checks
added:
1. the within-rig drone-against-mower probe first;
2. the rig-ID sanity probe.

The success and failure rule above is fixed as of this entry. It is the
start of iteration 7 once Mads approves:
- a new dependency (PANNs model code, or loading its checkpoint directly in
  torch);
- a checkpoint download of about 300 MB (`Cnn14_16k`, size not verified).

Nothing is downloaded and no code is written before that.

**Next.** Ask Mads:
- for the dependency and download;
- whether this should run before or after the field test.

## 054 - Decision: embeddings wait until after the field test
- Date: 2026-10-26
- Iteration: 6
- Type: Requirements
- Commit: uncommitted (on top of bf8e1ac)

**What changed.** Mads decided that the pretrained-embedding experiment of
entry 053 waits until after the field test. Nothing was downloaded and no
dependency or code was added.

**Why.** Mads's call on timing (entry 053, "Next"). Model 6 (entry 043)
stays the candidate for the field test.

**Next.** After the field test, start iteration 7 as fixed in entry 053:
1. the within-rig drone-against-mower probe;
2. the rig-ID sanity probe;
3. the full rule.

The rule is unchanged, but the field-test results may add a confuser
figure. If they do, that is logged before iteration 7 starts.

# Iteration 7 - An Edge Impulse candidate (side experiment)

> **Changed from iteration 6:** a second training path. A model is built in
> Edge Impulse Studio (MFE frontend + small Keras CNN, trained in the
> vendor's cloud) on the data whose licences allow upload to a third party:
> Nightjar (CC BY) for training, Halmstad (CC0), DroneNoise (CC BY, one
> microphone per event) and two held-out Nightjar sessions for testing.
>
> **Why:** Mads asked (this conversation, 2026-10-26) to try building it in
> Edge Impulse with his school access. Deployment hardware is still OPEN,
> and Edge Impulse profiles and exports models for a Pi or a microcontroller.
>
> **Kept the same:** model 6 and the baselines, unchanged; the 1 s window and
> 0.125 s hop of `dad info`; R2-R4 measured by the repo's own scorer
> (`dad wav`, 1 s guard at drone edges) at one threshold frozen on a
> validation session. Edge Impulse's own model-testing numbers are not
> reported as results.
>
> **Hypothesis:** none that the evidence favours (entry 055): the Edge
> Impulse model is the same family as model 6 trained on a subset of its
> data, so it is not expected to beat it on detection. The experiment asks
> whether the vendor's pipeline is competitive on the same windows, and what
> it costs on embedded targets.
>
> **Judged by:** recall, false-positive rate and precision at one
> validation-frozen threshold on the same test clips, Edge Impulse next to
> model 6, `lgbm` and `comb`; Halmstad's false-positive rate reported first.
> Keep it as a candidate only if it is clearly above model 6 there, or
> matches it and fits a target model 6 cannot. This iteration is
> time-boxed and must not delay the field test. The embedding plan of entry
> 054, called "iteration 7" there, becomes iteration 8.

## 055 - Council: an Edge Impulse model next to model 6
- Date: 2026-10-26
- Iteration: 7
- Type: Council
- Commit: uncommitted (on top of ddfc33d)

**Question.** Build a model in Edge Impulse Studio, and if so (a) not at all,
(b) as an extra candidate next to model 6, or (c) move development there?

**Devil's Advocate.** Decisive: the Edge Impulse model trains on a strict
subset of model 6's data (one drone type, one rig) on exactly the axis that
fails, and that drone type is already at chance off-rig (entry 037); it
cannot beat model 6. Serious: (1) the test sets are asymmetric (DroneNoise is
drone-only, Halmstad is off-rig), so a model that learned "Nightjar room tone
= no drone" shows high recall and hides its cost in Halmstad false alarms;
(2) Studio reports per-window accuracy at its own threshold, not R2/R3;
(3) looking at Studio's test page while tuning is tuning on test; (4) it costs
field-test preparation time. Verdict: reject (c), postpone (b).
*Answered:* (1) Halmstad's false-positive rate is reported first, DroneNoise
gives recall only; (2) and (3) the model is exported and scored locally on
the same clips as model 6 with `dad wav`, threshold frozen on a Nightjar
session that is never uploaded, and Studio's test page is not used for
choices; (4) left standing: Mads asked for it now, so it is time-boxed and
kept off the field-test path. The decisive objection is accepted as the
expected outcome, not refuted.

**Scholar.** Standard: log-mel + small CNN, split by recording, threshold
fixed on validation, recall at a low false-positive rate on unseen setups
(DCASE 2022-2026 Task 2; Mesaros et al. 2016). A hosted platform used as a
model builder with grouping by recording and metrics computed outside it is
a defensible variation; its default random split with per-window accuracy
reported as the result is against practice. Edge Impulse staff confirm its
train/validation split is random across windows and can leak with overlap.
*Departure:* none, if the conditions hold. Edge Impulse's internal
validation split (early stopping only) is random by window inside the
training sessions; this inflates only its own validation accuracy, which is
not reported.

**Problem Solver.** Try a modified version of (b), at most about a day:
split fixed before upload, a local validation session, score the exported
model (WebAssembly) on the same windows as model 6, `lgbm` and `comb`, and
collect Edge Impulse's latency/RAM/flash estimates for the open hardware
question. Key control: model 6's recipe retrained on the same restricted
data, so a gap is not just "less data". Success: at least 0.10 recall above
the restricted model 6 at 5% false positives; partial: Edge Impulse becomes
the deployment toolchain only; failure: logged as a negative result.
Confidence medium.

**Decision.** (b), modified as the Problem Solver proposes; (c) rejected.
The restricted-data model 6 control is planned but not part of the first
pass; if it is not run, the comparison is reported as confounded by data.
Model 6 itself trained on DroneNoise near drones and on every Nightjar
session (entry 043), so for model 6 only Halmstad is unseen; the comparison
rests on Halmstad.

**Next.** Export the clips (`scripts/export_edge_impulse.py`), upload,
train, export, score locally.

## 056 - Edge Impulse: data, impulse and first training; the repo's models on the same clips
- Date: 2026-10-26
- Iteration: 7
- Type: Model
- Commit: uncommitted (on top of ddfc33d)

**What changed.**

- Raw Nightjar, DroneNoise and Halmstad were linked into `data/` from the
  Google Drive copy with directory junctions (nothing copied or changed) and
  prepared with the existing scripts into `data/processed/`.
- New `scripts/export_edge_impulse.py`: 10 s clips per drone (label 1) and
  no-drone (label 0) stretch, at most 900 s per session and label, guard bands
  skipped, label in the file name, a `dad wav` sidecar next to each clip.
  Training: Nightjar sessions 0804, 0804-2, 0804-4, 0804-night, 0806-2 and
  both night-ambient sessions. Validation (kept local, never uploaded):
  Nightjar 0806-1. Testing: Nightjar 0804-3 and the lawn-mower session,
  Halmstad (30 drone, 59 helicopter/background), DroneNoise microphone M5
  only. Clips are also joined into files under 10 MB (`packed/`), because the
  browser bridge takes at most 10 MB per call.
- New `scripts/score_hops.py`: scores per-hop CSVs per source at one
  threshold, given or frozen at 5% false positives on a validation CSV.
- Edge Impulse project `drone-acoustic-detection` (ID 1149245, AAU-CREATE
  organisation, private). 29 training samples, 1 h 42 min, uploaded through
  the browser. Two of them are 10 s clips that are also inside a packed file
  (20 s duplicated of 102 min).
- Impulse: 1000 ms window, 250 ms training stride (24,433 windows), 16 kHz.
  MFE: frame 0.025 s, stride 0.01 s, 64 filters, FFT 1024, 50 Hz to 8 kHz,
  noise floor -100 dB (the default -52 dB set every feature of a Nightjar
  sample to zero). Classifier: Edge Impulse's default 1D CNN (8 and 16
  filters, kernel 3, dropout 0.25), 100 epochs, learning rate 0.005, GPU,
  Edge Impulse's audio augmentation on (noise high, time masks low).

**How.**

```powershell
cmd /c mklink /J data\Nightjar_HF "G:\My Drive\drone-acoustic-detection-data\Nightjar_HF"   # and DroneNoise_Salford, Halmstad
uv run python -I scripts/prepare_data.py dronenoise halmstad
uv run python -I scripts/prepare_new_data.py nightjar
uv run python -I scripts/export_edge_impulse.py
uv run dad wav data/edge_impulse/testing --detector artifacts/models/it6_final_none_simdist_dscnn.pt --threshold-from artifacts/runs/it6_final.json --out artifacts/edge_impulse/hops_dscnn_testing.csv
uv run python -I scripts/score_hops.py artifacts/edge_impulse/hops_dscnn_testing.csv --val artifacts/edge_impulse/hops_dscnn_validation.csv
```

(`comb` and `lgbm` the same way; logs and CSVs in `artifacts/edge_impulse/`.)

**Result.** Edge Impulse's own figures, on its internal validation split,
which is random by window inside the training sessions and so leaks and
flatters (entry 055); not a detection result:

| Run | Architecture | Model | Accuracy | AUC |
|---|---|---|---|---|
| 1 | 1D CNN (default) | int8 | 0.692 | 0.69 |
| 2 | 2D CNN preset (dropout 0.5) | int8 | 0.533 (every window called drone) | 0.50 |
| 3 | 1D CNN again | int8 | 0.687 | 0.69 |

Run 1 on Cortex-M4F 80 MHz, EON compiler: inference 12 ms, 20.3 KB RAM,
41.1 KB flash. The MFE block was estimated at 150 ms with FFT 512 and 592 ms
with FFT 1024 per 1 s window, so the FFT 1024 frontend does not keep up with
a 0.125 s hop on that chip.

The repo's models on the same exported clips, 1 s window, 0.125 s hop,
`dad wav` window labels. Model 6 trained on all of Nightjar (including the
validation session 0806-1) and on DroneNoise near drones (entry 043), so
only Halmstad is unseen for it. Thresholds: (a) each model's own, frozen in
entry 043; (b) frozen at 5% false positives on Nightjar 0806-1.

| Scorer | Threshold | Halmstad recall | Halmstad FP rate | Halmstad precision | Nightjar test recall / FP | DroneNoise recall |
|---|---|---|---|---|---|---|
| comb | (a) 1.914 | 0.268 | 0.261 | 0.343 | 0.028 / 0.054 | 0.494 |
| comb | (b) 4.42 | 0.142 | 0.088 | 0.450 | 0.011 / 0.003 | 0.365 |
| lgbm | (a) 0.612 | 0.503 | 0.200 | 0.561 | 0.402 / 0.068 | 0.502 |
| lgbm | (b) 0.723 | 0.396 | 0.140 | 0.591 | 0.312 / 0.016 | 0.333 |
| model 6 | (a) 0.297 | 0.831 | 0.365 | 0.537 | 0.716 / 0.083 | 0.871 |
| model 6 | (b) 0.879 | 0.551 | 0.048 | 0.854 | 0.239 / 0.001 | 0.497 |

Halmstad: 2,190 drone and 4,307 no-drone windows (drone share 34%).
Nightjar test: 5,037 / 5,694. DroneNoise: 3,139 drone windows, no
negatives. One seed each.

At threshold (b), the validation session is hard even for model 6, which
trained on it: recall 0.057 at 5% false positives. Its drone labels come
from motor-on logs, and much of a flight is far and quiet.

**Assessment.**

- The Edge Impulse model has not yet been scored on the shared windows. That
  needs its WebAssembly export downloaded, which waits for Mads's go-ahead.
- Even on its own leaky split it separates the classes poorly (AUC 0.69).
  The Devil's Advocate's decisive objection looks right: Nightjar alone, with
  motor-on labels, does not teach a usable detector. Model 6 at its own
  threshold still has 37% false positives on Halmstad, so R2/R3 are not met by
  any model.
- The 2D preset collapsed to one class after int8 quantisation. Not
  investigated further (the float32 model was not checked).

**Next.** Ask Mads for the WebAssembly download. Then score the Edge Impulse
model locally on the same clips at threshold (b), and give iteration 7 its
verdict.

## 057 - The Edge Impulse model scored on the shared windows, and iteration 7's verdict
- Date: 2026-10-26
- Iteration: 7
- Type: Evaluation
- Commit: uncommitted (on top of ddfc33d)

**What changed.** On Mads's go-ahead, the run-3 model (1D CNN, int8) was
built as a WebAssembly library in Studio and downloaded
(`drone-acoustic-detection-wasm-v1-impulse-#1.zip`, 3.6 MB, unpacked into
`artifacts/edge_impulse/wasm/`). New `scripts/edge_impulse/run_wasm.js` runs
it over the exported clips hop by hop like `dad wav`: 1 s window, 0.125 s
hop, decision at the window's end, the clip's raw int16 values as input (as
Studio feeds them).

**Why.** Entry 055: Edge Impulse's own figures are not results; the model is
judged on the same windows as model 6, `lgbm` and `comb`.

**How.**

```powershell
node scripts/edge_impulse/run_wasm.js artifacts/edge_impulse/wasm/node data/edge_impulse/validation artifacts/edge_impulse/hops_ei_validation.csv
node scripts/edge_impulse/run_wasm.js artifacts/edge_impulse/wasm/node data/edge_impulse/testing artifacts/edge_impulse/hops_ei_testing.csv
uv run python -I scripts/score_hops.py artifacts/edge_impulse/hops_ei_testing.csv --truth-from artifacts/edge_impulse/hops_dscnn_testing.csv --val artifacts/edge_impulse/hops_ei_validation.csv --val-truth-from artifacts/edge_impulse/hops_dscnn_validation.csv
```

The window counts match `dad wav` exactly (7,811 validation, 20,367 test),
and the window labels are taken from its CSVs. As a sanity check of the
runner, validation AUC is 0.647, close to Studio's 0.69 on its own split. The
runner's output was not compared sample by sample with Studio.

**Result.** Threshold frozen at 5% false positives on Nightjar 0806-1:
0.703, validation recall 0.002.

| Scorer | Threshold | Halmstad recall / FP rate | Nightjar test recall / FP rate | DroneNoise recall | AUC test, all | AUC Halmstad |
|---|---|---|---|---|---|---|
| Edge Impulse 1D CNN | (b) 0.703 | 0.000 / 0.000 | 0.023 / 0.000 | 0.000 | 0.358 | 0.462 |
| Edge Impulse 1D CNN | 0.5 (not frozen, for reference) | 0.000 / 0.000 | 0.364 / 0.846 | 0.000 | | |
| model 6 | (b) 0.879 | 0.551 / 0.048 | 0.239 / 0.001 | 0.497 | 0.856 | 0.854 |
| lgbm | (b) 0.723 | 0.396 / 0.140 | 0.312 / 0.016 | 0.333 | 0.727 | 0.755 |
| comb | (b) 4.42 | 0.142 / 0.088 | 0.011 / 0.003 | 0.365 | 0.534 | 0.510 |

Validation AUC: Edge Impulse 0.647, model 6 0.680, lgbm 0.708, comb 0.431.
Same windows and conditions as entry 056. One seed.

Score distribution of the Edge Impulse model: on every Halmstad and
DroneNoise window the score sits at the floor (0.133, max 0.191 on Halmstad
drone, 0.133 on all DroneNoise). On Nightjar's test sessions no-drone windows
score higher than drone windows (means 0.548 against 0.358).

**Assessment.**

- **Failure, by the rule of entry 055.** The Edge Impulse model is far below
  model 6 and below `lgbm` on every source, and its AUC on the test windows is
  below chance. It is not a candidate.
- Off the training rig it returns an almost constant score, so it detects
  nothing there: the recording-chain confound of iteration 1, as the Devil's
  Advocate predicted. A likely contributor, not verified: Edge Impulse's MFE
  scales log energy by the noise floor and clips it, and with the -100 dB
  floor chosen in entry 056 the louder Halmstad and DroneNoise recordings may
  saturate. The other contributor is the data itself: one drone, one rig and
  motor-on labels (entry 056: even model 6 finds 6% of 0806-1's drone
  windows at 5% false positives).
- **Deployment toolchain:** partly useful. The int8 classifier is small
  (12 ms, 20 KB RAM, 41 KB flash on Cortex-M4F; 2 ms in all on a laptop
  CPU, Studio's estimate). The MFE frontend at FFT 1024 does not keep up with
  a 0.125 s hop on that chip (592 ms per window). Profiling model 6 itself
  through Edge Impulse (bring your own model) was not tried.
- The restricted-data control (model 6's recipe retrained on the uploaded
  data only) was not run. So "Edge Impulse pipeline" and "less data" are not
  separated; given the size of the gap this does not change the verdict.
- Test sources were scored once for this verdict. No choice was made from
  them.

**Verdict on iteration 7: hypothesis-free side experiment, negative
result.** An Edge Impulse MFE + 1D CNN trained on Nightjar alone does not
detect drones on other rigs (Halmstad recall 0 at the frozen threshold). Model
6 remains the candidate for the field test. Edge Impulse is kept only as an
option for deployment profiling.

**Next.** Report to Mads. Nothing further in Edge Impulse unless Mads wants
either (a) model 6 profiled through bring-your-own-model for the open
hardware question, or (b) one retry with a level-robust MFE setting, which
needs the noise floor chosen on validation, not on test. The plan of entry
054 (embeddings after the field test, now iteration 8) is unchanged.

## 058 - Requirements: Lund may be uploaded to the private Edge Impulse project
- Date: 2026-10-26
- Iteration: 7
- Type: Requirements
- Commit: uncommitted (on top of 68cb072)

**What changed.** New row in `REQUIREMENTS.md`: Lund may be used for
research, including upload to the private Edge Impulse project. Publishing
results or audio on Lund still needs the authors' permission (the row of
2026-10-14 is unchanged).

**Why.** Mads's decision, in this conversation: "vi bruger det til
forskning". Before deciding, he was told that a dataset with no licence
falls under ordinary copyright (all rights reserved), that local research
use is the safe part, and that an upload hands a copy to a third party.

**Next.** Lund can go into a leave-one-dataset-out run in Edge Impulse. It
holds drone audio only, so it can be a test set for recall, or training
positives.

# Iteration 8 - Edge Impulse on the pool, d3set held out, level-normalised

> **Changed from iteration 7:** (1) the Edge Impulse model trains on every
> prepared dataset but one (Nightjar, FPV, DDL, Lund, DroneNoise M5,
> Halmstad) and is tested on the held-out one, d3set; (2) every training clip
> is scaled to -30 dBFS RMS before upload, and the local runner scales each
> test clip the same way before the Edge Impulse model sees it.
>
> **Why:** Mads asked for leave-one-dataset-out with all data in Edge
> Impulse, with d3set first. Iteration 7's model returned a constant score
> off its training rig, and the recording levels explain it: Nightjar's
> median RMS is about -78 dBFS against about -27 dBFS for Halmstad and
> DroneNoise (entry 059).
>
> **Kept the same:** the MFE settings and the default 1D CNN of entry 056; 1 s
> window, 0.125 s hop; threshold frozen at 5% false positives on a local
> validation session (now Nightjar 0804-3); scoring in the repo with
> `score_hops.py`. Not in the pool: FSD50K and Glasgow (7-Zip is not
> installed on this PC), and the Nightjar mower session (a confuser, as in
> iteration 5).
>
> **Hypothesis:** with the level cue removed and six datasets in training,
> the Edge Impulse model detects drones on the unseen d3set rig (recall clearly
> above `comb`, at most 5% false positives on d3set's own no-drone audio).
>
> **Judged by:** d3set recall, false-positive rate and precision at the
> validation-frozen threshold, against `comb` on the same windows. Model 6
> and `lgbm` trained on d3set (entry 043) and are shown only as an
> in-distribution reference. Iteration 5's held-out-d3set figures (entry 037)
> come from other windows and a 10% operating point, and are quoted for
> context only. No council was convened: the protocol is iteration 5's
> (council entry 033) and the model family is iteration 7's (council entry
> 055). The changes are the data pool and one preprocessing step.

## 059 - The pool exported for Edge Impulse, level-normalised, uploaded
- Date: 2026-10-26
- Iteration: 8
- Type: Preprocessing
- Commit: uncommitted (on top of 68cb072)

**What changed.**

- DDL, d3set, FPV and Lund linked from Google Drive with directory junctions
  and prepared with the existing scripts.
- `export_edge_impulse.py` gained:
  - `--hold SOURCE`: leave one dataset out. Validation is Nightjar 0804-3,
    the mower session is left out, and DroneNoise is limited to microphone M5.
  - `--norm-dbfs X`: scale each training clip to X dBFS RMS, with the peak
    kept below full scale. Silent clips are skipped.
  - A 3600 s cap for no-drone audio per group in hold mode (900 s for drone).
- `run_wasm.js` gained the same scaling as an optional fourth argument.
- Level check that motivated the scaling, `rms_dbfs` in the manifest, median
  per recording:

  | Source | Drone | No drone |
  |---|---|---|
  | Nightjar | -76.7 | -80.7 |
  | Halmstad | -25.0 | -26.8 |
  | DroneNoise | -30.3 | |

- Edge Impulse:
  - The 29 iteration-7 training samples were disabled in Studio, not deleted.
  - The API key has the ingestion role only. Mads put it in
    `data/.ei_api_key`, which is git-ignored.
  - Uploads used the Edge Impulse CLI uploader 1.40.0, installed without its
    native serial-port build.

**How.**

```powershell
uv run python -I scripts/prepare_data.py ddl
uv run python -I scripts/prepare_new_data.py d3set fpv lund
uv run python -I scripts/export_edge_impulse.py --hold d3set --norm-dbfs -30
# upload: edge-impulse-uploader --category training|testing, 200 files per call
```

**Result.** Exported clips (10 s each):

| Split | Source | Drone | No drone |
|---|---|---|---|
| Training | Nightjar | 59.5 min | 60.7 min |
| Training | FPV | 40.3 min | 94.5 min |
| Training | DDL | 96.7 min | |
| Training | Lund | 68.5 min | |
| Training | DroneNoise | 7.2 min | |
| Training | Halmstad | 5.0 min | 9.8 min |
| Training | All (2,653 clips) | 277 min | 165 min |
| Validation | Nightjar 0804-3 | 2.8 min | 12.2 min |
| Testing | d3set | 108.8 min | 11.2 min |

Five test clips are digitally silent. They are not in the normalised copy
uploaded to Edge Impulse's test set, and the runner skips them.

**Next.** Train in Edge Impulse, export, score d3set locally.

## 060 - Requirements: a microcontroller is out of scope
- Date: 2026-10-26
- Iteration: 8
- Type: Requirements
- Commit: uncommitted (on top of 68cb072)

**What changed.** New row in `REQUIREMENTS.md`: deployment on a
microcontroller is out of scope (Mads, this conversation). Laptop and
Raspberry Pi remain open.

**Why.** Mads's decision. It removes the main reason left for Edge Impulse
(entry 057: profiling and export for small targets). Training in Edge
Impulse is also not faster end to end here. Model 6 trained in 459 s on the
local GPU (entry 043). Edge Impulse needed about 20 min for the features of
98,161 windows, plus the upload, before training even began.

**Next.** Finish iteration 8 as planned, so its result is on record. Unless
it is clearly better than expected, training stays in the repo.

## 061 - Edge Impulse on the pool, scored on held-out d3set, and iteration 8's verdict
- Date: 2026-10-26
- Iteration: 8
- Type: Evaluation
- Commit: uncommitted (on top of 68cb072)

**What changed.** Features (98,161 windows; the 29 disabled samples are
excluded) and the same 1D CNN as entry 056 were trained in Edge Impulse on
the upload of entry 059. The int8 WebAssembly build 2 was downloaded through
the deployment API with the ingestion/deployment key, because the browser
download stalled. It was run locally with each clip scaled to -30 dBFS, as
for training.

**How.**

```powershell
node scripts/edge_impulse/run_wasm.js artifacts/edge_impulse/hold_d3set/wasm/node data/edge_impulse_hold_d3set/validation artifacts/edge_impulse/hold_d3set/hops_ei_validation.csv -30
node scripts/edge_impulse/run_wasm.js artifacts/edge_impulse/hold_d3set/wasm/node data/edge_impulse_hold_d3set/testing artifacts/edge_impulse/hold_d3set/hops_ei_testing.csv -30
uv run python -I scripts/score_hops.py artifacts/edge_impulse/hold_d3set/hops_ei_testing.csv --truth-from artifacts/edge_impulse/hold_d3set/hops_dscnn_testing.csv --val artifacts/edge_impulse/hold_d3set/hops_ei_validation.csv --val-truth-from artifacts/edge_impulse/hold_d3set/hops_dscnn_validation.csv
```

(`comb`, `lgbm` and model 6 via `dad wav` on the raw clips, then
`score_hops.py` the same way.)

**Result.** Studio's own validation split (random by window, leaks): int8
accuracy 0.698, AUC 0.67.

Local, 1 s window, 0.125 s hop. Threshold frozen at 5% false positives on
Nightjar 0804-3 (1,241 drone and 5,329 no-drone windows). d3set: 47,669
drone and 4,891 no-drone windows (drone share 91%); Edge Impulse 4,526
no-drone windows, because five silent clips are skipped.

| Scorer | Trained on d3set? | Threshold | d3set recall | d3set FP rate | d3set precision | d3set AUC | Validation AUC |
|---|---|---|---|---|---|---|---|
| Edge Impulse 1D CNN | no | 0.113 | 0.662 | 1.000 | 0.875 | 0.124 | 0.916 |
| comb | no (no training) | 1.905 | 0.198 | 0.136 | 0.934 | 0.533 | |
| lgbm (entry 043) | yes | 0.608 | 0.735 | 0.055 | 0.992 | 0.936 | |
| model 6 (entry 043) | yes | 0.291 | 0.916 | 0.170 | 0.981 | 0.956 | |

Mean Edge Impulse score on d3set: drone 0.230, no drone 0.576. Per
session, the no-drone stretches (before take-off, after landing, and the
pure "SNR test/background noise" recording) score 0.21 to 0.85, and drone
stretches 0.10 to 0.33. The order is inverted in almost every session.

d3set levels are the same for both classes (median RMS -47.9 dBFS drone,
-46.6 no drone), so the scaling does not cause the inversion.

For context only, on other windows at 10% false positives per decision:
iteration 5's repo model with d3set held out (pool + speed perturbation,
three seeds, entry 037) reached recall 0.555 within 50 m and 0.333 at
100-200 m.

**Assessment.**

- **Hypothesis rejected.** On the unseen d3set rig the Edge Impulse model is
  worse than chance (AUC 0.12) and worse than the untrained `comb` (0.53).
  At the frozen threshold it alarms on every no-drone window.
- **Level normalisation worked where it was aimed.** In-distribution
  (Nightjar 0804-3) AUC rose from 0.65 (iteration 7) to 0.92. The constant
  off-rig score of entry 057 is gone; it is replaced by a confident but wrong
  ranking.
- **Likely reason, not verified:** the pool's no-drone audio is narrow
  (Nightjar, FPV, Halmstad; no FSD50K, no Glasgow). Three of six training
  datasets hold drone audio only (DDL, Lund, DroneNoise). So "a microphone
  chain unlike Nightjar, FPV or Halmstad" is evidence for drone, and d3set's
  ambient noise is such a chain. That is iteration 1's confound again.
  Iteration 5's repo pool had FSD50K as varied negatives, and its recipe
  buried near drones in real backgrounds.
- Model 6 and `lgbm` trained on d3set, so their figures here are
  in-distribution references, not a comparison. One seed.
- d3set was scored once for this verdict. Nothing was chosen from it.

**Verdict on iteration 8: negative.** Leave-one-dataset-out in Edge Impulse
with the level cue removed does not give a detector that works on an unseen
rig; it ranks d3set's background above its drones. Combined with entry 060
(no microcontroller), Edge Impulse is stopped for this project. Model 6
stays the candidate for the field test. The embedding plan of entry 054 is
next (iteration 9 after this renumbering).

**Next.** Report to Mads. Open data question: FSD50K and Glasgow need 7-Zip
on this PC before the repo's own pool can be rebuilt here.

## 064 - One synthetic drone in the war soundscape env_001, CNN, lgbm and comb over time
- Date: 2026-10-29
- Iteration: between 8 and 9 (synthetic check)
- Type: Evaluation
- Commit: uncommitted, branch PySound (on top of 3f97135). Entries 062-063
  and the first `scripts/plot_confidence.py` are on master (8d90f3d); the
  script was copied here and extended.

**What changed.** The first Pure Data drone, `drone_s01_01` (seed 1,
100 km/h, start 277.8 m, 40 deg elevation, impact at 10.0 s), added to the
first 10 s of the war soundscape `artifacts/environment/saved/env_001.wav`
(20 s, calm war phase, wind, distant front, machine gun at 3.3-7.2 s
(1.4 km) and 10.5-12.6 s (356 m), small arms at 14.6 s). Both start at 0 s.
`plot_confidence.py` now also plots `comb`: the threshold comes from the
`comb (no ML)` row of `it6_final.json` (hold out none, 1.914), a single
file gets one panel per detector, and a panel whose scores exceed 1 uses
a symlog axis.

**Why.** Mads asked for the drone in a war soundscape, scored by all three
detectors over time.

**How.** Both WAVs are calibrated 1.0 = 110 dB SPL, so they were summed
with no gain. Sidecar: drone from 0.81 s (start distance / 343 m/s, when
the first sound arrives) to impact at 10.0 s, none after. Output in
`artifacts/soundscape/env001_mix/` (git-ignored).
```
uv run python scripts/plot_confidence.py artifacts/soundscape/env001_mix/env001_drone_s01_01.wav --detector artifacts/models/it6_final_none_simdist_dscnn.pt --detector artifacts/models/it6_final_none_simdist_lgbm.pkl --detector comb --out artifacts/soundscape/env001_mix/confidence_env001_drone_s01_01.png --csv artifacts/soundscape/env001_mix/hops.csv
uv run dad wav artifacts/soundscape/env001_mix/env001_drone_s01_01.wav --detector <each> --threshold-from artifacts/runs/it6_final.json   # comb: --threshold 1.9136685717165247
```

**Result.** Synthetic scaffolding, one clip, one drone: never detection
performance (CLAUDE.md section 5). Window 1 s, hop 0.125 s, 153 decisions:
66 drone, 73 no drone, 14 left out (window partly drone).
Levels: drone 31.6 dB SPL at 0-5 s, 39.4 at 5-8 s, 48.8 at 8-9 s,
70.6 at 9-10 s; war noise 62-69 dB SPL over the same spans.

| Detector | Threshold | Recall | False pos. | First hit | Alarm >= 3 s before impact |
|---|---|---|---|---|---|
| CNN | 0.297 | 30.3% | 0.0% | 7.63 s (2.38 s before) | no |
| `lgbm` | 0.612 | 0.0% | 0.0% | none | no |
| `comb` | 1.914 | 19.7% | 12.3% | 6.00 s (4.00 s before) | yes |

`comb` false alarms at 11.25-12.0 s and at 14.9 and 15.6 s, during the
356 m machine gun and the small arms. The CNN stays below 0.12 through
the war noise, including both machine-gun bursts, and rises to 1.0 at
7.6-9.5 s, when the drone is within about 20 dB of the background.

**Assessment.** The CNN gives no false alarms on this clip but alarms
only 2.4 s before impact (R5 asks for 3 s). It needs the drone within
about 20 dB of the war noise. The near-impact drop of entry 062 does not
show here; seed 1 is the lowest-pitched drone. `comb` alarms earlier but
fires on gunfire. One clip and one drone: not evidence for a requirement.

**Next.** None decided. The soundscape test in
`Pure-data+Script/EnviromentTest/soundscape_test.py` does the same for
every drone over many noise clips if a broader check is wanted.

## 065 - Drone speeds x war clips x war levels grid; a Pd render bug fixed
- Date: 2026-10-29
- Iteration: between 8 and 9 (synthetic check)
- Type: Evaluation, Correction
- Commit: uncommitted, branch PySound (on top of 3f97135)

**What changed.**
1. New `scripts/drone_war_grid.py`. For each war level, each drone speed and
   each war clip it mixes one Pure Data drone into a window of the war
   track, scores the mix with CNN, `lgbm` and `comb`, and plots one figure
   per war level: rows are speeds, columns are war clips, and all three
   detectors share each panel. The number of speeds, war clips and levels
   and the level step are arguments (`--speeds`, `--wars`, `--levels`,
   `--war-start-db`, `--war-step-db`). Drones are rendered by
   `DroneSound.pd` through `drone_sim.make_scene`, as `drone_only.py` does,
   and cached per speed. `out.wav` and `throttle.txt` next to the patch are
   restored after rendering.
2. **Bug fix in `Pure-data+Script/EnviromentTest/render.py`.** Pd sometimes
   quits (`wr stop; pd quit`, 300 ms after the last step) before
   writesf~'s disk thread has written the last buffer. The render is then
   about 0.6 s short, and `drone_sim.make_scene` pads it with zeros, which
   removes the closest part of the flight. `render_dry` now checks the
   length, renders again (up to 3 attempts), and fails if it stays short.
   It also reads the raw samples whenever the header holds fewer frames
   than the file.

**Why.** Mads asked for drones at several speeds mixed into several war
clips, repeated at rising war levels, with the counts easy to change, and
all three detectors in one plot.

**How.**
```
uv run python scripts/drone_war_grid.py --out artifacts/drone_war/run1
```
Defaults: drone seed 1, pitch scale 0.85 (as `drone_only.py` seed 1),
10 speeds 100-200 km/h, each starting speed x 10 s away at 40 deg
elevation and hitting the mic 10 s after the clip starts; 10 war clips of
20 s, offsets 0-580 s spread evenly over the 600 s track; war gain -20,
-10 and 0 dB (0 = the track as rendered); no wind. Truth: drone from its
first sound at the mic (start distance / 343 m/s) to impact at 10 s, none
after. Window 1 s, hop 0.125 s, frozen iteration 6 thresholds (CNN 0.297,
`lgbm` 0.612, `comb` 1.914). In the plots `comb` is shown as
s / (s + 1.914), so its threshold sits at 0.5 on the same 0-1 axis.
Outputs (git-ignored): `results.csv` (one row per mix and detector),
`hops.csv`, `grid_war-20dB.png`, `grid_war-10dB.png`, `grid_war+0dB.png`.

The render check: the 100 km/h drone, rendered again after the fix, is
sample-identical to `drone_only/drone_s01_01.wav`. One of the 10 renders
in this run came out short and was redone. **10 of the 50 existing
`drone_only` renders are short** (last sound before 10 s): s01_06 (9.70 s),
s01_10 (7.76 s), s02_01, s02_04, s02_05, s02_06, s02_07, s03_07, s05_03,
s05_08 (9.69-9.71 s). They were not re-rendered. Entry 062's drop to near
zero before impact holds for seed 4 (no short render), but two of its
seed 5 flights are short.

**Result.** Synthetic scaffolding, never detection performance (CLAUDE.md
section 5). 100 mixes per level, each 153 decisions; drone share of the
scored windows about 0.48.

| War gain | Detector | Recall | False pos. | Alarm >= 3 s before impact | No alarm |
|---|---|---|---|---|---|
| -20 dB | CNN | 52.9% | 0.8% | 83/100 | 1 |
| -20 dB | `lgbm` | 3.0% | 0.0% | 0/100 | 38 |
| -20 dB | `comb` | 23.7% | 13.5% | 80/100 | 0 |
| -10 dB | CNN | 35.5% | 0.9% | 70/100 | 6 |
| -10 dB | `lgbm` | 1.2% | 0.0% | 0/100 | 76 |
| -10 dB | `comb` | 15.3% | 13.5% | 56/100 | 1 |
| 0 dB | CNN | 20.1% | 0.9% | 46/100 | 16 |
| 0 dB | `lgbm` | 0.4% | 0.0% | 0/100 | 90 |
| 0 dB | `comb` | 10.3% | 13.5% | 58/100 | 6 |

The war clip matters more than the speed. CNN recall per clip at -10 dB:
0 s 60%, 64 s 48%, 129 s 2%, 193 s 21%, 258 s 43%, 322 s 1%, 387 s 6%,
451 s 56%, 516 s 74%, 580 s 45%. The clips at 129, 322 and 387 s lie in
or next to the track's intense phases (121-224 s and 309-388 s).
`comb`'s false-positive rate does not change with the war level: it
reacts to gunfire whatever the gain.

**Assessment.** On the CNN, each 10 dB of war costs about 15-17 points of
recall and 13-24 alarms in 100 at 3 s before impact, with false
positives under 1% throughout. In intense firefights it detects almost
nothing. `lgbm` is not useful here. `comb` gives early alarms but about
13% false positives on gunfire. One drone sound (seed 1, the lowest
pitch), no wind, one synthetic war track: no requirement is judged from
this.

**Next.** None decided. Re-rendering the 10 short `drone_only` files
(`uv run python drone_only.py` in `Pure-data+Script/EnviromentTest`) would
make entries 062-064 cleaner.

## 066 - A window for the drone-in-war test
- Date: 2026-10-29
- Iteration: between 8 and 9 (tooling)
- Type: Infrastructure
- Commit: uncommitted, branch PySound (on top of 3f97135)

**What changed.** New `scripts/drone_war_gui.py`, a matplotlib window in the
style of `stress_gui.py`. Sliders set the parameters of
`scripts/drone_war_grid.py`: number of speeds and their range, number of war
clips, number of war levels, first level and step in dB, wind, and the drone
(seed, pitch scale, elevation, time to impact). Run starts the grid in a
background thread, Stop ends it after the current clip, and each war
level's grid (all three detectors per panel) appears as it finishes,
next to a table of recall, false positives, alarms 3 s before impact and
missed approaches. `< level` and `level >` page between levels. Output in
`artifacts/drone_war/gui/`.

To serve both the CLI and the window, `drone_war_grid.py` was split into
`run()` (the grid, writing `results.csv` and `hops.csv`), `draw_grid()`
(draws a level into any figure) and `summarise()`. Models and the war
track load once per session. The drone cache moved to
`artifacts/drone_war/drones/`, shared by all runs. Its file names now hold
every drone parameter (seed, pitch scale, elevation, flight time, speed),
because the old names (seed and speed only) would have reused a wrong
drone when the GUI changed the pitch. The 10 drones of entry 065 were moved
there under the new names.

**Why.** Mads asked for a GUI, started from the command line, to change the
parameters, run the virtual test and see the graphs.

**How.**
```
uv run python scripts/drone_war_gui.py
```
Checks: after the split, `drone_war_grid.py --speeds 2 --wars 2 --levels 1`
gave results identical to entry 065's `results.csv` for the same 12
(mix, detector) cells. The window was driven headless (Agg backend,
sliders set to 2 speeds x 3 war clips x 2 levels, Run pressed): it
finished 12 mixes, drew both levels and filled the table. It was not
opened as a real window here.

**Result.** Tooling; no new numbers.

**Next.** None decided.

## 067 - The drone-in-war window takes three counts, no sliders
- Date: 2026-10-29
- Iteration: between 8 and 9 (tooling)
- Type: Infrastructure
- Commit: uncommitted, branch PySound (on top of f3dd9f8)

**What changed.** Mads: the drone should be set only by how many speeds to
run between 100 and 200 km/h, the war level the same way, and no sliders.
`scripts/drone_war_gui.py` (entry 066) now has three number fields:
drone speeds (100-200 km/h), war clips (spread over the 10 min track) and
war levels (-20 to 0 dB), each 1 to 20. The other drone and wind settings
are fixed at the defaults of entry 065 (seed 1, pitch scale 0.85,
40 deg, 10 s to impact, no wind). Bad input shows a message in the window.
In `scripts/drone_war_grid.py` the war levels now work like the speeds:
`--levels` gains spread evenly from `--war-min-db` (-20) to `--war-max-db`
(0). These replace `--war-start-db` and `--war-step-db`. The default of
3 levels gives -20, -10 and 0 dB, the same as entry 065.

**How.**
```
uv run python scripts/drone_war_gui.py
```
Check: driven headless with "abc" and "0" typed in the speeds field (both
refused with a message), then 2 speeds x 3 war clips x 2 levels: levels
-20 and 0 dB drawn, 12 mixes, table filled. Not opened as a real window.

**Result.** Tooling; no new numbers.

## 068 - Flush plots button with confirmation in the drone-in-war window
- Date: 2026-10-30
- Iteration: between 8 and 9 (tooling)
- Type: Infrastructure
- Commit: uncommitted, branch master (on top of a45d424)

**What changed.** `scripts/drone_war_gui.py` has a "Flush plots" button.
It deletes nothing by itself: it shows how many PNG files under
`artifacts/drone_war/` would go and their size, and reveals a red
"Confirm" button. Confirm deletes those PNGs and clears the grid and the
table in the window. Results (CSV) and the cached drones are kept. Run
cancels a pending flush, and neither button deletes anything while a run
is going.

**Why.** Mads asked for a way to free disk space from the plots, with a
confirmation step against mistakes.

**How.** Check: the window driven headless with its root pointed at a
scratch folder holding two PNGs and a CSV. Flush alone deleted nothing,
Run hid Confirm, then Flush and Confirm deleted both PNGs and kept the
CSV. At the time of writing the plots under `artifacts/drone_war/` are
6 files, 4.4 MB. The CSVs (7.7 MB) and WAVs (drone cache and listening
files, 28 MB) are larger. Not opened as a real window.

**Result.** Tooling; no new numbers.

## 069 - Drone-in-war window: field limits, war off, lgbm dropped
- Date: 2026-10-30
- Iteration: between 8 and 9 (tooling)
- Type: Infrastructure
- Commit: uncommitted, branch master

**What changed.**
1. Each number field in `scripts/drone_war_gui.py` shows its limits
   ("min 1, max 20") and refuses values outside them with a message. Drone
   speeds and war clips: 1 to 20. War levels: 0 to 20.
2. **War levels 0 = no war.** In `scripts/drone_war_grid.py` (and the
   window), `--levels 0` runs the drone alone. The war gain is minus
   infinity (multiplier 0), shown and saved as "off" (`grid_waroff.png`,
   `war_gain_db = -inf` in the CSVs). There is one war clip column, since
   every clip would be the same silence. Before this, 0 was refused (entry
   067).
3. **`lgbm` removed from the drone-in-war tool** at Mads's request: only the
   CNN and `comb` are scored and plotted. CLAUDE.md section 5 asks for both
   baselines next to any reported model. This tool produces synthetic
   scaffolding, not reported performance, so the rule is not broken here.
   Any number taken from this tool into a result needs `lgbm` back.

**Why.** Mads tried 0 war levels to hear and score the drone alone, and got
no run.

**How.** Check, headless: 21 war levels and 0 war clips were refused with
their limits. Then 2 speeds, 3 war clips, 0 levels gave one level "off",
2 mixes (100 and 200 km/h, one column) and detectors CNN and `comb` only
in `results.csv`. Not opened as a real window.

**Result.** Tooling. For orientation only, synthetic, drone seed 1 alone
with no background: CNN recall 34.9%, `comb` 94.4% over the two mixes.
That fits entry 062: the CNN hardly reacts to this low-pitched drone until
the last seconds.

## 070 - A hovering Pd drone at fixed distances
- Date: 2026-10-30
- Iteration: between 8 and 9 (synthetic data)
- Type: Data
- Commit: uncommitted, branch master

**What changed.** New `scripts/hover_drones.py`. It renders the seed 1 drone
of `drone_war_grid.py` hovering (0 km/h) with `DroneSound.pd`: the same
throttle model and parameters (base 66.6, wobble 5.6, pitch scale 0.85,
L_1m 78.5 dB SPL) but no dive, for 10 s. The result is calibrated to L_1m
at 1 m, as `drone_sim.make_scene` does, and carried to each distance by
`dad.propagation.propagate` (1/r and ISO 9613-1 air absorption). The moving
flights use 1/r only (`drone_sim.make_scene`), so at a given distance the
two differ by the air absorption: under 1 dB in RMS here.

**Why.** Mads asked for drone sounds at 0 km/h from 50, 100, 200 and 300 m.

**How.**
```
uv run python scripts/hover_drones.py --distances 50 100 200 300
```
Output in `artifacts/drone_war/hover/` (git-ignored): a calibrated WAV
(1.0 = 110 dB SPL), a JSON and a `_listen` WAV per distance. The listening
files share one gain (the 50 m file peaks at -1 dBFS), so their level
difference stays audible.

**Result.** RMS at the microphone: 50 m 44.5, 100 m 38.4, 200 m 32.2,
300 m 28.6 dB SPL (1/r alone gives 44.5, 38.5, 32.5 and 29.0). All four
files have sound to the end (10.00 s). Not scored by any detector.

## 071 - Listening files are no longer rescaled
- Date: 2026-10-30
- Iteration: between 8 and 9 (synthetic data)
- Type: Correction
- Commit: uncommitted, branch master

**What changed.** Every `_listen.wav` made in this session was rescaled for
playback: the flights and mixes to a -1 dBFS peak, the hover files (entry
070) to one common gain. Mads judged audibility by ear and the rescaling
misled him. A flight rescaled to its impact peak puts the 300 m part
60 dB down, while the hover files, rescaled to the 50 m file, put 300 m
only 16 dB down. All seven listening files were rewritten unscaled, as
float32 at the calibration of every render here (1.0 = 110 dB SPL). They
are now sample-identical to the calibrated files they came from.
`scripts/hover_drones.py` writes its `_listen.wav` unscaled from now on.

Files: `artifacts/drone_war/run1/listen/` (war -20 dB and 0 dB mixes,
drone alone at 200 km/h) and `artifacts/drone_war/hover/` (50, 100, 200,
300 m).

**Result.** Peaks: drone alone at 200 km/h -22.8 dBFS (impact), hover at
50/100/200/300 m -55.1/-61.2/-67.5/-71.2 dBFS, war 0 dB mix +1.0 dBFS,
which may clip in a player. The 50 m hover is about 32 dB below the flight's
impact peak, so the hover files are very quiet at a normal volume. That is
their calibrated level. The `_listen.wav` files from `drone_only.py`
(Pure-data folder, someone else's code) are still peak-normalised and
were not touched.

## 072 - Drone-in-war window: min, max and count; grey truth shading removed
- Date: 2026-10-30
- Iteration: between 8 and 9 (tooling)
- Type: Infrastructure
- Commit: uncommitted, branch master

**What changed.**
1. The drone speeds and the war levels are now set as min, max and count
   in `scripts/drone_war_gui.py`. The run takes `count` values from min to
   max, both included and evenly spaced. Mads's examples: speed min 200,
   max 200, count 1 gives 200 km/h; war min -190, max +10, count 5 gives
   -190, -140, -90, -40 and +10 dB. The rules live in
   `drone_war_grid.speeds_and_gains()`, so the CLI (`--speed-min`,
   `--speed-max`, `--speeds`, `--war-min-db`, `--war-max-db`, `--levels`)
   follows them too. A min above its max is refused, and so is a count
   that disagrees with min and max: count 1 needs min = max, and min = max
   needs count 1. War level count 0 still means no war. Field limits:
   speed 1-1000 km/h (sound is 1235 km/h), war -300 to +30 dB, counts
   1-20 (war levels 0-20). A comma is accepted as the decimal point.
   Before this change, an n = 1 level silently used the min.
2. **The grey "drone in truth" shading is gone** from the grid
   (`draw_grid`, so the GUI and the CLI PNGs) and from
   `scripts/plot_confidence.py`, at Mads's request. The truth is still
   used for every number in the tables and CSVs; it is only not drawn.
3. File names now print speeds and war gains in full (`:g`), so close values
   such as -0.75 and -1 dB no longer share a name. The 10 cached drones were
   renamed to match, so nothing is rendered again.

**How.** Check, headless: Mads's two examples gave [200] km/h and
[-190, -140, -90, -40, 10] dB. Count 3 at min = max, count 1 at
100-200 km/h, min above max and speed 0 were each refused with a message.
A run at 200 km/h, war -190 to +10 dB in 5 levels and 1 war clip drew all
5 levels with no shading. `plot_confidence.py` and the CLI refusal were
also checked. Not opened as a real window.

**Result.** Tooling. One observation, synthetic, one mix: at war -190 dB
(the war track about 170 dB below full scale, far below anything audible
or any real microphone's noise floor) `comb` still alarmed on the machine
gun (false positives 20.5%), while the CNN gave none. `comb`'s score
ignores level, so a harmonic sound at any level, however faint, can set it
off.

## 073 - Drone-in-war window: one grid of drone speed x war level
- Date: 2026-10-30
- Iteration: between 8 and 9 (tooling)
- Type: Infrastructure
- Commit: uncommitted, branch master

**What changed.** Mads cares only about drone speed and war noise for now,
and wants every graph on screen at the end of a run.
`scripts/drone_war_gui.py` now draws one grid: rows are drone speeds,
columns are war noise levels. Each column is headed with its level in
bold ("war -40 dB", or "no war") and each row with its speed in bold
("150 km/h"). The headings "war noise level →" and "drone speed ↓" sit
over the columns and beside the rows. The grid fills in column by column
as each level finishes, and holds every speed x level panel at the end.
The "< level" / "level >" paging and the war clip count are gone. The
window uses one war clip, the first 20 s of the war track (calm phase).
The table on the left has one line per war level: recall, false
positives and alarms 3 s before impact, for the CNN and `comb`. The run
saves `grid_speed_x_war.png` in place of one PNG per level.

`drone_war_grid.draw_grid()` now takes any rows and columns. `draw_level()`
(rows = speeds, columns = war clips, used by the CLI's per-level PNGs) and
`draw_speed_war()` (rows = speeds, columns = war levels) are two layouts
on it. The CLI is otherwise unchanged.

**How.** Check, headless: 100-200 km/h in 3 speeds and -40 to 0 dB in 5
levels gave 15 panels, columns "war -40 dB" ... "war +0 dB", rows
"100 km/h", "150 km/h" and "200 km/h", and a filled table. 200 km/h with
war count 0 gave one panel, "no war". The CLI was also checked: 2 speeds
x 2 clips x 1 level. Not opened as a real window.

**Result.** Tooling. Numbers from the check, for orientation only
(synthetic, 3 mixes per level, war clip at 0 s): CNN recall 88.9, 87.8,
82.5, 60.3 and 26.5% at war -40 to 0 dB with no false positives. `comb`
recall 71.4 to 15.3% with 21.2% false positives at every level.

## 074 - Drone-in-war window: "no drone" row and "no war" column
- Date: 2026-10-30
- Iteration: between 8 and 9 (tooling)
- Type: Infrastructure
- Commit: uncommitted, branch master

**What changed.** Mads wants a control at both ends: a run with no drone
(war only) and a run with no war (drone only), set with checkboxes beside
the drone fields and the war fields.
`scripts/drone_war_gui.py` has two checkboxes. "no drone" adds a row with
no drone, so every window there is a no-drone window and only false
alarms are measured. "no war" adds a column with no war, the drone alone.
They come on top of the speed and level sweeps, and with a box ticked its
count may be 0. The cell with neither (silence) is not run and shows
"silence: not run". This replaces "war level count 0 = no war" (entry 069):
a count of 0 now means "none from the sweep".
`scripts/drone_war_grid.py` has the matching flags `--no-drone` and
`--no-war`. The rules sit in `speeds_and_gains()`: at least one row and
one column, and not only silence. The no-drone row is speed 0
(`NO_DRONE`) in the CSVs, and the no-war column is `-inf` dB. The table's
"no alarm" count leaves out no-drone mixes. The legend now takes its lines
from the first panel that has any, since the first panel can be the blank
silence cell.

**How.** Check, headless: count 0 with no box ticked was refused for both
the drone and the war, and so was "no drone + no war only". 100-200 km/h
in 2 speeds plus no drone, and -20 to 0 dB in 2 levels plus no war, gave
3 x 3 panels: rows "no drone", "100 km/h", "200 km/h"; columns "no war",
"war -20 dB", "war +0 dB"; one blank silence panel; 8 mixes. The no-drone
row has empty recall and a false-positive rate in `results.csv`. CLI:
`--speeds 0 --no-drone` ran. (The CLI check of `--speeds 0` alone was
refused by the war-level rule first, so the speed refusal was checked in
the window only.) Not opened as a real window.

**Result.** Tooling. From the check, for orientation only (synthetic, war
clip at 0 s, no drone): the CNN raised no false alarm at -20 or 0 dB;
`comb` alarmed on 10.5% of the windows at both levels, on the machine
gun at 12.8-14.8 s.

## 075 - The virtual test uses war-shaped noise in place of the war track
- Date: 2026-10-30
- Iteration: between 8 and 9 (synthetic check)
- Type: Evaluation
- Commit: uncommitted, branch master (on top of 1dd077d)

Note on numbering: the merge 1dd077d kept this log's entries 068-074 and
dropped the war-shaped-noise entries 068-075 from commit 5b7161b. Their
text is still in git (`git show 5b7161b -- docs/REPORT.md`). Restoring them
is Mads's call. This entry is numbered after the last one in the file.

**What changed.** Mads: substitute the war noise in the virtual test with
the new war-shaped noise.
`scripts/drone_war_grid.py` has `--war-source shaped` (the default) and
`track` (the old war track). The shaped noise comes from
`war_shaped_noise.shaped_noise()`: random-phase noise with exactly the
committed 44.1 kHz gain table
(`Pure-data+Script/WarShapedNoise/war_gain.txt`), seed 1000 + clip index
(the doc's test seeds start at 1000). Its level is set to the RMS of the
war-track window it replaces, so "war 0 dB" is as loud as before (83.2 dB
SPL for the first 20 s) and the dB columns stay comparable with entries
065-074. `results.csv` gains a `war_source` column. The window always uses
the shaped noise.
`scripts/war_shaped_noise.py` loaded pyplot on the Agg backend at import,
which would have taken the GUI's window away. It now loads pyplot only when
it plots. Its build and `--check` behaviour are unchanged.

**How.**
```
uv run python scripts/drone_war_gui.py
uv run python scripts/drone_war_grid.py --war-source shaped ...
uv run python scripts/war_shaped_noise.py --sr 44100 --check artifacts/drone_war/run1/listen/war_shaped_noise_0dB_listen.wav
```
The listening file (20 s, 0 dB, seed 1000, unscaled, 1.0 = 110 dB SPL):
RMS 83.2 dB SPL, the same as the war track's first 20 s, with peak -13.3 dBFS
against +1.0 dBFS for the track. The spectrum check against the table gave
0.40 dB at most per 1/3-octave band, RMS 0.14 dB.

**Result.** Synthetic, one noise clip, 2 speeds (100, 200 km/h) plus a
no-drone row, war -20 and 0 dB plus a no-war column:

| War | CNN recall | CNN false pos. | `comb` recall | `comb` false pos. |
|---|---|---|---|---|
| none (drone only) | 34.9% | - | 94.4% | - |
| shaped -20 dB | 100.0% | **100.0%** | 4.8% | 3.6% |
| shaped 0 dB | 100.0% | **100.0%** | 3.2% | 3.6% |

**The CNN takes war-shaped noise for a drone.** With no drone at all, its
score stays at 0.4-0.9 for the whole clip, over its threshold of 0.30, at
both levels. The same score at -20 and 0 dB is expected: the frontend
removes each window's level, and the noise is all there is. Its recall of
100% is therefore no evidence of detection. `comb` barely reacts to the
noise, and barely to the drone in it. Likely reason, not verified: the
noise is steady and carries strong energy at 600-750 Hz from the shout
and siren clips (`docs/war_shaped_noise.md`), the band of the drone's
harmonics, and the CNN never saw steady noise with that spectrum as a
negative. One noise seed, one drone sound.

**Next.** None decided. The CNN's false alarms on steady noise matter more
than any recall in this condition. A check with the shout and siren clips
left out of the table would test the likely reason.

## 076 - Listening files: 150 km/h drone in war-shaped noise at -40 and -80 dB

**Why.** Mads asked for listening files for the 150 km/h drone mixed into
war-shaped noise at -40 dB and at -80 dB.

**What.** The mixes come from the virtual test itself, with `--keep`, so
they are what the detectors scored, unscaled (1.0 = 110 dB SPL). Commit
1dd077d, plus the uncommitted changes from entry 075.
```
uv run python scripts/drone_war_grid.py --speeds 1 --speed-min 150 --speed-max 150 --wars 1 --levels 2 --war-min-db -80 --war-max-db -40 --keep --out artifacts/drone_war/listen150
```
Files: `artifacts/drone_war/listen150/mixes/war-40dB_150kmh_off0.wav` and
`war-80dB_150kmh_off0.wav`, 20 s at 44.1 kHz. Seed-1 drone, first sound at
1.21 s, impact at 10 s. The noise is 43.2 dB SPL at -40 dB and 3.2 dB SPL
at -80 dB, so the -80 dB file is in effect the drone alone. The whole file
is 55.8 and 55.5 dB SPL RMS, peak 0.079.

**Result.** Synthetic, one clip each, 63 drone windows and 75 no-drone
windows, with 15 left out at the boundaries:

| War | CNN recall | CNN false pos. | `comb` recall | `comb` false pos. |
|---|---|---|---|---|
| shaped -80 dB | 28.6% | 0.0% | 90.5% | 4.0% |
| shaped -40 dB | 100.0% | **100.0%** | 25.4% | 4.0% |

At -40 dB the noise sits about 12 dB under the drone's RMS, and the CNN
still alarms on every window. This agrees with entry 075: the CNN reacts
to the noise itself, not to its level. At -80 dB the CNN behaves as it
does on the drone alone (entry 075: 34.9%).

**Next.** None. These are listening files.

## 077 - Drone-in-war window: click to zoom, and page 2 with confidence over SNR

**Why.** Mads asked for panels that can be clicked and zoomed, and for a
second page with confidence over SNR.

**What.** Commit 1dd077d, plus the uncommitted changes from entries 075-076.
- `scripts/drone_war_grid.py`: `hop_snr` gives each hop's SNR in dB. It
  takes the drone's energy over the noise's energy (war plus wind) in the
  hop's 1 s window, both band-passed with a causal 4th-order Butterworth
  filter to 50-8000 Hz, the frontend's mel range. A window with no drone
  sound is -inf; a window with no noise is +inf.
  - `run()` keeps the drone and the noise apart until the mix, stores the
    SNR per hop in each cell, and writes it to `hops.csv` as `snr_db`.
  - `draw_grid` takes `x="time"` or `x="snr"`. On the SNR page each hop is
    a dot, and a panel with no finite SNR says so ("no drone", "no noise").
- `scripts/drone_war_gui.py`:
  - A "Page 2: SNR" / "Page 1: time" button switches the x axis.
  - Clicking a panel opens it alone in its own window, with matplotlib's
    zoom and pan toolbar. Scrolling zooms the x axis around the mouse.
  - After a run, both pages are saved: `grid_speed_x_war.png` and
    `grid_speed_x_war_snr.png`.

**First version.** It filtered with `sosfiltfilt`. The backward pass leaked
filter ringing into the silence before the drone's first sound, which gave
SNRs down to -749 dB. The causal filter leaves those windows at exactly
zero drone energy (-inf, not plotted).

**Checks.**
- `hop_snr` on a 1 kHz sine at amplitude 1 against a 500 Hz sine at 0.1
  gives [-inf, 20.0, 20.0] for windows ending at 1, 2 and 3 s, with the
  drone silent for the first second.
- A headless GUI run: speeds 100 and 200 km/h plus no drone, war -40, -20
  and 0 dB plus no war, 11 mixes. Both pages draw; the five panels with
  no SNR are labelled; a simulated click opens the panel; a scroll narrows
  its x range.
- In that run, at 200 km/h the SNR reaches 24.2 dB at -40 dB, and the
  -20 dB column sits 20 dB lower, as it should.
- `uv run pytest`: all pass.
- Not checked in a live window: real mouse clicks, and the toolbar.

**What page 2 shows (synthetic, one drone, one noise clip).** The CNN
stays at 0.5-0.85 from -60 to -10 dB SNR at every war level, so its score
does not follow SNR (see entry 075). `comb` stays near 0 below about
-10 dB SNR. It reaches its threshold only in the -40 dB column, at
100 km/h, from about -10 dB SNR up.

**Next.** None. A tool change.
