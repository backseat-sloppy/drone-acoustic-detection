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
| Iteration 9 | 090 to 094 | Iteration 5's recipe on real data, with the frontend as the only change: MFE (log-mel), MFCC, linear spectrogram | The supervisor's feedback (entry 086) and Mads's choice to compare frontends on real data, after dropping synthetic training (entries 087-089) | **Hypothesis rejected for both.** FPV mean at 10% own false alarms: MFE 0.669, MFCC 0.485, linear 0.692 (+2.3 against +5 required). Linear gains near the rigs (+23-26 points within 50 m) but loses on d3set at 100-200 m (-6.7), on Lund (-18) and alarms 12 points more on mower audio. MFE stays. |
| Iteration 11 | 100 to 105 | Steady-noise family (random slopes, random-phase copies of rig no-drone windows) as negatives and maskers; no war audio in training. Judged on unseen real noise stems and the mower session, with recall guards | All networks alarm on 99-100% of war-shaped noise and on pink/brown noise, but on 0% of the war track: the failure is steady noise, not war (entry 100) | In progress. Council held (entry 100). War-shaped noise matches war audio only in its long-term spectrum; real tank clips alarm 45-57% (entry 101). Step 0 (entry 102): the networks rely on the harmonic lines; steadiness pushes toward drone; noise alarms only at -3 to -6 dB/octave. Step 1 (entry 103): generators and the `simdist_steady` recipe in the sampler, tested. War-shaped noise dropped on Mads's decision; war is tested on real war audio only (entry 104). 36 of 97 FSD50K stems are from FSD50K dev, which training draws from: reported apart (entry 105). Step 2 next. |
| Soundscape line (branch it11-soundscapes, own section at the end) | 095 to 105 and 101 to 118 of that section | Synthetic Pd drones in random labelled soundscapes (confusers, transients), then realism (rotor detune, mic, room), real Glasgow drones, a CRNN, steady-noise arms | The supervisor's notes (entry 086); synthetic training allowed (entry 087) | **Not met; validation only.** On 10 sealed ARIS excerpts with the war track at 0 dB, the share of alarmed windows is it11 CNN 13.5%, it11r CNN 23.1%, it9 CRNN (real rigs) 58.3%. Steady-noise arms are training. |
| Iteration 11.1 (branch 11.1) | 135 to 136 | AudioSet-pretrained PANNs Cnn14_16k on the e133 compliant data: P0 AudioSet scores (no training), P1 frozen encoder + linear probe, P2 LP-FT | Entry 134: from scratch, the only allowed real drones (Glasgow d1-d16, one rig) teach "a Glasgow recording"; pretraining is the established lever for unseen recording chains (entries 053, 131) | **P2 rejected, P1 mixed** (validation only). Frozen encoder: best Nightjar AUC (0.908) and drone-vs-mower separation, mower gate passes, but recall at 5% FA misses (b)/(c) and pooled real recall is 0.49. Fine-tuning overfits the Glasgow rig (perfect there, worse everywhere else). The training-data rule stays the binding limit (entry 142). |
| Iteration 11.2 | 143 to | Real drones from several rigs in training (Mads, entry 143): it9's pooled recipe without FPV plus steady noise, leave one rig out, CRNN against the frozen Cnn14_16k probe | Entries 134 and 142: under the single-rig rule neither training from scratch nor pretraining reaches gates (b) and (c) | **No gate passes** (validation only). CRNN on unseen d3set 0.64-0.70 recall @5% FA (e133: 0.58), on unseen Nightjar 0.64-0.70 (e133: 0.65-0.67); frozen probe worse (0.39-0.58). With FPV test, leave-one-rig-out leaves one real rig per fold (entry 146). |

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

## 078 - Drone-in-war window: a click opens four graphs for one case

**Why.** Mads asked that a click on a grid panel open a window with four
graphs for that case: drone and war levels in dB over time, SNR over
time (noise = war + wind), confidence over time, and confidence over SNR.

**What.** On top of commit 66d6b0a, uncommitted.
- `scripts/drone_war_grid.py`: `hop_snr` is replaced by `hop_db`, the RMS
  level in dBFS of one signal in each hop's 1 s window, with the same
  causal 50-8000 Hz band-pass. `run()` stores `drone_db` and `noise_db`
  per hop in each cell; `snr` is now their difference, so the SNR values
  are the same as in entry 077. `hops.csv` is unchanged.
- `scripts/drone_war_gui.py`: the click no longer opens the single big
  panel (entry 077). It opens a 2 x 2 window: (1) drone and noise level in
  dBFS over time, (2) SNR over time, (3) CNN and comb confidence over
  time with thresholds, (4) confidence over SNR. The three time graphs
  share the x axis; scroll zoom still works. The "noise" line is war plus
  wind, since that is what the SNR divides by.

**Checks.**
- `hop_db` on a 1 kHz sine at half of full scale gives -9.03 dBFS, as
  expected (assert in a scratch script, not in `tests/`).
- A headless run: 150 km/h plus no drone, war -10 dB, one clip each, then
  a simulated click on each panel. Both windows draw. At 150 km/h the
  drone rises from about -107 to -41.5 dBFS before impact at about 11 s;
  the noise sits at -37.5 to -36.9 dBFS. The no-drone window labels its
  SNR graphs "no SNR: no drone".
- `uv run pytest`: all pass.
- Not checked in a live window: real mouse clicks.

**Next.** None. A tool change. Synthetic scaffolding (section 5), not
detection performance.

## 079 - Drone-in-war detail window: frequency over time replaces SNR over time

**Why.** Mads asked to swap graph 2 (SNR over time, entry 078) for
frequency over time, with the drone and the war as separate lines like
graph 1.

**What.** On top of commit a6a359d, uncommitted.
- `scripts/drone_war_grid.py`: new `hop_peak_hz`, each hop's dominant
  frequency: the peak of the Hann-windowed spectrum of the 1 s window
  (1 Hz bins), searched within 50-8000 Hz. NaN where the window is
  silent. `run()` stores `drone_hz` and `noise_hz` per hop in each cell.
  `hops.csv` is unchanged.
- `scripts/drone_war_gui.py`: graph 2 is now "dominant frequency over
  time", drone and noise (war + wind) lines, linear Hz axis. The SNR is
  still computed and shown in graph 4.

**Checks.**
- `hop_peak_hz` on 1 s of silence then a 1 kHz sine gives [NaN, 1000,
  1000] for windows ending at 1, 2 and 3 s (assert in a scratch script).
- Headless run as in entry 078 (150 km/h plus no drone, war -10 dB): the
  drone's peak runs from 221 to 315 Hz, rising in the last 2-3 s before
  impact; the noise's peak jumps between 50 and 179 Hz.
- `uv run pytest`: all pass.
- Not checked in a live window: real mouse clicks.

**Next.** None. A tool change. Synthetic scaffolding (section 5).

## 080 - A window that looks for drones in chosen audio files

**Why.** Mads asked for a GUI where you only open one or more audio files
and the system tries to find drones in them: the files are the whole test,
with nothing to set. (A first attempt that added a file button to
`scripts/drone_war_gui.py`, mixing the files into war noise, was undone by
Mads before it was logged; it is not part of the repo.)

**What.** On top of commit 3545b69, uncommitted. New
`scripts/detect_gui.py`:
- "Open audio files..." (Tk multi-select dialog: WAV, FLAC, OGG, MP3).
- Each file is read by `dad.wavio.read_wav` (channel 0, 16 kHz) and scored
  by `dad.wavio.stream` with the CNN and comb at their frozen iteration 6
  thresholds (`drone_war_grid._detectors()`, the same as the drone-in-war
  window): 1 s window, 0.125 s hop. No noise is added and nothing is set.
- A list gives each file's verdict per detector: DRONE if any window
  alarmed, with the share of windows that alarmed. Clicking a file (or
  Previous / Next) shows its spectrogram and each detector's confidence
  over time, with thresholds, alarms shaded, first alarm and longest
  unbroken alarm. Play / Stop sound plays the file as the detectors hear
  it, normalised.
- Unreadable files and files shorter than the 1 s window are listed with
  the reason; the batch goes on. Scoring runs in a background thread.
- `artifacts/detect_gui/results.csv`: one row per file and detector.
- The docstring warns that the detectors assume 110 dB SPL full scale, so
  a verdict on an uncalibrated recording is a hint, not a measurement.

**Checks.**
- Headless run (Agg backend, dialog stubbed) on five made-up files: an 8 s
  WAV with a 180 Hz tone from 3 s, 8 s stereo noise, 25 s of 16 kHz noise
  (FLAC), a 0.5 s file and a file that is not audio. comb alarmed on the
  tone file (first alarm 3.12 s, longest 5.00 s, 70 % of windows) and on
  nothing else; the CNN alarmed on none; the short and broken files were
  listed with their reasons. Next, click-to-select and wrap-around worked;
  the CSV held 10 rows. Made-up audio: these numbers say nothing about
  detection performance (section 5).
- `uv run pytest`: all pass.
- Not checked in a live window: the real file dialog, mouse clicks,
  sound.

**Next.** None. A tool.

## 081 - Detection window: stats per file (peaks and time above threshold)

**Why.** Mads asked that clicking a red file in `scripts/detect_gui.py`
(entry 080) opens stats for that track: the number of peaks above
threshold and the time above threshold, for both the DSP detector (comb)
and the CNN.

**What.** On top of commit 3545b69, uncommitted.
- A peak = an unbroken stretch of decisions at or above the threshold
  (`peaks()`). Time above threshold = alarmed decisions x the 0.125 s hop,
  since each decision holds until the next one. Both are stored per file
  and detector, and added to `artifacts/detect_gui/results.csv`
  (`peaks_above_threshold`, `time_above_threshold_s`).
- Clicking a red file (any detector alarmed) still selects it, and also
  opens a stats window. For each detector it shows confidence over time
  with each peak shaded and numbered, plus peaks above threshold, time
  above threshold (and its share of the scored time), longest peak, first
  alarm, and a table of the peaks (start, end, duration, highest
  confidence; the first 12). Clicking a file that is not red opens nothing.

**Checks.**
- Headless (Agg backend, dialog stubbed): a made-up 20 s file with a
  180 Hz tone on at 2-5 s, 8-9.5 s and 13-18 s gave comb 3 peaks, 12.125 s
  above threshold, longest 5.875 s, peaks at 2.12-6.00, 8.12-10.50 and
  13.12-19.00 s. Each peak runs about 0.9 s past its burst, because a 1 s
  window still holds tone after the tone stops. The CNN gave 0 peaks. A
  noise file gave 0 for both. A click on the red file opened one window;
  a click on the noise file opened none. Made-up audio: no detection
  performance (section 5).
- `uv run pytest`: all pass.
- Not checked in a live window: real mouse clicks.

**Next.** None. A tool change.

## 082 - Detection window: play the file from its stats window

**Why.** Mads asked to be able to play the audio file from the stats
window (entry 081).

**What.** On top of commit 3545b69, uncommitted. `scripts/detect_gui.py`:
the stats window has Play and Stop sound buttons. Play plays that
window's file, whichever file the main window shows; `play()` takes the
file index. The buttons are kept on the window object so they keep
working after `open_stats` returns.

**Checks.** Headless (Agg backend, `winsound` stubbed to record what it
is sent): after opening the stats window for file 1 of 2 and then selecting
file 2 in the main window, Play sent 320000 samples at 16 kHz, the length
of file 1 (20 s); Stop sent `PlaySound(None)`. Not checked: real sound
output and real mouse clicks.

**Next.** None. A tool change.

Note on numbering (2026-11-03): entries 083 to 090 below were first
written, uncommitted, as entries 080 to 087. A `git pull` that brought in
commit 0576e0b (Nicolai's detection window, entries 080 to 082 above)
discarded the uncommitted copy of this file; nothing was stashed or left in
git. The text was restored from the session that wrote it and renumbered
after 082, with every cross-reference between them renumbered to match.
The content is otherwise unchanged.

## 083 - Listening file and spectrogram: 200 km/h drone in war-shaped noise at -70 dB

**Why.** Mads asked for a spectrogram and a WAV for 200 km/h at -70 dB.

**What.** As in entry 076, the mix comes from the virtual test with
`--keep`, so it is what the detectors scored, unscaled (1.0 = 110 dB SPL).
Commit 3545b69.
```
uv run python scripts/drone_war_grid.py --speeds 1 --speed-min 200 --speed-max 200 --wars 1 --levels 1 --war-min-db -70 --war-max-db -70 --keep --out artifacts/drone_war/listen200
```
Files in `artifacts/drone_war/listen200/mixes/`: `war-70dB_200kmh_off0.wav`
(20 s at 44.1 kHz, sidecar `.json`) and
`war-70dB_200kmh_off0_spectrogram.png` (4096-point FFT, 75% overlap,
0-8000 Hz, first drone sound and impact marked). The spectrogram was drawn
by a scratch script (matplotlib `specgram`), not a repo script.

Seed-1 drone, first sound at 1.62 s, impact at 10 s. Levels: whole file
-56.1 dBFS RMS, peak -22.8 dBFS; the noise alone (11-20 s) -96.9 dBFS;
the drone part (1.62-10 s) -52.3 dBFS; the last second before impact
-43.1 dBFS. So the noise sits about 45 dB under the drone: in effect the
drone alone, as at -80 dB in entry 076. The spectrogram shows the
fundamental near 250-330 Hz and its harmonics rising towards impact.

**Result.** Synthetic, one clip, 60 drone windows and 78 no-drone
windows, with 15 left out at the boundaries:

| War | CNN recall | CNN false pos. | `comb` recall | `comb` false pos. |
|---|---|---|---|---|
| shaped -70 dB | 100.0% | 0.0% | 86.7% | 3.8% |

Both first alarm at 2.1 s, 7.9 s before impact. Unlike 150 km/h at
-80 dB (entry 076, CNN recall 28.6%), the CNN catches every window here.
One clip each: not a speed effect until more seeds say so.

**Next.** None. These are listening files.

## 084 - Log-mel spectrogram of the 200 km/h, -70 dB listening file

**Why.** Mads asked for the spectrogram of entry 083 as a log-mel
spectrogram.

**What.** Commit 3545b69, plus the uncommitted entry 083. The log-mel is
the project frontend itself: `dad.wavio.read_wav` (resampled to 16 kHz)
then `dad.features.LogMel`, i.e. 64 mel bands 50-8000 Hz, 25 ms frames,
10 ms stride, the frozen contract (`uv run dad info`). The natural-log
power is shown in dB (times 10 / ln 10), relative to full scale, unscaled
input. The y axis is the 64 bands, labelled in Hz. Drawn by a scratch
script (matplotlib `imshow`), not a repo script. File:
`artifacts/drone_war/listen200/mixes/war-70dB_200kmh_off0_logmel.png`.

**Checks.** 64 x 1998 frames covering 0.01-19.98 s; band centres 79 to
7678 Hz; values -100.0 dB (the frontend's 1e-10 floor) to -2.7 dB (at
impact). The picture matches the linear spectrogram of entry 083: the
drone's harmonics from 1.62 s, rising and growing towards impact at 10 s,
and only faint noise after it.

**Next.** None. A listening and viewing aid.

## 085 - Listening file: 200 km/h drone in war-shaped noise at -20 dB

**Why.** Mads asked for a WAV for 200 km/h at -20 dB.

**What.** As in entry 083, from the virtual test with `--keep`, unscaled
(1.0 = 110 dB SPL). Commit 3545b69, plus the uncommitted entries 083-084.
```
uv run python scripts/drone_war_grid.py --speeds 1 --speed-min 200 --speed-max 200 --wars 1 --levels 1 --war-min-db -20 --war-max-db -20 --keep --out artifacts/drone_war/listen200m20
```
File: `artifacts/drone_war/listen200m20/mixes/war-20dB_200kmh_off0.wav`
(20 s at 44.1 kHz, sidecar `.json`). Same seed-1 drone as entry 083:
first sound at 1.62 s, impact at 10 s.

Levels: whole file -46.3 dBFS RMS, peak -23.0 dBFS; the noise alone
(11-20 s) -46.9 dBFS; drone plus noise (1.62-10 s) -45.7 dBFS; the last
second before impact -41.6 dBFS. Taking the drone's own level from entry
083 (-52.3 dBFS over 1.62-10 s, -43.1 dBFS in the last second, where the
noise was 45 dB lower), the drone sits about 5 dB under the noise on
average and about 4 dB over it in the last second.

**Result.** Synthetic, one clip, 60 drone windows and 78 no-drone
windows, with 15 left out at the boundaries:

| War | CNN recall | CNN false pos. | `comb` recall | `comb` false pos. |
|---|---|---|---|---|
| shaped -20 dB | 100.0% | **100.0%** | 3.3% | 3.8% |

The CNN alarms on every window, drone or not, as at -40 dB in entry 076:
it reacts to the war-shaped noise itself (entry 075). `comb` finds the
drone in 2 of 60 windows, first at 5.9 s (4.1 s before impact).

**Next.** None. A listening file.

## 086 - Supervisor feedback and council: synthetic training, CRNN, linear STFT
- Date: 2026-11-03
- Iteration: between 8 and 9
- Type: Council
- Commit: 3545b69, plus the uncommitted entries 083-085

**What changed.** Nothing in the code. Mads passed on written feedback
from the supervisor, with a drone-in-war detail plot (150 and 200 km/h,
war-shaped noise at -70 dB, seed 1000). The council was convened because
the feedback proposes a new model family, a new frontend and a new source
of training data (CLAUDE.md section 4).

The feedback, in short:
1. Train on synthetic data and use real data only for validation. Reasons
   given: the real data is unreliable, mostly DJI-type drones flying slower
   than an FPV, and its labels are inconsistent.
2. Drone synth with a random fundamental per example (100 Hz to 600-700 Hz)
   and integer harmonics, plus a random harmonic falloff rate, so the model
   learns the relations between harmonics, not fixed frequencies.
3. Labelled random soundscapes (drone, light_wind, artillery_explosion,
   small_gun_fire, truck_engine, helicopter_close, shouting_close, ...):
   30-50% of examples with confusers (helicopters, trucks, cars,
   motorcycles, delivery scooters), 20-30% without a drone.
4. A CRNN (3-4 2D conv layers, then a recurrent layer), so loud transients
   that mask whole frames can be bridged in time.
5. A linear STFT spectrogram (or log-mel), 16 kHz, window 64-128 ms
   (1024-2048 samples), hop 160-320 samples, log amplitude compression.
6. Drone-to-noise ratio in training about -20 to +10 dB, never below -40 dB.

**Question to the council.** Should the project (A) train a CRNN on fully
synthetic data and use real data only to validate and test, (B) adopt a
modified or partial version, or (C) keep the current approach (model 6:
DS-CNN, 1 s of 64 log-mel bands, real recordings pooled across rigs with
speed perturbation and mixing into real backgrounds)?

**Facts checked in the repo before weighing.**
- `REQUIREMENTS.md` has "Only real recordings, no synthetic training data"
  as LOCKED (Mads, 2026-10-10). Option A, and any use of synthetic drones in
  training, needs Mads to change it.
- The Pd drone (`src/dad/pdsynth.py`) has its fundamental at
  (150 + 180 u) x pscale Hz, about 150-330 Hz, from four detuned sawtooths,
  so the ratio between harmonics is fixed. The supervisor's point 2 applies
  to our test drone.
- `comb` (`src/dad/gate.py`) searches fundamentals from 35 to 330 Hz and
  scores only a drone band of 100-300 Hz. In the supervisor's plot both
  `comb` traces drop at about 9.5 s, when the drone's dominant frequency
  rises from about 240 Hz to about 320 Hz, just before impact. One drone
  seed; the drop matches the band edge, but no other cause was ruled out.
- The frontend's 25 ms frames give 40 Hz bins; `gate.py` itself notes that
  a 100-300 Hz harmonic comb cannot be resolved at that resolution. The
  supervisor's 64-128 ms window gives 8-16 Hz bins.
- A CRNN already exists (`src/dad/models/zoo.py`, `Crnn`, 739,747
  parameters) but has never been trained here: its GRU fails on DirectML
  (entry 009's timing table), and the model screens of entries
  016-019 ran it only conditionally and did not reach it.
- Labelled confuser stems already exist: `scripts/build_stems.py` (75 stems
  with source and licence, entry 052) and `scripts/distracters.py` /
  `scripts/env_sim.py` (car, truck, motorcycle, scooter, helicopter, mower,
  gunfire, explosions, shouting, wind). They are used only in the
  synthetic test environments, not in training.
- The dominant spectral peak (Welch, 4096 points, 80-4000 Hz, first 20 s)
  of every drone recording in `data/processed`, by a scratch script, not a
  repo script:

  | Source | Recordings | Peak p10 / p50 / p90 (Hz) |
  |---|---|---|
  | DroneNoise | 164 | 87 / 160 / 543 |
  | Halmstad | 30 | 109 / 270 / 932 |
  | Nightjar | 24 | 87 / 111 / 925 |
  | DDL | 36 | 82 / 107 / 953 |
  | d3set | 21 | 94 / 246 / 270 |
  | FPV campaigns | 124 | 82 / 109 / 219 |
  | Lund | 13 | 82 / 82 / 82 |

  Many peaks sit at the 80 Hz floor, which is likely wind or background,
  not a rotor, so this is a weak check. It does show drone peaks up to
  about 950 Hz, above the supervisor's 700 Hz, and it does not support the
  Devil's Advocate's guess that FPV tones sit at 1-2 kHz.

**Devil's Advocate.** Against A and most of B.
- Decisive: (1) A breaks the LOCKED rule. **Stands**; only Mads can change
  it. (2) If drones are synthetic and confusers real, "synthetic" becomes
  the shortcut for "drone": iteration 1's confound by construction.
  **Stands for A**; for a hybrid it is answered only by mixing real and
  synthetic positives and by checking that a classifier cannot tell
  synthetic from real positives. (3) The measured gap between the Pd drone
  and real drones is large (model 6 finds 29-35% of the clean Pd drone,
  entry 075). **Stands**; nothing measures the gap in the other direction
  yet.
- Serious: (4) The premise is partly contradicted: three FPV campaigns are
  in the pool and FPV on an unseen campaign is the best result (0.66,
  iteration 5); label inconsistency is asserted, not shown. **Partly
  answered**: the label claim should be audited, not assumed. (5) Synthetic
  knobs tuned against a few real validation rigs overfit them. **Answered
  by design**: arms and success rules are fixed in advance, test rigs are
  scored once. (6) 100-700 Hz may miss FPV tones. **Answered** by the peak
  table above, which suggests widening to about 1 kHz, not to 2 kHz.
  (7) Four changes at once. **Accepted**: one change per arm. (8) Score
  smoothing is a cheaper answer to masked frames than a CRNN (iteration 4:
  +0.07 from averaging 8 windows). **Accepted** as the baseline any CRNN
  must beat.

**Scholar** (no project context). The best-supported practice for UAV
detection trains on real recordings, with general-audio pretraining as the
largest lever and microphone transfer-function simulation as the largest
augmentation lever, and scores at a fixed false-positive rate on held-out
out-of-domain data (Reuter, Ohlenbusch and Rollwage, arXiv 2605.31329:
out-of-domain TPR at 1% FPR 0.701 without and 0.783 with AudioSet
pretraining, 0.816 with transfer-function simulation; no synthetic
drones). Sound event detection (DCASE Task 4) mixes synthetic soundscapes
with real data but never trains on synthetic alone; synthetic accuracy is
a weak proxy for real (Wei et al., arXiv 2609.22830, abstract only).
Placement: real only is standard but incomplete; synthetic only is against
practice for a final model; a hybrid is defensible (Al-Emadi et al.,
Sensors 2021: GAN drone clips added to real data raised recall on an
unseen drone by 1-38%); CRNN is the DCASE standard for temporal detection,
while a CNN on 1 s is standard for presence; a linear STFT at 64-128 ms is
a defensible variation (DCASE baselines use a 128 ms window), at the cost
of losing log-mel pretrained weights. Unsettled: whether synthetic drones
help once pretraining is in place, and CNN versus CRNN under domain shift.
Only the Reuter paper was read in full; the others from abstracts or
excerpts.

**Problem Solver.** Try a modified version (B). The measured failures are
real ones (mower-only false positives 55%, Lund recall 0.50, Halmstad
0.55, entry 043), and real confuser negatives are the lock-compliant lever.
Proposed arms, all DS-CNN, 1 s, frozen log-mel, iteration-5 speed
perturbation, three seeds, threshold frozen at 5% false alarms on
validation, tested on Lund, Halmstad and the mower-only session, with
`comb`, `lgbm` and model 6 alongside:

| Arm | Drone positives | Negatives | Needs the lock changed |
|---|---|---|---|
| R0 | model 6's recipe | as model 6 | no |
| R1 | real, drone-to-noise -20 to +10 dB | 30-50% real confusers, about 25% of examples without a drone | no |
| S | randomised synth only | as R1 | yes |
| R+S | real plus synth | as R1 | yes |

Success rules proposed in advance: R1 is kept if mower and helicopter
false positives drop by at least 15 points with Lund and Halmstad recall
down by at most 0.03; R+S is kept if it adds at least 0.05 recall on both
over R1 with no more false positives; S has failed if Lund recall is below
0.25 or its test AUC is at most 0.6. Then, one at a time on the winner: a
64 ms STFT frame, then a CRNN at a 2 s window against score smoothing.
Confidence medium.

**Where the members disagree.** The Scholar and the Problem Solver agree
that synthetic-only training is the riskiest step and a hybrid the
defensible one. The Scholar's two largest levers (general-audio
pretraining, transfer-function simulation) are not in the Problem
Solver's plan; pretraining was put off until after the field test in
entry 054.

**Decision (Claude).**
1. Option A is not taken. It breaks a LOCKED requirement, which only Mads
   can change, and it is against practice for a final model.
2. The lock-compliant part of the feedback is adopted as the next
   iteration: arm R1 against R0 (real confusers from the existing stems and
   FSD50K classes, 30-50% of examples with a confuser, about 25% without a
   drone, drone-to-noise -20 to +10 dB). It changes one thing, the training
   mix, and targets the measured mower false alarms.
3. Mads is asked whether to change the lock in a scoped way: "synthetic
   drone audio allowed in training arms, judged only on real held-out
   rigs". If yes, arms S and R+S run with a new randomised harmonic synth
   (random fundamental about 80-1000 Hz, random falloff, rotor detuning and
   amplitude modulation; not the Pd drone), and with a check that real and
   synthetic positives cannot be told apart.
4. After that, one at a time: a 64 ms frame, then a CRNN against score
   smoothing. `comb`'s 100-300 Hz drone band is a separate finding that
   needs widening for fast drones; a baseline change, logged when made.
5. The supervisor's points about labels and slow DJI drones go into a label
   audit (the worst validation errors, listened to) rather than being
   assumed.

**Next.** Report to Mads and ask about the lock. R1 can start without it.

## 087 - Requirements: synthetic training allowed; the synthetic drone sampler
- Date: 2026-11-03
- Iteration: 9 (preparation)
- Type: Requirements, Data
- Commit: 3545b69, plus the uncommitted entries 083-086

**What changed.**
1. **Requirements, decided by Mads.** Mads's reply to entry 086: in spot
   checks of the data uploaded to Edge Impulse they found drone-labelled
   clips with almost no sound, so they want to train on synthetic data only
   and test on the real FPV drone clips, which sounded good when they
   listened. Synthetic fly-bys (a drone passing, not only flying at the
   microphone) are also to be included. In `REQUIREMENTS.md` the LOCKED row
   "Only real recordings, no synthetic training data" is now WITHDRAWN, and a
   new row allows synthetic training with real drone recordings for
   validation and test, the FPV clips as test. Entry 086's decision 1
   (option A not taken) is overruled by this; the Devil's Advocate's
   decisive objections 2 and 3 there (synthetic positives as a shortcut,
   the measured synthetic-to-real gap) still stand and are what iteration 9
   must test.
2. **Mads's further instruction:** synthetic data is not to be generated,
   and models not trained, on this PC; that runs on JupyterLab resources.
   Nothing was generated or trained here beyond the smoke check below.
3. **New `src/dad/dronesynth.py`.** `drone()` draws one synthetic drone
   per window: fundamental log-uniform 80-900 Hz, harmonic falloff
   k ** -alpha with alpha 0.3-2.5 and +-3 dB per harmonic, 1-4 detuned and
   wandering rotors, 2-3 blades with amplitude modulation at the rotation
   rate, band-passed turbulence -30 to -5 dB, approach or fly-by at
   0-60 m/s (Doppler and 1/r from the geometry), an air low-pass at
   2-8 kHz. `SynthSampler` mixes it into a soundscape: a bed (a real
   no-drone window from given training rigs, or a background or wind stem),
   with p = 0.4 one distracter or war stem at -10 to +15 dB against the
   bed, then the drone at -20 to +10 dB in-band SNR; 75% of windows hold a
   drone. It has the interface of `RangeSampler`, so `NetScorer` and
   `LgbmScorer` train on it unchanged. The 80-900 Hz range widens the
   supervisor's 100-700 Hz to the drone peaks measured in entry 086.

**Data observation.** The supervisor's and Mads's point about near-empty
drone clips: Nightjar, both drone and background, sits at about -84 dBFS
RMS (manifest median -76.7 dBFS drone, -80.6 background; 10th percentile
-84.2 and -86.4), close to digital silence before level normalisation. That
is a likely source of the "empty" clips seen in Edge Impulse, which is not
verified: the Edge Impulse upload was not traced clip by clip. Quiet is not
the same as empty; iteration 3 found the drone at the edge of audibility
at 100-200 m. Nightjar windows can be selected by range.

**How.**
```
uv run pytest tests/test_dronesynth.py
```
Smoke check by a scratch script (not a repo script): 50 drones at 5.2 ms
each; one batch of 64 windows from `SynthSampler([])` (stems only, no rig
beds) in 0.37 s, shape (64, 64, 98), all finite, 49 of 64 with a drone.
Spectrograms of eight drawn drones show the intended variety: fundamentals
226-804 Hz, steep and flat falloffs, Doppler sweeps in fly-bys, slow
wander.

**Result.** No detection numbers: nothing trained.

**Next.** Set up the JupyterLab run: what the server has (GPU, storage,
network), how the repo and the data get there, and a notebook or script
that trains on `SynthSampler` and scores real validation and the FPV test.
Iteration 9's box (split, hypothesis, judging rule) is written when the
runner is fixed.

## 088 - Listening files: twelve synthetic drones from `dronesynth`

**Why.** Mads asked to hear some of the new synthetic drones.

**What.** Scratch scripts (not repo scripts) on top of entry 087's
uncommitted `src/dad/dronesynth.py`. Twelve clips of 5 s at 16 kHz, each as
`_drone_only.wav` and `_mix.wav`, in `artifacts/synth_listen/`
(git-ignored), parameters in `clips.csv` for clips 1-8. The beds are stems
only (no rig audio); clips 2, 4, 6 and 8 add one confuser stem; the SNR is
drawn from -10 to +10 dB, the audible end of the training range. Clips 9-12
were drawn until the fundamental was at least 400 Hz, because clips 1-8
all landed at 94-248 Hz by chance (1000 draws: 10th, 50th and 90th
percentile 101, 258 and 717 Hz, as intended).

The synth has no physical calibration; its level is set only by the SNR
against the bed (stems at -26 dBFS active RMS, entry 052). The files are
written unscaled at exactly that level: the drone-only file holds the
drone at the gain it has in the mix. Mix levels -27.5 to -13.7 dBFS RMS;
clip 6 (confuser +10 dB) peaks at 1.30 and can clip in a player.

**Next.** None. Listening files; nothing trained, and generation for
training stays on JupyterLab (entry 087).

## 089 - The training drone is the Pure Data drone, not a new harmonic synth
- Date: 2026-11-03
- Iteration: 9 (preparation)
- Type: Data, Correction
- Commit: 3545b69, plus the uncommitted entries 083-088

**What changed.** Mads: the training drones are to come from the realistic
drone generator already built in Pure Data. Entry 087's harmonic
`drone()` is replaced (not kept as an option); `SynthSampler` is unchanged
around it.
- `src/dad/dronesynth.py`: `drone()` now renders `DroneSound.pd` through
  its numpy port `dad.pdsynth.render` at 44.1 kHz (intended mapping,
  `PATCH_DOUBLE_DIVIDE = False`, as entry 062's port), resamples to 16 kHz,
  and propagates it with `dad.propagation`: 1/r, Doppler and ISO 9613-1
  air absorption, straight at the microphone (approach, ending 10-400 m
  away) or past it (`pass_by`, closest approach 5-200 m), 0-60 m/s. Two
  seconds are emitted and the window is cut after the first emitted
  sample has arrived, so it has no silent lead-in.
- Randomised per window: the patch's `pscale` log-uniform 0.4-3.0
  (`make_scene` used 0.85-1.25), so the motor fundamental
  (150 + 180 u) x pscale spans about 80-900 Hz, the supervisor's point of
  entry 086; the throttle by render.py's model, cruise 30-110 of 127,
  wobble 2-10, a dive ramp in 30% of flights. The harmonic falloff is the
  patch's own (sawtooth, `lop~`, `hip~`, noise wash, all following
  throttle); it is not randomised separately, unlike the supervisor's
  suggestion, because that would change the patch.
- `make_throttle` moved into `dad.pdsynth`; `scripts/stress_test.py` now
  uses it (`st.make_throttle` still works for `env_grid.py`, `scenes.py`,
  `stress_gui.py`). `Pure-data+Script/EnviromentTest/render.py` keeps its
  own copy, as before.

**How.**
```
uv run pytest
```
**Result.** All tests pass, including `tests/test_dronesynth.py` (rewritten:
finite, unit RMS, no silent lead-in). Cost on this PC, one core: 26 ms per
drone, 2.8 s per batch of 64 windows (0.37 s with entry 087's synth), so
200,000 training windows take about 2.4 h of one core for the data alone;
`NetScorer.fit` prefetches with worker threads. Listening files (scratch
script, as entry 088): ten 5 s clips in `artifacts/synth_listen_pd/`,
`_drone_only` and `_mix`, unscaled, stems-only beds, confuser in the even
clips, SNR -10 to +10 dB, parameters in `clips.csv`. Cruise fundamentals
143-507 Hz, speeds 5-54 m/s, four with a dive. Clips 4 and 10 peak above
1.0 (1.39, 1.84) and can clip in a player. The clips in
`artifacts/synth_listen/` (entry 088) are from the replaced synth.

**Next.** As entry 087: the JupyterLab runner.

# Iteration 9 - MFE, MFCC or a linear spectrogram, on real data

> **Changed from iteration 5 (the last iteration that trained the repo's
> own models on held-out units):** only the frontend, what the CNN sees.
> Three arms: `mfe`, 64 log-mel bands, 25 ms frames, 10 ms stride (model 6's
> frontend, the reference); `mfcc`, the first 20 MFCC of that log-mel; and
> `linear`, the log power of every STFT bin 50-8000 Hz from a 64 ms frame
> (15.6 Hz bins, 509 rows), 10 ms stride. The linear arm changes both the
> frequency scale and the frame length, as the supervisor proposed; a
> linear arm at 25 ms would have 40 Hz bins and could not resolve the
> harmonics, which is the reason to try it.
> **Why:** the supervisor's feedback (entry 086) that a linear spectrogram
> keeps the harmonic spacing and resolves it at 64-128 ms, the frontend
> note in `gate.py` that 25 ms frames cannot resolve a 100-300 Hz comb, and
> Mads's decision (2026-11-03) to drop synthetic training (entries 087-089)
> and test the frontends on real data, trained on JupyterLab.
> **Kept the same:** iteration 5's best recipe: pooled rigs, leave one unit
> out (Nightjar, d3set, FPV `ariel_25.02`, `palmahim_10.02`,
> `palmahim_27.05`), `simdist`, speed perturbation 0.8-1.25, DS-CNN, 1 s
> window, 0.5 s evaluation hop, 200,000 training windows, seeds 0-2. One
> known difference: Glasgow trained in iteration 5 but is not in this
> machine's processed store, so the `mfe` arm, not iteration 5's figures,
> is the reference. Baselines `comb` and `lgbm` (log-mel) run in the `mfe`
> arm only. The deployment threshold is frozen at 5% false alarms on
> validation (R3), as in iteration 6.
> **Hypothesis:** the linear arm separates drone harmonics from other
> motors better, so it raises recall on unseen FPV campaigns and lowers
> mower false alarms; MFCC does no better than MFE, because the DCT spreads
> a local harmonic pattern over all coefficients.
> **Judged by** (seed means, decided before any run): the primary figure is
> the FPV mean, recall on all drone windows at 10% false alarms on the
> held-out campaign's own background (iteration 5: 0.655 for MFE with
> Glasgow). A frontend beats `mfe` if its FPV mean is at least 5 points
> higher, all three of its seeds lie above every `mfe` seed, no held-out
> unit loses more than 5 points of recall (all windows, 10%), and its
> mower-only false alarms at the deployment threshold are not more than 5
> points higher. Also reported, without a rule: recall and false alarms at
> the deployment threshold (R2, R3), d3set at 100-200 m, Lund recall,
> `comb` and `lgbm`. Adopting a new frontend for the field model needs the
> council (CLAUDE.md section 4) once the numbers are in.

## 090 - A selectable frontend and the JupyterLab notebook
- Date: 2026-11-03
- Iteration: 9
- Type: Preprocessing, Infrastructure
- Commit: 3545b69, plus the uncommitted entries 083-089

**What changed.**
- Mads: "glem det syntetiske" (forget the synthetic). Train again on real
  data, on JupyterLab, to test whether MFE, MFCC or a linear spectrogram is
  best. Entries 087-089's synthetic plan is dropped. `src/dad/dronesynth.py`
  and its test stay in the tree, unused (not committed, so deleting them
  would lose them); `make_throttle` stays in `dad.pdsynth` (entry 089).
  The requirements row allowing synthetic training (entry 087) is left as
  it is: allowed, not used.
- `FrontendConfig` has `kind` ("mfe", the default, "mfcc", "linear") and
  `n_mfcc` (20), and `n_bands`, the rows the network gets. `features.py`:
  `band_matrix` is the mel filterbank, or for "linear" the identity rows of
  the bins in fmin-fmax; `LogMel` and `StreamingLogMel` use it.
  `realrun.to_model` applies the orthonormal DCT-II for "mfcc" after level
  normalisation and the training EQ (both act on log-mel bands); `featurise`
  and `RangeSampler.batch` end with it. `NetScorer` builds the network for
  `n_bands`, scales the SpecAugment frequency mask to the same share of
  bands (8 of 64: 63 linear bins, 2 MFCC), and saves the frontend in the
  checkpoint; `load` restores it, so `dad wav` scores a checkpoint with the
  frontend it was trained on. Checkpoints without it (model 6) load as
  log-mel, as before.
- `scripts/run_range.py`: `--frontend`, `--n-fft` (default 400, or 1024 for
  linear) and `--n-mfcc`.
- New `notebooks/iteration9_frontends.ipynb`: install, data check, FSD50K
  extraction and conversion on the server (7-Zip is not on this PC; FSD50K
  is only here as the raw multi-part zip on Google Drive), tests, the nine
  runs (3 frontends x 3 seeds, each over the 5 held-out units, resumable),
  and the summary table (`artifacts/runs/it9_summary.csv`).

**How.**
```
uv run pytest
```
New `tests/test_frontends.py`: the linear frontend keeps 509 bins and puts
a 1 kHz tone in bin 64; MFCC coefficient 0 is the scaled sum of the
level-normalised log-mel; a linear checkpoint loads back as linear and
scores. All tests pass. The notebook's code cells compile; the notebook
has not been run.

The DS-CNN has 22,530 parameters for every frontend (global pooling). One
CPU forward pass of 64 windows on this PC: MFE 64 x 98, 146 ms; MFCC
20 x 98, 36 ms; linear 509 x 94, 1,250 ms. The linear arm costs about 8.5
times the network compute of MFE.

**Result.** Nothing trained (Mads: training runs on JupyterLab, not on
this PC).

**Next.** Mads to say how the repo and the data reach JupyterLab and
what GPU it has; then run the notebook and log the results as entry 091.

## 091 - The JupyterLab server: Pascal GPUs need a CUDA 12.6 torch

**Type.** Setup. **Date.** 2026-11-03. **Commit.** 242b3b4 plus the
uncommitted notebook change below.

**What.** The training server for iteration 9 (JupyterLab at the
university): 3 NVIDIA TITAN X (Pascal, sm_61, 12 GB), driver 580, 24
cores, 125 GB RAM, Python 3.12.6.
- `uv sync --group dev --extra dl` installs torch 2.14.1+cu130 from
  `uv.lock`. Its kernels start at sm_75, so it cannot run on these GPUs
  (`cudaErrorNoKernelImageForDevice`). On the server only, torch was
  replaced by the CUDA 12.6 build of the same version (2.14.1+cu126,
  kernels sm_50-sm_90), which runs on all three GPUs. `uv.lock` is
  unchanged; on the server the runs call `.venv/bin/python` directly,
  because `uv run` would re-sync torch back to the lock.
- The notebook runs every step with `.venv/bin/python` when it exists,
  instead of `pip install` into the kernel's Python, and runs the three
  frontends at the same time, one per GPU (`CUDA_VISIBLE_DEVICES`), with
  the seeds of each one after another. Nothing in the runs themselves
  changed.
- FSD50K is downloaded raw from Zenodo (record 4060432) to
  `data/FSD50K/` on the server; the other processed stores are uploaded
  from this PC.

**How.**
```
uv pip install --python .venv/bin/python --reinstall-package torch torch --index-url https://download.pytorch.org/whl/cu126
.venv/bin/python -c "import torch; ..."   # torch 2.14.1+cu126, sm_50..sm_90, 3 x TITAN X, a CUDA op runs
.venv/bin/python -m pytest -q -p no:cacheprovider   # all pass on the server
```

**Result.** The server environment works. Nothing trained yet.

**Next.** The processed stores uploaded, FSD50K extracted and converted,
then the nine runs; results go in entry 092.

## 092 - The frontend runs batched on the GPU; the iteration 9 runs restarted

**Type.** Engineering (speed only; what the models see is unchanged).
**Date.** 2026-11-03. **Commit.** 242b3b4 plus the uncommitted changes
below.

**Why.** The first iteration 9 runs on the server (entry 091) trained at
102-145 windows/s (MFE, MFCC) and under 17 windows/s (linear, which had not
reached 50,000 windows after about 55 minutes), against 571 windows/s for
model 6 on this PC (entry `it6_final`, 350.6 s for 200,000 windows). The
GPUs sat at 0-5%. The server's CPUs are two Xeon E5-2620 v3 (12 cores at
2.4 GHz, 2014), and the training windows were built one at a time in numpy
on three Python threads per run. Mads asked for the window building to use
the GPU instead of overloading the CPU.

Profile of `RangeSampler.batch` (MFE, simdist, speed 0.8-1.25, 10 batches
of 64, one thread, server under load; cProfile):

| Step | Share of the time |
|---|---|
| speed change (`resample_poly`) | 46% (of which the filter design, redone for every window, about half) |
| in-band power (`sosfilt` high-pass) | 22% |
| STFT and bands (`LogMel.frames`) | 17% |
| level normalisation and the rest | 15% |

**What.**
- `rangerun.speed_filter`: the low-pass filter `resample_poly` designs
  for itself, designed once per speed factor and passed in. The output is
  bit-identical (`tests/test_gpufront.py`).
- `RangeSampler.raw_batch`: the CPU part of a batch (drawing, mixing at the
  drawn SNR, the speed factor, the EQ curve), with the random numbers drawn
  in the same order as before; `batch` now calls it and does the rest in
  numpy as before.
- New `dad/gpufront.py`, `GpuFrontend`: the speed change (scipy's
  polyphase filter, the same design and offset, as a gather), the STFT
  (windowed in float32, FFT in float64, as numpy), bands, log, EQ, level
  normalisation and the MFCC DCT for a whole batch on the GPU.
- `NetScorer` uses it on CUDA for training, the warm-up statistics and
  scoring; on CPU and DirectML the numpy path is used as before.
- `train.spec_augment` builds its masks for the whole batch at once, with
  the same random draws.

**How.**
```
uv run pytest                               # this PC: all pass, 10 new in tests/test_gpufront.py
uv run python <scratch>/diff.py             # GPU frontend on CPU torch vs numpy
.venv/bin/python it9_golden.py ...          # server, old code: batches 3-4 of seed 0, hold out Nightjar, per frontend
.venv/bin/python it9_check.py ... 3         # server, new code: same batches, speed, NetScorer.fit 25,600 windows
```
The scratch scripts are not in the repo.

**Result.** Same features:
- GPU frontend vs numpy on this PC (CPU torch): speed change max
  difference 0.0; features max 1e-6 (MFE, linear) and 1.1e-5 (MFCC) in
  log units.
- On the server against the old code's own output (real batches): the new
  numpy path is bit-identical for linear and differs in the last bits for
  MFE and MFCC (the reference was made with 2 BLAS threads, so the mel
  matrix product summed in another order); the GPU path is within 8.4e-5
  of the old training features and 6.5e-5 of the old evaluation features
  (log units; 1e-4 is 0.0004 dB). The labels are identical. The GPU
  numbers for linear were measured before the float32 windowing fix
  above, which brought linear from 3e-5 to 1e-6 on this PC.

Speed on the server, measured while the three old runs still loaded it:

| Frontend | One batch of 64, one thread: old numpy path | New: CPU part + GPU part | `NetScorer.fit`, 3 threads | Old runs |
|---|---|---|---|---|
| MFE | 644 ms | 74 + 13 ms | 571 windows/s | 102-145 windows/s |
| MFCC | 642 ms | 87 + 24 ms | 659 windows/s | 112-145 windows/s |
| linear | 9,389 ms | 60 + 15 ms | 191 windows/s | under 17 windows/s |

The linear network itself has 509 input rows, so its GPU part stays the
largest.

**Decision.** The partial runs with the old code were stopped (seed 0
only; nothing finished) and moved to `artifacts/runs/it9_old_code/` and
`artifacts/models/it9_old_code/` on the server, not deleted. All nine runs
were restarted with the new code, so every iteration 9 number comes from
one code version. The recipe, data, seeds and judging rule of iteration 9
are unchanged.

**Next.** The results of the nine runs, entry 093.

## 093 - The iteration 9 runs share the GPUs through a queue

**Type.** Infrastructure (scheduling only).
**Date.** 2026-11-03. **Commit.** 242b3b4 plus the uncommitted changes of
entry 092.

**Why.** After entry 092 the linear runs are GPU-bound (about 190
windows/s, the GPU at 99%) and MFE and MFCC CPU-bound (about 1,000
windows/s or more). The notebook gave each frontend its own GPU and ran
its three seeds one after another, so GPUs 0 and 1 would sit idle once MFE
and MFCC finished while GPU 2 still had about four hours of linear runs
left.

**What.** `notebooks/iteration9_frontends.ipynb`, run cell: the nine runs
(three frontends, three seeds) go into one queue, linear first, and each
GPU takes the next run when it finishes one. The command of each run,
its log and its `.done` marker are unchanged. The partial runs started
under entry 092 (seed 0, nothing finished) were stopped and deleted; all
nine start from the beginning with the new scheduling.

**How.** On the server, `artifacts/runs/it9_runs.py` is generated from
notebook cells 1 and 10 and started with
`nohup .venv/bin/python -u artifacts/runs/it9_runs.py > artifacts/runs/it9_go.log`.

**Result.** `it9_linear_s0`, `s1` and `s2` started on GPUs 0, 1 and 2; one
minute later the GPUs were at 90-99% and about 4.4 GB each. The total run
time is not yet measured; the expectation is about 2-2.5 hours.

**Next.** The results of the nine runs, entry 094.

## 094 - Iteration 9 results: neither MFCC nor linear beats MFE

- Date: 2026-11-04
- Iteration: 9
- Type: Evaluation
- Commit: 242b3b4 plus the uncommitted changes of entries 092-093

**What changed.** Nothing; the nine runs of entry 093 finished and were
scored against the rule fixed in the iteration 9 box.

**Why.** The iteration's verdict.

**How.** On the JupyterLab server: the nine runs started 2026-11-03 17:14
and finished at 19:21 (linear 18:27, MFE 19:04, MFCC 19:21); no error in
any log. Summary and rule check:
```
python3 artifacts/runs/it9_judge.py > artifacts/runs/it9_judge.txt
```
`python3` is the server's Jupyter Python (pandas 3.0.1; the project venv
has no pandas). The script is notebook cell 12 plus the rule check; it is
not in the repo. Outputs on the server: `artifacts/runs/it9_summary.csv`
(75 rows: 5 scorers x 5 held-out units x 3 seeds), `it9_judge.txt`, and
the run files `it9_<frontend>_s<seed>.json`; not copied to this PC.

**Result.** Pooled rigs, leave one unit out, `simdist`, speed 0.8-1.25,
DS-CNN, 1 s window, 0.5 s evaluation hop, 200,000 training windows. Recall
on all drone windows of the held-out unit at 10% false alarms on that
unit's own no-drone audio, seeds 0 / 1 / 2 and mean. `comb` and `lgbm` are
log-mel baselines, trained in the MFE runs.

| Held out | comb | lgbm | MFE | MFCC | linear |
|---|---|---|---|---|---|
| FPV `ariel_25.02` | 0.198 | 0.373 | 0.686 | 0.377 | 0.714 |
| FPV `palmahim_10.02` | 0.078 | 0.763 | 0.809 | 0.708 | 0.807 |
| FPV `palmahim_27.05` | 0.083 | 0.436 | 0.512 | 0.369 | 0.555 |
| d3set | 0.153 | 0.206 | 0.550 | 0.256 | 0.748 |
| Nightjar | 0.201 | 0.132 | 0.242 | 0.132 | 0.407 |
| **FPV mean** | 0.120 | 0.524 | **0.669** (0.672 / 0.667 / 0.669) | 0.485 (0.500 / 0.483 / 0.471) | **0.692** (0.702 / 0.688 / 0.686) |

By range, same figure (FPV has no range):

| Held out | Range | comb | lgbm | MFE | MFCC | linear |
|---|---|---|---|---|---|---|
| d3set | 0-50 m | 0.204 | 0.212 | 0.612 | 0.268 | 0.845 |
| d3set | 100-200 m | 0.002 | 0.097 | **0.252** | 0.156 | 0.185 |
| Nightjar | 0-50 m | 0.209 | 0.197 | 0.389 | 0.211 | 0.644 |
| Nightjar | 100-200 m | 0.108 | 0.029 | 0.024 | 0.067 | 0.038 |

At the deployment threshold (frozen at 5% false alarms on validation),
mean over the five held-out units and three seeds:

| | comb | lgbm | MFE | MFCC | linear |
|---|---|---|---|---|---|
| Recall (R2, target 0.95) | 0.200 | 0.478 | 0.668 | 0.461 | 0.692 |
| False alarms on the held-out unit (R3, target 0.05) | 0.164 | 0.274 | 0.292 | 0.245 | 0.260 |
| False alarms on mower-only audio | 0.226 | 0.363 | 0.352 | 0.291 | 0.475 |
| Recall on Lund (never trained on) | 0.020 | 0.063 | 0.430 | 0.095 | 0.251 |

The held-out false alarms are dominated by Nightjar, where every learned scorer
alarms on 76-86% of the no-drone windows at the frozen threshold (comb 3%); on the
other units they range from 0.1% to 41%.

**Assessment against the rule** (iteration 9 box):

| Condition | MFCC | linear |
|---|---|---|
| FPV mean at least +5 points over MFE | -18.4: no | +2.3: no |
| All three seeds above every MFE seed | no | yes (0.686 > 0.672) |
| No held-out unit loses more than 5 points | no (`ariel_25.02` -30.9, d3set -29.4) | yes (worst `palmahim_10.02` -0.1) |
| Mower false alarms not more than 5 points higher | yes (-6.1) | no (+12.3) |
| **Verdict** | **does not beat MFE** | **does not beat MFE** |

- **MFE stays the frontend.** Its FPV mean, 0.669, is at iteration 5's
  0.655 without Glasgow, so the reference reproduces.
- **MFCC is worse everywhere** except the mower, where it alarms less
  because it detects less. The hypothesis that it does no better than MFE
  is confirmed, more strongly than stated.
- **Linear is a mixed result, and the opposite of its hypothesis on the
  confusers.** It gains a little on FPV (+2.3, consistent over seeds) and
  much near the rigs (+23 on d3set and +26 on Nightjar within 50 m), but
  loses 6.7 points on d3set at 100-200 m, 18 points of recall on Lund, and
  alarms 12 points more on the mower. The finer frequency resolution helps
  where the drone is loud, and it does not separate a drone from another
  motor better; it does so worse.
- Against REQUIREMENTS.md: no scorer is near R2 (0.95 recall) and R3 (5%
  false alarms) at one threshold; the best, linear and MFE, reach about
  0.67-0.69 recall at 26-29% false alarms on unseen units. Not met.
- The held-out units are development folds (entry 025's council), so
  these are development figures; no sealed test was scored.

**Next.** Iteration 9 is closed: MFE stays; no frontend change, so no
council is needed. Report to Mads. Not decided: whether the linear arm's
gain near the rigs is worth a follow-up (for example on the mower false
alarms) or whether the next iteration targets something else.

## 095 - The no-drone audio exported as WAV for a second JupyterLab account

- Date: 2026-11-04
- Iteration: none (data handling)
- Type: Data
- Commit: uncommitted

**What changed.** Every no-drone recording (manifest label 0) of
`data/processed`, plus the no-drone stems, written as WAV with a CSV and
copied to the JupyterLab account `jupyter-mariusq` (same server as entry
091), at `~/nodrone/`:

| Part | Files | Duration |
|---|---|---|
| `recordings/nightjar` (background) | 28 | 75.3 min |
| `recordings/fpv` (background) | 121 | 109.6 min |
| `recordings/d3set` (background) | 37 | 14.1 min |
| `recordings/halmstad` (background and helicopter) | 59 | 9.8 min |
| `stems/` background, distracter, war, wind (drone stems left out) | 97 | - |

`recordings.csv` holds the manifest rows (source, group, licence and so on)
with the WAV path; `stems.csv` the stems manifest rows; attribution in
`stems_ATTRIBUTION.md`. 499 MB.

**Why.** Mads asked for the no-drone data on the second account.

**How.** `uv run python -I scripts/export_nodrone.py . data/extracted/nodrone`.
Samples are written as stored: int16 stores as 16-bit WAV, 16 kHz mono,
never rescaled. Checked on this PC: all 245 recordings read back
bit-identical to `realdata.Recording.audio()`. Uploaded as a tar in 55
parts through JupyterLab's own upload (the browser tool takes 10 MB per
file), joined on the server; MD5 6da945827252f84e82ca8b6b415d69d3 on both
sides; 342 WAV files after unpacking.

**Result.** Not a result; data handling only.

**Not done.** FSD50K: its processed store exists only on the `mads`
account, which this browser can no longer open while it is logged in as
`mariusq` (HTTP 403), and the raw FSD50K archives are not on this PC.

**Next.** FSD50K once Mads is logged in as `mads` again (export there, copy
through a folder both accounts can read), if still wanted.

## 096 - Final CNN and CRNN trained on every rig, for the virtual tests

- Date: 2026-11-04
- Iteration: none (checkpoints, as iteration 6)
- Type: Model
- Commit: 87a7bdf plus the uncommitted `scripts/run_range.py` change below

**What changed.** Two models trained once on every rig, with nothing held
out, at Mads's request, so both can drive the virtual tests (the drone-in-war
GUI and the other synthetic test benches) and go into the repo:
- `final_dscnn`: the DS-CNN of model 6 (22,530 parameters).
- `final_crnn`: the CRNN of `models/zoo.py` (739,747 parameters; three
  conv blocks that pool frequency only, a bidirectional GRU, attention
  pooling over time). It had never been trained before (entry 006's backend
  had no GRU).
- `scripts/run_range.py`: `crnn` added to `NETS`, so the runner trains it.

Both use iteration 5/6's recipe and model 6's frontend: MFE, 64 log-mel
bands, 25/10 ms, 1 s window, `simdist`, speed perturbation 0.8-1.25,
200,000 training windows, seed 0, deployment threshold frozen at 5% false
alarms on validation. Glasgow is not in the server's store, so these are
model 6's recipe without Glasgow.

**Why.** Mads asked for a final CNN and CRNN to compare in the virtual tests,
without a frontend evaluation. No council was held for the CRNN: no model
choice is made here, and no claim about its performance.

**How.** On the JupyterLab server, one GPU each:
```
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u scripts/run_range.py --name final_dscnn --pool --arms none --recipes simdist --models dscnn --device cuda --windows 200000 --speed 0.8 1.25 --deploy-fa 0.05 --seed 0 --frontend mfe
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u scripts/run_range.py --name final_crnn --pool --arms none --recipes simdist --models crnn --device cuda --windows 200000 --speed 0.8 1.25 --deploy-fa 0.05 --seed 0 --frontend mfe
```
258 s (DS-CNN) and 280 s (CRNN) in total, 760-1,070 windows/s. Outputs,
copied to this PC with matching MD5:
`artifacts/models/final_dscnn_none_simdist_dscnn.pt`,
`artifacts/models/final_crnn_none_simdist_crnn.pt`, and
`artifacts/runs/final_{dscnn,crnn}.json` and `.log`. Both load locally
through `dad.wavio.load_scorer`; thresholds from the run files: DS-CNN
0.380, CRNN 0.483. `artifacts/` is git-ignored, so pushing them needs
`git add -f`.

**Result.** Not a test result. The run files hold only validation figures
(21 segments from the training rigs, in distribution) and Lund at the
deployment threshold:

| | DS-CNN | CRNN |
|---|---|---|
| Validation recall, all drone windows, at the deployment threshold (FA 5%) | 0.812 | 0.798 |
| Validation recall 0-50 m at 10% FA | 0.840 | 0.828 |
| Validation recall 100-200 m at 10% FA | 0.082 | 0.021 |
| Lund recall at the deployment threshold (rig never trained on) | 0.561 | 0.454 |

One seed and in-distribution validation: these numbers cannot rank the two
models. The leave-one-unit-out runs of iteration 9 would be needed for that.

**Next.** Mads runs the virtual tests with both. Synthetic numbers from
them are not detection performance (CLAUDE.md section 5).

## 097 - REPORT.md merges by union

- Date: 2026-11-04
- Iteration: none (repo tooling)
- Type: Infrastructure
- Commit: uncommitted

**What changed.** New `.gitattributes`: `docs/REPORT.md merge=union`. When
two branches both append entries, git keeps both instead of stopping with a
conflict.

**Why.** Several people now append to this report on their own branches
(for example iteration 10 on `it10-SynthDrone`), and every merge conflicted
at the end of the file.

**How.** Tested on a throwaway repo: two branches each appending an entry
merged without a conflict and with both entries kept.

**Result.** Known limits, seen in that test:
- Two branches can pick the same entry number; both are kept. Check the
  numbers after a merge and renumber in a correcting entry if needed.
- The blank line between the two appended blocks can be lost.
- A line both sides changed (a row of the glance table) ends up twice,
  without warning. Check the table after a merge.
- Applies to merges made with local git. Not verified for GitHub's web
  merge button.

**Next.** None. A tool change.

## 098 - Detect GUI: pick the network per batch

- Date: 2026-11-04
- Iteration: none (tooling)
- Type: Infrastructure
- Commit: 791d95e plus the uncommitted `scripts/detect_gui.py` change below

**What changed.** `scripts/detect_gui.py` gets radio buttons at the bottom
left that pick the network run next to comb, for the next batch opened:

| Name in the GUI | Checkpoint | Threshold (from its run file) |
|---|---|---|
| it6 CNN | `artifacts/models/it6_final_none_simdist_dscnn.pt` (model 6, the old default) | 0.297 |
| it9 CNN | `artifacts/models/final_dscnn_none_simdist_dscnn.pt` (entry 096) | 0.380 |
| it9 CRNN | `artifacts/models/final_crnn_none_simdist_crnn.pt` (entry 096) | 0.483 |
| it10 CNN | `artifacts/it10_clean/clean_s0_dscnn.pt` (seed 0 of commit 32142e1) | 5.7e-06 |

"it9" is Mads's name for the entry 096 checkpoints (MFE frontend, after
iteration 9); the report files them under no iteration. comb keeps model 6's
frozen threshold whatever is picked. The model's name replaces "CNN" in the
list, the graphs, the stats window and the `detector` column of
`artifacts/detect_gui/results.csv`. The list shows 18 files at a time
instead of 22, to make room.

**Why.** Mads asked to run his own files through each of the three new
models in the GUI.

**How.** Loaded headless (`MPLBACKEND=Agg`), set each model, and ran
`detect`, `redraw`, `write_csv` and `open_stats` on
`data/extracted/sealed_peek/Drone_01.wav`. All four load and score.

**Result.** A smoke test, not a detection result: one file, no sidecar.
it6 CNN alarmed on 97% of windows, it9 CNN 74%, it9 CRNN 71%, it10 CNN 0%
(its threshold of 5.7e-06 comes from a validation set it separated
perfectly, AUC 1.0, so it says little about real audio); comb 14%.

**Next.** None. A tool change.

## 099 - Drone-in-war GUI: pick the network per run

- Date: 2026-11-04
- Iteration: none (tooling)
- Type: Infrastructure
- Commit: 791d95e plus the uncommitted changes below and in entry 098

**What changed.**
- `scripts/drone_war_grid.py`: the model list of entry 098 moves here as
  `MODELS` (it6 CNN, it9 CNN, it9 CRNN, it10 CNN; same checkpoints and
  thresholds). `_detectors(model)` loads one of them next to comb, cached
  per model. New `--model` option (default `it6 CNN`, the old behaviour).
  `results.csv` and `hops.csv` get a first column `model`. The folder
  constant `MODELS` is renamed `MODEL_DIR` to free the name.
- `scripts/drone_war_gui.py`: horizontal radio buttons under "Flush plots"
  pick the network for the next Run. Its name heads the table column and
  appears in the grid's title ("CNN = ..."), the detail window's title and
  legend, and the saved PNGs' titles.
- `scripts/detect_gui.py`: uses `drone_war_grid.MODELS` instead of its own
  copy. No change in behaviour.

**Why.** Mads asked for the same model choice in the drone-in-war GUI as
in entry 098.

**How.**
```
uv run python scripts/drone_war_grid.py --model "<each of the four>" --speeds 1 --speed-min 100 --speed-max 100 --levels 1 --war-min-db -10 --war-max-db -10 --wars 1 --no-drone --out <scratch>
```
and the GUI loaded headless (`MPLBACKEND=Agg`): it9 CRNN picked, Run,
table, grid and detail window drawn. `detect_gui.py` rerun on
`Drone_01.wav` with all four models: same window shares as entry 098.

**Result.** All four models run in both GUIs and on the command line.
Synthetic scaffolding, not detection performance (CLAUDE.md section 5):
in this one 100 km/h, war -10 dB case every network alarmed on 99-100% of
war-only windows, and comb on 3.3%.

**Next.** None. A tool change.

# Iteration 11 - Steady noise as negatives, without war audio in training

> **Changed from iteration 9 / entry 096:** only the training negatives
> and maskers. A family of steady noises is added to training: (a) Gaussian
> noise with a random spectral slope (-12 to +3 dB/octave) and a random
> smooth EQ, (b) random-phase copies of the training rigs' own no-drone
> windows (same spectrum, no texture), each with a slow level wobble
> (0.2-2 Hz, +-3 dB). About 15% of negative windows are such noise alone,
> and about 30% of positive windows get one under the drone at -5 to +15 dB
> in-band SNR (the existing `buried` range). Whether random-phase copies of
> drone windows are added as hard negatives (c) is decided by step 0. No
> war audio (the 23 FSD50K war clips, the war track or war-shaped noise)
> and no noise stem goes into training.
> **Why:** entries 075-076 and 100: with no drone present, model 6 and both
> entry 096 networks alarm on 99-100% of windows of war-shaped noise at
> every level from -40 to -10 dB, and on 0% of the war track at the same
> levels. Pink noise gives 54-100%, brown 9-89%, white 0-3%. The mower
> session gave 55% for model 6. The failure is steady noise with
> low-frequency emphasis, not war. The council (entry 100) rejected
> training on war-shaped noise made from the war clips.
> **Kept the same:** MFE (64 log-mel, 25/10 ms), per-window level
> normalisation, DS-CNN, iteration 5's `simdist` recipe with speed
> perturbation 0.8-1.25, 200,000 windows, 1 s window, deployment threshold
> frozen at 5% false alarms on validation, the validation set itself.
> **Hypothesis:** the networks alarm on steady noise because no training
> negative is steady and broadband after level normalisation. Steady
> negatives remove that, and false alarms fall on steady noise the model
> never saw, at a small cost in recall.
> **Judged by** (seed means of 3 seeds per arm, the treated arm against the
> same recipe without steady noise on the same seeds; decided before any
> run). All at the deployment threshold frozen on validation; the steady
> suite is never used to choose between variants.
> - Success needs all of these:
>   - Real unseen no-drone audio: the FSD50K distracter, background and
>     wind stems (none trained on), pooled false alarms, with the aircraft
>     classes (helicopter, prop plane, aircraft) reported apart. They fall
>     by at least half.
>   - The mower session: at most 20% false alarms.
>   - Synthetic tripwires, never reported as performance: war-shaped noise
>     from all 23 clips, white, pink and brown noise, each at most 5%. The
>     war track stays at most 1%.
>   - Recall guards: validation recall at the deployment threshold drops at
>     most 2 points. Lund recall drops at most 3 points. FPV mean recall at
>     10% own false alarms on the iteration 9 leave-one-unit-out arms drops
>     at most 3 points, and no unit more than 5.
> - Failure: any held-out colour above 20%, or a recall guard broken. That
>   would be the evidence that MFE with per-window level removal cannot
>   separate a steady drone from steady noise, and the next council would
>   weigh PCEN with these numbers.
> - `comb` and `lgbm` are reported next to both arms. This iteration targets
>   R3 robustness. R2 (Lund recall 0.56 against 0.95) stays the larger gap.
> - Iteration 10 (`it10-SynthDrone`) is not logged in this file yet; its
>   number is left to it.

## 100 - Council: what to do about false alarms on steady noise

- Date: 2026-11-04
- Iteration: 11
- Type: Council
- Commit: 791d95e plus the uncommitted entries 096-099

**Question.** Iteration 11: (A) train on war-shaped noise (under drones
and alone) and test false alarms on held-out real war clips (the 23 FSD50K
war clips split by uploader), (B) a PCEN frontend, (C) back to MFCC,
(D) wait for real war recordings, or (E) keep the current approach?

**Evidence gathered for the brief** (synthetic diagnostics, not detection
performance). `drone_war_grid.py`, no drone, one 20 s clip, war-shaped
noise seed 1000 against the war track's first 20 s (calm phase):
```
uv run python scripts/drone_war_grid.py --model "<m>" --war-source shaped|track --speeds 0 --no-drone --levels 4 --war-min-db -40 --war-max-db -10 --wars 1 --out <scratch>
```

| Model | War-shaped noise, -40/-30/-20/-10 dB | War track, same levels | `comb`, shaped / track |
|---|---|---|---|
| it6 CNN (model 6) | 100 / 100 / 100 / 100% | 0% at all | 3.3% / 10.5% |
| it9 CNN (entry 096 DS-CNN) | 98.7 / 99.3 / 99.3 / 99.3% | 0% at all | same |
| it9 CRNN (entry 096) | 100% at all | 0% at all | same |

Plain noise, 20 s, -30 dBFS RMS, seed 0, scored with `dad.wavio.stream` at
each model's frozen threshold (alarm share, median score):

| Model | White | Pink | Brown |
|---|---|---|---|
| it6 CNN | 0.0% (0.05) | 94.8% (0.43) | 88.9% (0.42) |
| it9 CNN | 2.6% (0.25) | 100% (0.59) | 9.2% (0.27) |
| it9 CRNN | 0.0% (0.16) | 53.6% (0.48) | 37.3% (0.45) |

Entry 075's guess (steady 600-750 Hz energy from the shout and siren clips)
does not explain pink and brown noise; the broader reading is steady noise
with low-frequency emphasis.

**Devil's Advocate.** Decisive and serious objections, and how each was
answered:
- *Decisive: the proposed test is at the floor before the change.* Real
  war audio already gives under 1% (entry 065) and 0% (above), so held-out
  war clips can only show a regression; the only number that moves is
  synthetic. **Accepted.** The test is replaced by real unseen steady noise
  (the FSD50K distracter, background and wind stems, which no it6 or it9
  model trained on; checked: `rangerun.py`, `realrun.py` and `run_range.py`
  do not read the stems) and the mower session. The synthetic noises stay
  as tripwires only.
- *Decisive: the failure is stationarity and spectral tilt, not war; A
  fixes one spectrum.* **Accepted.** Training gets a family of steady
  noises, not the war spectrum.
- *Serious: the generator becomes a "no drone" fingerprint.* **Answered in
  the design.** Two different generators (random slopes, random-phase
  copies of real windows), a slow level wobble, and a test made of noise
  from neither: real stems, the mower, war-shaped noise, exact colours.
- *Serious: low-SNR positives are label noise and cost recall.* **Partly
  answered.** Maskers go under only 30% of positives, in the existing
  `buried` range (-5 to +15 dB in-band), and recall guards are part of the
  verdict. It stands as a risk.
- *Serious: 23 clips, 201.6 s, one siren: too few to split, and an invite
  to tune on test.* **Accepted.** No war clip is split or trained on; all
  stay test-only, and the steady suite is never used for selection.
- *Serious: a one-seed guard cannot detect a regression.* **Accepted.**
  Three seeds per arm, both arms on the same seeds.
- *Serious: opportunity cost against Lund recall.* **Stands, noted.** One
  iteration of about 30-60 GPU minutes; R2 remains the main gap.

**Scholar** (given the problem without our context). Best practice is
multi-condition training: real or realistic noise mixed under positives at
random SNR plus noise-only negatives, with false alarms reported on real
background from sources never seen, seen and unseen noise reported apart
(the Aurora-2 set A / set B idea; checked via secondary sources only).
Pitfalls: a long-term spectrum computed from the test recordings is a
statistic of the test set (leakage); steady noise as negatives can teach
"stationary = no target" and hurt steady targets (Salamon & Bello 2017 saw
augmentation hurt the steady air-conditioner class; checked); augmentation
gains are often modest (Reuter, Ohlenbusch & Rollwage, arXiv:2605.31329,
2026, acoustic drone detection: little gain out of domain; checked).
Placement: A is standard if noise and test are disjoint by source; B (PCEN)
a defensible variation, with the caveat that it suppresses what is
stationary and could attenuate a hovering drone (Lostanlen et al. 2019;
checked); C against practice for a CNN; D against practice as a blanket
rule; E against practice.
**Departure:** the Scholar prefers real noise recordings in training; we
train on synthetic steady noise and keep the real noise stems for the
test. Reason: the stems are the only real, unseen steady noise we have;
training on them would leave no unseen-noise test. When real field noise
arrives, it can be split by source between the two.

**Problem Solver.** Verdict: try a modified version. Train on a
steady-noise family, not war audio; keep all war material for testing;
score a steady false-alarm suite; three seeds per arm; step 0 first:
score the existing models on random-phase copies of drone and no-drone
windows and on a slope sweep, to see whether they use the spectral
envelope rather than the harmonics, before any GPU time. Confidence
medium. One factual error, checked against entry 094: it says MFCC "lost
18 points on Lund"; that was the linear spectrogram. MFCC lost on the FPV
mean (0.485 against 0.669), which is enough to rule C out.

**Decision.** Iteration 11 as boxed above: the Problem Solver's modified
A, with the Devil's Advocate's two decisive objections built in and the
Scholar's leakage rule kept by putting no war or stem audio in training.
B (PCEN) is held back for a later council if iteration 11 fails. C, D and
E are rejected. Real war recordings are still wanted, as test data.

**Next.** Step 0 (no training): score it6, it9 CNN and CRNN on
random-phase copies of real drone and no-drone windows and on a
noise-slope sweep from -12 to +6 dB/octave, and score the steady suite as
the baseline.

## Note on numbering

The entries below were written on branch `NicolaiBranchGUI` as 095 to 104, and later on `master`, while entries 095 to 100 above were written elsewhere. They were renumbered twice when the work met on `master` (2026-11-04) and are now 101 to 115; references between them were renumbered with them.


## 101 - Ten ITU ARIS excerpts drawn for Mads to listen to (sealed set opened)

- Date: 2026-11-04
- Iteration: none (between iterations)
- Type: Data, Evaluation
- Commit: 87a7bdf, uncommitted

**What changed.** Mads asked for 10 random drone clips from the sealed test
that he can open in `scripts/detect_gui.py` himself. ITU ARIS was not on this
PC, so it was downloaded again from Zenodo, and 10 excerpts of 10 s were
written out.

**Why.** Mads's request. He was told first that clips he scores are no longer
sealed.

**How.**
- Download: Zenodo record 22980617, `itu-aris-lab-acoustic-drone-dataset-v1.1.0.zip`
  (161.3 MB, CC BY 4.0), with `curl.exe` into `data/ITU_ARIS_Lab/`. MD5
  `90eb3f25769a72415432ba62f5687514` matches the host and entry 027, so this
  is the same version as before (v1.1.0 changed documentation only; audio,
  `groups.json` and splits are identical to v1.0.0, record 22682339). The
  host record is saved as `_source_record.json`. The record was found by web
  search, since the `manifest_zenodo.tsv` and `download_new_datasets.ps1` of
  entry 027 are not on this PC and not in git.
- Unzipped to `data/extracted/ITU_ARIS_Lab/` (5,499 files); the zip is
  untouched.
- Draw: a scratch script (not in the repo) takes the 35 drone intervals of
  `groups.json`, keeps those with at least 19 segments (10 s), picks 10
  distinct intervals and one random start in each with numpy
  `default_rng(20261104)`, and joins every second 1 s segment (segments
  overlap 50 %), so each excerpt is 10 s of continuous audio at 22.05 kHz,
  unchanged otherwise. Continuity was checked: the second half of each
  segment equals the first half of the next, max difference 0.0.
- Output: `data/extracted/sealed_peek/aris_peek_01.wav` to `_10.wav`, and
  `draw.json` with the interval, segment range and offset of each.

| File | Interval | Segments | Offset in interval |
|---|---|---|---|
| aris_peek_01 | drone_interval_10 | drone_0712-0730 | 10.5 s |
| aris_peek_02 | drone_interval_11 | drone_0750-0768 | 6.0 s |
| aris_peek_03 | drone_interval_18 | drone_1311-1329 | 2.0 s |
| aris_peek_04 | drone_interval_19 | drone_1336-1354 | 0.0 s |
| aris_peek_05 | drone_interval_25 | drone_2546-2564 | 32.0 s |
| aris_peek_06 | drone_interval_26 | drone_2586-2604 | 6.5 s |
| aris_peek_07 | drone_interval_27 | drone_2930-2948 | 1.0 s |
| aris_peek_08 | drone_interval_28 | drone_3021-3039 | 28.0 s |
| aris_peek_09 | drone_interval_32 | drone_4185-4203 | 0.0 s |
| aris_peek_10 | drone_interval_34 | drone_4454-4472 | 17.5 s |

**Result.** No model was run on the clips by Claude. Whatever Mads sees in
`detect_gui` is a look at the sealed set, not a reported figure.

**Assessment.** ITU ARIS is partly opened: 10 of 35 drone intervals have
been seen. Every interval holds one drone pass, so a later sealed score
should leave out these 10 intervals (listed above and in `draw.json`), or
say that they were seen. If what Mads sees steers a design choice, that
must be written down (CLAUDE.md section 5).

**Next.** Mads tests the clips in `detect_gui`.

## 102 - Detection window: optional war-shaped noise at a chosen SNR

**Why.** Mads asked for a tick box in `scripts/detect_gui.py` that adds
war-shaped noise to the opened files at a chosen level in dB, with a
sensible min and max, scores the noisy clip and lets him listen to it
afterwards.

**What.** On top of commit 87a7bdf, uncommitted. `scripts/detect_gui.py`:
- "add war-shaped noise" tick box and an "SNR dB" box above "Open audio
  files...": -20 to +30 dB, default +10, lower = more noise. A value
  outside the range, or not a number, is refused in the status line before
  the file dialog opens. The setting applies to the next batch.
- The level is an SNR, as in `docs/war_shaped_noise.md`: the file's
  broadband RMS over the noise's RMS, both at 16 kHz after `read_wav`. A
  file-relative level was chosen over an absolute dBFS level because the
  opened files have unknown recording levels (entry 080).
- `war_noise()`: `war_shaped_noise.shaped_noise` with the 44.1 kHz gain
  table (`Pure-data+Script/WarShapedNoise/war_gain.txt`), seed 1000 + the
  file's place in the batch (test seeds start at 1000), resampled to
  16 kHz; at most 60 s, repeated for longer files (the noise is periodic,
  so the loop has no click). A digitally silent file is listed as an error,
  since there is no level to set the noise against.
- The detectors score the mix; Play (main and stats window) plays the mix;
  the mix is written as a 16 kHz float WAV to
  `artifacts/detect_gui/mixes/<stem>_wsn_snr<SNR>dB_seed<seed>.wav`. The
  spectrogram title names the SNR and seed; `results.csv` has
  `noise_snr_db`, `noise_seed` and `mix_file`.

**Checks.** Headless (Agg backend, dialog stubbed), on two made-up files
(8 s 180 Hz tone plus noise at 44.1 kHz; 3 s white noise at 16 kHz), not on
the ITU ARIS clips:
- "abc", "40" and "-25" were refused, and no batch started.
- SNR +10 and -5.5 (typed "-5,5"): measured SNR of the saved mix against the
  clean file +10.000 and -5.500 dB; seeds 1000 and 1001; the played audio
  equals the saved WAV. Tick box off: no noise, audio equal to the file.
- 150 s of noise: 2,400,000 samples, the 60 s loop repeats exactly.
- `uv run pytest`: all pass. The window was rendered to PNG and looked at.
- Seen in passing, made-up audio only: at SNR -5.5 dB the CNN alarmed on 59 %
  of the windows of the white-noise file with war noise added, and on none
  at +10 dB. Steady war-shaped noise at a high level can make the CNN alarm
  by itself. Synthetic scaffolding (section 5): not a detection figure.
- Not checked in a live window: the tick box and text box by mouse, sound.

**Next.** None. A tool change.

## 103 - Correction: Play in the detection window never made a sound on Windows

**Why.** Mads could not hear anything from Play in `scripts/detect_gui.py`.

**What was wrong.** `play()` called `winsound.PlaySound(bytes, SND_MEMORY |
SND_ASYNC)`. Python's `winsound` refuses that combination with
`RuntimeError: Cannot play asynchronously from memory` (reproduced on this
PC), and the error was lost in the matplotlib button callback. So Play has
been silent on Windows since entry 080. Entry 082's check stubbed `winsound`
and only recorded what it was sent, so it could not catch this; its "Play
sent 320000 samples" was true of the stub, not of the sound card.
`scripts/stress_gui.py` had the same call.

**What.** On top of commit 87a7bdf, uncommitted. Both scripts now stop any
sound, write the clip to a temp WAV, play it with `SND_FILENAME |
SND_ASYNC`, and delete the previous temp file. What the detectors score is
unchanged.

**Checks.** `play()` and `hush()` of `detect_gui.py` with the real `winsound`
(Agg backend, a 1 s 440 Hz tone): no error, a second Play deleted the first
temp file, Stop raised nothing. `stress_gui.py` compiles; not run. Whether
the speakers sound was not observable from here; Mads to confirm.

**Next.** None. A bug fix. (Mads confirmed afterwards that he hears the sound.)

## 104 - Detection window: several SNRs at once, one graph per SNR

**Why.** Mads asked to test several war-noise levels in one go (SNR 0, 10
and 30 dB) and get one graph per level.

**What.** On top of commit 87a7bdf, uncommitted. `scripts/detect_gui.py`:
- The SNR box takes one or more values, apart with spaces, commas or
  semicolons, each -20 to +30 dB, at most 6, repeats dropped; default
  "0 10 30". Decimals take a point, so entry 102's "-5,5" now reads as two
  values, -5 and 5.
- Each file is scored once per SNR. Every SNR is its own row in the list
  (the SNR after the name), its own mix WAV and its own rows in
  `results.csv`. All SNRs of a file use the same noise (seed 1000 + the
  file's place in the batch), so only the level differs between them.
- Under the spectrogram the file gets one confidence graph per SNR, both
  detectors in each, the selected row in bold with a dark frame. Clicking a
  graph selects that SNR: the spectrogram switches to it and Play plays it.
  Without noise there is one graph, as before.
- Fixed in passing: an empty list divided by zero when laying out the
  graphs (found by the check below before Mads ran it).

**Checks.** Headless (Agg backend, dialog stubbed), the two made-up files of
entry 102:
- "0 10 30" and "0, 10; 30" give [0, 10, 30]; "0 10 30 10" drops the repeat;
  7 values, "x", 40 dB and an empty box are refused.
- Six rows (2 files x 3 SNRs); measured SNR of each saved mix +0.000,
  +10.000, +30.000 dB; the noise of one file is the same at every SNR up to
  scale. Row 0 shows three graphs (rows 0-2), row 3 shows rows 3-5; a click
  on the third graph selected the +30 dB row. 12 CSV rows. Noise off: two
  rows, one graph each.
- `uv run pytest`: all pass. The window was rendered to PNG and looked at.
- Made-up audio only, not a detection figure (section 5): the CNN alarmed
  on 41 % of the white-noise file's windows at SNR 0 dB, on none at 10 and
  30 dB, which matches entry 102's observation that strong war-shaped noise
  alone can trigger it.
- Not checked in a live window: mouse clicks on the graphs.

**Next.** None. A tool change.

## 105 - Detection window: a single decision was invisible; ITU ARIS background opened

**Why.** Mads opened `background_0001.wav` (ITU ARIS, the no-drone class, 1 s)
at SNR -20, 0, 10 and 30 dB. The list said CNN DRONE 100 % at -20 and 0 dB,
but the graphs were empty, and he heard footsteps.

**What was wrong.** A 1 s file gives exactly one 1 s window, so one decision,
stamped at 1.00 s. "100 %" was 1 of 1 windows alarmed, and correct. A line
through one point draws nothing, and `fill_between` with one point shades
nothing, so the alarm did not show. The footsteps are in the ITU ARIS
recording itself (its background class comes from the session's opening
25 minutes, with people around), not in the war-shaped noise, which has no
events.

**What.** On top of commit 87a7bdf, uncommitted. `scripts/detect_gui.py`:
with fewer than 20 decisions, the confidence curves get dots (not clipped at
the right edge), and each alarmed decision shades the 1 s window it heard;
the same dots in the stats window. Longer files are drawn as before.

**Checks.** A made-up 1 s white-noise file at -20, 0, 10 and 30 dB: CNN
scores 0.585, 0.292, 0.146, 0.133 (threshold 0.297), so DRONE only at
-20 dB; the dots and the shaded window show in the rendered window.
`uv run pytest`: all pass. Made-up audio, not a detection figure.

**Sealed set.** Mads's run scored at least one ITU ARIS background clip
(`background_0001.wav`), beyond the 10 drone excerpts of entry 101. What he
saw there: CNN alarm at SNR -20 and 0 dB, none at +10 and +30 dB; comb none.
It was looked at, not counted as a sealed figure. A later sealed score
should leave out the background clips that have been opened, or say that
they were seen.

**Next.** None. A tool fix.

## 106 - A 10 s ITU ARIS background excerpt next to the drone excerpts

- Date: 2026-11-04
- Type: Data
- Commit: 87a7bdf, uncommitted

**Why.** Mads asked for a 10 s no-drone clip in `data/extracted/sealed_peek/`
to check false positives in `detect_gui` next to the drone excerpts of entry
101.

**How.** A scratch script (not in the repo), numpy
`default_rng(20261105)`: one random background block of `groups.json` (20
blocks of 50 segments), one random start, 10 consecutive segments.
- The first attempt, joining every second segment as for the drone
  excerpts (entry 101), was stopped by its own continuity check: unlike the
  drone segments, the background segments do not overlap. A search over
  lags for three segment pairs found no shared audio. The README's "spaced
  0.5 s apart" therefore does not mean a 0.5 s hop with overlap; the gap
  between segments is not documented.
- So the 10 segments are put end to end, with a 5 ms raised-cosine fade out
  and in at each of the 9 joints so a joint is not a click. The clip is 10 s
  of real ITU ARIS background, not 10 s of continuous audio: each joint
  skips an unknown stretch of the recording.

**Result.** `data/extracted/sealed_peek/aris_peek_background_01.wav`: block
`background_block_15`, segments `background_0782` to `background_0791`,
10.0 s at 22.05 kHz, 16-bit. `draw.json` has a `background` entry with the
same facts. Not scored by Claude.

**Assessment.** Ten more ITU ARIS background segments are opened (with
`background_0001`, entry 105). Leave them out of, or name them in, a later
sealed score. The fades and joints are not in the original audio; an alarm
exactly at a joint could come from the joint.

**Next.** Mads tests it in `detect_gui`.

## 107 - The war scenes no longer need the wind and distracter stems

**Why.** On this PC, Run in `scripts/drone_war_gui.py` failed with
`LibsndfileError: Error opening ...stems\wind\wind\fsd50k_eval_382591.wav`.
Only `build_stems.py --picked` has been run here: `stems/wind/` is absent,
and so are the distracter classes `car`, `motorcycle` and `birds`.
`env_sim.load()` read the wind stem and every distracter stem up front,
although the drone-in-war test uses neither (wind step 1 = off, no
distracters).

**What.** On top of commit 87a7bdf, uncommitted. `scripts/env_sim.py`:
`load()` returns a `Sources` dict that reads `"wind"` and `"distracters"`
on first access and keeps them. A missing wind stem is then reported by
name with "run scripts/build_stems.py"; missing distracters keep their
existing message from `distracters.load_stems()`. Every caller
(`env_sim.build`, `env_grid.py`, `env_gui.py`, `drone_war_grid.py`) is
unchanged: they index `src["wind"]` as before. Nothing about what a scene
sounds like changed.

**Checks.**
- New `tests/test_env_sim.py::test_wind_and_distracters_read_only_when_used`:
  a war-only build with no wind stem on disk reads neither and gives the
  same mix as before; turning wind on raises the message naming the stem.
  `uv run pytest`: all pass.
- On this PC: `drone_war_grid._sources()` loads the 600 s war track without
  touching wind or distracters, and
  `drone_war_grid.py --speeds 1 --speed-min 150 --speed-max 150 --wars 1
  --levels 1 --war-min-db -10 --war-max-db -10` ran to the end (one cell,
  grid PNG, `results.csv`, `hops.csv`, in a scratch folder). Synthetic
  scaffolding; no figure from it is reported.
- Not checked: the GUI window itself by mouse. `env_gui.py` and
  `env_grid.py` still need the stems when they use wind or distracters.

**Next.** None. A bug fix.

## 108 - The sealed_peek clips committed to git, on Mads's request

**Why.** Mads asked to push `data/extracted/sealed_peek/` to GitHub. This is
an exception to section 6 (`data/` is not in git), made by Mads for these
11 small clips only.

**What.** Commit 55175a1 on branch `NicolaiBranchGUI`, holding only that
folder (13 files, about 4.9 MB), force-added past the `data/*` and `*.wav`
ignore rules. Mads had renamed two clips; checked against the source
segments: `Drone 10.wav` is `aris_peek_10.wav`, `Dummy.wav` is
`aris_peek_background_01.wav`. A `README.md` was added with the CC BY 4.0
attribution the licence requires, the renames, and the note that these
sealed-set clips are opened. The other uncommitted changes (entries 102 to
107) are not in this commit.

**Not done.** The push failed here: git needs Mads's GitHub login, which
Claude cannot enter. Mads pushes from GitHub Desktop. If the repository is
public, the clips are public once pushed.

**Next.** Mads pushes.

## 109 - Correction to entry 108: the sealed_peek commit is Mads's be2db19, without the README

The first push from GitHub Desktop failed with `remote rejected ... (Internal
Server Error)` from GitHub (githubstatus.com said all systems operational
minutes later, so taken as transient; the 3.2 MB pack is far below GitHub's
limits). Commit 55175a1 of entry 108 was then undone, and Mads committed the
folder himself as be2db19, "Add sealed audio evaluation peek": the 11 WAVs
and `draw.json`, **without** the attribution `README.md`, which is no longer
on disk. That commit was undone in turn and has been restored by
fast-forward; it is what goes to GitHub. The other changes (entries 102 to
109) were stashed meanwhile and restored unchanged.

Open point: CC BY 4.0 requires attribution where the clips are shared. With
no README in the folder, the attribution lives only in this report
(entries 101, 106) and `docs/datasets.md`. Mads to decide whether the folder
gets it back. Not pushed from here (no GitHub login in Claude's shell).

## 110 - Correction: entries 101 to 109 were asked for by Nicolai, not Mads; the push keeps failing

**Who.** Entries 101 to 109 say "Mads asked". The requests came from Nicolai
Lennart Larsson (git user and e-mail of this machine, GitHub
`TwoStepSpinAround`), working in this repo. Every "Mads" in those entries
means Nicolai, except where an entry quotes REQUIREMENTS.md or an older
decision. The exception to CLAUDE.md section 6 in entry 108 (sealed_peek in
git) was Nicolai's call; Mads owns that rule and has not been asked.

**Push.** The second push of be2db19 from GitHub Desktop failed the same way
(`remote rejected ... (Internal Server Error)`, 2026-11-04 09:45:06Z). Not the
cause, checked: GitHub status (Git Operations operational, no incident);
branch rules (none on `NicolaiBranchGUI`); write access (Nicolai's account
has a commit on `origin` already); LFS (no `.gitattributes`); pack size
(3.2 MB). The repository is public. The remote branch is still at 87a7bdf.

To find out whether the content is the trigger, the push is split. be2db19 is
kept on the local branch `sealed-peek-backup`. `NicolaiBranchGUI` now holds
ebe3bf7, "Add sealed_peek draw metadata", with `draw.json` only; the WAVs
are on disk, untracked (ignored), to follow in a second commit. Nicolai
pushes ebe3bf7 next. If it goes through, the WAVs are the trigger; if it
fails too, the problem is not the files.

Follow-up, same day: ebe3bf7 (`draw.json` only) pushed fine and is on
`origin`. Nicolai then renamed the clips without spaces: `aris_peek_01` to
`_09` became `Drone_01` to `_09`, `Drone 10` became `Drone_10`, `Dummy`
unchanged (each checked by blob hash against the committed file). The WAVs
are now one commit, 029980f "Add sealed_peek WAVs", on top of ebe3bf7, for
Nicolai to push. `draw.json` keeps the original names. Whether the space in
"Drone 10.wav" caused the server error is not established.

Then: 029980f pushed. Nicolai committed `scripts/detect_gui.py` as 03c725d
"Detect_gui" and merged `NicolaiBranchGUI` into `master` (076072e). The rest
of the uncommitted work (this report's entries, `env_sim.py`,
`stress_gui.py`, `tests/test_env_sim.py`) went into a GitHub Desktop stash on
`NicolaiBranchGUI`; it was brought onto `master` from that stash, and these
entries renumbered (note above). The stash itself is kept.

## 111 - Detection window: the 600 s war track as a noise source

- Date: 2026-11-04
- Type: Tooling
- Commit: on top of 076072e (master), uncommitted

**Why.** Nicolai asked to add a random stretch of the 600 s war soundscape
instead of war-shaped noise, still at a chosen SNR.

**What.** `scripts/detect_gui.py`:
- The tick box is now "add war noise", with a source below it:
  "war-shaped noise" (as entries 102 and 104) or "war track (600 s)".
- War track: `data/processed/war/war_track.wav` (600 s, 44.1 kHz,
  `scripts/war_track.py` seed 0: FSD50K war sounds in calm, normal and
  intense phases), read once and resampled to 16 kHz. Each file gets a
  window of its own length from a start drawn with seed 1000 + the file's
  place in the batch, the same window at every SNR. A file longer than the
  track gets the track repeated from 0. A window with no sound at all is
  listed as an error.
- SNR as before: the file's broadband RMS over the window's RMS. The track's
  level changes with its phases and events, so at one SNR a calm window is
  amplified more than an intense one, and its events stand out against it.
- The title, the mix name (`<stem>_wartrack<start>s_snr<SNR>dB_seed<seed>.wav`)
  and `results.csv` (`noise_source`, `noise_offset_s`) give the window's
  start.

**Checks.** Headless (Agg backend, dialog stubbed), two made-up files (8 s
tone, 3 s white noise), war track at 0, 10 and 30 dB: measured SNR +0.000,
+10.000, +30.000; the added signal correlates 1.000000 with the track window
at the stated start (120.6 s and 539.3 s); the same seed gives the same
start; a file longer than the track starts at 0; the war-shaped source still
works. `uv run pytest`: all pass (with the restored `env_sim.py` and test).
The window was rendered to PNG and looked at. Made-up audio, not a detection
figure (section 5). Not checked in a live window: the radio buttons by mouse.

**Next.** None. A tool change.

## 112 - Detection window: a slide show of the SNR graphs

- Date: 2026-11-04
- Type: Tooling
- Commit: on top of 076072e (master), uncommitted

**Why.** Nicolai asked for a button that turns the stacked SNR graphs into a
slide show in the same window, each graph large.

**What.** `scripts/detect_gui.py`:
- "Slide show" (top right) hides the spectrogram and shows the current row's
  confidence graph alone over the right side, larger text, the CNN and comb
  summaries on two lines, and the file, noise and "slide i of n" above it.
- "< SNR" / "SNR >" (shown only in the slide show) and the Left / Right keys
  step through the file's SNRs, wrapping around; Play plays the slide shown;
  Previous / Next and the list still change file. Escape or the button
  ("Back to all graphs") returns to the normal view.
- Fixed in passing: for short files (under 20 decisions, entry 105) each
  alarmed decision shaded its own 1 s window, so overlapping windows stacked
  into darker bands. Overlapping windows are now merged and shaded once.

**Checks.** Headless (Agg backend, dialog stubbed), two made-up files at
five SNRs with the war track: toggling before any file is open does not
fail; slide show on gives one graph, spectrogram hidden, arrows shown;
"SNR >" five times visits rows 0, 1, 2, 3, 4, 0; Left from row 0 goes to the
+30 dB row; selecting row 7 shows that file's slide 3 of 5; Escape gives back
five graphs and the spectrogram. Both views rendered to PNG and looked at.
`uv run pytest`: all pass. Not checked in a live window: keys and clicks.

**Next.** None. A tool change.

## 113 - Detection window: "dummy" scores the added noise alone

- Date: 2026-11-04
- Type: Tooling
- Commit: on top of 076072e (master), uncommitted

**Why.** Nicolai asked for a "dummy" tick box that shows the confidence on
the war sound alone, to see what difference it makes when a drone is in the
file too.

**What.** `scripts/detect_gui.py`:
- "dummy" tick box beside the noise source; it needs "add war noise" (with
  it alone, the status line says so and nothing runs).
- With it, each row's added noise (the same stretch, at the level it has in
  that mix) is scored on its own as well. Both detectors' noise-alone curves
  are drawn dotted in the same graph, in the normal view and the slide show,
  not shaded; the title adds "(noise: N%)", the share of noise-alone windows
  that alarmed. `results.csv` gets `noise_alone_verdict` and
  `noise_alone_flagged_share`.
- Reading it: the gap between the solid (file + noise) and dotted (noise
  alone) curves is what the file adds; an alarm on the dotted curve is a
  false alarm from the noise. `comb` is a level-free ratio, so its dotted
  curve is the same at every SNR; the CNN's depends on level.
- The scoring of a clip moved into `score()`, used for the mix and the
  noise alone; `add_noise` also returns the added layer.

**Checks.** Headless (Agg backend, dialog stubbed), two made-up files, war
track at -10 and +10 dB: "dummy" without noise is refused; the noise-alone
scores equal an independent rescoring of (mix - file) (CNN, `allclose`);
noise alone gave CNN 0 % everywhere and comb 0 % (tone file's window) and
53 % (white-noise file's window) at both SNRs; dummy off gives no
noise-alone curves; both views rendered to PNG and looked at. `uv run
pytest`: all pass. Made-up audio, not a detection figure (section 5).

**Next.** None. A tool change.

## 114 - Detection window: the noise alone gets rows of its own

**Why.** Nicolai ticked "dummy" and could not see it. The noise-alone curves
of entry 113 are dotted lines in the graphs only; where the noise alone
scores near 0 they lie along the bottom of the graph and are hard to see.
Nothing was in the list.

**What.** On top of 076072e (master), uncommitted. `scripts/detect_gui.py`:
with "dummy", each file's noise alone also gets list rows, after the file's
own, named "NOISE <file> <SNR>", one per SNR, as their own group. Selecting
one shows the noise's spectrogram, its graphs at every SNR (titled "NOISE
ALONE") and a title saying which stretch it is and at whose level; Play
plays the noise alone. The dotted overlay stays. `results.csv` keeps one row
per file, SNR and detector (the noise rows are already in the
`noise_alone_*` columns).

**Checks.** Headless (Agg backend, dialog stubbed), two made-up files, war
track at 0 and 10 dB: 8 rows (4 file, 4 noise); a noise row's audio equals
mix minus file; its graphs are the noise rows of that file; 8 CSV rows.
Rendered to PNG and looked at. `uv run pytest`: all pass.

Seen on the way: the stretch of `war_track.wav` drawn for the tone file
starts with 1.2 s of digital silence. Over the whole track, 0.8 % of 100 ms
blocks are exactly silent; much more of the calm phases is very quiet. At a
fixed SNR such quiet stretches are what the file is measured against, so
the noise in a mix can be near nothing for seconds at a time.

**Next.** None. A tool change.

## 115 - Detection window: one dummy row, and one war-track start per batch (replaces 113-114)

**Why.** Nicolai: the dummy should be just the war soundscape with no drone,
once, not a noise row per file; and every file should use the same
soundscape offset.

**What.** On top of 076072e (master), uncommitted. `scripts/detect_gui.py`:
- Entries 113 and 114 are undone: no dotted noise-alone curves, no "NOISE"
  rows per file, no `noise_alone_*` CSV columns.
- "dummy" adds one row at the top of the list, "DUMMY war track": the war
  track alone, no file, no drone, at the track's own level (as rendered;
  full scale 110 dB SPL like the detectors' input), as long as the first
  readable file (10 s if none). It does not need "add war noise" and does
  not depend on the SNR. Every alarm on it is a false alarm. Play plays it;
  `results.csv` marks its rows `dummy=yes`.
- One seed (1000) for the whole batch instead of 1000 + the file's place:
  the war track has one start per batch, drawn so the longest file fits,
  used by every file at every SNR and by the dummy; a window past the end
  wraps to 0. War-shaped noise also uses seed 1000 for every file.

**Checks.** Headless (Agg backend, dialog stubbed), two made-up files (8 s,
3 s): dummy without noise gives one dummy row plus the files; with war track
at 0 and 10 dB, 5 rows, one dummy; the dummy equals the track from 120.6 s;
all four mixes use the start 120.6 s (added signal correlates 1.000000 with
the track there); the CSV has 10 rows with the dummy marked; a 5 s window
from 598 s wraps to the track's start. Rendered to PNG and looked at.
`uv run pytest`: all pass. Made-up audio, not a detection figure.

**Next.** None. A tool change.


## 116 - Merge of Mads's model picker with entries 111-115 in the detection window

- Date: 2026-11-04
- Type: Infrastructure
- Commit: on top of 3d1cc2e (master), uncommitted

**What happened.** Mads pushed 3d1cc2e "gui push": `scripts/detect_gui.py`
got radio buttons that pick the network per batch (it6 CNN, it9 CNN, it9
CRNN, it10 CNN; his entry 098). Pulling it onto Nicolai's uncommitted work
(entries 111 to 115: war track, slide show, dummy) gave four conflicts in
`detect_gui.py`. `docs/REPORT.md` merged by union (his entry 097), and both
sides had numbered new entries from 097: Nicolai's were renumbered 101 to
115 (note above them).

**Resolution.** Both kept. Left column: noise tick box, SNR, source,
dummy, buttons, then the list (now 16 rows, 0.15 to 0.64 of the window),
then Mads's model picker at the bottom. State holds his `model` and
`loaded` with the slide-show flag; the batch worker uses his
loaded-model check with Nicolai's one-start-per-batch noise; `results.csv`
keeps the noise source, start and dummy columns with his per-model
threshold lookup and detector names.

**Checks.** Compiles; `uv run pytest`: all pass. Headless (Agg backend,
dialog stubbed), two made-up files: the dummy and war-track checks of entry
115 and the slide-show checks of entry 112 give the same results as
before the merge; with "it9 CRNN" picked, dummy and war track at 0 and
10 dB, the batch runs on that model and the list and graphs name it.
Rendered to PNG and looked at. Not checked: a live window.

Not touched: the staged deletion of `data/extracted/sealed_peek/Dummy.wav`
(the file is not on disk; whether that is meant is Nicolai's call), and
the GitHub Desktop stash `stash@{0}`, which is kept.

Afterwards: Nicolai committed `detect_gui.py` alone (c0187b0) and moved to
`NicolaiBranchGUI`; the staged `docs/REPORT.md` with entries 101 to 116 was
not committed and was gone from the working tree. It was restored from git's
unreachable objects (blob d28e415: the committed report plus those entries,
nothing removed, no conflict markers).

## 117 - Detection window: tick several models at once

- Date: 2026-11-04
- Type: Tooling
- Commit: on top of c0187b0 (`NicolaiBranchGUI`), uncommitted

**Why.** Nicolai asked for the models to be tick boxes instead of radio
buttons, so several iterations' percentages can be seen side by side.

**What.** `scripts/detect_gui.py`: Mads's model radio buttons (entry 098) are
now tick boxes, each label coloured as its model's curve; at least one must
be ticked (otherwise the status line says so). Each batch scores every ticked
model next to comb, each with its own frozen threshold (loaded once and
kept). The list gets one column per model and comb; with more than one model
the columns are compact ("38%*", * = alarmed). Each model has its own colour
and dashed threshold in the graphs, the slide show and the stats window;
with several models the graph titles are shortened to "model: DRONE N%" or
"none". `results.csv` has one row per file, SNR and detector, the detector
named as the model.

**Checks.** Headless (Agg backend, dialog stubbed), two made-up files, war
track at 0 dB, dummy, all four models: none ticked is refused; five columns;
"it9 CRNN" in the batch scores exactly as it does alone; 15 CSV rows with
the five detector names; list, graphs and stats window rendered to PNG and
looked at. `uv run pytest`: all pass. Made-up audio, not a detection figure.

Seen on the way, not changed: the threshold the GUI reads for "it10 CNN"
(`artifacts/it10_clean/clean_s0.json`) is 5.7e-6. Its scores on the made-up
files and the war-track dummy stay near 0 but above that, so it alarmed on
53 to 93 % of their windows, the dummy included. Whether that threshold is
right is a question for the it10 run, not for the GUI.

**Next.** None. A tool change.

## 118 - Detection window: comb is a tick box too

- Date: 2026-11-04
- Type: Tooling
- Commit: on top of c0187b0 (`NicolaiBranchGUI`), uncommitted

**Why.** Nicolai asked for comb to be switchable like the models.

**What.** `scripts/detect_gui.py`: "DSP (comb)" is the last box in the
detector tick boxes (now titled "detectors"), ticked by default with it6
CNN. Any one or more detectors can run, comb alone included; with none
ticked the status line says so. comb, when ticked, keeps model 6's frozen
threshold and is loaded once on its own, no longer through a network's
loader. The notes under the graphs and in the stats window mention comb's
scaling only when comb ran.

**Checks.** Headless (Agg backend, dialog stubbed), two made-up files, war
track at 0 dB, dummy: comb alone gives one column (`DSP (comb)` in the CSV);
it6 CNN and it9 CNN without comb give two columns, no comb curve or note;
nothing ticked is refused; the stats window and slide show open in each case.
Rendered to PNG and looked at. `uv run pytest`: all pass. Made-up audio, not
a detection figure.

**Next.** None. A tool change.

## 119 - sealed_peek: Dummy.wav renamed Fugle.wav

Nicolai renamed `data/extracted/sealed_peek/Dummy.wav` to `Fugle.wav` and
asked to push it. Same file, checked by blob hash (4b69db3): the 10 s ITU
ARIS background clip of entry 106 (`aris_peek_background_01` in
`draw.json`), not a new recording. Staged with `git add -f` next to the
staged deletion of `Dummy.wav`, so git records a rename. Not committed or
pushed by Claude. Its ITU ARIS origin and CC BY 4.0 attribution are as in
entries 101 and 106.

Committed afterwards as 6f37520 on `NicolaiBranchGUI`, with entries 101 to
119 and the code changes of entries 107 and 111 to 118; git records
`Dummy.wav => Fugle.wav` as a rename (R100). The push failed from Claude's
shell (no GitHub login); Nicolai pushes from GitHub Desktop.

## 120 - Detection window: the list shows mean confidence, not the share of alarmed windows

- Date: 2026-11-05
- Type: Tooling
- Commit: on top of 6f37520 (`NicolaiBranchGUI`), uncommitted

**Why.** Nicolai asked for the list's percentages to be the area under the
confidence curve, as a percentage: the average confidence.

**What.** `scripts/detect_gui.py`: each detector's summary gets `mean`, the
mean of its confidence over all decisions. The decisions are evenly spaced
(0.125 s), so this is the area under the confidence curve divided by the
scored time. comb is averaged as drawn, s/(s+threshold), so 50 % is its
threshold. The list shows it in place of the share of alarmed windows; the
DRONE / "*" verdict is unchanged (any window at or above threshold). The
graph titles keep the share of alarmed windows. `results.csv` gets a
`mean_confidence` column before `flagged_share`.

**Checks.** Headless, two made-up files, it6 CNN and comb: the tone file
gives it6 CNN mean 1.2 % (0 % of windows alarmed) and comb 99.95 % (100 %);
the white-noise file it6 CNN 6.4 % (0 %), comb 0 %. `uv run pytest`: all
pass. Made-up audio, not a detection figure. Not checked: a live window.

A mean confidence is not a probability of a drone and is not comparable
between networks: each has its own threshold (it6 0.297, it9 CNN 0.380, it9
CRNN 0.483, it10 5.7e-6), so the same mean can sit far above one network's
threshold and below another's.

**Next.** None. A tool change.

## 121 - Detection window: the log-mel the networks see, in place of the spectrogram

- Date: 2026-11-05
- Type: Tooling
- Commit: on top of 6f37520 (`NicolaiBranchGUI`), uncommitted

**Why.** Nicolai asked to see the log-mel, since that is what the networks
look at, instead of the linear spectrogram.

**What.** `scripts/detect_gui.py`: the top panel draws
`dad.features.LogMel(DEFAULT).frames(audio)`, the frozen frontend itself (64
mel bands 50-8000 Hz, 25 ms frames, 10 ms stride, Slaney-normalised
filters), converted to dB and shown over an 80 dB range from its maximum. One
row per band; the axis is labelled with band centre frequencies (79 Hz to
7678 Hz), so the low bands take as much height as the high ones. It is the
log-mel before the level normalisation and training EQ each model applies,
so absolute colours are not what the network gets, the pattern is.

**Checks.** Headless: a made-up 8 s 180 Hz tone with war track at 0 dB gives
a 64 x 798 log-mel whose strongest band is band 4 (centre 206 Hz); rendered
to PNG and looked at. `uv run pytest`: all pass. Not checked: a live window.

**Next.** None. A tool change.

## 101 - How well does war-shaped noise stand in for war audio?

- Date: 2026-11-04
- Iteration: 11 (diagnostic)
- Type: Evaluation
- Commit: 791d95e plus the uncommitted entries 096-100 and the new `scripts/noise_stationarity.py`

**What changed.** New `scripts/noise_stationarity.py`. It compares
war-shaped noise (60 s, seed 1000) with the war track
(`data/processed/war/war_track.wav`, 600 s) and with the 23 FSD50K war
clips it was built from (201.6 s, each measured on its own). All three are
read at the frontend's 16 kHz. Measures:
- L10/L50/L90: levels of 125 ms frames, dBFS.
- Crest factor of each 1 s window.
- Within: spread inside a 1 s window of each level-normalised log-mel band.
- Between: spread from window to window of each band's window mean.
- To-shaped: RMS distance of each window's level-normalised mean spectrum
  from the shaped noise's.
- Alarm shares of the three networks at their frozen thresholds, for the
  clips also per class.

A sanity check in the session confirmed that the measures separate steady
noise from the same noise in bursts: L10-L90 1.5 dB at most, a lower
within-window spread and a crest factor under 15 dB for the steady noise.

**Why.** Mads asked whether war-shaped noise can misrepresent random war
sound, and whether that can be checked.

**How.**
```
uv run python scripts/noise_stationarity.py
```
Output: `artifacts/noise_stationarity/stationarity.csv`.

**Result.** Medians. For the clips: the median over clips, with the range
in brackets.

| | War-shaped noise | War track | 23 war clips |
|---|---|---|---|
| L10 - L90 (dB) | 1.1 | 32.7 | 44.2 |
| Crest factor per 1 s window (dB) | 11.9 | 12.3 | 13.4 (9.3-21.3) |
| Within a window, per band (dB) | 3.1 | 3.0 | 5.6 (1.1-17.1) |
| Between windows, per band (dB) | 0.4 | 10.5 | 5.7 (0.6-15.2) |
| Window spectrum to the shaped noise's (dB RMS) | 0.4 | 29.6 | 16.9 (4.6-35.4) |
| Alarms it6 / it9 CNN / it9 CRNN | 100 / 100 / 100% | 1.2 / 0.6 / 0.7% | 8.4 / 7.8 / 7.2% |

Alarms on the clips, per class (it6 / it9 CNN / it9 CRNN):
- tank (2 clips): 56.8 / 52.3 / 45.0%
- small arms (3 clips): 20.5 / 17.4 / 20.5%
- jet (3 clips): 6.7 / 8.1 / 4.3%
- siren, far explosion, machine gun, mortar/artillery, shout: 0% for all three

What it shows:
- **War-shaped noise matches the war audio only in its long-term average
  spectrum.** Over time it is steady: its level spreads 1 dB against 33-44
  dB, and its spectrum changes 0.4 dB from window to window against 6-10
  dB.
- **No real window looks like it.** The median real window's spectrum sits
  17 dB (clips) to 30 dB (track) from the shaped noise's. An average of very
  different sounds (shouts, sirens, gunfire, engines) is a spectrum that
  none of them has. That spectrum is what triggers the 100% alarms.
- **Inside a single window the median track window is as steady as the
  shaped noise** (3.0 against 3.1 dB): most of the track is steady
  background between events. The difference is its spectrum, not its
  steadiness within a window.
- **Correction to entry 100.** Entry 100 accepted the Devil's Advocate's
  point that "real war audio already gives under 1%". That holds for the
  war track, not for the clips. The tank clips (a steady engine) give
  45-57% alarms, and small arms about 20%. Real war audio is therefore not
  at the floor. The tank result fits the steady-noise reading of entry 100.
  The entry 100 decision stands: no war audio goes into training. The
  iteration 11 steady suite adds the 23 war clips, per class, as real
  test-only audio, beside the noise stems and the mower session.

**Limits.**
- One shaped-noise seed.
- The war track is built from war sound effects; it is not a field
  recording.
- The clips are short sound effects, many recorded close.
- None of this says how a real battlefield sounds at a microphone hundreds
  of metres away. Only field recordings would.
- Synthetic noise numbers are diagnostics, not detection performance.

**Next.** Step 0 of iteration 11, with the war clips added to the steady
suite.

## 102 - Step 0: the networks use the harmonic lines, and steadiness pushes them toward "drone"

- Date: 2026-11-04
- Iteration: 11 (step 0, diagnostic)
- Type: Evaluation
- Commit: 791d95e plus the uncommitted entries 096-101 and the new `scripts/phase_probe.py`

**What changed.** New `scripts/phase_probe.py`. It takes real 1 s windows
from the processed store, up to 300 per source and label:
- sources Nightjar, d3set, FPV, Halmstad and Lund;
- drone windows from recordings labelled 1, no-drone windows from those
  labelled 0;
- the mower session and Lund's suspect recording left out;
- random positions, seed 0.

Each window is scored as it is and in two copies, both with the window's
own RMS:
- **phase:** the window's FFT magnitude with random phases. The fine
  spectrum, harmonic lines included, is kept; the time structure (blade
  modulation, onsets) is not.
- **envelope:** the magnitude smoothed over about one octave (at least
  300 Hz) before the random phases. Only the coarse spectral shape is left:
  no harmonic lines and no time structure.

Plus Gaussian noise with slopes from -12 to +6 dB/octave, 20 s each,
-30 dBFS. Checked in the session on a 200 Hz harmonic comb with amplitude
modulation:
- the phase copy keeps the magnitude spectrum, with a peak-to-median ratio
  of 3054, as the original;
- the envelope copy removes the lines, with a ratio of 1.1;
- measured slopes of -6.0 and +2.98 dB/octave for -6 and +3 requested.

**Why.** Entry 100's step 0: before any GPU time, find out what in a
window makes the networks say "drone". Mads asked whether drones simply
are pink or brown noise to the networks.

**How.**
```
uv run python scripts/phase_probe.py
```
Output: `artifacts/phase_probe/phase_probe.csv`. Every network at its frozen
threshold.

**Result.** These windows come from rigs the networks trained on (Lund
excepted). This is a diagnostic of what they use, not detection
performance.

Drone windows the network detects as they are: the share still detected
as a copy (it6 / it9 CNN / it9 CRNN):

| Source | Phase copy (harmonics kept) | Envelope copy (shape only) |
|---|---|---|
| Nightjar | 95 / 94 / 93% | 42 / 37 / 24% |
| d3set | 99 / 98 / 97% | 23 / 7 / 13% |
| FPV | 91 / 93 / 89% | 48 / 32 / 28% |
| Halmstad | 98 / 98 / 96% | 66 / 71 / 54% |
| Lund (never trained on) | 80 / 76 / 93% | 1 / 3 / 0% |

No-drone windows, alarm share as they are, as a phase copy and as an
envelope copy:

| Source | it6 CNN | it9 CNN | it9 CRNN |
|---|---|---|---|
| Nightjar | 33 / 49 / 20% | 28 / 40 / 17% | 14 / 28 / 11% |
| d3set | 32 / 67 / 19% | 17 / 56 / 16% | 4 / 42 / 4% |
| FPV | 28 / 52 / 27% | 19 / 45 / 10% | 11 / 31 / 9% |
| Halmstad | 37 / 69 / 48% | 34 / 64 / 29% | 22 / 51 / 35% |

Noise slope sweep, alarm share (it6 / it9 CNN / it9 CRNN):
- -12 and -9 dB/octave: 0% for all three
- -6 dB/octave (brown): 97 / 14 / 42%
- -3 dB/octave (pink): 97 / 99 / 71%
- 0, +3 and +6 dB/octave: 0% for all three, except it9 CNN with 5% at 0

What it shows:
- **The harmonic lines carry the decision.** Removing the time structure
  keeps 76-99% of detections. Removing the lines as well keeps 0-3% on
  Lund, 7-23% on d3set, 24-48% on Nightjar and FPV, and 54-71% on Halmstad.
  The networks are not detecting "pink noise". On Halmstad, and partly on
  Nightjar and FPV, the coarse shape alone still carries a good part.
- **Steadiness pushes toward "drone".** The phase copy of a no-drone window
  has the same spectrum, only steadier. It alarms 1.4 to 10 times as often
  as the window itself: for example d3set in the CRNN, 4% to 42%. That is
  the shortcut behind entries 075 and 101, and generator (b) of iteration
  11 (random-phase copies of rig no-drone windows as negatives) targets it
  directly.
- **The noise alarms sit in a narrow band of slopes:** -3 and -6
  dB/octave, which is pink and brown. Steeper or flatter noise does not
  alarm. Generator (a), with slopes from -12 to +3, covers that band.
- **No-drone windows drawn at random alarm far above 5%:** 4-37% as they
  are, against a threshold frozen at 5% on validation. Some of these
  recordings hold handling sounds or other events, and the windows are not
  the validation set, so this is not a false-alarm figure. Not looked into
  further here.

**Decision on (c)** (entry 100 left it to step 0). Random-phase copies of
drone windows are **not** added as negatives. They keep the harmonic
lines, which are the networks' main evidence, and a hovering drone is
itself steady and harmonic, so they would teach the networks to reject
real drones. Envelope copies of drone windows (shape without lines) would
target the Halmstad shape reliance. They are noted as a candidate for a
later step, not added now, so that iteration 11 changes one thing.

**Not done.** Entry 100 also asked step 0 to score the steady suite as
the baseline. The war clips were scored in entry 101; the noise stems and
the mower session are not scored yet. That goes with step 2's suite
script.

**Next.** Step 1: generators (a) and (b) in `src/dad/`, wired into
`RangeSampler`, with a flag in `run_range.py` and a test. Then the steady
suite script, and the 3 + 3 seed runs on JupyterLab.

## 103 - Step 1: steady-noise generators in the training sampler

- Date: 2026-11-04
- Iteration: 11 (step 1, code)
- Type: Data preparation
- Commit: c0187b0 plus uncommitted work: entries 096-102, `src/dad/steady.py`, the change to
  `src/dad/rangerun.py`, `tests/test_steady.py`

**What changed.** New `src/dad/steady.py` with three functions:
- `slope_noise`: generator (a). Gaussian noise with a spectral slope drawn from -12 to +3
  dB/octave. The box's "random smooth EQ" is not added here. Every training window, these
  included, already gets the random smooth EQ of `realrun._eq_curve` (±3 dB tilt plus a
  5-point shape, σ 2 dB), so a second EQ would double it.
- `phase_copy`: generator (b). A rig no-drone window's FFT magnitude with random phases, at
  the window's RMS.
- `wobble`: a sinusoidal level change of 0.2-2 Hz with a depth drawn from 0 to 3 dB.

`RangeSampler.steady` draws a training-rig background window (the same draw the
simulated-distance positives use, so the background is from the training sources only). Then:
- **Half the time** it returns the phase copy of that window (b).
- **The other half** it returns slope noise (a) at that window's in-band level.
- **Either way** it applies the wobble.

`Recipe` has two new fields, `steady_neg` and `steady_mask`, both 0 by default. The new recipe
`simdist_steady` is `simdist` with `steady_neg=0.15` and `steady_mask=0.30`. With the new recipe:
- **Negatives:** 15% are steady noise alone, with no FSD50K mixing.
- **Positives:** 30% get steady noise added. The drone over the noise is drawn from -5 to +15 dB
  in-band SNR, the `buried` range.
  - Simulated-distance positive: the "drone" is the scaled near drone, before the rig background
    is added.
  - Real positive: it is the drawn window, before any FSD50K burial.

All other recipes draw nothing new: the test checks that. No war audio and no noise stem is
read.

Entry 102 planned a flag in `run_range.py`. A recipe does the same with no change to the
script. It also lets both arms of the iteration run in one call on the same seed:
`--recipes simdist simdist_steady`. The checkpoint names already carry the recipe.

**Why.** Entry 100's design and entry 102's step 0.

**How.**
```
uv run pytest -p no:cacheprovider
uv run python <scratchpad>/smoke_steady.py
uv run python <scratchpad>/time_steady.py
```
The two smoke scripts are not in the repo. They build the iteration 9 sampler
(`make_pool_arm(manifest, "none")`, speed 0.8-1.25) for both recipes and draw batches. This
machine's processed store has no FSD50K clips (`fsd_train` is empty), so `simdist` itself fails
here. The smoke scripts replace the FSD50K background with the rig background on this machine
only. The FSD50K branch was therefore not exercised on real data, only in the unit test's stub.

**Result.**
- Tests: 105 passed, among them six new ones in `tests/test_steady.py`. They check:
  - **slope_noise:** the slope within 0.5 dB/octave of the one asked for, from -12 to +3, and
    unit RMS;
  - **phase_copy:** the same magnitude spectrum and RMS, with the waveform correlation under 0.5;
  - **wobble:** within ±3 dB;
  - **shares:** the shares of steady negatives and masked positives (15% and 30%, 4,000 draws
    each);
  - **masker SNR:** within -5 to +15 dB;
  - **old recipes:** no steady noise drawn.
- Smoke run on real rig data: both recipes give (64, 64, 98) float32 batches, all finite, with
  32/32 labels. `describe()` names the steady shares.
- Steady draws over 200 draws: median in-band level -58.7 dBFS, 5-95% from -87.5 to -29.4
  dBFS, the spread of the rig backgrounds.
- CPU time of `raw_batch` (64 windows, mean of 30 batches, one process): 44 ms for `simdist`,
  82 ms for `simdist_steady`. Most of the extra time is FFTs at the 20,032-sample drawn length,
  which has a prime factor of 313. Training draws batches in parallel workers (`prefetch`), and
  iteration 9 took about 77 ms per step on the GPU, so this should not slow training much. Not
  measured on JupyterLab.

**Not done.** The steady suite script (step 2) and the 3 + 3 seed runs.

**Next.** Step 2: a script that scores a checkpoint on the steady suite. It scores false alarms on:
- the FSD50K distracter, background and wind stems, pooled, with the aircraft classes apart;
- the mower session;
- the 23 war clips per class;
- the tripwires: war-shaped, white, pink and brown noise, and the war track.

It also scores the iteration 6 and 9 models as the baseline. Then the runs on JupyterLab:
```
uv run python scripts/run_range.py --name it11_s<k> --pool --arms none --speed 0.8 1.25 --recipes simdist simdist_steady --models dscnn --deploy-fa 0.05 --device cuda --seed <k>
```
for k = 0, 1, 2.

## 104 - War-shaped noise dropped; real war audio is the war test

- Date: 2026-11-04
- Iteration: 11
- Type: Evaluation
- Commit: c0187b0 plus uncommitted work (entries 096-103 and their code)

**What changed.** Mads decided to drop war-shaped noise. It goes out of iteration 11's verdict,
and out of future tests. War is tested with real war audio only:
- **Now:** the 23 FSD50K war clips per class, and soundscapes mixed from them.
- **Distracters:** helicopters and tanks are kept as distracters in future tests. The stems hold
  3 helicopter, 3 aircraft and 3 prop-plane clips, and the war stems hold 2 tank clips.
- **Later:** field recordings when they exist.

The rest of the iteration 11 box is unchanged:
- **Pass criteria:** unseen FSD50K stems at least halved, mower session at most 20%, the recall
  guards.
- **Tripwires:** white, pink and brown noise stay as synthetic tripwires. The war track stays at
  at most 1%.

**Why.** Mads's reasoning: war-shaped noise is not what a microphone in the field hears, and the
real soundscapes are. The evidence agrees:
- **Real windows look nothing like it.** War-shaped noise matches war audio only in its long-term
  average spectrum. Real windows sit 17-30 dB from it, and it is steady where war audio swings
  33-44 dB (entry 101).
- **It has done its job.** It exposed the steady-noise failure, and entries 101-102 pinned that
  failure down without it.
- **Real audio already shows the problem.** The real tank clips alarm on 45-57% of windows
  (entry 101), and the mower session alarms on 55% for model 6. Iteration 11 therefore keeps its
  purpose without war-shaped noise.

**What is lost.** War-shaped noise was unlimited and reproducible, and it could be set to any
level. The real war audio is 201.6 s in 23 clips, with 2 tank clips and 1 air-raid siren.
Soundscapes mixed from them vary the timing and the levels, not the sounds. A result on them
rests on few sources, and that goes with every number reported.

**No council.** Before any run, this removes one synthetic tripwire from the verdict. The real
test sets were already the deciding ones, and nothing about the primary metric or the split
changes. The scripts and GUIs that can make war-shaped noise (`drone_war_grid.py`, the GUIs) are
left as they are. They are no longer used to judge models.

**Next.** Step 2's suite script scores the real audio only: stems, mower, war clips per class,
helicopters and aircraft apart, and the war track. White, pink and brown noise are scored as
tripwires.

## 105 - Correction: part of the steady suite comes from FSD50K dev, which training draws from

- Date: 2026-11-05
- Iteration: 11
- Type: Correction
- Commit: c0187b0 plus uncommitted work (entries 096-104 and their code)

**What changed.** Nothing in the code. The stem manifest
(`data/processed/stems/manifest.csv`) was checked against how training draws its FSD50K
negatives.

**Why.** Mads asked whether cars should be labelled no-drone. Answering that meant checking
which FSD50K audio training already holds.

**Result.**
- **Training already holds FSD50K.** It draws FSD50K negatives from `fsd_train`, which is FSD50K
  dev with split `train` (`realrun.make_splits`, `prepare_data.py`). It holds every licence-cleared
  dev clip without a drone keyword, cars, engines and aircraft included.
- **36 of the 97 FSD50K stems come from FSD50K dev, not eval.**

  | Stems | From dev | Classes |
  |---|---|---|
  | Distracter | 18 of 61 | buzzing_insect 2, chainsaw 3, generator 2, helicopter 1, mosquito 2, mower 3, prop_plane 2, scooter 3 |
  | War | 18 of 23 | both tank clips, all mortar/artillery, machine gun, far explosion and shout clips, the air-raid siren, 1 jet |
  | Background and wind | 0 | all from eval |

  Only small arms (3) and 2 jet clips among the war clips are from eval.
- **The dev stems may have been training negatives** for models 6 and 9. That is the case unless
  their dev split is `val`, or they were excluded by licence or drone keyword. This is **not
  verified**: this machine has neither the FSD50K ground truth nor the FSD50K store.
- **Corrections.** Entry 100 said that no it6 or it9 model trained on the stems. That holds for
  the stem files, not for the FSD50K clips they were cut from. Entry 104 used the war clips as
  the war test, so the same applies there.
- **Entry 101's reading still stands.** If the tank clips were training negatives, their 45-57%
  alarms happened despite that, which strengthens the steady-noise reading rather than weakening
  it.

**Decision.** In iteration 11's verdict:
- "Unseen real noise" means the eval stems only: 43 distracter, 10 background and 3 wind clips.
- The dev stems are reported apart, as "possibly seen".
- The war clips are reported per class with their subset.
- No criterion in the iteration 11 box changes. Its stem criterion is narrowed to the stems that
  are unseen.

**Next.** Check the dev stems' splits against `FSD50K.ground_truth/dev.csv`: the 334,701-byte
ground-truth zip from Zenodo record 4060432, or the copy on JupyterLab. Also check whether eval
uploaders also appear in dev `train`.

## 122 - Complete_Test_Gui: a control panel for large grids of test rounds

- Date: 2026-11-05
- Type: Tooling
- Commit: on top of 1fb0db7 (`NicolaiBranchGUI`), uncommitted

Numbering: entries 101 to 105 just above are Mads's (iteration 11), written
on another branch; Nicolai's entries 101 to 121 sit further up. Both sets
are kept as they are; this entry continues Nicolai's numbers.

**Why.** Nicolai asked for one GUI with complete control over a test, built
for 1000+ rounds: two pages in a slide show (real drone test, synthetic drone
test, working like `detect_gui.py` and `drone_war_gui.py`), every background
layer settable, the soundscape length setting the rounds (the drone starts at
every second until it ends with the soundscape), no graphs on the front, and
results in a separate window and a CSV. His answers to the questions asked
first: as many starts as fit (length - 10 + 1, so his example of 20 s, 3
drones and 3 war levels is 11 x 3 x 3 = 99 rounds, not 90); levels are real
levels in dB SPL for every layer and the drone, no SNR; all combinations of
the layers' levels; CSV plus a results window.

**What.**
- `scripts/complete_test_engine.py`: `Plan` (the grid: starts, drones, drone
  levels, every combination of the ticked layers' levels, optional no-drone
  rounds per background), `Layer` (war soundscape, war-shaped noise, wind;
  cars, helicopter and insects listed as not set up), `Drones` (real files cut
  to their first 10 s, or Pure Data drones per speed through
  `drone_war_grid.render_drones`, 10 s to impact), `mix_round`, `score_round`,
  `hop_levels` and `Test.run` (rounds.csv appended round by round, Stop
  between rounds, scores.npz, config.json, summary.csv). Levels: full scale is
  110 dB SPL; a drone level is the RMS over its sound; a background layer's
  level is its long-term RMS over the whole source (all 600 s of the war
  track), so calm and loud stretches keep their difference. The war track
  stretch has one start per batch (seed 1000 unless set). Windows are
  labelled as in `dad.wavio.window_labels` (straddling windows left out). Per
  round and detector: counts, recall, false-positive rate, precision, first
  alarm and delay after the drone starts, lead before impact (synthetic), mean
  confidence in drone and no-drone windows. Mixes are rebuilt on demand, not
  stored.
- `scripts/Complete_Test_Gui.py` (Tkinter): the two pages behind arrows,
  page buttons and Ctrl+Left/Right; sections Drones, Background layers
  (collapsible), Test rounds, Detectors (networks and comb), Run (live round
  count and its breakdown, problems in words, Start/Stop, progress, time
  left). The results window: summary table, a sortable and filterable table
  of every round and detector (false alarms in red), and for the selected
  round its log-mel, each detector's confidence with the drone's span and the
  dB SPL of every layer and the mix over time (1 s windows), with Play mix,
  Play drone, Play background (one gain for all three). "Open results..."
  reopens an earlier run folder.

**Checks.** Engine, made-up 10 s harmonic files: the example plan counts 99;
12 s gives starts 0, 1, 2; a 26-round plan (2 drones x 2 drone levels x 3
starts x 2 backgrounds + 2 no-drone) ran in 16 s with it6 CNN and comb
(about 0.6 s a round); measured levels: drone 60.0 dB SPL as asked,
war-shaped noise 40.0 over its loop, war track 55.0 over the whole track
(60.1 in that 12 s stretch); rounds.csv 52 rows, scores.npz 52 arrays,
summary.csv 22 rows; a synthetic 150 km/h round (cached render) has its
truth from first arrival (1.21 s) to impact (10 s). GUI, driven by a script:
the counts and problem messages update live (102 rounds for the example with
no-drone rounds, 99 without); a 14-round run filled the progress bar,
opened the results window (18 summary rows, 28 table rows; filter, search,
sort, selection, plotting and Play work) and reopened from its folder.
Screenshots looked at. `uv run pytest`: all pass. The made-up run folder was
deleted. Made-up audio and synthetic mixes: not detection figures (section
5). Not checked by a person: the live window, mouse use, sound.


# Soundscape line (branch it11-soundscapes)

> **What this section is.** The work on branch `it11-soundscapes` from
> 2026-11-04 on: synthetic Pd drones in random labelled soundscapes, then
> `--realism`, real Glasgow drones, a CRNN and steady-noise arms. It is a
> different line from master's "Iteration 11 - Steady noise as negatives"
> above. Its artifacts are named `it11*` and `itG*`.
> **Numbering.** Its entries were numbered on the branch, in parallel with
> master's, so their numbers repeat master's: 095-105 and 101-118. Inside
> this section an entry number means this section's entry.
> **Restored entries.** The first subsection (095-105) was dropped from
> REPORT.md by the merge of master on 2026-11-04 14:42 (c96f8d4). It is
> restored here unchanged from commit 850d1f6, except that its opening box
> is no longer headed "Iteration 11". Four entries written after 850d1f6
> (the it11 training start, its validation results, the checkpoint copy
> and the it11 GUI entry) were never committed and are lost. Their results
> are in `artifacts/it11_soundscapes/it11_s{0,1,2}.json`: DS-CNN validation
> AUC 0.826 / 0.812 / 0.822 against lgbm 0.719 / 0.773 / 0.742.

## Soundscape line, part 1 (restored from 850d1f6)

## 095 - Scene generator: random Pd drones in random war-shaped noise
- Date: 2026-11-04
- Iteration: 10
- Type: Data
- Commit: uncommitted (on top of c83c810)

**What changed.** `scripts/make_scenes.py --war`: the only noise in a scene
is war-shaped noise (`scripts/war_shaped_noise.py`, `war_gain.txt`). Each
scene gets its own random-phase draw (noise seed 1 000 000 + scene seed,
clear of the test seeds from 1000 used by `drone_war_grid.py`), made at
44.1 kHz and resampled to 16 kHz. A random Pd drone (the same `render_drone`
as before: random flight, f0 and harmonic falloff) is added with probability
`--p-drone`, at a uniform `--snr` against the noise. No wind, bugs or
helicopter. `scripts/train_clean.py` now takes the scenes that have no drone
and adds them to the no-drone clips, split by the same `--val-seed`.
**Why.** Mads asked for drone-in-war-noise training data for iteration 10.
If war noise appeared only with a drone, the model could learn "war noise
means drone". The war-only scenes rule that shortcut out without needing
new recordings.
**How.**
`uv run python scripts/make_scenes.py --war --p-drone 0.5 --n 2000 --out data/processed/scenes/war1`,
then `train_clean.py --drones <that folder> --nodrone ~/nodrone ...` on the
JupyterLab server. Test: `tests/test_make_scenes.py::test_war_noise_is_seeded_and_war_shaped`.
**Result.** `uv run pytest tests/test_make_scenes.py`: 3 passed. Smoke run
with `--war --p-drone 0 --n 2`: two 10 s war-only scenes, labelled
correctly. Not run: scenes with a drone (Pd was not run here), and no
training.
**Assessment.** Data tooling only, so no numbers to judge against
REQUIREMENTS.md. These are synthetic drones in synthetic noise (CLAUDE.md
section 5), so anything trained on them is scaffolding, not detection
performance. The war-shaped noise is steady and has none of the transients
of real war audio.
**Next.** Generate the scenes and train on the GPU server when Mads says so.

## 096 - Listening files: one war-only scene and one drone-in-war scene
- Date: 2026-11-04
- Iteration: 10
- Type: Data
- Commit: uncommitted (on top of c83c810)

**What changed.** Two 10 s, 16 kHz scenes from entry 095's `--war` mode,
made for Mads to listen to: `data/processed/scenes/war_listen/war_nodrone.wav`
(seed 0, war-shaped noise only) and `war_drone.wav` (seed 1, Pd drone over
the whole 10 s, cruise f0 186.5 Hz, falloff 2.374, SNR +8.5 dB). Each one's
labels are in `nodrone/labels/` or `drone/labels/` next to it.
**How.** `uv run python scripts/make_scenes.py --war --p-drone 0 --n 1 --seed 0 --out data/processed/scenes/war_listen/nodrone`,
and the same with `--p-drone 1 --seed 1 --out .../drone`.
**Result.** Both rendered, and Pd ran. No scoring.
**Assessment.** Listening material only. Synthetic data, not a result.
**Next.** Mads listens. Generate the training set when he says so.

## 097 - Correction: the war-shaped-noise training mode is removed
- Date: 2026-11-04
- Iteration: 10
- Type: Correction
- Commit: uncommitted (on top of c83c810)

**What changed.** The `--war` mode from entry 095 has been removed from
`scripts/make_scenes.py`, along with its test. Entry 096's listening files
stay on disk. `train_clean.py` keeps the part of entry 095 that is still
useful: scenes without a drone become no-drone clips. It now labels them
"scene" instead of "war", and it skips scenes marked heldout.
**Why.** Reviewed against the supervisor's notes (entry 086, re-sent by
Mads on 2026-11-04), training on war-shaped noise has four problems:
- The noise is one steady spectrum, the same one the drone-in-war tests
  (`drone_war_grid.py`, `drone_war_gui.py`) use, so a test would measure
  the training noise.
- `war_gain.txt` averages 5 `fsd50k_eval` clips (a test source) with 18
  dev clips.
- It has none of the transients the supervisor wants the model to handle.
- The supervisor asks for random labelled soundscapes instead.

War-shaped noise remains a controlled test background, which is what
`docs/war_shaped_noise.md` built it for.
**Result.** Nothing was trained on it.
**Next.** Iteration 11.

### The soundscape line's opening box (headed "Iteration 11" on the branch; master's iteration 11 is another line)

> **Changed from iteration 10:** the training audio. Iteration 10
> (`scripts/train_clean.py`, checkpoints in `artifacts/it10_clean/`, commit
> 32142e1; not logged as entries before this one) trained on clean Pd
> drones, with nothing mixed in, against real no-drone audio. Iteration 11
> trains on the scenes of `scripts/make_scenes.py` in full mode:
> - every drone sits in a soundscape: a bed, quiet events, loud transients
>   and, in 40% of scenes, an engine or tonal confuser;
> - drone-to-noise ratio -20 to +10 dB;
> - 25% of scenes have no drone, and those scenes are negatives next to the
>   real no-drone clips.
>
> **Why:** the supervisor's notes (entry 086): train on random labelled
> soundscapes with 30-50% confusers and 20-30% without a drone. A clean
> drone teaches nothing about masking or about engines that are not drones.
> Mads chose this as the first step, before a CRNN or the linear frontend.
>
> **Kept the same:** the Pd drone (`DroneSoundHarm.pd`, cruise f0
> U(100, 700) Hz, falloff log-U(1, 16)), the MFE DS-CNN and the `lgbm` and
> `comb` baselines, the "level" variant, a 1 s window, the real no-drone
> clips, the validation rig (nightjar) and the scene validation split
> (seed >= 1800).
>
> **Hypothesis:** on real drones and real confusers the model raises fewer
> false alarms and loses less recall than iteration 10's clean model, because
> it has seen engines and masking during training.
>
> **Judged by:** three seeds each of iteration 10 and iteration 11, scored
> once on the real FPV test clips at each model's threshold of 5% false
> alarms on validation. Iteration 11 must gain at least 5 points of recall
> over iteration 10 without a higher false-alarm rate on the real no-drone
> validation rig. The `comb` and `lgbm` baselines are reported alongside.
> (Proposed by Claude; Mads may change it before the test is scored.)

## 098 - Scene generator: the supervisor's confuser list, transients outside the SNR
- Date: 2026-11-04
- Iteration: 11
- Type: Data
- Commit: uncommitted (on top of c83c810)

**What changed.** In `scripts/make_scenes.py`, the roles now draw on every
stem class in `data/processed/stems/`:
- **bed:** wind, rain, stream or traffic.
- **events:** insects, buzzing insect, mosquito, birds.
- **transients** (new role, Poisson `--mean-transients`, default 1, at -6
  to +12 dB): gunfire, machine gun, small arms, mortar and artillery,
  explosions near and far, thunder, shouting, dog.
- **confuser:** helicopter, truck, car, motorcycle, scooter, aircraft,
  prop plane, jet, tank, generator, idling engine, mower, chainsaw, power
  tool, siren, air-raid siren.

The drone's SNR is now set against the bed, events and confuser only.
Transients are added afterwards, so a loud transient masks the drone's
frames without turning the drone down for the whole scene. `train_clean.py`
takes no-drone scenes as negatives and skips heldout scenes.
**How.** `uv run pytest tests/test_make_scenes.py`: 3 passed (new:
`test_transients_are_outside_the_snr_reference`).
`uv run python scripts/make_scenes.py --n 4 --seed 7 --mean-transients 2 --out <scratch>`:
4 scenes rendered with Pd (3 with a drone, SNR -12.3 to +9.1 dB). All 4
are marked heldout because they use FSD50K eval stems.
**Result.** No training and no scoring.
**Assessment.** A blocker against CLAUDE.md section 5. Most of these
classes exist only as `fsd50k_eval` stems, and FSD50K eval is a held-out
test source (`docs/datasets.md`). These classes have no dev stem at all:
wind, rain, stream, traffic, birds, insects, truck, car, motorcycle,
aircraft, idling, power tool, siren, gunfire, explosion, small arms,
thunder, dog. With `--sources fsd50k_dev` there is no bed, so
`make_scenes.py` stops. There are also only 1-5 stems per class, too few
for training without memorising them. Not verified: whether iteration 10's
`~/nodrone/stems` on the server includes eval stems.
**Next.** Training stems are needed from FSD50K dev, many per class (for
example 20-50), fetched by range request the way `fetch_env_clips.py`
already reads the dev parts. This is a download of new data, so it waits
for Mads.

## 099 - Training stems from FSD50K dev, up to 30 per class
- Date: 2026-11-04
- Iteration: 11
- Type: Data
- Commit: uncommitted (on top of c83c810)

**What changed.** `scripts/build_stems.py --split dev` builds the
non-drone classes from FSD50K dev, the split that may be trained on.
- Classes with an FSD50K label use the same rules as before: CC0 or CC BY,
  labels within the class's branch, no drone words.
- Classes without an FSD50K label (helicopter, scooter, mower, chainsaw,
  generator, mosquito, buzzing insect, prop plane, jet, tank, machine gun,
  small arms, mortar and artillery, far explosions, air-raid siren) are
  picked by a keyword in the title or tags plus a fitting label. Music and
  speech labels are excluded.
- A clip goes to the first class that takes it, so no stem is in two
  classes.

The manifest now merges by stem file instead of by class, so the earlier
hand-picked stems keep their rows. `fetch_env_clips.get` waits and retries
on HTTP 429 and 503. `--resume` keeps stems already on disk.
**Why.** Entry 098: most classes existed only as `fsd50k_eval` stems, which
are a held-out test source, and with 1-5 stems per class a model would
memorise the clips. Mads approved 30 per class (2026-11-04).
**How.**
`uv run python scripts/build_stems.py --split dev --per-class 30 --skip-drones`
stopped at 756 of 795 stems on Zenodo's HTTP 429. The same command with
`--resume` built the remaining 39. Check:
`make_scenes.py --n 4 --seed 20 --sources fsd50k_dev --out <scratch>`.
**Result.**
- `data/processed/stems/manifest.csv`: 899 stems (814 `fsd50k_dev`, 61
  `fsd50k_eval`, 24 Glasgow).
- Dev stems per role:
  - bed 120 (30 each of wind, rain, stream, traffic);
  - events 102 (mosquito 10);
  - transients 240 (machine gun 14, shout 10);
  - confusers 322 (helicopter 23, scooter 20, chainsaw 15, siren 19,
    generator 9, tank 4, prop plane 3, air-raid siren 3).
- The 756 resumed stems have `denoise = resumed` and blank gate figures,
  because the first run died before writing its manifest.
- The 4 check scenes use only dev stems and the Pd drone (heldout 0).
- `throttle.txt`, which Pd rewrites on every render, was restored to the
  committed version.
**Assessment.** Training scenes can now be made without the test source.
Classes with fewer than 10 stems will be memorised more easily. Keyword
picks have not been listened to, and some may not fit their class.
**Next.** Not done here: generating the training set
(`make_scenes.py --sources fsd50k_dev`) and training on JupyterLab. Both
wait for Mads. A listening pass over a sample of keyword stems would catch
the worst picks.

## 100 - Listening files: five dev-only soundscape scenes
- Date: 2026-11-04
- Iteration: 11
- Type: Data
- Commit: uncommitted (on top of c83c810)

**What changed.** Five 10 s scenes for Mads to listen to, in
`data/processed/scenes/listen_it11/`, each with a descriptive name:
- seed 100: no drone; wind, birds, artillery, explosion, thunder.
- seed 101: drone at 313 Hz cruise, SNR -8.3 dB; stream, insects, explosion.
- seed 102: drone at 697 Hz, SNR +7.7 dB; stream, birds, insects.
- seed 103: no drone; traffic, birds, insects, gunfire.
- seed 104: drone at 554 Hz, SNR -3.9 dB; traffic, mosquito, dog.
**How.** `uv run python scripts/make_scenes.py --n 5 --seed 100 --sources fsd50k_dev --out data/processed/scenes/listen_it11`.
`throttle.txt` was restored afterwards.
**Result.** All five use only dev stems (heldout 0). None of the five
happens to have a confuser, a 0.6^5 = 8% chance at p = 0.4.
**Observation.** A transient's label runs over the whole stem, and dev
stems are up to 10 s long. An "explosion" stem can therefore cover most of
a scene (seed 100: 0.10-9.42 s), and its label marks where the stem lies,
not when the boom happens. This does not matter for the window labels,
which come only from the drone event, but it is not the event timing the
supervisor's label list implies.
**Next.** Mads listens.

## 101 - The iteration 11 training set: 6 h drone and 6 h no-drone scenes (command only)
- Date: 2026-11-04
- Iteration: 11
- Type: Data
- Commit: uncommitted (on top of c83c810)

**What changed.** `train_clean.py --drones` now takes several scene
folders. The set, as Mads asked: 2160 scenes of 10 s with a drone and 2160
without (6 h each), FSD50K dev stems only, rendered on the Mac (Pd is not
on the server) and uploaded. Both runs use seeds 0-2159, so no-drone scene
k has the same background as drone scene k without the drone, and each
pair falls on the same side of the validation split (seed >= 1800, 360
scenes of each).
**How.** Mads runs:
`uv run python scripts/make_scenes.py --n 2160 --seed 0 --p-drone 0 --sources fsd50k_dev --out data/processed/scenes/it11/nodrone`
and the same with `--p-drone 1 --out data/processed/scenes/it11/drone`.
Rendering takes about 0.4 s per drone scene on the Mac, measured on 5
scenes. Claude started the same runs, then stopped them at 2324 scenes
when Mads chose to run them himself, and deleted that partial output.
**Result.** Not run yet. Drone-labelled time will be under 6 h, because a
drone event starts at its first arriving sound; the actual figure is to
be measured on the finished set.
**Next.** Mads renders, then uploads `data/processed/scenes/it11/`, and trains on JupyterLab.

## 102 - The iteration 11 training set, checked
- Date: 2026-11-04
- Iteration: 11
- Type: Data
- Commit: uncommitted (on top of c83c810)

**What changed.** Mads rendered entry 101's two runs on the Mac:
`data/processed/scenes/it11/drone` and `.../nodrone`, 675 MB each.
**How.** Checked with a short script over the label files (not a repo
script). `throttle.txt` was restored afterwards.
**Result.**
- Both sets: 2160 scenes and 2160 WAVs, 6.00 h, seeds 0-2159 (360 validation
  scenes each at seed >= 1800), none heldout. Every non-drone stem is
  `fsd50k_dev`: 9561 stem uses per set.
- Drone set: 2160 drone events, 5.37 h drone-labelled (the rest is before
  the first arriving sound), every event at least 1 s long, SNR -20.0 to
  +10.0 dB.
- Confuser in 838 scenes (39%), at least one transient in 1384 (64%), the
  same in both sets.
- The pairing holds: a spot check of seed 5 shows the same stems in the
  drone scene and in the no-drone scene.

**Assessment.** It matches the supervisor's mix: 39% of scenes have a
confuser (asked: 30-50%), and the SNR runs -20 to +10 dB. The drone-free
share depends on how the sets are combined. In training, batches are half
drone and half no-drone (`CleanSampler`), whatever the folder sizes.
Synthetic data: no numbers from it are detection performance.
**Next.** Commit and push the code, upload `data/processed/scenes/it11/`,
and train on JupyterLab. All of these wait for Mads.

## 103 - The iteration 11 scenes uploaded to JupyterLab
- Date: 2026-11-04
- Iteration: 11
- Type: Infrastructure
- Commit: 95a395c (branch it11-soundscapes)

**What changed.** `data/processed/scenes/it11/` (entry 102) is on the
server at `~/drone-acoustic-detection/data/processed/scenes/it11/`.
**How.** The folder was zipped without compression (1 391 359 704 bytes,
MD5 8cbb0f645fd9f3fa5f0c2802de040003). The JupyterLab page in the browser
read it in 166 chunks of 8 MB from a temporary localhost server on the Mac
and wrote it with the Jupyter contents API's chunked upload (112 s).
`md5sum`, `unzip` and a file count ran in a JupyterLab terminal. The zip
and the local server were then removed.
**Result.** The MD5 on the server matches. 4320 WAVs and 4320 label files;
676 MB each in `drone/` and `nodrone/`.
**Next.** Training on JupyterLab with `train_clean.py` (entry 101) when
Mads starts it. The server repo still has to be on branch it11-soundscapes
for the new `train_clean.py`. Its current branch was not checked.

## 104 - The server repo is on branch it11-soundscapes
- Date: 2026-11-04
- Iteration: 11
- Type: Infrastructure
- Commit: 95a395c

**What changed.** Nothing. Mads confirmed that the server repo is on
it11-soundscapes, which answers the open point in entry 103.
**Next.** Training (entry 101's command), when Mads starts it.

## 105 - Correction: iteration 10 trained on FSD50K eval stems; train_clean now skips them
- Date: 2026-11-04
- Iteration: 11 (correction to iteration 10)
- Type: Correction
- Commit: uncommitted (on top of 95a395c)

**What changed.** `load_nodrone` in `scripts/train_clean.py` skips stems.csv
rows with source `fsd50k_eval`.
**Why.** Checked on the server: `~/nodrone/stems.csv` lists 61
`fsd50k_eval` stems. That is the stem set from before entry 099, and every
one of its non-drone FSD50K stems is from eval. Iteration 10
(`artifacts/it10_clean/`) took its no-drone clips from `~/nodrone`, so it
trained on a held-out test source (CLAUDE.md section 5, `docs/datasets.md`).
Any score of an iteration 10 model on FSD50K eval audio is therefore not a
clean held-out figure. The real recordings in `~/nodrone/recordings` are
not affected.
**Also found.** The server's Python environment is `~/dad/.venv`. `~/dad`
is an older copy of the repo without git. The current code is in
`~/drone-acoustic-detection` (branch it11-soundscapes), which has no
`.venv`, so the training command uses `~/dad/.venv/bin/python` with the
repo's scripts. `train_clean.py` puts the repo's own `src/` first on the
path.
**Result.** No run. `py_compile` passes.
**Next.** Commit and push, `git pull` on the server, then train.

## Soundscape line, part 2 (entries written after the 14:42 merge)

## 101 - Soundscape line: a per-drone fingerprint and a recording chain (`--realism`)
- Date: 2026-11-04
- Iteration: the soundscape line on branch it11-soundscapes (artifacts `it11_*`). It is not
  master's iteration 11. Its earlier entries were dropped by the 14:42 merge of master
  (c96f8d4) and are still to be restored; they survive in commit 850d1f6.
- Type: Data
- Commit: uncommitted (on top of c96f8d4)

**What changed.**
1. `DroneSoundHarm.pd` gets four receivers, `det0`..`det3`, wired to the
   right inlets of the motors' detune `+` objects. `render.render_dry(...,
   detune=)` sets them through the qlist. Without it, the patch keeps
   0, +2.3, -3.7 and +5.1 Hz.
2. `drone_sim.make_scene` takes three optional extras:
   - `detune`;
   - prop noise: white noise band-passed at `noise_fc` with Q `noise_q`,
     following the throttle, added at `noise_db` against the motor tone at
     1 m, from its own rng;
   - a ground reflection: the image source mirrored in flat ground
     `mic_h` below the microphone, `refl_db` below the direct path.
3. `make_scenes.py --realism` draws those extras per drone:
   - detune U(-5, +5)% of the cruise pitch for each motor;
   - noise U(-25, -3) dB, log-U(300, 4000) Hz, Q U(0.5, 2);
   - mic height U(0.5, 3) m, reflection U(-8, -1) dB.

   It also applies a recording chain per seed to the whole mix:
   - a high-pass at log-U(30, 200) Hz;
   - a smooth EQ of U(-6, +6) dB at 6 log-even points over 50-8000 Hz;
   - a decaying-noise reverb with RT60 U(0.05, 0.5) s at U(-25, -8) dB;

   then levels the mix again. The chain comes from `rng([seed, 2])`, so the
   drone and no-drone scenes of one seed share it, and the chain cannot
   tell the label. The label JSON records the drone extras and `chain`.
**Why.** Mads asked for both changes after the it11 CNN did badly on his
GUI run of `sealed_peek` clips (10 real ARIS drone excerpts with the war
track at 0 dB):
- the CNN alarmed on 7 of 10 clips, with a mean of 14% of windows flagged
  (median 5%);
- comb alarmed on 4 of 10 (mean 4%);
- neither alarmed on the war-only rows.

Hypotheses, not tested: every Pd drone had the same rotor detune (so the
same beating) and the same noise band, and no drone passed through a
microphone, a room or a ground reflection. Mads declined a measurement on
the real development rigs. **Test-hygiene note:** `sealed_peek` holds
excerpts of ITU ARIS, a sealed test source (docs/datasets.md). Its result
is what prompted this change, so this design step was steered by a look at
sealed test audio (CLAUDE.md section 5).
**How.** `uv run pytest tests/test_make_scenes.py`: 5 passed (new:
`test_chain_is_seeded_and_keeps_the_level_rule`,
`test_realism_detune_scales_with_pitch`).

Pd check: a steady throttle-60 render with `falloff 4`.
- Fixed detune: one motor peak in 180-320 Hz, at 237.1 Hz.
- Detune (0, +40, -40, +80) Hz: four peaks, at 195.8, 234.5, 274.9 and
  315.0 Hz.

Smoke render: `make_scenes.py --realism --n 3 --seed 300 --p-drone {1,0}
--sources fsd50k_dev --out data/processed/scenes/listen_realism/p{1,0}`.
Each pair shares its chain. Detunes are -24.5 to +12.0 Hz at f0 316-516 Hz.
**Result.** No training and no scoring.
**Assessment.** These are scaffolding changes. The hypothesis is that the
model then learns "a drone" rather than "this patch". It is to be judged
on real drones, and preferably on a real development set rather than on
the sealed ARIS excerpts.
**Next.** Mads listens. Then, if he wants: render 6 h + 6 h with
`--realism`, upload and train three seeds as before.

## 102 - Listening check of `--realism`: the drone stays audible
- Date: 2026-11-04
- Iteration: the soundscape line (see entry 101)
- Type: Data
- Commit: uncommitted

**What changed.** Nothing in the code.
**How.** Mads found the three entry 101 scenes hard to hear: seeds 300-302
at SNR +7.8, -3.9 and +7.6 dB, each with a bed, events, transients and in
two a confuser. An A/B for seed 300, in
`data/processed/scenes/listen_ab/`:
- A: the drone alone, as before;
- B: the same drone alone with `--realism`;
- C: a `--realism` scene at +10 dB.
**Result.** Mads: B and C are good. The realism changes do not bury the
drone. The difficulty came from the soundscape at those SNRs, which is
intended by the -20 to +10 dB range.
**Next.** Ranges unchanged. Render the training set with `--realism` when
Mads says so.

## 103 - The `--realism` training set, uploaded, and three seeds training
- Date: 2026-11-04
- Iteration: the soundscape line (see entry 101)
- Type: Data, Model
- Commit: rendering uncommitted (on top of c96f8d4); training on the server at 850d1f6

**What changed.** Mads rendered `data/processed/scenes/it11r/` on the Mac.
It uses the commands of the first set with `--realism` added: seeds
0-2159, `--p-drone 0` and `1`, `--sources fsd50k_dev`. It was then
uploaded and trained, at Mads's request.
**Check.** Label files, checked with a short script:
- 2160 + 2160 scenes, 6.00 h each, 5.38 h of drone, none heldout, every
  non-drone stem from `fsd50k_dev`;
- confusers in 838 scenes (39%);
- every scene has a chain, all 2160 drone/no-drone pairs share theirs, and
  all 2160 drones have their own detune.

`throttle.txt` was restored.
**Upload.** As in the first set: an uncompressed zip of 1 392 648 171 bytes,
MD5 1f59e8fa4734a42559f8c843ccd44edb. It went in 167 chunks of 8 MB in
106 s, and the MD5 on the server matches. 4320 WAVs and 4320 labels, at
`~/drone-acoustic-detection/data/processed/scenes/it11r/`.
**Training.** Started at 15:25, one TITAN X per seed, named
`it11r_s{0,1,2}`, with the same command and defaults as the it11 runs. The
server's `train_clean.py` at 850d1f6 is all training needs; `--realism`
only matters for rendering.
- Split: 1800 drone scenes and 2053 no-drone clips for training;
  validation of 5956 drone and 15 839 no-drone windows.
- `comb` on this validation: AUC 0.598, recall 0.151 at 5% false alarms.

The validation set differs from the it11 one (it has the chain and the
fingerprints), so the two cannot be compared as a gain or a loss.
**Next.** Collect the results and the checkpoints, then add an "it11r CNN"
to the GUIs if Mads wants.

## 104 - `--realism` results (validation only), checkpoints copied, "it11r CNN" in the GUIs
- Date: 2026-11-04
- Iteration: the soundscape line (see entry 101)
- Type: Model, Infrastructure
- Commit: uncommitted (on top of c96f8d4)

**Result.** Entry 103's three runs finished at 15:34. Validation: 5956
drone and 15 839 no-drone windows, 1 s window, 0.5 s hop.

| Model | Seed | AUC | Recall at 5% FA | FA at 95% recall | Threshold |
|---|---|---|---|---|---|
| comb | (untrained) | 0.598 | 0.151 | 1.000 | 7.020 |
| lgbm | 0 / 1 / 2 | 0.681 / 0.699 / 0.641 | 0.237 / 0.253 / 0.241 | 0.856 / 0.821 / 0.891 | 0.68-0.69 |
| dscnn | 0 / 1 / 2 | 0.824 / 0.817 / 0.813 | 0.374 / 0.382 / 0.374 | 0.551 / 0.574 / 0.576 | 0.622 / 0.644 / 0.635 |

DS-CNN training took 491-500 s per seed. The loss at 50k / 100k / 150k /
200k windows was 0.645-0.651 / 0.609-0.611 / 0.594-0.598 / 0.584-0.589,
still falling at the end.

The validation set is not the one of the runs without `--realism`, which
reached dscnn AUC 0.812-0.826. It now holds the recording chain and the
per-drone fingerprints, so the two cannot be compared as a gain or a loss.
**Assessment.** Synthetic validation, so this is not detection performance
(CLAUDE.md section 5). The DS-CNN beats both baselines on every seed, by
AUC +0.12 to +0.17 over lgbm, and its seeds agree (0.813-0.824). Whether
`--realism` helps on real drones is untested.
**Checkpoints.** The 12 files (`it11r_s{0,1,2}` `.json`, `.log`,
`_dscnn.pt`, `_lgbm.pkl`) were copied from the server into
`artifacts/it11r_realism/` as in the first set, and all 12 MD5s match. The
JupyterLab page posted them to a localhost receiver that accepts only
these names.
**GUIs.** `drone_war_grid.MODELS` has a new "it11r CNN" entry:
`artifacts/it11r_realism/it11r_s0_dscnn.pt`, threshold 0.6224 from
`it11r_s0.json`. It sits next to "it11 CNN", which this branch added
before (entry still to be restored, see entry 101). `detect_gui.py` lists
it. In `drone_war_gui.py`, six names no longer fit the horizontal model
row, so the picker became a column at (0.19, 0.005), next to Flush plots.
The load check and the offscreen renders of both windows show all six.
**Next.** Mads tries "it11r CNN" against "it11 CNN" in the GUI. A test on
a real development set would be the fair comparison.

## 105 - Mads's GUI comparison on the sealed_peek clips: `--realism` helps, synthetic training still lags
- Date: 2026-11-04
- Iteration: the soundscape line (see entry 101)
- Type: Evaluation
- Commit: uncommitted

**What changed.** Nothing. Mads ran `detect_gui.py` three times on the same
batch:
- the 10 `sealed_peek` drone clips (ITU ARIS excerpts), each mixed with
  the war track at 0 dB;
- the Dummy clip;
- the "DUMMY war track" row.

One run each with "it11r CNN", "it11 CNN" and "it9 CRNN". The figures below
are read from his screenshot of the three windows (% of windows alarmed).
**Result.** Drone clips 10 to 01, in that order:

| Model | Drone_10 ... Drone_01 | Clips alarmed | Mean | Dummy rows |
|---|---|---|---|---|
| it11 CNN | 11, 0, 1, 25, 0, 0, 3, 64, 5, 26 | 7 / 10 | 13.5% | 0% |
| it11r CNN | 14, 1, 40, 25, 1, 15, 12, 63, 11, 49 | 10 / 10 | 23.1% | 0% |
| it9 CRNN | 62, 45, 82, 79, 67, 77, 19, 74, 41, 37 | 10 / 10 | 58.3% | 0% |
| comb | 0, 0, 0, 0, 0, 22, 8, 0, 1, 5 | 4 / 10 | 3.6% | 0% |

**Assessment.** These are ten clips from one sealed source, under one
noise condition, so this is indicative and not a test result. The share of
alarmed windows is not recall against a label: the clips have no sidecar,
and the drone is taken to be present throughout.
- `--realism` almost doubled the it11 CNN's alarmed share and found every
  clip.
- The model trained on real drones (it9 CRNN, entry 096) still alarms
  2.5 times as often.
- None of them alarmed on the war-only rows.

**Test-hygiene note:** this is the second design discussion steered by the
sealed ARIS excerpts (see entry 101). A final verdict needs data that has
not been looked at.
**Next.** For Mads to decide:
- (A) allow real drones in training beside the synthetic ones, which needs
  a change to REQUIREMENTS.md;
- (B) set the threshold on real, unsealed development data first, with no
  training;
- (C) a CRNN on the synthetic set, which needs a council first;
- (D) more realistic synthesis.

Claude recommends B first.

## 106 - Council: real Glasgow drones in training next to the Pd drones
- Date: 2026-11-04
- Iteration: the soundscape line (see entry 101)
- Type: Council
- Commit: uncommitted

**Question.** Should the next model train on real drone recordings next
to the synthetic Pd drones? Mads chose this (option A, entry 105). The
alternatives were:
- (a) keep synthetic-only training;
- (b) add the 24-drone Glasgow set as drone sources in the same
  `--realism` soundscapes;
- (c) set the threshold on real development data first;
- (d) a CRNN.

Constraints: off campus, the GPU server cannot be reached and Mads allows
training on the Mac (M1 Pro, MPS). The real field rigs and real no-drone
audio are not on the Mac. Glasgow, FSD50K stems, the war track and the 10
ARIS excerpts are.

**Devil's Advocate (against (b)).**
- *Decisive, partly answered.* There is no uncontaminated real validation
  set on the Mac, so a gain could only be seen on the sealed ARIS
  excerpts, which have already steered two decisions. Answer: held-out
  Glasgow drones become the validation and test sources, the success rule
  is fixed below before anything runs, and ARIS is scored once at the end.
  What stands: the Glasgow figures are in-domain, so the out-of-domain
  verdict rests on 10 clips (about +-10 points of noise) until a real
  field rig is scored on campus.
- *Decisive, answered by design.* Every real positive carries the Glasgow
  room and microphone, and no negative does (the iteration 1 failure
  mode). Answer: the Glasgow drone-free room recording
  (`noise/_uav_2022noise_n_noise_1m_.WAV`, 60 min) goes into the negatives
  through the same chain. More than 5% alarms on it, at the frozen
  threshold, counts as a leak and fails the experiment.
- *Serious, answered.* It overturns a data rule from the day before,
  before Glasgow has been audited for the near-silent clips that motivated
  the rule. Answer: Mads makes the rule change himself (below), and an
  audit of the 256 clips' levels and silence comes before rendering.
- *Serious, stands, stated.* A split by drone still shares one session.
  Glasgow validation and test therefore measure "unseen drone, seen room"
  and are not R2/R3 evidence.
- *Serious, partly answered.* Pushing near-field 1 m audio out to range is
  physically doubtful. Answer: the 5 m recordings are used too. Whether it
  transfers is exactly what the ARIS check measures.

**Scholar (context-free).** Synthetic data for scale plus real data to
anchor the model is standard practice: DESED/Scaper soundscapes, and
multi-condition training with simulated RIRs and noise (Ko et al., ICASSP
2017). Option (b) is standard on three conditions: split by drone model,
weight the small real set, and report synthetic-only (a) beside it as an
ablation. A threshold from synthetic validation does not carry over to
real data (Ovadia et al., NeurIPS 2019), so thresholds belong on a real,
in-domain development set. (d) is a defensible variation but no remedy
for domain shift, and changing data and architecture at once confounds
the result. Unsettled: the synthetic-to-real mix ratio, and whether
re-propagated near-field indoor drones stand in for far-field outdoor
ones. Sources as the Scholar reported them; their checking was its own.
**No departure from best practice in the decision.**

**Problem Solver.** Try a modified version of (b).
- Mix, do not replace: 50% of drone scenes take a Glasgow hover clip as
  the dry source in `drone_sim.make_scene`, in place of `render_dry`.
- Split by drone: train d1-d16, validation d17-d20, test d21-d24.
- Run A (it11r recipe, no real drones) against run B (identical except
  the Glasgow drones), 3 seeds each, on the Mac. `train_clean.py`'s
  `--nodrone` becomes optional.
- Step 0 is option (c) for free: score the it11r models on validation
  scenes with d17-d20 drones. If the AUC is high and only the recall is
  low, the threshold is the problem.

**Decision (Claude).** The modified (b), as the Problem Solver laid it
out, with the Devil's Advocate's conditions built in. Success rule, fixed
now, at the threshold frozen on Glasgow validation (5% false alarms):
- *Success:* on the ARIS excerpts with the war track at 0 dB, B alarms on
  at least 45% of windows (about twice it11r's 23.1%), with at most 5%
  false alarms on the war-only rows, on the Glasgow room recording and on
  FSD50K-eval no-drone stems.
- *Failure:* B at or below 30% on ARIS. Also failure: B is high on Glasgow
  d21-d24 while ARIS does not move (it learned Glasgow).
- *Leak:* more than 5% alarms on the Glasgow room recording, whatever else
  holds.

Run A's figures, comb and LightGBM are reported alongside. Everything is
scored once on ARIS and on Glasgow d21-d24, and each scoring is logged.
No CRNN in this step.

**Needed from Mads before training:** approve the change to the data rule
in REQUIREMENTS.md. Mac training is already approved (2026-11-04).

## 107 - Requirements: real Glasgow drones allowed in training
- Date: 2026-11-04
- Iteration: the soundscape line (see entry 101)
- Type: Requirements
- Commit: uncommitted

**What changed.** At Mads's word ("Yes, change it") after entry 106's
council, the REQUIREMENTS.md row "Synthetic training data" now allows
real drone recordings from the Glasgow set in training as well as
synthetic drones: drones d1-d16 train, d17-d20 validate, d21-d24 test.
Other real drone recordings stay for validation and test, with the FPV
clips as test. Before, training could use only synthetic drones (entry
087).
**Why.** Entry 105: models trained only on synthetic drones alarm on
13-23% of windows of the ARIS excerpts, against 58% for a model trained
on real rigs.
**Next.** Entry 106's plan: audit the Glasgow clips, then build, render
and train runs A and B on the Mac.

## 108 - Glasgow audit; real drones and the room recording in `make_scenes.py`; `--nodrone` optional
- Date: 2026-11-04
- Iteration: the soundscape line (see entry 101)
- Type: Data
- Commit: uncommitted

**Audit (entry 106's condition).** All 256 Glasgow hover recordings are
44.1 kHz:
- 24 drones; 16 drones have 10 files and 8 have 12; each file 120-148 s;
- RMS at 1 m -54.9 to -51.5 dBFS, at 5 m -58.3 to -55.0;
- no 0.5 s frame more than 30 dB below its clip's loudest, digital zeros
  at most 1%, no clip more than 10 dB below the median.

So none has the near-silent defect that motivated the synthetic-only rule.
**A finding about the room recording.** It is 60.5 min at -56.5 dBFS. In
Welch spectra (8192 points), the drone clips are quieter than the room
recording in every band below 2 kHz: 1 m clips -20.8 dB at 50-200 Hz,
-7.9 dB at 200-500 Hz, -8.8 dB at 500-1000 Hz, -2.6 dB at 1-2 kHz. They
are louder only at 2-4 kHz (+3.3 dB) and 4-8 kHz (+11.4 dB); the 5 m clips
are 1.5-4 dB lower. The room recording was therefore made at another gain
or under other conditions than the background inside the drone clips. As a
leak probe it tests the room's spectral shape, not the exact background
under the drones. That weakens entry 106's leak check, which still stands
as the only one available here.
**What changed.**
1. `drone_sim.make_scene(..., dry=)`: a real 44.1 kHz recording replaces
   the Pd render. It is looped from a random start (rng `[seed, 5]`) and
   then flown the same way.
2. `make_scenes.py --real-drones DIR --p-real P`: a drone scene flies a
   Glasgow hover clip (1 m or 5 m) with probability P (rng `[seed, 4]`, so
   a Pd scene is identical with or without the option).
   - The drone number comes from `--drone-ids` (default 1-16), or from
     `--val-drone-ids` (17-20) for seeds >= `--val-seed` (1800, as in
     `train_clean.py`).
   - Overlapping id sets are refused.
   - Real drones keep `--realism`'s ground reflection and chain, but not
     its detune or prop noise.
3. `--room-noise FILE --room-db` (default -20 to -5 dB against a stem):
   every scene, drone or not, gets an excerpt of the room recording (rng
   `[seed, 3]`). It is part of the SNR reference.
4. `train_clean.py --nodrone` is optional. Without it, the negatives are
   the no-drone scenes only, because the real no-drone recordings are on
   the server.
**How.** `uv run pytest tests/test_make_scenes.py`: 6 passed (new:
`test_drone_ids_and_the_glasgow_split_by_seed`). Smoke render, seeds
1798-1801 with `--p-real 1`:
- training seeds flew d15 and d2, validation seeds flew d17;
- every drone scene's other events and chain equal its no-drone
  partner's;
- 4 scenes rendered in 8 s.
**Next.** Render, in `data/processed/scenes/itG/`:
- `nodrone`: seeds 0-2159, shared by A and B;
- `droneA`: Pd only;
- `droneB`: `--p-real 0.5`;
- `test_drone`: seeds 5000-5359, all Glasgow d21-d24;
- `test_nodrone`: seeds 5000-5359.

All with `--realism --room-noise ... --sources fsd50k_dev`. Then time
training on MPS.

## 109 - The A and B training sets rendered; Pd retries raised to 6
- Date: 2026-11-04
- Iteration: the soundscape line (see entry 101)
- Type: Data, Bug fix
- Commit: uncommitted

**Result.** In `data/processed/scenes/itG/`, all with `--realism`
`--room-noise` and `--sources fsd50k_dev`:

| Set | Seeds | Scenes | Drones |
|---|---|---|---|
| `nodrone` (shared by A and B) | 0-2159 | 2160 | none |
| `droneB` | 0-2159 | 2160 | 1058 Glasgow, 1102 Pd |
| `droneA` + `droneA2` | 0-2159 | 2160 | all Pd |
| `test_drone` | 5000-5359 | 360 | all Glasgow, d21-d24 only |
| `test_nodrone` | 5000-5359 | 360 | none |

- In `droneB`, training seeds use only d1-d16 and validation seeds
  (>= 1800) only d17-d20.
- Every scene holds a room excerpt, and none is heldout.
- The drone and no-drone scenes of a seed share their bed and chain.

**Bug.** The first `droneA` run stopped at scene 62:
`render.py` raised "Pd render stayed short after 3 attempts" (267 008 of
515 970 samples). Pd quit before `writesf~` had written the file, while
Claude's MPS timing run was loading the Mac. `render_dry` now tries 6
times. The remaining seeds 62-2159 went to `droneA2` with nothing else
running, needing one retry. A covers every seed 0-2159 exactly once
(checked from the labels).
**MPS timing** (not a result): `train_clean.py --device mps` on the it11r
set, `--limit 1850 --windows 51200`, with a render running at the same
time, trained the DS-CNN at 575 windows/s. The TITAN X ran at about 400.
**Next.** Train A_s0-2 and B_s0-2 on the Mac (two at a time,
`artifacts/itG/`), with `--nodrone` left out (the real no-drone
recordings are on the server). Then score once with
`scripts/score_itG.py`.

## 110 - Runs A and B trained on the Mac; the test scoring stopped unfinished
- Date: 2026-11-05
- Iteration: the soundscape line (see entry 101)
- Type: Model, Evaluation
- Commit: uncommitted

**Training.** `train_clean.py --device mps` without `--nodrone`, two runs at
a time, 3 seeds each, in `artifacts/itG/`. Each run took 467-508 s.
Validation was 5956 drone and 6840 no-drone windows.

| Run | Validation | Seed | dscnn AUC | dscnn recall at 5% FA | lgbm AUC | comb AUC |
|---|---|---|---|---|---|---|
| A (Pd only) | Pd scenes | 0 / 1 / 2 | 0.691 / 0.686 / 0.691 | 0.31-0.32 | 0.64 | 0.545 |
| B (half Glasgow) | half Glasgow d17-d20 | 0 / 1 / 2 | 0.730 / 0.728 / 0.732 | 0.39 | 0.69 | 0.529 |

A and B validate on different drones, so the gap is not a test result.
"itG-A CNN" and "itG-B CNN" (seed 0) are in both GUIs, with a note on
what the model names mean.
**Test scoring.** `scripts/score_itG.py` did not finish:
1. A pipe hid a crash. Attempt 1 ended with no output after loading the
   sets.
2. Attempt 2 showed exit 139. LightGBM segfaults on macOS when torch is
   already loaded (two OpenMP runtimes). The scorer now scores LightGBM
   before any network.
3. Attempt 3 was still running after 12.5 h, with 34 min of CPU, because
   the Mac slept. Mads stopped it ("Forget it", 2026-11-05).

No test figure was seen, except one: while debugging, comb's alarm share
on the Glasgow room recording at threshold 8.84 was 0.047 on the first
5 min and 0.0067 on the whole hour. **Entry 106's verdict is therefore
open.** The test sets have not been scored by any model.

## 111 - CRNN on the it11r scenes: three seeds training on JupyterLab
- Date: 2026-11-05
- Iteration: the soundscape line (see entry 101)
- Type: Model
- Commit: uncommitted. The server runs 850d1f6 plus this `train_clean.py`,
  copied by the contents API (MD5 d14194338f43992a55ea75b2f2cd8e69).

**What changed.** `train_clean.py --arch {dscnn,crnn}`, default dscnn. It
saves `<name>_<arch>.pt` and `models.<arch>` in the JSON. The CRNN is
`dad.models.zoo.Crnn`, the same architecture master's entry 096 trained on
real rigs (739 747 parameters). Seeds 0-2 started at 09:30 on the three
TITAN X, named `it11r_crnn_s{0,1,2}`. Everything else is entry 103's it11r
command: same scenes, `--nodrone ~/nodrone`, defaults. So the architecture
is the only change, and the it11r DS-CNN validation (entry 104) compares
directly.
**Why.** Mads chose option 2 of 2026-11-05: the CRNN, now that he is on
campus. Entry 086's council already covered the supervisor's CRNN proposal
("a CRNN against score smoothing"). Entry 106's Scholar advised testing it
without changing the data at the same time, which is why only the
architecture changes. No new council: the architecture is not new to the
project.
**Check.** A local smoke run of 2048 windows on MPS trained, scored and
saved `_crnn.pt` (140 windows/s). On the server all three logs show
entry 103's split: 1800 drone scenes, 2053 no-drone clips, validation
5956 / 15 839 windows, comb AUC 0.598.
**Next.** Collect the three results, copy the checkpoints, and add
"it11r CRNN" to the GUIs.

## 112 - CRNN on the it11r scenes: validation results; "it11r CRNN" in the GUIs
- Date: 2026-11-05
- Iteration: the soundscape line (see entry 101)
- Type: Model
- Commit: uncommitted

**Result.** Entry 111's runs finished at 09:39 after 200 000 windows each.
Validation was the same as the it11r DS-CNN's (entry 104): 5956 drone and
15 839 no-drone windows, 1 s window, 0.5 s hop.

| Model | Seed 0 / 1 / 2 AUC | Recall at 5% FA | Threshold |
|---|---|---|---|
| it11r CRNN | 0.775 / 0.798 / 0.811 | 0.329 / 0.355 / 0.346 | 0.702 / 0.641 / 0.659 |
| it11r DS-CNN (entry 104) | 0.824 / 0.817 / 0.813 | 0.374 / 0.382 / 0.374 | 0.622 / 0.644 / 0.635 |
| lgbm (same runs) | 0.681 / 0.699 / 0.641 | 0.24-0.25 | |
| comb | 0.598 | 0.151 | |

- CRNN training took about 504 s per seed (seed 0).
- The loss at 50k / 100k / 150k / 200k windows was 0.645 / 0.626 / 0.607 /
  0.592 for seed 0, still falling at the end and higher than the DS-CNN's
  0.584-0.589.
**Assessment.** Synthetic validation (CLAUDE.md section 5). On it, the
CRNN is a little below the DS-CNN (AUC -0.002 to -0.049) and varies more
between seeds. Its rising-then-falling loss suggests it is under-trained
at 200 000 windows. Whether it does better on real drones, as it9's CRNN
did against it9's CNN on the ARIS excerpts (entry 105), is untested.
**Checkpoints and GUIs.** The 12 files are in `artifacts/it11r_crnn/`, and
all MD5s match the server's. "it11r CRNN" (seed 0, threshold 0.7024) is
in `drone_war_grid.MODELS`, so both GUIs list it. The name note in both
windows explains it. Offscreen renders of both windows show all nine
models.

## 113 - CRNN on the it11r scenes with 3x the training windows (started)
- Date: 2026-11-05
- Iteration: the soundscape line (see entry 101)
- Type: Tuning
- Commit: uncommitted (server: 850d1f6 + entry 111's `train_clean.py`)

**What changed.** Only `--windows 600000` instead of 200 000. Everything
else is the same as entry 111. Seeds 0-2 started at 09:46, one TITAN X
each, named `it11r_crnn600k_s{0,1,2}`. The logs show the same split and
comb as entries 103 and 111.
**Why.** Entry 112: the CRNN's loss was still falling at 200 000 windows
(0.607 to 0.592 over the last 50 000), so it was probably under-trained.
Mads asked to train it longer.
**Judged by.** The same validation as entry 112, compared with the CRNN
at 200 000 windows and with the it11r DS-CNN. Synthetic validation, so
this is not detection performance.

## 114 - CRNN at 600 000 windows: lower training loss, same validation AUC
- Date: 2026-11-05
- Iteration: the soundscape line (see entry 101)
- Type: Tuning
- Commit: uncommitted

**Result.** Entry 113's runs finished at 10:10-10:14. Validation is the
same as in entries 104 and 112.

| Model | Windows | AUC seed 0 / 1 / 2 (mean) | Recall at 5% FA | Threshold |
|---|---|---|---|---|
| it11r CRNN | 600 000 | 0.803 / 0.787 / 0.793 (0.794) | 0.367 / 0.370 / 0.374 | 0.631 / 0.663 / 0.631 |
| it11r CRNN | 200 000 | 0.775 / 0.798 / 0.811 (0.795) | 0.329 / 0.355 / 0.346 | |
| it11r DS-CNN | 200 000 | 0.824 / 0.817 / 0.813 (0.818) | 0.374 / 0.382 / 0.374 | |

- Training took 1366-1561 s per seed, at about 400 windows/s.
- Training loss at 600 000 windows: 0.551-0.556, against 0.592 at 200 000.
**Assessment.** Synthetic validation (CLAUDE.md section 5).
- Three times the training lowered the training loss a lot but left the
  validation AUC unchanged (mean 0.794 against 0.795).
- Recall at 5% false alarms rose a little (+0.02 to +0.04, all seeds),
  matching the DS-CNN's.
- The extra training is fitting the training scenes, not generalising
  further. The CRNN still does not beat the DS-CNN on this validation.
- Real-drone behaviour is untested.
**Checkpoints and GUIs.** The 12 files are in `artifacts/it11r_crnn600k/`,
and all MD5s match the server's. "it11r CRNN 600k" (seed 0, threshold
0.6311) is in both GUIs; the name note says "600k: 3x trained".
`detect_gui.py`'s model box grew from 0.12 to 0.17 of the window height,
and the file list moved up, so ten names fit. An offscreen render with
one scene loaded and scored with the new model shows the layout and the
scoring.

## 115 - Mads's drone-in-war run with "it11r CRNN 600k"; silent windows now score 0
- Date: 2026-11-05
- Iteration: the soundscape line (see entry 101)
- Type: Evaluation, Algorithm
- Commit: uncommitted

**Observation (Mads's screenshot of `drone_war_gui.py`).** Setup:
war-shaped noise, seed 1000; Pd drones at 100, 150 and 200 km/h; war at
-40, -20 and 0 dB plus a no-war column and a no-drone row; recall and
false alarms are % of 1 s windows.

| War | it11r CRNN 600k recall / false+ / >=3 s | comb recall / false+ / >=3 s |
|---|---|---|
| no war | 100.0 / 0.0 / 3 | 92.6 / 0.0 / 3 |
| -40 dB | 25.9 / 0.3 / 0 | 30.2 / 3.7 / 3 |
| -20 dB | 7.9 / 0.3 / 0 | 4.2 / 3.7 / 3 |
| 0 dB | 1.6 / 0.3 / 0 | 3.2 / 3.7 / 3 |

Mads: better than the earlier models, but still bad.
1. In war noise alone the CRNN scores 0.4-0.5, just under its threshold
   (0.631). It does not separate war-shaped noise from a drone, so it
   alarms only in the last 1-2 s before impact. No speed reaches R5's 3 s
   at any war level.
2. After impact the clip is digital silence, and the score sits flat at
   0.34: "confident there is a drone when there is no audio".

Synthetic test: Pd drones in war-shaped noise, so this is not detection
performance.
**Cause and fix of 2.** No model has seen silence: `train_clean.py`
skips windows below `realrun.SILENCE_POWER`, and level normalisation turns
silence into amplified rounding noise. `dad.wavio.stream`, used by both
GUIs and `drone_war_grid.py`, now sets the score of a window below that
in-band power to 0. Check with the it11r CRNN 600k:
- 3 s of white noise scored 0.334 / 0.305 / 0.335;
- the 3 s of zeros after it scored 0 / 0 / 0, with no alarm.

`uv run pytest -q tests`: all passed.
**Causes of 1 (hypotheses, not tested).**
- The it11r scenes hold no steady broadband noise: war-shaped noise was
  removed from training on purpose (entry 097, it is the test background).
  Master's iteration 11 ("steady noise as negatives", entry 100) targets
  exactly this, with random-slope noise, not the war spectrum.
- The training SNR range is -20 to +10 dB, while a drone 3 s before impact
  (about 170 m at 200 km/h) sits far below its noise.
**Also.** In `drone_war_gui.py` the model-name note overlapped the
results table. It moved to y 0.17 at font size 6.

## 116 - Two arms against entry 115's failures: steady noise, and steady noise with fainter drones
- Date: 2026-11-05
- Iteration: the soundscape line (see entry 101)
- Type: Data
- Commit: accf0e4 (pushed; the server pulled it)

**What changed.** `make_scenes.py --steady P --steady-db LO HI` (default
-6 to +12 dB): with probability P (rng `[seed, 7]`), a scene gets a
steady broadband noise bed made of:
- Gaussian noise with a random spectral slope, U(-12, +3) dB/octave;
- a smooth random EQ, U(-6, +6) dB at 6 log-even points over 50-8000 Hz;
- a slow level wobble, 0.2-2 Hz and +-3 dB.

The noise is rng `[seed, 6]` and the level rng `[seed, 8]`, so the drone
and no-drone scenes of a seed get the same noise. It is part of the SNR
reference. It is never the war spectrum, which stays the test background
(entry 097). Master's iteration 11 (entry 100) uses the same family.
New test `test_steady_noise_slope_is_seeded`: the noise is seeded and has
unit RMS, and a slope below -9 dB/octave puts more than 10 times the
power at 100-400 Hz than at 3-6 kHz.
**The arms.** Both start from the it11r recipe (`--realism --sources
fsd50k_dev`, seeds 0-2159) and both get `--steady 0.5`. Both share
`it11s/nodrone`, because a no-drone scene does not depend on the SNR range.
- Arm 1 (`it11s/drone`): SNR -20 to +10 dB, as before.
- Arm 2 (`it11sl/drone`): SNR -30 to +10 dB, so far drones are seen
  (entry 115's second failure). This stays above the supervisor's
  -40 dB floor.

Both train as the it11r CRNN 600k (entry 113), 3 seeds each, so arm 1
against entry 113 measures the steady noise, and arm 2 against arm 1
measures the SNR range.
**Where it runs.** Mads asked for everything to be tested on the server.
The server has no Pd: no sudo, and conda-forge does not respond, while
PyPI does. The numpy port `pdsynth` would change the renderer, a second
change. So scenes are rendered on the Mac, and training and all tests run
on the server.
**Server tests so far.** After `git pull` to accf0e4,
`~/dad/.venv/bin/python -m pytest -q tests` on the server: all passed, 1
skipped (the Glasgow split test; Glasgow is not on the server).

## 117 - Both arms rendered, uploaded and training on JupyterLab (no tests, Mads's instruction)
- Date: 2026-11-05
- Iteration: the soundscape line (see entry 101)
- Type: Data, Model
- Commit: accf0e4 (server and Mac)

**Render (Mac).** Started at 10:30, done by 10:47. Each set holds every
seed 0-2159 once, none heldout, and 1063 scenes (49%) with steady noise:
- `it11s/nodrone`;
- `it11s/drone`, SNR -20.0 to +10.0 dB;
- `it11sl/drone`, SNR -30.0 to +10.0 dB.

In both arms, all 2160 drone scenes share their bed and chain with
`it11s/nodrone`. `throttle.txt` was restored.
**Upload.** One uncompressed zip of 2 090 502 342 bytes, MD5
323100f53ef3b4beee3e33cde83c58c2, in 250 chunks of 8 MB (150 s). The MD5
on the server matches, and 2160 WAVs and 2160 labels per set are in place.
**Training.** At about 10:52 the six runs started:
`it11s_crnn600k_s{0,1,2}` and `it11sl_crnn600k_s{0,1,2}`. The command is
entry 113's (`--arch crnn --windows 600000 --nodrone ~/nodrone`), with
`--drones <arm>/drone it11s/nodrone`, two runs per TITAN X. Another
user's jobs held about 5.4 GB on each GPU at the time.
**Tests.** Mads said "Only train dont test": no drone-war test or other
scoring runs on the server for these models. Mads tests them himself in
the GUIs.
## 118 - Merge of master (1fb0db7) into it11-soundscapes, and this section
- Date: 2026-11-05
- Iteration: the soundscape line
- Type: Infrastructure, Correction
- Commit: the merge commit that adds this entry

**What changed.**
1. `origin/master` (6f37520, ad25f45, 1fb0db7) was merged into
   it11-soundscapes. REPORT.md merged by union with no conflict markers,
   but it left this line's entries 101-117 in the middle of master's
   iteration 11 (after entry 100) and the entries dropped on 2026-11-04
   still missing. Both are fixed by this section: part 2 was moved here
   unchanged, and part 1 was restored. Master's text is untouched.
2. `scripts/detect_gui.py` conflicted in three places.
   - Master's tick boxes (several detectors at once, comb as a box) are
     kept.
   - This branch's models, the note on their names and the taller detector
     box (17% of the height, file list from 20%) are kept, as are the macOS
     file dialog in a thread and `start_batch`, which now passes master's
     `models` and `comb`.
   - `MODEL_COLORS` had 6 colours for 10 models; it now has 10, with an
     assert.
3. The "Iterations at a glance" table has a row for this line.
**Check.** Headless `detect_gui.py`: a picked file scored with "it11r CRNN
600k", "it9 CRNN" and comb, then drew the window. `uv run pytest -q tests`
is in the merge commit's log.


## 119 - The arms restarted with the frontend on the GPU
- Date: 2026-11-05
- Iteration: the soundscape line
- Type: Infrastructure
- Commit: ce75020 (server and Mac)

**What changed.** `train_clean.CleanSampler.raw_batch` returns a batch's
windows before the frontend: no speed change (`SPEED_UP`), zero EQ, and
the labels. `NetScorer` uses it on CUDA and runs the log-mel on the GPU
(`gpufront`, master's entry 092). `batch()` now builds on the same draws.
**Why.** At 11:03 the six runs of entry 117 trained at 110 windows/s,
about 90 min for 600 000 windows. The server's load average was 68 on 24
cores, with another user's jobs as well, while the GPUs were 17-45% busy.
The CPU frontend was the bottleneck. Mads chose to switch at once ("No
switch now i mean").
**Check.** On a 16-window batch from the it11r scenes, `batch()` and
`GpuFrontend("cpu").train_features(raw_batch())` agree to a maximum
absolute difference of 1.9e-6 (mean |L| 3.2). So the switch changes speed,
not what the model sees, and the comparison with the it11r CRNN 600k
holds. `uv run pytest -q tests`: all passed.
**What happened.** The six runs of entry 117 were stopped at about 50 000
windows each and restarted at 11:06 with the same names and settings.
Their earlier logs are overwritten.

## 120 - The steady-noise arms: validation results; "it11s CRNN" and "it11sl CRNN" in the GUIs
- Date: 2026-11-05
- Iteration: the soundscape line
- Type: Model
- Commit: uncommitted (on top of ce75020)

**Result.** Entry 119's six runs finished at 11:33-11:36: CRNN, 600 000
windows, frontend on the GPU, 1288-1384 s of training each. The
validation is each arm's own: scenes with seed >= 1800 and the real
nightjar rig.

| Arm | Seed | CRNN AUC | Recall at 5% FA | FA at 95% recall | Threshold | lgbm AUC | Final loss |
|---|---|---|---|---|---|---|---|
| 1, steady noise (`it11s`) | 0 / 1 / 2 | 0.788 / 0.800 / 0.807 | 0.409 / 0.412 / 0.413 | 0.805 / 0.735 / 0.670 | 0.618 / 0.626 / 0.622 | 0.670 / 0.734 / 0.682 | 0.539 / 0.536 / 0.542 |
| 2, + drones to -30 dB (`it11sl`) | 0 / 1 / 2 | 0.748 / 0.720 / 0.742 | 0.307 / 0.306 / 0.306 | 0.793 / 0.892 / 0.839 | 0.623 / 0.645 / 0.619 | 0.640 / 0.729 / 0.604 | 0.583 / 0.579 / 0.584 |

comb: AUC 0.589 (arm 1), 0.579 (arm 2).
**Assessment.** Synthetic validation (CLAUDE.md section 5). The three
validation sets (it11r, arm 1, arm 2) differ, so the arms cannot be ranked
against it11r or against each other from these numbers:
- arm 1's validation adds steady noise;
- arm 2's validation also holds drones down to -30 dB, which are harder by
  construction. That is why its AUC and recall are lower.

What does compare:
- in both arms the CRNN beats lgbm and comb on every seed;
- arm 1's recall at 5% FA (0.41) is above the it11r CRNN 600k's (0.37),
  even though its validation is harder.

Whether either arm separates war-shaped noise from drones, and alarms
earlier (entry 115's failures), is for Mads to test in `drone_war_gui.py`.
Mads asked for no tests here.
**Checkpoints and GUIs.** The 24 files are in `artifacts/it11s_steady/`.
The combined MD5 over the 24 per-file MD5s matches the server's
(750b8017d0165287637699d9eca44f4d).
- "it11s CRNN" (threshold 0.6184) and "it11sl CRNN" (0.6233), seed 0, are
  in `drone_war_grid.MODELS`, so both GUIs list them.
- `detect_gui.py` has two more model colours, a taller detector box (21%
  of the height, file list from 24%) and the note at x 0.155.
- `drone_war_gui.py`'s model column is 0.24 tall.

Offscreen renders of both windows show all 12 models.

## 121 - Merge of master (4ad0c1a) into it11-soundscapes
- Date: 2026-11-05
- Iteration: the soundscape line
- Type: Infrastructure
- Commit: the merge commit that adds this entry

**What changed.** Nicolai's 4ad0c1a ("Add complete test GUI and log-mel
views": `Complete_Test_Gui.py`, `complete_test_engine.py`, a log-mel view
in `detect_gui.py`, entries 121-122 and a "Note on numbering" in master's
part) was merged at Mads's request. Git merged `detect_gui.py` and
REPORT.md without conflicts. The union merge put master's note and entry
122 inside this section (before entry 118 and after entry 120). Both were
moved back to where master has them: the note before master's entry 101,
and 122 at the end of master's part. This section again holds only this
line's entries.
**Check.** No REPORT.md line of `origin/master` or of f1e36d9 is missing.
The other checks are in the merge commit's message.


## 123 - Why the synthetic-trained models lag it9's CRNN: a code comparison (no run)
- Date: 2026-11-05
- Iteration: the soundscape line
- Type: Analysis
- Commit: uncommitted (on top of 55b990f)

**Question.** Mads: it makes no sense that it11, it11r, it11s and the other
synthetic-trained models are so far below it9's CRNN (entry 105: 58.3% of
windows alarmed against 13.5-23.1%). The it11r CRNN (entries 111-114) uses
the same architecture (`dad.models.zoo.Crnn`), the same frontend (MFE,
64 log-mel, 1 s window, level normalisation), the same `NetScorer.fit`
(AdamW, OneCycle, SpecAugment, per-band mu/sigma), and up to 3x the
windows. Whatever separates them is in the data path, not the network.
**Method.** Read `scripts/train_clean.py` (it11 line) against
`dad.rangerun.RangeSampler` with recipe `simdist`, `--pool`, speed
0.8-1.25 (the args in `artifacts/runs/final_crnn.json`, entry 096). No
model was trained or scored.
**Differences found.**

| | it9 CRNN (`rangerun`, simdist) | it11 line (`train_clean.py`) |
|---|---|---|
| Positive source | real drones recorded at 0-50 m (pooled rigs) | Pd synthesis (it11, it11r, it11s); half Glasgow in itG-B only |
| Positive background | 80%: a real near drone summed into the **same training rig's own background** at -10..+5 dB in-band SNR; 20%: near drone buried in FSD50K | the scene's synthetic soundscape (FSD50K stems, beds, confusers), drone at -20..+10 dB vs the bed |
| Negatives | the same rigs' backgrounds (share per rig matched to positives), plus FSD50K | real no-drone recordings and stems, and no-drone scenes |
| Can "real recording vs mixed scene" predict the label? | no: each rig gives both classes | partly: a real recording window is never a positive |
| Speed (pitch/BPF) augmentation | 0.8-1.25 on every window | none (`raw_batch` passes SPEED_UP) |
| Random spectral EQ (tilt + shape, `_eq_curve`) | on every window | none (zeros); it11r adds a mic EQ inside the scene renderer instead |
| Threshold | 5% FA on real held-out rig data | 5% FA on synthetic validation (val scenes + nightjar rig) |

**Assessment (hypotheses, not results).**
1. *Source gap.* Pd drones are not real drones. Entry 110's itG-B (half
   real Glasgow drones) is the only run that touches this, and it was
   never scored on test data, so how big this gap is is still unknown.
2. *Shortcut on the recording.* In it9 the rig is uninformative by
   construction (entry 025). In the it11 line every window of a real
   recording is a negative, and every positive is a rendered scene. A
   network can lower the loss by learning "real recording texture means
   no drone". That would make it under-alarm on real drone audio, which is
   the failure entry 105 shows (alarms too rare, not too frequent).
3. *Missing augmentation.* Neither speed nor EQ, both of which it9 had on
   every window. Without speed, the network sees only the BPF range of the
   Pd presets.
4. *Threshold calibrated on synthetic validation.* Entry 105's alarm
   shares depend on it. Option B of entry 105 (re-thresholding on real
   development data) was never run.

These four are confounded in every it11-line versus it9 comparison so far.
**Next.** Change one at a time, in order of cost: (4) is no training;
(3) is two lines in `train_clean.py`; (2) means mixing the Pd drones into
real rig backgrounds as `simdist` does. Waiting for Mads to pick (no tests
unasked).

## 124 - Entry 123's fixes: speed + EQ and real-background mixing in `train_clean.py`; three CRNN arms training
- Date: 2026-11-05
- Iteration: the soundscape line
- Type: Data, Model
- Commit: uncommitted (server: ce75020 + this `train_clean.py` and `score_real_dev.py`, applied as a git
  patch, MD5s 05be6c34827c1d6bdbbc237c3584e640 and 3a4e1b2d64bb5fbabcea3a033b7e077a, matching the Mac)

**What changed.** Mads: "Do them all" (entry 123's three fixes).
- `train_clean.py --speed LO HI`: every training window is resampled by a factor from LO-HI. This is
  `rangerun`'s speed change, as it9 used it (0.8-1.25).
- `--eq`: `realrun._eq_curve` on every training window, as it9 had.
- `--mix-real P` (`--mix-snr`, default -10..+5 dB): a share P of *scene* windows is summed into a real
  no-drone recording (from `--nodrone` recordings.csv, never stems or scenes), with the scene over the
  recording in-band, as `simdist` does. This applies to drone scenes and no-drone scenes alike, so a real
  recording's texture no longer marks "no drone" and mixing does not mark "drone".
- Every option is off by default. With all of them off, the draws are the same as before, so earlier runs
  reproduce.
- New `scripts/score_real_dev.py` scores checkpoints on it9's real validation sessions (Nightjar
  flight 3, d3set Validation). It reports per-rig AUC, recall at 5/10% FA on that rig's own background,
  and each model's frozen threshold.
**Check.** A local smoke run (30 scenes, 1024 windows, CPU, all three options on) trained, scored and
saved a CRNN. On a 16-window batch with speed + EQ + mixing, `batch()` and the GPU path
(`GpuFrontend("cpu").train_features(raw_batch())`) agree to a maximum difference of 1.9e-6.
**Training.** At 13:27, one TITAN X each, CRNN, 600 000 windows, seed 0, entry 113's data and command
(`--drones it11r/drone it11r/nodrone --nodrone ~/nodrone`). The logs show entry 113's split (1800 drone
scenes, 2053 no-drone clips; validation 5956 / 15 839 windows):
- `e123_aug`: `--speed 0.8 1.25 --eq`
- `e123_mix`: `--mix-real 0.8`
- `e123_augmix`: both

The baseline is the it11r CRNN 600k seed 0 (entry 114). Each arm changes one thing against it, and
`augmix` combines both.
**Fix 1 (threshold on real audio) is blocked.** `score_real_dev.py` failed on the server: Marius's account
has no `data/processed/manifest.csv` or rig stores (Nightjar, d3set, FPV), and the Mac has none either.
it9 was trained from another account. Nothing was scored.
**Judged by.** Synthetic validation (each run's own JSON) only, until real drone data can be scored. That
is not detection performance (CLAUDE.md section 5), and it cannot show whether the shortcut of entry 123
is gone. Only real drone audio can.

## 125 - Glasgow check of the existing models: the synthetic-trained CRNNs find real Glasgow drones, it9 alarms on the empty room
- Date: 2026-11-05
- Iteration: the soundscape line
- Type: Evaluation
- Commit: uncommitted (on top of 55b990f)

**Why.** Entry 124's real-rig scoring is blocked, and Mads chose Glasgow instead.
**What.** New `scripts/score_glasgow.py`, run on the Mac (MPS):
`uv run python scripts/score_glasgow.py --device mps --out artifacts/runs/e125_glasgow_old.json "it9 CRNN=...@0.4828" "it11r CRNN 600k=...@0.6311" "it11s CRNN=...@0.6184"`.
- Positives: every hover clip of Glasgow drones d17-d20, at 1 m and 5 m. These are the validation drones
  of entry 108; the test drones d21-d24 were not read. 9796 windows.
- Negatives: the Glasgow room recording (`noise/`), 7255 windows.
- Both: 1 s windows at a 0.5 s hop, skipping digital silence.
- Thresholds: each model's frozen one (it9 from `final_crnn.json`; the others from their synthetic
  validation).

**Result.** Validation, one seed per model.

| Model | AUC | AUC 1 m / 5 m | Recall at 5% FA (room) | Frozen threshold: room FA | Frozen threshold: recall (1 m / 5 m) |
|---|---|---|---|---|---|
| comb | 0.735 | 0.707 / 0.764 | (ties; not meaningful) | | |
| it9 CRNN (entry 096) | **0.119** | 0.059 / 0.178 | 0.016 | **96.7%** | 53.2% (36.2 / 70.2) |
| it11r CRNN 600k s0 (entry 114) | 0.997 | 0.996 / 0.998 | 0.988 | 0.4% | 94.0% (93.4 / 94.7) |
| it11s CRNN s0 (entry 120) | 0.996 | 0.995 / 0.997 | 0.985 | 0.9% | 95.4% (94.0 / 96.7) |

**Assessment.**
- This weakens entry 123's hypotheses 1 and 2 *for close, clean drones*. Models that never heard a real
  drone separate real Glasgow hover drones at 1-5 m from the Glasgow room almost perfectly. At their
  synthetic thresholds they alarm on 94-95% of drone windows and on under 1% of room windows.
- it9 is the model that fails here. It ranks the empty room *above* the drones (AUC 0.12) and alarms on
  97% of the room. it9 trained on Glasgow drones as positives, and Glasgow has no negatives of its own in
  that pool (`make_pool_arm`: "Glasgow always trains"). So "Glasgow lab sound" became a drone cue: the
  same recording shortcut that entry 123 suspected in the it11 line, but here in it9.
- So the gap of entry 105 (ARIS excerpts in war noise at 0 dB) is probably less about "real versus
  synthetic drone" than about faint or masked drones. There it9's eagerness to alarm may be what helps
  it. That fits the it11 line's low alarm share there, and entry 115's late alarms.

**Caveats.**
- Entry 108: the room recording was made at another gain or setting than the background under the
  drones. Level is removed per window, but its spectral shape differs. A model could therefore separate
  "drone take" from "room take" without hearing the drone, and part of the 0.997 may be that.
- it9 has seen Glasgow drones, including d17-d20 (the pool trains on every Glasgow drone). Its recall
  here is in-distribution. Its room false alarms are not.
- Hover at 1-5 m indoors is far easier than the deployment case.

**A hygiene finding (not new to this entry).** The server's `~/nodrone/recordings.csv` holds 59 Halmstad
recordings (and 37 d3set, 121 FPV, 28 Nightjar). Halmstad is a held-out test source (REQUIREMENTS.md).
`train_clean.py` trains on every non-Nightjar row, so every it11-line model since entry 101 has trained
on Halmstad no-drone audio. Entry 124's three arms do too, kept that way so they still differ from their
baseline by one change only. Any Halmstad false-alarm figure for these models is therefore not a test
result. Excluding Halmstad in `load_nodrone`, as FSD50K eval already is, is a one-line fix and needs
Mads's go-ahead, because it changes the training set of every later run.
**Next.** Score entry 124's three arms the same way once their checkpoints are on the Mac. The Mac
could not receive them directly: a receiving port on the Mac was refused by the sandbox.

## 126 - Entry 124's three arms: no gain on synthetic validation, a loss on Glasgow
- Date: 2026-11-05
- Iteration: the soundscape line
- Type: Model, Evaluation
- Commit: uncommitted

**Runs.** Entry 124's arms finished at 13:45 (`aug`, `mix`) and 13:48 (`augmix`), after 838-1002 s of
training. Mads approved the download. The three checkpoints are in `artifacts/e123_arms/`, and their
MD5s match the server's (aug 2fbaf292..., mix 59fbc4e7..., augmix 6a0e2ca6...). The three JSONs did not
save through the browser pane and are still only on the server (`artifacts/runs/e123_*.json`); the
figures below were read from them there.
**Synthetic validation** (each run's own; the same set as the baseline, since only the training
draws changed):

| Model | CRNN AUC | Recall at 5% FA | FA at 95% recall | Threshold | Final loss | lgbm AUC |
|---|---|---|---|---|---|---|
| it11r CRNN 600k s0 (baseline, entry 114) | 0.803 | 0.367 | 0.638 | 0.631 | 0.551 | |
| `e123_aug` (speed + EQ) | 0.790 | 0.370 | 0.682 | 0.618 | 0.577 | 0.632 |
| `e123_mix` (real mixing 0.8) | 0.799 | 0.365 | 0.630 | 0.647 | 0.605 | 0.713 |
| `e123_augmix` (both) | 0.788 | 0.352 | 0.641 | 0.674 | 0.621 | 0.675 |

comb AUC 0.598 in all three.
**Glasgow** (entry 125's set-up: `uv run python scripts/score_glasgow.py --device mps --out
artifacts/e123_arms/e125_glasgow_new.json ...`; d17-d20, 9796 drone and 7255 room windows):

| Model | AUC | AUC 1 m / 5 m | Recall at 5% FA (room) | Frozen threshold: room FA | Frozen threshold: recall |
|---|---|---|---|---|---|
| it11r CRNN 600k s0 (baseline, entry 125) | 0.997 | 0.996 / 0.998 | 0.988 | 0.4% | 94.0% |
| `e123_aug` | 0.983 | 0.980 / 0.987 | 0.914 | 0.2% | 76.4% |
| `e123_mix` | 0.972 | 0.964 / 0.980 | 0.915 | 0.9% | 82.1% |
| `e123_augmix` | 0.884 | 0.847 / 0.920 | 0.763 | 0.5% | 58.8% |
| it9 CRNN (entry 125) | 0.119 | | 0.016 | 96.7% | 53.2% |

**Assessment.** One seed each, so differences of about 0.01 on synthetic validation are within the
seed spread (entry 112: 0.775-0.811).
- Neither fix, nor both together, improves anything measured here. On synthetic validation all three
  are level with the baseline or slightly below it. On Glasgow each one alone costs a little AUC and a
  lot of recall at the frozen threshold, and the two together cost the most (AUC 0.884).
- The training loss is higher in every arm (0.577-0.621 against 0.551). The augmented and mixed task is
  harder, and 600 000 windows may be too few for it. Lower recall at the frozen threshold also reflects
  higher thresholds (0.647 and 0.674 for the mixing arms).
- Glasgow was already near its ceiling for the baseline (AUC 0.997), so it can show these arms losing
  but cannot show them gaining. What they target, faint drones in real noise (entry 105), is not in this
  check.

Verdict on entry 123's hypotheses, as far as this data can tell:
- (1) source gap and (2) recording shortcut: not supported for close, clean drones (entry 125).
- (3) missing augmentation: adding it did not help.
- (4) threshold on synthetic data: still untested on real far or masked drones.

**Next.** The case that fails is a faint drone under real noise. Two ways to measure it without test
data:
- Glasgow d17-d20 mixed into the war track or real rig noise at a range of SNRs;
- Mads's `drone_war_gui.py` with these three models added.

For now, do not adopt any of the three arms.

## 127 - Entry 124's three arms in `Complete_Test_Gui.py`
- Date: 2026-11-05
- Iteration: the soundscape line
- Type: Infrastructure
- Commit: uncommitted

**What changed.** Mads asked for the three arms in the complete test GUI.
- The three run JSONs were downloaded too (Mads approved all six files). Their MD5s match the server's:
  aug 2fe4f50b..., mix 80091613..., augmix 60163ce6.... All six files are in `artifacts/e123_arms/`.
- `Complete_Test_Gui.py` adds "e123 aug CRNN", "e123 mix CRNN" and "e123 augmix CRNN" to
  `drone_war_grid.MODELS` when it starts, with their own colours. They are registered there only, so
  `detect_gui.py` and `drone_war_gui.py` keep their twelve models and their layouts. The test engine
  runs in the same process, so this is enough.
- The detector checkboxes now wrap into rows of six. With 16 detectors, a single row would no longer
  fit the window.

**Check.** Run headless:
- `detector_names()` lists the three before comb.
- `complete_test_engine.load_detectors` loads them at thresholds 0.6183, 0.6468 and 0.6741, the
  `threshold_5pct_fa` of each run.
- With the window built (1180 x 917), all checkboxes are mapped, in three rows, the last at y = 774.

The GUI was not run on a test round.

## 128 - Real-rig validation (it9's own validation sessions): the it11r CRNN ranks as well as it9 on Nightjar; its synthetic threshold is far too strict
- Date: 2026-11-05
- Iteration: the soundscape line
- Type: Evaluation
- Commit: uncommitted

**Where.** Marius allowed work in Mads's JupyterHub account (`mads`), which holds the rig stores and
`data/processed/manifest.csv` that it9 was trained from (entry 124's block). Mads's repo
(`iteration9-frontends` at 242b3b4, with his uncommitted changes) was not modified. A separate folder
`~/e123_score` has `scripts/score_real_dev.py` (MD5 3a4e1b2d..., as on the Mac), a link to his `src`,
and the five synthetic-trained checkpoints, uploaded from the Mac with MD5s matching. His it9 checkpoint
`final_crnn_none_simdist_crnn.pt` has the same MD5 as ours (1a5728b1...).
**Command.** `CUDA_VISIBLE_DEVICES=0 python -u scripts/score_real_dev.py --device cuda --out
artifacts/runs/e128_real_dev.json "<label>=<ckpt>@<frozen threshold>" ...`
(result MD5 705e9fb332f0672050330096938bbb24, on the server).
**Set.** `rangerun.make_pool_arm(manifest, "none").val`: Nightjar flight 3 and d3set Validation.
These are the sessions that froze it9's threshold. 1 s windows at a 0.5 s hop, digital silence
skipped. Per rig, the threshold is set on that rig's own background windows.
- Nightjar: 357 drone and 1482 background windows.
- d3set: 2291 drone and only 83 background windows, so 5% FA there is about 4 windows. The d3set rows
  are coarse.

**Result.** One seed per model.

| Model | Nightjar AUC | Nightjar recall at 5% / 10% FA | Nightjar at frozen thr: FA / recall | d3set AUC | d3set recall at 5% FA | d3set at frozen thr: FA / recall |
|---|---|---|---|---|---|---|
| comb | 0.516 | 0.115 / 0.120 | | 0.643 | 0.312 | |
| it9 CRNN (entry 096) | 0.759 | 0.678 / 0.695 | 5.3% / 68.1% | **0.930** | **0.823** | 0% / 80.2% |
| it11r CRNN 600k s0 (entry 114) | **0.929** | **0.770 / 0.824** | 0.3% / 29.1% | 0.884 | 0.691 | 0% / 36.7% |
| it11s CRNN s0 (entry 120) | 0.897 | 0.675 / 0.745 | 0.3% / 32.5% | 0.847 | 0.539 | 0% / 40.9% |
| e123 aug | 0.871 | 0.697 / 0.756 | 0.3% / 20.7% | 0.874 | 0.644 | 0% / 37.4% |
| e123 mix | 0.910 | 0.686 / 0.773 | 0.3% / 49.0% | 0.863 | 0.542 | 0% / 38.6% |
| e123 augmix | 0.844 | 0.686 / 0.703 | 0.1% / 34.2% | 0.879 | 0.576 | 0% / 36.3% |

**Assessment.**
1. **Ranking:** the synthetic-trained it11r CRNN 600k *matches or beats* it9 on Nightjar (AUC 0.929
   against 0.759; recall at 5% FA 0.770 against 0.678). It has never heard a real drone or any Nightjar
   audio, while it9 trained on other Nightjar sessions. On d3set it9 is ahead (0.930 against 0.884). The
   it11 models trained on d3set no-drone audio, so their d3set background may not even be unseen.
2. **Threshold** (entry 123's hypothesis 4): confirmed, and it is the big one. At its frozen synthetic
   threshold, the it11r CRNN raises almost no false alarms on real background (0.3%) but catches only
   29-37% of real drones. At a threshold set on real background at 5% FA, it catches 69-77%. it9's
   threshold was frozen on these very sessions, so it is calibrated here by construction (5.3% FA on
   Nightjar). A large part of "it9 wins" in the GUIs (entries 105, 115; Marius on 2026-11-05) is
   probably this calibration gap, not detection ability. The GUIs use each model's frozen threshold.
3. **Entry 124's arms:** none beats the it11r CRNN 600k on real rigs either. This agrees with entries
   125-126.

**Caveats.** One seed each. These sessions are validation, but they are also it9's threshold set, so
it9's frozen-threshold row is in-sample. The d3set background is 83 windows. Choosing a new threshold
for the it11 models on these sessions would be tuning on validation, which is allowed, but it then
needs a fresh check elsewhere.
**Next.** Re-threshold the it11r CRNN 600k on real background (these Nightjar and d3set sessions, at the
FA budget the GUIs use), then rerun Marius's Complete_Test_Gui comparison with that threshold.

## 129 - The it11r CRNN 600k at a real-background threshold; "it11r CRNN 600k real-thr" in Complete_Test_Gui
- Date: 2026-11-05
- Iteration: the soundscape line
- Type: Tuning (threshold on validation)
- Commit: uncommitted

**What.** Marius: re-threshold it. The rule is it9's own (`run_range.py`): `rangerun.fa_threshold` at
5% false alarms on the *pooled* background windows of `make_pool_arm(manifest, "none").val`, 1 s windows
at a 0.5 s hop. Run in Mads's account with `~/e123_score/scripts/real_threshold.py`
(`CUDA_VISIBLE_DEVICES=0 python -u scripts/real_threshold.py`).
**Check of the rule.** Applied to it9, it returns 0.48279285..., exactly the `deploy_threshold` in
`final_crnn.json`. So the rule is the one that set it9's threshold.
**Result.** 2648 drone and 1565 background windows (Nightjar and d3set pooled):

| Model | Threshold | FA on real background | Recall |
|---|---|---|---|
| it11r CRNN 600k s0, frozen synthetic threshold (entry 114) | 0.6311 | (entry 128: 0-0.3% per rig) | (entry 128: 29-37% per rig) |
| it11r CRNN 600k s0, real threshold | **0.3370** | 5.05% | 59.2% |
| it9 CRNN, same rule (= its frozen threshold) | 0.4828 | 5.05% | 78.5% |

The pooled drone windows are 87% d3set (2291 of 2648), where it9 ranks better (entry 128). On Nightjar
alone the it11r CRNN was ahead at equal FA.
**GUI.** `artifacts/e129_realthr/it11r_crnn600k_s0_realthr.json` holds the threshold. It is
`train_clean`-shaped, so `wavio.threshold_from_run` reads it, and it points at the unchanged checkpoint.
`Complete_Test_Gui.py` registers "it11r CRNN 600k real-thr" (GUI-only, as in entry 127).
`load_detectors` returns 0.6311 / 0.3370 / 0.4828 for "it11r CRNN 600k" / "... real-thr" / "it9 CRNN".
**Hygiene.** The threshold is chosen on validation sessions, which is allowed. It is it9's
threshold set, so the two models are now calibrated on the same data. Marius's Complete_Test_Gui rounds
are the fresh check. Nothing was scored on test sources.

## 130 - Marius's Complete_Test_Gui run at 14:30: on Drone_10 in the war soundscape, it6 and it9 rank far better than the it11r CRNN, also at equal false alarms
- Date: 2026-11-05
- Iteration: the soundscape line
- Type: Evaluation (reading of a GUI run)
- Commit: uncommitted

**Run.** `artifacts/complete_test/20261105_143050_real/` (Marius):
- one real drone clip, `data/extracted/sealed_peek/Drone_10.wav` (first 10 s), at 60 dB SPL;
- the war soundscape at 30, 60, 70, 90 and 120 dB, 11 starts each, plus one drone-free round per level;
- 20 s rounds, 0.125 s hop;
- detectors: it6 CNN, it9 CRNN, it11r CRNN 600k, it11r CRNN 600k real-thr (entry 129), comb.

Marius: "it6 is the best and it11 is far behind".
**At each model's own threshold** (`rounds.csv`, pooled per war level, recall / FA):

| War dB | it6 CNN | it9 CRNN | it11r CRNN 600k | it11r real-thr | comb |
|---|---|---|---|---|---|
| 30 | 0.95 / 0.000 | 0.76 / 0.000 | 0.19 / 0.000 | 0.48 / 0.030 | 0.00 / 0.008 |
| 60 | 0.61 / 0.000 | 0.55 / 0.000 | 0.13 / 0.000 | 0.47 / 0.054 | 0.01 / 0.008 |
| 70 | 0.44 / 0.000 | 0.40 / 0.000 | 0.13 / 0.000 | 0.38 / 0.054 | 0.01 / 0.008 |
| 90 | 0.16 / 0.000 | 0.14 / 0.000 | 0.07 / 0.000 | 0.26 / 0.054 | 0.00 / 0.008 |
| 120 | 0.05 / 0.000 | 0.03 / 0.000 | 0.02 / 0.000 | 0.16 / 0.054 | 0.00 / 0.008 |

**At equal false alarms.** Re-read from `scores.npz`:
- Windows wholly inside the drone's 10 s count as drone; windows wholly outside it, and every window
  of the drone-free rounds, count as no drone. Straddling windows are left out, as in the engine.
- Per level, each model's threshold is set at 1% FA on that level's drone-free windows (893 at
  120 dB; 803 drone windows).

| War dB | it6 CNN AUC / recall | it9 CRNN AUC / recall | it11r CRNN 600k AUC / recall | comb AUC / recall |
|---|---|---|---|---|
| 30 | 1.000 / 1.00 | 1.000 / 1.00 | 0.850 / 0.32 | 0.961 / 0.00 |
| 60 | 0.977 / 0.78 | 0.986 / 0.92 | 0.867 / 0.26 | 0.960 / 0.01 |
| 70 | 0.919 / 0.60 | 0.921 / 0.72 | 0.804 / 0.19 | 0.961 / 0.01 |
| 90 | 0.746 / 0.30 | 0.709 / 0.32 | 0.708 / 0.10 | 0.960 / 0.00 |
| 120 | 0.614 / 0.07 | 0.578 / 0.08 | 0.601 / 0.02 | 0.960 / 0.00 |

The real-thr variant is the same network, so its ranking is identical to the it11r CRNN 600k's.
**Assessment.**
- Marius is right, and here the threshold is not the explanation. On this drone the synthetic-trained
  CRNN ranks worse than it6 and it9 even with the war track almost silent (30 dB: AUC 0.85 against
  1.00). At 1% FA it catches about a third as many windows as they do.
- Entry 129's real threshold only trades false alarms for recall: about 5% FA, which it6 and it9 do
  not pay here.
- So the it11r CRNN separates *some* real drones well (Glasgow hover, entry 125; Nightjar, entry 128)
  but not Drone_10. Entry 123's hypothesis 1 (the Pd drones lack something real drones have) stands for
  this kind of drone.
- comb's AUC (0.96 at every level) with near-zero recall at 1% FA means its scores are tied or flat on
  the drone-free windows. Its AUC is not comparable.

**Hygiene.** One drone clip, so this is anecdotal. Drone_10 is a `sealed_peek` ITU ARIS excerpt, the
third design discussion steered by it (entries 101, 105). Decisions taken on it need confirming on
drones that have not been looked at.

## 131 - Council: drop the synthetic-only line and build on it9's real-drone recipe?
- Date: 2026-11-05
- Iteration: the soundscape line (decision on its future)
- Type: Council
- Commit: uncommitted

**Question.** Should the project (A) abandon synthetic-only training and build on it9's real-drone
recipe (`run_range.py --pool`, simdist: real drones from several rigs summed into the same rig's
background, speed 0.8-1.25, EQ), adding what the it11 line contributed (steady and war-like noise,
soundscape backgrounds, Glasgow no-drone audio as negatives)? The alternatives are (B), keep the current
approach (Pd drones in soundscapes, plus Glasgow d1-d16), and (C), a hybrid of real data plus synthetic
scenes, or it9's weights fine-tuned on scenes. The evidence given was entries 105 and 123-130, with
REQUIREMENTS.md rows R2, R3 and "Synthetic training data". The last of these (Mads, entry 107) allows only
synthetic drones and Glasgow d1-d16 in training, so A needs a requirements change.

**Devil's Advocate.**
- Decisive 1: *source confounded with label.* The it9 recipe puts each rig's channel into every positive,
  and Glasgow proves the risk (it9 AUC 0.119, 97% FA on the empty room; entry 125). Adding Glasgow
  negatives patches only the rig that happened to be caught.
  **Answer:** partly answered by design. Every rig that gives drone positives must also give its own
  no-drone audio. DDL, DroneNoise and Glasgow have none in it9's pool, and this is the mechanism to remove,
  not a single patch. The Glasgow room probe becomes a hard gate. If an A model still fails it, the
  objection stands and A is rejected.
- Decisive 2: *the case for A rests on sealed ARIS data and one drone.* **Answer:** this objection stands
  in part. The non-sealed evidence is mixed: d3set favours it9 (0.930 against 0.884), Nightjar favours
  it11r (0.929 against 0.759), and Glasgow favours it11r. The evidence clearly for A is ARIS (entries 105,
  130), which has now steered four discussions. From here, ARIS is not used for decisions until a finished
  candidate is scored. Gates use non-sealed probes only.
- Serious objections:
  - *The rule change removes validation and test rigs.* Answered by protocol: leave-one-rig-out within
    validation, and DroneNoise, Halmstad, FSD50K-eval and FPV stay test.
  - *One seed.* Answered: at least 2 seeds per arm.
  - *Tiny d3set background (83 windows).* Answered: the threshold is pooled per rig, weighted, not by
    window count.
  - *itG-B (half real Glasgow, allowed now) never scored.* Answered: a compliant arm, with Glasgow
    d1-d16 as the only real drones, runs next to A and shows what the extra rigs add.
  - *Time cost.* This stands; it is the price of A.

**Scholar** (no project context; it checked the sources below, abstracts only, and I have not
re-verified them).
- Best practice is (b): real target audio from as many rigs and drone types as possible, augmented by
  mixing into real background at controlled SNR, pitch/speed shift, EQ and SpecAugment, plus hard
  negatives. On top of that, pretraining on a broad audio corpus (AudioSet/PANNs, Kong et al. 2020).
  Reuter, Ohlenbusch & Rollwage (arXiv 2605.31329) find that pretraining is the biggest lever for
  out-of-domain robustness.
- Synthetic Scaper-style soundscapes are mainstream as a complement, with a known synthetic-to-real gap
  (Serizel et al. 2020; Turpault & Serizel 2020).
- Procedurally synthesised *target* sound as the main source: no established practice found. (a) is
  *against practice*, (b) is *standard*, (c) is a *defensible variation* only as an ablation.
- Evaluation: leave-one-source-out or an external out-of-domain corpus, TPR at a fixed FPR measured on
  large realistic negatives, TPR by SNR or distance, thresholds on training-source validation.
- Unsettled: whether synthetic target audio helps beyond generic pretraining, and log-mel against PCEN
  under heavy noise (Vilhurin et al., arXiv 2608.14287).

**Problem Solver.** Verdict: *try a modified version of A.* Start from the `simdist_steady` recipe, then
add Glasgow no-drone audio, war-like noise as negatives and masks, and the soundscape backgrounds as extra
negatives. Remove Halmstad from every training pool. Keep the CRNN and the frontend (data change only).
- Cheapest experiment: "it9+neg", 2 seeds, about 20 minutes, beside it9 and an it11r reference without
  Halmstad, all thresholds frozen on validation. Gates:
  - (a) Glasgow d17-d20 against the room: AUC ≥ 0.97 and FA ≤ 5% (required);
  - (b) Nightjar recall at 5% FA ≥ 0.77;
  - (c) d3set recall at 5% FA ≥ 0.80 (required);
  - (d) war test within 5 points of it9.
- Success means (a) and (c) pass, plus (b) or (d). If (a) fails, balance negatives per rig instead of
  going back to synthetic.
- Fallback if Mads refuses the rule change: the same recipe with Glasgow d1-d16 as the only real drones.
- Confidence: medium.

**Weighing.**
- The Scholar and the Problem Solver agree: real drones at the core, synthetic material as negatives and
  diversity at most. No departure from best practice is proposed.
- The Problem Solver's gate (d) uses the war GUI, which runs on sealed ARIS clips. It is replaced by a
  non-sealed war check: Glasgow d17-d20 mixed into the war track. ARIS is kept for the final scoring.
- The Scholar's strongest lever, pretraining on a broad corpus, is a new model family or frontend change.
  It needs its own council and is noted as the next candidate, not part of this decision.

**Decision.** Recommend modified A to Mads. Its training data needs his change to the "Synthetic training
data" requirement: real drones from DDL, Nightjar and d3set (training sessions only), and Glasgow d1-d16, may train. FPV stays test.
Until he decides:
1. The synthetic line is frozen (no further tuning).
2. The Halmstad leak is fixed in every training list, with a check that no test source reaches training.
3. The compliant arm (it9's recipe plus the it11 negatives, Glasgow d1-d16 as the only real drones) may
   run, since it needs no rule change.

Gates as above, with (d) replaced, at least 2 seeds, and leave-one-rig-out validation. The sealed ARIS
clips are not used again until a finished candidate is scored.

## 132 - Test-source leak fixed in `train_clean.py`; correction to entries 125 and 131: it9 never trained on Glasgow
- Date: 2026-11-05
- Iteration: the soundscape line
- Type: Bug fix, Correction
- Commit: uncommitted

**Leak fix.** Entry 125 found the Halmstad no-drone recordings in `~/nodrone`. A second test source was
there too: 121 FPV no-drone recordings, and REQUIREMENTS.md makes the FPV clips test.
- `train_clean.load_nodrone` now skips every row whose source is in `TEST_SOURCES = {halmstad, fpv,
  dronenoise}`. FSD50K eval stems were already skipped.
- New `tests/test_train_clean.py` checks this (`uv run pytest -q tests/test_train_clean.py`: 1 passed).
- Every it11-line model so far, including entry 124's arms and the it11r and it11s CRNNs, trained on
  Halmstad and FPV no-drone audio. Their Halmstad or FPV false-alarm figures, if ever taken, are not
  test results. No model was retrained here.

**Correction.** Mads's `data/processed/manifest.csv`, the data it9 was trained from, read in his account
on 2026-11-05, holds no Glasgow rows. Its sources are d3set, ddl, dronenoise, fpv, fsd50k, halmstad, lund
and nightjar. `make_pool_arm` adds Glasgow only when the manifest has it, so **it9 never trained on
Glasgow**.
- Entry 125's explanation ("it9 trained on Glasgow drones as positives ... so 'Glasgow lab sound' became
  a drone cue") is wrong.
- What stands is the measurement: it9 ranks the Glasgow empty room above the Glasgow drones (AUC 0.119)
  and alarms on 97% of it. The room and the drones were *both* unseen. So it9 false-alarms on an unseen
  quiet room recording, which is a generalisation failure, not a learned microphone cue.
- In entry 131, the Devil's Advocate's decisive objection 1 rested on that wrong explanation, and so did
  my answer to it ("every rig that gives drone positives must also give its own no-drone audio").
  - Restated, the objection is weaker but still serious: it9 alarms on an unseen room, and the it9 recipe
    has not shown it can avoid that.
  - The Glasgow room gate (a) stays. It now tests generalisation to an unseen room, not removal of a
    learned cue.
- Also wrong in entry 128: "it9 has seen Glasgow drones, including d17-d20". It has not, so its
  Glasgow drone recall in entry 125 is not in-distribution either.

**Consequence for the compliant arm (entry 131).** Its only allowed real drones, Glasgow d1-d16, exist
on the Mac and not on the server where the rig backgrounds are. The Glasgow audio has to be moved first,
or the arm built differently.

## 133 - The compliant arm: Glasgow d1-d16 as the only real drones, mixed into real backgrounds (started); gate set-up and reference scores
- Date: 2026-11-05
- Iteration: the soundscape line (entry 131's compliant arm)
- Type: Data, Model
- Commit: uncommitted (server: ce75020 + entry 124's and 132's `train_clean.py`, MD5 a8669939...)

**Data.** Allowed now (REQUIREMENTS.md, entry 107): Glasgow d1-d16 train.
- Hover clips are stationary, so a 30 s excerpt was taken from each of the 176 d1-d16 hover files
  (random start, rng 132) instead of all 5.96 h, resampled to 16 kHz. Moving all 687 MB through the
  browser's 10 MB uploads was not practical.
- Added: the first half (30.2 min) of the Glasgow room recording as one no-drone clip, so "Glasgow room"
  cannot mean "drone". Gate (a) uses only the second half.
- Path: Mac → tar (MD5 504805bd...) → Mads's account in 11 uploads → shared `/tmp` → Marius's account,
  where the MD5 matched. Now `data/processed/scenes/glasgow_d1_16/` holds 177 WAVs and labels.
- Validation drones: `scenes/it11s_valdrone/`, symlinks to the 360 it11s drone scenes with seed ≥ 1800.

**Runs** (14:53, CRNN, 600 000 windows, `--nodrone ~/nodrone`, test sources excluded by entry 132's fix:
1873 no-drone clips instead of 2053):
- `e133_glasgow_s0` / `_s1` (GPUs 0, 1):
  `--drones scenes/glasgow_d1_16 scenes/it11s_valdrone scenes/it11s/nodrone --speed 0.8 1.25 --eq
  --mix-real 0.8`.
  - Positives: only Glasgow d1-d16, 80% summed into real rig recordings at -10..+5 dB (simdist-like),
    with speed and EQ as in it9.
  - Negatives: real Nightjar, d3set and stems, plus the it11s soundscape scenes (steady noise in half of
    them) and the room half.
- `e133_it11s_clean_s0` (GPU 2): the it11s command of entry 117, now without Halmstad and FPV, as the
  leak-free reference.

**Gate set-up.** `scripts/score_glasgow.py` gained two options:
- `--room-second-half`;
- `--war-snr`: every d17-d20 drone window is mixed into a random window of
  `data/processed/war/war_track.wav` at the given in-band SNR. These mixes are scored against
  war-track windows alone, with recall at 1% FA on those.

This war check (d) replaces the sealed ARIS war test (entry 131). Caveat: the it11 line has `stem:war`
stems in training, so the war track may not be fully unseen by it.
**Reference scores**:
`uv run python scripts/score_glasgow.py --device mps --room-second-half --war-snr 0 -5 -10 --out
artifacts/runs/e133_gates_ref.json ...`. There are 9796 drone and 3627 room windows (second half), and
the war-track windows come from the whole war track.

| Model | (a) room AUC | room FA at frozen thr | recall at frozen thr | war AUC 0 / -5 / -10 dB | war recall at 1% FA, 0 / -5 / -10 dB | war FA at frozen thr |
|---|---|---|---|---|---|---|
| it6 CNN* | 1.000 | 11.6% | 100% | 1.000 / 1.000 / 1.000 | 1.00 / 1.00 / 1.00 | 1.3% |
| it9 CRNN | 0.092 | 99.2% | 53.2% | 0.999 / 0.999 / 0.995 | 0.985 / 0.971 / 0.937 | 0.8% |
| it11r CRNN 600k | 0.998 | 0.3% | 94.0% | 0.996 / 0.995 / 0.993 | 0.876 / 0.839 / 0.787 | 0.6% |
| it11s CRNN | 0.997 | 0.8% | 95.4% | 0.998 / 0.997 / 0.995 | 0.952 / 0.930 / 0.895 | 0.3% |
| comb | 0.737 | | | 0.679 / 0.646 / 0.612 | 0.166 / 0.133 / 0.097 | |

\* **it6 trained on every Glasgow drone**, including d17-d20 and the test drones d21-d24 (iteration 6
box: "Nightjar, d3set, the three FPV campaigns, Glasgow, DDL and DroneNoise"), and on DroneNoise, a test
source. Its Glasgow figures here are in-training and say nothing about generalisation. Any Glasgow
d21-d24 or DroneNoise score of it6 is not a test result either. This bears on Marius's "it6 is the best"
(entry 130): whether Drone_10 resembles a source it6 trained on is unknown.

**Reading of the references.**
- it9 alarms on almost all of the unseen Glasgow room (99%; correction in entry 132).
- On Glasgow drones in the war track, though, it9 is the best of the models that never heard Glasgow:
  93.7% at -10 dB against 89.5% (it11s) and 78.7% (it11r) at 1% FA on the war track.
- Gate (a) separates the models: it9 fails it, and the it11 models pass. Gate (d) favours it9.

## 134 - The compliant arm fails its gates: Glasgow-only real drones overfit to the Glasgow recordings
- Date: 2026-11-06 (runs finished 2026-11-05, 15:06-15:16)
- Iteration: the soundscape line (entry 131's compliant arm)
- Type: Model, Evaluation (validation only)
- Commit: uncommitted

**Files.** Marius asked for the download.
- Checkpoints and run JSONs are in `artifacts/e133_arms/`. The checkpoint MD5s match the server's:
  glasgow s0 67d685db..., s1 aaa0f7ec..., it11s clean 5e435b0d....
- The real-rig scoring ran in Mads's account, with the checkpoints passed through `/tmp/e133`:
  - `~/e123_score/artifacts/runs/e133_real_dev.json`;
  - `e133_real_threshold.json` (`scripts/real_threshold.py`, it9's rule);
  - Glasgow gates: `artifacts/e133_arms/e133_gates_new.json`.

**Training.**
- The Glasgow arm's loss fell to 0.003 (s0) and 0.003 (s1); the it11s CRNNs ended near 0.55.
- On their own synthetic validation (Pd drone scenes), the Glasgow arms score CRNN AUC 0.560 / 0.420,
  with a 5% FA threshold of 8e-5 / 1e-4. Their scores collapse to about 0 for anything that does not
  sound like a Glasgow recording.
- it11s clean (leak-free): AUC 0.796 and recall at 5% FA 0.399, about the same as it11s with the leak
  (0.788-0.807, entry 120).

**Gates** (entry 131). The frozen threshold is each model's real-background threshold (5% FA pooled,
it9's rule).

| Model | (a) Glasgow room AUC / FA at thr | (b) Nightjar AUC / recall at 5% FA | (c) d3set AUC / recall at 5% FA | Pooled real recall at 5% FA (thr) | (d) war recall at 1% FA, 0 / -5 / -10 dB |
|---|---|---|---|---|---|
| e133 glasgow s0 | 1.000 / 99.9% | 0.713 / 0.650 | 0.812 / 0.582 | 71.0% (4.7e-6) | 1.00 / 1.00 / 0.999 |
| e133 glasgow s1 | 1.000 / 99.2% | 0.755 / 0.672 | 0.807 / 0.586 | 70.3% (1.3e-5) | 1.00 / 1.00 / 0.999 |
| e133 it11s clean | 1.000 / 8.7% | 0.890 / 0.678 | 0.847 / 0.557 | 58.3% (0.373) | 0.963 / 0.938 / 0.880 |
| it11r CRNN 600k (entries 128, 129, 133) | 0.998 / | 0.929 / 0.770 | 0.884 / 0.691 | 59.2% (0.337) | 0.876 / 0.839 / 0.787 |
| it9 CRNN (same) | 0.092 / 99.2% | 0.759 / 0.678 | 0.930 / 0.823 | 78.5% (0.483) | 0.985 / 0.971 / 0.937 |
| Gate | AUC ≥ 0.97, FA ≤ 5% | recall ≥ 0.77 | recall ≥ 0.80 | | within 5 points of it9 |

**Verdict: the compliant arm fails.**
- It fails (b) and (c), the decisive real-rig gates.
- It fails (a) on false alarms: at the threshold real rigs need, it alarms on 99% of the Glasgow room.
- Its perfect (a) AUC and (d) war scores are uninformative. d17-d20 come from the same Glasgow rig and
  room as the d1-d16 it trained on, which is exactly the shortcut the Devil's Advocate warned of
  (entry 131).
- The near-zero loss, the near-zero thresholds and the collapsed synthetic validation all show a model
  that learned "a Glasgow recording" more than "a drone".

**What it does show.** Ranked on the pooled real rigs at 5% FA, even this overfit model catches more
drones than either synthetic-trained CRNN (70-71% against 58-59%), from one indoor rig of real drones.
it9, trained on real drones from several outdoor rigs, is still well ahead at 78.5%. Under the current
rule (Glasgow d1-d16 the only real drones), real drone audio helps, but one rig is not enough.
**Leak fix.** Removing Halmstad and FPV changed it11s by about one seed's spread (Nightjar AUC 0.890
against 0.897; recall at 5% FA 0.678 against 0.675). The earlier conclusions do not depend on the leak.
**Next.** This is the evidence entry 131 said Mads needs. The compliant route stalls at Glasgow's single
rig. The next step is his decision on the "Synthetic training data" requirement: real drones from DDL,
Nightjar and d3set training sessions in training, with FPV, DroneNoise, Halmstad and FSD50K-eval kept as
test. Until then, no further training on this line.

# Iteration 11.1 - AudioSet-pretrained encoder (PANNs Cnn14_16k)

> **Changed from the soundscape line (entries 133-134):** the network only. The CRNN trained from scratch
> is replaced by the AudioSet-pretrained PANNs Cnn14_16k: frozen with a linear probe (P1), then fine-tuned
> from that probe (P2, LP-FT). P0 scores its 527 AudioSet outputs with no training.
> **Why:** entry 134. Under entry 107's rule, the only real training drones are Glasgow d1-d16, from one
> indoor rig. From scratch, the model learned "a Glasgow recording", not "a drone". It failed gates (a), (b)
> and (c), with a near-zero loss and near-zero thresholds. Pretraining is the established lever for unseen
> recording chains: the entry 053 and 131 councils, and Reuter, Ohlenbusch & Rollwage, arXiv 2605.31329.
> The latter was read in full on 2026-11-06. It reports out-of-domain TPR at 1% FPR of 0.701 ± 0.015 from
> scratch against 0.783 ± 0.014 with AudioSet pretraining, over 10 runs, with full fine-tuning of their own
> SE-ResNet.
> **Kept the same:** the e133 compliant training data, split and augmentation, by importing
> `train_clean.py`. That is Glasgow d1-d16, the it11s scenes, real rig negatives, speed 0.8-1.25, EQ,
> `--mix-real 0.8`, and the entry 132 test-source exclusion. Also kept: the validation split
> (train_clean's `--val-seed 1800` drones against the Nightjar no-drone clips), 1 s windows and 0.5 s
> evaluation hop.
> **Hypothesis:** a general-audio prior stops the single-rig shortcut. On real rigs, the pretrained arm
> beats the e133 from-scratch arm by more than the seed spread, and passes the room gate.
> **Judged by:** entry 131's gates plus entry 053's mower gate, fixed in entry 135 before any run.

## 135 - Requirements: Mads accepts the pretrained-encoder plan; entry 054's deferral is lifted
- Date: 2026-11-06
- Iteration: 11.1
- Type: Requirements
- Commit: uncommitted (branch 11.1, from 55b990f)

**What changed.**
- A written plan was put to Mads: P0, P1 and P2 on the e133 data, the gates below, and four open decisions.
  In reply he wrote "Im mads i accept it".
- He was then asked to approve each of the four decisions one by one. He answered "use your best judgment".
- Earlier in the same conversation the user was taken to be Marius, from the Mac account. Mads identified
  himself at this point. This is recorded as stated.
- Read as follows:
  1. **Entry 054's deferral ("embeddings wait until after the field test") is lifted.** The report does not
     record whether the field test has happened.
  2. **AudioSet-pretrained weights are allowed as initialisation** under the "Synthetic training data" row
     (entry 107). They contain no project audio and no test source. PANNs are trained on AudioSet, which is
     YouTube; FSD50K is Freesound. The training audio itself is unchanged.
  3. **The new code and the checkpoint download are approved:**
     - the Cnn14 port (reference code MIT);
     - `Cnn14_16k_mAP=0.438.pth` from Zenodo record 3987831, weights CC BY 4.0 according to several mirrors.
       The record itself could not be opened from the Mac. The download is about 330 MB (a mirror's figure,
       not verified), fetched on the server.
  4. **No new council**, my judgment as delegated:
     - Entry 053's council decided this arm: frozen `Cnn14_16k` plus a probe, with LP-FT on `Cnn14_16k` as
       the next step if the probe shows the encoder helps.
     - P2 is bound to that condition: it runs only if P1 beats the e133 from-scratch arm on gate (b) or (c).
     - The gates below are entry 131's council's, plus entry 053's mower gate.
     - Departure: entry 053's Lund gate and its iteration-5-arm comparison are replaced by entry 131's gates.
       Entry 107 changed the training data, so the iteration-5 arms are no longer the reference. The e133
       arms are, being on the same data.
- Also from this conversation: all training runs on the JupyterLab server and none on the Mac. Server:
  3× TITAN X (Pascal, 12 GB), fp32, internet access, no sshd.
- REQUIREMENTS.md was updated to match: two Data rows and the training-hardware row.

**Pass/fail rule, fixed now, before any run.**
- Frozen threshold: each model's real-background threshold (5% FA pooled per rig, it9's rule, entry 129).
  It is computed with `real_threshold.py`, which is in Mads's account and not in the repo.
- Gates:

  | Gate | Pass |
  |---|---|
  | (a) Glasgow room, second half | AUC ≥ 0.97 and FA ≤ 5% at the frozen threshold |
  | (b) Nightjar | recall ≥ 0.77 at 5% FA |
  | (c) d3set | recall ≥ 0.80 at 5% FA (required) |
  | (d) Glasgow d17-d20 in the war track | recall at 1% FA within 5 points of it9 at 0 / -5 / -10 dB |
  | (e) Mower-only windows of `nightjar-lawnmower1-20260805` | FA ≤ 25% at the frozen threshold |

- **Success:** (a) and (c) pass, plus (b) or (d), and (e) is no worse than it9. For P2 this must hold on
  at least 2 of 3 seeds.
- **Informative failure:** better than the e133 Glasgow arms (entry 134) on (b) and (c) by more than their
  seed spread, without passing.
- **Rejection:** not better than e133 on (b) and (c).
- **Also reported:**
  - entry 053's rig-ID probe (how well the rig is predicted from the embedding), next to the same probe on
    the CRNN's penultimate features;
  - the `comb` baseline, and the e133 `lgbm` baseline from the same data and split.
- **Sealed data stays sealed.** ARIS, RODENT and the held-out test sources are scored once, for a candidate
  that passes.

**Next.** Entry 136: the code. Then the server stages fetch, test, smoke, P0 and P1.

## 136 - The Cnn14_16k port, its scorer, scripts and tests (branch 11.1; no run on the server)
- Date: 2026-11-06
- Iteration: 11.1
- Type: Infrastructure
- Commit: uncommitted (branch 11.1, from 55b990f)

**What changed.**
- `src/dad/models/panns.py`: Cnn14 ported from the reference code, with module names that load the
  checkpoint with `strict=True`. Also its own pretraining frontend in plain torch and numpy (the repo's
  no-librosa rule):
  - 512/160-sample periodic-Hann STFT, centred, reflect-padded;
  - power spectrum, the Slaney mel filterbank with 64 bands over 50-8000 Hz, then 10·log10 with amin 1e-10.
- **Level.** Each window's mean log-mel (dB over time and bands) is set to the mean of the checkpoint's `bn0`
  running mean. This is my choice, not taken from the paper and not tuned. It makes loudness say nothing,
  as the repo's "level" variant does, and keeps the input at the encoder's training level.
- **EQ.** The training EQ curves, in natural-log units over 64 bands, are converted to dB and added to
  Cnn14's 64 bands before the level step.
- `src/dad/pannsrun.py`: `PannsScorer`, with the interface of `NetScorer`.
  - P1 `fit_probe`: frozen encoder in eval mode, 2048-d embeddings of sampled training windows, standardised;
    logistic regression with C picked by validation AUC from a grid (entry 137); the scaler is folded
    into a 2-logit head. Its checkpoint is head only, about 20 KB, and stores the encoder checkpoint's MD5.
  - P2 `fit_finetune`: from a probe head, AdamW at a one-cycle peak learning rate of 1e-4 with weight decay
    1e-2, SpecAugment after `bn0` (where the reference applies its own), and the reference's dropout. The
    state with the best validation AUC is kept, checked every 100 000 windows, as `dad.train` keeps the best
    epoch. The checkpoint is the whole network (about 320 MB) and loads without the encoder file.
  - Speed is changed on the device by `gpufront.GpuFrontend.speed`.
- `src/dad/realrun.py`: 3 lines. `NetScorer.load` hands checkpoints with arch `panns*` to `PannsScorer`, so
  `score_glasgow.py`, `score_real_dev.py` and the GUIs take them unchanged.
- `scripts/train_panns.py` runs P1 and P2. It imports train_clean's `load_drones`, `load_nodrone`,
  `CleanSampler`, `eval_windows` and `scores`.
- `scripts/panns_p0.py` runs P0: the per-class AUC of all 527 AudioSet outputs on each comparison whose data
  is present (train_clean validation, real validation rigs, the mower session, Glasgow room and war track).
  Classes picked by these AUCs are picked on validation, so P0 is a diagnostic only.
- `tests/test_panns.py`: 9 tests. One compares our filterbank and STFT kernels with the checkpoint's frozen
  copies; it runs only where the checkpoint exists.
- `scripts/experiments/run_it11_1.sh`: the server stages `fetch`, `test`, `smoke`, `p0`, `p1`, `p2` (seeds 0-2
  on GPUs 0-2, in the background), `gates` and `pack`. It calls `.venv/bin/python`, never `uv run`, so the
  Pascal-capable torch on the server is not replaced.

**How (checks on the Mac, CPU, random weights; no training).**
- `.venv/bin/python -m pytest -q tests`: 121 collected, 119 passed, 2 skipped. The skips are the librosa
  comparison (librosa is not a repo dependency) and the checkpoint test (no checkpoint on the Mac).
  `tests/test_realrun.py` and `tests/test_train_clean.py` pass with the `realrun.py` change.
- Against the reference code, in a scratch environment outside the repo (librosa 1.0.0, torchlibrosa 0.1.0,
  panns_inference 0.1.1, torch 2.14.1):
  - mel filterbank against `librosa.filters.mel(sr=16000, n_fft=512, n_mels=64, fmin=50, fmax=8000)`: max
    absolute difference 1.9e-9 (largest weight 0.0216);
  - STFT kernels against torchlibrosa's: 1.5e-8 (real), 4.7e-10 (imaginary);
  - all 84 state tensors of the reference `Cnn14(sample_rate=16000, window_size=512, hop_size=160,
    mel_bins=64, fmin=50, fmax=8000)` load strictly into the port;
  - with the same random weights and the level step bypassed, embedding and clipwise output differ by 0.0.

**Result.** No detection figure. Not run: the checkpoint test, P0, P1, P2 and every gate.

**Known gaps before a gate verdict.**
- `gates` scores with each run's own train_clean validation threshold. That is the synthetic-validation
  threshold entry 128 found far too strict. The decisive real-background threshold needs
  `real_threshold.py` (Mads's account).
- The real-rig and mower gates need `data/processed/manifest.csv` (Mads's account). Gates (a) and (d) need
  `data/extracted/Glasgow`, which may be on the Mac only.
- The level normalisation is a design choice, not tuned.

**Next.** On the server: `bash scripts/experiments/run_it11_1.sh fetch`, `test`, `smoke`, `p0`, `p1`. The
checkpoint's MD5, the test result and the P0 table go in the next entry.

## 137 - The Cnn14_16k checkpoint on the server, tests, smoke run and P0 (synthetic validation only)
- Date: 2026-11-06
- Iteration: 11.1
- Type: Infrastructure, Evaluation (validation only)
- Commit: 8573817 (server, branch 11.1; Marius's account)

**Checkpoint.** `run_it11_1.sh fetch` downloaded from Zenodo record 3987831 into `artifacts/pretrained/`.
The MD5s are in `artifacts/pretrained/MD5SUMS`:
- `Cnn14_16k_mAP=0.438.pth`: 358 668 570 bytes, MD5 `362fc5ff18f1d6ad2f6d464b45893f2c`;
- `class_labels_indices.csv`: 14 675 bytes, MD5 `68c0b5c199d6cc789852b144cc3ab500`.

**Interpreter.** Marius's server account has no repo `.venv`. Its CRNN runs use `~/dad/.venv/bin/python`
(from the shell history), and it reports torch 2.14.1+cu126 with CUDA on 3 devices.
- `run_it11_1.sh` now takes the interpreter from `$PY`, falling back to the repo `.venv` and then
  `~/dad/.venv`. A preflight check prints the interpreter, torch and CUDA before each stage (commits
  fa4d1e7 and 8573817).
- P2 starts with `nohup setsid ... < /dev/null`, as the CRNN runs do.

**Tests.** `run_it11_1.sh test`: 9 tests, 8 passed, 1 skipped (the librosa comparison; librosa is not in
that environment). **The checkpoint test passed.** The published state loads into the port with
`strict=True`, and the checkpoint's own `logmel_extractor.melW` and STFT kernels match our filterbank
and kernels within the test tolerances (rtol 1e-4 / atol 1e-6 and atol 1e-5). Entry 136's frontend is
therefore the one the weights were trained on.

**Smoke.** `run_it11_1.sh smoke` is P1 with `--limit 20 --windows 512` on the e133 flags. It ran end to
end on one GPU at 340 windows/s while embedding. Its numbers come from 20 scenes and 512 windows and
are not results.
- The probe picked C = 1e-3, the smallest value in the grid.
- So the grid is now {1e-4, 1e-3, 1e-2, 1e-1, 1}, and a pick at either edge is flagged in the run JSON
  (`C_at_grid_edge`).

**P0** (`run_it11_1.sh p0`, `artifacts/it11_1/p0.json`). Only `val_synthetic` ran: train_clean's validation
Pd drone scenes (`it11s_valdrone`, 5956 windows) against the Nightjar no-drone clips (8999 windows), at a
0.5 s hop. The real rigs, the mower session and Glasgow were skipped, because there is no
`data/processed/manifest.csv` and no `data/extracted/Glasgow` in this account.
- AudioSet classes with the highest AUC (drone over no-drone):

  | Class | AUC |
  |---|---|
  | Tender music | 0.810 |
  | Theme music | 0.783 |
  | Accelerating, revving, vroom | 0.776 |
  | Heavy engine (low frequency) | 0.764 |
  | Background music | 0.763 |

- **Reading.** To a general-audio tagger, the Pd drones resemble tonal music more than aircraft. This is a
  diagnostic, chosen on validation data, not a result. It is a warning that synthetic-drone validation
  may mislead model choice here, as entry 128 found for thresholds. It does not bear on P1, which fits
  its own classifier on the 2048-d embedding.
- The a-priori classes (Aircraft, Propeller, Helicopter, Lawn mower and so on) are in the JSON and not
  copied here.
- P0 on real drones needs Mads's account (manifest) or the Mac (Glasgow).

**Next.** P1: `run_it11_1.sh p1` (100 000 windows). Then the real-rig gates, scored where the manifest and
`real_threshold.py` are.

## 138 - P1 (frozen Cnn14_16k + probe) on the e133 data: weak on synthetic validation, which cannot judge it
- Date: 2026-11-06
- Iteration: 11.1
- Type: Model, Evaluation (validation only)
- Commit: 78dded7 (server, branch 11.1)

**Run.** `bash scripts/experiments/run_it11_1.sh p1` in Marius's account, GPU 0.
- P1 used the e133 data flags (176 Glasgow d1-d16 drone scenes, 1874 no-drone clips) and seed 0, with
  100 000 training windows embedded at about 820 windows/s. Training took 212.9 s in total.
- Validation was train_clean's: 5956 drone windows (Pd scenes, seed ≥ 1800) against 15 839 no-drone windows
  (the Nightjar clips plus the no-drone scenes), 1 s windows at a 0.5 s hop.
- Output: `artifacts/it11_1/p1_s0.json` and `p1_s0_panns.pt` (head only).

**Result** (validation only, synthetic drones):

| C | 1e-4 | 1e-3 | 1e-2 | 1e-1 | 1 |
|---|---|---|---|---|---|
| validation AUC | **0.591** | 0.546 | 0.522 | 0.518 | 0.521 |

- Chosen C = 1e-4, at the edge of the grid (flagged).
- The probe at that C: AUC 0.591, recall 0.202 at 5% FA, FA 0.943 at 95% recall, threshold 0.2173 at
  5% FA.
- `comb` on the same windows: AUC 0.589, recall 0.156 at 5% FA.
- For reference, the e133 from-scratch Glasgow arms on their own synthetic validation: CRNN AUC 0.560 (s0)
  and 0.420 (s1) (entry 134; their validation had the same drone scenes).
- P0's a-priori AudioSet classes on the synthetic validation, for the record (AUC):
  - Engine 0.689, Light engine 0.691, Helicopter 0.643, Propeller 0.612, Aircraft 0.603, Aircraft engine 0.585;
  - Lawn mower 0.556, Mechanical fan 0.506;
  - Insect 0.151, Bee/wasp 0.192, Fly 0.229, Silence 0.299.

**Reading.**
- On this validation, P1 is level with the from-scratch Glasgow arms and with `comb`. Nothing more can be
  said from it.
- The C sweep shows why. The more closely the probe fits its training data, the worse it scores on
  validation. Training drones are real Glasgow recordings, while validation drones are synthetic Pd
  scenes, which the encoder hears as music (P0, entry 137). So this validation measures transfer from
  Glasgow to Pd drones, and it rewards the most regularised, nearly flat probe.
- That is a weakness of the e133 protocol, which P1 inherited by design ("only the network changes"). It is
  not evidence about real drones.
- Lowering C further would chase that mismatch, so the grid is left as it is.
- **Not a verdict.** Entry 135's rule is decided by the real-rig gates (b, c), the mower gate (e) and the
  Glasgow gates (a, d), at the real-background threshold. None has run.
- P2 waits, as entry 135 fixed: it starts only if P1 beats the e133 arms on gate (b) or (c).

**Gate (e) scorer.** `scripts/score_mower.py` is new. It scores the mower-only windows of
`nightjar-lawnmower1-20260805` (rangerun.CONFUSER) and reports, per model:
- the false-alarm rate at the frozen threshold, with a Wilson 95% interval;
- the within-session AUC of drone against mower-only windows;
- drone recall.
It needs the manifest.

**Next.** Score P1 on gates (b), (c) and (e) in Mads's account, where `data/processed/manifest.csv` and
`real_threshold.py` are, next to it9 and the e133 Glasgow arms. Then gates (a) and (d) on the Mac, where
Glasgow d17-d20 and the room recording are.

## 139 - Real data copied to Marius's account; P1 on gates (b) and (c): Nightjar AUC jumps, recall at 5% FA barely moves, d3set unchanged
- Date: 2026-11-06
- Iteration: 11.1
- Type: Data, Evaluation (validation only)
- Commit: 78dded7 on the server for the scoring; `scripts/real_threshold.py` is added in this entry's commit

**Data.** On Mads's choice, the real data the gates read was copied once from his account (`jupyter-mads`)
into Marius's (`jupyter-mariusq`), so training and gates run in one account.
- What was copied: from `~/drone-acoustic-detection/data/processed` (which `~/dad/data/processed` links to),
  `manifest.csv`, `range_track.csv`, `ddl_clips.csv`, `summary.json`, `nightjar.i16` (305 484 000 bytes) and
  `d3set.i16` (264 420 040 bytes).
- It moved as one tar through `/tmp`: `real_gates.tar`, MD5 `5f6fc8a2db9f473cc191022a98672d4e`, `md5sum -c` OK.
  It was unpacked with `tar --skip-old-files`, so nothing in Marius's `data/processed` was overwritten.
- Not copied: FSD50K (9.6 GB), FPV, Halmstad, DroneNoise, DDL and Lund. The gate scripts read only the
  Nightjar and d3set validation sessions and the mower session, because recordings are memory-mapped when
  first read. Any code that reads another source in Marius's account will fail with "file not found".
- The `/tmp` copies are to be deleted once this entry is logged.

**Copy check.** it9 CRNN on the copied data, scored with the command below, reproduces entry 134 to the
third decimal: d3set AUC 0.930 and recall 0.823 at 5% FA; Nightjar 0.759 and 0.678.

**`real_threshold.py`.** Mads's script (`~/e123_score/scripts/real_threshold.py`, MD5
`1fdbd6c6a47974e43827b7a8396826cb`), which set the thresholds of entries 129 and 134. It is now
`scripts/real_threshold.py`, with the model list and output path as arguments and the rule unchanged:
`fa_threshold` at 5% on the background windows of `make_pool_arm(..., "none").val`, at a 0.5 s hop.
- **Correction to entries 131 and 135.** They describe this threshold as "pooled per rig, weighted, not by
  window count". The script pools by window, so the rig with more background windows weighs more. Entry
  134's thresholds, and every one from here on, use the script's rule.
- `run_it11_1.sh gates` now runs, in order, `score_real_dev.py`, `real_threshold.py`, `score_mower.py` at
  that threshold, and `score_glasgow.py` where Glasgow exists. it9 is scored alongside as the reference.

**How.** In Marius's account:
`CUDA_VISIBLE_DEVICES=0 ~/dad/.venv/bin/python -u scripts/score_real_dev.py --device cuda --out
artifacts/it11_1/gates_real_dev.json "it9 CRNN=artifacts/models/final_crnn_none_simdist_crnn.pt"
"p1=artifacts/it11_1/p1_s0_panns.pt"`.
The figures are recall at 5% (10%) FA on each rig's own validation background, all windows, 0.5 s hop.

**Result** (validation, real rigs; e133 figures from entry 134):

| Model | Nightjar AUC | Nightjar recall @5% FA (@10%) | d3set AUC | d3set recall @5% FA (@10%) |
|---|---|---|---|---|
| **P1** (frozen Cnn14_16k + probe, C 1e-4) | **0.908** | 0.683 (0.742) | 0.823 | 0.560 (0.617) |
| e133 Glasgow s0 (CRNN from scratch, same data) | 0.713 | 0.650 | 0.812 | 0.582 |
| e133 Glasgow s1 | 0.755 | 0.672 | 0.807 | 0.586 |
| it9 CRNN (real drones of several rigs; not under entry 107's rule) | 0.759 | 0.678 (0.695) | 0.930 | 0.823 (0.841) |
| comb | 0.516 | 0.115 (0.120) | 0.643 | 0.312 (0.337) |
| Gate (entry 135) | | (b) ≥ 0.77 | | (c) ≥ 0.80 |

**Reading.**
- **Gates (b) and (c) both fail.**
- **Nightjar.** Pretraining changes the ranking, but the operating point barely moves.
  - AUC is 0.908, against 0.713 and 0.755 for the from-scratch arms on the same data, and above it9 (0.759).
    That is far outside the e133 seed spread.
  - Recall at 5% FA is 0.683, which is +0.011 over the better e133 seed and inside the e133 seed spread
    (0.650 to 0.672, so 0.022).
  - So the encoder ranks Nightjar drones above Nightjar background much better overall, but the
    top-scoring 5% of background windows still outrank many drone windows.
- **d3set.** No gain: AUC 0.823 against 0.81, and recall 0.560 against 0.58.
- Against the synthetic validation of entry 138 (AUC 0.591), the real Nightjar AUC (0.908) confirms that
  the synthetic validation could not judge P1.
- **P2 trigger (entry 135).** "P2 runs only if P1 beats the e133 from-scratch arm on gate (b) or (c)." On (b)
  it does by the letter (0.683 against 0.650 and 0.672), but by less than e133's seed spread. The condition
  is met narrowly. The stronger evidence that pretraining helps is the Nightjar AUC, not the recall at 5% FA.
- P1 is one probe (seed 0). Its spread over seeds is not measured.

**Next.**
1. The real threshold and gate (e): `run_it11_1.sh gates`.
2. P2 (LP-FT), seeds 0-2 on GPUs 0-2, under entry 135's rule.
3. Gates (a) and (d) on the Mac, where Glasgow is.

## 140 - P2 keeps its final state, not the best synthetic-validation state (before any P2 run)
- Date: 2026-11-06
- Iteration: 11.1
- Type: Tuning (protocol detail, fixed before the run)
- Commit: this entry's commit

**What changed.** `PannsScorer.fit_finetune` now keeps the final state by default (`--select last`). Entry
136 had it keep the state with the best validation AUC, checked every 100 000 windows. Validation AUC is
still logged at each check, and the JSON records the best and final values. `--select best` remains as an
option.

**Why.**
- Entry 136's choice followed `dad.train`, which keeps the best validation epoch.
- On the e133 data, however, the validation drones are synthetic Pd scenes. Entries 138 and 139 show
  that this validation misjudges the encoder: the P1 probe scored AUC 0.591 there against 0.908 on real
  Nightjar audio, and the less it fitted, the higher it scored.
- Selection on it would favour the flattest state, possibly the untrained starting probe.
- The real validation sets (Nightjar, d3set, Glasgow d17-d20) are the gate data and cannot be used for
  selection.
- Keeping the final state is what `NetScorer` does for the e133 CRNNs. With it, "only the network
  changes" holds for the stopping rule too.

**Result.** Nothing run. `tests/test_panns.py` passes on the Mac (7 passed, 2 skipped).

## 141 - P1 at the real-background threshold: pooled recall 0.49; mower gate (e) passes; P2 started
- Date: 2026-11-06
- Iteration: 11.1
- Type: Evaluation (validation only)
- Commit: c1337a0 (server)

**How.** `bash scripts/experiments/run_it11_1.sh gates` in Marius's account. It reran (b) and (c), with the
same figures as entry 139, then ran `real_threshold.py` and `score_mower.py`. Output files are
`artifacts/it11_1/real_threshold.json` and `gates_mower.json`.

**Real-background threshold** (entry 129's rule; 2648 drone and 1565 background windows of the pooled
Nightjar and d3set validation sessions):

| Model | Threshold | FA | Pooled recall at the threshold |
|---|---|---|---|
| it9 CRNN | 0.4828 | 5.05% | 0.785 |
| P1 | 0.2458 | 5.05% | **0.491** |
| e133 Glasgow s0 / s1 (entry 134) | 4.7e-6 / 1.3e-5 | | 0.710 / 0.703 |
| it11r CRNN 600k / e133 it11s clean (entry 134) | 0.337 / 0.373 | | 0.592 / 0.583 |

it9's threshold and recall reproduce entry 134 exactly.

**Gate (e), the mower session** (`nightjar-lawnmower1-20260805`): 1045 drone windows and 125 mower-only
windows at a 0.5 s hop. Each model is at its own threshold from the table above.

| Model | Mower-only FA (Wilson 95%) | AUC, drone against mower-only | Drone recall in the session |
|---|---|---|---|
| **P1** | **9.6%** (5.6-16.0%), passes (≤ 25%) | **0.740** | 0.357 |
| it9 CRNN | 26.4% (19.5-34.7%), fails | 0.695 | 0.584 |
| comb | | 0.374 | |

**Reading.**
- **Pooled recall is low.** P1's recall at the pooled real threshold (0.491) is the lowest of the models
  compared, although its Nightjar ranking is the best (AUC 0.908, entry 139). One threshold has to serve
  Nightjar and d3set, and P1's scores are not on the same scale on the two rigs. The probe ranks well
  within a rig, but it is poorly calibrated across rigs.
- **The mower gate passes, partly because P1 is less sensitive at its threshold.** Its drone recall in the
  same session is 0.357 against it9's 0.584. The within-session AUC is the fairer comparison. There P1
  separates drone from mower a little better than it9 (0.740 against 0.695). This is entry 053's
  within-rig check. Entry 053's decisive objection, that a frozen AudioSet encoder merges mowers with
  drones, does not hold for this session.
- The figure rests on 125 mower-only windows from one session (entry 025). The interval is stated with it.
- **P1 against entry 135's rule:** (b) fails (0.683), (c) fails (0.560), (e) passes; (a) and (d) are not
  scored yet.
  - Not a success.
  - Not the "informative failure" either: (c) shows no gain over e133.
  - Not a rejection: (b) is narrowly better than e133.
  - Logged as mixed.
- **P2 started** under entry 135 (condition met narrowly, entry 139) and entry 140 (final state kept). It
  runs at 10:39:03 on GPUs 0, 1 and 2, seeds 0-2, 600 000 windows each, all three from `p1_s0_panns.pt`.
  Logs: `artifacts/it11_1/logs/p2_s{0,1,2}_20261106_103903.log`.

**Glasgow for gates (a) and (d).** Glasgow d17-d20, the room recording and the war track are on the Mac
only. They are packed for a single upload to the server, so the P2 checkpoints (about 320 MB each)
need not travel to the Mac.
- `artifacts/transfer/glasgow_gates.tar` (Mac, git-ignored): 866 500 608 bytes, MD5
  `281451f0416e6f90841fca0b57a5b645`.
- It holds 42 files, the originals unmodified (44.1 kHz): the 40 `17to24/` hover files of d17-d20, the
  room recording `noise/_uav_2022noise_n_noise_1m_.WAV`, and `data/processed/war/war_track.wav`.
- The test drones d21-d24 share that folder and are left out. This was checked: 0 matching files in the tar.

## 142 - P2 (LP-FT) overfits the Glasgow rig and is rejected; P1 passes gates (a) and (e) and fails (b), (c) and (d)
- Date: 2026-11-06
- Iteration: 11.1 (verdict)
- Type: Model, Evaluation (validation only)
- Commit: c1337a0 (server)

**Runs.** `run_it11_1.sh p2`: seeds 0-2 on GPUs 0-2, each from `p1_s0_panns.pt`.
- 600 000 windows each, AdamW at a peak learning rate of 1e-4, the final state kept (entry 140).
- 1755 / 1795 / 1804 s, at about 345 windows/s per seed.
- Training loss (mean per 100 000 windows): 0.80 → 0.06 → 0.03 → 0.02 → 0.016, with all three seeds alike.
- train_clean's synthetic validation (AUC at the end): 0.397 / 0.386 / 0.408, below chance and below the
  starting probe's 0.591. This is the e133 pattern of entry 134.

**Gates.** `run_it11_1.sh gates` scored it9 CRNN, P1 and P2 s0-s2.
- Thresholds are `real_threshold.py`'s (entry 139).
- Glasgow: d17-d20 hover (9796 windows) against the second half of the room recording (3627 windows).
  The war mixes are at 0 / -5 / -10 dB in-band, scored against the war track.
- Files: `artifacts/it11_1/gates_real_dev.json`, `real_threshold.json`, `gates_mower.json` and
  `gates_glasgow.json`.
- Glasgow came from the Mac tar of entry 141. The war-track MD5 matches the Mac's
  (`966937647ba588030f411ca1eaf079ee`).
- macOS `tar` had added `._*` AppleDouble files. They were deleted before scoring. From now on, Mac tars
  are made with `COPYFILE_DISABLE=1`.
- it9 reproduces entries 133-134 throughout: room AUC 0.092, room FA 99.2%, war 0.985 / 0.971 / 0.937, and
  threshold 0.4828.

| Model | (b) Nightjar AUC / recall @5% FA | (c) d3set AUC / recall @5% FA | Real threshold / pooled recall | (e) mower FA (Wilson 95%) / drone-vs-mower AUC | (a) room AUC / FA at threshold | (d) war recall @1% FA, 0 / -5 / -10 dB |
|---|---|---|---|---|---|---|
| P1 (frozen + probe) | **0.908** / 0.683 | 0.823 / 0.560 | 0.2458 / 0.491 | 9.6% (5.6-16.0) / **0.740** | 1.000 / 3.3% | 0.977 / 0.898 / 0.789 |
| P2 s0 | 0.759 / 0.513 | 0.764 / 0.498 | 0.0243 / 0.299 | 11.2% (6.8-17.9) / 0.398 | 1.000 / 0.8% | 1.000 / 1.000 / 0.997 |
| P2 s1 | 0.775 / 0.555 | 0.783 / 0.504 | 0.0101 / 0.333 | 14.4% (9.3-21.6) / 0.386 | 1.000 / 0.6% | 1.000 / 1.000 / 0.995 |
| P2 s2 | 0.766 / 0.543 | 0.722 / 0.421 | 0.0084 / 0.304 | 25.6% (18.8-33.9) / 0.332 | 1.000 / 0.8% | 1.000 / 1.000 / 0.994 |
| e133 Glasgow s0 / s1 (entry 134) | 0.713 / 0.650, 0.755 / 0.672 | 0.812 / 0.582, 0.807 / 0.586 | 4.7e-6, 1.3e-5 / 0.710, 0.703 | not scored | 1.000 / 99.9%, 99.2% | 1.00 / 1.00 / 0.999 |
| it9 CRNN | 0.759 / 0.678 | 0.930 / 0.823 | 0.4828 / 0.785 | 26.4% (19.5-34.7) / 0.695 | 0.092 / 99.2% | 0.985 / 0.971 / 0.937 |
| comb | 0.516 / 0.115 | 0.643 / 0.312 | | / 0.374 | 0.737 / | 0.166 / 0.133 / 0.097 |
| Gate (entry 135) | recall ≥ 0.77 | recall ≥ 0.80 | | FA ≤ 25% | AUC ≥ 0.97 and FA ≤ 5% | within 5 points of it9 |

Notes on the table:
- The mower session has 125 mower-only windows and 1045 drone windows.
- Drone recall in that session at each model's threshold: P1 0.357, P2 0.216 / 0.225 / 0.241, it9 0.584.

**Verdict (entry 135's rule, fixed before any run).**
- **P2 is rejected.** It is worse than the e133 arms on both (b) and (c), on all three seeds. Mean over seeds:
  - Nightjar recall 0.537 against e133's 0.650-0.672;
  - d3set recall 0.474 against e133's 0.582-0.586.
- **P1 is mixed.**
  - (a) passes: room AUC 1.000, FA 3.3%.
  - (e) passes: 9.6%, better than it9's 26.4%.
  - (b) fails: 0.683.
  - (c) fails: 0.560.
  - (d) fails: -0.8 / -7.3 / -14.8 points against it9.
  - It is neither the rule's success nor its informative failure.
- The Glasgow passes are weak evidence either way. d17-d20 come from the same rig and room as the training
  drones d1-d16, and the first half of the room recording is in training (entries 133-134).

**Reading.**
1. **Full fine-tuning on one rig's real drones overwrites the pretraining.**
   - P2 becomes perfect on Glasgow and worse than the frozen P1 on every rig it never heard: Nightjar AUC
     0.908 → 0.76-0.78, d3set 0.823 → 0.72-0.78, pooled recall 0.49 → about 0.31.
   - It now ranks mower-only windows above drone windows within the mower session (AUC 0.33-0.40).
   - This is the single-rig shortcut of entry 134 in a pretrained network. It is also the effect Kumar et
     al. (ICLR 2022) report, that full fine-tuning can lose out-of-distribution accuracy against a linear
     probe; entry 053's Scholar named it as a pitfall.
   - It is consistent across the three seeds.
2. **The frozen encoder keeps general structure.**
   - It gives the best Nightjar ranking of every model in this report's real-rig table (0.908; it11r CRNN
     600k had 0.929 in entry 134).
   - It gives the best within-session drone-vs-mower separation (0.740). Entry 053's decisive objection,
     that a frozen AudioSet encoder merges mowers with drones, does not hold for this session.
   - What fails is the operating point. A single threshold does not transfer across rigs: pooled recall is
     0.491 while within-rig AUC is 0.908.
3. **The binding limit is still the training data, not the network.**
   - Under entry 107's rule, the only real training drones are one indoor rig's.
   - Pretraining improved ranking on one unseen rig (Nightjar) but not the other (d3set). It did not
     reach gates (b) or (c), and tuning it fully made things worse.
   - This agrees with entries 131 and 134: the next lever is real drones from several rigs in training,
     which is Mads's decision on the "Synthetic training data" row.
   - If that is allowed, the frozen Cnn14_16k probe (P1) is the pretrained arm to pair with it, not LP-FT.
     That is the 2×2 of the plan (data rule × pretraining).

**Not run.** Partial fine-tuning (only the last blocks, or a lower learning rate) and recalibration across
rigs. Both would be new arms. They are not tried here, because the evidence points at the data, and
entry 135 fixed one encoder with one P2 recipe.

**Files.** The three P2 checkpoints, about 320 MB each, stay on the server in `artifacts/it11_1/`. `pack`
leaves them out.

## 143 - Requirements: real drones from several rigs may train (Mads)
- Date: 2026-11-06
- Iteration: 11.2 (opens it)
- Type: Requirements
- Commit: this entry's commit

**What changed.** After entry 142, Mads was asked how to continue. The options were to allow real
drones from several rigs, to keep the rule and try fixes within it, or to stop and report. He chose:
**"Allow real multi-rig drones in training"**.
- This is entry 131's "modified A", with the wording entry 131 proposed.
- REQUIREMENTS.md has a new row, "Real drones in training". The "Synthetic training data" row (entry 107)
  is marked superseded.
- Training may use real drones from the training sessions of DDL, Nightjar and d3set, and Glasgow
  d1-d16, next to synthetic drones.
- FPV, DroneNoise, Halmstad and FSD50K-eval stay test. Glasgow d17-d20 validate and d21-d24 test.
- The rig a gate is scored on is held out of training for that gate (leave one rig out, as entry 131 required).

# Iteration 11.2 - Real drones from several rigs, CRNN and the frozen Cnn14_16k probe

> **Changed from iteration 11.1:** the training data. Real drones from several rigs, under entry 143's rule:
> it9's pooled recipe (`run_range.py --pool`, simdist, speed 0.8-1.25) with FPV taken out of the pool,
> because FPV is test now. Steady noise as in iteration 11 (`simdist_steady`). Two networks on the same
> sampler: the CRNN from scratch, and the frozen Cnn14_16k with a probe (P1's method, entry 136). Not
> LP-FT, which entry 142 rejected.
> **Why:** entries 134 and 142. With one indoor rig of real drones, neither training from scratch nor
> pretraining reaches gates (b) and (c). Full fine-tuning overfits the rig. it9, trained on real drones
> from several rigs, reaches 0.785 pooled real recall but alarms on 99% of the unseen Glasgow room.
> **Kept the same:**
> - the gates of entry 135: (a) Glasgow room, (b) Nightjar, (c) d3set, (d) war mix, (e) mower;
> - the real-background threshold rule of entry 129, computed from the training rigs' validation
>   sessions only;
> - the 1 s window and 0.5 s evaluation hop;
> - test and sealed data, untouched.
> **Design:** leave one rig out.
> - Fold N trains with every Nightjar session held out and is judged on (b).
> - Fold D trains with every d3set session held out and is judged on (c).
> - Both folds are scored on (a), (d) and (e).
> - The mower session belongs to Nightjar, so (e) is unseen-rig only in fold N.
> - 2 seeds per CRNN fold, 1 probe per fold.
> **Hypothesis:** real drones from several rigs lift (b) and (c) towards it9. The frozen encoder keeps
> the unseen Glasgow room quiet (gate a), where it9 fails, so the probe arm passes (a), (c) and one of
> (b) or (d).
> **Judged by:** entry 135's success rule, per fold. If (a) fails, the next step is entry 131's: add
> Glasgow room and soundscape negatives, and balance negatives per rig.

**Needed before any run.**
1. `rangerun`: a pool arm without FPV (entry 143), with Nightjar or d3set held out.
2. `run_range.py`: that arm, and the frozen probe as a model next to the CRNN.
3. `real_threshold.py`: the threshold from the training rigs' validation sessions only, per fold.
4. Data in Marius's account: `ddl.f32` (DDL near drones, 364 MB) and `fsd50k.i16` (FSD50K training
   backgrounds, 9.6 GB). Both come from Mads's account as in entry 139. FSD50K-eval stays excluded by
   `make_splits`.

## 144 - Iteration 11.2 code: the pool arm without FPV, the probe in run_range.py, and the runner (no run)
- Date: 2026-11-06
- Iteration: 11.2
- Type: Infrastructure
- Commit: this entry's commit

**What changed.**
- `rangerun.make_pool_arm_143(manifest, held)` builds the arm for entry 143's rule.
  - The pool is Nightjar and d3set, plus Glasgow if the manifest has it. FPV is out of the pool because it
    is a test source.
  - `held` is "nightjar" or "d3set", and it is wholly out of training and validation. Validation, and with
    it the deployment threshold, comes from the other rig only.
  - The mower session is never trained on. Nothing is scored as unseen; Lund is not in Marius's account.
  - DDL near drones and FSD50K training backgrounds enter through `RangeSampler`, as for it9. DroneNoise,
    Halmstad and FSD50K-eval are not read by this path.
  - `tests/test_rule143.py`: 3 tests pass.
- `scripts/run_range.py` has two additions.
  - `--rule143` selects that arm.
  - `panns_probe` is a model: the frozen Cnn14_16k plus probe of entry 136, fitted on `RangeSampler.raw_batch`.
    C is picked on `arm.val`, the training rig's *real* validation sessions, not on synthetic drones as in
    iteration 11.1. `--probe-windows` defaults to 100 000.
  - The existing `comb`, `lgbm` and `crnn` paths are unchanged.
- `scripts/it11_2_models.py` lists the trained CRNN and probe checkpoints as `label=checkpoint@threshold`,
  using each run's `deploy_threshold`: 5% FA on the training rig's validation sessions.
- `scripts/experiments/run_it11_2.sh` has four stages: `check`, `train`, `status` and `gates`.
  - `train` starts three background jobs, all with recipes `simdist` (it9) and `simdist_steady`, speed
    0.8-1.25, 200 000 windows per CRNN (as it9), and `--deploy-fa 0.05`:
    - GPU 0: seed 0, Nightjar held out, with comb, lgbm, crnn and panns_probe;
    - GPU 2: seed 0, d3set held out, with the same four models;
    - GPU 1: seed 1, both folds, crnn only.
  - `gates` scores it9 and every trained model with `score_real_dev.py` (b, c), `score_mower.py` (e) and
    `score_glasgow.py` (a, d). A fold counts on (b) or (c) only for its held-out rig.

**Data still to arrive in Marius's account.** `ddl.f32` and `fsd50k.i16` (entry 143).

**Result.** Nothing run. On the Mac, `tests/test_rule143.py`, `tests/test_rangerun.py` and
`tests/test_panns.py` pass (2 skips, as before).

## 145 - Correction: RangeSampler always trained on DroneNoise drones; off under entry 143, so 11.2 restarts
- Date: 2026-11-06
- Iteration: 11.2
- Type: Bug fix, Correction
- Commit: this entry's commit

**What happened.** The first `run_it11_2.sh train` (11:26:31) stopped in all three jobs before any training
step, with `FileNotFoundError: data/processed/dronenoise.f32`.
- `RangeSampler.__init__` always adds a DroneNoise cell (`rr.dronenoise_pool`) as training positives, next to
  the DDL ≤ 50 m cell.
- DroneNoise is a test source under entry 143 and was therefore not copied to Marius's account (entry 143).

**Fix.**
- `RangeSampler(..., dronenoise=True)` keeps the old behaviour by default. `run_range.py` passes
  `dronenoise=False` under `--rule143`.
- `run_it11_2.sh check` now builds both folds' samplers for both recipes, exactly as `train` does. This reads
  the audio, so a missing source fails there and not on the GPUs. It also asserts that no DroneNoise cell is
  present, and prints each sampler's composition.

**Correction to earlier entries.** Every `run_range.py` model trained DroneNoise near-pass-by drones as
positives: iterations 3-6 and 9, it6, and **it9 CRNN**.
- Entry 133 noted this for it6 only.
- Their DroneNoise figures, where any were taken, are not test results.
- it9 also trained on the FPV campaigns. Under entry 143's rule it is a reference, not a compliant candidate.

**Result.** Nothing trained. `tests/test_rule143.py` and `tests/test_rangerun.py` pass. 11.2 restarts after
`check`.

## 146 - Iteration 11.2 results: no gate passes; a real outdoor rig lifts the CRNN on unseen d3set, the frozen probe falls behind
- Date: 2026-11-06
- Iteration: 11.2 (verdict)
- Type: Model, Evaluation (validation only)
- Commit: df9a21e (server)

**Runs.** `run_it11_2.sh train` started at 11:31:44, after entry 145's fix.
- `run_range.py --rule143` ran recipes `simdist` and `simdist_steady`, speed 0.8-1.25, 200 000 windows per
  CRNN, 100 000 per probe, `--deploy-fa 0.05`, at a 0.5 s evaluation hop.
- Fold N holds out Nightjar: it trains on d3set (85 segments) plus DDL ≤ 50 m, and its threshold comes from
  d3set validation (13 segments).
- Fold D holds out d3set: it trains on Nightjar (54 segments) plus DDL ≤ 50 m, and its threshold comes from
  Nightjar validation (8 segments).
- Seed 0 ran comb, lgbm, crnn and panns_probe; seed 1 ran crnn only.
- `run_it11_2.sh gates` produced `artifacts/it11_2/gates_real_dev.json`, `gates_mower.json` and
  `gates_glasgow.json`, with thresholds listed in `artifacts/it11_2/models.txt`.
- The LightGBM and comb figures on the held-out rigs are in the run JSONs (`artifacts/runs/it11_2_*.json`).
  They are not copied here; comb on the held-out rig at the frozen threshold has recall 0.201 (Nightjar) and
  0.153 (d3set).

**Gates (b) and (c).** Recall at 5% FA on the held-out rig's own validation background (`score_real_dev.py`).
Only the held-out rig counts for a fold; the other column is that model's in-rig figure, given for scale.

| Fold, recipe, model, seed | Nightjar AUC / recall | d3set AUC / recall |
|---|---|---|
| N, simdist, CRNN s0 | **0.941 / 0.697** | 0.922 / 0.825 (in-rig) |
| N, simdist, CRNN s1 | **0.911 / 0.675** | 0.928 / 0.828 (in-rig) |
| N, steady, CRNN s0 | **0.834 / 0.639** | 0.928 / 0.837 (in-rig) |
| N, steady, CRNN s1 | **0.872 / 0.678** | 0.925 / 0.830 (in-rig) |
| N, simdist, probe | **0.843 / 0.583** | 0.910 / 0.808 (in-rig) |
| N, steady, probe | **0.818 / 0.546** | 0.899 / 0.772 (in-rig) |
| D, simdist, CRNN s0 | 0.837 / 0.728 (in-rig) | **0.876 / 0.641** |
| D, simdist, CRNN s1 | 0.831 / 0.737 (in-rig) | **0.883 / 0.662** |
| D, steady, CRNN s0 | 0.846 / 0.723 (in-rig) | **0.879 / 0.696** |
| D, steady, CRNN s1 | 0.831 / 0.742 (in-rig) | **0.889 / 0.685** |
| D, simdist, probe | 0.934 / 0.796 (in-rig) | **0.802 / 0.430** |
| D, steady, probe | 0.925 / 0.773 (in-rig) | **0.789 / 0.387** |
| it9 CRNN (trained in-rig on both, and on FPV and DroneNoise) | 0.759 / 0.678 | 0.930 / 0.823 |
| e133 Glasgow s0 / s1 (entry 134) | 0.713 / 0.650, 0.755 / 0.672 | 0.812 / 0.582, 0.807 / 0.586 |
| Gate | (b) ≥ 0.77 | (c) ≥ 0.80 |

**Gates (a), (d) and (e)**, each model at its fold's frozen threshold.
- Glasgow is in no 11.2 training set: the manifest has no Glasgow rows (entry 132). So (a) and (d) are
  unseen-rig for every model here, unlike in entries 133-142.
- The mower session is a Nightjar session. It is unseen-rig only for fold N; for fold D it is in-rig but
  never trained on.

| Model | (a) room AUC / FA | (d) war recall @1% FA, 0 / -5 / -10 dB | (e) mower FA / drone-vs-mower AUC / drone recall |
|---|---|---|---|
| N simdist CRNN s0 | 0.969 / 94.1% | 0.999 / 0.997 / 0.982 | 16.8% / 0.491 / 0.385 |
| N simdist CRNN s1 | 0.995 / 90.4% | 1.000 / 0.997 / 0.967 | 11.2% / 0.520 / 0.393 |
| N steady CRNN s0 | 0.994 / 69.9% | 0.999 / 0.991 / 0.963 | 16.8% / 0.477 / 0.365 |
| N steady CRNN s1 | 0.994 / 58.3% | 1.000 / 0.990 / 0.947 | 22.4% / 0.462 / 0.364 |
| N simdist probe | 0.983 / 29.0% | 0.859 / 0.754 / 0.636 | 33.6% / 0.637 / 0.538 |
| N steady probe | 0.964 / 26.6% | 0.908 / 0.811 / 0.674 | 48.0% / 0.557 / 0.576 |
| D simdist CRNN s0 | 0.709 / 17.7% | 0.989 / 0.986 / 0.979 | 61.6% / 0.527 / 0.618 |
| D simdist CRNN s1 | 0.619 / 39.2% | 0.976 / 0.986 / 0.981 | 60.8% / 0.417 / 0.462 |
| D steady CRNN s0 | 0.545 / 10.9% | 0.987 / 0.991 / 0.983 | 53.6% / 0.438 / 0.475 |
| D steady CRNN s1 | 0.583 / 5.7% | 0.908 / 0.948 / 0.937 | 53.6% / 0.489 / 0.515 |
| D simdist probe | 0.888 / 31.6% | 0.566 / 0.482 / 0.355 | 36.8% / 0.594 / 0.505 |
| D steady probe | 0.850 / 28.7% | 0.809 / 0.710 / 0.551 | 39.2% / 0.602 / 0.528 |
| it9 CRNN | 0.092 / 99.2% | 0.985 / 0.971 / 0.937 | 26.4% / 0.695 / 0.584 |
| Gate | AUC ≥ 0.97 and FA ≤ 5% | within 5 points of it9 | ≤ 25% |

**Verdict (entry 135's rule, per fold).**
- **No model succeeds.** (a) fails for every model and (c) for every fold-D model.
- **Fold N (unseen Nightjar).** All four CRNNs pass (d) and (e) and fail (a) and (b). The probes fail (a),
  (b), (d) and (e).
- **Fold D (unseen d3set).** The CRNNs pass (d), except steady s1 at 0 dB. They fail (a), (c) and (e); (e) is
  53-62%, in-rig.
- **Against the e133 arms.** On (c), every fold-D CRNN is better by more than the e133 seed spread: 0.641-0.696
  against 0.582-0.586. On (b), the fold-N CRNNs average 0.686 (simdist) and 0.659 (steady) against 0.650-0.672,
  which is not more than the spread. That is entry 135's "informative failure" on (c) only.

**Reading.**
1. **Real drones from an outdoor rig help the CRNN on the other outdoor rig.**
   - Unseen d3set recall rises by 6-11 points over the Glasgow-only arms.
   - Unseen Nightjar barely moves.
   - `simdist_steady` helps on d3set (0.69 against 0.65) and costs a little on Nightjar (0.66 against 0.69).
     It lowers the room FA in both folds.
2. **The frozen Cnn14 probe falls behind once real outdoor drones are available.**
   - Unseen recall is 0.39-0.58 against the CRNN's 0.64-0.70.
   - In 11.1 (Glasgow drones only) the frozen encoder ranked Nightjar better (AUC 0.908). Here the
     d3set-trained probe reaches 0.843 on Nightjar and the CRNN 0.91-0.94.
   - The probe's thresholds transfer better to the Glasgow room (FA 27-32% against 58-94% for the fold-N
     CRNNs), and it separates drone from mower better (AUC 0.56-0.64 against 0.42-0.52). It fails the war gate.
   - With real drones in training, the pretrained embedding is not the better feature here.
3. **The protocol leaves one real rig per fold.**
   - With FPV a test source (entry 143), the pool is Nightjar and d3set. Holding one out leaves the other, plus
     DDL near drones. Each fold is single-rig training again.
   - Gates (b) ≥ 0.77 and (c) ≥ 0.80 (entry 131) were set from in-rig figures (it9: d3set 0.823; it11r CRNN:
     Nightjar 0.770). The fold-N CRNNs reach the same level in-rig (d3set 0.825-0.837).
   - The gap to the gates is the unseen-rig penalty, not a weaker model.
   - Glasgow d1-d16 (allowed, entry 143) are not in `RangeSampler`'s pool, because the manifest has no Glasgow
     rows. Adding them would give every fold two real rigs.
4. **A clue to it9's room failure (hypothesis, not tested).**
   - No 11.2 model inverts the unseen Glasgow room as it9 does (AUC 0.092).
   - The d3set-trained CRNNs rank it at 0.97-0.995; the Nightjar-trained ones at 0.55-0.71.
   - it9 alone trained on FPV and DroneNoise (entry 145), so the room inversion likely comes from one of them.
   - Separately, every fold-N CRNN's room FA at its frozen threshold is far above 5%: the threshold frozen on
     d3set's 12.3 min of background does not transfer to the room.

**Not run.** Glasgow d1-d16 in the pool; per-rig negative balancing (entry 131's fallback); the
FPV/DroneNoise ablation of the room inversion. These are the candidate next steps.

