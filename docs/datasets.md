# Raw audio datasets: provenance

Everything in `data/` is exactly as the host published it. **Nothing has
been unzipped, resampled, renamed or otherwise changed.** Each file was given
its final name only after its MD5 matched the checksum the host publishes.

Fetched with `scripts/download_datasets.ps1` on 2026-10-10. Rerun that script
on any machine to reproduce `data/`; it resumes interrupted downloads and
skips files that are already complete.

Each dataset folder also holds `_source_record.json`: the host's own API record
(title, authors, licence, description, per-file sizes and checksums), saved
verbatim at download time.

`data/` is git-ignored, so the audio never reaches GitHub. To get it on a new
machine, clone the repo and run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\download_datasets.ps1
```

That needs about 40 GB free and takes roughly an hour. It only requires
`curl.exe`, which ships with Windows 10 and 11.

## Selection criteria

Ground-based acoustic drone **detection** (localisation is not a goal). A
dataset was accepted only if it has real recordings rather than synthetic or
noise-augmented ones, documented provenance, per-sample labels or metadata, and
a clear licence.

## What is here

| Folder | Role | Size | Licence |
|---|---|---|---|
| `DDL_Surrey/` | Core drone positives, plus two noise-only sessions | 12.63 GB | CC BY 4.0 |
| `DroneNoise_Salford/` | Calibrated drone positives, other drones and mics | 0.74 GB | CC BY 4.0 |
| `Halmstad/` | Drone, helicopter and background clips | 0.31 GB | CC0 1.0 |
| `FSD50K/` | Negatives, to be filtered by per-clip licence | 24.68 GB | per clip |

### DDL_Surrey

- **Title:** DDL: Dataset and Baseline Methods for Drone Detection and Localization Using Sound
- **Authors:** Saeid Safavi (University of Surrey), Mohammad Amin Safavi (DeepX Inc.), Ben Cook (Airspeed), Wenwu Wang (University of Surrey)
- **Source:** https://zenodo.org/records/6459183
- **DOI:** 10.5281/zenodo.6459183 (published 2023-04-18)
- **Licence:** CC BY 4.0
- **File:** `MLSP_2022_Real_Data.zip`, 12,628,837,686 bytes
- **Deliberately not downloaded:** `MLSP_2022_Synth_Data.zip` (4.6 GB), which is rendered rather than recorded.
- **Content, as verified in the files** (report entry 006): 62,103 WAV files,
  96 kHz, 8 channels, 32-bit PCM, in 36 session folders. DJI Mini 2 on
  2021-03-29 (5 sessions) and 2021-03-31 (19 sessions), DJI Phantom 4 Pro on
  2021-03-31 (12 sessions).
- **There are no no-drone recordings.** The host describes a class `XXXX`
  for them; no such file is in the archive. Every file is a drone.
- **Metadata** is encoded in each filename. The layout below is what the
  files actually use; it differs from the host's description (which gives a
  13-character timestamp and two separate session IDs):

  | Index | Field |
  |---|---|
  | 0-13 | Timestamp, `YYYYMMDDHHMMSS` |
  | 14-17 | Class: `MINI` (DJI Mini 2) or `PRO4` (DJI Phantom 4 Pro) |
  | 18-20 | Bearing to target, degrees |
  | 21-23 | Range to target, metres |
  | 24-26 | Target altitude above ground, metres (negative written as `0-1`) |
  | 27-30 | Ambient temperature, 0.1 K steps |
  | 31 | `R` real |
  | 32-42 | Session ID, `DDMMYY-Tnnn` |
  | 44-49 | Sequence number within the recording |

- **Known defects** (handled in `scripts/prepare_data.py`):
  - 3,945 files are header-only and hold no audio.
  - 2,629 files repeat the audio of an earlier file in the same session under
    different labels (sessions `290321-T004` and `290321-T005`).
  - In 10 sessions one or two files hold many seconds of audio under one
    label; from there on the range, bearing and time in the file names do not
    line up with the audio.
- **Use:** files are 100 ms and consecutive ones are continuous audio, so
  they are joined per session. Split by session, never randomly. Use the
  range labels only where `ddl_clips.csv` marks them reliable.

### DroneNoise_Salford

- **Title:** DroneNoise Database (version 3)
- **Authors:** Carlos Ramos-Romero, Antonio J. Torija Martinez, Nathan Green, César Asensio (University of Salford; EPSRC-funded DroneNoise project)
- **Source:** https://salford.figshare.com/articles/dataset/DroneNoise_Database/22133411
- **DOI:** 10.17866/rd.salford.22133411.v3 (published 2024-05-13, modified 2025-01-23)
- **Licence:** CC BY 4.0
- **Files:** 174 files, 742 MB: per-channel calibration WAVs, one WAV per event
  per microphone (`M1` to `M9`), and `DNM_SQM_FLYO_AESCT.xlsx` with
  sound-quality metrics. Per-file MD5s are in `_source_record.json`.
- **Host quirk:** figshare lists 175 entries because
  `Ed_3p_10_H00_N_C_nw_ev1_M5.wav` was uploaded twice with an identical MD5.
  One copy is kept.
- **Content, as verified in the files** (report entry 006): 50 kHz, mono,
  32-bit float. 9 calibration files and 164 event files covering 20 distinct
  flight events (14 flyovers at 15 m/s, 6 hovers), all at 10 m above ground,
  recorded at Edzell, Scotland, on 17 August 2022. Mostly 20 s or 30 s long.
- **File names** are `site_drone_height_operation_payload_start_wind_event_mic`,
  for example `Ed_M3_10_F15_N_E_dw_ev1_M5`: Edzell, Matrice 300, 10 m,
  flyover at 15 m/s, no payload, starting east, downwind, event 1,
  microphone 5. `H00` is a hover.
- **Drone codes:** `3p` DJI Mini 3 Pro, `M3` DJI Matrice 300 RTK, `Yn` Yuneec
  H520E. `Fp` is not named by the host; the spreadsheet gives it a gross
  weight of 795 g.
- **Use:** every file contains a drone, so this supplies positives only. The
  nine microphones record the same event, so they are not independent: keep
  all microphones of one event on the same side of any split. In flyovers
  the drone is loud only around the pass-by.

### Halmstad

- **Title:** Drone-detection-dataset, first release v1.0.0 (Svanström et al., Halmstad University)
- **Author on record:** Fredrik Svanström
- **Source:** https://zenodo.org/records/5500576, a snapshot of https://github.com/DroneDetectionThesis/Drone-detection-dataset/tree/v1.0.0
- **DOI:** 10.5281/zenodo.5500576 (published 2020-05-16)
- **Paper:** Svanström, Englund, Alonso-Fernandez, "Real-Time Drone Detection and Tracking With Visible, Thermal and Acoustic Sensors", ICPR 2020
- **Licence:** CC0 1.0, confirmed from the `LICENSE` file inside the zip
  (report entry 006). Zenodo tags the record only as `other-open`.
- **File:** `Drone-detection-dataset-v1.0.0.zip`, 311,348,350 bytes, MD5 `5d9b891a87857b3d9ee8872e8a1f0f0f`
- **Content:** 90 ten-second audio clips, 44.1 kHz stereo 16-bit: 30 Drone,
  30 Helicopter, 30 Background, recorded at three Swedish airports. The zip
  is the whole repository, so it also contains the video part of the dataset.
  It is the only source here where drone and non-drone share one recording
  setup.
- **Check before use:** part of the overall dataset comes from YouTube, so
  check whether any audio clip is affected. The background class includes the
  camera rig's own servo noise, which a model could key on.

### FSD50K

- **Title:** FSD50K, version 1.0
- **Authors:** Eduardo Fonseca, Xavier Favory, Jordi Pons, Frederic Font, Xavier Serra (Music Technology Group, Universitat Pompeu Fabra)
- **Source:** https://zenodo.org/records/4060432
- **DOI:** 10.5281/zenodo.4060432 (published 2020-10-02)
- **Paper:** Fonseca et al., "FSD50K: an open dataset of human-labeled sound events", IEEE/ACM TASLP 30, 2022 (arXiv:2010.00475)
- **Licence:** each clip carries its own Creative Commons licence, listed per
  clip in the metadata. The dataset as a whole is CC BY.
- **Files and MD5s:**

  | File | Bytes | MD5 |
  |---|---|---|
  | `FSD50K.dev_audio.z01` | 3,221,225,472 | `faa7cf4cc076fc34a44a479a5ed862a3` |
  | `FSD50K.dev_audio.z02` | 3,221,225,472 | `8f9b66153e68571164fb1315d00bc7bc` |
  | `FSD50K.dev_audio.z03` | 3,221,225,472 | `1196ef47d267a993d30fa98af54b7159` |
  | `FSD50K.dev_audio.z04` | 3,221,225,472 | `d088ac4e11ba53daf9f7574c11cccac9` |
  | `FSD50K.dev_audio.z05` | 3,221,225,472 | `81356521aa159accd3c35de22da28c7f` |
  | `FSD50K.dev_audio.zip` | 2,306,663,327 | `c480d119b8f7a7e32fdb58f3ea4d6c5a` |
  | `FSD50K.eval_audio.z01` | 3,221,225,472 | `3090670eaeecc013ca1ff84fe4442aeb` |
  | `FSD50K.eval_audio.zip` | 3,037,675,767 | `6fa47636c3a3ad5c7dfeba99f2637982` |
  | `FSD50K.ground_truth.zip` | 334,701 | `ca27382c195e37d2269c4c866dd73485` |
  | `FSD50K.metadata.zip` | 6,700,838 | `b9ea0c829a411c1d42adb9da539ed237` |
  | `FSD50K.doc.zip` | 6,984 | `3516162b82dc2945d3e7feba0904e800` |

- **Content:** 51,197 human-labelled Freesound clips in 200 AudioSet-ontology
  classes. The dev and eval sets share no Freesound uploader.
- **The audio is a split zip.** `.z01` to `.z05` plus `.zip` are one archive;
  all parts must be present to extract. The Python standard library cannot
  read it; extract with 7-Zip (see `scripts/prepare_data.py`).
- **As processed** (report entry 007): 43,200 clips, 89.3 hours. 7,818 clips
  were left out for their licence (CC BY-NC or Sampling+) and 179 because
  title, description or tags mention a drone or a related word.
- **Use:**
  - The dev set's own train/val division shares 774 uploaders, so it is not
    leak-free. Split dev by uploader instead. Dev and eval share none.
  - About 12% of clips are shorter than one second.

## Processed form

`scripts/prepare_data.py` writes everything to `data/processed/` as 16 kHz
mono, with `manifest.csv` (one row per recording: source, label, kind, split
group, licence, level) and `ddl_clips.csv` (one row per original DDL file).
Only the format is changed there: no gain, normalisation, filtering or
mixing.

| Source | Recordings | Audio | Label |
|---|---|---|---|
| DDL | 36 sessions | 99.1 min | drone |
| DroneNoise | 164 files, 20 events | 59.2 min (about 6.6 min per microphone) | drone |
| Halmstad | 90 clips | 15.0 min | 30 drone, 30 helicopter, 30 background |
| FSD50K | 43,200 clips | 89.3 h | not drone |

## Considered and left out

| Dataset | Reason |
|---|---|
| DDL synthesized zip | Rendered, not recorded |
| DroneAudioDataset (Al-Emadi) | Indoor recordings artificially mixed with noise; no licence file; reported restrictions on defence use |
| DADS (Hugging Face) | Aggregates Al-Emadi, SPCup19 and DREGON; inherited augmentation, likely duplicates, licence conflicts |
| DREGON, DroneAudioSet | Microphones mounted on the drone itself: wrong perspective for ground-based detection |
| ESC-50 | CC BY-NC overall; class-dependent bandlimiting in some sources |
| Ljungbyhed (Lund) | Value was position ground truth; no licence found for the audio |
| MMAUD (NTU) | CC BY-NC-SA; 11-20 GB ROS bags; value was trajectory ground truth |
| RWDA (IEEE DataPort) | Good fit for detection, but needs an IEEE DataPort subscription |
| Wang et al. 32-class set | Environment confounded with class; laptop microphone; no confirmed licence |

## Known gap

No source except Halmstad has drone and non-drone audio from the same
recording setup, and Halmstad is 15 minutes. DDL, the main source of
positives, has no negatives at all, so in training the label is confounded
with the recording chain: a model can tell the classes apart by microphone,
level or background without detecting a drone. Nothing here was recorded
with the deployment microphone at a deployment-like site. The single most
valuable addition would be non-drone recordings from a setup that also
recorded drones.

## Added on 2026-10-12 and 2026-10-14

Twelve more datasets were found after iteration 2. Their full provenance
(authors, DOIs, licences, per-file checksums, what was deliberately not
downloaded) is in `README_new_datasets.md` in the Google Drive data folder,
copied to `data/README_new_datasets.md`; the download script and log are in
`data/_download_tools/`. Fetched so far (all checked against the host's
checksums, report entries 024 and 027):

| Folder | Role | Licence |
|---|---|---|
| `Nightjar_HF/` | Training and test rig: DJI Mini 2, UMA-16 array, drone and no-drone audio, GPS range | CC BY 4.0 |
| `D3set_IntRoLab/` | Training and test rig: Holybro X500, 16-mic array, drone and no-drone audio, GPS range | GPL-3.0 (internal use, decided by Mads 2026-10-14) |
| `Ljungbyhed_Lund/` | Further rig, recall only: three drone types, 12-mic array, x/y/z ground truth | none stated (internal use only; ask the authors before publishing) |
| `Glasgow_Drone_Authentication/` | Further rig: 24 drones at 1 m and 5 m, one hour of noise | CC BY 4.0 |
| `ITU_ARIS_Lab/` | Sealed test: drone presence out to about 2 km, background from the same session; no range per segment | CC BY 4.0 |
| `RODENT_ICAERUS/` | Sealed test: four recordings with DJI and Litchi logs; alignment not yet established | CC BY 4.0 |

Held back at Mads's request (only `_source_record.json` present):
`Acoustic_UAV_Field_Recordings`, `UaVirBASE`, `Fraunhofer_IDMT_2022`,
`DJI_Mini2_Wojcik`, `CiO_Drone_Acoustic`, `Drone_Hums_Beirut`.

Facts established on the files, not in the hosts' descriptions:

- Lund's WAVs are 32-bit integer PCM, 12 channels, 96 kHz; the 13
  recordings hold 90.5 min. In `10_flower3` the ground truth runs 29 s past
  the end of the audio. The ground truth's `drone_live` flag is True on every
  row; before the ground truth the drone stands at its take-off point and
  whether its motors run is not recorded.
- d3set's `actuator.csv` uses milliseconds from the start of the audio,
  unlike its other telemetry (Unix milliseconds). `Training/25 oct` was
  recorded at 8 kHz and is not used.
- Nightjar's rig position is in its own analysis files (41.534765,
  -75.595254); `OSD.height` in the DJI CSVs is in feet.
- No dataset records wind speed.

Processing: `scripts/prepare_new_data.py` (entry 023) writes each recording
once and one manifest row per stretch of drone, background or uncertain
state, plus `range_track.csv` with the slant range at 10 Hz.
