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
| `Halmstad/` | Drone, helicopter and background clips | 0.31 GB | see below |
| `FSD50K/` | Negatives, to be filtered by per-clip licence | 24.68 GB | per clip |

### DDL_Surrey

- **Title:** DDL: Dataset and Baseline Methods for Drone Detection and Localization Using Sound
- **Authors:** Saeid Safavi (University of Surrey), Mohammad Amin Safavi (DeepX Inc.), Ben Cook (Airspeed), Wenwu Wang (University of Surrey)
- **Source:** https://zenodo.org/records/6459183
- **DOI:** 10.5281/zenodo.6459183 (published 2023-04-18)
- **Licence:** CC BY 4.0
- **File:** `MLSP_2022_Real_Data.zip`, 12,628,837,686 bytes
- **Deliberately not downloaded:** `MLSP_2022_Synth_Data.zip` (4.6 GB), which is rendered rather than recorded.
- **Content:** DJI Phantom 4 Pro and DJI Mini 2, 8-channel array, rural and suburban UK sites.
- **Metadata** is encoded in each filename, by character index:

  | Index | Field |
  |---|---|
  | 0-12 | UTC timestamp |
  | 13-16 | Class: `MINI` (DJI Mini 2), `PRO4` (DJI Phantom 4 Pro), `XXXX` (no drone) |
  | 17-19 | Bearing to target, degrees |
  | 20-22 | Range to target, metres |
  | 23-25 | Target altitude above ground, metres |
  | 26-29 | Ambient temperature, 0.1 K steps |
  | 30 | `R` real, `S` synthesized |
  | 31-36 | Flight session ID |
  | 37-42 | Recording session ID |
  | 43-48 | Sample sequence ID, for merging consecutive samples |

- **Check before use:** the Zenodo page implies 200 ms after merging while the
  paper says 100 ms clips. Read the clip length and sample rate from the WAV
  headers. Split by session ID, never randomly, or adjacent clips leak between
  train and test.

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
- **Content:** sUAS overflights recorded at Edzell, Scotland, on 17 August 2022
  with a 9-microphone array. Filename codes `3p`, `M3` and `Yn` correspond to
  DJI Mini 3 Pro, DJI Matrice 300 RTK and Yuneec H520E. The code `Fp` is not
  explained by the host.
- **Check before use:** every file contains a drone, so this supplies positives
  only. The sample rate is not stated on the record; read it from the headers.

### Halmstad

- **Title:** Drone-detection-dataset, first release v1.0.0 (Svanström et al., Halmstad University)
- **Author on record:** Fredrik Svanström
- **Source:** https://zenodo.org/records/5500576, a snapshot of https://github.com/DroneDetectionThesis/Drone-detection-dataset/tree/v1.0.0
- **DOI:** 10.5281/zenodo.5500576 (published 2020-05-16)
- **Paper:** Svanström, Englund, Alonso-Fernandez, "Real-Time Drone Detection and Tracking With Visible, Thermal and Acoustic Sensors", ICPR 2020
- **Licence:** Zenodo tags the record `other-open`. The earlier survey for this
  project recorded the audio as CC0. **Confirm against the licence file inside
  the zip before redistributing.**
- **File:** `Drone-detection-dataset-v1.0.0.zip`, 311,348,350 bytes, MD5 `5d9b891a87857b3d9ee8872e8a1f0f0f`
- **Content:** 90 ten-second audio clips labelled Drone, Helicopter or
  Background, recorded at three Swedish airports. The zip is the whole
  repository, so it also contains the video part of the dataset.
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
  all parts must be present to extract.
- **Check before use:**
  - Keep only CC0 and CC BY clips. Drop CC BY-NC and Sampling+.
  - Search clip titles and tags for `drone`, `quadcopter`, `UAV` and
    `multicopter` and remove the hits, or real drones end up as negatives.
  - Pick outdoor-relevant classes: vehicles, aircraft, engines, wind, rain,
    birds, tools.

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

Negatives are thin. DDL has only two noise-only sessions, and nothing here was
recorded with the deployment microphone at a deployment-like site. Evaluate by
holding out whole sources, for example train on DDL plus FSD50K-dev and test on
DroneNoise, Halmstad and FSD50K-eval.
