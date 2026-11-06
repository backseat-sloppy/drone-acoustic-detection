# PeterDrones

The most static, clean drone recordings found in the project's datasets:
steady level, steady spectrum, clear rotor lines and little wind. Picked by
measuring every 10 s window (hop 5 s) of drone audio in `data/processed/`
and the Glasgow stems; not checked by ear.

All files are mono float32 WAV at the level they were recorded at: no
normalisation, no common gain. Quiet files are quiet on purpose.

| File | Drone | Distance | Length | RMS | Source |
|---|---|---|---|---|---|
| `01_glasgow_hover_1m_d03/d06/d07/d10.wav` | four of the 24 Glasgow drones, hover | 1 m | 10 s each, 44.1 kHz | -52 to -54 dBFS | Glasgow |
| `02_ddl_310321-T015_P4Pro_12m.wav` | DJI Phantom 4 Pro, hover | 12-13 m | 100 s, 16 kHz | -53 dBFS | DDL |
| `03_d3set_23oct_vol3_X500_3m.wav` | Holybro X500, hover (GPS range +-0.16 m) | 3.3 m | 35 s, 16 kHz | -48 dBFS | d3set |
| `04_nightjar_0804-4_Mini2_13m.wav` | DJI Mini 2 (GPS range +-0.05 m) | 12.8 m | 30 s, 16 kHz | -83 dBFS | Nightjar |
| `05_ddl_310321-T010_P4Pro_23m.wav` | DJI Phantom 4 Pro | 23 m | 20 s, 16 kHz | -53 dBFS | DDL |

Where in the source each excerpt starts:
- Glasgow: the 10 s stems built by `scripts/build_stems.py` from
  `_UAV_2022…_d{3,6,7,10}_hover_1m_…WAV`, with the stem gain removed again.
- DDL `310321-T015`: 0-100 s; `310321-T010`: 35-55 s (session audio joined
  from the 100 ms files, channel 0, resampled to 16 kHz).
- d3set `Validation/23_oct_vol3`, segment #02: 270-305 s (`matrix.wav`, one channel).
- Nightjar `nightjar-flight-20260804-4`, segment #01: 140-170 s (UMA-16, one channel).

## Credits and licences

- **Glasgow**: Diao, Zhang, Zhao, Khamis, "Drone Authentication via Acoustic
  Fingerprint", ACSAC 2022. University of Glasgow,
  DOI 10.5525/gla.researchdata.1348. CC BY 4.0.
- **DDL**: Safavi, Safavi, Cook, Wang, "DDL: Dataset and Baseline Methods for
  Drone Detection and Localization Using Sound". DOI 10.5281/zenodo.6459183.
  CC BY 4.0.
- **Nightjar**: joshruby, `nightjar-flight-20260804-4` on Hugging Face.
  CC BY 4.0.
- **d3set**: IntRoLab, "d3set - Drone delivery detection dataset",
  https://github.com/introlab/d3set. GPL-3.0.

The excerpts were cut and resampled; no other change was made.
