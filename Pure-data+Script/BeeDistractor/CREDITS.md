# Sources and licences

The bees in `bee_distractor.pd` play real FSD50K dev recordings with the background removed
(`make_tables.py`: only the bee's own harmonics are kept, frame by frame) as `bee_clean<k>.wav`,
looped, pitched and moved along synthetic flight paths. These loops and the WAV files here are
derived works of:

| Type | FSD50K id | Title | Uploader | Licence |
|---|---|---|---|---|
| 0 honeybee | 59451 | Bees_1.wav | digifishmusic | CC BY 3.0 |
| 1 bumblebee | 233783 | BUMBLEBEE 12.wav | delphidebrain | CC BY 3.0 |
| 2 bumblebee_2 | 79931 | bourdon.wav | klaskatt | CC BY 3.0 |

Rendered examples (`bee-seed 1` unless stated):

- `bee_example.wav`: type 0 (59451), defaults: 2 bees, 2 m away, flying in +-1 m, 1 m/s, 35 dB(A) at 1 m per bee, 10 s.
- `bee_swarm_3m.wav`: type 0 (59451), 8 bees, 3 m away, in +-1.5 m, 10 s.
- `bee_bumblebee.wav`: type 1 (233783), 1 bee, 1.5 m away, seed 3, 10 s.

Comparison figure `bee_vs_real.png` (`compare_real.py`) shows spectrograms of these real recordings,
next to Pd renders:

| FSD50K id | Title | Uploader | Licence |
|---|---|---|---|
| 201613 | Bees.wav | peachmaster | CC0 1.0 |
| 17349 | 20060326.marshes.beeswarm01.wav | dobroide | CC BY 3.0 |

CC BY 3.0: https://creativecommons.org/licenses/by/3.0/ - CC0 1.0: https://creativecommons.org/publicdomain/zero/1.0/

`bee_zoom_vs_real.png` (entry 161) shows 3 s and 0.6 s of 59451 and 233783 (above) next to Pd renders.
