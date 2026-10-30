# Sound taxonomy (supervisor note S10)

What each class of sound in the stem library (`data/processed/stems/`,
report entry 052) is like, measured from its stems with
`uv run python scripts/scenes.py describe`. The point of describing them
first: a class is defined by what it does to the signal, which is also what
decides whether it can mask or imitate a drone.

Columns (means over the class's stems):

- **Spectral centroid**: where the energy sits.
- **Energy below 300 Hz**: the low end, where wind and engines live.
- **Frame-level spread**: standard deviation of 10 ms frame levels (within
  60 dB of the loudest). Low = steady (wind, engines, drones); high = events.
- **Crest factor**: peak over RMS. High = impulsive (gunfire, explosions).
- **Cepstral peak**: height of the largest cepstral peak for a fundamental
  of 60-600 Hz, in standard deviations. High = harmonic, with a fundamental
  and its overtones, as a drone's blade-pass tone has (cepstrum: S5).
- **Spectral overlap with drone**: the shared area of the class's and the
  drones' mean spectra in 32 ERB bands (1 = same shape, 0 = disjoint). High
  overlap means it can mask the drone, and a harmonic, steady sound with
  high overlap can also be taken for one.

| Group | Class | Stems | Spectral centroid (Hz) | Energy below 300 Hz (%) | Frame-level spread (dB) | Crest factor (dB) | Cepstral peak (z) | Spectral overlap with drone |
|---|---|---|---|---|---|---|---|---|
| drone | glasgow_1m | 24 | 3136 | 4 | 0.8 | 14.8 | 5.1 | 0.87 |
| background | rain | 3 | 2907 | 8 | 1.5 | 19.1 | 2.6 | 0.66 |
| background | stream | 3 | 3033 | 35 | 2.7 | 16.2 | 3.1 | 0.64 |
| background | insects | 3 | 3276 | 41 | 3.1 | 19.9 | 3.8 | 0.36 |
| background | traffic | 1 | 81 | 94 | 4.2 | 19.3 | 3.4 | 0.33 |
| distracter | gunfire | 3 | 1430 | 25 | 12.7 | 23.6 | 4.1 | 0.47 |
| distracter | explosion | 3 | 1113 | 73 | 13.6 | 19.5 | 3.2 | 0.42 |
| distracter | car | 3 | 373 | 88 | 8.9 | 17.9 | 3.5 | 0.49 |
| distracter | truck | 3 | 1090 | 47 | 2.7 | 16.7 | 2.7 | 0.35 |
| distracter | motorcycle | 3 | 232 | 86 | 8.8 | 19.3 | 2.7 | 0.14 |
| distracter | aircraft | 3 | 294 | 82 | 2.9 | 10.5 | 2.8 | 0.18 |
| distracter | dog | 3 | 553 | 27 | 13.0 | 18.1 | 4.4 | 0.28 |
| distracter | shout | 3 | 1109 | 8 | 11.7 | 12.5 | 5.0 | 0.32 |
| distracter | birds | 3 | 2737 | 30 | 12.1 | 19.2 | 5.5 | 0.32 |
| distracter | siren | 3 | 840 | 1 | 15.9 | 15.5 | 5.9 | 0.27 |
| distracter | idling | 2 | 1922 | 11 | 4.1 | 16.6 | 2.5 | 0.56 |
| distracter | power_tool | 3 | 4067 | 1 | 6.3 | 17.2 | 5.6 | 0.57 |
| distracter | thunder | 3 | 75 | 98 | 9.8 | 21.6 | 3.4 | 0.18 |
| wind | wind | 3 | 426 | 45 | 3.5 | 12.9 | 3.8 | 0.16 |
| war | small_arms | 3 | 967 | 47 | 12.9 | 21.3 | 3.1 | 0.42 |
| war | jet | 3 | 644 | 33 | 3.8 | 13.8 | 2.6 | 0.31 |
| distracter | helicopter | 3 | 437 | 44 | 7.5 | 18.5 | 3.9 | 0.28 |
| distracter | generator | 3 | 1150 | 43 | 3.4 | 13.9 | 3.3 | 0.40 |
| distracter | mosquito | 3 | 2314 | 22 | 8.2 | 17.7 | 7.5 | 0.36 |
| distracter | buzzing_insect | 3 | 2311 | 49 | 8.0 | 21.6 | 4.2 | 0.46 |
| distracter | prop_plane | 3 | 729 | 47 | 4.5 | 19.5 | 2.7 | 0.36 |
| war | machine_gun | 3 | 901 | 49 | 15.6 | 19.2 | 3.5 | 0.46 |
| war | mortar_artillery | 5 | 445 | 58 | 13.3 | 19.1 | 3.3 | 0.30 |
| war | explosion_far | 3 | 61 | 100 | 15.1 | 23.4 | 5.4 | 0.08 |
| war | tank | 2 | 375 | 75 | 3.0 | 12.2 | 2.4 | 0.19 |
| war | air_raid_siren | 1 | 795 | 0 | 2.4 | 13.2 | 6.5 | 0.17 |
| war | shout | 3 | 1038 | 4 | 15.0 | 21.3 | 5.6 | 0.40 |
| distracter | scooter | 3 | 1213 | 34 | 8.1 | 22.1 | 7.0 | 0.54 |
| distracter | mower | 3 | 1818 | 52 | 3.3 | 14.5 | 3.6 | 0.39 |
| distracter | chainsaw | 3 | 1444 | 41 | 2.3 | 14.0 | 4.6 | 0.34 |

The drones' strongest ERB bands (mean of the 24 Glasgow stems, at 1 m):
343 Hz, 626 Hz, 1886 Hz, 2110 Hz, 6170 Hz, 6847 Hz.

How to read it as a classification, by the columns: steady (low spread) +
harmonic (high cepstral peak) + high overlap describes a drone look-alike,
the likely source of false alarms; steady + high overlap without harmonicity
masks without imitating; impulsive classes (high spread and crest factor)
mask only for moments. Which classes fall where is read from the table above,
and tested against the detectors in report entry 053.

Means over 1-3 stems per class are a description, not a classifier: a class
with one or two stems (traffic, idling) is described by those recordings.
