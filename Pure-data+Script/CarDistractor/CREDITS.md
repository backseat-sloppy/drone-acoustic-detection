# Sources and licences

All source audio is from FSD50K (dev split), via `data/processed/stems/` (see `docs/datasets.md`).
The patch and the WAV files here are derived works: the engines are looped, pitched, filtered and
mixed with noise shaped like the pass-bys below.

Engine recordings (`car_engines.txt`):

| FSD50K id | Title | Uploader | Licence |
|---|---|---|---|
| 369049 | car_idle.wav | AndrewAlexander | CC0 1.0 |
| 369053 | car_idle_loop.wav | AndrewAlexander | CC0 1.0 |
| 387676 | Car engine running 4 | Jagadamba | CC BY 3.0 |
| 387677 | Car engine running 3 | Jagadamba | CC BY 3.0 |
| 387679 | Car engine idle running | Jagadamba | CC BY 3.0 |
| 401552 | SFX_Car_Engine_Outside_Idle.wav | GiocoSound | CC0 1.0 |

Tyre-noise spectrum (`car_tyre_gain.txt`), averaged from:

| FSD50K id | Title | Uploader | Licence |
|---|---|---|---|
| 425381 | audi a4 allroad 20TFSI exterior passby 70kmph following mono NTG3.wav | Soundholder | CC BY 3.0 |
| 425492 | BMW 114i E87 Exterior Passby Fast Mono.wav | Soundholder | CC BY 3.0 |
| 425865 | Volkswagen Golf II 1.6 Diesel Exterior Passby Fast Mono.wav | Soundholder | CC BY 3.0 |
| 425868 | Volkswagen Golf V 1.6 FSI exterior passby fast speed stereo ORTF 8040.wav | Soundholder | CC BY 3.0 |

Rendered examples:

- `car_example.wav`: engine 3 (387677, Jagadamba, CC BY 3.0), 70 km/h, 15 m, defaults otherwise.
- `car_60kmh_20m.wav`: engine 2 (387676, Jagadamba, CC BY 3.0), 60 km/h, 20 m, 85 dB(A) at 1 m, 12 s.

CC BY 3.0: https://creativecommons.org/licenses/by/3.0/ - CC0 1.0: https://creativecommons.org/publicdomain/zero/1.0/
