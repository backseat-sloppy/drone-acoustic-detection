# War-shaped noise

Steady noise whose spectrum is the long-term average spectrum (LTAS) of war
recordings, the way speech-shaped noise follows the LTAS of speech. It is a
controlled background for testing the CNN and the DSP baselines at exact SNRs.
The Pd patch only generates the noise. The drone is added, and the SNR set,
when the noise is mixed with a drone recording (section "Set the SNR").

| File | What it is |
|---|---|
| `scripts/war_shaped_noise.py` | builds the gain table, the LTAS plot and a reference WAV; `--check` compares a recording with the target |
| `Pure-data+Script/WarShapedNoise/war_noise.pd` | Pd vanilla patch: `noise~` (seeded) -> FFT shaper -> level -> record / listen |
| `Pure-data+Script/WarShapedNoise/war_gain.txt` | 513 magnitude gains, FFT bins 0..512 (nfft 1024, 44.1 kHz), loaded by the patch |

## Build the gain table

```bash
uv run python scripts/war_shaped_noise.py --sr 44100 --octave-frac 3
```

The project runs at 44.1 kHz: the war clips and the drone renders are 44.1
kHz. The script's default is 48 kHz, so pass `--sr 44100` (also to `--check`).
Add `--drone FILE` to overlay a drone recording on the LTAS plot.

The default input is `data/processed/stems/war/` (23 FSD50K clips, 8 classes).
Any folder of WAV/MP3 files works as the first argument. Each file is mixed to
mono, resampled to `--sr`, and its Welch spectrum (Hann, `--nfft` 1024, 50 %
overlap) is scaled to unit total power before the files are averaged in the
power domain, so a long or loud file counts as much as a short quiet one. The
gain is the square root of that average, scaled so white noise keeps its RMS
(unit power gain), with bin 0 (DC) set to 0.

`--nfft` must equal the block size in the patch (`block~ 1024 4`). If you
change one, change the other, the `1024` in the Hann-window `expr`, the
array sizes and the `/~ 1536` (= nfft x 1.5).

## Run the patch

Open `war_noise.pd` in Pd 0.55 or newer (0.57 tested) with Pd at 44100 Hz
(Media > Audio Settings). At load it reads `war_gain.txt`, builds the Hann
window, sets the level to -20 dBFS and turns DSP on. If Pd runs at another
rate, the console prints a warning: the table's bins would then sit at the
wrong frequencies.

Two panels:

1. **NOISE**: seed (click 1 or 1001, or send `wsn-seed N`), `noise~`,
   `pd shaper` (the FFT part), the gain table.
2. **LEVEL AND OUTPUT**: noise level in dBFS RMS (-30/-20/-10, type a value,
   or send `wsn-level L`), the record buttons (mono 32-bit float
   `war_noise_out.wav`, or `wsn-rec open -bytes 4 FILE` and `wsn-rec start` /
   `stop`), and `monitor_volume` for the speakers (starts at 0.1, followed by
   `clip~ -1 1`; it does not change the recording).

The level is set from a fixed calibration: with the unit-gain table the
shaper's output has RMS -5.0 dBFS, so the gain is `10^((L + 5)/20)`. Measured:
-19.99 dBFS for L = -20, -29.99 for L = -30. Peaks sit about 13 dB above the
RMS, so keep L at -14 dBFS or lower if the file will be stored as integer
PCM.

Headless, like `Pure-data+Script/EnviromentTest/render.py`: open the patch
and a small launcher patch with
`pd -nogui -nosound -batch -r 44100 -open war_noise.pd -open launcher.pd`. The
launcher's loadbang sends `wsn-seed`, `wsn-level`, `wsn-rec open ...` and
`wsn-rec start`, and after a `delay` sends `wsn-rec stop` and, from a second
`delay`, `pd quit`. (A message box that starts with `;` has no outlet, so
chain the delays from the `delay` object, not from the message box.)

## Set the SNR

Mix in Python, where both signals are known exactly. SNR here is broadband
RMS; for an approaching drone it is set at a reference distance of 150 m and
then held (Mads, 2026-10-30): the noise level stays fixed and the drone rises
out of it.

```python
import numpy as np, soundfile as sf
drone, fs = sf.read("drone.wav")            # 44.1 kHz mono
noise, _ = sf.read("war_noise_out.wav")     # at least as long as the drone
noise = noise[:len(drone)]
t_ref = (start_dist - 150) / speed          # s, from the flight's .json
w = int(0.68 * fs)                          # reference window around t_ref
i = int(t_ref * fs)
rms = lambda v: np.sqrt(np.mean(v ** 2))
g = rms(drone[i - w // 2:i + w // 2]) / rms(noise) / 10 ** (snr_db / 20)
mix = drone + g * noise
```

Report a result as "SNR X dB at 150 m, broadband RMS", with the seed, the
sample rate and the detector's window and hop.

## 1. Check that the Pd output matches the target

```bash
uv run python scripts/war_shaped_noise.py --sr 44100 --check Pure-data+Script/WarShapedNoise/war_noise_out.wav
```

It prints the RMS, target and measured level per 1/3-octave band from 50 Hz
to 20 kHz, the largest and RMS difference, and writes
`war_noise_out_check.png`. With 20 s of noise expect about 0.5 dB at most per
band (measured: 0.51 dB max, 0.25 dB RMS). The same check on
`war_noise_ref.wav` gives 0.39 dB max, 0.14 dB RMS, all of it below 1 kHz:
that is the resolution of the check itself (a 1024-point Hann window on
bands only one or two table bins wide), not an error in the file.

### The most precise file: `war_noise_ref.wav`

For a fixed test clip, use `war_noise_ref.wav` (20 s, 44.1 kHz, -20 dBFS RMS,
peak -6.5 dBFS, seed 0; `--seconds` and `--seed` change it). It is made in
the frequency domain with exactly the target magnitude in every FFT bin and
a random phase, so its spectrum has no random error: measured on its own
full-length FFT, every 1/3-octave band from 50 Hz to 20 kHz matches the
table to 0.0000 dB. Gaussian or Pd noise has a random magnitude per bin
(about +-0.5 dB per band at 20 s). The clip is periodic, so it also loops
without a click. Pd stays useful for live listening and for new noise
per seed in real time.

### Pd FFT scaling: what is known and how it was checked

Pd vanilla's `rfft~` and `rifft~` are unnormalised: forward then inverse gives
N times the input. With a Hann window before `rfft~` and after `rifft~` and 4x
overlap, the overlap-added Hann^2 sums to 1.5. So the patch divides by
N x 1.5 = 1536, which is also what Pd's own example `I03.resynthesis.pd` does
(512 x 1.5 = 768). Rather than trust that, it was measured (entry 068 of
`docs/REPORT.md`): with the unit-power gain table the shaper's output RMS is
-5.0 dBFS against -4.8 dBFS for `noise~` (uniform -1..1). A wrong
normalisation would be off by 6 dB or more (a factor of 2 for a missed
overlap, 60 dB for a missed N). To repeat the check with a flat table, write
513 ones into `war_gain.txt`: the shaper output must then have the `noise~`
RMS, -4.8 dBFS.

The gain array has 513 points; `tabreceive~` fills the rest of the
1024-sample block with zeros, which `rifft~` ignores (it uses bins 0..512).

The shaper is a filter applied per block, so it is circular convolution with
time aliasing. For noise and a smooth gain curve this does not matter, which
is what the per-band check shows. A gain curve with sharp peaks would need a
larger block.

## 2. Keep training noise and test noise apart

The model is frozen at present (no training in this phase), so war-shaped
noise is used for testing only. If it ever enters training:

- Seeds: training noise uses seeds 1-999, validation and test noise 1000 and
  up. Every rendered file names its seed. Same seed gives the same samples;
  seeds 7 and 1001 correlate at 0.005.
- Spectrum: the LTAS itself is something a model can learn. Build the
  training table and the test table from disjoint war recordings (split by
  source file, never by clip), for example two folders, and check the two LTAS
  plots look alike. If the ML model trains on war-shaped noise and the DSP
  baseline is tuned on nothing, say so next to the result.
- Thresholds: the frozen thresholds stay frozen. Do not re-tune a threshold on
  a war-shaped-noise condition and then score that same condition.

## 3. Limitations and a "realistic" condition

War-shaped noise is stationary. It has the long-term spectrum of war but none
of its time structure: no explosions, no bursts of gunfire, no passing
vehicles, no silence between events. Detectors fail differently on that:
energy and comb detectors are sensitive to broadband transients, a CNN may
alarm on a single blast. So stationary noise alone overstates how a detector
does in the field, and an SNR in one condition is not comparable with the
same number in the other.

Run a second, "realistic" condition on top:

1. Stationary bed: war-shaped noise at the chosen SNR, as above.
2. Events: the war track from `scripts/war_track.py` (events from
   `data/processed/stems/war/`, logged in `events.csv` with start times), mixed
   at its own event-to-drone level, swept like the SNR (for example the -20 to
   0 dB grid of `scripts/drone_war_grid.py`).
3. Report both conditions side by side, and in the realistic one also score
   false alarms inside and outside event windows from `events.csv`, so a burst
   of alarms on explosions shows up as such.

Keep the events for training and testing in disjoint source files, like the
noise spectrum above.

The FSD50K war clips are 44.1 kHz, so the table has no content above
22.05 kHz.

What the spectrum is made of (44.1 kHz, per class, equal weight per clip):
the low end comes from explosions, tanks and artillery, while three `shout`
clips cause 47 % and the one air-raid siren 14 % of the energy at
600-750 Hz, the band where the drone's harmonics are strongest. The table
keeps them (Mads, 2026-10-30, report entry 073); report that the noise
includes shouting when comparing results.

Plots in `Pure-data+Script/WarShapedNoise/` (all 44.1 kHz): `war_ltas.png`
(LTAS with a drone overlaid), `war_ltas_by_class.png` (each class's share of
the average; the lines add up to the total), `war_noise_ref_check.png` and
`war_noise_out_check.png` (Python reference and a 20 s Pd render, seed 7,
-20 dBFS, against the target).
