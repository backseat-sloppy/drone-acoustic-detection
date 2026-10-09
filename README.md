# Acoustic Drone Detection, Latency First

A pipeline built around one finding: **the classifier is 1 to 5 percent of
end-to-end latency.** Window fill and hop dominate. So "which model is
fastest" is a second-order question, and the headline result of this project is
a latency budget plus an accuracy-versus-delay curve, not a model.

## The latency budget

```
L = T_window_fill + T_hop_wait + T_features + T_model + T_smoothing + T_io
```

At the default configuration, a 1 second window is 1000 ms of that. A small
INT8 depthwise-separable CNN is single-digit milliseconds. Optimising the
architecture optimises the smallest term.

Two numbers get conflated constantly, so `dad bench` reports both:

| Quantity | Meaning |
|---|---|
| **cold-onset** | A silent drone starts emitting now. You wait for the window to fill with signal. Quote this one. |
| **steady-state** | The drone is already audible and the sliding window is full. Median extra wait is hop/2 plus compute. |

## Design decisions and why

**Frozen frontend contract** in `src/dad/config.py`. 16 kHz mono, 1 s window,
125 ms hop, 64 log-mel bands over 50 to 8000 Hz, 25 ms frame with 10 ms
stride. Every model is benchmarked against this identical frontend, so a model
comparison isolates the classifier head rather than measuring two different
feature pipelines.

**16 kHz, not 48 and not 8.** Blade-pass fundamentals and the useful harmonic
stack sit below 8 kHz, so 48 kHz is a threefold tax on FFT and mel cost for
nothing. Dropping to 8 kHz is the opposite mistake: it discards the 4 to 8 kHz
band holding electronic speed controller whine and the upper harmonics.

**Incremental log-mel on a ring buffer** in `src/dad/features.py`. Push one
hop of new audio and only genuinely new frames are computed. Per-decision
feature cost scales with hop rather than window, roughly an eightfold saving
at the defaults. `test_streaming_matches_full_window` asserts the incremental
path produces the same features as the naive one, because the latency numbers
are meaningless otherwise.

**No librosa.** A general-purpose mel implementation can cost more than the
network it feeds, and you cannot report an honest budget for code you did not
time. The frontend is plain numpy so every millisecond is attributable.

**Tier-0 harmonic comb gate** in `src/dad/gate.py`. A multirotor is a harmonic
comb at the blade-pass frequency, typically 100 to 300 Hz. It runs every hop
and is calibrated **recall-first**: the threshold is set at the 1st percentile
of positive scores, and the forwarded-background fraction is then *measured*,
not targeted. Calibrating on the negative distribution instead is the
intuitive move and it silently trades away detections you can never recover
downstream. The comb's argmax is the blade-pass frequency, so a passing gate
hands you an RPM estimate free.

Getting this detector right took three attempts, and the failures are
documented in the module docstring because they are more instructive than the
fix. It needs its own longer FFT, since 25 ms frames give 40 Hz bins that
cannot resolve the comb at all. It needs subharmonic disambiguation, or a
steady low-fundamental source such as a mower beats a wobbling drone and the
detector locks onto its harmonic. And it needs spectral whitening, or it
measures power rather than tonality and the 1/f background tilt makes every
low-frequency candidate win.

**INT8 depthwise-separable CNN** in `src/dad/models/dscnn.py`, the MLPerf Tiny
DS-CNN class. The decisive number: the same DS-CNN measures 181.92 ms
unoptimised and 0.39 ms optimised on the same microcontroller class. That
465-fold spread within one architecture dwarfs any gap between CNN, CRNN and
transformer. Your latency work is quantisation and kernel selection, not
architecture search.

**Classical baseline always reported** in `src/dad/models/baseline.py`. MFCC
summary statistics into a gradient-boosted tree. It is genuinely the fastest
classifier by one to two orders of magnitude and it is your honest floor. If
the CNN cannot beat it on a session-level split, the CNN is not earning its
milliseconds. Expect it to tie on clean data and lose off-distribution, where
published field tests show MFCC plus SVM false-alarming on lawn equipment and
helicopters.

**Rejected outright.** Audio transformers, on quadratic attention, data
appetite and a poor INT8 story. Useful only as a distillation teacher or an
offline accuracy ceiling. CRNNs, because recurrence serialises and buys little
over a CNN that already sees a full second. One real exception survives: a
stateful streaming GRU updating every 10 ms has constant per-frame cost and
frame-level output, which makes it the genuine lowest-latency option if you
need sub-100 ms reaction. Revisit it only if that is a hard requirement.

## Without the microcontroller constraint

If the detector runs on a laptop or a GPU workstation, INT8, kilobyte RAM
budgets and the CNN-only rule stop applying. Compute stays a rounding error
next to window fill, so bigger models buy accuracy, not speed. `dad compare`
trains every candidate on the same windows, split the same way by recording
session, and ranks them as a ladder where each rung must beat the one below:

| Rung | Model | What it tests |
|---|---|---|
| 1 | `comb` | No ML. The harmonic detector from `gate.py`, used as a classifier. The floor. |
| 2 | `lgbm` | MFCC summary statistics into LightGBM. Classical ML. |
| 3 | `dscnn` | The MCU-class network, now actually trained. |
| 4 | `resnet` | A wider residual CNN, about 0.7M parameters. More texture capacity. |
| 5 | `crnn` | CNN plus a bidirectional GRU. Tests whether the time evolution (rotor wobble, Doppler) carries information pooling throws away. |
| 6 | `transformer` | A small spectrogram transformer trained from scratch. Data-hungry; included to measure that. |
| 7 | `ensemble` | Mean probability of every ML model in the run. |

All neural models share one training recipe in `src/dad/train.py`: AdamW,
one-cycle schedule, SpecAugment, best validation epoch kept. The metric is the
false-alarm rate at 95% drone recall, per confuser kind, at each window
length. A model that matches a longer window's false-alarm rate at a shorter
window is the one way a model can reduce latency. Training uses CUDA when it
is available and falls back to CPU.

The README's earlier rejection of CRNNs and transformers was about streaming
on a microcontroller. On a laptop both are back in contention, and the
comparison decides.

## Window length is a detection-range knob

This is the part that is easy to get wrong. Shortening the window does not
mainly buy latency, it spends signal-to-noise ratio, and SNR is range.

Incoherent integration buys about 1.5 dB per doubling of window length.
Source SNR falls about 6 dB per doubling of range. So cutting 1 s to 125 ms
costs about 4.5 dB, which is 0.75 of a range doubling.

| Window | SNR delta | Usable range |
|---|---|---|
| 0.125 s | -4.5 dB | 0.59x |
| 0.25 s | -3.0 dB | 0.71x |
| 0.5 s | -1.5 dB | 0.84x |
| 1.0 s | reference | 1.00x |
| 2.0 s | +1.5 dB | 1.19x |

So range falls by about 40 percent, not 50. `range_penalty` in
`src/dad/bench.py` computes this and the test pins the arithmetic.

Frequency resolution is **not** the binding constraint. A 25 ms frame gives
40 Hz resolution against a comb spaced 100 to 300 Hz, which is ample.
Integration time against noise is the constraint. This matters because it kills
a tempting shortcut: short windows with N-of-M voting do not recover the lost
integration gain, since hard-decision fusion is strictly lossy compared with
integrating over the same duration. Voting also *adds* latency, which
`test_n_of_m_smoothing_costs_latency` demonstrates.

## What data does to inference time

For a fixed parametric graph, dataset size has **zero** effect on inference
time. It affects training only. State that once and move on.

Two qualifications. Non-parametric models break the rule, since kNN and kernel
SVM inference cost scales with neighbours or support vectors. And data
influences latency indirectly through input dimensionality, meaning sample
rate, mel count and frame count, and through quality, because better and
better-augmented data lets a smaller model generalise. Data genuinely buys
latency that way. Reported gains worth citing: AudioSet pretraining moves true
positive rate at 1 percent false positives from 86 to 92 percent, and
augmentation moves the hardest out-of-domain case from 16.6 to 66.2 percent.

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python and all dependencies are
managed for you.

```powershell
uv sync --group dev          # core pipeline, no torch
uv sync --group dev --extra dl   # adds torch for the DS-CNN
```

## Commands

```powershell
uv run dad info     # print the frozen frontend contract
uv run dad bench    # per-stage latency, p50/p95/p99. The headline figure.
uv run dad gate     # tier-0 gate calibration and the free RPM estimate
uv run dad sweep    # accuracy vs end-to-end delay table
uv run dad compare  # every model on the same windows (needs the dl extra for neural models)
uv run dad compare --models comb lgbm resnet --windows 0.5 1.0 --device cuda
uv run pytest       # 29 tests, including the streaming-equivalence invariant
```

In VS Code, open `drone-acoustic-detection.code-workspace`. Three debug
configurations are preset in `.vscode/launch.json`.

## Measured on this machine

A Windows laptop, single-threaded, so these are a development baseline and not
a target-hardware result. Re-run on the Pi before quoting anything.

| Stage | p50 | p95 | p99 |
|---|---|---|---|
| log-mel, full window | 0.434 | 0.665 | 0.754 |
| log-mel, incremental | 0.047 | 0.052 | 0.058 |
| tier-0 harmonic gate | 0.277 | 0.430 | 0.523 |
| MFCC summary stats | 0.075 | 0.093 | 0.115 |
| LightGBM on MFCC stats | 1.521 | 1.945 | 2.359 |
| DS-CNN fp32, 22.5k params | 1.680 | 2.166 | 2.482 |

Three results worth putting in the report.

**The incremental frontend is 9.2 times cheaper** than recomputing the window,
0.434 ms down to 0.047 ms per decision.

**Total compute is 0.20 percent of cold-onset latency**, 2.0 ms against
1002 ms. That is the council's central claim, reproduced locally.

**The DS-CNN and the gradient-boosted tree are the same speed here**, 1.68 ms
against 1.52 ms, despite one being a neural network and the other a tree
ensemble. Both are dominated by Python call overhead rather than arithmetic.
The classical model's supposed one-to-two orders of magnitude advantage does
not survive contact with a real pipeline on this host, which is the strongest
possible argument that architecture choice is not where your latency lives.

The gate, on synthetic data, holds 97.9 percent recall while rejecting
83.3 percent of background. The per-source breakdown is the useful part: wind
and pink noise are rejected outright, the mower confuser leaks 58 percent, and
the in-band "machine" confuser passes 100 percent. That last one is by design.
It is exactly the case a tier-0 gate cannot resolve on fundamental frequency
alone, and therefore exactly what the second tier exists for.

## The honest caveats

**The synthetic data is scaffolding, not a result.** `src/dad/synth.py`
generates drones, wind, pink noise and a deliberately drone-like mower
confuser. The generator and the models share assumptions, so any accuracy it
produces is fiction and the AUC column in `dad sweep` is meaningless. What
survives the synthetic stage is the latency measurement, which is
hardware-dependent rather than data-dependent, and the shape of the real
dataset. Swap in real audio before reporting anything.

**Session-level splits are mandatory.** `session_split` groups by recording
session, never by clip. Clips from one session share microphone, placement,
wind and background, so a clip-level split leaks and inflates accuracy, which
tricks you into thinking you need a bigger and slower model.

**Benchmark on the target, not on your laptop.** Latency on a development
machine says nothing about a Raspberry Pi or a Cortex-M. If the deployment
target is a laptop or a GPU, the entire latency question dissolves and even a
transformer finishes inside one window.

**torch dynamic quantisation is not embedded-grade.** It covers the Linear
head, not the convolutions. The real 465-fold win needs TFLite-Micro, CMSIS-NN
or microTVM with full static INT8. Say so rather than implying otherwise.

## Real datasets to plug in

- DREGON, drone audio with ego-noise, from INRIA
- DroneAudioDataset, by Al-Emadi and colleagues
- ESC-50 and AudioSet, for negatives and for pretraining
- Record your own 30 minutes of local background. It is the most valuable half
  hour you will spend, because your false alarms come from your own site.

## For the report

Three independent reviewers converged on the same deliverable, which is a
strong signal that it is the right one. Plot accuracy against end-to-end delay,
sweeping window across 0.25, 0.5, 1 and 2 seconds crossed with hop. Report
measured p95 per-stage latency on the actual target hardware alongside
multiply-accumulate counts, INT8 model size and real-time factor. That curve is
the contribution, not the model.
