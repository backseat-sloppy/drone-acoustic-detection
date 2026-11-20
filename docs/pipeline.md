---
title: Processing Pipeline
layout: home
nav_order: 3
---

# Processing pipeline
{: .no_toc }

<details open markdown="block">
  <summary>Contents</summary>
  {: .text-delta }
- TOC
{:toc}
</details>

## Overview

The detector listens to one microphone and, eight times a second, answers the
question *is there a drone?* with a **confidence** between 0 and 1. The same
steps run whether the audio comes from a WAV file, a test round built by the
test GUIs, or a live microphone:

<style>
.pl { margin: 1.5rem 0 2rem; max-width: 46rem; }
.pl-src { display: inline-block; padding: .35rem .9rem; border-radius: 999px;
  border: 1px dashed #8c8a96; color: #c5c2cc; font-size: .85rem; }
.pl-group { position: relative; border-radius: 10px; padding: 1.4rem .9rem .9rem;
  margin-top: 1.1rem; border: 1px solid; }
.pl-group-label { position: absolute; top: -.7rem; left: .9rem; padding: 0 .5rem;
  font-size: .7rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase;
  background: #27262b; }
.pl-front { border-color: #2c84fa66; background: #2c84fa0d; }
.pl-front .pl-group-label { color: #6aa9ff; }
.pl-model { border-color: #9f7aea66; background: #9f7aea0d; }
.pl-model .pl-group-label { color: #b794f4; }
.pl-out { border-color: #38b2ac66; background: #38b2ac0d; }
.pl-out .pl-group-label { color: #4fd1c5; }
.pl-step { display: grid; grid-template-columns: 2rem 1fr auto; gap: .2rem .8rem;
  align-items: center; padding: .65rem .8rem; border-radius: 8px;
  background: #302d36; border: 1px solid #44434d; text-decoration: none !important;
  color: inherit !important; transition: border-color .15s, transform .15s; }
.pl-step:hover { border-color: #8c8a96; transform: translateX(2px); }
.pl-num { grid-row: span 2; width: 2rem; height: 2rem; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-weight: 700; font-size: .9rem; color: #fff; }
.pl-front .pl-num { background: #2c84fa; }
.pl-model .pl-num { background: #9f7aea; }
.pl-out .pl-num { background: #319795; }
.pl-title { font-weight: 600; color: #f5f6fa; }
.pl-desc { grid-column: 2 / 4; font-size: .82rem; color: #b8b5bf; line-height: 1.4; }
.pl-tag { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: .72rem; padding: .15rem .5rem; border-radius: 4px;
  background: #1f1e24; color: #d6d3dc; border: 1px solid #44434d; white-space: nowrap; }
.pl-arrow { height: 1.1rem; margin-left: 1.75rem; position: relative; }
.pl-arrow::before { content: ""; position: absolute; left: 0; top: 0; bottom: 4px;
  border-left: 2px solid #5c5a66; }
.pl-arrow::after { content: ""; position: absolute; left: -4px; bottom: 0;
  border: 5px solid transparent; border-top: 6px solid #5c5a66; border-bottom: 0; }
.pl-arrow.pl-first { margin-left: 2.65rem; }
@media (max-width: 40rem) {
  .pl-step { grid-template-columns: 2rem 1fr; }
  .pl-tag { grid-column: 2; justify-self: start; }
  .pl-desc { grid-column: 2; }
}
</style>

<div class="pl" role="img" aria-label="Pipeline: audio in, sliding window, log-mel frontend, normalisation, neural network, confidence, decision">
  <span class="pl-src">🎤 microphone or WAV file</span>
  <div class="pl-arrow pl-first"></div>

  <div class="pl-group pl-front">
    <span class="pl-group-label">Frontend · fixed for every model</span>
    <a class="pl-step" href="#1-audio-in">
      <span class="pl-num">1</span><span class="pl-title">Audio in</span><span class="pl-tag">16 kHz mono</span>
      <span class="pl-desc">First channel only, resampled to 16 kHz</span>
    </a>
    <div class="pl-arrow"></div>
    <a class="pl-step" href="#2-sliding-window">
      <span class="pl-num">2</span><span class="pl-title">Sliding window</span><span class="pl-tag">1 s every 125 ms</span>
      <span class="pl-desc">The last second of audio, eight times per second</span>
    </a>
    <div class="pl-arrow"></div>
    <a class="pl-step" href="#3-log-mel-frontend">
      <span class="pl-num">3</span><span class="pl-title">Log-mel frontend</span><span class="pl-tag">64 × 100</span>
      <span class="pl-desc">25 ms frames every 10 ms → 64 mel bands, 50 Hz to 8 kHz → log</span>
    </a>
    <div class="pl-arrow"></div>
    <a class="pl-step" href="#4-normalisation">
      <span class="pl-num">4</span><span class="pl-title">Normalisation</span><span class="pl-tag">64 × 100</span>
      <span class="pl-desc">Remove the window's own level, then standardise each band</span>
    </a>
  </div>
  <div class="pl-arrow pl-first"></div>

  <div class="pl-group pl-model">
    <span class="pl-group-label">Model · swappable</span>
    <a class="pl-step" href="#5-neural-network">
      <span class="pl-num">5</span><span class="pl-title">Neural network</span><span class="pl-tag">2 outputs</span>
      <span class="pl-desc">CRNN: convolutions → bidirectional GRU → attention pooling</span>
    </a>
    <div class="pl-arrow"></div>
    <a class="pl-step" href="#6-confidence">
      <span class="pl-num">6</span><span class="pl-title">Confidence</span><span class="pl-tag">0 – 1</span>
      <span class="pl-desc">Softmax probability of "drone"</span>
    </a>
  </div>
  <div class="pl-arrow pl-first"></div>

  <div class="pl-group pl-out">
    <span class="pl-group-label">Output</span>
    <a class="pl-step" href="#7-decision">
      <span class="pl-num">7</span><span class="pl-title">Decision</span><span class="pl-tag">drone / no drone</span>
      <span class="pl-desc">Drone if confidence ≥ the threshold frozen on validation</span>
    </a>
  </div>
</div>

Steps 1 to 4 are the *frontend*. They are fixed for the whole project (see
`src/dad/config.py`), so every model is fed exactly the same input and can be
compared fairly. Only the model in step 5, and its threshold, change between
models.

## 1. Audio in

The detector works on mono audio at **16 kHz**.

- A file of any sample rate, bit depth and channel count is read, and only its
  first channel is kept.
- If the sample rate is not 16 kHz, the audio is resampled with a polyphase
  filter.
- On a live microphone the same is done block by block as the audio arrives.

**Why 16 kHz.** The sound of a drone's propellers lies mainly below 8 kHz: the
blade-pass tone (typically 100 to 300 Hz) and its stack of harmonics, plus the
whine of the motor electronics at a few kHz. A 16 kHz rate keeps everything up
to 8 kHz. A higher rate costs about three times the computation for little
gain; iteration 22 tested whether the band from 8 to 11 kHz adds information
and did not adopt it. A lower rate (8 kHz) would throw away the upper
harmonics and the motor whine.

**Levels.** All audio in the project uses the same convention: digital full
scale corresponds to 110 dB SPL. The networks were trained on audio at that
scale, so an uncalibrated microphone gives indications rather than
measurements. The live tool has a gain setting to shift the input level.

## 2. Sliding window

The detector does not look at a single moment but at the **last 1 second** of
audio. A new window starts every **125 ms** (the *hop*), so windows overlap by
seven eighths, and a new confidence is produced eight times per second.

- The window length decides how much evidence the detector has. Longer windows
  hear fainter drones, but take longer to fill. Some later models use a 2 s
  window, and their results say so.
- The hop decides how quickly the detector reacts. It is independent of the
  window length.
- Each confidence is stamped with the time its window *ends*: the earliest
  moment it could exist in a live system. The first confidence therefore comes
  one window length after the audio starts.

## 3. Log-mel frontend

Each window is turned into a time-frequency picture, a **log-mel
spectrogram**, which is what the network looks at.

| Step | Setting |
|---|---|
| Frames | 25 ms (400 samples), Hann window, a new frame every 10 ms (160 samples) |
| Spectrum | 400-point FFT, power per frequency bin (40 Hz per bin) |
| Mel filterbank | 64 triangular filters spaced on the mel scale from 50 Hz to 8 kHz, each normalised by its width so narrow low bands are not quieter than wide high ones |
| Compression | natural logarithm of each band's energy |
| Result | 64 bands × 100 frames per 1 s window |

The mel scale gives fine resolution at low frequencies, where the blade-pass
tone and its first harmonics are, and coarser resolution higher up. The
logarithm compresses the large dynamic range of real sound, as the ear does.

**Streaming.** On live audio, only the frames that are new since the last hop
are computed (12 or 13 frames), and the window is reassembled from frames kept
in a ring buffer. The cost per decision therefore scales with the hop, not with
the window: about eight times less work than recomputing the whole window.

The frontend is written in plain NumPy, without an audio library, so that every
millisecond of it can be timed (`uv run dad bench`).

## 4. Normalisation

Two normalisations happen before the network sees the spectrogram.

**Level normalisation, per window.** The mean energy across all bands and
frames of the window is subtracted (in the log domain). This is the same as
turning the volume of every window to a fixed level, so the network cannot use
*how loud* the audio is, only *what it sounds like*. Values more than 50 dB
below the window's mean are clipped to a floor, so that digital silence does
not become a cue of its own.

This step exists because of iteration 1: without it, the network learned the
loudness and character of the recording equipment rather than the drone.

**Per-band standardisation.** Each of the 64 bands is then shifted and scaled
by a mean and standard deviation measured on the training data and stored with
the model. This puts all bands on a similar scale for the network.

## 5. Neural network

The default model is the **it9 CRNN** (convolutional recurrent neural network,
about 740,000 parameters), defined in `src/dad/models/zoo.py`. It takes the
64 × 100 spectrogram and works in three stages:

1. **Convolutions: local texture.** Three blocks of 3 × 3 convolutions (32, 64
   and 128 channels), each with batch normalisation and ReLU. After each block
   the frequency axis is halved by max-pooling, but the time axis is kept, so
   every one of the 100 frames survives. The convolutions learn local patterns
   such as harmonic lines and their spacing.
2. **Bidirectional GRU: evolution over time.** The 100 frames are read in order,
   forwards and backwards, by a recurrent layer (96 units each way). This lets
   the model use how the sound changes over the second, such as the slight
   wobble of a propeller's speed, which separates a drone from steady machines.
3. **Attention pooling: which moments matter.** A learned weight per frame
   decides how much each moment counts. A drone that is clear in only part of
   the window is not averaged away by the rest of it.

A final linear layer gives two outputs, *no drone* and *drone*.

Other networks can be put in the same place without changing anything else,
because they all take the same input and give the same two outputs. The most
used alternative is the **DS-CNN** (depthwise-separable CNN, about 22,500
parameters, `src/dad/models/dscnn.py`), which has no recurrent stage.

## 6. Confidence

The two outputs are turned into probabilities with a softmax, and the
probability of *drone* is the **confidence**: a number between 0 and 1 for
each window. This is the number the tests are judged on (see
[Evaluation](evaluation.html)).

## 7. Decision

Where a yes or no answer is needed, as in the live demo, the confidence is
compared with a **threshold**: at or above it, the answer is *drone*. Each
model's threshold was fixed once, on validation data from its training
sources, never on the test data. For the it9 CRNN it is 0.483, the level that
gave 5 % false alarms on validation.

No smoothing over time is applied by default: each decision rests on its own
window. Iteration 15 tested a running median and averaging several networks,
and neither was adopted.

## Training-only steps

Some steps happen only while a model is trained, never at inference. They make
the training data more varied, so the model generalises to recordings it has
not heard:

- mixing drone recordings with real background sounds at random levels,
  including "simulated distance", where a near drone is mixed in at a lower
  level,
- small random changes of playback speed (0.8 to 1.25 times),
- random spectral tilt and shape (equalisation),
- masking random bands and frames of the spectrogram (SpecAugment).

The training procedure is described under [Model Training](model-training.html).

## Baselines

Every result is reported next to two simple detectors. They share steps 1 and
2, and give a score per window in the same way:

- **comb** needs no training. It runs its own, longer FFT (2048 points) to
  resolve the blade-pass tone, searches for an evenly spaced stack of
  harmonics with a fundamental between 100 and 300 Hz, and allows the upper
  harmonics to smear, as a drone's do. A steady machine, such as a lawn mower,
  is less likely to pass, because its fundamental is checked separately.
- **lgbm** uses the same log-mel frontend and level normalisation (steps 3
  and 4), summarises each window as the mean and spread of 20 MFCCs and their
  changes over time, and classifies that summary with a gradient-boosted tree
  model (LightGBM).

## Frontend changes tested and not adopted

The frontend above has been challenged several times. Each change was tested
against it and none was kept:

| Iteration | Change | Outcome |
|---|---|---|
| 9 | MFCC or a linear spectrogram in place of log-mel | Log-mel was as good or better overall |
| 20 | Bandpass filter (80 Hz to 6 kHz) and Wiener noise suppression | Wiener removed too much of the drone; the bandpass raised false alarms on a mower |
| 22 | PCEN (per-channel energy normalisation) in place of the logarithm | Lower separation in war, and wind alone gave high confidence |
| 22 | 22.05 kHz sample rate, bands up to 11 kHz | No consistent gain on distant drones |

The details are in [Model iterations](model-iterations.html).

## Where it lives in the code

| Step | Code |
|---|---|
| Frontend settings (rate, window, hop, bands) | `src/dad/config.py` |
| Reading files, sliding window, scoring a file | `src/dad/wavio.py` |
| Live microphone, block by block | `src/dad/live.py` |
| Log-mel, streaming ring buffer | `src/dad/features.py` |
| Level normalisation, scorers (network, comb, LightGBM) | `src/dad/realrun.py` |
| Networks | `src/dad/models/zoo.py`, `src/dad/models/dscnn.py` |
| Comb baseline | `src/dad/gate.py` |

To run the it9 CRNN over WAV files, with its frozen threshold:

```powershell
uv run dad wav FILES --detector artifacts/models/final_crnn_none_simdist_crnn.pt --threshold-from artifacts/runs/final_crnn.json
```
