---
title: Evaluation methods
parent: Evaluation
layout: home
nav_order: 1
---

# Evaluation methods
{: .no_toc }

<details open markdown="block">
  <summary>Contents</summary>
  {: .text-delta }
- TOC
{:toc}
</details>

## What is measured

The detector reads the audio in short windows that overlap. For every window it
outputs a **confidence** between 0 and 1 that a drone is present. With the
standard settings, a window is 1 second of audio, and a new window starts every
125 ms (the *hop*). Some later models use a 2 second window, and their results
say so.

A test round is a stretch of audio where it is known exactly when a drone is
present. From the confidence in each window, two numbers are read:

- **Confidence with a drone**: the mean confidence over the windows where a
  drone is present. This should be as high as possible.
- **Confidence without a drone**: the mean confidence over rounds where no
  drone is present. This should be as low as possible.

There are no fixed pass marks. Models are ranked against each other: first on
confidence with a drone, then on confidence without a drone when distractors are
present.

## Sound levels

All levels in the tests are real sound pressure levels in **dB SPL**, the
level a microphone would measure. In every test file, digital full scale
corresponds to 110 dB SPL, so a drone at 80 dB SPL is 30 dB below full scale.

Each sound is mixed in at its own level. This lets the tests control how loud
the drone is compared with its background, which decides whether it can be
heard at all.

## Test 1: Real drone test

**The most important test of the project.**

*Question:* with real drone recordings in a war soundscape, is the confidence
high with a drone and low without one, and does it stay low when distractors
are added?

- **Drones:** the 35 real drone recordings of the ITU ARIS dataset. Each
  recording is cut to its first 10 seconds.
- **Background:** a war soundscape at a fixed **90 dB SPL**.
- **Drone levels:** 100, 90, 80, 70 and 60 dB SPL. Against the war soundscape,
  this is a range from 10 dB louder to 30 dB quieter than the background.
- **Distractors**, in order of priority: wind, then cars, helicopters and
  insects. They are added on top of the war soundscape, and they are also
  played without any drone to check that they do not raise the confidence.
- **No-drone rounds:** the same backgrounds without a drone, so each drone round
  has a matching round to compare against.

## Test 2: Time before impact

*Question:* when a drone flies straight at the microphone, how high is the
confidence in the last seconds before it arrives?

A real drone recording is treated as a steady drone heard at 1 metre and is
"flown" in a straight line at the microphone. The last sample of the file is
the moment of impact. At each moment, the drone's distance follows from its
speed and the time left before impact. The level is lowered for that distance,
by spherical spreading and air absorption (ISO 9613-1). Doppler shift can be
turned on, but it is off by default.

- **Speeds:** chosen by the tester, between 50 and 200 km/h.
- **Result:** the mean confidence over the window **3 to 7 seconds before
  impact**, with every second counted equally. The curve of confidence over
  time is shown next to it.

For example, at 150 km/h a drone 7 seconds from impact is about 290 metres away.

## Test 3: Natural environment simulation

*Question:* how does the detector behave outside a virtual test?

The detector runs live as a demo in a surround-sound speaker array, with drone
and background sounds played around the microphone.

> **To be specified.** The details of this test are still being decided.
{: .text-grey-dk-000 }

## Keeping test data apart

A test result is only meaningful if the model has never seen the test audio, or
anything close to it. The project follows a few fixed rules:

Split by recording, never by clip
: Short clips cut from the same recording sound alike. If some of them are used
  for training and others for testing, the test is too easy. Data is therefore
  always divided by whole recording session or source.

Tune on validation, not on test
: Model choices, thresholds and when to stop training are decided on a
  validation set taken from the training sources. The test sources are only
  used to report a finished model.

ITU ARIS is for testing only
: The ITU ARIS recordings used in test 1 are never used for training, for
  background audio in training, or for tuning. This rule is also enforced in
  the code. The recordings were listened to during development, and results on
  them state that.

A development copy of test 1
: To tune without touching ITU ARIS, a copy of test 1 is built from other
  drone recordings (the Lund and Halmstad datasets), with the same war
  soundscape, levels and distractors. None of the models are trained on these
  recordings. Figures from the copy are development figures, not results.

## Reporting a result

Every reported number states the conditions it was measured under: which test
and sources, the background and levels, the window and hop, and how many
drone and no-drone windows it covers.

New models are always reported next to two simple **baselines**:

- **comb**: a hand-made detector that looks for the evenly spaced tones
  (harmonics) that a drone's propellers produce. It needs no training.
- **lgbm**: a gradient-boosted decision tree model (LightGBM) trained on
  summary statistics of MFCCs, computed from the same mel frontend as the
  neural networks.

If a neural network cannot beat these, its complexity is not paying off.
