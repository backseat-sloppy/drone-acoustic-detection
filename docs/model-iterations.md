---
title: Model iterations
parent: Model Training
layout: home
nav_order: 1
---

# Model iterations
{: .no_toc }

A condensed account of how the detector was developed: every training
iteration in one table, then the choices that shaped the work and what
each one taught us. The full lab notebook, with every command, commit and
number, is [`docs/REPORT.md`](https://github.com/backseat-sloppy/drone-acoustic-detection/blob/master/docs/REPORT.md)
in the repository. Entry numbers on this page (for example *entry 037*)
refer to it.

<details open markdown="block">
  <summary>Contents</summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

---

## How to read the numbers

- **Recall** is the share of drone windows the detector flags; **false
  alarms** is the share of no-drone windows it flags. Unless stated
  otherwise a window is 1 s of 16 kHz mono audio.
- **"At 10% own false alarms"** means the threshold is set so that 10% of
  the held-out recording setup's own no-drone audio is flagged, and recall is
  read at that point. It compares models fairly but assumes the site has been
  listened to first.
- **"At the deployment threshold"** means one threshold frozen at 5% false
  alarms on the validation data and then carried unchanged to unseen data.
  This is how a deployed detector would behave, and it is the honest reading
  of the requirements.
- **Held out / unseen rig** means a recording setup (site, microphones and
  drone) that the model never trained on. Splits are always by session or
  setup, never by clip.
- **The requirements changed during the project.** Up to iteration 15 the
  targets were **95% recall (R2) at no more than 5% false alarms (R3), at one
  threshold**, and an alarm at least 3 s before the drone arrives (R5). No
  iteration met R2 and R3 together on unseen data. On 2026-11-09 these were
  replaced by three tests judged in **confidence** (the network's score,
  0-100%): as high as possible with a drone, as low as possible without one,
  first against a war soundscape, then with distractors (wind, cars,
  helicopter, insects). There are no numeric targets. See
  [The requirements become three tests](#the-requirements-become-three-tests).
- **Confidence "pinned"** means a constant is added to each model's logit so
  that its confidence with a drone matches the reference model's. A gain that
  disappears when pinned is only the operating point moving.
- Numbers from synthetic audio or virtual test benches are diagnostics, not
  detection performance.
- **Results on d3set and Lund are not shown on this page.** Two of the
  datasets used in training and testing come with terms that do not yet
  allow their results to be published: d3set is GPL-3.0 and kept for
  internal use, and Lund has no licence stated. Their per-dataset figures
  are in the report. Where they carried a verdict, the page cites the
  figures on FPV, Nightjar, UAV16 or ITU ARIS (all CC BY) instead, or marks
  the verdict with **†** (decided on a withheld figure). Validation figures
  are pooled over the training rigs' validation sessions. The development
  copy of test 1 (iterations 21-23) uses Lund and Halmstad drones, so only
  its no-drone figures are shown.

---

## Iterations at a glance

| # | What was tried | Why | Best honest result | Verdict |
|---|---|---|---|---|
| Phase 0 | Select, download, inspect and convert four datasets (DDL, DroneNoise, Halmstad, FSD50K) | Supervised training needs real, labelled audio | DDL has no no-drone recordings, so drone label and recording chain are confounded | Data ready, confound found |
| 1 | Small CNN on DDL drones against FSD50K, with level removed, backgrounds mixed in and random EQ | The standard remedy for a confounded label | Validation AUC 0.99, but transfer AUC 0.85 and a score that *rises* with distance | **Failed.** The model learned the DDL site, not the drone |
| 2 | Train only on drones that are loud in their own recording, each buried in a background; judge with a same-background probe | Iteration 1's faint, distant drones taught the background | Transfer AUC 0.93-0.95; about 23% false alarms at 95% recall on an unseen chain | **Hypothesis confirmed, target not met.** Stopped: more data needed |
| Phase 1 | New priorities (recall first, about 150 m) and four new datasets (Nightjar, d3set, Lund, Glasgow), two with no-drone audio from the drone's own rig | Iteration 2's request for data | At 100-200 m the drone is at the edge of audibility | Data ready |
| 3 | Leave one rig out; weight positives toward 150 m with real far drones or simulated distance | Same-rig negatives allow far drones again | DJI Mini 2 (Nightjar) at 100-200 m: recall 0.04 at 10% false alarms, no better than chance | **Rejected.** Simulated distance helped a little; the audio holds too little drone at 150 m |
| 4 | Average the scores of consecutive windows; a 2 s window | Per-window evidence at 150 m is weak | Mini 2 at 100-200 m still at chance; 2 s window adds nothing | **Rejected by its rule**, small consistent gain |
| 5 | Pool every rig but the held-out one; speed perturbation 0.8-1.25 | Data diversity had never been tested; the scenario's drones are fast | Unseen FPV campaign 0.61 → 0.66 with speed perturbation; Mini 2 at 100-200 m still at chance | **Pooling + speed perturbation is the best recipe** |
| 6 | Iteration 5's recipe trained once on every rig ("model 6") | A final model for the real test | Validation recall 0.83 at 5% false alarms; 55% false alarms on mower audio | Checkpoint, not evidence |
| 7 | Edge Impulse (MFE + small CNN) on upload-licensed data | Try the vendor toolchain for embedded deployment | Recall 0 off its training rig | **Negative.** Model 6 stays |
| 8 | Edge Impulse on a pool of six datasets, one held out, level-normalised | Iteration 7 failed on a 50 dB level gap between rigs | In-distribution AUC 0.65 → 0.92; held-out result not published (licence) | **Negative.** Edge Impulse stopped |
| 9 | Frontend only: log-mel (MFE) vs MFCC vs linear spectrogram | Supervisor's feedback: finer frequency resolution for harmonics | FPV mean: MFE 0.669, MFCC 0.485, linear 0.692 | **MFE stays** |
| 10 | `it10-SynthDrone` | | | Not logged in the report yet |
| 11 | Steady synthetic noise as negatives and maskers (CRNN) | All networks alarmed on steady pink/brown noise | Pink/brown alarms to 0%, mower 38% → 13%, but ITU ARIS drones alarmed 71% → 37% of windows | **Failed its recall guards** |
| 12 | Real FSD50K confusers (engines, vehicles, insects, water) as hard negatives | Synthetic noise cost real-drone recall | Confuser false alarms 31% → 18%, ITU ARIS drones 71% → 42%; same separation (AUC 0.932 vs 0.930) | **Failed its recall guards.** Recipes move the threshold, not the separation |
| 13 | Frozen pretrained PANNs Cnn14 embedding + logistic regression | Training-data levers had run out | Mower false alarms 46%; separation AUC 0.84 vs CRNN 0.93 | **Failed** |
| 14 | PANNs Cnn14 fine-tuned end to end (LP-FT) | The frozen embedding underfit | Failed 2 of 3 gates, among them validation recall 0.59 (gate 0.77) | **Failed at go/no-go** (one untuned setup) |
| 15 | Decision stage only: causal median and averaging several networks; new "war coupling" readout | War spikes appeared to lift the drone score | War mostly *lowers* the drone score; averaging cuts the coupling by 17-21% (30% required) | **Rejected** |
| Req. | All requirements deleted and replaced by three tests judged in confidence (entries 148-150) | Decided by Mads and Nicolai | | No numeric targets from here on |
| 16 | it9 CNN and CRNN retrained four ways: 600k training windows, 2 s window, wider networks, 128 mel bands | The it9 training settings had never been varied | 2 s window: the only change that passed for both networks †. 600k windows: mower false alarms CRNN 41% → 17%, DS-CNN 24% → 15% | **2 s window wins**; more training cuts mower alarms |
| 17 | 2 s window and 600k windows together | The two levers of iteration 16 acted on different readouts | CRNN mower false alarms 41% → 28% with the gain of iteration 16 kept † | **Passes for the CRNN, not the DS-CNN** |
| 18 | "It9B": the it9 CRNN with drone windows weighted ×2 in the loss | Mads wanted more confidence with a drone, accepting worse separation | Drone windows +0.05 (validation); mower alone 0.44 → 0.60; matched-FA recall unchanged (0.789 vs 0.790) | **Holds.** Scores shift up, separation unchanged |
| Test 1 | The first run of test 1 on ITU ARIS drones, seven models | The new requirements' main test | War: CRNNs 66-85% confident with a drone, 0.1-7% without. Cars alone: 59-81% with no drone | **War holds, cars fail.** it9 CRNN ranked first |
| 19 | Pink and brown noise as 15% of no-drone training windows (`600k` arm) | Networks alarmed on pink/brown noise; iteration 11's broader noise had cost too much recall | Pink/brown alarms to 0%; mower CRNN 17% → 11%, DS-CNN 15% → 5%; validation unchanged | **Fails narrowly †**; `it19 CRNN colour s1` goes to the test GUIs |
| 20 | Bandpass (80 Hz-6 kHz) and Wiener gain in the frontend, CPU gates first | Mads asked; the frontend has no noise suppression | Wiener keeps only 66% of drone line bins at 0 dB (80% required). Bandpass: mower false alarms more than double | **Both fail.** Frontend unchanged |
| 21 (UAV16) | A fifth rig in the pool: UAV16 (DJI Flip, 16 channels, CC BY) with its 28 backgrounds | Same-array drone and no-drone audio is the known data gap | Recall on held UAV16 batches *falls*: DS-CNN 0.91 → 0.89, CRNN 0.86 → 0.83; unseen rig also falls † | **Rejected** |
| 22 (UAV16) | UAV16 with only its 4 plain-environment recordings as negatives | Its other backgrounds are fans, exhausts and other places | Held UAV16: CRNN +0.04, DS-CNN -0.01; unseen rig falls further † | **Rejected.** The loss comes with UAV16 in the pool at all |
| 21 (cars) | A development copy of test 1 (non-ARIS drones); then road vehicles from FSD50K as negatives and maskers | Test 1 fails on cars; ITU ARIS may not steer design | Cars alone (no drone): CRNN 64% → 36%, DS-CNN 62% → 42%, pinned. "With cars" only -6.9 / -6.6 points (10 required) | **Fails its box**, but the car gain is real |
| 22 (frontend) | PCEN in place of the log; probe of whether 8-11 kHz adds drone information | PCEN was the named next frontend; the microphone samples at 42 kHz | PCEN: wind alone with no drone 37-50% (was 1-3%). 8-11 kHz: +0.015 AUC on near drones, no gain at range † | **Both rejected.** 16 kHz log-mel stays |
| 23 | Ten Nightjar "no-drone" segments that Mads heard as drones relabelled | "No drone" in Nightjar means "no logged DJI flight" | Drone confidence up by less than the seed spread †; no-drone confidence equal or lower | **No effect, no harm.** Relabels kept |

The default model is still the **it9 CRNN** (entry 096): log-mel frontend,
iteration 5's pooled recipe with speed perturbation, trained on every rig.
It ranked first on test 1 (entry 182), although several later models are
less confident without a drone.

---

## The baseline pipeline

What stayed fixed through most of the work, so that iterations could be
compared:

- **Signal.** 16 kHz mono, 1 s windows. During training windows are drawn at
  random; at test time the hop is 0.5 s (iterations 3-5, 9) or 0.125 s
  (deployment and the GUIs).
- **Frontend (MFE).** 64 log-mel bands from 50 Hz to 8 kHz, 25 ms frames,
  10 ms stride. Each window is **level-normalised**: the log of its own mean
  mel power is subtracted, so loudness cannot be used as a cue, and values
  are floored 50 dB below that mean so digital silence is not a cue either.
- **Models.**
  - *DS-CNN*, a depthwise-separable CNN of 22,530 parameters (the "small
    CNN" of iterations 1-9 and model 6).
  - *CRNN*, 739,747 parameters: three convolution blocks that pool only along
    frequency, a bidirectional GRU and attention pooling over time. First
    trained in entry 096, the main network from iteration 11 on.
- **Training.** AdamW, learning rate 2e-3 on a one-cycle schedule, weight
  decay 1e-2, batch 64, SpecAugment, a fixed budget of 200,000 windows,
  half drone and half not, final weights kept (no best-epoch selection).
- **Baselines**, reported next to every model:
  - `comb`, a harmonic comb detector with no training, which looks for a
    stack of harmonics with a fundamental between 35 and 330 Hz;
  - `lgbm`, LightGBM on log-mel features.

---

## Phase 0: what the data really contained

The first decision was to read every archive before touching it, instead of
trusting the hosts' descriptions. That paid off immediately. The DDL
dataset, the main source of drone audio, was described as having a no-drone
class, but **no no-drone file exists**; it also held 2,629 byte-identical
files with different labels, and timestamps a year off from its name.

The consequence shaped everything that followed: every training drone came
from one recording chain (DDL's microphones and site) and every non-drone
from another (FSD50K's crowd-sourced clips). A model can separate those two
sets perfectly without ever hearing a drone. FSD50K's own train/validation
division also shares 774 uploaders, so it was re-split by uploader.

## Iterations 1-2: the recording-chain shortcut

**Iteration 1** applied the textbook remedy: remove level, mix drone windows
into real backgrounds, randomise the EQ. On validation it looked excellent
(AUC 0.99, 2-6% false alarms). It was not. Its score **rose** with distance
to the drone, being most confident beyond 150 m where no drone is audible.
Faint, distant DDL windows are mostly DDL background with a drone label, so
the model learned the background. Validation could not see this, because
validation shared the same confound.

Two lasting changes came out of this:

1. **A paired probe** (entry 014): the same background window is scored
   alone and with a drone added. Both sides share the recording chain, so the
   probe cannot be passed by recognising a site. Controls (an ordinary sound
   added, or a far DDL window added) must stay near AUC 0.5.
2. **Only audible drones as positives** (iteration 2): DDL within 50 m,
   each buried in a background down to 5 dB below it.

Iteration 2 confirmed the hypothesis: the model now found drones from
recording chains it never trained on (transfer AUC 0.93-0.95), and its
score fell with distance. But false alarms at 95% recall on an unseen
chain were about 23%, against a 5% target. Four sampling changes and two
models about 30 times larger did not help. The council and the experiments
agreed that **the limit was data, not model capacity**: about 15 minutes of
two drone models within 50 m from one afternoon at one site, plus about
3 minutes of four others. Work stopped until new data arrived.

## Phase 1 and iterations 3-4: the 150 m question

Mads set new priorities: recall first, then time to detection, at about
150 m. New datasets arrived, and crucially, two of them (Nightjar and d3set)
recorded **no-drone audio with the same rig** as the drone flights. That
removes the confound at its source and allows an honest test: train on one
rig, test on the other, and set false alarms on the test rig's own
background.

**Iteration 3** weighted training positives toward 150 m, either with real
far drones or by summing near drones into the rig's background at a lower
level (*simulated distance*, recipe `simdist`). Neither reached the target.
On Nightjar at 100-200 m, recall at 10% false alarms was 0.04 for the DJI
Mini 2: an alarm followed a moment with the drone at that range less often
than it followed a moment of background. A level check found the Mini 2
only about 1 dB above background at 50-100 m and below it further out. Real
far drones did not help; simulated distance helped slightly and never hurt
(on Nightjar it raised recall within 50 m by 8 points), so it became the
standard recipe.

**Iteration 4** traded time for evidence by averaging up to 8 consecutive
window scores (4.5 s). The gain was small and consistent but short of the
+10 points the rule asked for, and on Nightjar nothing helped at
100-200 m, because there was nothing to integrate. A 2 s window added
nothing. Thresholds also did not transfer between rigs: one set for 10%
false alarms on one rig gave very different false-alarm rates on another.

## Iteration 5: pooling and speed perturbation

Mads described the operational scenario: small drones at up to 50 m/s,
heading for a soldier, in heavy noise, with an alarm needed at least 3 s
ahead. The council recommended testing **data diversity**, which had never
been tried, since each earlier arm trained on one new rig.

Iteration 5 trained on every rig except one held-out unit (Nightjar, d3set,
or one of three FPV campaigns, held out by campaign, never by device, since
several phones recorded the same flights). Pooling alone did not help at
150 m. **Speed perturbation**, resampling every training window by a random
factor of 0.8-1.25 to imitate faster motors and Doppler shift, did: mean
recall on an unseen FPV campaign rose from 0.61 to 0.66 (three seeds). The
DJI Mini 2 at 100-200 m stayed at chance. The same resampling was applied to negatives, so an emptied top band
could not become a drone cue. Rescoring with two microphones (the planned
deployment array) moved recall by -7 to +7 points with no consistent
direction.

This recipe (pool, `simdist`, speed 0.8-1.25) is still the base of every
model trained since.

## Iteration 6: model 6 and the MoSCoW requirements

For the real field test, Mads set the MoSCoW requirements (R1-R5) and asked
for a final model. Model 6 is iteration 5's recipe trained on every rig, with
the threshold frozen at 5% false alarms on validation. With nothing held
out it produces checkpoints, not evidence. Its most important finding was a
weakness: it alarms on **55% of mower-only audio**. A loud motor without a
drone became the clearest failure mode.

## Iterations 7-8: Edge Impulse

Edge Impulse was tried as a second training path because it profiles and
exports models for embedded hardware. Trained on Nightjar alone it detected
nothing on other rigs. A 50 dB level difference between rigs explained part
of that, so iteration 8 normalised every clip to -30 dBFS and trained on a
pool of six datasets with one held out. In-distribution AUC rose from 0.65 to
0.92, but the model still did not work on the held-out rig (figures in the
report). Three of its six training datasets hold drone audio only, so "an
unfamiliar microphone chain" had probably again become evidence for
"drone", the confound from iteration 1. With
microcontroller deployment ruled out of scope, Edge Impulse was stopped.

## Iteration 9: which frontend?

The supervisor proposed synthetic training, a CRNN and a linear
spectrogram. The argument for the spectrogram is concrete: 25 ms frames give
40 Hz bins, too coarse to resolve a 100-300 Hz harmonic comb, while a 64 ms
frame gives 15.6 Hz bins. A synthetic training sampler (the Pure Data drone
with randomised pitch, Doppler and air absorption) was built, then set aside
by Mads in favour of testing frontends on real data.

Three frontends were compared with everything else fixed:

| Frontend | FPV mean recall (10% own FA) | Nightjar recall within 50 m (10% own FA) | Mower false alarms (deployment) |
|---|---|---|---|
| MFE (64 log-mel, 25 ms) | 0.669 | 0.389 | 35% |
| MFCC (20 coefficients) | 0.485 | 0.211 | 29% |
| Linear (509 bins, 64 ms) | 0.692 | 0.644 | 48% |

The linear spectrogram gained a lot close to the rig (+26 points within
50 m on Nightjar) and a little on FPV (+2.3, under the +5 the rule asked
for), and it alarmed 12 points more on the mower. Finer frequency
resolution helps where the drone is loud and **does not** help tell a drone
from another motor. **MFE stayed.** Afterwards a DS-CNN and the CRNN were
trained on every rig with this recipe (the "it9" models); the CRNN became
the reference for the rest of the work.

## Iterations 11-12: false alarms on steady noise

Virtual tests showed that every network alarmed on 99-100% of *war-shaped
noise* (noise with the long-term spectrum of a war soundtrack), yet on 0% of
the real war track it was made from. Pink and brown noise did the same. The
council concluded the failure was **steady noise with low-frequency
emphasis**, not war, and rejected training on war-shaped noise.

- **Iteration 11** added a family of steady noises as negatives and as
  maskers under drones. Pink and brown alarms fell to 0% and mower alarms
  from 38% to 13%. But it cost recall on real drones it never trained on:
  the ten ITU ARIS drone excerpts went from 71% to 37% of windows alarmed,
  breaking the iteration's recall guards. The network learned "steady means no
  drone", and a distant or hovering drone is steady.
- **Iteration 12** used real confusers from FSD50K instead (engines,
  vehicles, power tools, fans, insects, water). Confuser false alarms fell
  by 42%, but validation recall dropped 2.3 points (the guard allowed 2),
  and the ITU ARIS drones fell from 71% to 42%.

The key result came from a threshold-free comparison (entry 127): **at
matched false alarms, the recipes separate drones from noise equally well**
(AUC 0.930-0.932, none better than the it9 CRNN's 0.937). Harder negatives
only move where the validation threshold lands; they buy fewer false alarms
with recall in a fixed ratio, which a higher threshold would buy just as
well. The training-data levers had run out.

## Iterations 13-14: pretrained audio networks

With data recipes exhausted, the next lever was the representation. PANNs
Cnn14, pretrained on AudioSet, has heard thousands of hours of engines,
vehicles and aircraft.

- **Frozen embedding + logistic regression** (iteration 13): worse overall.
  Separation AUC 0.84 against the CRNN's 0.93, 46% mower false alarms, 100%
  on brown noise, and 12-13 points worse than the CRNN on every held-out FPV
  campaign. The linear probe reached only 0.81 accuracy on its own training
  data.
- **Fine-tuned end to end (LP-FT)** (iteration 14): failed two of its three
  go/no-go gates after one seed, among them validation recall 0.59 against
  0.77 required. Only one untuned
  setup was tried, so this rules out that setup, not fine-tuning in general.

## Iteration 15: the decision stage

Mads observed that in the war soundscape the CRNN's score shows spikes that
seem to lift the drone's score. Iteration 15 trained nothing and changed only
what happens after the network: a causal 1 s median and averaging several
networks. A new readout, the **war coupling** (score of drone + war minus
score of drone alone, hop by hop), measured the effect directly.

The war mostly **lowers** the drone's score instead: by more than 0.1 on
37-51% of drone hops, while it raises it on only 5-6%. A median leaves this
unchanged; averaging networks cuts it by 17-21%, short of the 30% asked. The
two levers left for masking are war-like impulsive sounds under the positives
in training, and a level normalisation that a single loud event cannot
dominate. The second is a frontend change and needs a council first.

## The requirements become three tests

On 2026-11-09 Mads and Nicolai deleted every requirement (R1-R5, the
priorities, the data rows) and replaced them with three tests, all judged in
**confidence** rather than recall at a threshold:

| Test | What it tests | What matters |
|---|---|---|
| 1. Real drone test | The 35 real ITU ARIS drones, mixed into a war soundscape at fixed levels, then with distractors (wind, cars) | Confidence with a drone as high as possible, without one as low as possible. The most important test |
| 2. Modified drone | A real drone recording flown at the microphone at 50-200 km/h | Mean confidence 3 to 7 s before impact |
| 3. Natural environment | The detector as a live demo in a surround speaker array | How it behaves outside a virtual test |

Models are ranked first on confidence with a drone in war, then on
confidence without a drone with distractors. These tests run in
`Complete_Test_Gui`, which mixes the drones and background layers at set
levels and scores every network on the same audio.

This changed what an iteration could be judged on. Iterations 16-17 were
designed and judged under the old rules while the change happened, so their
verdicts say which variant separates drones better on an unseen rig, not
how it does in the three tests (entry 159).

**ITU ARIS** is a sealed test source: it may only score finished models,
never train, tune or select them, and `src/dad/sealed.py` enforces this in
code. Test 1 is built from it, so ranking models on test 1 conflicts with
that rule. The team chose to rank on it anyway (entries 180, 182), and the report
states that from then on ITU ARIS results steered a choice. To avoid this,
iteration 21 built a **development copy of test 1** from drones that no model
trained on and that are not ITU ARIS.

## Iterations 16-18: training settings, and It9B

The it9 networks had never had their training settings varied: 200,000
windows at batch 64 is only about 3,100 optimiser steps for a 740,000-parameter
CRNN. **Iteration 16** retrained both networks four ways, one change each,
three seeds, against a re-run reference:

- a **2 s window** was the only change that passed the rule for both
  networks †, at the cost of 1 s more audio before the first possible alarm;
- **600,000 training windows** did not pass the rule but cut mower false
  alarms clearly (CRNN 41% → 17%, DS-CNN 24% → 15%) at unchanged validation
  recall;
- **wider networks** (up to 2.95 M parameters) changed nothing beyond seed
  noise;
- **128 mel bands** on a 50 ms frame were worse, much worse on the mower for
  the CRNN.

**Iteration 17** combined the 2 s window with 600,000 windows. For the CRNN
the two levers added up: the best CRNN so far on both readouts, with mower
false alarms at 28% against the reference's 41% †. For the DS-CNN the
longer training gave back part of the 2 s gain.

**Iteration 18 (It9B)** answered a different wish: Mads wanted a model that
is more confident when a drone is present, accepting worse separation.
Moving the threshold does not change the confidence the test GUIs plot, so
the drone class was weighted twice in the loss instead. As predicted, every
score rose (validation drone windows +0.05, the mower alone from 0.44 to
0.60, above the GUIs' 50% line) while recall at matched false alarms stayed
the same (0.789 against 0.790). It9B is a shifted it9 CRNN, not a better one.

## Test 1: war holds, cars fail

The first full run of test 1 scored seven models on ten steady ITU ARIS
drone excerpts over 1,331 background combinations. Against the war
soundscape alone, the CRNNs kept 66-85% confidence with a drone down to about
-14 dB measured SNR, and 0.1-7% without one. Wind alone gave 0.2-5%. **Cars
did not**: a car at 50 km/h, far quieter than the drone, made every model
about as confident as a drone does (it9 CRNN 59-81% with no drone at all).
Car rejection became the main open failure.

Ranked strictly as the requirements state, the it9 CRNN came first (81.4% with
a drone in war) and stayed the default, although it was **last** on
confidence without a drone with distractors (33.9%, against 22.2% for the
best `600k` CRNN). The top three were within 1.1 points on the first
criterion, from ten excerpts, with no uncertainty computed.

## Iterations 19-20: pink and brown noise, and filters

**Iteration 19** took the narrowest part of iteration 11's steady-noise
recipe: pink and brown noise as 15% of no-drone windows, nothing under the
drones. Pink and brown alarms went to 0% (the DS-CNN had alarmed on 74% of
pink), mower false alarms fell (CRNN 17% → 11%, DS-CNN 15% → 5%) and
validation held. It failed narrowly on its unseen-rig recall rule †, by less
than the seed spread, and cost far less than iteration 11 did. One seed,
`it19 CRNN colour s1`, went into the test GUIs.

**Iteration 20** tested two filters Mads asked for, each behind a CPU gate
fixed before any training (a council in entry 172 rejected Wiener estimates
taken over time, because they would remove a drone's steady harmonic lines):

- The **Wiener gain** (noise estimated as a median across frequency) failed
  its gate: at 0 dB SNR it kept only 66% of the drone's harmonic line bins
  (80% required), 34% at -10 dB, and lowered the background by only 2.3 dB
  (6 dB required). No Wiener model was trained.
- The **bandpass** (80 Hz-6 kHz) was trained, three seeds, and failed: mower
  false alarms more than doubled (DS-CNN 15% → 40%, CRNN 17% → 36%), brown
  noise alarms rose, and the unseen-rig gain stayed under its bar †.

What the bandpass mostly removes is 6-8 kHz. The networks appear to use that
band to tell a mower or broadband noise from a drone, while air absorption
removes it from a distant drone anyway. That reading is untested.

## Iterations 21-22, UAV16 line: a fifth rig

Two iterations were numbered 21 and 22 on parallel branches. On one, at
Marius's request, a newly found dataset, **UAV16** (a 16-channel array
recording a DJI Flip hovering at 5-70 m, with 28 field backgrounds, CC BY
4.0), was added to the training pool of iteration 19, with two of its
acquisition batches held out.

It did not help. The colour model, which had never heard this rig, already
found 86-91% of the held DJI Flip windows. With UAV16 in the pool, recall
on those held batches *fell* (DS-CNN 0.91 → 0.89, CRNN 0.86 → 0.83), and so
did recall on the unseen rig †. A look at the backgrounds suggested a cause:
most were fans, exhaust pipes, a starting car and other places, not the
drone sites' ambience. Keeping only the four plain-environment recordings as
negatives (iteration 22) rejected that cause: the unseen-rig loss grew †. It
comes with having UAV16 in the pool at all.

A useful side result: the colour model alarms on 83-95% of UAV16's fan,
exhaust and starting-car recordings, and on 30-40% of its traffic, indoor and
wind recordings. That is the same false-alarm weakness test 1 found with cars.

## Iteration 21, cars line: car rejection on a development test

The other iteration 21 attacked test 1's car failure. First it built a
**development copy of test 1** (`scripts/dev_test.py`) with drones no model
trained on and that are not ITU ARIS, split into a tune half and a confirm
half, and with two readouts a constant cannot fake: the pinned comparison and
an AUC of drone-in-war against no-drone hops. The copy reproduced the car
failure: the it9 CRNN gave 76.6% confidence on cars alone, as in test 1.

Then FSD50K road-vehicle and engine clips were added as 25% of negatives and
under 30% of drones. On **cars alone**, confidence fell from 64% to 36%
(CRNN) and from 62% to 42% (DS-CNN), pinned, on engines the arm never
trained on, on every seed. But the box's deciding figure, no-drone
confidence *with* cars, fell only 6.9 and 6.6 points pinned (10 required),
because two of its three backgrounds are a loud war that already hides the
cars. For the CRNN it also cost separation in war † and raised the war-alone
and helicopter confidence. **The box fails**; the car gain is real but not
free.

## Iteration 22, frontend line: PCEN and the band above 8 kHz

PCEN (per-channel energy normalisation, which divides each mel band by its
own recent level) had been the named next frontend since iteration 9. It
passed a CPU gate that checked the drone survives it, then failed on the
development test:

- **Wind alone, with no drone**, gave 37% (CRNN) and 50% (DS-CNN) confidence
  against 1-3% before. Wind's fluctuating broadband energy looks to PCEN like
  the band-wise modulation a drone gives.
- Cars alone fell from 64% to 47% for the CRNN, close to what iteration 21's
  vehicle data bought, without new data. The helicopter rose 6-7 points.
- Separation in war fell for both networks, past the pre-set fail line †.

The **8-11 kHz probe** asked whether a 22.05 kHz frontend would help, since
the deployed microphone samples at 42 kHz. On near drones (Nightjar and FPV,
CRNN, grouped folds) the extra band added +0.015 AUC (0.783 → 0.798). At
range it added nothing † and could not have shown much, since the probe
networks detect almost nothing beyond 100 m. Air absorption also makes
8-11 kHz the first band to fade. **22.05 kHz was not adopted**, and with PCEN
rejected too, the frozen 16 kHz log-mel frontend stays.

## Iteration 23: relabelled Nightjar segments

A data audit found that Nightjar's "no drone" label means only "no logged
DJI flight". Mads listened to a ranked list and heard drones in ten 10 s
segments. They were relabelled as drone, and three borderline segments were
left out of training.
One CRNN was retrained per recipe, one seed each. Confidence with a drone
rose in every condition but by less than the seed spread †, and confidence
without a drone stayed equal or fell. **No measurable effect, no harm**: the
corrected labels are kept because they are more correct. 100 s is small
against about 75 minutes of Nightjar no-drone audio.

---

## Recurring lessons

- **Validation that shares the confound proves nothing.** Iteration 1 scored
  AUC 0.99 while learning the wrong thing. Every later judgement uses unseen
  rigs, same-rig negatives or a same-background probe.
- **The limit has mostly been data, not model size.** Larger CNNs
  (iteration 2), pretrained networks (iterations 13-14) and new frontends
  (iteration 9) never beat a small network trained on more diverse data.
  The two changes that clearly helped, pooling with speed perturbation and
  same-rig negatives, are both about the data.
- **Faint positives are label noise.** A drone label on audio without an
  audible drone teaches the background (iterations 1 and 3).
- **Separate the operating point from the separation.** Read at one frozen
  threshold, hard-negative recipes look like a trade-off; at matched false
  alarms they are the same model (entry 127).
- **Thresholds do not transfer between rigs.** A threshold frozen on
  validation lets through 5% there and far more elsewhere; on Nightjar's
  no-drone audio every learned model alarmed on 76-86% of windows at the
  deployment threshold (iteration 9). This is the largest gap between
  development figures and deployment.
- **Confidence can be bought without separation.** Weighting the drone class
  (It9B), harder negatives (iterations 11-12, 21) and relabelling
  (iteration 23) all moved scores up or down together. Pinned comparisons and
  AUCs are what tell a real gain from a shifted operating point.
- **The frontend has held.** MFCC, a linear spectrogram, 128 mel bands, a
  bandpass, a Wiener gain, PCEN and a 22.05 kHz band were all tried, and
  none beat 64 log-mel bands at 16 kHz. Apart from MFCC, which lost drone
  recall outright, they failed on noise (mower, wind, steady noise) rather
  than on the drone.
- **Cars and other engines are the main open failure.** The networks hear an
  engine's harmonic comb as a drone. Real vehicle negatives help on cars
  alone; no change yet fixes them under every background.

## How decisions were made

- Each iteration changes **one thing** from the one before, states a
  hypothesis and fixes the pass rule before any run.
- Big decisions (a new model family, frontend, evaluation protocol or
  abandoning a line of work) go to a three-member **council**: a Devil's
  Advocate, a Scholar who sees no project context and states best practice,
  and a Problem Solver. The council advises; the decision and any dissent are
  logged.
- Thresholds, model selection and early decisions use validation carved from
  the training rigs; held-out test sources are scored only to close an
  iteration, and every such scoring is logged. Where a test result did steer
  a choice (ITU ARIS in entry 180), the report says so.
- From iteration 21, candidates are judged on a development copy of test 1
  built from non-ITU ARIS drones, so the real test 1 stays a test.
- Negative results are recorded with the same care as positive ones.

## Where it stands

Under the old requirements, no model met R2 (95% recall) and R3 (5% false
alarms) at one threshold on unseen data. In the last leave-one-out
comparison (iteration 9) the DS-CNN with the MFE frontend found 0.67 of the
drone windows on an unseen FPV campaign at 10% of that campaign's own false
alarms.

Under the current requirements, judged in confidence:

- **Drone against the war soundscape (first priority): holds.** On test 1
  the CRNNs stay at 66-85% confidence with a drone and 0.1-7% without, down
  to about -14 dB measured SNR.
- **Distractors (second priority): wind holds, cars fail.** A car alone,
  with no drone, still gives 59-81% confidence on the default model.
  Vehicle negatives (iteration 21) and PCEN (iteration 22) lowered the
  CRNN's confidence on cars alone from 64% to 36% and to 47%, but each cost
  something elsewhere.
- **The default model** is the it9 CRNN, ranked first on test 1 by its
  confidence with a drone in war (81.4%). The best `600k` CRNN is almost as
  confident with a drone (80.7%) and clearly less confident without one with
  distractors (22.2% against 33.9%). It and the later `colour` CRNNs are
  available in the test GUIs.
- **Open levers named in the report:** vehicle negatives without the masker
  under drones, war-like impulsive sounds under the positives, PCEN with a
  longer context or alongside log-mel, and more listening to clean up
  Nightjar's labels.
