---
title: Purpose of the Project
layout: home
nav_order: 2
---

# Purpose of the Project
{: .no_toc }

<details open markdown="block">
  <summary>Contents</summary>
  {: .text-delta }
- TOC
{:toc}
</details>

## Motivation

> **Draft.** This section is to be written from the interviews in the
> [appendix](interview-transcriptions.html): who needs to know that a drone is
> coming, in what situations, and why sound is a useful signal for it.
{: .text-grey-dk-000 }

Small drones are cheap, fast and hard to see. A microphone on the ground can
hear a drone before it is in view, works in the dark and in fog, and does not
need a line of sight. This project asks how well a machine learning model can
turn that sound into a reliable warning.

## Final problem statement

The project builds a detector that listens to a single ground-based microphone
and answers one question: **is there a drone, or not?** Finding where the drone
is (localisation) is not part of the goal.

The detector outputs a *confidence* between 0 and 1. Its performance is judged
on that confidence, under specific and methodical conditions:

1. **With a drone**, the confidence must be as high as possible.
2. **Without a drone**, the confidence must be as low as possible.
3. First priority: both hold for a drone against a **war soundscape**.
4. After that: both hold with **distractors** as well, such as wind, cars,
   helicopters and insects.

There are no numeric targets. "High confidence" is deliberately left undefined:
models are compared with each other under the same conditions, not against a
fixed pass mark.

## The three tests

The same model is used in all three tests. They are described in detail under
[Evaluation methods](evaluation-methods.html).

| Test | What it asks |
|---|---|
| **1. Real drone test** | With real drone recordings mixed into a war soundscape at controlled levels: is confidence high with a drone and low without one, also when distractors are added? This is the most important test of the project. |
| **2. Time before impact** | When a drone flies straight at the microphone, how high is the confidence 3 to 7 seconds before it arrives? |
| **3. Natural environment simulation** | How does the detector behave as a live demo in a surround-sound speaker array, outside a purely virtual test? |

## Scope

In scope
: Detecting the presence of a drone from audio, in real time, with one
  microphone.

Out of scope
: Localising or tracking the drone, identifying the drone model, and running
  the detector on a microcontroller.

## Ranking models

When two models are compared, the order of importance is:

1. Confidence when a drone is present.
2. Confidence when no drone is present, with distractors.

The backgrounds are ranked the same way: the war soundscape is the main
condition, wind is secondary, and cars, helicopters and insects come after that.
