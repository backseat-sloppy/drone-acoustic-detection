# Supervisor notes

Notes from the project supervisor, passed on by Mads. Each note is kept as
given (lightly cleaned), followed by how it reads against the repo and what
it would take. Nothing here is started yet; each item becomes work only when
Mads says so. Report entry 049 records that these notes were received.

## 2026-10-26

| # | Note (as given) | Reading | Depends on | Conflicts |
|---|---|---|---|---|
| S1 | When humans listen to audio data, the sounds must be masked to test it correctly: how loud is the drone compared with car sounds, dogs barking, helicopters etc. | Test at controlled drone-to-distracter level ratios, and take auditory masking into account: a drone is hidden by a louder sound in the same frequency band at the same time, not by the overall level. The stress test (entries 044 to 048) already sets each sound's level in dB SPL, but only for gunfire, explosion, wind, birds and car. Dogs and helicopters are not in it yet. FSD50K has `Dog`/`Bark`, and Halmstad has helicopters. | - | - |
| S2 | Pre-processing will make the model perform better; we are trying to mimic the human brain's sound processing. | A front end modelled on human hearing: an auditory filterbank (gammatone or ERB bands) in place of or beside log-mel, loudness compression, and perhaps masking-aware features. Today the front end is 64 log-mel bands (REQUIREMENTS: FREE). | Council (CLAUDE.md section 4: replacing the feature front end) | - |
| S3 | Look for mono audio recordings (ambients and drones); normalize and denoise them (background, distracters, drones and wind). | Build clean stems by class: drone, background, distracter, wind. Mono, level-normalised, denoised (for example spectral gating). The existing sets (DDL, d3set, Nightjar, FPV, FSD50K) are the first place to look. | - | Denoising drone recordings can also remove drone content. It must be checked by ear and by spectrum, never only by eye. |
| S4 | You can skip the drone synthesiser if you have clean drone data (without environment noise). | Prefer real clean drone recordings over the Pd synth for building test scenes. DDL and the FPV set hold close, loud drones that may be clean enough. | S3 | - |
| S5 | "Capstugrams" | Most likely cepstrograms: cepstral features over time (MFCC, or the real cepstrum, which shows harmonic spacing such as a drone's blade-pass fundamental as one peak). To confirm with Mads. | S2 | - |
| S6 | Try to train the model to match the synthesised sounds to the clean drone data. | Reading A: tune the synth (Pd patch) until its output matches real clean drones (a calibration of the synth, judged by features). Reading B: domain adaptation, so the model treats synthetic and real drones alike. | S3, S7 | Reading B means training on synthetic audio, which REQUIREMENTS.md has LOCKED ("Only real recordings, no synthetic training data", Mads 2026-10-10). Needs Mads's decision before any training on synthesised audio. |
| S7 | Transcribe the Pure Data patch to Python. | Port `DroneSound.pd` (4 detuned sawtooth motors with wandering pitch, prop-wash noise through a band-pass, 1500 Hz low-pass, 30 Hz high-pass) to numpy. Removes the Pd dependency and makes it fast and parameterisable. The Pd render is the reference to match. | - | - |
| S8 | IMPORTANT: talk to Cumher to get access to the supercomputer. | Only Mads can do this. | Mads | - |
| S9 | Do an impulse response of the microphone and feed it to the model, so it can be filtered out in the test itself. | Measure the test microphone's impulse/frequency response (sine sweep in a quiet room, or against a reference mic). Then either convolve the training data with it (so the model learns the test chain) or equalise the test audio by its inverse. This addresses the recording-chain confound from iteration 1. | Mads: the microphone, a speaker, a quiet room | - |
| S10 | Build synthetic background sounds; start by describing them so you can classify them. | First a taxonomy of the background/distracter classes (for example: wind, rain, vehicle, aircraft/helicopter, gunfire, explosion, voices, animals, machinery/generator), each with its acoustic signature. Then a generator per class. | S1 | If used for training, the same LOCKED synthetic-data rule as S6. Fine for test scenes and the stress test. |
| S11 | PLAY CALL OF DUTY. | Read as a pointer: battlefield games are a reference for how a war soundscape is layered (distant artillery, small arms, vehicles, shouting) and at what relative levels. A listening reference for S1 and S10, not a data source: game audio is copyrighted. | - | Not usable as data. |

### Decisions needed from Mads

1. S6 and S10: may synthesised audio be used for training (amends the LOCKED rule), or only for testing?
2. S5: is "capstugrams" cepstrograms?
3. S8: access to the supercomputer (Cumher).
4. S9: which microphone the test uses, and when it can be measured.
5. Order of work. Suggested: S7 (Pd to Python), then S3/S4 (clean stems), then S1/S10 (masking-aware scenes with more distracter classes), then S2/S5 (auditory front end, via the council).

## Scope decision, 2026-10-26

Mads: "You should not train the model, we are only creating the environment."
So the notes are read as environment work only (report entry 054):

- **S2, S5:** a hearing-like or cepstral analysis is used to describe and
  measure scenes, not as a new model front end.
- **S6:** reading A only (tune the synth to sound like clean real drones).
  Reading B (training) is out of scope.
- **S9:** the microphone's impulse response is applied to the scenes, to
  make them sound as the test microphone hears them, not fed to training.
- **S10:** synthetic backgrounds are for test scenes only.
- **S8:** the supercomputer matters less without training.
