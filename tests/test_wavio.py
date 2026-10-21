"""WAV files through a detector, hop by hop, scored against a sidecar."""
from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf

from dad import wavio
from dad.cli import main
from dad.config import DEFAULT

SR_IN = 48_000


class RmsScorer:
    """Stands in for a model: a loud window scores high."""
    name = "rms"

    def score(self, audio: np.ndarray) -> np.ndarray:
        return np.sqrt(np.mean(np.asarray(audio, dtype=np.float64) ** 2, axis=1))


@pytest.fixture
def clip(tmp_path):
    """20 s, 48 kHz, 16-bit stereo: quiet noise, a loud tone in channel 0
    from 7 s to 12 s, and silence in channel 1."""
    rng = np.random.default_rng(0)
    t = np.arange(20 * SR_IN) / SR_IN
    left = 0.01 * rng.standard_normal(len(t))
    on = (t >= 7.0) & (t < 12.0)
    left[on] += 0.3 * np.sin(2 * np.pi * 300 * t[on])
    path = tmp_path / "approach.wav"
    sf.write(path, np.stack([left, np.zeros_like(left)], axis=1), SR_IN, subtype="PCM_16")
    path.with_suffix(".json").write_text(json.dumps(
        {"drones": [{"start_s": 7.0, "end_s": 12.0, "impact_s": 12.0}], "wind": "level 2"}))
    return path


def test_read_wav_resamples_to_16k_and_keeps_channel_0(clip):
    c = wavio.read_wav(clip)
    assert (c.sr_orig, c.channels_orig, c.subtype) == (SR_IN, 2, "PCM_16")
    assert c.audio.dtype == np.float32
    assert len(c.audio) == 20 * DEFAULT.sample_rate
    assert np.sqrt(np.mean(c.audio[8 * 16_000:9 * 16_000] ** 2)) > 0.1  # the tone, so channel 0


def test_one_decision_per_hop_stamped_when_its_window_ends(clip):
    d = wavio.stream(wavio.read_wav(clip), RmsScorer(), threshold=0.1)
    assert d.t_end[0] == pytest.approx(1.0)
    assert np.allclose(np.diff(d.t_end), DEFAULT.hop_seconds)
    assert len(d.t_end) == (20 * 16_000 - 16_000) // 2_000 + 1
    assert d.drone.dtype == bool


def test_short_clip_gives_no_decisions():
    c = wavio.Clip("short", np.zeros(8_000, np.float32), 16_000, 1, "PCM_16")
    assert len(wavio.stream(c, RmsScorer(), threshold=0.1).score) == 0


def test_windows_straddling_a_start_or_end_are_left_out():
    d = wavio.Decisions(np.array([7.0, 7.5, 8.0, 12.0, 12.5, 13.0]), np.zeros(6), 0.5)
    truth = wavio.Truth([{"start_s": 7.0, "end_s": 12.0}])
    assert wavio.window_labels(d, truth, 1.0).tolist() == [0, -1, 1, 1, -1, 0]


def test_scores_recall_false_positives_precision_and_lead(clip):
    r = wavio.evaluate_file(clip, RmsScorer(), threshold=0.1)
    q = wavio.rates(r["counts"], r["leads"])
    assert q["recall"] == 1.0 and q["false_positive_rate"] == 0.0 and q["precision"] == 1.0
    assert r["counts"]["left_out"] == 14          # 7 windows either side of each edge
    (lead,) = r["leads"]
    assert 7.5 <= lead["first_alarm_s"] < 8.0      # credited once half the window is drone
    assert lead["met"] and lead["lead_s"] == pytest.approx(12.0 - lead["first_alarm_s"])
    assert r["truth"].meta == {"wind": "level 2"}


def test_late_or_missing_alarm_fails_r5():
    d = wavio.Decisions(np.array([9.0, 10.0, 11.0]), np.array([0.0, 0.9, 0.9]), 0.5)
    late = wavio.lead_times(d, wavio.Truth([{"start_s": 8.0, "end_s": 12.0, "impact_s": 12.0}]), 1.0)
    assert late[0]["lead_s"] == pytest.approx(2.0) and not late[0]["met"]
    none = wavio.lead_times(d, wavio.Truth([{"start_s": 8.0, "end_s": 12.0, "impact_s": 9.5}]), 1.0)
    assert none[0]["first_alarm_s"] is None and not none[0]["met"]


def test_alarm_from_mostly_pre_drone_audio_does_not_count_for_r5():
    # an explosion fires the windows ending at 5.0 and 6.25; the drone appears at 6.0
    d = wavio.Decisions(np.array([5.0, 6.25, 9.0]), np.array([0.9, 0.9, 0.0]), 0.5)
    out = wavio.lead_times(d, wavio.Truth([{"start_s": 6.0, "end_s": 12.0, "impact_s": 12.0}]), 1.0)
    assert out[0]["first_alarm_s"] is None


@pytest.mark.parametrize("drone", [{"start_s": 5.0, "end_s": 4.0},
                                   {"start_s": 5.0, "end_s": 9.0, "impact_s": 4.0}])
def test_bad_sidecar_is_refused(tmp_path, drone):
    (tmp_path / "x.json").write_text(json.dumps({"drones": [drone]}))
    with pytest.raises(ValueError):
        wavio.read_truth(tmp_path / "x.wav")


def test_threshold_from_run(tmp_path):
    run = dict(args=dict(name="it5_pool_speed_s1", window_seconds=1.0), results=[
        dict(arm="hold out d3set", recipe="simdist", scorer="dscnn", integrate=1, deploy_threshold=0.41),
        dict(arm="hold out d3set", recipe="simdist", scorer="dscnn", integrate=8, deploy_threshold=0.30),
        dict(arm="hold out fpv:ariel_25.02", recipe="simdist", scorer="dscnn", integrate=1,
             deploy_threshold=0.62)])
    p = tmp_path / "run.json"
    p.write_text(json.dumps(run))
    assert wavio.threshold_from_run(p, "it5_pool_speed_s1_d3set_simdist_dscnn.pt", 1.0) == 0.41
    assert wavio.threshold_from_run(p, "it5_pool_speed_s1_fpv_ariel_25.02_simdist_dscnn.pt", 1.0) == 0.62
    for bad, window in (("other_run_d3set_simdist_dscnn.pt", 1.0), ("comb", 1.0),
                        ("it5_pool_speed_s1_d3set_simdist_dscnn.pt", 2.0)):  # 2.0: wrong window
        with pytest.raises(ValueError):
            wavio.threshold_from_run(p, bad, window)


def test_cli_end_to_end_with_the_comb(clip, tmp_path, capsys):
    out, summary = tmp_path / "hops.csv", tmp_path / "summary.json"
    assert main(["wav", str(clip.parent), "--threshold", "0.5", "--out", str(out),
                 "--summary", str(summary)]) == 0
    s = json.loads(summary.read_text())
    assert s["summary"]["files_with_truth"] == 1
    assert s["files"][0]["meta"] == {"wind": "level 2"}
    assert out.read_text().splitlines()[0] == "file,t_end_s,score,drone,truth"
    assert "R5 alarm" in capsys.readouterr().out


def test_a_saved_network_checkpoint_runs_over_a_wav(clip, tmp_path):
    pytest.importorskip("torch")
    from dad import realrun as rr
    from dad.models.zoo import build

    net = rr.NetScorer("dscnn", rr.VARIANTS["buried"], DEFAULT)
    net.mu = np.zeros((DEFAULT.n_mels, 1), np.float32)
    net.sigma = np.ones((DEFAULT.n_mels, 1), np.float32)
    net.model = build("dscnn", DEFAULT.n_mels).eval()
    ckpt = tmp_path / "untrained_dscnn.pt"
    net.save(ckpt)
    d = wavio.stream(wavio.read_wav(clip), wavio.load_scorer(str(ckpt)), threshold=0.5)
    assert len(d.score) == 153 and np.all((d.score >= 0) & (d.score <= 1))


def test_a_saved_lightgbm_model_runs_over_a_wav_and_finds_its_threshold(clip, tmp_path):
    from dad import realrun as rr

    rng = np.random.default_rng(0)
    m = rr.LgbmScorer(rr.VARIANTS["buried"], DEFAULT)
    m.model.fit(rng.standard_normal((200, 80)), np.r_[np.zeros(100), np.ones(100)])
    ckpt = tmp_path / "it9_d3set_simdist_lgbm.pkl"
    m.save(ckpt)
    audio = wavio.read_wav(clip).audio[:3 * 16_000].reshape(3, 16_000)
    loaded = wavio.load_scorer(str(ckpt))
    assert np.allclose(loaded.score(audio), m.score(audio))
    run = tmp_path / "it9.json"
    run.write_text(json.dumps(dict(args=dict(name="it9", window_seconds=1.0), results=[
        dict(arm="hold out d3set", recipe="simdist", scorer="lgbm", integrate=1, deploy_threshold=0.37)])))
    assert wavio.threshold_from_run(run, ckpt, 1.0) == 0.37
