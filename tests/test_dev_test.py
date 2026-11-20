"""dev_test's statistics (report entries 243-244): the AUC and the detection rate against brute force, the
bootstrap's weights, and judge leaving out scores made on other audio."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("dev_test", Path(__file__).parents[1] / "scripts" / "dev_test.py")
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)


def design() -> list[dict]:
    """A small copy of dev_test's rounds: every condition readout reads, two stretch strata, rig rounds."""
    rs = [dict(kind="none", stretch=s, bg="+".join(bg)) for s in range(6) for bg in D.NO_DRONE_BGS]
    rs += [dict(kind="none", stretch=s, bg="war") for s in range(6, 9)]
    rs += [dict(kind="drone", file=f"{src} {j}", stretch=s, bg="+".join(bg), drone_db=db)
           for src in ("lund", "halmstad") for j in range(3) for s in (j, j + 3) for bg in D.WAR_BGS
           for db in (100.0, 60.0)]
    rs += [dict(kind="drone", file=f"recorded take{t}_{c}", stretch=s, bg="+".join(bg), drone_db=db, unit=f"recorded take{t}")
           for t in range(2) for c in range(2) for s in (t, t + 3) for bg in D.WAR_BGS for db in (100.0, 60.0)]
    rs += [dict(kind="helicopter", file=f"halmstad h{j}", stretch=j, bg=b) for j in range(2) for b in ("alone", "war")]
    rs += [dict(kind="rig", file=f"halmstad b{j}", stretch=s, bg="war", drone_db=60.0) for j in range(2) for s in (j, j + 3)]
    return rs


def grid(pos, neg):
    u, inv = np.unique(np.concatenate([pos, neg]), return_inverse=True)
    return inv[: len(pos)], inv[len(pos):], len(u)


def test_auc_and_detection_rate_against_brute_force():
    rng = np.random.default_rng(1)
    pos, neg = rng.integers(0, 20, 300) / 20, rng.integers(0, 20, 200) / 20 - 0.1     # many ties
    brute = np.mean([(p > q) + 0.5 * (p == q) for p in pos for q in neg])
    assert abs(D._auc(pos, neg) - brute) < 1e-12
    wp, wn = rng.integers(0, 3, 300), rng.integers(0, 3, 200)                          # weights = repeats
    w = D._wauc(*grid(pos, neg), wp.astype(float), wn.astype(float))
    assert abs(w - D._auc(np.repeat(pos, wp), np.repeat(neg, wn))) < 1e-12
    pos, neg = rng.normal(1, 1, 500), rng.normal(0, 1, 400)
    thr = min(t for t in np.concatenate([pos, neg]) if np.mean(neg >= t) <= D.FA)
    det = D._wdet(*grid(pos, neg), np.ones(500), np.ones(400))
    assert abs(det - np.mean(pos >= thr)) < 1e-12


def test_bootstrap_weights_resample_files_and_stretches():
    rs = design()
    w = D.boot_weights(rs, 200, seed=3)
    assert w.shape == (201, len(rs)) and (w[0] == 1).all()
    extra = [i for i, r in enumerate(rs) if not D.core(r)]
    assert (w[1:, extra].sum(1) == len(extra)).all()           # the extra stretches among themselves
    war = [i for i, r in enumerate(rs) if r["kind"] == "none" and r["bg"] == "war" and D.core(r)]
    assert (w[1:, war].sum(1) == 6).all()                      # the six stretches of entry 207 among themselves
    alone = [i for i, r in enumerate(rs) if r["bg"] == "alone"]
    assert (w[1:, alone] <= 2).all()                           # no stretch: a helicopter's own count only
    d = [i for i, r in enumerate(rs) if r["kind"] == "drone"]
    assert (w[1:, d] == 0).any() and (w[1:, d] > 1).any()      # files drawn out and drawn twice
    take = [i for i, r in enumerate(rs) if r.get("unit") == "recorded take0" and r["stretch"] == 0]
    assert len(take) == 16 and len({tuple(w[:, i]) for i in take}) == 1   # a recording's clips are drawn together


def test_judge_leaves_out_scores_made_on_other_audio(tmp_path, monkeypatch):
    rs, rng = design(), np.random.default_rng(0)
    monkeypatch.setattr(D, "OUT", tmp_path)
    out = tmp_path / "t"
    out.mkdir()
    (out / "rounds.json").write_text(json.dumps(dict(half="tune", audio_md5="abc", rounds=rs)), encoding="utf-8")
    for m, md5 in (("good", "abc"), ("stale", "xyz"), ("old", None)):
        np.savez_compressed(out / f"{m}.npz", *[rng.random(9) * 0.5 + 0.4 * (r["kind"] == "drone") for r in rs])
        if md5:
            (out / f"{m}.json").write_text(json.dumps(dict(audio_md5=md5)), encoding="utf-8")
    D.judge(argparse.Namespace(tag="t", ref="", boot=20, vs=["good"]))
    j = json.loads((out / "judge.json").read_text(encoding="utf-8"))
    assert list(j["unpinned"]) == ["good"] and list(j["boot"]["auc war"]) == ["good"]
    o = j["unpinned"]["good"]
    assert o["auc war"] > 0.9 and "auc rig" in j["boot"]
    assert abs(j["boot"]["gap war"]["good"]["est"] - (o["drone war"] - o["none war"])) < 1e-9 and o["gap war"] > 30
    assert abs(j["boot"]["gap war recorded"]["good"]["est"] - o["gap war recorded"]) < 1e-9 and "auc war recorded" in o


def test_run_extend_scores_only_the_new_rounds(tmp_path, monkeypatch):
    rng, calls = np.random.default_rng(0), []
    rs = [(dict(kind="none", stretch=i, bg="war"), rng.standard_normal(9).astype(np.float32)) for i in range(5)]

    def fake_stream(clip, sc, cfg, thr):
        calls.append(clip)
        return argparse.Namespace(score=clip.audio[:1].copy())

    monkeypatch.setattr(D, "OUT", tmp_path)
    monkeypatch.setattr(D, "stream", fake_stream)
    monkeypatch.setattr(D.E, "load_detectors", lambda ms, dev: ({m: 0.5 for m in ms}, {m: argparse.Namespace(cfg=None) for m in ms}))

    def go(tag, rounds, models=("m",), extend=""):
        monkeypatch.setattr(D, "rounds", lambda half: rounds)
        D.run(argparse.Namespace(tag=tag, half="tune", models=list(models), device="cpu", extend=extend))

    go("a", rs[:3])
    calls.clear()
    go("b", rs, extend="a")
    assert len(calls) == 2                                        # the two new rounds only
    z = np.load(tmp_path / "b" / "m.npz")
    assert [z[f"arr_{i}"][0] for i in range(5)] == [x[0] for _, x in rs] and D.scored_on(tmp_path / "b", "m", D.audio_md5(rs))
    for other, models in ((rs[1:], ("m",)), (rs, ("m", "unscored"))):   # another prefix; a model "a" lacks
        with pytest.raises(SystemExit):
            go("c", other, models, extend="a")
