"""The REAPER live test must score every round of an automatic run, also when the rounds play back to back."""
import queue
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
reaper_live = pytest.importorskip("reaper_live")


def test_back_to_back_rounds_are_all_scored(monkeypatch, tmp_path):
    SR, clock = reaper_live.SR, [0.0]
    monkeypatch.setattr(reaper_live, "time", SimpleNamespace(time=lambda: clock[0], sleep=lambda s: clock.__setitem__(0, clock[0] + s)))
    (tmp_path / "reaper").mkdir()
    run = reaper_live.ReaperRun.__new__(reaper_live.ReaperRun)
    run.test = SimpleNamespace(out=tmp_path, rebuild=lambda r: dict(round=r["round"]))
    run.heard = SimpleNamespace(end=0)
    run.slots = [dict(r=dict(round=k + 1), P=3.0 + 2 * k, n=2 * SR, ref=k) for k in range(6)]
    run.end_P, run.next_i, run.current, run.placed, run.manual = 15.5, 0, None, [], False
    run.stop_flag, run.lock, run.events = threading.Event(), threading.Lock(), queue.Queue()
    run.rp = SimpleNamespace(transport=lambda: (1, clock[0]))
    run._listen = lambda inq: setattr(run.heard, "end", int(clock[0] * SR))
    run._announce = run._levels = lambda *a: None
    run._round_job = lambda p: (p["m"]["round"], p["ref"])
    jobs = queue.Queue()
    run._loop(None, jobs)
    assert list(jobs.queue) == [(k + 1, k) for k in range(6)]
    assert not any("m" in p or "ref" in p for p in run.placed + run.slots)    # nothing is kept after scoring
