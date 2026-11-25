"""A plan split into runs: the parts cover every round once, and the GUI plays them one after the other."""
import queue
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
L = pytest.importorskip("live_test_engine")


@pytest.mark.parametrize("n", [1, 2, 3, 5])
def test_parts_cover_the_plan(n, monkeypatch, tmp_path):
    rounds = [dict(round=k + 1, drone="d") for k in range(7)]
    monkeypatch.setattr(L, "LiveTest", lambda p, o: SimpleNamespace(rounds=list(rounds)))
    got, skips = [], []
    for i in range(1, n + 1):
        plan = SimpleNamespace(live=dict(source="virtual", trigger="automatic", part=(i, n)), layers={}, length_s=20.0)
        run = L.LiveRun(plan, queue.Queue())
        skips.append(run.skip_s)
        assert run.skip_s == 20.0 * len(got)
        got += [r["round"] for r in run.order]
    assert got == list(range(1, 8)) and skips[0] == 0


def _page(monkeypatch, n, ends):
    G = pytest.importorskip("Complete_Test_Gui")
    import threading
    runs, shown = [], []

    class FakeRun:
        def __init__(self, p, ev, **k):
            runs.append(p.live.get("part"))
            self.test = SimpleNamespace(out=Path(f"run{len(runs)}"), results={1: 1})
            self.part_names, self.order, self.stop_flag = [], [], threading.Event()
            self.end = ends[len(runs) - 1]

        def start(self):
            pass

    monkeypatch.setitem(sys.modules, "reaper_live", SimpleNamespace(ReaperRun=FakeRun))
    monkeypatch.setattr(G, "Results", lambda app, t: shown.append(t.out.name))
    w = SimpleNamespace(state=lambda *_: None, configure=lambda **k: None)
    pg = SimpleNamespace(b_start=w, b_stop=w, b_results=w, b_physical=w, l_status=w, pb=w, app=None, run=None, test=None,
                         _reset_view=lambda *a: None, _lock=lambda on: None, _show_play=None, _poll=None, error=None)
    plan = SimpleNamespace(live=dict(source="reaper"), detectors=[])
    pg.batch = dict(plan=plan, dev=(None, 0, "8,9", 0, 0), n=n, i=0, tests=[], notes=[], stopped=False)
    for m in ("_start_part", "_batch_over", "_after_run"):
        setattr(pg, m, getattr(G.LivePage, m).__get__(pg))
    todo = []
    pg.after = lambda ms, f: todo.append(f) if ms == 5000 else None
    pg._start_part()
    for _ in range(20):
        if pg.run.end == "error":
            pg.error, pg.test.results = "boom", {}
        if pg.run.end == "stop":
            pg.run.stop_flag.set()
        pg._after_run()
        if not todo:
            break
        todo.pop()()
    return runs, shown, pg


def test_runs_follow_each_other(monkeypatch):
    runs, shown, _ = _page(monkeypatch, 3, ["ok"] * 3)
    assert runs == [[1, 3], [2, 3], [3, 3]] and shown == ["run1", "run2", "run3"]


def test_an_error_goes_on_and_stop_ends(monkeypatch):
    runs, shown, pg = _page(monkeypatch, 3, ["ok", "error", "ok"])
    assert len(runs) == 3 and shown == ["run1", "run3"] and pg.batch["notes"] == ["run 2: boom"]
    runs, shown, _ = _page(monkeypatch, 3, ["ok", "stop", "ok"])
    assert len(runs) == 2 and shown == ["run1", "run2"]
