"""rangerun.make_pool_arm_143 (iteration 11.2, entry 143): FPV never trains, the held rig is wholly out
of training and validation, the mower session is never trained on, validation is the other rig's."""
import pytest

from dad import chains as ch
from dad import rangerun as rg
from dad.realdata import Recording


def _rec(source, session, subset="", label=1, rid=None):
    return Recording(source=source, recording_id=rid or f"{source}-{session}-{subset}-{label}", label=label,
                     kind="drone" if label else "background", group=session, subset=subset, n_samples=160_000,
                     labels="", attributes={"session": session})


@pytest.fixture()
def manifest():
    nj_val = next(iter(rg.NIGHTJAR_VAL))
    mower = next(iter(rg.CONFUSER))
    return [
        _rec("nightjar", "nightjar-flight-1"), _rec("nightjar", "nightjar-flight-1", label=0),
        _rec("nightjar", nj_val), _rec("nightjar", nj_val, label=0),
        _rec("nightjar", mower), _rec("nightjar", mower, label=0),
        _rec("d3set", "d3-a", subset="Train"), _rec("d3set", "d3-a", subset="Train", label=0),
        _rec("d3set", "d3-v", subset=rg.D3SET_VAL), _rec("d3set", "d3-v", subset=rg.D3SET_VAL, label=0),
        _rec("fpv", "ariel_25.02"), _rec("fpv", "palmahim_10.02", label=0),
        _rec("ddl", "ddl-1"), _rec("dronenoise", "dn-1"), _rec("halmstad", "h-1"),
    ]


@pytest.mark.parametrize("held", ["nightjar", "d3set"])
def test_rule143_arm(manifest, held):
    arm = rg.make_pool_arm_143(manifest, held)
    every = arm.train + arm.val + arm.test
    assert not [r for r in every if r.source in ("fpv", "dronenoise", "halmstad", "ddl")]
    assert not [r for r in arm.train + arm.val if r.source == held]               # held rig: no train, no val
    assert arm.test and all(r.source == held for r in arm.test)
    other = "d3set" if held == "nightjar" else "nightjar"
    assert arm.val and all(r.source == other for r in arm.val)                   # threshold from the other rig
    assert not [r for r in arm.train + arm.val + arm.test if ch.session(r) in rg.CONFUSER]
    assert arm.confuser and all(ch.session(r) in rg.CONFUSER for r in arm.confuser)
    assert arm.unseen == {}


def test_rule143_refuses_other_units(manifest):
    with pytest.raises(ValueError):
        rg.make_pool_arm_143(manifest, "fpv:ariel_25.02")
    with pytest.raises(ValueError):
        rg.make_pool_arm_143(manifest, "none")
