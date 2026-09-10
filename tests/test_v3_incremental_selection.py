from types import SimpleNamespace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from scripts.train_biomaster_v3_incremental_ablation import Runtime


def fixture():
    return SimpleNamespace(protocol={'old_validation_ap_tolerance':.01,'measured_validation_ap_tolerance':.01},
        base_validation={'d2t':{'ap':.5,'measured_ap':.95},'t2d':{'ap':.5,'measured_ap':.94}})


def test_forward_gain_cannot_hide_reverse_regression():
    candidate={'d2t':{'ap':.8,'measured_ap':.96},'t2d':{'ap':.48,'measured_ap':.95}}
    assert not Runtime.eligible(fixture(),candidate)


def test_retrieval_gain_cannot_hide_measured_activity_regression():
    candidate={'d2t':{'ap':.6,'measured_ap':.93},'t2d':{'ap':.6,'measured_ap':.94}}
    assert not Runtime.eligible(fixture(),candidate)


def test_unchanged_backbone_is_always_eligible():
    rt=fixture()
    assert Runtime.eligible(rt,rt.base_validation)
