import importlib.util
from pathlib import Path
from types import SimpleNamespace


def test_matrix_keeps_fixed_budgets_and_separates_local_comparison(tmp_path):
    path = Path(__file__).parents[1] / "scripts/run_biomaster_v3_experiments.py"
    spec = importlib.util.spec_from_file_location("v3_experiment_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = SimpleNamespace(out=tmp_path, cache_dir=tmp_path / "cache", seeds=[1, 2],
                           epochs=6, dense_test_drugs=64, skip_local=False)
    jobs = module.build_jobs(args)
    assert len(jobs) == 12
    assert len({j["out"] for j in jobs}) == 12
    for job in jobs:
        command = job["command"]
        assert command[command.index("--epochs") + 1] == "6"
        assert "--max-steps" not in command and "--max-entities" not in command
        assert ("--local" in command) == job["name"].startswith(("E_", "F_"))
    assert jobs[0]["name"] == "D_support_rank"
