from pathlib import Path
import yaml


def test_workflows_parse():
    wf_dir = Path(".github/workflows")
    files = list(wf_dir.glob("*.yml"))
    assert {f.name for f in files} >= {"tests.yml", "daily.yml", "weekly.yml"}
    for f in files:
        data = yaml.safe_load(f.read_text(encoding="utf-8"))
        assert "jobs" in data
