import json
import shutil
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")
REAL = Path(__file__).resolve().parents[1] / "artifacts" / "champion"


def run(art_dir):
    import os
    os.environ["ARTIFACT_DIR"] = str(art_dir)
    try:
        return AppTest.from_file(APP, default_timeout=60).run()
    finally:
        del os.environ["ARTIFACT_DIR"]


def page_text(at):
    parts = [e.value for e in at.markdown] + [e.value for e in at.metric] + [e.value for e in at.caption]
    parts += [e.value for e in at.success] + [e.value for e in at.warning] + [e.value for e in at.info]
    return "\n".join(map(str, parts))


def test_app_renders_three_sections_from_real_artifacts():
    at = run(REAL)
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Forecast", "Model performance", "MLOps monitoring"]
    champ = json.loads((REAL / "champion.json").read_text())
    assert any(champ["version"][:8] in m.value for m in at.metric)  # active model version is shown
    assert "7 observations" in page_text(at)  # small-sample warning


def test_displayed_values_come_from_artifacts_not_hardcoded(tmp_path):
    shutil.copytree(REAL, tmp_path / "a")
    art = tmp_path / "a"
    champ = json.loads((art / "champion.json").read_text())
    champ["version"] = "TESTVERSION123"
    champ["forecast"] = 123.4
    champ["all_validation_metrics"]["naive"]["RMSE"] = 98.76
    (art / "champion.json").write_text(json.dumps(champ))
    mon = json.loads((art / "monitoring.json").read_text())
    mon["drift"]["columns"]["pm2_5"]["p_value"] = 0.4321
    mon["drift"]["drifted"] = True
    (art / "monitoring.json").write_text(json.dumps(mon))
    at = run(art)
    assert not at.exception
    text = page_text(at)
    assert "TESTVERS" in text and "123.4" in text and "98.76" in text
    assert "0.432" in text and "Drift detected" in text


def test_missing_artifacts_show_error_not_crash(tmp_path):
    at = run(tmp_path)
    assert not at.exception
    assert any("Run `python -m src.train`" in e.value for e in at.error)
