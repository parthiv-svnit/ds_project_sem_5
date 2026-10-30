"""build_deliverables must never write into presentation/ from the fixture and must refuse invalid runs."""
import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    return subprocess.run([sys.executable, str(ROOT / "src" / "build_deliverables.py"), *args], cwd=ROOT, capture_output=True, text=True)


def test_refuses_fixture_into_presentation():
    r = run("--preview-fixture", "--out-dir", "presentation")
    assert r.returncode != 0 and "must not write" in (r.stdout + r.stderr)


def test_fixture_preview_is_stamped(tmp_path):
    r = run("--preview-fixture", "--out-dir", str(tmp_path))
    assert r.returncode == 0, r.stderr
    assert "FIXTURE PREVIEW" in (tmp_path / "speaker_script.txt").read_text()
    assert (tmp_path / "project_presentation.pptx").exists()
