"""Test de la captura de procedencia."""
from __future__ import annotations

import json

from src import config as config_mod
from src.evaluate import capture_provenance


def test_run_captura_procedencia_aunque_falten_artefactos(tmp_path, monkeypatch):
    """Sin git/mlflow.db/raw.dvc en el directorio aislado, no debe fallar: cada
    pieza se degrada a None/{} en vez de lanzar una excepción."""
    monkeypatch.setattr(capture_provenance, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config_mod, "PROJECT_ROOT", tmp_path)
    config_mod.load_config.cache_clear()

    result = capture_provenance.run()

    assert "captured_at_utc" in result
    assert result["dvc_raw_hash"] is None
    assert result["mlflow_latest_runs"] == {}
    out_path = tmp_path / "reports" / "provenance.json"
    assert out_path.exists()
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["captured_at_utc"] == result["captured_at_utc"]
