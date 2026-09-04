"""Test de la captura de procedencia."""
from __future__ import annotations

import json

from src import config as config_mod
from src.evaluate import capture_provenance, run_registry


def test_run_captura_procedencia_aunque_falten_artefactos(tmp_path, monkeypatch):
    """Sin git/mlflow.db/raw.dvc en el directorio aislado, no debe fallar: cada
    pieza se degrada a None/{} en vez de lanzar una excepción."""
    monkeypatch.setattr(capture_provenance, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config_mod, "PROJECT_ROOT", tmp_path)
    # También el registro de runs: sin esto el test leía y escribía el
    # `canonical_runs.json` real del repositorio, contaminando un artefacto
    # entregable desde la propia suite de pruebas.
    monkeypatch.setattr(run_registry, "PROJECT_ROOT", tmp_path)
    config_mod.load_config.cache_clear()

    result = capture_provenance.run()

    assert "captured_at_utc" in result
    assert result["dvc_raw_hash"] is None
    # `canonical_runs` es lo que declara cada etapa al ejecutarse; sin etapas
    # ejecutadas está vacío. `mlflow_latest_runs_seen_in_backend` es solo la vista
    # del backend, que se conserva como contraste y no como fuente de verdad.
    assert result["canonical_runs"] == {}
    assert result["mlflow_latest_runs_seen_in_backend"] == {}
    out_path = tmp_path / "reports" / "provenance.json"
    assert out_path.exists()
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["captured_at_utc"] == result["captured_at_utc"]
