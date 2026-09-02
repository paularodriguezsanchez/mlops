"""Test de integración end-to-end: ejecuta las orquestaciones `run` reales
de ingesta, anotación, dataset, entrenamiento y monitorización sobre un
subconjunto minúsculo, con todas las salidas redirigidas a un directorio
temporal (nunca toca `data/`, `models/`, `reports/` ni `docs/` del proyecto real).
"""
from __future__ import annotations

import yaml

from src import config as config_mod
from src.annotate import annotate
from src.features import build_dataset
from src.ingest import download
from src.monitor import drift_report
from src.train import train


def _tiny_config(tmp_path) -> dict:
    cfg = yaml.safe_load(config_mod.CONFIG_PATH.read_text(encoding="utf-8"))
    cfg["data"]["raw_dir"] = "data/raw"
    cfg["data"]["interim_dir"] = "data/interim"
    cfg["data"]["processed_dir"] = "data/processed"
    # Fixture de test explícitamente sintética y aislada del valor de producción
    # (config.yaml real usa multi_source, ver ADR 005 revisado 2026-07-30).
    cfg["data"]["annotation_source"] = "synthetic"
    cfg["synthetic"]["n_variants_train"] = 300
    cfg["synthetic"]["n_new_in_test"] = 60
    (tmp_path / "config").mkdir()
    cfg_path = tmp_path / "config" / "config.yaml"
    cfg_path.write_text(yaml.dump(cfg), encoding="utf-8")
    return cfg_path


def test_pipeline_completo_end_to_end(tmp_path, monkeypatch):
    cfg_path = _tiny_config(tmp_path)

    monkeypatch.setattr(config_mod, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config_mod, "CONFIG_PATH", cfg_path)
    config_mod.load_config.cache_clear()
    monkeypatch.setattr(train, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(drift_report, "PROJECT_ROOT", tmp_path)

    manifest = download.run(offline=True)
    assert manifest["source"] == "synthetic_offline"

    annotated = annotate.run()
    assert set(annotated) == {"2023-12", "2025-06"}
    for path in annotated.values():
        assert path.exists()

    dataset = build_dataset.run()
    train_path = tmp_path / "data" / "processed" / "train.parquet"
    test_path = tmp_path / "data" / "processed" / "test.parquet"
    assert train_path.exists() and test_path.exists()
    assert all(s["n"] > 0 for s in dataset["summaries"])

    result = train.run()
    assert result["best"] in {"logistic_regression", "random_forest",
                              "gradient_boosting", "hist_gradient_boosting"}
    assert (tmp_path / "models" / "best_model").exists()
    assert (tmp_path / "docs" / "MODEL_CARD.md").exists()

    summary = drift_report.run()
    assert "overall_alert" in summary
    assert (tmp_path / "reports" / "monitoring" / "drift_summary.json").exists()

    config_mod.load_config.cache_clear()
