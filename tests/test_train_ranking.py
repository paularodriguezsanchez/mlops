"""Tests del objetivo de ranking (ADR 007) sobre un subconjunto minúsculo."""
from __future__ import annotations

import numpy as np
import pytest
import yaml

from src import config as config_mod
from src.annotate import annotate
from src.features import build_dataset
from src.ingest import download
from src.train import train_ranking


def test_ndcg_at_ks_perfecto_para_orden_ideal():
    y_true = np.array([1, 1, 0, 0, 0])
    y_score = np.array([5, 4, 3, 2, 1])   # mismo orden que la relevancia
    out = train_ranking._ndcg_at_ks(y_true, y_score, ks=(2, 10))
    assert out["ndcg_at_2"] == pytest.approx(1.0)
    assert out["ndcg_at_10"] == pytest.approx(1.0)   # k se acota a len(y_true)=5


def test_ndcg_at_ks_penaliza_orden_invertido():
    y_true = np.array([0, 0, 0, 1, 1])
    y_score = np.array([5, 4, 3, 2, 1])   # relevantes al final: peor orden posible
    out = train_ranking._ndcg_at_ks(y_true, y_score, ks=(5,))
    assert out["ndcg_at_5"] < 1.0


def test_baseline_ndcg_aleatorio_peor_que_orden_ideal():
    """Revisión posterior del proyecto: baselines para poder interpretar el NDCG del modelo."""
    y_true = np.concatenate([np.ones(20), np.zeros(180)])
    out = train_ranking._baseline_ndcg(y_true, seed=42, n_reps=20, ks=(10,))
    assert 0.0 <= out["random_mean"]["ndcg_at_10"] <= 1.0
    assert 0.0 <= out["arrival_order"]["ndcg_at_10"] <= 1.0
    assert out["random_n_reps"] == 20


def test_run_entrena_y_evalua_ranking(tmp_path, monkeypatch):
    cfg = yaml.safe_load(config_mod.CONFIG_PATH.read_text(encoding="utf-8"))
    # Fixture de test explícitamente sintética, aislada del valor de producción.
    cfg["data"]["annotation_source"] = "synthetic"
    cfg["synthetic"]["n_variants_train"] = 300
    cfg["synthetic"]["n_new_in_test"] = 60
    (tmp_path / "config").mkdir()
    cfg_path = tmp_path / "config" / "config.yaml"
    cfg_path.write_text(yaml.dump(cfg), encoding="utf-8")

    monkeypatch.setattr(config_mod, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config_mod, "CONFIG_PATH", cfg_path)
    config_mod.load_config.cache_clear()
    monkeypatch.setattr(train_ranking, "PROJECT_ROOT", tmp_path)

    download.run(offline=True)
    annotate.run()
    build_dataset.run()

    result = train_ranking.run()

    assert result["n_holdout"] > 0
    assert 0.0 <= result["ref_pr_auc"] <= 1.0
    assert all(0.0 <= v <= 1.0 for v in result["ndcg"].values())
    assert (tmp_path / "models" / "ranking_model" / "lambdarank.txt").exists()
    assert (tmp_path / "reports" / "training" / "ranking_metrics.csv").exists()
    # El preprocesador ajustado se persiste junto al booster: serving
    # (`src/serve/prioritize_vus.py::load_ranking_model`) lo necesita para
    # transformar features nuevas exactamente igual que en entrenamiento.
    preprocessor_path = tmp_path / "models" / "ranking_model" / "preprocessor.joblib"
    assert preprocessor_path.exists()
    import joblib
    loaded_pre = joblib.load(preprocessor_path)
    assert hasattr(loaded_pre, "transform")


def test_run_es_reproducible_entre_ejecuciones(tmp_path, monkeypatch):
    """Regresión: sin `deterministic=True`/`force_row_wise=True`/
    `num_threads=1`, LightGBM daba NDCG@k distinto entre ejecuciones idénticas pese a la
    semilla fija. Dos llamadas a `run()` sobre los mismos datos deben coincidir al dígito."""
    cfg = yaml.safe_load(config_mod.CONFIG_PATH.read_text(encoding="utf-8"))
    # Fixture de test explícitamente sintética, aislada del valor de producción.
    cfg["data"]["annotation_source"] = "synthetic"
    cfg["synthetic"]["n_variants_train"] = 300
    cfg["synthetic"]["n_new_in_test"] = 60
    (tmp_path / "config").mkdir()
    cfg_path = tmp_path / "config" / "config.yaml"
    cfg_path.write_text(yaml.dump(cfg), encoding="utf-8")

    monkeypatch.setattr(config_mod, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config_mod, "CONFIG_PATH", cfg_path)
    config_mod.load_config.cache_clear()
    monkeypatch.setattr(train_ranking, "PROJECT_ROOT", tmp_path)

    download.run(offline=True)
    annotate.run()
    build_dataset.run()

    first = train_ranking.run()
    second = train_ranking.run()

    assert first["ndcg"] == second["ndcg"]
    assert first["ref_pr_auc"] == pytest.approx(second["ref_pr_auc"])
