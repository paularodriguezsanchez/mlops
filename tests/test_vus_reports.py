"""Tests de los informes automáticos de VUS por plantilla (ADR 007)."""
from __future__ import annotations

import json

import pandas as pd
import yaml

from src import config as config_mod
from src.annotate import annotate
from src.features import build_dataset
from src.ingest import download
from src.serve import predictor as predictor_mod
from src.serve import vus_reports
from src.serve.annotator import get_annotator
from src.train import train


def _isolated_project(tmp_path, monkeypatch, n_train=300, n_new=60):
    cfg = yaml.safe_load(config_mod.CONFIG_PATH.read_text(encoding="utf-8"))
    # Fixture de test explícitamente sintética, aislada del valor de producción.
    cfg["data"]["annotation_source"] = "synthetic"
    cfg["synthetic"]["n_variants_train"] = n_train
    cfg["synthetic"]["n_new_in_test"] = n_new
    (tmp_path / "config").mkdir()
    cfg_path = tmp_path / "config" / "config.yaml"
    cfg_path.write_text(yaml.dump(cfg), encoding="utf-8")

    monkeypatch.setattr(config_mod, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config_mod, "CONFIG_PATH", cfg_path)
    config_mod.load_config.cache_clear()
    monkeypatch.setattr(train, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(vus_reports, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(predictor_mod, "_MODEL_DIR", tmp_path / "models" / "best_model")
    monkeypatch.setattr(vus_reports, "_RECLASS_MODEL_DIR",
                        tmp_path / "models" / "reclassification_model")
    predictor_mod.get_predictor.cache_clear()
    get_annotator.cache_clear()

    download.run(offline=True)
    annotate.run()
    build_dataset.run()
    train.run()


def test_generate_reports_sin_modelo_de_reclasificacion(tmp_path, monkeypatch):
    """Sin el modelo de reclasificación entrenado, degrada con mensaje explícito, no rompe (mismo
    patrón ADR 005).
    """
    _isolated_project(tmp_path, monkeypatch)

    result = vus_reports.generate_reports(split="test", top_n=3)

    assert result["n"] <= 3
    assert result["doc"].exists()
    assert result["json"].exists()
    doc_text = result["doc"].read_text(encoding="utf-8")
    assert "no disponible" in doc_text
    records = json.loads(result["json"].read_text(encoding="utf-8"))
    assert len(records) == result["n"]
    for rec in records:
        assert rec["probabilidad_reclasificacion"] is None
        assert rec["probabilidad_reclasificacion_fiable"] is None
        assert 0.0 <= rec["probabilidad_patogenica"] <= 1.0
        assert isinstance(rec["evidencia"], list)
        for e in rec["evidencia"]:
            assert e["acmg_tag"].endswith("-like")


def test_generate_reports_con_modelo_de_reclasificacion(tmp_path, monkeypatch):
    from src.train import train_reclass
    monkeypatch.setattr(train_reclass, "PROJECT_ROOT", tmp_path)

    # Con solo 60 VUS nuevas, frac_reclassified=0.06 no deja suficientes
    # positivos para el holdout estratificado del modelo de reclasificación: se sube
    # para este test.
    _isolated_project(tmp_path, monkeypatch, n_train=300, n_new=60)
    cfg_path = tmp_path / "config" / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["synthetic"]["frac_reclassified"] = 0.3
    cfg_path.write_text(yaml.dump(cfg), encoding="utf-8")
    config_mod.load_config.cache_clear()
    download.run(offline=True, force=True)
    annotate.run()
    build_dataset.run()
    train_reclass.run()

    result = vus_reports.generate_reports(split="test", top_n=3)

    records = json.loads(result["json"].read_text(encoding="utf-8"))
    assert any(r["probabilidad_reclasificacion"] is not None for r in records)
    for rec in records:
        # Siempre que hay probabilidad, viene acompañada de su fiabilidad.
        if rec["probabilidad_reclasificacion"] is not None:
            assert isinstance(rec["probabilidad_reclasificacion_fiable"], bool)
        else:
            assert rec["probabilidad_reclasificacion_fiable"] is None
    assert (tmp_path / "models" / "reclassification_model" / "metrics.json").exists()


def test_render_variant_avisa_si_reclasificacion_no_fiable():
    row = pd.Series({"chrom": "1", "pos": 1, "ref": "A", "alt": "G", "gene": "GENE1",
                     "consequence": "missense_variant", "probabilidad_patogenica": 0.5})
    text = vus_reports._render_variant(row, [], 0.2, reclass_reliable=False)
    assert "AVISO" in text
    assert "20.0%" in text


def test_render_variant_sin_aviso_si_reclasificacion_fiable():
    row = pd.Series({"chrom": "1", "pos": 1, "ref": "A", "alt": "G", "gene": "GENE1",
                     "consequence": "missense_variant", "probabilidad_patogenica": 0.5})
    text = vus_reports._render_variant(row, [], 0.2, reclass_reliable=True)
    assert "AVISO" not in text
    assert "20.0%" in text


def test_generate_reports_usa_ranking_si_esta_entrenado(tmp_path, monkeypatch):
    """Con el objetivo de ranking entrenado, el generador de informes debe ordenar y reportar
    por su score (no solo por el modelo de patogenicidad).
    """
    from src.serve import prioritize_vus as prioritize_vus_mod
    from src.train import train_ranking

    monkeypatch.setattr(train_ranking, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(prioritize_vus_mod, "_RANKING_MODEL_DIR",
                        tmp_path / "models" / "ranking_model")

    _isolated_project(tmp_path, monkeypatch)
    train_ranking.run()

    result = vus_reports.generate_reports(split="test", top_n=3)

    records = json.loads(result["json"].read_text(encoding="utf-8"))
    assert len(records) == result["n"]
    for rec in records:
        assert "ranking_score" in rec
    doc_text = result["doc"].read_text(encoding="utf-8")
    assert "score del modelo de ranking" in doc_text
