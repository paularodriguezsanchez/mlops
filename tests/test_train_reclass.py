"""Tests del modelo de reclasificación de VUS (ADR 007)."""
from __future__ import annotations

import json

import pandas as pd
import pytest
import yaml

from src import config as config_mod
from src.annotate import annotate
from src.features import build_dataset
from src.ingest import download
from src.train import train_reclass


def test_label_reclassified_marca_solo_las_resueltas():
    vus = pd.DataFrame([
        {"chrom": "1", "pos": 1, "ref": "A", "alt": "G", "consequence": "missense_variant"},
        {"chrom": "1", "pos": 2, "ref": "A", "alt": "G", "consequence": "missense_variant"},
        {"chrom": "1", "pos": 3, "ref": "A", "alt": "G", "consequence": "missense_variant"},
    ])
    annotated_new = pd.DataFrame([
        {"chrom": "1", "pos": 1, "ref": "A", "alt": "G", "clnsig": "Likely_pathogenic"},
        {"chrom": "1", "pos": 2, "ref": "A", "alt": "G", "clnsig": "Uncertain_significance"},
        # pos 3 no aparece en la release nueva -> no reclasificada
    ])
    out = train_reclass._label_reclassified(vus, annotated_new)
    assert list(out["label"]) == [1, 0, 0]
    assert "clnsig_new" not in out.columns


def test_run_lanza_error_si_apenas_hay_reclasificadas(tmp_path, monkeypatch):
    """Salvaguarda: con muy pocos positivos, un holdout estratificado no tiene sentido."""
    cfg_path = _tiny_config(tmp_path, frac_reclassified=0.0)
    _patch_project_root(monkeypatch, tmp_path, cfg_path)
    download.run(offline=True)
    annotate.run()
    build_dataset.run()
    with pytest.raises(ValueError, match="pocos ejemplos"):
        train_reclass.run()


def test_run_entrena_y_registra_modelo_de_reclasificacion(tmp_path, monkeypatch):
    cfg_path = _tiny_config(tmp_path, frac_reclassified=0.3)
    _patch_project_root(monkeypatch, tmp_path, cfg_path)
    download.run(offline=True)
    annotate.run()
    build_dataset.run()

    result = train_reclass.run()

    assert result["best"] in {"logistic_regression", "random_forest",
                              "gradient_boosting", "hist_gradient_boosting"}
    assert result["n_reclassified"] > 0
    assert (tmp_path / "models" / "reclassification_model").exists()
    assert (tmp_path / "docs" / "MODEL_CARD_RECLASSIFICATION.md").exists()
    comp = pd.read_csv(tmp_path / "reports" / "training" / "reclassification_model_comparison.csv")
    assert len(comp) == 4
    assert (comp["pr_auc"] >= 0).all() and (comp["pr_auc"] <= 1).all()

    # ADR 008: ablación "temporalmente segura" (mitigación parcial de leakage).
    assert "ablation" in result
    assert result["ablation"]["features"] == ["consequence", "review_stars"]
    assert 0.0 <= result["ablation"]["roc_auc"] <= 1.0
    metrics_json = json.loads(
        (tmp_path / "models" / "reclassification_model" / "metrics.json").read_text())
    assert "leakage_ablation_temporally_safe" in metrics_json
    card_text = (tmp_path / "docs" / "MODEL_CARD_RECLASSIFICATION.md").read_text(encoding="utf-8")
    assert "Leakage temporal" in card_text
    assert "Ablación temporalmente segura" in card_text


def test_run_prospective_evalua_sin_reentrenar(tmp_path, monkeypatch):
    """Revisión posterior del proyecto: validación temporal prospectiva real."""
    prospective_rel = "2099-01"
    cfg_path = _tiny_config(tmp_path, frac_reclassified=0.3, prospective_rel=prospective_rel)
    _patch_project_root(monkeypatch, tmp_path, cfg_path)
    download.run(offline=True)
    annotate.run()
    build_dataset.run()
    train_reclass.run()

    # VUS que en la release de entrenamiento seguían "abiertas" (label=0):
    # se resuelve una de ellas en la release prospectiva de mentira, el resto no.
    still_open = train_reclass.build_reclass_dataset("2025-06")
    still_open = still_open.loc[still_open["label"] == 0]
    still_open_keys = still_open[["chrom", "pos", "ref", "alt"]].head(3)
    from src.ingest import synthetic as syn
    n = len(still_open_keys)
    release = {
        "chrom": still_open_keys["chrom"].to_numpy(),
        "pos": still_open_keys["pos"].to_numpy(),
        "ref": still_open_keys["ref"].to_numpy(),
        "alt": still_open_keys["alt"].to_numpy(),
        "gene": ["G"] * n,
        "consequence": ["missense_variant"] * n,
        "review_status": ["criteria_provided,_single_submitter"] * n,
        "clnsig": ["Pathogenic"] + ["Uncertain_significance"] * (n - 1),
    }
    prospective_path = tmp_path / "data" / "raw" / f"clinvar_{prospective_rel}.vcf.gz"
    syn.write_clinvar_vcf(release, prospective_path)

    result = train_reclass.run_prospective()

    assert result["prospective_release_c"] == prospective_rel
    assert result["n_resolved_by_c"] == 1
    assert result["n_found_in_c"] == n
    assert (tmp_path / "models" / "reclassification_model" / "prospective_metrics.json").exists()
    card = (tmp_path / "docs" / "MODEL_CARD_RECLASSIFICATION_PROSPECTIVE.md").read_text(
        encoding="utf-8")
    assert "PROSPECTIVA" in card


def test_run_prospective_sin_modelo_entrenado_falla_explicito(tmp_path, monkeypatch):
    cfg_path = _tiny_config(tmp_path, frac_reclassified=0.3, prospective_rel="2099-01")
    _patch_project_root(monkeypatch, tmp_path, cfg_path)
    download.run(offline=True)
    annotate.run()
    build_dataset.run()
    with pytest.raises(RuntimeError, match="No hay modelo"):
        train_reclass.run_prospective()


def _tiny_config(tmp_path, frac_reclassified: float, prospective_rel: str | None = None):
    cfg = yaml.safe_load(config_mod.CONFIG_PATH.read_text(encoding="utf-8"))
    # Fixture de test explícitamente sintética, aislada del valor de producción.
    cfg["data"]["annotation_source"] = "synthetic"
    cfg["synthetic"]["n_variants_train"] = 300
    cfg["synthetic"]["n_new_in_test"] = 30
    cfg["synthetic"]["frac_reclassified"] = frac_reclassified
    if prospective_rel:
        cfg["data"]["clinvar_prospective_release"] = prospective_rel
    (tmp_path / "config").mkdir()
    cfg_path = tmp_path / "config" / "config.yaml"
    cfg_path.write_text(yaml.dump(cfg), encoding="utf-8")
    return cfg_path


def _patch_project_root(monkeypatch, tmp_path, cfg_path):
    monkeypatch.setattr(config_mod, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config_mod, "CONFIG_PATH", cfg_path)
    config_mod.load_config.cache_clear()
    monkeypatch.setattr(train_reclass, "PROJECT_ROOT", tmp_path)
