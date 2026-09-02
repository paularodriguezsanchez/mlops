"""Tests de la traducción SHAP -> evidencia tipo ACMG (ADR 007)."""
from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline

from src.evaluate import acmg_evidence as acmg
from src.features.preprocess import FEATURE_COLUMNS, build_preprocessor
from src.ingest import synthetic as syn
from src.train.train import _models


def test_score_evidence_pp3_like_cuando_coherente():
    ev = acmg._score_evidence("cadd_phred", 32.0, shap_value=0.4)
    assert ev is not None
    assert ev["acmg_tag"] == "PP3-like"
    assert ev["direction"] == "patogénica"


def test_score_evidence_bp4_like_cuando_coherente_benigno():
    ev = acmg._score_evidence("revel_score", 0.05, shap_value=-0.3)
    assert ev is not None
    assert ev["acmg_tag"] == "BP4-like"
    assert ev["direction"] == "benigna"


def test_score_evidence_none_si_incoherente():
    # Score "dañino" pero el modelo lo usa para bajar el riesgo: no se reporta.
    assert acmg._score_evidence("cadd_phred", 32.0, shap_value=-0.2) is None
    assert acmg._score_evidence("cadd_phred", 2.0, shap_value=0.2) is None


def test_score_evidence_none_si_valor_ausente():
    assert acmg._score_evidence("cadd_phred", None, shap_value=0.4) is None
    assert acmg._score_evidence("sift_score", float("nan"), shap_value=0.4) is None


def test_population_frequency_pm2_like_variante_rara():
    ev = acmg._population_frequency_evidence(1e-6, shap_value=0.3)
    assert ev["acmg_tag"] == "PM2-like"


def test_population_frequency_ba1_like_variante_comun():
    ev = acmg._population_frequency_evidence(0.05, shap_value=-0.3)
    assert ev["acmg_tag"] == "BA1-like"


def test_population_frequency_none_en_rango_intermedio():
    assert acmg._population_frequency_evidence(0.001, shap_value=0.3) is None


def test_consequence_evidence_pvs1_like():
    ev = acmg._consequence_evidence("stop_gained", shap_value=0.5)
    assert ev["acmg_tag"] == "PVS1-like"
    assert acmg._consequence_evidence("missense_variant", shap_value=0.5) is None
    assert acmg._consequence_evidence("stop_gained", shap_value=-0.1) is None


def test_explain_variant_acmg_end_to_end():
    tr, _ = syn.generate_releases(syn.SyntheticConfig(
        seed=5, chromosomes=("1",), n_variants_train=300, n_new_in_test=10))
    df = pd.DataFrame({k: tr[k] for k in
                       [*FEATURE_COLUMNS, "chrom", "pos", "ref", "alt"]})
    y = pd.Series(tr["clnsig"]).isin(["Pathogenic", "Likely_pathogenic"]).astype(int)

    pipe = Pipeline([("pre", build_preprocessor()),
                     ("clf", _models(42)["gradient_boosting"])])
    pipe.fit(df[FEATURE_COLUMNS], y)

    background = df.iloc[:200]
    explain = df.iloc[200:210]
    sample, evidence = acmg.explain_variant_acmg(pipe, background, explain, seed=42)

    assert len(sample) == len(explain) == len(evidence)
    assert {"chrom", "pos", "ref", "alt"} <= set(sample.columns)
    for row_evidence in evidence:
        shap_abs = [abs(e["shap"]) for e in row_evidence]
        assert shap_abs == sorted(shap_abs, reverse=True)
        for e in row_evidence:
            assert e["acmg_tag"].endswith("-like")   # nunca una etiqueta ACMG "real"
