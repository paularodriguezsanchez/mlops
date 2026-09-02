"""Tests de la priorización de VUS por riesgo estimado (ligado a ADR 006)."""
import numpy as np
import pandas as pd

from src.serve.predictor import VariantPredictor
from src.serve.prioritize_vus import load_ranking_model, rank_vus


class _DummyModel:
    """Patogénica si CADD alto, igual que el dummy de test_serve.py."""

    def predict_proba(self, X):
        p = (X["cadd_phred"] >= 20).astype(float).to_numpy()
        return np.array([[1 - x, x] for x in p])


def _vus_frame() -> pd.DataFrame:
    return pd.DataFrame([
        {"chrom": "1", "pos": 100, "ref": "A", "alt": "G", "gene": "GENE1",
         "consequence": "missense_variant", "clnsig": "Uncertain_significance",
         "sift_score": 0.4, "polyphen_score": 0.5, "revel_score": 0.5,
         "alphamissense_score": 0.5,
         "cadd_phred": 10.0, "gerp_rs": 2.0, "phylop": 1.0, "gnomad_af": 1e-3},
        {"chrom": "2", "pos": 200, "ref": "C", "alt": "T", "gene": "GENE2",
         "consequence": "missense_variant", "clnsig": "Uncertain_significance",
         "sift_score": 0.1, "polyphen_score": 0.9, "revel_score": 0.8,
         "alphamissense_score": 0.9,
         "cadd_phred": 32.0, "gerp_rs": 5.0, "phylop": 6.0, "gnomad_af": 1e-7},
        {"chrom": "3", "pos": 300, "ref": "G", "alt": "A", "gene": "GENE3",
         "consequence": "intron_variant", "clnsig": "Uncertain_significance",
         "sift_score": None, "polyphen_score": None, "revel_score": None,
         "alphamissense_score": None,
         "cadd_phred": 5.0, "gerp_rs": -1.0, "phylop": -2.0, "gnomad_af": 1e-2},
    ])


def test_rank_vus_ordena_por_riesgo_descendente():
    vus = _vus_frame()
    predictor = VariantPredictor(_DummyModel(), annotator=None)

    ranked = rank_vus(predictor, vus)

    assert list(ranked["chrom"]) == ["2", "1", "3"]  # CADD 32 > 10 > 5
    assert list(ranked["probabilidad_patogenica"]) == sorted(
        ranked["probabilidad_patogenica"], reverse=True
    )


def test_rank_vus_no_pierde_ni_altera_filas():
    vus = _vus_frame()
    predictor = VariantPredictor(_DummyModel(), annotator=None)

    ranked = rank_vus(predictor, vus)

    assert len(ranked) == len(vus)
    assert set(ranked["gene"]) == set(vus["gene"])
    assert "probabilidad_patogenica" in ranked.columns


class _DummyRankingPreprocessor:
    """Transform mínimo: solo `cadd_phred`, como matriz 2D (forma que espera un booster)."""

    def transform(self, X):
        return X[["cadd_phred"]].to_numpy()


class _DummyRankingModel:
    """Puntúa al revés que CADD, deliberadamente: prueba que el orden lo decide
    este modelo  y no la probabilidad de patogenicidad (que sí sigue CADD)."""

    def predict(self, X):
        return -X[:, 0]


def test_rank_vus_usa_el_modelo_de_ranking_si_se_proporciona():
    vus = _vus_frame()
    predictor = VariantPredictor(_DummyModel(), annotator=None)

    ranked = rank_vus(predictor, vus, _DummyRankingModel(), _DummyRankingPreprocessor())

    assert "ranking_score" in ranked.columns
    assert list(ranked["chrom"]) == ["3", "1", "2"]  # -CADD: -5 > -10 > -32
    assert list(ranked["ranking_score"]) == sorted(ranked["ranking_score"], reverse=True)
    # La probabilidad de patogenicidad se sigue calculando y mostrando, aunque no ordene.
    assert "probabilidad_patogenica" in ranked.columns


def test_rank_vus_sin_modelo_de_ranking_no_anade_columna():
    vus = _vus_frame()
    predictor = VariantPredictor(_DummyModel(), annotator=None)

    ranked = rank_vus(predictor, vus)

    assert "ranking_score" not in ranked.columns
    assert list(ranked["chrom"]) == ["2", "1", "3"]  # comportamiento previo intacto (CADD)


def test_load_ranking_model_degrada_si_no_existe(tmp_path, monkeypatch):
    import src.serve.prioritize_vus as pv

    monkeypatch.setattr(pv, "_RANKING_MODEL_DIR", tmp_path / "no_existe")

    model, preprocessor = load_ranking_model()

    assert model is None
    assert preprocessor is None
