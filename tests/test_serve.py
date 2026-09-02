"""Tests del servicio de inferencia ."""
import pandas as pd
import pytest

from src.features.preprocess import FEATURE_COLUMNS
from src.serve.annotator import VariantAnnotator


@pytest.fixture
def annotator():
    store = pd.DataFrame([{
        "chrom": "1", "pos": 100, "ref": "A", "alt": "G", "gene": "GENE1",
        "consequence": "missense_variant", "sift_score": 0.1, "polyphen_score": 0.9,
        "revel_score": 0.8, "alphamissense_score": 0.85,
        "cadd_phred": 30.0, "gerp_rs": 5.0, "phylop": 6.0,
        "gnomad_af": 1e-6,
    }])
    return VariantAnnotator(store)


def test_annotator_devuelve_features(annotator):
    ann = annotator.annotate("1", 100, "A", "G")
    assert ann is not None
    assert set(FEATURE_COLUMNS) <= set(ann)
    assert "clnsig" not in ann          # no filtra la etiqueta


def test_annotator_variante_desconocida(annotator):
    assert annotator.annotate("9", 999, "T", "C") is None


class _DummyModel:
    def predict_proba(self, X):
        # patogénica si CADD alto
        p = (X["cadd_phred"] >= 20).astype(float).to_numpy()
        import numpy as np
        return np.array([[1 - x, x] for x in p])


def test_predictor_anota_y_predice(annotator):
    from src.serve.predictor import VariantPredictor
    pred = VariantPredictor(_DummyModel(), annotator)
    r = pred.predict("1", 100, "A", "G")
    assert r["annotated"] is True
    assert r["prediction"] == "Patogénica"
    assert 0.0 <= r["probability_pathogenic"] <= 1.0


def test_predictor_variante_no_anotable(annotator):
    from src.serve.predictor import VariantPredictor
    pred = VariantPredictor(_DummyModel(), annotator)
    r = pred.predict("9", 999, "T", "C")
    assert r["annotated"] is False
