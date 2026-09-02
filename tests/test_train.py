"""Tests del entrenamiento y las métricas ."""
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.evaluate.metrics import compute_metrics
from src.features.preprocess import build_preprocessor
from src.ingest import synthetic as syn
from src.train import train


def test_cuatro_algoritmos_estandar():
    models = train._models(seed=42)
    assert set(models) == {"logistic_regression", "random_forest",
                           "gradient_boosting", "hist_gradient_boosting"}


def test_metricas_claves_y_rango():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=200)
    prob = np.clip(y * 0.6 + rng.normal(0, 0.3, size=200), 0, 1)
    m = compute_metrics(y, prob)
    assert set(m) == {"pr_auc", "roc_auc", "f1", "precision", "recall"}
    assert all(0.0 <= v <= 1.0 for v in m.values())


def test_pipeline_entrena_y_predice():
    tr, _ = syn.generate_releases(syn.SyntheticConfig(
        seed=3, chromosomes=("1",), n_variants_train=300, n_new_in_test=10))
    df = pd.DataFrame({k: tr[k] for k in
                       ["sift_score", "polyphen_score", "revel_score", "alphamissense_score",
                        "cadd_phred", "gerp_rs", "phylop", "gnomad_af", "consequence"]})
    y = np.isin(tr["clnsig"], ["Pathogenic", "Likely_pathogenic"]).astype(int)
    pipe = Pipeline([("pre", build_preprocessor()),
                     ("clf", train._models(42)["logistic_regression"])])
    pipe.fit(df, y)
    prob = pipe.predict_proba(df)[:, 1]
    assert prob.shape == (len(df),)
    assert compute_metrics(y, prob)["roc_auc"] > 0.7   # señal aprendible
