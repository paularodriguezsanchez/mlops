"""Tests de explicabilidad SHAP (C4, bloque de trabajo)."""
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.evaluate.explain import explain_model
from src.features.preprocess import FEATURE_COLUMNS, build_preprocessor
from src.ingest import synthetic as syn
from src.train import train


def _synthetic_frame(seed: int, n: int) -> tuple[pd.DataFrame, np.ndarray]:
    tr, _ = syn.generate_releases(
        syn.SyntheticConfig(seed=seed, chromosomes=("1",), n_variants_train=n, n_new_in_test=1)
    )
    df = pd.DataFrame({k: tr[k] for k in FEATURE_COLUMNS})
    y = np.isin(tr["clnsig"], ["Pathogenic", "Likely_pathogenic"]).astype(int)
    return df, y


def test_explain_model_produce_ranking_valido(tmp_path):
    X_train, y_train = _synthetic_frame(seed=1, n=300)
    X_explain, _ = _synthetic_frame(seed=2, n=80)

    pipe = Pipeline([
        ("pre", build_preprocessor()),
        ("clf", train._models(42)["logistic_regression"]),
    ])
    pipe.fit(X_train, y_train)

    csv_path = explain_model(pipe, X_train, X_explain, tmp_path, seed=42)

    assert csv_path.exists()
    plot_path = tmp_path / "shap_summary.png"
    assert plot_path.exists()

    importance = pd.read_csv(csv_path)
    assert set(importance["feature"]) == set(FEATURE_COLUMNS)
    assert (importance["shap_importance"] >= 0).all()
    # Orden descendente por importancia.
    assert list(importance["shap_importance"]) == sorted(
        importance["shap_importance"], reverse=True
    )


def test_explain_model_respeta_limites_de_muestreo(tmp_path):
    X_train, y_train = _synthetic_frame(seed=3, n=500)
    X_explain, _ = _synthetic_frame(seed=4, n=500)

    pipe = Pipeline([
        ("pre", build_preprocessor()),
        ("clf", train._models(42)["logistic_regression"]),
    ])
    pipe.fit(X_train, y_train)

    csv_path = explain_model(pipe, X_train, X_explain, tmp_path, seed=42)
    importance = pd.read_csv(csv_path)
    # No debe fallar ni disparar coste con datasets mayores que los límites internos.
    assert len(importance) == len(FEATURE_COLUMNS)
