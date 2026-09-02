"""Explicabilidad SHAP del modelo seleccionado.

Complementa la importancia por permutación con valores por instancia, que dan
dirección y magnitud además de un ranking global.

Se explica el pipeline completo como caja negra sobre las columnas de entrada, no
sobre las expandidas por one-hot, para que ambos análisis sean comparables.

El masker tabular de SHAP compara con `np.isclose`, que no admite texto: la columna
categórica se codifica a enteros antes de invocarlo y se decodifica justo antes de
cada llamada real al pipeline.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from src.features.preprocess import CATEGORICAL_FEATURES

# Acotan el coste: SHAP agnóstico al modelo evalúa el pipeline muchas veces por
# instancia. Suficiente para un ranking estable.
_MAX_BACKGROUND = 50
_MAX_EXPLAIN = 200


def _category_maps(*frames: pd.DataFrame) -> dict[str, list]:
    """Categorías únicas por columna, ordenadas, excluyendo nulos.

    En ClinVar real hay variantes sin consecuencia anotada, y `sorted` no compara
    NaN con texto. Se excluyen del catálogo y `_encode` les asigna código -1. Es una
    aproximación aceptable aquí: el pipeline entrenado sí imputa por moda.
    """
    return {
        col: sorted({v for f in frames for v in f[col] if pd.notna(v)})
        for col in CATEGORICAL_FEATURES
    }


def _encode(df: pd.DataFrame, categories: dict[str, list]) -> pd.DataFrame:
    """Categórica → códigos enteros (float), para que todo el frame sea numérico."""
    out = df.copy()
    for col, cats in categories.items():
        out[col] = pd.Categorical(out[col], categories=cats).codes.astype(float)
    return out


def _decode(arr: np.ndarray, columns: list[str], categories: dict[str, list]) -> pd.DataFrame:
    """Inversa de `_encode`: reconstruye el DataFrame que espera el pipeline real."""
    df = pd.DataFrame(np.asarray(arr), columns=columns)
    for col, cats in categories.items():
        codes = df[col].round().astype(int).clip(lower=0, upper=len(cats) - 1)
        df[col] = [cats[c] for c in codes]
    return df


def compute_shap_values(
    pipe: Pipeline,
    X_background: pd.DataFrame,
    X_explain: pd.DataFrame,
    seed: int = 42,
    max_background: int = _MAX_BACKGROUND,
    max_explain: int = _MAX_EXPLAIN,
):
    """SHAP de la clase positiva para una muestra de `X_explain`.

    Devuelve `(shap_values, sample_raw, sample_encoded)`. `sample_raw` conserva el
    índice original para que el llamante pueda reidentificar cada fila tras el
    muestreo interno; `sample_encoded` es la misma muestra con la categórica ya
    codificada, que es lo que espera `shap.summary_plot`.
    """
    background_raw = shap.sample(
        X_background, min(max_background, len(X_background)), random_state=seed
    )
    sample_raw = X_explain.sample(n=min(max_explain, len(X_explain)), random_state=seed)

    categories = _category_maps(background_raw, sample_raw)
    background = _encode(background_raw, categories)
    sample = _encode(sample_raw, categories)
    columns = list(sample.columns)

    def predict_proba_numeric(arr: np.ndarray) -> np.ndarray:
        return pipe.predict_proba(_decode(arr, columns, categories))

    explainer = shap.Explainer(predict_proba_numeric, background, seed=seed)
    shap_values = explainer(sample)
    return shap_values[..., 1], sample_raw, sample


def explain_model(
    pipe: Pipeline,
    X_background: pd.DataFrame,
    X_explain: pd.DataFrame,
    out_dir: Path,
    seed: int = 42,
) -> Path:
    """Calcula SHAP sobre una muestra de `X_explain` y guarda los artefactos.

    `X_background` fija la distribución de referencia (train) y `X_explain` es lo que
    se explica (el holdout no visto); ambos se submuestrean.

    Escribe `shap_importance.csv` (ranking global por media del valor absoluto) y
    `shap_summary.png`. Devuelve la ruta del CSV.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    positive, _, sample = compute_shap_values(pipe, X_background, X_explain, seed=seed)
    columns = list(sample.columns)

    mean_abs = np.abs(positive.values).mean(axis=0)
    importance = (
        pd.DataFrame({"feature": columns, "shap_importance": mean_abs})
        .sort_values("shap_importance", ascending=False)
    )
    csv_path = out_dir / "shap_importance.csv"
    importance.to_csv(csv_path, index=False)

    plt.figure()
    shap.summary_plot(positive, sample, show=False)
    plot_path = out_dir / "shap_summary.png"
    plt.savefig(plot_path, bbox_inches="tight", dpi=120)
    plt.close()

    print("Top features (SHAP):", ", ".join(importance.head(5)["feature"]))
    return csv_path
