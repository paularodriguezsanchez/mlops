"""Explicabilidad del modelo mejor mediante SHAP (C4, P1).

Complementa la importancia por permutación (`src/train/train.py`) con SHAP:
valores por instancia basados en teoría de juegos (Shapley values), no solo
un ranking global por barrido de una feature.

Se explica el Pipeline COMPLETO (preprocesado + clasificador) como caja negra
sobre las columnas de ENTRADA (mismo criterio que la importancia por
permutación), para que ambos análisis sean directamente comparables y
legibles en la memoria sin exponer las columnas expandidas por one-hot.

Nota técnica: el masker tabular de SHAP compara valores con `np.isclose`,
que no admite texto. Como `consequence` es categórica, se codifica a enteros
antes de invocar SHAP y se decodifica de vuelta a texto justo antes de cada
llamada real al pipeline (que sí espera la categoría como string).
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

# Acotan el coste: SHAP, agnóstico al modelo, evalúa el pipeline muchas veces por
# instancia explicada. Suficientes para un ranking e informe estables.
_MAX_BACKGROUND = 50
_MAX_EXPLAIN = 200


def _category_maps(*frames: pd.DataFrame) -> dict[str, list]:
    """Categorías únicas por columna, ordenadas, EXCLUYENDO nulos.

    Con ClinVar real (a diferencia del generador sintético, que siempre
    asignaba una `consequence` de un catálogo fijo) algunas variantes no
    tienen consecuencia molecular anotada (`MC=` ausente en el VCF) -> NaN/
    None, que `sorted` no puede comparar con texto. Se excluyen de la lista
    de categorías; `_encode` ya les asigna código -1 (pandas Categorical.codes
    para valores fuera de catálogo), distinto del resto — es una aproximación
    aceptable para SHAP (heurística de explicabilidad, no el modelo real: el
    pipeline entrenado sí imputa `consequence` ausente por moda, `preprocess.py`).
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
    """Calcula SHAP (clase positiva) para una muestra de `X_explain`.

    Devuelve `(shap_values, sample_raw, sample_encoded)`: `shap_values` es el
    objeto de `shap.Explainer` (clase positiva únicamente, `.values` de forma
    `(n, n_features)`); `sample_raw` es el DataFrame original (sin codificar, útil
    para citar el valor real de cada feature junto a su contribución
    SHAP, la evidencia ACMG-símil ADR 007) — **conserva el índice original de `X_explain`** (no se
    resetea) precisamente para que un caller pueda re-identificar cada fila
    tras el muestreo/posible reordenado interno; `sample_encoded` es la misma
    muestra con la columna categórica codificada a enteros (la que espera
    `shap.summary_plot`).

    Factorizado de `explain_model` para reutilizarse en la traducción a
    evidencia tipo ACMG (`src/evaluate/acmg_evidence.py`) sin duplicar la
    lógica de codificación/decodificación de la columna categórica.
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
    """Calcula SHAP values sobre una muestra de `X_explain` y guarda artefactos.

    * `X_background` referencia la distribución base (típicamente train); se
      submuestrea a `_MAX_BACKGROUND` filas.
    * `X_explain` es la muestra a explicar (típicamente el holdout no visto);
      se submuestrea a `_MAX_EXPLAIN` filas.

    Guarda en `out_dir`: `shap_importance.csv` (ranking global, media del
    valor absoluto) y `shap_summary.png` (summary plot por instancia; la
    columna categórica aparece codificada como entero, no como texto).
    Devuelve la ruta del CSV.
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
