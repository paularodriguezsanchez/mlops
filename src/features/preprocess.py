"""Preprocesamiento compartido por los tres modelos.

Un único `ColumnTransformer`, embebido dentro del `Pipeline` que se registra en
MLflow, garantiza la misma transformación en entrenamiento y en serving.

Decisiones:
  * `gnomad_af` es muy asimétrica: log10 antes de imputar y escalar.
  * SIFT, PolyPhen, REVEL y AlphaMissense son NaN fuera de missense: imputación
    por mediana con indicador de ausencia, porque la ausencia es informativa.
  * CADD, GERP++ y phyloP reciben el mismo tratamiento: con ClinVar real su tasa
    de nulos es tan alta como la del grupo anterior (~75 %), porque la fuente solo
    cubre variantes no sinónimas. El nombre del grupo viene del supuesto inicial,
    que solo se cumplía con el generador sintético.
  * `consequence`: one-hot con `handle_unknown="ignore"`, para no romper en
    serving ante categorías nuevas.
  * `gene` se excluye por alta cardinalidad y riesgo de fuga.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.config import processed_dir

# Scores in silico que pueden faltar (NaN en no missense) → imputar + indicador.
SCORE_FEATURES = ["sift_score", "polyphen_score", "revel_score", "alphamissense_score"]
# Numéricas prácticamente siempre presentes.
DENSE_NUMERIC = ["cadd_phred", "gerp_rs", "phylop"]
# Numérica muy asimétrica → escala logarítmica.
LOG_FEATURES = ["gnomad_af"]
CATEGORICAL_FEATURES = ["consequence"]

FEATURE_COLUMNS = SCORE_FEATURES + DENSE_NUMERIC + LOG_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "label"

# Estado de revisión de ClinVar (0-4 estrellas), solo para el modelo de
# reclasificación (ADR 008): se lee del VCF fechado de cada release, así que es
# temporalmente seguro, y más consenso entre submitters predice resolución. No se
# añade al modelo de patogenicidad porque ahí el estado de revisión está confundido
# con la certeza de la propia etiqueta que se quiere predecir.
RECLASS_EXTRA_FEATURES = ["review_stars"]
RECLASS_FEATURE_COLUMNS = FEATURE_COLUMNS + RECLASS_EXTRA_FEATURES

# Subconjunto temporalmente seguro para la ablación de leakage (ADR 008): las dos
# únicas features que se leen del VCF fechado, sin re-consulta "de hoy".
SAFE_RECLASS_FEATURE_COLUMNS = [*CATEGORICAL_FEATURES, *RECLASS_EXTRA_FEATURES]


def _log10_clip(x):
    """log10 con recorte inferior para evitar log(0). Módulo-level (picklable)."""
    return np.log10(np.clip(np.asarray(x, dtype=float), 1e-7, None))


def build_preprocessor() -> ColumnTransformer:
    """Devuelve el ColumnTransformer del proyecto (sin ajustar)."""
    score_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    dense_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    log_pipe = Pipeline([
        ("log", FunctionTransformer(_log10_clip, feature_names_out="one-to-one")),
        # La ausencia de gnomAD no equivale a rareza poblacional: también puede ser
        # un fallo de cobertura de la fuente. El indicador conserva esa distinción en
        # vez de disolverla en la mediana.
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("scores", score_pipe, SCORE_FEATURES),
        ("dense", dense_pipe, DENSE_NUMERIC),
        ("log", log_pipe, LOG_FEATURES),
        ("cat", cat_pipe, CATEGORICAL_FEATURES),
    ], remainder="drop", verbose_feature_names_out=True)


def build_reclass_preprocessor() -> ColumnTransformer:
    """`build_preprocessor` más `review_stars` (ADR 008).

    `review_stars` recibe el mismo tratamiento que el resto de numéricas. En la
    práctica casi siempre está presente, pero se mantiene el patrón defensivo por si
    una release futura no lo trae para alguna variante.
    """
    base = build_preprocessor()
    review_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    transformers = [*base.transformers, ("review_stars", review_pipe, RECLASS_EXTRA_FEATURES)]
    # `sparse_threshold=0` fuerza salida densa: sin esto depende de una heurística
    # de sklearn sensible a la proporción de columnas one-hot, y
    # `HistGradientBoostingClassifier` exige entrada densa.
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=True,
                             sparse_threshold=0)


def build_safe_reclass_preprocessor() -> ColumnTransformer:
    """Preprocesador de la ablación temporalmente segura (ADR 008).

    Solo `consequence` y `review_stars`, las dos features ancladas a la fecha real
    de cada release.
    """
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    review_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer([
        ("cat", cat_pipe, CATEGORICAL_FEATURES),
        ("review_stars", review_pipe, RECLASS_EXTRA_FEATURES),
    ], remainder="drop", verbose_feature_names_out=True, sparse_threshold=0)


def load_split(name: str) -> tuple[pd.DataFrame, pd.Series]:
    """Carga un split GOLD ('train' | 'test') → (X[FEATURE_COLUMNS], y)."""
    df = pd.read_parquet(processed_dir() / f"{name}.parquet")
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].astype(int)
    return X, y
