"""Preprocesamiento y feature engineering para el clasificador de patogenicidad.

Construye un `ColumnTransformer` de scikit-learn reutilizable por todos los
modelos (garantiza el MISMO preprocesamiento en entrenamiento y en serving).

Decisiones de diseño:
  * `gnomad_af` es fuertemente asimétrica (frecuencias muy pequeñas) → log10.
  * SIFT/PolyPhen/REVEL/AlphaMissense son NaN fuera de *missense* (como en
    dbNSFP real) → imputación por mediana + **indicador de ausencia**
    (`add_indicator`), porque la propia ausencia es informativa (correlaciona
    con el tipo de consecuencia).
  * CADD/GERP++/phyloP (`DENSE_NUMERIC`): pese al nombre del grupo, con
    ClinVar real (dominado por variantes no codificantes) su tasa de nulos es
    tan alta como la del grupo anterior (~75-76%, medido en
    `data/processed/train.parquet` — dbNSFP, y por tanto myvariant.info, solo
    cubre variantes no-sinónimas por diseño, ver ADR 007/§2 de
    la revisión técnica del proyecto). Llevan **el mismo indicador de
    ausencia** que `scores`, no una imputación simple: el supuesto original
    ("numéricas prácticamente siempre presentes") solo era cierto con el
    generador sintético de dbNSFP, no con datos reales.
  * `consequence` (categórica) → one-hot con `handle_unknown="ignore"` para no
    romper en serving ante categorías no vistas.
  * `gene` se excluye a propósito: alta cardinalidad y riesgo de fuga/overfitting.

El preprocesador va DENTRO de un Pipeline junto al clasificador, de modo que el
objeto registrado en MLflow es autocontenido (preprocesa la variante de entrada).
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

# Estado de revisión de ClinVar (0-4 estrellas, `src/annotate/schema.py`),
# usado SOLO por el modelo de reclasificación (potencial de reclasificación,
# `src/train/train_reclass.py`),
# no por los tres modelos (ver ADR 008): a diferencia del resto de `FEATURE_COLUMNS`
# (recalculadas "hoy" vía myvariant.info, con riesgo de fuga temporal hacia
# el pasado), `review_stars` se lee del VCF fechado de cada release y es un
# predictor con respaldo en la literatura de reclasificación de VUS (más
# revisión/submitters -> más probable que se resuelva). No se añade a los tres modelos
# para no alterar esos modelos ya validados fuera del alcance de este cambio,
# y porque para el modelo de patogenicidad (predecir la propia etiqueta) el estado de revisión está
# confundido con la certeza de esa etiqueta -- un argumento distinto, no de
# fuga temporal sino casi definicional (ver ADR 008).
RECLASS_EXTRA_FEATURES = ["review_stars"]
RECLASS_FEATURE_COLUMNS = FEATURE_COLUMNS + RECLASS_EXTRA_FEATURES

# Subconjunto "temporalmente seguro" para la ablación de leakage del modelo de potencial de
# reclasificación (ADR
# 008): `consequence` y `review_stars` se leen directamente del VCF fechado
# de cada release, sin la re-consulta "de hoy" a myvariant.info que sí afecta
# a CADD/REVEL/AlphaMissense/gnomAD/SIFT/PolyPhen/GERP/phyloP. Comparar el
# modelo completo contra este subconjunto cuantifica cuánta señal del modelo de potencial de
# reclasificación
# depende de features potencialmente contaminadas con información posterior
# a la release de entrenamiento.
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
        # `add_indicator=True`: la ausencia de gnomAD no se puede asumir sin más
        # como rareza poblacional real -- también puede deberse a un fallo de
        # cobertura de la fuente de anotación (revisión posterior del proyecto, hallazgo
        # #15). Conservar el indicador aparte del valor imputado permite que el
        # modelo, y un análisis posterior, distingan ambas señales en vez de
        # mezclarlas en un único valor de mediana.
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
    """Preprocesador del modelo de reclasificación: `build_preprocessor` más
    `review_stars` (ADR 008).

    `review_stars` recibe el mismo tratamiento que `DENSE_NUMERIC` (imputación
    por mediana + indicador de ausencia + escalado): en la práctica casi
    siempre está presente (CLNREVSTAT es un campo estándar del VCF de
    ClinVar), pero se mantiene el mismo patrón defensivo que el resto de
    numéricas por si una release futura careciera de él para alguna variante.
    """
    base = build_preprocessor()
    review_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    transformers = [*base.transformers, ("review_stars", review_pipe, RECLASS_EXTRA_FEATURES)]
    # `sparse_threshold=0`: fuerza salida siempre densa. Sin esto, la salida
    # (dispersa o densa) depende de una heurística de densidad de sklearn que
    # varía con la proporción de columnas one-hot frente a numéricas -- y
    # `HistGradientBoostingClassifier` exige entrada densa (detectado al
    # ejecutar la ablación de `build_safe_reclass_preprocessor`, con muchas
    # menos columnas numéricas que el preprocesador base, ver ADR 008).
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=True,
                             sparse_threshold=0)


def build_safe_reclass_preprocessor() -> ColumnTransformer:
    """Preprocesador de la ablación "temporalmente segura" del modelo de reclasificación (ADR 008).

    Solo `consequence` (categórica) y `review_stars` (numérica): las dos
    únicas features del modelo de reclasificación ancladas a la fecha real de cada
    release, sin la
    re-consulta "de hoy" a myvariant.info que afecta al resto.
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
