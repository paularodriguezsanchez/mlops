"""Tests del preprocesamiento (contrato de features y robustez)."""
import numpy as np
import pandas as pd

from src.features import preprocess as pp


def _sample(n=6):
    return pd.DataFrame({
        "sift_score": [0.1, np.nan, 0.9, np.nan, 0.3, np.nan],
        "polyphen_score": [0.9, np.nan, 0.1, np.nan, 0.7, np.nan],
        "revel_score": [0.8, np.nan, 0.2, np.nan, 0.6, np.nan],
        "alphamissense_score": [0.85, np.nan, 0.15, np.nan, 0.65, np.nan],
        "cadd_phred": [30, 5, 25, 3, 20, 10],
        "gerp_rs": [5, -1, 4, -2, 3, 0],
        "phylop": [7, -1, 6, -3, 5, 0],
        "gnomad_af": [1e-6, 0.1, 1e-5, 0.2, 1e-4, 0.05],
        "consequence": ["missense_variant", "synonymous_variant", "stop_gained",
                        "intron_variant", "missense_variant", "splice_donor_variant"],
    })


def test_transform_sin_nan_y_expande_features():
    X = _sample()
    pre = pp.build_preprocessor()
    Xt = pre.fit_transform(X)
    arr = Xt.toarray() if hasattr(Xt, "toarray") else Xt
    assert not np.isnan(arr).any()          # imputación completa
    assert arr.shape[0] == len(X)
    assert arr.shape[1] > X.shape[1]          # one-hot + indicadores de ausencia


def test_maneja_categoria_no_vista_en_serving():
    pre = pp.build_preprocessor().fit(_sample())
    nuevo = _sample(1).iloc[[0]].copy()
    nuevo.loc[:, "consequence"] = "categoria_desconocida"
    out = pre.transform(nuevo)              # handle_unknown='ignore' -> no lanza
    arr = out.toarray() if hasattr(out, "toarray") else out
    assert arr.shape[0] == 1


def test_feature_columns_coinciden_con_gold():
    assert set(pp.FEATURE_COLUMNS) == set(
        pp.SCORE_FEATURES + pp.DENSE_NUMERIC + pp.LOG_FEATURES + pp.CATEGORICAL_FEATURES)


def _sample_reclass(n=6):
    df = _sample(n)
    df["review_stars"] = [1.0, 2.0, 0.0, 3.0, 1.0, np.nan][:n]
    return df


def test_reclass_feature_columns_extiende_feature_columns_con_review_stars():
    """ADR 008: el modelo de reclasificación usa FEATURE_COLUMNS + review_stars; los tres modelos no
    cambian.
    """
    assert pp.RECLASS_FEATURE_COLUMNS == [*pp.FEATURE_COLUMNS, "review_stars"]
    assert "review_stars" not in pp.FEATURE_COLUMNS


def test_reclass_preprocessor_transforma_sin_nan():
    X = _sample_reclass()
    pre = pp.build_reclass_preprocessor()
    Xt = pre.fit_transform(X[pp.RECLASS_FEATURE_COLUMNS])
    arr = Xt.toarray() if hasattr(Xt, "toarray") else Xt
    assert not np.isnan(arr).any()
    assert arr.shape[0] == len(X)


def test_safe_reclass_preprocessor_solo_usa_consequence_y_review_stars():
    """La ablación de leakage (ADR 008) no debe usar ninguna feature de myvariant.info."""
    X = _sample_reclass()
    pre = pp.build_safe_reclass_preprocessor()
    Xt = pre.fit_transform(X[pp.SAFE_RECLASS_FEATURE_COLUMNS])
    arr = Xt.toarray() if hasattr(Xt, "toarray") else Xt
    assert set(pp.SAFE_RECLASS_FEATURE_COLUMNS) == {"consequence", "review_stars"}
    assert not np.isnan(arr).any()
    assert arr.shape[0] == len(X)
