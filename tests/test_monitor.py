"""Tests de la detección de drift ."""
import numpy as np
import pandas as pd

from src.monitor.drift import compute_drift, drift_summary


def _frame(cadd_mean, n=500, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "sift_score": rng.random(n), "polyphen_score": rng.random(n),
        "revel_score": rng.random(n), "cadd_phred": rng.normal(cadd_mean, 3, n),
        "gerp_rs": rng.normal(2, 1, n), "phylop": rng.normal(3, 1, n),
        "gnomad_af": 10 ** rng.uniform(-6, -1, n),
        "consequence": rng.choice(["missense_variant", "stop_gained"], n),
    })


NUM = ["sift_score", "polyphen_score", "revel_score", "cadd_phred",
       "gerp_rs", "phylop", "gnomad_af"]
CAT = ["consequence"]


def test_sin_drift_no_alerta():
    ref, cur = _frame(20, seed=1), _frame(20, seed=2)
    table = compute_drift(ref, cur, NUM, CAT)
    s = drift_summary(table, threshold=0.15)
    assert s["alert"] is False


def test_con_drift_dispara_alerta():
    ref, cur = _frame(20, seed=1), _frame(35, seed=2)   # CADD desplazado
    table = compute_drift(ref, cur, NUM, CAT)
    s = drift_summary(table, threshold=0.1)   # 1/8=0.125 supera el umbral
    assert s["alert"] is True
    assert "cadd_phred" in s["drifted_features"]
