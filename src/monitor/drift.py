"""Motor estadístico de detección de drift entre dos poblaciones de variantes.

Por columna numérica, contraste de Kolmogorov-Smirnov y PSI; por categórica, solo
PSI. Basta que se cumpla una de las dos condiciones para marcar drift.

La alerta agregada se dispara si la proporción de columnas con drift supera
`monitor.drift_threshold`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.config import ks_alpha, psi_threshold


def _psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index entre dos muestras numéricas."""
    ref, cur = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    if len(ref) == 0 or len(cur) == 0:
        return 0.0
    edges = np.quantile(ref, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    edges = np.unique(edges)
    r = np.histogram(ref, bins=edges)[0] / len(ref)
    c = np.histogram(cur, bins=edges)[0] / len(cur)
    r, c = np.clip(r, 1e-6, None), np.clip(c, 1e-6, None)
    return float(np.sum((c - r) * np.log(c / r)))


def _psi_categorical(ref: pd.Series, cur: pd.Series) -> float:
    cats = sorted(set(ref.dropna()) | set(cur.dropna()))
    r = ref.value_counts(normalize=True).reindex(cats).fillna(0).to_numpy()
    c = cur.value_counts(normalize=True).reindex(cats).fillna(0).to_numpy()
    r, c = np.clip(r, 1e-6, None), np.clip(c, 1e-6, None)
    return float(np.sum((c - r) * np.log(c / r)))


def compute_drift(reference: pd.DataFrame, current: pd.DataFrame,
                  numeric: list[str], categorical: list[str]) -> pd.DataFrame:
    """Devuelve un DataFrame por feature con estadístico, PSI y flag de drift."""
    # Los dos umbrales se resuelven contra config.yaml en cada llamada, igual que
    # las rutas: ningún valor que gobierne una decisión vive como literal aquí.
    # Referencia habitual del PSI: <0.1 sin cambio, 0.1-0.2 moderado, >0.2 drift.
    psi_max, alpha = psi_threshold(), ks_alpha()
    rows = []
    for col in numeric:
        ref, cur = reference[col].to_numpy(float), current[col].to_numpy(float)
        ks = ks_2samp(ref[~np.isnan(ref)], cur[~np.isnan(cur)])
        psi = _psi(ref, cur)
        drift = bool(ks.pvalue < alpha or psi > psi_max)
        rows.append({"feature": col, "type": "numeric",
                     "ks_stat": round(float(ks.statistic), 4),
                     "p_value": round(float(ks.pvalue), 6),
                     "psi": round(psi, 4), "drift": drift})
    for col in categorical:
        psi = _psi_categorical(reference[col], current[col])
        rows.append({"feature": col, "type": "categorical",
                     "ks_stat": None, "p_value": None,
                     "psi": round(psi, 4), "drift": bool(psi > psi_max)})
    return pd.DataFrame(rows)


def drift_summary(drift_table: pd.DataFrame, threshold: float) -> dict:
    """Resume el drift y decide la alerta según el umbral de proporción."""
    n = len(drift_table)
    n_drift = int(drift_table["drift"].sum())
    share = n_drift / n if n else 0.0
    return {
        "n_features": n,
        "n_drifted": n_drift,
        "share_drifted": round(share, 4),
        "threshold": threshold,
        "alert": bool(share >= threshold),
        "drifted_features": drift_table.loc[drift_table["drift"], "feature"].tolist(),
    }
