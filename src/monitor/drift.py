"""Detección de drift entre dos poblaciones de variantes.

Motor estadístico propio (sin dependencias pesadas), reutilizable por el informe
de monitorización. Detecta:
  * Drift de features numéricas: test KS de Kolmogorov y Smirnov + PSI.
  * Drift de features categóricas (consecuencia): PSI sobre categorías.
  * Drift de etiqueta/predicción: cambio de prevalencia (concept drift).

Regla de alerta (config `monitor.drift_threshold`): se dispara si la **proporción
de columnas con drift** supera el umbral.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

# PSI: <0.1 sin cambio · 0.1 a 0.2 cambio moderado · >0.2 drift relevante.
PSI_DRIFT = 0.2
KS_ALPHA = 0.05


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
    rows = []
    for col in numeric:
        ref, cur = reference[col].to_numpy(float), current[col].to_numpy(float)
        ks = ks_2samp(ref[~np.isnan(ref)], cur[~np.isnan(cur)])
        psi = _psi(ref, cur)
        drift = bool(ks.pvalue < KS_ALPHA or psi > PSI_DRIFT)
        rows.append({"feature": col, "type": "numeric",
                     "ks_stat": round(float(ks.statistic), 4),
                     "p_value": round(float(ks.pvalue), 6),
                     "psi": round(psi, 4), "drift": drift})
    for col in categorical:
        psi = _psi_categorical(reference[col], current[col])
        rows.append({"feature": col, "type": "categorical",
                     "ks_stat": None, "p_value": None,
                     "psi": round(psi, 4), "drift": bool(psi > PSI_DRIFT)})
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
