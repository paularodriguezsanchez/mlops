"""Métricas de evaluación del clasificador de patogenicidad.

Se priorizan métricas robustas al (posible) desbalanceo de clases, conforme a
las convenciones del proyecto: **PR AUC, F1 y ROC AUC**, más la matriz de confusión.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict[str, float]:
    """Calcula el panel de métricas del proyecto a partir de probabilidades."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def bootstrap_pr_auc_ci(y_true, y_prob, n_boot: int = 1000, ci: float = 0.95,
                        seed: int = 42) -> dict[str, float]:
    """Intervalo de confianza del PR AUC por remuestreo bootstrap del holdout.

    Sin necesidad de una release adicional: remuestrea con reemplazo el
    propio conjunto de evaluación `n_boot` veces y recalcula el PR AUC en
    cada remuestreo. `compute_metrics` (y, con ella, la selección del mejor
    algoritmo en `train.py`) solo devuelve un valor puntual; con un holdout
    de tamaño moderado, diferencias pequeñas entre modelos (p. ej. gradient
    boosting vs. random forest) pueden no ser distinguibles del ruido de
    muestreo. Este intervalo permite afirmarlo con rigor en vez de asumirlo
    (revisión técnica del proyecto).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)
    rng = np.random.default_rng(seed)
    scores = np.full(n_boot, np.nan)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        if yt.min() == yt.max():  # remuestreo degenerado (una sola clase): PR AUC no definido
            continue
        scores[i] = average_precision_score(yt, y_prob[idx])
    valid = scores[~np.isnan(scores)]
    lo_q, hi_q = (1 - ci) / 2, 1 - (1 - ci) / 2
    return {
        "pr_auc_ci_low": float(np.quantile(valid, lo_q)) if len(valid) else float("nan"),
        "pr_auc_ci_high": float(np.quantile(valid, hi_q)) if len(valid) else float("nan"),
        "pr_auc_ci_n_boot": int(len(valid)),
    }


def bootstrap_roc_auc_ci(y_true, y_prob, n_boot: int = 1000, ci: float = 0.95,
                         seed: int = 42) -> dict[str, float]:
    """Intervalo de confianza del ROC AUC por remuestreo bootstrap del holdout.

    Revisión posterior del proyecto: el ROC AUC del modelo de reclasificación se citaba
    como
    estimador puntual sin incertidumbre pese al reducido número de positivos
    en el holdout, mientras que el PR AUC sí llevaba IC desde antes. Mismo
    procedimiento que `bootstrap_pr_auc_ci`, aplicado a ROC AUC.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)
    rng = np.random.default_rng(seed)
    scores = np.full(n_boot, np.nan)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        if yt.min() == yt.max():  # remuestreo degenerado (una sola clase): ROC AUC no definido
            continue
        scores[i] = roc_auc_score(yt, y_prob[idx])
    valid = scores[~np.isnan(scores)]
    lo_q, hi_q = (1 - ci) / 2, 1 - (1 - ci) / 2
    return {
        "roc_auc_ci_low": float(np.quantile(valid, lo_q)) if len(valid) else float("nan"),
        "roc_auc_ci_high": float(np.quantile(valid, hi_q)) if len(valid) else float("nan"),
        "roc_auc_ci_n_boot": int(len(valid)),
    }


def bootstrap_pr_auc_diff_ci(y_true, y_prob_a, y_prob_b, n_boot: int = 1000, ci: float = 0.95,
                             seed: int = 42) -> dict[str, float]:
    """IC bootstrap de la DIFERENCIA de PR AUC entre dos puntuaciones sobre el mismo conjunto.

    Revisión posterior del proyecto: comparar dos PR-AUC puntuales (p. ej.
    el modelo ensemble frente a CADD solo) sin cuantificar si la diferencia es
    distinguible del ruido de muestreo no basta para decir "competitivo con".
    Remuestrea las MISMAS filas para `y_prob_a` e `y_prob_b` en cada iteración
    (bootstrap pareado): si el IC de la diferencia no cruza el cero, la
    diferencia es estadísticamente distinguible con esta muestra.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob_a = np.asarray(y_prob_a, dtype=float)
    y_prob_b = np.asarray(y_prob_b, dtype=float)
    n = len(y_true)
    rng = np.random.default_rng(seed)
    diffs = np.full(n_boot, np.nan)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        if yt.min() == yt.max():
            continue
        diffs[i] = (average_precision_score(yt, y_prob_a[idx])
                   - average_precision_score(yt, y_prob_b[idx]))
    valid = diffs[~np.isnan(diffs)]
    lo_q, hi_q = (1 - ci) / 2, 1 - (1 - ci) / 2
    lo = float(np.quantile(valid, lo_q)) if len(valid) else float("nan")
    hi = float(np.quantile(valid, hi_q)) if len(valid) else float("nan")
    return {
        "pr_auc_diff_ci_low": lo, "pr_auc_diff_ci_high": hi,
        "pr_auc_diff_ci_n_boot": int(len(valid)),
        "crosses_zero": bool(lo <= 0 <= hi) if len(valid) else True,
    }


def precision_recall_lift_at_k(y_true, y_prob, ks: tuple[int, ...] = (10, 20, 50, 100)
                               ) -> dict[str, float]:
    """precision@k / recall@k / lift@k: métricas naturales para una cola de revisión.

    Revisión posterior del proyecto: con una prevalencia baja (p. ej. el
    1, 4 % de VUS reclasificadas en el modelo de reclasificación), el PR AUC/ROC AUC
    agregados no dicen
    cuántos aciertos reales contiene la cabeza de la lista ordenada, que es
    justo lo que le importa a un revisor que solo llega a mirar las primeras
    k variantes. `lift@k` compara la precisión en la cabeza frente a la
    prevalencia global (1.0 = igual que ordenar al azar).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)
    n_pos = int(y_true.sum())
    prevalence = n_pos / n if n else 0.0
    order = np.argsort(-y_prob)
    out: dict[str, float] = {}
    for k in ks:
        k_eff = min(k, n)
        if k_eff < 1:
            continue
        top_idx = order[:k_eff]
        hits = int(y_true[top_idx].sum())
        precision_k = hits / k_eff
        recall_k = hits / n_pos if n_pos else float("nan")
        lift_k = precision_k / prevalence if prevalence else float("nan")
        out[f"precision_at_{k}"] = round(precision_k, 4)
        out[f"recall_at_{k}"] = round(recall_k, 4)
        out[f"lift_at_{k}"] = round(lift_k, 4)
    return out


def calibration_report(y_true, y_prob, n_bins: int = 10) -> dict:
    """Brier score + tabla de calibración por deciles de probabilidad predicha.

    Revisión posterior del proyecto: el proyecto habla de "probabilidad
    de reclasificación" sin haber estudiado nunca si esa probabilidad está
    calibrada (si de las VUS con score ~0.3, de verdad se reclasifica
    alrededor del 30 %). Con pocos positivos el Brier score global puede ser
    engañosamente bajo por la propia prevalencia baja; la tabla por bins
    hace explícito dónde hay pocos datos para confiar en la calibración.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    brier = float(brier_score_loss(y_true, y_prob))
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(y_prob, edges[1:-1]), 0, n_bins - 1)
    bins = []
    for b in range(n_bins):
        mask = bin_idx == b
        n_b = int(mask.sum())
        bins.append({
            "bin_low": round(float(edges[b]), 3), "bin_high": round(float(edges[b + 1]), 3),
            "n": n_b,
            "mean_predicted": round(float(y_prob[mask].mean()), 4) if n_b else None,
            "observed_rate": round(float(y_true[mask].mean()), 4) if n_b else None,
        })
    return {"brier_score": brier, "n_bins": n_bins, "bins": bins}


def export_pr_curve(y_true, y_prob, path: Path) -> Path:
    """Exporta la curva precisión-cobertura completa a CSV (revisión posterior del proyecto)."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    path.parent.mkdir(parents=True, exist_ok=True)
    # `thresholds` tiene un elemento menos que precision/recall (sklearn no define
    # un umbral para el último punto, recall=0); se rellena con NaN para alinear.
    thr = np.append(thresholds, np.nan)
    import pandas as pd
    pd.DataFrame({"precision": precision, "recall": recall, "threshold": thr}).to_csv(
        path, index=False)
    return path


def export_roc_curve(y_true, y_prob, path: Path) -> Path:
    """Exporta la curva ROC completa a CSV (revisión posterior del proyecto)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    path.parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds}).to_csv(path, index=False)
    return path


def save_confusion_matrix(y_true, y_prob, path: Path, threshold: float = 0.5,
                          title: str = "Matriz de confusión",
                          labels: tuple[str, str] = ("Benigna", "Patogénica")) -> Path:
    """Guarda la matriz de confusión como PNG y devuelve la ruta.

    `labels` por defecto asume una tarea de patogenicidad (Benigna/Patogénica);
    otras tareas binarias (p. ej. reclasificación sí/no) deben pasar sus
    propias etiquetas explícitas (una revisión posterior del proyecto: la
    matriz del modelo de reclasificación usaba por defecto "Benigna", "Patogénica" para
    una tarea que
    en realidad es "No reclasificada", "Reclasificada", una tarea distinta).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = (np.asarray(y_prob, dtype=float) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], list(labels))
    ax.set_yticks([0, 1], list(labels))
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
