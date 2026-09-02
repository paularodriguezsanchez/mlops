"""Informe de monitorización y detección de drift [OE5, esa etapa+esa etapa].

Compara la release de referencia (ClinVar 2023-12) con la actual (2025-06) y produce:
  1. **Drift de covariables** (features): tabla propia (KS/PSI) + informe HTML de Evidently.
  2. **Concept drift**: variantes reclasificadas (VUS→resuelta) y su impacto en el
     rendimiento del modelo entrenado con las etiquetas antiguas → señal real de OE5.
  3. **Stress test** (perturbación controlada, plan R6): valida que la ALERTA por
     umbral se dispara cuando sí hay drift de features.

Salida:
  reports/monitoring/evidently_drift.html (informe citable)
  reports/monitoring/drift_summary.json (decisión de alerta)
  reports/monitoring/drift_features.csv (tabla por feature)

Uso:
    python -m src.monitor.drift_report
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, get_seed, interim_dir, load_config
from src.features.preprocess import (
    CATEGORICAL_FEATURES,
    DENSE_NUMERIC,
    FEATURE_COLUMNS,
    LOG_FEATURES,
    SCORE_FEATURES,
)
from src.features.reclassification import POSITIVE_LABELS, is_resolved, is_vus
from src.monitor.drift import compute_drift, drift_summary

NUMERIC = SCORE_FEATURES + DENSE_NUMERIC + LOG_FEATURES
_KEY = ["chrom", "pos", "ref", "alt"]


def _load(release: str) -> pd.DataFrame:
    return pd.read_parquet(interim_dir() / f"annotated_{release}.parquet")


def _covariate_drift(ref: pd.DataFrame, cur: pd.DataFrame, threshold: float) -> dict:
    table = compute_drift(ref, cur, NUMERIC, CATEGORICAL_FEATURES)
    summary = drift_summary(table, threshold)
    return {"table": table, "summary": summary}


def _concept_drift(ref: pd.DataFrame, cur: pd.DataFrame) -> dict:
    """Reclasificaciones VUS→resuelta y rendimiento del modelo en ellas.

    Usa la misma definición estricta de VUS que `build_dataset.py`/el modelo de potencial de
    reclasificación
    (`is_vus`, únicamente "Uncertain_significance"), en vez del literal
    hardcodeado de forma independiente que tenía antes este módulo -- eran
    ya la misma cadena, pero la duplicación es la que dejó pasar sin
    detectar la discrepancia real: la población de "VUS" del modelo de reclasificación
    (definida en
    `build_dataset.py`) incluía además clasificaciones conflictivas, no
    solo VUS estricta. `n_shared`
    sigue siendo el total de variantes compartidas entre ambas releases
    (cualquier `clnsig`), no solo VUS: es el denominador de la tasa de
    alerta, una magnitud distinta y deliberadamente más amplia que la
    población de VUS del modelo de reclasificación.
    """
    m = ref[_KEY + ["clnsig"]].merge(cur[_KEY + ["clnsig"]], on=_KEY,
                                     suffixes=("_old", "_new"))
    reclass = m[is_vus(m["clnsig_old"]) & is_resolved(m["clnsig_new"])].copy()
    out = {"n_shared": int(len(m)), "n_reclassified": int(len(reclass)),
           "n_new_variants": int(len(cur) - len(m))}
    # Impacto en el modelo: predicción vs nueva etiqueta de las reclasificadas.
    try:
        import mlflow.sklearn
        model = mlflow.sklearn.load_model(str(PROJECT_ROOT / "models" / "best_model"))
        feats = cur.merge(reclass[_KEY], on=_KEY)
        y_new = feats["clnsig"].isin(POSITIVE_LABELS).astype(int)
        prob = model.predict_proba(feats[FEATURE_COLUMNS])[:, 1]
        acc = float(((prob >= 0.5).astype(int) == y_new).mean())
        out["model_accuracy_on_reclassified"] = round(acc, 4)
        out["interpretation"] = (
            "El modelo se entrenó cuando estas variantes eran VUS (excluidas). "
            "Su acierto frente a la nueva verdad clínica mide el impacto del "
            "concept drift; por debajo de lo esperado justifica reentrenar.")
    except Exception as exc:  # noqa: BLE001
        out["model_accuracy_on_reclassified"] = None
        out["note"] = f"modelo no disponible: {exc}"
    # Alerta de concept drift: proporción de variantes compartidas reclasificadas
    # por encima de un umbral (no "cualquier" reclasificación). Ver la revisión interna del
    # proyecto.
    share = len(reclass) / len(m) if len(m) else 0.0
    out["share_reclassified"] = round(share, 4)
    out["reclassified_threshold"] = 0.02
    out["alert"] = bool(share >= 0.02)
    return out


def _stress_test(cur: pd.DataFrame, threshold: float, seed: int) -> dict:
    """Perturbación controlada (plan R6): desplaza features para validar la alerta."""
    rng = np.random.default_rng(seed)
    perturbed = cur.copy()
    perturbed["cadd_phred"] = np.clip(perturbed["cadd_phred"] + 8, 0, 60)
    perturbed["gnomad_af"] = np.clip(perturbed["gnomad_af"] * 0.1, 1e-7, 1)
    perturbed["gerp_rs"] = np.clip(perturbed["gerp_rs"] + rng.normal(2, 0.5, len(cur)), -15, 7)
    table = compute_drift(cur, perturbed, NUMERIC, CATEGORICAL_FEATURES)
    summary = drift_summary(table, threshold)
    summary["description"] = ("Perturbación controlada (+8 CADD, AF×0.1, +GERP) "
                              "para verificar que la alerta por umbral se dispara.")
    return summary


def _evidently_html(ref: pd.DataFrame, cur: pd.DataFrame, path: Path) -> bool:
    """Genera el informe HTML de Evidently. Devuelve True si tuvo éxito."""
    try:
        from evidently import DataDefinition, Dataset, Report
        from evidently.presets import DataDriftPreset
        cols = NUMERIC + CATEGORICAL_FEATURES
        dd = DataDefinition(numerical_columns=NUMERIC,
                            categorical_columns=CATEGORICAL_FEATURES)
        rds = Dataset.from_pandas(ref[cols], data_definition=dd)
        cds = Dataset.from_pandas(cur[cols], data_definition=dd)
        snapshot = Report([DataDriftPreset()]).run(reference_data=rds, current_data=cds)
        path.parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(str(path))
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[aviso] Evidently no disponible ({exc}); informe HTML omitido.")
        return False


def run() -> dict:
    cfg = load_config()
    threshold = float(cfg["monitor"]["drift_threshold"])
    ref = _load(cfg["data"]["clinvar_train_release"])
    cur = _load(cfg["data"]["clinvar_test_release"])
    out_dir = PROJECT_ROOT / "reports" / "monitoring"
    out_dir.mkdir(parents=True, exist_ok=True)

    cov = _covariate_drift(ref, cur, threshold)
    concept = _concept_drift(ref, cur)
    stress = _stress_test(cur, threshold, get_seed())
    evidently_ok = _evidently_html(ref, cur, out_dir / "evidently_drift.html")

    cov["table"].to_csv(out_dir / "drift_features.csv", index=False)
    summary = {
        "reference_release": cfg["data"]["clinvar_train_release"],
        "current_release": cfg["data"]["clinvar_test_release"],
        "covariate_drift": cov["summary"],
        "concept_drift": concept,
        "stress_test": stress,
        "evidently_report": "evidently_drift.html" if evidently_ok else None,
        "overall_alert": bool(cov["summary"]["alert"] or concept["alert"]),
    }
    (out_dir / "drift_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Covariate drift: {cov['summary']['n_drifted']}/{cov['summary']['n_features']} "
          f"features (alert={cov['summary']['alert']})")
    print(f"Concept drift: {concept['n_reclassified']} reclasificadas, "
          f"acc modelo en ellas={concept.get('model_accuracy_on_reclassified')} "
          f"(alert={concept['alert']})")
    print(f"Stress test (perturbación): alert={stress['alert']} "
          f"({stress['n_drifted']}/{stress['n_features']} features)")
    print(f">> ALERTA GLOBAL: {summary['overall_alert']}")
    return summary


if __name__ == "__main__":
    run()
