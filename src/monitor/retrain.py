"""Reentrenamiento gobernado, disparado por alerta de deriva (ADR 001).

En un contexto clínico promover un modelo automáticamente sería inaceptable: el
sistema recomienda y una persona aprueba. Sin `--execute` solo recomienda.

El modelo reentrenado entra en stage Staging, nunca directo a Production: la
promoción es un segundo gate humano, independiente de este módulo.
"""
from __future__ import annotations

import argparse

from src.monitor.drift_report import run as run_drift


def decide(summary: dict) -> dict:
    """Traduce el resumen de drift en una recomendación accionable."""
    cov = summary["covariate_drift"]["alert"]
    reclass = summary["reclassification_drift"]["alert"]
    reasons = []
    if cov:
        reasons.append("drift de covariables por encima del umbral")
    if reclass:
        n = summary["reclassification_drift"]["n_reclassified"]
        acc = summary["reclassification_drift"].get("model_accuracy_on_reclassified")
        reasons.append(f"deriva de reclasificación: {n} variantes reclasificadas"
                       + (f", acierto del modelo en ellas={acc}" if acc is not None else ""))
    return {"retrain_recommended": bool(cov or reclass), "reasons": reasons}


def run(execute: bool = False) -> dict:
    summary = run_drift()
    decision = decide(summary)
    print("\n=== Recomendación de reentrenamiento ===")
    if not decision["retrain_recommended"]:
        print("Sin drift relevante: no se recomienda reentrenar.")
        return {"decision": decision, "retrained": False}

    print("Reentrenamiento RECOMENDADO por:")
    for r in decision["reasons"]:
        print(f"  * {r}")

    if not execute:
        print("\n[simulación] No se reentrena. Ejecuta con --execute para aprobar "
              "el reentrenamiento.")
        return {"decision": decision, "retrained": False}

    print("\n[aprobado] Lanzando reentrenamiento (nuevo modelo -> stage Staging)...")
    from src.train.train import run as train_run
    train_run()
    return {"decision": decision, "retrained": True}


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Reentrenamiento gobernado por deriva.")
    p.add_argument("--execute", action="store_true",
                   help="Aprueba y ejecuta el reentrenamiento; por defecto solo recomienda.")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    run(execute=args.execute)
