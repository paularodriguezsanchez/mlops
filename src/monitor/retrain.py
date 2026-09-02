"""Flujo de reentrenamiento disparado por alerta de drift [esa etapa, OE5].

Gobernanza (ADR 001): en un contexto clínico el reentrenamiento **no es
automático**. La monitorización evalúa el drift y EMITE UNA RECOMENDACIÓN; un
humano (revisor clínico/ML) aprueba antes de promover un nuevo modelo. Por eso:

  * `python -m src.monitor.retrain` → evalúa y RECOMIENDA (dry-run).
  * `python -m src.monitor.retrain --execute` → aprueba y lanza el reentrenamiento.

El reentrenamiento reutiliza `src.train.train.run`, que registra el nuevo modelo
en MLflow y lo deja en stage **Staging** (no Production): la promoción a producción
es un segundo gate humano.
"""
from __future__ import annotations

import argparse

from src.monitor.drift_report import run as run_drift


def decide(summary: dict) -> dict:
    """Traduce el resumen de drift en una recomendación accionable."""
    cov = summary["covariate_drift"]["alert"]
    concept = summary["concept_drift"]["alert"]
    reasons = []
    if cov:
        reasons.append("drift de covariables por encima del umbral")
    if concept:
        n = summary["concept_drift"]["n_reclassified"]
        acc = summary["concept_drift"].get("model_accuracy_on_reclassified")
        reasons.append(f"concept drift: {n} variantes reclasificadas"
                       + (f", acierto del modelo en ellas={acc}" if acc is not None else ""))
    return {"retrain_recommended": bool(cov or concept), "reasons": reasons}


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
        print("\n[dry-run] No se reentrena. Ejecuta con --execute para aprobar "
              "el reentrenamiento (gate humano, gobernanza clínica).")
        return {"decision": decision, "retrained": False}

    print("\n[aprobado] Lanzando reentrenamiento (nuevo modelo -> stage Staging)...")
    from src.train.train import run as train_run
    train_run()
    return {"decision": decision, "retrained": True}


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Flujo de reentrenamiento por drift.")
    p.add_argument("--execute", action="store_true",
                   help="Aprueba y ejecuta el reentrenamiento (por defecto: dry-run).")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    run(execute=args.execute)
