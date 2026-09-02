"""Definición única de "resuelta" y "VUS", compartida por entrenamiento y monitor.

`train_reclass.py` entrena un modelo para predecir la reclasificación y
`drift_report.py` la detecta a posteriori: sin una regla común, ambos módulos
divergen y producen recuentos distintos para la misma transición.
"""
from __future__ import annotations

import pandas as pd

POSITIVE_LABELS = ("Pathogenic", "Likely_pathogenic", "Pathogenic/Likely_pathogenic")
NEGATIVE_LABELS = ("Benign", "Likely_benign", "Benign/Likely_benign")
RESOLVED_LABELS = POSITIVE_LABELS + NEGATIVE_LABELS

# VUS es estrictamente el término oficial de ClinVar, no "cualquier cosa sin
# veredicto": mezclar conflictivas y not_provided crea dos poblaciones distintas
# bajo el mismo nombre.
VUS_LABEL = "Uncertain_significance"


def is_resolved(clnsig: pd.Series) -> pd.Series:
    """True si `clnsig` ya tiene un veredicto resuelto (Patogénica/Benigna, cualquier grado)."""
    return clnsig.isin(RESOLVED_LABELS)


def is_vus(clnsig: pd.Series) -> pd.Series:
    """True si `clnsig` es estrictamente "Uncertain_significance".

    Las conflictivas, `not_provided` y demás categorías ambiguas se tratan como
    excluidas en `build_dataset.binarize_target`, no como VUS.
    """
    return clnsig == VUS_LABEL
