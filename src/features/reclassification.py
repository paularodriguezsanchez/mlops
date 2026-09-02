"""Regla compartida de reclasificación VUS -> resuelta, usada por el modelo de
reclasificación y por el monitor de deriva.

Una única definición de "¿esta variante se reclasificó?" para que
`src/train/train_reclass.py` (entrena un modelo a predecirla) y
`src/monitor/drift_report.py` (la detecta a posteriori, tras una nueva
release de ClinVar) no diverjan si la regla cambia -- antes cada módulo
calculaba, con código casi idéntico pero independiente, el mismo merge y el
mismo filtro (revisión técnica del proyecto).
"""
from __future__ import annotations

import pandas as pd

POSITIVE_LABELS = ("Pathogenic", "Likely_pathogenic", "Pathogenic/Likely_pathogenic")
NEGATIVE_LABELS = ("Benign", "Likely_benign", "Benign/Likely_benign")
RESOLVED_LABELS = POSITIVE_LABELS + NEGATIVE_LABELS

# Definición estricta de VUS (una revisión posterior del proyecto/#12): antes
# `build_dataset.py` reservaba como "VUS" cualquier `clnsig` no positivo/negativo
# (incluía "Conflicting_interpretations_of_pathogenicity", "not_provided", etc.),
# mientras que `drift_report.py` filtraba de forma independiente el literal
# "Uncertain_significance" -- dos poblaciones distintas sin documentar, origen de
# la discrepancia 67 vs 54 reclasificaciones citada en la memoria. Ahora ambos
# módulos usan esta misma constante como única fuente de verdad.
VUS_LABEL = "Uncertain_significance"


def is_resolved(clnsig: pd.Series) -> pd.Series:
    """True si `clnsig` ya tiene un veredicto resuelto (Patogénica/Benigna, cualquier grado)."""
    return clnsig.isin(RESOLVED_LABELS)


def is_vus(clnsig: pd.Series) -> pd.Series:
    """True si `clnsig` es estrictamente "Uncertain_significance" (definición ClinVar).

    No incluye clasificaciones conflictivas, "not_provided" u otras categorías
    ambiguas: esas se tratan como excluidas (ver `build_dataset.binarize_target`),
    no como VUS, para no mezclar poblaciones de significado distinto bajo la
    misma etiqueta "VUS" citada en la memoria.
    """
    return clnsig == VUS_LABEL
