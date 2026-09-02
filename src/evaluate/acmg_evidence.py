"""Traduce contribuciones SHAP a evidencia heurística tipo ACMG/AMP (ADR 007 §5).

No calcula una clasificación ACMG: eso exige segregación familiar, evidencia
funcional y revisión de comité, datos que este proyecto no tiene. Lo que hace es
traducir la contribución SHAP de cada feature a un lenguaje que un genetista
reconoce -PVS1, PP3/BP4, PM2, BA1-, siempre con el sufijo "-like" para que no se
confunda con una clasificación certificada.
"""
from __future__ import annotations

import pandas as pd

from src.evaluate.explain import compute_shap_values

_LOF_CONSEQUENCES = {
    "stop_gained", "nonsense", "frameshift_variant",
    "splice_donor_variant", "splice_acceptor_variant",
}


def _lof_term(consequence) -> str | None:
    """Normaliza `consequence` para compararlo contra `_LOF_CONSEQUENCES`.

    ClinVar antepone el término Sequence Ontology (`SO:0001587|nonsense`), mientras
    que el generador usa el término plano. Sin normalizar, PVS1-like -la regla de
    mayor fuerza clínica- no coincide con ningún valor real. Se aceptan `nonsense`
    (término de ClinVar) y `stop_gained` (término de VEP) como sinónimos.
    """
    if not isinstance(consequence, str):
        return None
    return consequence.rsplit("|", 1)[-1]

# Cada regla: (feature, condición sobre el valor crudo, tag, plantilla del texto).
# `high`/`low` se evalúan sobre el valor crudo de la feature; la DIRECCIÓN del
# SHAP decide si la regla aplica hacia patogénica o hacia benigna.
_THRESHOLDS = {
    "cadd_phred": {"damaging": lambda v: v is not None and v >= 20},
    "revel_score": {"damaging": lambda v: v is not None and v >= 0.7},
    "alphamissense_score": {"damaging": lambda v: v is not None and v >= 0.7},
    "polyphen_score": {"damaging": lambda v: v is not None and v >= 0.85},
    "sift_score": {"damaging": lambda v: v is not None and v <= 0.05},
    "gerp_rs": {"damaging": lambda v: v is not None and v >= 4.0},
    "phylop": {"damaging": lambda v: v is not None and v >= 5.0},
}

_EVIDENCE_LABELS = {
    "cadd_phred": ("CADD combinado sugiere efecto deletéreo", "CADD no sugiere efecto deletéreo"),
    "revel_score": ("REVEL sugiere patogenicidad (missense)", "REVEL no sugiere patogenicidad"),
    "alphamissense_score": (
        "AlphaMissense sugiere patogenicidad (basado en estructura, AlphaFold)",
        "AlphaMissense no sugiere patogenicidad"),
    "polyphen_score": ("PolyPhen-2 predice \"probably damaging\"", "PolyPhen-2 predice benigno"),
    "sift_score": ("SIFT predice \"deleterious\" (intolerante)", "SIFT predice \"tolerated\""),
    "gerp_rs": ("posición evolutivamente muy conservada (GERP++)", "posición poco conservada"),
    "phylop": ("posición evolutivamente muy conservada (phyloP)", "posición poco conservada"),
}


def _score_evidence(feature: str, value, shap_value: float) -> dict | None:
    rule = _THRESHOLDS.get(feature)
    if rule is None or value is None or pd.isna(value):
        return None
    damaging = rule["damaging"](value)
    pushes_pathogenic = shap_value > 0
    # Solo se reporta evidencia si el valor del score y el signo de SHAP apuntan en
    # la misma dirección. Si no, el modelo está capturando una interacción más
    # compleja y forzar un código unívoco sería inventar precisión.
    if damaging and not pushes_pathogenic:
        return None
    if not damaging and pushes_pathogenic:
        return None
    label_pathogenic, label_benign = _EVIDENCE_LABELS[feature]
    tag = "PP3-like" if pushes_pathogenic else "BP4-like"
    text = label_pathogenic if pushes_pathogenic else label_benign
    return {
        "feature": feature, "value": float(value), "shap": float(shap_value),
        "direction": "patogénica" if pushes_pathogenic else "benigna",
        "acmg_tag": tag, "note": text,
    }


def _population_frequency_evidence(gnomad_af, shap_value: float) -> dict | None:
    if gnomad_af is None or pd.isna(gnomad_af):
        return None
    if gnomad_af < 1e-4 and shap_value > 0:
        return {"feature": "gnomad_af", "value": float(gnomad_af), "shap": float(shap_value),
                "direction": "patogénica", "acmg_tag": "PM2-like",
                "note": "variante rara/ausente en gnomAD (frecuencia poblacional muy baja)"}
    if gnomad_af > 0.01 and shap_value < 0:
        return {"feature": "gnomad_af", "value": float(gnomad_af), "shap": float(shap_value),
                "direction": "benigna", "acmg_tag": "BA1-like",
                "note": "frecuencia poblacional demasiado alta para una enfermedad rara"}
    return None


def _consequence_evidence(consequence, shap_value: float) -> dict | None:
    if _lof_term(consequence) in _LOF_CONSEQUENCES and shap_value > 0:
        return {"feature": "consequence", "value": str(consequence), "shap": float(shap_value),
                "direction": "patogénica", "acmg_tag": "PVS1-like",
                "note": f"consecuencia de pérdida de función predicha ({consequence})"}
    return None


def _evidence_for_row(row_raw: pd.Series, row_shap: pd.Series) -> list[dict]:
    items: list[dict] = []
    for feature in _THRESHOLDS:
        ev = _score_evidence(feature, row_raw.get(feature), row_shap.get(feature, 0.0))
        if ev:
            items.append(ev)
    pop_ev = _population_frequency_evidence(
        row_raw.get("gnomad_af"), row_shap.get("gnomad_af", 0.0))
    if pop_ev:
        items.append(pop_ev)
    cons_ev = _consequence_evidence(row_raw.get("consequence"), row_shap.get("consequence", 0.0))
    if cons_ev:
        items.append(cons_ev)
    return sorted(items, key=lambda e: abs(e["shap"]), reverse=True)


def explain_variant_acmg(
    pipe, X_background: pd.DataFrame, X_explain: pd.DataFrame, seed: int = 42,
) -> tuple[pd.DataFrame, list[list[dict]]]:
    """Evidencia tipo ACMG por fila, ordenada por |SHAP| descendente.

    Devuelve `(sample, evidence)`. Identificar cada variante por `sample`, no por
    `X_explain`: `compute_shap_values` muestrea internamente y puede reordenar filas.

    SHAP evalúa el pipeline muchas veces por instancia, así que conviene pasar todas
    las variantes en una sola llamada.
    """
    key_cols = [c for c in ("chrom", "pos", "ref", "alt") if c in X_explain.columns]
    feature_cols = [c for c in X_explain.columns if c not in key_cols]

    # El índice original se conserva para recuperar la identidad de cada fila con
    # `.loc`, sin un merge por valor que sería ambiguo entre variantes con las
    # mismas features.
    shap_values, sample_raw, _ = compute_shap_values(
        pipe, X_background[feature_cols], X_explain[feature_cols],
        seed=seed, max_explain=len(X_explain))

    sample = sample_raw
    if key_cols:
        sample = pd.concat([X_explain.loc[sample_raw.index, key_cols], sample_raw], axis=1)

    out: list[list[dict]] = []
    for i in range(len(sample_raw)):
        row_raw = sample_raw.iloc[i]
        row_shap = pd.Series(shap_values.values[i], index=feature_cols)
        out.append(_evidence_for_row(row_raw, row_shap))
    return sample.reset_index(drop=True), out
