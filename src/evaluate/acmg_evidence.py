"""Traduce contribuciones SHAP a evidencia heurística tipo ACMG/AMP (ADR 007 §5).

Los criterios ACMG/AMP (Richards et al. 2015) son el marco que usan de
verdad los clínicos para clasificar variantes: PVS1 (pérdida de función),
PP3/BP4 (evidencia computacional), PM2 (rara en población), BA1/BS1 (demasiado
común para ser causante de enfermedad rara), etc. Este módulo NO calcula una
clasificación ACMG (eso exige segregación familiar, evidencia funcional
directa, revisión experta — datos que este proyecto no tiene, ver ADR 007).
Lo que sí hace: toma la contribución SHAP de cada feature (ya calculada por
`src/evaluate/explain.py`) para una variante concreta y la traduce a un
lenguaje de evidencia que un clínico reconoce, con la etiqueta "-like" para
dejar explícito que es una heurística, no una clasificación ACMG certificada.

Uso:
    from src.evaluate.acmg_evidence import explain_variant_acmg
    evidence = explain_variant_acmg(pipe, X_background, X_row)
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

    ClinVar real antepone el término Sequence Ontology al campo `MC=`
    (p. ej. `SO:0001587|nonsense`, `SO:0001575|splice_donor_variant`; ver
    `src/annotate/annotate.py`); el generador sintético (ADR 005, solo
    fixture de tests) usa el término plano sin prefijo. Sin esta
    normalización, `_LOF_CONSEQUENCES` nunca coincidía con ningún valor real
    de ClinVar (0/3.504 en `data/processed/train.parquet`, bug detectado en
    la revisión técnica del 2026-08-05, la revisión técnica del proyecto
    §6) — la etiqueta PVS1-like, la de mayor fuerza clínica del catálogo,
    era código muerto frente a datos reales. `nonsense` es además el
    término real que usa ClinVar para pérdida de función por codón de
    parada prematuro (VEP/Ensembl usa `stop_gained` para el mismo concepto;
    se aceptan ambos).
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
    # Coherente: el score dice "dañino" Y el modelo lo usa para subir el
    # riesgo (o al revés, "no dañino" y lo usa para bajarlo). Si son
    # incoherentes (p. ej. score dañino pero SHAP negativo), no se reporta
    # como evidencia clara — es una interacción más compleja del modelo.
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
    """Evidencia tipo ACMG por fila de `X_explain`, ordenada por |SHAP| descendente.

    Devuelve `(sample, evidence)`: `sample` es el DataFrame de las variantes
    realmente explicadas (con las columnas clave si `X_explain` las traía,
    más las de features) y `evidence[i]` es la lista de evidencia de
    `sample.iloc[i]` — **usar `sample`, no `X_explain`, para identificar cada
    variante**: `compute_shap_values` muestrea internamente (para acotar
    coste) y puede reordenar filas, así que el orden de salida no tiene por
    qué coincidir con el de `X_explain` de entrada.

    Nota de coste: SHAP evalúa el pipeline muchas veces por instancia; para
    explicar muchas variantes de golpe, pasar todas en un único `X_explain`
    (una sola llamada), no una por variante.
    """
    key_cols = [c for c in ("chrom", "pos", "ref", "alt") if c in X_explain.columns]
    feature_cols = [c for c in X_explain.columns if c not in key_cols]

    # `compute_shap_values` conserva el índice original de `X_explain` en la
    # muestra devuelta (no lo resetea) precisamente para poder recuperar la
    # identidad de cada fila con `.loc[sample_raw.index]` — sin depender de
    # un merge por valor, que sería ambiguo si dos variantes comparten los
    # mismos valores de features.
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
