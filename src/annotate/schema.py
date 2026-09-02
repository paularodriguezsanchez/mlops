"""Contrato de datos de variantes anotadas (capa SILVER).

Define el esquema esperado y una validación explícita. Cumplir un contrato de
datos es un requisito de calidad del TFM (T3): cualquier cambio de esquema
aguas arriba falla de forma ruidosa en vez de propagar datos corruptos.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ALLOWED_CHROMS = {str(c) for c in range(1, 23)} | {"X", "Y", "MT"}
BASES = set("ACGT")

# columna → (dtype_kind, permite_nulos)
KEY_COLUMNS = ["chrom", "pos", "ref", "alt"]

# Features numéricas con su rango plausible (validación de contrato de datos).
NUMERIC_RANGES = {
    # CADD phred (Rentzsch et al. 2019, GRCh38 v1.6+): el escalado phred de
    # -10*log10(rango/total) permite valores >60 para las variantes más extremas
    # (confirmado con datos reales 2026-08-01: nonsense en TTN/NPHP3 hasta 73).
    # El límite anterior (60) venía calibrado sobre el techo arbitrario del
    # generador sintético (`np.clip(..., 0, 45)`, ADR 005), no sobre el rango
    # real de la métrica — ver la revisión interna del proyecto.
    "cadd_phred": (0.0, 99.0),
    "sift_score": (0.0, 1.0),
    "polyphen_score": (0.0, 1.0),
    "revel_score": (0.0, 1.0),
    "alphamissense_score": (0.0, 1.0),
    "gerp_rs": (-15.0, 7.0),
    "phylop": (-25.0, 12.0),
    "gnomad_af": (0.0, 1.0),
}

CATEGORICAL_COLUMNS = ["gene", "consequence", "clnsig"]

# Estado de revisión de ClinVar (CLNREVSTAT del VCF), y su traducción a la
# escala oficial de "gold stars" (0-4, https://www.ncbi.nlm.nih.gov/clinvar/
# docs/review_status/). Añadido en ADR 008 como feature del modelo de reclasificación
# (potencial de
# reclasificación): a diferencia de CADD/REVEL/AlphaMissense/gnomAD (anotadas
# "hoy" vía myvariant.info, ver leakage temporal en ADR 008), `review_status`
# se lee directamente del VCF fechado de cada release, así que su valor en la
# release t0 es el que existía realmente en t0 -- temporalmente seguro.
#
# ClinVar cambió su vocabulario de CLNREVSTAT entre las dos releases
# (verificado por inspección directa de ambos VCF, no supuesto): 2023-12 usa
# "…_interpretations"/"no_assertion_provided", 2025-06 usa
# "…_classifications"/"no_classification_provided". Se mapean ambos
# vocabularios a la misma escala de estrellas; un valor no reconocido (p. ej.
# una futura release con terminología nueva otra vez) queda NaN documentado,
# no un supuesto silencioso -- mismo criterio que el resto del contrato.
REVIEW_STATUS_STARS: dict[str, int] = {
    "no_assertion_provided": 0,
    "no_classification_provided": 0,
    "no_assertion_criteria_provided": 0,
    "no_interpretation_for_the_single_variant": 0,
    "no_classification_for_the_single_variant": 0,
    "no_classifications_from_unflagged_records": 0,
    "criteria_provided,_single_submitter": 1,
    "criteria_provided,_conflicting_interpretations": 1,
    "criteria_provided,_conflicting_classifications": 1,
    "criteria_provided,_multiple_submitters,_no_conflicts": 2,
    "reviewed_by_expert_panel": 3,
    "practice_guideline": 4,
}


def review_stars(status) -> float:
    """Estrellas ClinVar (0.0-4.0) a partir de CLNREVSTAT; NaN si no se reconoce."""
    if status is None or (isinstance(status, float) and np.isnan(status)):
        return np.nan
    return float(REVIEW_STATUS_STARS.get(str(status), np.nan))


CATEGORICAL_COLUMNS_ANNOTATED = [*CATEGORICAL_COLUMNS, "review_status"]
NUMERIC_RANGES["review_stars"] = (0.0, 4.0)

REQUIRED_COLUMNS = KEY_COLUMNS + CATEGORICAL_COLUMNS_ANNOTATED + list(NUMERIC_RANGES)


class SchemaError(ValueError):
    """Violación del contrato de datos de variantes anotadas."""


@dataclass
class ValidationReport:
    n_rows: int
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_annotated(df: pd.DataFrame, *, strict: bool = True) -> ValidationReport:
    """Valida el DataFrame anotado contra el contrato. Si strict, lanza SchemaError."""
    errors: list[str] = []

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"faltan columnas: {missing}")
        report = ValidationReport(len(df), errors)
        if strict:
            raise SchemaError("; ".join(errors))
        return report

    # Claves: sin nulos y sin duplicados
    if df[KEY_COLUMNS].isnull().any().any():
        errors.append("hay nulos en columnas clave (chrom/pos/ref/alt)")
    n_dup = df.duplicated(subset=KEY_COLUMNS).sum()
    if n_dup:
        errors.append(f"{n_dup} variantes duplicadas por clave")

    # chrom permitido
    bad_chrom = set(df["chrom"].astype(str)) - ALLOWED_CHROMS
    if bad_chrom:
        errors.append(f"cromosomas no permitidos: {sorted(bad_chrom)[:5]}")

    # pos entero positivo
    if not np.issubdtype(df["pos"].dtype, np.integer):
        errors.append("pos no es entero")
    elif (df["pos"] <= 0).any():
        errors.append("hay posiciones <= 0")

    # ref/alt bases válidas y distintas
    for col in ("ref", "alt"):
        bad = ~df[col].astype(str).apply(lambda s: len(s) >= 1 and set(s) <= BASES)
        if bad.any():
            errors.append(f"{int(bad.sum())} valores no-ACGT en {col}")
    if (df["ref"] == df["alt"]).any():
        errors.append("hay variantes con ref == alt")

    # rangos numéricos (ignorando NaN, que es válido en scores dbNSFP ausentes)
    for col, (lo, hi) in NUMERIC_RANGES.items():
        s = pd.to_numeric(df[col], errors="coerce")
        out_of_range = ((s < lo) | (s > hi)).sum()
        if out_of_range:
            errors.append(f"{int(out_of_range)} valores fuera de rango en {col} [{lo},{hi}]")

    report = ValidationReport(len(df), errors)
    if strict and errors:
        raise SchemaError("Contrato de datos violado: " + "; ".join(errors))
    return report
