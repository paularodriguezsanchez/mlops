"""Anotación multi-fuente sin dbNSFP (ADR 007).

dbNSFP exige un registro académico que no llega de forma inmediata, así que consulto
directamente el agregador que expone las mismas fuentes sin registro:
myvariant.info devuelve CADD, SIFT, PolyPhen-2, GERP++, phyloP, REVEL,
AlphaMissense y la frecuencia de gnomAD en una sola llamada por lote.

SpliceAI se intenta contra el servicio del Broad Institute; no está accesible desde
este entorno, así que degrada a NaN documentado en vez de bloquear el pipeline.

No se descarga nada masivo: se consulta por lotes, y solo las variantes que
realmente se van a anotar tras acotar cromosomas y volumen.
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

_MYVARIANT_URL = "https://myvariant.info/v1/variant"
# myvariant.info asume hg19 por defecto: sin `assembly=hg38` explícito toda consulta
# devuelve "notfound" aunque el dato exista. Bajo hg38, además, CADD, SIFT, PolyPhen
# y conservación viven en el namespace `dbnsfp.*`, no en `cadd.*`.
_MYVARIANT_ASSEMBLY = "hg38"
# Se pide el objeto `dbnsfp` completo, no subcampos por dot-path: la proyección de
# `dbnsfp.gerp++.rs` devuelve vacío pese a existir el dato, porque la API no maneja
# bien los nombres con "+". Payload mayor, pero fiable.
_MYVARIANT_FIELDS = "dbnsfp,gnomad_exome.af.af,gnomad_genome.af.af"
_BATCH_SIZE = 1000
_TIMEOUT = 30

_SPLICEAI_URL = "https://spliceailookup-api.broadinstitute.org/spliceai/"
_SPLICEAI_TIMEOUT = 8

FEATURE_COLUMNS = [
    "chrom", "pos", "ref", "alt", "sift_score", "polyphen_score",
    "cadd_phred", "revel_score", "alphamissense_score", "gerp_rs",
    "phylop", "gnomad_af",
]


def _variant_id(chrom: str, pos: int, ref: str, alt: str) -> str:
    """Notación HGVS genómica que espera myvariant.info: chr7:g.140453136A>T."""
    c = str(chrom)
    if not c.startswith("chr"):
        c = f"chr{c}"
    return f"{c}:g.{int(pos)}{ref}>{alt}"


def _first(*values: float | None) -> float | None:
    for v in values:
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            return v
    return None


def _mean_or_none(x) -> float | None:
    """Media cuando el valor llega como lista, una entrada por transcrito."""
    if x is None:
        return None
    if isinstance(x, list):
        vals = [v for v in x if v is not None]
        return float(np.mean(vals)) if vals else None
    return float(x)


def _parse_hit(hit: dict) -> dict:
    """Extrae las columnas del objeto `dbnsfp` completo.

    Cualquiera de los campos puede llegar como lista si la variante mapea a varios
    transcritos, no solo AlphaMissense: todos se promedian, o el contrato de datos
    recibe listas donde espera numéricos.
    """
    if hit.get("notfound"):
        return {}
    dbnsfp = hit.get("dbnsfp", {}) or {}
    gnomad_ex = ((hit.get("gnomad_exome") or {}).get("af") or {}).get("af")
    gnomad_gen = ((hit.get("gnomad_genome") or {}).get("af") or {}).get("af")
    return {
        "cadd_phred": _mean_or_none((dbnsfp.get("cadd") or {}).get("phred")),
        "sift_score": _mean_or_none((dbnsfp.get("sift") or {}).get("score")),
        "polyphen_score": _mean_or_none(
            ((dbnsfp.get("polyphen2") or {}).get("hdiv") or {}).get("score")),
        "gerp_rs": _mean_or_none((dbnsfp.get("gerp++") or {}).get("rs")),
        "phylop": _mean_or_none(
            ((dbnsfp.get("phylop") or {}).get("100way_vertebrate") or {}).get("score")),
        "revel_score": _mean_or_none((dbnsfp.get("revel") or {}).get("score")),
        "alphamissense_score": _mean_or_none((dbnsfp.get("alphamissense") or {}).get("score")),
        "gnomad_af": _first(gnomad_ex, gnomad_gen),
    }


def fetch_myvariant(variants: pd.DataFrame, batch_size: int = _BATCH_SIZE,
                    timeout: int = _TIMEOUT) -> pd.DataFrame:
    """Consulta myvariant.info en lotes.

    Devuelve las claves más `FEATURE_COLUMNS`. Las variantes fuera de la cobertura
    del agregador quedan con NaN, igual que ocurriría con dbNSFP.
    """
    keys = variants[["chrom", "pos", "ref", "alt"]].drop_duplicates().reset_index(drop=True)
    ids = [_variant_id(*row) for row in keys.itertuples(index=False)]

    records: list[dict] = []
    n_failed_batches = 0
    for start in range(0, len(ids), batch_size):
        batch_ids = ids[start:start + batch_size]
        data = urllib.parse.urlencode({
            "ids": ",".join(batch_ids),
            "fields": _MYVARIANT_FIELDS,
            "assembly": _MYVARIANT_ASSEMBLY,
        }).encode()
        # S310: la URL es una constante HTTPS del módulo, no entrada externa.
        req = urllib.request.Request(  # noqa: S310
            _MYVARIANT_URL, data=data, headers={"User-Agent": "tfm-mlops-variantes"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
                hits = json.loads(resp.read())
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            print(f"  [myvariant.info no disponible] lote {start}: {exc}", file=sys.stderr)
            n_failed_batches += 1
            hits = [{} for _ in batch_ids]
        if isinstance(hits, dict):
            hits = [hits]
        batch_keys = keys.iloc[start:start + batch_size].itertuples(index=False)
        for row, hit in zip(batch_keys, hits, strict=False):
            rec = {"chrom": row.chrom, "pos": row.pos, "ref": row.ref, "alt": row.alt}
            rec.update(_parse_hit(hit if isinstance(hit, dict) else {}))
            records.append(rec)

    if n_failed_batches:
        print(f"  [aviso] {n_failed_batches} lote(s) de myvariant.info fallaron; "
              "esas variantes quedan sin features (NaN), igual que fuera de cobertura dbNSFP.",
              file=sys.stderr)

    out = pd.DataFrame.from_records(records)
    for col in FEATURE_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    return out[FEATURE_COLUMNS]


_spliceai_warned = False


def fetch_spliceai(chrom: str, pos: int, ref: str, alt: str,
                   timeout: int = _SPLICEAI_TIMEOUT) -> float | None:
    """Score SpliceAI (delta máximo) de una variante, en modo mejor esfuerzo.

    El servicio del Broad no es accesible desde este entorno (ADR 007 §1): si falla,
    avisa una sola vez por proceso y devuelve None, sin reintentar por variante.
    """
    global _spliceai_warned
    variant = f"{chrom}-{pos}-{ref}-{alt}"
    try:
        # S310: host constante; solo la query string lleva coordenadas ya validadas.
        req = urllib.request.Request(  # noqa: S310
            f"{_SPLICEAI_URL}?hg=38&variant={variant}&distance=50&mask=0",
            headers={"User-Agent": "tfm-mlops-variantes"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            data = json.loads(resp.read())
        scores = data.get("scores") or []
        deltas = [max(s.get(k, 0) for k in ("DS_AG", "DS_AL", "DS_DG", "DS_DL"))
                 for s in scores if isinstance(s, dict)]
        return float(max(deltas)) if deltas else None
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError, KeyError):
        if not _spliceai_warned:
            print("  [aviso] SpliceAI no accesible desde este entorno; "
                  "columna splice_ai_score queda NaN (ver ADR 007 §1).", file=sys.stderr)
            _spliceai_warned = True
        return None


def annotate_multi_source(clinvar: pd.DataFrame, chroms: list[str] | None = None,
                          include_spliceai: bool = False) -> pd.DataFrame:
    """Anota `clinvar` contra myvariant.info, con la misma salida que `annotate`.

    `include_spliceai=True` añade `splice_ai_score`: una consulta por variante,
    lento, desactivado por defecto.
    """
    from src.annotate import schema

    feats = fetch_myvariant(clinvar)
    merged = clinvar.merge(feats, on=schema.KEY_COLUMNS, how="left")
    if chroms:
        merged = merged[merged["chrom"].isin(chroms)].reset_index(drop=True)

    if include_spliceai:
        merged["splice_ai_score"] = [
            fetch_spliceai(r.chrom, r.pos, r.ref, r.alt)
            for r in merged.itertuples(index=False)
        ]

    # `review_status` y `review_stars` vienen del VCF, no del agregador: son
    # temporalmente seguros por release (ADR 008).
    ordered = (schema.KEY_COLUMNS + ["gene", "consequence", "clnsig", "review_status"]
              + list(schema.NUMERIC_RANGES))
    cols = ordered + (["splice_ai_score"] if include_spliceai else [])
    return merged[cols]
