"""Pipeline de anotación: variante (ClinVar) x features (dbNSFP) → capa SILVER.

RAW (VCF + dbNSFP) → [este módulo] → SILVER (parquet anotado por release)

Pasos:
  1. Parsear el VCF de ClinVar (chrom, pos, ref, alt, gene, consequence, clnsig).
  2. Cargar el subset dbNSFP (features in silico + frecuencia gnomAD).
  3. Join por clave (chrom, pos, ref, alt) → tabla anotada.
  4. Acotar a los cromosomas de la Fase I y validar el contrato de datos.

Uso:
    python -m src.annotate.annotate # anota train y test
    python -m src.annotate.annotate --release 2023-12
"""
from __future__ import annotations

import argparse
import gzip
import re
from pathlib import Path

import pandas as pd

from src.annotate import schema
from src.config import (
    annotation_source,
    chromosomes_subset,
    get_seed,
    interim_dir,
    load_config,
    max_new_variants_per_release,
    max_variants_per_release,
    raw_dir,
)

_INFO_RE = {
    "clnsig": re.compile(r"CLNSIG=([^;]+)"),
    "gene": re.compile(r"GENEINFO=([^;:]+)"),
    "consequence": re.compile(r"MC=([^;,]+)"),
    # Estado de revisión de ClinVar (ADR 008): a diferencia de las demás
    # features (recalculadas "hoy" vía myvariant.info), este campo se lee
    # directamente del VCF fechado de cada release -- temporalmente seguro,
    # ver `schema.review_stars`.
    "review_status": re.compile(r"CLNREVSTAT=([^;]+)"),
}


def parse_clinvar_vcf(path: Path) -> pd.DataFrame:
    """Parsea un VCF de ClinVar (gzip) a DataFrame. Solo SNVs (ref/alt de 1 base)."""
    rows: list[dict] = []
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 8:
                continue
            chrom, pos, _id, ref, alt = f[0], f[1], f[2], f[3], f[4]
            info = f[7]
            rec = {
                "chrom": str(chrom),
                "pos": int(pos),
                "ref": ref,
                "alt": alt,
            }
            for key, rx in _INFO_RE.items():
                m = rx.search(info)
                rec[key] = m.group(1) if m else None
            rows.append(rec)
    df = pd.DataFrame(rows)
    df["review_stars"] = df["review_status"].apply(schema.review_stars)
    # Foco del anteproyecto: SNVs (una base ref y alt). El VCF real de ClinVar
    # también incluye entradas de una sola letra que NO son bases (p. ej. "."
    # para variantes estructurales truncadas, o "N" ambigua): se descartan aquí,
    # antes del contrato de datos, en vez de dejar que lo rechace aguas abajo
    # (revisión interna del proyecto).
    df = df[(df["ref"].str.len() == 1) & (df["alt"].str.len() == 1)]
    df = df[df["ref"].isin(schema.BASES) & df["alt"].isin(schema.BASES)
            & (df["ref"] != df["alt"])]
    return df.reset_index(drop=True)


def load_dbnsfp(path: Path) -> pd.DataFrame:
    """Carga el subset dbNSFP a DataFrame con nombres de columna normalizados."""
    df = pd.read_csv(path, sep="\t", dtype={"#chr": str}, na_values=["."])
    rename = {
        "#chr": "chrom",
        "pos(1-based)": "pos",
        "ref": "ref",
        "alt": "alt",
        "genename": "gene_dbnsfp",
        "SIFT_score": "sift_score",
        "Polyphen2_HDIV_score": "polyphen_score",
        "CADD_phred": "cadd_phred",
        "REVEL_score": "revel_score",
        "AlphaMissense_score": "alphamissense_score",
        "GERP++_RS": "gerp_rs",
        "phyloP100way_vertebrate": "phylop",
        "gnomAD_exomes_AF": "gnomad_af",
    }
    df = df.rename(columns=rename)
    df["chrom"] = df["chrom"].astype(str)
    df["pos"] = df["pos"].astype(int)
    return df


def annotate(clinvar: pd.DataFrame, dbnsfp: pd.DataFrame,
             chroms: list[str] | None = None) -> pd.DataFrame:
    """Une variantes de ClinVar con features dbNSFP por clave (chrom,pos,ref,alt)."""
    feature_cols = ["chrom", "pos", "ref", "alt", "sift_score", "polyphen_score",
                    "cadd_phred", "revel_score", "alphamissense_score",
                    "gerp_rs", "phylop", "gnomad_af"]
    merged = clinvar.merge(dbnsfp[feature_cols], on=schema.KEY_COLUMNS, how="left")
    if chroms:
        merged = merged[merged["chrom"].isin(chroms)].reset_index(drop=True)
    # Orden estable de columnas segun el contrato. `review_status` viaja desde
    # `clinvar` (ver `parse_clinvar_vcf`); `review_stars` entra vía
    # `schema.NUMERIC_RANGES` (ADR 008).
    ordered = (schema.KEY_COLUMNS + ["gene", "consequence", "clnsig", "review_status"]
              + list(schema.NUMERIC_RANGES))
    return merged[ordered]


def _sample_release(clinvar: pd.DataFrame, max_n: int, seed: int,
                    cohort_df: pd.DataFrame | None, max_new_n: int | None) -> pd.DataFrame:
    """Muestreo determinista con continuidad de cohorte entre releases.

    La PRIMERA release (`cohort_df is None`) se muestrea a `max_n` sin más.
    Las siguientes CONSERVAN todas las variantes de `cohort_df` (claves de la
    primera release ya muestreada) que sigan presentes -- imprescindible para
    medir persistencia y reclasificación real entre releases (`src, train, train_reclass.py`, y el
    concept drift de `src, monitor, `;
    muestrear cada release de forma independiente hace que, con ClinVar real, dos muestras de una
    población de cientos de miles de variantes casi no
    compartan ninguna clave por puro azar: 0 de 4496 VUS "reclasificadas" en
    la primera prueba, no porque no se hubiera reclasificado ninguna, sino
    porque ninguna de las VUS muestreadas en train aparecía en la muestra de
    test) -- y AÑADEN hasta `max_new_n` variantes genuinamente nuevas (no en
    la cohorte). Sin este segundo cupo, y como ClinVar real es acumulativo
    (casi toda variante de train sigue existiendo en test), la cohorte
    retenida por sí sola agota `max_n` y no queda casi ninguna variante
    genuinamente nueva con la que evaluar honestamente el modelo principal
    (bug detectado 2026-08-02: solo 75/3532, 2,1%, tras el fix de cohorte
    sin este segundo parámetro). Incluir `max_new_n` nuevas hace además que
    el total de la release pueda superar `max_n`: es intencional.
    """
    if cohort_df is None:
        return (clinvar.sample(n=max_n, random_state=seed)
               .sort_values(schema.KEY_COLUMNS).reset_index(drop=True))
    merged = clinvar.merge(cohort_df, on=schema.KEY_COLUMNS, how="left", indicator=True)
    in_cohort = (merged["_merge"] == "both").to_numpy()
    kept, rest = clinvar[in_cohort], clinvar[~in_cohort]
    if len(kept) > max_n:
        kept = kept.sample(n=max_n, random_state=seed)
    n_new = max_new_n if max_new_n is not None else max(0, max_n - len(kept))
    new_part = (rest.sample(n=min(n_new, len(rest)), random_state=seed)
               if n_new and len(rest) else rest.iloc[0:0])
    return (pd.concat([kept, new_part])
           .sort_values(schema.KEY_COLUMNS).reset_index(drop=True))


def run(release: str | None = None, source: str | None = None) -> dict[str, Path]:
    """Anota las releases indicadas y escribe parquet en data/interim. Devuelve rutas.

    `source`: 'synthetic' (dbNSFP-like offline, ADR 005) o 'multi_source'
    (myvariant.info real, sin dbNSFP, ADR 007/B5). Por defecto, el de
    `config.yaml` (`data.annotation_source`).
    """
    cfg = load_config()
    rawdir, outdir = raw_dir(), interim_dir()
    outdir.mkdir(parents=True, exist_ok=True)
    chroms = chromosomes_subset()
    src = source or annotation_source()

    releases = [release] if release else [
        cfg["data"]["clinvar_train_release"], cfg["data"]["clinvar_test_release"]
    ]
    dbnsfp = None if src == "multi_source" else load_dbnsfp(rawdir / "dbnsfp_subset.tsv.gz")
    max_n = max_variants_per_release()
    max_new_n = max_new_variants_per_release()

    outputs: dict[str, Path] = {}
    cohort_df: pd.DataFrame | None = None
    for rel in releases:
        clinvar = parse_clinvar_vcf(rawdir / f"clinvar_{rel}.vcf.gz")
        # Cromosomas y volumen se acotan ANTES de anotar, no después (bug de
        # rendimiento detectado 2026-08-01 al ejecutar contra ClinVar real: con
        # `multi_source`, cada variante cuesta una consulta de red real a
        # myvariant.info, así que acotar solo la SALIDA no acotaba el coste —
        # se disparaban decenas de miles de lotes de red para variantes que se
        # descartaban igualmente después, dejando el proceso prácticamente
        # colgado durante horas. `max_variants_per_release` existe precisamente
        # para evitar esto; tenía que aplicarse antes del fetch, no después.
        if chroms:
            clinvar = clinvar[clinvar["chrom"].isin(chroms)].reset_index(drop=True)
        if max_n and len(clinvar) > max_n:
            clinvar = _sample_release(clinvar, max_n, get_seed(), cohort_df, max_new_n)
        if cohort_df is None:
            cohort_df = clinvar[schema.KEY_COLUMNS].copy()
        if src == "multi_source":
            from src.annotate.multi_source import annotate_multi_source
            annotated = annotate_multi_source(clinvar, chroms)
        else:
            annotated = annotate(clinvar, dbnsfp, chroms)
        report = schema.validate_annotated(annotated, strict=True)
        out = outdir / f"annotated_{rel}.parquet"
        annotated.to_parquet(out, index=False)
        outputs[rel] = out
        n_feat = annotated["cadd_phred"].notna().sum()
        print(f"[{rel}] {report.n_rows} variantes anotadas "
              f"({n_feat} con features, fuente={src}) -> {out.name}")
    return outputs


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Anotación de variantes (SILVER).")
    p.add_argument("--release", default=None, help="Anota solo una release (AAAA-MM).")
    p.add_argument("--source", default=None, choices=["synthetic", "multi_source"],
                   help="Fuente de features (por defecto, la de config.yaml).")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    run(release=args.release, source=args.source)
