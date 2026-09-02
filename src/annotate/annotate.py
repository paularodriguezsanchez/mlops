"""Capa SILVER: parsea el VCF de ClinVar y le añade las features de anotación.

Parsea el VCF (clave, gen, consecuencia, significancia clínica y estado de
revisión), acota a los cromosomas y al volumen configurados, cruza con la fuente
de features -myvariant.info en producción, el subset sintético en pruebas- y
valida el contrato de datos antes de escribir el parquet.

    python -m src.annotate.annotate [--release 2023-12] [--source multi_source]
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
    # Estado de revisión: se lee del VCF fechado, así que es temporalmente
    # seguro a diferencia del resto de features (ADR 008).
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
    # Solo SNVs. El VCF real trae entradas de una letra que no son bases ("." de
    # variantes estructurales truncadas, "N" ambigua): se descartan aquí, antes del
    # contrato de datos, en vez de dejar que las rechace aguas abajo.
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
    # Orden estable según el contrato: `review_status` viene del VCF y
    # `review_stars` entra por `schema.NUMERIC_RANGES`.
    ordered = (schema.KEY_COLUMNS + ["gene", "consequence", "clnsig", "review_status"]
              + list(schema.NUMERIC_RANGES))
    return merged[ordered]


def _sample_release(clinvar: pd.DataFrame, max_n: int, seed: int,
                    cohort_df: pd.DataFrame | None, max_new_n: int | None) -> pd.DataFrame:
    """Muestreo aleatorio simple con semilla fija y continuidad de cohorte.

    La primera release se muestrea a `max_n`. Las siguientes conservan la cohorte
    ya muestreada y añaden hasta `max_new_n` variantes nuevas.

    Ambas piezas son necesarias. Sin la cohorte, dos muestras independientes de una
    población de cientos de miles de variantes apenas comparten claves por azar, y
    la deriva de reclasificación se mide sobre un conjunto vacío. Sin el cupo de
    nuevas, la cohorte retenida agota `max_n` -ClinVar es acumulativo- y el holdout
    no visto queda reducido a una fracción marginal. El total puede superar `max_n`:
    es intencional.

    El muestreo no está estratificado por clase, cromosoma, gen ni consecuencia.
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
        # Cromosomas y volumen se acotan antes de anotar: con `multi_source` cada
        # variante cuesta una consulta de red, así que acotar solo la salida no
        # acota el coste.
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
