"""Representatividad de los cromosomas 1-3 frente al resto del genoma.

revisión posterior del proyecto: la Fase I local restringe la
anotación a SNVs de los cromosomas 1-3 por volumen y coste de consultas de
red (declarado en la memoria, sección~5.9), pero hasta esta revisión no se
había estudiado si chr1-3 son representativos del resto de ClinVar en genes,
consecuencia funcional, estado de revisión o distribución de clases -- los
resultados del proyecto se generalizan implícitamente a "las VUS" sin
acotar esa generalización a la población realmente estudiada.

Este módulo NO requiere red: el VCF crudo de ClinVar ya descargado
(`data/raw/clinvar_*.vcf.gz`) contiene el genoma completo; el recorte a
chr1-3 ocurre más adelante, en `annotate.py`. Aquí se parsea el VCF completo
sin recortar y se compara la distribución de chr1-3 frente al resto del
genoma (autosomas 4-22 + X + Y; se excluyen contigs alternativos/no
estándar) en tres ejes: bucket de CLNSIG (positivo/negativo/VUS/excluido,
misma taxonomía que `build_dataset.py`), estado de revisión (`review_stars`)
y consecuencia funcional (`MC`, top-10 categorías). La métrica es el mismo
Population Stability Index (PSI) ya usado por `src/monitor/drift.py` para
detectar deriva entre releases, aplicado aquí entre poblaciones geográficas
(cromosómicas) en vez de temporales.

Uso:
    python -m src.evaluate.chr_representativeness --release 2025-06
"""
from __future__ import annotations

import argparse

import pandas as pd

from src.annotate.annotate import parse_clinvar_vcf
from src.config import PROJECT_ROOT, chromosomes_subset, load_config, raw_dir
from src.features.build_dataset import _label_map
from src.monitor.drift import _psi_categorical

_STANDARD_CHROMS = {str(i) for i in range(1, 23)} | {"X", "Y"}


def _clnsig_bucket(clnsig: pd.Series, mapping: dict[str, int]) -> pd.Series:
    def bucket(v):
        if v in mapping:
            return "positivo" if mapping[v] == 1 else "negativo"
        if v == "Uncertain_significance":
            return "vus"
        return "excluido"
    return clnsig.map(bucket)


def _top_consequence(mc: pd.Series, top_n: int = 10) -> pd.Series:
    """Colapsa consecuencias fuera del top-N en 'otra' para que el PSI categórico
    no explote con decenas de categorías raras casi vacías en algún grupo."""
    top = mc.value_counts().head(top_n).index
    return mc.where(mc.isin(top), "otra")


def compute_representativeness(raw: pd.DataFrame, cfg: dict) -> dict:
    subset = set(chromosomes_subset())
    raw = raw[raw["chrom"].isin(_STANDARD_CHROMS)].copy()
    is_subset = raw["chrom"].isin(subset)
    chr13, resto = raw[is_subset], raw[~is_subset]

    mapping = _label_map(cfg)
    chr13_bucket = _clnsig_bucket(chr13["clnsig"], mapping)
    resto_bucket = _clnsig_bucket(resto["clnsig"], mapping)
    psi_clnsig = _psi_categorical(resto_bucket, chr13_bucket)

    psi_review = _psi_categorical(
        resto["review_stars"].astype("Int64").astype(str),
        chr13["review_stars"].astype("Int64").astype(str))

    all_mc = pd.concat([chr13["consequence"], resto["consequence"]])
    top_mc = _top_consequence(all_mc)
    chr13_mc = top_mc.loc[chr13.index]
    resto_mc = top_mc.loc[resto.index]
    psi_consequence = _psi_categorical(resto_mc, chr13_mc)

    genes_chr13 = set(chr13["gene"].dropna())
    genes_resto = set(resto["gene"].dropna())

    return {
        "n_chr1_3": int(len(chr13)), "n_resto_genoma": int(len(resto)),
        "psi_clnsig_bucket": round(psi_clnsig, 4),
        "psi_review_stars": round(psi_review, 4),
        "psi_consequence_top10": round(psi_consequence, 4),
        "clnsig_bucket_chr1_3": chr13_bucket.value_counts(normalize=True).round(4).to_dict(),
        "clnsig_bucket_resto": resto_bucket.value_counts(normalize=True).round(4).to_dict(),
        "review_stars_mean_chr1_3": round(float(chr13["review_stars"].mean()), 4),
        "review_stars_mean_resto": round(float(resto["review_stars"].mean()), 4),
        "n_genes_chr1_3": len(genes_chr13), "n_genes_resto": len(genes_resto),
        "genes_chr1_3_cobertura_pct": round(
            100 * len(genes_chr13) / (len(genes_chr13) + len(genes_resto)), 2)
            if (genes_chr13 or genes_resto) else 0.0,
    }


def run(release: str | None = None) -> dict:
    cfg = load_config()
    release = release or cfg["data"]["clinvar_test_release"]
    path = raw_dir() / f"clinvar_{release}.vcf.gz"
    raw = parse_clinvar_vcf(path)
    result = compute_representativeness(raw, cfg)
    result["release"] = release

    out_dir = PROJECT_ROOT / "reports" / "training"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "release": release, **{k: v for k, v in result.items()
                               if not isinstance(v, dict)},
    }]).to_csv(out_dir / "chr_representativeness.csv", index=False)

    print(f"Representatividad chr1-3 vs resto del genoma (release {release}):")
    print(f"  n chr1-3={result['n_chr1_3']} | n resto={result['n_resto_genoma']}")
    print(f"  PSI bucket CLNSIG={result['psi_clnsig_bucket']} "
          f"| PSI review_stars={result['psi_review_stars']} "
          f"| PSI consequence(top10)={result['psi_consequence_top10']}")
    print(f"  bucket chr1-3={result['clnsig_bucket_chr1_3']}")
    print(f"  bucket resto ={result['clnsig_bucket_resto']}")
    # Cada gen vive en un único cromosoma por definición biológica: el 100%
    # de solape nulo entre `genes_chr1_3` y `genes_resto` es esperado y NO es
    # una señal de sesgo, así que se reporta como cobertura (qué fracción del
    # total de genes con variantes queda representada en chr1-3), no como
    # solapamiento.
    total_genes = result["n_genes_chr1_3"] + result["n_genes_resto"]
    cobertura_pct = round(100 * result["n_genes_chr1_3"] / total_genes, 1) if total_genes else 0.0
    print(f"  genes distintos: chr1-3={result['n_genes_chr1_3']} "
          f"resto={result['n_genes_resto']} "
          f"(chr1-3 cubre el {cobertura_pct}% del total de genes con variantes "
          "en la release; el resto es, por definición biológica, exclusivo de "
          "otros cromosomas, no una señal de sesgo de selección)")
    return result


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Representatividad de chr1-3 frente al resto del genoma "
                    ".")
    p.add_argument("--release", default=None, help="Release a analizar (por defecto, test).")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    run(release=args.release)
