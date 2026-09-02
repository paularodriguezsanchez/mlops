"""Test de la representatividad de chr1-3."""
from __future__ import annotations

import pandas as pd

from src.evaluate import chr_representativeness as cr


def _raw():
    # 4 en chr1-3, 4 en el resto (chr4/chr10), mismas proporciones de clase en
    # ambos grupos: el PSI resultante debe ser ~0 (sin deriva).
    rows = []
    for chrom in ["1", "2", "1", "3", "4", "10", "4", "10"]:
        rows.append({
            "chrom": chrom, "pos": 1, "ref": "A", "alt": "T", "gene": f"GENE_{chrom}",
            "clnsig": "Pathogenic" if chrom in ("1", "4") else "Benign",
            "consequence": "missense_variant", "review_stars": 2,
        })
    return pd.DataFrame(rows)


def _cfg():
    return {"target": {
        "positive_labels": ["Pathogenic"],
        "negative_labels": ["Benign"],
    }}


def test_clnsig_bucket_clasifica_positivo_negativo_vus_excluido():
    mapping = {"Pathogenic": 1, "Benign": 0}
    s = pd.Series(["Pathogenic", "Benign", "Uncertain_significance", "other"])
    buckets = cr._clnsig_bucket(s, mapping)
    assert list(buckets) == ["positivo", "negativo", "vus", "excluido"]


def test_top_consequence_colapsa_categorias_fuera_del_top_n():
    mc = pd.Series(["missense_variant"] * 5 + ["rare_one"] * 1)
    top = cr._top_consequence(mc, top_n=1)
    assert (top == "otra").sum() == 1
    assert (top == "missense_variant").sum() == 5


def test_compute_representativeness_sin_deriva_da_psi_bajo(monkeypatch):
    monkeypatch.setattr(cr, "chromosomes_subset", lambda: ["1", "2", "3"])
    result = cr.compute_representativeness(_raw(), _cfg())
    assert result["n_chr1_3"] == 4
    assert result["n_resto_genoma"] == 4
    # Mismas proporciones de clase en ambos grupos (2 patogénica/2 benigna en
    # cada uno): PSI debe ser numéricamente ~0, no solo "bajo".
    assert result["psi_clnsig_bucket"] < 1e-6
    # chr1-3: GENE_1 (x2, mismo gen), GENE_2, GENE_3 -> 3 genes distintos.
    # resto: GENE_4 (x2), GENE_10 (x2) -> 2 genes distintos.
    assert result["n_genes_chr1_3"] == 3
    assert result["genes_chr1_3_cobertura_pct"] == 60.0
