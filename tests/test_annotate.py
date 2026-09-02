"""Tests del pipeline de anotación y del contrato de datos."""

import pandas as pd
import pytest

from src.annotate import annotate, schema
from src.ingest import synthetic as syn


def _make_raw(tmp_path):
    cfg = syn.SyntheticConfig(seed=1, chromosomes=("1",), n_variants_train=200,
                              n_new_in_test=40, frac_reclassified=0.1)
    tr, te = syn.generate_releases(cfg)
    vcf = tmp_path / "clinvar_2023-12.vcf.gz"
    dbnsfp = tmp_path / "dbnsfp_subset.tsv.gz"
    syn.write_clinvar_vcf(tr, vcf)
    syn.write_dbnsfp(te, dbnsfp)
    return vcf, dbnsfp


def test_parse_vcf_extrae_campos(tmp_path):
    vcf, _ = _make_raw(tmp_path)
    df = annotate.parse_clinvar_vcf(vcf)
    assert {"chrom", "pos", "ref", "alt", "clnsig", "gene", "consequence",
           "review_status", "review_stars"} <= set(df.columns)
    assert (df["pos"] > 0).all()
    assert df["clnsig"].notna().all()
    assert df["review_status"].notna().all()


def test_parse_vcf_mapea_review_status_a_estrellas(tmp_path):
    """CLNREVSTAT -> review_stars (0-4), ADR 008 (regresión del vocabulario real de ClinVar)."""
    vcf, _ = _make_raw(tmp_path)
    df = annotate.parse_clinvar_vcf(vcf)
    assert df["review_stars"].between(0, 4).all()
    assert df["review_stars"].notna().all()  # el generador sintético usa vocabulario reconocido


def test_annotate_join_y_contrato(tmp_path):
    vcf, dbnsfp = _make_raw(tmp_path)
    clinvar = annotate.parse_clinvar_vcf(vcf)
    feats = annotate.load_dbnsfp(dbnsfp)
    out = annotate.annotate(clinvar, feats, chroms=["1"])
    report = schema.validate_annotated(out, strict=False)
    assert report.ok, report.errors
    # missense tiene REVEL; no missense es NaN (como dbNSFP real)
    assert out.loc[out.consequence == "missense_variant", "revel_score"].notna().any()
    assert out.loc[out.consequence != "missense_variant", "revel_score"].isna().all()


def test_contrato_detecta_datos_corruptos():
    bad = pd.DataFrame({
        "chrom": ["1"], "pos": [-5], "ref": ["A"], "alt": ["A"],
        "gene": ["G"], "consequence": ["missense_variant"], "clnsig": ["Benign"],
        "review_status": ["reviewed_by_expert_panel"], "review_stars": [3.0],
        "cadd_phred": [999.0], "sift_score": [0.1], "polyphen_score": [0.1],
        "revel_score": [0.1], "alphamissense_score": [0.1],
        "gerp_rs": [1.0], "phylop": [1.0], "gnomad_af": [0.1],
    })
    with pytest.raises(schema.SchemaError):
        schema.validate_annotated(bad, strict=True)
