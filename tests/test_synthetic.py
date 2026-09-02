"""Tests del generador offline determinista (ADR 005)."""
import numpy as np

from src.ingest import synthetic as syn


def _small_cfg():
    return syn.SyntheticConfig(seed=42, chromosomes=("1", "2"),
                               n_variants_train=400, n_new_in_test=80,
                               frac_reclassified=0.1)


def test_determinismo_misma_semilla():
    a_tr, a_te = syn.generate_releases(_small_cfg())
    b_tr, b_te = syn.generate_releases(_small_cfg())
    assert np.array_equal(a_tr["clnsig"], b_tr["clnsig"])
    assert np.array_equal(a_te["pos"], b_te["pos"])
    assert np.allclose(a_tr["cadd_phred"], b_tr["cadd_phred"])


def test_release_test_es_superconjunto_y_tiene_drift():
    cfg = _small_cfg()
    tr, te = syn.generate_releases(cfg)
    assert len(te["pos"]) == len(tr["pos"]) + cfg.n_new_in_test
    # Alguna VUS del train debe haberse reclasificado en test (drift real).
    shared = len(tr["pos"])
    changed = (tr["clnsig"] != te["clnsig"][:shared]).sum()
    assert changed > 0


def test_features_correlacionan_con_patogenicidad():
    tr, _ = syn.generate_releases(_small_cfg())
    patho = np.isin(tr["clnsig"], ["Pathogenic", "Likely_pathogenic"])
    benign = np.isin(tr["clnsig"], ["Benign", "Likely_benign"])
    # CADD medio mayor en patogénicas; AF media menor.
    assert tr["cadd_phred"][patho].mean() > tr["cadd_phred"][benign].mean()
    assert tr["gnomad_af"][patho].mean() < tr["gnomad_af"][benign].mean()


def test_ficheros_escritos_tienen_esquema(tmp_path):
    tr, te = syn.generate_releases(_small_cfg())
    vcf = tmp_path / "clinvar.vcf.gz"
    dbnsfp = tmp_path / "dbnsfp.tsv.gz"
    syn.write_clinvar_vcf(tr, vcf)
    syn.write_dbnsfp(te, dbnsfp)
    import gzip
    head = gzip.open(vcf, "rt").readline()
    assert head.startswith("##fileformat=VCF")
    cols = gzip.open(dbnsfp, "rt").readline().strip().split("\t")
    assert cols[0] == "#chr" and "CADD_phred" in cols
