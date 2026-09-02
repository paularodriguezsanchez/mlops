"""Tests de la binarización del target (GOLD)."""
import pandas as pd

from src.config import load_config
from src.features import build_dataset as bd


def test_binarize_mapea_y_reserva_vus():
    cfg = load_config()
    df = pd.DataFrame({
        "chrom": ["1"] * 6, "pos": [1, 2, 3, 4, 5, 6],
        "ref": list("ACGTAC"), "alt": list("GTACGT"),
        "gene": ["G"] * 6, "consequence": ["missense_variant"] * 6,
        "clnsig": ["Pathogenic", "Likely_pathogenic", "Benign",
                   "Likely_benign", "Uncertain_significance",
                   "Conflicting_interpretations_of_pathogenicity"],
        "cadd_phred": [30, 28, 3, 5, 15, 12], "sift_score": [0.1] * 6,
        "polyphen_score": [0.9] * 6, "revel_score": [0.8] * 6,
        "gerp_rs": [5] * 6, "phylop": [5] * 6, "gnomad_af": [1e-5] * 6,
    })
    labeled, vus, excluded = bd.binarize_target(df, cfg)
    assert set(labeled["label"]) == {0, 1}
    assert labeled["label"].tolist() == [1, 1, 0, 0]
    # VUS estricta: solo "Uncertain_significance", no clasificaciones conflictivas.
    assert len(vus) == 1 and vus.iloc[0]["clnsig"] == "Uncertain_significance"
    assert len(excluded) == 1
    assert excluded.iloc[0]["clnsig"] == "Conflicting_interpretations_of_pathogenicity"


def test_label_taxonomy_clasifica_cada_clnsig_crudo():
    cfg = load_config()
    df = pd.DataFrame({
        "clnsig": ["Pathogenic", "Benign", "Uncertain_significance",
                   "not_provided", "not_provided"],
    })
    tax = bd._label_taxonomy(df, cfg)
    buckets = dict(zip(tax["clnsig_crudo"], tax["bucket"], strict=True))
    assert buckets["Pathogenic"] == "positivo"
    assert buckets["Benign"] == "negativo"
    assert buckets["Uncertain_significance"] == "vus"
    assert buckets["not_provided"] == "excluido"
    assert tax.loc[tax["clnsig_crudo"] == "not_provided", "n"].item() == 2
