"""Tests de la anotación multi-fuente real (B5/ADR 007), con HTTP simulado."""
import json
from unittest.mock import MagicMock, patch

import pandas as pd

from src.annotate import multi_source as ms


def _variants():
    # `review_status`/`review_stars` normalmente los añade
    # `annotate.parse_clinvar_vcf` (ADR 008); este fixture bypassa ese paso,
    # así que los aporta directamente, igual que gene/consequence/clnsig.
    return pd.DataFrame([
        {"chrom": "7", "pos": 140753336, "ref": "A", "alt": "T",
         "gene": "BRAF", "consequence": "missense_variant", "clnsig": "Pathogenic",
         "review_status": "reviewed_by_expert_panel", "review_stars": 3.0},
        {"chrom": "1", "pos": 123456, "ref": "C", "alt": "G",
         "gene": "GENE1", "consequence": "intron_variant", "clnsig": "Benign",
         "review_status": "criteria_provided,_single_submitter", "review_stars": 1.0},
    ])


def _fake_hits():
    return [
        {
            "query": "chr7:g.140753336A>T",
            "dbnsfp": {
                "cadd": {"phred": 32},
                "sift": {"score": 0},
                "polyphen2": {"hdiv": {"score": 0.967}},
                "gerp++": {"rs": 5.65},
                "phylop": {"100way_vertebrate": {"score": 5.101}},
                "revel": {"score": [0.931, 0.931]},  # multi-transcrito: debe promediarse
                "alphamissense": {"score": [0.98, 0.99]},
            },
            "gnomad_exome": {"af": {"af": 3.97994e-06}},
        },
        {"query": "chr1:g.123456C>G", "notfound": True},
    ]


def _mock_response(payload: bytes):
    resp = MagicMock()
    resp.read.return_value = payload
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    return resp


def test_variant_id_formato_hgvs():
    assert ms._variant_id("7", 140453136, "A", "T") == "chr7:g.140453136A>T"
    assert ms._variant_id("chrX", 5, "G", "C") == "chrX:g.5G>C"


def test_parse_hit_extrae_todas_las_columnas():
    parsed = ms._parse_hit(_fake_hits()[0])
    assert parsed["cadd_phred"] == 32
    assert parsed["sift_score"] == 0
    assert parsed["polyphen_score"] == 0.967
    assert parsed["gerp_rs"] == 5.65
    assert parsed["revel_score"] == 0.931
    assert parsed["alphamissense_score"] == 0.985   # media de [0.98, 0.99]
    assert abs(parsed["gnomad_af"] - 3.97994e-06) < 1e-12


def test_parse_hit_notfound_devuelve_vacio():
    assert ms._parse_hit({"notfound": True}) == {}


@patch("src.annotate.multi_source.urllib.request.urlopen")
def test_fetch_myvariant_rellena_encontradas_y_nan_las_ausentes(mock_urlopen):
    mock_urlopen.return_value = _mock_response(json.dumps(_fake_hits()).encode())
    out = ms.fetch_myvariant(_variants())
    assert list(out.columns) == ms.FEATURE_COLUMNS
    row = out[(out["chrom"] == "7") & (out["pos"] == 140753336)].iloc[0]
    assert row["cadd_phred"] == 32
    assert row["alphamissense_score"] == 0.985
    missing = out[(out["chrom"] == "1") & (out["pos"] == 123456)].iloc[0]
    assert pd.isna(missing["cadd_phred"])


@patch("src.annotate.multi_source.urllib.request.urlopen")
def test_fetch_myvariant_degrada_a_nan_si_falla_la_red(mock_urlopen):
    mock_urlopen.side_effect = OSError("red no disponible")
    out = ms.fetch_myvariant(_variants())
    assert len(out) == 2
    assert out["cadd_phred"].isna().all()


@patch("src.annotate.multi_source.urllib.request.urlopen")
def test_annotate_multi_source_forma_igual_que_annotate_clasico(mock_urlopen):
    mock_urlopen.return_value = _mock_response(json.dumps(_fake_hits()).encode())
    from src.annotate import schema
    out = ms.annotate_multi_source(_variants())
    report = schema.validate_annotated(out, strict=False)
    assert report.ok, report.errors
    assert list(out.columns) == (
        schema.KEY_COLUMNS + ["gene", "consequence", "clnsig", "review_status"]
        + list(schema.NUMERIC_RANGES)
    )


def test_fetch_spliceai_degrada_a_none_sin_reintentar_ruido(monkeypatch):
    ms._spliceai_warned = False

    def _boom(*a, **k):
        raise OSError("no accesible")
    monkeypatch.setattr(ms.urllib.request, "urlopen", _boom)
    assert ms.fetch_spliceai("7", 140453136, "A", "T") is None
    assert ms._spliceai_warned is True
