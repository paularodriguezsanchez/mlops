"""Test de la sensibilidad de agregación por transcrito (revisión posterior del proyecto)."""
from __future__ import annotations

from src.evaluate import transcript_aggregation_sensitivity as tas


def test_max_or_none_lista_y_escalar():
    assert tas._max_or_none([0.1, 0.9, 0.5]) == 0.9
    assert tas._max_or_none(0.42) == 0.42
    assert tas._max_or_none(None) is None
    assert tas._max_or_none([None, None]) is None


def test_dig_navega_diccionarios_anidados():
    d = {"a": {"b": {"c": 7}}}
    assert tas._dig(d, "a", "b", "c") == 7
    assert tas._dig(d, "a", "x", "c") is None
    assert tas._dig({}, "a", "b") is None


def test_parse_hit_both_multi_transcrito_vs_escalar():
    hit = {
        "dbnsfp": {
            "cadd": {"phred": 20},
            "revel": {"score": [0.2, 0.8]},
        },
    }
    out = tas._parse_hit_both(hit)
    assert out["cadd_phred_mean"] == 20
    assert out["cadd_phred_max"] == 20
    assert out["cadd_phred_is_list"] is False
    assert out["revel_score_mean"] == 0.5
    assert out["revel_score_max"] == 0.8
    assert out["revel_score_is_list"] is True


def test_parse_hit_both_notfound_devuelve_vacio():
    assert tas._parse_hit_both({"notfound": True}) == {}
