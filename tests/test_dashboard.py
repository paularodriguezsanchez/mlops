"""Tests del dashboard interactivo de VUS (ADR 007)."""
from __future__ import annotations

import json

from flask import Flask

from src.serve import dashboard as dashboard_mod
from src.serve.dashboard import dashboard_bp


def _client(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard_mod, "PROJECT_ROOT", tmp_path)
    app = Flask(__name__)
    app.register_blueprint(dashboard_bp)
    return app.test_client()


def test_dashboard_sin_informe_muestra_instrucciones(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    resp = client.get("/dashboard/test")
    assert resp.status_code == 200
    assert b"vus_reports" in resp.data
    assert b"python -m src.serve.vus_reports" in resp.data


def test_dashboard_con_informe_renderiza_tabla(tmp_path, monkeypatch):
    out_dir = tmp_path / "reports" / "serving"
    out_dir.mkdir(parents=True)
    records = [
        {"chrom": "1", "pos": 100, "ref": "A", "alt": "G", "gene": "GENE1",
         "consequence": "missense_variant", "probabilidad_patogenica": 0.87,
         "probabilidad_reclasificacion": 0.21,
         "evidencia": [
             {"feature": "cadd_phred", "value": 32.0, "shap": 0.3,
              "direction": "patogénica", "acmg_tag": "PP3-like", "note": "CADD alto"},
             {"feature": "gnomad_af", "value": 1e-6, "shap": 0.2,
              "direction": "patogénica", "acmg_tag": "PM2-like", "note": "variante rara"},
             {"feature": "gerp_rs", "value": 5.5, "shap": 0.1,
              "direction": "patogénica", "acmg_tag": "PP3-like", "note": "conservada"},
         ]},
        {"chrom": "2", "pos": 200, "ref": "C", "alt": "T", "gene": None,
         "consequence": None, "probabilidad_patogenica": 0.12,
         "probabilidad_reclasificacion": None, "evidencia": []},
    ]
    (out_dir / "vus_reports_test.json").write_text(
        json.dumps(records, ensure_ascii=False), encoding="utf-8")

    client = _client(tmp_path, monkeypatch)
    resp = client.get("/dashboard/test")

    assert resp.status_code == 200
    html = resp.data.decode("utf-8")
    assert "1:100 A&gt;G" in html or "1:100 A>G" in html
    assert "GENE1" in html
    assert "87.0%" in html
    assert "no disponible" in html   # segunda fila sin reclasificación
    assert "PP3-like" in html
    assert "+1 m" in html   # 3 evidencias, 2 mostradas, "+1 más" colapsado


def test_dashboard_url_por_defecto_es_test(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    resp = client.get("/dashboard")
    assert resp.status_code == 200


def test_dashboard_split_no_permitido_devuelve_404(tmp_path, monkeypatch):
    """la revisión interna del proyecto: `split` debe validarse contra {"train","test"},
    no confiar solo en que el converter de Flask excluya '/'."""
    client = _client(tmp_path, monkeypatch)
    resp = client.get("/dashboard/produccion")
    assert resp.status_code == 404


def test_dashboard_indica_criterio_de_orden_segun_ranking_score(tmp_path, monkeypatch):
    """Cabecera dinámica: menciona el objetivo de ranking si los registros traen `ranking_score`, si
    no, el modelo de patogenicidad.
    """
    out_dir = tmp_path / "reports" / "serving"
    out_dir.mkdir(parents=True)
    sin_ranking = [{"chrom": "1", "pos": 1, "ref": "A", "alt": "G", "gene": "G1",
                    "consequence": "missense_variant", "probabilidad_patogenica": 0.5,
                    "probabilidad_reclasificacion": None, "evidencia": []}]
    (out_dir / "vus_reports_test.json").write_text(
        json.dumps(sin_ranking, ensure_ascii=False), encoding="utf-8")
    client = _client(tmp_path, monkeypatch)
    html_sin = client.get("/dashboard/test").data.decode("utf-8")
    assert "todavía no está entrenado" in html_sin

    con_ranking = [{**sin_ranking[0], "ranking_score": 1.23}]
    (out_dir / "vus_reports_test.json").write_text(
        json.dumps(con_ranking, ensure_ascii=False), encoding="utf-8")
    html_con = client.get("/dashboard/test").data.decode("utf-8")
    assert "modelo de ranking dedicado" in html_con


def test_dashboard_avisa_si_reclasificacion_no_es_fiable(tmp_path, monkeypatch):
    """Si el modelo de reclasificación no supera el umbral de fiabilidad,
    el dashboard debe marcarlo junto al número, no solo en el Model Card."""
    out_dir = tmp_path / "reports" / "serving"
    out_dir.mkdir(parents=True)
    records = [
        {"chrom": "1", "pos": 100, "ref": "A", "alt": "G", "gene": "GENE1",
         "consequence": "missense_variant", "probabilidad_patogenica": 0.5,
         "probabilidad_reclasificacion": 0.2, "probabilidad_reclasificacion_fiable": False,
         "evidencia": []},
        {"chrom": "2", "pos": 200, "ref": "C", "alt": "T", "gene": "GENE2",
         "consequence": "missense_variant", "probabilidad_patogenica": 0.5,
         "probabilidad_reclasificacion": 0.3, "probabilidad_reclasificacion_fiable": True,
         "evidencia": []},
    ]
    (out_dir / "vus_reports_test.json").write_text(
        json.dumps(records, ensure_ascii=False), encoding="utf-8")

    client = _client(tmp_path, monkeypatch)
    resp = client.get("/dashboard/test")

    assert resp.status_code == 200
    html = resp.data.decode("utf-8")
    assert "señal débil" in html
    assert html.count("señal débil") == 1   # solo la fila no fiable la muestra
