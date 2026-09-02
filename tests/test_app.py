"""Tests del servicio REST (src/serve/app.py): endpoints /health y /predict.

Hasta ahora `test_serve.py` solo testeaba `VariantAnnotator`/`VariantPredictor`
directamente, nunca a través de las rutas Flask reales (routing, códigos de
estado HTTP, validación de payload). Este módulo cierra ese hueco con
`app.test_client`, sustituyendo `get_predictor` por un doble de test para no
depender de un modelo entrenado en disco.
"""
from __future__ import annotations

import pytest

from src.serve import app as app_mod


class _StubPredictor:
    def __init__(self):
        self._annotator = ["v1", "v2"]

    def predict(self, chrom, pos, ref, alt):
        return {
            "variant": {"chrom": str(chrom), "pos": int(pos), "ref": ref, "alt": alt},
            "annotated": True,
            "gene": "GENE1",
            "consequence": "missense_variant",
            "prediction": "Patogénica",
            "probability_pathogenic": 0.9,
        }


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(app_mod, "get_predictor", lambda: _StubPredictor())
    app_mod.app.testing = True
    return app_mod.app.test_client()


def test_health_devuelve_estado_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "ok"
    assert body["n_annotations"] == 2


def test_predict_variante_valida(client):
    resp = client.post("/predict", json={"chrom": "1", "pos": 100, "ref": "A", "alt": "G"})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["prediction"] == "Patogénica"
    assert body["probability_pathogenic"] == 0.9


@pytest.mark.parametrize("payload", [
    {},
    {"chrom": "1"},
    {"chrom": "1", "pos": 100, "ref": "A"},
])
def test_predict_campos_faltantes_devuelve_400(client, payload):
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 400
    assert "faltan campos" in resp.get_json()["error"]


def test_predict_sin_json_devuelve_400(client):
    """Cuerpo no-JSON: `get_json(silent=True)` no debe tumbar el servicio con 500."""
    resp = client.post("/predict", data="esto no es json", content_type="text/plain")
    assert resp.status_code == 400


def test_predict_pos_no_numerico_devuelve_400_no_500(client, monkeypatch):
    """Un `pos` que no castea a int debe traducirse en 400, no en un 500 sin control."""
    class _RaisingPredictor(_StubPredictor):
        def predict(self, chrom, pos, ref, alt):
            int(pos)  # misma conversión que el annotator real: dispara ValueError
            raise AssertionError("no debería llegar aquí")

    monkeypatch.setattr(app_mod, "get_predictor", lambda: _RaisingPredictor())
    resp = client.post("/predict", json={"chrom": "1", "pos": "no-numerico",
                                         "ref": "A", "alt": "G"})
    assert resp.status_code == 400


def test_dashboard_registrado_en_la_app(client):
    resp = client.get("/dashboard")
    assert resp.status_code == 200


# --- Autenticación opcional por API key (revisión interna del proyecto) ---


def test_sin_api_key_configurada_no_exige_autenticacion(client, monkeypatch):
    """Comportamiento por defecto (TFM_API_KEY no definida): sin cambios, Fase I local."""
    monkeypatch.setattr(app_mod, "_API_KEY", None)
    resp = client.post("/predict", json={"chrom": "1", "pos": 100, "ref": "A", "alt": "G"})
    assert resp.status_code == 200


def test_con_api_key_configurada_rechaza_peticion_sin_cabecera(client, monkeypatch):
    monkeypatch.setattr(app_mod, "_API_KEY", "clave-secreta")
    resp = client.post("/predict", json={"chrom": "1", "pos": 100, "ref": "A", "alt": "G"})
    assert resp.status_code == 401


def test_con_api_key_configurada_rechaza_clave_incorrecta(client, monkeypatch):
    monkeypatch.setattr(app_mod, "_API_KEY", "clave-secreta")
    resp = client.post("/predict", json={"chrom": "1", "pos": 100, "ref": "A", "alt": "G"},
                       headers={"X-API-Key": "otra-cosa"})
    assert resp.status_code == 401


def test_con_api_key_configurada_acepta_clave_correcta(client, monkeypatch):
    monkeypatch.setattr(app_mod, "_API_KEY", "clave-secreta")
    resp = client.post("/predict", json={"chrom": "1", "pos": 100, "ref": "A", "alt": "G"},
                       headers={"X-API-Key": "clave-secreta"})
    assert resp.status_code == 200


def test_con_api_key_configurada_dashboard_tambien_protegido(client, monkeypatch):
    monkeypatch.setattr(app_mod, "_API_KEY", "clave-secreta")
    assert client.get("/dashboard").status_code == 401
    assert client.get("/dashboard", headers={"X-API-Key": "clave-secreta"}).status_code == 200


def test_con_api_key_configurada_health_sigue_abierto(client, monkeypatch):
    """/health no exige autenticación aunque haya API key: uso estándar para probes."""
    monkeypatch.setattr(app_mod, "_API_KEY", "clave-secreta")
    resp = client.get("/health")
    assert resp.status_code == 200
