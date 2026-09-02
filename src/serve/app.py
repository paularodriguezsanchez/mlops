"""Servicio REST de inferencia de patogenicidad.

  GET  /health              estado del servicio y del almacén de anotación.
  POST /predict             {chrom, pos, ref, alt} -> anotación y predicción.
  GET  /dashboard[/<split>] exploración interactiva de las VUS priorizadas.

La autenticación es opcional: si `TFM_API_KEY` está definida, `/predict` y
`/dashboard` exigen la cabecera `X-API-Key`; `/health` queda siempre abierta, uso
estándar para sondas de disponibilidad. Sin esa variable el servicio funciona sin
autenticación y avisa al arrancar, aceptable en local pero no antes de exponerlo.

    python -m src.serve.app   # servidor de desarrollo; en Docker corre con gunicorn
"""
from __future__ import annotations

import os
import secrets
import sys

from flask import Flask, abort, jsonify, request

from src.serve.dashboard import dashboard_bp
from src.serve.predictor import get_predictor

app = Flask(__name__)
app.register_blueprint(dashboard_bp)
# 1 MB, holgado para el JSON diminuto que espera el servicio: corta cuerpos de
# petición desproporcionados.
app.config["MAX_CONTENT_LENGTH"] = 1_000_000

_REQUIRED = ("chrom", "pos", "ref", "alt")
_API_KEY = os.environ.get("TFM_API_KEY")
_PROTECTED_PREFIXES = ("/predict", "/dashboard")

if not _API_KEY:
    print("[aviso] TFM_API_KEY no definida: /predict y /dashboard quedan sin "
          "autenticación; obligatorio definirla antes de exponer el servicio.",
          file=sys.stderr)


@app.before_request
def _require_api_key():
    if not _API_KEY or not request.path.startswith(_PROTECTED_PREFIXES):
        return None
    given = request.headers.get("X-API-Key", "")
    if not secrets.compare_digest(given, _API_KEY):
        abort(401, description="falta o es incorrecta la cabecera X-API-Key")
    return None


@app.get("/health")
def health():
    predictor = get_predictor()
    return jsonify({"status": "ok",
                    "n_annotations": len(predictor._annotator),
                    "model": "variant_pathogenicity_clf"})


@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    missing = [k for k in _REQUIRED if k not in data]
    if missing:
        return jsonify({"error": f"faltan campos: {missing}"}), 400
    try:
        result = get_predictor().predict(
            data["chrom"], data["pos"], data["ref"], data["alt"])
    except (ValueError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(result)


def main():
    # 127.0.0.1 por defecto: no expone el servidor de desarrollo a toda la LAN. Lo
    # que sí debe ser accesible desde fuera es gunicorn dentro de Docker, que no pasa
    # por aquí y ya escucha en 0.0.0.0.
    host = os.environ.get("TFM_SERVE_HOST", "127.0.0.1")
    app.run(host=host, port=8000)


if __name__ == "__main__":
    main()
