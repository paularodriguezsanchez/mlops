"""Servicio REST de inferencia de patogenicidad [OE4].

Endpoints:
  GET /health → estado del servicio y del almacén de anotación (sin auth,
                             uso estándar para probes de liveness/readiness).
  POST /predict → {"chrom","pos","ref","alt"} → anotación + predicción.
  GET /dashboard[/<split>] → exploración interactiva de VUS priorizadas.

Autenticación (revisión interna del proyecto): opcional vía la variable de entorno
`TFM_API_KEY`. Si está definida, `/predict` y `/dashboard*` exigen la cabecera
`X-API-Key` con ese valor (401 si falta o no coincide); `/health` queda siempre
abierto. Si NO está definida (por defecto en Fase I local), el servicio sigue
funcionando sin autenticación, como hasta ahora, avisando una vez al arrancar —
mismo patrón de degradación explícita que el resto del proyecto (ver ADR 005).
Antes de exponer el servicio en Fase II (Cloud Run), definir `TFM_API_KEY` es
obligatorio.

Arranque local:
    python -m src.serve.app # servidor de desarrollo en:8000
    # o vía Docker / gunicorn en producción (Cloud Run en Fase II).
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
# 1 MB: generoso para el payload JSON diminuto que espera este servicio;
# corta de raíz cuerpos de petición desproporcionados (revisión interna del proyecto).
app.config["MAX_CONTENT_LENGTH"] = 1_000_000

_REQUIRED = ("chrom", "pos", "ref", "alt")
_API_KEY = os.environ.get("TFM_API_KEY")
_PROTECTED_PREFIXES = ("/predict", "/dashboard")

if not _API_KEY:
    print("[aviso] TFM_API_KEY no definida: /predict y /dashboard quedan sin "
          "autenticación (aceptable en Fase I local; obligatorio antes de Fase II).",
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
    # 127.0.0.1 por defecto (revisión interna del proyecto): evita exponer el servidor de
    # desarrollo Flask a toda la LAN al ejecutarlo directamente en el host. La
    # ruta que sí debe ser accesible desde fuera es gunicorn dentro de Docker
    # (`compose.yaml`, servicio `serve`), que no pasa por esta función y ya usa
    # 0.0.0.0 explícitamente. Override para casos que sí lo necesiten.
    host = os.environ.get("TFM_SERVE_HOST", "127.0.0.1")
    app.run(host=host, port=8000)


if __name__ == "__main__":
    main()
