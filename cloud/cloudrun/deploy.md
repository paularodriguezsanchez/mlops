# Despliegue en Cloud Run (diseño, no ejecutado — Opción A)

Empaquetado del mismo servicio que corre en local (`src/serve/app.py`, gunicorn), reutilizando `docker/app.Dockerfile` sin cambios: la portabilidad local↔cloud es una decisión de arquitectura explícita (ADR 001), no algo que haya que rediseñar para la nube.

## Build y publicación de la imagen

```bash
gcloud artifacts repositories create tfm-repo \
  --repository-format=docker --location=europe-west1

docker build -t europe-west1-docker.pkg.dev/<PROYECTO>/tfm-repo/serve:latest \
  -f docker/app.Dockerfile.

docker push europe-west1-docker.pkg.dev/<PROYECTO>/tfm-repo/serve:latest
```

## Despliegue

```bash
gcloud run deploy tfm-serve \
  --image europe-west1-docker.pkg.dev/<PROYECTO>/tfm-repo/serve:latest \
  --region europe-west1 \
  --command gunicorn --args="--bind,0.0.0.0:8080,src.serve.app:app" \
  --port 8080 \
  --min-instances 0 --max-instances 2 \
  --set-env-vars MLFLOW_TRACKING_URI=<URI del MLflow en Cloud Run o Cloud SQL> \
  --set-secrets TFM_API_KEY=tfm-api-key:latest \
  --no-allow-unauthenticated
```

Decisiones de diseño, explícitas para no dejar dudas de por qué se elige cada valor:

- **`--min-instances 0`**: escala a cero entre peticiones — coste cero fuera de uso real, requisito no negociable del proyecto (§4 el plan del proyecto).
- **`--no-allow-unauthenticated`**: usa IAM de Cloud Run como primera capa de autenticación (además de `TFM_API_KEY`, ya implementada en `src/serve/app.py`) — en Fase II, con el servicio potencialmente accesible por URL pública, la autenticación deja de ser opcional.
- **`TFM_API_KEY` vía Secret Manager** (`--set-secrets`), nunca como variable de entorno en claro en el manifiesto de despliegue.
- **El modelo (`models/best_model`, `models/ranking_model`, `models/reclassification_model`) no se empaqueta en la imagen**: se monta desde Cloud Storage en el arranque del contenedor (mismo patrón que el bind mount de `compose.yaml` en local), para no tener que reconstruir la imagen en cada reentrenamiento.

## Coste

Dentro del *free tier* de Cloud Run (2 millones de peticiones/mes, 360.000 GB-segundo de memoria) para el volumen de uso de una demo de TFM. Riesgo de cobro real: dejar instancias `min-instances > 0` o el servicio de MLflow en Cloud Run sin escalar a cero — evitado por diseño (`--min-instances 0`) y por desmontar los recursos tras cada demo, como fija `docs/retraining_flow.md`,.
