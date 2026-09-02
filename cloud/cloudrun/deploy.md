# Despliegue en Cloud Run (diseño, no ejecutado)

Empaqueta el mismo servicio que corre en local (`src/serve/app.py` tras gunicorn) reutilizando `docker/app.Dockerfile` sin cambios: la portabilidad local-nube es una decisión de arquitectura explícita (ADR 001), no algo que haya que rediseñar para la nube.

## Build y publicación

```bash
gcloud artifacts repositories create tfm-repo \
  --repository-format=docker --location=europe-west1

docker build -t europe-west1-docker.pkg.dev/<PROYECTO>/tfm-repo/serve:latest \
  -f docker/app.Dockerfile .

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
  --set-env-vars MLFLOW_TRACKING_URI=<URI de MLflow> \
  --set-secrets TFM_API_KEY=tfm-api-key:latest \
  --no-allow-unauthenticated
```

Cada valor responde a una decisión concreta:

- **`--min-instances 0`**: escala a cero entre peticiones, coste cero fuera de uso real.
- **`--no-allow-unauthenticated`**: IAM como primera capa de autenticación, además de `TFM_API_KEY`. Con el servicio accesible por URL pública, la autenticación deja de ser opcional.
- **`TFM_API_KEY` vía Secret Manager**, nunca como variable de entorno en claro en el manifiesto.
- **El modelo no se empaqueta en la imagen**: se monta desde Cloud Storage al arrancar el contenedor, mismo patrón que el bind mount local, para no reconstruir la imagen en cada reentrenamiento.

## Coste

Dentro del *free tier* de Cloud Run (2 millones de peticiones y 360 000 GB-segundo al mes) para el volumen de una demo. El riesgo real de cobro es dejar `min-instances > 0` o el servicio de MLflow sin escalar a cero; se evita por diseño y desmontando los recursos tras cada demo.
