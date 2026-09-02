# Servidor de tracking MLflow con backend SQLite y artefactos en volumen local.
FROM python:3.12-slim

RUN pip install --no-cache-dir mlflow==3.14.0

WORKDIR /mlflow
EXPOSE 5000

# Backend store en SQLite; artefactos en /mlflow/mlartifacts (montado como volumen).
CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--backend-store-uri", "sqlite:////mlflow/mlflow.db", \
     "--artifacts-destination", "/mlflow/mlartifacts"]
