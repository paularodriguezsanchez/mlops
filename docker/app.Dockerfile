# Imagen de la aplicación: pipeline + serving. Reutilizable en local y en Cloud Run.
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ENV PYTHONPATH=/app
# Por defecto abre una shell; el comando concreto lo fija compose.yaml o Cloud Run.
CMD ["bash"]
