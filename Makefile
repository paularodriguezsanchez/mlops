# Comandos reproducibles del proyecto. Documentacion ejecutable: cada objetivo
# invoca literalmente el modulo de Python correspondiente, sin logica propia.
.PHONY: help pipeline core data setup up down lint test ingest annotate dataset eda train train-reclass train-ranking serve examples prioritize vus-reports monitor retrain ingest-prospective validate-prospective clean

help:                  ## Muestra esta ayuda
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-22s %s\n", $$1, $$2}'

pipeline: data train   ## Capa de datos y modelo de patogenicidad; ver 'core' para el ciclo completo

core: data train train-reclass train-ranking examples prioritize vus-reports monitor ## Ciclo completo: los tres modelos, priorizacion, informes y monitorizacion

data: ingest annotate dataset ## Capa de datos completa (RAW -> SILVER -> GOLD)

setup:                  ## Instala dependencias en un entorno virtual local
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

up:                     ## Levanta MLflow + la aplicacion (Docker)
	docker compose up -d --build

down:                   ## Detiene los contenedores
	docker compose down

lint:                   ## Analisis de estilo y seguridad con ruff
	ruff check src tests

test:                   ## Ejecuta la suite de pruebas
	pytest -q

ingest:                 ## Capa de datos: descarga de las releases de ClinVar
	python -m src.ingest.download

annotate:               ## Capa de datos: anotacion multi-fuente de variantes (SILVER)
	python -m src.annotate.annotate

dataset:                ## Capa de datos: objetivo binario y particion temporal (GOLD)
	python -m src.features.build_dataset

eda:                    ## Ejecuta el notebook de analisis exploratorio y regenera figuras
	jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda_variantes.ipynb

train:                  ## Modelo de patogenicidad: entrenamiento y registro en MLflow
	python -m src.train.train

train-reclass:          ## Modelo de potencial de reclasificacion de VUS
	python -m src.train.train_reclass

train-ranking:          ## Objetivo de ranking (LightGBM lambdarank, evaluado con NDCG)
	python -m src.train.train_ranking

serve:                  ## Servicio de inferencia REST y dashboard (:8000)
	python -m src.serve.app

examples:               ## Documenta las variantes de prueba del servicio
	python -m src.serve.examples

prioritize:             ## Prioriza las VUS reservadas por riesgo estimado
	python -m src.serve.prioritize_vus

vus-reports:            ## Informes por VUS generados por plantilla
	python -m src.serve.vus_reports

monitor:                ## Informe de deriva entre releases (propio y Evidently)
	python -m src.monitor.drift_report

retrain:                ## Evalua la deriva y recomienda reentrenar (simulacion)
	python -m src.monitor.retrain

ingest-prospective:     ## Descarga la release prospectiva de ClinVar
	python -m src.ingest.download --prospective

validate-prospective:   ## Validacion temporal prospectiva del modelo de reclasificacion
	python -m src.train.train_reclass --prospective

clean:                  ## Limpia artefactos locales regenerables
	rm -rf mlruns mlartifacts .pytest_cache .ruff_cache
