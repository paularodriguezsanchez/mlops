# Plataforma MLOps para la priorización de variantes genéticas de significado incierto

Plataforma de aprendizaje automático que prioriza variantes de significado incierto (**VUS**) para dirigir la revisión clínica manual. Puntúa cada VUS por probabilidad de patogenicidad y por potencial de reclasificación próxima, traduce esa puntuación a un lenguaje de evidencia inspirado en los criterios ACMG/AMP, y cierra el ciclo con monitorización de deriva entre *releases* de ClinVar y reentrenamiento sujeto a aprobación humana.

Este repositorio contiene el código, la configuración y los artefactos de la plataforma. Es el que referencia la memoria del Trabajo Fin de Máster para reproducir los resultados.

**Autora:** Paula Rodríguez Sánchez · **Director:** Sergio Pérez Iglesias
Máster Universitario en Análisis de Datos Masivos, Universidad Europea de Madrid, curso 2025-2026.

## Alcance

Variantes de un solo nucleótido (SNV) de los cromosomas 1 a 3, construidas exclusivamente sobre bases públicas agregadas, sin ninguna capa de datos de paciente.

No reclasifica variantes que ClinVar ya ha resuelto, no sustituye la curación clínica experta ni una clasificación ACMG/AMP certificada, y no compite con herramientas de priorización fenotípica como Exomiser o Talos, que requieren datos de paciente fuera de este alcance. Toda la evidencia que genera el sistema lleva el sufijo `-like` para que no se confunda con una clasificación certificada.

## Componentes

1. **Anotación multi-fuente** contra `myvariant.info`, un agregador público sin registro: gnomAD, CADD, SIFT, PolyPhen-2, GERP++, phyloP, REVEL y AlphaMissense, resolviendo cada variante en notación HGVS bajo GRCh38.
2. **Modelo de patogenicidad**: clasificación binaria sobre variantes ya resueltas, comparando cuatro algoritmos y seleccionando por validación cruzada sobre el conjunto de entrenamiento.
3. **Modelo de potencial de reclasificación**: estima qué VUS tienen mayor probabilidad de resolverse pronto, entrenado sobre el par real de *releases* fechadas de ClinVar, con un umbral de fiabilidad explícito antes de presentar su salida como probabilidad utilizable.
4. **Objetivo de *ranking***: la priorización como problema de ordenación (LightGBM `lambdarank`), evaluado con NDCG@k.
5. **Explicabilidad SHAP traducida a evidencia ACMG/AMP-símil**, declarada en todo momento como heurística de apoyo.
6. **Dashboard interactivo** servido desde el propio servicio de inferencia.
7. **Informes automáticos por VUS**, generados por plantilla y no por un modelo de lenguaje libre, para evitar texto inventado en un contexto clínico.
8. **Monitorización y reentrenamiento gobernado**: deriva de covariables y de reclasificación entre *releases*, con recomendación automática pero ejecución y promoción sujetas siempre a una acción humana explícita.

## Fuentes de datos

Todas públicas y sin registro previo, consultadas a través de `myvariant.info`, que las agrega en una sola llamada.

| Fuente | Aporta |
|---|---|
| ClinVar (NCBI) | Significancia clínica y reclasificaciones fechadas entre *releases* |
| gnomAD | Frecuencia poblacional |
| CADD, REVEL, AlphaMissense | Puntuaciones *in silico* de patogenicidad |
| GERP++, phyloP | Conservación evolutiva |
| SIFT, PolyPhen-2 | Predictores clásicos de impacto funcional |

**dbNSFP** se descartó por exigir un registro académico previo. **SpliceAI** se intentó y se descartó con evidencia: el servicio de consulta del Broad Institute resultó inaccesible desde el entorno de desarrollo y la ejecución local del paquete no era viable dentro del alcance (`docs/adr/ADR_008_reclasificacion_leakage_temporal_y_review_status.md`).

## Reproducción

Python 3.12 y Docker. `requirements.txt` fija las dependencias directas;
`requirements.lock.txt` fija además las 198 transitivas con su huella
criptográfica, generado con `pip-compile --generate-hashes`. Para reproducir el
entorno exacto de la ejecución canónica:

```bash
python -m venv .venv && .venv/Scripts/activate   # o source .venv/bin/activate
pip install --require-hashes -r requirements.lock.txt
```

En Windows hace falta además `PYTHONUTF8=1`: MLflow escribe emojis en la salida
estándar y la consola en cp1252 aborta el entrenamiento al cerrar cada *run*.

El entorno es reproducible bit a bit desde ese fichero, pero la anotación se
consulta a una API externa mutable: el *pipeline* es reproducible **sujeto al
estado de las fuentes externas**, no garantiza los mismos valores al cabo del
tiempo.

```bash
make up
```

Levanta MLflow (interfaz en `http://localhost:5000`) y el contenedor de trabajo.

```bash
make core
```

Ejecuta la secuencia completa: ingesta y anotación de las dos *releases*, entrenamiento de los tres modelos, priorización de VUS, informes automáticos y monitorización. Requiere acceso de red al NCBI y a `myvariant.info`.

```bash
make serve
```

Arranca el servicio REST y el dashboard en el puerto 8000.

```bash
make test
```

Ejecuta la suite completa de pruebas.

`make pipeline` ejecuta solo la capa de datos y el modelo de patogenicidad. La validación temporal prospectiva se reproduce aparte con `make ingest-prospective && make validate-prospective`. `make help` lista todos los objetivos.

## Seguridad del servicio

Por defecto el servicio REST no exige autenticación, pensado para uso local. Para exigirla —obligatorio antes de exponerlo públicamente— define la variable de entorno `TFM_API_KEY` y envía la cabecera `X-API-Key` en cada petición. La ruta `/health` queda siempre abierta, uso estándar para sondas de disponibilidad.

## Estructura

```
config/    Configuración centralizada (config.yaml)
data/      raw (inmutable, versionado con DVC) / interim / processed
src/       ingest, annotate, features, train, evaluate, serve, monitor
tests/     Suite de pruebas (pytest)
docs/      Informe técnico, model cards, datasheet y ADRs
cloud/     Diseño de la extensión a GCP: BigQuery, Cloud Run e IaC
docker/    Imágenes de la aplicación y de MLflow
reports/   Artefactos generados: entrenamiento, servicio y monitorización
notebooks/ Análisis exploratorio
```

## Documentación

* `docs/INFORME_TECNICO_COMPLETO.md` — recorrido de extremo a extremo de lo construido, cómo se ejecuta y en qué orden.
* `docs/MODEL_CARD.md`, `docs/MODEL_CARD_RECLASSIFICATION.md`, `docs/MODEL_CARD_RECLASSIFICATION_PROSPECTIVE.md` — métricas y limitaciones de cada modelo.
* `docs/datasheet.md` — origen, composición y taxonomía de etiquetas del conjunto de datos.
* `docs/adr/` — decisiones de arquitectura, cada una con su contexto, alternativas y consecuencias.
* `docs/EDA_variantes.md` — análisis exploratorio; `docs/retraining_flow.md` — ciclo de monitorización y reentrenamiento.
* `docs/serving_examples.md`, `docs/vus_priorizadas.md`, `docs/vus_informes_test.md` — salidas del servicio, regeneradas por el pipeline.
* `docs/FASE_II_DISENO_GCP.md` — diseño de la extensión a Google Cloud Platform, documentado y no ejecutado.

## Licencia

MIT. Ver `LICENSE`.
