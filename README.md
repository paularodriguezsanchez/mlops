# Plataforma MLOps para la priorización de variantes genéticas de significado incierto

Plataforma de aprendizaje automático que prioriza variantes genéticas de significado incierto (**VUS**, *Variants of Uncertain Significance*) para dirigir la revisión clínica manual. Puntúa cada VUS por probabilidad de patogenicidad y por **potencial de reclasificación próxima**, traduce esa puntuación a un lenguaje de evidencia inspirado en los criterios ACMG/AMP, y cierra el ciclo con monitorización de deriva entre *releases* de ClinVar y reentrenamiento sujeto a aprobación humana.

Este repositorio contiene el código, la configuración y los artefactos de la plataforma. Es el repositorio que referencia la memoria del Trabajo Fin de Máster para la reproducción de resultados.

**Autora:** Paula Rodríguez Sánchez · **Director:** Sergio Pérez Iglesias
Máster Universitario en Análisis de Datos Masivos, Universidad Europea de Madrid, curso 2025-2026.

## Alcance

Variantes de un solo nucleótido (SNV) de los cromosomas 1 a 3, construidas exclusivamente sobre bases de datos públicas y agregadas, **sin ninguna capa de datos de paciente** (fenotipo, pedigrí, herencia).

**Qué no hace:** no reclasifica variantes que ClinVar ya ha resuelto, no sustituye la curación clínica experta ni una clasificación ACMG/AMP certificada, y no compite con herramientas de priorización fenotípica como Exomiser o Talos, que requieren datos de paciente fuera del alcance de este trabajo. Toda la evidencia que genera el sistema lleva el sufijo `-like` precisamente para que no se confunda con una clasificación certificada.

## Componentes

1. **Anotación multi-fuente** contra `myvariant.info`, un agregador público de libre acceso: gnomAD, CADD, SIFT, PolyPhen-2, GERP++, phyloP, REVEL y AlphaMissense, resolviendo cada variante en notación HGVS bajo el ensamblaje GRCh38.
2. **Modelo de patogenicidad**: clasificación binaria patogénica/benigna sobre variantes ya resueltas, comparando cuatro algoritmos y seleccionando por validación cruzada sobre el conjunto de entrenamiento.
3. **Modelo de potencial de reclasificación**: estima qué VUS tienen mayor probabilidad de resolverse pronto, entrenado sobre el par real de *releases* fechadas de ClinVar (antigua/nueva), con un umbral de fiabilidad explícito antes de presentar su salida como probabilidad utilizable.
4. **Objetivo de *ranking***: la priorización como problema de ordenación (LightGBM `lambdarank`), evaluado con NDCG@k además de las métricas de clasificación.
5. **Explicabilidad SHAP → evidencia ACMG/AMP-símil**: traduce las contribuciones del modelo a códigos de evidencia reconocibles por un genetista clínico, declarados en todo momento como heurística de apoyo.
6. **Dashboard interactivo** servido desde el propio servicio de inferencia.
7. **Informes automáticos por VUS**, generados por plantilla (no por un modelo de lenguaje libre, para evitar texto inventado en un contexto clínico), citando la evidencia que motivó cada prioridad.
8. **Monitorización y reentrenamiento gobernado**: deriva de covariables y de reclasificación entre *releases*, con recomendación automática pero ejecución y promoción siempre sujetas a una acción humana explícita.

## Fuentes de datos

Todas públicas y sin registro previo:

| Fuente | Aporta |
|---|---|
| ClinVar (NCBI) | Etiquetas de significancia clínica y reclasificaciones fechadas entre *releases* |
| gnomAD | Frecuencia poblacional |
| CADD, REVEL, AlphaMissense | Puntuaciones *in silico* de patogenicidad |
| GERP++, phyloP | Conservación evolutiva |
| SIFT, PolyPhen-2 | Predictores clásicos de impacto funcional |

Se consultan a través de `myvariant.info`, que las agrega en una sola llamada. **dbNSFP** se descartó como fuente por exigir un registro académico previo, y **SpliceAI** se intentó y se descartó con evidencia: el servicio de consulta del Broad Institute resultó inaccesible desde el entorno de desarrollo y la ejecución local del paquete no era viable dentro del alcance del trabajo (`docs/adr/ADR_008_reclasificacion_leakage_temporal_y_review_status.md`).

## Requisitos

Python 3.12 y Docker. Las versiones exactas de todas las dependencias están fijadas en `requirements.txt`.

## Reproducción

```bash
make up
```

Levanta MLflow (interfaz en `http://localhost:5000`) y el servicio de la aplicación.

```bash
make core
```

Ejecuta la secuencia completa: ingesta y anotación de las dos *releases* de ClinVar, entrenamiento de los tres modelos, priorización de VUS, informes automáticos y monitorización de deriva. Requiere acceso de red a los servidores del NCBI y a `myvariant.info`.

```bash
make serve
```

Arranca el servicio de inferencia REST y el dashboard.

```bash
make test
```

Ejecuta la suite completa de pruebas automatizadas.

`make pipeline` ejecuta solo la capa de datos y el modelo de patogenicidad; no reproduce el dashboard ni los informes por VUS. La validación temporal prospectiva contra una tercera *release* se reproduce aparte con `make ingest-prospective && make validate-prospective`. `make help` lista todos los objetivos disponibles.

## Seguridad del servicio

Por defecto el servicio REST (`/predict`, `/dashboard`) no exige autenticación, pensado para uso local. Para exigirla —obligatorio antes de exponerlo públicamente— define la variable de entorno `TFM_API_KEY` y envía la cabecera `X-API-Key` en cada petición. La ruta `/health` queda siempre abierta, uso estándar para sondas de disponibilidad.

## Estructura

```
config/    Configuración centralizada del proyecto (config.yaml)
data/      raw (inmutable, versionado con DVC) / interim / processed
src/       ingest, annotate, features, train, evaluate, serve, monitor
tests/     Suite de pruebas (pytest)
docs/      Documentación técnica: informe completo, model cards, datasheet y ADRs
cloud/     Diseño de la extensión a GCP: BigQuery, Cloud Run e IaC
docker/    Imágenes de la aplicación y de MLflow
reports/   Artefactos generados: entrenamiento, servicio y monitorización
notebooks/ Análisis exploratorio
```

## Documentación

* `docs/INFORME_TECNICO_COMPLETO.md` — visión de extremo a extremo de todo lo construido, cómo se ejecuta y en qué orden.
* `docs/INTRODUCCION.md` — explicación no técnica del problema y del enfoque.
* `docs/MODEL_CARD.md`, `docs/MODEL_CARD_RECLASSIFICATION.md` y `docs/MODEL_CARD_RECLASSIFICATION_PROSPECTIVE.md` — detalle, métricas y limitaciones de cada modelo.
* `docs/datasheet.md` — origen, composición y taxonomía de etiquetas del conjunto de datos.
* `docs/adr/` — decisiones de arquitectura, cada una con su contexto, alternativas consideradas y consecuencias.
* `docs/FASE_II_DISENO_GCP.md` — diseño de la extensión a Google Cloud Platform, documentado y no ejecutado por decisión deliberada.

## Licencia

Publicado bajo licencia MIT. Ver `LICENSE`.
