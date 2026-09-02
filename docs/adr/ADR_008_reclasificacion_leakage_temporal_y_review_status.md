# ADR 008: Cierre del escenario RECOMENDADO del informe de viabilidad MuSA — features de review status de ClinVar, mitigación del leakage temporal del modelo de reclasificación, y cierre de SpliceAI

**Fecha:** 2026-08-12 · **Estado:** **Confirmado** e implementado.

## Contexto

Un análisis de viabilidad (solo lectura) sobre incorporar componentes tipo MuSA (VEP, CADD, AlphaMissense, SpliceAI, dbNSFP, RENOVO, GeneBe, InterVar) concluyó que la mayor parte de MuSA ya está cubierta por el núcleo del proyecto (ADR 007) y que el valor añadido real no estaba en sumar más anotadores, sino en tres piezas concretas de menor coste y mayor impacto directo sobre el modelo de reclasificación (modelo de reclasificación):

1. Reparar o cerrar de forma justificada la integración de SpliceAI (documentada como NaN desde ADR 007, nunca resuelta).
2. Añadir el estado de revisión de ClinVar (`CLNREVSTAT`) como feature del modelo de reclasificación: es un predictor con respaldo en la literatura de reclasificación de variantes (más revisión/consenso entre submitters → más probable que una VUS se resuelva) que el pipeline actual no extrae en absoluto del VCF, pese a estar ya disponible sin coste adicional.
3. Un hallazgo nuevo del propio análisis, no documentado hasta ahora en el proyecto: las features de anotación multi-fuente (CADD, REVEL, AlphaMissense, gnomAD, SIFT, PolyPhen, GERP, phyloP) se consultan a myvariant.info **en el momento de ejecución del pipeline**, sin ningún anclaje a la fecha de la release de ClinVar. Esto es un riesgo de fuga de información del futuro hacia el pasado específicamente relevante para el modelo de reclasificación, cuyo objetivo es predecir con información de t0 si una VUS se resolverá en t1.

Este ADR documenta las tres decisiones y su implementación. No se ha tocado RENOVO/GeneBe/InterVar/VEP: el propio informe de viabilidad concluyó que no aportan valor suficiente frente al coste/riesgo de licencia para el alcance de este TFM (quedan como comparadores bibliográficos en el estado del arte, no como dependencias técnicas).

## 1. SpliceAI: investigación y cierre

Se investigó, con evidencia reproducible, si la ejecución **local** del paquete `spliceai` (alternativa a la API de Broad, ya documentada como inaccesible desde ADR 007) era viable en este entorno:

* **Conectividad a la API de Broad:** confirmada de nuevo como no accesible (`curl` con timeout de 10s a `spliceailookup-api.broadinstitute.org` → `HTTP 000`, `exit 28` = timeout). Coherente con lo ya documentado en `multi_source.py`.
* **Dependencias del paquete `spliceai` (PyPI 1.3.1):** `keras`, `tensorflow`/`tensorflow-gpu`, `pysam`, `pyfaidx`, `numpy`, `pandas` (metadata verificada vía `pypi.org/pypi/spliceai/1.3.1/json`).
* **Bloqueo real: `pysam` no publica wheels para Windows.** Verificado contra el índice de PyPI (`pip index versions pysam`, sin artefactos `win`): `pysam` depende de `htslib`/`samtools` vía extensiones C con soporte oficial solo en Linux/macOS. Sin wheel, requeriría compilar desde código fuente con un toolchain C completo — no realista en el entorno Windows nativo de este proyecto.
* **Alternativa: Docker.** El proyecto ya usa contenedores Linux (`docker/app.Dockerfile`, `python:3.12-slim`) donde `pysam` sí tiene wheels. Pero el daemon de Docker Desktop no estaba activo en esta revisión, y aun activándolo, `spliceai` requiere además un genoma de referencia GRCh38 completo (fichero FASTA, del orden de varios GB) más los ficheros de anotación de genes — una descarga que no es razonable acometer dentro del alcance de esta revisión, y que además solo se paga una vez pero pesa de forma permanente en el repositorio/entorno de cualquier reproducción futura.

**Decisión:** no se persigue la ejecución local de SpliceAI en este TFM. Se documenta como intentado y descartado con evidencia (no como "pendiente" sin más), y se revisa la estimación de esfuerzo del informe de viabilidad: de MEDIA a **ALTA en Windows nativo / MEDIA-ALTA en Docker** (por el genoma de referencia), no MEDIA sin matices. El código existente (`src/annotate/multi_source.py::fetch_spliceai`, `include_spliceai=False` por defecto) ya degrada de forma explícita a NaN documentado — se mantiene sin cambios, es el comportamiento correcto dado este cierre. Se registra como trabajo futuro en el trabajo futuro del proyecto, condicionado a disponer de un entorno Linux con más tiempo/espacio de almacenamiento dedicado.

## 2. Review status de ClinVar como feature del modelo de reclasificación

**Qué se añade:** `review_status` (categórica, texto crudo de `CLNREVSTAT`) y `review_stars` (numérica, 0-4, escala oficial de ["gold stars" de ClinVar](https://www.ncbi.nlm.nih.gov/clinvar/docs/review_status/)), extraídos directamente del VCF de cada release (`src/annotate/annotate.py::parse_clinvar_vcf`, `src/annotate/schema.py::REVIEW_STATUS_STARS`/`review_stars`).

**Hallazgo durante la implementación, no anticipado:** al inspeccionar directamente los dos VCF reales del proyecto (`data/raw/clinvar_2023-12.vcf.gz`, `data/raw/clinvar_2025-06.vcf.gz`), se confirmó que **ClinVar cambió su vocabulario de `CLNREVSTAT` entre ambas releases** — p. ej. `criteria_provided,_conflicting_interpretations` (2023-12) pasa a `criteria_provided,_conflicting_classifications` (2025-06), y `no_assertion_provided` pasa a incluir también `no_classification_provided`. `REVIEW_STATUS_STARS` mapea explícitamente ambos vocabularios a la misma escala de estrellas; un valor no reconocido (p. ej. una futura release con terminología nueva otra vez) se deja `NaN` documentado, nunca un supuesto silencioso — mismo criterio que el resto del contrato de datos (`schema.validate_annotated`).

**Por qué solo en el modelo de reclasificación, no en los tres modelos:** `review_stars` no se añade a `FEATURE_COLUMNS` (compartido por el modelo de patogenicidad patogenicidad y el objetivo de ranking ranking), sino a un nuevo `RECLASS_FEATURE_COLUMNS = FEATURE_COLUMNS + ["review_stars"]` (`src/features/preprocess.py`), usado únicamente por el modelo de reclasificación (`src/train/train_reclass.py`, `src/serve/vus_reports.py`). Dos motivos:

1. **Para el modelo de patogenicidad** (predecir la propia significancia clínica), el estado de revisión está confundido con la certeza de la etiqueta que se intenta predecir — un argumento casi definicional, distinto del leakage temporal de §3, pero igualmente indeseable: no es una entrada causal de la biología de la variante.
2. **Para el modelo de reclasificación** (predecir si una VUS se resolverá), el estado de revisión en t0 es una entrada legítima, con respaldo en la literatura de reclasificación de VUS, y no cambia el alcance ya validado de los tres modelos.

**Implementación:** `build_reclass_preprocessor` (`src/features/preprocess.py`) extiende `build_preprocessor` con una rama adicional para `review_stars` (imputación por mediana + indicador de ausencia + escalado, mismo patrón que el resto de numéricas). `src/train/train_reclass.py` y `src/serve/vus_reports.py` actualizados para usar `RECLASS_FEATURE_COLUMNS`/`build_reclass_preprocessor` en vez de los compartidos con los tres modelos.

## 3. Leakage temporal de las features de anotación en el modelo de reclasificación — hallazgo y mitigación parcial

### El problema

`annotate_multi_source` consulta myvariant.info **en el momento de ejecución del pipeline**, sin ningún parámetro de fecha/versión. El mismo snapshot "de hoy" se aplica tanto a las variantes de la release t0 (train, 2023-12) como a las de t1 (test, 2025-06). No existe ningún "dbNSFP congelado a fecha t0".

Para el modelo de reclasificación, cuyo objetivo es predecir con información de t0 si una VUS se resolverá en t1, esto es un problema real y no solo teórico: si una VUS se reclasificó entre t0 y t1 precisamente porque llegó evidencia computacional nueva (p. ej. AlphaMissense, publicado en 2023, incorporándose como evidencia PP3/BP4 a muchas VUS con el tiempo), el modelo entrena con el valor **posterior** de esa misma evidencia — no con el que existía cuando la VUS seguía sin resolver en t0. Es fuga de información del futuro hacia el pasado.

la revisión técnica del proyecto ya reconocía que "myvariant.info sirve lo que tenga indexado en el momento de la consulta, que puede cambiar entre ejecuciones", pero lo enmarcaba como problema de reproducibilidad entre ejecuciones del pipeline, no como fuga train→predicción del modelo de reclasificación. Ningún documento del proyecto (revisión interna del proyecto) trazaba esta implicación causal hasta este ADR.

### Clasificación de features por riesgo (ver informe de viabilidad, §6)

| Feature | Riesgo | Motivo |
|---|---|---|
| `consequence`, `review_status`/`review_stars` | Seguro temporalmente | Se leen del VCF fechado de cada release, sin re-consulta "de hoy" |
| `sift_score`, `polyphen_score`, `gerp_rs`, `phylop` | Riesgo bajo | Scores relativamente estables desde hace más de una década |
| `cadd_phred`, `revel_score`, `gnomad_af` | Riesgo de leakage | Se recalculan/reindexan con el tiempo |
| `alphamissense_score` | Leakage probable | Publicado en 2023, posterior a t0; es precisamente el tipo de evidencia que motiva reclasificaciones PP3/BP4 |

### Mitigación aplicada: ablación "temporalmente segura"

No es viable, en el alcance de esta revisión, congelar snapshots históricos de CADD/REVEL/AlphaMissense/gnomAD anclados a la fecha real de cada release (requeriría descargas versionadas por fuente que no están garantizadas como archivadas y públicamente disponibles para todas las fuentes — trabajo futuro, no bloqueante). En su lugar, se implementa una mitigación parcial honesta: **cuantificar** cuánta señal del modelo de reclasificación depende de las features con riesgo, en vez de solo documentar el riesgo en prosa.

`src/train/train_reclass.py::_run_safe_ablation` entrena, sobre el **mismo split train/holdout** que el modelo completo, un modelo idéntico en algoritmo pero restringido a `SAFE_RECLASS_FEATURE_COLUMNS = ["consequence", "review_stars"]` (`build_safe_reclass_preprocessor`). El resultado se registra en MLflow (run separado, tag `leakage_ablation=temporally_safe_only`), en `models/reclassification_model/metrics.json` (`leakage_ablation_temporally_safe`), y en una nueva sección de `docs/MODEL_CARD_RECLASSIFICATION.md` ("Leakage temporal de las features"), con una interpretación automática según la brecha entre ambos ROC AUC: si la ablación se queda cerca del azar, se avisa explícitamente de que el modelo completo puede depender en gran medida de las features de riesgo y su ROC AUC debe leerse con esa salvedad.

### Qué NO se hizo (documentado, no una carencia oculta)

* No se congelaron snapshots históricos de CADD/REVEL/AlphaMissense/gnomAD por release — la mitigación completa, no solo parcial.
* No se corrigió `annotate_multi_source` para exigir una fecha de referencia — cambiaría el comportamiento de producción de forma no trivial (qué fuente usar para congelar cada release) y excede el alcance de este ADR.
* No se re-entrenó el modelo de patogenicidad sin `alphamissense_score` u otras features de riesgo: el modelo de patogenicidad no predice un cambio de estado en el tiempo (predice la etiqueta ya fijada de cada release), así que el mismo argumento de fuga temporal no aplica con la misma fuerza — se mantiene fuera de alcance, documentado aquí para que quede explícito por qué no se tocó.

## Decisión final

* SpliceAI: cerrado como "intentado, no viable en este entorno" con evidencia verificable, no como pendiente sin más.
* `review_status`/`review_stars`: implementado como feature del modelo de reclasificación únicamente, con contrato de datos actualizado (`schema.py`) y mapeo robusto a los dos vocabularios reales observados en el VCF.
* Leakage temporal del modelo de reclasificación: documentado con precisión causal (no solo "puede haber ruido de versión") y mitigado parcialmente vía ablación cuantitativa, con el resultado citado explícitamente en la Model Card antes de cualquier cifra del modelo de reclasificación en la memoria.
* RENOVO/GeneBe/InterVar/VEP: sin cambios respecto a ADR 007 — quedan como comparadores bibliográficos, no como dependencias técnicas (ver informe de viabilidad, §3-4).
