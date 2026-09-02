# Informe técnico

Recorrido de extremo a extremo de la plataforma: qué existe, por qué existe, cómo se ejecuta y en qué orden. No sustituye a los ADR, que justifican cada decisión; los sintetiza y los ordena por flujo de ejecución.

Salvo indicación contraria, todas las cifras proceden de una única ejecución completa del pipeline con datos reales de ClinVar y `myvariant.info`, registrada en MLflow y con su procedencia exacta en `reports/provenance.json`. Es la misma ejecución que cita la memoria.

---

## 1. El problema y las dos preguntas de ML

ClinVar clasifica cada variante en cinco categorías; la de significado incierto (VUS) es un cuello de botella real, porque crece más rápido que la capacidad de revisión manual experta. El sistema toma cada VUS reservada, la puntúa por probabilidad de patogenicidad y por probabilidad de resolverse pronto, explica esa puntuación en un lenguaje reconocible clínicamente, y la sitúa en una lista ordenada para que un revisor decida qué mirar primero.

El proyecto nació como un pipeline de anotación y clasificación. Cuando confirmé que dbNSFP exigía un registro académico que no llegaba a tiempo y que sin sus columnas el entrenamiento era literalmente imposible (cero columnas con datos, no señal débil), tuve que resolver de todos modos con qué anotar. Esa investigación abrió la puerta a repensar el foco: el elemento diferencial dejó de ser la anotación y pasó a ser el motor de priorización de VUS, con la anotación como etapa que lo alimenta (ADR 007).

Los dos modelos principales responden preguntas distintas:

| | Pregunta | Precedente |
|---|---|---|
| **Patogenicidad** | ¿Es dañina esta variante? Clasificación binaria sobre variantes ya resueltas | Abundante: REVEL, CADD, RENOVO |
| **Potencial de reclasificación** | ¿Se va a resolver pronto esta VUS concreta? Entrenado sobre dos *releases* fechadas como par supervisado | Sin precedente exacto en la revisión realizada (ADR 007 §4) |

Un tercer modelo reformula el problema de clasificar a ordenar, más fiel a lo que se necesita en priorización.

**Lo que el sistema no hace:** no predice el veredicto de variantes ya resueltas en ClinVar, no sustituye la curación experta ni una clasificación ACMG/AMP certificada, y no integra datos de paciente. Trabaja siempre a nivel de variante poblacional agregada.

---

## 2. Mapa del repositorio

```
config/config.yaml            Configuración central; nada hardcodeado en el código
docs/
  INFORME_TECNICO_COMPLETO.md Este documento
  adr/ADR_001 a ADR_008       Ocho decisiones de arquitectura
  MODEL_CARD*.md              Fichas de los modelos, incluida la validación prospectiva
  datasheet.md                Ficha del dataset y taxonomía CLNSIG -> etiqueta
  EDA_variantes.md            Resumen del análisis exploratorio
  retraining_flow.md          Ciclo de monitorización y reentrenamiento
  serving_examples.md         Variantes de prueba del servicio
  vus_priorizadas.md          VUS ordenadas por prioridad
  vus_informes_test.md        Informes automáticos por VUS
  FASE_II_DISENO_GCP.md       Diseño de la extensión a GCP, no ejecutado
data/                         raw (inmutable, DVC) / interim (SILVER) / processed (GOLD)
src/                          config, ingest, annotate, features, train, evaluate, monitor, serve
models/                       best_model, reclassification_model, ranking_model
reports/                      training, monitoring, serving y provenance.json
tests/                        95 pruebas (pytest)
docker/, compose.yaml         Tres servicios: mlflow, app y serve
Makefile                      Documentación ejecutable de cada etapa
.github/workflows/ci.yml      Estilo, auditoría de dependencias y pruebas
cloud/                        Diseño de la extensión a GCP (BigQuery, Cloud Run, IaC)
```

Convención de nombres: los ficheros de `data/interim/` se nombran por *release* (`annotated_2023-12.parquet`); los de `data/processed/`, por rol de partición (`train.parquet`, `vus_test.parquet`). `train` es siempre la release antigua y `test` la nueva: la partición es temporal, no aleatoria.

---

## 3. Orden de ejecución

```bash
make up                 # MLflow (UI en :5000) y contenedor de trabajo

make data               # ingest + annotate + dataset (RAW -> SILVER -> GOLD)
make train              # modelo de patogenicidad
make train-reclass      # modelo de potencial de reclasificación
make train-ranking      # objetivo de ranking (lambdarank, NDCG@k)

make serve              # API REST y dashboard en :8000
make examples           # variantes de prueba documentadas
make prioritize         # ordena las VUS reservadas
make vus-reports        # informes por VUS; alimenta el dashboard

make monitor            # informe de deriva
make retrain            # evalúa la deriva y recomienda; no reentrena
python -m src.monitor.retrain --execute   # aprueba y ejecuta el reentrenamiento

make ingest-prospective && make validate-prospective   # validación temporal prospectiva
python -m src.evaluate.compare_predictors              # modelo vs. CADD/REVEL/AlphaMissense solos
python -m src.evaluate.capture_provenance              # commit, hashes y run IDs

make test && make lint
```

El orden no es intercambiable: cada etapa consume el fichero que produce la anterior. `predictor.py` no arranca sin `models/best_model`; `vus_reports.py` degrada mostrando "no disponible" si falta el modelo de reclasificación o el de ranking, en vez de fallar; y el dashboard no calcula nada, solo lee el JSON ya generado.

Cada `make <objetivo>` es literalmente `python -m src.<paquete>.<módulo>`: el Makefile no añade lógica, solo documenta el comando canónico.

---

## 4. Configuración (`src/config.py`, `config/config.yaml`)

Punto único de acceso, bajo la regla de no dejar ninguna ruta, semilla ni umbral suelto en el código. `load_config` cachea con `lru_cache` y resuelve la ruta en cada llamada, no como valor por defecto del parámetro: de otro modo queda fijada en el import y un test que la sustituye sigue leyendo la configuración real, un fallo que detecté con un test supuestamente aislado que corría contra las 6000 variantes reales en vez de las 300 del test.

Parámetros que gobiernan el sistema: semilla 42; `clinvar_train_release` 2023-12 y `clinvar_test_release` 2025-06; `chromosomes_subset` 1-3; `max_variants_per_release` 8000 y `max_new_variants_per_release` 4000, que acotan volumen y coste de red; `annotation_source: multi_source`, con `synthetic` reservado a pruebas; las etiquetas del target; y `drift_threshold` 0,15.

---

## 5. Ingesta (`src/ingest/`)

`download.py` descarga las dos *releases* (VCF GRCh38) y escribe `data/raw/MANIFEST.json` con SHA-256 y procedencia de cada fichero. Por defecto **exige red real y falla con `RuntimeError`** si no hay conexión, en vez de sustituir en silencio por datos inventados: son variantes asociadas a enfermedades reales y un resultado sintético no es válido como resultado del trabajo. Solo `--offline` activa el generador determinista (ADR 005). Valida además que la URL use esquema `http`/`https`, nunca `file://`.

`synthetic.py` produce, sin red, ficheros con el mismo esquema que las fuentes reales, con semilla fija. Imita tres propiedades del dato real: las features correlacionan con la patogenicidad a través de una señal latente ruidosa, de modo que ningún clasificador alcance un rendimiento perfecto; la release nueva añade variantes y reclasifica una fracción de las VUS; y todo depende de una única semilla. La selección de qué VUS se reclasifican está ponderada por `|latent - 0,5|`, no es uniforme: con selección uniforme no queda señal aprendible por construcción, y el modelo de reclasificación no podría superar al azar hiciera lo que hiciera.

`data/raw/` es inmutable y se versiona con DVC.

---

## 6. Anotación (`src/annotate/`): RAW -> SILVER

**`schema.py`** define y valida el contrato: columnas requeridas, claves sin nulos ni duplicados, bases ACGT y rango plausible por feature. No es decorativo, se invoca con `strict=True` en el flujo real. El límite superior de `cadd_phred` estaba fijado a 60, calibrado sobre el techo del generador sintético, y tuve que ampliarlo a 99 al validar contra ClinVar real: las variantes nonsense llegan a 73.

**`annotate.py`** parsea el VCF por expresiones regulares sobre `INFO` (CLNSIG, GENEINFO, MC, CLNREVSTAT), filtra a SNVs descartando ALT multialélico y estructural, acota por cromosoma y volumen, cruza con las features y valida el contrato.

Su pieza más delicada es `_sample_release`, con continuidad de cohorte. ClinVar es acumulativo y tiene cientos de miles de variantes en chr1-3: muestrear cada release de forma independiente casi no deja claves compartidas por azar, lo que anula la medición de deriva. La solución fija una cohorte tras procesar la primera release y la conserva en la segunda, más un cupo explícito de variantes nuevas. Sin ese segundo cupo la cohorte retenida agota ella sola el límite y el holdout no visto queda en el 2,1 % (75 de 3532), justo el subconjunto con el que se evalúa el modelo principal. El muestreo es aleatorio simple con semilla fija, sin estratificar.

El acotado por cromosoma y volumen se aplica **antes** de anotar: con consultas de red reales, acotar solo la salida deja el proceso disparando consultas durante horas para variantes que se descartan igualmente.

**`multi_source.py`** resuelve el bloqueo que disparó el pivote. Consulta myvariant.info en lotes de hasta 1000 variantes por POST, obteniendo en una sola fuente lo que aportaba dbNSFP más AlphaMissense. Cuatro problemas concretos, todos encontrados al validar contra ClinVar real:

1. myvariant.info asume hg19; sin `assembly=hg38` explícito, las 60 variantes de prueba devolvían "notfound" pese a existir el dato.
2. Bajo hg38, CADD, SIFT y PolyPhen viven en `dbnsfp.*`, no en `cadd.*`.
3. Proyectar `dbnsfp.gerp++.rs` por nombre devuelve vacío: la API no maneja bien los nombres con "+". Se pide el objeto `dbnsfp` completo y se extrae en el cliente.
4. REVEL, SIFT, PolyPhen-2, CADD, GERP y phyloP pueden llegar como lista, una entrada por transcrito, no solo AlphaMissense. Se promedian todos.

SpliceAI se intentó vía el servicio del Broad Institute; no accesible desde el entorno de desarrollo. Degrada a `NaN`, avisando una sola vez por proceso.

Cobertura validada sobre muestra real (chr1-3, missense, n=100): CADD 100 %, phyloP 100 %, AlphaMissense 98 %, GERP 99 %, SIFT 96 %, REVEL 95 %, PolyPhen 90 %, gnomAD AF 72 %.

---

## 7. Dataset (`src/features/build_dataset.py`): SILVER -> GOLD

Binariza el target y separa temporalmente train (2023-12) y test (2025-06).

**Las VUS no se descartan.** Se persisten en un fichero paralelo por partición, sin columna `label`, como conjunto de inferencia realista. Incluirlas en el entrenamiento inyectaría ruido de etiqueta —por definición no tienen ground truth fiable—, pero en la práctica clínica no se tiran, se reevalúan (ADR 006).

**Taxonomía de etiquetas.** VUS se define de forma estricta como `clnsig == "Uncertain_significance"`. Antes, `build_dataset.py` metía también clasificaciones conflictivas y otras categorías ambiguas en la población de VUS, mientras `drift_report.py` usaba por su cuenta el literal estricto: una divergencia no documentada que era la causa exacta de una inconsistencia numérica real (67 reclasificaciones citadas para el modelo de reclasificación frente a 54 para el monitor, en la misma transición). Con la definición unificada ambas cifras coinciden en **55**. Las variantes que no son positivas, negativas ni VUS estricta se persisten aparte en `excluded_{split}.parquet`, por trazabilidad. La tabla exhaustiva CLNSIG -> bucket se genera en cada ejecución (`reports/training/clinvar_label_taxonomy.csv`) y se reproduce en `docs/datasheet.md`.

---

## 8. Preprocesamiento (`src/features/`)

`preprocess.py` construye el `ColumnTransformer` compartido por los tres entrenamientos, embebido dentro del `Pipeline` que se registra en MLflow, de modo que entrenamiento y serving apliquen exactamente la misma transformación.

| Grupo | Columnas | Tratamiento |
|---|---|---|
| Scores in silico | `sift_score, polyphen_score, revel_score, alphamissense_score` | Mediana con indicador de ausencia, luego escalado: la ausencia es informativa fuera de missense |
| Densas | `cadd_phred, gerp_rs, phylop` | Mismo tratamiento. Con ClinVar real su tasa de nulos es tan alta (~75 %) como la del grupo anterior; el supuesto de partida solo se cumplía con el generador sintético |
| Logarítmica | `gnomad_af` | `log10` recortado, mediana con indicador de ausencia, escalado. Conservar el indicador aparte, en vez de disolver la ausencia en la mediana, mejoró de forma medible el PR AUC del modelo de patogenicidad y el NDCG@10 del de ranking |
| Categórica | `consequence` | One-hot con `handle_unknown="ignore"` |

`gene` se excluye por alta cardinalidad y riesgo de fuga. `review_stars` se añade solo al modelo de reclasificación (ADR 008).

`reclassification.py` define las funciones únicas `is_resolved` e `is_vus`, para que entrenamiento, construcción del dataset y monitor no dupliquen la misma lógica. `RESOLVED_LABELS` no incluía las formas compuestas `Pathogenic/Likely_pathogenic` y `Benign/Likely_benign` pese a que la configuración sí las trata como positivas y negativas, lo que afectaba al recuento de reclasificaciones; ahora incluye las seis.

---

## 9. Los tres modelos (`src/train/`)

### 9.1 Patogenicidad (`train.py`)

Clasificación binaria sobre variantes ya resueltas, comparando `logistic_regression`, `random_forest`, `gradient_boosting` y `hist_gradient_boosting`.

El criterio de evaluación incorpora dos correcciones metodológicas independientes. La primera: el test temporal es, por construcción, un superconjunto parcial del train, porque ClinVar es acumulativo, así que evaluar sobre el test completo es optimista por memorización —en una versión temprana infló el PR AUC a 0,988—. `unseen_mask` calcula qué filas del test tienen una clave ausente del train, y ese holdout no visto (n=1764) es el conjunto de evaluación honesto. La segunda: hasta una revisión posterior, el algoritmo ganador se elegía por su PR AUC en ese mismo holdout, el que después se citaba como resultado final, de modo que dejaba de ser un test independiente. La selección usa ahora únicamente el PR AUC medio en validación cruzada de 5 particiones estratificadas sobre el conjunto de entrenamiento; el holdout se evalúa una sola vez, sobre el algoritmo ya elegido.

| Algoritmo | PR AUC CV (train) | PR AUC holdout [IC 95 %] | ROC AUC holdout | F1 holdout | PR AUC full |
|---|---|---|---|---|---|
| **logistic_regression** (ganador) | **0,9787±0,0090** | 0,9700 [0,9511, 0,9849] | 0,9945 | 0,9359 | 0,9772 |
| random_forest | 0,9750±0,0084 | 0,9657 [0,9388, 0,9858] | 0,9925 | 0,9216 | 0,9827 |
| gradient_boosting | 0,9760±0,0055 | 0,9721 [0,9556, 0,9864] | 0,9947 | 0,9212 | 0,9831 |
| hist_gradient_boosting | 0,9759±0,0090 | 0,9581 [0,9310, 0,9803] | 0,9875 | 0,9289 | 0,9809 |

`gradient_boosting` tiene un holdout algo mejor que el ganador pero pierde la selección porque su PR AUC en validación cruzada es marginalmente inferior, dentro de una desviación estándar. Es exactamente el caso que ilustra por qué seleccionar y evaluar sobre el mismo conjunto es un sesgo real: con el criterio correcto, el ganador no coincide con el que más alto puntúa en la columna que un lector desprevenido miraría primero.

**Frente a los predictores in silico en solitario** (`compare_predictors.py`, IC bootstrap pareado de la diferencia, 1000 remuestreos, sobre las mismas variantes del holdout): supera a CADD solo (0,986 frente a 0,935 en las 379 variantes con CADD disponible, diferencia [0,026, 0,079]), a REVEL solo (0,970 frente a 0,918 en 252 missense, diferencia [0,024, 0,087]) y a AlphaMissense solo (0,968 frente a 0,910 en 254 missense, diferencia [0,022, 0,106]). Ninguna diferencia cruza cero.

Calibración sobre el mismo holdout: Brier score 0,0122, con la tabla por deciles en `docs/MODEL_CARD.md`.

Se registra en MLflow (experimento `variant_pathogenicity`, un run por algoritmo) y el mejor pasa al Model Registry en stage **Staging**, nunca directo a Production.

Importancia de features por SHAP: `cadd_phred` (0,077) > `gnomad_af` (0,060) > `phylop` (0,057) > `gerp_rs` (0,057) > `alphamissense_score` (0,040) > `consequence` (0,029) > `sift_score` (0,022) > `revel_score` (0,018) > `polyphen_score` (0,013).

### 9.2 Potencial de reclasificación (`train_reclass.py`)

Dada una VUS de la release antigua, predecir si estará resuelta en la nueva. Es la aportación más genuina del proyecto: explota dos *releases* fechadas con reclasificaciones reales conocidas como par de entrenamiento supervisado directo.

Población: **4056** VUS de 2023-12, de las que **55** (1,4 %) se reclasificaron en 2025-06.

| Algoritmo | PR AUC | ROC AUC | F1 |
|---|---|---|---|
| **logistic_regression** (ganador) | **0,1093** | 0,7786 | 0,0000 |
| random_forest | 0,0668 | 0,7703 | 0,0000 |
| gradient_boosting | 0,0746 | 0,8065 | 0,0000 |
| hist_gradient_boosting | 0,0717 | 0,7178 | 0,0000 |

Los cuatro algoritmos dan F1=0 al umbral 0,5: ninguno produce una sola predicción positiva correcta a ese umbral. Es justo el motivo por el que la selección usa PR AUC, que no depende de un umbral fijo, y por el que existe además un umbral de fiabilidad independiente por ROC AUC antes de servir cualquier probabilidad sin aviso.

Un PR AUC de 0,109 parece bajo en absoluto, pero la clase positiva es el 1,4 %: el valor esperado de un clasificador aleatorio coincide con la prevalencia, así que la cifra está varias veces por encima de esa referencia. El holdout tiene 14 positivos sobre n=1014, y el IC 95 % bootstrap del PR AUC es, en consecuencia, ancho: [0,033, 0,273]. Más informativas para el uso real —ordenar una cola de revisión— son las métricas de cola: **precision@10=0,20 y lift@10≈14,5**, degradando hacia precision@100=0,07 (lift≈5,1). Brier score 0,013, con la salvedad de que 14 positivos no permiten validar la calibración.

El umbral de fiabilidad `RELIABLE_ROC_AUC_THRESHOLD = 0,6` es una convención operativa mía, sin respaldo estadístico ni de la literatura; el criterio informativo es que el IC del ROC AUC, [0,649, 0,901], quede por encima de 0,5. Con ROC AUC 0,779 el modelo se marca fiable y su probabilidad se muestra sin aviso adicional; por debajo del umbral, dashboard e informes la marcan como señal débil.

**Ablación de fuga temporal.** Un modelo restringido a las dos features ancladas a la fecha real de cada release (`consequence`, `review_stars`) alcanza ROC AUC 0,7476, ligeramente por debajo del completo (0,7786, diferencia +0,031). Antes de corregir la taxonomía de VUS la ablación superaba claramente al modelo completo (0,8744 frente a 0,7300), lo que se leía como tranquilizador respecto al riesgo de fuga; con la población corregida esa lectura ya no se sostiene, y no puede descartarse que parte del rendimiento del modelo completo dependa de features con riesgo de fuga temporal (ADR 008).

### 9.3 Validación temporal prospectiva

Responde a la pregunta que ni el holdout retrospectivo ni la ablación pueden responder: ¿predice este modelo, ya entrenado, una reclasificación genuinamente futura? Descargué una tercera *release* real de ClinVar (2026-08, unos catorce meses posterior a la de test), publicada después de fijar el par de entrenamiento y nunca usada para ajustar nada. `run_prospective` toma las VUS que en 2025-06 seguían sin resolver (4001, el subconjunto realmente abierto), aplica el modelo persistido **sin reentrenar** y comprueba cuáles se resolvieron en 2026-08. Esa release solo se parsea para leer CLNSIG; no se reanota, porque las features ya están fijadas en la release de entrenamiento.

De 4001 VUS abiertas, 3986 se localizaron y solo **7 (0,17 %)** se habían resuelto. Modelo completo: **ROC AUC 0,604**, PR AUC 0,009. Ablación segura: ROC AUC 0,447, por debajo del azar.

Con 7 positivos la estimación es muy ruidosa y no permite confirmar ni refutar la capacidad prospectiva. Lo que sí es cierto es que el resultado, tal como sale, es sustancialmente más débil que el retrospectivo: la señal que el modelo captura dentro del intervalo 2023-2025 no se traslada, con la evidencia actual, a una predicción confiable sobre una release genuinamente futura. El modelo se presenta como prueba de concepto retrospectiva, con esta validación como limitación cuantificada.

### 9.4 Objetivo de ranking (`train_ranking.py`)

`LGBMRanker(objective="lambdarank")` entrenado directamente sobre el orden y evaluado con NDCG@k, que pondera la posición: penaliza más un error cerca de la cabeza de la lista, donde el revisor realmente mira. LambdaMART agrupa ítems por *query* y aquí no hay agrupación natural, así que uso un único grupo global; es una simplificación, documentada como tal.

**Hallazgo de reproducibilidad:** LightGBM con `lambdarank` no es reproducible entre ejecuciones pese al `random_state` fijo, salvo fijando `deterministic=True`, `force_row_wise=True` y `num_threads=1`: la construcción de histogramas multihilo introduce no determinismo en la reducción en coma flotante. El valor versionado era NDCG@10=1,0, un ranking perfecto, sospechoso en sí mismo; al reejecutar dio 0,7015 de forma estable. Corregido y verificado con un test de regresión que exige NDCG idéntico entre dos entrenamientos.

Resultado sobre el holdout no visto (n=1764): NDCG@10=**0,890**, NDCG@50=0,944, NDCG@100=0,966, PR AUC de referencia 0,798. Sin baseline, un NDCG alto no dice si el modelo aporta algo: el orden de llegada sin reordenar da NDCG@10=0,000 y la media de 50 órdenes aleatorios, 0,140±0,119. El margen sobre ambos es amplio.

**Limitación de proxy:** la etiqueta de relevancia es el mismo binario de patogenicidad, y tanto el entrenamiento como esta evaluación usan variantes **ya resueltas**. En producción, sin embargo, este modelo ordena **VUS**, una población distinta sobre la que no existe verdad terreno de ranking. El NDCG reportado mide la calidad del orden sobre variantes resueltas; no se ha medido, ni puede medirse hoy, sobre la cola de VUS que el sistema prioriza en la práctica.

El booster se guarda en formato nativo de LightGBM junto con el preprocesador ya ajustado, porque `LGBMRanker` no es de confianza por defecto para el serializador que usa MLflow. No se registra en el Model Registry. Desde su integración en serving es el criterio de orden real de `prioritize_vus.py` y `vus_reports.py`, con reversión explícita a la probabilidad de patogenicidad si no está entrenado.

### 9.5 Resumen

| | Patogenicidad | Reclasificación | Ranking |
|---|---|---|---|
| Tarea | Clasificación binaria | Clasificación binaria | Learning-to-rank |
| Población | Variantes resueltas | VUS de la release antigua | Variantes resueltas |
| Evaluación | Holdout temporal no visto | Holdout aleatorio estratificado, más validación prospectiva real | Holdout temporal no visto |
| Selección | PR AUC medio en CV sobre train | Igual | NDCG@10/50/100 |
| Model Registry | Sí, `variant_pathogenicity_clf` @Staging | Sí, `vus_reclassification_clf` @Staging | No |
| Consumidor | `predictor.py`, `prioritize_vus.py` | `vus_reports.py`, con aviso de fiabilidad | `prioritize_vus.py`, `vus_reports.py`: decide el orden real |

---

## 10. Evaluación y explicabilidad (`src/evaluate/`)

**`metrics.py`** reúne PR AUC, ROC AUC, F1, precisión y cobertura, deliberadamente sin exactitud, por el desbalance de clases; intervalos bootstrap del PR AUC, del ROC AUC y de la diferencia pareada entre dos puntuaciones; `precision_recall_lift_at_k` para la cola de revisión; `calibration_report` con Brier score y tabla por bins; y exportación de las curvas PR y ROC completas. NDCG vive aparte, en `train_ranking.py`, porque MLflow no admite "@" en nombres de métrica.

**`explain.py`** calcula valores SHAP sobre el pipeline completo como caja negra, con `shap.Explainer` agnóstico al modelo, sobre las columnas de entrada crudas. El masker tabular de SHAP compara con `np.isclose`, que no admite texto, así que `consequence` se codifica a enteros antes de invocarlo y se decodifica justo antes de cada llamada real al pipeline. Conserva el índice original de las filas muestreadas, para poder recuperar la identidad de cada variante sin ambigüedad.

**`acmg_evidence.py`** traduce cada contribución SHAP a un código de evidencia reconocible por un genetista, siempre con el sufijo `-like`:

| Código | Señal | Condición |
|---|---|---|
| PP3-like | CADD≥20, REVEL≥0,7, AlphaMissense≥0,7, PolyPhen≥0,85, SIFT≤0,05, GERP≥4,0, phyloP≥5,0 | Score dañino y SHAP hacia patogénica |
| BP4-like | Los mismos scores | Score no dañino y SHAP hacia benigna |
| PM2-like | gnomAD < 1e-4 | Variante rara o ausente, SHAP hacia patogénica |
| BA1-like | gnomAD > 0,01 | Demasiado común para enfermedad rara, SHAP hacia benigna |
| PVS1-like | stop_gained/nonsense, frameshift, splice donor/acceptor | Pérdida de función, SHAP hacia patogénica |

La regla de coherencia es el corazón de la heurística: solo se reporta evidencia si el valor del score y el signo de SHAP apuntan en la misma dirección. Si un score parece dañino pero el SHAP es negativo, no se reporta nada; el modelo está capturando una interacción más compleja y forzar un código unívoco sería inventar precisión.

Un detalle de normalización, determinante para el resultado: ClinVar antepone el término Sequence Ontology al campo de consecuencia (`SO:0001587|nonsense`), mientras el generador usa el término plano. Sin normalizar, PVS1-like —la regla de mayor fuerza clínica— no coincidía con ningún valor real de ClinVar, cero de 3504 filas. Se aceptan además `nonsense` (término de ClinVar) y `stop_gained` (término de VEP) como sinónimos.

No hay heurística para PS1, PM1, PM3-PM6, PP1/PP2/PP4/PP5, BS1-BS4 ni BP1-BP3/BP5-BP7: exigirían datos de segregación o funcionales que el proyecto no tiene.

**Dos análisis de sensibilidad**, fuera del pipeline canónico. `chr_representativeness.py` (sin red) parsea el VCF crudo ya descargado, que contiene el genoma completo, y compara chr1-3 (761 027 variantes) frente al resto (2 471 777) con el mismo PSI que usa el monitor entre *releases*: PSI 0,0003 en distribución de clases, 0,0019 en estado de revisión y 0,0013 en consecuencia funcional, muy por debajo del umbral de 0,1. La única diferencia estructural es que chr1-3 cubre 4108 de los 18 403 genes con variantes (22,3 %), un límite de cobertura génica, no un sesgo de selección dentro de chr1-3. `transcript_aggregation_sensitivity.py` (con red) compara agregar por media frente a por máximo: el mapeo a varios transcritos afecta a una minoría de variantes (SIFT 9,2 %, PolyPhen 7,2 %, REVEL 9,8 %, AlphaMissense 10,3 %) y la diferencia de PR AUC por validación cruzada es de +0,0001, indistinguible del ruido entre particiones.

---

## 11. Monitorización y reentrenamiento (`src/monitor/`)

**`drift.py`** aplica dos pruebas complementarias por columna numérica —Kolmogorov-Smirnov con p<0,05 y PSI con umbral 0,2— y solo PSI a las categóricas. Basta que se cumpla una para marcar drift. `drift_summary` agrega: `alert = n_drifted / n_features >= threshold`.

**`drift_report.py`** compara la release de referencia contra la actual en tres señales:

1. **Drift de covariables** sobre las nueve features de anotación.
2. **Deriva de reclasificación**: variantes que eran VUS estricta y aparecen resueltas. No se llama concept drift porque que una VUS reciba veredicto no demuestra por sí solo un cambio en P(Y|X), solo que se completó una etiqueta que faltaba. Si hay modelo disponible, calcula además su acierto sobre esas variantes, entrenado cuando estaban excluidas. Alerta si la proporción supera el 2 %; originalmente alertaba con cualquier reclasificación, demasiado laxo.
3. **Prueba de estrés**: perturbación controlada (`cadd_phred += 8`, `gnomad_af *= 0,1`, `gerp_rs += N(2, 0,5)`) que verifica que el mecanismo de alerta responde. No cuenta para la alerta global.

Genera además un informe HTML con Evidently, envuelto en `try/except`: si no está disponible se omite con aviso, sin romper el flujo.

Cifras de la ejecución canónica: drift de covariables 0/9 (sin alerta); deriva de reclasificación **55 de 7997 variantes compartidas (0,69 %)**, por debajo del umbral, con acierto del modelo **0,964**; prueba de estrés 3/9 (alerta, confirma que el mecanismo funciona); alerta global falsa. El recuento de 55 coincide exactamente con el de la población del modelo de reclasificación tras unificar la definición de VUS.

**`retrain.py`** implementa la gobernanza: en un contexto clínico promover un modelo automáticamente sería inaceptable, así que el sistema recomienda y una persona aprueba. `decide` es lógica de negocio pura, sin efectos secundarios. Sin `--execute` solo recomienda; el modelo reentrenado entra siempre en Staging, y la promoción es un segundo gate humano. No existe en el repositorio ningún cron ni acción programada que dispare un reentrenamiento sin intervención.

---

## 12. Servicio (`src/serve/`)

```
annotator.py  -> predictor.py -> app.py (REST + blueprint del dashboard)
                              -> examples.py
                              -> prioritize_vus.py -> vus_reports.py -> dashboard.py
```

**`annotator.py`** reutiliza la capa SILVER ya anotada como almacén indexado por clave; no ejecuta VEP en vivo. Nunca expone `clnsig`, que es la etiqueta, no una feature. Una variante fuera del almacén se marca como no anotable.

**`predictor.py`** anota y aplica `predict_proba` sobre `models/best_model`. `predict_batch` puntúa miles de VUS de golpe, y lo usa la priorización.

**`app.py`** expone tres rutas:

| Ruta | Método | Autenticación | Notas |
|---|---|---|---|
| `/health` | GET | Nunca | Sonda de disponibilidad |
| `/predict` | POST | Si `TFM_API_KEY` está definida | 400 ante campos ausentes, JSON inválido o `pos` no numérico; nunca 500 |
| `/dashboard[/<split>]` | GET | Si `TFM_API_KEY` está definida | Valida `split` contra `{train, test}`, 404 en cualquier otro valor |

La clave se compara con `secrets.compare_digest`, que evita ataques de temporización, y `MAX_CONTENT_LENGTH` corta payloads desproporcionados en 1 MB. Verificado con `curl` contra un servidor real: sin cabecera 401, clave incorrecta 401, clave correcta 200, `/health` abierta, payload de 2 MB 413. El servidor de desarrollo escucha en `127.0.0.1`; el `CMD` de gunicorn en Docker sí usa `0.0.0.0`, aislado por el mapeo de puertos.

**`examples.py`** documenta diez variantes de prueba muestreadas excluyendo las ya vistas en train. Resultado: **9 de 10** aciertos; el fallo es una variante missense de ATG7, patogénica real predicha benigna con probabilidad 0,105, documentado sin ocultarlo.

**`prioritize_vus.py`** ordena las VUS reservadas. `rank_vus` es una función pura, reutilizada por los informes. `load_ranking_model` carga booster y preprocesador ajustado; si existen, el orden lo decide `ranking_score`, y si no, revierte a `probabilidad_patogenica`. Resultado: **6042** VUS en la release de test; el primero (gen ARL6IP6, nonsense) alcanza `ranking_score=6,346` con probabilidad 0,997.

**`vus_reports.py`** es donde converge todo: cada informe cita probabilidad de patogenicidad, score de ranking, probabilidad de reclasificación con su aviso de fiabilidad y evidencia SHAP traducida a ACMG. Generado por plantilla y no por un modelo de lenguaje libre: en un informe que puede influir en qué VUS revisa antes un clínico, un texto inventado es un riesgo real que una plantilla con datos verificables no tiene.

**`dashboard.py`** es un blueprint Flask con tabla ordenable en JS sin dependencias, sobre los informes ya generados. No recalcula SHAP por petición ni carga el booster: lee el `ranking_score` ya presente en el JSON. La cabecera declara de forma dinámica si el orden viene del ranking o de la probabilidad, y marca como señal débil la probabilidad de reclasificación cuando no es fiable.

---

## 13. Infraestructura y calidad

Dos imágenes sobre `python:3.12-slim` y tres servicios en `compose.yaml`: `mlflow` (tracking y registro, backend SQLite, puerto 5000), `app` (shell de trabajo con bind mount del repositorio) y `serve` (`gunicorn src.serve.app:app`, puerto 8000).

La integración continua ejecuta en cada push y PR a `main`: `ruff check src tests` (bloqueante), `pip-audit` (no bloqueante, porque detecta un CVE de terceros sin corrección publicada que no debe tumbar el pipeline) y `pytest` (bloqueante). `pyproject.toml` activa los rulesets `E, F, I, W, UP, B, S` de ruff, incluido el de seguridad. `requirements.txt` fija versiones exactas.

`data/raw` se versiona con DVC, sin remoto configurado: tras clonar hay que regenerar con `make data`, que es determinista.

La suite tiene **95 pruebas** agrupadas por capa —ingesta y anotación, features, los tres entrenamientos, evaluación y procedencia, monitorización, servicio— más un test de integración que ejecuta la cadena completa sobre datos aislados en `tmp_path`, sin tocar el repositorio real. `RandomForestClassifier` y `permutation_importance` usaban `n_jobs=-1`, lo que en Windows dejaba procesos de `loky` sin cerrar entre tests sucesivos hasta degradar el sistema; corregido a `n_jobs=1`, sin alterar ningún resultado citado, porque con `random_state` fijo el modelo entrenado es idéntico.

---

## 14. Cifras de la ejecución canónica

`reports/provenance.json` registra el commit, los hashes de datos y los run IDs de MLflow de esta ejecución.

| Métrica | Valor | Fuente |
|---|---|---|
| Anotación | 2023-12: 8000 variantes · 2025-06: 11 997 · release prospectiva 2026-08 descargada, solo verdad terreno | `make annotate` |
| Dataset | train 3504 etiquetadas (prevalencia 0,150), 4056 VUS, 437 excluidas · test 5221 etiquetadas (0,137), 6042 VUS, 734 excluidas | `docs/datasheet.md` |
| Patogenicidad | logistic_regression; PR AUC holdout **0,9700** [0,9511, 0,9849], ROC AUC 0,9945, n=1764 | `docs/MODEL_CARD.md` |
| Frente a predictores solos | Mejor que CADD, REVEL y AlphaMissense; ningún IC de la diferencia cruza cero | `reports/training/compare_predictors.csv` |
| Reclasificación, retrospectivo | logistic_regression; PR AUC 0,1093, ROC AUC 0,7786; ablación segura 0,7476 | `docs/MODEL_CARD_RECLASSIFICATION.md` |
| Reclasificación, prospectivo | ROC AUC **0,604** frente a 0,447 de la ablación, sobre 7 positivos de 3986 VUS | `docs/MODEL_CARD_RECLASSIFICATION_PROSPECTIVE.md` |
| Ranking | NDCG@10=**0,890**, @50=0,944, @100=0,966; baselines 0,000 y 0,140±0,119 | `reports/training/ranking_metrics.csv` |
| Cobertura de anotación | 90-100 % salvo gnomAD AF, 72 % | validación de `multi_source` |
| Servicio | 9 de 10 variantes no vistas clasificadas correctamente | `docs/serving_examples.md` |
| Monitorización | covariables 0/9; reclasificación 55/7997 (0,69 %), acierto 0,964; estrés 3/9; alerta global falsa | `reports/monitoring/drift_summary.json` |
| Pruebas | 95 en verde | `make test` |
| VUS priorizadas | 6042 en la release de test | `docs/vus_priorizadas.md` |

Documentos de ejecuciones anteriores citan cifras ligeramente distintas: son fotos de otros momentos, no errores. La fuente única es siempre la ejecución canónica registrada en `reports/provenance.json`.

---

## 15. Estado y pendientes

El núcleo está completo, probado y revisado. Queda pendiente, sin ser bloqueante:

* Ejecutar realmente la extensión a GCP, hoy documentada en `docs/FASE_II_DISENO_GCP.md` y `cloud/`.
* Ampliar la validación prospectiva con más *releases*: el experimento actual tiene 7 positivos, insuficiente para una conclusión firme.
* Congelar snapshots históricos de CADD, REVEL, AlphaMissense y gnomAD anclados por fecha de release, para cerrar por completo el riesgo de fuga temporal.
* Evaluar el modelo de ranking sobre la propia población de VUS que prioriza, en cuanto exista verdad terreno de prioridad clínica.

Descartado con motivo, no por falta de tiempo: integración con Talos, Exomiser, LIRICAL o Seqr, y modelado de tríos y herencia, porque requieren datos de paciente; arquitecturas GNN o *transformer*, porque no hay estructura de grafo ni de secuencia en un problema tabular como este; y PrimateAI-3D o EVE como fuente primaria, por acceso restringido y cobertura limitada.

---

## 16. Glosario

- **VUS:** variante cuya patogenicidad no se ha podido determinar con la evidencia disponible.
- **ClinVar:** archivo público del NCBI con la interpretación clínica de variantes, publicado en *releases* fechadas.
- **ACMG/AMP:** criterios estándar (Richards et al., 2015) para clasificar formalmente una variante. Exigen evidencia que este proyecto no tiene, de ahí el sufijo `-like`.
- **SNV:** cambio de una sola base; el único tipo de variante cubierto.
- **gnomAD:** base de frecuencias alélicas poblacionales.
- **CADD, REVEL, AlphaMissense, SIFT, PolyPhen, GERP, phyloP:** puntuaciones computacionales de deletereidad o conservación.
- **PR AUC:** área bajo la curva precisión-cobertura, robusta al desbalance de clases.
- **NDCG@k:** métrica de calidad de un orden que pondera más los errores cerca de la cabeza.
- **SHAP:** atribución con signo de la contribución de cada feature a una predicción concreta.
- **Deriva de reclasificación:** el evento VUS -> resuelta entre *releases*. Distinto del *concept drift* en sentido estricto, que es un cambio en P(Y|X).
- **Validación retrospectiva frente a prospectiva:** la primera entrena y evalúa dentro del mismo intervalo histórico; la segunda aplica el modelo, sin reentrenar, sobre verdad terreno publicada después de fijarlo.
- **Medallion:** patrón de capas de calidad creciente, RAW inmutable -> SILVER anotado -> GOLD listo para modelar.
