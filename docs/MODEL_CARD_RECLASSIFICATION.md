# Model Card: Potencial de reclasificación de VUS

## Procedencia de los datos de este run
**Datos REALES de extremo a extremo.**

## Detalles del modelo
* **Tarea:** dada una VUS de la release 2023-12, predecir si se
  reclasificará (a Patogénica o Benigna) en la release 2025-06.
* **Mejor algoritmo:** logistic_regression (por PR AUC en holdout aleatorio).
* **Población:** 4056 VUS de 2023-12; 55 (1.4%)
  reclasificadas en 2025-06.

## Fiabilidad de la señal (revisión interna del proyecto)
El mejor modelo supera el umbral de fiabilidad (ROC AUC 0.7786 >= 0.6): se presenta como probabilidad en el dashboard y los informes sin aviso adicional.

**Naturaleza del umbral 0.6:** es una convención operativa interna del proyecto, sin respaldo estadístico, clínico ni de la literatura -no existe una referencia que establezca 0.6 como punto de corte de 'fiabilidad' para un ROC AUC-. Se documenta explícitamente como tal, no como una demostración de que el modelo sea fiable en sentido estadístico; el criterio real y más informativo es el intervalo de confianza del ROC AUC frente a 0.5 (azar), no la comparación puntual contra 0.6.

**Intervalo de confianza del ROC AUC** (bootstrap, 95%, 1000 remuestreos): [0.6492, 0.9014] (n_boot=1000). El intervalo completo queda por encima de 0.5 (azar), aunque su límite inferior queda también por encima del umbral operativo de 0.6: la discriminación por encima del azar es más defendible que la superación concreta del umbral interno.

## Limitación metodológica (léase antes de citar en la memoria)
Todas las métricas de esta ficha usan un **holdout aleatorio estratificado**
dentro de la población de VUS de la release antigua (2023-12), etiquetada
por si se resuelve en la release nueva (2025-06): miden señal discriminativa
retrospectiva dentro de ese mismo intervalo histórico, NO capacidad de
predicción prospectiva. **Existe una validación temporal prospectiva real,
aparte**: aplica este mismo
modelo, ya entrenado y sin reentrenar, sobre una release publicada después de
fijar este par -- ver `docs/MODEL_CARD_RECLASSIFICATION_PROSPECTIVE.md`
(`python -m src.train.train_reclass --prospective`). Es esa cifra, no esta, la
que responde a "¿predice el modelo una reclasificación genuinamente futura?".

## Leakage temporal de las features (ADR 008, léase antes de citar en la memoria)
Las features de anotación (CADD, REVEL, AlphaMissense, gnomAD, SIFT, PolyPhen,
GERP, phyloP) se consultan en vivo a myvariant.info **en el momento en que se
ejecuta el pipeline**, sin ningún anclaje a la fecha de la release de ClinVar:
el mismo snapshot "de hoy" se aplica tanto a la release 2023-12 (t0) como a
la 2025-06 (t1). Si una VUS se resolvió entre t0 y t1 precisamente porque
llegó evidencia computacional nueva (p. ej. AlphaMissense, publicado en 2023,
incorporándose como evidencia PP3/BP4), el modelo puede estar entrenando con
el valor **posterior** de esa evidencia, no con el que existía cuando la VUS
seguía sin resolver en t0 -- fuga de información del futuro hacia el pasado.
`review_status`/`review_stars` y `consequence`, en cambio, se leen directamente
del VCF fechado de cada release y son temporalmente seguros (ver
`src/features/preprocess.py::SAFE_RECLASS_FEATURE_COLUMNS`).

**Mitigación parcial aplicada:** se entrena, sobre el mismo split
train/holdout, un modelo de ablación que usa SOLO las features temporalmente
seguras (`consequence`, `review_stars`), para cuantificar cuánta señal del
modelo completo depende de las features con riesgo de fuga.

| Modelo | Features | PR AUC | ROC AUC |
|--------|----------|--------|---------|
| Completo (logistic_regression) | 10 (incluye CADD/REVEL/AlphaMissense/gnomAD/...) | 0.1093 | 0.7786 |
| Ablación temporalmente segura (logistic_regression) | `consequence`, `review_stars` | 0.0870 | 0.7476 |

La ablación (ROC AUC 0.7476) es notablemente peor que el modelo completo (ROC AUC 0.7786, diferencia +0.0310): una parte relevante de la señal del modelo completo depende de features con riesgo de fuga temporal (ver arriba). No citar el ROC AUC del modelo completo sin esta salvedad.

**Mitigación NO aplicada (trabajo futuro):** congelar snapshots históricos de
CADD/REVEL/AlphaMissense/gnomAD anclados a la fecha real de cada release, en
vez de la consulta única "de hoy", requeriría descargas versionadas por fuente
que no están garantizadas como archivadas y disponibles públicamente para
todas las fuentes; queda fuera del alcance de esta mitigación (ver ADR 008).

## Métricas (holdout aleatorio)
| Algoritmo | PR AUC | ROC AUC | F1 |
|-----------|--------|---------|-----|
| logistic_regression | 0.1093 | 0.7786 | 0.0000 |
| random_forest | 0.0668 | 0.7703 | 0.0000 |
| gradient_boosting | 0.0746 | 0.8065 | 0.0000 |
| hist_gradient_boosting | 0.0717 | 0.7178 | 0.0000 |

## Rigor estadístico adicional del mejor modelo
Holdout: n=1014, positivos=14 (1.4%).

**Intervalo de confianza del PR AUC** (bootstrap, 95%, 1000 remuestreos): [0.0326, 0.2735] (n_boot=1000).
Con solo 14 positivos en el holdout, este intervalo es necesariamente
ancho: se reporta explícitamente en vez de omitirlo, como pedía esa revisión,
no porque sea estrecho.

**Métricas de cola de revisión** (precision@k / recall@k / lift@k; lift=1.0
equivale a ordenar al azar):

| k | precision@k | recall@k | lift@k |
|---|---|---|---|
| 10 | 0.1 | 0.0714 | 7.2429 |
| 20 | 0.2 | 0.2857 | 14.4857 |
| 50 | 0.12 | 0.4286 | 8.6914 |
| 100 | 0.07 | 0.5 | 5.07 |

**Calibración** (Brier score = 0.0131; más bajo es
mejor, 0.25 es el de un clasificador que siempre predice 0.5 bajo esta
prevalencia). Tabla por quintil de probabilidad predicha (`n` bajo por bin es
esperable con solo 14 positivos; una tasa observada muy distinta
de la predicha en un bin con `n` pequeño no debe leerse como una calibración
pobre, sino como falta de datos para estimarla):

| Rango predicho | n | Media predicha | Tasa observada |
|---|---|---|---|
| [0.0, 0.2) | 1013 | 0.0131 | 0.0138 |
| [0.2, 0.4) | 0 | None | None |
| [0.4, 0.6) | 1 | 0.4908 | 0.0 |
| [0.6, 0.8) | 0 | None | None |
| [0.8, 1.0) | 0 | None | None |

Curva PR completa: `reports/training/reclass_pr_curve.csv`. Curva ROC completa:
`reports/training/reclass_roc_curve.csv`.

## Uso previsto
Complementa (no sustituye) la priorización por probabilidad de patogenicidad
(`src/serve/prioritize_vus.py`): además de "cuánto riesgo estimado tiene esta
VUS", indica "cuánta probabilidad hay de que esta VUS concreta se resuelva
pronto", útil para decidir qué VUS reanalizar primero cuando llega evidencia
nueva (bucle de reentrenamiento continuo, ADR 007 §5.5).
