# Model Card: Potencial de reclasificación de VUS

## Procedencia de los datos de este run
**Datos REALES de extremo a extremo.**

## Modelo
* **Tarea:** dada una VUS de la release 2023-12, predecir si estará resuelta
  (patogénica o benigna) en la release 2025-06.
* **Algoritmo:** hist_gradient_boosting, elegido por PR AUC medio en validación cruzada
  repetida (5 pliegues x 5 repeticiones, estratificada)
  **sobre el conjunto de entrenamiento**. El holdout no interviene en esa
  decisión: se evalúa una sola vez, con el algoritmo ya elegido. Las columnas de
  holdout del resto de candidatos son descriptivas, no criterio de selección.
* **Población:** 4056 VUS de 2023-12, de las que 55
  (1.4 %) se reclasificaron en 2025-06.

## Fiabilidad de la señal
Supera el umbral (ROC AUC 0.7178 >= 0.6): la probabilidad se muestra en el dashboard y en los informes sin aviso adicional.

El umbral 0.6 es una convención operativa mía, sin respaldo estadístico ni de la literatura: no existe una referencia que fije 0.6 como punto de corte de fiabilidad de un ROC AUC. Lo documento como tal. El criterio informativo es el intervalo de confianza del ROC AUC frente a 0.5, no la comparación contra 0.6.

**Intervalo de confianza del ROC AUC** (bootstrap, 95%, 1000 remuestreos): [0.5302, 0.8689] (n_boot=1000). El intervalo completo queda por encima de 0.5 (azar), aunque su límite inferior sí queda por debajo del umbral operativo de 0.6: la discriminación por encima del azar es más defendible que la superación concreta del umbral interno.

## Alcance de esta evaluación
El holdout es aleatorio estratificado dentro del par 2023-12/2025-06: mide señal
discriminativa retrospectiva dentro de ese intervalo, no capacidad prospectiva. La
validación temporal externa -mismo modelo, sin reentrenar, sobre una release posterior
que no intervino en la selección ni en el ajuste- está en
`docs/MODEL_CARD_RECLASSIFICATION_PROSPECTIVE.md`
(`python -m src.train.train_reclass --prospective`). Es esa cifra, no esta, la que
responde a si el modelo predice una reclasificación genuinamente futura.

## Leakage temporal de las features (ADR 008)
Las features de anotación (CADD, REVEL, AlphaMissense, gnomAD, SIFT, PolyPhen, GERP,
phyloP) se consultan a myvariant.info en el momento de ejecutar el pipeline, sin anclaje
a la fecha de la release: el mismo estado "de hoy" se aplica a 2023-12 (t0) y a
2025-06 (t1). Si una VUS se resolvió entre t0 y t1 porque llegó evidencia
computacional nueva -AlphaMissense, publicado en 2023, es el caso claro-, el modelo puede
estar entrenando con el valor posterior de esa evidencia. `consequence` y
`review_status`/`review_stars` sí se leen del VCF fechado y son temporalmente seguras
(`src/features/preprocess.py::SAFE_RECLASS_FEATURE_COLUMNS`).

Como mitigación parcial entreno, sobre el mismo split, un modelo restringido a esas dos
features seguras, para cuantificar la dependencia en vez de solo advertirla.

| Modelo | Features | PR AUC | ROC AUC |
|--------|----------|--------|---------|
| Completo (hist_gradient_boosting) | 10 (incluye CADD/REVEL/AlphaMissense/gnomAD/...) | 0.0717 | 0.7178 |
| Ablación temporalmente segura (hist_gradient_boosting) | `consequence`, `review_stars` | 0.0911 | 0.7516 |

La ablación (ROC AUC 0.7516) **iguala o supera** al modelo completo (ROC AUC 0.7178, diferencia -0.0339): las dos features temporalmente seguras (`consequence`, `review_stars`) por sí solas discriminan tanto o más que el conjunto completo. Esto es tranquilizador respecto al riesgo de fuga (el resultado del modelo completo no depende de las features de riesgo para alcanzar su nivel de discriminación), pero abre una pregunta distinta que no cierra esta ablación: por qué añadir más features (incluidas las de riesgo) no mejora, e incluso empeora, el ROC AUC del holdout -- con solo 55 positivos en la población, es plausible que el modelo completo sobreajuste con más dimensiones sin más señal que aportar. No se presenta el ROC AUC del modelo completo como superior al de la ablación.

La mitigación completa -congelar snapshots de CADD, REVEL, AlphaMissense y gnomAD
anclados a la fecha de cada release- exige descargas versionadas por fuente que no están
garantizadas como archivadas públicamente; queda como trabajo futuro (ADR 008).

## Métricas (holdout aleatorio)
| Algoritmo | PR AUC | ROC AUC | F1 |
|-----------|--------|---------|-----|
| logistic_regression | 0.1093 | 0.7786 | 0.0000 |
| random_forest | 0.0668 | 0.7703 | 0.0000 |
| gradient_boosting | 0.0746 | 0.8065 | 0.0000 |
| hist_gradient_boosting | 0.0717 | 0.7178 | 0.0000 |

## Incertidumbre y métricas de cola
Holdout: n=1014, positivos=14 (1.4 %).

IC 95 % bootstrap del PR AUC (1000 remuestreos): [0.0227, 0.1909] (n_boot=1000). Con 14 positivos
el intervalo es necesariamente ancho; lo reporto igualmente.

El uso real del modelo es ordenar una cola de revisión, así que las métricas de cola
(lift=1.0 equivale a ordenar al azar) son más informativas que el agregado:

| k | precision@k | recall@k | lift@k |
|---|---|---|---|
| 10 | 0.0 | 0.0 | 0.0 |
| 20 | 0.15 | 0.2143 | 10.8643 |
| 50 | 0.1 | 0.3571 | 7.2429 |
| 100 | 0.06 | 0.4286 | 4.3457 |

Calibración: Brier score = 0.0138 (más bajo, mejor). El
baseline informativo con esta prevalencia **no** es el clasificador que siempre predice
0.5, cuyo Brier de 0.25 es trivial de batir cuando la clase positiva ronda el 1 %: es el
que predice siempre la prevalencia observada, con Brier = p(1-p) =
0.0136. Un Brier bajo aquí
refleja sobre todo el desbalance, no discriminación. Por eso la salida se describe como
*score probabilístico* y no como probabilidad clínicamente calibrada: con
14 positivos no hay datos para sostener lo segundo, y una tasa observada muy
distinta de la predicha en un bin poco poblado indica falta de datos para estimarla, no
mala calibración:

| Rango predicho | n | Media predicha | Tasa observada |
|---|---|---|---|
| [0.0, 0.2) | 1008 | 0.0039 | 0.0139 |
| [0.2, 0.4) | 5 | 0.3102 | 0.0 |
| [0.4, 0.6) | 1 | 0.4539 | 0.0 |
| [0.6, 0.8) | 0 | None | None |
| [0.8, 1.0) | 0 | None | None |

Curvas completas: `reports/training/reclass_pr_curve.csv` y `reclass_roc_curve.csv`.

## Uso previsto
Complementa la priorización por probabilidad de patogenicidad
(`src/serve/prioritize_vus.py`): además de cuánto riesgo estimado tiene una VUS, indica
cuál es más probable que se resuelva dentro del horizonte de evaluación, para
decidir qué reanalizar primero cuando
llega evidencia nueva (ADR 007 §5.5).
