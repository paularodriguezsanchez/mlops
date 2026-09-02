# Model Card: Clasificador de patogenicidad de variantes

## Procedencia de los datos de este run
**Datos REALES de extremo a extremo** (ClinVar descargado de NCBI + features de myvariant.info real). Estos números son citables como resultado del proyecto.

## Detalles del modelo
* **Mejor algoritmo:** logistic_regression (seleccionado por PR AUC media en
  validación cruzada de 5 particiones **sobre el conjunto de entrenamiento
  únicamente**, sin tocar el holdout).
* **Tarea:** clasificación binaria patogénica (1) vs benigna (0) de SNVs.
* **Pipeline:** preprocesamiento (imputación + escalado + one-hot) + clasificador, autocontenido.
* **Datos:** entrenamiento ClinVar 2023-12;
  evaluación en split temporal ClinVar 2025-06.

## Metodología de selección
Hasta esta revisión, el algoritmo "ganador" se elegía por PR AUC sobre el
mismo holdout que después se citaba como su evaluación final: dejaba de ser
un test genuinamente independiente (*selection bias*). Ahora la selección
usa **únicamente** la media de PR AUC en validación cruzada de 5 particiones
estratificadas sobre el conjunto de entrenamiento (columna **CV (train)**
abajo); el holdout no visto se evalúa **una sola vez**, sobre el algoritmo ya
elegido, y es esa evaluación —no la comparación de los 4 algoritmos en el
holdout— la que se cita como resultado final del proyecto. Las columnas de
holdout de los otros tres algoritmos se conservan por transparencia
descriptiva, no como criterio de selección.

## Métricas
La columna **holdout** evalúa solo variantes NO vistas en entrenamiento (evaluación
honesta de generalización). El intervalo junto al PR AUC de holdout es un intervalo
de confianza al 95% por bootstrap (1000 remuestreos): diferencias entre algoritmos
que se solapan en su intervalo no deben leerse como "el ganador es claramente mejor",
solo como el resultado de esta comparación puntual (ver punto 5 de
la revisión técnica del proyecto). El tamaño de este holdout es una consecuencia del
parámetro `data.max_new_variants_per_release` (config.yaml), no un valor elegido por
potencia estadística de esta comparación. La columna **full** incluye variantes que
persisten entre releases (optimista por memorización); se muestra por transparencia.
Ver la revisión interna del proyecto.

| Algoritmo | PR AUC CV (train, 5-fold) | PR AUC (holdout) [IC 95%] | ROC AUC (holdout) | F1 (holdout) | PR AUC (full) |
|-----------|----------------------------|---------------------------|--------------------|--------------|----------------|
| logistic_regression | 0.9787±0.0090 | 0.9700 [0.9511, 0.9849] | 0.9945 | 0.9359 | 0.9772 |
| random_forest | 0.9750±0.0084 | 0.9657 [0.9388, 0.9858] | 0.9925 | 0.9216 | 0.9827 |
| gradient_boosting | 0.9760±0.0055 | 0.9721 [0.9556, 0.9864] | 0.9947 | 0.9212 | 0.9831 |
| hist_gradient_boosting | 0.9759±0.0090 | 0.9581 [0.9310, 0.9803] | 0.9875 | 0.9289 | 0.9809 |

## Features
Entradas: scores in silico (CADD, SIFT, PolyPhen, REVEL), conservación (GERP++, phyloP),
frecuencia gnomAD (log) y consecuencia funcional. Los scores ausentes (no missense)
se imputan por mediana con indicador de ausencia.

Dos análisis de importancia, complementarios:
* **Permutación** (`reports/training/feature_importance.csv`): heurística agnóstica al modelo
  por barrido de una feature; rápida, ranking global únicamente.
* **SHAP** (`reports/training/shap_importance.csv` + `shap_summary.png`): valores de
  Shapley sobre el pipeline completo (mismas columnas de entrada); aporta además la
  dirección y magnitud del efecto por instancia, no solo un ranking agregado.

## Calibración
El modelo se presenta como "probabilidad de patogenicidad" en dashboard e informes; el
PR-AUC/ROC-AUC miden discriminación (orden), no calibración (si un score de 0,7 corresponde
de verdad a un 70 % de probabilidad real). Sobre el holdout no visto (misma población que la
sección de Métricas anterior):
**Brier score = 0.0122** (0 = calibración perfecta; 0,25 es el score
de un clasificador que siempre predice 0,5). Tabla de calibración por deciles (predicción
media frente a tasa observada en cada bin; bins vacíos omitidos):

| Bin de probabilidad predicha | n | Predicción media | Tasa observada |
|---|---|---|---|
| [0.0, 0.1) | 1528 | 0.0099 | 0.0046 |
| [0.1, 0.2) | 10 | 0.1247 | 0.1 |
| [0.2, 0.3) | 4 | 0.2433 | 0.25 |
| [0.3, 0.4) | 8 | 0.3533 | 0.375 |
| [0.4, 0.5) | 2 | 0.479 | 0.0 |
| [0.5, 0.6) | 2 | 0.5519 | 0.5 |
| [0.6, 0.7) | 6 | 0.6409 | 0.3333 |
| [0.7, 0.8) | 11 | 0.7435 | 0.7273 |
| [0.8, 0.9) | 11 | 0.8596 | 0.6364 |
| [0.9, 1.0) | 182 | 0.9873 | 0.9835 |

Con el desbalance de clases de este holdout, algunos bins de probabilidad alta tienen pocos
casos: la calibración en esos bins es menos fiable que el Brier score global sugiere por sí
solo, y se reporta con esa salvedad explícita en vez de solo el agregado.

## Usos y limitaciones
* **Uso previsto:** el modelo **no** predice ni sustituye el veredicto de variantes que
  ya tienen significado clínico resuelto en ClinVar (Patogénica/Benigna): eso ya está
  disponible, consultarlo. Su objetivo es **dirigir y agilizar la investigación manual
  posterior sobre las VUS** (variantes de significado incierto, sin veredicto): puntúa cada
  VUS con una probabilidad de patogenicidad a partir del conocimiento previo ya construido
  en ClinVar/dbNSFP/gnomAD, y permite **ordenarlas por riesgo estimado** para priorizar cuáles
  revisar primero (ver `src/serve/prioritize_vus.py`, ADR 006). Apoyo a la priorización, no
  un veredicto clínico ni un sustituto de la curación experta (ACMG/AMP).
* **Limitaciones:** solo SNVs; subconjunto de cromosomas en Fase I; si los datos provienen
  del generador offline (ADR 005), las métricas son de validación del pipeline, no clínicas.
* **Ética:** solo bases públicas y agregadas; sin datos genómicos individuales identificables.

## Trazabilidad
Registrado en MLflow (experimento `variant_pathogenicity`,
modelo `variant_pathogenicity_clf`). Ver `reports/training/model_comparison.csv`.
