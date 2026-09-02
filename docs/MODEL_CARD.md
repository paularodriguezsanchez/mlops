# Model Card: Clasificador de patogenicidad de variantes

## Procedencia
Datos reales de extremo a extremo: ClinVar descargado del NCBI y features de myvariant.info. Estas cifras son citables como resultado del proyecto.

## Modelo
* **Algoritmo:** logistic_regression.
* **Tarea:** clasificación binaria patogénica (1) / benigna (0) de SNVs.
* **Pipeline:** imputación + escalado + one-hot y clasificador, autocontenido.
* **Datos:** entrenamiento ClinVar 2023-12, evaluación
  temporal sobre ClinVar 2025-06.

## Selección y evaluación
Elijo el algoritmo por PR AUC medio en validación cruzada de 5 particiones
estratificadas **sobre el conjunto de entrenamiento** (columna CV). El holdout no
visto se evalúa una sola vez, sobre el algoritmo ya elegido: seleccionar y reportar
sobre el mismo conjunto introduce sesgo de selección. Las columnas de holdout del
resto de algoritmos son descriptivas, no criterio de decisión.

El holdout contiene solo variantes ausentes del entrenamiento; su tamaño lo fija
`data.max_new_variants_per_release`, no un cálculo de potencia estadística. El
intervalo es bootstrap al 95 % (1000 remuestreos): dos algoritmos con intervalos
solapados no son distinguibles con esta muestra. La columna *full* incluye variantes
que persisten entre releases y es optimista por memorización; se muestra por
transparencia.

| Algoritmo | PR AUC CV (train, 5-fold) | PR AUC (holdout) [IC 95%] | ROC AUC (holdout) | F1 (holdout) | PR AUC (full) |
|-----------|----------------------------|---------------------------|--------------------|--------------|----------------|
| logistic_regression | 0.9787±0.0090 | 0.9700 [0.9511, 0.9849] | 0.9945 | 0.9359 | 0.9772 |
| random_forest | 0.9750±0.0084 | 0.9657 [0.9388, 0.9858] | 0.9925 | 0.9216 | 0.9827 |
| gradient_boosting | 0.9760±0.0055 | 0.9721 [0.9556, 0.9864] | 0.9947 | 0.9212 | 0.9831 |
| hist_gradient_boosting | 0.9759±0.0090 | 0.9581 [0.9310, 0.9803] | 0.9875 | 0.9289 | 0.9809 |

## Features
Scores in silico (CADD, SIFT, PolyPhen, REVEL, AlphaMissense), conservación (GERP++,
phyloP), frecuencia de gnomAD en escala logarítmica y consecuencia funcional. Los
valores ausentes se imputan por mediana conservando un indicador de ausencia, que es
informativo en sí mismo. `gene` se excluye por alta cardinalidad y riesgo de fuga.

La importancia se mide de dos formas complementarias: por permutación
(`reports/training/feature_importance.csv`), que da un ranking global agnóstico al
modelo, y por SHAP (`shap_importance.csv`, `shap_summary.png`), que añade dirección y
magnitud del efecto por instancia.

## Calibración
PR AUC y ROC AUC miden discriminación, no calibración: que un score de 0,7 corresponda
de verdad a un 70 % de probabilidad. Sobre el mismo holdout, **Brier score =
0.0122** (0 es calibración perfecta; 0,25, el de un clasificador
que siempre predice 0,5). Predicción media frente a tasa observada por decil, omitiendo
bins vacíos:

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

Con este desbalance, los bins de probabilidad alta tienen pocos casos: su calibración es
menos fiable de lo que sugiere el Brier score global.

## Uso y limitaciones
* **Uso previsto:** ordenar VUS por riesgo estimado para dirigir la revisión manual
  (`src/serve/prioritize_vus.py`, ADR 006). No predice ni sustituye el veredicto de las
  variantes que ClinVar ya ha resuelto, ni la curación experta ACMG/AMP.
* **Limitaciones:** solo SNVs, cromosomas 1-3. Con el generador offline (ADR 005) las
  métricas validan el pipeline, no tienen valor clínico.
* **Ética:** solo bases públicas agregadas, sin datos genómicos identificables.

## Trazabilidad
Registrado en MLflow (experimento `variant_pathogenicity`,
modelo `variant_pathogenicity_clf`). Ver `reports/training/model_comparison.csv`.
