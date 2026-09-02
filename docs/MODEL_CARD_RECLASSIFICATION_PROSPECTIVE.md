# Validación temporal prospectiva del modelo de reclasificación

## Diseño
* Entrenamiento, sin cambios: VUS de 2023-12, etiquetadas según se
  resuelvan en 2025-06.
* Validación: de esas VUS, las que a fecha de 2025-06 seguían sin
  resolver (4001) se puntúan con el modelo ya entrenado, sin
  reentrenar, y se comprueba si se resolvieron en 2026-08
  (3986 localizadas, 7 resueltas,
  0.0017).
* 2026-08 se publicó después de fijar el par de entrenamiento y
  no intervino en la selección de algoritmo ni en ningún hiperparámetro.

## Resultado
| Modelo | Features | ROC AUC | PR AUC |
|--------|----------|---------|--------|
| Completo (logistic_regression) | todas (RECLASS_FEATURE_COLUMNS) | 0.6036 | 0.0087 |
| Ablación segura (logistic_regression) | `consequence`, `review_stars` | 0.4467 | 0.0042 |

A diferencia del resto de cifras del modelo, estas proceden de una verdad terreno
publicada después de fijar el par de entrenamiento y nunca usada para entrenar ni
seleccionar: es la única evaluación del proyecto que mide capacidad prospectiva.

Con 7 casos positivos -una ventana más corta que la
retrospectiva, con menos tiempo para acumular reclasificaciones- la estimación es muy
ruidosa e insuficiente por sí sola para afirmar o refutar esa capacidad. La reporto sin
redondear al alza ni ocultar el tamaño muestral, porque es la evidencia disponible.

## Cómo citar
Es la única cifra del modelo que responde a si predice una reclasificación genuinamente
futura. El resto de métricas (holdout aleatorio dentro del par de entrenamiento) miden
señal retrospectiva y no deben presentarse como equivalentes.
