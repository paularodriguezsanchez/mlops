# Validación temporal externa del modelo de reclasificación

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
* No es una validación prospectiva en sentido estricto: las features de anotación
  se consultan a myvariant.info al ejecutar el pipeline, sin anclaje a la fecha de
  cada release, de modo que el estado de las fuentes externas no quedó congelado
  antes de que existiera 2026-08.

## Resultado
| Modelo | Features | ROC AUC | PR AUC |
|--------|----------|---------|--------|
| Completo (hist_gradient_boosting) | todas (RECLASS_FEATURE_COLUMNS) | 0.6440 | 0.0045 |
| Ablación segura (hist_gradient_boosting) | `consequence`, `review_stars` | 0.3414 | 0.0039 |

A diferencia del resto de cifras del modelo, estas proceden de una verdad terreno
publicada después de fijar el par de entrenamiento y nunca usada para entrenar ni
seleccionar: es la evaluación del proyecto más cercana a medir capacidad
prospectiva, con la salvedad de las features no ancladas.

Con 7 casos positivos -una ventana más corta que la
retrospectiva, con menos tiempo para acumular reclasificaciones- la estimación es muy
ruidosa e insuficiente por sí sola para afirmar o refutar esa capacidad. La reporto sin
redondear al alza ni ocultar el tamaño muestral, porque es la evidencia disponible.


## Cómo citar
Es la única cifra del modelo que se enfrenta a una verdad terreno posterior al par de
entrenamiento. El resto de métricas (holdout aleatorio dentro del par de entrenamiento) miden
señal retrospectiva y no deben presentarse como equivalentes.
