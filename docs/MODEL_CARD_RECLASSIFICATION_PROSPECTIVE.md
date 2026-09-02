# Validación temporal PROSPECTIVA del modelo de reclasificación

Revisión posterior del proyecto:.

## Diseño
* Entrenamiento (sin cambios): VUS de 2023-12, etiquetadas por si se
  resuelven en 2025-06.
* Validación prospectiva (esta ejecución): de esas VUS, las que a fecha de
  2025-06 **seguían sin resolver** (4001
  variantes) se puntúan con el modelo YA entrenado (sin reentrenar) y se comprueba si se
  resolvieron en 2026-08 (3986 encontradas,
  7 resueltas, 0.0017).
* 2026-08 se publicó después de fijar el par de entrenamiento y
  no participó en el ajuste de ningún hiperparámetro ni en la selección de algoritmo.

## Resultado
| Modelo | Features | ROC AUC | PR AUC |
|--------|----------|---------|--------|
| Completo (logistic_regression) | todas (RECLASS_FEATURE_COLUMNS) | 0.6036 | 0.0087 |
| Ablación segura (logistic_regression) | `consequence`, `review_stars` | 0.4467 | 0.0042 |

Estas cifras, a diferencia de todas las demás del modelo de reclasificación en este repositorio, proceden
de una verdad terreno publicada DESPUÉS de fijar el par de entrenamiento y nunca
usada para entrenar ni para elegir nada: es la única evaluación de este proyecto
que mide capacidad de predicción prospectiva real, no señal retrospectiva dentro
de un mismo intervalo histórico.

**Aviso de potencia estadística:** con solo 7 casos
positivos en esta ventana prospectiva (una ventana más corta que la retrospectiva
2023-2025 usada para entrenar, con menos tiempo para acumular reclasificaciones),
este ROC AUC/PR AUC es una estimación muy ruidosa e insuficiente por sí sola para
afirmar ni refutar con seguridad la capacidad prospectiva del modelo; se reporta
igualmente, sin redondear al alza ni ocultar el tamaño muestral, porque es la
evidencia prospectiva real disponible, no una aproximación conveniente.


## Cómo citar esto en la memoria
Esta es la única cifra del modelo de reclasificación que responde de verdad a "¿predice este modelo una
reclasificación genuinamente futura?" (revisión posterior del proyecto, hallazgo
CRÍTICO #1). El resto de métricas del modelo de reclasificación (holdout aleatorio 2023-2025) miden
señal retrospectiva dentro del mismo intervalo, no capacidad prospectiva; no
deben presentarse como equivalentes a esta.
