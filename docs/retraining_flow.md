# Monitorización y reentrenamiento

Cómo se detecta la degradación del modelo y cómo se decide reentrenar.

## Las tres señales

Cada *release* nueva de ClinVar dispara la monitorización, que evalúa en paralelo:

1. **Drift de covariables.** Kolmogorov-Smirnov y PSI por feature (`src/monitor/drift.py`). Alerta si la proporción de features con drift supera `monitor.drift_threshold` (0,15). Se genera además un informe HTML con Evidently como verificación cruzada.
2. **Deriva de reclasificación.** Variantes que pasan de VUS estricta a un veredicto resuelto entre *releases*, y el acierto del modelo —entrenado cuando esas variantes estaban excluidas— sobre ellas. Un acierto bajo indica que la relación entre features y patogenicidad puede haber cambiado, aunque el evento en sí, una etiqueta que se completa, no lo demuestre por sí solo: por eso no lo llamo *concept drift*.
3. **Prueba de estrés.** Perturbación controlada que verifica que el mecanismo de alerta se dispara cuando el drift es real. No cuenta para la alerta global.

Si ninguna supera su umbral, el sistema sigue sirviendo sin cambios. Si alguna lo supera, emite una recomendación de reentrenar.

## Por qué el reentrenamiento es manual

En genómica clínica, promover un modelo automáticamente sería inaceptable: un cambio en el clasificador puede alterar la interpretación de variantes con impacto en pacientes. El sistema recomienda y una persona aprueba (ADR 001). El modelo nuevo entra en **Staging**, nunca directo a **Production**, y la promoción es un segundo gate humano independiente.

```bash
make monitor    # informes y drift_summary.json
make retrain    # evalúa la deriva y recomienda; no reentrena
python -m src.monitor.retrain --execute   # aprueba y lanza el reentrenamiento
```

## Resultado de la ejecución canónica (2023-12 -> 2025-06)

* Drift de covariables: **0 de 9 features**. Las covariables son estables.
* Deriva de reclasificación: **55 variantes reclasificadas** de 7997 compartidas (0,69 %, por debajo del umbral del 2 %); acierto del modelo de patogenicidad sobre ellas, **0,964**.
* Prueba de estrés: la perturbación dispara 3 de 9 features, lo que confirma que la alerta funciona.
* Recomendación: **no reentrenar** (`overall_alert=false`).

Es el caso más realista del dominio: no cambia qué variantes se ven, sino cómo se interpretan con el tiempo, que es exactamente lo que refleja ClinVar.
