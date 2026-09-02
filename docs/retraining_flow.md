# Flujo de monitorización y reentrenamiento [OE5]

Documenta el ciclo cerrado de MLOps de la plataforma: cómo se detecta la
degradación del modelo y cómo se decide reentrenar.

## Diagrama del ciclo

Cada release nueva de ClinVar dispara la monitorización (Evidently + `drift.py`), que evalúa tres señales en paralelo:

1. **Drift de covariables** (features: KS/PSI).
2. **Deriva de reclasificación** (VUS reclasificadas + acierto del modelo en ellas; no se llama "concept drift" porque una VUS que recibe veredicto no demuestra por sí sola un cambio en P(Y|X), solo que se completó una etiqueta que antes faltaba).
3. **Stress test** (perturbación controlada, valida que la alerta funciona).

Si ninguna supera el umbral (`monitor.drift_threshold`), el sistema sigue sirviendo sin cambios. Si alguna lo supera, se emite una recomendación de reentrenar, sujeta a un gate humano (revisión clínica/ML). Si se aprueba, `train.run` genera un nuevo modelo en stage Staging; un segundo gate humano decide su promoción a Production.

## Señales de alerta

1. **Drift de covariables (features).** KS + PSI por feature (`src/monitor/drift.py`).
   Alerta si la proporción de features con drift supera `monitor.drift_threshold`
   (config: 0,15). Informe HTML citable con **Evidently** (`reports/monitoring/evidently_drift.html`).
2. **Deriva de reclasificación.** Variantes que pasan de VUS (estrictamente
   `Uncertain_significance`) a etiqueta resuelta entre releases, y el **acierto
   del modelo** (entrenado con las etiquetas antiguas) sobre ellas. Un acierto
   bajo es indicio de que la relación *features→patogenicidad* podría haber
   cambiado, aunque el evento en sí (una etiqueta que se completa) no lo
   demuestra por sí solo.
3. **Stress test.** Perturbación controlada (plan R6) que garantiza poder verificar
   que el mecanismo de alerta se dispara cuando el drift es real.

## Por qué el reentrenamiento es MANUAL (defendible clínicamente)

En genómica clínica, promover un modelo automáticamente sería inaceptable: un
cambio en el clasificador puede alterar la interpretación de variantes con impacto
en pacientes. Por eso el sistema **recomienda** y un humano **aprueba** (ADR 001).
El nuevo modelo entra en **Staging**, nunca directamente en **Production**; la
promoción es un segundo gate humano.

## Uso

```bash
make monitor # genera informes y drift_summary.json (decisión de alerta)
make retrain # evalúa el drift y RECOMIENDA (dry-run, no reentrena)
python -m src.monitor.retrain --execute # aprueba y lanza el reentrenamiento
```

## Resultado observado (releases 2023-12 → 2025-06, ejecución vigente)

* Drift de covariables: **0/9 features** (las covariables son estables).
* Deriva de reclasificación: **55 variantes reclasificadas** de 7997 compartidas
  (0,69 %, por debajo del umbral de alerta del 2 %); acierto del modelo
  (patogenicidad, entrenado con las etiquetas antiguas) en ellas ≈ **0,964**.
  Sin alerta por este criterio.
* Stress test: perturbación controlada dispara 3/9 features (confirma que el
  mecanismo de alerta funciona).
* Recomendación: **no reentrenar** (`overall_alert=false`); ver
  `reports/monitoring/drift_summary.json` de la ejecución vigente.

Esto ilustra el caso más realista del dominio: no cambia *qué* variantes se ven,
sino *cómo se interpretan* con el tiempo, exactamente lo que ClinVar refleja.
