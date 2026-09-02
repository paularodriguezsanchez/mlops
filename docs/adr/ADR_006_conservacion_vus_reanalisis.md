# ADR 006: Conservación de las VUS para reanálisis

**Fecha:** 2026-07-24 · **Estado:** implementado en su parte operativa; el registro persistente queda como trabajo futuro

## Contexto

Las VUS se excluyen del entrenamiento por una razón sólida: VUS significa que ClinVar no tiene ground truth fiable para esa variante, así que incluirlas como clase positiva o negativa inyectaría ruido de etiqueta. Eso es correcto para *entrenar*, pero deja una pregunta abierta: qué pasa con ellas después. En la práctica clínica no se descartan, se reevalúan periódicamente conforme aparece evidencia nueva. El propio monitor ya mide esa reevaluación —en cada release una fracción de VUS se resuelve—, pero hasta esta decisión solo la medía, no la usaba para nada operativo.

**Referencia del estado del arte.** Talos, del Centre for Population Genomics con el Broad Institute y Microsoft Research (*Nature Medicine*, 2026), automatiza el reanálisis iterativo de variantes en enfermedad rara integrando ClinVar y PanelApp actualizados con lógica ACMG/AMP; en una cohorte de 4735 individuos sin diagnóstico, el reanálisis periódico por sí solo aportó un 5,1 % de diagnósticos nuevos. Confirma que tratar las VUS como candidatas a reanálisis es una línea activa, no una idea aislada.

**Diferencia de alcance con Talos:** prioriza a nivel de paciente, con trío familiar, fenotipo y patrón de herencia. Este trabajo opera a nivel de variante poblacional agregada, sin contexto de paciente. Talos no es aplicable ni integrable aquí; sirve como referencia conceptual, no como dependencia técnica.

## Decisión

Las VUS se persisten aparte, sin etiqueta, como conjunto de inferencia realista (`vus_{split}.parquet`), y se puntúan con el modelo ya registrado para ordenarlas por riesgo estimado (`src/serve/prioritize_vus.py`, `docs/vus_priorizadas.md`). Es una ayuda de **priorización** para dirigir la investigación manual posterior, nunca un veredicto que sustituya la curación de las variantes ya resueltas.

Queda sin implementar, como trabajo futuro acotado, el registro persistente: una tabla propia que conserve cada VUS con su clave, sus features y la *release* en que se observó por última vez, y que el monitor marque cuáles cambiaron de estado y con qué veredicto, convirtiendo una medición pasiva en un histórico accionable.

## Consecuencias

* Fundamenta con literatura reciente una decisión de diseño —excluir las VUS del entrenamiento— que de otro modo podría leerse como una simplificación poco reflexionada.
* Deja una línea de trabajo futuro concreta y acotada, no una vaguedad.
* La parte no implementada no bloquea nada: la priorización funciona sin el registro histórico.

## Alternativas descartadas

* **Incluir las VUS como tercera clase:** rompe la comparabilidad con la literatura, toda binaria, y no aporta señal fiable, porque por definición no hay ground truth.
* **Implementar el registro persistente ahora:** riesgo de ampliar el alcance de un bloque ya cerrado sin que ningún objetivo lo exija.
