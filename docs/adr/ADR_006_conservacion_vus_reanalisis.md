# ADR 006: Conservación de VUS para reanálisis futuro

**Estado:** Parcialmente implementado (priorización sí; registro persistente, no) · **Fecha:** 2026-07-24 · **Bloque:** I / B3

## Contexto

B3 excluye las variantes de significado incierto (VUS) del entrenamiento: VUS significa
literalmente que ClinVar no tiene un ground-truth fiable para esa variante, así que
incluirlas como clase positiva o negativa inyectaría ruido de etiqueta en un problema
supervisado. Esto es correcto para *entrenar*, pero plantea una pregunta legítima: ¿qué
pasa con esas VUS después? En la práctica clínica no se descartan: se **reevalúan
periódicamente** a medida que aparece evidencia nueva (nuevas publicaciones, más
población de referencia, reclasificaciones de ClinVar). De hecho, el propio pipeline de
monitorización (`src/monitor/drift.py`, esa etapa) ya mide esto: en cada release nueva, una
fracción de VUS del release anterior se reclasifica como patogénica o benigna. Esa es
la señal de concept drift que sustenta OE5, pero hoy el pipeline solo la **mide**, no
la **usa** para nada operativo.

**Herramienta relacionada (referencia para el estado del arte, A2):** [TALOS](https://github.com/populationgenomics/talos),
desarrollada por el Centre for Population Genomics (Australia) con Broad Institute y
Microsoft Research, publicada en *Nature Medicine* (2026). Automatiza el reanálisis
iterativo de variantes en enfermedad rara integrando ClinVar/PanelApp actualizados y
lógica ACMG/AMP; en una cohorte de 4.735 individuos sin diagnóstico, el reanálisis
periódico por sí solo aportó un 5,1 % de diagnósticos nuevos. Confirma que "las VUS
son candidatas a reanálisis, no un callejón sin salida" es una línea de trabajo real y
activa, no una idea aislada.

**Diferencia de alcance con TALOS:** TALOS prioriza variantes a nivel de **paciente**
(trío familiar, fenotipo, patrón de herencia) para diagnóstico clínico individual. Este
TFM clasifica patogenicidad a nivel de **variante poblacional**, sin contexto de
paciente. TALOS no es directamente aplicable ni integrable en la arquitectura actual;
sirve como referencia conceptual y bibliográfica, no como dependencia técnica.

## Decisión

No se implementa en Fase I, para evitar scope creep sobre un núcleo ya cerrado y
defendible. Se documenta como diseño propuesto y como línea de trabajo futuro (a
incluir en la guía de buenas prácticas I2 y/o en conclusiones y trabajo futuro I9):

1. Las VUS reservadas (`target: exclude_labels` en `config/config.yaml`) se conservarían
   en una tabla propia (`data/processed/vus_registry.*`), con su clave de variante,
   features anotadas y la fecha/release en que se observaron por última vez.
2. En cada ejecución de `src/monitor/drift.py` (ya compara release antigua vs. nueva),
   además de medir el % reclasificado, se marcarían en el registro las VUS que
   **efectivamente** cambiaron de estado, con su nueva etiqueta y release de origen,
   convirtiendo una medición pasiva en un registro accionable.
3. El "reanálisis" en sí (aplicar el modelo ya entrenado a las VUS reclasificadas para
   validar si su predicción coincidía con el nuevo veredicto clínico) sería un informe,
   no un flujo de reentrenamiento nuevo: reutiliza el modelo y el pipeline de inferencia
   ya existentes (`src/serve/predictor.py`).

## Consecuencias

* A favor: Fundamenta con literatura reciente (TALOS, *Nature Medicine* 2026) una decisión
  de diseño (exclusión de VUS) que de otro modo podría leerse como una simplificación
  poco reflexionada.
* A favor: Deja una línea de trabajo futuro concreta y acotada, no una vaguedad genérica.
* En contra: No aporta a ningún OE de forma directa en Fase I; correctamente fuera de alcance.
* **Mitigación de scope creep:** si en algún momento se decide implementarlo, entra
  como línea de trabajo futuro explícita y pasa primero por este ADR.

## Actualización (2026-07-24)

Implementado el punto 3 de la decisión, adelantado desde "trabajo futuro": `src/serve/prioritize_vus.py`
puntúa las VUS reservadas (`data/processed/vus_{split}.parquet`) con el modelo ya registrado y las
ordena por riesgo estimado de patogenicidad (`docs/vus_priorizadas.md`). Aclara de forma explícita
(README, Model Card) que esto es una ayuda de **priorización** para dirigir la investigación manual
posterior, nunca un veredicto que sustituya la curación clínica de las variantes ya resueltas.

Siguen pendientes, sin implementar, los puntos 1 y 2 (registro persistente `vus_registry.*` con
seguimiento de qué VUS se reclasificaron entre releases y cuándo): eso sigue siendo trabajo futuro,
no bloqueante para Fase I.

## Alternativas descartadas

* *Incluir VUS en el entrenamiento con una tercera clase:* rompe la comparabilidad con
  la literatura (REVEL, ClinPred, etc., todas binarias) y no aporta señal de entrenamiento
  fiable (por definición, VUS no tiene ground-truth).
* *Implementar el registro de VUS ahora, dentro de Fase I:* descartado por riesgo de
  scope creep sobre un bloque ya cerrado (B3) sin que ningún OE lo exija.
