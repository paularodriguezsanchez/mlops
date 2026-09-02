# ADR 001: Arquitectura de referencia de la plataforma

* **Fecha:** 2026-07-06
* **Estado:** Aceptada

## Contexto
El anteproyecto (OE2) exige una arquitectura MLOps de referencia con las fases: ingesta,
preprocesamiento, entrenamiento, registro de experimentos, servicio de inferencia,
monitorización y reentrenamiento. El dominio es genómica (clasificación de patogenicidad de variantes).

## Decisión
Arquitectura de 7 componentes desacoplados: (1) ingesta de VCF/ClinVar inmutable,
(2) anotación como capa medallion (RAW→SILVER→GOLD), (3) entrenamiento con scikit-learn,
(4) MLflow Tracking + Model Registry, (5) inferencia REST, (6) monitorización con Evidently,
(7) reentrenamiento manual disparado por alerta. La misma arquitectura lógica se implementa
en local (Fase I) y en GCP (Fase II) para permitir la comparativa.

## Consecuencias
* A favor: Portabilidad local↔cloud gracias a la contenerización.
* A favor: La capa medallion reutiliza experiencia profesional (GCP/Kyndryl) y da contenido a la memoria.
* A favor: Reentrenamiento manual (no automático) es defendible en el dominio clínico.
* En contra: Mantener paridad entre los dos entornos exige verificación explícita (tarea G2).
