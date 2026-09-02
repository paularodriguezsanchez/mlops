# ADR 001: Arquitectura de referencia

**Fecha:** 2026-07-06 · **Estado:** aceptada

## Contexto

El trabajo exige una arquitectura MLOps completa —ingesta, preprocesamiento, entrenamiento, registro de experimentos, servicio de inferencia, monitorización y reentrenamiento— aplicada a la clasificación de patogenicidad de variantes genéticas.

## Decisión

Siete componentes desacoplados: ingesta de VCF inmutable; anotación como capa medallion (RAW, SILVER, GOLD); entrenamiento con scikit-learn; MLflow para tracking y Model Registry; inferencia REST; monitorización con Evidently junto al motor estadístico propio; y reentrenamiento manual disparado por alerta. La misma arquitectura lógica se implementa en local y se diseña para GCP, de modo que la comparativa entre ambas sea posible.

## Consecuencias

* La contenerización da portabilidad real entre local y nube, no solo declarada.
* La capa medallion reutiliza experiencia profesional previa en ingeniería de datos.
* El reentrenamiento manual, no automático, es defendible en el dominio clínico: un cambio de modelo puede alterar la interpretación de variantes con impacto en pacientes.
* Mantener paridad entre ambos entornos exigiría una verificación explícita que no se ha llegado a hacer, porque la extensión a la nube quedó como diseño.
