# ADR 004: Estrategia GCP de coste cero y comparativa MLflow vs. gestionado

* **Fecha:** 2026-07-06
* **Estado:** Aceptada (D6 parcialmente abierta)

## Contexto
Restricción de la autora: no introducir tarjeta ni pagar. GCP separa servicios sin tarjeta
(BigQuery Sandbox) de servicios que exigen cuenta de facturación (Cloud Run, Vertex AI).
Además, al pasar a GCP la comparativa de plataformas (OE6) puede reenfocarse.

## Decisión
1. **Datos/anotación en GCP** vía **BigQuery Sandbox** (sin tarjeta: 1 TB/mes, datasets genómicos
   públicos ya alojados).
2. **Despliegue**: empezar por diseño documentado + demo local; el despliegue real en Cloud Run
   (que requiere tarjeta de verificación, sin cargo dentro del free tier) queda como decisión D6 abierta.
3. **Comparativa OE6** reenfocada a **MLflow autogestionado vs. Vertex AI/GCP gestionado**
   (open source vs. managed). DVC se mantiene como versionador de datos, no como objeto de comparativa.

## Consecuencias
* A favor: La mayor parte de la Fase II se hace sin tarjeta y sin riesgo de coste.
* A favor: La comparativa open source vs. managed es más rica y alineada con el perfil de la autora.
* En contra: El despliegue real en Cloud Run exige decisión posterior (D6) sobre registrar tarjeta.
* En contra: BigQuery Sandbox expira tablas a 60 días (irrelevante para el TFM; se documenta).
