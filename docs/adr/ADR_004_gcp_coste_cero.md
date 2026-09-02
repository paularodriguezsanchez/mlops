# ADR 004: Estrategia de coste cero en la nube y enfoque de la comparativa

**Fecha:** 2026-07-06 · **Estado:** aceptada

## Contexto

Restricción autoimpuesta: no registrar tarjeta ni pagar nada. GCP separa los servicios que no exigen cuenta de facturación (BigQuery Sandbox) de los que sí la exigen (Cloud Run, Vertex AI), aunque el *free tier* de estos últimos no llegue a cobrar.

## Decisión

1. La capa de datos y anotación en la nube se resuelve con **BigQuery Sandbox**: sin tarjeta, 1 TB de consultas al mes y datasets genómicos públicos ya alojados.
2. El despliegue del servicio se aborda como diseño documentado más demo local. El despliegue real en Cloud Run queda condicionado a aceptar la verificación con tarjeta, decisión que finalmente no se tomó.
3. La comparativa se reenfoca a **MLflow autogestionado frente a la pila gestionada de GCP**, es decir código abierto frente a servicio administrado. DVC se mantiene como versionador de datos, no como objeto de comparación.

## Consecuencias

* La mayor parte de la extensión a la nube puede hacerse sin tarjeta y sin riesgo de coste.
* La comparativa entre autogestionado y gestionado es más rica y está más alineada con mi perfil profesional que una comparación de herramientas equivalentes.
* El despliegue real queda pendiente de una decisión posterior.
* BigQuery Sandbox expira las tablas a los 60 días, irrelevante aquí pero conviene documentarlo.
