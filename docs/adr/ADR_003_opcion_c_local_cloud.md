# ADR 003: Enfoque en dos fases (Opción C: local → cloud)

* **Fecha:** 2026-07-06
* **Estado:** Aceptada

## Contexto
El anteproyecto excluía explícitamente el cloud (ejecución 100 % local). La autora trabaja sobre
GCP y desea incorporar el despliegue cloud y una comparativa local vs. cloud, sin poner en riesgo
el cumplimiento del anteproyecto ni incurrir en coste.

## Decisión
Ejecutar el TFM en dos fases coordinadas: **Fase I** construye el sistema completo en local
(fiel al anteproyecto, coste cero, TFM ya defendible); **Fase II** despliega la misma arquitectura
en GCP y produce la comparativa. La extensión a GCP se comunicará al director para su visto bueno.

## Consecuencias
* A favor: El TFM es autosuficiente al terminar la Fase I; la Fase II es extensión de alto valor.
* A favor: La decisión de coste/tarjeta se pospone al máximo (ver ADR 004 y D6).
* En contra: Amplía el alcance firmado, requiere aprobación del director.
* En contra: Añade unas 2 a 3 semanas de trabajo (acotado, y recortable a "documentado sin ejecutar").
