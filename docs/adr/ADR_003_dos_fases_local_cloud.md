# ADR 003: Enfoque en dos fases, local y extensión a la nube

**Fecha:** 2026-07-06 · **Estado:** aceptada

## Contexto

El anteproyecto excluía explícitamente la nube y fijaba una ejecución íntegramente local. Trabajo a diario sobre Google Cloud Platform y quiero incorporar el despliegue en la nube y una comparativa entre plataforma autogestionada y gestionada, sin poner en riesgo el cumplimiento de lo firmado ni incurrir en coste.

## Decisión

Divido el trabajo en dos fases coordinadas. La **fase local** construye el sistema completo, es fiel al anteproyecto, no cuesta nada y basta por sí sola para defender el trabajo. La **extensión a la nube** reproduce la misma arquitectura sobre GCP y produce la comparativa; se comunica a la dirección del trabajo para su visto bueno antes de acometerla.

La razón de ordenarlas así es de riesgo: si el calendario se ajusta, la segunda fase se recorta a diseño documentado sin que la primera se resienta. Es exactamente lo que acabó ocurriendo (ADR 004, `docs/FASE_II_DISENO_GCP.md`).

## Consecuencias

* El trabajo es autosuficiente al terminar la fase local.
* La decisión sobre registrar tarjeta se pospone al máximo.
* Amplía el alcance firmado y requiere aprobación de la dirección.
* Añade entre dos y tres semanas de trabajo, acotadas y recortables a documentación sin ejecución.
