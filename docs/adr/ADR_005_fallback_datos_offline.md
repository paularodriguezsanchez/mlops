# ADR 005: Generador offline determinista, acotado a pruebas

**Fecha:** 2026-07-06 · **Revisión:** 2026-07-30 · **Estado:** aceptada, revisada

## Contexto

El pipeline necesita ClinVar y las features in silico, servidas desde el NCBI y agregadores públicos. Algunos entornos de ejecución —sandboxes con allowlist de red, CI aislada— no alcanzan esos hosts. Sin alternativa, ni el pipeline ni las pruebas serían reproducibles en cualquier entorno.

## Decisión original

Implementé `src/ingest/synthetic.py`, un generador determinista con semilla fija que produce ficheros con el mismo esquema que las fuentes reales. `download.py` intentaba primero la descarga real y **caía automáticamente** al generador si la red fallaba, sin pedir confirmación.

## Revisión: elimino el fallback automático

Este proyecto trabaja con variantes asociadas a enfermedades reales. Aunque ClinVar, gnomAD y los predictores son bases públicas y agregadas, sustituir esos valores por datos inventados —por bien calibrados que estén— no es aceptable como fuente de un resultado que se presenta como válido.

Al revisarlo encontré además que el fallback no dejaba más rastro que un mensaje de consola y una entrada en `MANIFEST.json` que podía sobrescribirse sin querer en una reejecución, perdiendo la procedencia real; y que mezclar ClinVar real con el subset sintético rompía el entrenamiento por completo, con cero columnas con datos.

La decisión revisada:

1. `download.py` **exige** la descarga real y lanza `RuntimeError` si falla. Solo `--offline`, pasado a mano, activa el generador; nunca como efecto colateral de que la red no responda.
2. `annotation_source` pasa a `multi_source` por defecto: las features de producción vienen de myvariant.info real (ADR 007), no del subset sintético.
3. `synthetic.py` queda reservado a pruebas y CI. Cada test que lo necesita fija `annotation_source: synthetic` de forma explícita en su propia configuración aislada, en vez de heredarlo del valor por defecto.
4. La procedencia real se registra como parámetro de MLflow y como banner al principio de cada Model Card, para que ningún resultado pueda citarse como real sin poder verificarlo.
5. Corrijo el bug por el que recalcular el manifest sin `--force` pisaba la procedencia original con un valor genérico.

No elimino el generador. Sigue siendo necesario para que las pruebas y la CI corran rápido y sin depender de la disponibilidad de servicios externos: es el patrón estándar de fixtures deterministas de cualquier proyecto, incluso de uno que solo publique resultados reales. Lo que cambia es que deja de poder alcanzar, ni por accidente ni en silencio, ningún artefacto que se presente como resultado.

## Consecuencias

* Ningún artefacto citable puede generarse con datos sintéticos sin que quien ejecuta lo pida de forma explícita.
* La procedencia queda trazada de forma estructurada, no en comentarios de código.
* Las pruebas siguen siendo deterministas, rápidas y sin red.
* Ejecutar el pipeline completo exige ahora red real: si NCBI o myvariant.info no responden, el comando falla en vez de producir igualmente un resultado.

## Alternativas descartadas

* **Mantener el fallback automático:** el riesgo de que un resultado sintético se cite por descuido como real supera el beneficio de conveniencia, especialmente en un trabajo sobre patogenicidad de variantes.
* **Eliminar `synthetic.py` por completo:** obligaría a que la CI dependiera de red real, lo que es frágil y lento, y no aporta rigor: las pruebas nunca fueron fuente de ningún resultado citado.
* **Versionar un volcado estático real:** CADD, AlphaMissense y dbNSFP pesan decenas de gigabytes y tienen licencias heterogéneas; no es versionable en el repositorio.
