# ADR 005: Generador offline determinista, acotado a tests (revisado)

**Estado:** Aceptado, revisado · **Fecha:** 2026-07-06 · **Revisión:** 2026-07-30 · **Bloque:** I / B1

## Contexto original
El pipeline de datos requiere ClinVar (etiquetas, releases fechadas) y
features in silico (CADD/SIFT/PolyPhen/REVEL/AlphaMissense/GERP/phyloP/gnomAD AF).
Las fuentes reales se sirven desde NCBI FTP y agregadores públicos. Algunos
entornos de ejecución (p. ej. sandboxes con *allowlist* de red, CI aislada)
**no** pueden alcanzar esos hosts. Sin una alternativa, el pipeline, el EDA y
los tests no serían reproducibles en cualquier entorno, y CI no podría validar
la cadena de datos.

## Decisión original (2026-07-06)
Se implementó `src/ingest/synthetic.py`, un **generador offline determinista**
(seed 42) que produce ficheros con el **mismo esquema** que las fuentes reales.
`src/ingest/download.py` intentaba primero la descarga real y **caía
automáticamente** al generador si la red fallaba, sin exigir confirmación.

## Revisión (2026-07-30): se elimina el fallback automático en el pipeline "oficial"

Este proyecto trabaja con variantes genéticas asociadas a enfermedades reales.
Aunque ClinVar/gnomAD/CADD/REVEL/AlphaMissense son bases **públicas y
agregadas** (no contienen datos identificables de pacientes, ver `README.md`),
sustituir esos valores por datos inventados —por bien calibrados que estén
estadísticamente— no es aceptable como fuente de un resultado que se presenta
como válido en la memoria. La revisión de la revisión interna del proyecto (punto 1 de la
revisión guiada) detectó además que el fallback automático no dejaba rastro
estructural fuera de un `print` y de `MANIFEST.json` (que además se podía
sobrescribir sin querer, perdiendo la procedencia real en una re-ejecución sin
`--force`), y que mezclar ClinVar real con el subset dbNSFP sintético rompía
el entrenamiento por completo (0 columnas con datos).

**Nueva decisión:**
1. `src/ingest/download.py` con `offline=False` (por defecto) **exige** la
   descarga real de ClinVar y **lanza `RuntimeError`** si falla, en vez de
   sustituir en silencio. Solo `offline=True`, pasado explícitamente, activa
   el generador — nunca como efecto colateral de que la red no responda.
2. `config/config.yaml` (`data.annotation_source`) pasa a `multi_source` por
   defecto: las features de producción vienen de myvariant.info real (B5/ADR
   007), no del subset dbNSFP sintético.
3. `synthetic.py` queda **reservado exclusivamente a tests/CI**: cada test que
   lo necesita fija `annotation_source: synthetic` de forma explícita en su
   propia configuración aislada (`tmp_path`), en vez de heredarlo del valor
   por defecto del proyecto. No es alcanzable desde ningún comando `make`
   "oficial" salvo pasando `--offline` a mano.
4. La procedencia real (`MANIFEST.json.source` + `annotation_source`) se
   registra ahora como parámetro de MLflow y como banner explícito al
   principio de cada Model Card (`data_provenance` en `src/train/train.py`),
   para que ningún resultado pueda citarse como real sin poder verificarlo.
5. Se corrige el bug por el que recalcular el manifest sin `--force` (RAW ya
   presente) pisaba la procedencia original con el valor genérico `"existing"`:
   ahora se conserva el `source` real del manifest anterior.

El generador determinista en sí (`synthetic.py`) NO se elimina del repositorio:
sigue siendo necesario para que los tests y el CI se ejecuten rápido y sin
depender de la disponibilidad de NCBI/myvariant.info en cada ejecución — es el
mismo patrón estándar de fixtures deterministas que usaría cualquier proyecto,
incluso uno que solo publique resultados con datos reales. Lo que cambia es
que deja de poder alcanzar, ni por accidente ni en silencio, ningún artefacto
que se presente como resultado del proyecto.

## Consecuencias
* A favor: ningún resultado "oficial" (memoria, Model Cards, informes de VUS)
  puede generarse con datos sintéticos sin que quien lo ejecuta lo pida
  explícitamente con `--offline`.
* A favor: la procedencia queda trazada de forma estructurada (MLflow + Model
  Card), no solo en comentarios de código.
* A favor: los tests siguen siendo deterministas, rápidos y sin red — no
  dependen de la disponibilidad de NCBI/myvariant.info en cada `pytest`.
* En contra: ejecutar el pipeline completo (`make data`, `make train`,...)
  ahora requiere red real; si NCBI o myvariant.info no están disponibles, el
  comando falla explícitamente en vez de producir igualmente un resultado
  (degradación intencionadamente eliminada para el camino "oficial").

## Alternativas descartadas
* *Mantener el fallback automático y silencioso:* es lo que había hasta esta
  revisión; el riesgo de que un resultado sintético se cite por descuido como
  real (o de perder la trazabilidad en una re-ejecución) supera el beneficio
  de conveniencia, especialmente en un TFM sobre patogenicidad de variantes.
* *Eliminar `synthetic.py` por completo, incluidos los tests:* obligaría a que
  CI y la suite de tests dependieran de red real (NCBI/myvariant.info), frágil
  y lento; no aporta más rigor científico, porque los tests nunca fueron la
  fuente de ningún resultado citado en la memoria.
* *Fijar un dump estático real en el repo:* CADD/AlphaMissense/dbNSFP son
  enormes y de licencia variable; no versionable en el repositorio.
