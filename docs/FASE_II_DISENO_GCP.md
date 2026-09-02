# Fase II: diseño de la extensión a GCP (Opción A — documentada, no ejecutada)

**Decisión (2026-08-11):** Fase II se aborda primero como diseño documentado y razonado, sin desplegar nada real. Se reconsidera ejecutar (Opción B) solo si sobra tiempo antes del cierre del TFM — ver el trabajo futuro del proyecto (G1, G2, H1, H2) y el plan del proyecto (D6). Esta decisión evita el registro de una tarjeta de crédito (verificación de identidad para Cloud Run/Vertex AI) sin renunciar a la comparativa razonada que exige OE6.

## 1. Por qué Opción A y no Opción B

- La Fase I local ya es, por sí sola, un TFM completo y defendible.
- BigQuery Sandbox es gratis y sin tarjeta, pero Cloud Run/Vertex AI sí exigen una cuenta de facturación verificada con tarjeta (ADR 004).
- El tiempo restante hasta la entrega es más valioso invertido en la memoria (Fase III) que en operar una demo real en la nube que no cambia ninguna conclusión de fondo del proyecto.
- Si en algún momento sobra tiempo real, la Opción B (desplegar de verdad) está completamente especificada en `cloud/cloudrun/deploy.md` y `cloud/iac/main.tf.example` — no hay diseño pendiente, solo ejecución.

## 2. Arquitectura de datos: medallion en BigQuery

Mismas tres capas que la Fase I local (`data/raw` → `data/interim` → `data/processed`), reproducidas como esquemas de BigQuery. Diseño completo en `cloud/bigquery/`:

| Fichero | Capa | Equivalente local |
|---|---|---|
| `cloud/bigquery/01_raw_ddl.sql` | RAW | `src/ingest/download.py` |
| `cloud/bigquery/02_silver_annotate.sql` | SILVER | `src/annotate/annotate.py` / `multi_source.py` |
| `cloud/bigquery/03_gold_build_dataset.sql` | GOLD | `src/features/build_dataset.py` |

**Hallazgo de diseño relevante, verificado (no asumido):** `bigquery-public-data.gnomAD.v3_genomes__chr{N}` es un dataset público real, consultable sin coste dentro del *free tier* del Sandbox. `bigquery-public-data.human_variant_annotation.ncbi_clinvar_hg38_20180701` también existe, pero es un volcado **estático fechado a julio de 2018** — no las dos *releases* fechadas (2023-12 / 2025-06) que este proyecto necesita para medir la deriva de reclasificación real. Por tanto, la capa RAW de la Fase II seguiría cargando los mismos VCF que descarga `src/ingest/download.py` (subidos a Cloud Storage), no el snapshot público de ClinVar. CADD, REVEL y AlphaMissense tampoco están alojados como dataset público en BigQuery: en una ejecución real (Opción B) habría que cargar sus ficheros precalculados a Cloud Storage, igual que hoy se resuelve vía `myvariant.info` en local.

## 3. Despliegue del servicio: Cloud Run

Diseño completo en `cloud/cloudrun/deploy.md`: reutiliza `docker/app.Dockerfile` sin cambios (portabilidad local↔cloud, ADR 001), `min-instances=0` (coste cero fuera de uso), autenticación IAM + `TFM_API_KEY` vía Secret Manager (nunca en claro), y el modelo montado desde Cloud Storage en vez de empaquetado en la imagen.

## 4. Infraestructura como código

Esquema de referencia en `cloud/iac/main.tf.example` (extensión `.example` deliberada: no se ha ejecutado `terraform plan`/`apply`, es documentación de qué recursos harían falta, no IaC lista para producción): datasets de BigQuery, bucket de Cloud Storage para RAW, servicio de Cloud Run con escalado a cero, y binding de IAM que explícitamente nunca usa `allUsers` (coherente con de la auditoría de la Fase I).

## 5. Comparativa razonada: local (MLflow autogestionado) vs. GCP gestionado [OE6]

| Dimensión | Fase I — local (MLflow + Docker) | Fase II — GCP (BigQuery + Cloud Run + Vertex AI) |
|---|---|---|
| **Coste** | Cero, sin condiciones. | Cero en BigQuery Sandbox; Cloud Run/Vertex AI exigen cuenta de facturación verificada con tarjeta, aunque el *free tier* no cobre si se respeta el escalado a cero. |
| **Reproducibilidad** | Total y verificada: mismo commit + mismos datos → mismos resultados (con las salvedades de determinismo de LightGBM ya corregidas, ver la revisión interna del proyecto). | Equivalente en teoría (mismo código, mismos contenedores); no verificado empíricamente en este proyecto porque no se ha ejecutado (Opción A). |
| **Esfuerzo operativo** | Bajo: `docker compose up`, sin gestión de cuotas ni IAM más allá de lo local. | Mayor: gestión de proyectos GCP, cuentas de servicio, IAM, cuotas del *free tier*, Secret Manager. |
| **Escalabilidad** | Limitada por la máquina local; acotado deliberadamente a chr 1-3 (riesgo R2 del plan). | BigQuery escala a todo el genoma sin coste añadido dentro del Sandbox (1 TB de consultas/mes); Cloud Run escala horizontalmente el servicio de inferencia. |
| **Gobernanza del modelo** | MLflow Model Registry autogestionado, *stages* `None→Staging→Production`, gate humano en `src/monitor/retrain.py`. | Vertex AI Model Registry gestionado ofrece lo mismo con menos mantenimiento operativo propio, a cambio de menor control directo y de acoplamiento al proveedor. |
| **Adecuación al dominio biomédico** | Adecuada para un volumen acotado (Fase I); sin garantías de cumplimiento normativo específico más allá de trabajar solo con datos públicos agregados. | Los datasets genómicos públicos (gnomAD, 1000 Genomes) ya están alojados y optimizados en BigQuery por el propio proveedor — ventaja real para escalar sin reingeniería. |
| **Riesgo de coste inesperado** | Ninguno. | Bajo si se respeta `min-instances=0` y se desmontan recursos tras cada demo (`docs/retraining_flow.md`); el riesgo real está en dejar recursos *always-on* olvidados. |

**Conclusión de la comparativa:** para el volumen y el calendario de este TFM, la plataforma local autogestionada (MLflow + Docker) es la opción correcta — coste cero garantizado, reproducibilidad ya verificada, sin fricción operativa de IAM/cuotas. La ventaja real de GCP no es el coste ni la reproducibilidad (ambos ya resueltos en local), sino la **escalabilidad de la capa de datos** (BigQuery sobre todo el genoma, no solo chr 1-3) y la **reducción de mantenimiento operativo** de un Model Registry gestionado — ninguna de las dos es una necesidad real de un TFM, pero sí lo sería de una puesta en producción real a mayor escala.

## 6. Trabajo futuro (Opción B)

Si se decide ejecutar realmente: (1) crear el proyecto GCP y habilitar BigQuery Sandbox (sin tarjeta); (2) cargar los VCF de ClinVar y los ficheros de CADD/REVEL/AlphaMissense a Cloud Storage; (3) ejecutar el DDL de `cloud/bigquery/`; (4) verificar paridad de features frente a la capa SILVER local (G2); (5) solo si se decide registrar tarjeta, desplegar `cloud/cloudrun/deploy.md` y desmontar los recursos tras la demo.
