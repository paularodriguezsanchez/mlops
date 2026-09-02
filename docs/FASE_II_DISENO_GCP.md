# Extensión a GCP: diseño documentado, no ejecutado

Reproduzco sobre Google Cloud Platform la misma arquitectura lógica de la fase local, con el único objetivo de sostener una comparativa razonada entre plataforma autogestionada y gestionada. Documenté el diseño completo sin llegar a desplegarlo: BigQuery Sandbox es gratuito y no exige tarjeta, pero Cloud Run y Vertex AI sí requieren una cuenta de facturación verificada (ADR 004), y el tiempo restante rendía más en la memoria que en operar una demo en la nube que no cambia ninguna conclusión de fondo. El despliegue real queda como trabajo futuro, con todo lo necesario ya especificado en `cloud/cloudrun/deploy.md` y `cloud/iac/main.tf.example`.

## Arquitectura de datos: medallion en BigQuery

Las mismas tres capas que en local, reproducidas como esquemas de BigQuery (`cloud/bigquery/`):

| Fichero | Capa | Equivalente local |
|---|---|---|
| `01_raw_ddl.sql` | RAW | `src/ingest/download.py` |
| `02_silver_annotate.sql` | SILVER | `src/annotate/annotate.py`, `multi_source.py` |
| `03_gold_build_dataset.sql` | GOLD | `src/features/build_dataset.py` |

Verifiqué, en vez de asumirlo, qué hay realmente disponible como dataset público. `bigquery-public-data.gnomAD.v3_genomes__chr{N}` existe y es consultable sin coste dentro del *free tier*. `bigquery-public-data.human_variant_annotation.ncbi_clinvar_hg38_20180701` también existe, pero es un volcado estático de julio de 2018, no las dos *releases* fechadas que este proyecto necesita para medir deriva: la capa RAW seguiría cargando los mismos VCF que descarga `download.py`, subidos a Cloud Storage. CADD, REVEL y AlphaMissense tampoco están alojados como dataset público en BigQuery; habría que cargar sus ficheros precalculados a Cloud Storage, igual que hoy se resuelven vía `myvariant.info`.

## Servicio: Cloud Run

Diseño en `cloud/cloudrun/deploy.md`. Reutiliza `docker/app.Dockerfile` sin cambios —la portabilidad local-nube es una decisión de arquitectura explícita (ADR 001)—, con `min-instances=0` para coste cero fuera de uso, autenticación IAM junto a `TFM_API_KEY` vía Secret Manager, y el modelo montado desde Cloud Storage en vez de empaquetado en la imagen, para no reconstruirla en cada reentrenamiento.

## Infraestructura como código

`cloud/iac/main.tf.example` describe los recursos necesarios: datasets de BigQuery, bucket para RAW, servicio de Cloud Run con escalado a cero y binding de IAM que nunca usa `allUsers`. La extensión `.example` es deliberada: no he ejecutado `terraform plan` ni `apply`, así que es documentación de qué haría falta, no infraestructura lista para producción.

## Comparativa local frente a nube

| Dimensión | Local (MLflow y Docker) | GCP (BigQuery, Cloud Run, Vertex AI) |
|---|---|---|
| Coste | Cero, sin condiciones | Cero en BigQuery Sandbox; Cloud Run y Vertex AI exigen tarjeta verificada, aunque el *free tier* no cobre con escalado a cero |
| Reproducibilidad | Verificada: mismo commit y mismos datos dan el mismo resultado, con el determinismo de LightGBM ya corregido | Equivalente en teoría, mismo código y mismos contenedores; no verificada empíricamente aquí |
| Esfuerzo operativo | Bajo: `docker compose up` | Mayor: proyectos, cuentas de servicio, IAM, cuotas y Secret Manager |
| Escalabilidad | Limitada por la máquina local; acotado a chr1-3 | BigQuery cubre todo el genoma dentro del Sandbox (1 TB de consultas/mes); Cloud Run escala el servicio horizontalmente |
| Gobernanza del modelo | Model Registry autogestionado con estados y gate humano en `retrain.py` | Vertex AI Model Registry ofrece lo mismo con menos mantenimiento, a cambio de menor control y acoplamiento al proveedor |
| Dominio biomédico | Adecuada para un volumen acotado; sin garantías normativas más allá de trabajar solo con datos públicos | Los datasets genómicos públicos ya están alojados y optimizados por el proveedor, ventaja real para escalar sin reingeniería |
| Riesgo de coste | Ninguno | Bajo con `min-instances=0`; el riesgo real es dejar recursos *always-on* olvidados |

**Conclusión.** Para el volumen y el calendario de este trabajo, la plataforma local autogestionada es la opción correcta: coste cero garantizado, reproducibilidad ya verificada y sin fricción de IAM ni cuotas. La ventaja real de GCP no está en el coste ni en la reproducibilidad, ambos resueltos en local, sino en la escalabilidad de la capa de datos y en la reducción del mantenimiento operativo de un Model Registry gestionado. Ninguna de las dos es una necesidad de este trabajo, pero sí lo serían de una puesta en producción a mayor escala.

## Si se decidiera ejecutarlo

Crear el proyecto y habilitar BigQuery Sandbox, que no exige tarjeta; cargar los VCF de ClinVar y los ficheros de CADD, REVEL y AlphaMissense a Cloud Storage; ejecutar el DDL de `cloud/bigquery/`; verificar paridad de features frente a la capa SILVER local; y, solo si se acepta registrar tarjeta, desplegar según `cloud/cloudrun/deploy.md` y desmontar los recursos tras la demo.
