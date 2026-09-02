# Datasheet del conjunto de datos

Siguiendo el marco de Gebru et al. (2021), adaptado al alcance de este trabajo.

## Motivación

Entrenar y evaluar modelos de patogenicidad y de potencial de reclasificación de variantes de un solo nucleótido, y medir la deriva real entre *releases* fechadas de ClinVar.

## Composición

* **Unidad:** una SNV identificada por `(chrom, pos, ref, alt)` sobre GRCh38. La unidad es la variante, no el par variante-condición: `CLNSIG` llega del VCF como una clasificación ya agregada por el NCBI a partir de los envíos de múltiples laboratorios y condiciones. Cuando esos envíos discrepan, ClinVar lo refleja como clasificación conflictiva, una de las categorías que este trabajo excluye.
* **Etiqueta:** significancia clínica binarizada según `config.yaml::target` y `src/features/reclassification.py`.
  * Positivo (1): `Pathogenic`, `Likely_pathogenic`, `Pathogenic/Likely_pathogenic`.
  * Negativo (0): `Benign`, `Likely_benign`, `Benign/Likely_benign`.
  * VUS reservada: **estrictamente** `Uncertain_significance`, el término oficial de ClinVar, no cualquier variante sin veredicto.
  * Excluida: cualquier otro valor de `clnsig`. Se persiste aparte en `excluded_{split}.parquet` por trazabilidad, no se descarta en silencio.
* **Features:** `cadd_phred`, `sift_score`, `polyphen_score`, `revel_score`, `alphamissense_score`, `gerp_rs`, `phylop`, `gnomad_af`, más `gene`, `consequence` y `review_status`/`review_stars`. Los tres scores específicos de missense solo están definidos para ese tipo de variante.
* **Acotación:** cromosomas 1 a 3 (`data.chromosomes_subset`).

La definición estricta de VUS importa por una razón concreta: mezclar clasificaciones conflictivas dentro de esa población infla su tamaño y produce recuentos de reclasificación distintos según qué módulo los calcule. Esa ambigüedad causó una discrepancia real en el proyecto, 67 frente a 54 reclasificaciones citadas para la misma transición; con la definición unificada, ambas cifras coinciden en 55.

## Fuentes

| Fuente | Papel | Acceso |
|---|---|---|
| ClinVar (NCBI) | Etiquetas y reclasificaciones fechadas entre *releases* | VCF GRCh38 del archivo público |
| myvariant.info | Agregador de las features in silico y de la frecuencia poblacional | API pública, sin registro |
| gnomAD, CADD, REVEL, AlphaMissense, SIFT, PolyPhen-2, GERP++, phyloP | Frecuencia, deletereidad y conservación | Vía myvariant.info |

*Releases* empleadas: `2023-12` para entrenar, `2025-06` para evaluar y `2026-08` como verdad terreno prospectiva.

dbNSFP se descartó como fuente por exigir registro académico (ADR 007). El generador offline determinista de `src/ingest/synthetic.py` sigue imitando su esquema, pero está reservado a pruebas y CI: nunca alimenta un resultado citable (ADR 005).

## Proceso

`src/ingest/download.py` descarga las *releases* reales y falla de forma explícita si no hay red. `src/annotate/annotate.py` parsea el VCF, acota, cruza con las features y valida el contrato de datos de `src/annotate/schema.py`. `src/features/build_dataset.py` binariza el target y separa las particiones.

## Usos y limitaciones

* **Uso previsto:** clasificación binaria supervisada con algoritmos estándar, estimación del potencial de reclasificación y monitorización de deriva entre *releases*.
* **Limitaciones:** solo SNVs, sin CNV ni variantes estructurales; cromosomas 1 a 3; sin datos genómicos individuales identificables. Con datos del generador offline, los valores no son clínicos y no deben interpretarse como tales.

## Taxonomía CLNSIG -> bucket

Generada en cada `make dataset` a partir de `reports/training/clinvar_label_taxonomy.csv`. ClinVar cambia el vocabulario de este campo entre *releases* —`Conflicting_interpretations_of_pathogenicity` en 2023-12 pasa a `Conflicting_classifications_of_pathogenicity` en 2025-06—, así que el código no depende de una lista cerrada de valores excluidos: solo de los tres buckets con significado explícito. Todo lo demás cae en "excluido" automáticamente, sea cual sea su ortografía.

| Partición | Release | CLNSIG crudo | n | Bucket |
|---|---|---|---:|---|
| train | 2023-12 | Uncertain_significance | 4056 | vus |
| train | 2023-12 | Likely_benign | 2272 | negativo |
| train | 2023-12 | Benign | 568 | negativo |
| train | 2023-12 | Conflicting_interpretations_of_pathogenicity | 410 | excluido |
| train | 2023-12 | Pathogenic | 263 | positivo |
| train | 2023-12 | Likely_pathogenic | 208 | positivo |
| train | 2023-12 | Benign/Likely_benign | 138 | negativo |
| train | 2023-12 | Pathogenic/Likely_pathogenic | 55 | positivo |
| train | 2023-12 | not_provided | 20 | excluido |
| train | 2023-12 | other | 2 | excluido |
| train | 2023-12 | Likely_risk_allele | 2 | excluido |
| train | 2023-12 | drug_response | 1 | excluido |
| train | 2023-12 | protective\|risk_factor | 1 | excluido |
| train | 2023-12 | Affects | 1 | excluido |
| test | 2025-06 | Uncertain_significance | 6042 | vus |
| test | 2025-06 | Likely_benign | 3524 | negativo |
| test | 2025-06 | Benign | 752 | negativo |
| test | 2025-06 | Conflicting_classifications_of_pathogenicity | 699 | excluido |
| test | 2025-06 | Pathogenic | 325 | positivo |
| test | 2025-06 | Likely_pathogenic | 285 | positivo |
| test | 2025-06 | Benign/Likely_benign | 228 | negativo |
| test | 2025-06 | Pathogenic/Likely_pathogenic | 107 | positivo |
| test | 2025-06 | not_provided | 24 | excluido |
| test | 2025-06 | no_classification_for_the_single_variant | 4 | excluido |
| test | 2025-06 | other | 2 | excluido |
| test | 2025-06 | drug_response | 1 | excluido |
| test | 2025-06 | protective\|risk_factor | 1 | excluido |
| test | 2025-06 | Likely_risk_allele | 1 | excluido |
| test | 2025-06 | association | 1 | excluido |
| test | 2025-06 | Affects | 1 | excluido |

Totales: train 3504 etiquetadas, 4056 VUS y 437 excluidas; test 5221 etiquetadas, 6042 VUS y 734 excluidas.

## Mantenimiento

`data/raw/` es inmutable y se versiona con DVC. `data/raw/MANIFEST.json` registra fuente, tamaño y SHA-256 de cada fichero.
