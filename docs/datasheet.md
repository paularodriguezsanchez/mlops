# Datasheet del dataset: variantes anotadas (Fase I)

*Siguiendo el marco "Datasheets for Datasets" (Gebru et al., 2021), adaptado al TFM.*

## Motivación
Dataset para entrenar y evaluar un clasificador de **patogenicidad de variantes
genéticas** (SNVs) y para estudiar el *concept drift* temporal entre releases de
ClinVar. Creado en el marco del TFM "Plataforma MLOps para anotación y
clasificación de variantes genéticas" (UEM).

## Composición
* **Unidad:** una variante SNV identificada por `(chrom, pos, ref, alt)` sobre GRCh38.
* **Etiqueta (target):** significancia clínica de ClinVar binarizada (`config.yaml::target`,
  `src/features/reclassification.py`). Revisión posterior del proyecto: la
  taxonomía completa se documenta aquí de forma exhaustiva porque antes no lo estaba,
  y esa falta de precisión fue la causa concreta de una discrepancia numérica (67 vs.
  54 reclasificaciones citadas para la misma transición en dos sitios distintos de una
  versión anterior de la memoria del TFM).
  * Positivo (1): `Pathogenic`, `Likely_pathogenic`, `Pathogenic/Likely_pathogenic`.
  * Negativo (0): `Benign`, `Likely_benign`, `Benign/Likely_benign`.
  * VUS (reservada, población de inferencia del modelo de reclasificación, priorización): **estrictamente**
    `Uncertain_significance` — el término oficial de ClinVar, no cualquier variante
    sin veredicto positivo/negativo.
  * Excluida (ni entrena ni se prioriza como VUS, se persiste aparte por
    trazabilidad en `excluded_{split}.parquet`): cualquier otro valor de `clnsig`
    — clasificaciones conflictivas (en cualquier vocabulario de release), `not_provided`,
    `other`, `drug_response`, `protective|risk_factor`, `Likely_risk_allele`,
    `association`, `no_classification_for_the_single_variant`. Ver tabla exhaustiva
    más abajo, generada automáticamente en cada ejecución canónica.
* **Features (dbNSFP / gnomAD):** `cadd_phred`, `sift_score`, `polyphen_score`,
  `revel_score` (los tres últimos solo definidos para variantes *missense*),
  `gerp_rs`, `phylop`, `gnomad_af`. Más `gene` y `consequence` (consecuencia funcional).
* **Acotación Fase I:** cromosomas 1 a 3 (config `data.chromosomes_subset`).

## Fuentes
| Fuente | Papel | Acceso |
|--------|-------|--------|
| ClinVar (NCBI) | etiquetas + drift temporal (releases fechadas) | VCF GRCh38 archive |
| dbNSFP | scores in silico precomputados (SNVs no sinónimas) | descarga académica |
| gnomAD | frecuencia alélica poblacional | incluida vía dbNSFP / BigQuery |

**Releases usadas:** entrenamiento `2023-12`, evaluación `2025-06` (drift temporal real).

## Proceso de recolección
`src/ingest/download.py` descarga las fuentes reales cuando hay red. En entornos
sin acceso a NCBI (allowlist), un **generador offline determinista** (ADR 005,
seed 42) produce ficheros con **idéntico esquema** (VCF de ClinVar + tabla
dbNSFP), preservando: correlación entre features y patogenicidad, ausencia de scores en no
missense, y drift temporal (variantes nuevas + VUS reclasificadas). Esto garantiza
reproducibilidad total del pipeline con o sin red.

## Preprocesamiento
`src/annotate/annotate.py` parsea el VCF, une con dbNSFP por clave, acota a los
cromosomas de config y valida el **contrato de datos** (`src/annotate/schema.py`).
`src/features/build_dataset.py` binariza el target y separa el conjunto GOLD.

## Usos y limitaciones
* **Uso previsto:** clasificación binaria supervisada con algoritmos estándar
  (regresión logística, random forest, gradient boosting) y monitorización de drift.
* **Limitaciones:** solo SNVs (no CNV/estructurales); subconjunto de cromosomas en
  Fase I; sin datos genómicos individuales identificables (solo bases agregadas y
  públicas). Cuando la fuente es el generador offline, los valores **no** son datos
  clínicos reales y no deben usarse para interpretación clínica.

## Tabla exhaustiva: CLNSIG crudo -> bucket (generada por `src/features/build_dataset.py`)

Regenerada en cada `make dataset`; fuente: `reports/training/clinvar_label_taxonomy.csv`.
ClinVar cambia el vocabulario de este campo entre releases (p. ej. `Conflicting_interpretations_of_pathogenicity`
en 2023-12 pasa a `Conflicting_classifications_of_pathogenicity` en 2025-06); el
código no depende de una lista cerrada de valores excluidos, solo de los tres
buckets con significado positivo/negativo/VUS explícito — todo lo demás cae en
"excluido" automáticamente, sin importar la ortografía exacta.

| split | release | clnsig crudo | n | bucket |
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

## Mantenimiento
Versionado de datos con DVC; `data/raw/MANIFEST.json` registra fuente, tamaños y
sha256 de cada fichero RAW (trazabilidad T2). `data/raw/` es inmutable.
