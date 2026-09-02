-- Extensión a GCP: diseño, no ejecutado. Capa GOLD: equivalente SQL de
-- `src/features/build_dataset.py` — misma regla de binarización del target
-- (config.yaml: target.positive_labels / negative_labels) y misma separación
-- de VUS reservadas (ADR 006), pero como vistas SQL en vez de ficheros parquet.

CREATE SCHEMA IF NOT EXISTS `tfm_variantes.gold`
  OPTIONS (location = 'EU', description = 'Capa GOLD: dataset de modelado (train/test) + VUS reservadas');

CREATE OR REPLACE VIEW `tfm_variantes.gold.train` AS
SELECT *,
  CASE
    WHEN clnsig IN ('Pathogenic', 'Likely_pathogenic', 'Pathogenic/Likely_pathogenic') THEN 1
    WHEN clnsig IN ('Benign', 'Likely_benign', 'Benign/Likely_benign') THEN 0
  END AS label
FROM `tfm_variantes.silver.annotated_2023-12`
WHERE clnsig IN ('Pathogenic', 'Likely_pathogenic', 'Pathogenic/Likely_pathogenic',
                  'Benign', 'Likely_benign', 'Benign/Likely_benign');

CREATE OR REPLACE VIEW `tfm_variantes.gold.vus_train` AS
SELECT * EXCEPT(clnsig), clnsig
FROM `tfm_variantes.silver.annotated_2023-12`
WHERE clnsig NOT IN ('Pathogenic', 'Likely_pathogenic', 'Pathogenic/Likely_pathogenic',
                      'Benign', 'Likely_benign', 'Benign/Likely_benign');

-- `gold.test` y `gold.vus_test` son análogas, sobre `silver.annotated_2025-06`.
