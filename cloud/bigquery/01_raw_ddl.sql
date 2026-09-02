-- Fase II (diseño, no ejecutado — Opción A, ver docs/FASE_II_DISENO_GCP.md).
-- Capa RAW: variantes de ClinVar cargadas desde los mismos VCF fechados que usa
-- la Fase I local (data/raw/clinvar_{release}.vcf.gz), subidos a Cloud Storage.
--
-- Nota importante de diseño: el dataset público `bigquery-public-data.human_variant_annotation`
-- solo expone un volcado estático de ClinVar (hg19/hg38, fechado 2018-07-01), no las
-- releases fechadas (2023-12 / 2025-06) que este proyecto necesita para el drift
-- temporal real. Por eso la capa RAW se carga desde los mismos ficheros VCF que
-- ya descarga `src/ingest/download.py`, subidos a un bucket de Cloud Storage
-- (gs://<proyecto>-tfm-raw/clinvar/), y no directamente desde el dataset público.

CREATE SCHEMA IF NOT EXISTS `tfm_variantes.raw`
  OPTIONS (location = 'EU', description = 'Capa RAW: VCF de ClinVar tal cual, inmutable, por release');

-- Tabla externa sobre el VCF ya descomprimido y convertido a NDJSON/Parquet en GCS
-- (la conversión VCF->Parquet se hace una vez, fuera de BigQuery, reutilizando
-- `src/annotate/annotate.py::parse_clinvar_vcf`, no reimplementada en SQL).
CREATE OR REPLACE EXTERNAL TABLE `tfm_variantes.raw.clinvar_variants`
(
  chrom     STRING,
  pos       INT64,
  ref       STRING,
  alt       STRING,
  clnsig    STRING,
  gene      STRING,
  consequence STRING,
  release   STRING   -- '2023-12' | '2025-06', para poder filtrar por release fechada
)
OPTIONS (
  format = 'PARQUET',
  uris = ['gs://<proyecto>-tfm-raw/clinvar/*.parquet']
);
