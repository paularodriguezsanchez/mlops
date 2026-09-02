-- Fase II (diseño, no ejecutado). Capa SILVER: variante x features, equivalente
-- a `src/annotate/annotate.py` / `src/annotate/multi_source.py`, pero como join
-- SQL contra fuentes ya alojadas en BigQuery en vez de consultas HTTP a
-- myvariant.info. Reproduce la paridad de columnas con la capa SILVER local
-- (mismas 15 columnas del contrato de datos, `src/annotate/schema.py`).
--
-- gnomAD SÍ está disponible como dataset público real en BigQuery (verificado):
-- `bigquery-public-data.gnomAD.v3_genomes__chr{N}` (particionado/clustered por
-- cromosoma). CADD, REVEL y AlphaMissense NO están alojados como dataset público
-- en BigQuery; en un despliegue real habría que cargar sus ficheros precalculados
-- a Cloud Storage (igual que se hace hoy vía myvariant.info en local) y crear
-- tablas externas equivalentes — no reimplementadas aquí porque exceden el
-- alcance de "diseño documentado" de la Opción A.

CREATE SCHEMA IF NOT EXISTS `tfm_variantes.silver`
  OPTIONS (location = 'EU', description = 'Capa SILVER: variantes anotadas por release');

CREATE OR REPLACE TABLE `tfm_variantes.silver.annotated_{release}` AS
SELECT
  c.chrom,
  c.pos,
  c.ref,
  c.alt,
  c.gene,
  c.consequence,
  c.clnsig,
  -- CADD / SIFT / PolyPhen / REVEL / AlphaMissense: en un despliegue real,
  -- joins contra tablas externas cargadas desde Cloud Storage (ver nota arriba).
  NULL AS sift_score,
  NULL AS polyphen_score,
  NULL AS cadd_phred,
  NULL AS revel_score,
  NULL AS alphamissense_score,
  NULL AS gerp_rs,
  NULL AS phylop,
  g.af  AS gnomad_af
FROM `tfm_variantes.raw.clinvar_variants` c
LEFT JOIN `bigquery-public-data.gnomAD.v3_genomes__chr1` g
  ON g.chrom = c.chrom AND g.pos = c.pos AND g.ref = c.ref AND g.alt = c.alt
WHERE c.release = '{release}'
  AND c.chrom IN ('1', '2', '3');  -- mismo acotado que `chromosomes_subset` en config.yaml
