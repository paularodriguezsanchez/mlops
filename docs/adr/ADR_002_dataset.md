# ADR 002: Selección del dataset

* **Fecha:** 2026-07-06
* **Estado:** Aceptada

## Contexto
Se necesita un dataset de genómica público, con variantes e información asociada, que permita
un problema supervisado y simular el ciclo completo. El anteproyecto ponía como ejemplo datasets
clínicos tabulares (Heart Disease/Diabetes), pero el requisito real es "dataset biomédico público".

## Decisión
Usar **ClinVar** como fuente de etiquetas (significancia clínica: patogénica/benigna) y
**dbNSFP + gnomAD** como fuente de features (scores CADD/SIFT/PolyPhen/REVEL, conservación,
frecuencia alélica). Consecuencia funcional vía VEP/Ensembl. Foco en SNVs, ensamblaje GRCh38.

## Consecuencias
* A favor: Combinación estándar en la literatura de predicción de patogenicidad (REVEL, ClinPred).
* A favor: 100 % pública y reproducible; disponible en local (descarga) y en BigQuery (cloud).
* A favor: Habilita *concept drift* temporal real usando releases fechadas de ClinVar.
* En contra: Volumen grande (dbNSFP/gnomAD): se acota por cromosoma/tipo en Fase I (config).
* En contra: Desbalanceo de clases: se maneja con métricas PR AUC/F1 y balanceo.
