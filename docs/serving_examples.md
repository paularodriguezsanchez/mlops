# Servicio de inferencia: variantes de prueba

El servicio recibe una variante `(chrom, pos, ref, alt)`, la anota -consecuencia
funcional, scores in silico y frecuencia poblacional- y devuelve la predicción de
patogenicidad con su probabilidad. Estas 10 variantes, mitad patogénicas y mitad
benignas según ClinVar, no se vieron en entrenamiento; aciertos: **9/10**.

Es una prueba de integración extremo a extremo del servicio, no una estimación de
rendimiento: para eso está el PR AUC sobre el conjunto de evaluación completo.

| variante | gen | consecuencia | clnsig_real | clase_real | prediccion | prob_patogenica | acierto |
|:----------------|:---------|:-----------------------------------|:------------------|:-------------|:-------------|------------------:|:----------|
| 1-63402252-A-G | ALG6 | SO:0001574|splice_acceptor_variant | Likely_pathogenic | Patogénica | Patogénica | 0.9757 | ✓ |
| 1-173833410-T-C | DARS2 | SO:0001583|missense_variant | Likely_pathogenic | Patogénica | Patogénica | 0.992 | ✓ |
| 3-11332986-A-G | ATG7 | SO:0001583|missense_variant | Pathogenic | Patogénica | Benigna | 0.1047 | ✗ |
| 2-54618178-C-G | SPTBN1 | SO:0001583|missense_variant | Likely_pathogenic | Patogénica | Patogénica | 0.9873 | ✓ |
| 2-227293255-G-A | COL4A3 | SO:0001583|missense_variant | Likely_pathogenic | Patogénica | Patogénica | 0.9922 | ✓ |
| 2-203870699-C-A | CTLA4 | SO:0001819|synonymous_variant | Likely_benign | Benigna | Benigna | 0.0085 | ✓ |
| 3-52372038-C-T | DNAH1 | SO:0001819|synonymous_variant | Likely_benign | Benigna | Benigna | 0.0003 | ✓ |
| 3-129034216-G-A | EFCC1 | SO:0001583|missense_variant | Likely_benign | Benigna | Benigna | 0.005 | ✓ |
| 3-193363248-C-T | ATP13A5 | SO:0001819|synonymous_variant | Benign | Benigna | Benigna | 0 | ✓ |
| 2-135130628-A-G | RAB3GAP1 | SO:0001819|synonymous_variant | Likely_benign | Benigna | Benigna | 0.0002 | ✓ |

`clnsig_real` es la etiqueta de referencia y no se pasa al modelo; sirve solo para
verificar el acierto. La respuesta completa de cada variante está en
`reports/serving/example_variants.json`.

## Reproducir
```bash
make serve
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' \
     -d '{"chrom":"3","pos":11825913,"ref":"G","alt":"T"}'
```
