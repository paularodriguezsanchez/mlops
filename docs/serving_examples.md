# Servicio de inferencia: variantes de prueba documentadas [OE4]

El servicio recibe una variante `(chrom, pos, ref, alt)`, la **anota**
(consecuencia funcional + scores in silico + frecuencia) y devuelve la
**predicción de patogenicidad** con su probabilidad. Se documentan 10 variantes
(mitad patogénicas, mitad benignas según ClinVar); aciertos: **9/10**.

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

## Reproducir
```bash
make serve # levanta el servicio REST en:8000
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' \
     -d '{"chrom":"3","pos":11825913,"ref":"G","alt":"T"}'
```

La respuesta completa (entrada + salida) de cada variante está en
`reports/serving/example_variants.json`.

> Nota: `clnsig_real` es la etiqueta de ClinVar (verdad de referencia) y **no** se
> pasa al modelo; sirve solo para verificar el acierto. Si los datos provienen del
> generador offline (ADR 005), estas predicciones validan el pipeline, no tienen
> valor clínico.
