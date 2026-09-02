# Priorización de VUS: release test

No es un veredicto clínico ni sustituye la curación manual: dirige la revisión que
un experto haría de todos modos, apoyándose en el conocimiento ya acumulado en
ClinVar, gnomAD, CADD, REVEL y AlphaMissense. Se revisan antes las de arriba,
ordenadas por el score del modelo de ranking (LightGBM `lambdarank`, entrenado para optimizar directamente el orden y evaluado con NDCG@k; ver ADR 007 §5.3 y `reports/training/ranking_metrics.csv`). La probabilidad de patogenicidad se muestra como referencia adicional.

6042 VUS reservadas en total. Top 20 por prioridad:

| chrom | pos | ref | alt | gene | consequence | clnsig | probabilidad_patogenica | ranking_score |
|--------:|----------:|:------|:------|:--------|:----------------------------|:-----------------------|--------------------------:|----------------:|
| 2 | 152718816 | G | A | ARL6IP6 | SO:0001587|nonsense | Uncertain_significance | 0.996631 | 6.34602 |
| 2 | 27522472 | C | T | GCKR | SO:0001587|nonsense | Uncertain_significance | 0.971916 | 6.32651 |
| 2 | 162288281 | G | A | IFIH1 | SO:0001587|nonsense | Uncertain_significance | 0.98011 | 6.32649 |
| 2 | 29383840 | T | A | ALK | SO:0001587|nonsense | Uncertain_significance | 0.999566 | 6.32626 |
| 3 | 179414906 | G | A | GNB4 | SO:0001587|nonsense | Uncertain_significance | 0.992687 | 6.23912 |
| 2 | 70224622 | G | A | TIA1 | SO:0001587|nonsense | Uncertain_significance | 0.999139 | 6.22806 |
| 3 | 119523649 | G | T | TIMMDC1 | SO:0001587|nonsense | Uncertain_significance | 0.999292 | 6.21943 |
| 2 | 169812501 | G | A | METTL5 | SO:0001587|nonsense | Uncertain_significance | 0.999713 | 6.21905 |
| 2 | 85371131 | C | T | ELMOD3 | SO:0001587|nonsense | Uncertain_significance | 0.984371 | 6.04251 |
| 1 | 201226983 | G | T | IGFN1 | SO:0001587|nonsense | Uncertain_significance | 0.99945 | 5.98247 |
| 2 | 29717584 | G | A | ALK | SO:0001587|nonsense | Uncertain_significance | 0.99297 | 5.65057 |
| 1 | 55008467 | C | T | BSND | SO:0001587|nonsense | Uncertain_significance | 0.998843 | 5.65039 |
| 1 | 186081383 | C | T | HMCN1 | SO:0001587|nonsense | Uncertain_significance | 0.981729 | 5.60357 |
| 2 | 74462871 | C | T | MOGS | SO:0001587|nonsense | Uncertain_significance | 0.993067 | 5.50812 |
| 3 | 14129463 | C | T | TMEM43 | SO:0001587|nonsense | Uncertain_significance | 0.999038 | 5.50742 |
| 2 | 49922063 | G | A | NRXN1 | SO:0001587|nonsense | Uncertain_significance | 0.999542 | 5.49437 |
| 1 | 39485594 | C | T | MACF1 | SO:0001587|nonsense | Uncertain_significance | 0.999911 | 5.41398 |
| 1 | 210683309 | G | T | KCNH1 | SO:0001587|nonsense | Uncertain_significance | 0.988478 | 5.35572 |
| 3 | 52402293 | G | A | BAP1 | SO:0001587|nonsense | Uncertain_significance | 0.991834 | 5.06996 |
| 1 | 155611080 | G | T | MSTO1 | SO:0001583|missense_variant | Uncertain_significance | 0.925834 | 4.69179 |

Lista completa (todas las VUS, ordenadas): `reports/serving/vus_priorizadas_test.csv`.

## Reproducir
```bash
make prioritize # release de test, top 20
python -m src.serve.prioritize_vus --split train --top-n 30
```
