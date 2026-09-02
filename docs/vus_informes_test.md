# Informes de VUS priorizadas: release test

Generados por plantilla (no por un LLM libre) a partir de los modelos y la
explicabilidad ya construidos: probabilidad de patogenicidad
(`src/serve/predictor.py`), probabilidad de reclasificación próxima
(`src, train, train_reclass.py`) y evidencia SHAP traducida a lenguaje tipo
ACMG/AMP (`src, evaluate, acmg_evidence.py`). **No es una clasificación
ACMG certificada** ni sustituye la curación clínica experta. Las variantes se ordenan por el score del modelo de ranking dedicado (LightGBM `lambdarank`) — ver `src/serve/prioritize_vus.py`.

6042 VUS reservadas en total en esta release. Informes de las
10 priorizadas primero:

### 2:152718816 G>A (ARL6IP6)

* **Consecuencia:** SO:0001587|nonsense
* **Probabilidad de patogenicidad:** 99.7%
* **Probabilidad de reclasificación próxima:** 26.1%
* **Evidencia computacional** (heurística tipo ACMG/AMP, no una clasificación certificada):
  * **PP3-like** — CADD combinado sugiere efecto deletéreo (valor=35.0, contribución SHAP=+0.288)
  * **PVS1-like** — consecuencia de pérdida de función predicha (SO:0001587|nonsense) (valor=SO:0001587|nonsense, contribución SHAP=+0.074)
  * **PM2-like** — variante rara/ausente en gnomAD (frecuencia poblacional muy baja) (valor=4.26519e-06, contribución SHAP=+0.007)

### 2:27522472 C>T (GCKR)

* **Consecuencia:** SO:0001587|nonsense
* **Probabilidad de patogenicidad:** 97.2%
* **Probabilidad de reclasificación próxima:** 0.8%
* **Evidencia computacional** (heurística tipo ACMG/AMP, no una clasificación certificada):
  * **PP3-like** — CADD combinado sugiere efecto deletéreo (valor=37.0, contribución SHAP=+0.371)
  * **PVS1-like** — consecuencia de pérdida de función predicha (SO:0001587|nonsense) (valor=SO:0001587|nonsense, contribución SHAP=+0.083)

### 2:162288281 G>A (IFIH1)

* **Consecuencia:** SO:0001587|nonsense
* **Probabilidad de patogenicidad:** 98.0%
* **Probabilidad de reclasificación próxima:** 4.2%
* **Evidencia computacional** (heurística tipo ACMG/AMP, no una clasificación certificada):
  * **PP3-like** — CADD combinado sugiere efecto deletéreo (valor=37.0, contribución SHAP=+0.363)
  * **PVS1-like** — consecuencia de pérdida de función predicha (SO:0001587|nonsense) (valor=SO:0001587|nonsense, contribución SHAP=+0.081)

### 2:29383840 T>A (ALK)

* **Consecuencia:** SO:0001587|nonsense
* **Probabilidad de patogenicidad:** 100.0%
* **Probabilidad de reclasificación próxima:** 4.8%
* **Evidencia computacional** (heurística tipo ACMG/AMP, no una clasificación certificada):
  * **PP3-like** — CADD combinado sugiere efecto deletéreo (valor=38.0, contribución SHAP=+0.288)
  * **PVS1-like** — consecuencia de pérdida de función predicha (SO:0001587|nonsense) (valor=SO:0001587|nonsense, contribución SHAP=+0.065)

### 3:179414906 G>A (GNB4)

* **Consecuencia:** SO:0001587|nonsense
* **Probabilidad de patogenicidad:** 99.3%
* **Probabilidad de reclasificación próxima:** 0.7%
* **Evidencia computacional** (heurística tipo ACMG/AMP, no una clasificación certificada):
  * **PP3-like** — CADD combinado sugiere efecto deletéreo (valor=36.0, contribución SHAP=+0.329)
  * **PVS1-like** — consecuencia de pérdida de función predicha (SO:0001587|nonsense) (valor=SO:0001587|nonsense, contribución SHAP=+0.078)

### 2:70224622 G>A (TIA1)

* **Consecuencia:** SO:0001587|nonsense
* **Probabilidad de patogenicidad:** 99.9%
* **Probabilidad de reclasificación próxima:** 4.5%
* **Evidencia computacional** (heurística tipo ACMG/AMP, no una clasificación certificada):
  * **PP3-like** — CADD combinado sugiere efecto deletéreo (valor=38.0, contribución SHAP=+0.309)
  * **PP3-like** — posición evolutivamente muy conservada (phyloP) (valor=5.825, contribución SHAP=+0.122)
  * **PVS1-like** — consecuencia de pérdida de función predicha (SO:0001587|nonsense) (valor=SO:0001587|nonsense, contribución SHAP=+0.067)

### 3:119523649 G>T (TIMMDC1)

* **Consecuencia:** SO:0001587|nonsense
* **Probabilidad de patogenicidad:** 99.9%
* **Probabilidad de reclasificación próxima:** 3.4%
* **Evidencia computacional** (heurística tipo ACMG/AMP, no una clasificación certificada):
  * **PP3-like** — CADD combinado sugiere efecto deletéreo (valor=36.0, contribución SHAP=+0.273)
  * **PVS1-like** — consecuencia de pérdida de función predicha (SO:0001587|nonsense) (valor=SO:0001587|nonsense, contribución SHAP=+0.068)

### 2:169812501 G>A (METTL5)

* **Consecuencia:** SO:0001587|nonsense
* **Probabilidad de patogenicidad:** 100.0%
* **Probabilidad de reclasificación próxima:** 4.4%
* **Evidencia computacional** (heurística tipo ACMG/AMP, no una clasificación certificada):
  * **PP3-like** — CADD combinado sugiere efecto deletéreo (valor=41.0, contribución SHAP=+0.317)
  * **PVS1-like** — consecuencia de pérdida de función predicha (SO:0001587|nonsense) (valor=SO:0001587|nonsense, contribución SHAP=+0.062)

### 2:85371131 C>T (ELMOD3)

* **Consecuencia:** SO:0001587|nonsense
* **Probabilidad de patogenicidad:** 98.4%
* **Probabilidad de reclasificación próxima:** 3.1%
* **Evidencia computacional** (heurística tipo ACMG/AMP, no una clasificación certificada):
  * **PP3-like** — CADD combinado sugiere efecto deletéreo (valor=37.0, contribución SHAP=+0.362)
  * **PP3-like** — posición evolutivamente muy conservada (GERP++) (valor=4.2, contribución SHAP=+0.156)
  * **PVS1-like** — consecuencia de pérdida de función predicha (SO:0001587|nonsense) (valor=SO:0001587|nonsense, contribución SHAP=+0.080)

### 1:201226983 G>T (IGFN1)

* **Consecuencia:** SO:0001587|nonsense
* **Probabilidad de patogenicidad:** 99.9%
* **Probabilidad de reclasificación próxima:** 5.0%
* **Evidencia computacional** (heurística tipo ACMG/AMP, no una clasificación certificada):
  * **PP3-like** — CADD combinado sugiere efecto deletéreo (valor=42.0, contribución SHAP=+0.355)
  * **PP3-like** — posición evolutivamente muy conservada (GERP++) (valor=4.28, contribución SHAP=+0.124)
  * **PP3-like** — posición evolutivamente muy conservada (phyloP) (valor=6.118, contribución SHAP=+0.112)
  * **PVS1-like** — consecuencia de pérdida de función predicha (SO:0001587|nonsense) (valor=SO:0001587|nonsense, contribución SHAP=+0.063)


Datos estructurados (para el dashboard): `reports/serving/vus_reports_test.json`.

## Reproducir
```bash
python -m src.serve.vus_reports --split test --top-n 10
```
