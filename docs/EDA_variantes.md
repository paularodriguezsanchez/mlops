# Análisis exploratorio de las variantes anotadas

Resumen de `notebooks/01_eda_variantes.ipynb`, sobre la ejecución canónica del pipeline.

## Volumen y balance del target

| Partición | Release | Etiquetadas | Patogénicas | Prevalencia | VUS reservadas | Excluidas |
|---|---|---:|---:|---:|---:|---:|
| train | 2023-12 | 3504 | 526 | 0,150 | 4056 | 437 |
| test | 2025-06 | 5221 | 717 | 0,137 | 6042 | 734 |

La clase positiva es minoritaria, lo que fija PR AUC como métrica principal en todo el proyecto. Las VUS no se descartan: se reservan como conjunto de inferencia (ADR 006). La taxonomía completa `CLNSIG` -> etiqueta está en `datasheet.md`.

![Balance del target](figuras/eda_target_balance.png)

## Separabilidad de las features

CADD, REVEL y la frecuencia de gnomAD en escala logarítmica separan las clases, pero con solapamiento apreciable. Ese solapamiento no es un defecto de los datos: es el error de Bayes que también limita a los predictores reales de la literatura, y confirma que el problema es aprendible sin ser trivial.

![Separabilidad](figuras/eda_feature_separability.png)

Las consecuencias truncantes (`stop_gained`, `frameshift_variant`, `splice_*`) concentran patogenicidad; las sinónimas, benignidad.

![Consecuencia](figuras/eda_consequence.png)

## Deriva entre releases (2023-12 -> 2025-06)

* Variantes compartidas entre ambas releases: **7997**.
* VUS estrictas reclasificadas a un veredicto resuelto: **55** (0,69 % de las compartidas).
* Variantes presentes solo en la release nueva: **4000**.

Es deriva real, no simulada: reproduce el fenómeno clínico de resolución progresiva de VUS entre publicaciones de ClinVar, y es la señal que sostiene tanto la monitorización como el entrenamiento del modelo de potencial de reclasificación.

![Deriva de CLNSIG](figuras/eda_clnsig_drift.png)

## Implicaciones para el modelado

* Métricas robustas al desbalance: PR AUC, F1 y ROC AUC, además de la matriz de confusión; nunca exactitud.
* Imputación obligatoria: SIFT, PolyPhen y REVEL son nulos fuera de missense, y con datos reales también CADD, GERP y phyloP presentan una tasa alta de nulos. La ausencia se conserva como indicador explícito, no se disuelve en la mediana.
* La partición temporal permite medir degradación realista, pero exige filtrar las variantes que persisten entre releases antes de evaluar.
