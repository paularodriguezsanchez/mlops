# ADR 002: Selección del dominio y del conjunto de datos

**Fecha:** 2026-07-06 · **Estado:** aceptada

## Contexto

El anteproyecto pedía un conjunto biomédico público que permitiera un problema supervisado y el ciclo MLOps completo, y ponía como ejemplo conjuntos tabulares clásicos del tipo Heart Disease o Diabetes. Son conjuntos estáticos, publicados en una única versión: cualquier deriva solo podría simularse inyectando un cambio artificial.

## Decisión

Concreto el dominio a la clasificación de patogenicidad de variantes genéticas: **ClinVar** como fuente de etiquetas y **gnomAD junto a los predictores in silico de referencia** como fuente de features, con la consecuencia funcional del propio VCF. Foco en SNVs sobre GRCh38.

Descarto los conjuntos tabulares clásicos por dos motivos: no permiten observar deriva real, solo simulada, y no aprovechan mi experiencia previa en genómica y en plataformas de datos en la nube. Tras la anotación el problema sigue siendo tabular, así que se mantiene el requisito de usar algoritmos estándar.

## Consecuencias

* Es la combinación habitual en la literatura de predicción de patogenicidad, lo que permite comparar con REVEL, CADD o RENOVO.
* Es reproducible por cualquier tercero: todo público, sin registro.
* Habilita deriva temporal real usando dos *releases* fechadas, que acabó siendo la base de la aportación central del trabajo.
* El volumen obliga a acotar por cromosoma y por número de variantes.
* El desbalance de clases es estructural del dominio y fija PR AUC como métrica principal desde el principio.
