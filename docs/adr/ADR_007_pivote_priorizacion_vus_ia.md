# ADR 007: Pivote de enfoque — de "pipeline de anotación" a "plataforma de IA para priorización y reanálisis continuo de VUS"

**Fecha:** 2026-07-27 · **Estado:** **Confirmado** (alcance del núcleo ampliado a 7 puntos, ver §5).

## Contexto

Hasta ahora el proyecto se ha descrito y construido como un **pipeline de anotación y clasificación de variantes** (ingesta → anotación → entrenamiento → serving → monitorización), con el modelo de ML como un componente más de una cadena MLOps genérica. El proyecto reformula el foco: que el **elemento diferencial deje de ser el pipeline de anotación y pase a ser el modelo de IA de priorización de VUS** (Variants of Uncertain Significance), con la anotación como una etapa que alimenta ese núcleo, no el objetivo en sí. Se aborda además una revisión crítica y una investigación del estado del arte 2025-2026 para identificar dónde puede haber una aportación real, no solo "otro pipeline".

Motivo inmediato de esta decisión: al intentar ejecutar el pipeline con ClinVar real (revisión interna del proyecto), se confirmó que **dbNSFP requiere registro académico** no disponible, y que sin sus features el entrenamiento **no es posible en absoluto** (0 columnas con datos, no solo señal débil). Esto obliga a resolver de todos modos "qué fuente usar para anotar" — y esa misma pregunta es la puerta de entrada al pivote: en vez de depender de un agregador de pago/registro (dbNSFP), se puede construir la capa de anotación directamente sobre fuentes públicas sin registro, y usar esa libertad para escoger fuentes más modernas y más alineadas con 2025-2026 que las que trae dbNSFP.

## 1. Alternativa inmediata a dbNSFP (sin registro, 100% pública)

dbNSFP es, en esencia, un agregador que junta scores de otras fuentes. Se puede prescindir de él y consultar esas fuentes directamente:

| Fuente | Qué aporta | Acceso | Notas |
|---|---|---|---|
| **gnomAD** | Frecuencia alélica poblacional | Público, sin registro (Google Cloud Storage / Hail / API del navegador) | Ya se usa conceptualmente; hay que cambiar de "subset sintético" a la fuente real |
| **CADD** | Deleteriedad combinada (sustituye a el `cadd_phred`) | Público, descarga directa sin login ([cadd.gs.washington.edu](https://cadd.gs.washington.edu/api)) | Cubre todo el genoma, SNVs precalculados |
| **REVEL** | Score ensemble específico de missense | Público, descarga directa ([sites.google.com/site/revelgenomics](https://www.sites.google.com/site/revelgenomics)) | Sin registro conocido |
| **AlphaMissense** (DeepMind, 2023) | Patogenicidad de TODAS las variantes missense posibles, basado en AlphaFold | Público, Zenodo, CC-BY-NC-SA (uso académico OK) | Estado del arte en benchmarks clínicos curados; más "2025-2026" que SIFT/PolyPhen |
| **SpliceAI** (Illumina, open source) | Impacto en splicing | Paquete Python open source, se puede **calcular en local** sobre el genoma de referencia sin depender de un fichero precalculado gigante | Feature que dbNSFP ni siquiera cubre bien — hueco real que es posible llenar |
| **PrimateAI-3D** / **EVE** | Predictores basados en estructura / MSA, superan a AlphaMissense en algunos cohortes reales | Acceso más restringido (PrimateAI-3D es de Illumina; EVE cubre menos genes) | Candidatos de "trabajo futuro", no MVP |

**Recomendación:** sustituir `dbnsfp_subset.tsv.gz` por un módulo de anotación multi-fuente (gnomAD + CADD + REVEL + AlphaMissense + SpliceAI calculado), unidas por `(chrom,pos,ref,alt)` igual que ahora. Esto resuelve el bloqueo real de acceso a la fuente **y** es en sí mismo un argumento de "innovación": en vez de depender de un agregador desactualizado (dbNSFP usa SIFT/PolyPhen, de hace más de una década), el pipeline anota con las fuentes que de verdad se usan en 2025-2026.

Fuentes: [dbNSFP](https://sites.google.com/site/jpopgen/dbNSFP), [CADD API](https://cadd.gs.washington.edu/api), [REVEL](https://www.sites.google.com/site/revelgenomics), [AlphaMissense (Science 2023)](https://www.science.org/doi/10.1126/science.adg7492).

## 2. Estado del arte actual (2025-2026)

**Herramientas de priorización fenotípica (Exomiser, LIRICAL).** Combinan variante + fenotipo (HPO) + modo de herencia; Exomiser sitúa la variante diagnóstica en el top-1 el 74% de las veces (top-5 el 94%) cuando hay HPO disponible, pero cae a 3%/27% sin fenotipo. LIRICAL añade razonamiento probabilístico (likelihood ratio) sobre los mismos datos. **Limitación clave para este proyecto: requieren datos de paciente (fenotipo, a veces trío familiar) no disponible** — el dataset es ClinVar/gnomAD agregado, sin pacientes. ([Jacobsen et al. 2022](https://onlinelibrary.wiley.com/doi/full/10.1002/humu.24380), [Exomiser/Genomiser 2025](https://link.springer.com/article/10.1186/s13073-025-01546-1))

**Talos** (Centre for Population Genomics, Broad, Microsoft; Nature Medicine 2026). Reanálisis automático e iterativo de cohortes: combina PanelApp (relación gen-enfermedad, actualizada) + ClinVar (patogenicidad, actualizada) + filtrado por herencia, y vuelve a analizar pacientes ya secuenciados cuando cambia la evidencia. En un cohorte de 4.735 individuos sin diagnóstico encontró 241 diagnósticos nuevos (5,1%): 32% por nuevas relaciones gen-enfermedad, 22% por nueva evidencia de variante, 45% por mejoras del propio análisis. ([Nature Medicine 2026](https://www.nature.com/articles/s41591-026-04477-5), [GitHub](https://github.com/populationgenomics/talos))

**Modelos ML recientes para reclasificar VUS.** RENOVO (random forest sobre VUS de ClinVar reclasificadas en el tiempo) reclasifica ~67% de VUS existentes con alta confianza — es el precedente más cercano a lo que ya hace `prioritize_vus.py`. Trabajos 2025-2026 combinan datos de fenotipo poblacional + inferencia bayesiana (reclasificaron >1.000 VUS afectando a ~45.000 personas en un subconjunto de 17 genes), o usan ensayos funcionales masivos (MAVE, 62.215 variantes/10 genes) para calibrar automáticamente evidencia clínica y reclasificar el 75% de las VUS de esos genes con <1% de error. También hay modelos de lenguaje que detectan "huecos de evidencia" en ClinVar/ClinGen para priorizar qué VUS son más reclasificables. ([RENOVO](https://www.sciencedirect.com/science/article/pii/S000292972100094X), [ML+fenotipo 2025](https://link.springer.com/article/10.1007/s00439-025-02743-z), [MAVE/calibración 2026](https://www.biorxiv.org/content/10.64898/2026.02.14.705848v2.full))

**Predictores de efecto de variante de última generación.** AlphaMissense (estructura vía AlphaFold) es el más consistente en benchmarks clínicos curados; PrimateAI-3D (también basado en estructura) lo supera en cohortes reales (DDD, ASD, CHD, UK Biobank); EVE (alineamientos múltiples) y ESM-1v (modelo de lenguaje de proteínas) lideran benchmarks independientes. La tendencia 2023-2026 es incorporar **estructura de proteína**, no solo conservación/frecuencia. ([PrimateAI-3D vs AlphaMissense](https://www.medrxiv.org/content/10.1101/2024.01.12.24301193.full.pdf), [AlphaMissense](https://www.science.org/doi/10.1126/science.adg7492))

**Explicabilidad y ACMG.** La adopción clínica depende tanto de la explicabilidad como del rendimiento: los clínicos necesitan scores interpretables alineados con criterios ACMG/AMP, no solo una probabilidad. Herramientas como BIAS-2015 automatizan la clasificación ACMG completa con lógica basada en reglas, transparente. La propia industria anticipa la migración a un ACMG v4 cuantitativo. ([GigaScience 2026](https://academic.oup.com/gigascience/article/doi/10.1093/gigascience/giag004/8419681), [BIAS-2015](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12706976/))

## 3. Qué hacen los mejores proyectos — y sus límites (para el caso concreto)

| Proyecto | Qué hace bien | Por qué no es replicable tal cual |
|---|---|---|
| Talos | Reanálisis automático a escala de cohorte, gen-enfermedad + variante | Necesita PanelApp + datos de paciente/pedigrí/herencia; este proyecto solo hay variantes agregadas públicas, sin pacientes |
| Exomiser/LIRICAL | Prioriza con fenotipo (HPO) | Necesita HPO **del paciente**; este proyecto no está disponible casos clínicos, solo ClinVar/gnomAD |
| RENOVO | RF sobre reclasificaciones históricas de ClinVar | Es el más cercano a lo que YA hace el proyecto (`prioritize_vus.py` + la anotación y la monitorización); la diferencia es que no explota el par de releases fechadas como *par de entrenamiento supervisado* para un segundo modelo (ver §5) |
| AlphaMissense/PrimateAI-3D/EVE | Predicción de efecto muy precisa | Son **inputs** (features), no sustituyen la priorización orientada a decisión clínica ni la explicabilidad ACMG |

**Conclusión:** ningún proyecto de referencia hace exactamente lo que permiten los datos disponibles (datos agregados públicos, sin pacientes, con dos releases fechadas de ClinVar). Eso es también la limitación de el alcance: **no es posible** replicar Talos/Exomiser porque no está disponible capa de paciente/fenotipo — y no deberíamos fingir que sí.

## 4. Oportunidad de innovación real (dado lo que hay)

La ventaja distintiva ya disponible y que **ningún proyecto de referencia explota exactamente así**: dos releases fechadas de ClinVar (train=antigua, test=nueva) con reclasificaciones reales conocidas entre ellas. Eso permite entrenar no solo "¿es patogénica?" sino **"¿qué probabilidad tiene esta VUS concreta de ser reclasificada pronto?"** — un modelo de **potencial de reclasificación**, no solo de patogenicidad. Es la pieza que más se alinea con lo que El proyecto reformula en el punto de "predicción del potencial de reclasificación futura" y es factible con los datos ya disponibles (no requiere fenotipo de paciente ni cohortes).

## 5. Funcionalidades diferenciales — priorizadas por impacto/factibilidad

**Núcleo confirmado del TFM (7 puntos, factibles sin datos de paciente, sin tarjeta, en el tiempo disponible):**

1. **Anotación multi-fuente sin dbNSFP** (§1): CADD + REVEL + AlphaMissense + SpliceAI + gnomAD reales, directamente. Resuelve el bloqueo de acceso y es en sí mismo el diferencial de anotación.
2. **Segundo modelo: "potencial de reclasificación"** (§4). Usa el par train=2023-12/test=2025-06 como entrenamiento supervisado directo (¿se reclasificó esta VUS de train a test?), en vez de solo medir drift a posteriori. Complementa (no sustituye) el modelo de patogenicidad ya existente.
3. **Objetivo de *ranking*, no solo clasificación.** El entregable real es un orden de prioridad, no una probabilidad aislada. Cambiar (o añadir) un objetivo de *learning-to-rank* (p. ej. LightGBM `lambdarank`) y evaluar con métricas de ranking (NDCG@k) además de PR-AUC. Encaja de forma natural con "priorización", que es justo el nuevo foco.
4. **Explicabilidad → evidencia ACMG.** Traducir las contribuciones SHAP ya existentes a un lenguaje de evidencia tipo ACMG/AMP (p. ej. CADD/REVEL/AlphaMissense altos → "PP3-like"; AF poblacional muy baja → "PM2-like"), con la advertencia explícita de que es una heurística, no una clasificación ACMG certificada. Es la pieza que la literatura señala como crítica para adopción clínica y que hoy nadie de el tamaño hace bien de forma automática y transparente.
5. **Reanálisis por cambio de evidencia como bucle central** (ya parcialmente construido: esa etapa, esa etapa). Reformular la narrativa: el pipeline no es "entrena una vez y sirve", es un bucle continuo variante→score→revisión→reclasificación en ClinVar→reentrenamiento. Esto es, en pequeño y sin cohorte de pacientes, la misma idea de fondo que Talos aplica a escala clínica — y se puede citar así en la memoria (inspiración explícita, no réplica).
6. **Dashboard interactivo** para explorar variantes, scores y evidencia (buen valor de demo para la defensa). Añadido al núcleo incorporado al alcance; se decidirá stack (Streamlit vs. HTML+Flask ya existente) en la fase de implementación.
7. **Informes automáticos por VUS basados en plantilla** (no LLM libre, para evitar alucinaciones en un contexto clínico) que citen qué evidencia empujó el score. Añadido al núcleo incorporado al alcance.

**Descartado explícitamente para esta TFM (documentar como trabajo futuro, no construir):**

* Integración literal con Talos/Exomiser/Seqr/LIRICAL: todos requieren datos de paciente (fenotipo HPO, pedigrí, herencia) no disponible y que están fuera del alcance de datos públicos agregados del anteproyecto.
* Modelado de tríos/herencia y recomendación de genes candidatos: requiere cohortes de pacientes, no datos públicos de variantes.
* Arquitecturas GNN/Transformer: no hay una estructura de grafo o secuencia clara a nivel de variante en el problema tabular; el coste de ingeniería y datos no está justificado frente al alcance y el calendario (el calendario del proyecto). Documentar como alternativa evaluada y descartada, con motivo.
* PrimateAI-3D/EVE como fuente primaria: acceso más restringido/cobertura limitada; mencionar como trabajo futuro.

## 6. Qué tendría mayor impacto científico y clínico

Por este orden: (1) el modelo de **potencial de reclasificación** (§5.2) — es la idea genuinamente nueva y defendible como aportación, no solo ingeniería; (2) la **explicabilidad ACMG-aware** (§5.4) — es lo que la literatura marca como determinante para adopción clínica real, y hoy es un hueco; (3) la anotación multi-fuente sin dbNSFP (§5.1) — necesaria, pero es "ponerse al día", no innovación en sí; (4) el objetivo de ranking (§5.3) — mejora de rigor metodológico, buen argumento metodológico en la memoria.

## 7. Propuesta de arquitectura (modelo de IA como núcleo)

```
                    ┌─────────────────────────────────────────┐
                    │ NÚCLEO: motor de priorización de VUS │
                    │ (patogenicidad + potencial de reclasif.)│
                    │ objetivo ranking + explicable ACMG │
                    └───────────────┬───────────────────────────┘
                                    │ alimentado por
        ┌───────────────────────────┼───────────────────────────┐
        │ │ │
┌───────▼────────┐ ┌────────▼────────┐ ┌────────▼────────┐
│ Anotación │ │ Historial de │ │ Registro/ │
│ multi-fuente │ │ releases ClinVar │ │ reentrenamiento │
│ (gnomAD, CADD, │ │ (train=antigua, │ │ MLflow + gate │
│ REVEL, Alpha- │ │ test=nueva) → │ │ humano, ya │
│ Missense, │ │ etiqueta de │ │ construido │
│ SpliceAI) │ │ reclasificación │ │ (esa etapa) │
└─────────────────┘ └──────────────────┘ └─────────────────┘
        │ ▲
        └──────────────── nueva release de ClinVar ────────────────┘
                       (dispara re-anotación → re-score → alerta)

Salida: lista de VUS ordenada por prioridad, cada una con:
  - probabilidad de patogenicidad
  - probabilidad de reclasificación próxima
  - explicación SHAP → evidencia tipo ACMG
  - servida vía REST (ya construido, D1/D2) — diseñada para poder ser
    consumida por un sistema tipo Talos/Seqr como fuente de evidencia
    computacional, sin pretender sustituirlo.
```

Esto conserva casi todo lo ya construido (ingesta, contrato de datos, MLflow, REST, monitorización, reentrenamiento) — el pivote es de **narrativa y de dónde está el peso**: la anotación pasa de ser "el proyecto" a ser "el input del modelo", y se añaden las piezas de §5 (1-5) como núcleo diferencial.

## Decisión final (2026-07-27)

* Núcleo confirmado: los 7 puntos de §5 (se añadieron dashboard e informes por plantilla al núcleo, inicialmente propuestos como extensión opcional).
* Orden de trabajo: **primero** actualizar `README.md` y el plan del proyecto (narrativa y alcance documentado) y **después** implementar, empezando por la anotación multi-fuente (§1), que desbloquea datos reales para todo lo demás.
* Pendiente explícito: comunicar este pivote al director (amplía el alcance firmado más allá de lo ya comunicado en D1/D3/D4 del el plan del proyecto; ver riesgo R8). No se ha redactado el correo (igual que A2 original, lo redacta la autora).

## Actualización (2026-08-07): el objetivo de ranking (§5.3) integrado como criterio de orden real

Hasta esta fecha, el objetivo de *ranking* (§5.3) se entrenaba y evaluaba
(`src/train/train_ranking.py`, NDCG@k) pero el orden de prioridad servido
realmente por `src/serve/prioritize_vus.py` y `src/serve/vus_reports.py`
 usaba solo la probabilidad de patogenicidad del modelo de patogenicidad — una auditoría
posterior de la plataforma detectó esta desconexión entre "lo que se evalúa"
y "lo que decide el orden real" y la señaló como riesgo de sobrerrepresentar
la integración del objetivo de ranking en la memoria.

**Corrección aplicada:** `train_ranking.py` persiste ahora, junto al booster
(`models/ranking_model/lambdarank.txt`), el preprocesador ya ajustado con el
que se entrenó (`models/ranking_model/preprocessor.joblib`) — necesario para
transformar features nuevas exactamente igual que en entrenamiento sin
reajustarlo en cada carga. `src/serve/prioritize_vus.py::load_ranking_model`
carga ambos; `rank_vus` ordena por el score del objetivo de ranking cuando el modelo existe, y
degrada explícitamente (mismo patrón que el resto del proyecto, ADR 005) a la
probabilidad del modelo de patogenicidad si `make train-ranking` no se ha ejecutado todavía. el generador de informes por VUS y el
dashboard reutilizan el mismo criterio y lo declaran de forma visible
(cabecera dinámica, columna `ranking_score` en los JSON de informes), para que
nunca haya ambigüedad sobre qué decidió el orden mostrado.

Esto cierra la brecha entre §5.3 ("el entregable real es un orden de
prioridad") y el comportamiento real del sistema: el objetivo de ranking deja de ser una pieza de
validación metodológica aislada y pasa a ser el criterio de orden efectivo del
núcleo de priorización.
