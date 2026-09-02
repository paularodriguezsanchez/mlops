# Informe técnico completo del TFM — Plataforma de IA de priorización de VUS

**Propósito de este documento:** dar una visión de extremo a extremo, verificada contra el código real, de todo lo construido en el repositorio. Está pensado para que alguien sin contexto previo del proyecto pueda entender qué existe, por qué existe, cómo se ejecuta y en qué orden, sin tener que leer el código fuente. No sustituye a los ADR (justificación de cada decisión); los sintetiza y los organiza por flujo de ejecución.

**Trazabilidad de las cifras:** todas las cifras numéricas de este documento proceden, salvo indicación contraria, de una única ejecución completa y consistente del pipeline con datos reales de ClinVar y de `myvariant.info`, registrada en MLflow y con su procedencia exacta en `reports/provenance.json`. Es la misma ejecución que cita la memoria del TFM.

---

## 1. Qué es este proyecto, en una página

### 1.1 El problema

Cada variante genética que se secuencia en un paciente necesita una interpretación clínica: ¿causa enfermedad o es un cambio genético inocuo? ClinVar, la base de datos de referencia mundial (NCBI), acumula millones de variantes clasificadas según evidencia clínica en cinco categorías: **Patogénica**, **Probablemente patogénica**, **Benigna**, **Probablemente benigna**, y **VUS — Variant of Uncertain Significance** (significado incierto). Esta última categoría es, en la práctica, un cuello de botella: representa variantes que no se han podido clasificar con la evidencia disponible hasta ahora, y su volumen crece más rápido que la capacidad de revisión manual experta.

### 1.2 Qué construye el proyecto (y qué evolución ha tenido)

El proyecto nació (julio de 2026) como un **pipeline MLOps de anotación y clasificación de variantes** (anteproyecto original): ingesta → anotación → entrenamiento → serving → monitorización → reentrenamiento, con un clasificador binario de patogenicidad como componente. El 27 de julio de 2026 sufrió un **pivote de enfoque documentado en `docs/adr/ADR_007_pivote_priorizacion_vus_ia.md` (decisión D7)**: el elemento diferencial deja de ser "otro pipeline de anotación" y pasa a ser un **motor de IA de priorización y reanálisis continuo de VUS**. La anotación pasa a ser la etapa que *alimenta* ese núcleo, no el objetivo del proyecto.

El motivo inmediato de este pivote fue un bloqueo real: dbNSFP (la fuente de features prevista originalmente) exige registro académico no disponible, y sin sus columnas el entrenamiento era literalmente imposible (0 columnas con datos, no solo señal débil). Resolver "qué usar en su lugar" abrió la puerta a repensar el proyecto entero a la luz del estado del arte 2025-2026 (Talos, RENOVO, Exomiser/LIRICAL, AlphaMissense/PrimateAI-3D/EVE).

**Lo que el sistema hace hoy, en una frase:** toma cada VUS reservada, la puntúa por probabilidad de patogenicidad y por probabilidad de resolverse pronto, explica esa puntuación en un lenguaje reconocible clínicamente (heurística tipo ACMG/AMP, no una certificación), y la sitúa en una lista ordenada de prioridad para que un revisor humano decida qué mirar primero — todo ello reentrenable de forma continua a medida que ClinVar publica nuevas reclasificaciones.

**Lo que el sistema explícitamente NO hace** (declarado en `README.md` y en cada Model Card): no predice el veredicto de variantes que ya tienen significado clínico resuelto (eso ya está en ClinVar, se consulta); no sustituye la curación clínica experta (criterios ACMG/AMP reales exigen segregación familiar, evidencia funcional y revisión de un comité, datos que este proyecto no tiene); no integra datos de paciente (fenotipo HPO, pedigrí, herencia) — trabaja solo con bases públicas agregadas a nivel de variante poblacional, nunca a nivel de paciente individual.

### 1.3 Las dos preguntas de ML que resuelve, y por qué son distintas

| | Pregunta | Analogía |
|---|---|---|
| **Modelo de patogenicidad** | "¿Es esta variante dañina?" — clasificación binaria clásica, entrenada sobre variantes YA resueltas por ClinVar | Como REVEL o CADD: un score de deletereidad |
| **Modelo de potencial de reclasificación** | "¿Esta VUS concreta se va a resolver pronto?" — entrenado explotando dos releases fechadas de ClinVar como par supervisado | Idea sin precedente exacto en la literatura consultada (ADR 007 §4): la mayoría de proyectos de referencia no tienen dos releases fechadas con reclasificaciones reales conocidas entre ellas |

Ambos modelos son complementarios: el primero dice "cuánto riesgo estimado tiene esta VUS"; el segundo dice "cuánta probabilidad hay de que se resuelva pronto, para decidir si merece la pena reanalizarla ahora". Un tercer objetivo (ranking) reformula el problema de "clasificar" a "ordenar" — más fiel a lo que realmente se necesita en priorización.

### 1.4 Autoría y marco académico

Autora: Paula Rodríguez Sánchez. Director: Sergio Pérez Iglesias. Máster en Análisis de Datos Masivos, UEM, curso 2025-2026. El proyecto se ejecuta en dos fases coordinadas ("Opción C", ADR 003): **Fase I** (local, Docker+Python, coste cero, ya completa y es lo que documenta este informe) y **Fase II** (GCP, BigQuery Sandbox + comparativa local/cloud, todavía no iniciada).

---

## 2. Mapa del repositorio

```
mlops/
├── README.md                          Qué es, qué no es y cómo reproducirlo
├── LICENSE                            Licencia MIT
├── config/config.yaml                 Configuración central versionada; nada hardcodeado en el código
├── docs/
│   ├── INFORME_TECNICO_COMPLETO.md    Este documento
│   ├── INTRODUCCION.md                Explicación no técnica del problema y el enfoque
│   ├── adr/ADR_001 a ADR_008          Ocho decisiones de arquitectura justificadas
│   ├── MODEL_CARD.md                  Ficha del modelo de patogenicidad
│   ├── MODEL_CARD_RECLASSIFICATION.md Ficha del modelo de reclasificación, evaluación retrospectiva
│   ├── MODEL_CARD_RECLASSIFICATION_PROSPECTIVE.md
│   │                                  Validación temporal prospectiva real del modelo de reclasificación
│   ├── datasheet.md                   Ficha del dataset y tabla exhaustiva CLNSIG -> etiqueta
│   ├── EDA_variantes.md               Resumen del análisis exploratorio
│   ├── retraining_flow.md             Ciclo de monitorización y reentrenamiento
│   ├── serving_examples.md            Diez variantes de prueba documentadas del servicio REST
│   ├── vus_priorizadas.md             VUS ordenadas por prioridad de revisión
│   ├── vus_informes_test.md           Informes automáticos por VUS
│   ├── FASE_II_DISENO_GCP.md          Diseño de la extensión a GCP, no ejecutado
│   └── figuras/                       Figuras del análisis exploratorio
├── data/
│   ├── raw/                           RAW, inmutable, versionado con DVC (incluye la release prospectiva)
│   ├── interim/                       SILVER: variantes anotadas, un fichero por release
│   └── processed/                     GOLD: dataset de modelado (train/test, VUS reservadas y excluidas)
├── src/
│   ├── config.py                      Carga centralizada de config/config.yaml
│   ├── ingest/                        download.py, synthetic.py
│   ├── annotate/                      schema.py, annotate.py, multi_source.py
│   ├── features/                      build_dataset.py, preprocess.py, reclassification.py
│   ├── train/                         train.py, train_reclass.py, train_ranking.py
│   ├── evaluate/                      metrics.py, explain.py, acmg_evidence.py, compare_predictors.py,
│   │                                  capture_provenance.py, chr_representativeness.py,
│   │                                  transcript_aggregation_sensitivity.py
│   ├── monitor/                       drift.py, drift_report.py, retrain.py
│   └── serve/                         annotator.py, predictor.py, app.py, examples.py,
│                                      prioritize_vus.py, vus_reports.py, dashboard.py
├── models/                            best_model/, reclassification_model/, ranking_model/
├── reports/                           training/, monitoring/, serving/ y provenance.json
├── notebooks/                         01_eda_variantes.ipynb, único uso de notebooks (exploratorio)
├── tests/                             21 ficheros de test, pytest
├── docker/                            app.Dockerfile, mlflow.Dockerfile
├── compose.yaml                       Tres servicios: mlflow, app y serve
├── Makefile                           Comandos reproducibles (documentación ejecutable)
├── .github/workflows/ci.yml           CI: estilo, auditoría de dependencias y pruebas
└── cloud/                             Diseño de la Fase II (BigQuery, Cloud Run, IaC), no ejecutado
```

Repositorio de código: `https://github.com/paularodriguezsanchez/mlops`. Es el repositorio que referencia la memoria del TFM para la reproducción de resultados; no incluye el documento de la memoria en sí (LaTeX), que se gestiona aparte.

**Convención clave para leer rutas de datos:** los ficheros de `data/interim/` se nombran por **release** (`annotated_2023-12.parquet`, `annotated_2025-06.parquet`); los de `data/processed/` se nombran por **rol de split** (`train.parquet`, `test.parquet`, `vus_train.parquet`, `vus_test.parquet`), no por fecha. `train` siempre corresponde a la release antigua (2023-12) y `test` a la release nueva (2025-06) — es un split **temporal**, no una partición aleatoria.

---

## 3. El flujo real, de principio a fin

### 3.1 Diagrama del pipeline completo

```
                         config/config.yaml (fuente única de verdad: rutas, semilla,
                         releases, umbrales, fuente de anotación) — src/config.py

  ┌──────────┐ ┌───────────┐ ┌────────────┐ ┌───────────┐ ┌─────────┐
  │ INGESTA │ --> │ ANOTACIÓN │ --> │ DATASET │ --> │ENTRENAMIENTO│ --> │ SERVING │
  │ (RAW) │ │ (SILVER) │ │ (GOLD) │ │ │ │ │
  └──────────┘ └───────────┘ └────────────┘ └───────────┘ └─────────┘
       │ │ │ │ │
  ClinVar NCBI myvariant.info binariza target el modelo de patogenicidad patogenicidad API REST
  (2 releases (real) o + separa VUS el modelo de reclasificación reclasificación /predict
   fechadas) generador (no se descartan) el objetivo de ranking ranking Dashboard
                     sintético Informes VUS
                     (fallback, Priorización
                      solo tests)
                                                                    │
                                                                    v
                                                          ┌──────────────────┐
                                                          │ EVALUACIÓN / │
                                                          │ EXPLICABILIDAD │
                                                          │ SHAP → ACMG │
                                                          └──────────────────┘
                                                                    │
                                                                    v
                                                          ┌──────────────────┐
                                                          │ MONITORIZACIÓN │
                                                          │ drift covariables │
                                                          │ + deriva reclasif.│
                                                          │ + stress test │
                                                          └────────┬─────────┘
                                                                    │ alerta
                                                                    v
                                                          ┌──────────────────┐
                                                          │ REENTRENAMIENTO │
                                                          │ gate humano │
                                                          │ (dry-run/--execute)│
                                                          └────────┬─────────┘
                                                                    │
                                                    retroalimenta al bloque de ENTRENAMIENTO
```

Este ciclo cerrado (variante → score → revisión → reclasificación en ClinVar → reentrenamiento) es, según ADR 007 §5.5, la narrativa central del proyecto desde el pivote: no es "se entrena una vez y se sirve", es un bucle continuo.

### 3.2 Orden real de ejecución (comandos)

```bash
make up # Levanta MLflow (UI:5000) + contenedor app (Docker)

# --- Capa de datos (RAW -> SILVER -> GOLD) ---
make data # = ingest + annotate + dataset (ver detalle abajo)
  make ingest # python -m src.ingest.download (RAW)
  make annotate # python -m src.annotate.annotate (SILVER)
  make dataset # python -m src.features.build_dataset (GOLD)

# --- Modelado ---
make train # el modelo de patogenicidad: modelo de patogenicidad (3-4 algoritmos, MLflow)
make train-reclass # el modelo de reclasificación: modelo de reclasificación de VUS
make train-ranking # el objetivo de ranking: LightGBM lambdarank + NDCG@k

# --- Serving ---
make serve # API REST:8000 (/health, /predict) + dashboard (/dashboard)
make examples # Documenta >=10 variantes de prueba [OE4]
make prioritize # la priorización de VUS: ordena VUS reservadas por riesgo estimado
make vus-reports # el generador de informes por VUS: informes por VUS (alimenta el dashboard)

# --- Monitorización y reentrenamiento ---
make monitor # Informe de drift (Evidently + propio) [OE5]
make retrain # Evalúa drift y RECOMIENDA reentrenar (dry-run)
python -m src.monitor.retrain --execute # Aprueba y ejecuta el reentrenamiento real

# --- Validación temporal prospectiva del modelo de reclasificación y comparativas (revisión posterior del proyecto) ---
make ingest-prospective # Descarga la 3ª release real de ClinVar (verdad terreno futura)
make validate-prospective # Aplica el modelo el modelo de reclasificación ya entrenado, sin reentrenar, sobre esa release
python -m src.evaluate.compare_predictors # el modelo de patogenicidad vs. CADD/REVEL/AlphaMissense en solitario
python -m src.evaluate.capture_provenance # Registra commit/hashes/run IDs de la ejecución (reports/provenance.json)

make test # Suite completa (pytest, 95 tests)
make lint # ruff check src tests
```

Cada `make <target>` es literalmente `python -m src.<paquete>.<módulo>` — el Makefile no añade lógica, solo documenta el comando canónico (es "documentación ejecutable", como dice su cabecera).

### 3.3 Por qué este orden es obligatorio y no intercambiable

1. **`ingest` antes que nada**: sin `data/raw/clinvar_*.vcf.gz`, `annotate.py` no tiene nada que leer.
2. **`annotate` antes que `dataset`**: `build_dataset.py` lee `data/interim/annotated_{release}.parquet`, que solo existe tras anotar.
3. **`dataset` antes de cualquier entrenamiento**: los tres scripts de `src/train/` leen `data/processed/{train,test}.parquet` y/o `vus_{train,test}.parquet`.
4. **`train` antes de casi todo el serving**: `predictor.py` carga `models/best_model`; sin él, `/predict`, `examples.py` y `prioritize_vus.py` no tienen modelo que cargar.
5. **`train-reclass` antes de `vus-reports`**: si `models/reclassification_model` no existe, `vus_reports.py` degrada mostrando "no disponible" en vez de fallar — pero para un informe completo hace falta entrenarlo primero.
6. **`vus-reports` antes de que el dashboard muestre algo**: el dashboard **no calcula nada por sí mismo**, solo lee el JSON que genera `vus_reports.py`. Si no existe, muestra instrucciones de cómo generarlo.
7. **`monitor` antes de `retrain`**: `retrain.py` reutiliza internamente todo el informe de `drift_report.py` para decidir si recomienda reentrenar.

### 3.4 El modelo de ranking: integrado como criterio de orden real (actualización 2026-08-07)

Hallazgo detectado en la primera versión de este informe (2026-08-05): `models/ranking_model` (LightGBM `lambdarank`) se entrenaba y evaluaba, pero ningún módulo de `src/serve/` lo cargaba — el orden que veían realmente `prioritize_vus.py`, `vus_reports.py` y el dashboard se basaba exclusivamente en la probabilidad de patogenicidad del modelo el modelo de patogenicidad. Esto se corrigió el 2026-08-07:

* `src/train/train_ranking.py` persiste ahora, junto al booster (`lambdarank.txt`), el **preprocesador ya ajustado** (`models/ranking_model/preprocessor.joblib`, vía `joblib.dump`) — necesario para transformar features nuevas exactamente igual que en entrenamiento, sin reajustarlo en cada carga.
* `src/serve/prioritize_vus.py` añade `load_ranking_model` (carga booster + preprocesador; degrada a `(None, None)` si no existen, mismo patrón de degradación explícita que el resto del proyecto) y `rank_vus` ordena por `ranking_score` cuando el modelo está disponible, con **fallback explícito** a `probabilidad_patogenica` si `make train-ranking` no se ha ejecutado todavía.
* `src/serve/vus_reports.py` reutiliza el mismo `load_ranking_model`, para que la priorización de VUS y el generador de informes por VUS nunca usen criterios distintos; el JSON de cada informe incluye `ranking_score` cuando existe.
* `src/serve/dashboard.py` declara en la cabecera, de forma dinámica, si el orden mostrado viene del objetivo de ranking o del modelo de patogenicidad.

Verificado con la regeneración real del pipeline (2026-08-07): `make prioritize` imprime *"orden por score de ranking "* y `docs/vus_priorizadas.md` lo declara en su propio texto — el objetivo de ranking ya no es una pieza de validación metodológica aislada, es el criterio de orden efectivo del núcleo de priorización (ver también ADR 007, actualización 2026-08-07).

---

## 4. Capa 1 — Configuración central (`src/config.py`, `config/config.yaml`)

Punto único de acceso a toda la configuración del proyecto. La regla de `docs/retraining_flow.md`, es categórica: *"nada hardcodeado (rutas, semillas, umbrales)"*.

**Funciones clave** (`src/config.py`):
- `load_config(path=None)`: carga el YAML con `yaml.safe_load` (nunca `yaml.load` inseguro), cacheada con `lru_cache`. Detalle de diseño no obvio y corregido tras un bug real: `path` se resuelve **en el momento de la llamada**, no en la firma de la función — de lo contrario, un test que monkeypatchea `CONFIG_PATH` no tendría ningún efecto (bug detectado validando el modelo de reclasificación, donde un test "aislado" seguía leyendo la config real de 6000 variantes en vez de las 300 del test).
- `get_seed` → semilla global `42`, usada en todo paso estocástico del proyecto.
- `raw_dir`, `interim_dir`, `processed_dir` → rutas de las tres capas.
- `chromosomes_subset` → `["1", "2", "3"]` en Fase I (acota volumen, riesgo R2 del plan).
- `max_variants_per_release` (8000) y `max_new_variants_per_release` (4000): acotan el volumen y, con `annotation_source: multi_source`, el **coste de red** (cada variante cuesta una consulta a myvariant.info).
- `annotation_source` → `"multi_source"` en producción, `"synthetic"` reservado exclusivamente a tests/CI.

**Bloques relevantes de `config/config.yaml`:**

```yaml
project: { random_seed: 42, assembly: GRCh38 }
data:
  clinvar_train_release: "2023-12" # drift temporal: train=antigua
  clinvar_test_release: "2025-06" # test=nueva
  chromosomes_subset: ["1", "2", "3"]
  max_variants_per_release: 8000
  max_new_variants_per_release: 4000
  annotation_source: multi_source # myvariant.info real; "synthetic" solo en tests
target:
  positive_labels: [Pathogenic, Likely_pathogenic, Pathogenic/Likely_pathogenic]
  negative_labels: [Benign, Likely_benign, Benign/Likely_benign]
  vus_labels: [Uncertain_significance] # VUS estricta; el resto se excluye automáticamente
mlflow:
  experiment_name: variant_pathogenicity
  reclass_experiment_name: vus_reclassification_potential
  ranking_experiment_name: vus_ranking
monitor: { drift_threshold: 0.15 }
```

---

## 5. Capa 2 — Ingesta (`src/ingest/`): RAW, inmutable

### 5.1 `download.py` — descarga real de ClinVar

Descarga las dos releases fechadas de ClinVar (VCF GRCh38, desde `sources.clinvar.base_url` de config) y escribe `data/raw/MANIFEST.json` con SHA-256 y procedencia de cada fichero, para trazabilidad extremo a extremo (T2).

**Decisión de política crítica (ADR 005, revisado 2026-07-30):** el comportamiento por defecto **exige red real y falla con `RuntimeError` explícito** si no hay conexión, en vez de sustituir en silencio por datos inventados. Solo `--offline`, pasado explícitamente, activa el generador determinista. La razón, citada literalmente del propio código: *"Este proyecto NO sustituye datos reales por sintéticos de forma automática: son variantes genéticas asociadas a enfermedades reales, y un resultado basado en datos inventados no es válido como resultado del TFM."* Esta política se revisó porque la versión original permitía un fallback automático y silencioso, con el riesgo real de que un resultado sintético se citara por descuido como si fuera clínico.

También valida activamente que la URL de descarga use esquema `http`/`https` (nunca `file://`), como mitigación explícita frente a un hallazgo de seguridad de linter (`S310`).

```bash
python -m src.ingest.download # exige red real; falla si no hay conexión
python -m src.ingest.download --offline # opt-in EXPLÍCITO al generador determinista
python -m src.ingest.download --force # regenera aunque ya exista
```

### 5.2 `synthetic.py` — generador offline determinista

Produce, sin tocar la red, ficheros con el **mismo esquema exacto** que las fuentes reales (VCF de ClinVar + tabla dbNSFP), con semilla fija (42). Reservado **exclusivamente a tests/CI** (cada test que lo necesita lo fija explícitamente; no es alcanzable desde ningún `make` oficial salvo `--offline` a mano).

Tres propiedades imitadas deliberadamente del dato real:
1. Las features **correlacionan** con la patogenicidad a través de una señal latente ruidosa (`feat_signal = clip(latent + N(0, 0.38), 0, 1)`) — el ruido impone un error de Bayes realista (ningún clasificador puede llegar a AUC=1, igual que con REVEL/CADD reales).
2. La release "nueva" añade variantes y reclasifica una fracción de VUS de la "antigua" — concept drift real, no trivial. La selección de qué VUS se reclasifica **no es uniforme**: está ponderada por `|latent - 0.5|` (las VUS con más evidencia previa, aunque insuficiente para veredicto, se resuelven antes) — decisión correctiva tomada tras descubrir que una selección uniforme dejaba al modelo el modelo de reclasificación sin ninguna señal aprendible por construcción del propio generador.
3. Todo depende de una única semilla: reproducibilidad total (hash idéntico al regenerar).

### 5.3 Salida de la capa RAW

`data/raw/clinvar_2023-12.vcf.gz`, `clinvar_2025-06.vcf.gz`, `MANIFEST.json` (y, solo en modo offline, `dbnsfp_subset.tsv.gz`). `data/raw/` es **inmutable**: nunca se edita a mano ni por código; se versiona con DVC (`data/raw.dvc`, sin remoto configurado todavía — pendiente para cuando entre dato real a escala).

---

## 6. Capa 3 — Anotación (`src/annotate/`): RAW → SILVER

### 6.1 `schema.py` — el contrato de datos

Define y valida activamente el esquema de cualquier tabla de variantes anotadas: 15 columnas requeridas (`chrom, pos, ref, alt` como clave; `gene, consequence, clnsig` categóricas; 8 numéricas con rango válido declarado). Ejemplo de por qué esto importa en la práctica: el límite superior de `cadd_phred` estaba fijado a 60 (calibrado sobre el techo arbitrario del generador sintético) y tuvo que ampliarse a 99 al validar contra ClinVar real (variantes nonsense reales llegan a CADD phred ≈73). `validate_annotated(df, strict=True)` lanza `SchemaError` con todos los problemas detectados si algo no cumple el contrato — no es decorativo, se invoca con `strict=True` en el flujo real, no solo en tests.

### 6.2 `annotate.py` — pipeline clásico (dbNSFP o delegación a multi-fuente)

Parsea el VCF de ClinVar por regex sobre el campo `INFO` (extrae `CLNSIG`, `GENEINFO`, `MC`), filtra explícitamente a SNVs (descarta ALT multialélico/estructural — corrección de un hallazgo real al probar contra ClinVar real, que sí incluye este tipo de entradas), cruza con las features por clave `(chrom,pos,ref,alt)`, acota por cromosomas y volumen, y valida contra el contrato de `schema.py`.

**Pieza más delicada del módulo — `_sample_release` con "continuidad de cohorte":** ClinVar real es acumulativo (casi toda variante de una release sigue existiendo en la siguiente), y tiene cientos de miles de variantes en chr 1-3. Muestrear cada release de forma independiente casi no deja claves compartidas por azar, lo que rompería por completo la medición de drift real. La solución fija una cohorte tras procesar la primera release y, en la segunda, conserva esa cohorte (hasta el cupo `max_variants_per_release`) más un cupo explícito de variantes genuinamente nuevas (`max_new_variants_per_release`). Sin este segundo cupo se detectó un bug real (2026-08-02): la cohorte retenida agotaba ella sola el cupo total, dejando un holdout "no visto" casi vacío (2,1 %, 75 de 3532 variantes) — justo el subconjunto que se usa para evaluar honestamente el modelo principal.

También hay un detalle de rendimiento documentado: el acotado por cromosomas y volumen se aplica **antes** de anotar, no después — con `multi_source` (consultas de red reales), acotar solo la salida dejaba el proceso colgado horas, disparando consultas para variantes que se iban a descartar igualmente.

### 6.3 `multi_source.py` — anotación real sin dbNSFP (ADR 007 §1 / B5)

El módulo que resuelve el bloqueo que disparó el pivote. Consulta **myvariant.info**, un agregador público sin registro, en lotes de hasta 1000 variantes por POST, obteniendo en una sola fuente lo mismo que aportaba dbNSFP más AlphaMissense (nuevo). Tres bugs reales encontrados y corregidos al validar contra ClinVar real, documentados explícitamente en el código:

1. **Ensamblaje**: myvariant.info usa hg19 por defecto; con GRCh38 hace falta `assembly=hg38` explícito — sin él, todas las consultas devolvían "notfound" pese a existir el dato (0/60 variantes con cualquier feature hasta corregirlo).
2. **Namespace de campos**: bajo `assembly=hg38`, CADD/SIFT/PolyPhen no viven en `cadd.*` (solo hg19) sino en `dbnsfp.*`.
3. **Proyección de campo con "+"**: pedir `dbnsfp.gerp++.rs` por nombre devuelve vacío pese a existir el dato (myvariant.info no proyecta bien nombres con "+"); solución: pedir el objeto `dbnsfp` completo y extraer el campo del lado del cliente.
4. **Listas por transcrito**: REVEL, SIFT, PolyPhen2, CADD, GERP y phyloP pueden venir como lista (una entrada por transcrito), no solo AlphaMissense; se promedian todos con `_mean_or_none`.

**SpliceAI** se intentó vía el servicio de consulta de Broad Institute; no accesible desde el entorno de desarrollo (timeout/DNS en dos hosts probados). Se documenta como limitación real, con degradación a `NaN`, avisando una sola vez por proceso (no en cada variante).

**Cobertura real validada** (muestra ClinVar real, chr 1-3, missense, n=100, semilla 42): CADD 100 %, phyloP 100 %, AlphaMissense 98 %, REVEL 95 %, SIFT 96 %, PolyPhen 90 %, GERP 99 %, gnomAD AF 72 %.

### 6.4 `synthetic` vs `multi_source` — cuándo se usa cada uno

| | `synthetic` | `multi_source` |
|---|---|---|
| Origen | Generador determinista local | Consultas HTTP reales a myvariant.info |
| Requiere red | No | Sí |
| Uso permitido | **Exclusivamente tests/CI** | Fuente única de cualquier resultado citable (memoria, Model Cards, informes) |
| Valor real en `config.yaml` | — | `multi_source` |

### 6.5 Salida de la capa SILVER

`data/interim/annotated_2023-12.parquet`, `annotated_2025-06.parquet`. 15 columnas del contrato de `schema.py`.

---

## 7. Capa 4 — Construcción del dataset (`src/features/build_dataset.py`): SILVER → GOLD

Binariza el target de significancia clínica según `config.yaml` (`target.positive_labels`/`negative_labels`) y separa temporalmente train/test: **train = release antigua (2023-12), test = release nueva (2025-06)** — no es una partición aleatoria de una única muestra.

**La decisión más importante de este módulo: las VUS no se descartan.** Se persisten en un fichero paralelo por split (`vus_train.parquet`, `vus_test.parquet`), sin columna `label`, como "conjunto de inferencia realista". Es la materialización directa de **ADR 006** (conservación de VUS para reanálisis futuro): incluir VUS en el entrenamiento inyectaría ruido de etiqueta (por definición, VUS no tiene ground-truth fiable), pero en la práctica clínica las VUS no se tiran, se reevalúan periódicamente. El ADR cita explícitamente **TALOS** (Centre for Population Genomics/Broad/Microsoft, *Nature Medicine* 2026) como precedente de esta idea a escala clínica de paciente — aclarando que TALOS no es aplicable literalmente aquí, porque trabaja a nivel de paciente (fenotipo, pedigrí) y este proyecto solo a nivel de variante poblacional agregada.

**Taxonomía de etiquetas corregida (revisión posterior del proyecto 2026-08-18):** "VUS" se define ahora de forma **estricta** como `clnsig == "Uncertain_significance"` (`src/features/reclassification.py::is_vus`), no como "cualquier variante sin veredicto positivo/negativo". Antes, `build_dataset.py` mezclaba en la población de VUS también clasificaciones conflictivas y otras categorías ambiguas de ClinVar, mientras que `drift_report.py` usaba por su cuenta, de forma independiente, el literal estricto — una divergencia no documentada que era la causa exacta de una inconsistencia numérica detectada por una revisión externa (67 VUS reclasificadas citadas para el modelo de reclasificación frente a 54 citadas para el monitor, para la misma transición). Con la definición unificada, ambas cifras coinciden: **55**. Las variantes que no son positivas, negativas ni VUS estricta (conflictivas, `not_provided`, `other`, etc.) se persisten aparte en `excluded_{split}.parquet`, por trazabilidad, no se descartan en silencio. Tabla exhaustiva `CLNSIG` crudo → *bucket*, generada automáticamente por este módulo en `reports/training/clinvar_label_taxonomy.csv` y reproducida en `docs/datasheet.md`.

**Convención de nombres importante:** las salidas se llaman por **rol de split**, no por fecha de release — hay que recordar que `train.parquet` ↔ release 2023-12 y `test.parquet` ↔ release 2025-06.

---

## 8. Capa 5 — Preprocesamiento y regla de reclasificación (`src/features/`)

### 8.1 `preprocess.py` — puente hacia el modelado

No genera datos: construye el `ColumnTransformer` de scikit-learn compartido por los tres scripts de entrenamiento, garantizando que entrenamiento y serving usen exactamente el mismo preprocesamiento (va embebido dentro del `Pipeline` que se registra en MLflow).

**Los 8 features numéricos de anotación** (mencionados en el estado del proyecto) + 1 categórica:

| Grupo | Columnas | Tratamiento |
|---|---|---|
| `SCORE_FEATURES` (4) | `sift_score, polyphen_score, revel_score, alphamissense_score` | Imputación por mediana **+ indicador de ausencia** (NaN fuera de missense, como en dbNSFP real; la propia ausencia es informativa) → escalado |
| `DENSE_NUMERIC` (3) | `cadd_phred, gerp_rs, phylop` | Mismo tratamiento que arriba — **hallazgo real**: con ClinVar real su tasa de nulos es tan alta como la del grupo anterior (~75-76 %), a diferencia del supuesto original ("numéricas casi siempre presentes"), que solo era cierto con el generador sintético |
| `LOG_FEATURES` (1) | `gnomad_af` | `log10` recortado (la frecuencia poblacional es muy asimétrica) → imputación por mediana **+ indicador de ausencia** (añadido en la revisión posterior del proyecto 2026-08-18: la ausencia de gnomAD no siempre es rareza biológica, también puede ser un fallo de cobertura de la fuente; conservar el indicador aparte, en vez de disolverlo en la mediana, mejoró de forma real y no trivial el PR-AUC del modelo de patogenicidad y el NDCG@10 del objetivo de ranking) → escalado |
| `CATEGORICAL_FEATURES` (1) | `consequence` | One-hot con `handle_unknown="ignore"` (no rompe ante categorías nuevas en serving) |

`gene` se **excluye deliberadamente** por alta cardinalidad y riesgo de fuga/sobreajuste.

### 8.2 `reclassification.py` — regla compartida "¿está resuelta esta variante?"

Módulo diminuto pero deliberado: define las funciones de verdad únicas `is_resolved(clnsig)` e `is_vus(clnsig)` para que el entrenamiento del modelo de reclasificación, `build_dataset.py` y el monitor de drift no dupliquen, con riesgo de divergencia, la misma lógica de "¿esta variante ya tiene veredicto?" / "¿es esto una VUS?". **Bug real corregido (revisión posterior del proyecto 2026-08-18):** `RESOLVED_LABELS` no incluía las formas compuestas `Pathogenic/Likely_pathogenic` / `Benign/Likely_benign`, pese a que `config.yaml` sí las trata como positivas/negativas para el modelo de patogenicidad — afectaba al recuento de reclasificaciones del modelo de reclasificación y del monitor. `RESOLVED_LABELS` ahora incluye las seis formas; `VUS_LABEL = "Uncertain_significance"` es la única fuente de verdad de qué es VUS, usada tanto por `build_dataset.py` como por `drift_report.py`.

---

## 9. Capa 6 — Entrenamiento (`src/train/`): los tres modelos del núcleo

Los tres comparten infraestructura (mismo `build_preprocessor`, mismo MLflow, mismo criterio de "holdout no visto" cuando aplica) pero resuelven tareas de ML distintas.

### 9.1 `train.py` — Modelo de patogenicidad

**Tarea:** clasificación binaria supervisada, patogénica (1) vs. benigna (0), sobre las variantes ya resueltas de ClinVar.

**Algoritmos comparados (4):** `logistic_regression`, `random_forest` (300 árboles), `gradient_boosting`, `hist_gradient_boosting` (histogram-based, se añadió como candidato adicional del estado del arte más allá de los 3 del anteproyecto original).

**Criterio de selección — dos correcciones metodológicas independientes, en dos revisiones distintas:**

1. **(corrección de la fuga entre entrenamiento y evaluación)** el test temporal (release 2025-06) es, por construcción, un superconjunto parcial del train (release 2023-12) — ClinVar es acumulativo. Evaluar sobre el test completo es optimista por memorización (en una versión temprana, esto infló el PR-AUC a 0,988). La corrección: `unseen_mask` calcula qué filas del test tienen una clave `(chrom,pos,ref,alt)` que NO aparece en train; ese **holdout no visto** (n=1764) es el conjunto de evaluación honesto.
2. **(revisión posterior del proyecto 2026-08-18)** hasta esa revisión, el algoritmo "ganador" se seguía eligiendo por su propio PR-AUC en ese mismo holdout, el que después se citaba como resultado final: dejaba de ser un test genuinamente independiente (*selection bias*). La corrección: la selección usa ahora **únicamente** la media de PR-AUC en validación cruzada de 5 particiones estratificadas **sobre el conjunto de entrenamiento**, sin tocar el holdout; el holdout se evalúa una única vez, sobre el algoritmo ya elegido.

**Resultado real, extremo a extremo con datos reales** (`docs/MODEL_CARD.md`, `reports/training/model_comparison.csv`):

| Algoritmo | PR-AUC CV (train, 5-fold) | PR-AUC holdout [IC 95 %] | ROC-AUC holdout | F1 holdout | PR-AUC full |
|---|---|---|---|---|---|
| **logistic_regression (ganador por CV)** | **0,9787±0,0090** | 0,9700 [0,9511, 0,9849] | 0,9945 | 0,9359 | 0,9772 |
| random_forest | 0,9750±0,0084 | 0,9657 [0,9388, 0,9858] | 0,9925 | 0,9216 | 0,9827 |
| gradient_boosting | 0,9760±0,0055 | 0,9721 [0,9556, 0,9864] | 0,9947 | 0,9212 | 0,9831 |
| hist_gradient_boosting | 0,9759±0,0090 | 0,9581 [0,9310, 0,9803] | 0,9875 | 0,9289 | 0,9809 |

Nótese que `gradient_boosting` tiene un holdout ligeramente mejor que el ganador (`logistic_regression`), pero pierde la selección porque su PR-AUC de validación cruzada en train es marginalmente inferior (0,9760 frente a 0,9787, dentro de una desviación estándar de ambos): es precisamente el caso que ilustra por qué seleccionar por el holdout y evaluar en él a la vez es un sesgo real, no solo teórico — con el criterio correcto, el "ganador" no siempre coincide con el que más alto puntúa en el conjunto que un lector desprevenido miraría primero.

**Comparativa frente a los predictores in silico usados en solitario** (`src/evaluate/compare_predictors.py`, nuevo, revisión 2026-08-18): sobre las mismas variantes del holdout, con IC bootstrap de la diferencia (pareado, 1000 remuestreos). El ensemble supera de forma estadísticamente distinguible a CADD solo (0,935 frente a 0,986 sobre 379 variantes con CADD disponible, diferencia [0,026, 0,079]), a REVEL solo (0,918 frente a 0,970 sobre 252 missense, diferencia [0,024, 0,087]) y a AlphaMissense solo (0,910 frente a 0,968 sobre 254 missense, diferencia [0,022, 0,106]) — ver `reports/training/compare_predictors.csv`.

**Registro en MLflow:** experimento `variant_pathogenicity`, un run por algoritmo con params/métricas/artefactos (matriz de confusión + modelo cloudpickle); el mejor se registra en el Model Registry como `variant_pathogenicity_clf`, stage **Staging** (nunca directo a Production — segundo gate humano).

**Importancia de features** (dos análisis complementarios, ambos generados por este script): por **permutación** (`cadd_phred` y `gnomad_af` dominan) y por **SHAP** (mismo orden relativo aproximado: `cadd_phred`, `gnomad_af`, `consequence`; cifras exactas en `reports/training/{feature_importance,shap_importance}.csv` de la ejecución vigente).

**Salidas:** `models/best_model/` (consumido por `predictor.py`, `prioritize_vus.py`), `docs/MODEL_CARD.md`, `reports/training/{model_comparison.csv, feature_importance.csv, shap_importance.csv, cm_*.png}`.

### 9.2 `train_reclass.py` — Modelo de potencial de reclasificación de VUS

**Tarea:** dada una VUS de la release antigua, predecir si se resolverá (a Patogénica o Benigna) en la release nueva. Es, según ADR 007, la aportación más genuina del proyecto: explota algo que la mayoría de proyectos de referencia no explotan exactamente así (dos releases fechadas con reclasificaciones reales conocidas, tratadas como par de entrenamiento supervisado directo).

**Datos:** población = `vus_train.parquet` (VUS de la release antigua); etiqueta = 1 si esa misma variante aparece con `clnsig` resuelto en `annotated_{release_test}.parquet` (vía `is_resolved` de `reclassification.py`), 0 en caso contrario.

**Evaluación retrospectiva — limitación metodológica explícita y honesta:** el *holdout* aleatorio estratificado usado aquí mide señal discriminativa **dentro** del mismo intervalo histórico 2023-2025, no capacidad de predicción prospectiva genuina — distinción sobre la que alertó explícitamente una revisión posterior del proyecto (2026-08-18). Ver §9.2 bis para la validación prospectiva real que se construyó en respuesta.

**Resultado real, evaluación retrospectiva** (`docs/MODEL_CARD_RECLASSIFICATION.md`): población de **4056** VUS de 2023-12 (VUS estricta, tras la corrección de taxonomía de §7), **55** (1,4 %) reclasificadas en 2025-06 (antes: 67, con la definición de VUS sin corregir que mezclaba clasificaciones conflictivas).

| Algoritmo | PR-AUC | ROC-AUC | F1 |
|---|---|---|---|
| **logistic_regression (ganador)** | **0,1093** | 0,7786 | 0,0000 |
| random_forest | 0,0668 | — | — |
| gradient_boosting | 0,0746 | — | — |
| hist_gradient_boosting | 0,0717 | — | — |

**Selección del ganador:** por PR-AUC (mismo criterio que el modelo de patogenicidad), no por ROC-AUC. `logistic_regression` tiene F1=0,0000 en el holdout al umbral de decisión por defecto (0,5): no produce ni una sola predicción positiva correcta a ese umbral. Es precisamente el motivo por el que la selección se hace por PR-AUC (no depende de un umbral fijo) y por el que existe además un umbral de fiabilidad independiente por ROC-AUC antes de servir cualquier probabilidad sin aviso.

**Rigor estadístico adicional (revisión posterior del proyecto 2026-08-18):** el holdout tiene solo 14 positivos (n=1014); el IC 95 % bootstrap del PR-AUC es, en consecuencia, ancho: [0,033, 0,273], reportado explícitamente en vez de omitido. Más informativas para el uso real (ordenar una cola de revisión) son las métricas de cola: **precision@10=0,20, lift@10≈14,5** (catorce veces más aciertos en las diez primeras posiciones que un orden aleatorio), degradando hacia precision@100=0,07 (lift≈5,1). Calibración: Brier score 0,013, con la salvedad explícita de que 14 positivos no permiten validar la calibración con confianza. Ver `src/evaluate/metrics.py::{precision_recall_lift_at_k, calibration_report}` y `reports/training/{reclass_pr_curve.csv, reclass_roc_curve.csv}`.

**Por qué el PR-AUC es bajo aunque el modelo tenga señal real:** la clase positiva es minoritaria (1,4 %); PR-AUC es sensible a la prevalencia (el valor esperado de un clasificador aleatorio coincide con la prevalencia), así que 0,1093 supone varias veces ese nivel de referencia, pese a parecer un valor absoluto bajo. Por eso el proyecto define además un **umbral de fiabilidad explícito e independiente de PR-AUC**: `RELIABLE_ROC_AUC_THRESHOLD = 0.6`. Si el ROC-AUC del mejor modelo no lo supera, se marca `reliable=False` en `models/reclassification_model/metrics.json`, y esa señal se propaga como aviso visible en el dashboard y en los informes por VUS — hallazgo de la revisión interna del proyecto, resuelto exactamente así. Con la cifra vigente (ROC-AUC 0,7786), el modelo se marca `reliable=true`.

**Ablación de fuga temporal — ya no tranquilizadora:** un modelo restringido a las dos características ancladas a la fecha real de cada release (`consequence`, `review_stars`) alcanza ROC-AUC 0,7476, **ligeramente por debajo** del modelo completo (0,7786, diferencia +0,031). Antes de la corrección de taxonomía de §7, la ablación superaba claramente al modelo completo (0,8744 frente a 0,7300), lo que se leía como tranquilizador respecto al riesgo de fuga; con la población corregida, esa lectura ya no se sostiene, y no puede descartarse que parte del rendimiento del modelo completo dependa de características con riesgo de fuga temporal (CADD/REVEL/AlphaMissense/gnomAD, recalculadas "hoy" sin anclaje a la fecha de cada release — ver ADR 008).

### 9.2 bis. Validación temporal PROSPECTIVA real del modelo de reclasificación (nueva, revisión posterior del proyecto 2026-08-18)

Responde directamente a la pregunta que ni el holdout retrospectivo ni la ablación podían responder: ¿predice este modelo, ya entrenado, una reclasificación genuinamente futura? Se descargó una **tercera release real de ClinVar** (2026-08, `clinvar_20260808.vcf.gz`, ~14 meses después de la release de test), publicada después de fijar el par de entrenamiento y nunca usada para ajustar nada. `src/train/train_reclass.py::run_prospective`: toma las VUS que en la release de test (2025-06) seguían sin resolver (4001, el subconjunto "abierto" real del modelo ya entrenado), aplica el modelo persistido **sin reentrenar** y comprueba cuáles se resolvieron en la release de 2026-08 (parseada solo para verdad terreno CLNSIG, sin reanotar contra myvariant.info — las features ya están fijadas en la release de entrenamiento).

**Resultado** (`docs/MODEL_CARD_RECLASSIFICATION_PROSPECTIVE.md`, `models/reclassification_model/prospective_metrics.json`): de 4001 VUS abiertas, 3986 se localizaron en la release de 2026-08 y solo **7 (0,17 %)** se habían resuelto. Modelo completo: **ROC-AUC 0,604**, PR-AUC 0,009. Ablación segura: ROC-AUC 0,447 (por debajo del azar), PR-AUC 0,004.

**Lectura honesta, no forzada:** con solo 7 positivos, esta estimación es muy ruidosa y no permite confirmar ni refutar con seguridad la capacidad prospectiva del modelo. Lo que sí es cierto es que el resultado, tal cual sale, es sustancialmente más débil que el retrospectivo (ROC-AUC 0,779): la señal que el modelo captura dentro del intervalo histórico 2023-2025 no se traslada, con la evidencia actual, a una predicción confiable de una release genuinamente futura. El modelo de reclasificación se presenta en la memoria como prueba de concepto retrospectiva, con esta primera validación prospectiva real como limitación cuantificada, no como una laguna metodológica sin abordar.

Reproducible con `make ingest-prospective && make validate-prospective` (requiere haber entrenado el modelo de reclasificación antes con `make train-reclass`).

### 9.3 `train_ranking.py` — Objetivo de ranking con LightGBM

**Tarea:** no clasificación sino ***learning-to-rank***: el entregable real del proyecto es un orden de prioridad, no una probabilidad aislada. Entrena `LGBMRanker(objective="lambdarank")` (LambdaMART) directamente sobre ese objetivo, evaluado con **NDCG@k** (Normalized Discounted Cumulative Gain), la métrica estándar de ranking porque pondera la posición: penaliza más un error cerca del top de la lista (donde el revisor humano realmente mira).

**Simplificación documentada explícitamente:** LambdaMART agrupa ítems por "query"; aquí no hay una agrupación natural (no hay "una query por paciente"), así que se trata como un único grupo global. Es una simplificación razonable para un ranking de prioridad, no disfrazada de otra cosa.

**Hallazgo relevante:** LightGBM con `objective="lambdarank"` **no es reproducible entre ejecuciones pese a `random_state` fijo**, salvo que se fijen explícitamente `deterministic=True, force_row_wise=True, num_threads=1` (la construcción de histogramas multi-hilo introduce no-determinismo en la reducción en coma flotante). Confirmado empíricamente: el valor versionado en git era NDCG@10=1,0 ("ranking perfecto", sospechoso en sí mismo), y al reejecutar dio 0,7015 de forma estable en dos ejecuciones repetidas. Corregido fijando esos tres flags; verificado con un test de regresión (`test_run_es_reproducible_entre_ejecuciones`) que exige NDCG idéntico al bit entre dos entrenamientos.

**Resultado real reproducible** (`reports/training/ranking_metrics.csv`, holdout no visto n=1764): NDCG@10=**0,890**, NDCG@50=0,944, NDCG@100=0,966, PR-AUC de referencia=0,798 (mejora respecto a la cifra de 2026-08-07 por el indicador de ausencia de gnomAD añadido en la revisión posterior del proyecto de 2026-08-18, §8.1).

**Baselines añadidos (revisión posterior del proyecto 2026-08-18):** sin una referencia, un NDCG@10 alto no dice si el modelo aporta algo — puede que el propio desbalance de clases favorezca casi cualquier orden. `_baseline_ndcg` calcula el NDCG del **orden de llegada** del holdout sin reordenar (NDCG@10=0,000) y la media de 50 **órdenes aleatorios** independientes (NDCG@10=0,140±0,119). El margen del modelo sobre ambos es amplio.

**Limitación de proxy, declarada explícitamente:** el objetivo de entrenamiento (*relevance label*) es el mismo binario de patogenicidad que el modelo de patogenicidad, y tanto el entrenamiento como esta evaluación usan variantes **ya resueltas** — la misma población que el modelo de patogenicidad, no VUS. En producción, sin embargo, este modelo ordena **VUS** (`prioritize_vus.py`, `vus_reports.py`), una población distinta sobre la que no existe verdad terreno de ranking. El NDCG@k reportado mide la calidad del orden sobre variantes resueltas; no se ha medido, ni puede medirse hoy, directamente sobre la cola de VUS que el sistema prioriza en la práctica.

**Registro:** experimento `vus_ranking`, un único run. El booster se guarda como texto nativo de LightGBM (`ranker.booster_.save_model(...)`), subido como artefacto genérico — no usa el flavor `mlflow.lightgbm` porque `LGBMRanker` no es de confianza por defecto para el serializador `skops`. **No se registra en el Model Registry** (a diferencia del modelo de patogenicidad y el modelo de reclasificación).

**Salida:** `models/ranking_model/lambdarank.txt` + `models/ranking_model/preprocessor.joblib` (el preprocesador ajustado, persistido desde el 2026-08-07 para que serving pueda transformar features nuevas sin reajustarlo), `reports/training/ranking_metrics.csv`. Sin Model Card propia. Desde el 2026-08-07 (ver §3.4), **es el criterio de orden real** que usan `prioritize_vus.py` y `vus_reports.py` vía `load_ranking_model`, con fallback al modelo de patogenicidad si no está entrenado.

### 9.4 Resumen comparativo de los tres modelos

| | el modelo de patogenicidad `train.py` | el modelo de reclasificación `train_reclass.py` | el objetivo de ranking `train_ranking.py` |
|---|---|---|---|
| Tarea | Clasificación patogenicidad | Clasificación potencial de reclasificación | Ranking (lambdarank) |
| Evaluación | Holdout temporal "no visto" | Holdout aleatorio estratificado (retrospectivo) **+ validación prospectiva real (§9.2 bis)** | Holdout temporal "no visto" |
| Selección de algoritmo | PR-AUC medio en CV (train únicamente), evaluación final única en holdout | Igual que el modelo de patogenicidad | NDCG@10/50/100 |
| Model Registry | Sí, `variant_pathogenicity_clf` @Staging | Sí, `vus_reclassification_clf` @Staging | No |
| Consumidor productivo | `predictor.py`, `prioritize_vus.py`, `vus_reports.py` | `vus_reports.py` (con aviso de fiabilidad) | `prioritize_vus.py` y `vus_reports.py` — decide el orden real desde 2026-08-07, con fallback al modelo de patogenicidad |
| Comando | `make train` | `make train-reclass` | `make train-ranking` |

---

## 10. Capa 7 — Evaluación y explicabilidad (`src/evaluate/`)

### 10.1 `metrics.py` — métricas compartidas

`compute_metrics`: PR-AUC (`average_precision_score`), ROC-AUC, F1, precision, recall — deliberadamente **no accuracy**, porque son robustas al desbalanceo entre patogénicas/benignas. `bootstrap_pr_auc_ci`: intervalo de confianza al 95 % por 1000 remuestreos, para no confundir "ganó esta comparación puntual" con "es genuinamente mejor". `save_confusion_matrix`: figura PNG por algoritmo, con `labels` configurable (el modelo de reclasificación usa "Reclasificada", "No reclasificada", no "Patogénica", "Benigna" — corrección de la revisión posterior del proyecto 2026-08-18, la tarea del modelo de reclasificación no es la misma que la del modelo de patogenicidad). NDCG@k vive aparte, en `train_ranking.py` (usa `sklearn.metrics.ndcg_score`), porque MLflow no admite `@` en nombres de métrica (se registra como `ndcg_at_k`).

**Añadido en la revisión posterior del proyecto 2026-08-18:** `precision_recall_lift_at_k` (métricas de cola de revisión), `calibration_report` (Brier score + tabla de calibración por bins, ahora usado también por el modelo de patogenicidad, no solo el modelo de reclasificación), `export_pr_curve`/`export_roc_curve` (curvas completas a CSV), `bootstrap_pr_auc_diff_ci` (IC bootstrap pareado de la diferencia de PR-AUC entre dos puntuaciones sobre el mismo conjunto, usado por `compare_predictors.py`) y `bootstrap_roc_auc_ci` (mismo procedimiento que `bootstrap_pr_auc_ci` pero para ROC-AUC, cerrando el un hallazgo de esa revisión, #9 de la revisión posterior del proyecto: el ROC-AUC del modelo de reclasificación se citaba como estimador puntual sin incertidumbre).

### 10.1 bis `chr_representativeness.py` y `transcript_aggregation_sensitivity.py` — dos análisis de sensibilidad (revisión posterior del proyecto, hallazgos #16 y #20)

Dos módulos nuevos, no parte del pipeline canónico (`make core`), pensados para responder puntualmente a dos preguntas de la revisión posterior del proyecto sobre sesgos no evaluados:

* **`chr_representativeness.py`** (sin red): parsea el VCF crudo de ClinVar ya descargado (contiene el genoma completo; el recorte a chr1-3 ocurre después, en `annotate.py`) y compara la distribución de chr1-3 frente al resto del genoma con el mismo PSI (`_psi_categorical`, reutilizado de `src/monitor/drift.py`) que el proyecto ya usa para detectar deriva entre *releases* — aquí aplicado entre poblaciones cromosómicas en vez de temporales. Resultado real: PSI≈0 en distribución de clases, estado de revisión y consecuencia funcional.
* **`transcript_aggregation_sensitivity.py`** (con red: re-consulta myvariant.info): compara agregar por media (comportamiento actual de `multi_source.py`) frente a por máximo para las variantes ya etiquetadas, y mide el efecto en PR-AUC por validación cruzada. Resultado real: diferencia de +0,0001, indistinguible del ruido entre particiones.

Ambos se documentan con detalle en `Discusion.tex` (sección de alcance final del sistema, y sección de anotación multi-fuente respectivamente) y generan sus propios CSV en `reports/training/`.

### 10.2 `explain.py` — Explicabilidad SHAP

Complementa la importancia por permutación con **valores SHAP** (dirección y magnitud del efecto por instancia, no solo un ranking agregado). Explica el **Pipeline completo** (preprocesado + clasificador) como caja negra, con `shap.Explainer` agnóstico al modelo (debe funcionar igual para LR, RF o GB), sobre las columnas de entrada crudas.

**Problema técnico resuelto:** el masker tabular de SHAP compara valores con `np.isclose`, que no admite texto, y `consequence` es categórica — se codifica a enteros antes de invocar SHAP y se decodifica justo antes de cada llamada real al pipeline. También conserva el índice original de las filas muestreadas (en vez de resetearlo), para poder recuperar la identidad de cada variante sin ambigüedad tras el muestreo interno de SHAP.

**Resultado real** (`reports/training/shap_importance.csv`): `cadd_phred` (0,1076) > `gnomad_af` (0,0472) > `consequence` (0,0351) > `revel_score` (0,0193) > `alphamissense_score` (0,0161) > `phylop` (0,0154) > `polyphen_score` (0,0031) > `gerp_rs` (0,0023) > `sift_score` (0,0018).

### 10.3 `acmg_evidence.py` — SHAP → evidencia tipo ACMG/AMP

El módulo más específico del dominio clínico, y quizá el que mejor ilustra el estilo del proyecto: traduce cada contribución SHAP a un código de evidencia reconocible por un genetista clínico, con la advertencia explícita de que **no es una clasificación ACMG certificada** (los criterios reales exigen segregación familiar, evidencia funcional directa y revisión de comité — datos que este proyecto no tiene). Cada etiqueta lleva el sufijo `-like` obligatorio.

**Reglas concretas implementadas:**

| Código | Señal que lo dispara | Condición |
|---|---|---|
| **PP3-like** | Scores in silico (CADD≥20, REVEL≥0,7, AlphaMissense≥0,7, PolyPhen≥0,85, SIFT≤0,05, GERP≥4,0, phyloP≥5,0) coherentes con SHAP positivo | Score "dañino" Y SHAP empuja a patogénica |
| **BP4-like** | Mismos scores, coherentes con SHAP negativo | Score "no dañino" Y SHAP empuja a benigna |
| **PM2-like** | Frecuencia gnomAD muy baja (<1e-4) coherente con SHAP positivo | Variante rara/ausente en población |
| **BA1-like** | Frecuencia gnomAD alta (>0,01) coherente con SHAP negativo | Demasiado común para enfermedad rara |
| **PVS1-like** | Consecuencia de pérdida de función (stop_gained/nonsense, frameshift, splice_donor/acceptor) coherente con SHAP positivo | Pérdida de función predicha |

**Regla de coherencia (el corazón de la heurística):** solo se reporta evidencia si el valor del score y el signo de SHAP apuntan en la **misma dirección**. Si un score parece "dañino" pero el SHAP es negativo (el modelo lo está usando para bajar el riesgo en ese contexto concreto), no se reporta nada — se interpreta como una interacción más compleja del modelo, no como evidencia clara y unívoca.

**Bug real corregido, relevante para entender el dominio:** ClinVar real antepone el término Sequence Ontology al campo de consecuencia (`SO:0001587|nonsense`), mientras que el generador sintético usa el término plano. Sin normalizar (`consequence.rsplit("|", 1)[-1]`), la regla PVS1-like — la de mayor fuerza clínica del catálogo — **no coincidía con ningún valor real de ClinVar** (0 de 3.504 filas). También hay que aceptar tanto `nonsense` (término de ClinVar) como `stop_gained` (término de VEP/Ensembl) como sinónimos del mismo concepto.

**Salida por variante:** lista de evidencias ordenada por `|SHAP|` descendente, cada una con `feature, value, shap, direction, acmg_tag, note` (texto legible en español). No hay heurística para PS1, PM1, PM3-PM6, PP1/PP2/PP4/PP5, BS1-BS4, BP1-BP3/BP5-BP7 — exigirían datos de segregación/funcionales que el proyecto no tiene, consistente con la limitación declarada.

---

## 11. Capa 8 — Monitorización y reentrenamiento (`src/monitor/`) [OE5]

### 11.1 `drift.py` — motor estadístico

Dos pruebas complementarias por columna numérica: **KS** (Kolmogorov-Smirnov de dos muestras, `p<0,05`) y **PSI** (Population Stability Index, umbral `>0,2`); para categóricas, solo PSI. Basta que **una** de las dos condiciones se cumpla para marcar drift en esa columna. `drift_summary` agrega a nivel de dataset: `share = n_drifted / n_features`, `alert = share >= threshold` (`monitor.drift_threshold` de config, `0.15`).

### 11.2 `drift_report.py` — el informe completo [esa etapa+esa etapa]

Compara la release de referencia (2023-12) contra la actual (2025-06) en tres señales paralelas:

1. **Drift de covariables** (las 9 features de anotación): vía `drift.py`.
2. **Deriva de reclasificación** (antes descrita como "concept drift" sin matiz; corregido en la revisión posterior del proyecto 2026-08-18 — que una VUS reciba veredicto no demuestra por sí solo un cambio en $P(Y\mid X)$, solo que se completó una etiqueta que antes faltaba): variantes que eran VUS estricta en la release antigua y aparecen resueltas en la nueva (`is_vus` + `is_resolved` de `reclassification.py`, unificado con `build_dataset.py` — antes este módulo usaba su propio literal `"Uncertain_significance"` de forma independiente). Si hay modelo entrenado disponible, calcula además `model_accuracy_on_reclassified` — cuánto acierta el modelo (entrenado cuando estas variantes eran VUS, excluidas del entrenamiento) frente al veredicto clínico nuevo. **Alerta si la proporción de reclasificadas ≥ 2 %** (umbral hardcodeado, corrección explícita de un hallazgo de la revisión interna: originalmente alertaba con *cualquier* reclasificación, "demasiado laxo").
3. **Stress test**: perturbación controlada (`cadd_phred += 8`, `gnomad_af *= 0.1`, `gerp_rs += N(2, 0.5)`) para verificar que el mecanismo de alerta realmente se dispara ante un drift claro — no cuenta para la alerta global, es solo una validación del propio mecanismo.

También genera un **informe HTML con Evidently AI** (`reports/monitoring/evidently_drift.html`), envuelto en `try/except` — si Evidently no está disponible, se omite con aviso, sin romper el flujo (patrón de degradación explícita consistente con el resto del proyecto).

`overall_alert = covariate_drift.alert OR concept_drift.alert` (el stress test no cuenta).

**Cifras canónicas (ejecución vigente, 2026-08-18):** `reference_release=2023-12`, `current_release=2025-06`; drift de covariables 0/9 (`alert=false`); deriva de reclasificación **55 reclasificadas de 7997 compartidas (0,69 %)**, por debajo del umbral 2 % (`alert=false`), acierto del modelo en ellas **0,964**; stress test dispara 3/9 (`alert=true`, confirma que el mecanismo funciona); `overall_alert=false`. El recuento (55) coincide ahora exactamente con el de la población del modelo de reclasificación (§9.2), tras unificar la definición de VUS entre ambos módulos — antes citaban 54 y 67 respectivamente, sin reconciliar. Documentos de ejecuciones anteriores citan cifras de ejecuciones previas del pipeline con datos ligeramente distintos — no son un error del código, sino la consecuencia esperable de regenerar el dataset en sesiones distintas sin fijar una única ejecución citable; la fuente vigente es siempre la que cita `memoria/capitulos/Desarrollo.tex` en cada momento.

### 11.3 `retrain.py` — el gate humano [esa etapa]

Gobernanza declarada en la cabecera del módulo, conforme a ADR 001: en un contexto clínico, promover un modelo automáticamente sería inaceptable — el sistema **recomienda**, un humano **aprueba**.

`decide(summary)`: `retrain_recommended = covariate_drift.alert OR concept_drift.alert` (lógica de negocio pura, sin efectos secundarios).

```bash
python -m src.monitor.retrain # dry-run (por defecto): solo recomienda, no reentrena
python -m src.monitor.retrain --execute # aprueba y ejecuta train.run de verdad
```

El modelo reentrenado entra siempre en stage **Staging** (nunca directo a Production) — la promoción es un **segundo** gate humano, independiente de este módulo. Verificado activamente en la auditoría: no existe ningún cron ni GitHub Action que dispare `retrain.py` sin intervención humana.

---

## 12. Capa 9 — Serving (`src/serve/`): dónde converge todo

### 12.1 Orden de dependencia entre los 7 módulos

```
annotator.py (almacén de anotación, reutiliza SILVER)
      │
      v
predictor.py (carga models/best_model + predice)
      │
      ├──→ app.py API REST: /health, /predict (+ registra el blueprint de dashboard.py)
      ├──→ examples.py Documenta >=10 variantes de prueba [OE4]
      └──→ prioritize_vus.py rank_vus: ordena VUS por riesgo estimado
                 │
                 v
           vus_reports.py: combina el modelo de patogenicidad + el modelo de reclasificación + la evidencia ACMG-símil en informes por VUS
                 │
                 v
           dashboard.py: SOLO LEE el JSON ya generado por vus_reports.py
```

### 12.2 `annotator.py` — almacén de anotación on-demand

En Fase I local, "anotar una variante nueva" no ejecuta VEP/gnomAD en vivo: reutiliza directamente la capa SILVER ya anotada (`annotated_{release}.parquet`) como un almacén indexado por clave. `annotate` **nunca expone `clnsig`** (la etiqueta, no un input) — probado explícitamente. Limitación documentada: una variante fuera del almacén se marca "no anotable" (en producción real llamaría a VEP).

### 12.3 `predictor.py` — inferencia pura, reutilizable

`VariantPredictor.predict`: anota + `model.predict_proba` sobre `models/best_model`, devuelve `{"prediction": "Patogénica"|"Benigna", "probability_pathogenic":...}`. `predict_batch`: variante para lotes ya anotados, usada exclusivamente por `prioritize_vus.py` para puntuar miles de VUS de golpe.

### 12.4 `app.py` — API REST [OE4]

| Ruta | Método | Auth | Notas |
|---|---|---|---|
| `/health` | GET | **Nunca** (siempre abierta) | Uso estándar de liveness probe |
| `/predict` | POST | Sí, si `TFM_API_KEY` definida | `{chrom,pos,ref,alt}` → predicción; 400 si faltan campos/JSON inválido/`pos` no numérico (nunca 500) |
| `/dashboard[/&lt;split&gt;]` | GET | Sí, si `TFM_API_KEY` definida | Monta el blueprint de `dashboard.py` |

**Autenticación:** cabecera `X-API-Key`, comparada con `secrets.compare_digest` (evita timing attacks), exigida solo si la variable de entorno `TFM_API_KEY` está definida (opt-in, no rompe el uso local por defecto). `MAX_CONTENT_LENGTH = 1_000_000` (1 MB) corta payloads desproporcionados. Verificado con `curl` contra un servidor real: sin cabecera → 401; clave incorrecta → 401; clave correcta → 200; `/health` sigue abierta con la API key activa; payload de 2 MB → 413. El servidor de desarrollo arranca en `127.0.0.1` por defecto (no `0.0.0.0`), evitando exponerlo a toda la LAN al ejecutarlo directamente en el host; el `CMD` de gunicorn en Docker sí usa `0.0.0.0` explícitamente, correctamente aislado por el mapeo de puertos.

### 12.5 `examples.py` — documentación del servicio [OE4]

Genera y documenta ≥10 variantes de prueba, muestreadas **explícitamente excluyendo las ya vistas en train** (evaluación honesta, corrección de un hallazgo de la revisión interna). Resultado real (regeneración 2026-08-07): **9/10 aciertos** (`docs/serving_examples.md`); el único fallo es una variante missense de ATG7 (Pathogenic real, predicha Benigna con probabilidad 0,0296), documentado sin ocultarlo.

### 12.6 `prioritize_vus.py` — priorización de VUS (ADR006)

`rank_vus`: función pura (sin I/O), reutilizada también por `vus_reports.py`. Desde el 2026-08-07, `load_ranking_model` carga el booster del objetivo de ranking (`models/ranking_model/lambdarank.txt`) y su preprocesador ya ajustado (`preprocessor.joblib`); si existen, `rank_vus` ordena las VUS por `ranking_score` (el criterio del objetivo de ranking, entrenado para optimizar directamente el orden — ver ADR 007 §5.3); si no, cae explícitamente a `probabilidad_patogenica` del modelo de patogenicidad. El criterio explícito en el docstring: *"revisar antes las de arriba"*, dirige la investigación manual, no la sustituye. Genera `reports/serving/vus_priorizadas_{split}.csv` (todas las VUS, con columna `ranking_score` cuando aplica) y `docs/vus_priorizadas.md` (top-N). Resultado real (ejecución vigente, orden por el objetivo de ranking): **6042** VUS en la release test (VUS estricta, tras la corrección de taxonomía de §7); el top 1 (gen ARL6IP6, `nonsense`) alcanza `ranking_score=6,346` con `probabilidad_patogenica=0,9966`.

### 12.7 `vus_reports.py` — informes automáticos por VUS

El punto donde converge casi todo lo demás: cada informe cita señales de cuatro módulos distintos — probabilidad de patogenicidad, score de ranking (el objetivo de ranking, si está entrenado — mismo criterio de orden que la priorización de VUS, vía el `load_ranking_model` compartido), probabilidad de reclasificación próxima (con su aviso de fiabilidad si aplica) y evidencia SHAP→ACMG. **Deliberadamente generado por plantilla, no por un LLM libre** — el docstring lo justifica: *"en un informe que puede influir en qué VUS revisa antes un clínico, un texto inventado es un riesgo real, y una plantilla con los datos reales del modelo no lo tiene."* Degrada explícitamente si el modelo de reclasificación o el objetivo de ranking no están entrenados, nunca falla en silencio.

### 12.8 `dashboard.py` — dashboard interactivo (ADR007 §5)

Blueprint Flask con tabla ordenable (JS vanilla, sin dependencias nuevas) sobre los informes **ya generados** por el generador de informes por VUS — deliberadamente no recalcula SHAP por petición HTTP (sería demasiado lento), ni carga tampoco el booster del objetivo de ranking directamente: lee el `ranking_score` ya calculado en el JSON del generador de informes por VUS. Cabecera **dinámica** (desde 2026-08-07): declara si el orden mostrado viene del score del objetivo de ranking o de la probabilidad del modelo de patogenicidad, según si los registros traen `ranking_score`. Muestra también la etiqueta visual "señal débil" junto a la probabilidad de reclasificación cuando `probabilidad_reclasificacion_fiable=False`. Validación explícita de `split` contra `{"train","test"}` (404 en cualquier otro valor, defensa en profundidad aunque el path traversal no fuera realmente explotable).

### 12.9 Qué modelo carga cada componente

| Componente | el modelo de patogenicidad `best_model` | el modelo de reclasificación `reclassification_model` | el objetivo de ranking `ranking_model` |
|---|---|---|---|
| `predictor.py` / `/predict` | Sí (único) | No | No |
| `prioritize_vus.py` | Sí (probabilidad mostrada siempre) | No | Sí, si existe — **decide el orden**; fallback al modelo de patogenicidad si no |
| `vus_reports.py` | Sí | Sí, si existe | Sí, si existe — mismo criterio que la priorización de VUS |
| `dashboard.py` | No (lee JSON ya generado) | No | No directamente (lee `ranking_score` ya calculado por el generador de informes por VUS) |

---

## 13. Infraestructura, calidad y CI/CD

### 13.1 Contenedores

Dos imágenes (`python:3.12-slim`): `docker/app.Dockerfile` (pipeline/desarrollo, reutilizable también para Cloud Run en Fase II) y `docker/mlflow.Dockerfile` (servidor MLflow, backend SQLite). `compose.yaml` define 3 servicios:

| Servicio | Puerto | Rol |
|---|---|---|
| `mlflow` | 5000 | Tracking + Model Registry (backend SQLite, artefactos en volumen `mlflow_data`) |
| `app` | — | Shell de trabajo (`sleep infinity`), para ejecutar `make...` dentro con bind mount del repo |
| `serve` | 8000 | `gunicorn src.serve.app:app` — servicio real de inferencia + dashboard |

### 13.2 CI (`.github/workflows/ci.yml`)

Un job (`lint_test`) en `ubuntu-latest`, Python 3.12, en push/PR a `main`: `ruff check src tests` (bloqueante) → `pip-audit -r requirements.txt` (no bloqueante, `continue-on-error: true`, porque detecta un CVE de terceros sin fix publicado que no debe tumbar el pipeline) → `pytest -q` (bloqueante).

### 13.3 Calidad de código

`pyproject.toml`: `ruff` con rulesets `E, F, I, W, UP, B, S` (el ruleset de seguridad `S` se activó como remediación de un hallazgo de la revisión interna). `requirements.txt` con **versiones exactas fijadas** (`==`), sin rangos abiertos. `pytest` con `pythonpath=["."]`.

### 13.4 Versionado de datos (DVC)

`data/raw` trackeado con `dvc add` → `data/raw.dvc` (hash MD5 del directorio, 4 ficheros). **Sin remoto configurado** todavía (no hay dato real a gran escala que empujar); tras clonar el repo hay que regenerar con `make data` (determinista, mismo hash).

### 13.5 Suite de tests — 95 tests, agrupados por capa

| Capa | Ficheros de test |
|---|---|
| Ingesta/anotación | `test_synthetic.py`, `test_annotate.py`, `test_multi_source.py`, `test_config.py` |
| Features | `test_build_dataset.py` (+ taxonomía de etiquetas), `test_preprocess.py` |
| Entrenamiento | `test_train.py`, `test_train_reclass.py` (+ validación prospectiva), `test_train_ranking.py` (+ baselines), `test_explain.py`, `test_acmg_evidence.py` |
| Evaluación/procedencia | `test_capture_provenance.py`, `test_chr_representativeness.py` (nuevo), `test_transcript_aggregation_sensitivity.py` (nuevo) |
| Monitorización | `test_monitor.py` |
| Serving | `test_serve.py`, `test_app.py`, `test_dashboard.py`, `test_prioritize_vus.py`, `test_vus_reports.py` |
| Integración | `test_e2e_pipeline.py` (único test que ejecuta ingesta→anotación→dataset→entrenamiento→drift en cadena, sobre datos aislados en `tmp_path`, sin tocar el repo real) |

La suite pasó de 55 a 63 (cierre de la falta de pruebas de la capa HTTP), a 74 (remediación completa del 29/07/2026), a 83 tras el cierre de ADR 008, a 88 tras la primera ronda de la revisión posterior del proyecto de 2026-08-18 (taxonomía de etiquetas, validación prospectiva del modelo de reclasificación ×2, baselines de ranking, captura de procedencia), y a **95 tests** tras cerrar los hallazgos restantes de la misma auditoría ese mismo día (calibración del modelo de patogenicidad, representatividad de chr1-3, sensibilidad de agregación por transcrito). Corregido además un problema de reproducibilidad: `RandomForestClassifier` y `permutation_importance` usaban `n_jobs=-1`, lo que en Windows podía dejar procesos de `joblib`/`loky` sin cerrar entre tests sucesivos que reentrenan el modelo, degradando el sistema hasta bloquear la ejecución completa de la suite en sesiones largas; corregido a `n_jobs=1` sin alterar ningún resultado citado (con `random_state` fijo, el modelo entrenado es idéntico).

---

## 14. Auditorías: qué se encontró y cómo se corrigió

El proyecto se ha revisado a sí mismo con severidad real en tres rondas sucesivas. Tres rondas relevantes:

### 14.1 Revisión inicial de la Fase I

- **Fuga entre entrenamiento y evaluación (crítico):** fuga train/test — el 79,7 % del test ya estaba en train, inflando PR-AUC a 0,988. Corregido con el criterio de "holdout no visto" (ver §9.1); el modelo ganador cambió al evaluar honestamente.
- **Variantes de prueba ya vistas (crítico):** las variantes documentadas del servicio estaban mayoritariamente ya vistas en train. Corregido muestreando solo variantes no vistas.
- **Umbral de alerta demasiado laxo (medio):** la alerta de concept drift se disparaba con cualquier reclasificación. Corregida a un umbral de proporción (2 %).
- **Bloqueos con datos reales:** ClinVar real tiene ALT multialélico/estructural que el contrato rechazaba; y sin features reales (dbNSFP no accesible), el entrenamiento fallaba con 0 columnas de datos — este último hallazgo fue el disparador directo del pivote (ADR 007).

### 14.2 Revisión del estado del sistema construido

| Hallazgo | Severidad | Resumen | Estado |
|---|---|---|---|
| Fiabilidad del modelo | Alta | El modelo de reclasificación sin poder predictivo real en una ejecución concreta (ROC-AUC≈0,5) | Resuelto: umbral de fiabilidad + aviso explícito en dashboard/informes/Model Card |
| Reproducibilidad del ranking | Alta | El objetivo de ranking no reproducible pese a semilla fija (NDCG@10=1,0 vs 0,70 al reejecutar) | Resuelto: flags de determinismo de LightGBM, verificado empíricamente |
| Autenticación del servicio | Media | Sin autenticación en `/predict`/`/dashboard` | Resuelto: `X-API-Key` opt-in + límite de payload, probado con `curl` real |
| Validación de entrada | Baja | `split` de `/dashboard/&lt;split&gt;` sin validar contra una lista permitida | Resuelto: `abort(404)` si no está en `{train,test}` |
| Dependencia vulnerable | Baja | CVE `PYSEC-2026-2447` en `diskcache` (transitiva de DVC) | Abierto por diseño — sin fix publicado, vigilado en CI |
| Cobertura de pruebas | Media | La API Flask no tenía ningún test | Resuelto: `tests/test_app.py`, 14 tests |
| Integración continua | Baja | CI sin `pip-audit`; `ruff` sin conjunto de reglas de seguridad | Ambos resueltos |

Todo lo marcado "resuelto" se verificó con ejecución real (reentrenamiento real del modelo de reclasificación, servidor Flask real probado con `curl`), no solo con tests aislados — este nivel de rigor en la propia auditoría es, en sí mismo, un argumento defendible en la memoria.

### 14.3 Revisión posterior del proyecto (2026-08-18)

Tercera revisión, con foco en validez científica y metodológica, no solo en calidad de ingeniería. Los hallazgos CRÍTICOS y ALTOS se corrigieron con evidencia de ejecución real, no solo reformulando el texto a la baja:

| Hallazgo | Severidad | Resumen | Corrección |
|---|---|---|---|
| Sin validación temporal prospectiva del modelo de reclasificación | Crítico | el modelo de reclasificación se entrenaba y evaluaba dentro del mismo intervalo histórico 2023-2025; ninguna evidencia de predicción de una reclasificación genuinamente futura | Descarga y evaluación real contra una 3ª release de ClinVar (2026-08); resultado más débil (ROC-AUC 0,604), reportado tal cual (§9.2 bis) |
| Leakage temporal en el modelo de reclasificación, ablación "confirmaba" ausencia de fuga | Crítico | El lenguaje afirmaba que la ablación segura demostraba que el modelo completo no dependía de fuga | Tras corregir la taxonomía de VUS, la ablación ya no supera al modelo completo; el texto ya no afirma ausencia de fuga, dice que no puede descartarse (§9.2) |
| Ranking sin target/grupo/baseline definidos | Crítico | NDCG@10=0,78 sin ninguna referencia con la que interpretarlo, ni documentación del objetivo de entrenamiento | Baselines de orden de llegada y aleatorio añadidos; población de entrenamiento (variantes resueltas) vs. de aplicación (VUS) documentada como limitación de proxy (§9.3) |
| Target no distingue "anticipar reclasificación" de "acelerar diagnóstico" | Crítico | Claims de beneficio clínico (tiempo de revisión, más diagnósticos) sin evidencia que los sostenga | Separación explícita de ambos claims en la memoria; el segundo se retira como resultado no demostrado |
| Mismo holdout para seleccionar y evaluar el modelo | Alto | Selection bias: el "ganador" se elegía por el mismo conjunto citado como resultado final | Selección por CV en train únicamente; el holdout se evalúa una sola vez (§9.1, §9.2) |
| Evidencia estadística insuficiente en el modelo de reclasificación | Alto | Sin IC, sin calibración, sin métricas de cola de revisión | precision@k/recall@k/lift@k, calibración, IC bootstrap, curvas PR/ROC completas (§9.2) |
| "Competitivo con predictores in silico" sin comparación directa | Alto | CADD/REVEL/AlphaMissense son features del propio modelo, nunca evaluados solos | Comparativa head-to-head real con IC bootstrap de la diferencia (§9.1) |
| Ground truth de ClinVar / taxonomía de VUS no documentada | Alto | No estaba claro qué valores de CLNSIG eran VUS, y esa ambigüedad causaba una discrepancia numérica real (67 vs. 54) | Taxonomía estricta y unificada; tabla exhaustiva CLNSIG→etiqueta (§7, `docs/datasheet.md`) |
| Reproducibilidad no verificable desde la memoria | Alto | Sin commit, sin run IDs, sin hash de datos citados | `src/evaluate/capture_provenance.py`, `reports/provenance.json` (nuevo) |
| Terminología "concept drift" imprecisa | Alto | El evento VUS→resuelta se llamaba concept drift sin prueba de cambio en P(Y\|X) | Renombrado a "deriva de reclasificación" en el uso operativo del proyecto (§11.2) |
| Ausencia de gnomAD interpretada como rareza sin matiz | Alto | No se distinguía rareza biológica real de fallo de cobertura de la anotación | Matizado en el texto; indicador de ausencia añadido como feature real, con mejora medible de los tres modelos (§8.1) |

No abordado en esta revisión, declarado explícitamente como pendiente: sensibilidad de la agregación de anotaciones por transcrito (media vs. transcrito canónico). Detalle completo de los hallazgos y su estado en la bitácora del proyecto (entrada 2026-08-18) y en `memoria/capitulos/Discusion.tex`/`TrabajoFuturo.tex`.

---

## 15. Cifras clave del proyecto (para citar en la memoria, con procedencia)

*Las cifras de esta tabla se sincronizaron con la memoria (`memoria/capitulos/Desarrollo.tex`), que cita la ejecución canónica más reciente, posterior a la revisión posterior del proyecto de 2026-08-18. Es la ejecución a citar en la memoria; `reports/provenance.json` registra su commit, hashes de datos y run IDs de MLflow exactos.*

| Métrica | Valor | Fuente |
|---|---|---|
| Anotación (multi_source) | 2023-12: 8000 var. anotadas · 2025-06: 11997 var. anotadas · release prospectiva 2026-08: descargada, solo verdad terreno (sin reanotar) | `make annotate`, `make ingest-prospective` |
| Dataset | train 3504 etiquetadas (prev. 0,150), **4056 VUS estricta + 440 excluidas** · test 5221 etiquetadas (prev. 0,137), **6042 VUS estricta + 734 excluidas** | `make dataset`, `docs/datasheet.md` |
| Modelo de patogenicidad, mejor algoritmo | **logistic_regression** (seleccionado por CV en train), PR-AUC holdout **0,9700** [IC 0,9511-0,9849], ROC-AUC 0,9945, n_holdout=1764 | `docs/MODEL_CARD.md` |
| el modelo de patogenicidad vs. predictores in silico en solitario | Ensemble distinguiblemente mejor que CADD, REVEL y AlphaMissense solos (IC bootstrap de la diferencia, ninguno cruza cero) | `reports/training/compare_predictors.csv` |
| Modelo de reclasificación, mejor algoritmo, retrospectivo | **logistic_regression**, PR-AUC 0,1093, ROC-AUC 0,7786 (fiable, umbral 0,6); ablación temporalmente segura: ROC-AUC 0,7476 (ya no supera al modelo completo) | `docs/MODEL_CARD_RECLASSIFICATION.md` |
| Modelo de reclasificación, **validación prospectiva real** | ROC-AUC **0,604** (modelo completo) / 0,447 (ablación), sobre 7 casos positivos de 3986 VUS evaluadas — más débil que el retrospectivo, potencia estadística limitada | `docs/MODEL_CARD_RECLASSIFICATION_PROSPECTIVE.md` |
| Ranking | NDCG@10=**0,890**, NDCG@50=0,944, NDCG@100=0,966, PR-AUC ref.=0,798; baselines: orden de llegada NDCG@10=0,000, aleatorio NDCG@10=0,140±0,119 | `reports/training/ranking_metrics.csv` |
| el objetivo de ranking, integración en serving | Criterio de orden real de los tres modelos (desde 2026-08-07), sobre población de VUS (limitación de proxy declarada, entrena sobre variantes resueltas) | `src/serve/prioritize_vus.py::load_ranking_model` |
| Cobertura de anotación real (myvariant.info) | 90-100 % en CADD/SIFT/PolyPhen/REVEL/AlphaMissense/GERP/phyloP; 72 % gnomAD AF (indicador de ausencia conservado como feature) | validación B5 |
| SpliceAI | No accesible desde el entorno de desarrollo; descartado con evidencia tras una segunda investigación (ADR 008) | ADR 007 §1, ADR 008, `multi_source.py` |
| Servicio REST, aciertos en variantes documentadas | 9/10 (no vistas en train) | `docs/serving_examples.md` |
| Monitorización | covariables 0/9 (alerta=False); deriva de reclasificación 55/7997 (0,69 %, alerta=False), acierto=0,964; stress test 3/9 (alerta=True); alerta global=False | `reports/monitoring/drift_summary.json` |
| Suite de tests | **95 passed**, 0 fallos | ver también nota de reproducibilidad más abajo |
| VUS reservadas priorizadas (release test) | 6042, ordenadas por el objetivo de ranking | `docs/vus_priorizadas.md` |

**Nota de honestidad metodológica:** documentos de ejecuciones anteriores (revisión interna del proyecto) citan cifras de ejecuciones previas del pipeline, ligeramente distintas de las de la tabla anterior — no son errores, son fotos de otros momentos. La fuente única para la memoria es siempre la ejecución más reciente citada en `memoria/capitulos/Desarrollo.tex`; si se regenera el pipeline de nuevo más adelante, hay que repetir este ejercicio de fijar una única foto citable en vez de mezclar cifras de ejecuciones distintas.

**Nota de reproducibilidad de la suite de tests:** `RandomForestClassifier` y `permutation_importance` en `src/train/train.py` usaban `n_jobs=-1`; en Windows esto podía dejar procesos de `joblib`/`loky` sin cerrar entre tests sucesivos que reentrenan el modelo, degradando el sistema hasta bloquear la ejecución completa de la suite en sesiones largas. Corregido a `n_jobs=1` (no afecta a ningún resultado citado: con `random_state` fijo, el modelo entrenado es idéntico, solo cambia la paralelización interna).

---

## 16. Estado actual y próximos pasos

El **núcleo del proyecto está completo** — B5 (anotación multi-fuente real) y los 7 puntos del bloque de trabajo (los componentes del núcleo) construidos, probados y auditados. La revisión posterior del proyecto de 2026-08-18 (§14.3) cerró los hallazgos de una revisión posterior del proyecto, con los CRÍTICOS y ALTOS corregidos con evidencia de ejecución real. Nada bloqueante pendiente en Fase I.

**Pendiente explícito, no bloqueante:**
- Fase II (GCP): BigQuery Sandbox, arquitectura medallion, comparativa local vs. cloud — diseño documentado (`docs/FASE_II_DISENO_GCP.md`, `cloud/`), no ejecutado.
- Ampliar la validación prospectiva del modelo de reclasificación con más releases (el experimento actual tiene solo 7 positivos, insuficiente para una conclusión firme por sí solo).
- Sensibilidad de la agregación de anotaciones por transcrito (no abordada en la revisión de 2026-08-18 por límite de tiempo).
- Snapshots históricos de CADD/REVEL/AlphaMissense/gnomAD anclados por fecha de release (mitigación completa, no solo parcial, de la fuga temporal del modelo de reclasificación).
- Trabajo futuro explícitamente descartado con motivo (no construir): integración literal con Talos/Exomiser/LIRICAL/Seqr (requieren datos de paciente), modelado de tríos/herencia, arquitecturas GNN/Transformer, PrimateAI-3D/EVE como fuente primaria.

**Resuelto el 2026-08-07:** el objetivo de ranking integrado como criterio de orden real en los tres modelos. **Resuelto el 2026-08-18:** validación temporal prospectiva real del modelo de reclasificación, selección de modelo separada de la evaluación final, taxonomía de etiquetas unificada, rigor estadístico adicional, comparativa con predictores in silico, procedencia reproducible (ver §14.3, §15).

---

## 17. Glosario mínimo para seguir leyendo sin conocimiento previo del dominio

- **VUS (Variant of Uncertain Significance):** variante genética cuya patogenicidad no se ha podido determinar con la evidencia disponible; ni "Pathogenic" ni "Benign" en ClinVar.
- **ClinVar:** base de datos pública de NCBI que archiva la interpretación clínica de variantes genéticas, con releases fechadas periódicas.
- **ACMG/AMP:** criterios estándar (Richards et al. 2015) usados por genetistas clínicos para clasificar formalmente la patogenicidad de una variante; exigen evidencia que este proyecto no tiene (segregación familiar, ensayos funcionales, revisión de comité) — por eso toda evidencia generada aquí se marca "-like", heurística, no certificada.
- **SNV (Single Nucleotide Variant):** cambio de una sola base del ADN; el único tipo de variante que cubre este proyecto (no CNV ni estructurales).
- **gnomAD:** base de datos pública de frecuencias alélicas poblacionales; una variante muy común en población general es rara vez patogénica.
- **CADD, REVEL, AlphaMissense, SIFT, PolyPhen, GERP, phyloP:** distintos scores computacionales ("in silico") que estiman cuán perjudicial es una variante o cuán conservada evolutivamente está esa posición del genoma.
- **PR-AUC (Average Precision):** área bajo la curva precisión-recall; métrica de clasificación robusta al desbalanceo de clases (preferida sobre accuracy en este proyecto).
- **NDCG@k (Normalized Discounted Cumulative Gain):** métrica estándar de calidad de un ranking, que pondera más los errores cerca de la cima de la lista.
- **SHAP (SHapley Additive exPlanations):** técnica de explicabilidad que atribuye a cada feature una contribución (con signo) a la predicción de una instancia concreta.
- **Concept drift / drift de covariables:** degradación de un modelo con el tiempo; *concept drift*, en sentido estricto, es cuando cambia la relación $P(Y\mid X)$ entre features y etiqueta; *drift de covariables* es cuando cambia la distribución de las features en sí. Este proyecto usa el término más preciso **"deriva de reclasificación"** para el evento VUS→resuelta entre releases de ClinVar, porque una etiqueta que se completa no demuestra por sí sola un cambio en $P(Y\mid X)$ (revisión posterior del proyecto 2026-08-18).
- **Validación prospectiva vs. retrospectiva:** un modelo se valida de forma *retrospectiva* cuando se entrena y se evalúa dentro del mismo intervalo histórico ya conocido (p. ej. un holdout aleatorio sobre datos de 2023-2025); se valida de forma *prospectiva* cuando se aplica, sin reentrenar, sobre verdad terreno publicada después de fijar el modelo — la única evaluación que demuestra capacidad de predicción futura genuina.
- **MLflow (Tracking + Model Registry):** herramienta de registro de experimentos (parámetros, métricas, artefactos) y de gestión del ciclo de vida de modelos (stages `None→Staging→Production`).
- **Medallion (RAW/SILVER/GOLD):** patrón de arquitectura de datos en capas de calidad creciente: crudo inmutable → enriquecido/anotado → listo para modelado.
