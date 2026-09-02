# ADR 008: Cierre de SpliceAI, estado de revisión de ClinVar y fuga temporal

**Fecha:** 2026-08-12 · **Estado:** confirmado e implementado

## Contexto

Analicé, solo como lectura, si tenía sentido incorporar componentes de una arquitectura de anotación multi-fuente más amplia (VEP, dbNSFP, SpliceAI, RENOVO, GeneBe, InterVar). La conclusión fue que la mayor parte ya estaba cubierta por el núcleo (ADR 007) y que el valor real no estaba en sumar anotadores, sino en tres piezas de menor coste y mayor impacto directo sobre el modelo de reclasificación:

1. Resolver o cerrar de forma justificada la integración de SpliceAI, documentada como NaN desde ADR 007 y nunca cerrada.
2. Añadir el estado de revisión de ClinVar (`CLNREVSTAT`) como feature de ese modelo: es un predictor con respaldo en la literatura de reclasificación —más consenso entre submitters, más probable que una VUS se resuelva— que el pipeline no extraía en absoluto del VCF pese a estar ya disponible sin coste.
3. Un hallazgo del propio análisis, no documentado hasta entonces: las features de anotación se consultan **en el momento de ejecutar el pipeline**, sin anclaje a la fecha de la *release*.

No toco RENOVO, GeneBe, InterVar ni VEP: no aportan capacidad que el sistema no tenga y exponen al mismo riesgo de bloqueo de licencia o registro que ya obligó a descartar dbNSFP. Quedan como comparadores bibliográficos.

## 1. SpliceAI: cierre con evidencia

Investigué si la ejecución **local** del paquete `spliceai` era viable, como alternativa a la API del Broad ya documentada como inaccesible:

* **API del Broad:** confirmada de nuevo inaccesible (`curl` con timeout de 10 s a `spliceailookup-api.broadinstitute.org`, `HTTP 000`, exit 28).
* **Dependencias del paquete:** `keras`, `tensorflow`, `pysam`, `pyfaidx`, `numpy`, `pandas`.
* **Bloqueo real:** `pysam` no publica wheels para Windows. Depende de `htslib` vía extensiones C con soporte oficial solo en Linux y macOS; sin wheel, exigiría compilar desde fuente con un toolchain C completo.
* **Alternativa Docker:** el proyecto ya usa contenedores Linux donde `pysam` sí tiene wheels, pero `spliceai` requiere además el genoma de referencia GRCh38 completo, del orden de varios gigabytes, más los ficheros de anotación de genes. Es una descarga desproporcionada para el alcance, y pesa de forma permanente en cualquier reproducción futura.

**Decisión:** no persigo la ejecución local de SpliceAI. Lo documento como intentado y descartado con evidencia, no como pendiente sin más, y corrijo la estimación de esfuerzo: alta en Windows nativo, media-alta en Docker por el genoma de referencia. El código existente ya degrada a NaN documentado y se mantiene sin cambios. Queda como trabajo futuro, condicionado a un entorno Linux con más tiempo y almacenamiento.

## 2. Estado de revisión de ClinVar como feature

Añado `review_status` (texto crudo de `CLNREVSTAT`) y `review_stars` (0 a 4, escala oficial de ClinVar), extraídos directamente del VCF de cada *release*.

**Hallazgo no anticipado:** al inspeccionar los dos VCF reales confirmé que **ClinVar cambió el vocabulario de `CLNREVSTAT` entre ambas releases**: `criteria_provided,_conflicting_interpretations` pasa a `criteria_provided,_conflicting_classifications`, y `no_assertion_provided` incorpora también `no_classification_provided`. El mapeo reconoce ambos vocabularios; un valor no reconocido queda NaN documentado, nunca un supuesto silencioso.

**Por qué solo en el modelo de reclasificación.** `review_stars` no entra en las features compartidas, sino en un conjunto propio usado únicamente por ese modelo, por dos motivos. Para predecir la propia significancia clínica, el estado de revisión está confundido con la certeza de la etiqueta que se intenta predecir: es un argumento casi definicional, distinto de la fuga temporal, pero igualmente indeseable, y no es una entrada causal de la biología de la variante. Para predecir si una VUS se resolverá, en cambio, el estado de revisión en el momento inicial es una entrada legítima y con respaldo en la literatura.

## 3. Fuga temporal de las features

### El problema

`annotate_multi_source` consulta myvariant.info en el momento de ejecutar el pipeline, sin ningún parámetro de fecha ni de versión. El mismo estado "de hoy" se aplica tanto a las variantes de la *release* antigua como a las de la nueva. No existe un snapshot congelado a la fecha de cada una.

Para el modelo de reclasificación, cuyo objetivo es predecir con información del momento inicial si una VUS se resolverá después, esto es un problema real y no solo teórico: si una VUS se reclasificó precisamente porque llegó evidencia computacional nueva —AlphaMissense, publicado en 2023, incorporándose como evidencia PP3 o BP4 a muchas VUS con el tiempo—, el modelo entrena con el valor **posterior** de esa misma evidencia, no con el que existía cuando la variante seguía sin resolver. Es fuga de información del futuro hacia el pasado.

Ya se sabía que myvariant.info sirve lo que tenga indexado en cada consulta, pero eso se enmarcaba como un problema de reproducibilidad entre ejecuciones, no como una fuga hacia el objetivo del modelo. Ningún documento del proyecto trazaba esa implicación causal hasta aquí.

### Clasificación de features por riesgo

| Feature | Riesgo | Motivo |
|---|---|---|
| `consequence`, `review_status`, `review_stars` | Seguro | Se leen del VCF fechado de cada release |
| `sift_score`, `polyphen_score`, `gerp_rs`, `phylop` | Bajo | Scores relativamente estables desde hace más de una década |
| `cadd_phred`, `revel_score`, `gnomad_af` | Con riesgo | Se recalculan y reindexan con el tiempo |
| `alphamissense_score` | Fuga probable | Publicado en 2023, posterior a la release de entrenamiento; es justo el tipo de evidencia que motiva reclasificaciones |

### Mitigación aplicada

No es viable, dentro de este alcance, congelar snapshots históricos de CADD, REVEL, AlphaMissense y gnomAD anclados a la fecha de cada *release*: exigiría descargas versionadas por fuente que no están garantizadas como archivadas y públicamente disponibles. En su lugar aplico una mitigación parcial honesta: **cuantificar** cuánta señal depende de las features con riesgo, en vez de solo advertirlo en prosa.

`_run_safe_ablation` entrena, sobre el mismo split que el modelo completo, un modelo idéntico en algoritmo pero restringido a `consequence` y `review_stars`. El resultado se registra en MLflow con su propia etiqueta, en `metrics.json` y en la Model Card, con una interpretación automática según la brecha entre ambos ROC AUC.

### Qué no se hizo

* No congelé snapshots históricos por *release*: sería la mitigación completa, no parcial.
* No modifiqué `annotate_multi_source` para exigir una fecha de referencia: cambiaría el comportamiento de producción de forma no trivial y excede este alcance.
* No reentrené el modelo de patogenicidad sin las features con riesgo: no predice un cambio de estado en el tiempo, sino la etiqueta ya fijada de cada *release*, así que el mismo argumento no aplica con la misma fuerza. Lo dejo explícito para que quede claro por qué no se tocó.

## Decisión

SpliceAI queda cerrado como intentado y no viable, con evidencia verificable. `review_status` y `review_stars` se implementan como features del modelo de reclasificación únicamente, con el contrato de datos actualizado y un mapeo robusto a los dos vocabularios observados. La fuga temporal queda documentada con precisión causal y mitigada parcialmente mediante ablación cuantitativa, cuyo resultado se cita en la Model Card antes de cualquier cifra del modelo.
