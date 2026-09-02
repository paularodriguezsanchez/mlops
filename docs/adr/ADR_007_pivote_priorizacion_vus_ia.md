# ADR 007: Del pipeline de anotación al motor de priorización de VUS

**Fecha:** 2026-07-27 · **Estado:** confirmado

## Contexto

Hasta esta decisión el proyecto era un pipeline de anotación y clasificación con el modelo como un componente más de una cadena MLOps genérica. El motivo inmediato del cambio fue un bloqueo real: al ejecutar con ClinVar real confirmé que **dbNSFP exige un registro académico** que no llegaba a tiempo, y que sin sus features el entrenamiento no era posible en absoluto —cero columnas con datos, no señal débil—.

Eso obligaba a resolver de todos modos con qué anotar, y esa pregunta fue la puerta de entrada al cambio de foco: en vez de depender de un agregador con registro, podía construir la anotación sobre fuentes públicas directas y aprovechar esa libertad para elegir predictores más actuales que los que trae dbNSFP.

## 1. Alternativa a dbNSFP

dbNSFP es, en esencia, un agregador de otras fuentes. Se puede prescindir de él y consultarlas directamente:

| Fuente | Aporta | Acceso |
|---|---|---|
| gnomAD | Frecuencia alélica poblacional | Público, sin registro |
| CADD | Deletereidad combinada, todo el genoma | Descarga directa sin login |
| REVEL | Ensemble específico de missense | Descarga directa |
| AlphaMissense | Patogenicidad de todas las missense posibles, basado en AlphaFold | Zenodo, CC-BY-NC-SA, uso académico permitido |
| SpliceAI | Impacto en *splicing* | Paquete de código abierto, calculable en local |
| PrimateAI-3D, EVE | Predictores estructurales y de alineamientos múltiples | Acceso restringido; candidatos a trabajo futuro |

Sustituyo el subset de dbNSFP por un módulo de anotación multi-fuente, unido por la misma clave. Resuelve el bloqueo de acceso y, de paso, actualiza la capa de anotación: dbNSFP se apoya en SIFT y PolyPhen, de hace más de una década.

## 2. Estado del arte revisado

**Priorización fenotípica (Exomiser, LIRICAL).** Combinan variante, fenotipo HPO y modo de herencia. Exomiser sitúa la variante causal en primera posición el 82,6 % de las veces cuando hay fenotipo disponible, y cae drásticamente sin él. Requieren datos de paciente que las bases públicas agregadas no contienen.

**Talos** (Nature Medicine, 2026). Reanálisis automático e iterativo de cohortes combinando PanelApp, ClinVar actualizado y filtrado por herencia. En 4735 individuos sin diagnóstico encontró 241 diagnósticos nuevos (5,1 %), de los cuales un 22 % por nueva evidencia a nivel de variante.

**Modelos sobre el histórico de ClinVar.** RENOVO, un random forest sobre variantes ya resueltas, propone reclasificación para el 67 % de las VUS existentes con alta confianza, y una validación posterior sobre instantáneas trimestrales muestra que acertaba el 82,6 % de las que ClinVar acabó reclasificando cuatro años después. Es el precedente más cercano.

**Predictores de última generación.** AlphaMissense es el más consistente en benchmarks clínicos curados; PrimateAI-3D lo supera en algunas cohortes reales. La tendencia desde 2023 es incorporar estructura de proteína, no solo conservación y frecuencia.

**Explicabilidad.** La literatura señala que la adopción clínica depende tanto de la explicabilidad como del rendimiento: hacen falta scores interpretables alineados con criterios ACMG/AMP, no solo una probabilidad.

## 3. Dónde queda el hueco

| Proyecto | Qué hace bien | Por qué no es replicable aquí |
|---|---|---|
| Talos | Reanálisis a escala de cohorte | Necesita PanelApp y datos de paciente |
| Exomiser, LIRICAL | Prioriza con fenotipo | Necesita HPO del paciente |
| RENOVO | Random forest sobre reclasificaciones históricas | No explota el par de *releases* fechadas como par de entrenamiento supervisado para un modelo dedicado |
| AlphaMissense, PrimateAI-3D, EVE | Predicción de efecto muy precisa | Son entradas del sistema, no priorización orientada a decisión |

Ningún proyecto revisado hace exactamente lo que permiten los datos disponibles: bases públicas agregadas, sin pacientes, pero con dos *releases* fechadas y reclasificaciones conocidas entre ellas. Y esa misma restricción marca el límite: no puedo replicar Talos ni Exomiser porque no tengo capa de paciente, y no voy a fingir lo contrario.

## 4. La oportunidad

Las dos *releases* fechadas permiten entrenar no solo "¿es patogénica?" sino **"¿qué probabilidad tiene esta VUS de reclasificarse pronto?"**. Es factible con los datos disponibles y no requiere fenotipo ni cohortes.

## 5. Alcance confirmado

1. **Anotación multi-fuente sin dbNSFP.** Resuelve el bloqueo y actualiza la capa de anotación.
2. **Modelo de potencial de reclasificación.** Usa el par de *releases* como entrenamiento supervisado directo, en vez de limitarse a medir deriva a posteriori.
3. **Objetivo de ranking.** El entregable real es un orden de prioridad, así que entreno directamente sobre esa función objetivo (`lambdarank`) y evalúo con NDCG@k además de con métricas de clasificación.
4. **Explicabilidad traducida a evidencia ACMG.** Traduzco las contribuciones SHAP a códigos reconocibles por un genetista, siempre marcados como heurística.
5. **Reanálisis por cambio de evidencia como bucle central.** El pipeline no entrena una vez y sirve: es un ciclo continuo variante, score, revisión, reclasificación en ClinVar, reentrenamiento. Es la misma idea de fondo que Talos aplica a escala clínica, citada como inspiración, no como réplica.
6. **Dashboard interactivo** para explorar variantes, scores y evidencia.
7. **Informes automáticos por VUS basados en plantilla**, no en generación libre de un modelo de lenguaje, que citen qué evidencia empujó cada score.

**Descartado con motivo, no por falta de tiempo:** integración con Talos, Exomiser, Seqr o LIRICAL, y modelado de tríos y herencia, porque exigen datos de paciente; arquitecturas GNN o *transformer*, porque no hay estructura de grafo ni de secuencia en un problema tabular como este y el coste de ingeniería no está justificado; PrimateAI-3D y EVE como fuente primaria, por acceso restringido y cobertura limitada.

## 6. Prioridad por impacto

Por este orden: el modelo de potencial de reclasificación, que es la idea genuinamente nueva; la explicabilidad orientada a ACMG, que la literatura marca como determinante para la adopción clínica; la anotación multi-fuente, necesaria pero que es ponerse al día, no innovar; y el objetivo de ranking, que aporta rigor metodológico.

## 7. Arquitectura resultante

El núcleo es el motor de priorización —patogenicidad, potencial de reclasificación, ranking y explicabilidad—, alimentado por tres bloques: la anotación multi-fuente, el historial de *releases* de ClinVar que produce la etiqueta de reclasificación, y el registro de modelos con su gate humano. Una *release* nueva dispara reanotación, repuntuación y alerta, cerrando el ciclo.

La salida es una lista de VUS ordenada, cada una con su probabilidad de patogenicidad, su probabilidad de reclasificación próxima y su evidencia SHAP traducida a ACMG, servida por REST y diseñada para poder consumirse desde un sistema tipo Talos o Seqr como fuente de evidencia computacional, sin pretender sustituirlo.

Esto conserva casi todo lo ya construido —ingesta, contrato de datos, MLflow, REST, monitorización, reentrenamiento—: el cambio es de dónde está el peso. La anotación pasa de ser el proyecto a ser la entrada del modelo.

## Decisión

Confirmo los siete puntos anteriores como núcleo. Orden de trabajo: primero actualizar la documentación de alcance y después implementar, empezando por la anotación multi-fuente, que desbloquea datos reales para todo lo demás. El cambio amplía el alcance firmado y se comunica a la dirección.

## Actualización: el ranking como criterio de orden real

Durante un tiempo el objetivo de ranking se entrenaba y evaluaba con NDCG@k, pero el orden que veía realmente el usuario seguía viniendo de la probabilidad de patogenicidad: una revisión posterior detectó esa desconexión entre lo que se evalúa y lo que decide el orden. Valoré documentarla como limitación conocida y lo descarté, porque contradecía directamente la idea central del proyecto.

La corrección: `train_ranking.py` persiste ahora el preprocesador ya ajustado junto al booster —necesario para transformar features nuevas exactamente igual que en entrenamiento—, `load_ranking_model` carga ambos, y `rank_vus` ordena por el score de ranking cuando el modelo existe, con reversión explícita a la probabilidad de patogenicidad si no está entrenado. Los informes y el dashboard reutilizan el mismo criterio y lo declaran de forma visible, para que nunca haya ambigüedad sobre qué decidió el orden mostrado.
