# Introducción al proyecto (explicación no técnica)

Este documento explica el proyecto en lenguaje sencillo, sin dar por supuesto
conocimiento previo de genética ni de machine learning. Sirve como puerta de
entrada antes de leer el `README.md` (más técnico) o el plan del proyecto.

## El problema de partida

**ClinVar** es una base de datos pública donde se recogen variantes genéticas
humanas (cambios puntuales en el ADN) junto con su interpretación clínica: si
una variante causa enfermedad, si es inofensiva, o si no se sabe. La
alimentan laboratorios clínicos y expertos de todo el mundo.

El problema es de volumen: cada día se secuencian más genomas, y cada
secuenciación genera nuevas variantes que revisar. Pero los expertos humanos
que clasifican esas variantes no crecen al mismo ritmo. El resultado es una
acumulación creciente de variantes etiquetadas como **VUS (Variant of
Uncertain Significance)**: "no se sabe si esto es peligroso o no". Para un
paciente, tener una VUS significa quedarse sin respuesta clínica.

## Por qué no basta con las herramientas que ya existen

Hay herramientas muy buenas para priorizar variantes en un caso concreto
(Exomiser, Talos), pero funcionan con datos de un **paciente individual**:
sus síntomas (fenotipo), el árbol genealógico de su familia (pedigrí), cómo
se hereda la enfermedad. Son datos privados, caso por caso, que no están en
bases públicas.

Este proyecto plantea la pregunta contraria: ¿se puede hacer algo útil
usando solo lo que ya es público y agregado (no de un paciente concreto,
sino estadísticas y anotaciones de millones de variantes)? Las fuentes que
usa son:

- **ClinVar**: las variantes y sus clasificaciones clínicas.
- **gnomAD**: qué tan frecuente es cada variante en la población general (si
  algo es muy común en gente sana, probablemente no cause una enfermedad
  grave).
- **CADD, REVEL, AlphaMissense**: predictores computacionales que ya estiman,
  a partir de propiedades del ADN o de la proteína, cuán dañina podría ser
  una variante.

## La idea central: usar el propio calendario de ClinVar como maestro

Aquí está el giro más original del proyecto. ClinVar publica **versiones
(releases) fechadas**: cada cierto tiempo saca una foto actualizada de su
base de datos. Eso permite coger, por ejemplo, la versión de hace un año y
la versión de hoy, y comparar: ¿qué variantes que antes eran "VUS"
(inciertas) ahora ya tienen una clasificación clara (patogénica o benigna)?

Esa comparación entre dos fotos temporales genera automáticamente una especie
de examen con respuestas: variantes que ya se sabe que pasaron de "inciertas"
a "resueltas", y en qué se resolvieron. Eso es información muy valiosa para
entrenar un modelo de machine learning, porque en vez de depender de que un
humano etiquete a mano miles de casos, el propio histórico de publicaciones
de ClinVar sirve como fuente de verdad para el entrenamiento.

Con esos datos se entrena un **modelo de reclasificación**:
dado un VUS actual, ¿qué probabilidad tiene de ser reclasificado (resuelto)
próximamente? Esto es distinto de decir "es patogénica o benigna" — es más
bien decir "esta variante tiene pinta de que pronto se va a poder resolver,
hay que priorizarla para revisión".

## Las tres piezas de modelado

El sistema no usa un único modelo, sino tres complementarios:

1. **Modelo de patogenicidad**: estima si una variante es dañina o no (la
   pregunta clásica). Resultado: PR-AUC de 0,968, una métrica exigente en
   este contexto (con clases desbalanceadas, pocas variantes patogénicas
   frente a muchas benignas), y ese valor es muy alto: el modelo distingue
   muy bien.

2. **Modelo de potencial de reclasificación**: el que se entrena con el
   truco de las dos fechas de ClinVar. Estima qué VUS es más probable que se
   resuelva pronto. Resultado: ROC-AUC de 0,734, más modesto, lo cual tiene
   sentido porque predecir "qué se va a reclasificar en el futuro" es un
   problema mucho más difícil e incierto que "es dañino o no".

3. **Objetivo de ranking (NDCG@k)**: en vez de dar una etiqueta a cada
   variante por separado, este componente ordena una lista de VUS de mayor
   a menor prioridad de revisión. NDCG (Normalized Discounted Cumulative
   Gain) es una métrica típica de sistemas de recomendación y buscadores:
   mide si las cosas más importantes quedan arriba de la lista. Un
   NDCG@10 de 0,78 significa que, mirando el top 10 de la lista que genera
   el sistema, el orden es bastante bueno comparado con el orden ideal.

En conjunto: el modelo de patogenicidad da una probabilidad de daño, el de
reclasificación da una probabilidad de "esto se va a resolver pronto", y el
ranking combina esas señales para decirle a un experto humano por dónde
empezar a revisar.

## La explicabilidad: no es una caja negra

Un modelo que dice "prioriza esta variante" sin explicar por qué es poco
útil y poco fiable en medicina. Por eso el proyecto usa **SHAP**, una
técnica estándar de explicabilidad en IA que descompone la predicción de un
modelo en la contribución de cada variable de entrada (por ejemplo: "esta
variante es rara en la población", "el predictor CADD la puntúa muy alto").

Esas contribuciones SHAP se traducen a un lenguaje inspirado en los
**criterios ACMG/AMP**, las reglas oficiales que usan los genetistas
clínicos para justificar una clasificación (criterios con nombres como PM2,
PP3, etc.). Es decir, el sistema no solo dice "prioriza esto", sino que lo
argumenta en el mismo vocabulario que usaría un genetista humano.

Importante: el propio proyecto es explícito en que esto es una **heurística
de apoyo**, no una clasificación certificada. No sustituye al experto, le da
un punto de partida razonado para decidir más rápido.

## La plataforma completa

El proyecto no se queda en "un modelo entrenado en un notebook". Construye
una plataforma con las piezas típicas de un sistema de **MLOps** (las
prácticas de ingeniería para llevar modelos de machine learning a producción
de forma sostenible):

- **Ingesta**: proceso automático para traer los datos de ClinVar, gnomAD,
  CADD, REVEL, AlphaMissense.
- **Anotación**: cruzar cada variante con toda esa información adicional.
- **Entrenamiento con registro automático**: cuando se entrena un modelo,
  queda un registro (versión, métricas, parámetros) para poder auditar y
  reproducir resultados — evita el problema típico de "no se sabe qué
  versión del modelo está en producción".
- **Servicio de inferencia REST**: una API a la que se le puede preguntar por
  una variante y devuelve la predicción, para que otras aplicaciones la
  puedan consumir.
- **Dashboard interactivo con informes automáticos por VUS**: una interfaz
  visual donde un experto puede ver las variantes priorizadas y generar un
  informe legible de cada una.
- **Monitorización de concept drift**: "concept drift" es cuando el mundo
  cambia y el modelo entrenado con datos viejos deja de reflejar la realidad
  (por ejemplo, si cambian los criterios de clasificación o llega mucha más
  evidencia nueva). El sistema vigila esto.
- **Reentrenamiento gobernado por un responsable humano**: cuando se detecta
  que el modelo se ha quedado desactualizado, no se reentrena solo de forma
  automática y ciega — un responsable humano decide y aprueba el
  reentrenamiento. Esto es clave en un contexto clínico, donde no conviene
  que un sistema cambie su comportamiento sin supervisión.

## La validación

79 pruebas automatizadas en verde significa que el software tiene una
batería de tests que comprueban que cada pieza (ingesta, anotación,
modelos, API, dashboard) funciona correctamente, y todas pasan. Esto da
confianza de que la plataforma es robusta, no solo un prototipo frágil.

## La conclusión, en una frase

La aportación principal no es "un modelo que predice patogenicidad" (eso ya
existe). Es una idea más sutil: **el simple hecho de que ClinVar publique
versiones fechadas a lo largo del tiempo es, en sí mismo, una fuente de
datos de entrenamiento supervisado** — porque comparar dos fotos temporales
revela automáticamente qué variantes se resolvieron y en qué se resolvieron,
sin necesidad de que nadie las etiquete a mano expresamente para este
propósito. Hasta ahora, ese historial de versiones se usaba solo para
vigilar cambios a posteriori (monitorización); este proyecto demuestra que
también sirve, de forma proactiva, para entrenar un modelo que anticipe qué
se va a resolver antes de que ocurra.
