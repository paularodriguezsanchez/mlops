# Auditoría de dependencias

Estado de `pip-audit` sobre `requirements.txt`. Este documento existe porque
"auditoría limpia" no era cierto: la CI ejecutaba `pip-audit` con
`continue-on-error: true`, de modo que el paso aparecía en verde aunque el
comando terminase con código 1 y vulnerabilidades reales sin revisar.

## Hallazgos y decisión

Auditoría de partida, sobre `mlflow==3.14.0`: **9 vulnerabilidades en 4 paquetes**.

| Paquete | Versión auditada | Identificadores | Fix publicado | Decisión |
|---|---|---|---|---|
| `mlflow` | 3.14.0 | PYSEC-2026-3687, GHSA-gqvg-gmmx-x4hm, CVE-2026-71211 | 3.15.0 / 3.16.0 | **Actualizado** a 3.16.0 |
| `cryptography` | 48.0.1 | PYSEC-2026-3552, -3553, -3554 | 49.0.0 / 50.0.0 | **Pin** `>=50.0.0,<51` |
| `diskcache` | 5.6.3 | PYSEC-2026-2447 | sin fix publicado | Excepción documentada |
| `nltk` | 3.10.3 | PYSEC-2026-3740 | sin fix publicado | Excepción documentada |

El orden de las dos primeras filas no es independiente: `mlflow 3.15.2` declara
`cryptography<50`, de modo que la serie 50 —la única que corrige
PYSEC-2026-3552— exige subir a `mlflow 3.16.0`, que ya declara `<51`. Subir
`mlflow` no fue, por tanto, una preferencia por la última versión, sino la
condición para poder cerrar las vulnerabilidades de `cryptography`.

`cryptography`, `diskcache` y `nltk` son dependencias transitivas: la primera la
arrastran `mlflow`, `evidently` y `google-auth`; `diskcache` viene de
`dvc-data` y `nltk` de `evidently`. Ninguna se usa de forma directa en el código
del proyecto.

Resultado tras las correcciones: `No known vulnerabilities found, 2 ignored`.

## Excepciones: por qué no bloquean

Las dos vulnerabilidades sin corrección publicada se aceptan de forma
explícita, no por omisión, y con esta evaluación de exposición:

* El sistema se ejecuta en local, sobre datos públicos agregados, sin datos de
  paciente y sin exponer ningún puerto fuera de `127.0.0.1` (ver `compose.yaml`).
* `diskcache` y `nltk` se cargan como dependencias de `dvc-data` y `evidently`;
  el proyecto no expone ninguna de sus superficies a entrada no confiable.
* `mlflow` corre como servidor de seguimiento local sin autenticación, y por eso
  mismo su puerto está limitado a loopback.

Subir `mlflow` de 3.14 a 3.16 cambia el entorno de ejecución, así que obliga a
reinstalar dependencias y a repetir la suite y el pipeline antes de citar
cualquier cifra: las métricas de una ejecución con 3.14 no son atribuibles a
este entorno.

Esa evaluación acota el riesgo **en este despliegue**; no afirma que las
vulnerabilidades sean inocuas en general. Un despliegue expuesto a red exigiría
revisarlas de nuevo.

## Comportamiento en integración continua

`pip-audit` es **bloqueante**, con las tres excepciones anteriores declaradas por
identificador mediante `--ignore-vuln`. Así el paso solo pasa a verde si no
aparece ninguna vulnerabilidad nueva, y cualquier hallazgo distinto de los tres
documentados tumba el pipeline y obliga a revisarlo.

## Reproducir

```bash
pip-audit -r requirements.txt
```

Última ejecución registrada: 2026-09-04, con los resultados de la tabla anterior.
La fecha importa: el resultado depende del estado de las bases de datos de
vulnerabilidades en el momento de consultarlas, así que una ejecución posterior
puede encontrar hallazgos que aquí no aparecen.
