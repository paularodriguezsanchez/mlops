"""Registro del run de MLflow de cada etapa de la ejecución canónica.

`capture_provenance` tomaba, por experimento, el run más reciente del backend de
seguimiento. Ese backend es una base mutable y compartida: la suite de pruebas, una
reejecución parcial o un experimento suelto añaden runs posteriores, y la procedencia
acababa apuntando a ejecuciones que no eran la citada. El síntoma era silencioso, que
es lo peor que puede tener un registro de procedencia.

Aquí cada etapa deja constancia de su propio `run_id` en el momento de ejecutarse. La
procedencia lee ese registro en vez de adivinar, así que solo cambia cuando la etapa
correspondiente se vuelve a ejecutar de verdad.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.config import PROJECT_ROOT


def registry_path() -> Path:
    """Ruta del registro, resuelta en cada llamada.

    No como constante de módulo: fijarla al importar hacía que un test que
    sustituye `PROJECT_ROOT` siguiera escribiendo en el registro real del
    repositorio. Es el mismo fallo de configuración diferida que el proyecto ya
    documenta para `load_config`.
    """
    return PROJECT_ROOT / "reports" / "canonical_runs.json"


def record_run(stage: str, run_id: str, **extra) -> None:
    """Anota el run de una etapa, conservando las demás."""
    path = registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {}
    data[stage] = {
        "run_id": run_id,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
        **extra,
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_runs() -> dict:
    """Runs registrados por las etapas ya ejecutadas."""
    path = registry_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
