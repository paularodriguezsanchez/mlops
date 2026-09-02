"""Carga centralizada de la configuración del proyecto (config/config.yaml)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


@lru_cache(maxsize=1)
def _load_config_cached(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_config(path: str | Path | None = None) -> dict:
    """Configuración del proyecto como diccionario, cacheada.

    `CONFIG_PATH` se resuelve en cada llamada, no como valor por defecto del
    parámetro: de lo contrario queda fijado en el import y sustituirlo desde un test
    no tiene efecto, de modo que un test "aislado" seguiría leyendo la config real.
    """
    return _load_config_cached(Path(path) if path is not None else CONFIG_PATH)


load_config.cache_clear = _load_config_cached.cache_clear  # type: ignore[attr-defined]


def get_seed() -> int:
    """Semilla aleatoria global para reproducibilidad."""
    return int(load_config()["project"]["random_seed"])


def _dir(key: str) -> Path:
    """Ruta absoluta de un directorio de datos declarado en config."""
    return (PROJECT_ROOT / load_config()["data"][key]).resolve()


def raw_dir() -> Path:
    """Directorio de datos crudos (inmutable)."""
    return _dir("raw_dir")


def interim_dir() -> Path:
    """Directorio de datos intermedios (capa SILVER)."""
    return _dir("interim_dir")


def processed_dir() -> Path:
    """Directorio de datos procesados listos para modelar (capa GOLD)."""
    return _dir("processed_dir")


def chromosomes_subset() -> list[str] | None:
    """Cromosomas a los que se acota el análisis. Lista vacía o None = todos."""
    subset = load_config()["data"].get("chromosomes_subset") or None
    return [str(c) for c in subset] if subset else None


def max_variants_per_release() -> int | None:
    """Variantes por release antes de anotar: acota volumen y coste de red."""
    limit = load_config()["data"].get("max_variants_per_release")
    return int(limit) if limit else None


def max_new_variants_per_release() -> int | None:
    """Variantes nuevas a añadir en releases posteriores, además de la cohorte retenida."""
    limit = load_config()["data"].get("max_new_variants_per_release")
    return int(limit) if limit else None


def clinvar_prospective_release() -> str | None:
    """Release adicional para la validación temporal prospectiva."""
    return load_config()["data"].get("clinvar_prospective_release")


def annotation_source() -> str:
    """Fuente de features de anotación: 'synthetic' (ADR 005) o 'multi_source' (ADR 007/B5)."""
    return str(load_config()["data"].get("annotation_source", "synthetic"))
