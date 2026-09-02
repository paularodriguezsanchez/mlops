"""Anotador de variantes para el servicio de inferencia.

En producción, anotar una variante nueva implica ejecutar VEP (consecuencia
funcional) + dbNSFP/gnomAD (scores). En la Fase I local se reutiliza la capa
SILVER ya anotada como **almacén de anotación**: dada una variante
`(chrom, pos, ref, alt)` devuelve sus features de entrada al modelo.

Limitación documentada: solo resuelve variantes presentes en el almacén; una
variante desconocida devolvería "no anotable" (en producción se llamaría a VEP).
NO expone la etiqueta clínica (`clnsig`): eso es el target, no una feature.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.config import interim_dir, load_config
from src.features.preprocess import FEATURE_COLUMNS

_KEY = ["chrom", "pos", "ref", "alt"]


class VariantAnnotator:
    """Almacén de anotación en memoria indexado por (chrom, pos, ref, alt)."""

    def __init__(self, store: pd.DataFrame):
        self._store = store.set_index(_KEY)

    @classmethod
    def from_release(cls, release: str | None = None) -> VariantAnnotator:
        cfg = load_config()
        release = release or cfg["data"]["clinvar_test_release"]
        path = interim_dir() / f"annotated_{release}.parquet"
        df = pd.read_parquet(path)
        keep = _KEY + ["gene"] + FEATURE_COLUMNS
        return cls(df[keep].drop_duplicates(_KEY))

    def annotate(self, chrom: str, pos: int, ref: str, alt: str) -> dict | None:
        """Devuelve features de la variante o None si no está en el almacén."""
        key = (str(chrom), int(pos), str(ref), str(alt))
        if key not in self._store.index:
            return None
        row = self._store.loc[key]
        return {"gene": row["gene"], **{c: row[c] for c in FEATURE_COLUMNS}}

    def __len__(self) -> int:
        return len(self._store)


@lru_cache(maxsize=1)
def get_annotator() -> VariantAnnotator:
    """Anotador cacheado (se carga una vez por proceso)."""
    return VariantAnnotator.from_release()


def sample_documented_variants(n_per_class: int = 5, release: str | None = None,
                               seed: int | None = None) -> pd.DataFrame:
    """Selecciona variantes etiquetadas (mitad patogénicas / mitad benignas) para
    documentar el servicio (OE4). Prioriza variantes NO vistas en entrenamiento
    (evaluación honesta). Incluye clnsig real solo para el informe."""
    from src.config import get_seed, processed_dir
    cfg = load_config()
    seed = get_seed() if seed is None else seed
    release = release or cfg["data"]["clinvar_test_release"]
    df = pd.read_parquet(interim_dir() / f"annotated_{release}.parquet")
    # Excluir variantes ya presentes en el conjunto de entrenamiento.
    train = pd.read_parquet(processed_dir() / "train.parquet")
    train_keys = set(map(tuple, train[_KEY].to_numpy()))
    unseen = ~pd.Series(list(map(tuple, df[_KEY].to_numpy())), index=df.index).isin(train_keys)
    df = df[unseen]
    pos = df[df["clnsig"].isin(["Pathogenic", "Likely_pathogenic"])].sample(
        n_per_class, random_state=seed)
    neg = df[df["clnsig"].isin(["Benign", "Likely_benign"])].sample(
        n_per_class, random_state=seed)
    return pd.concat([pos, neg]).reset_index(drop=True)


def _default_store_exists() -> bool:
    cfg = load_config()
    p = Path(interim_dir()) / f"annotated_{cfg['data']['clinvar_test_release']}.parquet"
    return p.exists()
