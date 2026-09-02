"""Inferencia pura: variante, anotación y predicción de patogenicidad.

Combina el pipeline autocontenido de `models/best_model` con el almacén de
anotación. Lo reutilizan el servicio REST, la priorización y los tests, sin
necesidad de levantar el servidor.
"""
from __future__ import annotations

from functools import lru_cache

import mlflow.sklearn
import pandas as pd

from src.config import PROJECT_ROOT
from src.features.preprocess import FEATURE_COLUMNS
from src.serve.annotator import VariantAnnotator, get_annotator

_MODEL_DIR = PROJECT_ROOT / "models" / "best_model"
_POS_LABEL = "Patogénica"
_NEG_LABEL = "Benigna"


class VariantPredictor:
    def __init__(self, model, annotator: VariantAnnotator, threshold: float = 0.5):
        self._model = model
        self._annotator = annotator
        self._threshold = threshold

    @property
    def model(self):
        """El pipeline sklearn subyacente (para explicabilidad)."""
        return self._model

    def predict(self, chrom: str, pos: int, ref: str, alt: str) -> dict:
        """Anota y predice. Devuelve un dict serializable a JSON."""
        annotation = self._annotator.annotate(chrom, pos, ref, alt)
        variant = {"chrom": str(chrom), "pos": int(pos), "ref": ref, "alt": alt}
        if annotation is None:
            return {"variant": variant, "annotated": False,
                    "message": "Variante no anotable en el almacén local "
                               "(en producción se anotaría con VEP+dbNSFP)."}
        X = pd.DataFrame([{c: annotation[c] for c in FEATURE_COLUMNS}])
        prob = float(self._model.predict_proba(X)[0, 1])
        pred = _POS_LABEL if prob >= self._threshold else _NEG_LABEL
        return {
            "variant": variant,
            "annotated": True,
            "gene": annotation["gene"],
            "consequence": annotation["consequence"],
            "prediction": pred,
            "probability_pathogenic": round(prob, 4),
        }

    def predict_batch(self, X: pd.DataFrame):
        """Probabilidad de patogenicidad para un lote ya anotado (uso: priorización de VUS).

        A diferencia de `predict`, no pasa por el anotador: `X` debe traer ya
        las columnas de `FEATURE_COLUMNS` (p. ej. desde la capa SILVER/GOLD).
        """
        return self._model.predict_proba(X[FEATURE_COLUMNS])[:, 1]


@lru_cache(maxsize=1)
def get_predictor() -> VariantPredictor:
    """Predictor cacheado: carga el modelo una vez por proceso."""
    model = mlflow.sklearn.load_model(str(_MODEL_DIR))
    return VariantPredictor(model, get_annotator())
