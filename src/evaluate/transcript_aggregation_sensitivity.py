"""Sensibilidad de la agregación por transcrito (media vs. máximo).

Motivo: cuando myvariant.info devuelve
una variante mapeada a varios transcritos/isoformas, `multi_source.py`
promedia los valores (`_mean_or_none`) sin justificar por qué la media es la
agregación adecuada ni analizar la sensibilidad frente a otras estrategias
(p. ej. el score más extremo, más conservador desde un punto de vista de
priorización clínica). Una revisión anterior lo dejó explícitamente reconocido como no abordado;
este módulo lo cuantifica.

Este módulo SÍ requiere red (re-consulta myvariant.info para los mismos
IDs de variante ya anotados, esta vez conservando el valor MÁXIMO por
transcrito en vez de la media, para los 7 campos que pueden venir como
lista). Alcance deliberadamente acotado: un análisis de sensibilidad por
validación cruzada sobre la población YA etiquetada (train+test combinados),
no una sustitución del protocolo de evaluación canónico del modelo de patogenicidad -- los
resultados de esta memoria siguen siendo los de la agregación por media.

Uso:
    python -m src.evaluate.transcript_aggregation_sensitivity
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from src.annotate.multi_source import (
    _BATCH_SIZE,
    _MYVARIANT_ASSEMBLY,
    _MYVARIANT_FIELDS,
    _MYVARIANT_URL,
    _TIMEOUT,
    _variant_id,
)
from src.config import PROJECT_ROOT, get_seed, processed_dir
from src.features.preprocess import FEATURE_COLUMNS, build_preprocessor

_LIST_FIELDS = {
    "cadd_phred": ("cadd", "phred"),
    "sift_score": ("sift", "score"),
    "polyphen_score": ("polyphen2", "hdiv", "score"),
    "gerp_rs": ("gerp++", "rs"),
    "phylop": ("phylop", "100way_vertebrate", "score"),
    "revel_score": ("revel", "score"),
    "alphamissense_score": ("alphamissense", "score"),
}


def _dig(d: dict, *path):
    for p in path:
        d = (d or {}).get(p)
    return d


def _max_or_none(x):
    if x is None:
        return None
    if isinstance(x, list):
        vals = [v for v in x if v is not None]
        return float(np.max(vals)) if vals else None
    return float(x)


def _parse_hit_both(hit: dict) -> dict:
    """Igual que `multi_source._parse_hit` pero devuelve media Y máximo,
    más si el valor crudo era una lista (multi-transcrito) para cada campo."""
    if hit.get("notfound"):
        return {}
    dbnsfp = hit.get("dbnsfp", {}) or {}
    out = {}
    for col, path in _LIST_FIELDS.items():
        raw = _dig(dbnsfp, *path)
        out[f"{col}_mean"] = float(np.mean(raw)) if isinstance(raw, list) and raw else (
            float(raw) if raw is not None else None)
        out[f"{col}_max"] = _max_or_none(raw)
        out[f"{col}_is_list"] = isinstance(raw, list)
    return out


def _fetch_both_aggregations(keys: pd.DataFrame) -> pd.DataFrame:
    ids = [_variant_id(*row) for row in keys.itertuples(index=False)]
    records = []
    for start in range(0, len(ids), _BATCH_SIZE):
        batch_ids = ids[start:start + _BATCH_SIZE]
        data = urllib.parse.urlencode({
            "ids": ",".join(batch_ids), "fields": _MYVARIANT_FIELDS,
            "assembly": _MYVARIANT_ASSEMBLY,
        }).encode()
        req = urllib.request.Request(  # noqa: S310
            _MYVARIANT_URL, data=data, headers={"User-Agent": "tfm-mlops-variantes"})
        try:
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:  # noqa: S310
                hits = json.loads(resp.read())
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            print(f"  [aviso] lote {start} fallo: {exc}", file=sys.stderr)
            hits = [{} for _ in batch_ids]
        if isinstance(hits, dict):
            hits = [hits]
        batch_keys = keys.iloc[start:start + _BATCH_SIZE].itertuples(index=False)
        for row, hit in zip(batch_keys, hits, strict=False):
            rec = {"chrom": row.chrom, "pos": row.pos, "ref": row.ref, "alt": row.alt}
            rec.update(_parse_hit_both(hit if isinstance(hit, dict) else {}))
            records.append(rec)
        print(f"  lote {start // _BATCH_SIZE + 1}/{-(-len(ids) // _BATCH_SIZE)} OK")
    return pd.DataFrame.from_records(records)


def run() -> dict:
    seed = get_seed()
    train_df = pd.read_parquet(processed_dir() / "train.parquet")
    test_df = pd.read_parquet(processed_dir() / "test.parquet")
    labeled = pd.concat([train_df, test_df], ignore_index=True)
    keys = labeled[["chrom", "pos", "ref", "alt"]].drop_duplicates().reset_index(drop=True)
    print(f"Re-consultando myvariant.info para {len(keys)} variantes ya etiquetadas "
          "(media Y máximo por transcrito)...")
    both = _fetch_both_aggregations(keys)

    list_prevalence = {
        col: round(100 * both[f"{col}_is_list"].mean(), 2) for col in _LIST_FIELDS
    }

    merged = labeled.merge(both, on=["chrom", "pos", "ref", "alt"], how="inner")
    print(f"Variantes con re-anotación encontrada: {len(merged)}/{len(labeled)}")

    diffs = {}
    for col in _LIST_FIELDS:
        sub = merged[merged[f"{col}_is_list"]]
        if len(sub):
            abs_diff = (sub[f"{col}_max"] - sub[f"{col}_mean"]).abs()
            feature_std = merged[f"{col}_mean"].std()
            diffs[col] = {
                "n_multi_transcrito": int(len(sub)),
                "diff_media_absoluta": round(float(abs_diff.mean()), 4),
                "diff_maxima": round(float(abs_diff.max()), 4),
                "relativa_a_std_del_feature": round(
                    float(abs_diff.mean() / feature_std), 4) if feature_std else None,
            }

    # Comparación por validación cruzada: mismo preprocesador/algoritmo,
    # mismas filas, features de MEDIA (ya en el dataset canónico) vs MÁXIMO
    # (recién obtenido). Estratificado, 5 folds, misma semilla en ambos casos
    # -- comparación pareada, no dos muestreos independientes.
    y = merged["label"].astype(int)
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    X_mean = merged[FEATURE_COLUMNS].copy()
    pipe_mean = Pipeline([("pre", build_preprocessor()), ("clf", clf)])
    cv_mean = cross_val_score(pipe_mean, X_mean, y, cv=cv, scoring="average_precision", n_jobs=1)

    X_max = X_mean.copy()
    for col in _LIST_FIELDS:
        if col in X_max.columns:
            X_max[col] = merged[f"{col}_max"]
    pipe_max = Pipeline([("pre", build_preprocessor()), ("clf", clf)])
    cv_max = cross_val_score(pipe_max, X_max, y, cv=cv, scoring="average_precision", n_jobs=1)

    result = {
        "n_variantes_analizadas": len(merged),
        "prevalencia_multi_transcrito_pct": list_prevalence,
        "diferencia_media_vs_maximo": diffs,
        "cv_pr_auc_media_mean": round(float(cv_mean.mean()), 4),
        "cv_pr_auc_media_std": round(float(cv_mean.std()), 4),
        "cv_pr_auc_maximo_mean": round(float(cv_max.mean()), 4),
        "cv_pr_auc_maximo_std": round(float(cv_max.std()), 4),
        "diferencia_cv_pr_auc": round(float(cv_max.mean() - cv_mean.mean()), 4),
    }

    out_dir = PROJECT_ROOT / "reports" / "training"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "campo": col, **vals,
        "prevalencia_multi_transcrito_pct": list_prevalence[col],
    } for col, vals in diffs.items()]).to_csv(
        out_dir / "transcript_aggregation_sensitivity.csv", index=False)

    print(f"\nPrevalencia de multi-transcrito por campo (%): {list_prevalence}")
    print(f"Diferencia media vs máximo (entre multi-transcrito): {diffs}")
    print(f"\nCV 5-fold PR-AUC (media, dataset canónico) = "
          f"{result['cv_pr_auc_media_mean']}±{result['cv_pr_auc_media_std']}")
    print(f"CV 5-fold PR-AUC (máximo, re-anotado)       = "
          f"{result['cv_pr_auc_maximo_mean']}±{result['cv_pr_auc_maximo_std']}")
    print(f"Diferencia = {result['diferencia_cv_pr_auc']:+.4f}")
    return result


if __name__ == "__main__":
    run()
