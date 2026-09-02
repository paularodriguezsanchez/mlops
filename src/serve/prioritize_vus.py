"""Priorización de VUS por riesgo estimado de patogenicidad (ADR 006).

Implementa de forma concreta la idea registrada en ADR 006 / la priorización de VUS: el modelo NO
emite un veredicto clínico ni sustituye la curación manual; puntúa cada VUS
reservada (variante sin significado clínico resuelto en ClinVar) y esta tabla
las ordena de mayor a menor prioridad. El objetivo es dirigir y agilizar la
investigación posterior (el revisor prioriza las de arriba), no reemplazarla.

Criterio de orden: si el modelo de ranking (`src, train, train_ranking.py`, LightGBM `lambdarank`)
está entrenado, el orden lo decide su score — es el
entregable real del proyecto (un orden de prioridad, no solo una probabilidad
de clasificación aislada, ver ADR 007 §5.3) y evita el desajuste de que la priorización de VUS
muestre un orden distinto del que el objetivo de ranking fue entrenado y evaluado (NDCG@k) para
producir. Si el objetivo de ranking no está entrenado todavía (`make train-ranking`), se cae de
forma explícita a ordenar por la probabilidad de patogenicidad del modelo de patogenicidad — la
probabilidad de patogenicidad se calcula y se muestra siempre, se use o no
para ordenar.

Uso:
    python -m src.serve.prioritize_vus # release de test (VUS más recientes)
    python -m src.serve.prioritize_vus --split train --top-n 30
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT, processed_dir
from src.features.preprocess import FEATURE_COLUMNS
from src.serve.predictor import VariantPredictor, get_predictor

_REPORT_COLUMNS = ["chrom", "pos", "ref", "alt", "gene", "consequence", "clnsig"]
_RANKING_MODEL_DIR = PROJECT_ROOT / "models" / "ranking_model"


def load_ranking_model() -> tuple[object, object] | tuple[None, None]:
    """Carga el booster de ranking y el preprocesador con el que se entrenó.

    Degrada a `(None, None)` si `models/ranking_model` no existe o no se pudo
    cargar (mismo patrón de degradación explícita que el resto del proyecto,
    ver ADR 005): `rank_vus` cae entonces a ordenar por la probabilidad de
    patogenicidad del modelo de patogenicidad. Compartida por `prioritize_vus.py` y
    `vus_reports.py` para que ambos usen siempre el mismo criterio.
    """
    booster_path = _RANKING_MODEL_DIR / "lambdarank.txt"
    preprocessor_path = _RANKING_MODEL_DIR / "preprocessor.joblib"
    if not booster_path.exists() or not preprocessor_path.exists():
        return None, None
    try:
        import joblib
        import lightgbm as lgb

        booster = lgb.Booster(model_file=str(booster_path))
        preprocessor = joblib.load(preprocessor_path)
    except Exception as exc:  # noqa: BLE001
        print("[aviso] modelo de ranking no disponible, se usa el modelo de "
              f"patogenicidad para ordenar: {exc}")
        return None, None
    return booster, preprocessor


def rank_vus(
    predictor: VariantPredictor,
    vus: pd.DataFrame,
    ranking_model: object | None = None,
    ranking_preprocessor: object | None = None,
) -> pd.DataFrame:
    """Puntúa y ordena VUS por prioridad (mayor primero). Pura, sin I/O.

    Con `ranking_model`/`ranking_preprocessor` (ver `load_ranking_model`),
    ordena por `ranking_score`. Sin ellos (por defecto), ordena por
    `probabilidad_patogenica`, como antes de integrar el objetivo de ranking.
    """
    probs = predictor.predict_batch(vus)
    out = vus.assign(probabilidad_patogenica=probs)

    if ranking_model is not None and ranking_preprocessor is not None:
        X_t = ranking_preprocessor.transform(vus[FEATURE_COLUMNS])
        out = out.assign(ranking_score=ranking_model.predict(X_t))
        sort_col = "ranking_score"
    else:
        sort_col = "probabilidad_patogenica"

    return out.sort_values(sort_col, ascending=False).reset_index(drop=True)


def run(split: str = "test", top_n: int = 20) -> dict:
    vus = pd.read_parquet(processed_dir() / f"vus_{split}.parquet")
    predictor = get_predictor()
    ranking_model, ranking_preprocessor = load_ranking_model()
    ranked = rank_vus(predictor, vus, ranking_model, ranking_preprocessor)
    using_ranking = ranking_model is not None

    extra_cols = ["ranking_score"] if using_ranking else []
    cols = [c for c in _REPORT_COLUMNS if c in ranked.columns] + [
        "probabilidad_patogenica", *extra_cols]
    out_dir = PROJECT_ROOT / "reports" / "serving"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"vus_priorizadas_{split}.csv"
    ranked[cols].to_csv(csv_path, index=False)

    _write_report(ranked[cols].head(top_n), split, len(ranked), csv_path, using_ranking)
    criterio = "score de ranking " if using_ranking else "probabilidad de patogenicidad "
    print(f"{len(ranked)} VUS priorizadas (release {split}) -> {csv_path} "
          f"(top {min(top_n, len(ranked))} en docs/vus_priorizadas.md, orden por {criterio})")
    return {"n": len(ranked), "csv": csv_path, "using_ranking": using_ranking}


def _write_report(
    top: pd.DataFrame, split: str, n_total: int, csv_path: Path, using_ranking: bool,
) -> None:
    criterio = (
        "el score del modelo de ranking dedicado (LightGBM `lambdarank`, entrenado "
        "para optimizar directamente el orden de prioridad y evaluado con NDCG@k — ver "
        "ADR 007 §5.3, `reports, training, ranking_metrics.csv`). La probabilidad de "
        "patogenicidad del modelo de patogenicidad se muestra también, como referencia adicional."
        if using_ranking else
        "la probabilidad de patogenicidad (el modelo de ranking no está "
        "entrenado todavía; ejecuta `make train-ranking` para que la "
        "priorización use ese criterio)."
    )
    doc = PROJECT_ROOT / "docs" / "vus_priorizadas.md"
    doc.write_text(f"""# Priorización de VUS por riesgo estimado: release {split}

**Esto no es un veredicto clínico ni sustituye la curación manual.** Es una
ayuda para dirigir y agilizar la investigación posterior que un revisor
tendría que hacer de todos modos, apoyándose en el conocimiento previo de
ClinVar/gnomAD/CADD/REVEL/AlphaMissense ya construido. El criterio de
priorización: revisar antes las de arriba, ordenadas por {criterio}

{n_total} VUS reservadas en total. Top {len(top)} por prioridad:

{top.to_markdown(index=False)}

Lista completa (todas las VUS, ordenadas): `{csv_path.relative_to(PROJECT_ROOT).as_posix}`.

## Reproducir
```bash
make prioritize # release de test, top 20
python -m src.serve.prioritize_vus --split train --top-n 30
```
""", encoding="utf-8")


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Prioriza VUS por riesgo estimado.")
    p.add_argument("--split", default="test", choices=["train", "test"],
                   help="Qué conjunto de VUS reservadas priorizar (por defecto: test).")
    p.add_argument("--top-n", type=int, default=20,
                   help="Cuántas variantes mostrar en el informe (por defecto: 20).")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    run(split=args.split, top_n=args.top_n)
