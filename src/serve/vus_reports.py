"""Informes automáticos por VUS, basados en plantilla (ADR 007 §5).

Deliberadamente **basados en plantilla, no en generación libre de un LLM**:
en un informe que puede influir en qué VUS revisa antes un clínico, un texto
inventado (alucinado) es un riesgo real, y una plantilla con los datos reales
del modelo no lo tiene. Cada informe cita: la probabilidad de patogenicidad,
la probabilidad de reclasificación próxima (si el modelo de reclasificación
está
disponible) y la evidencia SHAP→ACMG del evidencia ACMG-símil (`src/evaluate/acmg_evidence.py`).

Uso:
    python -m src.serve.vus_reports # top 10, release test
    python -m src.serve.vus_reports --top-n 15 --split train
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow.sklearn
import pandas as pd

from src.config import PROJECT_ROOT, processed_dir
from src.evaluate.acmg_evidence import explain_variant_acmg
from src.features.preprocess import FEATURE_COLUMNS, RECLASS_FEATURE_COLUMNS
from src.serve.predictor import get_predictor
from src.serve.prioritize_vus import load_ranking_model, rank_vus

_KEY = ["chrom", "pos", "ref", "alt"]
_RECLASS_MODEL_DIR = PROJECT_ROOT / "models" / "reclassification_model"


def _load_reclass_model() -> tuple[object, bool] | tuple[None, None]:
    """Carga el modelo de reclasificación si existe; si no, degrada sin bloquear
    (igual que Evidently/ADR005).

    Devuelve `(modelo, reliable)`: `reliable` viene de `models/reclassification_model/
    metrics.json` (escrito por `train_reclass.py`, ROC AUC >= umbral, ver la revisión interna del
    proyecto). Si el fichero de métricas no existe (modelo entrenado antes de este fix),
    se asume `reliable=False` — más seguro no dar por buena una señal sin verificar.
    """
    if not _RECLASS_MODEL_DIR.exists():
        return None, None
    try:
        model = mlflow.sklearn.load_model(str(_RECLASS_MODEL_DIR))
    except Exception as exc:  # noqa: BLE001
        print(f"[aviso] modelo de reclasificación no disponible: {exc}")
        return None, None
    metrics_path = _RECLASS_MODEL_DIR / "metrics.json"
    reliable = False
    if metrics_path.exists():
        reliable = bool(json.loads(metrics_path.read_text(encoding="utf-8")).get("reliable"))
    return model, reliable


def _render_variant(row: pd.Series, evidence: list[dict], reclass_prob: float | None,
                    reclass_reliable: bool | None) -> str:
    ev_lines = "\n".join(
        f"  * **{e['acmg_tag']}** — {e['note']} "
        f"(valor={e['value']}, contribución SHAP={e['shap']:+.3f})"
        for e in evidence[:5]
    ) or "  * Sin evidencia computacional destacable por encima de los umbrales."

    if reclass_prob is None:
        reclass_line = (
            "* Probabilidad de reclasificación próxima: no disponible "
            "(ejecutar `python -m src.train.train_reclass`).\n"
        )
    elif reclass_reliable:
        reclass_line = f"* **Probabilidad de reclasificación próxima:** {reclass_prob:.1%}\n"
    else:
        reclass_line = (
            f"* **Probabilidad de reclasificación próxima:** {reclass_prob:.1%} "
            "(AVISO: modelo con señal predictiva no confirmada — ROC AUC cercano al azar, "
            "ver `docs/MODEL_CARD_RECLASSIFICATION.md`; no tratar como resultado validado)\n"
        )
    gene = row.get("gene", "gen desconocido")
    header = f"### {row['chrom']}:{row['pos']} {row['ref']}>{row['alt']} ({gene})"
    evidence_header = ("* **Evidencia computacional** "
                       "(heurística tipo ACMG/AMP, no una clasificación certificada):")

    return f"""{header}

* **Consecuencia:** {row.get('consequence', 'desconocida')}
* **Probabilidad de patogenicidad:** {row['probabilidad_patogenica']:.1%}
{reclass_line}{evidence_header}
{ev_lines}
"""


def generate_reports(split: str = "test", top_n: int = 10) -> dict:
    vus = pd.read_parquet(processed_dir() / f"vus_{split}.parquet")
    train_df = pd.read_parquet(processed_dir() / "train.parquet")

    predictor = get_predictor()
    # Mismo criterio de orden que la priorización de VUS (`prioritize_vus.py`): ranking dedicado
    # si está entrenado, si no, probabilidad de patogenicidad. Ambos
    # módulos comparten `load_ranking_model`/`rank_vus` para no divergir.
    ranking_model, ranking_preprocessor = load_ranking_model()
    ranked = rank_vus(predictor, vus, ranking_model, ranking_preprocessor).head(top_n)
    ranked = ranked.reset_index(drop=True)

    reclass_model, reclass_reliable = _load_reclass_model()
    # el modelo de reclasificación usa RECLASS_FEATURE_COLUMNS (incluye `review_stars`,
    # ADR 008), no
    # FEATURE_COLUMNS -- el pipeline guardado ya sabe seleccionar sus
    # propias columnas de entrada, basta con ofrecérselas todas disponibles.
    reclass_probs = (
        reclass_model.predict_proba(ranked[RECLASS_FEATURE_COLUMNS])[:, 1]
        if reclass_model is not None else [None] * len(ranked)
    )

    explain_cols = [*FEATURE_COLUMNS, *_KEY]
    sample, evidence = explain_variant_acmg(
        predictor.model, train_df[FEATURE_COLUMNS], ranked[explain_cols])
    # `sample` puede venir en otro orden que `ranked` (ver acmg_evidence.py):
    # se re-alinea por clave antes de renderizar.
    order = {tuple(r[c] for c in _KEY): i for i, (_, r) in enumerate(sample.iterrows())}

    sections, records = [], []
    for i, row in ranked.iterrows():
        key = tuple(row[c] for c in _KEY)
        ev = evidence[order[key]] if key in order else []
        reclass_prob = None if reclass_probs[i] is None else float(reclass_probs[i])
        sections.append(_render_variant(row, ev, reclass_prob, reclass_reliable))
        record = {
            **{c: row[c] for c in _KEY},
            "gene": row.get("gene"), "consequence": row.get("consequence"),
            "probabilidad_patogenica": float(row["probabilidad_patogenica"]),
            "probabilidad_reclasificacion": reclass_prob,
            "probabilidad_reclasificacion_fiable": (
                reclass_reliable if reclass_prob is not None else None),
            "evidencia": ev,
        }
        if ranking_model is not None:
            record["ranking_score"] = float(row["ranking_score"])
        records.append(record)

    out_dir = PROJECT_ROOT / "reports" / "serving"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"vus_reports_{split}.json"
    json_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")

    doc_path = _write_doc(sections, split, len(vus), json_path, ranking_model is not None)
    criterio = "ranking" if ranking_model is not None else "patogenicidad"
    print(f"{len(ranked)} informes de VUS generados (release {split}, orden por {criterio}) "
          f"-> {doc_path}")
    return {"n": len(ranked), "json": json_path, "doc": doc_path}


def _write_doc(
    sections: list[str], split: str, n_total: int, json_path: Path, using_ranking: bool,
) -> Path:
    orden_nota = (
        "Las variantes se ordenan por el score del modelo de ranking dedicado "
        "(LightGBM `lambdarank`) — ver `src/serve/prioritize_vus.py`."
        if using_ranking else
        "Las variantes se ordenan por la probabilidad de patogenicidad : el "
        "modelo de ranking no está entrenado todavía (`make train-ranking`)."
    )
    doc = PROJECT_ROOT / "docs" / f"vus_informes_{split}.md"
    body = "\n".join(sections)
    doc.write_text(f"""# Informes de VUS priorizadas: release {split}

Generados por plantilla (no por un LLM libre) a partir de los modelos y la
explicabilidad ya construidos: probabilidad de patogenicidad
(`src/serve/predictor.py`), probabilidad de reclasificación próxima
(`src, train, train_reclass.py`) y evidencia SHAP traducida a lenguaje tipo
ACMG/AMP (`src, evaluate, acmg_evidence.py`). **No es una clasificación
ACMG certificada** ni sustituye la curación clínica experta. {orden_nota}

{n_total} VUS reservadas en total en esta release. Informes de las
{len(sections)} priorizadas primero:

{body}

Datos estructurados (para el dashboard): `{json_path.relative_to(PROJECT_ROOT).as_posix}`.

## Reproducir
```bash
python -m src.serve.vus_reports --split {split} --top-n {len(sections)}
```
""", encoding="utf-8")
    return doc


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Genera informes de VUS priorizadas .")
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--top-n", type=int, default=10)
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    generate_reports(split=args.split, top_n=args.top_n)
