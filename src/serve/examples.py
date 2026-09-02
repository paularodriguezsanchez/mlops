"""Documenta al menos diez variantes de prueba del servicio de inferencia.

Escribe la entrada y la salida completas en `reports/serving/example_variants.json`
y una tabla legible, etiqueta real frente a predicción, en `docs/serving_examples.md`.
"""
from __future__ import annotations

import json

import pandas as pd

from src.config import PROJECT_ROOT
from src.serve.annotator import sample_documented_variants
from src.serve.predictor import get_predictor


def run(n_per_class: int = 5) -> dict:
    predictor = get_predictor()
    sample = sample_documented_variants(n_per_class=n_per_class)
    rows, records = [], []
    for _, v in sample.iterrows():
        res = predictor.predict(v["chrom"], int(v["pos"]), v["ref"], v["alt"])
        real = "Patogénica" if v["clnsig"] in ("Pathogenic", "Likely_pathogenic") else "Benigna"
        ok = "✓" if res.get("prediction") == real else "✗"
        rows.append({
            "variante": f"{v['chrom']}-{int(v['pos'])}-{v['ref']}-{v['alt']}",
            "gen": v["gene"], "consecuencia": v["consequence"],
            "clnsig_real": v["clnsig"], "clase_real": real,
            "prediccion": res.get("prediction"),
            "prob_patogenica": res.get("probability_pathogenic"), "acierto": ok,
        })
        records.append({"input": {"chrom": v["chrom"], "pos": int(v["pos"]),
                                  "ref": v["ref"], "alt": v["alt"]},
                        "clnsig_real": v["clnsig"], "output": res})

    rep_dir = PROJECT_ROOT / "reports" / "serving"
    rep_dir.mkdir(parents=True, exist_ok=True)
    (rep_dir / "example_variants.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")

    table = pd.DataFrame(rows)
    n_ok = (table["acierto"] == "✓").sum()
    _write_md(table, n_ok, len(table))
    print(f"Variantes documentadas: {len(table)} | aciertos: {n_ok}/{len(table)}")
    return {"table": table, "n_ok": int(n_ok), "n": len(table)}


def _write_md(table: pd.DataFrame, n_ok: int, n: int) -> None:
    doc = PROJECT_ROOT / "docs" / "serving_examples.md"
    md = table.to_markdown(index=False)
    doc.write_text(f"""# Servicio de inferencia: variantes de prueba

El servicio recibe una variante `(chrom, pos, ref, alt)`, la anota -consecuencia
funcional, scores in silico y frecuencia poblacional- y devuelve la predicción de
patogenicidad con su probabilidad. Estas {n} variantes, mitad patogénicas y mitad
benignas según ClinVar, no se vieron en entrenamiento; aciertos: **{n_ok}/{n}**.

Es una prueba de integración extremo a extremo del servicio, no una estimación de
rendimiento: para eso está el PR AUC sobre el conjunto de evaluación completo.

{md}

`clnsig_real` es la etiqueta de referencia y no se pasa al modelo; sirve solo para
verificar el acierto. La respuesta completa de cada variante está en
`reports/serving/example_variants.json`.

## Reproducir
```bash
make serve
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' \\
     -d '{{"chrom":"3","pos":11825913,"ref":"G","alt":"T"}}'
```
""", encoding="utf-8")


if __name__ == "__main__":
    run()
