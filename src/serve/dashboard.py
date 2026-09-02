"""Dashboard interactivo de exploración de VUS priorizadas (ADR 007 §5).

Lee los informes ya generados por el generador de informes por VUS (`src/serve/vus_reports.py` →
`reports/serving/vus_reports_{split}.json`); no recalcula SHAP en cada
petición HTTP (sería demasiado lento para una página web). Si el informe no
existe todavía, muestra cómo generarlo en vez de un error 500 (mismo patrón
de degradación explícita que el resto del proyecto).

Ruta: GET /dashboard, GET /dashboard/<split> (train|test).
"""
from __future__ import annotations

import json

from flask import Blueprint, abort, render_template_string

from src.config import PROJECT_ROOT

dashboard_bp = Blueprint("dashboard", __name__)

_ALLOWED_SPLITS = {"train", "test"}

_PAGE = """<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Priorización de VUS — dashboard</title>
<style>
  body { font-family: -apple-system, "Segoe UI", Roboto, sans-serif; margin: 0;
         background: #f6f7f5; color: #1a231f; }
  header { padding: 20px 28px; background: #17301f; color: #eef5f0; }
  header h1 { margin: 0 0 4px; font-size: 1.3rem; }
  header p { margin: 0; font-size: 0.85rem; color: #b8ccbe; }
  main { padding: 20px 28px; }
.split-links a { margin-right: 12px; font-size: 0.85rem; }
  table { width: 100%; border-collapse: collapse; background: #fff;
          box-shadow: 0 1px 3px rgba(0,0,0,0.08); margin-top: 14px; }
  th, td { padding: 8px 10px; text-align: left; border-bottom: 1px solid #e4e8e5;
           font-size: 0.85rem; vertical-align: top; }
  th { background: #eef2ef; cursor: pointer; user-select: none; position: sticky; top: 0; }
  th:hover { background: #e1e8e3; }
.bar-wrap { width: 90px; background: #e4e8e5; border-radius: 4px; overflow: hidden;
              display: inline-block; height: 8px; vertical-align: middle; margin-right: 6px; }
.bar { height: 8px; background: #2f7d4f; }
.bar.reclass { background: #a8631f; }
.tag { display: inline-block; font-size: 0.7rem; padding: 1px 6px; border-radius: 10px;
         background: #dfeee5; color: #1f5c3a; margin: 1px 2px 1px 0; }
.tag.benign { background: #f3e6de; color: #8a4a1f; }
.tag.weak { background: #f6e2a8; color: #6b4d06; margin-left: 4px; cursor: help; }
  details summary { cursor: pointer; font-size: 0.8rem; color: #4a5750; }
.empty { padding: 40px; text-align: center; color: #6b7a71; }
  code { background: #eef2ef; padding: 1px 5px; border-radius: 3px; }
</style>
</head>
<body>
<header>
  <h1>Priorización de VUS por IA</h1>
  <p>{% if using_ranking %}Ordenadas por el score del modelo de ranking dedicado.
     {% else %}Ordenadas por probabilidad de patogenicidad; el modelo de ranking el objetivo de
     ranking
     todavía no está entrenado.{% endif %}
     Apoyo a la revisión manual, no un veredicto clínico ni una clasificación ACMG certificada.</p>
</header>
<main>
  <div class="split-links">
    Release:
    <a href="/dashboard/train">train</a>
    <a href="/dashboard/test">test</a>
  </div>
  {% if not records %}
  <div class="empty">
    <p>No hay informe generado todavía para <code>{{ split }}</code>.</p>
    <p>Ejecuta: <code>python -m src.serve.vus_reports --split {{ split }}</code></p>
  </div>
  {% else %}
  <table id="vus-table">
    <thead>
      <tr>
        <th onclick="sortBy(0)">Variante</th>
        <th onclick="sortBy(1)">Gen</th>
        <th onclick="sortBy(2)">Consecuencia</th>
        <th onclick="sortBy(3)">Prob. patogenicidad</th>
        <th onclick="sortBy(4)">Prob. reclasificación</th>
        <th>Evidencia principal</th>
      </tr>
    </thead>
    <tbody>
      {% for r in records %}
      <tr>
        <td>{{ r.chrom }}:{{ r.pos }} {{ r.ref }}&gt;{{ r.alt }}</td>
        <td>{{ r.gene or "—" }}</td>
        <td>{{ r.consequence or "—" }}</td>
        <td data-sort="{{ r.probabilidad_patogenica }}">
          <span class="bar-wrap"><span class="bar" style="width:{{ r.pct_pat_px }}px"></span></span>
          {{ r.pct_pat }}%
        </td>
        <td data-sort="{{ r.probabilidad_reclasificacion or -1 }}">
          {% if r.pct_rec is not none %}
          <span class="bar-wrap">
            <span class="bar reclass" style="width:{{ r.pct_rec_px }}px"></span>
          </span>
          {{ r.pct_rec }}%
          {% if not r.probabilidad_reclasificacion_fiable %}
          <span class="tag weak" title="ROC AUC cercano al azar: ver Model Card del modelo de
          potencial de reclasificación">
            señal débil</span>
          {% endif %}
          {% else %}<span style="color:#9aa8a0">no disponible</span>{% endif %}
        </td>
        <td>
          {% for e in r.evidencia[:2] %}
          <span class="tag {{ 'benign' if e.benign else '' }}">{{ e.acmg_tag }}</span>
          {% endfor %}
          {% if r.evidencia|length > 2 %}
          <details><summary>+{{ r.evidencia|length - 2 }} más</summary>
            {% for e in r.evidencia[2:] %}<div>{{ e.acmg_tag }} — {{ e.note }}</div>{% endfor %}
          </details>
          {% endif %}
        </td>
      </tr>
      {% endfor %}
    </tbody>
  </table>
  {% endif %}
</main>
<script>
function sortBy(col) {
  const table = document.getElementById("vus-table");
  const rows = Array.from(table.tBodies[0].rows);
  const asc = table.dataset.sortCol == col ? table.dataset.sortDir !== "asc": true;
  rows.sort((a, b) => {
    const cellA = a.cells[col], cellB = b.cells[col];
    const va = cellA.dataset.sort ?? cellA.innerText;
    const vb = cellB.dataset.sort ?? cellB.innerText;
    const na = parseFloat(va), nb = parseFloat(vb);
    const cmp = (!isNaN(na) && !isNaN(nb)) ? na - nb: va.localeCompare(vb);
    return asc ? cmp: -cmp;
  });
  rows.forEach(r => table.tBodies[0].appendChild(r));
  table.dataset.sortCol = col;
  table.dataset.sortDir = asc ? "asc": "desc";
}
</script>
</body>
</html>
"""


def _view_record(r: dict) -> dict:
    """Precalcula lo que el template solo necesita mostrar (nada de aritmética en Jinja)."""
    pat = r["probabilidad_patogenica"]
    rec = r.get("probabilidad_reclasificacion")
    evidencia = [
        {**e, "benign": e["acmg_tag"].split("-")[0] in ("BP4", "BA1")}
        for e in r.get("evidencia", [])
    ]
    return {
        **r, "evidencia": evidencia,
        "pct_pat": round(pat * 100, 1), "pct_pat_px": round(pat * 90),
        "pct_rec": round(rec * 100, 1) if rec is not None else None,
        "pct_rec_px": round(rec * 90) if rec is not None else 0,
    }


@dashboard_bp.get("/dashboard")
@dashboard_bp.get("/dashboard/<split>")
def dashboard(split: str = "test"):
    # la revisión interna del proyecto: `split` no se valida contra una lista permitida a
    # nivel de Flask (antes solo el CLI de vus_reports.py usaba `choices=[...]`).
    # No era explotable como path traversal (el converter <string> de Werkzeug
    # excluye "/", y `split` se interpola en el NOMBRE de fichero, no como
    # segmento de ruta), pero faltaba el control explícito de defensa en profundidad.
    if split not in _ALLOWED_SPLITS:
        abort(404, description=f"split debe ser uno de {sorted(_ALLOWED_SPLITS)}")
    path = PROJECT_ROOT / "reports" / "serving" / f"vus_reports_{split}.json"
    raw = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    records = [_view_record(r) for r in raw]
    using_ranking = any("ranking_score" in r for r in raw)
    return render_template_string(_PAGE, split=split, records=records, using_ranking=using_ranking)
