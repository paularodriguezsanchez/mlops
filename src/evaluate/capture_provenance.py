"""Procedencia de la ejecución canónica, en un único fichero.

Commit de git, hash DVC de los datos crudos, identificadores de run de MLflow de
cada modelo y marca de tiempo de la anotación. Sin esto, una cifra citada en la
memoria no se puede trazar hasta el experimento exacto que la produjo.

    python -m src.evaluate.capture_provenance
"""
from __future__ import annotations

import json
import sqlite3
import subprocess
from datetime import datetime, timezone

import yaml

from src.config import PROJECT_ROOT, raw_dir
from src.evaluate.run_registry import load_runs


def _git_sha() -> str | None:
    try:
        out = subprocess.run(  # noqa: S603
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,  # noqa: S607
            capture_output=True, text=True, check=True, timeout=10)
        return out.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return None


def _dvc_hash() -> dict | None:
    path = PROJECT_ROOT / "data" / "raw.dvc"
    if not path.exists():
        return None
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    outs = doc.get("outs", [{}])[0]
    return {"md5": outs.get("md5"), "size": outs.get("size"), "nfiles": outs.get("nfiles")}


def _latest_run_per_experiment() -> dict:
    """Último run_id por experimento de MLflow, consultado directamente sobre el
    backend SQLite -- evita depender de un servidor MLflow levantado solo para
    generar la memoria."""
    db_path = PROJECT_ROOT / "mlflow.db"
    if not db_path.exists():
        return {}
    con = sqlite3.connect(str(db_path))
    try:
        cur = con.cursor()
        cur.execute("SELECT experiment_id, name FROM experiments")
        experiments = dict(cur.fetchall())
        out = {}
        for exp_id, name in experiments.items():
            cur.execute(
                "SELECT run_uuid, start_time FROM runs WHERE experiment_id=? "
                "ORDER BY start_time DESC LIMIT 1", (exp_id,))
            row = cur.fetchone()
            if row:
                out[name] = {"run_id": row[0], "start_time_ms": row[1]}
        return out
    finally:
        con.close()


def _annotation_timestamps() -> dict:
    manifest_path = raw_dir() / "MANIFEST.json"
    if not manifest_path.exists():
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {"raw_generated_utc": manifest.get("generated_utc"), "source": manifest.get("source")}


def run() -> dict:
    result = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
        "git_sha": _git_sha(),
        "dvc_raw_hash": _dvc_hash(),
        # Runs declarados por cada etapa al ejecutarse. El backend de
        # seguimiento es mutable y compartido, asi que "el run mas reciente"
        # no identifica la ejecucion canonica: la suite de pruebas o una
        # reejecucion parcial lo desplazan sin dejar rastro visible.
        "canonical_runs": load_runs(),
        # Se conserva la vista del backend como contraste: si difiere del
        # registro, es que hubo runs posteriores a la ejecucion canonica.
        "mlflow_latest_runs_seen_in_backend": _latest_run_per_experiment(),
        "raw_data": _annotation_timestamps(),
    }
    out_dir = PROJECT_ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "provenance.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Procedencia -> {out_path}")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    run()
