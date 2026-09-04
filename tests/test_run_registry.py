"""El registro de runs existe porque "el run más reciente del backend" no
identificaba la ejecución canónica: la suite de pruebas o una reejecución parcial
lo desplazaban en silencio. Estos tests fijan justo esa propiedad.
"""
import json

import pytest

from src.evaluate import run_registry


@pytest.fixture
def registry(tmp_path, monkeypatch):
    monkeypatch.setattr(run_registry, "PROJECT_ROOT", tmp_path)
    return tmp_path / "reports" / "canonical_runs.json"


def test_record_run_crea_el_fichero_y_guarda_el_run(registry):
    run_registry.record_run("variant_pathogenicity", "abc123", algorithm="logistic_regression")

    data = json.loads(registry.read_text(encoding="utf-8"))
    assert data["variant_pathogenicity"]["run_id"] == "abc123"
    assert data["variant_pathogenicity"]["algorithm"] == "logistic_regression"
    assert data["variant_pathogenicity"]["recorded_at_utc"]


def test_record_run_no_pisa_las_etapas_ya_registradas(registry):
    """Cada etapa se ejecuta por separado: registrar una no puede borrar las demás."""
    run_registry.record_run("variant_pathogenicity", "abc123")
    run_registry.record_run("vus_ranking", "def456")

    data = json.loads(registry.read_text(encoding="utf-8"))
    assert set(data) == {"variant_pathogenicity", "vus_ranking"}
    assert data["variant_pathogenicity"]["run_id"] == "abc123"


def test_record_run_reemplaza_solo_su_propia_etapa(registry):
    """Reejecutar una etapa debe actualizar su run, y solo el suyo."""
    run_registry.record_run("vus_ranking", "viejo")
    run_registry.record_run("variant_pathogenicity", "intacto")
    run_registry.record_run("vus_ranking", "nuevo")

    data = json.loads(registry.read_text(encoding="utf-8"))
    assert data["vus_ranking"]["run_id"] == "nuevo"
    assert data["variant_pathogenicity"]["run_id"] == "intacto"


def test_load_runs_sin_fichero_devuelve_vacio(registry):
    assert run_registry.load_runs() == {}


def test_load_runs_tolera_un_json_corrupto(registry):
    """Un registro ilegible no debe tumbar la captura de procedencia: se prefiere
    una procedencia incompleta y visible a un fallo en el último paso del pipeline."""
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text("{esto no es json", encoding="utf-8")

    assert run_registry.load_runs() == {}
    run_registry.record_run("vus_ranking", "tras_corrupcion")
    assert run_registry.load_runs()["vus_ranking"]["run_id"] == "tras_corrupcion"
