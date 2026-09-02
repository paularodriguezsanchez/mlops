"""Tests base de la configuración (garantiza que el proyecto arranca correctamente)."""
from src.config import get_seed, load_config


def test_config_loads():
    cfg = load_config()
    assert cfg["project"]["name"]
    assert cfg["project"]["assembly"] == "GRCh38"


def test_seed_is_int():
    assert isinstance(get_seed(), int)


def test_target_labels_disjoint():
    t = load_config()["target"]
    pos, neg = set(t["positive_labels"]), set(t["negative_labels"])
    assert pos.isdisjoint(neg), "Etiquetas positivas y negativas no deben solaparse"
