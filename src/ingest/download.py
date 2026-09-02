"""Capa RAW inmutable: descarga las releases fechadas de ClinVar.

Sin red, `run` falla con una excepción explícita en vez de degradar a datos
sintéticos (ADR 005): el generador determinista solo se alcanza con `--offline`, y
queda reservado a pruebas y a comprobaciones manuales del pipeline, nunca a
producir un resultado citable.

Escribe `data/raw/MANIFEST.json` con la procedencia, el tamaño y el SHA-256 de cada
fichero, que es lo que sostiene la trazabilidad extremo a extremo.

    python -m src.ingest.download [--offline] [--force] [--prospective]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from src.config import clinvar_prospective_release, load_config, raw_dir
from src.ingest import synthetic

_TIMEOUT = 20


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _try_download(url: str, dest: Path) -> bool:
    """Descarga `url` en `dest`; True si tuvo éxito.

    La URL se reconstruye desde `config.yaml`, así que valido el esquema: un fichero
    mal editado con `file://` no debe poder leer ficheros locales por esta vía.
    """
    if urllib.parse.urlparse(url).scheme not in ("http", "https"):
        raise ValueError(f"esquema de URL no permitido en config.yaml: {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "tfm-mlops-variantes"})  # noqa: S310
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp, open(dest, "wb") as out:  # noqa: S310
            out.write(resp.read())
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
        print(f"  [red no disponible] {url} -> {exc}", file=sys.stderr)
        if dest.exists():
            dest.unlink()
        return False


def _clinvar_targets(cfg: dict, rawdir: Path) -> dict[str, tuple[str, Path]]:
    """Mapea release → (url_real, ruta_destino_local)."""
    base = cfg["sources"]["clinvar"]["base_url"]
    rels = cfg["sources"]["clinvar"]["releases"]
    train = cfg["data"]["clinvar_train_release"]
    test = cfg["data"]["clinvar_test_release"]
    out = {}
    for rel in (train, test):
        out[rel] = (f"{base}/{rels[rel]}", rawdir / f"clinvar_{rel}.vcf.gz")
    return out


def run(offline: bool = False, force: bool = False) -> dict:
    """Ejecuta la ingesta y devuelve el manifest.

    Por defecto exige ClinVar real y lanza `RuntimeError` si la descarga falla.
    `offline=True` usa el generador determinista, y nunca se activa por sí solo.
    """
    cfg = load_config()
    rawdir = raw_dir()
    rawdir.mkdir(parents=True, exist_ok=True)
    targets = _clinvar_targets(cfg, rawdir)

    if offline:
        return _run_offline(cfg, rawdir, targets, force)
    return _run_real(rawdir, targets, force)


def _run_real(rawdir: Path, targets, force: bool) -> dict:
    existing = all(p.exists() for _, p in targets.values())
    if existing and not force:
        prev = _read_manifest(rawdir)
        source = prev.get("source", "unknown_preexisting") if prev else "unknown_preexisting"
        print(f"RAW ya presente (usar --force para regenerar). "
              f"Procedencia registrada conservada: {source}.")
        return _write_manifest(rawdir, targets, dbnsfp_path=None, source=source)

    print("Descargando ClinVar real (NCBI)...")
    ok = all(_try_download(url, dest) for url, dest in targets.values())
    if not ok:
        raise RuntimeError(
            "No se pudo descargar ClinVar real desde NCBI (red no disponible o "
            "fuente caída). Este proyecto NO sustituye datos reales por sintéticos "
            "de forma automática: son variantes genéticas asociadas a enfermedades "
            "reales, y un resultado basado en datos inventados no es válido como "
            "resultado del TFM (ver ADR 005, revisado 2026-07-30). Si de verdad "
            "quieres datos deterministas sintéticos para probar el pipeline (nunca "
            "para generar un resultado citable), ejecuta con --offline explícito."
        )
    print("ClinVar real descargado. Las features se obtienen en la "
          "anotación (annotation_source=multi_source, myvariant.info real), no "
          "aquí: este comando ya no genera ningún fichero de features sintético.")
    return _write_manifest(rawdir, targets, dbnsfp_path=None, source="ncbi_clinvar")


def _run_offline(cfg: dict, rawdir: Path, targets, force: bool) -> dict:
    dbnsfp_path = rawdir / "dbnsfp_subset.tsv.gz"
    existing = all(p.exists() for _, p in targets.values()) and dbnsfp_path.exists()
    if existing and not force:
        print("RAW sintético ya presente (usar --force para regenerar).")
        return _write_manifest(rawdir, targets, dbnsfp_path, source="synthetic_offline")

    train_rel = cfg["data"]["clinvar_train_release"]
    test_rel = cfg["data"]["clinvar_test_release"]
    scfg = synthetic.config_from_dict(cfg.get("synthetic", {}))
    scfg = synthetic.SyntheticConfig(
        seed=scfg.seed,
        chromosomes=tuple(str(c) for c in cfg["data"]["chromosomes_subset"]),
        n_variants_train=scfg.n_variants_train,
        n_new_in_test=scfg.n_new_in_test,
        frac_reclassified=scfg.frac_reclassified,
    )
    train, test = synthetic.generate_releases(scfg)
    synthetic.write_clinvar_vcf(train, targets[train_rel][1])
    synthetic.write_clinvar_vcf(test, targets[test_rel][1])
    synthetic.write_dbnsfp(test, dbnsfp_path)
    return _write_manifest(rawdir, targets, dbnsfp_path, source="synthetic_offline")


def _read_manifest(rawdir: Path) -> dict | None:
    path = rawdir / "MANIFEST.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _write_manifest(rawdir: Path, targets, dbnsfp_path: Path | None, source: str) -> dict:
    files = {}
    for rel, (_, path) in targets.items():
        files[path.name] = {
            "release": rel, "bytes": path.stat().st_size, "sha256": _sha256(path),
        }
    if dbnsfp_path is not None and dbnsfp_path.exists():
        files[dbnsfp_path.name] = {
            "role": "features", "bytes": dbnsfp_path.stat().st_size,
            "sha256": _sha256(dbnsfp_path),
        }
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
        "source": source,
        "assembly": "GRCh38",
        "files": files,
    }
    (rawdir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"RAW listo (fuente={source}). Manifest en {rawdir / 'MANIFEST.json'}")
    for name, meta in files.items():
        print(f"  * {name}: {meta['bytes']:,} bytes")
    return manifest


def run_prospective(force: bool = False) -> Path:
    """Descarga la release prospectiva, independiente del par train/test.

    Solo sirve para leer la verdad terreno de una release publicada después de fijar
    ese par (`src.train.train_reclass.run_prospective`). Sin fallback sintético.
    """
    cfg = load_config()
    rel = clinvar_prospective_release()
    if not rel:
        raise RuntimeError(
            "config.data.clinvar_prospective_release no está definido en config.yaml.")
    rawdir = raw_dir()
    rawdir.mkdir(parents=True, exist_ok=True)
    rels = cfg["sources"]["clinvar"]["releases"]
    if rel not in rels:
        raise RuntimeError(f"Release prospectiva '{rel}' sin URL en sources.clinvar.releases.")
    base = cfg["sources"]["clinvar"]["base_url"]
    url = f"{base}/{rels[rel]}"
    dest = rawdir / f"clinvar_{rel}.vcf.gz"
    if dest.exists() and not force:
        print(f"RAW prospectiva ya presente: {dest.name} (usar --force para regenerar).")
        return dest
    print(f"Descargando release prospectiva {rel} (NCBI)...")
    if not _try_download(url, dest):
        raise RuntimeError(
            f"No se pudo descargar la release prospectiva {rel} desde NCBI. La validación "
            "temporal real del modelo de reclasificación no se genera sin este "
            "fichero (no hay fallback sintético: "
            "sería una reclasificación inventada, no una prospectiva real).")
    print(f"RAW prospectiva lista: {dest} ({dest.stat().st_size:,} bytes, sha256={_sha256(dest)})")
    return dest


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Ingesta RAW (ClinVar real).")
    p.add_argument("--offline", action="store_true",
                   help="Opt-in EXPLÍCITO al generador determinista sintético, sin tocar la "
                        "red (tests/desarrollo). Sin este flag, la descarga real es "
                        "obligatoria y el comando falla si no hay red (ver ADR 005).")
    p.add_argument("--force", action="store_true",
                   help="Regenera aunque los ficheros RAW ya existan.")
    p.add_argument("--prospective", action="store_true",
                   help="Descarga solo la release prospectiva de validación temporal, "
                        "en vez del par train/test habitual.")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    if args.prospective:
        run_prospective(force=args.force)
    else:
        run(offline=args.offline, force=args.force)
