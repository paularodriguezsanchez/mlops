"""Construcción del dataset GOLD: binarización del target y split temporal.

SILVER (anotado por release) → [este módulo] → GOLD (train/test + VUS + excluidas)

Reglas del target (desde `config/config.yaml`, ver `docs/datasheet.md`):
  * Positivo (1): Pathogenic, Likely_pathogenic, Pathogenic/Likely_pathogenic.
  * Negativo (0): Benign, Likely_benign, Benign/Likely_benign.
  * VUS (población reservada para el modelo de reclasificación, priorización):
  estrictamente
    "Uncertain_significance" (`src/features/reclassification.py::VUS_LABEL`).
  * Excluidas (ni entrenan ni se priorizan como VUS): cualquier otro `clnsig`
    (clasificaciones conflictivas -- en cualquier vocabulario de release --,
    not_provided, other, drug_response, etc.). Se persisten aparte por
    trazabilidad, no se descartan en silencio (una revisión posterior del proyecto: antes se
    mezclaban con la población de VUS sin documentarlo,
    y esa mezcla es la causa de la discrepancia 67 vs 54 reclasificaciones
    citada en la memoria -- ver un hallazgo de esa revisión).

Split temporal (clave para el drift, OE5):
  * TRAIN = release antigua (config.data.clinvar_train_release).
  * TEST = release nueva (config.data.clinvar_test_release).

Uso:
    python -m src.features.build_dataset
"""
from __future__ import annotations

import pandas as pd

from src.config import interim_dir, load_config, processed_dir
from src.features.reclassification import is_vus


def _label_map(cfg: dict) -> dict[str, int]:
    m: dict[str, int] = {}
    for lab in cfg["target"]["positive_labels"]:
        m[lab] = 1
    for lab in cfg["target"]["negative_labels"]:
        m[lab] = 0
    return m


def binarize_target(
    df: pd.DataFrame, cfg: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Devuelve (df_etiquetado_con_label, df_vus_reservado_estricto, df_excluido)."""
    mapping = _label_map(cfg)
    label = df["clnsig"].map(mapping)
    labeled = df.assign(label=label)
    train_set = labeled[labeled["label"].notna()].copy()
    train_set["label"] = train_set["label"].astype(int)
    unlabeled = labeled[labeled["label"].isna()].drop(columns=["label"]).copy()
    vus = unlabeled[is_vus(unlabeled["clnsig"])].copy()
    excluded = unlabeled[~is_vus(unlabeled["clnsig"])].copy()
    return (
        train_set.reset_index(drop=True),
        vus.reset_index(drop=True),
        excluded.reset_index(drop=True),
    )


def _summary(name: str, df: pd.DataFrame) -> dict:
    pos = int(df["label"].sum())
    n = len(df)
    return {
        "split": name, "n": n, "positivos": pos, "negativos": n - pos,
        "prevalencia_pos": round(pos / n, 4) if n else 0.0,
    }


def _label_taxonomy(raw: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Tabla exhaustiva CLNSIG crudo -> bucket asignado (revisión posterior del proyecto)."""
    mapping = _label_map(cfg)
    counts = raw["clnsig"].value_counts()
    rows = []
    for clnsig, n in counts.items():
        if clnsig in mapping:
            bucket = "positivo" if mapping[clnsig] == 1 else "negativo"
        elif clnsig == "Uncertain_significance":
            bucket = "vus"
        else:
            bucket = "excluido"
        rows.append({"clnsig_crudo": clnsig, "n": int(n), "bucket": bucket})
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


def run() -> dict:
    cfg = load_config()
    indir, outdir = interim_dir(), processed_dir()
    outdir.mkdir(parents=True, exist_ok=True)

    train_rel = cfg["data"]["clinvar_train_release"]
    test_rel = cfg["data"]["clinvar_test_release"]
    splits = {"train": train_rel, "test": test_rel}

    summaries = []
    taxonomy_frames = []
    for split, rel in splits.items():
        df = pd.read_parquet(indir / f"annotated_{rel}.parquet")
        labeled, vus, excluded = binarize_target(df, cfg)
        labeled.to_parquet(outdir / f"{split}.parquet", index=False)
        vus.to_parquet(outdir / f"vus_{split}.parquet", index=False)
        excluded.to_parquet(outdir / f"excluded_{split}.parquet", index=False)
        s = _summary(split, labeled)
        s["release"] = rel
        s["vus_reservadas"] = len(vus)
        s["excluidas"] = len(excluded)
        summaries.append(s)
        tax = _label_taxonomy(df, cfg)
        tax.insert(0, "split", split)
        tax.insert(1, "release", rel)
        taxonomy_frames.append(tax)
        print(f"[{split} · {rel}] etiquetadas={s['n']} "
              f"(pos={s['positivos']}, neg={s['negativos']}, "
              f"prev={s['prevalencia_pos']}); VUS reservadas={len(vus)}; "
              f"excluidas={len(excluded)}")

    taxonomy = pd.concat(taxonomy_frames, ignore_index=True)
    reports_dir = outdir.parent.parent / "reports" / "training"
    reports_dir.mkdir(parents=True, exist_ok=True)
    taxonomy.to_csv(reports_dir / "clinvar_label_taxonomy.csv", index=False)

    return {"outdir": outdir, "summaries": summaries, "taxonomy": taxonomy}


if __name__ == "__main__":
    run()
