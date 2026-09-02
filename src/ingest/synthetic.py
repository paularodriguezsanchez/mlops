"""Generador OFFLINE determinista de datos tipo ClinVar + dbNSFP.

Motivación (ADR 005): las fuentes reales (NCBI ClinVar FTP, dbNSFP académico) no
son accesibles desde todos los entornos de ejecución (p. ej. sandboxes con
allowlist de red). Para que el pipeline de anotación, el EDA y los tests sean
100 % reproducibles sin red, este módulo genera ficheros *con el mismo esquema*
que los reales:

  * ClinVar VCF (GRCh38) por release fechada, con INFO/CLNSIG realista.
  * dbNSFP subset tab delimitado con scores in silico y frecuencias.

Propiedades clave que imitan al dato real:
  1. Las features (CADD, REVEL, SIFT, PolyPhen, conservación, AF) CORRELACIONAN
     con la patogenicidad mediante una señal latente: el problema es aprendible.
  2. La release "nueva" (test) AÑADE variantes y RECLASIFICA una fracción de VUS
     de la release "antigua" (train): concept drift temporal real, no simulado
     de forma trivial (refuerza OE5).
  3. Todo depende de una única semilla: reproducibilidad total.

Este módulo NO sustituye al dato real: `download.py` intenta primero la descarga
real y solo cae aquí si la red no está disponible.
"""
from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Consecuencias funcionales (Sequence Ontology, subconjunto habitual en ClinVar).
# Cada una lleva un "peso" de deleteriedad a priori usado para la señal latente.
CONSEQUENCES = {
    "missense_variant": 0.55,
    "synonymous_variant": 0.05,
    "stop_gained": 0.95,
    "splice_donor_variant": 0.90,
    "splice_acceptor_variant": 0.90,
    "frameshift_variant": 0.92,
    "intron_variant": 0.10,
    "5_prime_UTR_variant": 0.15,
}

BASES = ("A", "C", "G", "T")

# Estado de revisión de ClinVar (CLNREVSTAT), con pesos aproximados a la
# distribución real observada en `data/raw/clinvar_2023-12.vcf.gz` (ADR 008).
# El generador usa un único vocabulario (a diferencia de ClinVar real, que
# cambió de terminología entre las dos releases, ver `schema.py`): no es
# necesario reproducir ese detalle para que el pipeline sea ejercitable
# offline, solo que `review_status` exista y sea mapeable a estrellas.
_REVIEW_STATUSES = (
    "criteria_provided,_single_submitter",
    "criteria_provided,_multiple_submitters,_no_conflicts",
    "criteria_provided,_conflicting_interpretations",
    "no_assertion_criteria_provided",
    "reviewed_by_expert_panel",
)
_REVIEW_WEIGHTS = (0.78, 0.15, 0.05, 0.015, 0.005)

# Genes de ejemplo por cromosoma (nombres reales, uso ilustrativo).
GENES = {
    "1": ["BRCA1_like", "MTHFR", "PADI4", "CASP9"],
    "2": ["MSH2", "MSH6", "ALK", "SOS1"],
    "3": ["MLH1", "VHL", "BAP1", "PIK3CA"],
}


@dataclass(frozen=True)
class SyntheticConfig:
    seed: int = 42
    chromosomes: tuple[str, ...] = ("1", "2", "3")
    n_variants_train: int = 6000
    n_new_in_test: int = 1200
    frac_reclassified: float = 0.06


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _build_core_variants(cfg: SyntheticConfig, n: int, rng: np.random.Generator) -> dict:
    """Genera el núcleo de variantes con features y una señal latente de patogenicidad."""
    chroms = rng.choice(cfg.chromosomes, size=n)
    pos = rng.integers(1_000_000, 60_000_000, size=n)
    ref = rng.choice(BASES, size=n)
    # alt distinto de ref
    alt = ref.copy()
    for i in range(n):
        choices = [b for b in BASES if b != ref[i]]
        alt[i] = rng.choice(choices)

    cons_names = np.array(list(CONSEQUENCES.keys()))
    cons_weights = np.array(list(CONSEQUENCES.values()))
    cons_idx = rng.integers(0, len(cons_names), size=n)
    consequence = cons_names[cons_idx]
    cons_deleteriousness = cons_weights[cons_idx]

    genes = np.array(
        [rng.choice(GENES[c]) for c in chroms]
    )
    review_status = rng.choice(_REVIEW_STATUSES, size=n, p=_REVIEW_WEIGHTS)

    # Dos señales para lograr un error de Bayes REALISTA (no separable al 100 %)
    # `latent` (señal de etiqueta) genera la significancia clínica de ClinVar.
    latent = _sigmoid(2.2 * (cons_deleteriousness - 0.5) + rng.normal(0, 0.9, size=n))
    # `feat_signal` es un PROXY RUIDOSO de la etiqueta: es lo que ven las features.
    # La brecha entre ambas (ruido 0.38) impone un límite superior realista al
    # rendimiento de cualquier clasificador (como ocurre con REVEL/CADD reales, AUC~0.9).
    feat_signal = np.clip(latent + rng.normal(0, 0.38, size=n), 0, 1)

    # Features in silico derivadas del proxy ruidoso (con ruido adicional propio)
    # Frecuencia alélica poblacional (gnomAD): patogénicas suelen ser MUY raras.
    gnomad_af = np.clip(
        10 ** (-1.0 - 5.0 * feat_signal + rng.normal(0, 0.8, size=n)), 1e-7, 0.5
    )
    # CADD phred (0..40+): mayor = más deletéreo.
    cadd = np.clip(2 + 33 * feat_signal + rng.normal(0, 6, size=n), 0, 45)
    # REVEL (0..1): mayor = más patogénico.
    revel = np.clip(feat_signal + rng.normal(0, 0.22, size=n), 0, 1)
    # SIFT (0..1): MENOR = más dañino (invertido).
    sift = np.clip(1 - feat_signal + rng.normal(0, 0.22, size=n), 0, 1)
    # PolyPhen-2 (0..1): mayor = más dañino.
    polyphen = np.clip(feat_signal + rng.normal(0, 0.22, size=n), 0, 1)
    # AlphaMissense (0..1): mayor = más patogénico (estructura vía AlphaFold, B5/ADR 007).
    alphamissense = np.clip(feat_signal + rng.normal(0, 0.18, size=n), 0, 1)
    # Conservación GERP++ (-12..6) y phyloP (-20..10): mayor = más conservado.
    gerp = np.clip(-2 + 8 * feat_signal + rng.normal(0, 2.6, size=n), -12, 6)
    phylop = np.clip(-1 + 9 * feat_signal + rng.normal(0, 3.5, size=n), -20, 10)

    return {
        "chrom": chroms,
        "pos": pos,
        "ref": ref,
        "alt": alt,
        "gene": genes,
        "consequence": consequence,
        "review_status": review_status,
        "latent": latent,
        "gnomad_af": gnomad_af,
        "cadd_phred": cadd,
        "revel_score": revel,
        "sift_score": sift,
        "polyphen_score": polyphen,
        "alphamissense_score": alphamissense,
        "gerp_rs": gerp,
        "phylop": phylop,
    }


def _assign_clnsig(latent: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Asigna significancia clínica tipo ClinVar a partir de la señal latente.

    Reparte en Pathogenic / Likely_pathogenic / Uncertain_significance /
    Likely_benign / Benign de forma coherente con `latent`, dejando una banda
    intermedia grande de VUS (como en ClinVar real).
    """
    n = latent.size
    u = rng.uniform(0, 1, size=n)
    labels = np.empty(n, dtype=object)
    for i in range(n):
        lat = latent[i]
        if lat > 0.80:
            labels[i] = "Pathogenic" if u[i] < 0.7 else "Likely_pathogenic"
        elif lat > 0.60:
            labels[i] = "Likely_pathogenic" if u[i] < 0.55 else "Uncertain_significance"
        elif lat > 0.40:
            labels[i] = "Uncertain_significance"
        elif lat > 0.20:
            labels[i] = "Likely_benign" if u[i] < 0.55 else "Uncertain_significance"
        else:
            labels[i] = "Benign" if u[i] < 0.7 else "Likely_benign"
    return labels


def generate_releases(cfg: SyntheticConfig) -> tuple[dict, dict]:
    """Devuelve (release_train, release_test) como dicts de columnas alineadas.

    * release_train: núcleo de `n_variants_train` variantes.
    * release_test: mismas variantes (misma clave y features) PERO con una
      fracción de VUS reclasificada, MÁS `n_new_in_test` variantes nuevas.
      Esto produce concept drift temporal realista.
    """
    rng = np.random.default_rng(cfg.seed)

    core = _build_core_variants(cfg, cfg.n_variants_train, rng)
    clnsig_train = _assign_clnsig(core["latent"], rng)

    # Release test: copia + reclasificación de VUS + variantes nuevas
    clnsig_test = clnsig_train.copy()
    vus_idx = np.where(clnsig_train == "Uncertain_significance")[0]
    n_reclass = int(len(vus_idx) * cfg.frac_reclassified)
    # Selección de qué VUS se reclasifican: PONDERADA por |latent-0.5|, no
    # uniforme. Motivo (ADR 007): en la práctica clínica, las VUS con más
    # evidencia previa (aunque insuficiente para un veredicto formal) tienden
    # a resolverse antes que las genuinamente ambiguas. Una selección
    # uniforme no deja ninguna señal aprendible sobre "qué VUS se
    # reclasificará", lo que haría el modelo de reclasificación
    # no mejor que el azar por construcción del propio generador.
    lat_vus = core["latent"][vus_idx]
    weights = np.abs(lat_vus - 0.5) + 0.05  # +0.05: nunca probabilidad cero
    weights = weights / weights.sum()
    reclass = rng.choice(vus_idx, size=n_reclass, replace=False, p=weights)
    for i in reclass:
        # Las VUS se resuelven según su señal latente (como en la práctica clínica).
        lat = core["latent"][i]
        if lat >= 0.5:
            clnsig_test[i] = "Likely_pathogenic"
        else:
            clnsig_test[i] = "Likely_benign"

    # Variantes nuevas que solo aparecen en la release nueva.
    new_core = _build_core_variants(cfg, cfg.n_new_in_test, rng)
    clnsig_new = _assign_clnsig(new_core["latent"], rng)

    def pack(core_d: dict, clnsig: np.ndarray) -> dict:
        d = {k: v for k, v in core_d.items() if k != "latent"}
        d["clnsig"] = clnsig
        return d

    train = pack(core, clnsig_train)
    # concatenar núcleo (con clnsig test) + nuevas
    test_core = pack(core, clnsig_test)
    test = {}
    for k in test_core:
        test[k] = np.concatenate([test_core[k], pack(new_core, clnsig_new)[k]])
    return train, test


# =============================================================================
# Serialización al FORMATO REAL (VCF de ClinVar y tabla dbNSFP)
# =============================================================================

_VCF_HEADER = """##fileformat=VCFv4.2
##source=SYNTHETIC_ClinVar_like (ADR 005, offline determinista)
##reference=GRCh38
##INFO=<ID=CLNSIG,Number=.,Type=String,Description="Clinical significance">
##INFO=<ID=GENEINFO,Number=1,Type=String,Description="Gene symbol">
##INFO=<ID=MC,Number=.,Type=String,Description="Molecular consequence">
##INFO=<ID=CLNREVSTAT,Number=.,Type=String,Description="ClinVar review status">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO
"""


def write_clinvar_vcf(release: dict, path: Path) -> None:
    """Escribe una release en formato VCF de ClinVar (gzip), ordenada por chrom,pos."""
    path.parent.mkdir(parents=True, exist_ok=True)
    order = np.lexsort((release["pos"], release["chrom"]))
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(_VCF_HEADER)
        for j, i in enumerate(order, start=1):
            info = (
                f"CLNSIG={release['clnsig'][i]};"
                f"GENEINFO={release['gene'][i]};"
                f"MC={release['consequence'][i]};"
                f"CLNREVSTAT={release['review_status'][i]}"
            )
            fh.write(
                f"{release['chrom'][i]}\t{int(release['pos'][i])}\t{j}\t"
                f"{release['ref'][i]}\t{release['alt'][i]}\t.\t.\t{info}\n"
            )


# Cabecera dbNSFP (subconjunto de columnas reales relevantes al TFM).
_DBNSFP_COLS = [
    "#chr", "pos(1-based)", "ref", "alt", "genename",
    "SIFT_score", "Polyphen2_HDIV_score", "CADD_phred", "REVEL_score",
    "AlphaMissense_score", "GERP++_RS", "phyloP100way_vertebrate", "gnomAD_exomes_AF",
]


def write_dbnsfp(union_release: dict, path: Path) -> None:
    """Escribe la tabla dbNSFP (gzip, tab-delimitado) para el UNIVERSO de variantes.

    dbNSFP es independiente de ClinVar: contiene features por variante. Se escribe
    la unión de variantes (release nueva incluye a todas) para poder anotar ambas.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    n = union_release["pos"].size
    order = np.lexsort((union_release["pos"], union_release["chrom"]))
    seen: set[tuple] = set()
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write("\t".join(_DBNSFP_COLS) + "\n")
        for i in order:
            key = (
                union_release["chrom"][i], int(union_release["pos"][i]),
                union_release["ref"][i], union_release["alt"][i],
            )
            if key in seen:
                continue
            seen.add(key)
            # SIFT/PolyPhen/REVEL solo son informativos para variantes missense
            # (dbNSFP los marca "." en el resto). CADD/GERP/phyloP/AF, para todas.
            is_missense = union_release["consequence"][i] == "missense_variant"
            sift = f"{union_release['sift_score'][i]:.4f}" if is_missense else "."
            pph = f"{union_release['polyphen_score'][i]:.4f}" if is_missense else "."
            revel = f"{union_release['revel_score'][i]:.4f}" if is_missense else "."
            amis = f"{union_release['alphamissense_score'][i]:.4f}" if is_missense else "."
            fh.write(
                "\t".join([
                    str(union_release["chrom"][i]),
                    str(int(union_release["pos"][i])),
                    str(union_release["ref"][i]),
                    str(union_release["alt"][i]),
                    str(union_release["gene"][i]),
                    sift, pph,
                    f"{union_release['cadd_phred'][i]:.3f}",
                    revel, amis,
                    f"{union_release['gerp_rs'][i]:.3f}",
                    f"{union_release['phylop'][i]:.3f}",
                    f"{union_release['gnomad_af'][i]:.3e}",
                ]) + "\n"
            )
    _ = n  # documentado: n variantes generadas antes de deduplicar por clave


def config_from_dict(d: dict) -> SyntheticConfig:
    """Construye SyntheticConfig desde el bloque `synthetic` de config.yaml."""
    return SyntheticConfig(
        seed=int(d.get("seed", 42)),
        n_variants_train=int(d.get("n_variants_train", 6000)),
        n_new_in_test=int(d.get("n_new_in_test", 1200)),
        frac_reclassified=float(d.get("frac_reclassified", 0.06)),
    )
