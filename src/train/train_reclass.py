"""Modelo de potencial de reclasificación de VUS (ADR 007 §5).

Población: las VUS reservadas de la release antigua. Etiqueta: 1 si esa misma
variante aparece resuelta (patogénica o benigna) en la release nueva, 0 si sigue
sin resolver o desaparece.

La evaluación por defecto es un holdout aleatorio estratificado dentro del par de
releases: mide señal retrospectiva, no capacidad prospectiva. `run_prospective`
aplica el modelo ya entrenado sobre una tercera release posterior.

    python -m src.train.train_reclass [--prospective]
"""
from __future__ import annotations

import argparse
import json

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from src.config import (
    PROJECT_ROOT,
    get_seed,
    interim_dir,
    load_config,
    processed_dir,
    reclass_cv_repeats,
    reclass_cv_splits,
    reclass_reliable_roc_auc,
)
from src.evaluate.metrics import (
    bootstrap_pr_auc_ci,
    bootstrap_roc_auc_ci,
    calibration_report,
    compute_metrics,
    export_pr_curve,
    export_roc_curve,
    precision_recall_lift_at_k,
    save_confusion_matrix,
)
from src.features.preprocess import (
    RECLASS_FEATURE_COLUMNS,
    SAFE_RECLASS_FEATURE_COLUMNS,
    build_reclass_preprocessor,
    build_safe_reclass_preprocessor,
)
from src.features.reclassification import is_resolved
from src.train.train import _models, _resolve_tracking_uri, data_provenance

_KEY = ["chrom", "pos", "ref", "alt"]

# Umbral operativo (0.5 = azar): por debajo, dashboard e informes leen
# `metrics.json` y avisan de señal débil en vez de mostrar la probabilidad sin más.
# Vive en config.yaml (`train.reclass_reliable_roc_auc`), igual que el resto de
# umbrales que gobiernan una decisión del sistema; este alias lo resuelve una vez
# al importar para no repetir la llamada en los cinco puntos que lo usan.
RELIABLE_ROC_AUC_THRESHOLD = reclass_reliable_roc_auc()
CV_SPLITS = reclass_cv_splits()
CV_REPEATS = reclass_cv_repeats()


def build_reclass_dataset(test_release: str) -> pd.DataFrame:
    """VUS de la release de train con su etiqueta de reclasificación en `test_release`.

    Lee `vus_train.parquet` (GOLD, nombrado por split) y
    `annotated_{test_release}.parquet` (SILVER, nombrado por release).
    """
    vus = pd.read_parquet(processed_dir() / "vus_train.parquet")
    annotated_new = pd.read_parquet(interim_dir() / f"annotated_{test_release}.parquet")
    return _label_reclassified(vus, annotated_new)


def _label_reclassified(vus: pd.DataFrame, annotated_new: pd.DataFrame) -> pd.DataFrame:
    """Etiqueta cada VUS: 1 si su clave aparece resuelta en `annotated_new`."""
    new_clnsig = annotated_new[_KEY + ["clnsig"]].rename(columns={"clnsig": "clnsig_new"})
    merged = vus.merge(new_clnsig, on=_KEY, how="left")
    merged["label"] = is_resolved(merged["clnsig_new"]).astype(int)
    return merged.drop(columns=["clnsig_new"])


def _run_safe_ablation(labeled: pd.DataFrame, train_idx, hold_idx, y_train: pd.Series,
                       y_hold: pd.Series, algorithm: str, seed: int, provenance: dict) -> dict:
    """Ablación temporalmente segura: mitigación parcial del leakage (ADR 008).

    Mismo split y mismo algoritmo que el modelo completo, restringido a
    `consequence` y `review_stars`, las dos únicas features ancladas a la fecha
    real de cada release. El resto se consulta a myvariant.info "hoy", sin
    anclaje de versión. Comparar ambos cuantifica cuánta señal depende de esa fuga
    potencial en vez de solo documentarla.
    """
    Xs_train = labeled.loc[train_idx, SAFE_RECLASS_FEATURE_COLUMNS]
    Xs_hold = labeled.loc[hold_idx, SAFE_RECLASS_FEATURE_COLUMNS]
    clf = _models(seed)[algorithm]
    with mlflow.start_run(run_name=f"reclass_{algorithm}_safe_ablation") as mlrun:
        pipe = Pipeline([("pre", build_safe_reclass_preprocessor()), ("clf", clf)])
        pipe.fit(Xs_train, y_train)
        y_prob = pipe.predict_proba(Xs_hold)[:, 1]
        metrics = compute_metrics(y_hold, y_prob)
        mlflow.log_params({
            "algorithm": algorithm, "seed": seed, "task": "reclassification_safe_ablation",
            "features": ",".join(SAFE_RECLASS_FEATURE_COLUMNS),
            "clinvar_data_source": provenance["clinvar_source"],
            "annotation_source": provenance["annotation_source"],
        })
        mlflow.set_tag("real_data_end_to_end", provenance["is_real_data"])
        mlflow.set_tag("leakage_ablation", "temporally_safe_only")
        mlflow.log_metrics(metrics)
        run_id = mlrun.info.run_id
    print(f"[reclass_{algorithm}_safe_ablation] PR AUC={metrics['pr_auc']:.4f} "
          f"ROC AUC={metrics['roc_auc']:.4f} (solo consequence+review_stars, ADR 008)")
    return {"algorithm": algorithm, "features": SAFE_RECLASS_FEATURE_COLUMNS,
            "run_id": run_id, **metrics}


def run(tracking_uri: str | None = None, test_size: float = 0.25) -> dict:
    cfg = load_config()
    seed = get_seed()
    train_rel = cfg["data"]["clinvar_train_release"]
    test_rel = cfg["data"]["clinvar_test_release"]

    labeled = build_reclass_dataset(test_rel)
    X, y = labeled[RECLASS_FEATURE_COLUMNS], labeled["label"]
    n_pos = int(y.sum())
    print(f"VUS población={len(labeled)} (release {train_rel}) | "
          f"reclasificadas en {test_rel}={n_pos} ({100 * y.mean():.1f}%)")
    if n_pos < 2 or n_pos > len(y) - 2:
        raise ValueError(
            f"Muy pocos ejemplos de una clase ({n_pos}/{len(y)}) para entrenar "
            "el modelo de reclasificación con un holdout estratificado.")

    X_train, X_hold, y_train, y_hold = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y)

    provenance = data_provenance()

    mlflow.set_tracking_uri(_resolve_tracking_uri(cfg, tracking_uri))
    mlflow.set_experiment(cfg["mlflow"]["reclass_experiment_name"])

    art_dir = PROJECT_ROOT / "reports" / "training"
    # Selección y evaluación sobre conjuntos distintos. `X_hold` es el holdout
    # externo y no interviene en la elección del algoritmo: cada candidato se
    # puntúa por PR AUC medio en validación cruzada repetida y estratificada
    # DENTRO de `X_train`. Con tan pocos positivos una única partición de 5
    # pliegues es muy inestable, de ahí las repeticiones.
    cv = RepeatedStratifiedKFold(n_splits=CV_SPLITS, n_repeats=CV_REPEATS,
                                 random_state=seed)
    results: list[dict] = []
    for name, clf in _models(seed).items():
        with mlflow.start_run(run_name=f"reclass_{name}") as mlrun:
            pipe = Pipeline([("pre", build_reclass_preprocessor()), ("clf", clf)])
            cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv,
                                        scoring="average_precision", n_jobs=1)
            cv_mean, cv_std = float(cv_scores.mean()), float(cv_scores.std())
            # El ajuste sobre X_train y las métricas de holdout se registran para
            # todos los candidatos por transparencia, pero son descriptivas: la
            # decisión ya está tomada con `cv_mean`.
            pipe.fit(X_train, y_train)
            y_prob = pipe.predict_proba(X_hold)[:, 1]
            metrics = compute_metrics(y_hold, y_prob)
            mlflow.log_params({"algorithm": name, "seed": seed, "task": "reclassification",
                               "cv_splits": CV_SPLITS, "cv_repeats": CV_REPEATS,
                               "selection_criterion": "cv_pr_auc_mean_on_outer_train",
                               "clinvar_data_source": provenance["clinvar_source"],
                               "annotation_source": provenance["annotation_source"]})
            mlflow.set_tag("real_data_end_to_end", provenance["is_real_data"])
            mlflow.log_metrics({**metrics, "cv_pr_auc_mean": cv_mean, "cv_pr_auc_std": cv_std})
            cm_path = save_confusion_matrix(
                y_hold, y_prob, art_dir / f"cm_reclass_{name}.png",
                title=f"CM · reclasificación · {name}",
                labels=("No reclasificada", "Reclasificada"))
            mlflow.log_artifact(str(cm_path), artifact_path="figures")
            mlflow.sklearn.log_model(
                pipe, name="model",
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE)
            results.append({"name": name, "run_id": mlrun.info.run_id, "pipe": pipe,
                            "metrics": metrics,
                            "cv_pr_auc_mean": cv_mean, "cv_pr_auc_std": cv_std})
            print(f"[reclass_{name}] CV PR AUC={cv_mean:.4f}±{cv_std:.4f} "
                  f"| holdout PR AUC={metrics['pr_auc']:.4f} (descriptivo)")

    best = max(results, key=lambda r: r["cv_pr_auc_mean"])
    print(f"\n>> Mejor modelo de reclasificación por CV sobre el conjunto de "
          f"entrenamiento: {best['name']} "
          f"(CV PR AUC={best['cv_pr_auc_mean']:.4f}±{best['cv_pr_auc_std']:.4f})")

    # Con la clase positiva tan minoritaria, el estimador puntual no basta: se
    # añaden IC bootstrap, curvas completas, calibración y métricas de cola
    # (precision/recall/lift@k), que son las que importan al ordenar una lista.
    best_prob = best["pipe"].predict_proba(X_hold)[:, 1]
    n_hold_pos = int(y_hold.sum())
    if n_hold_pos >= 2 and n_hold_pos <= len(y_hold) - 2:
        holdout_ci = bootstrap_pr_auc_ci(y_hold, best_prob, seed=seed)
        roc_ci = bootstrap_roc_auc_ci(y_hold, best_prob, seed=seed)
    else:
        holdout_ci = {"pr_auc_ci_low": None, "pr_auc_ci_high": None, "pr_auc_ci_n_boot": 0}
        roc_ci = {"roc_auc_ci_low": None, "roc_auc_ci_high": None, "roc_auc_ci_n_boot": 0}
    topk_ks = tuple(k for k in (10, 20, 50, 100) if k <= len(y_hold))
    topk = precision_recall_lift_at_k(y_hold, best_prob, ks=topk_ks)
    calibration = calibration_report(y_hold, best_prob, n_bins=5)
    export_pr_curve(y_hold, best_prob, art_dir / "reclass_pr_curve.csv")
    export_roc_curve(y_hold, best_prob, art_dir / "reclass_roc_curve.csv")
    print(f"   Holdout: n={len(y_hold)}, positivos={n_hold_pos}. "
          f"IC 95% PR AUC=[{holdout_ci['pr_auc_ci_low']}, {holdout_ci['pr_auc_ci_high']}] "
          f"(n_boot={holdout_ci['pr_auc_ci_n_boot']}). "
          f"IC 95% ROC AUC=[{roc_ci['roc_auc_ci_low']}, {roc_ci['roc_auc_ci_high']}] "
          f"(n_boot={roc_ci['roc_auc_ci_n_boot']}). Brier={calibration['brier_score']:.4f}. "
          + ", ".join(f"{k}={v}" for k, v in topk.items()))

    ablation = _run_safe_ablation(
        labeled, X_train.index, X_hold.index, y_train, y_hold, best["name"], seed, provenance)

    models_dir = PROJECT_ROOT / "models" / "reclassification_model"
    if models_dir.exists():
        import shutil
        shutil.rmtree(models_dir)
    mlflow.sklearn.save_model(
        best["pipe"], str(models_dir),
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE)
    reliable = best["metrics"]["roc_auc"] >= RELIABLE_ROC_AUC_THRESHOLD
    (models_dir / "metrics.json").write_text(json.dumps({
        "best_algorithm": best["name"],
        "roc_auc": best["metrics"]["roc_auc"], "pr_auc": best["metrics"]["pr_auc"],
        "reliable": reliable, "reliable_threshold": RELIABLE_ROC_AUC_THRESHOLD,
        "holdout_n": len(y_hold), "holdout_n_positivos": n_hold_pos,
        "holdout_pr_auc_ci": holdout_ci,
        "holdout_roc_auc_ci": roc_ci,
        "holdout_top_k": topk,
        "holdout_calibration": calibration,
        "leakage_ablation_temporally_safe": {
            "roc_auc": ablation["roc_auc"], "pr_auc": ablation["pr_auc"],
            "features": ablation["features"],
        },
        **provenance,
    }, indent=2), encoding="utf-8")

    comp = pd.DataFrame([{"name": r["name"], **r["metrics"]} for r in results])
    comp.to_csv(art_dir / "reclassification_model_comparison.csv", index=False)

    _register_best(best["name"], best["run_id"])
    _write_card(best, results, train_rel, test_rel, n_pos, len(labeled), reliable, provenance,
               ablation, holdout_ci, topk, calibration, len(y_hold), n_hold_pos, roc_ci)
    return {"best": best["name"], "results": results, "n_reclassified": n_pos,
            "n_population": len(labeled), "reliable": reliable, "ablation": ablation,
            "holdout_ci": holdout_ci, "holdout_top_k": topk, "holdout_calibration": calibration,
            **provenance}


def _register_best(name: str, run_id: str) -> None:
    cfg = load_config()
    model_name = cfg["mlflow"]["reclass_registered_model_name"]
    try:
        result = mlflow.register_model(f"runs:/{run_id}/model", model_name)
        client = mlflow.tracking.MlflowClient()
        try:
            client.transition_model_version_stage(model_name, result.version, "Staging")
            print(f"Registrado {model_name} v{result.version} -> stage Staging")
        except Exception:  # noqa: BLE001
            client.set_registered_model_alias(model_name, "staging", result.version)
            print(f"Registrado {model_name} v{result.version} -> alias @staging")
    except Exception as exc:  # noqa: BLE001
        print(f"[aviso] no se pudo registrar en el Model Registry: {exc}")


def _write_card(best: dict, results: list, train_rel: str, test_rel: str,
               n_pos: int, n_total: int, reliable: bool, provenance: dict,
               ablation: dict, holdout_ci: dict, topk: dict, calibration: dict,
               holdout_n: int, holdout_n_pos: int, roc_ci: dict) -> None:
    card = PROJECT_ROOT / "docs" / "MODEL_CARD_RECLASSIFICATION.md"
    card.parent.mkdir(parents=True, exist_ok=True)
    rows = "\n".join(
        f"| {r['name']} | {r['metrics']['pr_auc']:.4f} | {r['metrics']['roc_auc']:.4f} "
        f"| {r['metrics']['f1']:.4f} |" for r in results)
    if holdout_ci.get("pr_auc_ci_low") is not None:
        ci_text = (f"[{holdout_ci['pr_auc_ci_low']:.4f}, {holdout_ci['pr_auc_ci_high']:.4f}] "
                  f"(n_boot={holdout_ci['pr_auc_ci_n_boot']})")
    else:
        ci_text = "no calculado (menos de 2 positivos en el holdout)"
    if roc_ci.get("roc_auc_ci_low") is not None:
        roc_ci_text = (f"[{roc_ci['roc_auc_ci_low']:.4f}, {roc_ci['roc_auc_ci_high']:.4f}] "
                       f"(n_boot={roc_ci['roc_auc_ci_n_boot']})")
        roc_ci_crosses_threshold = roc_ci["roc_auc_ci_low"] < RELIABLE_ROC_AUC_THRESHOLD
        roc_ci_crosses_half = roc_ci["roc_auc_ci_low"] < 0.5
    else:
        roc_ci_text = "no calculado (menos de 2 positivos en el holdout)"
        roc_ci_crosses_threshold = True
        roc_ci_crosses_half = True
    topk_rows = "\n".join(
        f"| {k} | {topk.get(f'precision_at_{k}', '-')} | {topk.get(f'recall_at_{k}', '-')} "
        f"| {topk.get(f'lift_at_{k}', '-')} |"
        for k in (10, 20, 50, 100) if f"precision_at_{k}" in topk)
    calibration_rows = "\n".join(
        f"| [{b['bin_low']}, {b['bin_high']}) | {b['n']} | {b['mean_predicted']} "
        f"| {b['observed_rate']} |" for b in calibration["bins"])
    provenance_banner = (
        "**Datos REALES de extremo a extremo.**"
        if provenance["is_real_data"] else
        f"**AVISO: datos SINTÉTICOS** (ClinVar: `{provenance['clinvar_source']}`, "
        f"features: `{provenance['annotation_source']}`). No citar como resultado clínico."
    )
    # delta > 0: el modelo completo discrimina mejor que la ablación segura.
    # delta <= 0: la ablación iguala o supera al completo, un caso distinto de "empata".
    delta = best["metrics"]["roc_auc"] - ablation["roc_auc"]
    if ablation["roc_auc"] < 0.55:
        ablation_note = (
            f"La ablación por sí sola apenas discrimina (ROC AUC {ablation['roc_auc']:.4f}, "
            "cerca de 0.5 = azar): buena parte de la señal del modelo completo depende de "
            "features recalculadas 'hoy' vía myvariant.info (CADD, REVEL, AlphaMissense, "
            "gnomAD, SIFT, PolyPhen, GERP, phyloP), con el riesgo de fuga temporal descrito "
            "arriba. El ROC AUC del modelo completo debe leerse con esta salvedad explícita: "
            "no se puede descartar que una parte relevante sea optimista por este motivo."
        )
    elif delta <= 0:
        ablation_note = (
            f"La ablación (ROC AUC {ablation['roc_auc']:.4f}) **iguala o supera** al modelo "
            f"completo (ROC AUC {best['metrics']['roc_auc']:.4f}, diferencia {delta:+.4f}): "
            "las dos features temporalmente seguras (`consequence`, `review_stars`) por sí "
            "solas discriminan tanto o más que el conjunto completo. Esto es tranquilizador "
            "respecto al riesgo de fuga (el resultado del modelo completo no depende de las "
            "features de riesgo para alcanzar su nivel de discriminación), pero abre una "
            "pregunta distinta que no cierra esta ablación: por qué añadir más features "
            "(incluidas las de riesgo) no mejora, e incluso empeora, el ROC AUC del holdout "
            f"-- con solo {n_pos} positivos en la población, es plausible que el modelo "
            "completo sobreajuste con más dimensiones sin más señal que aportar. No se "
            "presenta el ROC AUC del modelo completo como superior al de la ablación."
        )
    elif delta <= 0.05:
        ablation_note = (
            f"La ablación (ROC AUC {ablation['roc_auc']:.4f}) queda ligeramente por debajo "
            f"del modelo completo (ROC AUC {best['metrics']['roc_auc']:.4f}, diferencia "
            f"{delta:+.4f}). La brecha es pequeña, pero ya no favorece a la ablación: no "
            "puede descartarse que parte de la ventaja del modelo completo venga de las "
            "features con riesgo de fuga temporal. El ROC AUC del modelo completo debe "
            "leerse con esa salvedad."
        )
    else:
        ablation_note = (
            f"La ablación (ROC AUC {ablation['roc_auc']:.4f}) queda claramente por debajo "
            f"del modelo completo (ROC AUC {best['metrics']['roc_auc']:.4f}, diferencia "
            f"{delta:+.4f}): una parte relevante de la señal depende de features con riesgo "
            "de fuga temporal. No citar el ROC AUC del modelo completo sin esta salvedad."
        )
    full_row = (
        f"| Completo ({best['name']}) | {len(RECLASS_FEATURE_COLUMNS)} "
        f"(incluye CADD/REVEL/AlphaMissense/gnomAD/...) "
        f"| {best['metrics']['pr_auc']:.4f} | {best['metrics']['roc_auc']:.4f} |"
    )
    ablation_row = (
        f"| Ablación temporalmente segura ({ablation['algorithm']}) | "
        f"`consequence`, `review_stars` "
        f"| {ablation['pr_auc']:.4f} | {ablation['roc_auc']:.4f} |"
    )
    threshold_justification = (
        f"El umbral {RELIABLE_ROC_AUC_THRESHOLD} es una convención operativa mía, sin respaldo "
        "estadístico ni de la literatura: no existe una referencia que fije 0.6 como punto de "
        "corte de fiabilidad de un ROC AUC. Lo documento como tal. El criterio informativo es "
        "el intervalo de confianza del ROC AUC frente a 0.5, no la comparación contra 0.6."
    )
    roc_ci_note = (
        f"**Intervalo de confianza del ROC AUC** (bootstrap, 95%, 1000 remuestreos): "
        f"{roc_ci_text}. "
        + (
            "El límite inferior del intervalo queda por debajo de 0.5 (azar): con este tamaño "
            "de holdout no puede excluirse que el modelo no discrimine mejor que el azar, pese "
            "a que su estimador puntual supere el umbral de 0.6."
            if roc_ci_crosses_half else
            "El intervalo completo queda por encima de 0.5 (azar), aunque su límite inferior "
            f"{'sí queda por debajo' if roc_ci_crosses_threshold else 'queda también por encima'} "
            f"del umbral operativo de {RELIABLE_ROC_AUC_THRESHOLD}: la discriminación por encima "
            "del azar es más defendible que la superación concreta del umbral interno."
        )
    )
    reliability_note = (
        f"Supera el umbral (ROC AUC {best['metrics']['roc_auc']:.4f} >= "
        f"{RELIABLE_ROC_AUC_THRESHOLD}): la probabilidad se muestra en el dashboard y en los "
        "informes sin aviso adicional.\n\n"
        f"{threshold_justification}\n\n{roc_ci_note}"
        if reliable else
        f"**No supera el umbral** (ROC AUC {best['metrics']['roc_auc']:.4f} < "
        f"{RELIABLE_ROC_AUC_THRESHOLD}; 0.5 = azar): el modelo no discrimina mejor que el azar "
        "con estas features. Dashboard e informes siguen mostrando el número, pero con una "
        "marca de señal débil junto a cada valor. No citable como resultado predictivo.\n\n"
        f"{threshold_justification}\n\n{roc_ci_note}"
    )
    card.write_text(f"""# Model Card: Potencial de reclasificación de VUS

## Procedencia de los datos de este run
{provenance_banner}

## Modelo
* **Tarea:** dada una VUS de la release {train_rel}, predecir si estará resuelta
  (patogénica o benigna) en la release {test_rel}.
* **Algoritmo:** {best['name']}, elegido por PR AUC medio en validación cruzada
  repetida ({CV_SPLITS} pliegues x {CV_REPEATS} repeticiones, estratificada)
  **sobre el conjunto de entrenamiento**. El holdout no interviene en esa
  decisión: se evalúa una sola vez, con el algoritmo ya elegido. Las columnas de
  holdout del resto de candidatos son descriptivas, no criterio de selección.
* **Población:** {n_total} VUS de {train_rel}, de las que {n_pos}
  ({100 * n_pos / n_total:.1f} %) se reclasificaron en {test_rel}.

## Fiabilidad de la señal
{reliability_note}

## Alcance de esta evaluación
El holdout es aleatorio estratificado dentro del par {train_rel}/{test_rel}: mide señal
discriminativa retrospectiva dentro de ese intervalo, no capacidad prospectiva. La
validación temporal externa -mismo modelo, sin reentrenar, sobre una release posterior
que no intervino en la selección ni en el ajuste- está en
`docs/MODEL_CARD_RECLASSIFICATION_PROSPECTIVE.md`
(`python -m src.train.train_reclass --prospective`). Es esa cifra, no esta, la que
responde a si el modelo predice una reclasificación genuinamente futura.

## Leakage temporal de las features (ADR 008)
Las features de anotación (CADD, REVEL, AlphaMissense, gnomAD, SIFT, PolyPhen, GERP,
phyloP) se consultan a myvariant.info en el momento de ejecutar el pipeline, sin anclaje
a la fecha de la release: el mismo estado "de hoy" se aplica a {train_rel} (t0) y a
{test_rel} (t1). Si una VUS se resolvió entre t0 y t1 porque llegó evidencia
computacional nueva -AlphaMissense, publicado en 2023, es el caso claro-, el modelo puede
estar entrenando con el valor posterior de esa evidencia. `consequence` y
`review_status`/`review_stars` sí se leen del VCF fechado y son temporalmente seguras
(`src/features/preprocess.py::SAFE_RECLASS_FEATURE_COLUMNS`).

Como mitigación parcial entreno, sobre el mismo split, un modelo restringido a esas dos
features seguras, para cuantificar la dependencia en vez de solo advertirla.

| Modelo | Features | PR AUC | ROC AUC |
|--------|----------|--------|---------|
{full_row}
{ablation_row}

{ablation_note}

La mitigación completa -congelar snapshots de CADD, REVEL, AlphaMissense y gnomAD
anclados a la fecha de cada release- exige descargas versionadas por fuente que no están
garantizadas como archivadas públicamente; queda como trabajo futuro (ADR 008).

## Métricas (holdout aleatorio)
| Algoritmo | PR AUC | ROC AUC | F1 |
|-----------|--------|---------|-----|
{rows}

## Incertidumbre y métricas de cola
Holdout: n={holdout_n}, positivos={holdout_n_pos} ({100 * holdout_n_pos / holdout_n:.1f} %).

IC 95 % bootstrap del PR AUC (1000 remuestreos): {ci_text}. Con {holdout_n_pos} positivos
el intervalo es necesariamente ancho; lo reporto igualmente.

El uso real del modelo es ordenar una cola de revisión, así que las métricas de cola
(lift=1.0 equivale a ordenar al azar) son más informativas que el agregado:

| k | precision@k | recall@k | lift@k |
|---|---|---|---|
{topk_rows}

Calibración: Brier score = {calibration['brier_score']:.4f} (más bajo, mejor). El
baseline informativo con esta prevalencia **no** es el clasificador que siempre predice
0.5, cuyo Brier de 0.25 es trivial de batir cuando la clase positiva ronda el 1 %: es el
que predice siempre la prevalencia observada, con Brier = p(1-p) =
{holdout_n_pos / holdout_n * (1 - holdout_n_pos / holdout_n):.4f}. Un Brier bajo aquí
refleja sobre todo el desbalance, no discriminación. Por eso la salida se describe como
*score probabilístico* y no como probabilidad clínicamente calibrada: con
{holdout_n_pos} positivos no hay datos para sostener lo segundo, y una tasa observada muy
distinta de la predicha en un bin poco poblado indica falta de datos para estimarla, no
mala calibración:

| Rango predicho | n | Media predicha | Tasa observada |
|---|---|---|---|
{calibration_rows}

Curvas completas: `reports/training/reclass_pr_curve.csv` y `reclass_roc_curve.csv`.

## Uso previsto
Complementa la priorización por probabilidad de patogenicidad
(`src/serve/prioritize_vus.py`): además de cuánto riesgo estimado tiene una VUS, indica
cuál es más probable que se resuelva dentro del horizonte de evaluación, para
decidir qué reanalizar primero cuando
llega evidencia nueva (ADR 007 §5.5).
""", encoding="utf-8")
    print(f"Model Card (reclasificación) -> {card}")


def run_prospective(prospective_release: str | None = None) -> dict:
    """Validación temporal prospectiva: walk-forward sobre una tercera release.

    Reutiliza el modelo ya entrenado y persistido por `run` (ajustado sobre A,
    etiquetado por B), lo aplica sobre las VUS de A que a fecha de B seguían sin
    resolver, y comprueba cuáles se resolvieron en una release C publicada después
    de fijar el par A/B. La verdad terreno de C no existía al entrenar.

    Requiere haber descargado C (`--prospective` en `src.ingest.download`). Solo se
    parsea su VCF para leer CLNSIG: las features ya están fijadas en A.
    """
    cfg = load_config()
    seed = get_seed()
    train_rel = cfg["data"]["clinvar_train_release"]
    test_rel = cfg["data"]["clinvar_test_release"]
    c_rel = prospective_release or cfg["data"].get("clinvar_prospective_release")
    if not c_rel:
        raise RuntimeError("config.data.clinvar_prospective_release no está definido.")

    models_dir = PROJECT_ROOT / "models" / "reclassification_model"
    metrics_path = models_dir / "metrics.json"
    if not models_dir.exists() or not metrics_path.exists():
        raise RuntimeError(
            "No hay modelo de reclasificación entrenado (ejecuta antes `python -m "
            "src.train.train_reclass`). La validación prospectiva reutiliza ese modelo, "
            "no entrena uno nuevo.")
    saved_meta = json.loads(metrics_path.read_text(encoding="utf-8"))

    # Reconstruye la misma población que `run` para identificar qué VUS seguían
    # abiertas (label=0) en B. No altera nada del entrenamiento original.
    labeled = build_reclass_dataset(test_rel)
    X, y = labeled[RECLASS_FEATURE_COLUMNS], labeled["label"]
    still_open = labeled.loc[y == 0].copy()

    # Verdad terreno en C: parsear (no reanotar) su VCF.
    from src.annotate.annotate import parse_clinvar_vcf
    from src.config import chromosomes_subset, raw_dir
    c_path = raw_dir() / f"clinvar_{c_rel}.vcf.gz"
    if not c_path.exists():
        raise RuntimeError(
            f"Falta {c_path}. Descarga la release prospectiva con "
            "`python -m src.ingest.download --prospective` antes de ejecutar esto.")
    c_raw = parse_clinvar_vcf(c_path)
    chroms = chromosomes_subset()
    if chroms:
        c_raw = c_raw[c_raw["chrom"].isin(chroms)].reset_index(drop=True)

    merged = still_open[_KEY + RECLASS_FEATURE_COLUMNS].merge(
        c_raw[_KEY + ["clnsig"]], on=_KEY, how="left")
    merged["label_c"] = is_resolved(merged["clnsig"]).astype(int)
    n_found_in_c = int(merged["clnsig"].notna().sum())

    # Mismo split que `run` (determinista con el seed fijo), solo para reajustar la
    # ablación segura, que `run` no persiste a disco.
    X_train, _X_hold, y_train, _y_hold = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y)

    full_model = mlflow.sklearn.load_model(str(models_dir))
    prob_full = full_model.predict_proba(merged[RECLASS_FEATURE_COLUMNS])[:, 1]

    algorithm = saved_meta["best_algorithm"]
    safe_pipe = Pipeline([("pre", build_safe_reclass_preprocessor()),
                          ("clf", _models(seed)[algorithm])])
    safe_pipe.fit(X_train[SAFE_RECLASS_FEATURE_COLUMNS], y_train)
    prob_safe = safe_pipe.predict_proba(merged[SAFE_RECLASS_FEATURE_COLUMNS])[:, 1]

    y_c = merged["label_c"].to_numpy()
    n_pos_c = int(y_c.sum())
    result = {
        "train_release_a": train_rel, "label_release_b": test_rel,
        "prospective_release_c": c_rel,
        "n_still_open_at_b": len(still_open), "n_found_in_c": n_found_in_c,
        "n_resolved_by_c": n_pos_c,
        "share_resolved_by_c": round(n_pos_c / len(y_c), 4) if len(y_c) else None,
    }
    if n_pos_c >= 2 and n_pos_c <= len(y_c) - 2:
        result["full_model"] = {**compute_metrics(y_c, prob_full),
                                "algorithm": algorithm}
        result["safe_ablation"] = {**compute_metrics(y_c, prob_safe),
                                   "algorithm": algorithm,
                                   "features": SAFE_RECLASS_FEATURE_COLUMNS}
        note = ("Validación prospectiva real: ROC AUC/PR AUC de un modelo ya "
                f"entrenado en {train_rel}->{test_rel}, aplicado sin reentrenar "
                f"sobre verdad terreno de {c_rel}, publicada después.")
    else:
        result["full_model"] = None
        result["safe_ablation"] = None
        note = (f"Muy pocos casos resueltos en {c_rel} ({n_pos_c}) para calcular "
                "ROC AUC/PR AUC con fiabilidad; se reporta solo el recuento.")
    result["note"] = note
    print(f">> Validación prospectiva ({train_rel}->{test_rel}, verdad en {c_rel}): "
          f"{len(still_open)} VUS abiertas en {test_rel}, {n_found_in_c} encontradas en "
          f"{c_rel}, {n_pos_c} resueltas ({result['share_resolved_by_c']}). {note}")
    if result["full_model"]:
        print(f"   Modelo completo: ROC AUC={result['full_model']['roc_auc']:.4f} "
              f"PR AUC={result['full_model']['pr_auc']:.4f}")
        print(f"   Ablación segura: ROC AUC={result['safe_ablation']['roc_auc']:.4f} "
              f"PR AUC={result['safe_ablation']['pr_auc']:.4f}")

    out_path = models_dir / "prospective_metrics.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_prospective_card(result)
    return result


def _write_prospective_card(result: dict) -> None:
    card = PROJECT_ROOT / "docs" / "MODEL_CARD_RECLASSIFICATION_PROSPECTIVE.md"
    card.parent.mkdir(parents=True, exist_ok=True)
    if result["full_model"]:
        body = f"""## Resultado
| Modelo | Features | ROC AUC | PR AUC |
|--------|----------|---------|--------|
| Completo ({result['full_model']['algorithm']}) | todas (RECLASS_FEATURE_COLUMNS) \
| {result['full_model']['roc_auc']:.4f} | {result['full_model']['pr_auc']:.4f} |
| Ablación segura ({result['safe_ablation']['algorithm']}) | \
{', '.join(f'`{f}`' for f in result['safe_ablation']['features'])} \
| {result['safe_ablation']['roc_auc']:.4f} | {result['safe_ablation']['pr_auc']:.4f} |

A diferencia del resto de cifras del modelo, estas proceden de una verdad terreno
publicada después de fijar el par de entrenamiento y nunca usada para entrenar ni
seleccionar: es la única evaluación del proyecto que mide capacidad prospectiva.

Con {result['n_resolved_by_c']} casos positivos -una ventana más corta que la
retrospectiva, con menos tiempo para acumular reclasificaciones- la estimación es muy
ruidosa e insuficiente por sí sola para afirmar o refutar esa capacidad. La reporto sin
redondear al alza ni ocultar el tamaño muestral, porque es la evidencia disponible.
"""
    else:
        body = f"**{result['note']}**\n"
    card.write_text(f"""# Validación temporal prospectiva del modelo de reclasificación

## Diseño
* Entrenamiento, sin cambios: VUS de {result['train_release_a']}, etiquetadas según se
  resuelvan en {result['label_release_b']}.
* Validación: de esas VUS, las que a fecha de {result['label_release_b']} seguían sin
  resolver ({result['n_still_open_at_b']}) se puntúan con el modelo ya entrenado, sin
  reentrenar, y se comprueba si se resolvieron en {result['prospective_release_c']}
  ({result['n_found_in_c']} localizadas, {result['n_resolved_by_c']} resueltas,
  {result['share_resolved_by_c']}).
* {result['prospective_release_c']} se publicó después de fijar el par de entrenamiento y
  no intervino en la selección de algoritmo ni en ningún hiperparámetro.

{body}

## Cómo citar
Es la única cifra del modelo que responde a si predice una reclasificación genuinamente
futura. El resto de métricas (holdout aleatorio dentro del par de entrenamiento) miden
señal retrospectiva y no deben presentarse como equivalentes.
""", encoding="utf-8")
    print(f"Model Card (validación prospectiva) -> {card}")


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Modelo de potencial de reclasificación de VUS.")
    p.add_argument("--prospective", action="store_true",
                   help="Ejecuta la validación temporal prospectiva real (requiere "
                        "haber entrenado antes y descargado la release prospectiva).")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    if args.prospective:
        run_prospective()
    else:
        run()
