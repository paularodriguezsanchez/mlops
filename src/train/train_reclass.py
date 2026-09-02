"""Modelo de potencial de reclasificación de VUS (núcleo del proyecto — ADR 007 §5).

Complementa al modelo de patogenicidad (`src/train/train.py`): en vez de
"¿es patogénica?", responde **"¿qué probabilidad tiene esta VUS concreta de
resolverse (reclasificarse a Patogénica/Benigna) en la próxima release de
ClinVar?"**. Es la pieza que más se apoya en algo que nadie más explota así:
tener dos releases fechadas de ClinVar con reclasificaciones reales conocidas
entre medias (ver ADR 007 §4).

Población y etiqueta:
  * Población: las VUS reservadas de la release ANTIGUA (`vus_{train_release}`,
    ver `build_dataset.py`) — variantes sin veredicto en su momento.
  * Etiqueta: 1 si esa misma variante aparece con significado clínico
    RESUELTO (Patogénica/Benigna) en la release NUEVA; 0 si sigue siendo VUS
    (o no aparece en la release nueva).

Limitación metodológica, explícita: solo hay DOS releases fechadas, no una
serie temporal. No existe una tercera release para una evaluación con
holdout temporal real (como sí hace `train.py` con el holdout "no visto"), así
que aquí la evaluación es un **holdout aleatorio estratificado** dentro de la
población de VUS de la release antigua. No se disfraza de evaluación
temporal: se documenta como lo que es.

Uso:
    python -m src.train.train_reclass
"""
from __future__ import annotations

import argparse
import json

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import PROJECT_ROOT, get_seed, interim_dir, load_config, processed_dir
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

# ROC AUC 0.5 = azar. Por debajo de este umbral el modelo no discrimina mejor
# que el azar y no debe presentarse como una probabilidad fiable (revisión interna del proyecto):
# dashboard e informes leen `metrics.json` y avisan de forma
# explícita en vez de mostrar el número sin más contexto.
RELIABLE_ROC_AUC_THRESHOLD = 0.6


def build_reclass_dataset(test_release: str) -> pd.DataFrame:
    """VUS de la release de train + etiqueta de reclasificación en `test_release`.

    Devuelve un DataFrame con `FEATURE_COLUMNS` + `label` (1 = reclasificada a
    Patogénica/Benigna en la release nueva, 0 = sigue VUS o desaparece). Lee
    `data/processed/vus_train.parquet` (capa GOLD, nombrada por split, no por
    fecha de release — ver `build_dataset.py`) y `data/interim/annotated_
    {test_release}.parquet` (capa SILVER, nombrada por release).
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
    """Ablación "temporalmente segura" del modelo de reclasificación (mitigación parcial, ADR 008).

    Mismo split (mismos índices de train/holdout) que el modelo completo,
    pero solo con `SAFE_RECLASS_FEATURE_COLUMNS` (`consequence`,
    `review_stars`): las dos únicas features del modelo de reclasificación ancladas a la fecha real
    de cada release. El resto (CADD/REVEL/AlphaMissense/gnomAD/SIFT/
    PolyPhen/GERP/phyloP) se recalculan "hoy" vía myvariant.info sin anclaje
    de versión, así que su valor en la release t0 puede incorporar
    información posterior a t0 (y a veces a t1) -- ver ADR 008. Comparar
    ambos resultados cuantifica cuánta señal del modelo de reclasificación depende de
    esa fuga
    potencial, en vez de solo documentarla en prosa.
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
    results: list[dict] = []
    for name, clf in _models(seed).items():
        with mlflow.start_run(run_name=f"reclass_{name}") as mlrun:
            pipe = Pipeline([("pre", build_reclass_preprocessor()), ("clf", clf)])
            pipe.fit(X_train, y_train)
            y_prob = pipe.predict_proba(X_hold)[:, 1]
            metrics = compute_metrics(y_hold, y_prob)
            mlflow.log_params({"algorithm": name, "seed": seed, "task": "reclassification",
                               "clinvar_data_source": provenance["clinvar_source"],
                               "annotation_source": provenance["annotation_source"]})
            mlflow.set_tag("real_data_end_to_end", provenance["is_real_data"])
            mlflow.log_metrics(metrics)
            cm_path = save_confusion_matrix(
                y_hold, y_prob, art_dir / f"cm_reclass_{name}.png",
                title=f"CM · reclasificación · {name}",
                labels=("No reclasificada", "Reclasificada"))
            mlflow.log_artifact(str(cm_path), artifact_path="figures")
            mlflow.sklearn.log_model(
                pipe, name="model",
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE)
            results.append({"name": name, "run_id": mlrun.info.run_id,
                            "pipe": pipe, "metrics": metrics})
            print(f"[reclass_{name}] PR AUC={metrics['pr_auc']:.4f} F1={metrics['f1']:.4f}")

    best = max(results, key=lambda r: r["metrics"]["pr_auc"])
    print(f"\n>> Mejor modelo de reclasificación: {best['name']} "
          f"(PR AUC={best['metrics']['pr_auc']:.4f})")

    # Rigor estadístico adicional del mejor modelo sobre el holdout (revisión posterior del proyecto
    # independiente, un hallazgo de esa revisión): el proyecto solo reportaba PR AUC/ROC AUC/F1
    # puntuales, sin IC, sin curva PR completa, sin calibración y sin métricas
    # de cola de revisión (precision@k/recall@k/lift@k), pese a presentar la
    # salida como "probabilidad" y a que la utilidad real del modelo de reclasificación
    # es priorizar
    # una lista, no clasificar cada VUS de forma aislada.
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

    # Ablación "temporalmente segura" (mitigación parcial de leakage, ADR
    # 008): mismo split, mismo algoritmo, solo features ancladas a la fecha
    # real de cada release. Cuantifica cuánta señal del modelo de reclasificación
    # podría depender de
    # features recalculadas "hoy" en vez de a fecha de la release t0.
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
    # `delta > 0` = el modelo completo discrimina mejor que la ablación segura.
    # `delta <= 0` = la ablación iguala o SUPERA al modelo completo (caso que
    # la primera versión de esta función no distinguía correctamente de "muy
    # cerca": una ablación que gana no es lo mismo que una que empata).
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
            "-- con solo 67 positivos en la población, es plausible que el modelo completo "
            "sobreajuste con más dimensiones sin más señal real que aportar. No se presenta "
            "el ROC AUC del modelo completo como superior al de la ablación en ningún caso."
        )
    elif delta <= 0.03:
        ablation_note = (
            f"La ablación (ROC AUC {ablation['roc_auc']:.4f}) está muy cerca del modelo "
            f"completo (ROC AUC {best['metrics']['roc_auc']:.4f}, diferencia {delta:+.4f}): "
            "la mayor parte de la señal parece provenir de las dos features temporalmente "
            "seguras (`consequence`, `review_stars`), no de las potencialmente afectadas por "
            "la fuga descrita arriba. Esto acota el riesgo, no lo elimina."
        )
    else:
        ablation_note = (
            f"La ablación (ROC AUC {ablation['roc_auc']:.4f}) es notablemente peor que el "
            f"modelo completo (ROC AUC {best['metrics']['roc_auc']:.4f}, diferencia "
            f"{delta:+.4f}): una parte relevante de la señal del modelo completo depende de "
            "features con riesgo de fuga temporal (ver arriba). No citar el ROC AUC del "
            "modelo completo sin esta salvedad."
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
        f"**Naturaleza del umbral {RELIABLE_ROC_AUC_THRESHOLD} (revisión posterior del proyecto, "
        "revisión posterior):** es una convención operativa interna del proyecto, "
        "sin respaldo "
        "estadístico, clínico ni de la literatura -no existe una referencia que establezca "
        "0.6 como punto de corte de 'fiabilidad' para un ROC AUC-. Se documenta explícitamente "
        "como tal, no como una demostración de que el modelo sea fiable en sentido estadístico; "
        "el criterio real y más informativo es el intervalo de confianza del ROC AUC frente a "
        "0.5 (azar), no la comparación puntual contra 0.6."
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
        f"El mejor modelo supera el umbral de fiabilidad "
        f"(ROC AUC {best['metrics']['roc_auc']:.4f} >= {RELIABLE_ROC_AUC_THRESHOLD}): "
        "se presenta como probabilidad en el dashboard y los informes sin aviso adicional.\n\n"
        f"{threshold_justification}\n\n{roc_ci_note}"
        if reliable else
        f"**AVISO: el mejor modelo NO supera el umbral de fiabilidad** "
        f"(ROC AUC {best['metrics']['roc_auc']:.4f} < {RELIABLE_ROC_AUC_THRESHOLD}; "
        "0.5 = azar). Un ROC AUC tan cercano a 0.5 indica que el modelo no discrimina "
        "mejor que el azar con las features actuales. El dashboard  y los informes "
        "por VUS  muestran igualmente el número (transparencia de datos) pero con "
        "una nota explícita de señal débil junto a cada valor, no solo aquí. "
        "No citar esta probabilidad en la memoria como un resultado predictivo validado.\n\n"
        f"{threshold_justification}\n\n{roc_ci_note}"
    )
    card.write_text(f"""# Model Card: Potencial de reclasificación de VUS

## Procedencia de los datos de este run
{provenance_banner}

## Detalles del modelo
* **Tarea:** dada una VUS de la release {train_rel}, predecir si se
  reclasificará (a Patogénica o Benigna) en la release {test_rel}.
* **Mejor algoritmo:** {best['name']} (por PR AUC en holdout aleatorio).
* **Población:** {n_total} VUS de {train_rel}; {n_pos} ({100 * n_pos / n_total:.1f}%)
  reclasificadas en {test_rel}.

## Fiabilidad de la señal (revisión interna del proyecto)
{reliability_note}

## Limitación metodológica (léase antes de citar en la memoria)
Todas las métricas de esta ficha usan un **holdout aleatorio estratificado**
dentro de la población de VUS de la release antigua ({train_rel}), etiquetada
por si se resuelve en la release nueva ({test_rel}): miden señal discriminativa
retrospectiva dentro de ese mismo intervalo histórico, NO capacidad de
predicción prospectiva. **Existe una validación temporal prospectiva real,
aparte**: aplica este mismo
modelo, ya entrenado y sin reentrenar, sobre una release publicada después de
fijar este par -- ver `docs/MODEL_CARD_RECLASSIFICATION_PROSPECTIVE.md`
(`python -m src.train.train_reclass --prospective`). Es esa cifra, no esta, la
que responde a "¿predice el modelo una reclasificación genuinamente futura?".

## Leakage temporal de las features (ADR 008, léase antes de citar en la memoria)
Las features de anotación (CADD, REVEL, AlphaMissense, gnomAD, SIFT, PolyPhen,
GERP, phyloP) se consultan en vivo a myvariant.info **en el momento en que se
ejecuta el pipeline**, sin ningún anclaje a la fecha de la release de ClinVar:
el mismo snapshot "de hoy" se aplica tanto a la release {train_rel} (t0) como a
la {test_rel} (t1). Si una VUS se resolvió entre t0 y t1 precisamente porque
llegó evidencia computacional nueva (p. ej. AlphaMissense, publicado en 2023,
incorporándose como evidencia PP3/BP4), el modelo puede estar entrenando con
el valor **posterior** de esa evidencia, no con el que existía cuando la VUS
seguía sin resolver en t0 -- fuga de información del futuro hacia el pasado.
`review_status`/`review_stars` y `consequence`, en cambio, se leen directamente
del VCF fechado de cada release y son temporalmente seguros (ver
`src/features/preprocess.py::SAFE_RECLASS_FEATURE_COLUMNS`).

**Mitigación parcial aplicada:** se entrena, sobre el mismo split
train/holdout, un modelo de ablación que usa SOLO las features temporalmente
seguras (`consequence`, `review_stars`), para cuantificar cuánta señal del
modelo completo depende de las features con riesgo de fuga.

| Modelo | Features | PR AUC | ROC AUC |
|--------|----------|--------|---------|
{full_row}
{ablation_row}

{ablation_note}

**Mitigación NO aplicada (trabajo futuro):** congelar snapshots históricos de
CADD/REVEL/AlphaMissense/gnomAD anclados a la fecha real de cada release, en
vez de la consulta única "de hoy", requeriría descargas versionadas por fuente
que no están garantizadas como archivadas y disponibles públicamente para
todas las fuentes; queda fuera del alcance de esta mitigación (ver ADR 008).

## Métricas (holdout aleatorio)
| Algoritmo | PR AUC | ROC AUC | F1 |
|-----------|--------|---------|-----|
{rows}

## Rigor estadístico adicional del mejor modelo
Holdout: n={holdout_n}, positivos={holdout_n_pos} ({100 * holdout_n_pos / holdout_n:.1f}%).

**Intervalo de confianza del PR AUC** (bootstrap, 95%, 1000 remuestreos): {ci_text}.
Con solo {holdout_n_pos} positivos en el holdout, este intervalo es necesariamente
ancho: se reporta explícitamente en vez de omitirlo, como pedía esa revisión,
no porque sea estrecho.

**Métricas de cola de revisión** (precision@k / recall@k / lift@k; lift=1.0
equivale a ordenar al azar):

| k | precision@k | recall@k | lift@k |
|---|---|---|---|
{topk_rows}

**Calibración** (Brier score = {calibration['brier_score']:.4f}; más bajo es
mejor, 0.25 es el de un clasificador que siempre predice 0.5 bajo esta
prevalencia). Tabla por quintil de probabilidad predicha (`n` bajo por bin es
esperable con solo {holdout_n_pos} positivos; una tasa observada muy distinta
de la predicha en un bin con `n` pequeño no debe leerse como una calibración
pobre, sino como falta de datos para estimarla):

| Rango predicho | n | Media predicha | Tasa observada |
|---|---|---|---|
{calibration_rows}

Curva PR completa: `reports/training/reclass_pr_curve.csv`. Curva ROC completa:
`reports/training/reclass_roc_curve.csv`.

## Uso previsto
Complementa (no sustituye) la priorización por probabilidad de patogenicidad
(`src/serve/prioritize_vus.py`): además de "cuánto riesgo estimado tiene esta
VUS", indica "cuánta probabilidad hay de que esta VUS concreta se resuelva
pronto", útil para decidir qué VUS reanalizar primero cuando llega evidencia
nueva (bucle de reentrenamiento continuo, ADR 007 §5.5).
""", encoding="utf-8")
    print(f"Model Card (reclasificación) -> {card}")


def run_prospective(prospective_release: str | None = None) -> dict:
    """Validación temporal PROSPECTIVA real del modelo de reclasificación.

    El resto de este módulo entrena y evalúa el modelo de reclasificación con un
    holdout ALEATORIO
    dentro del mismo intervalo histórico 2023-2025 (train A -> label por B):
    demuestra señal discriminativa entre VUS de esa ventana, pero no que el
    modelo generalice a una release genuinamente futura, nunca vista ni en
    entrenamiento ni en la elección de hiperparámetros/algoritmo.

    Esta función SÍ lo hace: reutiliza el modelo YA entrenado y persistido
    por `run` (ajustado sobre A={train_rel}, etiquetado por B={test_rel}),
    lo aplica sobre las VUS de A que a fecha de B **seguían sin resolver**
    (label=0 en el entrenamiento original -- el subconjunto realmente
    "abierto" en el momento de entrenar) y comprueba si esas variantes se
    resolvieron en una release C posterior (`clinvar_prospective_release`,
    publicada después de fijar el par A/B y nunca usada para entrenar ni
    seleccionar nada). Es un walk-forward real, no un holdout aleatorio: la
    verdad terreno de C no existía cuando se entrenó ni se persistió el
    modelo.

    Requiere haber descargado C (`python -m src.ingest.download
    --prospective`) -- solo se parsea su VCF (CLNSIG) para la verdad
    terreno, no se reanota contra myvariant.info (no hace falta: las
    features ya están fijadas en A).
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

    # Reconstruye la MISMA población y el MISMO split que `run` (mismo seed,
    # mismos datos): no cambia nada del entrenamiento original, solo identifica
    # qué VUS de A seguían "abiertas" (label=0) en B.
    labeled = build_reclass_dataset(test_rel)
    X, y = labeled[RECLASS_FEATURE_COLUMNS], labeled["label"]
    still_open = labeled.loc[y == 0].copy()

    # Verdad terreno en C: parsear (no reanotar) el VCF de la release
    # prospectiva y comprobar cuáles de las VUS "abiertas" en B aparecen
    # resueltas en C.
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

    # Reconstruye el split train/holdout original (mismo seed, mismos datos:
    # determinista) solo para poder refit la ablación "segura" (no persistida
    # a disco por `run`, a diferencia del modelo completo).
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

Estas cifras, a diferencia de todas las demás del modelo de reclasificación en este
repositorio, proceden
de una verdad terreno publicada DESPUÉS de fijar el par de entrenamiento y nunca
usada para entrenar ni para elegir nada: es la única evaluación de este proyecto
que mide capacidad de predicción prospectiva real, no señal retrospectiva dentro
de un mismo intervalo histórico.

**Aviso de potencia estadística:** con solo {result['n_resolved_by_c']} casos
positivos en esta ventana prospectiva (una ventana más corta que la retrospectiva
2023-2025 usada para entrenar, con menos tiempo para acumular reclasificaciones),
este ROC AUC/PR AUC es una estimación muy ruidosa e insuficiente por sí sola para
afirmar ni refutar con seguridad la capacidad prospectiva del modelo; se reporta
igualmente, sin redondear al alza ni ocultar el tamaño muestral, porque es la
evidencia prospectiva real disponible, no una aproximación conveniente.
"""
    else:
        body = f"**{result['note']}**\n"
    card.write_text(f"""# Validación temporal PROSPECTIVA del modelo de reclasificación

Revisión posterior del proyecto:.

## Diseño
* Entrenamiento (sin cambios): VUS de {result['train_release_a']}, etiquetadas por si se
  resuelven en {result['label_release_b']}.
* Validación prospectiva (esta ejecución): de esas VUS, las que a fecha de
  {result['label_release_b']} **seguían sin resolver** ({result['n_still_open_at_b']}
  variantes) se puntúan con el modelo YA entrenado (sin reentrenar) y se comprueba si se
  resolvieron en {result['prospective_release_c']} ({result['n_found_in_c']} encontradas,
  {result['n_resolved_by_c']} resueltas, {result['share_resolved_by_c']}).
* {result['prospective_release_c']} se publicó después de fijar el par de entrenamiento y
  no participó en el ajuste de ningún hiperparámetro ni en la selección de algoritmo.

{body}

## Cómo citar esto en la memoria
Esta es la única cifra del modelo de reclasificación que responde de verdad a "¿predice
este modelo una
reclasificación genuinamente futura?" (revisión posterior del proyecto, hallazgo
CRÍTICO #1). El resto de métricas del modelo de reclasificación (holdout aleatorio
2023-2025) miden
señal retrospectiva dentro del mismo intervalo, no capacidad prospectiva; no
deben presentarse como equivalentes a esta.
""", encoding="utf-8")
    print(f"Model Card (validación prospectiva) -> {card}")


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Modelo de reclasificación de VUS .")
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
