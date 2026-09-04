"""Modelo de patogenicidad: compara cuatro algoritmos y registra el mejor en MLflow.

Cada algoritmo entrena un `Pipeline(preprocesador + clasificador)` sobre la
release antigua y se evalúa sobre la nueva. La selección usa el PR AUC medio en
validación cruzada sobre train; el holdout no visto se evalúa una sola vez,
sobre el algoritmo ya elegido. Salidas: `models/best_model`, Model Card,
importancias y tabla comparativa.

    python -m src.train.train [--tracking-uri sqlite:///mlflow.db]
"""
from __future__ import annotations

import argparse
import json
import socket
import urllib.parse
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from src.config import (
    PROJECT_ROOT,
    annotation_source,
    get_seed,
    load_config,
    processed_dir,
    raw_dir,
)
from src.evaluate.explain import explain_model
from src.evaluate.metrics import (
    bootstrap_pr_auc_ci,
    calibration_report,
    compute_metrics,
    save_confusion_matrix,
)
from src.evaluate.run_registry import record_run
from src.features.preprocess import FEATURE_COLUMNS, build_preprocessor

_KEY = ["chrom", "pos", "ref", "alt"]


def unseen_mask(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Filas de `test_df` cuya clave no aparece en `train_df`.

    ClinVar es acumulativo, así que la release de test conserva buena parte de la
    de train: solo este subconjunto mide generalización. Compartida con
    `train_ranking.py` para que ambos apliquen el mismo criterio.
    """
    train_keys = set(map(tuple, train_df[_KEY].to_numpy()))
    unseen = ~pd.Series(list(map(tuple, test_df[_KEY].to_numpy()))).isin(train_keys)
    return unseen.to_numpy()


def data_provenance() -> dict:
    """Procedencia de los datos del run: ClinVar real o generador sintético (ADR 005).

    Se registra en MLflow y en la Model Card para que ningún resultado pueda
    citarse como real sin poder verificarlo.
    """
    manifest_path = raw_dir() / "MANIFEST.json"
    clinvar_source = "desconocida (MANIFEST.json no encontrado, ejecuta `make ingest`)"
    if manifest_path.exists():
        try:
            clinvar_source = json.loads(
                manifest_path.read_text(encoding="utf-8")).get("source", "desconocida")
        except (json.JSONDecodeError, OSError):
            pass
    ann_source = annotation_source()
    real = clinvar_source == "ncbi_clinvar" and ann_source == "multi_source"
    return {"clinvar_source": clinvar_source, "annotation_source": ann_source, "is_real_data": real}


def _models(seed: int) -> dict[str, object]:
    """Los tres algoritmos del anteproyecto más `hist_gradient_boosting`.

    Los cuatro reciben el mismo preprocesador ya imputado, para que la
    comparación sea homogénea: la ventaja de la variante por histogramas viene
    del binning y la regularización, no del manejo nativo de NaN.
    """
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=seed),
        "random_forest": RandomForestClassifier(
            # n_jobs=1: con -1, loky deja procesos huérfanos en Windows entre tests
            # sucesivos. Con random_state fijo el modelo resultante es idéntico.
            n_estimators=300, max_depth=None, n_jobs=1, random_state=seed),
        "gradient_boosting": GradientBoostingClassifier(random_state=seed),
        "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=seed),
    }


def _resolve_tracking_uri(cfg: dict, override: str | None) -> str:
    """Usa el override/config; si es un servidor HTTP no accesible, cae a file store."""
    uri = override or cfg["mlflow"]["tracking_uri"]
    if uri.startswith("http"):
        parsed = urllib.parse.urlparse(uri)
        host, port = parsed.hostname or "localhost", parsed.port or 80
        try:
            with socket.create_connection((host, port), timeout=1.5):
                return uri
        except OSError:
            # SQLite local: habilita el Model Registry sin servidor.
            local = f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
            print(f"MLflow server {uri} no accesible -> tracking local {local}")
            return local
    return uri


def _register_best(name: str, run_id: str, metrics: dict) -> None:
    """Registra el mejor modelo en el Model Registry con stage/alias (con fallback)."""
    cfg = load_config()
    model_name = cfg["mlflow"]["registered_model_name"]
    try:
        result = mlflow.register_model(f"runs:/{run_id}/model", model_name)
        client = mlflow.tracking.MlflowClient()
        # Stages clásicos, con fallback a alias en MLflow 3.x.
        try:
            client.transition_model_version_stage(
                model_name, result.version, stage="Staging",
                archive_existing_versions=True)
            print(f"Registrado {model_name} v{result.version} -> stage Staging")
        except Exception:
            client.set_registered_model_alias(model_name, "staging", result.version)
            print(f"Registrado {model_name} v{result.version} -> alias @staging")
    except Exception as exc:  # p. ej. backend sin soporte de registry
        print(f"[aviso] Model Registry no disponible en este backend: {exc}. "
              f"El mejor modelo ({name}) queda guardado como artefacto y en models/.")


def _feature_importance(pipe: Pipeline, X_test, y_test, seed: int, out_dir: Path) -> Path:
    """Importancia por permutación sobre las columnas de entrada. Guarda CSV."""
    result = permutation_importance(
        pipe, X_test, y_test, n_repeats=5, random_state=seed,
        # n_jobs=1: mismo motivo que en RandomForestClassifier.
        scoring="average_precision", n_jobs=1)
    imp = (pd.DataFrame({"feature": list(X_test.columns),
                         "importance": result.importances_mean})
           .sort_values("importance", ascending=False))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "feature_importance.csv"
    imp.to_csv(path, index=False)
    print("Top features (permutación):", ", ".join(imp.head(5)["feature"]))
    return path


def run(tracking_uri: str | None = None) -> dict:
    cfg = load_config()
    seed = get_seed()
    mlflow.set_tracking_uri(_resolve_tracking_uri(cfg, tracking_uri))
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    train_df = pd.read_parquet(processed_dir() / "train.parquet")
    test_df = pd.read_parquet(processed_dir() / "test.parquet")
    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["label"].astype(int)
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df["label"].astype(int)

    provenance = data_provenance()
    print(f"Procedencia de datos: ClinVar={provenance['clinvar_source']} | "
          f"features={provenance['annotation_source']} | "
          + ("REAL de extremo a extremo."
             if provenance["is_real_data"] else
             "AVISO: contiene datos SINTÉTICOS (generador ADR 005) — no citar "
             "estos números como resultado clínico ni de la memoria."))

    unseen = unseen_mask(train_df, test_df)
    print(f"train={len(X_train)} (prev {y_train.mean():.3f}) | "
          f"test={len(X_test)} (prev {y_test.mean():.3f}) | "
          f"holdout no visto={int(unseen.sum())} ({100 * unseen.mean():.1f}%)")

    art_dir = PROJECT_ROOT / "reports" / "training"
    results: list[dict] = []
    for name, clf in _models(seed).items():
        with mlflow.start_run(run_name=name) as run:
            # La selección usa solo la CV sobre train: elegir por el mismo holdout
            # que después se cita como evaluación final es sesgo de selección. El
            # holdout se calcula para los cuatro, pero no decide.
            cv_pipe = Pipeline([("pre", build_preprocessor()), ("clf", clf)])
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            cv_scores = cross_val_score(
                cv_pipe, X_train, y_train, cv=cv, scoring="average_precision", n_jobs=1)
            cv_pr_auc_mean, cv_pr_auc_std = float(cv_scores.mean()), float(cv_scores.std())

            pipe = Pipeline([("pre", build_preprocessor()), ("clf", clf)])
            pipe.fit(X_train, y_train)
            y_prob = pipe.predict_proba(X_test)[:, 1]
            metrics_full = compute_metrics(y_test, y_prob)                    # release completa
            metrics_holdout = compute_metrics(y_test[unseen], y_prob[unseen])  # no visto
            holdout_ci = bootstrap_pr_auc_ci(y_test[unseen], y_prob[unseen], seed=seed)

            mlflow.log_params({"algorithm": name, "seed": seed,
                               **{f"clf__{k}": v for k, v in clf.get_params().items()
                                  if isinstance(v, (int, float, str, bool)) or v is None}})
            mlflow.log_metrics({"cv_pr_auc_mean": cv_pr_auc_mean, "cv_pr_auc_std": cv_pr_auc_std})
            mlflow.log_metrics({f"full_{k}": v for k, v in metrics_full.items()})
            mlflow.log_metrics({f"holdout_{k}": v for k, v in metrics_holdout.items()})
            mlflow.log_metrics({f"holdout_{k}": v for k, v in holdout_ci.items()})
            mlflow.log_param("clinvar_train_release", cfg["data"]["clinvar_train_release"])
            mlflow.log_param("clinvar_test_release", cfg["data"]["clinvar_test_release"])
            mlflow.log_param("clinvar_data_source", provenance["clinvar_source"])
            mlflow.log_param("annotation_source", provenance["annotation_source"])
            mlflow.set_tag("real_data_end_to_end", provenance["is_real_data"])
            cm_path = save_confusion_matrix(
                y_test[unseen], y_prob[unseen], art_dir / f"cm_{name}.png",
                title=f"CM · {name} (holdout no visto)")
            mlflow.log_artifact(str(cm_path), artifact_path="figures")
            mlflow.sklearn.log_model(
                pipe, name="model",
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE)

            results.append({"name": name, "run_id": run.info.run_id, "pipe": pipe,
                            "holdout": metrics_holdout, "full": metrics_full,
                            "holdout_ci": holdout_ci,
                            "cv_pr_auc_mean": cv_pr_auc_mean, "cv_pr_auc_std": cv_pr_auc_std})
            print(f"[{name}] CV (train, 5-fold) PR AUC={cv_pr_auc_mean:.4f}"
                  f"±{cv_pr_auc_std:.4f} | holdout PR AUC={metrics_holdout['pr_auc']:.4f} "
                  f"[{holdout_ci['pr_auc_ci_low']:.4f}, {holdout_ci['pr_auc_ci_high']:.4f}] "
                  f"F1={metrics_holdout['f1']:.4f} | full PR AUC={metrics_full['pr_auc']:.4f}")

    best = max(results, key=lambda r: r["cv_pr_auc_mean"])
    print(f"\n>> Mejor modelo (seleccionado por CV en train, no por el holdout): {best['name']} "
          f"(CV PR AUC={best['cv_pr_auc_mean']:.4f}; "
          f"holdout PR AUC={best['holdout']['pr_auc']:.4f}, evaluación final única)")

    models_dir = PROJECT_ROOT / "models" / "best_model"
    if models_dir.exists():
        import shutil
        shutil.rmtree(models_dir)
    mlflow.sklearn.save_model(
        best["pipe"], str(models_dir),
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE)
    imp_path = _feature_importance(best["pipe"], X_test[unseen], y_test[unseen], seed, art_dir)
    shap_path = explain_model(best["pipe"], X_train, X_test[unseen], art_dir, seed)
    # La salida se presenta como probabilidad, así que se mide también calibración.
    best_prob_holdout = best["pipe"].predict_proba(X_test[unseen])[:, 1]
    calibration = calibration_report(y_test[unseen], best_prob_holdout, n_bins=10)
    pd.DataFrame(calibration["bins"]).to_csv(art_dir / "calibration_bins.csv", index=False)
    _write_model_card(best, results, cfg, imp_path, shap_path, provenance, calibration)
    record_run("variant_pathogenicity", best["run_id"], algorithm=best["name"])
    _register_best(best["name"], best["run_id"], best["holdout"])

    comp = pd.DataFrame([{
        "name": r["name"],
        "cv_pr_auc_mean": r["cv_pr_auc_mean"], "cv_pr_auc_std": r["cv_pr_auc_std"],
        "holdout_pr_auc": r["holdout"]["pr_auc"], "holdout_roc_auc": r["holdout"]["roc_auc"],
        "holdout_f1": r["holdout"]["f1"],
        "holdout_pr_auc_ci_low": r["holdout_ci"]["pr_auc_ci_low"],
        "holdout_pr_auc_ci_high": r["holdout_ci"]["pr_auc_ci_high"],
        "full_pr_auc": r["full"]["pr_auc"], "full_f1": r["full"]["f1"],
    } for r in results])
    comp.to_csv(art_dir / "model_comparison.csv", index=False)
    return {"best": best["name"], "results": results, "artifacts": art_dir}


def _write_model_card(
    best: dict, results: list, cfg: dict, imp_path: Path, shap_path: Path, provenance: dict,
    calibration: dict,
) -> None:
    card = PROJECT_ROOT / "docs" / "MODEL_CARD.md"
    card.parent.mkdir(parents=True, exist_ok=True)
    rows = "\n".join(
        f"| {r['name']} | {r['cv_pr_auc_mean']:.4f}±{r['cv_pr_auc_std']:.4f} "
        f"| {r['holdout']['pr_auc']:.4f} "
        f"[{r['holdout_ci']['pr_auc_ci_low']:.4f}, {r['holdout_ci']['pr_auc_ci_high']:.4f}] "
        f"| {r['holdout']['roc_auc']:.4f} "
        f"| {r['holdout']['f1']:.4f} | {r['full']['pr_auc']:.4f} |" for r in results)
    calibration_rows = "\n".join(
        f"| [{b['bin_low']}, {b['bin_high']}) | {b['n']} | {b['mean_predicted']} "
        f"| {b['observed_rate']} |" for b in calibration["bins"] if b["n"])
    provenance_banner = (
        "Datos reales de extremo a extremo: ClinVar descargado del NCBI y features de "
        "myvariant.info. Estas cifras son citables como resultado del proyecto."
        if provenance["is_real_data"] else
        "**Aviso: esta ejecución contiene datos sintéticos** (generador determinista, "
        f"ADR 005). ClinVar: `{provenance['clinvar_source']}`, features: "
        f"`{provenance['annotation_source']}`. Validan que el pipeline funciona, no son un "
        "resultado clínico. Reejecuta con red real y `annotation_source: multi_source` "
        "para obtener cifras citables."
    )
    card.write_text(f"""# Model Card: Clasificador de patogenicidad de variantes

## Procedencia
{provenance_banner}

## Modelo
* **Algoritmo:** {best['name']}.
* **Tarea:** clasificación binaria patogénica (1) / benigna (0) de SNVs.
* **Pipeline:** imputación + escalado + one-hot y clasificador, autocontenido.
* **Datos:** entrenamiento ClinVar {cfg['data']['clinvar_train_release']}, evaluación
  temporal sobre ClinVar {cfg['data']['clinvar_test_release']}.

## Selección y evaluación
Elijo el algoritmo por PR AUC medio en validación cruzada de 5 particiones
estratificadas **sobre el conjunto de entrenamiento** (columna CV). El holdout no
visto se evalúa una sola vez, sobre el algoritmo ya elegido: seleccionar y reportar
sobre el mismo conjunto introduce sesgo de selección. Las columnas de holdout del
resto de algoritmos son descriptivas, no criterio de decisión.

El holdout contiene solo variantes ausentes del entrenamiento; su tamaño lo fija
`data.max_new_variants_per_release`, no un cálculo de potencia estadística. El
intervalo es bootstrap al 95 % (1000 remuestreos): dos algoritmos con intervalos
solapados no son distinguibles con esta muestra. La columna *full* incluye variantes
que persisten entre releases y es optimista por memorización; se muestra por
transparencia.

| Algoritmo | PR AUC CV (train, 5-fold) | PR AUC (holdout) [IC 95%] | ROC AUC (holdout) \
| F1 (holdout) | PR AUC (full) |
|-----------|----------------------------|---------------------------|--------------------\
|--------------|----------------|
{rows}

## Features
Scores in silico (CADD, SIFT, PolyPhen, REVEL, AlphaMissense), conservación (GERP++,
phyloP), frecuencia de gnomAD en escala logarítmica y consecuencia funcional. Los
valores ausentes se imputan por mediana conservando un indicador de ausencia, que es
informativo en sí mismo. `gene` se excluye por alta cardinalidad y riesgo de fuga.

La importancia se mide de dos formas complementarias: por permutación
(`reports/training/feature_importance.csv`), que da un ranking global agnóstico al
modelo, y por SHAP (`shap_importance.csv`, `shap_summary.png`), que añade dirección y
magnitud del efecto por instancia.

## Calibración
PR AUC y ROC AUC miden discriminación, no calibración: que un score de 0,7 corresponda
de verdad a un 70 % de probabilidad. Sobre el mismo holdout, **Brier score =
{calibration['brier_score']:.4f}** (0 es calibración perfecta; 0,25, el de un clasificador
que siempre predice 0,5). Predicción media frente a tasa observada por decil, omitiendo
bins vacíos:

| Bin de probabilidad predicha | n | Predicción media | Tasa observada |
|---|---|---|---|
{calibration_rows}

Con este desbalance, los bins de probabilidad alta tienen pocos casos: su calibración es
menos fiable de lo que sugiere el Brier score global.

## Uso y limitaciones
* **Uso previsto:** ordenar VUS por riesgo estimado para dirigir la revisión manual
  (`src/serve/prioritize_vus.py`, ADR 006). No predice ni sustituye el veredicto de las
  variantes que ClinVar ya ha resuelto, ni la curación experta ACMG/AMP.
* **Limitaciones:** solo SNVs, cromosomas 1-3. Con el generador offline (ADR 005) las
  métricas validan el pipeline, no tienen valor clínico.
* **Ética:** solo bases públicas agregadas, sin datos genómicos identificables.

## Trazabilidad
Registrado en MLflow (experimento `{cfg['mlflow']['experiment_name']}`,
modelo `{cfg['mlflow']['registered_model_name']}`). Ver `reports/training/model_comparison.csv`.
""", encoding="utf-8")
    print(f"Model Card -> {card}")


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Modelo de patogenicidad con tracking MLflow.")
    p.add_argument("--tracking-uri", default=None,
                   help="Override del tracking URI (p. ej. sqlite:///mlflow.db).")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    run(tracking_uri=args.tracking_uri)
