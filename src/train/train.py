"""Entrenamiento de 4 algoritmos con tracking automático en MLflow [OE3].

Para cada algoritmo (regresión logística, random forest, gradient boosting,
hist gradient boosting) se entrena un `Pipeline(preprocesador + clasificador)`
sobre el split de train (ClinVar 2023-12) y se evalúa en el split temporal de
test (ClinVar 2025-06). Cada run registra en MLflow: parámetros, métricas
(PR AUC, ROC AUC, F1, intervalo de confianza bootstrap del PR AUC de
holdout...), la matriz de confusión y el propio modelo. Se selecciona el
mejor por **PR AUC de holdout**, se **registra en el Model Registry** (stage
Staging / alias) y se genera su **Model Card** más la importancia de
features.

Uso:
    python -m src.train.train
    python -m src.train.train --tracking-uri sqlite:///mlflow.db
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
from src.features.preprocess import FEATURE_COLUMNS, build_preprocessor

_KEY = ["chrom", "pos", "ref", "alt"]


def unseen_mask(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Máscara booleana: filas de `test_df` cuya clave no aparece en `train_df`.

    Revisión interna: el test temporal contiene variantes que persisten desde
    train (ClinVar es acumulativo entre releases, ver punto 2 de
    la revisión técnica del proyecto); evaluar solo con estas filas mide
    generalización honesta en vez de memorización. Extraída aquí (antes
    duplicada de forma idéntica en `train_ranking.py`, ver punto 7 de la
    misma revisión) para que ambos módulos apliquen exactamente el mismo
    criterio de "no visto" sin poder divergir silenciosamente.
    """
    train_keys = set(map(tuple, train_df[_KEY].to_numpy()))
    unseen = ~pd.Series(list(map(tuple, test_df[_KEY].to_numpy()))).isin(train_keys)
    return unseen.to_numpy()


def data_provenance() -> dict:
    """Procedencia real de los datos de este run (ADR 005 revisado 2026-07-30).

    Antes, ningún artefacto (MLflow, Model Card) registraba si un run se había
    entrenado con ClinVar real o con el generador sintético de fallback: la
    distinción vivía solo en comentarios y en `MANIFEST.json`, fácil de perder
    de vista. Se registra explícitamente en cada run para que un resultado
    nunca pueda citarse como real sin poder verificarlo.
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
    """Los 3 algoritmos estándar del alcance del anteproyecto + 1 candidato del
    estado del arte (revisión técnica del proyecto).

    `hist_gradient_boosting` (histogram-based, inspirado en LightGBM) es más
    rápido que `GradientBoostingClassifier` clásico y admite `NaN` de forma
    nativa en el clasificador -- aquí igualmente recibe la entrada ya
    imputada del `ColumnTransformer` compartido (mismo preprocesador que el
    resto, para que la comparación entre los 4 sea homogénea, ver punto 3),
    así que su ventaja frente a `gradient_boosting` en este pipeline viene
    del *histogram binning* y la regularización, no de una lectura directa
    de `NaN` -- se compara igualmente porque es, aun así, el reemplazo
    directo de menor coste/mayor impacto identificado en la revisión.
    """
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=seed),
        "random_forest": RandomForestClassifier(
            # n_jobs=1: con n_jobs=-1, joblib/loky en Windows deja procesos huérfanos
            # tras cada re-entrenamiento; con decenas de tests reentrenando en la misma
            # sesión, esto degrada el sistema hasta colgarlo. No afecta al modelo
            # resultante (mismos árboles con random_state fijo), solo a la paralelización.
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
            # Backend SQLite local (`config/config.yaml`): habilita el Model Registry sin servidor.
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
        # Stages clásicos (`config/config.yaml`) con fallback a alias (MLflow 3.x).
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
    """Importancia de features por permutación (agnóstica al modelo). Guarda CSV."""
    # La importancia por permutación opera sobre las columnas de ENTRADA (X_test).
    result = permutation_importance(
        pipe, X_test, y_test, n_repeats=5, random_state=seed,
        # n_jobs=1: mismo motivo que en RandomForestClassifier más arriba (procesos
        # huérfanos de joblib/loky en Windows entre tests sucesivos).
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

    # Revisión interna: el test temporal contiene variantes ya presentes en train (persisten
    # entre releases de ClinVar). Para una evaluación HONESTA de la generalización se
    # evalúa también sobre el subconjunto DISJUNTO (variantes no vistas en train) y la
    # SELECCIÓN del mejor modelo usa ese holdout. Ver la revisión interna del proyecto.
    unseen = unseen_mask(train_df, test_df)
    print(f"train={len(X_train)} (prev {y_train.mean():.3f}) | "
          f"test={len(X_test)} (prev {y_test.mean():.3f}) | "
          f"holdout no visto={int(unseen.sum())} ({100 * unseen.mean():.1f}%)")

    art_dir = PROJECT_ROOT / "reports" / "training"
    results: list[dict] = []
    for name, clf in _models(seed).items():
        with mlflow.start_run(run_name=name) as run:
            # Selección del algoritmo por validación cruzada SOLO sobre train
            #: antes se seleccionaba el
            # "mejor" algoritmo por PR AUC del MISMO holdout que después se
            # reportaba como evaluación final -- deja de ser un test
            # verdaderamente independiente (selection bias). El holdout se
            # sigue calculando para los 4 algoritmos por transparencia
            # descriptiva, pero la decisión de cuál es "el mejor" no lo usa.
            cv_pipe = Pipeline([("pre", build_preprocessor()), ("clf", clf)])
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            cv_scores = cross_val_score(
                cv_pipe, X_train, y_train, cv=cv, scoring="average_precision", n_jobs=1)
            cv_pr_auc_mean, cv_pr_auc_std = float(cv_scores.mean()), float(cv_scores.std())

            pipe = Pipeline([("pre", build_preprocessor()), ("clf", clf)])
            pipe.fit(X_train, y_train)
            y_prob = pipe.predict_proba(X_test)[:, 1]
            metrics_full = compute_metrics(y_test, y_prob)               # test completo
            metrics_holdout = compute_metrics(                          # honesto (no visto)
                y_test[unseen], y_prob[unseen])
            # Intervalo de confianza del PR AUC de holdout (punto 5 de
            # la revisión técnica del proyecto): sin él, un ranking determinista por
            # `max` no distingue "este modelo es mejor" de "ganó este holdout
            # concreto por una diferencia pequeña dentro del ruido de muestreo".
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

    # Persistencia del mejor pipeline + Model Card + importancia
    models_dir = PROJECT_ROOT / "models" / "best_model"
    if models_dir.exists():
        import shutil
        shutil.rmtree(models_dir)
    mlflow.sklearn.save_model(
        best["pipe"], str(models_dir),
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE)
    imp_path = _feature_importance(best["pipe"], X_test[unseen], y_test[unseen], seed, art_dir)
    shap_path = explain_model(best["pipe"], X_train, X_test[unseen], art_dir, seed)
    # Calibración del modelo de patogenicidad:
    # se presenta la salida como "probabilidad de patogenicidad" pero, hasta esta
    # revisión, solo el modelo de reclasificación reportaba Brier score/reliability
    # diagram -- el modelo de patogenicidad no, pese a
    # ser el modelo cuya salida se muestra más prominentemente en dashboard/informes.
    best_prob_holdout = best["pipe"].predict_proba(X_test[unseen])[:, 1]
    calibration = calibration_report(y_test[unseen], best_prob_holdout, n_bins=10)
    pd.DataFrame(calibration["bins"]).to_csv(art_dir / "calibration_bins.csv", index=False)
    _write_model_card(best, results, cfg, imp_path, shap_path, provenance, calibration)
    _register_best(best["name"], best["run_id"], best["holdout"])

    # Tabla comparativa de modelos (para la memoria)
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
        "**Datos REALES de extremo a extremo** (ClinVar descargado de NCBI + features de "
        "myvariant.info real). Estos números son citables como resultado del proyecto."
        if provenance["is_real_data"] else
        "**AVISO: este run contiene datos SINTÉTICOS** (generador determinista, ADR 005) — "
        f"ClinVar: `{provenance['clinvar_source']}`, "
        f"features: `{provenance['annotation_source']}`. "
        "Estas métricas validan que el pipeline funciona, NO son un resultado clínico ni deben "
        "citarse como tal en la memoria. Reentrena con `make ingest` (red real) y "
        "`annotation_source: multi_source` en `config/config.yaml` para un resultado citable."
    )
    card.write_text(f"""# Model Card: Clasificador de patogenicidad de variantes

## Procedencia de los datos de este run
{provenance_banner}

## Detalles del modelo
* **Mejor algoritmo:** {best['name']} (seleccionado por PR AUC media en
  validación cruzada de 5 particiones **sobre el conjunto de entrenamiento
  únicamente**, sin tocar el holdout).
* **Tarea:** clasificación binaria patogénica (1) vs benigna (0) de SNVs.
* **Pipeline:** preprocesamiento (imputación + escalado + one-hot) + clasificador, autocontenido.
* **Datos:** entrenamiento ClinVar {cfg['data']['clinvar_train_release']};
  evaluación en split temporal ClinVar {cfg['data']['clinvar_test_release']}.

## Metodología de selección
Hasta esta revisión, el algoritmo "ganador" se elegía por PR AUC sobre el
mismo holdout que después se citaba como su evaluación final: dejaba de ser
un test genuinamente independiente (*selection bias*). Ahora la selección
usa **únicamente** la media de PR AUC en validación cruzada de 5 particiones
estratificadas sobre el conjunto de entrenamiento (columna **CV (train)**
abajo); el holdout no visto se evalúa **una sola vez**, sobre el algoritmo ya
elegido, y es esa evaluación —no la comparación de los 4 algoritmos en el
holdout— la que se cita como resultado final del proyecto. Las columnas de
holdout de los otros tres algoritmos se conservan por transparencia
descriptiva, no como criterio de selección.

## Métricas
La columna **holdout** evalúa solo variantes NO vistas en entrenamiento (evaluación
honesta de generalización). El intervalo junto al PR AUC de holdout es un intervalo
de confianza al 95% por bootstrap (1000 remuestreos): diferencias entre algoritmos
que se solapan en su intervalo no deben leerse como "el ganador es claramente mejor",
solo como el resultado de esta comparación puntual (ver punto 5 de
la revisión técnica del proyecto). El tamaño de este holdout es una consecuencia del
parámetro `data.max_new_variants_per_release` (config.yaml), no un valor elegido por
potencia estadística de esta comparación. La columna **full** incluye variantes que
persisten entre releases (optimista por memorización); se muestra por transparencia.
Ver la revisión interna del proyecto.

| Algoritmo | PR AUC CV (train, 5-fold) | PR AUC (holdout) [IC 95%] | ROC AUC (holdout) \
| F1 (holdout) | PR AUC (full) |
|-----------|----------------------------|---------------------------|--------------------\
|--------------|----------------|
{rows}

## Features
Entradas: scores in silico (CADD, SIFT, PolyPhen, REVEL), conservación (GERP++, phyloP),
frecuencia gnomAD (log) y consecuencia funcional. Los scores ausentes (no missense)
se imputan por mediana con indicador de ausencia.

Dos análisis de importancia, complementarios:
* **Permutación** (`reports/training/feature_importance.csv`): heurística agnóstica al modelo
  por barrido de una feature; rápida, ranking global únicamente.
* **SHAP** (`reports/training/shap_importance.csv` + `shap_summary.png`): valores de
  Shapley sobre el pipeline completo (mismas columnas de entrada); aporta además la
  dirección y magnitud del efecto por instancia, no solo un ranking agregado.

## Calibración
El modelo se presenta como "probabilidad de patogenicidad" en dashboard e informes; el
PR-AUC/ROC-AUC miden discriminación (orden), no calibración (si un score de 0,7 corresponde
de verdad a un 70 % de probabilidad real). Sobre el holdout no visto (misma población que la
sección de Métricas anterior):
**Brier score = {calibration['brier_score']:.4f}** (0 = calibración perfecta; 0,25 es el score
de un clasificador que siempre predice 0,5). Tabla de calibración por deciles (predicción
media frente a tasa observada en cada bin; bins vacíos omitidos):

| Bin de probabilidad predicha | n | Predicción media | Tasa observada |
|---|---|---|---|
{calibration_rows}

Con el desbalance de clases de este holdout, algunos bins de probabilidad alta tienen pocos
casos: la calibración en esos bins es menos fiable que el Brier score global sugiere por sí
solo, y se reporta con esa salvedad explícita en vez de solo el agregado.

## Usos y limitaciones
* **Uso previsto:** el modelo **no** predice ni sustituye el veredicto de variantes que
  ya tienen significado clínico resuelto en ClinVar (Patogénica/Benigna): eso ya está
  disponible, consultarlo. Su objetivo es **dirigir y agilizar la investigación manual
  posterior sobre las VUS** (variantes de significado incierto, sin veredicto): puntúa cada
  VUS con una probabilidad de patogenicidad a partir del conocimiento previo ya construido
  en ClinVar/dbNSFP/gnomAD, y permite **ordenarlas por riesgo estimado** para priorizar cuáles
  revisar primero (ver `src/serve/prioritize_vus.py`, ADR 006). Apoyo a la priorización, no
  un veredicto clínico ni un sustituto de la curación experta (ACMG/AMP).
* **Limitaciones:** solo SNVs; subconjunto de cromosomas en Fase I; si los datos provienen
  del generador offline (ADR 005), las métricas son de validación del pipeline, no clínicas.
* **Ética:** solo bases públicas y agregadas; sin datos genómicos individuales identificables.

## Trazabilidad
Registrado en MLflow (experimento `{cfg['mlflow']['experiment_name']}`,
modelo `{cfg['mlflow']['registered_model_name']}`). Ver `reports/training/model_comparison.csv`.
""", encoding="utf-8")
    print(f"Model Card -> {card}")


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Entrenamiento + tracking MLflow .")
    p.add_argument("--tracking-uri", default=None,
                   help="Override del tracking URI (p. ej. sqlite:///mlflow.db).")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    run(tracking_uri=args.tracking_uri)
