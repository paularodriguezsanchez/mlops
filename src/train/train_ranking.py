"""Objetivo de ranking para la priorización de VUS (ADR 007 §5).

El entregable real es un orden de prioridad, no una probabilidad aislada, así que
entreno directamente sobre esa función objetivo (LightGBM `lambdarank`) y evalúo
con NDCG@k en vez de con métricas de clasificación.

LambdaMART agrupa ítems por *query*; aquí no hay agrupación natural, así que uso
un único grupo global. Es una simplificación, documentada como tal.

El booster y el preprocesador ajustado se persisten en `models/ranking_model/` y
los carga `src/serve/prioritize_vus.py::load_ranking_model`, que es quien decide
el orden real servido, con reversión a la probabilidad de patogenicidad si este
modelo aún no está entrenado.

    python -m src.train.train_ranking
"""
from __future__ import annotations

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

from src.config import PROJECT_ROOT, get_seed, load_config, processed_dir
from src.evaluate.metrics import compute_metrics
from src.evaluate.run_registry import record_run
from src.features.preprocess import FEATURE_COLUMNS, build_preprocessor

_NDCG_KS = (10, 50, 100)


def _ndcg_at_ks(y_true: np.ndarray, y_score: np.ndarray, ks=_NDCG_KS) -> dict[str, float]:
    """NDCG@k para varios k, acotados al tamaño real de `y_true`."""
    y_true_2d = np.asarray(y_true, dtype=float).reshape(1, -1)
    y_score_2d = np.asarray(y_score, dtype=float).reshape(1, -1)
    out = {}
    for k in ks:
        k_eff = min(k, y_true_2d.shape[1])
        if k_eff < 1:
            continue
        # "ndcg_at_k", no "ndcg@k": MLflow no admite "@" en nombres de métrica.
        out[f"ndcg_at_{k}"] = float(ndcg_score(y_true_2d, y_score_2d, k=k_eff))
    return out


def _baseline_ndcg(y_true: np.ndarray, seed: int, n_reps: int = 50, ks=_NDCG_KS) -> dict:
    """NDCG de dos baselines triviales, sin los que un NDCG alto no es interpretable.

    `arrival_order` es el orden en que llegan las filas del holdout, sin reordenar:
    lo que vería un revisor sin ranking. `random_*` promedia `n_reps` órdenes
    aleatorios, para comprobar (no asumir) que el orden de llegada no es informativo
    por azar.
    """
    rng = np.random.default_rng(seed)
    arrival = _ndcg_at_ks(y_true, np.arange(len(y_true), 0, -1, dtype=float), ks)
    random_runs = [
        _ndcg_at_ks(y_true, rng.permutation(len(y_true)).astype(float), ks)
        for _ in range(n_reps)
    ]
    random_mean = {k: float(np.mean([r[k] for r in random_runs])) for k in arrival}
    random_std = {k: float(np.std([r[k] for r in random_runs])) for k in arrival}
    return {"arrival_order": arrival, "random_mean": random_mean, "random_std": random_std,
           "random_n_reps": n_reps}


def run(tracking_uri: str | None = None) -> dict:
    import mlflow

    from src.train.train import _resolve_tracking_uri, data_provenance, unseen_mask

    cfg = load_config()
    seed = get_seed()
    provenance = data_provenance()

    train_df = pd.read_parquet(processed_dir() / "train.parquet")
    test_df = pd.read_parquet(processed_dir() / "test.parquet")
    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["label"].astype(int)
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df["label"].astype(int)

    # Mismo criterio que train.py: evaluar solo sobre el holdout no visto.
    unseen = unseen_mask(train_df, test_df)
    X_hold, y_hold = X_test[unseen], y_test[unseen]

    pre = build_preprocessor()
    X_train_t = pre.fit_transform(X_train, y_train)
    X_hold_t = pre.transform(X_hold)

    # `deterministic`, `force_row_wise` y `num_threads=1` fijan la reducción en coma
    # flotante al construir histogramas: sin ellos LightGBM no es reproducible entre
    # ejecuciones pese al `random_state` fijo (dos ejecuciones dieron NDCG@10 distinto).
    ranker = lgb.LGBMRanker(
        objective="lambdarank", n_estimators=200, random_state=seed, verbosity=-1,
        deterministic=True, force_row_wise=True, num_threads=1)
    ranker.fit(X_train_t, y_train, group=[X_train_t.shape[0]])

    scores = ranker.predict(X_hold_t)
    ndcg = _ndcg_at_ks(y_hold.to_numpy(), scores)
    # Referencia de clasificación, para comparar con train.py.
    prob_like = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
    classif = compute_metrics(y_hold, prob_like)
    baseline = _baseline_ndcg(y_hold.to_numpy(), seed)

    print(f"Ranking (holdout no visto, n={len(y_hold)}): "
          + ", ".join(f"{k}={v:.4f}" for k, v in ndcg.items()))
    print(f"Referencia (score normalizado como prob.): PR AUC={classif['pr_auc']:.4f}")
    print("Baselines -- orden de llegada: "
          + ", ".join(f"{k}={v:.4f}" for k, v in baseline["arrival_order"].items())
          + " | aleatorio (media±sd, " + str(baseline["random_n_reps"]) + " reps): "
          + ", ".join(f"{k}={baseline['random_mean'][k]:.4f}±{baseline['random_std'][k]:.4f}"
                      for k in baseline["arrival_order"]))

    models_dir = PROJECT_ROOT / "models" / "ranking_model"
    models_dir.mkdir(parents=True, exist_ok=True)
    booster_path = models_dir / "lambdarank.txt"
    ranker.booster_.save_model(str(booster_path))
    # Persisto el preprocesador ya ajustado, no solo el booster: serving necesita
    # exactamente estas medianas, escalas y categorías para que su score sea
    # comparable al evaluado aquí. Reajustarlo en carga expondría el sistema a que
    # una regeneración posterior del dataset produjera una transformación distinta.
    preprocessor_path = models_dir / "preprocessor.joblib"
    joblib.dump(pre, preprocessor_path)

    mlflow.set_tracking_uri(_resolve_tracking_uri(cfg, tracking_uri))
    mlflow.set_experiment(cfg["mlflow"].get("ranking_experiment_name", "vus_ranking"))
    with mlflow.start_run(run_name="lightgbm_lambdarank") as mlrun:
        record_run("vus_ranking", mlrun.info.run_id, algorithm="lightgbm_lambdarank")
        mlflow.log_params({"algorithm": "lightgbm_lambdarank", "seed": seed,
                           "n_estimators": 200, "group_strategy": "single_global_group",
                           "clinvar_data_source": provenance["clinvar_source"],
                           "annotation_source": provenance["annotation_source"]})
        mlflow.set_tag("real_data_end_to_end", provenance["is_real_data"])
        mlflow.log_metrics({**ndcg, "ref_pr_auc": classif["pr_auc"]})
        mlflow.log_metrics({f"baseline_arrival_{k}": v
                            for k, v in baseline["arrival_order"].items()})
        mlflow.log_metrics({f"baseline_random_mean_{k}": v
                            for k, v in baseline["random_mean"].items()})
        # Booster nativo como artefacto de texto, no vía el flavor `mlflow.lightgbm`:
        # ese serializa con skops, que por defecto no confía en LGBMRanker.
        mlflow.log_artifact(str(booster_path), artifact_path="model")
        mlflow.log_artifact(str(preprocessor_path), artifact_path="model")

    art_dir = PROJECT_ROOT / "reports" / "training"
    art_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "algorithm": "lightgbm_lambdarank", **ndcg, "ref_pr_auc": classif["pr_auc"],
        **{f"baseline_arrival_{k}": v for k, v in baseline["arrival_order"].items()},
        **{f"baseline_random_mean_{k}": v for k, v in baseline["random_mean"].items()},
        **{f"baseline_random_std_{k}": v for k, v in baseline["random_std"].items()},
    }]).to_csv(art_dir / "ranking_metrics.csv", index=False)

    return {"ndcg": ndcg, "ref_pr_auc": classif["pr_auc"], "n_holdout": int(len(y_hold)),
           "baseline": baseline}


if __name__ == "__main__":
    run()
