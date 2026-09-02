"""Comparativa directa del modelo frente a cada predictor in silico en solitario.

CADD, REVEL y AlphaMissense son features del propio modelo, así que afirmar que lo
supera exige evaluarlos por separado sobre el mismo conjunto. Este módulo calcula,
sobre el mismo holdout no visto, el PR AUC y el ROC AUC de cada score usado
directamente como probabilidad, y compara cada uno con el modelo mediante un IC
bootstrap pareado de la diferencia.

Cada score solo está definido donde el predictor aplica -REVEL solo en missense-,
así que se reporta el denominador exacto de cada comparación.

    python -m src.evaluate.compare_predictors
"""
from __future__ import annotations

import json

import mlflow.sklearn
import pandas as pd

from src.config import PROJECT_ROOT, get_seed, processed_dir
from src.evaluate.metrics import bootstrap_pr_auc_ci, bootstrap_pr_auc_diff_ci, compute_metrics
from src.features.preprocess import FEATURE_COLUMNS
from src.train.train import unseen_mask

_SOLO_SCORES = {
    "cadd_phred": "CADD (solo)",
    "revel_score": "REVEL (solo, missense)",
    "alphamissense_score": "AlphaMissense (solo, missense)",
}


def run() -> dict:
    seed = get_seed()
    train_df = pd.read_parquet(processed_dir() / "train.parquet")
    test_df = pd.read_parquet(processed_dir() / "test.parquet")
    unseen = unseen_mask(train_df, test_df)
    hold = test_df[unseen].reset_index(drop=True)
    y_hold = hold["label"].astype(int)

    model = mlflow.sklearn.load_model(str(PROJECT_ROOT / "models" / "best_model"))
    prob_ensemble_full = model.predict_proba(hold[FEATURE_COLUMNS])[:, 1]

    rows = []
    for col, label in _SOLO_SCORES.items():
        mask = hold[col].notna()
        n = int(mask.sum())
        if n < 10:
            rows.append({"predictor": label, "column": col, "n": n,
                        "coverage": round(n / len(hold), 4), "note": "cobertura insuficiente"})
            continue
        y_sub = y_hold[mask].to_numpy()
        prob_solo = hold.loc[mask, col].to_numpy()
        prob_ensemble_sub = prob_ensemble_full[mask.to_numpy()]
        solo_metrics = compute_metrics(y_sub, prob_solo)
        ensemble_metrics_on_sub = compute_metrics(y_sub, prob_ensemble_sub)
        solo_ci = bootstrap_pr_auc_ci(y_sub, prob_solo, seed=seed)
        diff = bootstrap_pr_auc_diff_ci(y_sub, prob_ensemble_sub, prob_solo, seed=seed)
        rows.append({
            "predictor": label, "column": col, "n": n,
            "coverage": round(n / len(hold), 4),
            "solo_pr_auc": solo_metrics["pr_auc"], "solo_roc_auc": solo_metrics["roc_auc"],
            "solo_pr_auc_ci_low": solo_ci["pr_auc_ci_low"],
            "solo_pr_auc_ci_high": solo_ci["pr_auc_ci_high"],
            "ensemble_pr_auc_on_same_subset": ensemble_metrics_on_sub["pr_auc"],
            "ensemble_roc_auc_on_same_subset": ensemble_metrics_on_sub["roc_auc"],
            "pr_auc_diff_ensemble_minus_solo": (
                ensemble_metrics_on_sub["pr_auc"] - solo_metrics["pr_auc"]),
            "pr_auc_diff_ci_low": diff["pr_auc_diff_ci_low"],
            "pr_auc_diff_ci_high": diff["pr_auc_diff_ci_high"],
            "diff_crosses_zero": diff["crosses_zero"],
        })
        print(f"[{label}] n={n} ({100 * n / len(hold):.1f}% del holdout) "
              f"solo PR AUC={solo_metrics['pr_auc']:.4f} "
              f"[{solo_ci['pr_auc_ci_low']:.4f}, {solo_ci['pr_auc_ci_high']:.4f}] "
              f"| ensemble en el mismo subconjunto PR AUC="
              f"{ensemble_metrics_on_sub['pr_auc']:.4f} "
              f"| diferencia IC 95%=[{diff['pr_auc_diff_ci_low']:.4f}, "
              f"{diff['pr_auc_diff_ci_high']:.4f}] "
              f"({'NO distinguible de 0' if diff['crosses_zero'] else 'distinguible de 0'})")

    out_dir = PROJECT_ROOT / "reports" / "training"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "holdout_n": len(hold),
        "ensemble_pr_auc_full_holdout": compute_metrics(y_hold, prob_ensemble_full)["pr_auc"],
        "comparisons": rows,
    }
    (out_dir / "compare_predictors.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(rows).to_csv(out_dir / "compare_predictors.csv", index=False)
    print(f"\nComparativa guardada en {out_dir / 'compare_predictors.csv'}")
    return result


if __name__ == "__main__":
    run()
